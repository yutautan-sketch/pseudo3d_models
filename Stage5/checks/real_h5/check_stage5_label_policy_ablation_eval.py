from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from checks.real_h5.check_stage5_overlap_aggregation import (
    check_mean_probability_parity,
    load_reference_h5_metrics,
)
from evaluate_stage5 import (
    MetricTotals,
    compute_metrics,
    model_from_checkpoint,
    path_video_name,
    predict_h5,
    safe_divide,
    write_csv,
)
from export_anonymized_stage5_metrics import assert_share_bundle_anonymous, build_aliases
from infer_stage5 import checkpoint_config, load_checkpoint
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list
from stage5.utils.label_policy import (
    DEFAULT_LABEL_POLICY,
    LABEL_IGNORE,
    apply_bbox_noncontour_label_policy,
    compute_bbox_inside_mask,
)
from stage5.utils.visualization_export import write_diagnostic_segmentation_ply


PROBABILITY_PERCENTILES = (50, 90, 95, 99)
TARGET_CANONICAL = "canonical_v6"
TARGET_BG = "bbox_noncontour_as_background"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------
# S5-12 Step E7: evaluate one Run A/B checkpoint against three common targets
# that do not depend on which policy the checkpoint was trained under:
#   1. canonical_v6                    -- native teacher v6 valid points only
#      (identical to plain evaluate_stage5.py; used as a hard parity gate)
#   2. bbox_noncontour_as_background    -- audited BBox non-contour points added
#      as background+valid to the same GT positive contour (primary comparison)
#   3. bbox_noncontour_region           -- probability/prediction-rate diagnostics
#      restricted to the audited target points themselves
# One model forward per H5 (via evaluate_stage5.predict_h5); target 2/3 reuse
# the same aggregated probability/pred_label as target 1, just against
# different label/valid_mask arrays -- no extra inference cost.
# ----------------------------------------------------------------------------


def region_probability_stats(*, probability: np.ndarray, pred_label: np.ndarray, region_mask: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(region_mask))
    region_probability = probability[region_mask]
    region_pred_positive = pred_label[region_mask] == 1
    row: dict[str, Any] = {
        "target_point_count": count,
        "predicted_positive_count": int(np.sum(region_pred_positive)),
        "predicted_positive_rate": safe_divide(int(np.sum(region_pred_positive)), count),
        "prob_femur_mean": float(np.mean(region_probability)) if count else None,
    }
    for q in PROBABILITY_PERCENTILES:
        row[f"prob_femur_p{q}"] = float(np.percentile(region_probability, q)) if count else None
    return row


def process_h5(
    *,
    model: torch.nn.Module,
    path: Path,
    split: str,
    alias: str,
    features: tuple[str, ...],
    normalize_points: bool,
    device: torch.device,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    ignore_index: int,
    reference: dict[tuple[str, str], dict[str, str]],
    ply_output_dir: Path | None,
    requested_ply_aliases: set[str],
) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(path)
    video_name = path_video_name(path, data)
    require(
        "bbox_frame_order" in data and "bbox_local_xyxy" in data,
        f"{path}: missing frame_annotation BBox data required for the common ablation targets",
    )

    source_label = data["point_label"]
    source_valid = data["valid_mask"].astype(bool)
    bbox_inside_mask = compute_bbox_inside_mask(
        frame_order=data["frame_order"],
        pixel_xy=data["pixel_xy"],
        bbox_frame_order=data["bbox_frame_order"],
        bbox_local_xyxy=data["bbox_local_xyxy"],
    )
    target_bg_label, target_bg_valid, policy_stats = apply_bbox_noncontour_label_policy(
        "bbox_noncontour_background",
        point_label=source_label,
        valid_mask=source_valid,
        frame_order=data["frame_order"],
        pixel_xy=data["pixel_xy"],
        bbox_frame_order=data["bbox_frame_order"],
        bbox_local_xyxy=data["bbox_local_xyxy"],
        context=f"ablation eval common target for {path}",
    )
    region_mask = (source_label == LABEL_IGNORE) & bbox_inside_mask

    aggregated_probability, pred_label, vote_count, window_rows = predict_h5(
        model=model,
        data=data,
        features=features,
        normalize_points=normalize_points,
        device=device,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
        ignore_index=ignore_index,
        metadata={"split": split, "video_alias": alias, "h5_path": str(path)},
    )

    canonical_metrics, canonical_counts = compute_metrics(
        source_label, source_valid, pred_label, aggregated_probability, ignore_index=ignore_index
    )
    parity_mismatches = check_mean_probability_parity(
        split=split, video_name=video_name, counts=canonical_counts, reference=reference
    )

    bg_metrics, bg_counts = compute_metrics(
        target_bg_label, target_bg_valid, pred_label, aggregated_probability, ignore_index=ignore_index
    )

    region_stats = region_probability_stats(
        probability=aggregated_probability, pred_label=pred_label, region_mask=region_mask
    )

    if alias in requested_ply_aliases:
        require(ply_output_dir is not None, "ply_output_dir must be set when exporting PLY")
        alias_dir = ply_output_dir / alias
        write_diagnostic_segmentation_ply(
            alias_dir / f"{TARGET_CANONICAL}_diagnostic.ply",
            data["points"],
            source_label,
            source_valid,
            pred_label,
            ignore_index=ignore_index,
        )
        write_diagnostic_segmentation_ply(
            alias_dir / f"{TARGET_BG}_diagnostic.ply",
            data["points"],
            target_bg_label,
            target_bg_valid,
            pred_label,
            ignore_index=ignore_index,
        )

    return {
        "video_name": video_name,
        "canonical_metrics": canonical_metrics,
        "canonical_counts": canonical_counts,
        "bg_metrics": bg_metrics,
        "bg_counts": bg_counts,
        "region_stats": region_stats,
        "policy_stats": policy_stats,
        "parity_mismatches": parity_mismatches,
    }


def build_video_row(*, split: str, alias: str, target: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"split": split, "video_alias": alias, "target": target, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-12 Step E7: common-target evaluation for one Run A/B label-policy checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selected_train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument(
        "--selection_summary", required=True, help="evaluation_data/summary.json from prepare_stage5_evaluation_data.py"
    )
    parser.add_argument(
        "--reference_h5_metrics_csv",
        required=True,
        help="Existing plain evaluate_stage5.py h5_metrics.csv for this same checkpoint "
        "(canonical_v6 mean baseline parity gate)",
    )
    parser.add_argument("--share_output_dir", required=True)
    parser.add_argument("--private_output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--window_size_frames", type=int, default=None)
    parser.add_argument("--window_stride_frames", type=int, default=None)
    parser.add_argument("--export_ply_alias", action="append", default=None)
    parser.add_argument("--ply_output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    require(device.type == "cuda", "S5-12 ablation evaluator requires CUDA")
    require(torch.cuda.is_available(), "CUDA was requested but torch.cuda.is_available() is False")

    share_output_dir = Path(args.share_output_dir)
    private_output_dir = Path(args.private_output_dir)
    share_output_dir.mkdir(parents=True, exist_ok=True)
    private_output_dir.mkdir(parents=True, exist_ok=True)

    selected_train_paths = read_path_list(args.selected_train_list)
    val_paths = read_path_list(args.val_list)
    require(bool(selected_train_paths) or bool(val_paths), "No evaluation H5 files to process")
    for path in (*selected_train_paths, *val_paths):
        require(path.is_file(), f"H5 not found: {path}")

    selection_summary = json.loads(Path(args.selection_summary).read_text(encoding="utf-8"))
    reference = load_reference_h5_metrics(Path(args.reference_h5_metrics_csv))

    requested_ply_aliases = set(args.export_ply_alias or [])
    ply_output_dir: Path | None = None
    if requested_ply_aliases:
        require(args.ply_output_dir is not None, "--ply_output_dir is required when --export_ply_alias is set")
        ply_output_dir = Path(args.ply_output_dir)
        ply_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    trained_policy = str(checkpoint_config(checkpoint).get("label_policy", DEFAULT_LABEL_POLICY))
    model, config, features, normalize_points = model_from_checkpoint(
        checkpoint, device=device, strict=args.strict_checkpoint
    )
    del checkpoint

    window_size_frames = int(
        args.window_size_frames if args.window_size_frames is not None else config.get("window_size_frames", 16)
    )
    window_stride_frames = int(
        args.window_stride_frames if args.window_stride_frames is not None else config.get("window_stride_frames", 8)
    )
    include_tail_window = bool(config.get("include_tail_window", True))
    ignore_index = int(config.get("ignore_index", -1))

    h5_manifest_rows: list[dict[str, str]] = []
    for split, paths in (("train_sanity", selected_train_paths), ("validation", val_paths)):
        for path in paths:
            data = load_stage5_pointcloud_h5(path)
            video_name = path_video_name(path, data)
            h5_manifest_rows.append({"split": split, "video_name": video_name, "h5_path": str(path)})
    aliases, private_rows = build_aliases(h5_manifest_rows, selection_summary)
    write_csv(private_output_dir / "video_id_map_DO_NOT_SHARE.csv", private_rows)

    split_paths = (("train_sanity", selected_train_paths), ("validation", val_paths))
    video_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    parity_failure_rows: list[dict[str, Any]] = []
    canonical_totals: dict[str, MetricTotals] = {split: MetricTotals.empty() for split, _ in split_paths}
    bg_totals: dict[str, MetricTotals] = {split: MetricTotals.empty() for split, _ in split_paths}

    summary: dict[str, Any] = {
        "status": "running",
        "checkpoint": Path(args.checkpoint).name,
        "trained_label_policy": trained_policy,
        "features": list(features),
        "window_size_frames": window_size_frames,
        "window_stride_frames": window_stride_frames,
    }
    summary_path = share_output_dir / "label_policy_ablation_summary.json"

    try:
        for split, paths in split_paths:
            for path in tqdm(paths, desc=f"label policy ablation {split}"):
                video_name = next(
                    row["video_name"]
                    for row in h5_manifest_rows
                    if row["split"] == split and row["h5_path"] == str(path)
                )
                alias = aliases[(split, video_name)]
                result = process_h5(
                    model=model,
                    path=path,
                    split=split,
                    alias=alias,
                    features=features,
                    normalize_points=normalize_points,
                    device=device,
                    window_size_frames=window_size_frames,
                    window_stride_frames=window_stride_frames,
                    include_tail_window=include_tail_window,
                    ignore_index=ignore_index,
                    reference=reference,
                    ply_output_dir=ply_output_dir,
                    requested_ply_aliases=requested_ply_aliases,
                )

                if result["parity_mismatches"]:
                    parity_failure_rows.append(
                        {
                            "split": split,
                            "video_alias": alias,
                            "mismatches": json.dumps(result["parity_mismatches"], ensure_ascii=False),
                        }
                    )
                    write_csv(private_output_dir / "mean_baseline_parity_failure.csv", parity_failure_rows)
                    raise RuntimeError(
                        f"canonical_v6 mean_probability parity mismatch for split={split!r} "
                        f"video={result['video_name']!r}: {result['parity_mismatches']}"
                    )

                canonical_totals[split].update(result["canonical_counts"])
                bg_totals[split].update(result["bg_counts"])
                video_rows.append(build_video_row(split=split, alias=alias, target=TARGET_CANONICAL, metrics=result["canonical_metrics"]))
                video_rows.append(build_video_row(split=split, alias=alias, target=TARGET_BG, metrics=result["bg_metrics"]))
                region_rows.append({"split": split, "video_alias": alias, **result["region_stats"], **result["policy_stats"]})

        checkpoint_summary_rows: list[dict[str, Any]] = []
        for split, _ in split_paths:
            for target, totals in ((TARGET_CANONICAL, canonical_totals[split]), (TARGET_BG, bg_totals[split])):
                metrics = totals.metrics()
                target_video_rows = [r for r in video_rows if r["split"] == split and r["target"] == target]
                f1_values = [float(r["f1"]) for r in target_video_rows]
                iou_values = [float(r["iou_femur"]) for r in target_video_rows]
                tp_zero = sum(1 for r in target_video_rows if int(r["true_positive_count"]) == 0)
                checkpoint_summary_rows.append(
                    {
                        "split": split,
                        "target": target,
                        **metrics,
                        "num_videos": len(target_video_rows),
                        "video_mean_f1": float(np.mean(f1_values)) if f1_values else None,
                        "video_median_f1": float(np.median(f1_values)) if f1_values else None,
                        "video_mean_iou": float(np.mean(iou_values)) if iou_values else None,
                        "video_median_iou": float(np.median(iou_values)) if iou_values else None,
                        "tp_zero_video_count": tp_zero,
                    }
                )

        write_csv(share_output_dir / "label_policy_ablation_video_metrics.csv", video_rows)
        write_csv(share_output_dir / "label_policy_ablation_region_stats.csv", region_rows)
        write_csv(share_output_dir / "label_policy_ablation_checkpoint_summary.csv", checkpoint_summary_rows)

        summary.update(
            {
                "status": "passed",
                "mean_baseline_parity": "passed",
                "num_videos": len(selected_train_paths) + len(val_paths),
                "checkpoint_summary": checkpoint_summary_rows,
            }
        )
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        raise
    finally:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    assert_share_bundle_anonymous(share_output_dir, private_rows=private_rows)

    print("Stage5 label-policy ablation evaluator passed.")
    print(f"checkpoint: {args.checkpoint}")
    print(f"trained label_policy: {trained_policy}")
    print(f"videos evaluated: {summary['num_videos']}")
    print(f"share output: {share_output_dir}")
    print(f"private output: {private_output_dir}")


if __name__ == "__main__":
    main()
