from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from checks.real_h5.check_stage5_overlap_aggregation import (
    check_mean_probability_parity,
    load_reference_h5_metrics,
)
from checks.real_h5.check_stage5_padding_parity import set_seed
from evaluate_stage5 import (
    MetricTotals,
    compute_metrics,
    model_from_checkpoint,
    path_video_name,
    safe_divide,
    write_csv,
)
from export_anonymized_stage5_metrics import assert_share_bundle_anonymous, build_aliases
from infer_stage5 import build_features, load_checkpoint
from stage5.utils.feature_normalization import normalize_xyz
from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list


MODES: tuple[str, ...] = ("eval", "train")
GT_CLASS_IDS: dict[str, int] = {"valid_positive": 0, "valid_background": 1, "ignore": 2}
DIFF_ROW_NAMES: tuple[str, ...] = ("valid_positive", "valid_background", "ignore", "all")
PROBABILITY_PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_array_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator_safe = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, numerator / denominator_safe, 0.0)


class ParityMismatchError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# 6.1: BatchNorm running-buffer inspection (section 6.2's "running mean/variance,
# num_batches_tracked" report). Captured from the eval-mode instance only, since
# its buffers are the checkpoint's saved production statistics (eval-mode forward
# never mutates them).
# ----------------------------------------------------------------------------


def inspect_batchnorm_modules(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        running_mean = module.running_mean
        running_var = module.running_var
        num_batches_tracked = module.num_batches_tracked
        mean_np = running_mean.detach().cpu().numpy() if running_mean is not None else np.array([])
        var_np = running_var.detach().cpu().numpy() if running_var is not None else np.array([])
        rows.append(
            {
                "module_name": name,
                "num_features": int(module.num_features),
                "running_mean_finite": bool(np.all(np.isfinite(mean_np))) if mean_np.size else None,
                "running_mean_min": float(np.min(mean_np)) if mean_np.size else None,
                "running_mean_max": float(np.max(mean_np)) if mean_np.size else None,
                "running_var_finite": bool(np.all(np.isfinite(var_np))) if var_np.size else None,
                "running_var_min": float(np.min(var_np)) if var_np.size else None,
                "running_var_max": float(np.max(var_np)) if var_np.size else None,
                "running_var_negative_count": int(np.sum(var_np < 0)) if var_np.size else None,
                "num_batches_tracked": (
                    int(num_batches_tracked.item()) if num_batches_tracked is not None else None
                ),
            }
        )
    return rows


# ----------------------------------------------------------------------------
# Independent eval()/train() model instances from the same checkpoint.
# ----------------------------------------------------------------------------


def build_model_pair(
    checkpoint_path: str,
    *,
    device: torch.device,
    strict: bool,
) -> tuple[nn.Module, nn.Module, dict[str, Any], tuple[str, ...], bool]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    # model_from_checkpoint always returns model.eval(); build two independent
    # nn.Module instances from the same (unmodified) checkpoint dict so B never
    # shares parameter/buffer storage with A. load_state_dict copies values, so
    # reusing the same source dict for both calls does not alias the two models.
    model_eval, config, features, normalize_points = model_from_checkpoint(
        checkpoint, device=device, strict=strict
    )
    model_train, config_train, features_train, normalize_points_train = model_from_checkpoint(
        checkpoint, device=device, strict=strict
    )
    del checkpoint
    require(config == config_train, "eval/train model instances were built from different configs")
    require(features == features_train, "eval/train model instances resolved different features")
    require(
        normalize_points == normalize_points_train,
        "eval/train model instances resolved different normalize_points",
    )
    require(
        float(config.get("dropout", 0.0)) == 0.0,
        f"BatchNorm mode parity requires dropout=0.0, got {config.get('dropout')}",
    )
    model_train.train()
    require(model_eval.training is False, "model_eval must remain in eval() mode")
    require(model_train.training is True, "model_train must be in train() mode")
    return model_eval, model_train, config, features, normalize_points


# ----------------------------------------------------------------------------
# Per-window dual forward pass
# ----------------------------------------------------------------------------


def run_window_forward(
    model: nn.Module,
    points_window: np.ndarray,
    features_window: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    h5_path: str,
) -> np.ndarray:
    set_seed(seed)
    with torch.inference_mode():
        batch = {
            "points": torch.from_numpy(points_window[None]).float().to(device),
            "features": torch.from_numpy(features_window[None]).float().to(device),
        }
        logits = model(batch)["logits"][0]
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
    require(np.all(np.isfinite(probabilities)), f"Non-finite probabilities in {h5_path}")
    return probabilities


# ----------------------------------------------------------------------------
# Per-H5 processing: same window, same point order, same RNG seed fed to both
# model instances (constraint 6.1); only the eval()/train() BatchNorm mode differs.
# ----------------------------------------------------------------------------


def summarize_diff_group(
    mask: np.ndarray,
    *,
    abs_diff: np.ndarray,
    disagreement: np.ndarray,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    subset = abs_diff[mask]
    disagreement_count = int(np.sum(disagreement[mask]))
    row: dict[str, Any] = {
        "occurrence_count": count,
        "disagreement_count": disagreement_count,
        "disagreement_rate": safe_divide(disagreement_count, count),
        "mean_abs_diff": float(np.mean(subset)) if count else None,
        "max_abs_diff": float(np.max(subset)) if count else None,
    }
    for q in PROBABILITY_PERCENTILES:
        row[f"p{q}_abs_diff"] = float(np.percentile(subset, q)) if count else None
    return row


def process_h5_pair(
    *,
    model_eval: nn.Module,
    model_train: nn.Module,
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
    seed_base: int,
    reference: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(path)
    video_name = path_video_name(path, data)
    labels = data["point_label"]
    valid_mask = data["valid_mask"]
    frame_order = data["frame_order"]
    raw_points = data["points"].astype(np.float32)
    points = normalize_xyz(raw_points) if normalize_points else raw_points
    feature_values = build_features(data, features)

    windows = generate_frame_order_windows(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    require(bool(windows), f"No frame windows generated for {path}")

    num_points = points.shape[0]
    models = {"eval": model_eval, "train": model_train}
    prob_sum = {mode: np.zeros(num_points, dtype=np.float64) for mode in MODES}
    vote_count = {mode: np.zeros(num_points, dtype=np.int32) for mode in MODES}
    window_totals = {mode: MetricTotals.empty() for mode in MODES}
    diff_chunks_abs: list[np.ndarray] = []
    diff_chunks_disagree: list[np.ndarray] = []
    diff_chunks_gt: list[np.ndarray] = []

    for window_offset, window in enumerate(windows):
        indices = point_indices_for_window(frame_order, window)
        if indices.size == 0:
            continue
        seed = seed_base + window_offset
        probability_by_mode: dict[str, np.ndarray] = {}
        for mode in MODES:
            probability_by_mode[mode] = run_window_forward(
                models[mode],
                points[indices],
                feature_values[indices],
                device=device,
                seed=seed,
                h5_path=str(path),
            )

        window_labels = labels[indices]
        window_valid = valid_mask[indices]
        for mode in MODES:
            p1 = probability_by_mode[mode][:, 1]
            pred = (p1 > 0.5).astype(np.uint8)
            _, counts = compute_metrics(window_labels, window_valid, pred, p1, ignore_index=ignore_index)
            window_totals[mode].update(counts)
            prob_sum[mode][indices] += p1
            vote_count[mode][indices] += 1

        p1_eval = probability_by_mode["eval"][:, 1]
        p1_train = probability_by_mode["train"][:, 1]
        pred_eval = p1_eval > 0.5
        pred_train = p1_train > 0.5

        valid = window_valid & (window_labels != int(ignore_index))
        target_positive = window_labels == 1
        gt_class = np.full(indices.shape, GT_CLASS_IDS["ignore"], dtype=np.int8)
        gt_class[valid & target_positive] = GT_CLASS_IDS["valid_positive"]
        gt_class[valid & ~target_positive] = GT_CLASS_IDS["valid_background"]

        diff_chunks_abs.append(np.abs(p1_eval - p1_train))
        diff_chunks_disagree.append(pred_eval != pred_train)
        diff_chunks_gt.append(gt_class)

    for mode in MODES:
        missing = vote_count[mode] == 0
        require(
            not np.any(missing),
            f"{int(np.sum(missing))} points received no {mode}-mode prediction in {path}",
        )

    all_abs_diff = np.concatenate(diff_chunks_abs)
    all_disagreement = np.concatenate(diff_chunks_disagree)
    all_gt_class = np.concatenate(diff_chunks_gt)

    diff_rows: list[dict[str, Any]] = []
    for name in DIFF_ROW_NAMES:
        mask = (
            np.ones_like(all_gt_class, dtype=bool)
            if name == "all"
            else all_gt_class == GT_CLASS_IDS[name]
        )
        row = summarize_diff_group(mask, abs_diff=all_abs_diff, disagreement=all_disagreement)
        diff_rows.append({"split": split, "video_alias": alias, "gt_class": name, **row})

    aggregation_metrics: dict[str, dict[str, Any]] = {}
    aggregation_counts: dict[str, dict[str, Any]] = {}
    parity_mismatches: dict[str, Any] = {}
    for mode in MODES:
        mean_probability = safe_array_divide(prob_sum[mode], vote_count[mode].astype(np.float64))
        pred_label = (mean_probability > 0.5).astype(np.uint8)
        metrics, counts = compute_metrics(labels, valid_mask, pred_label, mean_probability, ignore_index=ignore_index)
        aggregation_metrics[mode] = metrics
        aggregation_counts[mode] = counts
        if mode == "eval":
            parity_mismatches = check_mean_probability_parity(
                split=split, video_name=video_name, counts=counts, reference=reference
            )

    window_metrics = {mode: window_totals[mode].metrics() for mode in MODES}
    window_counts = {mode: dict(window_totals[mode].values) for mode in MODES}

    return {
        "video_name": video_name,
        "diff_rows": diff_rows,
        "window_metrics": window_metrics,
        "window_counts": window_counts,
        "aggregation_metrics": aggregation_metrics,
        "aggregation_counts": aggregation_counts,
        "parity_mismatches": parity_mismatches,
    }


# ----------------------------------------------------------------------------
# Split/checkpoint-level rollups
# ----------------------------------------------------------------------------


def build_checkpoint_summary(
    *,
    window_totals: dict[tuple[str, str], MetricTotals],
    aggregation_totals: dict[tuple[str, str], MetricTotals],
    aggregation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity, totals_by_key in (("window", window_totals), ("aggregated", aggregation_totals)):
        for (split, mode), totals in totals_by_key.items():
            metrics = totals.metrics()
            row: dict[str, Any] = {"split": split, "mode": mode, "granularity": granularity, **metrics}
            if granularity == "aggregated":
                video_rows = [
                    r for r in aggregation_rows if r["split"] == split and r["mode"] == mode
                ]
                f1_values = [float(r["f1"]) for r in video_rows]
                iou_values = [float(r["iou_femur"]) for r in video_rows]
                tp_zero_video_count = sum(1 for r in video_rows if int(r["true_positive_count"]) == 0)
                row.update(
                    {
                        "num_videos": len(video_rows),
                        "video_mean_f1": float(np.mean(f1_values)) if f1_values else None,
                        "video_median_f1": float(np.median(f1_values)) if f1_values else None,
                        "video_mean_iou": float(np.mean(iou_values)) if iou_values else None,
                        "video_median_iou": float(np.median(iou_values)) if iou_values else None,
                        "tp_zero_video_count": tp_zero_video_count,
                    }
                )
            rows.append(row)
    return rows


def build_mode_deltas(checkpoint_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["split"], row["granularity"], row["mode"]): row for row in checkpoint_summary_rows}
    rows: list[dict[str, Any]] = []
    for (split, granularity, mode), row in by_key.items():
        if mode != "train":
            continue
        eval_row = by_key.get((split, granularity, "eval"))
        if eval_row is None:
            continue
        delta: dict[str, Any] = {
            "split": split,
            "granularity": granularity,
            "f1_delta_train_minus_eval": row["f1"] - eval_row["f1"],
            "iou_delta_train_minus_eval": row["iou_femur"] - eval_row["iou_femur"],
            "recall_delta_train_minus_eval": row["recall"] - eval_row["recall"],
            "false_positive_count_delta_train_minus_eval": row["false_positive_count"]
            - eval_row["false_positive_count"],
        }
        if granularity == "aggregated":
            delta["tp_zero_video_count_delta_train_minus_eval"] = (
                row["tp_zero_video_count"] - eval_row["tp_zero_video_count"]
            )
        rows.append(delta)
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("Stage5 BatchNorm mode parity self-test passed.")
        return

    require(args.checkpoint is not None, "--checkpoint is required unless --self_test is set")
    require(args.selected_train_list is not None, "--selected_train_list is required")
    require(args.val_list is not None, "--val_list is required")
    require(args.selection_summary is not None, "--selection_summary is required")
    require(args.reference_h5_metrics_csv is not None, "--reference_h5_metrics_csv is required")
    require(args.share_output_dir is not None, "--share_output_dir is required")
    require(args.private_output_dir is not None, "--private_output_dir is required")

    device = torch.device(args.device)
    require(device.type == "cuda", "BatchNorm mode parity check requires CUDA")
    require(torch.cuda.is_available(), "CUDA was requested but torch.cuda.is_available() is False")

    share_output_dir = Path(args.share_output_dir)
    private_output_dir = Path(args.private_output_dir)
    share_output_dir.mkdir(parents=True, exist_ok=True)
    private_output_dir.mkdir(parents=True, exist_ok=True)

    selected_train_paths = read_path_list(args.selected_train_list)
    val_paths = read_path_list(args.val_list)
    require(bool(selected_train_paths) or bool(val_paths), "No H5 files to process")
    for path in (*selected_train_paths, *val_paths):
        require(path.is_file(), f"H5 not found: {path}")
    duplicate_paths = set(selected_train_paths) & set(val_paths)
    require(not duplicate_paths, f"train_sanity and validation lists overlap: {sorted(duplicate_paths)[:3]}")

    selection_summary = json.loads(Path(args.selection_summary).read_text(encoding="utf-8"))
    reference = load_reference_h5_metrics(Path(args.reference_h5_metrics_csv))

    model_eval, model_train, config, features, normalize_points = build_model_pair(
        args.checkpoint, device=device, strict=args.strict_checkpoint
    )
    buffer_report = inspect_batchnorm_modules(model_eval)

    window_size_frames = int(
        args.window_size_frames if args.window_size_frames is not None else config.get("window_size_frames", 12)
    )
    window_stride_frames = int(
        args.window_stride_frames
        if args.window_stride_frames is not None
        else config.get("window_stride_frames", 6)
    )
    include_tail_window = (
        args.include_tail_window
        if args.include_tail_window is not None
        else bool(config.get("include_tail_window", True))
    )
    ignore_index = int(config.get("ignore_index", -1))

    split_paths = (("train_sanity", selected_train_paths), ("validation", val_paths))

    h5_manifest_rows: list[dict[str, str]] = []
    for split, paths in split_paths:
        for path in paths:
            with h5py.File(path, "r") as handle:
                file_attrs = dict(handle.attrs)
            video_name = path_video_name(path, {"file_attrs": file_attrs})
            h5_manifest_rows.append({"split": split, "video_name": video_name, "h5_path": str(path)})

    aliases, private_rows = build_aliases(h5_manifest_rows, selection_summary)
    write_csv(private_output_dir / "video_id_map_DO_NOT_SHARE.csv", private_rows)
    (private_output_dir / "DO_NOT_SHARE.txt").write_text(
        "This directory contains original video identifiers and H5 paths.\n",
        encoding="utf-8",
    )

    diff_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    aggregation_metric_rows: list[dict[str, Any]] = []
    parity_failure_rows: list[dict[str, Any]] = []
    window_totals: dict[tuple[str, str], MetricTotals] = {
        (split, mode): MetricTotals.empty() for split, _ in split_paths for mode in MODES
    }
    aggregation_totals: dict[tuple[str, str], MetricTotals] = {
        (split, mode): MetricTotals.empty() for split, _ in split_paths for mode in MODES
    }

    summary: dict[str, Any] = {
        "status": "running",
        "run_name": Path(args.checkpoint).resolve().parent.name,
        "checkpoint_name": Path(args.checkpoint).name,
        "features": list(features),
        "normalize_points": normalize_points,
        "ignore_index": ignore_index,
        "window_size_frames": window_size_frames,
        "window_stride_frames": window_stride_frames,
        "include_tail_window": bool(include_tail_window),
        "modes": list(MODES),
        "mean_baseline_parity_reference": Path(args.reference_h5_metrics_csv).name,
        "batchnorm_module_count": len(buffer_report),
    }
    summary_path = share_output_dir / "batchnorm_mode_parity_summary.json"
    private_run_info_path = private_output_dir / "batchnorm_mode_parity_run_info_DO_NOT_SHARE.json"
    with private_run_info_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "reference_h5_metrics_csv": str(Path(args.reference_h5_metrics_csv).resolve()),
                "selected_train_list": str(Path(args.selected_train_list).resolve()),
                "val_list": str(Path(args.val_list).resolve()),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    try:
        seed_base = int(args.seed)
        for split, paths in split_paths:
            for path in tqdm(paths, desc=f"batchnorm mode parity {split}"):
                video_name = next(
                    (
                        row["video_name"]
                        for row in h5_manifest_rows
                        if row["split"] == split and row["h5_path"] == str(path)
                    ),
                    None,
                )
                require(video_name is not None, f"Missing manifest entry for {path}")
                alias = aliases[(split, video_name)]

                result = process_h5_pair(
                    model_eval=model_eval,
                    model_train=model_train,
                    path=path,
                    split=split,
                    alias=alias,
                    features=features,
                    normalize_points=normalize_points,
                    device=device,
                    window_size_frames=window_size_frames,
                    window_stride_frames=window_stride_frames,
                    include_tail_window=bool(include_tail_window),
                    ignore_index=ignore_index,
                    seed_base=seed_base,
                    reference=reference,
                )

                if result["parity_mismatches"]:
                    parity_failure_rows.append(
                        {
                            "split": split,
                            "video_name": result["video_name"],
                            "video_alias": alias,
                            "mismatches": json.dumps(result["parity_mismatches"], ensure_ascii=False),
                        }
                    )
                    write_csv(private_output_dir / "mean_baseline_parity_failure.csv", parity_failure_rows)
                    raise ParityMismatchError(
                        f"eval-mode mean_probability parity mismatch for split={split!r} "
                        f"video={result['video_name']!r}: {result['parity_mismatches']}"
                    )

                diff_rows.extend(result["diff_rows"])
                for mode in MODES:
                    window_totals[(split, mode)].update(result["window_counts"][mode])
                    aggregation_totals[(split, mode)].update(result["aggregation_counts"][mode])
                    window_metric_rows.append(
                        {"split": split, "video_alias": alias, "mode": mode, **result["window_metrics"][mode]}
                    )
                    aggregation_metric_rows.append(
                        {"split": split, "video_alias": alias, "mode": mode, **result["aggregation_metrics"][mode]}
                    )

        checkpoint_summary_rows = build_checkpoint_summary(
            window_totals=window_totals,
            aggregation_totals=aggregation_totals,
            aggregation_rows=aggregation_metric_rows,
        )
        mode_delta_rows = build_mode_deltas(checkpoint_summary_rows)

        write_csv(share_output_dir / "batchnorm_probability_difference.csv", diff_rows)
        write_csv(share_output_dir / "batchnorm_window_metrics.csv", window_metric_rows)
        write_csv(share_output_dir / "batchnorm_aggregation_metrics.csv", aggregation_metric_rows)
        write_csv(share_output_dir / "batchnorm_checkpoint_summary.csv", checkpoint_summary_rows)
        write_csv(share_output_dir / "batchnorm_mode_delta.csv", mode_delta_rows)
        write_csv(share_output_dir / "batchnorm_buffer_report.csv", buffer_report)

        summary.update(
            {
                "status": "passed",
                "mean_baseline_parity": "passed",
                "num_videos": len(selected_train_paths) + len(val_paths),
                "checkpoint_summary": checkpoint_summary_rows,
                "mode_delta": mode_delta_rows,
            }
        )
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        if isinstance(error, ParityMismatchError):
            summary["mean_baseline_parity"] = "failed"
            summary["parity_failure_csv"] = "mean_baseline_parity_failure.csv (in private_output_dir)"
        raise
    finally:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    try:
        assert_share_bundle_anonymous(share_output_dir, private_rows=private_rows)
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = f"share bundle anonymization check failed: {error}"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        raise

    print("Stage5 BatchNorm mode parity checker passed.")
    print(f"checkpoint: {args.checkpoint}")
    print(f"videos processed: {summary['num_videos']}")
    print(f"share output: {share_output_dir}")
    print(f"private output: {private_output_dir}")


# ----------------------------------------------------------------------------
# Synthetic self-test (no checkpoint/H5 required for the pure-Python/NumPy
# pieces; a real BatchNorm module is used to test inspect_batchnorm_modules).
# ----------------------------------------------------------------------------


def test_inspect_batchnorm_modules_flags_bad_buffers() -> None:
    module = nn.Sequential(nn.BatchNorm1d(4))
    bn = module[0]
    bn.running_mean.data = torch.tensor([0.0, 1.0, float("nan"), -2.0])
    bn.running_var.data = torch.tensor([1.0, -0.5, 2.0, 3.0])
    bn.num_batches_tracked.data = torch.tensor(42)

    rows = inspect_batchnorm_modules(module)
    require(len(rows) == 1, f"Expected exactly one BatchNorm module, got {len(rows)}")
    row = rows[0]
    require(row["num_features"] == 4, "num_features must match the BatchNorm module")
    require(row["running_mean_finite"] is False, "NaN in running_mean must be flagged as not finite")
    require(row["running_var_negative_count"] == 1, "Exactly one negative running_var entry must be counted")
    require(row["num_batches_tracked"] == 42, "num_batches_tracked must be read from the module buffer")


def test_summarize_diff_group_basic() -> None:
    abs_diff = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    disagreement = np.array([False, False, True, True, False])
    mask = np.array([True, True, True, True, True])
    row = summarize_diff_group(mask, abs_diff=abs_diff, disagreement=disagreement)
    require(row["occurrence_count"] == 5, "occurrence_count must equal the mask size")
    require(row["disagreement_count"] == 2, "disagreement_count must equal the number of True flags")
    require(np.isclose(row["disagreement_rate"], 0.4), "disagreement_rate must be disagreement_count/occurrence_count")
    require(np.isclose(row["mean_abs_diff"], np.mean(abs_diff)), "mean_abs_diff must equal the mean of abs_diff")
    require(np.isclose(row["max_abs_diff"], 0.4), "max_abs_diff must equal the max of abs_diff")


def test_summarize_diff_group_empty_mask() -> None:
    abs_diff = np.array([0.1, 0.2])
    disagreement = np.array([False, True])
    mask = np.array([False, False])
    row = summarize_diff_group(mask, abs_diff=abs_diff, disagreement=disagreement)
    require(row["occurrence_count"] == 0, "occurrence_count must be 0 for an empty mask")
    require(row["mean_abs_diff"] is None, "mean_abs_diff must be None (not NaN/0) for an empty mask")
    require(row["disagreement_rate"] == 0.0, "disagreement_rate must safely default to 0.0 for an empty mask")


def test_mean_aggregation_matches_manual_computation() -> None:
    num_points = 3
    prob_sum = {"eval": np.zeros(num_points), "train": np.zeros(num_points)}
    vote_count = {"eval": np.zeros(num_points, dtype=np.int32), "train": np.zeros(num_points, dtype=np.int32)}

    windows = [
        {"eval": np.array([0.2, 0.6, 0.9]), "train": np.array([0.1, 0.7, 0.8])},
        {"eval": np.array([0.4, np.nan, np.nan]), "train": np.array([0.3, np.nan, np.nan])},
    ]
    indices_per_window = [np.array([0, 1, 2]), np.array([0])]
    for window_probs, indices in zip(windows, indices_per_window):
        for mode in ("eval", "train"):
            values = window_probs[mode][: indices.size]
            prob_sum[mode][indices] += values
            vote_count[mode][indices] += 1

    mean_eval = safe_array_divide(prob_sum["eval"], vote_count["eval"].astype(np.float64))
    require(np.isclose(mean_eval[0], (0.2 + 0.4) / 2), "point 0 mean (eval) must average both windows' votes")
    require(np.isclose(mean_eval[1], 0.6), "point 1 mean (eval) must equal its single vote")
    require(np.isclose(mean_eval[2], 0.9), "point 2 mean (eval) must equal its single vote")


def test_dropout_assertion_rejects_nonzero_dropout() -> None:
    raised = False
    try:
        require(float({"dropout": 0.1}.get("dropout", 0.0)) == 0.0, "dropout must be 0.0")
    except AssertionError:
        raised = True
    require(raised, "A nonzero dropout config must fail the dropout==0.0 assertion")


def run_self_test() -> None:
    test_inspect_batchnorm_modules_flags_bad_buffers()
    test_summarize_diff_group_basic()
    test_summarize_diff_group_empty_mask()
    test_mean_aggregation_matches_manual_computation()
    test_dropout_assertion_rejects_nonzero_dropout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PointNeXt-S eval()/train() BatchNorm-mode predictions on Stage5 H5 files"
    )
    parser.add_argument("--self_test", action="store_true", help="Run synthetic tests and exit")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--selected_train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument(
        "--selection_summary",
        default=None,
        help="Path to evaluation_data/summary.json produced by prepare_stage5_evaluation_data.py",
    )
    parser.add_argument(
        "--reference_h5_metrics_csv",
        default=None,
        help="Existing evaluate_stage5.py h5_metrics.csv for the same checkpoint (eval-mode sanity parity)",
    )
    parser.add_argument("--share_output_dir", default=None)
    parser.add_argument("--private_output_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--window_size_frames", type=int, default=None)
    parser.add_argument("--window_stride_frames", type=int, default=None)
    parser.add_argument(
        "--include_tail_window",
        dest="include_tail_window",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no_include_tail_window",
        dest="include_tail_window",
        action="store_false",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
