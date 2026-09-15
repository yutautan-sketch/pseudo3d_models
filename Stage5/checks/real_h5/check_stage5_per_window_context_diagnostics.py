from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from checks.real_h5.check_stage5_class_weight_threshold_free import (
    check_threshold_0p5_parity,
    load_h5_metrics_rows,
    load_prediction,
    safe_name,
)
from checks.real_h5.check_stage5_frame_spatial_overlap_diagnostics import require, window_occurrence_counts
from checks.real_h5.check_stage5_frame_xy_diagnostics import assign_xy_grid_bin
from checks.real_h5.check_stage5_xy_coordinate_provenance import normalize_pixel_xy_with_dimensions
from stage5.utils.h5_io import load_stage5_pointcloud_h5

# ----------------------------------------------------------------------------
# S5-14 Step H4: per-window context diagnostics for the fixed 21 videos.
# Reuses check_stage5_overlap_aggregation.py's OverlapAccumulator/
# run_overlap_forward (S5-08) for the actual per-window model forward pass
# (the ONE re-inference this step is allowed), and cross-checks its own
# freshly-derived mean-probability aggregate against W-A's already-saved
# evaluate_stage5.py prediction with NO tolerance (handoff 7 Step H4: "CUDA
# 非決定性等で不一致が出てもtoleranceで通さず停止").
#
# Everything in this module except `run_overlap_forward` itself (imported,
# not reimplemented) is torch-free and unit-testable with a hand-built
# OverlapAccumulator, exactly like check_stage5_overlap_aggregation.py's own
# synthetic self-tests.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Parity: this run's fresh per-window forward vs. the already-saved W-A
# aggregate prediction.
# ----------------------------------------------------------------------------


def canonical_predicted_class(
    probability_sum_2class: np.ndarray,
    vote_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce evaluate_stage5.predict_h5()'s exact final aggregation step:
    float64 probability_sum / vote_count -> cast to float32 -> 2-class argmax
    (a tie after the float32 cast resolves to index 0 = background, matching
    both predict_h5() and check_stage5_overlap_aggregation.py's own
    test_probability_half_is_background()). Never reconstructs background as
    1 - positive; both classes must come from the actual accumulated softmax
    output (Step H4.1 policy-chat finding: 1 - p1 is not bit-identical to a
    separately-accumulated background probability under floating point).

    Returns (aggregated_probability [N, 2] float32, pred_label [N] uint8).
    """
    require(probability_sum_2class.ndim == 2 and probability_sum_2class.shape[1] == 2, "expected a 2-class probability sum array")
    aggregated_probability = (probability_sum_2class / vote_count[:, None].astype(np.float64)).astype(np.float32)
    pred_label = np.argmax(aggregated_probability, axis=1).astype(np.uint8)
    return aggregated_probability, pred_label


def verify_h4_parity(
    *,
    vote_count: np.ndarray,
    aggregated_probability_positive: np.ndarray,
    pred_label: np.ndarray,
    saved_prob_femur: np.ndarray,
    saved_vote_count: np.ndarray,
    saved_pred_label: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    training_occurrence_count: np.ndarray,
    h5_metrics_row: dict[str, str],
    ignore_index: int,
    context: str,
    diff_artifact_path: Path | None = None,
    frame_order: np.ndarray | None = None,
) -> dict[str, Any]:
    """`pred_label`/`aggregated_probability_positive` must already come from
    `canonical_predicted_class()` (or an equivalent exact reproduction of
    evaluate_stage5.predict_h5()'s aggregation) -- this function does not
    derive a predicted class on its own (Step H4.1: an earlier `>= 0.5`
    single-class threshold here was the actual root cause of a spurious
    "parity mismatch", not GPU non-determinism).
    """
    require(vote_count.shape == saved_vote_count.shape, f"{context}: vote_count shape mismatch vs. saved prediction")
    vote_count_mismatch = int(np.sum(vote_count != saved_vote_count))
    require(
        vote_count_mismatch == 0,
        f"{context}: vote_count differs at {vote_count_mismatch} point(s) between this re-inference "
        "and the saved W-A prediction .npz",
    )

    require(
        bool(np.array_equal(training_occurrence_count, vote_count)),
        f"{context}: training window occurrence count differs from this re-inference's vote_count -- "
        "window generation must be identical between Step H2's structural exposure counting and this "
        "per-window forward pass",
    )

    # `overall_max_abs_prob_diff` is a video-wide informational statistic --
    # it is NOT the mismatched point's own probability difference. Each
    # mismatched point's own diff is written per-row into the diff artifact
    # below (`abs_probability_diff`); read that column, not this summary
    # value, when diagnosing a specific point (Step H4.1 policy-chat
    # correction: an earlier version of this function's error message
    # conflated the two).
    overall_max_abs_prob_diff = (
        float(np.max(np.abs(aggregated_probability_positive - saved_prob_femur))) if aggregated_probability_positive.size else 0.0
    )
    mismatch_indices = np.flatnonzero(pred_label != saved_pred_label)
    label_mismatch_count = int(mismatch_indices.size)

    if label_mismatch_count > 0 and diff_artifact_path is not None:
        diff_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with diff_artifact_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "point_index",
                "gt_label",
                "valid",
                "this_run_positive_probability",
                "saved_prob_femur",
                "abs_probability_diff",
                "distance_from_0.5_this_run",
                "distance_from_0.5_saved",
                "this_run_pred_label",
                "saved_pred_label",
                "this_run_vote_count",
                "saved_vote_count",
                "frame_order",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for index in mismatch_indices.tolist():
                writer.writerow(
                    {
                        "point_index": index,
                        "gt_label": int(labels[index]),
                        "valid": bool(valid_mask[index]),
                        "this_run_positive_probability": float(aggregated_probability_positive[index]),
                        "saved_prob_femur": float(saved_prob_femur[index]),
                        "abs_probability_diff": float(abs(aggregated_probability_positive[index] - saved_prob_femur[index])),
                        "distance_from_0.5_this_run": float(abs(aggregated_probability_positive[index] - 0.5)),
                        "distance_from_0.5_saved": float(abs(saved_prob_femur[index] - 0.5)),
                        "this_run_pred_label": int(pred_label[index]),
                        "saved_pred_label": int(saved_pred_label[index]),
                        "this_run_vote_count": int(vote_count[index]),
                        "saved_vote_count": int(saved_vote_count[index]),
                        "frame_order": int(frame_order[index]) if frame_order is not None else None,
                    }
                )

    require(
        label_mismatch_count == 0,
        f"{context}: {label_mismatch_count} point(s) have a different predicted class (canonical 2-class "
        "argmax, matching evaluate_stage5.predict_h5()) between this re-inference and the saved W-A "
        f"prediction .npz (video-wide max abs positive-probability diff={overall_max_abs_prob_diff:.3e}; "
        "this is NOT necessarily the mismatched point's own diff -- see the per-point diff artifact)"
        + (f"; diff artifact saved to {diff_artifact_path}" if diff_artifact_path is not None else ""),
    )

    check_threshold_0p5_parity(
        labels=labels,
        valid_mask=valid_mask,
        pred_label=pred_label,
        row=h5_metrics_row,
        ignore_index=ignore_index,
        context=context,
    )
    return {"max_abs_prob_diff": overall_max_abs_prob_diff, "vote_count_mismatch": vote_count_mismatch}


# ----------------------------------------------------------------------------
# Stratified point-level summaries.
# ----------------------------------------------------------------------------


def bucket_vote_count(vote_count: np.ndarray) -> np.ndarray:
    bounds = np.array([1, 2, 3, 5], dtype=np.int64)
    labels = np.searchsorted(bounds, vote_count, side="right")
    return labels  # 0:count<1(impossible), 1:{1}, 2:{2}, 3:{3,4}, 4:{5+}


VOTE_COUNT_BUCKET_NAMES = {0: "0", 1: "1", 2: "2", 3: "3-4", 4: "5+"}


def summarize_by_group(
    group_values: np.ndarray,
    *,
    scope_mask: np.ndarray,
    disagreement: np.ndarray,
    training_occurrence: np.ndarray,
    positive_vote_ratio: np.ndarray,
    min_edge_distance: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not np.any(scope_mask):
        return rows
    values_in_scope = group_values[scope_mask]
    for group_value in sorted(set(values_in_scope.tolist())):
        mask = scope_mask & (group_values == group_value)
        count = int(np.sum(mask))
        if count == 0:
            continue
        rows.append(
            {
                "group_value": group_value,
                "point_count": count,
                "disagreement_rate": float(np.mean(disagreement[mask])),
                "mean_training_window_occurrence": float(np.mean(training_occurrence[mask])),
                "mean_positive_vote_ratio": float(np.mean(positive_vote_ratio[mask])),
                "mean_min_edge_distance": float(np.mean(min_edge_distance[mask])),
            }
        )
    return rows


def classify_prediction(pred_label: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray, *, ignore_index: int) -> dict[str, np.ndarray]:
    """`pred_label` must be the canonical aggregate class (from
    `canonical_predicted_class()`), not an independently-thresholded
    probability -- see Step H4.1. This keeps the TP/FP/FN/TN stratification
    consistent with the parity gate's own predicted-class convention (per
    policy-chat 8.5 item 3: both must use the same aggregate predicted
    class). Per-window `positive_vote_ratio`/`classify_overlap()` disagreement
    stats intentionally use a *different*, purely diagnostic per-window
    convention (S5-08's `p1 > 0.5`) and must not be confused with this.
    """
    valid = valid_mask & (labels != int(ignore_index))
    target_positive = valid & (labels == 1)
    target_background = valid & (labels == 0)
    predicted_positive = pred_label == 1
    return {
        "tp": target_positive & predicted_positive,
        "fn": target_positive & ~predicted_positive,
        "fp": target_background & predicted_positive,
        "tn": target_background & ~predicted_positive,
    }


# ----------------------------------------------------------------------------
# XY-bin exposure/disagreement vs. Step H3.1 temporal-recurrence cross-check.
# ----------------------------------------------------------------------------


def bin_exposure_disagreement_table(
    *,
    video_alias: str,
    split: str,
    xy_normalized: np.ndarray,
    mask: np.ndarray,
    training_occurrence: np.ndarray,
    disagreement: np.ndarray,
    grid_resolution: int,
) -> list[dict[str, Any]]:
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size == 0:
        return []
    bins = assign_xy_grid_bin(xy_normalized[selected_idx], grid_resolution=grid_resolution)
    rows: list[dict[str, Any]] = []
    combo = np.stack([bins[:, 1], bins[:, 0]], axis=1)
    unique_bins = np.unique(combo, axis=0)
    for row_idx, col_idx in unique_bins.tolist():
        bin_mask = (combo[:, 0] == row_idx) & (combo[:, 1] == col_idx)
        indices = selected_idx[bin_mask]
        rows.append(
            {
                "video_alias": video_alias,
                "split": split,
                "grid_resolution": grid_resolution,
                "bin_row": row_idx,
                "bin_col": col_idx,
                "point_count": int(indices.size),
                "mean_training_window_occurrence": float(np.mean(training_occurrence[indices])),
                "disagreement_rate": float(np.mean(disagreement[indices])),
            }
        )
    return rows


def join_recurrence_with_bin_exposure(
    bin_exposure_rows: list[dict[str, Any]],
    recurrence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-check: does high unmatched-run recurrence (Step H3.1) coincide with
    high training exposure or high per-window disagreement (Step H4), at the
    same (video_alias, split, grid_resolution, bin_row, bin_col)?
    """
    exposure_by_key = {
        (row["video_alias"], row["split"], int(row["grid_resolution"]), int(row["bin_row"]), int(row["bin_col"])): row
        for row in bin_exposure_rows
    }
    matched_multi: list[dict[str, Any]] = []
    matched_single: list[dict[str, Any]] = []
    for row in recurrence_rows:
        key = (
            row["video_alias"],
            row["split"],
            int(row["grid_resolution"]),
            int(row["bin_row"]),
            int(row["bin_col"]),
        )
        exposure_row = exposure_by_key.get(key)
        if exposure_row is None:
            continue
        multiple_unmatched = str(row["multiple_unmatched_runs"]).strip().lower() == "true"
        target = matched_multi if multiple_unmatched else matched_single
        target.append(exposure_row)

    def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
        values = [row[field] for row in rows]
        return float(np.mean(values)) if values else None

    return {
        "bins_with_multiple_unmatched_runs": len(matched_multi),
        "bins_without_multiple_unmatched_runs": len(matched_single),
        "mean_training_window_occurrence_multiple_unmatched": mean_field(matched_multi, "mean_training_window_occurrence"),
        "mean_training_window_occurrence_other": mean_field(matched_single, "mean_training_window_occurrence"),
        "mean_disagreement_rate_multiple_unmatched": mean_field(matched_multi, "disagreement_rate"),
        "mean_disagreement_rate_other": mean_field(matched_single, "disagreement_rate"),
    }


def read_recurrence_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------
# Orchestration (torch required only here, via run_h4_forward).
# ----------------------------------------------------------------------------


def run_h4_forward(
    *,
    model: Any,
    data: dict[str, Any],
    features: tuple[str, ...],
    normalize_points: bool,
    device: Any,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    h5_path: str,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """One per-window forward pass, feeding both:

    - the S5-08 `OverlapAccumulator` (diagnostic-only: edge/center distance,
      per-window disagreement, etc. -- never used for the canonical predicted
      class);
    - a canonical two-class float64 probability sum that exactly reproduces
      evaluate_stage5.predict_h5()'s aggregation path (per-window softmax
      cast to float32 immediately, THEN accumulated in float64), so the
      final predicted class (via `canonical_predicted_class()`) matches the
      saved W-A prediction bit-for-bit absent genuine GPU non-determinism.

    Deliberately does not call check_stage5_overlap_aggregation.run_overlap_forward()
    (S5-08): that function casts each window's softmax straight to float64
    and only ever keeps the positive-class probability, which is what caused
    the Step H4.1 spurious parity mismatch (a tie-breaking/aggregation-path
    difference vs. predict_h5(), not GPU non-determinism). Reusing it here
    for anything parity-critical would silently reintroduce the same bug.
    """
    import torch

    from checks.real_h5.check_stage5_overlap_aggregation import OverlapAccumulator
    from infer_stage5 import build_features
    from stage5.utils.feature_normalization import normalize_xyz
    from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window

    raw_points = data["points"].astype(np.float32)
    points = normalize_xyz(raw_points) if normalize_points else raw_points
    feature_values = build_features(data, features)
    frame_order = data["frame_order"]
    windows = generate_frame_order_windows(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    require(bool(windows), f"No frame windows generated for {h5_path}")

    num_points = points.shape[0]
    accumulator = OverlapAccumulator.zeros(num_points)
    probability_sum_2class = np.zeros((num_points, 2), dtype=np.float64)
    canonical_vote_count = np.zeros(num_points, dtype=np.int32)

    with torch.inference_mode():
        for window in windows:
            indices = point_indices_for_window(frame_order, window)
            if indices.size == 0:
                continue
            batch = {
                "points": torch.from_numpy(points[indices][None]).float().to(device),
                "features": torch.from_numpy(feature_values[indices][None]).float().to(device),
            }
            logits = model(batch)["logits"][0]
            # Cast to float32 immediately after softmax, exactly matching
            # evaluate_stage5.predict_h5()'s per-window precision path.
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
            require(np.all(np.isfinite(probabilities)), f"Non-finite probabilities in {h5_path}")

            accumulator.update(
                indices=indices,
                p1=probabilities[:, 1],
                frame_order_indices=frame_order[indices].astype(np.float64),
                window=window,
            )
            probability_sum_2class[indices] += probabilities.astype(np.float64)
            canonical_vote_count[indices] += 1

    missing = accumulator.vote_count == 0
    require(not bool(np.any(missing)), f"{int(np.sum(missing))} points received no prediction in {h5_path}")
    require(
        bool(np.array_equal(canonical_vote_count, accumulator.vote_count)),
        f"{h5_path}: internal vote_count mismatch between the S5-08 accumulator and the canonical "
        "2-class accumulator -- both are updated from the same window loop and must always agree",
    )
    return accumulator, probability_sum_2class, canonical_vote_count


def process_video_h4(
    *,
    video_alias: str,
    split: str,
    model: Any,
    h5_path: Path,
    npz_path: Path,
    h5_metrics_row: dict[str, str],
    features: tuple[str, ...],
    normalize_points: bool,
    device: Any,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    ignore_index: int,
    pixel_width: int,
    pixel_height: int,
    grid_resolution: int,
    diff_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    # Local import: only this function needs torch, so the rest of the module
    # (and every synthetic test) stays importable/runnable without CUDA.
    from checks.real_h5.check_stage5_overlap_aggregation import (
        classify_overlap,
        derive_point_probabilities,
        gt_class_masks,
        relative_frame_deciles,
    )

    data = load_stage5_pointcloud_h5(h5_path)
    labels = np.asarray(data["point_label"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    frame_order = np.asarray(data["frame_order"])
    pixel_xy = np.asarray(data["pixel_xy"])

    saved_prob_femur, saved_pred_label = load_prediction(npz_path)
    with np.load(npz_path) as saved:
        saved_vote_count = saved["vote_count"].astype(np.int32)

    accumulator, probability_sum_2class, canonical_vote_count = run_h4_forward(
        model=model,
        data=data,
        features=features,
        normalize_points=normalize_points,
        device=device,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
        h5_path=str(h5_path),
    )
    aggregated_probability, pred_label = canonical_predicted_class(probability_sum_2class, canonical_vote_count)
    derived = derive_point_probabilities(accumulator)
    overlap_classes = classify_overlap(accumulator)
    training_occurrence, _num_windows = window_occurrence_counts(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )

    parity = verify_h4_parity(
        vote_count=canonical_vote_count,
        aggregated_probability_positive=aggregated_probability[:, 1],
        pred_label=pred_label,
        saved_prob_femur=saved_prob_femur,
        saved_vote_count=saved_vote_count,
        saved_pred_label=saved_pred_label,
        labels=labels,
        valid_mask=valid_mask,
        training_occurrence_count=training_occurrence,
        h5_metrics_row=h5_metrics_row,
        ignore_index=ignore_index,
        context=f"{split}/{video_alias}",
        diff_artifact_path=(diff_artifact_dir / f"{split}_{video_alias}_parity_diff_DO_NOT_SHARE.csv") if diff_artifact_dir else None,
        frame_order=frame_order,
    )

    gt_masks = gt_class_masks(labels, valid_mask, ignore_index)
    pred_classes = classify_prediction(pred_label, labels, valid_mask, ignore_index=ignore_index)
    deciles = relative_frame_deciles(frame_order)
    vote_buckets = bucket_vote_count(canonical_vote_count)
    xy_normalized = normalize_pixel_xy_with_dimensions(pixel_xy, width=pixel_width, height=pixel_height)

    stratified_rows: list[dict[str, Any]] = []
    for stratum_name, scope_mask in (
        ("valid_positive", gt_masks["valid_positive"]),
        ("valid_background", gt_masks["valid_background"]),
        ("ignore", gt_masks["ignore"]),
        ("tp", pred_classes["tp"]),
        ("fp", pred_classes["fp"]),
        ("fn", pred_classes["fn"]),
        ("tn", pred_classes["tn"]),
    ):
        for group_name, group_values in (
            ("relative_frame_decile", deciles),
            ("vote_count_bucket", vote_buckets),
        ):
            for row in summarize_by_group(
                group_values,
                scope_mask=scope_mask,
                disagreement=overlap_classes["disagreement"],
                training_occurrence=training_occurrence,
                positive_vote_ratio=derived["positive_vote_ratio"],
                min_edge_distance=accumulator.min_edge_distance,
            ):
                row["video_alias"] = video_alias
                row["split"] = split
                row["stratum"] = stratum_name
                row["group_name"] = group_name
                if group_name == "vote_count_bucket":
                    row["group_value"] = VOTE_COUNT_BUCKET_NAMES.get(int(row["group_value"]), str(row["group_value"]))
                stratified_rows.append(row)

    bin_rows = bin_exposure_disagreement_table(
        video_alias=video_alias,
        split=split,
        xy_normalized=xy_normalized,
        mask=pred_classes["tp"] | pred_classes["fp"],
        training_occurrence=training_occurrence,
        disagreement=overlap_classes["disagreement"],
        grid_resolution=grid_resolution,
    )

    return {
        "video_alias": video_alias,
        "split": split,
        "parity": parity,
        "stratified_rows": stratified_rows,
        "bin_rows": bin_rows,
        "video_summary": {
            "video_alias": video_alias,
            "split": split,
            "point_count": int(labels.size),
            "disagreement_rate": float(np.mean(overlap_classes["disagreement"])),
            "max_abs_prob_diff": parity["max_abs_prob_diff"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-14 Step H4: per-window context diagnostics for the fixed 21 videos. "
        "Requires the ONE re-inference this stage allows (per-window forward via the S5-08 "
        "OverlapAccumulator), verified with no-tolerance parity against the saved W-A aggregate."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation_dir", required=True, help="evaluate_stage5.py per-checkpoint output_dir")
    parser.add_argument("--checkpoint_row_name", default="last", help="checkpoint value in h5_metrics.csv")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--pixel_width", type=int, default=256)
    parser.add_argument("--pixel_height", type=int, default=256)
    parser.add_argument("--grid_resolution", type=int, default=16)
    parser.add_argument("--window_size_frames", type=int, default=None)
    parser.add_argument("--window_stride_frames", type=int, default=None)
    parser.add_argument("--include_tail_window", type=int, default=None, choices=[0, 1])
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--xy_temporal_recurrence_csv", default=None, help="Step H3.1 output, for the recurrence/exposure cross-check")
    parser.add_argument("--stratified_summary_csv", default=None)
    parser.add_argument("--bin_exposure_csv", default=None)
    parser.add_argument("--video_summary_csv", default=None)
    parser.add_argument("--recurrence_cross_check_json", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument(
        "--diff_artifact_dir",
        default=None,
        help="Directory for per-video parity-mismatch diff CSVs (private -- point indices/labels/probabilities, "
        "written only if a mismatch actually occurs)",
    )
    return parser.parse_args()


def main() -> None:
    import torch

    from evaluate_stage5 import model_from_checkpoint
    from infer_stage5 import load_checkpoint

    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)

    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but torch.cuda.is_available() is False")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model, config, features, normalize_points = model_from_checkpoint(checkpoint, device=device, strict=args.strict_checkpoint)
    del checkpoint
    window_size_frames = args.window_size_frames if args.window_size_frames is not None else int(config.get("window_size_frames", 16))
    window_stride_frames = args.window_stride_frames if args.window_stride_frames is not None else int(config.get("window_stride_frames", 8))
    include_tail_window = (
        bool(args.include_tail_window) if args.include_tail_window is not None else bool(config.get("include_tail_window", True))
    )

    all_stratified_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    all_video_summary_rows: list[dict[str, Any]] = []

    for split in ("train_sanity", "validation"):
        rows = load_h5_metrics_rows(evaluation_dir / "h5_metrics.csv", checkpoint=args.checkpoint_row_name, split=split)
        for index, row in enumerate(rows):
            # row["video_name"] (real video_name from the raw h5_metrics.csv) is
            # used ONLY to resolve paths on disk; every output row below uses
            # the enumeration-based video_alias. Kept consistent with the fix
            # applied to check_stage5_frame_xy_diagnostics.py in this same
            # session (see revision record for the incident).
            h5_path = Path(row["h5_path"])
            npz_path = evaluation_dir / "predictions" / split / f"{safe_name(row['video_name'])}.npz"
            video_alias = f"{split}_{index:03d}"
            result = process_video_h4(
                video_alias=video_alias,
                split=split,
                model=model,
                h5_path=h5_path,
                npz_path=npz_path,
                h5_metrics_row=row,
                features=features,
                normalize_points=normalize_points,
                device=device,
                window_size_frames=window_size_frames,
                window_stride_frames=window_stride_frames,
                include_tail_window=include_tail_window,
                ignore_index=args.ignore_index,
                pixel_width=args.pixel_width,
                pixel_height=args.pixel_height,
                grid_resolution=args.grid_resolution,
                diff_artifact_dir=Path(args.diff_artifact_dir) if args.diff_artifact_dir else None,
            )
            all_stratified_rows.extend(result["stratified_rows"])
            all_bin_rows.extend(result["bin_rows"])
            all_video_summary_rows.append(result["video_summary"])

    recurrence_cross_check: dict[str, Any] | None = None
    if args.xy_temporal_recurrence_csv:
        recurrence_rows = read_recurrence_csv(Path(args.xy_temporal_recurrence_csv))
        recurrence_cross_check = join_recurrence_with_bin_exposure(all_bin_rows, recurrence_rows)

    def write_csv(path_str: str | None, rows: list[dict[str, Any]]) -> None:
        if not path_str or not rows:
            return
        output = Path(path_str)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.stratified_summary_csv, all_stratified_rows)
    write_csv(args.bin_exposure_csv, all_bin_rows)
    write_csv(args.video_summary_csv, all_video_summary_rows)

    summary = {
        "status": "passed",
        "num_videos": len(all_video_summary_rows),
        "num_stratified_rows": len(all_stratified_rows),
        "num_bin_rows": len(all_bin_rows),
        "max_abs_prob_diff_overall": max((row["max_abs_prob_diff"] for row in all_video_summary_rows), default=0.0),
        "recurrence_cross_check": recurrence_cross_check,
    }
    if args.recurrence_cross_check_json and recurrence_cross_check is not None:
        output = Path(args.recurrence_cross_check_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(recurrence_cross_check, f, indent=2, ensure_ascii=False)
    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-14 Step H4 per-window context diagnostics passed.")
    print(f"videos: {summary['num_videos']}, stratified rows: {summary['num_stratified_rows']}, bin rows: {summary['num_bin_rows']}")
    print(f"max abs mean-probability diff vs saved W-A prediction (informational only): {summary['max_abs_prob_diff_overall']:.3e}")
    if recurrence_cross_check is not None:
        print(f"recurrence/exposure cross-check: {recurrence_cross_check}")
    if args.summary_json:
        print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
