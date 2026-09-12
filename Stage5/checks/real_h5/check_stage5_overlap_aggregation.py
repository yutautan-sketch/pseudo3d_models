from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from evaluate_stage5 import (
    MetricTotals,
    compute_metrics,
    model_from_checkpoint,
    path_video_name,
    write_csv,
)
from export_anonymized_stage5_metrics import assert_share_bundle_anonymous, build_aliases
from infer_stage5 import build_features, load_checkpoint
from stage5.utils.feature_normalization import normalize_xyz
from stage5.utils.frame_windows import FrameWindow, generate_frame_order_windows, point_indices_for_window
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list
from stage5.utils.visualization_export import write_probability_ply, write_selected_probability_ply


AGGREGATION_METHODS: tuple[str, ...] = (
    "mean_probability",
    "max_probability",
    "center_nearest_probability",
    "center_weighted_probability",
)
GT_CLASS_NAMES: tuple[str, ...] = ("valid_positive", "valid_background", "ignore")
STAT_CATEGORY_NAMES: tuple[str, ...] = (
    "valid_positive",
    "valid_background",
    "ignore",
    "all_valid",
    "vote_count_1",
    "vote_count_2plus",
)
PROBABILITY_PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)
COMPARE_COUNT_KEYS: tuple[str, ...] = (
    "true_positive_count",
    "false_positive_count",
    "true_negative_count",
    "false_negative_count",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def safe_array_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator_safe = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, numerator / denominator_safe, 0.0)


class ParityMismatchError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Section 4.1/4.2/4.3: per-point overlap accumulator
# ----------------------------------------------------------------------------


@dataclass
class OverlapAccumulator:
    vote_count: np.ndarray
    prob_sum: np.ndarray
    prob_sum_sq: np.ndarray
    prob_min: np.ndarray
    prob_max: np.ndarray
    positive_vote_count: np.ndarray
    center_weighted_sum: np.ndarray
    center_weight_sum: np.ndarray
    center_nearest_prob: np.ndarray
    center_nearest_distance: np.ndarray
    center_nearest_window_id: np.ndarray
    min_edge_distance: np.ndarray

    @classmethod
    def zeros(cls, num_points: int) -> "OverlapAccumulator":
        return cls(
            vote_count=np.zeros(num_points, dtype=np.int32),
            prob_sum=np.zeros(num_points, dtype=np.float64),
            prob_sum_sq=np.zeros(num_points, dtype=np.float64),
            prob_min=np.full(num_points, np.inf, dtype=np.float64),
            prob_max=np.full(num_points, -np.inf, dtype=np.float64),
            positive_vote_count=np.zeros(num_points, dtype=np.int32),
            center_weighted_sum=np.zeros(num_points, dtype=np.float64),
            center_weight_sum=np.zeros(num_points, dtype=np.float64),
            center_nearest_prob=np.zeros(num_points, dtype=np.float64),
            center_nearest_distance=np.full(num_points, np.inf, dtype=np.float64),
            center_nearest_window_id=np.full(num_points, np.iinfo(np.int64).max, dtype=np.int64),
            min_edge_distance=np.full(num_points, np.iinfo(np.int32).max, dtype=np.int32),
        )

    def update(
        self,
        *,
        indices: np.ndarray,
        p1: np.ndarray,
        frame_order_indices: np.ndarray,
        window: FrameWindow,
    ) -> None:
        require(indices.shape == p1.shape == frame_order_indices.shape, "update() array shapes must match")
        positive = p1 > 0.5

        self.vote_count[indices] += 1
        self.prob_sum[indices] += p1
        self.prob_sum_sq[indices] += p1 * p1
        self.prob_min[indices] = np.minimum(self.prob_min[indices], p1)
        self.prob_max[indices] = np.maximum(self.prob_max[indices], p1)
        self.positive_vote_count[indices] += positive.astype(np.int32)

        edge_distance = np.minimum(
            frame_order_indices - window.start_frame,
            window.end_frame - frame_order_indices,
        ).astype(np.int32)
        weight = (1.0 + edge_distance).astype(np.float64)
        self.center_weighted_sum[indices] += weight * p1
        self.center_weight_sum[indices] += weight
        self.min_edge_distance[indices] = np.minimum(self.min_edge_distance[indices], edge_distance)

        window_center = (window.start_frame + window.end_frame) / 2.0
        dist_to_center = np.abs(frame_order_indices - window_center)
        current_distance = self.center_nearest_distance[indices]
        current_window_id = self.center_nearest_window_id[indices]
        better = (dist_to_center < current_distance) | (
            (dist_to_center == current_distance) & (window.window_id < current_window_id)
        )
        self.center_nearest_prob[indices] = np.where(better, p1, self.center_nearest_prob[indices])
        self.center_nearest_distance[indices] = np.where(better, dist_to_center, current_distance)
        self.center_nearest_window_id[indices] = np.where(
            better, window.window_id, current_window_id
        )


def derive_point_probabilities(accumulator: OverlapAccumulator) -> dict[str, np.ndarray]:
    vote = accumulator.vote_count.astype(np.float64)
    mean = safe_array_divide(accumulator.prob_sum, vote)
    mean_sq = safe_array_divide(accumulator.prob_sum_sq, vote)
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    std = np.sqrt(variance)
    prob_range = np.where(accumulator.vote_count > 0, accumulator.prob_max - accumulator.prob_min, 0.0)
    positive_vote_ratio = safe_array_divide(accumulator.positive_vote_count.astype(np.float64), vote)
    center_weighted = safe_array_divide(accumulator.center_weighted_sum, accumulator.center_weight_sum)
    return {
        "mean_probability": mean,
        "std_probability": std,
        "range_probability": prob_range,
        "positive_vote_ratio": positive_vote_ratio,
        "max_probability": accumulator.prob_max,
        "center_nearest_probability": accumulator.center_nearest_prob,
        "center_weighted_probability": center_weighted,
    }


def classify_overlap(accumulator: OverlapAccumulator) -> dict[str, np.ndarray]:
    vote = accumulator.vote_count
    all_negative = accumulator.positive_vote_count == 0
    all_positive = accumulator.positive_vote_count == vote
    disagreement = (vote >= 2) & ~all_negative & ~all_positive
    return {"all_negative": all_negative, "all_positive": all_positive, "disagreement": disagreement}


def gt_class_masks(labels: np.ndarray, valid_mask: np.ndarray, ignore_index: int) -> dict[str, np.ndarray]:
    valid = valid_mask & (labels != int(ignore_index))
    target_positive = labels == 1
    return {
        "valid_positive": valid & target_positive,
        "valid_background": valid & ~target_positive,
        "ignore": ~valid,
    }


def stat_category_masks(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    ignore_index: int,
    vote_count: np.ndarray,
) -> dict[str, np.ndarray]:
    gt = gt_class_masks(labels, valid_mask, ignore_index)
    return {
        **gt,
        "all_valid": gt["valid_positive"] | gt["valid_background"],
        "vote_count_1": vote_count == 1,
        "vote_count_2plus": vote_count >= 2,
    }


def relative_frame_deciles(frame_order: np.ndarray) -> np.ndarray:
    frame_order = frame_order.astype(np.float64)
    video_min = float(np.min(frame_order))
    video_max = float(np.max(frame_order))
    span = video_max - video_min
    if span <= 0:
        return np.zeros(frame_order.shape, dtype=np.int32)
    relative = (frame_order - video_min) / span
    return np.minimum(np.floor(relative * 10.0), 9.0).astype(np.int32)


def summarize_probability_category(
    mask: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    prob_range: np.ndarray,
    suppressed: np.ndarray,
    vote_count: np.ndarray,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    overlap_count = int(np.sum(mask & (vote_count >= 2)))
    row: dict[str, Any] = {
        "point_count": count,
        "overlap_point_count": overlap_count,
        "overlap_point_rate": safe_divide(overlap_count, count),
        "suppressed_positive_count": int(np.sum(suppressed[mask])),
    }
    for name, values in (
        ("mean_probability", mean),
        ("std_probability", std),
        ("range_probability", prob_range),
    ):
        subset = values[mask]
        row[f"{name}_mean"] = float(np.mean(subset)) if count else None
    for name, values in (("std_probability", std), ("range_probability", prob_range)):
        subset = values[mask]
        for q in PROBABILITY_PERCENTILES:
            row[f"{name}_p{q}"] = float(np.percentile(subset, q)) if count else None
    return row


def summarize_disagreement_category(
    gt_mask: np.ndarray,
    *,
    vote_count: np.ndarray,
    all_negative: np.ndarray,
    all_positive: np.ndarray,
    disagreement: np.ndarray,
    mean_probability: np.ndarray,
    positive_vote_ratio: np.ndarray,
) -> dict[str, Any]:
    scope = gt_mask & (vote_count >= 2)
    count = int(np.sum(scope))
    disagreement_scope = disagreement & scope
    disagreement_count = int(np.sum(disagreement_scope))
    mean_background_count = int(np.sum(disagreement_scope & (mean_probability <= 0.5)))
    ratio_subset = positive_vote_ratio[scope]
    return {
        "point_count_vote2plus": count,
        "all_negative_count": int(np.sum(all_negative & scope)),
        "all_positive_count": int(np.sum(all_positive & scope)),
        "disagreement_count": disagreement_count,
        "disagreement_rate": safe_divide(disagreement_count, count),
        "disagreement_mean_aggregation_background_count": mean_background_count,
        "positive_vote_ratio_p50": float(np.percentile(ratio_subset, 50)) if count else None,
        "positive_vote_ratio_p90": float(np.percentile(ratio_subset, 90)) if count else None,
    }


def build_group_disagreement_rows(
    *,
    group_values: np.ndarray,
    gt_masks: dict[str, np.ndarray],
    disagreement: np.ndarray,
    vote_count: np.ndarray,
    group_column: str,
) -> list[dict[str, Any]]:
    base_mask = vote_count >= 2
    rows: list[dict[str, Any]] = []
    for gt_name, gt_mask in gt_masks.items():
        scope = base_mask & gt_mask
        if not np.any(scope):
            continue
        values_in_scope = group_values[scope]
        disagreement_in_scope = disagreement[scope]
        for group_value in sorted(int(v) for v in np.unique(values_in_scope)):
            value_mask = values_in_scope == group_value
            point_count = int(np.sum(value_mask))
            disagreement_count = int(np.sum(disagreement_in_scope[value_mask]))
            rows.append(
                {
                    "gt_class": gt_name,
                    group_column: group_value,
                    "point_count_vote2plus": point_count,
                    "disagreement_count": disagreement_count,
                    "disagreement_rate": safe_divide(disagreement_count, point_count),
                }
            )
    return rows


# ----------------------------------------------------------------------------
# Forward pass (mirrors evaluate_stage5.predict_h5, extended to keep per-window
# probabilities instead of only the aggregated mean).
# ----------------------------------------------------------------------------


def run_overlap_forward(
    *,
    model: torch.nn.Module,
    data: dict[str, Any],
    features: tuple[str, ...],
    normalize_points: bool,
    device: torch.device,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    h5_path: str,
) -> OverlapAccumulator:
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

    accumulator = OverlapAccumulator.zeros(points.shape[0])
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
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
            require(np.all(np.isfinite(probabilities)), f"Non-finite probabilities in {h5_path}")
            accumulator.update(
                indices=indices,
                p1=probabilities[:, 1],
                frame_order_indices=frame_order[indices].astype(np.float64),
                window=window,
            )

    missing = accumulator.vote_count == 0
    require(
        not np.any(missing),
        f"{int(np.sum(missing))} points received no prediction in {h5_path}",
    )
    return accumulator


# ----------------------------------------------------------------------------
# Mean-baseline parity (section C / F2: compare against the existing
# evaluate_stage5.py output for this checkpoint; no in-process re-run).
# ----------------------------------------------------------------------------


def load_reference_h5_metrics(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    require(path.is_file(), f"Reference h5_metrics.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reference: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["split"], row["video_name"])
        if key in reference:
            raise ValueError(f"Duplicate reference row for split={key[0]!r} video_name={key[1]!r}")
        reference[key] = row
    return reference


def check_mean_probability_parity(
    *,
    split: str,
    video_name: str,
    counts: dict[str, Any],
    reference: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    key = (split, video_name)
    if key not in reference:
        return {"missing_reference_row": True}
    reference_row = reference[key]
    mismatches: dict[str, Any] = {}
    for count_key in COMPARE_COUNT_KEYS:
        expected = int(reference_row[count_key])
        actual = int(counts[count_key])
        if expected != actual:
            mismatches[count_key] = {"expected": expected, "actual": actual}
    return mismatches


# ----------------------------------------------------------------------------
# Per-H5 processing
# ----------------------------------------------------------------------------


def method_metrics(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    probability: np.ndarray,
    *,
    ignore_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred_label = (probability > 0.5).astype(np.uint8)
    return compute_metrics(labels, valid_mask, pred_label, probability, ignore_index=ignore_index)


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
    requested_ply_aliases: set[str],
    ply_output_dir: Path | None,
) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(path)
    video_name = path_video_name(path, data)
    labels = data["point_label"]
    valid_mask = data["valid_mask"]
    frame_order = data["frame_order"]

    accumulator = run_overlap_forward(
        model=model,
        data=data,
        features=features,
        normalize_points=normalize_points,
        device=device,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
        h5_path=str(path),
    )
    probabilities = derive_point_probabilities(accumulator)
    overlap_classes = classify_overlap(accumulator)
    suppressed = (accumulator.prob_max > 0.5) & (probabilities["mean_probability"] <= 0.5)
    gt_masks = gt_class_masks(labels, valid_mask, ignore_index)
    stat_masks = stat_category_masks(labels, valid_mask, ignore_index, accumulator.vote_count)
    frame_decile = relative_frame_deciles(frame_order)

    probability_stat_rows = []
    for category in STAT_CATEGORY_NAMES:
        row = summarize_probability_category(
            stat_masks[category],
            mean=probabilities["mean_probability"],
            std=probabilities["std_probability"],
            prob_range=probabilities["range_probability"],
            suppressed=suppressed,
            vote_count=accumulator.vote_count,
        )
        probability_stat_rows.append({"split": split, "video_alias": alias, "category": category, **row})

    disagreement_rows = []
    for gt_class in GT_CLASS_NAMES:
        row = summarize_disagreement_category(
            gt_masks[gt_class],
            vote_count=accumulator.vote_count,
            all_negative=overlap_classes["all_negative"],
            all_positive=overlap_classes["all_positive"],
            disagreement=overlap_classes["disagreement"],
            mean_probability=probabilities["mean_probability"],
            positive_vote_ratio=probabilities["positive_vote_ratio"],
        )
        disagreement_rows.append({"split": split, "video_alias": alias, "gt_class": gt_class, **row})

    edge_distance_rows = [
        {"split": split, "video_alias": alias, **row}
        for row in build_group_disagreement_rows(
            group_values=accumulator.min_edge_distance,
            gt_masks=gt_masks,
            disagreement=overlap_classes["disagreement"],
            vote_count=accumulator.vote_count,
            group_column="min_edge_distance",
        )
    ]
    frame_decile_rows = [
        {"split": split, "video_alias": alias, **row}
        for row in build_group_disagreement_rows(
            group_values=frame_decile,
            gt_masks=gt_masks,
            disagreement=overlap_classes["disagreement"],
            vote_count=accumulator.vote_count,
            group_column="relative_frame_decile",
        )
    ]

    method_rows: list[dict[str, Any]] = []
    method_counts: dict[str, dict[str, Any]] = {}
    parity_mismatches: dict[str, Any] = {}
    for method in AGGREGATION_METHODS:
        metrics, counts = method_metrics(labels, valid_mask, probabilities[method], ignore_index=ignore_index)
        method_counts[method] = counts
        method_rows.append(
            {
                "split": split,
                "video_alias": alias,
                "method": method,
                "num_points": int(labels.size),
                **metrics,
            }
        )
        if method == "mean_probability":
            parity_mismatches = check_mean_probability_parity(
                split=split,
                video_name=video_name,
                counts=counts,
                reference=reference,
            )

    if alias in requested_ply_aliases:
        require(ply_output_dir is not None, "ply_output_dir must be set when exporting PLY")
        alias_dir = ply_output_dir / alias
        for method in AGGREGATION_METHODS:
            probability = probabilities[method].astype(np.float32)
            write_probability_ply(alias_dir / f"{method}_probability.ply", data["points"], probability)
            write_selected_probability_ply(
                alias_dir / f"{method}_positive.ply",
                data["points"],
                probability,
                probability > 0.5,
            )

    return {
        "video_name": video_name,
        "probability_stat_rows": probability_stat_rows,
        "disagreement_rows": disagreement_rows,
        "edge_distance_rows": edge_distance_rows,
        "frame_decile_rows": frame_decile_rows,
        "method_rows": method_rows,
        "method_counts": method_counts,
        "parity_mismatches": parity_mismatches,
    }


# ----------------------------------------------------------------------------
# Split/checkpoint-level rollups
# ----------------------------------------------------------------------------


def build_checkpoint_summary(
    *,
    totals_by_split_method: dict[tuple[str, str], MetricTotals],
    h5_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, method), totals in totals_by_split_method.items():
        metrics = totals.metrics()
        video_rows = [
            row for row in h5_rows if row["split"] == split and row["method"] == method
        ]
        f1_values = [float(row["f1"]) for row in video_rows]
        iou_values = [float(row["iou_femur"]) for row in video_rows]
        tp_zero_video_count = sum(1 for row in video_rows if int(row["true_positive_count"]) == 0)
        gt_positive_videos = [row for row in video_rows if int(row["valid_positive_count"]) > 0]
        gt_positive_detected = sum(
            1 for row in gt_positive_videos if int(row["true_positive_count"]) > 0
        )
        rows.append(
            {
                "split": split,
                "method": method,
                "num_videos": len(video_rows),
                **metrics,
                "video_mean_f1": float(np.mean(f1_values)) if f1_values else None,
                "video_median_f1": float(np.median(f1_values)) if f1_values else None,
                "video_mean_iou": float(np.mean(iou_values)) if iou_values else None,
                "video_median_iou": float(np.median(iou_values)) if iou_values else None,
                "tp_zero_video_count": tp_zero_video_count,
                "gt_positive_video_count": len(gt_positive_videos),
                "gt_positive_video_detected_count": gt_positive_detected,
                "gt_positive_video_detection_rate": safe_divide(
                    gt_positive_detected, len(gt_positive_videos)
                ),
            }
        )
    return rows


def build_comparison_rows(checkpoint_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_split_method = {(row["split"], row["method"]): row for row in checkpoint_summary_rows}
    rows: list[dict[str, Any]] = []
    for row in checkpoint_summary_rows:
        baseline = by_split_method.get((row["split"], "mean_probability"))
        delta = {}
        if baseline is not None:
            delta = {
                "f1_delta_vs_mean": row["f1"] - baseline["f1"],
                "iou_delta_vs_mean": row["iou_femur"] - baseline["iou_femur"],
                "recall_delta_vs_mean": row["recall"] - baseline["recall"],
                "false_positive_count_delta_vs_mean": row["false_positive_count"]
                - baseline["false_positive_count"],
                "tp_zero_video_count_delta_vs_mean": row["tp_zero_video_count"]
                - baseline["tp_zero_video_count"],
            }
        rows.append({**row, **delta})
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("Stage5 overlap aggregation self-test passed.")
        return

    require(args.checkpoint is not None, "--checkpoint is required unless --self_test is set")
    require(args.selected_train_list is not None, "--selected_train_list is required")
    require(args.val_list is not None, "--val_list is required")
    require(args.selection_summary is not None, "--selection_summary is required")
    require(args.reference_h5_metrics_csv is not None, "--reference_h5_metrics_csv is required")
    require(args.share_output_dir is not None, "--share_output_dir is required")
    require(args.private_output_dir is not None, "--private_output_dir is required")

    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but torch.cuda.is_available() is False")

    share_output_dir = Path(args.share_output_dir)
    private_output_dir = Path(args.private_output_dir)
    share_output_dir.mkdir(parents=True, exist_ok=True)
    private_output_dir.mkdir(parents=True, exist_ok=True)

    requested_ply_aliases = set(args.export_ply_alias or [])
    ply_output_dir: Path | None = None
    if requested_ply_aliases:
        require(args.ply_output_dir is not None, "--ply_output_dir is required when --export_ply_alias is set")
        ply_output_dir = Path(args.ply_output_dir)
        ply_output_dir.mkdir(parents=True, exist_ok=True)

    selected_train_paths = read_path_list(args.selected_train_list)
    val_paths = read_path_list(args.val_list)
    require(bool(selected_train_paths) or bool(val_paths), "No H5 files to process")
    for path in (*selected_train_paths, *val_paths):
        require(path.is_file(), f"H5 not found: {path}")
    duplicate_paths = set(selected_train_paths) & set(val_paths)
    require(not duplicate_paths, f"train_sanity and validation lists overlap: {sorted(duplicate_paths)[:3]}")

    selection_summary = json.loads(Path(args.selection_summary).read_text(encoding="utf-8"))
    reference = load_reference_h5_metrics(Path(args.reference_h5_metrics_csv))

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_epoch = checkpoint.get("epoch")
    model, config, features, normalize_points = model_from_checkpoint(
        checkpoint, device=device, strict=args.strict_checkpoint,
    )
    del checkpoint
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

    # Lightweight pre-pass: only H5 root attributes are read (no point arrays)
    # so alias assignment can happen before the per-H5 forward loop, without
    # loading every H5 twice.
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

    unknown_ply_aliases = requested_ply_aliases - set(aliases.values())
    require(
        not unknown_ply_aliases,
        f"--export_ply_alias names not present in this run's alias set: {sorted(unknown_ply_aliases)}",
    )

    probability_stat_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    edge_distance_rows: list[dict[str, Any]] = []
    frame_decile_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    parity_failure_rows: list[dict[str, Any]] = []
    fulfilled_ply_aliases: set[str] = set()
    totals_by_split_method: dict[tuple[str, str], MetricTotals] = {
        (split, method): MetricTotals.empty() for split, _ in split_paths for method in AGGREGATION_METHODS
    }

    # Only basenames (no absolute host paths) go into the share-side summary;
    # constraint 9 / section 8 forbid absolute paths in shared artifacts.
    summary: dict[str, Any] = {
        "status": "running",
        "run_name": Path(args.checkpoint).resolve().parent.name,
        "checkpoint_name": Path(args.checkpoint).name,
        "checkpoint_epoch": checkpoint_epoch,
        "features": list(features),
        "normalize_points": normalize_points,
        "ignore_index": ignore_index,
        "window_size_frames": window_size_frames,
        "window_stride_frames": window_stride_frames,
        "include_tail_window": bool(include_tail_window),
        "aggregation_methods": list(AGGREGATION_METHODS),
        "mean_baseline_parity_reference": Path(args.reference_h5_metrics_csv).name,
    }
    summary_path = share_output_dir / "overlap_aggregation_summary.json"
    private_run_info_path = private_output_dir / "overlap_aggregation_run_info_DO_NOT_SHARE.json"
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
        for split, paths in split_paths:
            for path in tqdm(paths, desc=f"overlap aggregation {split}"):
                video_name = None
                for row in h5_manifest_rows:
                    if row["split"] == split and row["h5_path"] == str(path):
                        video_name = row["video_name"]
                        break
                require(video_name is not None, f"Missing manifest entry for {path}")
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
                    include_tail_window=bool(include_tail_window),
                    ignore_index=ignore_index,
                    reference=reference,
                    requested_ply_aliases=requested_ply_aliases,
                    ply_output_dir=ply_output_dir,
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
                        f"mean_probability parity mismatch for split={split!r} "
                        f"video={result['video_name']!r}: {result['parity_mismatches']}"
                    )

                probability_stat_rows.extend(result["probability_stat_rows"])
                disagreement_rows.extend(result["disagreement_rows"])
                edge_distance_rows.extend(result["edge_distance_rows"])
                frame_decile_rows.extend(result["frame_decile_rows"])
                method_rows.extend(result["method_rows"])
                if alias in requested_ply_aliases:
                    fulfilled_ply_aliases.add(alias)
                for method in AGGREGATION_METHODS:
                    totals_by_split_method[(split, method)].update(result["method_counts"][method])

        missing_ply_aliases = requested_ply_aliases - fulfilled_ply_aliases
        require(
            not missing_ply_aliases,
            f"--export_ply_alias requested aliases not found in processed splits: {sorted(missing_ply_aliases)}",
        )

        checkpoint_summary_rows = build_checkpoint_summary(
            totals_by_split_method=totals_by_split_method,
            h5_rows=method_rows,
        )
        comparison_rows = build_comparison_rows(checkpoint_summary_rows)

        write_csv(share_output_dir / "overlap_probability_statistics.csv", probability_stat_rows)
        write_csv(share_output_dir / "overlap_class_disagreement.csv", disagreement_rows)
        write_csv(share_output_dir / "disagreement_by_edge_distance.csv", edge_distance_rows)
        write_csv(share_output_dir / "disagreement_by_relative_frame_decile.csv", frame_decile_rows)
        write_csv(share_output_dir / "aggregation_h5_metrics.csv", method_rows)
        write_csv(share_output_dir / "aggregation_checkpoint_summary.csv", checkpoint_summary_rows)
        write_csv(share_output_dir / "aggregation_comparison.csv", comparison_rows)

        summary.update(
            {
                "status": "passed",
                "mean_baseline_parity": "passed",
                "num_videos": len(method_rows) // len(AGGREGATION_METHODS) if method_rows else 0,
                "ply_export_aliases": sorted(fulfilled_ply_aliases),
                "checkpoint_summary": checkpoint_summary_rows,
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

    # Defense-in-depth: the same anonymization check export_anonymized_stage5_metrics.py
    # runs on its own share bundle (section 8 / constraint 9: no raw identifiers, no
    # absolute host paths, no private mapping in share_output_dir).
    try:
        assert_share_bundle_anonymous(share_output_dir, private_rows=private_rows)
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = f"share bundle anonymization check failed: {error}"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        raise

    print("Stage5 overlap aggregation checker passed.")
    print(f"checkpoint: {args.checkpoint}")
    print(f"videos processed: {summary['num_videos']}")
    print(f"share output: {share_output_dir}")
    print(f"private output: {private_output_dir}")
    if ply_output_dir is not None:
        print(f"ply output: {ply_output_dir}")


# ----------------------------------------------------------------------------
# Step A1: synthetic accumulator self-test (no checkpoint/H5 required)
# ----------------------------------------------------------------------------


def _make_window(window_id: int, start: int, end: int) -> FrameWindow:
    return FrameWindow(window_id=window_id, start_frame=start, end_frame=end)


def test_accumulator_basic_stats() -> None:
    # 3 points, one window covering all of them.
    acc = OverlapAccumulator.zeros(3)
    window = _make_window(0, 0, 15)
    indices = np.array([0, 1, 2], dtype=np.int64)
    p1 = np.array([0.2, 0.6, 0.9], dtype=np.float64)
    frame_order = np.array([0, 7, 15], dtype=np.float64)
    acc.update(indices=indices, p1=p1, frame_order_indices=frame_order, window=window)

    require(np.array_equal(acc.vote_count, [1, 1, 1]), "vote_count after one window must be 1 for all points")
    require(np.allclose(acc.prob_sum, p1), "prob_sum must equal p1 after a single vote")
    require(np.allclose(acc.prob_sum_sq, p1 * p1), "prob_sum_sq must equal p1^2 after a single vote")
    require(np.allclose(acc.prob_min, p1), "prob_min must equal p1 after a single vote")
    require(np.allclose(acc.prob_max, p1), "prob_max must equal p1 after a single vote")
    require(
        np.array_equal(acc.positive_vote_count, [0, 1, 1]),
        "positive_vote_count must reflect p1 > 0.5 (point 1: 0.6>0.5, point 2: 0.9>0.5)",
    )


def test_derived_stats_two_votes() -> None:
    acc = OverlapAccumulator.zeros(1)
    window_a = _make_window(0, 0, 15)
    window_b = _make_window(1, 8, 23)
    acc.update(
        indices=np.array([0]),
        p1=np.array([0.2]),
        frame_order_indices=np.array([10.0]),
        window=window_a,
    )
    acc.update(
        indices=np.array([0]),
        p1=np.array([0.8]),
        frame_order_indices=np.array([10.0]),
        window=window_b,
    )
    probabilities = derive_point_probabilities(acc)
    require(acc.vote_count[0] == 2, "vote_count must be 2 after two windows")
    expected_mean = (0.2 + 0.8) / 2
    require(
        np.isclose(probabilities["mean_probability"][0], expected_mean),
        f"mean_probability {probabilities['mean_probability'][0]} != {expected_mean}",
    )
    expected_std = float(np.std([0.2, 0.8]))
    require(
        np.isclose(probabilities["std_probability"][0], expected_std),
        f"std_probability {probabilities['std_probability'][0]} != {expected_std}",
    )
    expected_range = 0.8 - 0.2
    require(
        np.isclose(probabilities["range_probability"][0], expected_range),
        f"range_probability {probabilities['range_probability'][0]} != {expected_range}",
    )


def test_classification_all_negative_all_positive_disagreement() -> None:
    acc = OverlapAccumulator.zeros(3)
    windows = [_make_window(0, 0, 15), _make_window(1, 8, 23)]
    # point 0: both windows negative. point 1: both positive. point 2: disagreement.
    p1_by_window = {0: np.array([0.1, 0.9, 0.1]), 1: np.array([0.2, 0.8, 0.9])}
    for window in windows:
        acc.update(
            indices=np.array([0, 1, 2]),
            p1=p1_by_window[window.window_id],
            frame_order_indices=np.array([10.0, 10.0, 10.0]),
            window=window,
        )
    classes = classify_overlap(acc)
    require(bool(classes["all_negative"][0]), "point 0 must be all_negative")
    require(bool(classes["all_positive"][1]), "point 1 must be all_positive")
    require(bool(classes["disagreement"][2]), "point 2 must be disagreement")
    require(not bool(classes["disagreement"][0]), "point 0 must not be disagreement")
    require(not bool(classes["disagreement"][1]), "point 1 must not be disagreement")


def test_center_nearest_and_center_weighted() -> None:
    acc = OverlapAccumulator.zeros(1)
    window_a = _make_window(0, 0, 15)  # center 7.5
    window_b = _make_window(1, 8, 23)  # center 15.5
    # point at frame_order=12: distance to A center=4.5, to B center=3.5 -> B nearer.
    acc.update(indices=np.array([0]), p1=np.array([0.3]), frame_order_indices=np.array([12.0]), window=window_a)
    acc.update(indices=np.array([0]), p1=np.array([0.7]), frame_order_indices=np.array([12.0]), window=window_b)
    require(
        np.isclose(acc.center_nearest_prob[0], 0.7),
        "center_nearest must pick window B's probability (closer center)",
    )

    probabilities = derive_point_probabilities(acc)
    # edge_distance in A: min(12-0, 15-12)=3 -> weight 4. edge_distance in B: min(12-8, 23-12)=4 -> weight 5.
    expected_weighted = (4 * 0.3 + 5 * 0.7) / (4 + 5)
    require(
        np.isclose(probabilities["center_weighted_probability"][0], expected_weighted),
        f"center_weighted_probability {probabilities['center_weighted_probability'][0]} != {expected_weighted}",
    )


def test_center_nearest_tie_break_prefers_smaller_window_id() -> None:
    acc = OverlapAccumulator.zeros(1)
    window_a = _make_window(0, 0, 15)  # center 7.5
    window_b = _make_window(1, 8, 23)  # center 15.5
    # frame_order=11.5 is equidistant (4.0) from both centers.
    acc.update(indices=np.array([0]), p1=np.array([0.1]), frame_order_indices=np.array([11.5]), window=window_a)
    acc.update(indices=np.array([0]), p1=np.array([0.9]), frame_order_indices=np.array([11.5]), window=window_b)
    require(
        np.isclose(acc.center_nearest_prob[0], 0.1),
        "on an exact tie, center_nearest must keep the smaller window_id's probability",
    )

    # Process the tie in reverse order: the smaller window_id must still win.
    acc_reverse = OverlapAccumulator.zeros(1)
    acc_reverse.update(
        indices=np.array([0]), p1=np.array([0.9]), frame_order_indices=np.array([11.5]), window=window_b
    )
    acc_reverse.update(
        indices=np.array([0]), p1=np.array([0.1]), frame_order_indices=np.array([11.5]), window=window_a
    )
    require(
        np.isclose(acc_reverse.center_nearest_prob[0], 0.1),
        "tie-break must not depend on processing order",
    )


def test_tail_window_edge_distance() -> None:
    # window_size=16, stride=8, frames 0..20 -> windows [0,15], [8,20] (tail, len 13).
    frame_order = np.arange(21)
    windows = generate_frame_order_windows(
        frame_order, window_size_frames=16, window_stride_frames=8, include_tail_window=True
    )
    require(len(windows) == 2, f"Expected 2 windows for frames 0..20, got {len(windows)}")
    tail = windows[-1]
    require(tail.start_frame == 8 and tail.end_frame == 20, f"Unexpected tail window: {tail}")

    acc = OverlapAccumulator.zeros(int(frame_order.size))
    indices = point_indices_for_window(frame_order, tail)
    p1 = np.full(indices.shape, 0.5, dtype=np.float64)
    acc.update(
        indices=indices,
        p1=p1,
        frame_order_indices=frame_order[indices].astype(np.float64),
        window=tail,
    )
    # frame 20 is the tail window's end -> edge_distance 0. frame 14 (offset 6 of 12) -> min(6,6)=6.
    require(acc.min_edge_distance[20] == 0, "tail window end frame must have edge_distance 0")
    require(acc.min_edge_distance[14] == 6, f"unexpected edge_distance at frame 14: {acc.min_edge_distance[14]}")


def test_vote_count_one_matches_all_methods() -> None:
    acc = OverlapAccumulator.zeros(1)
    window = _make_window(0, 0, 15)
    acc.update(indices=np.array([0]), p1=np.array([0.37]), frame_order_indices=np.array([5.0]), window=window)
    probabilities = derive_point_probabilities(acc)
    for method in AGGREGATION_METHODS:
        require(
            np.isclose(probabilities[method][0], 0.37),
            f"{method} must equal the single vote's probability when vote_count==1",
        )


def test_probability_half_is_background() -> None:
    p1 = np.array([0.5])
    positive = p1 > 0.5
    require(not bool(positive[0]), "p1 == 0.5 must be classified as background (tie -> background)")


def run_self_test() -> None:
    test_accumulator_basic_stats()
    test_derived_stats_two_votes()
    test_classification_all_negative_all_positive_disagreement()
    test_center_nearest_and_center_weighted()
    test_center_nearest_tie_break_prefers_smaller_window_id()
    test_tail_window_edge_distance()
    test_vote_count_one_matches_all_methods()
    test_probability_half_is_background()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare overlap-window probability aggregation methods on Stage5 H5 files"
    )
    parser.add_argument("--self_test", action="store_true", help="Run Step A1 synthetic accumulator tests and exit")
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
        help="Existing evaluate_stage5.py h5_metrics.csv for the same checkpoint (mean baseline parity)",
    )
    parser.add_argument("--share_output_dir", default=None)
    parser.add_argument("--private_output_dir", default=None)
    parser.add_argument("--ply_output_dir", default=None)
    parser.add_argument(
        "--export_ply_alias",
        action="append",
        default=None,
        help="Anonymized alias (e.g. train_sanity_fixed_001) to export per-method PLY for; repeatable",
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
