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
from checks.real_h5.check_stage5_frame_spatial_overlap_diagnostics import (
    relative_frame_decile,
    require,
    window_occurrence_counts,
)
from checks.real_h5.check_stage5_frame_xy_diagnostics import assign_xy_grid_bin
from checks.real_h5.check_stage5_xy_coordinate_provenance import normalize_pixel_xy_with_dimensions
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list

# ----------------------------------------------------------------------------
# S5-14補足 Step S1-S5: denominator-carrying XY-bin FPR/recall, a train-GT-only
# XY prior with hot/cold FPR comparison, within-video paired temporal
# recall, and vote-count-joint exposure/FPR stratification. Reuses W-A's
# already-saved evaluate_stage5.py prediction .npz artifacts (pred_label,
# vote_count) and the Step H2.5-confirmed 256x256 local-crop normalization.
# CPU/h5py/NumPy only -- no torch/CUDA import anywhere in this module, and
# NO new model forward pass (per-window `vote_count` is taken from the
# already-saved Step H4 prediction .npz and cross-checked against a purely
# structural recount of window_occurrence_counts() from frame_order alone).
# ----------------------------------------------------------------------------

GRID_RESOLUTIONS_DEFAULT = (16, 8)


# ----------------------------------------------------------------------------
# Step S1: denominator-carrying per-video x grid x bin confusion counts.
# ----------------------------------------------------------------------------


def xy_bin_denominator_rows(
    *,
    video_alias: str,
    split: str,
    xy_normalized: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    ignore_index: int,
    grid_resolution: int,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid = valid_mask & (labels != int(ignore_index))
    target_positive = valid & (labels == 1)
    target_background = valid & (labels == 0)
    ignore = ~valid
    predicted_positive = np.asarray(pred_label) == 1

    tp_mask = target_positive & predicted_positive
    fp_mask = target_background & predicted_positive
    tn_mask = target_background & ~predicted_positive
    fn_mask = target_positive & ~predicted_positive
    ignore_predicted_positive_mask = ignore & predicted_positive

    bins = assign_xy_grid_bin(xy_normalized, grid_resolution=grid_resolution)
    row_idx = bins[:, 1]
    col_idx = bins[:, 0]

    def bin_sum(mask: np.ndarray) -> np.ndarray:
        counts = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
        if np.any(mask):
            np.add.at(counts, (row_idx[mask], col_idx[mask]), 1)
        return counts

    all_points = np.ones(labels.shape[0], dtype=bool)
    total_c = bin_sum(all_points)
    valid_c = bin_sum(valid)
    positive_c = bin_sum(target_positive)
    background_c = bin_sum(target_background)
    ignore_c = bin_sum(ignore)
    tp_c = bin_sum(tp_mask)
    fp_c = bin_sum(fp_mask)
    tn_c = bin_sum(tn_mask)
    fn_c = bin_sum(fn_mask)
    pred_pos_all_c = bin_sum(predicted_positive)
    pred_pos_ignore_c = bin_sum(ignore_predicted_positive_mask)

    rows: list[dict[str, Any]] = []
    for r in range(grid_resolution):
        for c in range(grid_resolution):
            tp = int(tp_c[r, c])
            fp = int(fp_c[r, c])
            tn = int(tn_c[r, c])
            fn = int(fn_c[r, c])
            v = int(valid_c[r, c])
            fpr_denominator = fp + tn
            recall_denominator = tp + fn
            rows.append(
                {
                    "video_alias": video_alias,
                    "split": split,
                    "grid_resolution": grid_resolution,
                    "bin_row": r,
                    "bin_col": c,
                    "total_point_count": int(total_c[r, c]),
                    "valid_point_count": v,
                    "valid_positive_count": int(positive_c[r, c]),
                    "valid_background_count": int(background_c[r, c]),
                    "ignore_count": int(ignore_c[r, c]),
                    "true_positive_count": tp,
                    "false_positive_count": fp,
                    "true_negative_count": tn,
                    "false_negative_count": fn,
                    "predicted_positive_count_all": int(pred_pos_all_c[r, c]),
                    "predicted_positive_count_ignore": int(pred_pos_ignore_c[r, c]),
                    "false_positive_rate": (fp / fpr_denominator) if fpr_denominator > 0 else None,
                    "false_positive_rate_available": fpr_denominator > 0,
                    "recall": (tp / recall_denominator) if recall_denominator > 0 else None,
                    "recall_available": recall_denominator > 0,
                    "predicted_positive_rate_valid": ((tp + fp) / v) if v > 0 else None,
                    "gt_positive_rate_valid": (int(positive_c[r, c]) / v) if v > 0 else None,
                }
            )
    return rows


def verify_bin_denominator_parity(
    rows: list[dict[str, Any]],
    *,
    h5_metrics_row: dict[str, str],
    total_point_count: int,
    context: str,
) -> None:
    """Every point falls in exactly one bin (assign_xy_grid_bin clips [0,1] into
    [0, grid_resolution)) -- so summing this video's bins must reproduce both
    the saved threshold-0.5 confusion counts AND the raw point count exactly.
    """
    sums = {
        key: sum(row[key] for row in rows)
        for key in (
            "true_positive_count",
            "false_positive_count",
            "true_negative_count",
            "false_negative_count",
            "total_point_count",
            "ignore_count",
        )
    }
    for key in ("true_positive_count", "false_positive_count", "true_negative_count", "false_negative_count"):
        expected = int(h5_metrics_row[key])
        require(
            sums[key] == expected,
            f"{context}: bin-summed {key}={sums[key]} != h5_metrics.csv {expected}",
        )
    require(
        sums["total_point_count"] == total_point_count,
        f"{context}: bin-summed total_point_count={sums['total_point_count']} != H5 point count {total_point_count}",
    )


# ----------------------------------------------------------------------------
# Step S2: train-GT-only XY prior (two denominator definitions) and per-video
# hot/cold-bin FPR comparison, with train-sanity self-exclusion.
# ----------------------------------------------------------------------------


def train_video_bin_prior_counts(
    h5_path: Path,
    *,
    ignore_index: int,
    pixel_width: int,
    pixel_height: int,
    grid_resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (raw_gt_positive_count, valid_point_count), each [grid, grid],
    for one train H5. GT/valid_mask only -- no prediction needed."""
    data = load_stage5_pointcloud_h5(h5_path)
    labels = np.asarray(data["point_label"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    pixel_xy = np.asarray(data["pixel_xy"])
    valid = valid_mask & (labels != int(ignore_index))
    target_positive = valid & (labels == 1)

    xy_normalized = normalize_pixel_xy_with_dimensions(pixel_xy, width=pixel_width, height=pixel_height)
    bins = assign_xy_grid_bin(xy_normalized, grid_resolution=grid_resolution)
    row_idx, col_idx = bins[:, 1], bins[:, 0]

    gt_pos = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
    valid_c = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
    if np.any(target_positive):
        np.add.at(gt_pos, (row_idx[target_positive], col_idx[target_positive]), 1)
    if np.any(valid):
        np.add.at(valid_c, (row_idx[valid], col_idx[valid]), 1)
    return gt_pos, valid_c


def build_train_prior(
    train_h5_paths: list[Path],
    *,
    ignore_index: int,
    pixel_width: int,
    pixel_height: int,
    grid_resolutions: list[int],
) -> dict[str, Any]:
    per_video: dict[Path, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    totals: dict[int, tuple[np.ndarray, np.ndarray]] = {
        grid: (np.zeros((grid, grid), dtype=np.int64), np.zeros((grid, grid), dtype=np.int64)) for grid in grid_resolutions
    }
    for h5_path in train_h5_paths:
        resolved = h5_path.resolve()
        per_video[resolved] = {}
        for grid in grid_resolutions:
            gt_pos, valid_c = train_video_bin_prior_counts(
                h5_path,
                ignore_index=ignore_index,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                grid_resolution=grid,
            )
            per_video[resolved][grid] = (gt_pos, valid_c)
            total_gt, total_valid = totals[grid]
            totals[grid] = (total_gt + gt_pos, total_valid + valid_c)
    return {"totals": totals, "per_video": per_video, "num_videos": len(train_h5_paths)}


def prior_excluding_video(
    prior: dict[str, Any], *, h5_path: Path, grid_resolution: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Returns (gt_pos, valid_count, self_excluded). Subtracts the given
    video's own contribution from the full train prior if (and only if) that
    video is itself part of the train list (train-sanity self-exclusion)."""
    total_gt, total_valid = prior["totals"][grid_resolution]
    resolved = h5_path.resolve()
    own = prior["per_video"].get(resolved)
    if own is None:
        return total_gt, total_valid, False
    own_gt, own_valid = own[grid_resolution]
    return total_gt - own_gt, total_valid - own_valid, True


def compute_prior_value_grid(gt_pos_grid: np.ndarray, valid_grid: np.ndarray, *, definition: str) -> np.ndarray:
    require(definition in ("raw_count", "rate"), f"unknown prior definition: {definition}")
    if definition == "raw_count":
        return gt_pos_grid.astype(np.float64)
    rate = np.full(gt_pos_grid.shape, np.nan, dtype=np.float64)
    mask = valid_grid > 0
    rate[mask] = gt_pos_grid[mask] / valid_grid[mask]
    return rate


def classify_hot_cold_bins(values: np.ndarray, eligible_mask: np.ndarray) -> dict[str, Any]:
    """Top/bottom-25% hot/cold classification over `values` restricted to
    `eligible_mask` (train valid-point-count > 0 bins). Ties at the quantile
    boundary are kept together via an inclusive threshold (never arbitrarily
    split by coordinate order). If the 75th and 25th percentile coincide
    (heavily tied distribution, e.g. mostly-zero bins), hot/cold cannot be
    separated and the comparison is reported undefined for this definition.
    """
    if not np.any(eligible_mask):
        return {
            "status": "no_eligible_bins",
            "quantile_method": "linear",
            "q75": None,
            "q25": None,
            "hot_mask": None,
            "cold_mask": None,
            "eligible_bin_count": 0,
        }
    eligible_values = values[eligible_mask]
    q75 = float(np.quantile(eligible_values, 0.75))
    q25 = float(np.quantile(eligible_values, 0.25))
    if q75 <= q25:
        return {
            "status": "undefined_boundary_collision",
            "quantile_method": "linear",
            "q75": q75,
            "q25": q25,
            "hot_mask": None,
            "cold_mask": None,
            "eligible_bin_count": int(np.sum(eligible_mask)),
        }
    hot_mask = eligible_mask & (values >= q75)
    cold_mask = eligible_mask & (values <= q25)
    return {
        "status": "ok",
        "quantile_method": "linear",
        "q75": q75,
        "q25": q25,
        "hot_mask": hot_mask,
        "cold_mask": cold_mask,
        "hot_bin_count": int(np.sum(hot_mask)),
        "cold_bin_count": int(np.sum(cold_mask)),
        "eligible_bin_count": int(np.sum(eligible_mask)),
    }


def hot_cold_fpr_for_video(
    bin_rows: list[dict[str, Any]],
    *,
    hot_mask: np.ndarray,
    cold_mask: np.ndarray,
    grid_resolution: int,
) -> dict[str, Any]:
    """Restricted to this evaluated video's own GT-positive-count==0 bins
    (never the train prior's positive/negative labeling), computes hot vs.
    cold FPR = FP / (FP + TN) over valid background points."""
    fp_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
    tn_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
    gt_pos_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
    for row in bin_rows:
        r, c = row["bin_row"], row["bin_col"]
        fp_grid[r, c] = row["false_positive_count"]
        tn_grid[r, c] = row["true_negative_count"]
        gt_pos_grid[r, c] = row["valid_positive_count"]

    no_gt_mask = gt_pos_grid == 0
    hot_eval_mask = hot_mask & no_gt_mask
    cold_eval_mask = cold_mask & no_gt_mask

    def pooled(mask: np.ndarray) -> tuple[int, int, float | None]:
        fp = int(fp_grid[mask].sum())
        tn = int(tn_grid[mask].sum())
        denom = fp + tn
        return fp, tn, (fp / denom if denom > 0 else None)

    hot_fp, hot_tn, hot_fpr = pooled(hot_eval_mask)
    cold_fp, cold_tn, cold_fpr = pooled(cold_eval_mask)
    hot_available = hot_fpr is not None
    cold_available = cold_fpr is not None

    diff = (hot_fpr - cold_fpr) if (hot_available and cold_available) else None
    ratio: float | None = None
    ratio_undefined_reason: str | None = None
    if hot_available and cold_available:
        if cold_fpr is not None and cold_fpr > 0:
            ratio = hot_fpr / cold_fpr
        else:
            ratio_undefined_reason = "cold_fpr_is_zero"
    else:
        ratio_undefined_reason = "hot_or_cold_fpr_unavailable"

    return {
        "hot_bin_no_gt_count": int(np.sum(hot_eval_mask)),
        "cold_bin_no_gt_count": int(np.sum(cold_eval_mask)),
        "hot_fp": hot_fp,
        "hot_tn": hot_tn,
        "cold_fp": cold_fp,
        "cold_tn": cold_tn,
        "hot_fpr": hot_fpr,
        "hot_fpr_available": hot_available,
        "cold_fpr": cold_fpr,
        "cold_fpr_available": cold_available,
        "fpr_diff_hot_minus_cold": diff,
        "fpr_ratio_hot_over_cold": ratio,
        "ratio_undefined_reason": ratio_undefined_reason,
    }


def aggregate_hot_cold_across_videos(per_video_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [r for r in per_video_rows if r["hot_fpr_available"] and r["cold_fpr_available"]]
    diffs = [r["fpr_diff_hot_minus_cold"] for r in comparable]
    pooled_hot_fp = sum(r["hot_fp"] for r in comparable)
    pooled_hot_tn = sum(r["hot_tn"] for r in comparable)
    pooled_cold_fp = sum(r["cold_fp"] for r in comparable)
    pooled_cold_tn = sum(r["cold_tn"] for r in comparable)
    pooled_hot_fpr = pooled_hot_fp / (pooled_hot_fp + pooled_hot_tn) if (pooled_hot_fp + pooled_hot_tn) > 0 else None
    pooled_cold_fpr = pooled_cold_fp / (pooled_cold_fp + pooled_cold_tn) if (pooled_cold_fp + pooled_cold_tn) > 0 else None
    return {
        "num_videos_total": len(per_video_rows),
        "num_videos_comparable": len(comparable),
        "num_videos_hot_gt_cold": sum(1 for d in diffs if d > 0),
        "num_videos_hot_lt_cold": sum(1 for d in diffs if d < 0),
        "num_videos_hot_eq_cold": sum(1 for d in diffs if d == 0),
        "point_weighted_pooled_hot_fpr": pooled_hot_fpr,
        "point_weighted_pooled_cold_fpr": pooled_cold_fpr,
        "point_weighted_pooled_fpr_diff": (
            (pooled_hot_fpr - pooled_cold_fpr) if (pooled_hot_fpr is not None and pooled_cold_fpr is not None) else None
        ),
        "video_equal_weight_mean_fpr_diff": float(np.mean(diffs)) if diffs else None,
        "video_equal_weight_median_fpr_diff": float(np.median(diffs)) if diffs else None,
    }


# ----------------------------------------------------------------------------
# Step S3/S4 shared primitive: per-video x time-half x vote-count{1,2} joint
# TP/FP/TN/FN (structural vote_count, no re-inference).
# ----------------------------------------------------------------------------


def joint_half_vote_count_rows(
    *,
    video_alias: str,
    split: str,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    vote_count: np.ndarray,
    half: np.ndarray,
    ignore_index: int,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool) & (labels != int(ignore_index))
    target_positive = valid & (labels == 1)
    target_background = valid & (labels == 0)
    predicted_positive = np.asarray(pred_label) == 1

    rows: list[dict[str, Any]] = []
    for half_name in ("first_half", "second_half"):
        half_mask = half == half_name
        for vc in (1, 2):
            vc_mask = half_mask & (vote_count == vc)
            tp = int(np.sum(vc_mask & target_positive & predicted_positive))
            fp = int(np.sum(vc_mask & target_background & predicted_positive))
            tn = int(np.sum(vc_mask & target_background & ~predicted_positive))
            fn = int(np.sum(vc_mask & target_positive & ~predicted_positive))
            valid_n = int(np.sum(vc_mask & valid))
            fpr_denom = fp + tn
            recall_denom = tp + fn
            rows.append(
                {
                    "video_alias": video_alias,
                    "split": split,
                    "time_half": half_name,
                    "vote_count": vc,
                    "valid_point_count": valid_n,
                    "true_positive_count": tp,
                    "false_positive_count": fp,
                    "true_negative_count": tn,
                    "false_negative_count": fn,
                    "false_positive_rate": (fp / fpr_denom) if fpr_denom > 0 else None,
                    "false_positive_rate_available": fpr_denom > 0,
                    "recall": (tp / recall_denom) if recall_denom > 0 else None,
                    "recall_available": recall_denom > 0,
                }
            )
    return rows


# ----------------------------------------------------------------------------
# Orchestration: one evaluated video (Step S1 bins for every grid + Step
# S3/S4 joint rows), with structural vote_count parity against the saved
# Step H4 prediction .npz (no model forward pass).
# ----------------------------------------------------------------------------


def process_video_supplement(
    *,
    video_alias: str,
    split: str,
    h5_path: Path,
    npz_path: Path,
    h5_metrics_row: dict[str, str],
    ignore_index: int,
    pixel_width: int,
    pixel_height: int,
    grid_resolutions: list[int],
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(h5_path)
    labels = np.asarray(data["point_label"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    frame_order = np.asarray(data["frame_order"])
    pixel_xy = np.asarray(data["pixel_xy"])

    prob_femur, pred_label = load_prediction(npz_path)
    require(prob_femur.shape == labels.shape, f"{video_alias}: prediction shape {prob_femur.shape} != H5 label shape {labels.shape}")
    with np.load(npz_path) as saved:
        require("vote_count" in saved, f"{npz_path}: missing vote_count (expected to be saved by the earlier Step H4 run)")
        saved_vote_count = saved["vote_count"].astype(np.int64)
    require(saved_vote_count.shape == labels.shape, f"{video_alias}: saved vote_count shape != H5 label shape")

    check_threshold_0p5_parity(
        labels=labels,
        valid_mask=valid_mask,
        pred_label=pred_label,
        row=h5_metrics_row,
        ignore_index=ignore_index,
        context=f"{split}/{video_alias}",
    )

    # Purely structural recount of window occurrence from frame_order alone
    # (no model forward pass) -- must exactly match the vote_count already
    # saved by the earlier Step H4 GPU run.
    reconstructed_vote_count, _num_windows = window_occurrence_counts(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    vote_count_mismatch = int(np.sum(reconstructed_vote_count != saved_vote_count))
    require(
        vote_count_mismatch == 0,
        f"{split}/{video_alias}: structurally reconstructed window-occurrence count differs from the saved "
        f"Step H4 prediction .npz vote_count at {vote_count_mismatch} point(s) -- window parameters must "
        "match the Step H4 run exactly",
    )

    xy_normalized = normalize_pixel_xy_with_dimensions(pixel_xy, width=pixel_width, height=pixel_height)

    point_decile = relative_frame_decile(frame_order, frame_order)
    half = np.where(point_decile <= 4, "first_half", "second_half")

    bin_rows_by_grid: dict[int, list[dict[str, Any]]] = {}
    for grid in grid_resolutions:
        rows = xy_bin_denominator_rows(
            video_alias=video_alias,
            split=split,
            xy_normalized=xy_normalized,
            labels=labels,
            valid_mask=valid_mask,
            pred_label=pred_label,
            ignore_index=ignore_index,
            grid_resolution=grid,
        )
        verify_bin_denominator_parity(
            rows,
            h5_metrics_row=h5_metrics_row,
            total_point_count=int(labels.shape[0]),
            context=f"{split}/{video_alias}/grid{grid}",
        )
        bin_rows_by_grid[grid] = rows

    joint_rows = joint_half_vote_count_rows(
        video_alias=video_alias,
        split=split,
        labels=labels,
        valid_mask=valid_mask,
        pred_label=pred_label,
        vote_count=saved_vote_count,
        half=half,
        ignore_index=ignore_index,
    )

    return {
        "video_alias": video_alias,
        "split": split,
        "h5_path": h5_path,
        "bin_rows_by_grid": bin_rows_by_grid,
        "joint_rows": joint_rows,
        "vote_count_parity_mismatch": vote_count_mismatch,
    }


# ----------------------------------------------------------------------------
# Step S3: frame_metrics.csv (existing Step H3 artifact) -> video x decile
# aggregation and within-video paired first-half/second-half recall.
# ----------------------------------------------------------------------------


def _parse_optional_float(value: str) -> float | None:
    if value is None or value == "" or value.lower() == "none":
        return None
    return float(value)


def read_frame_metrics_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing frame_metrics.csv (Step H3 output): {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_video_decile_counts(frame_metrics_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in frame_metrics_rows:
        key = (row["video_alias"], row["split"], int(row["relative_frame_decile"]))
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (video_alias, split, decile), group_rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])):
        tp = sum(int(r["true_positive_count"]) for r in group_rows)
        fp = sum(int(r["false_positive_count"]) for r in group_rows)
        tn = sum(int(r["true_negative_count"]) for r in group_rows)
        fn = sum(int(r["false_negative_count"]) for r in group_rows)
        gt_positive_point_count = sum(int(r["gt_positive_count"]) for r in group_rows)
        gt_positive_frame_count = sum(1 for r in group_rows if int(r["gt_positive_count"]) > 0)
        frame_recall_values = [v for v in (_parse_optional_float(r["recall"]) for r in group_rows) if v is not None]
        recall_denom = tp + fn
        rows.append(
            {
                "video_alias": video_alias,
                "split": split,
                "relative_frame_decile": decile,
                "frame_count": len(group_rows),
                "gt_positive_frame_count": gt_positive_frame_count,
                "gt_positive_point_count": gt_positive_point_count,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "true_negative_count": tn,
                "false_negative_count": fn,
                "point_weighted_recall": (tp / recall_denom) if recall_denom > 0 else None,
                "point_weighted_recall_available": recall_denom > 0,
                "frame_equal_weight_recall_mean": float(np.mean(frame_recall_values)) if frame_recall_values else None,
                "frame_equal_weight_recall_available_frame_count": len(frame_recall_values),
            }
        )
    return rows


def paired_half_temporal_rows(
    video_decile_rows: list[dict[str, Any]],
    joint_rows_by_video: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_video: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in video_decile_rows:
        by_video.setdefault((row["video_alias"], row["split"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for (video_alias, split), decile_rows in sorted(by_video.items(), key=lambda item: (item[0][1], item[0][0])):
        first = [r for r in decile_rows if r["relative_frame_decile"] <= 4]
        second = [r for r in decile_rows if r["relative_frame_decile"] >= 5]
        tp1 = sum(r["true_positive_count"] for r in first)
        fn1 = sum(r["false_negative_count"] for r in first)
        tp2 = sum(r["true_positive_count"] for r in second)
        fn2 = sum(r["false_negative_count"] for r in second)
        denom1 = tp1 + fn1
        denom2 = tp2 + fn2
        recall1 = (tp1 / denom1) if denom1 > 0 else None
        recall2 = (tp2 / denom2) if denom2 > 0 else None
        paired_eligible = denom1 > 0 and denom2 > 0
        exclusion_reason = None if paired_eligible else ("no_gt_positive_in_first_half" if denom1 == 0 and denom2 > 0 else ("no_gt_positive_in_second_half" if denom2 == 0 and denom1 > 0 else "no_gt_positive_in_either_half"))

        vote_entry: dict[str, Any] = {}
        for vc in (1, 2):
            joint = joint_rows_by_video.get((video_alias, split), [])
            r1 = next((j for j in joint if j["time_half"] == "first_half" and j["vote_count"] == vc), None)
            r2 = next((j for j in joint if j["time_half"] == "second_half" and j["vote_count"] == vc), None)
            v_recall1 = r1["recall"] if r1 else None
            v_recall2 = r2["recall"] if r2 else None
            v_eligible = bool(r1 and r1["recall_available"] and r2 and r2["recall_available"])
            vote_entry[f"vote{vc}_recall_first_half"] = v_recall1
            vote_entry[f"vote{vc}_recall_second_half"] = v_recall2
            vote_entry[f"vote{vc}_paired_eligible"] = v_eligible
            vote_entry[f"vote{vc}_recall_diff"] = (v_recall2 - v_recall1) if v_eligible else None

        rows.append(
            {
                "video_alias": video_alias,
                "split": split,
                "gt_positive_point_count_first_half": denom1,
                "gt_positive_point_count_second_half": denom2,
                "recall_first_half": recall1,
                "recall_second_half": recall2,
                "paired_eligible": paired_eligible,
                "exclusion_reason": exclusion_reason,
                "recall_diff_second_minus_first": (recall2 - recall1) if paired_eligible else None,
                **vote_entry,
            }
        )
    return rows


def summarize_paired_temporal(paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in paired_rows:
        by_split.setdefault(row["split"], []).append(row)
    summary: dict[str, Any] = {}
    for split, rows in by_split.items():
        eligible = [r for r in rows if r["paired_eligible"]]
        diffs = [r["recall_diff_second_minus_first"] for r in eligible]
        summary[split] = {
            "num_videos_total": len(rows),
            "num_videos_paired_eligible": len(eligible),
            "num_videos_excluded": len(rows) - len(eligible),
            "num_videos_recall_decreased": sum(1 for d in diffs if d < 0),
            "num_videos_recall_increased": sum(1 for d in diffs if d > 0),
            "num_videos_recall_unchanged": sum(1 for d in diffs if d == 0),
            "mean_recall_diff": float(np.mean(diffs)) if diffs else None,
            "median_recall_diff": float(np.median(diffs)) if diffs else None,
        }
    return summary


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-14補足 Step S1-S5: denominator-carrying XY FPR/recall, train-GT-only XY prior "
        "hot/cold comparison, within-video paired temporal recall, and vote-count joint stratification. "
        "Reuses saved W-A predictions (pred_label, vote_count) and the Step H2.5 256x256 normalization. "
        "No re-training, no GPU re-inference, no torch/CUDA import."
    )
    parser.add_argument("--evaluation_dir", required=True, help="evaluate_stage5.py per-checkpoint output_dir (h5_metrics.csv + predictions/)")
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--train_list", required=True, help="W-A train file list (162 H5 paths) for the Step S2 GT-only XY prior")
    parser.add_argument("--frame_metrics_csv", required=True, help="Step H3 frame_metrics.csv, reused for Step S3")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--pixel_width", type=int, default=256)
    parser.add_argument("--pixel_height", type=int, default=256)
    parser.add_argument("--grid_resolutions", default="16,8")
    parser.add_argument("--window_size_frames", type=int, default=16)
    parser.add_argument("--window_stride_frames", type=int, default=8)
    parser.add_argument("--no_include_tail_window", dest="include_tail_window", action="store_false")
    parser.set_defaults(include_tail_window=True)
    parser.add_argument("--xy_bin_denominators_csv", default=None)
    parser.add_argument("--train_xy_prior_csv", default=None)
    parser.add_argument("--xy_hot_cold_video_metrics_csv", default=None)
    parser.add_argument("--video_decile_counts_csv", default=None)
    parser.add_argument("--paired_temporal_metrics_csv", default=None)
    parser.add_argument("--video_time_exposure_metrics_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args()


def _write_csv(path_str: str | None, rows: list[dict[str, Any]]) -> None:
    if not path_str or not rows:
        return
    output = Path(path_str)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    grid_resolutions = [int(item) for item in args.grid_resolutions.split(",") if item.strip()]
    require(bool(grid_resolutions), "--grid_resolutions must list at least one resolution")

    # ---- evaluated videos: fixed train_sanity(3) + validation(18) ----
    video_results: list[dict[str, Any]] = []
    for split in ("train_sanity", "validation"):
        rows = load_h5_metrics_rows(evaluation_dir / "h5_metrics.csv", checkpoint=args.checkpoint, split=split)
        for index, row in enumerate(rows):
            h5_path = Path(row["h5_path"])
            npz_path = evaluation_dir / "predictions" / split / f"{safe_name(row['video_name'])}.npz"
            video_alias = f"{split}_{index:03d}"
            result = process_video_supplement(
                video_alias=video_alias,
                split=split,
                h5_path=h5_path,
                npz_path=npz_path,
                h5_metrics_row=row,
                ignore_index=args.ignore_index,
                pixel_width=args.pixel_width,
                pixel_height=args.pixel_height,
                grid_resolutions=grid_resolutions,
                window_size_frames=args.window_size_frames,
                window_stride_frames=args.window_stride_frames,
                include_tail_window=args.include_tail_window,
            )
            video_results.append(result)

    # ---- Step S1 output ----
    all_bin_rows: list[dict[str, Any]] = []
    for result in video_results:
        for grid in grid_resolutions:
            all_bin_rows.extend(result["bin_rows_by_grid"][grid])

    # ---- Step S2: train-only XY prior ----
    train_h5_paths = read_path_list(args.train_list)
    require(bool(train_h5_paths), f"No H5 paths resolved from --train_list {args.train_list}")
    prior = build_train_prior(
        train_h5_paths,
        ignore_index=args.ignore_index,
        pixel_width=args.pixel_width,
        pixel_height=args.pixel_height,
        grid_resolutions=grid_resolutions,
    )

    train_xy_prior_rows: list[dict[str, Any]] = []
    hot_cold_video_rows: list[dict[str, Any]] = []
    definitions = ("raw_count", "rate")

    for grid in grid_resolutions:
        total_gt, total_valid = prior["totals"][grid]
        classification_by_definition: dict[str, dict[str, Any]] = {}
        for definition in definitions:
            values = compute_prior_value_grid(total_gt, total_valid, definition=definition)
            classification = classify_hot_cold_bins(values, total_valid > 0)
            classification_by_definition[definition] = classification
            for r in range(grid):
                for c in range(grid):
                    hot = bool(classification["hot_mask"][r, c]) if classification["hot_mask"] is not None else None
                    cold = bool(classification["cold_mask"][r, c]) if classification["cold_mask"] is not None else None
                    train_xy_prior_rows.append(
                        {
                            "grid_resolution": grid,
                            "bin_row": r,
                            "bin_col": c,
                            "definition": definition,
                            "raw_gt_positive_count": int(total_gt[r, c]),
                            "valid_point_count": int(total_valid[r, c]),
                            "prior_value": None if np.isnan(values[r, c]) else float(values[r, c]),
                            "quantile_status": classification["status"],
                            "quantile_method": classification["quantile_method"],
                            "q75": classification["q75"],
                            "q25": classification["q25"],
                            "is_hot_shared_classification": hot,
                            "is_cold_shared_classification": cold,
                        }
                    )

        for result in video_results:
            video_alias, split, h5_path = result["video_alias"], result["split"], result["h5_path"]
            gt_pos_excl, valid_excl, self_excluded = prior_excluding_video(prior, h5_path=h5_path, grid_resolution=grid)
            for definition in definitions:
                values_excl = compute_prior_value_grid(gt_pos_excl, valid_excl, definition=definition)
                if self_excluded:
                    classification = classify_hot_cold_bins(values_excl, valid_excl > 0)
                else:
                    classification = classification_by_definition[definition]
                if classification["hot_mask"] is None:
                    hot_cold_video_rows.append(
                        {
                            "video_alias": video_alias,
                            "split": split,
                            "grid_resolution": grid,
                            "definition": definition,
                            "self_excluded_from_prior": self_excluded,
                            "quantile_status": classification["status"],
                            "hot_bin_no_gt_count": None,
                            "cold_bin_no_gt_count": None,
                            "hot_fp": None,
                            "hot_tn": None,
                            "cold_fp": None,
                            "cold_tn": None,
                            "hot_fpr": None,
                            "hot_fpr_available": False,
                            "cold_fpr": None,
                            "cold_fpr_available": False,
                            "fpr_diff_hot_minus_cold": None,
                            "fpr_ratio_hot_over_cold": None,
                            "ratio_undefined_reason": "quantile_" + classification["status"],
                        }
                    )
                    continue
                comparison = hot_cold_fpr_for_video(
                    result["bin_rows_by_grid"][grid],
                    hot_mask=classification["hot_mask"],
                    cold_mask=classification["cold_mask"],
                    grid_resolution=grid,
                )
                hot_cold_video_rows.append(
                    {
                        "video_alias": video_alias,
                        "split": split,
                        "grid_resolution": grid,
                        "definition": definition,
                        "self_excluded_from_prior": self_excluded,
                        "quantile_status": classification["status"],
                        **comparison,
                    }
                )

    hot_cold_aggregate: dict[str, Any] = {}
    for grid in grid_resolutions:
        for definition in definitions:
            for split in ("train_sanity", "validation"):
                scoped = [
                    r
                    for r in hot_cold_video_rows
                    if r["grid_resolution"] == grid and r["definition"] == definition and r["split"] == split and r["hot_fpr_available"] is True
                ]
                key = f"grid{grid}_{definition}_{split}"
                hot_cold_aggregate[key] = aggregate_hot_cold_across_videos(
                    [r for r in hot_cold_video_rows if r["grid_resolution"] == grid and r["definition"] == definition and r["split"] == split]
                )

    # ---- Step S3: video x decile + paired half comparison ----
    frame_metrics_rows = read_frame_metrics_csv(Path(args.frame_metrics_csv))
    video_decile_rows = aggregate_video_decile_counts(frame_metrics_rows)

    joint_rows_by_video: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_joint_rows: list[dict[str, Any]] = []
    for result in video_results:
        key = (result["video_alias"], result["split"])
        joint_rows_by_video[key] = result["joint_rows"]
        all_joint_rows.extend(result["joint_rows"])

    paired_rows = paired_half_temporal_rows(video_decile_rows, joint_rows_by_video)
    paired_summary = summarize_paired_temporal(paired_rows)

    # ---- write outputs ----
    _write_csv(args.xy_bin_denominators_csv, all_bin_rows)
    _write_csv(args.train_xy_prior_csv, train_xy_prior_rows)
    _write_csv(args.xy_hot_cold_video_metrics_csv, hot_cold_video_rows)
    _write_csv(args.video_decile_counts_csv, video_decile_rows)
    _write_csv(args.paired_temporal_metrics_csv, paired_rows)
    _write_csv(args.video_time_exposure_metrics_csv, all_joint_rows)

    summary = {
        "status": "passed",
        "stage": "S5-14_supplement",
        "checkpoint": args.checkpoint,
        "num_evaluated_videos": len(video_results),
        "num_train_prior_videos": prior["num_videos"],
        "grid_resolutions": grid_resolutions,
        "prior_definitions": list(definitions),
        "time_segmentation": {"first_half_deciles": [0, 1, 2, 3, 4], "second_half_deciles": [5, 6, 7, 8, 9]},
        "vote_counts_compared": [1, 2],
        "pixel_width": args.pixel_width,
        "pixel_height": args.pixel_height,
        "num_xy_bin_rows": len(all_bin_rows),
        "num_train_xy_prior_rows": len(train_xy_prior_rows),
        "num_hot_cold_video_rows": len(hot_cold_video_rows),
        "num_video_decile_rows": len(video_decile_rows),
        "num_paired_temporal_rows": len(paired_rows),
        "num_video_time_exposure_rows": len(all_joint_rows),
        "vote_count_parity": {
            "max_mismatch_any_video": max((r["vote_count_parity_mismatch"] for r in video_results), default=0),
        },
        "hot_cold_aggregate_by_grid_definition_split": hot_cold_aggregate,
        "paired_temporal_summary_by_split": paired_summary,
        "joint_disagreement_status": (
            "not_reconstructed: Step S4 requires per-window disagreement at the (video, time_half, "
            "vote_count) joint granularity; the saved Step H4 artifacts only carry disagreement_rate "
            "marginalized over (video, xy_bin) or (video, decile)/(video, vote_count_bucket) separately. "
            "Joining those marginals into a fabricated joint table is explicitly disallowed by the "
            "implementation handoff, and GPU re-inference to reconstruct it directly is out of scope for "
            "this supplement. video_time_exposure_metrics.csv above reports only FPR/recall (independently "
            "recomputable from saved pred_label/vote_count), not disagreement."
        ),
    }
    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-14 supplement (Step S1-S5) passed.")
    print(f"evaluated videos: {summary['num_evaluated_videos']}, train prior videos: {summary['num_train_prior_videos']}")
    print(f"grid resolutions: {grid_resolutions}")
    print(f"xy bin rows: {summary['num_xy_bin_rows']}, hot/cold video rows: {summary['num_hot_cold_video_rows']}")
    print(f"video decile rows: {summary['num_video_decile_rows']}, paired temporal rows: {summary['num_paired_temporal_rows']}")
    print(f"video time exposure rows: {summary['num_video_time_exposure_rows']}")
    if args.summary_json:
        print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
