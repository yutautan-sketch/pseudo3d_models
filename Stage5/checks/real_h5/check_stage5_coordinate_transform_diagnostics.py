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

from checks.real_h5.check_stage5_class_weight_threshold_free import load_h5_metrics_rows, load_prediction, safe_name
from checks.real_h5.check_stage5_frame_spatial_overlap_diagnostics import require, window_occurrence_counts
from checks.real_h5.check_stage5_s5_14_supplement import (
    build_train_prior,
    classify_hot_cold_bins,
    compute_prior_value_grid,
    hot_cold_fpr_for_video,
    prior_excluding_video,
    xy_bin_denominator_rows,
)
from checks.real_h5.check_stage5_xy_coordinate_provenance import normalize_pixel_xy_with_dimensions
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list

# ----------------------------------------------------------------------------
# S5-14補足2: coordinate-transform (prediction equivariance) diagnosis.
# Applies a fixed rigid transform to the model's own input XYZ (post
# normalize_xyz(), pre-window-extraction -- see check_stage5_s5_14_
# supplement2_report_to_policy_chat.md section 4 for the exact injection
# point audit) and compares the resulting per-window canonical aggregate
# prediction against the identity condition. No retraining; the ONE
# additional GPU forward pass per (video, condition) this diagnosis allows
# (mirrors S5-14 core Step H4's precedent) lives only in
# `run_condition_forward` / `main()` (function-local torch import) -- every
# other function in this module is torch-free and unit-testable without
# CUDA, matching check_stage5_per_window_context_diagnostics.py's convention.
#
# 2026-09-15 policy-chat correction (section 12 of the report-to-policy-chat
# doc): an earlier version of this module classified hot/cold bins for
# EVERY evaluated video (including the 3 train-sanity videos) from the
# unmodified, shared 162-video train prior. That does not reproduce S5-14
# supplement's own train-sanity self-exclusion (leave-one-video-out) and was
# not approved as a simplification. This version re-derives each video's
# hot/cold classification via `prior_excluding_video()` (imported from the
# supplement module, unmodified) so a train-sanity video's own GT
# contribution is subtracted before classifying; a validation video (never
# part of the train list) is unaffected. Condition/hot-cold summaries are
# also now computed per split (train_sanity vs. validation), with the
# previous 21-video-mixed aggregate kept only as an explicitly-labeled
# reference value.
# ----------------------------------------------------------------------------

TRANSLATION_MAGNITUDE = 0.1
ROTATION_DEGREES = 15.0

TRANSFORM_CONDITIONS: tuple[str, ...] = (
    "identity",
    "repeat_identity",
    "translate_x_plus",
    "translate_x_minus",
    "translate_y_plus",
    "translate_y_minus",
    "rotate_z_plus",
    "rotate_z_minus",
)

# identity/repeat_identity must reproduce the saved W-A prediction with zero
# tolerance (canonical parity gate); the other 6 conditions are expected to
# differ from identity -- that difference IS the diagnostic signal.
PARITY_GATED_CONDITIONS: tuple[str, ...] = ("identity", "repeat_identity")

# Must match S5-14 supplement Step S2 exactly (grid resolutions and prior
# denominator definitions) so the hot/cold classification this diagnosis
# reuses is the identical algorithm, not a re-invented variant.
GRID_RESOLUTIONS: tuple[int, ...] = (16, 8)
PRIOR_DEFINITIONS: tuple[str, ...] = ("raw_count", "rate")

EVALUATED_SPLITS: tuple[str, ...] = ("train_sanity", "validation")


# ----------------------------------------------------------------------------
# Step T1 primitives: pure geometry, no torch, no model.
# ----------------------------------------------------------------------------


def rotation_matrix_z(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_transform(points: np.ndarray, condition: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Applies one of the 8 fixed conditions to already-`normalize_xyz()`-ed
    points. `identity`/`repeat_identity` return an unmodified copy (the
    latter exists only to control for run-to-run forward-pass variability,
    not to change the input). Never re-centers/re-scales/clips afterward.
    """
    require(condition in TRANSFORM_CONDITIONS, f"unknown transform condition: {condition!r}")
    points = np.asarray(points, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3, f"expected [N, 3] points, got shape {points.shape}")

    if condition in ("identity", "repeat_identity"):
        transformed = points.copy()
        meta: dict[str, Any] = {"kind": "identity", "translation": None, "rotation_degrees": None}
    elif condition.startswith("translate_"):
        axis = 0 if "_x_" in condition else 1
        sign = 1.0 if condition.endswith("_plus") else -1.0
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = sign * TRANSLATION_MAGNITUDE
        transformed = points + offset[None, :]
        meta = {"kind": "translation", "translation": offset.tolist(), "rotation_degrees": None}
    elif condition.startswith("rotate_z_"):
        sign = 1.0 if condition.endswith("_plus") else -1.0
        degrees = sign * ROTATION_DEGREES
        matrix = rotation_matrix_z(degrees)
        transformed = points @ matrix.T
        meta = {"kind": "rotation", "translation": None, "rotation_degrees": degrees, "rotation_matrix": matrix.tolist()}
    else:  # pragma: no cover -- guarded by the require() above
        raise AssertionError(f"unhandled transform condition: {condition!r}")

    unit_norm_exceeded_rate = (
        float(np.mean(np.linalg.norm(transformed, axis=1) > 1.0)) if transformed.shape[0] else 0.0
    )
    meta["unit_norm_exceeded_rate"] = unit_norm_exceeded_rate
    return transformed.astype(np.float32), meta


# ----------------------------------------------------------------------------
# Step T3/T4 primitives: per-point comparison against the identity baseline,
# and hot/cold-bin FPR under each condition (reusing S5-14 supplement's
# already-built, already-tested XY-bin denominator machinery unmodified).
# ----------------------------------------------------------------------------


def per_point_condition_metrics(
    *,
    baseline_prob_positive: np.ndarray,
    baseline_pred_label: np.ndarray,
    condition_prob_positive: np.ndarray,
    condition_pred_label: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    ignore_index: int,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool) & (labels != int(ignore_index))
    target_positive = valid & (labels == 1)
    target_background = valid & (labels == 0)
    ignore_mask = ~valid

    prob_diff = condition_prob_positive.astype(np.float64) - baseline_prob_positive.astype(np.float64)
    flipped = condition_pred_label != baseline_pred_label
    flip_to_positive_count = int(np.sum(flipped & (condition_pred_label == 1)))
    flip_to_background_count = int(np.sum(flipped & (condition_pred_label == 0)))

    def confusion_and_rates(pred: np.ndarray) -> dict[str, float | int | None]:
        tp = int(np.sum(target_positive & (pred == 1)))
        fp = int(np.sum(target_background & (pred == 1)))
        tn = int(np.sum(target_background & (pred == 0)))
        fn = int(np.sum(target_positive & (pred == 0)))
        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": (tp / (tp + fp)) if (tp + fp) > 0 else None,
            "recall": (tp / (tp + fn)) if (tp + fn) > 0 else None,
            "fpr": (fp / (fp + tn)) if (fp + tn) > 0 else None,
            "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else None,
            "iou": (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else None,
        }

    condition_stats = confusion_and_rates(condition_pred_label)
    baseline_stats = confusion_and_rates(baseline_pred_label)

    def diff(key: str) -> float | None:
        a, b = condition_stats[key], baseline_stats[key]
        return (a - b) if (a is not None and b is not None) else None

    condition_ignore_rate = float(np.mean(condition_pred_label[ignore_mask] == 1)) if np.any(ignore_mask) else None
    baseline_ignore_rate = float(np.mean(baseline_pred_label[ignore_mask] == 1)) if np.any(ignore_mask) else None

    return {
        "prob_diff_signed_mean": float(np.mean(prob_diff)) if prob_diff.size else None,
        "prob_diff_abs_mean": float(np.mean(np.abs(prob_diff))) if prob_diff.size else None,
        "prob_diff_abs_p95": float(np.percentile(np.abs(prob_diff), 95)) if prob_diff.size else None,
        "prob_diff_abs_max": float(np.max(np.abs(prob_diff))) if prob_diff.size else None,
        "flip_rate": float(np.mean(flipped)) if flipped.size else None,
        "flip_to_positive_count": flip_to_positive_count,
        "flip_to_background_count": flip_to_background_count,
        "true_positive_count": condition_stats["tp"],
        "false_positive_count": condition_stats["fp"],
        "true_negative_count": condition_stats["tn"],
        "false_negative_count": condition_stats["fn"],
        "precision": condition_stats["precision"],
        "recall": condition_stats["recall"],
        "false_positive_rate": condition_stats["fpr"],
        "f1": condition_stats["f1"],
        "iou_femur": condition_stats["iou"],
        "precision_diff": diff("precision"),
        "recall_diff": diff("recall"),
        "false_positive_rate_diff": diff("fpr"),
        "f1_diff": diff("f1"),
        "iou_femur_diff": diff("iou"),
        "true_positive_count_is_zero": condition_stats["tp"] == 0,
        "ignore_predicted_positive_rate": condition_ignore_rate,
        "baseline_ignore_predicted_positive_rate": baseline_ignore_rate,
    }


def build_video_hot_cold_masks(
    prior: dict[str, Any],
    *,
    h5_path: Path,
    grid_resolutions: tuple[int, ...] = GRID_RESOLUTIONS,
    definitions: tuple[str, ...] = PRIOR_DEFINITIONS,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Per-video hot/cold classification, reusing S5-14 supplement Step S2's
    exact algorithm (`prior_excluding_video()` + `compute_prior_value_grid()`
    + `classify_hot_cold_bins()`) unmodified. For a video that is itself
    part of the train list (a train-sanity video), `prior_excluding_video()`
    subtracts that video's own GT contribution from the train-162 prior
    before classifying (leave-one-video-out self-exclusion) -- matching
    S5-14 supplement's own handling. A validation video (never part of the
    train list) gets the full, unmodified 162-video classification back
    unchanged, exactly as before.
    """
    masks: dict[tuple[int, str], dict[str, Any]] = {}
    for grid in grid_resolutions:
        gt_pos_excl, valid_excl, self_excluded = prior_excluding_video(prior, h5_path=h5_path, grid_resolution=grid)
        for definition in definitions:
            values = compute_prior_value_grid(gt_pos_excl, valid_excl, definition=definition)
            classification = classify_hot_cold_bins(values, valid_excl > 0)
            masks[(grid, definition)] = {
                "hot_mask": classification["hot_mask"],
                "cold_mask": classification["cold_mask"],
                "status": classification["status"],
                "self_excluded": self_excluded,
            }
    return masks


def condition_hot_cold_rows(
    *,
    video_alias: str,
    split: str,
    condition: str,
    xy_normalized: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    baseline_pred_label: np.ndarray,
    ignore_index: int,
    video_hot_cold_masks: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (grid, definition), mask_info in video_hot_cold_masks.items():
        hot_mask, cold_mask = mask_info["hot_mask"], mask_info["cold_mask"]
        base_row = {
            "video_alias": video_alias,
            "split": split,
            "condition": condition,
            "grid_resolution": grid,
            "definition": definition,
            "self_excluded_from_prior": mask_info["self_excluded"],
            "quantile_status": mask_info["status"],
        }
        if hot_mask is None or cold_mask is None:
            rows.append(
                {
                    **base_row,
                    "hot_fpr": None,
                    "hot_fpr_available": False,
                    "cold_fpr": None,
                    "cold_fpr_available": False,
                    "hot_fpr_diff_vs_identity": None,
                    "cold_fpr_diff_vs_identity": None,
                }
            )
            continue

        bin_rows = xy_bin_denominator_rows(
            video_alias=video_alias,
            split=split,
            xy_normalized=xy_normalized,
            labels=labels,
            valid_mask=valid_mask,
            pred_label=pred_label,
            ignore_index=ignore_index,
            grid_resolution=grid,
        )
        comparison = hot_cold_fpr_for_video(bin_rows, hot_mask=hot_mask, cold_mask=cold_mask, grid_resolution=grid)

        baseline_bin_rows = xy_bin_denominator_rows(
            video_alias=video_alias,
            split=split,
            xy_normalized=xy_normalized,
            labels=labels,
            valid_mask=valid_mask,
            pred_label=baseline_pred_label,
            ignore_index=ignore_index,
            grid_resolution=grid,
        )
        baseline_comparison = hot_cold_fpr_for_video(
            baseline_bin_rows, hot_mask=hot_mask, cold_mask=cold_mask, grid_resolution=grid
        )

        def diff(key: str, a=comparison, b=baseline_comparison) -> float | None:
            va, vb = a[key], b[key]
            return (va - vb) if (va is not None and vb is not None) else None

        rows.append(
            {
                **base_row,
                "hot_fpr": comparison["hot_fpr"],
                "hot_fpr_available": comparison["hot_fpr_available"],
                "cold_fpr": comparison["cold_fpr"],
                "cold_fpr_available": comparison["cold_fpr_available"],
                "hot_fpr_diff_vs_identity": diff("hot_fpr"),
                "cold_fpr_diff_vs_identity": diff("cold_fpr"),
            }
        )
    return rows


# ----------------------------------------------------------------------------
# Aggregation. Callers filter `point_rows`/`hot_cold_rows` by split BEFORE
# calling these (e.g. once per split, once unfiltered for the mixed
# reference value) -- these functions themselves are split-agnostic.
# ----------------------------------------------------------------------------


def aggregate_condition_summary(point_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in point_rows:
        by_condition.setdefault(row["condition"], []).append(row)

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "median": float(np.median(values)) if values else None,
            "mean": float(np.mean(values)) if values else None,
        }

    summary: dict[str, Any] = {}
    for condition, rows in by_condition.items():
        flip_rates = [r["flip_rate"] for r in rows if r["flip_rate"] is not None]
        prob_diff_abs_means = [r["prob_diff_abs_mean"] for r in rows if r["prob_diff_abs_mean"] is not None]
        fpr_diffs = [r["false_positive_rate_diff"] for r in rows if r["false_positive_rate_diff"] is not None]
        recall_diffs = [r["recall_diff"] for r in rows if r["recall_diff"] is not None]
        summary[condition] = {
            "num_videos": len(rows),
            "num_videos_tp_zero": sum(1 for r in rows if r["true_positive_count_is_zero"]),
            "flip_rate": stats(flip_rates),
            "prob_diff_abs_mean": stats(prob_diff_abs_means),
            "false_positive_rate_diff": {
                "num_videos_available": len(fpr_diffs),
                "num_videos_increased": sum(1 for d in fpr_diffs if d > 0),
                "num_videos_decreased": sum(1 for d in fpr_diffs if d < 0),
                **stats(fpr_diffs),
            },
            "recall_diff": {
                "num_videos_available": len(recall_diffs),
                "num_videos_increased": sum(1 for d in recall_diffs if d > 0),
                "num_videos_decreased": sum(1 for d in recall_diffs if d < 0),
                **stats(recall_diffs),
            },
        }
    return summary


def aggregate_hot_cold_condition_summary(hot_cold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in hot_cold_rows:
        key = (row["condition"], int(row["grid_resolution"]), row["definition"])
        by_key.setdefault(key, []).append(row)

    summary: dict[str, Any] = {}
    for (condition, grid, definition), rows in by_key.items():
        diffs = [r["hot_fpr_diff_vs_identity"] for r in rows if r["hot_fpr_diff_vs_identity"] is not None]
        summary[f"{condition}__grid{grid}__{definition}"] = {
            "num_videos_total": len(rows),
            "num_videos_available": len(diffs),
            "num_videos_hot_fpr_increased": sum(1 for d in diffs if d > 0),
            "num_videos_hot_fpr_decreased": sum(1 for d in diffs if d < 0),
            "median_hot_fpr_diff_vs_identity": float(np.median(diffs)) if diffs else None,
            "mean_hot_fpr_diff_vs_identity": float(np.mean(diffs)) if diffs else None,
        }
    return summary


# ----------------------------------------------------------------------------
# Orchestration (torch required only in run_condition_forward/main()).
# ----------------------------------------------------------------------------


def run_condition_forward(
    *,
    model: Any,
    data: dict[str, Any],
    features: tuple[str, ...],
    normalize_points: bool,
    device: Any,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    condition: str,
    h5_path: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One per-window forward pass with the transform applied ONCE to the
    whole video's `normalize_xyz()`-ed points, before any window slicing --
    matching evaluate_stage5.predict_h5()'s exact aggregation path
    (per-window softmax cast to float32 immediately, accumulated in
    float64) so identity/repeat_identity reproduce the saved W-A prediction
    bit-for-bit absent genuine GPU non-determinism.
    """
    import torch

    from infer_stage5 import build_features
    from stage5.utils.feature_normalization import normalize_xyz
    from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window

    raw_points = data["points"].astype(np.float32)
    normalized_points = normalize_xyz(raw_points) if normalize_points else raw_points
    transformed_points, transform_meta = apply_transform(normalized_points, condition)
    feature_values = build_features(data, features)
    frame_order = data["frame_order"]
    windows = generate_frame_order_windows(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    require(bool(windows), f"No frame windows generated for {h5_path}")

    num_points = transformed_points.shape[0]
    probability_sum_2class = np.zeros((num_points, 2), dtype=np.float64)
    vote_count = np.zeros(num_points, dtype=np.int32)

    with torch.inference_mode():
        for window in windows:
            indices = point_indices_for_window(frame_order, window)
            if indices.size == 0:
                continue
            batch = {
                "points": torch.from_numpy(transformed_points[indices][None]).float().to(device),
                "features": torch.from_numpy(feature_values[indices][None]).float().to(device),
            }
            logits = model(batch)["logits"][0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
            require(np.all(np.isfinite(probabilities)), f"Non-finite probabilities in {h5_path} condition={condition}")
            probability_sum_2class[indices] += probabilities.astype(np.float64)
            vote_count[indices] += 1

    missing = vote_count == 0
    require(not bool(np.any(missing)), f"{int(np.sum(missing))} points received no prediction in {h5_path} condition={condition}")
    return probability_sum_2class, vote_count, transform_meta


def process_video_coordinate_transform(
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
    video_hot_cold_masks: dict[tuple[int, str], dict[str, Any]],
    diff_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    # Local import: only this function needs torch (via canonical_predicted_class's
    # caller / verify_h4_parity, both torch-free themselves, but this module as a
    # whole stays importable/testable without CUDA except through this function).
    from checks.real_h5.check_stage5_per_window_context_diagnostics import canonical_predicted_class, verify_h4_parity

    data = load_stage5_pointcloud_h5(h5_path)
    labels = np.asarray(data["point_label"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    frame_order = np.asarray(data["frame_order"])
    pixel_xy = np.asarray(data["pixel_xy"])

    saved_prob_femur, saved_pred_label = load_prediction(npz_path)
    with np.load(npz_path) as saved:
        require("vote_count" in saved, f"{npz_path}: missing vote_count")
        saved_vote_count = saved["vote_count"].astype(np.int32)

    structural_vote_count, _num_windows = window_occurrence_counts(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    xy_normalized = normalize_pixel_xy_with_dimensions(pixel_xy, width=pixel_width, height=pixel_height)

    condition_results: dict[str, dict[str, Any]] = {}
    for condition in TRANSFORM_CONDITIONS:
        probability_sum_2class, vote_count, transform_meta = run_condition_forward(
            model=model,
            data=data,
            features=features,
            normalize_points=normalize_points,
            device=device,
            window_size_frames=window_size_frames,
            window_stride_frames=window_stride_frames,
            include_tail_window=include_tail_window,
            condition=condition,
            h5_path=str(h5_path),
        )
        # Window membership is a pure function of frame_order (never of the
        # transformed XYZ) -- vote_count must be identical to the structural
        # recount and to the saved W-A prediction for EVERY condition, not
        # just identity.
        require(
            bool(np.array_equal(vote_count, structural_vote_count)),
            f"{split}/{video_alias}/{condition}: vote_count differs from the structural window-occurrence "
            "recount -- window membership must be invariant to the input XYZ transform",
        )
        require(
            bool(np.array_equal(vote_count, saved_vote_count)),
            f"{split}/{video_alias}/{condition}: vote_count differs from the saved W-A prediction's vote_count",
        )
        aggregated_probability, pred_label = canonical_predicted_class(probability_sum_2class, vote_count)

        if condition in PARITY_GATED_CONDITIONS:
            verify_h4_parity(
                vote_count=vote_count,
                aggregated_probability_positive=aggregated_probability[:, 1],
                pred_label=pred_label,
                saved_prob_femur=saved_prob_femur,
                saved_vote_count=saved_vote_count,
                saved_pred_label=saved_pred_label,
                labels=labels,
                valid_mask=valid_mask,
                training_occurrence_count=structural_vote_count,
                h5_metrics_row=h5_metrics_row,
                ignore_index=ignore_index,
                context=f"{split}/{video_alias}/{condition}",
                diff_artifact_path=(
                    diff_artifact_dir / f"{split}_{video_alias}_{condition}_parity_diff_DO_NOT_SHARE.csv"
                )
                if diff_artifact_dir
                else None,
                frame_order=frame_order,
            )

        condition_results[condition] = {
            "probability_positive": aggregated_probability[:, 1],
            "pred_label": pred_label,
            "transform_meta": transform_meta,
        }

    baseline = condition_results["identity"]
    point_rows: list[dict[str, Any]] = []
    hot_cold_rows: list[dict[str, Any]] = []
    for condition in TRANSFORM_CONDITIONS:
        result = condition_results[condition]
        metrics = per_point_condition_metrics(
            baseline_prob_positive=baseline["probability_positive"],
            baseline_pred_label=baseline["pred_label"],
            condition_prob_positive=result["probability_positive"],
            condition_pred_label=result["pred_label"],
            labels=labels,
            valid_mask=valid_mask,
            ignore_index=ignore_index,
        )
        point_rows.append(
            {
                "video_alias": video_alias,
                "split": split,
                "condition": condition,
                "transform_kind": result["transform_meta"]["kind"],
                "translation": result["transform_meta"]["translation"],
                "rotation_degrees": result["transform_meta"]["rotation_degrees"],
                "unit_norm_exceeded_rate": result["transform_meta"]["unit_norm_exceeded_rate"],
                **metrics,
            }
        )
        hot_cold_rows.extend(
            condition_hot_cold_rows(
                video_alias=video_alias,
                split=split,
                condition=condition,
                xy_normalized=xy_normalized,
                labels=labels,
                valid_mask=valid_mask,
                pred_label=result["pred_label"],
                baseline_pred_label=baseline["pred_label"],
                ignore_index=ignore_index,
                video_hot_cold_masks=video_hot_cold_masks,
            )
        )

    return {"video_alias": video_alias, "split": split, "point_rows": point_rows, "hot_cold_rows": hot_cold_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-14補足2: coordinate-transform (prediction equivariance) diagnosis. "
        "8 fixed conditions (identity/repeat_identity/±0.1 X or Y translation/±15deg Z rotation) "
        "applied to the model's own normalize_xyz() input, pre-window-extraction. No retraining."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation_dir", required=True, help="evaluate_stage5.py per-checkpoint output_dir")
    parser.add_argument("--checkpoint_row_name", default="last", help="checkpoint value in h5_metrics.csv")
    parser.add_argument("--train_list", required=True, help="W-A train file list (162 H5 paths), for the leave-one-video-out hot/cold prior")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--pixel_width", type=int, default=256)
    parser.add_argument("--pixel_height", type=int, default=256)
    parser.add_argument("--window_size_frames", type=int, default=None)
    parser.add_argument("--window_stride_frames", type=int, default=None)
    parser.add_argument("--include_tail_window", type=int, default=None, choices=[0, 1])
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point_metrics_csv", default=None)
    parser.add_argument("--hot_cold_metrics_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument(
        "--diff_artifact_dir",
        default=None,
        help="Directory for per-video identity/repeat_identity parity-mismatch diff CSVs (private, "
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

    train_h5_paths = read_path_list(args.train_list)
    require(bool(train_h5_paths), f"No H5 paths resolved from --train_list {args.train_list}")
    prior = build_train_prior(
        train_h5_paths,
        ignore_index=args.ignore_index,
        pixel_width=args.pixel_width,
        pixel_height=args.pixel_height,
        grid_resolutions=list(GRID_RESOLUTIONS),
    )

    all_point_rows: list[dict[str, Any]] = []
    all_hot_cold_rows: list[dict[str, Any]] = []
    num_videos_by_split: dict[str, int] = {}

    for split in EVALUATED_SPLITS:
        rows = load_h5_metrics_rows(evaluation_dir / "h5_metrics.csv", checkpoint=args.checkpoint_row_name, split=split)
        num_videos_by_split[split] = len(rows)
        for index, row in enumerate(rows):
            h5_path = Path(row["h5_path"])
            npz_path = evaluation_dir / "predictions" / split / f"{safe_name(row['video_name'])}.npz"
            video_alias = f"{split}_{index:03d}"
            video_hot_cold_masks = build_video_hot_cold_masks(prior, h5_path=h5_path)
            result = process_video_coordinate_transform(
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
                video_hot_cold_masks=video_hot_cold_masks,
                diff_artifact_dir=Path(args.diff_artifact_dir) if args.diff_artifact_dir else None,
            )
            all_point_rows.extend(result["point_rows"])
            all_hot_cold_rows.extend(result["hot_cold_rows"])

    def write_csv(path_str: str | None, rows: list[dict[str, Any]]) -> None:
        if not path_str or not rows:
            return
        output = Path(path_str)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.point_metrics_csv, all_point_rows)
    write_csv(args.hot_cold_metrics_csv, all_hot_cold_rows)

    # Split-level summaries are the primary judgment layer (policy-chat
    # requirement); the 21-video-mixed aggregate is kept only as an
    # explicitly-labeled reference value, never the basis for a decision.
    condition_summary_by_split = {
        split: aggregate_condition_summary([r for r in all_point_rows if r["split"] == split]) for split in EVALUATED_SPLITS
    }
    hot_cold_summary_by_split = {
        split: aggregate_hot_cold_condition_summary([r for r in all_hot_cold_rows if r["split"] == split])
        for split in EVALUATED_SPLITS
    }
    condition_summary_all_videos_reference_mixes_splits = aggregate_condition_summary(all_point_rows)
    hot_cold_summary_all_videos_reference_mixes_splits = aggregate_hot_cold_condition_summary(all_hot_cold_rows)

    num_videos = sum(num_videos_by_split.values())

    summary = {
        "status": "passed",
        "stage": "S5-14_supplement2",
        "checkpoint": args.checkpoint_row_name,
        "num_videos": num_videos,
        "num_videos_by_split": num_videos_by_split,
        "num_train_prior_videos": prior["num_videos"],
        "num_conditions": len(TRANSFORM_CONDITIONS),
        "translation_magnitude": TRANSLATION_MAGNITUDE,
        "rotation_degrees": ROTATION_DEGREES,
        "grid_resolutions": list(GRID_RESOLUTIONS),
        "prior_definitions": list(PRIOR_DEFINITIONS),
        "num_point_metric_rows": len(all_point_rows),
        "num_hot_cold_rows": len(all_hot_cold_rows),
        "condition_summary_by_split": condition_summary_by_split,
        "hot_cold_condition_summary_by_split": hot_cold_summary_by_split,
        "condition_summary_all_videos_reference_mixes_splits": condition_summary_all_videos_reference_mixes_splits,
        "hot_cold_condition_summary_all_videos_reference_mixes_splits": hot_cold_summary_all_videos_reference_mixes_splits,
    }
    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-14 supplement2 coordinate-transform diagnosis passed.")
    print(f"videos: {num_videos} ({num_videos_by_split}), conditions: {len(TRANSFORM_CONDITIONS)}")
    print(f"point metric rows: {len(all_point_rows)}, hot/cold rows: {len(all_hot_cold_rows)}")
    if args.summary_json:
        print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
