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
    binary_counts,
    check_threshold_0p5_parity,
    load_h5_metrics_rows,
    load_prediction,
    safe_name,
)
from checks.real_h5.check_stage5_frame_spatial_overlap_diagnostics import (
    find_runs,
    relative_frame_decile,
    require,
)
from checks.real_h5.check_stage5_xy_coordinate_provenance import normalize_pixel_xy_with_dimensions
from stage5.utils.h5_io import load_stage5_pointcloud_h5

# ----------------------------------------------------------------------------
# S5-14 Step H3/H3.1: frame-level TP/FP/FN/probability/XY-centroid diagnostics
# and normalized-XY density/temporal-recurrence diagnostics, reusing W-A's
# already-saved evaluate_stage5.py prediction `.npz` artifacts (no
# re-inference, no torch/CUDA) plus the Step H2.5-confirmed per-video local
# crop dimensions (256x256, resolved for all 180 teacher v7 videos with zero
# exclusions) for cross-video XY comparability.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Step H3: per-frame diagnostics
# ----------------------------------------------------------------------------


def frame_confusion_counts(
    frame_order: np.ndarray,
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    *,
    ignore_index: int,
) -> dict[int, dict[str, int]]:
    frame_order = np.asarray(frame_order)
    point_label = np.asarray(point_label)
    valid = np.asarray(valid_mask, dtype=bool) & (point_label != int(ignore_index))
    target_positive = point_label == 1
    predicted_positive = np.asarray(pred_label) == 1

    unique_frames, inverse = np.unique(frame_order, return_inverse=True)
    n = unique_frames.size
    tp = np.zeros(n, dtype=np.int64)
    fp = np.zeros(n, dtype=np.int64)
    tn = np.zeros(n, dtype=np.int64)
    fn = np.zeros(n, dtype=np.int64)
    ignore_count = np.zeros(n, dtype=np.int64)
    np.add.at(tp, inverse, (valid & target_positive & predicted_positive).astype(np.int64))
    np.add.at(fp, inverse, (valid & ~target_positive & predicted_positive).astype(np.int64))
    np.add.at(tn, inverse, (valid & ~target_positive & ~predicted_positive).astype(np.int64))
    np.add.at(fn, inverse, (valid & target_positive & ~predicted_positive).astype(np.int64))
    np.add.at(ignore_count, inverse, (~valid).astype(np.int64))

    return {
        int(f): {"tp": int(a), "fp": int(b), "tn": int(c), "fn": int(d), "ignore_count": int(e)}
        for f, a, b, c, d, e in zip(unique_frames.tolist(), tp, fp, tn, fn, ignore_count)
    }


def _per_frame_grouped_stats(frame_order: np.ndarray, values: np.ndarray, group_mask: np.ndarray) -> dict[int, dict[str, Any]]:
    """Per distinct frame: count/mean/std/min/max of `values` restricted to `group_mask`."""
    unique_frames, inverse = np.unique(frame_order, return_inverse=True)
    n = unique_frames.size
    count = np.zeros(n, dtype=np.int64)
    total = np.zeros(n, dtype=np.float64)
    total_sq = np.zeros(n, dtype=np.float64)

    sel_inverse = inverse[group_mask]
    sel_values = values[group_mask].astype(np.float64)
    np.add.at(count, sel_inverse, 1)
    np.add.at(total, sel_inverse, sel_values)
    np.add.at(total_sq, sel_inverse, sel_values**2)

    result: dict[int, dict[str, Any]] = {}
    for i, frame in enumerate(unique_frames.tolist()):
        c = int(count[i])
        if c == 0:
            result[frame] = {"count": 0, "mean": None, "std": None}
            continue
        mean = total[i] / c
        variance = max(total_sq[i] / c - mean**2, 0.0)
        result[frame] = {"count": c, "mean": float(mean), "std": float(np.sqrt(variance))}
    return result


def frame_probability_stats(
    frame_order: np.ndarray,
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    prob_femur: np.ndarray,
    *,
    ignore_index: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    point_label = np.asarray(point_label)
    valid = np.asarray(valid_mask, dtype=bool) & (point_label != int(ignore_index))
    gt_positive_mask = valid & (point_label == 1)
    gt_background_mask = valid & (point_label == 0)
    ignore_mask = ~valid

    positive_stats = _per_frame_grouped_stats(frame_order, prob_femur, gt_positive_mask)
    background_stats = _per_frame_grouped_stats(frame_order, prob_femur, gt_background_mask)
    ignore_stats = _per_frame_grouped_stats(frame_order, prob_femur, ignore_mask)

    frames = sorted(positive_stats)
    return {
        frame: {"positive": positive_stats[frame], "background": background_stats[frame], "ignore": ignore_stats[frame]}
        for frame in frames
    }


def xy_centroid_and_variance(xy: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    selected = xy[mask]
    if selected.shape[0] == 0:
        return None, None
    return selected.mean(axis=0), selected.var(axis=0)


def centroid_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def nearest_positive_frame_distance(frame: int, positive_frames_sorted: np.ndarray) -> int | None:
    if positive_frames_sorted.size == 0:
        return None
    idx = int(np.searchsorted(positive_frames_sorted, frame))
    candidates = []
    if idx < positive_frames_sorted.size:
        candidates.append(abs(int(positive_frames_sorted[idx]) - frame))
    if idx > 0:
        candidates.append(abs(int(positive_frames_sorted[idx - 1]) - frame))
    return min(candidates)


def classify_relative_to_runs(frame: int, runs: list[tuple[int, int]]) -> str:
    if not runs:
        return "no_gt_run"
    for start, end in runs:
        if start <= frame <= end:
            return "inside"
    nearest = min(runs, key=lambda run: min(abs(frame - run[0]), abs(frame - run[1])))
    return "before" if frame < nearest[0] else "after"


def build_frame_metrics_rows(
    *,
    video_alias: str,
    split: str,
    frame_order: np.ndarray,
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    pixel_xy: np.ndarray,
    pred_label: np.ndarray,
    prob_femur: np.ndarray,
    ignore_index: int,
) -> list[dict[str, Any]]:
    point_label = np.asarray(point_label)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid = valid_mask & (point_label != int(ignore_index))
    gt_positive_mask_all = valid & (point_label == 1)
    predicted_positive_mask_all = np.asarray(pred_label) == 1

    confusion = frame_confusion_counts(frame_order, point_label, valid_mask, pred_label, ignore_index=ignore_index)
    probability = frame_probability_stats(frame_order, point_label, valid_mask, prob_femur, ignore_index=ignore_index)
    unique_frames = np.array(sorted(confusion), dtype=np.int64)
    deciles = relative_frame_decile(frame_order, unique_frames)

    gt_positive_frames = np.array(
        sorted(frame for frame, counts in confusion.items() if _frame_has_gt_positive(point_label, valid, frame_order, frame)),
        dtype=np.int64,
    )
    gt_runs = find_runs(gt_positive_frames) if gt_positive_frames.size else []

    rows: list[dict[str, Any]] = []
    for i, frame in enumerate(unique_frames.tolist()):
        frame_point_mask = frame_order == frame
        counts = confusion[frame]
        prob = probability[frame]

        gt_centroid, gt_variance = xy_centroid_and_variance(pixel_xy, frame_point_mask & gt_positive_mask_all)
        pred_centroid, pred_variance = xy_centroid_and_variance(pixel_xy, frame_point_mask & predicted_positive_mask_all)
        distance = centroid_distance(gt_centroid, pred_centroid)

        has_gt_positive = gt_centroid is not None
        is_no_gt_frame = not has_gt_positive
        no_gt_fp_probability_mean = (
            prob["background"]["mean"] if is_no_gt_frame and prob["background"]["count"] > 0 else None
        )

        tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        fnr = fn / (fn + tp) if (fn + tp) > 0 else None

        rows.append(
            {
                "video_alias": video_alias,
                "split": split,
                "frame": frame,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "true_negative_count": tn,
                "false_negative_count": fn,
                "ignore_count": counts["ignore_count"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou_femur": iou,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "gt_positive_count": prob["positive"]["count"],
                "predicted_positive_count": int(np.sum(frame_point_mask & predicted_positive_mask_all)),
                "valid_count": int(np.sum(frame_point_mask & valid)),
                "prob_positive_mean": prob["positive"]["mean"],
                "prob_background_mean": prob["background"]["mean"],
                "prob_ignore_mean": prob["ignore"]["mean"],
                "gt_centroid_x": float(gt_centroid[0]) if gt_centroid is not None else None,
                "gt_centroid_y": float(gt_centroid[1]) if gt_centroid is not None else None,
                "predicted_centroid_x": float(pred_centroid[0]) if pred_centroid is not None else None,
                "predicted_centroid_y": float(pred_centroid[1]) if pred_centroid is not None else None,
                "centroid_distance": distance,
                "centroid_distance_available": distance is not None,
                "is_no_gt_frame": is_no_gt_frame,
                "no_gt_frame_fp_count": fp if is_no_gt_frame else None,
                "no_gt_frame_background_prob_mean": no_gt_fp_probability_mean,
                "distance_to_nearest_gt_positive_frame": nearest_positive_frame_distance(frame, gt_positive_frames),
                "relative_frame_decile": int(deciles[i]),
                "gt_interval_position": classify_relative_to_runs(frame, gt_runs),
            }
        )
    return rows


def _frame_has_gt_positive(point_label: np.ndarray, valid: np.ndarray, frame_order: np.ndarray, frame: int) -> bool:
    frame_mask = frame_order == frame
    return bool(np.any(frame_mask & valid & (point_label == 1)))


# ----------------------------------------------------------------------------
# Step H3.1: normalized XY density and temporal recurrence
# ----------------------------------------------------------------------------


def assign_xy_grid_bin(xy_normalized: np.ndarray, *, grid_resolution: int) -> np.ndarray:
    require(grid_resolution > 0, "grid_resolution must be positive")
    require(bool(np.all(np.isfinite(xy_normalized))), "non-finite normalized xy")
    require(
        bool(np.all((xy_normalized >= 0.0) & (xy_normalized <= 1.0))),
        "normalized xy out of [0, 1] -- caller must normalize before binning",
    )
    bins = np.minimum((xy_normalized * grid_resolution).astype(np.int64), grid_resolution - 1)
    return bins


def compute_xy_density(xy_normalized: np.ndarray, mask: np.ndarray, *, grid_resolution: int) -> np.ndarray:
    density = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
    selected = xy_normalized[mask]
    if selected.shape[0] == 0:
        return density
    bins = assign_xy_grid_bin(selected, grid_resolution=grid_resolution)
    np.add.at(density, (bins[:, 1], bins[:, 0]), 1)
    return density


def bin_frame_activity(
    frame_order: np.ndarray, xy_normalized: np.ndarray, mask: np.ndarray, *, grid_resolution: int
) -> dict[tuple[int, int], np.ndarray]:
    """For points where `mask` is True, the distinct frames active in each XY bin."""
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size == 0:
        return {}
    bins = assign_xy_grid_bin(xy_normalized[selected_idx], grid_resolution=grid_resolution)
    frames = np.asarray(frame_order)[selected_idx]
    combo = np.stack([bins[:, 1], bins[:, 0], frames], axis=1)
    unique_combo = np.unique(combo, axis=0)
    activity: dict[tuple[int, int], list[int]] = {}
    for row, col, frame in unique_combo.tolist():
        activity.setdefault((int(row), int(col)), []).append(int(frame))
    return {key: np.array(sorted(frames)) for key, frames in activity.items()}


def temporal_recurrence_for_bin(predicted_frames: np.ndarray, gt_positive_frames: np.ndarray) -> dict[str, Any]:
    predicted_runs = find_runs(predicted_frames)
    gt_runs = find_runs(gt_positive_frames) if gt_positive_frames.size else []

    def overlaps_any(run: tuple[int, int], other_runs: list[tuple[int, int]]) -> bool:
        return any(not (run[1] < other[0] or other[1] < run[0]) for other in other_runs)

    matched = [run for run in predicted_runs if overlaps_any(run, gt_runs)]
    unmatched = [run for run in predicted_runs if not overlaps_any(run, gt_runs)]
    unmatched_sorted = sorted(unmatched)
    gaps = [unmatched_sorted[i + 1][0] - unmatched_sorted[i][1] - 1 for i in range(len(unmatched_sorted) - 1)]

    return {
        "predicted_run_count": len(predicted_runs),
        "predicted_run_lengths": [end - start + 1 for start, end in predicted_runs],
        "matched_with_gt_run_count": len(matched),
        "unmatched_run_count": len(unmatched),
        "unmatched_run_lengths": [end - start + 1 for start, end in unmatched],
        "unmatched_run_gaps": gaps,
        "multiple_unmatched_runs": len(unmatched) >= 2,
    }


# ----------------------------------------------------------------------------
# Orchestration: one video (H3 rows + H3.1 per-bin activity)
# ----------------------------------------------------------------------------


def process_video(
    *,
    video_alias: str,
    split: str,
    h5_path: Path,
    npz_path: Path,
    h5_metrics_row: dict[str, str],
    ignore_index: int,
    pixel_width: int,
    pixel_height: int,
) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(h5_path)
    labels = np.asarray(data["point_label"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    frame_order = np.asarray(data["frame_order"])
    pixel_xy = np.asarray(data["pixel_xy"])

    prob_femur, pred_label = load_prediction(npz_path)
    require(
        prob_femur.shape == labels.shape,
        f"{video_alias}: prediction shape {prob_femur.shape} != H5 label shape {labels.shape}",
    )
    check_threshold_0p5_parity(
        labels=labels,
        valid_mask=valid_mask,
        pred_label=pred_label,
        row=h5_metrics_row,
        ignore_index=ignore_index,
        context=f"{split}/{video_alias}",
    )

    xy_normalized = normalize_pixel_xy_with_dimensions(pixel_xy, width=pixel_width, height=pixel_height)

    frame_rows = build_frame_metrics_rows(
        video_alias=video_alias,
        split=split,
        frame_order=frame_order,
        point_label=labels,
        valid_mask=valid_mask,
        pixel_xy=pixel_xy,
        pred_label=pred_label,
        prob_femur=prob_femur,
        ignore_index=ignore_index,
    )

    valid = valid_mask & (labels != int(ignore_index))
    gt_positive_mask = valid & (labels == 1)
    predicted_positive_mask = pred_label == 1
    target_positive = labels == 1
    fp_mask = valid & ~target_positive & predicted_positive_mask
    fn_mask = valid & target_positive & ~predicted_positive_mask
    gt_positive_frames_sorted = np.array(sorted({int(f) for f in frame_order[gt_positive_mask].tolist()}), dtype=np.int64)

    return {
        "video_alias": video_alias,
        "split": split,
        "frame_rows": frame_rows,
        "xy_normalized": xy_normalized,
        "frame_order": frame_order,
        "gt_positive_mask": gt_positive_mask,
        "predicted_positive_mask": predicted_positive_mask,
        "fp_mask": fp_mask,
        "fn_mask": fn_mask,
        "gt_positive_frames_sorted": gt_positive_frames_sorted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-14 Step H3/H3.1: frame-level and normalized-XY-density/temporal-recurrence "
        "diagnostics reusing saved W-A prediction .npz artifacts. No re-inference/torch/CUDA."
    )
    parser.add_argument("--evaluation_dir", required=True, help="evaluate_stage5.py per-checkpoint output_dir")
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--pixel_width", type=int, default=256, help="Step H2.5-confirmed local-crop width")
    parser.add_argument("--pixel_height", type=int, default=256, help="Step H2.5-confirmed local-crop height")
    parser.add_argument("--grid_resolutions", default="16,8", help="Comma-separated grid resolutions; first is primary")
    parser.add_argument("--frame_metrics_csv", default=None)
    parser.add_argument("--video_summary_csv", default=None)
    parser.add_argument("--xy_density_bins_csv", default=None)
    parser.add_argument("--xy_temporal_recurrence_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    grid_resolutions = [int(item) for item in args.grid_resolutions.split(",") if item.strip()]
    require(bool(grid_resolutions), "--grid_resolutions must list at least one resolution")

    videos: list[dict[str, Any]] = []
    for split in ("train_sanity", "validation"):
        rows = load_h5_metrics_rows(evaluation_dir / "h5_metrics.csv", checkpoint=args.checkpoint, split=split)
        for index, row in enumerate(rows):
            # row["video_name"] is the REAL video_name from evaluate_stage5.py's raw
            # (non-anonymized) h5_metrics.csv -- it must only be used internally to
            # resolve the H5/prediction paths on disk, never written into any output
            # row. All output rows use `video_alias` (an enumeration-based label,
            # matching the video_alias convention used elsewhere in this project).
            h5_path = Path(row["h5_path"])
            npz_path = evaluation_dir / "predictions" / split / f"{safe_name(row['video_name'])}.npz"
            video_alias = f"{split}_{index:03d}"
            result = process_video(
                video_alias=video_alias,
                split=split,
                h5_path=h5_path,
                npz_path=npz_path,
                h5_metrics_row=row,
                ignore_index=args.ignore_index,
                pixel_width=args.pixel_width,
                pixel_height=args.pixel_height,
            )
            videos.append(result)

    all_frame_rows: list[dict[str, Any]] = []
    video_summary_rows: list[dict[str, Any]] = []
    for video in videos:
        all_frame_rows.extend(video["frame_rows"])
        no_gt_rows = [row for row in video["frame_rows"] if row["is_no_gt_frame"]]
        video_summary_rows.append(
            {
                "video_alias": video["video_alias"],
                "split": video["split"],
                "num_frames": len(video["frame_rows"]),
                "num_gt_positive_frames": len(video["gt_positive_frames_sorted"]),
                "num_no_gt_frames": len(no_gt_rows),
                "no_gt_frame_total_fp": sum(row["false_positive_count"] for row in no_gt_rows),
                "no_gt_frame_mean_background_prob": (
                    float(np.mean([row["no_gt_frame_background_prob_mean"] for row in no_gt_rows if row["no_gt_frame_background_prob_mean"] is not None]))
                    if any(row["no_gt_frame_background_prob_mean"] is not None for row in no_gt_rows)
                    else None
                ),
            }
        )

    density_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
    for grid_resolution in grid_resolutions:
        for video in videos:
            gt_density = compute_xy_density(video["xy_normalized"], video["gt_positive_mask"], grid_resolution=grid_resolution)
            pred_density = compute_xy_density(video["xy_normalized"], video["predicted_positive_mask"], grid_resolution=grid_resolution)
            fp_density = compute_xy_density(video["xy_normalized"], video["fp_mask"], grid_resolution=grid_resolution)
            fn_density = compute_xy_density(video["xy_normalized"], video["fn_mask"], grid_resolution=grid_resolution)
            for row_idx in range(grid_resolution):
                for col_idx in range(grid_resolution):
                    density_rows.append(
                        {
                            "video_alias": video["video_alias"],
                            "split": video["split"],
                            "grid_resolution": grid_resolution,
                            "bin_row": row_idx,
                            "bin_col": col_idx,
                            "gt_positive_count": int(gt_density[row_idx, col_idx]),
                            "predicted_positive_count": int(pred_density[row_idx, col_idx]),
                            "false_positive_count": int(fp_density[row_idx, col_idx]),
                            "false_negative_count": int(fn_density[row_idx, col_idx]),
                        }
                    )

            predicted_activity = bin_frame_activity(
                video["frame_order"], video["xy_normalized"], video["predicted_positive_mask"], grid_resolution=grid_resolution
            )
            for (row_idx, col_idx), frames in predicted_activity.items():
                recurrence = temporal_recurrence_for_bin(frames, video["gt_positive_frames_sorted"])
                recurrence_rows.append(
                    {
                        "video_alias": video["video_alias"],
                        "split": video["split"],
                        "grid_resolution": grid_resolution,
                        "bin_row": row_idx,
                        "bin_col": col_idx,
                        "predicted_run_count": recurrence["predicted_run_count"],
                        "predicted_run_length_mean": (
                            float(np.mean(recurrence["predicted_run_lengths"])) if recurrence["predicted_run_lengths"] else None
                        ),
                        "matched_with_gt_run_count": recurrence["matched_with_gt_run_count"],
                        "unmatched_run_count": recurrence["unmatched_run_count"],
                        "unmatched_run_gap_mean": (
                            float(np.mean(recurrence["unmatched_run_gaps"])) if recurrence["unmatched_run_gaps"] else None
                        ),
                        "multiple_unmatched_runs": recurrence["multiple_unmatched_runs"],
                    }
                )

    summary = {
        "status": "passed",
        "num_videos": len(videos),
        "grid_resolutions": grid_resolutions,
        "pixel_width": args.pixel_width,
        "pixel_height": args.pixel_height,
        "num_frame_rows": len(all_frame_rows),
        "num_density_rows": len(density_rows),
        "num_recurrence_rows": len(recurrence_rows),
    }

    def write_csv(path_str: str | None, rows: list[dict[str, Any]]) -> None:
        if not path_str or not rows:
            return
        output = Path(path_str)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.frame_metrics_csv, all_frame_rows)
    write_csv(args.video_summary_csv, video_summary_rows)
    write_csv(args.xy_density_bins_csv, density_rows)
    write_csv(args.xy_temporal_recurrence_csv, recurrence_rows)

    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-14 Step H3/H3.1 frame/XY diagnostics passed.")
    print(f"videos: {summary['num_videos']}, frame rows: {summary['num_frame_rows']}")
    print(f"grid resolutions: {grid_resolutions}, density rows: {summary['num_density_rows']}, recurrence rows: {summary['num_recurrence_rows']}")
    if args.summary_json:
        print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
