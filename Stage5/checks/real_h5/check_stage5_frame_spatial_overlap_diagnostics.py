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

from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list

# ----------------------------------------------------------------------------
# S5-14: coordinate-prior / frame-phase / overlap-exposure diagnostics.
#
# This module is organized in phases matching the request handoff:
#   Step H1  shared geometry/exposure primitives (below)
#   Step H2  teacher v7 GT + structural window-exposure audit (this file,
#            `audit_h5`/`main`) -- pure h5py/numpy, no predictions, no torch.
#   Step H3/H3.1 (frame/XY diagnostics on saved W-A predictions) and
#   Step H4 (per-window context, reusing check_stage5_overlap_aggregation's
#   accumulator) are implemented in follow-up work; H4 will need a
#   function-local torch import since only it needs a model forward pass.
#   Everything in this file stays torch-free so it can run without CUDA.
# ----------------------------------------------------------------------------


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------
# Step H1: shared primitives
# ----------------------------------------------------------------------------


def find_runs(sorted_unique_values: np.ndarray) -> list[tuple[int, int]]:
    """Split a sorted array of distinct integers into maximal contiguous runs.

    A run is a maximal subsequence where each value is exactly 1 more than the
    previous (i.e. a gap of >1 breaks the run). Used for both GT-positive
    frame runs and predicted-positive frame runs per XY bin (Step H3.1).
    """
    values = np.asarray(sorted_unique_values)
    if values.size == 0:
        return []
    require(bool(np.all(np.diff(values) > 0)), "find_runs requires strictly increasing, distinct values")
    breaks = np.flatnonzero(np.diff(values) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, values.size - 1]
    return [(int(values[s]), int(values[e])) for s, e in zip(starts, ends)]


def per_frame_label_counts(
    frame_order: np.ndarray,
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    *,
    ignore_index: int,
) -> dict[int, dict[str, int]]:
    """Per distinct frame_order value: positive/background/ignore point counts."""
    frame_order = np.asarray(frame_order)
    point_label = np.asarray(point_label)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    require(
        frame_order.shape == point_label.shape == valid_mask.shape,
        "frame_order/point_label/valid_mask must share shape",
    )
    unique_frames, inverse = np.unique(frame_order, return_inverse=True)
    num_frames = unique_frames.size

    is_positive = valid_mask & (point_label == 1)
    is_background = valid_mask & (point_label == 0)
    is_ignore = (~valid_mask) | (point_label == int(ignore_index))

    positive = np.zeros(num_frames, dtype=np.int64)
    background = np.zeros(num_frames, dtype=np.int64)
    ignore = np.zeros(num_frames, dtype=np.int64)
    np.add.at(positive, inverse, is_positive.astype(np.int64))
    np.add.at(background, inverse, is_background.astype(np.int64))
    np.add.at(ignore, inverse, is_ignore.astype(np.int64))

    return {
        int(frame): {"positive": int(p), "background": int(b), "ignore": int(g)}
        for frame, p, b, g in zip(unique_frames.tolist(), positive.tolist(), background.tolist(), ignore.tolist())
    }


def window_occurrence_counts(
    frame_order: np.ndarray,
    *,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
) -> tuple[np.ndarray, int]:
    """Per-point count of how many training-style overlap windows contain it.

    Pure structural geometry (mirrors the exact window generation used by
    Pseudo3DPointCloudDataset / train_stage5.py) -- no model forward pass.
    Returns (counts, num_windows).
    """
    frame_order = np.asarray(frame_order)
    windows = generate_frame_order_windows(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    counts = np.zeros(frame_order.shape[0], dtype=np.int64)
    for window in windows:
        indices = point_indices_for_window(frame_order, window)
        counts[indices] += 1
    return counts, len(windows)


def relative_frame_decile(frame_order: np.ndarray, frame_values: np.ndarray) -> np.ndarray:
    """Decile in [0, 9] of ``frame_values`` within [min(frame_order), max(frame_order)]."""
    frame_order = np.asarray(frame_order)
    frame_values = np.asarray(frame_values, dtype=np.float64)
    require(frame_order.size > 0, "relative_frame_decile requires a non-empty frame_order")
    min_frame = float(frame_order.min())
    max_frame = float(frame_order.max())
    span = max(max_frame - min_frame, 1.0)
    fraction = np.clip((frame_values - min_frame) / span, 0.0, 1.0)
    return np.minimum((fraction * 10.0).astype(np.int64), 9)


def class_exposure_summary(
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    occurrence_counts: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Unique-point count vs. total window-occurrence count, by GT class.

    Deliberately keeps "unique point" and "window occurrence" separate
    (handoff Step H2: "unique-point集計とwindow-occurrence集計を混ぜない").
    """
    point_label = np.asarray(point_label)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    is_positive = valid_mask & (point_label == 1)
    is_background = valid_mask & (point_label == 0)
    summary: dict[str, dict[str, Any]] = {}
    for name, mask in (("positive", is_positive), ("background", is_background)):
        unique_count = int(np.sum(mask))
        total_occurrence = int(np.sum(occurrence_counts[mask])) if unique_count > 0 else 0
        summary[name] = {
            "unique_point_count": unique_count,
            "window_occurrence_total": total_occurrence,
            "exposure_multiplier": (total_occurrence / unique_count) if unique_count > 0 else None,
        }
    return summary


def positive_frame_runs(frame_counts: dict[int, dict[str, int]]) -> list[tuple[int, int]]:
    positive_frames = np.array(sorted(frame for frame, counts in frame_counts.items() if counts["positive"] > 0), dtype=np.int64)
    return find_runs(positive_frames)


def require_no_duplicate_or_missing_points(point_indices: np.ndarray, *, expected_count: int, context: str) -> None:
    unique_indices = np.unique(point_indices)
    require(
        unique_indices.size == point_indices.size,
        f"{context}: duplicate point indices found ({point_indices.size} entries, {unique_indices.size} unique)",
    )
    require(
        unique_indices.size == expected_count
        and bool(np.array_equal(unique_indices, np.arange(expected_count, dtype=unique_indices.dtype))),
        f"{context}: point indices do not exactly cover [0, {expected_count})",
    )


def require_finite(values: np.ndarray, *, context: str) -> None:
    require(bool(np.all(np.isfinite(values))), f"{context}: non-finite value(s) present")


# ----------------------------------------------------------------------------
# Step H2: teacher v7 GT + structural window-exposure audit (CPU/h5py only)
# ----------------------------------------------------------------------------


def audit_h5(
    path: Path,
    *,
    ignore_index: int,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(path)
    frame_order = np.asarray(data["frame_order"])
    point_label = np.asarray(data["point_label"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    require(
        frame_order.shape == point_label.shape == valid_mask.shape,
        f"{path}: frame_order/point_label/valid_mask shape mismatch",
    )
    require(frame_order.size > 0, f"{path}: empty point cloud")

    frame_counts = per_frame_label_counts(frame_order, point_label, valid_mask, ignore_index=ignore_index)
    occurrence_counts, num_windows = window_occurrence_counts(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    require_no_duplicate_or_missing_points(
        np.arange(frame_order.size), expected_count=frame_order.size, context=str(path)
    )
    require(bool(np.all(occurrence_counts >= 0)), f"{path}: negative window-occurrence count")

    exposure = class_exposure_summary(point_label, valid_mask, occurrence_counts)
    runs = positive_frame_runs(frame_counts)
    run_lengths = [end - start + 1 for start, end in runs]
    run_gaps = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]

    unique_frames = np.array(sorted(frame_counts), dtype=np.int64)
    deciles = relative_frame_decile(frame_order, unique_frames)
    decile_exposure: dict[int, dict[str, float]] = {}
    for decile in range(10):
        frame_mask_for_decile = deciles == decile
        frames_in_decile = unique_frames[frame_mask_for_decile]
        if frames_in_decile.size == 0:
            decile_exposure[decile] = {"positive_count": 0, "background_count": 0, "mean_occurrence": None}
            continue
        point_mask = np.isin(frame_order, frames_in_decile)
        positive_point_mask = point_mask & valid_mask & (point_label == 1)
        background_point_mask = point_mask & valid_mask & (point_label == 0)
        decile_exposure[decile] = {
            "positive_count": int(np.sum(positive_point_mask)),
            "background_count": int(np.sum(background_point_mask)),
            "mean_occurrence": float(np.mean(occurrence_counts[point_mask])) if np.any(point_mask) else None,
        }

    has_positive_frames = [frame for frame, counts in frame_counts.items() if counts["positive"] > 0]
    return {
        "num_points": int(frame_order.size),
        "min_frame": int(frame_order.min()),
        "max_frame": int(frame_order.max()),
        "num_distinct_frames": len(frame_counts),
        "num_windows": num_windows,
        "num_frames_with_positive": len(has_positive_frames),
        "class_exposure": exposure,
        "positive_frame_run_count": len(runs),
        "positive_frame_run_lengths": run_lengths,
        "positive_frame_run_gaps": run_gaps,
        "decile_exposure": decile_exposure,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-14 Step H2: teacher v7 GT + structural window-exposure audit. CPU/h5py only, no CUDA."
    )
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument("--max_files", type=int, default=0, help="0 means all files in the given list(s)")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--window_size_frames", type=int, default=16)
    parser.add_argument("--window_stride_frames", type=int, default=8)
    parser.add_argument(
        "--no_include_tail_window", dest="include_tail_window", action="store_false"
    )
    parser.set_defaults(include_tail_window=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None, help="Per-video summary CSV (gt_overlap_exposure.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(bool(args.train_list) or bool(args.val_list), "Provide --train_list and/or --val_list")

    entries: list[tuple[str, Path]] = []
    for split, list_path in (("train", args.train_list), ("val", args.val_list)):
        if list_path:
            for path in read_path_list(list_path):
                entries.append((split, path))
    if args.max_files > 0:
        entries = entries[: args.max_files]
    require(bool(entries), "No H5 paths resolved from the given list(s)")

    per_video_rows: list[dict[str, Any]] = []
    for split, path in entries:
        require(path.is_file(), f"H5 not found: {path}")
        result = audit_h5(
            path,
            ignore_index=args.ignore_index,
            window_size_frames=args.window_size_frames,
            window_stride_frames=args.window_stride_frames,
            include_tail_window=args.include_tail_window,
        )
        row = {
            "split": split,
            "video_alias": f"{split}_{len(per_video_rows):03d}",
            "num_points": result["num_points"],
            "min_frame": result["min_frame"],
            "max_frame": result["max_frame"],
            "num_distinct_frames": result["num_distinct_frames"],
            "num_windows": result["num_windows"],
            "num_frames_with_positive": result["num_frames_with_positive"],
            "positive_unique_point_count": result["class_exposure"]["positive"]["unique_point_count"],
            "positive_window_occurrence_total": result["class_exposure"]["positive"]["window_occurrence_total"],
            "positive_exposure_multiplier": result["class_exposure"]["positive"]["exposure_multiplier"],
            "background_unique_point_count": result["class_exposure"]["background"]["unique_point_count"],
            "background_window_occurrence_total": result["class_exposure"]["background"]["window_occurrence_total"],
            "background_exposure_multiplier": result["class_exposure"]["background"]["exposure_multiplier"],
            "positive_frame_run_count": result["positive_frame_run_count"],
            "positive_frame_run_length_mean": (
                float(np.mean(result["positive_frame_run_lengths"])) if result["positive_frame_run_lengths"] else None
            ),
            "positive_frame_run_length_max": (
                max(result["positive_frame_run_lengths"]) if result["positive_frame_run_lengths"] else None
            ),
            "positive_frame_run_gap_mean": (
                float(np.mean(result["positive_frame_run_gaps"])) if result["positive_frame_run_gaps"] else None
            ),
        }
        per_video_rows.append(row)

    summary = {
        "status": "passed",
        "num_files": len(per_video_rows),
        "window_size_frames": args.window_size_frames,
        "window_stride_frames": args.window_stride_frames,
        "include_tail_window": args.include_tail_window,
        "positive_exposure_multiplier_mean": float(
            np.mean([row["positive_exposure_multiplier"] for row in per_video_rows if row["positive_exposure_multiplier"] is not None])
        ),
        "background_exposure_multiplier_mean": float(
            np.mean([row["background_exposure_multiplier"] for row in per_video_rows if row["background_exposure_multiplier"] is not None])
        ),
    }

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(per_video_rows[0].keys()) if per_video_rows else []
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_video_rows)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_video": per_video_rows}, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-14 Step H2 GT/exposure audit passed.")
    print(f"files audited: {summary['num_files']}")
    print(f"positive exposure multiplier (mean of per-video means): {summary['positive_exposure_multiplier_mean']:.4f}")
    print(f"background exposure multiplier (mean of per-video means): {summary['background_exposure_multiplier_mean']:.4f}")
    if args.output_csv:
        print(f"output csv: {args.output_csv}")
    if args.output_json:
        print(f"output json: {args.output_json}")


if __name__ == "__main__":
    main()
