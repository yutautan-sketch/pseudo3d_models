from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.annotation.stage4_manual_review import (
    LABEL_FEMUR_CANDIDATE,
    load_manual_review_config,
)
from pseudo3d.batch.annotation.batch_import_stage4_phase5_fullvideo_cvat import (
    FullVideoImportError,
    _package_contract,
    _resolve_source,
    _snapshot_contract,
    _task_map_contract,
)


FRAME_FIELDS = (
    "video_name",
    "task_id",
    "frame_stem",
    "frame_order",
    "frame_index",
    "frame_role",
    "review_target",
    "context_only",
    "cvat_mask_status",
    "cvat_mask_positive_pixels",
    "frame_points",
    "source_positive_points",
    "source_positive_inside_cvat_points",
    "source_positive_outside_cvat_points",
    "projected_positive_points",
    "retained_positive_points",
    "added_positive_points",
    "removed_positive_points",
    "bbox_count",
    "no_bbox_cvat_positive",
    "no_bbox_projected_positive_points",
)

VIDEO_FIELDS = (
    "video_name",
    "task_id",
    "task_frames",
    "review_target_frames",
    "context_only_frames",
    "unreviewable_target_frames",
    "cvat_positive_frames",
    "cvat_empty_frames",
    "cvat_omitted_as_empty_frames",
    "cvat_mask_positive_pixels",
    "frame_points",
    "source_positive_points",
    "source_positive_inside_cvat_points",
    "source_positive_outside_cvat_points",
    "projected_positive_points",
    "retained_positive_points",
    "added_positive_points",
    "removed_positive_points",
    "no_bbox_cvat_positive_frames",
    "no_bbox_projected_positive_points",
)


class AuthoritativePreflightError(FullVideoImportError):
    """Raised when the read-only CVAT-authority preflight contract fails."""


def summarize_authoritative_frame_points(
    *,
    point_labels: np.ndarray,
    pixel_xy: np.ndarray,
    cvat_mask: np.ndarray,
) -> dict[str, int]:
    labels = np.asarray(point_labels, dtype=np.int8)
    points = np.asarray(pixel_xy)
    mask = np.asarray(cvat_mask, dtype=bool)
    if labels.ndim != 1 or points.shape != (labels.size, 2):
        raise AuthoritativePreflightError(
            "Frame point labels/pixel_xy must have aligned shapes"
        )
    if mask.ndim != 2:
        raise AuthoritativePreflightError("CVAT mask must be 2-D")
    if not np.all(np.isfinite(points)):
        raise AuthoritativePreflightError("pixel_xy contains NaN or Inf")
    xy = np.rint(points).astype(np.int64)
    height, width = mask.shape
    if xy.size and (
        np.any(xy[:, 0] < 0)
        or np.any(xy[:, 0] >= width)
        or np.any(xy[:, 1] < 0)
        or np.any(xy[:, 1] >= height)
    ):
        raise AuthoritativePreflightError("pixel_xy lies outside CVAT mask")
    sampled = (
        mask[xy[:, 1], xy[:, 0]]
        if xy.size
        else np.zeros((0,), dtype=bool)
    )
    source_positive = labels == LABEL_FEMUR_CANDIDATE
    retained = source_positive & sampled
    removed = source_positive & ~sampled
    added = ~source_positive & sampled
    projected = sampled
    result = {
        "frame_points": int(labels.size),
        "source_positive_points": int(np.sum(source_positive)),
        "source_positive_inside_cvat_points": int(np.sum(retained)),
        "source_positive_outside_cvat_points": int(np.sum(removed)),
        "projected_positive_points": int(np.sum(projected)),
        "retained_positive_points": int(np.sum(retained)),
        "added_positive_points": int(np.sum(added)),
        "removed_positive_points": int(np.sum(removed)),
    }
    if result["source_positive_points"] != (
        result["source_positive_inside_cvat_points"]
        + result["source_positive_outside_cvat_points"]
    ):
        raise AssertionError("Source-positive partition is inconsistent")
    if result["projected_positive_points"] != (
        result["retained_positive_points"] + result["added_positive_points"]
    ):
        raise AssertionError("Projected-positive partition is inconsistent")
    if result["removed_positive_points"] != result[
        "source_positive_outside_cvat_points"
    ]:
        raise AssertionError("Removed-positive count is inconsistent")
    return result


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_frame_arrays(
    source_h5: Path,
    *,
    expected_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(source_h5, "r") as handle:
        for name in (
            "point_cloud/frame_order",
            "point_cloud/pixel_xy",
            "annotation/point_label",
            "annotation/valid_mask",
        ):
            if name not in handle:
                raise AuthoritativePreflightError(
                    f"Source H5 lacks required dataset {name}: {source_h5}"
                )
        orders = handle["point_cloud/frame_order"][:].astype(np.int32)
        pixel_xy = handle["point_cloud/pixel_xy"][:].astype(np.float32)
        labels = handle["annotation/point_label"][:].astype(np.int8)
        valid = handle["annotation/valid_mask"][:].astype(bool)
        stored_num_frames = int(handle.attrs.get("num_frames", -1))
        per_frame_count = (
            int(handle["point_cloud/per_frame_counts"].shape[0])
            if "per_frame_counts" in handle["point_cloud"]
            else -1
        )
    for value, source in (
        (stored_num_frames, "num_frames metadata"),
        (per_frame_count, "point_cloud/per_frame_counts"),
    ):
        if value >= 0 and value != expected_frames:
            raise AuthoritativePreflightError(
                f"Source {source} differs from CVAT Task inventory: "
                f"{source_h5}/{value}!={expected_frames}"
            )
    if orders.shape != labels.shape or valid.shape != labels.shape:
        raise AuthoritativePreflightError(
            f"Source point arrays have different lengths: {source_h5}"
        )
    if pixel_xy.shape != (labels.size, 2):
        raise AuthoritativePreflightError(
            f"Source pixel_xy shape differs from point labels: {source_h5}"
        )
    if not np.array_equal(valid, labels != -1):
        raise AuthoritativePreflightError(
            f"Source valid_mask differs from point_label != -1: {source_h5}"
        )
    unknown_labels = sorted(set(int(value) for value in np.unique(labels)) - {-1, 0, 1})
    if unknown_labels:
        raise AuthoritativePreflightError(
            f"Source point_label contains unknown values {unknown_labels}: {source_h5}"
        )
    if orders.size and (
        int(orders.min()) < 0 or int(orders.max()) >= expected_frames
    ):
        raise AuthoritativePreflightError(
            f"Source frame_order escapes CVAT Task inventory: {source_h5}"
        )
    counts = np.bincount(orders, minlength=expected_frames).astype(np.int64)
    sorted_indices = np.argsort(orders, kind="stable")
    offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(counts))
    )
    return orders, pixel_xy, labels, sorted_indices, offsets


def _sum_fields(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> dict[str, int]:
    return {
        name: sum(int(row[name]) for row in rows)
        for name in names
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    exclusion_path = Path(args.exclusion_manifest).resolve()
    source_root = Path(args.source_annotated_root).resolve()
    review_root = Path(args.review_root).resolve()
    snapshot_root = Path(args.snapshot_root).resolve()
    task_map_csv = Path(args.task_map_csv).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.is_symlink():
        raise AuthoritativePreflightError("output_root must not be a symlink")
    if output_root.exists() and not output_root.is_dir():
        raise AuthoritativePreflightError("output_root must be a directory")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output_root is not empty; pass --overwrite: {output_root}"
        )

    manual_config = load_manual_review_config(args.manual_review_config)
    if manual_config.label_name != "femur":
        raise AuthoritativePreflightError("Preflight requires the fixed femur label")
    manifest_items = load_stage4_sweep_manifest(manifest_path)
    exclusions = load_exclusions(exclusion_path)
    excluded = {row.video_name for row in exclusions}
    enabled = [
        row
        for row in manifest_items
        if row.enabled and row.video_name not in excluded
    ]
    if len(enabled) != int(args.expected_videos):
        raise AuthoritativePreflightError(
            f"Enabled video count mismatch: {len(enabled)} != {args.expected_videos}"
        )
    manifest_names = {row.video_name for row in manifest_items}
    if not excluded.issubset(manifest_names):
        raise AuthoritativePreflightError(
            "Exclusion manifest contains videos absent from training manifest"
        )

    snapshots = _snapshot_contract(
        snapshot_root, int(args.expected_reviewed_videos)
    )
    _task_map_contract(task_map_csv, snapshots)
    enabled_names = {row.video_name for row in enabled}
    if set(snapshots) - enabled_names:
        raise AuthoritativePreflightError(
            "Snapshot contains disabled/excluded videos: "
            f"{sorted(set(snapshots) - enabled_names)}"
        )
    review_video_root = review_root / "videos"
    if not review_video_root.is_dir():
        raise FileNotFoundError(f"Review videos directory not found: {review_video_root}")
    review_video_dirs = {
        path.name for path in review_video_root.iterdir() if path.is_dir()
    }
    if set(snapshots) != review_video_dirs - excluded:
        raise AuthoritativePreflightError(
            "Snapshot video set must equal review packages minus exclusions"
        )
    sources = {
        video: _resolve_source(source_root, video) for video in sorted(snapshots)
    }

    protected_paths = {
        manifest_path,
        exclusion_path,
        Path(args.manual_review_config).resolve(),
        snapshot_root / "export_manifest.csv",
        snapshot_root / "export_summary.json",
        task_map_csv,
    }
    for video, snapshot in snapshots.items():
        protected_paths.update(
            {
                sources[video],
                Path(snapshot["annotation_path"]),
                Path(snapshot["backup_path"]),
            }
        )
        package_root = review_video_root / video
        protected_paths.update(
            package_root / name
            for name in (
                "review_frames.csv",
                "review_bboxes.csv",
                "checksums.csv",
                "export_summary.json",
                "run_config.yaml",
            )
        )
    before_hashes = {
        path: file_sha256(path) for path in sorted(protected_paths)
    }

    print("Stage 4 CVAT-authoritative real-data preflight (read-only)")
    print(f"  manifest       : {manifest_path}")
    print(f"  source root    : {source_root}")
    print(f"  review root    : {review_root}")
    print(f"  snapshot root  : {snapshot_root}")
    print(f"  reviewed videos: {len(snapshots)}")
    print(f"  output root    : {output_root}")

    frame_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    actionable_bboxes = 0
    for video_index, video in enumerate(sorted(snapshots), start=1):
        snapshot = snapshots[video]
        source_h5 = sources[video]
        package = _package_contract(
            review_root,
            video,
            source_h5,
            snapshot,
            manual_config.fingerprint,
        )
        actionable_bboxes += int(package["actionable_bboxes"])
        authoritative_frames = package["authoritative_frames"]
        expected_orders = list(range(len(authoritative_frames)))
        actual_orders = [int(row["frame_order"]) for row in authoritative_frames]
        if actual_orders != expected_orders:
            raise AuthoritativePreflightError(
                f"Task frame inventory is not complete/in order: {video}"
            )
        _, pixel_xy, labels, sorted_indices, offsets = _source_frame_arrays(
            source_h5,
            expected_frames=len(authoritative_frames),
        )
        current_video_rows: list[dict[str, Any]] = []
        task_id = int(snapshot["task_id"])
        for frame in authoritative_frames:
            order = int(frame["frame_order"])
            indices = sorted_indices[offsets[order] : offsets[order + 1]]
            stats = summarize_authoritative_frame_points(
                point_labels=labels[indices],
                pixel_xy=pixel_xy[indices],
                cvat_mask=np.asarray(frame["mask"], dtype=bool),
            )
            stem = str(frame["frame_stem"])
            bbox_count = len(package["all_bbox_by_stem"].get(stem, ()))
            no_bbox_positive = int(
                bbox_count == 0 and int(frame["cvat_mask_positive_pixels"]) > 0
            )
            row = {
                "video_name": video,
                "task_id": task_id,
                "frame_stem": stem,
                "frame_order": order,
                "frame_index": int(frame["frame_index"]),
                "frame_role": str(frame["frame_role"]),
                "review_target": int(bool(frame["review_target"])),
                "context_only": int(bool(frame["context_only"])),
                "cvat_mask_status": str(frame["cvat_mask_status"]),
                "cvat_mask_positive_pixels": int(
                    frame["cvat_mask_positive_pixels"]
                ),
                **stats,
                "bbox_count": bbox_count,
                "no_bbox_cvat_positive": no_bbox_positive,
                "no_bbox_projected_positive_points": (
                    stats["projected_positive_points"] if no_bbox_positive else 0
                ),
            }
            current_video_rows.append(row)
            frame_rows.append(row)

        integer_fields = (
            "cvat_mask_positive_pixels",
            "frame_points",
            "source_positive_points",
            "source_positive_inside_cvat_points",
            "source_positive_outside_cvat_points",
            "projected_positive_points",
            "retained_positive_points",
            "added_positive_points",
            "removed_positive_points",
            "no_bbox_cvat_positive",
            "no_bbox_projected_positive_points",
        )
        totals = _sum_fields(current_video_rows, integer_fields)
        status_counts = Counter(
            row["cvat_mask_status"] for row in current_video_rows
        )
        review_target_frames = sum(
            int(row["review_target"]) for row in current_video_rows
        )
        context_only_frames = sum(
            int(row["context_only"]) for row in current_video_rows
        )
        unreviewable_target_frames = sum(
            int(row["review_target"]) and int(row["context_only"])
            for row in current_video_rows
        )
        video_row = {
            "video_name": video,
            "task_id": task_id,
            "task_frames": len(current_video_rows),
            "review_target_frames": review_target_frames,
            "context_only_frames": context_only_frames,
            "unreviewable_target_frames": unreviewable_target_frames,
            "cvat_positive_frames": status_counts["positive"],
            "cvat_empty_frames": status_counts["empty"],
            "cvat_omitted_as_empty_frames": status_counts["omitted_as_empty"],
            **{
                name: totals[name]
                for name in integer_fields
                if name != "no_bbox_cvat_positive"
            },
            "no_bbox_cvat_positive_frames": totals["no_bbox_cvat_positive"],
        }
        video_rows.append(video_row)
        print(
            f"  [{video_index}/{len(snapshots)}] [OK] {video}: "
            f"frames={len(current_video_rows)}, "
            f"CVAT+={video_row['cvat_positive_frames']}, "
            f"old_outside={video_row['source_positive_outside_cvat_points']}, "
            f"added={video_row['added_positive_points']}, "
            f"removed={video_row['removed_positive_points']}"
        )

    if actionable_bboxes != int(args.expected_actionable_bboxes):
        raise AuthoritativePreflightError(
            "Actionable BBox count mismatch: "
            f"{actionable_bboxes} != {args.expected_actionable_bboxes}"
        )
    if int(args.expected_task_frames) >= 0 and len(frame_rows) != int(
        args.expected_task_frames
    ):
        raise AuthoritativePreflightError(
            f"Task frame count mismatch: {len(frame_rows)} != {args.expected_task_frames}"
        )

    after_hashes = {path: file_sha256(path) for path in sorted(protected_paths)}
    changed = [
        str(path)
        for path in before_hashes
        if before_hashes[path] != after_hashes[path]
    ]
    if changed:
        raise RuntimeError(f"Preflight input files changed: {changed}")

    status_counts = Counter(row["cvat_mask_status"] for row in frame_rows)
    review_target_frames = sum(int(row["review_target"]) for row in frame_rows)
    context_only_frames = sum(int(row["context_only"]) for row in frame_rows)
    unreviewable_target_frames = sum(
        int(row["review_target"]) and int(row["context_only"])
        for row in frame_rows
    )
    aggregate_fields = (
        "cvat_mask_positive_pixels",
        "frame_points",
        "source_positive_points",
        "source_positive_inside_cvat_points",
        "source_positive_outside_cvat_points",
        "projected_positive_points",
        "retained_positive_points",
        "added_positive_points",
        "removed_positive_points",
        "no_bbox_cvat_positive",
        "no_bbox_projected_positive_points",
    )
    totals = _sum_fields(frame_rows, aggregate_fields)
    summary = {
        "schema_version": 1,
        "status": "ok",
        "preflight_kind": "cvat_snapshot_authoritative_point_projection",
        "h5_files_written": 0,
        "input_files_unchanged": True,
        "enabled_videos": len(enabled),
        "reviewed_videos": len(snapshots),
        "non_snapshot_videos": len(enabled) - len(snapshots),
        "excluded_videos": sorted(excluded),
        "task_frames": len(frame_rows),
        "review_target_frames": review_target_frames,
        "context_only_frames": context_only_frames,
        "unreviewable_target_frames": unreviewable_target_frames,
        "cvat_positive_frames": status_counts["positive"],
        "cvat_empty_frames": status_counts["empty"],
        "cvat_omitted_as_empty_frames": status_counts["omitted_as_empty"],
        **{
            name: totals[name]
            for name in aggregate_fields
            if name != "no_bbox_cvat_positive"
        },
        "old_automatic_positive_points": totals["source_positive_points"],
        "no_bbox_cvat_positive_frames": totals["no_bbox_cvat_positive"],
        "actionable_bboxes": actionable_bboxes,
        "manifest_sha256": file_sha256(manifest_path),
        "exclusion_manifest_sha256": file_sha256(exclusion_path),
        "snapshot_manifest_sha256": file_sha256(
            snapshot_root / "export_manifest.csv"
        ),
        "task_map_sha256": file_sha256(task_map_csv),
        "manual_review_fingerprint": manual_config.fingerprint,
        "failure_rows": 0,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        _write_csv(temporary / "frame_metrics.csv", frame_rows, FRAME_FIELDS)
        _write_csv(temporary / "video_summary.csv", video_rows, VIDEO_FIELDS)
        _write_csv(
            temporary / "input_checksums.csv",
            [
                {"path": str(path), "sha256": before_hashes[path]}
                for path in sorted(before_hashes)
            ],
            ("path", "sha256"),
        )
        _write_json(temporary / "preflight_summary.json", summary)
        if output_root.exists():
            if any(output_root.iterdir()) and not args.overwrite:
                raise FileExistsError(
                    f"output_root became non-empty: {output_root}"
                )
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("\nCVAT-authoritative preflight summary")
    for name in (
        "reviewed_videos",
        "task_frames",
        "review_target_frames",
        "context_only_frames",
        "unreviewable_target_frames",
        "cvat_positive_frames",
        "cvat_empty_frames",
        "cvat_omitted_as_empty_frames",
        "cvat_mask_positive_pixels",
        "source_positive_points",
        "source_positive_inside_cvat_points",
        "source_positive_outside_cvat_points",
        "projected_positive_points",
        "added_positive_points",
        "removed_positive_points",
        "no_bbox_cvat_positive_frames",
    ):
        print(f"  {name:38s}: {summary[name]}")
    print(f"  output_root                           : {output_root}")
    print("Stage 4 CVAT-authoritative real-data preflight passed.")
    return {"summary": summary, "frames": frame_rows, "videos": video_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only point-center comparison of the source Stage 4 labels "
            "against every returned full-video CVAT snapshot mask."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclusion_manifest", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--review_root", type=Path, required=True)
    parser.add_argument("--snapshot_root", type=Path, required=True)
    parser.add_argument("--task_map_csv", type=Path, required=True)
    parser.add_argument("--manual_review_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, default=181)
    parser.add_argument("--expected_reviewed_videos", type=int, default=59)
    parser.add_argument("--expected_actionable_bboxes", type=int, default=112)
    parser.add_argument("--expected_task_frames", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_preflight(args)
    except Exception as exc:
        raise SystemExit(
            "Stage 4 CVAT-authoritative real-data preflight failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
