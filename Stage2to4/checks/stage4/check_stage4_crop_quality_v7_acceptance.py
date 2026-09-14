from __future__ import annotations

import argparse
import csv
import json
import os
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
from pseudo3d.analysis.validate_stage4_crop_quality_manifests import (
    CropQualityInvalidation,
    invalidation_fingerprint,
    load_crop_quality_invalidations,
)
from pseudo3d.annotation.apply_crop_quality_invalidations import (
    FRAME_AUTHORITY,
    GROUP_NAME,
    OUTPUT_TOKEN,
    SOURCE_TOKEN,
)


DATASET_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711"
)
SOURCE_RUN_NAME = (
    "global_local_l75_w31_c12_area15_"
    "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
)
OUTPUT_RUN_NAME = (
    "global_local_l75_w31_c12_area15_"
    "bboxrank_v7_cvat_authoritative_crop_quality_v1"
)
MANIFEST_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/"
    "stage4_sampling_parameter_sweep/260711/manifests"
)

LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_POSITIVE = 1
NEW_FRAME_PROVENANCE_DATASETS = {
    "manual_review_fullvideo_frames/crop_invalidation_action",
    "manual_review_fullvideo_frames/crop_invalidation_reason_code",
    "manual_review_fullvideo_frames/crop_invalidation_manifest_sha256",
}


class AcceptanceError(RuntimeError):
    """Raised when the versioned teacher-v7 build violates its fixed contract."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"JSON root must be an object: {path}")
    return dict(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_or_verify_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> str:
    content = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return "verified_existing"
        if not overwrite:
            raise FileExistsError(f"Acceptance report differs; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return "written"


def _single_annotated(root: Path, video: str, token: str) -> Path:
    matches = sorted((root / video).glob(f"*{token}.h5"))
    if len(matches) != 1:
        raise AcceptanceError(
            f"Expected one {token} annotated H5 for {video}, found {len(matches)}"
        )
    return matches[0].resolve()


def _single_collected(root: Path, video: str) -> Path:
    matches = sorted(
        path
        for path in root.glob(f"*{OUTPUT_TOKEN}.h5")
        if path.name.startswith(f"{video}_")
    )
    if len(matches) != 1:
        raise AcceptanceError(
            f"Expected one collected v7 H5 for {video}, found {len(matches)}"
        )
    return matches[0].resolve()


def _dataset_paths(handle: h5py.File | h5py.Group) -> set[str]:
    result: set[str] = set()

    def visitor(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            result.add(name)

    handle.visititems(visitor)
    return result


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _read_dataset(dataset: h5py.Dataset) -> np.ndarray:
    """Read both regular and scalar HDF5 datasets as NumPy arrays."""
    value = dataset[()] if dataset.shape == () else dataset[:]
    return np.asarray(value)


def _assert_dataset_equal(
    source: h5py.File | h5py.Group,
    final: h5py.File | h5py.Group,
    path: str,
    *,
    video: str,
) -> None:
    if path not in final:
        raise AcceptanceError(f"v7 lacks preserved dataset {path}: {video}")
    left = source[path]
    right = final[path]
    if not isinstance(left, h5py.Dataset) or not isinstance(right, h5py.Dataset):
        raise AcceptanceError(f"Preserved path is not a dataset: {video}/{path}")
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AcceptanceError(f"Preserved dataset schema changed: {video}/{path}")
    if not _arrays_equal(_read_dataset(left), _read_dataset(right)):
        raise AcceptanceError(f"Preserved dataset values changed: {video}/{path}")


def _assert_unaffected_semantic_identity(
    source: h5py.File,
    final: h5py.File,
    *,
    video: str,
) -> None:
    source_paths = _dataset_paths(source)
    final_paths = _dataset_paths(final)
    missing = sorted(source_paths - final_paths)
    if missing:
        raise AcceptanceError(f"v7 lost source datasets for {video}: {missing}")
    extra = final_paths - source_paths
    unexpected = sorted(
        path
        for path in extra
        if not path.startswith(f"{GROUP_NAME}/")
        and path not in NEW_FRAME_PROVENANCE_DATASETS
    )
    if unexpected:
        raise AcceptanceError(f"v7 has unexpected datasets for {video}: {unexpected}")
    for path in sorted(source_paths):
        _assert_dataset_equal(source, final, path, video=video)


def _assert_filtered_group(
    source: h5py.File,
    final: h5py.File,
    group_name: str,
    remove: np.ndarray,
    *,
    video: str,
) -> None:
    if group_name not in source:
        if group_name in final:
            raise AcceptanceError(f"v7 unexpectedly added {group_name}: {video}")
        return
    if group_name not in final:
        raise AcceptanceError(f"v7 removed group {group_name}: {video}")
    source_group = source[group_name]
    final_group = final[group_name]
    source_paths = _dataset_paths(source_group)
    final_paths = _dataset_paths(final_group)
    if source_paths != final_paths:
        raise AcceptanceError(f"Filtered group schema changed: {video}/{group_name}")
    for relative in sorted(source_paths):
        source_data = _read_dataset(source_group[relative])
        final_data = _read_dataset(final_group[relative])
        if source_data.ndim < 1 or remove.shape != (source_data.shape[0],):
            raise AcceptanceError(f"Unaligned filtered group: {video}/{group_name}")
        expected = source_data[~remove]
        if not _arrays_equal(expected, final_data):
            raise AcceptanceError(
                f"Filtered group differs outside target: {video}/{group_name}/{relative}"
            )


def _assert_archive_matches(
    source_group: h5py.Group,
    archive_group: h5py.Group,
    selector: np.ndarray,
    *,
    video: str,
    archive_name: str,
) -> None:
    source_paths = _dataset_paths(source_group)
    archive_paths = _dataset_paths(archive_group)
    if source_paths != archive_paths:
        raise AcceptanceError(f"Archive schema differs: {video}/{archive_name}")
    for relative in sorted(source_paths):
        source_data = _read_dataset(source_group[relative])
        archive_data = _read_dataset(archive_group[relative])
        if source_data.ndim < 1 or selector.shape != (source_data.shape[0],):
            raise AcceptanceError(f"Archive selector is unaligned: {video}/{archive_name}")
        expected = source_data[selector]
        if not _arrays_equal(expected, archive_data):
            raise AcceptanceError(
                f"Archived values differ from v6: {video}/{archive_name}/{relative}"
            )


def _strict_label_stats(handle: h5py.File, *, video: str) -> dict[str, int]:
    required = (
        "point_cloud/frame_order",
        "point_cloud/pixel_xy",
        "annotation/point_label",
        "annotation/valid_mask",
        "frame_annotation/frame_order",
        "frame_annotation/bbox_local_xyxy",
    )
    missing = [name for name in required if name not in handle]
    if missing:
        raise AcceptanceError(f"v7 lacks label-policy datasets {missing}: {video}")
    frame_orders = handle["point_cloud/frame_order"][:].astype(np.int64)
    pixel_xy = handle["point_cloud/pixel_xy"][:].astype(np.float64)
    labels = handle["annotation/point_label"][:].astype(np.int8)
    valid = handle["annotation/valid_mask"][:].astype(bool)
    bbox_orders = handle["frame_annotation/frame_order"][:].astype(np.int64)
    bboxes = handle["frame_annotation/bbox_local_xyxy"][:].astype(np.float64)
    if frame_orders.shape != labels.shape or valid.shape != labels.shape:
        raise AcceptanceError(f"Point label arrays are unaligned: {video}")
    if pixel_xy.shape != (labels.size, 2):
        raise AcceptanceError(f"pixel_xy shape is invalid: {video}")
    if bboxes.shape != (bbox_orders.size, 4):
        raise AcceptanceError(f"BBox arrays are unaligned: {video}")
    if not np.array_equal(valid, labels != LABEL_IGNORE):
        raise AcceptanceError(f"valid_mask differs from point_label: {video}")
    unknown = set(int(value) for value in np.unique(labels)) - {
        LABEL_IGNORE,
        LABEL_BACKGROUND,
        LABEL_POSITIVE,
    }
    if unknown:
        raise AcceptanceError(f"Unsupported labels {sorted(unknown)}: {video}")

    rounded_xy = np.rint(pixel_xy).astype(np.int64)
    bbox_frame_set = set(int(value) for value in np.unique(bbox_orders))
    no_bbox_ignore = 0
    stray_ignore = 0
    background_inside_bbox = 0
    for order in np.unique(frame_orders):
        point_indices = np.flatnonzero(frame_orders == order)
        frame_labels = labels[point_indices]
        frame_boxes = bboxes[bbox_orders == order]
        if int(order) not in bbox_frame_set:
            no_bbox_ignore += int(np.sum(frame_labels == LABEL_IGNORE))
            continue
        inside = np.zeros(point_indices.size, dtype=bool)
        xy = rounded_xy[point_indices]
        for bbox in frame_boxes:
            if not np.all(np.isfinite(bbox)):
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            left = int(np.floor(max(0.0, x1)))
            top = int(np.floor(max(0.0, y1)))
            right = int(np.ceil(x2))
            bottom = int(np.ceil(y2))
            if right <= left or bottom <= top:
                continue
            inside |= (
                (xy[:, 0] >= left)
                & (xy[:, 0] <= right)
                & (xy[:, 1] >= top)
                & (xy[:, 1] <= bottom)
            )
        stray_ignore += int(np.sum((frame_labels == LABEL_IGNORE) & ~inside))
        background_inside_bbox += int(
            np.sum((frame_labels == LABEL_BACKGROUND) & inside)
        )
    if no_bbox_ignore:
        raise AcceptanceError(f"no-BBox ignore points={no_bbox_ignore}: {video}")
    if stray_ignore:
        raise AcceptanceError(f"stray ignore points={stray_ignore}: {video}")
    if background_inside_bbox:
        raise AcceptanceError(
            f"BBox-inside background points={background_inside_bbox}: {video}"
        )
    outside_cvat = 0
    if "manual_review_fullvideo_frames" in handle:
        frames = handle["manual_review_fullvideo_frames"]
        if "positive_outside_cvat_mask_points" in frames:
            outside_cvat = int(
                np.sum(frames["positive_outside_cvat_mask_points"][:].astype(np.int64))
            )
            if outside_cvat:
                raise AcceptanceError(
                    f"Positive points outside CVAT mask={outside_cvat}: {video}"
                )
    return {
        "points": int(labels.size),
        "positive_points": int(np.sum(labels == LABEL_POSITIVE)),
        "ignore_points": int(np.sum(labels == LABEL_IGNORE)),
        "background_points": int(np.sum(labels == LABEL_BACKGROUND)),
        "no_bbox_ignore_points": no_bbox_ignore,
        "stray_ignore_points": stray_ignore,
        "bbox_inside_background_points": background_inside_bbox,
        "positive_outside_cvat_mask_points": outside_cvat,
    }


def _validate_fixed_summaries(
    *,
    build_summary_path: Path,
    build_rows_path: Path,
    manifest_validation_path: Path,
    exclusion_summary_path: Path,
    manifest_path: Path,
    exclusion_path: Path,
    invalidation_path: Path,
    invalidations: Sequence[CropQualityInvalidation],
    expected_videos: int,
    expected_source_videos: int,
    expected_excluded: int,
    expected_ignore: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    build = _read_json(build_summary_path)
    rows = _read_csv(build_rows_path)
    validation = _read_json(manifest_validation_path)
    exclusion = _read_json(exclusion_summary_path)
    fingerprint = invalidation_fingerprint(invalidations)
    expected_build = {
        "status": "ok",
        "source_teacher_version": SOURCE_TOKEN,
        "teacher_version": OUTPUT_TOKEN,
        "videos": expected_videos,
        "source_videos": expected_source_videos,
        "affected_videos": len({item.video_name for item in invalidations}),
        "invalidated_frames": len(invalidations),
        "removed_bbox_rows": sum(item.expected_saved_bbox_rows for item in invalidations),
        "removed_positive_points": 0,
        "removed_ignore_points": expected_ignore,
        "train_manifest_sha256": file_sha256(manifest_path),
        "exclusion_manifest_sha256": file_sha256(exclusion_path),
        "invalidation_manifest_sha256": file_sha256(invalidation_path),
        "invalidation_fingerprint": fingerprint,
        "failure_rows": 0,
    }
    for name, expected in expected_build.items():
        if build.get(name) != expected:
            raise AcceptanceError(
                f"Build summary mismatch for {name}: {build.get(name)!r} != {expected!r}"
            )
    if len(rows) != expected_videos:
        raise AcceptanceError(f"Build summary CSV rows={len(rows)} != {expected_videos}")
    if {row.get("video_name", "") for row in rows} != {
        row.video_name
        for row in load_stage4_sweep_manifest(manifest_path)
        if row.enabled
    }:
        raise AcceptanceError("Build summary CSV video inventory differs from manifest")
    csv_totals = {
        "invalidated_frames": sum(int(row["invalidated_frames"]) for row in rows),
        "removed_bbox_rows": sum(int(row["removed_bbox_rows"]) for row in rows),
        "removed_positive_points": sum(int(row["removed_positive_points"]) for row in rows),
        "removed_ignore_points": sum(int(row["removed_ignore_points"]) for row in rows),
    }
    for name in csv_totals:
        if csv_totals[name] != expected_build[name]:
            raise AcceptanceError(f"Build CSV total mismatch for {name}")
    if validation.get("status") != "ok" or not validation.get("input_files_unchanged"):
        raise AcceptanceError("Manifest-validation summary is not an accepted read-only run")
    expected_validation = {
        "video_exclusions_sha256": file_sha256(exclusion_path),
        "crop_invalidations_sha256": file_sha256(invalidation_path),
        "crop_invalidation_fingerprint": fingerprint,
        "invalidated_frames": len(invalidations),
        "expected_stray_points_to_remove": expected_ignore,
        "h5_files_written": 0,
    }
    for name, expected in expected_validation.items():
        if validation.get(name) != expected:
            raise AcceptanceError(f"Manifest-validation summary mismatch for {name}")
    expected_exclusion = {
        "status": "ok",
        "output_manifest_sha256": file_sha256(manifest_path),
        "exclusions_sha256": file_sha256(exclusion_path),
        "rows": expected_videos + expected_excluded,
        "enabled": expected_videos,
        "disabled": expected_excluded,
    }
    for name, expected in expected_exclusion.items():
        if exclusion.get(name) != expected:
            raise AcceptanceError(f"Exclusion summary mismatch for {name}")
    return build, rows


def _validate_affected_video(
    source: h5py.File,
    final: h5py.File,
    *,
    video: str,
    items: Sequence[CropQualityInvalidation],
    invalidation_manifest_sha256: str,
) -> dict[str, int]:
    source_paths = _dataset_paths(source)
    skip_prefixes = (
        "annotation/",
        "frame_annotation/",
        "manual_review_fullvideo/",
        "manual_review_fullvideo_frames/",
    )
    for path in sorted(source_paths):
        if not path.startswith(skip_prefixes):
            _assert_dataset_equal(source, final, path, video=video)
    source_orders = source["point_cloud/frame_order"][:].astype(np.int64)
    source_labels = source["annotation/point_label"][:].astype(np.int8)
    final_orders = final["point_cloud/frame_order"][:].astype(np.int64)
    final_labels = final["annotation/point_label"][:].astype(np.int8)
    if not np.array_equal(source_orders, final_orders):
        raise AcceptanceError(f"Point frame order changed: {video}")
    target_orders = {item.frame_order for item in items}
    target_points = np.isin(source_orders, list(target_orders))
    if not np.array_equal(source_labels[~target_points], final_labels[~target_points]):
        raise AcceptanceError(f"Non-target point labels changed: {video}")
    if not np.all(final_labels[target_points] == LABEL_BACKGROUND):
        raise AcceptanceError(f"Crop-invalidated points are not all background: {video}")

    source_frame = source["frame_annotation"]
    source_frame_orders = source_frame["frame_order"][:].astype(np.int64)
    source_frame_indices = source_frame["frame_index"][:].astype(np.int64)
    frame_remove = np.zeros(source_frame_orders.size, dtype=bool)
    for item in items:
        frame_remove |= (
            (source_frame_orders == item.frame_order)
            & (source_frame_indices == item.frame_index)
        )
    _assert_filtered_group(
        source, final, "frame_annotation", frame_remove, video=video
    )
    group = final[GROUP_NAME]
    _assert_archive_matches(
        source_frame,
        group["removed_frame_annotation"],
        frame_remove,
        video=video,
        archive_name="removed_frame_annotation",
    )

    if "manual_review_fullvideo" in source:
        manual = source["manual_review_fullvideo"]
        manual_orders = manual["frame_order"][:].astype(np.int64)
        manual_indices = manual["frame_index"][:].astype(np.int64)
        manual_remove = np.zeros(manual_orders.size, dtype=bool)
        for item in items:
            manual_remove |= (
                (manual_orders == item.frame_order)
                & (manual_indices == item.frame_index)
            )
        _assert_filtered_group(
            source, final, "manual_review_fullvideo", manual_remove, video=video
        )
        _assert_archive_matches(
            manual,
            group["removed_manual_review_fullvideo"],
            manual_remove,
            video=video,
            archive_name="removed_manual_review_fullvideo",
        )
    elif "manual_review_fullvideo" in final:
        raise AcceptanceError(f"v7 unexpectedly added manual-review BBox group: {video}")

    if "manual_review_fullvideo_frames" not in source or (
        "manual_review_fullvideo_frames" not in final
    ):
        raise AcceptanceError(f"Affected video lacks CVAT frame provenance: {video}")
    source_frames = source["manual_review_fullvideo_frames"]
    final_frames = final["manual_review_fullvideo_frames"]
    source_provenance_paths = _dataset_paths(source_frames)
    if not source_provenance_paths.issubset(_dataset_paths(final_frames)):
        raise AcceptanceError(f"v7 lost CVAT frame provenance datasets: {video}")
    orders = final_frames["frame_order"][:].astype(np.int64)
    indices = final_frames["frame_index"][:].astype(np.int64)
    authorities = [_decode(value) for value in final_frames["label_authority"][:]]
    actions = [_decode(value) for value in final_frames["crop_invalidation_action"][:]]
    reasons = [_decode(value) for value in final_frames["crop_invalidation_reason_code"][:]]
    hashes = [_decode(value) for value in final_frames["crop_invalidation_manifest_sha256"][:]]
    archived_frames = group["source_manual_review_fullvideo_frames"]
    source_provenance_selector = np.zeros(len(source_frames["frame_order"]), dtype=bool)
    source_provenance_orders = source_frames["frame_order"][:].astype(np.int64)
    source_provenance_indices = source_frames["frame_index"][:].astype(np.int64)
    for item in items:
        source_provenance_selector |= (
            (source_provenance_orders == item.frame_order)
            & (source_provenance_indices == item.frame_index)
        )
    mutable_frame_fields = {
        "label_authority",
        "positive_points",
        "ignore_points",
        "background_points",
        "positive_outside_cvat_mask_points",
    }
    for relative in sorted(source_provenance_paths):
        source_data = _read_dataset(source_frames[relative])
        final_data = _read_dataset(final_frames[relative])
        if relative in mutable_frame_fields:
            if not _arrays_equal(
                source_data[~source_provenance_selector],
                final_data[~source_provenance_selector],
            ):
                raise AcceptanceError(
                    f"Non-target CVAT frame provenance changed: {video}/{relative}"
                )
        elif not _arrays_equal(source_data, final_data):
            raise AcceptanceError(
                f"CVAT frame provenance changed unexpectedly: {video}/{relative}"
            )
    _assert_archive_matches(
        source_frames,
        archived_frames,
        source_provenance_selector,
        video=video,
        archive_name="source_manual_review_fullvideo_frames",
    )
    total_before_ignore = 0
    total_removed_bbox = 0
    for item in items:
        selector = (orders == item.frame_order) & (indices == item.frame_index)
        if int(selector.sum()) != 1:
            raise AcceptanceError(f"Affected provenance row count mismatch: {item.frame_stem}")
        row = int(np.flatnonzero(selector)[0])
        if (
            authorities[row] != FRAME_AUTHORITY
            or actions[row] != item.action
            or reasons[row] != item.reason_code
            or hashes[row] != invalidation_manifest_sha256
        ):
            raise AcceptanceError(f"Affected frame authority mismatch: {item.frame_stem}")
        for name, expected in (
            ("positive_points", 0),
            ("ignore_points", 0),
            ("background_points", int(np.sum(final_orders == item.frame_order))),
        ):
            if name in final_frames and int(final_frames[name][row]) != expected:
                raise AcceptanceError(f"Affected provenance {name} mismatch: {item.frame_stem}")
        archive_selector = (
            (archived_frames["frame_order"][:].astype(np.int64) == item.frame_order)
            & (archived_frames["frame_index"][:].astype(np.int64) == item.frame_index)
        )
        if int(archive_selector.sum()) != 1:
            raise AcceptanceError(f"Archived CVAT frame row mismatch: {item.frame_stem}")
        total_before_ignore += item.expected_stray_ignore_points
        total_removed_bbox += item.expected_saved_bbox_rows
    if int(np.sum(group["before_ignore_points"][:])) != total_before_ignore:
        raise AcceptanceError(f"Archived ignore total mismatch: {video}")
    if int(np.sum(group["before_positive_points"][:])) != 0:
        raise AcceptanceError(f"Unexpected archived positive points: {video}")
    if int(np.sum(group["removed_frame_annotation_rows"][:])) != total_removed_bbox:
        raise AcceptanceError(f"Archived BBox total mismatch: {video}")
    if len(group["removed_frame_annotation/frame_order"]) != total_removed_bbox:
        raise AcceptanceError(f"Removed BBox archive length mismatch: {video}")
    return {
        "invalidated_frames": len(items),
        "removed_bbox_rows": total_removed_bbox,
        "removed_ignore_points": total_before_ignore,
    }


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    exclusion_path = Path(args.exclusion_manifest).resolve()
    invalidation_path = Path(args.invalidation_manifest).resolve()
    source_root = Path(args.source_annotated_root).resolve()
    annotated_root = Path(args.annotated_root).resolve()
    collected_root = Path(args.collected_root).resolve()
    build_summary_path = Path(args.build_summary).resolve()
    build_rows_path = Path(args.build_summary_csv).resolve()
    manifest_validation_path = Path(args.manifest_validation_summary).resolve()
    exclusion_summary_path = Path(args.exclusion_summary).resolve()
    output_json = Path(args.output_json).resolve()
    for path, description in (
        (manifest_path, "train manifest"),
        (exclusion_path, "exclusion manifest"),
        (invalidation_path, "crop invalidation manifest"),
        (build_summary_path, "v7 build summary"),
        (build_rows_path, "v7 build summary CSV"),
        (manifest_validation_path, "manifest-validation summary"),
        (exclusion_summary_path, "exclusion summary"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    for path, description in (
        (source_root, "v6 annotated root"),
        (annotated_root, "v7 annotated root"),
        (collected_root, "v7 collected root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} not found: {path}")

    manifest_rows = load_stage4_sweep_manifest(manifest_path)
    exclusions = load_exclusions(exclusion_path)
    excluded = {row.video_name for row in exclusions}
    disabled = {row.video_name for row in manifest_rows if not row.enabled}
    if disabled != excluded:
        raise AcceptanceError(
            f"Disabled/excluded inventory differs: {sorted(disabled)} != {sorted(excluded)}"
        )
    enabled = sorted(row.video_name for row in manifest_rows if row.enabled)
    if len(enabled) != args.expected_videos:
        raise AcceptanceError(f"Enabled videos={len(enabled)} != {args.expected_videos}")
    if len(excluded) != args.expected_excluded_videos:
        raise AcceptanceError(
            f"Excluded videos={len(excluded)} != {args.expected_excluded_videos}"
        )
    invalidations = load_crop_quality_invalidations(invalidation_path)
    by_video: dict[str, list[CropQualityInvalidation]] = {}
    for item in invalidations:
        by_video.setdefault(item.video_name, []).append(item)
    if len(invalidations) != args.expected_invalidated_frames:
        raise AcceptanceError(
            f"Invalidated frames={len(invalidations)} != {args.expected_invalidated_frames}"
        )
    if set(by_video) - set(enabled):
        raise AcceptanceError("Crop invalidation references an excluded/unknown video")

    _validate_fixed_summaries(
        build_summary_path=build_summary_path,
        build_rows_path=build_rows_path,
        manifest_validation_path=manifest_validation_path,
        exclusion_summary_path=exclusion_summary_path,
        manifest_path=manifest_path,
        exclusion_path=exclusion_path,
        invalidation_path=invalidation_path,
        invalidations=invalidations,
        expected_videos=args.expected_videos,
        expected_source_videos=args.expected_source_videos,
        expected_excluded=args.expected_excluded_videos,
        expected_ignore=args.expected_removed_ignore_points,
    )

    source_files = sorted(source_root.rglob(f"*{SOURCE_TOKEN}.h5"))
    annotated_files = sorted(annotated_root.rglob(f"*{OUTPUT_TOKEN}.h5"))
    collected_files = sorted(collected_root.glob(f"*{OUTPUT_TOKEN}.h5"))
    expected_inventory = {
        "v6 source": (len(source_files), args.expected_source_videos),
        "v7 annotated": (len(annotated_files), args.expected_videos),
        "v7 collected": (len(collected_files), args.expected_videos),
    }
    for name, (actual, expected) in expected_inventory.items():
        if actual != expected:
            raise AcceptanceError(f"{name} files={actual} != {expected}")
    for video in excluded:
        if (annotated_root / video).exists():
            raise AcceptanceError(f"Excluded video leaked into annotated v7: {video}")
        if any(path.name.startswith(f"{video}_") for path in collected_files):
            raise AcceptanceError(f"Excluded video leaked into collected v7: {video}")

    manifest_sha = file_sha256(invalidation_path)
    fingerprint = invalidation_fingerprint(invalidations)
    totals: Counter[str] = Counter()
    source_hashes: dict[str, str] = {}
    annotated_hashes: dict[str, str] = {}
    semantic_identity_videos = 0
    affected_videos = 0
    print("Stage 4 teacher v7 crop-quality acceptance check (read-only)")
    print(f"  source v6 root  : {source_root}")
    print(f"  annotated root  : {annotated_root}")
    print(f"  collected root  : {collected_root}")
    print(f"  videos          : {len(enabled)}")
    print(f"  excluded videos : {len(excluded)}")

    for index, video in enumerate(enabled, start=1):
        source_h5 = _single_annotated(source_root, video, SOURCE_TOKEN)
        final_h5 = _single_annotated(annotated_root, video, OUTPUT_TOKEN)
        collected_h5 = _single_collected(collected_root, video)
        source_sha = file_sha256(source_h5)
        final_sha = file_sha256(final_h5)
        if file_sha256(collected_h5) != final_sha:
            raise AcceptanceError(f"Annotated/collected byte mismatch: {video}")
        source_hashes[video] = source_sha
        annotated_hashes[video] = final_sha
        items = sorted(by_video.get(video, []), key=lambda item: item.frame_key)
        with h5py.File(source_h5, "r") as source, h5py.File(final_h5, "r") as final:
            if _decode(source.attrs.get("contour_teacher_schema", "")) != SOURCE_TOKEN:
                raise AcceptanceError(f"Wrong v6 source schema: {video}")
            if _decode(final.attrs.get("contour_teacher_schema", "")) != OUTPUT_TOKEN:
                raise AcceptanceError(f"Wrong v7 schema: {video}")
            if GROUP_NAME not in final:
                raise AcceptanceError(f"v7 lacks {GROUP_NAME}: {video}")
            provenance = final[GROUP_NAME]
            expected_attrs = {
                "source_h5_sha256": source_sha,
                "manifest_sha256": manifest_sha,
                "manifest_fingerprint": fingerprint,
                "source_teacher_version": SOURCE_TOKEN,
                "teacher_version": OUTPUT_TOKEN,
            }
            for name, expected in expected_attrs.items():
                if _decode(provenance.attrs.get(name, "")) != expected:
                    raise AcceptanceError(f"v7 provenance mismatch {name}: {video}")
            saved_keys = {
                (_decode(name), int(order), int(frame_index))
                for name, order, frame_index in zip(
                    provenance["video_name"][:],
                    provenance["frame_order"][:],
                    provenance["frame_index"][:],
                    strict=True,
                )
            }
            if saved_keys != {item.frame_key for item in items}:
                raise AcceptanceError(f"v7 invalidation inventory mismatch: {video}")
            if items:
                change = _validate_affected_video(
                    source,
                    final,
                    video=video,
                    items=items,
                    invalidation_manifest_sha256=manifest_sha,
                )
                totals.update(change)
                affected_videos += 1
            else:
                _assert_unaffected_semantic_identity(source, final, video=video)
                semantic_identity_videos += 1
            stats = _strict_label_stats(final, video=video)
            totals.update(stats)
        print(
            f"  [{index}/{len(enabled)}] [OK] {video}: "
            f"mode={'invalidated' if items else 'preserved'}, "
            f"points={stats['points']}, positive={stats['positive_points']}, "
            f"ignore={stats['ignore_points']}"
        )

    expected_changes = {
        "invalidated_frames": args.expected_invalidated_frames,
        "removed_bbox_rows": args.expected_removed_bbox_rows,
        "removed_ignore_points": args.expected_removed_ignore_points,
    }
    for name, expected in expected_changes.items():
        if totals[name] != expected:
            raise AcceptanceError(f"Accepted total {name}={totals[name]} != {expected}")
    if affected_videos != len(by_video):
        raise AcceptanceError("Affected-video count differs from invalidation manifest")
    if semantic_identity_videos != args.expected_videos - affected_videos:
        raise AcceptanceError("Unaffected semantic-identity count mismatch")
    for video, checksum in source_hashes.items():
        source_h5 = _single_annotated(source_root, video, SOURCE_TOKEN)
        if file_sha256(source_h5) != checksum:
            raise AcceptanceError(f"Source H5 changed during acceptance: {video}")
    for video, checksum in annotated_hashes.items():
        final_h5 = _single_annotated(annotated_root, video, OUTPUT_TOKEN)
        collected_h5 = _single_collected(collected_root, video)
        if file_sha256(final_h5) != checksum or file_sha256(collected_h5) != checksum:
            raise AcceptanceError(f"v7 H5 changed during acceptance: {video}")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "source_teacher_version": SOURCE_TOKEN,
        "teacher_version": OUTPUT_TOKEN,
        "source_videos": len(source_files),
        "videos": len(enabled),
        "excluded_videos": sorted(excluded),
        "affected_videos": affected_videos,
        "unaffected_semantic_identity_videos": semantic_identity_videos,
        "invalidated_frames": totals["invalidated_frames"],
        "removed_bbox_rows": totals["removed_bbox_rows"],
        "removed_ignore_points": totals["removed_ignore_points"],
        "points": totals["points"],
        "positive_points": totals["positive_points"],
        "ignore_points": totals["ignore_points"],
        "background_points": totals["background_points"],
        "stray_ignore_points": totals["stray_ignore_points"],
        "no_bbox_ignore_points": totals["no_bbox_ignore_points"],
        "bbox_inside_background_points": totals["bbox_inside_background_points"],
        "positive_outside_cvat_mask_points": totals[
            "positive_outside_cvat_mask_points"
        ],
        "annotated_collected_byte_identical": True,
        "source_inputs_unchanged": True,
        "failure_rows": 0,
        "files_written": 1,
    }
    report_status = _write_or_verify_json(
        output_json, summary, overwrite=bool(args.overwrite)
    )
    print("\nStage 4 teacher v7 acceptance summary")
    for name in (
        "source_videos",
        "videos",
        "affected_videos",
        "unaffected_semantic_identity_videos",
        "invalidated_frames",
        "removed_bbox_rows",
        "removed_ignore_points",
        "stray_ignore_points",
        "no_bbox_ignore_points",
        "bbox_inside_background_points",
        "positive_outside_cvat_mask_points",
    ):
        print(f"  {name:38s}: {summary[name]}")
    print(f"  acceptance_report_status              : {report_status}")
    print(f"  acceptance_report                     : {output_json}")
    print("Stage 4 teacher v7 crop-quality acceptance checks passed.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    source_root = DATASET_ROOT / SOURCE_RUN_NAME
    output_root = DATASET_ROOT / OUTPUT_RUN_NAME
    step2_root = source_root / "crop_quality_manifest_step2"
    parser = argparse.ArgumentParser(
        description="Read-only acceptance check for the Stage 4 teacher-v7 crop-quality build."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_ROOT / "train_manifest_cropclean_v2.csv",
    )
    parser.add_argument(
        "--exclusion_manifest",
        type=Path,
        default=REPO_ROOT / "pseudo3d/analysis/configs/stage4_video_exclusions_v2.csv",
    )
    parser.add_argument(
        "--invalidation_manifest",
        type=Path,
        default=REPO_ROOT
        / "pseudo3d/analysis/configs/stage4_crop_quality_invalidations_v1.csv",
    )
    parser.add_argument(
        "--source_annotated_root", type=Path, default=source_root / "annotated"
    )
    parser.add_argument("--annotated_root", type=Path, default=output_root / "annotated")
    parser.add_argument("--collected_root", type=Path, default=output_root / "collected")
    parser.add_argument(
        "--build_summary",
        type=Path,
        default=output_root / "crop_quality_invalidation_summary.json",
    )
    parser.add_argument(
        "--build_summary_csv",
        type=Path,
        default=output_root / "crop_quality_invalidation_summary.csv",
    )
    parser.add_argument(
        "--manifest_validation_summary",
        type=Path,
        default=step2_root / "manifest_validation_summary.json",
    )
    parser.add_argument(
        "--exclusion_summary",
        type=Path,
        default=step2_root / "excluded_manifest_summary.json",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=output_root / "crop_quality_acceptance_step5.json",
    )
    parser.add_argument("--expected_videos", type=int, default=180)
    parser.add_argument("--expected_source_videos", type=int, default=181)
    parser.add_argument("--expected_excluded_videos", type=int, default=2)
    parser.add_argument("--expected_invalidated_frames", type=int, default=1)
    parser.add_argument("--expected_removed_bbox_rows", type=int, default=1)
    parser.add_argument("--expected_removed_ignore_points", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    try:
        run_acceptance(build_parser().parse_args())
    except Exception as exc:
        raise SystemExit(
            "Stage 4 teacher v7 crop-quality acceptance failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
