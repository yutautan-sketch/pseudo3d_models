from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_contour_teacher import mask_shape_metrics
from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LocalBBox,
    summarize_labels_by_source,
)
from pseudo3d.annotation.import_cvat_segmentation_mask_corrections import (
    _review_package_contract,
)
from pseudo3d.annotation.stage4_manual_review import (
    LABEL_BACKGROUND,
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
    ManualReviewError,
    bbox_bounds,
    bbox_union_mask,
    load_manual_review_config,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    read_cvat_segmentation_class_masks,
)


SCHEMA_VERSION = 1
AUTHORITATIVE_SCHEMA_VERSION = 2
SOURCE_TOKEN = "bboxrank_v3_refined_auto_v1"
OUTPUT_TOKEN = "bboxrank_v4_manual_fullvideo_v1"
AUTHORITATIVE_OUTPUT_TOKEN = "bboxrank_v5_cvat_authoritative_v1"
MANUAL_SOURCE = "manual_cvat_fullvideo_v1"
MANUAL_EMPTY_SOURCE = "manual_cvat_fullvideo_empty_v1"
AUTHORITATIVE_MANUAL_SOURCE = "manual_cvat_fullvideo_snapshot_authoritative_v1"
AUTHORITATIVE_EMPTY_SOURCE = (
    "manual_cvat_fullvideo_snapshot_authoritative_empty_v1"
)
AUTHORITATIVE_LABEL_AUTHORITY = "cvat_snapshot"
CVAT_MASK_POSITIVE = "positive"
CVAT_MASK_EMPTY = "empty"
CVAT_MASK_OMITTED_AS_EMPTY = "omitted_as_empty"


class FullVideoImportError(ManualReviewError):
    """Raised when the final full-video review contract is violated."""


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false", ""}:
        return False
    raise FullVideoImportError(f"{name} must be boolean: {value!r}")


def _integer(value: Any, name: str) -> int:
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise FullVideoImportError(f"{name} must be an integer: {value!r}") from exc


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = str(value).split("|")
    if len(parts) != 4:
        raise FullVideoImportError(f"Invalid bbox_local_xyxy: {value!r}")
    result = tuple(float(item) for item in parts)
    if not all(math.isfinite(item) for item in result):
        raise FullVideoImportError(f"Non-finite bbox_local_xyxy: {value!r}")
    return result  # type: ignore[return-value]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FullVideoImportError(f"JSON root must be an object: {path}")
    return value


def _safe_video_name(value: Any) -> str:
    name = str(value).strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise FullVideoImportError(f"Unsafe video_name: {value!r}")
    return name


def _validate_zip(path: Path, expected_sha256: str, kind: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing/empty {kind}: {path}")
    if file_sha256(path) != expected_sha256:
        raise FullVideoImportError(f"{kind} checksum mismatch: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if archive.testzip() is not None:
                raise FullVideoImportError(f"Corrupt {kind}: {path}")
    except zipfile.BadZipFile as exc:
        raise FullVideoImportError(f"Invalid {kind} ZIP: {path}") from exc


def _snapshot_contract(snapshot_root: Path, expected_tasks: int) -> dict[str, dict[str, Any]]:
    summary = _read_json(snapshot_root / "export_summary.json")
    if str(summary.get("status")) != "complete":
        raise FullVideoImportError("CVAT snapshot export_summary status is not complete")
    if int(summary.get("completed_tasks", -1)) != expected_tasks:
        raise FullVideoImportError("CVAT snapshot completed task count mismatch")
    rows = read_csv_rows(snapshot_root / "export_manifest.csv")
    if len(rows) != expected_tasks:
        raise FullVideoImportError(
            f"CVAT snapshot manifest count mismatch: {len(rows)} != {expected_tasks}"
        )
    result: dict[str, dict[str, Any]] = {}
    task_ids: set[int] = set()
    for row in rows:
        video = _safe_video_name(row.get("video_name"))
        task_id = _integer(row.get("task_id"), "task_id")
        if video in result or task_id in task_ids:
            raise FullVideoImportError(f"Duplicate snapshot video/task: {video}/{task_id}")
        if str(row.get("status")) != "complete":
            raise FullVideoImportError(f"Snapshot task is incomplete: {video}")
        annotation = (
            snapshot_root
            / "annotations"
            / video
            / f"{video}__task{task_id}__reviewed_segmentation_mask_1_1.zip"
        )
        backup = (
            snapshot_root
            / "task_backups"
            / video
            / f"{video}__task{task_id}__post_review_backup.zip"
        )
        _validate_zip(annotation, str(row.get("annotation_sha256", "")), "annotation")
        _validate_zip(backup, str(row.get("backup_sha256", "")), "task backup")
        normalized = dict(row)
        normalized.update(
            video_name=video,
            task_id=task_id,
            annotation_path=annotation,
            backup_path=backup,
        )
        result[video] = normalized
        task_ids.add(task_id)
    return result


def _task_map_contract(task_map_csv: Path, snapshots: Mapping[str, Mapping[str, Any]]) -> None:
    rows = read_csv_rows(task_map_csv)
    if len(rows) != len(snapshots):
        raise FullVideoImportError("task_map.csv row count differs from snapshot")
    for row in rows:
        video = _safe_video_name(row.get("video_name"))
        snapshot = snapshots.get(video)
        if snapshot is None:
            raise FullVideoImportError(f"task_map video is absent from snapshot: {video}")
        if _integer(row.get("task_id"), "task_id") != int(snapshot["task_id"]):
            raise FullVideoImportError(f"task_map task ID mismatch: {video}")
        if str(row.get("task_name", "")) != str(snapshot.get("task_name", "")):
            raise FullVideoImportError(f"task_map task name mismatch: {video}")


def _resolve_source(source_root: Path, video_name: str) -> Path:
    matches = sorted((source_root / video_name).glob(f"*{SOURCE_TOKEN}.h5"))
    if len(matches) != 1:
        raise FullVideoImportError(
            f"Expected one v3 source H5 for {video_name}, found {len(matches)}"
        )
    return matches[0].resolve()


def _output_name(source: Path, output_token: str = OUTPUT_TOKEN) -> str:
    if SOURCE_TOKEN not in source.name:
        raise FullVideoImportError(f"Source filename lacks {SOURCE_TOKEN}: {source.name}")
    return source.name.replace(SOURCE_TOKEN, output_token)


def _replace_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    old = group[name]
    kwargs: dict[str, Any] = {}
    if old.compression is not None:
        kwargs["compression"] = old.compression
        kwargs["compression_opts"] = old.compression_opts
    if old.shuffle:
        kwargs["shuffle"] = True
    if old.fletcher32:
        kwargs["fletcher32"] = True
    del group[name]
    group.create_dataset(name, data=values, **kwargs)


def _replace_text_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype("utf-8")))


def _contour_area(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return float(sum(cv2.contourArea(contour) for contour in contours))


def _point_count(mask: np.ndarray, xy: np.ndarray) -> int:
    coords = np.rint(xy).astype(np.int64)
    if coords.size == 0:
        return 0
    height, width = mask.shape
    if (
        np.any(coords[:, 0] < 0)
        or np.any(coords[:, 0] >= width)
        or np.any(coords[:, 1] < 0)
        or np.any(coords[:, 1] >= height)
    ):
        raise FullVideoImportError("point_cloud/pixel_xy lies outside review image")
    return int(np.sum(mask[coords[:, 1], coords[:, 0]]))


def _frame_annotation_row(
    frame_group: h5py.Group, frame_order: int, frame_index: int, bbox_index: int
) -> int:
    matches = np.flatnonzero(
        (frame_group["frame_order"][:].astype(np.int64) == frame_order)
        & (frame_group["frame_index"][:].astype(np.int64) == frame_index)
        & (frame_group["bbox_index"][:].astype(np.int64) == bbox_index)
    )
    if matches.size != 1:
        raise FullVideoImportError(
            "Expected one frame_annotation row for "
            f"frame={frame_order}/{frame_index}, bbox={bbox_index}; got {matches.size}"
        )
    return int(matches[0])


def _apply_selected_frame(
    *,
    labels: np.ndarray,
    frame_orders: np.ndarray,
    pixel_xy: np.ndarray,
    frame_order: int,
    corrected_mask: np.ndarray,
    target_bboxes: Sequence[tuple[float, float, float, float]],
    allow_outside_single_bbox: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    corrected = np.asarray(corrected_mask, dtype=bool)
    allowed = bbox_union_mask(corrected.shape, target_bboxes)
    outside = corrected & ~allowed
    if np.any(outside):
        if allow_outside_single_bbox and len(target_bboxes) == 1:
            pass
        else:
            y, x = np.nonzero(outside)
            raise FullVideoImportError(
                f"Corrected mask has {int(outside.sum())} pixel(s) outside selected BBox union; "
                f"x_range={int(x.min())}..{int(x.max())}, y_range={int(y.min())}..{int(y.max())}"
            )
    result = labels.copy()
    selector = frame_orders.astype(np.int64) == frame_order
    xy = np.rint(pixel_xy[selector]).astype(np.int64)
    if xy.size:
        height, width = corrected.shape
        if (
            np.any(xy[:, 0] < 0)
            or np.any(xy[:, 0] >= width)
            or np.any(xy[:, 1] < 0)
            or np.any(xy[:, 1] >= height)
        ):
            raise FullVideoImportError("point_cloud/pixel_xy lies outside review image")
        inside = allowed[xy[:, 1], xy[:, 0]]
        positive = corrected[xy[:, 1], xy[:, 0]]
        frame_labels = result[selector].copy()
        frame_labels[inside] = LABEL_IGNORE
        frame_labels[positive] = LABEL_FEMUR_CANDIDATE
        result[selector] = frame_labels
    frame_labels = result[selector]
    return result, {
        "frame_points": int(selector.sum()),
        "positive_points": int(np.sum(frame_labels == LABEL_FEMUR_CANDIDATE)),
        "ignore_points": int(np.sum(frame_labels == LABEL_IGNORE)),
        "background_points": int(np.sum(frame_labels == LABEL_BACKGROUND)),
        "corrected_positive_pixels": int(corrected.sum()),
        "outside_bbox_positive_pixels": int(outside.sum()),
    }


def apply_authoritative_frame_mask(
    *,
    labels: np.ndarray,
    frame_orders: np.ndarray,
    pixel_xy: np.ndarray,
    frame_order: int,
    corrected_mask: np.ndarray,
    frame_bboxes: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Rebuild one Task frame solely from its CVAT mask and saved BBoxes.

    Source labels are deliberately ignored for points on ``frame_order``.  The
    CVAT mask is the only positive authority, while the union of every saved
    BBox on the frame defines the non-positive ignore region.  Human positives
    outside that union remain positive.
    """
    result = np.asarray(labels, dtype=np.int8).copy()
    orders = np.asarray(frame_orders, dtype=np.int64)
    if result.ndim != 1 or orders.shape != result.shape:
        raise FullVideoImportError("point labels/frame orders must be aligned 1-D arrays")
    points = np.asarray(pixel_xy)
    if points.shape != (result.size, 2):
        raise FullVideoImportError("point_cloud/pixel_xy must have shape [num_points, 2]")

    corrected = np.asarray(corrected_mask, dtype=bool)
    if corrected.ndim != 2:
        raise FullVideoImportError("Authoritative CVAT mask must be 2-D")
    selector = orders == int(frame_order)
    xy = np.rint(points[selector]).astype(np.int64)
    height, width = corrected.shape
    if xy.size and (
        np.any(xy[:, 0] < 0)
        or np.any(xy[:, 0] >= width)
        or np.any(xy[:, 1] < 0)
        or np.any(xy[:, 1] >= height)
    ):
        raise FullVideoImportError("point_cloud/pixel_xy lies outside review image")

    bbox_union = bbox_union_mask(corrected.shape, frame_bboxes)
    rebuilt = np.full(int(selector.sum()), LABEL_BACKGROUND, dtype=np.int8)
    sampled_positive = np.zeros(rebuilt.shape, dtype=bool)
    if xy.size:
        sampled_inside_bbox = bbox_union[xy[:, 1], xy[:, 0]]
        sampled_positive = corrected[xy[:, 1], xy[:, 0]]
        rebuilt[sampled_inside_bbox] = LABEL_IGNORE
        rebuilt[sampled_positive] = LABEL_FEMUR_CANDIDATE
    result[selector] = rebuilt

    outside_bbox = corrected & ~bbox_union
    positive_outside_cvat = int(
        np.sum((rebuilt == LABEL_FEMUR_CANDIDATE) & ~sampled_positive)
    )
    if positive_outside_cvat:
        raise FullVideoImportError(
            "Authoritative label reconstruction left positive points outside CVAT mask"
        )
    return result, {
        "frame_points": int(selector.sum()),
        "positive_points": int(np.sum(rebuilt == LABEL_FEMUR_CANDIDATE)),
        "ignore_points": int(np.sum(rebuilt == LABEL_IGNORE)),
        "background_points": int(np.sum(rebuilt == LABEL_BACKGROUND)),
        "corrected_positive_pixels": int(corrected.sum()),
        "outside_bbox_positive_pixels": int(outside_bbox.sum()),
        "positive_outside_cvat_mask_points": positive_outside_cvat,
    }


def _refresh_metadata(handle: h5py.File) -> None:
    ann = handle["annotation"]
    labels = ann["point_label"][:].astype(np.int8)
    valid = ann["valid_mask"][:].astype(bool)
    if not np.array_equal(valid, labels != LABEL_IGNORE):
        raise FullVideoImportError("valid_mask differs from point_label != ignore")
    frame_group = handle["frame_annotation"]
    valid_contours = frame_group["valid_contour"][:].astype(bool)
    valid_orders = frame_group["frame_order"][:].astype(np.int64)[valid_contours]
    sources = np.asarray([_decode(value) for value in frame_group["selected_contour_source"][:]])
    handle.attrs["num_labeled_points"] = int(np.sum(labels == LABEL_FEMUR_CANDIDATE))
    handle.attrs["num_valid_contours"] = int(valid_contours.sum())
    handle.attrs["num_valid_contour_frames"] = int(np.unique(valid_orders).size)
    for attr, source in (
        ("num_selected_global_contours", "global"),
        ("num_selected_local_contours", "local_percentile"),
        ("num_selected_shared_contours", "shared"),
    ):
        handle.attrs[attr] = int(np.sum(sources == source))
    handle.attrs["num_selected_phase3_refined_contours"] = int(
        np.sum(np.char.startswith(sources.astype(str), "phase3_"))
    )
    handle.attrs["num_selected_manual_contours"] = int(
        np.sum(np.char.startswith(sources.astype(str), "manual_cvat_fullvideo"))
    )
    source_stats = summarize_labels_by_source(
        point_cloud={"source_flags": handle["point_cloud/source_flags"][:]},
        point_label=labels,
        valid_mask=valid,
    )
    for name, value in source_stats.items():
        handle.attrs[name] = int(value)


def _write_provenance(
    handle: h5py.File,
    *,
    source_h5: Path,
    snapshot_manifest_sha256: str,
    package_summary: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]] = (),
    output_token: str = OUTPUT_TOKEN,
) -> None:
    if "manual_review_fullvideo" in handle:
        del handle["manual_review_fullvideo"]
    group = handle.create_group("manual_review_fullvideo")
    text_fields = ("frame_stem", "decision", "mask_sha256")
    integer_fields = (
        "frame_order",
        "frame_index",
        "bbox_index",
        "before_positive_points",
        "after_positive_points",
        "outside_bbox_positive_pixels",
    )
    for name in text_fields:
        group.create_dataset(
            name,
            data=np.asarray([str(row.get(name, "")) for row in rows], dtype=h5py.string_dtype("utf-8")),
        )
    for name in integer_fields:
        group.create_dataset(
            name,
            data=np.asarray([int(row.get(name, 0)) for row in rows], dtype=np.int64),
        )
    reviewed = snapshot is not None
    authoritative = output_token == AUTHORITATIVE_OUTPUT_TOKEN
    schema_version = AUTHORITATIVE_SCHEMA_VERSION if authoritative else SCHEMA_VERSION
    manual_source = (
        AUTHORITATIVE_MANUAL_SOURCE
        if authoritative and reviewed
        else MANUAL_SOURCE
        if reviewed or not authoritative
        else "inherited"
    )
    group.attrs["schema_version"] = schema_version
    group.attrs["teacher_version"] = output_token
    group.attrs["source_teacher_version"] = SOURCE_TOKEN
    group.attrs["reviewed_video"] = reviewed
    group.attrs["source_h5_sha256"] = file_sha256(source_h5)
    group.attrs["snapshot_manifest_sha256"] = snapshot_manifest_sha256
    if snapshot is not None:
        group.attrs["task_id"] = int(snapshot["task_id"])
        group.attrs["task_name"] = str(snapshot["task_name"])
        group.attrs["annotation_zip_sha256"] = str(snapshot["annotation_sha256"])
        group.attrs["task_backup_sha256"] = str(snapshot["backup_sha256"])
    if package_summary is not None:
        group.attrs["review_package_sha256"] = str(package_summary["review_package_sha256"])

    if "manual_review_fullvideo_frames" in handle:
        del handle["manual_review_fullvideo_frames"]
    frame_group = handle.create_group("manual_review_fullvideo_frames")
    for name in (
        "frame_stem",
        "frame_role",
        "cvat_mask_status",
        "label_authority",
    ):
        frame_group.create_dataset(
            name,
            data=np.asarray(
                [str(row.get(name, "")) for row in frame_rows],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
    for name in (
        "frame_order",
        "frame_index",
        "cvat_mask_positive_pixels",
        "before_positive_points",
        "positive_points",
        "ignore_points",
        "background_points",
        "outside_bbox_positive_pixels",
        "positive_outside_cvat_mask_points",
    ):
        frame_group.create_dataset(
            name,
            data=np.asarray(
                [int(row.get(name, 0)) for row in frame_rows], dtype=np.int64
            ),
        )
    frame_group.attrs["schema_version"] = schema_version
    frame_group.attrs["teacher_version"] = output_token
    frame_group.attrs["source_teacher_version"] = SOURCE_TOKEN
    frame_group.attrs["reviewed_video"] = reviewed
    frame_group.attrs["label_authority"] = (
        AUTHORITATIVE_LABEL_AUTHORITY if authoritative and reviewed else "inherited"
    )
    frame_group.attrs["snapshot_manifest_sha256"] = snapshot_manifest_sha256
    if snapshot is not None:
        frame_group.attrs["task_id"] = int(snapshot["task_id"])
        frame_group.attrs["annotation_zip_sha256"] = str(snapshot["annotation_sha256"])

    handle.attrs["contour_teacher_schema"] = output_token
    handle.attrs["manual_annotation_provenance"] = manual_source
    handle.attrs["manual_review_fullvideo_schema_version"] = schema_version
    handle.attrs["manual_review_fullvideo_reviewed_video"] = reviewed
    handle.attrs["manual_review_fullvideo_applied_bbox_count"] = len(rows)
    handle.attrs["manual_review_fullvideo_empty_bbox_count"] = sum(
        str(row.get("decision")) == "manual_empty" for row in rows
    )
    override_rows = frame_rows if authoritative else rows
    handle.attrs["manual_review_fullvideo_bbox_override_count"] = sum(
        int(row.get("outside_bbox_positive_pixels", 0)) > 0 for row in override_rows
    )
    handle.attrs["manual_review_fullvideo_bbox_override_pixels"] = sum(
        int(row.get("outside_bbox_positive_pixels", 0)) for row in override_rows
    )
    handle.attrs["manual_review_fullvideo_snapshot_manifest_sha256"] = snapshot_manifest_sha256
    handle.attrs["manual_review_fullvideo_source_h5_sha256"] = file_sha256(source_h5)
    handle.attrs["manual_review_fullvideo_authoritative_frame_count"] = len(frame_rows)
    handle.attrs["manual_review_fullvideo_positive_outside_cvat_mask_points"] = sum(
        int(row.get("positive_outside_cvat_mask_points", 0)) for row in frame_rows
    )
    handle["annotation"].attrs["label_source"] = output_token
    handle["annotation"].attrs["manual_annotation_provenance"] = manual_source
    handle["annotation"].attrs["label_authority"] = (
        AUTHORITATIVE_LABEL_AUTHORITY if authoritative and reviewed else "inherited"
    )
    _refresh_metadata(handle)


def read_authoritative_task_frame_masks(
    *,
    package_root: Path,
    frame_rows: Sequence[Mapping[str, Any]],
    annotation_zip: Path,
) -> dict[str, Any]:
    """Read and classify every mask in a full-video CVAT Task.

    This Step 2 reader intentionally does not apply labels.  It fixes
    review_frames.csv as the Task-frame inventory and preserves the distinction
    between an explicit empty mask and a class/object pair omitted by CVAT.
    """
    package_root = Path(package_root).resolve()
    annotation_zip = Path(annotation_zip).resolve()
    if not frame_rows:
        raise FullVideoImportError("Full-video review package has no frame rows")

    stems: list[str] = []
    orders: list[int] = []
    indices: list[int] = []
    normalized_rows: list[dict[str, Any]] = []
    for source_row in frame_rows:
        row = dict(source_row)
        stem = str(row.get("frame_stem", ""))
        if not stem or Path(stem).name != stem:
            raise FullVideoImportError(f"Unsafe or empty frame_stem: {stem!r}")
        order = _integer(row.get("frame_order"), "frame_order")
        index = _integer(row.get("frame_index"), "frame_index")
        height = _integer(row.get("height"), "height")
        width = _integer(row.get("width"), "width")
        if order < 0 or height <= 0 or width <= 0:
            raise FullVideoImportError(
                f"Invalid frame geometry/order: {stem}/{order}/{height}x{width}"
            )
        review_target = _bool(row.get("review_target"), "review_target")
        context_only = _bool(row.get("context_only"), "context_only")
        frame_role = str(row.get("frame_role", ""))
        # ``review_target`` records Phase 3 selection, while ``context_only``
        # records absence of an actionable/drawable manual BBox.  An
        # unreviewable selected target therefore has both flags set.
        expected_role = "review_target" if review_target else "context_only"
        if frame_role != expected_role or (not review_target and not context_only):
            raise FullVideoImportError(
                f"Frame role flags differ for {stem}: "
                f"role={frame_role!r}, review_target={review_target}, "
                f"context_only={context_only}"
            )
        stems.append(stem)
        orders.append(order)
        indices.append(index)
        normalized_rows.append(
            {
                "frame_stem": stem,
                "frame_order": order,
                "frame_index": index,
                "height": height,
                "width": width,
                "frame_role": frame_role,
                "review_target": review_target,
                "context_only": context_only,
            }
        )

    if len(stems) != len(set(stems)):
        raise FullVideoImportError("Full-video review package has duplicate frame stems")
    if len(orders) != len(set(orders)):
        raise FullVideoImportError("Full-video review package has duplicate frame orders")
    if len(indices) != len(set(indices)):
        raise FullVideoImportError("Full-video review package has duplicate frame indices")
    if orders != sorted(orders):
        raise FullVideoImportError("Full-video review frame rows are not in frame order")

    zip_summary, masks = read_cvat_segmentation_class_masks(
        annotation_zip,
        label_name="femur",
        images_dir=package_root / "images",
        allowed_missing_mask_stems=stems,
    )
    if zip_summary.stems != stems or set(masks) != set(stems):
        raise FullVideoImportError("CVAT frame stems/order differ from review_frames.csv")
    missing = set(zip_summary.missing_mask_stems)
    if not missing.issubset(stems):
        raise FullVideoImportError("CVAT ZIP reports unknown omitted-mask stems")

    authoritative_frames: list[dict[str, Any]] = []
    frames_by_stem: dict[str, dict[str, Any]] = {}
    frames_by_order: dict[int, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for row in normalized_rows:
        stem = str(row["frame_stem"])
        mask = np.asarray(masks[stem], dtype=np.uint8)
        expected_shape = (int(row["height"]), int(row["width"]))
        if mask.ndim != 2 or mask.shape != expected_shape:
            raise FullVideoImportError(
                f"CVAT mask shape differs from review frame: "
                f"{stem}/{mask.shape}/{expected_shape}"
            )
        positive_pixels = int(np.sum(mask != 0))
        if stem in missing:
            if positive_pixels:
                raise FullVideoImportError(
                    f"CVAT omitted mask was not normalized to empty: {stem}"
                )
            status = CVAT_MASK_OMITTED_AS_EMPTY
        elif positive_pixels:
            status = CVAT_MASK_POSITIVE
        else:
            status = CVAT_MASK_EMPTY
        record = {
            **row,
            "cvat_mask_status": status,
            "cvat_mask_positive_pixels": positive_pixels,
            "mask": mask,
        }
        authoritative_frames.append(record)
        frames_by_stem[stem] = record
        frames_by_order[int(row["frame_order"])] = record
        status_counts[status] += 1

    return {
        "frames": authoritative_frames,
        "frames_by_stem": frames_by_stem,
        "frames_by_order": frames_by_order,
        "masks": masks,
        "zip_summary": zip_summary,
        "mask_status_counts": dict(sorted(status_counts.items())),
    }


def _package_contract(
    review_root: Path,
    video: str,
    source_h5: Path,
    snapshot: Mapping[str, Any],
    manual_fingerprint: str,
) -> dict[str, Any]:
    package_root = review_root / "videos" / video
    frames, bboxes, summary = _review_package_contract(package_root)
    summary["review_root"] = str(package_root)
    if str(summary.get("export_mode")) != "full_video":
        raise FullVideoImportError(f"Review package is not full_video: {video}")
    if str(summary.get("manual_review_fingerprint")) != manual_fingerprint:
        raise FullVideoImportError(f"Manual-review config fingerprint mismatch: {video}")
    source_hashes = {str(row.get("source_annotated_h5_sha256")) for row in frames}
    if source_hashes != {file_sha256(source_h5)}:
        raise FullVideoImportError(f"Source H5 checksum differs from review package: {video}")
    authoritative = read_authoritative_task_frame_masks(
        package_root=package_root,
        frame_rows=frames,
        annotation_zip=Path(snapshot["annotation_path"]),
    )
    stems = [str(row["frame_stem"]) for row in frames]
    target_bbox_rows = [row for row in bboxes if _bool(row.get("review_target"), "review_target")]
    if len(target_bbox_rows) != int(snapshot["review_bboxes"]):
        raise FullVideoImportError(f"Target BBox count differs from task map: {video}")
    zip_summary = authoritative["zip_summary"]
    masks = authoritative["masks"]
    frame_by_stem = {str(row["frame_stem"]): row for row in frames}
    bbox_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_bbox_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bboxes:
        stem = str(row["frame_stem"])
        if stem not in frame_by_stem:
            raise FullVideoImportError(f"BBox has unknown frame stem: {stem}")
        all_bbox_by_stem[stem].append(dict(row))
    for rows in all_bbox_by_stem.values():
        rows.sort(key=lambda row: _integer(row.get("bbox_index"), "bbox_index"))
    for frame in frames:
        stem = str(frame["frame_stem"])
        declared = _integer(frame.get("bbox_count"), "bbox_count")
        actual = len(all_bbox_by_stem.get(stem, ()))
        if declared != actual:
            raise FullVideoImportError(
                f"Frame/BBox inventory count mismatch: {video}/{stem}/{declared}!={actual}"
            )
    actionable = 0
    for row in target_bbox_rows:
        stem = str(row["frame_stem"])
        frame = frame_by_stem.get(stem)
        if frame is None:
            raise FullVideoImportError(f"Target BBox has unknown frame stem: {stem}")
        bbox_by_stem[stem].append(dict(row))
        is_actionable = _bool(
            row.get("manual_review_required"), "manual_review_required"
        )
        has_drawable_geometry = (
            bbox_bounds(
                _parse_bbox(str(row["bbox_local_xyxy"])),
                (int(frame["height"]), int(frame["width"])),
            )
            is not None
        )
        if is_actionable and not has_drawable_geometry:
            raise FullVideoImportError(
                f"Actionable BBox has no drawable geometry: {video}/{stem}/"
                f"bbox={row['bbox_index']}"
            )
        if is_actionable:
            actionable += 1
    outside_violations: list[dict[str, Any]] = []
    for stem, rows in sorted(bbox_by_stem.items()):
        actionable_rows = [
            row
            for row in rows
            if _bool(row.get("manual_review_required"), "manual_review_required")
        ]
        if not actionable_rows:
            continue
        frame = frame_by_stem[stem]
        shape = (int(frame["height"]), int(frame["width"]))
        allowed = bbox_union_mask(
            shape,
            [_parse_bbox(str(row["bbox_local_xyxy"])) for row in actionable_rows],
        )
        corrected = np.asarray(masks[stem], dtype=bool)
        outside = corrected & ~allowed
        outside_count = int(outside.sum())
        if outside_count:
            outside_y, outside_x = np.nonzero(outside)
            positive_count = int(corrected.sum())
            outside_violations.append(
                {
                    "video_name": video,
                    "frame_stem": stem,
                    "frame_order": int(frame["frame_order"]),
                    "frame_index": int(frame["frame_index"]),
                    "bbox_indices": "|".join(
                        str(int(row["bbox_index"])) for row in actionable_rows
                    ),
                    "actionable_bbox_count": len(actionable_rows),
                    "outside_pixels": outside_count,
                    "corrected_positive_pixels": positive_count,
                    "outside_ratio": outside_count / max(positive_count, 1),
                    "outside_x_range": f"{int(outside_x.min())}..{int(outside_x.max())}",
                    "outside_y_range": f"{int(outside_y.min())}..{int(outside_y.max())}",
                }
            )
    if int(summary.get("selected_review_bboxes", -1)) != len(target_bbox_rows):
        raise FullVideoImportError(f"Package selected BBox count mismatch: {video}")
    if int(summary.get("actionable_review_bboxes", -1)) != actionable:
        raise FullVideoImportError(f"Package actionable BBox count mismatch: {video}")
    return {
        "root": package_root,
        "frames": frames,
        "bboxes": bboxes,
        "summary": summary,
        "masks": masks,
        "authoritative_frames": authoritative["frames"],
        "authoritative_frames_by_stem": authoritative["frames_by_stem"],
        "authoritative_frames_by_order": authoritative["frames_by_order"],
        "mask_status_counts": authoritative["mask_status_counts"],
        "bbox_by_stem": bbox_by_stem,
        "all_bbox_by_stem": all_bbox_by_stem,
        "actionable_bboxes": actionable,
        "missing_masks": int(zip_summary.missing_masks),
        "outside_violations": outside_violations,
    }


def _apply_authoritative_package(
    *,
    video: str,
    labels: np.ndarray,
    frame_orders: np.ndarray,
    pixel_xy: np.ndarray,
    frame_group: h5py.Group,
    selected_sources: list[str],
    reasons: list[str],
    package: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply every Task-frame mask and update saved BBox metadata in-place."""
    bbox_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    annotation_sha256 = str(snapshot["annotation_sha256"])
    saved_frame_orders = frame_group["frame_order"][:].astype(np.int64)
    saved_frame_indices = frame_group["frame_index"][:].astype(np.int64)

    for frame in package["authoritative_frames"]:
        stem = str(frame["frame_stem"])
        frame_order = int(frame["frame_order"])
        frame_index = int(frame["frame_index"])
        shape = (int(frame["height"]), int(frame["width"]))
        corrected = np.asarray(frame["mask"], dtype=bool)
        if corrected.shape != shape:
            raise FullVideoImportError(
                f"Authoritative mask shape changed after preflight: {video}/{stem}"
            )

        package_bboxes = list(package["all_bbox_by_stem"].get(stem, ()))
        parsed_bboxes = [
            _parse_bbox(str(row["bbox_local_xyxy"])) for row in package_bboxes
        ]
        saved_rows_for_frame = np.flatnonzero(
            (saved_frame_orders == frame_order) & (saved_frame_indices == frame_index)
        )
        if saved_rows_for_frame.size != len(package_bboxes):
            raise FullVideoImportError(
                "Saved/package all-BBox count mismatch: "
                f"{video}/{stem}/{saved_rows_for_frame.size}!={len(package_bboxes)}"
            )
        for row, bbox in zip(package_bboxes, parsed_bboxes, strict=True):
            bbox_index = int(row["bbox_index"])
            saved_index = _frame_annotation_row(
                frame_group, frame_order, frame_index, bbox_index
            )
            saved_bbox = np.asarray(
                frame_group["bbox_local_xyxy"][saved_index], dtype=np.float64
            )
            if not np.allclose(saved_bbox, bbox, rtol=0.0, atol=1e-4):
                raise FullVideoImportError(
                    f"Saved/package BBox mismatch: {video}/{stem}/{bbox_index}"
                )

        selector = frame_orders == frame_order
        before_frame = labels[selector].copy()
        labels, stats = apply_authoritative_frame_mask(
            labels=labels,
            frame_orders=frame_orders,
            pixel_xy=pixel_xy,
            frame_order=frame_order,
            corrected_mask=corrected,
            frame_bboxes=parsed_bboxes,
        )
        frame_xy = pixel_xy[selector]

        for row, bbox in zip(package_bboxes, parsed_bboxes, strict=True):
            bbox_index = int(row["bbox_index"])
            saved_index = _frame_annotation_row(
                frame_group, frame_order, frame_index, bbox_index
            )
            region = bbox_union_mask(shape, [bbox])
            bbox_mask = corrected & region
            bbox_values = np.asarray(bbox, dtype=np.float32)
            local_bbox = LocalBBox(
                xml_xyxy=bbox_values.copy(),
                raw_xyxy=bbox_values.copy(),
                local_xyxy=bbox_values.copy(),
                valid=bbox_bounds(bbox, shape) is not None,
            )
            metrics = mask_shape_metrics(bbox_mask, local_bbox)
            positive_points = _point_count(bbox_mask, frame_xy)
            before_positive = int(frame_group["num_labeled_points"][saved_index])
            is_empty = not np.any(bbox_mask)
            selected_sources[saved_index] = (
                AUTHORITATIVE_EMPTY_SOURCE if is_empty else AUTHORITATIVE_MANUAL_SOURCE
            )
            reasons[saved_index] = (
                "cvat_snapshot_authoritative_empty"
                if is_empty
                else "cvat_snapshot_authoritative"
            )
            updates = {
                "valid_contour": not is_empty,
                "contour_area": _contour_area(bbox_mask),
                "binary_area": int(bbox_mask.sum()),
                "foreground_ratio_in_bbox": float(metrics["area_ratio"]),
                "num_labeled_points": positive_points,
                "contour_positive_points": positive_points,
                "selected_area_ratio": float(metrics["area_ratio"]),
                "center_distance_norm": float(metrics["center_distance_norm"]),
                "contour_selection_score": math.nan,
                "fallback_used": False,
                "fallback_percentile": math.nan,
                "fallback_attempts": 0,
            }
            for name, value in updates.items():
                if name in frame_group:
                    frame_group[name][saved_index] = value
            bbox_rows.append(
                {
                    "frame_stem": stem,
                    "frame_order": frame_order,
                    "frame_index": frame_index,
                    "bbox_index": bbox_index,
                    "decision": "manual_empty" if is_empty else "manual_positive",
                    "mask_sha256": annotation_sha256,
                    "before_positive_points": before_positive,
                    "after_positive_points": positive_points,
                    "outside_bbox_positive_pixels": int(
                        np.sum(corrected & ~region)
                    ),
                }
            )

        if "num_labeled_points_union" in frame_group:
            for saved_index in saved_rows_for_frame:
                frame_group["num_labeled_points_union"][saved_index] = stats[
                    "positive_points"
                ]
        frame_rows.append(
            {
                "frame_stem": stem,
                "frame_order": frame_order,
                "frame_index": frame_index,
                "frame_role": str(frame["frame_role"]),
                "cvat_mask_status": str(frame["cvat_mask_status"]),
                "cvat_mask_positive_pixels": int(
                    frame["cvat_mask_positive_pixels"]
                ),
                "label_authority": AUTHORITATIVE_LABEL_AUTHORITY,
                "before_positive_points": int(
                    np.sum(before_frame == LABEL_FEMUR_CANDIDATE)
                ),
                "positive_points": stats["positive_points"],
                "ignore_points": stats["ignore_points"],
                "background_points": stats["background_points"],
                "outside_bbox_positive_pixels": stats[
                    "outside_bbox_positive_pixels"
                ],
                "positive_outside_cvat_mask_points": stats[
                    "positive_outside_cvat_mask_points"
                ],
            }
        )

    return labels, bbox_rows, frame_rows


def _apply_video(
    *,
    video: str,
    source_h5: Path,
    output_h5: Path,
    snapshot_manifest_sha256: str,
    package: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
    overwrite: bool,
    output_token: str = OUTPUT_TOKEN,
) -> dict[str, Any]:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists() and not overwrite:
        raise FileExistsError(f"Output H5 exists; pass --overwrite: {output_h5}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_h5.name}.", suffix=".tmp", dir=output_h5.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    shutil.copy2(source_h5, temporary)
    applied_rows: list[dict[str, Any]] = []
    applied_frame_rows: list[dict[str, Any]] = []
    try:
        with h5py.File(temporary, "r+") as handle:
            for group_name in ("point_cloud", "annotation", "frame_annotation"):
                if group_name not in handle:
                    raise FullVideoImportError(f"Source H5 lacks {group_name}: {video}")
            ann = handle["annotation"]
            labels_before = ann["point_label"][:].astype(np.int8)
            valid_before = ann["valid_mask"][:].astype(bool)
            if not np.array_equal(valid_before, labels_before != LABEL_IGNORE):
                raise FullVideoImportError(f"Source label/valid contract is invalid: {video}")
            labels = labels_before.copy()
            frame_orders = handle["point_cloud/frame_order"][:].astype(np.int64)
            pixel_xy = handle["point_cloud/pixel_xy"][:]
            frame_group = handle["frame_annotation"]
            selected_sources = [_decode(value) for value in frame_group["selected_contour_source"][:]]
            reasons = [_decode(value) for value in frame_group["annotation_reason"][:]]

            if package is not None and output_token == AUTHORITATIVE_OUTPUT_TOKEN:
                if snapshot is None:
                    raise FullVideoImportError(
                        f"Authoritative package lacks CVAT snapshot: {video}"
                    )
                authoritative_orders = [
                    int(row["frame_order"]) for row in package["authoritative_frames"]
                ]
                num_frames = int(handle.attrs.get("num_frames", -1))
                if num_frames < 0 and "per_frame_counts" in handle["point_cloud"]:
                    num_frames = int(handle["point_cloud/per_frame_counts"].shape[0])
                if num_frames < 0:
                    # review_frames.csv is already checksum-bound to this
                    # source H5. It remains the canonical inventory for old
                    # annotated schemas that did not persist a frame count.
                    num_frames = len(authoritative_orders)
                    if frame_orders.size and (
                        int(frame_orders.min()) < 0
                        or int(frame_orders.max()) >= num_frames
                    ):
                        raise FullVideoImportError(
                            f"Source point frame_order escapes Task inventory: {video}"
                        )
                if num_frames < 0 or authoritative_orders != list(range(num_frames)):
                    raise FullVideoImportError(
                        "CVAT Task frame inventory does not cover every source frame: "
                        f"{video}/{len(authoritative_orders)}!={num_frames}"
                    )
                labels, applied_rows, applied_frame_rows = _apply_authoritative_package(
                    video=video,
                    labels=labels,
                    frame_orders=frame_orders,
                    pixel_xy=pixel_xy,
                    frame_group=frame_group,
                    selected_sources=selected_sources,
                    reasons=reasons,
                    package=package,
                    snapshot=snapshot,
                )

            legacy_package = (
                package if output_token != AUTHORITATIVE_OUTPUT_TOKEN else None
            )
            if legacy_package is not None:
                frame_by_stem = {str(row["frame_stem"]): row for row in package["frames"]}
                for stem, target_rows_all in sorted(package["bbox_by_stem"].items()):
                    frame = frame_by_stem[stem]
                    shape = (int(frame["height"]), int(frame["width"]))
                    target_rows = [
                        row
                        for row in target_rows_all
                        if _bool(
                            row.get("manual_review_required"),
                            "manual_review_required",
                        )
                    ]
                    if not target_rows:
                        continue
                    for row in target_rows:
                        if (
                            bbox_bounds(
                                _parse_bbox(str(row["bbox_local_xyxy"])), shape
                            )
                            is None
                        ):
                            raise FullVideoImportError(
                                "Actionable BBox lost drawable geometry after preflight: "
                                f"{video}/{stem}/bbox={row['bbox_index']}"
                            )
                    frame_order = int(frame["frame_order"])
                    frame_index = int(frame["frame_index"])
                    corrected = np.asarray(package["masks"][stem], dtype=np.uint8)
                    target_bboxes = [_parse_bbox(str(row["bbox_local_xyxy"])) for row in target_rows]
                    before_frame_labels = labels[frame_orders == frame_order].copy()
                    labels, stats = _apply_selected_frame(
                        labels=labels,
                        frame_orders=frame_orders,
                        pixel_xy=pixel_xy,
                        frame_order=frame_order,
                        corrected_mask=corrected,
                        target_bboxes=target_bboxes,
                        allow_outside_single_bbox=True,
                    )
                    frame_selector = frame_orders == frame_order
                    frame_xy = pixel_xy[frame_selector]
                    for row, bbox in zip(target_rows, target_bboxes, strict=True):
                        bbox_index = int(row["bbox_index"])
                        saved_index = _frame_annotation_row(
                            frame_group, frame_order, frame_index, bbox_index
                        )
                        saved_bbox = np.asarray(frame_group["bbox_local_xyxy"][saved_index], dtype=np.float64)
                        if not np.allclose(saved_bbox, bbox, rtol=0.0, atol=1e-4):
                            raise FullVideoImportError(f"Saved/package BBox mismatch: {video}/{stem}/{bbox_index}")
                        region = bbox_union_mask(shape, [bbox])
                        if len(target_rows) == 1:
                            # A completed human mask is authoritative when its
                            # assignment is unambiguous. The original VOC BBox
                            # remains as weak-teacher provenance, but does not
                            # clip a corrected femur contour.
                            bbox_mask = corrected.astype(bool)
                        else:
                            bbox_mask = corrected.astype(bool) & region
                        bbox_values = np.asarray(bbox, dtype=np.float32)
                        local_bbox = LocalBBox(
                            xml_xyxy=bbox_values.copy(),
                            raw_xyxy=bbox_values.copy(),
                            local_xyxy=bbox_values.copy(),
                            valid=True,
                        )
                        metrics = mask_shape_metrics(bbox_mask, local_bbox)
                        positive_points = _point_count(bbox_mask, frame_xy)
                        before_positive = int(frame_group["num_labeled_points"][saved_index])
                        is_empty = not np.any(bbox_mask)
                        selected_sources[saved_index] = MANUAL_EMPTY_SOURCE if is_empty else MANUAL_SOURCE
                        reasons[saved_index] = "manual_review_empty" if is_empty else "manual_review_completed"
                        updates = {
                            "valid_contour": not is_empty,
                            "contour_area": _contour_area(bbox_mask),
                            "binary_area": int(bbox_mask.sum()),
                            "foreground_ratio_in_bbox": float(metrics["area_ratio"]),
                            "num_labeled_points": positive_points,
                            "contour_positive_points": positive_points,
                            "selected_area_ratio": float(metrics["area_ratio"]),
                            "center_distance_norm": float(metrics["center_distance_norm"]),
                            "contour_selection_score": math.nan,
                            "fallback_used": False,
                            "fallback_percentile": math.nan,
                            "fallback_attempts": 0,
                        }
                        for name, value in updates.items():
                            if name in frame_group:
                                frame_group[name][saved_index] = value
                        applied_rows.append(
                            {
                                "frame_stem": stem,
                                "frame_order": frame_order,
                                "frame_index": frame_index,
                                "bbox_index": bbox_index,
                                "decision": "manual_empty" if is_empty else "manual_positive",
                                "mask_sha256": file_sha256(Path(snapshot["annotation_path"])),
                                "before_positive_points": before_positive,
                                "after_positive_points": positive_points,
                                "outside_bbox_positive_pixels": int(
                                    np.sum(bbox_mask & ~region)
                                ),
                            }
                        )
                    if "num_labeled_points_union" in frame_group:
                        same_frame_rows = np.flatnonzero(
                            (frame_group["frame_order"][:].astype(np.int64) == frame_order)
                            & (frame_group["frame_index"][:].astype(np.int64) == frame_index)
                        )
                        for saved_index in same_frame_rows:
                            frame_group["num_labeled_points_union"][saved_index] = stats["positive_points"]
                    if np.array_equal(before_frame_labels, labels[frame_selector]) and np.any(corrected):
                        # A valid mask can legitimately contain no sampled points. This branch is
                        # intentionally informational rather than an error.
                        pass

            _replace_dataset(ann, "point_label", labels.astype(np.int8))
            _replace_dataset(ann, "valid_mask", (labels != LABEL_IGNORE).astype(bool))
            _replace_text_dataset(frame_group, "selected_contour_source", selected_sources)
            _replace_text_dataset(frame_group, "annotation_reason", reasons)
            _write_provenance(
                handle,
                source_h5=source_h5,
                snapshot_manifest_sha256=snapshot_manifest_sha256,
                package_summary=package["summary"] if package is not None else None,
                snapshot=snapshot,
                rows=applied_rows,
                frame_rows=applied_frame_rows,
                output_token=output_token,
            )
            handle.flush()
        if output_h5.exists():
            output_h5.unlink()
        os.replace(temporary, output_h5)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    with h5py.File(output_h5, "r") as handle:
        labels_after = handle["annotation/point_label"][:].astype(np.int8)
    return {
        "video_name": video,
        "status": "reviewed" if package is not None else "inherited",
        "source_h5": str(source_h5),
        "source_h5_sha256": file_sha256(source_h5),
        "output_h5": str(output_h5.resolve()),
        "output_h5_sha256": file_sha256(output_h5),
        "points": int(labels_after.size),
        "positive_before": int(np.sum(labels_before == LABEL_FEMUR_CANDIDATE)),
        "positive_after": int(np.sum(labels_after == LABEL_FEMUR_CANDIDATE)),
        "added_positive_points": int(np.sum((labels_after == 1) & (labels_before != 1))),
        "removed_positive_points": int(np.sum((labels_after != 1) & (labels_before == 1))),
        "applied_bboxes": len(applied_rows),
        "empty_bboxes": sum(row["decision"] == "manual_empty" for row in applied_rows),
        "authoritative_frames": len(applied_frame_rows),
        "positive_outside_cvat_mask_points": sum(
            int(row["positive_outside_cvat_mask_points"])
            for row in applied_frame_rows
        ),
        "task_id": int(snapshot["task_id"]) if snapshot is not None else "",
    }


def _validate_existing(
    output_h5: Path,
    source_h5: Path,
    snapshot_manifest_sha256: str,
    output_token: str = OUTPUT_TOKEN,
) -> dict[str, Any]:
    with h5py.File(output_h5, "r") as handle:
        if _decode(handle.attrs.get("contour_teacher_schema", "")) != output_token:
            raise FullVideoImportError(f"Existing output has wrong teacher schema: {output_h5}")
        if _decode(handle.attrs.get("manual_review_fullvideo_source_h5_sha256", "")) != file_sha256(source_h5):
            raise FullVideoImportError(f"Existing output source checksum mismatch: {output_h5}")
        if _decode(handle.attrs.get("manual_review_fullvideo_snapshot_manifest_sha256", "")) != snapshot_manifest_sha256:
            raise FullVideoImportError(f"Existing output snapshot checksum mismatch: {output_h5}")
        labels = handle["annotation/point_label"][:].astype(np.int8)
        valid = handle["annotation/valid_mask"][:].astype(bool)
        if not np.array_equal(valid, labels != LABEL_IGNORE):
            raise FullVideoImportError(f"Existing output label/valid mismatch: {output_h5}")
        authoritative_frames = int(
            handle.attrs.get("manual_review_fullvideo_authoritative_frame_count", 0)
        )
        positive_outside = int(
            handle.attrs.get(
                "manual_review_fullvideo_positive_outside_cvat_mask_points", 0
            )
        )
        reviewed = bool(
            handle.attrs.get("manual_review_fullvideo_reviewed_video", False)
        )
        if output_token == AUTHORITATIVE_OUTPUT_TOKEN:
            if "manual_review_fullvideo_frames" not in handle:
                raise FullVideoImportError(
                    f"Existing output lacks authoritative frame provenance: {output_h5}"
                )
            frame_provenance = handle["manual_review_fullvideo_frames"]
            for name in (
                "frame_order",
                "label_authority",
                "positive_outside_cvat_mask_points",
            ):
                if name not in frame_provenance:
                    raise FullVideoImportError(
                        f"Existing output lacks frame provenance field {name}: {output_h5}"
                    )
            if len(frame_provenance["frame_order"]) != authoritative_frames:
                raise FullVideoImportError(
                    f"Existing output authoritative frame count mismatch: {output_h5}"
                )
            saved_positive_outside = int(
                np.sum(frame_provenance["positive_outside_cvat_mask_points"][:])
            )
            if saved_positive_outside != positive_outside:
                raise FullVideoImportError(
                    f"Existing output CVAT-boundary summary mismatch: {output_h5}"
                )
            if reviewed and authoritative_frames <= 0:
                raise FullVideoImportError(
                    f"Reviewed output has no authoritative frames: {output_h5}"
                )
            authorities = {
                _decode(value) for value in frame_provenance["label_authority"][:]
            }
            expected_authorities = {AUTHORITATIVE_LABEL_AUTHORITY} if reviewed else set()
            if authorities != expected_authorities:
                raise FullVideoImportError(
                    f"Existing output frame authority mismatch: {output_h5}"
                )
            if positive_outside:
                raise FullVideoImportError(
                    f"Existing output has positives outside CVAT masks: {output_h5}"
                )
        return {
            "points": int(labels.size),
            "positive_after": int(np.sum(labels == LABEL_FEMUR_CANDIDATE)),
            "applied_bboxes": int(handle.attrs.get("manual_review_fullvideo_applied_bbox_count", 0)),
            "empty_bboxes": int(handle.attrs.get("manual_review_fullvideo_empty_bbox_count", 0)),
            "authoritative_frames": authoritative_frames,
            "positive_outside_cvat_mask_points": positive_outside,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_token = str(getattr(args, "output_teacher_token", OUTPUT_TOKEN))
    if output_token not in {OUTPUT_TOKEN, AUTHORITATIVE_OUTPUT_TOKEN}:
        raise FullVideoImportError(
            "output_teacher_token must be one of "
            f"{OUTPUT_TOKEN!r}, {AUTHORITATIVE_OUTPUT_TOKEN!r}"
        )
    authoritative_mode = output_token == AUTHORITATIVE_OUTPUT_TOKEN
    manifest_path = Path(args.manifest).resolve()
    source_root = Path(args.source_annotated_root).resolve()
    review_root = Path(args.review_root).resolve()
    snapshot_root = Path(args.snapshot_root).resolve()
    task_map_csv = Path(
        getattr(args, "task_map_csv", None)
        or review_root / "task_management" / "task_map.csv"
    ).resolve()
    output_root = Path(args.output_root).resolve()
    manual_config = load_manual_review_config(args.manual_review_config)
    if manual_config.label_name != "femur":
        raise FullVideoImportError("Final import requires the fixed femur label")
    manifest_items = load_stage4_sweep_manifest(manifest_path)
    exclusions = load_exclusions(args.exclusion_manifest)
    excluded = {row.video_name for row in exclusions}
    enabled = [row for row in manifest_items if row.enabled and row.video_name not in excluded]
    if len(enabled) != int(args.expected_videos):
        raise FullVideoImportError(
            f"Enabled final video count mismatch: {len(enabled)} != {args.expected_videos}"
        )
    manifest_names = {row.video_name for row in manifest_items}
    if not excluded.issubset(manifest_names):
        raise FullVideoImportError("Exclusion manifest contains videos absent from training manifest")
    snapshots = _snapshot_contract(snapshot_root, int(args.expected_reviewed_videos))
    _task_map_contract(task_map_csv, snapshots)
    enabled_names = {row.video_name for row in enabled}
    if set(snapshots) - enabled_names:
        raise FullVideoImportError(
            f"Snapshot contains disabled/excluded videos: {sorted(set(snapshots) - enabled_names)}"
        )
    review_video_dirs = {
        path.name for path in (review_root / "videos").iterdir() if path.is_dir()
    }
    if set(snapshots) != review_video_dirs - excluded:
        raise FullVideoImportError(
            "Snapshot video set must equal review packages minus explicit exclusions"
        )
    sources = {row.video_name: _resolve_source(source_root, row.video_name) for row in enabled}
    snapshot_manifest_sha256 = file_sha256(snapshot_root / "export_manifest.csv")

    print("Stage 4 Phase 5 full-video CVAT final import")
    print(f"  manifest            : {manifest_path}")
    print(f"  source annotated root: {source_root}")
    print(f"  review root          : {review_root}")
    print(f"  snapshot root        : {snapshot_root}")
    print(f"  task map             : {task_map_csv}")
    print(f"  output root          : {output_root}")
    print(f"  output teacher       : {output_token}")
    print(f"  CVAT authority       : {'all_task_frames' if authoritative_mode else 'actionable_bboxes'}")
    print(f"  enabled videos       : {len(enabled)}")
    print(f"  reviewed videos      : {len(snapshots)}")
    print(f"  excluded videos      : {sorted(excluded)}")
    print(f"  preflight only       : {bool(args.preflight_only)}")

    packages: dict[str, dict[str, Any]] = {}
    actionable = missing_masks = 0
    mask_status_counts: Counter[str] = Counter()
    outside_violations: list[dict[str, Any]] = []
    for index, video in enumerate(sorted(snapshots), start=1):
        package = _package_contract(
            review_root,
            video,
            sources[video],
            snapshots[video],
            manual_config.fingerprint,
        )
        packages[video] = package
        actionable += int(package["actionable_bboxes"])
        missing_masks += int(package["missing_masks"])
        mask_status_counts.update(package["mask_status_counts"])
        outside_violations.extend(package["outside_violations"])
        print(
            f"  [{index}/{len(snapshots)}] [OK preflight] {video}: "
            f"targets={snapshots[video]['review_bboxes']}, "
            f"actionable={package['actionable_bboxes']}, "
            f"mask_status={package['mask_status_counts']}"
        )
    if actionable != int(args.expected_actionable_bboxes):
        raise FullVideoImportError(
            f"Actionable BBox count mismatch: {actionable} != {args.expected_actionable_bboxes}"
        )
    if outside_violations:
        print("\nAuthoritative manual-mask pixels outside original selected BBox")
        for index, row in enumerate(outside_violations, start=1):
            print(
                f"  {index}. video={row['video_name']}, stem={row['frame_stem']}, "
                f"frame={row['frame_order']}/{row['frame_index']}, "
                f"bbox_indices={row['bbox_indices']}, outside={row['outside_pixels']}/"
                f"{row['corrected_positive_pixels']} "
                f"({float(row['outside_ratio']):.6f}), "
                f"x={row['outside_x_range']}, y={row['outside_y_range']}"
            )
    ambiguous_outside = [
        row
        for row in outside_violations
        if int(row["actionable_bbox_count"]) != 1
    ]
    if ambiguous_outside and not authoritative_mode:
        raise FullVideoImportError(
            "Corrected masks escape a multi-BBox target union and cannot be "
            f"assigned unambiguously: frames={len(ambiguous_outside)}"
        )
    override_frames = len(outside_violations)
    override_pixels = sum(int(row["outside_pixels"]) for row in outside_violations)
    if (
        int(args.expected_bbox_override_frames) >= 0
        and override_frames != int(args.expected_bbox_override_frames)
    ):
        raise FullVideoImportError(
            "Manual-mask BBox override frame count mismatch: "
            f"{override_frames} != {args.expected_bbox_override_frames}"
        )
    if (
        int(args.expected_bbox_override_pixels) >= 0
        and override_pixels != int(args.expected_bbox_override_pixels)
    ):
        raise FullVideoImportError(
            "Manual-mask BBox override pixel count mismatch: "
            f"{override_pixels} != {args.expected_bbox_override_pixels}"
        )
    if args.preflight_only:
        print("Stage 4 Phase 5 full-video final-import preflight passed; no H5 was written.")
        return {
            "status": "preflight_ok",
            "videos": len(enabled),
            "actionable_bboxes": actionable,
            "cvat_frame_mask_status_counts": dict(sorted(mask_status_counts.items())),
        }

    if output_root.exists() and output_root.is_symlink():
        raise FullVideoImportError("output_root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(enabled, key=lambda item: item.video_name), start=1):
        video = row.video_name
        source_h5 = sources[video]
        output_h5 = output_root / video / _output_name(source_h5, output_token)
        if output_h5.exists() and args.skip_existing:
            checked = _validate_existing(
                output_h5,
                source_h5,
                snapshot_manifest_sha256,
                output_token,
            )
            result = {
                "video_name": video,
                "status": "skipped_verified",
                "source_h5": str(source_h5),
                "source_h5_sha256": file_sha256(source_h5),
                "output_h5": str(output_h5),
                "output_h5_sha256": file_sha256(output_h5),
                "positive_before": "",
                "added_positive_points": "",
                "removed_positive_points": "",
                "task_id": int(snapshots[video]["task_id"]) if video in snapshots else "",
                **checked,
            }
        else:
            result = _apply_video(
                video=video,
                source_h5=source_h5,
                output_h5=output_h5,
                snapshot_manifest_sha256=snapshot_manifest_sha256,
                package=packages.get(video),
                snapshot=snapshots.get(video),
                overwrite=bool(args.overwrite),
                output_token=output_token,
            )
        results.append(result)
        print(
            f"  [{index}/{len(enabled)}] [OK {result['status']}] {video}: "
            f"points={result['points']}, positive={result['positive_after']}, "
            f"applied_bboxes={result['applied_bboxes']}"
        )

    write_csv_rows(output_root / "summary.csv", results)
    status_counts = Counter(str(row["status"]) for row in results)
    summary = {
        "schema_version": (
            AUTHORITATIVE_SCHEMA_VERSION if authoritative_mode else SCHEMA_VERSION
        ),
        "status": "ok",
        "teacher_version": output_token,
        "source_teacher_version": SOURCE_TOKEN,
        "label_authority": (
            AUTHORITATIVE_LABEL_AUTHORITY if authoritative_mode else "actionable_bboxes"
        ),
        "videos": len(results),
        "reviewed_videos": len(snapshots),
        "inherited_videos": len(results) - len(snapshots),
        "excluded_videos": sorted(excluded),
        "selected_review_bboxes": sum(int(row["review_bboxes"]) for row in snapshots.values()),
        "actionable_review_bboxes": actionable,
        "unreviewable_review_bboxes": sum(int(row["review_bboxes"]) for row in snapshots.values()) - actionable,
        "applied_bboxes": sum(int(row["applied_bboxes"]) for row in results),
        "empty_bboxes": sum(int(row["empty_bboxes"]) for row in results),
        "authoritative_frames": sum(
            int(row["authoritative_frames"]) for row in results
        ),
        "positive_outside_cvat_mask_points": sum(
            int(row["positive_outside_cvat_mask_points"]) for row in results
        ),
        "points": sum(int(row["points"]) for row in results),
        "positive_after": sum(int(row["positive_after"]) for row in results),
        "missing_cvat_masks": missing_masks,
        "cvat_frame_mask_status_counts": dict(sorted(mask_status_counts.items())),
        "bbox_override_frames": override_frames,
        "bbox_override_pixels": override_pixels,
        "status_counts": dict(sorted(status_counts.items())),
        "manifest_sha256": file_sha256(manifest_path),
        "exclusion_manifest_sha256": file_sha256(args.exclusion_manifest),
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "manual_review_fingerprint": manual_config.fingerprint,
        "failure_rows": 0,
    }
    write_json(output_root / "import_summary.json", summary)
    print("Stage 4 Phase 5 full-video CVAT final import passed.")
    return {"summary": summary, "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight and apply all reviewed full-video CVAT masks to a new, "
            "exclusion-aware Stage 4 teacher run."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclusion_manifest", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--review_root", type=Path, required=True)
    parser.add_argument("--snapshot_root", type=Path, required=True)
    parser.add_argument(
        "--task_map_csv",
        type=Path,
        default=None,
        help=(
            "Task map returned from the CVAT host. Defaults to "
            "<review_root>/task_management/task_map.csv for self-contained fixtures."
        ),
    )
    parser.add_argument("--manual_review_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--output_teacher_token",
        choices=(OUTPUT_TOKEN, AUTHORITATIVE_OUTPUT_TOKEN),
        default=OUTPUT_TOKEN,
        help=(
            "v4 preserves the historical actionable-BBox behavior; "
            "v5 rebuilds every CVAT Task frame from the snapshot mask."
        ),
    )
    parser.add_argument("--expected_videos", type=int, default=181)
    parser.add_argument("--expected_reviewed_videos", type=int, default=59)
    parser.add_argument("--expected_actionable_bboxes", type=int, default=112)
    parser.add_argument(
        "--expected_bbox_override_frames",
        type=int,
        default=-1,
        help="Optional exact snapshot contract; -1 records without fixing the count.",
    )
    parser.add_argument(
        "--expected_bbox_override_pixels",
        type=int,
        default=-1,
        help="Optional exact snapshot contract; -1 records without fixing the count.",
    )
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.skip_existing and args.overwrite:
        raise SystemExit("--skip_existing and --overwrite are mutually exclusive")
    try:
        run(args)
    except Exception as exc:
        raise SystemExit(
            f"Stage 4 Phase 5 full-video final import failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
