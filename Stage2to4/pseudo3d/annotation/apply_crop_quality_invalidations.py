from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256
from pseudo3d.analysis.validate_stage4_crop_quality_manifests import (
    ACTION,
    REASON_CODE,
    CropQualityInvalidation,
    invalidation_fingerprint,
    load_crop_quality_invalidations,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_BACKGROUND,
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
)
from pseudo3d.annotation.apply_deleted_xml_invalidations import (
    _aligned_row_count,
    _archive_and_filter_group,
    _dataset_kwargs,
    _decode,
    _frame_counts,
    _refresh_metadata,
    _replace_text_dataset,
)


SOURCE_TOKEN = "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
OUTPUT_TOKEN = "bboxrank_v7_cvat_authoritative_crop_quality_v1"
INVALIDATION_SCHEMA_VERSION = 1
FRAME_AUTHORITY = "crop_quality_invalidation_manifest"
GROUP_NAME = "crop_quality_invalidation"


class CropQualityApplyError(RuntimeError):
    """Raised when a v6 H5 cannot be safely crop-invalidated."""


def _validate_source(
    handle: h5py.File,
    *,
    source_h5: Path,
    items: Sequence[CropQualityInvalidation],
) -> tuple[str, np.ndarray, np.ndarray]:
    if _decode(handle.attrs.get("contour_teacher_schema", "")) != SOURCE_TOKEN:
        raise CropQualityApplyError(
            f"Source H5 is not the fixed v6 teacher: {source_h5}"
        )
    for name in (
        "point_cloud/frame_order",
        "annotation/point_label",
        "annotation/valid_mask",
        "frame_annotation/frame_order",
        "frame_annotation/frame_index",
        "frame_annotation/bbox_local_xyxy",
        "frame_annotation/xml_path",
        "xml_annotation_invalidation",
    ):
        if name not in handle:
            raise CropQualityApplyError(f"Source H5 lacks {name}: {source_h5}")
    video = _decode(handle.attrs.get("video_name", ""))
    if not video:
        raise CropQualityApplyError(f"Source H5 lacks video_name: {source_h5}")
    if any(item.video_name != video for item in items):
        raise CropQualityApplyError(
            f"Crop invalidation video differs from source H5: {video}"
        )
    if len({item.frame_key for item in items}) != len(items):
        raise CropQualityApplyError("Duplicate crop invalidation frame")
    orders = handle["point_cloud/frame_order"][:].astype(np.int64)
    labels = handle["annotation/point_label"][:].astype(np.int8)
    valid = handle["annotation/valid_mask"][:].astype(bool)
    if orders.shape != labels.shape or valid.shape != labels.shape:
        raise CropQualityApplyError("Point frame/label/valid arrays are not aligned")
    if set(int(value) for value in np.unique(labels)) - {
        LABEL_IGNORE,
        LABEL_BACKGROUND,
        LABEL_FEMUR_CANDIDATE,
    }:
        raise CropQualityApplyError("Source point_label has unsupported values")
    if not np.array_equal(valid, labels != LABEL_IGNORE):
        raise CropQualityApplyError("Source valid_mask differs from point labels")
    return video, orders, labels


def _validate_frame_rows(
    handle: h5py.File,
    item: CropQualityInvalidation,
) -> np.ndarray:
    group = handle["frame_annotation"]
    count = _aligned_row_count(group, group_name="frame_annotation")
    orders = group["frame_order"][:].astype(np.int64)
    indices = group["frame_index"][:].astype(np.int64)
    selector = (orders == item.frame_order) & (indices == item.frame_index)
    if selector.shape != (count,):
        raise AssertionError("frame_annotation selector shape is inconsistent")
    matches = int(selector.sum())
    if matches != item.expected_saved_bbox_rows:
        raise CropQualityApplyError(
            f"Saved BBox count mismatch for {item.frame_stem}: "
            f"{matches} != {item.expected_saved_bbox_rows}"
        )
    if int(np.sum(orders == item.frame_order)) != matches:
        raise CropQualityApplyError(
            f"frame_order maps to multiple frame_index values: {item.frame_stem}"
        )
    boxes = group["bbox_local_xyxy"][:][selector].astype(np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise CropQualityApplyError("bbox_local_xyxy must have shape (N, 4)")
    degenerate = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    if not np.all(degenerate):
        raise CropQualityApplyError(
            f"Crop-invalidated BBox is not degenerate: {item.frame_stem}"
        )
    xml_paths = [_decode(value) for value in group["xml_path"][:][selector]]
    missing = [value for value in xml_paths if not Path(value).is_file()]
    if missing:
        raise CropQualityApplyError(
            f"Crop invalidation must retain source XML for {item.frame_stem}: {missing}"
        )
    return selector


def _selector_for_frame(
    group: h5py.Group,
    item: CropQualityInvalidation,
    *,
    group_name: str,
    allowed_counts: set[int],
) -> np.ndarray:
    count = _aligned_row_count(group, group_name=group_name)
    for name in ("frame_order", "frame_index"):
        if name not in group:
            raise CropQualityApplyError(f"{group_name} lacks {name}")
    selector = (
        (group["frame_order"][:].astype(np.int64) == item.frame_order)
        & (group["frame_index"][:].astype(np.int64) == item.frame_index)
    )
    if selector.shape != (count,) or int(selector.sum()) not in allowed_counts:
        raise CropQualityApplyError(
            f"{group_name} row count mismatch for {item.frame_stem}: "
            f"{int(selector.sum())}"
        )
    return selector


def _archive_selected_group(
    *,
    source: h5py.Group,
    archive: h5py.Group,
    selector: np.ndarray,
    group_name: str,
) -> int:
    count = _aligned_row_count(source, group_name=group_name)
    selector = np.asarray(selector, dtype=bool)
    if selector.shape != (count,):
        raise CropQualityApplyError(f"{group_name} archive selector has wrong shape")
    for key, value in source.attrs.items():
        archive.attrs[key] = value
    for name, dataset in source.items():
        if not isinstance(dataset, h5py.Dataset):
            raise CropQualityApplyError(
                f"{group_name} contains unsupported nested group: {name}"
            )
        copied = archive.create_dataset(
            name,
            data=dataset[:][selector],
            dtype=dataset.dtype,
            **_dataset_kwargs(dataset),
        )
        for key, value in dataset.attrs.items():
            copied.attrs[key] = value
    return int(selector.sum())


def _update_frame_provenance(
    handle: h5py.File,
    *,
    items: Sequence[CropQualityInvalidation],
    labels: np.ndarray,
    frame_orders: np.ndarray,
    manifest_sha256: str,
) -> None:
    if "manual_review_fullvideo_frames" not in handle:
        if items:
            raise CropQualityApplyError(
                "Affected v6 H5 lacks manual_review_fullvideo_frames"
            )
        return
    group = handle["manual_review_fullvideo_frames"]
    count = _aligned_row_count(group, group_name="manual_review_fullvideo_frames")
    for name in ("frame_order", "frame_index", "frame_stem", "label_authority"):
        if name not in group:
            raise CropQualityApplyError(
                f"manual_review_fullvideo_frames lacks {name}"
            )
    orders = group["frame_order"][:].astype(np.int64)
    indices = group["frame_index"][:].astype(np.int64)
    stems = [_decode(value) for value in group["frame_stem"][:]]
    authorities = [_decode(value) for value in group["label_authority"][:]]
    actions = [""] * count
    reasons = [""] * count
    hashes = [""] * count
    for item in items:
        selector = (orders == item.frame_order) & (indices == item.frame_index)
        if int(selector.sum()) != 1:
            raise CropQualityApplyError(
                f"Expected one CVAT frame provenance row for {item.frame_stem}, "
                f"found {int(selector.sum())}"
            )
        row = int(np.flatnonzero(selector)[0])
        if stems[row] != item.frame_stem:
            raise CropQualityApplyError(
                f"CVAT frame provenance stem mismatch for {item.frame_stem}"
            )
        authorities[row] = FRAME_AUTHORITY
        actions[row] = item.action
        reasons[row] = item.reason_code
        hashes[row] = manifest_sha256
        counts = _frame_counts(labels, frame_orders, item.frame_order)
        for name, value in (
            ("positive_points", 0),
            ("ignore_points", 0),
            ("background_points", counts["frame_points"]),
            ("positive_outside_cvat_mask_points", 0),
        ):
            if name in group:
                group[name][row] = value
    _replace_text_dataset(group, "label_authority", authorities)
    for name, values in (
        ("crop_invalidation_action", actions),
        ("crop_invalidation_reason_code", reasons),
        ("crop_invalidation_manifest_sha256", hashes),
    ):
        _replace_text_dataset(group, name, values)
    group.attrs["crop_quality_invalidation_frame_count"] = len(items)
    group.attrs["crop_quality_invalidation_manifest_sha256"] = manifest_sha256


def _write_provenance(
    handle: h5py.File,
    *,
    source_h5: Path,
    source_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    fingerprint: str,
    items: Sequence[CropQualityInvalidation],
    records: Sequence[Mapping[str, Any]],
    frame_remove: np.ndarray,
    manual_remove: np.ndarray | None,
    frame_provenance_selector: np.ndarray | None,
) -> None:
    if GROUP_NAME in handle:
        raise CropQualityApplyError(f"Source H5 already contains {GROUP_NAME}")
    group = handle.create_group(GROUP_NAME)
    for name, value in {
        "schema_version": INVALIDATION_SCHEMA_VERSION,
        "teacher_version": OUTPUT_TOKEN,
        "source_teacher_version": SOURCE_TOKEN,
        "policy": "fully-outside local-crop frame -> all background",
        "frame_authority": FRAME_AUTHORITY,
        "source_h5": str(source_h5),
        "source_h5_sha256": source_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_fingerprint": fingerprint,
    }.items():
        group.attrs[name] = value
    text_fields = (
        "video_name",
        "frame_stem",
        "expected_crop_status",
        "action",
        "reason_code",
        "recoverable",
        "notes",
        "cvat_mask_status",
    )
    integer_fields = (
        "frame_order",
        "frame_index",
        "expected_saved_bbox_rows",
        "expected_stray_ignore_points",
        "before_positive_points",
        "before_ignore_points",
        "before_background_points",
        "after_positive_points",
        "after_ignore_points",
        "after_background_points",
        "removed_frame_annotation_rows",
        "removed_manual_review_rows",
        "cvat_mask_positive_pixels",
    )
    for name in text_fields:
        group.create_dataset(
            name,
            data=np.asarray(
                [str(record.get(name, "")) for record in records],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
    for name in integer_fields:
        group.create_dataset(
            name,
            data=np.asarray([int(record.get(name, 0)) for record in records], dtype=np.int64),
        )
    group.create_dataset(
        "expected_visible_fraction",
        data=np.asarray(
            [float(record.get("expected_visible_fraction", 0.0)) for record in records],
            dtype=np.float64,
        ),
    )
    removed = _archive_and_filter_group(
        source=handle["frame_annotation"],
        archive=group.create_group("removed_frame_annotation"),
        remove=frame_remove,
        group_name="frame_annotation",
    )
    if removed != sum(item.expected_saved_bbox_rows for item in items):
        raise CropQualityApplyError("Removed frame_annotation count differs from manifest")
    if manual_remove is not None:
        removed_manual = _archive_and_filter_group(
            source=handle["manual_review_fullvideo"],
            archive=group.create_group("removed_manual_review_fullvideo"),
            remove=manual_remove,
            group_name="manual_review_fullvideo",
        )
        if removed_manual != sum(
            int(record["removed_manual_review_rows"]) for record in records
        ):
            raise CropQualityApplyError("Removed CVAT BBox count differs from preflight")
    if frame_provenance_selector is not None:
        archived = _archive_selected_group(
            source=handle["manual_review_fullvideo_frames"],
            archive=group.create_group("source_manual_review_fullvideo_frames"),
            selector=frame_provenance_selector,
            group_name="manual_review_fullvideo_frames",
        )
        if archived != len(items):
            raise CropQualityApplyError("Archived CVAT frame provenance count mismatch")


def _validate_existing_output(
    *,
    output_h5: Path,
    source_h5: Path,
    source_sha256: str,
    manifest_sha256: str,
    fingerprint: str,
    items: Sequence[CropQualityInvalidation],
) -> dict[str, Any]:
    with h5py.File(output_h5, "r") as handle:
        if _decode(handle.attrs.get("contour_teacher_schema", "")) != OUTPUT_TOKEN:
            raise CropQualityApplyError("Existing output has wrong teacher schema")
        if GROUP_NAME not in handle:
            raise CropQualityApplyError(f"Existing output lacks {GROUP_NAME}")
        group = handle[GROUP_NAME]
        for name, expected in (
            ("source_h5_sha256", source_sha256),
            ("manifest_sha256", manifest_sha256),
            ("manifest_fingerprint", fingerprint),
        ):
            if _decode(group.attrs.get(name, "")) != expected:
                raise CropQualityApplyError(
                    f"Existing output provenance mismatch: {name}"
                )
        saved = {
            (_decode(video), int(order), int(index))
            for video, order, index in zip(
                group["video_name"][:],
                group["frame_order"][:],
                group["frame_index"][:],
                strict=True,
            )
        }
        if saved != {item.frame_key for item in items}:
            raise CropQualityApplyError("Existing crop invalidation inventory mismatch")
        labels = handle["annotation/point_label"][:].astype(np.int8)
        valid = handle["annotation/valid_mask"][:].astype(bool)
        orders = handle["point_cloud/frame_order"][:].astype(np.int64)
        if not np.array_equal(valid, labels != LABEL_IGNORE):
            raise CropQualityApplyError("Existing output label/valid mismatch")
        frame = handle["frame_annotation"]
        for item in items:
            point_values = labels[orders == item.frame_order]
            if point_values.size == 0 or not np.all(point_values == LABEL_BACKGROUND):
                raise CropQualityApplyError(
                    f"Existing invalidated frame is not all background: {item.frame_stem}"
                )
            selector = (
                (frame["frame_order"][:].astype(np.int64) == item.frame_order)
                & (frame["frame_index"][:].astype(np.int64) == item.frame_index)
            )
            if np.any(selector):
                raise CropQualityApplyError(
                    f"Existing output retains crop-invalidated BBox: {item.frame_stem}"
                )
            frames = handle["manual_review_fullvideo_frames"]
            selector = (
                (frames["frame_order"][:].astype(np.int64) == item.frame_order)
                & (frames["frame_index"][:].astype(np.int64) == item.frame_index)
            )
            if int(selector.sum()) != 1:
                raise CropQualityApplyError("Existing CVAT frame provenance row mismatch")
            row = int(np.flatnonzero(selector)[0])
            if _decode(frames["label_authority"][row]) != FRAME_AUTHORITY:
                raise CropQualityApplyError("Existing CVAT frame authority mismatch")
        removed_bbox = int(np.sum(group["removed_frame_annotation_rows"][:]))
        removed_positive = int(np.sum(group["before_positive_points"][:]))
        removed_ignore = int(np.sum(group["before_ignore_points"][:]))
        return {
            "status": "verified_existing",
            "source_h5": str(source_h5),
            "source_h5_sha256": source_sha256,
            "output_h5": str(output_h5),
            "output_h5_sha256": file_sha256(output_h5),
            "points": int(labels.size),
            "positive_after": int(np.sum(labels == LABEL_FEMUR_CANDIDATE)),
            "removed_positive_points": removed_positive,
            "removed_ignore_points": removed_ignore,
            "removed_bbox_rows": removed_bbox,
            "invalidated_frames": len(items),
        }


def apply_crop_quality_invalidations_to_h5(
    *,
    source_h5: Path,
    output_h5: Path,
    manifest_path: Path,
    manifest_sha256: str,
    items: Sequence[CropQualityInvalidation],
    overwrite: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    source_h5 = Path(source_h5).resolve()
    output_h5 = Path(output_h5).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not source_h5.is_file():
        raise FileNotFoundError(f"Source H5 not found: {source_h5}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Invalidation manifest not found: {manifest_path}")
    if source_h5 == output_h5:
        raise CropQualityApplyError("Source and output H5 must differ")
    if file_sha256(manifest_path) != manifest_sha256:
        raise CropQualityApplyError("Crop invalidation manifest SHA-256 mismatch")
    source_sha = file_sha256(source_h5)
    all_items = load_crop_quality_invalidations(manifest_path)
    with h5py.File(source_h5, "r") as handle:
        video = _decode(handle.attrs.get("video_name", ""))
    expected = sorted(
        (item for item in all_items if item.video_name == video),
        key=lambda item: item.frame_key,
    )
    items = sorted(items, key=lambda item: item.frame_key)
    if [item.to_row() for item in items] != [item.to_row() for item in expected]:
        raise CropQualityApplyError(
            f"Supplied crop invalidations differ from manifest rows for {video}"
        )
    fingerprint = invalidation_fingerprint(all_items)
    if output_h5.exists() and not overwrite:
        if skip_existing:
            return _validate_existing_output(
                output_h5=output_h5,
                source_h5=source_h5,
                source_sha256=source_sha,
                manifest_sha256=manifest_sha256,
                fingerprint=fingerprint,
                items=items,
            )
        raise FileExistsError(f"Output H5 exists; pass --overwrite: {output_h5}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_h5.name}.", suffix=".tmp", dir=output_h5.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source_h5, temporary)
        with h5py.File(temporary, "r+") as handle:
            _, frame_orders, labels_before = _validate_source(
                handle, source_h5=source_h5, items=items
            )
            frame_remove = np.zeros(
                _aligned_row_count(handle["frame_annotation"], group_name="frame_annotation"),
                dtype=bool,
            )
            manual_remove: np.ndarray | None = None
            if "manual_review_fullvideo" in handle:
                manual_remove = np.zeros(
                    _aligned_row_count(
                        handle["manual_review_fullvideo"],
                        group_name="manual_review_fullvideo",
                    ),
                    dtype=bool,
                )
            provenance_selector: np.ndarray | None = None
            if "manual_review_fullvideo_frames" in handle:
                provenance_selector = np.zeros(
                    _aligned_row_count(
                        handle["manual_review_fullvideo_frames"],
                        group_name="manual_review_fullvideo_frames",
                    ),
                    dtype=bool,
                )
            labels = labels_before.copy()
            records: list[dict[str, Any]] = []
            for item in items:
                if item.action != ACTION or item.reason_code != REASON_CODE:
                    raise CropQualityApplyError(
                        f"Invalid manifest action/reason for {item.frame_stem}"
                    )
                point_selector = frame_orders == item.frame_order
                if not np.any(point_selector):
                    raise CropQualityApplyError(
                        f"Crop-invalidated frame has no points: {item.frame_stem}"
                    )
                before = _frame_counts(labels_before, frame_orders, item.frame_order)
                if before["positive_points"] != 0:
                    raise CropQualityApplyError(
                        f"Fully-outside frame unexpectedly has positive points: {item.frame_stem}"
                    )
                if before["ignore_points"] != item.expected_stray_ignore_points:
                    raise CropQualityApplyError(
                        f"Stray ignore count mismatch for {item.frame_stem}: "
                        f"{before['ignore_points']} != {item.expected_stray_ignore_points}"
                    )
                row_selector = _validate_frame_rows(handle, item)
                if np.any(frame_remove & row_selector):
                    raise CropQualityApplyError(
                        f"Overlapping frame invalidations: {item.frame_stem}"
                    )
                frame_remove |= row_selector
                removed_manual = 0
                if manual_remove is not None:
                    selector = _selector_for_frame(
                        handle["manual_review_fullvideo"],
                        item,
                        group_name="manual_review_fullvideo",
                        allowed_counts={0, item.expected_saved_bbox_rows},
                    )
                    if np.any(manual_remove & selector):
                        raise CropQualityApplyError(
                            f"Overlapping CVAT BBox invalidations: {item.frame_stem}"
                        )
                    manual_remove |= selector
                    removed_manual = int(selector.sum())
                cvat_status = ""
                cvat_pixels = 0
                if provenance_selector is not None:
                    selector = _selector_for_frame(
                        handle["manual_review_fullvideo_frames"],
                        item,
                        group_name="manual_review_fullvideo_frames",
                        allowed_counts={1},
                    )
                    provenance_selector |= selector
                    row = int(np.flatnonzero(selector)[0])
                    frames = handle["manual_review_fullvideo_frames"]
                    if "cvat_mask_status" in frames:
                        cvat_status = _decode(frames["cvat_mask_status"][row])
                    if "cvat_mask_positive_pixels" in frames:
                        cvat_pixels = int(frames["cvat_mask_positive_pixels"][row])
                labels[point_selector] = LABEL_BACKGROUND
                after = _frame_counts(labels, frame_orders, item.frame_order)
                records.append(
                    {
                        **item.to_row(),
                        "before_positive_points": before["positive_points"],
                        "before_ignore_points": before["ignore_points"],
                        "before_background_points": before["background_points"],
                        "after_positive_points": after["positive_points"],
                        "after_ignore_points": after["ignore_points"],
                        "after_background_points": after["background_points"],
                        "removed_frame_annotation_rows": int(row_selector.sum()),
                        "removed_manual_review_rows": removed_manual,
                        "cvat_mask_status": cvat_status,
                        "cvat_mask_positive_pixels": cvat_pixels,
                    }
                )
            handle["annotation/point_label"][...] = labels.astype(np.int8)
            handle["annotation/valid_mask"][...] = labels != LABEL_IGNORE
            _write_provenance(
                handle,
                source_h5=source_h5,
                source_sha256=source_sha,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                fingerprint=fingerprint,
                items=items,
                records=records,
                frame_remove=frame_remove,
                manual_remove=manual_remove,
                frame_provenance_selector=provenance_selector,
            )
            _update_frame_provenance(
                handle,
                items=items,
                labels=labels,
                frame_orders=frame_orders,
                manifest_sha256=manifest_sha256,
            )
            handle.attrs["contour_teacher_schema"] = OUTPUT_TOKEN
            handle.attrs["crop_quality_invalidation_schema_version"] = INVALIDATION_SCHEMA_VERSION
            handle.attrs["crop_quality_invalidation_frame_count"] = len(items)
            handle.attrs["crop_quality_invalidation_removed_bbox_count"] = int(frame_remove.sum())
            handle.attrs["crop_quality_invalidation_manifest_sha256"] = manifest_sha256
            handle.attrs["crop_quality_invalidation_fingerprint"] = fingerprint
            handle.attrs["crop_quality_invalidation_source_h5_sha256"] = source_sha
            handle.attrs["label_source"] = OUTPUT_TOKEN
            annotation = handle["annotation"]
            annotation.attrs["label_source"] = OUTPUT_TOKEN
            annotation.attrs["crop_quality_invalidation_policy"] = (
                "fully-outside local-crop frame -> all background"
            )
            annotation.attrs["crop_quality_invalidation_manifest_sha256"] = manifest_sha256
            _refresh_metadata(handle)
            handle.flush()
        if output_h5.exists():
            output_h5.unlink()
        os.replace(temporary, output_h5)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    result = _validate_existing_output(
        output_h5=output_h5,
        source_h5=source_h5,
        source_sha256=source_sha,
        manifest_sha256=manifest_sha256,
        fingerprint=fingerprint,
        items=items,
    )
    result["status"] = "processed"
    return result
