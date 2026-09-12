from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from pseudo3d.analysis.build_stage4_deleted_xml_invalidation_manifest import (
    ACTION,
    REASON_CODE,
    XmlFrameInvalidation,
    invalidation_fingerprint,
    load_xml_invalidation_manifest,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_BACKGROUND,
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
    summarize_labels_by_source,
)


SOURCE_TOKEN = "bboxrank_v5_cvat_authoritative_v1"
OUTPUT_TOKEN = "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
INVALIDATION_SCHEMA_VERSION = 1
FRAME_AUTHORITY = "xml_deletion_manifest"


class XmlInvalidationApplyError(RuntimeError):
    """Raised when a v5 H5 cannot be safely invalidated."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _dataset_kwargs(dataset: h5py.Dataset) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if dataset.compression is not None:
        kwargs["compression"] = dataset.compression
        kwargs["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        kwargs["shuffle"] = True
    if dataset.fletcher32:
        kwargs["fletcher32"] = True
    return kwargs


def _replace_dataset(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
) -> None:
    old = group[name]
    attrs = dict(old.attrs)
    dtype = old.dtype
    kwargs = _dataset_kwargs(old)
    del group[name]
    created = group.create_dataset(name, data=values, dtype=dtype, **kwargs)
    for key, value in attrs.items():
        created.attrs[key] = value


def _replace_text_dataset(
    group: h5py.Group,
    name: str,
    values: Sequence[str],
) -> None:
    old = group[name] if name in group else None
    attrs = dict(old.attrs) if old is not None else {}
    if old is not None:
        del group[name]
    created = group.create_dataset(
        name,
        data=np.asarray(list(values), dtype=h5py.string_dtype("utf-8")),
    )
    for key, value in attrs.items():
        created.attrs[key] = value


def _aligned_row_count(group: h5py.Group, *, group_name: str) -> int:
    datasets = [value for value in group.values() if isinstance(value, h5py.Dataset)]
    if not datasets:
        return 0
    for dataset in datasets:
        if dataset.ndim < 1:
            raise XmlInvalidationApplyError(
                f"{group_name} contains scalar dataset: {dataset.name}"
            )
    lengths = {int(dataset.shape[0]) for dataset in datasets}
    if len(lengths) != 1:
        raise XmlInvalidationApplyError(
            f"{group_name} datasets have different row counts"
        )
    return lengths.pop()


def _archive_and_filter_group(
    *,
    source: h5py.Group,
    archive: h5py.Group,
    remove: np.ndarray,
    group_name: str,
) -> int:
    row_count = _aligned_row_count(source, group_name=group_name)
    selector = np.asarray(remove, dtype=bool)
    if selector.shape != (row_count,):
        raise XmlInvalidationApplyError(
            f"{group_name} removal selector has wrong shape"
        )
    for key, value in source.attrs.items():
        archive.attrs[key] = value
    for name, dataset in list(source.items()):
        if not isinstance(dataset, h5py.Dataset):
            raise XmlInvalidationApplyError(
                f"{group_name} contains unsupported nested group: {name}"
            )
        data = dataset[:]
        archived = archive.create_dataset(
            name,
            data=data[selector],
            dtype=dataset.dtype,
            **_dataset_kwargs(dataset),
        )
        for key, value in dataset.attrs.items():
            archived.attrs[key] = value
        _replace_dataset(source, name, data[~selector])
    return int(selector.sum())


def _frame_counts(
    labels: np.ndarray,
    frame_orders: np.ndarray,
    order: int,
) -> dict[str, int]:
    values = labels[frame_orders == int(order)]
    return {
        "frame_points": int(values.size),
        "positive_points": int(np.sum(values == LABEL_FEMUR_CANDIDATE)),
        "ignore_points": int(np.sum(values == LABEL_IGNORE)),
        "background_points": int(np.sum(values == LABEL_BACKGROUND)),
    }


def _validate_source(
    handle: h5py.File,
    *,
    source_h5: Path,
    invalidations: Sequence[XmlFrameInvalidation],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if _decode(handle.attrs.get("contour_teacher_schema", "")) != SOURCE_TOKEN:
        raise XmlInvalidationApplyError(
            f"Source H5 is not the fixed v5 teacher: {source_h5}"
        )
    for name in (
        "point_cloud/frame_order",
        "annotation/point_label",
        "annotation/valid_mask",
        "frame_annotation/frame_order",
        "frame_annotation/frame_index",
        "frame_annotation/xml_path",
    ):
        if name not in handle:
            raise XmlInvalidationApplyError(f"Source H5 lacks {name}: {source_h5}")
    video = _decode(handle.attrs.get("video_name", ""))
    if not video:
        raise XmlInvalidationApplyError(f"Source H5 lacks video_name: {source_h5}")
    if any(item.video_name != video for item in invalidations):
        raise XmlInvalidationApplyError(
            f"Invalidation video differs from source H5: {video}"
        )
    frame_orders = handle["point_cloud/frame_order"][:].astype(np.int64)
    labels = handle["annotation/point_label"][:].astype(np.int8)
    valid = handle["annotation/valid_mask"][:].astype(bool)
    if frame_orders.shape != labels.shape or valid.shape != labels.shape:
        raise XmlInvalidationApplyError("Point frame/label/valid arrays are not aligned")
    if set(int(value) for value in np.unique(labels)) - {
        LABEL_IGNORE,
        LABEL_BACKGROUND,
        LABEL_FEMUR_CANDIDATE,
    }:
        raise XmlInvalidationApplyError("Source point_label has unsupported values")
    if not np.array_equal(valid, labels != LABEL_IGNORE):
        raise XmlInvalidationApplyError("Source valid_mask differs from point labels")
    if len({item.frame_key for item in invalidations}) != len(invalidations):
        raise XmlInvalidationApplyError("Duplicate invalidation frame")
    return video, frame_orders, labels, valid


def _validate_frame_rows(
    handle: h5py.File,
    item: XmlFrameInvalidation,
) -> np.ndarray:
    group = handle["frame_annotation"]
    row_count = _aligned_row_count(group, group_name="frame_annotation")
    orders = group["frame_order"][:].astype(np.int64)
    indices = group["frame_index"][:].astype(np.int64)
    selector = (orders == item.frame_order) & (indices == item.frame_index)
    if selector.shape != (row_count,):
        raise AssertionError("frame_annotation selector shape is inconsistent")
    matches = int(selector.sum())
    if matches != item.expected_saved_bbox_rows:
        raise XmlInvalidationApplyError(
            f"Saved BBox count mismatch for {item.frame_stem}: "
            f"{matches} != {item.expected_saved_bbox_rows}"
        )
    same_order = orders == item.frame_order
    if int(same_order.sum()) != matches:
        raise XmlInvalidationApplyError(
            f"frame_order maps to multiple frame_index values: {item.frame_stem}"
        )
    xml_paths = [_decode(value) for value in group["xml_path"][:][selector]]
    xml_names = {Path(value).name for value in xml_paths}
    if xml_names != {item.expected_xml_name}:
        raise XmlInvalidationApplyError(
            f"Saved XML basename mismatch for {item.frame_stem}: {sorted(xml_names)}"
        )
    restored = sorted(value for value in xml_paths if Path(value).is_file())
    if restored:
        raise XmlInvalidationApplyError(
            f"Invalidated XML exists again for {item.frame_stem}: {restored}"
        )
    return selector


def _manual_bbox_selector(
    handle: h5py.File,
    item: XmlFrameInvalidation,
) -> np.ndarray | None:
    if "manual_review_fullvideo" not in handle:
        return None
    group = handle["manual_review_fullvideo"]
    count = _aligned_row_count(group, group_name="manual_review_fullvideo")
    for name in ("frame_order", "frame_index"):
        if name not in group:
            raise XmlInvalidationApplyError(
                f"manual_review_fullvideo lacks {name}"
            )
    selector = (
        (group["frame_order"][:].astype(np.int64) == item.frame_order)
        & (group["frame_index"][:].astype(np.int64) == item.frame_index)
    )
    if selector.shape != (count,):
        raise AssertionError("manual-review selector shape is inconsistent")
    if int(selector.sum()) not in {0, item.expected_saved_bbox_rows}:
        raise XmlInvalidationApplyError(
            f"Manual-review BBox count mismatch for {item.frame_stem}"
        )
    return selector


def _update_frame_provenance(
    handle: h5py.File,
    *,
    items: Sequence[XmlFrameInvalidation],
    labels: np.ndarray,
    frame_orders: np.ndarray,
    manifest_sha256: str,
) -> None:
    if "manual_review_fullvideo_frames" not in handle:
        return
    group = handle["manual_review_fullvideo_frames"]
    count = _aligned_row_count(group, group_name="manual_review_fullvideo_frames")
    if count == 0:
        return
    for name in ("frame_order", "frame_index", "frame_stem", "label_authority"):
        if name not in group:
            raise XmlInvalidationApplyError(
                f"manual_review_fullvideo_frames lacks {name}"
            )
    orders = group["frame_order"][:].astype(np.int64)
    indices = group["frame_index"][:].astype(np.int64)
    stems = [_decode(value) for value in group["frame_stem"][:]]
    authorities = [_decode(value) for value in group["label_authority"][:]]
    actions = [""] * count
    reasons = [""] * count
    manifest_hashes = [""] * count
    for item in items:
        selector = (orders == item.frame_order) & (indices == item.frame_index)
        if int(selector.sum()) != 1:
            raise XmlInvalidationApplyError(
                f"Expected one frame provenance row for {item.frame_stem}, "
                f"found {int(selector.sum())}"
            )
        row_index = int(np.flatnonzero(selector)[0])
        if stems[row_index] != item.frame_stem:
            raise XmlInvalidationApplyError(
                f"Frame provenance stem mismatch for {item.frame_stem}"
            )
        authorities[row_index] = FRAME_AUTHORITY
        actions[row_index] = item.action
        reasons[row_index] = item.reason_code
        manifest_hashes[row_index] = manifest_sha256
        counts = _frame_counts(labels, frame_orders, item.frame_order)
        for name, value in (
            ("positive_points", 0),
            ("ignore_points", 0),
            ("background_points", counts["frame_points"]),
            ("positive_outside_cvat_mask_points", 0),
        ):
            if name in group:
                group[name][row_index] = value
    _replace_text_dataset(group, "label_authority", authorities)
    for name, values in (
        ("invalidation_action", actions),
        ("invalidation_reason_code", reasons),
        ("invalidation_manifest_sha256", manifest_hashes),
    ):
        _replace_text_dataset(group, name, values)
    group.attrs["xml_invalidation_frame_count"] = len(items)
    group.attrs["xml_invalidation_manifest_sha256"] = manifest_sha256


def _refresh_metadata(handle: h5py.File) -> None:
    labels = handle["annotation/point_label"][:].astype(np.int8)
    valid = handle["annotation/valid_mask"][:].astype(bool)
    if not np.array_equal(valid, labels != LABEL_IGNORE):
        raise XmlInvalidationApplyError("Output valid_mask differs from labels")
    frame = handle["frame_annotation"]
    row_count = _aligned_row_count(frame, group_name="frame_annotation")
    handle.attrs["num_labeled_points"] = int(
        np.sum(labels == LABEL_FEMUR_CANDIDATE)
    )
    handle.attrs["num_voc_bboxes"] = row_count
    if row_count:
        orders = frame["frame_order"][:].astype(np.int64)
        handle.attrs["num_voc_bbox_frames"] = int(np.unique(orders).size)
    else:
        handle.attrs["num_voc_bbox_frames"] = 0
    if "valid_contour" in frame:
        valid_contours = frame["valid_contour"][:].astype(bool)
        valid_orders = frame["frame_order"][:].astype(np.int64)[valid_contours]
        handle.attrs["num_valid_contours"] = int(valid_contours.sum())
        handle.attrs["num_valid_contour_frames"] = int(
            np.unique(valid_orders).size
        )
    if "fallback_used" in frame:
        handle.attrs["num_fallback_used"] = int(
            frame["fallback_used"][:].astype(bool).sum()
        )
    if "selected_contour_source" in frame:
        sources = np.asarray(
            [_decode(value) for value in frame["selected_contour_source"][:]],
            dtype=str,
        )
        for attr, source in (
            ("num_selected_global_contours", "global"),
            ("num_selected_local_contours", "local_percentile"),
            ("num_selected_shared_contours", "shared"),
        ):
            handle.attrs[attr] = int(np.sum(sources == source))
        handle.attrs["num_selected_phase3_refined_contours"] = int(
            np.sum(np.char.startswith(sources, "phase3_"))
        )
        handle.attrs["num_selected_manual_contours"] = int(
            np.sum(np.char.startswith(sources, "manual_cvat_fullvideo"))
        )
    if "source_flags" in handle["point_cloud"]:
        source_stats = summarize_labels_by_source(
            point_cloud={
                "source_flags": handle["point_cloud/source_flags"][:]
            },
            point_label=labels,
            valid_mask=valid,
        )
        for name, value in source_stats.items():
            handle.attrs[name] = int(value)
    if "manual_review_fullvideo" in handle:
        manual = handle["manual_review_fullvideo"]
        count = _aligned_row_count(
            manual, group_name="manual_review_fullvideo"
        )
        handle.attrs["manual_review_fullvideo_applied_bbox_count"] = count
        if "decision" in manual:
            decisions = [_decode(value) for value in manual["decision"][:]]
            handle.attrs["manual_review_fullvideo_empty_bbox_count"] = sum(
                value == "manual_empty" for value in decisions
            )
        if "outside_bbox_positive_pixels" in manual:
            outside = manual["outside_bbox_positive_pixels"][:].astype(np.int64)
            handle.attrs["manual_review_fullvideo_bbox_override_count"] = int(
                np.sum(outside > 0)
            )
            handle.attrs["manual_review_fullvideo_bbox_override_pixels"] = int(
                outside.sum()
            )


def _write_invalidation_provenance(
    handle: h5py.File,
    *,
    source_h5: Path,
    source_h5_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    manifest_fingerprint: str,
    items: Sequence[XmlFrameInvalidation],
    records: Sequence[Mapping[str, Any]],
    frame_remove: np.ndarray,
    manual_remove: np.ndarray | None,
) -> None:
    if "xml_annotation_invalidation" in handle:
        raise XmlInvalidationApplyError(
            "Source H5 already contains xml_annotation_invalidation"
        )
    group = handle.create_group("xml_annotation_invalidation")
    group.attrs["schema_version"] = INVALIDATION_SCHEMA_VERSION
    group.attrs["teacher_version"] = OUTPUT_TOKEN
    group.attrs["source_teacher_version"] = SOURCE_TOKEN
    group.attrs["policy"] = "explicit_deleted_xml_frame_tombstone"
    group.attrs["frame_authority"] = FRAME_AUTHORITY
    group.attrs["source_h5"] = str(source_h5.resolve())
    group.attrs["source_h5_sha256"] = source_h5_sha256
    group.attrs["manifest_path"] = str(manifest_path.resolve())
    group.attrs["manifest_sha256"] = manifest_sha256
    group.attrs["manifest_fingerprint"] = manifest_fingerprint
    text_fields = (
        "video_name",
        "frame_stem",
        "expected_xml_name",
        "action",
        "reason_code",
        "cvat_mask_status",
    )
    integer_fields = (
        "frame_order",
        "frame_index",
        "expected_saved_bbox_rows",
        "before_positive_points",
        "before_ignore_points",
        "before_background_points",
        "after_positive_points",
        "after_ignore_points",
        "after_background_points",
        "cvat_mask_positive_pixels",
        "removed_frame_annotation_rows",
        "removed_manual_review_rows",
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
            data=np.asarray(
                [int(record.get(name, 0)) for record in records],
                dtype=np.int64,
            ),
        )
    frame_archive = group.create_group("removed_frame_annotation")
    removed_frame_rows = _archive_and_filter_group(
        source=handle["frame_annotation"],
        archive=frame_archive,
        remove=frame_remove,
        group_name="frame_annotation",
    )
    if removed_frame_rows != sum(
        item.expected_saved_bbox_rows for item in items
    ):
        raise XmlInvalidationApplyError(
            "Removed frame_annotation row count differs from manifest"
        )
    if manual_remove is not None:
        manual_archive = group.create_group("removed_manual_review_fullvideo")
        removed_manual_rows = _archive_and_filter_group(
            source=handle["manual_review_fullvideo"],
            archive=manual_archive,
            remove=manual_remove,
            group_name="manual_review_fullvideo",
        )
        if removed_manual_rows != int(
            sum(record["removed_manual_review_rows"] for record in records)
        ):
            raise XmlInvalidationApplyError(
                "Removed manual-review row count differs from preflight"
            )


def _validate_existing_output(
    *,
    output_h5: Path,
    source_h5: Path,
    source_h5_sha256: str,
    manifest_sha256: str,
    manifest_fingerprint: str,
    items: Sequence[XmlFrameInvalidation],
) -> dict[str, Any]:
    with h5py.File(output_h5, "r") as handle:
        if _decode(handle.attrs.get("contour_teacher_schema", "")) != OUTPUT_TOKEN:
            raise XmlInvalidationApplyError(
                f"Existing output has wrong teacher schema: {output_h5}"
            )
        group_name = "xml_annotation_invalidation"
        if group_name not in handle:
            raise XmlInvalidationApplyError(
                f"Existing output lacks {group_name}: {output_h5}"
            )
        group = handle[group_name]
        expected_attrs = {
            "source_h5_sha256": source_h5_sha256,
            "manifest_sha256": manifest_sha256,
            "manifest_fingerprint": manifest_fingerprint,
        }
        for name, expected in expected_attrs.items():
            if _decode(group.attrs.get(name, "")) != expected:
                raise XmlInvalidationApplyError(
                    f"Existing output provenance mismatch: {name}"
                )
        saved_keys = {
            (
                _decode(video),
                int(order),
                int(index),
            )
            for video, order, index in zip(
                group["video_name"][:],
                group["frame_order"][:],
                group["frame_index"][:],
                strict=True,
            )
        }
        if saved_keys != {item.frame_key for item in items}:
            raise XmlInvalidationApplyError(
                "Existing output invalidation frame inventory mismatch"
            )
        labels = handle["annotation/point_label"][:].astype(np.int8)
        valid = handle["annotation/valid_mask"][:].astype(bool)
        orders = handle["point_cloud/frame_order"][:].astype(np.int64)
        if not np.array_equal(valid, labels != LABEL_IGNORE):
            raise XmlInvalidationApplyError("Existing output label/valid mismatch")
        for item in items:
            values = labels[orders == item.frame_order]
            if values.size == 0 or not np.all(values == LABEL_BACKGROUND):
                raise XmlInvalidationApplyError(
                    f"Existing invalidated frame is not all background: {item.frame_stem}"
                )
            frame = handle["frame_annotation"]
            selector = (
                (frame["frame_order"][:].astype(np.int64) == item.frame_order)
                & (frame["frame_index"][:].astype(np.int64) == item.frame_index)
            )
            if np.any(selector):
                raise XmlInvalidationApplyError(
                    f"Existing output retains invalidated BBox: {item.frame_stem}"
                )
            if "manual_review_fullvideo" in handle:
                manual = handle["manual_review_fullvideo"]
                manual_selector = (
                    (manual["frame_order"][:].astype(np.int64) == item.frame_order)
                    & (manual["frame_index"][:].astype(np.int64) == item.frame_index)
                )
                if np.any(manual_selector):
                    raise XmlInvalidationApplyError(
                        "Existing output retains invalidated manual-review BBox: "
                        f"{item.frame_stem}"
                    )
            if "manual_review_fullvideo_frames" in handle:
                frames = handle["manual_review_fullvideo_frames"]
                frame_selector = (
                    (frames["frame_order"][:].astype(np.int64) == item.frame_order)
                    & (frames["frame_index"][:].astype(np.int64) == item.frame_index)
                )
                if int(frame_selector.sum()) == 1:
                    row_index = int(np.flatnonzero(frame_selector)[0])
                    authority = _decode(frames["label_authority"][row_index])
                    if authority != FRAME_AUTHORITY:
                        raise XmlInvalidationApplyError(
                            f"Existing frame authority mismatch: {item.frame_stem}"
                        )
        archived = group["removed_frame_annotation"]
        if len(archived["frame_order"]) != sum(
            item.expected_saved_bbox_rows for item in items
        ):
            raise XmlInvalidationApplyError(
                "Existing removed-BBox archive count mismatch"
            )
        before_positive = int(np.sum(group["before_positive_points"][:]))
        before_ignore = int(np.sum(group["before_ignore_points"][:]))
        removed_bbox_rows = int(
            np.sum(group["removed_frame_annotation_rows"][:])
        )
        return {
            "status": "verified_existing",
            "source_h5": str(source_h5),
            "source_h5_sha256": source_h5_sha256,
            "output_h5": str(output_h5),
            "points": int(labels.size),
            "positive_before": int(
                np.sum(labels == LABEL_FEMUR_CANDIDATE) + before_positive
            ),
            "positive_after": int(np.sum(labels == LABEL_FEMUR_CANDIDATE)),
            "removed_positive_points": before_positive,
            "removed_ignore_points": before_ignore,
            "removed_bbox_rows": removed_bbox_rows,
            "invalidated_frames": len(items),
            "output_h5_sha256": file_sha256(output_h5),
        }


def apply_invalidations_to_h5(
    *,
    source_h5: Path,
    output_h5: Path,
    manifest_path: Path,
    manifest_sha256: str,
    items: Sequence[XmlFrameInvalidation],
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
        raise XmlInvalidationApplyError("Source and output H5 must differ")
    actual_manifest_sha = file_sha256(manifest_path)
    if actual_manifest_sha != manifest_sha256:
        raise XmlInvalidationApplyError("Invalidation manifest SHA-256 mismatch")
    source_sha = file_sha256(source_h5)
    all_manifest_items = load_xml_invalidation_manifest(manifest_path)
    with h5py.File(source_h5, "r") as source_handle:
        source_video = _decode(source_handle.attrs.get("video_name", ""))
    expected_items = sorted(
        (
            item
            for item in all_manifest_items
            if item.video_name == source_video
        ),
        key=lambda item: item.frame_key,
    )
    items = sorted(items, key=lambda item: item.frame_key)
    if [item.to_row() for item in items] != [
        item.to_row() for item in expected_items
    ]:
        raise XmlInvalidationApplyError(
            f"Supplied invalidations differ from manifest rows for {source_video}"
        )
    fingerprint = invalidation_fingerprint(all_manifest_items)
    if output_h5.exists() and not overwrite:
        if skip_existing:
            return _validate_existing_output(
                output_h5=output_h5,
                source_h5=source_h5,
                source_h5_sha256=source_sha,
                manifest_sha256=manifest_sha256,
                manifest_fingerprint=fingerprint,
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
            video, frame_orders, labels_before, _ = _validate_source(
                handle,
                source_h5=source_h5,
                invalidations=items,
            )
            frame_row_count = _aligned_row_count(
                handle["frame_annotation"], group_name="frame_annotation"
            )
            frame_remove = np.zeros(frame_row_count, dtype=bool)
            manual_remove: np.ndarray | None = None
            if "manual_review_fullvideo" in handle:
                manual_remove = np.zeros(
                    _aligned_row_count(
                        handle["manual_review_fullvideo"],
                        group_name="manual_review_fullvideo",
                    ),
                    dtype=bool,
                )
            records: list[dict[str, Any]] = []
            labels = labels_before.copy()
            for item in items:
                if item.action != ACTION or item.reason_code != REASON_CODE:
                    raise XmlInvalidationApplyError(
                        f"Invalid manifest action/reason for {item.frame_stem}"
                    )
                point_selector = frame_orders == item.frame_order
                if not np.any(point_selector):
                    raise XmlInvalidationApplyError(
                        f"Invalidated frame has no points: {item.frame_stem}"
                    )
                row_selector = _validate_frame_rows(handle, item)
                if np.any(frame_remove & row_selector):
                    raise XmlInvalidationApplyError(
                        f"Overlapping invalidation rows: {item.frame_stem}"
                    )
                frame_remove |= row_selector
                selected_manual = _manual_bbox_selector(handle, item)
                removed_manual = 0
                cvat_status = ""
                cvat_pixels = 0
                if selected_manual is not None:
                    if np.any(manual_remove & selected_manual):
                        raise XmlInvalidationApplyError(
                            f"Overlapping manual-review rows: {item.frame_stem}"
                        )
                    manual_remove |= selected_manual
                    removed_manual = int(selected_manual.sum())
                if "manual_review_fullvideo_frames" in handle:
                    frames = handle["manual_review_fullvideo_frames"]
                    if len(frames["frame_order"]):
                        selected_frame = (
                            (frames["frame_order"][:].astype(np.int64) == item.frame_order)
                            & (frames["frame_index"][:].astype(np.int64) == item.frame_index)
                        )
                        if int(selected_frame.sum()) == 1:
                            row_index = int(np.flatnonzero(selected_frame)[0])
                            if "cvat_mask_status" in frames:
                                cvat_status = _decode(
                                    frames["cvat_mask_status"][row_index]
                                )
                            if "cvat_mask_positive_pixels" in frames:
                                cvat_pixels = int(
                                    frames["cvat_mask_positive_pixels"][row_index]
                                )
                before = _frame_counts(labels_before, frame_orders, item.frame_order)
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
                        "cvat_mask_status": cvat_status,
                        "cvat_mask_positive_pixels": cvat_pixels,
                        "removed_frame_annotation_rows": int(row_selector.sum()),
                        "removed_manual_review_rows": removed_manual,
                    }
                )

            handle["annotation/point_label"][...] = labels.astype(np.int8)
            # Non-target labels are unchanged; target labels are background.
            # Therefore this both preserves all non-target validity and makes
            # every invalidated-frame point valid.
            handle["annotation/valid_mask"][...] = labels != LABEL_IGNORE
            _write_invalidation_provenance(
                handle,
                source_h5=source_h5,
                source_h5_sha256=source_sha,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                manifest_fingerprint=fingerprint,
                items=items,
                records=records,
                frame_remove=frame_remove,
                manual_remove=manual_remove,
            )
            _update_frame_provenance(
                handle,
                items=items,
                labels=labels,
                frame_orders=frame_orders,
                manifest_sha256=manifest_sha256,
            )
            handle.attrs["contour_teacher_schema"] = OUTPUT_TOKEN
            handle.attrs["xml_annotation_invalidation_schema_version"] = (
                INVALIDATION_SCHEMA_VERSION
            )
            handle.attrs["xml_annotation_invalidation_frame_count"] = len(items)
            handle.attrs["xml_annotation_invalidation_removed_bbox_count"] = int(
                frame_remove.sum()
            )
            handle.attrs["xml_annotation_invalidation_manifest_sha256"] = (
                manifest_sha256
            )
            handle.attrs["xml_annotation_invalidation_fingerprint"] = fingerprint
            handle.attrs["xml_annotation_invalidation_source_h5_sha256"] = source_sha
            handle.attrs["label_source"] = OUTPUT_TOKEN
            annotation = handle["annotation"]
            annotation.attrs["label_source"] = OUTPUT_TOKEN
            annotation.attrs["xml_invalidation_policy"] = (
                "deleted XML frame -> all background"
            )
            annotation.attrs["xml_invalidation_manifest_sha256"] = manifest_sha256
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
        source_h5_sha256=source_sha,
        manifest_sha256=manifest_sha256,
        manifest_fingerprint=fingerprint,
        items=items,
    )
    result.update(
        {
            "video_name": video,
            "status": "processed",
            "source_h5": str(source_h5),
            "output_h5": str(output_h5),
            "positive_before": int(
                np.sum(labels_before == LABEL_FEMUR_CANDIDATE)
            ),
            "removed_positive_points": int(
                np.sum(
                    (labels_before == LABEL_FEMUR_CANDIDATE)
                    & (labels != LABEL_FEMUR_CANDIDATE)
                )
            ),
            "removed_ignore_points": int(
                np.sum((labels_before == LABEL_IGNORE) & (labels != LABEL_IGNORE))
            ),
            "removed_bbox_rows": int(frame_remove.sum()),
        }
    )
    return result
