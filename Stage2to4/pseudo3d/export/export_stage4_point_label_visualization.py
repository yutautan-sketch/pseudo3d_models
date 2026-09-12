from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

from src.utils.alpha_texture_processing import image_to_uint8_gray
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    read_cvat_segmentation_class_masks,
    scan_images,
)


LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_FEMUR_CANDIDATE = 1
AUTHORITATIVE_SCHEMA = "bboxrank_v5_cvat_authoritative_v1"
XML_INVALIDATION_SCHEMA = "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
AUTHORITATIVE_SCHEMAS = {AUTHORITATIVE_SCHEMA, XML_INVALIDATION_SCHEMA}
AUTHORITATIVE_LABEL_AUTHORITY = "cvat_snapshot"
XML_INVALIDATION_LABEL_AUTHORITY = "xml_deletion_manifest"
XML_INVALIDATION_ACTION = "invalidate_entire_frame"
XML_INVALIDATION_REASON = "deleted_incorrect_bbox_xml"
SUPPRESSED_CVAT_STATUS = "suppressed_by_xml_invalidation"
CVAT_MASK_STATUSES = {"positive", "empty", "omitted_as_empty"}
KNOWN_EXACT_SOURCES = {
    "",
    "none",
    "global",
    "local_percentile",
    "shared",
}
KNOWN_SOURCE_PREFIXES = ("phase3_", "manual_cvat_")

FRAME_FIELDS = (
    "video_name",
    "frame_order",
    "frame_index",
    "bbox_index",
    "bbox_local_xyxy",
    "selected_contour_source",
    "annotation_reason",
    "valid_contour",
    "num_points",
    "positive_points",
    "ignore_points",
    "background_points",
    "label_authority",
    "xml_invalidated",
    "xml_invalidation_action",
    "xml_invalidation_reason_code",
    "cvat_mask_applied",
    "cvat_mask_suppressed",
    "cvat_mask_status",
    "cvat_mask_pixels",
    "positive_outside_cvat_mask_points",
    "cvat_point_mask_mismatch_points",
    "rendered_image",
)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _attr_text(attrs: Mapping[str, Any], name: str, default: str = "") -> str:
    value = attrs.get(name, default)
    text = _decode(value) if value is not None else default
    return text if text not in {"", "None"} else default


def parse_color(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Color must be R,G,B, got: {text}")
    color = tuple(int(part) for part in parts)
    if any(value < 0 or value > 255 for value in color):
        raise ValueError(f"Color values must be in [0,255], got: {text}")
    return color


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_array(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray([_decode(value) for value in dataset[:]], dtype=object)


def _known_source(source: str) -> bool:
    return source in KNOWN_EXACT_SOURCES or source.startswith(KNOWN_SOURCE_PREFIXES)


def _required_dataset(group: h5py.Group, name: str, qualified: str) -> h5py.Dataset:
    if name not in group:
        raise KeyError(f"Required dataset {qualified}/{name} is missing")
    return group[name]


def _read_manual_review(group: h5py.Group | None) -> dict[str, Any]:
    if group is None:
        return {
            "reviewed_video": False,
            "frame_stem": np.asarray([], dtype=object),
            "frame_order": np.asarray([], dtype=np.int32),
            "frame_index": np.asarray([], dtype=np.int64),
            "bbox_index": np.asarray([], dtype=np.int32),
            "decision": np.asarray([], dtype=object),
            "attrs": {},
        }
    required = (
        "frame_stem",
        "frame_order",
        "frame_index",
        "bbox_index",
        "decision",
    )
    for name in required:
        _required_dataset(group, name, "manual_review_fullvideo")
    result = {
        "reviewed_video": bool(group.attrs.get("reviewed_video", False)),
        "frame_stem": _read_text_array(group["frame_stem"]),
        "frame_order": group["frame_order"][:].astype(np.int32),
        "frame_index": group["frame_index"][:].astype(np.int64),
        "bbox_index": group["bbox_index"][:].astype(np.int32),
        "decision": _read_text_array(group["decision"]),
        "attrs": dict(group.attrs),
    }
    lengths = {values.shape[0] for key, values in result.items() if isinstance(values, np.ndarray)}
    if len(lengths) != 1:
        raise ValueError("manual_review_fullvideo datasets have different row counts")
    return result


def _read_manual_review_frames(group: h5py.Group | None) -> dict[str, Any]:
    empty = {
        "reviewed_video": False,
        "frame_stem": np.asarray([], dtype=object),
        "frame_order": np.asarray([], dtype=np.int32),
        "frame_index": np.asarray([], dtype=np.int64),
        "frame_role": np.asarray([], dtype=object),
        "cvat_mask_status": np.asarray([], dtype=object),
        "cvat_mask_positive_pixels": np.asarray([], dtype=np.int64),
        "label_authority": np.asarray([], dtype=object),
        "positive_points": np.asarray([], dtype=np.int64),
        "ignore_points": np.asarray([], dtype=np.int64),
        "background_points": np.asarray([], dtype=np.int64),
        "positive_outside_cvat_mask_points": np.asarray([], dtype=np.int64),
        "attrs": {},
    }
    if group is None:
        return empty
    text_fields = (
        "frame_stem",
        "frame_role",
        "cvat_mask_status",
        "label_authority",
    )
    integer_fields = (
        "frame_order",
        "frame_index",
        "cvat_mask_positive_pixels",
        "positive_points",
        "ignore_points",
        "background_points",
        "positive_outside_cvat_mask_points",
    )
    for name in (*text_fields, *integer_fields):
        _required_dataset(group, name, "manual_review_fullvideo_frames")
    result = {
        "reviewed_video": bool(group.attrs.get("reviewed_video", False)),
        "attrs": dict(group.attrs),
    }
    for name in text_fields:
        result[name] = _read_text_array(group[name])
    for name in integer_fields:
        dtype = np.int32 if name == "frame_order" else np.int64
        result[name] = group[name][:].astype(dtype)
    lengths = {
        values.shape[0]
        for values in result.values()
        if isinstance(values, np.ndarray)
    }
    if len(lengths) != 1:
        raise ValueError(
            "manual_review_fullvideo_frames datasets have different row counts"
        )
    return result


def _read_xml_invalidations(
    group: h5py.Group | None,
    *,
    schema: str,
    video_name: str,
) -> dict[str, Any]:
    empty = {
        "frame_order": np.asarray([], dtype=np.int32),
        "frame_index": np.asarray([], dtype=np.int64),
        "frame_stem": np.asarray([], dtype=object),
        "action": np.asarray([], dtype=object),
        "reason_code": np.asarray([], dtype=object),
        "attrs": {},
    }
    if group is None:
        if schema == XML_INVALIDATION_SCHEMA:
            raise ValueError(
                "v6 H5 lacks xml_annotation_invalidation provenance"
            )
        return empty
    if schema != XML_INVALIDATION_SCHEMA:
        raise ValueError(
            "Non-v6 H5 unexpectedly contains xml_annotation_invalidation"
        )
    text_fields = ("video_name", "frame_stem", "action", "reason_code")
    integer_fields = ("frame_order", "frame_index")
    for name in (*text_fields, *integer_fields):
        _required_dataset(group, name, "xml_annotation_invalidation")
    result: dict[str, Any] = {"attrs": dict(group.attrs)}
    for name in text_fields:
        result[name] = _read_text_array(group[name])
    result["frame_order"] = group["frame_order"][:].astype(np.int32)
    result["frame_index"] = group["frame_index"][:].astype(np.int64)
    lengths = {
        value.shape[0]
        for value in result.values()
        if isinstance(value, np.ndarray)
    }
    if len(lengths) != 1:
        raise ValueError(
            "xml_annotation_invalidation datasets have different row counts"
        )
    count = int(result["frame_order"].size)
    keys = list(
        zip(
            result["frame_order"].tolist(),
            result["frame_index"].tolist(),
            strict=True,
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError(
            "xml_annotation_invalidation contains duplicate frame keys"
        )
    if set(str(value) for value in result["video_name"]) not in ({video_name}, set()):
        raise ValueError("XML invalidation video_name differs from H5 video")
    if set(str(value) for value in result["action"]) not in (
        {XML_INVALIDATION_ACTION},
        set(),
    ):
        raise ValueError("Unknown XML invalidation action")
    if set(str(value) for value in result["reason_code"]) not in (
        {XML_INVALIDATION_REASON},
        set(),
    ):
        raise ValueError("Unknown XML invalidation reason code")
    expected_attrs = {
        "schema_version": 1,
        "teacher_version": XML_INVALIDATION_SCHEMA,
        "source_teacher_version": AUTHORITATIVE_SCHEMA,
        "policy": "explicit_deleted_xml_frame_tombstone",
        "frame_authority": XML_INVALIDATION_LABEL_AUTHORITY,
    }
    for name, expected in expected_attrs.items():
        actual = group.attrs.get(name)
        if isinstance(expected, str):
            actual = _decode(actual)
        else:
            actual = int(actual) if actual is not None else None
        if actual != expected:
            raise ValueError(
                f"XML invalidation provenance mismatch for {name}: "
                f"{actual!r} != {expected!r}"
            )
    for name in (
        "source_h5_sha256",
        "manifest_sha256",
        "manifest_fingerprint",
    ):
        if not _attr_text(group.attrs, name):
            raise ValueError(f"XML invalidation provenance lacks {name}")
    if "removed_frame_annotation" not in group:
        raise ValueError("XML invalidation provenance lacks removed BBox archive")
    archive = group["removed_frame_annotation"]
    if "frame_order" not in archive:
        raise ValueError("Removed BBox archive lacks frame_order")
    if int(archive["frame_order"].shape[0]) < count:
        raise ValueError("Removed BBox archive is smaller than invalidation inventory")
    return result


def load_stage4_point_label_visualization_input(
    annotated_h5: Path,
    *,
    pseudo3d_h5: Path | None = None,
) -> dict[str, Any]:
    annotated_h5 = Path(annotated_h5).resolve()
    if not annotated_h5.is_file():
        raise FileNotFoundError(f"Annotated H5 not found: {annotated_h5}")

    with h5py.File(annotated_h5, "r") as handle:
        for name in ("point_cloud", "annotation", "frame_annotation"):
            if name not in handle:
                raise KeyError(f"Required group '{name}' is missing: {annotated_h5}")
        pc = handle["point_cloud"]
        ann = handle["annotation"]
        frame = handle["frame_annotation"]
        attrs = dict(handle.attrs)

        pixel_xy = _required_dataset(pc, "pixel_xy", "point_cloud")[:].astype(
            np.float32
        )
        point_frame_order = _required_dataset(
            pc, "frame_order", "point_cloud"
        )[:].astype(np.int32)
        point_frame_index = (
            pc["frame_index"][:].astype(np.int64)
            if "frame_index" in pc
            else None
        )
        point_label = _required_dataset(
            ann, "point_label", "annotation"
        )[:].astype(np.int8)
        valid_mask = _required_dataset(
            ann, "valid_mask", "annotation"
        )[:].astype(bool)

        required_frame = (
            "frame_order",
            "frame_index",
            "bbox_index",
            "bbox_local_xyxy",
            "selected_contour_source",
            "annotation_reason",
            "valid_contour",
        )
        for name in required_frame:
            _required_dataset(frame, name, "frame_annotation")
        frame_annotation = {
            "frame_order": frame["frame_order"][:].astype(np.int32),
            "frame_index": frame["frame_index"][:].astype(np.int64),
            "bbox_index": frame["bbox_index"][:].astype(np.int32),
            "bbox_local_xyxy": frame["bbox_local_xyxy"][:].astype(np.float32),
            "selected_contour_source": _read_text_array(
                frame["selected_contour_source"]
            ),
            "annotation_reason": _read_text_array(frame["annotation_reason"]),
            "valid_contour": frame["valid_contour"][:].astype(bool),
        }
        manual_review = _read_manual_review(
            handle["manual_review_fullvideo"]
            if "manual_review_fullvideo" in handle
            else None
        )
        manual_review_frames = _read_manual_review_frames(
            handle["manual_review_fullvideo_frames"]
            if "manual_review_fullvideo_frames" in handle
            else None
        )
        schema = _attr_text(attrs, "contour_teacher_schema")
        video_name = _attr_text(attrs, "video_name", annotated_h5.parent.name)
        if not video_name:
            raise ValueError("Could not resolve video_name")
        xml_invalidations = _read_xml_invalidations(
            handle["xml_annotation_invalidation"]
            if "xml_annotation_invalidation" in handle
            else None,
            schema=schema,
            video_name=video_name,
        )

    source_path = Path(pseudo3d_h5).resolve() if pseudo3d_h5 else None
    if source_path is None:
        source_text = _attr_text(attrs, "source_pseudo3d_h5")
        if not source_text:
            raise KeyError(
                "source_pseudo3d_h5 metadata is missing; pass --pseudo3d_h5"
            )
        source_path = Path(source_text).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source pseudo3D H5 not found: {source_path}")
    with h5py.File(source_path, "r") as handle:
        for name in ("local_encoder_images", "frame_indices"):
            if name not in handle:
                raise KeyError(f"Required pseudo3D dataset '{name}' is missing")
        local_images = handle["local_encoder_images"][:]
        frame_indices = handle["frame_indices"][:].astype(np.int64)

    num_points = int(point_label.size)
    if pixel_xy.shape != (num_points, 2):
        raise ValueError(
            f"point_cloud/pixel_xy must have shape ({num_points},2), got {pixel_xy.shape}"
        )
    for name, values in (
        ("point_cloud/frame_order", point_frame_order),
        ("annotation/valid_mask", valid_mask),
    ):
        if values.shape != (num_points,):
            raise ValueError(f"{name} length differs from point_label")
    if point_frame_index is not None and point_frame_index.shape != (num_points,):
        raise ValueError("point_cloud/frame_index length differs from point_label")
    if not np.all(np.isfinite(pixel_xy)):
        raise ValueError("point_cloud/pixel_xy contains NaN or Inf")
    labels = set(int(value) for value in np.unique(point_label))
    if not labels.issubset(
        {LABEL_IGNORE, LABEL_BACKGROUND, LABEL_FEMUR_CANDIDATE}
    ):
        raise ValueError(f"point_label contains unknown values: {sorted(labels)}")
    if not np.array_equal(valid_mask, point_label != LABEL_IGNORE):
        raise ValueError("valid_mask differs from point_label != -1")

    num_frames = int(local_images.shape[0])
    if frame_indices.shape != (num_frames,):
        raise ValueError("pseudo3D frame_indices length differs from local images")
    if np.any(point_frame_order < 0) or np.any(point_frame_order >= num_frames):
        raise ValueError("point_cloud/frame_order is outside pseudo3D frame range")
    if point_frame_index is not None and not np.array_equal(
        point_frame_index, frame_indices[point_frame_order]
    ):
        raise ValueError("point_cloud/frame_index is not aligned with frame_order")

    frame_lengths = {values.shape[0] for values in frame_annotation.values()}
    if len(frame_lengths) != 1:
        raise ValueError("frame_annotation datasets have different row counts")
    num_bbox_rows = next(iter(frame_lengths), 0)
    if frame_annotation["bbox_local_xyxy"].shape != (num_bbox_rows, 4):
        raise ValueError("frame_annotation/bbox_local_xyxy must have shape (N,4)")
    if not np.all(np.isfinite(frame_annotation["bbox_local_xyxy"])):
        raise ValueError("frame_annotation/bbox_local_xyxy contains NaN or Inf")
    orders = frame_annotation["frame_order"]
    if np.any(orders < 0) or np.any(orders >= num_frames):
        raise ValueError("frame_annotation/frame_order is outside frame range")
    if not np.array_equal(
        frame_annotation["frame_index"], frame_indices[orders]
    ):
        raise ValueError("frame_annotation/frame_index is not aligned with frame_order")
    keys = list(
        zip(
            frame_annotation["frame_order"].tolist(),
            frame_annotation["frame_index"].tolist(),
            frame_annotation["bbox_index"].tolist(),
            strict=True,
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("frame_annotation contains duplicate frame/BBox keys")
    if np.any(frame_annotation["bbox_index"] < 0):
        raise ValueError("frame_annotation/bbox_index must be >= 0")
    unknown_sources = sorted(
        {
            str(source)
            for source in frame_annotation["selected_contour_source"]
            if not _known_source(str(source))
        }
    )
    if unknown_sources:
        raise ValueError(f"Unknown selected_contour_source values: {unknown_sources}")

    manual_keys = list(
        zip(
            manual_review["frame_order"].tolist(),
            manual_review["frame_index"].tolist(),
            manual_review["bbox_index"].tolist(),
            strict=True,
        )
    )
    if len(manual_keys) != len(set(manual_keys)):
        raise ValueError("manual_review_fullvideo contains duplicate frame/BBox keys")
    frame_key_to_row = {key: index for index, key in enumerate(keys)}
    for key in manual_keys:
        if key not in frame_key_to_row:
            raise ValueError(f"manual_review_fullvideo has unknown frame/BBox key: {key}")
        source = str(
            frame_annotation["selected_contour_source"][frame_key_to_row[key]]
        )
        if not source.startswith("manual_cvat_fullvideo"):
            raise ValueError(
                "manual_review_fullvideo key does not have manual provenance: "
                f"{key}/{source}"
            )

    authoritative = schema in AUTHORITATIVE_SCHEMAS
    invalidation_keys = set(
        zip(
            xml_invalidations["frame_order"].tolist(),
            xml_invalidations["frame_index"].tolist(),
            strict=True,
        )
    )
    invalidation_orders = {int(order) for order, _ in invalidation_keys}
    for row_index, (order, index, stem) in enumerate(
        zip(
            xml_invalidations["frame_order"],
            xml_invalidations["frame_index"],
            xml_invalidations["frame_stem"],
            strict=True,
        )
    ):
        order = int(order)
        index = int(index)
        expected_stem = f"{video_name}__fo{order:05d}__fi{index:08d}"
        if order < 0 or order >= num_frames:
            raise ValueError("XML invalidation frame order is outside frame range")
        if index != int(frame_indices[order]) or str(stem) != expected_stem:
            raise ValueError(
                f"XML invalidation frame identity mismatch: {video_name}/{row_index}"
            )
        frame_labels = point_label[point_frame_order == order]
        if frame_labels.size == 0 or not np.all(
            frame_labels == LABEL_BACKGROUND
        ):
            raise ValueError(
                f"XML-invalidated frame is not all background: {expected_stem}"
            )
        if np.any(
            (frame_annotation["frame_order"] == order)
            & (frame_annotation["frame_index"] == index)
        ):
            raise ValueError(
                f"XML-invalidated frame retains saved BBox: {expected_stem}"
            )
        if np.any(
            (manual_review["frame_order"] == order)
            & (manual_review["frame_index"] == index)
        ):
            raise ValueError(
                f"XML-invalidated frame retains manual BBox: {expected_stem}"
            )
    if schema == XML_INVALIDATION_SCHEMA:
        expected_count = int(
            attrs.get("xml_annotation_invalidation_frame_count", -1)
        )
        if expected_count != len(invalidation_keys):
            raise ValueError(
                "Root XML invalidation frame count differs from provenance"
            )
    frame_provenance_orders = manual_review_frames["frame_order"]
    if frame_provenance_orders.size:
        if not manual_review_frames["reviewed_video"]:
            raise ValueError(
                "manual_review_fullvideo_frames has rows but reviewed_video is false"
            )
        if len(set(manual_review_frames["frame_stem"].tolist())) != int(
            frame_provenance_orders.size
        ):
            raise ValueError(
                "manual_review_fullvideo_frames contains duplicate frame stems"
            )
        if len(set(frame_provenance_orders.tolist())) != int(
            frame_provenance_orders.size
        ):
            raise ValueError(
                "manual_review_fullvideo_frames contains duplicate frame orders"
            )
        if not np.array_equal(
            frame_provenance_orders,
            np.sort(frame_provenance_orders, kind="stable"),
        ):
            raise ValueError(
                "manual_review_fullvideo_frames is not in frame-order order"
            )
        if np.any(frame_provenance_orders < 0) or np.any(
            frame_provenance_orders >= num_frames
        ):
            raise ValueError(
                "manual_review_fullvideo_frames/frame_order is outside frame range"
            )
        if not np.array_equal(
            manual_review_frames["frame_index"],
            frame_indices[frame_provenance_orders],
        ):
            raise ValueError(
                "manual_review_fullvideo_frames/frame_index is not aligned"
            )
        statuses = set(
            str(value) for value in manual_review_frames["cvat_mask_status"]
        )
        if not statuses.issubset(CVAT_MASK_STATUSES):
            raise ValueError(f"Unknown CVAT mask statuses: {sorted(statuses)}")
        for order, index, authority in zip(
            manual_review_frames["frame_order"],
            manual_review_frames["frame_index"],
            manual_review_frames["label_authority"],
            strict=True,
        ):
            key = (int(order), int(index))
            expected_authority = (
                XML_INVALIDATION_LABEL_AUTHORITY
                if key in invalidation_keys
                else AUTHORITATIVE_LABEL_AUTHORITY
            )
            if str(authority) != expected_authority:
                raise ValueError(
                    "manual_review_fullvideo_frames label authority mismatch: "
                    f"{video_name}/{key}/{authority}/{expected_authority}"
                )
        provenance_keys = set(
            zip(
                manual_review_frames["frame_order"].tolist(),
                manual_review_frames["frame_index"].tolist(),
                strict=True,
            )
        )
        if invalidation_keys and not invalidation_keys.issubset(provenance_keys):
            raise ValueError(
                "Reviewed v6 video lacks invalidated-frame provenance rows"
            )
        if np.any(
            manual_review_frames["positive_outside_cvat_mask_points"] != 0
        ):
            raise ValueError(
                "Saved frame provenance reports positive points outside CVAT mask"
            )
    elif authoritative and manual_review_frames["reviewed_video"]:
        raise ValueError(
            "Authoritative reviewed video has no frame-level provenance rows"
        )
    if authoritative:
        expected_frame_rows = int(
            attrs.get("manual_review_fullvideo_authoritative_frame_count", 0)
        )
        if expected_frame_rows != int(frame_provenance_orders.size):
            raise ValueError(
                "Authoritative frame count metadata differs from frame provenance"
            )
        expected_outside = int(
            attrs.get(
                "manual_review_fullvideo_positive_outside_cvat_mask_points", 0
            )
        )
        actual_outside = int(
            np.sum(
                manual_review_frames[
                    "positive_outside_cvat_mask_points"
                ]
            )
        )
        if expected_outside != actual_outside or actual_outside != 0:
            raise ValueError(
                "Authoritative positive-outside metadata is not zero/consistent"
            )

    first_shape: tuple[int, int] | None = None
    rounded_xy = np.rint(pixel_xy).astype(np.int64)
    for order in range(num_frames):
        gray = image_to_uint8_gray(local_images[order])
        if first_shape is None:
            first_shape = tuple(gray.shape)
        elif tuple(gray.shape) != first_shape:
            raise ValueError("local_encoder_images contain inconsistent frame shapes")
    if first_shape is None:
        raise ValueError("Source pseudo3D H5 contains no local frames")
    height, width = first_shape
    if rounded_xy.size and (
        np.any(rounded_xy[:, 0] < 0)
        or np.any(rounded_xy[:, 0] >= width)
        or np.any(rounded_xy[:, 1] < 0)
        or np.any(rounded_xy[:, 1] >= height)
    ):
        raise ValueError("point_cloud/pixel_xy lies outside local image bounds")

    bbox_frame_orders = np.unique(frame_annotation["frame_order"])
    non_background_counts = np.bincount(
        point_frame_order[point_label != LABEL_BACKGROUND], minlength=num_frames
    )
    no_bbox_orders = np.setdiff1d(
        np.arange(num_frames, dtype=np.int32), bbox_frame_orders, assume_unique=True
    )
    invalid_no_bbox_orders = no_bbox_orders[non_background_counts[no_bbox_orders] > 0]
    if authoritative and invalid_no_bbox_orders.size:
        provenance_order_set = set(frame_provenance_orders.tolist())
        allowed = []
        for order in invalid_no_bbox_orders.tolist():
            frame_labels = point_label[point_frame_order == order]
            if (
                order in provenance_order_set
                and np.all(frame_labels != LABEL_IGNORE)
            ):
                allowed.append(order)
        invalid_no_bbox_orders = np.setdiff1d(
            invalid_no_bbox_orders,
            np.asarray(allowed, dtype=np.int32),
            assume_unique=True,
        )
    if invalid_no_bbox_orders.size:
        raise ValueError(
            "BBox-free frames contain non-background labels: "
            f"{invalid_no_bbox_orders.tolist()}"
        )

    return {
        "annotated_h5": annotated_h5,
        "pseudo3d_h5": source_path,
        "annotated_attrs": attrs,
        "video_name": video_name,
        "local_images": local_images,
        "frame_indices": frame_indices,
        "pixel_xy": pixel_xy,
        "point_frame_order": point_frame_order,
        "point_label": point_label,
        "valid_mask": valid_mask,
        "frame_annotation": frame_annotation,
        "manual_review": manual_review,
        "manual_review_frames": manual_review_frames,
        "xml_invalidations": xml_invalidations,
        "xml_invalidation_orders": invalidation_orders,
        "cvat_authoritative": authoritative,
    }


def load_applied_cvat_masks(
    data: Mapping[str, Any],
    *,
    review_root: Path,
    snapshot_root: Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    manual = data["manual_review"]
    manual_frames = data["manual_review_frames"]
    authoritative = bool(data["cvat_authoritative"])
    invalidation_orders = set(data.get("xml_invalidation_orders", set()))
    num_rows = int(manual["frame_order"].size)
    num_frame_rows = int(manual_frames["frame_order"].size)
    provenance = manual_frames if authoritative else manual
    if (num_frame_rows if authoritative else num_rows) == 0:
        return {}, {
            "reviewed_video": bool(provenance["reviewed_video"]),
            "label_authority": (
                "inherited_with_xml_invalidation"
                if authoritative and invalidation_orders
                else "inherited"
                if authoritative
                else "legacy"
            ),
            "authoritative_frame_count": 0,
            "suppressed_cvat_mask_frames": 0,
            "suppressed_cvat_mask_pixels": 0,
            "positive_outside_cvat_mask_points": 0,
            "cvat_point_mask_mismatch_points": 0,
        }
    if not provenance["reviewed_video"]:
        raise ValueError(
            "CVAT provenance has rows but reviewed_video is false"
        )

    review_root = Path(review_root).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    manifest_path = snapshot_root / "export_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CVAT snapshot manifest not found: {manifest_path}")
    expected_manifest_sha = _attr_text(
        data["annotated_attrs"],
        "manual_review_fullvideo_snapshot_manifest_sha256",
    )
    actual_manifest_sha = file_sha256(manifest_path)
    if expected_manifest_sha != actual_manifest_sha:
        raise ValueError("CVAT snapshot manifest checksum differs from annotated H5")

    video_name = str(data["video_name"])
    if Path(video_name).name != video_name:
        raise ValueError(f"Unsafe video_name in annotated H5: {video_name}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("video_name", "")) == video_name
        ]
    if len(manifest_rows) != 1:
        raise ValueError(
            f"Expected one CVAT snapshot manifest row for {video_name}, "
            f"got {len(manifest_rows)}"
        )
    manifest_row = manifest_rows[0]
    if str(manifest_row.get("status")) != "complete":
        raise ValueError(f"CVAT snapshot is not complete: {video_name}")
    task_id = int(manifest_row["task_id"])
    saved_task_id = int(provenance["attrs"].get("task_id", -1))
    if task_id != saved_task_id:
        raise ValueError(f"CVAT task ID differs from annotated H5: {video_name}")
    annotation_zip = (
        snapshot_root
        / "annotations"
        / video_name
        / f"{video_name}__task{task_id}__reviewed_segmentation_mask_1_1.zip"
    )
    annotation_sha = file_sha256(annotation_zip)
    expected_annotation_shas = {
        str(manifest_row.get("annotation_sha256", "")),
        _attr_text(provenance["attrs"], "annotation_zip_sha256"),
    }
    if expected_annotation_shas != {annotation_sha}:
        raise ValueError(f"CVAT annotation ZIP checksum mismatch: {video_name}")

    images_dir = review_root / "videos" / video_name / "images"
    image_stems = sorted(scan_images(images_dir))
    zip_summary, masks = read_cvat_segmentation_class_masks(
        annotation_zip,
        label_name="femur",
        images_dir=images_dir,
        allowed_missing_mask_stems=image_stems,
    )
    if authoritative:
        expected_orders = np.arange(
            int(data["local_images"].shape[0]), dtype=np.int32
        )
        if not np.array_equal(manual_frames["frame_order"], expected_orders):
            raise ValueError(
                "Authoritative frame provenance does not cover every source frame"
            )
        expected_stems = [str(value) for value in manual_frames["frame_stem"]]
        if set(image_stems) != set(expected_stems):
            raise ValueError(
                "CVAT review images differ from authoritative frame provenance"
            )
        missing_stems = set(zip_summary.missing_mask_stems)
        masks_by_order: dict[int, np.ndarray] = {}
        stems_by_order: dict[int, str] = {}
        statuses_by_order: dict[int, str] = {}
        outside_by_order: dict[int, int] = {}
        mismatch_by_order: dict[int, int] = {}
        status_counts: Counter[str] = Counter()
        rendered_status_counts: Counter[str] = Counter()
        suppressed_by_order: dict[int, int] = {}
        rounded_xy = np.rint(data["pixel_xy"]).astype(np.int64)
        for row_index in range(num_frame_rows):
            stem = expected_stems[row_index]
            order = int(manual_frames["frame_order"][row_index])
            index = int(manual_frames["frame_index"][row_index])
            if index != int(data["frame_indices"][order]):
                raise ValueError(
                    f"Authoritative CVAT frame index mismatch: {video_name}/{stem}"
                )
            expected_stem = (
                f"{video_name}__fo{order:05d}__fi{index:08d}"
            )
            if stem != expected_stem:
                raise ValueError(
                    "Authoritative CVAT frame stem/order/index mismatch: "
                    f"{video_name}/{stem}/{expected_stem}"
                )
            if stem not in masks:
                raise KeyError(
                    f"Authoritative CVAT mask stem is missing: {video_name}/{stem}"
                )
            mask = np.asarray(masks[stem], dtype=bool)
            expected_shape = image_to_uint8_gray(
                data["local_images"][order]
            ).shape
            if mask.shape != expected_shape:
                raise ValueError(
                    f"CVAT mask/image shape mismatch: {video_name}/{stem}"
                )
            positive_pixels = int(np.sum(mask))
            actual_status = (
                "omitted_as_empty"
                if stem in missing_stems
                else "positive"
                if positive_pixels
                else "empty"
            )
            saved_status = str(manual_frames["cvat_mask_status"][row_index])
            if saved_status != actual_status:
                raise ValueError(
                    "CVAT mask status differs from saved frame provenance: "
                    f"{video_name}/{stem}/{saved_status}/{actual_status}"
                )
            if positive_pixels != int(
                manual_frames["cvat_mask_positive_pixels"][row_index]
            ):
                raise ValueError(
                    "CVAT mask pixel count differs from saved frame provenance: "
                    f"{video_name}/{stem}"
                )

            point_selector = data["point_frame_order"] == order
            frame_xy = rounded_xy[point_selector]
            frame_labels = data["point_label"][point_selector]
            status_counts[actual_status] += 1
            if order in invalidation_orders:
                if not np.all(frame_labels == LABEL_BACKGROUND):
                    raise ValueError(
                        "XML-invalidated CVAT frame is not all background: "
                        f"{video_name}/{stem}"
                    )
                expected_counts = {
                    "positive_points": 0,
                    "ignore_points": 0,
                    "background_points": int(frame_labels.size),
                    "positive_outside_cvat_mask_points": 0,
                }
                for name, expected in expected_counts.items():
                    if int(manual_frames[name][row_index]) != expected:
                        raise ValueError(
                            "XML-invalidated frame provenance count mismatch: "
                            f"{video_name}/{stem}/{name}"
                        )
                statuses_by_order[order] = SUPPRESSED_CVAT_STATUS
                outside_by_order[order] = 0
                mismatch_by_order[order] = 0
                suppressed_by_order[order] = positive_pixels
                rendered_status_counts[SUPPRESSED_CVAT_STATUS] += 1
                continue
            sampled_mask = (
                mask[frame_xy[:, 1], frame_xy[:, 0]]
                if frame_xy.size
                else np.zeros((0,), dtype=bool)
            )
            positive = frame_labels == LABEL_FEMUR_CANDIDATE
            outside_count = int(np.sum(positive & ~sampled_mask))
            mismatch_count = int(np.sum(positive != sampled_mask))
            saved_counts = {
                "positive_points": int(np.sum(positive)),
                "ignore_points": int(np.sum(frame_labels == LABEL_IGNORE)),
                "background_points": int(
                    np.sum(frame_labels == LABEL_BACKGROUND)
                ),
            }
            for name, actual in saved_counts.items():
                if actual != int(manual_frames[name][row_index]):
                    raise ValueError(
                        "Saved point count differs from authoritative frame "
                        f"provenance: {video_name}/{stem}/{name}"
                    )
            saved_outside = int(
                manual_frames["positive_outside_cvat_mask_points"][row_index]
            )
            if outside_count or saved_outside:
                raise ValueError(
                    "Positive point centers lie outside authoritative CVAT mask: "
                    f"{video_name}/{stem}/{outside_count}"
                )
            if mismatch_count:
                raise ValueError(
                    "Saved positive labels differ from authoritative CVAT mask at "
                    f"sampled point centers: {video_name}/{stem}/{mismatch_count}"
                )
            masks_by_order[order] = mask
            stems_by_order[order] = stem
            statuses_by_order[order] = actual_status
            outside_by_order[order] = outside_count
            mismatch_by_order[order] = mismatch_count
            rendered_status_counts[actual_status] += 1
        return masks_by_order, {
            "reviewed_video": True,
            "task_id": task_id,
            "annotation_zip": str(annotation_zip),
            "annotation_zip_sha256": annotation_sha,
            "applied_bbox_rows": num_rows,
            "applied_frame_stems": dict(sorted(stems_by_order.items())),
            "label_authority": (
                "cvat_snapshot_with_xml_invalidation"
                if invalidation_orders
                else AUTHORITATIVE_LABEL_AUTHORITY
            ),
            "authoritative_frame_count": num_frame_rows,
            "cvat_mask_status_counts": dict(sorted(status_counts.items())),
            "rendered_cvat_mask_status_counts": dict(
                sorted(rendered_status_counts.items())
            ),
            "cvat_mask_status_by_order": dict(sorted(statuses_by_order.items())),
            "suppressed_cvat_mask_frames": len(suppressed_by_order),
            "suppressed_cvat_mask_pixels": int(sum(suppressed_by_order.values())),
            "suppressed_cvat_mask_pixels_by_order": dict(
                sorted(suppressed_by_order.items())
            ),
            "positive_outside_cvat_mask_points": int(sum(outside_by_order.values())),
            "positive_outside_cvat_mask_points_by_order": dict(
                sorted(outside_by_order.items())
            ),
            "cvat_point_mask_mismatch_points": int(sum(mismatch_by_order.values())),
            "cvat_point_mask_mismatch_points_by_order": dict(
                sorted(mismatch_by_order.items())
            ),
        }

    masks_by_order: dict[int, np.ndarray] = {}
    stems_by_order: dict[int, str] = {}
    decisions_by_order: dict[int, set[str]] = defaultdict(set)
    for stem_value, order_value, index_value, decision_value in zip(
        manual["frame_stem"],
        manual["frame_order"],
        manual["frame_index"],
        manual["decision"],
        strict=True,
    ):
        stem = str(stem_value)
        order = int(order_value)
        index = int(index_value)
        if index != int(data["frame_indices"][order]):
            raise ValueError(f"Manual CVAT frame index mismatch: {video_name}/{stem}")
        if stem not in masks:
            raise KeyError(f"Manual CVAT mask stem is missing: {video_name}/{stem}")
        if order in stems_by_order and stems_by_order[order] != stem:
            raise ValueError(f"Multiple CVAT stems map to frame order {order}")
        stems_by_order[order] = stem
        masks_by_order[order] = np.asarray(masks[stem], dtype=bool)
        decision = str(decision_value)
        if decision not in {"manual_positive", "manual_empty"}:
            raise ValueError(f"Unknown manual CVAT decision: {decision}")
        decisions_by_order[order].add(decision)
    for order, decisions in decisions_by_order.items():
        has_pixels = bool(np.any(masks_by_order[order]))
        if "manual_positive" in decisions and not has_pixels:
            raise ValueError(
                f"manual_positive frame has an empty CVAT mask: {video_name}/{order}"
            )
        if decisions == {"manual_empty"} and has_pixels:
            raise ValueError(
                f"manual_empty frame has a non-empty CVAT mask: {video_name}/{order}"
            )
    return masks_by_order, {
        "reviewed_video": True,
        "task_id": task_id,
        "annotation_zip": str(annotation_zip),
        "annotation_zip_sha256": annotation_sha,
        "applied_bbox_rows": num_rows,
        "applied_frame_stems": dict(sorted(stems_by_order.items())),
        "label_authority": "legacy_bbox_rows",
        "authoritative_frame_count": 0,
        "positive_outside_cvat_mask_points": 0,
        "cvat_point_mask_mismatch_points": 0,
    }


def _blend_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if not np.any(mask):
        return
    color_array = np.asarray(color, dtype=np.float32)
    blended = (
        (1.0 - alpha) * rgb[mask].astype(np.float32) + alpha * color_array
    )
    rgb[mask] = np.clip(blended, 0, 255).astype(np.uint8)


def render_point_labels(
    gray: np.ndarray,
    *,
    pixel_xy: np.ndarray,
    labels: np.ndarray,
    radius: int,
    alpha: float,
    positive_color: tuple[int, int, int],
    ignore_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    cvat_mask: np.ndarray | None = None,
    cvat_color: tuple[int, int, int] = (0, 255, 255),
    cvat_alpha: float = 0.35,
) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError(f"gray image must be 2-D, got {gray.shape}")
    pixel_xy = np.asarray(pixel_xy, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    if pixel_xy.shape != (labels.size, 2):
        raise ValueError("pixel_xy/labels shape mismatch")
    radius = int(radius)
    if radius < 0:
        raise ValueError("point radius must be >= 0")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("point alpha must be in [0,1]")
    cvat_alpha = float(cvat_alpha)
    if not 0.0 <= cvat_alpha <= 1.0:
        raise ValueError("CVAT mask alpha must be in [0,1]")
    rgb = np.stack([gray, gray, gray], axis=-1)
    height, width = gray.shape
    xy = np.rint(pixel_xy).astype(np.int64)
    kernel = None
    if radius > 0:
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # Draw the dense CVAT mask after ignore points and before positive points.
    for label, color in (
        (LABEL_BACKGROUND, background_color),
        (LABEL_IGNORE, ignore_color),
    ):
        selected = labels == label
        marker = np.zeros((height, width), dtype=np.uint8)
        if np.any(selected):
            marker[xy[selected, 1], xy[selected, 0]] = 1
            if kernel is not None:
                marker = cv2.dilate(marker, kernel)
        _blend_mask(rgb, marker.astype(bool), color=color, alpha=alpha)
    if cvat_mask is not None:
        cvat_mask = np.asarray(cvat_mask)
        if cvat_mask.shape != gray.shape:
            raise ValueError(
                f"CVAT mask/image shape mismatch: {cvat_mask.shape} != {gray.shape}"
            )
        _blend_mask(rgb, cvat_mask.astype(bool), color=cvat_color, alpha=cvat_alpha)
    positive = labels == LABEL_FEMUR_CANDIDATE
    marker = np.zeros((height, width), dtype=np.uint8)
    if np.any(positive):
        marker[xy[positive, 1], xy[positive, 0]] = 1
        if kernel is not None:
            marker = cv2.dilate(marker, kernel)
    _blend_mask(rgb, marker.astype(bool), color=positive_color, alpha=alpha)
    return rgb


def _draw_saved_bbox(
    rgb: np.ndarray,
    bbox: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width - 1, int(round(x2))))
    bottom = max(0, min(height - 1, int(round(y2))))
    if right <= left or bottom <= top:
        return
    cv2.rectangle(rgb, (left, top), (right, bottom), color, int(thickness))


def _bbox_text(bbox: np.ndarray) -> str:
    return "|".join(f"{float(value):.8g}" for value in bbox)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def export_stage4_point_label_visualization(
    annotated_h5: Path,
    output_dir: Path,
    *,
    pseudo3d_h5: Path | None = None,
    point_radius: int = 1,
    point_alpha: float = 0.85,
    positive_color: tuple[int, int, int] = (255, 32, 32),
    ignore_color: tuple[int, int, int] = (255, 210, 0),
    background_color: tuple[int, int, int] = (80, 140, 200),
    bbox_color: tuple[int, int, int] = (160, 255, 80),
    bbox_thickness: int = 1,
    cvat_review_root: Path | None = None,
    cvat_snapshot_root: Path | None = None,
    cvat_mask_color: tuple[int, int, int] = (0, 255, 255),
    cvat_mask_alpha: float = 0.35,
    require_cvat_authoritative: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.is_symlink():
        raise ValueError(f"output_dir must not be a symlink: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output_dir is not empty; pass --overwrite: {output_dir}")
    if bbox_thickness <= 0:
        raise ValueError("bbox_thickness must be > 0")

    data = load_stage4_point_label_visualization_input(
        annotated_h5, pseudo3d_h5=pseudo3d_h5
    )
    if require_cvat_authoritative and not data["cvat_authoritative"]:
        raise ValueError(
            "Annotated H5 is not a supported CVAT-authoritative teacher"
        )
    if (cvat_review_root is None) != (cvat_snapshot_root is None):
        raise ValueError(
            "cvat_review_root and cvat_snapshot_root must be specified together"
        )
    if cvat_review_root is not None:
        cvat_masks, cvat_meta = load_applied_cvat_masks(
            data,
            review_root=cvat_review_root,
            snapshot_root=cvat_snapshot_root,
        )
        cvat_masks_requested = True
    else:
        if require_cvat_authoritative:
            raise ValueError(
                "Authoritative visualization requires CVAT review/snapshot roots"
            )
        cvat_masks = {}
        cvat_meta = {"reviewed_video": False}
        cvat_masks_requested = False
    input_sha = file_sha256(data["annotated_h5"])
    pseudo3d_sha = file_sha256(data["pseudo3d_h5"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        frames_dir = temporary / "frames"
        frames_dir.mkdir(parents=True)
        frame_annotation = data["frame_annotation"]
        bbox_rows_by_order: dict[int, list[int]] = defaultdict(list)
        for row_index, order in enumerate(frame_annotation["frame_order"]):
            bbox_rows_by_order[int(order)].append(row_index)
        for order in bbox_rows_by_order:
            bbox_rows_by_order[order].sort(
                key=lambda index: int(frame_annotation["bbox_index"][index])
            )

        rows: list[dict[str, Any]] = []
        label_totals = Counter()
        source_counts = Counter()
        cvat_mask_pixels = 0
        status_by_order = {
            int(order): str(status)
            for order, status in cvat_meta.get(
                "cvat_mask_status_by_order", {}
            ).items()
        }
        outside_by_order = {
            int(order): int(count)
            for order, count in cvat_meta.get(
                "positive_outside_cvat_mask_points_by_order", {}
            ).items()
        }
        mismatch_by_order = {
            int(order): int(count)
            for order, count in cvat_meta.get(
                "cvat_point_mask_mismatch_points_by_order", {}
            ).items()
        }
        suppressed_by_order = {
            int(order): int(count)
            for order, count in cvat_meta.get(
                "suppressed_cvat_mask_pixels_by_order", {}
            ).items()
        }
        xml_invalidations = data["xml_invalidations"]
        invalidation_by_order = {
            int(order): {
                "action": str(action),
                "reason_code": str(reason),
            }
            for order, action, reason in zip(
                xml_invalidations["frame_order"],
                xml_invalidations["action"],
                xml_invalidations["reason_code"],
                strict=True,
            )
        }
        frame_authority_by_order = {
            int(order): str(authority)
            for order, authority in zip(
                data["manual_review_frames"]["frame_order"],
                data["manual_review_frames"]["label_authority"],
                strict=True,
            )
        }
        num_frames = int(data["local_images"].shape[0])
        sorted_point_indices = np.argsort(
            data["point_frame_order"], kind="stable"
        )
        points_per_frame = np.bincount(
            data["point_frame_order"], minlength=num_frames
        )
        point_offsets = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(points_per_frame))
        )
        for frame_order in range(num_frames):
            point_indices = sorted_point_indices[
                point_offsets[frame_order] : point_offsets[frame_order + 1]
            ]
            labels = data["point_label"][point_indices]
            gray = image_to_uint8_gray(data["local_images"][frame_order])
            cvat_mask = cvat_masks.get(frame_order)
            rgb = render_point_labels(
                gray,
                pixel_xy=data["pixel_xy"][point_indices],
                labels=labels,
                radius=point_radius,
                alpha=point_alpha,
                positive_color=positive_color,
                ignore_color=ignore_color,
                background_color=background_color,
                cvat_mask=cvat_mask,
                cvat_color=cvat_mask_color,
                cvat_alpha=cvat_mask_alpha,
            )
            bbox_row_indices = bbox_rows_by_order.get(frame_order, [])
            for row_index in bbox_row_indices:
                _draw_saved_bbox(
                    rgb,
                    frame_annotation["bbox_local_xyxy"][row_index],
                    color=bbox_color,
                    thickness=bbox_thickness,
                )
            relative_image = Path("frames") / f"annotation_frame_{frame_order:05d}.png"
            imageio.imwrite(temporary / relative_image, rgb)

            counts = {
                "num_points": int(labels.size),
                "positive_points": int(np.sum(labels == LABEL_FEMUR_CANDIDATE)),
                "ignore_points": int(np.sum(labels == LABEL_IGNORE)),
                "background_points": int(np.sum(labels == LABEL_BACKGROUND)),
                "label_authority": frame_authority_by_order.get(
                    frame_order,
                    XML_INVALIDATION_LABEL_AUTHORITY
                    if frame_order in invalidation_by_order
                    else "inherited",
                ),
                "xml_invalidated": int(frame_order in invalidation_by_order),
                "xml_invalidation_action": invalidation_by_order.get(
                    frame_order, {}
                ).get("action", ""),
                "xml_invalidation_reason_code": invalidation_by_order.get(
                    frame_order, {}
                ).get("reason_code", ""),
                "cvat_mask_applied": int(cvat_mask is not None),
                "cvat_mask_suppressed": int(frame_order in suppressed_by_order),
                "cvat_mask_status": status_by_order.get(frame_order, ""),
                "cvat_mask_pixels": (
                    int(np.sum(cvat_mask)) if cvat_mask is not None else 0
                ),
                "positive_outside_cvat_mask_points": outside_by_order.get(
                    frame_order, 0
                ),
                "cvat_point_mask_mismatch_points": mismatch_by_order.get(
                    frame_order, 0
                ),
            }
            cvat_mask_pixels += counts["cvat_mask_pixels"]
            label_totals.update(
                positive_points=counts["positive_points"],
                ignore_points=counts["ignore_points"],
                background_points=counts["background_points"],
            )
            common = {
                "video_name": data["video_name"],
                "frame_order": frame_order,
                "frame_index": int(data["frame_indices"][frame_order]),
                **counts,
                "rendered_image": relative_image.as_posix(),
            }
            if not bbox_row_indices:
                rows.append(
                    {
                        **common,
                        "bbox_index": "",
                        "bbox_local_xyxy": "",
                        "selected_contour_source": "",
                        "annotation_reason": "",
                        "valid_contour": "",
                    }
                )
            for row_index in bbox_row_indices:
                source = str(
                    frame_annotation["selected_contour_source"][row_index]
                )
                source_counts[source] += 1
                rows.append(
                    {
                        **common,
                        "bbox_index": int(frame_annotation["bbox_index"][row_index]),
                        "bbox_local_xyxy": _bbox_text(
                            frame_annotation["bbox_local_xyxy"][row_index]
                        ),
                        "selected_contour_source": source,
                        "annotation_reason": str(
                            frame_annotation["annotation_reason"][row_index]
                        ),
                        "valid_contour": int(
                            bool(frame_annotation["valid_contour"][row_index])
                        ),
                    }
                )

        _write_csv(temporary / "frame_labels.csv", rows)
        summary = {
            "status": "ok",
            "schema_version": 2,
            "visualization_kind": "saved_stage4_point_labels",
            "automatic_contours_recomputed": False,
            "input_files_modified": False,
            "video_name": data["video_name"],
            "contour_teacher_schema": _attr_text(
                data["annotated_attrs"], "contour_teacher_schema"
            ),
            "annotated_h5": str(data["annotated_h5"]),
            "annotated_h5_sha256": input_sha,
            "pseudo3d_h5": str(data["pseudo3d_h5"]),
            "pseudo3d_h5_sha256": pseudo3d_sha,
            "num_frames": num_frames,
            "num_frame_csv_rows": len(rows),
            "num_bbox_rows": int(frame_annotation["frame_order"].size),
            "num_points": int(data["point_label"].size),
            "positive_points": int(label_totals["positive_points"]),
            "ignore_points": int(label_totals["ignore_points"]),
            "background_points": int(label_totals["background_points"]),
            "cvat_masks_requested": cvat_masks_requested,
            "cvat_mask_frames": len(cvat_masks),
            "cvat_mask_nonempty_frames": sum(
                bool(np.any(mask)) for mask in cvat_masks.values()
            ),
            "cvat_mask_pixels": int(cvat_mask_pixels),
            "cvat_label_authority": cvat_meta.get("label_authority", "none"),
            "cvat_authoritative_frames": int(
                cvat_meta.get("authoritative_frame_count", 0)
            ),
            "cvat_mask_status_counts": cvat_meta.get(
                "cvat_mask_status_counts", {}
            ),
            "rendered_cvat_mask_status_counts": cvat_meta.get(
                "rendered_cvat_mask_status_counts", {}
            ),
            "suppressed_cvat_mask_frames": int(
                cvat_meta.get("suppressed_cvat_mask_frames", 0)
            ),
            "suppressed_cvat_mask_pixels": int(
                cvat_meta.get("suppressed_cvat_mask_pixels", 0)
            ),
            "positive_outside_cvat_mask_points": int(
                cvat_meta.get("positive_outside_cvat_mask_points", 0)
            ),
            "cvat_point_mask_mismatch_points": int(
                cvat_meta.get("cvat_point_mask_mismatch_points", 0)
            ),
            "cvat_mask_source": cvat_meta,
            "source_counts": dict(sorted(source_counts.items())),
            "xml_invalidation_frames": len(invalidation_by_order),
            "xml_invalidation_manifest_sha256": _attr_text(
                xml_invalidations["attrs"], "manifest_sha256"
            ),
            "xml_invalidation_fingerprint": _attr_text(
                xml_invalidations["attrs"], "manifest_fingerprint"
            ),
            "point_radius": int(point_radius),
            "point_alpha": float(point_alpha),
            "positive_color_rgb": list(positive_color),
            "ignore_color_rgb": list(ignore_color),
            "background_color_rgb": list(background_color),
            "bbox_color_rgb": list(bbox_color),
            "cvat_mask_color_rgb": list(cvat_mask_color),
            "cvat_mask_alpha": float(cvat_mask_alpha),
        }
        _write_json(temporary / "summary.json", summary)
        if file_sha256(data["annotated_h5"]) != input_sha:
            raise RuntimeError("Annotated input H5 changed during visualization")
        if file_sha256(data["pseudo3d_h5"]) != pseudo3d_sha:
            raise RuntimeError("Source pseudo3D H5 changed during visualization")
        cvat_zip = cvat_meta.get("annotation_zip")
        if cvat_zip and file_sha256(Path(cvat_zip)) != cvat_meta.get(
            "annotation_zip_sha256"
        ):
            raise RuntimeError("CVAT annotation ZIP changed during visualization")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render saved Stage 4 point_label values and saved BBoxes without "
            "recomputing automatic contours."
        )
    )
    parser.add_argument("--annotated_h5", type=Path, required=True)
    parser.add_argument("--pseudo3d_h5", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--point_alpha", type=float, default=0.85)
    parser.add_argument("--positive_color", default="255,32,32")
    parser.add_argument("--ignore_color", default="255,210,0")
    parser.add_argument("--background_color", default="80,140,200")
    parser.add_argument("--bbox_color", default="160,255,80")
    parser.add_argument("--bbox_thickness", type=int, default=1)
    parser.add_argument("--cvat_review_root", type=Path, default=None)
    parser.add_argument("--cvat_snapshot_root", type=Path, default=None)
    parser.add_argument("--cvat_mask_color", default="0,255,255")
    parser.add_argument("--cvat_mask_alpha", type=float, default=0.35)
    parser.add_argument("--require_cvat_authoritative", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = export_stage4_point_label_visualization(
            args.annotated_h5,
            args.output_dir,
            pseudo3d_h5=args.pseudo3d_h5,
            point_radius=args.point_radius,
            point_alpha=args.point_alpha,
            positive_color=parse_color(args.positive_color),
            ignore_color=parse_color(args.ignore_color),
            background_color=parse_color(args.background_color),
            bbox_color=parse_color(args.bbox_color),
            bbox_thickness=args.bbox_thickness,
            cvat_review_root=args.cvat_review_root,
            cvat_snapshot_root=args.cvat_snapshot_root,
            cvat_mask_color=parse_color(args.cvat_mask_color),
            cvat_mask_alpha=args.cvat_mask_alpha,
            require_cvat_authoritative=args.require_cvat_authoritative,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        raise SystemExit(
            f"Stage 4 point-label visualization failed: {type(exc).__name__}: {exc}"
        ) from exc
    print("Stage 4 point-label visualization passed.")
    print(f"  video       : {summary['video_name']}")
    print(f"  frames      : {summary['num_frames']}")
    print(f"  points      : {summary['num_points']}")
    print(f"  positive    : {summary['positive_points']}")
    print(f"  output      : {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
