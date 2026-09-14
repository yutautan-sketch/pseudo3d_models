from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_contour_teacher import (
    load_audit_teacher_config,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    Stage4SweepManifestItem,
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    load_voc_bboxes,
    xml_bbox_to_local,
)
from pseudo3d.annotation.stage4_manual_review import (
    prepare_output_root,
    write_csv_rows,
    write_json,
)
from pseudo3d.batch.export.batch_export_stage4_manual_review_cvat import (
    _bbox_crop_metrics,
    _project_bbox_to_local_unclipped,
)


V6_TOKEN = "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
DEFAULT_V6_GLOB = "{video_name}/*bboxrank_v6_cvat_authoritative_xml_invalidation_v1.h5"


class CropQualityAuditError(RuntimeError):
    """Raised when the read-only crop/label audit contract is not satisfied."""


@dataclass(frozen=True)
class TargetSpec:
    video_name: str
    frame_orders: tuple[int, ...]


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _parse_target(value: str) -> TargetSpec:
    video, separator, frames_text = str(value).partition(":")
    video = video.strip()
    if not separator or not video or not frames_text.strip():
        raise argparse.ArgumentTypeError(
            "--target must be VIDEO:FRAME_ORDER[,FRAME_ORDER...]"
        )
    try:
        frames = tuple(sorted({int(token) for token in frames_text.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid target frame list: {value!r}") from exc
    if not frames or any(frame < 0 for frame in frames):
        raise argparse.ArgumentTypeError(f"Target frame orders must be non-negative: {value!r}")
    return TargetSpec(video_name=video, frame_orders=frames)


def _local_shape(image_shape: Sequence[int]) -> tuple[int, int]:
    shape = tuple(int(value) for value in image_shape)
    if len(shape) == 3:
        return shape[1], shape[2]
    if len(shape) == 4 and shape[1] == 1:
        return shape[2], shape[3]
    raise CropQualityAuditError(f"Unsupported local frame layout: {shape}")


def _pipe(values: Sequence[float]) -> str:
    return "|".join(str(float(value)) for value in values)


def _single_h5(root: Path, video: str, template: str) -> Path:
    try:
        pattern = template.format(video_name=video)
    except KeyError as exc:
        raise CropQualityAuditError(
            "--annotated_glob_template only supports {video_name}"
        ) from exc
    matches = sorted(path for path in Path(root).glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one v6 H5 for {video}: root={root}, pattern={pattern}, "
            f"matches={len(matches)}"
        )
    return matches[0].resolve()


def _strict_bbox_inside(
    *,
    frame_orders: np.ndarray,
    pixel_xy: np.ndarray,
    bbox_orders: np.ndarray,
    bboxes: np.ndarray,
) -> np.ndarray:
    """Mirror the Stage 4/5 strict saved-BBox membership contract."""
    result = np.zeros(frame_orders.shape, dtype=bool)
    rounded = np.rint(pixel_xy).astype(np.int64)
    for order in np.unique(bbox_orders):
        point_indices = np.flatnonzero(frame_orders == int(order))
        if not point_indices.size:
            continue
        xy = rounded[point_indices]
        inside = np.zeros(point_indices.size, dtype=bool)
        for bbox in bboxes[bbox_orders == int(order)]:
            if not np.all(np.isfinite(bbox)):
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            left = int(math.floor(max(0.0, x1)))
            top = int(math.floor(max(0.0, y1)))
            right = int(math.ceil(x2))
            bottom = int(math.ceil(y2))
            if right <= left or bottom <= top:
                continue
            inside |= (
                (xy[:, 0] >= left)
                & (xy[:, 0] <= right)
                & (xy[:, 1] >= top)
                & (xy[:, 1] <= bottom)
            )
        result[point_indices] = inside
    return result


def _permissive_bbox_inside(
    *,
    frame_orders: np.ndarray,
    pixel_xy: np.ndarray,
    bbox_orders: np.ndarray,
    bboxes: np.ndarray,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Reproduce the pre-fix CVAT bbox behavior, including 1-pixel lines.

    This is deliberately independent of the current ``bbox_bounds`` helper so
    the read-only audit remains able to explain labels written by older code.
    """
    result = np.zeros(frame_orders.shape, dtype=bool)
    rounded = np.rint(pixel_xy).astype(np.int64)
    height, width = shape_hw
    for order in np.unique(bbox_orders):
        point_indices = np.flatnonzero(frame_orders == int(order))
        if not point_indices.size:
            continue
        xy = rounded[point_indices]
        inside = np.zeros(point_indices.size, dtype=bool)
        for bbox in bboxes[bbox_orders == order]:
            if not np.all(np.isfinite(bbox)):
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            left = int(math.floor(max(0.0, x1)))
            top = int(math.floor(max(0.0, y1)))
            right = int(math.ceil(min(float(width - 1), x2)))
            bottom = int(math.ceil(min(float(height - 1), y2)))
            if right < left or bottom < top:
                continue
            inside |= (
                (xy[:, 0] >= left)
                & (xy[:, 0] <= right)
                & (xy[:, 1] >= top)
                & (xy[:, 1] <= bottom)
            )
        result[point_indices] = inside
    return result


def _read_pseudo(path: Path) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        for name in ("local_encoder_images", "frame_indices"):
            if name not in handle:
                raise CropQualityAuditError(f"pseudo3D H5 lacks {name}: {path}")
        image_shape = tuple(int(value) for value in handle["local_encoder_images"].shape)
        indices = handle["frame_indices"][:].astype(np.int64)
        attrs = dict(handle.attrs)
    if image_shape[0] != len(indices) or len(np.unique(indices)) != len(indices):
        raise CropQualityAuditError(f"Invalid pseudo3D frame inventory: {path}")
    return indices, _local_shape(image_shape), attrs


def _annotation_values(group: h5py.Group, name: str, rows: int, default: Any) -> list[Any]:
    if name not in group:
        return [default] * rows
    values = group[name][:]
    if len(values) != rows:
        raise CropQualityAuditError(f"frame_annotation/{name} length mismatch")
    return values.tolist()


def _read_v6(path: Path, video: str, shape_hw: tuple[int, int]) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if _decode(handle.attrs.get("contour_teacher_schema", "")) != V6_TOKEN:
            raise CropQualityAuditError(f"Unexpected teacher schema: {path}")
        if _decode(handle.attrs.get("video_name", "")) != video:
            raise CropQualityAuditError(f"H5 video_name mismatch: {path}")
        required = (
            "point_cloud/frame_order",
            "point_cloud/pixel_xy",
            "annotation/point_label",
            "annotation/valid_mask",
            "frame_annotation/frame_order",
            "frame_annotation/frame_index",
            "frame_annotation/bbox_index",
            "frame_annotation/bbox_local_xyxy",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise CropQualityAuditError(f"v6 H5 lacks datasets {missing}: {path}")
        frame_orders = handle["point_cloud/frame_order"][:].astype(np.int64)
        pixel_xy = handle["point_cloud/pixel_xy"][:].astype(np.float64)
        labels = handle["annotation/point_label"][:].astype(np.int8)
        valid = handle["annotation/valid_mask"][:].astype(bool)
        frame = handle["frame_annotation"]
        bbox_orders = frame["frame_order"][:].astype(np.int64)
        bbox_indices = frame["bbox_index"][:].astype(np.int64)
        bbox_frame_indices = frame["frame_index"][:].astype(np.int64)
        bboxes = frame["bbox_local_xyxy"][:].astype(np.float64)
        rows = len(bbox_orders)
        selected_sources = [_decode(value) for value in _annotation_values(frame, "selected_contour_source", rows, "")]
        reasons = [_decode(value) for value in _annotation_values(frame, "annotation_reason", rows, "")]
        valid_contours = [bool(value) for value in _annotation_values(frame, "valid_contour", rows, False)]
        xml_paths = [_decode(value) for value in _annotation_values(frame, "xml_path", rows, "")]
        authorities: dict[int, str] = {}
        mask_statuses: dict[int, str] = {}
        if "manual_review_fullvideo_frames" in handle:
            provenance = handle["manual_review_fullvideo_frames"]
            if "frame_order" in provenance:
                prov_orders = provenance["frame_order"][:].astype(np.int64)
                authority_values = _annotation_values(provenance, "label_authority", len(prov_orders), "")
                status_values = _annotation_values(provenance, "cvat_mask_status", len(prov_orders), "")
                authorities = {
                    int(order): _decode(authority)
                    for order, authority in zip(prov_orders, authority_values, strict=True)
                }
                mask_statuses = {
                    int(order): _decode(status)
                    for order, status in zip(prov_orders, status_values, strict=True)
                }
        source_v5 = ""
        if "xml_annotation_invalidation" in handle:
            source_v5 = _decode(handle["xml_annotation_invalidation"].attrs.get("source_h5", ""))

    if not (frame_orders.shape == labels.shape == valid.shape):
        raise CropQualityAuditError(f"Point label arrays are not aligned: {path}")
    if pixel_xy.shape != (labels.size, 2):
        raise CropQualityAuditError(f"pixel_xy has wrong shape: {path}")
    if not np.array_equal(valid, labels != -1):
        raise CropQualityAuditError(f"valid_mask differs from point_label: {path}")
    if bboxes.shape != (rows, 4):
        raise CropQualityAuditError(f"bbox_local_xyxy has wrong shape: {path}")
    height, width = shape_hw
    rounded = np.rint(pixel_xy).astype(np.int64)
    if rounded.size and (
        np.any(rounded[:, 0] < 0)
        or np.any(rounded[:, 0] >= width)
        or np.any(rounded[:, 1] < 0)
        or np.any(rounded[:, 1] >= height)
    ):
        raise CropQualityAuditError(f"Rounded pixel_xy escapes local image: {path}")
    strict = _strict_bbox_inside(
        frame_orders=frame_orders,
        pixel_xy=pixel_xy,
        bbox_orders=bbox_orders,
        bboxes=bboxes,
    )
    permissive = _permissive_bbox_inside(
        frame_orders=frame_orders,
        pixel_xy=pixel_xy,
        bbox_orders=bbox_orders,
        bboxes=bboxes,
        shape_hw=shape_hw,
    )
    stray = (labels == -1) & ~strict
    explained = stray & permissive
    return {
        "frame_orders": frame_orders,
        "pixel_xy": pixel_xy,
        "rounded_xy": rounded,
        "labels": labels,
        "bbox_orders": bbox_orders,
        "bbox_indices": bbox_indices,
        "bbox_frame_indices": bbox_frame_indices,
        "bboxes": bboxes,
        "selected_sources": selected_sources,
        "reasons": reasons,
        "valid_contours": valid_contours,
        "xml_paths": xml_paths,
        "authorities": authorities,
        "mask_statuses": mask_statuses,
        "strict_inside": strict,
        "permissive_inside": permissive,
        "stray": stray,
        "explained": explained,
        "source_v5": source_v5,
    }


def _read_source_v5(
    path: Path,
    *,
    shape_hw: tuple[int, int],
    expected: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        required = (
            "point_cloud/frame_order",
            "point_cloud/pixel_xy",
            "annotation/point_label",
            "frame_annotation/frame_order",
            "frame_annotation/bbox_local_xyxy",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise CropQualityAuditError(f"source v5 H5 lacks datasets {missing}: {path}")
        frame_orders = handle["point_cloud/frame_order"][:].astype(np.int64)
        pixel_xy = handle["point_cloud/pixel_xy"][:].astype(np.float64)
        labels = handle["annotation/point_label"][:].astype(np.int8)
        bbox_orders = handle["frame_annotation/frame_order"][:].astype(np.int64)
        bboxes = handle["frame_annotation/bbox_local_xyxy"][:].astype(np.float64)
    if not np.array_equal(frame_orders, expected["frame_orders"]):
        raise CropQualityAuditError(f"v5/v6 point frame order differs: {path}")
    if not np.array_equal(pixel_xy, expected["pixel_xy"]):
        raise CropQualityAuditError(f"v5/v6 point pixel_xy differs: {path}")
    if labels.shape != expected["labels"].shape:
        raise CropQualityAuditError(f"v5/v6 point label shape differs: {path}")
    strict = _strict_bbox_inside(
        frame_orders=frame_orders,
        pixel_xy=pixel_xy,
        bbox_orders=bbox_orders,
        bboxes=bboxes,
    )
    return {
        "labels": labels,
        "stray": (labels == -1) & ~strict,
    }


def _bbox_row(
    *,
    video: str,
    frame_order: int,
    frame_index: int,
    bbox_index: int,
    voc: Any,
    attrs: Mapping[str, Any],
    shape_hw: tuple[int, int],
    h5: Mapping[str, Any] | None,
) -> dict[str, Any]:
    local = xml_bbox_to_local(
        voc.xml_xyxy,
        image_size_wh=voc.image_size_wh,
        attrs=dict(attrs),
        local_shape_hw=shape_hw,
    )
    projected = _project_bbox_to_local_unclipped(
        raw_xyxy=local.raw_xyxy,
        attrs=attrs,
        local_shape_hw=shape_hw,
    )
    crop = _bbox_crop_metrics(projected_xyxy=projected, local_shape_hw=shape_hw)
    saved_matches: list[int] = []
    if h5 is not None:
        saved_matches = np.flatnonzero(
            (h5["bbox_orders"] == frame_order)
            & (h5["bbox_frame_indices"] == frame_index)
            & (h5["bbox_indices"] == bbox_index)
        ).tolist()
    if len(saved_matches) > 1:
        raise CropQualityAuditError(
            f"Duplicate saved BBox: {video}/{frame_order}/{frame_index}/{bbox_index}"
        )
    saved = h5["bboxes"][saved_matches[0]] if saved_matches else None
    saved_width = float(saved[2] - saved[0]) if saved is not None else math.nan
    saved_height = float(saved[3] - saved[1]) if saved is not None else math.nan
    saved_degenerate = bool(
        saved is not None
        and np.all(np.isfinite(saved))
        and (saved_width <= 0.0 or saved_height <= 0.0)
    )
    row_index = saved_matches[0] if saved_matches else None
    return {
        "video_name": video,
        "frame_order": frame_order,
        "frame_index": frame_index,
        "bbox_index": bbox_index,
        "xml_path": str(voc.xml_path.resolve()),
        "bbox_xml_xyxy": _pipe(voc.xml_xyxy),
        "bbox_raw_xyxy": _pipe(local.raw_xyxy),
        "projected_bbox_xyxy": _pipe(crop["projected_bbox_xyxy"]),
        "clipped_bbox_xyxy": _pipe(crop["clipped_bbox_xyxy"]),
        "local_bbox_xyxy": _pipe(local.local_xyxy),
        "local_bbox_valid": local.valid,
        "crop_status": crop["crop_status"],
        "visible_fraction": crop["visible_fraction"],
        "touches_left": crop["touches_left"],
        "touches_right": crop["touches_right"],
        "touches_top": crop["touches_top"],
        "touches_bottom": crop["touches_bottom"],
        "saved_bbox_present": saved is not None,
        "saved_bbox_xyxy": _pipe(saved) if saved is not None else "",
        "saved_bbox_width": saved_width,
        "saved_bbox_height": saved_height,
        "saved_bbox_degenerate": saved_degenerate,
        "saved_valid_contour": h5["valid_contours"][row_index] if row_index is not None else "",
        "saved_selected_source": h5["selected_sources"][row_index] if row_index is not None else "",
        "saved_annotation_reason": h5["reasons"][row_index] if row_index is not None else "",
        "saved_xml_path": h5["xml_paths"][row_index] if row_index is not None else "",
    }


def _source_files(
    item: Stage4SweepManifestItem,
    teacher: Any,
    annotated_h5: Path | None,
    source_v5: str,
) -> list[tuple[str, Path]]:
    rows = [("pseudo3d_h5", item.pseudo3d_h5.resolve())]
    xml_dir = item.voc_xml_root / item.video_name / teacher.xml_annotation_dir_name
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"Strict XML directory not found: {xml_dir}")
    rows.extend(("strict_xml", path.resolve()) for path in sorted(xml_dir.glob("*.xml")))
    if annotated_h5 is not None:
        rows.append(("v6_annotated_h5", annotated_h5.resolve()))
    if source_v5:
        source = Path(source_v5).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"v6 source v5 H5 not found: {source}")
        rows.append(("v5_source_h5", source))
    return rows


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    targets: list[TargetSpec] = list(args.target)
    if len({item.video_name for item in targets}) != len(targets):
        raise CropQualityAuditError("Duplicate --target video")
    comparison_videos = tuple(dict.fromkeys(str(value) for value in args.comparison_video))
    overlap = sorted({item.video_name for item in targets} & set(comparison_videos))
    if overlap:
        raise CropQualityAuditError(f"Target/comparison videos overlap: {overlap}")
    teacher = load_audit_teacher_config(args.teacher_config)
    manifest_items = load_stage4_sweep_manifest(args.manifest)
    manifest_map = {item.video_name: item for item in manifest_items}
    requested_videos = [item.video_name for item in targets] + list(comparison_videos)
    missing = sorted(set(requested_videos) - set(manifest_map))
    if missing:
        raise CropQualityAuditError(f"Videos are missing from manifest: {missing}")

    bbox_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    stray_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    checksum_sources: list[tuple[str, str, Path]] = [
        ("", "manifest", Path(args.manifest).resolve()),
        ("", "teacher_config", Path(args.teacher_config).resolve()),
    ]

    for video in requested_videos:
        item = manifest_map[video]
        is_target = video in {target.video_name for target in targets}
        spec = next((target for target in targets if target.video_name == video), None)
        annotated_h5 = (
            _single_h5(args.annotated_root, video, args.annotated_glob_template)
            if is_target
            else None
        )
        frame_indices, shape_hw, pseudo_attrs = _read_pseudo(item.pseudo3d_h5)
        h5 = _read_v6(annotated_h5, video, shape_hw) if annotated_h5 else None
        if h5 is not None:
            if np.any(h5["frame_orders"] < 0) or np.any(
                h5["frame_orders"] >= len(frame_indices)
            ):
                raise CropQualityAuditError(
                    f"point_cloud/frame_order escapes pseudo3D inventory: {annotated_h5}"
                )
            if np.any(h5["bbox_orders"] < 0) or np.any(
                h5["bbox_orders"] >= len(frame_indices)
            ):
                raise CropQualityAuditError(
                    f"frame_annotation/frame_order escapes pseudo3D inventory: {annotated_h5}"
                )
        source_v5 = None
        if h5 is not None and h5["source_v5"]:
            source_v5_path = Path(h5["source_v5"]).expanduser().resolve()
            if not source_v5_path.is_file():
                raise FileNotFoundError(f"v6 source v5 H5 not found: {source_v5_path}")
            source_v5 = _read_source_v5(
                source_v5_path,
                shape_hw=shape_hw,
                expected=h5,
            )
        voc_by_order = load_voc_bboxes(
            item.voc_xml_root,
            video_name=video,
            frame_indices=frame_indices,
            xml_frame_number_offsets=teacher.xml_frame_number_offsets,
            xml_frame_id_source=teacher.xml_frame_id_source,
            xml_annotation_dir_name=teacher.xml_annotation_dir_name,
            strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
        )
        local_bbox_rows: list[dict[str, Any]] = []
        for order, vocs in sorted(voc_by_order.items()):
            for bbox_index, voc in enumerate(vocs):
                local_bbox_rows.append(
                    _bbox_row(
                        video=video,
                        frame_order=int(order),
                        frame_index=int(frame_indices[order]),
                        bbox_index=bbox_index,
                        voc=voc,
                        attrs=pseudo_attrs,
                        shape_hw=shape_hw,
                        h5=h5,
                    )
                )
        bbox_rows.extend(local_bbox_rows)
        bbox_by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in local_bbox_rows:
            bbox_by_order[int(row["frame_order"])].append(row)

        observed_stray_orders: set[int] = set()
        if h5 is not None:
            for point_index in np.flatnonzero(h5["stray"]):
                order = int(h5["frame_orders"][point_index])
                observed_stray_orders.add(order)
                same_order_bboxes = h5["bboxes"][h5["bbox_orders"] == order]
                stray_rows.append(
                    {
                        "video_name": video,
                        "frame_order": order,
                        "frame_index": int(frame_indices[order]),
                        "point_index": int(point_index),
                        "pixel_x": float(h5["pixel_xy"][point_index, 0]),
                        "pixel_y": float(h5["pixel_xy"][point_index, 1]),
                        "rounded_x": int(h5["rounded_xy"][point_index, 0]),
                        "rounded_y": int(h5["rounded_xy"][point_index, 1]),
                        "strict_bbox_inside": bool(h5["strict_inside"][point_index]),
                        "permissive_bbox_inside": bool(h5["permissive_inside"][point_index]),
                        "explained_by_degenerate_bbox_semantics": bool(h5["explained"][point_index]),
                        "saved_bboxes": ";".join(_pipe(row) for row in same_order_bboxes),
                        "label_authority": h5["authorities"].get(order, "inherited_automatic"),
                        "cvat_mask_status": h5["mask_statuses"].get(order, ""),
                        "source_v5_label": (
                            int(source_v5["labels"][point_index])
                            if source_v5 is not None
                            else ""
                        ),
                        "source_v5_stray_ignore": (
                            bool(source_v5["stray"][point_index])
                            if source_v5 is not None
                            else ""
                        ),
                    }
                )
            expected_orders = set(spec.frame_orders if spec else ())
            if observed_stray_orders != expected_orders:
                raise CropQualityAuditError(
                    f"Observed stray frame orders differ for {video}: "
                    f"observed={sorted(observed_stray_orders)}, expected={sorted(expected_orders)}"
                )

        for order, frame_index in enumerate(frame_indices.tolist()):
            boxes = bbox_by_order.get(order, [])
            crop_counts = Counter(str(row["crop_status"]) for row in boxes)
            if h5 is not None:
                selector = h5["frame_orders"] == order
                labels = h5["labels"][selector]
                stray_count = int(np.sum(h5["stray"][selector]))
                explained_count = int(np.sum(h5["explained"][selector]))
                saved_rows = h5["bboxes"][h5["bbox_orders"] == order]
                degenerate_rows = int(
                    sum(
                        np.all(np.isfinite(row))
                        and (float(row[2] - row[0]) <= 0.0 or float(row[3] - row[1]) <= 0.0)
                        for row in saved_rows
                    )
                )
                label_counts = {
                    "frame_points": int(labels.size),
                    "positive_points": int(np.sum(labels == 1)),
                    "ignore_points": int(np.sum(labels == -1)),
                    "background_points": int(np.sum(labels == 0)),
                    "source_v5_stray_ignore_points": (
                        int(np.sum(source_v5["stray"][selector]))
                        if source_v5 is not None
                        else ""
                    ),
                }
            else:
                stray_count = 0
                explained_count = 0
                saved_rows = np.zeros((0, 4))
                degenerate_rows = 0
                label_counts = {
                    "frame_points": "",
                    "positive_points": "",
                    "ignore_points": "",
                    "background_points": "",
                    "source_v5_stray_ignore_points": "",
                }
            frame_rows.append(
                {
                    "video_name": video,
                    "role": "target" if is_target else "comparison",
                    "split": item.split,
                    "frame_order": order,
                    "frame_index": int(frame_index),
                    "requested_problem_frame": bool(spec and order in spec.frame_orders),
                    "xml_bbox_rows": len(boxes),
                    "saved_bbox_rows": len(saved_rows),
                    "saved_degenerate_bbox_rows": degenerate_rows,
                    "fully_visible_bboxes": int(crop_counts.get("fully_visible", 0)),
                    "partially_clipped_bboxes": int(crop_counts.get("partially_clipped", 0)),
                    "fully_outside_crop_bboxes": int(crop_counts.get("fully_outside_crop", 0)),
                    "minimum_visible_fraction": min(
                        (float(row["visible_fraction"]) for row in boxes), default=math.nan
                    ),
                    "stray_ignore_points": stray_count,
                    "stray_explained_by_degenerate_bbox_semantics": explained_count,
                    "label_authority": h5["authorities"].get(order, "inherited_automatic") if h5 else "",
                    "cvat_mask_status": h5["mask_statuses"].get(order, "") if h5 else "",
                    **label_counts,
                }
            )

        crop_counts = Counter(str(row["crop_status"]) for row in local_bbox_rows)
        stray_count = int(np.sum(h5["stray"])) if h5 is not None else 0
        explained_count = int(np.sum(h5["explained"])) if h5 is not None else 0
        source_v5_stray_count = (
            int(np.sum(source_v5["stray"])) if source_v5 is not None else ""
        )
        video_rows.append(
            {
                "video_name": video,
                "role": "target" if is_target else "comparison",
                "split": item.split,
                "manifest_enabled": item.enabled,
                "frames": len(frame_indices),
                "xml_bbox_frames": len(voc_by_order),
                "xml_bbox_rows": len(local_bbox_rows),
                "fully_visible_bboxes": int(crop_counts.get("fully_visible", 0)),
                "partially_clipped_bboxes": int(crop_counts.get("partially_clipped", 0)),
                "fully_outside_crop_bboxes": int(crop_counts.get("fully_outside_crop", 0)),
                "minimum_visible_fraction": min(
                    (float(row["visible_fraction"]) for row in local_bbox_rows), default=math.nan
                ),
                "stray_ignore_points": stray_count,
                "stray_explained_by_degenerate_bbox_semantics": explained_count,
                "source_v5_stray_ignore_points": source_v5_stray_count,
                "stray_frame_orders": "|".join(str(value) for value in sorted(observed_stray_orders)),
                "v6_h5": str(annotated_h5) if annotated_h5 else "",
                "source_v5_h5": h5["source_v5"] if h5 else "",
            }
        )
        for role, path in _source_files(
            item, teacher, annotated_h5, h5["source_v5"] if h5 else ""
        ):
            checksum_sources.append((video, role, path))

    checksum_rows: list[dict[str, Any]] = []
    for video, role, path in sorted(checksum_sources, key=lambda row: (row[0], row[1], str(row[2]))):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        checksum_rows.append(
            {
                "video_name": video,
                "role": role,
                "path": str(path),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": file_sha256(path),
            }
        )

    total_stray = sum(int(row["stray_ignore_points"]) for row in video_rows)
    explained = sum(
        int(row["stray_explained_by_degenerate_bbox_semantics"]) for row in video_rows
    )
    stray_videos = sum(int(row["stray_ignore_points"]) > 0 for row in video_rows)
    requested_frames = sum(len(target.frame_orders) for target in targets)
    if args.expected_stray_points >= 0 and total_stray != args.expected_stray_points:
        raise CropQualityAuditError(
            f"Stray ignore count mismatch: {total_stray} != {args.expected_stray_points}"
        )
    if args.expected_stray_videos >= 0 and stray_videos != args.expected_stray_videos:
        raise CropQualityAuditError(
            f"Stray video count mismatch: {stray_videos} != {args.expected_stray_videos}"
        )
    if args.expected_problem_frames >= 0 and requested_frames != args.expected_problem_frames:
        raise CropQualityAuditError(
            f"Requested problem frame count mismatch: {requested_frames} != {args.expected_problem_frames}"
        )
    all_explained = total_stray > 0 and explained == total_stray

    summary = {
        "schema_version": 1,
        "status": "ok",
        "target_videos": len(targets),
        "comparison_videos": len(comparison_videos),
        "requested_problem_frames": requested_frames,
        "observed_stray_videos": stray_videos,
        "stray_ignore_points": total_stray,
        "stray_explained_by_degenerate_bbox_semantics": explained,
        "all_stray_explained_by_degenerate_bbox_semantics": all_explained,
        "source_checksum_rows": len(checksum_rows),
        "input_files_unchanged": True,
        "files_written": 6,
    }
    output_root = Path(args.output_root)
    prepare_output_root(output_root, overwrite=args.overwrite)
    bbox_rows.sort(key=lambda row: (row["video_name"], int(row["frame_order"]), int(row["bbox_index"])))
    frame_rows.sort(key=lambda row: (row["video_name"], int(row["frame_order"])))
    stray_rows.sort(key=lambda row: (row["video_name"], int(row["frame_order"]), int(row["point_index"])))
    video_rows.sort(key=lambda row: row["video_name"])
    write_csv_rows(output_root / "bbox_geometry.csv", bbox_rows)
    write_csv_rows(output_root / "frame_metrics.csv", frame_rows)
    write_csv_rows(output_root / "stray_points.csv", stray_rows)
    write_csv_rows(output_root / "video_summary.csv", video_rows)
    write_csv_rows(output_root / "input_checksums.csv", checksum_rows)
    write_json(output_root / "audit_summary.json", summary)

    after = {
        (row["path"], row["bytes"], row["mtime_ns"], row["sha256"])
        for row in checksum_rows
    }
    current = set()
    for row in checksum_rows:
        path = Path(row["path"])
        stat = path.stat()
        current.add((str(path), int(stat.st_size), int(stat.st_mtime_ns), file_sha256(path)))
    if current != after:
        raise CropQualityAuditError("An input file changed during the read-only audit")

    print("Stage 4 crop-quality/stray-ignore audit summary")
    print(f"  target videos       : {summary['target_videos']}")
    print(f"  comparison videos   : {summary['comparison_videos']}")
    print(f"  problem frames      : {summary['requested_problem_frames']}")
    print(f"  stray ignore points : {summary['stray_ignore_points']}")
    print(f"  degenerate explained: {summary['stray_explained_by_degenerate_bbox_semantics']}")
    print(f"  output root         : {output_root.resolve()}")
    print("Stage 4 crop-quality/stray-ignore read-only audit passed.")
    return {
        "summary": summary,
        "bbox_rows": bbox_rows,
        "frame_rows": frame_rows,
        "stray_rows": stray_rows,
        "video_rows": video_rows,
        "checksum_rows": checksum_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Stage 4 audit for teacher-v6 stray ignore points and "
            "unclipped XML BBox crop visibility."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--target", action="append", type=_parse_target, required=True)
    parser.add_argument("--comparison_video", action="append", default=[])
    parser.add_argument("--annotated_glob_template", default=DEFAULT_V6_GLOB)
    parser.add_argument("--expected_stray_points", type=int, default=-1)
    parser.add_argument("--expected_stray_videos", type=int, default=-1)
    parser.add_argument("--expected_problem_frames", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_audit(args)
    except Exception as exc:
        raise SystemExit(
            f"Stage 4 crop-quality/stray-ignore audit failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
