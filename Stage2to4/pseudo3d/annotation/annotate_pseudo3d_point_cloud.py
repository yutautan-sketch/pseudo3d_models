from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.utils.alpha_texture_processing import (
    AlphaTextureConfig,
    image_to_uint8_gray,
    make_texture_image,
)


LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_FEMUR_CANDIDATE = 1


@dataclass(frozen=True)
class VocBBox:
    xml_path: Path
    object_index: int
    object_name: str
    xml_xyxy: np.ndarray
    image_size_wh: np.ndarray | None


@dataclass(frozen=True)
class LocalBBox:
    xml_xyxy: np.ndarray
    raw_xyxy: np.ndarray
    local_xyxy: np.ndarray
    valid: bool


def _write_h5_attrs(group: Any, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        if isinstance(value, (np.bool_, bool)):
            group.attrs[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            group.attrs[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            group.attrs[key] = float(value)
        elif value is None:
            group.attrs[key] = "None"
        else:
            group.attrs[key] = str(value)


def load_point_cloud_h5(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud h5 not found: {path}")

    required = [
        "points",
        "intensity",
        "alpha",
        "confidence",
        "frame_index",
        "frame_order",
        "pixel_xy",
        "source_type",
    ]
    with h5py.File(path, "r") as f:
        if "point_cloud" not in f:
            raise KeyError(f"'point_cloud' group not found in {path}")
        group = f["point_cloud"]
        for key in required:
            if key not in group:
                raise KeyError(f"Required point_cloud/{key} not found in {path}")
        point_cloud = {key: group[key][:] for key in required}
        if "per_frame_counts" in group:
            point_cloud["per_frame_counts"] = group["per_frame_counts"][:]
        attrs = dict(group.attrs)

    return point_cloud, attrs


def load_pseudo3d_h5(path: Path, *, geometry_key: str) -> dict[str, Any]:
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pseudo3D h5 not found: {path}")

    if geometry_key == "prior":
        tracking_key = "pred_tracking"
    elif geometry_key == "corr":
        tracking_key = "pred_tracking_corr"
    else:
        raise ValueError(f"Unknown geometry_key: {geometry_key}")

    with h5py.File(path, "r") as f:
        required = [
            "local_encoder_images",
            "frame_indices",
            "pixel_to_image",
            tracking_key,
        ]
        for key in required:
            if key not in f:
                raise KeyError(
                    f"Required key '{key}' for geometry_key='{geometry_key}' "
                    f"not found in {path}"
                )
        data = {
            "local_encoder_images": f["local_encoder_images"][:],
            "frame_indices": f["frame_indices"][:].astype(np.int64),
            "pixel_to_image": f["pixel_to_image"][:].astype(np.float32),
            "pred_tracking": f[tracking_key][:].astype(np.float32),
            "attrs": dict(f.attrs),
            "tracking_key": tracking_key,
        }
    return data


def _parse_voc_bboxes(xml_path: Path) -> tuple[list[tuple[int, str, np.ndarray]], np.ndarray | None]:
    root = ET.parse(xml_path).getroot()
    image_size_wh = None
    size_node = root.find("size")
    if size_node is not None:
        width_node = size_node.find("width")
        height_node = size_node.find("height")
        if (
            width_node is not None
            and height_node is not None
            and width_node.text is not None
            and height_node.text is not None
        ):
            image_size_wh = np.asarray(
                [float(width_node.text), float(height_node.text)],
                dtype=np.float32,
            )

    boxes: list[tuple[int, str, np.ndarray]] = []
    for object_index, obj in enumerate(root.findall("object")):
        name_node = obj.find("name")
        object_name = name_node.text.strip() if name_node is not None and name_node.text else ""
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        values = []
        for name in ["xmin", "ymin", "xmax", "ymax"]:
            node = bnd.find(name)
            if node is None or node.text is None:
                raise ValueError(f"Missing {name} in {xml_path}")
            values.append(float(node.text))
        boxes.append((object_index, object_name, np.asarray(values, dtype=np.float32)))

    if not boxes:
        raise ValueError(f"No VOC object/bndbox found in {xml_path}")

    return boxes, image_size_wh


def _candidate_xml_paths(
    root: Path,
    *,
    video_name: str,
    frame_index: int,
    frame_order: int,
    offsets: tuple[int, ...],
    frame_id_source: str,
) -> list[Path]:
    names: list[str] = []
    base_nums = []
    if frame_id_source == "frame_index":
        values = [frame_index]
    elif frame_id_source == "frame_order":
        values = [frame_order]
    elif frame_id_source == "both":
        values = [frame_index, frame_order]
    else:
        raise ValueError(
            "frame_id_source must be one of: frame_index, frame_order, both"
        )

    for value in values:
        for offset in offsets:
            base_nums.append(int(value) + int(offset))

    for num in base_nums:
        num_tokens = [str(num), f"{num:04d}", f"{num:05d}", f"{num:06d}"]
        for token in num_tokens:
            names.extend(
                [
                    f"{token}.xml",
                    f"frame_{token}.xml",
                    f"{video_name}_{token}.xml",
                    f"{video_name}-{token}.xml",
                    f"{video_name}_frame_{token}.xml",
                ]
            )

    annotation_dir_names = [
        "annotations_renamed",
        "annotation_renamed",
        "annotations_rename",
        "annotation_rename",
        "annotations",
    ]

    if root.name in set(annotation_dir_names):
        search_roots = [
            root,
            *[root.with_name(dirname) for dirname in annotation_dir_names],
        ]
    else:
        search_roots = [
            root,
            root / video_name,
            *[root / dirname for dirname in annotation_dir_names],
            *[root / video_name / dirname for dirname in annotation_dir_names],
        ]

    seen: set[Path] = set()
    paths = []
    for search_root in search_roots:
        if search_root in seen:
            continue
        seen.add(search_root)
        for name in names:
            p = search_root / name
            if p not in seen:
                paths.append(p)
                seen.add(p)
    return paths


def load_voc_bboxes(
    voc_xml_root: Path,
    *,
    video_name: str,
    frame_indices: np.ndarray,
    xml_frame_number_offsets: tuple[int, ...],
    xml_frame_id_source: str,
) -> dict[int, list[VocBBox]]:
    voc_xml_root = Path(voc_xml_root)
    if not voc_xml_root.exists():
        raise FileNotFoundError(f"VOC XML root not found: {voc_xml_root}")

    by_order: dict[int, list[VocBBox]] = {}
    for frame_order, frame_index in enumerate(frame_indices.tolist()):
        xml_path = None
        for candidate in _candidate_xml_paths(
            voc_xml_root,
            video_name=video_name,
            frame_index=int(frame_index),
            frame_order=int(frame_order),
            offsets=xml_frame_number_offsets,
            frame_id_source=xml_frame_id_source,
        ):
            if candidate.exists():
                xml_path = candidate
                break

        if xml_path is None:
            continue

        boxes, image_size_wh = _parse_voc_bboxes(xml_path)
        by_order[int(frame_order)] = [
            VocBBox(
                xml_path=xml_path,
                object_index=object_index,
                object_name=object_name,
                xml_xyxy=xml_xyxy,
                image_size_wh=image_size_wh,
            )
            for object_index, object_name, xml_xyxy in boxes
        ]

    return by_order


def _as_float_attr(attrs: dict[str, Any], key: str, default: float) -> float:
    value = attrs.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def xml_bbox_to_local(
    xml_xyxy: np.ndarray,
    *,
    image_size_wh: np.ndarray | None,
    attrs: dict[str, Any],
    local_shape_hw: tuple[int, int],
) -> LocalBBox:
    xml_xyxy = np.asarray(xml_xyxy, dtype=np.float32)
    local_h, local_w = local_shape_hw

    preprocess = str(attrs.get("local_preprocess", ""))
    raw_w = _as_float_attr(attrs, "raw_width", np.nan)
    raw_h = _as_float_attr(attrs, "raw_height", np.nan)
    if image_size_wh is not None and np.all(np.isfinite(image_size_wh)):
        xml_w = float(image_size_wh[0])
        xml_h = float(image_size_wh[1])
    else:
        xml_w = raw_w
        xml_h = raw_h

    if not np.isfinite(raw_w) or raw_w <= 0:
        raw_w = xml_w
    if not np.isfinite(raw_h) or raw_h <= 0:
        raw_h = xml_h

    raw_xyxy = xml_xyxy.copy()
    if np.isfinite(xml_w) and np.isfinite(xml_h) and xml_w > 0 and xml_h > 0:
        raw_xyxy[[0, 2]] *= raw_w / xml_w
        raw_xyxy[[1, 3]] *= raw_h / xml_h

    if preprocess == "resize":
        if not np.isfinite(raw_w) or not np.isfinite(raw_h) or raw_w <= 0 or raw_h <= 0:
            raise ValueError("raw_width/raw_height attrs are required for resize mode")
        scale_x = local_w / raw_w
        scale_y = local_h / raw_h
        local = raw_xyxy.copy()
        local[[0, 2]] = local[[0, 2]] * scale_x
        local[[1, 3]] = local[[1, 3]] * scale_y
    else:
        scale = _as_float_attr(attrs, "local_resize_scale", 1.0)
        if not np.isfinite(scale):
            scale = 1.0
        crop_left = _as_float_attr(attrs, "local_crop_left", 0.0)
        crop_top = _as_float_attr(attrs, "local_crop_top", 0.0)
        local = raw_xyxy.copy()
        local[[0, 2]] = local[[0, 2]] * scale - crop_left
        local[[1, 3]] = local[[1, 3]] * scale - crop_top

    x1, y1, x2, y2 = local.tolist()
    ix1 = max(0.0, min(float(local_w - 1), x1))
    iy1 = max(0.0, min(float(local_h - 1), y1))
    ix2 = max(0.0, min(float(local_w - 1), x2))
    iy2 = max(0.0, min(float(local_h - 1), y2))
    clipped = np.asarray([ix1, iy1, ix2, iy2], dtype=np.float32)
    valid = bool(ix2 > ix1 and iy2 > iy1)
    return LocalBBox(
        xml_xyxy=xml_xyxy,
        raw_xyxy=raw_xyxy.astype(np.float32),
        local_xyxy=clipped,
        valid=valid,
    )


def make_point_labels(
    point_cloud: dict[str, np.ndarray],
    local_bboxes: dict[int, LocalBBox],
    *,
    bbox_ignore_margin: float,
    no_bbox_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_order = point_cloud["frame_order"].astype(np.int32)
    pixel_xy = point_cloud["pixel_xy"].astype(np.float32)
    labels = np.full(frame_order.shape, int(no_bbox_label), dtype=np.int8)

    for order, bbox in local_bboxes.items():
        if not bbox.valid:
            continue
        mask = frame_order == int(order)
        if not np.any(mask):
            continue

        xy = pixel_xy[mask]
        x1, y1, x2, y2 = bbox.local_xyxy
        inside = (
            (xy[:, 0] >= x1)
            & (xy[:, 0] <= x2)
            & (xy[:, 1] >= y1)
            & (xy[:, 1] <= y2)
        )
        frame_labels = np.full(xy.shape[0], LABEL_BACKGROUND, dtype=np.int8)
        frame_labels[inside] = LABEL_FEMUR_CANDIDATE

        margin = float(bbox_ignore_margin)
        if margin > 0:
            near_x = (
                (np.abs(xy[:, 0] - x1) <= margin)
                | (np.abs(xy[:, 0] - x2) <= margin)
            )
            near_y = (
                (np.abs(xy[:, 1] - y1) <= margin)
                | (np.abs(xy[:, 1] - y2) <= margin)
            )
            in_margin_band = inside & (near_x | near_y)
            frame_labels[in_margin_band] = LABEL_IGNORE

        labels[mask] = frame_labels

    valid_mask = labels != LABEL_IGNORE
    return labels, valid_mask


def make_contour_texture_config_from_args(args: argparse.Namespace) -> AlphaTextureConfig:
    if args.contour_preset == "original":
        cfg = AlphaTextureConfig(
            texture_style="original",
            white_foreground=False,
        )
    elif args.contour_preset == "otsu_area500":
        cfg = AlphaTextureConfig(
            texture_style="threshold_alpha",
            denoise=args.contour_denoise,
            denoise_ksize=args.contour_denoise_ksize,
            threshold_mode="otsu",
            open_ksize=args.contour_open_ksize,
            close_ksize=args.contour_close_ksize,
            morph_shape="ellipse",
            min_component_area=500,
            white_foreground=False,
        )
    elif args.contour_preset in {"percentile90_area100", "percentile_area100"}:
        cfg = AlphaTextureConfig(
            texture_style="threshold_alpha",
            denoise=args.contour_denoise,
            denoise_ksize=args.contour_denoise_ksize,
            threshold_mode="percentile",
            percentile=90.0,
            open_ksize=args.contour_open_ksize,
            close_ksize=args.contour_close_ksize,
            morph_shape="ellipse",
            min_component_area=100,
            white_foreground=False,
        )
    elif args.contour_preset == "percentile85_area100":
        cfg = AlphaTextureConfig(
            texture_style="threshold_alpha",
            denoise=args.contour_denoise,
            denoise_ksize=args.contour_denoise_ksize,
            threshold_mode="percentile",
            percentile=85.0,
            open_ksize=args.contour_open_ksize,
            close_ksize=args.contour_close_ksize,
            morph_shape="ellipse",
            min_component_area=100,
            white_foreground=False,
        )
    else:
        raise ValueError(f"Unknown contour_preset: {args.contour_preset}")

    if args.contour_threshold_mode is not None:
        cfg.threshold_mode = args.contour_threshold_mode
    if args.contour_percentile is not None:
        cfg.percentile = args.contour_percentile
    if args.contour_fixed_threshold is not None:
        cfg.fixed_threshold = args.contour_fixed_threshold
    if args.contour_min_component_area is not None:
        cfg.min_component_area = args.contour_min_component_area

    return cfg


def _bbox_to_roi_bounds(
    bbox: LocalBBox,
    *,
    image_shape_hw: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not bbox.valid:
        return None
    h, w = image_shape_hw
    x1, y1, x2, y2 = bbox.local_xyxy
    left = int(np.floor(max(0.0, x1)))
    top = int(np.floor(max(0.0, y1)))
    right = int(np.ceil(min(float(w - 1), x2)))
    bottom = int(np.ceil(min(float(h - 1), y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_full_frame_binary_mask(
    image: np.ndarray,
    *,
    texture_config: AlphaTextureConfig,
    min_alpha: int,
) -> np.ndarray:
    gray = image_to_uint8_gray(image)

    # Keep this identical to export_pseudo3d_visualization.py-style texture
    # creation: threshold/morphology is applied on the full local frame before
    # any VOC search bbox is cropped.
    texture = make_texture_image(image, texture_config)
    if texture.ndim == 3 and texture.shape[-1] == 4:
        return texture[..., 3].astype(np.uint8) >= int(min_alpha)
    return gray > 0


def build_largest_contour_mask_from_binary(
    full_binary: np.ndarray,
    bbox: LocalBBox,
    *,
    min_contour_area: float,
) -> dict[str, Any]:
    full_binary = np.asarray(full_binary, dtype=bool)
    bounds = _bbox_to_roi_bounds(bbox, image_shape_hw=full_binary.shape)
    full_mask = np.zeros(full_binary.shape, dtype=bool)
    if bounds is None:
        return {
            "valid": False,
            "reason": "invalid_bbox",
            "mask": full_mask,
            "contour_area": 0.0,
            "binary_area": 0,
            "foreground_ratio_in_bbox": 0.0,
        }

    left, top, right, bottom = bounds
    binary = full_binary[top : bottom + 1, left : right + 1]

    binary_area = int(binary.sum())
    foreground_ratio = float(binary.mean()) if binary.size else 0.0
    if binary_area == 0:
        return {
            "valid": False,
            "reason": "empty_binary_mask",
            "mask": full_mask,
            "contour_area": 0.0,
            "binary_area": binary_area,
            "foreground_ratio_in_bbox": foreground_ratio,
        }

    contours, _ = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return {
            "valid": False,
            "reason": "no_contour",
            "mask": full_mask,
            "contour_area": 0.0,
            "binary_area": binary_area,
            "foreground_ratio_in_bbox": foreground_ratio,
        }

    contour_areas = [float(cv2.contourArea(contour)) for contour in contours]
    best_idx = int(np.argmax(contour_areas))
    contour_area = contour_areas[best_idx]
    if contour_area < float(min_contour_area):
        return {
            "valid": False,
            "reason": "contour_area_below_threshold",
            "mask": full_mask,
            "contour_area": contour_area,
            "binary_area": binary_area,
            "foreground_ratio_in_bbox": foreground_ratio,
        }

    roi_mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(roi_mask, contours, best_idx, color=1, thickness=-1)
    full_mask[top : bottom + 1, left : right + 1] = roi_mask.astype(bool)

    return {
        "valid": True,
        "reason": "ok",
        "mask": full_mask,
        "contour_area": contour_area,
        "binary_area": binary_area,
        "foreground_ratio_in_bbox": foreground_ratio,
    }


def build_largest_contour_mask_in_bbox(
    image: np.ndarray,
    bbox: LocalBBox,
    *,
    texture_config: AlphaTextureConfig,
    min_alpha: int,
    min_contour_area: float,
) -> dict[str, Any]:
    full_binary = build_full_frame_binary_mask(
        image,
        texture_config=texture_config,
        min_alpha=min_alpha,
    )
    return build_largest_contour_mask_from_binary(
        full_binary,
        bbox,
        min_contour_area=min_contour_area,
    )


def count_points_in_mask(
    *,
    point_cloud: dict[str, np.ndarray],
    frame_order_value: int,
    mask: np.ndarray,
) -> int:
    frame_order = point_cloud["frame_order"].astype(np.int32)
    pixel_xy = point_cloud["pixel_xy"].astype(np.float32)

    frame_mask = frame_order == int(frame_order_value)
    if not np.any(frame_mask):
        return 0

    xy = np.rint(pixel_xy[frame_mask]).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)

    return int(mask[xy[:, 1], xy[:, 0]].sum())


def _with_contour_selection_metadata(
    result: dict[str, Any],
    *,
    positive_points: int,
    fallback_used: bool,
    fallback_percentile: float | None,
    fallback_attempts: int,
) -> dict[str, Any]:
    out = dict(result)
    out["positive_points"] = int(positive_points)
    out["fallback_used"] = bool(fallback_used)
    out["fallback_percentile"] = (
        np.nan if fallback_percentile is None else float(fallback_percentile)
    )
    out["fallback_attempts"] = int(fallback_attempts)
    return out


def _mark_contour_result_failed(
    result: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    out = dict(result)
    out["valid"] = False
    out["reason"] = reason
    out["mask"] = np.zeros_like(result["mask"], dtype=bool)
    return out


def try_contour_with_fallbacks(
    *,
    image: np.ndarray,
    bbox: LocalBBox,
    base_texture_config: AlphaTextureConfig,
    min_alpha: int,
    min_contour_area: float,
    point_cloud: dict[str, np.ndarray],
    frame_order_value: int,
    enable_fallback: bool,
    fallback_percentiles: list[float],
    fallback_min_positive_points: int,
) -> dict[str, Any]:
    base_result = build_largest_contour_mask_in_bbox(
        image,
        bbox,
        texture_config=base_texture_config,
        min_alpha=min_alpha,
        min_contour_area=min_contour_area,
    )
    base_positive = (
        count_points_in_mask(
            point_cloud=point_cloud,
            frame_order_value=frame_order_value,
            mask=base_result["mask"],
        )
        if base_result["valid"]
        else 0
    )
    min_positive = max(0, int(fallback_min_positive_points))
    base_with_meta = _with_contour_selection_metadata(
        base_result,
        positive_points=base_positive,
        fallback_used=False,
        fallback_percentile=None,
        fallback_attempts=0,
    )
    if not enable_fallback:
        return base_with_meta
    if base_result["valid"] and base_positive >= min_positive:
        return base_with_meta

    best_result = base_with_meta
    best_positive = base_positive
    attempts = 0
    for percentile in fallback_percentiles:
        attempts += 1
        cfg = replace(
            base_texture_config,
            texture_style="threshold_alpha",
            threshold_mode="percentile",
            percentile=float(percentile),
        )
        candidate = build_largest_contour_mask_in_bbox(
            image,
            bbox,
            texture_config=cfg,
            min_alpha=min_alpha,
            min_contour_area=min_contour_area,
        )
        positive = (
            count_points_in_mask(
                point_cloud=point_cloud,
                frame_order_value=frame_order_value,
                mask=candidate["mask"],
            )
            if candidate["valid"]
            else 0
        )
        candidate = _with_contour_selection_metadata(
            candidate,
            positive_points=positive,
            fallback_used=True,
            fallback_percentile=float(percentile),
            fallback_attempts=attempts,
        )
        if positive > best_positive:
            best_result = candidate
            best_positive = positive
        if candidate["valid"] and positive >= min_positive:
            return candidate

    if best_result["valid"] and best_positive < min_positive:
        return _mark_contour_result_failed(
            best_result,
            reason="fallback_positive_points_below_threshold",
        )
    if not best_result["valid"]:
        return _with_contour_selection_metadata(
            best_result,
            positive_points=best_positive,
            fallback_used=bool(attempts > 0),
            fallback_percentile=best_result.get("fallback_percentile", np.nan),
            fallback_attempts=attempts,
        )
    return best_result


def build_bbox_union_mask(
    bboxes: list[LocalBBox],
    *,
    image_shape_hw: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(image_shape_hw, dtype=bool)
    for bbox in bboxes:
        bounds = _bbox_to_roi_bounds(bbox, image_shape_hw=image_shape_hw)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        mask[top : bottom + 1, left : right + 1] = True
    return mask


def build_frame_contour_mask_results(
    *,
    local_images: np.ndarray,
    voc_bboxes: dict[int, list[VocBBox]],
    local_bboxes: dict[int, list[LocalBBox]],
    texture_config: AlphaTextureConfig,
    min_alpha: int,
    min_contour_area: float,
    point_cloud: dict[str, np.ndarray] | None = None,
    enable_fallback: bool = False,
    fallback_percentiles: list[float] | None = None,
    fallback_min_positive_points: int = 0,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, np.ndarray], dict[str, int]]:
    results_by_order: dict[int, list[dict[str, Any]]] = {}
    union_masks_by_order: dict[int, np.ndarray] = {}
    stats = {
        "num_frames": int(local_images.shape[0]),
        "num_xml_frames": int(len(voc_bboxes)),
        "num_voc_bboxes": int(sum(len(boxes) for boxes in voc_bboxes.values())),
        "num_valid_contours": 0,
        "num_valid_contour_frames": 0,
        "num_invalid_contours": 0,
        "num_invalid_contour_frames": 0,
    }

    for order in sorted(local_bboxes):
        order_results = []
        contour_union = None
        frame_has_valid_contour = False
        frame_has_invalid_contour = False
        use_fallback = bool(enable_fallback and point_cloud is not None)
        full_binary = None
        if not use_fallback:
            full_binary = build_full_frame_binary_mask(
                local_images[order],
                texture_config=texture_config,
                min_alpha=min_alpha,
            )

        for bbox_idx, (voc, bbox) in enumerate(
            zip(voc_bboxes[order], local_bboxes[order], strict=True)
        ):
            if use_fallback:
                result = try_contour_with_fallbacks(
                    image=local_images[order],
                    bbox=bbox,
                    base_texture_config=texture_config,
                    min_alpha=min_alpha,
                    min_contour_area=min_contour_area,
                    point_cloud=point_cloud,
                    frame_order_value=int(order),
                    enable_fallback=True,
                    fallback_percentiles=fallback_percentiles or [],
                    fallback_min_positive_points=fallback_min_positive_points,
                )
            else:
                result = build_largest_contour_mask_from_binary(
                    full_binary,
                    bbox,
                    min_contour_area=min_contour_area,
                )
                positive_points = 0
                if point_cloud is not None and result["valid"]:
                    positive_points = count_points_in_mask(
                        point_cloud=point_cloud,
                        frame_order_value=int(order),
                        mask=result["mask"],
                    )
                result = _with_contour_selection_metadata(
                    result,
                    positive_points=positive_points,
                    fallback_used=False,
                    fallback_percentile=None,
                    fallback_attempts=0,
                )
            if contour_union is None:
                contour_union = np.zeros_like(result["mask"], dtype=bool)
            if result["valid"]:
                contour_union |= result["mask"]
                frame_has_valid_contour = True
                stats["num_valid_contours"] += 1
            else:
                frame_has_invalid_contour = True
                stats["num_invalid_contours"] += 1

            order_results.append(
                {
                    "bbox_index": int(bbox_idx),
                    "voc": voc,
                    "bbox": bbox,
                    "result": result,
                }
            )

        if contour_union is not None:
            union_masks_by_order[int(order)] = contour_union
        if frame_has_valid_contour:
            stats["num_valid_contour_frames"] += 1
        if frame_has_invalid_contour and not frame_has_valid_contour:
            stats["num_invalid_contour_frames"] += 1
        results_by_order[int(order)] = order_results

    return results_by_order, union_masks_by_order, stats


def build_contour_point_annotations(
    *,
    point_cloud: dict[str, np.ndarray],
    pseudo3d: dict[str, Any],
    voc_bboxes: dict[int, list[VocBBox]],
    local_bboxes: dict[int, list[LocalBBox]],
    texture_config: AlphaTextureConfig,
    min_alpha: int,
    min_contour_area: float,
    no_bbox_label: int,
    enable_contour_fallback: bool = False,
    fallback_percentiles: list[float] | None = None,
    fallback_min_positive_points: int = 5,
    bbox_inside_non_contour_label: str = "ignore",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    local_images = pseudo3d["local_encoder_images"]
    frame_indices = pseudo3d["frame_indices"]
    frame_order = point_cloud["frame_order"].astype(np.int32)
    pixel_xy = point_cloud["pixel_xy"].astype(np.float32)
    labels = np.full(frame_order.shape, int(no_bbox_label), dtype=np.int8)
    rows = []
    results_by_order, union_masks_by_order, _ = build_frame_contour_mask_results(
        local_images=local_images,
        voc_bboxes=voc_bboxes,
        local_bboxes=local_bboxes,
        texture_config=texture_config,
        min_alpha=min_alpha,
        min_contour_area=min_contour_area,
        point_cloud=point_cloud,
        enable_fallback=enable_contour_fallback,
        fallback_percentiles=fallback_percentiles or [],
        fallback_min_positive_points=fallback_min_positive_points,
    )

    for order in sorted(results_by_order):
        frame_mask = frame_order == int(order)
        num_points = int(frame_mask.sum())
        frame_labels = np.full(num_points, LABEL_BACKGROUND, dtype=np.int8)
        frame_rows = []
        frame_xy = None
        if num_points > 0:
            frame_xy = np.rint(pixel_xy[frame_mask]).astype(np.int64)

            if bbox_inside_non_contour_label == "ignore":
                image_shape_hw = image_to_uint8_gray(local_images[order]).shape
                bbox_union = build_bbox_union_mask(
                    local_bboxes[order],
                    image_shape_hw=image_shape_hw,
                )
                xy_bbox = frame_xy.copy()
                xy_bbox[:, 0] = np.clip(xy_bbox[:, 0], 0, bbox_union.shape[1] - 1)
                xy_bbox[:, 1] = np.clip(xy_bbox[:, 1], 0, bbox_union.shape[0] - 1)
                inside_bbox = bbox_union[xy_bbox[:, 1], xy_bbox[:, 0]]
                frame_labels[inside_bbox] = LABEL_IGNORE
            elif bbox_inside_non_contour_label != "background":
                raise ValueError(
                    "bbox_inside_non_contour_label must be one of: "
                    "ignore, background"
                )

        for item in results_by_order[order]:
            bbox_idx = int(item["bbox_index"])
            voc = item["voc"]
            bbox = item["bbox"]
            result = item["result"]
            num_labeled_bbox = 0
            if num_points > 0 and result["valid"] and frame_xy is not None:
                xy = frame_xy.copy()
                xy[:, 0] = np.clip(xy[:, 0], 0, result["mask"].shape[1] - 1)
                xy[:, 1] = np.clip(xy[:, 1], 0, result["mask"].shape[0] - 1)
                inside_bbox_mask = result["mask"][xy[:, 1], xy[:, 0]]
                num_labeled_bbox = int(inside_bbox_mask.sum())

            frame_rows.append(
                {
                    "frame_order": int(order),
                    "frame_index": int(frame_indices[order]),
                    "object_index": np.int32(voc.object_index),
                    "bbox_index": np.int32(bbox_idx),
                    "bbox_xml_xyxy": voc.xml_xyxy.astype(np.float32),
                    "bbox_xml_size_wh": (
                        voc.image_size_wh.astype(np.float32)
                        if voc.image_size_wh is not None
                        else np.full((2,), np.nan, dtype=np.float32)
                    ),
                    "bbox_raw_xyxy": bbox.raw_xyxy.astype(np.float32),
                    "bbox_local_xyxy": bbox.local_xyxy.astype(np.float32),
                    "valid_contour": bool(result["valid"]),
                    "contour_area": np.float32(result["contour_area"]),
                    "binary_area": np.int32(result["binary_area"]),
                    "foreground_ratio_in_bbox": np.float32(
                        result["foreground_ratio_in_bbox"]
                    ),
                    "num_frame_points": np.int32(num_points),
                    "num_labeled_points": np.int32(num_labeled_bbox),
                    "contour_positive_points": np.int32(
                        result.get("positive_points", num_labeled_bbox)
                    ),
                    "fallback_used": bool(result.get("fallback_used", False)),
                    "fallback_percentile": np.float32(
                        result.get("fallback_percentile", np.nan)
                    ),
                    "fallback_attempts": np.int32(
                        result.get("fallback_attempts", 0)
                    ),
                    "annotation_reason": str(result["reason"]),
                    "object_name": str(voc.object_name),
                    "xml_path": str(voc.xml_path),
                }
            )

        num_labeled = 0
        contour_union = union_masks_by_order.get(int(order))
        if (
            num_points > 0
            and frame_xy is not None
            and contour_union is not None
            and np.any(contour_union)
        ):
            xy = frame_xy.copy()
            xy[:, 0] = np.clip(xy[:, 0], 0, contour_union.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, contour_union.shape[0] - 1)
            inside = contour_union[xy[:, 1], xy[:, 0]]
            frame_labels[inside] = LABEL_FEMUR_CANDIDATE
            num_labeled = int(inside.sum())
        if num_points > 0:
            labels[frame_mask] = frame_labels

        for row in frame_rows:
            row["num_labeled_points_union"] = np.int32(num_labeled)
        rows.extend(frame_rows)

    if not rows:
        return (
            labels,
            labels != LABEL_IGNORE,
            {
                "frame_order": np.zeros((0,), dtype=np.int32),
                "frame_index": np.zeros((0,), dtype=np.int64),
                "object_index": np.zeros((0,), dtype=np.int32),
                "bbox_index": np.zeros((0,), dtype=np.int32),
                "bbox_xml_xyxy": np.zeros((0, 4), dtype=np.float32),
                "bbox_xml_size_wh": np.zeros((0, 2), dtype=np.float32),
                "bbox_raw_xyxy": np.zeros((0, 4), dtype=np.float32),
                "bbox_local_xyxy": np.zeros((0, 4), dtype=np.float32),
                "valid_contour": np.zeros((0,), dtype=bool),
                "contour_area": np.zeros((0,), dtype=np.float32),
                "binary_area": np.zeros((0,), dtype=np.int32),
                "foreground_ratio_in_bbox": np.zeros((0,), dtype=np.float32),
                "num_frame_points": np.zeros((0,), dtype=np.int32),
                "num_labeled_points": np.zeros((0,), dtype=np.int32),
                "contour_positive_points": np.zeros((0,), dtype=np.int32),
                "fallback_used": np.zeros((0,), dtype=bool),
                "fallback_percentile": np.zeros((0,), dtype=np.float32),
                "fallback_attempts": np.zeros((0,), dtype=np.int32),
                "num_labeled_points_union": np.zeros((0,), dtype=np.int32),
                "annotation_reason": np.asarray([], dtype=object),
                "object_name": np.asarray([], dtype=object),
                "xml_path": np.asarray([], dtype=object),
            },
        )

    frame_annotation: dict[str, np.ndarray] = {}
    for key in rows[0]:
        if key in {"annotation_reason", "object_name", "xml_path"}:
            frame_annotation[key] = np.asarray([row[key] for row in rows], dtype=object)
        else:
            frame_annotation[key] = np.asarray([row[key] for row in rows])

    return labels, labels != LABEL_IGNORE, frame_annotation


def make_empty_measurement() -> dict[str, Any]:
    return {
        "endpoint_1": np.full((3,), np.nan, dtype=np.float32),
        "endpoint_2": np.full((3,), np.nan, dtype=np.float32),
        "length": np.float32(np.nan),
        "source_frame_order": np.int32(-1),
        "source_frame_index": np.int64(-1),
        "source_method": "not_computed_contour_point_annotation",
    }


def save_annotated_h5(
    output_h5: Path,
    *,
    point_cloud: dict[str, np.ndarray],
    point_cloud_attrs: dict[str, Any],
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    frame_annotation: dict[str, np.ndarray],
    measurement: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    import h5py

    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        pc_group = f.create_group("point_cloud")
        for key, value in point_cloud.items():
            pc_group.create_dataset(key, data=value, compression="gzip")
        _write_h5_attrs(pc_group, point_cloud_attrs)

        ann_group = f.create_group("annotation")
        ann_group.create_dataset("point_label", data=point_label, compression="gzip")
        ann_group.create_dataset("valid_mask", data=valid_mask, compression="gzip")
        ann_group.attrs["label_source"] = "voc_bbox_largest_contour_weak_label"
        ann_group.attrs["label_definition"] = (
            "-1=ignore, 0=background, 1=femur_candidate"
        )
        ann_group.attrs["weak_annotation_note"] = (
            "VOC bbox is treated as a weak search region. Labels are assigned "
            "from the largest filled contour after intensity/alpha binarization "
            "inside the bbox; they are still weak annotations."
        )

        frame_group = f.create_group("frame_annotation")
        for key, value in frame_annotation.items():
            if value.dtype == object:
                frame_group.create_dataset(
                    key,
                    data=value.astype(h5py.string_dtype("utf-8")),
                )
            else:
                frame_group.create_dataset(key, data=value, compression="gzip")

        meas_group = f.create_group("measurement")
        for key in ["endpoint_1", "endpoint_2", "length", "source_frame_order", "source_frame_index"]:
            meas_group.create_dataset(key, data=measurement[key])
        meas_group.attrs["source_method"] = str(measurement["source_method"])

        _write_h5_attrs(f, meta)

    print(f"Saved annotated point cloud h5: {output_h5}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate sparse pseudo-3D evidence point cloud with weak VOC BBox "
            "labels. The bbox is used as a search region; labels are assigned "
            "from the largest filled contour after binarization inside the bbox."
        )
    )
    parser.add_argument("--point_cloud_h5", type=Path, required=True)
    parser.add_argument("--pseudo3d_h5", type=Path, required=True)
    parser.add_argument("--voc_xml_root", type=Path, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--output_h5", type=Path, required=True)
    parser.add_argument("--geometry_key", type=str, default="corr", choices=["prior", "corr"])
    parser.add_argument(
        "--label_mode",
        type=str,
        default="contour_in_bbox",
        choices=["contour_in_bbox"],
    )
    parser.add_argument(
        "--endpoint_mode",
        type=str,
        default="none",
        choices=["none", "pca_in_bbox"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--bbox_ignore_margin", type=float, default=0.0)
    parser.add_argument(
        "--no_bbox_label",
        type=int,
        default=LABEL_IGNORE,
        choices=[LABEL_IGNORE, LABEL_BACKGROUND],
        help="Label for points whose frame has no usable VOC bbox.",
    )
    parser.add_argument(
        "--xml_frame_number_offsets",
        type=str,
        default="0,1",
        help="Comma-separated offsets tried for frame_index/frame_order XML names.",
    )
    parser.add_argument(
        "--xml_frame_id_source",
        type=str,
        default="frame_index",
        choices=["frame_index", "frame_order", "both"],
        help=(
            "Which frame id is used in XML filenames. "
            "frame_index is the original video frame index saved in pseudo3d h5; "
            "frame_order is the sampled sequence order. Use both only for legacy "
            "ambiguous datasets."
        ),
    )
    parser.add_argument(
        "--contour_preset",
        type=str,
        default="percentile90_area100",
        choices=[
            "original",
            "otsu_area500",
            "percentile90_area100",
            "percentile_area100",
            "percentile85_area100",
        ],
        help="Binarization preset used inside each local bbox.",
    )
    parser.add_argument(
        "--contour_threshold_mode",
        type=str,
        default=None,
        choices=["fixed", "otsu", "percentile", "adaptive"],
        help="Override preset threshold mode.",
    )
    parser.add_argument("--contour_percentile", type=float, default=None)
    parser.add_argument("--contour_fixed_threshold", type=int, default=None)
    parser.add_argument("--contour_min_component_area", type=int, default=None)
    parser.add_argument("--contour_min_area", type=float, default=20.0)
    parser.add_argument("--contour_min_alpha", type=int, default=1)
    parser.add_argument("--contour_open_ksize", type=int, default=3)
    parser.add_argument("--contour_close_ksize", type=int, default=5)
    parser.add_argument(
        "--contour_denoise",
        type=str,
        default="median",
        choices=["none", "median", "gaussian", "bilateral"],
    )
    parser.add_argument("--contour_denoise_ksize", type=int, default=3)
    parser.add_argument(
        "--enable_contour_fallback",
        action="store_true",
        help=(
            "Enable relaxed contour extraction fallback when the initial contour "
            "is invalid or yields too few positive points. Mainly intended for "
            "non-foreground point clouds such as point_mode=grid."
        ),
    )
    parser.add_argument(
        "--fallback_only_non_foreground_point_cloud",
        action="store_true",
        default=True,
        help=(
            "Apply contour fallback only when point_cloud attrs indicate "
            "point_mode != foreground."
        ),
    )
    parser.add_argument(
        "--fallback_percentiles",
        type=str,
        default="85,80,75,70",
        help="Comma-separated percentile thresholds tried during contour fallback.",
    )
    parser.add_argument(
        "--fallback_min_positive_points",
        type=int,
        default=5,
        help=(
            "If a valid contour yields fewer than this number of positive points "
            "on the current point cloud, try fallback when enabled."
        ),
    )
    parser.add_argument(
        "--bbox_inside_non_contour_label",
        type=str,
        default="ignore",
        choices=["ignore", "background"],
        help=(
            "Label assigned to points inside a VOC bbox but outside the accepted "
            "contour. For grid point clouds, ignore is safer because contour "
            "extraction may miss true femur pixels."
        ),
    )
    return parser


def _parse_offsets(text: str) -> tuple[int, ...]:
    offsets = []
    for token in re.split(r"[, ]+", text.strip()):
        if token:
            offsets.append(int(token))
    if not offsets:
        offsets = [0, 1]
    return tuple(dict.fromkeys(offsets))


def _parse_float_list(text: str) -> list[float]:
    values = []
    for token in re.split(r"[, ]+", text.strip()):
        if token:
            values.append(float(token))
    return values


def _attr_to_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    text = str(value)
    return text if text else default


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    point_cloud, point_cloud_attrs = load_point_cloud_h5(args.point_cloud_h5)
    pseudo3d = load_pseudo3d_h5(args.pseudo3d_h5, geometry_key=args.geometry_key)
    point_mode = _attr_to_str(point_cloud_attrs.get("point_mode"), "foreground")
    fallback_percentiles = _parse_float_list(args.fallback_percentiles)
    enable_contour_fallback = bool(args.enable_contour_fallback)
    if args.fallback_only_non_foreground_point_cloud and point_mode == "foreground":
        enable_contour_fallback = False

    local_images = pseudo3d["local_encoder_images"]
    local_shape_hw = tuple(image_to_uint8_gray(local_images[0]).shape)
    voc_bboxes = load_voc_bboxes(
        args.voc_xml_root,
        video_name=args.video_name,
        frame_indices=pseudo3d["frame_indices"],
        xml_frame_number_offsets=_parse_offsets(args.xml_frame_number_offsets),
        xml_frame_id_source=args.xml_frame_id_source,
    )
    local_bboxes = {
        order: [
            xml_bbox_to_local(
                voc.xml_xyxy,
                image_size_wh=voc.image_size_wh,
                attrs=pseudo3d["attrs"],
                local_shape_hw=local_shape_hw,
            )
            for voc in boxes
        ]
        for order, boxes in voc_bboxes.items()
    }

    texture_config = make_contour_texture_config_from_args(args)
    point_label, valid_mask, frame_annotation = build_contour_point_annotations(
        point_cloud=point_cloud,
        pseudo3d=pseudo3d,
        voc_bboxes=voc_bboxes,
        local_bboxes=local_bboxes,
        texture_config=texture_config,
        min_alpha=args.contour_min_alpha,
        min_contour_area=args.contour_min_area,
        no_bbox_label=args.no_bbox_label,
        enable_contour_fallback=enable_contour_fallback,
        fallback_percentiles=fallback_percentiles,
        fallback_min_positive_points=args.fallback_min_positive_points,
        bbox_inside_non_contour_label=args.bbox_inside_non_contour_label,
    )
    measurement = make_empty_measurement()

    num_points = int(point_cloud["points"].shape[0])
    num_xml_frames = int(len(voc_bboxes))
    num_bbox = int(sum(len(boxes) for boxes in voc_bboxes.values()))
    valid_contour_mask = frame_annotation["valid_contour"].astype(bool)
    num_valid_contour = int(valid_contour_mask.sum())
    num_valid_contour_frames = int(
        np.unique(frame_annotation["frame_order"][valid_contour_mask]).size
    )
    num_labeled_points = int(np.sum(point_label == LABEL_FEMUR_CANDIDATE))
    num_fallback_used = int(frame_annotation["fallback_used"].astype(bool).sum())
    meta = {
        "source_point_cloud_h5": str(args.point_cloud_h5),
        "source_pseudo3d_h5": str(args.pseudo3d_h5),
        "voc_xml_root": str(args.voc_xml_root),
        "video_name": args.video_name,
        "geometry_key": args.geometry_key,
        "tracking_key": pseudo3d["tracking_key"],
        "point_cloud_point_mode": point_mode,
        "label_mode": args.label_mode,
        "bbox_ignore_margin": float(args.bbox_ignore_margin),
        "bbox_inside_non_contour_label": args.bbox_inside_non_contour_label,
        "no_bbox_label": int(args.no_bbox_label),
        "xml_frame_id_source": args.xml_frame_id_source,
        "xml_frame_number_offsets": args.xml_frame_number_offsets,
        "contour_preset": args.contour_preset,
        "contour_threshold_mode": texture_config.threshold_mode,
        "contour_percentile": float(texture_config.percentile),
        "contour_fixed_threshold": int(texture_config.fixed_threshold),
        "contour_min_component_area": int(texture_config.min_component_area),
        "contour_min_area": float(args.contour_min_area),
        "contour_min_alpha": int(args.contour_min_alpha),
        "contour_open_ksize": int(texture_config.open_ksize),
        "contour_close_ksize": int(texture_config.close_ksize),
        "contour_denoise": texture_config.denoise,
        "contour_denoise_ksize": int(texture_config.denoise_ksize),
        "enable_contour_fallback": bool(enable_contour_fallback),
        "requested_enable_contour_fallback": bool(args.enable_contour_fallback),
        "fallback_only_non_foreground_point_cloud": bool(
            args.fallback_only_non_foreground_point_cloud
        ),
        "fallback_percentiles": args.fallback_percentiles,
        "fallback_min_positive_points": int(args.fallback_min_positive_points),
        "num_points": num_points,
        "num_voc_bbox_frames": num_xml_frames,
        "num_voc_bboxes": num_bbox,
        "num_valid_contours": num_valid_contour,
        "num_valid_contour_frames": num_valid_contour_frames,
        "num_fallback_used": num_fallback_used,
        "num_labeled_points": num_labeled_points,
        "label_source": (
            "VOC BBox weak annotation; largest filled contour in bbox after "
            "binarization is projected to point labels"
        ),
    }

    print("Pseudo-3D point cloud annotation:")
    print(f"  num_points              : {num_points}")
    print(f"  num_voc_bbox_frames     : {num_xml_frames}")
    print(f"  num_voc_bboxes          : {num_bbox}")
    print(f"  num_valid_contours      : {num_valid_contour}")
    print(f"  num_valid_contour_frames: {num_valid_contour_frames}")
    print(f"  num_fallback_used       : {num_fallback_used}")
    print(f"  num_labeled_points      : {num_labeled_points}")

    save_annotated_h5(
        args.output_h5,
        point_cloud=point_cloud,
        point_cloud_attrs=point_cloud_attrs,
        point_label=point_label,
        valid_mask=valid_mask,
        frame_annotation=frame_annotation,
        measurement=measurement,
        meta=meta,
    )


if __name__ == "__main__":
    main()
