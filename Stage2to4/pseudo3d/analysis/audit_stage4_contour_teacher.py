from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.stage4_sampling_sweep_config import (
    Stage4SweepManifestItem,
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_FEMUR_CANDIDATE,
    LocalBBox,
    RankedContourConfig,
    build_bbox_ranked_contour_candidates,
    build_bbox_ranked_contour_mask,
    build_full_frame_binary_mask,
    build_ranked_local_binary_mask,
    load_voc_bboxes,
    ranked_contour_candidate_sort_key,
    ranked_contour_config_from_metadata,
    xml_bbox_to_local,
)
from src.utils.alpha_texture_processing import AlphaTextureConfig, image_to_uint8_gray


AUDIT_SCHEMA_VERSION = 1
STRING_FRAME_DATASETS = {
    "annotation_reason",
    "selected_contour_source",
    "object_name",
    "xml_path",
}
REQUIRED_FRAME_DATASETS = {
    "frame_order",
    "frame_index",
    "object_index",
    "bbox_index",
    "bbox_local_xyxy",
    "valid_contour",
    "contour_area",
    "foreground_ratio_in_bbox",
    "num_frame_points",
    "num_labeled_points",
    "contour_positive_points",
    "annotation_reason",
    "selected_contour_source",
    "contour_selection_score",
    "center_distance_norm",
    "selected_area_ratio",
    "num_global_candidates",
    "num_local_candidates",
    "num_global_eligible_candidates",
    "num_local_eligible_candidates",
    "global_candidate_score",
    "local_candidate_score",
    "object_name",
    "xml_path",
}


@dataclass(frozen=True)
class AuditTeacherConfig:
    path: Path
    raw: dict[str, Any]
    config_name: str
    file_sha256: str
    fingerprint: str
    xml_frame_number_offsets: tuple[int, ...]
    xml_frame_id_source: str
    xml_annotation_dir_name: str
    strict_xml_annotation_dir: bool
    no_bbox_label: int
    bbox_inside_non_contour_label: str
    texture_config: AlphaTextureConfig
    min_alpha: int
    min_contour_area: float
    ranked_config: RankedContourConfig


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ResolvedInput:
    item: Stage4SweepManifestItem
    annotated_h5: Path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"teacher config '{name}' must be a mapping")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"teacher config '{name}' must be true or false")
    return value


def load_audit_teacher_config(path: Path) -> AuditTeacherConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Teacher config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw_loaded = yaml.safe_load(handle)
    if not isinstance(raw_loaded, Mapping):
        raise ValueError("Teacher config root must be a mapping")
    raw = dict(raw_loaded)
    if int(raw.get("schema_version", 0)) != 2:
        raise ValueError("Phase 1 requires BBox-ranked teacher schema_version=2")
    config_name = str(raw.get("config_name", "")).strip()
    if not config_name:
        raise ValueError("Teacher config_name must not be empty")

    labels = _mapping(raw.get("labels", {}), "labels")
    xml = _mapping(raw.get("xml", {}), "xml")
    global_cfg = _mapping(raw.get("global_candidate", {}), "global_candidate")
    local_cfg = _mapping(raw.get("local_candidate", {}), "local_candidate")
    ranking = _mapping(raw.get("ranking", {}), "ranking")
    selection = _mapping(raw.get("selection", {}), "selection")

    if (
        int(labels.get("ignore", -999)),
        int(labels.get("background", -999)),
        int(labels.get("positive", -999)),
    ) != (-1, 0, 1):
        raise ValueError("Teacher labels must be ignore=-1, background=0, positive=1")
    no_bbox_label = int(labels.get("no_bbox_label", -999))
    if no_bbox_label != 0:
        raise ValueError("Phase 1 fixed teacher requires no_bbox_label=0")
    bbox_policy = str(labels.get("bbox_inside_non_contour_label", ""))
    if bbox_policy != "ignore":
        raise ValueError("Phase 1 fixed teacher requires BBox non-contour=ignore")

    offsets_raw = xml.get("frame_number_offsets", [1])
    if not isinstance(offsets_raw, Sequence) or isinstance(offsets_raw, str):
        raise ValueError("xml.frame_number_offsets must be a list")
    offsets = tuple(int(value) for value in offsets_raw)
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("xml.frame_number_offsets must be non-empty and unique")
    frame_id_source = str(xml.get("frame_id_source", "frame_index"))
    if frame_id_source not in {"frame_index", "frame_order", "both"}:
        raise ValueError(f"Unknown xml.frame_id_source: {frame_id_source}")
    annotation_dir_name = str(xml.get("annotation_dir_name", "")).strip()
    annotation_dir = Path(annotation_dir_name)
    if (
        not annotation_dir_name
        or annotation_dir.is_absolute()
        or len(annotation_dir.parts) != 1
        or annotation_dir.name in {".", ".."}
    ):
        raise ValueError("xml.annotation_dir_name must be one relative directory")
    strict_xml = _bool(xml.get("strict_annotation_dir", False), "xml.strict_annotation_dir")
    if not strict_xml:
        raise ValueError("Phase 1 requires strict XML annotation lookup")

    texture = AlphaTextureConfig(
        texture_style="threshold_alpha",
        denoise=str(global_cfg.get("denoise", "median")),
        denoise_ksize=int(global_cfg.get("denoise_ksize", 3)),
        threshold_mode=str(global_cfg.get("threshold_mode", "percentile")),
        fixed_threshold=int(global_cfg.get("fixed_threshold", 180)),
        percentile=float(global_cfg.get("percentile", 85.0)),
        open_ksize=int(global_cfg.get("open_ksize", 3)),
        close_ksize=int(global_cfg.get("close_ksize", 5)),
        morph_shape=str(global_cfg.get("morph_shape", "ellipse")),
        min_component_area=int(global_cfg.get("min_component_area", 100)),
        white_foreground=True,
    )
    if texture.threshold_mode not in {"fixed", "otsu", "percentile", "adaptive"}:
        raise ValueError(f"Unknown global threshold mode: {texture.threshold_mode}")
    if texture.denoise not in {"none", "median", "gaussian", "bilateral"}:
        raise ValueError(f"Unknown global denoise mode: {texture.denoise}")
    if texture.morph_shape not in {"rect", "ellipse", "cross"}:
        raise ValueError(f"Unknown global morph shape: {texture.morph_shape}")
    if not 0.0 <= texture.percentile <= 100.0:
        raise ValueError("global percentile must be in [0, 100]")
    if min(texture.open_ksize, texture.close_ksize, texture.min_component_area) < 0:
        raise ValueError("global cleanup values must be >= 0")

    ranked = RankedContourConfig(
        local_window_size=int(local_cfg.get("window_size", 31)),
        local_percentile=float(local_cfg.get("percentile", 75.0)),
        local_min_contrast=float(local_cfg.get("min_contrast", 12.0)),
        local_cleanup_open_ksize=int(local_cfg.get("cleanup_open_ksize", 3)),
        local_cleanup_close_ksize=int(local_cfg.get("cleanup_close_ksize", 5)),
        local_cleanup_morph_shape=str(local_cfg.get("cleanup_morph_shape", "ellipse")),
        local_cleanup_min_component_area=int(local_cfg.get("cleanup_min_component_area", 15)),
        min_area_ratio=float(ranking.get("min_area_ratio", 0.02)),
        sufficient_area_ratio=float(ranking.get("sufficient_area_ratio", 0.10)),
        max_center_distance_norm=float(ranking.get("max_center_distance_norm", 0.50)),
        center_weight=float(ranking.get("center_weight", 0.75)),
        area_weight=float(ranking.get("area_weight", 0.25)),
    )
    ranked.validate()
    if selection.get("union_sources_before_ranking", False):
        raise ValueError("Phase 1 requires independent global/local candidates")
    if selection.get("equal_rank_different_masks", "reject") != "reject":
        raise ValueError("Phase 1 requires equal-rank different masks to be rejected")

    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return AuditTeacherConfig(
        path=path.resolve(),
        raw=raw,
        config_name=config_name,
        file_sha256=file_sha256(path),
        fingerprint=fingerprint,
        xml_frame_number_offsets=offsets,
        xml_frame_id_source=frame_id_source,
        xml_annotation_dir_name=annotation_dir_name,
        strict_xml_annotation_dir=strict_xml,
        no_bbox_label=no_bbox_label,
        bbox_inside_non_contour_label=bbox_policy,
        texture_config=texture,
        min_alpha=int(global_cfg.get("min_alpha", 1)),
        min_contour_area=float(global_cfg.get("min_area", 20.0)),
        ranked_config=ranked,
    )


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _resolve_annotated_h5(
    root: Path,
    video_name: str,
    glob_template: str,
) -> Path:
    try:
        pattern = glob_template.format(video_name=video_name)
    except KeyError as exc:
        raise ValueError(
            "annotated_glob_template only supports {video_name}"
        ) from exc
    matches = sorted(path for path in Path(root).glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one annotated H5 for {video_name}: "
            f"root={root}, pattern={pattern}, matches={len(matches)}"
        )
    return matches[0].resolve()


def resolve_inputs(
    items: Sequence[Stage4SweepManifestItem],
    *,
    annotated_root: Path,
    annotated_glob_template: str,
    expected_videos: int,
) -> list[ResolvedInput]:
    enabled = [item for item in items if item.enabled]
    if expected_videos > 0 and len(enabled) != expected_videos:
        raise ValueError(
            f"Enabled video count mismatch: expected={expected_videos}, actual={len(enabled)}"
        )
    resolved = [
        ResolvedInput(
            item=item,
            annotated_h5=_resolve_annotated_h5(
                annotated_root,
                item.video_name,
                annotated_glob_template,
            ),
        )
        for item in enabled
    ]
    if len({str(item.annotated_h5) for item in resolved}) != len(resolved):
        raise ValueError("One annotated H5 resolved to multiple manifest rows")
    return resolved


def _snapshot(paths: Iterable[Path]) -> dict[str, FileSnapshot]:
    result: dict[str, FileSnapshot] = {}
    for path in sorted({Path(value).resolve() for value in paths}, key=str):
        stat = path.stat()
        result[str(path)] = FileSnapshot(str(path), int(stat.st_size), int(stat.st_mtime_ns))
    return result


def _assert_snapshot_unchanged(before: Mapping[str, FileSnapshot]) -> None:
    after = _snapshot(Path(path) for path in before)
    if before != after:
        changed = [path for path in before if before.get(path) != after.get(path)]
        raise RuntimeError(f"Read-only audit input changed during execution: {changed}")


def _read_pseudo3d(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        missing = [key for key in ("local_encoder_images", "frame_indices") if key not in handle]
        if missing:
            raise KeyError(f"pseudo3D H5 missing datasets: {missing}")
        images = handle["local_encoder_images"][:]
        frame_indices = handle["frame_indices"][:].astype(np.int64)
        attrs = dict(handle.attrs)
    if images.ndim not in {3, 4}:
        raise ValueError(f"Unsupported local_encoder_images shape: {images.shape}")
    if images.shape[0] != frame_indices.shape[0]:
        raise ValueError("local_encoder_images/frame_indices length mismatch")
    if np.unique(frame_indices).size != frame_indices.size:
        raise ValueError("frame_indices contains duplicates")
    return images, frame_indices, attrs


def _preflight_pseudo3d(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with h5py.File(path, "r") as handle:
        missing = [key for key in ("local_encoder_images", "frame_indices") if key not in handle]
        if missing:
            raise KeyError(f"pseudo3D H5 missing datasets: {missing}")
        image_shape = tuple(int(value) for value in handle["local_encoder_images"].shape)
        frame_indices = handle["frame_indices"][:].astype(np.int64)
    if len(image_shape) not in {3, 4}:
        raise ValueError(f"Unsupported local_encoder_images shape: {image_shape}")
    if image_shape[0] != frame_indices.shape[0]:
        raise ValueError("local_encoder_images/frame_indices length mismatch")
    if np.unique(frame_indices).size != frame_indices.size:
        raise ValueError("frame_indices contains duplicates")
    return image_shape, frame_indices


def _read_frame_annotation(group: h5py.Group) -> dict[str, np.ndarray]:
    missing = REQUIRED_FRAME_DATASETS - set(group.keys())
    if missing:
        raise KeyError(f"annotated H5 missing frame_annotation datasets: {sorted(missing)}")
    result: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for key in REQUIRED_FRAME_DATASETS:
        values = group[key][:]
        lengths.add(int(values.shape[0]))
        if key in STRING_FRAME_DATASETS:
            values = np.asarray([_decode_text(value) for value in values], dtype=object)
        result[key] = values
    if len(lengths) != 1:
        raise ValueError(f"frame_annotation dataset lengths differ: {sorted(lengths)}")
    return result


def _read_annotated_h5(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    required_pc = ("frame_order", "frame_index", "pixel_xy")
    with h5py.File(path, "r") as handle:
        for group_name in ("point_cloud", "annotation", "frame_annotation"):
            if group_name not in handle:
                raise KeyError(f"annotated H5 missing group: {group_name}")
        pc_group = handle["point_cloud"]
        missing_pc = [key for key in required_pc if key not in pc_group]
        if missing_pc:
            raise KeyError(f"annotated H5 missing point_cloud datasets: {missing_pc}")
        if "point_label" not in handle["annotation"]:
            raise KeyError("annotated H5 missing annotation/point_label")
        point_cloud = {key: pc_group[key][:] for key in required_pc}
        point_cloud["point_label"] = handle["annotation/point_label"][:].astype(np.int8)
        frame_annotation = _read_frame_annotation(handle["frame_annotation"])
        attrs = dict(handle.attrs)
    num_points = int(point_cloud["frame_order"].shape[0])
    if point_cloud["pixel_xy"].shape != (num_points, 2):
        raise ValueError("point_cloud/pixel_xy must have shape [N,2]")
    if point_cloud["point_label"].shape != (num_points,):
        raise ValueError("annotation/point_label must align with point_cloud")
    return point_cloud, frame_annotation, attrs


def _preflight_annotated_h5(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    """Validate schema without loading point-level arrays during preflight."""
    with h5py.File(path, "r") as handle:
        for group_name in ("point_cloud", "annotation", "frame_annotation"):
            if group_name not in handle:
                raise KeyError(f"annotated H5 missing group: {group_name}")
        pc_group = handle["point_cloud"]
        for key in ("frame_order", "frame_index", "pixel_xy"):
            if key not in pc_group:
                raise KeyError(f"annotated H5 missing point_cloud/{key}")
        num_points = int(pc_group["frame_order"].shape[0])
        if pc_group["frame_index"].shape != (num_points,):
            raise ValueError("point_cloud/frame_index must align with frame_order")
        if pc_group["pixel_xy"].shape != (num_points, 2):
            raise ValueError("point_cloud/pixel_xy must have shape [N,2]")
        if "point_label" not in handle["annotation"]:
            raise KeyError("annotated H5 missing annotation/point_label")
        if handle["annotation/point_label"].shape != (num_points,):
            raise ValueError("annotation/point_label must align with point_cloud")
        frame_annotation = _read_frame_annotation(handle["frame_annotation"])
        attrs = dict(handle.attrs)
    return frame_annotation, attrs, num_points


def _attr_text(attrs: Mapping[str, Any], key: str, default: str = "") -> str:
    return _decode_text(attrs.get(key, default))


def _assert_numeric_equal(name: str, actual: float, expected: float, atol: float = 1e-5) -> None:
    actual_f = float(actual)
    expected_f = float(expected)
    if math.isnan(actual_f) and math.isnan(expected_f):
        return
    if not math.isfinite(actual_f) or not math.isfinite(expected_f):
        if actual_f == expected_f:
            return
        raise ValueError(f"{name} mismatch: actual={actual_f}, expected={expected_f}")
    if not math.isclose(actual_f, expected_f, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{name} mismatch: actual={actual_f}, expected={expected_f}")


def _assert_metadata_matches_teacher(attrs: Mapping[str, Any], teacher: AuditTeacherConfig) -> None:
    expected_text = {
        "label_mode": "bbox_ranked_global_local",
        "bbox_inside_non_contour_label": teacher.bbox_inside_non_contour_label,
        "xml_frame_id_source": teacher.xml_frame_id_source,
        "xml_annotation_dir_name": teacher.xml_annotation_dir_name,
        "ranked_teacher_schema": "stage4_bbox_ranked_teacher_v2",
        "ranked_local_cleanup_morph_shape": teacher.ranked_config.local_cleanup_morph_shape,
    }
    for key, expected in expected_text.items():
        actual = _attr_text(attrs, key)
        if actual != expected:
            raise ValueError(f"annotated metadata {key} mismatch: {actual!r} != {expected!r}")
    if int(attrs.get("no_bbox_label", -999)) != teacher.no_bbox_label:
        raise ValueError("annotated metadata no_bbox_label mismatch")
    if not bool(attrs.get("strict_xml_annotation_dir", False)):
        raise ValueError("annotated metadata strict_xml_annotation_dir is false")

    stored_ranked = ranked_contour_config_from_metadata(dict(attrs))
    if stored_ranked != teacher.ranked_config:
        raise ValueError(
            f"annotated ranked config differs from teacher config: {stored_ranked} != {teacher.ranked_config}"
        )
    numeric = {
        "contour_percentile": teacher.texture_config.percentile,
        "contour_fixed_threshold": teacher.texture_config.fixed_threshold,
        "contour_min_component_area": teacher.texture_config.min_component_area,
        "contour_min_area": teacher.min_contour_area,
        "contour_min_alpha": teacher.min_alpha,
        "contour_open_ksize": teacher.texture_config.open_ksize,
        "contour_close_ksize": teacher.texture_config.close_ksize,
        "contour_denoise_ksize": teacher.texture_config.denoise_ksize,
    }
    for key, expected in numeric.items():
        _assert_numeric_equal(f"metadata/{key}", attrs.get(key, np.nan), expected)
    for key, expected in {
        "contour_threshold_mode": teacher.texture_config.threshold_mode,
        "contour_denoise": teacher.texture_config.denoise,
    }.items():
        if _attr_text(attrs, key) != expected:
            raise ValueError(f"annotated metadata {key} mismatch")


def _bbox_bounds(bbox: LocalBBox, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if not bbox.valid:
        return None
    h, w = shape
    x1, y1, x2, y2 = (float(value) for value in bbox.local_xyxy)
    left = int(math.floor(max(0.0, x1)))
    top = int(math.floor(max(0.0, y1)))
    right = int(math.ceil(min(float(w - 1), x2)))
    bottom = int(math.ceil(min(float(h - 1), y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def mask_shape_metrics(mask: np.ndarray, bbox: LocalBBox) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    bounds = _bbox_bounds(bbox, mask.shape)
    empty = {
        "filled_area": 0,
        "bbox_area": 0,
        "area_ratio": math.nan,
        "center_x": math.nan,
        "center_y": math.nan,
        "center_distance_norm": math.nan,
        "border_contact_pixels": 0,
        "border_contact_ratio": math.nan,
        "touch_top": False,
        "touch_bottom": False,
        "touch_left": False,
        "touch_right": False,
        "component_count": 0,
        "contour_area": 0.0,
        "perimeter": 0.0,
        "solidity": math.nan,
        "extent": math.nan,
        "compactness": math.nan,
        "contour_bbox_area_ratio": math.nan,
    }
    if bounds is None:
        return empty
    left, top, right, bottom = bounds
    roi = mask[top : bottom + 1, left : right + 1].astype(np.uint8)
    bbox_area = int(roi.size)
    filled_area = int(roi.sum())
    result = dict(empty)
    result.update(
        bbox_area=bbox_area,
        filled_area=filled_area,
        area_ratio=float(filled_area / bbox_area) if bbox_area else math.nan,
    )
    if filled_area == 0:
        return result

    num_components, _ = cv2.connectedComponents(roi, connectivity=8)
    result["component_count"] = int(num_components - 1)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    x, y, width, height = cv2.boundingRect(contour)
    contour_rect_area = int(width * height)

    moments = cv2.moments(roi, binaryImage=True)
    if moments["m00"] > 0:
        center_x = float(left + moments["m10"] / moments["m00"])
        center_y = float(top + moments["m01"] / moments["m00"])
        bbox_center_x = 0.5 * (left + right)
        bbox_center_y = 0.5 * (top + bottom)
        bbox_diag = float(math.hypot(right - left + 1, bottom - top + 1))
        center_distance = float(
            math.hypot(center_x - bbox_center_x, center_y - bbox_center_y) / bbox_diag
        )
    else:
        center_x = center_y = center_distance = math.nan

    boundary = np.zeros_like(roi)
    cv2.drawContours(boundary, [contour], 0, color=1, thickness=1)
    edge = np.zeros_like(roi, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    boundary_bool = boundary.astype(bool)
    contact_pixels = int(np.sum(boundary_bool & edge))
    boundary_pixels = int(boundary_bool.sum())

    result.update(
        center_x=center_x,
        center_y=center_y,
        center_distance_norm=center_distance,
        border_contact_pixels=contact_pixels,
        border_contact_ratio=(
            float(contact_pixels / boundary_pixels) if boundary_pixels else math.nan
        ),
        touch_top=bool(np.any(boundary_bool[0, :])),
        touch_bottom=bool(np.any(boundary_bool[-1, :])),
        touch_left=bool(np.any(boundary_bool[:, 0])),
        touch_right=bool(np.any(boundary_bool[:, -1])),
        contour_area=contour_area,
        perimeter=perimeter,
        solidity=float(contour_area / hull_area) if hull_area > 0 else math.nan,
        extent=float(filled_area / contour_rect_area) if contour_rect_area else math.nan,
        compactness=(
            float(4.0 * math.pi * contour_area / (perimeter * perimeter))
            if perimeter > 0
            else math.nan
        ),
        contour_bbox_area_ratio=(
            float(contour_rect_area / bbox_area) if bbox_area else math.nan
        ),
    )
    return result


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_b = np.asarray(first, dtype=bool)
    second_b = np.asarray(second, dtype=bool)
    if first_b.shape != second_b.shape:
        raise ValueError("mask IoU requires identical shapes")
    union = int(np.sum(first_b | second_b))
    if union == 0:
        return math.nan
    return float(np.sum(first_b & second_b) / union)


def candidate_comparison_metrics(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if bool(candidate["eligible"])]
    eligible.sort(key=ranked_contour_candidate_sort_key, reverse=True)
    global_candidates = [candidate for candidate in eligible if candidate["source"] == "global"]
    local_candidates = [
        candidate for candidate in eligible if candidate["source"] == "local_percentile"
    ]
    global_candidates.sort(key=ranked_contour_candidate_sort_key, reverse=True)
    local_candidates.sort(key=ranked_contour_candidate_sort_key, reverse=True)

    result: dict[str, Any] = {
        "top1_score": math.nan,
        "top2_score": math.nan,
        "score_margin": math.nan,
        "top1_source": "none",
        "top2_source": "none",
        "top1_top2_iou": math.nan,
        "global_local_best_iou": math.nan,
        "candidate_center_distance": math.nan,
    }
    if eligible:
        result["top1_score"] = float(eligible[0]["score"])
        result["top1_source"] = str(eligible[0]["source"])
    if len(eligible) >= 2:
        first, second = eligible[:2]
        result["top2_score"] = float(second["score"])
        result["top2_source"] = str(second["source"])
        result["score_margin"] = float(first["score"] - second["score"])
        result["top1_top2_iou"] = mask_iou(first["mask"], second["mask"])
        result["candidate_center_distance"] = float(
            math.hypot(
                float(first["center_x"]) - float(second["center_x"]),
                float(first["center_y"]) - float(second["center_y"]),
            )
        )
    if global_candidates and local_candidates:
        result["global_local_best_iou"] = mask_iou(
            global_candidates[0]["mask"], local_candidates[0]["mask"]
        )
    return result


def _saved_row_map(frame: Mapping[str, np.ndarray]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for index, (order, bbox_index) in enumerate(
        zip(frame["frame_order"], frame["bbox_index"], strict=True)
    ):
        key = (int(order), int(bbox_index))
        if key in result:
            raise ValueError(f"Duplicate saved frame_annotation key: {key}")
        result[key] = index
    return result


def _points_in_bbox(
    frame_xy: np.ndarray,
    bbox: LocalBBox,
    shape: tuple[int, int],
) -> int:
    bounds = _bbox_bounds(bbox, shape)
    if bounds is None or frame_xy.size == 0:
        return 0
    left, top, right, bottom = bounds
    return int(
        np.sum(
            (frame_xy[:, 0] >= left)
            & (frame_xy[:, 0] <= right)
            & (frame_xy[:, 1] >= top)
            & (frame_xy[:, 1] <= bottom)
        )
    )


def _points_in_mask(frame_xy: np.ndarray, mask: np.ndarray) -> int:
    if frame_xy.size == 0:
        return 0
    xy = frame_xy.copy()
    xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
    return int(np.sum(mask[xy[:, 1], xy[:, 0]]))


def _compare_saved_result(
    frame: Mapping[str, np.ndarray],
    saved_index: int,
    result: Mapping[str, Any],
    *,
    positive_points: int,
) -> None:
    exact = {
        "valid_contour": bool(result["valid"]),
        "annotation_reason": str(result["reason"]),
        "selected_contour_source": str(result["selected_contour_source"]),
        "contour_positive_points": int(positive_points),
        "num_global_candidates": int(result["num_global_candidates"]),
        "num_local_candidates": int(result["num_local_candidates"]),
        "num_global_eligible_candidates": int(result["num_global_eligible_candidates"]),
        "num_local_eligible_candidates": int(result["num_local_eligible_candidates"]),
    }
    for key, expected in exact.items():
        actual = frame[key][saved_index]
        if isinstance(expected, str):
            actual = _decode_text(actual)
        elif isinstance(expected, bool):
            actual = bool(actual)
        else:
            actual = int(actual)
        if actual != expected:
            raise ValueError(f"saved {key} mismatch: actual={actual!r}, expected={expected!r}")
    numeric = {
        "contour_area": result["contour_area"],
        "foreground_ratio_in_bbox": result["foreground_ratio_in_bbox"],
        "contour_selection_score": result["contour_selection_score"],
        "center_distance_norm": result["center_distance_norm"],
        "selected_area_ratio": result["selected_area_ratio"],
        "global_candidate_score": result["global_candidate_score"],
        "local_candidate_score": result["local_candidate_score"],
    }
    for key, expected in numeric.items():
        _assert_numeric_equal(f"saved frame_annotation/{key}", frame[key][saved_index], expected)


def _finite_or_nan(value: Any) -> float:
    value_f = float(value)
    return value_f if math.isfinite(value_f) else math.nan


def _validate_audit_row_metrics(row: Mapping[str, Any]) -> None:
    if not str(row.get("annotation_reason", "")).strip():
        raise ValueError("Audit row has no annotation_reason")
    for key in (
        "num_frame_points",
        "bbox_point_count",
        "positive_point_count",
        "num_global_candidates",
        "num_local_candidates",
        "num_global_eligible_candidates",
        "num_local_eligible_candidates",
    ):
        if int(row[key]) < 0:
            raise ValueError(f"Audit row has negative {key}")
    if int(row["positive_point_count"]) > int(row["bbox_point_count"]):
        raise ValueError("positive_point_count exceeds bbox_point_count")
    if bool(row["valid_contour"]):
        required_finite = (
            "contour_selection_score",
            "contour_area",
            "selected_area_ratio",
            "foreground_ratio_in_bbox",
            "area_ratio",
            "center_x",
            "center_y",
            "center_distance_norm",
            "border_contact_ratio",
            "perimeter",
            "solidity",
            "extent",
            "compactness",
            "contour_bbox_area_ratio",
        )
        non_finite = [
            key for key in required_finite if not math.isfinite(float(row[key]))
        ]
        if non_finite:
            raise ValueError(f"Valid audit row has non-finite metrics: {non_finite}")


def audit_video(resolved: ResolvedInput, teacher: AuditTeacherConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item = resolved.item
    images, frame_indices, pseudo_attrs = _read_pseudo3d(item.pseudo3d_h5)
    point_cloud, saved_frame, annotated_attrs = _read_annotated_h5(resolved.annotated_h5)
    _assert_metadata_matches_teacher(annotated_attrs, teacher)
    if _attr_text(annotated_attrs, "video_name") != item.video_name:
        raise ValueError("annotated metadata video_name mismatch")
    if point_cloud["frame_order"].size:
        orders = point_cloud["frame_order"].astype(np.int64)
        if orders.min() < 0 or orders.max() >= len(frame_indices):
            raise ValueError("point_cloud/frame_order is outside pseudo3D frame range")
        if not np.array_equal(
            point_cloud["frame_index"].astype(np.int64), frame_indices[orders]
        ):
            raise ValueError("point_cloud frame_order/frame_index alignment mismatch")

    voc_bboxes = load_voc_bboxes(
        item.voc_xml_root,
        video_name=item.video_name,
        frame_indices=frame_indices,
        xml_frame_number_offsets=teacher.xml_frame_number_offsets,
        xml_frame_id_source=teacher.xml_frame_id_source,
        xml_annotation_dir_name=teacher.xml_annotation_dir_name,
        strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
    )
    if not voc_bboxes:
        raise ValueError("No strict VOC BBox matched this video")
    shape = tuple(int(value) for value in image_to_uint8_gray(images[0]).shape)
    local_bboxes = {
        order: [
            xml_bbox_to_local(
                voc.xml_xyxy,
                image_size_wh=voc.image_size_wh,
                attrs=pseudo_attrs,
                local_shape_hw=shape,
            )
            for voc in boxes
        ]
        for order, boxes in voc_bboxes.items()
    }
    expected_rows = sum(len(value) for value in voc_bboxes.values())
    if len(saved_frame["frame_order"]) != expected_rows:
        raise ValueError(
            f"saved BBox row count mismatch: saved={len(saved_frame['frame_order'])}, XML={expected_rows}"
        )
    saved_map = _saved_row_map(saved_frame)
    expected_keys = {
        (int(order), int(index))
        for order, boxes in voc_bboxes.items()
        for index in range(len(boxes))
    }
    if set(saved_map) != expected_keys:
        raise ValueError("saved frame_annotation keys differ from strict XML BBoxes")

    frame_order_values = point_cloud["frame_order"].astype(np.int64)
    rounded_xy = np.rint(point_cloud["pixel_xy"]).astype(np.int64)
    rows: list[dict[str, Any]] = []
    contour_unions: dict[int, np.ndarray] = {}
    for order in sorted(voc_bboxes):
        global_binary = build_full_frame_binary_mask(
            images[order], texture_config=teacher.texture_config, min_alpha=teacher.min_alpha
        )
        local_binary = build_ranked_local_binary_mask(
            images[order], config=teacher.ranked_config
        )
        frame_point_mask = frame_order_values == int(order)
        frame_xy = rounded_xy[frame_point_mask]
        num_frame_points = int(frame_xy.shape[0])
        contour_union = np.zeros(global_binary.shape, dtype=bool)
        for bbox_index, (voc, bbox) in enumerate(
            zip(voc_bboxes[order], local_bboxes[order], strict=True)
        ):
            result = build_bbox_ranked_contour_mask(
                global_binary=global_binary,
                local_binary=local_binary,
                bbox=bbox,
                min_contour_area=teacher.min_contour_area,
                config=teacher.ranked_config,
            )
            candidates = build_bbox_ranked_contour_candidates(
                global_binary=global_binary,
                local_binary=local_binary,
                bbox=bbox,
                min_contour_area=teacher.min_contour_area,
                config=teacher.ranked_config,
            )
            positive_points = _points_in_mask(frame_xy, result["mask"]) if result["valid"] else 0
            bbox_points = _points_in_bbox(frame_xy, bbox, global_binary.shape)
            saved_index = saved_map[(int(order), int(bbox_index))]
            _compare_saved_result(
                saved_frame, saved_index, result, positive_points=positive_points
            )
            if int(saved_frame["num_frame_points"][saved_index]) != num_frame_points:
                raise ValueError("saved num_frame_points mismatch")
            if int(saved_frame["num_labeled_points"][saved_index]) != positive_points:
                raise ValueError("saved num_labeled_points mismatch")
            if int(saved_frame["frame_index"][saved_index]) != int(frame_indices[order]):
                raise ValueError("saved frame_index mismatch")
            if int(saved_frame["object_index"][saved_index]) != int(voc.object_index):
                raise ValueError("saved object_index mismatch")
            if _decode_text(saved_frame["object_name"][saved_index]) != str(voc.object_name):
                raise ValueError("saved object_name mismatch")
            if _decode_text(saved_frame["xml_path"][saved_index]) != str(voc.xml_path):
                raise ValueError("saved XML path mismatch")
            np.testing.assert_allclose(
                saved_frame["bbox_local_xyxy"][saved_index].astype(np.float64),
                bbox.local_xyxy.astype(np.float64),
                rtol=0.0,
                atol=1e-5,
                err_msg="saved bbox_local_xyxy mismatch",
            )
            if result["valid"]:
                contour_union |= np.asarray(result["mask"], dtype=bool)

            selected_metrics = mask_shape_metrics(result["mask"], bbox)
            comparison = candidate_comparison_metrics(candidates)
            source_component_count = int(len(candidates))
            global_source_ratio = next(
                (
                    float(candidate["source_foreground_ratio"])
                    for candidate in candidates
                    if candidate["source"] == "global"
                ),
                0.0,
            )
            local_source_ratio = next(
                (
                    float(candidate["source_foreground_ratio"])
                    for candidate in candidates
                    if candidate["source"] == "local_percentile"
                ),
                0.0,
            )
            row: dict[str, Any] = {
                "video_name": item.video_name,
                "split": item.split,
                "pseudo3d_h5": str(item.pseudo3d_h5),
                "annotated_h5": str(resolved.annotated_h5),
                "xml_path": str(voc.xml_path),
                "frame_order": int(order),
                "frame_index": int(frame_indices[order]),
                "object_index": int(voc.object_index),
                "bbox_index": int(bbox_index),
                "object_name": str(voc.object_name),
                "image_height": int(global_binary.shape[0]),
                "image_width": int(global_binary.shape[1]),
                "bbox_xml_xyxy": json.dumps([float(value) for value in voc.xml_xyxy]),
                "bbox_raw_xyxy": json.dumps([float(value) for value in bbox.raw_xyxy]),
                "bbox_local_xyxy": json.dumps([float(value) for value in bbox.local_xyxy]),
                "bbox_valid": bool(bbox.valid),
                "valid_contour": bool(result["valid"]),
                "annotation_reason": str(result["reason"]),
                "selected_contour_source": str(result["selected_contour_source"]),
                "contour_selection_score": _finite_or_nan(result["contour_selection_score"]),
                "contour_area": float(result["contour_area"]),
                "selected_area_ratio": _finite_or_nan(result["selected_area_ratio"]),
                "foreground_ratio_in_bbox": float(result["foreground_ratio_in_bbox"]),
                "num_frame_points": num_frame_points,
                "bbox_point_count": bbox_points,
                "positive_point_count": positive_points,
                "positive_point_ratio_in_bbox": (
                    float(positive_points / bbox_points) if bbox_points else math.nan
                ),
                "num_global_candidates": int(result["num_global_candidates"]),
                "num_local_candidates": int(result["num_local_candidates"]),
                "num_global_eligible_candidates": int(result["num_global_eligible_candidates"]),
                "num_local_eligible_candidates": int(result["num_local_eligible_candidates"]),
                "global_candidate_score": _finite_or_nan(result["global_candidate_score"]),
                "local_candidate_score": _finite_or_nan(result["local_candidate_score"]),
                "global_foreground_ratio_in_bbox": global_source_ratio,
                "local_foreground_ratio_in_bbox": local_source_ratio,
                "source_component_count": source_component_count,
                **selected_metrics,
                **comparison,
            }
            _validate_audit_row_metrics(row)
            rows.append(row)
        contour_unions[int(order)] = contour_union

    actual_labels = point_cloud["point_label"].astype(np.int8)
    unexpected_labels = np.setdiff1d(
        np.unique(actual_labels), np.asarray([-1, 0, 1], dtype=np.int8)
    )
    if unexpected_labels.size:
        raise ValueError(f"Unexpected point labels: {unexpected_labels.tolist()}")
    for order in np.unique(frame_order_values):
        order_int = int(order)
        frame_mask = frame_order_values == order_int
        frame_xy = rounded_xy[frame_mask]
        expected = np.zeros(frame_xy.shape[0], dtype=np.int8)
        if order_int in local_bboxes:
            bbox_union = np.zeros(shape, dtype=bool)
            for bbox in local_bboxes[order_int]:
                bounds = _bbox_bounds(bbox, shape)
                if bounds is None:
                    continue
                left, top, right, bottom = bounds
                bbox_union[top : bottom + 1, left : right + 1] = True
            xy = frame_xy.copy()
            xy[:, 0] = np.clip(xy[:, 0], 0, shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, shape[0] - 1)
            expected[bbox_union[xy[:, 1], xy[:, 0]]] = -1
            contour_union = contour_unions[order_int]
            expected[contour_union[xy[:, 1], xy[:, 0]]] = 1
        if not np.array_equal(actual_labels[frame_mask], expected):
            differing = int(np.sum(actual_labels[frame_mask] != expected))
            raise ValueError(
                f"saved point labels differ from recomputed teacher at frame_order={order_int}: "
                f"differing_points={differing}"
            )

    stats = {
        "video_name": item.video_name,
        "num_frames": int(images.shape[0]),
        "num_points": int(frame_order_values.size),
        "num_bbox_rows": len(rows),
        "num_valid_contours": int(sum(bool(row["valid_contour"]) for row in rows)),
        "num_invalid_contours": int(sum(not bool(row["valid_contour"]) for row in rows)),
        "num_positive_points": int(np.sum(point_cloud["point_label"] == LABEL_FEMUR_CANDIDATE)),
    }
    return rows, stats


def _sort_float(value: Any, *, missing: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return missing
    return result if math.isfinite(result) else missing


def select_category_rows(rows: Sequence[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    valid = [row for row in rows if bool(row["valid_contour"])]
    invalid = [row for row in rows if not bool(row["valid_contour"])]
    multiple = [
        row
        for row in rows
        if int(row["num_global_eligible_candidates"])
        + int(row["num_local_eligible_candidates"])
        >= 2
    ]

    def identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
        return (str(row["video_name"]), int(row["frame_order"]), int(row["bbox_index"]))

    categories = {
        "overfilled": sorted(
            valid,
            key=lambda row: (
                max(
                    _sort_float(row["selected_area_ratio"], missing=-1.0),
                    _sort_float(row["foreground_ratio_in_bbox"], missing=-1.0),
                ),
                identity(row),
            ),
            reverse=True,
        ),
        "border_contact": sorted(
            valid,
            key=lambda row: (
                _sort_float(row["border_contact_ratio"], missing=-1.0),
                int(row["border_contact_pixels"]),
                identity(row),
            ),
            reverse=True,
        ),
        "ambiguous": sorted(
            multiple,
            key=lambda row: (
                _sort_float(row["score_margin"], missing=math.inf),
                _sort_float(row["top1_top2_iou"], missing=math.inf),
                identity(row),
            ),
        ),
        "off_center": sorted(
            valid,
            key=lambda row: (
                _sort_float(row["center_distance_norm"], missing=-1.0),
                identity(row),
            ),
            reverse=True,
        ),
        "invalid": sorted(invalid, key=identity),
        "small_contour": sorted(
            valid,
            key=lambda row: (
                int(row["positive_point_count"]),
                _sort_float(row["area_ratio"], missing=math.inf),
                identity(row),
            ),
        ),
        "candidate_good_control": sorted(
            valid,
            key=lambda row: (
                abs(_sort_float(row["area_ratio"], missing=1.0) - 0.10)
                + _sort_float(row["center_distance_norm"], missing=1.0)
                + _sort_float(row["border_contact_ratio"], missing=1.0),
                -_sort_float(row["score_margin"], missing=0.0),
                identity(row),
            ),
        ),
    }
    return {key: value[:top_k] for key, value in categories.items()}


def _row_identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["video_name"]), int(row["frame_order"]), int(row["bbox_index"])


def _draw_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness)


def _render_overlay(
    *,
    output_path: Path,
    row: Mapping[str, Any],
    image: np.ndarray,
    teacher: AuditTeacherConfig,
) -> None:
    gray = image_to_uint8_gray(image)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    local_xyxy = np.asarray(json.loads(str(row["bbox_local_xyxy"])), dtype=np.float32)
    bbox = LocalBBox(local_xyxy.copy(), local_xyxy.copy(), local_xyxy.copy(), bool(row["bbox_valid"]))
    global_binary = build_full_frame_binary_mask(
        image, texture_config=teacher.texture_config, min_alpha=teacher.min_alpha
    )
    local_binary = build_ranked_local_binary_mask(image, config=teacher.ranked_config)
    candidates = build_bbox_ranked_contour_candidates(
        global_binary=global_binary,
        local_binary=local_binary,
        bbox=bbox,
        min_contour_area=teacher.min_contour_area,
        config=teacher.ranked_config,
    )
    result = build_bbox_ranked_contour_mask(
        global_binary=global_binary,
        local_binary=local_binary,
        bbox=bbox,
        min_contour_area=teacher.min_contour_area,
        config=teacher.ranked_config,
    )
    for candidate in candidates:
        color = (255, 255, 255) if candidate["source"] == "global" else (255, 128, 0)
        _draw_contours(canvas, candidate["mask"], color, 1)
    if result["valid"]:
        _draw_contours(canvas, result["mask"], (0, 255, 0), 2)
    left, top, right, bottom = (int(round(value)) for value in local_xyxy)
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 255), 1)
    bbox_center = (int(round(0.5 * (left + right))), int(round(0.5 * (top + bottom))))
    cv2.drawMarker(canvas, bbox_center, (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
    if math.isfinite(_sort_float(row["center_x"], missing=math.nan)):
        contour_center = (int(round(float(row["center_x"]))), int(round(float(row["center_y"]))))
        cv2.drawMarker(canvas, contour_center, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 1)
    lines = [
        f"{row['video_name']} fo={row['frame_order']} fi={row['frame_index']} bbox={row['bbox_index']}",
        f"source={row['selected_contour_source']} valid={int(bool(row['valid_contour']))} reason={row['annotation_reason']}",
        f"score={_sort_float(row['contour_selection_score'], missing=math.nan):.3f} area={_sort_float(row['area_ratio'], missing=math.nan):.3f} center={_sort_float(row['center_distance_norm'], missing=math.nan):.3f}",
        f"border={_sort_float(row['border_contact_ratio'], missing=math.nan):.3f} positive={row['positive_point_count']}/{row['bbox_point_count']}",
    ]
    panel_height = 18 * len(lines) + 6
    output = cv2.copyMakeBorder(canvas, panel_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (4, 16 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise OSError(f"Failed to save overlay: {output_path}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = {key for row in rows for key in row}
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _video_summary(rows: Sequence[dict[str, Any]], base: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_name"])].append(row)
    base_map = {str(row["video_name"]): dict(row) for row in base}
    result = []
    for video_name in sorted(grouped):
        video_rows = grouped[video_name]
        valid = [row for row in video_rows if row["valid_contour"]]
        areas = np.asarray([float(row["area_ratio"]) for row in valid], dtype=np.float64)
        contacts = np.asarray([float(row["border_contact_ratio"]) for row in valid], dtype=np.float64)
        row = base_map[video_name]
        row.update(
            num_valid_contours=len(valid),
            num_invalid_contours=len(video_rows) - len(valid),
            selected_global=sum(row_["selected_contour_source"] == "global" for row_ in video_rows),
            selected_local=sum(row_["selected_contour_source"] == "local_percentile" for row_ in video_rows),
            selected_shared=sum(row_["selected_contour_source"] == "shared" for row_ in video_rows),
            selected_area_ratio_mean=float(np.nanmean(areas)) if areas.size else math.nan,
            selected_area_ratio_max=float(np.nanmax(areas)) if areas.size else math.nan,
            border_contact_ratio_mean=float(np.nanmean(contacts)) if contacts.size else math.nan,
            border_contact_ratio_max=float(np.nanmax(contacts)) if contacts.size else math.nan,
        )
        result.append(row)
    return result


def _prepare_output_root(path: Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output root is not empty; pass --overwrite to replace it: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def preflight_input(resolved: ResolvedInput, teacher: AuditTeacherConfig) -> dict[str, Any]:
    item = resolved.item
    if not item.pseudo3d_h5.is_file():
        raise FileNotFoundError(f"pseudo3D H5 not found: {item.pseudo3d_h5}")
    image_shape, frame_indices = _preflight_pseudo3d(item.pseudo3d_h5)
    frame_annotation, attrs, num_points = _preflight_annotated_h5(
        resolved.annotated_h5
    )
    _assert_metadata_matches_teacher(attrs, teacher)
    voc = load_voc_bboxes(
        item.voc_xml_root,
        video_name=item.video_name,
        frame_indices=frame_indices,
        xml_frame_number_offsets=teacher.xml_frame_number_offsets,
        xml_frame_id_source=teacher.xml_frame_id_source,
        xml_annotation_dir_name=teacher.xml_annotation_dir_name,
        strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
    )
    num_bbox = sum(len(value) for value in voc.values())
    if num_bbox != len(frame_annotation["frame_order"]):
        raise ValueError("strict XML and saved BBox row counts differ")
    return {
        "video_name": item.video_name,
        "frames": int(image_shape[0]),
        "bbox_rows": int(num_bbox),
        "points": int(num_points),
        "annotated_h5": str(resolved.annotated_h5),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    teacher = load_audit_teacher_config(args.teacher_config)
    items = load_stage4_sweep_manifest(args.manifest)
    resolved_inputs = resolve_inputs(
        items,
        annotated_root=args.annotated_root,
        annotated_glob_template=args.annotated_glob_template,
        expected_videos=args.expected_videos,
    )
    print("Stage 4 contour-teacher Phase 1 audit")
    print(f"manifest            : {args.manifest}")
    print(f"annotated_root      : {args.annotated_root}")
    print(f"teacher_config      : {args.teacher_config}")
    print(f"teacher_fingerprint : {teacher.fingerprint}")
    print(f"videos              : {len(resolved_inputs)}")

    preflight_rows = []
    preflight_failures = []
    for index, resolved in enumerate(resolved_inputs, start=1):
        try:
            row = preflight_input(resolved, teacher)
            preflight_rows.append(row)
            print(
                f"[{index}/{len(resolved_inputs)}] [OK] preflight {row['video_name']}: "
                f"frames={row['frames']}, bboxes={row['bbox_rows']}"
            )
        except Exception as exc:
            failure = {
                "stage": "preflight",
                "video_name": resolved.item.video_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            preflight_failures.append(failure)
            print(f"[{index}/{len(resolved_inputs)}] [FAIL] {failure['video_name']}: {failure['error']}")
            if not args.continue_on_error:
                break
    if preflight_failures:
        raise RuntimeError(
            f"Phase 1 preflight failed for {len(preflight_failures)} video(s): "
            f"{preflight_failures[0]['error']}"
        )
    print("Stage 4 contour-teacher Phase 1 preflight passed.")
    if args.preflight_only:
        return {"preflight_rows": preflight_rows, "bbox_rows": [], "failures": []}

    _prepare_output_root(args.output_root, args.overwrite)
    input_paths: list[Path] = []
    for resolved in resolved_inputs:
        input_paths.extend([resolved.item.pseudo3d_h5, resolved.annotated_h5])
        annotation_dir = (
            resolved.item.voc_xml_root
            / resolved.item.video_name
            / teacher.xml_annotation_dir_name
        )
        input_paths.extend(sorted(annotation_dir.glob("*.xml")))
    before = _snapshot(input_paths)

    all_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, resolved in enumerate(resolved_inputs, start=1):
        try:
            rows, stats = audit_video(resolved, teacher)
            all_rows.extend(rows)
            video_rows.append(stats)
            print(
                f"[{index}/{len(resolved_inputs)}] [OK] audit {resolved.item.video_name}: "
                f"bboxes={len(rows)}, valid={stats['num_valid_contours']}, "
                f"invalid={stats['num_invalid_contours']}"
            )
        except Exception as exc:
            failure = {
                "stage": "audit",
                "video_name": resolved.item.video_name,
                "pseudo3d_h5": str(resolved.item.pseudo3d_h5),
                "annotated_h5": str(resolved.annotated_h5),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[{index}/{len(resolved_inputs)}] [FAIL] {failure['video_name']}: {failure['error']}")
            if not args.continue_on_error:
                break

    all_rows.sort(key=_row_identity)
    _write_csv(args.output_root / "bbox_audit.csv", all_rows)
    summary_rows = _video_summary(all_rows, video_rows) if all_rows else []
    _write_csv(args.output_root / "video_summary.csv", summary_rows)
    _write_csv(
        args.output_root / "failures.csv",
        failures,
        fieldnames=("stage", "video_name", "pseudo3d_h5", "annotated_h5", "error"),
    )

    category_rows: list[dict[str, Any]] = []
    categories = select_category_rows(all_rows, args.top_k)
    selected_by_video: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for category, rows in categories.items():
        for rank, row in enumerate(rows, start=1):
            record = {
                "category": category,
                "rank": rank,
                "video_name": row["video_name"],
                "frame_order": row["frame_order"],
                "frame_index": row["frame_index"],
                "bbox_index": row["bbox_index"],
                "valid_contour": row["valid_contour"],
                "annotation_reason": row["annotation_reason"],
                "selected_contour_source": row["selected_contour_source"],
                "selected_area_ratio": row["selected_area_ratio"],
                "border_contact_ratio": row["border_contact_ratio"],
                "center_distance_norm": row["center_distance_norm"],
                "score_margin": row["score_margin"],
                "positive_point_count": row["positive_point_count"],
            }
            category_rows.append(record)
            selected_by_video[str(row["video_name"])].append((category, rank, row))
    _write_csv(args.output_root / "category_summary.csv", category_rows)

    if not args.no_overlays:
        for category in categories:
            (args.output_root / "overlays" / category).mkdir(
                parents=True, exist_ok=True
            )
        resolved_map = {resolved.item.video_name: resolved for resolved in resolved_inputs}
        for video_name in sorted(selected_by_video):
            resolved = resolved_map[video_name]
            images, _, _ = _read_pseudo3d(resolved.item.pseudo3d_h5)
            for category, rank, row in selected_by_video[video_name]:
                name = (
                    f"{rank:03d}_{video_name}__fo{int(row['frame_order']):05d}"
                    f"__fi{int(row['frame_index']):08d}__bbox{int(row['bbox_index']):03d}.png"
                )
                _render_overlay(
                    output_path=args.output_root / "overlays" / category / name,
                    row=row,
                    image=images[int(row["frame_order"])],
                    teacher=teacher,
                )

    _assert_snapshot_unchanged(before)
    reason_counts = Counter(str(row["annotation_reason"]) for row in all_rows)
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "teacher_config": str(teacher.path),
        "teacher_config_sha256": teacher.file_sha256,
        "teacher_fingerprint": teacher.fingerprint,
        "annotated_root": str(Path(args.annotated_root).resolve()),
        "videos_expected": int(args.expected_videos),
        "videos_checked": len(video_rows),
        "bbox_rows": len(all_rows),
        "valid_contours": int(sum(bool(row["valid_contour"]) for row in all_rows)),
        "invalid_contours": int(sum(not bool(row["valid_contour"]) for row in all_rows)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "category_counts": {key: len(value) for key, value in categories.items()},
        "failure_rows": len(failures),
        "input_files_unchanged": True,
    }
    with (args.output_root / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    run_config = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "manifest": str(Path(args.manifest).resolve()),
        "annotated_root": str(Path(args.annotated_root).resolve()),
        "annotated_glob_template": args.annotated_glob_template,
        "expected_videos": int(args.expected_videos),
        "top_k": int(args.top_k),
        "overlays_enabled": not bool(args.no_overlays),
        "teacher_fingerprint": teacher.fingerprint,
        "teacher": teacher.raw,
    }
    with (args.output_root / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, allow_unicode=True, sort_keys=True)

    print("\nPhase 1 audit summary")
    print(f"  videos_checked        : {len(video_rows)}/{len(resolved_inputs)}")
    print(f"  bbox_rows             : {len(all_rows)}")
    print(f"  valid_contours        : {summary['valid_contours']}")
    print(f"  invalid_contours      : {summary['invalid_contours']}")
    print(f"  failure_rows          : {len(failures)}")
    print("  input_files_unchanged : true")
    print(f"  output_root           : {args.output_root}")
    if failures:
        raise RuntimeError(f"Phase 1 audit failed for {len(failures)} video(s)")
    if len(video_rows) != len(resolved_inputs):
        raise RuntimeError("Phase 1 audit did not process every resolved video")
    print("Stage 4 contour-teacher Phase 1 audit passed.")
    return {"bbox_rows": all_rows, "video_rows": video_rows, "failures": failures, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Phase 1 audit of the Stage 4 BBox-ranked contour teacher. "
            "Recomputes current global/local candidates, verifies saved H5 results, "
            "and writes deterministic BBox metrics and ranked overlays."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("stage4_contour_teacher_audit"))
    parser.add_argument(
        "--annotated_glob_template",
        type=str,
        default="{video_name}/*bboxrank_v2_nobbox_bg.h5",
        help="Glob relative to annotated_root; only {video_name} is supported.",
    )
    parser.add_argument("--expected_videos", type=int, default=182)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--no_overlays", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_videos < 0:
        raise SystemExit("--expected_videos must be >= 0")
    if args.top_k <= 0:
        raise SystemExit("--top_k must be > 0")
    try:
        run_audit(args)
    except Exception as exc:
        raise SystemExit(f"Phase 1 audit failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
