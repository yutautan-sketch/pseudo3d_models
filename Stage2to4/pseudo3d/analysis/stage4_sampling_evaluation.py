from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pseudo3d.analysis.stage4_sampling_sweep_config import DenseTeacherConfig
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LocalBBox,
    VocBBox,
    build_bbox_union_mask,
    build_frame_contour_mask_results,
    load_voc_bboxes,
    xml_bbox_to_local,
)
from src.utils.alpha_texture_processing import (
    AlphaTextureConfig,
    image_to_uint8_gray,
)
from src.utils.pseudo3d_sampling import (
    SOURCE_CONTEXT_GRID,
    SOURCE_GLOBAL,
    SOURCE_LOCAL_PERCENTILE,
    SOURCE_TOPHAT,
)


SOURCE_BITS = {
    "global": int(SOURCE_GLOBAL),
    "local": int(SOURCE_LOCAL_PERCENTILE),
    "tophat": int(SOURCE_TOPHAT),
    "context": int(SOURCE_CONTEXT_GRID),
}


@dataclass(frozen=True)
class TeacherBBox:
    video_name: str
    frame_order: int
    frame_index: int
    bbox_index: int
    object_index: int
    object_name: str
    xml_path: str
    local_xyxy: tuple[float, float, float, float]
    roi_bounds: tuple[int, int, int, int] | None
    positive_y: np.ndarray
    positive_x: np.ndarray
    teacher_status: str
    annotation_reason: str
    contour_area: float
    binary_area: int
    foreground_ratio_in_bbox: float

    @property
    def teacher_positive_count(self) -> int:
        return int(self.positive_y.size)

    @property
    def teacher_valid(self) -> bool:
        return self.teacher_status == "teacher_valid"


@dataclass(frozen=True)
class DenseTeacherVideo:
    video_name: str
    frame_indices: np.ndarray
    labels: np.ndarray
    bboxes: tuple[TeacherBBox, ...]
    contour_stats: dict[str, int]


def make_teacher_texture_config(config: DenseTeacherConfig) -> AlphaTextureConfig:
    config.validate()
    return AlphaTextureConfig(
        texture_style="threshold_alpha",
        denoise=config.contour_denoise,
        denoise_ksize=config.contour_denoise_ksize,
        threshold_mode=config.contour_threshold_mode,
        percentile=config.contour_percentile,
        open_ksize=config.contour_open_ksize,
        close_ksize=config.contour_close_ksize,
        morph_shape=config.contour_morph_shape,
        min_component_area=config.contour_min_component_area,
        white_foreground=False,
    )


def _bbox_roi_bounds(
    bbox: LocalBBox,
    *,
    image_shape_hw: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not bbox.valid:
        return None
    h, w = image_shape_hw
    x1, y1, x2, y2 = bbox.local_xyxy.tolist()
    left = int(np.floor(max(0.0, x1)))
    top = int(np.floor(max(0.0, y1)))
    right = int(np.ceil(min(float(w - 1), x2)))
    bottom = int(np.ceil(min(float(h - 1), y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_dense_teacher_video(
    *,
    video_name: str,
    local_images: np.ndarray,
    frame_indices: np.ndarray,
    pseudo3d_attrs: dict[str, Any],
    voc_xml_root: Path,
    config: DenseTeacherConfig,
) -> DenseTeacherVideo:
    """Build sampling-independent dense {-1, 0, 1} teacher maps."""
    config.validate()
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if int(local_images.shape[0]) != int(frame_indices.size):
        raise ValueError(
            "local image/frame index count mismatch: "
            f"images={local_images.shape[0]}, frame_indices={frame_indices.size}"
        )
    if frame_indices.size == 0:
        raise ValueError("Dense teacher input contains no frames")

    frame_shapes = [
        tuple(image_to_uint8_gray(local_images[order]).shape)
        for order in range(int(frame_indices.size))
    ]
    if len(set(frame_shapes)) != 1:
        raise ValueError(f"Local frame shapes are inconsistent: {frame_shapes}")
    image_shape_hw = frame_shapes[0]

    voc_bboxes = load_voc_bboxes(
        Path(voc_xml_root),
        video_name=video_name,
        frame_indices=frame_indices,
        xml_frame_number_offsets=config.xml_frame_number_offsets,
        xml_frame_id_source=config.xml_frame_id_source,
        xml_annotation_dir_name=config.xml_annotation_dir_name,
        strict_xml_annotation_dir=config.strict_xml_annotation_dir,
    )
    local_bboxes: dict[int, list[LocalBBox]] = {
        order: [
            xml_bbox_to_local(
                voc.xml_xyxy,
                image_size_wh=voc.image_size_wh,
                attrs=pseudo3d_attrs,
                local_shape_hw=image_shape_hw,
            )
            for voc in boxes
        ]
        for order, boxes in voc_bboxes.items()
    }

    results_by_order, union_masks, contour_stats = (
        build_frame_contour_mask_results(
            local_images=local_images,
            voc_bboxes=voc_bboxes,
            local_bboxes=local_bboxes,
            texture_config=make_teacher_texture_config(config),
            min_alpha=config.contour_min_alpha,
            min_contour_area=config.contour_min_area,
            point_cloud=None,
            enable_fallback=False,
        )
    )

    labels = np.full(
        (int(frame_indices.size), *image_shape_hw),
        config.label_ignore,
        dtype=np.int8,
    )
    teacher_bboxes: list[TeacherBBox] = []

    for order, order_results in sorted(results_by_order.items()):
        valid_local_boxes = [bbox for bbox in local_bboxes[order] if bbox.valid]
        if valid_local_boxes:
            labels[order].fill(config.label_background)
            bbox_union = build_bbox_union_mask(
                valid_local_boxes,
                image_shape_hw=image_shape_hw,
            )
            labels[order][bbox_union] = config.label_ignore
            positive_union = union_masks.get(order)
            if positive_union is not None:
                labels[order][positive_union] = config.label_positive

        for item in order_results:
            bbox_index = int(item["bbox_index"])
            voc: VocBBox = item["voc"]
            bbox: LocalBBox = item["bbox"]
            result = item["result"]
            mask = np.asarray(result["mask"], dtype=bool)
            positive_y, positive_x = np.nonzero(mask)
            status = (
                "teacher_valid"
                if positive_y.size > 0
                else "teacher_invalid"
            )
            teacher_bboxes.append(
                TeacherBBox(
                    video_name=video_name,
                    frame_order=int(order),
                    frame_index=int(frame_indices[order]),
                    bbox_index=bbox_index,
                    object_index=int(voc.object_index),
                    object_name=str(voc.object_name),
                    xml_path=str(voc.xml_path),
                    local_xyxy=tuple(float(value) for value in bbox.local_xyxy),
                    roi_bounds=_bbox_roi_bounds(
                        bbox,
                        image_shape_hw=image_shape_hw,
                    ),
                    positive_y=positive_y.astype(np.int32),
                    positive_x=positive_x.astype(np.int32),
                    teacher_status=status,
                    annotation_reason=str(result["reason"]),
                    contour_area=float(result["contour_area"]),
                    binary_area=int(result["binary_area"]),
                    foreground_ratio_in_bbox=float(
                        result["foreground_ratio_in_bbox"]
                    ),
                )
            )

    return DenseTeacherVideo(
        video_name=video_name,
        frame_indices=frame_indices.copy(),
        labels=labels,
        bboxes=tuple(teacher_bboxes),
        contour_stats=dict(contour_stats),
    )


def label_counts(selected_mask: np.ndarray, labels: np.ndarray) -> dict[str, int | float]:
    selected = np.asarray(selected_mask, dtype=bool)
    labels = np.asarray(labels, dtype=np.int8)
    if selected.shape != labels.shape:
        raise ValueError(
            f"selected/teacher shape mismatch: {selected.shape} vs {labels.shape}"
        )
    total = int(selected.sum())
    positive = int(np.count_nonzero(selected & (labels == 1)))
    background = int(np.count_nonzero(selected & (labels == 0)))
    ignore = int(np.count_nonzero(selected & (labels == -1)))
    if positive + background + ignore != total:
        raise ValueError("Dense teacher contains labels outside {-1, 0, 1}")
    return {
        "selected_total_count": total,
        "selected_positive_count": positive,
        "selected_background_count": background,
        "selected_ignore_count": ignore,
        "selected_positive_ratio": positive / total if total else 0.0,
        "selected_background_ratio": background / total if total else 0.0,
        "selected_ignore_ratio": ignore / total if total else 0.0,
        "useful_point_ratio": (positive + background) / total if total else 0.0,
    }


def source_label_counts(source_flags: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    flags = np.asarray(source_flags, dtype=np.uint16)
    labels = np.asarray(labels, dtype=np.int8)
    if flags.shape != labels.shape:
        raise ValueError(f"source/teacher shape mismatch: {flags.shape} vs {labels.shape}")
    result: dict[str, int] = {}
    for name, bit in SOURCE_BITS.items():
        mask = (flags & np.uint16(bit)) != 0
        result[f"source_{name}_selected"] = int(mask.sum())
        result[f"source_{name}_positive"] = int(
            np.count_nonzero(mask & (labels == 1))
        )
        result[f"source_{name}_background"] = int(
            np.count_nonzero(mask & (labels == 0))
        )
        result[f"source_{name}_ignore"] = int(
            np.count_nonzero(mask & (labels == -1))
        )

    evidence_bits = np.uint16(
        int(SOURCE_GLOBAL) | int(SOURCE_LOCAL_PERCENTILE) | int(SOURCE_TOPHAT)
    )
    context_only = (
        ((flags & SOURCE_CONTEXT_GRID) != 0)
        & ((flags & evidence_bits) == 0)
    )
    result["context_only_selected"] = int(context_only.sum())
    result["context_only_positive"] = int(
        np.count_nonzero(context_only & (labels == 1))
    )
    result["context_only_background"] = int(
        np.count_nonzero(context_only & (labels == 0))
    )
    result["context_only_ignore"] = int(
        np.count_nonzero(context_only & (labels == -1))
    )
    return result


def ablate_source(source_flags: np.ndarray, source_name: str) -> np.ndarray:
    flags = np.asarray(source_flags, dtype=np.uint16)
    if source_name == "none":
        return flags.copy()
    if source_name not in SOURCE_BITS:
        raise ValueError(f"Unknown source ablation: {source_name}")
    keep_mask = np.uint16(0xFFFF ^ SOURCE_BITS[source_name])
    return flags & keep_mask


def bbox_selected_counts(
    bbox: TeacherBBox,
    *,
    selected_mask: np.ndarray,
    labels: np.ndarray,
) -> dict[str, int | float | bool]:
    selected = np.asarray(selected_mask, dtype=bool)
    labels = np.asarray(labels, dtype=np.int8)
    if selected.shape != labels.shape:
        raise ValueError("selected mask and teacher labels must have equal shape")

    selected_positive = int(
        np.count_nonzero(selected[bbox.positive_y, bbox.positive_x])
    )
    selected_total = 0
    selected_background = 0
    selected_ignore = 0
    if bbox.roi_bounds is not None:
        left, top, right, bottom = bbox.roi_bounds
        roi_selected = selected[top : bottom + 1, left : right + 1]
        roi_labels = labels[top : bottom + 1, left : right + 1]
        selected_total = int(roi_selected.sum())
        selected_background = int(
            np.count_nonzero(roi_selected & (roi_labels == 0))
        )
        selected_ignore = int(
            np.count_nonzero(roi_selected & (roi_labels == -1))
        )

    teacher_positive = bbox.teacher_positive_count
    recall = (
        selected_positive / teacher_positive
        if teacher_positive > 0
        else float("nan")
    )
    return {
        "teacher_positive_count": teacher_positive,
        "selected_positive_count": selected_positive,
        "selected_background_count": selected_background,
        "selected_ignore_count": selected_ignore,
        "selected_total_count": selected_total,
        "positive_pixel_recall": recall,
        "zero_positive": bool(bbox.teacher_valid and selected_positive == 0),
    }


def positive_extent_metrics(
    bbox: TeacherBBox,
    *,
    selected_mask: np.ndarray,
    min_elongation: float,
) -> tuple[float, float]:
    """Return (teacher elongation, sampled/teacher PC1 extent)."""
    if bbox.teacher_positive_count < 3:
        return float("nan"), float("nan")
    coordinates = np.stack([bbox.positive_x, bbox.positive_y], axis=1).astype(
        np.float64
    )
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, coordinates.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = float(eigenvalues[order[0]])
    minor = float(eigenvalues[order[1]])
    elongation = major / max(minor, 1e-12)
    if not np.isfinite(elongation) or elongation < float(min_elongation):
        return elongation, float("nan")

    axis = eigenvectors[:, order[0]]
    teacher_projection = coordinates @ axis
    teacher_extent = float(teacher_projection.max() - teacher_projection.min())
    selected_values = np.asarray(selected_mask, dtype=bool)[
        bbox.positive_y,
        bbox.positive_x,
    ]
    sampled_coordinates = coordinates[selected_values]
    if teacher_extent <= 0.0:
        return elongation, float("nan")
    if sampled_coordinates.shape[0] < 2:
        return elongation, 0.0
    sampled_projection = sampled_coordinates @ axis
    sampled_extent = float(sampled_projection.max() - sampled_projection.min())
    return elongation, float(np.clip(sampled_extent / teacher_extent, 0.0, 1.0))

