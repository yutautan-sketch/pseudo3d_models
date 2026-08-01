from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from scipy.ndimage import median_filter, percentile_filter

from src.utils.alpha_texture_processing import (
    apply_morphology,
    image_to_uint8_gray,
    remove_small_components,
)


SamplingMode = Literal["legacy", "combined_v2"]
ContextGridPhase = Literal["origin", "centered", "dual"]
MorphShape = Literal["rect", "ellipse", "cross"]
OriginalForegroundMode = Literal["all", "nonzero", "intensity"]

SOURCE_GLOBAL = np.uint16(1 << 0)
SOURCE_LOCAL_PERCENTILE = np.uint16(1 << 1)
SOURCE_TOPHAT = np.uint16(1 << 2)
SOURCE_CONTEXT_GRID = np.uint16(1 << 3)

SOURCE_FLAG_NAMES = {
    int(SOURCE_GLOBAL): "global",
    int(SOURCE_LOCAL_PERCENTILE): "local_percentile",
    int(SOURCE_TOPHAT): "tophat",
    int(SOURCE_CONTEXT_GRID): "context_grid",
}


@dataclass(frozen=True)
class PointSamplingConfig:
    """
    Configuration for BBox-independent Stage 4 point sampling.

    The default is legacy-compatible: only the supplied global evidence mask is
    used. Use combined_v2_defaults() for the provisional Stage 5 rescue setup.
    """

    sampling_mode: SamplingMode = "legacy"

    include_context_grid: bool = False
    context_grid_stride: int = 4
    context_grid_phase: ContextGridPhase = "origin"

    enable_local_percentile: bool = False
    local_window_size: int = 41
    local_percentile: float = 80.0
    local_min_contrast: float = 4.0

    enable_tophat: bool = False
    tophat_kernel_size: int = 21
    tophat_percentile: float = 85.0
    tophat_min_response: float = 2.0
    tophat_morph_shape: MorphShape = "ellipse"

    enable_evidence_cleanup: bool = False
    evidence_open_ksize: int = 3
    evidence_close_ksize: int = 5
    evidence_morph_shape: MorphShape = "ellipse"
    evidence_min_component_area: int = 100

    local_source_confidence: float = 0.7
    tophat_source_confidence: float = 0.7
    context_source_confidence: float = 0.0

    @classmethod
    def combined_v2_defaults(cls) -> "PointSamplingConfig":
        """
        Provisional combined_v2 configuration from stage4_edit_prompt.md.
        """
        return cls(
            sampling_mode="combined_v2",
            include_context_grid=True,
            context_grid_stride=4,
            context_grid_phase="origin",
            enable_local_percentile=True,
            local_window_size=41,
            local_percentile=80.0,
            local_min_contrast=4.0,
            enable_tophat=True,
            tophat_kernel_size=21,
            tophat_percentile=85.0,
            tophat_min_response=2.0,
            tophat_morph_shape="ellipse",
            enable_evidence_cleanup=True,
            evidence_open_ksize=3,
            evidence_close_ksize=5,
            evidence_morph_shape="ellipse",
            evidence_min_component_area=100,
        )


def _validate_2d_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"Expected 2D image shape, got {shape}")
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Image shape must be positive, got {shape}")
    return h, w


def _ensure_positive_odd(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd, got {value}")
    return value


def _morph_kernel(ksize: int, shape: MorphShape) -> np.ndarray:
    ksize = _ensure_positive_odd(ksize, "morphology kernel size")

    if shape == "rect":
        morph_shape = cv2.MORPH_RECT
    elif shape == "ellipse":
        morph_shape = cv2.MORPH_ELLIPSE
    elif shape == "cross":
        morph_shape = cv2.MORPH_CROSS
    else:
        raise ValueError(f"Unknown morph_shape: {shape}")

    return cv2.getStructuringElement(morph_shape, (ksize, ksize))


def build_global_evidence_mask(
    image: np.ndarray,
    texture: np.ndarray | None = None,
    *,
    min_alpha: int = 1,
    min_intensity: int = 1,
    original_foreground_mode: OriginalForegroundMode = "nonzero",
    sample_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the legacy global/texture-alpha evidence mask.

    Returns:
        mask:
            bool [H, W] selected pixels.
        alpha:
            uint8 [H, W] alpha/confidence source map.
    """
    gray = image_to_uint8_gray(image)

    if texture is not None and texture.ndim == 3 and texture.shape[-1] == 4:
        if tuple(texture.shape[:2]) != tuple(gray.shape):
            raise ValueError(
                "Texture and image shapes do not match: "
                f"texture={texture.shape}, image={gray.shape}"
            )
        alpha = texture[..., 3].astype(np.uint8)
        mask = alpha >= int(min_alpha)
    else:
        alpha = np.full(gray.shape, 255, dtype=np.uint8)
        if original_foreground_mode == "all":
            mask = np.ones(gray.shape, dtype=bool)
        elif original_foreground_mode == "nonzero":
            mask = gray > 0
        elif original_foreground_mode == "intensity":
            mask = gray >= int(min_intensity)
        else:
            raise ValueError(
                "original_foreground_mode must be one of: all, nonzero, intensity"
            )

    if sample_stride < 1:
        raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")

    if sample_stride > 1:
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::sample_stride, ::sample_stride] = True
        mask &= stride_mask

    return mask.astype(bool), alpha


def build_context_grid_mask(
    shape: tuple[int, int],
    *,
    stride: int = 4,
    phase: ContextGridPhase = "origin",
) -> np.ndarray:
    """
    Build a regular whole-image context grid mask.
    """
    h, w = _validate_2d_shape(shape)
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"context_grid_stride must be >= 1, got {stride}")

    if phase == "origin":
        offsets = [(0, 0)]
    elif phase == "centered":
        offsets = [(stride // 2, stride // 2)]
    elif phase == "dual":
        offsets = [(0, 0), (stride // 2, stride // 2)]
    else:
        raise ValueError(f"Unknown context_grid_phase: {phase}")

    mask = np.zeros((h, w), dtype=bool)
    for off_y, off_x in offsets:
        mask[off_y:h:stride, off_x:w:stride] = True
    return mask


def build_local_percentile_evidence(
    image: np.ndarray,
    *,
    window_size: int = 41,
    percentile: float = 80.0,
    min_contrast: float = 4.0,
    boundary_mode: str = "reflect",
    return_debug_maps: bool = False,
) -> dict[str, np.ndarray]:
    """
    Select pixels that are bright relative to their local neighborhood.
    """
    gray = image_to_uint8_gray(image)
    window_size = _ensure_positive_odd(window_size, "local_window_size")
    percentile = float(np.clip(percentile, 0.0, 100.0))

    threshold_map = percentile_filter(
        gray,
        percentile=percentile,
        size=window_size,
        mode=boundary_mode,
    ).astype(np.float32)
    median_map = median_filter(
        gray,
        size=window_size,
        mode=boundary_mode,
    ).astype(np.float32)

    gray_f = gray.astype(np.float32)
    contrast_map = gray_f - median_map
    mask = (gray_f >= threshold_map) & (contrast_map >= float(min_contrast))

    result: dict[str, np.ndarray] = {"mask": mask.astype(bool)}
    if return_debug_maps:
        result.update(
            {
                "threshold_map": threshold_map,
                "median_map": median_map,
                "contrast_map": contrast_map,
            }
        )
    return result


def build_tophat_evidence(
    image: np.ndarray,
    *,
    kernel_size: int = 21,
    percentile: float = 85.0,
    min_response: float = 2.0,
    morph_shape: MorphShape = "ellipse",
    return_debug_maps: bool = False,
) -> dict[str, np.ndarray]:
    """
    Select pixels with positive white top-hat response.
    """
    gray = image_to_uint8_gray(image)
    kernel = _morph_kernel(kernel_size, morph_shape)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    response = gray.astype(np.float32) - opened.astype(np.float32)
    positive = response[response > 0.0]

    if positive.size == 0:
        mask = np.zeros(gray.shape, dtype=bool)
        threshold = np.float32(np.inf)
    else:
        p = float(np.clip(percentile, 0.0, 100.0))
        threshold = np.float32(np.percentile(positive, p))
        mask = (response >= float(threshold)) & (response >= float(min_response))

    result: dict[str, np.ndarray] = {"mask": mask.astype(bool)}
    if return_debug_maps:
        result.update(
            {
                "response_map": response.astype(np.float32),
                "opened_map": opened.astype(np.uint8),
                "threshold": np.asarray(threshold, dtype=np.float32),
            }
        )
    return result


def cleanup_evidence_mask(
    mask: np.ndarray,
    *,
    open_ksize: int = 3,
    close_ksize: int = 5,
    morph_shape: MorphShape = "ellipse",
    min_component_area: int = 100,
    return_debug_maps: bool = False,
) -> dict[str, np.ndarray]:
    """
    Apply the global-foreground cleanup sequence to an evidence mask.

    Local-percentile and top-hat masks call this independently so a noisy
    source cannot be retained merely because it overlaps another source.
    """
    raw_mask = np.asarray(mask, dtype=bool)
    _validate_2d_shape(raw_mask.shape)

    open_ksize = int(open_ksize)
    close_ksize = int(close_ksize)
    min_component_area = int(min_component_area)
    if open_ksize < 0:
        raise ValueError(
            f"evidence_open_ksize must be >= 0, got {open_ksize}"
        )
    if close_ksize < 0:
        raise ValueError(
            f"evidence_close_ksize must be >= 0, got {close_ksize}"
        )
    if morph_shape not in {"rect", "ellipse", "cross"}:
        raise ValueError(f"Unknown evidence_morph_shape: {morph_shape}")
    if min_component_area < 0:
        raise ValueError(
            "evidence_min_component_area must be >= 0, "
            f"got {min_component_area}"
        )

    morphology_mask = apply_morphology(
        raw_mask,
        open_ksize=open_ksize,
        close_ksize=close_ksize,
        morph_shape=morph_shape,
    ).astype(bool)
    cleaned_mask = remove_small_components(
        morphology_mask,
        min_area=min_component_area,
    ).astype(bool)

    result: dict[str, np.ndarray] = {"mask": cleaned_mask}
    if return_debug_maps:
        result.update(
            {
                "raw_mask": raw_mask.copy(),
                "morphology_mask": morphology_mask,
            }
        )
    return result


def build_sampling_source_flags(
    image: np.ndarray,
    global_mask: np.ndarray | None,
    config: PointSamplingConfig,
    *,
    return_debug_maps: bool = False,
) -> dict[str, Any]:
    """
    Build source flags and selected mask for one frame.
    """
    gray = image_to_uint8_gray(image)
    h, w = _validate_2d_shape(gray.shape)

    if config.sampling_mode not in {"legacy", "combined_v2"}:
        raise ValueError(f"Unknown sampling_mode: {config.sampling_mode}")

    if global_mask is None:
        global_mask_bool = np.zeros((h, w), dtype=bool)
    else:
        global_mask_bool = np.asarray(global_mask, dtype=bool)
        if global_mask_bool.shape != (h, w):
            raise ValueError(
                "global_mask shape does not match image: "
                f"global_mask={global_mask_bool.shape}, image={(h, w)}"
            )

    source_flags = np.zeros((h, w), dtype=np.uint16)
    source_masks: dict[str, np.ndarray] = {
        "global_mask": global_mask_bool,
        "local_percentile_mask": np.zeros((h, w), dtype=bool),
        "tophat_mask": np.zeros((h, w), dtype=bool),
        "context_grid_mask": np.zeros((h, w), dtype=bool),
    }
    debug_maps: dict[str, np.ndarray] = {}

    source_flags[global_mask_bool] |= SOURCE_GLOBAL

    if config.sampling_mode == "combined_v2":
        if config.include_context_grid:
            context_mask = build_context_grid_mask(
                (h, w),
                stride=config.context_grid_stride,
                phase=config.context_grid_phase,
            )
            source_masks["context_grid_mask"] = context_mask
            source_flags[context_mask] |= SOURCE_CONTEXT_GRID

        if config.enable_local_percentile:
            local_result = build_local_percentile_evidence(
                gray,
                window_size=config.local_window_size,
                percentile=config.local_percentile,
                min_contrast=config.local_min_contrast,
                return_debug_maps=return_debug_maps,
            )
            local_mask = local_result["mask"].astype(bool)
            if config.enable_evidence_cleanup:
                local_cleanup = cleanup_evidence_mask(
                    local_mask,
                    open_ksize=config.evidence_open_ksize,
                    close_ksize=config.evidence_close_ksize,
                    morph_shape=config.evidence_morph_shape,
                    min_component_area=config.evidence_min_component_area,
                    return_debug_maps=return_debug_maps,
                )
                local_mask = local_cleanup["mask"].astype(bool)
                if return_debug_maps:
                    debug_maps["local_raw_evidence_mask"] = local_cleanup[
                        "raw_mask"
                    ]
                    debug_maps["local_morphology_mask"] = local_cleanup[
                        "morphology_mask"
                    ]
            source_masks["local_percentile_mask"] = local_mask
            source_flags[local_mask] |= SOURCE_LOCAL_PERCENTILE
            if return_debug_maps:
                for key, value in local_result.items():
                    if key != "mask":
                        debug_maps[f"local_{key}"] = value

        if config.enable_tophat:
            tophat_result = build_tophat_evidence(
                gray,
                kernel_size=config.tophat_kernel_size,
                percentile=config.tophat_percentile,
                min_response=config.tophat_min_response,
                morph_shape=config.tophat_morph_shape,
                return_debug_maps=return_debug_maps,
            )
            tophat_mask = tophat_result["mask"].astype(bool)
            if config.enable_evidence_cleanup:
                tophat_cleanup = cleanup_evidence_mask(
                    tophat_mask,
                    open_ksize=config.evidence_open_ksize,
                    close_ksize=config.evidence_close_ksize,
                    morph_shape=config.evidence_morph_shape,
                    min_component_area=config.evidence_min_component_area,
                    return_debug_maps=return_debug_maps,
                )
                tophat_mask = tophat_cleanup["mask"].astype(bool)
                if return_debug_maps:
                    debug_maps["tophat_raw_evidence_mask"] = tophat_cleanup[
                        "raw_mask"
                    ]
                    debug_maps["tophat_morphology_mask"] = tophat_cleanup[
                        "morphology_mask"
                    ]
            source_masks["tophat_mask"] = tophat_mask
            source_flags[tophat_mask] |= SOURCE_TOPHAT
            if return_debug_maps:
                for key, value in tophat_result.items():
                    if key != "mask":
                        debug_maps[f"tophat_{key}"] = value

    selected_mask = source_flags != 0
    result: dict[str, Any] = {
        "selected_mask": selected_mask,
        "source_flags": source_flags,
        **source_masks,
        "stats": summarize_source_flags(source_flags),
    }
    if return_debug_maps:
        result["debug_maps"] = debug_maps
    return result


def source_type_from_flags(source_flags: np.ndarray) -> np.ndarray:
    """
    Convert bit flags to a single representative source id.

    Priority:
        1 = global, 2 = local percentile, 3 = top-hat, 4 = context grid.
        0 means unselected.
    """
    flags = np.asarray(source_flags, dtype=np.uint16)
    source_type = np.zeros(flags.shape, dtype=np.uint8)
    source_type[(flags & SOURCE_CONTEXT_GRID) != 0] = 4
    source_type[(flags & SOURCE_TOPHAT) != 0] = 3
    source_type[(flags & SOURCE_LOCAL_PERCENTILE) != 0] = 2
    source_type[(flags & SOURCE_GLOBAL) != 0] = 1
    return source_type


def build_sampling_confidence(
    source_flags: np.ndarray,
    *,
    global_confidence: np.ndarray | None = None,
    local_confidence: float = 0.7,
    tophat_confidence: float = 0.7,
    context_confidence: float = 0.0,
) -> np.ndarray:
    """
    Build a provisional sampling confidence map from source flags.
    """
    flags = np.asarray(source_flags, dtype=np.uint16)
    confidence = np.zeros(flags.shape, dtype=np.float32)

    if global_confidence is not None:
        global_confidence_arr = np.asarray(global_confidence, dtype=np.float32)
        if global_confidence_arr.shape != flags.shape:
            raise ValueError(
                "global_confidence shape does not match source_flags: "
                f"global_confidence={global_confidence_arr.shape}, "
                f"source_flags={flags.shape}"
            )
        confidence = np.maximum(
            confidence,
            np.where((flags & SOURCE_GLOBAL) != 0, global_confidence_arr, 0.0),
        )

    confidence = np.maximum(
        confidence,
        np.where((flags & SOURCE_LOCAL_PERCENTILE) != 0, local_confidence, 0.0),
    )
    confidence = np.maximum(
        confidence,
        np.where((flags & SOURCE_TOPHAT) != 0, tophat_confidence, 0.0),
    )
    confidence = np.maximum(
        confidence,
        np.where((flags & SOURCE_CONTEXT_GRID) != 0, context_confidence, 0.0),
    )
    return confidence.astype(np.float32)


def selected_pixel_arrays(
    source_flags: np.ndarray,
    *,
    sampling_confidence: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Convert a source flag map into per-point pixel arrays.
    """
    flags = np.asarray(source_flags, dtype=np.uint16)
    selected_mask = flags != 0
    ys, xs = np.nonzero(selected_mask)

    result = {
        "xs": xs.astype(np.float32),
        "ys": ys.astype(np.float32),
        "source_flags": flags[ys, xs].astype(np.uint16),
        "source_type": source_type_from_flags(flags)[ys, xs].astype(np.uint8),
    }
    if sampling_confidence is not None:
        conf = np.asarray(sampling_confidence, dtype=np.float32)
        if conf.shape != flags.shape:
            raise ValueError(
                "sampling_confidence shape does not match source_flags: "
                f"sampling_confidence={conf.shape}, source_flags={flags.shape}"
            )
        result["sampling_confidence"] = conf[ys, xs].astype(np.float32)
    return result


def summarize_source_flags(source_flags: np.ndarray) -> dict[str, int | float]:
    """
    Summarize selected pixels for one frame or an already-flattened collection.
    """
    flags = np.asarray(source_flags, dtype=np.uint16)
    selected = flags != 0
    evidence = (flags & (SOURCE_GLOBAL | SOURCE_LOCAL_PERCENTILE | SOURCE_TOPHAT)) != 0
    source_count = (
        ((flags & SOURCE_GLOBAL) != 0).astype(np.uint8)
        + ((flags & SOURCE_LOCAL_PERCENTILE) != 0).astype(np.uint8)
        + ((flags & SOURCE_TOPHAT) != 0).astype(np.uint8)
        + ((flags & SOURCE_CONTEXT_GRID) != 0).astype(np.uint8)
    )

    total = int(np.count_nonzero(selected))
    global_points = int(np.count_nonzero((flags & SOURCE_GLOBAL) != 0))

    return {
        "total_selected_points": total,
        "global_evidence_points": global_points,
        "local_percentile_points": int(
            np.count_nonzero((flags & SOURCE_LOCAL_PERCENTILE) != 0)
        ),
        "tophat_points": int(np.count_nonzero((flags & SOURCE_TOPHAT) != 0)),
        "context_grid_points": int(
            np.count_nonzero((flags & SOURCE_CONTEXT_GRID) != 0)
        ),
        "context_only_points": int(
            np.count_nonzero(((flags & SOURCE_CONTEXT_GRID) != 0) & ~evidence)
        ),
        "evidence_points": int(np.count_nonzero(evidence)),
        "overlap_points": int(np.count_nonzero(source_count > 1)),
        "legacy_point_count_multiplier": (
            float(total) / float(global_points) if global_points > 0 else 0.0
        ),
    }


def summarize_per_frame_counts(counts: np.ndarray) -> dict[str, int | float]:
    """
    Summarize per-frame selected point counts.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        return {
            "points_per_frame_min": 0,
            "points_per_frame_mean": 0.0,
            "points_per_frame_median": 0.0,
            "points_per_frame_max": 0,
        }
    return {
        "points_per_frame_min": int(counts.min()),
        "points_per_frame_mean": float(counts.mean()),
        "points_per_frame_median": float(np.median(counts)),
        "points_per_frame_max": int(counts.max()),
    }
