from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


DenoiseMode = Literal["none", "median", "gaussian", "bilateral"]
ThresholdMode = Literal["fixed", "otsu", "percentile", "adaptive"]
TextureStyle = Literal["original", "threshold_alpha"]


@dataclass
class AlphaTextureConfig:
    """
    Configuration for generic, inference-time usable alpha texture generation.

    This uses only:
      - the input image itself
      - frame-local intensity statistics
      - generic morphology / component filtering

    It does not use:
      - class labels
      - BBox annotations
      - manually specified target anatomy
    """

    texture_style: TextureStyle = "original"

    # Denoising
    denoise: DenoiseMode = "none"
    denoise_ksize: int = 3

    # Thresholding
    threshold_mode: ThresholdMode = "fixed"
    fixed_threshold: int = 180
    percentile: float = 90.0

    # Adaptive threshold
    adaptive_block_size: int = 31
    adaptive_c: float = 5.0

    # Morphology
    open_ksize: int = 0
    close_ksize: int = 0
    morph_shape: Literal["rect", "ellipse", "cross"] = "ellipse"

    # Connected component filtering
    min_component_area: int = 0

    # Foreground rendering
    white_foreground: bool = True


def image_to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 grayscale.

    Accepts:
        [H, W] uint8
        [H, W] float
        [1, H, W] float

    Returns:
        [H, W] uint8
    """
    img = np.asarray(image)

    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[-1] == 1:
            img = img[..., 0]
        else:
            img = img[0]

    if img.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape={img.shape}")

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    img = np.nan_to_num(img)

    min_v = float(img.min())
    max_v = float(img.max())

    # local_encoder_images are usually float32 in [0, 1].
    if min_v >= -0.5 and max_v <= 1.5:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        denom = max(max_v - min_v, 1e-6)
        img = (img - min_v) / denom * 255.0

    return img.astype(np.uint8)


def _ensure_odd_ksize(ksize: int, name: str) -> int:
    if ksize <= 1:
        return 0
    if ksize % 2 == 0:
        raise ValueError(f"{name} must be odd when > 1, got {ksize}")
    return int(ksize)


def denoise_gray(gray: np.ndarray, mode: DenoiseMode, ksize: int) -> np.ndarray:
    """
    Generic denoising before thresholding.
    """
    mode = mode.lower()

    if mode == "none":
        return gray

    ksize = _ensure_odd_ksize(ksize, "denoise_ksize")
    if ksize == 0:
        return gray

    if mode == "median":
        return cv2.medianBlur(gray, ksize)

    if mode == "gaussian":
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)

    if mode == "bilateral":
        # ksize is used as diameter.
        # sigma values are intentionally moderate for ultrasound texture.
        return cv2.bilateralFilter(gray, d=ksize, sigmaColor=50, sigmaSpace=50)

    raise ValueError(f"Unknown denoise mode: {mode}")


def threshold_gray(
    gray: np.ndarray,
    *,
    mode: ThresholdMode,
    fixed_threshold: int,
    percentile: float,
    adaptive_block_size: int,
    adaptive_c: float,
) -> np.ndarray:
    """
    Create binary foreground mask from denoised uint8 grayscale image.

    Returns:
        bool mask [H, W]
    """
    mode = mode.lower()

    if mode == "fixed":
        return gray >= int(fixed_threshold)

    if mode == "otsu":
        _, th = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return th > 0

    if mode == "percentile":
        p = float(np.clip(percentile, 0.0, 100.0))
        thr = np.percentile(gray, p)
        return gray >= thr

    if mode == "adaptive":
        block = _ensure_odd_ksize(adaptive_block_size, "adaptive_block_size")
        if block <= 1:
            raise ValueError("adaptive_block_size must be odd and > 1")

        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            adaptive_c,
        )
        return th > 0

    raise ValueError(f"Unknown threshold mode: {mode}")


def _morph_kernel(ksize: int, shape: str) -> np.ndarray | None:
    if ksize <= 1:
        return None

    if shape == "rect":
        morph_shape = cv2.MORPH_RECT
    elif shape == "ellipse":
        morph_shape = cv2.MORPH_ELLIPSE
    elif shape == "cross":
        morph_shape = cv2.MORPH_CROSS
    else:
        raise ValueError(f"Unknown morph_shape: {shape}")

    return cv2.getStructuringElement(morph_shape, (ksize, ksize))


def apply_morphology(
    mask: np.ndarray,
    *,
    open_ksize: int,
    close_ksize: int,
    morph_shape: str,
) -> np.ndarray:
    """
    Apply opening and closing to a boolean mask.

    Opening:
      removes small isolated speckles.

    Closing:
      fills small gaps and connects nearby fragmented contours.
    """
    x = mask.astype(np.uint8) * 255

    kernel_open = _morph_kernel(open_ksize, morph_shape)
    if kernel_open is not None:
        x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel_open)

    kernel_close = _morph_kernel(close_ksize, morph_shape)
    if kernel_close is not None:
        x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel_close)

    return x > 0


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components smaller than min_area.

    This is class-independent and uses only image-derived connected components.
    """
    if min_area <= 0:
        return mask

    x = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        x,
        connectivity=8,
    )

    keep = np.zeros_like(mask, dtype=bool)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == label] = True

    return keep


def mask_to_rgba_texture(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    white_foreground: bool,
) -> np.ndarray:
    """
    Convert mask to RGBA texture.

    If white_foreground=True:
      visible pixels become pure white.

    Else:
      visible pixels preserve grayscale intensity.
    """
    if white_foreground:
        rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
        rgb[mask] = 255
    else:
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[~mask] = 0

    alpha = np.zeros(gray.shape, dtype=np.uint8)
    alpha[mask] = 255

    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def make_texture_image(
    image: np.ndarray,
    config: AlphaTextureConfig,
) -> np.ndarray:
    """
    Create texture image for OBJ export.

    Returns:
        original:
            [H, W, 3] uint8

        threshold_alpha:
            [H, W, 4] uint8
    """
    gray = image_to_uint8_gray(image)

    if config.texture_style == "original":
        return np.stack([gray, gray, gray], axis=-1)

    if config.texture_style == "threshold_alpha":
        denoised = denoise_gray(
            gray,
            mode=config.denoise,
            ksize=config.denoise_ksize,
        )

        mask = threshold_gray(
            denoised,
            mode=config.threshold_mode,
            fixed_threshold=config.fixed_threshold,
            percentile=config.percentile,
            adaptive_block_size=config.adaptive_block_size,
            adaptive_c=config.adaptive_c,
        )

        mask = apply_morphology(
            mask,
            open_ksize=config.open_ksize,
            close_ksize=config.close_ksize,
            morph_shape=config.morph_shape,
        )

        mask = remove_small_components(
            mask,
            min_area=config.min_component_area,
        )

        return mask_to_rgba_texture(
            gray,
            mask,
            white_foreground=config.white_foreground,
        )

    raise ValueError(f"Unknown texture_style: {config.texture_style}")