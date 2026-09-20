from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from stage5.utils.visualization_export import (
    DIAGNOSTIC_CATEGORY_COLORS,
    DIAGNOSTIC_CATEGORY_NAMES,
    diagnostic_segmentation_categories,
)

# ----------------------------------------------------------------------------
# Step 1 of the S5-15 prediction frame visualization (handoff section 10.3).
#
# Draws the six-way diagnostic categories (TP / FP / FN / TN /
# ignore_predicted_positive / ignore_other) onto the local-crop frame images
# that `pixel_xy` is expressed in. The categories themselves come from
# `stage5.utils.visualization_export.diagnostic_segmentation_categories`, the
# same function the diagnostic PLY writer uses, so the PNG and the PLY cannot
# disagree about what a point is. numpy + cv2 only; no torch, no CUDA.
# ----------------------------------------------------------------------------

CATEGORY_TRUE_POSITIVE = 0
CATEGORY_FALSE_POSITIVE = 1
CATEGORY_FALSE_NEGATIVE = 2
CATEGORY_TRUE_NEGATIVE = 3
CATEGORY_IGNORE_PREDICTED_POSITIVE = 4
CATEGORY_IGNORE_OTHER = 5

NUM_CATEGORIES = len(DIAGNOSTIC_CATEGORY_NAMES)

# Back-to-front draw order (handoff 10.1(b)). Later entries paint over earlier
# ones where points coincide, so false_positive is drawn last and is never
# hidden by another category -- observing FPs is the point of this export.
DEFAULT_DRAW_ORDER: tuple[int, ...] = (
    CATEGORY_TRUE_NEGATIVE,
    CATEGORY_IGNORE_OTHER,
    CATEGORY_FALSE_NEGATIVE,
    CATEGORY_TRUE_POSITIVE,
    CATEGORY_IGNORE_PREDICTED_POSITIVE,
    CATEGORY_FALSE_POSITIVE,
)

# true_negative and ignore_other are off by default: they fill the frame and
# bury the categories this export exists to show (handoff 5.1).
DEFAULT_DRAW_CATEGORIES: tuple[int, ...] = (
    CATEGORY_TRUE_POSITIVE,
    CATEGORY_FALSE_POSITIVE,
    CATEGORY_FALSE_NEGATIVE,
    CATEGORY_IGNORE_PREDICTED_POSITIVE,
)

__all__ = [
    "CATEGORY_FALSE_NEGATIVE",
    "CATEGORY_FALSE_POSITIVE",
    "CATEGORY_IGNORE_OTHER",
    "CATEGORY_IGNORE_PREDICTED_POSITIVE",
    "CATEGORY_TRUE_NEGATIVE",
    "CATEGORY_TRUE_POSITIVE",
    "DEFAULT_DRAW_CATEGORIES",
    "DEFAULT_DRAW_ORDER",
    "DIAGNOSTIC_CATEGORY_COLORS",
    "DIAGNOSTIC_CATEGORY_NAMES",
    "FramePointIndex",
    "NUM_CATEGORIES",
    "blend_mask",
    "count_categories",
    "draw_bbox",
    "diagnostic_segmentation_categories",
    "frame_point_index_ranges",
    "load_image_to_uint8_gray",
    "render_diagnostic_categories",
    "validate_pixel_xy_bounds",
]


def load_image_to_uint8_gray() -> Callable[[np.ndarray], np.ndarray]:
    """Return Stage2to4's `image_to_uint8_gray`.

    Imported lazily and by function call, never at module import time: the
    Stage2to4 repository root is put on `sys.path` by the CLI (handoff 10.3
    Step 2), in one place, so that importing this module for rendering or for
    the synthetic checks does not require Stage2to4 to be present. Raises with
    the reason rather than falling back to a local reimplementation -- the
    grayscale conversion must be the same one the Stage 4 GT visualization
    used, or the two exports are not comparable.
    """
    try:
        from src.utils.alpha_texture_processing import image_to_uint8_gray
    except ImportError as exc:  # pragma: no cover - depends on sys.path setup
        raise ImportError(
            "Could not import src.utils.alpha_texture_processing.image_to_uint8_gray. "
            "Add the Stage2to4 repository root to sys.path before calling this "
            "(the CLI does so via --stage2to4_root). Original error: "
            f"{exc}"
        ) from exc
    return image_to_uint8_gray


# ----------------------------------------------------------------------------
# Point-to-frame assignment
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePointIndex:
    """Point indices grouped by frame, in stable `frame_order` order.

    `sorted_point_indices` holds every point index sorted by its frame, and
    `offsets` slices it per frame, so `indices(frame)` never mixes frames.
    """

    sorted_point_indices: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.counts.size)

    def indices(self, frame_order: int) -> np.ndarray:
        frame_order = int(frame_order)
        if not 0 <= frame_order < self.num_frames:
            raise IndexError(
                f"frame_order {frame_order} out of range [0, {self.num_frames})"
            )
        start = int(self.offsets[frame_order])
        stop = int(self.offsets[frame_order + 1])
        return self.sorted_point_indices[start:stop]


def frame_point_index_ranges(
    frame_order: np.ndarray,
    num_frames: int,
) -> FramePointIndex:
    """Group point indices by frame.

    Same method as the Stage 4 GT exporter (`np.argsort(..., kind="stable")` +
    `np.bincount` + `cumsum`; handoff 4.5), so a point lands on the same frame
    in both exports.
    """
    frame_order = np.asarray(frame_order)
    if frame_order.ndim != 1:
        raise ValueError(f"frame_order must have shape [N], got {frame_order.shape}")
    if not np.issubdtype(frame_order.dtype, np.integer):
        raise ValueError(f"frame_order must be an integer array, got {frame_order.dtype}")
    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if frame_order.size:
        minimum = int(np.min(frame_order))
        maximum = int(np.max(frame_order))
        if minimum < 0:
            raise ValueError(f"frame_order contains a negative value: {minimum}")
        if maximum >= num_frames:
            raise ValueError(
                f"frame_order max {maximum} exceeds the available frames "
                f"[0, {num_frames}); the intermediate H5 images and the final "
                "H5 points do not belong to the same video"
            )

    frame_order = frame_order.astype(np.int64, copy=False)
    sorted_point_indices = np.argsort(frame_order, kind="stable").astype(np.int64)
    counts = np.bincount(frame_order, minlength=num_frames).astype(np.int64)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    return FramePointIndex(
        sorted_point_indices=sorted_point_indices,
        offsets=offsets,
        counts=counts,
    )


# ----------------------------------------------------------------------------
# Coordinate validation
# ----------------------------------------------------------------------------


def validate_pixel_xy_bounds(
    pixel_xy: np.ndarray,
    *,
    width: int,
    height: int,
    context: str = "",
) -> None:
    """Fail loudly when `pixel_xy` does not fit the local-crop image.

    Same criterion as the S5-14 provenance audit (finite, x in [0, width),
    y in [0, height); `check_stage5_xy_coordinate_provenance.py:215-220`).
    Out-of-range coordinates mean the points and the image are not in the same
    space, so nothing here clips or drops them silently (handoff 10.3 Step 3).
    """
    pixel_xy = np.asarray(pixel_xy)
    prefix = f"{context}: " if context else ""
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise ValueError(f"{prefix}pixel_xy must have shape [N, 2], got {pixel_xy.shape}")
    width = int(width)
    height = int(height)
    if width <= 1 or height <= 1:
        raise ValueError(f"{prefix}degenerate image size: width={width}, height={height}")
    if not pixel_xy.size:
        return
    if not bool(np.all(np.isfinite(pixel_xy))):
        raise ValueError(f"{prefix}pixel_xy contains non-finite value(s)")
    x = pixel_xy[:, 0]
    y = pixel_xy[:, 1]
    if not bool(np.all((x >= 0) & (x < width))):
        raise ValueError(
            f"{prefix}pixel_xy x out of bounds [0, {width}): "
            f"observed [{float(np.min(x)):.6g}, {float(np.max(x)):.6g}]"
        )
    if not bool(np.all((y >= 0) & (y < height))):
        raise ValueError(
            f"{prefix}pixel_xy y out of bounds [0, {height}): "
            f"observed [{float(np.min(y)):.6g}, {float(np.max(y)):.6g}]"
        )


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------


def blend_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: Sequence[int],
    alpha: float,
) -> None:
    """Alpha-blend a flat color into `rgb` where `mask` is set, in place.

    Follows the Stage 4 exporter's `_blend_mask`
    (`export_stage4_point_label_visualization.py:1259`) so that points drawn
    here look like the ones drawn there.
    """
    if not np.any(mask):
        return
    color_array = np.asarray(color, dtype=np.float32)
    blended = (1.0 - alpha) * rgb[mask].astype(np.float32) + alpha * color_array
    rgb[mask] = np.clip(blended, 0, 255).astype(np.uint8)


def render_diagnostic_categories(
    gray: np.ndarray,
    *,
    pixel_xy: np.ndarray,
    categories: np.ndarray,
    radius: int,
    alpha: float,
    draw_categories: Iterable[int] | None = None,
    draw_order: Sequence[int] = DEFAULT_DRAW_ORDER,
    colors: np.ndarray = DIAGNOSTIC_CATEGORY_COLORS,
) -> np.ndarray:
    """Draw diagnostic categories over one grayscale frame.

    Args:
        gray: `[H, W]` uint8 local-crop frame, as returned by
            `image_to_uint8_gray`.
        pixel_xy: `[N, 2]` local-crop coordinates of this frame's points.
        categories: `[N]` values from `diagnostic_segmentation_categories`.
        radius: point radius in pixels; 0 draws single pixels.
        alpha: blend weight in [0, 1]; 1.0 writes the color exactly.
        draw_categories: which categories to draw. Defaults to
            `DEFAULT_DRAW_CATEGORIES` (true_negative and ignore_other off).
        draw_order: back-to-front order; see `DEFAULT_DRAW_ORDER`.
        colors: `[6, 3]` uint8 RGB table, by default the same one the
            diagnostic PLY uses.

    Returns:
        A new `[H, W, 3]` uint8 RGB image. `gray` is not modified.
    """
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError(f"gray image must be 2-D, got {gray.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"gray image must be uint8, got {gray.dtype}")

    categories = np.asarray(categories)
    if categories.ndim != 1:
        raise ValueError(f"categories must have shape [N], got {categories.shape}")
    pixel_xy = np.asarray(pixel_xy, dtype=np.float32)
    if pixel_xy.shape != (categories.size, 2):
        raise ValueError(
            f"pixel_xy/categories shape mismatch: {pixel_xy.shape} vs {categories.shape}"
        )

    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape != (NUM_CATEGORIES, 3):
        raise ValueError(f"colors must have shape [{NUM_CATEGORIES}, 3], got {colors.shape}")
    if categories.size:
        category_min = int(np.min(categories))
        category_max = int(np.max(categories))
        if category_min < 0 or category_max >= NUM_CATEGORIES:
            raise ValueError(
                f"categories out of range [0, {NUM_CATEGORIES}): "
                f"observed [{category_min}, {category_max}]"
            )

    radius = int(radius)
    if radius < 0:
        raise ValueError(f"point radius must be >= 0, got {radius}")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"point alpha must be in [0,1], got {alpha}")

    selected = (
        tuple(DEFAULT_DRAW_CATEGORIES)
        if draw_categories is None
        else tuple(int(value) for value in draw_categories)
    )
    for value in selected:
        if not 0 <= value < NUM_CATEGORIES:
            raise ValueError(
                f"draw_categories contains {value}, outside [0, {NUM_CATEGORIES})"
            )
    order = tuple(int(value) for value in draw_order)
    if sorted(order) != list(range(NUM_CATEGORIES)):
        raise ValueError(
            f"draw_order must be a permutation of 0..{NUM_CATEGORIES - 1}, got {order}"
        )

    height, width = gray.shape
    validate_pixel_xy_bounds(pixel_xy, width=width, height=height)

    rgb = np.stack([gray, gray, gray], axis=-1)
    if not categories.size or not selected:
        return rgb

    # Bounds are already validated, so rounding can only overshoot by one at
    # the outermost half pixel (e.g. x = width - 0.4 rounds to width); clip
    # that back onto the last pixel rather than dropping the point.
    xy = np.rint(pixel_xy).astype(np.int64)
    np.clip(xy[:, 0], 0, width - 1, out=xy[:, 0])
    np.clip(xy[:, 1], 0, height - 1, out=xy[:, 1])

    cv2 = None
    kernel = None
    if radius > 0:
        import cv2  # imported only when a radius is actually requested

        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    for category in order:
        if category not in selected:
            continue
        chosen = categories == category
        if not np.any(chosen):
            continue
        marker = np.zeros((height, width), dtype=np.uint8)
        marker[xy[chosen, 1], xy[chosen, 0]] = 1
        if kernel is not None:
            marker = cv2.dilate(marker, kernel)
        blend_mask(
            rgb,
            marker.astype(bool),
            color=colors[category],
            alpha=alpha,
        )
    return rgb


def count_categories(categories: np.ndarray) -> dict[str, int]:
    """Count points per diagnostic category, keyed by category name."""
    categories = np.asarray(categories)
    if categories.ndim != 1:
        raise ValueError(f"categories must have shape [N], got {categories.shape}")
    counts = np.bincount(categories.astype(np.int64), minlength=NUM_CATEGORIES)
    if counts.size > NUM_CATEGORIES:
        raise ValueError(
            f"categories out of range [0, {NUM_CATEGORIES}): max {counts.size - 1}"
        )
    return {
        name: int(counts[index])
        for index, name in enumerate(DIAGNOSTIC_CATEGORY_NAMES)
    }


def draw_bbox(
    rgb: np.ndarray,
    bbox: Sequence[float],
    *,
    color: Sequence[int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw one saved local-crop bbox onto `rgb`, in place.

    Follows the Stage 4 exporter's `_draw_saved_bbox`
    (`export_stage4_point_label_visualization.py:1341`): coordinates are
    clamped to the image and a degenerate box is skipped rather than raising,
    because the bbox is optional context here, not the subject of the export.
    """
    import cv2

    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width - 1, int(round(x2))))
    bottom = max(0, min(height - 1, int(round(y2))))
    if right <= left or bottom <= top:
        return
    cv2.rectangle(rgb, (left, top), (right, bottom), tuple(int(v) for v in color), int(thickness))
