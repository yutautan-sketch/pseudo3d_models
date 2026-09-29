from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from stage5.geometry.types import (
    CoordSpace,
    GeometryContractError,
    InputReason,
    InputUnevaluable,
    Points2D,
    require_space,
)

# ----------------------------------------------------------------------------
# S5-17 S17-1: local crop <-> original frame, and original frame <-> mm.
#
# The forward map is the one Stage 2 wrote the local encoder images with, and
# the one Stage 4 used to put the CVAT BBoxes into local coordinates
# (annotate_pseudo3d_point_cloud.py, xml_bbox_to_local):
#
#   crop modes   u = x * s - left,        v = y * s - top        (isotropic s)
#   resize mode  u = x * local_w / raw_w, v = y * local_h / raw_h (anisotropic)
#
# so the inverse used here is x = (u + left) / s, y = (v + top) / s, or the
# per-axis ratio for `resize`. The mode is taken from
# `local_preprocess_effective`, never from the requested `local_preprocess`:
# Stage 4 branches on the requested mode and replaces a NaN scale by 1.0, which
# is wrong for `offset_crop_fallback_resize` (report 1.3 F4). That mode is
# therefore input-unevaluable here rather than guessed at (management 5.4).
#
# Sub-pixel resampling offsets of cv2.INTER_AREA are not modelled; the scale
# recorded for `resize_shorter_then_offset_crop` is the requested one, and the
# rounded resize can differ per axis by up to 0.5 / side. Both are recorded
# limits, not calibrated here.
#
# PixelToMM is constructed only from a confirmed external source. There is no
# default: the intermediate H5 `spacing` (1.0 / 1.0) is a visualisation
# placeholder and is never accepted. A frame without a value stops the run.
# ----------------------------------------------------------------------------

CROP_MODES = ("offset_crop", "resize_shorter_then_offset_crop")
RESIZE_MODE = "resize"
FALLBACK_RESIZE_MODE = "offset_crop_fallback_resize"

# Original-frame coordinates after inversion may fall slightly outside the
# frame because of the rounded resize; one pixel of slack, anything beyond
# that is a contract violation (report 1.7.3).
RAW_RANGE_SLACK_PX = 1.0


def _attr(attrs: Mapping[str, Any], key: str) -> Any:
    if key not in attrs:
        raise InputUnevaluable(InputReason.CROP_FIELD_MISSING, key)
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _finite_float(attrs: Mapping[str, Any], key: str, *, reason: str = InputReason.CROP_SCALE_INVALID) -> float:
    value = _attr(attrs, key)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InputUnevaluable(InputReason.CROP_FIELD_MISSING, f"{key} is not numeric") from error
    if not math.isfinite(number):
        raise InputUnevaluable(reason, f"{key} is not finite")
    return number


@dataclass(frozen=True)
class CropTransform:
    mode: str
    raw_width: int
    raw_height: int
    local_width: int
    local_height: int
    crop_left: float
    crop_top: float
    scale_x: float
    scale_y: float

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any], *, local_shape_hw: tuple[int, int]) -> "CropTransform":
        """Build from intermediate-H5 attrs. Raises InputUnevaluable, never guesses."""
        mode = str(_attr(attrs, "local_preprocess_effective"))
        local_height, local_width = (int(local_shape_hw[0]), int(local_shape_hw[1]))
        raw_width = _finite_float(attrs, "raw_width", reason=InputReason.CROP_DIMENSION_INVALID)
        raw_height = _finite_float(attrs, "raw_height", reason=InputReason.CROP_DIMENSION_INVALID)
        for name, value in (("raw_width", raw_width), ("raw_height", raw_height)):
            if value <= 1 or value != int(value):
                raise InputUnevaluable(InputReason.CROP_DIMENSION_INVALID, f"{name}={value}")
        if local_width <= 1 or local_height <= 1:
            raise InputUnevaluable(InputReason.CROP_DIMENSION_INVALID, f"local shape {local_shape_hw}")

        if mode == FALLBACK_RESIZE_MODE:
            raise InputUnevaluable(
                InputReason.CROP_FALLBACK_RESIZE,
                "frame smaller than the crop; Stage 4 BBox mapping may disagree with the inverse",
            )
        if mode == RESIZE_MODE:
            return cls(
                mode=mode,
                raw_width=int(raw_width),
                raw_height=int(raw_height),
                local_width=local_width,
                local_height=local_height,
                crop_left=0.0,
                crop_top=0.0,
                scale_x=local_width / raw_width,
                scale_y=local_height / raw_height,
            )
        if mode not in CROP_MODES:
            raise InputUnevaluable(InputReason.CROP_UNKNOWN_MODE, mode)

        scale = _finite_float(attrs, "local_resize_scale")
        if scale <= 0:
            raise InputUnevaluable(InputReason.CROP_SCALE_INVALID, f"local_resize_scale={scale}")
        left = _finite_float(attrs, "local_crop_left", reason=InputReason.CROP_DIMENSION_INVALID)
        top = _finite_float(attrs, "local_crop_top", reason=InputReason.CROP_DIMENSION_INVALID)
        if left < 0 or top < 0:
            raise InputUnevaluable(InputReason.CROP_DIMENSION_INVALID, "negative crop offset")
        # The crop must lie inside the (possibly resized) frame it was cut from.
        if (left + local_width) > raw_width * scale + 1.0 or (top + local_height) > raw_height * scale + 1.0:
            raise InputUnevaluable(InputReason.CROP_DIMENSION_INVALID, "crop exceeds the resized frame")
        return cls(
            mode=mode,
            raw_width=int(raw_width),
            raw_height=int(raw_height),
            local_width=local_width,
            local_height=local_height,
            crop_left=left,
            crop_top=top,
            scale_x=scale,
            scale_y=scale,
        )

    # -- range checks: violations stop the run -------------------------------

    def validate_local(self, points: Points2D) -> None:
        require_space(points.space, CoordSpace.LOCAL_CROP_PX, what="validate_local input")
        xy = points.xy
        if not np.all(np.isfinite(xy)):
            raise GeometryContractError("pixel_xy contains non-finite values")
        if xy.size and not (
            np.all((xy[:, 0] >= 0) & (xy[:, 0] < self.local_width))
            and np.all((xy[:, 1] >= 0) & (xy[:, 1] < self.local_height))
        ):
            raise GeometryContractError(
                f"pixel_xy outside the local crop [0, {self.local_width}) x [0, {self.local_height})"
            )

    def validate_raw(self, points: Points2D) -> None:
        require_space(points.space, CoordSpace.RAW_FRAME_PX, what="validate_raw input")
        xy = points.xy
        slack = RAW_RANGE_SLACK_PX
        if xy.size and not (
            np.all((xy[:, 0] >= -slack) & (xy[:, 0] <= self.raw_width + slack))
            and np.all((xy[:, 1] >= -slack) & (xy[:, 1] <= self.raw_height + slack))
        ):
            raise GeometryContractError("inverted coordinates fall outside the original frame")

    # -- maps -----------------------------------------------------------------

    def local_to_raw(self, points: Points2D, *, check_range: bool = True) -> Points2D:
        """`check_range=False` is for synthetic perturbations (jitter) only."""
        require_space(points.space, CoordSpace.LOCAL_CROP_PX, what="local_to_raw input")
        if check_range:
            self.validate_local(points)
        xy = points.xy
        raw = np.empty_like(xy)
        raw[:, 0] = (xy[:, 0] + self.crop_left) / self.scale_x
        raw[:, 1] = (xy[:, 1] + self.crop_top) / self.scale_y
        result = Points2D(raw, CoordSpace.RAW_FRAME_PX)
        if check_range:
            self.validate_raw(result)
        return result

    def raw_to_local(self, points: Points2D) -> Points2D:
        require_space(points.space, CoordSpace.RAW_FRAME_PX, what="raw_to_local input")
        xy = points.xy
        local = np.empty_like(xy)
        local[:, 0] = xy[:, 0] * self.scale_x - self.crop_left
        local[:, 1] = xy[:, 1] * self.scale_y - self.crop_top
        return Points2D(local, CoordSpace.LOCAL_CROP_PX)

    def raw_to_norm(self, points: Points2D) -> Points2D:
        require_space(points.space, CoordSpace.RAW_FRAME_PX, what="raw_to_norm input")
        return Points2D(points.xy / np.array([self.raw_width, self.raw_height], dtype=np.float64), CoordSpace.RAW_FRAME_NORM)

    def norm_to_raw(self, points: Points2D) -> Points2D:
        require_space(points.space, CoordSpace.RAW_FRAME_NORM, what="norm_to_raw input")
        return Points2D(points.xy * np.array([self.raw_width, self.raw_height], dtype=np.float64), CoordSpace.RAW_FRAME_PX)


@dataclass(frozen=True)
class PixelToMM:
    """Original-frame pixels <-> mm, x and y separate.

    Either one video-level pair, or a per-frame table. A per-frame table has no
    fallback to a video value and no fallback to a neighbouring frame.
    """

    source_sha256: str
    mm_per_px_x: float | None = None
    mm_per_px_y: float | None = None
    per_frame: Mapping[int, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        video_level = self.mm_per_px_x is not None or self.mm_per_px_y is not None
        if video_level == (self.per_frame is not None):
            raise GeometryContractError("PixelToMM needs exactly one of a video-level pair or a per-frame table")
        if not self.source_sha256:
            raise GeometryContractError("PixelToMM requires the source's SHA-256; an unsourced scale is not accepted")
        pairs = [(self.mm_per_px_x, self.mm_per_px_y)] if video_level else list((self.per_frame or {}).values())
        for sx, sy in pairs:
            for value in (sx, sy):
                if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                    raise GeometryContractError(f"mm/pixel must be finite and positive, got {value!r}")

    @property
    def granularity(self) -> str:
        return "frame" if self.per_frame is not None else "video"

    def scale_for(self, frame_order: int | None) -> tuple[float, float]:
        if self.per_frame is None:
            return float(self.mm_per_px_x), float(self.mm_per_px_y)  # type: ignore[arg-type]
        if frame_order is None or int(frame_order) not in self.per_frame:
            raise GeometryContractError(
                "no mm/pixel for this frame; a per-frame scale is never borrowed from another frame"
            )
        sx, sy = self.per_frame[int(frame_order)]
        return float(sx), float(sy)

    def raw_px_to_mm(self, points: Points2D, *, frame_order: int | None = None) -> Points2D:
        require_space(points.space, CoordSpace.RAW_FRAME_PX, what="raw_px_to_mm input")
        sx, sy = self.scale_for(frame_order)
        return Points2D(points.xy * np.array([sx, sy], dtype=np.float64), CoordSpace.MM_XY)

    def mm_to_raw_px(self, points: Points2D, *, frame_order: int | None = None) -> Points2D:
        require_space(points.space, CoordSpace.MM_XY, what="mm_to_raw_px input")
        sx, sy = self.scale_for(frame_order)
        return Points2D(points.xy / np.array([sx, sy], dtype=np.float64), CoordSpace.RAW_FRAME_PX)


def length_change_under_anisotropy(angle_rad: float, ratio_k: float) -> float:
    """Relative length change when y is scaled by k relative to x.

    A segment of unit length at angle theta (from the x axis) becomes
    sqrt(cos^2 + k^2 sin^2). Used to report how a pixel-space length would
    move if the unknown x/y mm scales differed by k (report 1.7.2).
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return math.sqrt(c * c + (ratio_k * s) ** 2) - 1.0
