from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------
# S5-17 S17-1: coordinate spaces, units, geometry records and failure states.
#
# Management document 4.1 / 5. Three kinds of "no answer" are kept apart on
# purpose, because they end up in different denominators:
#
#   * GeometryContractError -- a contract violation (mixed coordinate spaces,
#     out-of-range coordinates, a missing mm scale that was asked for). The
#     run stops; nothing is counted.
#   * InputUnevaluable -- the input side cannot be evaluated (no usable crop
#     inverse, e.g. `offset_crop_fallback_resize`). The video is removed from
#     the evaluable count and reported by reason.
#   * FailureReason on a record -- the METHOD failed (no prediction, no usable
#     frame, degenerate points). It stays in the evaluable count as a failure.
#
# An undefined quantity (a zero denominator, L = 0, nothing to measure) is
# None, never 0. A positive denominator with a zero numerator is a real 0.
#
# `selection_score` is an internal ranking value for choosing a frame. It is
# not a probability and not a confidence, and is never reported as either.
# ----------------------------------------------------------------------------


class CoordSpace(str, Enum):
    LOCAL_CROP_PX = "local_crop_px"      # `pixel_xy` as stored in the teacher H5
    RAW_FRAME_PX = "raw_frame_px"        # original frame pixels (crop/resize inverted)
    RAW_FRAME_NORM = "raw_frame_norm"    # original frame / (raw_width, raw_height)
    MM_XY = "mm_xy"                      # original-frame millimetres; only via a confirmed PixelToMM
    PSEUDO3D = "pseudo3d"                # teacher `points` XYZ; physical unit unconfirmed


class Unit(str, Enum):
    LOCAL_PX = "local_px"
    RAW_PX = "raw_px"
    NORM = "normalized"
    MM = "mm"
    PSEUDO3D_UNITS = "pseudo3d_units"


UNIT_OF_SPACE = {
    CoordSpace.LOCAL_CROP_PX: Unit.LOCAL_PX,
    CoordSpace.RAW_FRAME_PX: Unit.RAW_PX,
    CoordSpace.RAW_FRAME_NORM: Unit.NORM,
    CoordSpace.MM_XY: Unit.MM,
    CoordSpace.PSEUDO3D: Unit.PSEUDO3D_UNITS,
}


class GeometryContractError(ValueError):
    """A contract violation. The run must stop; nothing is counted."""


class InputUnevaluable(ValueError):
    """The input side cannot be evaluated. Carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class InputReason:
    CROP_FALLBACK_RESIZE = "crop_fallback_resize"
    CROP_UNKNOWN_MODE = "crop_unknown_preprocess_mode"
    CROP_FIELD_MISSING = "crop_field_missing"
    CROP_SCALE_INVALID = "crop_scale_not_finite_positive"
    CROP_DIMENSION_INVALID = "crop_dimension_invalid"


class FailureReason:
    """Method-side failures. These stay in the evaluable denominator."""

    EMPTY = "empty"                              # no points of the class in the frame
    INSUFFICIENT_POINTS = "insufficient_points"  # points, but no component of the minimum size
    COINCIDENT = "coincident_points"             # all points identical: no axis, zero extent
    AXIS_AMBIGUOUS = "axis_ambiguous"            # lambda2 / lambda1 above the fixed ratio
    ZERO_WIDTH = "zero_width"                    # collinear: minor extent is zero
    ZERO_AREA = "zero_area_aabb"                 # AABB area 0: area-based quantities only, never length
    NO_PREDICTION = "no_prediction"              # the saved prediction has no positive point in the video
    REMOVED_BY_POSTPROCESS = "removed_by_postprocess"  # positives existed, the post-process removed them all
    GT_REFERENCE_UNDEFINED = "gt_reference_undefined"  # the GT value to compare with is itself undefined
    NO_USABLE_FRAME = "no_usable_frame"          # no frame usable for this representation
    HOLDOUT_TOO_SMALL = "holdout_too_small"      # n < 10 before the A/B split
    HOLDOUT_TOO_FEW_FRAMES = "holdout_too_few_frames"


class InstanceStatus:
    OK = "ok"
    AXIS_AMBIGUOUS = FailureReason.AXIS_AMBIGUOUS
    ZERO_WIDTH = FailureReason.ZERO_WIDTH
    COINCIDENT = FailureReason.COINCIDENT
    PRIOR = "prior_only"


def require_space(actual: CoordSpace, expected: CoordSpace, *, what: str) -> None:
    if actual is not expected:
        raise GeometryContractError(
            f"{what} is in {actual.value}, expected {expected.value}; coordinate spaces are never mixed implicitly"
        )


@dataclass(frozen=True)
class Points2D:
    """An [N, 2] float64 coordinate array tagged with its space."""

    xy: np.ndarray
    space: CoordSpace

    def __post_init__(self) -> None:
        array = np.asarray(self.xy, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise GeometryContractError(f"Points2D expects shape [N, 2], got {array.shape}")
        if self.space is CoordSpace.PSEUDO3D:
            raise GeometryContractError("PSEUDO3D is three-dimensional and is not a Points2D space")
        object.__setattr__(self, "xy", array)

    @property
    def unit(self) -> Unit:
        return UNIT_OF_SPACE[self.space]

    def __len__(self) -> int:
        return int(self.xy.shape[0])


@dataclass(frozen=True)
class InstanceGeometry:
    """Geometry of one instance (one connected component, or a merged set).

    Oriented quantities are expressed in a frame whose origin is `centroid`,
    with `t` along the major axis and `s` along the minor axis. The box of
    representation (b) is t in t_range, s in s_range; `center` is the box
    centre and the endpoints of representation (c) sit on the box centre line
    (s = s_mid), so (b) and (c) cover exactly the same region.
    """

    n_points: int
    status: str
    coord_space: CoordSpace
    centroid: tuple[float, float]
    aabb: tuple[float, float, float, float]          # x_min, y_min, x_max, y_max
    aabb_long_side: float
    eig_ratio: float | None                          # lambda2 / lambda1; None when lambda1 == 0
    axis_angle: float | None = None                  # [0, pi); no sign
    t_range: tuple[float, float] | None = None       # (t_0.02, t_0.98)
    s_range: tuple[float, float] | None = None       # (s_0.02, s_0.98)
    center: tuple[float, float] | None = None        # box centre (t_mid, s_mid) in coord_space
    length: float | None = None                      # t_0.98 - t_0.02
    width: float | None = None                       # s_0.98 - s_0.02
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None
    touches_crop_border: bool = False
    mean_probability: float | None = None            # predictions only; not a confidence

    @property
    def unit(self) -> Unit:
        return UNIT_OF_SPACE[self.coord_space]

    @property
    def axis_usable(self) -> bool:
        """Whether representations (b)-(d) may use this instance.

        A prior-only instance is synthetic (no points) but carries a defined
        axis and extent, so it is usable; it is never mistaken for data
        because its status and n_points == 0 say what it is.
        """
        return self.status in (InstanceStatus.OK, InstanceStatus.PRIOR)

    @property
    def axis_unit(self) -> tuple[float, float] | None:
        if self.axis_angle is None:
            return None
        return (float(np.cos(self.axis_angle)), float(np.sin(self.axis_angle)))


@dataclass(frozen=True)
class FrameGeometry:
    frame_order: int
    present: bool
    status: str                                      # "ok", FailureReason.EMPTY or INSUFFICIENT_POINTS
    instances: tuple[InstanceGeometry, ...] = ()     # descending point count (M1 is instances[0])
    merged: InstanceGeometry | None = None           # all kept components as one (M2)
    instance_point_indices: tuple[np.ndarray, ...] = field(default=(), compare=False, repr=False)

    def instance(self, rule: str) -> InstanceGeometry | None:
        if not self.present:
            return None
        if rule == "M1":
            return self.instances[0]
        if rule == "M2":
            return self.merged
        raise GeometryContractError(f"unknown instance rule {rule!r}; expected 'M1' or 'M2'")


@dataclass(frozen=True)
class VideoFLEstimate:
    """A video-level length estimate. `value` is None whenever `failure` is set."""

    representation: str
    aggregation: str
    instance_rule: str
    coord_space: CoordSpace
    value: float | None
    frames_used: tuple[int, ...]
    failure: str | None = None
    selection_basis: str | None = None               # "gt", "gt_holdout_a", "prediction", "oracle_gt"
    selection_score: float | None = None             # ranking value only; not a probability
    is_oracle: bool = False
    candidate_eligible: bool = True                  # False for (e) pseudo-3D

    @property
    def unit(self) -> Unit:
        return UNIT_OF_SPACE[self.coord_space]

    def as_record(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "aggregation": self.aggregation,
            "instance_rule": self.instance_rule,
            "coord_space": self.coord_space.value,
            "unit": self.unit.value,
            "value": self.value,
            "frames_used": list(self.frames_used),
            "failure": self.failure,
            "selection_basis": self.selection_basis,
            "selection_score": self.selection_score,
            "is_oracle": self.is_oracle,
            "candidate_eligible": self.candidate_eligible,
        }
