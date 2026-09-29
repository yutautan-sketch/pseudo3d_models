from __future__ import annotations

import math
from typing import Callable, Mapping

import numpy as np

from stage5.geometry.frame_geometry import MIN_COMPONENT_POINTS, QUANTILE_HIGH, QUANTILE_LOW
from stage5.geometry.types import (
    CoordSpace,
    FailureReason,
    FrameGeometry,
    GeometryContractError,
    InstanceGeometry,
    VideoFLEstimate,
)

# ----------------------------------------------------------------------------
# S5-17 S17-1: video-level length estimates for representations (a)-(e).
#
# Management document 4.2.
#
#   (a) a_aabb       frame AABB long side,            video q90 over usable frames
#   (b) b_obb        oriented-box major extent,        video q90
#   (c) c_axis       endpoint distance (= (b) length), video q90
#   (d) d_bestframe  (c) at ONE selected frame
#   (e) e_pseudo3d   robust major extent of the video's pseudo-3D points;
#                    reference only, unit pseudo3d_units, never a candidate
#
# Frame selection for (d) is three separate functions, never one function
# with a flag, so that a GT-informed choice cannot leak into a practical
# result:
#   * GT representation:  score = GT points of the instance ("gt"); in the
#                         held-out evaluation, points of half A only
#   * practical:          score = points x mean predicted probability
#                         ("prediction"); prediction information only
#   * oracle:             the GT-selected frame applied to the prediction
#                         ("oracle_gt", is_oracle=True); reported separately
# Frames touching the crop border are not selectable (the femur may be cut
# off). Ties go to the smallest frame_order.
#
# The quantities returned are lengths in the instances' coordinate space. No
# value is converted to mm here, and nothing here is a clinical FL.
# ----------------------------------------------------------------------------

REP_AABB = "a_aabb"
REP_OBB = "b_obb"
REP_AXIS = "c_axis"
REP_BEST_FRAME = "d_bestframe"
REP_PSEUDO3D = "e_pseudo3d"
FRAME_REPRESENTATIONS = (REP_AABB, REP_OBB, REP_AXIS)
CANDIDATE_REPRESENTATIONS = (REP_AABB, REP_OBB, REP_AXIS, REP_BEST_FRAME)
# Fixed precedence used only to break ties (management 4.4). Not a ranking of correctness.
CANDIDATE_PRECEDENCE = (REP_AXIS, REP_BEST_FRAME, REP_OBB, REP_AABB)

VIDEO_QUANTILE = 0.90
AGG_Q90 = "q90"
AGG_BEST_FRAME = "best_frame"
AGG_VIDEO_PCA = "video_pca"

BASIS_GT = "gt"
BASIS_GT_HOLDOUT_A = "gt_holdout_a"
BASIS_PREDICTION = "prediction"
BASIS_ORACLE_GT = "oracle_gt"

ScoreFn = Callable[[InstanceGeometry], float]


def frame_length(instance: InstanceGeometry, representation: str) -> float | None:
    """Per-frame length of one instance, or None when the representation cannot use it.

    (a) needs only a positive long side. A horizontal or vertical segment has a
    zero-area AABB but a well-defined long side, so it is not refused here --
    otherwise axis-parallel segments alone would be penalised. The zero-area
    condition belongs to area-based quantities (held-out coverage, relative
    inflation), which apply it themselves (management review of I-4).
    """
    if representation == REP_AABB:
        return instance.aabb_long_side if instance.aabb_long_side > 0 else None
    if representation in (REP_OBB, REP_AXIS, REP_BEST_FRAME):
        return instance.length if instance.axis_usable else None
    raise GeometryContractError(f"frame_length does not handle {representation!r}")


def _failure(representation: str, aggregation: str, rule: str, space: CoordSpace, reason: str,
             **extra) -> VideoFLEstimate:
    return VideoFLEstimate(
        representation=representation,
        aggregation=aggregation,
        instance_rule=rule,
        coord_space=space,
        value=None,
        frames_used=(),
        failure=reason,
        **extra,
    )


def _check_space(frames: Mapping[int, FrameGeometry], space: CoordSpace) -> None:
    for geometry in frames.values():
        for instance in geometry.instances:
            if instance.coord_space is not space:
                raise GeometryContractError(
                    f"frame {geometry.frame_order} is in {instance.coord_space.value}, expected {space.value}"
                )


def estimate_quantile(
    frames: Mapping[int, FrameGeometry],
    representation: str,
    *,
    instance_rule: str,
    coord_space: CoordSpace,
) -> VideoFLEstimate:
    """(a)-(c): q90 of the per-frame lengths over usable frames."""
    if representation not in FRAME_REPRESENTATIONS:
        raise GeometryContractError(f"estimate_quantile handles {FRAME_REPRESENTATIONS}, not {representation!r}")
    _check_space(frames, coord_space)
    lengths: list[float] = []
    used: list[int] = []
    any_present = False
    for frame in sorted(frames):
        instance = frames[frame].instance(instance_rule)
        if instance is None:
            continue
        any_present = True
        length = frame_length(instance, representation)
        if length is not None:
            lengths.append(length)
            used.append(frame)
    if not lengths:
        reason = FailureReason.NO_USABLE_FRAME if any_present else FailureReason.EMPTY
        return _failure(representation, AGG_Q90, instance_rule, coord_space, reason)
    return VideoFLEstimate(
        representation=representation,
        aggregation=AGG_Q90,
        instance_rule=instance_rule,
        coord_space=coord_space,
        value=float(np.quantile(np.asarray(lengths), VIDEO_QUANTILE)),
        frames_used=tuple(used),
    )


def gt_point_score(instance: InstanceGeometry) -> float:
    return float(instance.n_points)


def prediction_score(instance: InstanceGeometry) -> float:
    if instance.mean_probability is None:
        raise GeometryContractError("prediction_score needs the instance's mean predicted probability")
    return float(instance.n_points) * float(instance.mean_probability)


def select_best_frame(
    frames: Mapping[int, FrameGeometry],
    *,
    instance_rule: str,
    score: ScoreFn,
) -> tuple[int, float] | None:
    """The selectable frame with the highest score; ties -> smallest frame_order."""
    best: tuple[float, int] | None = None
    for frame in sorted(frames):
        instance = frames[frame].instance(instance_rule)
        if instance is None or instance.touches_crop_border or frame_length(instance, REP_BEST_FRAME) is None:
            continue
        value = score(instance)
        if not math.isfinite(value):
            raise GeometryContractError("a selection score must be finite")
        if best is None or value > best[0]:
            best = (value, frame)
    return None if best is None else (best[1], best[0])


def _best_frame_estimate(
    frames: Mapping[int, FrameGeometry],
    *,
    instance_rule: str,
    coord_space: CoordSpace,
    score: ScoreFn,
    basis: str,
) -> VideoFLEstimate:
    _check_space(frames, coord_space)
    chosen = select_best_frame(frames, instance_rule=instance_rule, score=score)
    if chosen is None:
        any_present = any(geometry.present for geometry in frames.values())
        reason = FailureReason.NO_USABLE_FRAME if any_present else FailureReason.EMPTY
        return _failure(REP_BEST_FRAME, AGG_BEST_FRAME, instance_rule, coord_space, reason, selection_basis=basis)
    frame, value = chosen
    instance = frames[frame].instance(instance_rule)
    assert instance is not None
    return VideoFLEstimate(
        representation=REP_BEST_FRAME,
        aggregation=AGG_BEST_FRAME,
        instance_rule=instance_rule,
        coord_space=coord_space,
        value=instance.length,
        frames_used=(frame,),
        selection_basis=basis,
        selection_score=value,
    )


def estimate_best_frame_gt(
    gt_frames: Mapping[int, FrameGeometry], *, instance_rule: str, coord_space: CoordSpace
) -> VideoFLEstimate:
    """(d) on GT: frame chosen by GT point count."""
    return _best_frame_estimate(
        gt_frames, instance_rule=instance_rule, coord_space=coord_space, score=gt_point_score, basis=BASIS_GT
    )


def estimate_best_frame_prediction(
    pred_frames: Mapping[int, FrameGeometry], *, instance_rule: str, coord_space: CoordSpace
) -> VideoFLEstimate:
    """(d) practical: frame chosen from prediction information only."""
    return _best_frame_estimate(
        pred_frames, instance_rule=instance_rule, coord_space=coord_space, score=prediction_score,
        basis=BASIS_PREDICTION,
    )


def estimate_best_frame_oracle(
    gt_frames: Mapping[int, FrameGeometry],
    pred_frames: Mapping[int, FrameGeometry],
    *,
    instance_rule: str,
    coord_space: CoordSpace,
) -> VideoFLEstimate:
    """(d) oracle: the GT-selected frame, measured on the prediction. Never mixed into practical results."""
    _check_space(gt_frames, coord_space)
    _check_space(pred_frames, coord_space)
    chosen = select_best_frame(gt_frames, instance_rule=instance_rule, score=gt_point_score)
    if chosen is None:
        return _failure(REP_BEST_FRAME, AGG_BEST_FRAME, instance_rule, coord_space,
                        FailureReason.NO_USABLE_FRAME, selection_basis=BASIS_ORACLE_GT, is_oracle=True)
    frame, _gt_score = chosen
    geometry = pred_frames.get(frame)
    instance = None if geometry is None else geometry.instance(instance_rule)
    length = None if instance is None else frame_length(instance, REP_BEST_FRAME)
    if length is None:
        reason = FailureReason.EMPTY if instance is None else FailureReason.NO_USABLE_FRAME
        return _failure(REP_BEST_FRAME, AGG_BEST_FRAME, instance_rule, coord_space, reason,
                        selection_basis=BASIS_ORACLE_GT, is_oracle=True)
    return VideoFLEstimate(
        representation=REP_BEST_FRAME,
        aggregation=AGG_BEST_FRAME,
        instance_rule=instance_rule,
        coord_space=coord_space,
        value=length,
        frames_used=(frame,),
        selection_basis=BASIS_ORACLE_GT,
        selection_score=None,
        is_oracle=True,
    )


def estimate_pseudo3d_axis(points_xyz: np.ndarray) -> VideoFLEstimate:
    """(e) reference value: robust major extent of the video's pseudo-3D points.

    The Z scale of the pseudo-3D reconstruction is not a confirmed physical
    unit, so the value is in pseudo3d_units and is never a selection candidate.
    """
    xyz = np.asarray(points_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise GeometryContractError(f"pseudo-3D points must be [N, 3], got {xyz.shape}")
    common = dict(candidate_eligible=False)
    if xyz.shape[0] < MIN_COMPONENT_POINTS:
        return _failure(REP_PSEUDO3D, AGG_VIDEO_PCA, "all_points", CoordSpace.PSEUDO3D,
                        FailureReason.INSUFFICIENT_POINTS, **common)
    if not np.all(np.isfinite(xyz)):
        raise GeometryContractError("pseudo-3D points contain non-finite values")
    centered = xyz - xyz.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / xyz.shape[0])
    if float(eigenvalues[2]) <= 1e-18:
        return _failure(REP_PSEUDO3D, AGG_VIDEO_PCA, "all_points", CoordSpace.PSEUDO3D,
                        FailureReason.COINCIDENT, **common)
    if max(float(eigenvalues[1]), 0.0) / float(eigenvalues[2]) > 0.9:
        return _failure(REP_PSEUDO3D, AGG_VIDEO_PCA, "all_points", CoordSpace.PSEUDO3D,
                        FailureReason.AXIS_AMBIGUOUS, **common)
    t = centered @ eigenvectors[:, 2]
    lo, hi = np.quantile(t, [QUANTILE_LOW, QUANTILE_HIGH])
    return VideoFLEstimate(
        representation=REP_PSEUDO3D,
        aggregation=AGG_VIDEO_PCA,
        instance_rule="all_points",
        coord_space=CoordSpace.PSEUDO3D,
        value=float(hi - lo),
        frames_used=(),
        candidate_eligible=False,
    )


def estimate_representation(
    frames: Mapping[int, FrameGeometry],
    representation: str,
    *,
    instance_rule: str,
    coord_space: CoordSpace,
    source: str,
) -> VideoFLEstimate:
    """Dispatch (a)-(d) for GT (`source="gt"`) or a prediction (`source="prediction"`)."""
    if representation in FRAME_REPRESENTATIONS:
        return estimate_quantile(frames, representation, instance_rule=instance_rule, coord_space=coord_space)
    if representation == REP_BEST_FRAME:
        if source == "gt":
            return estimate_best_frame_gt(frames, instance_rule=instance_rule, coord_space=coord_space)
        if source == "prediction":
            return estimate_best_frame_prediction(frames, instance_rule=instance_rule, coord_space=coord_space)
        raise GeometryContractError(f"unknown source {source!r}; the oracle has its own function")
    raise GeometryContractError(f"estimate_representation does not handle {representation!r}")
