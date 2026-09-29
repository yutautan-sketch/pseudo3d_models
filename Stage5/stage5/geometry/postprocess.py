from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stage5.geometry.frame_geometry import component_index_sets, group_by_frame
from stage5.geometry.types import CoordSpace, GeometryContractError, Points2D, require_space

# ----------------------------------------------------------------------------
# S5-17 S17-1: non-learned post-processing of a saved prediction.
#
# Management document 4.2. Exactly five, no threshold search:
#
#   P0     none -- the saved `pred_label`
#   P1     per frame, the largest connected component only
#          (link distance 4 local px, minimum 5 points; same as the GT side)
#   P2     frame persistence -- keep a component only if a component of the
#          same mask lies in a frame within +-2 frame_order values with its
#          centroid within `p2_distance_raw_px`
#   P3     probability top-K -- in each frame that has at least one
#          predicted-positive point, the K points of highest `prob_femur`
#          (points below 0.5 included)
#   P1+P2  P1, then P2
#
# The two constants come from train_core GT only and are passed in; they are
# never tuned on validation:
#   p2_distance_raw_px = 0.5 x median train_core GT length (raw px)
#   p3_top_k           = median train_core valid GT-positive points per positive
#                        frame (a frame with >= 1 valid GT-positive point; this is
#                        not the >= 5-point instance condition)
#
# P3 is restricted to frames the saved prediction already marks as present.
# Applied to every frame, a top-K rule would mark every frame present by
# construction and frame-presence metrics would stop meaning anything; this
# restriction is recorded in the S17-1 report as a specification detail.
# ----------------------------------------------------------------------------

POSTPROCESSES = ("P0", "P1", "P2", "P3", "P1+P2")
# Tie-break order among post-processes (report 1.6.5): simplest first.
POSTPROCESS_PRECEDENCE = ("P0", "P1", "P2", "P1+P2", "P3")
P2_FRAME_WINDOW = 2
P2_DISTANCE_FACTOR = 0.5


@dataclass(frozen=True)
class PostprocessConstants:
    p2_distance_raw_px: float
    p3_top_k: int
    source: str = "train_core_gt"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.p2_distance_raw_px) and self.p2_distance_raw_px > 0):
            raise GeometryContractError("p2_distance_raw_px must be finite and positive")
        if int(self.p3_top_k) != self.p3_top_k or self.p3_top_k < 1:
            raise GeometryContractError("p3_top_k must be a positive integer")

    @classmethod
    def from_train_core(cls, gt_lengths_raw_px: Sequence[float], gt_points_per_positive_frame: Sequence[int]
                        ) -> "PostprocessConstants":
        """P2 from M1 frames with a defined length; P3 from frames with >= 1 valid GT-positive point."""
        lengths = np.asarray([v for v in gt_lengths_raw_px if v is not None], dtype=np.float64)
        counts = np.asarray(gt_points_per_positive_frame, dtype=np.float64)
        if lengths.size == 0 or counts.size == 0:
            raise GeometryContractError("post-process constants need train_core GT lengths and counts")
        # Round half up, stated explicitly rather than inherited from round().
        k = max(1, int(math.floor(float(np.median(counts)) + 0.5)))
        return cls(p2_distance_raw_px=P2_DISTANCE_FACTOR * float(np.median(lengths)), p3_top_k=k)


def _check(frame_order: np.ndarray, mask: np.ndarray, *points: Points2D) -> None:
    n = np.asarray(frame_order).shape[0]
    if np.asarray(mask).shape != (n,):
        raise GeometryContractError("the mask does not align with frame_order")
    for p in points:
        if len(p) != n:
            raise GeometryContractError("per-point coordinates do not align with frame_order")


def p1_largest_component(frame_order: np.ndarray, local_points: Points2D, mask: np.ndarray) -> np.ndarray:
    require_space(local_points.space, CoordSpace.LOCAL_CROP_PX, what="P1 local points")
    _check(frame_order, mask, local_points)
    out = np.zeros_like(np.asarray(mask, dtype=bool))
    for members in group_by_frame(frame_order, mask).values():
        components = component_index_sets(local_points.xy[members])
        if components:
            out[members[components[0]]] = True
    return out


def p2_persistence(
    frame_order: np.ndarray,
    local_points: Points2D,
    raw_points: Points2D,
    mask: np.ndarray,
    *,
    distance_raw_px: float,
) -> np.ndarray:
    require_space(local_points.space, CoordSpace.LOCAL_CROP_PX, what="P2 local points")
    require_space(raw_points.space, CoordSpace.RAW_FRAME_PX, what="P2 raw points")
    _check(frame_order, mask, local_points, raw_points)
    components: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for frame, members in group_by_frame(frame_order, mask).items():
        for component in component_index_sets(local_points.xy[members]):
            indices = members[component]
            components.setdefault(frame, []).append((indices, raw_points.xy[indices].mean(axis=0)))

    out = np.zeros_like(np.asarray(mask, dtype=bool))
    for frame, items in components.items():
        neighbours = [
            centroid
            for other in range(frame - P2_FRAME_WINDOW, frame + P2_FRAME_WINDOW + 1)
            if other != frame
            for _indices, centroid in components.get(other, ())
        ]
        if not neighbours:
            continue
        stacked = np.stack(neighbours)
        for indices, centroid in items:
            if np.any(np.linalg.norm(stacked - centroid, axis=1) <= distance_raw_px):
                out[indices] = True
    return out


def p3_top_k(frame_order: np.ndarray, pred_mask: np.ndarray, probabilities: np.ndarray, *, top_k: int) -> np.ndarray:
    prob = np.asarray(probabilities, dtype=np.float64)
    _check(frame_order, pred_mask)
    if prob.shape != np.asarray(pred_mask).shape:
        raise GeometryContractError("probabilities do not align with the mask")
    present_frames = set(group_by_frame(frame_order, pred_mask))
    out = np.zeros(prob.shape, dtype=bool)
    for frame, members in group_by_frame(frame_order, np.ones(prob.shape, dtype=bool)).items():
        if frame not in present_frames:
            continue
        ranking = np.lexsort((members, -prob[members]))     # prob desc, then lower index
        out[members[ranking[:top_k]]] = True
    return out


def apply_postprocess(
    name: str,
    *,
    frame_order: np.ndarray,
    local_points: Points2D,
    raw_points: Points2D,
    pred_mask: np.ndarray,
    probabilities: np.ndarray,
    constants: PostprocessConstants,
) -> np.ndarray:
    pred = np.asarray(pred_mask, dtype=bool)
    if name == "P0":
        return pred.copy()
    if name == "P1":
        return p1_largest_component(frame_order, local_points, pred)
    if name == "P2":
        return p2_persistence(frame_order, local_points, raw_points, pred, distance_raw_px=constants.p2_distance_raw_px)
    if name == "P3":
        return p3_top_k(frame_order, pred, probabilities, top_k=constants.p3_top_k)
    if name == "P1+P2":
        largest = p1_largest_component(frame_order, local_points, pred)
        return p2_persistence(frame_order, local_points, raw_points, largest,
                              distance_raw_px=constants.p2_distance_raw_px)
    raise GeometryContractError(f"unknown post-process {name!r}; expected one of {POSTPROCESSES}")
