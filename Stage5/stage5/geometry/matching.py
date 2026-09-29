from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stage5.geometry.types import GeometryContractError, InstanceGeometry

# ----------------------------------------------------------------------------
# S5-17 S17-1: multiple instances per frame.
#
# Management document 4.2. Three rules, kept apart:
#   M1  the largest instance only (FrameGeometry.instance("M1"))
#   M2  all kept instances as one (FrameGeometry.instance("M2"))
#   M3  GT <-> prediction assignment by centroid distance with a gate of
#       0.5 x the GT instance's length (its AABB long side when the axis is
#       not usable). Used for frame-level detection counts and matched-pair
#       errors only.
#
# The assignment is a minimum-cost Hungarian solve (implemented here; no scipy
# dependency). Pairs beyond the gate are given a cost larger than any sum of
# admissible costs, so the solver first maximises the number of admissible
# pairs and then minimises their total distance; inadmissible pairs it is
# forced to make are dropped afterwards. Centroids, not box centres, are used
# because a centroid exists for every instance.
# ----------------------------------------------------------------------------

M3_GATE_FACTOR = 0.5


def hungarian_min_cost(cost: np.ndarray) -> list[int]:
    """Row -> column assignment minimising total cost; rows <= columns required."""
    matrix = np.asarray(cost, dtype=np.float64)
    n, m = matrix.shape
    if n > m:
        raise GeometryContractError("hungarian_min_cost expects rows <= columns; transpose first")
    if not np.all(np.isfinite(matrix)):
        raise GeometryContractError("assignment costs must be finite")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j] = current
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[tuple[int, int, float], ...]     # (gt index, prediction index, centroid distance)
    unmatched_gt: tuple[int, ...]
    unmatched_pred: tuple[int, ...]

    @property
    def tp(self) -> int:
        return len(self.pairs)

    @property
    def fn(self) -> int:
        return len(self.unmatched_gt)

    @property
    def fp(self) -> int:
        return len(self.unmatched_pred)


def gate_for(instance: InstanceGeometry, factor: float = M3_GATE_FACTOR) -> float:
    scale = instance.length if (instance.axis_usable and instance.length is not None) else instance.aabb_long_side
    return factor * float(scale)


def match_instances(
    gt: Sequence[InstanceGeometry],
    pred: Sequence[InstanceGeometry],
    *,
    gate_factor: float = M3_GATE_FACTOR,
) -> MatchResult:
    for a in gt:
        for b in pred:
            if a.coord_space is not b.coord_space:
                raise GeometryContractError("GT and prediction instances are in different coordinate spaces")
    if not gt or not pred:
        return MatchResult(pairs=(), unmatched_gt=tuple(range(len(gt))), unmatched_pred=tuple(range(len(pred))))

    g = np.asarray([a.centroid for a in gt], dtype=np.float64)
    q = np.asarray([b.centroid for b in pred], dtype=np.float64)
    distance = np.linalg.norm(g[:, None, :] - q[None, :, :], axis=2)
    gates = np.asarray([gate_for(a, gate_factor) for a in gt])
    admissible = distance <= gates[:, None]
    big = 1.0 + float(distance.sum()) * 2.0 + 1.0e6
    cost = np.where(admissible, distance, big)

    if len(gt) <= len(pred):
        rows = hungarian_min_cost(cost)
        assigned = [(i, j) for i, j in enumerate(rows) if j >= 0]
    else:
        cols = hungarian_min_cost(cost.T)
        assigned = [(i, j) for j, i in enumerate(cols) if i >= 0]

    pairs = tuple(sorted((i, j, float(distance[i, j])) for i, j in assigned if admissible[i, j]))
    matched_gt = {i for i, _j, _d in pairs}
    matched_pred = {j for _i, j, _d in pairs}
    return MatchResult(
        pairs=pairs,
        unmatched_gt=tuple(i for i in range(len(gt)) if i not in matched_gt),
        unmatched_pred=tuple(j for j in range(len(pred)) if j not in matched_pred),
    )


def centroid_distance(a: InstanceGeometry, b: InstanceGeometry) -> float:
    return math.dist(a.centroid, b.centroid)
