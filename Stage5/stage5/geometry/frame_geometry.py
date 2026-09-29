from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from stage5.geometry.types import (
    CoordSpace,
    FailureReason,
    FrameGeometry,
    GeometryContractError,
    InstanceGeometry,
    InstanceStatus,
    Points2D,
    require_space,
)

# ----------------------------------------------------------------------------
# S5-17 S17-1: points of one frame -> instances -> oriented geometry.
#
# Management document 4.2 / 4.3. Constants are fixed before any real data is
# read and are not tuned on validation:
#
#   * instances are single-linkage components in LOCAL crop pixels, link
#     distance 4, components under 5 points dropped (the S5-15 GT region
#     count used the same linkage; its O(n^2) `component_sizes` is not
#     imported because the core must not depend on `checks/` and prediction
#     frames can exceed its 20000-point cap -- parity is a synthetic test)
#   * geometry is computed in RAW_FRAME_PX (or MM_XY once a scale exists),
#     never in local pixels, because the `resize` mode is anisotropic
#   * the oriented box is the 2..98 % quantile range along the PCA major (t)
#     and minor (s) axes, measured from the centroid. The endpoints sit on the
#     BOX centre line s = s_mid, not on the line through the centroid, so the
#     band of representation (c) and the box of (b) are the same region
#     (report 3.5). centroid and centre are kept as separate fields.
#   * lambda2 / lambda1 > 0.9 marks the axis ambiguous; a zero minor extent
#     marks the instance collinear. Both keep their numbers but are not
#     usable for (b)-(d).
# ----------------------------------------------------------------------------

LINK_DISTANCE_LOCAL_PX = 4.0
MIN_COMPONENT_POINTS = 5
QUANTILE_LOW = 0.02
QUANTILE_HIGH = 0.98
AXIS_AMBIGUOUS_RATIO = 0.9
CROP_BORDER_MARGIN_PX = 2.0

# Numerical floors, not tuning parameters: an eigenvalue at or below
# COINCIDENT_EIGEN_FLOOR (raw px^2) is "all points identical", and a minor
# extent at or below ZERO_WIDTH_RELATIVE * max(1, length) is "collinear".
COINCIDENT_EIGEN_FLOOR = 1e-18
ZERO_WIDTH_RELATIVE = 1e-9

_HALF_NEIGHBOURHOOD = ((0, 0), (1, -1), (1, 0), (1, 1), (0, 1))


def connected_components(xy: np.ndarray, *, link_distance: float = LINK_DISTANCE_LOCAL_PX) -> np.ndarray:
    """Single-linkage labels for [N, 2] points; label 0 is the largest component.

    Two points join when their Euclidean distance is <= link_distance. Points
    are bucketed into square cells of side link_distance, so any linked pair
    lies in the same or an adjacent cell. Ties in component size are broken by
    the smallest member index, so labels are deterministic.
    """
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise GeometryContractError(f"connected_components expects [N, 2], got {points.shape}")
    count = int(points.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.all(np.isfinite(points)):
        raise GeometryContractError("connected_components received non-finite coordinates")
    if not link_distance > 0:
        raise GeometryContractError("link_distance must be positive")

    cells = np.floor(points / float(link_distance)).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    members_by_cell = {
        (int(unique_cells[inverse[group[0]], 0]), int(unique_cells[inverse[group[0]], 1])): group
        for group in np.split(order, boundaries)
    }

    parent = np.arange(count)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    limit = float(link_distance) ** 2
    for (cx, cy), members in members_by_cell.items():
        a = points[members]
        for dx, dy in _HALF_NEIGHBOURHOOD:
            other = members_by_cell.get((cx + dx, cy + dy))
            if other is None:
                continue
            b = points[other]
            delta = a[:, None, :] - b[None, :, :]
            close = np.einsum("ijk,ijk->ij", delta, delta) <= limit
            if dx == 0 and dy == 0:
                close = np.triu(close, k=1)
            for i, j in zip(*np.nonzero(close)):
                ra, rb = find(int(members[i])), find(int(other[j]))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(count)], dtype=np.int64)
    root_ids, root_inverse, sizes = np.unique(roots, return_inverse=True, return_counts=True)
    first_member = np.full(root_ids.shape[0], count, dtype=np.int64)
    np.minimum.at(first_member, np.asarray(root_inverse).reshape(-1), np.arange(count))
    ranking = np.lexsort((first_member, -sizes))            # size desc, then first member
    new_label = np.empty_like(ranking)
    new_label[ranking] = np.arange(ranking.shape[0])
    return new_label[np.asarray(root_inverse).reshape(-1)]


def component_index_sets(
    local_xy: np.ndarray,
    *,
    link_distance: float = LINK_DISTANCE_LOCAL_PX,
    min_points: int = MIN_COMPONENT_POINTS,
) -> list[np.ndarray]:
    """Indices (into local_xy) of each kept component, largest first."""
    labels = connected_components(local_xy, link_distance=link_distance)
    kept: list[np.ndarray] = []
    for label in range(int(labels.max()) + 1 if labels.size else 0):
        members = np.flatnonzero(labels == label)
        if members.size >= min_points:
            kept.append(members)
    return kept


def instance_geometry(
    points: Points2D,
    *,
    local_points: Points2D | None = None,
    local_bounds_wh: tuple[int, int] | None = None,
    probabilities: np.ndarray | None = None,
) -> InstanceGeometry:
    """Oriented geometry of one point set.

    RAW_FRAME_PX or MM_XY for measurement. RAW_FRAME_NORM is accepted for the
    prior only, which is built in normalised original-frame coordinates; a
    normalised length is not isotropic and is never reported as a length.
    """
    if points.space not in (CoordSpace.RAW_FRAME_PX, CoordSpace.MM_XY, CoordSpace.RAW_FRAME_NORM):
        raise GeometryContractError(
            f"geometry is computed in raw_frame_px, mm_xy or (prior only) raw_frame_norm, not {points.space.value}"
        )
    xy = points.xy
    count = len(points)
    if count == 0:
        raise GeometryContractError("instance_geometry needs at least one point")
    if not np.all(np.isfinite(xy)):
        raise GeometryContractError("instance_geometry received non-finite coordinates")

    touches = False
    if local_points is not None and local_bounds_wh is not None:
        require_space(local_points.space, CoordSpace.LOCAL_CROP_PX, what="border test input")
        width, height = local_bounds_wh
        m = CROP_BORDER_MARGIN_PX
        lxy = local_points.xy
        touches = bool(
            np.any(lxy[:, 0] < m) or np.any(lxy[:, 0] > width - 1 - m)
            or np.any(lxy[:, 1] < m) or np.any(lxy[:, 1] > height - 1 - m)
        )
    mean_probability = None
    if probabilities is not None:
        prob = np.asarray(probabilities, dtype=np.float64)
        if prob.shape != (count,):
            raise GeometryContractError("probabilities must align with the instance points")
        mean_probability = float(prob.mean())

    centroid = xy.mean(axis=0)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    aabb = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
    base = dict(
        n_points=count,
        coord_space=points.space,
        centroid=(float(centroid[0]), float(centroid[1])),
        aabb=aabb,
        aabb_long_side=float(max(hi[0] - lo[0], hi[1] - lo[1])),
        touches_crop_border=touches,
        mean_probability=mean_probability,
    )

    centered = xy - centroid
    covariance = centered.T @ centered / count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)     # ascending
    lambda1 = float(eigenvalues[1])
    lambda2 = max(float(eigenvalues[0]), 0.0)
    if lambda1 <= COINCIDENT_EIGEN_FLOOR:
        return InstanceGeometry(status=InstanceStatus.COINCIDENT, eig_ratio=None, **base)

    ratio = lambda2 / lambda1
    major = eigenvectors[:, 1]
    angle = math.atan2(float(major[1]), float(major[0])) % math.pi
    if angle >= math.pi:
        angle = 0.0
    u = np.array([math.cos(angle), math.sin(angle)])
    p = np.array([-math.sin(angle), math.cos(angle)])
    t = centered @ u
    s = centered @ p
    t_lo, t_hi = (float(v) for v in np.quantile(t, [QUANTILE_LOW, QUANTILE_HIGH]))
    s_lo, s_hi = (float(v) for v in np.quantile(s, [QUANTILE_LOW, QUANTILE_HIGH]))
    length = t_hi - t_lo
    width = s_hi - s_lo
    t_mid, s_mid = (t_lo + t_hi) / 2.0, (s_lo + s_hi) / 2.0
    center = centroid + t_mid * u + s_mid * p
    e0 = centroid + t_lo * u + s_mid * p
    e1 = centroid + t_hi * u + s_mid * p

    if ratio > AXIS_AMBIGUOUS_RATIO:
        status = InstanceStatus.AXIS_AMBIGUOUS
    elif width <= ZERO_WIDTH_RELATIVE * max(1.0, length):
        status = InstanceStatus.ZERO_WIDTH
    else:
        status = InstanceStatus.OK

    return InstanceGeometry(
        status=status,
        eig_ratio=ratio,
        axis_angle=angle,
        t_range=(t_lo, t_hi),
        s_range=(s_lo, s_hi),
        center=(float(center[0]), float(center[1])),
        length=float(length),
        width=float(width),
        endpoints=((float(e0[0]), float(e0[1])), (float(e1[0]), float(e1[1]))),
        **base,
    )


# ----------------------------------------------------------------------------
# region tests (held-out coverage). Representation (c) is computed from its
# endpoints on purpose -- not by calling the box test -- so the synthetic test
# that the two regions agree is a real check.
# ----------------------------------------------------------------------------


def _check_region_input(instance: InstanceGeometry, points: Points2D) -> np.ndarray:
    require_space(points.space, instance.coord_space, what="region test points")
    return points.xy


def inside_aabb(instance: InstanceGeometry, points: Points2D) -> np.ndarray:
    xy = _check_region_input(instance, points)
    x0, y0, x1, y1 = instance.aabb
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def inside_oriented_box(instance: InstanceGeometry, points: Points2D) -> np.ndarray:
    xy = _check_region_input(instance, points)
    if instance.axis_unit is None or instance.t_range is None or instance.s_range is None:
        raise GeometryContractError("the oriented box is undefined for this instance")
    ux, uy = instance.axis_unit
    centered = xy - np.asarray(instance.centroid)
    t = centered @ np.array([ux, uy])
    s = centered @ np.array([-uy, ux])
    return (
        (t >= instance.t_range[0]) & (t <= instance.t_range[1])
        & (s >= instance.s_range[0]) & (s <= instance.s_range[1])
    )


def inside_endpoint_band(instance: InstanceGeometry, points: Points2D) -> np.ndarray:
    """Points whose axial projection lies between the endpoints AND whose
    distance from the endpoint line is at most width / 2."""
    xy = _check_region_input(instance, points)
    if instance.endpoints is None or instance.width is None or instance.axis_unit is None:
        raise GeometryContractError("the endpoint band is undefined for this instance")
    e0 = np.asarray(instance.endpoints[0])
    e1 = np.asarray(instance.endpoints[1])
    ux, uy = instance.axis_unit
    u = np.array([ux, uy])
    p = np.array([-uy, ux])
    relative = xy - e0
    t = relative @ u
    s = relative @ p
    t_end = float((e1 - e0) @ u)
    return (t >= 0.0) & (t <= t_end) & (np.abs(s) <= instance.width / 2.0)


# ----------------------------------------------------------------------------
# frame and video level
# ----------------------------------------------------------------------------


def frame_geometry(
    frame_order: int,
    local_points: Points2D,
    raw_points: Points2D,
    *,
    point_indices: np.ndarray | None = None,
    local_bounds_wh: tuple[int, int] | None = None,
    probabilities: np.ndarray | None = None,
    link_distance: float = LINK_DISTANCE_LOCAL_PX,
    min_points: int = MIN_COMPONENT_POINTS,
) -> FrameGeometry:
    """Instances of one frame's selected points (GT positives or a prediction mask)."""
    require_space(local_points.space, CoordSpace.LOCAL_CROP_PX, what="frame_geometry local points")
    if len(local_points) != len(raw_points):
        raise GeometryContractError("local and raw points of a frame differ in length")
    count = len(local_points)
    indices = np.arange(count) if point_indices is None else np.asarray(point_indices, dtype=np.int64)
    if indices.shape != (count,):
        raise GeometryContractError("point_indices must align with the frame's points")
    if count == 0:
        return FrameGeometry(frame_order=int(frame_order), present=False, status=FailureReason.EMPTY)

    components = component_index_sets(local_points.xy, link_distance=link_distance, min_points=min_points)
    if not components:
        return FrameGeometry(frame_order=int(frame_order), present=False, status=FailureReason.INSUFFICIENT_POINTS)

    def build(members: np.ndarray) -> InstanceGeometry:
        return instance_geometry(
            Points2D(raw_points.xy[members], raw_points.space),
            local_points=Points2D(local_points.xy[members], CoordSpace.LOCAL_CROP_PX),
            local_bounds_wh=local_bounds_wh,
            probabilities=None if probabilities is None else np.asarray(probabilities)[members],
        )

    instances = tuple(build(members) for members in components)
    merged = instances[0] if len(components) == 1 else build(np.concatenate(components))
    return FrameGeometry(
        frame_order=int(frame_order),
        present=True,
        status="ok",
        instances=instances,
        merged=merged,
        instance_point_indices=tuple(indices[members] for members in components),
    )


def group_by_frame(frame_order: np.ndarray, selected: np.ndarray) -> dict[int, np.ndarray]:
    """Global point indices of the selected points, grouped by frame_order."""
    frames = np.asarray(frame_order, dtype=np.int64)
    mask = np.asarray(selected, dtype=bool)
    if frames.shape != mask.shape:
        raise GeometryContractError("frame_order and the selection mask differ in shape")
    chosen = np.flatnonzero(mask)
    if chosen.size == 0:
        return {}
    order = chosen[np.argsort(frames[chosen], kind="stable")]
    boundaries = np.flatnonzero(np.diff(frames[order])) + 1
    return {int(frames[group[0]]): group for group in np.split(order, boundaries)}


def video_frame_geometries(
    frame_order: np.ndarray,
    local_points: Points2D,
    raw_points: Points2D,
    selected: np.ndarray,
    *,
    frame_universe: Sequence[int],
    local_bounds_wh: tuple[int, int] | None = None,
    probabilities: np.ndarray | None = None,
) -> dict[int, FrameGeometry]:
    """FrameGeometry for every frame in `frame_universe` (EMPTY where none selected)."""
    if len(local_points) != len(raw_points) or len(local_points) != np.asarray(frame_order).shape[0]:
        raise GeometryContractError("per-point arrays of a video differ in length")
    groups = group_by_frame(frame_order, selected)
    unknown = set(groups) - {int(f) for f in frame_universe}
    if unknown:
        raise GeometryContractError(f"{len(unknown)} selected frame(s) are outside the frame universe")
    result: dict[int, FrameGeometry] = {}
    for frame in sorted(int(f) for f in frame_universe):
        members = groups.get(frame, np.zeros(0, dtype=np.int64))
        result[frame] = frame_geometry(
            frame,
            Points2D(local_points.xy[members], CoordSpace.LOCAL_CROP_PX),
            Points2D(raw_points.xy[members], raw_points.space),
            point_indices=members,
            local_bounds_wh=local_bounds_wh,
            probabilities=None if probabilities is None else np.asarray(probabilities)[members],
        )
    return result


def frame_universe_of(frame_order: np.ndarray) -> tuple[int, ...]:
    """Frames holding at least one point of the video (the presence universe)."""
    return tuple(int(f) for f in np.unique(np.asarray(frame_order, dtype=np.int64)))


def frames_present(frames: Mapping[int, FrameGeometry]) -> dict[int, bool]:
    return {frame: geometry.present for frame, geometry in frames.items()}
