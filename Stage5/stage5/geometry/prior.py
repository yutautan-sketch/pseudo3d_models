from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from stage5.geometry.fl_estimate import (
    AGG_BEST_FRAME,
    AGG_Q90,
    REP_AABB,
    REP_AXIS,
    REP_BEST_FRAME,
    REP_OBB,
    VIDEO_QUANTILE,
)
from stage5.geometry.frame_geometry import instance_geometry
from stage5.geometry.types import (
    CoordSpace,
    FailureReason,
    FrameGeometry,
    GeometryContractError,
    InstanceGeometry,
    InstanceStatus,
    Points2D,
    VideoFLEstimate,
    require_space,
)

# ----------------------------------------------------------------------------
# S5-17 S17-1: prior-only baseline (H17-4).
#
# Management document 4.4: built from train_core GT only, in ORIGINAL-FRAME
# NORMALISED coordinates (x / raw_width, y / raw_height). Local crop
# coordinates are not used because the crop position differs per video
# (report 1.3 F9).
#
# ALL geometric quantities of the prior live in the normalised space, and one
# mapping is used in both directions (management review of 5.3, 2026-09-28;
# the first version normalised only the centre and kept length/width/angle in
# raw pixels):
#
#   build    per frame, the M1 GT points are mapped to (x/W, y/H) and the
#            oriented box is re-fitted THERE (PCA, 2..98 % ranges). A box
#            fitted in pixels and then divided by (W, H) would be a
#            parallelogram, not a box, whenever W != H.
#   video    centre = mean of box centres, angle = doubled-angle mean,
#            length = q90 of the per-frame lengths, width = median -- all
#            normalised
#   prior    mean over videos (angle: doubled-angle mean)
#   inverse  the normalised box (centre, endpoints, corners) is multiplied by
#            the TARGET video's (W, H). Length, angle, width and AABB in raw
#            pixels are derived from the mapped points, so a 45 degree
#            normalised axis becomes atan(H / W) in a W x H frame.
#
# A normalised length is not isotropic (x and y have different
# denominators), and videos of different resolution and field of view are
# averaged in that space. This is recorded as a limit, not corrected. In-sample
# use excludes the target video by IDENTITY (never by path); same-exam sibling
# files stay in, which makes in-sample values optimistic.
# ----------------------------------------------------------------------------

PRESENCE_BINS = 10
PRESENCE_THRESHOLD = 0.5
_RESULTANT_FLOOR = 1e-9


def _doubled_angle_mean(angles: Sequence[float]) -> float | None:
    if not angles:
        return None
    c = float(np.mean([math.cos(2.0 * a) for a in angles]))
    s = float(np.mean([math.sin(2.0 * a) for a in angles]))
    if math.hypot(c, s) <= _RESULTANT_FLOOR:
        return None
    return (math.atan2(s, c) / 2.0) % math.pi


def presence_bin(rank: int, count: int) -> int:
    t = 0.0 if count <= 1 else rank / (count - 1)
    return min(int(t * PRESENCE_BINS), PRESENCE_BINS - 1)


def _check_wh(raw_wh: tuple[int, int]) -> np.ndarray:
    w, h = raw_wh
    if not (w > 1 and h > 1):
        raise GeometryContractError(f"raw frame size must exceed 1 x 1, got {raw_wh}")
    return np.array([float(w), float(h)])


@dataclass(frozen=True)
class VideoPriorSummary:
    """One train_core video, every geometric field in RAW_FRAME_NORM."""

    identity: str
    center_norm: tuple[float, float] | None
    axis_angle_norm: float | None
    length_norm: float | None
    width_norm: float | None
    presence_rate_by_bin: tuple[float | None, ...]


def summarize_video_for_prior(
    identity: str,
    gt_frames: Mapping[int, FrameGeometry],
    raw_points: Points2D,
    *,
    raw_wh: tuple[int, int],
) -> VideoPriorSummary:
    """`raw_points` must be the array that `gt_frames`' instance indices point into."""
    require_space(raw_points.space, CoordSpace.RAW_FRAME_PX, what="prior input points")
    scale = _check_wh(raw_wh)
    frames = sorted(gt_frames)
    centers: list[tuple[float, float]] = []
    angles: list[float] = []
    lengths: list[float] = []
    widths: list[float] = []
    hits = [0] * PRESENCE_BINS
    totals = [0] * PRESENCE_BINS
    for rank, frame in enumerate(frames):
        geometry = gt_frames[frame]
        b = presence_bin(rank, len(frames))
        totals[b] += 1
        hits[b] += int(geometry.present)
        if not geometry.present:
            continue
        members = geometry.instance_point_indices[0]
        normalised = instance_geometry(Points2D(raw_points.xy[members] / scale, CoordSpace.RAW_FRAME_NORM))
        if not normalised.axis_usable:
            continue
        centers.append(normalised.center)  # type: ignore[arg-type]
        angles.append(float(normalised.axis_angle))  # type: ignore[arg-type]
        lengths.append(float(normalised.length))  # type: ignore[arg-type]
        widths.append(float(normalised.width))  # type: ignore[arg-type]
    return VideoPriorSummary(
        identity=identity,
        center_norm=tuple(np.mean(np.asarray(centers), axis=0).tolist()) if centers else None,  # type: ignore[arg-type]
        axis_angle_norm=_doubled_angle_mean(angles),
        length_norm=float(np.quantile(lengths, VIDEO_QUANTILE)) if lengths else None,
        width_norm=float(np.median(widths)) if widths else None,
        presence_rate_by_bin=tuple((hits[b] / totals[b]) if totals[b] else None for b in range(PRESENCE_BINS)),
    )


@dataclass(frozen=True)
class Prior:
    """The train_core prior, every geometric field in RAW_FRAME_NORM."""

    n_videos: int
    identities: frozenset[str]
    center_norm: tuple[float, float] | None
    axis_angle_norm: float | None
    length_norm: float | None
    width_norm: float | None
    presence_rate_by_bin: tuple[float | None, ...]

    @property
    def geometry_defined(self) -> bool:
        return None not in (self.center_norm, self.axis_angle_norm, self.length_norm, self.width_norm)


def build_prior(summaries: Sequence[VideoPriorSummary], *, exclude_identity: str | None = None) -> Prior:
    identities = [s.identity for s in summaries]
    if len(set(identities)) != len(identities):
        raise GeometryContractError("prior summaries contain the same identity twice")
    if exclude_identity is not None and exclude_identity not in identities:
        raise GeometryContractError("the identity to exclude is not part of the prior")
    kept = [s for s in summaries if s.identity != exclude_identity]

    def mean_of(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    centers = [s.center_norm for s in kept if s.center_norm is not None]
    presence: list[float | None] = []
    for b in range(PRESENCE_BINS):
        rates = [s.presence_rate_by_bin[b] for s in kept if s.presence_rate_by_bin[b] is not None]
        presence.append(mean_of(rates))  # type: ignore[arg-type]
    return Prior(
        n_videos=len(kept),
        identities=frozenset(s.identity for s in kept),
        center_norm=(tuple(np.mean(np.asarray(centers), axis=0).tolist()) if centers else None),  # type: ignore[arg-type]
        axis_angle_norm=_doubled_angle_mean([s.axis_angle_norm for s in kept if s.axis_angle_norm is not None]),
        length_norm=mean_of([s.length_norm for s in kept if s.length_norm is not None]),
        width_norm=mean_of([s.width_norm for s in kept if s.width_norm is not None]),
        presence_rate_by_bin=tuple(presence),
    )


def prior_instance(prior: Prior, *, raw_wh: tuple[int, int]) -> InstanceGeometry:
    """The normalised prior box mapped into a W x H frame (RAW_FRAME_PX).

    The normalised box maps to a parallelogram in pixels. Endpoints and centre
    map exactly; length and angle come from the mapped endpoints, the AABB
    from the mapped corners, and the width is the mapped minor half-vector's
    extent perpendicular to the mapped axis. t/s ranges describe that
    perpendicular extent; the prior is not used for region (coverage) tests.
    """
    if not prior.geometry_defined:
        raise GeometryContractError("the prior geometry is undefined (no usable train_core video)")
    scale = _check_wh(raw_wh)
    angle_n = float(prior.axis_angle_norm)  # type: ignore[arg-type]
    u_n = np.array([math.cos(angle_n), math.sin(angle_n)])
    p_n = np.array([-math.sin(angle_n), math.cos(angle_n)])
    c_n = np.asarray(prior.center_norm, dtype=np.float64)
    half_major = u_n * float(prior.length_norm) / 2.0    # type: ignore[arg-type]
    half_minor = p_n * float(prior.width_norm) / 2.0     # type: ignore[arg-type]

    c = c_n * scale
    e0, e1 = (c_n - half_major) * scale, (c_n + half_major) * scale
    corners = np.stack([(c_n + a * half_major + b * half_minor) * scale for a in (-1, 1) for b in (-1, 1)])

    axis = e1 - e0
    length = float(np.linalg.norm(axis))
    angle = math.atan2(float(axis[1]), float(axis[0])) % math.pi
    if angle >= math.pi:
        angle = 0.0
    p = np.array([-math.sin(angle), math.cos(angle)])
    width = 2.0 * abs(float((half_minor * scale) @ p))
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return InstanceGeometry(
        n_points=0,
        status=InstanceStatus.PRIOR,
        coord_space=CoordSpace.RAW_FRAME_PX,
        centroid=(float(c[0]), float(c[1])),
        aabb=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
        aabb_long_side=float(max(hi[0] - lo[0], hi[1] - lo[1])),
        eig_ratio=None,
        axis_angle=angle,
        t_range=(-length / 2, length / 2),
        s_range=(-width / 2, width / 2),
        center=(float(c[0]), float(c[1])),
        length=length,
        width=width,
        endpoints=((float(e0[0]), float(e0[1])), (float(e1[0]), float(e1[1]))),
    )


def prior_frames(prior: Prior, *, frame_universe: Sequence[int], raw_wh: tuple[int, int]) -> dict[int, FrameGeometry]:
    """Prior-only frame geometries: present where the bin's train_core presence rate exceeds 0.5."""
    frames = sorted(int(f) for f in frame_universe)
    instance = prior_instance(prior, raw_wh=raw_wh)
    result: dict[int, FrameGeometry] = {}
    for rank, frame in enumerate(frames):
        rate = prior.presence_rate_by_bin[presence_bin(rank, len(frames))]
        if rate is not None and rate > PRESENCE_THRESHOLD:
            result[frame] = FrameGeometry(frame_order=frame, present=True, status="ok",
                                          instances=(instance,), merged=instance)
        else:
            result[frame] = FrameGeometry(frame_order=frame, present=False, status=FailureReason.EMPTY)
    return result


def prior_estimates(prior: Prior, *, frame_universe: Sequence[int], raw_wh: tuple[int, int]) -> dict[str, VideoFLEstimate]:
    """Video-level prior-only lengths for (a)-(d), in the target frame's raw pixels.

    The geometry is the same in every present frame, so (d) takes the first
    present frame; with no present frame every representation fails.
    """
    frames = prior_frames(prior, frame_universe=frame_universe, raw_wh=raw_wh)
    present = tuple(f for f, g in frames.items() if g.present)
    instance = prior_instance(prior, raw_wh=raw_wh)
    out: dict[str, VideoFLEstimate] = {}
    for rep, value, agg, used in (
        (REP_AABB, instance.aabb_long_side, AGG_Q90, present),
        (REP_OBB, instance.length, AGG_Q90, present),
        (REP_AXIS, instance.length, AGG_Q90, present),
        (REP_BEST_FRAME, instance.length, AGG_BEST_FRAME, present[:1]),
    ):
        out[rep] = VideoFLEstimate(
            representation=rep,
            aggregation=agg,
            instance_rule="prior",
            coord_space=CoordSpace.RAW_FRAME_PX,
            value=value if present else None,
            frames_used=used,
            failure=None if present else FailureReason.EMPTY,
            selection_basis="prior",
        )
    return out
