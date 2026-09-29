from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from stage5.geometry.fl_estimate import CANDIDATE_PRECEDENCE, REP_AABB, REP_AXIS, REP_BEST_FRAME, REP_OBB
from stage5.geometry.frame_geometry import (
    MIN_COMPONENT_POINTS,
    group_by_frame,
    inside_aabb,
    inside_endpoint_band,
    inside_oriented_box,
    instance_geometry,
)
from stage5.geometry.matching import match_instances
from stage5.geometry.types import (
    CoordSpace,
    FailureReason,
    FrameGeometry,
    GeometryContractError,
    InstanceGeometry,
    Points2D,
    require_space,
)

# ----------------------------------------------------------------------------
# S5-17 S17-1: metrics, denominators and the pre-fixed selection rules.
#
# Management document 4.3 / 4.4 / 5.
#
# Undefined is None with a count, never 0: a zero denominator, L = 0, or
# nothing to measure. A POSITIVE denominator with a zero numerator is a real
# 0 (e.g. precision with TP = 0 and FP > 0).
#
# Three detection measures are kept apart and never merged:
#   * frame presence over every frame of the video (GT absent and prediction
#     absent is a TN, not a missing value)
#   * instance detection after M3 matching
#   * point-level FP classes, with the denominator in the field name
#
# Held-out coverage C_ho is DESCRIPTIVE (not a cut-off). Instances are fixed
# BEFORE the A/B split; A builds the geometry and selects the (d) frame, B is
# only ever measured.
#
# Stability is the pre-fixed candidate rule with failure policy F-A (management
# decision on report 10.4): a method-side failure -- undefined unperturbed GT
# length, undefined perturbed length, or ANY of the three seeds undefined --
# counts as the WORST value for that video and cell, and stays in the
# denominator. Selection statistics use the nearest-rank quantile without
# interpolation (rank ceil(q n), 1-based), so a worst value never enters an
# interpolation and no NaN can arise; the P90 is worst exactly when the number
# of worst videos exceeds n - ceil(0.9 n). "Worst" is internal only: outputs
# carry null plus an explicit flag and counts, never Infinity. Descriptive
# success-only statistics are reported separately and never decide anything.
# All videos undefined in every cell is NOT_SELECTABLE, which is kept apart
# from FAIL.
# ----------------------------------------------------------------------------

HOLDOUT_MIN_POINTS = 10
HOLDOUT_SEEDS = (0, 1, 2)
HOLDOUT_MIN_FRAMES = 3

FP_NEAR_FACTOR = 0.1
_NEAREST_CHUNK = 2048

PERTURBATION_CELLS = (
    ("thin", 0.50),
    ("thin", 0.75),
    ("fp", 0.10),
    ("fp", 0.30),
    ("jitter", 1.0),
)
PERTURBATION_SEEDS = (0, 1, 2)
PERTURBATION_CONDITIONS = tuple(
    (kind, strength, seed) for kind, strength in PERTURBATION_CELLS for seed in PERTURBATION_SEEDS
)
assert len(PERTURBATION_CONDITIONS) == 15

STABILITY_MEDIAN_MAX = 0.05
STABILITY_P90_MAX = 0.15
PRIMARY_TIE_PERCENTAGE_POINTS = 0.01   # 1 percentage point on the relative-change P90

RULE_PASS = "pass"
RULE_FAIL = "fail"
RULE_NOT_SELECTABLE = "not_selectable"
SELECTION_QUANTILE_METHOD = "nearest_rank_no_interpolation"
_WORST = math.inf        # internal marker only; never written to any output


def cell_name(kind: str, strength: float) -> str:
    return f"{kind}_{strength:g}"


# ----------------------------------------------------------------------------
# undefined-aware helpers
# ----------------------------------------------------------------------------


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def relative_change(reference: float | None, value: float | None) -> float | None:
    """|value - reference| / reference; None when either is undefined or reference <= 0."""
    if reference is None or value is None or not reference > 0:
        return None
    return abs(float(value) - float(reference)) / float(reference)


def signed_relative_error(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or not reference > 0:
        return None
    return (float(value) - float(reference)) / float(reference)


def summarize(values: Sequence[float | None]) -> dict[str, Any]:
    defined = np.asarray([v for v in values if v is not None], dtype=np.float64)
    out: dict[str, Any] = {
        "n_total": len(values),
        "n_defined": int(defined.size),
        "n_undefined": len(values) - int(defined.size),
        "median": None,
        "q1": None,
        "q3": None,
        "p90": None,
    }
    if defined.size:
        q1, med, q3, p90 = np.quantile(defined, [0.25, 0.5, 0.75, 0.90])
        out.update(median=float(med), q1=float(q1), q3=float(q3), p90=float(p90))
    return out


# ----------------------------------------------------------------------------
# geometric errors (sign- and endpoint-swap-invariant)
# ----------------------------------------------------------------------------


def _same_space(a: InstanceGeometry, b: InstanceGeometry) -> None:
    if a.coord_space is not b.coord_space:
        raise GeometryContractError("errors are only measured between instances in the same coordinate space")


def axis_angle_error_deg(a: InstanceGeometry, b: InstanceGeometry) -> float | None:
    _same_space(a, b)
    if a.axis_angle is None or b.axis_angle is None or not (a.axis_usable and b.axis_usable):
        return None
    diff = abs(a.axis_angle - b.axis_angle) % math.pi
    return math.degrees(min(diff, math.pi - diff))


def endpoint_error(a: InstanceGeometry, b: InstanceGeometry) -> tuple[float, float] | None:
    """(mean, max) endpoint distance under the better of the two pairings."""
    _same_space(a, b)
    if a.endpoints is None or b.endpoints is None or not (a.axis_usable and b.axis_usable):
        return None
    (a0, a1), (b0, b1) = a.endpoints, b.endpoints
    straight = (math.dist(a0, b0), math.dist(a1, b1))
    swapped = (math.dist(a0, b1), math.dist(a1, b0))
    best = min(straight, swapped, key=lambda pair: (sum(pair), max(pair)))
    return (sum(best) / 2.0, max(best))


def center_error(a: InstanceGeometry, b: InstanceGeometry) -> float | None:
    _same_space(a, b)
    if a.center is None or b.center is None:
        return None
    return math.dist(a.center, b.center)


def pair_errors(gt: InstanceGeometry, pred: InstanceGeometry) -> dict[str, float | None]:
    endpoints = endpoint_error(gt, pred)
    return {
        "center_error": center_error(gt, pred),          # box centre (t_mid, s_mid), not the centroid
        "endpoint_error_mean": None if endpoints is None else endpoints[0],
        "endpoint_error_max": None if endpoints is None else endpoints[1],
        "axis_angle_error_deg": axis_angle_error_deg(gt, pred),
        "length_relative_error": signed_relative_error(
            gt.length if gt.axis_usable else None, pred.length if pred.axis_usable else None
        ),
    }


# ----------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------


def _same_universe(gt_frames: Mapping[int, FrameGeometry], pred_frames: Mapping[int, FrameGeometry]) -> list[int]:
    if set(gt_frames) != set(pred_frames):
        raise GeometryContractError("GT and prediction frame universes differ")
    return sorted(gt_frames)


def frame_presence_confusion(
    gt_frames: Mapping[int, FrameGeometry], pred_frames: Mapping[int, FrameGeometry]
) -> dict[str, Any]:
    counts = Counter()
    for frame in _same_universe(gt_frames, pred_frames):
        g, p = gt_frames[frame].present, pred_frames[frame].present
        counts["tp" if g and p else "fn" if g else "fp" if p else "tn"] += 1
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {
        "n_frames": tp + fp + fn + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
    }


def instance_detection(
    gt_frames: Mapping[int, FrameGeometry], pred_frames: Mapping[int, FrameGeometry]
) -> dict[str, Any]:
    tp = fp = fn = 0
    pairs: list[dict[str, Any]] = []
    for frame in _same_universe(gt_frames, pred_frames):
        g, p = gt_frames[frame], pred_frames[frame]
        if not (g.present or p.present):
            continue
        result = match_instances(g.instances, p.instances)
        tp, fp, fn = tp + result.tp, fp + result.fp, fn + result.fn
        for gi, pj, distance in result.pairs:
            # matching_centroid_distance is the M3 assignment distance (centroids);
            # center_error is the box-centre error. They are different quantities.
            pairs.append({"frame_order": frame, "matching_centroid_distance": distance,
                          **pair_errors(g.instances[gi], p.instances[pj])})
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "n_matched_pairs": len(pairs),
        "matched_pairs": pairs,
    }


def _gt_scale(frame: FrameGeometry) -> float | None:
    instance = frame.instance("M1")
    if instance is None:
        return None
    if instance.axis_usable and instance.length:
        return float(instance.length)
    return float(instance.aabb_long_side) if instance.aabb_long_side > 0 else None


def _nearest_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], _NEAREST_CHUNK):
        chunk = points[start:start + _NEAREST_CHUNK]
        delta = chunk[:, None, :] - targets[None, :, :]
        out[start:start + _NEAREST_CHUNK] = np.sqrt(np.min(np.einsum("ijk,ijk->ij", delta, delta), axis=1))
    return out


def fp_spatial_classes(
    *,
    pred_mask: np.ndarray,
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    frame_order: np.ndarray,
    raw_points: Points2D,
    gt_frames: Mapping[int, FrameGeometry],
    near_factor: float = FP_NEAR_FACTOR,
) -> dict[str, Any]:
    """Point-level classes of predicted positives. Distance classes partition N_FP.

    "Far" and "GT-absent frame" are distances, not anatomy: they are never
    reported as a different structure.
    """
    require_space(raw_points.space, CoordSpace.RAW_FRAME_PX, what="FP classification points")
    pred = np.asarray(pred_mask, dtype=bool)
    label = np.asarray(point_label)
    valid = np.asarray(valid_mask, dtype=bool)
    frames = np.asarray(frame_order, dtype=np.int64)
    if not (pred.shape == label.shape == valid.shape == frames.shape == (len(raw_points),)):
        raise GeometryContractError("FP classification arrays do not align")

    ignore = (label == -1) | ~valid
    fp_mask = pred & valid & (label == 0)
    gt_positive = valid & (label == 1)
    gt_by_frame = group_by_frame(frames, gt_positive)
    counts = Counter()
    for frame, members in group_by_frame(frames, fp_mask).items():
        targets = gt_by_frame.get(frame)
        if targets is None:
            counts["gt_absent_frame"] += members.size
            continue
        geometry = gt_frames.get(frame)
        scale = None if geometry is None else _gt_scale(geometry)
        if scale is None:
            counts["gt_scale_undefined"] += members.size
            continue
        near = _nearest_distance(raw_points.xy[members], raw_points.xy[targets]) <= near_factor * scale
        counts["gt_near"] += int(near.sum())
        counts["gt_frame_far"] += int((~near).sum())

    n_pred_all = int(pred.sum())
    n_pred_ignore = int((pred & ignore).sum())
    n_pred_valid = n_pred_all - n_pred_ignore
    n_fp = int(fp_mask.sum())
    classes = ("gt_near", "gt_frame_far", "gt_absent_frame", "gt_scale_undefined")
    if sum(counts[c] for c in classes) != n_fp:
        raise GeometryContractError("FP distance classes do not partition N_FP")
    out: dict[str, Any] = {
        "n_pred_all": n_pred_all,
        "n_pred_ignore": n_pred_ignore,
        "n_pred_valid": n_pred_valid,
        "n_fp": n_fp,
        "ignore_share_of_pred_all": safe_ratio(n_pred_ignore, n_pred_all),
    }
    for c in classes:
        out[f"n_{c}"] = counts[c]
        out[f"{c}_share_of_fp"] = safe_ratio(counts[c], n_fp)
        out[f"{c}_share_of_pred_valid"] = safe_ratio(counts[c], n_pred_valid)
    return out


# ----------------------------------------------------------------------------
# held-out coverage (descriptive) and relative inflation
# ----------------------------------------------------------------------------


def holdout_split(n: int, *, seed: int, frame_order: int) -> tuple[np.ndarray, np.ndarray]:
    """A = first ceil(n/2), B = the rest, of a permutation seeded by (seed, frame)."""
    rng = np.random.default_rng([int(seed), int(frame_order)])
    permutation = rng.permutation(n)
    cut = int(math.ceil(n / 2))
    return np.sort(permutation[:cut]), np.sort(permutation[cut:])


@dataclass(frozen=True)
class FrameHoldout:
    frame_order: int
    n_points: int
    n_a: int
    touches_crop_border: bool
    coverage: Mapping[str, float | None]      # a_aabb / b_obb / c_axis
    reasons: Mapping[str, str | None]


def frame_holdout(
    frame: FrameGeometry,
    raw_points: Points2D,
    *,
    seed: int,
    local_points: Points2D | None = None,
    local_bounds_wh: tuple[int, int] | None = None,
) -> FrameHoldout | None:
    """Held-out coverage of one GT frame's M1 instance. None if the frame has no GT."""
    if not frame.present:
        return None
    reps = (REP_AABB, REP_OBB, REP_AXIS)
    members = frame.instance_point_indices[0]
    n = int(members.size)
    if n < HOLDOUT_MIN_POINTS:
        reason = FailureReason.HOLDOUT_TOO_SMALL
        return FrameHoldout(frame.frame_order, n, 0, False, {r: None for r in reps}, {r: reason for r in reps})
    a_local, b_local = holdout_split(n, seed=seed, frame_order=frame.frame_order)
    a_idx, b_idx = members[a_local], members[b_local]
    a_instance = instance_geometry(
        Points2D(raw_points.xy[a_idx], raw_points.space),
        local_points=None if local_points is None else Points2D(local_points.xy[a_idx], CoordSpace.LOCAL_CROP_PX),
        local_bounds_wh=local_bounds_wh,
    )
    b = Points2D(raw_points.xy[b_idx], raw_points.space)
    coverage: dict[str, float | None] = {}
    reasons: dict[str, str | None] = {}
    # Coverage is area-based: a zero-area AABB (e.g. an axis-parallel segment)
    # is undefined here, although its long side remains a valid (a) length.
    x0, y0, x1, y1 = a_instance.aabb
    if x1 > x0 and y1 > y0:
        coverage[REP_AABB], reasons[REP_AABB] = float(inside_aabb(a_instance, b).mean()), None
    else:
        coverage[REP_AABB], reasons[REP_AABB] = None, FailureReason.ZERO_AREA
    for rep, test in ((REP_OBB, inside_oriented_box), (REP_AXIS, inside_endpoint_band)):
        if a_instance.axis_usable:
            coverage[rep], reasons[rep] = float(test(a_instance, b).mean()), None
        else:
            coverage[rep], reasons[rep] = None, a_instance.status
    return FrameHoldout(frame.frame_order, n, int(a_idx.size), a_instance.touches_crop_border, coverage, reasons)


def video_holdout(
    gt_frames: Mapping[int, FrameGeometry],
    raw_points: Points2D,
    *,
    local_points: Points2D | None = None,
    local_bounds_wh: tuple[int, int] | None = None,
    seeds: Sequence[int] = HOLDOUT_SEEDS,
) -> dict[str, Any]:
    """Video-level held-out coverage on the common frame set.

    Per seed: frames where (a)-(c) are all defined form the common set; with
    at least 3 of them, (a)-(c) take the median over the set and (d) takes the
    (c) coverage of the common frame with the most A points (ties: smallest
    frame; border frames not selectable). The video value is the median of the
    per-seed values, defined only when every seed is defined -- a seed is never
    dropped to make a video count (the policy for partial seeds is proposed in
    the S17-3 report and fixed before real data).
    """
    reps_frame = (REP_AABB, REP_OBB, REP_AXIS)
    reps_all = reps_frame + (REP_BEST_FRAME,)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        holdouts = [
            h for h in (
                frame_holdout(gt_frames[f], raw_points, seed=seed, local_points=local_points,
                              local_bounds_wh=local_bounds_wh)
                for f in sorted(gt_frames)
            ) if h is not None
        ]
        common = [h for h in holdouts if all(h.coverage[r] is not None for r in reps_frame)]
        own = {r: [h.coverage[r] for h in holdouts if h.coverage[r] is not None] for r in reps_frame}
        record: dict[str, Any] = {
            "seed": int(seed),
            "n_gt_frames": len(holdouts),
            "n_common_frames": len(common),
            "reasons": dict(Counter(reason for h in holdouts for reason in h.reasons.values() if reason)),
            "own_frame_median": {r: (float(np.median(v)) if v else None) for r, v in own.items()},
            "own_frame_count": {r: len(v) for r, v in own.items()},
        }
        if len(common) < HOLDOUT_MIN_FRAMES:
            record["value"] = {r: None for r in reps_all}
            record["undefined_reason"] = FailureReason.HOLDOUT_TOO_FEW_FRAMES
        else:
            values = {r: float(np.median([h.coverage[r] for h in common])) for r in reps_frame}
            selectable = [h for h in common if not h.touches_crop_border]
            if selectable:
                chosen = max(selectable, key=lambda h: (h.n_a, -h.frame_order))
                values[REP_BEST_FRAME] = chosen.coverage[REP_AXIS]
                record["d_frame"] = chosen.frame_order
            else:
                values[REP_BEST_FRAME] = None
            record["value"] = values
            record["undefined_reason"] = None
        per_seed.append(record)

    video_value: dict[str, float | None] = {}
    for rep in reps_all:
        seed_values = [s["value"][rep] for s in per_seed]
        video_value[rep] = float(np.median(seed_values)) if all(v is not None for v in seed_values) else None
    return {"per_seed": per_seed, "value": video_value}


def convex_hull_area(xy: np.ndarray) -> float | None:
    """Area of the convex hull (monotone chain). None when fewer than 3 non-collinear points."""
    points = sorted(set(map(tuple, np.asarray(xy, dtype=np.float64).tolist())))
    if len(points) < 3:
        return None

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None
    area = 0.0
    for i in range(len(hull)):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % len(hull)]
        area += x0 * y1 - x1 * y0
    area = abs(area) / 2.0
    return area if area > 0 else None


def relative_inflation(instance: InstanceGeometry, points: Points2D, representation: str) -> float | None:
    """Box area / convex-hull area of the same instance's points (both raw px^2). Descriptive only."""
    require_space(points.space, instance.coord_space, what="inflation points")
    hull = convex_hull_area(points.xy)
    if hull is None:
        return None
    if representation == REP_AABB:
        x0, y0, x1, y1 = instance.aabb
        area = (x1 - x0) * (y1 - y0)
    elif representation == REP_OBB:
        if not instance.axis_usable:
            return None
        area = float(instance.length) * float(instance.width)  # type: ignore[arg-type]
    else:
        raise GeometryContractError("relative inflation is defined for the box representations (a) and (b)")
    return area / hull if area > 0 else None


# ----------------------------------------------------------------------------
# perturbations (15 conditions)
# ----------------------------------------------------------------------------

_CELL_CODE = {cell: index for index, cell in enumerate(PERTURBATION_CELLS)}


def _rng(kind: str, strength: float, seed: int, frame: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), _CELL_CODE[(kind, strength)], int(frame)])


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def perturb_gt_mask(
    kind: str,
    strength: float,
    seed: int,
    *,
    frame_order: np.ndarray,
    gt_mask: np.ndarray,
    background_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Thinning or FP injection on the GT-positive mask.

    FP candidates are VALID BACKGROUND points of the same frame
    (`valid_mask & point_label == 0`), so GT positives and ignore points are
    never drawn and no point is added twice. A shortfall is recorded, not hidden.
    """
    if (kind, strength) not in _CELL_CODE or kind == "jitter":
        raise GeometryContractError(f"unknown mask perturbation {(kind, strength)!r}")
    gt = np.asarray(gt_mask, dtype=bool)
    background = np.asarray(background_mask, dtype=bool)
    if np.any(gt & background):
        raise GeometryContractError("the background mask overlaps GT positives")
    out = gt.copy()
    stats = Counter()
    background_by_frame = group_by_frame(frame_order, background)
    for frame, members in group_by_frame(frame_order, gt).items():
        rng = _rng(kind, strength, seed, frame)
        n = int(members.size)
        if kind == "thin":
            keep = max(int(math.ceil(strength * n)), min(n, MIN_COMPONENT_POINTS))
            kept = rng.choice(members, size=keep, replace=False)
            out[members] = False
            out[kept] = True
            stats["removed"] += n - keep
        else:
            requested = _round_half_up(strength * n)
            candidates = background_by_frame.get(frame, np.zeros(0, dtype=np.int64))
            added = min(requested, int(candidates.size))
            if added:
                out[rng.choice(candidates, size=added, replace=False)] = True
            stats["requested"] += requested
            stats["added"] += added
            stats["shortfall_frames"] += int(added < requested)
    stats["frames"] = len(group_by_frame(frame_order, gt))
    return out, dict(stats)


def perturb_local_jitter(local_points: Points2D, gt_mask: np.ndarray, seed: int, *, frame_order: np.ndarray,
                         sigma: float = 1.0) -> Points2D:
    """Gaussian jitter (sigma local px) on GT-positive coordinates. Not clipped; the
    range check applies to the original data only (call local_to_raw with check_range=False)."""
    require_space(local_points.space, CoordSpace.LOCAL_CROP_PX, what="jitter input")
    xy = local_points.xy.copy()
    for frame, members in group_by_frame(frame_order, gt_mask).items():
        rng = _rng("jitter", sigma, seed, frame)
        xy[members] += rng.normal(0.0, sigma, size=(members.size, 2))
    return Points2D(xy, CoordSpace.LOCAL_CROP_PX)


# ----------------------------------------------------------------------------
# stability and the pre-fixed candidate / primary rules
# ----------------------------------------------------------------------------


def nearest_rank(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile: the value at 1-based rank ceil(q n). No interpolation."""
    n = len(sorted_values)
    if n == 0:
        raise GeometryContractError("nearest_rank of an empty sequence")
    return float(sorted_values[max(1, math.ceil(q * n)) - 1])


def _selection_value(value: float) -> tuple[float | None, bool]:
    return (None, True) if value == _WORST else (value, False)


def stability_cell_summary(per_video_seed_changes: Mapping[str, Sequence[float | None]]) -> dict[str, Any]:
    """One perturbation cell under F-A.

    Per video: the median of the seeds when EVERY seed is defined (three
    values, so the median is the middle one), otherwise WORST. Returns the
    selection statistics (worst included, nearest rank) and, separately, the
    descriptive success-only summary.
    """
    selection: list[float] = []
    descriptive: list[float | None] = []
    seed_incomplete = all_undefined = 0
    for changes in per_video_seed_changes.values():
        defined = [c for c in changes if c is not None]
        if changes and len(defined) == len(changes):
            value = float(np.median(defined))
            selection.append(value)
            descriptive.append(value)
        else:
            selection.append(_WORST)
            descriptive.append(None)
            if defined:
                seed_incomplete += 1
            else:
                all_undefined += 1
    ordered = sorted(selection)
    stats: dict[str, Any] = {
        "method": SELECTION_QUANTILE_METHOD,
        "n_videos": len(selection),
        "n_worst": seed_incomplete + all_undefined,
        "n_seed_incomplete": seed_incomplete,
        "n_all_seeds_undefined": all_undefined,
        "median": None, "median_is_worst": None, "p90": None, "p90_is_worst": None,
    }
    if ordered:
        stats["median"], stats["median_is_worst"] = _selection_value(nearest_rank(ordered, 0.5))
        stats["p90"], stats["p90_is_worst"] = _selection_value(nearest_rank(ordered, 0.9))
    return {
        "n_defined": len(selection) - stats["n_worst"],
        "selection": stats,
        "descriptive_success_only": summarize(descriptive),
    }


def candidate_rule(cell_summaries: Mapping[str, Mapping[str, Any]], *, baseline_failures: int) -> dict[str, Any]:
    """Pre-fixed stability rule under F-A: every cell's selection median <= 5 % and P90 <= 15 %.

    A worst median or P90 fails the cell. All videos undefined in every cell is
    NOT_SELECTABLE (kept apart from FAIL). `baseline_failures` (videos whose
    unperturbed GT length is undefined) is reported; those videos are already
    worst in every cell.
    """
    if set(cell_summaries) != {cell_name(k, s) for k, s in PERTURBATION_CELLS}:
        raise GeometryContractError("candidate_rule needs all five perturbation cells")
    if all(s["n_defined"] == 0 for s in cell_summaries.values()):
        return {"status": RULE_NOT_SELECTABLE, "baseline_failures": baseline_failures, "max_p90": None}
    failing: list[str] = []
    for name, summary in cell_summaries.items():
        sel = summary["selection"]
        if sel["median_is_worst"] or sel["p90_is_worst"] \
                or sel["median"] > STABILITY_MEDIAN_MAX or sel["p90"] > STABILITY_P90_MAX:
            failing.append(name)
    finite = [s["selection"]["p90"] for s in cell_summaries.values() if not s["selection"]["p90_is_worst"]]
    all_finite = len(finite) == len(cell_summaries)
    return {
        "status": RULE_FAIL if failing else RULE_PASS,
        "failing_cells": sorted(failing),
        "max_p90": max(finite) if all_finite else None,
        "max_p90_is_worst": not all_finite,
        "baseline_failures": baseline_failures,
        "worst_videos_by_cell": {n: s["selection"]["n_worst"] for n, s in cell_summaries.items()},
    }


def select_primary(max_p90_by_rep: Mapping[str, float], *, precedence: Sequence[str] = CANDIDATE_PRECEDENCE
                   ) -> dict[str, Any]:
    """Primary = smallest max-P90; P90s within 1 percentage point are ties, broken by precedence.

    Secondaries: the remaining candidates by (max-P90, precedence), at most two.
    """
    if not max_p90_by_rep:
        return {"primary": None, "secondary": (), "tied": ()}
    unknown = set(max_p90_by_rep) - set(precedence)
    if unknown:
        raise GeometryContractError(f"representations outside the precedence list: {sorted(unknown)}")
    rank = {rep: i for i, rep in enumerate(precedence)}
    best = min(max_p90_by_rep.values())
    tied = sorted((r for r, v in max_p90_by_rep.items() if v - best <= PRIMARY_TIE_PERCENTAGE_POINTS + 1e-12),
                  key=lambda r: rank[r])
    primary = tied[0]
    rest = sorted((r for r in max_p90_by_rep if r != primary), key=lambda r: (max_p90_by_rep[r], rank[r]))
    return {"primary": primary, "secondary": tuple(rest[:2]), "tied": tuple(tied)}


# ----------------------------------------------------------------------------
# denominators: input-side unevaluable vs method-side failure
# ----------------------------------------------------------------------------


@dataclass
class DenominatorBreakdown:
    """total = input_unevaluable + evaluable; evaluable = success + method failures."""

    total: int = 0
    input_unevaluable: Counter = field(default_factory=Counter)
    method_failure: Counter = field(default_factory=Counter)
    success: int = 0

    def add_input_unevaluable(self, reason: str) -> None:
        self.total += 1
        self.input_unevaluable[reason] += 1

    def add_result(self, failure: str | None) -> None:
        self.total += 1
        if failure is None:
            self.success += 1
        else:
            self.method_failure[failure] += 1

    @property
    def evaluable(self) -> int:
        return self.total - sum(self.input_unevaluable.values())

    def as_record(self) -> dict[str, Any]:
        failures = sum(self.method_failure.values())
        return {
            "total": self.total,
            "input_unevaluable": sum(self.input_unevaluable.values()),
            "input_unevaluable_by_reason": dict(sorted(self.input_unevaluable.items())),
            "evaluable": self.evaluable,
            "success": self.success,
            "method_failure": failures,
            "method_failure_by_reason": dict(sorted(self.method_failure.items())),
            "method_failure_rate_of_evaluable": safe_ratio(failures, self.evaluable),
        }
