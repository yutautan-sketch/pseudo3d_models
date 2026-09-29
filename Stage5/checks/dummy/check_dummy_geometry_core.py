from __future__ import annotations

import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from checks.real_h5.check_stage5_gt_component_count import component_sizes  # noqa: E402
from stage5.geometry import fl_estimate as fl  # noqa: E402
from stage5.geometry import metrics as mt  # noqa: E402
from stage5.geometry import postprocess as pp  # noqa: E402
from stage5.geometry import prior as pr  # noqa: E402
from stage5.geometry.frame_geometry import (  # noqa: E402
    connected_components,
    frame_geometry,
    inside_endpoint_band,
    inside_oriented_box,
    instance_geometry,
    video_frame_geometries,
)
from stage5.geometry.matching import hungarian_min_cost, match_instances  # noqa: E402
from stage5.geometry.transform import CropTransform, PixelToMM  # noqa: E402
from stage5.geometry.types import (  # noqa: E402
    CoordSpace,
    FailureReason,
    GeometryContractError,
    InputReason,
    InputUnevaluable,
    InstanceStatus,
    Points2D,
)

# ----------------------------------------------------------------------------
# S5-17 S17-1 / S17-3: deterministic synthetic coverage of the geometry core.
#
# Management document 7. Known values are written here as constants, computed
# by hand -- never recomputed from the implementation -- so a forward and an
# inverse map sharing the same mistake cannot pass on a round trip alone.
# Success paths are exercised, not only refusals. The probabilistic held-out
# coverage OBSERVATIONS on known shapes are a separate S17-3 report item and
# are not asserted here.
#
# numpy only (plus the S5-15 `component_sizes` for a parity check). No H5, no
# pins, no real data, no torch, no CUDA.
# ----------------------------------------------------------------------------

FAILURES: list[str] = []
CHECKS = 0
RAW = CoordSpace.RAW_FRAME_PX
LOCAL = CoordSpace.LOCAL_CROP_PX


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def raises(exc: type[BaseException], fn, *args, **kwargs) -> BaseException | None:
    try:
        fn(*args, **kwargs)
    except exc as error:
        return error
    return None


def close(a, b, tol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0.0))


def rotated_grid(length: float, width: float, angle_deg: float, center=(300.0, 200.0), nt=101, ns=21) -> np.ndarray:
    t, s = np.meshgrid(np.linspace(-length / 2, length / 2, nt), np.linspace(-width / 2, width / 2, ns))
    a = math.radians(angle_deg)
    u = np.array([math.cos(a), math.sin(a)])
    p = np.array([-math.sin(a), math.cos(a)])
    return np.asarray(center) + t.reshape(-1, 1) * u + s.reshape(-1, 1) * p


# ----------------------------------------------------------------------------


def test_transform_known_values() -> None:
    print("[1] crop inverse: hand-computed expected values, Stage 4 forward formula, refusals")
    offset = CropTransform.from_attrs(
        {"local_preprocess_effective": b"offset_crop", "raw_width": np.int64(800), "raw_height": 600,
         "local_crop_left": 272, "local_crop_top": 172, "local_resize_scale": 1.0},
        local_shape_hw=(256, 256),
    )
    got = offset.local_to_raw(Points2D([[0.0, 0.0], [255.5, 10.0]], LOCAL)).xy
    check(close(got, [[272.0, 172.0], [527.5, 182.0]]), "offset_crop local->raw expected values")

    s = 448.0 / 480.0
    shorter = CropTransform.from_attrs(
        {"local_preprocess_effective": "resize_shorter_then_offset_crop", "raw_width": 640, "raw_height": 480,
         "local_crop_left": 170, "local_crop_top": 0, "local_resize_scale": s},
        local_shape_hw=(256, 256),
    )
    got = shorter.local_to_raw(Points2D([[0.0, 0.0], [56.0, 112.0]], LOCAL)).xy
    check(close(got, [[182.14285714285714, 0.0], [242.14285714285714, 120.0]]),
          "resize_shorter local->raw expected values")
    # Stage 4's own forward formula (annotate_pseudo3d_point_cloud.py): local = raw * s - left.
    stage4_forward = got * s - np.array([170.0, 0.0])
    check(close(stage4_forward, [[0.0, 0.0], [56.0, 112.0]]), "inverse agrees with the Stage 4 forward convention")

    resize = CropTransform.from_attrs(
        {"local_preprocess_effective": "resize", "raw_width": 640, "raw_height": 480},
        local_shape_hw=(256, 256),
    )
    got = resize.local_to_raw(Points2D([[128.0, 64.0]], LOCAL)).xy
    check(close(got, [[320.0, 120.0]]), "resize (anisotropic) local->raw expected value")

    rng = np.random.default_rng(0)
    local = Points2D(rng.uniform(0, 255.999, size=(500, 2)), LOCAL)
    for name, transform in (("offset", offset), ("shorter", shorter), ("resize", resize)):
        back = transform.raw_to_local(transform.local_to_raw(local)).xy
        check(float(np.max(np.abs(back - local.xy))) <= 1e-6, f"{name}: local->raw->local round trip <= 1e-6")

    base = {"raw_width": 640, "raw_height": 480, "local_crop_left": 0, "local_crop_top": 0}
    cases = (
        ({**base, "local_preprocess_effective": "offset_crop_fallback_resize", "local_resize_scale": float("nan")},
         InputReason.CROP_FALLBACK_RESIZE),
        ({**base, "local_preprocess_effective": "offset_crop", "local_resize_scale": float("nan")},
         InputReason.CROP_SCALE_INVALID),
        ({**base, "local_preprocess_effective": "offset_crop"}, InputReason.CROP_FIELD_MISSING),
        ({**base, "local_preprocess_effective": "something_else", "local_resize_scale": 1.0},
         InputReason.CROP_UNKNOWN_MODE),
        ({**base, "local_preprocess_effective": "offset_crop", "local_resize_scale": 1.0, "raw_width": float("nan")},
         InputReason.CROP_DIMENSION_INVALID),
    )
    for attrs, reason in cases:
        error = raises(InputUnevaluable, CropTransform.from_attrs, attrs, local_shape_hw=(256, 256))
        check(error is not None and error.reason == reason, f"input-unevaluable reason {reason}")

    check(raises(GeometryContractError, offset.local_to_raw, Points2D([[256.0, 0.0]], LOCAL)) is not None,
          "pixel_xy on the crop edge (x == width) stops")
    check(raises(GeometryContractError, offset.local_to_raw, Points2D([[1.0, 1.0]], RAW)) is not None,
          "a raw point handed to local_to_raw stops (no implicit space mixing)")


def test_pixel_to_mm() -> None:
    print("[2] PixelToMM: anisotropic expected values, round trip, refusals")
    scale = PixelToMM(source_sha256="ab" * 32, mm_per_px_x=0.1, mm_per_px_y=0.13)
    mm = scale.raw_px_to_mm(Points2D([[100.0, 200.0]], RAW))
    check(mm.space is CoordSpace.MM_XY and close(mm.xy, [[10.0, 26.0]]), "raw(100,200) -> (10, 26) mm")
    check(not close(mm.xy, [[13.0, 20.0]], tol=1e-3), "x/y swap would be detected")
    back = scale.mm_to_raw_px(mm)
    check(back.space is RAW and float(np.max(np.abs(back.xy - [[100.0, 200.0]]) / 200.0)) <= 1e-9,
          "mm round trip relative error <= 1e-9")
    check(raises(GeometryContractError, PixelToMM, source_sha256="", mm_per_px_x=0.1, mm_per_px_y=0.1) is not None,
          "an unsourced scale is refused")
    check(raises(GeometryContractError, PixelToMM, source_sha256="x", mm_per_px_x=0.1, mm_per_px_y=None) is not None,
          "a missing y scale is refused (no default)")
    frame_scale = PixelToMM(source_sha256="x", per_frame={3: (0.1, 0.1)})
    check(raises(GeometryContractError, frame_scale.raw_px_to_mm, Points2D([[1.0, 1.0]], RAW), frame_order=4)
          is not None, "a per-frame scale is not borrowed from another frame")


def test_instance_geometry() -> None:
    print("[3] instance geometry: known rectangle, line, degenerate cases, sign/swap invariance")
    grid = rotated_grid(100.0, 10.0, 30.0)
    inst = instance_geometry(Points2D(grid, RAW))
    check(inst.status == InstanceStatus.OK, "rectangle is OK")
    check(abs(inst.length - 96.0) < 1e-6 and abs(inst.width - 10.0) < 1e-6,
          "rectangle length 96 (2..98 % of a 101-step grid), width 10 (21-step minor grid)")
    check(abs(inst.axis_angle - math.radians(30.0)) < 1e-6, "axis angle 30 deg")
    check(close(inst.center, (300.0, 200.0), 1e-6) and close(inst.centroid, (300.0, 200.0), 1e-6),
          "symmetric rectangle: centre == centroid == (300, 200)")
    u = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
    expected_e = sorted([tuple(np.array([300.0, 200.0]) - 48 * u), tuple(np.array([300.0, 200.0]) + 48 * u)])
    check(close(sorted(inst.endpoints), expected_e, 1e-6), "endpoints at centre -+ 48 along the axis")

    flipped = instance_geometry(Points2D(rotated_grid(100.0, 10.0, 210.0)[::-1], RAW))
    check(abs(flipped.axis_angle - inst.axis_angle) < 1e-9, "axis angle has no sign (30 vs 210 deg)")
    check(mt.endpoint_error(inst, flipped)[1] < 1e-6, "endpoint error invariant to direction")
    swapped = replace(inst, endpoints=(inst.endpoints[1], inst.endpoints[0]))
    check(mt.endpoint_error(inst, swapped) == (0.0, 0.0), "endpoint error invariant to endpoint swap")
    a1 = replace(inst, axis_angle=math.radians(1.0))
    a179 = replace(inst, axis_angle=math.radians(179.0))
    check(abs(mt.axis_angle_error_deg(a1, a179) - 2.0) < 1e-9, "axis error 1 vs 179 deg is 2 deg")

    t = np.linspace(-50, 50, 1001)
    line = np.stack([100 + t * math.cos(0.5), 100 + t * math.sin(0.5)], axis=1)
    line_inst = instance_geometry(Points2D(line, RAW))
    check(line_inst.status == InstanceStatus.ZERO_WIDTH and not line_inst.axis_usable, "collinear -> zero_width")
    check(abs(line_inst.length - 96.0) < 1e-6, "line length 96")
    check(fl.frame_length(line_inst, fl.REP_AABB) is not None, "(a) still usable on a sloped line")

    # I-4: an axis-parallel segment has a zero-area AABB but a valid (a) length;
    # the zero-area condition applies to area-based quantities only.
    horizontal = np.stack([np.arange(0.0, 60.0, 2.0), np.full(30, 100.0)], axis=1)     # x 0..58
    vertical = horizontal[:, ::-1]
    for name, seg in (("horizontal", horizontal), ("vertical", vertical)):
        seg_inst = instance_geometry(Points2D(seg, RAW))
        check(fl.frame_length(seg_inst, fl.REP_AABB) == 58.0, f"{name} segment: (a) length = long side 58")
        check(mt.relative_inflation(seg_inst, Points2D(seg, RAW), fl.REP_AABB) is None,
              f"{name} segment: area-based inflation undefined")
        seg_frame = frame_geometry(0, Points2D(seg, LOCAL), Points2D(seg, RAW))
        held = mt.frame_holdout(seg_frame, Points2D(seg, RAW), seed=0)
        check(held.coverage[fl.REP_AABB] is None and held.reasons[fl.REP_AABB] == FailureReason.ZERO_AREA,
              f"{name} segment: held-out (a) coverage undefined as zero_area_aabb")

    same = instance_geometry(Points2D(np.full((6, 2), 7.0), RAW))
    check(same.status == InstanceStatus.COINCIDENT and same.eig_ratio is None and same.length is None,
          "identical points -> coincident, no axis, no length")
    check(fl.frame_length(same, fl.REP_AABB) is None, "(a) undefined when the AABB has zero extent")

    angles = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    disc = np.concatenate([np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1) for r in (5, 10, 15)])
    check(instance_geometry(Points2D(disc + 50, RAW)).status == InstanceStatus.AXIS_AMBIGUOUS,
          "isotropic disc -> axis_ambiguous")
    check(raises(GeometryContractError, instance_geometry, Points2D(grid, LOCAL)) is not None,
          "geometry refuses local-crop coordinates")


def test_region_alignment() -> None:
    print("[4] (b) box and (c) endpoint band are the same region on an asymmetric minor axis")
    rng = np.random.default_rng(1)
    t = rng.uniform(-50, 50, 3000)
    s = rng.exponential(2.0, 3000)
    a = math.radians(20.0)
    u, p = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    pts = 200 + t[:, None] * u + s[:, None] * p
    inst = instance_geometry(Points2D(pts, RAW))
    s_mid = sum(inst.s_range) / 2
    check(inst.axis_usable and abs(s_mid) > 0.1 * inst.width, "minor-axis quantile range is asymmetric")
    probe = Points2D(200 + rng.uniform(-70, 70, (20000, 1)) * u + rng.uniform(-8, 14, (20000, 1)) * p, RAW)
    box, band = inside_oriented_box(inst, probe), inside_endpoint_band(inst, probe)
    check(bool(np.array_equal(box, band)), "box and band agree point by point")
    check(0 < int(box.sum()) < 20000, "the probe set straddles the region")
    rel = probe.xy - np.asarray(inst.centroid)
    centroid_band = np.abs(rel @ p) <= inst.width / 2
    check(bool(np.any(centroid_band != box)), "a band on the centroid line would differ (the alignment matters)")


def test_components() -> None:
    print("[5] connected components: parity with the S5-15 implementation, link boundary, ordering")
    rng = np.random.default_rng(2)
    blobs = [rng.normal(c, 1.5, size=(n, 2)) for c, n in (((20, 20), 60), ((60, 20), 25), ((20, 70), 40))]
    noise = rng.uniform(0, 100, size=(15, 2))
    pts = np.concatenate(blobs + [noise])
    labels = connected_components(pts, link_distance=4.0)
    ours = sorted(np.bincount(labels).tolist(), reverse=True)
    check(ours == component_sizes(pts, link_distance=4.0), "component sizes equal component_sizes()")
    check(int(np.bincount(labels)[0]) == max(ours), "label 0 is the largest component")
    check(int(connected_components(np.array([[0.0, 0.0], [4.0, 0.0]]), link_distance=4.0).max()) == 0,
          "distance exactly 4 links")
    check(int(connected_components(np.array([[0.0, 0.0], [4.0001, 0.0]]), link_distance=4.0).max()) == 1,
          "distance above 4 does not link")


def _frame_points(center, length, width, count_t, angle=0.0):
    return rotated_grid(length, width, angle, center=center, nt=count_t, ns=5)


def _grid_length(length: float, count_t: int, ns: int = 5) -> float:
    """Expected 2..98 % extent of a count_t x ns grid, from the construction (not the implementation)."""
    t = np.repeat(np.linspace(-length / 2, length / 2, count_t), ns)
    lo, hi = np.quantile(t, [0.02, 0.98])
    return float(hi - lo)


def test_frame_and_video_estimates() -> None:
    print("[6] frames, M1/M2, q90, (d) GT / practical / oracle kept apart, border exclusion")
    two = np.concatenate([_frame_points((60, 60), 30, 4, 12), _frame_points((180, 180), 20, 4, 8)])
    geom = frame_geometry(0, Points2D(two, LOCAL), Points2D(two, RAW))
    check(geom.present and len(geom.instances) == 2, "two separated blobs -> two instances")
    check(geom.instance("M1").n_points == 60 and geom.instance("M2").n_points == 100, "M1 = largest, M2 = all")
    check(frame_geometry(1, Points2D(two[:3], LOCAL), Points2D(two[:3], RAW)).status
          == FailureReason.INSUFFICIENT_POINTS, "3 points -> insufficient")
    empty = np.zeros((0, 2))
    check(frame_geometry(2, Points2D(empty, LOCAL), Points2D(empty, RAW)).status == FailureReason.EMPTY,
          "no points -> empty")

    lengths = (40.0, 60.0, 80.0, 100.0, 120.0)
    counts = (21, 41, 31, 27, 35)            # grid columns (x5 rows); every step <= 4 px, one component
    frame_prob = (0.6, 0.3, 0.6, 0.95, 0.6)
    frames_xy, frame_order, prob = [], [], []
    for f, (length, count) in enumerate(zip(lengths, counts)):
        xy = _frame_points((128, 128), length, 6, count)
        frames_xy.append(xy)
        frame_order += [f] * len(xy)
        prob += [frame_prob[f]] * len(xy)
    xy = np.concatenate(frames_xy)
    frame_order = np.asarray(frame_order)
    universe = tuple(range(6))                                   # frame 5 exists but has no GT
    frames = video_frame_geometries(frame_order, Points2D(xy, LOCAL), Points2D(xy, RAW),
                                    np.ones(len(xy), dtype=bool), frame_universe=universe,
                                    local_bounds_wh=(256, 256), probabilities=np.asarray(prob))
    check(not frames[5].present, "frame without points is present=False in the universe")
    per_frame = [_grid_length(v, c) for v, c in zip(lengths, counts)]
    q90 = fl.estimate_quantile(frames, fl.REP_OBB, instance_rule="M1", coord_space=RAW)
    check(abs(q90.value - float(np.quantile(per_frame, 0.9))) < 1e-6, "(b) = q90 of per-frame lengths")
    check(q90.unit.value == "raw_px" and q90.failure is None, "(b) carries its unit")

    gt_d = fl.estimate_best_frame_gt(frames, instance_rule="M1", coord_space=RAW)
    check(gt_d.frames_used == (1,) and gt_d.selection_basis == fl.BASIS_GT, "(d) GT picks the frame with most points")
    pred_d = fl.estimate_best_frame_prediction(frames, instance_rule="M1", coord_space=RAW)
    # points x mean prob: f0 105*0.6=63, f1 205*0.3=61.5, f2 155*0.6=93, f3 135*0.95=128.25, f4 175*0.6=105
    check(pred_d.frames_used == (3,) and not pred_d.is_oracle and pred_d.selection_basis == fl.BASIS_PREDICTION,
          "(d) practical picks by prediction information (frame 3), not by GT (frame 1)")
    oracle = fl.estimate_best_frame_oracle(frames, frames, instance_rule="M1", coord_space=RAW)
    check(oracle.is_oracle and oracle.selection_basis == fl.BASIS_ORACLE_GT and oracle.frames_used == (1,),
          "oracle applies the GT-selected frame and is flagged")
    check(raises(GeometryContractError, fl.estimate_representation, frames, fl.REP_BEST_FRAME,
                 instance_rule="M1", coord_space=RAW, source="oracle") is not None,
          "the oracle is not reachable through the practical dispatcher")

    border = _frame_points((18, 128), 36, 4, 50)                  # x from 0: inside the 2 px margin
    bframes = {0: frame_geometry(0, Points2D(border, LOCAL), Points2D(border, RAW), local_bounds_wh=(256, 256))}
    check(bframes[0].instances[0].touches_crop_border, "instance at the crop edge is flagged")
    check(fl.estimate_best_frame_gt(bframes, instance_rule="M1", coord_space=RAW).failure
          == FailureReason.NO_USABLE_FRAME, "a border frame is not selectable for (d)")
    check(raises(GeometryContractError, fl.estimate_quantile, frames, fl.REP_OBB, instance_rule="M1",
                 coord_space=CoordSpace.MM_XY) is not None, "estimating in the wrong space stops")
    empty_frames = {0: frames[5]}
    check(fl.estimate_quantile(empty_frames, fl.REP_AXIS, instance_rule="M1", coord_space=RAW).failure
          == FailureReason.EMPTY, "no present frame -> failure, value None")

    rng = np.random.default_rng(3)
    line3 = np.stack([np.linspace(0, 10, 200), np.linspace(0, 20, 200), np.linspace(0, 30, 200)], 1)
    e = fl.estimate_pseudo3d_axis(line3 + rng.normal(0, 0.01, line3.shape))
    check(e.coord_space is CoordSpace.PSEUDO3D and not e.candidate_eligible and e.unit.value == "pseudo3d_units",
          "(e) is pseudo3d_units and never a candidate")


def test_postprocess() -> None:
    print("[7] post-processing P1 / P2 / P3")
    big = _frame_points((60, 60), 30, 4, 12)
    small = _frame_points((180, 180), 12, 4, 4)
    xy = np.concatenate([big, small, big + [1, 0], big + [150, 150]])
    frame_order = np.concatenate([np.zeros(len(big) + len(small)), np.ones(len(big)), np.full(len(big), 5)]).astype(int)
    pred = np.ones(len(xy), dtype=bool)
    local, raw = Points2D(xy, LOCAL), Points2D(xy, RAW)
    p1 = pp.p1_largest_component(frame_order, local, pred)
    check(int(p1[:len(big)].sum()) == len(big) and not p1[len(big):len(big) + len(small)].any(),
          "P1 keeps the largest component only")
    p2 = pp.p2_persistence(frame_order, local, raw, p1, distance_raw_px=10.0)
    check(bool(p2[frame_order == 0].sum() == len(big)) and bool(p2[frame_order == 1].all()),
          "P2 keeps components persistent across neighbouring frames")
    check(not p2[frame_order == 5].any(), "P2 drops an isolated frame")

    fo = np.array([0, 0, 0, 0, 1, 1, 1])
    prob = np.array([0.2, 0.9, 0.4, 0.3, 0.8, 0.7, 0.6])
    pred3 = np.array([False, True, False, False, False, False, False])
    p3 = pp.p3_top_k(fo, pred3, prob, top_k=2)
    check(p3.tolist() == [False, True, True, False, False, False, False],
          "P3: top-K in frames with a prediction, including prob < 0.5; none elsewhere")
    c = pp.PostprocessConstants.from_train_core([80.0, 100.0, 120.0], [10, 11, 12, 13])
    check(abs(c.p2_distance_raw_px - 50.0) < 1e-12 and c.p3_top_k == 12, "constants: 0.5 x median, round half up")


def test_matching() -> None:
    print("[8] Hungarian parity with brute force; gated M3 matching")
    rng = np.random.default_rng(4)
    for trial in range(20):
        cost = rng.uniform(0, 10, size=(4, 5))
        assignment = hungarian_min_cost(cost)
        ours = sum(cost[i, j] for i, j in enumerate(assignment))
        best = min(sum(cost[i, perm[i]] for i in range(4)) for perm in itertools.permutations(range(5), 4))
        check(abs(ours - best) < 1e-9, f"hungarian optimal (trial {trial})")
    gt = instance_geometry(Points2D(rotated_grid(10.0, 2.0, 0.0, center=(0, 0)), RAW))    # length 9.6 -> gate 4.8
    near = instance_geometry(Points2D(rotated_grid(10.0, 2.0, 0.0, center=(3, 0)), RAW))
    far = instance_geometry(Points2D(rotated_grid(10.0, 2.0, 0.0, center=(20, 0)), RAW))
    result = match_instances([gt], [far, near])
    check(result.pairs[0][:2] == (0, 1) and result.unmatched_pred == (0,), "nearest admissible pair matched")
    none = match_instances([gt], [far])
    check(none.tp == 0 and none.fn == 1 and none.fp == 1, "beyond the gate: unmatched, not forced")


def test_detection_and_undefined() -> None:
    print("[9] frame presence (TN), undefined vs 0, FP classes and denominators")
    blob = _frame_points((100, 100), 20, 4, 10)
    gt_frames = {
        0: frame_geometry(0, Points2D(blob, LOCAL), Points2D(blob, RAW)),
        1: frame_geometry(1, Points2D(np.zeros((0, 2)), LOCAL), Points2D(np.zeros((0, 2)), RAW)),
    }
    pred_frames = {0: gt_frames[1], 1: gt_frames[1]}
    conf = mt.frame_presence_confusion(gt_frames, pred_frames)
    check((conf["tp"], conf["fn"], conf["fp"], conf["tn"]) == (0, 1, 0, 1), "absent/absent frame is a TN")
    check(conf["precision"] is None and conf["recall"] == 0.0, "precision undefined (0/0), recall 0 (0/1)")
    fp_only = mt.frame_presence_confusion(gt_frames, {0: gt_frames[1], 1: gt_frames[0]})
    check(fp_only["precision"] == 0.0, "TP = 0 with a positive denominator is 0, not undefined")
    check(mt.relative_change(0.0, 5.0) is None and mt.relative_change(None, 5.0) is None, "L = 0 -> undefined")
    summary = mt.summarize([1.0, None, 3.0])
    check(summary["n_defined"] == 2 and summary["n_undefined"] == 1 and summary["median"] == 2.0,
          "summaries count undefined values instead of zero-filling")

    xy = np.array([[100, 100], [101, 100], [102, 100], [103, 100], [104, 100],   # GT cluster, frame 0
                   [101, 101],                                                   # FP near
                   [200, 200],                                                   # FP far
                   [50, 50],                                                     # FP in a GT-absent frame
                   [120, 120]], dtype=float)                                     # predicted on ignore
    label = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1])
    valid = np.array([True] * 8 + [False])
    frames = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    pred = np.array([True, False, False, False, False, True, True, True, True])
    gts = {0: frame_geometry(0, Points2D(xy[:5], LOCAL), Points2D(xy[:5], RAW))}
    fp = mt.fp_spatial_classes(pred_mask=pred, point_label=label, valid_mask=valid, frame_order=frames,
                               raw_points=Points2D(xy, RAW), gt_frames=gts, near_factor=1.0)
    check((fp["n_pred_all"], fp["n_pred_ignore"], fp["n_pred_valid"], fp["n_fp"]) == (5, 1, 4, 3),
          "denominators N_pred_all / ignore / valid / FP")
    check((fp["n_gt_near"], fp["n_gt_frame_far"], fp["n_gt_absent_frame"]) == (1, 1, 1),
          "near / far / GT-absent frame partition N_FP")
    check(abs(fp["gt_near_share_of_fp"] - 1 / 3) < 1e-12 and abs(fp["gt_near_share_of_pred_valid"] - 1 / 4) < 1e-12,
          "shares carry their denominators")


def test_perturbations_and_rules() -> None:
    print("[10] 15 perturbation conditions, FP from valid background only, stability rules")
    check(len(mt.PERTURBATION_CONDITIONS) == 15, "exactly 15 perturbation conditions")
    frame_order = np.array([0] * 20 + [1] * 12)
    gt = np.zeros(32, dtype=bool)
    gt[:10] = True
    gt[20:26] = True
    background = np.zeros(32, dtype=bool)
    background[10:20] = True            # frame 0 background
    background[26:27] = True            # frame 1: one background point only
    thinned, stats = mt.perturb_gt_mask("thin", 0.5, 0, frame_order=frame_order, gt_mask=gt, background_mask=background)
    check(int(thinned[:10].sum()) == 5 and int(thinned[20:26].sum()) == 5 and not thinned[~gt].any(),
          "thinning keeps max(ceil(r n), min(n, 5)) GT points and nothing else")
    injected, stats = mt.perturb_gt_mask("fp", 0.3, 0, frame_order=frame_order, gt_mask=gt, background_mask=background)
    added = injected & ~gt
    check(bool(np.all(background[added])), "FP points come from valid background only")
    check(stats["requested"] == 5 and stats["added"] == 4 and stats["shortfall_frames"] == 1,
          "requested 3 + 2, added 3 + 1, one shortfall frame recorded")
    again, _ = mt.perturb_gt_mask("fp", 0.3, 0, frame_order=frame_order, gt_mask=gt, background_mask=background)
    check(bool(np.array_equal(again, injected)), "perturbations are deterministic per (seed, cell, frame)")

    def uniform_cells(per_video):
        return {mt.cell_name(k, s): mt.stability_cell_summary(per_video) for k, s in mt.PERTURBATION_CELLS}

    cells = uniform_cells({"v1": [0.01, 0.02, 0.01], "v2": [0.03, 0.02, 0.02]})
    check(mt.candidate_rule(cells, baseline_failures=0)["status"] == mt.RULE_PASS, "all defined and small -> pass")

    # F-A: a failed video is WORST and stays in the denominator (never passed on successes alone).
    failed = uniform_cells({"v1": [0.01, 0.02, 0.01], "v2": [None, None, None]})
    rule = mt.candidate_rule(failed, baseline_failures=1)
    check(rule["status"] == mt.RULE_FAIL and rule["max_p90"] is None and rule["max_p90_is_worst"],
          "a failed video makes the P90 worst -> fail, distinct from not selectable")
    partial = uniform_cells({"v1": [0.01, None, 0.01], "v2": [0.01, 0.01, 0.01]})
    sel = partial[mt.cell_name("fp", 0.1)]["selection"]
    check(sel["n_seed_incomplete"] == 1 and sel["n_worst"] == 1, "one undefined seed makes that video worst")
    check(mt.candidate_rule(partial, baseline_failures=0)["status"] == mt.RULE_FAIL, "incomplete seeds fail the cell")
    empty = uniform_cells({"v1": [None, None, None]})
    check(mt.candidate_rule(empty, baseline_failures=1)["status"] == mt.RULE_NOT_SELECTABLE,
          "all undefined -> not selectable (not fail)")

    # Nearest-rank boundaries: P90 is worst exactly when worst videos > n - ceil(0.9 n).
    def cell(n_ok, n_bad, value=0.01):
        videos = {f"ok{i}": [value] * 3 for i in range(n_ok)}
        videos.update({f"bad{i}": [None] * 3 for i in range(n_bad)})
        return mt.stability_cell_summary(videos)["selection"]
    for n, bad, worst in ((10, 1, False), (10, 2, True), (20, 2, False), (20, 3, True), (143, 14, False), (143, 15, True)):
        check(cell(n - bad, bad)["p90_is_worst"] is worst, f"n={n}, {bad} worst -> P90 worst is {worst}")
    check(cell(5, 5)["median_is_worst"] is False and cell(4, 6)["median_is_worst"] is True,
          "median worst from 6 of 10 (rank ceil(5) = 5)")
    mixed = mt.stability_cell_summary({"a": [0.01] * 3, "b": [0.02] * 3, "c": [0.03] * 3, "d": [None] * 3})
    check(mixed["descriptive_success_only"]["median"] == 0.02 and mixed["descriptive_success_only"]["n_undefined"] == 1
          and mixed["selection"]["median"] == 0.02 and mixed["selection"]["p90_is_worst"],
          "descriptive success-only statistics are kept apart from the selection statistics")
    check(mt.nearest_rank([0.1, 0.2, 0.3, 0.4], 0.9) == 0.4 and mt.nearest_rank([0.1, 0.2, 0.3, 0.4], 0.5) == 0.2,
          "nearest rank uses no interpolation")
    import json as _json
    try:
        _json.dumps({"cells": failed, "rule": rule}, allow_nan=False)
        check(True, "selection outputs contain no Infinity or NaN (worst is null plus a flag)")
    except ValueError:
        check(False, "selection outputs contain no Infinity or NaN (worst is null plus a flag)")

    pick = mt.select_primary({fl.REP_OBB: 0.040, fl.REP_AXIS: 0.049})
    check(pick["primary"] == fl.REP_AXIS, "P90 within 1 percentage point -> precedence decides")
    pick = mt.select_primary({fl.REP_OBB: 0.030, fl.REP_AXIS: 0.050, fl.REP_BEST_FRAME: 0.045})
    check(pick["primary"] == fl.REP_OBB and pick["secondary"] == (fl.REP_BEST_FRAME, fl.REP_AXIS),
          "clear minimum wins; secondaries by P90 then precedence")


def test_holdout_prior_denominators() -> None:
    print("[11] held-out split conditions, prior self-exclusion, denominator breakdown")
    a, b = mt.holdout_split(10, seed=0, frame_order=3)
    check(len(a) == 5 and len(b) == 5 and not set(a) & set(b), "n = 10 -> A 5, B 5, disjoint")
    a2, _ = mt.holdout_split(10, seed=0, frame_order=3)
    check(bool(np.array_equal(a, a2)), "split is deterministic")
    small = np.array([[x, y] for x in (100.0, 101.0, 102.0) for y in (100.0, 101.0, 102.0)])   # one 9-point component
    tiny = frame_geometry(0, Points2D(small, LOCAL), Points2D(small, RAW))
    held = mt.frame_holdout(tiny, Points2D(small, RAW), seed=0)
    check(all(v is None for v in held.coverage.values())
          and set(held.reasons.values()) == {FailureReason.HOLDOUT_TOO_SMALL}, "n < 10 -> undefined, not 0")
    grid = rotated_grid(60, 8, 10, center=(128, 128), nt=31, ns=5)
    two_frames = {f: frame_geometry(f, Points2D(grid, LOCAL), Points2D(grid, RAW)) for f in (0, 1)}
    video = mt.video_holdout(two_frames, Points2D(grid, RAW))
    check(all(v is None for v in video["value"].values()), "fewer than 3 common frames -> video undefined")

    frames = {0: frame_geometry(0, Points2D(grid, LOCAL), Points2D(grid, RAW))}
    summaries = [pr.summarize_video_for_prior(name, frames, Points2D(grid, RAW), raw_wh=(256, 256))
                 for name in ("v1", "v2", "v3")]
    full = pr.build_prior(summaries)
    loo = pr.build_prior(summaries, exclude_identity="v2")
    check(full.n_videos == 3 and loo.n_videos == 2 and "v2" not in loo.identities, "prior excludes by identity")
    check(raises(GeometryContractError, pr.build_prior, summaries, exclude_identity="v9") is not None,
          "excluding an unknown identity stops")
    check(abs(((pr._doubled_angle_mean([math.radians(1), math.radians(179)]) + math.pi / 2) % math.pi)
              - math.pi / 2) < 1e-9, "doubled-angle mean of 1 and 179 deg is 0")
    estimates = pr.prior_estimates(full, frame_universe=(0, 1, 2), raw_wh=(256, 256))
    check(all(e.instance_rule == "prior" for e in estimates.values()), "prior estimates are labelled")

    breakdown = mt.DenominatorBreakdown()
    for _ in range(2):
        breakdown.add_input_unevaluable(InputReason.CROP_FALLBACK_RESIZE)
    for failure in [None] * 13 + [FailureReason.NO_PREDICTION] * 2 + [FailureReason.NO_USABLE_FRAME]:
        breakdown.add_result(failure)
    record = breakdown.as_record()
    check((record["total"], record["input_unevaluable"], record["evaluable"], record["success"],
           record["method_failure"]) == (18, 2, 16, 13, 3), "18 = 2 input + 16 evaluable (13 ok + 3 failed)")
    check(abs(record["method_failure_rate_of_evaluable"] - 3 / 16) < 1e-12, "failure rate over evaluable")


def _prior_video(raw_wh, angle_deg, center_n=(0.5, 0.5), length_n=0.4, width_n=0.04):
    """A GT box defined in NORMALISED coordinates, rendered into a W x H frame."""
    raw = rotated_grid(length_n, width_n, angle_deg, center=center_n) * np.asarray(raw_wh, dtype=float)
    return {0: frame_geometry(0, Points2D(raw, LOCAL), Points2D(raw, RAW))}, Points2D(raw, RAW)


def test_prior_normalised() -> None:
    print("[12] prior in normalised coordinates across frame sizes and aspect ratios")
    # (A) the same normalised box in 640x480 and 960x540 frames, mapped into 1000x600.
    #     101 x 21 grid: normalised length 0.96 x 0.4 = 0.384, width 0.04.
    summaries = []
    for name, wh in (("v1", (640, 480)), ("v2", (960, 540))):
        frames, raw = _prior_video(wh, 0.0, center_n=(0.5, 0.4))
        summaries.append(pr.summarize_video_for_prior(name, frames, raw, raw_wh=wh))
    prior = pr.build_prior(summaries)
    check(close(prior.center_norm, (0.5, 0.4), 1e-9) and abs(prior.length_norm - 0.384) < 1e-9
          and abs(prior.width_norm - 0.04) < 1e-9, "(A) centre, length and width are all normalised")
    target = pr.prior_instance(prior, raw_wh=(1000, 600))
    check(abs(target.length - 384.0) < 1e-6 and abs(target.width - 24.0) < 1e-6
          and close(target.center, (500.0, 240.0), 1e-6),
          "(A) mapped by the TARGET size: length 0.384 x 1000, width 0.04 x 600 (raw-pixel mean would give 307.2)")

    # (B) a 45 deg normalised axis in a 1000x500 frame (26.57 deg in pixels), mapped into 500x1000.
    frames, raw = _prior_video((1000, 500), 45.0)
    diagonal = pr.build_prior([pr.summarize_video_for_prior("v", frames, raw, raw_wh=(1000, 500))])
    check(abs(diagonal.axis_angle_norm - math.pi / 4) < 1e-9, "(B) angle is fitted in normalised space (45 deg)")
    mapped = pr.prior_instance(diagonal, raw_wh=(500, 1000))
    c45 = math.cos(math.pi / 4)
    check(abs(math.degrees(mapped.axis_angle) - math.degrees(math.atan2(1000, 500))) < 1e-6,
          "(B) mapped angle = atan(H/W) = 63.43 deg, not the source frame's 26.57 deg")
    check(abs(mapped.length - 0.384 * math.hypot(c45 * 500, c45 * 1000)) < 1e-6,
          "(B) mapped length from the mapped endpoints")
    minor = np.array([-c45, c45]) * 0.02 * np.array([500.0, 1000.0])
    perp = np.array([-1000.0, 500.0]) / math.hypot(500, 1000)
    check(abs(mapped.width - 2 * abs(float(minor @ perp))) < 1e-6, "(B) width perpendicular to the mapped axis")
    check(close(mapped.center, (250.0, 500.0), 1e-6), "(B) centre mapped by the target size")
    estimates = pr.prior_estimates(diagonal, frame_universe=(0,), raw_wh=(500, 1000))
    check(estimates[fl.REP_OBB].value == mapped.length and estimates[fl.REP_AABB].value == mapped.aabb_long_side,
          "(B) prior estimates use the mapped geometry")
    check(raises(GeometryContractError, pr.summarize_video_for_prior, "v", frames, Points2D(raw.xy, LOCAL),
                 raw_wh=(1000, 500)) is not None, "prior input must be raw_frame_px")


def main() -> int:
    print("Stage5 S5-17 geometry core synthetic test")
    for test in (
        test_transform_known_values,
        test_pixel_to_mm,
        test_instance_geometry,
        test_region_alignment,
        test_components,
        test_frame_and_video_estimates,
        test_postprocess,
        test_matching,
        test_detection_and_undefined,
        test_perturbations_and_rules,
        test_holdout_prior_denominators,
        test_prior_normalised,
    ):
        test()
    print(f"checks: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        return 1
    print("All geometry core checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
