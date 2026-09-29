from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from stage5.geometry import metrics as mt  # noqa: E402
from stage5.geometry.fl_estimate import REP_AABB, REP_AXIS, REP_OBB  # noqa: E402
from stage5.geometry.frame_geometry import frame_geometry, instance_geometry  # noqa: E402
from stage5.geometry.types import CoordSpace, FrameGeometry, Points2D  # noqa: E402

# ----------------------------------------------------------------------------
# S5-17 S17-3: OBSERVATION of held-out coverage C_ho on known shapes.
#
# Management document 5.2. This is not a pass/fail test. It reports what the
# fixed held-out procedure (instance fixed before the split, n >= 10,
# A = ceil(n/2) builds, B = floor(n/2) is measured, seeds 0/1/2) gives on
# synthetic shapes, so that the use of C_ho can be decided before real data.
# The analytic 0.92-0.96 range for (b)/(c) describes an idealised
# distribution; a finite sample, an estimated PCA axis and held-out points
# differ from it, and a difference alone is not an implementation defect.
#
# Shapes x points: rectangle (uniform 100 x 20), Gaussian ellipse (sd 25 x 5,
# 5:1), curved band (arc radius 150, 40 deg, width 10), two regions (two
# 60 x 12 rectangles 90 px apart, standard 4 px linkage so M1 is the larger
# region) x 20 / 100 / 1000 points, 50 random frames each. Single-shape cases
# use one instance of all points (built directly, without linkage) so that the
# box geometry is observed on its own.
#
# numpy only. Synthetic data only.
# ----------------------------------------------------------------------------

SHAPES = ("rectangle", "gaussian_ellipse_5to1", "curved_band", "two_regions")
POINT_COUNTS = (20, 100, 1000)
FRAMES_PER_CASE = 50
REPS = (REP_AABB, REP_OBB, REP_AXIS)


def _rotate(xy: np.ndarray, angle_deg: float, center=(300.0, 200.0)) -> np.ndarray:
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return xy @ rot.T + np.asarray(center)


def sample(shape: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if shape == "rectangle":
        xy = np.column_stack([rng.uniform(-50, 50, n), rng.uniform(-10, 10, n)])
    elif shape == "gaussian_ellipse_5to1":
        xy = np.column_stack([rng.normal(0, 25, n), rng.normal(0, 5, n)])
    elif shape == "curved_band":
        theta = np.radians(rng.uniform(-20, 20, n))
        radius = 150 + rng.uniform(-5, 5, n)
        xy = np.column_stack([radius * np.sin(theta), radius * np.cos(theta) - 150])
    elif shape == "two_regions":
        half = n // 2
        left = np.column_stack([rng.uniform(-75, -15, n - half), rng.uniform(-6, 6, n - half)])
        right = np.column_stack([rng.uniform(15, 75, half), rng.uniform(-6, 6, half)])
        xy = np.concatenate([left, right])
    else:
        raise ValueError(shape)
    return _rotate(xy, 25.0)


def analytic_reference(shape: str, n: int) -> dict[str, str]:
    m = math.ceil(n / 2)
    ref = {REP_OBB: "idealised 0.92-0.96 (independent axes ~0.922)", REP_AXIS: "same region as (b)"}
    if shape == "rectangle":
        ref[REP_AABB] = (f"axis-aligned uniform ((m-1)/(m+1))^2 = {((m - 1) / (m + 1)) ** 2:.3f} "
                         "(the shape is rotated 25 deg, so only indicative)")
    else:
        ref[REP_AABB] = "no closed form used"
    return ref


def observe() -> list[dict]:
    rows = []
    for s_index, shape in enumerate(SHAPES):
        for n in POINT_COUNTS:
            values: dict[str, list[float]] = {rep: [] for rep in REPS}
            reasons = Counter()
            frames_absent = 0
            for frame in range(FRAMES_PER_CASE):
                rng = np.random.default_rng([s_index, n, frame])
                xy = sample(shape, n, rng)
                raw = Points2D(xy, CoordSpace.RAW_FRAME_PX)
                if shape == "two_regions":             # standard linkage: M1 is the larger region
                    geometry = frame_geometry(frame, Points2D(xy, CoordSpace.LOCAL_CROP_PX), raw)
                else:                                   # one instance of all points
                    whole = instance_geometry(raw)
                    geometry = FrameGeometry(frame_order=frame, present=True, status="ok", instances=(whole,),
                                             merged=whole, instance_point_indices=(np.arange(n),))
                if not geometry.present:
                    frames_absent += 1
                    continue
                for seed in mt.HOLDOUT_SEEDS:
                    held = mt.frame_holdout(geometry, Points2D(xy, CoordSpace.RAW_FRAME_PX), seed=seed)
                    for rep in REPS:
                        if held.coverage[rep] is None:
                            reasons[f"{rep}:{held.reasons[rep]}"] += 1
                        else:
                            values[rep].append(held.coverage[rep])
            reference = analytic_reference(shape, n)
            for rep in REPS:
                summary = mt.summarize(values[rep])
                rows.append({
                    "shape": shape, "n_points": n, "representation": rep,
                    "frames": FRAMES_PER_CASE, "frames_without_instance": frames_absent,
                    "defined_frame_seeds": summary["n_defined"],
                    "median": summary["median"], "q1": summary["q1"], "q3": summary["q3"],
                    "undefined_reasons": {k.split(":", 1)[1]: v for k, v in reasons.items() if k.startswith(rep)},
                    "reference": reference[rep],
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe held-out coverage on synthetic shapes (no pass/fail).")
    parser.add_argument("--json", default=None, help="optional output path for the observation table")
    args = parser.parse_args()
    rows = observe()
    print("Stage5 S5-17 held-out coverage observation (synthetic, descriptive)")
    print(f"{'shape':24s} {'n':>5s} {'rep':12s} {'defined':>8s} {'median':>8s} {'q1':>8s} {'q3':>8s}  undefined")
    for row in rows:
        fmt = lambda v: "   -    " if v is None else f"{v:8.3f}"  # noqa: E731
        print(f"{row['shape']:24s} {row['n_points']:5d} {row['representation']:12s} {row['defined_frame_seeds']:8d} "
              f"{fmt(row['median'])} {fmt(row['q1'])} {fmt(row['q3'])}  {row['undefined_reasons'] or ''}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"table written: {Path(args.json).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
