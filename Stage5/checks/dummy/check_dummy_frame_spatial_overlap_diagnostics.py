from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_frame_spatial_overlap_diagnostics import (
    audit_h5,
    class_exposure_summary,
    find_runs,
    per_frame_label_counts,
    positive_frame_runs,
    relative_frame_decile,
    require_finite,
    require_no_duplicate_or_missing_points,
    window_occurrence_counts,
)

# ----------------------------------------------------------------------------
# S5-14 Step H1: synthetic pass/fail coverage for
# check_stage5_frame_spatial_overlap_diagnostics.py's shared primitives, plus
# a hand-built H5 integration test for audit_h5() (Step H2). No CUDA needed.
# ----------------------------------------------------------------------------


def expect_pass(label: str, fn) -> None:
    fn()
    print(f"ok (expected pass): {label}")


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def test_find_runs() -> None:
    assert find_runs(np.array([], dtype=np.int64)) == []
    assert find_runs(np.array([5], dtype=np.int64)) == [(5, 5)]
    assert find_runs(np.array([1, 2, 3, 7, 8, 12], dtype=np.int64)) == [(1, 3), (7, 8), (12, 12)]
    # contiguous run of consecutive integers with no gaps
    assert find_runs(np.arange(10, 20)) == [(10, 19)]
    print("ok: find_runs basic/gap/single/empty cases")

    expect_fail(
        "find_runs rejects non-strictly-increasing input",
        lambda: find_runs(np.array([1, 1, 2], dtype=np.int64)),
    )


def test_per_frame_label_counts() -> None:
    frame_order = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    point_label = np.array([1, 0, -1, 1, 1, 0, 0, -1], dtype=np.int64)
    valid_mask = point_label != -1
    counts = per_frame_label_counts(frame_order, point_label, valid_mask, ignore_index=-1)
    assert counts == {
        0: {"positive": 1, "background": 1, "ignore": 1},
        1: {"positive": 2, "background": 0, "ignore": 0},
        2: {"positive": 0, "background": 2, "ignore": 1},
    }, counts
    print(f"ok: per_frame_label_counts = {counts}")


def test_window_occurrence_counts() -> None:
    # frames 0..19, window size 16 stride 8 tail included -> matches S5-13/S5-14
    # baseline window config. Two windows: [0,15] and tail [8,19].
    frame_order = np.repeat(np.arange(20), 2)
    counts, num_windows = window_occurrence_counts(
        frame_order, window_size_frames=16, window_stride_frames=8, include_tail_window=True
    )
    assert num_windows == 2, num_windows
    # frame 0..7 only in window 0 -> count 1 per point; frame 8..15 in both -> count 2; frame 16..19 only tail -> count 1
    frame_to_count = {frame: counts[frame_order == frame][0] for frame in range(20)}
    for frame in range(0, 8):
        assert frame_to_count[frame] == 1, (frame, frame_to_count[frame])
    for frame in range(8, 16):
        assert frame_to_count[frame] == 2, (frame, frame_to_count[frame])
    for frame in range(16, 20):
        assert frame_to_count[frame] == 1, (frame, frame_to_count[frame])
    print(f"ok: window_occurrence_counts num_windows={num_windows}, overlap band counts correct")


def test_relative_frame_decile() -> None:
    frame_order = np.arange(0, 100)
    deciles = relative_frame_decile(frame_order, np.array([0, 9, 10, 50, 98, 99]))
    assert deciles.tolist() == [0, 0, 1, 5, 9, 9], deciles.tolist()
    print(f"ok: relative_frame_decile boundary values = {deciles.tolist()}")

    # degenerate span (single frame value) must not divide by zero
    single = relative_frame_decile(np.array([5]), np.array([5]))
    assert single.tolist() == [0]
    print("ok: relative_frame_decile handles single-frame (zero span) video")


def test_class_exposure_summary() -> None:
    point_label = np.array([1, 1, 0, 0, 0, -1], dtype=np.int64)
    valid_mask = point_label != -1
    occurrence_counts = np.array([2, 3, 1, 1, 4, 0], dtype=np.int64)
    summary = class_exposure_summary(point_label, valid_mask, occurrence_counts)
    assert summary["positive"]["unique_point_count"] == 2
    assert summary["positive"]["window_occurrence_total"] == 5
    assert abs(summary["positive"]["exposure_multiplier"] - 2.5) < 1e-9
    assert summary["background"]["unique_point_count"] == 3
    assert summary["background"]["window_occurrence_total"] == 6
    print(f"ok: class_exposure_summary = {summary}")

    empty_label = np.array([0, 0, 0], dtype=np.int64)
    empty_summary = class_exposure_summary(empty_label, np.ones(3, dtype=bool), np.array([1, 1, 1]))
    assert empty_summary["positive"]["unique_point_count"] == 0
    assert empty_summary["positive"]["exposure_multiplier"] is None
    print("ok: class_exposure_summary handles an empty class without dividing by zero")


def test_positive_frame_runs() -> None:
    frame_counts = {
        0: {"positive": 1, "background": 0, "ignore": 0},
        1: {"positive": 1, "background": 0, "ignore": 0},
        2: {"positive": 0, "background": 1, "ignore": 0},
        5: {"positive": 1, "background": 0, "ignore": 0},
        6: {"positive": 1, "background": 0, "ignore": 0},
        7: {"positive": 1, "background": 0, "ignore": 0},
    }
    runs = positive_frame_runs(frame_counts)
    assert runs == [(0, 1), (5, 7)], runs
    print(f"ok: positive_frame_runs = {runs}")


def test_fail_fast_helpers() -> None:
    expect_pass(
        "require_no_duplicate_or_missing_points accepts a full 0..N-1 cover",
        lambda: require_no_duplicate_or_missing_points(np.arange(5), expected_count=5, context="ok"),
    )
    expect_fail(
        "require_no_duplicate_or_missing_points rejects a duplicate index",
        lambda: require_no_duplicate_or_missing_points(np.array([0, 1, 1, 3, 4]), expected_count=5, context="dup"),
    )
    expect_fail(
        "require_no_duplicate_or_missing_points rejects a missing index",
        lambda: require_no_duplicate_or_missing_points(np.array([0, 1, 2, 3]), expected_count=5, context="missing"),
    )
    expect_pass("require_finite accepts an all-finite array", lambda: require_finite(np.array([0.0, 1.0, -1.0]), context="ok"))
    expect_fail(
        "require_finite rejects a NaN",
        lambda: require_finite(np.array([0.0, float("nan")]), context="nan"),
    )
    expect_fail(
        "require_finite rejects an Inf",
        lambda: require_finite(np.array([0.0, float("inf")]), context="inf"),
    )


def make_dummy_pointcloud_h5(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    num_frames = 24
    points_per_frame = 10
    n = num_frames * points_per_frame
    frame_order = np.repeat(np.arange(num_frames, dtype=np.int64), points_per_frame)
    point_label = np.zeros(n, dtype=np.int8)
    # frames 5-7 and 15 are GT-positive-bearing (two separate runs)
    positive_frames = {5, 6, 7, 15}
    for i in range(n):
        if int(frame_order[i]) in positive_frames and (i % points_per_frame) < 3:
            point_label[i] = 1
    valid_mask = np.ones(n, dtype=bool)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        pc = f.create_group("point_cloud")
        pc.attrs["point_mode"] = "grid"
        pc.create_dataset("points", data=rng.normal(size=(n, 3)).astype(np.float32))
        pc.create_dataset("intensity", data=rng.integers(0, 256, size=n, dtype=np.uint8))
        pc.create_dataset("alpha", data=rng.integers(0, 256, size=n, dtype=np.uint8))
        pc.create_dataset("confidence", data=rng.random(size=n).astype(np.float32))
        pc.create_dataset("frame_order", data=frame_order)
        pc.create_dataset("pixel_xy", data=rng.integers(0, 256, size=(n, 2)).astype(np.float32))
        ann = f.create_group("annotation")
        ann.create_dataset("point_label", data=point_label)
        ann.create_dataset("valid_mask", data=valid_mask)
        f.attrs["video_name"] = "dummy_video"


def test_audit_h5_integration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dummy.h5"
        make_dummy_pointcloud_h5(path, seed=0)
        result = audit_h5(
            path,
            ignore_index=-1,
            window_size_frames=16,
            window_stride_frames=8,
            include_tail_window=True,
        )
        assert result["num_points"] == 240, result["num_points"]
        assert result["min_frame"] == 0 and result["max_frame"] == 23
        assert result["num_distinct_frames"] == 24
        assert result["positive_frame_run_count"] == 2, result["positive_frame_run_count"]
        assert result["positive_frame_run_lengths"] == [3, 1], result["positive_frame_run_lengths"]
        assert result["class_exposure"]["positive"]["unique_point_count"] == 4 * 3
        assert result["class_exposure"]["positive"]["exposure_multiplier"] is not None
        assert set(result["decile_exposure"].keys()) == set(range(10))
        print(f"ok: audit_h5 integration -- {result['num_windows']} windows, "
              f"positive runs={result['positive_frame_run_lengths']}, "
              f"positive exposure multiplier={result['class_exposure']['positive']['exposure_multiplier']:.3f}")


def main() -> None:
    test_find_runs()
    test_per_frame_label_counts()
    test_window_occurrence_counts()
    test_relative_frame_decile()
    test_class_exposure_summary()
    test_positive_frame_runs()
    test_fail_fast_helpers()
    test_audit_h5_integration()
    print("check_dummy_frame_spatial_overlap_diagnostics: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
