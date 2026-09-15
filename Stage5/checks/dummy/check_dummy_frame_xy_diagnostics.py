from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_frame_xy_diagnostics import (
    assign_xy_grid_bin,
    bin_frame_activity,
    build_frame_metrics_rows,
    centroid_distance,
    classify_relative_to_runs,
    compute_xy_density,
    frame_confusion_counts,
    frame_probability_stats,
    nearest_positive_frame_distance,
    process_video,
    temporal_recurrence_for_bin,
    xy_centroid_and_variance,
)

# ----------------------------------------------------------------------------
# S5-14 Step H3/H3.1: synthetic pass/fail coverage for
# check_stage5_frame_xy_diagnostics.py -- frame-level confusion/probability
# stats, XY centroid availability handling, GT-interval classification, XY
# grid binning boundaries, per-bin temporal recurrence run matching, and a
# hand-built H5+prediction-npz integration test. No CUDA needed.
# ----------------------------------------------------------------------------


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def test_frame_confusion_counts() -> None:
    frame_order = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    point_label = np.array([1, 0, -1, 1, 0, 1], dtype=np.int64)
    valid_mask = point_label != -1
    pred_label = np.array([1, 1, 0, 0, 0, 1], dtype=np.uint8)
    result = frame_confusion_counts(frame_order, point_label, valid_mask, pred_label, ignore_index=-1)
    assert result == {
        0: {"tp": 1, "fp": 1, "tn": 0, "fn": 0, "ignore_count": 1},
        1: {"tp": 1, "fp": 0, "tn": 1, "fn": 1, "ignore_count": 0},
    }, result
    print(f"ok: frame_confusion_counts = {result}")


def test_frame_probability_stats() -> None:
    frame_order = np.array([0, 0, 0], dtype=np.int64)
    point_label = np.array([1, 0, -1], dtype=np.int64)
    valid_mask = point_label != -1
    prob = np.array([0.8, 0.2, 0.5], dtype=np.float32)
    result = frame_probability_stats(frame_order, point_label, valid_mask, prob, ignore_index=-1)
    assert abs(result[0]["positive"]["mean"] - 0.8) < 1e-5 and result[0]["positive"]["count"] == 1
    assert abs(result[0]["background"]["mean"] - 0.2) < 1e-5
    assert abs(result[0]["ignore"]["mean"] - 0.5) < 1e-5 and result[0]["ignore"]["count"] == 1
    print(f"ok: frame_probability_stats = {result}")

    empty_frame = np.array([5], dtype=np.int64)
    empty_label = np.array([0], dtype=np.int64)
    result_empty = frame_probability_stats(empty_frame, empty_label, np.array([True]), np.array([0.1]), ignore_index=-1)
    assert result_empty[5]["positive"]["count"] == 0 and result_empty[5]["positive"]["mean"] is None
    print("ok: frame_probability_stats handles an empty group without dividing by zero")


def test_xy_centroid_and_distance() -> None:
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    centroid, variance = xy_centroid_and_variance(xy, np.array([True, True, False]))
    assert np.allclose(centroid, [1.0, 0.0]), centroid
    print(f"ok: xy_centroid_and_variance = centroid {centroid}, variance {variance}")

    empty_centroid, empty_variance = xy_centroid_and_variance(xy, np.array([False, False, False]))
    assert empty_centroid is None and empty_variance is None
    print("ok: xy_centroid_and_variance returns None (not zeros) for an empty mask")

    assert centroid_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0
    assert centroid_distance(None, np.array([1.0, 1.0])) is None
    assert centroid_distance(np.array([1.0, 1.0]), None) is None
    print("ok: centroid_distance handles both available and unavailable cases")


def test_nearest_positive_frame_distance() -> None:
    positive_frames = np.array([5, 10, 20], dtype=np.int64)
    assert nearest_positive_frame_distance(5, positive_frames) == 0
    assert nearest_positive_frame_distance(7, positive_frames) == 2
    assert nearest_positive_frame_distance(15, positive_frames) == 5
    assert nearest_positive_frame_distance(0, positive_frames) == 5
    assert nearest_positive_frame_distance(0, np.array([], dtype=np.int64)) is None
    print("ok: nearest_positive_frame_distance boundary/empty cases")


def test_classify_relative_to_runs() -> None:
    runs = [(5, 7), (20, 22)]
    assert classify_relative_to_runs(6, runs) == "inside"
    assert classify_relative_to_runs(5, runs) == "inside"
    assert classify_relative_to_runs(0, runs) == "before"
    assert classify_relative_to_runs(30, runs) == "after"
    assert classify_relative_to_runs(13, runs) == "after"  # nearer to (5,7) (distance 6) than (20,22) (distance 7), and past its end
    assert classify_relative_to_runs(10, []) == "no_gt_run"
    print("ok: classify_relative_to_runs inside/before/after/no_gt_run")


def test_assign_xy_grid_bin() -> None:
    xy = np.array([[0.0, 0.0], [0.999, 0.999], [1.0, 1.0], [0.5, 0.5]])
    bins = assign_xy_grid_bin(xy, grid_resolution=4)
    assert bins.tolist() == [[0, 0], [3, 3], [3, 3], [2, 2]], bins.tolist()
    print(f"ok: assign_xy_grid_bin boundary values (0, near-1, exactly-1, mid) = {bins.tolist()}")

    expect_fail(
        "assign_xy_grid_bin rejects values outside [0, 1]",
        lambda: assign_xy_grid_bin(np.array([[1.5, 0.0]]), grid_resolution=4),
    )
    expect_fail(
        "assign_xy_grid_bin rejects non-finite values",
        lambda: assign_xy_grid_bin(np.array([[float("nan"), 0.0]]), grid_resolution=4),
    )


def test_compute_xy_density_and_bin_frame_activity() -> None:
    xy = np.array([[0.1, 0.1], [0.1, 0.1], [0.9, 0.9]])
    mask = np.array([True, True, True])
    density = compute_xy_density(xy, mask, grid_resolution=2)
    assert density[0, 0] == 2 and density[1, 1] == 1, density
    print(f"ok: compute_xy_density = {density.tolist()}")

    frame_order = np.array([0, 1, 5], dtype=np.int64)
    activity = bin_frame_activity(frame_order, xy, mask, grid_resolution=2)
    assert set(activity.keys()) == {(0, 0), (1, 1)}, activity
    assert activity[(0, 0)].tolist() == [0, 1]
    assert activity[(1, 1)].tolist() == [5]
    print(f"ok: bin_frame_activity = {activity}")

    empty_activity = bin_frame_activity(frame_order, xy, np.array([False, False, False]), grid_resolution=2)
    assert empty_activity == {}
    print("ok: bin_frame_activity handles an empty mask without error")


def test_temporal_recurrence_for_bin() -> None:
    predicted_frames = np.array([1, 2, 3, 10, 11, 30], dtype=np.int64)
    gt_positive_frames = np.array([2, 3, 4], dtype=np.int64)
    result = temporal_recurrence_for_bin(predicted_frames, gt_positive_frames)
    assert result["predicted_run_count"] == 3, result
    assert result["matched_with_gt_run_count"] == 1, result
    assert result["unmatched_run_count"] == 2, result
    assert result["multiple_unmatched_runs"] is True, result
    assert result["unmatched_run_gaps"] == [30 - 11 - 1], result
    print(f"ok: temporal_recurrence_for_bin matched/unmatched/gap computation = {result}")

    no_gt_result = temporal_recurrence_for_bin(predicted_frames, np.array([], dtype=np.int64))
    assert no_gt_result["matched_with_gt_run_count"] == 0
    assert no_gt_result["unmatched_run_count"] == 3
    print("ok: temporal_recurrence_for_bin with no GT-positive frames -> everything unmatched")


def test_build_frame_metrics_rows_availability_flags() -> None:
    # frame 0: GT positive and predicted positive both present -> centroid distance available.
    # frame 1: no GT positive at all (no-GT frame) but a stray FP prediction -> centroid distance unavailable.
    frame_order = np.array([0, 0, 1, 1], dtype=np.int64)
    point_label = np.array([1, 0, 0, 0], dtype=np.int64)
    valid_mask = np.ones(4, dtype=bool)
    pixel_xy = np.array([[10.0, 10.0], [50.0, 50.0], [20.0, 20.0], [60.0, 60.0]])
    pred_label = np.array([1, 0, 1, 0], dtype=np.uint8)
    prob_femur = np.array([0.9, 0.1, 0.6, 0.2], dtype=np.float32)

    rows = build_frame_metrics_rows(
        video_alias="dummy",
        split="validation",
        frame_order=frame_order,
        point_label=point_label,
        valid_mask=valid_mask,
        pixel_xy=pixel_xy,
        pred_label=pred_label,
        prob_femur=prob_femur,
        ignore_index=-1,
    )
    by_frame = {row["frame"]: row for row in rows}
    assert by_frame[0]["centroid_distance_available"] is True
    assert by_frame[0]["centroid_distance"] == 0.0, by_frame[0]["centroid_distance"]
    assert by_frame[0]["is_no_gt_frame"] is False

    assert by_frame[1]["is_no_gt_frame"] is True
    assert by_frame[1]["gt_centroid_x"] is None and by_frame[1]["gt_centroid_y"] is None
    assert by_frame[1]["centroid_distance_available"] is False
    assert by_frame[1]["false_positive_count"] == 1
    assert by_frame[1]["no_gt_frame_fp_count"] == 1
    print(f"ok: build_frame_metrics_rows availability flags -- frame0(has GT)={by_frame[0]}, "
          f"frame1(no GT)={by_frame[1]}")


def make_video_h5_and_prediction(root: Path, *, seed: int) -> tuple[Path, Path, dict]:
    rng = np.random.default_rng(seed)
    n = 40
    frame_order = np.repeat(np.arange(20, dtype=np.int64), 2)
    point_label = np.zeros(n, dtype=np.int8)
    point_label[(frame_order >= 5) & (frame_order <= 6)] = 1  # a short GT-positive run
    valid_mask = np.ones(n, dtype=bool)
    pixel_xy = rng.integers(0, 256, size=(n, 2)).astype(np.float32)

    h5_path = root / "video.h5"
    with h5py.File(h5_path, "w") as f:
        pc = f.create_group("point_cloud")
        pc.attrs["point_mode"] = "grid"
        pc.create_dataset("points", data=rng.normal(size=(n, 3)).astype(np.float32))
        pc.create_dataset("intensity", data=rng.integers(0, 256, size=n, dtype=np.uint8))
        pc.create_dataset("alpha", data=rng.integers(0, 256, size=n, dtype=np.uint8))
        pc.create_dataset("confidence", data=rng.random(size=n).astype(np.float32))
        pc.create_dataset("frame_order", data=frame_order)
        pc.create_dataset("pixel_xy", data=pixel_xy)
        ann = f.create_group("annotation")
        ann.create_dataset("point_label", data=point_label)
        ann.create_dataset("valid_mask", data=valid_mask)
        f.attrs["video_name"] = "video_a"

    prob_femur = np.where(point_label == 1, 0.8, 0.1).astype(np.float32)
    pred_label = (prob_femur >= 0.5).astype(np.uint8)
    npz_path = root / "video_a.npz"
    np.savez_compressed(npz_path, point_indices=np.arange(n, dtype=np.int64), prob_femur=prob_femur, pred_label=pred_label, vote_count=np.ones(n, dtype=np.int32))

    tp = int(np.sum(valid_mask & (point_label == 1) & (pred_label == 1)))
    fp = int(np.sum(valid_mask & (point_label == 0) & (pred_label == 1)))
    tn = int(np.sum(valid_mask & (point_label == 0) & (pred_label == 0)))
    fn = int(np.sum(valid_mask & (point_label == 1) & (pred_label == 0)))
    h5_metrics_row = {
        "true_positive_count": str(tp),
        "false_positive_count": str(fp),
        "true_negative_count": str(tn),
        "false_negative_count": str(fn),
    }
    return h5_path, npz_path, h5_metrics_row


def test_process_video_integration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h5_path, npz_path, h5_metrics_row = make_video_h5_and_prediction(root, seed=0)
        result = process_video(
            video_alias="video_a",
            split="validation",
            h5_path=h5_path,
            npz_path=npz_path,
            h5_metrics_row=h5_metrics_row,
            ignore_index=-1,
            pixel_width=256,
            pixel_height=256,
        )
        assert len(result["frame_rows"]) == 20, len(result["frame_rows"])
        assert np.all((result["xy_normalized"] >= 0.0) & (result["xy_normalized"] <= 1.0))
        assert result["gt_positive_frames_sorted"].tolist() == [5, 6], result["gt_positive_frames_sorted"].tolist()
        print("ok: process_video integration -- parity gate passed, 20 frame rows, normalized xy in bounds")

        # tamper with the h5_metrics row -> parity gate must reject
        tampered_row = dict(h5_metrics_row)
        tampered_row["true_positive_count"] = "999"
        try:
            process_video(
                video_alias="video_a",
                split="validation",
                h5_path=h5_path,
                npz_path=npz_path,
                h5_metrics_row=tampered_row,
                ignore_index=-1,
                pixel_width=256,
                pixel_height=256,
            )
        except AssertionError:
            print("ok (expected fail): process_video rejects a tampered h5_metrics parity row")
        else:
            raise AssertionError("expected process_video to fail-fast on a tampered parity row")


def test_main_cli_does_not_leak_real_video_name() -> None:
    """Regression test for a real bug found in this session: main() once wrote
    the RAW (non-anonymized) evaluate_stage5.py video_name straight into every
    output row as "video_alias". Build a fixture with a realistic
    timestamp-style video_name, run the actual CLI end-to-end, and assert it
    never appears in any output file -- only the enumeration-based alias
    (e.g. "validation_000") should appear.
    """
    fake_real_video_name = "20250626_999999_1234"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h5_path, npz_path_tmp, h5_metrics_row = make_video_h5_and_prediction(root, seed=1)

        evaluation_dir = root / "eval"
        # main() requires at least one row per split (train_sanity AND
        # validation) -- reuse the same synthetic H5/prediction under both.
        rows = []
        for split in ("train_sanity", "validation"):
            predictions_dir = evaluation_dir / "predictions" / split
            predictions_dir.mkdir(parents=True)
            with np.load(npz_path_tmp) as data:
                np.savez_compressed(
                    predictions_dir / f"{fake_real_video_name}.npz",
                    point_indices=data["point_indices"],
                    prob_femur=data["prob_femur"],
                    pred_label=data["pred_label"],
                    vote_count=data["vote_count"],
                )
            rows.append(
                {
                    "checkpoint": "last",
                    "split": split,
                    "video_name": fake_real_video_name,
                    "h5_path": str(h5_path),
                    **h5_metrics_row,
                }
            )

        h5_metrics_csv = evaluation_dir / "h5_metrics.csv"
        h5_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with h5_metrics_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        frame_metrics_csv = root / "out" / "frame_metrics.csv"
        video_summary_csv = root / "out" / "video_summary.csv"
        xy_density_csv = root / "out" / "xy_density_bins.csv"
        xy_recurrence_csv = root / "out" / "xy_temporal_recurrence.csv"
        summary_json = root / "out" / "summary.json"

        script = REPO_ROOT / "checks" / "real_h5" / "check_stage5_frame_xy_diagnostics.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--evaluation_dir", str(evaluation_dir),
                "--checkpoint", "last",
                "--pixel_width", "256",
                "--pixel_height", "256",
                "--grid_resolutions", "4",
                "--frame_metrics_csv", str(frame_metrics_csv),
                "--video_summary_csv", str(video_summary_csv),
                "--xy_density_bins_csv", str(xy_density_csv),
                "--xy_temporal_recurrence_csv", str(xy_recurrence_csv),
                "--summary_json", str(summary_json),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed: stdout={result.stdout}\nstderr={result.stderr}"

        for output_path in (frame_metrics_csv, video_summary_csv, xy_density_csv, xy_recurrence_csv):
            content = output_path.read_text()
            assert fake_real_video_name not in content, (
                f"{output_path.name} leaks the real video_name {fake_real_video_name!r} -- "
                "video_alias must be an enumeration-based label, never the raw evaluate_stage5.py video_name"
            )
            assert "validation_000" in content, f"{output_path.name} missing the expected anonymized alias"
        print(
            "ok: main() CLI never writes the real video_name into any output file "
            "(frame_metrics/video_summary/xy_density_bins/xy_temporal_recurrence all use validation_000)"
        )


def main() -> None:
    test_frame_confusion_counts()
    test_frame_probability_stats()
    test_xy_centroid_and_distance()
    test_nearest_positive_frame_distance()
    test_classify_relative_to_runs()
    test_assign_xy_grid_bin()
    test_compute_xy_density_and_bin_frame_activity()
    test_temporal_recurrence_for_bin()
    test_build_frame_metrics_rows_availability_flags()
    test_process_video_integration()
    test_main_cli_does_not_leak_real_video_name()
    print("check_dummy_frame_xy_diagnostics: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
