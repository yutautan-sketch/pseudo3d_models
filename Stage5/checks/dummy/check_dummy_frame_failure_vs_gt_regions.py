from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))

from check_stage5_frame_failure_vs_gt_regions import (  # noqa: E402
    aggregate_paired,
    build_frame_table,
    exact_sign_test,
    resolve_checkpoint,
    summarize_frame_localization,
    within_video_region_comparison,
)

# ----------------------------------------------------------------------------
# S5-15 verification (2): synthetic coverage for the within-video frame-level
# comparison.
#
# The properties that matter: the sign test is the exact binomial and not an
# approximation, a video with only one kind of frame is EXCLUDED rather than
# counted as a tie (counting it as a tie would silently dilute the result),
# ignore points never enter recall, and the pooled figures stay separate from
# the paired ones so the confound the checker exists to remove is not
# reintroduced by the summary.
# ----------------------------------------------------------------------------

CHECKER = REPO_ROOT / "checks" / "real_h5" / "check_stage5_frame_failure_vs_gt_regions.py"


def test_sign_test_is_the_exact_binomial() -> None:
    # 6 pairs all in one direction: 2 * (1/2)^6 = 0.03125, exactly.
    result = exact_sign_test(negative=6, positive=0)
    assert result["num_pairs"] == 6
    assert abs(result["p_value"] - 0.03125) < 1e-12
    # A perfectly split set cannot be evidence of a direction.
    even = exact_sign_test(negative=3, positive=3)
    assert abs(even["p_value"] - 1.0) < 1e-12
    # 5 of 6 in one direction: 2 * (6+1)/64.
    assert abs(exact_sign_test(negative=5, positive=1)["p_value"] - 2 * 7 / 64) < 1e-12
    assert exact_sign_test(0, 0)["p_value"] is None
    print("  ok: the sign test returns exact binomial tails, not a normal approximation")


def frame_table_inputs(*, frames: list[dict]) -> dict:
    """Build point arrays from a compact per-frame description."""
    pixel_xy, frame_order, point_label, valid, pred = [], [], [], [], []
    for index, spec in enumerate(frames):
        for center in spec["centers"]:
            count = spec["points_per_region"]
            pixel_xy.append(np.stack([np.full(count, center[0], float), np.arange(count, dtype=float) + center[1]], axis=1))
            frame_order.append(np.full(count, index, dtype=np.int64))
            point_label.append(np.ones(count, dtype=np.int64))
            valid.append(np.ones(count, dtype=bool))
            hits = int(round(spec["recall"] * count))
            pred.append(np.array([1] * hits + [0] * (count - hits), dtype=np.uint8))
        # One background point per frame, always predicted negative.
        pixel_xy.append(np.array([[200.0, 200.0]]))
        frame_order.append(np.array([index], dtype=np.int64))
        point_label.append(np.zeros(1, dtype=np.int64))
        valid.append(np.ones(1, dtype=bool))
        pred.append(np.zeros(1, dtype=np.uint8))
    return {
        "pixel_xy": np.concatenate(pixel_xy),
        "frame_order": np.concatenate(frame_order),
        "point_label": np.concatenate(point_label),
        "valid_mask": np.concatenate(valid),
        "pred_label": np.concatenate(pred),
    }


def build(frames: list[dict], *, prob_for_positive: float = 0.4):
    arrays = frame_table_inputs(frames=frames)
    gt_data = {
        "pixel_xy": arrays["pixel_xy"],
        "frame_order": arrays["frame_order"],
        "gt_positive": arrays["valid_mask"] & (arrays["point_label"] == 1),
    }
    prob = np.where(arrays["point_label"] == 1, prob_for_positive, 0.1).astype(np.float64)
    return build_frame_table(
        gt_data=gt_data,
        point_label=arrays["point_label"],
        valid_mask=arrays["valid_mask"],
        pred_label=arrays["pred_label"],
        prob_femur=prob,
        ignore_index=-1,
        link_distance=4.0,
        min_component_points=5,
    )


def test_per_frame_recall_and_region_count_are_both_per_frame() -> None:
    rows = build(
        [
            {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.8},
            {"centers": [(10, 10), (90, 10)], "points_per_region": 10, "recall": 0.2},
        ]
    )
    assert [row["num_regions"] for row in rows] == [1, 2]
    assert abs(rows[0]["recall"] - 0.8) < 1e-9
    assert abs(rows[1]["recall"] - 0.2) < 1e-9
    assert rows[0]["gt_positive_points"] == 10 and rows[1]["gt_positive_points"] == 20
    assert rows[0]["true_positive"] == 8 and rows[1]["true_positive"] == 4
    assert all(row["mean_prob_on_gt_background"] is not None for row in rows)
    print("  ok: each frame carries its own region count and its own recall")


def test_ignore_points_never_enter_recall() -> None:
    arrays = frame_table_inputs(frames=[{"centers": [(10, 10)], "points_per_region": 10, "recall": 0.5}])
    # Turn 4 GT points into ignore points that the model calls positive. If they
    # leaked in they would change both the denominator and the TP count.
    arrays["point_label"][:4] = -1
    arrays["pred_label"][:4] = 1
    gt_data = {
        "pixel_xy": arrays["pixel_xy"],
        "frame_order": arrays["frame_order"],
        "gt_positive": arrays["valid_mask"] & (arrays["point_label"] == 1),
    }
    rows = build_frame_table(
        gt_data=gt_data,
        point_label=arrays["point_label"],
        valid_mask=arrays["valid_mask"],
        pred_label=arrays["pred_label"],
        prob_femur=np.full(arrays["point_label"].size, 0.3),
        ignore_index=-1,
        link_distance=4.0,
        min_component_points=5,
    )
    assert rows[0]["gt_positive_points"] == 6, "ignore points must leave the denominator"
    assert rows[0]["true_positive"] == 1, "the 5 predicted-positive GT points minus the 4 ignored ones"
    print("  ok: ignore points are excluded from both the numerator and the denominator")


def test_single_kind_video_is_excluded_not_counted_as_a_tie() -> None:
    only_single = within_video_region_comparison(
        build([{"centers": [(10, 10)], "points_per_region": 10, "recall": 0.5}])
    )
    assert only_single["comparable"] is False
    assert only_single["recall_median_difference"] is None
    assert only_single["multi_region"] is None

    both = within_video_region_comparison(
        build(
            [
                {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.9},
                {"centers": [(10, 10), (90, 10)], "points_per_region": 10, "recall": 0.1},
            ]
        )
    )
    assert both["comparable"] is True
    assert abs(both["recall_median_difference"] - (0.1 - 0.9)) < 1e-9
    print("  ok: a video with one kind of frame is excluded, not folded in as a zero difference")


def test_aggregate_excludes_single_kind_videos_from_the_sign_test() -> None:
    comparable = {
        "anonymous_id": "validation_001",
        "frame_rows": build(
            [
                {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.9},
                {"centers": [(10, 10), (90, 10)], "points_per_region": 10, "recall": 0.1},
            ]
        ),
    }
    single_only = {
        "anonymous_id": "validation_002",
        "frame_rows": build([{"centers": [(10, 10)], "points_per_region": 10, "recall": 0.4}]),
    }
    for video in (comparable, single_only):
        video["within_video"] = within_video_region_comparison(video["frame_rows"])
        video["localization"] = summarize_frame_localization(video["frame_rows"])

    aggregate = aggregate_paired([comparable, single_only])
    assert aggregate["videos_with_both_kinds_of_frame"] == 1
    assert aggregate["videos_excluded_as_single_kind"] == 1
    assert aggregate["sign_test"]["num_pairs"] == 1, "the excluded video must not enter the test"
    # The pooled view still sees every frame of both videos.
    assert aggregate["pooled_single_region_frames"]["num_frames"] == 2
    assert aggregate["pooled_multi_region_frames"]["num_frames"] == 1
    assert "NOT independent of the video-level confound" in aggregate["pooled_note"]
    print("  ok: single-kind videos leave the paired test but stay in the pooled view, which is labelled")


def test_pooled_recall_is_point_weighted_not_frame_averaged() -> None:
    """A big frame and a small frame must not count equally in the pooled figure;
    the two weightings disagree here, so a silent swap would be visible."""
    videos = [
        {
            "anonymous_id": "validation_001",
            "frame_rows": [
                {"num_regions": 1, "true_positive": 90, "gt_positive_points": 100, "recall": 0.9,
                 "false_positive": 0, "mean_prob_on_gt_positive": 0.6},
                {"num_regions": 1, "true_positive": 0, "gt_positive_points": 10, "recall": 0.0,
                 "false_positive": 0, "mean_prob_on_gt_positive": 0.1},
            ],
        }
    ]
    videos[0]["within_video"] = within_video_region_comparison(videos[0]["frame_rows"])
    aggregate = aggregate_paired(videos)
    pooled = aggregate["pooled_single_region_frames"]
    assert abs(pooled["point_weighted_recall"] - 90 / 110) < 1e-12
    assert abs(pooled["frame_recall_median"] - 0.45) < 1e-12
    print("  ok: the pooled recall is point-weighted, and the frame-median is reported separately")


def test_zero_recall_frames_are_counted() -> None:
    summary = summarize_frame_localization(
        build(
            [
                {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.0},
                {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.0},
                {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.6},
                {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.8},
            ]
        )
    )
    assert summary["num_gt_frames"] == 4
    assert summary["frames_with_zero_recall"] == 2
    assert abs(summary["fraction_frames_zero_recall"] - 0.5) < 1e-12
    assert abs(summary["recall_median"] - 0.3) < 1e-12
    print("  ok: frames with no true positive at all are counted separately from the median")


def test_checkpoint_comes_from_the_directory_and_a_mismatch_stops_the_run() -> None:
    """Naming one checkpoint while pointing at another's directory is the easy
    mistake here, and it was made on the first real run. The directory wins by
    default, and a contradiction is refused rather than silently resolved."""
    assert resolve_checkpoint(Path("/eval/run/last"), None) == "last"
    assert resolve_checkpoint(Path("/eval/run/best"), "") == "best"
    assert resolve_checkpoint(Path("/eval/run/last"), "last") == "last"
    try:
        resolve_checkpoint(Path("/eval/run/last"), "best")
    except ValueError as error:
        assert "refusing to guess" in str(error)
        assert "last" in str(error) and "best" in str(error)
    else:
        raise AssertionError("a checkpoint that contradicts the directory must stop the run")
    print("  ok: the checkpoint is taken from the evaluation directory, and a contradiction is refused")


def write_video(path: Path, arrays: dict) -> None:
    count = arrays["point_label"].size
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        group = f.create_group("point_cloud")
        group.create_dataset("points", data=np.zeros((count, 3), dtype=np.float32))
        group.create_dataset("intensity", data=np.zeros(count, dtype=np.float32))
        group.create_dataset("alpha", data=np.zeros(count, dtype=np.float32))
        group.create_dataset("confidence", data=np.ones(count, dtype=np.float32))
        group.create_dataset("frame_order", data=arrays["frame_order"])
        group.create_dataset("pixel_xy", data=arrays["pixel_xy"].astype(np.float32))
        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=arrays["point_label"])
        annotation.create_dataset("valid_mask", data=arrays["valid_mask"])


def test_cli_end_to_end_and_privacy_self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_dir = root / "eval" / "best"
        rows = []
        for index in (1, 2):
            arrays = frame_table_inputs(
                frames=[
                    {"centers": [(10, 10)], "points_per_region": 10, "recall": 0.9},
                    {"centers": [(10, 10), (90, 10)], "points_per_region": 10, "recall": 0.1},
                ]
            )
            # A real recording name, to prove it never reaches the outputs.
            video_name = f"20250101_1200{index:02d}_{index}"
            h5_path = root / "h5" / f"{video_name}.h5"
            write_video(h5_path, arrays)
            prediction_dir = eval_dir / "predictions" / "validation"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_dir / f"{video_name}.npz",
                point_indices=np.arange(arrays["point_label"].size, dtype=np.int64),
                prob_femur=np.where(arrays["point_label"] == 1, 0.4, 0.1).astype(np.float32),
                pred_label=arrays["pred_label"],
                vote_count=np.ones(arrays["point_label"].size, dtype=np.int32),
            )
            rows.append({"video_name": video_name, "h5_path": str(h5_path), "index": index})

        with (eval_dir / "h5_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", "split", "video_name", "h5_path"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {"checkpoint": "best", "split": "validation", "video_name": row["video_name"], "h5_path": row["h5_path"]}
                )

        map_path = root / "video_id_map_DO_NOT_SHARE.csv"
        with map_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["anonymous_id", "split", "original_h5_path"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "anonymous_id": f"validation_{row['index']:03d}",
                        "split": "validation",
                        "original_h5_path": row["h5_path"],
                    }
                )

        shareable = root / "out" / "shareable.json"
        per_frame = root / "out" / "per_frame.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--video_id_map", str(map_path),
                "--evaluation_dir", str(eval_dir),
                "--checkpoint", "best",
                "--split", "validation",
                "--shareable_json", str(shareable),
                "--per_frame_csv", str(per_frame),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(shareable.read_text(encoding="utf-8"))
        assert payload["num_videos"] == 2
        assert payload["aggregate"]["videos_with_both_kinds_of_frame"] == 2
        assert payload["aggregate"]["sign_test"]["num_multi_region_worse"] == 2
        assert payload["requires_policy_chat_judgment"] is True

        # The alias numbering is the 1-based one from the map, not a fresh
        # enumeration: that off-by-one already cost a round of confusion.
        aliases = [video["anonymous_id"] for video in payload["videos"]]
        assert aliases == ["validation_001", "validation_002"]

        text = shareable.read_text(encoding="utf-8") + per_frame.read_text(encoding="utf-8")
        for row in rows:
            assert row["video_name"] not in text, "a real recording name reached a shareable output"
        assert all(payload["privacy_self_check"].values())
    print("  ok: the CLI runs end to end, keeps the 1-based aliases, and leaks no real video name")


def test_cli_refuses_a_prediction_with_the_wrong_point_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_dir = root / "eval" / "best"
        arrays = frame_table_inputs(frames=[{"centers": [(10, 10)], "points_per_region": 10, "recall": 0.5}])
        video_name = "20250101_120001_1"
        h5_path = root / "h5" / f"{video_name}.h5"
        write_video(h5_path, arrays)
        (eval_dir / "predictions" / "validation").mkdir(parents=True, exist_ok=True)
        short = arrays["point_label"].size - 3
        np.savez_compressed(
            eval_dir / "predictions" / "validation" / f"{video_name}.npz",
            point_indices=np.arange(short, dtype=np.int64),
            prob_femur=np.full(short, 0.3, dtype=np.float32),
            pred_label=np.zeros(short, dtype=np.uint8),
            vote_count=np.ones(short, dtype=np.int32),
        )
        with (eval_dir / "h5_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", "split", "video_name", "h5_path"])
            writer.writeheader()
            writer.writerow({"checkpoint": "best", "split": "validation", "video_name": video_name, "h5_path": str(h5_path)})
        map_path = root / "map.csv"
        with map_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["anonymous_id", "split", "original_h5_path"])
            writer.writeheader()
            writer.writerow({"anonymous_id": "validation_001", "split": "validation", "original_h5_path": str(h5_path)})

        result = subprocess.run(
            [sys.executable, str(CHECKER), "--video_id_map", str(map_path), "--evaluation_dir", str(eval_dir)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode != 0, "a point-count mismatch must stop the run"
        assert "points" in result.stderr
    print("  ok: a prediction that does not match the H5 point set is refused, not aligned by truncation")


def main() -> None:
    tests = [
        test_sign_test_is_the_exact_binomial,
        test_per_frame_recall_and_region_count_are_both_per_frame,
        test_ignore_points_never_enter_recall,
        test_single_kind_video_is_excluded_not_counted_as_a_tie,
        test_aggregate_excludes_single_kind_videos_from_the_sign_test,
        test_pooled_recall_is_point_weighted_not_frame_averaged,
        test_zero_recall_frames_are_counted,
        test_checkpoint_comes_from_the_directory_and_a_mismatch_stops_the_run,
        test_cli_end_to_end_and_privacy_self_check,
        test_cli_refuses_a_prediction_with_the_wrong_point_count,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 frame-failure-vs-GT-regions synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
