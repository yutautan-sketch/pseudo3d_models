from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))

from check_stage5_gt_component_count import (  # noqa: E402
    analyze_video,
    classify_video,
    component_sizes,
    group_comparison,
    privacy_self_check,
    read_gt_points,
)

# ----------------------------------------------------------------------------
# S5-15: synthetic coverage for the mechanical GT region count that replaces
# the by-eye "the femur appears in two places" grouping.
#
# The properties that matter: a region is what single linkage says it is, the
# count genuinely depends on the link distance (so the sweep is not decorative),
# a small speck does not become a second region, ignore/background points never
# count as GT, and the recall comparison never claims causation.
# ----------------------------------------------------------------------------

CHECKER = REPO_ROOT / "checks" / "real_h5" / "check_stage5_gt_component_count.py"


def write_h5(path: Path, *, pixel_xy: np.ndarray, frame_order: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = pixel_xy.shape[0]
    with h5py.File(path, "w") as f:
        group = f.create_group("point_cloud")
        group.create_dataset("points", data=np.zeros((count, 3), dtype=np.float32))
        group.create_dataset("pixel_xy", data=pixel_xy.astype(np.float32))
        group.create_dataset("frame_order", data=frame_order.astype(np.int64))
        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=labels.astype(np.int64))
        annotation.create_dataset("valid_mask", data=valid.astype(bool))
    return path


def blob(center: tuple[float, float], *, count: int, spread: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.normal(center[0], spread, count), rng.normal(center[1], spread, count)], axis=1
    )


def test_single_linkage_counts_separated_blobs() -> None:
    left = blob((10, 10), count=30, seed=1)
    right = blob((80, 10), count=30, seed=2)
    points = np.concatenate([left, right])
    assert len(component_sizes(points, link_distance=4.0)) == 2
    # A link distance wide enough to bridge the 70px gap merges them.
    assert len(component_sizes(points, link_distance=80.0)) == 1
    assert component_sizes(np.empty((0, 2)), link_distance=4.0) == []
    assert component_sizes(np.array([[5.0, 5.0]]), link_distance=4.0) == [1]
    print("  ok: two separated blobs are two regions, and a wide enough link distance merges them")


def test_link_distance_genuinely_changes_the_count() -> None:
    """If the count were insensitive to the radius the sweep would be
    decorative; a chain of points 5px apart is one region at 6px and many at 4."""
    chain = np.stack([np.arange(0, 50, 5.0), np.zeros(10)], axis=1)
    assert len(component_sizes(chain, link_distance=6.0)) == 1
    assert len(component_sizes(chain, link_distance=4.0)) == 10
    print("  ok: the region count really does depend on the link distance")


def test_small_speck_is_not_counted_as_a_second_region() -> None:
    main = blob((20, 20), count=40, seed=3)
    speck = np.array([[90.0, 90.0], [90.5, 90.5]])
    points = np.concatenate([main, speck])
    data = {
        "pixel_xy": points,
        "frame_order": np.zeros(points.shape[0], dtype=np.int64),
        "gt_positive": np.ones(points.shape[0], dtype=bool),
    }
    with_filter = analyze_video(data, link_distance=4.0, min_component_points=5)
    without_filter = analyze_video(data, link_distance=4.0, min_component_points=1)
    assert with_filter["per_frame"][0]["num_regions"] == 1
    assert without_filter["per_frame"][0]["num_regions"] == 2
    assert with_filter["per_frame"][0]["num_regions_all"] == 2, "the unfiltered count is still recorded"
    print("  ok: a 2-point speck is excluded by min_component_points but still recorded as num_regions_all")


def test_frames_are_analyzed_independently() -> None:
    frame0 = blob((10, 10), count=20, seed=4)
    frame1 = np.concatenate([blob((10, 10), count=20, seed=5), blob((80, 80), count=20, seed=6)])
    points = np.concatenate([frame0, frame1])
    frames = np.concatenate([np.zeros(20, dtype=np.int64), np.ones(40, dtype=np.int64)])
    data = {"pixel_xy": points, "frame_order": frames, "gt_positive": np.ones(60, dtype=bool)}
    result = analyze_video(data, link_distance=4.0, min_component_points=5)
    counts = {row["frame_order"]: row["num_regions"] for row in result["per_frame"]}
    assert counts == {0: 1, 1: 2}
    assert result["multi_region_frames"] == 1
    assert result["multi_region_frame_fraction"] == 0.5
    assert result["num_gt_frames"] == 2
    print("  ok: each frame is clustered on its own; one of two frames is multi-region")


def test_only_valid_gt_positive_points_are_counted() -> None:
    positives = blob((10, 10), count=20, seed=7)
    ignored = blob((80, 80), count=20, seed=8)
    background = blob((50, 50), count=20, seed=9)
    invalid_positive = blob((30, 90), count=20, seed=10)
    points = np.concatenate([positives, ignored, background, invalid_positive])
    labels = np.concatenate([np.ones(20), -np.ones(20), np.zeros(20), np.ones(20)]).astype(np.int64)
    valid = np.concatenate([np.ones(60, dtype=bool), np.zeros(20, dtype=bool)])
    with tempfile.TemporaryDirectory() as tmp:
        path = write_h5(
            Path(tmp) / "v.h5",
            pixel_xy=points,
            frame_order=np.zeros(80, dtype=np.int64),
            labels=labels,
            valid=valid,
        )
        data = read_gt_points(path)
        assert int(data["gt_positive"].sum()) == 20, "ignore, background and invalidated points must not count"
        result = analyze_video(data, link_distance=4.0, min_component_points=5)
        assert result["per_frame"][0]["num_regions"] == 1
    print("  ok: ignore, background and valid_mask=False points are all excluded from the GT regions")


def test_classification_reports_instability_across_radii() -> None:
    stable = [
        {"link_distance": d, "multi_region_frame_fraction": 1.0} for d in (2.0, 4.0, 8.0)
    ]
    result = classify_video(stable, frame_fraction_threshold=0.5)
    assert result["stable_across_link_distances"] and result["multi_region_all"]

    flipping = [
        {"link_distance": 2.0, "multi_region_frame_fraction": 1.0},
        {"link_distance": 8.0, "multi_region_frame_fraction": 0.0},
    ]
    unstable = classify_video(flipping, frame_fraction_threshold=0.5)
    assert unstable["stable_across_link_distances"] is False
    assert unstable["multi_region_any"] and not unstable["multi_region_all"]
    print("  ok: a label that flips with the radius is reported unstable, not silently resolved")


def test_any_frame_label_is_separate_from_the_threshold_label() -> None:
    """The S5-15 data is bimodal on 'does any frame have two regions': videos
    sit at 0% of frames or at >=11%, never between. That split is reported
    independently of frame_fraction_threshold, which at its 50% default would
    have labelled only 2 of the 8 affected videos."""
    below = [{"link_distance": d, "multi_region_frame_fraction": 0.11} for d in (2.0, 12.0)]
    result = classify_video(below, frame_fraction_threshold=0.5)
    assert result["multi_region_all"] is False, "11% of frames is below the 50% threshold"
    assert result["multi_region_any_frame"] is True, "but the video does contain multi-region frames"
    assert result["multi_region_any_frame_all_radii"] is True

    none = [{"link_distance": d, "multi_region_frame_fraction": 0.0} for d in (2.0, 12.0)]
    empty = classify_video(none, frame_fraction_threshold=0.5)
    assert empty["multi_region_any_frame"] is False
    assert empty["multi_region_all"] is False

    missing = classify_video([{"link_distance": 2.0, "multi_region_frame_fraction": None}], frame_fraction_threshold=0.5)
    assert missing["multi_region_any_frame"] is False, "a video with no GT frames is not multi-region"
    print("  ok: the any-frame label separates videos the 50% threshold would have missed")


def test_group_comparison_reports_association_not_cause() -> None:
    videos = [
        {"anonymous_id": "validation_001", "classification": {"multi_region_all": True}},
        {"anonymous_id": "validation_002", "classification": {"multi_region_all": True}},
        {"anonymous_id": "validation_003", "classification": {"multi_region_all": False}},
        {"anonymous_id": "validation_004", "classification": {"multi_region_all": False}},
    ]
    metrics = {
        "validation_001": {"recall": 0.10, "f1": 0.05, "gt_positive": 4000},
        "validation_002": {"recall": 0.20, "f1": 0.06, "gt_positive": 3000},
        "validation_003": {"recall": 0.60, "f1": 0.10, "gt_positive": 2000},
        "validation_004": {"recall": 0.80, "f1": 0.12, "gt_positive": 2500},
    }
    result = group_comparison(videos, metrics, label_key="multi_region_all")
    assert result["multi_region"]["num_videos"] == 2 and result["single_region"]["num_videos"] == 2
    assert abs(result["multi_region"]["recall_median"] - 0.15) < 1e-9
    assert abs(result["single_region"]["recall_median"] - 0.70) < 1e-9
    assert abs(result["recall_median_difference"] + 0.55) < 1e-9
    assert "not a demonstrated cause" in result["note"]

    one_sided = group_comparison(videos[:2], metrics, label_key="multi_region_all")
    assert one_sided["recall_median_difference"] is None, "no difference is reported when a group is empty"
    print("  ok: group medians are computed, an empty group yields no difference, and causation is disclaimed")


def test_cli_end_to_end_with_alias_map_and_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specs = [
            ("validation_001", [(10, 10)], 0.70),
            ("validation_002", [(10, 10), (80, 80)], 0.15),
            ("validation_003", [(20, 20)], 0.60),
            ("validation_004", [(20, 20), (85, 20)], 0.10),
        ]
        map_rows = []
        metrics_rows = []
        for index, (alias, centers, recall) in enumerate(specs):
            blobs, frames = [], []
            for frame in range(4):
                for center in centers:
                    blobs.append(blob(center, count=15, seed=index * 10 + frame))
                    frames.append(np.full(15, frame, dtype=np.int64))
            points = np.concatenate(blobs)
            frame_order = np.concatenate(frames)
            h5_path = write_h5(
                root / "h5" / f"20260711_120000_{index}_video.h5",
                pixel_xy=points,
                frame_order=frame_order,
                labels=np.ones(points.shape[0], dtype=np.int64),
                valid=np.ones(points.shape[0], dtype=bool),
            )
            map_rows.append(
                {
                    "anonymous_id": alias,
                    "split": "validation",
                    "selection_role": "validation",
                    "original_video_name": h5_path.stem,
                    "original_h5_path": str(h5_path),
                }
            )
            gt = int(points.shape[0])
            metrics_rows.append(
                {
                    "checkpoint": "best",
                    "split": "validation",
                    "video_name": alias,
                    "recall": recall,
                    "f1": recall / 5,
                    "valid_positive_count": gt,
                    "true_positive_count": int(gt * recall),
                }
            )

        map_path = root / "video_id_map_DO_NOT_SHARE.csv"
        with map_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(map_rows[0]))
            writer.writeheader()
            writer.writerows(map_rows)
        metrics_path = root / "validation_h5_metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0]))
            writer.writeheader()
            writer.writerows(metrics_rows)

        result = subprocess.run(
            [
                sys.executable, str(CHECKER),
                "--video_id_map", str(map_path),
                "--link_distances", "3,6",
                "--metrics_csv", str(metrics_path),
                "--private_json", str(root / "private.json"),
                "--shareable_json", str(root / "shareable.json"),
                "--per_frame_csv", str(root / "per_frame.csv"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        private = json.loads((root / "private.json").read_text())
        shareable = json.loads((root / "shareable.json").read_text())

        labels = {v["anonymous_id"]: v["classification"]["multi_region_all"] for v in private["videos"]}
        assert labels == {
            "validation_001": False,
            "validation_002": True,
            "validation_003": False,
            "validation_004": True,
        }, labels

        comparison = private["recall_comparison"]["multi_region_all"]
        assert comparison["multi_region"]["num_videos"] == 2
        assert comparison["single_region"]["num_videos"] == 2
        assert comparison["recall_median_difference"] < 0, "the two-region group should score lower here"

        assert private["videos"][0]["original_video_name"]
        assert "original_video_name" not in shareable["videos"][0]
        assert shareable["privacy_self_check"]["timestamp_like_video_ids_absent"]
        assert shareable["privacy_self_check"]["absolute_host_paths_absent"]
        assert not privacy_self_check(private)["timestamp_like_video_ids_absent"]

        per_frame = list(csv.DictReader((root / "per_frame.csv").open()))
        assert len(per_frame) == 4 * 4 * 2, "4 videos x 4 frames x 2 link distances"
    print("  ok: the CLI classifies 4 synthetic videos, compares recall, and keeps names out of the shareable copy")


def test_cli_fails_loudly_on_a_missing_h5() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        map_path = root / "map.csv"
        with map_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["anonymous_id", "split", "selection_role", "original_video_name", "original_h5_path"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "anonymous_id": "validation_001",
                    "split": "validation",
                    "selection_role": "validation",
                    "original_video_name": "gone",
                    "original_h5_path": str(root / "absent.h5"),
                }
            )
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--video_id_map", str(map_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "H5 not found" in result.stderr
    print("  ok: a missing H5 stops the run instead of being skipped")


def main() -> None:
    tests = [
        test_single_linkage_counts_separated_blobs,
        test_link_distance_genuinely_changes_the_count,
        test_small_speck_is_not_counted_as_a_second_region,
        test_frames_are_analyzed_independently,
        test_only_valid_gt_positive_points_are_counted,
        test_classification_reports_instability_across_radii,
        test_any_frame_label_is_separate_from_the_threshold_label,
        test_group_comparison_reports_association_not_cause,
        test_cli_end_to_end_with_alias_map_and_metrics,
        test_cli_fails_loudly_on_a_missing_h5,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 GT region-count synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
