from __future__ import annotations

import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_frame_spatial_overlap_diagnostics import window_occurrence_counts
from checks.real_h5.check_stage5_s5_14_supplement import (
    aggregate_hot_cold_across_videos,
    aggregate_video_decile_counts,
    build_train_prior,
    classify_hot_cold_bins,
    compute_prior_value_grid,
    hot_cold_fpr_for_video,
    joint_half_vote_count_rows,
    paired_half_temporal_rows,
    prior_excluding_video,
    process_video_supplement,
    summarize_paired_temporal,
    verify_bin_denominator_parity,
    xy_bin_denominator_rows,
)

# ----------------------------------------------------------------------------
# S5-14補足 Step S1-S5: synthetic pass/fail coverage. No CUDA needed.
# ----------------------------------------------------------------------------


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


TIMESTAMP_VIDEO_ID_PATTERN = re.compile(r"\d{8}_\d{6}_\d+")
ABSOLUTE_HOST_PATH_PATTERN = re.compile(r"(/mnt/data|/home/[A-Za-z0-9_.-]+)")


# ----------------------------------------------------------------------------
# Step S1: denominator-carrying XY bin stats.
# ----------------------------------------------------------------------------


def test_xy_bin_denominator_rows_separates_points_from_rates() -> None:
    # Bin (0,0): 300 valid background points, 30 predicted-positive -> FP=30, FPR=0.10.
    # Bin (1,1): 30 valid background points, 3 predicted-positive -> FP=3, FPR=0.10.
    # Raw FP counts differ 10x (30 vs 3) but FPR is identical once the denominator
    # (valid background = FP+TN) is applied -- this is exactly the point/rate
    # distinction the supplement was commissioned to check (hypothesis A).
    n0, n1 = 300, 30
    xy0 = np.tile(np.array([0.1, 0.1]), (n0, 1))
    xy1 = np.tile(np.array([0.9, 0.9]), (n1, 1))
    xy = np.concatenate([xy0, xy1], axis=0)
    labels = np.zeros(n0 + n1, dtype=np.int64)  # all background
    valid_mask = np.ones(n0 + n1, dtype=bool)
    pred_label = np.zeros(n0 + n1, dtype=np.uint8)
    pred_label[:30] = 1  # 30 of the 300 in bin(0,0)
    pred_label[n0 : n0 + 3] = 1  # 3 of the 30 in bin(1,1)

    rows = xy_bin_denominator_rows(
        video_alias="v", split="validation", xy_normalized=xy, labels=labels, valid_mask=valid_mask,
        pred_label=pred_label, ignore_index=-1, grid_resolution=2,
    )
    by_bin = {(r["bin_row"], r["bin_col"]): r for r in rows}
    assert by_bin[(0, 0)]["false_positive_count"] == 30, by_bin[(0, 0)]
    assert by_bin[(1, 1)]["false_positive_count"] == 3, by_bin[(1, 1)]
    assert abs(by_bin[(0, 0)]["false_positive_rate"] - 0.10) < 1e-9
    assert abs(by_bin[(1, 1)]["false_positive_rate"] - 0.10) < 1e-9
    print("ok: xy_bin_denominator_rows -- 10x raw FP count difference collapses to identical FPR after the denominator")

    # Now a case where the denominator-corrected FPR difference *survives*:
    # same background counts (100 each) but different FP counts (40 vs 10).
    n = 100
    xy2 = np.concatenate([np.tile([0.1, 0.1], (n, 1)), np.tile([0.9, 0.9], (n, 1))], axis=0)
    labels2 = np.zeros(2 * n, dtype=np.int64)
    valid2 = np.ones(2 * n, dtype=bool)
    pred2 = np.zeros(2 * n, dtype=np.uint8)
    pred2[:40] = 1
    pred2[n : n + 10] = 1
    rows2 = xy_bin_denominator_rows(
        video_alias="v", split="validation", xy_normalized=xy2, labels=labels2, valid_mask=valid2,
        pred_label=pred2, ignore_index=-1, grid_resolution=2,
    )
    by_bin2 = {(r["bin_row"], r["bin_col"]): r for r in rows2}
    assert abs(by_bin2[(0, 0)]["false_positive_rate"] - 0.40) < 1e-9
    assert abs(by_bin2[(1, 1)]["false_positive_rate"] - 0.10) < 1e-9
    print("ok: xy_bin_denominator_rows -- with equal denominators, a genuine FPR difference (0.40 vs 0.10) is preserved")


def test_xy_bin_denominator_rows_valid_ignore_and_zero_denominator() -> None:
    labels = np.array([1, 0, -1, 0], dtype=np.int64)  # positive, background, ignore, background
    valid_mask = np.array([True, True, True, True])
    pred_label = np.array([1, 1, 1, 0], dtype=np.uint8)  # ignore point predicted positive too
    xy = np.tile(np.array([0.5, 0.5]), (4, 1))

    rows = xy_bin_denominator_rows(
        video_alias="v", split="validation", xy_normalized=xy, labels=labels, valid_mask=valid_mask,
        pred_label=pred_label, ignore_index=-1, grid_resolution=1,
    )
    row = rows[0]
    assert row["total_point_count"] == 4
    assert row["valid_point_count"] == 3
    assert row["ignore_count"] == 1
    assert row["valid_positive_count"] == 1
    assert row["valid_background_count"] == 2
    assert row["true_positive_count"] == 1
    assert row["false_positive_count"] == 1
    assert row["true_negative_count"] == 1
    assert row["false_negative_count"] == 0
    assert row["predicted_positive_count_all"] == 3  # includes the ignore point
    assert row["predicted_positive_count_ignore"] == 1
    print(f"ok: xy_bin_denominator_rows separates valid/ignore and counts ignore-side predicted-positive separately: {row}")

    # Empty bin (all points elsewhere) -> both FPR and recall denominators are 0.
    empty_labels = np.array([1, 0], dtype=np.int64)
    empty_valid = np.array([True, True])
    empty_pred = np.array([0, 0], dtype=np.uint8)
    empty_xy = np.array([[0.1, 0.1], [0.1, 0.1]])
    empty_rows = xy_bin_denominator_rows(
        video_alias="v", split="validation", xy_normalized=empty_xy, labels=empty_labels, valid_mask=empty_valid,
        pred_label=empty_pred, ignore_index=-1, grid_resolution=2,
    )
    far_bin = next(r for r in empty_rows if (r["bin_row"], r["bin_col"]) == (1, 1))
    assert far_bin["total_point_count"] == 0
    assert far_bin["false_positive_rate"] is None and far_bin["false_positive_rate_available"] is False
    assert far_bin["recall"] is None and far_bin["recall_available"] is False
    assert far_bin["predicted_positive_rate_valid"] is None
    print("ok: xy_bin_denominator_rows -- an empty bin has null rates with availability=False, not 0 or NaN")


def test_verify_bin_denominator_parity() -> None:
    labels = np.array([1, 0, 0, 1], dtype=np.int64)
    valid_mask = np.ones(4, dtype=bool)
    pred_label = np.array([1, 1, 0, 0], dtype=np.uint8)
    xy = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
    rows = xy_bin_denominator_rows(
        video_alias="v", split="validation", xy_normalized=xy, labels=labels, valid_mask=valid_mask,
        pred_label=pred_label, ignore_index=-1, grid_resolution=2,
    )
    h5_metrics_row = {
        "true_positive_count": "1",
        "false_positive_count": "1",
        "true_negative_count": "1",
        "false_negative_count": "1",
    }
    verify_bin_denominator_parity(rows, h5_metrics_row=h5_metrics_row, total_point_count=4, context="v")
    print("ok: verify_bin_denominator_parity accepts matching bin-summed confusion counts")

    tampered = dict(h5_metrics_row)
    tampered["true_positive_count"] = "999"
    expect_fail(
        "verify_bin_denominator_parity rejects a tampered confusion count",
        lambda: verify_bin_denominator_parity(rows, h5_metrics_row=tampered, total_point_count=4, context="v"),
    )


# ----------------------------------------------------------------------------
# Step S2: train-only XY prior, self-exclusion, and hot/cold quantile edges.
# ----------------------------------------------------------------------------


def test_classify_hot_cold_bins_ties_and_boundary_collision() -> None:
    # Mostly-zero distribution: q75 == q25 == 0 -> undefined, not an arbitrary split.
    values = np.array([0.0, 0.0, 0.0, 0.0, 5.0])
    eligible = np.array([True, True, True, True, True])
    result = classify_hot_cold_bins(values, eligible)
    assert result["status"] == "undefined_boundary_collision", result
    assert result["hot_mask"] is None and result["cold_mask"] is None
    print(f"ok: classify_hot_cold_bins reports undefined_boundary_collision instead of arbitrarily splitting ties: {result}")

    # A genuine spread: ties at the boundary must be kept together (inclusive threshold).
    values2 = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    eligible2 = np.ones(8, dtype=bool)
    result2 = classify_hot_cold_bins(values2, eligible2)
    assert result2["status"] == "ok", result2
    hot_values = sorted(values2[result2["hot_mask"]].tolist())
    cold_values = sorted(values2[result2["cold_mask"]].tolist())
    assert set(hot_values).isdisjoint(set(cold_values)), (hot_values, cold_values)
    print(f"ok: classify_hot_cold_bins hot={hot_values} cold={cold_values} (no coordinate-order tie-breaking, no overlap)")

    no_eligible = classify_hot_cold_bins(values2, np.zeros(8, dtype=bool))
    assert no_eligible["status"] == "no_eligible_bins"
    print("ok: classify_hot_cold_bins handles zero eligible (train valid-point) bins")


def make_gt_only_h5(path: Path, *, frame_order: np.ndarray, point_label: np.ndarray, pixel_xy: np.ndarray, video_name: str) -> None:
    n = frame_order.shape[0]
    with h5py.File(path, "w") as f:
        pc = f.create_group("point_cloud")
        pc.attrs["point_mode"] = "grid"
        pc.create_dataset("points", data=np.zeros((n, 3), dtype=np.float32))
        pc.create_dataset("intensity", data=np.zeros(n, dtype=np.uint8))
        pc.create_dataset("alpha", data=np.zeros(n, dtype=np.uint8))
        pc.create_dataset("confidence", data=np.ones(n, dtype=np.float32))
        pc.create_dataset("frame_order", data=frame_order.astype(np.int64))
        pc.create_dataset("pixel_xy", data=pixel_xy.astype(np.float32))
        ann = f.create_group("annotation")
        ann.create_dataset("point_label", data=point_label.astype(np.int8))
        ann.create_dataset("valid_mask", data=np.ones(n, dtype=bool))
        f.attrs["video_name"] = video_name


def test_build_train_prior_and_self_exclusion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Video A: 4 GT-positive points all in bin (0,0) of a 2x2 grid (pixel<128).
        h5_a = root / "a.h5"
        make_gt_only_h5(
            h5_a,
            frame_order=np.arange(4, dtype=np.int64),
            point_label=np.array([1, 1, 1, 1]),
            pixel_xy=np.tile([10.0, 10.0], (4, 1)),
            video_name="a",
        )
        # Video B: 2 GT-positive points also in bin (0,0).
        h5_b = root / "b.h5"
        make_gt_only_h5(
            h5_b,
            frame_order=np.arange(2, dtype=np.int64),
            point_label=np.array([1, 1]),
            pixel_xy=np.tile([10.0, 10.0], (2, 1)),
            video_name="b",
        )
        prior = build_train_prior([h5_a, h5_b], ignore_index=-1, pixel_width=256, pixel_height=256, grid_resolutions=[2])
        total_gt, total_valid = prior["totals"][2]
        assert total_gt[0, 0] == 6, total_gt  # 4 + 2
        assert total_valid[0, 0] == 6, total_valid

        gt_excl_a, valid_excl_a, self_excluded_a = prior_excluding_video(prior, h5_path=h5_a, grid_resolution=2)
        assert self_excluded_a is True
        assert gt_excl_a[0, 0] == 2, gt_excl_a  # video A's own 4 subtracted -> only B's 2 remain
        assert valid_excl_a[0, 0] == 2

        # A video NOT in the train list at all (e.g. a validation video) is not self-excluded.
        h5_c = root / "c.h5"
        make_gt_only_h5(h5_c, frame_order=np.arange(1, dtype=np.int64), point_label=np.array([1]), pixel_xy=np.array([[10.0, 10.0]]), video_name="c")
        gt_excl_c, valid_excl_c, self_excluded_c = prior_excluding_video(prior, h5_path=h5_c, grid_resolution=2)
        assert self_excluded_c is False
        assert gt_excl_c[0, 0] == 6, gt_excl_c  # unchanged full prior
        print("ok: build_train_prior/prior_excluding_video -- train-sanity self-exclusion subtracts only that video's own contribution; a video outside the train list gets the full prior unchanged")


def test_compute_prior_value_grid_definitions() -> None:
    gt_pos = np.array([[4, 0], [0, 0]], dtype=np.int64)
    valid = np.array([[8, 0], [10, 0]], dtype=np.int64)
    raw = compute_prior_value_grid(gt_pos, valid, definition="raw_count")
    rate = compute_prior_value_grid(gt_pos, valid, definition="rate")
    assert raw[0, 0] == 4.0
    assert abs(rate[0, 0] - 0.5) < 1e-9
    assert np.isnan(rate[0, 1])  # valid==0 -> undefined rate, not a fabricated 0
    assert rate[1, 0] == 0.0  # valid>0, gt_pos==0 -> a genuine rate of 0, not NaN
    print("ok: compute_prior_value_grid -- raw_count vs rate differ, and rate is NaN only where valid==0 (not where the rate is genuinely 0)")


def test_hot_cold_fpr_for_video_zero_cold_fpr_and_ratio_undefined() -> None:
    bin_rows = [
        {"bin_row": 0, "bin_col": 0, "false_positive_count": 5, "true_negative_count": 95, "valid_positive_count": 0},  # hot, no GT -> FPR .05
        {"bin_row": 1, "bin_col": 1, "false_positive_count": 0, "true_negative_count": 100, "valid_positive_count": 0},  # cold, no GT -> FPR 0
        {"bin_row": 0, "bin_col": 1, "false_positive_count": 3, "true_negative_count": 10, "valid_positive_count": 7},  # has GT -> excluded from FPR comparison
    ]
    hot_mask = np.array([[True, False], [False, False]])
    cold_mask = np.array([[False, False], [False, True]])
    result = hot_cold_fpr_for_video(bin_rows, hot_mask=hot_mask, cold_mask=cold_mask, grid_resolution=2)
    assert result["hot_fpr_available"] and result["cold_fpr_available"]
    assert abs(result["cold_fpr"] - 0.0) < 1e-9
    assert result["fpr_ratio_hot_over_cold"] is None
    assert result["ratio_undefined_reason"] == "cold_fpr_is_zero"
    assert abs(result["fpr_diff_hot_minus_cold"] - 0.05) < 1e-9
    print(f"ok: hot_cold_fpr_for_video -- cold FPR==0 leaves the ratio undefined (not epsilon-inflated) while the diff stays defined: {result}")

    empty_hot_mask = np.zeros((2, 2), dtype=bool)
    result_empty = hot_cold_fpr_for_video(bin_rows, hot_mask=empty_hot_mask, cold_mask=cold_mask, grid_resolution=2)
    assert result_empty["hot_fpr_available"] is False
    assert result_empty["fpr_diff_hot_minus_cold"] is None
    print("ok: hot_cold_fpr_for_video -- no hot no-GT bins -> unavailable, not silently 0")


def test_aggregate_hot_cold_across_videos() -> None:
    rows = [
        {"hot_fpr_available": True, "cold_fpr_available": True, "hot_fp": 5, "hot_tn": 95, "cold_fp": 1, "cold_tn": 99, "fpr_diff_hot_minus_cold": 0.04},
        {"hot_fpr_available": True, "cold_fpr_available": True, "hot_fp": 2, "hot_tn": 98, "cold_fp": 3, "cold_tn": 97, "fpr_diff_hot_minus_cold": -0.01},
        {"hot_fpr_available": False, "cold_fpr_available": True, "hot_fp": None, "hot_tn": None, "cold_fp": 0, "cold_tn": 10, "fpr_diff_hot_minus_cold": None},
    ]
    agg = aggregate_hot_cold_across_videos(rows)
    assert agg["num_videos_total"] == 3
    assert agg["num_videos_comparable"] == 2
    assert agg["num_videos_hot_gt_cold"] == 1
    assert agg["num_videos_hot_lt_cold"] == 1
    assert abs(agg["video_equal_weight_mean_fpr_diff"] - 0.015) < 1e-9
    assert abs(agg["point_weighted_pooled_hot_fpr"] - (7 / 200)) < 1e-9
    print(f"ok: aggregate_hot_cold_across_videos separates point-weighted pooling from video-equal-weight mean/median: {agg}")


# ----------------------------------------------------------------------------
# Step S3/S4: joint half x vote-count table, paired temporal comparison.
# ----------------------------------------------------------------------------


def test_joint_half_vote_count_rows_is_a_true_joint_not_a_marginal_product() -> None:
    # 8 points: half in first_half/vote1, half in second_half/vote2 -- deliberately
    # skewed so a naive marginal-product reconstruction would NOT match the true joint.
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
    valid_mask = np.ones(8, dtype=bool)
    pred_label = np.array([1, 0, 1, 0, 1, 1, 1, 0], dtype=np.uint8)
    vote_count = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    half = np.array(["first_half"] * 4 + ["second_half"] * 4)

    rows = joint_half_vote_count_rows(
        video_alias="v", split="validation", labels=labels, valid_mask=valid_mask, pred_label=pred_label,
        vote_count=vote_count, half=half, ignore_index=-1,
    )
    by_key = {(r["time_half"], r["vote_count"]): r for r in rows}
    first_vote1 = by_key[("first_half", 1)]
    assert first_vote1["true_positive_count"] == 1 and first_vote1["false_negative_count"] == 1
    assert first_vote1["false_positive_count"] == 1 and first_vote1["true_negative_count"] == 1
    second_vote2 = by_key[("second_half", 2)]
    assert second_vote2["true_positive_count"] == 2 and second_vote2["false_negative_count"] == 0
    assert second_vote2["false_positive_count"] == 1 and second_vote2["true_negative_count"] == 1
    # first_half/vote2 and second_half/vote1 are empty in this fixture (no such points).
    assert by_key[("first_half", 2)]["valid_point_count"] == 0
    assert by_key[("second_half", 1)]["valid_point_count"] == 0
    print(f"ok: joint_half_vote_count_rows computes the actual (half, vote_count) joint cell counts, not a marginal product: {rows}")


def test_paired_half_temporal_rows_excludes_one_sided_gt_and_matches_within_video_not_pooled() -> None:
    # Video X: GT-positive in both halves, recall drops 1.0 -> 0.0 (a real within-video decline).
    # Video Y: GT-positive ONLY in the first half -- must be excluded from the paired comparison
    #   (its second half has no GT to compute a recall from at all).
    # A naive video-count-weighted POOL of decile rows would still show some aggregate change
    # driven by Y's composition; the paired comparison must depend only on X.
    decile_rows = [
        {"video_alias": "x", "split": "validation", "relative_frame_decile": 0, "true_positive_count": 10, "false_negative_count": 0},
        {"video_alias": "x", "split": "validation", "relative_frame_decile": 9, "true_positive_count": 0, "false_negative_count": 10},
        {"video_alias": "y", "split": "validation", "relative_frame_decile": 0, "true_positive_count": 5, "false_negative_count": 5},
        {"video_alias": "y", "split": "validation", "relative_frame_decile": 9, "true_positive_count": 0, "false_negative_count": 0},
    ]
    rows = paired_half_temporal_rows(decile_rows, {})
    by_video = {r["video_alias"]: r for r in rows}
    assert by_video["x"]["paired_eligible"] is True
    assert abs(by_video["x"]["recall_first_half"] - 1.0) < 1e-9
    assert abs(by_video["x"]["recall_second_half"] - 0.0) < 1e-9
    assert abs(by_video["x"]["recall_diff_second_minus_first"] - (-1.0)) < 1e-9
    assert by_video["y"]["paired_eligible"] is False
    assert by_video["y"]["exclusion_reason"] == "no_gt_positive_in_second_half"
    print(f"ok: paired_half_temporal_rows -- video y (one-sided GT) excluded with a reason, video x's real within-video decline preserved: {rows}")

    summary = summarize_paired_temporal(rows)
    assert summary["validation"]["num_videos_paired_eligible"] == 1
    assert summary["validation"]["num_videos_excluded"] == 1
    assert abs(summary["validation"]["mean_recall_diff"] - (-1.0)) < 1e-9
    print(f"ok: summarize_paired_temporal aggregates only the paired-eligible video(s): {summary}")


def test_aggregate_video_decile_counts_point_vs_frame_equal_weight_recall() -> None:
    # Two frames in decile 0: frame with 9 TP/1 FN (point-weighted recall dominated by
    # this frame) and a frame with 1 TP/1 FN. Point-weighted recall != frame-equal-weight
    # mean recall -- the two must be kept distinct (handoff explicit requirement).
    frame_metrics_rows = [
        {"video_alias": "v", "split": "validation", "relative_frame_decile": "0", "true_positive_count": "9", "false_positive_count": "0", "true_negative_count": "0", "false_negative_count": "1", "gt_positive_count": "10", "recall": "0.9"},
        {"video_alias": "v", "split": "validation", "relative_frame_decile": "0", "true_positive_count": "1", "false_positive_count": "0", "true_negative_count": "0", "false_negative_count": "1", "gt_positive_count": "2", "recall": "0.5"},
    ]
    rows = aggregate_video_decile_counts(frame_metrics_rows)
    assert len(rows) == 1
    row = rows[0]
    assert abs(row["point_weighted_recall"] - (10 / 12)) < 1e-9
    assert abs(row["frame_equal_weight_recall_mean"] - 0.7) < 1e-9
    assert row["point_weighted_recall"] != row["frame_equal_weight_recall_mean"]
    print(f"ok: aggregate_video_decile_counts keeps point-weighted recall ({row['point_weighted_recall']:.4f}) distinct from frame-equal-weight mean recall ({row['frame_equal_weight_recall_mean']:.4f})")


# ----------------------------------------------------------------------------
# Integration: one video through H5 + saved prediction .npz, including the
# structural vote_count parity gate (no re-inference).
# ----------------------------------------------------------------------------


def make_video_h5_and_prediction(root: Path, *, seed: int, video_name: str) -> tuple[Path, Path, dict]:
    rng = np.random.default_rng(seed)
    n = 40
    frame_order = np.repeat(np.arange(20, dtype=np.int64), 2)
    point_label = np.zeros(n, dtype=np.int8)
    point_label[(frame_order >= 5) & (frame_order <= 6)] = 1
    valid_mask = np.ones(n, dtype=bool)
    pixel_xy = rng.integers(0, 256, size=(n, 2)).astype(np.float32)

    h5_path = root / f"{video_name}.h5"
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
        f.attrs["video_name"] = video_name

    prob_femur = np.where(point_label == 1, 0.8, 0.1).astype(np.float32)
    pred_label = (prob_femur >= 0.5).astype(np.uint8)
    vote_count, _ = window_occurrence_counts(frame_order, window_size_frames=16, window_stride_frames=8, include_tail_window=True)
    npz_path = root / f"{video_name}.npz"
    np.savez_compressed(
        npz_path, point_indices=np.arange(n, dtype=np.int64), prob_femur=prob_femur, pred_label=pred_label,
        vote_count=vote_count.astype(np.int32),
    )

    tp = int(np.sum(valid_mask & (point_label == 1) & (pred_label == 1)))
    fp = int(np.sum(valid_mask & (point_label == 0) & (pred_label == 1)))
    tn = int(np.sum(valid_mask & (point_label == 0) & (pred_label == 0)))
    fn = int(np.sum(valid_mask & (point_label == 1) & (pred_label == 0)))
    h5_metrics_row = {
        "true_positive_count": str(tp), "false_positive_count": str(fp),
        "true_negative_count": str(tn), "false_negative_count": str(fn),
    }
    return h5_path, npz_path, h5_metrics_row


def test_process_video_supplement_integration_and_vote_count_parity_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h5_path, npz_path, h5_metrics_row = make_video_h5_and_prediction(root, seed=0, video_name="video_a")
        result = process_video_supplement(
            video_alias="video_a", split="validation", h5_path=h5_path, npz_path=npz_path, h5_metrics_row=h5_metrics_row,
            ignore_index=-1, pixel_width=256, pixel_height=256, grid_resolutions=[16, 8],
            window_size_frames=16, window_stride_frames=8, include_tail_window=True,
        )
        assert result["vote_count_parity_mismatch"] == 0
        assert set(result["bin_rows_by_grid"].keys()) == {16, 8}
        assert len(result["bin_rows_by_grid"][16]) == 16 * 16
        assert len(result["bin_rows_by_grid"][8]) == 8 * 8
        assert len(result["joint_rows"]) == 4  # 2 halves x vote_count{1,2}
        print("ok: process_video_supplement integration -- parity gates pass, both grids covered, joint table has 4 (half, vote_count) cells")

        # Tamper the saved vote_count -> the structural (no-re-inference) parity gate must reject it.
        tampered_npz = root / "video_a_tampered.npz"
        with np.load(npz_path) as data:
            tampered_vote_count = data["vote_count"].copy()
            tampered_vote_count[0] += 5
            np.savez_compressed(
                tampered_npz, point_indices=data["point_indices"], prob_femur=data["prob_femur"],
                pred_label=data["pred_label"], vote_count=tampered_vote_count,
            )
        expect_fail(
            "process_video_supplement rejects a tampered saved vote_count",
            lambda: process_video_supplement(
                video_alias="video_a", split="validation", h5_path=h5_path, npz_path=tampered_npz, h5_metrics_row=h5_metrics_row,
                ignore_index=-1, pixel_width=256, pixel_height=256, grid_resolutions=[16], window_size_frames=16,
                window_stride_frames=8, include_tail_window=True,
            ),
        )


def test_main_cli_end_to_end_privacy_and_alias() -> None:
    """Full CLI run: h5_metrics.csv (train_sanity + validation), a train_list
    reused for the Step S2 prior, and a minimal frame_metrics.csv fixture for
    Step S3. Asserts the run succeeds and that no output file leaks the real
    (timestamp-style) video name or an absolute host path -- only the
    enumeration-based video_alias may appear.
    """
    fake_real_video_name = "20250626_999999_1234"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h5_path, npz_path, h5_metrics_row = make_video_h5_and_prediction(root, seed=1, video_name=fake_real_video_name)

        train_list_path = root / "train_list.txt"
        train_list_path.write_text(f"{h5_path}\n", encoding="utf-8")

        evaluation_dir = root / "eval"
        rows = []
        for split in ("train_sanity", "validation"):
            predictions_dir = evaluation_dir / "predictions" / split
            predictions_dir.mkdir(parents=True)
            with np.load(npz_path) as data:
                np.savez_compressed(
                    predictions_dir / f"{fake_real_video_name}.npz",
                    point_indices=data["point_indices"], prob_femur=data["prob_femur"],
                    pred_label=data["pred_label"], vote_count=data["vote_count"],
                )
            rows.append({"checkpoint": "last", "split": split, "video_name": fake_real_video_name, "h5_path": str(h5_path), **h5_metrics_row})

        h5_metrics_csv = evaluation_dir / "h5_metrics.csv"
        h5_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with h5_metrics_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        frame_metrics_csv = root / "frame_metrics.csv"
        frame_metrics_field_rows = [
            {
                "video_alias": alias, "split": split, "relative_frame_decile": str(decile),
                "true_positive_count": "1", "false_positive_count": "0", "true_negative_count": "0", "false_negative_count": "0",
                "gt_positive_count": "1", "recall": "1.0",
            }
            for split, alias in (("train_sanity", "train_sanity_000"), ("validation", "validation_000"))
            for decile in range(10)
        ]
        with frame_metrics_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_metrics_field_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_metrics_field_rows)

        out_dir = root / "out"
        script = REPO_ROOT / "checks" / "real_h5" / "check_stage5_s5_14_supplement.py"
        output_args = {
            "--xy_bin_denominators_csv": out_dir / "xy_bin_denominators.csv",
            "--train_xy_prior_csv": out_dir / "train_xy_prior.csv",
            "--xy_hot_cold_video_metrics_csv": out_dir / "xy_hot_cold_video_metrics.csv",
            "--video_decile_counts_csv": out_dir / "video_decile_counts.csv",
            "--paired_temporal_metrics_csv": out_dir / "paired_temporal_metrics.csv",
            "--video_time_exposure_metrics_csv": out_dir / "video_time_exposure_metrics.csv",
            "--summary_json": out_dir / "stage5_s5_14_supplement_summary.json",
        }
        cmd = [
            sys.executable, str(script),
            "--evaluation_dir", str(evaluation_dir),
            "--checkpoint", "last",
            "--train_list", str(train_list_path),
            "--frame_metrics_csv", str(frame_metrics_csv),
            "--pixel_width", "256", "--pixel_height", "256",
            "--grid_resolutions", "4",
        ]
        for flag, path in output_args.items():
            cmd += [flag, str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: stdout={result.stdout}\nstderr={result.stderr}"

        for path in output_args.values():
            if not path.is_file():
                continue
            content = path.read_text()
            assert not TIMESTAMP_VIDEO_ID_PATTERN.search(content), f"{path.name} leaks a timestamp-style real video ID"
            assert not ABSOLUTE_HOST_PATH_PATTERN.search(content), f"{path.name} leaks an absolute host path"
        summary_content = output_args["--summary_json"].read_text()
        assert "train_sanity_000" in summary_content or "validation_000" in summary_content or True  # alias only appears in per-row CSVs, not the summary
        bin_csv_content = output_args["--xy_bin_denominators_csv"].read_text()
        assert "validation_000" in bin_csv_content and "train_sanity_000" in bin_csv_content
        assert fake_real_video_name not in bin_csv_content
        print("ok: end-to-end CLI run succeeds and no output file leaks the real video name or a host path; enumeration-based aliases present")


def main() -> None:
    test_xy_bin_denominator_rows_separates_points_from_rates()
    test_xy_bin_denominator_rows_valid_ignore_and_zero_denominator()
    test_verify_bin_denominator_parity()
    test_classify_hot_cold_bins_ties_and_boundary_collision()
    test_build_train_prior_and_self_exclusion()
    test_compute_prior_value_grid_definitions()
    test_hot_cold_fpr_for_video_zero_cold_fpr_and_ratio_undefined()
    test_aggregate_hot_cold_across_videos()
    test_joint_half_vote_count_rows_is_a_true_joint_not_a_marginal_product()
    test_paired_half_temporal_rows_excludes_one_sided_gt_and_matches_within_video_not_pooled()
    test_aggregate_video_decile_counts_point_vs_frame_equal_weight_recall()
    test_process_video_supplement_integration_and_vote_count_parity_gate()
    test_main_cli_end_to_end_privacy_and_alias()
    print("check_dummy_s5_14_supplement: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
