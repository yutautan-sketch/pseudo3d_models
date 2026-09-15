from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_per_window_context_diagnostics import (
    bin_exposure_disagreement_table,
    bucket_vote_count,
    canonical_predicted_class,
    classify_prediction,
    join_recurrence_with_bin_exposure,
    summarize_by_group,
    verify_h4_parity,
)

# ----------------------------------------------------------------------------
# S5-14 Step H4: synthetic pass/fail coverage for
# check_stage5_per_window_context_diagnostics.py's torch-free logic (parity
# gate against a saved W-A prediction, vote-count bucketing, stratified group
# summaries, TP/FP/FN/TN classification, per-bin exposure/disagreement
# aggregation, and the Step H3.1 recurrence cross-check join). No CUDA/torch
# needed -- this module never imports torch at module scope.
# ----------------------------------------------------------------------------


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def base_parity_kwargs() -> dict:
    labels = np.array([1, 0, 0, 1], dtype=np.int64)
    valid_mask = np.ones(4, dtype=bool)
    vote_count = np.array([2, 3, 1, 2], dtype=np.int32)
    positive_probability = np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32)
    pred_label = (positive_probability >= 0.5).astype(np.uint8)
    tp = int(np.sum(valid_mask & (labels == 1) & (pred_label == 1)))
    fp = int(np.sum(valid_mask & (labels == 0) & (pred_label == 1)))
    tn = int(np.sum(valid_mask & (labels == 0) & (pred_label == 0)))
    fn = int(np.sum(valid_mask & (labels == 1) & (pred_label == 0)))
    return {
        "vote_count": vote_count,
        "aggregated_probability_positive": positive_probability,
        "pred_label": pred_label,
        "saved_prob_femur": positive_probability.copy(),
        "saved_vote_count": vote_count.copy(),
        "saved_pred_label": pred_label.copy(),
        "labels": labels,
        "valid_mask": valid_mask,
        "training_occurrence_count": vote_count.copy(),
        "h5_metrics_row": {
            "true_positive_count": str(tp),
            "false_positive_count": str(fp),
            "true_negative_count": str(tn),
            "false_negative_count": str(fn),
        },
        "ignore_index": -1,
        "context": "validation/video_000",
    }


def test_verify_h4_parity() -> None:
    kwargs = base_parity_kwargs()
    result = verify_h4_parity(**kwargs)
    assert result["vote_count_mismatch"] == 0 and result["max_abs_prob_diff"] == 0.0, result
    print(f"ok: verify_h4_parity passes on matching data = {result}")

    bad_vote = base_parity_kwargs()
    bad_vote["saved_vote_count"] = bad_vote["saved_vote_count"].copy()
    bad_vote["saved_vote_count"][0] = 999
    expect_fail("verify_h4_parity rejects a vote_count mismatch vs saved prediction", lambda: verify_h4_parity(**bad_vote))

    bad_occurrence = base_parity_kwargs()
    bad_occurrence["training_occurrence_count"] = bad_occurrence["training_occurrence_count"].copy()
    bad_occurrence["training_occurrence_count"][0] = 999
    expect_fail(
        "verify_h4_parity rejects training-occurrence vs. this-run's vote_count mismatch",
        lambda: verify_h4_parity(**bad_occurrence),
    )

    bad_label = base_parity_kwargs()
    bad_label["saved_pred_label"] = bad_label["saved_pred_label"].copy()
    bad_label["saved_pred_label"][0] = 1 - bad_label["saved_pred_label"][0]
    expect_fail(
        "verify_h4_parity rejects a threshold-0.5 predicted-class mismatch vs saved prediction",
        lambda: verify_h4_parity(**bad_label),
    )

    bad_csv = base_parity_kwargs()
    bad_csv["h5_metrics_row"] = dict(bad_csv["h5_metrics_row"])
    bad_csv["h5_metrics_row"]["true_positive_count"] = "999"
    expect_fail(
        "verify_h4_parity rejects a tampered h5_metrics.csv aggregate row",
        lambda: verify_h4_parity(**bad_csv),
    )


def test_verify_h4_parity_diff_artifact() -> None:
    """Regression test for the real GPU-side incident in this session: a
    single-point threshold-0.5 mismatch (recomputed=8456 vs h5_metrics.csv
    saved=8455) must leave a diagnostic CSV identifying exactly which point(s)
    differ and how close they were to the 0.5 boundary, not just a bare
    AssertionError.
    """
    with tempfile.TemporaryDirectory() as tmp:
        diff_path = Path(tmp) / "parity_diff.csv"

        bad_label = base_parity_kwargs()
        bad_label["saved_pred_label"] = bad_label["saved_pred_label"].copy()
        bad_label["saved_pred_label"][0] = 1 - bad_label["saved_pred_label"][0]
        bad_label["diff_artifact_path"] = diff_path
        bad_label["frame_order"] = np.array([10, 11, 12, 13], dtype=np.int64)

        expect_fail(
            "verify_h4_parity still fails on a predicted-class mismatch when a diff_artifact_path is given",
            lambda: verify_h4_parity(**bad_label),
        )
        assert diff_path.is_file(), "expected a diff artifact CSV to be written before the fail-fast raise"
        with diff_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1, rows
        assert int(rows[0]["point_index"]) == 0
        assert int(rows[0]["frame_order"]) == 10
        print(f"ok: verify_h4_parity writes a diagnostic diff artifact on mismatch = {rows[0]}")

        # A passing case must NOT write a diff artifact (nothing to diagnose).
        clean_path = Path(tmp) / "parity_diff_clean.csv"
        ok_kwargs = base_parity_kwargs()
        ok_kwargs["diff_artifact_path"] = clean_path
        verify_h4_parity(**ok_kwargs)
        assert not clean_path.exists(), "no diff artifact should be written when parity actually passes"
        print("ok: verify_h4_parity writes no diff artifact when parity passes")


def test_canonical_predicted_class_tie_cases() -> None:
    """Step H4.1: the three boundary cases the policy chat explicitly asked
    for, confirming canonical_predicted_class() reproduces
    evaluate_stage5.predict_h5()'s exact aggregation (float64 sum -> cast to
    float32 -> 2-class argmax, tie -> index 0 = background) and that this
    differs from the naive single-class ">= 0.5" rule that caused the
    original spurious parity mismatch.
    """
    # (a) Both classes already exactly tied in float64 (and stay tied after
    # the float32 cast) -- must resolve to background (index 0), matching
    # both predict_h5()'s argmax and check_stage5_overlap_aggregation.py's
    # own test_probability_half_is_background().
    probability_sum_a = np.array([[1.0, 1.0]], dtype=np.float64)
    vote_count_a = np.array([2], dtype=np.int32)
    aggregated_a, pred_label_a = canonical_predicted_class(probability_sum_a, vote_count_a)
    assert pred_label_a.tolist() == [0], pred_label_a
    assert aggregated_a[0, 0] == aggregated_a[0, 1] == np.float32(0.5)
    print(f"ok: canonical_predicted_class exact float64 tie -> background = {aggregated_a}, pred_label={pred_label_a}")

    # (b) NOT tied in float64 (positive slightly higher, so a naive float64
    # argmax would pick positive), but the float32 cast collapses both to
    # exactly 0.5 -- canonical result must follow the POST-CAST tie rule
    # (background), not the pre-cast float64 comparison.
    background_sum = np.float64(1.0 - 1e-9)
    positive_sum = np.float64(1.0 + 1e-9)
    probability_sum_b = np.array([[background_sum, positive_sum]], dtype=np.float64)
    vote_count_b = np.array([2], dtype=np.int32)
    naive_float64_mean = probability_sum_b[0] / vote_count_b[0]
    assert naive_float64_mean[1] > naive_float64_mean[0], "test setup must have positive > background in raw float64"
    assert int(np.argmax(naive_float64_mean)) == 1, "raw float64 argmax should (incorrectly, if used directly) pick positive"

    aggregated_b, pred_label_b = canonical_predicted_class(probability_sum_b, vote_count_b)
    assert aggregated_b[0, 0] == aggregated_b[0, 1], "float32 cast should make both classes exactly equal here"
    assert pred_label_b.tolist() == [0], (
        f"canonical (post-float32-cast) argmax must pick background here, got {pred_label_b}"
    )
    print(
        f"ok: canonical_predicted_class float64-not-tied-but-float32-tied case -> background "
        f"(float64 raw argmax would have said positive) = {aggregated_b}, pred_label={pred_label_b}"
    )

    # (c) Same case (b) data, framed as the actual historical bug: a naive
    # single-class "positive_probability >= 0.5" threshold (what Step H4
    # originally did, in float64, no cast) disagrees with the canonical
    # 2-class result -- this is exactly the class of mismatch that produced
    # the spurious "GPU non-determinism" report in this session.
    naive_pred_positive = bool(naive_float64_mean[1] >= 0.5)
    canonical_pred_positive = bool(pred_label_b[0] == 1)
    assert naive_pred_positive is True and canonical_pred_positive is False, (
        "this case must reproduce the exact naive-vs-canonical disagreement that caused the Step H4.1 incident"
    )
    print(
        f"ok: naive '>= 0.5' rule (positive={naive_pred_positive}) disagrees with the canonical "
        f"2-class post-cast argmax (positive={canonical_pred_positive}) -- confirms the fix changes behavior "
        "in exactly the scenario that caused the original spurious parity mismatch"
    )


def test_bucket_vote_count() -> None:
    vote_count = np.array([1, 2, 3, 4, 5, 10], dtype=np.int32)
    buckets = bucket_vote_count(vote_count)
    assert buckets.tolist() == [1, 2, 3, 3, 4, 4], buckets.tolist()
    print(f"ok: bucket_vote_count = {buckets.tolist()}")


def test_summarize_by_group() -> None:
    group_values = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    scope_mask = np.array([True, True, True, True, False])
    disagreement = np.array([True, False, False, False, True])
    training_occurrence = np.array([2, 2, 4, 4, 4], dtype=np.int64)
    positive_vote_ratio = np.array([0.5, 0.5, 0.25, 0.25, 0.25])
    min_edge_distance = np.array([1, 1, 2, 2, 2], dtype=np.int32)

    rows = summarize_by_group(
        group_values,
        scope_mask=scope_mask,
        disagreement=disagreement,
        training_occurrence=training_occurrence,
        positive_vote_ratio=positive_vote_ratio,
        min_edge_distance=min_edge_distance,
    )
    by_group = {row["group_value"]: row for row in rows}
    assert by_group[0]["point_count"] == 2 and abs(by_group[0]["disagreement_rate"] - 0.5) < 1e-9
    assert by_group[1]["point_count"] == 2 and by_group[1]["disagreement_rate"] == 0.0  # 5th point excluded by scope_mask
    print(f"ok: summarize_by_group = {rows}")

    empty_rows = summarize_by_group(
        group_values,
        scope_mask=np.zeros(5, dtype=bool),
        disagreement=disagreement,
        training_occurrence=training_occurrence,
        positive_vote_ratio=positive_vote_ratio,
        min_edge_distance=min_edge_distance,
    )
    assert empty_rows == []
    print("ok: summarize_by_group returns no rows for an empty scope mask")


def test_classify_prediction() -> None:
    labels = np.array([1, 0, -1, 1, 0], dtype=np.int64)
    valid_mask = labels != -1
    # classify_prediction() takes the already-decided canonical pred_label,
    # not a probability array (Step H4.1: it must not re-derive predicted
    # class on its own).
    pred_label = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
    result = classify_prediction(pred_label, labels, valid_mask, ignore_index=-1)
    assert result["tp"].tolist() == [True, False, False, False, False]
    assert result["fp"].tolist() == [False, True, False, False, False]
    assert result["fn"].tolist() == [False, False, False, True, False]
    assert result["tn"].tolist() == [False, False, False, False, True]
    # the ignore-index point (index 2) must never be classified into any of TP/FP/FN/TN
    assert not any(result[key][2] for key in ("tp", "fp", "fn", "tn"))
    print(f"ok: classify_prediction TP/FP/FN/TN (ignore point excluded) = {result}")


def test_bin_exposure_disagreement_table() -> None:
    xy_normalized = np.array([[0.1, 0.1], [0.1, 0.1], [0.9, 0.9]])
    mask = np.array([True, True, True])
    training_occurrence = np.array([2, 4, 1], dtype=np.int64)
    disagreement = np.array([True, False, True])
    rows = bin_exposure_disagreement_table(
        video_alias="validation_000",
        split="validation",
        xy_normalized=xy_normalized,
        mask=mask,
        training_occurrence=training_occurrence,
        disagreement=disagreement,
        grid_resolution=2,
    )
    by_bin = {(row["bin_row"], row["bin_col"]): row for row in rows}
    assert by_bin[(0, 0)]["point_count"] == 2
    assert abs(by_bin[(0, 0)]["mean_training_window_occurrence"] - 3.0) < 1e-9
    assert abs(by_bin[(0, 0)]["disagreement_rate"] - 0.5) < 1e-9
    assert by_bin[(1, 1)]["point_count"] == 1
    print(f"ok: bin_exposure_disagreement_table = {rows}")

    empty_rows = bin_exposure_disagreement_table(
        video_alias="validation_000", split="validation", xy_normalized=xy_normalized,
        mask=np.zeros(3, dtype=bool), training_occurrence=training_occurrence,
        disagreement=disagreement, grid_resolution=2,
    )
    assert empty_rows == []
    print("ok: bin_exposure_disagreement_table returns no rows for an empty mask")


def test_join_recurrence_with_bin_exposure() -> None:
    bin_exposure_rows = [
        {"video_alias": "validation_000", "split": "validation", "grid_resolution": 16, "bin_row": 1, "bin_col": 2,
         "point_count": 10, "mean_training_window_occurrence": 5.0, "disagreement_rate": 0.4},
        {"video_alias": "validation_000", "split": "validation", "grid_resolution": 16, "bin_row": 3, "bin_col": 4,
         "point_count": 8, "mean_training_window_occurrence": 1.5, "disagreement_rate": 0.05},
    ]
    recurrence_rows = [
        {"video_alias": "validation_000", "split": "validation", "grid_resolution": "16", "bin_row": "1", "bin_col": "2",
         "multiple_unmatched_runs": "True"},
        {"video_alias": "validation_000", "split": "validation", "grid_resolution": "16", "bin_row": "3", "bin_col": "4",
         "multiple_unmatched_runs": "False"},
        {"video_alias": "validation_000", "split": "validation", "grid_resolution": "16", "bin_row": "9", "bin_col": "9",
         "multiple_unmatched_runs": "True"},  # no matching exposure row -- must be skipped, not crash
    ]
    result = join_recurrence_with_bin_exposure(bin_exposure_rows, recurrence_rows)
    assert result["bins_with_multiple_unmatched_runs"] == 1
    assert result["bins_without_multiple_unmatched_runs"] == 1
    assert abs(result["mean_training_window_occurrence_multiple_unmatched"] - 5.0) < 1e-9
    assert abs(result["mean_training_window_occurrence_other"] - 1.5) < 1e-9
    assert abs(result["mean_disagreement_rate_multiple_unmatched"] - 0.4) < 1e-9
    print(f"ok: join_recurrence_with_bin_exposure = {result}")

    empty_result = join_recurrence_with_bin_exposure([], recurrence_rows)
    assert empty_result["bins_with_multiple_unmatched_runs"] == 0
    assert empty_result["mean_training_window_occurrence_multiple_unmatched"] is None
    print("ok: join_recurrence_with_bin_exposure handles no matching exposure rows without crashing")


def main() -> None:
    test_verify_h4_parity()
    test_verify_h4_parity_diff_artifact()
    test_canonical_predicted_class_tie_cases()
    test_bucket_vote_count()
    test_summarize_by_group()
    test_classify_prediction()
    test_bin_exposure_disagreement_table()
    test_join_recurrence_with_bin_exposure()
    print("check_dummy_per_window_context_diagnostics: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
