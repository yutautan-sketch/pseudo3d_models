from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_class_weight_threshold_free import (
    average_precision,
    binary_counts,
    check_threshold_0p5_parity,
    binary_clf_curve,
    max_f1_from_curve,
    recall_at_max_fpr,
    roc_auc,
    threshold_grid_table,
)

# ----------------------------------------------------------------------------
# S5-13補足 Step 3: synthetic pass/fail coverage for
# checks/real_h5/check_stage5_class_weight_threshold_free.py's threshold-free
# curve math (AUPRC, AUROC, max F1, fixed-FPR lookup, threshold-0.5 parity
# gate) using known score/GT arrays with a hand-verified reference value.
# No CUDA/H5/real run data required.
# ----------------------------------------------------------------------------


def expect_close(label: str, actual: float, expected: float, *, tol: float = 1e-6) -> None:
    if not (np.isfinite(actual) and abs(actual - expected) <= tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")
    print(f"ok: {label} = {actual:.6f}")


def expect_nan(label: str, actual: float) -> None:
    if not np.isnan(actual):
        raise AssertionError(f"{label}: expected NaN, got {actual}")
    print(f"ok: {label} is NaN as expected")


def brute_force_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    total_positive = float(np.sum(y_true))
    if total_positive <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    ap = 0.0
    prev_recall = 0.0
    for k in range(1, y_sorted.size + 1):
        tp = float(np.sum(y_sorted[:k]))
        recall = tp / total_positive
        precision = tp / k
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def main() -> None:
    # --- classic textbook example (sklearn docs): AP = 0.8333.../AUC = 0.75 ---
    y_true = np.array([1, 0, 1, 0], dtype=np.int64)
    scores = np.array([0.9, 0.8, 0.4, 0.3], dtype=np.float64)
    thresholds, tps, fps = binary_clf_curve(y_true, scores)
    total_positive = float(tps[-1])
    total_negative = float(fps[-1])
    expect_close("AP (textbook example)", average_precision(tps, fps, total_positive), 5.0 / 6.0)
    expect_close("AUROC (textbook example)", roc_auc(tps, fps, total_positive, total_negative), 0.75)

    # --- discrimination preserved despite scores all below 0.5 (calibration
    # shift, not collapse -- exactly the S5-13補足 diagnostic question) ---
    y_true_shifted = np.array([1, 0, 1, 0], dtype=np.int64)
    scores_shifted = np.array([0.30, 0.20, 0.15, 0.05], dtype=np.float64)
    thresholds_s, tps_s, fps_s = binary_clf_curve(y_true_shifted, scores_shifted)
    ap_shifted = average_precision(tps_s, fps_s, float(tps_s[-1]))
    expect_close("AP unchanged under score shift (ranking preserved)", ap_shifted, 5.0 / 6.0)
    parity = binary_counts(
        y_true_shifted,
        np.ones(4, dtype=bool),
        scores_shifted >= 0.5,
        ignore_index=-1,
    )
    if parity["true_positive_count"] != 0 or parity["false_positive_count"] != 0:
        raise AssertionError("expected threshold-0.5 collapse (TP=FP=0) despite preserved ranking")
    print("ok: threshold-0.5 predicts all-negative even though AUPRC shows preserved discrimination")

    # --- cross-check against a brute-force O(N^2) reference on random data ---
    rng = np.random.default_rng(42)
    for trial in range(5):
        n = 20
        y_rand = (rng.random(n) < 0.3).astype(np.int64)
        scores_rand = rng.random(n)
        thresholds_r, tps_r, fps_r = binary_clf_curve(y_rand, scores_rand)
        ap_vectorized = average_precision(tps_r, fps_r, float(tps_r[-1]))
        ap_reference = brute_force_average_precision(y_rand, scores_rand)
        expect_close(f"AP vectorized vs brute-force (trial {trial})", ap_vectorized, ap_reference, tol=1e-9)

    # --- tie handling: all scores identical ---
    y_tie = np.array([1, 1, 0, 0], dtype=np.int64)
    scores_tie = np.full(4, 0.5, dtype=np.float64)
    thresholds_t, tps_t, fps_t = binary_clf_curve(y_tie, scores_tie)
    if thresholds_t.size != 1:
        raise AssertionError(f"expected exactly 1 distinct threshold for all-tied scores, got {thresholds_t.size}")
    expect_close("AP under all-tied scores == prevalence", average_precision(tps_t, fps_t, float(tps_t[-1])), 0.5)

    # --- empty positive class -> NaN, not a crash ---
    y_empty = np.zeros(5, dtype=np.int64)
    scores_empty = rng.random(5)
    thresholds_e, tps_e, fps_e = binary_clf_curve(y_empty, scores_empty)
    expect_nan("AP with no positives", average_precision(tps_e, fps_e, float(tps_e[-1])))
    expect_nan("AUROC with no positives", roc_auc(tps_e, fps_e, float(tps_e[-1]), float(fps_e[-1])))

    # --- max F1 from curve matches a direct grid search ---
    y_true2 = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=np.int64)
    scores2 = np.array([0.9, 0.7, 0.65, 0.6, 0.55, 0.4, 0.3, 0.1], dtype=np.float64)
    thresholds2, tps2, fps2 = binary_clf_curve(y_true2, scores2)
    max_f1 = max_f1_from_curve(tps2, fps2, float(tps2[-1]))
    best_f1_grid = 0.0
    for t in np.unique(scores2):
        pred = scores2 >= t
        tp = int(np.sum(pred & (y_true2 == 1)))
        fp = int(np.sum(pred & (y_true2 == 0)))
        fn = int(np.sum(~pred & (y_true2 == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        best_f1_grid = max(best_f1_grid, f1)
    expect_close("max F1 from curve vs grid search", max_f1["f1"], best_f1_grid)

    # --- fixed-FPR lookup: recall at FPR<=0 must use the strictest feasible threshold ---
    result = recall_at_max_fpr(
        thresholds2, tps2, fps2, total_positive=float(tps2[-1]), total_negative=float(fps2[-1]), max_fpr=0.0
    )
    if result is None:
        raise AssertionError("expected a feasible point at max_fpr=0.0 (highest-score threshold has fpr=0)")
    if result["fpr"] > 1e-9:
        raise AssertionError(f"recall_at_max_fpr(0.0) returned fpr={result['fpr']}, expected 0")
    print(f"ok: recall_at_max_fpr(0.0) = {result}")

    # --- threshold_grid_table sanity: threshold=0.0 includes every point ---
    grid = threshold_grid_table(y_true2, scores2, np.array([0.0, 1.01]))
    if grid[0]["predicted_positive_count"] != y_true2.size:
        raise AssertionError("threshold=0.0 should predict positive for every point")
    if grid[1]["predicted_positive_count"] != 0:
        raise AssertionError("threshold=1.01 (above every score) should predict positive for no point")
    print("ok: threshold_grid_table boundary thresholds behave as expected")

    # --- threshold-0.5 parity gate: mismatch must raise, match must pass ---
    labels = np.array([1, 0, 1, 0, -1], dtype=np.int64)
    valid_mask = np.array([True, True, True, True, False], dtype=bool)
    pred_label = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    matching_row = {
        "true_positive_count": "1",
        "false_positive_count": "0",
        "true_negative_count": "2",
        "false_negative_count": "1",
    }
    check_threshold_0p5_parity(
        labels=labels, valid_mask=valid_mask, pred_label=pred_label, row=matching_row, ignore_index=-1, context="ok-case"
    )
    print("ok: threshold-0.5 parity gate passes on matching counts")
    mismatching_row = dict(matching_row)
    mismatching_row["true_positive_count"] = "999"
    try:
        check_threshold_0p5_parity(
            labels=labels,
            valid_mask=valid_mask,
            pred_label=pred_label,
            row=mismatching_row,
            ignore_index=-1,
            context="fail-case",
        )
    except AssertionError:
        print("ok (expected fail): threshold-0.5 parity gate rejects mismatched counts")
    else:
        raise AssertionError("expected AssertionError for mismatched threshold-0.5 parity counts")

    print("check_dummy_class_weight_threshold_free: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
