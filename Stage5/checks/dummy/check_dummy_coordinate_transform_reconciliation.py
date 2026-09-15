from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_coordinate_transform_reconciliation import (
    median_vs_diff_of_medians,
    pooled_confusion_metrics,
    read_point_metrics_csv,
    split_pooled_rows,
    verify_identity_self_diff_is_zero,
    verify_split_video_counts,
)

# ----------------------------------------------------------------------------
# S5-14補足3 synthetic pass/fail coverage for
# check_stage5_coordinate_transform_reconciliation.py -- split-pooled
# (point-weighted) confusion-derived rates vs. the mean/median of per-video
# rates (a deliberately different statistic), median-of-per-video-diffs vs.
# diff-of-split-medians for F1/IoU (two more distinct statistics), and the
# two fail-fast consistency checks (duplicate/missing video_alias, identity
# self-diff must be exactly 0). No CUDA needed -- this module is pure
# CSV/JSON re-aggregation.
# ----------------------------------------------------------------------------


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def test_pooled_confusion_metrics_is_not_the_mean_of_per_video_rates() -> None:
    """A small video (few points, FPR=100%) pooled with a huge video (many
    points, FPR=0%) must NOT average to 50% -- point-weighted pooling must
    be dominated by the huge video, unlike a naive mean of per-video rates."""
    rows = [
        {"true_positive_count": "0", "false_positive_count": "1", "true_negative_count": "0", "false_negative_count": "0"},
        {"true_positive_count": "0", "false_positive_count": "0", "true_negative_count": "999", "false_negative_count": "0"},
    ]
    pooled = pooled_confusion_metrics(rows)
    naive_mean_of_rates = (1.0 + 0.0) / 2  # what a (wrong) mean-of-per-video-FPR would give: 50%
    assert pooled["false_positive_rate"] < 0.01, pooled
    assert abs(pooled["false_positive_rate"] - naive_mean_of_rates) > 0.4
    assert pooled["false_positive_rate"] == 1 / 1000
    print(f"ok: pooled_confusion_metrics point-weights by summed counts (FPR={pooled['false_positive_rate']}), not a naive per-video-rate mean ({naive_mean_of_rates})")


def test_pooled_confusion_metrics_undefined_denominators_stay_none() -> None:
    rows = [{"true_positive_count": "0", "false_positive_count": "0", "true_negative_count": "0", "false_negative_count": "0"}]
    pooled = pooled_confusion_metrics(rows)
    assert pooled["precision"] is None and pooled["recall"] is None and pooled["false_positive_rate"] is None
    print("ok: pooled_confusion_metrics leaves all-zero-denominator rates as None, not 0 or NaN")


def make_row(*, video_alias: str, split: str, condition: str, tp: int, fp: int, tn: int, fn: int, f1: float | None, iou: float | None, precision_diff=None, recall_diff=None, fpr_diff=None, f1_diff=None, iou_diff=None, tp_zero: bool = False) -> dict[str, str]:
    def fmt(v):
        return "" if v is None else str(v)

    return {
        "video_alias": video_alias,
        "split": split,
        "condition": condition,
        "true_positive_count": str(tp),
        "false_positive_count": str(fp),
        "true_negative_count": str(tn),
        "false_negative_count": str(fn),
        "precision": "",
        "recall": "",
        "false_positive_rate": "",
        "f1": fmt(f1),
        "iou_femur": fmt(iou),
        "precision_diff": fmt(precision_diff),
        "recall_diff": fmt(recall_diff),
        "false_positive_rate_diff": fmt(fpr_diff),
        "f1_diff": fmt(f1_diff),
        "iou_femur_diff": fmt(iou_diff),
        "true_positive_count_is_zero": "True" if tp_zero else "False",
    }


def test_split_pooled_rows_diff_vs_identity() -> None:
    rows = [
        make_row(video_alias="validation_000", split="validation", condition="identity", tp=8, fp=2, tn=90, fn=0, f1=0.9, iou=0.8),
        make_row(video_alias="validation_001", split="validation", condition="identity", tp=4, fp=1, tn=95, fn=0, f1=0.85, iou=0.7),
        make_row(video_alias="validation_000", split="validation", condition="rotate_z_plus", tp=6, fp=1, tn=91, fn=2, f1=0.8, iou=0.6),
        make_row(video_alias="validation_001", split="validation", condition="rotate_z_plus", tp=4, fp=2, tn=94, fn=0, f1=0.8, iou=0.65),
    ]
    pooled = split_pooled_rows(rows)
    by_condition = {r["condition"]: r for r in pooled}
    identity_row = by_condition["identity"]
    # pooled identity: tp=12, fp=3, fn=0 -> recall = 12/12 = 1.0
    assert identity_row["true_positive_count"] == 12
    assert abs(identity_row["recall"] - 1.0) < 1e-9
    rotate_row = by_condition["rotate_z_plus"]
    # pooled rotate_z_plus: tp=10, fp=3, fn=2 -> recall = 10/12
    assert abs(rotate_row["recall"] - (10 / 12)) < 1e-9
    assert abs(rotate_row["recall_pooled_diff_vs_identity"] - (10 / 12 - 1.0)) < 1e-9
    print(f"ok: split_pooled_rows sums confusion counts before deriving rates, and diffs against the POOLED identity baseline: {rotate_row}")


def test_median_vs_diff_of_medians_are_genuinely_different_statistics() -> None:
    """Constructed so median(diffs) != diff(medians) -- the exact
    distinction the policy chat required be reported separately."""
    rows = [
        make_row(video_alias="v0", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=0.10, iou=0.10),
        make_row(video_alias="v1", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=0.50, iou=0.50),
        make_row(video_alias="v2", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=0.90, iou=0.90),
        make_row(video_alias="v0", split="validation", condition="rotate_z_plus", tp=1, fp=0, tn=1, fn=0, f1=0.20, iou=0.20),  # +0.10
        make_row(video_alias="v1", split="validation", condition="rotate_z_plus", tp=1, fp=0, tn=1, fn=0, f1=0.55, iou=0.55),  # +0.05
        make_row(video_alias="v2", split="validation", condition="rotate_z_plus", tp=1, fp=0, tn=1, fn=0, f1=0.95, iou=0.95),  # +0.05
    ]
    result = median_vs_diff_of_medians(rows, metric_key="f1")
    row = next(r for r in result if r["condition"] == "rotate_z_plus")
    # per-video diffs: [0.10, 0.05, 0.05] -> median 0.05
    assert abs(row["median_of_per_video_diffs"] - 0.05) < 1e-9, row
    # split medians: condition median(0.20,0.55,0.95)=0.55, identity median(0.10,0.50,0.90)=0.50 -> diff 0.05
    # (deliberately close here; make a second case below where they clearly diverge)
    assert row["num_videos_improved"] == 3 and row["num_videos_worsened"] == 0
    print(f"ok: median_vs_diff_of_medians computes median_of_per_video_diffs={row['median_of_per_video_diffs']} and diff_of_split_medians={row['diff_of_split_medians']} as tracked separately: {row}")

    # Now a case engineered so the two statistics clearly diverge: one video
    # improves a lot, the rest stay flat -- the median-of-diffs stays ~0
    # (dominated by the flat majority) while the diff-of-medians can shift
    # by a different amount because it medians each side independently.
    rows2 = [
        make_row(video_alias="v0", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=0.10, iou=0.10),
        make_row(video_alias="v1", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=0.50, iou=0.50),
        make_row(video_alias="v2", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=0.50, iou=0.50),
        make_row(video_alias="v3", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=0.90, iou=0.90),
        make_row(video_alias="v0", split="validation", condition="rotate_z_minus", tp=1, fp=0, tn=1, fn=0, f1=0.95, iou=0.95),  # +0.85, big jump
        make_row(video_alias="v1", split="validation", condition="rotate_z_minus", tp=1, fp=0, tn=1, fn=0, f1=0.50, iou=0.50),  # +0
        make_row(video_alias="v2", split="validation", condition="rotate_z_minus", tp=1, fp=0, tn=1, fn=0, f1=0.50, iou=0.50),  # +0
        make_row(video_alias="v3", split="validation", condition="rotate_z_minus", tp=1, fp=0, tn=1, fn=0, f1=0.90, iou=0.90),  # +0
    ]
    result2 = median_vs_diff_of_medians(rows2, metric_key="f1")
    row2 = next(r for r in result2 if r["condition"] == "rotate_z_minus")
    # per-video diffs: [0.85, 0, 0, 0] -> median 0.0 (dominated by the flat majority)
    assert abs(row2["median_of_per_video_diffs"] - 0.0) < 1e-9, row2
    # split medians: condition median(0.95,0.50,0.50,0.90)=0.70, identity median(0.10,0.50,0.50,0.90)=0.50 -> diff 0.20
    assert abs(row2["diff_of_split_medians"] - 0.20) < 1e-9, row2
    assert row2["median_of_per_video_diffs"] != row2["diff_of_split_medians"]
    print(
        f"ok: a single large per-video jump changes diff_of_split_medians ({row2['diff_of_split_medians']}) "
        f"without moving median_of_per_video_diffs ({row2['median_of_per_video_diffs']}) -- these are genuinely different statistics"
    )


def test_verify_split_video_counts_rejects_duplicate_and_missing() -> None:
    good_rows = [
        make_row(video_alias="train_sanity_000", split="train_sanity", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=1.0, iou=1.0),
        make_row(video_alias="train_sanity_001", split="train_sanity", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=1.0, iou=1.0),
    ]
    verify_split_video_counts(good_rows, expected_counts={"train_sanity": 2, "validation": 18})
    print("ok: verify_split_video_counts accepts a split/condition with exactly the expected distinct video count")

    duplicate_rows = good_rows + [
        make_row(video_alias="train_sanity_000", split="train_sanity", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=1.0, iou=1.0)
    ]
    expect_fail(
        "verify_split_video_counts rejects a duplicate video_alias row",
        lambda: verify_split_video_counts(duplicate_rows, expected_counts={"train_sanity": 2, "validation": 18}),
    )

    missing_rows = good_rows[:1]
    expect_fail(
        "verify_split_video_counts rejects a missing video (fewer than expected)",
        lambda: verify_split_video_counts(missing_rows, expected_counts={"train_sanity": 2, "validation": 18}),
    )


def test_verify_identity_self_diff_is_zero() -> None:
    ok_rows = [make_row(video_alias="v0", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=1.0, iou=1.0, f1_diff=0.0, recall_diff=0.0)]
    verify_identity_self_diff_is_zero(ok_rows)
    print("ok: verify_identity_self_diff_is_zero accepts an identity row whose own diff columns are exactly 0")

    bad_rows = [make_row(video_alias="v0", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=1.0, iou=1.0, f1_diff=0.01)]
    expect_fail(
        "verify_identity_self_diff_is_zero rejects a non-zero identity self-diff",
        lambda: verify_identity_self_diff_is_zero(bad_rows),
    )


def test_read_point_metrics_csv_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "point_metrics.csv"
        rows = [make_row(video_alias="v0", split="validation", condition="identity", tp=1, fp=0, tn=1, fn=0, f1=1.0, iou=1.0)]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        loaded = read_point_metrics_csv(path)
        assert len(loaded) == 1 and loaded[0]["video_alias"] == "v0"
        print("ok: read_point_metrics_csv round-trips a saved CSV")

        missing_path = Path(tmp) / "does_not_exist.csv"
        expect_fail("read_point_metrics_csv rejects a missing file", lambda: read_point_metrics_csv(missing_path))


def main() -> None:
    test_pooled_confusion_metrics_is_not_the_mean_of_per_video_rates()
    test_pooled_confusion_metrics_undefined_denominators_stay_none()
    test_split_pooled_rows_diff_vs_identity()
    test_median_vs_diff_of_medians_are_genuinely_different_statistics()
    test_verify_split_video_counts_rejects_duplicate_and_missing()
    test_verify_identity_self_diff_is_zero()
    test_read_point_metrics_csv_round_trip()
    print("check_dummy_coordinate_transform_reconciliation: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
