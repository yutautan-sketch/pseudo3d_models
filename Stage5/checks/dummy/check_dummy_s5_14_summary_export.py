from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_s5_14_summary_export import (
    aggregate_frame_position_deciles,
    aggregate_vote_count_error_rates,
    assert_bundle_anonymous,
    merge_video_summaries,
)

# ----------------------------------------------------------------------------
# S5-14 Step H5: synthetic pass/fail coverage for
# check_stage5_s5_14_summary_export.py -- frame-position-decile aggregation
# (including frames with an undefined/None recall), video_summary merge
# (matching and mismatched key sets), vote-count-stratified error-rate
# aggregation (cross-checked by hand), and the bundle privacy self-check.
# Pure CSV/JSON, no CUDA needed.
# ----------------------------------------------------------------------------


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def test_aggregate_frame_position_deciles() -> None:
    rows = [
        {"split": "validation", "relative_frame_decile": "0", "recall": "0.8", "precision": "0.1",
         "f1": "0.18", "iou_femur": "0.1", "false_positive_rate": "0.1", "false_negative_rate": "0.2"},
        {"split": "validation", "relative_frame_decile": "0", "recall": "0.6", "precision": "0.2",
         "f1": "0.3", "iou_femur": "0.15", "false_positive_rate": "0.12", "false_negative_rate": "0.4"},
        # A frame with no valid GT positive at all -> recall is undefined (empty string), must be
        # excluded from the mean rather than treated as 0.
        {"split": "validation", "relative_frame_decile": "0", "recall": "", "precision": "0.0",
         "f1": "0.0", "iou_femur": "0.0", "false_positive_rate": "0.05", "false_negative_rate": ""},
        {"split": "validation", "relative_frame_decile": "1", "recall": "0.4", "precision": "0.3",
         "f1": "0.34", "iou_femur": "0.2", "false_positive_rate": "0.2", "false_negative_rate": "0.6"},
    ]
    result = aggregate_frame_position_deciles(rows)
    by_decile = {row["relative_frame_decile"]: row for row in result}

    decile0 = by_decile[0]
    assert decile0["frame_count"] == 3, decile0
    assert decile0["recall_available_frame_count"] == 2, decile0
    assert abs(decile0["recall_mean"] - 0.7) < 1e-9, decile0  # (0.8 + 0.6) / 2, the undefined one excluded
    assert abs(decile0["recall_median"] - 0.7) < 1e-9, decile0

    decile1 = by_decile[1]
    assert decile1["frame_count"] == 1
    assert abs(decile1["recall_mean"] - 0.4) < 1e-9

    print(f"ok: aggregate_frame_position_deciles excludes undefined (empty-string) recall from the mean = {by_decile}")


def test_aggregate_frame_position_deciles_all_undefined() -> None:
    rows = [
        {"split": "train_sanity", "relative_frame_decile": "5", "recall": "", "precision": "",
         "f1": "", "iou_femur": "", "false_positive_rate": "0.0", "false_negative_rate": ""},
    ]
    result = aggregate_frame_position_deciles(rows)
    assert result[0]["recall_mean"] is None and result[0]["recall_available_frame_count"] == 0
    print("ok: aggregate_frame_position_deciles returns None (not a crash or 0) when every value is undefined")


def test_merge_video_summaries() -> None:
    h3_rows = [
        {"video_alias": "validation_000", "split": "validation", "num_frames": "20", "num_gt_positive_frames": "5"},
        {"video_alias": "train_sanity_000", "split": "train_sanity", "num_frames": "10", "num_gt_positive_frames": "2"},
    ]
    h4_rows = [
        {"video_alias": "validation_000", "split": "validation", "point_count": "100000", "disagreement_rate": "0.1"},
        {"video_alias": "train_sanity_000", "split": "train_sanity", "point_count": "50000", "disagreement_rate": "0.05"},
    ]
    merged = merge_video_summaries(h3_rows, h4_rows)
    by_key = {(row["video_alias"], row["split"]): row for row in merged}
    assert by_key[("validation_000", "validation")]["num_frames"] == "20"
    assert by_key[("validation_000", "validation")]["point_count"] == "100000"
    assert by_key[("train_sanity_000", "train_sanity")]["disagreement_rate"] == "0.05"
    print(f"ok: merge_video_summaries combines H3 and H4 columns per (video_alias, split) = {merged}")

    mismatched_h4 = h4_rows[:1]  # missing train_sanity_000
    expect_fail(
        "merge_video_summaries rejects mismatched (video_alias, split) sets between H3 and H4",
        lambda: merge_video_summaries(h3_rows, mismatched_h4),
    )


def test_aggregate_vote_count_error_rates() -> None:
    # Mirrors the real S5-14 finding shape: FP rate roughly flat across vote
    # counts, but FP disagreement much higher than TN disagreement at vote
    # count 2.
    rows = [
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "tp", "group_value": "1", "point_count": "100", "disagreement_rate": "0.0"},
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "fn", "group_value": "1", "point_count": "150", "disagreement_rate": "0.0"},
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "fp", "group_value": "1", "point_count": "200", "disagreement_rate": "0.0"},
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "tn", "group_value": "1", "point_count": "1800", "disagreement_rate": "0.0"},
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "tp", "group_value": "2", "point_count": "300", "disagreement_rate": "0.2"},
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "fn", "group_value": "2", "point_count": "350", "disagreement_rate": "0.15"},
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "fp", "group_value": "2", "point_count": "400", "disagreement_rate": "0.34"},
        {"split": "validation", "group_name": "vote_count_bucket", "stratum": "tn", "group_value": "2", "point_count": "3600", "disagreement_rate": "0.07"},
        # A relative_frame_decile row that must be ignored by this aggregation (different group_name).
        {"split": "validation", "group_name": "relative_frame_decile", "stratum": "tp", "group_value": "5", "point_count": "999", "disagreement_rate": "0.9"},
    ]
    result = aggregate_vote_count_error_rates(rows)
    by_bucket = {row["vote_count_bucket"]: row for row in result}

    bucket1 = by_bucket["1"]
    assert abs(bucket1["recall"] - 100 / 250) < 1e-9, bucket1
    assert abs(bucket1["fp_rate_among_background"] - 200 / 2000) < 1e-9, bucket1
    assert bucket1["fp_disagreement_rate"] == 0.0

    bucket2 = by_bucket["2"]
    assert abs(bucket2["recall"] - 300 / 650) < 1e-9, bucket2
    assert abs(bucket2["fp_rate_among_background"] - 400 / 4000) < 1e-9, bucket2
    assert abs(bucket2["fp_disagreement_rate"] - 0.34) < 1e-9, bucket2
    assert abs(bucket2["tn_disagreement_rate"] - 0.07) < 1e-9, bucket2

    print(f"ok: aggregate_vote_count_error_rates matches hand-computed recall/FP-rate/disagreement-rate = {result}")


def test_assert_bundle_anonymous() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        clean_dir = Path(tmp) / "clean"
        clean_dir.mkdir()
        (clean_dir / "summary.csv").write_text("video_alias,split\nvalidation_000,validation\n", encoding="utf-8")
        result = assert_bundle_anonymous(clean_dir)
        assert result["status"] == "passed" and result["files_checked"] == 1
        print(f"ok: assert_bundle_anonymous passes on a clean bundle = {result}")

        leaky_dir = Path(tmp) / "leaky_video_id"
        leaky_dir.mkdir()
        (leaky_dir / "oops.csv").write_text("video_name\n20250626_124212_7300\n", encoding="utf-8")
        expect_fail(
            "assert_bundle_anonymous rejects a timestamp-like video ID",
            lambda: assert_bundle_anonymous(leaky_dir),
        )

        leaky_path_dir = Path(tmp) / "leaky_path"
        leaky_path_dir.mkdir()
        (leaky_path_dir / "oops.csv").write_text("h5_path\n/mnt/data/3d_projects/foo.h5\n", encoding="utf-8")
        expect_fail(
            "assert_bundle_anonymous rejects an absolute host path",
            lambda: assert_bundle_anonymous(leaky_path_dir),
        )


def main() -> None:
    test_aggregate_frame_position_deciles()
    test_aggregate_frame_position_deciles_all_undefined()
    test_merge_video_summaries()
    test_aggregate_vote_count_error_rates()
    test_assert_bundle_anonymous()
    print("check_dummy_s5_14_summary_export: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
