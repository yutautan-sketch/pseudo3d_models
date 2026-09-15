from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from checks.real_h5.check_stage5_frame_spatial_overlap_diagnostics import require

# ----------------------------------------------------------------------------
# S5-14補足3: CPU-only reconciliation of check_stage5_coordinate_transform_
# diagnostics.py's already-saved `coordinate_transform_point_metrics.csv`.
# No re-inference, no new training, no new conditions/videos/seeds/
# checkpoints -- pure re-aggregation of already-saved per-video confusion
# counts and metrics, per the 2026-09-15 policy-chat request (report-to-
# policy-chat section 13).
#
# Adds two things check_stage5_coordinate_transform_diagnostics.py's
# condition_summary_by_split never computed:
#
# 1. Split-pooled (point-weighted) precision/recall/FPR/F1/IoU, derived
#    from SUMMED TP/FP/TN/FN across every video in a split -- distinct from
#    (and not approximated by) the mean/median of already-per-video rates
#    that condition_summary_by_split reports.
# 2. For F1/IoU: "median of per-video diffs" vs. "diff of split medians"
#    reported as two separate, distinctly-named statistics (median(a_i -
#    b_i) != median(a_i) - median(b_i) in general) -- a 2026-09-15
#    policy-chat review found the prior reporting conflated these.
#
# Also fail-fasts on two consistency checks the same review asked for:
# each split/condition has exactly the expected distinct-video count (no
# duplicate/missing video_alias), and identity's diff against itself is
# exactly 0 for every already-saved *_diff column.
# ----------------------------------------------------------------------------

EVALUATED_SPLITS: tuple[str, ...] = ("train_sanity", "validation")
DIFF_METRICS: tuple[str, ...] = ("f1", "iou_femur")
DIFF_COLUMNS: tuple[str, ...] = ("precision_diff", "recall_diff", "false_positive_rate_diff", "f1_diff", "iou_femur_diff")


def _parse_optional_float(value: str) -> float | None:
    if value is None or value == "" or value.lower() == "none":
        return None
    return float(value)


def read_point_metrics_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing coordinate_transform_point_metrics.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    require(bool(rows), f"Empty point_metrics.csv: {path}")
    return rows


def verify_split_video_counts(rows: list[dict[str, str]], *, expected_counts: dict[str, int]) -> None:
    """Fail-fast (13.2): every (split, condition) must contain exactly the
    expected number of DISTINCT videos, with no duplicate or missing
    `video_alias` row."""
    aliases_by_key: dict[tuple[str, str], set[str]] = {}
    row_counts_by_key: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["condition"])
        aliases_by_key.setdefault(key, set()).add(row["video_alias"])
        row_counts_by_key[key] = row_counts_by_key.get(key, 0) + 1

    for (split, condition), aliases in aliases_by_key.items():
        expected = expected_counts.get(split)
        require(expected is not None, f"Unknown split {split!r} in point_metrics.csv (expected one of {sorted(expected_counts)})")
        require(
            len(aliases) == expected,
            f"{split}/{condition}: expected {expected} distinct videos, found {len(aliases)}",
        )
        require(
            row_counts_by_key[(split, condition)] == expected,
            f"{split}/{condition}: expected {expected} rows (one per video), found "
            f"{row_counts_by_key[(split, condition)]} -- duplicate video_alias rows present",
        )


def verify_identity_self_diff_is_zero(rows: list[dict[str, str]]) -> None:
    """13.2: identity's diff against itself must be exactly 0 for every
    already-saved *_diff column and every video."""
    for row in rows:
        if row["condition"] != "identity":
            continue
        for column in DIFF_COLUMNS:
            value = _parse_optional_float(row[column])
            if value is None:
                continue
            require(
                value == 0.0,
                f"{row['split']}/{row['video_alias']}: identity's own {column} is {value!r}, expected exactly 0.0",
            )


def pooled_confusion_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Sums TP/FP/TN/FN across every given row, then derives precision/
    recall/FPR/F1/IoU from the SUMMED (point-weighted/pooled) counts --
    never from averaging already-computed per-video rates (13.2 item 1)."""
    tp = sum(int(r["true_positive_count"]) for r in rows)
    fp = sum(int(r["false_positive_count"]) for r in rows)
    tn = sum(int(r["true_negative_count"]) for r in rows)
    fn = sum(int(r["false_negative_count"]) for r in rows)
    return {
        "num_videos_pooled": len(rows),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "precision": (tp / (tp + fp)) if (tp + fp) > 0 else None,
        "recall": (tp / (tp + fn)) if (tp + fn) > 0 else None,
        "false_positive_rate": (fp / (fp + tn)) if (fp + tn) > 0 else None,
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else None,
        "iou_femur": (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else None,
    }


def split_pooled_rows(point_metrics_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in point_metrics_rows:
        by_key.setdefault((row["split"], row["condition"]), []).append(row)

    identity_pooled: dict[str, dict[str, Any]] = {
        split: pooled_confusion_metrics(by_key[(split, "identity")]) for split in EVALUATED_SPLITS if (split, "identity") in by_key
    }

    rows: list[dict[str, Any]] = []
    for (split, condition), group in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        pooled = pooled_confusion_metrics(group)
        baseline = identity_pooled.get(split, {})

        def diff(key: str) -> float | None:
            a, b = pooled.get(key), baseline.get(key)
            return (a - b) if (a is not None and b is not None) else None

        rows.append(
            {
                "split": split,
                "condition": condition,
                **pooled,
                "precision_pooled_diff_vs_identity": diff("precision"),
                "recall_pooled_diff_vs_identity": diff("recall"),
                "false_positive_rate_pooled_diff_vs_identity": diff("false_positive_rate"),
                "f1_pooled_diff_vs_identity": diff("f1"),
                "iou_femur_pooled_diff_vs_identity": diff("iou_femur"),
            }
        )
    return rows


def median_vs_diff_of_medians(point_metrics_rows: list[dict[str, str]], *, metric_key: str) -> list[dict[str, Any]]:
    """For each (split, condition): BOTH `median_of_per_video_diffs`
    (median over videos of metric_condition - metric_identity, matched by
    video_alias) and `diff_of_split_medians` (median(metric_condition) -
    median(metric_identity), each medianed independently first) -- kept as
    two separate, distinctly-named statistics (13.2 item 2)."""
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in point_metrics_rows:
        by_key.setdefault((row["split"], row["condition"]), []).append(row)

    identity_by_split: dict[str, dict[str, dict[str, str]]] = {
        split: {r["video_alias"]: r for r in by_key.get((split, "identity"), [])} for split in EVALUATED_SPLITS
    }

    rows: list[dict[str, Any]] = []
    for (split, condition), group in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        identity_rows_by_video = identity_by_split.get(split, {})
        condition_values: list[float] = []
        identity_values: list[float] = []
        per_video_diffs: list[float] = []
        for row in group:
            identity_row = identity_rows_by_video.get(row["video_alias"])
            c_val = _parse_optional_float(row[metric_key])
            i_val = _parse_optional_float(identity_row[metric_key]) if identity_row else None
            if c_val is not None:
                condition_values.append(c_val)
            if i_val is not None:
                identity_values.append(i_val)
            if c_val is not None and i_val is not None:
                per_video_diffs.append(c_val - i_val)

        rows.append(
            {
                "split": split,
                "condition": condition,
                "metric": metric_key,
                "num_videos_total": len(group),
                "num_videos_condition_value_defined": len(condition_values),
                "num_videos_identity_value_defined": len(identity_values),
                "num_videos_diff_defined": len(per_video_diffs),
                "median_of_per_video_diffs": float(np.median(per_video_diffs)) if per_video_diffs else None,
                "diff_of_split_medians": (
                    float(np.median(condition_values) - np.median(identity_values))
                    if condition_values and identity_values
                    else None
                ),
                "num_videos_improved": sum(1 for d in per_video_diffs if d > 0),
                "num_videos_worsened": sum(1 for d in per_video_diffs if d < 0),
                "num_videos_unchanged": sum(1 for d in per_video_diffs if d == 0),
                "num_videos_tp_zero": sum(1 for r in group if r["true_positive_count_is_zero"] == "True"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-14補足3: CPU-only reconciliation of check_stage5_coordinate_transform_"
        "diagnostics.py's saved point_metrics.csv -- split-pooled (point-weighted) confusion-derived "
        "precision/recall/FPR/F1/IoU, and median-of-diffs vs diff-of-medians for F1/IoU. "
        "No re-inference, no new training, no new videos/conditions."
    )
    parser.add_argument("--point_metrics_csv", required=True, help="check_stage5_coordinate_transform_diagnostics.py output")
    parser.add_argument("--expected_train_sanity_videos", type=int, default=3)
    parser.add_argument("--expected_validation_videos", type=int, default=18)
    parser.add_argument("--split_pooled_metrics_csv", default=None)
    parser.add_argument("--median_diff_metrics_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args()


def _write_csv(path_str: str | None, rows: list[dict[str, Any]]) -> None:
    if not path_str or not rows:
        return
    output = Path(path_str)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_point_metrics_csv(Path(args.point_metrics_csv))

    verify_split_video_counts(
        rows,
        expected_counts={
            "train_sanity": args.expected_train_sanity_videos,
            "validation": args.expected_validation_videos,
        },
    )
    verify_identity_self_diff_is_zero(rows)

    pooled_rows = split_pooled_rows(rows)
    diff_rows: list[dict[str, Any]] = []
    for metric_key in DIFF_METRICS:
        diff_rows.extend(median_vs_diff_of_medians(rows, metric_key=metric_key))

    _write_csv(args.split_pooled_metrics_csv, pooled_rows)
    _write_csv(args.median_diff_metrics_csv, diff_rows)

    summary = {
        "status": "passed",
        "stage": "S5-14_supplement3",
        "num_input_rows": len(rows),
        "num_split_pooled_rows": len(pooled_rows),
        "num_median_diff_rows": len(diff_rows),
        "checks": {
            "split_video_counts_verified": True,
            "identity_self_diff_verified_zero": True,
        },
    }
    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-14 supplement3 reconciliation passed.")
    print(f"input rows: {len(rows)}, split-pooled rows: {len(pooled_rows)}, median-diff rows: {len(diff_rows)}")
    print("checks: split video counts verified, identity self-diff verified exactly 0")
    if args.summary_json:
        print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
