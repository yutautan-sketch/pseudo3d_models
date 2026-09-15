from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# ----------------------------------------------------------------------------
# S5-14 Step H5: aggregation, shareable-bundle export, and completion summary.
# Consumes the already-generated Step H2/H3/H3.1/H4 CSVs (all already
# video_alias-only) and produces the two remaining required artifacts
# (frame_position_deciles.csv, stage5_s5_14_summary.json), merges the two
# separate video_summary.csv outputs (Step H3 and Step H4) into the single
# file the handoff expects, and assembles a privacy-checked shareable bundle.
# Pure CSV/JSON processing -- no h5py/torch/CUDA.
# ----------------------------------------------------------------------------


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


TIMESTAMP_VIDEO_ID_PATTERN = re.compile(r"\d{8}_\d{6}_\d+")
ABSOLUTE_HOST_PATH_PATTERN = re.compile(r"(/mnt/data|/home/[A-Za-z0-9_.-]+)")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing input CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# frame_position_deciles.csv: aggregate Step H3's per-frame rows by
# (split, relative_frame_decile), pooled across all videos.
# ----------------------------------------------------------------------------

_FRAME_METRIC_FIELDS: tuple[str, ...] = (
    "recall",
    "precision",
    "f1",
    "iou_femur",
    "false_positive_rate",
    "false_negative_rate",
)


def _parse_optional_float(value: str) -> float | None:
    if value is None or value == "" or value.lower() == "none":
        return None
    return float(value)


def aggregate_frame_position_deciles(frame_metrics_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in frame_metrics_rows:
        key = (row["split"], int(row["relative_frame_decile"]))
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (split, decile), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        row: dict[str, Any] = {"split": split, "relative_frame_decile": decile, "frame_count": len(group_rows)}
        for field in _FRAME_METRIC_FIELDS:
            values = [v for v in (_parse_optional_float(r[field]) for r in group_rows) if v is not None]
            row[f"{field}_mean"] = float(np.mean(values)) if values else None
            row[f"{field}_median"] = float(np.median(values)) if values else None
            row[f"{field}_available_frame_count"] = len(values)
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# video_summary.csv: merge Step H3's and Step H4's separate outputs.
# ----------------------------------------------------------------------------


def merge_video_summaries(
    h3_rows: list[dict[str, str]],
    h4_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    h3_by_key = {(row["video_alias"], row["split"]): row for row in h3_rows}
    h4_by_key = {(row["video_alias"], row["split"]): row for row in h4_rows}
    require(
        set(h3_by_key) == set(h4_by_key),
        f"Step H3 and Step H4 video_summary.csv cover different (video_alias, split) sets: "
        f"only in H3={sorted(set(h3_by_key) - set(h4_by_key))[:5]}, "
        f"only in H4={sorted(set(h4_by_key) - set(h3_by_key))[:5]}",
    )
    merged: list[dict[str, Any]] = []
    for key in sorted(h3_by_key):
        row = {**h3_by_key[key]}
        h4_row = dict(h4_by_key[key])
        h4_row.pop("video_alias", None)
        h4_row.pop("split", None)
        row.update(h4_row)
        merged.append(row)
    return merged


# ----------------------------------------------------------------------------
# Vote-count-stratified error rates (reproduces the recall/FP-rate-by-
# vote_count-bucket analysis backing the hypothesis C write-up).
# ----------------------------------------------------------------------------


def aggregate_vote_count_error_rates(point_overlap_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    scoped = [row for row in point_overlap_rows if row["group_name"] == "vote_count_bucket"]
    totals: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in scoped:
        key = (row["split"], row["stratum"], row["group_value"])
        entry = totals.setdefault(key, {"point_count": 0.0, "disagreement_weighted": 0.0})
        point_count = float(row["point_count"])
        entry["point_count"] += point_count
        entry["disagreement_weighted"] += point_count * float(row["disagreement_rate"])

    counts: dict[tuple[str, str], float] = {}
    for (split, stratum, group_value), entry in totals.items():
        counts[(split, group_value, stratum)] = entry["point_count"]

    rows: list[dict[str, Any]] = []
    splits_group_values = sorted({(split, group_value) for split, _stratum, group_value in totals})
    for split, group_value in splits_group_values:
        tp = counts.get((split, group_value, "tp"), 0.0)
        fp = counts.get((split, group_value, "fp"), 0.0)
        fn = counts.get((split, group_value, "fn"), 0.0)
        tn = counts.get((split, group_value, "tn"), 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        fp_rate_among_background = fp / (fp + tn) if (fp + tn) > 0 else None
        fp_entry = totals.get((split, "fp", group_value))
        tn_entry = totals.get((split, "tn", group_value))
        fp_disagreement_rate = fp_entry["disagreement_weighted"] / fp_entry["point_count"] if fp_entry and fp_entry["point_count"] > 0 else None
        tn_disagreement_rate = tn_entry["disagreement_weighted"] / tn_entry["point_count"] if tn_entry and tn_entry["point_count"] > 0 else None
        rows.append(
            {
                "split": split,
                "vote_count_bucket": group_value,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": recall,
                "fp_rate_among_background": fp_rate_among_background,
                "fp_disagreement_rate": fp_disagreement_rate,
                "tn_disagreement_rate": tn_disagreement_rate,
            }
        )
    return rows


# ----------------------------------------------------------------------------
# Privacy self-check.
# ----------------------------------------------------------------------------


def assert_bundle_anonymous(directory: Path) -> dict[str, Any]:
    violations: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if TIMESTAMP_VIDEO_ID_PATTERN.search(text):
            violations.append(f"{path}: timestamp-like video ID pattern found")
        if ABSOLUTE_HOST_PATH_PATTERN.search(text):
            violations.append(f"{path}: absolute host path pattern found")
    require(not violations, "Shareable bundle privacy check failed:\n" + "\n".join(violations[:10]))
    return {"status": "passed", "files_checked": sum(1 for p in directory.rglob("*") if p.is_file())}


# ----------------------------------------------------------------------------
# Orchestration.
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5-14 Step H5: aggregation, shareable export, completion summary.")
    parser.add_argument("--gt_overlap_exposure_csv", required=True, help="Step H2 output")
    parser.add_argument("--frame_metrics_csv", required=True, help="Step H3 output")
    parser.add_argument("--h3_video_summary_csv", required=True, help="Step H3 output")
    parser.add_argument("--xy_density_bins_csv", required=True, help="Step H3.1 output")
    parser.add_argument("--xy_temporal_recurrence_csv", required=True, help="Step H3.1 output")
    parser.add_argument("--point_overlap_error_statistics_csv", required=True, help="Step H4 output")
    parser.add_argument("--h4_video_summary_csv", required=True, help="Step H4 output")
    parser.add_argument("--h4_summary_json", required=True, help="Step H4 summary.json (for the recurrence cross-check)")
    parser.add_argument("--xy_coordinate_provenance_summary_json", required=True, help="Step H2.5 shareable summary")
    parser.add_argument("--output_dir", required=True, help="Directory to write the shareable S5-14 bundle into")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_overlap_exposure_rows = read_csv_rows(Path(args.gt_overlap_exposure_csv))
    frame_metrics_rows = read_csv_rows(Path(args.frame_metrics_csv))
    h3_video_summary_rows = read_csv_rows(Path(args.h3_video_summary_csv))
    xy_density_bins_rows = read_csv_rows(Path(args.xy_density_bins_csv))
    xy_temporal_recurrence_rows = read_csv_rows(Path(args.xy_temporal_recurrence_csv))
    point_overlap_rows = read_csv_rows(Path(args.point_overlap_error_statistics_csv))
    h4_video_summary_rows = read_csv_rows(Path(args.h4_video_summary_csv))
    h4_summary = json.loads(Path(args.h4_summary_json).read_text(encoding="utf-8"))
    provenance_summary = json.loads(Path(args.xy_coordinate_provenance_summary_json).read_text(encoding="utf-8"))

    frame_position_deciles = aggregate_frame_position_deciles(frame_metrics_rows)
    merged_video_summary = merge_video_summaries(h3_video_summary_rows, h4_video_summary_rows)
    vote_count_error_rates = aggregate_vote_count_error_rates(point_overlap_rows)

    # Copy already-anonymized inputs through unchanged, plus the two newly
    # derived artifacts, matching the handoff's required file list exactly.
    for source_path, dest_name in (
        (Path(args.gt_overlap_exposure_csv), "gt_overlap_exposure.csv"),
        (Path(args.frame_metrics_csv), "frame_metrics.csv"),
        (Path(args.xy_density_bins_csv), "xy_density_bins.csv"),
        (Path(args.xy_temporal_recurrence_csv), "xy_temporal_recurrence.csv"),
        (Path(args.point_overlap_error_statistics_csv), "point_overlap_error_statistics.csv"),
    ):
        shutil.copyfile(source_path, output_dir / dest_name)

    write_csv_rows(output_dir / "frame_position_deciles.csv", frame_position_deciles)
    write_csv_rows(output_dir / "video_summary.csv", merged_video_summary)
    write_csv_rows(output_dir / "vote_count_error_rates.csv", vote_count_error_rates)

    summary = {
        "status": "passed",
        "stage": "S5-14",
        "num_videos": len(merged_video_summary),
        "step_h2_gt_exposure_rows": len(gt_overlap_exposure_rows),
        "step_h3_frame_metrics_rows": len(frame_metrics_rows),
        "step_h3_1_xy_density_bins_rows": len(xy_density_bins_rows),
        "step_h3_1_xy_temporal_recurrence_rows": len(xy_temporal_recurrence_rows),
        "step_h4_point_overlap_error_statistics_rows": len(point_overlap_rows),
        "step_h4_parity": {
            "max_abs_prob_diff_overall": h4_summary.get("max_abs_prob_diff_overall"),
            "num_videos": h4_summary.get("num_videos"),
        },
        "step_h4_recurrence_exposure_cross_check": h4_summary.get("recurrence_cross_check"),
        "step_h2_5_xy_coordinate_provenance": {
            "resolved_dimension_count": provenance_summary.get("resolved_dimension_count"),
            "num_videos": provenance_summary.get("num_videos"),
            "policy_option": (provenance_summary.get("policy") or {}).get("option"),
        },
        "hypothesis_verdicts": {
            "A_coordinate_prior": "strongly_suggested_normalization_pending",
            "B_temporal_position": "clearly_supported",
            "C_overlap_exposure": "weakly_supported_not_primary",
        },
        "files": [
            "gt_overlap_exposure.csv",
            "frame_metrics.csv",
            "frame_position_deciles.csv",
            "xy_density_bins.csv",
            "xy_temporal_recurrence.csv",
            "point_overlap_error_statistics.csv",
            "video_summary.csv",
            "vote_count_error_rates.csv",
            "stage5_s5_14_summary.json",
        ],
    }
    with (output_dir / "stage5_s5_14_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    privacy_result = assert_bundle_anonymous(output_dir)

    print("Stage5 S5-14 Step H5 summary export passed.")
    print(f"output dir: {output_dir}")
    print(f"videos: {summary['num_videos']}")
    print(f"privacy check: {privacy_result}")
    print(f"files written: {summary['files']}")


if __name__ == "__main__":
    main()
