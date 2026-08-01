from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.stage4_sampling_sweep_config import (
    Stage4SweepManifestItem,
    file_sha256,
    load_stage4_sweep_manifest,
)


MANIFEST_FIELDNAMES = (
    "video_name",
    "pseudo3d_h5",
    "voc_xml_root",
    "split",
    "enabled",
    "notes",
)


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0")))
    except (TypeError, ValueError):
        return 0


def _load_per_video_rows(
    path: Path,
    *,
    config_name: str | None,
) -> tuple[str, list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"per_video CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"per_video CSV contains no rows: {path}")
    names = sorted({(row.get("config_name") or "").strip() for row in rows})
    if config_name is None:
        if len(names) != 1:
            raise ValueError(
                "per_video CSV contains multiple configs; pass --config_name. "
                f"Available={names}"
            )
        config_name = names[0]
    selected = [
        row for row in rows if (row.get("config_name") or "").strip() == config_name
    ]
    if not selected:
        raise ValueError(f"config_name={config_name!r} not found; available={names}")
    video_names = [(row.get("video_name") or "").strip() for row in selected]
    if len(video_names) != len(set(video_names)):
        raise ValueError(f"Duplicate video rows for config_name={config_name!r}")
    return config_name, selected


def _rank_finite(
    rows: list[dict[str, str]],
    key: str,
    *,
    reverse: bool,
) -> list[dict[str, str]]:
    finite = [row for row in rows if math.isfinite(_float(row, key))]
    return sorted(
        finite,
        key=lambda row: (
            -_float(row, key) if reverse else _float(row, key),
            row["video_name"],
        ),
    )


def _hashed_order(rows: list[dict[str, str]], seed: str) -> list[dict[str, str]]:
    def key(row: dict[str, str]) -> tuple[str, str]:
        video_name = row["video_name"]
        digest = hashlib.sha256(f"{seed}:{video_name}".encode("utf-8")).hexdigest()
        return digest, video_name

    return sorted(rows, key=key)


def select_screening_videos(
    rows: list[dict[str, str]],
    *,
    target_size: int,
    seed: str,
) -> tuple[list[str], dict[str, list[str]]]:
    if target_size < 1:
        raise ValueError("target_size must be >= 1")
    if target_size > len(rows):
        target_size = len(rows)

    valid_rows = [row for row in rows if _int(row, "teacher_valid_bbox_count") > 0]
    if not valid_rows:
        raise ValueError("No video has teacher-valid BBoxes")

    reasons: dict[str, set[str]] = defaultdict(set)

    def mark(selected_rows: list[dict[str, str]], reason: str) -> None:
        for row in selected_rows:
            reasons[row["video_name"]].add(reason)

    mandatory = [
        row
        for row in valid_rows
        if _int(row, "zero_positive_bbox_count") > 0
        or (
            math.isfinite(_float(row, "positive_bbox_recall_at_16"))
            and _float(row, "positive_bbox_recall_at_16") < 1.0
        )
    ]
    mark(mandatory, "baseline_failure_or_recall16_below_1")

    # Overlapping strata deliberately emphasize hard positive-retention cases
    # while retaining high-cost/noise examples. Hash filling supplies a stable
    # broad sample without deriving a validation/test split.
    hard_quota = max(1, target_size // 4)
    cost_quota = max(1, target_size // 6)
    mark(
        _rank_finite(valid_rows, "positive_pixel_recall_min", reverse=False)[
            :hard_quota
        ],
        "lowest_positive_recall_min",
    )
    mark(
        _rank_finite(valid_rows, "positive_pixel_recall_p05", reverse=False)[
            :hard_quota
        ],
        "lowest_positive_recall_p05",
    )
    mark(
        _rank_finite(valid_rows, "positive_pixel_recall_micro", reverse=False)[
            :cost_quota
        ],
        "lowest_positive_recall_micro",
    )
    mark(
        _rank_finite(rows, "selected_ignore_ratio", reverse=True)[:cost_quota],
        "highest_selected_ignore_ratio",
    )
    mark(
        _rank_finite(rows, "points_per_frame_mean", reverse=True)[:cost_quota],
        "highest_points_per_frame",
    )

    if len(reasons) < target_size:
        remaining = [row for row in rows if row["video_name"] not in reasons]
        for row in _hashed_order(remaining, seed)[: target_size - len(reasons)]:
            reasons[row["video_name"]].add("deterministic_hash_fill")

    # Mandatory/stratified overlap normally keeps this at target_size. If the
    # union exceeds it, keep all selected hard/cost cases instead of silently
    # discarding an explicit stratum.
    selected_names = sorted(reasons)
    return selected_names, {
        name: sorted(reason_set) for name, reason_set in sorted(reasons.items())
    }


def _write_subset_manifest(
    path: Path,
    *,
    items_by_video: dict[str, Stage4SweepManifestItem],
    selected_names: list[str],
    reasons: dict[str, list[str]],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output manifest exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for name in selected_names:
            item = items_by_video[name]
            source_notes = item.notes.strip()
            selection_notes = "screening_reasons=" + "|".join(reasons[name])
            notes = f"{source_notes}; {selection_notes}" if source_notes else selection_notes
            writer.writerow(
                {
                    "video_name": item.video_name,
                    "pseudo3d_h5": str(item.pseudo3d_h5),
                    "voc_xml_root": str(item.voc_xml_root),
                    "split": item.split,
                    "enabled": "true",
                    "notes": notes,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic Phase A screening subset from a full-train "
            "baseline per_video.csv. This does not create a new data split."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per_video_csv", type=Path, required=True)
    parser.add_argument("--config_name", default=None)
    parser.add_argument("--target_size", type=int, default=36)
    parser.add_argument("--seed", default="stage4_phase_a_screening_v1")
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    items = [item for item in load_stage4_sweep_manifest(args.manifest) if item.enabled]
    if any(item.split != "train" for item in items):
        raise ValueError("Screening subset input must contain only enabled train rows")
    items_by_video = {item.video_name: item for item in items}

    config_name, per_video_rows = _load_per_video_rows(
        args.per_video_csv,
        config_name=args.config_name,
    )
    rows_by_video = {row["video_name"]: row for row in per_video_rows}
    missing = sorted(set(items_by_video) - set(rows_by_video))
    extra = sorted(set(rows_by_video) - set(items_by_video))
    if missing or extra:
        raise ValueError(
            "Manifest/per_video video mismatch: "
            f"missing_results={missing}, extra_results={extra}"
        )

    selected_names, reasons = select_screening_videos(
        per_video_rows,
        target_size=args.target_size,
        seed=args.seed,
    )
    report_path = args.report_json or args.output_manifest.with_suffix(
        ".selection.json"
    )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Selection report exists: {report_path}; pass --overwrite to replace it"
        )
    _write_subset_manifest(
        args.output_manifest,
        items_by_video=items_by_video,
        selected_names=selected_names,
        reasons=reasons,
        overwrite=args.overwrite,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    reason_counts: dict[str, int] = defaultdict(int)
    for video_reasons in reasons.values():
        for reason in video_reasons:
            reason_counts[reason] += 1
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.manifest),
        "per_video_csv": str(args.per_video_csv.resolve()),
        "per_video_csv_sha256": file_sha256(args.per_video_csv),
        "config_name": config_name,
        "seed": args.seed,
        "target_size": args.target_size,
        "source_video_count": len(items),
        "selected_video_count": len(selected_names),
        "reason_counts": dict(sorted(reason_counts.items())),
        "selected_videos": [
            {"video_name": name, "reasons": reasons[name]}
            for name in selected_names
        ],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print("Stage 4 Phase A screening subset selected")
    print(f"source_manifest : {args.manifest}")
    print(f"per_video_csv   : {args.per_video_csv}")
    print(f"config_name     : {config_name}")
    print(f"source_videos   : {len(items)}")
    print(f"target_size     : {args.target_size}")
    print(f"selected_videos : {len(selected_names)}")
    print(f"output_manifest : {args.output_manifest}")
    print(f"report_json     : {report_path}")
    print(f"reason_counts   : {dict(sorted(reason_counts.items()))}")


if __name__ == "__main__":
    main()
