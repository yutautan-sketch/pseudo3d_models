from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)


SCHEMA_VERSION = 1
ACTION = "invalidate_entire_frame"
REASON_CODE = "local_crop_fully_outside"
CROP_STATUS = "fully_outside_crop"
MANIFEST_FIELDS = (
    "schema_version",
    "video_name",
    "frame_order",
    "frame_index",
    "frame_stem",
    "expected_saved_bbox_rows",
    "expected_stray_ignore_points",
    "expected_crop_status",
    "expected_visible_fraction",
    "action",
    "reason_code",
    "recoverable",
    "notes",
)
AUDIT_FILES = (
    "audit_summary.json",
    "bbox_geometry.csv",
    "frame_metrics.csv",
    "video_summary.csv",
    "input_checksums.csv",
)
VIDEO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CropQualityManifestError(RuntimeError):
    """Raised when the fixed crop-quality evidence contract is inconsistent."""


@dataclass(frozen=True)
class CropQualityInvalidation:
    schema_version: int
    video_name: str
    frame_order: int
    frame_index: int
    frame_stem: str
    expected_saved_bbox_rows: int
    expected_stray_ignore_points: int
    expected_crop_status: str
    expected_visible_fraction: float
    action: str
    reason_code: str
    recoverable: bool
    notes: str

    @property
    def frame_key(self) -> tuple[str, int, int]:
        return self.video_name, self.frame_order, self.frame_index

    def to_row(self) -> dict[str, str]:
        return {
            "schema_version": str(self.schema_version),
            "video_name": self.video_name,
            "frame_order": str(self.frame_order),
            "frame_index": str(self.frame_index),
            "frame_stem": self.frame_stem,
            "expected_saved_bbox_rows": str(self.expected_saved_bbox_rows),
            "expected_stray_ignore_points": str(self.expected_stray_ignore_points),
            "expected_crop_status": self.expected_crop_status,
            "expected_visible_fraction": str(self.expected_visible_fraction),
            "action": self.action,
            "reason_code": self.reason_code,
            "recoverable": "true" if self.recoverable else "false",
            "notes": self.notes,
        }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CropQualityManifestError(f"CSV has no header: {path}")
        rows = [
            {key: value or "" for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]
    return list(reader.fieldnames), rows


def _integer(value: str, *, field: str, row_number: int, minimum: int = 0) -> int:
    try:
        result = int(str(value).strip())
    except ValueError as exc:
        raise CropQualityManifestError(
            f"Row {row_number}: {field} must be an integer, got {value!r}"
        ) from exc
    if result < minimum:
        raise CropQualityManifestError(
            f"Row {row_number}: {field} must be >= {minimum}, got {result}"
        )
    return result


def _boolean(value: str, *, field: str, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise CropQualityManifestError(
        f"Row {row_number}: {field} must be true/false, got {value!r}"
    )


def _floating(value: str, *, field: str, row_number: int) -> float:
    try:
        result = float(str(value).strip())
    except ValueError as exc:
        raise CropQualityManifestError(
            f"Row {row_number}: {field} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise CropQualityManifestError(
            f"Row {row_number}: {field} must be finite, got {value!r}"
        )
    return result


def _frame_stem(video: str, order: int, index: int) -> str:
    return f"{video}__fo{order:05d}__fi{index:08d}"


def load_crop_quality_invalidations(path: Path) -> list[CropQualityInvalidation]:
    fields, rows = _read_csv(path)
    if tuple(fields) != MANIFEST_FIELDS:
        raise CropQualityManifestError(
            f"Crop invalidation columns/order differ from schema: {path}"
        )
    if not rows:
        raise CropQualityManifestError(f"Crop invalidation manifest is empty: {path}")
    result: list[CropQualityInvalidation] = []
    seen: set[tuple[str, int, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        schema = _integer(row["schema_version"], field="schema_version", row_number=row_number)
        if schema != SCHEMA_VERSION:
            raise CropQualityManifestError(
                f"Row {row_number}: unsupported schema_version={schema}"
            )
        video = row["video_name"].strip()
        if not VIDEO_PATTERN.fullmatch(video):
            raise CropQualityManifestError(
                f"Row {row_number}: invalid video_name={video!r}"
            )
        order = _integer(row["frame_order"], field="frame_order", row_number=row_number)
        index = _integer(row["frame_index"], field="frame_index", row_number=row_number)
        stem = row["frame_stem"].strip()
        expected_stem = _frame_stem(video, order, index)
        if stem != expected_stem:
            raise CropQualityManifestError(
                f"Row {row_number}: frame_stem mismatch: {stem!r} != {expected_stem!r}"
            )
        bbox_rows = _integer(
            row["expected_saved_bbox_rows"],
            field="expected_saved_bbox_rows",
            row_number=row_number,
            minimum=1,
        )
        stray = _integer(
            row["expected_stray_ignore_points"],
            field="expected_stray_ignore_points",
            row_number=row_number,
            minimum=1,
        )
        crop_status = row["expected_crop_status"].strip()
        if crop_status != CROP_STATUS:
            raise CropQualityManifestError(
                f"Row {row_number}: expected_crop_status must be {CROP_STATUS!r}"
            )
        visible = _floating(
            row["expected_visible_fraction"],
            field="expected_visible_fraction",
            row_number=row_number,
        )
        if visible != 0.0:
            raise CropQualityManifestError(
                f"Row {row_number}: fully outside crop must have visible fraction 0"
            )
        action = row["action"].strip()
        reason = row["reason_code"].strip()
        if action != ACTION:
            raise CropQualityManifestError(
                f"Row {row_number}: action must be {ACTION!r}, got {action!r}"
            )
        if reason != REASON_CODE:
            raise CropQualityManifestError(
                f"Row {row_number}: reason_code must be {REASON_CODE!r}, got {reason!r}"
            )
        notes = row["notes"].strip()
        if not notes:
            raise CropQualityManifestError(f"Row {row_number}: notes must not be empty")
        item = CropQualityInvalidation(
            schema_version=schema,
            video_name=video,
            frame_order=order,
            frame_index=index,
            frame_stem=stem,
            expected_saved_bbox_rows=bbox_rows,
            expected_stray_ignore_points=stray,
            expected_crop_status=crop_status,
            expected_visible_fraction=visible,
            action=action,
            reason_code=reason,
            recoverable=_boolean(
                row["recoverable"], field="recoverable", row_number=row_number
            ),
            notes=notes,
        )
        if item.frame_key in seen:
            raise CropQualityManifestError(
                f"Row {row_number}: duplicate frame key={item.frame_key}"
            )
        seen.add(item.frame_key)
        result.append(item)
    return result


def invalidation_fingerprint(rows: Sequence[CropQualityInvalidation]) -> str:
    payload = [row.to_row() for row in sorted(rows, key=lambda item: item.frame_key)]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CropQualityManifestError(f"JSON root must be an object: {path}")
    return value


def _single_row(
    rows: Sequence[Mapping[str, str]],
    *,
    video: str,
    order: int,
    index: int,
    source: str,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if row.get("video_name") == video
        and int(row.get("frame_order", "-1")) == order
        and int(row.get("frame_index", "-1")) == index
    ]
    if len(matches) != 1:
        raise CropQualityManifestError(
            f"Expected one {source} row for {(video, order, index)}, got {len(matches)}"
        )
    return matches[0]


def _write_or_verify_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> str:
    content = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path = Path(path)
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return "verified_existing"
        if not overwrite:
            raise FileExistsError(f"Summary differs; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return "written"


def validate(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = Path(args.audit_root)
    missing = [name for name in AUDIT_FILES if not (audit_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Audit root lacks files {missing}: {audit_root}")
    audit_summary = _json(audit_root / "audit_summary.json")
    if audit_summary.get("status") != "ok" or not audit_summary.get(
        "input_files_unchanged"
    ):
        raise CropQualityManifestError("Crop audit is not a successful read-only run")

    previous = load_exclusions(args.previous_exclusions)
    current = load_exclusions(args.video_exclusions)
    previous_map = {row.video_name: row for row in previous}
    current_map = {row.video_name: row for row in current}
    lost = sorted(set(previous_map) - set(current_map))
    changed = sorted(
        video
        for video, row in previous_map.items()
        if current_map.get(video) != row
    )
    if lost or changed:
        raise CropQualityManifestError(
            f"Video exclusion v2 does not preserve v1: lost={lost}, changed={changed}"
        )
    expected_exclusions = set(args.expected_excluded_video)
    if expected_exclusions and set(current_map) != expected_exclusions:
        raise CropQualityManifestError(
            "Excluded video set mismatch: "
            f"{sorted(current_map)} != {sorted(expected_exclusions)}"
        )
    if any(row.reason_code != "local_crop_tracking_drift" for row in current):
        raise CropQualityManifestError(
            "Every v2 exclusion must use local_crop_tracking_drift"
        )

    source_manifest = load_stage4_sweep_manifest(args.source_manifest)
    source_map = {row.video_name: row for row in source_manifest}
    missing_source = sorted(set(current_map) - set(source_map))
    if missing_source:
        raise CropQualityManifestError(
            f"Excluded videos are absent from source manifest: {missing_source}"
        )

    invalidations = load_crop_quality_invalidations(args.crop_invalidations)
    overlap = sorted({row.video_name for row in invalidations} & set(current_map))
    if overlap:
        raise CropQualityManifestError(
            f"Frame invalidation must not target excluded videos: {overlap}"
        )
    missing_source = sorted(
        {row.video_name for row in invalidations} - set(source_map)
    )
    if missing_source:
        raise CropQualityManifestError(
            f"Invalidated videos are absent from source manifest: {missing_source}"
        )

    _, frame_rows = _read_csv(audit_root / "frame_metrics.csv")
    _, bbox_rows = _read_csv(audit_root / "bbox_geometry.csv")
    evidence: list[dict[str, Any]] = []
    for item in invalidations:
        frame = _single_row(
            frame_rows,
            video=item.video_name,
            order=item.frame_order,
            index=item.frame_index,
            source="frame_metrics",
        )
        boxes = [
            row
            for row in bbox_rows
            if row.get("video_name") == item.video_name
            and int(row.get("frame_order", "-1")) == item.frame_order
            and int(row.get("frame_index", "-1")) == item.frame_index
        ]
        checks = {
            "saved_bbox_rows": int(frame["saved_bbox_rows"]),
            "stray_ignore_points": int(frame["stray_ignore_points"]),
            "fully_outside_crop_bboxes": int(frame["fully_outside_crop_bboxes"]),
            "saved_degenerate_bbox_rows": int(frame["saved_degenerate_bbox_rows"]),
        }
        if checks["saved_bbox_rows"] != item.expected_saved_bbox_rows:
            raise CropQualityManifestError(
                f"Saved BBox count changed for {item.frame_key}: "
                f"{checks['saved_bbox_rows']} != {item.expected_saved_bbox_rows}"
            )
        if checks["stray_ignore_points"] != item.expected_stray_ignore_points:
            raise CropQualityManifestError(
                f"Stray count changed for {item.frame_key}: "
                f"{checks['stray_ignore_points']} != {item.expected_stray_ignore_points}"
            )
        if len(boxes) != item.expected_saved_bbox_rows:
            raise CropQualityManifestError(
                f"BBox evidence count changed for {item.frame_key}: {len(boxes)}"
            )
        if checks["fully_outside_crop_bboxes"] != len(boxes) or checks[
            "saved_degenerate_bbox_rows"
        ] != len(boxes):
            raise CropQualityManifestError(
                f"Invalidation target is not fully outside and degenerate: {item.frame_key}"
            )
        for bbox in boxes:
            visible = float(bbox["visible_fraction"])
            if bbox["crop_status"] != item.expected_crop_status or not math.isclose(
                visible, item.expected_visible_fraction, rel_tol=0.0, abs_tol=1e-12
            ):
                raise CropQualityManifestError(
                    f"Crop evidence changed for {item.frame_key}"
                )
            if bbox["saved_bbox_degenerate"].strip().lower() != "true":
                raise CropQualityManifestError(
                    f"Saved BBox is no longer degenerate: {item.frame_key}"
                )
        evidence.append({"frame_key": list(item.frame_key), **checks})

    if args.expected_invalidations is not None and len(invalidations) != args.expected_invalidations:
        raise CropQualityManifestError(
            f"Invalidation count mismatch: {len(invalidations)} != {args.expected_invalidations}"
        )
    summary = {
        "schema_version": 1,
        "status": "ok",
        "audit_root": str(audit_root.resolve()),
        "audit_summary_sha256": file_sha256(audit_root / "audit_summary.json"),
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "previous_exclusions": str(Path(args.previous_exclusions).resolve()),
        "previous_exclusions_sha256": file_sha256(args.previous_exclusions),
        "video_exclusions": str(Path(args.video_exclusions).resolve()),
        "video_exclusions_sha256": file_sha256(args.video_exclusions),
        "excluded_videos": sorted(current_map),
        "crop_invalidations": str(Path(args.crop_invalidations).resolve()),
        "crop_invalidations_sha256": file_sha256(args.crop_invalidations),
        "crop_invalidation_fingerprint": invalidation_fingerprint(invalidations),
        "invalidated_frames": len(invalidations),
        "expected_stray_points_to_remove": sum(
            row.expected_stray_ignore_points for row in invalidations
        ),
        "evidence": evidence,
        "h5_files_written": 0,
        "input_files_unchanged": True,
    }
    status = _write_or_verify_json(
        args.output_summary, summary, overwrite=args.overwrite
    )
    print("Stage 4 crop-quality manifests preflight passed.")
    print(f"  exclusions          : {len(current)}")
    print(f"  invalidated frames  : {len(invalidations)}")
    print(f"  stray to remove     : {summary['expected_stray_points_to_remove']}")
    print(f"  summary status      : {status}")
    print(f"  summary             : {args.output_summary}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate fixed Stage 4 crop-quality exclusion/invalidation manifests."
    )
    parser.add_argument("--audit_root", type=Path, required=True)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--previous_exclusions", type=Path, required=True)
    parser.add_argument("--video_exclusions", type=Path, required=True)
    parser.add_argument("--crop_invalidations", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--expected_excluded_video", action="append", default=[])
    parser.add_argument("--expected_invalidations", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    try:
        validate(build_parser().parse_args())
    except Exception as exc:
        raise SystemExit(
            f"Stage 4 crop-quality manifest preflight failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
