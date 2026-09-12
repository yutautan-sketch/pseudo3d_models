from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


MANIFEST_REQUIRED_COLUMNS = (
    "video_name",
    "pseudo3d_h5",
    "voc_xml_root",
    "split",
    "enabled",
    "notes",
)
EXCLUSION_COLUMNS = (
    "video_name",
    "reason_code",
    "evidence_frames",
    "scope",
    "recoverable",
    "notes",
)
VIDEO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REASON_CODE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class Exclusion:
    video_name: str
    reason_code: str
    evidence_frames: str
    scope: str
    recoverable: bool
    notes: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str, *, field: str, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Row {row_number}: {field} must be true/false, got {value!r}"
    )


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader if any((value or "").strip() for value in row.values())]
    return fieldnames, rows


def load_exclusions(path: Path) -> list[Exclusion]:
    fieldnames, rows = _read_rows(path)
    missing = [name for name in EXCLUSION_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f"Exclusion CSV missing columns {missing}: {path}")
    if not rows:
        raise ValueError(f"Exclusion CSV has no rows: {path}")

    exclusions: list[Exclusion] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        video_name = (row.get("video_name") or "").strip()
        reason_code = (row.get("reason_code") or "").strip()
        evidence_frames = (row.get("evidence_frames") or "").strip()
        scope = (row.get("scope") or "").strip()
        notes = (row.get("notes") or "").strip()
        if not VIDEO_NAME_PATTERN.fullmatch(video_name):
            raise ValueError(f"Row {row_number}: invalid video_name={video_name!r}")
        if video_name in seen:
            raise ValueError(f"Row {row_number}: duplicate video_name={video_name!r}")
        seen.add(video_name)
        if not REASON_CODE_PATTERN.fullmatch(reason_code):
            raise ValueError(f"Row {row_number}: invalid reason_code={reason_code!r}")
        if scope != "stage4_teacher":
            raise ValueError(
                f"Row {row_number}: scope must be 'stage4_teacher', got {scope!r}"
            )
        if not notes:
            raise ValueError(f"Row {row_number}: notes must not be empty")
        exclusions.append(
            Exclusion(
                video_name=video_name,
                reason_code=reason_code,
                evidence_frames=evidence_frames,
                scope=scope,
                recoverable=_parse_bool(
                    row.get("recoverable") or "",
                    field="recoverable",
                    row_number=row_number,
                ),
                notes=notes,
            )
        )
    return exclusions


def build_excluded_manifest(
    source_manifest: Path,
    exclusions: Sequence[Exclusion],
) -> tuple[list[str], list[dict[str, str]], int]:
    fieldnames, source_rows = _read_rows(source_manifest)
    missing = [name for name in MANIFEST_REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f"Source manifest missing columns {missing}: {source_manifest}")
    if not source_rows:
        raise ValueError(f"Source manifest has no rows: {source_manifest}")

    by_name: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(source_rows, start=2):
        video_name = (row.get("video_name") or "").strip()
        if not VIDEO_NAME_PATTERN.fullmatch(video_name):
            raise ValueError(
                f"Source manifest row {row_number}: invalid video_name={video_name!r}"
            )
        if video_name in by_name:
            raise ValueError(f"Source manifest duplicate video_name={video_name!r}")
        _parse_bool(
            row.get("enabled") or "",
            field="enabled",
            row_number=row_number,
        )
        by_name[video_name] = row

    missing_videos = sorted(
        exclusion.video_name
        for exclusion in exclusions
        if exclusion.video_name not in by_name
    )
    if missing_videos:
        raise ValueError(f"Excluded videos not found in source manifest: {missing_videos}")

    exclusion_map = {row.video_name: row for row in exclusions}
    output_rows: list[dict[str, str]] = []
    for source in source_rows:
        row = dict(source)
        video_name = row["video_name"].strip()
        exclusion = exclusion_map.get(video_name)
        if exclusion is not None:
            row["enabled"] = "false"
            evidence = exclusion.evidence_frames or "unspecified"
            row["notes"] = (
                f"excluded: {exclusion.reason_code}; evidence_frames={evidence}; "
                f"recoverable={'true' if exclusion.recoverable else 'false'}; "
                f"source_h5_unchanged=true; source_xml_unchanged=true"
            )
        output_rows.append(row)

    enabled = sum(
        _parse_bool(row["enabled"], field="enabled", row_number=index)
        for index, row in enumerate(output_rows, start=2)
    )
    return fieldnames, output_rows, enabled


def _atomic_write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
    overwrite: bool,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output manifest exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reversible Stage 4 teacher manifest by disabling videos "
            "listed in a versioned exclusion CSV."
        )
    )
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path)
    parser.add_argument("--expected_source_rows", type=int)
    parser.add_argument("--expected_excluded", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.source_manifest.resolve() == args.output_manifest.resolve():
        raise ValueError("output_manifest must differ from source_manifest")
    if args.exclusions.resolve() == args.output_manifest.resolve():
        raise ValueError("output_manifest must differ from exclusions")
    exclusions = load_exclusions(args.exclusions)
    fieldnames, rows, enabled = build_excluded_manifest(
        args.source_manifest,
        exclusions,
    )
    if args.expected_source_rows is not None and len(rows) != args.expected_source_rows:
        raise ValueError(
            f"Source row count mismatch: {len(rows)} != {args.expected_source_rows}"
        )
    if args.expected_excluded is not None and len(exclusions) != args.expected_excluded:
        raise ValueError(
            f"Exclusion count mismatch: {len(exclusions)} != {args.expected_excluded}"
        )
    if enabled != len(rows) - len(exclusions):
        raise ValueError(
            "Enabled count does not equal source rows minus exclusions; source "
            "manifest contains an independently disabled row"
        )

    _atomic_write_csv(
        args.output_manifest,
        fieldnames=fieldnames,
        rows=rows,
        overwrite=args.overwrite,
    )
    summary = {
        "schema_version": 1,
        "status": "ok",
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "exclusions": str(args.exclusions.resolve()),
        "exclusions_sha256": _sha256(args.exclusions),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": _sha256(args.output_manifest),
        "rows": len(rows),
        "enabled": enabled,
        "disabled": len(rows) - enabled,
        "excluded_videos": sorted(row.video_name for row in exclusions),
    }
    summary_path = args.summary_json or args.output_manifest.with_suffix(
        ".summary.json"
    )
    _write_json(summary_path, summary)

    print("Stage 4 exclusion manifest created")
    print(f"  source manifest : {args.source_manifest}")
    print(f"  exclusions      : {args.exclusions}")
    print(f"  output manifest : {args.output_manifest}")
    print(f"  summary         : {summary_path}")
    print(f"  rows            : {len(rows)}")
    print(f"  enabled         : {enabled}")
    print(f"  disabled        : {len(rows) - enabled}")
    print(f"  excluded videos : {sorted(row.video_name for row in exclusions)}")


if __name__ == "__main__":
    main()
