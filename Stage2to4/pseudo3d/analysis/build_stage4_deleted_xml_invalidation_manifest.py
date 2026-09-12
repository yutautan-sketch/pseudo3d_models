from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pseudo3d.analysis.audit_stage4_deleted_xml_annotations import (
    INVENTORY_FIELDS,
    V5_TEACHER_TOKEN,
    _canonical_xml_path,
    _single_v5_h5,
    audit_video_h5,
)
from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    Stage4SweepManifestItem,
    file_sha256,
    load_stage4_sweep_manifest,
)


SCHEMA_VERSION = 1
ACTION = "invalidate_entire_frame"
REASON_CODE = "deleted_incorrect_bbox_xml"
MANIFEST_FIELDS = (
    "schema_version",
    "video_name",
    "frame_order",
    "frame_index",
    "frame_stem",
    "expected_xml_name",
    "expected_saved_bbox_rows",
    "action",
    "reason_code",
    "notes",
)
AUDIT_REQUIRED_FILES = (
    "audit_summary.json",
    "xml_inventory.csv",
    "missing_xml_candidates.csv",
    "video_summary.csv",
    "input_checksums.csv",
)
_VIDEO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_XML_PATTERN = re.compile(r"^[^/\\]+\.xml$", re.IGNORECASE)


class XmlInvalidationManifestError(RuntimeError):
    """Raised when deleted-XML invalidation evidence is inconsistent."""


@dataclass(frozen=True)
class XmlFrameInvalidation:
    schema_version: int
    video_name: str
    frame_order: int
    frame_index: int
    frame_stem: str
    expected_xml_name: str
    expected_saved_bbox_rows: int
    action: str
    reason_code: str
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
            "expected_xml_name": self.expected_xml_name,
            "expected_saved_bbox_rows": str(self.expected_saved_bbox_rows),
            "action": self.action,
            "reason_code": self.reason_code,
            "notes": self.notes,
        }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise XmlInvalidationManifestError(f"CSV has no header: {path}")
        rows = [
            {key: value or "" for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]
        return list(reader.fieldnames), rows


def _require_columns(
    path: Path,
    fields: Sequence[str],
    required: Sequence[str],
) -> None:
    missing = [name for name in required if name not in fields]
    if missing:
        raise XmlInvalidationManifestError(
            f"CSV missing columns {missing}: {path}"
        )


def _parse_int(value: str, *, field: str, row_number: int, minimum: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise XmlInvalidationManifestError(
            f"Row {row_number}: {field} must be an integer, got {value!r}"
        ) from exc
    if parsed < minimum:
        raise XmlInvalidationManifestError(
            f"Row {row_number}: {field} must be >= {minimum}, got {parsed}"
        )
    return parsed


def _parse_bool(value: str, *, field: str, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise XmlInvalidationManifestError(
        f"Row {row_number}: {field} must be true/false, got {value!r}"
    )


def _expected_frame_stem(video: str, order: int, index: int) -> str:
    return f"{video}__fo{order:05d}__fi{index:08d}"


def load_xml_invalidation_manifest(path: Path) -> list[XmlFrameInvalidation]:
    fields, rows = _read_csv(path)
    _require_columns(path, fields, MANIFEST_FIELDS)
    if tuple(fields) != MANIFEST_FIELDS:
        raise XmlInvalidationManifestError(
            f"Invalidation manifest columns/order differ from schema: {path}"
        )
    invalidations: list[XmlFrameInvalidation] = []
    seen_frames: set[tuple[str, int, int]] = set()
    seen_stems: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        schema = _parse_int(
            row["schema_version"], field="schema_version", row_number=row_number
        )
        if schema != SCHEMA_VERSION:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: unsupported schema_version={schema}"
            )
        video = row["video_name"].strip()
        if not _VIDEO_PATTERN.fullmatch(video):
            raise XmlInvalidationManifestError(
                f"Row {row_number}: invalid video_name={video!r}"
            )
        order = _parse_int(
            row["frame_order"], field="frame_order", row_number=row_number
        )
        index = _parse_int(
            row["frame_index"], field="frame_index", row_number=row_number
        )
        stem = row["frame_stem"].strip()
        expected_stem = _expected_frame_stem(video, order, index)
        if stem != expected_stem:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: frame_stem mismatch: {stem!r} != "
                f"{expected_stem!r}"
            )
        xml_name = row["expected_xml_name"].strip()
        if not _XML_PATTERN.fullmatch(xml_name):
            raise XmlInvalidationManifestError(
                f"Row {row_number}: invalid expected_xml_name={xml_name!r}"
            )
        bbox_rows = _parse_int(
            row["expected_saved_bbox_rows"],
            field="expected_saved_bbox_rows",
            row_number=row_number,
            minimum=1,
        )
        action = row["action"].strip()
        reason = row["reason_code"].strip()
        notes = row["notes"].strip()
        if action != ACTION:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: action must be {ACTION!r}, got {action!r}"
            )
        if reason != REASON_CODE:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: reason_code must be {REASON_CODE!r}, "
                f"got {reason!r}"
            )
        if not notes:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: notes must not be empty"
            )
        item = XmlFrameInvalidation(
            schema_version=schema,
            video_name=video,
            frame_order=order,
            frame_index=index,
            frame_stem=stem,
            expected_xml_name=xml_name,
            expected_saved_bbox_rows=bbox_rows,
            action=action,
            reason_code=reason,
            notes=notes,
        )
        if item.frame_key in seen_frames:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: duplicate frame key={item.frame_key}"
            )
        if item.frame_stem in seen_stems:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: duplicate frame_stem={item.frame_stem!r}"
            )
        seen_frames.add(item.frame_key)
        seen_stems.add(item.frame_stem)
        invalidations.append(item)
    if not invalidations:
        raise XmlInvalidationManifestError(
            f"Invalidation manifest has no data rows: {path}"
        )
    return invalidations


def invalidation_fingerprint(rows: Sequence[XmlFrameInvalidation]) -> str:
    payload = [row.to_row() for row in sorted(rows, key=lambda item: item.frame_key)]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _csv_bytes(rows: Sequence[XmlFrameInvalidation]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(MANIFEST_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(row.to_row() for row in rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_or_verify(
    path: Path,
    content: bytes,
    *,
    overwrite: bool,
    write: bool = True,
) -> str:
    path = Path(path)
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return "verified_existing"
        if not overwrite:
            raise FileExistsError(
                f"Output differs from expected content; pass --overwrite: {path}"
            )
    if not write:
        return "written"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return "written"


def _verify_audit_checksums(audit_root: Path) -> int:
    path = audit_root / "input_checksums.csv"
    fields, rows = _read_csv(path)
    _require_columns(path, fields, ("kind", "path", "sha256"))
    if not rows:
        raise XmlInvalidationManifestError("Audit input checksum inventory is empty")
    seen: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        source = Path(row["path"]).expanduser().resolve(strict=False)
        expected = row["sha256"].strip()
        if source in seen:
            raise XmlInvalidationManifestError(
                f"Duplicate audit checksum path at row {row_number}: {source}"
            )
        seen.add(source)
        if not source.is_file():
            raise XmlInvalidationManifestError(
                f"Audited input is missing or changed: {source}"
            )
        actual = file_sha256(source)
        if actual != expected:
            raise XmlInvalidationManifestError(
                f"Audited input checksum changed: {source}"
            )
    return len(rows)


def _candidate_key(row: Mapping[str, str]) -> tuple[str, int, int, str]:
    return (
        row["video_name"].strip(),
        int(row["frame_order"]),
        int(row["frame_index"]),
        row["xml_name"].strip(),
    )


def _validate_candidate_row(
    row: Mapping[str, str], *, row_number: int
) -> XmlFrameInvalidation:
    required_truth = ("missing_candidate", "frame_all_xml_missing")
    for field in required_truth:
        if not _parse_bool(row[field], field=field, row_number=row_number):
            raise XmlInvalidationManifestError(
                f"Row {row_number}: {field} must be true"
            )
    if _parse_bool(
        row["frame_has_mixed_xml_presence"],
        field="frame_has_mixed_xml_presence",
        row_number=row_number,
    ):
        raise XmlInvalidationManifestError(
            f"Row {row_number}: mixed XML presence cannot invalidate an entire frame"
        )
    for field in ("saved_path_exists", "canonical_path_exists"):
        if _parse_bool(row[field], field=field, row_number=row_number):
            raise XmlInvalidationManifestError(
                f"Row {row_number}: {field} must be false for deleted XML"
            )
    if row["xml_status"].strip() != "missing":
        raise XmlInvalidationManifestError(
            f"Row {row_number}: xml_status must be 'missing'"
        )
    if row["resolved_xml_path"].strip() or row["xml_sha256"].strip():
        raise XmlInvalidationManifestError(
            f"Row {row_number}: missing XML unexpectedly has resolved path/checksum"
        )
    video = row["video_name"].strip()
    order = _parse_int(row["frame_order"], field="frame_order", row_number=row_number)
    index = _parse_int(row["frame_index"], field="frame_index", row_number=row_number)
    stem = row["frame_stem"].strip()
    if stem != _expected_frame_stem(video, order, index):
        raise XmlInvalidationManifestError(
            f"Row {row_number}: audit frame_stem does not match frame identity"
        )
    xml_name = row["xml_name"].strip()
    for field in ("saved_xml_path", "canonical_xml_path"):
        path = Path(row[field]).expanduser().resolve(strict=False)
        if path.name != xml_name:
            raise XmlInvalidationManifestError(
                f"Row {row_number}: {field} basename differs from xml_name"
            )
        if path.exists():
            raise XmlInvalidationManifestError(
                f"Row {row_number}: deleted XML now exists: {path}"
            )
    xml_bbox_rows = _parse_int(
        row["xml_bbox_rows"], field="xml_bbox_rows", row_number=row_number, minimum=1
    )
    frame_bbox_rows = _parse_int(
        row["frame_bbox_rows"],
        field="frame_bbox_rows",
        row_number=row_number,
        minimum=1,
    )
    if xml_bbox_rows != frame_bbox_rows:
        raise XmlInvalidationManifestError(
            f"Row {row_number}: frame contains more than one saved XML entry; "
            "object-level ambiguity requires separate handling"
        )
    return XmlFrameInvalidation(
        schema_version=SCHEMA_VERSION,
        video_name=video,
        frame_order=order,
        frame_index=index,
        frame_stem=stem,
        expected_xml_name=xml_name,
        expected_saved_bbox_rows=frame_bbox_rows,
        action=ACTION,
        reason_code=REASON_CODE,
        notes="Confirmed intentional XML deletion; invalidate the entire frame.",
    )


def _compare_live_candidate(
    expected: Mapping[str, str], live: Mapping[str, Any]
) -> None:
    exact_fields = (
        "video_name",
        "frame_order",
        "frame_index",
        "frame_stem",
        "xml_name",
        "saved_xml_path",
        "canonical_xml_path",
        "xml_bbox_rows",
        "frame_bbox_rows",
        "positive_points",
        "ignore_points",
        "background_points",
        "cvat_mask_status",
        "cvat_mask_positive_pixels",
    )
    for field in exact_fields:
        if str(expected[field]).strip() != str(live[field]).strip():
            raise XmlInvalidationManifestError(
                f"Audit candidate differs from current v5 evidence for {field}: "
                f"{expected[field]!r} != {live[field]!r}"
            )
    for field in ("missing_candidate", "frame_all_xml_missing"):
        if not bool(live[field]):
            raise XmlInvalidationManifestError(
                f"Current v5 evidence no longer marks {field}=true"
            )
    if bool(live["frame_has_mixed_xml_presence"]):
        raise XmlInvalidationManifestError(
            "Current v5 evidence has mixed XML presence"
        )


def build_invalidation_manifest(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = Path(args.audit_root).resolve()
    train_manifest = Path(args.train_manifest).resolve()
    exclusion_manifest = Path(args.exclusion_manifest).resolve()
    annotated_root = Path(args.annotated_root).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_summary = Path(args.output_summary).resolve()
    for name in AUDIT_REQUIRED_FILES:
        if not (audit_root / name).is_file():
            raise FileNotFoundError(f"Audit artifact not found: {audit_root / name}")
    if not annotated_root.is_dir():
        raise FileNotFoundError(f"v5 annotated root not found: {annotated_root}")

    audit_summary = json.loads(
        (audit_root / "audit_summary.json").read_text(encoding="utf-8")
    )
    if audit_summary.get("status") != "ok":
        raise XmlInvalidationManifestError("Step 1 audit status is not ok")
    if audit_summary.get("teacher_version") != V5_TEACHER_TOKEN:
        raise XmlInvalidationManifestError("Step 1 teacher version mismatch")
    if not bool(audit_summary.get("input_files_unchanged")):
        raise XmlInvalidationManifestError("Step 1 inputs were not unchanged")
    if int(audit_summary.get("failure_rows", -1)) != 0:
        raise XmlInvalidationManifestError("Step 1 audit contains failures")
    if file_sha256(train_manifest) != str(audit_summary.get("manifest_sha256", "")):
        raise XmlInvalidationManifestError("Train manifest differs from Step 1 audit")
    if file_sha256(exclusion_manifest) != str(
        audit_summary.get("exclusion_manifest_sha256", "")
    ):
        raise XmlInvalidationManifestError(
            "Exclusion manifest differs from Step 1 audit"
        )
    checksum_rows = _verify_audit_checksums(audit_root)

    manifest_items = load_stage4_sweep_manifest(train_manifest)
    excluded = {item.video_name for item in load_exclusions(exclusion_manifest)}
    enabled_items = {
        item.video_name: item
        for item in manifest_items
        if item.enabled and item.video_name not in excluded
    }
    if len(enabled_items) != int(args.expected_videos):
        raise XmlInvalidationManifestError(
            f"Enabled video count mismatch: {len(enabled_items)} != "
            f"{args.expected_videos}"
        )

    candidate_path = audit_root / "missing_xml_candidates.csv"
    candidate_fields, candidates = _read_csv(candidate_path)
    candidate_required = (
        "xml_status",
        "missing_candidate",
        "frame_all_xml_missing",
        "frame_has_mixed_xml_presence",
        "video_name",
        "frame_order",
        "frame_index",
        "frame_stem",
        "xml_name",
        "saved_xml_path",
        "canonical_xml_path",
        "resolved_xml_path",
        "saved_path_exists",
        "canonical_path_exists",
        "xml_sha256",
        "xml_bbox_rows",
        "frame_bbox_rows",
        "positive_points",
        "ignore_points",
        "background_points",
        "cvat_mask_status",
        "cvat_mask_positive_pixels",
    )
    _require_columns(candidate_path, candidate_fields, candidate_required)
    if tuple(candidate_fields) != INVENTORY_FIELDS:
        raise XmlInvalidationManifestError(
            "Step 1 candidate CSV columns/order differ from the audit schema"
        )
    if len(candidates) != int(args.expected_invalidations):
        raise XmlInvalidationManifestError(
            f"Invalidation candidate count mismatch: {len(candidates)} != "
            f"{args.expected_invalidations}"
        )
    for summary_field in (
        "missing_xml_entries",
        "missing_frames",
        "all_xml_missing_frames",
    ):
        if int(audit_summary.get(summary_field, -1)) != len(candidates):
            raise XmlInvalidationManifestError(
                f"Step 1 {summary_field} differs from candidate count"
            )
    if int(audit_summary.get("mixed_xml_presence_frames", -1)) != 0:
        raise XmlInvalidationManifestError(
            "Step 1 includes mixed-presence frames; automatic frame invalidation refused"
        )
    positive_points = sum(int(row["positive_points"]) for row in candidates)
    if positive_points != int(args.expected_positive_points):
        raise XmlInvalidationManifestError(
            f"Candidate positive-point count mismatch: {positive_points} != "
            f"{args.expected_positive_points}"
        )
    if positive_points != int(
        audit_summary.get("all_missing_frame_positive_points", -1)
    ):
        raise XmlInvalidationManifestError(
            "Step 1 positive-point total differs from candidates"
        )

    inventory_path = audit_root / "xml_inventory.csv"
    inventory_fields, inventory = _read_csv(inventory_path)
    _require_columns(inventory_path, inventory_fields, candidate_required)
    if tuple(inventory_fields) != INVENTORY_FIELDS:
        raise XmlInvalidationManifestError(
            "Step 1 XML inventory columns/order differ from the audit schema"
        )
    inventory_missing = [
        row
        for row_number, row in enumerate(inventory, start=2)
        if _parse_bool(
            row["missing_candidate"],
            field="missing_candidate",
            row_number=row_number,
        )
    ]
    candidate_keys = [_candidate_key(row) for row in candidates]
    inventory_keys = [_candidate_key(row) for row in inventory_missing]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise XmlInvalidationManifestError("Step 1 candidate CSV has duplicate keys")
    if sorted(candidate_keys) != sorted(inventory_keys):
        raise XmlInvalidationManifestError(
            "Step 1 candidate CSV differs from missing rows in XML inventory"
        )
    inventory_by_key = {
        _candidate_key(row): row for row in inventory_missing
    }
    for candidate in candidates:
        key = _candidate_key(candidate)
        if candidate != inventory_by_key[key]:
            raise XmlInvalidationManifestError(
                f"Step 1 candidate row differs from XML inventory: {key}"
            )

    invalidations: list[XmlFrameInvalidation] = []
    candidates_by_video: dict[str, list[dict[str, str]]] = {}
    for row_number, candidate in enumerate(candidates, start=2):
        invalidation = _validate_candidate_row(candidate, row_number=row_number)
        if invalidation.video_name not in enabled_items:
            raise XmlInvalidationManifestError(
                f"Candidate video is not enabled in crop-clean manifest: "
                f"{invalidation.video_name}"
            )
        invalidations.append(invalidation)
        candidates_by_video.setdefault(invalidation.video_name, []).append(candidate)

    invalidations.sort(key=lambda item: item.frame_key)
    if len({item.frame_key for item in invalidations}) != len(invalidations):
        raise XmlInvalidationManifestError("Candidates contain duplicate frames")

    live_h5_paths: list[Path] = []
    for video, video_candidates in sorted(candidates_by_video.items()):
        item: Stage4SweepManifestItem = enabled_items[video]
        h5_path = _single_v5_h5(annotated_root, video)
        live_h5_paths.append(h5_path)
        live_rows, _, _ = audit_video_h5(item=item, h5_path=h5_path)
        live_missing = {
            _candidate_key({key: str(value) for key, value in row.items()}): row
            for row in live_rows
            if bool(row["missing_candidate"])
        }
        for candidate in video_candidates:
            key = _candidate_key(candidate)
            if key not in live_missing:
                raise XmlInvalidationManifestError(
                    f"Candidate is not missing in current v5 evidence: {key}"
                )
            _compare_live_candidate(candidate, live_missing[key])
            saved = Path(candidate["saved_xml_path"]).resolve(strict=False)
            canonical = _canonical_xml_path(item, saved)
            if canonical != Path(candidate["canonical_xml_path"]).resolve(strict=False):
                raise XmlInvalidationManifestError(
                    f"Canonical XML path changed for candidate: {key}"
                )
            if saved.exists() or canonical.exists():
                raise XmlInvalidationManifestError(
                    f"Deleted XML was restored after audit: {key}"
                )

    manifest_content = _csv_bytes(invalidations)
    parsed_for_fingerprint = invalidations
    fingerprint = invalidation_fingerprint(parsed_for_fingerprint)
    manifest_sha = hashlib.sha256(manifest_content).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "action": ACTION,
        "reason_code": REASON_CODE,
        "teacher_version": V5_TEACHER_TOKEN,
        "invalidations": len(invalidations),
        "videos": len(candidates_by_video),
        "expected_positive_points_to_remove": positive_points,
        "mixed_xml_presence_frames": 0,
        "input_files_unchanged": True,
        "h5_files_written": 0,
        "audit_root": str(audit_root),
        "audit_summary_sha256": file_sha256(audit_root / "audit_summary.json"),
        "audit_candidates_sha256": file_sha256(candidate_path),
        "audit_inventory_sha256": file_sha256(inventory_path),
        "audit_input_checksums_sha256": file_sha256(
            audit_root / "input_checksums.csv"
        ),
        "audit_checksum_rows_verified": checksum_rows,
        "train_manifest_sha256": file_sha256(train_manifest),
        "exclusion_manifest_sha256": file_sha256(exclusion_manifest),
        "v5_candidate_h5_sha256": {
            path.name: file_sha256(path) for path in sorted(live_h5_paths)
        },
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": manifest_sha,
        "invalidation_fingerprint": fingerprint,
        "failure_rows": 0,
    }
    summary_content = _json_bytes(summary)
    manifest_status = _write_or_verify(
        output_manifest,
        manifest_content,
        overwrite=bool(args.overwrite),
        write=False,
    )
    summary_status = _write_or_verify(
        output_summary,
        summary_content,
        overwrite=bool(args.overwrite),
        write=False,
    )
    if manifest_status == "written":
        _write_or_verify(
            output_manifest, manifest_content, overwrite=bool(args.overwrite)
        )
    if summary_status == "written":
        _write_or_verify(
            output_summary, summary_content, overwrite=bool(args.overwrite)
        )

    loaded = load_xml_invalidation_manifest(output_manifest)
    if invalidation_fingerprint(loaded) != fingerprint:
        raise XmlInvalidationManifestError(
            "Written invalidation manifest fingerprint mismatch"
        )
    print("Stage 4 deleted-XML invalidation manifest fixed")
    print(f"  audit root          : {audit_root}")
    print(f"  output manifest     : {output_manifest}")
    print(f"  summary             : {output_summary}")
    print(f"  manifest status     : {manifest_status}")
    print(f"  summary status      : {summary_status}")
    print(f"  invalidations       : {len(invalidations)}")
    print(f"  affected videos     : {len(candidates_by_video)}")
    print(f"  positive to remove  : {positive_points}")
    print(f"  fingerprint         : {fingerprint}")
    print("Stage 4 deleted-XML invalidation manifest preflight passed.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fix a versioned frame-invalidation manifest from a verified Stage 4 "
            "deleted-XML audit. No H5 file is modified."
        )
    )
    parser.add_argument("--audit_root", type=Path, required=True)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--exclusion_manifest", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, required=True)
    parser.add_argument("--expected_invalidations", type=int, required=True)
    parser.add_argument("--expected_positive_points", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    try:
        build_invalidation_manifest(build_parser().parse_args())
    except Exception as exc:
        raise SystemExit(
            f"Stage 4 deleted-XML invalidation manifest failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
