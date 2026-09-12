from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from checks.stage4.check_stage4_deleted_xml_annotation_audit import (
    _args as audit_args,
)
from checks.stage4.check_stage4_deleted_xml_annotation_audit import (
    _make_fixture,
)
from pseudo3d.analysis.audit_stage4_deleted_xml_annotations import run_audit
from pseudo3d.analysis.build_stage4_deleted_xml_invalidation_manifest import (
    ACTION,
    MANIFEST_FIELDS,
    REASON_CODE,
    XmlInvalidationManifestError,
    build_invalidation_manifest,
    invalidation_fingerprint,
    load_xml_invalidation_manifest,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256


def _build_args(
    paths: dict[str, Path],
    audit_root: Path,
    output_root: Path,
    **overrides: object,
) -> argparse.Namespace:
    values: dict[str, object] = {
        "audit_root": audit_root,
        "train_manifest": paths["manifest"],
        "exclusion_manifest": paths["exclusion"],
        "annotated_root": paths["annotated_root"],
        "output_manifest": output_root / "invalidations.csv",
        "output_summary": output_root / "summary.json",
        "expected_videos": 1,
        "expected_invalidations": 1,
        "expected_positive_points": 1,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _rewrite_csv(path: Path, mutate) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _expect_failure(args: argparse.Namespace, text: str) -> None:
    try:
        build_invalidation_manifest(args)
    except (XmlInvalidationManifestError, FileExistsError) as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected failure containing {text!r}")
    assert not Path(args.output_manifest).exists()
    assert not Path(args.output_summary).exists()


def test_end_to_end(root: Path) -> None:
    paths = _make_fixture(root / "fixture")
    audit_root = root / "audit"
    run_audit(audit_args(paths, audit_root))
    protected_before = {
        path: file_sha256(path)
        for path in (
            paths["manifest"],
            paths["exclusion"],
            paths["h5"],
            paths["present_xml"],
        )
    }
    output_root = root / "fixed"
    args = _build_args(paths, audit_root, output_root)
    summary_1 = build_invalidation_manifest(args)
    manifest_bytes = Path(args.output_manifest).read_bytes()
    summary_bytes = Path(args.output_summary).read_bytes()
    summary_2 = build_invalidation_manifest(args)
    assert Path(args.output_manifest).read_bytes() == manifest_bytes
    assert Path(args.output_summary).read_bytes() == summary_bytes
    assert summary_1 == summary_2

    rows = load_xml_invalidation_manifest(args.output_manifest)
    assert len(rows) == 1
    row = rows[0]
    assert row.video_name == "synthetic_deleted_xml"
    assert row.frame_order == 1
    assert row.frame_index == 11
    assert row.expected_xml_name == paths["missing_xml"].name
    assert row.expected_saved_bbox_rows == 1
    assert row.action == ACTION
    assert row.reason_code == REASON_CODE
    assert summary_1["invalidations"] == 1
    assert summary_1["expected_positive_points_to_remove"] == 1
    assert summary_1["invalidation_fingerprint"] == invalidation_fingerprint(rows)
    assert summary_1["h5_files_written"] == 0
    assert {path: file_sha256(path) for path in protected_before} == protected_before
    print("[OK] deterministic manifest fixation and verified-existing behavior")


def test_manifest_loader_rejections(root: Path) -> None:
    valid = {
        "schema_version": "1",
        "video_name": "video_a",
        "frame_order": "2",
        "frame_index": "12",
        "frame_stem": "video_a__fo00002__fi00000012",
        "expected_xml_name": "video_a_00013.xml",
        "expected_saved_bbox_rows": "1",
        "action": ACTION,
        "reason_code": REASON_CODE,
        "notes": "intentional deletion",
    }

    def write(name: str, rows: list[dict[str, str]]) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        return path

    duplicate = write("duplicate.csv", [valid, valid])
    try:
        load_xml_invalidation_manifest(duplicate)
    except XmlInvalidationManifestError as exc:
        assert "duplicate frame key" in str(exc)
    else:
        raise AssertionError("Duplicate frame must be rejected")

    for field, value, expected in (
        ("frame_stem", "wrong", "frame_stem mismatch"),
        ("expected_xml_name", "../bad.xml", "invalid expected_xml_name"),
        ("expected_saved_bbox_rows", "0", "must be >= 1"),
        ("action", "ignore", "action must be"),
        ("reason_code", "other", "reason_code must be"),
    ):
        invalid = dict(valid)
        invalid[field] = value
        path = write(f"invalid_{field}.csv", [invalid])
        try:
            load_xml_invalidation_manifest(path)
        except XmlInvalidationManifestError as exc:
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError(f"Invalid {field} must be rejected")
    print("[OK] strict manifest schema, identity, and action validation")


def test_evidence_rejections(root: Path) -> None:
    paths = _make_fixture(root / "fixture")
    audit_root = root / "audit"
    run_audit(audit_args(paths, audit_root))

    bad_stem_root = root / "audit_bad_stem"
    shutil.copytree(audit_root, bad_stem_root)
    _rewrite_csv(
        bad_stem_root / "missing_xml_candidates.csv",
        lambda rows: rows[0].__setitem__("frame_stem", "wrong"),
    )
    _rewrite_csv(
        bad_stem_root / "xml_inventory.csv",
        lambda rows: rows[1].__setitem__("frame_stem", "wrong"),
    )
    args = _build_args(paths, bad_stem_root, root / "bad_stem_output")
    _expect_failure(args, "audit frame_stem")

    mixed_root = root / "audit_mixed"
    shutil.copytree(audit_root, mixed_root)
    _rewrite_csv(
        mixed_root / "missing_xml_candidates.csv",
        lambda rows: rows[0].__setitem__(
            "frame_has_mixed_xml_presence", "True"
        ),
    )
    _rewrite_csv(
        mixed_root / "xml_inventory.csv",
        lambda rows: rows[1].__setitem__(
            "frame_has_mixed_xml_presence", "True"
        ),
    )
    args = _build_args(paths, mixed_root, root / "mixed_output")
    _expect_failure(args, "mixed XML presence")

    restored_root = root / "restored_output"
    paths["missing_xml"].write_text("<annotation/>\n", encoding="utf-8")
    args = _build_args(paths, audit_root, restored_root)
    _expect_failure(args, "deleted XML now exists")
    paths["missing_xml"].unlink()

    checksum_root = root / "checksum_output"
    paths["present_xml"].write_text("<changed/>\n", encoding="utf-8")
    args = _build_args(paths, audit_root, checksum_root)
    _expect_failure(args, "checksum changed")
    print("[OK] candidate drift, restored XML, and audited-input change rejection")


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="stage4_deleted_xml_invalidation_manifest_"
    ) as directory:
        root = Path(directory)
        test_end_to_end(root / "end_to_end")
        test_manifest_loader_rejections(root / "loader")
        test_evidence_rejections(root / "evidence")
    print("Stage 4 deleted-XML invalidation manifest synthetic checks passed.")


if __name__ == "__main__":
    main()
