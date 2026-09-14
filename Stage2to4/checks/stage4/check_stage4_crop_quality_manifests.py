from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pseudo3d.analysis.validate_stage4_crop_quality_manifests import (
    MANIFEST_FIELDS,
    CropQualityManifestError,
    invalidation_fingerprint,
    load_crop_quality_invalidations,
    validate,
)


OLD_VIDEO = "synthetic_old_exclusion"
NEW_VIDEO = "synthetic_new_exclusion"
FRAME_VIDEO = "synthetic_frame_invalidation"


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _exclusion(video: str) -> dict[str, object]:
    return {
        "video_name": video,
        "reason_code": "local_crop_tracking_drift",
        "evidence_frames": "all_bbox_frames",
        "scope": "stage4_teacher",
        "recoverable": "true",
        "notes": "Preserve all source evidence for future recropping.",
    }


def _crop_row() -> dict[str, object]:
    return {
        "schema_version": 1,
        "video_name": FRAME_VIDEO,
        "frame_order": 51,
        "frame_index": 51,
        "frame_stem": f"{FRAME_VIDEO}__fo00051__fi00000051",
        "expected_saved_bbox_rows": 1,
        "expected_stray_ignore_points": 6,
        "expected_crop_status": "fully_outside_crop",
        "expected_visible_fraction": 0.0,
        "action": "invalidate_entire_frame",
        "reason_code": "local_crop_fully_outside",
        "recoverable": "true",
        "notes": "Synthetic crop-quality invalidation.",
    }


def _fixture(root: Path) -> dict[str, Path]:
    manifest = root / "manifest.csv"
    _write_csv(
        manifest,
        ("video_name", "pseudo3d_h5", "voc_xml_root", "split", "enabled", "notes"),
        [
            {
                "video_name": video,
                "pseudo3d_h5": root / f"{video}.h5",
                "voc_xml_root": root / "voc",
                "split": "train",
                "enabled": "true",
                "notes": "synthetic",
            }
            for video in (OLD_VIDEO, NEW_VIDEO, FRAME_VIDEO)
        ],
    )
    previous = root / "exclusions_v1.csv"
    current = root / "exclusions_v2.csv"
    exclusion_fields = (
        "video_name",
        "reason_code",
        "evidence_frames",
        "scope",
        "recoverable",
        "notes",
    )
    _write_csv(previous, exclusion_fields, [_exclusion(OLD_VIDEO)])
    _write_csv(current, exclusion_fields, [_exclusion(OLD_VIDEO), _exclusion(NEW_VIDEO)])
    crop = root / "crop.csv"
    _write_csv(crop, MANIFEST_FIELDS, [_crop_row()])

    audit = root / "audit"
    audit.mkdir()
    (audit / "audit_summary.json").write_text(
        json.dumps({"status": "ok", "input_files_unchanged": True}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        audit / "frame_metrics.csv",
        (
            "video_name",
            "frame_order",
            "frame_index",
            "saved_bbox_rows",
            "stray_ignore_points",
            "fully_outside_crop_bboxes",
            "saved_degenerate_bbox_rows",
        ),
        [
            {
                "video_name": FRAME_VIDEO,
                "frame_order": 51,
                "frame_index": 51,
                "saved_bbox_rows": 1,
                "stray_ignore_points": 6,
                "fully_outside_crop_bboxes": 1,
                "saved_degenerate_bbox_rows": 1,
            }
        ],
    )
    _write_csv(
        audit / "bbox_geometry.csv",
        (
            "video_name",
            "frame_order",
            "frame_index",
            "crop_status",
            "visible_fraction",
            "saved_bbox_degenerate",
        ),
        [
            {
                "video_name": FRAME_VIDEO,
                "frame_order": 51,
                "frame_index": 51,
                "crop_status": "fully_outside_crop",
                "visible_fraction": 0.0,
                "saved_bbox_degenerate": "True",
            }
        ],
    )
    _write_csv(audit / "video_summary.csv", ("video_name",), [{"video_name": FRAME_VIDEO}])
    _write_csv(audit / "input_checksums.csv", ("path", "sha256"), [])
    return {
        "manifest": manifest,
        "previous": previous,
        "current": current,
        "crop": crop,
        "audit": audit,
    }


def _args(paths: dict[str, Path], output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        audit_root=paths["audit"],
        source_manifest=paths["manifest"],
        previous_exclusions=paths["previous"],
        video_exclusions=paths["current"],
        crop_invalidations=paths["crop"],
        output_summary=output,
        expected_excluded_video=[OLD_VIDEO, NEW_VIDEO],
        expected_invalidations=1,
        overwrite=False,
    )


def test_validated_contract_and_verified_resume(root: Path) -> None:
    paths = _fixture(root)
    output = root / "summary.json"
    first = validate(_args(paths, output))
    content = output.read_bytes()
    second = validate(_args(paths, output))
    assert first == second
    assert output.read_bytes() == content
    rows = load_crop_quality_invalidations(paths["crop"])
    assert len(rows) == 1 and rows[0].frame_key == (FRAME_VIDEO, 51, 51)
    assert first["crop_invalidation_fingerprint"] == invalidation_fingerprint(rows)
    assert first["expected_stray_points_to_remove"] == 6
    assert first["h5_files_written"] == 0
    print("[OK] v1 exclusion preservation, fixed crop evidence, and verified resume")


def _expect_failure(args: argparse.Namespace, expected: str) -> None:
    try:
        validate(args)
    except (CropQualityManifestError, FileExistsError) as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected failure containing {expected!r}")
    assert not Path(args.output_summary).exists()


def test_drift_and_overlap_rejections(root: Path) -> None:
    paths = _fixture(root / "evidence")
    with (paths["audit"] / "frame_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    rows[0]["stray_ignore_points"] = "5"
    _write_csv(paths["audit"] / "frame_metrics.csv", fields, rows)
    _expect_failure(_args(paths, root / "bad_evidence.json"), "Stray count changed")

    paths = _fixture(root / "overlap")
    crop = _crop_row()
    crop["video_name"] = NEW_VIDEO
    crop["frame_stem"] = f"{NEW_VIDEO}__fo00051__fi00000051"
    _write_csv(paths["crop"], MANIFEST_FIELDS, [crop])
    _expect_failure(_args(paths, root / "bad_overlap.json"), "must not target excluded")

    paths = _fixture(root / "lost_v1")
    exclusion_fields, _ = _read_fields_rows(paths["current"])
    _write_csv(paths["current"], exclusion_fields, [_exclusion(NEW_VIDEO)])
    _expect_failure(_args(paths, root / "bad_v1.json"), "does not preserve v1")
    print("[OK] audit drift, exclusion/invalidation overlap, and lost-v1 rejection")


def _read_fields_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def test_loader_rejections(root: Path) -> None:
    paths = _fixture(root)
    row = _crop_row()
    _write_csv(paths["crop"], MANIFEST_FIELDS, [row, row])
    try:
        load_crop_quality_invalidations(paths["crop"])
    except CropQualityManifestError as exc:
        assert "duplicate frame key" in str(exc)
    else:
        raise AssertionError("Duplicate crop invalidation was accepted")
    row = _crop_row()
    row["expected_visible_fraction"] = "0.1"
    _write_csv(paths["crop"], MANIFEST_FIELDS, [row])
    try:
        load_crop_quality_invalidations(paths["crop"])
    except CropQualityManifestError as exc:
        assert "visible fraction 0" in str(exc)
    else:
        raise AssertionError("Nonzero visibility was accepted for fully-outside crop")
    print("[OK] duplicate and nonzero-visible fully-outside manifest rejection")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_crop_quality_manifests_") as directory:
        root = Path(directory)
        test_validated_contract_and_verified_resume(root / "valid")
        test_drift_and_overlap_rejections(root / "invalid")
        test_loader_rejections(root / "loader")
    print("Stage 4 crop-quality manifest synthetic checks passed.")


if __name__ == "__main__":
    main()
