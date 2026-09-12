from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_deleted_xml_annotations import (
    V5_TEACHER_TOKEN,
    DeletedXmlAuditError,
    classify_xml_paths,
    run_audit,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_v5_h5(path: Path, video: str, xml_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["video_name"] = video
        handle.attrs["num_frames"] = 2
        handle.attrs["contour_teacher_schema"] = V5_TEACHER_TOKEN
        handle.attrs["manual_review_fullvideo_reviewed_video"] = True
        point_cloud = handle.create_group("point_cloud")
        point_cloud.create_dataset(
            "frame_order", data=np.asarray([0, 0, 1, 1], dtype=np.int32)
        )
        point_cloud.create_dataset(
            "per_frame_counts", data=np.asarray([2, 2], dtype=np.int64)
        )
        annotation = handle.create_group("annotation")
        labels = np.asarray([1, -1, 1, 0], dtype=np.int8)
        annotation.create_dataset("point_label", data=labels)
        annotation.create_dataset("valid_mask", data=labels != -1)

        frame = handle.create_group("frame_annotation")
        frame.create_dataset("frame_order", data=np.asarray([0, 1], dtype=np.int32))
        frame.create_dataset("frame_index", data=np.asarray([10, 11], dtype=np.int64))
        frame.create_dataset("bbox_index", data=np.asarray([0, 0], dtype=np.int32))
        frame.create_dataset("valid_contour", data=np.asarray([True, True]))
        frame.create_dataset(
            "xml_path",
            data=np.asarray([str(path) for path in xml_paths], dtype=text_dtype),
        )

        provenance = handle.create_group("manual_review_fullvideo_frames")
        provenance.create_dataset(
            "frame_order", data=np.asarray([0, 1], dtype=np.int32)
        )
        provenance.create_dataset(
            "frame_index", data=np.asarray([10, 11], dtype=np.int64)
        )
        provenance.create_dataset(
            "frame_stem",
            data=np.asarray(
                [
                    f"{video}__fo00000__fi00000010",
                    f"{video}__fo00001__fi00000011",
                ],
                dtype=text_dtype,
            ),
        )
        provenance.create_dataset(
            "cvat_mask_status",
            data=np.asarray(["positive", "positive"], dtype=text_dtype),
        )
        provenance.create_dataset(
            "cvat_mask_positive_pixels",
            data=np.asarray([8, 7], dtype=np.int64),
        )
        provenance.create_dataset(
            "label_authority",
            data=np.asarray(["cvat_snapshot", "cvat_snapshot"], dtype=text_dtype),
        )


def _make_fixture(root: Path) -> dict[str, Path]:
    video = "synthetic_deleted_xml"
    excluded_video = "synthetic_crop_excluded"
    voc_root = root / "voc"
    annotation_dir = voc_root / video / "annotations_renamed"
    annotation_dir.mkdir(parents=True)
    present_xml = annotation_dir / f"{video}_00011.xml"
    missing_xml = annotation_dir / f"{video}_00012.xml"
    present_xml.write_text("<annotation/>\n", encoding="utf-8")

    annotated_root = root / "annotated"
    h5_path = (
        annotated_root
        / video
        / f"{video}_pointcloud_annotated_foreground_{V5_TEACHER_TOKEN}.h5"
    )
    _write_v5_h5(h5_path, video, [present_xml, missing_xml])

    manifest = root / "manifest.csv"
    _write_csv(
        manifest,
        ["video_name", "pseudo3d_h5", "voc_xml_root", "split", "enabled", "notes"],
        [
            {
                "video_name": video,
                "pseudo3d_h5": str(root / "pseudo" / f"{video}.h5"),
                "voc_xml_root": str(voc_root),
                "split": "train",
                "enabled": "true",
                "notes": "synthetic enabled",
            },
            {
                "video_name": excluded_video,
                "pseudo3d_h5": str(root / "pseudo" / f"{excluded_video}.h5"),
                "voc_xml_root": str(voc_root),
                "split": "train",
                "enabled": "false",
                "notes": "synthetic excluded",
            },
        ],
    )
    exclusion = root / "exclusions.csv"
    _write_csv(
        exclusion,
        [
            "video_name",
            "reason_code",
            "evidence_frames",
            "scope",
            "recoverable",
            "notes",
        ],
        [
            {
                "video_name": excluded_video,
                "reason_code": "synthetic_crop_failure",
                "evidence_frames": "0",
                "scope": "stage4_teacher",
                "recoverable": "true",
                "notes": "synthetic exclusion row",
            }
        ],
    )
    return {
        "manifest": manifest,
        "exclusion": exclusion,
        "annotated_root": annotated_root,
        "h5": h5_path,
        "present_xml": present_xml,
        "missing_xml": missing_xml,
    }


def _args(paths: dict[str, Path], output_root: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "manifest": paths["manifest"],
        "exclusion_manifest": paths["exclusion"],
        "annotated_root": paths["annotated_root"],
        "output_root": output_root,
        "expected_videos": 1,
        "expected_excluded_videos": 1,
        "expected_missing_frames": 1,
        "expected_missing_xml_entries": 1,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_path_classification(root: Path) -> None:
    saved = root / "old" / "frame.xml"
    canonical = root / "new" / "frame.xml"
    saved.parent.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    saved.write_text("old\n", encoding="utf-8")
    canonical.write_text("new\n", encoding="utf-8")
    conflict = classify_xml_paths(
        saved_xml_path=saved,
        canonical_xml_path=canonical,
    )
    assert conflict["xml_status"] == "canonical_saved_conflict"
    assert conflict["missing_candidate"] is False
    saved.unlink()
    relocated = classify_xml_paths(
        saved_xml_path=saved,
        canonical_xml_path=canonical,
    )
    assert relocated["xml_status"] == "canonical_only"
    canonical.unlink()
    missing = classify_xml_paths(
        saved_xml_path=saved,
        canonical_xml_path=canonical,
    )
    assert missing["xml_status"] == "missing"
    assert missing["missing_candidate"] is True
    print("[OK] missing/relocated/conflicting XML path classification")


def test_end_to_end(root: Path) -> None:
    paths = _make_fixture(root / "fixture")
    protected_before = {
        path: file_sha256(path)
        for path in (paths["manifest"], paths["exclusion"], paths["h5"], paths["present_xml"])
    }
    output_1 = root / "audit_1"
    output_2 = root / "audit_2"
    summary = run_audit(_args(paths, output_1))
    run_audit(_args(paths, output_2))
    assert summary["videos"] == 1
    assert summary["bbox_rows"] == 2
    assert summary["missing_xml_entries"] == 1
    assert summary["missing_frames"] == 1
    assert summary["all_xml_missing_frames"] == 1
    assert summary["mixed_xml_presence_frames"] == 0
    assert summary["all_missing_frame_positive_points"] == 1
    assert summary["h5_files_written"] == 0

    with (output_1 / "missing_xml_candidates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        candidates = list(csv.DictReader(handle))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["frame_order"] == "1"
    assert candidate["frame_index"] == "11"
    assert candidate["frame_all_xml_missing"] == "True"
    assert candidate["positive_points"] == "1"
    assert candidate["cvat_mask_status"] == "positive"

    for name in (
        "xml_inventory.csv",
        "missing_xml_candidates.csv",
        "video_summary.csv",
        "input_checksums.csv",
        "audit_summary.json",
    ):
        assert (output_1 / name).read_bytes() == (output_2 / name).read_bytes()
    protected_after = {path: file_sha256(path) for path in protected_before}
    assert protected_after == protected_before
    print("[OK] deterministic read-only missing-XML audit and candidate metrics")

    try:
        run_audit(
            _args(
                paths,
                root / "failed_audit",
                expected_missing_frames=0,
            )
        )
    except DeletedXmlAuditError as exc:
        assert "Missing frame count mismatch" in str(exc)
    else:
        raise AssertionError("Wrong expected missing-frame count must fail")
    assert not (root / "failed_audit").exists()
    print("[OK] strict expected-count rejection without partial report")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_deleted_xml_audit_") as directory:
        root = Path(directory)
        test_path_classification(root / "classification")
        test_end_to_end(root)
    source = (
        REPO_ROOT / "pseudo3d/analysis/audit_stage4_deleted_xml_annotations.py"
    ).read_text(encoding="utf-8")
    assert 'h5py.File(h5_path, "r")' in source
    assert 'h5py.File(h5_path, "r+")' not in source
    print("[OK] H5 read-only implementation contract")
    print("Stage 4 deleted-XML annotation audit synthetic checks passed.")


if __name__ == "__main__":
    main()
