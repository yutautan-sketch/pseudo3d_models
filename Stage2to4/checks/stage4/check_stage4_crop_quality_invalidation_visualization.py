from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_ROOT = Path(__file__).resolve().parent
for candidate in (REPO_ROOT, CHECK_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


from check_stage4_deleted_xml_invalidation_visualization import (  # noqa: E402
    _make_fixture as make_v6_fixture,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256  # noqa: E402
from pseudo3d.analysis.validate_stage4_crop_quality_manifests import (  # noqa: E402
    ACTION,
    MANIFEST_FIELDS,
    REASON_CODE,
    CropQualityInvalidation,
)
from pseudo3d.annotation.apply_crop_quality_invalidations import (  # noqa: E402
    OUTPUT_TOKEN,
    apply_crop_quality_invalidations_to_h5,
)
from pseudo3d.export.export_stage4_point_label_visualization import (  # noqa: E402
    CROP_INVALIDATION_LABEL_AUTHORITY,
    SUPPRESSED_CROP_CVAT_STATUS,
    export_stage4_point_label_visualization,
)


VIDEO = "synthetic_xml_invalidation"


def _write_manifest(path: Path, item: CropQualityInvalidation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(MANIFEST_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(item.to_row())


def _make_v7_fixture(root: Path) -> dict[str, Path]:
    paths = make_v6_fixture(root)
    v6_h5 = paths["output_h5"]
    with h5py.File(v6_h5, "r+") as handle:
        rows = handle["frame_annotation/frame_order"][:].astype(np.int64)
        target = np.flatnonzero(rows == 0)
        assert target.size == 1
        handle["frame_annotation/bbox_local_xyxy"][int(target[0])] = [
            15.0,
            1.0,
            15.0,
            5.0,
        ]
        point_rows = handle["point_cloud/frame_order"][:].astype(np.int64) == 0
        labels = handle["annotation/point_label"][:].astype(np.int8)
        labels[point_rows] = [0, -1, 0]
        handle["annotation/point_label"][...] = labels
        handle["annotation/valid_mask"][...] = labels != -1
        handle.attrs["num_labeled_points"] = int(np.sum(labels == 1))
        frames = handle["manual_review_fullvideo_frames"]
        frames["positive_points"][0] = 0
        frames["ignore_points"][0] = 1
        frames["background_points"][0] = 2

    item = CropQualityInvalidation(
        schema_version=1,
        video_name=VIDEO,
        frame_order=0,
        frame_index=0,
        frame_stem=f"{VIDEO}__fo00000__fi00000000",
        expected_saved_bbox_rows=1,
        expected_stray_ignore_points=1,
        expected_crop_status="fully_outside_crop",
        expected_visible_fraction=0.0,
        action=ACTION,
        reason_code=REASON_CODE,
        recoverable=True,
        notes="synthetic crop-quality tombstone",
    )
    manifest = root / "crop_quality_invalidations.csv"
    _write_manifest(manifest, item)
    output_h5 = (
        root
        / "v7"
        / "annotated"
        / VIDEO
        / f"{VIDEO}_pointcloud_annotated_{OUTPUT_TOKEN}.h5"
    )
    apply_crop_quality_invalidations_to_h5(
        source_h5=v6_h5,
        output_h5=output_h5,
        manifest_path=manifest,
        manifest_sha256=file_sha256(manifest),
        items=[item],
    )
    return {**paths, "v6_h5": v6_h5, "output_h5": output_h5}


def test_v7_render_contract(root: Path) -> None:
    paths = _make_v7_fixture(root)
    input_hashes = {
        name: file_sha256(paths[name])
        for name in ("v6_h5", "output_h5", "pseudo3d_h5", "annotation_zip")
    }
    output_dir = root / "visualization"
    summary = export_stage4_point_label_visualization(
        paths["output_h5"],
        output_dir,
        point_radius=0,
        point_alpha=1.0,
        cvat_review_root=paths["review_root"],
        cvat_snapshot_root=paths["snapshot_root"],
        cvat_mask_alpha=1.0,
        require_cvat_authoritative=True,
    )
    assert summary["contour_teacher_schema"] == OUTPUT_TOKEN
    assert summary["xml_invalidation_frames"] == 1
    assert summary["crop_invalidation_frames"] == 1
    assert summary["suppressed_cvat_mask_frames"] == 2
    assert summary["suppressed_cvat_mask_pixels"] == 16
    assert summary["cvat_mask_frames"] == 1
    assert summary["cvat_authoritative_frames"] == 3
    assert summary["cvat_label_authority"] == (
        "cvat_snapshot_with_xml_and_crop_invalidation"
    )
    assert summary["positive_outside_cvat_mask_points"] == 0
    assert summary["cvat_point_mask_mismatch_points"] == 0

    with (output_dir / "frame_labels.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    crop_rows = [row for row in rows if row["frame_order"] == "0"]
    assert len(crop_rows) == 1
    row = crop_rows[0]
    assert row["bbox_index"] == ""
    assert row["positive_points"] == "0"
    assert row["ignore_points"] == "0"
    assert row["background_points"] == "3"
    assert row["label_authority"] == CROP_INVALIDATION_LABEL_AUTHORITY
    assert row["xml_invalidated"] == "0"
    assert row["crop_invalidated"] == "1"
    assert row["crop_invalidation_action"] == ACTION
    assert row["crop_invalidation_reason_code"] == REASON_CODE
    assert row["cvat_mask_applied"] == "0"
    assert row["cvat_mask_suppressed"] == "1"
    assert row["cvat_mask_status"] == SUPPRESSED_CROP_CVAT_STATUS

    image = imageio.imread(
        output_dir / "frames" / "annotation_frame_00000.png"
    )
    np.testing.assert_array_equal(image[2, 2], [80, 140, 200])
    np.testing.assert_array_equal(image[3, 3], [20, 20, 20])
    for name, expected in input_hashes.items():
        assert file_sha256(paths[name]) == expected
    print(
        "[OK] crop-invalidated frame hides stale CVAT mask/BBox and renders "
        "saved all-background labels with v7 provenance"
    )


def test_v7_batch_and_verified_resume(root: Path) -> None:
    paths = _make_v7_fixture(root)
    annotated_root = paths["output_h5"].parents[1]
    output_root = root / "batch_visualization"
    summary_csv = output_root / "summary.csv"
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py"
        ),
        "--annotated_dir",
        str(annotated_root),
        "--pattern",
        f"*{OUTPUT_TOKEN}.h5",
        "--recursive",
        "--output_root",
        str(output_root),
        "--summary_csv",
        str(summary_csv),
        "--expected_files",
        "1",
        "--cvat_review_root",
        str(paths["review_root"]),
        "--cvat_snapshot_root",
        str(paths["snapshot_root"]),
        "--require_cvat_authoritative",
        "--require_xml_invalidation",
        "--require_crop_quality_invalidation",
        "--expected_xml_invalidated_frames",
        "1",
        "--expected_xml_invalidated_videos",
        "1",
        "--expected_crop_invalidated_frames",
        "1",
        "--expected_crop_invalidated_videos",
        "1",
        "--expected_suppressed_cvat_frames",
        "2",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    completed = subprocess.run(
        command + ["--skip_existing"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped_verified"
    assert rows[0]["xml_invalidation_frames"] == "1"
    assert rows[0]["crop_invalidation_frames"] == "1"
    assert rows[0]["suppressed_cvat_mask_frames"] == "2"
    print("[OK] v7 visualization batch counts and verified resume")


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="stage4_crop_quality_invalidation_visualization_"
    ) as directory:
        root = Path(directory)
        test_v7_render_contract(root / "render")
        test_v7_batch_and_verified_resume(root / "batch")
    print("Stage 4 crop-quality invalidation saved-label checks passed.")


if __name__ == "__main__":
    main()
