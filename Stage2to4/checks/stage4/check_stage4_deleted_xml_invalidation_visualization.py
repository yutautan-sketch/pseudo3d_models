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


from check_stage4_deleted_xml_invalidation_apply import (  # noqa: E402
    _manifest,
    _write_source,
)
from pseudo3d.analysis.build_stage4_deleted_xml_invalidation_manifest import (  # noqa: E402
    ACTION,
    REASON_CODE,
    XmlFrameInvalidation,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256  # noqa: E402
from pseudo3d.annotation.apply_deleted_xml_invalidations import (  # noqa: E402
    OUTPUT_TOKEN,
    SOURCE_TOKEN,
    apply_invalidations_to_h5,
)
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (  # noqa: E402
    convert_masks_to_cvat_zip,
)
from pseudo3d.export.export_stage4_point_label_visualization import (  # noqa: E402
    SUPPRESSED_CVAT_STATUS,
    export_stage4_point_label_visualization,
)


VIDEO = "synthetic_xml_invalidation"


def _replace_text(group: h5py.Group, name: str, values: list[str]) -> None:
    if name in group:
        del group[name]
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=h5py.string_dtype("utf-8")),
    )


def _make_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    missing_xml = root / "voc" / "annotations" / f"{VIDEO}_00002.xml"
    source_h5 = (
        root
        / "v5"
        / VIDEO
        / f"{VIDEO}_pointcloud_annotated_{SOURCE_TOKEN}.h5"
    )
    _write_source(source_h5, missing_xml, video=VIDEO)

    pseudo3d_h5 = root / "pseudo" / f"{VIDEO}_ts448_oym96_corr.h5"
    pseudo3d_h5.parent.mkdir(parents=True)
    images = np.zeros((3, 12, 16), dtype=np.uint8)
    images[0] = 20
    images[1] = 40
    images[2] = 60
    with h5py.File(pseudo3d_h5, "w") as handle:
        handle.create_dataset("local_encoder_images", data=images)
        handle.create_dataset("frame_indices", data=np.arange(3, dtype=np.int64))

    pixel_xy = np.asarray(
        [
            [2, 2],
            [4, 2],
            [6, 2],
            [2, 5],
            [4, 5],
            [6, 5],
            [2, 8],
            [4, 8],
            [6, 8],
        ],
        dtype=np.float32,
    )
    review_root = root / "review"
    images_dir = review_root / "videos" / VIDEO / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    stems = [f"{VIDEO}__fo{order:05d}__fi{order:08d}" for order in range(3)]
    masks = [np.zeros((12, 16), dtype=np.uint8) for _ in range(3)]
    masks[0][2, 2] = 255
    masks[1][4:7, 1:6] = 255
    masks[2][8, 6] = 255
    for order, stem in enumerate(stems):
        imageio.imwrite(images_dir / f"{stem}.png", images[order])
        imageio.imwrite(masks_dir / f"{stem}.png", masks[order])

    snapshot_root = root / "snapshot"
    task_id = 17
    annotation_zip = (
        snapshot_root
        / "annotations"
        / VIDEO
        / f"{VIDEO}__task{task_id}__reviewed_segmentation_mask_1_1.zip"
    )
    convert_masks_to_cvat_zip(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_zip=annotation_zip,
    )
    annotation_sha = file_sha256(annotation_zip)
    snapshot_manifest = snapshot_root / "export_manifest.csv"
    snapshot_manifest.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video_name", "task_id", "status", "annotation_sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": VIDEO,
                "task_id": task_id,
                "status": "complete",
                "annotation_sha256": annotation_sha,
            }
        )

    with h5py.File(source_h5, "r+") as handle:
        handle.attrs["source_pseudo3d_h5"] = str(pseudo3d_h5.resolve())
        handle.attrs["manual_review_fullvideo_snapshot_manifest_sha256"] = (
            file_sha256(snapshot_manifest)
        )
        handle.attrs["manual_review_fullvideo_authoritative_frame_count"] = 3
        handle.attrs[
            "manual_review_fullvideo_positive_outside_cvat_mask_points"
        ] = 0
        handle["point_cloud"].create_dataset("pixel_xy", data=pixel_xy)
        handle["point_cloud"].create_dataset(
            "frame_index", data=np.repeat(np.arange(3), 3).astype(np.int64)
        )
        handle["annotation/point_label"][6:9] = [0, 0, 1]
        handle["annotation/valid_mask"][6:9] = [True, True, True]
        _replace_text(
            handle["frame_annotation"],
            "selected_contour_source",
            ["manual_cvat_fullvideo_v1"] * 3,
        )
        manual = handle["manual_review_fullvideo"]
        manual.attrs["task_id"] = task_id
        manual.attrs["annotation_zip_sha256"] = annotation_sha
        frames = handle["manual_review_fullvideo_frames"]
        frames.attrs["task_id"] = task_id
        frames.attrs["annotation_zip_sha256"] = annotation_sha
        frames.attrs["label_authority"] = "cvat_snapshot"
        _replace_text(frames, "cvat_mask_status", ["positive"] * 3)
        frames["cvat_mask_positive_pixels"][:] = [1, 15, 1]
        frames["ignore_points"][2] = 0
        frames["background_points"][2] = 2

    item = XmlFrameInvalidation(
        schema_version=1,
        video_name=VIDEO,
        frame_order=1,
        frame_index=1,
        frame_stem=stems[1],
        expected_xml_name=missing_xml.name,
        expected_saved_bbox_rows=2,
        action=ACTION,
        reason_code=REASON_CODE,
        notes="synthetic intentional XML deletion",
    )
    invalidation_manifest = root / "invalidations.csv"
    _manifest(invalidation_manifest, item)
    output_h5 = (
        root
        / "v6"
        / "annotated"
        / VIDEO
        / f"{VIDEO}_pointcloud_annotated_{OUTPUT_TOKEN}.h5"
    )
    apply_invalidations_to_h5(
        source_h5=source_h5,
        output_h5=output_h5,
        manifest_path=invalidation_manifest,
        manifest_sha256=file_sha256(invalidation_manifest),
        items=[item],
    )
    return {
        "source_h5": source_h5,
        "output_h5": output_h5,
        "pseudo3d_h5": pseudo3d_h5,
        "review_root": review_root,
        "snapshot_root": snapshot_root,
        "annotation_zip": annotation_zip,
    }


def test_v6_invalidation_render_contract(root: Path) -> None:
    paths = _make_fixture(root)
    input_hashes = {
        name: file_sha256(path)
        for name, path in paths.items()
        if name in {"source_h5", "output_h5", "pseudo3d_h5", "annotation_zip"}
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
    assert summary["suppressed_cvat_mask_frames"] == 1
    assert summary["suppressed_cvat_mask_pixels"] == 15
    assert summary["cvat_mask_frames"] == 2
    assert summary["cvat_authoritative_frames"] == 3
    assert summary["cvat_label_authority"] == (
        "cvat_snapshot_with_xml_invalidation"
    )
    assert summary["positive_outside_cvat_mask_points"] == 0
    assert summary["cvat_point_mask_mismatch_points"] == 0

    with (output_dir / "frame_labels.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    target = [row for row in rows if row["frame_order"] == "1"]
    assert len(target) == 1
    row = target[0]
    assert row["bbox_index"] == ""
    assert row["positive_points"] == "0"
    assert row["ignore_points"] == "0"
    assert row["background_points"] == "3"
    assert row["label_authority"] == "xml_deletion_manifest"
    assert row["xml_invalidated"] == "1"
    assert row["xml_invalidation_action"] == ACTION
    assert row["xml_invalidation_reason_code"] == REASON_CODE
    assert row["cvat_mask_applied"] == "0"
    assert row["cvat_mask_suppressed"] == "1"
    assert row["cvat_mask_status"] == SUPPRESSED_CVAT_STATUS

    image = imageio.imread(
        output_dir / "frames" / "annotation_frame_00001.png"
    )
    np.testing.assert_array_equal(image[4, 1], [40, 40, 40])
    np.testing.assert_array_equal(image[5, 2], [80, 140, 200])
    for name, expected in input_hashes.items():
        assert file_sha256(paths[name]) == expected
    print(
        "[OK] invalidated frame hides stale CVAT mask/BBox and renders saved "
        "all-background labels with tombstone provenance"
    )


def test_v6_batch_and_verified_resume(root: Path) -> None:
    paths = _make_fixture(root)
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
        "--expected_xml_invalidated_frames",
        "1",
        "--expected_xml_invalidated_videos",
        "1",
        "--expected_suppressed_cvat_frames",
        "1",
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
    assert rows[0]["suppressed_cvat_mask_frames"] == "1"
    print("[OK] v6 visualization batch counts and verified resume")


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="stage4_deleted_xml_invalidation_visualization_"
    ) as directory:
        root = Path(directory)
        test_v6_invalidation_render_contract(root / "render")
        test_v6_batch_and_verified_resume(root / "batch")
    print("Stage 4 deleted-XML invalidation Step 5 synthetic checks passed.")


if __name__ == "__main__":
    main()
