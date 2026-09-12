from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import imageio.v2 as imageio
import numpy as np

from pseudo3d.export.export_stage4_point_label_visualization import (
    LABEL_IGNORE,
    export_stage4_point_label_visualization,
    file_sha256,
    load_stage4_point_label_visualization_input,
    render_point_labels,
)
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    convert_masks_to_cvat_zip,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_text(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=h5py.string_dtype("utf-8")),
    )


def make_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    pseudo3d_h5 = root / "synthetic_ts448_oym96_corr.h5"
    images = np.zeros((3, 12, 16), dtype=np.uint8)
    images[0] = 20
    images[1] = 40
    images[2] = 60
    with h5py.File(pseudo3d_h5, "w") as handle:
        handle.create_dataset("local_encoder_images", data=images)
        handle.create_dataset(
            "frame_indices", data=np.asarray([10, 20, 40], dtype=np.int64)
        )

    annotated_h5 = root / "synthetic" / (
        "synthetic_pointcloud_annotated_foreground_combined_v2_"
        "global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1.h5"
    )
    annotated_h5.parent.mkdir()
    pixel_xy = np.asarray(
        [
            [2, 2],
            [4, 2],
            [6, 2],
            [8, 2],
            [3, 8],
            [5, 8],
            [7, 8],
            [2, 5],
            [5, 5],
        ],
        dtype=np.float32,
    )
    frame_order = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
    frame_index = np.asarray([10, 10, 10, 10, 20, 20, 20, 40, 40], dtype=np.int64)
    labels = np.asarray([-1, 0, 1, 1, -1, 0, 0, 0, 0], dtype=np.int8)
    with h5py.File(annotated_h5, "w") as handle:
        handle.attrs["video_name"] = "synthetic"
        handle.attrs["source_pseudo3d_h5"] = str(pseudo3d_h5.resolve())
        handle.attrs["contour_teacher_schema"] = "bboxrank_v4_manual_fullvideo_v1"
        pc = handle.create_group("point_cloud")
        pc.create_dataset("pixel_xy", data=pixel_xy)
        pc.create_dataset("frame_order", data=frame_order)
        pc.create_dataset("frame_index", data=frame_index)
        ann = handle.create_group("annotation")
        ann.create_dataset("point_label", data=labels)
        ann.create_dataset("valid_mask", data=labels != -1)
        frame = handle.create_group("frame_annotation")
        frame.create_dataset("frame_order", data=np.asarray([0, 0, 1], dtype=np.int32))
        frame.create_dataset("frame_index", data=np.asarray([10, 10, 20], dtype=np.int64))
        frame.create_dataset("bbox_index", data=np.asarray([0, 1, 0], dtype=np.int32))
        frame.create_dataset(
            "bbox_local_xyxy",
            data=np.asarray(
                [[1, 1, 9, 6], [10, 1, 14, 6], [1, 7, 9, 10]],
                dtype=np.float32,
            ),
        )
        frame.create_dataset("valid_contour", data=np.asarray([1, 1, 0], dtype=bool))
        _write_text(
            frame,
            "selected_contour_source",
            ["manual_cvat_fullvideo_v1", "global", "manual_cvat_fullvideo_empty_v1"],
        )
        _write_text(
            frame,
            "annotation_reason",
            ["manual_review_completed", "bbox_ranked_selected_global", "manual_review_empty"],
        )
    return annotated_h5, pseudo3d_h5


def _relative_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def make_cvat_artifacts(
    root: Path,
    annotated_h5: Path,
    pseudo3d_h5: Path,
) -> tuple[Path, Path, np.ndarray]:
    review_root = root / "review"
    images_dir = review_root / "videos" / "synthetic" / "images"
    masks_dir = root / "cvat_masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    stems = [
        "synthetic__fo00000__fi00000010",
        "synthetic__fo00001__fi00000020",
        "synthetic__fo00002__fi00000040",
    ]
    with h5py.File(pseudo3d_h5, "r") as handle:
        images = handle["local_encoder_images"][:]
    applied_mask = np.zeros(images[0].shape, dtype=np.uint8)
    applied_mask[2:5, 2:8] = 255
    for order, stem in enumerate(stems):
        imageio.imwrite(images_dir / f"{stem}.png", images[order])
        mask = applied_mask if order == 0 else np.zeros_like(applied_mask)
        imageio.imwrite(masks_dir / f"{stem}.png", mask)

    snapshot_root = root / "snapshot"
    task_id = 7
    annotation_zip = (
        snapshot_root
        / "annotations"
        / "synthetic"
        / f"synthetic__task{task_id}__reviewed_segmentation_mask_1_1.zip"
    )
    convert_masks_to_cvat_zip(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_zip=annotation_zip,
    )
    annotation_sha = file_sha256(annotation_zip)
    manifest_path = snapshot_root / "export_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video_name", "task_id", "status", "annotation_sha256"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": "synthetic",
                "task_id": task_id,
                "status": "complete",
                "annotation_sha256": annotation_sha,
            }
        )
    with h5py.File(annotated_h5, "r+") as handle:
        handle.attrs["manual_review_fullvideo_snapshot_manifest_sha256"] = (
            file_sha256(manifest_path)
        )
        manual = handle.create_group("manual_review_fullvideo")
        manual.attrs["reviewed_video"] = True
        manual.attrs["task_id"] = task_id
        manual.attrs["annotation_zip_sha256"] = annotation_sha
        _write_text(manual, "frame_stem", [stems[0]])
        manual.create_dataset("frame_order", data=np.asarray([0], dtype=np.int32))
        manual.create_dataset("frame_index", data=np.asarray([10], dtype=np.int64))
        manual.create_dataset("bbox_index", data=np.asarray([0], dtype=np.int32))
        _write_text(manual, "decision", ["manual_positive"])
    return review_root, snapshot_root, applied_mask.astype(bool)


def make_authoritative_cvat_artifacts(
    root: Path,
    annotated_h5: Path,
    pseudo3d_h5: Path,
) -> tuple[Path, Path, dict[int, np.ndarray]]:
    review_root = root / "review"
    images_dir = review_root / "videos" / "synthetic" / "images"
    masks_dir = root / "cvat_masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    stems = [
        "synthetic__fo00000__fi00000010",
        "synthetic__fo00001__fi00000020",
        "synthetic__fo00002__fi00000040",
    ]
    with h5py.File(pseudo3d_h5, "r") as handle:
        images = handle["local_encoder_images"][:]
    masks = {order: np.zeros(images[order].shape, dtype=np.uint8) for order in range(3)}
    masks[0][2:5, 2:8] = 255
    masks[2][4:7, 1:4] = 255
    for order, stem in enumerate(stems):
        imageio.imwrite(images_dir / f"{stem}.png", images[order])
        imageio.imwrite(masks_dir / f"{stem}.png", masks[order])

    snapshot_root = root / "snapshot"
    task_id = 8
    annotation_zip = (
        snapshot_root
        / "annotations"
        / "synthetic"
        / f"synthetic__task{task_id}__reviewed_segmentation_mask_1_1.zip"
    )
    convert_masks_to_cvat_zip(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_zip=annotation_zip,
    )
    annotation_sha = file_sha256(annotation_zip)
    manifest_path = snapshot_root / "export_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video_name", "task_id", "status", "annotation_sha256"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": "synthetic",
                "task_id": task_id,
                "status": "complete",
                "annotation_sha256": annotation_sha,
            }
        )
    labels = np.asarray([1, 1, 1, -1, -1, -1, -1, 1, 0], dtype=np.int8)
    with h5py.File(annotated_h5, "r+") as handle:
        handle.attrs["contour_teacher_schema"] = (
            "bboxrank_v5_cvat_authoritative_v1"
        )
        handle.attrs["manual_review_fullvideo_snapshot_manifest_sha256"] = (
            file_sha256(manifest_path)
        )
        handle.attrs["manual_review_fullvideo_authoritative_frame_count"] = 3
        handle.attrs[
            "manual_review_fullvideo_positive_outside_cvat_mask_points"
        ] = 0
        handle["annotation/point_label"][:] = labels
        handle["annotation/valid_mask"][:] = labels != -1
        handle["annotation"].attrs["label_authority"] = "cvat_snapshot"
        manual = handle.create_group("manual_review_fullvideo")
        manual.attrs["reviewed_video"] = True
        manual.attrs["task_id"] = task_id
        manual.attrs["annotation_zip_sha256"] = annotation_sha
        _write_text(manual, "frame_stem", [stems[0]])
        manual.create_dataset("frame_order", data=np.asarray([0], dtype=np.int32))
        manual.create_dataset("frame_index", data=np.asarray([10], dtype=np.int64))
        manual.create_dataset("bbox_index", data=np.asarray([0], dtype=np.int32))
        _write_text(manual, "decision", ["manual_positive"])

        frames = handle.create_group("manual_review_fullvideo_frames")
        frames.attrs["reviewed_video"] = True
        frames.attrs["task_id"] = task_id
        frames.attrs["annotation_zip_sha256"] = annotation_sha
        frames.attrs["label_authority"] = "cvat_snapshot"
        _write_text(frames, "frame_stem", stems)
        _write_text(frames, "frame_role", ["review_target", "context_only", "context_only"])
        _write_text(frames, "cvat_mask_status", ["positive", "empty", "positive"])
        _write_text(frames, "label_authority", ["cvat_snapshot"] * 3)
        for name, values in {
            "frame_order": [0, 1, 2],
            "frame_index": [10, 20, 40],
            "cvat_mask_positive_pixels": [int(masks[0].astype(bool).sum()), 0, int(masks[2].astype(bool).sum())],
            "positive_points": [3, 0, 1],
            "ignore_points": [1, 3, 0],
            "background_points": [0, 0, 1],
            "positive_outside_cvat_mask_points": [0, 0, 0],
        }.items():
            frames.create_dataset(name, data=np.asarray(values, dtype=np.int64))
    return review_root, snapshot_root, {
        order: mask.astype(bool) for order, mask in masks.items()
    }


def test_render_colors() -> None:
    gray = np.zeros((7, 9), dtype=np.uint8)
    cvat_mask = np.zeros_like(gray, dtype=bool)
    cvat_mask[1, 1] = True
    cvat_mask[1, 7] = True
    cvat_mask[4, 4] = True
    image = render_point_labels(
        gray,
        pixel_xy=np.asarray([[1, 1], [4, 1], [7, 1]], dtype=np.float32),
        labels=np.asarray([-1, 0, 1], dtype=np.int8),
        radius=0,
        alpha=1.0,
        positive_color=(255, 32, 32),
        ignore_color=(255, 210, 0),
        background_color=(80, 140, 200),
        cvat_mask=cvat_mask,
        cvat_color=(0, 255, 255),
        cvat_alpha=1.0,
    )
    np.testing.assert_array_equal(image[1, 1], (0, 255, 255))
    np.testing.assert_array_equal(image[1, 4], (80, 140, 200))
    np.testing.assert_array_equal(image[1, 7], (255, 32, 32))
    np.testing.assert_array_equal(image[4, 4], (0, 255, 255))
    print("[OK] yellow -> CVAT cyan -> positive red overlay order")


def test_cvat_mask_integration(root: Path) -> None:
    annotated_h5, pseudo3d_h5 = make_fixture(root)
    review_root, snapshot_root, applied_mask = make_cvat_artifacts(
        root, annotated_h5, pseudo3d_h5
    )
    output = root / "output"
    summary = export_stage4_point_label_visualization(
        annotated_h5,
        output,
        point_radius=0,
        point_alpha=1.0,
        cvat_review_root=review_root,
        cvat_snapshot_root=snapshot_root,
        cvat_mask_alpha=1.0,
    )
    assert summary["cvat_masks_requested"] is True
    assert summary["cvat_mask_frames"] == 1
    assert summary["cvat_mask_pixels"] == int(applied_mask.sum())
    image = imageio.imread(output / "frames" / "annotation_frame_00000.png")
    np.testing.assert_array_equal(image[4, 5], (0, 255, 255))
    np.testing.assert_array_equal(image[2, 6], (255, 32, 32))
    with (output / "frame_labels.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    frame_zero = [row for row in rows if row["frame_order"] == "0"]
    assert {row["cvat_mask_applied"] for row in frame_zero} == {"1"}
    assert {int(row["cvat_mask_pixels"]) for row in frame_zero} == {
        int(applied_mask.sum())
    }
    print("[OK] corrected CVAT ZIP mask loading, checksum, and frame alignment")


def test_authoritative_cvat_mask_integration(root: Path) -> None:
    annotated_h5, pseudo3d_h5 = make_fixture(root)
    review_root, snapshot_root, masks = make_authoritative_cvat_artifacts(
        root, annotated_h5, pseudo3d_h5
    )
    output = root / "output"
    summary = export_stage4_point_label_visualization(
        annotated_h5,
        output,
        point_radius=0,
        point_alpha=1.0,
        cvat_review_root=review_root,
        cvat_snapshot_root=snapshot_root,
        cvat_mask_alpha=1.0,
        require_cvat_authoritative=True,
    )
    assert summary["cvat_label_authority"] == "cvat_snapshot"
    assert summary["cvat_mask_frames"] == 3
    assert summary["cvat_authoritative_frames"] == 3
    assert summary["cvat_mask_nonempty_frames"] == 2
    assert summary["cvat_mask_status_counts"] == {"empty": 1, "positive": 2}
    assert summary["positive_outside_cvat_mask_points"] == 0
    assert summary["cvat_point_mask_mismatch_points"] == 0
    assert summary["cvat_mask_pixels"] == sum(int(mask.sum()) for mask in masks.values())
    context_image = imageio.imread(
        output / "frames" / "annotation_frame_00002.png"
    )
    np.testing.assert_array_equal(context_image[5, 2], (255, 32, 32))
    np.testing.assert_array_equal(context_image[4, 3], (0, 255, 255))
    with (output / "frame_labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_order = {int(row["frame_order"]): row for row in rows if row["bbox_index"] == ""}
    assert by_order[2]["cvat_mask_status"] == "positive"
    assert by_order[2]["positive_points"] == "1"
    assert by_order[2]["positive_outside_cvat_mask_points"] == "0"

    with h5py.File(annotated_h5, "r+") as handle:
        handle["annotation/point_label"][8] = 1
        handle["manual_review_fullvideo_frames/positive_points"][2] = 2
        handle["manual_review_fullvideo_frames/background_points"][2] = 0
    try:
        export_stage4_point_label_visualization(
            annotated_h5,
            root / "invalid_output",
            cvat_review_root=review_root,
            cvat_snapshot_root=snapshot_root,
            require_cvat_authoritative=True,
        )
    except ValueError as exc:
        assert "outside authoritative CVAT mask" in str(exc)
    else:
        raise AssertionError("Positive point center outside CVAT mask must fail")
    print(
        "[OK] authoritative all-frame CVAT overlay and point-center subset contract"
    )


def test_end_to_end(root: Path) -> None:
    annotated_h5, pseudo3d_h5 = make_fixture(root)
    before = file_sha256(annotated_h5)
    pseudo3d_before = file_sha256(pseudo3d_h5)
    loaded = load_stage4_point_label_visualization_input(annotated_h5)
    assert loaded["pseudo3d_h5"] == pseudo3d_h5.resolve()
    assert loaded["video_name"] == "synthetic"

    first = root / "output_1"
    second = root / "output_2"
    summary_1 = export_stage4_point_label_visualization(
        annotated_h5,
        first,
        point_radius=0,
        point_alpha=1.0,
    )
    summary_2 = export_stage4_point_label_visualization(
        annotated_h5,
        second,
        point_radius=0,
        point_alpha=1.0,
    )
    assert file_sha256(annotated_h5) == before
    assert file_sha256(pseudo3d_h5) == pseudo3d_before
    assert summary_1 == summary_2
    assert summary_1["automatic_contours_recomputed"] is False
    assert summary_1["input_files_modified"] is False
    assert summary_1["num_frames"] == 3
    assert summary_1["num_bbox_rows"] == 3
    assert summary_1["num_points"] == 9
    assert summary_1["positive_points"] == 2
    assert summary_1["ignore_points"] == 2
    assert summary_1["background_points"] == 5
    assert summary_1["source_counts"] == {
        "global": 1,
        "manual_cvat_fullvideo_empty_v1": 1,
        "manual_cvat_fullvideo_v1": 1,
    }

    files_1 = _relative_files(first)
    files_2 = _relative_files(second)
    assert files_1 == files_2
    for relative in files_1:
        assert _sha(first / relative) == _sha(second / relative), relative
    assert len(list((first / "frames").glob("annotation_frame_*.png"))) == 3
    with (first / "frame_labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    no_bbox = [row for row in rows if row["frame_order"] == "2"]
    assert len(no_bbox) == 1
    assert no_bbox[0]["bbox_index"] == ""
    assert no_bbox[0]["background_points"] == "2"
    manual_empty = [
        row
        for row in rows
        if row["selected_contour_source"] == "manual_cvat_fullvideo_empty_v1"
    ]
    assert len(manual_empty) == 1
    assert manual_empty[0]["positive_points"] == "0"

    frame_zero = imageio.imread(first / "frames" / "annotation_frame_00000.png")
    np.testing.assert_array_equal(frame_zero[2, 2], (255, 210, 0))
    np.testing.assert_array_equal(frame_zero[2, 4], (80, 140, 200))
    np.testing.assert_array_equal(frame_zero[2, 6], (255, 32, 32))
    # BBox comes from the saved H5 and is drawn after point labels.
    np.testing.assert_array_equal(frame_zero[1, 1], (160, 255, 80))
    print("[OK] saved labels, saved BBoxes, provenance CSV, and deterministic output")


def test_validation(root: Path) -> None:
    annotated_h5, _ = make_fixture(root / "invalid")
    with h5py.File(annotated_h5, "r+") as handle:
        handle["annotation/valid_mask"][0] = True
    try:
        load_stage4_point_label_visualization_input(annotated_h5)
    except ValueError as exc:
        assert "valid_mask differs" in str(exc)
    else:
        raise AssertionError("valid_mask mismatch must be rejected")

    annotated_h5, _ = make_fixture(root / "unknown")
    with h5py.File(annotated_h5, "r+") as handle:
        values = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["frame_annotation/selected_contour_source"][:]
        ]
        values[0] = "mystery"
        del handle["frame_annotation/selected_contour_source"]
        _write_text(handle["frame_annotation"], "selected_contour_source", values)
    try:
        load_stage4_point_label_visualization_input(annotated_h5)
    except ValueError as exc:
        assert "Unknown selected_contour_source" in str(exc)
    else:
        raise AssertionError("unknown provenance must be rejected")

    annotated_h5, _ = make_fixture(root / "no_bbox")
    with h5py.File(annotated_h5, "r+") as handle:
        handle["annotation/point_label"][7] = LABEL_IGNORE
        handle["annotation/valid_mask"][7] = False
    try:
        load_stage4_point_label_visualization_input(annotated_h5)
    except ValueError as exc:
        assert "BBox-free frames contain non-background labels" in str(exc)
    else:
        raise AssertionError("non-background label on BBox-free frame must be rejected")

    source = (
        REPO_ROOT
        / "pseudo3d/export/export_stage4_point_label_visualization.py"
    ).read_text(encoding="utf-8")
    assert "build_frame_contour_mask_results" not in source
    assert "load_voc_bboxes" not in source
    print(
        "[OK] label/provenance/no-BBox validation and no contour/XML recomputation"
    )


def test_batch_cli(root: Path) -> None:
    annotated_h5, _ = make_fixture(root / "batch_fixture")
    output_root = root / "batch_output"
    summary_csv = output_root / "summary.csv"
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py"
        ),
        "--annotated_dir",
        str(annotated_h5.parent),
        "--pattern",
        "*.h5",
        "--output_root",
        str(output_root),
        "--summary_csv",
        str(summary_csv),
        "--expected_files",
        "1",
        "--point_radius",
        "0",
        "--point_alpha",
        "1",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "processed"
    assert rows[0]["video_name"] == "synthetic"
    assert rows[0]["positive_points"] == "2"
    summary = json.loads(
        (output_root / "synthetic" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["automatic_contours_recomputed"] is False

    completed = subprocess.run(
        command + ["--skip_existing"], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "skipped_verified"
    print("[OK] batch CLI, summary, and verified resume")


def test_authoritative_batch_cli(root: Path) -> None:
    annotated_h5, pseudo3d_h5 = make_fixture(root / "fixture")
    review_root, snapshot_root, _ = make_authoritative_cvat_artifacts(
        root / "fixture", annotated_h5, pseudo3d_h5
    )
    output_root = root / "output"
    summary_csv = output_root / "summary.csv"
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py"
        ),
        "--annotated_dir",
        str(annotated_h5.parent),
        "--pattern",
        "*.h5",
        "--output_root",
        str(output_root),
        "--summary_csv",
        str(summary_csv),
        "--expected_files",
        "1",
        "--point_radius",
        "0",
        "--cvat_review_root",
        str(review_root),
        "--cvat_snapshot_root",
        str(snapshot_root),
        "--require_cvat_authoritative",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["cvat_label_authority"] == "cvat_snapshot"
    assert rows[0]["cvat_authoritative_frames"] == "3"
    assert rows[0]["positive_outside_cvat_mask_points"] == "0"
    assert rows[0]["cvat_point_mask_mismatch_points"] == "0"
    completed = subprocess.run(
        command + ["--skip_existing"], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "skipped_verified"
    print("[OK] authoritative batch CLI, summary, and verified resume")


def main() -> None:
    test_render_colors()
    with tempfile.TemporaryDirectory(prefix="stage4_point_label_vis_") as directory:
        root = Path(directory)
        test_cvat_mask_integration(root / "cvat_integration")
        test_authoritative_cvat_mask_integration(root / "authoritative_integration")
        test_end_to_end(root / "end_to_end")
        test_validation(root / "validation")
        test_batch_cli(root / "batch")
        test_authoritative_batch_cli(root / "authoritative_batch")
    print("Stage 4 saved point-label visualization synthetic checks passed.")


if __name__ == "__main__":
    main()
