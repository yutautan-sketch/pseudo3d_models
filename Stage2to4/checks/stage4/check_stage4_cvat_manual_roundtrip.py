from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import h5py
import numpy as np
import yaml

from checks.stage4.check_stage4_contour_auto_refine import (
    _phase3_args,
    _write_config,
)
from checks.stage4.check_stage4_contour_teacher_audit import (
    _args as phase1_args,
    _make_integration_fixture,
)
from pseudo3d.analysis.audit_stage4_contour_teacher import (
    load_audit_teacher_config,
    run_audit,
)
from pseudo3d.analysis.prototype_stage4_contour_auto_refine import (
    _checksum_rows as phase3_checksum_rows,
    _write_csv as phase3_write_csv,
    run_prototype,
)
from pseudo3d.annotation.import_cvat_segmentation_mask_corrections import (
    _resolve_context_only_stems,
    run_manual_import,
)
from pseudo3d.annotation.stage4_manual_review import (
    ManualReviewError,
    apply_corrected_frame_mask,
    file_sha256,
    load_manual_review_config,
    read_csv_rows,
)
from pseudo3d.batch.export.batch_export_stage4_manual_review_cvat import (
    CVAT_CLIPPED_BBOX_COLOR_BGR,
    CVAT_MANUAL_BBOX_COLOR_BGR,
    CVAT_OTHER_BBOX_COLOR_BGR,
    CVAT_OUTSIDE_BBOX_COLOR_BGR,
    _bbox_crop_metrics,
    _render_full_video_review_image,
    _render_cvat_review_image,
    run_review_export,
)
from pseudo3d.batch.export.rebuild_stage4_phase5_textfree_review_package import (
    RENDER_VERSION,
    run as run_textfree_migration,
)
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    convert_masks_to_cvat_zip,
)


def _force_one_manual_review(phase3_root: Path) -> None:
    decisions = read_csv_rows(phase3_root / "bbox_decisions.csv")
    assert len(decisions) == 2
    decisions[0]["proposed_decision"] = "manual_review"
    decisions[0]["reason_codes"] = "synthetic_manual_roundtrip"
    decisions[1]["proposed_decision"] = "auto_accept"
    phase3_write_csv(phase3_root / "bbox_decisions.csv", decisions)
    summary_path = phase3_root / "refine_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision_counts"] = {"auto_accept": 1, "manual_review": 1}
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase3_write_csv(
        phase3_root / "checksums.csv", phase3_checksum_rows(phase3_root)
    )


def _export_args(
    paths: dict[str, Path],
    phase1: Path,
    phase3: Path,
    refine: Path,
    output: Path,
    manual_config: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=paths["manifest"],
        annotated_root=paths["annotated_root"],
        phase1_audit_root=phase1,
        phase3_root=phase3,
        teacher_config=paths["teacher"],
        refine_config=refine,
        manual_review_config=manual_config,
        output_root=output,
        annotated_glob_template="{video_name}/*bboxrank_v2_nobbox_bg.h5",
        expected_videos=1,
        overwrite=False,
    )


def _read_binary(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None and image.ndim == 2 and image.dtype == np.uint8
    return image > 0


def _write_binary(path: Path, mask: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)
    assert ok


def _make_edited_export(review_root: Path, output_zip: Path) -> Path:
    edited = review_root.parent / "edited_masks"
    edited.mkdir()
    bbox_rows = read_csv_rows(review_root / "review_bboxes.csv")
    manual = next(row for row in bbox_rows if row["manual_review_required"] == "True")
    stem = manual["frame_stem"]
    for source in sorted((review_root / "masks").glob("*.png")):
        mask = _read_binary(source)
        if source.stem == stem:
            left, top, right, bottom = (
                int(value) for value in manual["bbox_bounds_ltrb"].split("|")
            )
            mask[top : min(bottom + 1, top + 4), left : min(right + 1, left + 4)] = True
            if right - left > 8 and bottom - top > 8:
                mask[top + 5, left + 5] = False
        _write_binary(edited / source.name, mask)
    convert_masks_to_cvat_zip(
        images_dir=review_root / "images",
        masks_dir=edited,
        output_zip=output_zip,
    )
    return output_zip


def _write_session(review_root: Path, edited_zip: Path, path: Path) -> Path:
    summary = json.loads((review_root / "export_summary.json").read_text(encoding="utf-8"))
    payload = {
        "review_schema_version": 1,
        "review_package_sha256": summary["review_package_sha256"],
        "cvat_version": "synthetic-local",
        "cvat_task_identifier": "synthetic-task-1",
        "reviewer_identifier": "synthetic-reviewer",
        "imported_zip_sha256": summary["cvat_import_zip_sha256"],
        "pre_import_backup_sha256": "synthetic-backup",
        "exported_zip_sha256": file_sha256(edited_zip),
        "review_started_utc": "2026-08-19T00:00:00+00:00",
        "review_completed_utc": "2026-08-19T00:10:00+00:00",
        "all_manual_bboxes_reviewed": True,
        "notes": "synthetic",
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _snapshot_non_annotation(source: Path) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    result: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    with h5py.File(source, "r") as handle:
        for group_name in ("point_cloud", "measurement"):
            group = handle[group_name]
            for name, dataset in group.items():
                value = dataset[()] if dataset.shape == () else dataset[:]
                result[f"{group_name}/{name}"] = (np.asarray(value), dict(dataset.attrs))
    return result


def test_pure_label_contract() -> None:
    labels = np.zeros(6, dtype=np.int8)
    orders = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    xy = np.asarray([[0, 0], [2, 2], [5, 5], [0, 0], [2, 2], [5, 5]], dtype=np.float32)
    corrected = np.zeros((6, 6), dtype=np.uint8)
    corrected[2, 2] = 1
    updated, stats = apply_corrected_frame_mask(
        point_labels=labels,
        frame_orders=orders,
        pixel_xy=xy,
        target_frame_order=0,
        corrected_mask=corrected,
        bboxes=[(1, 1, 4, 4)],
        manual_bbox_indices=[0],
    )
    np.testing.assert_array_equal(updated, [0, 1, 0, 0, 0, 0])
    assert stats["positive_points"] == 1
    outside = corrected.copy()
    outside[0, 0] = 1
    try:
        apply_corrected_frame_mask(
            point_labels=labels,
            frame_orders=orders,
            pixel_xy=xy,
            target_frame_order=0,
            corrected_mask=outside,
            bboxes=[(1, 1, 4, 4)],
            manual_bbox_indices=[0],
        )
    except ManualReviewError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("BBox-union outside correction must be rejected")
    try:
        apply_corrected_frame_mask(
            point_labels=labels,
            frame_orders=orders,
            pixel_xy=xy,
            target_frame_order=0,
            corrected_mask=np.zeros((6, 6), dtype=np.uint8),
            bboxes=[(1, 1, 4, 4)],
            manual_bbox_indices=[0],
        )
    except ManualReviewError as exc:
        assert "empty correction" in str(exc)
    else:
        raise AssertionError("Empty manual BBox correction must be rejected")
    print("[OK] strict mask-to-point labels, BBox union, and empty correction")


def test_cvat_review_image_bbox_overlay() -> None:
    image = np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)
    original = image.copy()
    rendered_1 = _render_cvat_review_image(
        image=image,
        bboxes=[(1.0, 1.0, 6.0, 5.0), (9.0, 3.0, 14.0, 10.0)],
        decisions=["manual_review", "auto_accept"],
    )
    rendered_2 = _render_cvat_review_image(
        image=image,
        bboxes=[(1.0, 1.0, 6.0, 5.0), (9.0, 3.0, 14.0, 10.0)],
        decisions=["manual_review", "auto_accept"],
    )
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(rendered_1, rendered_2)
    assert rendered_1.shape == (12, 16, 3) and rendered_1.dtype == np.uint8
    np.testing.assert_array_equal(rendered_1[1, 1], CVAT_MANUAL_BBOX_COLOR_BGR)
    np.testing.assert_array_equal(rendered_1[3, 9], CVAT_OTHER_BBOX_COLOR_BGR)
    np.testing.assert_array_equal(rendered_1[7, 7], np.repeat(image[7, 7], 3))
    print("[OK] deterministic CVAT review images contain strict BBox boundaries")


def test_full_video_crop_metrics_and_rendering() -> None:
    shape = (100, 110)
    fully_visible = _bbox_crop_metrics(
        projected_xyxy=(10.0, 20.0, 50.0, 60.0), local_shape_hw=shape
    )
    partially_clipped = _bbox_crop_metrics(
        projected_xyxy=(-10.0, 20.0, 30.0, 60.0), local_shape_hw=shape
    )
    fully_outside = _bbox_crop_metrics(
        projected_xyxy=(120.0, 20.0, 140.0, 60.0), local_shape_hw=shape
    )
    right_boundary = _bbox_crop_metrics(
        projected_xyxy=(80.0, 20.0, 110.0, 60.0), local_shape_hw=shape
    )

    assert fully_visible["crop_status"] == "fully_visible"
    assert fully_visible["fully_visible"] is True
    assert fully_visible["visible_fraction"] == 1.0
    assert partially_clipped["crop_status"] == "partially_clipped"
    assert partially_clipped["partially_clipped"] is True
    np.testing.assert_allclose(partially_clipped["visible_fraction"], 0.75)
    assert partially_clipped["touches_left"] is True
    assert fully_outside["crop_status"] == "fully_outside_crop"
    assert fully_outside["fully_outside_crop"] is True
    assert fully_outside["visible_fraction"] == 0.0
    assert right_boundary["crop_status"] == "fully_visible"
    assert right_boundary["fully_visible"] is True
    assert right_boundary["touches_right"] is True
    assert right_boundary["visible_fraction"] == 1.0

    image = np.full(shape, 40, dtype=np.uint8)
    rendered_1 = _render_full_video_review_image(
        image=image,
        bboxes=[(10, 20, 50, 60), (-10, 20, 30, 60), (120, 20, 140, 60)],
        crop_metrics=[fully_visible, partially_clipped, fully_outside],
        review_required=[False, True, True],
        frame_order=3,
        frame_index=19,
        has_review_target=True,
    )
    rendered_2 = _render_full_video_review_image(
        image=image,
        bboxes=[(10, 20, 50, 60), (-10, 20, 30, 60), (120, 20, 140, 60)],
        crop_metrics=[fully_visible, partially_clipped, fully_outside],
        review_required=[False, True, True],
        frame_order=3,
        frame_index=19,
        has_review_target=True,
    )
    np.testing.assert_array_equal(rendered_1, rendered_2)
    np.testing.assert_array_equal(rendered_1[50, 10], CVAT_OTHER_BBOX_COLOR_BGR)
    np.testing.assert_array_equal(rendered_1[50, 0], CVAT_CLIPPED_BBOX_COLOR_BGR)
    np.testing.assert_array_equal(rendered_1[40, -1], CVAT_OUTSIDE_BBOX_COLOR_BGR)
    # Only source grayscale and the fixed BBox/marker colors may be introduced.
    # Anti-aliased text would add several other colors and fail this contract.
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    changed = np.any(rendered_1 != base, axis=2)
    changed_colors = {
        tuple(int(value) for value in pixel)
        for pixel in rendered_1[changed]
    }
    assert changed_colors <= {
        CVAT_OTHER_BBOX_COLOR_BGR,
        CVAT_MANUAL_BBOX_COLOR_BGR,
        CVAT_CLIPPED_BBOX_COLOR_BGR,
        CVAT_OUTSIDE_BBOX_COLOR_BGR,
    }
    # The outside marker is a local tick, not a full-height red edge.
    red_edge_rows = np.flatnonzero(
        np.all(rendered_1[:, -1] == CVAT_OUTSIDE_BBOX_COLOR_BGR, axis=1)
    )
    assert 0 < red_edge_rows.size < shape[0] // 2
    np.testing.assert_array_equal(rendered_1[12, 4], base[12, 4])
    assert rendered_1.shape == (100, 110, 3)

    for invalid in (
        (10.0, 10.0, 10.0, 20.0),
        (10.0, 10.0, np.nan, 20.0),
    ):
        try:
            _bbox_crop_metrics(projected_xyxy=invalid, local_shape_hw=shape)
        except ManualReviewError:
            pass
        else:
            raise AssertionError(f"Invalid projected BBox was accepted: {invalid}")
    print("[OK] full-video text-free BBox/tick rendering, crop states, and validation")


def test_context_only_bbox_contract(root: Path) -> None:
    review_root = root / "context_only_contract"
    review_root.mkdir(parents=True)
    frame_rows = [
        {
            "frame_stem": "actionable",
            "height": "10",
            "width": "10",
            "context_only": "False",
        },
        {
            "frame_stem": "context",
            "height": "10",
            "width": "10",
            "context_only": "True",
        },
    ]
    bbox_rows = [
        {
            "frame_stem": "actionable",
            "bbox_local_xyxy": "1|1|6|6",
            "manual_review_required": "True",
            "phase3_decision": "manual_review",
        },
        {
            "frame_stem": "context",
            "bbox_local_xyxy": "100|100|110|110",
            "manual_review_required": "True",
            "phase3_decision": "manual_review",
        },
    ]
    allowlist = review_root / "context_only_stems.txt"
    allowlist.write_text("context\n", encoding="utf-8")
    assert _resolve_context_only_stems(
        review_root=review_root,
        frame_rows=frame_rows,
        bbox_rows=bbox_rows,
    ) == ["context"]
    allowlist.write_text("actionable\ncontext\n", encoding="utf-8")
    try:
        _resolve_context_only_stems(
            review_root=review_root,
            frame_rows=frame_rows,
            bbox_rows=bbox_rows,
        )
    except ManualReviewError as exc:
        assert "differs from review_frames.csv" in str(exc)
    else:
        raise AssertionError("Modern context-only CSV/file contracts must agree")

    legacy_rows = [
        {key: value for key, value in row.items() if key != "context_only"}
        for row in frame_rows
    ]
    allowlist.write_text("actionable\n", encoding="utf-8")
    assert _resolve_context_only_stems(
        review_root=review_root,
        frame_rows=legacy_rows,
        bbox_rows=bbox_rows,
    ) == ["actionable"]
    print("[OK] modern context contract and explicit legacy context override")


def test_end_to_end_roundtrip(root: Path) -> None:
    fixture = root / "integration"
    fixture.mkdir()
    paths = _make_integration_fixture(fixture)
    phase1 = fixture / "phase1"
    run_audit(phase1_args(paths, phase1))
    teacher = load_audit_teacher_config(paths["teacher"])
    refine = _write_config(
        fixture / "refine.yaml",
        teacher.fingerprint,
        production_thresholds_fixed=True,
    )
    phase3 = fixture / "phase3"
    phase3_result = run_prototype(
        _phase3_args(
            paths,
            phase1,
            refine,
            phase3,
            candidate_generation_only=False,
        )
    )
    assert phase3_result["failures"] == []
    _force_one_manual_review(phase3)
    manual_config = REPO_ROOT / "pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml"
    review_1 = fixture / "review_1"
    review_2 = fixture / "review_2"
    source_hash = file_sha256(paths["annotated"])
    result_1 = run_review_export(
        _export_args(paths, phase1, phase3, refine, review_1, manual_config)
    )
    result_2 = run_review_export(
        _export_args(paths, phase1, phase3, refine, review_2, manual_config)
    )
    assert result_1["summary"]["review_frames"] == 1
    assert result_1["summary"]["manual_review_bboxes"] == 1
    assert result_1["summary"]["export_mode"] == "selected_frames"
    assert not (review_1 / "crop_review_decisions.csv").exists()
    assert len(result_1["bbox_rows"]) == 2
    assert file_sha256(paths["annotated"]) == source_hash
    for relative in (
        "images",
        "masks",
        "overlays",
        "cvat/annotations_segmentation_mask_1_1.zip",
        "review_frames.csv",
        "review_bboxes.csv",
    ):
        first = review_1 / relative
        second = review_2 / relative
        if first.is_dir():
            files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
            assert files == sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
            for item in files:
                assert (first / item).read_bytes() == (second / item).read_bytes()
        else:
            assert first.read_bytes() == second.read_bytes()
    print("[OK] deterministic review frame export and exact CVAT ZIP")

    video_name = paths["annotated"].parent.name
    refined_auto_root = fixture / "refined_auto"
    refined_auto_h5 = (
        refined_auto_root
        / video_name
        / f"{video_name}_pointcloud_annotated_bboxrank_v3_refined_auto_v1.h5"
    )
    refined_auto_h5.parent.mkdir(parents=True)
    shutil.copy2(paths["annotated"], refined_auto_h5)
    all_args = _export_args(
        paths,
        phase1,
        phase3,
        refine,
        fixture / "review_all_phase3",
        manual_config,
    )
    all_args.review_scope = "all_phase3"
    all_args.partition_by_video = True
    all_args.source_annotated_root = refined_auto_root
    all_args.source_annotated_glob_template = (
        "{video_name}/*bboxrank_v3_refined_auto_v1.h5"
    )
    all_result = run_review_export(all_args)
    assert all_result["summary"]["manual_review_bboxes"] == 2
    assert all_result["summary"]["review_scope"] == "all_phase3"
    partition = all_result["partition_result"]
    assert partition is not None
    assert partition["video_packages"] == 1
    assert partition["review_bboxes"] == 2
    review_index = read_csv_rows(all_args.output_root / "review_index.csv")
    assert len(review_index) == 2
    assert {row["phase3_decision"] for row in review_index} == {
        "auto_accept",
        "manual_review",
    }
    video_root = all_args.output_root / "videos" / video_name
    video_summary = json.loads(
        (video_root / "export_summary.json").read_text(encoding="utf-8")
    )
    assert video_summary["manual_review_bboxes"] == 2
    assert video_summary["phase3_decision_counts"] == {
        "auto_accept": 1,
        "manual_review": 1,
    }
    case_dirs = sorted(path for path in (video_root / "cases").iterdir() if path.is_dir())
    assert len(case_dirs) == video_summary["review_frames"]
    assert all((path / "case_metadata.yaml").is_file() for path in case_dirs)
    assert {
        Path(row["source_annotated_h5"]).resolve()
        for row in read_csv_rows(video_root / "review_frames.csv")
    } == {refined_auto_h5.resolve()}

    full_video_root = fixture / "review_full_video"
    full_args = _export_args(
        paths,
        phase1,
        phase3,
        refine,
        full_video_root,
        manual_config,
    )
    full_args.review_scope = "all_phase3"
    full_args.partition_by_video = True
    full_args.include_all_video_frames = True
    full_args.expected_selected_videos = 1
    full_args.expected_review_bboxes = 2
    full_args.expected_auto_accept = 1
    full_args.expected_auto_refine = 0
    full_args.expected_manual_review = 1
    full_args.source_annotated_root = refined_auto_root
    full_args.source_annotated_glob_template = (
        "{video_name}/*bboxrank_v3_refined_auto_v1.h5"
    )
    full_source_hashes = {
        "pseudo3d": file_sha256(paths["pseudo"]),
        "annotated_v2": file_sha256(paths["annotated"]),
        "annotated_v3": file_sha256(refined_auto_h5),
    }
    full_result = run_review_export(full_args)
    full_summary = full_result["summary"]
    assert full_summary["export_mode"] == "full_video"
    assert full_summary["selected_videos"] == 1
    assert full_summary["review_frames"] == 2
    assert full_summary["target_review_frames"] == 1
    assert full_summary["selected_review_bboxes"] == 2
    assert full_summary["crop_review_decision_rows"] == 2
    assert full_summary["crop_status_counts"] == {"fully_visible": 2}
    assert full_source_hashes == {
        "pseudo3d": file_sha256(paths["pseudo"]),
        "annotated_v2": file_sha256(paths["annotated"]),
        "annotated_v3": file_sha256(refined_auto_h5),
    }
    full_frame_rows = read_csv_rows(full_video_root / "review_frames.csv")
    target_stem, context_stem = [row["frame_stem"] for row in full_frame_rows]

    # Simulate the first full-video package, where text was burned into the
    # images, then rebuild only its review pixels from the immutable source H5.
    source_target_image = full_video_root / "images" / f"{target_stem}.png"
    source_video_image = (
        full_video_root / "videos" / video_name / "images" / f"{target_stem}.png"
    )
    for path in (source_target_image, source_video_image):
        legacy = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert legacy is not None
        cv2.putText(
            legacy,
            "LEGACY LABEL",
            (2, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        assert cv2.imwrite(str(path), legacy)
    source_mask_hashes = {
        path.relative_to(full_video_root).as_posix(): file_sha256(path)
        for path in (full_video_root / "videos" / video_name / "masks").glob("*.png")
    }
    source_zip_hash = file_sha256(
        full_video_root
        / "videos"
        / video_name
        / "cvat"
        / "annotations_segmentation_mask_1_1.zip"
    )
    textfree_root = fixture / "review_full_video_v3_textfree"
    migration = run_textfree_migration(
        argparse.Namespace(
            source_review_root=full_video_root,
            output_root=textfree_root,
            expected_video_packages=1,
            expected_review_bboxes=2,
            overwrite=False,
        )
    )
    assert migration["status"] == "ok"
    assert migration["review_image_render_version"] == RENDER_VERSION
    assert migration["changed_images"] >= 1
    assert source_mask_hashes == {
        path.relative_to(textfree_root).as_posix(): file_sha256(path)
        for path in (textfree_root / "videos" / video_name / "masks").glob("*.png")
    }
    assert source_zip_hash == file_sha256(
        textfree_root
        / "videos"
        / video_name
        / "cvat"
        / "annotations_segmentation_mask_1_1.zip"
    )
    migrated_frames = read_csv_rows(
        textfree_root / "videos" / video_name / "review_frames.csv"
    )
    assert [row["frame_stem"] for row in migrated_frames] == [
        row["frame_stem"]
        for row in read_csv_rows(full_video_root / "videos" / video_name / "review_frames.csv")
    ]
    assert {row["review_image_render_version"] for row in migrated_frames} == {
        RENDER_VERSION
    }
    assert {row["review_image_has_text_overlay"] for row in migrated_frames} == {
        "False"
    }
    assert (textfree_root / "mac_transfer_manifest.csv").is_file()
    assert (textfree_root / "package_validation_summary.json").is_file()
    print("[OK] text-free image-only migration preserves masks, stems, shapes, and ZIPs")

    assert [int(row["frame_order"]) for row in full_frame_rows] == [0, 1]
    assert [int(row["frame_index"]) for row in full_frame_rows] == [10, 30]
    assert [row["frame_role"] for row in full_frame_rows] == [
        "review_target",
        "context_only",
    ]
    assert [row["review_target"] for row in full_frame_rows] == ["True", "False"]
    assert [int(row["bbox_count"]) for row in full_frame_rows] == [2, 0]
    assert _read_binary(full_video_root / "masks" / f"{target_stem}.png").any()
    assert not _read_binary(full_video_root / "masks" / f"{context_stem}.png").any()
    assert (full_video_root / "context_only_stems.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [context_stem]

    full_bbox_rows = read_csv_rows(full_video_root / "review_bboxes.csv")
    assert len(full_bbox_rows) == 2
    assert all(row["review_target"] == "True" for row in full_bbox_rows)
    assert all(row["crop_status"] == "fully_visible" for row in full_bbox_rows)
    crop_decisions = read_csv_rows(
        full_video_root / "crop_review_decisions.csv"
    )
    assert len(crop_decisions) == 2
    assert {row["review_disposition"] for row in crop_decisions} == {"pending"}
    assert {row["bbox_key"] for row in crop_decisions} == {
        row["bbox_key"] for row in full_bbox_rows
    }

    full_partition = full_result["partition_result"]
    assert full_partition is not None
    assert full_partition["video_packages"] == 1
    assert full_partition["exported_frames"] == 2
    assert full_partition["case_frames"] == 1
    assert full_partition["review_bboxes"] == 2
    assert full_partition["progress_csv"] == "video_progress.csv"
    full_video_package = full_video_root / "videos" / video_name
    assert len(list((full_video_package / "images").glob("*.png"))) == 2
    assert len(list((full_video_package / "masks").glob("*.png"))) == 2
    assert len(list((full_video_package / "cases").iterdir())) == 1
    assert len(read_csv_rows(full_video_package / "crop_review_decisions.csv")) == 2
    assert len(read_csv_rows(full_video_root / "video_progress.csv")) == 1
    print(
        "[OK] full-video target/context masks, ordering, multiple BBoxes, "
        "non-contiguous indices, and package contract"
    )

    edited_partition = _make_edited_export(
        video_root, fixture / "edited_partition.zip"
    )
    partition_session = _write_session(
        video_root, edited_partition, fixture / "partition_session.yaml"
    )
    partition_import = run_manual_import(
        argparse.Namespace(
            review_root=video_root,
            cvat_export_zip=edited_partition,
            review_session=partition_session,
            source_annotated_root=refined_auto_root,
            manual_review_config=manual_config,
            output_root=fixture / "partition_manual_output",
            summary_csv=None,
            overwrite=False,
        )
    )
    assert partition_import["summary"]["reviewed_bboxes"] == 2
    print("[OK] all-Phase3 per-video packages, case folders, v3 source, and import contract")

    phase3_empty = fixture / "phase3_no_manual"
    shutil.copytree(phase3, phase3_empty)
    empty_decisions = read_csv_rows(phase3_empty / "bbox_decisions.csv")
    for row in empty_decisions:
        row["proposed_decision"] = "auto_accept"
        row["reason_codes"] = "synthetic_auto_accept"
    phase3_write_csv(phase3_empty / "bbox_decisions.csv", empty_decisions)
    empty_summary_path = phase3_empty / "refine_summary.json"
    empty_summary = json.loads(empty_summary_path.read_text(encoding="utf-8"))
    empty_summary["decision_counts"] = {"auto_accept": len(empty_decisions)}
    empty_summary_path.write_text(
        json.dumps(empty_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase3_write_csv(
        phase3_empty / "checksums.csv", phase3_checksum_rows(phase3_empty)
    )
    no_review_root = fixture / "review_none"
    no_review = run_review_export(
        _export_args(paths, phase1, phase3_empty, refine, no_review_root, manual_config)
    )
    assert no_review["summary"]["status"] == "no_manual_review"
    assert not (no_review_root / "cvat").exists()
    print("[OK] zero manual-review result does not create an empty CVAT ZIP")

    edited_zip = _make_edited_export(review_1, fixture / "edited.zip")
    session = _write_session(review_1, edited_zip, fixture / "review_session.yaml")
    output_root = fixture / "manual_output"
    import_result = run_manual_import(
        argparse.Namespace(
            review_root=review_1,
            cvat_export_zip=edited_zip,
            review_session=session,
            source_annotated_root=paths["annotated_root"],
            manual_review_config=manual_config,
            output_root=output_root,
            summary_csv=None,
            overwrite=False,
        )
    )
    assert import_result["summary"]["videos"] == 1
    output_h5 = Path(import_result["results"][0]["output_h5"])
    assert output_h5.is_file() and file_sha256(paths["annotated"]) == source_hash
    before = _snapshot_non_annotation(paths["annotated"])
    after = _snapshot_non_annotation(output_h5)
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_array_equal(before[key][0], after[key][0])
        assert before[key][1] == after[key][1]
    with h5py.File(output_h5, "r") as handle:
        labels = handle["annotation/point_label"][:]
        valid = handle["annotation/valid_mask"][:]
        np.testing.assert_array_equal(valid, labels != -1)
        assert set(int(value) for value in np.unique(labels)) <= {-1, 0, 1}
        assert handle.attrs["manual_annotation_provenance"] == (
            "manual_cvat_segmentation_mask_1_1_v1"
        )
        assert int(handle.attrs["manual_reviewed_frame_count"]) == 1
        assert int(handle.attrs["manual_reviewed_bbox_count"]) == 1
        assert "manual_review" in handle
    second_output_root = fixture / "manual_output_repeat"
    repeated = run_manual_import(
        argparse.Namespace(
            review_root=review_1,
            cvat_export_zip=edited_zip,
            review_session=session,
            source_annotated_root=paths["annotated_root"],
            manual_review_config=manual_config,
            output_root=second_output_root,
            summary_csv=None,
            overwrite=False,
        )
    )
    repeated_h5 = Path(repeated["results"][0]["output_h5"])
    with h5py.File(output_h5, "r") as first, h5py.File(repeated_h5, "r") as second:
        for dataset in (
            "annotation/point_label",
            "annotation/valid_mask",
            "frame_annotation/annotation_reason",
            "frame_annotation/selected_contour_source",
            "manual_review/frame_stem",
            "manual_review/positive_points",
        ):
            np.testing.assert_array_equal(first[dataset][()], second[dataset][()])
        for attr in (
            "manual_annotation_provenance",
            "manual_review_package_sha256",
            "cvat_export_sha256",
            "manual_reviewed_frame_count",
            "manual_reviewed_bbox_count",
        ):
            assert first.attrs[attr] == second.attrs[attr]
    print("[OK] edited CVAT ZIP import, H5 schema preservation, labels, and provenance")
    print("[OK] deterministic repeated import annotation arrays and metadata")


def test_config_contract() -> None:
    config = load_manual_review_config(
        REPO_ROOT / "pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml"
    )
    assert config.label_name == "femur"
    assert config.no_bbox_label == 0
    assert config.bbox_inside_non_contour_label == "ignore"
    assert config.require_nonempty_manual_bbox is True
    print("[OK] project Phase 4 config and fixed label contract")


def main() -> None:
    test_pure_label_contract()
    test_cvat_review_image_bbox_overlay()
    test_full_video_crop_metrics_and_rendering()
    with tempfile.TemporaryDirectory(prefix="stage4_cvat_manual_roundtrip_") as temporary:
        root = Path(temporary)
        test_context_only_bbox_contract(root)
        test_end_to_end_roundtrip(root)
    test_config_contract()
    print("Stage 4 CVAT manual round-trip synthetic checks passed.")


if __name__ == "__main__":
    main()
