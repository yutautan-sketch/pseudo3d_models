from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import h5py
import numpy as np

from checks.stage4.check_stage4_contour_auto_refine import _phase3_args, _write_config
from checks.stage4.check_stage4_bbox_ranked_label_policy import audit_file
from checks.stage4.check_stage4_contour_teacher_audit import (
    _args as phase1_args,
    _make_integration_fixture,
)
from pseudo3d.analysis.audit_stage4_contour_teacher import (
    load_audit_teacher_config,
    run_audit,
)
from pseudo3d.analysis.prototype_stage4_contour_auto_refine import run_prototype
from pseudo3d.annotation.stage4_manual_review import (
    LABEL_BACKGROUND,
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
    bbox_union_mask,
    file_sha256,
    read_csv_rows,
)
from pseudo3d.batch.annotation.batch_import_stage4_phase5_fullvideo_cvat import (
    AUTHORITATIVE_LABEL_AUTHORITY,
    AUTHORITATIVE_OUTPUT_TOKEN,
    CVAT_MASK_EMPTY,
    CVAT_MASK_OMITTED_AS_EMPTY,
    CVAT_MASK_POSITIVE,
    FullVideoImportError,
    _apply_selected_frame,
    apply_authoritative_frame_mask,
    read_authoritative_task_frame_masks,
    run,
)
from pseudo3d.batch.export.batch_export_stage4_manual_review_cvat import run_review_export
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    convert_masks_to_cvat_zip,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _export_args(
    paths: dict[str, Path],
    phase1: Path,
    phase3: Path,
    refine: Path,
    output: Path,
    manual_config: Path,
    refined_root: Path,
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
        review_scope="all_phase3",
        partition_by_video=True,
        source_annotated_root=refined_root,
        source_annotated_glob_template="{video_name}/*bboxrank_v3_refined_auto_v1.h5",
        include_all_video_frames=True,
        expected_selected_videos=1,
        expected_review_bboxes=2,
        expected_auto_accept=0,
        expected_auto_refine=0,
        expected_manual_review=2,
    )


def _snapshot(
    root: Path,
    *,
    video: str,
    task_id: int,
    task_name: str,
    annotation: Path,
    review_bboxes: int,
) -> Path:
    snapshot = root / "snapshot"
    annotation_out = (
        snapshot
        / "annotations"
        / video
        / f"{video}__task{task_id}__reviewed_segmentation_mask_1_1.zip"
    )
    backup_out = (
        snapshot
        / "task_backups"
        / video
        / f"{video}__task{task_id}__post_review_backup.zip"
    )
    annotation_out.parent.mkdir(parents=True)
    backup_out.parent.mkdir(parents=True)
    shutil.copy2(annotation, annotation_out)
    with zipfile.ZipFile(backup_out, "w") as archive:
        archive.writestr("task.json", "{}\n")
    row = {
        "video_name": video,
        "task_id": task_id,
        "task_name": task_name,
        "status": "complete",
        "image_count": 2,
        "review_bboxes": review_bboxes,
        "annotation_zip": "/untrusted/mac/path/annotation.zip",
        "annotation_bytes": annotation_out.stat().st_size,
        "annotation_sha256": file_sha256(annotation_out),
        "backup_zip": "/untrusted/mac/path/backup.zip",
        "backup_bytes": backup_out.stat().st_size,
        "backup_sha256": file_sha256(backup_out),
    }
    _write_csv(snapshot / "export_manifest.csv", [row])
    (snapshot / "export_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "registered_tasks": 1,
                "completed_tasks": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot


def _copy_zip_without_mask_pair(source: Path, destination: Path, stem: str) -> None:
    removed = 0
    forbidden = {
        f"SegmentationClass/{stem}.png",
        f"SegmentationClass/images/{stem}.png",
        f"SegmentationObject/{stem}.png",
        f"SegmentationObject/images/{stem}.png",
    }
    with zipfile.ZipFile(source, "r") as input_zip, zipfile.ZipFile(
        destination, "w"
    ) as output_zip:
        for info in input_zip.infolist():
            if info.filename in forbidden:
                removed += 1
                continue
            output_zip.writestr(info, input_zip.read(info.filename))
    assert removed == 2


def _reference_authoritative_frame_labels(
    *,
    labels: np.ndarray,
    frame_orders: np.ndarray,
    pixel_xy: np.ndarray,
    frame_order: int,
    corrected_mask: np.ndarray,
    frame_bboxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Return the fixed Step 1 oracle without consulting source-frame labels."""
    result = np.asarray(labels, dtype=np.int8).copy()
    selector = np.asarray(frame_orders, dtype=np.int64) == int(frame_order)
    xy = np.rint(np.asarray(pixel_xy)[selector]).astype(np.int64)
    mask = np.asarray(corrected_mask, dtype=bool)
    height, width = mask.shape
    if xy.size and (
        np.any(xy[:, 0] < 0)
        or np.any(xy[:, 0] >= width)
        or np.any(xy[:, 1] < 0)
        or np.any(xy[:, 1] >= height)
    ):
        raise AssertionError("Synthetic authoritative point lies outside mask")

    frame_labels = np.full(int(selector.sum()), LABEL_BACKGROUND, dtype=np.int8)
    bbox_union = bbox_union_mask(mask.shape, frame_bboxes)
    if xy.size:
        frame_labels[bbox_union[xy[:, 1], xy[:, 0]]] = LABEL_IGNORE
        frame_labels[mask[xy[:, 1], xy[:, 0]]] = LABEL_FEMUR_CANDIDATE
    result[selector] = frame_labels
    return result


def test_snapshot_authoritative_label_oracle() -> None:
    orders = np.asarray([0, 0, 0, 0, 0, 1], dtype=np.int32)
    xy = np.asarray(
        [[0, 0], [1, 1], [2, 2], [4, 1], [5, 5], [3, 3]],
        dtype=np.float32,
    )
    bboxes = [(1.0, 1.0, 3.0, 3.0), (4.0, 0.0, 5.0, 3.0)]
    corrected = np.zeros((6, 6), dtype=np.uint8)
    corrected[0, 0] = 1  # Human positive outside every BBox remains positive.
    corrected[2, 2] = 1

    source_variants = (
        np.ones(6, dtype=np.int8),
        np.asarray([-1, 1, 0, 1, 1, -1], dtype=np.int8),
    )
    expected = (
        np.asarray([1, -1, 1, -1, 0, 1], dtype=np.int8),
        np.asarray([1, -1, 1, -1, 0, -1], dtype=np.int8),
    )
    for source, wanted in zip(source_variants, expected, strict=True):
        actual, stats = apply_authoritative_frame_mask(
            labels=source,
            frame_orders=orders,
            pixel_xy=xy,
            frame_order=0,
            corrected_mask=corrected,
            frame_bboxes=bboxes,
        )
        np.testing.assert_array_equal(actual, wanted)
        assert stats["positive_outside_cvat_mask_points"] == 0
        selected = orders == 0
        rounded = np.rint(xy[selected]).astype(np.int64)
        positives = actual[selected] == LABEL_FEMUR_CANDIDATE
        assert np.all(corrected[rounded[positives, 1], rounded[positives, 0]])

    # An empty or CVAT-omitted-as-empty mask never falls back to source positives.
    empty = np.zeros_like(corrected)
    emptied, empty_stats = apply_authoritative_frame_mask(
        labels=np.ones(6, dtype=np.int8),
        frame_orders=orders,
        pixel_xy=xy,
        frame_order=0,
        corrected_mask=empty,
        frame_bboxes=bboxes,
    )
    np.testing.assert_array_equal(emptied, [0, -1, -1, -1, 0, 1])
    assert not np.any(emptied[orders == 0] == LABEL_FEMUR_CANDIDATE)
    assert empty_stats["positive_points"] == 0

    # A no-BBox context frame still accepts a human mask; all mask-external
    # points are background because there is no BBox-defined ignore region.
    no_bbox, no_bbox_stats = apply_authoritative_frame_mask(
        labels=np.full(6, LABEL_IGNORE, dtype=np.int8),
        frame_orders=orders,
        pixel_xy=xy,
        frame_order=0,
        corrected_mask=corrected,
        frame_bboxes=[],
    )
    np.testing.assert_array_equal(no_bbox, [1, 0, 1, 0, 0, -1])
    assert no_bbox_stats["ignore_points"] == 0
    print(
        "[OK] fixed authoritative target/context, empty, no-BBox, and "
        "multiple-BBox label oracle"
    )


def test_authoritative_task_frame_reader(
    *,
    package: Path,
    frame_rows: list[dict[str, str]],
    corrected_zip: Path,
    target: dict[str, str],
    context: dict[str, str],
    root: Path,
) -> None:
    loaded = read_authoritative_task_frame_masks(
        package_root=package,
        frame_rows=frame_rows,
        annotation_zip=corrected_zip,
    )
    assert [row["frame_stem"] for row in loaded["frames"]] == [
        row["frame_stem"] for row in frame_rows
    ]
    assert loaded["mask_status_counts"] == {
        CVAT_MASK_EMPTY: 1,
        CVAT_MASK_POSITIVE: 1,
    }
    target_record = loaded["frames_by_stem"][target["frame_stem"]]
    context_record = loaded["frames_by_stem"][context["frame_stem"]]
    assert target_record["cvat_mask_status"] == CVAT_MASK_EMPTY
    assert target_record["frame_role"] == "review_target"
    assert target_record["cvat_mask_positive_pixels"] == 0
    assert context_record["cvat_mask_status"] == CVAT_MASK_POSITIVE
    assert context_record["frame_role"] == "context_only"
    assert context_record["cvat_mask_positive_pixels"] == 1
    assert (
        loaded["frames_by_order"][int(context["frame_order"])]["frame_stem"]
        == context["frame_stem"]
    )

    omitted_zip = root / "reviewed_target_omitted.zip"
    _copy_zip_without_mask_pair(
        corrected_zip,
        omitted_zip,
        target["frame_stem"],
    )
    omitted = read_authoritative_task_frame_masks(
        package_root=package,
        frame_rows=frame_rows,
        annotation_zip=omitted_zip,
    )
    omitted_target = omitted["frames_by_stem"][target["frame_stem"]]
    assert omitted_target["cvat_mask_status"] == CVAT_MASK_OMITTED_AS_EMPTY
    assert omitted_target["cvat_mask_positive_pixels"] == 0
    assert not np.any(omitted_target["mask"])
    assert omitted["mask_status_counts"] == {
        CVAT_MASK_OMITTED_AS_EMPTY: 1,
        CVAT_MASK_POSITIVE: 1,
    }

    duplicate_rows = [dict(row) for row in frame_rows]
    duplicate_rows.append(dict(frame_rows[0]))
    try:
        read_authoritative_task_frame_masks(
            package_root=package,
            frame_rows=duplicate_rows,
            annotation_zip=corrected_zip,
        )
    except FullVideoImportError as exc:
        assert "duplicate frame stems" in str(exc)
    else:
        raise AssertionError("Duplicate Task-frame rows must be rejected")

    unreviewable_target_rows = [dict(row) for row in frame_rows]
    unreviewable_target_rows[0]["context_only"] = "True"
    unreviewable_target = read_authoritative_task_frame_masks(
        package_root=package,
        frame_rows=unreviewable_target_rows,
        annotation_zip=corrected_zip,
    )
    assert unreviewable_target["frames"][0]["review_target"] is True
    assert unreviewable_target["frames"][0]["context_only"] is True

    invalid_role_rows = [dict(row) for row in frame_rows]
    invalid_role_rows[0]["frame_role"] = "context_only"
    try:
        read_authoritative_task_frame_masks(
            package_root=package,
            frame_rows=invalid_role_rows,
            annotation_zip=corrected_zip,
        )
    except FullVideoImportError as exc:
        assert "Frame role flags differ" in str(exc)
    else:
        raise AssertionError("Inconsistent Task-frame role flags must be rejected")
    print(
        "[OK] all-Task-frame CVAT reader, explicit empty, omitted-as-empty, "
        "roles, and indexes"
    )


def test_single_bbox_manual_override() -> None:
    labels = np.asarray([0, 0, 0], dtype=np.int8)
    orders = np.zeros(3, dtype=np.int32)
    xy = np.asarray([[0, 0], [2, 2], [5, 5]], dtype=np.float32)
    corrected = np.zeros((6, 6), dtype=np.uint8)
    corrected[0, 0] = 1
    updated, stats = _apply_selected_frame(
        labels=labels,
        frame_orders=orders,
        pixel_xy=xy,
        frame_order=0,
        corrected_mask=corrected,
        target_bboxes=[(1.0, 1.0, 3.0, 3.0)],
        allow_outside_single_bbox=True,
    )
    np.testing.assert_array_equal(updated, [1, -1, 0])
    assert stats["outside_bbox_positive_pixels"] == 1
    try:
        _apply_selected_frame(
            labels=labels,
            frame_orders=orders,
            pixel_xy=xy,
            frame_order=0,
            corrected_mask=corrected,
            target_bboxes=[(1.0, 1.0, 3.0, 3.0)],
            allow_outside_single_bbox=False,
        )
    except FullVideoImportError:
        pass
    else:
        raise AssertionError("Strict mode must reject a BBox-external mask")
    print("[OK] authoritative single-BBox manual mask override")


def test_end_to_end_final_import(root: Path) -> None:
    fixture = root / "fixture"
    fixture.mkdir()
    paths = _make_integration_fixture(fixture)
    video = paths["annotated"].parent.name
    phase1 = fixture / "phase1"
    run_audit(phase1_args(paths, phase1))
    teacher = load_audit_teacher_config(paths["teacher"])
    refine = _write_config(
        fixture / "refine.yaml",
        teacher.fingerprint,
        production_thresholds_fixed=True,
    )
    phase3 = fixture / "phase3"
    run_prototype(
        _phase3_args(
            paths,
            phase1,
            refine,
            phase3,
            candidate_generation_only=False,
        )
    )
    # The final importer only needs a stable all-Phase3 selection. The fixture's
    # actual proposed decisions do not affect the selected BBox contract.
    decisions = read_csv_rows(phase3 / "bbox_decisions.csv")
    for row in decisions:
        row["proposed_decision"] = "manual_review"
        row["reason_codes"] = "synthetic_final_import"
    _write_csv(phase3 / "bbox_decisions.csv", decisions)
    summary_path = phase3 / "refine_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision_counts"] = {"manual_review": 2}
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    # Recreate the Phase 3 checksums after the deterministic fixture edit.
    from pseudo3d.analysis.prototype_stage4_contour_auto_refine import (
        _checksum_rows,
        _write_csv as phase3_write_csv,
    )

    phase3_write_csv(phase3 / "checksums.csv", _checksum_rows(phase3))

    refined_root = fixture / "refined"
    refined_h5 = (
        refined_root
        / video
        / f"{video}_pointcloud_annotated_bboxrank_v3_refined_auto_v1.h5"
    )
    refined_h5.parent.mkdir(parents=True)
    shutil.copy2(paths["annotated"], refined_h5)
    manual_config = REPO_ROOT / "pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml"
    review_root = fixture / "review"
    exported = run_review_export(
        _export_args(
            paths,
            phase1,
            phase3,
            refine,
            review_root,
            manual_config,
            refined_root,
        )
    )
    assert exported["summary"]["selected_review_bboxes"] == 2
    package = review_root / "videos" / video
    frame_rows = read_csv_rows(package / "review_frames.csv")
    target = next(row for row in frame_rows if row["review_target"] == "True")
    context = next(row for row in frame_rows if row["review_target"] == "False")

    edited_masks = fixture / "edited_masks"
    edited_masks.mkdir()
    for frame in frame_rows:
        source = package / frame["mask_path"]
        mask = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        assert mask is not None
        mask[...] = 0
        if frame["frame_stem"] == context["frame_stem"]:
            # This non-empty context edit reproduces the v4 scope defect. Step
            # 3 will make it authoritative instead of inheriting source labels.
            mask[0, 0] = 255
        assert cv2.imwrite(str(edited_masks / f"{frame['frame_stem']}.png"), mask)
    corrected_zip = fixture / "reviewed.zip"
    convert_masks_to_cvat_zip(
        images_dir=package / "images",
        masks_dir=edited_masks,
        output_zip=corrected_zip,
    )
    test_authoritative_task_frame_reader(
        package=package,
        frame_rows=frame_rows,
        corrected_zip=corrected_zip,
        target=target,
        context=context,
        root=fixture,
    )
    task_name = f"stage4_phase5_fullvideo_v3_textfree__{video}"
    snapshot = _snapshot(
        fixture,
        video=video,
        task_id=7,
        task_name=task_name,
        annotation=corrected_zip,
        review_bboxes=2,
    )
    returned_task_map = fixture / "returned" / "task_management" / "task_map.csv"
    _write_csv(
        returned_task_map,
        [
            {
                "video_name": video,
                "task_id": 7,
                "task_name": task_name,
                "status": "backup_verified",
                "image_count": 2,
                "review_bboxes": 2,
            }
        ],
    )
    excluded = "synthetic_excluded"
    manifest = fixture / "manifest_final.csv"
    _write_csv(
        manifest,
        [
            {
                "video_name": video,
                "pseudo3d_h5": str(paths["pseudo"]),
                "voc_xml_root": str(paths["xml"].parents[2]),
                "split": "train",
                "enabled": "true",
                "notes": "",
            },
            {
                "video_name": excluded,
                "pseudo3d_h5": str(fixture / "unused.h5"),
                "voc_xml_root": str(paths["xml"].parents[2]),
                "split": "train",
                "enabled": "false",
                "notes": "excluded fixture",
            },
        ],
    )
    exclusion_manifest = fixture / "exclusions.csv"
    _write_csv(
        exclusion_manifest,
        [
            {
                "video_name": excluded,
                "reason_code": "synthetic_exclusion",
                "evidence_frames": "0",
                "scope": "stage4_teacher",
                "recoverable": "true",
                "notes": "synthetic exclusion contract",
            }
        ],
    )
    output = fixture / "output"
    args = argparse.Namespace(
        manifest=manifest,
        exclusion_manifest=exclusion_manifest,
        source_annotated_root=refined_root,
        review_root=review_root,
        snapshot_root=snapshot,
        task_map_csv=returned_task_map,
        manual_review_config=manual_config,
        output_root=output,
        expected_videos=1,
        expected_reviewed_videos=1,
        expected_actionable_bboxes=2,
        expected_bbox_override_frames=0,
        expected_bbox_override_pixels=0,
        preflight_only=False,
        skip_existing=False,
        overwrite=False,
        output_teacher_token=AUTHORITATIVE_OUTPUT_TOKEN,
    )
    with h5py.File(refined_h5, "r") as handle:
        before_labels = handle["annotation/point_label"][:]
        orders = handle["point_cloud/frame_order"][:]
        pixel_xy = handle["point_cloud/pixel_xy"][:]
    result = run(args)
    assert result["summary"]["videos"] == 1
    assert result["summary"]["reviewed_videos"] == 1
    assert result["summary"]["applied_bboxes"] == 2
    assert result["summary"]["empty_bboxes"] == 2
    assert result["summary"]["authoritative_frames"] == 2
    assert result["summary"]["positive_outside_cvat_mask_points"] == 0
    output_h5 = next((output / video).glob(f"*{AUTHORITATIVE_OUTPUT_TOKEN}.h5"))
    with h5py.File(output_h5, "r") as handle:
        after_labels = handle["annotation/point_label"][:]
        after_valid = handle["annotation/valid_mask"][:]
        assert handle.attrs["contour_teacher_schema"] == AUTHORITATIVE_OUTPUT_TOKEN
        assert int(handle.attrs["manual_review_fullvideo_applied_bbox_count"]) == 2
        assert int(handle.attrs["manual_review_fullvideo_empty_bbox_count"]) == 2
        assert int(handle.attrs["manual_review_fullvideo_authoritative_frame_count"]) == 2
        assert (
            int(
                handle.attrs[
                    "manual_review_fullvideo_positive_outside_cvat_mask_points"
                ]
            )
            == 0
        )
        assert len(handle["manual_review_fullvideo/frame_stem"]) == 2
        frame_provenance = handle["manual_review_fullvideo_frames"]
        assert len(frame_provenance["frame_stem"]) == 2
        assert frame_provenance.attrs["label_authority"] == AUTHORITATIVE_LABEL_AUTHORITY
        assert set(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in frame_provenance["cvat_mask_status"][:]
        ) == {CVAT_MASK_EMPTY, CVAT_MASK_POSITIVE}
        assert np.all(frame_provenance["positive_outside_cvat_mask_points"][:] == 0)
    context_order = int(context["frame_order"])
    target_order = int(target["frame_order"])
    context_mask = cv2.imread(
        str(edited_masks / f"{context['frame_stem']}.png"),
        cv2.IMREAD_UNCHANGED,
    )
    assert context_mask is not None and int(np.sum(context_mask > 0)) == 1
    expected_context = _reference_authoritative_frame_labels(
        labels=before_labels,
        frame_orders=orders,
        pixel_xy=pixel_xy,
        frame_order=context_order,
        corrected_mask=context_mask,
        frame_bboxes=[],
    )
    context_selector = orders == context_order
    np.testing.assert_array_equal(
        after_labels[context_selector], expected_context[context_selector]
    )
    context_xy = np.rint(pixel_xy[context_selector]).astype(np.int64)
    expected_positive = context_mask[context_xy[:, 1], context_xy[:, 0]] > 0
    assert int(np.sum(expected_positive)) == 1
    assert int(np.sum(after_labels[context_selector][expected_positive] == 1)) == 1
    assert not np.any(
        (after_labels[context_selector] == 1) & ~expected_positive
    )
    assert np.all(after_valid == (after_labels != -1))
    # An empty CVAT mask removes every target-frame automatic positive.
    assert int(np.sum(after_labels[orders == target_order] == 1)) == 0
    assert not (output / excluded).exists()
    policy_stats = audit_file(
        output_h5, allow_cvat_authoritative_no_bbox_positive=True
    )
    assert policy_stats["no_bbox_cvat_positive_points"] == 1

    args.skip_existing = True
    rerun = run(args)
    assert rerun["summary"]["status_counts"] == {"skipped_verified": 1}
    print(
        "[OK] authoritative all-frame import, context/empty handling, exclusion, "
        "frame provenance, and verified resume"
    )


def main() -> None:
    test_snapshot_authoritative_label_oracle()
    test_single_bbox_manual_override()
    with tempfile.TemporaryDirectory(prefix="stage4_phase5_fullvideo_final_") as directory:
        test_end_to_end_final_import(Path(directory))
    print(
        "Stage 4 CVAT snapshot-authoritative Step 3 importer contracts passed."
    )


if __name__ == "__main__":
    main()
