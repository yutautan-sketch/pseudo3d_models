from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.validate_stage4_phase5_fullvideo_cvat_package import (
    _write_transfer_manifest,
)
from pseudo3d.annotation.stage4_manual_review import (
    ManualReviewError,
    file_sha256,
    prepare_output_root,
    read_csv_rows,
    write_csv_rows,
    write_json,
    write_png,
)
from pseudo3d.batch.export.batch_export_stage4_manual_review_cvat import (
    _render_full_video_review_image,
    hashlib_sha256_rows,
)
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    validate_cvat_segmentation_mask_zip,
)
from src.utils.alpha_texture_processing import image_to_uint8_gray


MIGRATION_SCHEMA_VERSION = 1
RENDER_VERSION = "bbox_only_textfree_v3"
CONTROL_FILES_TO_REBUILD = {
    "mac_transfer_manifest.csv",
    "mac_transfer_summary.json",
    "package_validation_summary.json",
    "textfree_migration_summary.json",
}
GENERATED_STATE_DIRECTORIES = {
    "task_management",
    "cvat_exports_before_textfree_v1",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManualReviewError(f"JSON root must be an object: {path}")
    return value


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ManualReviewError(f"{name} must be boolean: {value!r}")


def _pipe_floats(value: Any, name: str, *, count: int = 4) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in str(value).split("|"))
    except ValueError as exc:
        raise ManualReviewError(f"Invalid {name}: {value!r}") from exc
    if len(parsed) != count or not np.all(np.isfinite(parsed)):
        raise ManualReviewError(f"Invalid {name}: {value!r}")
    return parsed


def _copy_ignore(source_root: Path):
    def ignore(directory: str, names: list[str]) -> set[str]:
        path = Path(directory).resolve()
        ignored = {name for name in names if name == ".DS_Store" or name.startswith("._")}
        if path == source_root:
            ignored.update(CONTROL_FILES_TO_REBUILD & set(names))
            ignored.update(GENERATED_STATE_DIRECTORIES & set(names))
        if "task_backups" in names:
            ignored.add("task_backups")
        return ignored

    return ignore


def _rewrite_checksum_contract(source_root: Path, destination_root: Path) -> tuple[int, str]:
    source_rows = read_csv_rows(source_root / "checksums.csv")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        relative = str(row.get("relative_path", ""))
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts or relative in seen:
            raise ManualReviewError(f"Unsafe/duplicate checksum member: {relative!r}")
        seen.add(relative)
        destination = destination_root / path
        if not destination.is_file():
            raise FileNotFoundError(destination)
        rows.append(
            {
                "relative_path": relative,
                "bytes": int(destination.stat().st_size),
                "sha256": file_sha256(destination),
            }
        )
    write_csv_rows(
        destination_root / "checksums.csv",
        rows,
        fieldnames=("relative_path", "bytes", "sha256"),
    )
    return len(rows), hashlib_sha256_rows(rows)


def _frame_bbox_inputs(
    bbox_rows: Sequence[Mapping[str, Any]], frame_stem: str
) -> tuple[list[tuple[float, ...]], list[dict[str, Any]], list[bool]]:
    selected = sorted(
        (row for row in bbox_rows if str(row["frame_stem"]) == frame_stem),
        key=lambda row: int(row["bbox_index"]),
    )
    bboxes: list[tuple[float, ...]] = []
    metrics: list[dict[str, Any]] = []
    review_required: list[bool] = []
    for expected_index, row in enumerate(selected):
        if int(row["bbox_index"]) != expected_index:
            raise ManualReviewError(
                f"Non-contiguous BBox indices for {frame_stem}: {row['bbox_index']}"
            )
        projected = _pipe_floats(row["projected_bbox_xyxy"], "projected_bbox_xyxy")
        clipped = _pipe_floats(row["clipped_bbox_xyxy"], "clipped_bbox_xyxy")
        bboxes.append(projected)
        metrics.append(
            {
                "projected_bbox_xyxy": projected,
                "clipped_bbox_xyxy": clipped,
                "visible_intersection_area": float(row["visible_intersection_area"]),
                "visible_fraction": float(row["visible_fraction"]),
                "crop_status": str(row["crop_status"]),
            }
        )
        review_required.append(_bool(row.get("review_target", False), "review_target"))
    return bboxes, metrics, review_required


def _load_source_frame(
    images: h5py.Dataset,
    frame_indices: h5py.Dataset,
    *,
    frame_order: int,
    frame_index: int,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if not 0 <= frame_order < int(images.shape[0]):
        raise ManualReviewError(f"frame_order is outside source H5: {frame_order}")
    if int(frame_indices[frame_order]) != frame_index:
        raise ManualReviewError(
            f"Source frame index mismatch at order {frame_order}: "
            f"{int(frame_indices[frame_order])} != {frame_index}"
        )
    gray = image_to_uint8_gray(images[frame_order])
    if gray.shape != expected_shape:
        raise ManualReviewError(
            f"Source frame shape mismatch at order {frame_order}: "
            f"{gray.shape} != {expected_shape}"
        )
    return gray


def _update_yaml(path: Path, values: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ManualReviewError(f"YAML root must be a mapping: {path}")
    payload.update(values)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=True)


def _render_video(
    *,
    source_root: Path,
    output_root: Path,
    video_name: str,
) -> dict[str, Any]:
    source_video = source_root / "videos" / video_name
    output_video = output_root / "videos" / video_name
    source_frames = read_csv_rows(source_video / "review_frames.csv")
    output_frames = read_csv_rows(output_video / "review_frames.csv")
    bbox_rows = read_csv_rows(source_video / "review_bboxes.csv")
    if source_frames != output_frames:
        raise ManualReviewError(f"Copied review_frames.csv changed unexpectedly: {video_name}")
    source_h5_paths = {str(row["source_pseudo3d_h5"]) for row in source_frames}
    source_h5_hashes = {str(row["source_pseudo3d_h5_sha256"]) for row in source_frames}
    if len(source_h5_paths) != 1 or len(source_h5_hashes) != 1:
        raise ManualReviewError(f"Video has multiple source pseudo3D H5 contracts: {video_name}")
    source_h5 = Path(next(iter(source_h5_paths))).resolve()
    expected_h5_sha = next(iter(source_h5_hashes))
    if not source_h5.is_file() or file_sha256(source_h5) != expected_h5_sha:
        raise ManualReviewError(f"Source pseudo3D H5 missing or changed: {video_name}")

    changed_images = 0
    masks_preserved = 0
    case_images_updated = 0
    updated_rows: list[dict[str, Any]] = []
    with h5py.File(source_h5, "r") as handle:
        if "local_encoder_images" not in handle or "frame_indices" not in handle:
            raise ManualReviewError(f"Source pseudo3D H5 lacks frame datasets: {source_h5}")
        images = handle["local_encoder_images"]
        frame_indices = handle["frame_indices"]
        if int(images.shape[0]) != len(source_frames) or int(frame_indices.shape[0]) != len(source_frames):
            raise ManualReviewError(f"Source H5/review frame count mismatch: {video_name}")
        for row in sorted(source_frames, key=lambda value: int(value["frame_order"])):
            row = dict(row)
            stem = str(row["frame_stem"])
            shape = (int(row["height"]), int(row["width"]))
            frame_order = int(row["frame_order"])
            frame_index = int(row["frame_index"])
            gray = _load_source_frame(
                images,
                frame_indices,
                frame_order=frame_order,
                frame_index=frame_index,
                expected_shape=shape,
            )
            bboxes, crop_metrics, review_required = _frame_bbox_inputs(bbox_rows, stem)
            rendered = _render_full_video_review_image(
                image=gray,
                bboxes=bboxes,
                crop_metrics=crop_metrics,
                review_required=review_required,
                frame_order=frame_order,
                frame_index=frame_index,
                has_review_target=any(review_required),
            )
            source_image = source_video / str(row["image_path"])
            output_image = output_video / str(row["image_path"])
            old_image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
            if old_image is None or old_image.shape != rendered.shape:
                raise ManualReviewError(f"Source/new review image shape mismatch: {source_image}")
            source_before = file_sha256(source_image)
            write_png(output_image, rendered)
            if file_sha256(source_image) != source_before:
                raise ManualReviewError(f"Source review image changed during migration: {source_image}")
            changed_images += int(not np.array_equal(old_image, rendered))

            root_image = output_root / str(row["image_path"])
            if root_image.is_file():
                write_png(root_image, rendered)
            for case_image in (
                output_video / "cases" / stem / "image_with_bbox.png",
                output_root / "cases" / stem / "image_with_bbox.png",
            ):
                if case_image.is_file():
                    write_png(case_image, rendered)
                    case_images_updated += 1

            source_mask = source_video / str(row["mask_path"])
            output_mask = output_video / str(row["mask_path"])
            if file_sha256(source_mask) != file_sha256(output_mask):
                raise ManualReviewError(f"Initial mask changed during migration: {video_name}/{stem}")
            masks_preserved += 1
            row["image_sha256"] = file_sha256(output_image)
            row["review_image_has_text_overlay"] = False
            row["review_image_render_version"] = RENDER_VERSION
            updated_rows.append(row)

    write_csv_rows(output_video / "review_frames.csv", updated_rows)
    _update_yaml(
        output_video / "run_config.yaml",
        {
            "review_image_has_text_overlay": False,
            "review_image_render_version": RENDER_VERSION,
            "textfree_source_review_root": str(source_root),
        },
    )
    checksum_rows, package_hash = _rewrite_checksum_contract(source_video, output_video)
    video_summary = _read_json(output_video / "export_summary.json")
    source_zip = source_video / "cvat" / "annotations_segmentation_mask_1_1.zip"
    output_zip = output_video / "cvat" / "annotations_segmentation_mask_1_1.zip"
    if file_sha256(source_zip) != file_sha256(output_zip):
        raise ManualReviewError(f"Initial CVAT mask ZIP changed: {video_name}")
    zip_result = validate_cvat_segmentation_mask_zip(
        output_zip,
        label_name="femur",
        images_dir=output_video / "images",
        reference_masks_dir=output_video / "masks",
    )
    if zip_result.images != len(updated_rows) or zip_result.reference_pixel_mismatches != 0:
        raise ManualReviewError(f"Migrated CVAT ZIP/image/mask contract mismatch: {video_name}")
    video_summary.update(
        {
            "review_images_have_text_overlay": False,
            "review_image_render_version": RENDER_VERSION,
            "textfree_source_review_root": str(source_root),
            "review_package_sha256": package_hash,
            "checksum_rows": checksum_rows,
        }
    )
    write_json(output_video / "export_summary.json", video_summary)
    return {
        "video_name": video_name,
        "frames": len(updated_rows),
        "review_bboxes": int(video_summary["selected_review_bboxes"]),
        "changed_images": changed_images,
        "masks_preserved": masks_preserved,
        "case_images_updated": case_images_updated,
        "checksum_rows": checksum_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_review_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if source_root == output_root:
        raise ManualReviewError("source_review_root and output_root must differ")
    for parent, child in ((source_root, output_root), (output_root, source_root)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise ManualReviewError(
            "source_review_root and output_root must not contain one another"
        )
    required = (
        "partition_summary.json",
        "video_progress.csv",
        "review_frames.csv",
        "review_bboxes.csv",
        "export_summary.json",
        "checksums.csv",
    )
    for name in required:
        if not (source_root / name).is_file():
            raise FileNotFoundError(source_root / name)

    progress = read_csv_rows(source_root / "video_progress.csv")
    videos = sorted(str(row["video_name"]) for row in progress)
    if len(videos) != int(args.expected_video_packages) or len(videos) != len(set(videos)):
        raise ManualReviewError(
            f"Source video package count mismatch: {len(videos)} != {args.expected_video_packages}"
        )
    expected_bboxes = sum(int(row["review_bboxes"]) for row in progress)
    if expected_bboxes != int(args.expected_review_bboxes):
        raise ManualReviewError(
            f"Source target BBox count mismatch: {expected_bboxes} != {args.expected_review_bboxes}"
        )

    prepare_output_root(output_root, overwrite=bool(args.overwrite))
    shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root, ignore=_copy_ignore(source_root))
    print("Stage 4 Phase 5 full-video text-free image-only migration")
    print(f"  source_root       : {source_root}")
    print(f"  output_root       : {output_root}")
    print(f"  video_packages    : {len(videos)}")
    print(f"  target_bboxes     : {expected_bboxes}")
    print(f"  render_version    : {RENDER_VERSION}")

    results: list[dict[str, Any]] = []
    for index, video_name in enumerate(videos, start=1):
        result = _render_video(
            source_root=source_root,
            output_root=output_root,
            video_name=video_name,
        )
        results.append(result)
        print(
            f"  [{index}/{len(videos)}] [OK] {video_name}: "
            f"frames={result['frames']}, changed={result['changed_images']}"
        )

    root_frame_rows = read_csv_rows(output_root / "review_frames.csv")
    rows_by_stem: dict[str, dict[str, Any]] = {}
    for video_name in videos:
        for row in read_csv_rows(output_root / "videos" / video_name / "review_frames.csv"):
            stem = str(row["frame_stem"])
            if stem in rows_by_stem:
                raise ManualReviewError(f"Duplicate full-video frame stem: {stem}")
            rows_by_stem[stem] = dict(row)
    if {str(row["frame_stem"]) for row in root_frame_rows} != set(rows_by_stem):
        raise ManualReviewError("Root/per-video frame stem sets differ")
    write_csv_rows(
        output_root / "review_frames.csv",
        [rows_by_stem[str(row["frame_stem"])] for row in root_frame_rows],
    )
    _update_yaml(
        output_root / "run_config.yaml",
        {
            "review_image_has_text_overlay": False,
            "review_image_render_version": RENDER_VERSION,
            "textfree_source_review_root": str(source_root),
        },
    )
    _, root_package_hash = _rewrite_checksum_contract(source_root, output_root)
    export_summary = _read_json(output_root / "export_summary.json")
    export_summary.update(
        {
            "review_images_have_text_overlay": False,
            "review_image_render_version": RENDER_VERSION,
            "textfree_source_review_root": str(source_root),
            "review_package_sha256": root_package_hash,
        }
    )
    write_json(output_root / "export_summary.json", export_summary)
    partition_summary = _read_json(output_root / "partition_summary.json")
    partition_summary.update(
        {
            "review_images_have_text_overlay": False,
            "review_image_render_version": RENDER_VERSION,
            "textfree_source_review_root": str(source_root),
        }
    )
    write_json(output_root / "partition_summary.json", partition_summary)

    total_frames = sum(int(result["frames"]) for result in results)
    validation_summary = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "status": "ok",
        "videos_checked": len(results),
        "frames_checked": total_frames,
        "target_bboxes_checked": sum(int(result["review_bboxes"]) for result in results),
        "masks_byte_identical": sum(int(result["masks_preserved"]) for result in results),
        "review_image_render_version": RENDER_VERSION,
        "review_images_have_text_overlay": False,
        "source_review_root_unchanged": True,
        "failure_rows": 0,
    }
    write_json(output_root / "package_validation_summary.json", validation_summary)
    migration_summary = {
        **validation_summary,
        "source_review_root": str(source_root),
        "output_root": str(output_root),
        "changed_images": sum(int(result["changed_images"]) for result in results),
        "case_images_updated": sum(int(result["case_images_updated"]) for result in results),
        "video_results": results,
    }
    write_json(output_root / "textfree_migration_summary.json", migration_summary)
    transfer_rows, transfer_summary = _write_transfer_manifest(
        review_root=output_root,
        hash_cache={},
        overwrite=True,
    )
    print("Stage 4 Phase 5 text-free review package migration passed.")
    print(f"  frames_preserved  : {total_frames}")
    print(f"  masks_preserved   : {validation_summary['masks_byte_identical']}")
    print(f"  changed_images    : {migration_summary['changed_images']}")
    print(f"  transfer_files    : {len(transfer_rows)}")
    print(f"  transfer_manifest : {output_root / 'mac_transfer_manifest.csv'}")
    return migration_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new Stage 4 full-video CVAT package by re-rendering only "
            "the review images without text; preserve masks, stems, shapes, and order."
        )
    )
    parser.add_argument("--source_review_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_video_packages", type=int, default=60)
    parser.add_argument("--expected_review_bboxes", type=int, default=129)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_video_packages <= 0 or args.expected_review_bboxes <= 0:
        raise SystemExit("expected counts must be > 0")
    try:
        run(args)
    except Exception as exc:
        raise SystemExit(
            f"Stage 4 Phase 5 text-free migration failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
