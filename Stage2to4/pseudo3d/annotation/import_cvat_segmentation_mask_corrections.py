from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_contour_teacher import mask_shape_metrics
from pseudo3d.annotation.stage4_manual_review import (
    MANUAL_REVIEW_SCHEMA_VERSION,
    ManualReviewError,
    apply_corrected_frame_mask,
    bbox_bounds,
    bbox_union_mask,
    file_sha256,
    load_manual_review_config,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import LocalBBox
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    load_allowed_missing_stems_file,
    read_cvat_segmentation_class_masks,
)


MANUAL_SOURCE = "manual_cvat_segmentation_mask_1_1_v1"


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ManualReviewError(f"{name} must be boolean: {value!r}")


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = str(value).split("|")
    if len(parts) != 4:
        raise ManualReviewError(f"Invalid bbox_local_xyxy: {value!r}")
    x1, y1, x2, y2 = (float(item) for item in parts)
    return x1, y1, x2, y2


def _parse_indices(value: str) -> list[int]:
    if not str(value).strip():
        return []
    result = [int(item) for item in str(value).split("|")]
    if result != sorted(set(result)):
        raise ManualReviewError(f"manual_bbox_indices must be sorted and unique: {value}")
    return result


def _preflight_corrected_masks(
    *,
    frame_rows: Sequence[Mapping[str, str]],
    bbox_rows: Sequence[Mapping[str, str]],
    corrected_masks: Mapping[str, np.ndarray],
    require_nonempty_manual_bbox: bool,
) -> None:
    """Validate every corrected frame before any output H5 is created."""
    bbox_by_stem: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in bbox_rows:
        bbox_by_stem[str(row["frame_stem"])].append(row)
    empty_labels = np.empty((0,), dtype=np.int8)
    empty_orders = np.empty((0,), dtype=np.int64)
    empty_xy = np.empty((0, 2), dtype=np.float32)
    for frame_row in sorted(frame_rows, key=lambda row: str(row["frame_stem"])):
        stem = str(frame_row["frame_stem"])
        mask = np.asarray(corrected_masks[stem], dtype=np.uint8)
        expected_shape = (int(frame_row["height"]), int(frame_row["width"]))
        if mask.shape != expected_shape:
            raise ManualReviewError(
                f"{stem}: corrected mask shape mismatch: {mask.shape} != {expected_shape}"
            )
        rows = sorted(
            bbox_by_stem.get(stem, []), key=lambda row: int(row["bbox_index"])
        )
        if [int(row["bbox_index"]) for row in rows] != list(range(len(rows))):
            raise ManualReviewError(f"{stem}: BBox indices are not contiguous")
        bboxes = [_parse_bbox(str(row["bbox_local_xyxy"])) for row in rows]
        manual_indices = _parse_indices(str(frame_row["manual_bbox_indices"]))
        expected_manual = [
            int(row["bbox_index"])
            for row in rows
            if _bool(row["manual_review_required"], "manual_review_required")
        ]
        if manual_indices != expected_manual:
            raise ManualReviewError(f"{stem}: manual BBox manifest mismatch")
        try:
            apply_corrected_frame_mask(
                point_labels=empty_labels,
                frame_orders=empty_orders,
                pixel_xy=empty_xy,
                target_frame_order=int(frame_row["frame_order"]),
                corrected_mask=mask,
                bboxes=bboxes,
                manual_bbox_indices=manual_indices,
                require_nonempty_manual_bbox=require_nonempty_manual_bbox,
            )
        except ManualReviewError as exc:
            raise ManualReviewError(f"{stem}: {exc}") from exc


def _resolve_context_only_stems(
    *,
    review_root: Path,
    frame_rows: Sequence[Mapping[str, str]],
    bbox_rows: Sequence[Mapping[str, str]],
) -> list[str]:
    """Resolve frames that have context value but no drawable manual BBox.

    New packages record this in both review_frames.csv and
    context_only_stems.txt.  Geometry-based inference keeps already exported
    representative packages usable without another CVAT editing pass.
    """
    rows_by_stem: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in bbox_rows:
        rows_by_stem[str(row["frame_stem"])].append(row)
    expected: set[str] = set()
    frame_stems = {str(row["frame_stem"]) for row in frame_rows}
    for frame_row in frame_rows:
        stem = str(frame_row["frame_stem"])
        shape = (int(frame_row["height"]), int(frame_row["width"]))
        requested_rows = []
        for bbox_row in rows_by_stem.get(stem, []):
            decision = str(bbox_row.get("phase3_decision", ""))
            manual_required = _bool(
                bbox_row.get("manual_review_required", False),
                "manual_review_required",
            )
            if manual_required or decision in {
                "manual_review",
                "manual_review_unrenderable",
            }:
                requested_rows.append(bbox_row)
        drawable = any(
            bbox_bounds(_parse_bbox(str(row["bbox_local_xyxy"])), shape) is not None
            for row in requested_rows
        )
        if not drawable:
            expected.add(stem)

    allowlist_path = Path(review_root) / "context_only_stems.txt"
    listed: set[str] | None = None
    if allowlist_path.is_file():
        listed = set(load_allowed_missing_stems_file(allowlist_path))
        unknown = sorted(listed - frame_stems)
        if unknown:
            raise ManualReviewError(
                f"context_only_stems.txt contains unknown frames: {unknown}"
            )

    explicit_columns = all("context_only" in row for row in frame_rows)
    if explicit_columns:
        declared = {
            str(row["frame_stem"])
            for row in frame_rows
            if _bool(row["context_only"], "context_only")
        }
        if listed is not None and listed != declared:
            raise ManualReviewError(
                "context_only_stems.txt differs from review_frames.csv: "
                f"listed={sorted(listed)}, declared={sorted(declared)}"
            )
        return sorted(declared)

    # Legacy representative packages predate the context_only CSV columns.
    # For those packages an explicit allowlist records the human decision that
    # a frame is useful as context but is not safely annotatable. The ZIP
    # validator and unchanged-mask check still prevent silent mask corruption.
    if listed is not None:
        return sorted(listed)
    return sorted(expected)


def _review_package_contract(review_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    review_root = Path(review_root).resolve()
    required = {
        "review_frames.csv",
        "review_bboxes.csv",
        "checksums.csv",
        "export_summary.json",
        "run_config.yaml",
    }
    missing = sorted(name for name in required if not (review_root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Review package is incomplete: missing={missing}")
    summary = json.loads((review_root / "export_summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping) or str(summary.get("status", "")) != "ok":
        raise ManualReviewError("Review package status is not ok")
    if int(summary.get("schema_version", 0)) != MANUAL_REVIEW_SCHEMA_VERSION:
        raise ManualReviewError("Review package schema_version mismatch")
    checksum_rows = read_csv_rows(review_root / "checksums.csv")
    seen: set[str] = set()
    for row in checksum_rows:
        relative = str(row.get("relative_path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ManualReviewError(f"Invalid/duplicate review checksum path: {relative!r}")
        seen.add(relative)
        target = review_root / relative
        if not target.is_file() or file_sha256(target) != str(row.get("sha256", "")):
            raise ManualReviewError(f"Review package checksum mismatch: {relative}")
    from pseudo3d.batch.export.batch_export_stage4_manual_review_cvat import (
        hashlib_sha256_rows,
    )

    if hashlib_sha256_rows(checksum_rows) != str(summary.get("review_package_sha256", "")):
        raise ManualReviewError("Review package aggregate checksum mismatch")
    frames = read_csv_rows(review_root / "review_frames.csv")
    bboxes = read_csv_rows(review_root / "review_bboxes.csv")
    if len(frames) != int(summary.get("review_frames", -1)):
        raise ManualReviewError("review_frames.csv count differs from summary")
    if len(bboxes) != int(summary.get("bbox_rows", -1)):
        raise ManualReviewError("review_bboxes.csv count differs from summary")
    stems = [row["frame_stem"] for row in frames]
    if stems != sorted(set(stems)):
        raise ManualReviewError("review frame stems must be sorted and unique")
    return frames, bboxes, dict(summary)


def _load_review_session(
    path: Path,
    *,
    package_summary: Mapping[str, Any],
    cvat_export_zip: Path,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"review_session not found: {path}")
    with path.open(encoding="utf-8") as handle:
        session = yaml.safe_load(handle)
    if not isinstance(session, Mapping):
        raise ManualReviewError("review_session root must be a mapping")
    result = dict(session)
    if int(result.get("review_schema_version", 0)) != MANUAL_REVIEW_SCHEMA_VERSION:
        raise ManualReviewError("review_session schema_version mismatch")
    if str(result.get("review_package_sha256", "")) != str(
        package_summary.get("review_package_sha256", "")
    ):
        raise ManualReviewError("review_session package checksum mismatch")
    if str(result.get("imported_zip_sha256", "")) != str(
        package_summary.get("cvat_import_zip_sha256", "")
    ):
        raise ManualReviewError("review_session imported ZIP checksum mismatch")
    actual_export_hash = file_sha256(cvat_export_zip)
    if str(result.get("exported_zip_sha256", "")) != actual_export_hash:
        raise ManualReviewError("review_session exported ZIP checksum mismatch")
    if not _bool(result.get("all_manual_bboxes_reviewed", False), "all_manual_bboxes_reviewed"):
        raise ManualReviewError("review_session is not marked complete")
    for key in (
        "cvat_version",
        "cvat_task_identifier",
        "reviewer_identifier",
        "pre_import_backup_sha256",
        "review_started_utc",
        "review_completed_utc",
    ):
        if not str(result.get(key, "")).strip():
            raise ManualReviewError(f"review_session field must not be empty: {key}")
    return result


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    if name not in group:
        raise KeyError(f"Required H5 dataset not found: {group.name}/{name}")
    old = group[name]
    kwargs: dict[str, Any] = {}
    if old.compression is not None:
        kwargs["compression"] = old.compression
        kwargs["compression_opts"] = old.compression_opts
    if old.shuffle:
        kwargs["shuffle"] = True
    if old.fletcher32:
        kwargs["fletcher32"] = True
    del group[name]
    group.create_dataset(name, data=data, **kwargs)


def _replace_text_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype("utf-8")))


def _decode_text_array(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _contour_area(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return float(sum(cv2.contourArea(contour) for contour in contours))


def import_video_corrections(
    *,
    source_h5: Path,
    output_h5: Path,
    frame_rows: Sequence[Mapping[str, str]],
    bbox_rows: Sequence[Mapping[str, str]],
    corrected_masks: Mapping[str, np.ndarray],
    session: Mapping[str, Any],
    package_summary: Mapping[str, Any],
    manual_config: Any,
    cvat_export_zip: Path,
    overwrite: bool,
) -> dict[str, Any]:
    source_h5 = Path(source_h5).resolve()
    output_h5 = Path(output_h5).resolve(strict=False)
    if source_h5 == output_h5:
        raise ManualReviewError("Manual import must not overwrite the source H5")
    if output_h5.exists() and not overwrite:
        raise FileExistsError(f"Output H5 exists; pass --overwrite: {output_h5}")
    expected_source_hash = {row["source_annotated_h5_sha256"] for row in frame_rows}
    if len(expected_source_hash) != 1 or file_sha256(source_h5) not in expected_source_hash:
        raise ManualReviewError("source annotated H5 checksum differs from review package")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_h5.name}.", suffix=".tmp", dir=output_h5.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    shutil.copy2(source_h5, temporary)
    added_pixels = removed_pixels = reviewed_bboxes = 0
    try:
        with h5py.File(temporary, "r+") as handle:
            if "manual_annotation_provenance" in handle.attrs:
                raise ManualReviewError(
                    "source H5 already has manual provenance; start a new revision from the original source"
                )
            for group_name in ("point_cloud", "annotation", "frame_annotation"):
                if group_name not in handle:
                    raise KeyError(f"source H5 missing group: {group_name}")
            pc = handle["point_cloud"]
            ann = handle["annotation"]
            frame_group = handle["frame_annotation"]
            for name in ("frame_order", "pixel_xy"):
                if name not in pc:
                    raise KeyError(f"source H5 missing point_cloud/{name}")
            for name in ("point_label", "valid_mask"):
                if name not in ann:
                    raise KeyError(f"source H5 missing annotation/{name}")
            labels = ann["point_label"][:].astype(np.int8)
            valid = ann["valid_mask"][:].astype(bool)
            if labels.shape != valid.shape or not np.array_equal(valid, labels != -1):
                raise ManualReviewError("source annotation label/valid_mask contract is invalid")
            frame_orders = pc["frame_order"][:]
            pixel_xy = pc["pixel_xy"][:]
            bbox_by_stem: dict[str, list[Mapping[str, str]]] = defaultdict(list)
            for row in bbox_rows:
                bbox_by_stem[row["frame_stem"]].append(row)
            per_frame_stats: list[dict[str, Any]] = []
            for frame_row in sorted(frame_rows, key=lambda row: row["frame_stem"]):
                stem = frame_row["frame_stem"]
                mask = np.asarray(corrected_masks[stem], dtype=np.uint8)
                expected_shape = (int(frame_row["height"]), int(frame_row["width"]))
                if mask.shape != expected_shape:
                    raise ManualReviewError(f"Corrected mask shape mismatch for {stem}")
                rows = sorted(bbox_by_stem[stem], key=lambda row: int(row["bbox_index"]))
                if [int(row["bbox_index"]) for row in rows] != list(range(len(rows))):
                    raise ManualReviewError(f"BBox indices are not contiguous for {stem}")
                bboxes = [_parse_bbox(row["bbox_local_xyxy"]) for row in rows]
                manual_indices = _parse_indices(frame_row["manual_bbox_indices"])
                if manual_indices != [
                    int(row["bbox_index"])
                    for row in rows
                    if _bool(row["manual_review_required"], "manual_review_required")
                ]:
                    raise ManualReviewError(f"Manual BBox manifest mismatch for {stem}")
                labels, stats = apply_corrected_frame_mask(
                    point_labels=labels,
                    frame_orders=frame_orders,
                    pixel_xy=pixel_xy,
                    target_frame_order=int(frame_row["frame_order"]),
                    corrected_mask=mask,
                    bboxes=bboxes,
                    manual_bbox_indices=manual_indices,
                    require_nonempty_manual_bbox=manual_config.require_nonempty_manual_bbox,
                )
                initial_path = Path(frame_row["mask_path"])
                review_root = Path(package_summary["review_root"])
                initial = cv2.imread(str(review_root / initial_path), cv2.IMREAD_UNCHANGED)
                if initial is None:
                    raise FileNotFoundError(f"Initial review mask missing: {initial_path}")
                initial_bool = initial > 0
                corrected_bool = mask.astype(bool)
                added = int(np.sum(corrected_bool & ~initial_bool))
                removed = int(np.sum(initial_bool & ~corrected_bool))
                added_pixels += added
                removed_pixels += removed
                reviewed_bboxes += len(manual_indices)
                per_frame_stats.append(
                    {
                        "frame_stem": stem,
                        "frame_order": int(frame_row["frame_order"]),
                        "frame_index": int(frame_row["frame_index"]),
                        "manual_bbox_count": len(manual_indices),
                        "added_positive_pixels": added,
                        "removed_positive_pixels": removed,
                        **stats,
                    }
                )

                if not {"frame_order", "frame_index", "bbox_index"}.issubset(frame_group.keys()):
                    raise KeyError("source frame_annotation lacks alignment datasets")
                row_orders = frame_group["frame_order"][:].astype(np.int64)
                row_indices = frame_group["frame_index"][:].astype(np.int64)
                row_bbox_indices = frame_group["bbox_index"][:].astype(np.int64)
                selected_source = _decode_text_array(frame_group["selected_contour_source"][:])
                reasons = _decode_text_array(frame_group["annotation_reason"][:])
                for bbox_row, bbox in zip(rows, bboxes, strict=True):
                    bbox_index = int(bbox_row["bbox_index"])
                    matches = np.flatnonzero(
                        (row_orders == int(frame_row["frame_order"]))
                        & (row_indices == int(frame_row["frame_index"]))
                        & (row_bbox_indices == bbox_index)
                    )
                    if matches.size != 1:
                        raise ManualReviewError(
                            f"Expected one frame_annotation row for {stem}/bbox={bbox_index}"
                        )
                    saved_index = int(matches[0])
                    region = bbox_union_mask(mask.shape, [bbox])
                    bbox_mask = corrected_bool & region
                    point_selector = frame_orders.astype(np.int64) == int(frame_row["frame_order"])
                    xy = np.rint(pixel_xy[point_selector]).astype(np.int64)
                    positive_points = int(np.sum(bbox_mask[xy[:, 1], xy[:, 0]])) if xy.size else 0
                    bbox_values = np.asarray(bbox, dtype=np.float32)
                    local_bbox = LocalBBox(
                        xml_xyxy=bbox_values.copy(),
                        raw_xyxy=bbox_values.copy(),
                        local_xyxy=bbox_values.copy(),
                        valid=True,
                    )
                    metrics = mask_shape_metrics(bbox_mask, local_bbox)
                    if bbox_index in manual_indices:
                        selected_source[saved_index] = MANUAL_SOURCE
                        reasons[saved_index] = "manual_review_completed"
                    for name, value in (
                        ("valid_contour", True),
                        ("contour_area", _contour_area(bbox_mask)),
                        ("binary_area", int(bbox_mask.sum())),
                        ("foreground_ratio_in_bbox", float(metrics["area_ratio"])),
                        ("num_labeled_points", positive_points),
                        ("contour_positive_points", positive_points),
                        ("num_labeled_points_union", int(stats["positive_points"])),
                        ("selected_area_ratio", float(metrics["area_ratio"])),
                        ("center_distance_norm", float(metrics["center_distance_norm"])),
                    ):
                        if name in frame_group:
                            frame_group[name][saved_index] = value
                _replace_text_dataset(frame_group, "selected_contour_source", selected_source)
                _replace_text_dataset(frame_group, "annotation_reason", reasons)

            _replace_dataset(ann, "point_label", labels.astype(np.int8))
            _replace_dataset(ann, "valid_mask", (labels != -1).astype(bool))
            ann.attrs["label_source"] = MANUAL_SOURCE
            ann.attrs["manual_annotation_provenance"] = MANUAL_SOURCE
            if "manual_review" in handle:
                del handle["manual_review"]
            review_group = handle.create_group("manual_review")
            review_group.create_dataset(
                "frame_stem",
                data=np.asarray(
                    [row["frame_stem"] for row in per_frame_stats],
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            for name in (
                "frame_order",
                "frame_index",
                "manual_bbox_count",
                "added_positive_pixels",
                "removed_positive_pixels",
                "frame_points",
                "positive_points",
                "ignore_points",
                "background_points",
                "corrected_positive_pixels",
            ):
                review_group.create_dataset(name, data=np.asarray([row[name] for row in per_frame_stats]))
            handle.attrs["manual_annotation_provenance"] = MANUAL_SOURCE
            handle.attrs["manual_review_schema_version"] = MANUAL_REVIEW_SCHEMA_VERSION
            handle.attrs["manual_review_package_sha256"] = str(package_summary["review_package_sha256"])
            handle.attrs["manual_review_session_sha256"] = file_sha256(Path(session["session_path"]))
            handle.attrs["cvat_export_sha256"] = file_sha256(cvat_export_zip)
            handle.attrs["cvat_task_identifier"] = str(session["cvat_task_identifier"])
            handle.attrs["reviewer_identifier"] = str(session["reviewer_identifier"])
            handle.attrs["manual_reviewed_frame_count"] = len(per_frame_stats)
            handle.attrs["manual_reviewed_bbox_count"] = reviewed_bboxes
            handle.attrs["manual_added_positive_pixels"] = added_pixels
            handle.attrs["manual_removed_positive_pixels"] = removed_pixels
            handle.attrs["source_annotated_h5_sha256"] = file_sha256(source_h5)
            handle.attrs["teacher_fingerprint"] = str(package_summary["teacher_fingerprint"])
            handle.attrs["refine_fingerprint"] = str(package_summary["refine_fingerprint"])
            handle.attrs["manual_review_fingerprint"] = manual_config.fingerprint
            handle.flush()
        if output_h5.exists():
            output_h5.unlink()
        os.replace(temporary, output_h5)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "source_h5": str(source_h5),
        "output_h5": str(output_h5),
        "output_h5_sha256": file_sha256(output_h5),
        "reviewed_frames": len(frame_rows),
        "reviewed_bboxes": reviewed_bboxes,
        "added_positive_pixels": added_pixels,
        "removed_positive_pixels": removed_pixels,
    }


def run_manual_import(args: argparse.Namespace) -> dict[str, Any]:
    manual_config = load_manual_review_config(args.manual_review_config)
    review_root = Path(args.review_root).resolve()
    frame_rows, bbox_rows, package_summary = _review_package_contract(review_root)
    package_summary["review_root"] = str(review_root)
    if str(package_summary.get("manual_review_fingerprint", "")) != manual_config.fingerprint:
        raise ManualReviewError("Current manual-review config differs from review package")
    session = _load_review_session(
        args.review_session,
        package_summary=package_summary,
        cvat_export_zip=args.cvat_export_zip,
    )
    session["session_path"] = str(Path(args.review_session).resolve())
    context_only_stems = _resolve_context_only_stems(
        review_root=review_root,
        frame_rows=frame_rows,
        bbox_rows=bbox_rows,
    )
    context_only_set = set(context_only_stems)
    zip_summary, corrected_masks = read_cvat_segmentation_class_masks(
        args.cvat_export_zip,
        label_name=manual_config.label_name,
        images_dir=review_root / "images",
        allowed_missing_mask_stems=context_only_stems,
    )
    expected_stems = [row["frame_stem"] for row in frame_rows]
    if zip_summary.stems != expected_stems or set(corrected_masks) != set(expected_stems):
        raise ManualReviewError("CVAT export stems differ from review manifest")
    missing_context = set(zip_summary.missing_mask_stems)
    changed_context: list[str] = []
    for stem in context_only_stems:
        if stem in missing_context:
            continue
        frame_row = next(row for row in frame_rows if row["frame_stem"] == stem)
        initial = cv2.imread(
            str(review_root / frame_row["mask_path"]), cv2.IMREAD_UNCHANGED
        )
        if initial is None:
            raise FileNotFoundError(
                f"Initial context-only mask is missing: {frame_row['mask_path']}"
            )
        initial_binary = (np.asarray(initial) > 0).astype(np.uint8)
        if not np.array_equal(corrected_masks[stem], initial_binary):
            changed_context.append(stem)
    if changed_context:
        raise ManualReviewError(
            "Context-only frames must be omitted or remain unchanged in the CVAT "
            f"export: {changed_context}"
        )
    source_root = Path(args.source_annotated_root).resolve()
    by_video_frames: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_video_bboxes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        source = Path(row["source_annotated_h5"]).resolve()
        if not source.is_relative_to(source_root):
            raise ManualReviewError(f"Source H5 is outside source_annotated_root: {source}")
        if file_sha256(source) != row["source_annotated_h5_sha256"]:
            raise ManualReviewError(f"Source H5 checksum is stale: {source}")
        pseudo = Path(row["source_pseudo3d_h5"]).resolve()
        if file_sha256(pseudo) != row["source_pseudo3d_h5_sha256"]:
            raise ManualReviewError(f"Source pseudo3D H5 checksum is stale: {pseudo}")
        image = review_root / row["image_path"]
        mask = review_root / row["mask_path"]
        if file_sha256(image) != row["image_sha256"] or file_sha256(mask) != row["mask_sha256"]:
            raise ManualReviewError(f"Review image/mask checksum is stale: {row['frame_stem']}")
        if row["frame_stem"] not in context_only_set:
            by_video_frames[row["video_name"]].append(row)
    for row in bbox_rows:
        xml = Path(row["xml_path"])
        if not xml.is_file() or file_sha256(xml) != row["xml_sha256"]:
            raise ManualReviewError(f"Strict XML checksum is stale: {xml}")
        if row["frame_stem"] not in context_only_set:
            by_video_bboxes[row["video_name"]].append(row)
    if set(by_video_frames) != set(by_video_bboxes):
        raise ManualReviewError("Review frame/BBox video sets differ")

    actionable_frame_rows = [
        row for row in frame_rows if row["frame_stem"] not in context_only_set
    ]
    actionable_bbox_rows = [
        row for row in bbox_rows if row["frame_stem"] not in context_only_set
    ]
    _preflight_corrected_masks(
        frame_rows=actionable_frame_rows,
        bbox_rows=actionable_bbox_rows,
        corrected_masks=corrected_masks,
        require_nonempty_manual_bbox=manual_config.require_nonempty_manual_bbox,
    )

    output_root = Path(args.output_root)
    if output_root.exists() and output_root.is_symlink():
        raise ManualReviewError("output_root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for video_name in sorted(by_video_frames):
        sources = {Path(row["source_annotated_h5"]).resolve() for row in by_video_frames[video_name]}
        if len(sources) != 1:
            raise ManualReviewError(f"Video maps to multiple source H5 files: {video_name}")
        output_h5 = output_root / video_name / f"{video_name}{manual_config.output_suffix}"
        result = import_video_corrections(
            source_h5=next(iter(sources)),
            output_h5=output_h5,
            frame_rows=by_video_frames[video_name],
            bbox_rows=by_video_bboxes[video_name],
            corrected_masks=corrected_masks,
            session=session,
            package_summary=package_summary,
            manual_config=manual_config,
            cvat_export_zip=args.cvat_export_zip,
            overwrite=args.overwrite,
        )
        result["video_name"] = video_name
        results.append(result)
        print(f"[OK] {video_name}: frames={result['reviewed_frames']}, bboxes={result['reviewed_bboxes']}")
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary.csv"
    write_csv_rows(summary_csv, results)
    summary = {
        "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
        "status": "ok",
        "videos": len(results),
        "reviewed_frames": sum(int(row["reviewed_frames"]) for row in results),
        "reviewed_bboxes": sum(int(row["reviewed_bboxes"]) for row in results),
        "context_only_frames": len(context_only_stems),
        "cvat_missing_context_masks": int(zip_summary.missing_masks),
        "added_positive_pixels": sum(int(row["added_positive_pixels"]) for row in results),
        "removed_positive_pixels": sum(int(row["removed_positive_pixels"]) for row in results),
        "cvat_export_sha256": file_sha256(args.cvat_export_zip),
        "review_package_sha256": package_summary["review_package_sha256"],
        "failure_rows": 0,
    }
    write_json(output_root / "import_summary.json", summary)
    print("Stage 4 Phase 4 manual import passed.")
    return {"summary": summary, "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly import a local CVAT Segmentation Mask 1.1 correction into new Stage 4 H5 files."
    )
    parser.add_argument("--review_root", type=Path, required=True)
    parser.add_argument("--cvat_export_zip", type=Path, required=True)
    parser.add_argument("--review_session", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--manual_review_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_manual_import(args)
    except Exception as exc:
        raise SystemExit(f"Phase 4 manual import failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
