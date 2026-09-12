from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_contour_teacher import (
    _assert_metadata_matches_teacher,
    _assert_snapshot_unchanged,
    _read_annotated_h5,
    _read_pseudo3d,
    _snapshot,
    load_audit_teacher_config,
    mask_shape_metrics,
    resolve_inputs,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_FEMUR_CANDIDATE,
    build_bbox_ranked_contour_mask,
    build_full_frame_binary_mask,
    build_ranked_local_binary_mask,
    load_voc_bboxes,
    summarize_labels_by_source,
    xml_bbox_to_local,
)
from pseudo3d.annotation.contour_teacher_refinement import (
    load_contour_refinement_config,
    mask_sha256,
)
from pseudo3d.annotation.stage4_manual_review import (
    ManualReviewError,
    apply_corrected_frame_mask,
    compose_initial_frame_mask,
    write_csv_rows,
    write_json,
)
from pseudo3d.batch.export.batch_export_stage4_manual_review_cvat import (
    _phase3_contract,
    _read_proposal,
)
from src.utils.alpha_texture_processing import image_to_uint8_gray


PHASE5_SCHEMA_VERSION = 1
SOURCE_TOKEN = "bboxrank_v2_nobbox_bg"
OUTPUT_TOKEN = "bboxrank_v3_refined_auto_v1"
OUTPUT_SUFFIX = f"_{OUTPUT_TOKEN}.h5"
DEFAULT_ANNOTATED_GLOB = "{video_name}/*bboxrank_v2_nobbox_bg.h5"


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _output_name(source: Path) -> str:
    if SOURCE_TOKEN not in source.name:
        raise ValueError(f"Source filename lacks fixed teacher token: {source.name}")
    return source.name.replace(SOURCE_TOKEN, OUTPUT_TOKEN)


def _replace_text_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    if name not in group:
        raise KeyError(f"Required text dataset not found: {group.name}/{name}")
    del group[name]
    group.create_dataset(
        name,
        data=np.asarray(list(values), dtype=h5py.string_dtype("utf-8")),
    )


def _contour_area(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return float(sum(cv2.contourArea(contour) for contour in contours))


def _point_count(mask: np.ndarray, xy: np.ndarray) -> int:
    points = np.rint(np.asarray(xy)).astype(np.int64)
    if points.size == 0:
        return 0
    height, width = mask.shape
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        raise ManualReviewError("point_cloud/pixel_xy lies outside the local image")
    return int(np.sum(mask[points[:, 1], points[:, 0]]))


def _frame_row_index(
    frame_annotation: Mapping[str, np.ndarray],
    *,
    frame_order: int,
    frame_index: int,
    bbox_index: int,
) -> int:
    matches = np.flatnonzero(
        (frame_annotation["frame_order"].astype(np.int64) == int(frame_order))
        & (frame_annotation["frame_index"].astype(np.int64) == int(frame_index))
        & (frame_annotation["bbox_index"].astype(np.int64) == int(bbox_index))
    )
    if matches.size != 1:
        raise ManualReviewError(
            "Expected one source frame_annotation row for "
            f"frame_order={frame_order}, frame_index={frame_index}, "
            f"bbox_index={bbox_index}; matches={matches.size}"
        )
    return int(matches[0])


def _write_refinement_group(
    handle: h5py.File,
    *,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, str],
) -> None:
    if "contour_refinement" in handle:
        del handle["contour_refinement"]
    group = handle.create_group("contour_refinement")
    text_fields = (
        "video_name",
        "reason_codes",
        "proposal_source",
        "baseline_mask_sha256",
        "proposal_mask_sha256",
    )
    int_fields = (
        "frame_order",
        "frame_index",
        "bbox_index",
        "before_positive_points",
        "after_positive_points",
    )
    float_fields = (
        "score_gain",
        "score_margin",
        "proposal_score",
        "proposal_area_ratio",
        "proposal_center_distance_norm",
        "proposal_stability_iou",
    )
    for name in text_fields:
        group.create_dataset(
            name,
            data=np.asarray(
                [str(row.get(name, "")) for row in rows],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
    for name in int_fields:
        group.create_dataset(
            name,
            data=np.asarray([int(row.get(name, 0)) for row in rows], dtype=np.int64),
        )
    for name in float_fields:
        group.create_dataset(
            name,
            data=np.asarray(
                [float(row.get(name, math.nan)) for row in rows], dtype=np.float64
            ),
        )
    for name, value in metadata.items():
        group.attrs[name] = str(value)


def _set_provenance(
    handle: h5py.File,
    *,
    source_h5: Path,
    teacher_fingerprint: str,
    refine_fingerprint: str,
    phase3_root: Path,
    decision_sha256: str,
    refined_rows: Sequence[Mapping[str, Any]],
) -> None:
    metadata = {
        "schema_version": str(PHASE5_SCHEMA_VERSION),
        "teacher_version": OUTPUT_TOKEN,
        "base_teacher_version": SOURCE_TOKEN,
        "policy": "auto_refine_only; auto_accept/manual_review/unselected preserve v2",
        "teacher_fingerprint": teacher_fingerprint,
        "refine_fingerprint": refine_fingerprint,
        "phase3_root": str(Path(phase3_root).resolve()),
        "phase3_bbox_decisions_sha256": decision_sha256,
        "source_annotated_h5": str(Path(source_h5).resolve()),
        "source_annotated_h5_sha256": file_sha256(source_h5),
    }
    _write_refinement_group(handle, rows=refined_rows, metadata=metadata)
    handle.attrs["contour_teacher_schema"] = OUTPUT_TOKEN
    handle.attrs["contour_refinement_policy"] = metadata["policy"]
    handle.attrs["contour_refinement_teacher_fingerprint"] = teacher_fingerprint
    handle.attrs["contour_refinement_config_fingerprint"] = refine_fingerprint
    handle.attrs["contour_refinement_decisions_sha256"] = decision_sha256
    handle.attrs["contour_refinement_source_h5_sha256"] = metadata[
        "source_annotated_h5_sha256"
    ]
    handle.attrs["contour_refined_bbox_count"] = len(refined_rows)
    annotation = handle["annotation"]
    annotation.attrs["label_source"] = OUTPUT_TOKEN
    annotation.attrs["contour_refinement_note"] = metadata["policy"]
    labels = annotation["point_label"][:].astype(np.int8)
    valid = annotation["valid_mask"][:].astype(bool)
    if not np.array_equal(valid, labels != -1):
        raise ManualReviewError("Phase 5 valid_mask differs from point_label != ignore")
    frame_group = handle["frame_annotation"]
    valid_contours = frame_group["valid_contour"][:].astype(bool)
    valid_orders = frame_group["frame_order"][:].astype(np.int64)[valid_contours]
    selected_sources = np.asarray(
        [_decode_text(value) for value in frame_group["selected_contour_source"][:]]
    )
    handle.attrs["num_labeled_points"] = int(np.sum(labels == 1))
    handle.attrs["num_valid_contours"] = int(valid_contours.sum())
    handle.attrs["num_valid_contour_frames"] = int(np.unique(valid_orders).size)
    if "fallback_used" in frame_group:
        handle.attrs["num_fallback_used"] = int(
            frame_group["fallback_used"][:].astype(bool).sum()
        )
    handle.attrs["num_selected_global_contours"] = int(
        np.sum(selected_sources == "global")
    )
    handle.attrs["num_selected_local_contours"] = int(
        np.sum(selected_sources == "local_percentile")
    )
    handle.attrs["num_selected_shared_contours"] = int(
        np.sum(selected_sources == "shared")
    )
    handle.attrs["num_selected_phase3_refined_contours"] = int(
        np.sum(np.char.startswith(selected_sources.astype(str), "phase3_"))
    )
    source_stats = summarize_labels_by_source(
        point_cloud={"source_flags": handle["point_cloud/source_flags"][:]},
        point_label=labels,
        valid_mask=valid,
    )
    for name, value in source_stats.items():
        handle.attrs[name] = int(value)
    handle.attrs["label_source"] = (
        "VOC BBox-ranked teacher v2 with fixed Phase 3 auto-refine proposals"
    )


def _validate_existing_output(
    path: Path,
    *,
    source_h5: Path,
    teacher_fingerprint: str,
    refine_fingerprint: str,
    decision_sha256: str,
) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        required = {"point_cloud", "annotation", "frame_annotation", "contour_refinement"}
        missing = required - set(handle.keys())
        if missing:
            raise ManualReviewError(f"Existing Phase 5 output is incomplete: {sorted(missing)}")
        expected = {
            "contour_teacher_schema": OUTPUT_TOKEN,
            "contour_refinement_teacher_fingerprint": teacher_fingerprint,
            "contour_refinement_config_fingerprint": refine_fingerprint,
            "contour_refinement_decisions_sha256": decision_sha256,
            "contour_refinement_source_h5_sha256": file_sha256(source_h5),
        }
        for name, value in expected.items():
            if _decode_text(handle.attrs.get(name, "")) != str(value):
                raise ManualReviewError(f"Existing Phase 5 provenance mismatch: {name}")
        labels = handle["annotation/point_label"][:].astype(np.int8)
        valid = handle["annotation/valid_mask"][:].astype(bool)
        if not np.array_equal(valid, labels != -1):
            raise ManualReviewError("Existing Phase 5 valid_mask is inconsistent")
        return {
            "points": int(labels.size),
            "positive": int(np.sum(labels == 1)),
            "ignore": int(np.sum(labels == -1)),
            "refined": int(handle.attrs.get("contour_refined_bbox_count", 0)),
        }


def _apply_one_video(
    *,
    video_name: str,
    pseudo3d_h5: Path,
    source_h5: Path,
    output_h5: Path,
    voc_xml_root: Path,
    decisions: Sequence[Mapping[str, str]],
    teacher: Any,
    refine_fingerprint: str,
    phase3_root: Path,
    decision_sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists() and not overwrite:
        raise FileExistsError(f"Output H5 exists; pass --overwrite: {output_h5}")

    point_cloud, frame_annotation, source_attrs = _read_annotated_h5(source_h5)
    _assert_metadata_matches_teacher(source_attrs, teacher)
    original_labels = point_cloud["point_label"].astype(np.int8)
    labels = original_labels.copy()
    frame_orders = point_cloud["frame_order"].astype(np.int64)
    pixel_xy = point_cloud["pixel_xy"]
    refined_decisions = [
        dict(row) for row in decisions if str(row["proposed_decision"]) == "auto_refine"
    ]
    refined_by_frame: dict[tuple[int, int], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in refined_decisions:
        key = (int(row["frame_order"]), int(row["frame_index"]))
        bbox_index = int(row["bbox_index"])
        if bbox_index in refined_by_frame[key]:
            raise ManualReviewError(f"Duplicate auto_refine BBox decision: {video_name}/{key}/{bbox_index}")
        refined_by_frame[key][bbox_index] = row

    frame_updates: dict[int, dict[str, Any]] = {}
    union_updates: dict[int, int] = {}
    refinement_rows: list[dict[str, Any]] = []
    if refined_by_frame:
        images, frame_indices, pseudo_attrs = _read_pseudo3d(pseudo3d_h5)
        voc_bboxes = load_voc_bboxes(
            voc_xml_root,
            video_name=video_name,
            frame_indices=frame_indices,
            xml_frame_number_offsets=teacher.xml_frame_number_offsets,
            xml_frame_id_source=teacher.xml_frame_id_source,
            xml_annotation_dir_name=teacher.xml_annotation_dir_name,
            strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
        )
        for (frame_order, frame_index), decisions_by_bbox in sorted(refined_by_frame.items()):
            if frame_order < 0 or frame_order >= len(frame_indices):
                raise ManualReviewError(f"Phase 5 frame_order outside pseudo3D input: {frame_order}")
            if int(frame_indices[frame_order]) != frame_index:
                raise ManualReviewError(
                    f"Phase 5 frame_index mismatch: {frame_index} != {frame_indices[frame_order]}"
                )
            vocs = voc_bboxes.get(frame_order, [])
            if not vocs:
                raise ManualReviewError(f"Auto-refine frame has no strict XML BBox: {video_name}/{frame_order}")
            image = images[frame_order]
            shape = tuple(int(value) for value in image_to_uint8_gray(image).shape)
            local_bboxes = [
                xml_bbox_to_local(
                    voc.xml_xyxy,
                    image_size_wh=voc.image_size_wh,
                    attrs=pseudo_attrs,
                    local_shape_hw=shape,
                )
                for voc in vocs
            ]
            unknown = sorted(set(decisions_by_bbox) - set(range(len(local_bboxes))))
            if unknown:
                raise ManualReviewError(f"Auto-refine BBox indices not found: {unknown}")
            global_binary = build_full_frame_binary_mask(
                image,
                texture_config=teacher.texture_config,
                min_alpha=teacher.min_alpha,
            )
            local_binary = build_ranked_local_binary_mask(
                image, config=teacher.ranked_config
            )
            baseline_masks: list[np.ndarray] = []
            target_masks: list[np.ndarray] = []
            baseline_results: list[dict[str, Any]] = []
            for bbox_index, bbox in enumerate(local_bboxes):
                baseline = build_bbox_ranked_contour_mask(
                    global_binary=global_binary,
                    local_binary=local_binary,
                    bbox=bbox,
                    min_contour_area=teacher.min_contour_area,
                    config=teacher.ranked_config,
                )
                baseline_mask = np.asarray(baseline["mask"], dtype=bool)
                baseline_results.append(baseline)
                baseline_masks.append(baseline_mask)
                decision = decisions_by_bbox.get(bbox_index)
                if decision is None:
                    target_masks.append(baseline_mask)
                    continue
                expected_baseline_sha = str(decision["baseline_mask_sha256"])
                baseline_valid = str(decision.get("baseline_valid", "")).lower() in {
                    "true",
                    "1",
                }
                if expected_baseline_sha:
                    if mask_sha256(baseline_mask) != expected_baseline_sha:
                        raise ManualReviewError(
                            "Phase 3 baseline mask checksum is stale: "
                            f"{video_name}/{frame_order}/{bbox_index}"
                        )
                elif baseline_valid or bool(baseline.get("valid", False)):
                    raise ManualReviewError(
                        "Phase 3 omitted the checksum of a valid baseline mask"
                    )
                proposal = _read_proposal(
                    Path(phase3_root) / str(decision["proposal_mask_path"]),
                    shape,
                    str(decision["proposal_mask_sha256"]),
                )
                if not np.any(proposal):
                    raise ManualReviewError("auto_refine proposal must not be empty")
                target_masks.append(proposal)

            baseline_union = compose_initial_frame_mask(
                shape_hw=shape,
                bboxes=local_bboxes,
                bbox_masks=baseline_masks,
            )
            rebuilt_baseline, _ = apply_corrected_frame_mask(
                point_labels=original_labels,
                frame_orders=frame_orders,
                pixel_xy=pixel_xy,
                target_frame_order=frame_order,
                corrected_mask=baseline_union,
                bboxes=local_bboxes,
                manual_bbox_indices=[],
                require_nonempty_manual_bbox=False,
            )
            frame_selector = frame_orders == frame_order
            if not np.array_equal(
                rebuilt_baseline[frame_selector], original_labels[frame_selector]
            ):
                mismatch = int(
                    np.sum(
                        rebuilt_baseline[frame_selector]
                        != original_labels[frame_selector]
                    )
                )
                raise ManualReviewError(
                    f"Rebuilt v2 baseline differs from source labels: "
                    f"{video_name}/frame={frame_order}, points={mismatch}"
                )

            target_union = compose_initial_frame_mask(
                shape_hw=shape,
                bboxes=local_bboxes,
                bbox_masks=target_masks,
            )
            labels, stats = apply_corrected_frame_mask(
                point_labels=labels,
                frame_orders=frame_orders,
                pixel_xy=pixel_xy,
                target_frame_order=frame_order,
                corrected_mask=target_union,
                bboxes=local_bboxes,
                manual_bbox_indices=[],
                require_nonempty_manual_bbox=False,
            )
            frame_row_indices = np.flatnonzero(
                (frame_annotation["frame_order"].astype(np.int64) == frame_order)
                & (frame_annotation["frame_index"].astype(np.int64) == frame_index)
            )
            if frame_row_indices.size != len(local_bboxes):
                raise ManualReviewError(
                    f"Frame annotation/strict XML BBox count mismatch: {video_name}/{frame_order}"
                )
            for row_index in frame_row_indices:
                union_updates[int(row_index)] = int(stats["positive_points"])

            frame_xy = pixel_xy[frame_selector]
            for bbox_index, decision in sorted(decisions_by_bbox.items()):
                row_index = _frame_row_index(
                    frame_annotation,
                    frame_order=frame_order,
                    frame_index=frame_index,
                    bbox_index=bbox_index,
                )
                bbox = local_bboxes[bbox_index]
                saved_bbox = np.asarray(
                    frame_annotation["bbox_local_xyxy"][row_index], dtype=np.float64
                )
                if not np.allclose(saved_bbox, bbox.local_xyxy, rtol=0.0, atol=1e-4):
                    raise ManualReviewError("Source frame_annotation BBox differs from strict XML")
                proposal = target_masks[bbox_index]
                metrics = mask_shape_metrics(proposal, bbox)
                after_positive = _point_count(proposal, frame_xy)
                before_positive = int(
                    frame_annotation["num_labeled_points"][row_index]
                )
                if not math.isclose(
                    float(metrics["area_ratio"]),
                    float(decision["proposal_area_ratio"]),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                ):
                    raise ManualReviewError("Phase 3 proposal area metric is stale")
                frame_updates[row_index] = {
                    "valid_contour": True,
                    "contour_area": _contour_area(proposal),
                    "binary_area": int(proposal.sum()),
                    "foreground_ratio_in_bbox": float(metrics["area_ratio"]),
                    "num_labeled_points": after_positive,
                    "contour_positive_points": after_positive,
                    "fallback_used": False,
                    "fallback_percentile": math.nan,
                    "fallback_attempts": 0,
                    "annotation_reason": "phase3_auto_refine",
                    "selected_contour_source": f"phase3_{decision['proposal_source']}",
                    "contour_selection_score": float(decision["proposal_score"]),
                    "center_distance_norm": float(metrics["center_distance_norm"]),
                    "selected_area_ratio": float(metrics["area_ratio"]),
                }
                refinement_rows.append(
                    {
                        "video_name": video_name,
                        "frame_order": frame_order,
                        "frame_index": frame_index,
                        "bbox_index": bbox_index,
                        "reason_codes": decision["reason_codes"],
                        "proposal_source": decision["proposal_source"],
                        "baseline_mask_sha256": decision["baseline_mask_sha256"],
                        "proposal_mask_sha256": decision["proposal_mask_sha256"],
                        "before_positive_points": before_positive,
                        "after_positive_points": after_positive,
                        "score_gain": float(decision["score_gain"]),
                        "score_margin": float(decision["score_margin"]),
                        "proposal_score": float(decision["proposal_score"]),
                        "proposal_area_ratio": float(decision["proposal_area_ratio"]),
                        "proposal_center_distance_norm": float(
                            decision["proposal_center_distance_norm"]
                        ),
                        "proposal_stability_iou": float(
                            decision["proposal_stability_iou"]
                        ),
                    }
                )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_h5.name}.", suffix=".tmp", dir=output_h5.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source_h5, temporary)
        with h5py.File(temporary, "r+") as handle:
            annotation = handle["annotation"]
            annotation["point_label"][...] = labels.astype(np.int8)
            annotation["valid_mask"][...] = (labels != -1)
            frame_group = handle["frame_annotation"]
            text_updates: dict[str, list[str]] = {}
            for name in ("annotation_reason", "selected_contour_source"):
                text_updates[name] = [_decode_text(value) for value in frame_group[name][:]]
            for row_index, updates in frame_updates.items():
                for name, value in updates.items():
                    if name in text_updates:
                        text_updates[name][row_index] = str(value)
                    elif name in frame_group:
                        frame_group[name][row_index] = value
            if "num_labeled_points_union" in frame_group:
                for row_index, value in union_updates.items():
                    frame_group["num_labeled_points_union"][row_index] = value
            for name, values in text_updates.items():
                _replace_text_dataset(frame_group, name, values)
            _set_provenance(
                handle,
                source_h5=source_h5,
                teacher_fingerprint=teacher.fingerprint,
                refine_fingerprint=refine_fingerprint,
                phase3_root=phase3_root,
                decision_sha256=decision_sha256,
                refined_rows=refinement_rows,
            )
            handle.flush()
        if output_h5.exists():
            output_h5.unlink()
        os.replace(temporary, output_h5)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "video_name": video_name,
        "source_h5": str(source_h5),
        "output_h5": str(output_h5),
        "output_h5_sha256": file_sha256(output_h5),
        "points": int(labels.size),
        "positive_before": int(np.sum(original_labels == LABEL_FEMUR_CANDIDATE)),
        "positive_after": int(np.sum(labels == LABEL_FEMUR_CANDIDATE)),
        "added_positive_points": int(np.sum((labels == 1) & (original_labels != 1))),
        "removed_positive_points": int(np.sum((labels != 1) & (original_labels == 1))),
        "refined_frames": len(refined_by_frame),
        "refined_bboxes": len(refinement_rows),
        "status": "processed",
    }


def run_phase5(args: argparse.Namespace) -> dict[str, Any]:
    teacher = load_audit_teacher_config(args.teacher_config)
    refine = load_contour_refinement_config(args.refine_config)
    if refine.base_teacher_config_name != teacher.config_name:
        raise ManualReviewError("Phase 5 refine/teacher config_name mismatch")
    if refine.base_teacher_fingerprint not in {None, teacher.fingerprint}:
        raise ManualReviewError("Phase 5 refine/teacher fingerprint mismatch")
    refine.validate(require_production_thresholds=True)
    decisions, phase3_summary, _ = _phase3_contract(
        phase3_root=args.phase3_root,
        manifest=args.manifest,
        annotated_root=args.source_annotated_root,
        phase1_audit_root=args.phase1_audit_root,
        teacher_fingerprint=teacher.fingerprint,
        refine_fingerprint=refine.fingerprint,
    )
    if int(phase3_summary.get("failure_rows", -1)) != 0:
        raise ManualReviewError("Phase 3 production run contains failures")
    decision_counts = Counter(str(row["proposed_decision"]) for row in decisions)
    decision_sha256 = file_sha256(Path(args.phase3_root) / "bbox_decisions.csv")

    items = load_stage4_sweep_manifest(args.manifest)
    resolved = resolve_inputs(
        items,
        annotated_root=args.source_annotated_root,
        annotated_glob_template=args.annotated_glob_template,
        expected_videos=args.expected_videos,
    )
    decisions_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in decisions:
        decisions_by_video[str(row["video_name"])].append(row)
    unknown = sorted(set(decisions_by_video) - {item.item.video_name for item in resolved})
    if unknown:
        raise ManualReviewError(f"Phase 3 decisions reference unknown manifest videos: {unknown}")

    output_root = Path(args.output_root)
    if output_root.exists() and output_root.is_symlink():
        raise ManualReviewError("Phase 5 output_root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    input_paths = [
        Path(args.manifest),
        Path(args.teacher_config),
        Path(args.refine_config),
        Path(args.phase1_audit_root) / "audit_summary.json",
        Path(args.phase3_root) / "bbox_decisions.csv",
        Path(args.phase3_root) / "refine_summary.json",
        *(item.annotated_h5 for item in resolved),
    ]
    before = _snapshot(input_paths)

    print("Stage 4 contour-teacher Phase 5 full-dataset apply")
    print(f"manifest             : {args.manifest}")
    print(f"source_annotated_root: {args.source_annotated_root}")
    print(f"phase3_root          : {args.phase3_root}")
    print(f"output_root          : {output_root}")
    print(f"videos               : {len(resolved)}")
    print(f"decision_counts      : {dict(sorted(decision_counts.items()))}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, item in enumerate(resolved, start=1):
        video_name = item.item.video_name
        output_h5 = output_root / video_name / _output_name(item.annotated_h5)
        try:
            if output_h5.exists() and args.skip_existing and not args.overwrite:
                existing = _validate_existing_output(
                    output_h5,
                    source_h5=item.annotated_h5,
                    teacher_fingerprint=teacher.fingerprint,
                    refine_fingerprint=refine.fingerprint,
                    decision_sha256=decision_sha256,
                )
                row = {
                    "video_name": video_name,
                    "source_h5": str(item.annotated_h5),
                    "output_h5": str(output_h5),
                    "output_h5_sha256": file_sha256(output_h5),
                    "points": existing["points"],
                    "positive_before": "",
                    "positive_after": existing["positive"],
                    "added_positive_points": "",
                    "removed_positive_points": "",
                    "refined_frames": "",
                    "refined_bboxes": existing["refined"],
                    "status": "skipped",
                }
            else:
                row = _apply_one_video(
                    video_name=video_name,
                    pseudo3d_h5=item.item.pseudo3d_h5,
                    source_h5=item.annotated_h5,
                    output_h5=output_h5,
                    voc_xml_root=item.item.voc_xml_root,
                    decisions=decisions_by_video.get(video_name, []),
                    teacher=teacher,
                    refine_fingerprint=refine.fingerprint,
                    phase3_root=args.phase3_root,
                    decision_sha256=decision_sha256,
                    overwrite=args.overwrite,
                )
            rows.append(row)
            print(
                f"[{index}/{len(resolved)}] [OK] {video_name}: "
                f"refined_bboxes={row['refined_bboxes']}, status={row['status']}"
            )
        except Exception as exc:
            failure = {
                "video_name": video_name,
                "source_h5": str(item.annotated_h5),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[{index}/{len(resolved)}] [FAIL] {video_name}: {failure['error']}")
            if not args.continue_on_error:
                break

    _assert_snapshot_unchanged(before)
    write_csv_rows(output_root / "summary.csv", rows)
    write_csv_rows(
        output_root / "failures.csv",
        failures,
        fieldnames=("video_name", "source_h5", "error"),
    )
    run_config = {
        "schema_version": PHASE5_SCHEMA_VERSION,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "source_annotated_root": str(Path(args.source_annotated_root).resolve()),
        "phase1_audit_root": str(Path(args.phase1_audit_root).resolve()),
        "phase3_root": str(Path(args.phase3_root).resolve()),
        "phase3_bbox_decisions_sha256": decision_sha256,
        "teacher_config": str(Path(args.teacher_config).resolve()),
        "teacher_fingerprint": teacher.fingerprint,
        "refine_config": str(Path(args.refine_config).resolve()),
        "refine_fingerprint": refine.fingerprint,
        "policy": "auto_refine_only; auto_accept/manual_review/unselected preserve v2",
        "expected_videos": int(args.expected_videos),
    }
    with (output_root / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, allow_unicode=True, sort_keys=True)
    processed = [row for row in rows if row["status"] == "processed"]
    summary = {
        "schema_version": PHASE5_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failures else "ok",
        "videos_expected": len(resolved),
        "videos_written_or_verified": len(rows),
        "videos_processed": len(processed),
        "videos_skipped": len(rows) - len(processed),
        "phase3_decision_counts": dict(sorted(decision_counts.items())),
        "refined_bboxes": sum(int(row["refined_bboxes"]) for row in rows),
        "refined_frames": sum(
            int(row["refined_frames"]) for row in processed
        ),
        "added_positive_points": sum(
            int(row["added_positive_points"]) for row in processed
        ),
        "removed_positive_points": sum(
            int(row["removed_positive_points"]) for row in processed
        ),
        "failure_rows": len(failures),
        "input_files_unchanged": True,
        "output_root": str(output_root.resolve()),
    }
    write_json(output_root / "phase5_summary.json", summary)
    print("\nPhase 5 summary")
    for key in (
        "videos_written_or_verified",
        "refined_bboxes",
        "refined_frames",
        "added_positive_points",
        "removed_positive_points",
        "failure_rows",
    ):
        print(f"  {key:27s}: {summary[key]}")
    if failures:
        raise RuntimeError(f"Phase 5 failed for {len(failures)} video(s)")
    if len(rows) != len(resolved):
        raise RuntimeError("Phase 5 did not produce or verify every manifest video")
    if int(summary["refined_bboxes"]) != int(decision_counts.get("auto_refine", 0)):
        raise RuntimeError("Phase 5 refined BBox count differs from Phase 3 decisions")
    print("Stage 4 contour-teacher Phase 5 full-dataset apply passed.")
    return {"summary": summary, "rows": rows, "failures": failures}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply fixed Phase 3 auto_refine decisions to all Stage 4 teacher v2 "
            "H5 files while preserving auto_accept/manual_review/unselected BBoxes."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--phase1_audit_root", type=Path, required=True)
    parser.add_argument("--phase3_root", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--refine_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, default=182)
    parser.add_argument("--annotated_glob_template", default=DEFAULT_ANNOTATED_GLOB)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.skip_existing and args.overwrite:
        raise SystemExit("--skip_existing and --overwrite are mutually exclusive")
    try:
        run_phase5(args)
    except Exception as exc:
        raise SystemExit(f"Phase 5 apply failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
