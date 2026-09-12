from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_contour_teacher import (
    _assert_metadata_matches_teacher,
    _preflight_annotated_h5,
    _read_annotated_h5,
    _read_pseudo3d,
    load_audit_teacher_config,
    preflight_input,
    resolve_inputs,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import load_stage4_sweep_manifest
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    build_bbox_ranked_contour_mask,
    build_full_frame_binary_mask,
    build_ranked_local_binary_mask,
    load_voc_bboxes,
    xml_bbox_to_local,
)
from pseudo3d.annotation.contour_teacher_refinement import (
    load_contour_refinement_config,
    mask_sha256,
)
from pseudo3d.annotation.stage4_manual_review import (
    MANUAL_REVIEW_SCHEMA_VERSION,
    ManualReviewError,
    bbox_bounds,
    bbox_key,
    clip_mask_to_bbox,
    compose_initial_frame_mask,
    file_sha256,
    frame_stem,
    load_manual_review_config,
    prepare_output_root,
    read_csv_rows,
    write_csv_rows,
    write_json,
    write_png,
)
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    convert_masks_to_cvat_zip,
)
from src.utils.alpha_texture_processing import image_to_uint8_gray


DEFAULT_ANNOTATED_GLOB = "{video_name}/*bboxrank_v2_nobbox_bg.h5"
DEFAULT_SOURCE_ANNOTATED_GLOB = DEFAULT_ANNOTATED_GLOB
ALL_PHASE3_DECISIONS = ("auto_accept", "auto_refine", "manual_review")
REQUIRED_PHASE3_FILES = {
    "bbox_decisions.csv",
    "candidate_metrics.csv",
    "failures.csv",
    "run_config.yaml",
    "refine_summary.json",
    "checksums.csv",
}
CVAT_MANUAL_BBOX_COLOR_BGR = (0, 255, 255)
CVAT_OTHER_BBOX_COLOR_BGR = (255, 255, 0)
CVAT_CLIPPED_BBOX_COLOR_BGR = (255, 0, 255)
CVAT_OUTSIDE_BBOX_COLOR_BGR = (0, 0, 255)
FULL_VIDEO_REVIEW_DISPOSITIONS = (
    "pending",
    "annotate",
    "exclude_crop_boundary",
    "needs_recrop",
)


def _full_video_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "include_all_video_frames", False))


def _review_decisions(args: argparse.Namespace) -> tuple[str, ...]:
    scope = str(getattr(args, "review_scope", "manual_review"))
    if scope == "manual_review":
        return ("manual_review",)
    if scope == "all_phase3":
        return ALL_PHASE3_DECISIONS
    raise ManualReviewError(f"Unknown review_scope: {scope!r}")


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ManualReviewError(f"{name} must be boolean: {value!r}")


def _float_attr(attrs: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(attrs.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _project_bbox_to_local_unclipped(
    *,
    raw_xyxy: Sequence[float],
    attrs: Mapping[str, Any],
    local_shape_hw: tuple[int, int],
) -> np.ndarray:
    """Reproduce raw-to-local projection without clipping to the local image."""
    local_h, local_w = (int(value) for value in local_shape_hw)
    projected = np.asarray(raw_xyxy, dtype=np.float64).copy()
    if projected.shape != (4,) or not np.all(np.isfinite(projected)):
        raise ManualReviewError(f"BBox raw coordinates must be finite xyxy: {raw_xyxy}")
    preprocess = str(attrs.get("local_preprocess", ""))
    if preprocess == "resize":
        raw_w = _float_attr(attrs, "raw_width", np.nan)
        raw_h = _float_attr(attrs, "raw_height", np.nan)
        if not np.isfinite(raw_w) or not np.isfinite(raw_h) or raw_w <= 0 or raw_h <= 0:
            raise ManualReviewError(
                "raw_width/raw_height attrs are required for resize BBox projection"
            )
        projected[[0, 2]] *= float(local_w) / raw_w
        projected[[1, 3]] *= float(local_h) / raw_h
    else:
        scale = _float_attr(attrs, "local_resize_scale", 1.0)
        if not np.isfinite(scale):
            scale = 1.0
        crop_left = _float_attr(attrs, "local_crop_left", 0.0)
        crop_top = _float_attr(attrs, "local_crop_top", 0.0)
        if not np.isfinite(crop_left) or not np.isfinite(crop_top):
            raise ManualReviewError("local crop offsets must be finite")
        projected[[0, 2]] = projected[[0, 2]] * scale - crop_left
        projected[[1, 3]] = projected[[1, 3]] * scale - crop_top
    if not np.all(np.isfinite(projected)):
        raise ManualReviewError("Projected local BBox contains non-finite coordinates")
    return projected.astype(np.float32)


def _bbox_crop_metrics(
    *,
    projected_xyxy: Sequence[float],
    local_shape_hw: tuple[int, int],
) -> dict[str, Any]:
    """Describe how an unclipped local BBox intersects the local crop."""
    height, width = (int(value) for value in local_shape_hw)
    if height <= 0 or width <= 0:
        raise ManualReviewError(f"Invalid local image shape: {local_shape_hw}")
    values = np.asarray(projected_xyxy, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ManualReviewError(f"Projected BBox must be finite xyxy: {projected_xyxy}")
    x1, y1, x2, y2 = (float(value) for value in values)
    projected_width = x2 - x1
    projected_height = y2 - y1
    if projected_width <= 0 or projected_height <= 0:
        raise ManualReviewError(f"Projected BBox has non-positive area: {values.tolist()}")
    projected_area = projected_width * projected_height
    clipped_x1 = max(0.0, min(float(width), x1))
    clipped_y1 = max(0.0, min(float(height), y1))
    clipped_x2 = max(0.0, min(float(width), x2))
    clipped_y2 = max(0.0, min(float(height), y2))
    visible_width = max(0.0, clipped_x2 - clipped_x1)
    visible_height = max(0.0, clipped_y2 - clipped_y1)
    visible_area = visible_width * visible_height
    visible_fraction = min(1.0, max(0.0, visible_area / projected_area))
    fully_outside = visible_area <= 0.0
    fully_visible = bool(
        not fully_outside
        and x1 >= 0.0
        and y1 >= 0.0
        and x2 <= float(width)
        and y2 <= float(height)
    )
    partially_clipped = bool(not fully_outside and not fully_visible)
    if fully_outside:
        crop_status = "fully_outside_crop"
    elif partially_clipped:
        crop_status = "partially_clipped"
    else:
        crop_status = "fully_visible"
    return {
        "projected_bbox_xyxy": tuple(float(value) for value in values),
        "clipped_bbox_xyxy": (
            float(clipped_x1),
            float(clipped_y1),
            float(clipped_x2),
            float(clipped_y2),
        ),
        "projected_area": float(projected_area),
        "visible_intersection_area": float(visible_area),
        "visible_fraction": float(visible_fraction),
        "touches_left": bool(x1 <= 0.0),
        "touches_right": bool(x2 >= float(width)),
        "touches_top": bool(y1 <= 0.0),
        "touches_bottom": bool(y2 >= float(height)),
        "fully_outside_crop": fully_outside,
        "partially_clipped": partially_clipped,
        "fully_visible": fully_visible,
        "crop_status": crop_status,
    }


def _pipe_floats(values: Sequence[float]) -> str:
    return "|".join(str(float(value)) for value in values)


def _phase3_contract(
    *,
    phase3_root: Path,
    manifest: Path,
    annotated_root: Path,
    phase1_audit_root: Path,
    teacher_fingerprint: str,
    refine_fingerprint: str,
) -> tuple[list[dict[str, str]], Mapping[str, Any], Mapping[str, Any]]:
    phase3_root = Path(phase3_root).resolve()
    missing = sorted(name for name in REQUIRED_PHASE3_FILES if not (phase3_root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Phase 3 root is incomplete: missing={missing}")
    summary = json.loads((phase3_root / "refine_summary.json").read_text(encoding="utf-8"))
    with (phase3_root / "run_config.yaml").open(encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle)
    if not isinstance(summary, Mapping) or not isinstance(run_config, Mapping):
        raise ManualReviewError("Phase 3 summary/run_config must be mappings")
    if str(summary.get("status", "")) != "ok" or int(summary.get("failure_rows", -1)) != 0:
        raise ManualReviewError("Phase 3 run is not complete and failure-free")
    if _bool(summary.get("candidate_generation_only", True), "summary candidate_generation_only"):
        raise ManualReviewError("Phase 4 requires a Phase 3 production decision run")
    expected_paths = {
        "manifest": Path(manifest).resolve(),
        "annotated_root": Path(annotated_root).resolve(),
        "phase1_audit_root": Path(phase1_audit_root).resolve(),
    }
    for key, expected in expected_paths.items():
        if Path(str(run_config.get(key, ""))).resolve() != expected:
            raise ManualReviewError(f"Phase 3 {key} differs from Phase 4 input")
    if str(run_config.get("manifest_sha256", "")) != file_sha256(manifest):
        raise ManualReviewError("Phase 3 manifest checksum is stale")
    if str(run_config.get("teacher_fingerprint", "")) != teacher_fingerprint:
        raise ManualReviewError("Phase 3 teacher fingerprint mismatch")
    if str(run_config.get("refine_fingerprint", "")) != refine_fingerprint:
        raise ManualReviewError("Phase 3 refine fingerprint mismatch")
    if _bool(run_config.get("candidate_generation_only", True), "run_config candidate_generation_only"):
        raise ManualReviewError("Phase 3 run_config is candidate-only")
    phase1_summary = Path(phase1_audit_root) / "audit_summary.json"
    if not phase1_summary.is_file():
        raise FileNotFoundError(f"Phase 1 audit summary not found: {phase1_summary}")
    if str(run_config.get("phase1_audit_summary_sha256", "")) != file_sha256(phase1_summary):
        raise ManualReviewError("Phase 3 Phase 1 audit checksum is stale")

    checksum_rows = read_csv_rows(phase3_root / "checksums.csv")
    seen: set[str] = set()
    for row in checksum_rows:
        relative = str(row.get("relative_path", ""))
        if not relative or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ManualReviewError(f"Invalid/duplicate Phase 3 checksum path: {relative!r}")
        seen.add(relative)
        path = phase3_root / relative
        if not path.is_file() or file_sha256(path) != str(row.get("sha256", "")):
            raise ManualReviewError(f"Phase 3 checksum mismatch: {relative}")

    decisions = read_csv_rows(phase3_root / "bbox_decisions.csv")
    expected_count = int(summary.get("processed_bboxes", -1))
    if len(decisions) != expected_count:
        raise ManualReviewError("Phase 3 decision row count differs from summary")
    keys: set[tuple[str, int, int, int]] = set()
    for row in decisions:
        key = (
            str(row["video_name"]),
            int(row["frame_order"]),
            int(row["frame_index"]),
            int(row["bbox_index"]),
        )
        if key in keys:
            raise ManualReviewError(f"Duplicate Phase 3 decision key: {key}")
        keys.add(key)
        if str(row.get("proposed_decision", "")) not in {
            "auto_accept", "auto_refine", "manual_review"
        }:
            raise ManualReviewError(f"Unknown Phase 3 decision: {row.get('proposed_decision')}")
        proposal = phase3_root / str(row.get("proposal_mask_path", ""))
        if not proposal.is_file():
            raise FileNotFoundError(f"Phase 3 proposal mask not found: {proposal}")
        if str(row.get("proposed_decision")) == "manual_review" and not str(
            row.get("reason_codes", "")
        ).strip():
            raise ManualReviewError(f"manual_review decision has no reason code: {key}")
    actual_counts = Counter(str(row["proposed_decision"]) for row in decisions)
    stored_counts = summary.get("decision_counts", {})
    if not isinstance(stored_counts, Mapping) or {
        str(key): int(value) for key, value in stored_counts.items()
    } != dict(sorted(actual_counts.items())):
        raise ManualReviewError("Phase 3 decision counts differ from summary")
    return decisions, summary, run_config


def _read_proposal(path: Path, expected_shape: tuple[int, int], expected_sha: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ManualReviewError(f"Proposal mask must be 2-D uint8: {path}")
    if mask.shape != expected_shape:
        raise ManualReviewError(f"Proposal mask shape mismatch: {path}: {mask.shape}")
    unexpected = sorted(set(int(value) for value in np.unique(mask)) - {0, 255})
    if unexpected:
        raise ManualReviewError(f"Proposal mask contains unexpected values: {unexpected}")
    result = mask > 0
    if expected_sha and mask_sha256(result) != expected_sha:
        raise ManualReviewError(f"Proposal semantic mask checksum mismatch: {path}")
    return result


def _render_frame_overlay(
    *,
    image: np.ndarray,
    frame_mask: np.ndarray,
    bboxes: Sequence[Any],
    decisions: Sequence[str],
) -> np.ndarray:
    gray = image_to_uint8_gray(image)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tint = output.copy()
    tint[frame_mask] = (80, 180, 80)
    output = cv2.addWeighted(output, 0.65, tint, 0.35, 0.0)
    colors = {
        "manual_review": (0, 0, 255),
        "manual_review_unrenderable": (128, 128, 128),
        "review_unrenderable": (128, 128, 128),
        "auto_refine": (0, 165, 255),
        "auto_accept": (0, 255, 0),
        "phase3_unselected": (255, 255, 0),
    }
    for index, (bbox, decision) in enumerate(zip(bboxes, decisions, strict=True)):
        bounds = bbox_bounds(bbox, frame_mask.shape)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        color = colors[decision]
        cv2.rectangle(output, (left, top), (right, bottom), color, 1)
        cv2.putText(
            output,
            f"b{index}:{decision}",
            (left, max(10, top - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _render_cvat_review_image(
    *,
    image: np.ndarray,
    bboxes: Sequence[Any],
    decisions: Sequence[str],
    review_required: Sequence[bool] | None = None,
) -> np.ndarray:
    """Render strict BBox boundaries directly into the image uploaded to CVAT."""
    if len(bboxes) != len(decisions):
        raise ManualReviewError("CVAT review image BBox/decision length mismatch")
    if review_required is None:
        review_required = [decision == "manual_review" for decision in decisions]
    if len(review_required) != len(bboxes):
        raise ManualReviewError("CVAT review image BBox/review flag length mismatch")
    gray = image_to_uint8_gray(image)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    entries = list(zip(bboxes, decisions, review_required, strict=True))
    # Draw context BBoxes first so every review-target boundary remains visible.
    entries.sort(key=lambda entry: bool(entry[2]))
    for bbox, _decision, is_review_target in entries:
        bounds = bbox_bounds(bbox, gray.shape)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        color = (
            CVAT_MANUAL_BBOX_COLOR_BGR
            if is_review_target
            else CVAT_OTHER_BBOX_COLOR_BGR
        )
        cv2.rectangle(
            output,
            (left, top),
            (right, bottom),
            color,
            1,
            lineType=cv2.LINE_8,
        )
    return output


def _draw_outside_edge_marker(
    image: np.ndarray,
    projected_xyxy: Sequence[float],
    color: tuple[int, int, int],
) -> None:
    """Draw short inward ticks indicating a target outside the local crop."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in projected_xyxy)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    tick_length = max(3, min(12, int(round(min(height, width) * 0.08))))
    marker_x = int(np.clip(round(center_x), 0, width - 1))
    marker_y = int(np.clip(round(center_y), 0, height - 1))
    if center_x < 0:
        cv2.line(image, (0, marker_y), (tick_length, marker_y), color, 2, cv2.LINE_8)
    elif center_x > width:
        cv2.line(
            image,
            (width - 1 - tick_length, marker_y),
            (width - 1, marker_y),
            color,
            2,
            cv2.LINE_8,
        )
    if center_y < 0:
        cv2.line(image, (marker_x, 0), (marker_x, tick_length), color, 2, cv2.LINE_8)
    elif center_y > height:
        cv2.line(
            image,
            (marker_x, height - 1 - tick_length),
            (marker_x, height - 1),
            color,
            2,
            cv2.LINE_8,
        )


def _render_full_video_review_image(
    *,
    image: np.ndarray,
    bboxes: Sequence[Any],
    crop_metrics: Sequence[Mapping[str, Any]],
    review_required: Sequence[bool],
    frame_order: int,
    frame_index: int,
    has_review_target: bool,
) -> np.ndarray:
    """Render a coordinate-preserving, text-free image for full-video CVAT review."""
    if not (len(bboxes) == len(crop_metrics) == len(review_required)):
        raise ManualReviewError("Full-video BBox/metric/review flag length mismatch")
    # These values remain in review_frames.csv, but are intentionally not burned
    # into pixels because text can hide the anatomy being segmented.
    _ = (frame_order, frame_index, has_review_target)
    gray = image_to_uint8_gray(image)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    entries = [
        (bbox_index, bbox, metrics, is_target)
        for bbox_index, (bbox, metrics, is_target) in enumerate(
            zip(bboxes, crop_metrics, review_required, strict=True)
        )
    ]
    entries.sort(key=lambda entry: bool(entry[3]))
    for bbox_index, bbox, metrics, is_target in entries:
        crop_status = str(metrics["crop_status"])
        if not is_target:
            color = CVAT_OTHER_BBOX_COLOR_BGR
        elif crop_status == "fully_visible":
            color = CVAT_MANUAL_BBOX_COLOR_BGR
        elif crop_status == "partially_clipped":
            color = CVAT_CLIPPED_BBOX_COLOR_BGR
        else:
            color = CVAT_OUTSIDE_BBOX_COLOR_BGR
        clipped = tuple(float(value) for value in metrics["clipped_bbox_xyxy"])
        bounds = bbox_bounds(clipped, gray.shape)
        if float(metrics["visible_intersection_area"]) > 0.0 and bounds is not None:
            left, top, right, bottom = bounds
            cv2.rectangle(output, (left, top), (right, bottom), color, 1, cv2.LINE_8)
        elif is_target:
            _draw_outside_edge_marker(
                output,
                metrics["projected_bbox_xyxy"],
                CVAT_OUTSIDE_BBOX_COLOR_BGR,
            )
    return output


def _checksum_rows(root: Path) -> list[dict[str, Any]]:
    excluded = {"checksums.csv", "export_summary.json"}
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {"relative_path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return rows


def run_review_export(args: argparse.Namespace) -> dict[str, Any]:
    teacher = load_audit_teacher_config(args.teacher_config)
    refine = load_contour_refinement_config(args.refine_config)
    refine.validate(require_production_thresholds=True)
    manual = load_manual_review_config(args.manual_review_config)
    review_decisions = _review_decisions(args)
    full_video_mode = _full_video_mode(args)
    if full_video_mode:
        if str(getattr(args, "review_scope", "manual_review")) != "all_phase3":
            raise ManualReviewError("Full-video export requires --review_scope all_phase3")
        if not bool(getattr(args, "partition_by_video", False)):
            raise ManualReviewError("Full-video export requires --partition_by_video")
    if refine.base_teacher_config_name != teacher.config_name:
        raise ManualReviewError("refine/teacher config_name mismatch")
    if refine.base_teacher_fingerprint is not None and refine.base_teacher_fingerprint != teacher.fingerprint:
        raise ManualReviewError("refine/teacher fingerprint mismatch")
    decisions, phase3_summary, phase3_run = _phase3_contract(
        phase3_root=args.phase3_root,
        manifest=args.manifest,
        annotated_root=args.annotated_root,
        phase1_audit_root=args.phase1_audit_root,
        teacher_fingerprint=teacher.fingerprint,
        refine_fingerprint=refine.fingerprint,
    )
    review_rows = [row for row in decisions if row["proposed_decision"] in review_decisions]
    expected_review_bboxes = int(getattr(args, "expected_review_bboxes", 0))
    if expected_review_bboxes > 0 and len(review_rows) != expected_review_bboxes:
        raise ManualReviewError(
            "Review BBox count mismatch: "
            f"expected={expected_review_bboxes}, actual={len(review_rows)}"
        )
    review_decision_counts = Counter(str(row["proposed_decision"]) for row in review_rows)
    selected_video_names = sorted({str(row["video_name"]) for row in review_rows})
    expected_selected_videos = int(getattr(args, "expected_selected_videos", 0))
    if expected_selected_videos > 0 and len(selected_video_names) != expected_selected_videos:
        raise ManualReviewError(
            "Selected review video count mismatch: "
            f"expected={expected_selected_videos}, actual={len(selected_video_names)}"
        )
    for decision in ALL_PHASE3_DECISIONS:
        expected = int(getattr(args, f"expected_{decision}", -1))
        actual = int(review_decision_counts.get(decision, 0))
        if expected >= 0 and actual != expected:
            raise ManualReviewError(
                f"Review decision count mismatch for {decision}: "
                f"expected={expected}, actual={actual}"
            )
    prepare_output_root(args.output_root, overwrite=args.overwrite)
    if not review_rows:
        summary = {
            "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
            "status": "no_manual_review",
            "review_frames": 0,
            "manual_review_bboxes": 0,
            "teacher_fingerprint": teacher.fingerprint,
            "refine_fingerprint": refine.fingerprint,
            "manual_review_fingerprint": manual.fingerprint,
            "export_mode": "full_video" if full_video_mode else "selected_frames",
            "review_scope": str(getattr(args, "review_scope", "manual_review")),
            "review_decisions": list(review_decisions),
            "failure_rows": 0,
        }
        write_json(Path(args.output_root) / "export_summary.json", summary)
        print("Stage 4 Phase 4 review export: no manual-review BBoxes")
        return {"summary": summary, "frame_rows": [], "bbox_rows": []}

    items = load_stage4_sweep_manifest(args.manifest)
    resolved = resolve_inputs(
        items,
        annotated_root=args.annotated_root,
        annotated_glob_template=args.annotated_glob_template,
        expected_videos=args.expected_videos,
    )
    resolved_map = {item.item.video_name: item for item in resolved}
    source_annotated_root = Path(
        getattr(args, "source_annotated_root", None) or args.annotated_root
    )
    source_annotated_glob_template = str(
        getattr(args, "source_annotated_glob_template", None)
        or args.annotated_glob_template
    )
    source_resolved = resolve_inputs(
        items,
        annotated_root=source_annotated_root,
        annotated_glob_template=source_annotated_glob_template,
        expected_videos=args.expected_videos,
    )
    source_resolved_map = {item.item.video_name: item for item in source_resolved}
    decision_map: dict[tuple[str, int, int, int], dict[str, str]] = {}
    decisions_by_frame: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in decisions:
        key = (row["video_name"], int(row["frame_order"]), int(row["frame_index"]), int(row["bbox_index"]))
        decision_map[key] = row
        decisions_by_frame[key[:3]].append(row)
    review_frames = sorted(
        {
            (row["video_name"], int(row["frame_order"]), int(row["frame_index"]))
            for row in review_rows
        }
    )
    unknown = sorted({key[0] for key in review_frames} - set(resolved_map))
    if unknown:
        raise ManualReviewError(f"Phase 3 decisions reference unknown videos: {unknown}")
    for video_name in sorted({key[0] for key in review_frames}):
        preflight_input(resolved_map[video_name], teacher)
        _, source_attrs, _ = _preflight_annotated_h5(
            source_resolved_map[video_name].annotated_h5
        )
        _assert_metadata_matches_teacher(source_attrs, teacher)

    images_dir = Path(args.output_root) / "images"
    masks_dir = Path(args.output_root) / "masks"
    overlays_dir = Path(args.output_root) / "overlays"
    frame_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    unreviewable_rows: list[dict[str, Any]] = []
    phase3_root = Path(args.phase3_root)
    frames_by_video: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for key in review_frames:
        frames_by_video[key[0]].append(key)

    for video_name in sorted(frames_by_video):
        item = resolved_map[video_name]
        source_item = source_resolved_map[video_name]
        images, frame_indices, pseudo_attrs = _read_pseudo3d(item.item.pseudo3d_h5)
        _, _, annotated_attrs = _read_annotated_h5(item.annotated_h5)
        _assert_metadata_matches_teacher(annotated_attrs, teacher)
        voc_bboxes = load_voc_bboxes(
            item.item.voc_xml_root,
            video_name=video_name,
            frame_indices=frame_indices,
            xml_frame_number_offsets=teacher.xml_frame_number_offsets,
            xml_frame_id_source=teacher.xml_frame_id_source,
            xml_annotation_dir_name=teacher.xml_annotation_dir_name,
            strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
        )
        selected_frame_keys = {
            (int(frame_order), int(frame_index))
            for _, frame_order, frame_index in frames_by_video[video_name]
        }
        frames_to_export = (
            [
                (video_name, int(frame_order), int(frame_index))
                for frame_order, frame_index in enumerate(frame_indices.tolist())
            ]
            if full_video_mode
            else frames_by_video[video_name]
        )
        if full_video_mode and not selected_frame_keys:
            raise ManualReviewError(f"Full-video export selected no target frames: {video_name}")
        source_pseudo3d_h5_sha256 = file_sha256(item.item.pseudo3d_h5)
        source_annotated_h5_sha256 = file_sha256(source_item.annotated_h5)
        for _, frame_order, frame_index in frames_to_export:
            if frame_order >= len(frame_indices) or int(frame_indices[frame_order]) != frame_index:
                raise ManualReviewError(f"Phase 3 frame alignment mismatch: {video_name}/{frame_order}")
            vocs = voc_bboxes.get(frame_order, [])
            if not vocs and not full_video_mode:
                raise ManualReviewError(f"Manual-review frame has no strict XML BBox: {video_name}/{frame_order}")
            gray = image_to_uint8_gray(images[frame_order])
            shape = tuple(int(value) for value in gray.shape)
            has_review_target_frame = (
                int(frame_order), int(frame_index)
            ) in selected_frame_keys
            local_bboxes = [
                xml_bbox_to_local(
                    voc.xml_xyxy,
                    image_size_wh=voc.image_size_wh,
                    attrs=pseudo_attrs,
                    local_shape_hw=shape,
                )
                for voc in vocs
            ]
            needs_initial_masks = bool(vocs) and (
                not full_video_mode or has_review_target_frame
            )
            global_binary = (
                build_full_frame_binary_mask(
                    images[frame_order],
                    texture_config=teacher.texture_config,
                    min_alpha=teacher.min_alpha,
                )
                if needs_initial_masks
                else np.zeros(shape, dtype=bool)
            )
            local_binary = (
                build_ranked_local_binary_mask(
                    images[frame_order], config=teacher.ranked_config
                )
                if needs_initial_masks
                else np.zeros(shape, dtype=bool)
            )
            masks: list[np.ndarray] = []
            bbox_decisions: list[str] = []
            bbox_review_required: list[bool] = []
            bbox_crop_metrics: list[dict[str, Any]] = []
            stem = frame_stem(video_name, frame_order, frame_index)
            manual_indices: list[int] = []
            selected_target_indices: list[int] = []
            local_bbox_rows: list[dict[str, Any]] = []
            for bbox_index, (voc, bbox) in enumerate(zip(vocs, local_bboxes, strict=True)):
                projected_bbox = _project_bbox_to_local_unclipped(
                    raw_xyxy=bbox.raw_xyxy,
                    attrs=pseudo_attrs,
                    local_shape_hw=shape,
                )
                crop_metrics = _bbox_crop_metrics(
                    projected_xyxy=projected_bbox,
                    local_shape_hw=shape,
                )
                bbox_crop_metrics.append(crop_metrics)
                row = decision_map.get((video_name, frame_order, frame_index, bbox_index))
                phase3_decision = (
                    str(row["proposed_decision"]) if row else "phase3_unselected"
                )
                render_decision = phase3_decision
                bounds = bbox_bounds(bbox, shape)
                requested_manual_review = bool(
                    row and phase3_decision in review_decisions
                )
                if full_video_mode and not requested_manual_review:
                    baseline_mask = np.zeros(shape, dtype=bool)
                else:
                    baseline = build_bbox_ranked_contour_mask(
                        global_binary=global_binary,
                        local_binary=local_binary,
                        bbox=bbox,
                        min_contour_area=teacher.min_contour_area,
                        config=teacher.ranked_config,
                    )
                    baseline_mask = np.asarray(baseline["mask"], dtype=bool)
                if requested_manual_review:
                    selected_target_indices.append(bbox_index)
                if requested_manual_review and bounds is None:
                    render_decision = "review_unrenderable"
                    unreviewable_rows.append(
                        {
                            "bbox_key": bbox_key(
                                video_name, frame_order, frame_index, bbox_index
                            ),
                            "video_name": video_name,
                            "frame_order": frame_order,
                            "frame_index": frame_index,
                            "bbox_index": bbox_index,
                            "xml_path": str(voc.xml_path.resolve()),
                            "bbox_xml_xyxy": "|".join(
                                str(float(value)) for value in voc.xml_xyxy
                            ),
                            "bbox_local_xyxy": "|".join(
                                str(float(value)) for value in bbox.local_xyxy
                            ),
                            "projected_bbox_xyxy": _pipe_floats(
                                crop_metrics["projected_bbox_xyxy"]
                            ),
                            "crop_status": crop_metrics["crop_status"],
                            "visible_fraction": crop_metrics["visible_fraction"],
                            "phase3_reason_codes": row.get("reason_codes", ""),
                            "disposition": (
                                "unreviewable_outside_crop"
                                if crop_metrics["fully_outside_crop"]
                                else "unreviewable_invalid_bbox"
                            ),
                        }
                    )
                if full_video_mode and not requested_manual_review:
                    selected_source = "context_all_background"
                    chosen = np.zeros(shape, dtype=bool)
                else:
                    selected_source = "v2_baseline"
                    chosen = baseline_mask
                    if row and phase3_decision in {"auto_refine", "manual_review"}:
                        proposal_path = phase3_root / row["proposal_mask_path"]
                        chosen = _read_proposal(
                            proposal_path,
                            shape,
                            row.get("proposal_mask_sha256", ""),
                        )
                        selected_source = str(
                            row.get("proposal_source", "phase3_proposal")
                        )
                    chosen = clip_mask_to_bbox(chosen, bbox)
                if requested_manual_review and bounds is not None:
                    manual_indices.append(bbox_index)
                masks.append(chosen)
                bbox_decisions.append(render_decision)
                bbox_review_required.append(requested_manual_review and bounds is not None)
                local_bbox_rows.append(
                    {
                        "frame_stem": stem,
                        "bbox_key": bbox_key(video_name, frame_order, frame_index, bbox_index),
                        "video_name": video_name,
                        "frame_order": frame_order,
                        "frame_index": frame_index,
                        "bbox_index": bbox_index,
                        "object_index": int(voc.object_index),
                        "object_name": voc.object_name,
                        "xml_path": str(voc.xml_path.resolve()),
                        "xml_sha256": file_sha256(voc.xml_path),
                        "bbox_xml_xyxy": "|".join(str(float(value)) for value in voc.xml_xyxy),
                        "bbox_local_xyxy": "|".join(str(float(value)) for value in bbox.local_xyxy),
                        "bbox_bounds_ltrb": "|".join(str(value) for value in bounds) if bounds else "",
                        "projected_bbox_xyxy": _pipe_floats(
                            crop_metrics["projected_bbox_xyxy"]
                        ),
                        "clipped_bbox_xyxy": _pipe_floats(
                            crop_metrics["clipped_bbox_xyxy"]
                        ),
                        "projected_area": crop_metrics["projected_area"],
                        "visible_intersection_area": crop_metrics[
                            "visible_intersection_area"
                        ],
                        "visible_fraction": crop_metrics["visible_fraction"],
                        "touches_left": crop_metrics["touches_left"],
                        "touches_right": crop_metrics["touches_right"],
                        "touches_top": crop_metrics["touches_top"],
                        "touches_bottom": crop_metrics["touches_bottom"],
                        "fully_outside_crop": crop_metrics["fully_outside_crop"],
                        "partially_clipped": crop_metrics["partially_clipped"],
                        "fully_visible": crop_metrics["fully_visible"],
                        "crop_status": crop_metrics["crop_status"],
                        "review_target": requested_manual_review,
                        "review_disposition": (
                            "pending" if requested_manual_review else ""
                        ),
                        "phase3_decision": phase3_decision,
                        "phase3_reason_codes": row.get("reason_codes", "") if row else "phase3_unselected",
                        "initial_mask_source": selected_source,
                        "initial_mask_sha256": mask_sha256(chosen),
                        "initial_bbox_positive_pixels": int(np.sum(chosen)),
                        "manual_review_required": requested_manual_review and bounds is not None,
                        "manual_review_disposition": (
                            "actionable"
                            if requested_manual_review and bounds is not None
                            else (
                                "unreviewable_invalid_bbox"
                                if requested_manual_review
                                else "not_requested"
                            )
                        ),
                        "proposal_score": row.get("proposal_score", "") if row else "",
                        "proposal_area_ratio": row.get("proposal_area_ratio", "") if row else "",
                        "proposal_center_distance_norm": row.get("proposal_center_distance_norm", "") if row else "",
                        "proposal_border_contact_ratio": row.get("proposal_border_contact_ratio", "") if row else "",
                        "proposal_stability_iou": row.get("proposal_stability_iou", "") if row else "",
                    }
                )
            bbox_rows.extend(local_bbox_rows)
            frame_mask = compose_initial_frame_mask(
                shape_hw=shape, bboxes=local_bboxes, bbox_masks=masks
            )
            image_path = images_dir / f"{stem}.png"
            mask_path = masks_dir / f"{stem}.png"
            overlay_path = overlays_dir / f"{stem}.png"
            review_image = (
                _render_full_video_review_image(
                    image=gray,
                    bboxes=local_bboxes,
                    crop_metrics=bbox_crop_metrics,
                    review_required=[
                        index in selected_target_indices
                        for index in range(len(local_bboxes))
                    ],
                    frame_order=frame_order,
                    frame_index=frame_index,
                    has_review_target=bool(selected_target_indices),
                )
                if full_video_mode
                else _render_cvat_review_image(
                    image=gray,
                    bboxes=local_bboxes,
                    decisions=bbox_decisions,
                    review_required=bbox_review_required,
                )
            )
            write_png(
                image_path,
                review_image,
            )
            write_png(mask_path, frame_mask.astype(np.uint8) * 255)
            write_png(
                overlay_path,
                _render_frame_overlay(
                    image=images[frame_order],
                    frame_mask=frame_mask,
                    bboxes=local_bboxes,
                    decisions=bbox_decisions,
                ),
            )
            frame_rows.append(
                {
                    "frame_stem": stem,
                    "video_name": video_name,
                    "frame_order": frame_order,
                    "frame_index": frame_index,
                    "height": shape[0],
                    "width": shape[1],
                    "image_path": image_path.relative_to(args.output_root).as_posix(),
                    "mask_path": mask_path.relative_to(args.output_root).as_posix(),
                    "overlay_path": overlay_path.relative_to(args.output_root).as_posix(),
                    "image_sha256": file_sha256(image_path),
                    "mask_sha256": file_sha256(mask_path),
                    "overlay_sha256": file_sha256(overlay_path),
                    "source_pseudo3d_h5": str(item.item.pseudo3d_h5.resolve()),
                    "source_pseudo3d_h5_sha256": source_pseudo3d_h5_sha256,
                    "source_annotated_h5": str(source_item.annotated_h5.resolve()),
                    "source_annotated_h5_sha256": source_annotated_h5_sha256,
                    "voc_xml_root": str(item.item.voc_xml_root.resolve()),
                    "bbox_count": len(vocs),
                    "frame_role": (
                        "review_target" if selected_target_indices else "context_only"
                    ),
                    "review_target": bool(selected_target_indices),
                    "selected_review_bbox_count": len(selected_target_indices),
                    "selected_bbox_indices": "|".join(
                        str(value) for value in selected_target_indices
                    ),
                    "manual_review_bbox_count": len(manual_indices),
                    "manual_bbox_indices": "|".join(str(value) for value in manual_indices),
                    "annotation_required": bool(manual_indices),
                    "context_only": not bool(manual_indices),
                    "initial_mask_positive_pixels": int(frame_mask.sum()),
                    "review_image_has_bbox_overlay": True,
                    "review_image_has_text_overlay": False,
                    "review_image_render_version": "bbox_only_textfree_v3",
                    "manual_bbox_color_rgb": "255|255|0",
                    "other_bbox_color_rgb": "0|255|255",
                    "clipped_bbox_color_rgb": "255|0|255",
                    "outside_bbox_color_rgb": "255|0|0",
                    "teacher_fingerprint": teacher.fingerprint,
                    "refine_fingerprint": refine.fingerprint,
                    "manual_review_fingerprint": manual.fingerprint,
                }
            )

    frame_rows.sort(key=lambda row: row["frame_stem"])
    bbox_rows.sort(key=lambda row: row["bbox_key"])
    unreviewable_rows.sort(key=lambda row: row["bbox_key"])
    target_bbox_rows = [
        row
        for row in bbox_rows
        if _bool(row.get("review_target", False), "review_target")
    ]
    if len(target_bbox_rows) != len(review_rows):
        raise ManualReviewError(
            "Exported review-target BBox count mismatch: "
            f"expected={len(review_rows)}, actual={len(target_bbox_rows)}"
        )
    target_bbox_keys = [str(row["bbox_key"]) for row in target_bbox_rows]
    if len(set(target_bbox_keys)) != len(target_bbox_keys):
        raise ManualReviewError("Exported review-target BBox keys are not unique")
    crop_decision_rows = [
        {
            "bbox_key": row["bbox_key"],
            "video_name": row["video_name"],
            "frame_order": row["frame_order"],
            "frame_index": row["frame_index"],
            "bbox_index": row["bbox_index"],
            "phase3_decision": row["phase3_decision"],
            "crop_status": row["crop_status"],
            "visible_fraction": row["visible_fraction"],
            "review_disposition": "pending",
            "reviewer_notes": "",
        }
        for row in target_bbox_rows
    ]
    write_csv_rows(
        Path(args.output_root) / "unreviewable_bboxes.csv",
        unreviewable_rows,
        fieldnames=(
            "bbox_key",
            "video_name",
            "frame_order",
            "frame_index",
            "bbox_index",
            "xml_path",
            "bbox_xml_xyxy",
            "bbox_local_xyxy",
            "projected_bbox_xyxy",
            "crop_status",
            "visible_fraction",
            "phase3_reason_codes",
            "disposition",
        ),
    )
    write_csv_rows(Path(args.output_root) / "review_frames.csv", frame_rows)
    write_csv_rows(Path(args.output_root) / "review_bboxes.csv", bbox_rows)
    if full_video_mode:
        write_csv_rows(
            Path(args.output_root) / "crop_review_decisions.csv",
            crop_decision_rows,
            fieldnames=(
                "bbox_key",
                "video_name",
                "frame_order",
                "frame_index",
                "bbox_index",
                "phase3_decision",
                "crop_status",
                "visible_fraction",
                "review_disposition",
                "reviewer_notes",
            ),
        )
    context_only_stems = [
        str(row["frame_stem"]) for row in frame_rows if bool(row["context_only"])
    ]
    (Path(args.output_root) / "context_only_stems.txt").write_text(
        "".join(f"{stem}\n" for stem in context_only_stems),
        encoding="utf-8",
    )
    session = {
        "review_schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
        "review_package_sha256": "FILL_FROM_export_summary_review_package_sha256",
        "cvat_version": "",
        "cvat_task_identifier": "",
        "reviewer_identifier": "",
        "imported_zip_sha256": "FILL_FROM_export_summary_cvat_import_zip_sha256",
        "pre_import_backup_sha256": "",
        "exported_zip_sha256": "",
        "review_started_utc": "",
        "review_completed_utc": "",
        "all_manual_bboxes_reviewed": False,
        "notes": "",
    }
    with (Path(args.output_root) / "review_session_template.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(session, handle, allow_unicode=True, sort_keys=True)
    run_config = {
        "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "annotated_root": str(Path(args.annotated_root).resolve()),
        "source_annotated_root": str(source_annotated_root.resolve()),
        "source_annotated_glob_template": source_annotated_glob_template,
        "phase1_audit_root": str(Path(args.phase1_audit_root).resolve()),
        "phase3_root": str(Path(args.phase3_root).resolve()),
        "phase3_refine_summary_sha256": file_sha256(Path(args.phase3_root) / "refine_summary.json"),
        "teacher_config": str(Path(args.teacher_config).resolve()),
        "teacher_fingerprint": teacher.fingerprint,
        "refine_config": str(Path(args.refine_config).resolve()),
        "refine_fingerprint": refine.fingerprint,
        "manual_review_config": str(Path(args.manual_review_config).resolve()),
        "manual_review_fingerprint": manual.fingerprint,
        "export_mode": "full_video" if full_video_mode else "selected_frames",
        "include_all_video_frames": full_video_mode,
        "selected_videos": len(selected_video_names),
        "full_video_review_dispositions": list(FULL_VIDEO_REVIEW_DISPOSITIONS),
        "review_scope": str(getattr(args, "review_scope", "manual_review")),
        "review_decisions": list(review_decisions),
    }
    with (Path(args.output_root) / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, allow_unicode=True, sort_keys=True)
    cvat_dir = Path(args.output_root) / "cvat"
    cvat_summary = convert_masks_to_cvat_zip(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_zip=cvat_dir / "annotations_segmentation_mask_1_1.zip",
        label_name=manual.label_name,
        summary_json=cvat_dir / "conversion_summary.json",
        keep_unpacked_dir=cvat_dir / "unpacked_reference",
    )
    checksum_rows = _checksum_rows(Path(args.output_root))
    write_csv_rows(Path(args.output_root) / "checksums.csv", checksum_rows)
    package_hash = hashlib_sha256_rows(checksum_rows)
    requested_bbox_rows = [
        row
        for row in bbox_rows
        if _bool(row["manual_review_required"], "manual_review_required")
    ]
    requested_decision_counts = dict(
        sorted(review_decision_counts.items())
    )
    summary = {
        "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
        "status": "ok",
        "export_mode": "full_video" if full_video_mode else "selected_frames",
        "selected_videos": len(selected_video_names),
        "review_frames": len(frame_rows),
        "target_review_frames": sum(
            _bool(row["review_target"], "review_target") for row in frame_rows
        ),
        "actionable_review_frames": sum(
            _bool(row["annotation_required"], "annotation_required")
            for row in frame_rows
        ),
        "context_only_frames": len(context_only_stems),
        "bbox_rows": len(bbox_rows),
        "selected_review_bboxes": len(review_rows),
        "crop_review_decision_rows": (
            len(crop_decision_rows) if full_video_mode else 0
        ),
        "crop_status_counts": dict(
            sorted(Counter(str(row["crop_status"]) for row in target_bbox_rows).items())
        ),
        "actionable_review_bboxes": len(requested_bbox_rows),
        "manual_review_bboxes": len(requested_bbox_rows),
        "phase3_decision_counts": requested_decision_counts,
        "unreviewable_manual_bboxes": len(unreviewable_rows),
        "review_images_have_bbox_overlay": True,
        "review_images_have_text_overlay": False,
        "review_image_render_version": "bbox_only_textfree_v3",
        "manual_bbox_color_rgb": "255|255|0",
        "other_bbox_color_rgb": "0|255|255",
        "clipped_bbox_color_rgb": "255|0|255",
        "outside_bbox_color_rgb": "255|0|0",
        "teacher_fingerprint": teacher.fingerprint,
        "refine_fingerprint": refine.fingerprint,
        "manual_review_fingerprint": manual.fingerprint,
        "review_scope": str(getattr(args, "review_scope", "manual_review")),
        "review_decisions": list(review_decisions),
        "cvat_import_zip_sha256": cvat_summary.zip_sha256,
        "review_package_sha256": package_hash,
        "checksum_rows": len(checksum_rows),
        "failure_rows": 0,
    }
    write_json(Path(args.output_root) / "export_summary.json", summary)
    partition_result = None
    if bool(getattr(args, "partition_by_video", False)):
        partition_result = _partition_review_packages(
            output_root=Path(args.output_root),
            frame_rows=frame_rows,
            bbox_rows=bbox_rows,
            unreviewable_rows=unreviewable_rows,
            root_summary=summary,
        )
        summary["partition_by_video"] = True
        summary["video_packages"] = partition_result["video_packages"]
        summary["case_frames"] = partition_result["case_frames"]
        summary["review_index"] = partition_result["review_index"]
        summary["progress_csv"] = partition_result["progress_csv"]
        write_json(Path(args.output_root) / "export_summary.json", summary)
    print("Stage 4 Phase 4 manual-review export passed.")
    print(f"  export_mode         : {summary['export_mode']}")
    print(f"  selected_videos     : {summary['selected_videos']}")
    print(f"  review_frames       : {len(frame_rows)}")
    print(f"  target_frames       : {summary['target_review_frames']}")
    print(f"  actionable_frames   : {summary['actionable_review_frames']}")
    print(f"  context_only_frames : {summary['context_only_frames']}")
    print(f"  selected_bboxes     : {summary['selected_review_bboxes']}")
    print(f"  manual_review_bboxes: {summary['manual_review_bboxes']}")
    print(f"  unreviewable_bboxes : {summary['unreviewable_manual_bboxes']}")
    if partition_result is not None:
        print(f"  video_packages      : {partition_result['video_packages']}")
        print(f"  case_frames         : {partition_result['case_frames']}")
    print(f"  output_root         : {args.output_root}")
    return {
        "summary": summary,
        "frame_rows": frame_rows,
        "bbox_rows": bbox_rows,
        "unreviewable_rows": unreviewable_rows,
        "partition_result": partition_result,
    }


def hashlib_sha256_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _partition_review_packages(
    *,
    output_root: Path,
    frame_rows: Sequence[Mapping[str, Any]],
    bbox_rows: Sequence[Mapping[str, Any]],
    unreviewable_rows: Sequence[Mapping[str, Any]],
    root_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Create independent, importable CVAT packages and case folders per video."""
    output_root = Path(output_root)
    videos_root = output_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=False)
    frames_by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    bboxes_by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    bboxes_by_stem: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unreviewable_by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_video[str(row["video_name"])].append(row)
    for row in bbox_rows:
        bboxes_by_video[str(row["video_name"])].append(row)
        bboxes_by_stem[str(row["frame_stem"])].append(row)
    for row in unreviewable_rows:
        unreviewable_by_video[str(row["video_name"])].append(row)

    root_run_config_path = output_root / "run_config.yaml"
    with root_run_config_path.open(encoding="utf-8") as handle:
        root_run_config = yaml.safe_load(handle)
    if not isinstance(root_run_config, Mapping):
        raise ManualReviewError("Root review run_config must be a mapping")

    index_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    for video_name in sorted(frames_by_video):
        video_root = videos_root / video_name
        video_root.mkdir(parents=True, exist_ok=False)
        video_frames = sorted(
            frames_by_video[video_name], key=lambda row: str(row["frame_stem"])
        )
        video_bboxes = sorted(
            bboxes_by_video[video_name], key=lambda row: str(row["bbox_key"])
        )
        video_unreviewable = sorted(
            unreviewable_by_video.get(video_name, []),
            key=lambda row: str(row["bbox_key"]),
        )
        for frame_row in video_frames:
            stem = str(frame_row["frame_stem"])
            source_files: dict[str, tuple[str, Path]] = {}
            for key, case_name in (
                ("image_path", "image_with_bbox.png"),
                ("mask_path", "initial_mask.png"),
                ("overlay_path", "overlay.png"),
            ):
                relative = Path(str(frame_row[key]))
                source = output_root / relative
                destination = video_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                source_files[key] = (case_name, source)
            case_bboxes = sorted(
                bboxes_by_stem[stem], key=lambda row: int(row["bbox_index"])
            )
            selected = [
                row
                for row in case_bboxes
                if str(row["phase3_decision"]) in root_summary["review_decisions"]
            ]
            if selected:
                case_root = video_root / "cases" / stem
                case_root.mkdir(parents=True, exist_ok=False)
                copied: dict[str, str] = {}
                for key, (case_name, source) in source_files.items():
                    case_destination = case_root / case_name
                    shutil.copy2(source, case_destination)
                    copied[key] = case_destination.relative_to(
                        video_root
                    ).as_posix()
                case_payload = {
                    "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
                    "case_id": stem,
                    "video_name": video_name,
                    "frame_order": int(frame_row["frame_order"]),
                    "frame_index": int(frame_row["frame_index"]),
                    "frame_role": frame_row.get("frame_role", "review_target"),
                    "context_only": _bool(
                        frame_row["context_only"], "context_only"
                    ),
                    "annotation_required": _bool(
                        frame_row["annotation_required"], "annotation_required"
                    ),
                    "case_files": copied,
                    "selected_bbox_keys": [str(row["bbox_key"]) for row in selected],
                    "phase3_decisions": [
                        str(row["phase3_decision"]) for row in selected
                    ],
                    "bboxes": [dict(row) for row in case_bboxes],
                }
                with (case_root / "case_metadata.yaml").open(
                    "w", encoding="utf-8", newline="\n"
                ) as handle:
                    yaml.safe_dump(
                        case_payload, handle, allow_unicode=True, sort_keys=True
                    )
            for row in selected:
                actionable = _bool(
                    row["manual_review_required"], "manual_review_required"
                )
                index_rows.append(
                    {
                        "bbox_key": row["bbox_key"],
                        "video_name": video_name,
                        "frame_stem": stem,
                        "frame_order": frame_row["frame_order"],
                        "frame_index": frame_row["frame_index"],
                        "bbox_index": row["bbox_index"],
                        "phase3_decision": row["phase3_decision"],
                        "phase3_reason_codes": row["phase3_reason_codes"],
                        "review_disposition": row["manual_review_disposition"],
                        "crop_status": row.get("crop_status", ""),
                        "visible_fraction": row.get("visible_fraction", ""),
                        "video_package": (Path("videos") / video_name).as_posix(),
                        "case_directory": (
                            Path("videos") / video_name / "cases" / stem
                        ).as_posix(),
                        "review_status": "pending",
                        "correction_status": (
                            "pending" if actionable else "not_applicable"
                        ),
                    }
                )

        write_csv_rows(video_root / "review_frames.csv", video_frames)
        write_csv_rows(video_root / "review_bboxes.csv", video_bboxes)
        selected_rows = [
            row
            for row in video_bboxes
            if _bool(row.get("review_target", False), "review_target")
        ]
        if str(root_summary.get("export_mode", "selected_frames")) == "full_video":
            write_csv_rows(
                video_root / "crop_review_decisions.csv",
                [
                    {
                        "bbox_key": row["bbox_key"],
                        "video_name": row["video_name"],
                        "frame_order": row["frame_order"],
                        "frame_index": row["frame_index"],
                        "bbox_index": row["bbox_index"],
                        "phase3_decision": row["phase3_decision"],
                        "crop_status": row["crop_status"],
                        "visible_fraction": row["visible_fraction"],
                        "review_disposition": "pending",
                        "reviewer_notes": "",
                    }
                    for row in selected_rows
                ],
                fieldnames=(
                    "bbox_key",
                    "video_name",
                    "frame_order",
                    "frame_index",
                    "bbox_index",
                    "phase3_decision",
                    "crop_status",
                    "visible_fraction",
                    "review_disposition",
                    "reviewer_notes",
                ),
            )
        write_csv_rows(
            video_root / "unreviewable_bboxes.csv",
            video_unreviewable,
            fieldnames=(
                "bbox_key",
                "video_name",
                "frame_order",
                "frame_index",
                "bbox_index",
                "xml_path",
                "bbox_xml_xyxy",
                "bbox_local_xyxy",
                "projected_bbox_xyxy",
                "crop_status",
                "visible_fraction",
                "phase3_reason_codes",
                "disposition",
            ),
        )
        context_stems = [
            str(row["frame_stem"])
            for row in video_frames
            if _bool(row["context_only"], "context_only")
        ]
        (video_root / "context_only_stems.txt").write_text(
            "".join(f"{stem}\n" for stem in context_stems), encoding="utf-8"
        )
        shutil.copy2(
            output_root / "review_session_template.yaml",
            video_root / "review_session_template.yaml",
        )
        video_run_config = dict(root_run_config)
        video_run_config["partition_video_name"] = video_name
        video_run_config["partition_parent_root"] = str(output_root.resolve())
        with (video_root / "run_config.yaml").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            yaml.safe_dump(video_run_config, handle, allow_unicode=True, sort_keys=True)
        cvat_summary = convert_masks_to_cvat_zip(
            images_dir=video_root / "images",
            masks_dir=video_root / "masks",
            output_zip=video_root / "cvat" / "annotations_segmentation_mask_1_1.zip",
            label_name="femur",
            summary_json=video_root / "cvat" / "conversion_summary.json",
            keep_unpacked_dir=video_root / "cvat" / "unpacked_reference",
        )
        checksum_rows = _checksum_rows(video_root)
        write_csv_rows(video_root / "checksums.csv", checksum_rows)
        package_hash = hashlib_sha256_rows(checksum_rows)
        actionable_rows = [
            row
            for row in selected_rows
            if _bool(row["manual_review_required"], "manual_review_required")
        ]
        decision_counts = dict(
            sorted(Counter(str(row["phase3_decision"]) for row in selected_rows).items())
        )
        video_summary = {
            "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
            "status": "ok",
            "export_mode": root_summary.get("export_mode", "selected_frames"),
            "partition_video_name": video_name,
            "review_frames": len(video_frames),
            "target_review_frames": sum(
                _bool(row.get("review_target", False), "review_target")
                for row in video_frames
            ),
            "actionable_review_frames": sum(
                _bool(row["annotation_required"], "annotation_required")
                for row in video_frames
            ),
            "context_only_frames": len(context_stems),
            "bbox_rows": len(video_bboxes),
            "selected_review_bboxes": len(selected_rows),
            "crop_review_decision_rows": (
                len(selected_rows)
                if str(root_summary.get("export_mode", "selected_frames"))
                == "full_video"
                else 0
            ),
            "crop_status_counts": dict(
                sorted(
                    Counter(
                        str(row.get("crop_status", "")) for row in selected_rows
                    ).items()
                )
            ),
            "actionable_review_bboxes": len(actionable_rows),
            "manual_review_bboxes": len(actionable_rows),
            "unreviewable_manual_bboxes": len(video_unreviewable),
            "phase3_decision_counts": decision_counts,
            "review_scope": root_summary["review_scope"],
            "review_decisions": root_summary["review_decisions"],
            "review_images_have_bbox_overlay": True,
            "manual_bbox_color_rgb": "255|255|0",
            "other_bbox_color_rgb": "0|255|255",
            "clipped_bbox_color_rgb": "255|0|255",
            "outside_bbox_color_rgb": "255|0|0",
            "teacher_fingerprint": root_summary["teacher_fingerprint"],
            "refine_fingerprint": root_summary["refine_fingerprint"],
            "manual_review_fingerprint": root_summary["manual_review_fingerprint"],
            "cvat_import_zip_sha256": cvat_summary.zip_sha256,
            "review_package_sha256": package_hash,
            "checksum_rows": len(checksum_rows),
            "failure_rows": 0,
        }
        write_json(video_root / "export_summary.json", video_summary)
        progress_rows.append(
            {
                "video_name": video_name,
                "export_mode": root_summary.get("export_mode", "selected_frames"),
                "review_frames": len(video_frames),
                "target_review_frames": sum(
                    _bool(row.get("review_target", False), "review_target")
                    for row in video_frames
                ),
                "review_bboxes": len(selected_rows),
                "actionable_bboxes": len(actionable_rows),
                "unreviewable_bboxes": len(video_unreviewable),
                "auto_accept": decision_counts.get("auto_accept", 0),
                "auto_refine": decision_counts.get("auto_refine", 0),
                "manual_review": decision_counts.get("manual_review", 0),
                "context_only_frames": len(context_stems),
                "package_root": (Path("videos") / video_name).as_posix(),
                "cvat_import_zip": (
                    Path("videos")
                    / video_name
                    / "cvat"
                    / "annotations_segmentation_mask_1_1.zip"
                ).as_posix(),
                "review_status": "pending",
                "correction_status": (
                    "pending" if actionable_rows else "not_applicable"
                ),
            }
        )

    index_rows.sort(key=lambda row: str(row["bbox_key"]))
    progress_rows.sort(key=lambda row: str(row["video_name"]))
    write_csv_rows(output_root / "review_index.csv", index_rows)
    progress_filename = (
        "video_progress.csv"
        if str(root_summary.get("export_mode", "selected_frames")) == "full_video"
        else "progress.csv"
    )
    write_csv_rows(output_root / progress_filename, progress_rows)
    result = {
        "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
        "status": "ok",
        "video_packages": len(progress_rows),
        "exported_frames": len(frame_rows),
        "case_frames": len({str(row["frame_stem"]) for row in index_rows}),
        "review_bboxes": len(index_rows),
        "review_index": "review_index.csv",
        "progress_csv": progress_filename,
        "videos_root": "videos",
        "failure_rows": 0,
    }
    write_json(output_root / "partition_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Phase 3 manual-review frames for a local CVAT Segmentation Mask 1.1 task."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument(
        "--source_annotated_root",
        type=Path,
        default=None,
        help=(
            "H5 root recorded as the correction/import source. Defaults to "
            "--annotated_root; Phase 5 should point this at refined-auto v3."
        ),
    )
    parser.add_argument("--phase1_audit_root", type=Path, required=True)
    parser.add_argument("--phase3_root", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--refine_config", type=Path, required=True)
    parser.add_argument("--manual_review_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--annotated_glob_template", default=DEFAULT_ANNOTATED_GLOB)
    parser.add_argument(
        "--source_annotated_glob_template",
        default=None,
        help="Glob template for --source_annotated_root.",
    )
    parser.add_argument(
        "--review_scope",
        choices=("manual_review", "all_phase3"),
        default="manual_review",
        help="Export only manual_review decisions or all Phase 3 decision rows.",
    )
    parser.add_argument(
        "--partition_by_video",
        action="store_true",
        help="Also create independent importable CVAT packages under videos/<video_name>.",
    )
    parser.add_argument(
        "--include_all_video_frames",
        action="store_true",
        help=(
            "Export every local-crop frame from each selected video. Requires "
            "--review_scope all_phase3 and --partition_by_video."
        ),
    )
    parser.add_argument("--expected_review_bboxes", type=int, default=0)
    parser.add_argument("--expected_selected_videos", type=int, default=0)
    parser.add_argument("--expected_auto_accept", type=int, default=-1)
    parser.add_argument("--expected_auto_refine", type=int, default=-1)
    parser.add_argument("--expected_manual_review", type=int, default=-1)
    parser.add_argument("--expected_videos", type=int, default=182)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_videos < 0:
        raise SystemExit("--expected_videos must be >= 0")
    if args.expected_review_bboxes < 0:
        raise SystemExit("--expected_review_bboxes must be >= 0")
    if args.expected_selected_videos < 0:
        raise SystemExit("--expected_selected_videos must be >= 0")
    for name in ("expected_auto_accept", "expected_auto_refine", "expected_manual_review"):
        if int(getattr(args, name)) < -1:
            raise SystemExit(f"--{name} must be >= -1")
    try:
        run_review_export(args)
    except Exception as exc:
        raise SystemExit(f"Phase 4 review export failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
