from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_contour_teacher import (
    AuditTeacherConfig,
    ResolvedInput,
    _assert_metadata_matches_teacher,
    _assert_snapshot_unchanged,
    _read_annotated_h5,
    _read_pseudo3d,
    _snapshot,
    load_audit_teacher_config,
    preflight_input,
    resolve_inputs,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LocalBBox,
    build_bbox_ranked_contour_candidates,
    build_bbox_ranked_contour_mask,
    build_full_frame_binary_mask,
    build_ranked_local_binary_mask,
    load_voc_bboxes,
    xml_bbox_to_local,
)
from pseudo3d.annotation.contour_teacher_refinement import (
    ContourRefinementConfig,
    candidate_to_serializable,
    evaluate_bbox_refinement,
    load_contour_refinement_config,
    mask_sha256,
)
from src.utils.alpha_texture_processing import image_to_uint8_gray


PROTOTYPE_SCHEMA_VERSION = 1
DEFAULT_ANNOTATED_GLOB = "{video_name}/*bboxrank_v2_nobbox_bg.h5"
REQUIRED_PHASE1_FILES = {
    "bbox_audit.csv",
    "category_summary.csv",
    "audit_summary.json",
    "run_config.yaml",
}
KEY_FIELDS = ("video_name", "frame_order", "frame_index", "bbox_index")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["video_name"]),
        int(row["frame_order"]),
        int(row["frame_index"]),
        int(row["bbox_index"]),
    )


def _key_name(key: tuple[str, int, int, int]) -> str:
    video_name, frame_order, frame_index, bbox_index = key
    return (
        f"{video_name}__fo{frame_order:05d}__fi{frame_index:08d}"
        f"__bbox{bbox_index:03d}"
    )


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be a boolean: {value!r}")


def _float_equal(first: Any, second: Any, *, atol: float = 1e-5) -> bool:
    first_value = float(first)
    second_value = float(second)
    if math.isnan(first_value) and math.isnan(second_value):
        return True
    return math.isclose(first_value, second_value, rel_tol=0.0, abs_tol=atol)


def _load_phase1_contract(
    *,
    phase1_audit_root: Path,
    manifest: Path,
    annotated_root: Path,
    teacher: AuditTeacherConfig,
    top_per_category: int,
    selection_csv: Path | None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int, int], dict[str, str]], dict[str, Any]]:
    phase1_audit_root = Path(phase1_audit_root).resolve()
    missing = sorted(
        name for name in REQUIRED_PHASE1_FILES if not (phase1_audit_root / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Phase 1 audit root is incomplete: missing={missing}")
    summary = json.loads(
        (phase1_audit_root / "audit_summary.json").read_text(encoding="utf-8")
    )
    with (phase1_audit_root / "run_config.yaml").open(encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle)
    if not isinstance(summary, Mapping) or not isinstance(run_config, Mapping):
        raise ValueError("Phase 1 summary/run_config must be mappings")
    if int(summary.get("failure_rows", -1)) != 0:
        raise ValueError("Phase 1 audit has failures")
    if not _parse_bool(summary.get("input_files_unchanged", False), "input_files_unchanged"):
        raise ValueError("Phase 1 audit did not preserve its inputs")
    if Path(str(summary.get("manifest", ""))).resolve() != Path(manifest).resolve():
        raise ValueError("Phase 1 manifest path differs from Phase 3 manifest")
    if str(summary.get("manifest_sha256", "")) != file_sha256(manifest):
        raise ValueError("Phase 1 manifest checksum differs from current manifest")
    if Path(str(summary.get("annotated_root", ""))).resolve() != Path(annotated_root).resolve():
        raise ValueError("Phase 1 annotated_root differs from Phase 3 annotated_root")
    if str(summary.get("teacher_fingerprint", "")) != teacher.fingerprint:
        raise ValueError("Phase 1 teacher fingerprint differs from current teacher")
    if str(run_config.get("teacher_fingerprint", "")) != teacher.fingerprint:
        raise ValueError("Phase 1 run_config teacher fingerprint mismatch")

    audit_rows = _read_csv(phase1_audit_root / "bbox_audit.csv")
    audit_map: dict[tuple[str, int, int, int], dict[str, str]] = {}
    for row in audit_rows:
        key = _row_key(row)
        if key in audit_map:
            raise ValueError(f"Duplicate Phase 1 BBox key: {key}")
        audit_map[key] = row
    if len(audit_map) != int(summary.get("bbox_rows", -1)):
        raise ValueError("Phase 1 bbox_audit row count differs from summary")

    selected: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    if selection_csv is None:
        category_rows = _read_csv(phase1_audit_root / "category_summary.csv")
        counts: Counter[str] = Counter()
        for row in category_rows:
            category = str(row.get("category", "")).strip()
            if not category or counts[category] >= top_per_category:
                continue
            key = _row_key(row)
            if key not in audit_map:
                raise ValueError(f"Phase 1 category row has unknown BBox key: {key}")
            counts[category] += 1
            entry = selected.setdefault(
                key,
                {
                    **{name: value for name, value in zip(KEY_FIELDS, key, strict=True)},
                    "categories": set(),
                },
            )
            entry["categories"].add(category)
    else:
        for row in _read_csv(selection_csv):
            key = _row_key(row)
            if key not in audit_map:
                raise ValueError(f"selection_csv has unknown BBox key: {key}")
            category = str(row.get("category", "manual_selection")).strip() or "manual_selection"
            entry = selected.setdefault(
                key,
                {
                    **{name: value for name, value in zip(KEY_FIELDS, key, strict=True)},
                    "categories": set(),
                },
            )
            entry["categories"].add(category)
    if not selected:
        raise ValueError("Phase 3 selection is empty")
    if selection_csv is None and not any(
        "candidate_good_control" in value["categories"] for value in selected.values()
    ):
        raise ValueError("Phase 3 automatic selection must include candidate_good_control")
    rows = []
    for key in sorted(selected):
        item = dict(selected[key])
        item["categories"] = tuple(sorted(item["categories"]))
        rows.append(item)
    return rows, audit_map, dict(summary)


def _prepare_output_root(path: Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"output_root must be a plain directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"output_root is not empty; pass --overwrite: {path}"
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_binary_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".png",
        np.asarray(mask, dtype=np.uint8) * 255,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    if not success:
        raise OSError(f"Failed to encode mask PNG: {path}")
    path.write_bytes(encoded.tobytes())


def _draw_mask_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness)


def _render_overlay(
    *,
    path: Path,
    image: np.ndarray,
    bbox: LocalBBox,
    baseline_mask: np.ndarray,
    proposal_mask: np.ndarray,
    row: Mapping[str, Any],
) -> None:
    gray = image_to_uint8_gray(image)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    baseline_panel = base.copy()
    proposal_panel = base.copy()
    difference_panel = base.copy()
    _draw_mask_contour(baseline_panel, baseline_mask, (0, 255, 0), 2)
    _draw_mask_contour(proposal_panel, proposal_mask, (255, 0, 255), 2)
    added = np.asarray(proposal_mask, dtype=bool) & ~np.asarray(baseline_mask, dtype=bool)
    removed = np.asarray(baseline_mask, dtype=bool) & ~np.asarray(proposal_mask, dtype=bool)
    difference_panel[added] = (0, 128, 255)
    difference_panel[removed] = (255, 128, 0)
    left, top, right, bottom = (int(round(value)) for value in bbox.local_xyxy)
    for panel in (baseline_panel, proposal_panel, difference_panel):
        cv2.rectangle(panel, (left, top), (right, bottom), (0, 255, 255), 1)
    panels = [baseline_panel, proposal_panel, difference_panel]
    labels = ("v2 baseline", "Phase 3 proposal", "added=orange removed=blue")
    for panel, label in zip(panels, labels, strict=True):
        cv2.putText(
            panel,
            label,
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    canvas = np.concatenate(panels, axis=1)
    lines = [
        (
            f"{row['video_name']} fo={row['frame_order']} fi={row['frame_index']} "
            f"bbox={row['bbox_index']}"
        ),
        f"decision={row['proposed_decision']} categories={row['categories']}",
        f"reason={row['reason_codes']}",
        (
            f"candidate={row['num_candidates']} eligible={row['num_eligible_candidates']} "
            f"gain={float(row['score_gain']):.3f} margin={float(row['score_margin']):.3f}"
        ),
        (
            f"v2_area={float(row['baseline_area_ratio']):.3f} "
            f"proposal_area={float(row['proposal_area_ratio']):.3f}"
        ),
    ]
    panel_height = len(lines) * 18 + 6
    output = cv2.copyMakeBorder(
        canvas, panel_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (4, 16 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".png", output, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not success:
        raise OSError(f"Failed to encode overlay: {path}")
    path.write_bytes(encoded.tobytes())


def _validate_baseline_against_audit(
    audit_row: Mapping[str, str], result: Mapping[str, Any]
) -> None:
    exact = {
        "valid_contour": bool(result["valid"]),
        "annotation_reason": str(result["reason"]),
        "selected_contour_source": str(result["selected_contour_source"]),
    }
    for name, expected in exact.items():
        actual: Any = audit_row[name]
        if isinstance(expected, bool):
            actual = _parse_bool(actual, name)
        if actual != expected:
            raise ValueError(
                f"Phase 1/current baseline {name} mismatch: {actual!r} != {expected!r}"
            )
    numeric = {
        "contour_selection_score": result["contour_selection_score"],
        "selected_area_ratio": result["selected_area_ratio"],
        "foreground_ratio_in_bbox": result["foreground_ratio_in_bbox"],
        "center_distance_norm": result["center_distance_norm"],
    }
    for name, expected in numeric.items():
        if not _float_equal(audit_row[name], expected):
            raise ValueError(
                f"Phase 1/current baseline {name} mismatch: "
                f"{audit_row[name]!r} != {expected!r}"
            )


def _decision_row(
    *,
    key: tuple[str, int, int, int],
    categories: Sequence[str],
    audit_row: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    proposal_relative_path: str,
) -> dict[str, Any]:
    baseline = evaluation["baseline_candidate"]
    selected = evaluation["selected_candidate"]
    video_name, frame_order, frame_index, bbox_index = key
    return {
        "video_name": video_name,
        "frame_order": frame_order,
        "frame_index": frame_index,
        "bbox_index": bbox_index,
        "categories": "|".join(categories),
        "proposed_decision": evaluation["proposed_decision"],
        "reason_codes": "|".join(evaluation["reason_codes"]),
        "triggered": bool(evaluation["triggered"]),
        "trigger_reasons": "|".join(evaluation["trigger_reasons"]),
        "num_candidates": int(evaluation["num_candidates"]),
        "num_eligible_candidates": int(evaluation["num_eligible_candidates"]),
        "baseline_valid": _parse_bool(audit_row["valid_contour"], "valid_contour"),
        "baseline_source": str(audit_row["selected_contour_source"]),
        "baseline_mask_sha256": (
            str(baseline["mask_sha256"]) if baseline is not None else ""
        ),
        "baseline_score": float(baseline["score"]) if baseline is not None else math.nan,
        "baseline_area_ratio": (
            float(baseline["area_ratio"]) if baseline is not None else math.nan
        ),
        "baseline_border_contact_ratio": (
            float(baseline["border_contact_ratio"])
            if baseline is not None
            else math.nan
        ),
        "baseline_center_distance_norm": (
            float(baseline["center_distance_norm"])
            if baseline is not None
            else math.nan
        ),
        "proposal_source": (
            str(selected["primary_source"]) if selected is not None else "none"
        ),
        "proposal_mask_sha256": (
            str(selected["mask_sha256"]) if selected is not None else ""
        ),
        "proposal_score": float(selected["score"]) if selected is not None else math.nan,
        "proposal_area_ratio": (
            float(selected["area_ratio"]) if selected is not None else math.nan
        ),
        "proposal_border_contact_ratio": (
            float(selected["border_contact_ratio"])
            if selected is not None
            else math.nan
        ),
        "proposal_center_distance_norm": (
            float(selected["center_distance_norm"])
            if selected is not None
            else math.nan
        ),
        "proposal_stability_iou": (
            float(selected["stability_iou_max"]) if selected is not None else math.nan
        ),
        "proposal_support_count": (
            int(selected["support_count"]) if selected is not None else 0
        ),
        "score_gain": float(evaluation["score_gain"]),
        "score_margin": float(evaluation["score_margin"]),
        "baseline_score_margin": float(evaluation["baseline_score_margin"]),
        "proposal_mask_path": proposal_relative_path,
    }


def _candidate_rows(
    key: tuple[str, int, int, int],
    categories: Sequence[str],
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for rank, candidate in enumerate(evaluation["candidates"], start=1):
        row = {
            **{name: value for name, value in zip(KEY_FIELDS, key, strict=True)},
            "categories": "|".join(categories),
            "candidate_rank": rank,
            "proposed_decision": evaluation["proposed_decision"],
            "selected_proposal": bool(candidate is evaluation["selected_candidate"]),
            **candidate_to_serializable(candidate),
        }
        rows.append(row)
    return rows


def _video_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_name"])].append(row)
    result = []
    for video_name in sorted(grouped):
        values = grouped[video_name]
        decisions = Counter(str(row["proposed_decision"]) for row in values)
        result.append(
            {
                "video_name": video_name,
                "bbox_rows": len(values),
                "auto_accept": decisions["auto_accept"],
                "auto_refine": decisions["auto_refine"],
                "manual_review": decisions["manual_review"],
                "candidate_generation_only": decisions["candidate_generation_only"],
                "candidates": sum(int(row["num_candidates"]) for row in values),
                "eligible_candidates": sum(
                    int(row["num_eligible_candidates"]) for row in values
                ),
            }
        )
    return result


def _category_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for category in str(row["categories"]).split("|"):
            if category:
                grouped[category].append(row)
    result = []
    for category in sorted(grouped):
        values = grouped[category]
        decisions = Counter(str(row["proposed_decision"]) for row in values)
        result.append(
            {
                "category": category,
                "bbox_rows": len(values),
                "auto_accept": decisions["auto_accept"],
                "auto_refine": decisions["auto_refine"],
                "manual_review": decisions["manual_review"],
                "candidate_generation_only": decisions["candidate_generation_only"],
            }
        )
    return result


def _checksum_rows(output_root: Path) -> list[dict[str, Any]]:
    excluded = {"refine_summary.json", "checksums.csv"}
    rows = []
    for path in sorted(value for value in output_root.rglob("*") if value.is_file()):
        relative = path.relative_to(output_root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def run_prototype(args: argparse.Namespace) -> dict[str, Any]:
    teacher = load_audit_teacher_config(args.teacher_config)
    refine_config = load_contour_refinement_config(args.refine_config)
    if refine_config.base_teacher_config_name != teacher.config_name:
        raise ValueError(
            "refine config base_teacher_config_name differs from teacher config"
        )
    if (
        refine_config.base_teacher_fingerprint is not None
        and refine_config.base_teacher_fingerprint != teacher.fingerprint
    ):
        raise ValueError(
            "refine config optional base_teacher_fingerprint differs from teacher config"
        )
    refine_config.validate(
        require_production_thresholds=not bool(args.candidate_generation_only)
    )
    selections, audit_map, phase1_summary = _load_phase1_contract(
        phase1_audit_root=args.phase1_audit_root,
        manifest=args.manifest,
        annotated_root=args.annotated_root,
        teacher=teacher,
        top_per_category=args.top_per_category,
        selection_csv=args.selection_csv,
    )
    items = load_stage4_sweep_manifest(args.manifest)
    resolved_inputs = resolve_inputs(
        items,
        annotated_root=args.annotated_root,
        annotated_glob_template=args.annotated_glob_template,
        expected_videos=args.expected_videos,
    )
    resolved_map = {value.item.video_name: value for value in resolved_inputs}
    selected_videos = sorted({str(row["video_name"]) for row in selections})
    unknown_videos = sorted(set(selected_videos) - set(resolved_map))
    if unknown_videos:
        raise ValueError(f"Phase 3 selection contains unknown videos: {unknown_videos}")
    for video_name in selected_videos:
        preflight_input(resolved_map[video_name], teacher)

    print("Stage 4 contour-teacher Phase 3 auto-refine prototype (read-only)")
    print(f"manifest                  : {args.manifest}")
    print(f"annotated_root            : {args.annotated_root}")
    print(f"phase1_audit_root         : {args.phase1_audit_root}")
    print(f"teacher_fingerprint       : {teacher.fingerprint}")
    print(f"refine_fingerprint        : {refine_config.fingerprint}")
    print(f"candidate_generation_only : {bool(args.candidate_generation_only)}")
    print(f"selected_videos           : {len(selected_videos)}")
    print(f"selected_bboxes           : {len(selections)}")

    _prepare_output_root(args.output_root, args.overwrite)
    input_paths: list[Path] = [
        Path(args.manifest),
        Path(args.teacher_config),
        Path(args.refine_config),
        Path(args.phase1_audit_root) / "bbox_audit.csv",
        Path(args.phase1_audit_root) / "category_summary.csv",
        Path(args.phase1_audit_root) / "audit_summary.json",
        Path(args.phase1_audit_root) / "run_config.yaml",
    ]
    if args.selection_csv is not None:
        input_paths.append(Path(args.selection_csv))
    for video_name in selected_videos:
        resolved = resolved_map[video_name]
        input_paths.extend([resolved.item.pseudo3d_h5, resolved.annotated_h5])
        annotation_dir = (
            resolved.item.voc_xml_root
            / video_name
            / teacher.xml_annotation_dir_name
        )
        input_paths.extend(sorted(annotation_dir.glob("*.xml")))
    before = _snapshot(input_paths)

    selection_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        selection_by_video[str(row["video_name"])].append(row)
    decision_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for video_index, video_name in enumerate(selected_videos, start=1):
        resolved = resolved_map[video_name]
        try:
            images, frame_indices, pseudo_attrs = _read_pseudo3d(
                resolved.item.pseudo3d_h5
            )
            point_cloud, _, annotated_attrs = _read_annotated_h5(
                resolved.annotated_h5
            )
            _assert_metadata_matches_teacher(annotated_attrs, teacher)
            voc_bboxes = load_voc_bboxes(
                resolved.item.voc_xml_root,
                video_name=video_name,
                frame_indices=frame_indices,
                xml_frame_number_offsets=teacher.xml_frame_number_offsets,
                xml_frame_id_source=teacher.xml_frame_id_source,
                xml_annotation_dir_name=teacher.xml_annotation_dir_name,
                strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
            )
            shape = tuple(int(value) for value in image_to_uint8_gray(images[0]).shape)
            local_bboxes = {
                order: [
                    xml_bbox_to_local(
                        voc.xml_xyxy,
                        image_size_wh=voc.image_size_wh,
                        attrs=pseudo_attrs,
                        local_shape_hw=shape,
                    )
                    for voc in boxes
                ]
                for order, boxes in voc_bboxes.items()
            }
            frame_orders = point_cloud["frame_order"].astype(np.int64)
            pixel_xy = point_cloud["pixel_xy"]
            for selected_row in sorted(
                selection_by_video[video_name], key=lambda value: _row_key(value)
            ):
                key = _row_key(selected_row)
                _, frame_order, frame_index, bbox_index = key
                if frame_order < 0 or frame_order >= len(frame_indices):
                    raise ValueError(f"selection frame_order outside input: {key}")
                if int(frame_indices[frame_order]) != frame_index:
                    raise ValueError(f"selection frame_index mismatch: {key}")
                if frame_order not in local_bboxes or bbox_index >= len(local_bboxes[frame_order]):
                    raise ValueError(f"selection BBox index not found in strict XML: {key}")
                bbox = local_bboxes[frame_order][bbox_index]
                global_binary = build_full_frame_binary_mask(
                    images[frame_order],
                    texture_config=teacher.texture_config,
                    min_alpha=teacher.min_alpha,
                )
                local_binary = build_ranked_local_binary_mask(
                    images[frame_order], config=teacher.ranked_config
                )
                baseline_candidates = build_bbox_ranked_contour_candidates(
                    global_binary=global_binary,
                    local_binary=local_binary,
                    bbox=bbox,
                    min_contour_area=teacher.min_contour_area,
                    config=teacher.ranked_config,
                )
                baseline_result = build_bbox_ranked_contour_mask(
                    global_binary=global_binary,
                    local_binary=local_binary,
                    bbox=bbox,
                    min_contour_area=teacher.min_contour_area,
                    config=teacher.ranked_config,
                )
                audit_row = audit_map[key]
                _validate_baseline_against_audit(audit_row, baseline_result)
                frame_xy = pixel_xy[frame_orders == frame_order]
                evaluation = evaluate_bbox_refinement(
                    image=images[frame_order],
                    bbox=bbox,
                    baseline_candidates=baseline_candidates,
                    baseline_result=baseline_result,
                    config=refine_config,
                    frame_xy=frame_xy,
                    audit_categories=selected_row["categories"],
                    candidate_generation_only=args.candidate_generation_only,
                )
                stem = _key_name(key)
                proposal_relative = f"proposal_masks/{stem}.png"
                _write_binary_png(
                    Path(args.output_root) / proposal_relative,
                    evaluation["proposal_mask"],
                )
                decision_row = _decision_row(
                    key=key,
                    categories=selected_row["categories"],
                    audit_row=audit_row,
                    evaluation=evaluation,
                    proposal_relative_path=proposal_relative,
                )
                decision_rows.append(decision_row)
                candidate_rows.extend(
                    _candidate_rows(
                        key, selected_row["categories"], evaluation
                    )
                )
                if not args.no_overlays:
                    baseline_mask = (
                        np.asarray(baseline_result["mask"], dtype=bool)
                        if baseline_result["valid"]
                        else np.zeros(shape, dtype=bool)
                    )
                    overlay_category = str(evaluation["proposed_decision"])
                    is_good_control = "candidate_good_control" in selected_row["categories"]
                    if is_good_control and (
                        evaluation["proposed_decision"] != "auto_accept"
                        and not args.candidate_generation_only
                    ):
                        overlay_category = "regressions"
                    _render_overlay(
                        path=(
                            Path(args.output_root)
                            / "overlays"
                            / overlay_category
                            / f"{stem}.png"
                        ),
                        image=images[frame_order],
                        bbox=bbox,
                        baseline_mask=baseline_mask,
                        proposal_mask=evaluation["proposal_mask"],
                        row=decision_row,
                    )
            print(
                f"[{video_index}/{len(selected_videos)}] [OK] {video_name}: "
                f"bboxes={len(selection_by_video[video_name])}"
            )
        except Exception as exc:
            failure = {
                "video_name": video_name,
                "pseudo3d_h5": str(resolved.item.pseudo3d_h5),
                "annotated_h5": str(resolved.annotated_h5),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(
                f"[{video_index}/{len(selected_videos)}] [FAIL] {video_name}: "
                f"{failure['error']}"
            )
            if not args.continue_on_error:
                break

    decision_rows.sort(key=_row_key)
    candidate_rows.sort(
        key=lambda row: (*_row_key(row), int(row["candidate_rank"]))
    )
    selection_output = [
        {
            **{name: row[name] for name in KEY_FIELDS},
            "categories": "|".join(row["categories"]),
        }
        for row in selections
    ]
    _write_csv(Path(args.output_root) / "selection_resolved.csv", selection_output)
    _write_csv(Path(args.output_root) / "bbox_decisions.csv", decision_rows)
    _write_csv(Path(args.output_root) / "candidate_metrics.csv", candidate_rows)
    video_rows = _video_summary(decision_rows)
    category_rows = _category_summary(decision_rows)
    _write_csv(Path(args.output_root) / "video_summary.csv", video_rows)
    _write_csv(Path(args.output_root) / "category_summary.csv", category_rows)
    _write_csv(
        Path(args.output_root) / "failures.csv",
        failures,
        fieldnames=("video_name", "pseudo3d_h5", "annotated_h5", "error"),
    )
    run_config = {
        "schema_version": PROTOTYPE_SCHEMA_VERSION,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "annotated_root": str(Path(args.annotated_root).resolve()),
        "phase1_audit_root": str(Path(args.phase1_audit_root).resolve()),
        "phase1_audit_summary_sha256": file_sha256(
            Path(args.phase1_audit_root) / "audit_summary.json"
        ),
        "teacher_config": str(Path(args.teacher_config).resolve()),
        "teacher_fingerprint": teacher.fingerprint,
        "refine_config": str(Path(args.refine_config).resolve()),
        "refine_fingerprint": refine_config.fingerprint,
        "candidate_generation_only": bool(args.candidate_generation_only),
        "top_per_category": int(args.top_per_category),
        "selection_csv": (
            str(Path(args.selection_csv).resolve())
            if args.selection_csv is not None
            else None
        ),
    }
    with (Path(args.output_root) / "run_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(run_config, handle, allow_unicode=True, sort_keys=True)
    _assert_snapshot_unchanged(before)
    checksum_rows = _checksum_rows(Path(args.output_root))
    _write_csv(Path(args.output_root) / "checksums.csv", checksum_rows)

    decisions = Counter(str(row["proposed_decision"]) for row in decision_rows)
    summary = {
        "schema_version": PROTOTYPE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failures else "ok",
        "manifest": str(Path(args.manifest).resolve()),
        "phase1_audit_root": str(Path(args.phase1_audit_root).resolve()),
        "phase1_bbox_rows": int(phase1_summary["bbox_rows"]),
        "teacher_fingerprint": teacher.fingerprint,
        "refine_fingerprint": refine_config.fingerprint,
        "candidate_generation_only": bool(args.candidate_generation_only),
        "selected_videos": len(selected_videos),
        "selected_bboxes": len(selections),
        "processed_bboxes": len(decision_rows),
        "candidate_rows": len(candidate_rows),
        "decision_counts": dict(sorted(decisions.items())),
        "failure_rows": len(failures),
        "input_files_unchanged": True,
        "checksum_rows": len(checksum_rows),
    }
    with (Path(args.output_root) / "refine_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print("\nPhase 3 auto-refine prototype summary")
    print(f"  videos_checked        : {len(selected_videos) - len(failures)}/{len(selected_videos)}")
    print(f"  selected_bboxes       : {len(selections)}")
    print(f"  processed_bboxes      : {len(decision_rows)}")
    print(f"  candidate_rows        : {len(candidate_rows)}")
    print(f"  decision_counts       : {dict(sorted(decisions.items()))}")
    print(f"  failure_rows          : {len(failures)}")
    print("  input_files_unchanged : true")
    print(f"  output_root           : {args.output_root}")
    if failures:
        raise RuntimeError(f"Phase 3 prototype failed for {len(failures)} video(s)")
    if len(decision_rows) != len(selections):
        raise RuntimeError("Phase 3 prototype did not process every selected BBox")
    print("Stage 4 contour-teacher Phase 3 auto-refine prototype passed.")
    return {
        "selection_rows": selection_output,
        "decision_rows": decision_rows,
        "candidate_rows": candidate_rows,
        "failures": failures,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Stage 4 Phase 3 prototype: generate bounded BBox refine "
            "candidates and proposed decisions without modifying H5/XML inputs."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument("--phase1_audit_root", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--refine_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--selection_csv", type=Path, default=None)
    parser.add_argument(
        "--annotated_glob_template",
        type=str,
        default=DEFAULT_ANNOTATED_GLOB,
    )
    parser.add_argument("--expected_videos", type=int, default=182)
    parser.add_argument("--top_per_category", type=int, default=20)
    parser.add_argument("--candidate_generation_only", action="store_true")
    parser.add_argument("--no_overlays", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_videos < 0:
        raise SystemExit("--expected_videos must be >= 0")
    if args.top_per_category <= 0:
        raise SystemExit("--top_per_category must be > 0")
    try:
        run_prototype(args)
    except Exception as exc:
        raise SystemExit(
            f"Phase 3 auto-refine prototype failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
