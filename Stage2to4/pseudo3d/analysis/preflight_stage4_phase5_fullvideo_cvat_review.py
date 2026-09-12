from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.audit_stage4_contour_teacher import (
    _assert_metadata_matches_teacher,
    _assert_snapshot_unchanged,
    _preflight_annotated_h5,
    _preflight_pseudo3d,
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
    load_voc_bboxes,
    xml_bbox_to_local,
)
from pseudo3d.annotation.contour_teacher_refinement import (
    load_contour_refinement_config,
)
from pseudo3d.annotation.stage4_manual_review import (
    ManualReviewError,
    load_manual_review_config,
    prepare_output_root,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from pseudo3d.batch.export.batch_export_stage4_manual_review_cvat import (
    ALL_PHASE3_DECISIONS,
    DEFAULT_ANNOTATED_GLOB,
    _bbox_crop_metrics,
    _phase3_contract,
    _project_bbox_to_local_unclipped,
)


PREFLIGHT_SCHEMA_VERSION = 1
DEFAULT_SOURCE_ANNOTATED_GLOB = (
    "{video_name}/*bboxrank_v3_refined_auto_v1.h5"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _pipe_floats(values: Sequence[float]) -> str:
    return "|".join(str(float(value)) for value in values)


def _local_shape(image_shape: Sequence[int]) -> tuple[int, int]:
    shape = tuple(int(value) for value in image_shape)
    if len(shape) == 3:
        return shape[1], shape[2]
    if len(shape) == 4 and shape[1] == 1:
        return shape[2], shape[3]
    raise ManualReviewError(f"Unsupported local frame layout: {shape}")


def _strict_xml_paths(
    *, voc_xml_root: Path, video_name: str, annotation_dir_name: str
) -> list[Path]:
    directory = Path(voc_xml_root) / video_name / annotation_dir_name
    if not directory.is_dir():
        raise FileNotFoundError(f"Strict XML directory not found: {directory}")
    paths = sorted(directory.glob("*.xml"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"Strict XML directory contains no XML files: {directory}")
    return [path.resolve() for path in paths]


def _source_paths(
    *,
    selected_videos: Sequence[str],
    resolved_map: Mapping[str, Any],
    source_resolved_map: Mapping[str, Any],
    annotation_dir_name: str,
) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for video_name in selected_videos:
        resolved = resolved_map[video_name]
        source = source_resolved_map[video_name]
        rows.extend(
            (
                (video_name, "pseudo3d_h5", resolved.item.pseudo3d_h5.resolve()),
                (video_name, "annotated_v2_h5", resolved.annotated_h5.resolve()),
                (video_name, "annotated_v3_h5", source.annotated_h5.resolve()),
            )
        )
        rows.extend(
            (video_name, "strict_xml", path)
            for path in _strict_xml_paths(
                voc_xml_root=resolved.item.voc_xml_root,
                video_name=video_name,
                annotation_dir_name=annotation_dir_name,
            )
        )
    paths = [str(path) for _, _, path in rows]
    if len(set(paths)) != len(paths):
        duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
        raise ManualReviewError(f"Duplicate source file paths: {duplicates}")
    return rows


def _checksum_rows(
    sources: Sequence[tuple[str, str, Path]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_name, role, path in sources:
        stat = path.stat()
        rows.append(
            {
                "video_name": video_name,
                "role": role,
                "path": str(path),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": file_sha256(path),
            }
        )
    rows.sort(key=lambda row: (str(row["video_name"]), str(row["role"]), str(row["path"])))
    return rows


def verify_source_checksum_rows(
    current_rows: Sequence[Mapping[str, Any]], expected_manifest: Path
) -> None:
    expected_manifest = Path(expected_manifest).resolve()
    expected_rows = read_csv_rows(expected_manifest)

    def normalized(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], tuple[int, str]]:
        result: dict[tuple[str, str, str], tuple[int, str]] = {}
        for row in rows:
            key = (
                str(row.get("video_name", "")),
                str(row.get("role", "")),
                str(Path(str(row.get("path", ""))).resolve()),
            )
            if not all(key) or key in result:
                raise ManualReviewError(
                    f"Invalid/duplicate source checksum identity: {key}"
                )
            result[key] = (int(row.get("bytes", -1)), str(row.get("sha256", "")))
        return result

    current = normalized(current_rows)
    expected = normalized(expected_rows)
    if current != expected:
        missing = sorted(set(expected) - set(current))
        extra = sorted(set(current) - set(expected))
        changed = sorted(
            key for key in set(current) & set(expected) if current[key] != expected[key]
        )
        raise ManualReviewError(
            "Source checksum manifest mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}, changed={changed[:3]}"
        )


def _report_checksums(output_root: Path) -> list[dict[str, Any]]:
    excluded = {"report_checksums.csv", "preflight_summary.json"}
    rows: list[dict[str, Any]] = []
    for path in sorted(value for value in Path(output_root).iterdir() if value.is_file()):
        if path.name in excluded:
            continue
        rows.append(
            {
                "relative_path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return rows


def _load_h5_attrs(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return dict(handle.attrs)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output root is not empty; pass --overwrite: {output_root}"
        )
    teacher = load_audit_teacher_config(args.teacher_config)
    refine = load_contour_refinement_config(args.refine_config)
    refine.validate(require_production_thresholds=True)
    manual = load_manual_review_config(args.manual_review_config)
    if refine.base_teacher_config_name != teacher.config_name:
        raise ManualReviewError("refine/teacher config_name mismatch")
    if (
        refine.base_teacher_fingerprint is not None
        and refine.base_teacher_fingerprint != teacher.fingerprint
    ):
        raise ManualReviewError("refine/teacher fingerprint mismatch")

    decisions, phase3_summary, _ = _phase3_contract(
        phase3_root=args.phase3_root,
        manifest=args.manifest,
        annotated_root=args.annotated_root,
        phase1_audit_root=args.phase1_audit_root,
        teacher_fingerprint=teacher.fingerprint,
        refine_fingerprint=refine.fingerprint,
    )
    decision_counts = Counter(str(row["proposed_decision"]) for row in decisions)
    selected_videos = sorted({str(row["video_name"]) for row in decisions})
    if len(decisions) != int(args.expected_review_bboxes):
        raise ManualReviewError(
            "Phase 3 target BBox count mismatch: "
            f"expected={args.expected_review_bboxes}, actual={len(decisions)}"
        )
    if len(selected_videos) != int(args.expected_selected_videos):
        raise ManualReviewError(
            "Phase 3 selected video count mismatch: "
            f"expected={args.expected_selected_videos}, actual={len(selected_videos)}"
        )
    for decision in ALL_PHASE3_DECISIONS:
        expected = int(getattr(args, f"expected_{decision}"))
        actual = int(decision_counts.get(decision, 0))
        if actual != expected:
            raise ManualReviewError(
                f"Phase 3 {decision} count mismatch: expected={expected}, actual={actual}"
            )

    items = load_stage4_sweep_manifest(args.manifest)
    resolved = resolve_inputs(
        items,
        annotated_root=args.annotated_root,
        annotated_glob_template=args.annotated_glob_template,
        expected_videos=args.expected_videos,
    )
    source_resolved = resolve_inputs(
        items,
        annotated_root=args.source_annotated_root,
        annotated_glob_template=args.source_annotated_glob_template,
        expected_videos=args.expected_videos,
    )
    resolved_map = {value.item.video_name: value for value in resolved}
    source_resolved_map = {value.item.video_name: value for value in source_resolved}
    unknown = sorted(set(selected_videos) - set(resolved_map))
    if unknown:
        raise ManualReviewError(f"Phase 3 references unknown manifest videos: {unknown}")

    sources = _source_paths(
        selected_videos=selected_videos,
        resolved_map=resolved_map,
        source_resolved_map=source_resolved_map,
        annotation_dir_name=teacher.xml_annotation_dir_name,
    )
    before = _snapshot(path for _, _, path in sources)

    decisions_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in decisions:
        decisions_by_video[str(row["video_name"])].append(row)

    target_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    seen_target_keys: set[tuple[str, int, int, int]] = set()
    print("Stage 4 Phase 5 full-video real-data preflight (read-only)")
    print(f"  manifest        : {Path(args.manifest).resolve()}")
    print(f"  annotated v2    : {Path(args.annotated_root).resolve()}")
    print(f"  annotated v3    : {Path(args.source_annotated_root).resolve()}")
    print(f"  selected videos : {len(selected_videos)}")
    print(f"  target BBoxes   : {len(decisions)}")

    for number, video_name in enumerate(selected_videos, start=1):
        resolved_item = resolved_map[video_name]
        source_item = source_resolved_map[video_name]
        preflight = preflight_input(resolved_item, teacher)
        source_frame_annotation, source_attrs, _ = _preflight_annotated_h5(
            source_item.annotated_h5
        )
        _assert_metadata_matches_teacher(source_attrs, teacher)
        if len(source_frame_annotation["frame_order"]) != int(preflight["bbox_rows"]):
            raise ManualReviewError(
                f"v2/v3 frame_annotation row count mismatch: {video_name}"
            )

        image_shape, frame_indices = _preflight_pseudo3d(
            resolved_item.item.pseudo3d_h5
        )
        shape_hw = _local_shape(image_shape)
        pseudo_attrs = _load_h5_attrs(resolved_item.item.pseudo3d_h5)
        voc_bboxes = load_voc_bboxes(
            resolved_item.item.voc_xml_root,
            video_name=video_name,
            frame_indices=frame_indices,
            xml_frame_number_offsets=teacher.xml_frame_number_offsets,
            xml_frame_id_source=teacher.xml_frame_id_source,
            xml_annotation_dir_name=teacher.xml_annotation_dir_name,
            strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
        )

        local_target_rows: list[dict[str, Any]] = []
        for decision_row in sorted(
            decisions_by_video[video_name],
            key=lambda row: (
                int(row["frame_order"]),
                int(row["frame_index"]),
                int(row["bbox_index"]),
            ),
        ):
            frame_order = int(decision_row["frame_order"])
            frame_index = int(decision_row["frame_index"])
            bbox_index = int(decision_row["bbox_index"])
            key = (video_name, frame_order, frame_index, bbox_index)
            if key in seen_target_keys:
                raise ManualReviewError(f"Duplicate target BBox key: {key}")
            seen_target_keys.add(key)
            if frame_order < 0 or frame_order >= len(frame_indices):
                raise ManualReviewError(f"Target frame_order is out of range: {key}")
            if int(frame_indices[frame_order]) != frame_index:
                raise ManualReviewError(f"Target frame_index alignment mismatch: {key}")
            frame_vocs = voc_bboxes.get(frame_order, [])
            if bbox_index < 0 or bbox_index >= len(frame_vocs):
                raise ManualReviewError(
                    f"Target BBox is missing from strict XML geometry: {key}"
                )
            voc = frame_vocs[bbox_index]
            bbox = xml_bbox_to_local(
                voc.xml_xyxy,
                image_size_wh=voc.image_size_wh,
                attrs=pseudo_attrs,
                local_shape_hw=shape_hw,
            )
            projected = _project_bbox_to_local_unclipped(
                raw_xyxy=bbox.raw_xyxy,
                attrs=pseudo_attrs,
                local_shape_hw=shape_hw,
            )
            crop = _bbox_crop_metrics(
                projected_xyxy=projected,
                local_shape_hw=shape_hw,
            )
            target = {
                "bbox_key": f"{video_name}__fo{frame_order:05d}__fi{frame_index:08d}__bbox{bbox_index:03d}",
                "video_name": video_name,
                "frame_order": frame_order,
                "frame_index": frame_index,
                "bbox_index": bbox_index,
                "phase3_decision": decision_row["proposed_decision"],
                "phase3_reason_codes": decision_row.get("reason_codes", ""),
                "xml_path": str(voc.xml_path.resolve()),
                "xml_sha256": file_sha256(voc.xml_path),
                "bbox_xml_xyxy": _pipe_floats(voc.xml_xyxy),
                "projected_bbox_xyxy": _pipe_floats(crop["projected_bbox_xyxy"]),
                "clipped_bbox_xyxy": _pipe_floats(crop["clipped_bbox_xyxy"]),
                "projected_area": crop["projected_area"],
                "visible_intersection_area": crop["visible_intersection_area"],
                "visible_fraction": crop["visible_fraction"],
                "touches_left": crop["touches_left"],
                "touches_right": crop["touches_right"],
                "touches_top": crop["touches_top"],
                "touches_bottom": crop["touches_bottom"],
                "fully_outside_crop": crop["fully_outside_crop"],
                "partially_clipped": crop["partially_clipped"],
                "fully_visible": crop["fully_visible"],
                "crop_status": crop["crop_status"],
            }
            target_rows.append(target)
            local_target_rows.append(target)

        local_crop_counts = Counter(str(row["crop_status"]) for row in local_target_rows)
        local_decision_counts = Counter(
            str(row["phase3_decision"]) for row in local_target_rows
        )
        target_frame_count = len(
            {
                (int(row["frame_order"]), int(row["frame_index"]))
                for row in local_target_rows
            }
        )
        video_row = {
            "video_name": video_name,
            "frames": int(image_shape[0]),
            "height": int(shape_hw[0]),
            "width": int(shape_hw[1]),
            "strict_xml_bbox_rows": int(preflight["bbox_rows"]),
            "target_frames": target_frame_count,
            "target_bboxes": len(local_target_rows),
            "auto_accept": int(local_decision_counts.get("auto_accept", 0)),
            "auto_refine": int(local_decision_counts.get("auto_refine", 0)),
            "manual_review": int(local_decision_counts.get("manual_review", 0)),
            "fully_visible": int(local_crop_counts.get("fully_visible", 0)),
            "partially_clipped": int(local_crop_counts.get("partially_clipped", 0)),
            "fully_outside_crop": int(local_crop_counts.get("fully_outside_crop", 0)),
            "pseudo3d_h5": str(resolved_item.item.pseudo3d_h5.resolve()),
            "annotated_v2_h5": str(resolved_item.annotated_h5.resolve()),
            "annotated_v3_h5": str(source_item.annotated_h5.resolve()),
        }
        video_rows.append(video_row)
        print(
            f"  [{number}/{len(selected_videos)}] [OK] {video_name}: "
            f"frames={video_row['frames']}, targets={video_row['target_bboxes']}, "
            f"crop={dict(sorted(local_crop_counts.items()))}"
        )

    target_rows.sort(key=lambda row: str(row["bbox_key"]))
    video_rows.sort(key=lambda row: str(row["video_name"]))
    if len(target_rows) != int(args.expected_review_bboxes):
        raise ManualReviewError("Not every Phase 3 target was resolved to strict XML")
    if len(seen_target_keys) != len(target_rows):
        raise ManualReviewError("Resolved target BBox keys are not unique")

    source_role_counts = Counter(role for _, role, _ in sources)
    print(
        "  hashing source files : "
        f"{len(sources)} {dict(sorted(source_role_counts.items()))}"
    )
    source_checksum_rows = _checksum_rows(sources)
    print("  [OK] source SHA-256 manifest created")
    verified_source_manifest = getattr(args, "verify_source_checksums", None)
    if verified_source_manifest is not None:
        verify_source_checksum_rows(
            source_checksum_rows, Path(verified_source_manifest)
        )
    _assert_snapshot_unchanged(before)
    prepare_output_root(output_root, overwrite=args.overwrite)
    write_csv_rows(output_root / "video_summary.csv", video_rows)
    write_csv_rows(output_root / "target_bboxes.csv", target_rows)
    write_csv_rows(
        output_root / "source_checksums.csv", source_checksum_rows
    )
    run_config = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "annotated_root": str(Path(args.annotated_root).resolve()),
        "source_annotated_root": str(Path(args.source_annotated_root).resolve()),
        "phase1_audit_root": str(Path(args.phase1_audit_root).resolve()),
        "phase3_root": str(Path(args.phase3_root).resolve()),
        "teacher_config": str(Path(args.teacher_config).resolve()),
        "teacher_fingerprint": teacher.fingerprint,
        "refine_config": str(Path(args.refine_config).resolve()),
        "refine_fingerprint": refine.fingerprint,
        "manual_review_config": str(Path(args.manual_review_config).resolve()),
        "manual_review_fingerprint": manual.fingerprint,
        "expected_videos": int(args.expected_videos),
        "expected_selected_videos": int(args.expected_selected_videos),
        "expected_review_bboxes": int(args.expected_review_bboxes),
        "expected_auto_accept": int(args.expected_auto_accept),
        "expected_auto_refine": int(args.expected_auto_refine),
        "expected_manual_review": int(args.expected_manual_review),
    }
    with (output_root / "run_config.yaml").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        yaml.safe_dump(run_config, handle, allow_unicode=True, sort_keys=True)

    report_checksum_rows = _report_checksums(output_root)
    write_csv_rows(
        output_root / "report_checksums.csv", report_checksum_rows
    )
    crop_counts = Counter(str(row["crop_status"]) for row in target_rows)
    summary = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "ok",
        "selected_videos": len(video_rows),
        "total_frames": int(sum(int(row["frames"]) for row in video_rows)),
        "target_frames": int(sum(int(row["target_frames"]) for row in video_rows)),
        "target_bboxes": len(target_rows),
        "phase3_decision_counts": dict(sorted(decision_counts.items())),
        "crop_status_counts": dict(sorted(crop_counts.items())),
        "source_checksum_rows": len(source_checksum_rows),
        "source_checksums_sha256": file_sha256(
            output_root / "source_checksums.csv"
        ),
        "verified_source_checksums": (
            str(Path(verified_source_manifest).resolve())
            if verified_source_manifest is not None
            else ""
        ),
        "phase3_processed_bboxes": int(phase3_summary["processed_bboxes"]),
        "input_files_unchanged": True,
        "failure_rows": 0,
    }
    write_json(output_root / "preflight_summary.json", summary)

    print("\nFull-video preflight summary")
    print(f"  videos_checked        : {summary['selected_videos']}")
    print(f"  total_frames          : {summary['total_frames']}")
    print(f"  target_frames         : {summary['target_frames']}")
    print(f"  target_bboxes         : {summary['target_bboxes']}")
    print(f"  decision_counts       : {summary['phase3_decision_counts']}")
    print(f"  crop_status_counts    : {summary['crop_status_counts']}")
    print(f"  source_checksum_rows  : {summary['source_checksum_rows']}")
    print("  input_files_unchanged : true")
    print(f"  output_root           : {output_root.resolve()}")
    print("Stage 4 Phase 5 full-video real-data preflight passed.")
    return {
        "summary": summary,
        "video_rows": video_rows,
        "target_rows": target_rows,
        "source_checksum_rows": source_checksum_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only real-data preflight for Stage 4 Phase 5 full-video CVAT export."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--phase1_audit_root", type=Path, required=True)
    parser.add_argument("--phase3_root", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--refine_config", type=Path, required=True)
    parser.add_argument("--manual_review_config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--annotated_glob_template", default=DEFAULT_ANNOTATED_GLOB)
    parser.add_argument(
        "--source_annotated_glob_template",
        default=DEFAULT_SOURCE_ANNOTATED_GLOB,
    )
    parser.add_argument("--expected_videos", type=int, default=182)
    parser.add_argument("--expected_selected_videos", type=int, default=60)
    parser.add_argument("--expected_review_bboxes", type=int, default=129)
    parser.add_argument("--expected_auto_accept", type=int, default=31)
    parser.add_argument("--expected_auto_refine", type=int, default=35)
    parser.add_argument("--expected_manual_review", type=int, default=63)
    parser.add_argument(
        "--verify_source_checksums",
        type=Path,
        default=None,
        help=(
            "Optional earlier source_checksums.csv. Requiring an exact match makes "
            "the same preflight usable after package generation."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "expected_videos",
        "expected_selected_videos",
        "expected_review_bboxes",
        "expected_auto_accept",
        "expected_auto_refine",
        "expected_manual_review",
    ):
        if int(getattr(args, name)) < 0:
            raise SystemExit(f"--{name} must be >= 0")
    try:
        run_preflight(args)
    except Exception as exc:
        raise SystemExit(
            f"Full-video preflight failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
