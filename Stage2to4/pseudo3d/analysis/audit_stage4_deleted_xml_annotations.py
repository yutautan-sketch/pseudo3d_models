from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    Stage4SweepManifestItem,
    file_sha256,
    load_stage4_sweep_manifest,
)


V5_TEACHER_TOKEN = "bboxrank_v5_cvat_authoritative_v1"

INVENTORY_FIELDS = (
    "xml_status",
    "missing_candidate",
    "frame_all_xml_missing",
    "frame_has_mixed_xml_presence",
    "video_name",
    "frame_order",
    "frame_index",
    "frame_stem",
    "xml_name",
    "xml_annotation_dir",
    "saved_xml_path",
    "canonical_xml_path",
    "resolved_xml_path",
    "saved_path_exists",
    "canonical_path_exists",
    "xml_sha256",
    "xml_bbox_rows",
    "frame_bbox_rows",
    "valid_contour_rows",
    "frame_points",
    "positive_points",
    "ignore_points",
    "background_points",
    "reviewed_video",
    "label_authority",
    "cvat_mask_status",
    "cvat_mask_positive_pixels",
)

VIDEO_FIELDS = (
    "video_name",
    "reviewed_video",
    "xml_entries",
    "bbox_rows",
    "missing_xml_entries",
    "missing_frames",
    "missing_bbox_rows",
    "missing_frame_points",
    "missing_positive_points",
    "missing_ignore_points",
    "mixed_presence_frames",
    "saved_only_xml_entries",
    "canonical_only_xml_entries",
    "conflicting_copy_entries",
)

CHECKSUM_FIELDS = ("kind", "path", "sha256")


class DeletedXmlAuditError(RuntimeError):
    """Raised when the read-only deleted-XML audit contract is invalid."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _single_v5_h5(root: Path, video_name: str) -> Path:
    matches = sorted(
        (Path(root) / video_name).glob(f"*{V5_TEACHER_TOKEN}.h5")
    )
    if len(matches) != 1:
        raise DeletedXmlAuditError(
            f"Expected one v5 H5 for {video_name}, found {len(matches)}"
        )
    return matches[0].resolve()


def _canonical_xml_path(
    item: Stage4SweepManifestItem,
    saved_xml_path: Path,
) -> Path:
    if saved_xml_path.suffix.lower() != ".xml" or not saved_xml_path.name:
        raise DeletedXmlAuditError(
            f"Invalid saved frame_annotation/xml_path: {saved_xml_path}"
        )
    annotation_dir = saved_xml_path.parent.name
    if not annotation_dir or annotation_dir in {".", ".."}:
        raise DeletedXmlAuditError(
            f"Could not resolve XML annotation directory: {saved_xml_path}"
        )
    canonical_dir = (
        item.voc_xml_root
        / item.video_name
        / annotation_dir
    ).resolve(strict=False)
    if not canonical_dir.is_dir():
        raise DeletedXmlAuditError(
            f"Canonical XML annotation directory not found: {canonical_dir}"
        )
    return canonical_dir / saved_xml_path.name


def classify_xml_paths(
    *,
    saved_xml_path: Path,
    canonical_xml_path: Path,
) -> dict[str, Any]:
    saved = Path(saved_xml_path).resolve(strict=False)
    canonical = Path(canonical_xml_path).resolve(strict=False)
    saved_exists = saved.is_file()
    canonical_exists = canonical.is_file()
    same_path = saved == canonical
    resolved: Path | None = None
    digest = ""

    if same_path and canonical_exists:
        status = "canonical"
        resolved = canonical
        digest = file_sha256(canonical)
    elif canonical_exists and saved_exists:
        canonical_sha = file_sha256(canonical)
        saved_sha = file_sha256(saved)
        status = (
            "canonical_saved_identical"
            if canonical_sha == saved_sha
            else "canonical_saved_conflict"
        )
        resolved = canonical
        digest = canonical_sha
    elif canonical_exists:
        status = "canonical_only"
        resolved = canonical
        digest = file_sha256(canonical)
    elif saved_exists:
        status = "saved_only"
        resolved = saved
        digest = file_sha256(saved)
    else:
        status = "missing"

    return {
        "xml_status": status,
        "missing_candidate": status == "missing",
        "saved_path_exists": saved_exists,
        "canonical_path_exists": canonical_exists,
        "resolved_xml_path": str(resolved) if resolved is not None else "",
        "xml_sha256": digest,
    }


def _read_frame_provenance(handle: h5py.File) -> dict[int, dict[str, Any]]:
    if "manual_review_fullvideo_frames" not in handle:
        raise DeletedXmlAuditError("v5 H5 lacks manual_review_fullvideo_frames")
    group = handle["manual_review_fullvideo_frames"]
    required = (
        "frame_order",
        "frame_index",
        "frame_stem",
        "cvat_mask_status",
        "cvat_mask_positive_pixels",
        "label_authority",
    )
    for name in required:
        if name not in group:
            raise DeletedXmlAuditError(
                f"manual_review_fullvideo_frames lacks {name}"
            )
    lengths = {len(group[name]) for name in required}
    if len(lengths) != 1:
        raise DeletedXmlAuditError(
            "manual_review_fullvideo_frames datasets have different lengths"
        )
    result: dict[int, dict[str, Any]] = {}
    for values in zip(
        group["frame_order"][:],
        group["frame_index"][:],
        group["frame_stem"][:],
        group["cvat_mask_status"][:],
        group["cvat_mask_positive_pixels"][:],
        group["label_authority"][:],
        strict=True,
    ):
        order = int(values[0])
        if order in result:
            raise DeletedXmlAuditError(
                f"Duplicate frame provenance order: {order}"
            )
        result[order] = {
            "frame_index": int(values[1]),
            "frame_stem": _decode(values[2]),
            "cvat_mask_status": _decode(values[3]),
            "cvat_mask_positive_pixels": int(values[4]),
            "label_authority": _decode(values[5]),
        }
    return result


def _frame_point_counts(
    *,
    frame_orders: np.ndarray,
    labels: np.ndarray,
    num_frames: int,
) -> dict[str, np.ndarray]:
    orders = np.asarray(frame_orders, dtype=np.int64)
    values = np.asarray(labels, dtype=np.int8)
    if orders.ndim != 1 or values.shape != orders.shape:
        raise DeletedXmlAuditError("point frame_order and point_label are not aligned")
    if set(int(value) for value in np.unique(values)) - {-1, 0, 1}:
        raise DeletedXmlAuditError("point_label contains values outside {-1,0,1}")
    if orders.size and (
        int(orders.min()) < 0 or int(orders.max()) >= int(num_frames)
    ):
        raise DeletedXmlAuditError("point frame_order lies outside frame range")
    return {
        "frame_points": np.bincount(orders, minlength=num_frames),
        "positive_points": np.bincount(
            orders[values == 1], minlength=num_frames
        ),
        "ignore_points": np.bincount(
            orders[values == -1], minlength=num_frames
        ),
        "background_points": np.bincount(
            orders[values == 0], minlength=num_frames
        ),
    }


def audit_video_h5(
    *,
    item: Stage4SweepManifestItem,
    h5_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[Path]]:
    with h5py.File(h5_path, "r") as handle:
        if _decode(handle.attrs.get("contour_teacher_schema", "")) != V5_TEACHER_TOKEN:
            raise DeletedXmlAuditError(
                f"Unexpected teacher schema for {item.video_name}: {h5_path}"
            )
        if _decode(handle.attrs.get("video_name", "")) != item.video_name:
            raise DeletedXmlAuditError(f"H5 video_name mismatch: {h5_path}")
        for name in (
            "point_cloud/frame_order",
            "annotation/point_label",
            "annotation/valid_mask",
            "frame_annotation/frame_order",
            "frame_annotation/frame_index",
            "frame_annotation/xml_path",
            "frame_annotation/valid_contour",
        ):
            if name not in handle:
                raise DeletedXmlAuditError(f"v5 H5 lacks {name}: {h5_path}")

        frame_orders = handle["point_cloud/frame_order"][:].astype(np.int64)
        labels = handle["annotation/point_label"][:].astype(np.int8)
        valid = handle["annotation/valid_mask"][:].astype(bool)
        if not np.array_equal(valid, labels != -1):
            raise DeletedXmlAuditError(f"valid_mask differs from labels: {h5_path}")
        num_frames = int(handle.attrs.get("num_frames", -1))
        if num_frames < 0 and "point_cloud/per_frame_counts" in handle:
            num_frames = len(handle["point_cloud/per_frame_counts"])
        if num_frames < 0:
            raise DeletedXmlAuditError(f"Could not resolve num_frames: {h5_path}")
        point_counts = _frame_point_counts(
            frame_orders=frame_orders,
            labels=labels,
            num_frames=num_frames,
        )

        frame_group = handle["frame_annotation"]
        row_count = len(frame_group["frame_order"])
        for dataset in frame_group.values():
            if isinstance(dataset, h5py.Dataset) and len(dataset) != row_count:
                raise DeletedXmlAuditError(
                    f"frame_annotation datasets have different lengths: {h5_path}"
                )
        annotation_orders = frame_group["frame_order"][:].astype(np.int64)
        annotation_indices = frame_group["frame_index"][:].astype(np.int64)
        xml_paths = [_decode(value) for value in frame_group["xml_path"][:]]
        valid_contours = frame_group["valid_contour"][:].astype(bool)
        if annotation_orders.size and (
            int(annotation_orders.min()) < 0
            or int(annotation_orders.max()) >= num_frames
        ):
            raise DeletedXmlAuditError(
                f"frame_annotation order lies outside frame range: {h5_path}"
            )
        provenance = _read_frame_provenance(handle)
        reviewed_video = bool(
            handle.attrs.get("manual_review_fullvideo_reviewed_video", False)
        )
        if reviewed_video and len(provenance) != num_frames:
            raise DeletedXmlAuditError(
                f"Reviewed v5 provenance does not cover every frame: {h5_path}"
            )
        if not reviewed_video and provenance:
            raise DeletedXmlAuditError(
                f"Inherited v5 H5 unexpectedly has frame provenance: {h5_path}"
            )

    grouped_rows: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    paths_by_key: dict[tuple[int, int, str], Path] = {}
    frame_keys: dict[tuple[int, int], list[tuple[int, int, str]]] = defaultdict(list)
    for row_index, (order, index, xml_text) in enumerate(
        zip(annotation_orders, annotation_indices, xml_paths, strict=True)
    ):
        saved = Path(xml_text).expanduser().resolve(strict=False)
        key = (int(order), int(index), str(saved))
        grouped_rows[key].append(row_index)
        paths_by_key[key] = saved
    for key in grouped_rows:
        frame_keys[(key[0], key[1])].append(key)

    current_xml_paths: set[Path] = set()
    rows: list[dict[str, Any]] = []
    classifications: dict[tuple[int, int, str], dict[str, Any]] = {}
    for key in sorted(grouped_rows):
        saved = paths_by_key[key]
        canonical = _canonical_xml_path(item, saved)
        classification = classify_xml_paths(
            saved_xml_path=saved,
            canonical_xml_path=canonical,
        )
        classifications[key] = classification
        if saved.is_file():
            current_xml_paths.add(saved)
        if canonical.is_file():
            current_xml_paths.add(canonical)

    for key in sorted(grouped_rows):
        order, index, _ = key
        saved = paths_by_key[key]
        canonical = _canonical_xml_path(item, saved)
        classification = classifications[key]
        same_frame_keys = frame_keys[(order, index)]
        missing_flags = [
            bool(classifications[frame_key]["missing_candidate"])
            for frame_key in same_frame_keys
        ]
        frame_all_missing = bool(missing_flags) and all(missing_flags)
        frame_mixed = any(missing_flags) and not frame_all_missing
        provenance_row = provenance.get(order)
        if provenance_row is not None:
            if int(provenance_row["frame_index"]) != index:
                raise DeletedXmlAuditError(
                    f"Frame provenance index mismatch: {item.video_name}/{order}"
                )
            frame_stem = str(provenance_row["frame_stem"])
            label_authority = str(provenance_row["label_authority"])
            cvat_status = str(provenance_row["cvat_mask_status"])
            cvat_pixels = int(provenance_row["cvat_mask_positive_pixels"])
        else:
            frame_stem = f"{item.video_name}__fo{order:05d}__fi{index:08d}"
            label_authority = "inherited"
            cvat_status = "not_reviewed"
            cvat_pixels = 0
        row_indices = grouped_rows[key]
        rows.append(
            {
                **classification,
                "frame_all_xml_missing": frame_all_missing,
                "frame_has_mixed_xml_presence": frame_mixed,
                "video_name": item.video_name,
                "frame_order": order,
                "frame_index": index,
                "frame_stem": frame_stem,
                "xml_name": saved.name,
                "xml_annotation_dir": saved.parent.name,
                "saved_xml_path": str(saved),
                "canonical_xml_path": str(canonical),
                "xml_bbox_rows": len(row_indices),
                "frame_bbox_rows": sum(
                    len(grouped_rows[frame_key]) for frame_key in same_frame_keys
                ),
                "valid_contour_rows": int(
                    np.sum(valid_contours[np.asarray(row_indices, dtype=np.int64)])
                ),
                "frame_points": int(point_counts["frame_points"][order]),
                "positive_points": int(point_counts["positive_points"][order]),
                "ignore_points": int(point_counts["ignore_points"][order]),
                "background_points": int(point_counts["background_points"][order]),
                "reviewed_video": reviewed_video,
                "label_authority": label_authority,
                "cvat_mask_status": cvat_status,
                "cvat_mask_positive_pixels": cvat_pixels,
            }
        )

    missing_rows = [row for row in rows if row["missing_candidate"]]
    missing_frames = {
        (int(row["frame_order"]), int(row["frame_index"])) for row in missing_rows
    }
    all_missing_frame_rows = {
        (int(row["frame_order"]), int(row["frame_index"])): row
        for row in missing_rows
        if row["frame_all_xml_missing"]
    }
    status_counts = Counter(str(row["xml_status"]) for row in rows)
    video_summary = {
        "video_name": item.video_name,
        "reviewed_video": reviewed_video,
        "xml_entries": len(rows),
        "bbox_rows": row_count,
        "missing_xml_entries": len(missing_rows),
        "missing_frames": len(missing_frames),
        "missing_bbox_rows": sum(int(row["xml_bbox_rows"]) for row in missing_rows),
        "missing_frame_points": sum(
            int(row["frame_points"]) for row in all_missing_frame_rows.values()
        ),
        "missing_positive_points": sum(
            int(row["positive_points"]) for row in all_missing_frame_rows.values()
        ),
        "missing_ignore_points": sum(
            int(row["ignore_points"]) for row in all_missing_frame_rows.values()
        ),
        "mixed_presence_frames": len(
            {
                (int(row["frame_order"]), int(row["frame_index"]))
                for row in rows
                if row["frame_has_mixed_xml_presence"]
            }
        ),
        "saved_only_xml_entries": status_counts["saved_only"],
        "canonical_only_xml_entries": status_counts["canonical_only"],
        "conflicting_copy_entries": status_counts["canonical_saved_conflict"],
    }
    return rows, video_summary, current_xml_paths


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    exclusion_path = Path(args.exclusion_manifest).resolve()
    annotated_root = Path(args.annotated_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output_root is not empty; pass --overwrite: {output_root}"
        )
    if not annotated_root.is_dir():
        raise FileNotFoundError(f"annotated_root not found: {annotated_root}")

    manifest_items = load_stage4_sweep_manifest(manifest_path)
    exclusions = load_exclusions(exclusion_path)
    excluded = {row.video_name for row in exclusions}
    enabled_items = sorted(
        (
            item
            for item in manifest_items
            if item.enabled and item.video_name not in excluded
        ),
        key=lambda item: item.video_name,
    )
    if len(enabled_items) != int(args.expected_videos):
        raise DeletedXmlAuditError(
            f"Enabled video count mismatch: {len(enabled_items)} != "
            f"{args.expected_videos}"
        )
    if len(excluded) != int(args.expected_excluded_videos):
        raise DeletedXmlAuditError(
            f"Excluded video count mismatch: {len(excluded)} != "
            f"{args.expected_excluded_videos}"
        )
    enabled_names = {item.video_name for item in enabled_items}
    for item in enabled_items:
        if not item.voc_xml_root.is_dir():
            raise FileNotFoundError(
                f"VOC XML root not found for {item.video_name}: "
                f"{item.voc_xml_root}"
            )
        video_xml_root = item.voc_xml_root / item.video_name
        if not video_xml_root.is_dir():
            raise FileNotFoundError(
                f"VOC video directory not found: {video_xml_root}"
            )
    h5_files = sorted(annotated_root.rglob(f"*{V5_TEACHER_TOKEN}.h5"))
    if len(h5_files) != len(enabled_items):
        raise DeletedXmlAuditError(
            f"v5 H5 count mismatch: {len(h5_files)} != {len(enabled_items)}"
        )
    for video in excluded:
        if (annotated_root / video).exists():
            raise DeletedXmlAuditError(
                f"Excluded video exists in v5 annotated root: {video}"
            )

    protected: dict[Path, str] = {
        manifest_path: "manifest",
        exclusion_path: "exclusion_manifest",
    }
    resolved_h5: dict[str, Path] = {}
    for item in enabled_items:
        h5_path = _single_v5_h5(annotated_root, item.video_name)
        resolved_h5[item.video_name] = h5_path
        protected[h5_path] = "v5_annotated_h5"
    if set(resolved_h5) != enabled_names:
        raise AssertionError("Resolved v5 video inventory is inconsistent")
    before_hashes = {path: file_sha256(path) for path in protected}

    print("Stage 4 deleted-XML annotation audit (read-only)")
    print(f"  manifest       : {manifest_path}")
    print(f"  annotated root : {annotated_root}")
    print(f"  videos         : {len(enabled_items)}")
    inventory_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    xml_paths: set[Path] = set()
    for index, item in enumerate(enabled_items, start=1):
        rows, video_summary, video_xml_paths = audit_video_h5(
            item=item,
            h5_path=resolved_h5[item.video_name],
        )
        inventory_rows.extend(rows)
        video_rows.append(video_summary)
        xml_paths.update(video_xml_paths)
        print(
            f"  [{index}/{len(enabled_items)}] [OK] {item.video_name}: "
            f"bbox_rows={video_summary['bbox_rows']}, "
            f"missing_frames={video_summary['missing_frames']}, "
            f"positive={video_summary['missing_positive_points']}"
        )

    for path in sorted(xml_paths):
        protected[path] = "current_voc_xml"
        before_hashes[path] = file_sha256(path)
    after_hashes = {path: file_sha256(path) for path in protected}
    changed = [
        str(path)
        for path in sorted(protected)
        if before_hashes[path] != after_hashes[path]
    ]
    if changed:
        raise DeletedXmlAuditError(f"Audit input files changed: {changed}")

    candidate_rows = [
        row for row in inventory_rows if bool(row["missing_candidate"])
    ]
    missing_frame_keys = {
        (str(row["video_name"]), int(row["frame_order"]), int(row["frame_index"]))
        for row in candidate_rows
    }
    all_missing_frame_keys = {
        (str(row["video_name"]), int(row["frame_order"]), int(row["frame_index"]))
        for row in candidate_rows
        if bool(row["frame_all_xml_missing"])
    }
    mixed_frame_keys = {
        (str(row["video_name"]), int(row["frame_order"]), int(row["frame_index"]))
        for row in inventory_rows
        if bool(row["frame_has_mixed_xml_presence"])
    }
    all_missing_frame_rows = {
        (
            str(row["video_name"]),
            int(row["frame_order"]),
            int(row["frame_index"]),
        ): row
        for row in candidate_rows
        if bool(row["frame_all_xml_missing"])
    }
    if int(args.expected_missing_frames) >= 0 and len(missing_frame_keys) != int(
        args.expected_missing_frames
    ):
        raise DeletedXmlAuditError(
            f"Missing frame count mismatch: {len(missing_frame_keys)} != "
            f"{args.expected_missing_frames}"
        )
    if int(args.expected_missing_xml_entries) >= 0 and len(candidate_rows) != int(
        args.expected_missing_xml_entries
    ):
        raise DeletedXmlAuditError(
            f"Missing XML entry count mismatch: {len(candidate_rows)} != "
            f"{args.expected_missing_xml_entries}"
        )

    status_counts = Counter(str(row["xml_status"]) for row in inventory_rows)
    reviewed_count = sum(bool(row["reviewed_video"]) for row in video_rows)
    summary = {
        "schema_version": 1,
        "status": "ok",
        "audit_kind": "deleted_xml_annotation_candidates",
        "teacher_version": V5_TEACHER_TOKEN,
        "input_files_unchanged": True,
        "h5_files_written": 0,
        "videos": len(enabled_items),
        "reviewed_videos": reviewed_count,
        "inherited_videos": len(enabled_items) - reviewed_count,
        "excluded_videos": sorted(excluded),
        "bbox_rows": sum(int(row["bbox_rows"]) for row in video_rows),
        "xml_entries": len(inventory_rows),
        "xml_status_counts": dict(sorted(status_counts.items())),
        "missing_xml_entries": len(candidate_rows),
        "missing_frames": len(missing_frame_keys),
        "all_xml_missing_frames": len(all_missing_frame_keys),
        "mixed_xml_presence_frames": len(mixed_frame_keys),
        "missing_bbox_rows": sum(
            int(row["xml_bbox_rows"]) for row in candidate_rows
        ),
        "all_missing_frame_positive_points": sum(
            int(row["positive_points"])
            for row in all_missing_frame_rows.values()
        ),
        "manifest_sha256": file_sha256(manifest_path),
        "exclusion_manifest_sha256": file_sha256(exclusion_path),
        "failure_rows": 0,
        "report_files_written": 5,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        _write_csv(temporary / "xml_inventory.csv", inventory_rows, INVENTORY_FIELDS)
        _write_csv(
            temporary / "missing_xml_candidates.csv",
            candidate_rows,
            INVENTORY_FIELDS,
        )
        _write_csv(temporary / "video_summary.csv", video_rows, VIDEO_FIELDS)
        _write_csv(
            temporary / "input_checksums.csv",
            [
                {
                    "kind": protected[path],
                    "path": str(path),
                    "sha256": after_hashes[path],
                }
                for path in sorted(protected)
            ],
            CHECKSUM_FIELDS,
        )
        _write_json(temporary / "audit_summary.json", summary)
        if output_root.exists():
            if any(output_root.iterdir()) and not args.overwrite:
                raise FileExistsError(
                    f"output_root became non-empty: {output_root}"
                )
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("\nDeleted-XML audit summary")
    for name in (
        "videos",
        "bbox_rows",
        "xml_entries",
        "missing_xml_entries",
        "missing_frames",
        "all_xml_missing_frames",
        "mixed_xml_presence_frames",
        "all_missing_frame_positive_points",
    ):
        print(f"  {name:36s}: {summary[name]}")
    print(f"  output_root                         : {output_root}")
    print("Stage 4 deleted-XML annotation audit passed.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of v5 saved frame_annotation/xml_path values "
            "against the current canonical VOC XML roots."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclusion_manifest", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, default=181)
    parser.add_argument("--expected_excluded_videos", type=int, default=1)
    parser.add_argument("--expected_missing_frames", type=int, default=-1)
    parser.add_argument("--expected_missing_xml_entries", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_audit(args)
    except Exception as exc:
        raise SystemExit(
            "Stage 4 deleted-XML annotation audit failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
