from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.preflight_stage4_phase5_fullvideo_cvat_review import (
    verify_source_checksum_rows,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256
from pseudo3d.annotation.stage4_manual_review import (
    ManualReviewError,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    validate_cvat_segmentation_mask_zip,
)


VALIDATION_SCHEMA_VERSION = 1


def _read_json(path: Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ManualReviewError(f"JSON root must be an object: {path}")
    return value


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ManualReviewError(f"{name} must be boolean: {value!r}")


def _target_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["video_name"]),
        int(row["frame_order"]),
        int(row["frame_index"]),
        int(row["bbox_index"]),
    )


def _frame_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["video_name"]),
        int(row["frame_order"]),
        int(row["frame_index"]),
    )


def _video_key(row: Mapping[str, Any]) -> str:
    return str(row["video_name"])


def _unique_map(
    rows: Sequence[Mapping[str, Any]], key_function: Any, name: str
) -> dict[Any, Mapping[str, Any]]:
    result: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        key = key_function(row)
        if key in result:
            raise ManualReviewError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def _verify_checksum_file(
    root: Path, checksum_csv: Path, *, hash_cache: dict[Path, str]
) -> int:
    rows = read_csv_rows(checksum_csv)
    expected_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"checksums.csv", "export_summary.json"}
    }
    listed = {str(row.get("relative_path", "")) for row in rows}
    if "" in listed or len(listed) != len(rows):
        raise ManualReviewError(f"Invalid/duplicate checksum rows: {checksum_csv}")
    if listed != expected_files:
        raise ManualReviewError(
            f"Checksum member mismatch for {root}: "
            f"missing={sorted(expected_files - listed)[:3]}, "
            f"extra={sorted(listed - expected_files)[:3]}"
        )
    for row in rows:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ManualReviewError(f"Unsafe checksum path: {relative}")
        path = (root / relative).resolve()
        if not path.is_file() or int(row["bytes"]) != int(path.stat().st_size):
            raise ManualReviewError(f"Checksum size mismatch: {path}")
        actual = hash_cache.get(path)
        if actual is None:
            actual = file_sha256(path)
            hash_cache[path] = actual
        if actual != str(row["sha256"]):
            raise ManualReviewError(f"Checksum SHA-256 mismatch: {path}")
    return len(rows)


def _verify_context_masks(
    *, video_root: Path, frame_rows: Sequence[Mapping[str, Any]]
) -> int:
    checked = 0
    for row in frame_rows:
        if str(row.get("frame_role", "")) != "context_only":
            continue
        mask_path = video_root / str(row["mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
            raise ManualReviewError(f"Invalid context mask: {mask_path}")
        if np.any(mask != 0):
            raise ManualReviewError(
                f"Context-only initial mask is not all-background: {mask_path}"
            )
        checked += 1
    return checked


def _transfer_paths(review_root: Path) -> list[Path]:
    top_level = sorted(
        path
        for path in review_root.iterdir()
        if path.is_file()
        and path.name
        not in {
            "mac_transfer_manifest.csv",
            "mac_transfer_summary.json",
            "package_validation_summary.json",
        }
    )
    video_files = sorted(
        path for path in (review_root / "videos").rglob("*") if path.is_file()
    )
    paths = top_level + video_files
    if len({str(path.resolve()) for path in paths}) != len(paths):
        raise ManualReviewError("Transfer file list contains duplicates")
    return paths


def _write_transfer_manifest(
    *, review_root: Path, hash_cache: dict[Path, str], overwrite: bool
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    manifest_path = review_root / "mac_transfer_manifest.csv"
    summary_path = review_root / "mac_transfer_summary.json"
    if (manifest_path.exists() or summary_path.exists()) and not overwrite:
        raise FileExistsError(
            "Mac transfer artifacts already exist; pass --overwrite_transfer_manifest"
        )
    rows: list[dict[str, Any]] = []
    for path in _transfer_paths(review_root):
        resolved = path.resolve()
        digest = hash_cache.get(resolved)
        if digest is None:
            digest = file_sha256(resolved)
            hash_cache[resolved] = digest
        rows.append(
            {
                "relative_path": path.relative_to(review_root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": digest,
            }
        )
    write_csv_rows(
        manifest_path,
        rows,
        fieldnames=("relative_path", "bytes", "sha256"),
    )
    summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "ok",
        "transfer_files": len(rows),
        "transfer_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "manifest": manifest_path.name,
        "manifest_sha256": file_sha256(manifest_path),
        "scope": "top-level metadata plus videos/; root staging images/masks/overlays/cvat excluded",
        "control_files_not_listed_in_manifest": [
            "mac_transfer_manifest.csv",
            "mac_transfer_summary.json",
            "package_validation_summary.json",
        ],
    }
    write_json(summary_path, summary)
    return rows, summary


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    review_root = Path(args.review_root).resolve()
    preflight_root = Path(args.preflight_root).resolve()
    if not review_root.is_dir() or not preflight_root.is_dir():
        raise FileNotFoundError("review_root and preflight_root must be directories")

    preflight_summary = _read_json(preflight_root / "preflight_summary.json")
    export_summary = _read_json(review_root / "export_summary.json")
    partition_summary = _read_json(review_root / "partition_summary.json")
    for name, summary in (
        ("preflight", preflight_summary),
        ("export", export_summary),
        ("partition", partition_summary),
    ):
        if str(summary.get("status", "")) != "ok" or int(
            summary.get("failure_rows", 0)
        ) != 0:
            raise ManualReviewError(f"{name} summary is not complete and failure-free")
    if str(export_summary.get("export_mode", "")) != "full_video":
        raise ManualReviewError("Review export is not full_video mode")

    expected_videos = int(args.expected_selected_videos)
    expected_frames = int(preflight_summary["total_frames"])
    expected_target_frames = int(preflight_summary["target_frames"])
    expected_targets = int(args.expected_review_bboxes)
    expected_crop_counts = {
        str(key): int(value)
        for key, value in preflight_summary["crop_status_counts"].items()
    }
    expected_decision_counts = {
        "auto_accept": int(args.expected_auto_accept),
        "auto_refine": int(args.expected_auto_refine),
        "manual_review": int(args.expected_manual_review),
    }
    for actual, expected, name in (
        (int(preflight_summary["selected_videos"]), expected_videos, "preflight videos"),
        (int(preflight_summary["target_bboxes"]), expected_targets, "preflight targets"),
        (int(export_summary["selected_videos"]), expected_videos, "export videos"),
        (int(export_summary["review_frames"]), expected_frames, "export frames"),
        (
            int(export_summary["target_review_frames"]),
            expected_target_frames,
            "export target frames",
        ),
        (
            int(export_summary["selected_review_bboxes"]),
            expected_targets,
            "export targets",
        ),
        (int(partition_summary["video_packages"]), expected_videos, "video packages"),
        (int(partition_summary["exported_frames"]), expected_frames, "partition frames"),
        (
            int(partition_summary["case_frames"]),
            expected_target_frames,
            "case frames",
        ),
        (int(partition_summary["review_bboxes"]), expected_targets, "review index"),
    ):
        if actual != expected:
            raise ManualReviewError(f"{name} mismatch: {actual} != {expected}")
    if {
        str(key): int(value)
        for key, value in export_summary["crop_status_counts"].items()
    } != expected_crop_counts:
        raise ManualReviewError("Export/preflight crop status counts differ")
    if {
        str(key): int(value)
        for key, value in export_summary["phase3_decision_counts"].items()
    } != expected_decision_counts:
        raise ManualReviewError("Export decision counts differ from fixed contract")
    if str(partition_summary.get("progress_csv", "")) != "video_progress.csv":
        raise ManualReviewError("Full-video partition does not use video_progress.csv")

    preflight_targets = read_csv_rows(preflight_root / "target_bboxes.csv")
    export_bboxes = read_csv_rows(review_root / "review_bboxes.csv")
    export_targets = [
        row for row in export_bboxes if _as_bool(row.get("review_target", False), "review_target")
    ]
    decisions = read_csv_rows(review_root / "crop_review_decisions.csv")
    review_index = read_csv_rows(review_root / "review_index.csv")
    preflight_map = _unique_map(preflight_targets, _target_key, "preflight target")
    export_map = _unique_map(export_targets, _target_key, "export target")
    decision_map = _unique_map(decisions, _target_key, "crop decision")
    index_map = _unique_map(review_index, _target_key, "review index")
    if not (
        set(preflight_map)
        == set(export_map)
        == set(decision_map)
        == set(index_map)
    ):
        raise ManualReviewError("Target BBox key sets differ across preflight/export/index")
    if len(preflight_map) != expected_targets:
        raise ManualReviewError("Target BBox key count differs from fixed contract")
    for key in sorted(preflight_map):
        expected = preflight_map[key]
        exported = export_map[key]
        decision = decision_map[key]
        indexed = index_map[key]
        if str(exported["phase3_decision"]) != str(expected["phase3_decision"]):
            raise ManualReviewError(f"Phase 3 decision mismatch: {key}")
        if str(exported["crop_status"]) != str(expected["crop_status"]):
            raise ManualReviewError(f"Crop status mismatch: {key}")
        if not np.isclose(
            float(exported["visible_fraction"]),
            float(expected["visible_fraction"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ManualReviewError(f"Visible fraction mismatch: {key}")
        if str(decision["review_disposition"]) != "pending":
            raise ManualReviewError(f"New crop decision is not pending: {key}")
        if str(indexed["review_status"]) != "pending":
            raise ManualReviewError(f"New review index is not pending: {key}")
        for field in ("phase3_decision", "crop_status", "visible_fraction"):
            if str(indexed[field]) != str(decision[field]):
                raise ManualReviewError(
                    f"Review index/crop decision field mismatch: {key} {field}"
                )

    preflight_videos = read_csv_rows(preflight_root / "video_summary.csv")
    progress_rows = read_csv_rows(review_root / "video_progress.csv")
    preflight_video_map = _unique_map(
        preflight_videos, _video_key, "preflight video"
    )
    progress_map = _unique_map(progress_rows, _video_key, "progress video")
    if len(preflight_video_map) != expected_videos or set(preflight_video_map) != set(
        progress_map
    ):
        raise ManualReviewError("Video set differs between preflight and export progress")

    hash_cache: dict[Path, str] = {}
    total_checksum_rows = 0
    total_context_frames = 0
    total_zip_bytes = 0
    print("Stage 4 Phase 5 full-video package validation (read-only package check)")
    print(f"  review_root     : {review_root}")
    print(f"  preflight_root  : {preflight_root}")
    print(f"  videos          : {expected_videos}")
    print(f"  frames          : {expected_frames}")
    print(f"  target BBoxes   : {expected_targets}")

    for number, video_name in enumerate(sorted(preflight_video_map), start=1):
        expected_video = preflight_video_map[video_name]
        progress = progress_map[video_name]
        video_root = review_root / "videos" / video_name
        video_summary = _read_json(video_root / "export_summary.json")
        if str(video_summary.get("status", "")) != "ok" or str(
            video_summary.get("export_mode", "")
        ) != "full_video":
            raise ManualReviewError(f"Invalid video summary: {video_name}")
        expected_video_frames = int(expected_video["frames"])
        expected_video_targets = int(expected_video["target_bboxes"])
        expected_video_target_frames = int(expected_video["target_frames"])
        for actual, expected, name in (
            (int(progress["review_frames"]), expected_video_frames, "progress frames"),
            (
                int(progress["target_review_frames"]),
                expected_video_target_frames,
                "progress target frames",
            ),
            (int(progress["review_bboxes"]), expected_video_targets, "progress targets"),
            (int(video_summary["review_frames"]), expected_video_frames, "summary frames"),
            (
                int(video_summary["selected_review_bboxes"]),
                expected_video_targets,
                "summary targets",
            ),
        ):
            if actual != expected:
                raise ManualReviewError(
                    f"{video_name} {name} mismatch: {actual} != {expected}"
                )

        frame_rows = read_csv_rows(video_root / "review_frames.csv")
        if len(frame_rows) != expected_video_frames:
            raise ManualReviewError(f"Frame CSV count mismatch: {video_name}")
        _unique_map(frame_rows, _frame_key, "video frame")
        ordered = sorted(frame_rows, key=lambda row: int(row["frame_order"]))
        if [int(row["frame_order"]) for row in ordered] != list(
            range(expected_video_frames)
        ):
            raise ManualReviewError(f"Non-contiguous frame_order: {video_name}")
        expected_stems = {str(row["frame_stem"]) for row in frame_rows}
        for directory in ("images", "masks", "overlays"):
            stems = {
                path.stem for path in (video_root / directory).glob("*.png") if path.is_file()
            }
            if stems != expected_stems:
                raise ManualReviewError(
                    f"{video_name} {directory} stems differ from frame CSV"
                )
        context_count = _verify_context_masks(
            video_root=video_root, frame_rows=frame_rows
        )
        total_context_frames += context_count
        case_stems = {
            path.name
            for path in (video_root / "cases").iterdir()
            if path.is_dir()
        }
        target_frame_stems = {
            str(row["frame_stem"])
            for row in frame_rows
            if _as_bool(row.get("review_target", False), "review_target")
        }
        if case_stems != target_frame_stems or len(case_stems) != expected_video_target_frames:
            raise ManualReviewError(f"Target case directory mismatch: {video_name}")
        video_decisions = read_csv_rows(video_root / "crop_review_decisions.csv")
        video_decision_map = _unique_map(
            video_decisions, _target_key, "video crop decision"
        )
        expected_video_decisions = {
            key: row for key, row in decision_map.items() if key[0] == video_name
        }
        if set(video_decision_map) != set(expected_video_decisions):
            raise ManualReviewError(f"Video decision CSV mismatch: {video_name}")
        for key, video_decision in video_decision_map.items():
            root_decision = expected_video_decisions[key]
            for field in (
                "phase3_decision",
                "crop_status",
                "visible_fraction",
                "review_disposition",
            ):
                if str(video_decision[field]) != str(root_decision[field]):
                    raise ManualReviewError(
                        f"Video/root decision field mismatch: {key} {field}"
                    )

        zip_path = video_root / "cvat" / "annotations_segmentation_mask_1_1.zip"
        zip_validation = validate_cvat_segmentation_mask_zip(
            zip_path,
            label_name="femur",
            images_dir=video_root / "images",
            reference_masks_dir=video_root / "masks",
        )
        if (
            zip_validation.images != expected_video_frames
            or zip_validation.masks != expected_video_frames
            or zip_validation.reference_pixel_mismatches != 0
        ):
            raise ManualReviewError(f"CVAT ZIP contract mismatch: {video_name}")
        total_zip_bytes += int(zip_path.stat().st_size)
        total_checksum_rows += _verify_checksum_file(
            video_root,
            video_root / "checksums.csv",
            hash_cache=hash_cache,
        )
        print(
            f"  [{number}/{expected_videos}] [OK] {video_name}: "
            f"frames={expected_video_frames}, targets={expected_video_targets}, "
            f"zip_bytes={zip_path.stat().st_size}"
        )

    source_manifest = preflight_root / "source_checksums.csv"
    expected_source_rows = read_csv_rows(source_manifest)
    current_source_rows: list[dict[str, Any]] = []
    print(f"  verifying source SHA-256: {len(expected_source_rows)} files")
    for row in expected_source_rows:
        path = Path(str(row["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        current_source_rows.append(
            {
                "video_name": row["video_name"],
                "role": row["role"],
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    verify_source_checksum_rows(current_source_rows, source_manifest)
    print("  [OK] source H5/XML checksums match preflight")

    transfer_rows, transfer_summary = _write_transfer_manifest(
        review_root=review_root,
        hash_cache=hash_cache,
        overwrite=bool(args.overwrite_transfer_manifest),
    )
    summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "ok",
        "videos_checked": expected_videos,
        "frames_checked": expected_frames,
        "target_frames_checked": expected_target_frames,
        "target_bboxes_checked": expected_targets,
        "context_masks_checked": total_context_frames,
        "video_checksum_rows": total_checksum_rows,
        "cvat_zip_bytes": total_zip_bytes,
        "source_checksum_rows": len(expected_source_rows),
        "source_files_unchanged": True,
        "transfer_files": len(transfer_rows),
        "transfer_bytes": int(transfer_summary["transfer_bytes"]),
        "transfer_manifest_sha256": transfer_summary["manifest_sha256"],
        "failure_rows": 0,
    }
    write_json(review_root / "package_validation_summary.json", summary)
    print("\nFull-video package validation summary")
    print(f"  videos_checked        : {summary['videos_checked']}")
    print(f"  frames_checked        : {summary['frames_checked']}")
    print(f"  target_bboxes_checked : {summary['target_bboxes_checked']}")
    print(f"  context_masks_checked : {summary['context_masks_checked']}")
    print(f"  video_checksum_rows   : {summary['video_checksum_rows']}")
    print(f"  source_checksum_rows  : {summary['source_checksum_rows']}")
    print("  source_files_unchanged: true")
    print(f"  transfer_files        : {summary['transfer_files']}")
    print(f"  transfer_bytes        : {summary['transfer_bytes']}")
    print("Stage 4 Phase 5 full-video package validation passed.")
    return {"summary": summary, "transfer_rows": transfer_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the generated Stage 4 Phase 5 full-video CVAT packages and "
            "write a Mac transfer manifest."
        )
    )
    parser.add_argument("--review_root", type=Path, required=True)
    parser.add_argument("--preflight_root", type=Path, required=True)
    parser.add_argument("--expected_selected_videos", type=int, default=60)
    parser.add_argument("--expected_review_bboxes", type=int, default=129)
    parser.add_argument("--expected_auto_accept", type=int, default=31)
    parser.add_argument("--expected_auto_refine", type=int, default=35)
    parser.add_argument("--expected_manual_review", type=int, default=63)
    parser.add_argument("--overwrite_transfer_manifest", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "expected_selected_videos",
        "expected_review_bboxes",
        "expected_auto_accept",
        "expected_auto_refine",
        "expected_manual_review",
    ):
        if int(getattr(args, name)) < 0:
            raise SystemExit(f"--{name} must be >= 0")
    try:
        run_validation(args)
    except Exception as exc:
        raise SystemExit(
            f"Full-video package validation failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
