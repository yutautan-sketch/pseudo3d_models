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


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.analysis.validate_stage4_crop_quality_manifests import (
    CropQualityInvalidation,
    invalidation_fingerprint,
    load_crop_quality_invalidations,
)
from pseudo3d.annotation.apply_crop_quality_invalidations import (
    OUTPUT_TOKEN,
    SOURCE_TOKEN,
    apply_crop_quality_invalidations_to_h5,
)


SUMMARY_FIELDS = (
    "video_name",
    "status",
    "collected_status",
    "source_h5",
    "source_h5_sha256",
    "output_h5",
    "output_h5_sha256",
    "collected_h5",
    "collected_h5_sha256",
    "points",
    "positive_after",
    "removed_positive_points",
    "removed_ignore_points",
    "invalidated_frames",
    "removed_bbox_rows",
)


class CropQualityBatchError(RuntimeError):
    """Raised when the v6-to-v7 batch contract is inconsistent."""


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _single_source(root: Path, video: str) -> Path:
    matches = sorted((root / video).glob(f"*{SOURCE_TOKEN}.h5"))
    if len(matches) != 1:
        raise CropQualityBatchError(
            f"Expected one v6 source H5 for {video}, found {len(matches)}"
        )
    return matches[0].resolve()


def _output_name(source: Path) -> str:
    if SOURCE_TOKEN not in source.name:
        raise CropQualityBatchError(f"Source filename lacks v6 token: {source.name}")
    return source.name.replace(SOURCE_TOKEN, OUTPUT_TOKEN)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {name: row.get(name, "") for name in SUMMARY_FIELDS} for row in rows
    )
    _atomic_write(path, stream.getvalue().encode("utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _copy_or_verify(source: Path, destination: Path, *, overwrite: bool) -> tuple[str, str]:
    source_sha = file_sha256(source)
    if destination.exists():
        if destination.is_file() and file_sha256(destination) == source_sha:
            return "verified_existing", source_sha
        if not overwrite:
            raise CropQualityBatchError(
                f"Collected H5 differs from annotated H5: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if file_sha256(temporary) != source_sha:
            raise CropQualityBatchError(f"Temporary collected checksum mismatch: {source}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "copied", source_sha


def _load_fixed_summaries(
    *,
    validation_path: Path,
    exclusion_path: Path,
    train_manifest: Path,
    exclusion_manifest: Path,
    invalidation_manifest: Path,
    invalidations: Sequence[CropQualityInvalidation],
    expected_videos: int,
    expected_excluded: int,
    expected_stray: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    expected_validation = {
        "schema_version": 1,
        "status": "ok",
        "video_exclusions_sha256": file_sha256(exclusion_manifest),
        "crop_invalidations_sha256": file_sha256(invalidation_manifest),
        "crop_invalidation_fingerprint": invalidation_fingerprint(invalidations),
        "invalidated_frames": len(invalidations),
        "expected_stray_points_to_remove": expected_stray,
        "h5_files_written": 0,
        "input_files_unchanged": True,
    }
    for name, expected in expected_validation.items():
        if validation.get(name) != expected:
            raise CropQualityBatchError(
                f"Manifest-validation summary mismatch for {name}: "
                f"{validation.get(name)!r} != {expected!r}"
            )
    expected_exclusion = {
        "schema_version": 1,
        "status": "ok",
        "output_manifest_sha256": file_sha256(train_manifest),
        "exclusions_sha256": file_sha256(exclusion_manifest),
        "rows": expected_videos + expected_excluded,
        "enabled": expected_videos,
        "disabled": expected_excluded,
    }
    for name, expected in expected_exclusion.items():
        if exclusion.get(name) != expected:
            raise CropQualityBatchError(
                f"Exclusion summary mismatch for {name}: "
                f"{exclusion.get(name)!r} != {expected!r}"
            )
    return validation, exclusion


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    train_manifest = Path(args.train_manifest).resolve()
    exclusion_manifest = Path(args.exclusion_manifest).resolve()
    invalidation_manifest = Path(args.invalidation_manifest).resolve()
    validation_summary = Path(args.manifest_validation_summary).resolve()
    exclusion_summary = Path(args.exclusion_summary).resolve()
    source_root = Path(args.source_annotated_root).resolve()
    annotated_root = Path(args.output_annotated_root).resolve()
    collected_root = Path(args.output_collected_root).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    summary_json = Path(args.summary_json).resolve()
    for path, description in (
        (train_manifest, "train manifest"),
        (exclusion_manifest, "exclusion manifest"),
        (invalidation_manifest, "crop invalidation manifest"),
        (validation_summary, "manifest-validation summary"),
        (exclusion_summary, "exclusion summary"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"v6 source root not found: {source_root}")
    if any(
        _paths_overlap(left, right)
        for left, right in (
            (source_root, annotated_root),
            (source_root, collected_root),
            (annotated_root, collected_root),
        )
    ):
        raise CropQualityBatchError("Source/annotated/collected roots must not overlap")

    invalidations = load_crop_quality_invalidations(invalidation_manifest)
    if len(invalidations) != args.expected_invalidations:
        raise CropQualityBatchError(
            f"Invalidation count mismatch: {len(invalidations)} != {args.expected_invalidations}"
        )
    by_video: dict[str, list[CropQualityInvalidation]] = defaultdict(list)
    for item in invalidations:
        by_video[item.video_name].append(item)
    if len(by_video) != args.expected_affected_videos:
        raise CropQualityBatchError(
            f"Affected-video count mismatch: {len(by_video)} != {args.expected_affected_videos}"
        )
    _load_fixed_summaries(
        validation_path=validation_summary,
        exclusion_path=exclusion_summary,
        train_manifest=train_manifest,
        exclusion_manifest=exclusion_manifest,
        invalidation_manifest=invalidation_manifest,
        invalidations=invalidations,
        expected_videos=args.expected_videos,
        expected_excluded=args.expected_excluded_videos,
        expected_stray=args.expected_stray_ignore_points,
    )

    manifest_items = load_stage4_sweep_manifest(train_manifest)
    exclusions = load_exclusions(exclusion_manifest)
    excluded = {item.video_name for item in exclusions}
    disabled = {item.video_name for item in manifest_items if not item.enabled}
    if disabled != excluded:
        raise CropQualityBatchError(
            "Disabled videos differ from exclusion manifest: "
            f"{sorted(disabled)} != {sorted(excluded)}"
        )
    enabled = sorted(
        (item for item in manifest_items if item.enabled), key=lambda item: item.video_name
    )
    if len(enabled) != args.expected_videos or len(excluded) != args.expected_excluded_videos:
        raise CropQualityBatchError(
            f"Manifest counts differ: enabled={len(enabled)}, excluded={len(excluded)}"
        )
    enabled_names = {item.video_name for item in enabled}
    unknown = sorted(set(by_video) - enabled_names)
    if unknown:
        raise CropQualityBatchError(
            f"Crop invalidations reference excluded/unknown videos: {unknown}"
        )
    sources = {item.video_name: _single_source(source_root, item.video_name) for item in enabled}
    all_source_files = sorted(source_root.rglob(f"*{SOURCE_TOKEN}.h5"))
    if len(all_source_files) != args.expected_source_videos:
        raise CropQualityBatchError(
            f"v6 source inventory mismatch: {len(all_source_files)} != "
            f"{args.expected_source_videos}"
        )
    if len(set(sources.values())) != len(sources):
        raise CropQualityBatchError("Duplicate v6 source H5 path")
    manifest_video_names = {item.video_name for item in manifest_items}
    unexpected_source_dirs = sorted(
        path.parent.name
        for path in all_source_files
        if path.parent.name not in manifest_video_names
    )
    if unexpected_source_dirs:
        raise CropQualityBatchError(
            f"v6 source root has unknown video directories: {unexpected_source_dirs}"
        )

    annotated_root.mkdir(parents=True, exist_ok=True)
    collected_root.mkdir(parents=True, exist_ok=True)
    manifest_sha = file_sha256(invalidation_manifest)
    print("Stage 4 crop-quality invalidation full-dataset apply")
    print(f"  source root        : {source_root}")
    print(f"  annotated root     : {annotated_root}")
    print(f"  collected root     : {collected_root}")
    print(f"  enabled videos     : {len(enabled)}")
    print(f"  excluded videos    : {len(excluded)}")
    print(f"  invalidated frames : {len(invalidations)}")
    rows: list[dict[str, Any]] = []
    for index, manifest_item in enumerate(enabled, start=1):
        video = manifest_item.video_name
        source = sources[video]
        output = annotated_root / video / _output_name(source)
        result = apply_crop_quality_invalidations_to_h5(
            source_h5=source,
            output_h5=output,
            manifest_path=invalidation_manifest,
            manifest_sha256=manifest_sha,
            items=by_video.get(video, []),
            overwrite=bool(args.overwrite),
            skip_existing=bool(args.skip_existing),
        )
        collected = collected_root / output.name
        collected_status, collected_sha = _copy_or_verify(
            output, collected, overwrite=bool(args.overwrite)
        )
        if result["output_h5_sha256"] != collected_sha:
            raise CropQualityBatchError(f"Annotated/collected checksum mismatch: {video}")
        row = {
            **result,
            "video_name": video,
            "collected_status": collected_status,
            "collected_h5": str(collected),
            "collected_h5_sha256": collected_sha,
        }
        rows.append(row)
        print(
            f"  [{index}/{len(enabled)}] [OK] {video}: "
            f"status={result['status']}, invalidated={result['invalidated_frames']}, "
            f"removed_ignore={result['removed_ignore_points']}"
        )

    totals = {
        "invalidated_frames": sum(int(row["invalidated_frames"]) for row in rows),
        "removed_bbox_rows": sum(int(row["removed_bbox_rows"]) for row in rows),
        "removed_positive_points": sum(int(row["removed_positive_points"]) for row in rows),
        "removed_ignore_points": sum(int(row["removed_ignore_points"]) for row in rows),
    }
    expected_totals = {
        "invalidated_frames": args.expected_invalidations,
        "removed_bbox_rows": args.expected_removed_bbox_rows,
        "removed_positive_points": 0,
        "removed_ignore_points": args.expected_stray_ignore_points,
    }
    if totals != expected_totals:
        raise CropQualityBatchError(f"Applied totals differ: {totals} != {expected_totals}")
    annotated_files = sorted(annotated_root.rglob(f"*{OUTPUT_TOKEN}.h5"))
    collected_files = sorted(collected_root.glob(f"*{OUTPUT_TOKEN}.h5"))
    if len(annotated_files) != len(enabled) or len(collected_files) != len(enabled):
        raise CropQualityBatchError("Final v7 file count differs from enabled-video count")
    if any((annotated_root / video).exists() for video in excluded):
        raise CropQualityBatchError("Excluded video unexpectedly exists in v7 output")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "source_teacher_version": SOURCE_TOKEN,
        "teacher_version": OUTPUT_TOKEN,
        "videos": len(rows),
        "source_videos": len(all_source_files),
        "excluded_videos": sorted(excluded),
        "affected_videos": len(by_video),
        **totals,
        "points": sum(int(row["points"]) for row in rows),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
        "collected_status_counts": dict(
            sorted(Counter(str(row["collected_status"]) for row in rows).items())
        ),
        "train_manifest_sha256": file_sha256(train_manifest),
        "exclusion_manifest_sha256": file_sha256(exclusion_manifest),
        "invalidation_manifest_sha256": manifest_sha,
        "invalidation_fingerprint": invalidation_fingerprint(invalidations),
        "manifest_validation_summary_sha256": file_sha256(validation_summary),
        "exclusion_summary_sha256": file_sha256(exclusion_summary),
        "annotated_root": str(annotated_root),
        "collected_root": str(collected_root),
        "input_files_unchanged": True,
        "failure_rows": 0,
    }
    _write_csv(summary_csv, rows)
    _write_json(summary_json, summary)
    print("Stage 4 crop-quality invalidation full-dataset apply passed.")
    print(f"  videos             : {len(rows)}")
    print(f"  excluded videos    : {len(excluded)}")
    print(f"  invalidated frames : {totals['invalidated_frames']}")
    print(f"  removed BBoxes     : {totals['removed_bbox_rows']}")
    print(f"  removed ignore     : {totals['removed_ignore_points']}")
    print(f"  summary CSV        : {summary_csv}")
    print(f"  summary JSON       : {summary_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply fixed crop-quality exclusions/invalidations to Stage 4 v6 H5 files."
    )
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--exclusion_manifest", type=Path, required=True)
    parser.add_argument("--invalidation_manifest", type=Path, required=True)
    parser.add_argument("--manifest_validation_summary", type=Path, required=True)
    parser.add_argument("--exclusion_summary", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--output_annotated_root", type=Path, required=True)
    parser.add_argument("--output_collected_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, required=True)
    parser.add_argument("--expected_source_videos", type=int, required=True)
    parser.add_argument("--expected_excluded_videos", type=int, required=True)
    parser.add_argument("--expected_invalidations", type=int, required=True)
    parser.add_argument("--expected_affected_videos", type=int, required=True)
    parser.add_argument("--expected_stray_ignore_points", type=int, required=True)
    parser.add_argument("--expected_removed_bbox_rows", type=int, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--skip_existing", action="store_true")
    modes.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    try:
        run_batch(build_parser().parse_args())
    except Exception as exc:
        raise SystemExit(
            "Stage 4 crop-quality invalidation batch failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
