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


from pseudo3d.analysis.build_stage4_deleted_xml_invalidation_manifest import (
    XmlFrameInvalidation,
    invalidation_fingerprint,
    load_xml_invalidation_manifest,
)
from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.annotation.apply_deleted_xml_invalidations import (
    OUTPUT_TOKEN,
    SOURCE_TOKEN,
    apply_invalidations_to_h5,
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
    "positive_before",
    "positive_after",
    "removed_positive_points",
    "removed_ignore_points",
    "invalidated_frames",
    "removed_bbox_rows",
)


class XmlInvalidationBatchError(RuntimeError):
    """Raised when the v5-to-v6 batch contract is inconsistent."""


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _single_source(root: Path, video: str) -> Path:
    matches = sorted((root / video).glob(f"*{SOURCE_TOKEN}.h5"))
    if len(matches) != 1:
        raise XmlInvalidationBatchError(
            f"Expected one v5 source H5 for {video}, found {len(matches)}"
        )
    return matches[0].resolve()


def _output_name(source: Path) -> str:
    if SOURCE_TOKEN not in source.name:
        raise XmlInvalidationBatchError(
            f"Source filename lacks fixed v5 token: {source.name}"
        )
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
    writer = csv.DictWriter(
        stream, fieldnames=list(SUMMARY_FIELDS), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(
        {name: row.get(name, "") for name in SUMMARY_FIELDS} for row in rows
    )
    _atomic_write(path, stream.getvalue().encode("utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8"),
    )


def _copy_or_verify(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> tuple[str, str]:
    source_sha = file_sha256(source)
    if destination.exists():
        if not destination.is_file():
            raise XmlInvalidationBatchError(
                f"Collected destination is not a file: {destination}"
            )
        if file_sha256(destination) == source_sha:
            return "verified_existing", source_sha
        if not overwrite:
            raise XmlInvalidationBatchError(
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
            raise XmlInvalidationBatchError(
                f"Temporary collected copy checksum mismatch: {source}"
            )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "copied", source_sha


def _validate_step2_summary(
    *,
    path: Path,
    train_manifest: Path,
    exclusion_manifest: Path,
    invalidation_manifest: Path,
    invalidations: Sequence[XmlFrameInvalidation],
    expected_invalidations: int,
    expected_affected_videos: int,
    expected_positive_points: int,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Step 2 summary not found: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "ok",
        "teacher_version": SOURCE_TOKEN,
        "invalidations": expected_invalidations,
        "videos": expected_affected_videos,
        "expected_positive_points_to_remove": expected_positive_points,
        "mixed_xml_presence_frames": 0,
        "failure_rows": 0,
        "failure_rows": 0,
        "train_manifest_sha256": file_sha256(train_manifest),
        "exclusion_manifest_sha256": file_sha256(exclusion_manifest),
        "output_manifest_sha256": file_sha256(invalidation_manifest),
        "invalidation_fingerprint": invalidation_fingerprint(invalidations),
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise XmlInvalidationBatchError(
                f"Step 2 summary mismatch for {name}: "
                f"{summary.get(name)!r} != {value!r}"
            )
    if not bool(summary.get("input_files_unchanged")):
        raise XmlInvalidationBatchError("Step 2 inputs were not unchanged")
    if int(summary.get("h5_files_written", -1)) != 0:
        raise XmlInvalidationBatchError("Step 2 unexpectedly wrote H5 files")
    return summary


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    train_manifest = Path(args.train_manifest).resolve()
    exclusion_manifest = Path(args.exclusion_manifest).resolve()
    invalidation_manifest = Path(args.invalidation_manifest).resolve()
    step2_summary = Path(args.step2_summary).resolve()
    source_root = Path(args.source_annotated_root).resolve()
    annotated_root = Path(args.output_annotated_root).resolve()
    collected_root = Path(args.output_collected_root).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    summary_json = Path(args.summary_json).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"v5 source root not found: {source_root}")
    if _paths_overlap(source_root, annotated_root) or _paths_overlap(
        source_root, collected_root
    ):
        raise XmlInvalidationBatchError("Source and output roots must not overlap")
    if _paths_overlap(annotated_root, collected_root):
        raise XmlInvalidationBatchError(
            "Annotated and collected roots must not overlap"
        )
    for path, name in (
        (annotated_root, "annotated root"),
        (collected_root, "collected root"),
    ):
        if path.exists() and path.is_symlink():
            raise XmlInvalidationBatchError(f"Output {name} must not be a symlink")

    invalidations = load_xml_invalidation_manifest(invalidation_manifest)
    if len(invalidations) != int(args.expected_invalidations):
        raise XmlInvalidationBatchError(
            f"Invalidation count mismatch: {len(invalidations)} != "
            f"{args.expected_invalidations}"
        )
    invalidations_by_video: dict[str, list[XmlFrameInvalidation]] = defaultdict(list)
    for item in invalidations:
        invalidations_by_video[item.video_name].append(item)
    if len(invalidations_by_video) != int(args.expected_affected_videos):
        raise XmlInvalidationBatchError(
            f"Affected-video count mismatch: {len(invalidations_by_video)} != "
            f"{args.expected_affected_videos}"
        )
    fixed_summary = _validate_step2_summary(
        path=step2_summary,
        train_manifest=train_manifest,
        exclusion_manifest=exclusion_manifest,
        invalidation_manifest=invalidation_manifest,
        invalidations=invalidations,
        expected_invalidations=int(args.expected_invalidations),
        expected_affected_videos=int(args.expected_affected_videos),
        expected_positive_points=int(args.expected_positive_points),
    )

    manifest_items = load_stage4_sweep_manifest(train_manifest)
    exclusions = load_exclusions(exclusion_manifest)
    excluded = {item.video_name for item in exclusions}
    manifest_by_video = {item.video_name: item for item in manifest_items}
    unknown_exclusions = sorted(excluded - set(manifest_by_video))
    if unknown_exclusions:
        raise XmlInvalidationBatchError(
            f"Exclusion manifest references unknown videos: {unknown_exclusions}"
        )
    disabled = {item.video_name for item in manifest_items if not item.enabled}
    if disabled != excluded:
        raise XmlInvalidationBatchError(
            "Disabled train-manifest videos differ from exclusion manifest: "
            f"disabled={sorted(disabled)}, excluded={sorted(excluded)}"
        )
    enabled = sorted(
        (
            item
            for item in manifest_items
            if item.enabled and item.video_name not in excluded
        ),
        key=lambda item: item.video_name,
    )
    if len(enabled) != int(args.expected_videos):
        raise XmlInvalidationBatchError(
            f"Enabled-video count mismatch: {len(enabled)} != {args.expected_videos}"
        )
    if len(excluded) != int(args.expected_excluded_videos):
        raise XmlInvalidationBatchError(
            f"Excluded-video count mismatch: {len(excluded)} != "
            f"{args.expected_excluded_videos}"
        )
    enabled_names = {item.video_name for item in enabled}
    unknown = sorted(set(invalidations_by_video) - enabled_names)
    if unknown:
        raise XmlInvalidationBatchError(
            f"Invalidations reference disabled/unknown videos: {unknown}"
        )
    sources = {item.video_name: _single_source(source_root, item.video_name) for item in enabled}
    fixed_source_hashes = fixed_summary.get("v5_candidate_h5_sha256")
    if not isinstance(fixed_source_hashes, dict):
        raise XmlInvalidationBatchError(
            "Step 2 summary lacks v5_candidate_h5_sha256 mapping"
        )
    expected_fixed_source_hashes = {
        sources[video].name: file_sha256(sources[video])
        for video in sorted(invalidations_by_video)
    }
    if fixed_source_hashes != expected_fixed_source_hashes:
        raise XmlInvalidationBatchError(
            "Affected v5 H5 checksums differ from the fixed Step 2 evidence"
        )
    all_source_files = sorted(
        path
        for path in source_root.rglob(f"*{SOURCE_TOKEN}.h5")
        if path.is_file()
    )
    if len(all_source_files) != len(enabled):
        raise XmlInvalidationBatchError(
            f"v5 source inventory mismatch: {len(all_source_files)} != {len(enabled)}"
        )
    if len(set(sources.values())) != len(sources):
        raise XmlInvalidationBatchError("Duplicate v5 source H5 path")
    for video in excluded:
        if (source_root / video).exists():
            raise XmlInvalidationBatchError(
                f"Excluded video exists in v5 source root: {video}"
            )

    annotated_root.mkdir(parents=True, exist_ok=True)
    collected_root.mkdir(parents=True, exist_ok=True)
    manifest_sha = file_sha256(invalidation_manifest)
    print("Stage 4 deleted-XML invalidation full-dataset apply")
    print(f"  source root        : {source_root}")
    print(f"  annotated root     : {annotated_root}")
    print(f"  collected root     : {collected_root}")
    print(f"  videos             : {len(enabled)}")
    print(f"  affected videos    : {len(invalidations_by_video)}")
    print(f"  invalidated frames : {len(invalidations)}")
    rows: list[dict[str, Any]] = []
    for index, manifest_item in enumerate(enabled, start=1):
        video = manifest_item.video_name
        source_h5 = sources[video]
        filename = _output_name(source_h5)
        output_h5 = annotated_root / video / filename
        video_invalidations = invalidations_by_video.get(video, [])
        result = apply_invalidations_to_h5(
            source_h5=source_h5,
            output_h5=output_h5,
            manifest_path=invalidation_manifest,
            manifest_sha256=manifest_sha,
            items=video_invalidations,
            overwrite=bool(args.overwrite),
            skip_existing=bool(args.skip_existing),
        )
        collected_h5 = collected_root / filename
        collected_status, collected_sha = _copy_or_verify(
            output_h5,
            collected_h5,
            overwrite=bool(args.overwrite),
        )
        if result["output_h5_sha256"] != collected_sha:
            raise XmlInvalidationBatchError(
                f"Annotated/collected checksum mismatch: {video}"
            )
        row = {
            **result,
            "video_name": video,
            "collected_status": collected_status,
            "collected_h5": str(collected_h5),
            "collected_h5_sha256": collected_sha,
        }
        rows.append(row)
        print(
            f"  [{index}/{len(enabled)}] [OK] {video}: "
            f"status={result['status']}, invalidated={result['invalidated_frames']}, "
            f"removed_positive={result['removed_positive_points']}"
        )

    total_removed_positive = sum(
        int(row["removed_positive_points"]) for row in rows
    )
    total_removed_bbox = sum(int(row["removed_bbox_rows"]) for row in rows)
    total_invalidated = sum(int(row["invalidated_frames"]) for row in rows)
    if total_removed_positive != int(args.expected_positive_points):
        raise XmlInvalidationBatchError(
            f"Removed-positive total mismatch: {total_removed_positive} != "
            f"{args.expected_positive_points}"
        )
    if total_removed_bbox != int(args.expected_removed_bbox_rows):
        raise XmlInvalidationBatchError(
            f"Removed-BBox total mismatch: {total_removed_bbox} != "
            f"{args.expected_removed_bbox_rows}"
        )
    if total_invalidated != len(invalidations):
        raise XmlInvalidationBatchError(
            "Applied invalidation-frame count differs from manifest"
        )
    annotated_files = sorted(annotated_root.rglob(f"*{OUTPUT_TOKEN}.h5"))
    collected_files = sorted(collected_root.glob(f"*{OUTPUT_TOKEN}.h5"))
    if len(annotated_files) != len(enabled) or len(collected_files) != len(enabled):
        raise XmlInvalidationBatchError(
            "Final v6 file count differs from enabled-video count"
        )

    status_counts = Counter(str(row["status"]) for row in rows)
    collected_counts = Counter(str(row["collected_status"]) for row in rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "source_teacher_version": SOURCE_TOKEN,
        "teacher_version": OUTPUT_TOKEN,
        "videos": len(rows),
        "affected_videos": len(invalidations_by_video),
        "invalidated_frames": total_invalidated,
        "removed_bbox_rows": total_removed_bbox,
        "removed_positive_points": total_removed_positive,
        "removed_ignore_points": sum(
            int(row["removed_ignore_points"]) for row in rows
        ),
        "points": sum(int(row["points"]) for row in rows),
        "status_counts": dict(sorted(status_counts.items())),
        "collected_status_counts": dict(sorted(collected_counts.items())),
        "train_manifest_sha256": file_sha256(train_manifest),
        "exclusion_manifest_sha256": file_sha256(exclusion_manifest),
        "invalidation_manifest_sha256": manifest_sha,
        "invalidation_fingerprint": invalidation_fingerprint(invalidations),
        "step2_summary_sha256": file_sha256(step2_summary),
        "annotated_root": str(annotated_root),
        "collected_root": str(collected_root),
        "input_files_unchanged": True,
        "failure_rows": 0,
    }
    _write_csv(summary_csv, rows)
    _write_json(summary_json, summary)
    print("Stage 4 deleted-XML invalidation full-dataset apply passed.")
    print(f"  videos             : {len(rows)}")
    print(f"  invalidated frames : {total_invalidated}")
    print(f"  removed BBoxes     : {total_removed_bbox}")
    print(f"  removed positive   : {total_removed_positive}")
    print(f"  summary CSV        : {summary_csv}")
    print(f"  summary JSON       : {summary_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a fixed deleted-XML frame manifest to all Stage 4 v5 H5 "
            "files and collect byte-identical v6 outputs."
        )
    )
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--exclusion_manifest", type=Path, required=True)
    parser.add_argument("--invalidation_manifest", type=Path, required=True)
    parser.add_argument("--step2_summary", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--output_annotated_root", type=Path, required=True)
    parser.add_argument("--output_collected_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, required=True)
    parser.add_argument("--expected_excluded_videos", type=int, required=True)
    parser.add_argument("--expected_invalidations", type=int, required=True)
    parser.add_argument("--expected_affected_videos", type=int, required=True)
    parser.add_argument("--expected_positive_points", type=int, required=True)
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
            "Stage 4 deleted-XML invalidation batch failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
