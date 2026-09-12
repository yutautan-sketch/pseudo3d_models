from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.build_stage4_exclusion_manifest import load_exclusions
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    file_sha256,
    load_stage4_sweep_manifest,
)
from pseudo3d.batch.annotation.batch_import_stage4_phase5_fullvideo_cvat import (
    AUTHORITATIVE_LABEL_AUTHORITY,
    AUTHORITATIVE_OUTPUT_TOKEN,
    _resolve_source,
    _snapshot_contract,
)


class AcceptanceError(RuntimeError):
    """Raised when the versioned v5 build fails an acceptance contract."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _single_h5(root: Path, video: str) -> Path:
    matches = sorted(
        (Path(root) / video).glob(f"*{AUTHORITATIVE_OUTPUT_TOKEN}.h5")
    )
    if len(matches) != 1:
        raise AcceptanceError(
            f"Expected one v5 H5 for {video}, found {len(matches)}: {root}"
        )
    return matches[0].resolve()


def _single_collected_h5(root: Path, video: str) -> Path:
    matches = sorted(
        path
        for path in Path(root).glob(f"*{AUTHORITATIVE_OUTPUT_TOKEN}.h5")
        if path.name.startswith(f"{video}_")
    )
    if len(matches) != 1:
        raise AcceptanceError(
            f"Expected one collected v5 H5 for {video}, found {len(matches)}"
        )
    return matches[0].resolve()


def _point_cloud_schema(handle: h5py.File) -> dict[str, tuple[Any, ...]]:
    if "point_cloud" not in handle:
        raise AcceptanceError("H5 lacks point_cloud group")
    result: dict[str, tuple[Any, ...]] = {}

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            result[name] = (
                tuple(obj.shape),
                str(obj.dtype),
                obj.compression,
                obj.compression_opts,
            )

    handle["point_cloud"].visititems(visitor)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"JSON root must be a mapping: {path}")
    return dict(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_preflight_inputs(preflight_root: Path) -> dict[str, Any]:
    summary = _read_json(preflight_root / "preflight_summary.json")
    if summary.get("status") != "ok" or int(summary.get("h5_files_written", -1)) != 0:
        raise AcceptanceError("Step 5 preflight summary is not an accepted read-only run")
    rows = _read_csv(preflight_root / "input_checksums.csv")
    if not rows:
        raise AcceptanceError("Step 5 input checksum inventory is empty")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise AcceptanceError(f"Step 5 protected input changed: {path}")
    return summary


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    exclusion_path = Path(args.exclusion_manifest).resolve()
    source_root = Path(args.source_annotated_root).resolve()
    annotated_root = Path(args.annotated_root).resolve()
    collected_root = Path(args.collected_root).resolve()
    visualization_root = Path(args.visualization_root).resolve()
    snapshot_root = Path(args.snapshot_root).resolve()
    preflight_root = Path(args.preflight_root).resolve()

    preflight = _validate_preflight_inputs(preflight_root)
    manifest_items = load_stage4_sweep_manifest(manifest_path)
    excluded = {row.video_name for row in load_exclusions(exclusion_path)}
    enabled = sorted(
        row.video_name
        for row in manifest_items
        if row.enabled and row.video_name not in excluded
    )
    if len(enabled) != int(args.expected_videos):
        raise AcceptanceError(
            f"Enabled video count mismatch: {len(enabled)} != {args.expected_videos}"
        )
    if len(excluded) != int(args.expected_excluded_videos):
        raise AcceptanceError(
            f"Excluded video count mismatch: {len(excluded)} != "
            f"{args.expected_excluded_videos}"
        )
    snapshots = _snapshot_contract(
        snapshot_root, int(args.expected_reviewed_videos)
    )
    reviewed = set(snapshots)
    inherited = set(enabled) - reviewed
    if reviewed - set(enabled):
        raise AcceptanceError("Snapshot contains an excluded/disabled video")
    if len(inherited) != int(args.expected_inherited_videos):
        raise AcceptanceError(
            "Inherited video count mismatch: "
            f"{len(inherited)} != {args.expected_inherited_videos}"
        )
    if int(preflight.get("task_frames", -1)) != int(args.expected_authoritative_frames):
        raise AcceptanceError("Step 5 authoritative frame count differs from acceptance")
    expected_preflight_values = {
        "enabled_videos": len(enabled),
        "reviewed_videos": len(reviewed),
        "non_snapshot_videos": len(inherited),
    }
    for name, expected in expected_preflight_values.items():
        if int(preflight.get(name, -1)) != expected:
            raise AcceptanceError(
                f"Step 5 {name} mismatch: {preflight.get(name)} != {expected}"
            )
    expected_preflight_hashes = {
        "manifest_sha256": file_sha256(manifest_path),
        "exclusion_manifest_sha256": file_sha256(exclusion_path),
        "snapshot_manifest_sha256": file_sha256(
            snapshot_root / "export_manifest.csv"
        ),
    }
    for name, expected in expected_preflight_hashes.items():
        if preflight.get(name) != expected:
            raise AcceptanceError(f"Step 5 {name} differs from current input")

    artifact_counts = {
        "annotated H5": len(
            list(annotated_root.rglob(f"*{AUTHORITATIVE_OUTPUT_TOKEN}.h5"))
        ),
        "collected H5": len(
            list(collected_root.glob(f"*{AUTHORITATIVE_OUTPUT_TOKEN}.h5"))
        ),
        "visualization summary": len(
            list(visualization_root.glob("*/summary.json"))
        ),
    }
    for name, actual in artifact_counts.items():
        if actual != len(enabled):
            raise AcceptanceError(
                f"{name} count mismatch: {actual} != {len(enabled)}"
            )

    snapshot_manifest_sha = file_sha256(snapshot_root / "export_manifest.csv")
    status_counts: Counter[str] = Counter()
    mask_status_counts: Counter[str] = Counter()
    authoritative_frames = 0
    authoritative_positive = 0
    positive_outside = 0
    visualization_mismatches = 0
    total_points = 0
    total_positive = 0

    print("Stage 4 teacher v5 CVAT-authoritative acceptance check (read-only)")
    print(f"  annotated root    : {annotated_root}")
    print(f"  collected root    : {collected_root}")
    print(f"  visualization root: {visualization_root}")
    print(f"  videos            : {len(enabled)}")
    print(f"  reviewed/inherited: {len(reviewed)}/{len(inherited)}")

    for index, video in enumerate(enabled, start=1):
        source_h5 = _resolve_source(source_root, video)
        final_h5 = _single_h5(annotated_root, video)
        collected_h5 = _single_collected_h5(collected_root, video)
        final_sha = file_sha256(final_h5)
        if file_sha256(collected_h5) != final_sha:
            raise AcceptanceError(f"Collected H5 differs from annotated H5: {video}")

        with h5py.File(source_h5, "r") as source, h5py.File(final_h5, "r") as final:
            if _decode(
                final.attrs.get("contour_teacher_schema", "")
            ) != AUTHORITATIVE_OUTPUT_TOKEN:
                raise AcceptanceError(f"Wrong teacher schema: {video}")
            if _decode(
                final.attrs.get("manual_review_fullvideo_source_h5_sha256", "")
            ) != file_sha256(source_h5):
                raise AcceptanceError(f"Source H5 checksum metadata mismatch: {video}")
            if _decode(
                final.attrs.get(
                    "manual_review_fullvideo_snapshot_manifest_sha256", ""
                )
            ) != snapshot_manifest_sha:
                raise AcceptanceError(f"Snapshot manifest checksum mismatch: {video}")
            if _point_cloud_schema(source) != _point_cloud_schema(final):
                raise AcceptanceError(f"point_cloud schema changed: {video}")
            labels = final["annotation/point_label"][:].astype(np.int8)
            valid = final["annotation/valid_mask"][:].astype(bool)
            if not np.array_equal(valid, labels != -1):
                raise AcceptanceError(f"Final label/valid_mask mismatch: {video}")
            total_points += int(labels.size)
            total_positive += int(np.sum(labels == 1))
            is_reviewed = bool(
                final.attrs.get("manual_review_fullvideo_reviewed_video", False)
            )
            frame_group = final.get("manual_review_fullvideo_frames")
            if frame_group is None:
                raise AcceptanceError(f"Missing frame provenance: {video}")
            frame_count = int(len(frame_group["frame_order"]))
            frame_outside = int(
                np.sum(frame_group["positive_outside_cvat_mask_points"][:])
            )
            root_outside = int(
                final.attrs.get(
                    "manual_review_fullvideo_positive_outside_cvat_mask_points", -1
                )
            )
            if frame_outside != root_outside or root_outside != 0:
                raise AcceptanceError(f"Positive point outside CVAT mask: {video}")
            positive_outside += root_outside

            if video in reviewed:
                if not is_reviewed or frame_count <= 0:
                    raise AcceptanceError(f"Reviewed video lacks frame authority: {video}")
                authorities = {
                    _decode(value) for value in frame_group["label_authority"][:]
                }
                if authorities != {AUTHORITATIVE_LABEL_AUTHORITY}:
                    raise AcceptanceError(f"Wrong reviewed label authority: {video}")
                saved_positive = int(np.sum(frame_group["positive_points"][:]))
                if saved_positive != int(np.sum(labels == 1)):
                    raise AcceptanceError(
                        f"Frame provenance/final positive count mismatch: {video}"
                    )
                authoritative_frames += frame_count
                authoritative_positive += saved_positive
                mask_status_counts.update(
                    _decode(value) for value in frame_group["cvat_mask_status"][:]
                )
                status_counts["reviewed"] += 1
            else:
                if is_reviewed or frame_count != 0:
                    raise AcceptanceError(
                        f"Inherited video unexpectedly has CVAT frame authority: {video}"
                    )
                source_labels = source["annotation/point_label"][:].astype(np.int8)
                source_valid = source["annotation/valid_mask"][:].astype(bool)
                if not np.array_equal(labels, source_labels) or not np.array_equal(
                    valid, source_valid
                ):
                    raise AcceptanceError(f"Inherited labels changed: {video}")
                status_counts["inherited"] += 1

        vis_dir = visualization_root / video
        vis_summary = _read_json(vis_dir / "summary.json")
        if vis_summary.get("status") != "ok":
            raise AcceptanceError(f"Visualization status is not ok: {video}")
        if vis_summary.get("annotated_h5_sha256") != final_sha:
            raise AcceptanceError(f"Visualization input checksum mismatch: {video}")
        if int(vis_summary.get("num_points", -1)) != int(labels.size):
            raise AcceptanceError(f"Visualization point count mismatch: {video}")
        if int(vis_summary.get("positive_points", -1)) != int(
            np.sum(labels == 1)
        ):
            raise AcceptanceError(f"Visualization positive count mismatch: {video}")
        if int(vis_summary.get("positive_outside_cvat_mask_points", -1)) != 0:
            raise AcceptanceError(f"Visualization reports outside positive: {video}")
        mismatch = int(vis_summary.get("cvat_point_mask_mismatch_points", -1))
        if mismatch != 0:
            raise AcceptanceError(f"Visualization reports CVAT mismatch: {video}")
        visualization_mismatches += mismatch
        if video in reviewed:
            if vis_summary.get("cvat_label_authority") != AUTHORITATIVE_LABEL_AUTHORITY:
                raise AcceptanceError(f"Visualization authority mismatch: {video}")
            if int(vis_summary.get("cvat_authoritative_frames", -1)) != int(
                vis_summary.get("num_frames", -2)
            ):
                raise AcceptanceError(f"Visualization frame coverage mismatch: {video}")
        elif vis_summary.get("cvat_label_authority") != "inherited":
            raise AcceptanceError(f"Inherited visualization authority mismatch: {video}")
        rendered = list((vis_dir / "frames").glob("annotation_frame_*.png"))
        if len(rendered) != int(vis_summary["num_frames"]):
            raise AcceptanceError(f"Visualization image count mismatch: {video}")

        print(
            f"  [{index}/{len(enabled)}] [OK] {video}: "
            f"status={'reviewed' if video in reviewed else 'inherited'}, "
            f"points={int(vis_summary['num_points'])}, "
            f"positive={int(vis_summary['positive_points'])}"
        )

    if authoritative_frames != int(args.expected_authoritative_frames):
        raise AcceptanceError(
            "Authoritative frame count mismatch: "
            f"{authoritative_frames} != {args.expected_authoritative_frames}"
        )
    expected_mask_status = {
        "positive": int(preflight["cvat_positive_frames"]),
        "empty": int(preflight["cvat_empty_frames"]),
        "omitted_as_empty": int(preflight["cvat_omitted_as_empty_frames"]),
    }
    actual_mask_status = {
        name: int(mask_status_counts[name]) for name in expected_mask_status
    }
    if actual_mask_status != expected_mask_status:
        raise AcceptanceError(
            f"CVAT mask status counts differ from Step 5: {actual_mask_status}"
        )
    if authoritative_positive != int(preflight["projected_positive_points"]):
        raise AcceptanceError(
            "Final reviewed positive count differs from Step 5 projection: "
            f"{authoritative_positive} != {preflight['projected_positive_points']}"
        )

    regression_video = str(args.regression_video)
    regression_order = int(args.regression_frame_order)
    regression_dir = visualization_root / regression_video
    regression_rows = [
        row
        for row in _read_csv(regression_dir / "frame_labels.csv")
        if int(row["frame_order"]) == regression_order
    ]
    if not regression_rows:
        raise AcceptanceError("Regression frame is absent from frame_labels.csv")
    for row in regression_rows:
        if (
            row.get("cvat_mask_status") != "positive"
            or int(row.get("positive_points", 0)) <= 0
            or int(row.get("positive_outside_cvat_mask_points", -1)) != 0
            or int(row.get("cvat_point_mask_mismatch_points", -1)) != 0
        ):
            raise AcceptanceError(
                f"Regression frame CVAT-authority contract failed: {row}"
            )
    regression_image = (
        regression_dir
        / "frames"
        / f"annotation_frame_{regression_order:05d}.png"
    )
    if not regression_image.is_file():
        raise AcceptanceError(f"Regression visualization is missing: {regression_image}")

    for video in excluded:
        if (annotated_root / video).exists() or (visualization_root / video).exists():
            raise AcceptanceError(f"Excluded video leaked into v5 output: {video}")
        if list(collected_root.glob(f"{video}_*{AUTHORITATIVE_OUTPUT_TOKEN}.h5")):
            raise AcceptanceError(f"Excluded video leaked into collected output: {video}")

    # Recheck the Step 5 input inventory after traversing every output.  The
    # acceptance command writes only its own JSON report.
    _validate_preflight_inputs(preflight_root)

    summary = {
        "schema_version": 1,
        "status": "ok",
        "teacher_version": AUTHORITATIVE_OUTPUT_TOKEN,
        "videos": len(enabled),
        "reviewed_videos": len(reviewed),
        "inherited_videos": len(inherited),
        "excluded_videos": sorted(excluded),
        "authoritative_frames": authoritative_frames,
        "authoritative_positive_points": authoritative_positive,
        "positive_outside_cvat_mask_points": positive_outside,
        "cvat_point_mask_mismatch_points": visualization_mismatches,
        "cvat_mask_status_counts": actual_mask_status,
        "total_points": total_points,
        "total_positive_points": total_positive,
        "status_counts": dict(sorted(status_counts.items())),
        "collected_byte_identical": True,
        "step5_projection_matched": True,
        "protected_step5_inputs_unchanged": True,
        "regression_video": regression_video,
        "regression_frame_order": regression_order,
        "regression_image": str(regression_image),
        "failure_rows": 0,
        "files_written": 1,
    }
    _write_json_atomic(Path(args.output_json), summary)
    print("\nStep 6 acceptance summary")
    for name in (
        "videos",
        "reviewed_videos",
        "inherited_videos",
        "authoritative_frames",
        "authoritative_positive_points",
        "positive_outside_cvat_mask_points",
        "cvat_point_mask_mismatch_points",
    ):
        print(f"  {name:38s}: {summary[name]}")
    print(f"  regression_image                      : {regression_image}")
    print("Stage 4 teacher v5 CVAT-authoritative acceptance checks passed.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only acceptance check for the versioned Stage 4 v5 teacher build."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclusion_manifest", type=Path, required=True)
    parser.add_argument("--source_annotated_root", type=Path, required=True)
    parser.add_argument("--annotated_root", type=Path, required=True)
    parser.add_argument("--collected_root", type=Path, required=True)
    parser.add_argument("--visualization_root", type=Path, required=True)
    parser.add_argument("--snapshot_root", type=Path, required=True)
    parser.add_argument("--preflight_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--expected_videos", type=int, default=181)
    parser.add_argument("--expected_reviewed_videos", type=int, default=59)
    parser.add_argument("--expected_inherited_videos", type=int, default=122)
    parser.add_argument("--expected_excluded_videos", type=int, default=1)
    parser.add_argument("--expected_authoritative_frames", type=int, default=3014)
    parser.add_argument("--regression_video", default="1-3_14")
    parser.add_argument("--regression_frame_order", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_acceptance(args)
    except Exception as exc:
        raise SystemExit(
            "Stage 4 teacher v5 CVAT-authoritative acceptance failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
