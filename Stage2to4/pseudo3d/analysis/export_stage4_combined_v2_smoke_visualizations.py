from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES_LIST = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_smoke/candidates.txt"
)
DEFAULT_COMBINED_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_smoke/"
    "combined_v2_default_corr"
)
DEFAULT_LEGACY_ANNOTATED_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_visualizations/260711"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_smoke/"
    "combined_v2_default_corr_visualization"
)
DEFAULT_VIDEO_SUFFIX = "_ts448_oym96_corr"

SOURCE_GLOBAL = np.uint16(1 << 0)
SOURCE_LOCAL_PERCENTILE = np.uint16(1 << 1)
SOURCE_TOPHAT = np.uint16(1 << 2)
SOURCE_CONTEXT_GRID = np.uint16(1 << 3)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _attr_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    text = str(value)
    return text if text else default


def _read_candidates(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidates list not found: {path}")
    result = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            result.append(Path(value))
    if not result:
        raise ValueError(f"Candidates list is empty: {path}")
    if len(set(result)) != len(result):
        raise ValueError("Candidates list contains duplicate paths")
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"Candidate pseudo3D H5 not found: {path}")
    return result


def _video_name(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _find_unique(root: Path, filename: str) -> Path:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Legacy annotated root not found: {root}")
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} below {root}")
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {filename!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _inside_bbox_union(pixel_xy: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inside = np.zeros((pixel_xy.shape[0],), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        inside |= (
            (pixel_xy[:, 0] >= x1)
            & (pixel_xy[:, 0] <= x2)
            & (pixel_xy[:, 1] >= y1)
            & (pixel_xy[:, 1] <= y2)
        )
    return inside


def _source_counts(flags: np.ndarray) -> dict[str, int]:
    flags = np.asarray(flags, dtype=np.uint16)
    evidence = (
        flags & (SOURCE_GLOBAL | SOURCE_LOCAL_PERCENTILE | SOURCE_TOPHAT)
    ) != 0
    return {
        "global": int(np.count_nonzero((flags & SOURCE_GLOBAL) != 0)),
        "local_percentile": int(
            np.count_nonzero((flags & SOURCE_LOCAL_PERCENTILE) != 0)
        ),
        "tophat": int(np.count_nonzero((flags & SOURCE_TOPHAT) != 0)),
        "context_grid": int(
            np.count_nonzero((flags & SOURCE_CONTEXT_GRID) != 0)
        ),
        "context_only": int(
            np.count_nonzero(((flags & SOURCE_CONTEXT_GRID) != 0) & ~evidence)
        ),
    }


def _load_annotated_inputs(
    path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        for group_name in ("point_cloud", "frame_annotation"):
            if group_name not in handle:
                raise KeyError(f"{group_name} group missing: {path}")
        point_cloud = handle["point_cloud"]
        frame_annotation = handle["frame_annotation"]
        for key in ("frame_order", "pixel_xy"):
            if key not in point_cloud:
                raise KeyError(f"point_cloud/{key} missing: {path}")
        for key in ("frame_order", "bbox_local_xyxy"):
            if key not in frame_annotation:
                raise KeyError(f"frame_annotation/{key} missing: {path}")
        attrs = dict(handle.attrs)
        legacy_order = point_cloud["frame_order"][:].astype(np.int64)
        legacy_xy = point_cloud["pixel_xy"][:].astype(np.float64)
        bbox_order = frame_annotation["frame_order"][:].astype(np.int64)
        bbox_xyxy = frame_annotation["bbox_local_xyxy"][:].astype(np.float64)
    _require(
        legacy_xy.shape == (legacy_order.size, 2),
        f"Invalid legacy pixel array: {path}",
    )
    _require(
        bbox_xyxy.shape == (bbox_order.size, 4),
        f"Invalid BBox array: {path}",
    )
    return attrs, legacy_order, legacy_xy, bbox_order, bbox_xyxy


def _load_combined_inputs(
    path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Combined point-cloud H5 not found: {path}")
    with h5py.File(path, "r") as handle:
        if "point_cloud" not in handle:
            raise KeyError(f"point_cloud group missing: {path}")
        group = handle["point_cloud"]
        for key in ("frame_order", "pixel_xy", "source_flags"):
            if key not in group:
                raise KeyError(f"point_cloud/{key} missing: {path}")
        attrs = dict(group.attrs)
        frame_order = group["frame_order"][:].astype(np.int64)
        pixel_xy = group["pixel_xy"][:].astype(np.float64)
        source_flags = group["source_flags"][:].astype(np.uint16)
    _require(
        pixel_xy.shape == (frame_order.size, 2),
        f"Invalid combined pixel array: {path}",
    )
    _require(
        source_flags.shape == (frame_order.size,),
        f"Invalid combined source_flags: {path}",
    )
    _require(
        _attr_text(attrs.get("sampling_mode")) == "combined_v2",
        f"Expected combined_v2 point cloud: {path}",
    )
    return attrs, frame_order, pixel_xy, source_flags


def _valid_bboxes_by_order(
    frame_order: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> dict[int, np.ndarray]:
    valid = (
        np.all(np.isfinite(bbox_xyxy), axis=1)
        & (bbox_xyxy[:, 2] > bbox_xyxy[:, 0])
        & (bbox_xyxy[:, 3] > bbox_xyxy[:, 1])
    )
    return {
        int(order): bbox_xyxy[(frame_order == order) & valid]
        for order in np.unique(frame_order[valid])
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _prepare_rows(
    candidates: list[Path],
    *,
    video_suffix: str,
    combined_root: Path,
    combined_h5_template: str,
    legacy_annotated_root: Path,
    legacy_annotated_template: str,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = []
    target_rows = []
    for pseudo3d_h5 in candidates:
        video_name = _video_name(pseudo3d_h5, video_suffix)
        combined_h5 = combined_root / combined_h5_template.format(
            video_name=video_name
        )
        legacy_annotated_h5 = _find_unique(
            legacy_annotated_root,
            legacy_annotated_template.format(video_name=video_name),
        )
        (
            annotated_attrs,
            legacy_order,
            legacy_xy,
            bbox_order,
            bbox_xyxy,
        ) = _load_annotated_inputs(legacy_annotated_h5)
        (
            combined_attrs,
            combined_order,
            combined_xy,
            combined_flags,
        ) = _load_combined_inputs(combined_h5)

        source_pseudo3d = Path(
            _attr_text(annotated_attrs.get("source_pseudo3d_h5"))
        )
        _require(
            source_pseudo3d == pseudo3d_h5,
            f"Annotated pseudo3D source mismatch for {video_name}",
        )
        voc_xml_root = Path(_attr_text(annotated_attrs.get("voc_xml_root")))
        _require(voc_xml_root.exists(), f"VOC XML root not found: {voc_xml_root}")
        _require(
            Path(_attr_text(combined_attrs.get("source_h5"))) == pseudo3d_h5,
            f"Combined pseudo3D source mismatch for {video_name}",
        )

        video_output_dir = output_root / video_name
        output_obj = (
            video_output_dir
            / f"{video_name}_annotation_mask_debug_combined_v2.obj"
        )
        texture_dir = video_output_dir / "combined_v2_annotation_textures"
        manifest_rows.append(
            {
                "annotated_h5": str(legacy_annotated_h5),
                "pseudo3d_h5": str(pseudo3d_h5),
                "point_cloud_h5": str(combined_h5),
                "voc_xml_root": str(voc_xml_root),
                "video_name": video_name,
                "output_obj": str(output_obj),
            }
        )

        with h5py.File(pseudo3d_h5, "r") as handle:
            frame_indices = handle["frame_indices"][:].astype(np.int64)
        legacy_frame_counts = np.bincount(
            legacy_order,
            minlength=frame_indices.size,
        )
        bboxes = _valid_bboxes_by_order(bbox_order, bbox_xyxy)
        for order, boxes in sorted(bboxes.items()):
            legacy_frame = legacy_order == order
            combined_frame = combined_order == order
            legacy_inside = _inside_bbox_union(legacy_xy[legacy_frame], boxes)
            legacy_bbox_points = int(np.count_nonzero(legacy_inside))
            if legacy_bbox_points != 0 or legacy_frame_counts[order] == 0:
                continue
            combined_inside = _inside_bbox_union(
                combined_xy[combined_frame], boxes
            )
            inside_flags = combined_flags[combined_frame][combined_inside]
            source_counts = _source_counts(inside_flags)
            target_rows.append(
                {
                    "video_name": video_name,
                    "frame_order": order,
                    "frame_index": int(frame_indices[order]),
                    "legacy_frame_points": int(legacy_frame_counts[order]),
                    "legacy_bbox_points": legacy_bbox_points,
                    "combined_bbox_points": int(inside_flags.size),
                    "bbox_global_points": source_counts["global"],
                    "bbox_local_percentile_points": source_counts[
                        "local_percentile"
                    ],
                    "bbox_tophat_points": source_counts["tophat"],
                    "bbox_context_grid_points": source_counts["context_grid"],
                    "bbox_context_only_points": source_counts["context_only"],
                    "texture_png": str(
                        texture_dir / f"annotation_frame_{order:05d}.png"
                    ),
                }
            )
    return manifest_rows, target_rows


def _visualization_command(
    *,
    manifest_csv: Path,
    summary_csv: Path,
    skip_existing: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "pseudo3d/batch/export/"
            "batch_export_annotation_mask_visualization.py"
        ),
        "--h5_list",
        str(manifest_csv),
        "--texture_dir_name_template",
        "combined_v2_annotation_textures",
        "--summary_csv",
        str(summary_csv),
        "--geometry_key",
        "corr",
        "--xml_frame_id_source",
        "frame_index",
        "--xml_frame_number_offsets",
        "1",
        "--xml_annotation_dir_name",
        "annotations_renamed",
        "--strict_xml_annotation_dir",
        "--contour_preset",
        "percentile85_area100",
        "--contour_percentile",
        "85",
        "--contour_min_component_area",
        "100",
        "--contour_min_area",
        "20",
        "--contour_min_alpha",
        "1",
        "--contour_open_ksize",
        "3",
        "--contour_close_ksize",
        "5",
        "--draw_point_cloud_samples",
        "--sample_point_radius",
        "1",
        "--sample_point_alpha",
        "0.75",
        "--global_sample_color",
        "255,255,255",
        "--local_percentile_sample_color",
        "64,200,255",
        "--tophat_sample_color",
        "255,200,64",
        "--context_grid_sample_color",
        "120,120,120",
        "--continue_on_error",
    ]
    if skip_existing:
        command.append("--skip_existing")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and batch-export source-colored annotation-mask "
            "visualizations for the three Stage 4 combined_v2 smoke candidates."
        )
    )
    parser.add_argument(
        "--candidates_list", type=Path, default=DEFAULT_CANDIDATES_LIST
    )
    parser.add_argument("--combined_root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument(
        "--legacy_annotated_root",
        type=Path,
        default=DEFAULT_LEGACY_ANNOTATED_ROOT,
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video_suffix_to_strip", default=DEFAULT_VIDEO_SUFFIX)
    parser.add_argument(
        "--combined_h5_template",
        default=(
            "{video_name}/{video_name}_pointcloud_foreground_combined_v2.h5"
        ),
    )
    parser.add_argument(
        "--legacy_annotated_filename_template",
        default="{video_name}_pointcloud_annotated_foreground.h5",
    )
    parser.add_argument("--expected_videos", type=int, default=3)
    parser.add_argument("--expected_target_frames", type=int, default=51)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.expected_videos < 0:
        raise ValueError("expected_videos must be >= 0")
    if args.expected_target_frames < 0:
        raise ValueError("expected_target_frames must be >= 0")
    for name in (
        "combined_h5_template",
        "legacy_annotated_filename_template",
    ):
        try:
            getattr(args, name).format(video_name="example")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid {name}: {getattr(args, name)!r}") from exc


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    candidates = _read_candidates(args.candidates_list)
    if args.expected_videos:
        _require(
            len(candidates) == args.expected_videos,
            f"Expected {args.expected_videos} videos, found {len(candidates)}",
        )

    manifest_rows, target_rows = _prepare_rows(
        candidates,
        video_suffix=args.video_suffix_to_strip,
        combined_root=args.combined_root,
        combined_h5_template=args.combined_h5_template,
        legacy_annotated_root=args.legacy_annotated_root,
        legacy_annotated_template=args.legacy_annotated_filename_template,
        output_root=args.output_root,
    )
    if args.expected_target_frames:
        _require(
            len(target_rows) == args.expected_target_frames,
            f"Expected {args.expected_target_frames} target frames, "
            f"detected {len(target_rows)}",
        )
    _require(
        all(int(row["combined_bbox_points"]) > 0 for row in target_rows),
        "One or more target frames have no combined_v2 BBox points",
    )

    manifest_csv = args.output_root / "visualization_manifest.csv"
    target_csv = args.output_root / "target_frames.csv"
    summary_csv = args.output_root / "visualization_summary.csv"
    _write_csv(
        manifest_csv,
        manifest_rows,
        [
            "annotated_h5",
            "pseudo3d_h5",
            "point_cloud_h5",
            "voc_xml_root",
            "video_name",
            "output_obj",
        ],
    )
    _write_csv(
        target_csv,
        target_rows,
        [
            "video_name",
            "frame_order",
            "frame_index",
            "legacy_frame_points",
            "legacy_bbox_points",
            "combined_bbox_points",
            "bbox_global_points",
            "bbox_local_percentile_points",
            "bbox_tophat_points",
            "bbox_context_grid_points",
            "bbox_context_only_points",
            "texture_png",
        ],
    )

    print("Stage 4 combined_v2 smoke visualization preparation")
    print(f"  videos             : {len(manifest_rows)}")
    print(f"  target_frames      : {len(target_rows)}")
    print(f"  visualization_root : {args.output_root}")
    print(f"  manifest_csv       : {manifest_csv}")
    print(f"  target_frames_csv  : {target_csv}")
    if args.prepare_only:
        print("  visualization_run  : skipped (--prepare_only)")
        return

    command = _visualization_command(
        manifest_csv=manifest_csv,
        summary_csv=summary_csv,
        skip_existing=args.skip_existing,
    )
    print("  visualization_run  : starting")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("Stage 4 combined_v2 smoke visualizations exported.")


if __name__ == "__main__":
    main()
