from __future__ import annotations

import argparse
import csv
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.export.export_pseudo3d_point_cloud import (
    build_frame_point_cloud,
    load_point_cloud_inputs,
    make_sampling_config_from_args,
    make_texture_config_from_args,
    save_point_cloud_h5,
    save_point_cloud_ply,
)
from src.utils.pseudo3d_sampling import (
    SOURCE_CONTEXT_GRID,
    SOURCE_GLOBAL,
    SOURCE_LOCAL_PERCENTILE,
    SOURCE_TOPHAT,
    summarize_per_frame_counts,
    summarize_source_flags,
)


@dataclass(frozen=True)
class BatchItem:
    input_h5: Path
    video_name: str = ""
    output_h5: Path | None = None
    output_ply: Path | None = None


def _read_h5_list(h5_list: Path) -> list[BatchItem]:
    """
    Read pseudo3D h5 paths from a text or CSV file.

    Text format:
      one pseudo3D h5 path per line

    CSV format:
      input_h5/h5_path/pseudo3d_h5 column, optional video_name, output_h5,
      and output_ply columns.
    """
    h5_list = Path(h5_list)
    if not h5_list.exists():
        raise FileNotFoundError(f"h5_list not found: {h5_list}")

    if h5_list.suffix.lower() == ".csv":
        with h5_list.open(newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample)

            if has_header:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                for candidate in ["input_h5", "h5_path", "pseudo3d_h5"]:
                    if candidate in reader.fieldnames:
                        path_column = candidate
                        break
                else:
                    path_column = reader.fieldnames[0]

                items = []
                for row in reader:
                    value = row.get(path_column, "").strip()
                    if not value:
                        continue
                    items.append(
                        BatchItem(
                            input_h5=Path(value),
                            video_name=row.get("video_name", "").strip(),
                            output_h5=(
                                Path(row["output_h5"])
                                if row.get("output_h5", "").strip()
                                else None
                            ),
                            output_ply=(
                                Path(row["output_ply"])
                                if row.get("output_ply", "").strip()
                                else None
                            ),
                        )
                    )
                return items

            reader = csv.reader(f)
            return [
                BatchItem(input_h5=Path(row[0]))
                for row in reader
                if row and row[0].strip()
            ]

    items = []
    with h5_list.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(BatchItem(input_h5=Path(line)))
    return items


def collect_batch_items(
    *,
    input_dir: Path | None,
    h5_list: Path | None,
    pattern: str,
    recursive: bool,
) -> list[BatchItem]:
    if h5_list is not None:
        items = _read_h5_list(h5_list)
    elif input_dir is not None:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")
        globber = input_dir.rglob if recursive else input_dir.glob
        items = [
            BatchItem(input_h5=path)
            for path in sorted(globber(pattern))
            if path.is_file()
        ]
    else:
        raise ValueError("Either --input_dir or --h5_list must be specified.")

    if not items:
        raise FileNotFoundError("No pseudo3D h5 files were found.")
    return items


def derive_video_name(input_h5: Path, *, strip_suffix: str) -> str:
    stem = Path(input_h5).stem
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def _format_template(template: str, context: dict[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        raise KeyError(
            f"Unknown template field {exc} in '{template}'. "
            f"Available fields: {sorted(context)}"
        ) from exc


def resolve_output_paths(
    item: BatchItem,
    *,
    output_root: Path,
    preset: str,
    geometry_key: str,
    point_mode: str,
    sampling_mode: str,
    video_suffix_to_strip: str,
    output_subdir_template: str,
    output_h5_filename_template: str,
    output_ply_filename_template: str,
) -> tuple[str, Path, Path]:
    input_h5 = Path(item.input_h5)
    pseudo3d_stem = input_h5.stem
    video_name = item.video_name or derive_video_name(
        input_h5,
        strip_suffix=video_suffix_to_strip,
    )
    context = {
        "pseudo3d_stem": pseudo3d_stem,
        "video_name": video_name,
        "preset": preset,
        "geometry_key": geometry_key,
        "point_mode": point_mode,
        "sampling_mode": sampling_mode,
    }

    output_subdir = _format_template(output_subdir_template, context)
    output_dir = Path(output_root) / output_subdir
    output_h5 = (
        Path(item.output_h5)
        if item.output_h5 is not None
        else output_dir / _format_template(output_h5_filename_template, context)
    )
    output_ply = (
        Path(item.output_ply)
        if item.output_ply is not None
        else output_dir / _format_template(output_ply_filename_template, context)
    )
    return video_name, output_h5, output_ply


def export_one(
    *,
    input_h5: Path,
    output_h5: Path,
    output_ply: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    data = load_point_cloud_inputs(input_h5, geometry_key=args.geometry_key)
    texture_config = make_texture_config_from_args(args)
    sampling_config = make_sampling_config_from_args(args)
    point_cloud = build_frame_point_cloud(
        local_images=data["local_encoder_images"],
        pred_tracking=data["pred_tracking"],
        pixel_to_image=data["pixel_to_image"],
        frame_indices=data["frame_indices"],
        texture_config=texture_config,
        min_alpha=args.min_alpha,
        min_intensity=args.min_intensity,
        original_foreground_mode=args.original_foreground_mode,
        point_mode=args.point_mode,
        sampling_config=sampling_config,
        sample_stride=args.sample_stride,
        max_points_per_frame=args.max_points_per_frame,
        random_seed=args.random_seed,
    )

    counts = point_cloud["per_frame_counts"]
    source_stats = summarize_source_flags(point_cloud["source_flags"])
    count_stats = summarize_per_frame_counts(counts)
    meta = {
        "source_h5": str(input_h5),
        "geometry_key": args.geometry_key,
        "tracking_key": data["tracking_key"],
        "preset": args.preset,
        "point_mode": args.point_mode,
        "sampling_mode": sampling_config.sampling_mode,
        "include_context_grid": sampling_config.include_context_grid,
        "context_grid_stride": sampling_config.context_grid_stride,
        "context_grid_phase": sampling_config.context_grid_phase,
        "enable_local_percentile": sampling_config.enable_local_percentile,
        "local_window_size": sampling_config.local_window_size,
        "local_percentile": sampling_config.local_percentile,
        "local_min_contrast": sampling_config.local_min_contrast,
        "enable_tophat": sampling_config.enable_tophat,
        "tophat_kernel_size": sampling_config.tophat_kernel_size,
        "tophat_percentile": sampling_config.tophat_percentile,
        "tophat_min_response": sampling_config.tophat_min_response,
        "tophat_morph_shape": sampling_config.tophat_morph_shape,
        "enable_evidence_cleanup": sampling_config.enable_evidence_cleanup,
        "evidence_open_ksize": sampling_config.evidence_open_ksize,
        "evidence_close_ksize": sampling_config.evidence_close_ksize,
        "evidence_morph_shape": sampling_config.evidence_morph_shape,
        "evidence_min_component_area": (
            sampling_config.evidence_min_component_area
        ),
        "local_source_confidence": sampling_config.local_source_confidence,
        "tophat_source_confidence": sampling_config.tophat_source_confidence,
        "context_source_confidence": sampling_config.context_source_confidence,
        "texture_style": texture_config.texture_style,
        "texture_denoise": texture_config.denoise,
        "texture_denoise_ksize": texture_config.denoise_ksize,
        "texture_threshold_mode": texture_config.threshold_mode,
        "texture_percentile": texture_config.percentile,
        "texture_min_component_area": texture_config.min_component_area,
        "min_alpha": int(args.min_alpha),
        "min_intensity": int(args.min_intensity),
        "original_foreground_mode": args.original_foreground_mode,
        "sample_stride": int(args.sample_stride),
        "max_points_per_frame": int(args.max_points_per_frame),
        "random_seed": int(args.random_seed),
        "num_frames": int(counts.shape[0]),
        "num_points": int(point_cloud["points"].shape[0]),
        **count_stats,
        **source_stats,
        "source_type_frame": 0,
        "source_type_global": 1,
        "source_type_local_percentile": 2,
        "source_type_tophat": 3,
        "source_type_context_grid": 4,
        "source_flags_global": int(SOURCE_GLOBAL),
        "source_flags_local_percentile": int(SOURCE_LOCAL_PERCENTILE),
        "source_flags_tophat": int(SOURCE_TOPHAT),
        "source_flags_context_grid": int(SOURCE_CONTEXT_GRID),
        "batch_source": "batch_export_pseudo3d_point_cloud.py",
    }

    save_point_cloud_h5(output_h5, point_cloud=point_cloud, meta=meta)
    if output_ply is not None:
        save_point_cloud_ply(output_ply, point_cloud=point_cloud)

    return {
        "input_h5": str(input_h5),
        "output_h5": str(output_h5),
        "output_ply": str(output_ply) if output_ply is not None else "",
        "num_frames": meta["num_frames"],
        "num_points": meta["num_points"],
        "sampling_mode": meta["sampling_mode"],
        "points_per_frame_min": meta["points_per_frame_min"],
        "points_per_frame_max": meta["points_per_frame_max"],
        "points_per_frame_mean": meta["points_per_frame_mean"],
        "points_per_frame_median": meta["points_per_frame_median"],
        "global_evidence_points": meta["global_evidence_points"],
        "local_percentile_points": meta["local_percentile_points"],
        "tophat_points": meta["tophat_points"],
        "context_grid_points": meta["context_grid_points"],
        "context_only_points": meta["context_only_points"],
        "evidence_points": meta["evidence_points"],
        "overlap_points": meta["overlap_points"],
        "legacy_point_count_multiplier": meta["legacy_point_count_multiplier"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "video_name",
        "input_h5",
        "output_h5",
        "output_ply",
        "num_frames",
        "num_points",
        "sampling_mode",
        "points_per_frame_min",
        "points_per_frame_median",
        "points_per_frame_max",
        "points_per_frame_mean",
        "global_evidence_points",
        "local_percentile_points",
        "tophat_points",
        "context_grid_points",
        "context_only_points",
        "evidence_points",
        "overlap_points",
        "legacy_point_count_multiplier",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-export pseudo-3D frame-plane point cloud h5/ply files."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=Path, default=None)
    input_group.add_argument("--h5_list", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default="*.h5")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_ply", action="store_true")

    parser.add_argument("--video_suffix_to_strip", type=str, default="")
    parser.add_argument(
        "--output_subdir_template",
        type=str,
        default="{pseudo3d_stem}/{preset}_{geometry_key}",
    )
    parser.add_argument(
        "--output_h5_filename_template",
        type=str,
        default="{video_name}_pointcloud_{point_mode}.h5",
    )
    parser.add_argument(
        "--output_ply_filename_template",
        type=str,
        default="{video_name}_pointcloud_{point_mode}.ply",
    )

    parser.add_argument("--geometry_key", type=str, default="corr", choices=["prior", "corr"])
    parser.add_argument(
        "--preset",
        type=str,
        default="percentile90_area100",
        choices=[
            "original",
            "otsu_area500",
            "percentile90_area100",
            "percentile_area100",
            "percentile85_area100",
        ],
    )
    parser.add_argument(
        "--point_mode",
        type=str,
        default="foreground",
        choices=["foreground", "grid", "dense"],
    )
    parser.add_argument("--min_alpha", type=int, default=1)
    parser.add_argument("--min_intensity", type=int, default=1)
    parser.add_argument(
        "--original_foreground_mode",
        type=str,
        default="nonzero",
        choices=["all", "nonzero", "intensity"],
    )
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_points_per_frame", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="legacy",
        choices=["legacy", "combined_v2"],
    )
    parser.add_argument(
        "--include_context_grid",
        default=None,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--context_grid_stride", type=int, default=4)
    parser.add_argument(
        "--context_grid_phase",
        type=str,
        default="origin",
        choices=["origin", "centered", "dual"],
    )
    parser.add_argument(
        "--enable_local_percentile",
        default=None,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--local_window_size", type=int, default=41)
    parser.add_argument("--local_percentile", type=float, default=80.0)
    parser.add_argument("--local_min_contrast", type=float, default=4.0)
    parser.add_argument(
        "--enable_tophat",
        default=None,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--tophat_kernel_size", type=int, default=21)
    parser.add_argument("--tophat_percentile", type=float, default=85.0)
    parser.add_argument("--tophat_min_response", type=float, default=2.0)
    parser.add_argument(
        "--tophat_morph_shape",
        type=str,
        default="ellipse",
        choices=["rect", "ellipse", "cross"],
    )
    parser.add_argument(
        "--enable_evidence_cleanup",
        default=None,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--evidence_open_ksize", type=int, default=3)
    parser.add_argument("--evidence_close_ksize", type=int, default=5)
    parser.add_argument(
        "--evidence_morph_shape",
        type=str,
        default="ellipse",
        choices=["rect", "ellipse", "cross"],
    )
    parser.add_argument("--evidence_min_component_area", type=int, default=100)
    parser.add_argument("--local_source_confidence", type=float, default=0.7)
    parser.add_argument("--tophat_source_confidence", type=float, default=0.7)
    parser.add_argument("--context_source_confidence", type=float, default=0.0)

    parser.add_argument(
        "--texture_style",
        type=str,
        default=None,
        choices=["original", "threshold_alpha"],
    )
    parser.add_argument(
        "--texture_denoise",
        type=str,
        default=None,
        choices=["none", "median", "gaussian", "bilateral"],
    )
    parser.add_argument("--texture_denoise_ksize", type=int, default=None)
    parser.add_argument(
        "--texture_threshold_mode",
        type=str,
        default=None,
        choices=["fixed", "otsu", "percentile", "adaptive"],
    )
    parser.add_argument("--texture_threshold", type=int, default=None)
    parser.add_argument("--texture_percentile", type=float, default=None)
    parser.add_argument("--texture_adaptive_block_size", type=int, default=None)
    parser.add_argument("--texture_adaptive_c", type=float, default=None)
    parser.add_argument("--texture_open_ksize", type=int, default=None)
    parser.add_argument("--texture_close_ksize", type=int, default=None)
    parser.add_argument(
        "--texture_morph_shape",
        type=str,
        default=None,
        choices=["rect", "ellipse", "cross"],
    )
    parser.add_argument("--texture_min_component_area", type=int, default=None)
    parser.add_argument("--texture_white_foreground", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    items = collect_batch_items(
        input_dir=args.input_dir,
        h5_list=args.h5_list,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    print(f"Found pseudo3D h5 files: {len(items)}")
    print(f"Output root             : {args.output_root}")
    print(f"Geometry key            : {args.geometry_key}")
    print(f"Preset                  : {args.preset}")
    print(f"Point mode              : {args.point_mode}")
    print(f"Sampling mode           : {args.sampling_mode}")
    print(f"Sample stride           : {args.sample_stride}")

    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []

    for index, item in enumerate(items, start=1):
        input_h5 = Path(item.input_h5)
        video_name, output_h5, output_ply = resolve_output_paths(
            item,
            output_root=args.output_root,
            preset=args.preset,
            geometry_key=args.geometry_key,
            point_mode=args.point_mode,
            sampling_mode=args.sampling_mode,
            video_suffix_to_strip=args.video_suffix_to_strip,
            output_subdir_template=args.output_subdir_template,
            output_h5_filename_template=args.output_h5_filename_template,
            output_ply_filename_template=args.output_ply_filename_template,
        )
        if args.no_ply:
            output_ply = None

        print("\n" + "=" * 72)
        print(f"[{index}/{len(items)}] video_name: {video_name}")
        print(f"input_h5 : {input_h5}")
        print(f"output_h5: {output_h5}")
        if output_ply is not None:
            print(f"output_ply: {output_ply}")

        requested = [output_h5]
        if output_ply is not None:
            requested.append(output_ply)
        if args.skip_existing and all(path.exists() for path in requested):
            print("Skipping existing outputs.")
            skipped += 1
            rows.append(
                {
                    "status": "skipped",
                    "video_name": video_name,
                    "input_h5": str(input_h5),
                    "output_h5": str(output_h5),
                    "output_ply": str(output_ply) if output_ply is not None else "",
                }
            )
            continue

        try:
            result = export_one(
                input_h5=input_h5,
                output_h5=output_h5,
                output_ply=output_ply,
                args=args,
            )
            result["status"] = "ok"
            result["video_name"] = video_name
            result["error"] = ""
            rows.append(result)
            processed += 1
            print(
                "Exported: "
                f"points={result['num_points']}, "
                f"frames={result['num_frames']}, "
                f"mean/frame={result['points_per_frame_mean']:.2f}, "
                f"multiplier={result['legacy_point_count_multiplier']:.2f}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((input_h5, message))
            rows.append(
                {
                    "status": "failed",
                    "video_name": video_name,
                    "input_h5": str(input_h5),
                    "output_h5": str(output_h5),
                    "output_ply": str(output_ply) if output_ply is not None else "",
                    "error": message,
                }
            )
            print(f"ERROR: {message}", file=sys.stderr)
            traceback.print_exc()
            if not args.continue_on_error:
                break

    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, rows)
        print(f"\nSaved summary CSV: {args.summary_csv}")

    print("\nBatch point cloud export summary:")
    print(f"  processed: {processed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {len(failures)}")

    if failures:
        for path, message in failures:
            print(f"  FAILED {path}: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
