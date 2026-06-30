from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_h5_list(h5_list: Path) -> list[Path]:
    """
    Read h5 paths from a text or CSV file.

    Text format:
      one path per line

    CSV format:
      a column named h5_path, input_h5, or the first column.
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
                if "h5_path" in reader.fieldnames:
                    column = "h5_path"
                elif "input_h5" in reader.fieldnames:
                    column = "input_h5"
                else:
                    column = reader.fieldnames[0]

                return [
                    Path(row[column])
                    for row in reader
                    if row.get(column, "").strip()
                ]

            reader = csv.reader(f)
            return [Path(row[0]) for row in reader if row and row[0].strip()]

    paths = []
    with h5_list.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))

    return paths


def collect_h5_paths(
    *,
    input_dir: Path | None,
    h5_list: Path | None,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    """
    Collect h5 paths from a directory or explicit list file.
    """
    if h5_list is not None:
        paths = _read_h5_list(h5_list)
    elif input_dir is not None:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

        globber = input_dir.rglob if recursive else input_dir.glob
        paths = sorted(p for p in globber(pattern) if p.is_file())
    else:
        raise ValueError("Either --input_dir or --h5_list must be specified.")

    if not paths:
        raise FileNotFoundError("No input h5 files were found.")

    return paths


def build_export_paths(
    input_h5: Path,
    *,
    output_root: Path,
    preset: str,
    geometry_key: str,
    obj_suffix: str,
    ply_suffix: str,
    screenshot_suffix: str,
    tracking_debug_suffix: str,
) -> dict[str, Path]:
    """
    Build per-h5 export paths.
    """
    stem = Path(input_h5).stem
    detail_dir = Path(output_root) / stem / f"{preset}_{geometry_key}"

    return {
        "detail_dir": detail_dir,
        "obj": detail_dir / f"{stem}{obj_suffix}.obj",
        "ply": detail_dir / f"{stem}{ply_suffix}.ply",
        "screenshot": detail_dir / f"{stem}{screenshot_suffix}.png",
        "tracking_debug": detail_dir / f"{stem}{tracking_debug_suffix}.obj",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-export visualization files from pseudo-3D h5 files. "
            "This script does not run DualTrack inference."
        )
    )

    # ------------------------------------------------------------
    # Batch input / output
    # ------------------------------------------------------------
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory containing pseudo-3D h5 files.",
    )
    input_group.add_argument(
        "--h5_list",
        type=Path,
        default=None,
        help="Text or CSV file listing input h5 paths.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.h5",
        help="Glob pattern used with --input_dir. Default: *.h5.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input_dir recursively.",
    )

    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory for exported visualization files.",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip an h5 when all requested outputs already exist.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing later h5 files after an error.",
    )

    # ------------------------------------------------------------
    # Main export outputs
    # ------------------------------------------------------------
    parser.add_argument(
        "--no_textured_obj",
        action="store_true",
        help="Disable textured OBJ export. By default, OBJ export is enabled.",
    )

    parser.add_argument(
        "--export_ply",
        action="store_true",
        help="Also export PLY geometry for each h5.",
    )

    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Also export a geometry-only screenshot for each h5.",
    )

    parser.add_argument(
        "--export_tracking_debug_obj",
        action="store_true",
        help="Also export center trajectory and representative axis OBJ for each h5.",
    )

    parser.add_argument(
        "--obj_suffix",
        type=str,
        default="_textured",
        help="Suffix appended to each h5 stem before .obj.",
    )

    parser.add_argument(
        "--ply_suffix",
        type=str,
        default="_mesh",
        help="Suffix appended to each h5 stem before .ply.",
    )

    parser.add_argument(
        "--screenshot_suffix",
        type=str,
        default="_screenshot",
        help="Suffix appended to each h5 stem before .png.",
    )

    parser.add_argument(
        "--tracking_debug_suffix",
        type=str,
        default="_tracking_debug",
        help="Suffix appended to each h5 stem before tracking debug .obj.",
    )

    parser.add_argument(
        "--texture_dir_name",
        type=str,
        default="textures",
        help="Texture directory name relative to each OBJ file.",
    )

    parser.add_argument(
        "--geometry_key",
        type=str,
        default="prior",
        choices=["prior", "corr"],
        help="Geometry to export from h5.",
    )

    # ------------------------------------------------------------
    # Preset
    # ------------------------------------------------------------
    parser.add_argument(
        "--preset",
        type=str,
        default="original",
        choices=[
            "original",
            "otsu_area500",
            "percentile90_area100",
            "percentile_area100",
            "percentile85_area100",
        ],
        help="Texture processing preset.",
    )

    # ------------------------------------------------------------
    # Optional texture overrides
    # ------------------------------------------------------------
    parser.add_argument(
        "--texture_style",
        type=str,
        default=None,
        choices=["original", "threshold_alpha"],
        help="Override texture style.",
    )

    parser.add_argument(
        "--texture_denoise",
        type=str,
        default=None,
        choices=["none", "median", "gaussian", "bilateral"],
        help="Override denoise mode.",
    )

    parser.add_argument(
        "--texture_denoise_ksize",
        type=int,
        default=None,
        help="Override denoise kernel size.",
    )

    parser.add_argument(
        "--texture_threshold_mode",
        type=str,
        default=None,
        choices=["fixed", "otsu", "percentile", "adaptive"],
        help="Override threshold mode.",
    )

    parser.add_argument(
        "--texture_threshold",
        type=int,
        default=None,
        help="Override fixed threshold.",
    )

    parser.add_argument(
        "--texture_percentile",
        type=float,
        default=None,
        help="Override percentile value.",
    )

    parser.add_argument(
        "--texture_adaptive_block_size",
        type=int,
        default=None,
        help="Override adaptive threshold block size.",
    )

    parser.add_argument(
        "--texture_adaptive_c",
        type=float,
        default=None,
        help="Override adaptive threshold C.",
    )

    parser.add_argument(
        "--texture_open_ksize",
        type=int,
        default=None,
        help="Override morphology opening kernel size.",
    )

    parser.add_argument(
        "--texture_close_ksize",
        type=int,
        default=None,
        help="Override morphology closing kernel size.",
    )

    parser.add_argument(
        "--texture_morph_shape",
        type=str,
        default=None,
        choices=["rect", "ellipse", "cross"],
        help="Override morphology kernel shape.",
    )

    parser.add_argument(
        "--texture_min_component_area",
        type=int,
        default=None,
        help="Override minimum connected component area.",
    )

    parser.add_argument(
        "--texture_white_foreground",
        action="store_true",
        help="Force visible alpha foreground pixels to pure white.",
    )

    # ------------------------------------------------------------
    # Geometry preview options
    # ------------------------------------------------------------
    parser.add_argument(
        "--show_only_slices",
        action="store_true",
        help="For PLY/screenshot, rebuild mesh with one quad per slice and no side faces.",
    )

    parser.add_argument(
        "--show_edges",
        action="store_true",
        help="Show mesh edges in screenshot.",
    )

    parser.add_argument(
        "--show_trajectory",
        action="store_true",
        help="Show trajectory in screenshot.",
    )

    parser.add_argument(
        "--label_every",
        type=int,
        default=0,
        help="Show original frame index label every N frames in screenshot.",
    )

    parser.add_argument(
        "--window_width",
        type=int,
        default=1400,
        help="Screenshot window width.",
    )

    parser.add_argument(
        "--window_height",
        type=int,
        default=1000,
        help="Screenshot window height.",
    )

    parser.add_argument(
        "--tracking_debug_axis_method",
        type=str,
        default="attrs",
        choices=["attrs", "first_valid", "early_mean", "global_mean", "start_to_end"],
        help="Representative axis method for tracking debug OBJ.",
    )

    parser.add_argument(
        "--tracking_debug_axis_window",
        type=int,
        default=None,
        help="Override early_mean valid-step window for tracking debug OBJ.",
    )

    parser.add_argument(
        "--tracking_debug_axis_length",
        type=float,
        default=0.0,
        help="Axis line length for tracking debug OBJ. 0 uses trajectory extent.",
    )

    parser.add_argument(
        "--tracking_debug_point_size",
        type=float,
        default=0.25,
        help="Center cross size for tracking debug OBJ.",
    )

    return parser


def _all_requested_outputs_exist(
    paths: dict[str, Path],
    *,
    export_textured_obj: bool,
    export_ply_enabled: bool,
    screenshot_enabled: bool,
    tracking_debug_enabled: bool,
) -> bool:
    requested = []
    if export_textured_obj:
        requested.append(paths["obj"])
    if export_ply_enabled:
        requested.append(paths["ply"])
    if screenshot_enabled:
        requested.append(paths["screenshot"])
    if tracking_debug_enabled:
        requested.append(paths["tracking_debug"])

    return bool(requested) and all(path.exists() for path in requested)


def main():
    parser = build_parser()
    args = parser.parse_args()

    from pseudo3d.export.export_pseudo3d_visualization import (
        export_ply,
        export_screenshot,
        export_tracking_debug_obj,
        export_textured_slices_obj,
        load_pseudo3d_h5,
        make_texture_config_from_args,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    export_textured_obj = not args.no_textured_obj
    if (
        not export_textured_obj
        and not args.export_ply
        and not args.screenshot
        and not args.export_tracking_debug_obj
    ):
        raise ValueError(
            "No outputs requested. Enable textured OBJ, --export_ply, "
            "--screenshot, or --export_tracking_debug_obj."
        )

    input_h5_paths = collect_h5_paths(
        input_dir=args.input_dir,
        h5_list=args.h5_list,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    texture_config = make_texture_config_from_args(args)

    print(f"Found h5 files: {len(input_h5_paths)}")
    print(f"Output root   : {output_root}")
    print(f"Geometry key  : {args.geometry_key}")
    print(f"Preset        : {args.preset}")

    failures: list[tuple[Path, str]] = []
    processed = 0
    skipped = 0

    for index, input_h5 in enumerate(input_h5_paths, start=1):
        input_h5 = Path(input_h5)
        paths = build_export_paths(
            input_h5,
            output_root=output_root,
            preset=args.preset,
            geometry_key=args.geometry_key,
            obj_suffix=args.obj_suffix,
            ply_suffix=args.ply_suffix,
            screenshot_suffix=args.screenshot_suffix,
            tracking_debug_suffix=args.tracking_debug_suffix,
        )

        print("\n" + "=" * 72)
        print(f"[{index}/{len(input_h5_paths)}] input_h5: {input_h5}")
        print(f"detail_dir: {paths['detail_dir']}")

        if args.skip_existing and _all_requested_outputs_exist(
            paths,
            export_textured_obj=export_textured_obj,
            export_ply_enabled=args.export_ply,
            screenshot_enabled=args.screenshot,
            tracking_debug_enabled=args.export_tracking_debug_obj,
        ):
            print("Skipping existing outputs.")
            skipped += 1
            continue

        try:
            data = load_pseudo3d_h5(input_h5, geometry_key=args.geometry_key)

            local_images = data["local_encoder_images"]
            frame_corners_world = data["frame_corners_world"]
            vertices = data["frame_mesh_vertices"]
            faces = data["frame_mesh_faces"]
            pred_tracking = data["pred_tracking"]
            pred_tracking_prior = data.get("pred_tracking_prior", pred_tracking)
            frame_indices = data["frame_indices"]

            if export_textured_obj:
                export_textured_slices_obj(
                    paths["obj"],
                    frame_corners_world=frame_corners_world,
                    texture_images=local_images,
                    texture_config=texture_config,
                    texture_dir_name=args.texture_dir_name,
                )

            if args.export_ply:
                export_ply(
                    paths["ply"],
                    vertices=vertices,
                    faces=faces,
                    frame_corners_world=frame_corners_world,
                    show_only_slices=args.show_only_slices,
                )

            if args.screenshot:
                export_screenshot(
                    paths["screenshot"],
                    vertices=vertices,
                    faces=faces,
                    frame_corners_world=frame_corners_world,
                    pred_tracking=pred_tracking,
                    frame_indices=frame_indices,
                    show_only_slices=args.show_only_slices,
                    show_edges=args.show_edges,
                    show_trajectory=args.show_trajectory,
                    label_every=args.label_every,
                    window_width=args.window_width,
                    window_height=args.window_height,
                )

            if args.export_tracking_debug_obj:
                export_tracking_debug_obj(
                    paths["tracking_debug"],
                    pred_tracking=pred_tracking,
                    pred_tracking_for_axis=pred_tracking_prior,
                    tracking_correction_attrs=data.get("tracking_correction_attrs", {}),
                    axis_method=args.tracking_debug_axis_method,
                    axis_window=args.tracking_debug_axis_window,
                    axis_length=args.tracking_debug_axis_length,
                    point_size=args.tracking_debug_point_size,
                )

            processed += 1
        except Exception as exc:
            failures.append((input_h5, repr(exc)))
            print(f"ERROR: failed to export {input_h5}: {exc}", file=sys.stderr)
            traceback.print_exc()
            if not args.continue_on_error:
                raise

    print("\n" + "=" * 72)
    print("Batch export done.")
    print(f"  processed: {processed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {len(failures)}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for input_h5, error in failures:
            print(f"  {input_h5}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
