from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_video_list(video_list: Path) -> list[Path]:
    """
    Read video paths from a text or CSV file.

    Text format:
      one path per line

    CSV format:
      a column named video_path, or the first column when no header is present
    """
    video_list = Path(video_list)
    if not video_list.exists():
        raise FileNotFoundError(f"video_list not found: {video_list}")

    if video_list.suffix.lower() == ".csv":
        with video_list.open(newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample)

            if has_header:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return []
                column = "video_path" if "video_path" in reader.fieldnames else reader.fieldnames[0]
                return [
                    Path(row[column])
                    for row in reader
                    if row.get(column, "").strip()
                ]

            reader = csv.reader(f)
            return [Path(row[0]) for row in reader if row and row[0].strip()]

    paths = []
    with video_list.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))

    return paths


def collect_video_paths(
    *,
    input_dir: Path | None,
    video_list: Path | None,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    """
    Collect video paths from a directory or an explicit list file.
    """
    if video_list is not None:
        paths = _read_video_list(video_list)
    elif input_dir is not None:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

        globber = input_dir.rglob if recursive else input_dir.glob
        paths = sorted(p for p in globber(pattern) if p.is_file())
    else:
        raise ValueError("Either --input_dir or --video_list must be specified.")

    if not paths:
        raise FileNotFoundError("No input videos were found.")

    return paths


def _infer_video_name_from_frame_dir(frame_dir: Path) -> str:
    if frame_dir.name in {"object_renamed", "object_rename"}:
        return frame_dir.parent.name
    return frame_dir.name


def collect_frame_dir_sources(
    *,
    input_dir: Path | None,
    video_list: Path | None,
    frame_root: Path | None,
    frame_object_dir_name: str,
    pattern: str,
    recursive: bool,
) -> list[tuple[Path, str]]:
    """
    Collect frame directory sources.

    With --input_dir, the expected layout is:
      input_dir/{video_name}/{frame_object_dir_name}

    With --video_list, entries can be video names used under --frame_root, or
    explicit frame directory paths.
    """
    sources: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add_frame_source(path: Path) -> None:
        path = Path(path)
        if path.name == frame_object_dir_name:
            frame_dir = path
            video_name = _infer_video_name_from_frame_dir(frame_dir)
        else:
            frame_dir = path / frame_object_dir_name
            video_name = path.name

        if not frame_dir.is_dir():
            return

        resolved = frame_dir.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        sources.append((frame_dir, video_name))

    if video_list is not None:
        entries = _read_video_list(video_list)
        for entry in entries:
            entry = Path(entry)
            if frame_root is not None and len(entry.parts) == 1:
                video_name = entry.name
                frame_dir = Path(frame_root) / video_name / frame_object_dir_name
                add_frame_source(frame_dir)
            else:
                add_frame_source(entry)

    elif input_dir is not None:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

        frame_pattern = "*" if pattern == "*.mp4" else pattern
        globber = input_dir.rglob if recursive else input_dir.glob
        add_frame_source(input_dir)
        video_dirs = sorted(p for p in globber(frame_pattern) if p.is_dir())
        for video_dir in video_dirs:
            add_frame_source(video_dir)

    else:
        raise ValueError("Either --input_dir or --video_list must be specified.")

    if not sources:
        raise FileNotFoundError("No frame directories were found.")

    return sources


def collect_input_sources(
    *,
    input_mode: str,
    input_dir: Path | None,
    video_list: Path | None,
    pattern: str,
    recursive: bool,
    frame_root: Path | None,
    frame_object_dir_name: str,
) -> list[tuple[Path, str | None]]:
    """
    Collect input sources as (path, frame_video_name).
    """
    if input_mode == "video":
        return [
            (path, None)
            for path in collect_video_paths(
                input_dir=input_dir,
                video_list=video_list,
                pattern=pattern,
                recursive=recursive,
            )
        ]

    if input_mode == "frame_dir":
        return collect_frame_dir_sources(
            input_dir=input_dir,
            video_list=video_list,
            frame_root=frame_root,
            frame_object_dir_name=frame_object_dir_name,
            pattern=pattern,
            recursive=recursive,
        )

    raise ValueError(f"Unknown input_mode: {input_mode}")


def build_output_h5_path(
    source_name: str,
    *,
    output_dir: Path,
    output_suffix: str,
) -> Path:
    """
    Build output h5 path for one input source.
    """
    return Path(output_dir) / f"{source_name}{output_suffix}.h5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run DualTrack on multiple videos and export pseudo-3D h5 files. "
            "This script only performs inference and h5 export."
        )
    )

    # ------------------------------------------------------------
    # Batch input / output
    # ------------------------------------------------------------
    parser.add_argument(
        "--input_mode",
        type=str,
        default="video",
        choices=["video", "frame_dir"],
        help=(
            "Input mode. video reads MP4 files. frame_dir reads JPEG/PNG "
            "directories under {root}/{video_name}/object_renamed."
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory containing input video files.",
    )
    input_group.add_argument(
        "--video_list",
        type=Path,
        default=None,
        help="Text or CSV file listing input video paths.",
    )

    parser.add_argument(
        "--frame_root",
        type=Path,
        default=None,
        help=(
            "Root path used with --input_mode frame_dir and --video_list entries "
            "that are video names."
        ),
    )

    parser.add_argument(
        "--frame_object_dir_name",
        type=str,
        default="object_renamed",
        help="Frame subdirectory name under {root}/{video_name}.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help=(
            "Glob pattern used with --input_dir. For video mode this matches "
            "video files. For frame_dir mode this matches video directories."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input_dir recursively.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for output pseudo-3D h5 files.",
    )

    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_ts448_oym96_corr",
        help="Suffix appended to each video stem before .h5.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint path. "
            "If omitted, uses DUALTRACK_CHECKPOINT, "
            "DUALTRACK_CHECKPOINT_DIR/dualtrack_final.pt, or the legacy "
            "repo-local dualtrack_checkpoints/dualtrack_final.pt."
        ),
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a video when its output h5 already exists.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing later videos after an error.",
    )

    # ------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Frame sampling stride. stride=2 keeps frames [0,2,4,...].",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First 0-based frame index to consider.",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
        help="Maximum number of sampled frames. Use -1 for no limit.",
    )

    # ------------------------------------------------------------
    # Local branch preprocessing
    # ------------------------------------------------------------
    parser.add_argument(
        "--local_preprocess",
        type=str,
        default="resize_shorter_then_center_crop",
        choices=["center_crop", "resize_shorter_then_center_crop", "resize"],
        help="Preprocessing mode for local_encoder_images.",
    )

    parser.add_argument(
        "--local_target_short_side",
        type=int,
        default=448,
        help="Target short side before local crop.",
    )

    parser.add_argument(
        "--local_crop_offset_y",
        type=int,
        default=-96,
        help="Crop offset in y direction after optional resizing.",
    )

    parser.add_argument(
        "--local_crop_offset_x",
        type=int,
        default=0,
        help="Crop offset in x direction after optional resizing.",
    )

    parser.add_argument(
        "--local_crop_size",
        type=int,
        default=256,
        help="Final local input crop/resize size.",
    )

    # ------------------------------------------------------------
    # 3D plane geometry
    # ------------------------------------------------------------
    parser.add_argument(
        "--spacing_x",
        type=float,
        default=1.0,
        help="Temporary pixel spacing in x direction for visualization geometry.",
    )

    parser.add_argument(
        "--spacing_y",
        type=float,
        default=1.0,
        help="Temporary pixel spacing in y direction for visualization geometry.",
    )

    parser.add_argument(
        "--connect_slices",
        action="store_true",
        help="Also create side faces between adjacent frame planes in h5 mesh.",
    )

    # ------------------------------------------------------------
    # Tracking correction
    # ------------------------------------------------------------
    parser.add_argument(
        "--tracking_correction_preset",
        type=str,
        default="none",
        help=(
            "Named tracking correction preset. "
            "When combined with individual tracking arguments, explicitly "
            "specified individual arguments override the preset values."
        ),
    )

    parser.add_argument(
        "--enable_tracking_correction",
        action="store_true",
        help="Generate T_t^corr by correcting the translation trajectory.",
    )

    parser.add_argument(
        "--disable_tracking_correction",
        action="store_false",
        dest="enable_tracking_correction",
        help="Disable tracking correction even when a preset enables it.",
    )

    parser.add_argument(
        "--tracking_min_cos",
        type=float,
        default=-0.1,
        help="Cosine threshold for detecting reverse or sharp-turn steps.",
    )

    parser.add_argument(
        "--tracking_min_step_ratio",
        type=float,
        default=0.25,
        help="Minimum corrected step length as a ratio of median prior step.",
    )

    parser.add_argument(
        "--tracking_max_step_ratio",
        type=float,
        default=2.0,
        help="Maximum corrected step length as a ratio of median prior step.",
    )

    parser.add_argument(
        "--tracking_direction_update_alpha",
        type=float,
        default=0.2,
        help="Smoothing factor for updating the reference travel direction.",
    )

    parser.add_argument(
        "--tracking_initial_direction_method",
        type=str,
        default="first_valid",
        choices=["first_valid", "early_mean", "global_mean"],
        help="How to initialize the reference travel direction.",
    )

    parser.add_argument(
        "--tracking_initial_direction_window",
        type=int,
        default=15,
        help="Number of valid steps used by early_mean initial direction.",
    )

    parser.add_argument(
        "--tracking_enforce_progress_monotonic",
        action="store_true",
        help="Force corrected centers to be monotonic along the progress axis.",
    )

    parser.add_argument(
        "--no_tracking_enforce_progress_monotonic",
        action="store_false",
        dest="tracking_enforce_progress_monotonic",
        help=(
            "Disable progress-axis monotonic correction even when a preset "
            "enables it."
        ),
    )

    parser.add_argument(
        "--tracking_progress_monotonic_min_step_ratio",
        type=float,
        default=0.0,
        help="Minimum monotonic progress increment as a ratio of median prior step.",
    )

    parser.add_argument(
        "--tracking_fix_frame_normal_same_side",
        action="store_true",
        help=(
            "Detect when frames n-1 and n+1 lie on the same side of frame n's "
            "plane normal, then push frame n+1 and later frames to the opposite side."
        ),
    )

    parser.add_argument(
        "--no_tracking_fix_frame_normal_same_side",
        action="store_false",
        dest="tracking_fix_frame_normal_same_side",
        help=(
            "Disable frame-normal same-side correction even when a preset "
            "enables it."
        ),
    )

    parser.add_argument(
        "--tracking_frame_normal_same_side_min_step_ratio",
        type=float,
        default=0.0,
        help=(
            "Minimum opposite-side normal offset as a ratio of median prior step "
            "when --tracking_fix_frame_normal_same_side is enabled."
        ),
    )

    parser.add_argument(
        "--tracking_progress_axis_scale",
        type=float,
        default=1.0,
        help="Scale corrected centers along the progress axis. 1.0 disables scaling.",
    )

    parser.set_defaults(
        enable_tracking_correction=False,
        tracking_enforce_progress_monotonic=False,
        tracking_fix_frame_normal_same_side=False,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from src.utils.pseudo3d_processing import (
        get_tracking_correction_preset_names,
        get_tracking_correction_override_keys_from_argv,
        run_pseudo3d_inference_for_video,
    )

    if (
        args.tracking_correction_preset not in ("", "none")
        and args.tracking_correction_preset not in get_tracking_correction_preset_names()
    ):
        names = ", ".join(get_tracking_correction_preset_names())
        raise ValueError(
            "Unknown tracking correction preset: "
            f"{args.tracking_correction_preset}. Available presets: {names}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_correction_override_keys = get_tracking_correction_override_keys_from_argv(
        sys.argv[1:]
    )

    max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames

    input_sources = collect_input_sources(
        input_mode=args.input_mode,
        input_dir=args.input_dir,
        video_list=args.video_list,
        pattern=args.pattern,
        recursive=args.recursive,
        frame_root=args.frame_root,
        frame_object_dir_name=args.frame_object_dir_name,
    )

    print(f"Input mode  : {args.input_mode}")
    print(f"Found inputs: {len(input_sources)}")
    print(f"Output dir  : {output_dir}")

    failures: list[tuple[Path, str]] = []
    processed = 0
    skipped = 0

    for index, (input_path, frame_video_name) in enumerate(input_sources, start=1):
        input_path = Path(input_path)
        source_name = (
            frame_video_name
            if frame_video_name is not None
            else input_path.stem
        )
        output_h5 = build_output_h5_path(
            source_name,
            output_dir=output_dir,
            output_suffix=args.output_suffix,
        )

        print("\n" + "=" * 72)
        print(f"[{index}/{len(input_sources)}] input: {input_path}")
        print(f"input_mode: {args.input_mode}")
        if frame_video_name is not None:
            print(f"frame_video_name: {frame_video_name}")
        print(f"output_h5: {output_h5}")

        if args.skip_existing and output_h5.exists():
            print("Skipping existing output.")
            skipped += 1
            continue

        try:
            run_pseudo3d_inference_for_video(
                repo_root=REPO_ROOT,
                video_path=input_path,
                output_h5=output_h5,
                input_mode=args.input_mode,
                frame_video_name=frame_video_name,
                checkpoint_path=args.checkpoint,
                stride=args.stride,
                start_frame=args.start_frame,
                max_frames=max_frames,
                local_preprocess=args.local_preprocess,
                local_target_short_side=args.local_target_short_side,
                local_crop_offset_y=args.local_crop_offset_y,
                local_crop_offset_x=args.local_crop_offset_x,
                local_crop_size=args.local_crop_size,
                spacing_x=args.spacing_x,
                spacing_y=args.spacing_y,
                connect_slices=args.connect_slices,
                tracking_correction_preset=args.tracking_correction_preset,
                tracking_correction_override_keys=tracking_correction_override_keys,
                enable_tracking_correction=args.enable_tracking_correction,
                tracking_min_cos=args.tracking_min_cos,
                tracking_min_step_ratio=args.tracking_min_step_ratio,
                tracking_max_step_ratio=args.tracking_max_step_ratio,
                tracking_direction_update_alpha=args.tracking_direction_update_alpha,
                tracking_initial_direction_method=args.tracking_initial_direction_method,
                tracking_initial_direction_window=args.tracking_initial_direction_window,
                tracking_enforce_progress_monotonic=(
                    args.tracking_enforce_progress_monotonic
                ),
                tracking_progress_monotonic_min_step_ratio=(
                    args.tracking_progress_monotonic_min_step_ratio
                ),
                tracking_fix_frame_normal_same_side=(
                    args.tracking_fix_frame_normal_same_side
                ),
                tracking_frame_normal_same_side_min_step_ratio=(
                    args.tracking_frame_normal_same_side_min_step_ratio
                ),
                tracking_progress_axis_scale=args.tracking_progress_axis_scale,
            )
            processed += 1
        except Exception as exc:
            failures.append((input_path, repr(exc)))
            print(f"ERROR: failed to process {input_path}: {exc}", file=sys.stderr)
            traceback.print_exc()
            if not args.continue_on_error:
                raise

    print("\n" + "=" * 72)
    print("Batch inference done.")
    print(f"  processed: {processed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {len(failures)}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for input_path, error in failures:
            print(f"  {input_path}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
