from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.pseudo3d_processing import (
    get_tracking_correction_preset_names,
    get_tracking_correction_override_keys_from_argv,
    run_pseudo3d_inference_for_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DualTrack on one video and export pseudo-3D teacher data as h5. "
            "This script only performs inference and h5 export. "
            "Texture/OBJ/visualization export should be done from the generated h5."
        )
    )

    # ------------------------------------------------------------
    # Input / output
    # ------------------------------------------------------------
    parser.add_argument(
        "--input_mode",
        type=str,
        default="video",
        choices=["video", "frame_dir"],
        help="Input mode. video reads --video_path. frame_dir reads JPEG/PNG frames.",
    )

    parser.add_argument(
        "--video_path",
        type=Path,
        default=None,
        help="Path to input video file.",
    )

    parser.add_argument(
        "--frame_dir",
        type=Path,
        default=None,
        help=(
            "Path to frame directory. Expected layout is usually "
            "{frame_root}/{frame_video_name}/object_renamed."
        ),
    )

    parser.add_argument(
        "--frame_root",
        type=Path,
        default=None,
        help="Root path containing {frame_video_name}/object_renamed.",
    )

    parser.add_argument(
        "--frame_video_name",
        type=str,
        default=None,
        help=(
            "Video name used for frame directory layout and filenames. "
            "Frames are named {frame_video_name}_{frame_number:05d}.jpg/png."
        ),
    )

    parser.add_argument(
        "--frame_object_dir_name",
        type=str,
        default="object_renamed",
        help="Frame subdirectory name under {frame_root}/{frame_video_name}.",
    )

    parser.add_argument(
        "--output_h5",
        type=Path,
        required=True,
        help="Output pseudo-3D h5 path.",
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
        help=(
            "Preprocessing mode for local_encoder_images. "
            "center_crop: crop from raw image with offset. "
            "resize_shorter_then_center_crop: resize min(H,W) to target short side, "
            "then crop with offset. "
            "resize: resize whole frame to local_crop_size x local_crop_size."
        ),
    )

    parser.add_argument(
        "--local_target_short_side",
        type=int,
        default=448,
        help=(
            "Target short side before local crop. "
            "Used only when --local_preprocess resize_shorter_then_center_crop."
        ),
    )

    parser.add_argument(
        "--local_crop_offset_y",
        type=int,
        default=-96,
        help=(
            "Crop offset in y direction after optional resizing. "
            "Negative values move the crop upward."
        ),
    )

    parser.add_argument(
        "--local_crop_offset_x",
        type=int,
        default=0,
        help=(
            "Crop offset in x direction after optional resizing. "
            "Negative values move the crop left."
        ),
    )

    parser.add_argument(
        "--local_crop_size",
        type=int,
        default=256,
        help="Final local input crop/resize size. Default: 256.",
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
        choices=["none", *get_tracking_correction_preset_names()],
        help=(
            "Named tracking correction preset. "
            "When combined with individual tracking arguments, explicitly "
            "specified individual arguments override the preset values."
        ),
    )

    parser.add_argument(
        "--enable_tracking_correction",
        action="store_true",
        help=(
            "Generate T_t^corr by correcting the translation trajectory. "
            "Existing pred_tracking remains the uncorrected prior."
        ),
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
        help=(
            "How to initialize the reference travel direction. "
            "first_valid preserves the original behavior. "
            "early_mean averages the first valid steps. "
            "global_mean averages all valid steps."
        ),
    )

    parser.add_argument(
        "--tracking_initial_direction_window",
        type=int,
        default=15,
        help=(
            "Number of valid steps used by "
            "--tracking_initial_direction_method early_mean."
        ),
    )

    parser.add_argument(
        "--tracking_progress_axis_scale",
        type=float,
        default=1.0,
        help=(
            "Scale the corrected center trajectory along the initial progress axis. "
            "1.0 disables this post-correction scaling."
        ),
    )

    parser.add_argument(
        "--tracking_enforce_progress_monotonic",
        action="store_true",
        help=(
            "Force corrected centers to be monotonic along the estimated "
            "progress axis before progress-axis scaling."
        ),
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
        help=(
            "Minimum progress-axis increment as a ratio of median prior step "
            "when --tracking_enforce_progress_monotonic is enabled. "
            "0.0 allows non-decreasing projection."
        ),
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

    parser.set_defaults(
        enable_tracking_correction=False,
        tracking_enforce_progress_monotonic=False,
        tracking_fix_frame_normal_same_side=False,
    )

    return parser


def resolve_input_source(args: argparse.Namespace) -> tuple[Path, str | None]:
    """
    Resolve CLI input arguments to the source path passed to inference.
    """
    if args.input_mode == "video":
        if args.video_path is None:
            raise ValueError("--video_path is required when --input_mode video.")
        return args.video_path, None

    if args.input_mode != "frame_dir":
        raise ValueError(f"Unknown input_mode: {args.input_mode}")

    if args.frame_dir is not None:
        return args.frame_dir, args.frame_video_name

    if args.frame_root is None or args.frame_video_name is None:
        raise ValueError(
            "When --input_mode frame_dir, specify either --frame_dir or both "
            "--frame_root and --frame_video_name."
        )

    frame_dir = (
        Path(args.frame_root)
        / args.frame_video_name
        / args.frame_object_dir_name
    )
    return frame_dir, args.frame_video_name


def main():
    parser = build_parser()
    args = parser.parse_args()
    tracking_correction_override_keys = get_tracking_correction_override_keys_from_argv(
        sys.argv[1:]
    )

    # This script is located at:
    #   DualTrack/pseudo3d/inference/infer_video_to_pseudo3d_h5.py
    # Therefore repo_root is:
    #   DualTrack/
    repo_root = REPO_ROOT

    input_path, frame_video_name = resolve_input_source(args)
    max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames

    run_pseudo3d_inference_for_video(
        repo_root=repo_root,
        video_path=input_path,
        output_h5=args.output_h5,
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
        tracking_enforce_progress_monotonic=args.tracking_enforce_progress_monotonic,
        tracking_progress_monotonic_min_step_ratio=(
            args.tracking_progress_monotonic_min_step_ratio
        ),
        tracking_fix_frame_normal_same_side=args.tracking_fix_frame_normal_same_side,
        tracking_frame_normal_same_side_min_step_ratio=(
            args.tracking_frame_normal_same_side_min_step_ratio
        ),
        tracking_progress_axis_scale=args.tracking_progress_axis_scale,
    )


if __name__ == "__main__":
    main()
