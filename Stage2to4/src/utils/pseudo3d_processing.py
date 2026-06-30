from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Any

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from src.models import get_model
from src.utils.pose import get_global_and_relative_pred_trackings_from_vectors


STAGE2_TRACKING_CORRECTION_PRESET_NAME = "stage2_v1"

# prior 値修正の設定プリセット
STAGE2_TRACKING_CORRECTION_PRESET = {
    "enable_tracking_correction": True,
    "tracking_min_cos": 0.0,
    "tracking_min_step_ratio": 0.25,
    "tracking_max_step_ratio": 2.5,
    "tracking_direction_update_alpha": 0.2,
    "tracking_initial_direction_method": "global_mean",
    "tracking_initial_direction_window": 15,
    "tracking_enforce_progress_monotonic": True,
    "tracking_progress_monotonic_min_step_ratio": 0.02,
    "tracking_fix_frame_normal_same_side": True,
    "tracking_frame_normal_same_side_min_step_ratio": 0.02,
    "tracking_progress_axis_scale": 2.0,
}

TRACKING_CORRECTION_PRESETS = {
    STAGE2_TRACKING_CORRECTION_PRESET_NAME: STAGE2_TRACKING_CORRECTION_PRESET,
    "stage2_global_mono002_sameside002_scale2": STAGE2_TRACKING_CORRECTION_PRESET,
}

TRACKING_CORRECTION_CLI_OPTION_TO_PARAM = {
    "--enable_tracking_correction": "enable_tracking_correction",
    "--disable_tracking_correction": "enable_tracking_correction",
    "--tracking_min_cos": "tracking_min_cos",
    "--tracking_min_step_ratio": "tracking_min_step_ratio",
    "--tracking_max_step_ratio": "tracking_max_step_ratio",
    "--tracking_direction_update_alpha": "tracking_direction_update_alpha",
    "--tracking_initial_direction_method": "tracking_initial_direction_method",
    "--tracking_initial_direction_window": "tracking_initial_direction_window",
    "--tracking_enforce_progress_monotonic": "tracking_enforce_progress_monotonic",
    "--no_tracking_enforce_progress_monotonic": "tracking_enforce_progress_monotonic",
    "--tracking_progress_monotonic_min_step_ratio": (
        "tracking_progress_monotonic_min_step_ratio"
    ),
    "--tracking_fix_frame_normal_same_side": "tracking_fix_frame_normal_same_side",
    "--no_tracking_fix_frame_normal_same_side": "tracking_fix_frame_normal_same_side",
    "--tracking_frame_normal_same_side_min_step_ratio": (
        "tracking_frame_normal_same_side_min_step_ratio"
    ),
    "--tracking_progress_axis_scale": "tracking_progress_axis_scale",
}


def get_tracking_correction_preset_names() -> list[str]:
    """
    Return available named tracking-correction presets.
    """
    return sorted(TRACKING_CORRECTION_PRESETS.keys())


def get_tracking_correction_override_keys_from_argv(argv: list[str]) -> list[str]:
    """
    Return tracking-correction params explicitly specified on the CLI.
    """
    override_keys = set()

    for token in argv:
        option = token.split("=", 1)[0]
        param_name = TRACKING_CORRECTION_CLI_OPTION_TO_PARAM.get(option)
        if param_name is not None:
            override_keys.add(param_name)

    return sorted(override_keys)


def _format_tracking_correction_overrides(
    *,
    override_keys: list[str],
    params: dict[str, Any],
    preset_params: dict[str, Any],
) -> str:
    """
    Format CLI overrides that actually changed preset values.
    """
    entries = []
    for key in override_keys:
        if key not in params or key not in preset_params:
            continue
        if params[key] != preset_params[key]:
            entries.append(f"{key}={params[key]}")
    return ",".join(entries)


def resolve_tracking_correction_params(
    preset_name: str | None,
    params: dict[str, Any],
    *,
    override_keys: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[dict[str, Any], str, str]:
    """
    Resolve tracking-correction params from a named preset plus optional overrides.
    """
    params = dict(params)
    override_keys = sorted(set(override_keys or []))

    if preset_name is None or preset_name == "" or preset_name == "none":
        label = "custom" if params.get("enable_tracking_correction") else "none"
        return params, label, ""

    if preset_name not in TRACKING_CORRECTION_PRESETS:
        names = ", ".join(get_tracking_correction_preset_names())
        raise ValueError(
            f"Unknown tracking correction preset: {preset_name}. "
            f"Available presets: {names}"
        )

    cli_params = dict(params)
    preset_params = dict(TRACKING_CORRECTION_PRESETS[preset_name])
    params.update(preset_params)

    for key in override_keys:
        if key in cli_params:
            params[key] = cli_params[key]

    overrides = _format_tracking_correction_overrides(
        override_keys=override_keys,
        params=params,
        preset_params=preset_params,
    )
    return params, preset_name, overrides


def apply_tracking_correction_preset(
    preset_name: str | None,
    params: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """
    Apply a named tracking-correction preset to a parameter dictionary.

    The explicit "none" and empty values do not change params. In that case the
    returned label is "custom" when correction is enabled, otherwise "none".
    """
    resolved, label, _ = resolve_tracking_correction_params(preset_name, params)
    return resolved, label


def resolve_dualtrack_checkpoint(
    repo_root: Path,
    checkpoint_path: Optional[Path] = None,
) -> Path:
    """
    Resolve the DualTrack checkpoint path.

    Priority:
      1. Explicit checkpoint_path, e.g. CLI --checkpoint
      2. DUALTRACK_CHECKPOINT
      3. DUALTRACK_CHECKPOINT_DIR / dualtrack_final.pt
      4. repo_root / dualtrack_checkpoints / dualtrack_final.pt
    """
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    env_checkpoint = os.environ.get("DUALTRACK_CHECKPOINT")
    if env_checkpoint:
        return Path(env_checkpoint)

    env_checkpoint_dir = os.environ.get("DUALTRACK_CHECKPOINT_DIR")
    if env_checkpoint_dir:
        return Path(env_checkpoint_dir) / "dualtrack_final.pt"

    return Path(repo_root) / "dualtrack_checkpoints" / "dualtrack_final.pt"


def load_dualtrack_model(
    repo_root: Path,
    checkpoint_path: Optional[Path],
    device: torch.device,
):
    """
    Load DualTrack model from repo config and checkpoint.

    Args:
        repo_root:
            DualTrack repository root.
        checkpoint_path:
            Optional checkpoint path.
            If None, resolved from DUALTRACK_CHECKPOINT,
            DUALTRACK_CHECKPOINT_DIR, or the legacy repo-local path.
        device:
            torch device.

    Returns:
        Loaded eval-mode model on device.
    """
    cfg_path = repo_root / "configs" / "model" / "dualtrack.yaml"

    checkpoint_path = resolve_dualtrack_checkpoint(repo_root, checkpoint_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Config:", cfg_path)
    print("Checkpoint:", checkpoint_path)

    cfg = OmegaConf.load(cfg_path)
    cfg.checkpoint = str(checkpoint_path)

    model = get_model(**cfg)
    model = model.to(device)
    model.eval()

    print("\nModel loaded successfully.")
    print("Model class:", model.__class__.__name__)
    print(f"num_params    : {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"num_trainable : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    return model


def read_video_frames_gray(
    video_path: Path,
    stride: int = 1,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read grayscale frames from a video using OpenCV.

    Returns:
        frames_gray:
            uint8 array, shape [N, H, W]
        frame_indices:
            int64 array, shape [N], 0-based original video frame indices
        fps:
            video FPS if available, otherwise 0.0
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    indices = []

    frame_idx = 0
    kept = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx >= start_frame and (frame_idx - start_frame) % stride == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            indices.append(frame_idx)
            kept += 1

            if max_frames is not None and kept >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(
            f"Need at least 2 frames after stride filtering, got {len(frames)}. "
            f"video={video_path}, total_frames={total_frames}, stride={stride}"
        )

    frames_gray = np.stack(frames, axis=0).astype(np.uint8)
    frame_indices = np.asarray(indices, dtype=np.int64)

    print("\nLoaded video frames:")
    print(f"  video_path   : {video_path}")
    print(f"  total_frames : {total_frames}")
    print(f"  fps          : {fps}")
    print(f"  stride       : {stride}")
    print(f"  kept frames  : {len(frames_gray)}")
    print(f"  frame_indices[:10]: {frame_indices[:10].tolist()}")

    return frames_gray, frame_indices, fps


def _infer_video_name_from_frame_dir(frame_dir: Path) -> str:
    """
    Infer video_name from .../{video_name}/object_renamed.
    """
    if frame_dir.name in {"object_renamed", "object_rename"}:
        return frame_dir.parent.name
    return frame_dir.name


def read_frame_dir_frames_gray(
    frame_dir: Path,
    *,
    video_name: str | None = None,
    stride: int = 1,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read grayscale frames from a JPEG/PNG frame directory using OpenCV.

    Expected layout:
      {root}/{video_name}/object_renamed/{video_name}_{frame_number:05d}.{ext}

    Frame numbers are expected to start from 00001. Returned frame_indices are
    0-based, matching video input behavior.
    """
    frame_dir = Path(frame_dir)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if not frame_dir.is_dir():
        raise NotADirectoryError(f"frame_dir is not a directory: {frame_dir}")

    if video_name is None or video_name == "":
        video_name = _infer_video_name_from_frame_dir(frame_dir)

    name_pattern = re.compile(
        rf"^{re.escape(video_name)}_(\d{{5}})\.(jpe?g|png)$",
        re.IGNORECASE,
    )

    indexed_paths: list[tuple[int, Path]] = []
    for path in frame_dir.iterdir():
        if not path.is_file():
            continue
        match = name_pattern.match(path.name)
        if match is None:
            continue
        frame_number = int(match.group(1))
        indexed_paths.append((frame_number, path))

    if not indexed_paths:
        raise FileNotFoundError(
            f"No JPEG/PNG frames matched '{video_name}_00001.jpg/png' style "
            f"names in {frame_dir}"
        )

    indexed_paths.sort(key=lambda item: item[0])
    frame_numbers = [number for number, _ in indexed_paths]

    if frame_numbers[0] != 1:
        raise RuntimeError(
            f"Frame numbering must start from 00001, got {frame_numbers[0]:05d} "
            f"in {frame_dir}"
        )

    expected_numbers = list(range(1, frame_numbers[-1] + 1))
    if frame_numbers != expected_numbers:
        present = set(frame_numbers)
        missing = [number for number in expected_numbers if number not in present]
        preview = ", ".join(f"{number:05d}" for number in missing[:10])
        raise RuntimeError(
            f"Frame numbering has missing files in {frame_dir}. "
            f"First missing frame numbers: {preview}"
        )

    frames = []
    indices = []
    kept = 0

    for frame_number, path in indexed_paths:
        zero_based_idx = frame_number - 1
        if zero_based_idx < start_frame:
            continue
        if (zero_based_idx - start_frame) % stride != 0:
            continue

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Failed to read frame image: {path}")

        frames.append(gray)
        indices.append(zero_based_idx)
        kept += 1

        if max_frames is not None and kept >= max_frames:
            break

    if len(frames) < 2:
        raise RuntimeError(
            f"Need at least 2 frames after stride filtering, got {len(frames)}. "
            f"frame_dir={frame_dir}, total_frames={len(indexed_paths)}, "
            f"stride={stride}"
        )

    frames_gray = np.stack(frames, axis=0).astype(np.uint8)
    frame_indices = np.asarray(indices, dtype=np.int64)

    print("\nLoaded frame directory:")
    print(f"  frame_dir    : {frame_dir}")
    print(f"  video_name   : {video_name}")
    print(f"  total_frames : {len(indexed_paths)}")
    print("  fps          : 0.0")
    print(f"  stride       : {stride}")
    print(f"  kept frames  : {len(frames_gray)}")
    print(f"  frame_indices[:10]: {frame_indices[:10].tolist()}")

    return frames_gray, frame_indices, 0.0


def resize_frames(
    frames: np.ndarray,
    size_hw: tuple[int, int],
) -> np.ndarray:
    """
    Resize [N,H,W] uint8 frames to [N,new_H,new_W].
    """
    out_h, out_w = size_hw
    resized = []

    for frame in frames:
        resized.append(
            cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        )

    return np.stack(resized, axis=0).astype(frames.dtype)


def crop_frames_with_offset(
    frames: np.ndarray,
    crop_size: tuple[int, int] = (256, 256),
    offset_y: int = 0,
    offset_x: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Crop frames with an offset from the image center.

    offset_y < 0:
        crop upward
    offset_y > 0:
        crop downward
    offset_x < 0:
        crop left
    offset_x > 0:
        crop right

    Offsets are measured in pixels after any resizing step.
    """
    crop_h, crop_w = crop_size
    _, h, w = frames.shape

    if h < crop_h or w < crop_w:
        resized = resize_frames(frames, crop_size)
        meta = {
            "local_preprocess_effective": "offset_crop_fallback_resize",
            "local_resized_h": crop_h,
            "local_resized_w": crop_w,
            "local_crop_top": 0,
            "local_crop_left": 0,
            "local_crop_offset_y": offset_y,
            "local_crop_offset_x": offset_x,
            "local_resize_scale": np.nan,
            "local_crop_clipped": True,
        }
        return resized, meta

    center_y = h / 2.0 + offset_y
    center_x = w / 2.0 + offset_x

    top = int(round(center_y - crop_h / 2.0))
    left = int(round(center_x - crop_w / 2.0))

    clipped = False

    if top < 0:
        top = 0
        clipped = True
    if left < 0:
        left = 0
        clipped = True
    if top + crop_h > h:
        top = h - crop_h
        clipped = True
    if left + crop_w > w:
        left = w - crop_w
        clipped = True

    cropped = frames[:, top:top + crop_h, left:left + crop_w]

    meta = {
        "local_preprocess_effective": "offset_crop",
        "local_resized_h": h,
        "local_resized_w": w,
        "local_crop_top": top,
        "local_crop_left": left,
        "local_crop_offset_y": offset_y,
        "local_crop_offset_x": offset_x,
        "local_resize_scale": 1.0,
        "local_crop_clipped": clipped,
    }

    return cropped, meta


def resize_shorter_then_offset_crop(
    frames: np.ndarray,
    target_short_side: int,
    crop_size: tuple[int, int] = (256, 256),
    offset_y: int = 0,
    offset_x: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Resize with preserved aspect ratio so that min(H,W)=target_short_side,
    then crop crop_size with an offset from the resized image center.
    """
    crop_h, crop_w = crop_size

    if target_short_side < min(crop_h, crop_w):
        raise ValueError(
            f"target_short_side must be >= min(crop_size). "
            f"got target_short_side={target_short_side}, crop_size={crop_size}"
        )

    _, h, w = frames.shape
    short = min(h, w)
    scale = target_short_side / float(short)

    resized_h = int(round(h * scale))
    resized_w = int(round(w * scale))

    resized = resize_frames(frames, (resized_h, resized_w))

    cropped, meta = crop_frames_with_offset(
        resized,
        crop_size=crop_size,
        offset_y=offset_y,
        offset_x=offset_x,
    )

    meta.update(
        {
            "local_preprocess_effective": "resize_shorter_then_offset_crop",
            "local_target_short_side": target_short_side,
            "local_resized_h": resized_h,
            "local_resized_w": resized_w,
            "local_resize_scale": scale,
            "local_crop_offset_y": offset_y,
            "local_crop_offset_x": offset_x,
        }
    )

    return cropped, meta


def preprocess_local_frames(
    frames: np.ndarray,
    mode: str,
    crop_size: tuple[int, int] = (256, 256),
    target_short_side: int = 448,
    offset_y: int = -96,
    offset_x: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Create local_encoder_images.

    Args:
        frames:
            uint8 [N,H,W]
        mode:
            center_crop
            resize_shorter_then_center_crop
            resize
        crop_size:
            final local input size, usually (256,256)
        target_short_side:
            used for resize_shorter_then_center_crop
        offset_y, offset_x:
            crop offset after optional resizing

    Returns:
        local_frames:
            uint8 [N, crop_h, crop_w]
        local_preprocess_meta:
            preprocessing metadata saved into h5 attrs
    """
    if mode == "center_crop":
        local_frames, meta = crop_frames_with_offset(
            frames,
            crop_size=crop_size,
            offset_y=offset_y,
            offset_x=offset_x,
        )
        meta["local_preprocess"] = "center_crop"
        meta["local_target_short_side"] = -1
        return local_frames, meta

    if mode == "resize_shorter_then_center_crop":
        local_frames, meta = resize_shorter_then_offset_crop(
            frames,
            target_short_side=target_short_side,
            crop_size=crop_size,
            offset_y=offset_y,
            offset_x=offset_x,
        )
        meta["local_preprocess"] = "resize_shorter_then_center_crop"
        return local_frames, meta

    if mode == "resize":
        crop_h, crop_w = crop_size
        local_frames = resize_frames(frames, (crop_h, crop_w))
        meta = {
            "local_preprocess": "resize",
            "local_preprocess_effective": "resize",
            "local_target_short_side": -1,
            "local_resized_h": crop_h,
            "local_resized_w": crop_w,
            "local_crop_top": 0,
            "local_crop_left": 0,
            "local_crop_offset_y": 0,
            "local_crop_offset_x": 0,
            "local_resize_scale": np.nan,
            "local_crop_clipped": False,
        }
        return local_frames, meta

    raise ValueError(
        f"Unknown local_preprocess='{mode}'. "
        "Choose from: center_crop, resize_shorter_then_center_crop, resize"
    )


def frames_to_model_tensor(
    frames_uint8: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert uint8 frames [N,H,W] to float tensor [1,N,1,H,W] in [0,1].
    """
    x = torch.from_numpy(frames_uint8).float() / 255.0
    x = x[:, None, :, :]  # [N,1,H,W]
    x = x[None, ...]      # [1,N,1,H,W]
    return x.to(device)


def make_pixel_to_image_matrix(
    width: int,
    height: int,
    spacing_x: float = 1.0,
    spacing_y: float = 1.0,
) -> np.ndarray:
    """
    Create a simple pixel-to-image matrix for a 2D image plane.

    Pixel coordinate:
        [x, y, z=0, 1]

    Image coordinate:
        centered at the image center, with physical units defined by spacing_x/y.

    This is a temporary calibration for pseudo-3D testing.
    Replace with real ultrasound calibration if available.
    """
    pixel_to_image = np.eye(4, dtype=np.float32)
    pixel_to_image[0, 0] = spacing_x
    pixel_to_image[1, 1] = spacing_y
    pixel_to_image[2, 2] = 1.0

    pixel_to_image[0, 3] = -width * spacing_x / 2.0
    pixel_to_image[1, 3] = -height * spacing_y / 2.0

    return pixel_to_image


def make_frame_corners_world(
    pred_tracking: np.ndarray,
    pixel_to_image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Create 3D world coordinates of the four corners of each frame.

    Args:
        pred_tracking:
            [N,4,4]
        pixel_to_image:
            [4,4]
        width, height:
            image plane size used for visualization

    Returns:
        frame_corners_world:
            [N,4,3]
    """
    corners_px = np.array(
        [
            [0, 0, 0, 1],
            [width - 1, 0, 0, 1],
            [width - 1, height - 1, 0, 1],
            [0, height - 1, 0, 1],
        ],
        dtype=np.float32,
    )

    corners_img = (pixel_to_image @ corners_px.T).T

    all_corners = []
    for i in range(pred_tracking.shape[0]):
        corners_world_h = (pred_tracking[i] @ corners_img.T).T
        all_corners.append(corners_world_h[:, :3])

    return np.stack(all_corners, axis=0).astype(np.float32)


def _estimate_initial_tracking_direction(
    steps: np.ndarray,
    step_lengths: np.ndarray,
    valid_step_mask: np.ndarray,
    *,
    method: str,
    window: int,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate the initial reference travel direction for trajectory correction.
    """
    valid_indices = np.flatnonzero(valid_step_mask)
    if valid_indices.size == 0:
        raise ValueError("Cannot estimate initial direction without valid steps.")

    first_valid_idx = int(valid_indices[0])
    first_valid_dir = steps[first_valid_idx] / step_lengths[first_valid_idx]

    meta: dict[str, Any] = {
        "initial_direction_method": method,
        "initial_direction_window": int(window),
        "initial_direction_num_candidates": 1,
        "initial_direction_norm": 1.0,
        "initial_direction_fallback_to_first_valid": False,
    }

    if method == "first_valid":
        return first_valid_dir, meta

    if method == "early_mean":
        if window <= 0:
            raise ValueError(
                f"initial_direction_window must be > 0 for early_mean, got {window}"
            )
        candidate_indices = valid_indices[:window]
    elif method == "global_mean":
        candidate_indices = valid_indices
    else:
        raise ValueError(
            f"Unknown initial_direction_method='{method}'. "
            "Choose from: first_valid, early_mean, global_mean"
        )

    candidate_dirs = steps[candidate_indices] / step_lengths[candidate_indices, None]
    mean_dir = candidate_dirs.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_dir))

    meta["initial_direction_num_candidates"] = int(candidate_indices.size)
    meta["initial_direction_norm"] = mean_norm

    if mean_norm <= eps or not np.isfinite(mean_norm):
        meta["initial_direction_fallback_to_first_valid"] = True
        return first_valid_dir, meta

    return mean_dir / mean_norm, meta


def _apply_progress_axis_scale(
    centers: np.ndarray,
    axis: np.ndarray,
    *,
    scale: float,
    axis_source: str,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Scale center trajectory only along the estimated progress axis.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"progress_axis_scale must be finite and > 0, got {scale}")

    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= eps or not np.isfinite(axis_norm):
        raise ValueError(f"Invalid progress axis norm: {axis_norm}")

    axis = axis / axis_norm
    origin = centers[0]
    relative = centers - origin
    projection = relative @ axis
    extent_before = float(projection.max() - projection.min())

    meta: dict[str, Any] = {
        "progress_axis_scale": scale,
        "progress_axis_scale_enabled": bool(abs(scale - 1.0) > eps),
        "progress_axis_scale_axis_source": axis_source,
        "progress_axis_scale_axis_norm": axis_norm,
        "progress_axis_scale_extent_before": extent_before,
        "progress_axis_scale_extent_after": extent_before,
        "progress_axis_scale_displacement_mean": 0.0,
        "progress_axis_scale_displacement_max": 0.0,
    }

    if not meta["progress_axis_scale_enabled"]:
        return centers, meta

    parallel = projection[:, None] * axis[None, :]
    perpendicular = relative - parallel
    scaled_centers = origin + perpendicular + scale * parallel

    scaled_projection = (scaled_centers - origin) @ axis
    displacement = np.linalg.norm(scaled_centers - centers, axis=1)

    meta["progress_axis_scale_extent_after"] = float(
        scaled_projection.max() - scaled_projection.min()
    )
    meta["progress_axis_scale_displacement_mean"] = float(displacement.mean())
    meta["progress_axis_scale_displacement_max"] = float(displacement.max())

    return scaled_centers, meta


def _enforce_progress_axis_monotonicity(
    centers: np.ndarray,
    axis: np.ndarray,
    *,
    enable: bool,
    min_step: float,
    axis_source: str,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Enforce monotonic center positions along the estimated progress axis.
    """
    min_step = float(min_step)
    if min_step < 0.0 or not np.isfinite(min_step):
        raise ValueError(f"progress_monotonic_min_step must be finite and >= 0, got {min_step}")

    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= eps or not np.isfinite(axis_norm):
        raise ValueError(f"Invalid progress monotonic axis norm: {axis_norm}")

    axis = axis / axis_norm
    origin = centers[0]
    relative = centers - origin
    projection = relative @ axis

    meta: dict[str, Any] = {
        "progress_monotonic_enabled": bool(enable),
        "progress_monotonic_axis_source": axis_source,
        "progress_monotonic_axis_norm": axis_norm,
        "progress_monotonic_min_step": min_step,
        "num_progress_monotonic_adjusted": 0,
        "progress_monotonic_displacement_mean": 0.0,
        "progress_monotonic_displacement_max": 0.0,
    }

    if not enable:
        return centers, meta

    monotonic_projection = projection.copy()
    for i in range(1, len(monotonic_projection)):
        min_allowed = monotonic_projection[i - 1] + min_step
        if monotonic_projection[i] < min_allowed:
            monotonic_projection[i] = min_allowed
            meta["num_progress_monotonic_adjusted"] += 1

    perpendicular = relative - projection[:, None] * axis[None, :]
    monotonic_centers = origin + perpendicular + monotonic_projection[:, None] * axis[None, :]
    displacement = np.linalg.norm(monotonic_centers - centers, axis=1)

    meta["progress_monotonic_displacement_mean"] = float(displacement.mean())
    meta["progress_monotonic_displacement_max"] = float(displacement.max())

    return monotonic_centers, meta


def _get_frame_normal_from_tracking(
    pred_tracking: np.ndarray,
    frame_idx: int,
    *,
    eps: float,
) -> np.ndarray:
    """
    Estimate frame plane normal from the tracking rotation.
    """
    normal = pred_tracking[frame_idx, :3, :3].astype(np.float64) @ np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= eps or not np.isfinite(normal_norm):
        raise ValueError(
            f"Invalid frame normal for frame_idx={frame_idx}: norm={normal_norm}"
        )
    return normal / normal_norm


def _fix_frame_normal_same_side_steps(
    centers: np.ndarray,
    pred_tracking: np.ndarray,
    *,
    enable: bool,
    min_step: float,
    reference_direction: np.ndarray | None = None,
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Push future centers when neighboring frames lie on the same side of frame n.

    For frame n, compare the normal components of c[n-1]-c[n] and c[n+1]-c[n].
    If both are on the same side, c[n+1:] is shifted so c[n+1] moves to the
    opposite side of frame n by at least min_step along frame n's normal.
    For frame 0, -reference_direction is used as the virtual previous side.
    """
    min_step = float(min_step)
    if min_step < 0.0 or not np.isfinite(min_step):
        raise ValueError(
            "frame_normal_same_side_min_step must be finite and >= 0, "
            f"got {min_step}"
        )

    meta: dict[str, Any] = {
        "frame_normal_same_side_enabled": bool(enable),
        "frame_normal_same_side_min_step": min_step,
        "frame_normal_same_side_boundary_enabled": bool(
            reference_direction is not None
        ),
        "frame_normal_same_side_boundary_reference_norm": 0.0,
        "num_frame_normal_same_side_boundary_detected": 0,
        "num_frame_normal_same_side_boundary_corrected": 0,
        "num_frame_normal_same_side_detected": 0,
        "num_frame_normal_same_side_corrected": 0,
        "frame_normal_same_side_displacement_mean": 0.0,
        "frame_normal_same_side_displacement_max": 0.0,
    }

    if not enable or len(centers) < 2:
        return centers, meta

    corrected_centers = centers.copy()
    original_centers = centers.copy()

    reference_direction_norm = 0.0
    if reference_direction is not None:
        reference_direction = np.asarray(reference_direction, dtype=np.float64)
        reference_direction_norm = float(np.linalg.norm(reference_direction))
        if reference_direction_norm > eps and np.isfinite(reference_direction_norm):
            reference_direction = reference_direction / reference_direction_norm
        else:
            reference_direction = None

    meta["frame_normal_same_side_boundary_reference_norm"] = reference_direction_norm

    for i in range(0, len(corrected_centers) - 1):
        normal = _get_frame_normal_from_tracking(pred_tracking, i, eps=eps)

        is_boundary = i == 0
        if is_boundary:
            if reference_direction is None:
                continue
            prev_from_i = -reference_direction
        else:
            prev_from_i = corrected_centers[i - 1] - corrected_centers[i]

        next_from_i = corrected_centers[i + 1] - corrected_centers[i]

        prev_normal_delta = float(np.dot(prev_from_i, normal))
        next_normal_delta = float(np.dot(next_from_i, normal))

        if abs(prev_normal_delta) <= eps or abs(next_normal_delta) <= eps:
            continue

        if prev_normal_delta * next_normal_delta < 0.0:
            continue

        if is_boundary:
            meta["num_frame_normal_same_side_boundary_detected"] += 1
        else:
            meta["num_frame_normal_same_side_detected"] += 1

        target_next_delta = -np.sign(prev_normal_delta) * min_step
        correction_amount = target_next_delta - next_normal_delta
        delta = correction_amount * normal

        corrected_centers[i + 1:] += delta
        if is_boundary:
            meta["num_frame_normal_same_side_boundary_corrected"] += 1
        else:
            meta["num_frame_normal_same_side_corrected"] += 1

    displacement = np.linalg.norm(corrected_centers - original_centers, axis=1)
    meta["frame_normal_same_side_displacement_mean"] = float(displacement.mean())
    meta["frame_normal_same_side_displacement_max"] = float(displacement.max())

    return corrected_centers, meta


def correct_tracking_temporal_geometry(
    pred_tracking: np.ndarray,
    *,
    correction_preset: str = "custom",
    correction_overrides: str = "",
    enable: bool = True,
    min_cos: float = -0.1,
    min_step_ratio: float = 0.25,
    max_step_ratio: float = 2.0,
    direction_update_alpha: float = 0.2,
    initial_direction_method: str = "first_valid",
    initial_direction_window: int = 15,
    enforce_progress_monotonic: bool = False,
    progress_monotonic_min_step_ratio: float = 0.0,
    fix_frame_normal_same_side: bool = False,
    frame_normal_same_side_min_step_ratio: float = 0.0,
    progress_axis_scale: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    Correct temporal translation geometry while preserving prior rotations.

    The input tracking is treated as T_t^prior. This function creates T_t^corr
    by rebuilding only the center trajectory from clipped/filtered adjacent
    translation steps. Rotations are copied from the prior matrices.
    """
    pred_tracking = np.asarray(pred_tracking)
    if pred_tracking.ndim != 3 or pred_tracking.shape[1:] != (4, 4):
        raise ValueError(
            "pred_tracking must have shape [N,4,4], "
            f"got {pred_tracking.shape}"
        )

    pred_tracking_corr = pred_tracking.astype(np.float32, copy=True)
    centers = pred_tracking[:, :3, 3].astype(np.float64, copy=False)
    n = pred_tracking.shape[0]

    steps = np.diff(centers, axis=0) if n > 1 else np.zeros((0, 3), dtype=np.float64)
    finite_steps = np.isfinite(steps).all(axis=1)
    step_lengths = np.linalg.norm(np.where(finite_steps[:, None], steps, 0.0), axis=1)
    valid_step_mask = finite_steps & (step_lengths > eps)
    valid_step_lengths = step_lengths[valid_step_mask]

    # TODO: Evaluate which valid-step filtering and initial direction method
    # should become the default after multi-video metadata review.
    if valid_step_lengths.size > 0:
        median_step = float(np.median(valid_step_lengths))
        prior_mean = float(np.mean(valid_step_lengths))
        prior_median = median_step
    else:
        median_step = 0.0
        prior_mean = 0.0
        prior_median = 0.0

    min_step = float(min_step_ratio * median_step)
    max_step = float(max_step_ratio * median_step)

    correction_meta = {
        "tracking_correction_preset": str(correction_preset),
        "tracking_correction_overrides": str(correction_overrides),
        "correction_enabled": bool(enable),
        "median_step": median_step,
        "min_step": min_step,
        "max_step": max_step,
        "min_cos": float(min_cos),
        "min_step_ratio": float(min_step_ratio),
        "max_step_ratio": float(max_step_ratio),
        "direction_update_alpha": float(direction_update_alpha),
        "initial_direction_method": initial_direction_method,
        "initial_direction_window": int(initial_direction_window),
        "initial_direction_num_candidates": 0,
        "initial_direction_norm": 0.0,
        "initial_direction_fallback_to_first_valid": False,
        "progress_axis_scale": float(progress_axis_scale),
        "progress_axis_scale_enabled": False,
        "progress_axis_scale_axis_source": "initial_direction",
        "progress_axis_scale_axis_norm": 0.0,
        "progress_axis_scale_extent_before": 0.0,
        "progress_axis_scale_extent_after": 0.0,
        "progress_axis_scale_displacement_mean": 0.0,
        "progress_axis_scale_displacement_max": 0.0,
        "progress_monotonic_enabled": bool(enforce_progress_monotonic),
        "progress_monotonic_axis_source": "initial_direction",
        "progress_monotonic_axis_norm": 0.0,
        "progress_monotonic_min_step_ratio": float(progress_monotonic_min_step_ratio),
        "progress_monotonic_min_step": float(progress_monotonic_min_step_ratio * median_step),
        "num_progress_monotonic_adjusted": 0,
        "progress_monotonic_displacement_mean": 0.0,
        "progress_monotonic_displacement_max": 0.0,
        "frame_normal_same_side_enabled": bool(fix_frame_normal_same_side),
        "frame_normal_same_side_min_step_ratio": float(
            frame_normal_same_side_min_step_ratio
        ),
        "frame_normal_same_side_min_step": float(
            frame_normal_same_side_min_step_ratio * median_step
        ),
        "num_frame_normal_same_side_detected": 0,
        "num_frame_normal_same_side_corrected": 0,
        "frame_normal_same_side_displacement_mean": 0.0,
        "frame_normal_same_side_displacement_max": 0.0,
        "num_reverse_steps": 0,
        "num_short_steps": 0,
        "num_long_steps": 0,
        "num_zero_or_invalid_steps": int((~valid_step_mask).sum()),
        "step_lengths_prior_mean": prior_mean,
        "step_lengths_prior_median": prior_median,
        "step_lengths_corr_mean": prior_mean,
        "step_lengths_corr_median": prior_median,
    }

    if not enable or n <= 1 or valid_step_lengths.size == 0 or median_step <= eps:
        pred_tracking_corr[:, 3, :] = np.asarray([0, 0, 0, 1], dtype=np.float32)
        return pred_tracking_corr, correction_meta

    alpha = float(np.clip(direction_update_alpha, 0.0, 1.0))
    corrected_centers = np.empty_like(centers)
    corrected_centers[0] = centers[0]

    v_ref, initial_direction_meta = _estimate_initial_tracking_direction(
        steps,
        step_lengths,
        valid_step_mask,
        method=initial_direction_method,
        window=initial_direction_window,
        eps=eps,
    )
    correction_meta.update(initial_direction_meta)
    progress_axis = v_ref.copy()

    for i, step in enumerate(steps):
        is_valid = bool(valid_step_mask[i])

        if not is_valid:
            corrected_step = v_ref * median_step
        else:
            step_len = float(step_lengths[i])
            step_dir = step / step_len
            cos = float(np.dot(step_dir, v_ref))

            if cos < min_cos:
                correction_meta["num_reverse_steps"] += 1
                corrected_step = v_ref * median_step
            else:
                clipped_len = step_len
                if step_len < min_step:
                    correction_meta["num_short_steps"] += 1
                    clipped_len = min_step
                elif step_len > max_step:
                    correction_meta["num_long_steps"] += 1
                    clipped_len = max_step

                corrected_step = step_dir * clipped_len

                corrected_dir = corrected_step / max(np.linalg.norm(corrected_step), eps)
                updated_ref = (1.0 - alpha) * v_ref + alpha * corrected_dir
                updated_norm = float(np.linalg.norm(updated_ref))
                if updated_norm > eps and np.isfinite(updated_norm):
                    v_ref = updated_ref / updated_norm

        corrected_centers[i + 1] = corrected_centers[i] + corrected_step

    monotonic_min_step = float(progress_monotonic_min_step_ratio * median_step)
    corrected_centers, progress_monotonic_meta = _enforce_progress_axis_monotonicity(
        corrected_centers,
        progress_axis,
        enable=enforce_progress_monotonic,
        min_step=monotonic_min_step,
        axis_source="initial_direction",
        eps=eps,
    )
    progress_monotonic_meta["progress_monotonic_min_step_ratio"] = float(
        progress_monotonic_min_step_ratio
    )
    correction_meta.update(progress_monotonic_meta)

    frame_normal_min_step = float(frame_normal_same_side_min_step_ratio * median_step)
    corrected_centers, frame_normal_same_side_meta = (
        _fix_frame_normal_same_side_steps(
            corrected_centers,
            pred_tracking,
            enable=fix_frame_normal_same_side,
            min_step=frame_normal_min_step,
            reference_direction=progress_axis,
            eps=eps,
        )
    )
    frame_normal_same_side_meta["frame_normal_same_side_min_step_ratio"] = float(
        frame_normal_same_side_min_step_ratio
    )
    correction_meta.update(frame_normal_same_side_meta)

    corrected_centers, progress_scale_meta = _apply_progress_axis_scale(
        corrected_centers,
        progress_axis,
        scale=progress_axis_scale,
        axis_source="initial_direction",
        eps=eps,
    )
    correction_meta.update(progress_scale_meta)

    corr_steps = np.diff(corrected_centers, axis=0)
    corr_step_lengths = np.linalg.norm(corr_steps, axis=1)
    correction_meta["step_lengths_corr_mean"] = float(np.mean(corr_step_lengths))
    correction_meta["step_lengths_corr_median"] = float(np.median(corr_step_lengths))

    pred_tracking_corr[:, :3, :3] = pred_tracking[:, :3, :3]
    pred_tracking_corr[:, :3, 3] = corrected_centers.astype(np.float32)
    pred_tracking_corr[:, 3, :] = np.asarray([0, 0, 0, 1], dtype=np.float32)

    return pred_tracking_corr, correction_meta


def make_frame_mesh_from_corners(
    frame_corners_world: np.ndarray,
    connect_slices: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert [N,4,3] frame corners into mesh vertices/faces.

    If connect_slices=False:
        faces contain only one quad per frame: [N,4]

    If connect_slices=True:
        also create side faces between adjacent frames.
    """
    n = frame_corners_world.shape[0]
    vertices = frame_corners_world.reshape(n * 4, 3).astype(np.float32)

    faces = []

    for i in range(n):
        base = i * 4
        faces.append([base + 0, base + 1, base + 2, base + 3])

    if connect_slices:
        for i in range(n - 1):
            a = i * 4
            b = (i + 1) * 4

            faces.append([a + 0, a + 1, b + 1, b + 0])
            faces.append([a + 1, a + 2, b + 2, b + 1])
            faces.append([a + 2, a + 3, b + 3, b + 2])
            faces.append([a + 3, a + 0, b + 0, b + 3])

    return vertices, np.asarray(faces, dtype=np.int32)


def extract_threshold_point_cloud_placeholder(
    raw_images: np.ndarray,
    pred_tracking: np.ndarray,
    pixel_to_image: np.ndarray,
    threshold: Optional[float] = None,
):
    """
    Reserved for future implementation.

    Intended future output:
        point_cloud_xyz       [K,3]
        point_cloud_intensity [K]
        point_cloud_frame_idx [K]
        point_cloud_pixel_xy  [K,2]

    Currently returns None to indicate that no point cloud is saved.
    """
    return None


def _write_h5_attr(f: h5py.File, key: str, value: Any):
    """
    Safely write simple metadata values into h5 attrs.
    """
    if isinstance(value, (np.bool_, bool)):
        f.attrs[key] = bool(value)
    elif isinstance(value, (np.integer, int)):
        f.attrs[key] = int(value)
    elif isinstance(value, (np.floating, float)):
        f.attrs[key] = float(value)
    elif value is None:
        f.attrs[key] = "None"
    else:
        f.attrs[key] = str(value)


def save_pseudo3d_h5(
    output_path: Path,
    *,
    video_path: Path,
    input_mode: str,
    source_name: str,
    raw_images: np.ndarray,
    global_input: np.ndarray,
    local_input: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    stride: int,
    pred_rel_pose_vecs: np.ndarray,
    pred_tracking: np.ndarray,
    pred_tracking_corr: np.ndarray,
    pixel_to_image: np.ndarray,
    spacing: np.ndarray,
    dimensions: np.ndarray,
    frame_corners_world: np.ndarray,
    frame_mesh_vertices: np.ndarray,
    frame_mesh_faces: np.ndarray,
    frame_corners_world_corr: np.ndarray,
    frame_mesh_vertices_corr: np.ndarray,
    frame_mesh_faces_corr: np.ndarray,
    connect_slices: bool,
    local_preprocess_meta: dict[str, Any],
    tracking_correction_meta: dict[str, Any],
):
    """
    Save pseudo-3D inference result into h5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Original / input data
        f.create_dataset("raw_images", data=raw_images, compression="gzip")
        f.create_dataset("global_encoder_images", data=global_input, compression="gzip")
        f.create_dataset("local_encoder_images", data=local_input, compression="gzip")
        f.create_dataset("frame_indices", data=frame_indices)

        # DualTrack outputs
        f.create_dataset("pred_rel_pose_vecs", data=pred_rel_pose_vecs)
        f.create_dataset("pred_tracking", data=pred_tracking)
        f.create_dataset("pred_tracking_prior", data=pred_tracking)
        f.create_dataset("pred_tracking_corr", data=pred_tracking_corr)

        # Geometry / calibration-like information
        f.create_dataset("pixel_to_image", data=pixel_to_image)
        f.create_dataset("spacing", data=spacing)
        f.create_dataset("dimensions", data=dimensions)

        # Plane polygon / mesh data for visualization
        f.create_dataset("frame_corners_world", data=frame_corners_world)
        f.create_dataset("frame_mesh_vertices", data=frame_mesh_vertices)
        f.create_dataset("frame_mesh_faces", data=frame_mesh_faces)
        f.create_dataset("frame_corners_world_corr", data=frame_corners_world_corr)
        f.create_dataset("frame_mesh_vertices_corr", data=frame_mesh_vertices_corr)
        f.create_dataset("frame_mesh_faces_corr", data=frame_mesh_faces_corr)

        correction_group = f.create_group("tracking_correction")
        for key, value in tracking_correction_meta.items():
            if isinstance(value, (np.bool_, bool)):
                correction_group.attrs[key] = bool(value)
            elif isinstance(value, (np.integer, int)):
                correction_group.attrs[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                correction_group.attrs[key] = float(value)
            elif value is None:
                correction_group.attrs[key] = "None"
            else:
                correction_group.attrs[key] = str(value)

        # Placeholder for future point cloud extraction
        pc_group = f.create_group("point_cloud")
        pc_group.attrs["status"] = "not_generated"
        pc_group.attrs["note"] = (
            "Reserved for threshold/contour-derived point cloud. "
            "No point cloud is saved in this preliminary script."
        )

        # Metadata
        f.attrs["source_video"] = str(video_path)
        f.attrs["source_path"] = str(video_path)
        f.attrs["source_name"] = str(source_name)
        f.attrs["input_mode"] = str(input_mode)
        f.attrs["fps"] = fps
        f.attrs["stride"] = stride
        f.attrs["num_frames"] = int(raw_images.shape[0])
        f.attrs["raw_height"] = int(raw_images.shape[1])
        f.attrs["raw_width"] = int(raw_images.shape[2])
        f.attrs["global_input_shape"] = str(global_input.shape)
        f.attrs["local_input_shape"] = str(local_input.shape)
        f.attrs["connect_slices"] = bool(connect_slices)
        f.attrs["description"] = (
            "Pseudo-3D teacher data generated from a video using DualTrack. "
            "pred_tracking is generated from predicted relative pose vectors."
        )

        for key, value in local_preprocess_meta.items():
            _write_h5_attr(f, key, value)

    print(f"\nSaved pseudo-3D h5: {output_path}")


def run_pseudo3d_inference_for_video(
    *,
    repo_root: Path,
    video_path: Path,
    output_h5: Path,
    input_mode: str = "video",
    frame_video_name: str | None = None,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    stride: int = 2,
    start_frame: int = 0,
    max_frames: Optional[int] = 64,
    local_preprocess: str = "resize_shorter_then_center_crop",
    local_target_short_side: int = 448,
    local_crop_offset_y: int = -96,
    local_crop_offset_x: int = 0,
    local_crop_size: int = 256,
    spacing_x: float = 1.0,
    spacing_y: float = 1.0,
    connect_slices: bool = False,
    tracking_correction_preset: str | None = None,
    tracking_correction_override_keys: (
        list[str] | tuple[str, ...] | set[str] | None
    ) = None,
    enable_tracking_correction: bool = False,
    tracking_min_cos: float = -0.1,
    tracking_min_step_ratio: float = 0.25,
    tracking_max_step_ratio: float = 2.0,
    tracking_direction_update_alpha: float = 0.2,
    tracking_initial_direction_method: str = "first_valid",
    tracking_initial_direction_window: int = 15,
    tracking_enforce_progress_monotonic: bool = False,
    tracking_progress_monotonic_min_step_ratio: float = 0.0,
    tracking_fix_frame_normal_same_side: bool = False,
    tracking_frame_normal_same_side_min_step_ratio: float = 0.0,
    tracking_progress_axis_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Run DualTrack on one video and export pseudo-3D h5.

    This function intentionally does not export textured OBJ or alpha textures.
    Visualization/export should be done from the resulting h5.
    """
    repo_root = Path(repo_root)
    video_path = Path(video_path)
    input_mode = str(input_mode)
    output_h5 = Path(output_h5)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

    (
        tracking_params,
        tracking_correction_preset_label,
        tracking_correction_overrides,
    ) = resolve_tracking_correction_params(
        tracking_correction_preset,
        {
            "enable_tracking_correction": enable_tracking_correction,
            "tracking_min_cos": tracking_min_cos,
            "tracking_min_step_ratio": tracking_min_step_ratio,
            "tracking_max_step_ratio": tracking_max_step_ratio,
            "tracking_direction_update_alpha": tracking_direction_update_alpha,
            "tracking_initial_direction_method": tracking_initial_direction_method,
            "tracking_initial_direction_window": tracking_initial_direction_window,
            "tracking_enforce_progress_monotonic": tracking_enforce_progress_monotonic,
            "tracking_progress_monotonic_min_step_ratio": (
                tracking_progress_monotonic_min_step_ratio
            ),
            "tracking_fix_frame_normal_same_side": tracking_fix_frame_normal_same_side,
            "tracking_frame_normal_same_side_min_step_ratio": (
                tracking_frame_normal_same_side_min_step_ratio
            ),
            "tracking_progress_axis_scale": tracking_progress_axis_scale,
        },
        override_keys=tracking_correction_override_keys,
    )
    enable_tracking_correction = tracking_params["enable_tracking_correction"]
    tracking_min_cos = tracking_params["tracking_min_cos"]
    tracking_min_step_ratio = tracking_params["tracking_min_step_ratio"]
    tracking_max_step_ratio = tracking_params["tracking_max_step_ratio"]
    tracking_direction_update_alpha = tracking_params["tracking_direction_update_alpha"]
    tracking_initial_direction_method = tracking_params["tracking_initial_direction_method"]
    tracking_initial_direction_window = tracking_params["tracking_initial_direction_window"]
    tracking_enforce_progress_monotonic = tracking_params[
        "tracking_enforce_progress_monotonic"
    ]
    tracking_progress_monotonic_min_step_ratio = tracking_params[
        "tracking_progress_monotonic_min_step_ratio"
    ]
    tracking_fix_frame_normal_same_side = tracking_params[
        "tracking_fix_frame_normal_same_side"
    ]
    tracking_frame_normal_same_side_min_step_ratio = tracking_params[
        "tracking_frame_normal_same_side_min_step_ratio"
    ]
    tracking_progress_axis_scale = tracking_params["tracking_progress_axis_scale"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)

    model = load_dualtrack_model(repo_root, checkpoint_path, device)

    # ------------------------------------------------------------
    # 1. Read and sample input frames
    # ------------------------------------------------------------
    if input_mode == "video":
        raw_images, frame_indices, fps = read_video_frames_gray(
            video_path,
            stride=stride,
            start_frame=start_frame,
            max_frames=max_frames,
        )
        source_name = video_path.stem
    elif input_mode == "frame_dir":
        raw_images, frame_indices, fps = read_frame_dir_frames_gray(
            video_path,
            video_name=frame_video_name,
            stride=stride,
            start_frame=start_frame,
            max_frames=max_frames,
        )
        source_name = frame_video_name or _infer_video_name_from_frame_dir(video_path)
    else:
        raise ValueError("input_mode must be one of: video, frame_dir")

    n, raw_h, raw_w = raw_images.shape
    print("\nRaw images:")
    print(f"  shape: {raw_images.shape}")
    print(f"  dtype: {raw_images.dtype}")

    # ------------------------------------------------------------
    # 2. Create DualTrack inputs
    # ------------------------------------------------------------
    global_frames = resize_frames(raw_images, (224, 224))

    local_frames, local_preprocess_meta = preprocess_local_frames(
        raw_images,
        mode=local_preprocess,
        crop_size=(local_crop_size, local_crop_size),
        target_short_side=local_target_short_side,
        offset_y=local_crop_offset_y,
        offset_x=local_crop_offset_x,
    )

    print("\nLocal preprocessing:")
    for k, v in local_preprocess_meta.items():
        print(f"  {k}: {v}")

    global_x = frames_to_model_tensor(global_frames, device)
    local_x = frames_to_model_tensor(local_frames, device)

    print("\nModel inputs:")
    print(
        f"  global_x: shape={tuple(global_x.shape)}, "
        f"dtype={global_x.dtype}, device={global_x.device}"
    )
    print(
        f"  local_x : shape={tuple(local_x.shape)}, "
        f"dtype={local_x.dtype}, device={local_x.device}"
    )

    # ------------------------------------------------------------
    # 3. Run DualTrack inference
    # ------------------------------------------------------------
    print("\nRunning DualTrack inference...")

    with torch.inference_mode():
        pred = model(global_x, local_x)

    expected_shape = (1, n - 1, 6)
    print(f"  pred shape: {tuple(pred.shape)}")
    print(f"  expected  : {expected_shape}")

    if tuple(pred.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected prediction shape: got {tuple(pred.shape)}, "
            f"expected {expected_shape}"
        )

    if not torch.isfinite(pred).all():
        raise RuntimeError("Prediction contains NaN or Inf.")

    pred_rel_pose_vecs = pred[0].detach().float().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------
    # 4. Convert [N-1,6] relative pose vectors to [N,4,4] tracking
    # ------------------------------------------------------------
    pred_tracking = get_global_and_relative_pred_trackings_from_vectors(
        pred_rel_pose_vecs
    )[0].astype(np.float32)

    print("\nPredicted tracking:")
    print(f"  pred_rel_pose_vecs: {pred_rel_pose_vecs.shape}")
    print(f"  pred_tracking     : {pred_tracking.shape}")

    # ------------------------------------------------------------
    # 5. Correct translation trajectory for T_t^corr
    # ------------------------------------------------------------
    pred_tracking_corr, tracking_correction_meta = correct_tracking_temporal_geometry(
        pred_tracking,
        correction_preset=tracking_correction_preset_label,
        correction_overrides=tracking_correction_overrides,
        enable=enable_tracking_correction,
        min_cos=tracking_min_cos,
        min_step_ratio=tracking_min_step_ratio,
        max_step_ratio=tracking_max_step_ratio,
        direction_update_alpha=tracking_direction_update_alpha,
        initial_direction_method=tracking_initial_direction_method,
        initial_direction_window=tracking_initial_direction_window,
        enforce_progress_monotonic=tracking_enforce_progress_monotonic,
        progress_monotonic_min_step_ratio=tracking_progress_monotonic_min_step_ratio,
        fix_frame_normal_same_side=tracking_fix_frame_normal_same_side,
        frame_normal_same_side_min_step_ratio=(
            tracking_frame_normal_same_side_min_step_ratio
        ),
        progress_axis_scale=tracking_progress_axis_scale,
    )

    print("\nTracking correction:")
    for key, value in tracking_correction_meta.items():
        print(f"  {key}: {value}")

    # ------------------------------------------------------------
    # 6. Create frame plane geometry for 3D visualization
    # ------------------------------------------------------------
    plane_h, plane_w = local_frames.shape[-2:]

    pixel_to_image = make_pixel_to_image_matrix(
        width=plane_w,
        height=plane_h,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
    )

    spacing = np.asarray([spacing_x, spacing_y, 1.0], dtype=np.float32)
    dimensions = np.asarray([plane_w, plane_h, 1], dtype=np.float32)

    frame_corners_world = make_frame_corners_world(
        pred_tracking=pred_tracking,
        pixel_to_image=pixel_to_image,
        width=plane_w,
        height=plane_h,
    )

    frame_mesh_vertices, frame_mesh_faces = make_frame_mesh_from_corners(
        frame_corners_world,
        connect_slices=connect_slices,
    )

    frame_corners_world_corr = make_frame_corners_world(
        pred_tracking=pred_tracking_corr,
        pixel_to_image=pixel_to_image,
        width=plane_w,
        height=plane_h,
    )

    frame_mesh_vertices_corr, frame_mesh_faces_corr = make_frame_mesh_from_corners(
        frame_corners_world_corr,
        connect_slices=connect_slices,
    )

    print("\nFrame mesh:")
    print(f"  frame_corners_world     : {frame_corners_world.shape}")
    print(f"  frame_mesh_vertices     : {frame_mesh_vertices.shape}")
    print(f"  frame_mesh_faces        : {frame_mesh_faces.shape}")
    print(f"  frame_corners_world_corr: {frame_corners_world_corr.shape}")
    print(f"  frame_mesh_vertices_corr: {frame_mesh_vertices_corr.shape}")
    print(f"  frame_mesh_faces_corr   : {frame_mesh_faces_corr.shape}")

    # ------------------------------------------------------------
    # 7. Point cloud extraction placeholder
    # ------------------------------------------------------------
    _ = extract_threshold_point_cloud_placeholder(
        raw_images=raw_images,
        pred_tracking=pred_tracking,
        pixel_to_image=pixel_to_image,
        threshold=None,
    )

    # ------------------------------------------------------------
    # 8. Save h5
    # ------------------------------------------------------------
    global_input_np = global_x[0].detach().float().cpu().numpy().astype(np.float32)
    local_input_np = local_x[0].detach().float().cpu().numpy().astype(np.float32)

    save_pseudo3d_h5(
        output_h5,
        video_path=video_path,
        input_mode=input_mode,
        source_name=source_name,
        raw_images=raw_images,
        global_input=global_input_np,
        local_input=local_input_np,
        frame_indices=frame_indices,
        fps=fps,
        stride=stride,
        pred_rel_pose_vecs=pred_rel_pose_vecs,
        pred_tracking=pred_tracking,
        pred_tracking_corr=pred_tracking_corr,
        pixel_to_image=pixel_to_image,
        spacing=spacing,
        dimensions=dimensions,
        frame_corners_world=frame_corners_world,
        frame_mesh_vertices=frame_mesh_vertices,
        frame_mesh_faces=frame_mesh_faces,
        frame_corners_world_corr=frame_corners_world_corr,
        frame_mesh_vertices_corr=frame_mesh_vertices_corr,
        frame_mesh_faces_corr=frame_mesh_faces_corr,
        connect_slices=connect_slices,
        local_preprocess_meta=local_preprocess_meta,
        tracking_correction_meta=tracking_correction_meta,
    )

    print("\nDone.")

    return {
        "output_h5": str(output_h5),
        "video_path": str(video_path),
        "input_mode": input_mode,
        "source_name": source_name,
        "num_frames": int(n),
        "raw_shape": tuple(raw_images.shape),
        "pred_rel_pose_vecs_shape": tuple(pred_rel_pose_vecs.shape),
        "pred_tracking_shape": tuple(pred_tracking.shape),
        "pred_tracking_corr_shape": tuple(pred_tracking_corr.shape),
        "frame_corners_world_shape": tuple(frame_corners_world.shape),
        "frame_mesh_vertices_shape": tuple(frame_mesh_vertices.shape),
        "frame_mesh_faces_shape": tuple(frame_mesh_faces.shape),
        "frame_corners_world_corr_shape": tuple(frame_corners_world_corr.shape),
        "frame_mesh_vertices_corr_shape": tuple(frame_mesh_vertices_corr.shape),
        "frame_mesh_faces_corr_shape": tuple(frame_mesh_faces_corr.shape),
        "local_preprocess_meta": local_preprocess_meta,
        "tracking_correction_meta": tracking_correction_meta,
    }
