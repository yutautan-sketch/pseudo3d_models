from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.utils.alpha_texture_processing import (
    AlphaTextureConfig,
    image_to_uint8_gray,
    make_texture_image,
)


def make_texture_config_from_args(args: argparse.Namespace) -> AlphaTextureConfig:
    """
    Build AlphaTextureConfig from preset and command-line overrides.
    """
    if args.preset == "original":
        cfg = AlphaTextureConfig(
            texture_style="original",
            white_foreground=False,
        )
    elif args.preset == "otsu_area500":
        cfg = AlphaTextureConfig(
            texture_style="threshold_alpha",
            denoise="median",
            denoise_ksize=3,
            threshold_mode="otsu",
            open_ksize=3,
            close_ksize=5,
            morph_shape="ellipse",
            min_component_area=500,
            white_foreground=False,
        )
    elif args.preset in {"percentile90_area100", "percentile_area100"}:
        cfg = AlphaTextureConfig(
            texture_style="threshold_alpha",
            denoise="median",
            denoise_ksize=3,
            threshold_mode="percentile",
            percentile=90.0,
            open_ksize=3,
            close_ksize=5,
            morph_shape="ellipse",
            min_component_area=100,
            white_foreground=False,
        )
    elif args.preset == "percentile85_area100":
        cfg = AlphaTextureConfig(
            texture_style="threshold_alpha",
            denoise="median",
            denoise_ksize=3,
            threshold_mode="percentile",
            percentile=85.0,
            open_ksize=3,
            close_ksize=5,
            morph_shape="ellipse",
            min_component_area=100,
            white_foreground=False,
        )
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    if args.texture_style is not None:
        cfg.texture_style = args.texture_style
    if args.texture_denoise is not None:
        cfg.denoise = args.texture_denoise
    if args.texture_denoise_ksize is not None:
        cfg.denoise_ksize = args.texture_denoise_ksize
    if args.texture_threshold_mode is not None:
        cfg.threshold_mode = args.texture_threshold_mode
    if args.texture_threshold is not None:
        cfg.fixed_threshold = args.texture_threshold
    if args.texture_percentile is not None:
        cfg.percentile = args.texture_percentile
    if args.texture_adaptive_block_size is not None:
        cfg.adaptive_block_size = args.texture_adaptive_block_size
    if args.texture_adaptive_c is not None:
        cfg.adaptive_c = args.texture_adaptive_c
    if args.texture_open_ksize is not None:
        cfg.open_ksize = args.texture_open_ksize
    if args.texture_close_ksize is not None:
        cfg.close_ksize = args.texture_close_ksize
    if args.texture_morph_shape is not None:
        cfg.morph_shape = args.texture_morph_shape
    if args.texture_min_component_area is not None:
        cfg.min_component_area = args.texture_min_component_area
    if args.texture_white_foreground:
        cfg.white_foreground = True

    return cfg


def load_point_cloud_inputs(input_h5: Path, *, geometry_key: str) -> dict[str, Any]:
    """
    Load only the h5 datasets needed for frame-plane point-cloud projection.
    """
    import h5py

    input_h5 = Path(input_h5)
    if not input_h5.exists():
        raise FileNotFoundError(f"Input h5 not found: {input_h5}")

    if geometry_key == "prior":
        tracking_key = "pred_tracking"
    elif geometry_key == "corr":
        tracking_key = "pred_tracking_corr"
    else:
        raise ValueError(f"Unknown geometry_key: {geometry_key}")

    with h5py.File(input_h5, "r") as f:
        required_keys = [
            "local_encoder_images",
            "frame_indices",
            "pixel_to_image",
            tracking_key,
        ]
        for key in required_keys:
            if key not in f:
                raise KeyError(
                    f"Required key '{key}' for geometry_key='{geometry_key}' "
                    f"not found in {input_h5}"
                )

        data = {
            "local_encoder_images": f["local_encoder_images"][:].astype(np.float32),
            "frame_indices": f["frame_indices"][:].astype(np.int64),
            "pixel_to_image": f["pixel_to_image"][:].astype(np.float32),
            "pred_tracking": f[tracking_key][:].astype(np.float32),
            "attrs": dict(f.attrs),
            "geometry_key": geometry_key,
            "tracking_key": tracking_key,
        }

        if "tracking_correction" in f:
            data["tracking_correction_attrs"] = dict(f["tracking_correction"].attrs)

    return data


def _choose_foreground_pixels(
    image: np.ndarray,
    texture: np.ndarray,
    *,
    min_alpha: int,
    min_intensity: int,
    original_foreground_mode: str,
    sample_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select foreground pixel coordinates from one local image and its texture.
    """
    gray = image_to_uint8_gray(image)

    if texture.ndim == 3 and texture.shape[-1] == 4:
        alpha = texture[..., 3].astype(np.uint8)
        mask = alpha >= int(min_alpha)
    else:
        alpha = np.full(gray.shape, 255, dtype=np.uint8)
        if original_foreground_mode == "all":
            mask = np.ones(gray.shape, dtype=bool)
        elif original_foreground_mode == "nonzero":
            mask = gray > 0
        elif original_foreground_mode == "intensity":
            mask = gray >= int(min_intensity)
        else:
            raise ValueError(
                "original_foreground_mode must be one of: all, nonzero, intensity"
            )

    if sample_stride > 1:
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::sample_stride, ::sample_stride] = True
        mask &= stride_mask

    ys, xs = np.nonzero(mask)
    return xs.astype(np.float32), ys.astype(np.float32), alpha[ys, xs]


def _choose_grid_pixels(
    image: np.ndarray,
    texture: np.ndarray,
    *,
    sample_stride: int,
    dense: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select regular-grid local pixel coordinates without foreground removal.
    """
    gray = image_to_uint8_gray(image)
    h, w = gray.shape

    step = 1 if dense else max(1, int(sample_stride))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    xs = xs.reshape(-1).astype(np.float32)
    ys = ys.reshape(-1).astype(np.float32)

    if texture.ndim == 3 and texture.shape[-1] == 4:
        alpha = texture[..., 3].astype(np.uint8)
    else:
        alpha = np.full(gray.shape, 255, dtype=np.uint8)

    yi = ys.astype(np.int64)
    xi = xs.astype(np.int64)
    alpha_values = alpha[yi, xi]
    return xs, ys, alpha_values


def _subsample_indices(
    num_points: int,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    return np.sort(rng.choice(num_points, size=max_points, replace=False))


def project_frame_points_to_world(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    transform: np.ndarray,
    pixel_to_image: np.ndarray,
) -> np.ndarray:
    """
    Project pixel coordinates on one frame plane into world coordinates.
    """
    zeros = np.zeros_like(xs, dtype=np.float32)
    ones = np.ones_like(xs, dtype=np.float32)
    pixel_h = np.stack([xs, ys, zeros, ones], axis=1)
    image_h = (pixel_to_image @ pixel_h.T).T
    world_h = (transform @ image_h.T).T
    return world_h[:, :3].astype(np.float32)


def build_frame_point_cloud(
    *,
    local_images: np.ndarray,
    pred_tracking: np.ndarray,
    pixel_to_image: np.ndarray,
    frame_indices: np.ndarray,
    texture_config: AlphaTextureConfig,
    min_alpha: int,
    min_intensity: int,
    original_foreground_mode: str,
    point_mode: str,
    sample_stride: int,
    max_points_per_frame: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """
    Convert each corrected/prior frame plane into pseudo-3D points.
    """
    if sample_stride < 1:
        raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")

    n = int(local_images.shape[0])
    if pred_tracking.shape[0] != n:
        raise ValueError(
            "Number of tracking matrices must match local images: "
            f"pred_tracking={pred_tracking.shape[0]}, local_images={n}"
        )
    if frame_indices.shape[0] != n:
        raise ValueError(
            "Number of frame_indices must match local images: "
            f"frame_indices={frame_indices.shape[0]}, local_images={n}"
        )

    rng = np.random.default_rng(random_seed)

    all_points = []
    all_intensity = []
    all_alpha = []
    all_frame_index = []
    all_frame_order = []
    all_pixel_xy = []
    all_confidence = []

    per_frame_counts = np.zeros(n, dtype=np.int64)

    for frame_order in range(n):
        image = local_images[frame_order]
        gray = image_to_uint8_gray(image)
        texture = make_texture_image(image, texture_config)

        if point_mode == "foreground":
            xs, ys, alpha_values = _choose_foreground_pixels(
                image,
                texture,
                min_alpha=min_alpha,
                min_intensity=min_intensity,
                original_foreground_mode=original_foreground_mode,
                sample_stride=sample_stride,
            )
        elif point_mode == "grid":
            xs, ys, alpha_values = _choose_grid_pixels(
                image,
                texture,
                sample_stride=sample_stride,
                dense=False,
            )
        elif point_mode == "dense":
            xs, ys, alpha_values = _choose_grid_pixels(
                image,
                texture,
                sample_stride=1,
                dense=True,
            )
        else:
            raise ValueError(
                "point_mode must be one of: foreground, grid, dense; "
                f"got {point_mode}"
            )

        if xs.size == 0:
            continue

        keep = _subsample_indices(
            xs.size,
            max_points=max_points_per_frame,
            rng=rng,
        )
        xs = xs[keep]
        ys = ys[keep]
        alpha_values = alpha_values[keep]

        points = project_frame_points_to_world(
            xs=xs,
            ys=ys,
            transform=pred_tracking[frame_order],
            pixel_to_image=pixel_to_image,
        )

        yi = ys.astype(np.int64)
        xi = xs.astype(np.int64)
        intensity = gray[yi, xi].astype(np.uint8)
        confidence = alpha_values.astype(np.float32) / 255.0

        count = int(points.shape[0])
        per_frame_counts[frame_order] = count

        all_points.append(points)
        all_intensity.append(intensity)
        all_alpha.append(alpha_values.astype(np.uint8))
        all_frame_index.append(
            np.full(count, int(frame_indices[frame_order]), dtype=np.int64)
        )
        all_frame_order.append(np.full(count, frame_order, dtype=np.int32))
        all_pixel_xy.append(np.stack([xs, ys], axis=1).astype(np.float32))
        all_confidence.append(confidence)

    if not all_points:
        return {
            "points": np.zeros((0, 3), dtype=np.float32),
            "intensity": np.zeros((0,), dtype=np.uint8),
            "alpha": np.zeros((0,), dtype=np.uint8),
            "frame_index": np.zeros((0,), dtype=np.int64),
            "frame_order": np.zeros((0,), dtype=np.int32),
            "pixel_xy": np.zeros((0, 2), dtype=np.float32),
            "confidence": np.zeros((0,), dtype=np.float32),
            "source_type": np.zeros((0,), dtype=np.uint8),
            "per_frame_counts": per_frame_counts,
        }

    points = np.concatenate(all_points, axis=0)
    intensity = np.concatenate(all_intensity, axis=0)
    alpha = np.concatenate(all_alpha, axis=0)
    frame_index = np.concatenate(all_frame_index, axis=0)
    frame_order = np.concatenate(all_frame_order, axis=0)
    pixel_xy = np.concatenate(all_pixel_xy, axis=0)
    confidence = np.concatenate(all_confidence, axis=0)

    return {
        "points": points,
        "intensity": intensity,
        "alpha": alpha,
        "frame_index": frame_index,
        "frame_order": frame_order,
        "pixel_xy": pixel_xy,
        "confidence": confidence,
        "source_type": np.zeros(points.shape[0], dtype=np.uint8),
        "per_frame_counts": per_frame_counts,
    }


def _write_h5_attrs(group: Any, attrs: dict[str, Any]):
    for key, value in attrs.items():
        if isinstance(value, (np.bool_, bool)):
            group.attrs[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            group.attrs[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            group.attrs[key] = float(value)
        elif value is None:
            group.attrs[key] = "None"
        else:
            group.attrs[key] = str(value)


def save_point_cloud_h5(
    output_h5: Path,
    *,
    point_cloud: dict[str, np.ndarray],
    meta: dict[str, Any],
):
    """
    Save exported pseudo-3D point cloud into a standalone h5 file.
    """
    import h5py

    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        group = f.create_group("point_cloud")
        for key, value in point_cloud.items():
            group.create_dataset(key, data=value, compression="gzip")
        _write_h5_attrs(group, meta)

    print(f"Saved point cloud h5: {output_h5}")


def save_point_cloud_ply(
    output_ply: Path,
    *,
    point_cloud: dict[str, np.ndarray],
):
    """
    Save point cloud as ASCII PLY for Blender/MeshLab inspection.
    """
    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    points = point_cloud["points"]
    intensity = point_cloud["intensity"]
    alpha = point_cloud["alpha"]
    confidence = point_cloud["confidence"]
    frame_order = point_cloud["frame_order"]

    with output_ply.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("property float confidence\n")
        f.write("property int frame_order\n")
        f.write("end_header\n")

        for p, gray, a, conf, order in zip(
            points,
            intensity,
            alpha,
            confidence,
            frame_order,
            strict=True,
        ):
            g = int(gray)
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{g} {g} {g} {int(a)} {float(conf):.6f} {int(order)}\n"
            )

    print(f"Saved point cloud ply: {output_ply}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export frame-plane pseudo-3D point cloud from a pseudo-3D h5. "
            "This implements Step 1 only: each frame's alpha/foreground pixels "
            "are projected onto the selected prior/corr geometry."
        )
    )

    parser.add_argument("--input_h5", type=Path, required=True)
    parser.add_argument("--output_h5", type=Path, default=None)
    parser.add_argument("--output_ply", type=Path, default=None)

    parser.add_argument(
        "--geometry_key",
        type=str,
        default="corr",
        choices=["prior", "corr"],
        help="Use prior or corrected tracking geometry for projection.",
    )

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
        help="Alpha texture preset used to choose foreground pixels.",
    )

    parser.add_argument(
        "--point_mode",
        type=str,
        default="foreground",
        choices=["foreground", "grid", "dense"],
        help=(
            "Point selection mode. "
            "foreground: current behavior; select foreground pixels using "
            "alpha/intensity. "
            "grid: keep regular grid pixels without alpha/intensity removal. "
            "dense: same as grid with effective sample_stride=1."
        ),
    )
    parser.add_argument("--min_alpha", type=int, default=1)
    parser.add_argument("--min_intensity", type=int, default=1)
    parser.add_argument(
        "--original_foreground_mode",
        type=str,
        default="nonzero",
        choices=["all", "nonzero", "intensity"],
        help="Foreground rule when preset/texture_style produces no alpha channel.",
    )
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_points_per_frame", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)

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


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.output_h5 is None and args.output_ply is None:
        raise ValueError("Specify at least one of --output_h5 or --output_ply.")

    data = load_point_cloud_inputs(args.input_h5, geometry_key=args.geometry_key)
    texture_config = make_texture_config_from_args(args)

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
        sample_stride=args.sample_stride,
        max_points_per_frame=args.max_points_per_frame,
        random_seed=args.random_seed,
    )

    counts = point_cloud["per_frame_counts"]
    meta = {
        "source_h5": str(args.input_h5),
        "geometry_key": args.geometry_key,
        "tracking_key": data["tracking_key"],
        "preset": args.preset,
        "point_mode": args.point_mode,
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
        "points_per_frame_min": int(counts.min()) if counts.size else 0,
        "points_per_frame_max": int(counts.max()) if counts.size else 0,
        "points_per_frame_mean": float(counts.mean()) if counts.size else 0.0,
        "source_type_frame": 0,
    }

    print("Pseudo-3D frame point cloud:")
    for key in [
        "source_h5",
        "geometry_key",
        "preset",
        "point_mode",
        "num_frames",
        "num_points",
        "points_per_frame_min",
        "points_per_frame_max",
        "points_per_frame_mean",
    ]:
        print(f"  {key}: {meta[key]}")

    if args.output_h5 is not None:
        save_point_cloud_h5(args.output_h5, point_cloud=point_cloud, meta=meta)
    if args.output_ply is not None:
        save_point_cloud_ply(args.output_ply, point_cloud=point_cloud)


if __name__ == "__main__":
    main()
