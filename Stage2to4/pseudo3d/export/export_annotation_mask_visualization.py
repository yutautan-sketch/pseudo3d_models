from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import imageio.v2 as imageio
import numpy as np

from src.utils.alpha_texture_processing import image_to_uint8_gray
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    build_frame_contour_mask_results,
    load_point_cloud_h5,
    load_voc_bboxes,
    make_contour_texture_config_from_args,
    xml_bbox_to_local,
)


def load_pseudo3d_for_annotation_visualization(
    input_h5: Path,
    *,
    geometry_key: str,
) -> dict[str, Any]:
    import h5py

    input_h5 = Path(input_h5)
    if not input_h5.exists():
        raise FileNotFoundError(f"Pseudo3D h5 not found: {input_h5}")

    if geometry_key == "prior":
        frame_corners_key = "frame_corners_world"
    elif geometry_key == "corr":
        frame_corners_key = "frame_corners_world_corr"
    else:
        raise ValueError(f"Unknown geometry_key: {geometry_key}")

    with h5py.File(input_h5, "r") as f:
        required = ["local_encoder_images", "frame_indices", frame_corners_key]
        for key in required:
            if key not in f:
                raise KeyError(
                    f"Required key '{key}' for geometry_key='{geometry_key}' "
                    f"not found in {input_h5}"
                )

        return {
            "local_encoder_images": f["local_encoder_images"][:],
            "frame_indices": f["frame_indices"][:].astype(np.int64),
            "frame_corners_world": f[frame_corners_key][:].astype(np.float32),
            "attrs": dict(f.attrs),
            "geometry_key": geometry_key,
        }


def _parse_offsets(text: str) -> tuple[int, ...]:
    offsets = []
    for token in text.replace(",", " ").split():
        offsets.append(int(token))
    if not offsets:
        offsets = [0, 1]
    return tuple(dict.fromkeys(offsets))


def _parse_float_list(text: str) -> list[float]:
    values = []
    for token in text.replace(",", " ").split():
        if token:
            values.append(float(token))
    return values


def _attr_to_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    text = str(value)
    return text if text else default


def _blend_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_bbox(
    rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left = int(np.clip(round(float(x1)), 0, w - 1))
    top = int(np.clip(round(float(y1)), 0, h - 1))
    right = int(np.clip(round(float(x2)), 0, w - 1))
    bottom = int(np.clip(round(float(y2)), 0, h - 1))
    if right <= left or bottom <= top:
        return
    cv2.rectangle(rgb, (left, top), (right, bottom), color, int(thickness))


def _draw_mask_contours(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(rgb, contours, -1, color, int(thickness))


def make_overlay_textures(
    *,
    pseudo3d: dict[str, Any],
    voc_bboxes: dict[int, Any],
    local_bboxes: dict[int, Any],
    texture_config: Any,
    min_alpha: int,
    min_contour_area: float,
    mask_color: tuple[int, int, int],
    mask_alpha: float,
    bbox_color: tuple[int, int, int],
    contour_color: tuple[int, int, int],
    fallback_contour_color: tuple[int, int, int],
    bbox_thickness: int,
    contour_thickness: int,
    point_cloud: dict[str, np.ndarray] | None = None,
    enable_contour_fallback: bool = False,
    fallback_percentiles: list[float] | None = None,
    fallback_min_positive_points: int = 5,
) -> tuple[np.ndarray, dict[str, int]]:
    local_images = pseudo3d["local_encoder_images"]
    n = int(local_images.shape[0])
    textures = []
    results_by_order, union_masks_by_order, stats = build_frame_contour_mask_results(
        local_images=local_images,
        voc_bboxes=voc_bboxes,
        local_bboxes=local_bboxes,
        texture_config=texture_config,
        min_alpha=min_alpha,
        min_contour_area=min_contour_area,
        point_cloud=point_cloud,
        enable_fallback=enable_contour_fallback,
        fallback_percentiles=fallback_percentiles or [],
        fallback_min_positive_points=fallback_min_positive_points,
    )
    stats["num_fallback_used"] = int(
        sum(
            bool(item["result"].get("fallback_used", False))
            for items in results_by_order.values()
            for item in items
        )
    )

    for frame_order in range(n):
        gray = image_to_uint8_gray(local_images[frame_order])
        rgb = np.stack([gray, gray, gray], axis=-1)

        union_mask = union_masks_by_order.get(frame_order)
        if union_mask is not None and np.any(union_mask):
            rgb = _blend_mask(
                rgb,
                union_mask,
                color=mask_color,
                alpha=mask_alpha,
            )

        for item in results_by_order.get(frame_order, []):
            bbox = item["bbox"]
            result = item["result"]
            if result["valid"]:
                mask = result["mask"]
                draw_color = (
                    fallback_contour_color
                    if bool(result.get("fallback_used", False))
                    else contour_color
                )
                _draw_mask_contours(
                    rgb,
                    mask,
                    color=draw_color,
                    thickness=contour_thickness,
                )

            _draw_bbox(
                rgb,
                bbox.local_xyxy,
                color=bbox_color,
                thickness=bbox_thickness,
            )

        textures.append(rgb.astype(np.uint8))

    return np.stack(textures, axis=0), stats


def export_textured_frame_obj(
    output_obj: Path,
    *,
    frame_corners_world: np.ndarray,
    texture_images: np.ndarray,
    texture_dir_name: str,
    material_prefix: str = "ann_frame",
) -> None:
    output_obj = Path(output_obj)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    if output_obj.suffix.lower() != ".obj":
        raise ValueError(f"output_obj must end with .obj, got: {output_obj}")

    n = int(frame_corners_world.shape[0])
    if texture_images.shape[0] != n:
        raise ValueError(
            "Number of texture images must match frame corners: "
            f"texture_images={texture_images.shape[0]}, frame_corners={n}"
        )

    mtl_path = output_obj.with_suffix(".mtl")
    texture_dir = output_obj.parent / texture_dir_name
    texture_dir.mkdir(parents=True, exist_ok=True)

    texture_rel_paths = []
    for i, tex_img in enumerate(texture_images):
        tex_name = f"annotation_frame_{i:05d}.png"
        tex_path = texture_dir / tex_name
        imageio.imwrite(tex_path, tex_img.astype(np.uint8))
        texture_rel_paths.append(f"{texture_dir_name}/{tex_name}")

    with mtl_path.open("w") as f:
        f.write("# Annotation mask overlay materials\n\n")
        for i, rel_tex in enumerate(texture_rel_paths):
            mat_name = f"{material_prefix}_{i:05d}"
            f.write(f"newmtl {mat_name}\n")
            f.write("Ka 1.000 1.000 1.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write("d 1.0\n")
            f.write("illum 2\n")
            f.write(f"map_Kd {rel_tex}\n\n")

    with output_obj.open("w") as f:
        f.write("# Annotation mask overlay OBJ\n")
        f.write(f"mtllib {mtl_path.name}\n\n")

        for i in range(n):
            for p in frame_corners_world[i]:
                f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

        f.write("\n")
        for _ in range(n):
            f.write("vt 0.0 1.0\n")
            f.write("vt 1.0 1.0\n")
            f.write("vt 1.0 0.0\n")
            f.write("vt 0.0 0.0\n")

        f.write("\n")
        for i in range(n):
            mat_name = f"{material_prefix}_{i:05d}"
            v0 = i * 4 + 1
            v1 = i * 4 + 2
            v2 = i * 4 + 3
            v3 = i * 4 + 4
            vt0 = i * 4 + 1
            vt1 = i * 4 + 2
            vt2 = i * 4 + 3
            vt3 = i * 4 + 4
            f.write(f"usemtl {mat_name}\n")
            f.write(f"f {v0}/{vt0} {v1}/{vt1} {v2}/{vt2} {v3}/{vt3}\n\n")

    print(f"Saved annotation mask OBJ: {output_obj}")
    print(f"Saved annotation mask MTL: {mtl_path}")
    print(f"Saved annotation textures : {texture_dir}")


def parse_color(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Color must be R,G,B, got: {text}")
    color = tuple(int(part) for part in parts)
    if any(value < 0 or value > 255 for value in color):
        raise ValueError(f"Color values must be in [0,255], got: {text}")
    return color


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a textured OBJ where each pseudo-3D frame texture shows "
            "the local image with VOC-BBox-derived annotation mask overlay."
        )
    )
    parser.add_argument("--pseudo3d_h5", type=Path, required=True)
    parser.add_argument(
        "--point_cloud_h5",
        type=Path,
        default=None,
        help=(
            "Optional point cloud h5 used to evaluate fallback quality. "
            "Required when --enable_contour_fallback is used."
        ),
    )
    parser.add_argument("--voc_xml_root", type=Path, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--output_obj", type=Path, required=True)
    parser.add_argument("--geometry_key", type=str, default="corr", choices=["prior", "corr"])
    parser.add_argument("--texture_dir_name", type=str, default="annotation_textures")
    parser.add_argument("--xml_frame_number_offsets", type=str, default="0,1")
    parser.add_argument(
        "--xml_frame_id_source",
        type=str,
        default="frame_index",
        choices=["frame_index", "frame_order", "both"],
    )
    parser.add_argument(
        "--contour_preset",
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
        "--contour_threshold_mode",
        type=str,
        default=None,
        choices=["fixed", "otsu", "percentile", "adaptive"],
    )
    parser.add_argument("--contour_percentile", type=float, default=None)
    parser.add_argument("--contour_fixed_threshold", type=int, default=None)
    parser.add_argument("--contour_min_component_area", type=int, default=None)
    parser.add_argument("--contour_min_area", type=float, default=20.0)
    parser.add_argument("--contour_min_alpha", type=int, default=1)
    parser.add_argument("--contour_open_ksize", type=int, default=3)
    parser.add_argument("--contour_close_ksize", type=int, default=5)
    parser.add_argument(
        "--contour_denoise",
        type=str,
        default="median",
        choices=["none", "median", "gaussian", "bilateral"],
    )
    parser.add_argument("--contour_denoise_ksize", type=int, default=3)
    parser.add_argument(
        "--enable_contour_fallback",
        action="store_true",
        help=(
            "Enable the same relaxed contour fallback used by "
            "annotate_pseudo3d_point_cloud.py."
        ),
    )
    parser.add_argument(
        "--fallback_only_non_foreground_point_cloud",
        action="store_true",
        default=True,
        help=(
            "Apply fallback only when point_cloud attrs indicate "
            "point_mode != foreground."
        ),
    )
    parser.add_argument(
        "--fallback_percentiles",
        type=str,
        default="85,80,75,70",
        help="Comma-separated percentile thresholds tried during fallback.",
    )
    parser.add_argument(
        "--fallback_min_positive_points",
        type=int,
        default=5,
        help="Minimum positive points required before fallback is considered successful.",
    )
    parser.add_argument("--mask_color", type=str, default="255,32,32")
    parser.add_argument("--mask_alpha", type=float, default=0.55)
    parser.add_argument("--bbox_color", type=str, default="255,220,0")
    parser.add_argument("--contour_color", type=str, default="0,255,255")
    parser.add_argument(
        "--fallback_contour_color",
        type=str,
        default="0,255,96",
        help="Contour color used when a mask was selected by fallback.",
    )
    parser.add_argument("--bbox_thickness", type=int, default=2)
    parser.add_argument("--contour_thickness", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pseudo3d = load_pseudo3d_for_annotation_visualization(
        args.pseudo3d_h5,
        geometry_key=args.geometry_key,
    )
    point_cloud = None
    point_mode = "foreground"
    if args.point_cloud_h5 is not None:
        point_cloud, point_cloud_attrs = load_point_cloud_h5(args.point_cloud_h5)
        point_mode = _attr_to_str(point_cloud_attrs.get("point_mode"), "foreground")

    enable_contour_fallback = bool(args.enable_contour_fallback)
    if args.fallback_only_non_foreground_point_cloud and point_mode == "foreground":
        enable_contour_fallback = False
    if enable_contour_fallback and point_cloud is None:
        raise ValueError(
            "--point_cloud_h5 is required when contour fallback is enabled, "
            "because fallback success is judged by positive point count."
        )

    local_images = pseudo3d["local_encoder_images"]
    local_shape_hw = tuple(image_to_uint8_gray(local_images[0]).shape)

    voc_bboxes = load_voc_bboxes(
        args.voc_xml_root,
        video_name=args.video_name,
        frame_indices=pseudo3d["frame_indices"],
        xml_frame_number_offsets=_parse_offsets(args.xml_frame_number_offsets),
        xml_frame_id_source=args.xml_frame_id_source,
    )
    local_bboxes = {
        order: [
            xml_bbox_to_local(
                voc.xml_xyxy,
                image_size_wh=voc.image_size_wh,
                attrs=pseudo3d["attrs"],
                local_shape_hw=local_shape_hw,
            )
            for voc in boxes
        ]
        for order, boxes in voc_bboxes.items()
    }

    texture_config = make_contour_texture_config_from_args(args)
    textures, stats = make_overlay_textures(
        pseudo3d=pseudo3d,
        voc_bboxes=voc_bboxes,
        local_bboxes=local_bboxes,
        texture_config=texture_config,
        min_alpha=args.contour_min_alpha,
        min_contour_area=args.contour_min_area,
        mask_color=parse_color(args.mask_color),
        mask_alpha=args.mask_alpha,
        bbox_color=parse_color(args.bbox_color),
        contour_color=parse_color(args.contour_color),
        fallback_contour_color=parse_color(args.fallback_contour_color),
        bbox_thickness=args.bbox_thickness,
        contour_thickness=args.contour_thickness,
        point_cloud=point_cloud,
        enable_contour_fallback=enable_contour_fallback,
        fallback_percentiles=_parse_float_list(args.fallback_percentiles),
        fallback_min_positive_points=args.fallback_min_positive_points,
    )

    export_textured_frame_obj(
        args.output_obj,
        frame_corners_world=pseudo3d["frame_corners_world"],
        texture_images=textures,
        texture_dir_name=args.texture_dir_name,
    )

    print("Annotation mask visualization:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  geometry_key: {args.geometry_key}")
    print(f"  contour_preset: {args.contour_preset}")
    print(f"  point_cloud_h5: {args.point_cloud_h5}")
    print(f"  point_cloud_point_mode: {point_mode}")
    print(f"  enable_contour_fallback: {enable_contour_fallback}")
    print(f"  fallback_percentiles: {args.fallback_percentiles}")
    print(f"  fallback_min_positive_points: {args.fallback_min_positive_points}")


if __name__ == "__main__":
    main()
