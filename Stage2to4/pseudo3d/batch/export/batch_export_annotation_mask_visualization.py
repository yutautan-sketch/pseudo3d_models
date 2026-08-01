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

import h5py
import numpy as np

from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    load_point_cloud_h5,
    load_voc_bboxes,
    make_contour_texture_config_from_args,
    ranked_contour_config_from_metadata,
    xml_bbox_to_local,
)
from pseudo3d.export.export_annotation_mask_visualization import (
    _attr_to_str,
    _parse_float_list,
    _parse_offsets,
    export_textured_frame_obj,
    load_pseudo3d_for_annotation_visualization,
    make_overlay_textures,
    parse_color,
)
from src.utils.alpha_texture_processing import image_to_uint8_gray


LABEL_FEMUR_CANDIDATE = 1


@dataclass(frozen=True)
class BatchItem:
    annotated_h5: Path
    pseudo3d_h5: Path | None = None
    point_cloud_h5: Path | None = None
    voc_xml_root: Path | None = None
    video_name: str = ""
    output_obj: Path | None = None


@dataclass(frozen=True)
class AnnotatedManifest:
    attrs: dict[str, Any]
    point_label: np.ndarray
    valid_mask: np.ndarray
    frame_order: np.ndarray
    valid_contour_frame_orders: np.ndarray


def _attr_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    text = str(value)
    return text if text and text != "None" else default


def _optional_path(value: Any) -> Path | None:
    text = _attr_text(value)
    return Path(text) if text else None


def load_annotated_manifest(path: Path) -> AnnotatedManifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotated h5 not found: {path}")

    with h5py.File(path, "r") as handle:
        for group_name in ("point_cloud", "annotation"):
            if group_name not in handle:
                raise KeyError(f"'{group_name}' group not found in {path}")
        point_cloud = handle["point_cloud"]
        annotation = handle["annotation"]
        for key in ("frame_order",):
            if key not in point_cloud:
                raise KeyError(f"Required point_cloud/{key} not found in {path}")
        for key in ("point_label", "valid_mask"):
            if key not in annotation:
                raise KeyError(f"Required annotation/{key} not found in {path}")

        point_label = annotation["point_label"][:].astype(np.int8)
        valid_mask = annotation["valid_mask"][:].astype(bool)
        frame_order = point_cloud["frame_order"][:].astype(np.int32)
        attrs = dict(handle.attrs)

        valid_contour_frame_orders = np.zeros((0,), dtype=np.int32)
        if "frame_annotation" in handle:
            frame_group = handle["frame_annotation"]
            if "frame_order" in frame_group and "valid_contour" in frame_group:
                annotation_orders = frame_group["frame_order"][:].astype(np.int32)
                valid_contours = frame_group["valid_contour"][:].astype(bool)
                if annotation_orders.shape != valid_contours.shape:
                    raise ValueError(
                        "frame_annotation frame_order/valid_contour shape mismatch"
                    )
                valid_contour_frame_orders = np.unique(
                    annotation_orders[valid_contours]
                )

    num_points = int(point_label.shape[0])
    if valid_mask.shape != (num_points,):
        raise ValueError(
            "annotation/valid_mask length mismatch: "
            f"shape={valid_mask.shape}, point_label={point_label.shape}"
        )
    if frame_order.shape != (num_points,):
        raise ValueError(
            "point_cloud/frame_order length mismatch: "
            f"shape={frame_order.shape}, point_label={point_label.shape}"
        )
    return AnnotatedManifest(
        attrs=attrs,
        point_label=point_label,
        valid_mask=valid_mask,
        frame_order=frame_order,
        valid_contour_frame_orders=valid_contour_frame_orders,
    )


def _read_h5_list(h5_list: Path) -> list[BatchItem]:
    """
    Read annotated H5 paths from text or CSV.

    CSV supports annotated_h5/input_h5/h5_path plus optional pseudo3d_h5,
    point_cloud_h5, voc_xml_root, video_name, and output_obj columns.
    """
    h5_list = Path(h5_list)
    if not h5_list.exists():
        raise FileNotFoundError(f"h5_list not found: {h5_list}")

    if h5_list.suffix.lower() == ".csv":
        with h5_list.open(newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            if has_header:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    return []
                for candidate in ("annotated_h5", "input_h5", "h5_path"):
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
                            annotated_h5=Path(value),
                            pseudo3d_h5=(
                                Path(row["pseudo3d_h5"])
                                if row.get("pseudo3d_h5", "").strip()
                                else None
                            ),
                            point_cloud_h5=(
                                Path(row["point_cloud_h5"])
                                if row.get("point_cloud_h5", "").strip()
                                else None
                            ),
                            voc_xml_root=(
                                Path(row["voc_xml_root"])
                                if row.get("voc_xml_root", "").strip()
                                else None
                            ),
                            video_name=row.get("video_name", "").strip(),
                            output_obj=(
                                Path(row["output_obj"])
                                if row.get("output_obj", "").strip()
                                else None
                            ),
                        )
                    )
                return items

            reader = csv.reader(handle)
            return [
                BatchItem(annotated_h5=Path(row[0]))
                for row in reader
                if row and row[0].strip()
            ]

    items = []
    with h5_list.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(BatchItem(annotated_h5=Path(line)))
    return items


def collect_batch_items(
    *,
    annotated_dir: Path | None,
    h5_list: Path | None,
    pattern: str,
    recursive: bool,
) -> list[BatchItem]:
    if h5_list is not None:
        items = _read_h5_list(h5_list)
    elif annotated_dir is not None:
        annotated_dir = Path(annotated_dir)
        if not annotated_dir.exists():
            raise FileNotFoundError(f"annotated_dir not found: {annotated_dir}")
        if not annotated_dir.is_dir():
            raise NotADirectoryError(
                f"annotated_dir is not a directory: {annotated_dir}"
            )
        globber = annotated_dir.rglob if recursive else annotated_dir.glob
        items = [
            BatchItem(annotated_h5=path)
            for path in sorted(globber(pattern))
            if path.is_file()
        ]
    else:
        raise ValueError("Either --annotated_dir or --h5_list must be specified.")

    if not items:
        raise FileNotFoundError("No annotated point cloud h5 files were found.")
    return items


def _format_template(template: str, context: dict[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        raise KeyError(
            f"Unknown template field {exc} in '{template}'. "
            f"Available fields: {sorted(context)}"
        ) from exc


def default_visualization_stem(stem: str) -> str:
    marker = "_pointcloud_annotated_"
    if marker in stem:
        prefix, tag = stem.rsplit(marker, 1)
        if prefix and tag:
            return f"{prefix}_annotation_mask_debug_{tag}"
    if stem.endswith("_pointcloud_annotated"):
        prefix = stem[: -len("_pointcloud_annotated")]
        return f"{prefix}_annotation_mask_debug"
    return f"{stem}_annotation_mask_debug"


def resolve_output(
    item: BatchItem,
    *,
    output_root: Path | None,
    output_subdir_template: str,
    output_filename_template: str,
    texture_dir_name_template: str,
) -> tuple[Path, str]:
    annotated_h5 = Path(item.annotated_h5)
    context = {
        "input_stem": annotated_h5.stem,
        "input_name": annotated_h5.name,
        "parent_name": annotated_h5.parent.name,
        "visualization_stem": default_visualization_stem(annotated_h5.stem),
    }
    if item.output_obj is not None:
        output_obj = Path(item.output_obj)
    else:
        output_filename = _format_template(output_filename_template, context)
        if output_root is None:
            output_obj = annotated_h5.parent / output_filename
        else:
            output_subdir = _format_template(output_subdir_template, context)
            output_obj = Path(output_root) / output_subdir / output_filename

    texture_dir_name = _format_template(texture_dir_name_template, context)
    texture_path = Path(texture_dir_name)
    if texture_path.is_absolute() or ".." in texture_path.parts:
        raise ValueError(
            "texture_dir_name_template must produce a safe relative path, "
            f"got: {texture_dir_name}"
        )
    return output_obj, texture_dir_name


def resolve_sources(
    item: BatchItem,
    manifest: AnnotatedManifest,
    *,
    voc_xml_root_override: Path | None,
    geometry_key_override: str | None,
) -> tuple[Path, Path | None, Path, str, str]:
    attrs = manifest.attrs
    pseudo3d_h5 = item.pseudo3d_h5 or _optional_path(
        attrs.get("source_pseudo3d_h5")
    )
    point_cloud_h5 = item.point_cloud_h5 or _optional_path(
        attrs.get("source_point_cloud_h5")
    )
    voc_xml_root = (
        item.voc_xml_root
        or voc_xml_root_override
        or _optional_path(attrs.get("voc_xml_root"))
    )
    video_name = item.video_name or _attr_text(attrs.get("video_name"))
    geometry_key = geometry_key_override or _attr_text(
        attrs.get("geometry_key"), "corr"
    )

    if pseudo3d_h5 is None:
        raise KeyError(
            "source_pseudo3d_h5 metadata is missing; provide it in --h5_list CSV"
        )
    if voc_xml_root is None:
        raise KeyError(
            "voc_xml_root metadata is missing; use --voc_xml_root or --h5_list CSV"
        )
    if not video_name:
        raise KeyError(
            "video_name metadata is missing; provide it in --h5_list CSV"
        )
    if geometry_key not in {"prior", "corr"}:
        raise ValueError(f"Unknown geometry_key: {geometry_key}")
    if not pseudo3d_h5.is_file():
        raise FileNotFoundError(f"Pseudo3D h5 not found: {pseudo3d_h5}")
    if point_cloud_h5 is not None and not point_cloud_h5.is_file():
        raise FileNotFoundError(f"Point cloud h5 not found: {point_cloud_h5}")
    if not voc_xml_root.exists():
        raise FileNotFoundError(f"VOC XML root not found: {voc_xml_root}")
    return pseudo3d_h5, point_cloud_h5, voc_xml_root, video_name, geometry_key


def _annotation_frame_stats(
    manifest: AnnotatedManifest,
    *,
    num_frames: int,
) -> dict[str, int]:
    frame_order = manifest.frame_order
    if np.any(frame_order < 0) or np.any(frame_order >= num_frames):
        raise ValueError(
            "Annotated point frame_order is outside pseudo3D frame range"
        )
    positive = (
        manifest.valid_mask
        & (manifest.point_label == LABEL_FEMUR_CANDIDATE)
    )
    positive_counts = np.bincount(
        frame_order[positive],
        minlength=num_frames,
    )
    zero_positive = positive_counts == 0
    valid_contour_orders = manifest.valid_contour_frame_orders
    if np.any(valid_contour_orders < 0) or np.any(
        valid_contour_orders >= num_frames
    ):
        raise ValueError(
            "frame_annotation frame_order is outside pseudo3D frame range"
        )
    return {
        "num_points": int(manifest.point_label.shape[0]),
        "num_femur_candidates": int(positive.sum()),
        "num_positive_frames": int(np.count_nonzero(positive_counts > 0)),
        "num_zero_positive_frames": int(np.count_nonzero(zero_positive)),
        "num_valid_contour_frames": int(valid_contour_orders.size),
        "num_zero_positive_valid_contour_frames": int(
            np.count_nonzero(positive_counts[valid_contour_orders] == 0)
        ),
    }


def export_one(
    *,
    item: BatchItem,
    output_obj: Path,
    texture_dir_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    annotated_h5 = Path(item.annotated_h5)
    manifest = load_annotated_manifest(annotated_h5)
    (
        pseudo3d_h5,
        point_cloud_h5,
        voc_xml_root,
        video_name,
        geometry_key,
    ) = resolve_sources(
        item,
        manifest,
        voc_xml_root_override=args.voc_xml_root,
        geometry_key_override=args.geometry_key,
    )

    pseudo3d = load_pseudo3d_for_annotation_visualization(
        pseudo3d_h5,
        geometry_key=geometry_key,
    )
    num_frames = int(pseudo3d["local_encoder_images"].shape[0])
    annotation_stats = _annotation_frame_stats(manifest, num_frames=num_frames)

    point_cloud = None
    point_mode = "foreground"
    sampling_mode = "legacy"
    if point_cloud_h5 is not None:
        point_cloud, point_cloud_attrs = load_point_cloud_h5(point_cloud_h5)
        point_mode = _attr_to_str(
            point_cloud_attrs.get("point_mode"), "foreground"
        )
        sampling_mode = _attr_to_str(
            point_cloud_attrs.get("sampling_mode"), "legacy"
        )

    enable_contour_fallback = bool(args.enable_contour_fallback)
    if args.fallback_only_non_foreground_point_cloud and point_mode == "foreground":
        enable_contour_fallback = False
    if enable_contour_fallback and point_cloud is None:
        raise ValueError(
            "A source point cloud is required when contour fallback is enabled"
        )
    if args.draw_point_cloud_samples and point_cloud is None:
        raise ValueError(
            "A source point cloud is required to draw point-cloud samples"
        )

    local_images = pseudo3d["local_encoder_images"]
    local_shape_hw = tuple(image_to_uint8_gray(local_images[0]).shape)
    voc_bboxes = load_voc_bboxes(
        voc_xml_root,
        video_name=video_name,
        frame_indices=pseudo3d["frame_indices"],
        xml_frame_number_offsets=_parse_offsets(args.xml_frame_number_offsets),
        xml_frame_id_source=args.xml_frame_id_source,
        xml_annotation_dir_name=args.xml_annotation_dir_name,
        strict_xml_annotation_dir=args.strict_xml_annotation_dir,
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
    ranked_config = ranked_contour_config_from_metadata(manifest.attrs)
    textures, visualization_stats = make_overlay_textures(
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
        draw_point_cloud_samples=args.draw_point_cloud_samples,
        sample_point_radius=args.sample_point_radius,
        sample_point_alpha=args.sample_point_alpha,
        global_sample_color=parse_color(args.global_sample_color),
        local_percentile_sample_color=parse_color(
            args.local_percentile_sample_color
        ),
        tophat_sample_color=parse_color(args.tophat_sample_color),
        context_grid_sample_color=parse_color(args.context_grid_sample_color),
        ranked_contour_config=ranked_config,
        ranked_global_contour_color=parse_color(
            args.ranked_global_contour_color
        ),
        ranked_local_contour_color=parse_color(args.ranked_local_contour_color),
        ranked_shared_contour_color=parse_color(
            args.ranked_shared_contour_color
        ),
    )
    export_textured_frame_obj(
        output_obj,
        frame_corners_world=pseudo3d["frame_corners_world"],
        texture_images=textures,
        texture_dir_name=texture_dir_name,
    )

    return {
        "annotated_h5": str(annotated_h5),
        "pseudo3d_h5": str(pseudo3d_h5),
        "point_cloud_h5": str(point_cloud_h5) if point_cloud_h5 else "",
        "voc_xml_root": str(voc_xml_root),
        "video_name": video_name,
        "output_obj": str(output_obj),
        "output_mtl": str(output_obj.with_suffix(".mtl")),
        "texture_dir": str(output_obj.parent / texture_dir_name),
        "geometry_key": geometry_key,
        "xml_annotation_dir_name": args.xml_annotation_dir_name,
        "strict_xml_annotation_dir": bool(args.strict_xml_annotation_dir),
        "point_mode": point_mode,
        "sampling_mode": sampling_mode,
        "label_mode": _attr_text(
            manifest.attrs.get("label_mode"), "contour_in_bbox"
        ),
        "num_frames": num_frames,
        **annotation_stats,
        **visualization_stats,
        "draw_point_cloud_samples": bool(args.draw_point_cloud_samples),
        "enable_contour_fallback": enable_contour_fallback,
    }


SUMMARY_FIELDS = (
    "status",
    "annotated_h5",
    "pseudo3d_h5",
    "point_cloud_h5",
    "voc_xml_root",
    "video_name",
    "output_obj",
    "output_mtl",
    "texture_dir",
    "geometry_key",
    "xml_annotation_dir_name",
    "strict_xml_annotation_dir",
    "point_mode",
    "sampling_mode",
    "label_mode",
    "num_frames",
    "num_points",
    "num_femur_candidates",
    "num_positive_frames",
    "num_zero_positive_frames",
    "num_valid_contour_frames",
    "num_zero_positive_valid_contour_frames",
    "num_xml_frames",
    "num_voc_bboxes",
    "num_valid_contours",
    "num_invalid_contours",
    "num_selected_global_contours",
    "num_selected_local_contours",
    "num_selected_shared_contours",
    "num_fallback_used",
    "num_drawn_point_samples",
    "draw_point_cloud_samples",
    "enable_contour_fallback",
    "error",
)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-export annotation-mask visualization OBJ/MTL/textures from "
            "the same annotated H5 inputs and output locations used by "
            "batch_export_annotated_point_cloud_ply.py."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--annotated_dir", type=Path, default=None)
    input_group.add_argument("--h5_list", type=Path, default=None)
    parser.add_argument("--pattern", default="*_pointcloud_annotated.h5")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--output_subdir_template", default="{parent_name}")
    parser.add_argument(
        "--output_filename_template",
        default="{visualization_stem}.obj",
    )
    parser.add_argument(
        "--texture_dir_name_template",
        default="{visualization_stem}_textures",
    )
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")

    parser.add_argument(
        "--voc_xml_root",
        type=Path,
        default=None,
        help="Override voc_xml_root metadata for every item.",
    )
    parser.add_argument(
        "--geometry_key",
        choices=["prior", "corr"],
        default=None,
        help="Override geometry_key metadata for every item.",
    )
    parser.add_argument("--xml_frame_number_offsets", default="0,1")
    parser.add_argument(
        "--xml_frame_id_source",
        default="frame_index",
        choices=["frame_index", "frame_order", "both"],
    )
    parser.add_argument(
        "--xml_annotation_dir_name",
        type=str,
        default="annotations_renamed",
    )
    parser.add_argument(
        "--strict_xml_annotation_dir",
        action="store_true",
        help=(
            "Use only the named per-video XML directory and disable fallback "
            "to annotations and other sibling directories."
        ),
    )
    parser.add_argument(
        "--contour_preset",
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
        default="median",
        choices=["none", "median", "gaussian", "bilateral"],
    )
    parser.add_argument("--contour_denoise_ksize", type=int, default=3)
    parser.add_argument("--enable_contour_fallback", action="store_true")
    parser.add_argument(
        "--fallback_only_non_foreground_point_cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fallback_percentiles", default="85,80,75,70")
    parser.add_argument("--fallback_min_positive_points", type=int, default=5)

    parser.add_argument("--mask_color", default="255,32,32")
    parser.add_argument("--mask_alpha", type=float, default=0.55)
    parser.add_argument("--bbox_color", default="255,220,0")
    parser.add_argument("--contour_color", default="0,255,255")
    parser.add_argument("--fallback_contour_color", default="0,255,96")
    parser.add_argument("--bbox_thickness", type=int, default=2)
    parser.add_argument("--contour_thickness", type=int, default=2)
    parser.add_argument(
        "--draw_point_cloud_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sample_point_radius", type=int, default=1)
    parser.add_argument("--sample_point_alpha", type=float, default=0.75)
    parser.add_argument("--global_sample_color", default="255,255,255")
    parser.add_argument("--local_percentile_sample_color", default="64,200,255")
    parser.add_argument("--tophat_sample_color", default="255,200,64")
    parser.add_argument("--context_grid_sample_color", default="120,120,120")
    parser.add_argument("--ranked_global_contour_color", default="0,255,0")
    parser.add_argument("--ranked_local_contour_color", default="255,0,255")
    parser.add_argument("--ranked_shared_contour_color", default="255,255,0")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask_alpha must be in [0,1]")
    if args.bbox_thickness < 1:
        raise ValueError("--bbox_thickness must be >= 1")
    if args.contour_thickness < 1:
        raise ValueError("--contour_thickness must be >= 1")
    if args.sample_point_radius < 0:
        raise ValueError("--sample_point_radius must be >= 0")
    if not 0.0 <= args.sample_point_alpha <= 1.0:
        raise ValueError("--sample_point_alpha must be in [0,1]")
    if args.fallback_min_positive_points < 0:
        raise ValueError("--fallback_min_positive_points must be >= 0")
    for value in (
        args.mask_color,
        args.bbox_color,
        args.contour_color,
        args.fallback_contour_color,
        args.global_sample_color,
        args.local_percentile_sample_color,
        args.tophat_sample_color,
        args.context_grid_sample_color,
        args.ranked_global_contour_color,
        args.ranked_local_contour_color,
        args.ranked_shared_contour_color,
    ):
        parse_color(value)


def _output_is_complete(output_obj: Path, texture_dir_name: str) -> bool:
    return (
        output_obj.is_file()
        and output_obj.with_suffix(".mtl").is_file()
        and (output_obj.parent / texture_dir_name).is_dir()
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    items = collect_batch_items(
        annotated_dir=args.annotated_dir,
        h5_list=args.h5_list,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    resolved_outputs = [
        resolve_output(
            item,
            output_root=args.output_root,
            output_subdir_template=args.output_subdir_template,
            output_filename_template=args.output_filename_template,
            texture_dir_name_template=args.texture_dir_name_template,
        )
        for item in items
    ]
    output_objs = [output_obj for output_obj, _ in resolved_outputs]
    if len(set(output_objs)) != len(output_objs):
        raise ValueError("Multiple inputs resolve to the same output OBJ")

    print(f"Found annotated h5 files : {len(items)}")
    print(f"Output root              : {args.output_root or 'same as input parent'}")
    print(f"Draw point-cloud samples : {args.draw_point_cloud_samples}")
    print(f"Contour preset           : {args.contour_preset}")

    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []
    for index, (item, resolved) in enumerate(
        zip(items, resolved_outputs, strict=True),
        start=1,
    ):
        annotated_h5 = Path(item.annotated_h5)
        output_obj, texture_dir_name = resolved
        print("\n" + "=" * 72)
        print(f"[{index}/{len(items)}] annotated_h5: {annotated_h5}")
        print(f"output_obj : {output_obj}")
        print(f"texture_dir: {output_obj.parent / texture_dir_name}")

        if args.skip_existing and _output_is_complete(
            output_obj,
            texture_dir_name,
        ):
            print("Skipping existing outputs.")
            skipped += 1
            rows.append(
                {
                    "status": "skipped",
                    "annotated_h5": str(annotated_h5),
                    "output_obj": str(output_obj),
                    "output_mtl": str(output_obj.with_suffix(".mtl")),
                    "texture_dir": str(output_obj.parent / texture_dir_name),
                    "error": "",
                }
            )
            continue

        try:
            result = export_one(
                item=item,
                output_obj=output_obj,
                texture_dir_name=texture_dir_name,
                args=args,
            )
            result["status"] = "ok"
            result["error"] = ""
            rows.append(result)
            processed += 1
            print(
                "Exported: "
                f"frames={result['num_frames']}, "
                f"positive_frames={result['num_positive_frames']}, "
                f"zero_positive_frames={result['num_zero_positive_frames']}, "
                f"drawn_samples={result['num_drawn_point_samples']}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((annotated_h5, message))
            rows.append(
                {
                    "status": "failed",
                    "annotated_h5": str(annotated_h5),
                    "output_obj": str(output_obj),
                    "output_mtl": str(output_obj.with_suffix(".mtl")),
                    "texture_dir": str(output_obj.parent / texture_dir_name),
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

    print("\nBatch annotation-mask visualization summary:")
    print(f"  processed: {processed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {len(failures)}")
    if failures:
        for path, message in failures:
            print(f"  FAILED {path}: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
