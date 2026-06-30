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

import numpy as np

from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
    LABEL_BACKGROUND,
    build_contour_point_annotations,
    load_point_cloud_h5,
    load_pseudo3d_h5,
    load_voc_bboxes,
    make_contour_texture_config_from_args,
    make_empty_measurement,
    save_annotated_h5,
    xml_bbox_to_local,
)
from src.utils.alpha_texture_processing import image_to_uint8_gray


@dataclass(frozen=True)
class BatchItem:
    pseudo3d_h5: Path
    video_name: str
    point_cloud_h5: Path | None = None
    output_h5: Path | None = None


def _read_h5_list(h5_list: Path) -> list[BatchItem]:
    """
    Read pseudo3D h5 paths from a text or CSV file.

    Text format:
      one pseudo3D h5 path per line

    CSV format:
      pseudo3d_h5/input_h5/h5_path column, optional video_name,
      point_cloud_h5, and output_h5 columns.
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
                for candidate in ["pseudo3d_h5", "input_h5", "h5_path"]:
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
                            pseudo3d_h5=Path(value),
                            video_name=row.get("video_name", "").strip(),
                            point_cloud_h5=(
                                Path(row["point_cloud_h5"])
                                if row.get("point_cloud_h5", "").strip()
                                else None
                            ),
                            output_h5=(
                                Path(row["output_h5"])
                                if row.get("output_h5", "").strip()
                                else None
                            ),
                        )
                    )
                return items

            reader = csv.reader(f)
            return [
                BatchItem(pseudo3d_h5=Path(row[0]), video_name="")
                for row in reader
                if row and row[0].strip()
            ]

    items = []
    with h5_list.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(BatchItem(pseudo3d_h5=Path(line), video_name=""))
    return items


def collect_batch_items(
    *,
    pseudo3d_dir: Path | None,
    h5_list: Path | None,
    pattern: str,
    recursive: bool,
) -> list[BatchItem]:
    if h5_list is not None:
        items = _read_h5_list(h5_list)
    elif pseudo3d_dir is not None:
        pseudo3d_dir = Path(pseudo3d_dir)
        if not pseudo3d_dir.exists():
            raise FileNotFoundError(f"pseudo3d_dir not found: {pseudo3d_dir}")
        if not pseudo3d_dir.is_dir():
            raise NotADirectoryError(f"pseudo3d_dir is not a directory: {pseudo3d_dir}")
        globber = pseudo3d_dir.rglob if recursive else pseudo3d_dir.glob
        items = [
            BatchItem(pseudo3d_h5=path, video_name="")
            for path in sorted(globber(pattern))
            if path.is_file()
        ]
    else:
        raise ValueError("Either --pseudo3d_dir or --h5_list must be specified.")

    if not items:
        raise FileNotFoundError("No pseudo3D h5 files were found.")
    return items


def derive_video_name(pseudo3d_h5: Path, *, strip_suffix: str) -> str:
    stem = Path(pseudo3d_h5).stem
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


def resolve_paths(
    item: BatchItem,
    *,
    point_cloud_root: Path,
    output_root: Path | None,
    point_cloud_detail: str,
    geometry_key: str,
    video_suffix_to_strip: str,
    point_cloud_subdir_template: str,
    point_cloud_filename_template: str,
    output_subdir_template: str,
    output_filename_template: str,
) -> tuple[str, Path, Path]:
    pseudo3d_h5 = Path(item.pseudo3d_h5)
    pseudo3d_stem = pseudo3d_h5.stem
    video_name = item.video_name or derive_video_name(
        pseudo3d_h5,
        strip_suffix=video_suffix_to_strip,
    )
    context = {
        "pseudo3d_stem": pseudo3d_stem,
        "video_name": video_name,
        "detail": point_cloud_detail,
        "geometry_key": geometry_key,
    }

    if item.point_cloud_h5 is not None:
        point_cloud_h5 = Path(item.point_cloud_h5)
    else:
        point_cloud_subdir = _format_template(point_cloud_subdir_template, context)
        point_cloud_filename = _format_template(point_cloud_filename_template, context)
        point_cloud_h5 = Path(point_cloud_root) / point_cloud_subdir / point_cloud_filename

    if item.output_h5 is not None:
        output_h5 = Path(item.output_h5)
    elif output_root is not None:
        output_subdir = _format_template(output_subdir_template, context)
        output_filename = _format_template(output_filename_template, context)
        output_h5 = Path(output_root) / output_subdir / output_filename
    else:
        output_filename = _format_template(output_filename_template, context)
        output_h5 = point_cloud_h5.parent / output_filename

    return video_name, point_cloud_h5, output_h5


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


def annotate_one(
    *,
    pseudo3d_h5: Path,
    point_cloud_h5: Path,
    output_h5: Path,
    voc_xml_root: Path,
    video_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    point_cloud, point_cloud_attrs = load_point_cloud_h5(point_cloud_h5)
    pseudo3d = load_pseudo3d_h5(pseudo3d_h5, geometry_key=args.geometry_key)
    point_mode = _attr_to_str(point_cloud_attrs.get("point_mode"), "foreground")
    fallback_percentiles = _parse_float_list(args.fallback_percentiles)
    enable_contour_fallback = bool(args.enable_contour_fallback)
    if args.fallback_only_non_foreground_point_cloud and point_mode == "foreground":
        enable_contour_fallback = False

    local_images = pseudo3d["local_encoder_images"]
    local_shape_hw = tuple(image_to_uint8_gray(local_images[0]).shape)
    voc_bboxes = load_voc_bboxes(
        voc_xml_root,
        video_name=video_name,
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
    point_label, valid_mask, frame_annotation = build_contour_point_annotations(
        point_cloud=point_cloud,
        pseudo3d=pseudo3d,
        voc_bboxes=voc_bboxes,
        local_bboxes=local_bboxes,
        texture_config=texture_config,
        min_alpha=args.contour_min_alpha,
        min_contour_area=args.contour_min_area,
        no_bbox_label=args.no_bbox_label,
        enable_contour_fallback=enable_contour_fallback,
        fallback_percentiles=fallback_percentiles,
        fallback_min_positive_points=args.fallback_min_positive_points,
        bbox_inside_non_contour_label=args.bbox_inside_non_contour_label,
    )
    measurement = make_empty_measurement()

    num_points = int(point_cloud["points"].shape[0])
    num_xml_frames = int(len(voc_bboxes))
    num_bbox = int(sum(len(boxes) for boxes in voc_bboxes.values()))
    valid_contour_mask = frame_annotation["valid_contour"].astype(bool)
    num_valid_contour = int(valid_contour_mask.sum())
    num_valid_contour_frames = int(
        np.unique(frame_annotation["frame_order"][valid_contour_mask]).size
    )
    num_labeled_points = int(np.sum(point_label == LABEL_FEMUR_CANDIDATE))
    num_fallback_used = int(frame_annotation["fallback_used"].astype(bool).sum())

    meta = {
        "source_point_cloud_h5": str(point_cloud_h5),
        "source_pseudo3d_h5": str(pseudo3d_h5),
        "voc_xml_root": str(voc_xml_root),
        "video_name": video_name,
        "geometry_key": args.geometry_key,
        "tracking_key": pseudo3d["tracking_key"],
        "point_cloud_point_mode": point_mode,
        "label_mode": args.label_mode,
        "bbox_ignore_margin": float(args.bbox_ignore_margin),
        "bbox_inside_non_contour_label": args.bbox_inside_non_contour_label,
        "no_bbox_label": int(args.no_bbox_label),
        "xml_frame_id_source": args.xml_frame_id_source,
        "xml_frame_number_offsets": args.xml_frame_number_offsets,
        "contour_preset": args.contour_preset,
        "contour_threshold_mode": texture_config.threshold_mode,
        "contour_percentile": float(texture_config.percentile),
        "contour_fixed_threshold": int(texture_config.fixed_threshold),
        "contour_min_component_area": int(texture_config.min_component_area),
        "contour_min_area": float(args.contour_min_area),
        "contour_min_alpha": int(args.contour_min_alpha),
        "contour_open_ksize": int(texture_config.open_ksize),
        "contour_close_ksize": int(texture_config.close_ksize),
        "contour_denoise": texture_config.denoise,
        "contour_denoise_ksize": int(texture_config.denoise_ksize),
        "enable_contour_fallback": bool(enable_contour_fallback),
        "requested_enable_contour_fallback": bool(args.enable_contour_fallback),
        "fallback_only_non_foreground_point_cloud": bool(
            args.fallback_only_non_foreground_point_cloud
        ),
        "fallback_percentiles": args.fallback_percentiles,
        "fallback_min_positive_points": int(args.fallback_min_positive_points),
        "num_points": num_points,
        "num_voc_bbox_frames": num_xml_frames,
        "num_voc_bboxes": num_bbox,
        "num_valid_contours": num_valid_contour,
        "num_valid_contour_frames": num_valid_contour_frames,
        "num_fallback_used": num_fallback_used,
        "num_labeled_points": num_labeled_points,
        "batch_source": "batch_annotate_pseudo3d_point_cloud.py",
        "label_source": (
            "VOC BBox weak annotation; largest filled contour in bbox after "
            "binarization is projected to point labels"
        ),
    }

    save_annotated_h5(
        output_h5,
        point_cloud=point_cloud,
        point_cloud_attrs=point_cloud_attrs,
        point_label=point_label,
        valid_mask=valid_mask,
        frame_annotation=frame_annotation,
        measurement=measurement,
        meta=meta,
    )

    return {
        "video_name": video_name,
        "pseudo3d_h5": str(pseudo3d_h5),
        "point_cloud_h5": str(point_cloud_h5),
        "output_h5": str(output_h5),
        "num_points": num_points,
        "num_voc_bbox_frames": num_xml_frames,
        "num_voc_bboxes": num_bbox,
        "num_valid_contours": num_valid_contour,
        "num_valid_contour_frames": num_valid_contour_frames,
        "num_fallback_used": num_fallback_used,
        "num_labeled_points": num_labeled_points,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "video_name",
        "pseudo3d_h5",
        "point_cloud_h5",
        "output_h5",
        "num_points",
        "num_voc_bbox_frames",
        "num_voc_bboxes",
        "num_valid_contours",
        "num_valid_contour_frames",
        "num_fallback_used",
        "num_labeled_points",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-annotate pseudo-3D point cloud h5 files with weak VOC BBox "
            "contour labels."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pseudo3d_dir", type=Path, default=None)
    input_group.add_argument("--h5_list", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default="*.h5")
    parser.add_argument("--recursive", action="store_true")

    parser.add_argument("--point_cloud_root", type=Path, required=True)
    parser.add_argument("--voc_xml_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")

    parser.add_argument("--video_suffix_to_strip", type=str, default="")
    parser.add_argument("--point_cloud_detail", type=str, default="percentile90_area100")
    parser.add_argument(
        "--point_cloud_subdir_template",
        type=str,
        default="{pseudo3d_stem}/{detail}_{geometry_key}",
    )
    parser.add_argument(
        "--point_cloud_filename_template",
        type=str,
        default="{video_name}_pointcloud.h5",
    )
    parser.add_argument(
        "--output_subdir_template",
        type=str,
        default="{pseudo3d_stem}/{detail}_{geometry_key}",
    )
    parser.add_argument(
        "--output_filename_template",
        type=str,
        default="{video_name}_pointcloud_annotated.h5",
    )

    parser.add_argument("--geometry_key", type=str, default="corr", choices=["prior", "corr"])
    parser.add_argument(
        "--label_mode",
        type=str,
        default="contour_in_bbox",
        choices=["contour_in_bbox"],
    )
    parser.add_argument("--bbox_ignore_margin", type=float, default=0.0)
    parser.add_argument(
        "--no_bbox_label",
        type=int,
        default=LABEL_IGNORE,
        choices=[LABEL_IGNORE, LABEL_BACKGROUND],
    )
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
    parser.add_argument("--enable_contour_fallback", action="store_true")
    parser.add_argument(
        "--fallback_only_non_foreground_point_cloud",
        action="store_true",
        default=True,
    )
    parser.add_argument("--fallback_percentiles", type=str, default="85,80,75,70")
    parser.add_argument("--fallback_min_positive_points", type=int, default=5)
    parser.add_argument(
        "--bbox_inside_non_contour_label",
        type=str,
        default="ignore",
        choices=["ignore", "background"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    items = collect_batch_items(
        pseudo3d_dir=args.pseudo3d_dir,
        h5_list=args.h5_list,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    print(f"Found pseudo3D h5 files: {len(items)}")
    print(f"Point cloud root       : {args.point_cloud_root}")
    print(f"VOC XML root           : {args.voc_xml_root}")
    print(f"Geometry key           : {args.geometry_key}")
    print(f"Contour preset         : {args.contour_preset}")

    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []

    for index, item in enumerate(items, start=1):
        pseudo3d_h5 = Path(item.pseudo3d_h5)
        video_name, point_cloud_h5, output_h5 = resolve_paths(
            item,
            point_cloud_root=args.point_cloud_root,
            output_root=args.output_root,
            point_cloud_detail=args.point_cloud_detail,
            geometry_key=args.geometry_key,
            video_suffix_to_strip=args.video_suffix_to_strip,
            point_cloud_subdir_template=args.point_cloud_subdir_template,
            point_cloud_filename_template=args.point_cloud_filename_template,
            output_subdir_template=args.output_subdir_template,
            output_filename_template=args.output_filename_template,
        )

        print("\n" + "=" * 72)
        print(f"[{index}/{len(items)}] video_name     : {video_name}")
        print(f"pseudo3d_h5     : {pseudo3d_h5}")
        print(f"point_cloud_h5  : {point_cloud_h5}")
        print(f"output_h5       : {output_h5}")

        if args.skip_existing and output_h5.exists():
            print("Skipping existing output.")
            skipped += 1
            rows.append(
                {
                    "status": "skipped",
                    "video_name": video_name,
                    "pseudo3d_h5": str(pseudo3d_h5),
                    "point_cloud_h5": str(point_cloud_h5),
                    "output_h5": str(output_h5),
                }
            )
            continue

        try:
            result = annotate_one(
                pseudo3d_h5=pseudo3d_h5,
                point_cloud_h5=point_cloud_h5,
                output_h5=output_h5,
                voc_xml_root=args.voc_xml_root,
                video_name=video_name,
                args=args,
            )
            result["status"] = "ok"
            result["error"] = ""
            rows.append(result)
            processed += 1
            print(
                "Annotated: "
                f"bbox={result['num_voc_bboxes']}, "
                f"valid_contours={result['num_valid_contours']}, "
                f"labeled_points={result['num_labeled_points']}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((pseudo3d_h5, message))
            rows.append(
                {
                    "status": "failed",
                    "video_name": video_name,
                    "pseudo3d_h5": str(pseudo3d_h5),
                    "point_cloud_h5": str(point_cloud_h5),
                    "output_h5": str(output_h5),
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

    print("\nBatch point cloud annotation summary:")
    print(f"  processed: {processed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {len(failures)}")

    if failures:
        for path, message in failures:
            print(f"  FAILED {path}: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
