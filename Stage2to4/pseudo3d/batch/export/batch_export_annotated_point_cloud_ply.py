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

from pseudo3d.export.export_annotated_point_cloud_ply import (
    LABEL_BACKGROUND,
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
    ensure_numpy,
    load_annotated_h5,
    make_point_colors,
    parse_color,
    parse_label_values,
    save_ply,
    select_annotation_points,
    select_points,
)


@dataclass(frozen=True)
class BatchItem:
    input_h5: Path
    output_ply: Path | None = None
    annotation_only_output_ply: Path | None = None


def _read_h5_list(h5_list: Path) -> list[BatchItem]:
    """
    Read annotated point cloud h5 paths from text or CSV.

    Text format:
      one annotated h5 path per line

    CSV format:
      annotated_h5/input_h5/h5_path column, optional output_ply and
      annotation_only_output_ply columns.
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
                for candidate in ["annotated_h5", "input_h5", "h5_path"]:
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
                            output_ply=(
                                Path(row["output_ply"])
                                if row.get("output_ply", "").strip()
                                else None
                            ),
                            annotation_only_output_ply=(
                                Path(row["annotation_only_output_ply"])
                                if row.get("annotation_only_output_ply", "").strip()
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
            raise NotADirectoryError(f"annotated_dir is not a directory: {annotated_dir}")

        globber = annotated_dir.rglob if recursive else annotated_dir.glob
        items = [
            BatchItem(input_h5=path)
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


def default_annotation_only_stem(stem: str) -> str:
    if stem.endswith("_pointcloud_annotated"):
        return stem[: -len("_pointcloud_annotated")] + "_pointcloud_annotation_only"
    marker = "_pointcloud_annotated_"
    if marker in stem:
        prefix, mode = stem.rsplit(marker, 1)
        if prefix and mode:
            return f"{prefix}_pointcloud_annotation_only_{mode}"
    return stem + "_annotation_only"


def resolve_output_paths(
    item: BatchItem,
    *,
    output_root: Path | None,
    output_subdir_template: str,
    output_filename_template: str,
    annotation_only_filename_template: str,
) -> tuple[Path, Path]:
    input_h5 = Path(item.input_h5)
    context = {
        "input_stem": input_h5.stem,
        "input_name": input_h5.name,
        "parent_name": input_h5.parent.name,
        "annotation_only_stem": default_annotation_only_stem(input_h5.stem),
    }

    if item.output_ply is not None:
        output_ply = Path(item.output_ply)
    else:
        output_filename = _format_template(output_filename_template, context)
        if output_root is None:
            output_ply = input_h5.parent / output_filename
        else:
            output_subdir = _format_template(output_subdir_template, context)
            output_ply = Path(output_root) / output_subdir / output_filename

    if item.annotation_only_output_ply is not None:
        annotation_only_output_ply = Path(item.annotation_only_output_ply)
    else:
        annotation_filename = _format_template(
            annotation_only_filename_template,
            context,
        )
        if output_root is None:
            annotation_only_output_ply = input_h5.parent / annotation_filename
        else:
            output_subdir = _format_template(output_subdir_template, context)
            annotation_only_output_ply = Path(output_root) / output_subdir / annotation_filename

    return output_ply, annotation_only_output_ply


def export_one(
    *,
    input_h5: Path,
    output_ply: Path,
    annotation_only_output_ply: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    data = load_annotated_h5(input_h5)
    femur_color = parse_color(args.femur_color)
    measurement_color = parse_color(args.measurement_color)
    frame_endpoint_color = parse_color(args.frame_endpoint_color)
    annotation_marker_color = parse_color(args.annotation_marker_color)
    label_values = parse_label_values(args.label_values)

    colors, out_alpha = make_point_colors(
        intensity=data["intensity"],
        alpha=data["alpha"],
        labels=data["point_label"],
        valid_mask=data["valid_mask"],
        femur_color=femur_color,
        background_mode=args.background_mode,
        background_alpha=args.background_alpha,
        ignore_alpha=args.ignore_alpha,
    )
    keep = select_points(
        data,
        background_mode=args.background_mode,
        label_values=label_values,
    )

    save_ply(
        output_ply,
        data=data,
        keep=keep,
        colors=colors,
        out_alpha=out_alpha,
        include_measurement_endpoint=args.include_measurement_endpoint,
        include_frame_endpoints=args.include_frame_endpoints,
        measurement_color=measurement_color,
        frame_endpoint_color=frame_endpoint_color,
        include_annotation_markers=args.include_annotation_markers,
        annotation_marker_color=annotation_marker_color,
        annotation_marker_size=args.annotation_marker_size,
        annotation_marker_stride=args.annotation_marker_stride,
        max_annotation_markers=args.max_annotation_markers,
    )

    annotation_keep = select_annotation_points(data)
    if annotation_only_output_ply is not None:
        annotation_colors, annotation_alpha = make_point_colors(
            intensity=data["intensity"],
            alpha=data["alpha"],
            labels=data["point_label"],
            valid_mask=data["valid_mask"],
            femur_color=femur_color,
            background_mode="hidden",
            background_alpha=args.background_alpha,
            ignore_alpha=args.ignore_alpha,
        )
        save_ply(
            annotation_only_output_ply,
            data=data,
            keep=annotation_keep,
            colors=annotation_colors,
            out_alpha=annotation_alpha,
            include_measurement_endpoint=False,
            include_frame_endpoints=False,
            measurement_color=measurement_color,
            frame_endpoint_color=frame_endpoint_color,
            include_annotation_markers=False,
            annotation_marker_color=annotation_marker_color,
            annotation_marker_size=args.annotation_marker_size,
            annotation_marker_stride=args.annotation_marker_stride,
            max_annotation_markers=args.max_annotation_markers,
        )

    labels = data["point_label"]
    frame_annotation = data.get("frame_annotation", {})
    valid_contours = frame_annotation.get("valid_contour")
    num_valid_contours = (
        int(np.sum(valid_contours.astype(bool)))
        if valid_contours is not None
        else ""
    )
    marker_source = int(
        np.sum(data["valid_mask"] & (labels == LABEL_FEMUR_CANDIDATE))
    )
    marker_count = 0
    if args.include_annotation_markers:
        stride = max(1, args.annotation_marker_stride)
        marker_count = max(0, (marker_source + stride - 1) // stride)
        if args.max_annotation_markers > 0:
            marker_count = min(marker_count, args.max_annotation_markers)

    return {
        "input_h5": str(input_h5),
        "output_ply": str(output_ply),
        "annotation_only_output_ply": (
            str(annotation_only_output_ply)
            if annotation_only_output_ply is not None
            else ""
        ),
        "total_points": int(labels.shape[0]),
        "exported_points": int(keep.sum()),
        "femur_candidates": int(np.sum(labels == LABEL_FEMUR_CANDIDATE)),
        "background_points": int(np.sum(labels == LABEL_BACKGROUND)),
        "ignore_points": int(np.sum(labels == LABEL_IGNORE)),
        "valid_contours": num_valid_contours,
        "annotation_markers": marker_count,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "input_h5",
        "output_ply",
        "annotation_only_output_ply",
        "total_points",
        "exported_points",
        "femur_candidates",
        "background_points",
        "ignore_points",
        "valid_contours",
        "annotation_markers",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-export annotated pseudo-3D point cloud h5 files to PLY."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--annotated_dir", type=Path, default=None)
    input_group.add_argument("--h5_list", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default="*_pointcloud_annotated.h5")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument(
        "--output_subdir_template",
        type=str,
        default="{parent_name}",
    )
    parser.add_argument(
        "--output_filename_template",
        type=str,
        default="{input_stem}.ply",
    )
    parser.add_argument(
        "--annotation_only_filename_template",
        type=str,
        default="{annotation_only_stem}.ply",
    )
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--no_annotation_only_output",
        action="store_true",
        help="Disable the second PLY containing only point_label=1 points.",
    )

    parser.add_argument(
        "--background_mode",
        type=str,
        default="dim",
        choices=["dim", "gray", "hidden"],
    )
    parser.add_argument("--label_values", type=str, default="1")
    parser.add_argument("--femur_color", type=str, default="255,64,32")
    parser.add_argument("--background_alpha", type=int, default=90)
    parser.add_argument("--ignore_alpha", type=int, default=35)
    parser.add_argument("--include_measurement_endpoint", action="store_true")
    parser.add_argument("--include_frame_endpoints", action="store_true")
    parser.add_argument("--measurement_color", type=str, default="32,144,255")
    parser.add_argument("--frame_endpoint_color", type=str, default="255,220,32")
    parser.add_argument("--include_annotation_markers", action="store_true")
    parser.add_argument("--annotation_marker_color", type=str, default="0,255,160")
    parser.add_argument("--annotation_marker_size", type=float, default=3.0)
    parser.add_argument("--annotation_marker_stride", type=int, default=8)
    parser.add_argument("--max_annotation_markers", type=int, default=5000)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not (0 <= args.background_alpha <= 255):
        raise ValueError("--background_alpha must be in [0,255]")
    if not (0 <= args.ignore_alpha <= 255):
        raise ValueError("--ignore_alpha must be in [0,255]")
    if args.annotation_marker_size <= 0:
        raise ValueError("--annotation_marker_size must be > 0")
    if args.annotation_marker_stride < 1:
        raise ValueError("--annotation_marker_stride must be >= 1")
    if args.max_annotation_markers < 0:
        raise ValueError("--max_annotation_markers must be >= 0")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    ensure_numpy()

    items = collect_batch_items(
        annotated_dir=args.annotated_dir,
        h5_list=args.h5_list,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    print(f"Found annotated h5 files: {len(items)}")
    print(f"Output root             : {args.output_root or 'same as input parent'}")
    print(f"Background mode         : {args.background_mode}")
    print(f"Annotation-only output  : {not args.no_annotation_only_output}")

    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []

    for index, item in enumerate(items, start=1):
        input_h5 = Path(item.input_h5)
        output_ply, annotation_only_output_ply = resolve_output_paths(
            item,
            output_root=args.output_root,
            output_subdir_template=args.output_subdir_template,
            output_filename_template=args.output_filename_template,
            annotation_only_filename_template=args.annotation_only_filename_template,
        )
        if args.no_annotation_only_output:
            annotation_only_output_ply = None

        print("\n" + "=" * 72)
        print(f"[{index}/{len(items)}] input_h5: {input_h5}")
        print(f"output_ply      : {output_ply}")
        if annotation_only_output_ply is not None:
            print(f"annotation_only : {annotation_only_output_ply}")

        requested_outputs = [output_ply]
        if annotation_only_output_ply is not None:
            requested_outputs.append(annotation_only_output_ply)
        if args.skip_existing and all(path.exists() for path in requested_outputs):
            print("Skipping existing outputs.")
            skipped += 1
            rows.append(
                {
                    "status": "skipped",
                    "input_h5": str(input_h5),
                    "output_ply": str(output_ply),
                    "annotation_only_output_ply": (
                        str(annotation_only_output_ply)
                        if annotation_only_output_ply is not None
                        else ""
                    ),
                }
            )
            continue

        try:
            result = export_one(
                input_h5=input_h5,
                output_ply=output_ply,
                annotation_only_output_ply=annotation_only_output_ply,
                args=args,
            )
            result["status"] = "ok"
            result["error"] = ""
            rows.append(result)
            processed += 1
            print(
                "Exported: "
                f"points={result['exported_points']}, "
                f"femur={result['femur_candidates']}, "
                f"markers={result['annotation_markers']}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((input_h5, message))
            rows.append(
                {
                    "status": "failed",
                    "input_h5": str(input_h5),
                    "output_ply": str(output_ply),
                    "annotation_only_output_ply": (
                        str(annotation_only_output_ply)
                        if annotation_only_output_ply is not None
                        else ""
                    ),
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

    print("\nBatch annotated PLY export summary:")
    print(f"  processed: {processed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {len(failures)}")

    if failures:
        for path, message in failures:
            print(f"  FAILED {path}: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
