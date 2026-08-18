from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import h5py
import numpy as np


DEFAULT_REBUILD_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722"
)
DEFAULT_ANNOTATED_ROOT = DEFAULT_REBUILD_ROOT / "combined_v2_annotated_corr_strict"
DEFAULT_PLY_ROOT = DEFAULT_REBUILD_ROOT / "combined_v2_annotated_ply"
DEFAULT_PATTERN = "*_pointcloud_annotated_foreground_combined_v2.h5"
INPUT_SUFFIX = "_pointcloud_annotated_foreground_combined_v2.h5"

SOURCE_GLOBAL = np.uint16(1 << 0)
SOURCE_LOCAL_PERCENTILE = np.uint16(1 << 1)
SOURCE_TOPHAT = np.uint16(1 << 2)
SOURCE_CONTEXT_GRID = np.uint16(1 << 3)

LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_FEMUR_CANDIDATE = 1

FEMUR_COLOR = np.asarray([255, 64, 32], dtype=np.uint8)
GLOBAL_COLOR = np.asarray([255, 255, 255], dtype=np.uint8)
LOCAL_PERCENTILE_COLOR = np.asarray([64, 200, 255], dtype=np.uint8)
TOPHAT_COLOR = np.asarray([255, 200, 64], dtype=np.uint8)
CONTEXT_GRID_COLOR = np.asarray([120, 120, 120], dtype=np.uint8)
BACKGROUND_ALPHA = np.uint8(90)
IGNORE_ALPHA = np.uint8(35)
FLOAT_TEXT_ATOL = 5.1e-7

VERTEX_PROPERTIES = (
    ("float", "x"),
    ("float", "y"),
    ("float", "z"),
    ("uchar", "red"),
    ("uchar", "green"),
    ("uchar", "blue"),
    ("uchar", "alpha"),
    ("int", "label"),
    ("int", "frame_order"),
    ("float", "confidence"),
    ("int", "source_flags"),
    ("int", "source_type"),
    ("float", "sampling_confidence"),
)


@dataclass(frozen=True)
class CheckItem:
    video_name: str
    input_h5: Path
    output_ply: Path
    annotation_only_ply: Path


@dataclass(frozen=True)
class PlyHeader:
    vertex_count: int
    vertex_properties: tuple[tuple[str, str], ...]
    edge_count: int


@dataclass(frozen=True)
class CheckResult:
    video_name: str
    num_points: int
    num_positive: int
    num_valid_contours: int
    full_ply_bytes: int
    annotation_only_ply_bytes: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_items(args: argparse.Namespace) -> list[CheckItem]:
    input_paths = sorted(
        path
        for path in Path(args.annotated_root).rglob(args.h5_pattern)
        if path.is_file()
    )
    _require(
        len(input_paths) == args.expected_videos,
        f"Matched annotated H5 files={len(input_paths)}, expected={args.expected_videos}",
    )

    items: list[CheckItem] = []
    seen: set[str] = set()
    for input_h5 in input_paths:
        _require(
            input_h5.name.endswith(INPUT_SUFFIX),
            f"Unexpected annotated H5 filename: {input_h5.name}",
        )
        video_name = input_h5.name[: -len(INPUT_SUFFIX)]
        _require(video_name not in seen, f"Duplicate video_name: {video_name}")
        seen.add(video_name)
        output_dir = Path(args.ply_root) / input_h5.parent.name
        items.append(
            CheckItem(
                video_name=video_name,
                input_h5=input_h5,
                output_ply=output_dir / f"{input_h5.stem}.ply",
                annotation_only_ply=(
                    output_dir
                    / f"{video_name}_pointcloud_annotation_only_foreground_combined_v2.ply"
                ),
            )
        )
    return items


def _read_ply_header(stream: TextIO, path: Path) -> PlyHeader:
    first = stream.readline().rstrip("\n")
    _require(first == "ply", f"Invalid PLY magic in {path}: {first!r}")
    format_line = stream.readline().rstrip("\n")
    _require(format_line == "format ascii 1.0", f"Expected ASCII PLY in {path}")

    current_element = ""
    vertex_count = -1
    edge_count = 0
    vertex_properties: list[tuple[str, str]] = []
    while True:
        line = stream.readline()
        _require(line != "", f"Unexpected EOF in PLY header: {path}")
        text = line.rstrip("\n")
        if text == "end_header":
            break
        if not text or text.startswith("comment "):
            continue
        tokens = text.split()
        if tokens[0] == "element":
            _require(len(tokens) == 3, f"Malformed element line in {path}: {text}")
            current_element = tokens[1]
            count = int(tokens[2])
            if current_element == "vertex":
                vertex_count = count
            elif current_element == "edge":
                edge_count = count
            else:
                raise AssertionError(f"Unexpected PLY element in {path}: {text}")
        elif tokens[0] == "property":
            _require(len(tokens) == 3, f"Malformed property line in {path}: {text}")
            if current_element == "vertex":
                vertex_properties.append((tokens[1], tokens[2]))
        else:
            raise AssertionError(f"Unexpected PLY header line in {path}: {text}")

    _require(vertex_count >= 0, f"PLY has no vertex element: {path}")
    return PlyHeader(
        vertex_count=vertex_count,
        vertex_properties=tuple(vertex_properties),
        edge_count=edge_count,
    )


def _expected_colors(
    intensity: np.ndarray,
    alpha: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    source_flags: np.ndarray,
    *,
    annotation_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gray = intensity.astype(np.uint8)
    colors = np.stack([gray, gray, gray], axis=1)
    out_alpha = alpha.astype(np.uint8).copy()

    if not annotation_only:
        colors = (colors.astype(np.float32) * 0.35).astype(np.uint8)
        out_alpha[:] = BACKGROUND_ALPHA

    flags = source_flags.astype(np.uint16)
    colors[(flags & SOURCE_CONTEXT_GRID) != 0] = CONTEXT_GRID_COLOR
    colors[(flags & SOURCE_TOPHAT) != 0] = TOPHAT_COLOR
    colors[(flags & SOURCE_LOCAL_PERCENTILE) != 0] = LOCAL_PERCENTILE_COLOR
    colors[(flags & SOURCE_GLOBAL) != 0] = GLOBAL_COLOR

    ignore = labels == LABEL_IGNORE
    colors[ignore] = np.asarray([60, 60, 60], dtype=np.uint8)
    out_alpha[ignore] = IGNORE_ALPHA

    positive = valid_mask & (labels == LABEL_FEMUR_CANDIDATE)
    colors[positive] = FEMUR_COLOR
    out_alpha[positive] = np.uint8(255)
    return colors, out_alpha


def _assert_float_columns(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> None:
    _require(np.isfinite(actual).all(), f"PLY {label} contains NaN or Inf")
    if not np.allclose(actual, expected, rtol=0.0, atol=FLOAT_TEXT_ATOL):
        max_error = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"PLY {label} differs from H5; max_abs_error={max_error}")


def _assert_integer_column(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> None:
    rounded = np.rint(actual)
    _require(np.array_equal(actual, rounded), f"PLY {label} contains non-integer text")
    _require(
        np.array_equal(rounded.astype(np.int64), expected.astype(np.int64)),
        f"PLY {label} differs from H5",
    )


def _selection_for_file(
    point_cloud: h5py.Group,
    annotation: h5py.Group,
    *,
    annotation_only: bool,
) -> tuple[int, np.ndarray | None]:
    num_points = int(point_cloud["points"].shape[0])
    if not annotation_only:
        return num_points, None
    labels = annotation["point_label"][:].astype(np.int8)
    valid_mask = annotation["valid_mask"][:].astype(bool)
    indices = np.flatnonzero(valid_mask & (labels == LABEL_FEMUR_CANDIDATE))
    return int(indices.size), indices.astype(np.int64)


def _read_vertex_chunk(stream: TextIO, count: int, path: Path) -> np.ndarray:
    lines = []
    for _ in range(count):
        line = stream.readline()
        _require(line != "", f"Unexpected EOF in PLY vertex body: {path}")
        lines.append(line)
    values = np.fromstring("".join(lines), sep=" ", dtype=np.float64)
    expected_values = count * len(VERTEX_PROPERTIES)
    _require(
        values.size == expected_values,
        f"PLY vertex field count differs in {path}: {values.size} != {expected_values}",
    )
    return values.reshape(count, len(VERTEX_PROPERTIES))


def _assert_vertex_chunk(
    actual: np.ndarray,
    point_cloud: h5py.Group,
    annotation: h5py.Group,
    selector: slice | np.ndarray,
    *,
    annotation_only: bool,
) -> None:
    points = point_cloud["points"][selector].astype(np.float32).astype(np.float64)
    intensity = point_cloud["intensity"][selector].astype(np.uint8)
    alpha = point_cloud["alpha"][selector].astype(np.uint8)
    confidence = (
        point_cloud["confidence"][selector].astype(np.float32).astype(np.float64)
    )
    frame_order = point_cloud["frame_order"][selector].astype(np.int64)
    source_flags = point_cloud["source_flags"][selector].astype(np.uint16)
    source_type = point_cloud["source_type"][selector].astype(np.uint8)
    sampling_confidence = (
        point_cloud["sampling_confidence"][selector]
        .astype(np.float32)
        .astype(np.float64)
    )
    labels = annotation["point_label"][selector].astype(np.int8)
    valid_mask = annotation["valid_mask"][selector].astype(bool)
    colors, out_alpha = _expected_colors(
        intensity,
        alpha,
        labels,
        valid_mask,
        source_flags,
        annotation_only=annotation_only,
    )

    _assert_float_columns(actual[:, 0:3], points, label="xyz")
    for column, expected, label in (
        (3, colors[:, 0], "red"),
        (4, colors[:, 1], "green"),
        (5, colors[:, 2], "blue"),
        (6, out_alpha, "alpha"),
        (7, labels, "label"),
        (8, frame_order, "frame_order"),
        (10, source_flags, "source_flags"),
        (11, source_type, "source_type"),
    ):
        _assert_integer_column(actual[:, column], expected, label=label)
    _assert_float_columns(actual[:, 9], confidence, label="confidence")
    _assert_float_columns(
        actual[:, 12],
        sampling_confidence,
        label="sampling_confidence",
    )


def _check_ply_body(
    h5_path: Path,
    ply_path: Path,
    *,
    annotation_only: bool,
    chunk_size: int,
) -> int:
    _require(ply_path.is_file(), f"PLY not found: {ply_path}")
    with h5py.File(h5_path, "r") as h5_file, ply_path.open() as stream:
        point_cloud = h5_file["point_cloud"]
        annotation = h5_file["annotation"]
        required_point_keys = {
            "points",
            "intensity",
            "alpha",
            "confidence",
            "frame_order",
            "source_flags",
            "source_type",
            "sampling_confidence",
        }
        missing = required_point_keys - set(point_cloud.keys())
        _require(not missing, f"H5 point_cloud fields missing: {sorted(missing)}")

        expected_count, selected_indices = _selection_for_file(
            point_cloud,
            annotation,
            annotation_only=annotation_only,
        )
        header = _read_ply_header(stream, ply_path)
        _require(
            header.vertex_count == expected_count,
            f"PLY vertex count={header.vertex_count}, expected={expected_count}: {ply_path}",
        )
        _require(
            header.vertex_properties == VERTEX_PROPERTIES,
            f"PLY vertex properties differ: {header.vertex_properties}",
        )
        _require(header.edge_count == 0, f"Unexpected PLY edges in {ply_path}")

        offset = 0
        while offset < expected_count:
            count = min(chunk_size, expected_count - offset)
            actual = _read_vertex_chunk(stream, count, ply_path)
            if selected_indices is None:
                selector: slice | np.ndarray = slice(offset, offset + count)
            else:
                selector = selected_indices[offset : offset + count]
            _assert_vertex_chunk(
                actual,
                point_cloud,
                annotation,
                selector,
                annotation_only=annotation_only,
            )
            offset += count

        trailing = stream.read().strip()
        _require(not trailing, f"Unexpected data after PLY vertices: {ply_path}")
    return expected_count


def check_one(item: CheckItem, args: argparse.Namespace) -> CheckResult:
    _require(item.input_h5.is_file(), f"Annotated H5 not found: {item.input_h5}")
    full_count = _check_ply_body(
        item.input_h5,
        item.output_ply,
        annotation_only=False,
        chunk_size=args.chunk_size,
    )
    positive_count = _check_ply_body(
        item.input_h5,
        item.annotation_only_ply,
        annotation_only=True,
        chunk_size=args.chunk_size,
    )

    with h5py.File(item.input_h5, "r") as h5_file:
        num_points = int(h5_file["point_cloud/points"].shape[0])
        valid_contours = h5_file["frame_annotation/valid_contour"][:].astype(bool)
        num_valid_contours = int(valid_contours.sum())
    _require(full_count == num_points, "Full PLY did not preserve all H5 points")
    return CheckResult(
        video_name=item.video_name,
        num_points=num_points,
        num_positive=positive_count,
        num_valid_contours=num_valid_contours,
        full_ply_bytes=item.output_ply.stat().st_size,
        annotation_only_ply_bytes=item.annotation_only_ply.stat().st_size,
    )


def _check_summary(path: Path, items: list[CheckItem], results: list[CheckResult]) -> None:
    _require(path.is_file(), f"Batch summary CSV not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    _require(len(rows) == len(items), f"Summary rows={len(rows)}, expected={len(items)}")
    rows_by_input = {Path(row.get("input_h5", "")).resolve(): row for row in rows}
    _require(len(rows_by_input) == len(rows), "Summary has duplicate input H5 paths")
    results_by_video = {result.video_name: result for result in results}

    for item in items:
        key = item.input_h5.resolve()
        _require(key in rows_by_input, f"Summary row missing: {item.input_h5}")
        row = rows_by_input[key]
        result = results_by_video[item.video_name]
        _require(row.get("status") == "ok", f"Summary status is not ok: {item.video_name}")
        _require(not row.get("error", "").strip(), f"Summary error: {row.get('error')}")
        expected_paths = {
            "output_ply": item.output_ply,
            "annotation_only_output_ply": item.annotation_only_ply,
        }
        for field, expected in expected_paths.items():
            actual = Path(row.get(field, ""))
            _require(
                actual.resolve() == expected.resolve(),
                f"Summary {field} differs for {item.video_name}: {actual} != {expected}",
            )
        expected_counts = {
            "total_points": result.num_points,
            "exported_points": result.num_points,
            "femur_candidates": result.num_positive,
            "valid_contours": result.num_valid_contours,
            "annotation_markers": 0,
        }
        for field, expected in expected_counts.items():
            actual = int(row.get(field, -1))
            _require(
                actual == expected,
                f"Summary {field} differs for {item.video_name}: {actual} != {expected}",
            )


def _check_output_file_set(ply_root: Path, items: list[CheckItem]) -> None:
    actual = {path.resolve() for path in Path(ply_root).rglob("*.ply") if path.is_file()}
    expected = {
        path.resolve()
        for item in items
        for path in (item.output_ply, item.annotation_only_ply)
    }
    _require(
        actual == expected,
        "PLY output file set differs: "
        f"missing={sorted(str(path) for path in expected - actual)}, "
        f"unexpected={sorted(str(path) for path in actual - expected)}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only full-body comparison of Stage 4 combined_v2 annotated "
            "PLY outputs against strict annotated H5 files."
        )
    )
    parser.add_argument("--annotated_root", type=Path, default=DEFAULT_ANNOTATED_ROOT)
    parser.add_argument("--ply_root", type=Path, default=DEFAULT_PLY_ROOT)
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--h5_pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--expected_videos", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=50000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summary_csv is None:
        args.summary_csv = Path(args.ply_root) / "summary.csv"
    _require(args.chunk_size > 0, "chunk_size must be positive")
    _require(
        args.annotated_root.is_dir(),
        f"Annotated root not found: {args.annotated_root}",
    )
    items = _build_items(args)
    _check_output_file_set(args.ply_root, items)

    print("Stage 4 combined_v2 annotated PLY check (read-only)")
    print(f"annotated_root : {args.annotated_root}")
    print(f"ply_root       : {args.ply_root}")
    print(f"summary_csv    : {args.summary_csv}")
    print(f"videos         : {len(items)}")
    print(f"chunk_size     : {args.chunk_size}")

    results: list[CheckResult] = []
    failures: list[str] = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] checking {item.video_name}")
        try:
            result = check_one(item, args)
        except Exception as exc:
            failures.append(f"{item.video_name}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        print(
            "  [OK] "
            f"vertices={result.num_points}, positive={result.num_positive}, "
            f"valid_contours={result.num_valid_contours}, "
            f"bytes={result.full_ply_bytes + result.annotation_only_ply_bytes}"
        )

    if not failures:
        try:
            _check_summary(args.summary_csv, items, results)
            print("[OK] batch summary CSV matches all H5/PLY outputs")
        except Exception as exc:
            failures.append(f"summary CSV: {type(exc).__name__}: {exc}")
            print(f"[FAIL] summary CSV: {type(exc).__name__}: {exc}")

    print("\nAnnotated PLY summary")
    print(f"  videos_checked       : {len(results)}/{len(items)}")
    print(f"  full_vertices        : {sum(result.num_points for result in results)}")
    print(f"  annotation_vertices  : {sum(result.num_positive for result in results)}")
    print(
        "  ply_bytes_checked     : "
        f"{sum(result.full_ply_bytes + result.annotation_only_ply_bytes for result in results)}"
    )
    print(f"  failures             : {len(failures)}")
    print("  files_written        : 0")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Stage 4 combined_v2 annotated PLY checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
