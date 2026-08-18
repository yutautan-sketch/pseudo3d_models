from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_REBUILD_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722"
)
DEFAULT_CANDIDATES_LIST = DEFAULT_REBUILD_ROOT / "candidates.txt"
DEFAULT_RAW_ROOT = DEFAULT_REBUILD_ROOT / "combined_v2_default_corr"
DEFAULT_ANNOTATED_ROOT = DEFAULT_REBUILD_ROOT / "combined_v2_annotated_corr_strict"

LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_FEMUR_CANDIDATE = 1

REQUIRED_COMBINED_DATASETS = {
    "points",
    "intensity",
    "alpha",
    "confidence",
    "frame_index",
    "frame_order",
    "pixel_xy",
    "source_type",
    "source_flags",
    "sampling_confidence",
}
EXPECTED_ADDED_POINT_CLOUD_ATTRS = {
    "source_flags_inferred",
    "sampling_confidence_inferred",
}
REQUIRED_FRAME_ANNOTATION_DATASETS = {
    "frame_order",
    "frame_index",
    "object_index",
    "bbox_index",
    "bbox_xml_xyxy",
    "bbox_xml_size_wh",
    "bbox_raw_xyxy",
    "bbox_local_xyxy",
    "valid_contour",
    "contour_area",
    "binary_area",
    "foreground_ratio_in_bbox",
    "num_frame_points",
    "num_labeled_points",
    "contour_positive_points",
    "fallback_used",
    "fallback_percentile",
    "fallback_attempts",
    "num_labeled_points_union",
    "annotation_reason",
    "object_name",
    "xml_path",
}


@dataclass(frozen=True)
class CheckItem:
    video_name: str
    pseudo3d_h5: Path
    raw_h5: Path
    annotated_h5: Path


@dataclass(frozen=True)
class CheckResult:
    video_name: str
    num_points: int
    num_frames: int
    num_point_cloud_datasets: int
    num_bbox_rows: int
    num_valid_contours: int
    num_labeled_points: int
    num_xml_paths: int


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _attr_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = _decode_text(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Cannot interpret as bool: {value!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _attr_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False

    if left_array.dtype.kind in {"S", "U", "O"} or right_array.dtype.kind in {
        "S",
        "U",
        "O",
    }:
        left_values = [_decode_text(value) for value in left_array.reshape(-1)]
        right_values = [_decode_text(value) for value in right_array.reshape(-1)]
        return left_values == right_values

    return bool(np.array_equal(left_array, right_array, equal_nan=True))


def _read_candidate_paths(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidates list not found: {path}")

    candidates: list[Path] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []
            path_key = next(
                (
                    key
                    for key in ("pseudo3d_h5", "input_h5", "h5_path")
                    if key in reader.fieldnames
                ),
                reader.fieldnames[0],
            )
            for row in reader:
                value = row.get(path_key, "").strip()
                if value:
                    candidates.append(Path(value))
    else:
        with path.open() as f:
            for line in f:
                value = line.strip()
                if value and not value.startswith("#"):
                    candidates.append(Path(value))

    if not candidates:
        raise ValueError(f"Candidates list is empty: {path}")
    return candidates


def _derive_video_name(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _build_items(args: argparse.Namespace) -> list[CheckItem]:
    items: list[CheckItem] = []
    seen: set[str] = set()
    for pseudo3d_h5 in _read_candidate_paths(args.candidates_list):
        video_name = _derive_video_name(pseudo3d_h5, args.video_suffix_to_strip)
        _require(video_name not in seen, f"Duplicate video_name: {video_name}")
        seen.add(video_name)

        context = {"video_name": video_name}
        raw_relative = args.raw_h5_template.format(**context)
        annotated_relative = args.annotated_h5_template.format(**context)
        items.append(
            CheckItem(
                video_name=video_name,
                pseudo3d_h5=pseudo3d_h5,
                raw_h5=Path(args.raw_root) / raw_relative,
                annotated_h5=Path(args.annotated_root) / annotated_relative,
            )
        )
    return items


def _assert_dataset_equal(
    raw_dataset: h5py.Dataset,
    annotated_dataset: h5py.Dataset,
    *,
    key: str,
) -> None:
    _require(
        raw_dataset.shape == annotated_dataset.shape,
        f"point_cloud/{key} shape changed: "
        f"raw={raw_dataset.shape}, annotated={annotated_dataset.shape}",
    )
    _require(
        raw_dataset.dtype == annotated_dataset.dtype,
        f"point_cloud/{key} dtype changed: "
        f"raw={raw_dataset.dtype}, annotated={annotated_dataset.dtype}",
    )
    _require(
        np.array_equal(raw_dataset[...], annotated_dataset[...], equal_nan=True),
        f"point_cloud/{key} values or ordering changed",
    )

    filter_fields = (
        "compression",
        "compression_opts",
        "shuffle",
        "fletcher32",
        "scaleoffset",
    )
    for field in filter_fields:
        raw_value = getattr(raw_dataset, field)
        annotated_value = getattr(annotated_dataset, field)
        _require(
            raw_value == annotated_value,
            f"point_cloud/{key} {field} changed: "
            f"raw={raw_value!r}, annotated={annotated_value!r}",
        )


def _assert_point_cloud_attrs(
    raw_group: h5py.Group,
    annotated_group: h5py.Group,
) -> None:
    raw_keys = set(raw_group.attrs.keys())
    annotated_keys = set(annotated_group.attrs.keys())
    missing = raw_keys - annotated_keys
    _require(not missing, f"Annotated point_cloud attributes missing: {sorted(missing)}")

    added = annotated_keys - raw_keys
    _require(
        added <= EXPECTED_ADDED_POINT_CLOUD_ATTRS,
        f"Unexpected point_cloud attributes added: {sorted(added)}",
    )
    for key in sorted(raw_keys):
        _require(
            _attr_equal(raw_group.attrs[key], annotated_group.attrs[key]),
            f"point_cloud attribute changed: {key}: "
            f"raw={raw_group.attrs[key]!r}, annotated={annotated_group.attrs[key]!r}",
        )

    for key in EXPECTED_ADDED_POINT_CLOUD_ATTRS:
        _require(key in annotated_group.attrs, f"Missing point_cloud attribute: {key}")
        _require(
            not _attr_bool(annotated_group.attrs[key]),
            f"Current combined_v2 field was unexpectedly inferred: {key}",
        )

    sampling_mode = _decode_text(annotated_group.attrs.get("sampling_mode", ""))
    _require(
        sampling_mode == "combined_v2",
        f"Expected point_cloud sampling_mode=combined_v2, got {sampling_mode!r}",
    )


def _assert_root_metadata(
    annotated_file: h5py.File,
    *,
    item: CheckItem,
    num_points: int,
    expected_xml_dir_name: str,
) -> Path:
    attrs = annotated_file.attrs
    expected_values = {
        "video_name": item.video_name,
        "geometry_key": "corr",
        "point_cloud_sampling_mode": "combined_v2",
        "xml_annotation_dir_name": expected_xml_dir_name,
    }
    for key, expected in expected_values.items():
        _require(key in attrs, f"Missing root metadata: {key}")
        actual = _decode_text(attrs[key])
        _require(actual == expected, f"Root metadata {key}: {actual!r} != {expected!r}")

    expected_paths = {
        "source_point_cloud_h5": item.raw_h5,
        "source_pseudo3d_h5": item.pseudo3d_h5,
    }
    for key, expected in expected_paths.items():
        _require(key in attrs, f"Missing root metadata: {key}")
        actual = Path(_decode_text(attrs[key]))
        _require(
            actual.resolve() == expected.resolve(),
            f"Root metadata {key}: {actual} != {expected}",
        )

    true_flags = (
        "point_cloud_has_source_flags",
        "point_cloud_has_sampling_confidence",
        "strict_xml_annotation_dir",
    )
    false_flags = (
        "source_flags_inferred",
        "sampling_confidence_inferred",
        "enable_contour_fallback",
    )
    for key in true_flags:
        _require(key in attrs and _attr_bool(attrs[key]), f"Expected root {key}=True")
    for key in false_flags:
        _require(key in attrs and not _attr_bool(attrs[key]), f"Expected root {key}=False")

    _require(
        int(attrs.get("num_points", -1)) == num_points,
        f"Root num_points does not match point_cloud: {attrs.get('num_points')}",
    )
    _require(
        int(attrs.get("num_fallback_used", -1)) == 0,
        f"Expected num_fallback_used=0, got {attrs.get('num_fallback_used')}",
    )

    _require("voc_xml_root" in attrs, "Missing root metadata: voc_xml_root")
    return Path(_decode_text(attrs["voc_xml_root"]))


def _assert_annotation_group(
    group: h5py.Group,
    *,
    num_points: int,
) -> tuple[int, int]:
    _require("point_label" in group, "Missing annotation/point_label")
    _require("valid_mask" in group, "Missing annotation/valid_mask")
    labels = group["point_label"][:]
    valid = group["valid_mask"][:]

    _require(labels.shape == (num_points,), f"Unexpected point_label shape: {labels.shape}")
    _require(valid.shape == (num_points,), f"Unexpected valid_mask shape: {valid.shape}")
    _require(labels.dtype == np.dtype("int8"), f"Unexpected point_label dtype: {labels.dtype}")
    _require(valid.dtype == np.dtype("bool"), f"Unexpected valid_mask dtype: {valid.dtype}")
    _require(group["point_label"].compression == "gzip", "point_label is not gzip")
    _require(group["valid_mask"].compression == "gzip", "valid_mask is not gzip")

    label_values = set(int(value) for value in np.unique(labels))
    expected_labels = {LABEL_IGNORE, LABEL_BACKGROUND, LABEL_FEMUR_CANDIDATE}
    _require(
        label_values <= expected_labels,
        f"Unexpected annotation label values: {sorted(label_values)}",
    )
    _require(
        np.array_equal(valid, labels != LABEL_IGNORE),
        "annotation/valid_mask differs from point_label != -1",
    )
    return int(np.sum(labels == LABEL_FEMUR_CANDIDATE)), int(valid.sum())


def _decode_string_dataset(dataset: h5py.Dataset) -> list[str]:
    return [_decode_text(value) for value in dataset[:].reshape(-1)]


def _assert_frame_annotation_group(
    group: h5py.Group,
    *,
    pseudo3d_h5: Path,
    voc_xml_root: Path,
    video_name: str,
    expected_xml_dir_name: str,
    expected_labeled_points: int,
) -> tuple[int, int, int]:
    keys = set(group.keys())
    missing = REQUIRED_FRAME_ANNOTATION_DATASETS - keys
    _require(not missing, f"Missing frame_annotation datasets: {sorted(missing)}")

    lengths = {key: int(group[key].shape[0]) for key in keys}
    row_counts = set(lengths.values())
    _require(
        len(row_counts) == 1,
        f"frame_annotation first dimensions differ: {lengths}",
    )
    num_rows = next(iter(row_counts), 0)

    string_keys = {
        "annotation_reason",
        "selected_contour_source",
        "object_name",
        "xml_path",
    }
    for key in keys - string_keys:
        _require(
            group[key].compression == "gzip",
            f"frame_annotation/{key} is not gzip-compressed",
        )

    with h5py.File(pseudo3d_h5, "r") as pseudo3d_file:
        _require("frame_indices" in pseudo3d_file, "Pseudo3D H5 has no frame_indices")
        frame_indices = pseudo3d_file["frame_indices"][:].astype(np.int64)

    frame_order = group["frame_order"][:].astype(np.int64)
    saved_frame_index = group["frame_index"][:].astype(np.int64)
    if num_rows:
        _require(int(frame_order.min()) >= 0, "Negative frame_annotation/frame_order")
        _require(
            int(frame_order.max()) < len(frame_indices),
            "frame_annotation/frame_order exceeds pseudo3D frame count",
        )
        _require(
            np.array_equal(saved_frame_index, frame_indices[frame_order]),
            "frame_annotation frame_index is not aligned with pseudo3D frame_indices",
        )

    expected_xml_parent = voc_xml_root / video_name / expected_xml_dir_name
    xml_paths = [Path(value) for value in _decode_string_dataset(group["xml_path"])]
    for xml_path in xml_paths:
        _require(xml_path.is_file(), f"Saved XML path does not exist: {xml_path}")
        _require(
            xml_path.parent.resolve() == expected_xml_parent.resolve(),
            f"Saved XML escaped strict annotation directory: {xml_path}",
        )

    fallback_used = group["fallback_used"][:].astype(bool)
    _require(not fallback_used.any(), "Foreground annotation unexpectedly used fallback")
    valid_contours = group["valid_contour"][:].astype(bool)
    union_counts = group["num_labeled_points_union"][:].astype(np.int64)
    total_union_points = 0
    for order in np.unique(frame_order):
        frame_counts = np.unique(union_counts[frame_order == order])
        _require(
            frame_counts.size == 1,
            f"Frame {order} has inconsistent num_labeled_points_union values",
        )
        total_union_points += int(frame_counts[0])
    _require(
        total_union_points == expected_labeled_points,
        "Per-frame contour-union counts do not equal annotation positive count",
    )

    return num_rows, int(valid_contours.sum()), len(xml_paths)


def check_one(item: CheckItem, args: argparse.Namespace) -> CheckResult:
    for label, path in (
        ("pseudo3D H5", item.pseudo3d_h5),
        ("raw point-cloud H5", item.raw_h5),
        ("annotated H5", item.annotated_h5),
    ):
        _require(path.is_file(), f"{label} not found: {path}")

    with h5py.File(item.raw_h5, "r") as raw_file, h5py.File(
        item.annotated_h5, "r"
    ) as annotated_file:
        _require("point_cloud" in raw_file, "Raw H5 has no point_cloud group")
        _require("point_cloud" in annotated_file, "Annotated H5 has no point_cloud group")
        raw_group = raw_file["point_cloud"]
        annotated_group = annotated_file["point_cloud"]

        raw_keys = set(raw_group.keys())
        annotated_keys = set(annotated_group.keys())
        _require(
            raw_keys == annotated_keys,
            "point_cloud dataset set changed: "
            f"missing={sorted(raw_keys - annotated_keys)}, "
            f"added={sorted(annotated_keys - raw_keys)}",
        )
        missing_combined = REQUIRED_COMBINED_DATASETS - raw_keys
        _require(
            not missing_combined,
            f"Raw combined_v2 H5 lacks datasets: {sorted(missing_combined)}",
        )

        for key in sorted(raw_keys):
            _assert_dataset_equal(raw_group[key], annotated_group[key], key=key)
        _assert_point_cloud_attrs(raw_group, annotated_group)

        num_points = int(raw_group["points"].shape[0])
        frame_order = raw_group["frame_order"][:].astype(np.int64)
        num_frames = int(np.unique(frame_order).size)

        required_groups = {"annotation", "frame_annotation", "measurement"}
        missing_groups = required_groups - set(annotated_file.keys())
        _require(not missing_groups, f"Missing annotated H5 groups: {sorted(missing_groups)}")

        num_labeled_points, _ = _assert_annotation_group(
            annotated_file["annotation"],
            num_points=num_points,
        )
        voc_xml_root = _assert_root_metadata(
            annotated_file,
            item=item,
            num_points=num_points,
            expected_xml_dir_name=args.expected_xml_dir_name,
        )
        num_bbox_rows, num_valid_contours, num_xml_paths = (
            _assert_frame_annotation_group(
                annotated_file["frame_annotation"],
                pseudo3d_h5=item.pseudo3d_h5,
                voc_xml_root=voc_xml_root,
                video_name=item.video_name,
                expected_xml_dir_name=args.expected_xml_dir_name,
                expected_labeled_points=num_labeled_points,
            )
        )

        root_counts = {
            "num_voc_bboxes": num_bbox_rows,
            "num_valid_contours": num_valid_contours,
            "num_labeled_points": num_labeled_points,
        }
        for key, expected in root_counts.items():
            actual = int(annotated_file.attrs.get(key, -1))
            _require(actual == expected, f"Root {key}: {actual} != {expected}")

    return CheckResult(
        video_name=item.video_name,
        num_points=num_points,
        num_frames=num_frames,
        num_point_cloud_datasets=len(raw_keys),
        num_bbox_rows=num_bbox_rows,
        num_valid_contours=num_valid_contours,
        num_labeled_points=num_labeled_points,
        num_xml_paths=num_xml_paths,
    )


def _check_summary_csv(
    path: Path,
    *,
    items: list[CheckItem],
    results: list[CheckResult],
    expected_xml_dir_name: str,
) -> None:
    path = Path(path)
    _require(path.is_file(), f"Annotation summary CSV not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    _require(len(rows) == len(items), f"Summary rows={len(rows)}, expected={len(items)}")
    rows_by_video = {row.get("video_name", ""): row for row in rows}
    _require(
        len(rows_by_video) == len(rows),
        "Summary CSV contains duplicate or empty video_name values",
    )
    results_by_video = {result.video_name: result for result in results}

    for item in items:
        _require(item.video_name in rows_by_video, f"Summary row missing: {item.video_name}")
        row = rows_by_video[item.video_name]
        result = results_by_video[item.video_name]
        _require(row.get("status") == "ok", f"Summary status is not ok: {item.video_name}")
        _require(not row.get("error", "").strip(), f"Summary has error: {row.get('error')}")
        _require(
            row.get("point_cloud_sampling_mode") == "combined_v2",
            f"Summary sampling mode is not combined_v2: {item.video_name}",
        )
        _require(
            row.get("xml_annotation_dir_name") == expected_xml_dir_name,
            f"Summary XML directory differs: {item.video_name}",
        )
        _require(
            _attr_bool(row.get("strict_xml_annotation_dir", "")),
            f"Summary strict_xml_annotation_dir is false: {item.video_name}",
        )
        _require(
            not _attr_bool(row.get("source_flags_inferred", "")),
            f"Summary says source_flags were inferred: {item.video_name}",
        )
        expected_paths = {
            "pseudo3d_h5": item.pseudo3d_h5,
            "point_cloud_h5": item.raw_h5,
            "output_h5": item.annotated_h5,
        }
        for key, expected in expected_paths.items():
            actual = Path(row.get(key, ""))
            _require(
                actual.resolve() == expected.resolve(),
                f"Summary {key} differs for {item.video_name}: {actual} != {expected}",
            )
        expected_counts = {
            "num_points": result.num_points,
            "num_voc_bboxes": result.num_bbox_rows,
            "num_valid_contours": result.num_valid_contours,
            "num_labeled_points": result.num_labeled_points,
            "num_fallback_used": 0,
        }
        for key, expected in expected_counts.items():
            actual = int(row.get(key, -1))
            _require(
                actual == expected,
                f"Summary {key} differs for {item.video_name}: {actual} != {expected}",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Stage 4 check that annotation preserves every combined_v2 "
            "point-cloud dataset and appends an aligned strict-VOC annotation schema."
        )
    )
    parser.add_argument("--candidates_list", type=Path, default=DEFAULT_CANDIDATES_LIST)
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--annotated_root", type=Path, default=DEFAULT_ANNOTATED_ROOT)
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--video_suffix_to_strip", default="_ts448_oym96_corr")
    parser.add_argument(
        "--raw_h5_template",
        default="{video_name}/{video_name}_pointcloud_foreground_combined_v2.h5",
    )
    parser.add_argument(
        "--annotated_h5_template",
        default=(
            "{video_name}/"
            "{video_name}_pointcloud_annotated_foreground_combined_v2.h5"
        ),
    )
    parser.add_argument("--expected_xml_dir_name", default="annotations_renamed")
    parser.add_argument("--expected_videos", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summary_csv is None:
        args.summary_csv = Path(args.annotated_root) / "summary.csv"

    items = _build_items(args)
    _require(
        len(items) == args.expected_videos,
        f"Candidates={len(items)}, expected={args.expected_videos}",
    )

    print("Stage 4 annotation schema propagation check (read-only)")
    print(f"candidates_list : {args.candidates_list}")
    print(f"raw_root        : {args.raw_root}")
    print(f"annotated_root  : {args.annotated_root}")
    print(f"summary_csv     : {args.summary_csv}")
    print(f"videos          : {len(items)}")

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
            f"points={result.num_points}, frames={result.num_frames}, "
            f"datasets={result.num_point_cloud_datasets}, "
            f"bbox_rows={result.num_bbox_rows}, "
            f"valid_contours={result.num_valid_contours}, "
            f"labeled_points={result.num_labeled_points}"
        )

    if not failures:
        try:
            _check_summary_csv(
                args.summary_csv,
                items=items,
                results=results,
                expected_xml_dir_name=args.expected_xml_dir_name,
            )
            print("[OK] batch summary CSV matches all annotated H5 outputs")
        except Exception as exc:
            failures.append(f"summary CSV: {type(exc).__name__}: {exc}")
            print(f"[FAIL] summary CSV: {type(exc).__name__}: {exc}")

    print("\nSchema-propagation summary")
    print(f"  videos_checked          : {len(results)}/{len(items)}")
    print(f"  point_cloud_datasets    : {sum(r.num_point_cloud_datasets for r in results)}")
    print(f"  points_preserved        : {sum(r.num_points for r in results)}")
    print(f"  strict_xml_paths        : {sum(r.num_xml_paths for r in results)}")
    print(f"  annotation_positive     : {sum(r.num_labeled_points for r in results)}")
    print(f"  failures                : {len(failures)}")
    print("  files_written           : 0")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Stage 4 annotation schema propagation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
