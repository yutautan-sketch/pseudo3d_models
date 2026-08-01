from __future__ import annotations

import argparse
import csv
import hashlib
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
DEFAULT_SOURCE_ROOT = DEFAULT_REBUILD_ROOT / "combined_v2_annotated_corr_strict"
DEFAULT_OUTPUT_DIR = DEFAULT_REBUILD_ROOT / "collected_combined_v2_annotated_corr_strict"


@dataclass(frozen=True)
class CollectItem:
    video_name: str
    source_h5: Path
    collected_h5: Path


@dataclass(frozen=True)
class CheckResult:
    video_name: str
    file_size: int
    sha256: str
    num_groups: int
    num_datasets: int
    num_points: int
    num_labeled_points: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _array_equal(left: Any, right: Any) -> bool:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _build_items(args: argparse.Namespace) -> list[CollectItem]:
    items: list[CollectItem] = []
    seen: set[str] = set()
    for candidate in _read_candidate_paths(args.candidates_list):
        video_name = _derive_video_name(candidate, args.video_suffix_to_strip)
        _require(video_name not in seen, f"Duplicate video_name: {video_name}")
        seen.add(video_name)

        filename = args.filename_template.format(video_name=video_name)
        items.append(
            CollectItem(
                video_name=video_name,
                source_h5=Path(args.source_root) / video_name / filename,
                collected_h5=Path(args.output_dir) / filename,
            )
        )
    return items


def _object_map(h5_file: h5py.File) -> dict[str, str]:
    objects = {"/": "group"}

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            objects[name] = "dataset"
        elif isinstance(obj, h5py.Group):
            objects[name] = "group"
        else:
            objects[name] = type(obj).__name__

    h5_file.visititems(visitor)
    return objects


def _get_object(h5_file: h5py.File, name: str) -> h5py.File | h5py.Group | h5py.Dataset:
    if name == "/":
        return h5_file
    return h5_file[name]


def _assert_attrs_equal(
    source: h5py.File | h5py.Group | h5py.Dataset,
    collected: h5py.File | h5py.Group | h5py.Dataset,
    *,
    object_name: str,
) -> None:
    source_keys = set(source.attrs.keys())
    collected_keys = set(collected.attrs.keys())
    _require(
        source_keys == collected_keys,
        f"Attribute keys changed at {object_name}: "
        f"missing={sorted(source_keys - collected_keys)}, "
        f"added={sorted(collected_keys - source_keys)}",
    )
    for key in sorted(source_keys):
        _require(
            _array_equal(source.attrs[key], collected.attrs[key]),
            f"Attribute value changed at {object_name}:{key}",
        )


def _assert_dataset_equal(
    source: h5py.Dataset,
    collected: h5py.Dataset,
    *,
    name: str,
) -> None:
    _require(
        source.shape == collected.shape,
        f"Dataset shape changed at {name}: {source.shape} != {collected.shape}",
    )
    _require(
        source.dtype == collected.dtype,
        f"Dataset dtype changed at {name}: {source.dtype} != {collected.dtype}",
    )
    _require(
        _array_equal(source[...], collected[...]),
        f"Dataset values or ordering changed at {name}",
    )

    properties = (
        "chunks",
        "compression",
        "compression_opts",
        "shuffle",
        "fletcher32",
        "scaleoffset",
        "fillvalue",
        "maxshape",
    )
    for property_name in properties:
        source_value = getattr(source, property_name)
        collected_value = getattr(collected, property_name)
        _require(
            _array_equal(source_value, collected_value),
            f"Dataset property {property_name} changed at {name}: "
            f"{source_value!r} != {collected_value!r}",
        )


def _assert_h5_equal(source_path: Path, collected_path: Path) -> tuple[int, int]:
    with h5py.File(source_path, "r") as source_file, h5py.File(
        collected_path, "r"
    ) as collected_file:
        source_objects = _object_map(source_file)
        collected_objects = _object_map(collected_file)
        _require(
            source_objects == collected_objects,
            "H5 object schema changed: "
            f"missing={sorted(set(source_objects) - set(collected_objects))}, "
            f"added={sorted(set(collected_objects) - set(source_objects))}",
        )

        for name, object_type in sorted(source_objects.items()):
            source_obj = _get_object(source_file, name)
            collected_obj = _get_object(collected_file, name)
            _assert_attrs_equal(source_obj, collected_obj, object_name=name)
            if object_type == "dataset":
                _assert_dataset_equal(source_obj, collected_obj, name=name)

        num_groups = sum(value == "group" for value in source_objects.values())
        num_datasets = sum(value == "dataset" for value in source_objects.values())
        return num_groups, num_datasets


def _read_required_counts(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as h5_file:
        _require("point_cloud/points" in h5_file, "Missing point_cloud/points")
        _require("annotation/point_label" in h5_file, "Missing annotation/point_label")
        points = h5_file["point_cloud/points"]
        labels = h5_file["annotation/point_label"][:]
        _require(
            labels.shape == (points.shape[0],),
            "Collected annotation label count does not match points",
        )
        return int(points.shape[0]), int(np.sum(labels == 1))


def check_one(item: CollectItem) -> CheckResult:
    _require(item.source_h5.is_file(), f"Source H5 not found: {item.source_h5}")
    _require(
        item.collected_h5.is_file(),
        f"Collected H5 not found: {item.collected_h5}",
    )

    source_size = item.source_h5.stat().st_size
    collected_size = item.collected_h5.stat().st_size
    _require(
        source_size == collected_size,
        f"File size changed: source={source_size}, collected={collected_size}",
    )
    source_hash = _sha256(item.source_h5)
    collected_hash = _sha256(item.collected_h5)
    _require(
        source_hash == collected_hash,
        f"SHA-256 changed: source={source_hash}, collected={collected_hash}",
    )

    num_groups, num_datasets = _assert_h5_equal(
        item.source_h5,
        item.collected_h5,
    )
    num_points, num_labeled_points = _read_required_counts(item.collected_h5)
    return CheckResult(
        video_name=item.video_name,
        file_size=source_size,
        sha256=source_hash,
        num_groups=num_groups,
        num_datasets=num_datasets,
        num_points=num_points,
        num_labeled_points=num_labeled_points,
    )


def _check_manifest(path: Path, items: list[CollectItem]) -> None:
    path = Path(path)
    _require(path.is_file(), f"Collect manifest not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    _require(len(rows) == len(items), f"Manifest rows={len(rows)}, expected={len(items)}")
    rows_by_destination = {
        Path(row.get("destination_path", "")).resolve(): row for row in rows
    }
    _require(
        len(rows_by_destination) == len(rows),
        "Manifest contains duplicate or empty destination paths",
    )

    accepted_statuses = {"copied", "skipped_exists"}
    for item in items:
        destination = item.collected_h5.resolve()
        _require(destination in rows_by_destination, f"Manifest row missing: {destination}")
        row = rows_by_destination[destination]
        _require(
            row.get("status") in accepted_statuses,
            f"Unexpected manifest status for {item.video_name}: {row.get('status')!r}",
        )
        _require(
            Path(row.get("source_path", "")).resolve() == item.source_h5.resolve(),
            f"Manifest source path differs for {item.video_name}",
        )
        _require(
            row.get("video_dir") == item.video_name,
            f"Manifest video_dir differs for {item.video_name}: {row.get('video_dir')!r}",
        )


def _check_output_file_set(output_dir: Path, items: list[CollectItem]) -> None:
    actual = {path.resolve() for path in Path(output_dir).glob("*.h5") if path.is_file()}
    expected = {item.collected_h5.resolve() for item in items}
    _require(
        actual == expected,
        "Collected H5 file set differs: "
        f"missing={sorted(str(path) for path in expected - actual)}, "
        f"unexpected={sorted(str(path) for path in actual - expected)}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Stage 4 check that collected annotated combined_v2 H5 "
            "files are byte-identical and schema-identical to their sources."
        )
    )
    parser.add_argument("--candidates_list", type=Path, default=DEFAULT_CANDIDATES_LIST)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest_csv", type=Path, default=None)
    parser.add_argument("--video_suffix_to_strip", default="_ts448_oym96_corr")
    parser.add_argument(
        "--filename_template",
        default=(
            "{video_name}_pointcloud_annotated_foreground_combined_v2.h5"
        ),
    )
    parser.add_argument("--expected_videos", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.manifest_csv is None:
        args.manifest_csv = Path(args.output_dir) / "manifest.csv"

    items = _build_items(args)
    _require(
        len(items) == args.expected_videos,
        f"Candidates={len(items)}, expected={args.expected_videos}",
    )
    _check_output_file_set(args.output_dir, items)

    print("Stage 4 collect schema propagation check (read-only)")
    print(f"candidates_list : {args.candidates_list}")
    print(f"source_root     : {args.source_root}")
    print(f"output_dir      : {args.output_dir}")
    print(f"manifest_csv    : {args.manifest_csv}")
    print(f"videos          : {len(items)}")

    results: list[CheckResult] = []
    failures: list[str] = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] checking {item.video_name}")
        try:
            result = check_one(item)
        except Exception as exc:
            failures.append(f"{item.video_name}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        print(
            "  [OK] "
            f"bytes={result.file_size}, sha256={result.sha256[:12]}..., "
            f"groups={result.num_groups}, datasets={result.num_datasets}, "
            f"points={result.num_points}, labeled_points={result.num_labeled_points}"
        )

    if not failures:
        try:
            _check_manifest(args.manifest_csv, items)
            print("[OK] collect manifest matches all source/destination pairs")
        except Exception as exc:
            failures.append(f"manifest: {type(exc).__name__}: {exc}")
            print(f"[FAIL] manifest: {type(exc).__name__}: {exc}")

    print("\nCollect-propagation summary")
    print(f"  videos_checked       : {len(results)}/{len(items)}")
    print(f"  bytes_preserved      : {sum(result.file_size for result in results)}")
    print(f"  groups_preserved     : {sum(result.num_groups for result in results)}")
    print(f"  datasets_preserved   : {sum(result.num_datasets for result in results)}")
    print(f"  points_preserved     : {sum(result.num_points for result in results)}")
    print(f"  annotation_positive  : {sum(result.num_labeled_points for result in results)}")
    print(f"  failures             : {len(failures)}")
    print("  files_written        : 0")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Stage 4 collect schema propagation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
