from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.datasets.pseudo3d_pointcloud_dataset import Pseudo3DPointCloudDataset
from stage5.utils.feature_normalization import normalize_uint8_feature
from stage5.utils.h5_io import (
    OPTIONAL_FRAME_LEVEL_KEYS,
    OPTIONAL_POINT_LEVEL_KEYS,
    load_pointcloud_for_inference,
    load_stage5_pointcloud_h5,
)


DEFAULT_INPUT_DIR = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722/"
    "collected_combined_v2_annotated_corr_strict"
)
DEFAULT_PATTERN = "*_pointcloud_annotated_foreground_combined_v2.h5"
KNOWN_SOURCE_FLAGS = np.uint16(1 | 2 | 4 | 8)


@dataclass(frozen=True)
class CheckResult:
    h5_path: Path
    num_points: int
    num_frames: int
    num_positive: int
    num_valid: int
    num_dataset_samples: int
    num_windows: int


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


def _assert_attrs_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    _require(
        set(actual) == set(expected),
        f"{label} keys differ: "
        f"missing={sorted(set(expected) - set(actual))}, "
        f"added={sorted(set(actual) - set(expected))}",
    )
    for key in expected:
        _require(
            _array_equal(actual[key], expected[key]),
            f"{label}/{key} value differs",
        )


def _assert_loaded_array(
    loaded: dict[str, Any],
    key: str,
    expected: np.ndarray,
    *,
    loader_name: str,
) -> None:
    _require(key in loaded, f"{loader_name} did not return {key}")
    actual = loaded[key]
    _require(
        actual.shape == expected.shape,
        f"{loader_name}/{key} shape: {actual.shape} != {expected.shape}",
    )
    _require(
        actual.dtype == expected.dtype,
        f"{loader_name}/{key} dtype: {actual.dtype} != {expected.dtype}",
    )
    _require(
        _array_equal(actual, expected),
        f"{loader_name}/{key} values or point order changed",
    )


def _check_loaders(path: Path) -> tuple[int, int, int, int]:
    training_data = load_stage5_pointcloud_h5(path)
    inference_data = load_pointcloud_for_inference(path)

    with h5py.File(path, "r") as h5_file:
        point_cloud = h5_file["point_cloud"]
        annotation = h5_file["annotation"]
        expected_training = {
            "points": point_cloud["points"][:].astype(np.float32),
            "intensity": point_cloud["intensity"][:],
            "alpha": point_cloud["alpha"][:],
            "confidence": point_cloud["confidence"][:].astype(np.float32),
            "frame_order": point_cloud["frame_order"][:],
            "pixel_xy": point_cloud["pixel_xy"][:].astype(np.float32),
            "point_label": annotation["point_label"][:].astype(np.int64),
            "valid_mask": annotation["valid_mask"][:].astype(bool),
        }
        for key in OPTIONAL_POINT_LEVEL_KEYS + OPTIONAL_FRAME_LEVEL_KEYS:
            _require(key in point_cloud, f"combined_v2 H5 is missing point_cloud/{key}")
            expected_training[key] = point_cloud[key][:]

        point_cloud_attrs = dict(point_cloud.attrs)
        annotation_attrs = dict(annotation.attrs)
        file_attrs = dict(h5_file.attrs)
        frame_annotation_attrs = dict(h5_file["frame_annotation"].attrs)

    for key, expected in expected_training.items():
        _assert_loaded_array(
            training_data,
            key,
            expected,
            loader_name="load_stage5_pointcloud_h5",
        )

    expected_inference = dict(expected_training)
    expected_inference["point_label"] = expected_training["point_label"].astype(np.int8)
    expected_inference["valid_mask"] = expected_training["valid_mask"].astype(bool)
    for key, expected in expected_inference.items():
        _assert_loaded_array(
            inference_data,
            key,
            expected,
            loader_name="load_pointcloud_for_inference",
        )

    _assert_attrs_equal(
        training_data["point_cloud_attrs"],
        point_cloud_attrs,
        label="training point_cloud_attrs",
    )
    _assert_attrs_equal(
        training_data["annotation_attrs"],
        annotation_attrs,
        label="training annotation_attrs",
    )
    _assert_attrs_equal(
        training_data["file_attrs"],
        file_attrs,
        label="training file_attrs",
    )
    _assert_attrs_equal(
        training_data["frame_annotation_attrs"],
        frame_annotation_attrs,
        label="training frame_annotation_attrs",
    )
    _assert_attrs_equal(
        inference_data["point_cloud_attrs"],
        point_cloud_attrs,
        label="inference point_cloud_attrs",
    )
    _assert_attrs_equal(
        inference_data["file_attrs"],
        file_attrs,
        label="inference file_attrs",
    )

    source_flags = training_data["source_flags"]
    sampling_confidence = training_data["sampling_confidence"]
    _require(source_flags.dtype == np.uint16, f"source_flags dtype: {source_flags.dtype}")
    _require(
        sampling_confidence.dtype == np.float32,
        f"sampling_confidence dtype: {sampling_confidence.dtype}",
    )
    unknown_bits = source_flags & np.uint16(0xFFFF ^ int(KNOWN_SOURCE_FLAGS))
    _require(not np.any(unknown_bits), "source_flags contains undefined bits")
    _require(np.all(source_flags != 0), "source_flags contains zero-source points")
    _require(
        np.isfinite(sampling_confidence).all(),
        "sampling_confidence contains NaN or Inf",
    )
    _require(
        np.all((sampling_confidence >= 0.0) & (sampling_confidence <= 1.0)),
        "sampling_confidence is outside [0, 1]",
    )

    num_points = int(expected_training["points"].shape[0])
    num_frames = int(expected_training["per_frame_counts"].shape[0])
    num_positive = int(np.sum(expected_training["point_label"] == 1))
    num_valid = int(expected_training["valid_mask"].sum())
    del training_data, inference_data, expected_training, expected_inference
    return num_points, num_frames, num_positive, num_valid


def _assert_tensor_equal(actual: torch.Tensor, expected: np.ndarray, label: str) -> None:
    actual_array = actual.detach().cpu().numpy()
    _require(
        actual_array.shape == expected.shape,
        f"Dataset {label} shape: {actual_array.shape} != {expected.shape}",
    )
    _require(
        _array_equal(actual_array, expected),
        f"Dataset {label} differs from the H5 source",
    )


def _check_full_dataset(path: Path) -> int:
    dataset = Pseudo3DPointCloudDataset(
        [path],
        num_points=None,
        features=("intensity", "confidence"),
        normalize_points=False,
        sampling_mode="video",
        seed=260722,
        cache_data=False,
    )
    _require(len(dataset) == 1, f"Expected one video sample, got {len(dataset)}")
    _require(dataset.num_feature_channels == 2, "Default Stage 5 feature count changed")
    sample = dataset[0]

    with h5py.File(path, "r") as h5_file:
        point_cloud = h5_file["point_cloud"]
        annotation = h5_file["annotation"]
        points = point_cloud["points"][:].astype(np.float32)
        intensity = point_cloud["intensity"][:]
        confidence = point_cloud["confidence"][:].astype(np.float32)
        expected_features = np.column_stack(
            [normalize_uint8_feature(intensity), confidence]
        ).astype(np.float32)
        labels = annotation["point_label"][:].astype(np.int64)
        valid_mask = annotation["valid_mask"][:].astype(bool)
        frame_order = point_cloud["frame_order"][:].astype(np.int64)

    point_indices = np.arange(points.shape[0], dtype=np.int64)
    _assert_tensor_equal(sample["points"], points, "points")
    _assert_tensor_equal(sample["features"], expected_features, "features")
    _assert_tensor_equal(sample["labels"], labels, "labels")
    _assert_tensor_equal(sample["valid_mask"], valid_mask, "valid_mask")
    _assert_tensor_equal(sample["frame_order"], frame_order, "frame_order")
    _assert_tensor_equal(sample["point_indices"], point_indices, "point_indices")
    _require(
        set(sample) == {
            "points",
            "features",
            "labels",
            "valid_mask",
            "frame_order",
            "point_indices",
            "window_start",
            "window_end",
            "meta",
        },
        "Optional sampling fields were unexpectedly forced into model inputs",
    )
    _require(
        Path(sample["meta"]["h5_path"]).resolve() == path.resolve(),
        "Dataset metadata H5 path differs",
    )
    return int(points.shape[0])


def _check_deterministic_sampling(path: Path, num_points: int) -> None:
    kwargs = {
        "num_points": num_points,
        "features": ("intensity", "confidence"),
        "normalize_points": False,
        "sampling_mode": "video",
        "seed": 260722,
        "cache_data": False,
    }
    first = Pseudo3DPointCloudDataset([path], **kwargs)[0]
    second = Pseudo3DPointCloudDataset([path], **kwargs)[0]
    for key in ("points", "features", "labels", "valid_mask", "frame_order", "point_indices"):
        _require(torch.equal(first[key], second[key]), f"Sampling is not deterministic: {key}")

    indices = first["point_indices"].detach().cpu().numpy().astype(np.int64)
    with h5py.File(path, "r") as h5_file:
        flags = h5_file["point_cloud/source_flags"][:]
        sampling_confidence = h5_file["point_cloud/sampling_confidence"][:]
    _require(flags[indices].shape == (num_points,), "source_flags cannot index sampled points")
    _require(
        sampling_confidence[indices].shape == (num_points,),
        "sampling_confidence cannot index sampled points",
    )


def _check_window_index(path: Path) -> int:
    dataset = Pseudo3DPointCloudDataset(
        [path],
        num_points=None,
        features=("intensity", "confidence"),
        normalize_points=False,
        window_mode="overlap",
        window_size_frames=20,
        window_stride_frames=10,
        include_tail_window=True,
        seed=260722,
    )
    _require(len(dataset) > 0, "Stage 5 overlap window index is empty")
    summaries = dataset.get_window_summaries()
    _require(len(summaries) == len(dataset), "Window summaries and dataset length differ")
    for summary in summaries:
        _require(int(summary["num_points"]) > 0, "Generated an empty Stage 5 window")
    return len(dataset)


def check_one(path: Path, deterministic_points: int) -> CheckResult:
    _require(path.is_file(), f"Input H5 not found: {path}")
    num_points, num_frames, num_positive, num_valid = _check_loaders(path)
    num_dataset_samples = _check_full_dataset(path)
    _require(num_dataset_samples == num_points, "Stage 5 dataset dropped or added points")
    sample_size = min(int(deterministic_points), num_points)
    _check_deterministic_sampling(path, sample_size)
    num_windows = _check_window_index(path)
    return CheckResult(
        h5_path=path,
        num_points=num_points,
        num_frames=num_frames,
        num_positive=num_positive,
        num_valid=num_valid,
        num_dataset_samples=num_dataset_samples,
        num_windows=num_windows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Stage 5 compatibility check for collected Stage 4 "
            "combined_v2 annotated H5 files."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--h5_pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--expected_videos", type=int, default=3)
    parser.add_argument("--deterministic_points", type=int, default=4096)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _require(args.input_dir.is_dir(), f"Input directory not found: {args.input_dir}")
    _require(args.deterministic_points > 0, "deterministic_points must be positive")
    paths = sorted(path for path in args.input_dir.glob(args.h5_pattern) if path.is_file())
    _require(
        len(paths) == args.expected_videos,
        f"Matched H5 files={len(paths)}, expected={args.expected_videos}",
    )

    print("Stage 4 -> Stage 5 combined_v2 loader compatibility check (read-only)")
    print(f"input_dir            : {args.input_dir}")
    print(f"h5_pattern           : {args.h5_pattern}")
    print(f"videos               : {len(paths)}")
    print(f"deterministic_points : {args.deterministic_points}")

    results: list[CheckResult] = []
    failures: list[str] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] checking {path.name}")
        try:
            result = check_one(path, args.deterministic_points)
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        print(
            "  [OK] "
            f"points={result.num_points}, frames={result.num_frames}, "
            f"valid={result.num_valid}, positive={result.num_positive}, "
            f"windows={result.num_windows}"
        )

    print("\nStage 5 compatibility summary")
    print(f"  videos_checked       : {len(results)}/{len(paths)}")
    print(f"  points_loaded        : {sum(result.num_points for result in results)}")
    print(f"  valid_points         : {sum(result.num_valid for result in results)}")
    print(f"  annotation_positive  : {sum(result.num_positive for result in results)}")
    print(f"  overlap_windows      : {sum(result.num_windows for result in results)}")
    print(f"  failures             : {len(failures)}")
    print("  files_written        : 0")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Stage 4 -> Stage 5 combined_v2 loader compatibility checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
