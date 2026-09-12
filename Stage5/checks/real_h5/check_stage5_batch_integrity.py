from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.utils.h5_io import read_path_list


CHECKED_FIELDS = (
    "points",
    "features",
    "labels",
    "valid_mask",
    "frame_order",
    "point_indices",
)


@dataclass(frozen=True)
class ExpectedWindow:
    window_id: int
    start: int
    end: int
    point_indices: np.ndarray


@dataclass
class SourceArrays:
    points: np.ndarray
    intensity: np.ndarray
    alpha: np.ndarray
    confidence: np.ndarray
    frame_order: np.ndarray
    pixel_xy: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    normalized_points: np.ndarray | None = None
    feature_cache: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)


class SourceCache:
    def __init__(self, paths: list[Path], max_items: int = 2) -> None:
        self.paths = paths
        self.max_items = max(int(max_items), 1)
        self._items: OrderedDict[int, SourceArrays] = OrderedDict()

    def get(self, h5_index: int) -> SourceArrays:
        if h5_index in self._items:
            value = self._items.pop(h5_index)
            self._items[h5_index] = value
            return value

        value = load_source_arrays(self.paths[h5_index])
        self._items[h5_index] = value
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def combined_sha256(field_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for field in CHECKED_FIELDS:
        digest.update(field.encode("ascii"))
        digest.update(field_hashes[field].encode("ascii"))
    return digest.hexdigest()


def assert_array_equal(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    require(actual.shape == expected.shape, f"{context} shape: {actual.shape} != {expected.shape}")
    require(actual.dtype == expected.dtype, f"{context} dtype: {actual.dtype} != {expected.dtype}")
    if np.issubdtype(actual.dtype, np.floating):
        equal = np.array_equal(actual, expected, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    require(equal, f"{context} values or point order differ")


def load_source_arrays(path: Path) -> SourceArrays:
    with h5py.File(path, "r") as handle:
        require("point_cloud" in handle, f"point_cloud group missing: {path}")
        require("annotation" in handle, f"annotation group missing: {path}")
        point_cloud = handle["point_cloud"]
        annotation = handle["annotation"]
        required_point_cloud = (
            "points",
            "intensity",
            "alpha",
            "confidence",
            "frame_order",
            "pixel_xy",
        )
        for key in required_point_cloud:
            require(key in point_cloud, f"point_cloud/{key} missing: {path}")
        for key in ("point_label", "valid_mask"):
            require(key in annotation, f"annotation/{key} missing: {path}")

        source = SourceArrays(
            points=point_cloud["points"][:].astype(np.float32),
            intensity=point_cloud["intensity"][:],
            alpha=point_cloud["alpha"][:],
            confidence=point_cloud["confidence"][:].astype(np.float32),
            frame_order=point_cloud["frame_order"][:],
            pixel_xy=point_cloud["pixel_xy"][:].astype(np.float32),
            labels=annotation["point_label"][:].astype(np.int64),
            valid_mask=annotation["valid_mask"][:].astype(bool),
        )

    num_points = int(source.points.shape[0])
    require(source.points.shape == (num_points, 3), f"Invalid points shape in {path}")
    for name in ("intensity", "alpha", "confidence", "frame_order", "labels", "valid_mask"):
        value = getattr(source, name)
        require(value.shape == (num_points,), f"Length mismatch for {name} in {path}: {value.shape}")
    require(source.pixel_xy.shape == (num_points, 2), f"Invalid pixel_xy shape in {path}")
    return source


def normalize_xyz_independent(points: np.ndarray) -> np.ndarray:
    values = points.astype(np.float32)
    if values.size == 0:
        return values
    center = values.mean(axis=0, keepdims=True)
    centered = values - center
    scale = np.linalg.norm(centered, axis=1).max()
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return centered / float(scale)


def build_expected_features(source: SourceArrays, features: tuple[str, ...]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for name in features:
        if name == "intensity":
            arrays.append(source.intensity.astype(np.float32)[:, None] / 255.0)
        elif name == "confidence":
            arrays.append(source.confidence.astype(np.float32)[:, None])
        elif name == "alpha":
            arrays.append(source.alpha.astype(np.float32)[:, None] / 255.0)
        elif name == "frame_order":
            arrays.append(source.frame_order.astype(np.float32)[:, None])
        elif name == "normalized_frame_order":
            values = source.frame_order.astype(np.float32)
            denominator = max(float(values.max() - values.min()), 1.0)
            arrays.append(((values - float(values.min())) / denominator)[:, None])
        elif name == "pixel_xy":
            values = source.pixel_xy.astype(np.float32)
            max_xy = np.maximum(values.max(axis=0), 1.0)
            arrays.append(values / max_xy)
        else:
            raise ValueError(f"Unsupported feature in integrity check: {name}")
    if not arrays:
        return np.zeros((source.points.shape[0], 0), dtype=np.float32)
    return np.concatenate(arrays, axis=1).astype(np.float32)


def generate_expected_windows(
    frame_order: np.ndarray,
    *,
    window_size: int,
    window_stride: int,
    include_tail: bool,
) -> list[ExpectedWindow]:
    require(frame_order.ndim == 1 and frame_order.size > 0, "frame_order must be non-empty and 1D")
    minimum = int(frame_order.min())
    maximum = int(frame_order.max())
    generated: list[ExpectedWindow] = []
    start = minimum
    window_id = 0
    while start <= maximum:
        end = min(start + window_size - 1, maximum)
        if end - start + 1 < window_size and window_id > 0 and not include_tail:
            break
        indices = np.flatnonzero((frame_order >= start) & (frame_order <= end)).astype(np.int64)
        if indices.size:
            generated.append(ExpectedWindow(window_id, int(start), int(end), indices))
        if end >= maximum:
            break
        start += window_stride
        window_id += 1
    return generated


def parse_features(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        features = tuple(token.strip() for token in value.split(",") if token.strip())
    else:
        features = tuple(str(token).strip() for token in value if str(token).strip())
    require(bool(features), "At least one feature is required")
    return features


def resolve_setting(args_value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    if args_value is not None:
        return args_value
    return config.get(key, default)


def read_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    require(path.is_file(), f"Run config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    require(isinstance(config, dict), f"Run config must contain a JSON object: {path}")
    return config


def validate_dataset_index(
    dataset: Pseudo3DPointCloudDataset,
    *,
    window_size: int,
    window_stride: int,
    include_tail: bool,
) -> dict[tuple[int, int], int]:
    expected_keys: list[tuple[int, int, int, int]] = []
    for h5_index, path in enumerate(dataset.h5_paths):
        with h5py.File(path, "r") as handle:
            frame_order = handle["point_cloud/frame_order"][:]
        windows = generate_expected_windows(
            frame_order,
            window_size=window_size,
            window_stride=window_stride,
            include_tail=include_tail,
        )
        for window in windows:
            expected_keys.append((h5_index, window.window_id, window.start, window.end))

    require(len(dataset.samples) == len(expected_keys), "Dataset sample count differs from independent window index")
    key_to_index: dict[tuple[int, int], int] = {}
    for dataset_index, (sample_info, expected) in enumerate(zip(dataset.samples, expected_keys)):
        h5_index, window_id, start, end = expected
        window = sample_info["window"]
        require(int(sample_info["h5_index"]) == h5_index, f"sample[{dataset_index}] h5_index differs")
        require(window is not None, f"sample[{dataset_index}] has no overlap window")
        require(int(window.window_id) == window_id, f"sample[{dataset_index}] window_id differs")
        require(int(window.start_frame) == start, f"sample[{dataset_index}] window_start differs")
        require(int(window.end_frame) == end, f"sample[{dataset_index}] window_end differs")
        key = (h5_index, window_id)
        require(key not in key_to_index, f"Duplicate Dataset key: {key}")
        key_to_index[key] = dataset_index
    return key_to_index


def expected_item_arrays(
    source: SourceArrays,
    *,
    window_start: int,
    window_end: int,
    features: tuple[str, ...],
    normalize_points: bool,
) -> dict[str, np.ndarray]:
    point_indices = np.flatnonzero(
        (source.frame_order >= window_start) & (source.frame_order <= window_end)
    ).astype(np.int64)
    if normalize_points:
        if source.normalized_points is None:
            source.normalized_points = normalize_xyz_independent(source.points)
        all_points = source.normalized_points
    else:
        all_points = source.points
    if features not in source.feature_cache:
        source.feature_cache[features] = build_expected_features(source, features)
    all_features = source.feature_cache[features]
    return {
        "points": all_points[point_indices].astype(np.float32),
        "features": all_features[point_indices].astype(np.float32),
        "labels": source.labels[point_indices].astype(np.int64),
        "valid_mask": source.valid_mask[point_indices].astype(bool),
        "frame_order": source.frame_order[point_indices].astype(np.int64),
        "point_indices": point_indices,
    }


def validate_padding(batch: dict[str, Any], item_id: int, num_points: int, context: str) -> None:
    total_points = int(batch["points"].shape[1])
    require(0 < num_points <= total_points, f"{context} invalid num_points={num_points}")
    fill_values: dict[str, Any] = {
        "points": 0.0,
        "features": 0.0,
        "labels": -1,
        "valid_mask": False,
        "frame_order": -1,
        "point_indices": -1,
    }
    for field, fill_value in fill_values.items():
        padding = tensor_numpy(batch[field][item_id, num_points:])
        if padding.size:
            require(np.all(padding == fill_value), f"{context} non-canonical padding in {field}")


def validate_batch_item_against_source(
    batch: dict[str, Any],
    *,
    item_id: int,
    split: str,
    key_to_index: dict[tuple[int, int], int],
    paths: list[Path],
    aliases: dict[int, str],
    source_cache: SourceCache,
    features: tuple[str, ...],
    normalize_points: bool,
) -> dict[str, Any]:
    meta = batch["meta"][item_id]
    h5_index = int(meta["h5_index"])
    window_id = int(meta["window_id"])
    key = (h5_index, window_id)
    require(key in key_to_index, f"{split}: unexpected batch key {key}")
    require(0 <= h5_index < len(paths), f"{split}: h5_index out of range: {h5_index}")
    require(
        Path(meta["h5_path"]).resolve() == paths[h5_index].resolve(),
        f"{split}: meta h5_path does not match h5_index={h5_index}",
    )

    num_points = int(batch["num_points"][item_id].item())
    window_start = int(batch["window_start"][item_id].item())
    window_end = int(batch["window_end"][item_id].item())
    source = source_cache.get(h5_index)
    expected = expected_item_arrays(
        source,
        window_start=window_start,
        window_end=window_end,
        features=features,
        normalize_points=normalize_points,
    )
    require(num_points == int(expected["point_indices"].size), f"{split}/{key}: num_points differs")
    require(int(meta["num_source_points"]) == source.points.shape[0], f"{split}/{key}: source count differs")
    sampled_indices = tensor_numpy(meta["sampled_indices"])
    assert_array_equal(sampled_indices, expected["point_indices"], f"{split}/{key}/meta.sampled_indices")

    field_hashes: dict[str, str] = {}
    for field in CHECKED_FIELDS:
        actual = tensor_numpy(batch[field][item_id, :num_points])
        assert_array_equal(actual, expected[field], f"{split}/{key}/{field}")
        field_hashes[field] = array_sha256(actual)

    point_indices = expected["point_indices"]
    require(np.unique(point_indices).size == point_indices.size, f"{split}/{key}: duplicate point_indices")
    require(point_indices.min() >= 0, f"{split}/{key}: negative source point index")
    require(point_indices.max() < source.points.shape[0], f"{split}/{key}: source point index overflow")
    frame_order = expected["frame_order"]
    require(
        np.all((frame_order >= window_start) & (frame_order <= window_end)),
        f"{split}/{key}: frame_order outside window",
    )
    validate_padding(batch, item_id, num_points, f"{split}/{key}")

    valid_mask = expected["valid_mask"]
    valid_labels = expected["labels"][valid_mask]
    return {
        "split": split,
        "dataset_index": key_to_index[key],
        "h5_alias": aliases[h5_index],
        "h5_index": h5_index,
        "window_id": window_id,
        "window_start": window_start,
        "window_end": window_end,
        "num_points": num_points,
        "num_valid": int(valid_mask.sum()),
        "num_positive": int(np.sum(valid_labels == 1)),
        "points_sha256": field_hashes["points"],
        "features_sha256": field_hashes["features"],
        "labels_sha256": field_hashes["labels"],
        "valid_mask_sha256": field_hashes["valid_mask"],
        "frame_order_sha256": field_hashes["frame_order"],
        "point_indices_sha256": field_hashes["point_indices"],
        "combined_sha256": combined_sha256(field_hashes),
    }


def batch_summary_row(pass_name: str, batch_id: int, batch: dict[str, Any]) -> dict[str, Any]:
    point_counts = tensor_numpy(batch["num_points"]).astype(np.int64)
    h5_indices = [int(meta["h5_index"]) for meta in batch["meta"]]
    total_slots = int(batch["points"].shape[0] * batch["points"].shape[1])
    real_points = int(point_counts.sum())
    return {
        "pass": pass_name,
        "batch_id": batch_id,
        "batch_size": int(point_counts.size),
        "max_points": int(point_counts.max()),
        "min_points": int(point_counts.min()),
        "real_points": real_points,
        "padding_points": total_slots - real_points,
        "padding_ratio": float((total_slots - real_points) / total_slots),
        "num_h5": len(set(h5_indices)),
        "mixed_h5": int(len(set(h5_indices)) > 1),
        "unequal_points": int(np.unique(point_counts).size > 1),
    }


def make_loader(
    dataset: Pseudo3DPointCloudDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=pad_point_window_collate,
        generator=generator,
    )


def verify_hashed_batch(
    batch: dict[str, Any],
    *,
    split: str,
    expected_rows: dict[tuple[int, int], dict[str, Any]],
    seen: set[tuple[int, int]],
) -> None:
    for item_id, meta in enumerate(batch["meta"]):
        key = (int(meta["h5_index"]), int(meta["window_id"]))
        require(key in expected_rows, f"{split}: shuffled loader returned unknown key {key}")
        require(key not in seen, f"{split}: shuffled loader returned duplicate key {key}")
        seen.add(key)
        expected = expected_rows[key]
        num_points = int(batch["num_points"][item_id].item())
        require(num_points == int(expected["num_points"]), f"{split}/{key}: shuffled point count differs")
        for field in CHECKED_FIELDS:
            actual_hash = array_sha256(tensor_numpy(batch[field][item_id, :num_points]))
            require(
                actual_hash == expected[f"{field}_sha256"],
                f"{split}/{key}: shuffled/multi-worker hash differs for {field}",
            )
        validate_padding(batch, item_id, num_points, f"{split}/shuffled/{key}")


def manual_batch_cases(
    dataset: Pseudo3DPointCloudDataset,
    rows_by_key: dict[tuple[int, int], dict[str, Any]],
) -> list[tuple[str, list[int]]]:
    by_h5: dict[int, list[int]] = defaultdict(list)
    for key, row in rows_by_key.items():
        by_h5[key[0]].append(int(row["dataset_index"]))

    cases: list[tuple[str, list[int]]] = []
    same_h5 = next((indices[:2] for indices in by_h5.values() if len(indices) >= 2), None)
    if same_h5 is not None:
        cases.append(("manual_same_h5_overlap", same_h5))

    if len(by_h5) >= 2:
        first_two = list(sorted(by_h5))[:2]
        cases.append(("manual_different_h5", [by_h5[index][0] for index in first_two]))

    sorted_rows = sorted(rows_by_key.values(), key=lambda row: int(row["num_points"]))
    if len(sorted_rows) >= 2 and int(sorted_rows[0]["num_points"]) != int(sorted_rows[-1]["num_points"]):
        min_index = int(sorted_rows[0]["dataset_index"])
        max_index = int(sorted_rows[-1]["dataset_index"])
        if min_index != max_index:
            cases.append(("manual_min_max_points", [min_index, max_index]))
    return cases


def verify_manual_cases(
    dataset: Pseudo3DPointCloudDataset,
    *,
    split: str,
    rows_by_key: dict[tuple[int, int], dict[str, Any]],
    batch_rows: list[dict[str, Any]],
) -> list[str]:
    completed: list[str] = []
    for case_id, (case_name, indices) in enumerate(manual_batch_cases(dataset, rows_by_key)):
        samples = [dataset[index] for index in indices]
        if case_name == "manual_same_h5_overlap":
            first_indices = tensor_numpy(samples[0]["point_indices"])
            second_indices = tensor_numpy(samples[1]["point_indices"])
            require(
                np.intersect1d(first_indices, second_indices).size > 0,
                f"{split}: adjacent same-H5 windows do not overlap",
            )
        elif case_name == "manual_different_h5":
            require(
                int(samples[0]["meta"]["h5_index"]) != int(samples[1]["meta"]["h5_index"]),
                f"{split}: cross-H5 manual batch selected the same H5",
            )
        batch = pad_point_window_collate(samples)
        seen: set[tuple[int, int]] = set()
        verify_hashed_batch(batch, split=split, expected_rows=rows_by_key, seen=seen)
        batch_rows.append(batch_summary_row(case_name, case_id, batch))
        completed.append(case_name)
    return completed


def verify_raw_point_spot_checks(
    dataset: Pseudo3DPointCloudDataset,
    *,
    features: tuple[str, ...],
    num_checks: int,
) -> int:
    if num_checks <= 0 or not dataset:
        return 0
    raw_dataset = Pseudo3DPointCloudDataset(
        dataset.h5_paths,
        num_points=None,
        features=features,
        normalize_points=False,
        window_mode="overlap",
        window_size_frames=dataset.window_size_frames,
        window_stride_frames=dataset.window_stride_frames,
        include_tail_window=dataset.include_tail_window,
        cache_data=False,
    )
    candidate_indices = np.linspace(0, len(raw_dataset) - 1, min(num_checks, len(raw_dataset)), dtype=int)
    cache = SourceCache(raw_dataset.h5_paths)
    checked = 0
    for dataset_index in np.unique(candidate_indices):
        sample = raw_dataset[int(dataset_index)]
        meta = sample["meta"]
        h5_index = int(meta["h5_index"])
        source = cache.get(h5_index)
        indices = tensor_numpy(sample["point_indices"])
        assert_array_equal(
            tensor_numpy(sample["points"]),
            source.points[indices],
            f"raw spot check sample[{dataset_index}]",
        )
        checked += 1
    return checked


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_split(
    *,
    split: str,
    paths: list[Path],
    output_dir: Path,
    features: tuple[str, ...],
    normalize_points: bool,
    window_size: int,
    window_stride: int,
    include_tail: bool,
    batch_size: int,
    num_workers: int,
    seed: int,
    raw_point_checks: int,
) -> dict[str, Any]:
    dataset = Pseudo3DPointCloudDataset(
        paths,
        num_points=None,
        features=features,
        normalize_points=normalize_points,
        window_mode="overlap",
        window_size_frames=window_size,
        window_stride_frames=window_stride,
        include_tail_window=include_tail,
        cache_data=False,
    )
    key_to_index = validate_dataset_index(
        dataset,
        window_size=window_size,
        window_stride=window_stride,
        include_tail=include_tail,
    )
    aliases = {index: f"{split}_{index:03d}" for index in range(len(paths))}
    source_cache = SourceCache(paths)
    ordered_loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        seed=seed,
    )

    sample_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for batch_id, batch in enumerate(ordered_loader):
        if batch_id == 0 or (batch_id + 1) % 25 == 0 or batch_id + 1 == len(ordered_loader):
            print(f"  {split} ordered source check: batch {batch_id + 1}/{len(ordered_loader)}")
        batch_rows.append(batch_summary_row("ordered_source_check", batch_id, batch))
        for item_id in range(int(batch["points"].shape[0])):
            row = validate_batch_item_against_source(
                batch,
                item_id=item_id,
                split=split,
                key_to_index=key_to_index,
                paths=paths,
                aliases=aliases,
                source_cache=source_cache,
                features=features,
                normalize_points=normalize_points,
            )
            key = (int(row["h5_index"]), int(row["window_id"]))
            require(key not in rows_by_key, f"{split}: ordered loader duplicated {key}")
            rows_by_key[key] = row
            sample_rows.append(row)

    require(len(rows_by_key) == len(dataset), f"{split}: ordered loader omitted samples")

    shuffled_loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    shuffled_seen: set[tuple[int, int]] = set()
    for batch_id, batch in enumerate(shuffled_loader):
        if batch_id == 0 or (batch_id + 1) % 25 == 0 or batch_id + 1 == len(shuffled_loader):
            print(f"  {split} shuffled worker check: batch {batch_id + 1}/{len(shuffled_loader)}")
        batch_rows.append(batch_summary_row("shuffled_worker_check", batch_id, batch))
        verify_hashed_batch(
            batch,
            split=split,
            expected_rows=rows_by_key,
            seen=shuffled_seen,
        )
    require(shuffled_seen == set(rows_by_key), f"{split}: shuffled loader omitted samples")

    manual_cases = verify_manual_cases(
        dataset,
        split=split,
        rows_by_key=rows_by_key,
        batch_rows=batch_rows,
    )
    raw_checked = verify_raw_point_spot_checks(
        dataset,
        features=features,
        num_checks=raw_point_checks,
    )

    sample_rows.sort(key=lambda row: int(row["dataset_index"]))
    sample_fields = list(sample_rows[0])
    write_csv(output_dir / f"{split}_sample_integrity.csv", sample_rows, sample_fields)
    write_csv(output_dir / f"{split}_batch_integrity.csv", batch_rows, list(batch_rows[0]))

    total_real_points = sum(int(row["num_points"]) for row in sample_rows)
    total_valid = sum(int(row["num_valid"]) for row in sample_rows)
    total_positive = sum(int(row["num_positive"]) for row in sample_rows)
    return {
        "num_h5": len(paths),
        "num_samples": len(dataset),
        "num_ordered_batches": len(ordered_loader),
        "num_shuffled_batches": len(shuffled_loader),
        "total_window_point_occurrences": total_real_points,
        "total_valid_point_occurrences": total_valid,
        "total_positive_point_occurrences": total_positive,
        "raw_point_spot_checks": raw_checked,
        "manual_batch_cases": manual_cases,
        "ordered_source_check": "passed",
        "shuffled_worker_check": "passed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Stage5 real-H5 Dataset, overlap windows, collate, and DataLoader integrity"
    )
    parser.add_argument("--run_dir", required=True, help="Training run containing config.json and split lists")
    parser.add_argument("--output_dir", required=True, help="Directory for anonymous integrity CSV/JSON")
    parser.add_argument("--splits", default="train,val", help="Comma-separated split names: train,val")
    parser.add_argument("--features", default=None, help="Override config feature list")
    parser.add_argument("--window_size_frames", type=int, default=None)
    parser.add_argument("--window_stride_frames", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_files_per_split", type=int, default=0, help="0 checks every split file")
    parser.add_argument("--raw_point_checks", type=int, default=3, help="Raw-coordinate spot checks per split")
    parser.add_argument(
        "--normalize_points",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the run's xyz normalization setting",
    )
    parser.add_argument(
        "--include_tail_window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the run's tail-window setting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_run_config(run_dir)

    features = parse_features(resolve_setting(args.features, config, "features", "intensity,confidence"))
    window_size = int(resolve_setting(args.window_size_frames, config, "window_size_frames", 12))
    window_stride = int(resolve_setting(args.window_stride_frames, config, "window_stride_frames", 6))
    batch_size = int(resolve_setting(args.batch_size, config, "batch_size", 1))
    num_workers = int(resolve_setting(args.num_workers, config, "num_workers", 0))
    seed = int(resolve_setting(args.seed, config, "seed", 42))
    include_tail = bool(
        resolve_setting(args.include_tail_window, config, "include_tail_window", True)
    )
    normalize_points = bool(
        resolve_setting(args.normalize_points, config, "normalize_points", not config.get("no_normalize_points", False))
    )
    require(config.get("window_mode", "overlap") == "overlap", "Integrity check requires overlap window mode")
    require(window_size > 0 and window_stride > 0, "Window size and stride must be positive")
    require(batch_size > 0, "batch_size must be positive")
    require(num_workers >= 0, "num_workers must be non-negative")
    require(args.max_files_per_split >= 0, "max_files_per_split must be non-negative")

    requested_splits = tuple(token.strip() for token in args.splits.split(",") if token.strip())
    require(bool(requested_splits), "No split was requested")
    require(set(requested_splits) <= {"train", "val"}, f"Unsupported splits: {requested_splits}")

    summary: dict[str, Any] = {
        "status": "running",
        "settings": {
            "features": list(features),
            "normalize_points": normalize_points,
            "window_size_frames": window_size,
            "window_stride_frames": window_stride,
            "include_tail_window": include_tail,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "max_files_per_split": args.max_files_per_split,
        },
        "splits": {},
    }
    summary_path = output_dir / "integrity_summary.json"
    try:
        for split in requested_splits:
            list_path = run_dir / f"{split}_files.txt"
            if not list_path.is_file():
                if split == "val":
                    summary["splits"][split] = {"status": "skipped", "reason": "split list not found"}
                    continue
                raise FileNotFoundError(f"Split list not found: {list_path}")
            paths = read_path_list(list_path)
            if args.max_files_per_split > 0:
                paths = paths[: args.max_files_per_split]
            require(bool(paths), f"{split} split contains no H5 files")
            missing = [path for path in paths if not path.is_file()]
            require(not missing, f"{split} split contains missing H5 files; first={missing[0] if missing else ''}")

            print(f"Checking {split}: files={len(paths)}")
            result = check_split(
                split=split,
                paths=paths,
                output_dir=output_dir,
                features=features,
                normalize_points=normalize_points,
                window_size=window_size,
                window_stride=window_stride,
                include_tail=include_tail,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed + (100000 if split == "val" else 0),
                raw_point_checks=args.raw_point_checks,
            )
            summary["splits"][split] = {"status": "passed", **result}
            print(f"  {split} passed: samples={result['num_samples']}, files={result['num_h5']}")
        summary["status"] = "passed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        raise
    finally:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Stage5 real-H5 batch integrity check passed.")
    print(f"summary: {summary_path}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
