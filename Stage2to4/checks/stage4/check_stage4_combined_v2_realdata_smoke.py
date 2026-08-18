from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_CANDIDATES_LIST = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_smoke/candidates.txt"
)
DEFAULT_COMBINED_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_smoke/"
    "combined_v2_default_corr"
)
DEFAULT_LEGACY_ANNOTATED_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_visualizations/260711"
)
DEFAULT_VIDEO_SUFFIX = "_ts448_oym96_corr"

SOURCE_GLOBAL = np.uint16(1 << 0)
SOURCE_LOCAL_PERCENTILE = np.uint16(1 << 1)
SOURCE_TOPHAT = np.uint16(1 << 2)
SOURCE_CONTEXT_GRID = np.uint16(1 << 3)
ALL_SOURCE_FLAGS = np.uint16(
    SOURCE_GLOBAL
    | SOURCE_LOCAL_PERCENTILE
    | SOURCE_TOPHAT
    | SOURCE_CONTEXT_GRID
)

POINT_LEVEL_DTYPES = {
    "points": np.dtype(np.float32),
    "intensity": np.dtype(np.uint8),
    "alpha": np.dtype(np.uint8),
    "frame_index": np.dtype(np.int64),
    "frame_order": np.dtype(np.int32),
    "pixel_xy": np.dtype(np.float32),
    "confidence": np.dtype(np.float32),
    "source_type": np.dtype(np.uint8),
    "source_flags": np.dtype(np.uint16),
    "sampling_confidence": np.dtype(np.float32),
}

FRAME_LEVEL_KEYS = (
    "per_frame_counts",
    "per_frame_global_counts",
    "per_frame_local_percentile_counts",
    "per_frame_tophat_counts",
    "per_frame_context_grid_counts",
    "per_frame_context_only_counts",
    "per_frame_evidence_counts",
    "per_frame_overlap_counts",
)


@dataclass(frozen=True)
class PointCloudFile:
    path: Path
    arrays: dict[str, np.ndarray]
    attrs: dict[str, Any]
    compression: dict[str, str | None]


@dataclass(frozen=True)
class VideoReport:
    video_name: str
    num_frames: int
    legacy_points: int
    combined_points: int
    multiplier: float
    valid_bbox_frames: int
    target_frames: int
    recovered_frames: int
    recovered_bbox_min: int
    recovered_bbox_median: float
    recovered_bbox_max: int
    preexisting_bbox_frames: int
    preexisting_min_added: int | None
    bbox_global_points: int
    bbox_local_points: int
    bbox_tophat_points: int
    bbox_context_points: int
    bbox_context_only_points: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _attr_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _read_candidates(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidates list not found: {path}")
    candidates = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            candidates.append(Path(value))
    if not candidates:
        raise ValueError(f"Candidates list is empty: {path}")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidates list contains duplicate paths")
    for candidate in candidates:
        if not candidate.is_file():
            raise FileNotFoundError(f"Candidate pseudo3D H5 not found: {candidate}")
    return candidates


def _video_name(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _find_unique(root: Path, filename: str) -> Path:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Search root not found: {root}")
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} below {root}")
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {filename!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _load_point_cloud(path: Path) -> PointCloudFile:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Point-cloud H5 not found: {path}")
    with h5py.File(path, "r") as handle:
        if "point_cloud" not in handle:
            raise KeyError(f"point_cloud group missing: {path}")
        group = handle["point_cloud"]
        arrays = {key: group[key][:] for key in group.keys()}
        attrs = dict(group.attrs)
        compression = {key: group[key].compression for key in group.keys()}
    return PointCloudFile(
        path=path,
        arrays=arrays,
        attrs=attrs,
        compression=compression,
    )


def _load_input_geometry(
    path: Path,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        for key in (
            "local_encoder_images",
            "frame_indices",
            "pixel_to_image",
            "pred_tracking_corr",
        ):
            if key not in handle:
                raise KeyError(f"{key} missing in pseudo3D H5: {path}")
        images = handle["local_encoder_images"]
        frame_indices = handle["frame_indices"][:].astype(np.int64)
        # Match the exporter's explicit float32 input conversion.
        pixel_to_image = handle["pixel_to_image"][:].astype(np.float32)
        tracking = handle["pred_tracking_corr"][:].astype(np.float32)
        _require(images.shape[0] == frame_indices.shape[0], "Input frame mismatch")
        frame_shape = tuple(int(value) for value in images.shape[1:])

    if len(frame_shape) == 2:
        height, width = frame_shape
    elif len(frame_shape) == 3:
        if frame_shape[0] in {1, 3, 4}:
            height, width = frame_shape[1:]
        elif frame_shape[-1] in {1, 3, 4}:
            height, width = frame_shape[:2]
        else:
            raise ValueError(f"Unsupported local frame layout: {frame_shape}")
    else:
        raise ValueError(f"Unsupported local frame layout: {frame_shape}")
    _require(height > 0 and width > 0, f"Invalid image size: {(height, width)}")
    _require(pixel_to_image.shape == (4, 4), "Invalid pixel_to_image shape")
    _require(
        tracking.shape == (frame_indices.size, 4, 4),
        "Invalid corrected tracking shape",
    )
    _require(np.all(np.isfinite(pixel_to_image)), "Non-finite pixel_to_image")
    _require(np.all(np.isfinite(tracking)), "Non-finite corrected tracking")
    return frame_indices, height, width, tracking, pixel_to_image


def _point_keys(
    frame_order: np.ndarray,
    pixel_xy: np.ndarray,
    *,
    num_frames: int,
    height: int,
    width: int,
    label: str,
) -> np.ndarray:
    frame_order = np.asarray(frame_order)
    pixel_xy = np.asarray(pixel_xy)
    _require(frame_order.ndim == 1, f"{label}: frame_order must be 1D")
    _require(
        pixel_xy.shape == (frame_order.shape[0], 2),
        f"{label}: invalid pixel_xy shape {pixel_xy.shape}",
    )
    _require(np.all(np.isfinite(pixel_xy)), f"{label}: non-finite pixel_xy")
    rounded = np.rint(pixel_xy).astype(np.int64)
    _require(
        np.allclose(pixel_xy, rounded, rtol=0.0, atol=1e-6),
        f"{label}: non-integral pixel coordinates",
    )
    orders = frame_order.astype(np.int64)
    xs = rounded[:, 0]
    ys = rounded[:, 1]
    _require(
        np.all((orders >= 0) & (orders < num_frames)),
        f"{label}: frame_order outside [0, {num_frames})",
    )
    _require(
        np.all((xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)),
        f"{label}: pixel coordinates outside {width}x{height}",
    )
    keys = orders * (height * width) + ys * width + xs
    _require(np.unique(keys).size == keys.size, f"{label}: duplicate pixels")
    return keys


def _source_counts(flags: np.ndarray) -> dict[str, int]:
    flags = np.asarray(flags, dtype=np.uint16)
    evidence = (
        flags & (SOURCE_GLOBAL | SOURCE_LOCAL_PERCENTILE | SOURCE_TOPHAT)
    ) != 0
    source_count = (
        ((flags & SOURCE_GLOBAL) != 0).astype(np.uint8)
        + ((flags & SOURCE_LOCAL_PERCENTILE) != 0).astype(np.uint8)
        + ((flags & SOURCE_TOPHAT) != 0).astype(np.uint8)
        + ((flags & SOURCE_CONTEXT_GRID) != 0).astype(np.uint8)
    )
    return {
        "total": int(flags.size),
        "global": int(np.count_nonzero((flags & SOURCE_GLOBAL) != 0)),
        "local": int(np.count_nonzero((flags & SOURCE_LOCAL_PERCENTILE) != 0)),
        "tophat": int(np.count_nonzero((flags & SOURCE_TOPHAT) != 0)),
        "context": int(np.count_nonzero((flags & SOURCE_CONTEXT_GRID) != 0)),
        "context_only": int(
            np.count_nonzero(((flags & SOURCE_CONTEXT_GRID) != 0) & ~evidence)
        ),
        "evidence": int(np.count_nonzero(evidence)),
        "overlap": int(np.count_nonzero(source_count > 1)),
    }


def _expected_source_type(flags: np.ndarray) -> np.ndarray:
    flags = np.asarray(flags, dtype=np.uint16)
    result = np.zeros(flags.shape, dtype=np.uint8)
    result[(flags & SOURCE_CONTEXT_GRID) != 0] = 4
    result[(flags & SOURCE_TOPHAT) != 0] = 3
    result[(flags & SOURCE_LOCAL_PERCENTILE) != 0] = 2
    result[(flags & SOURCE_GLOBAL) != 0] = 1
    return result


def _assert_attr_equal(attrs: dict[str, Any], key: str, expected: Any) -> None:
    _require(key in attrs, f"Combined metadata missing: {key}")
    actual = attrs[key]
    if isinstance(expected, str):
        _require(_attr_text(actual) == expected, f"Metadata mismatch: {key}")
    elif isinstance(expected, bool):
        _require(bool(actual) is expected, f"Metadata mismatch: {key}")
    elif isinstance(expected, float):
        _require(
            bool(np.isclose(float(actual), expected)),
            f"Metadata mismatch: {key}={actual!r}, expected={expected!r}",
        )
    else:
        _require(int(actual) == expected, f"Metadata mismatch: {key}")


def _validate_combined_schema(
    point_cloud: PointCloudFile,
    *,
    source_h5: Path,
    frame_indices: np.ndarray,
    height: int,
    width: int,
    tracking: np.ndarray,
    pixel_to_image: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    arrays = point_cloud.arrays
    attrs = point_cloud.attrs
    for key, expected_dtype in POINT_LEVEL_DTYPES.items():
        _require(key in arrays, f"Combined dataset missing: point_cloud/{key}")
        _require(
            arrays[key].dtype == expected_dtype,
            f"Combined dtype mismatch: {key}={arrays[key].dtype}",
        )
        _require(
            point_cloud.compression[key] == "gzip",
            f"Combined dataset is not gzip-compressed: {key}",
        )
    for key in FRAME_LEVEL_KEYS:
        _require(key in arrays, f"Combined dataset missing: point_cloud/{key}")
        _require(
            arrays[key].dtype == np.dtype(np.int64),
            f"Combined dtype mismatch: {key}={arrays[key].dtype}",
        )
        _require(
            point_cloud.compression[key] == "gzip",
            f"Combined dataset is not gzip-compressed: {key}",
        )

    num_points = int(arrays["points"].shape[0])
    num_frames = int(frame_indices.shape[0])
    for key in POINT_LEVEL_DTYPES:
        _require(
            arrays[key].shape[0] == num_points,
            f"Combined point-level length mismatch: {key}",
        )
    _require(arrays["points"].shape == (num_points, 3), "Invalid points shape")
    _require(arrays["pixel_xy"].shape == (num_points, 2), "Invalid pixel shape")
    for key in FRAME_LEVEL_KEYS:
        _require(
            arrays[key].shape == (num_frames,),
            f"Combined frame-level shape mismatch: {key}",
        )
        _require(np.all(arrays[key] >= 0), f"Negative frame counts: {key}")

    for key in ("points", "confidence", "sampling_confidence"):
        _require(np.all(np.isfinite(arrays[key])), f"Non-finite combined {key}")
    for key in ("confidence", "sampling_confidence"):
        _require(
            np.all((arrays[key] >= 0.0) & (arrays[key] <= 1.0)),
            f"Combined {key} outside [0,1]",
        )

    keys = _point_keys(
        arrays["frame_order"],
        arrays["pixel_xy"],
        num_frames=num_frames,
        height=height,
        width=width,
        label="combined_v2",
    )
    orders = arrays["frame_order"].astype(np.int64)
    recomputed_counts = np.bincount(orders, minlength=num_frames)
    _require(
        np.array_equal(recomputed_counts, arrays["per_frame_counts"]),
        "Combined per_frame_counts mismatch",
    )
    _require(int(recomputed_counts.sum()) == num_points, "Combined count sum mismatch")
    _require(
        np.array_equal(arrays["frame_index"], frame_indices[orders]),
        "Combined frame_index does not match input frame_indices",
    )

    for order in range(num_frames):
        frame_mask = orders == order
        xy = arrays["pixel_xy"][frame_mask].astype(np.float32)
        pixel_h = np.column_stack(
            (
                xy,
                np.zeros((xy.shape[0],), dtype=np.float32),
                np.ones((xy.shape[0],), dtype=np.float32),
            )
        )
        expected_points = (
            tracking[order] @ (pixel_to_image @ pixel_h.T)
        ).T[:, :3]
        _require(
            np.allclose(
                arrays["points"][frame_mask],
                expected_points,
                rtol=1e-5,
                atol=1e-4,
            ),
            f"Independent corrected projection mismatch at frame_order={order}",
        )

    flags = arrays["source_flags"]
    _require(np.all(flags != 0), "Combined source_flags contains zero")
    _require(
        np.all((flags & np.uint16(~ALL_SOURCE_FLAGS)) == 0),
        "Combined source_flags contains unknown bits",
    )
    _require(
        np.array_equal(arrays["source_type"], _expected_source_type(flags)),
        "Combined source_type priority mismatch",
    )

    stat_datasets = {
        "total": "per_frame_counts",
        "global": "per_frame_global_counts",
        "local": "per_frame_local_percentile_counts",
        "tophat": "per_frame_tophat_counts",
        "context": "per_frame_context_grid_counts",
        "context_only": "per_frame_context_only_counts",
        "evidence": "per_frame_evidence_counts",
        "overlap": "per_frame_overlap_counts",
    }
    for order in range(num_frames):
        frame_stats = _source_counts(flags[orders == order])
        for stat_key, dataset_key in stat_datasets.items():
            _require(
                int(arrays[dataset_key][order]) == frame_stats[stat_key],
                f"{dataset_key} mismatch at frame_order={order}",
            )

    source_stats = _source_counts(flags)
    attr_names = {
        "total": "total_selected_points",
        "global": "global_evidence_points",
        "local": "local_percentile_points",
        "tophat": "tophat_points",
        "context": "context_grid_points",
        "context_only": "context_only_points",
        "evidence": "evidence_points",
        "overlap": "overlap_points",
    }
    for stat_key, attr_key in attr_names.items():
        _require(attr_key in attrs, f"Combined metadata missing: {attr_key}")
        _require(
            int(attrs[attr_key]) == source_stats[stat_key],
            f"Combined aggregate source statistic mismatch: {attr_key}",
        )

    expected_attrs = {
        "sampling_mode": "combined_v2",
        "point_mode": "foreground",
        "geometry_key": "corr",
        "sample_stride": 2,
        "max_points_per_frame": 0,
        "random_seed": 0,
        "include_context_grid": True,
        "context_grid_stride": 4,
        "context_grid_phase": "origin",
        "enable_local_percentile": True,
        "local_window_size": 41,
        "local_percentile": 80.0,
        "local_min_contrast": 4.0,
        "enable_tophat": True,
        "tophat_kernel_size": 21,
        "tophat_percentile": 85.0,
        "tophat_min_response": 2.0,
        "tophat_morph_shape": "ellipse",
        "enable_evidence_cleanup": True,
        "evidence_open_ksize": 3,
        "evidence_close_ksize": 5,
        "evidence_morph_shape": "ellipse",
        "evidence_min_component_area": 100,
        "local_source_confidence": 0.7,
        "tophat_source_confidence": 0.7,
        "context_source_confidence": 0.0,
        "num_frames": num_frames,
        "num_points": num_points,
    }
    for key, expected in expected_attrs.items():
        _assert_attr_equal(attrs, key, expected)
    _require(
        Path(_attr_text(attrs.get("source_h5"))) == source_h5,
        "Combined source_h5 metadata mismatch",
    )
    return keys, source_stats


def _validate_legacy_and_inclusion(
    legacy: PointCloudFile,
    combined: PointCloudFile,
    *,
    combined_keys: np.ndarray,
    frame_indices: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    required = (
        "points",
        "intensity",
        "alpha",
        "frame_index",
        "frame_order",
        "pixel_xy",
        "confidence",
    )
    for key in required:
        _require(key in legacy.arrays, f"Legacy dataset missing: {key}")
    _require(
        _attr_text(legacy.attrs.get("sampling_mode"), "legacy") == "legacy",
        "Expected a legacy annotated point cloud",
    )
    num_frames = int(frame_indices.shape[0])
    legacy_keys = _point_keys(
        legacy.arrays["frame_order"],
        legacy.arrays["pixel_xy"],
        num_frames=num_frames,
        height=height,
        width=width,
        label="legacy",
    )
    legacy_orders = legacy.arrays["frame_order"].astype(np.int64)
    _require(
        np.array_equal(legacy.arrays["frame_index"], frame_indices[legacy_orders]),
        "Legacy frame_index does not match input frame_indices",
    )

    combined_sort = np.argsort(combined_keys)
    sorted_combined_keys = combined_keys[combined_sort]
    positions = np.searchsorted(sorted_combined_keys, legacy_keys)
    included = positions < sorted_combined_keys.size
    included[included] &= (
        sorted_combined_keys[positions[included]] == legacy_keys[included]
    )
    _require(np.all(included), "One or more legacy pixels are absent in combined_v2")
    matched = combined_sort[positions]

    for key in ("intensity", "alpha", "frame_index"):
        _require(
            np.array_equal(legacy.arrays[key], combined.arrays[key][matched]),
            f"Legacy/combined mismatch for retained {key}",
        )
    for key in ("points", "confidence"):
        _require(
            np.allclose(
                legacy.arrays[key],
                combined.arrays[key][matched],
                rtol=1e-6,
                atol=1e-5,
            ),
            f"Legacy/combined mismatch for retained {key}",
        )
    _require(
        np.all(
            (combined.arrays["source_flags"][matched] & SOURCE_GLOBAL) != 0
        ),
        "A retained legacy pixel is missing the global source flag",
    )
    global_count = int(
        np.count_nonzero(
            (combined.arrays["source_flags"] & SOURCE_GLOBAL) != 0
        )
    )
    _require(
        global_count == legacy_keys.size,
        f"Global source count={global_count} != legacy points={legacy_keys.size}",
    )
    return legacy_keys


def _inside_bbox_union(pixel_xy: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inside = np.zeros((pixel_xy.shape[0],), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        inside |= (
            (pixel_xy[:, 0] >= x1)
            & (pixel_xy[:, 0] <= x2)
            & (pixel_xy[:, 1] >= y1)
            & (pixel_xy[:, 1] <= y2)
        )
    return inside


def _load_bboxes(annotated_h5: Path, num_frames: int) -> dict[int, np.ndarray]:
    with h5py.File(annotated_h5, "r") as handle:
        if "frame_annotation" not in handle:
            raise KeyError("Legacy annotated H5 has no frame_annotation group")
        group = handle["frame_annotation"]
        for key in ("frame_order", "bbox_local_xyxy"):
            if key not in group:
                raise KeyError(f"frame_annotation/{key} missing")
        orders = group["frame_order"][:].astype(np.int64)
        boxes = group["bbox_local_xyxy"][:].astype(np.float64)
    _require(boxes.shape == (orders.size, 4), "Invalid BBox array shape")
    _require(
        np.all((orders >= 0) & (orders < num_frames)),
        "BBox frame_order outside input range",
    )
    valid = (
        np.all(np.isfinite(boxes), axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    result = {}
    for order in np.unique(orders[valid]):
        result[int(order)] = boxes[(orders == order) & valid]
    return result


def _check_ply_vertex_count(path: Path, expected: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Combined PLY not found: {path}")
    vertex_count = None
    with path.open(errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
            if line.strip() == "end_header":
                break
            if line_number > 100:
                raise ValueError(f"PLY header is unexpectedly long: {path}")
    _require(vertex_count == expected, "PLY/H5 vertex count mismatch")


def _inspect_video(
    source_h5: Path,
    *,
    video_name: str,
    combined_h5: Path,
    legacy_annotated_h5: Path,
) -> VideoReport:
    (
        frame_indices,
        height,
        width,
        tracking,
        pixel_to_image,
    ) = _load_input_geometry(source_h5)
    num_frames = int(frame_indices.size)
    combined = _load_point_cloud(combined_h5)
    legacy = _load_point_cloud(legacy_annotated_h5)

    combined_keys, source_stats = _validate_combined_schema(
        combined,
        source_h5=source_h5,
        frame_indices=frame_indices,
        height=height,
        width=width,
        tracking=tracking,
        pixel_to_image=pixel_to_image,
    )
    _validate_legacy_and_inclusion(
        legacy,
        combined,
        combined_keys=combined_keys,
        frame_indices=frame_indices,
        height=height,
        width=width,
    )

    legacy_orders = legacy.arrays["frame_order"].astype(np.int64)
    combined_orders = combined.arrays["frame_order"].astype(np.int64)
    legacy_frame_counts = np.bincount(legacy_orders, minlength=num_frames)
    bboxes = _load_bboxes(legacy_annotated_h5, num_frames)
    _require(bool(bboxes), "No geometrically valid BBoxes found")

    target_counts = []
    target_orders = []
    preexisting_added = []
    bbox_source_flags = []
    recovered = 0
    for order, boxes in sorted(bboxes.items()):
        legacy_frame = legacy_orders == order
        combined_frame = combined_orders == order
        legacy_inside = _inside_bbox_union(
            legacy.arrays["pixel_xy"][legacy_frame], boxes
        )
        combined_inside = _inside_bbox_union(
            combined.arrays["pixel_xy"][combined_frame], boxes
        )
        legacy_count = int(np.count_nonzero(legacy_inside))
        combined_count = int(np.count_nonzero(combined_inside))
        _require(
            combined_count >= legacy_count,
            f"BBox point loss at frame_order={order}",
        )
        if legacy_count == 0 and legacy_frame_counts[order] > 0:
            target_orders.append(order)
            target_counts.append(combined_count)
            if combined_count > 0:
                recovered += 1
            bbox_source_flags.append(
                combined.arrays["source_flags"][combined_frame][combined_inside]
            )
        elif legacy_count > 0:
            preexisting_added.append(combined_count - legacy_count)

    _require(bool(target_counts), "No primary zero-BBox target frames detected")
    unrecovered_orders = [
        order
        for order, count in zip(target_orders, target_counts, strict=True)
        if count == 0
    ]
    _require(
        recovered == len(target_counts),
        f"Recovered {recovered}/{len(target_counts)} target BBox frames; "
        f"unrecovered_orders={unrecovered_orders}",
    )
    target_array = np.asarray(target_counts, dtype=np.int64)
    target_flags = np.concatenate(bbox_source_flags)
    bbox_stats = _source_counts(target_flags)

    combined_points = int(combined.arrays["points"].shape[0])
    legacy_points = int(legacy.arrays["points"].shape[0])
    _require(legacy_points > 0, "Legacy point cloud is empty")
    multiplier = combined_points / legacy_points
    _require(
        np.isclose(
            float(combined.attrs["legacy_point_count_multiplier"]),
            multiplier,
        ),
        "legacy_point_count_multiplier metadata mismatch",
    )

    ply_path = combined_h5.with_suffix(".ply")
    _check_ply_vertex_count(ply_path, combined_points)
    _require(source_stats["global"] == legacy_points, "Legacy/global mismatch")

    return VideoReport(
        video_name=video_name,
        num_frames=num_frames,
        legacy_points=legacy_points,
        combined_points=combined_points,
        multiplier=multiplier,
        valid_bbox_frames=len(bboxes),
        target_frames=len(target_counts),
        recovered_frames=recovered,
        recovered_bbox_min=int(target_array.min()),
        recovered_bbox_median=float(np.median(target_array)),
        recovered_bbox_max=int(target_array.max()),
        preexisting_bbox_frames=len(preexisting_added),
        preexisting_min_added=(
            min(preexisting_added) if preexisting_added else None
        ),
        bbox_global_points=bbox_stats["global"],
        bbox_local_points=bbox_stats["local"],
        bbox_tophat_points=bbox_stats["tophat"],
        bbox_context_points=bbox_stats["context"],
        bbox_context_only_points=bbox_stats["context_only"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Stage 4 combined_v2 real-data smoke check against legacy "
            "annotated point clouds and their local BBoxes."
        )
    )
    parser.add_argument(
        "--candidates_list", type=Path, default=DEFAULT_CANDIDATES_LIST
    )
    parser.add_argument("--combined_root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument(
        "--legacy_annotated_root",
        type=Path,
        default=DEFAULT_LEGACY_ANNOTATED_ROOT,
    )
    parser.add_argument("--video_suffix_to_strip", default=DEFAULT_VIDEO_SUFFIX)
    parser.add_argument(
        "--combined_h5_template",
        default=(
            "{video_name}/{video_name}_pointcloud_foreground_combined_v2.h5"
        ),
    )
    parser.add_argument(
        "--legacy_annotated_filename_template",
        default="{video_name}_pointcloud_annotated_foreground.h5",
    )
    parser.add_argument("--expected_videos", type=int, default=3)
    parser.add_argument("--expected_target_frames", type=int, default=51)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.expected_videos < 0:
        raise ValueError("expected_videos must be >= 0")
    if args.expected_target_frames < 0:
        raise ValueError("expected_target_frames must be >= 0")
    context = {"video_name": "example"}
    for name in (
        "combined_h5_template",
        "legacy_annotated_filename_template",
    ):
        try:
            getattr(args, name).format(**context)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid {name}: {getattr(args, name)!r}") from exc


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    candidates = _read_candidates(args.candidates_list)
    if args.expected_videos:
        _require(
            len(candidates) == args.expected_videos,
            f"Expected {args.expected_videos} videos, found {len(candidates)}",
        )

    print("Stage 4 combined_v2 real-data smoke check (read-only)")
    print(f"candidates_list       : {args.candidates_list}")
    print(f"combined_root         : {args.combined_root}")
    print(f"legacy_annotated_root : {args.legacy_annotated_root}")
    print(f"videos                : {len(candidates)}")

    reports = []
    failures = []
    for index, source_h5 in enumerate(candidates, start=1):
        video_name = _video_name(source_h5, args.video_suffix_to_strip)
        combined_h5 = args.combined_root / args.combined_h5_template.format(
            video_name=video_name
        )
        try:
            legacy_h5 = _find_unique(
                args.legacy_annotated_root,
                args.legacy_annotated_filename_template.format(
                    video_name=video_name
                ),
            )
            print(f"[{index}/{len(candidates)}] checking {video_name}")
            report = _inspect_video(
                source_h5,
                video_name=video_name,
                combined_h5=combined_h5,
                legacy_annotated_h5=legacy_h5,
            )
            reports.append(report)
            print(
                "  [OK] "
                f"targets={report.recovered_frames}/{report.target_frames}, "
                "combined_bbox_points="
                f"{report.recovered_bbox_min}/"
                f"{report.recovered_bbox_median:.1f}/"
                f"{report.recovered_bbox_max}, "
                f"multiplier={report.multiplier:.3f}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((video_name, message))
            print(f"  [FAIL] {message}")

    total_targets = sum(report.target_frames for report in reports)
    total_recovered = sum(report.recovered_frames for report in reports)
    if (
        args.expected_target_frames
        and total_targets != args.expected_target_frames
    ):
        failures.append(
            (
                "aggregate",
                f"Expected {args.expected_target_frames} target frames, "
                f"detected {total_targets}",
            )
        )

    print("\nPer-video comparison")
    print(
        "video\tframes\tlegacy_points\tcombined_points\tmultiplier\t"
        "valid_bbox_frames\ttargets\trecovered\tbbox_min\tbbox_median\t"
        "bbox_max\tpreexisting_bbox_frames\tpreexisting_min_added\t"
        "bbox_global\tbbox_local\tbbox_tophat\tbbox_context\t"
        "bbox_context_only"
    )
    for report in reports:
        preexisting_min = (
            "NA"
            if report.preexisting_min_added is None
            else str(report.preexisting_min_added)
        )
        print(
            f"{report.video_name}\t{report.num_frames}\t"
            f"{report.legacy_points}\t{report.combined_points}\t"
            f"{report.multiplier:.6f}\t{report.valid_bbox_frames}\t"
            f"{report.target_frames}\t{report.recovered_frames}\t"
            f"{report.recovered_bbox_min}\t"
            f"{report.recovered_bbox_median:.3f}\t"
            f"{report.recovered_bbox_max}\t"
            f"{report.preexisting_bbox_frames}\t{preexisting_min}\t"
            f"{report.bbox_global_points}\t{report.bbox_local_points}\t"
            f"{report.bbox_tophat_points}\t{report.bbox_context_points}\t"
            f"{report.bbox_context_only_points}"
        )

    print("\nSmoke-check summary")
    print(f"  videos_checked   : {len(reports)}")
    print(f"  target_frames    : {total_targets}")
    print(f"  recovered_frames : {total_recovered}")
    print(f"  failures         : {len(failures)}")
    print("  files_written    : 0")
    if failures:
        print("\nFailures")
        for video_name, message in failures:
            print(f"  {video_name}: {message}")
        raise SystemExit(1)
    _require(total_recovered == total_targets, "Not all target frames recovered")
    print("Stage 4 combined_v2 real-data smoke checks passed.")


if __name__ == "__main__":
    main()
