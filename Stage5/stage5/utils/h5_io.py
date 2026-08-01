from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


POINT_CLOUD_REQUIRED_KEYS = (
    "points",
    "intensity",
    "alpha",
    "confidence",
    "frame_order",
    "pixel_xy",
)

ANNOTATION_REQUIRED_KEYS = (
    "point_label",
    "valid_mask",
)

OPTIONAL_POINT_LEVEL_KEYS = (
    "frame_index",
    "source_type",
    "source_flags",
    "sampling_confidence",
)

OPTIONAL_FRAME_LEVEL_KEYS = (
    "per_frame_counts",
    "per_frame_global_counts",
    "per_frame_local_percentile_counts",
    "per_frame_tophat_counts",
    "per_frame_context_grid_counts",
    "per_frame_context_only_counts",
    "per_frame_evidence_counts",
    "per_frame_overlap_counts",
)

OPTIONAL_POINT_CLOUD_KEYS = OPTIONAL_POINT_LEVEL_KEYS + OPTIONAL_FRAME_LEVEL_KEYS


def read_path_list(path: str | Path) -> list[Path]:
    """Read newline-delimited paths, preserving absolute external data paths."""
    path = Path(path)
    base_dir = path.parent
    items: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            item = Path(token)
            if not item.is_absolute():
                item = base_dir / item
            items.append(item)
    return items


def require_h5_keys(group: h5py.Group, keys: tuple[str, ...], *, path: Path, group_name: str) -> None:
    missing = [key for key in keys if key not in group]
    if missing:
        missing_text = ", ".join(f"{group_name}/{key}" for key in missing)
        raise KeyError(f"Missing required dataset(s) in {path}: {missing_text}")


def load_stage5_pointcloud_h5(path: str | Path) -> dict[str, Any]:
    """Load the point-cloud and annotation arrays needed by Stage 5 training."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotated point-cloud h5 not found: {path}")

    with h5py.File(path, "r") as f:
        if "point_cloud" not in f:
            raise KeyError(f"'point_cloud' group not found in {path}")
        if "annotation" not in f:
            raise KeyError(f"'annotation' group not found in {path}")

        point_cloud = f["point_cloud"]
        annotation = f["annotation"]
        require_h5_keys(point_cloud, POINT_CLOUD_REQUIRED_KEYS, path=path, group_name="point_cloud")
        require_h5_keys(annotation, ANNOTATION_REQUIRED_KEYS, path=path, group_name="annotation")

        data: dict[str, Any] = {
            "points": point_cloud["points"][:].astype(np.float32),
            "intensity": point_cloud["intensity"][:],
            "alpha": point_cloud["alpha"][:],
            "confidence": point_cloud["confidence"][:].astype(np.float32),
            "frame_order": point_cloud["frame_order"][:],
            "pixel_xy": point_cloud["pixel_xy"][:].astype(np.float32),
            "point_label": annotation["point_label"][:].astype(np.int64),
            "valid_mask": annotation["valid_mask"][:].astype(bool),
            "point_cloud_attrs": dict(point_cloud.attrs),
            "annotation_attrs": dict(annotation.attrs),
        }

        for key in OPTIONAL_POINT_CLOUD_KEYS:
            if key in point_cloud:
                data[key] = point_cloud[key][:]

        if "frame_annotation" in f:
            data["frame_annotation_attrs"] = dict(f["frame_annotation"].attrs)
        data["file_attrs"] = dict(f.attrs)

    num_points = data["points"].shape[0]
    for key in ("intensity", "alpha", "confidence", "frame_order", "point_label", "valid_mask"):
        if data[key].shape[0] != num_points:
            raise ValueError(
                f"Dataset length mismatch in {path}: point_cloud/points has {num_points} "
                f"rows but {key} has {data[key].shape[0]}"
            )
    if data["pixel_xy"].shape[0] != num_points:
        raise ValueError(
            f"Dataset length mismatch in {path}: point_cloud/points has {num_points} "
            f"rows but pixel_xy has {data['pixel_xy'].shape[0]}"
        )
    for key in OPTIONAL_POINT_LEVEL_KEYS:
        if key in data and data[key].shape[0] != num_points:
            raise ValueError(
                f"Dataset length mismatch in {path}: point_cloud/points has "
                f"{num_points} rows but {key} has {data[key].shape[0]}"
            )

    return data


def load_pointcloud_for_inference(path: str | Path) -> dict[str, Any]:
    """Load point-cloud arrays for inference from annotated or unannotated H5."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point-cloud h5 not found: {path}")

    with h5py.File(path, "r") as f:
        if "point_cloud" not in f:
            raise KeyError(f"'point_cloud' group not found in {path}")

        point_cloud = f["point_cloud"]
        require_h5_keys(point_cloud, POINT_CLOUD_REQUIRED_KEYS, path=path, group_name="point_cloud")

        data: dict[str, Any] = {
            "points": point_cloud["points"][:].astype(np.float32),
            "intensity": point_cloud["intensity"][:],
            "alpha": point_cloud["alpha"][:],
            "confidence": point_cloud["confidence"][:].astype(np.float32),
            "frame_order": point_cloud["frame_order"][:],
            "pixel_xy": point_cloud["pixel_xy"][:].astype(np.float32),
            "point_cloud_attrs": dict(point_cloud.attrs),
            "file_attrs": dict(f.attrs),
        }

        for key in OPTIONAL_POINT_CLOUD_KEYS:
            if key in point_cloud:
                data[key] = point_cloud[key][:]

        if "annotation" in f:
            annotation = f["annotation"]
            if "point_label" in annotation:
                data["point_label"] = annotation["point_label"][:]
            if "valid_mask" in annotation:
                data["valid_mask"] = annotation["valid_mask"][:]

    num_points = data["points"].shape[0]
    for key in ("intensity", "alpha", "confidence", "frame_order"):
        if data[key].shape[0] != num_points:
            raise ValueError(
                f"Dataset length mismatch in {path}: point_cloud/points has {num_points} "
                f"rows but {key} has {data[key].shape[0]}"
            )
    if data["pixel_xy"].shape[0] != num_points:
        raise ValueError(
            f"Dataset length mismatch in {path}: point_cloud/points has {num_points} "
            f"rows but pixel_xy has {data['pixel_xy'].shape[0]}"
        )
    for key in OPTIONAL_POINT_LEVEL_KEYS:
        if key in data and data[key].shape[0] != num_points:
            raise ValueError(
                f"Dataset length mismatch in {path}: point_cloud/points has "
                f"{num_points} rows but {key} has {data[key].shape[0]}"
            )

    return data


def copy_h5_group(src_file: h5py.File, dst_file: h5py.File, group_name: str) -> None:
    if group_name in dst_file:
        del dst_file[group_name]
    src_file.copy(group_name, dst_file)
