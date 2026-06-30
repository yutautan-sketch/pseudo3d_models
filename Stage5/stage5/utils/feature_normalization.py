from __future__ import annotations

import numpy as np


def normalize_uint8_feature(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32) / 255.0


def normalize_frame_order(frame_order: np.ndarray) -> np.ndarray:
    values = frame_order.astype(np.float32)
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    denom = max(max_value - min_value, 1.0)
    return (values - min_value) / denom


def normalize_pixel_xy(pixel_xy: np.ndarray) -> np.ndarray:
    values = pixel_xy.astype(np.float32)
    if values.size == 0:
        return values
    max_xy = np.maximum(values.max(axis=0), 1.0)
    return values / max_xy


def normalize_xyz(points: np.ndarray) -> np.ndarray:
    """Center and scale each point cloud for stable point-network inputs."""
    points = points.astype(np.float32)
    if points.size == 0:
        return points
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    scale = np.linalg.norm(centered, axis=1).max()
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return centered / float(scale)

