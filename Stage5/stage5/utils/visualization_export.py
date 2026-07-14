from __future__ import annotations

from pathlib import Path

import numpy as np


def probability_to_rgb(probability: np.ndarray) -> np.ndarray:
    prob = np.clip(probability.astype(np.float32), 0.0, 1.0)
    low = np.asarray([70, 110, 180], dtype=np.float32)
    high = np.asarray([230, 60, 50], dtype=np.float32)
    rgb = low[None, :] * (1.0 - prob[:, None]) + high[None, :] * prob[:, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def write_probability_ply(
    path: str | Path,
    points: np.ndarray,
    probability: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    probability = np.asarray(probability, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if probability.ndim != 1 or probability.shape[0] != points.shape[0]:
        raise ValueError(
            f"probability must have shape [N], got {probability.shape} for {points.shape[0]} points"
        )

    colors = probability_to_rgb(probability)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_selected_probability_ply(
    path: str | Path,
    points: np.ndarray,
    probability: np.ndarray,
    selected_mask: np.ndarray,
) -> int:
    """Write only selected points as a probability-colored PLY."""
    points = np.asarray(points, dtype=np.float32)
    probability = np.asarray(probability, dtype=np.float32)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    if selected_mask.ndim != 1 or selected_mask.shape[0] != points.shape[0]:
        raise ValueError(
            f"selected_mask must have shape [N], got {selected_mask.shape} "
            f"for {points.shape[0]} points"
        )

    selected_points = points[selected_mask]
    selected_probability = probability[selected_mask]
    write_probability_ply(path, selected_points, selected_probability)
    return int(selected_points.shape[0])
