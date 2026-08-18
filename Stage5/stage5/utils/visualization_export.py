from __future__ import annotations

from pathlib import Path

import numpy as np


DIAGNOSTIC_CATEGORY_NAMES = (
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "ignore_predicted_positive",
    "ignore_other",
)

DIAGNOSTIC_CATEGORY_COLORS = np.asarray(
    [
        [30, 180, 70],
        [230, 45, 45],
        [245, 190, 35],
        [65, 105, 180],
        [200, 65, 180],
        [145, 145, 145],
    ],
    dtype=np.uint8,
)


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


def diagnostic_segmentation_categories(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    *,
    ignore_index: int = -1,
    positive_label: int = 1,
) -> np.ndarray:
    labels = np.asarray(labels)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    pred_label = np.asarray(pred_label)
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {labels.shape}")
    if valid_mask.shape != labels.shape or pred_label.shape != labels.shape:
        raise ValueError("labels, valid_mask, and pred_label must share shape [N]")

    valid = valid_mask & (labels != int(ignore_index))
    target_positive = labels == int(positive_label)
    predicted_positive = pred_label == int(positive_label)

    categories = np.full(labels.shape, 5, dtype=np.uint8)
    categories[valid & target_positive & predicted_positive] = 0
    categories[valid & ~target_positive & predicted_positive] = 1
    categories[valid & target_positive & ~predicted_positive] = 2
    categories[valid & ~target_positive & ~predicted_positive] = 3
    categories[~valid & predicted_positive] = 4
    return categories


def write_diagnostic_segmentation_ply(
    path: str | Path,
    points: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    *,
    ignore_index: int = -1,
    positive_label: int = 1,
) -> dict[str, int]:
    """Write a PLY colored by valid confusion class and ignore prediction state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    categories = diagnostic_segmentation_categories(
        labels,
        valid_mask,
        pred_label,
        ignore_index=ignore_index,
        positive_label=positive_label,
    )
    if categories.shape[0] != points.shape[0]:
        raise ValueError(
            f"labels must contain {points.shape[0]} points, got {categories.shape[0]}"
        )

    colors = DIAGNOSTIC_CATEGORY_COLORS[categories]
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
        f.write("property uchar diagnostic_category\n")
        f.write("end_header\n")
        for point, color, category in zip(points, colors, categories):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(category)}\n"
            )

    counts = np.bincount(categories, minlength=len(DIAGNOSTIC_CATEGORY_NAMES))
    return {
        name: int(counts[index])
        for index, name in enumerate(DIAGNOSTIC_CATEGORY_NAMES)
    }
