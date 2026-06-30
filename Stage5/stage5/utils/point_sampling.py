from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LABEL_IGNORE = -1


@dataclass(frozen=True)
class PointSamplingConfig:
    mode: str = "video"
    num_points: int | None = 8192
    exclude_ignore: bool = False
    positive_oversample_ratio: float = 0.0
    frame_window_size: int | None = None


def _validate_labels_and_mask(labels: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    valid_mask = np.asarray(valid_mask).astype(bool)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if valid_mask.ndim != 1:
        raise ValueError(f"valid_mask must be 1D, got shape {valid_mask.shape}")
    if labels.shape[0] != valid_mask.shape[0]:
        raise ValueError(
            f"labels and valid_mask length mismatch: {labels.shape[0]} vs {valid_mask.shape[0]}"
        )
    return labels, valid_mask


def _valid_sampling_candidates(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    candidate_indices: np.ndarray | None = None,
    exclude_ignore: bool,
) -> np.ndarray:
    total_points = int(labels.shape[0])
    if total_points == 0:
        raise ValueError("Cannot sample from an empty point cloud")

    if candidate_indices is None:
        candidates = np.arange(total_points, dtype=np.int64)
    else:
        candidates = np.asarray(candidate_indices, dtype=np.int64)
        if candidates.ndim != 1:
            raise ValueError(f"candidate_indices must be 1D, got shape {candidates.shape}")
        if candidates.size == 0:
            candidates = np.arange(total_points, dtype=np.int64)

    if candidates.size > 0 and (candidates.min() < 0 or candidates.max() >= total_points):
        raise IndexError("candidate_indices contains values outside the point-cloud range")

    if exclude_ignore:
        candidates = candidates[valid_mask[candidates] & (labels[candidates] != LABEL_IGNORE)]
        if candidates.size == 0:
            candidates = np.arange(total_points, dtype=np.int64)

    return candidates


def _sample_from_candidates(
    labels: np.ndarray,
    candidates: np.ndarray,
    *,
    num_points: int | None,
    rng: np.random.Generator,
    positive_oversample_ratio: float,
) -> np.ndarray:
    if num_points is None or int(num_points) <= 0:
        return candidates.astype(np.int64)

    num_points = int(num_points)
    replace = candidates.size < num_points

    positive_oversample_ratio = float(positive_oversample_ratio)
    positive_oversample_ratio = min(max(positive_oversample_ratio, 0.0), 1.0)
    if positive_oversample_ratio <= 0.0:
        return rng.choice(candidates, size=num_points, replace=replace).astype(np.int64)

    positive_candidates = candidates[labels[candidates] == 1]
    if positive_candidates.size == 0:
        return rng.choice(candidates, size=num_points, replace=replace).astype(np.int64)

    num_positive = int(round(num_points * positive_oversample_ratio))
    num_positive = min(num_positive, num_points)
    num_other = num_points - num_positive

    sampled_positive = rng.choice(
        positive_candidates,
        size=num_positive,
        replace=positive_candidates.size < num_positive,
    ).astype(np.int64)

    sampled_other = rng.choice(
        candidates,
        size=num_other,
        replace=candidates.size < num_other,
    ).astype(np.int64)

    sampled = np.concatenate([sampled_positive, sampled_other], axis=0)
    rng.shuffle(sampled)
    return sampled


def sample_video_points(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    num_points: int | None,
    rng: np.random.Generator,
    exclude_ignore: bool = False,
    positive_oversample_ratio: float = 0.0,
) -> np.ndarray:
    """Sample point indices from one video-level point cloud."""
    labels, valid_mask = _validate_labels_and_mask(labels, valid_mask)
    candidates = _valid_sampling_candidates(
        labels,
        valid_mask,
        exclude_ignore=exclude_ignore,
    )
    return _sample_from_candidates(
        labels,
        candidates,
        num_points=num_points,
        rng=rng,
        positive_oversample_ratio=positive_oversample_ratio,
    )


def choose_frame_window_candidates(
    frame_order: np.ndarray,
    *,
    frame_window_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose point indices belonging to a random contiguous frame-order window."""
    frame_order = np.asarray(frame_order)
    if frame_order.ndim != 1:
        raise ValueError(f"frame_order must be 1D, got shape {frame_order.shape}")
    if frame_order.size == 0:
        raise ValueError("Cannot choose a frame window from an empty point cloud")
    frame_window_size = int(frame_window_size)
    if frame_window_size <= 0:
        raise ValueError("frame_window_size must be positive")

    unique_orders = np.unique(frame_order)
    if unique_orders.size <= frame_window_size:
        return np.arange(frame_order.shape[0], dtype=np.int64)

    max_start = unique_orders.size - frame_window_size
    start_pos = int(rng.integers(0, max_start + 1))
    selected_orders = unique_orders[start_pos : start_pos + frame_window_size]
    return np.flatnonzero(np.isin(frame_order, selected_orders)).astype(np.int64)


def sample_frame_window_points(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    frame_order: np.ndarray,
    *,
    num_points: int | None,
    frame_window_size: int,
    rng: np.random.Generator,
    exclude_ignore: bool = False,
    positive_oversample_ratio: float = 0.0,
) -> np.ndarray:
    """Sample points from a random contiguous frame window."""
    labels, valid_mask = _validate_labels_and_mask(labels, valid_mask)
    frame_candidates = choose_frame_window_candidates(
        frame_order,
        frame_window_size=frame_window_size,
        rng=rng,
    )
    candidates = _valid_sampling_candidates(
        labels,
        valid_mask,
        candidate_indices=frame_candidates,
        exclude_ignore=exclude_ignore,
    )
    return _sample_from_candidates(
        labels,
        candidates,
        num_points=num_points,
        rng=rng,
        positive_oversample_ratio=positive_oversample_ratio,
    )


def sample_points(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    config: PointSamplingConfig,
    frame_order: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch point sampling by mode."""
    if config.mode == "video":
        return sample_video_points(
            labels,
            valid_mask,
            num_points=config.num_points,
            rng=rng,
            exclude_ignore=config.exclude_ignore,
            positive_oversample_ratio=config.positive_oversample_ratio,
        )
    if config.mode == "frame_window":
        if frame_order is None:
            raise ValueError("frame_order is required when sampling mode is 'frame_window'")
        if config.frame_window_size is None:
            raise ValueError("frame_window_size is required when sampling mode is 'frame_window'")
        return sample_frame_window_points(
            labels,
            valid_mask,
            frame_order,
            num_points=config.num_points,
            frame_window_size=config.frame_window_size,
            rng=rng,
            exclude_ignore=config.exclude_ignore,
            positive_oversample_ratio=config.positive_oversample_ratio,
        )
    raise ValueError(f"Unknown sampling mode: {config.mode}")
