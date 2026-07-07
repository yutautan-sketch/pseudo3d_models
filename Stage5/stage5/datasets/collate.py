from __future__ import annotations

from typing import Any

import torch


def _pad_tensor(
    value: torch.Tensor,
    *,
    target_points: int,
    fill_value: float | int | bool,
) -> torch.Tensor:
    if value.ndim == 1:
        out_shape = (target_points,)
    elif value.ndim == 2:
        out_shape = (target_points, value.shape[1])
    else:
        raise ValueError(f"Expected [N] or [N, C] tensor, got shape {tuple(value.shape)}")

    out = torch.full(out_shape, fill_value, dtype=value.dtype)
    out[: value.shape[0]] = value
    return out


def pad_point_window_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-size point-window samples into a batch."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    max_points = max(int(sample["points"].shape[0]) for sample in samples)
    batch: dict[str, Any] = {
        "points": torch.stack(
            [_pad_tensor(sample["points"], target_points=max_points, fill_value=0.0) for sample in samples]
        ),
        "features": torch.stack(
            [_pad_tensor(sample["features"], target_points=max_points, fill_value=0.0) for sample in samples]
        ),
        "labels": torch.stack(
            [_pad_tensor(sample["labels"], target_points=max_points, fill_value=-1) for sample in samples]
        ),
        "valid_mask": torch.stack(
            [_pad_tensor(sample["valid_mask"], target_points=max_points, fill_value=False) for sample in samples]
        ),
        "frame_order": torch.stack(
            [_pad_tensor(sample["frame_order"], target_points=max_points, fill_value=-1) for sample in samples]
        ),
        "point_indices": torch.stack(
            [_pad_tensor(sample["point_indices"], target_points=max_points, fill_value=-1) for sample in samples]
        ),
        "window_start": torch.as_tensor([sample["window_start"] for sample in samples], dtype=torch.long),
        "window_end": torch.as_tensor([sample["window_end"] for sample in samples], dtype=torch.long),
        "num_points": torch.as_tensor([sample["points"].shape[0] for sample in samples], dtype=torch.long),
        "meta": [sample.get("meta", {}) for sample in samples],
    }

    return batch

