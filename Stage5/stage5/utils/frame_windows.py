from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameWindow:
    window_id: int
    start_frame: int
    end_frame: int


def generate_frame_order_windows(
    frame_order: np.ndarray,
    *,
    window_size_frames: int = 12,
    window_stride_frames: int = 6,
    include_tail_window: bool = True,
) -> list[FrameWindow]:
    """Generate overlapping inclusive frame-order windows.

    Tail windows are allowed to contain fewer than ``window_size_frames`` frame
    orders. They start one stride after the previous window start and end at the
    final observed frame order, preserving the intended overlap with the
    previous window.
    """
    frame_order = np.asarray(frame_order)
    if frame_order.ndim != 1:
        raise ValueError(f"frame_order must be 1D, got shape {frame_order.shape}")
    if frame_order.size == 0:
        return []

    window_size_frames = int(window_size_frames)
    window_stride_frames = int(window_stride_frames)
    if window_size_frames <= 0:
        raise ValueError("window_size_frames must be positive")
    if window_stride_frames <= 0:
        raise ValueError("window_stride_frames must be positive")

    min_frame = int(np.min(frame_order))
    max_frame = int(np.max(frame_order))

    windows: list[FrameWindow] = []
    start = min_frame
    while start <= max_frame:
        end = min(start + window_size_frames - 1, max_frame)
        if end - start + 1 < window_size_frames and windows and not include_tail_window:
            break
        windows.append(
            FrameWindow(
                window_id=len(windows),
                start_frame=int(start),
                end_frame=int(end),
            )
        )
        if end >= max_frame:
            break
        start += window_stride_frames

    return windows


def point_indices_for_window(
    frame_order: np.ndarray,
    window: FrameWindow,
) -> np.ndarray:
    frame_order = np.asarray(frame_order)
    mask = (frame_order >= window.start_frame) & (frame_order <= window.end_frame)
    return np.flatnonzero(mask).astype(np.int64)


def summarize_frame_windows(
    frame_order: np.ndarray,
    windows: list[FrameWindow],
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for window in windows:
        rows.append(
            {
                "window_id": int(window.window_id),
                "start_frame": int(window.start_frame),
                "end_frame": int(window.end_frame),
                "num_points": int(point_indices_for_window(frame_order, window).size),
            }
        )
    return rows

