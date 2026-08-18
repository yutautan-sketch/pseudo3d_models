from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_INPUT_DIR = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_outputs/260711"
)
DEFAULT_PATTERN = "*_ts448_oym96_corr.h5"
DEFAULT_VIDEO_SUFFIX = "_ts448_oym96_corr"


@dataclass(frozen=True)
class LegacyAnnotationStats:
    num_points: int
    total_positive_points: int
    positive_frames: int
    zero_positive_all_frames: int
    bbox_frames: int
    zero_positive_bbox_frames: int
    valid_contour_frames: int
    zero_positive_valid_contour_frames: int
    zero_positive_valid_contour_with_points_frames: int
    eligible_frames: int
    low_positive_frames: int
    zero_positive_eligible_frames: int
    positive_points_min: int | None
    positive_points_median: float | None
    positive_per_binary_area_min: float | None
    positive_per_binary_area_median: float | None
    valid_bbox_frames: int
    invalid_bbox_rows: int
    zero_bbox_point_frames: int
    zero_bbox_point_with_frame_points: int
    zero_bbox_point_empty_frames: int
    low_bbox_point_frames: int
    bbox_points_min: int | None
    bbox_points_median: float | None
    zero_bbox_point_frame_orders: tuple[int, ...]
    zero_bbox_point_with_frame_points_orders: tuple[int, ...]
    zero_bbox_point_empty_frame_orders: tuple[int, ...]
    low_bbox_point_frame_orders: tuple[int, ...]
    direct_bbox_point_details: tuple[str, ...]
    low_positive_frame_orders: tuple[int, ...]
    low_positive_frame_indices: tuple[int, ...]
    low_positive_details: tuple[str, ...]
    zero_positive_bbox_details: tuple[str, ...]


@dataclass(frozen=True)
class CandidateResult:
    input_h5: Path
    video_name: str
    num_frames: int
    frame_shape: tuple[int, ...]
    image_dtype: str
    nonfinite_values: int
    frame_mean_min: float
    frame_mean_median: float
    frame_mean_max: float
    frame_p95_min: float
    frame_p95_median: float
    frame_p95_max: float
    positive_ratio_min: float
    positive_ratio_median: float
    dark_frames: int
    low_positive_frames: int
    legacy_status: str
    legacy_h5: Path | None
    legacy_num_points: int | None
    legacy_zero_frames: int | None
    legacy_zero_frame_orders: tuple[int, ...]
    annotation_status: str
    legacy_annotated_h5: Path | None
    annotation_stats: LegacyAnnotationStats | None


def _attr_text(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    text = str(value)
    return text if text else default


def derive_video_name(input_h5: Path, *, strip_suffix: str) -> str:
    stem = Path(input_h5).stem
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def collect_inputs(
    *,
    input_dir: Path,
    pattern: str,
    recursive: bool,
    max_files: int,
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    globber = input_dir.rglob if recursive else input_dir.glob
    paths = [path for path in sorted(globber(pattern)) if path.is_file()]
    if max_files > 0:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(
            f"No input H5 files matched pattern={pattern!r} in {input_dir}"
        )
    return paths


def _as_gray_255(frame: np.ndarray) -> tuple[np.ndarray, int]:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3:
        channel_first = (
            frame.shape[0] in {1, 3, 4}
            and frame.shape[1] > 4
            and frame.shape[2] > 4
        )
        channel_last = (
            frame.shape[-1] in {1, 3, 4}
            and frame.shape[0] > 4
            and frame.shape[1] > 4
        )
        if channel_first:
            if frame.shape[0] == 1:
                gray = frame[0]
            else:
                gray = np.mean(frame[:3], axis=0)
        elif channel_last:
            if frame.shape[-1] == 1:
                gray = frame[..., 0]
            else:
                gray = np.mean(frame[..., :3], axis=-1)
        else:
            raise ValueError(
                "Unsupported or ambiguous 3D local frame layout; expected "
                f"[1|3|4,H,W] or [H,W,1|3|4], got {frame.shape}"
            )
    else:
        raise ValueError(f"Unsupported local frame shape: {frame.shape}")

    gray = gray.astype(np.float64, copy=False)
    finite = np.isfinite(gray)
    nonfinite = int(gray.size - finite.sum())
    if finite.any():
        finite_values = gray[finite]
        if finite_values.min() >= -0.5 and finite_values.max() <= 1.5:
            gray = np.clip(gray, 0.0, 1.0) * 255.0
    gray = np.where(finite, np.clip(gray, 0.0, 255.0), np.nan)
    return gray, nonfinite


def test_supported_frame_layouts() -> None:
    base = np.linspace(0.0, 1.0, 8 * 9, dtype=np.float32).reshape(8, 9)
    expected = base.astype(np.float64) * 255.0
    layouts = {
        "hw": base,
        "chw1": base[None, ...],
        "hwc1": base[..., None],
        "chw3": np.stack([base, base, base], axis=0),
        "hwc3": np.stack([base, base, base], axis=-1),
        "chw4": np.stack([base, base, base, base], axis=0),
        "hwc4": np.stack([base, base, base, base], axis=-1),
    }
    for name, frame in layouts.items():
        actual, nonfinite = _as_gray_255(frame)
        assert nonfinite == 0, name
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-5,
            err_msg=name,
        )

    with_nonfinite = base.copy()
    with_nonfinite[0, 0] = np.nan
    with_nonfinite[0, 1] = np.inf
    _, nonfinite = _as_gray_255(with_nonfinite[None, ...])
    assert nonfinite == 2


def _count_points_inside_bbox_union(
    pixel_xy: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> int:
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float64)
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2:
        raise ValueError(f"pixel_xy must have shape [K,2], got {pixel_xy.shape}")
    if bbox_xyxy.ndim != 2 or bbox_xyxy.shape[1] != 4:
        raise ValueError(
            f"bbox_xyxy must have shape [N,4], got {bbox_xyxy.shape}"
        )
    if pixel_xy.shape[0] == 0 or bbox_xyxy.shape[0] == 0:
        return 0

    inside_any = np.zeros((pixel_xy.shape[0],), dtype=bool)
    for x1, y1, x2, y2 in bbox_xyxy:
        inside_any |= (
            (pixel_xy[:, 0] >= x1)
            & (pixel_xy[:, 0] <= x2)
            & (pixel_xy[:, 1] >= y1)
            & (pixel_xy[:, 1] <= y2)
        )
    return int(inside_any.sum())


def _classify_direct_bbox_candidates(
    valid_bbox_mask: np.ndarray,
    per_frame_points: np.ndarray,
    bbox_point_counts: np.ndarray,
    *,
    max_legacy_bbox_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_bbox_mask = np.asarray(valid_bbox_mask, dtype=bool)
    per_frame_points = np.asarray(per_frame_points, dtype=np.int64)
    bbox_point_counts = np.asarray(bbox_point_counts, dtype=np.int64)
    if not (
        valid_bbox_mask.ndim == 1
        and valid_bbox_mask.shape == per_frame_points.shape
        and valid_bbox_mask.shape == bbox_point_counts.shape
    ):
        raise ValueError("Direct BBox candidate arrays must be aligned 1D arrays")
    if max_legacy_bbox_points < 1:
        raise ValueError("max_legacy_bbox_points must be >= 1")

    zero_mask = valid_bbox_mask & (bbox_point_counts == 0)
    zero_with_frame_points_mask = zero_mask & (per_frame_points > 0)
    zero_empty_frame_mask = zero_mask & (per_frame_points == 0)
    low_mask = (
        valid_bbox_mask
        & (bbox_point_counts > 0)
        & (bbox_point_counts <= max_legacy_bbox_points)
    )
    return zero_mask, zero_with_frame_points_mask, zero_empty_frame_mask, low_mask


def test_direct_bbox_candidate_logic() -> None:
    points = np.array(
        [[0, 0], [2, 2], [3, 3], [5, 5], [8, 8]],
        dtype=np.float64,
    )
    boxes = np.array([[1, 1, 3, 3], [3, 3, 6, 6]], dtype=np.float64)
    assert _count_points_inside_bbox_union(points, boxes) == 3

    masks = _classify_direct_bbox_candidates(
        np.array([True, True, True, True, False]),
        np.array([10, 0, 20, 20, 0]),
        np.array([0, 0, 3, 8, 0]),
        max_legacy_bbox_points=5,
    )
    expected = (
        [True, True, False, False, False],
        [True, False, False, False, False],
        [False, True, False, False, False],
        [False, False, True, False, False],
    )
    for actual, wanted in zip(masks, expected):
        np.testing.assert_array_equal(actual, wanted)


def _summarize_frame(frame: np.ndarray) -> tuple[float, float, float, int]:
    gray, nonfinite = _as_gray_255(frame)
    finite = np.isfinite(gray)
    if not finite.any():
        return float("nan"), float("nan"), 0.0, nonfinite

    values = gray[finite]
    return (
        float(values.mean()),
        float(np.percentile(values, 95.0)),
        float(np.count_nonzero(values > 0.0) / values.size),
        nonfinite,
    )


def _validate_pseudo3d_shapes(handle: h5py.File) -> tuple[h5py.Dataset, int]:
    required = (
        "local_encoder_images",
        "frame_indices",
        "pixel_to_image",
        "pred_tracking_corr",
    )
    for key in required:
        if key not in handle:
            raise KeyError(f"Required dataset missing: {key}")

    images = handle["local_encoder_images"]
    if images.ndim not in {3, 4}:
        raise ValueError(
            "local_encoder_images must have shape [N,H,W], [N,C,H,W], or "
            "[N,H,W,C], "
            f"got {images.shape}"
        )
    num_frames = int(images.shape[0])
    if num_frames <= 0:
        raise ValueError("local_encoder_images contains no frames")

    frame_indices = handle["frame_indices"]
    tracking = handle["pred_tracking_corr"]
    pixel_to_image = handle["pixel_to_image"]
    if frame_indices.ndim != 1 or int(frame_indices.shape[0]) != num_frames:
        raise ValueError(
            "frame_indices length mismatch: "
            f"shape={frame_indices.shape}, num_frames={num_frames}"
        )
    if tracking.shape != (num_frames, 4, 4):
        raise ValueError(
            "pred_tracking_corr shape mismatch: "
            f"shape={tracking.shape}, expected=({num_frames}, 4, 4)"
        )
    if pixel_to_image.shape != (4, 4):
        raise ValueError(
            f"pixel_to_image must have shape (4, 4), got {pixel_to_image.shape}"
        )
    return images, num_frames


def find_legacy_h5(
    *,
    legacy_root: Path | None,
    filename_template: str,
    video_name: str,
) -> tuple[str, Path | None]:
    if legacy_root is None:
        return "not_requested", None
    legacy_root = Path(legacy_root)
    if not legacy_root.exists():
        return "root_missing", None

    filename = filename_template.format(video_name=video_name)
    matches = sorted(path for path in legacy_root.rglob(filename) if path.is_file())
    if not matches:
        return "missing", None
    if len(matches) > 1:
        return f"ambiguous:{len(matches)}", None
    return "found", matches[0]


def read_legacy_counts(
    legacy_h5: Path,
    *,
    expected_num_frames: int,
) -> tuple[int, int, tuple[int, ...]]:
    with h5py.File(legacy_h5, "r") as handle:
        if "point_cloud" not in handle:
            raise KeyError("point_cloud group missing")
        group = handle["point_cloud"]
        if "points" not in group:
            raise KeyError("point_cloud/points missing")
        num_points = int(group["points"].shape[0])

        if "per_frame_counts" in group:
            counts = group["per_frame_counts"][:].astype(np.int64)
        elif "frame_order" in group:
            frame_order = group["frame_order"][:].astype(np.int64)
            counts = np.bincount(frame_order, minlength=expected_num_frames)
        else:
            raise KeyError("per_frame_counts and frame_order are both missing")

        sampling_mode = _attr_text(
            group.attrs.get("sampling_mode"),
            "legacy",
        )
        if sampling_mode != "legacy":
            raise ValueError(
                f"Expected legacy point cloud, sampling_mode={sampling_mode!r}"
            )
        point_mode = _attr_text(
            group.attrs.get("point_mode"),
            "foreground",
        )
        if point_mode != "foreground":
            raise ValueError(
                f"Expected foreground point cloud, point_mode={point_mode!r}"
            )

    if counts.shape != (expected_num_frames,):
        raise ValueError(
            "per_frame_counts length mismatch: "
            f"shape={counts.shape}, expected=({expected_num_frames},)"
        )
    if np.any(counts < 0):
        raise ValueError("per_frame_counts contains negative values")
    if int(counts.sum()) != num_points:
        raise ValueError(
            f"per_frame_counts sum={int(counts.sum())} != num_points={num_points}"
        )

    zero_orders = tuple(int(value) for value in np.flatnonzero(counts == 0))
    return num_points, len(zero_orders), zero_orders


def read_legacy_annotation_stats(
    annotated_h5: Path,
    *,
    expected_frame_indices: np.ndarray,
    min_contour_binary_area: int,
    max_legacy_positive_points: int,
    max_legacy_bbox_points: int,
) -> LegacyAnnotationStats:
    expected_frame_indices = np.asarray(expected_frame_indices, dtype=np.int64)
    if expected_frame_indices.ndim != 1:
        raise ValueError(
            "expected_frame_indices must be one-dimensional, "
            f"got {expected_frame_indices.shape}"
        )
    expected_num_frames = int(expected_frame_indices.shape[0])
    with h5py.File(annotated_h5, "r") as handle:
        for group_name in ("point_cloud", "annotation", "frame_annotation"):
            if group_name not in handle:
                raise KeyError(f"{group_name} group missing")
        point_cloud = handle["point_cloud"]
        annotation = handle["annotation"]
        frame_annotation = handle["frame_annotation"]

        for key in ("frame_order", "frame_index", "pixel_xy"):
            if key not in point_cloud:
                raise KeyError(f"point_cloud/{key} missing")
        for key in ("point_label", "valid_mask"):
            if key not in annotation:
                raise KeyError(f"annotation/{key} missing")
        for key in (
            "frame_order",
            "frame_index",
            "valid_contour",
            "binary_area",
            "bbox_local_xyxy",
        ):
            if key not in frame_annotation:
                raise KeyError(f"frame_annotation/{key} missing")

        frame_order = point_cloud["frame_order"][:].astype(np.int64)
        point_frame_index = point_cloud["frame_index"][:].astype(np.int64)
        pixel_xy = point_cloud["pixel_xy"][:].astype(np.float64)
        point_label = annotation["point_label"][:].astype(np.int8)
        valid_mask = annotation["valid_mask"][:].astype(bool)

        annotation_order = frame_annotation["frame_order"][:].astype(np.int64)
        annotation_index = frame_annotation["frame_index"][:].astype(np.int64)
        valid_contour = frame_annotation["valid_contour"][:].astype(bool)
        binary_area = frame_annotation["binary_area"][:].astype(np.int64)
        bbox_local_xyxy = frame_annotation["bbox_local_xyxy"][:].astype(
            np.float64
        )
        saved_union_counts = (
            frame_annotation["num_labeled_points_union"][:].astype(np.int64)
            if "num_labeled_points_union" in frame_annotation
            else None
        )

        sampling_mode = _attr_text(
            point_cloud.attrs.get("sampling_mode"),
            _attr_text(handle.attrs.get("point_cloud_sampling_mode"), "legacy"),
        )
        point_mode = _attr_text(
            point_cloud.attrs.get("point_mode"),
            _attr_text(handle.attrs.get("point_cloud_point_mode"), "foreground"),
        )

    num_points = int(frame_order.shape[0])
    for name, value in (
        ("point_cloud/frame_index", point_frame_index),
        ("point_cloud/pixel_xy", pixel_xy),
        ("annotation/point_label", point_label),
        ("annotation/valid_mask", valid_mask),
    ):
        if value.shape[0] != num_points:
            raise ValueError(
                f"{name} length={value.shape[0]} does not match points={num_points}"
            )
    num_rows = int(annotation_order.shape[0])
    for name, value in (
        ("frame_index", annotation_index),
        ("valid_contour", valid_contour),
        ("binary_area", binary_area),
        ("bbox_local_xyxy", bbox_local_xyxy),
    ):
        if value.shape[0] != num_rows:
            raise ValueError(
                f"frame_annotation/{name} length mismatch: {value.shape[0]} != "
                f"{num_rows}"
            )
    if pixel_xy.shape != (num_points, 2):
        raise ValueError(
            f"point_cloud/pixel_xy must have shape [K,2], got {pixel_xy.shape}"
        )
    if bbox_local_xyxy.shape != (num_rows, 4):
        raise ValueError(
            "frame_annotation/bbox_local_xyxy must have shape [N,4], "
            f"got {bbox_local_xyxy.shape}"
        )
    if saved_union_counts is not None and saved_union_counts.shape != (num_rows,):
        raise ValueError(
            "frame_annotation/num_labeled_points_union length mismatch: "
            f"{saved_union_counts.shape} != ({num_rows},)"
        )
    if sampling_mode != "legacy":
        raise ValueError(
            f"Expected legacy annotated H5, sampling_mode={sampling_mode!r}"
        )
    if point_mode != "foreground":
        raise ValueError(
            f"Expected foreground annotated H5, point_mode={point_mode!r}"
        )
    if np.any(frame_order < 0) or np.any(frame_order >= expected_num_frames):
        raise ValueError("point_cloud/frame_order is outside the input frame range")
    if np.any(annotation_order < 0) or np.any(
        annotation_order >= expected_num_frames
    ):
        raise ValueError("frame_annotation/frame_order is outside the input range")
    if np.any(binary_area < 0):
        raise ValueError("frame_annotation/binary_area contains negative values")

    if num_points:
        expected_point_indices = expected_frame_indices[frame_order]
        if not np.array_equal(point_frame_index, expected_point_indices):
            raise ValueError(
                "point_cloud/frame_index does not match input frame_indices"
            )
    if num_rows:
        expected_annotation_indices = expected_frame_indices[annotation_order]
        if not np.array_equal(annotation_index, expected_annotation_indices):
            raise ValueError(
                "frame_annotation/frame_index does not match input frame_indices"
            )

    per_frame_points = np.bincount(frame_order, minlength=expected_num_frames)
    positive_mask = valid_mask & (point_label == 1)
    per_frame_positive = np.bincount(
        frame_order[positive_mask],
        minlength=expected_num_frames,
    )

    bbox_orders = np.unique(annotation_order)
    valid_orders = np.unique(annotation_order[valid_contour])
    valid_bbox_rows = (
        np.all(np.isfinite(bbox_local_xyxy), axis=1)
        & (bbox_local_xyxy[:, 2] > bbox_local_xyxy[:, 0])
        & (bbox_local_xyxy[:, 3] > bbox_local_xyxy[:, 1])
    )
    valid_bbox_orders = np.unique(annotation_order[valid_bbox_rows])
    bbox_mask = np.zeros((expected_num_frames,), dtype=bool)
    bbox_mask[bbox_orders] = True
    valid_bbox_mask = np.zeros((expected_num_frames,), dtype=bool)
    valid_bbox_mask[valid_bbox_orders] = True
    valid_contour_mask = np.zeros((expected_num_frames,), dtype=bool)
    valid_contour_mask[valid_orders] = True
    max_binary_area = np.zeros((expected_num_frames,), dtype=np.int64)
    max_bbox_area = np.zeros((expected_num_frames,), dtype=np.float64)
    frame_indices = np.full((expected_num_frames,), -1, dtype=np.int64)
    bbox_point_counts = np.zeros((expected_num_frames,), dtype=np.int64)
    for order in bbox_orders.tolist():
        rows = annotation_order == order
        valid_bbox_rows_for_order = rows & valid_bbox_rows
        valid_rows = rows & valid_contour
        area_rows = valid_rows if np.any(valid_rows) else rows
        max_binary_area[order] = int(binary_area[area_rows].max())
        row_indices = np.unique(annotation_index[rows])
        if row_indices.size != 1:
            raise ValueError(
                f"frame_order={order} maps to multiple frame indices: "
                f"{row_indices.tolist()}"
            )
        frame_indices[order] = int(row_indices[0])

        if np.any(valid_bbox_rows_for_order):
            boxes = bbox_local_xyxy[valid_bbox_rows_for_order]
            bbox_areas = (
                (boxes[:, 2] - boxes[:, 0] + 1.0)
                * (boxes[:, 3] - boxes[:, 1] + 1.0)
            )
            max_bbox_area[order] = float(bbox_areas.max())

        point_mask = frame_order == order
        bbox_point_counts[order] = _count_points_inside_bbox_union(
            pixel_xy[point_mask],
            bbox_local_xyxy[valid_bbox_rows_for_order],
        )

        if saved_union_counts is not None:
            saved_values = np.unique(saved_union_counts[rows])
            if saved_values.size != 1 or int(saved_values[0]) != int(
                per_frame_positive[order]
            ):
                raise ValueError(
                    f"num_labeled_points_union mismatch at frame_order={order}"
                )

    eligible_mask = np.zeros((expected_num_frames,), dtype=bool)
    eligible_mask[valid_orders] = True
    eligible_mask &= per_frame_points > 0
    eligible_mask &= max_binary_area >= int(min_contour_binary_area)
    eligible_orders = np.flatnonzero(eligible_mask)
    low_mask = eligible_mask & (
        per_frame_positive <= int(max_legacy_positive_points)
    )
    low_orders = np.flatnonzero(low_mask)
    zero_positive_mask = per_frame_positive == 0
    zero_bbox_mask = zero_positive_mask & bbox_mask
    zero_valid_mask = zero_positive_mask & valid_contour_mask
    zero_valid_with_points_mask = zero_valid_mask & (per_frame_points > 0)

    # Stage 4 recovery candidates are selected from the legacy sampling result
    # itself.  Annotation contour validity and point labels are intentionally
    # not eligibility gates here: a failed/empty contour is one of the failure
    # modes that combined_v2 is expected to recover from.
    (
        zero_bbox_point_mask,
        zero_bbox_point_with_frame_points_mask,
        zero_bbox_point_empty_frame_mask,
        low_bbox_point_mask,
    ) = _classify_direct_bbox_candidates(
        valid_bbox_mask,
        per_frame_points,
        bbox_point_counts,
        max_legacy_bbox_points=max_legacy_bbox_points,
    )

    if eligible_orders.size:
        eligible_positive = per_frame_positive[eligible_orders]
        eligible_ratios = (
            eligible_positive.astype(np.float64)
            / max_binary_area[eligible_orders].astype(np.float64)
        )
        positive_min = int(eligible_positive.min())
        positive_median = float(np.median(eligible_positive))
        ratio_min = float(eligible_ratios.min())
        ratio_median = float(np.median(eligible_ratios))
    else:
        positive_min = None
        positive_median = None
        ratio_min = None
        ratio_median = None
    if valid_bbox_orders.size:
        valid_bbox_points = bbox_point_counts[valid_bbox_orders]
        bbox_min = int(valid_bbox_points.min())
        bbox_median = float(np.median(valid_bbox_points))
    else:
        bbox_min = None
        bbox_median = None

    direct_bbox_details = tuple(
        (
            f"order={int(order)},index={int(frame_indices[order])},"
            f"frame_points={int(per_frame_points[order])},"
            f"bbox_points={int(bbox_point_counts[order])},"
            f"bbox_area={float(max_bbox_area[order]):.1f},"
            f"annotation_positive={int(per_frame_positive[order])},"
            f"valid_contour={bool(valid_contour_mask[order])},"
            f"binary_area={int(max_binary_area[order])}"
        )
        for order in np.flatnonzero(
            zero_bbox_point_mask | low_bbox_point_mask
        )
    )

    details = tuple(
        (
            f"order={int(order)},index={int(frame_indices[order])},"
            f"positive={int(per_frame_positive[order])},"
            f"bbox_points={int(bbox_point_counts[order])},"
            f"binary_area={int(max_binary_area[order])},"
            "positive_per_binary_area="
            f"{float(per_frame_positive[order] / max_binary_area[order]):.6f}"
        )
        for order in low_orders
    )
    zero_bbox_details = []
    for order in np.flatnonzero(zero_bbox_mask):
        reasons = []
        if not valid_contour_mask[order]:
            reasons.append("no_valid_contour")
        if per_frame_points[order] == 0:
            reasons.append("no_legacy_points")
        if valid_contour_mask[order] and (
            max_binary_area[order] < min_contour_binary_area
        ):
            reasons.append("binary_area_below_limit")
        if eligible_mask[order]:
            reasons.append("eligible_zero_positive")
        if not reasons:
            reasons.append("not_eligible")
        zero_bbox_details.append(
            (
                f"order={int(order)},index={int(expected_frame_indices[order])},"
                f"frame_points={int(per_frame_points[order])},"
                f"valid_contour={bool(valid_contour_mask[order])},"
                f"bbox_points={int(bbox_point_counts[order])},"
                f"binary_area={int(max_binary_area[order])},"
                f"reason={'+'.join(reasons)}"
            )
        )
    return LegacyAnnotationStats(
        num_points=num_points,
        total_positive_points=int(positive_mask.sum()),
        positive_frames=int(np.count_nonzero(per_frame_positive > 0)),
        zero_positive_all_frames=int(np.count_nonzero(zero_positive_mask)),
        bbox_frames=int(bbox_orders.size),
        zero_positive_bbox_frames=int(np.count_nonzero(zero_bbox_mask)),
        valid_contour_frames=int(valid_orders.size),
        zero_positive_valid_contour_frames=int(
            np.count_nonzero(zero_valid_mask)
        ),
        zero_positive_valid_contour_with_points_frames=int(
            np.count_nonzero(zero_valid_with_points_mask)
        ),
        eligible_frames=int(eligible_orders.size),
        low_positive_frames=int(low_orders.size),
        zero_positive_eligible_frames=int(
            np.count_nonzero(low_mask & (per_frame_positive == 0))
        ),
        positive_points_min=positive_min,
        positive_points_median=positive_median,
        positive_per_binary_area_min=ratio_min,
        positive_per_binary_area_median=ratio_median,
        valid_bbox_frames=int(valid_bbox_orders.size),
        invalid_bbox_rows=int(np.count_nonzero(~valid_bbox_rows)),
        zero_bbox_point_frames=int(np.count_nonzero(zero_bbox_point_mask)),
        zero_bbox_point_with_frame_points=int(
            np.count_nonzero(zero_bbox_point_with_frame_points_mask)
        ),
        zero_bbox_point_empty_frames=int(
            np.count_nonzero(zero_bbox_point_empty_frame_mask)
        ),
        low_bbox_point_frames=int(np.count_nonzero(low_bbox_point_mask)),
        bbox_points_min=bbox_min,
        bbox_points_median=bbox_median,
        zero_bbox_point_frame_orders=tuple(
            int(order) for order in np.flatnonzero(zero_bbox_point_mask)
        ),
        zero_bbox_point_with_frame_points_orders=tuple(
            int(order)
            for order in np.flatnonzero(
                zero_bbox_point_with_frame_points_mask
            )
        ),
        zero_bbox_point_empty_frame_orders=tuple(
            int(order)
            for order in np.flatnonzero(zero_bbox_point_empty_frame_mask)
        ),
        low_bbox_point_frame_orders=tuple(
            int(order) for order in np.flatnonzero(low_bbox_point_mask)
        ),
        direct_bbox_point_details=direct_bbox_details,
        low_positive_frame_orders=tuple(int(order) for order in low_orders),
        low_positive_frame_indices=tuple(
            int(frame_indices[order]) for order in low_orders
        ),
        low_positive_details=details,
        zero_positive_bbox_details=tuple(zero_bbox_details),
    )


def inspect_candidate(
    input_h5: Path,
    *,
    video_suffix_to_strip: str,
    dark_p95_threshold: float,
    low_positive_ratio_threshold: float,
    legacy_root: Path | None,
    legacy_filename_template: str,
    legacy_annotated_root: Path | None,
    legacy_annotated_filename_template: str,
    min_contour_binary_area: int,
    max_legacy_positive_points: int,
    max_legacy_bbox_points: int,
) -> CandidateResult:
    input_h5 = Path(input_h5)
    video_name = derive_video_name(
        input_h5,
        strip_suffix=video_suffix_to_strip,
    )

    with h5py.File(input_h5, "r") as handle:
        images, num_frames = _validate_pseudo3d_shapes(handle)
        input_frame_indices = handle["frame_indices"][:].astype(np.int64)
        frame_means = np.zeros((num_frames,), dtype=np.float64)
        frame_p95 = np.zeros((num_frames,), dtype=np.float64)
        positive_ratios = np.zeros((num_frames,), dtype=np.float64)
        nonfinite_values = 0
        for frame_order in range(num_frames):
            mean, p95, positive_ratio, nonfinite = _summarize_frame(
                images[frame_order]
            )
            frame_means[frame_order] = mean
            frame_p95[frame_order] = p95
            positive_ratios[frame_order] = positive_ratio
            nonfinite_values += nonfinite
        if not np.all(np.isfinite(frame_means)) or not np.all(
            np.isfinite(frame_p95)
        ):
            raise ValueError("One or more frames contain no finite image values")
        frame_shape = tuple(int(value) for value in images.shape[1:])
        image_dtype = str(images.dtype)

    dark_frames = int(np.count_nonzero(frame_p95 <= dark_p95_threshold))
    low_positive_frames = int(
        np.count_nonzero(positive_ratios <= low_positive_ratio_threshold)
    )

    legacy_status, legacy_h5 = find_legacy_h5(
        legacy_root=legacy_root,
        filename_template=legacy_filename_template,
        video_name=video_name,
    )
    legacy_num_points = None
    legacy_zero_frames = None
    legacy_zero_orders: tuple[int, ...] = ()
    if legacy_h5 is not None:
        try:
            (
                legacy_num_points,
                legacy_zero_frames,
                legacy_zero_orders,
            ) = read_legacy_counts(
                legacy_h5,
                expected_num_frames=num_frames,
            )
        except Exception as exc:
            legacy_status = f"invalid:{type(exc).__name__}:{exc}"
            legacy_h5 = None

    annotation_status, legacy_annotated_h5 = find_legacy_h5(
        legacy_root=legacy_annotated_root,
        filename_template=legacy_annotated_filename_template,
        video_name=video_name,
    )
    annotation_stats = None
    if legacy_annotated_h5 is not None:
        try:
            annotation_stats = read_legacy_annotation_stats(
                legacy_annotated_h5,
                expected_frame_indices=input_frame_indices,
                min_contour_binary_area=min_contour_binary_area,
                max_legacy_positive_points=max_legacy_positive_points,
                max_legacy_bbox_points=max_legacy_bbox_points,
            )
        except Exception as exc:
            annotation_status = f"invalid:{type(exc).__name__}:{exc}"
            legacy_annotated_h5 = None

    return CandidateResult(
        input_h5=input_h5,
        video_name=video_name,
        num_frames=num_frames,
        frame_shape=frame_shape,
        image_dtype=image_dtype,
        nonfinite_values=nonfinite_values,
        frame_mean_min=float(np.nanmin(frame_means)),
        frame_mean_median=float(np.nanmedian(frame_means)),
        frame_mean_max=float(np.nanmax(frame_means)),
        frame_p95_min=float(np.nanmin(frame_p95)),
        frame_p95_median=float(np.nanmedian(frame_p95)),
        frame_p95_max=float(np.nanmax(frame_p95)),
        positive_ratio_min=float(np.nanmin(positive_ratios)),
        positive_ratio_median=float(np.nanmedian(positive_ratios)),
        dark_frames=dark_frames,
        low_positive_frames=low_positive_frames,
        legacy_status=legacy_status,
        legacy_h5=legacy_h5,
        legacy_num_points=legacy_num_points,
        legacy_zero_frames=legacy_zero_frames,
        legacy_zero_frame_orders=legacy_zero_orders,
        annotation_status=annotation_status,
        legacy_annotated_h5=legacy_annotated_h5,
        annotation_stats=annotation_stats,
    )


def _format_optional_int(value: int | None) -> str:
    return "NA" if value is None else str(value)


def _format_optional_float(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def print_result_table(results: list[CandidateResult]) -> None:
    header = (
        "idx\tvideo\tframes\tshape\tdtype\tnonfinite\tmean_min\tmean_med\t"
        "p95_min\tp95_med\tpositive_min\tdark_frames\tlow_positive_frames\t"
        "legacy_status\tlegacy_points\tlegacy_zero_frames\tannotation_status\t"
        "ann_total_positive\tann_positive_frames\tann_zero_all_frames\t"
        "ann_bbox_frames\tann_zero_bbox_frames\tann_valid_frames\t"
        "direct_valid_bbox_frames\tdirect_invalid_bbox_rows\t"
        "direct_zero_bbox_points\tdirect_zero_bbox_with_frame_points\t"
        "direct_zero_bbox_empty_frame\tdirect_low_bbox_points\t"
        "direct_bbox_points_min\tdirect_bbox_points_median\t"
        "ann_zero_valid_frames\tann_eligible_frames\t"
        "ann_low_positive_frames\tann_zero_eligible_frames\t"
        "ann_positive_min\tann_positive_median\tann_positive_per_binary_min"
    )
    print(header)
    for index, result in enumerate(results, start=1):
        shape = "x".join(str(value) for value in result.frame_shape)
        stats = result.annotation_stats
        ann_total_positive = stats.total_positive_points if stats else None
        ann_positive_frames = stats.positive_frames if stats else None
        ann_zero_all = stats.zero_positive_all_frames if stats else None
        ann_bbox = stats.bbox_frames if stats else None
        ann_zero_bbox = stats.zero_positive_bbox_frames if stats else None
        ann_valid = stats.valid_contour_frames if stats else None
        direct_valid_bbox = stats.valid_bbox_frames if stats else None
        direct_invalid_bbox = stats.invalid_bbox_rows if stats else None
        direct_zero_bbox = stats.zero_bbox_point_frames if stats else None
        direct_zero_with_points = (
            stats.zero_bbox_point_with_frame_points if stats else None
        )
        direct_zero_empty = (
            stats.zero_bbox_point_empty_frames if stats else None
        )
        direct_low_bbox = stats.low_bbox_point_frames if stats else None
        direct_bbox_min = stats.bbox_points_min if stats else None
        direct_bbox_median = stats.bbox_points_median if stats else None
        ann_zero_valid = (
            stats.zero_positive_valid_contour_frames if stats else None
        )
        ann_eligible = stats.eligible_frames if stats else None
        ann_low = stats.low_positive_frames if stats else None
        ann_zero_eligible = stats.zero_positive_eligible_frames if stats else None
        ann_positive_min = stats.positive_points_min if stats else None
        ann_positive_median = stats.positive_points_median if stats else None
        ann_ratio_min = (
            stats.positive_per_binary_area_min if stats else None
        )
        print(
            f"{index}\t{result.video_name}\t{result.num_frames}\t{shape}\t"
            f"{result.image_dtype}\t{result.nonfinite_values}\t"
            f"{result.frame_mean_min:.3f}\t{result.frame_mean_median:.3f}\t"
            f"{result.frame_p95_min:.3f}\t{result.frame_p95_median:.3f}\t"
            f"{result.positive_ratio_min:.6f}\t{result.dark_frames}\t"
            f"{result.low_positive_frames}\t{result.legacy_status}\t"
            f"{_format_optional_int(result.legacy_num_points)}\t"
            f"{_format_optional_int(result.legacy_zero_frames)}\t"
            f"{result.annotation_status}\t"
            f"{_format_optional_int(ann_total_positive)}\t"
            f"{_format_optional_int(ann_positive_frames)}\t"
            f"{_format_optional_int(ann_zero_all)}\t"
            f"{_format_optional_int(ann_bbox)}\t"
            f"{_format_optional_int(ann_zero_bbox)}\t"
            f"{_format_optional_int(ann_valid)}\t"
            f"{_format_optional_int(direct_valid_bbox)}\t"
            f"{_format_optional_int(direct_invalid_bbox)}\t"
            f"{_format_optional_int(direct_zero_bbox)}\t"
            f"{_format_optional_int(direct_zero_with_points)}\t"
            f"{_format_optional_int(direct_zero_empty)}\t"
            f"{_format_optional_int(direct_low_bbox)}\t"
            f"{_format_optional_int(direct_bbox_min)}\t"
            f"{_format_optional_float(direct_bbox_median)}\t"
            f"{_format_optional_int(ann_zero_valid)}\t"
            f"{_format_optional_int(ann_eligible)}\t"
            f"{_format_optional_int(ann_low)}\t"
            f"{_format_optional_int(ann_zero_eligible)}\t"
            f"{_format_optional_int(ann_positive_min)}\t"
            f"{_format_optional_float(ann_positive_median)}\t"
            f"{_format_optional_float(ann_ratio_min, 6)}"
        )


def _print_ranked(
    title: str,
    results: list[CandidateResult],
    *,
    top_k: int,
    detail: Any,
) -> None:
    print(f"\n{title}")
    if not results:
        print("  (no candidates)")
        return
    for rank, result in enumerate(results[:top_k], start=1):
        print(f"  {rank}. {result.video_name}: {detail(result)}")
        print(f"     {result.input_h5}")


def _print_direct_bbox_details(
    results: list[CandidateResult],
    *,
    top_k: int,
    order_attribute: str,
) -> None:
    for result in results[:top_k]:
        stats = result.annotation_stats
        assert stats is not None
        wanted_orders = set(getattr(stats, order_attribute))
        print(f"     annotated_h5: {result.legacy_annotated_h5}")
        printed = 0
        for detail in stats.direct_bbox_point_details:
            order = int(detail.split(",", 1)[0].split("=", 1)[1])
            if order not in wanted_orders:
                continue
            print(f"       {detail}")
            printed += 1
            if printed == 20:
                break
        if len(wanted_orders) > printed:
            print(f"       ... {len(wanted_orders) - printed} additional frames")


def print_recommendations(results: list[CandidateResult], *, top_k: int) -> None:
    direct_zero_with_points = sorted(
        [
            result
            for result in results
            if result.annotation_stats is not None
            and result.annotation_stats.zero_bbox_point_with_frame_points > 0
        ],
        key=lambda result: (
            -result.annotation_stats.zero_bbox_point_with_frame_points,
            -result.annotation_stats.zero_bbox_point_frames,
            result.annotation_stats.bbox_points_median,
            result.video_name,
        ),
    )
    direct_zero_empty = sorted(
        [
            result
            for result in results
            if result.annotation_stats is not None
            and result.annotation_stats.zero_bbox_point_empty_frames > 0
        ],
        key=lambda result: (
            -result.annotation_stats.zero_bbox_point_empty_frames,
            -result.annotation_stats.zero_bbox_point_frames,
            result.video_name,
        ),
    )
    direct_low_bbox = sorted(
        [
            result
            for result in results
            if result.annotation_stats is not None
            and result.annotation_stats.low_bbox_point_frames > 0
        ],
        key=lambda result: (
            -result.annotation_stats.low_bbox_point_frames,
            result.annotation_stats.bbox_points_min,
            result.annotation_stats.bbox_points_median,
            result.video_name,
        ),
    )
    zero_total_positive = sorted(
        [
            result
            for result in results
            if result.annotation_stats is not None
            and result.annotation_stats.total_positive_points == 0
        ],
        key=lambda result: (
            -result.annotation_stats.bbox_frames,
            result.video_name,
        ),
    )
    zero_bbox_positive = sorted(
        [
            result
            for result in results
            if result.annotation_stats is not None
            and result.annotation_stats.zero_positive_bbox_frames > 0
        ],
        key=lambda result: (
            -result.annotation_stats.zero_positive_bbox_frames,
            -result.annotation_stats.zero_positive_valid_contour_frames,
            result.video_name,
        ),
    )
    zero_valid_contour = sorted(
        [
            result
            for result in results
            if result.annotation_stats is not None
            and result.annotation_stats.zero_positive_valid_contour_frames > 0
        ],
        key=lambda result: (
            -result.annotation_stats.zero_positive_valid_contour_frames,
            -result.annotation_stats.zero_positive_valid_contour_with_points_frames,
            result.video_name,
        ),
    )
    annotation_low_positive = sorted(
        [
            result
            for result in results
            if result.annotation_stats is not None
            and result.annotation_stats.low_positive_frames > 0
        ],
        key=lambda result: (
            -result.annotation_stats.low_positive_frames,
            result.annotation_stats.positive_points_median,
            result.annotation_stats.positive_per_binary_area_min,
            result.video_name,
        ),
    )
    zero_positive = sorted(
        [result for result in results if (result.legacy_zero_frames or 0) > 0],
        key=lambda result: (
            -(result.legacy_zero_frames or 0),
            result.video_name,
        ),
    )
    darkest = sorted(
        results,
        key=lambda result: (
            result.frame_p95_min,
            result.positive_ratio_min,
            result.video_name,
        ),
    )
    normal_pool = [
        result
        for result in results
        if result.dark_frames == 0
        and result.low_positive_frames == 0
        and result.nonfinite_values == 0
    ]
    normal = sorted(
        normal_pool or results,
        key=lambda result: (
            -result.frame_p95_median,
            -result.frame_mean_median,
            result.video_name,
        ),
    )

    _print_ranked(
        "Recommended primary: zero legacy BBox points, frame has points",
        direct_zero_with_points,
        top_k=top_k,
        detail=lambda result: (
            "zero_bbox_with_frame_points="
            f"{result.annotation_stats.zero_bbox_point_with_frame_points}, "
            f"zero_bbox_total={result.annotation_stats.zero_bbox_point_frames}, "
            f"valid_bbox_frames={result.annotation_stats.valid_bbox_frames}, "
            f"bbox_points_median={result.annotation_stats.bbox_points_median:.3f}"
        ),
    )
    _print_direct_bbox_details(
        direct_zero_with_points,
        top_k=top_k,
        order_attribute="zero_bbox_point_with_frame_points_orders",
    )
    _print_ranked(
        "Recommended secondary: zero legacy BBox points, empty frame",
        direct_zero_empty,
        top_k=top_k,
        detail=lambda result: (
            f"zero_bbox_empty={result.annotation_stats.zero_bbox_point_empty_frames}, "
            f"zero_bbox_total={result.annotation_stats.zero_bbox_point_frames}, "
            f"valid_bbox_frames={result.annotation_stats.valid_bbox_frames}"
        ),
    )
    _print_direct_bbox_details(
        direct_zero_empty,
        top_k=top_k,
        order_attribute="zero_bbox_point_empty_frame_orders",
    )
    _print_ranked(
        "Recommended tertiary: low legacy BBox point count",
        direct_low_bbox,
        top_k=top_k,
        detail=lambda result: (
            f"low_bbox_frames={result.annotation_stats.low_bbox_point_frames}, "
            f"bbox_points_min={result.annotation_stats.bbox_points_min}, "
            f"bbox_points_median={result.annotation_stats.bbox_points_median:.3f}, "
            f"valid_bbox_frames={result.annotation_stats.valid_bbox_frames}"
        ),
    )
    _print_direct_bbox_details(
        direct_low_bbox,
        top_k=top_k,
        order_attribute="low_bbox_point_frame_orders",
    )

    _print_ranked(
        "Zero-positive audit: videos with zero total annotation-positive",
        zero_total_positive,
        top_k=top_k,
        detail=lambda result: (
            f"total_points={result.annotation_stats.num_points}, "
            f"bbox_frames={result.annotation_stats.bbox_frames}, "
            f"valid_contour_frames={result.annotation_stats.valid_contour_frames}, "
            f"zero_all_frames={result.annotation_stats.zero_positive_all_frames}"
        ),
    )
    _print_ranked(
        "Annotation-label audit: zero positive on BBox frames",
        zero_bbox_positive,
        top_k=top_k,
        detail=lambda result: (
            f"zero_bbox_frames={result.annotation_stats.zero_positive_bbox_frames}, "
            "zero_valid_contour_frames="
            f"{result.annotation_stats.zero_positive_valid_contour_frames}, "
            "zero_valid_with_points="
            f"{result.annotation_stats.zero_positive_valid_contour_with_points_frames}"
        ),
    )
    _print_ranked(
        "Annotation-label audit: valid-contour frames",
        zero_valid_contour,
        top_k=top_k,
        detail=lambda result: (
            "zero_valid_contour_frames="
            f"{result.annotation_stats.zero_positive_valid_contour_frames}, "
            "zero_valid_with_points="
            f"{result.annotation_stats.zero_positive_valid_contour_with_points_frames},"
            " "
            "zero_eligible="
            f"{result.annotation_stats.zero_positive_eligible_frames}"
        ),
    )
    for result in zero_bbox_positive[:top_k]:
        stats = result.annotation_stats
        assert stats is not None
        print(f"     annotated_h5: {result.legacy_annotated_h5}")
        for detail in stats.zero_positive_bbox_details[:20]:
            print(f"       {detail}")
        if len(stats.zero_positive_bbox_details) > 20:
            print(
                "       ... "
                f"{len(stats.zero_positive_bbox_details) - 20} additional frames"
            )

    _print_ranked(
        "Annotation-label diagnostic: low legacy annotation-positive",
        annotation_low_positive,
        top_k=top_k,
        detail=lambda result: (
            f"low_frames={result.annotation_stats.low_positive_frames}, "
            "zero_eligible="
            f"{result.annotation_stats.zero_positive_eligible_frames}, "
            f"positive_min={result.annotation_stats.positive_points_min}, "
            "positive_median="
            f"{result.annotation_stats.positive_points_median:.3f}, "
            "positive_per_binary_min="
            f"{result.annotation_stats.positive_per_binary_area_min:.6f}"
        ),
    )
    for result in annotation_low_positive[:top_k]:
        stats = result.annotation_stats
        assert stats is not None
        print(f"     annotated_h5: {result.legacy_annotated_h5}")
        for detail in stats.low_positive_details[:20]:
            print(f"       {detail}")
        if len(stats.low_positive_details) > 20:
            print(
                "       ... "
                f"{len(stats.low_positive_details) - 20} additional frames"
            )

    _print_ranked(
        "Recommended: legacy zero-positive",
        zero_positive,
        top_k=top_k,
        detail=lambda result: (
            f"zero_frames={result.legacy_zero_frames}, "
            f"orders={list(result.legacy_zero_frame_orders[:20])}"
        ),
    )
    _print_ranked(
        "Recommended: darkest",
        darkest,
        top_k=top_k,
        detail=lambda result: (
            f"p95_min={result.frame_p95_min:.3f}, "
            f"positive_min={result.positive_ratio_min:.6f}, "
            f"dark_frames={result.dark_frames}"
        ),
    )
    _print_ranked(
        "Recommended: normal brightness",
        normal,
        top_k=top_k,
        detail=lambda result: (
            f"p95_median={result.frame_p95_median:.3f}, "
            f"mean_median={result.frame_mean_median:.3f}"
        ),
    )

    signatures: dict[tuple[int, tuple[int, ...]], list[CandidateResult]] = {}
    for result in results:
        signatures.setdefault(
            (result.num_frames, result.frame_shape), []
        ).append(result)
    print("\nAvailable frame-count/image-shape signatures")
    for (num_frames, frame_shape), members in sorted(signatures.items()):
        shape = "x".join(str(value) for value in frame_shape)
        examples = ", ".join(member.video_name for member in members[:top_k])
        print(
            f"  frames={num_frames}, shape={shape}, count={len(members)}, "
            f"examples={examples}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only survey of pseudo3D H5 files for Stage 4 combined_v2 "
            "real-data smoke-test candidate selection. Legacy annotation/BBox "
            "data is optional and is used only for evaluation, never sampling."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--video_suffix_to_strip", default=DEFAULT_VIDEO_SUFFIX)
    parser.add_argument("--legacy_point_cloud_root", type=Path, default=None)
    parser.add_argument(
        "--legacy_filename_template",
        default="{video_name}_pointcloud_foreground.h5",
    )
    parser.add_argument(
        "--legacy_annotated_root",
        type=Path,
        default=None,
        help=(
            "Root searched recursively for legacy annotated H5 files. Defaults "
            "to --legacy_point_cloud_root when omitted."
        ),
    )
    parser.add_argument(
        "--legacy_annotated_filename_template",
        default="{video_name}_pointcloud_annotated_foreground.h5",
    )
    parser.add_argument("--min_contour_binary_area", type=int, default=100)
    parser.add_argument("--max_legacy_positive_points", type=int, default=5)
    parser.add_argument(
        "--max_legacy_bbox_points",
        type=int,
        default=5,
        help=(
            "Upper inclusive limit for the direct low-density BBox category. "
            "Zero-point BBox frames are reported separately."
        ),
    )
    parser.add_argument("--dark_p95_threshold", type=float, default=16.0)
    parser.add_argument(
        "--low_positive_ratio_threshold",
        type=float,
        default=0.005,
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_files < 0:
        raise ValueError("max_files must be >= 0")
    if args.dark_p95_threshold < 0.0 or args.dark_p95_threshold > 255.0:
        raise ValueError("dark_p95_threshold must be in [0, 255]")
    if not 0.0 <= args.low_positive_ratio_threshold <= 1.0:
        raise ValueError("low_positive_ratio_threshold must be in [0, 1]")
    if args.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if args.min_contour_binary_area < 1:
        raise ValueError("min_contour_binary_area must be >= 1")
    if args.max_legacy_positive_points < 0:
        raise ValueError("max_legacy_positive_points must be >= 0")
    if args.max_legacy_bbox_points < 1:
        raise ValueError("max_legacy_bbox_points must be >= 1")
    for name in (
        "legacy_filename_template",
        "legacy_annotated_filename_template",
    ):
        template = getattr(args, name)
        try:
            template.format(video_name="example")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"{name} must support only {{video_name}}"
            ) from exc


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    test_supported_frame_layouts()
    test_direct_bbox_candidate_logic()
    legacy_annotated_root = (
        args.legacy_annotated_root or args.legacy_point_cloud_root
    )
    inputs = collect_inputs(
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        max_files=args.max_files,
    )

    print("Stage 4 real-data candidate survey (read-only)")
    print(f"input_dir                 : {args.input_dir}")
    print(f"pattern                   : {args.pattern}")
    print(f"inputs_found              : {len(inputs)}")
    print(f"dark_p95_threshold        : {args.dark_p95_threshold}")
    print(
        "low_positive_ratio_limit  : "
        f"{args.low_positive_ratio_threshold}"
    )
    print(f"legacy_point_cloud_root   : {args.legacy_point_cloud_root}")
    print(f"legacy_annotated_root     : {legacy_annotated_root}")
    print(f"min_contour_binary_area   : {args.min_contour_binary_area}")
    print(f"max_legacy_positive_pts   : {args.max_legacy_positive_points}")
    print(f"max_legacy_bbox_points    : {args.max_legacy_bbox_points}")
    print("Primary candidate basis   : direct legacy points inside BBox")
    print("BBox/annotation usage     : evaluation only; never sampling")
    print("frame layout self-check   : OK (HW/CHW/HWC)")
    print("direct BBox self-check    : OK (union/count/categories)")

    results: list[CandidateResult] = []
    failures: list[tuple[Path, str]] = []
    for index, input_h5 in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] inspect: {input_h5.name}", file=sys.stderr)
        try:
            results.append(
                inspect_candidate(
                    input_h5,
                    video_suffix_to_strip=args.video_suffix_to_strip,
                    dark_p95_threshold=args.dark_p95_threshold,
                    low_positive_ratio_threshold=(
                        args.low_positive_ratio_threshold
                    ),
                    legacy_root=args.legacy_point_cloud_root,
                    legacy_filename_template=args.legacy_filename_template,
                    legacy_annotated_root=legacy_annotated_root,
                    legacy_annotated_filename_template=(
                        args.legacy_annotated_filename_template
                    ),
                    min_contour_binary_area=args.min_contour_binary_area,
                    max_legacy_positive_points=(
                        args.max_legacy_positive_points
                    ),
                    max_legacy_bbox_points=args.max_legacy_bbox_points,
                )
            )
        except Exception as exc:
            failures.append((input_h5, f"{type(exc).__name__}: {exc}"))

    if results:
        print("\nCandidate table")
        print_result_table(results)
        print_recommendations(results, top_k=args.top_k)

    print("\nFailures")
    if failures:
        for path, error in failures:
            print(f"  {path}: {error}")
    else:
        print("  none")

    annotation_results = [
        result.annotation_stats
        for result in results
        if result.annotation_stats is not None
    ]
    zero_total_videos = sum(
        stats.total_positive_points == 0 for stats in annotation_results
    )
    zero_bbox_videos = sum(
        stats.zero_positive_bbox_frames > 0 for stats in annotation_results
    )
    zero_valid_videos = sum(
        stats.zero_positive_valid_contour_frames > 0
        for stats in annotation_results
    )
    zero_direct_bbox_videos = sum(
        stats.zero_bbox_point_frames > 0 for stats in annotation_results
    )
    zero_direct_selective_videos = sum(
        stats.zero_bbox_point_with_frame_points > 0
        for stats in annotation_results
    )
    zero_direct_empty_videos = sum(
        stats.zero_bbox_point_empty_frames > 0
        for stats in annotation_results
    )
    low_direct_bbox_videos = sum(
        stats.low_bbox_point_frames > 0 for stats in annotation_results
    )
    print("\nSurvey summary")
    print(f"  inspected_ok : {len(results)}")
    print(f"  failures     : {len(failures)}")
    print(f"  annotation_ok: {len(annotation_results)}")
    print(f"  direct_zero_bbox_videos     : {zero_direct_bbox_videos}")
    print(
        "  direct_zero_selective_videos: "
        f"{zero_direct_selective_videos}"
    )
    print(f"  direct_zero_empty_videos    : {zero_direct_empty_videos}")
    print(f"  direct_low_bbox_videos      : {low_direct_bbox_videos}")
    print(f"  zero_total_positive_videos: {zero_total_videos}")
    print(f"  zero_positive_bbox_videos : {zero_bbox_videos}")
    print(f"  zero_positive_valid_videos: {zero_valid_videos}")
    print("  files_written: 0")

    if not results or (failures and args.fail_on_error):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
