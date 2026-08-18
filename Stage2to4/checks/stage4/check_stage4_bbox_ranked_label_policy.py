from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_POSITIVE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Stage 4 BBox-ranked H5 labels: no-BBox frames must be "
            "background, and non-positive points inside a BBox must be ignore."
        )
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--expected_files", type=int, default=182)
    return parser.parse_args()


def audit_file(path: Path) -> dict[str, int]:
    required = (
        "point_cloud/frame_order",
        "point_cloud/pixel_xy",
        "annotation/point_label",
        "annotation/valid_mask",
        "frame_annotation/frame_order",
        "frame_annotation/bbox_local_xyxy",
    )
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise AssertionError(f"missing datasets: {missing}")
        if str(handle.attrs.get("label_mode", "")) != "bbox_ranked_global_local":
            raise AssertionError("unexpected label_mode")
        if int(handle.attrs.get("no_bbox_label", -999)) != LABEL_BACKGROUND:
            raise AssertionError("no_bbox_label is not background (0)")
        if str(handle.attrs.get("bbox_inside_non_contour_label", "")) != "ignore":
            raise AssertionError("bbox_inside_non_contour_label is not ignore")

        frame_order = handle["point_cloud/frame_order"][:]
        pixel_xy = handle["point_cloud/pixel_xy"][:]
        labels = handle["annotation/point_label"][:]
        valid_mask = handle["annotation/valid_mask"][:].astype(bool)
        bbox_orders = handle["frame_annotation/frame_order"][:].astype(np.int64)
        bboxes = handle["frame_annotation/bbox_local_xyxy"][:].astype(np.float64)

    if frame_order.ndim != 1 or labels.shape != frame_order.shape:
        raise AssertionError("frame_order and point_label must be aligned 1-D arrays")
    if pixel_xy.shape != (frame_order.size, 2):
        raise AssertionError("pixel_xy must have shape [num_points, 2]")
    if valid_mask.shape != labels.shape:
        raise AssertionError("valid_mask and point_label must be aligned")
    if bbox_orders.ndim != 1 or bboxes.shape != (bbox_orders.size, 4):
        raise AssertionError("frame_annotation BBox arrays are not aligned")
    if not np.array_equal(valid_mask, labels != LABEL_IGNORE):
        raise AssertionError("valid_mask does not match point_label != ignore")
    unexpected = np.setdiff1d(
        np.unique(labels),
        np.asarray([LABEL_IGNORE, LABEL_BACKGROUND, LABEL_POSITIVE]),
    )
    if unexpected.size:
        raise AssertionError(f"unexpected point labels: {unexpected.tolist()}")

    bbox_frame_values = np.unique(bbox_orders)
    no_bbox_mask = ~np.isin(frame_order, bbox_frame_values)
    no_bbox_labels = np.unique(labels[no_bbox_mask])
    if no_bbox_labels.size and not np.array_equal(
        no_bbox_labels,
        np.asarray([LABEL_BACKGROUND], dtype=no_bbox_labels.dtype),
    ):
        raise AssertionError(
            f"no-BBox points are not all background: {no_bbox_labels.tolist()}"
        )

    rounded_xy = np.rint(pixel_xy).astype(np.int64)
    bbox_inside_points = 0
    bbox_nonpositive_ignore_points = 0
    for order in bbox_frame_values:
        point_indices = np.flatnonzero(frame_order == order)
        if point_indices.size == 0:
            continue
        xy = rounded_xy[point_indices]
        inside_union = np.zeros(point_indices.size, dtype=bool)
        for bbox in bboxes[bbox_orders == order]:
            if not np.isfinite(bbox).all():
                continue
            x1, y1, x2, y2 = bbox
            left = int(np.floor(max(0.0, x1)))
            top = int(np.floor(max(0.0, y1)))
            right = int(np.ceil(x2))
            bottom = int(np.ceil(y2))
            if right <= left or bottom <= top:
                continue
            inside_union |= (
                (xy[:, 0] >= left)
                & (xy[:, 0] <= right)
                & (xy[:, 1] >= top)
                & (xy[:, 1] <= bottom)
            )

        inside_labels = labels[point_indices[inside_union]]
        invalid_inside = inside_labels[
            (inside_labels != LABEL_POSITIVE) & (inside_labels != LABEL_IGNORE)
        ]
        if invalid_inside.size:
            raise AssertionError(
                "BBox contains non-positive points that are not ignore: "
                f"labels={np.unique(invalid_inside).tolist()}"
            )
        bbox_inside_points += int(inside_labels.size)
        bbox_nonpositive_ignore_points += int(
            np.sum(inside_labels == LABEL_IGNORE)
        )

    return {
        "points": int(labels.size),
        "no_bbox_frames": int(
            np.setdiff1d(np.unique(frame_order), bbox_frame_values).size
        ),
        "no_bbox_background_points": int(no_bbox_mask.sum()),
        "bbox_inside_points": bbox_inside_points,
        "bbox_nonpositive_ignore_points": bbox_nonpositive_ignore_points,
        "positive_points": int(np.sum(labels == LABEL_POSITIVE)),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob(args.pattern))
    if len(paths) != args.expected_files:
        raise SystemExit(
            f"Input count mismatch: expected={args.expected_files}, actual={len(paths)}"
        )

    totals = {
        "points": 0,
        "no_bbox_frames": 0,
        "no_bbox_background_points": 0,
        "bbox_inside_points": 0,
        "bbox_nonpositive_ignore_points": 0,
        "positive_points": 0,
    }
    for index, path in enumerate(paths, start=1):
        try:
            stats = audit_file(path)
        except Exception as exc:
            raise SystemExit(f"[FAIL] {path}: {type(exc).__name__}: {exc}") from exc
        for key, value in stats.items():
            totals[key] += value
        print(
            f"[{index}/{len(paths)}] [OK] {path.name}: "
            f"no_bbox_bg={stats['no_bbox_background_points']}, "
            f"bbox_ignore={stats['bbox_nonpositive_ignore_points']}, "
            f"positive={stats['positive_points']}"
        )

    print("\nStage 4 BBox-ranked label-policy summary")
    print(f"  files                         : {len(paths)}")
    for key, value in totals.items():
        print(f"  {key:<30}: {value}")
    print("Stage 4 BBox-ranked label-policy checks passed.")


if __name__ == "__main__":
    main()
