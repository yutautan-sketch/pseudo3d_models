from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch

from stage5.datasets import Pseudo3DPointCloudDataset
from stage5.utils.label_policy import (
    LABEL_BACKGROUND,
    LABEL_IGNORE,
    LABEL_POSITIVE,
    apply_bbox_noncontour_label_policy,
    compute_bbox_inside_mask,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------
# Synthetic H5 with hand-placed, deterministic BBox / label geometry (S5-12
# Step E2). No CUDA is needed anywhere in this checker: label-policy logic and
# Dataset sample construction are plain numpy/torch, independent of PointNeXt.
# ----------------------------------------------------------------------------

POINTS_PER_FRAME = 12
# Per-frame point layout (local index -> (pixel_x, pixel_y), label):
#   0-3: candidates inside the frame's BBox (if any)
#   4-11: always outside any BBox in this synthetic layout
_INSIDE_XY = [(5, 5), (6, 6), (7, 7), (8, 8)]
_OUTSIDE_XY = [(300 + i, 300 + i) for i in range(8)]
_BBOX_XYXY = (0.0, 0.0, 20.0, 20.0)
_BBOX_XYXY_NO_OVERLAP = (200.0, 200.0, 220.0, 220.0)


def _frame_layout(
    *, has_bbox: bool, bbox_overlaps_points: bool, inside_labels: list[int], stray_outside_label: int | None
) -> tuple[list[tuple[int, int]], list[int], tuple[float, float, float, float] | None]:
    xy = list(_INSIDE_XY) + list(_OUTSIDE_XY)
    labels = list(inside_labels) + [LABEL_BACKGROUND] * len(_OUTSIDE_XY)
    if stray_outside_label is not None:
        labels[len(_INSIDE_XY)] = stray_outside_label
    if not has_bbox:
        return xy, [LABEL_BACKGROUND] * len(xy), None
    bbox = _BBOX_XYXY if bbox_overlaps_points else _BBOX_XYXY_NO_OVERLAP
    if not bbox_overlaps_points:
        labels = [LABEL_BACKGROUND] * len(xy)
    return xy, labels, bbox


def make_bbox_noncontour_dummy_h5(path: Path, *, seed: int, inject_stray_ignore: bool = False) -> dict[str, Any]:
    """4 frames x 12 points, deterministic BBox/label geometry.

    frame 0: no BBox at all -> all background.
    frame 1: BBox covers indices 0-3; 2 positive + 2 ignore (the
        bbox_noncontour_background conversion targets) inside, 8 background
        outside (or, if inject_stray_ignore, one of the "outside" points is
        also ignore -- a contract violation the policy must refuse to fix up).
    frame 2: BBox stored but does not overlap any point (edge case).
    frame 3: no BBox at all -> all background.

    Returns the expected ground-truth counts for assertions.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame_specs = [
        _frame_layout(has_bbox=False, bbox_overlaps_points=False, inside_labels=[], stray_outside_label=None),
        _frame_layout(
            has_bbox=True,
            bbox_overlaps_points=True,
            inside_labels=[LABEL_POSITIVE, LABEL_POSITIVE, LABEL_IGNORE, LABEL_IGNORE],
            stray_outside_label=LABEL_IGNORE if inject_stray_ignore else None,
        ),
        _frame_layout(has_bbox=True, bbox_overlaps_points=False, inside_labels=[], stray_outside_label=None),
        _frame_layout(has_bbox=False, bbox_overlaps_points=False, inside_labels=[], stray_outside_label=None),
    ]

    frame_order: list[int] = []
    pixel_xy: list[tuple[int, int]] = []
    point_label: list[int] = []
    bbox_frame_order: list[int] = []
    bbox_local_xyxy: list[tuple[float, float, float, float]] = []
    for frame_index, (xy, labels, bbox) in enumerate(frame_specs):
        frame_order.extend([frame_index] * len(xy))
        pixel_xy.extend(xy)
        point_label.extend(labels)
        if bbox is not None:
            bbox_frame_order.append(frame_index)
            bbox_local_xyxy.append(bbox)

    frame_order_arr = np.asarray(frame_order, dtype=np.int32)
    pixel_xy_arr = np.asarray(pixel_xy, dtype=np.float32)
    point_label_arr = np.asarray(point_label, dtype=np.int8)
    valid_mask_arr = point_label_arr != LABEL_IGNORE
    num_points = frame_order_arr.shape[0]

    with h5py.File(path, "w") as f:
        point_cloud = f.create_group("point_cloud")
        point_cloud.create_dataset("points", data=rng.normal(size=(num_points, 3)).astype(np.float32))
        point_cloud.create_dataset("intensity", data=rng.integers(0, 256, size=num_points, dtype=np.uint8))
        point_cloud.create_dataset("alpha", data=rng.integers(0, 256, size=num_points, dtype=np.uint8))
        point_cloud.create_dataset("confidence", data=rng.random(size=num_points).astype(np.float32))
        point_cloud.create_dataset("frame_order", data=frame_order_arr)
        point_cloud.create_dataset("pixel_xy", data=pixel_xy_arr)

        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=point_label_arr)
        annotation.create_dataset("valid_mask", data=valid_mask_arr)

        frame_annotation = f.create_group("frame_annotation")
        frame_annotation.create_dataset("frame_order", data=np.asarray(bbox_frame_order, dtype=np.int64))
        frame_annotation.create_dataset(
            "bbox_local_xyxy", data=np.asarray(bbox_local_xyxy, dtype=np.float64).reshape(-1, 4)
        )

    target_count = 2 if not inject_stray_ignore else 2  # frame 1's in-BBox ignore points
    stray_count = 1 if inject_stray_ignore else 0
    return {
        "num_points": num_points,
        "positive_count": int(np.sum(point_label_arr == LABEL_POSITIVE)),
        "source_background_count": int(np.sum(point_label_arr == LABEL_BACKGROUND)),
        "source_ignore_count": int(np.sum(point_label_arr == LABEL_IGNORE)),
        "expected_target_count": target_count,
        "expected_stray_count": stray_count,
    }


# ----------------------------------------------------------------------------
# Pure-array tests of stage5.utils.label_policy (no H5/Dataset involved).
# ----------------------------------------------------------------------------


def test_compute_bbox_inside_mask_matches_hand_geometry() -> None:
    frame_order = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    pixel_xy = np.array([[5, 5], [50, 50], [5, 5], [500, 500], [1, 1], [1, 1]], dtype=np.float32)
    bbox_frame_order = np.array([0], dtype=np.int64)
    bbox_local_xyxy = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)

    mask = compute_bbox_inside_mask(
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=bbox_frame_order,
        bbox_local_xyxy=bbox_local_xyxy,
    )
    expected = np.array([True, False, True, False, False, False])
    require(np.array_equal(mask, expected), f"bbox_inside_mask mismatch: {mask} != {expected}")


def test_compute_bbox_inside_mask_empty_bbox_arrays() -> None:
    frame_order = np.array([0, 0], dtype=np.int64)
    pixel_xy = np.array([[1, 1], [2, 2]], dtype=np.float32)
    mask = compute_bbox_inside_mask(
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=np.array([], dtype=np.int64),
        bbox_local_xyxy=np.zeros((0, 4), dtype=np.float64),
    )
    require(not mask.any(), "No BBoxes at all must produce an all-False mask")


def test_compute_bbox_inside_mask_rejects_zero_area_boxes() -> None:
    frame_order = np.zeros(3, dtype=np.int64)
    pixel_xy = np.array([[5, 2], [2, 2], [3, 3]], dtype=np.float32)
    mask = compute_bbox_inside_mask(
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=np.array([0, 0], dtype=np.int64),
        bbox_local_xyxy=np.array(
            [[5.0, 1.0, 5.0, 4.0], [2.2, 1.0, 2.2, 4.0]],
            dtype=np.float64,
        ),
    )
    require(not mask.any(), "Zero-area BBoxes must not become one-pixel lines")


def test_ignore_policy_is_a_strict_noop() -> None:
    point_label = np.array([LABEL_BACKGROUND, LABEL_POSITIVE, LABEL_IGNORE, LABEL_IGNORE], dtype=np.int64)
    valid_mask = point_label != LABEL_IGNORE
    effective_label, effective_valid_mask, stats = apply_bbox_noncontour_label_policy(
        "bbox_noncontour_ignore", point_label=point_label, valid_mask=valid_mask
    )
    require(np.array_equal(effective_label, point_label), "ignore policy must not change labels")
    require(np.array_equal(effective_valid_mask, valid_mask), "ignore policy must not change valid_mask")
    require(stats["converted_point_count"] == 0, "ignore policy must convert 0 points")
    # Returned arrays must be independent copies, not aliases of the input.
    effective_label[0] = LABEL_POSITIVE
    require(point_label[0] == LABEL_BACKGROUND, "apply_bbox_noncontour_label_policy must not alias its input arrays")


def test_background_policy_converts_only_audited_targets() -> None:
    frame_order = np.array([0, 0, 0, 0], dtype=np.int64)
    pixel_xy = np.array([[5, 5], [5, 5], [500, 500], [5, 5]], dtype=np.float32)
    bbox_frame_order = np.array([0], dtype=np.int64)
    bbox_local_xyxy = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
    # index0: inside bbox, positive (must stay positive)
    # index1: inside bbox, ignore (bbox non-contour target -> background)
    # index2: outside bbox, background (must stay background)
    # index3: inside bbox, background (already background, no-op)
    point_label = np.array([LABEL_POSITIVE, LABEL_IGNORE, LABEL_BACKGROUND, LABEL_BACKGROUND], dtype=np.int64)
    valid_mask = point_label != LABEL_IGNORE

    effective_label, effective_valid_mask, stats = apply_bbox_noncontour_label_policy(
        "bbox_noncontour_background",
        point_label=point_label,
        valid_mask=valid_mask,
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=bbox_frame_order,
        bbox_local_xyxy=bbox_local_xyxy,
    )
    require(
        np.array_equal(effective_label, np.array([LABEL_POSITIVE, LABEL_BACKGROUND, LABEL_BACKGROUND, LABEL_BACKGROUND])),
        f"unexpected effective labels: {effective_label}",
    )
    require(bool(effective_valid_mask.all()), "background policy must make the converted point valid")
    require(stats["converted_point_count"] == 1, f"expected exactly 1 conversion, got {stats['converted_point_count']}")
    require(stats["stray_ignore_outside_bbox_count"] == 0, "no stray ignore points expected in this fixture")
    require(point_label[1] == LABEL_IGNORE, "source point_label must not be mutated in place")


def test_background_policy_rejects_stray_ignore_outside_bbox() -> None:
    frame_order = np.array([0, 0], dtype=np.int64)
    pixel_xy = np.array([[5, 5], [500, 500]], dtype=np.float32)
    bbox_frame_order = np.array([0], dtype=np.int64)
    bbox_local_xyxy = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
    # index1 is ignore but outside the only stored BBox: contract violation.
    point_label = np.array([LABEL_POSITIVE, LABEL_IGNORE], dtype=np.int64)
    valid_mask = point_label != LABEL_IGNORE

    raised = False
    try:
        apply_bbox_noncontour_label_policy(
            "bbox_noncontour_background",
            point_label=point_label,
            valid_mask=valid_mask,
            frame_order=frame_order,
            pixel_xy=pixel_xy,
            bbox_frame_order=bbox_frame_order,
            bbox_local_xyxy=bbox_local_xyxy,
        )
    except AssertionError:
        raised = True
    require(raised, "A stray ignore point outside any BBox must be rejected, not silently converted")


def test_background_policy_requires_bbox_arrays() -> None:
    point_label = np.array([LABEL_IGNORE], dtype=np.int64)
    valid_mask = point_label != LABEL_IGNORE
    raised = False
    try:
        apply_bbox_noncontour_label_policy(
            "bbox_noncontour_background", point_label=point_label, valid_mask=valid_mask
        )
    except ValueError:
        raised = True
    require(raised, "bbox_noncontour_background without BBox arrays must raise ValueError, not silently no-op")


def test_unsupported_policy_rejected() -> None:
    point_label = np.array([LABEL_BACKGROUND], dtype=np.int64)
    valid_mask = np.array([True])
    raised = False
    try:
        apply_bbox_noncontour_label_policy("bbox_all_background", point_label=point_label, valid_mask=valid_mask)
    except ValueError:
        raised = True
    require(raised, "Unknown label_policy values must raise ValueError")


# ----------------------------------------------------------------------------
# Dataset-level integration tests (H5 + Pseudo3DPointCloudDataset, CPU only).
# ----------------------------------------------------------------------------


def _collect_all_samples(dataset: Pseudo3DPointCloudDataset) -> list[dict[str, Any]]:
    return [dataset[i] for i in range(len(dataset))]


def test_run_a_dataset_matches_raw_h5(work_dir: Path) -> None:
    h5_path = work_dir / "label_policy_valid.h5"
    make_bbox_noncontour_dummy_h5(h5_path, seed=1)

    with h5py.File(h5_path, "r") as f:
        raw_label = f["annotation/point_label"][:].astype(np.int64)
        raw_valid = f["annotation/valid_mask"][:].astype(bool)

    dataset = Pseudo3DPointCloudDataset(
        [h5_path],
        num_points=None,
        features="intensity,confidence",
        window_mode="overlap",
        window_size_frames=2,
        window_stride_frames=1,
        include_tail_window=True,
        label_policy="bbox_noncontour_ignore",
    )
    for sample in _collect_all_samples(dataset):
        indices = sample["point_indices"].numpy()
        require(
            np.array_equal(sample["labels"].numpy(), raw_label[indices]),
            "bbox_noncontour_ignore Dataset labels must equal raw H5 point_label (no-op)",
        )
        require(
            np.array_equal(sample["valid_mask"].numpy(), raw_valid[indices]),
            "bbox_noncontour_ignore Dataset valid_mask must equal raw H5 valid_mask (no-op)",
        )


def test_run_b_dataset_converts_only_target_points(work_dir: Path) -> None:
    h5_path = work_dir / "label_policy_valid.h5"
    expected = make_bbox_noncontour_dummy_h5(h5_path, seed=2)

    with h5py.File(h5_path, "r") as f:
        raw_label = f["annotation/point_label"][:].astype(np.int64)
        raw_valid = f["annotation/valid_mask"][:].astype(bool)
        frame_order = f["point_cloud/frame_order"][:]
        pixel_xy = f["point_cloud/pixel_xy"][:]
        bbox_frame_order = f["frame_annotation/frame_order"][:]
        bbox_local_xyxy = f["frame_annotation/bbox_local_xyxy"][:]

    bbox_inside_mask = compute_bbox_inside_mask(
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=bbox_frame_order,
        bbox_local_xyxy=bbox_local_xyxy,
    )
    expected_target_mask = (raw_label == LABEL_IGNORE) & bbox_inside_mask
    require(
        int(expected_target_mask.sum()) == expected["expected_target_count"],
        f"fixture sanity check failed: {int(expected_target_mask.sum())} != {expected['expected_target_count']}",
    )

    def build(policy: str) -> Pseudo3DPointCloudDataset:
        return Pseudo3DPointCloudDataset(
            [h5_path],
            num_points=None,
            features="intensity,confidence",
            window_mode="overlap",
            window_size_frames=2,
            window_stride_frames=1,
            include_tail_window=True,
            label_policy=policy,
        )

    dataset_a = build("bbox_noncontour_ignore")
    dataset_b = build("bbox_noncontour_background")
    require(len(dataset_a) == len(dataset_b), "Run A/B must produce the same number of windows")

    positive_indices_a = set(np.flatnonzero(raw_label == LABEL_POSITIVE).tolist())
    total_converted = 0
    for sample_a, sample_b in zip(_collect_all_samples(dataset_a), _collect_all_samples(dataset_b)):
        require(
            torch.equal(sample_a["points"], sample_b["points"]), "Run A/B points must be identical"
        )
        require(
            torch.equal(sample_a["features"], sample_b["features"]), "Run A/B features must be identical"
        )
        require(
            torch.equal(sample_a["frame_order"], sample_b["frame_order"]),
            "Run A/B frame_order must be identical",
        )
        require(
            torch.equal(sample_a["point_indices"], sample_b["point_indices"]),
            "Run A/B point_indices (window membership) must be identical",
        )
        require(
            sample_a["window_start"] == sample_b["window_start"]
            and sample_a["window_end"] == sample_b["window_end"],
            "Run A/B window boundaries must be identical",
        )

        indices = sample_a["point_indices"].numpy()
        label_a = sample_a["labels"].numpy()
        label_b = sample_b["labels"].numpy()
        valid_a = sample_a["valid_mask"].numpy()
        valid_b = sample_b["valid_mask"].numpy()

        require(
            np.array_equal(label_a, raw_label[indices]),
            "Run A labels must still equal raw source labels",
        )
        diff_mask = label_a != label_b
        require(
            bool(np.all(expected_target_mask[indices[diff_mask]])) if diff_mask.any() else True,
            "Run A/B labels may only differ at audited BBox non-contour target points",
        )
        require(
            bool(np.all(label_b[diff_mask] == LABEL_BACKGROUND)) if diff_mask.any() else True,
            "Every changed point must become background under bbox_noncontour_background",
        )
        require(
            np.array_equal(valid_b, valid_a | expected_target_mask[indices]),
            "Run B valid_mask must equal Run A valid_mask with only target points added",
        )
        sample_positive_b = set(indices[label_b == LABEL_POSITIVE].tolist())
        require(
            sample_positive_b == (positive_indices_a & set(indices.tolist())),
            "Run B must not change which points are positive",
        )
        total_converted += int(diff_mask.sum())

    require(
        total_converted >= expected["expected_target_count"],
        "Overlapping windows should cover every converted target point at least once "
        f"(saw {total_converted}, expected >= {expected['expected_target_count']})",
    )


def test_dataset_construction_rejects_stray_ignore_h5(work_dir: Path) -> None:
    h5_path = work_dir / "label_policy_invalid.h5"
    make_bbox_noncontour_dummy_h5(h5_path, seed=3, inject_stray_ignore=True)

    raised = False
    try:
        dataset = Pseudo3DPointCloudDataset(
            [h5_path],
            num_points=None,
            features="intensity,confidence",
            window_mode="overlap",
            window_size_frames=2,
            window_stride_frames=1,
            include_tail_window=True,
            label_policy="bbox_noncontour_background",
        )
        _collect_all_samples(dataset)
    except AssertionError:
        raised = True
    require(raised, "A stray ignore point outside any BBox must fail fast when policy=bbox_noncontour_background")


def run_self_test(work_dir: Path) -> None:
    test_compute_bbox_inside_mask_matches_hand_geometry()
    test_compute_bbox_inside_mask_empty_bbox_arrays()
    test_compute_bbox_inside_mask_rejects_zero_area_boxes()
    test_ignore_policy_is_a_strict_noop()
    test_background_policy_converts_only_audited_targets()
    test_background_policy_rejects_stray_ignore_outside_bbox()
    test_background_policy_requires_bbox_arrays()
    test_unsupported_policy_rejected()
    test_run_a_dataset_matches_raw_h5(work_dir)
    test_run_b_dataset_converts_only_target_points(work_dir)
    test_dataset_construction_rejects_stray_ignore_h5(work_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-12 Step E2: synthetic label-policy and Dataset parity tests (CPU only)."
    )
    parser.add_argument("--work_dir", default=str(REPO_ROOT / "work_dirs" / "_dummy_label_policy_check"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    run_self_test(work_dir)
    print("Stage5 label-policy synthetic test passed.")
    print(f"work_dir: {work_dir}")


if __name__ == "__main__":
    main()
