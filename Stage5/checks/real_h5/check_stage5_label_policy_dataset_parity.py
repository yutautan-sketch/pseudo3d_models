from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from checks.real_h5.check_stage5_batch_integrity import array_sha256
from stage5.datasets import Pseudo3DPointCloudDataset
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list
from stage5.utils.label_policy import (
    LABEL_BACKGROUND,
    LABEL_IGNORE,
    LABEL_POSITIVE,
    compute_bbox_inside_mask,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------
# S5-12 Step E4: on real teacher v6 H5s, confirm that switching
# --label_policy between bbox_noncontour_ignore (Run A) and
# bbox_noncontour_background (Run B) changes nothing except labels/valid_mask
# at the audited BBox non-contour target points, and that Run A is a byte
# exact no-op against the raw source H5. CPU/h5py only -- no CUDA needed.
# ----------------------------------------------------------------------------

NON_LABEL_FIELDS = ("points", "features", "frame_order", "point_indices")


def build_dataset(h5_path: Path, *, policy: str, window_size_frames: int, window_stride_frames: int) -> Pseudo3DPointCloudDataset:
    return Pseudo3DPointCloudDataset(
        [h5_path],
        num_points=None,
        features="intensity,confidence",
        window_mode="overlap",
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=True,
        label_policy=policy,
    )


def check_file(
    path: Path, *, window_size_frames: int, window_stride_frames: int
) -> dict[str, Any]:
    raw = load_stage5_pointcloud_h5(path)
    require(
        "bbox_frame_order" in raw and "bbox_local_xyxy" in raw,
        f"{path}: missing frame_annotation BBox data required for Run B",
    )
    bbox_inside_mask = compute_bbox_inside_mask(
        frame_order=raw["frame_order"],
        pixel_xy=raw["pixel_xy"],
        bbox_frame_order=raw["bbox_frame_order"],
        bbox_local_xyxy=raw["bbox_local_xyxy"],
    )
    expected_target_mask = (raw["point_label"] == LABEL_IGNORE) & bbox_inside_mask

    dataset_a = build_dataset(
        path, policy="bbox_noncontour_ignore", window_size_frames=window_size_frames, window_stride_frames=window_stride_frames
    )
    dataset_b = build_dataset(
        path, policy="bbox_noncontour_background", window_size_frames=window_size_frames, window_stride_frames=window_stride_frames
    )
    require(len(dataset_a) == len(dataset_b), f"{path}: Run A/B produced different window counts")

    positive_index_set = set(np.flatnonzero(raw["point_label"] == LABEL_POSITIVE).tolist())
    seen_target_indices: set[int] = set()
    label_disagreement_beyond_target = 0

    for window_index in range(len(dataset_a)):
        sample_a = dataset_a[window_index]
        sample_b = dataset_b[window_index]

        for field in NON_LABEL_FIELDS:
            hash_a = array_sha256(sample_a[field].numpy())
            hash_b = array_sha256(sample_b[field].numpy())
            require(hash_a == hash_b, f"{path} window {window_index}: {field} differs between Run A/B")
        require(
            sample_a["window_start"] == sample_b["window_start"]
            and sample_a["window_end"] == sample_b["window_end"],
            f"{path} window {window_index}: window bounds differ between Run A/B",
        )

        indices = sample_a["point_indices"].numpy()
        label_a = sample_a["labels"].numpy()
        label_b = sample_b["labels"].numpy()
        valid_a = sample_a["valid_mask"].numpy()
        valid_b = sample_b["valid_mask"].numpy()

        require(
            np.array_equal(label_a, raw["point_label"][indices]),
            f"{path} window {window_index}: Run A labels diverge from raw source (must be a no-op)",
        )
        require(
            np.array_equal(valid_a, raw["valid_mask"][indices]),
            f"{path} window {window_index}: Run A valid_mask diverges from raw source (must be a no-op)",
        )

        diff_mask = label_a != label_b
        local_target_mask = expected_target_mask[indices]
        if diff_mask.any() and not np.all(local_target_mask[diff_mask]):
            label_disagreement_beyond_target += int(np.sum(diff_mask & ~local_target_mask))
        require(
            bool(np.all(label_b[diff_mask] == LABEL_BACKGROUND)) if diff_mask.any() else True,
            f"{path} window {window_index}: a changed point did not become background under Run B",
        )
        require(
            np.array_equal(valid_b, valid_a | local_target_mask),
            f"{path} window {window_index}: Run B valid_mask is not (Run A valid_mask | target mask)",
        )
        sample_positive_b = set(indices[label_b == LABEL_POSITIVE].tolist())
        require(
            sample_positive_b == (positive_index_set & set(indices.tolist())),
            f"{path} window {window_index}: Run B changed which points are positive",
        )
        seen_target_indices.update(indices[diff_mask].tolist())

    require(
        label_disagreement_beyond_target == 0,
        f"{path}: {label_disagreement_beyond_target} point(s) changed outside the audited target mask",
    )
    expected_target_indices = set(np.flatnonzero(expected_target_mask).tolist())
    require(
        expected_target_indices.issubset(seen_target_indices),
        f"{path}: {len(expected_target_indices - seen_target_indices)} target point(s) were never "
        "covered by any window (unexpected given overlap windows should cover all points)",
    )

    return {
        "point_count": int(raw["point_label"].size),
        "window_count": len(dataset_a),
        "expected_target_point_count": len(expected_target_indices),
        "target_points_observed_converted": len(seen_target_indices),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-12 Step E4: Dataset parity between bbox_noncontour_ignore (Run A) and "
        "bbox_noncontour_background (Run B) on real teacher v6 H5s."
    )
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument("--max_files", type=int, default=0, help="0 means all files in the given list(s)")
    parser.add_argument("--window_size_frames", type=int, default=16)
    parser.add_argument("--window_stride_frames", type=int, default=8)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(bool(args.train_list) or bool(args.val_list), "Provide --train_list and/or --val_list")

    paths: list[Path] = []
    for list_path in (args.train_list, args.val_list):
        if list_path:
            paths.extend(read_path_list(list_path))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    require(bool(paths), "No H5 paths resolved from the given list(s)")

    results: list[dict[str, Any]] = []
    for file_index, path in enumerate(tqdm(paths, desc="label policy dataset parity")):
        require(path.is_file(), f"H5 not found: {path}")
        results.append(
            {
                "video_alias": f"file_{file_index:03d}",
                **check_file(
                    path,
                    window_size_frames=args.window_size_frames,
                    window_stride_frames=args.window_stride_frames,
                ),
            }
        )

    summary = {
        "status": "passed",
        "num_files": len(results),
        "total_points": sum(row["point_count"] for row in results),
        "total_windows": sum(row["window_count"] for row in results),
        "total_expected_target_points": sum(row["expected_target_point_count"] for row in results),
    }
    if args.output_json:
        with Path(args.output_json).open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_file": results}, f, indent=2, ensure_ascii=False)

    print("Stage5 label-policy Dataset parity check passed.")
    print(f"files checked: {summary['num_files']}")
    print(f"total windows: {summary['total_windows']}")
    print(f"total expected BBox non-contour target points: {summary['total_expected_target_points']}")
    if args.output_json:
        print(f"output: {args.output_json}")


if __name__ == "__main__":
    main()
