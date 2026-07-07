from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model


def make_window_dummy_h5(path: Path, *, num_frames: int, points_per_frame: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame_order = np.repeat(np.arange(num_frames, dtype=np.int32), points_per_frame)
    num_points = frame_order.shape[0]
    pixel_xy = np.stack(
        [
            rng.integers(0, 256, size=num_points),
            rng.integers(0, 256, size=num_points),
        ],
        axis=1,
    ).astype(np.float32)

    with h5py.File(path, "w") as f:
        point_cloud = f.create_group("point_cloud")
        point_cloud.create_dataset("points", data=rng.normal(size=(num_points, 3)).astype(np.float32))
        point_cloud.create_dataset("intensity", data=rng.integers(0, 256, size=num_points, dtype=np.uint8))
        point_cloud.create_dataset("alpha", data=rng.integers(0, 256, size=num_points, dtype=np.uint8))
        point_cloud.create_dataset("confidence", data=rng.random(size=num_points).astype(np.float32))
        point_cloud.create_dataset("frame_order", data=frame_order)
        point_cloud.create_dataset("pixel_xy", data=pixel_xy)

        labels = rng.choice([-1, 0, 1], size=num_points, p=[0.15, 0.65, 0.20]).astype(np.int8)
        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=labels)
        annotation.create_dataset("valid_mask", data=(labels != -1))


def validate_sample(sample: dict[str, torch.Tensor], source_frame_order: np.ndarray) -> None:
    frame_order = sample["frame_order"].numpy()
    point_indices = sample["point_indices"].numpy()
    window_start = int(sample["window_start"])
    window_end = int(sample["window_end"])

    if frame_order.size == 0:
        raise AssertionError("Window sample has no points")
    if frame_order.min() < window_start or frame_order.max() > window_end:
        raise AssertionError(
            f"frame_order outside window: min={frame_order.min()} max={frame_order.max()} "
            f"window=[{window_start}, {window_end}]"
        )
    if not np.array_equal(source_frame_order[point_indices], frame_order):
        raise AssertionError("point_indices do not map back to source frame_order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage5 frame-order overlap window Dataset")
    parser.add_argument("--work_dir", default=str(REPO_ROOT / "_dummy_window_check"))
    parser.add_argument("--num_frames", type=int, default=35)
    parser.add_argument("--points_per_frame", type=int, default=8)
    parser.add_argument("--window_size_frames", type=int, default=12)
    parser.add_argument("--window_stride_frames", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    h5_path = work_dir / "dummy_window_pointcloud_annotated_grid.h5"
    make_window_dummy_h5(
        h5_path,
        num_frames=args.num_frames,
        points_per_frame=args.points_per_frame,
        seed=args.seed,
    )

    dataset = Pseudo3DPointCloudDataset(
        [h5_path],
        num_points=None,
        features="intensity,confidence",
        window_mode="overlap",
        window_size_frames=args.window_size_frames,
        window_stride_frames=args.window_stride_frames,
        include_tail_window=True,
    )

    with h5py.File(h5_path, "r") as f:
        source_frame_order = f["point_cloud/frame_order"][:]

    summaries = dataset.get_window_summaries()
    print("window_id,start_frame,end_frame,num_points")
    for row in summaries:
        print(f"{row['window_id']},{row['start_frame']},{row['end_frame']},{row['num_points']}")

    if len(dataset) < 2:
        raise AssertionError(f"Expected multiple window samples, got {len(dataset)}")

    previous_start = None
    for i in range(len(dataset)):
        sample = dataset[i]
        validate_sample(sample, source_frame_order)
        if previous_start is not None:
            actual_stride = int(sample["window_start"]) - int(previous_start)
            if actual_stride != args.window_stride_frames:
                raise AssertionError(
                    f"Unexpected window stride: {actual_stride} != {args.window_stride_frames}"
                )
        previous_start = int(sample["window_start"])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_point_window_collate,
    )
    batch = next(iter(loader))
    if batch["points"].ndim != 3 or batch["features"].ndim != 3:
        raise AssertionError("Padded batch does not have [B, N, C] tensors")
    if batch["point_indices"].shape[:2] != batch["points"].shape[:2]:
        raise AssertionError("point_indices were not padded to match points")

    model = build_stage5_model(
        "mlp_baseline",
        num_classes=2,
        feature_dim=dataset.num_feature_channels,
        width=16,
        depth=2,
        expansion=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        output = model(batch)
    if output["logits"].shape[:2] != batch["points"].shape[:2]:
        raise AssertionError(
            f"Unexpected logits shape {tuple(output['logits'].shape)} for points {tuple(batch['points'].shape)}"
        )

    print("Dummy frame-window dataset check passed.")
    print(f"num_windows: {len(dataset)}")
    print(f"batch points shape: {tuple(batch['points'].shape)}")
    print(f"logits shape: {tuple(output['logits'].shape)}")


if __name__ == "__main__":
    main()

