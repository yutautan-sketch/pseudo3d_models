from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model, list_stage5_models
from stage5.training import build_loss


def make_window_dummy_h5(
    path: Path,
    *,
    num_frames: int,
    points_per_frame: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame_order = np.repeat(np.arange(num_frames, dtype=np.int32), points_per_frame)
    num_points = int(frame_order.shape[0])
    pixel_xy = np.stack(
        [
            rng.integers(0, 512, size=num_points),
            rng.integers(0, 512, size=num_points),
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

        labels = rng.choice([-1, 0, 1], size=num_points, p=[0.05, 0.65, 0.30]).astype(np.int8)
        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=labels)
        annotation.create_dataset("valid_mask", data=(labels != -1))


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage5 PointNeXt-S wrapper forward pass")
    parser.add_argument("--work_dir", default=str(REPO_ROOT / "_dummy_pointnext_s_forward_check"))
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--points_per_frame", type=int, default=80)
    parser.add_argument("--window_size_frames", type=int, default=8)
    parser.add_argument("--window_stride_frames", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--nsample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("PointNeXt-S CUDA ops require --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if "pointnext_s" not in list_stage5_models():
        raise RuntimeError(f"pointnext_s is not registered. Available models: {list_stage5_models()}")

    h5_path = Path(args.work_dir) / "dummy_pointnext_s_pointcloud_annotated_grid.h5"
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_point_window_collate,
    )
    batch = move_batch_to_device(next(iter(loader)), device)

    model = build_stage5_model(
        "pointnext_s",
        num_classes=2,
        feature_dim=dataset.num_feature_channels,
        width=args.width,
        expansion=args.expansion,
        dropout=args.dropout,
        radius=args.radius,
        nsample=args.nsample,
    ).to(device)
    model.eval()

    loss_fn = build_loss("cross_entropy", ignore_index=-1).to(device)
    with torch.no_grad():
        output = model(batch)
        loss_dict = loss_fn(output, batch)

    logits = output["logits"]
    expected_shape = (batch["points"].shape[0], batch["points"].shape[1], 2)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)} != {expected_shape}")
    if not torch.isfinite(logits).all():
        raise AssertionError("logits contains NaN/Inf")
    if not torch.isfinite(loss_dict["loss"]):
        raise AssertionError(f"loss is not finite: {loss_dict['loss']}")

    print("PointNeXt-S Stage5 wrapper forward check passed.")
    print(f"registered models: {list_stage5_models()}")
    print(f"num_windows: {len(dataset)}")
    print(f"batch points shape: {tuple(batch['points'].shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {float(loss_dict['loss'].item()):.6f}")
    print(f"work_dir: {Path(args.work_dir)}")


if __name__ == "__main__":
    main()
