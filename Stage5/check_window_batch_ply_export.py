from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate


def label_colors(labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    colors[labels == 0] = np.asarray([170, 170, 170], dtype=np.uint8)
    colors[labels == 1] = np.asarray([230, 50, 40], dtype=np.uint8)
    colors[labels == -1] = np.asarray([50, 100, 220], dtype=np.uint8)
    colors[~valid_mask.astype(bool)] = np.asarray([20, 40, 100], dtype=np.uint8)
    return colors


def frame_order_colors(frame_order: np.ndarray) -> np.ndarray:
    values = frame_order.astype(np.float32)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    min_value = float(values.min())
    max_value = float(values.max())
    denom = max(max_value - min_value, 1.0)
    t = (values - min_value) / denom
    low = np.asarray([50, 110, 220], dtype=np.float32)
    mid = np.asarray([80, 220, 120], dtype=np.float32)
    high = np.asarray([230, 60, 50], dtype=np.float32)
    colors = np.empty((values.shape[0], 3), dtype=np.float32)
    first_half = t <= 0.5
    colors[first_half] = low + (mid - low) * (t[first_half, None] * 2.0)
    colors[~first_half] = mid + (high - mid) * ((t[~first_half, None] - 0.5) * 2.0)
    return np.clip(colors, 0, 255).astype(np.uint8)


def choose_colors(
    *,
    color_by: str,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    frame_order: np.ndarray,
) -> np.ndarray:
    if color_by == "label":
        return label_colors(labels, valid_mask)
    if color_by == "frame_order":
        return frame_order_colors(frame_order)
    raise ValueError(f"Unknown color_by: {color_by}")


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if colors.shape != (points.shape[0], 3):
        raise ValueError(f"colors must have shape [N, 3], got {colors.shape}")

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
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def export_batch_windows(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Pseudo3DPointCloudDataset(
        [args.input_h5],
        num_points=None,
        features=args.features,
        normalize_points=args.normalize_points,
        window_mode="overlap",
        window_size_frames=args.window_size_frames,
        window_stride_frames=args.window_stride_frames,
        include_tail_window=True,
        cache_data=args.cache_data,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_point_window_collate,
    )

    manifest_path = output_dir / "batch_window_manifest.csv"
    fieldnames = [
        "batch_id",
        "item_id",
        "window_id",
        "window_start",
        "window_end",
        "num_points",
        "min_frame_order",
        "max_frame_order",
        "min_point_index",
        "max_point_index",
        "ply_path",
    ]

    rows: list[dict[str, int | str]] = []
    exported = 0
    for batch_id, batch in enumerate(loader):
        if args.max_batches > 0 and batch_id >= args.max_batches:
            break

        batch_size = int(batch["points"].shape[0])
        for item_id in range(batch_size):
            num_points = int(batch["num_points"][item_id].item())
            points = tensor_to_numpy(batch["points"][item_id, :num_points])
            labels = tensor_to_numpy(batch["labels"][item_id, :num_points])
            valid_mask = tensor_to_numpy(batch["valid_mask"][item_id, :num_points]).astype(bool)
            frame_order = tensor_to_numpy(batch["frame_order"][item_id, :num_points])
            point_indices = tensor_to_numpy(batch["point_indices"][item_id, :num_points])
            window_start = int(batch["window_start"][item_id].item())
            window_end = int(batch["window_end"][item_id].item())
            meta = batch["meta"][item_id]
            window_id = int(meta.get("window_id", -1))

            if num_points == 0:
                raise AssertionError(f"Empty window at batch={batch_id}, item={item_id}")
            if frame_order.min() < window_start or frame_order.max() > window_end:
                raise AssertionError(
                    f"Frame order outside window at batch={batch_id}, item={item_id}: "
                    f"min={frame_order.min()}, max={frame_order.max()}, "
                    f"window=[{window_start}, {window_end}]"
                )
            if np.any(point_indices < 0):
                raise AssertionError(f"Negative point index found before padding at batch={batch_id}, item={item_id}")

            colors = choose_colors(
                color_by=args.color_by,
                labels=labels,
                valid_mask=valid_mask,
                frame_order=frame_order,
            )

            ply_name = (
                f"batch{batch_id:03d}_item{item_id:03d}_window{window_id:03d}_"
                f"f{window_start:04d}-{window_end:04d}_n{num_points:06d}.ply"
            )
            ply_path = output_dir / ply_name
            write_ply(ply_path, points, colors)

            rows.append(
                {
                    "batch_id": batch_id,
                    "item_id": item_id,
                    "window_id": window_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "num_points": num_points,
                    "min_frame_order": int(frame_order.min()),
                    "max_frame_order": int(frame_order.max()),
                    "min_point_index": int(point_indices.min()),
                    "max_point_index": int(point_indices.max()),
                    "ply_path": str(ply_path),
                }
            )
            exported += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Window batch PLY export complete.")
    print(f"input_h5: {args.input_h5}")
    print(f"output_dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"num_dataset_windows: {len(dataset)}")
    print(f"num_exported_ply: {exported}")
    print(f"normalize_points: {args.normalize_points}")
    print(f"color_by: {args.color_by}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export padded Stage5 window batches as per-window PLY files")
    parser.add_argument("--input_h5", required=True, help="Annotated Stage5 point-cloud H5")
    parser.add_argument("--output_dir", required=True, help="Directory for PLY files and manifest CSV")
    parser.add_argument("--features", default="intensity,confidence")
    parser.add_argument("--window_size_frames", type=int, default=12)
    parser.add_argument("--window_stride_frames", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=3, help="0 means all batches")
    parser.add_argument("--color_by", choices=["label", "frame_order"], default="label")
    parser.add_argument("--normalize_points", action="store_true", help="Export normalized model-input coordinates")
    parser.add_argument("--cache_data", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export_batch_windows(parse_args())

