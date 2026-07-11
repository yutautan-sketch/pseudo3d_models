from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from check_s3dis_pointnext_s_strict_load import extract_state_dict
from stage5.models.pointnext_s_segmentor import (
    _ensure_openpoints_importable,
    build_pointnext_s_config,
)


S3DIS_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
)

S3DIS_CMAP = np.asarray(
    [
        [0, 255, 0],
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [100, 100, 255],
        [200, 200, 100],
        [255, 0, 0],
        [170, 120, 200],
        [10, 200, 100],
        [200, 100, 100],
        [200, 200, 200],
        [50, 50, 50],
    ],
    dtype=np.uint8,
)


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


def find_room_file(data_root: Path, room_name: str | None) -> Path:
    raw_dir = data_root / "raw"
    if room_name:
        candidate = raw_dir / room_name
        if candidate.suffix != ".npy":
            candidate = raw_dir / f"{room_name}.npy"
        if not candidate.exists():
            raise FileNotFoundError(f"S3DIS room file not found: {candidate}")
        return candidate

    area5_files = sorted(raw_dir.glob("Area_5_*.npy"))
    if not area5_files:
        raise FileNotFoundError(f"No Area_5 room files found in {raw_dir}")
    # A small room keeps the smoke inference fast and easy to inspect.
    return min(area5_files, key=lambda path: path.stat().st_size)


def prepare_room_data(
    path: Path,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path).astype(np.float32)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"{path} must have shape [N, >=7], got {data.shape}")

    if max_points > 0 and data.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(data.shape[0], size=max_points, replace=False))
        data = data[indices]

    points = data[:, :3].astype(np.float32)
    rgb = data[:, 3:6].astype(np.float32)
    labels = data[:, 6].astype(np.int64)

    points_for_model = points - points.min(axis=0, keepdims=True)
    heights = points_for_model[:, 2:3].astype(np.float32)
    rgb_norm = np.clip(rgb / 255.0, 0.0, 1.0)
    features = np.concatenate([rgb_norm, heights], axis=1).astype(np.float32)

    return points_for_model, features, labels


def compute_metrics(pred: np.ndarray, labels: np.ndarray) -> tuple[float, float, list[dict[str, Any]]]:
    accuracy = float(np.mean(pred == labels)) if labels.size else 0.0
    rows: list[dict[str, Any]] = []
    ious: list[float] = []
    for class_id, name in enumerate(S3DIS_CLASSES):
        pred_mask = pred == class_id
        label_mask = labels == class_id
        intersection = int(np.logical_and(pred_mask, label_mask).sum())
        union = int(np.logical_or(pred_mask, label_mask).sum())
        label_count = int(label_mask.sum())
        pred_count = int(pred_mask.sum())
        iou = float(intersection / union) if union > 0 else float("nan")
        if union > 0:
            ious.append(iou)
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "label_count": label_count,
                "pred_count": pred_count,
                "intersection": intersection,
                "union": union,
                "iou": iou,
            }
        )
    miou = float(np.mean(ious)) if ious else 0.0
    return accuracy, miou, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PointNeXt-S S3DIS checkpoint inference on one room")
    parser.add_argument(
        "--data_root",
        default=str(REPO_ROOT / "external" / "PointNeXt" / "data" / "S3DIS" / "s3disfull"),
    )
    parser.add_argument("--room", default=None, help="Room npy filename or stem. Defaults to the smallest Area_5 room.")
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT
            / "external"
            / "PointNeXt"
            / "weight"
            / "s3dis-seg"
            / "checkpoint"
            / "s3dis-train-pointnext-s-ngpus1-seed1742-20220525-162639-Gz4ViiL95b6KP9MMH2ytG3_ckpt_best.pth"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "_s3dis_pointnext_s_single_room_inference"),
    )
    parser.add_argument("--max_points", type=int, default=24000, help="Randomly subsample room points; 0 means all")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    data_root = Path(args.data_root)
    room_path = find_room_file(data_root, args.room)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, features, labels = prepare_room_data(
        room_path,
        max_points=args.max_points,
        seed=args.seed,
    )

    _ensure_openpoints_importable()
    from openpoints.models import build_model_from_cfg

    cfg = build_pointnext_s_config(
        num_classes=13,
        feature_dim=4,
        width=32,
        expansion=4,
        dropout=0.5,
        radius=0.1,
        nsample=32,
        sa_layers=2,
        sa_use_res=True,
    )
    model = build_model_from_cfg(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()

    batch = {
        "pos": torch.from_numpy(points[None]).to(device=device, dtype=torch.float32).contiguous(),
        "x": torch.from_numpy(features.T[None]).to(device=device, dtype=torch.float32).contiguous(),
    }
    with torch.no_grad():
        logits = model(batch)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    if pred.shape[0] != labels.shape[0]:
        raise AssertionError(f"Prediction length mismatch: {pred.shape[0]} vs {labels.shape[0]}")

    accuracy, miou, rows = compute_metrics(pred, labels)
    room_stem = room_path.stem
    write_ply(output_dir / f"{room_stem}_input_rgb.ply", points, np.clip(features[:, :3] * 255, 0, 255).astype(np.uint8))
    write_ply(output_dir / f"{room_stem}_gt.ply", points, S3DIS_CMAP[labels])
    write_ply(output_dir / f"{room_stem}_pred.ply", points, S3DIS_CMAP[pred])

    metrics_path = output_dir / f"{room_stem}_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "label_count", "pred_count", "intersection", "union", "iou"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("S3DIS PointNeXt-S single-room inference passed.")
    print(f"room: {room_path}")
    print(f"num_points: {points.shape[0]}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"miou_present_classes: {miou:.4f}")
    print(f"output_dir: {output_dir}")
    print(f"metrics_csv: {metrics_path}")
    print(f"input_ply: {output_dir / f'{room_stem}_input_rgb.ply'}")
    print(f"gt_ply: {output_dir / f'{room_stem}_gt.ply'}")
    print(f"pred_ply: {output_dir / f'{room_stem}_pred.ply'}")


if __name__ == "__main__":
    main()
