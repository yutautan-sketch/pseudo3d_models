from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from checks.visualization.check_window_batch_ply_export import label_colors, write_ply
from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model, list_stage5_models
from stage5.training import build_loss


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def load_raw_points(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "point_cloud" not in f or "points" not in f["point_cloud"]:
            raise KeyError(f"point_cloud/points not found in {h5_path}")
        return f["point_cloud/points"][:].astype(np.float32)


def prediction_colors(predictions: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((predictions.shape[0], 3), dtype=np.uint8)
    colors[predictions == 0] = np.asarray([170, 170, 170], dtype=np.uint8)
    colors[predictions == 1] = np.asarray([230, 50, 40], dtype=np.uint8)
    colors[~valid_mask.astype(bool)] = np.asarray([20, 40, 100], dtype=np.uint8)
    return colors


def probability_colors(probabilities: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    low = np.asarray([45, 95, 220], dtype=np.float32)
    mid = np.asarray([70, 220, 120], dtype=np.float32)
    high = np.asarray([235, 55, 45], dtype=np.float32)
    colors = np.empty((values.shape[0], 3), dtype=np.float32)
    first_half = values <= 0.5
    colors[first_half] = low + (mid - low) * (values[first_half, None] * 2.0)
    colors[~first_half] = mid + (high - mid) * ((values[~first_half, None] - 0.5) * 2.0)
    colors[~valid_mask.astype(bool)] = np.asarray([20, 40, 100], dtype=np.float32)
    return np.clip(colors, 0, 255).astype(np.uint8)


def validate_window(
    *,
    frame_order: np.ndarray,
    point_indices: np.ndarray,
    window_start: int,
    window_end: int,
    batch_id: int,
    item_id: int,
) -> None:
    if frame_order.size == 0:
        raise AssertionError(f"Empty window at batch={batch_id}, item={item_id}")
    if frame_order.min() < window_start or frame_order.max() > window_end:
        raise AssertionError(
            f"Frame order outside window at batch={batch_id}, item={item_id}: "
            f"min={frame_order.min()}, max={frame_order.max()}, "
            f"window=[{window_start}, {window_end}]"
        )
    if np.any(point_indices < 0):
        raise AssertionError(f"Negative point index found before padding at batch={batch_id}, item={item_id}")


def export_window_plys(
    *,
    output_dir: Path,
    batch_id: int,
    item_id: int,
    window_id: int,
    window_start: int,
    window_end: int,
    points: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    predictions: np.ndarray,
    femur_probabilities: np.ndarray,
) -> dict[str, str]:
    stem = (
        f"batch{batch_id:03d}_item{item_id:03d}_window{window_id:03d}_"
        f"f{window_start:04d}-{window_end:04d}_n{points.shape[0]:06d}"
    )
    gt_path = output_dir / f"{stem}_gt.ply"
    pred_path = output_dir / f"{stem}_pred.ply"
    prob_path = output_dir / f"{stem}_prob_femur.ply"

    write_ply(gt_path, points, label_colors(labels, valid_mask))
    write_ply(pred_path, points, prediction_colors(predictions, valid_mask))
    write_ply(prob_path, points, probability_colors(femur_probabilities, valid_mask))

    return {
        "gt_ply_path": str(gt_path),
        "pred_ply_path": str(pred_path),
        "prob_ply_path": str(prob_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward-check Stage5 pointnext_s partial-transfer checkpoint on a real H5 window batch"
    )
    parser.add_argument("--input_h5", required=True, help="Annotated Stage5 point-cloud H5")
    parser.add_argument("--checkpoint", required=True, help="Stage5 pointnext_s checkpoint to load")
    parser.add_argument("--output_dir", required=True, help="Directory for PLY files and summary outputs")
    parser.add_argument("--features", default="intensity,confidence")
    parser.add_argument("--window_size_frames", type=int, default=12)
    parser.add_argument("--window_stride_frames", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=2, help="0 means all batches")
    parser.add_argument("--max_export_windows", type=int, default=8, help="0 means export all processed windows")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize_points", action="store_true", help="Use normalized xyz as model input")
    parser.add_argument("--export_raw_points", action="store_true", help="Export original H5 xyz coordinates in PLY")
    parser.add_argument("--cache_data", action="store_true")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pointnext_radius", type=float, default=0.1)
    parser.add_argument("--pointnext_nsample", type=int, default=32)
    parser.add_argument("--pointnext_sa_layers", type=int, default=2)
    parser.add_argument("--no_pointnext_sa_use_res", dest="pointnext_sa_use_res", action="store_false")
    parser.set_defaults(pointnext_sa_use_res=True)
    parser.add_argument("--strict_checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    input_h5 = Path(args.input_h5)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device.type != "cuda":
        raise RuntimeError("PointNeXt-S CUDA ops require --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if "pointnext_s" not in list_stage5_models():
        raise RuntimeError(f"pointnext_s is not registered. Available models: {list_stage5_models()}")
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input H5 not found: {input_h5}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dataset = Pseudo3DPointCloudDataset(
        [input_h5],
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
        num_workers=args.num_workers,
        collate_fn=pad_point_window_collate,
    )

    raw_points = load_raw_points(input_h5) if args.export_raw_points else None
    model = build_stage5_model(
        "pointnext_s",
        num_classes=2,
        feature_dim=dataset.num_feature_channels,
        checkpoint=checkpoint,
        strict_checkpoint=args.strict_checkpoint,
        width=args.width,
        expansion=args.expansion,
        dropout=args.dropout,
        pointnext_radius=args.pointnext_radius,
        pointnext_nsample=args.pointnext_nsample,
        pointnext_sa_layers=args.pointnext_sa_layers,
        pointnext_sa_use_res=args.pointnext_sa_use_res,
    ).to(device)
    loss_fn = build_loss("cross_entropy", ignore_index=-1).to(device)
    model.eval()

    manifest_path = output_dir / "real_h5_pointnext_s_forward_manifest.csv"
    fieldnames = [
        "batch_id",
        "item_id",
        "window_id",
        "window_start",
        "window_end",
        "num_points",
        "valid_points",
        "gt_positive_points",
        "pred_positive_points",
        "mean_femur_probability",
        "batch_loss",
        "min_frame_order",
        "max_frame_order",
        "min_point_index",
        "max_point_index",
        "gt_ply_path",
        "pred_ply_path",
        "prob_ply_path",
    ]

    rows: list[dict[str, Any]] = []
    processed_batches = 0
    processed_windows = 0
    exported_windows = 0
    batch_losses: list[float] = []

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            if args.max_batches > 0 and batch_id >= args.max_batches:
                break

            batch_on_device = move_batch_to_device(batch, device)
            output = model(batch_on_device)
            loss_dict = loss_fn(output, batch_on_device)
            logits = output["logits"]
            expected_shape = (batch_on_device["points"].shape[0], batch_on_device["points"].shape[1], 2)
            if tuple(logits.shape) != expected_shape:
                raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)} != {expected_shape}")
            if not torch.isfinite(logits).all():
                raise AssertionError(f"logits contains NaN/Inf at batch={batch_id}")
            if not torch.isfinite(loss_dict["loss"]):
                raise AssertionError(f"loss is not finite at batch={batch_id}: {loss_dict['loss']}")

            probabilities = torch.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
            batch_loss = float(loss_dict["loss"].item())
            batch_losses.append(batch_loss)
            processed_batches += 1

            batch_size = int(batch["points"].shape[0])
            for item_id in range(batch_size):
                num_points = int(batch["num_points"][item_id].item())
                labels = tensor_to_numpy(batch["labels"][item_id, :num_points])
                valid_mask = tensor_to_numpy(batch["valid_mask"][item_id, :num_points]).astype(bool)
                frame_order = tensor_to_numpy(batch["frame_order"][item_id, :num_points])
                point_indices = tensor_to_numpy(batch["point_indices"][item_id, :num_points])
                window_start = int(batch["window_start"][item_id].item())
                window_end = int(batch["window_end"][item_id].item())
                meta = batch["meta"][item_id]
                window_id = int(meta.get("window_id", -1))

                validate_window(
                    frame_order=frame_order,
                    point_indices=point_indices,
                    window_start=window_start,
                    window_end=window_end,
                    batch_id=batch_id,
                    item_id=item_id,
                )

                if raw_points is not None:
                    points = raw_points[point_indices]
                else:
                    points = tensor_to_numpy(batch["points"][item_id, :num_points])

                item_predictions = tensor_to_numpy(predictions[item_id, :num_points])
                femur_probabilities = tensor_to_numpy(probabilities[item_id, :num_points, 1])
                valid_labels = labels[valid_mask]
                valid_predictions = item_predictions[valid_mask]

                should_export = args.max_export_windows == 0 or exported_windows < args.max_export_windows
                ply_paths = {"gt_ply_path": "", "pred_ply_path": "", "prob_ply_path": ""}
                if should_export:
                    ply_paths = export_window_plys(
                        output_dir=output_dir,
                        batch_id=batch_id,
                        item_id=item_id,
                        window_id=window_id,
                        window_start=window_start,
                        window_end=window_end,
                        points=points,
                        labels=labels,
                        valid_mask=valid_mask,
                        predictions=item_predictions,
                        femur_probabilities=femur_probabilities,
                    )
                    exported_windows += 1

                rows.append(
                    {
                        "batch_id": batch_id,
                        "item_id": item_id,
                        "window_id": window_id,
                        "window_start": window_start,
                        "window_end": window_end,
                        "num_points": num_points,
                        "valid_points": int(valid_mask.sum()),
                        "gt_positive_points": int((valid_labels == 1).sum()),
                        "pred_positive_points": int((valid_predictions == 1).sum()),
                        "mean_femur_probability": float(femur_probabilities[valid_mask].mean())
                        if valid_mask.any()
                        else 0.0,
                        "batch_loss": batch_loss,
                        "min_frame_order": int(frame_order.min()),
                        "max_frame_order": int(frame_order.max()),
                        "min_point_index": int(point_indices.min()),
                        "max_point_index": int(point_indices.max()),
                        **ply_paths,
                    }
                )
                processed_windows += 1

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_h5": str(input_h5),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "num_dataset_windows": len(dataset),
        "processed_batches": processed_batches,
        "processed_windows": processed_windows,
        "exported_windows": exported_windows,
        "batch_size": args.batch_size,
        "window_size_frames": args.window_size_frames,
        "window_stride_frames": args.window_stride_frames,
        "features": args.features,
        "feature_dim": dataset.num_feature_channels,
        "normalize_points": args.normalize_points,
        "export_raw_points": args.export_raw_points,
        "avg_loss": float(np.mean(batch_losses)) if batch_losses else None,
        "min_loss": float(np.min(batch_losses)) if batch_losses else None,
        "max_loss": float(np.max(batch_losses)) if batch_losses else None,
    }
    summary_path = output_dir / "real_h5_pointnext_s_forward_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 real-H5 PointNeXt-S transfer forward check passed.")
    print(f"input_h5: {input_h5}")
    print(f"checkpoint: {checkpoint}")
    print(f"num_dataset_windows: {len(dataset)}")
    print(f"processed_batches: {processed_batches}")
    print(f"processed_windows: {processed_windows}")
    print(f"exported_windows: {exported_windows}")
    print(f"avg_loss: {summary['avg_loss']}")
    print(f"output_dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
