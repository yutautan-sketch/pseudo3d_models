from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from check_dummy_pointnext_s_forward import make_window_dummy_h5, move_batch_to_device
from check_s3dis_pointnext_s_strict_load import extract_state_dict
from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model
from stage5.training import build_loss


def prefix_counter(keys: list[str], *, depth: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for key in keys:
        parts = key.split(".")
        counter[".".join(parts[:depth])] += 1
    return dict(sorted(counter.items()))


def filter_s3dis_state_for_stage5(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    filtered: dict[str, torch.Tensor] = {}
    skipped_missing: list[dict[str, Any]] = []
    skipped_shape: list[dict[str, Any]] = []

    for source_key, source_value in source_state.items():
        target_key = f"pointnext.{source_key}"
        if target_key not in target_state:
            skipped_missing.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "reason": "missing_target_key",
                    "source_shape": tuple(source_value.shape),
                    "target_shape": None,
                }
            )
            continue

        target_value = target_state[target_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped_shape.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "reason": "shape_mismatch",
                    "source_shape": tuple(source_value.shape),
                    "target_shape": tuple(target_value.shape),
                }
            )
            continue

        filtered[target_key] = source_value

    return filtered, skipped_missing, skipped_shape


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_key", "target_key", "reason", "source_shape", "target_shape"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partially transfer S3DIS PointNeXt-S weights to Stage5 pointnext_s")
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
        "--work_dir",
        default=str(REPO_ROOT / "_s3dis_to_stage5_pointnext_s_transfer"),
    )
    parser.add_argument("--features", default="intensity,confidence")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--points_per_frame", type=int, default=80)
    parser.add_argument("--window_size_frames", type=int, default=8)
    parser.add_argument("--window_stride_frames", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--nsample", type=int, default=32)
    parser.add_argument("--sa_layers", type=int, default=2)
    parser.add_argument("--no_sa_use_res", dest="sa_use_res", action="store_false")
    parser.set_defaults(sa_use_res=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("PointNeXt-S CUDA ops require --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    dummy_h5 = work_dir / "dummy_stage5_pointcloud_annotated_grid.h5"
    make_window_dummy_h5(
        dummy_h5,
        num_frames=args.num_frames,
        points_per_frame=args.points_per_frame,
        seed=args.seed,
    )
    dataset = Pseudo3DPointCloudDataset(
        [dummy_h5],
        num_points=None,
        features=args.features,
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
        pointnext_radius=args.radius,
        pointnext_nsample=args.nsample,
        pointnext_sa_layers=args.sa_layers,
        pointnext_sa_use_res=args.sa_use_res,
    ).to(device)

    source_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    source_state = extract_state_dict(source_checkpoint)
    target_state = model.state_dict()
    filtered_state, skipped_missing, skipped_shape = filter_s3dis_state_for_stage5(
        source_state,
        target_state,
    )

    load_result = model.load_state_dict(filtered_state, strict=False)
    loaded_keys = sorted(filtered_state.keys())
    loaded_source_keys = [key.removeprefix("pointnext.") for key in loaded_keys]

    loss_fn = build_loss("cross_entropy", ignore_index=-1).to(device)
    model.eval()
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

    write_rows(work_dir / "skipped_missing.csv", skipped_missing)
    write_rows(work_dir / "skipped_shape_mismatch.csv", skipped_shape)

    summary = {
        "source_checkpoint": str(args.checkpoint),
        "source_keys": len(source_state),
        "target_keys": len(target_state),
        "loaded_keys": len(loaded_keys),
        "skipped_missing": len(skipped_missing),
        "skipped_shape_mismatch": len(skipped_shape),
        "load_result_missing_keys": len(load_result.missing_keys),
        "load_result_unexpected_keys": len(load_result.unexpected_keys),
        "loaded_prefix_counts": prefix_counter(loaded_source_keys),
        "skipped_shape_prefix_counts": prefix_counter([row["source_key"] for row in skipped_shape]),
        "feature_names": args.features,
        "feature_dim": dataset.num_feature_channels,
        "logits_shape": tuple(logits.shape),
        "loss": float(loss_dict["loss"].item()),
    }
    with (work_dir / "transfer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "model": "pointnext_s",
                    "num_classes": 2,
                    "features": args.features,
                    "feature_dim": dataset.num_feature_channels,
                    "width": args.width,
                    "expansion": args.expansion,
                    "dropout": args.dropout,
                    "pointnext_radius": args.radius,
                    "pointnext_nsample": args.nsample,
                    "pointnext_sa_layers": args.sa_layers,
                    "pointnext_sa_use_res": args.sa_use_res,
                },
                "transfer": summary,
            },
            work_dir / "stage5_pointnext_s_s3dis_partial_init.pt",
        )

    print("S3DIS -> Stage5 PointNeXt-S partial transfer check passed.")
    print(f"source checkpoint: {args.checkpoint}")
    print(f"source keys: {len(source_state)}")
    print(f"target keys: {len(target_state)}")
    print(f"loaded keys: {len(loaded_keys)}")
    print(f"skipped missing: {len(skipped_missing)}")
    print(f"skipped shape mismatch: {len(skipped_shape)}")
    print(f"loaded prefix counts: {summary['loaded_prefix_counts']}")
    print(f"skipped shape prefix counts: {summary['skipped_shape_prefix_counts']}")
    print(f"batch points shape: {tuple(batch['points'].shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {float(loss_dict['loss'].item()):.6f}")
    print(f"work_dir: {work_dir}")


if __name__ == "__main__":
    main()
