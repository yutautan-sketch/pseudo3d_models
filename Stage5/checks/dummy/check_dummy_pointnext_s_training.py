from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from checks.dummy.check_dummy_pointnext_s_forward import make_window_dummy_h5
from stage5.models import build_stage5_model


def write_list_file(path: Path, h5_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h5_path in h5_paths:
            f.write(f"{h5_path}\n")


def make_dummy_dataset(
    work_dir: Path,
    *,
    num_train: int,
    num_val: int,
    num_frames: int,
    points_per_frame: int,
    seed: int,
) -> tuple[Path, Path]:
    data_dir = work_dir / "data"
    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for index in range(num_train):
        path = data_dir / "train" / f"dummy_train_{index:03d}_pointcloud_annotated_grid.h5"
        make_window_dummy_h5(
            path,
            num_frames=num_frames,
            points_per_frame=points_per_frame,
            seed=seed + index,
        )
        train_paths.append(path)

    for index in range(num_val):
        path = data_dir / "val" / f"dummy_val_{index:03d}_pointcloud_annotated_grid.h5"
        make_window_dummy_h5(
            path,
            num_frames=num_frames,
            points_per_frame=points_per_frame,
            seed=seed + 1000 + index,
        )
        val_paths.append(path)

    train_list = work_dir / "train_h5_list.txt"
    val_list = work_dir / "val_h5_list.txt"
    write_list_file(train_list, train_paths)
    write_list_file(val_list, val_paths)
    return train_list, val_list


def run_training(
    *,
    train_list: Path,
    val_list: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train_stage5.py"),
        "--train_list",
        str(train_list),
        "--val_list",
        str(val_list),
        "--output_dir",
        str(output_dir),
        "--model",
        "pointnext_s",
        "--features",
        "intensity,confidence",
        "--num_points",
        "0",
        "--window_mode",
        "overlap",
        "--window_size_frames",
        str(args.window_size_frames),
        "--window_stride_frames",
        str(args.window_stride_frames),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--width",
        str(args.width),
        "--depth",
        "1",
        "--expansion",
        str(args.expansion),
        "--dropout",
        str(args.dropout),
        "--pointnext_radius",
        str(args.radius),
        "--pointnext_nsample",
        str(args.nsample),
        "--pointnext_sa_layers",
        str(args.sa_layers),
        "--num_workers",
        "0",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--val_every",
        "1",
    ]
    if not args.sa_use_res:
        cmd.append("--no_pointnext_sa_use_res")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_training_outputs(output_dir: Path, *, expected_epochs: int) -> dict:
    required_paths = [
        output_dir / "config.json",
        output_dir / "metrics.jsonl",
        output_dir / "history.json",
        output_dir / "last.pt",
        output_dir / "best.pt",
        output_dir / "train_window_summary.csv",
        output_dir / "val_window_summary.csv",
    ]
    for path in required_paths:
        if not path.exists():
            raise AssertionError(f"Expected training artifact not found: {path}")

    with (output_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model") != "pointnext_s":
        raise AssertionError(f"Unexpected model in config.json: {config.get('model')}")
    if config.get("window_mode") != "overlap":
        raise AssertionError(f"Unexpected window_mode in config.json: {config.get('window_mode')}")
    if int(config.get("feature_dim", -1)) != 2:
        raise AssertionError(f"Unexpected feature_dim in config.json: {config.get('feature_dim')}")

    records = []
    with (output_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) != expected_epochs:
        raise AssertionError(f"Expected {expected_epochs} metric records, got {len(records)}")

    for record in records:
        train = record.get("train") or {}
        val = record.get("val") or {}
        for split, metrics in [("train", train), ("val", val)]:
            if "loss" not in metrics:
                raise AssertionError(f"{split} metrics do not contain loss")
            if not math.isfinite(float(metrics["loss"])):
                raise AssertionError(f"{split} loss is not finite: {metrics['loss']}")

    return config


def validate_checkpoint_reload(output_dir: Path, config: dict, *, device: str) -> None:
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    model = build_stage5_model(
        "pointnext_s",
        num_classes=int(config["num_classes"]),
        feature_dim=int(config["feature_dim"]),
        width=int(config["width"]),
        expansion=int(config["expansion"]),
        dropout=float(config["dropout"]),
        pointnext_radius=float(config["pointnext_radius"]),
        pointnext_nsample=int(config["pointnext_nsample"]),
        pointnext_sa_layers=int(config["pointnext_sa_layers"]),
        pointnext_sa_use_res=bool(config["pointnext_sa_use_res"]),
    )
    result = model.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(
            f"Checkpoint reload mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    if device == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage5 PointNeXt-S training CLI")
    parser.add_argument("--work_dir", default=str(REPO_ROOT / "work_dirs" / "_dummy_pointnext_s_training_check"))
    parser.add_argument("--num_train", type=int, default=2)
    parser.add_argument("--num_val", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--points_per_frame", type=int, default=64)
    parser.add_argument("--window_size_frames", type=int, default=8)
    parser.add_argument("--window_stride_frames", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--nsample", type=int, default=16)
    parser.add_argument("--sa_layers", type=int, default=2)
    parser.add_argument("--no_sa_use_res", dest="sa_use_res", action="store_false")
    parser.set_defaults(sa_use_res=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("PointNeXt-S training smoke test requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    work_dir = Path(args.work_dir)
    train_list, val_list = make_dummy_dataset(
        work_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_frames=args.num_frames,
        points_per_frame=args.points_per_frame,
        seed=args.seed,
    )

    output_dir = work_dir / "run_pointnext_s"
    run_training(
        train_list=train_list,
        val_list=val_list,
        output_dir=output_dir,
        args=args,
    )
    config = validate_training_outputs(output_dir, expected_epochs=args.epochs)
    validate_checkpoint_reload(output_dir, config, device=args.device)

    print("PointNeXt-S Stage5 training CLI check passed.")
    print(f"output_dir: {output_dir}")
    print("Generated files are kept for inspection. Remove the work_dir manually when no longer needed.")


if __name__ == "__main__":
    main()
