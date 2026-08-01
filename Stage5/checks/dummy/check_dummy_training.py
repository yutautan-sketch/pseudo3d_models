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

import h5py
import numpy as np
import torch

from checks.dummy.check_dummy_inference import make_dummy_pointcloud_h5, validate_outputs
from stage5.models import build_stage5_model
from stage5.training import compute_segmentation_metrics


CONFUSION_METRIC_KEYS = (
    "true_positive_count",
    "false_positive_count",
    "true_negative_count",
    "false_negative_count",
    "false_positive_rate",
    "false_negative_rate",
)


def validate_confusion_metric_semantics() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 4.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [4.0, 0.0],
                [0.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1, 1, 0, 0, -1]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, True, True, False]])

    ignored_metrics = compute_segmentation_metrics(
        {"logits": logits},
        {"labels": labels, "valid_mask": valid_mask},
    )
    expected_ignored = {
        "true_positive_count": 1.0,
        "false_positive_count": 1.0,
        "true_negative_count": 1.0,
        "false_negative_count": 1.0,
        "false_positive_rate": 0.5,
        "false_negative_rate": 0.5,
    }
    for key, expected in expected_ignored.items():
        if not math.isclose(float(ignored_metrics[key]), expected):
            raise AssertionError(
                f"Unexpected ignored-point metric {key}: "
                f"{ignored_metrics[key]} != {expected}"
            )

    background_labels = labels.clone()
    background_valid_mask = valid_mask.clone()
    background_labels[0, 4] = 0
    background_valid_mask[0, 4] = True
    background_metrics = compute_segmentation_metrics(
        {"logits": logits},
        {"labels": background_labels, "valid_mask": background_valid_mask},
    )
    if float(background_metrics["false_positive_count"]) != 2.0:
        raise AssertionError(
            "Relabeling an ignored positive prediction as valid background "
            "must add one false positive"
        )
    if not math.isclose(float(background_metrics["false_positive_rate"]), 2.0 / 3.0):
        raise AssertionError(
            "False-positive rate did not include the relabeled background point"
        )


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
    num_points: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    data_dir = work_dir / "data"
    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for index in range(num_train):
        path = data_dir / "train" / f"dummy_train_{index:03d}_pointcloud_annotated_grid.h5"
        make_dummy_pointcloud_h5(path, num_points=num_points, seed=seed + index)
        train_paths.append(path)

    for index in range(num_val):
        path = data_dir / "val" / f"dummy_val_{index:03d}_pointcloud_annotated_grid.h5"
        make_dummy_pointcloud_h5(path, num_points=num_points, seed=seed + 1000 + index)
        val_paths.append(path)

    train_list = work_dir / "train_h5_list.txt"
    val_list = work_dir / "val_h5_list.txt"
    write_list_file(train_list, train_paths)
    write_list_file(val_list, val_paths)
    return train_list, val_list, val_paths[0]


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
        "mlp_baseline",
        "--features",
        "intensity,confidence",
        "--num_points",
        str(args.sample_points),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        "0.0",
        "--width",
        str(args.width),
        "--depth",
        str(args.depth),
        "--expansion",
        "2",
        "--dropout",
        "0.0",
        "--num_workers",
        "0",
        "--device",
        "cpu",
        "--seed",
        str(args.seed),
        "--window_mode",
        args.window_mode,
        "--window_size_frames",
        str(args.window_size_frames),
        "--window_stride_frames",
        str(args.window_stride_frames),
    ]
    if not args.include_tail_window:
        cmd.append("--no_include_tail_window")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_training_outputs(
    output_dir: Path,
    *,
    expected_epochs: int,
    expected_window_mode: str,
) -> None:
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.jsonl"
    history_path = output_dir / "history.json"
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"

    for path in [config_path, metrics_path, history_path, last_path, best_path]:
        if not path.exists():
            raise AssertionError(f"Expected training artifact not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model") != "mlp_baseline":
        raise AssertionError(f"Unexpected model in config.json: {config.get('model')}")
    if config.get("feature_dim") != 2:
        raise AssertionError(f"Unexpected feature_dim in config.json: {config.get('feature_dim')}")
    if config.get("window_mode") != expected_window_mode:
        raise AssertionError(f"Unexpected window_mode in config.json: {config.get('window_mode')}")
    if expected_window_mode == "overlap":
        for name in ["train_window_summary.csv", "val_window_summary.csv"]:
            path = output_dir / name
            if not path.exists():
                raise AssertionError(f"Expected window summary not found: {path}")
        if int(config.get("num_train_samples", 0)) <= int(config.get("num_train_files", 0)):
            raise AssertionError(
                "Expected overlap window training to produce more train samples than train files"
            )

    records = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) != expected_epochs:
        raise AssertionError(f"Expected {expected_epochs} metric records, got {len(records)}")

    for record in records:
        train = record.get("train") or {}
        val = record.get("val") or {}
        for name, metrics in [("train", train), ("val", val)]:
            if "loss" not in metrics:
                raise AssertionError(f"{name} metrics do not contain loss")
            if not math.isfinite(float(metrics["loss"])):
                raise AssertionError(f"{name} loss is not finite: {metrics['loss']}")
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "iou_femur",
                "valid_ratio",
                *CONFUSION_METRIC_KEYS,
            ]:
                if key not in metrics:
                    raise AssertionError(f"{name} metrics do not contain {key}")

    checkpoint = torch.load(best_path, map_location="cpu")
    if "model" not in checkpoint or "config" not in checkpoint:
        raise AssertionError("best.pt does not contain model/config")

    model = build_stage5_model(
        "mlp_baseline",
        num_classes=2,
        feature_dim=2,
        width=int(config["width"]),
        depth=int(config["depth"]),
        expansion=int(config["expansion"]),
        dropout=float(config["dropout"]),
        use_global_context=not bool(config.get("no_global_context", False)),
    )
    load_result = model.load_state_dict(checkpoint["model"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError(
            f"Checkpoint reload mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )


def run_inference_with_trained_checkpoint(
    *,
    input_h5: Path,
    checkpoint: Path,
    output_h5: Path,
    output_ply: Path,
    chunk_size: int,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "infer_stage5.py"),
        "--input_h5",
        str(input_h5),
        "--checkpoint",
        str(checkpoint),
        "--output_h5",
        str(output_h5),
        "--output_ply",
        str(output_ply),
        "--chunk_size",
        str(chunk_size),
        "--device",
        "cpu",
        "--strict_checkpoint",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage5 training and inference with dummy data")
    parser.add_argument(
        "--work_dir",
        default=str(REPO_ROOT / "work_dirs" / "_dummy_training_check"),
        help="Directory for generated dummy data, checkpoints, and predictions",
    )
    parser.add_argument("--num_train", type=int, default=4)
    parser.add_argument("--num_val", type=int, default=2)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--sample_points", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--infer_chunk_size", type=int, default=128)
    parser.add_argument("--window_mode", default="overlap", choices=["none", "overlap"])
    parser.add_argument("--window_size_frames", type=int, default=8)
    parser.add_argument("--window_stride_frames", type=int, default=4)
    parser.add_argument("--no_include_tail_window", dest="include_tail_window", action="store_false")
    parser.set_defaults(include_tail_window=True)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_train <= 0:
        raise ValueError("--num_train must be positive")
    if args.num_val <= 0:
        raise ValueError("--num_val must be positive")

    validate_confusion_metric_semantics()

    work_dir = Path(args.work_dir)
    train_list, val_list, inference_input = make_dummy_dataset(
        work_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_points=args.num_points,
        seed=args.seed,
    )

    run_dir = work_dir / "run_mlp_baseline"
    run_training(
        train_list=train_list,
        val_list=val_list,
        output_dir=run_dir,
        args=args,
    )
    validate_training_outputs(
        run_dir,
        expected_epochs=args.epochs,
        expected_window_mode=args.window_mode,
    )

    output_h5 = work_dir / "trained_dummy_prediction.h5"
    output_ply = work_dir / "trained_dummy_prediction.ply"
    run_inference_with_trained_checkpoint(
        input_h5=inference_input,
        checkpoint=run_dir / "best.pt",
        output_h5=output_h5,
        output_ply=output_ply,
        chunk_size=args.infer_chunk_size,
    )
    validate_outputs(
        output_h5=output_h5,
        output_ply=output_ply,
        num_points=args.num_points,
        num_classes=2,
    )

    with h5py.File(output_h5, "r") as f:
        prob = f["prediction/prob_femur"][:]
        print(
            "prob_femur summary:",
            f"min={float(np.min(prob)):.4f}",
            f"mean={float(np.mean(prob)):.4f}",
            f"max={float(np.max(prob)):.4f}",
        )

    print("Dummy training check passed.")
    print(f"work_dir: {work_dir}")
    print("Generated files are kept for inspection. Remove the work_dir manually when no longer needed.")


if __name__ == "__main__":
    main()
