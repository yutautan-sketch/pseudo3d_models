from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage5.datasets import Pseudo3DPointCloudDataset
from stage5.models import build_stage5_model
from stage5.training import SegmentationMetricAccumulator, build_loss


def parse_feature_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_class_weight(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def collect_h5_paths_from_dir(
    directory: str | Path,
    *,
    pattern: str,
    max_files: int = 0,
) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"H5 directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"H5 path is not a directory: {directory}")

    paths = sorted(path for path in directory.glob(pattern) if path.is_file())
    if max_files > 0:
        paths = paths[: int(max_files)]
    if not paths:
        raise FileNotFoundError(f"No H5 files matched {pattern!r} in {directory}")
    return paths


def read_list_paths(path: str | Path) -> list[Path]:
    list_path = Path(path)
    base_dir = list_path.parent
    paths: list[Path] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            h5_path = Path(token)
            if not h5_path.is_absolute():
                h5_path = base_dir / h5_path
            paths.append(h5_path)
    if not paths:
        raise ValueError(f"H5 list is empty: {list_path}")
    return paths


def split_train_val_paths(
    paths: list[Path],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    val_fraction = float(val_fraction)
    if val_fraction <= 0:
        return paths, []
    if val_fraction >= 1:
        raise ValueError("--val_fraction must be less than 1.0")
    if len(paths) < 2:
        raise ValueError("--val_fraction requires at least 2 H5 files")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    num_val = max(1, int(round(len(paths) * val_fraction)))
    num_val = min(num_val, len(paths) - 1)
    val_indices = set(indices[:num_val].tolist())
    train_paths = [path for i, path in enumerate(paths) if i not in val_indices]
    val_paths = [path for i, path in enumerate(paths) if i in val_indices]
    return train_paths, val_paths


def write_resolved_path_list(path: Path, h5_paths: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h5_path in h5_paths:
            f.write(f"{h5_path}\n")


def resolve_train_val_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    if args.train_list is not None and args.train_dir is not None:
        raise ValueError("Specify only one of --train_list or --train_dir")
    if args.train_list is None and args.train_dir is None:
        raise ValueError("Specify either --train_list or --train_dir")

    if args.train_list is not None:
        train_paths = read_list_paths(args.train_list)
    else:
        train_paths = collect_h5_paths_from_dir(
            args.train_dir,
            pattern=args.h5_pattern,
            max_files=args.max_train_files,
        )

    if args.val_list is not None and args.val_dir is not None:
        raise ValueError("Specify only one of --val_list or --val_dir")
    if args.val_list is not None:
        val_paths = read_list_paths(args.val_list)
    elif args.val_dir is not None:
        val_paths = collect_h5_paths_from_dir(
            args.val_dir,
            pattern=args.h5_pattern,
            max_files=args.max_val_files,
        )
    else:
        train_paths, val_paths = split_train_val_paths(
            train_paths,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

    if args.max_train_files > 0 and args.train_list is not None:
        train_paths = train_paths[: args.max_train_files]
    if args.max_val_files > 0 and args.val_list is not None:
        val_paths = val_paths[: args.max_val_files]

    return train_paths, val_paths


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    return value


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    best_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_score": float(best_score),
        },
        path,
    )


def make_dataset(
    h5_paths: str | Path | Iterable[str | Path],
    *,
    args: argparse.Namespace,
    seed: int,
    cache_data: bool,
) -> Pseudo3DPointCloudDataset:
    return Pseudo3DPointCloudDataset(
        h5_paths,
        num_points=args.num_points,
        features=parse_feature_list(args.features),
        normalize_points=not args.no_normalize_points,
        sampling_mode=args.sampling_mode,
        exclude_ignore_in_sampling=args.exclude_ignore_in_sampling,
        positive_oversample_ratio=args.positive_oversample_ratio,
        frame_window_size=args.frame_window_size,
        seed=seed,
        cache_data=cache_data,
    )


def make_loader(
    dataset: Pseudo3DPointCloudDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    desc: str,
    grad_clip_norm: float | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    metric_acc = SegmentationMetricAccumulator(device=device)
    total_loss = 0.0
    total_batches = 0

    iterator = tqdm(loader, desc=f"{desc} epoch {epoch}", leave=False)
    for batch in iterator:
        batch = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            output = model(batch)
            loss_dict = loss_fn(output, batch)
            loss = loss_dict["loss"]
            if is_train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        metric_acc.update(output, batch)
        total_loss += float(loss.detach().item())
        total_batches += 1
        iterator.set_postfix(loss=total_loss / max(total_batches, 1))

    metrics = metric_acc.compute()
    avg_loss = total_loss / max(total_batches, 1)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def build_config(args: argparse.Namespace, *, feature_dim: int) -> dict[str, Any]:
    config = vars(args).copy()
    config["feature_names"] = parse_feature_list(args.features)
    config["feature_dim"] = int(feature_dim)
    config["class_weight"] = parse_class_weight(args.class_weight)
    return make_jsonable(config)


def validate_args(args: argparse.Namespace) -> None:
    if args.val_every <= 0:
        raise ValueError("--val_every must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.max_train_files < 0:
        raise ValueError("--max_train_files must be non-negative")
    if args.max_val_files < 0:
        raise ValueError("--max_val_files must be non-negative")
    if args.sampling_mode == "frame_window" and args.frame_window_size is None:
        raise ValueError("--frame_window_size is required when --sampling_mode frame_window")
    if args.num_points is not None and args.num_points <= 0 and args.batch_size > 1:
        raise ValueError("--num_points <= 0 returns variable-size samples; use --batch_size 1")

    class_weight = parse_class_weight(args.class_weight)
    if class_weight is not None and len(class_weight) != args.num_classes:
        raise ValueError(
            f"--class_weight length must match --num_classes: {len(class_weight)} vs {args.num_classes}"
        )


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl_path = output_dir / "metrics.jsonl"
    metrics_jsonl_path.write_text("", encoding="utf-8")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    train_paths, val_paths = resolve_train_val_paths(args)
    write_resolved_path_list(output_dir / "train_files.txt", train_paths)
    if val_paths:
        write_resolved_path_list(output_dir / "val_files.txt", val_paths)

    train_dataset = make_dataset(
        train_paths,
        args=args,
        seed=args.seed,
        cache_data=args.cache_data,
    )
    val_dataset = None
    if val_paths:
        val_dataset = make_dataset(
            val_paths,
            args=args,
            seed=args.seed + 100000,
            cache_data=args.cache_data,
        )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = make_loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    feature_dim = train_dataset.num_feature_channels
    config = build_config(args, feature_dim=feature_dim)
    config["num_train_files"] = len(train_paths)
    config["num_val_files"] = len(val_paths)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    model = build_stage5_model(
        args.model,
        num_classes=args.num_classes,
        feature_dim=feature_dim,
        checkpoint=args.checkpoint,
        strict_checkpoint=args.strict_checkpoint,
        width=args.width,
        depth=args.depth,
        expansion=args.expansion,
        dropout=args.dropout,
        use_global_context=not args.no_global_context,
    ).to(device)

    loss_fn = build_loss(
        args.loss,
        ignore_index=args.ignore_index,
        class_weight=parse_class_weight(args.class_weight),
        label_smoothing=args.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = float("-inf")
    history: list[dict[str, Any]] = []

    print(f"Training Stage5 model on {device}")
    print(f"train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"val samples: {len(val_dataset)}")
    print(f"feature_dim: {feature_dim}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            desc="train",
            grad_clip_norm=args.grad_clip_norm,
        )

        val_metrics = None
        score = train_metrics.get(args.best_metric, -train_loss)
        if val_loader is not None and epoch % args.val_every == 0:
            with torch.no_grad():
                _, val_metrics = run_one_epoch(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    optimizer=None,
                    device=device,
                    epoch=epoch,
                    desc="val",
                )
            score = val_metrics.get(args.best_metric, -val_metrics["loss"])

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        with metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_score=max(best_score, score),
        )
        if score >= best_score:
            best_score = score
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_score=best_score,
            )

        train_summary = f"loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} iou={train_metrics['iou_femur']:.4f}"
        if val_metrics is None:
            print(f"epoch {epoch:03d}: train {train_summary}")
        else:
            val_summary = f"loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} iou={val_metrics['iou_femur']:.4f}"
            print(f"epoch {epoch:03d}: train {train_summary} | val {val_summary}")

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 5 pseudo-3D femur point-cloud segmentor")

    parser.add_argument("--train_list", default=None, help="Text file containing annotated H5 paths")
    parser.add_argument("--train_dir", default=None, help="Directory containing annotated H5 files")
    parser.add_argument("--val_list", default=None, help="Optional text file containing validation H5 paths")
    parser.add_argument("--val_dir", default=None, help="Optional directory containing validation H5 files")
    parser.add_argument("--h5_pattern", default="*_annotated_grid.h5", help="Pattern used with --train_dir/--val_dir")
    parser.add_argument("--val_fraction", type=float, default=0.0, help="Validation split fraction when no val source is provided")
    parser.add_argument("--max_train_files", type=int, default=0, help="Limit train files after sorting/list loading; 0 means all")
    parser.add_argument("--max_val_files", type=int, default=0, help="Limit val files after sorting/list loading; 0 means all")
    parser.add_argument("--output_dir", required=True, help="Directory for config, logs, and checkpoints")

    parser.add_argument("--model", default="mlp_baseline")
    parser.add_argument("--checkpoint", default=None, help="Optional Stage5 checkpoint to initialize from")
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_global_context", action="store_true")

    parser.add_argument("--features", default="intensity,confidence")
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--sampling_mode", default="video", choices=["video", "frame_window"])
    parser.add_argument("--frame_window_size", type=int, default=None)
    parser.add_argument("--exclude_ignore_in_sampling", action="store_true")
    parser.add_argument("--positive_oversample_ratio", type=float, default=0.0)
    parser.add_argument("--no_normalize_points", action="store_true")
    parser.add_argument("--cache_data", action="store_true")

    parser.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "ce"])
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--class_weight", default=None, help="Comma-separated class weights, e.g. '1.0,4.0'")
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--best_metric", default="iou_femur")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
