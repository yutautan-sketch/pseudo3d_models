from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model
from stage5.training import SegmentationMetricAccumulator, build_loss


def parse_feature_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_class_weight(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    if is_auto_class_weight(value):
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def is_auto_class_weight(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"auto", "pointnext_auto"}


def compute_class_counts_from_h5(
    h5_paths: Iterable[Path],
    *,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    counts = np.zeros((int(num_classes),), dtype=np.int64)
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            if "annotation" not in f:
                raise KeyError(f"'annotation' group not found in {h5_path}")
            annotation = f["annotation"]
            if "point_label" not in annotation:
                raise KeyError(f"annotation/point_label not found in {h5_path}")
            labels = annotation["point_label"][:].astype(np.int64)
            if "valid_mask" in annotation:
                valid_mask = annotation["valid_mask"][:].astype(bool)
            else:
                valid_mask = np.ones_like(labels, dtype=bool)

        if labels.shape != valid_mask.shape:
            raise ValueError(
                f"annotation length mismatch in {h5_path}: "
                f"point_label={labels.shape}, valid_mask={valid_mask.shape}"
            )

        mask = valid_mask & (labels != int(ignore_index))
        selected = labels[mask]
        if selected.size == 0:
            continue
        invalid = selected[(selected < 0) | (selected >= int(num_classes))]
        if invalid.size > 0:
            unique_invalid = sorted(set(int(item) for item in invalid.tolist()))
            raise ValueError(
                f"Found label(s) outside [0, {num_classes - 1}] in {h5_path}: {unique_invalid}"
            )
        counts += np.bincount(selected, minlength=int(num_classes))[: int(num_classes)]
    return counts


def pointnext_class_weights_from_counts(
    counts: np.ndarray,
    *,
    epsilon: float = 0.02,
    normalize: bool = True,
) -> list[float]:
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute class weights because no valid labeled points were found")
    frequency = counts.astype(np.float64) / float(total)
    weights = 1.0 / (frequency + float(epsilon))
    if normalize:
        weights = weights * len(weights) / weights.sum()
    return weights.astype(np.float32).tolist()


def resolve_class_weight(
    args: argparse.Namespace,
    train_paths: Iterable[Path],
) -> tuple[list[float] | None, dict[str, Any]]:
    requested = args.class_weight
    if not is_auto_class_weight(requested):
        manual_weight = parse_class_weight(requested)
        return manual_weight, {
            "mode": "manual" if manual_weight is not None else "none",
            "requested": requested,
            "counts": None,
            "epsilon": None,
            "normalize": None,
            "resolved": manual_weight,
        }

    counts = compute_class_counts_from_h5(
        train_paths,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )
    resolved = pointnext_class_weights_from_counts(
        counts,
        epsilon=args.auto_class_weight_epsilon,
        normalize=args.normalize_auto_class_weight,
    )
    return resolved, {
        "mode": "auto",
        "requested": requested,
        "counts": [int(item) for item in counts.tolist()],
        "epsilon": float(args.auto_class_weight_epsilon),
        "normalize": bool(args.normalize_auto_class_weight),
        "resolved": resolved,
    }


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
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any] | None,
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
    num_points = None if args.window_mode == "overlap" else args.num_points
    return Pseudo3DPointCloudDataset(
        h5_paths,
        num_points=num_points,
        features=parse_feature_list(args.features),
        normalize_points=not args.no_normalize_points,
        sampling_mode=args.sampling_mode,
        exclude_ignore_in_sampling=args.exclude_ignore_in_sampling,
        positive_oversample_ratio=args.positive_oversample_ratio,
        frame_window_size=args.frame_window_size,
        window_mode=args.window_mode,
        window_size_frames=args.window_size_frames,
        window_stride_frames=args.window_stride_frames,
        include_tail_window=args.include_tail_window,
        seed=seed,
        cache_data=cache_data,
    )


def make_loader(
    dataset: Pseudo3DPointCloudDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    window_mode: str,
) -> DataLoader:
    collate_fn = pad_point_window_collate if window_mode == "overlap" else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
    )


def write_window_summary(
    path: Path,
    *,
    split: str,
    dataset: Pseudo3DPointCloudDataset,
) -> None:
    rows = dataset.get_window_summaries()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "h5_index",
        "h5_path",
        "window_id",
        "start_frame",
        "end_frame",
        "num_points",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": split,
                    "h5_index": row["h5_index"],
                    "h5_path": row["h5_path"],
                    "window_id": row["window_id"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "num_points": row["num_points"],
                }
            )


class TrainingDebugAccumulator:
    """Collect epoch-level diagnostics without changing the training objective."""

    def __init__(self, *, ignore_index: int = -1, positive_label: int = 1) -> None:
        self.ignore_index = int(ignore_index)
        self.positive_label = int(positive_label)
        self.total_points = 0
        self.valid_points = 0
        self.ignore_points = 0
        self.unpadded_points = 0
        self.padded_points = 0
        self.num_samples = 0
        self.empty_valid_samples = 0
        self.label_background = 0
        self.label_positive = 0
        self.pred_background = 0
        self.pred_positive = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.valid_points_per_sample: list[int] = []
        self.num_points_per_sample: list[int] = []
        self.logit_background_sum = 0.0
        self.logit_positive_sum = 0.0
        self.prob_positive_sum = 0.0
        self.prob_positive_on_gt_background_sum = 0.0
        self.prob_positive_on_gt_positive_sum = 0.0
        self.logit_valid_count = 0
        self.gt_background_count_for_prob = 0
        self.gt_positive_count_for_prob = 0

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    @torch.no_grad()
    def update(self, output: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        logits = output["logits"].detach()
        labels = batch["labels"].detach()
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            valid_mask = valid_mask.detach().bool()

        metric_mask = valid_mask & (labels != self.ignore_index)
        predictions = torch.argmax(logits, dim=-1)

        total = int(labels.numel())
        valid = int(metric_mask.sum().item())
        self.total_points += total
        self.valid_points += valid
        self.ignore_points += total - valid

        if "num_points" in batch:
            num_points = batch["num_points"].detach().long()
        else:
            num_points = torch.full(
                (labels.shape[0],),
                labels.shape[1],
                dtype=torch.long,
                device=labels.device,
            )
        unpadded = int(num_points.sum().item())
        self.unpadded_points += unpadded
        self.padded_points += max(0, total - unpadded)
        self.num_samples += int(labels.shape[0])
        self.num_points_per_sample.extend(int(item) for item in num_points.detach().cpu().tolist())

        per_sample_valid = metric_mask.sum(dim=1).detach().cpu().tolist()
        self.valid_points_per_sample.extend(int(item) for item in per_sample_valid)
        self.empty_valid_samples += sum(1 for item in per_sample_valid if int(item) == 0)

        if valid == 0:
            return

        valid_labels = labels[metric_mask]
        valid_predictions = predictions[metric_mask]
        positive_label = int(self.positive_label)
        background_label = 0

        label_positive_mask = valid_labels == positive_label
        label_background_mask = valid_labels == background_label
        pred_positive_mask = valid_predictions == positive_label
        pred_background_mask = valid_predictions == background_label

        self.label_positive += int(label_positive_mask.sum().item())
        self.label_background += int(label_background_mask.sum().item())
        self.pred_positive += int(pred_positive_mask.sum().item())
        self.pred_background += int(pred_background_mask.sum().item())
        self.tp += int((pred_positive_mask & label_positive_mask).sum().item())
        self.fp += int((pred_positive_mask & label_background_mask).sum().item())
        self.tn += int((pred_background_mask & label_background_mask).sum().item())
        self.fn += int((pred_background_mask & label_positive_mask).sum().item())

        valid_logits = logits[metric_mask]
        if valid_logits.shape[-1] >= 2:
            probabilities = torch.softmax(valid_logits.float(), dim=-1)
            prob_positive = probabilities[:, positive_label]
            self.logit_background_sum += float(valid_logits[:, background_label].float().sum().item())
            self.logit_positive_sum += float(valid_logits[:, positive_label].float().sum().item())
            self.prob_positive_sum += float(prob_positive.sum().item())
            self.logit_valid_count += int(valid_logits.shape[0])

            if label_background_mask.any():
                self.prob_positive_on_gt_background_sum += float(
                    prob_positive[label_background_mask].sum().item()
                )
                self.gt_background_count_for_prob += int(label_background_mask.sum().item())
            if label_positive_mask.any():
                self.prob_positive_on_gt_positive_sum += float(
                    prob_positive[label_positive_mask].sum().item()
                )
                self.gt_positive_count_for_prob += int(label_positive_mask.sum().item())

    def compute(self) -> dict[str, float]:
        valid_array = np.asarray(self.valid_points_per_sample, dtype=np.float64)
        points_array = np.asarray(self.num_points_per_sample, dtype=np.float64)
        if valid_array.size == 0:
            valid_array = np.asarray([0.0], dtype=np.float64)
        if points_array.size == 0:
            points_array = np.asarray([0.0], dtype=np.float64)

        return {
            "debug_total_point_count": float(self.total_points),
            "debug_valid_point_count": float(self.valid_points),
            "debug_ignore_point_count": float(self.ignore_points),
            "debug_ignore_ratio": self._ratio(self.ignore_points, self.total_points),
            "debug_unpadded_point_count": float(self.unpadded_points),
            "debug_padded_point_count": float(self.padded_points),
            "debug_padding_ratio": self._ratio(self.padded_points, self.total_points),
            "debug_num_samples": float(self.num_samples),
            "debug_empty_valid_sample_count": float(self.empty_valid_samples),
            "debug_empty_valid_sample_ratio": self._ratio(self.empty_valid_samples, self.num_samples),
            "debug_min_valid_points_per_sample": float(valid_array.min()),
            "debug_mean_valid_points_per_sample": float(valid_array.mean()),
            "debug_max_valid_points_per_sample": float(valid_array.max()),
            "debug_min_num_points_per_sample": float(points_array.min()),
            "debug_mean_num_points_per_sample": float(points_array.mean()),
            "debug_max_num_points_per_sample": float(points_array.max()),
            "debug_label_background_count": float(self.label_background),
            "debug_label_positive_count": float(self.label_positive),
            "debug_label_positive_ratio": self._ratio(self.label_positive, self.valid_points),
            "debug_pred_background_count": float(self.pred_background),
            "debug_pred_positive_count": float(self.pred_positive),
            "debug_pred_positive_ratio": self._ratio(self.pred_positive, self.valid_points),
            "debug_tp": float(self.tp),
            "debug_fp": float(self.fp),
            "debug_tn": float(self.tn),
            "debug_fn": float(self.fn),
            "debug_mean_logit_background": self._ratio(self.logit_background_sum, self.logit_valid_count),
            "debug_mean_logit_positive": self._ratio(self.logit_positive_sum, self.logit_valid_count),
            "debug_mean_logit_margin_positive_minus_background": self._ratio(
                self.logit_positive_sum - self.logit_background_sum,
                self.logit_valid_count,
            ),
            "debug_mean_prob_positive": self._ratio(self.prob_positive_sum, self.logit_valid_count),
            "debug_mean_prob_positive_on_gt_background": self._ratio(
                self.prob_positive_on_gt_background_sum,
                self.gt_background_count_for_prob,
            ),
            "debug_mean_prob_positive_on_gt_positive": self._ratio(
                self.prob_positive_on_gt_positive_sum,
                self.gt_positive_count_for_prob,
            ),
        }


def add_class_weight_debug(
    metrics: dict[str, Any],
    *,
    class_weight_info: dict[str, Any] | None,
    label_smoothing: float,
) -> None:
    if class_weight_info is None:
        return
    resolved = class_weight_info.get("resolved")
    counts = class_weight_info.get("counts")
    metrics["debug_class_weight_mode"] = class_weight_info.get("mode")
    metrics["debug_class_weight"] = resolved
    metrics["debug_class_counts"] = counts
    metrics["debug_auto_class_weight_epsilon"] = class_weight_info.get("epsilon")
    metrics["debug_auto_class_weight_normalize"] = class_weight_info.get("normalize")
    metrics["debug_label_smoothing"] = float(label_smoothing)
    if isinstance(resolved, list):
        for index, value in enumerate(resolved):
            metrics[f"debug_class_weight_{index}"] = float(value)
    if isinstance(counts, list):
        for index, value in enumerate(counts):
            metrics[f"debug_class_count_{index}"] = float(value)


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
    ignore_index: int = -1,
    class_weight_info: dict[str, Any] | None = None,
    label_smoothing: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    is_train = optimizer is not None
    model.train(is_train)
    metric_acc = SegmentationMetricAccumulator(device=device, ignore_index=ignore_index)
    debug_acc = TrainingDebugAccumulator(ignore_index=ignore_index)
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
        debug_acc.update(output, batch)
        total_loss += float(loss.detach().item())
        total_batches += 1
        iterator.set_postfix(loss=total_loss / max(total_batches, 1))

    metrics = metric_acc.compute()
    metrics.update(debug_acc.compute())
    add_class_weight_debug(
        metrics,
        class_weight_info=class_weight_info,
        label_smoothing=label_smoothing,
    )
    avg_loss = total_loss / max(total_batches, 1)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def build_config(
    args: argparse.Namespace,
    *,
    feature_dim: int,
    class_weight_info: dict[str, Any],
) -> dict[str, Any]:
    config = vars(args).copy()
    config["feature_names"] = parse_feature_list(args.features)
    config["feature_dim"] = int(feature_dim)
    config["class_weight"] = class_weight_info["resolved"]
    config["class_weight_info"] = class_weight_info
    return make_jsonable(config)


def build_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "width": args.width,
        "depth": args.depth,
        "expansion": args.expansion,
        "dropout": args.dropout,
        "use_global_context": not args.no_global_context,
    }
    if args.model.lower() == "pointnext_s":
        kwargs.update(
            {
                "pointnext_radius": args.pointnext_radius,
                "pointnext_nsample": args.pointnext_nsample,
                "pointnext_sa_layers": args.pointnext_sa_layers,
                "pointnext_sa_use_res": args.pointnext_sa_use_res,
            }
        )
    return kwargs


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
    if args.window_mode not in {"none", "overlap"}:
        raise ValueError("--window_mode must be one of: none, overlap")
    if args.window_size_frames <= 0:
        raise ValueError("--window_size_frames must be positive")
    if args.window_stride_frames <= 0:
        raise ValueError("--window_stride_frames must be positive")
    if args.pointnext_radius <= 0:
        raise ValueError("--pointnext_radius must be positive")
    if args.pointnext_nsample <= 0:
        raise ValueError("--pointnext_nsample must be positive")
    if args.pointnext_sa_layers <= 0:
        raise ValueError("--pointnext_sa_layers must be positive")
    if args.save_every < 0:
        raise ValueError("--save_every must be non-negative")
    if (
        args.window_mode == "none"
        and args.num_points is not None
        and args.num_points <= 0
        and args.batch_size > 1
    ):
        raise ValueError("--num_points <= 0 returns variable-size samples; use --batch_size 1")

    if is_auto_class_weight(args.class_weight):
        if args.auto_class_weight_epsilon <= 0:
            raise ValueError("--auto_class_weight_epsilon must be positive")
        return

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
        window_mode=args.window_mode,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = make_loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            window_mode=args.window_mode,
        )

    class_weight, class_weight_info = resolve_class_weight(args, train_paths)

    feature_dim = train_dataset.num_feature_channels
    config = build_config(args, feature_dim=feature_dim, class_weight_info=class_weight_info)
    config["num_train_files"] = len(train_paths)
    config["num_val_files"] = len(val_paths)
    config["num_train_samples"] = len(train_dataset)
    config["num_val_samples"] = len(val_dataset) if val_dataset is not None else 0
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if args.window_mode == "overlap":
        write_window_summary(output_dir / "train_window_summary.csv", split="train", dataset=train_dataset)
        if val_dataset is not None:
            write_window_summary(output_dir / "val_window_summary.csv", split="val", dataset=val_dataset)

    model = build_stage5_model(
        args.model,
        num_classes=args.num_classes,
        feature_dim=feature_dim,
        checkpoint=args.checkpoint,
        strict_checkpoint=args.strict_checkpoint,
        **build_model_kwargs(args),
    ).to(device)

    loss_fn = build_loss(
        args.loss,
        ignore_index=args.ignore_index,
        class_weight=class_weight,
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
    print(f"train files: {len(train_paths)}")
    print(f"train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"val files: {len(val_paths)}")
        print(f"val samples: {len(val_dataset)}")
    print(f"feature_dim: {feature_dim}")
    if class_weight is not None:
        print(f"class weight: {class_weight}")
        if class_weight_info["mode"] == "auto":
            print(f"class counts: {class_weight_info['counts']}")
    if args.window_mode == "overlap":
        print(
            "frame windows: "
            f"size={args.window_size_frames} stride={args.window_stride_frames} "
            f"include_tail={args.include_tail_window}"
        )

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
            ignore_index=args.ignore_index,
            class_weight_info=class_weight_info,
            label_smoothing=args.label_smoothing,
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
                    ignore_index=args.ignore_index,
                    class_weight_info=class_weight_info,
                    label_smoothing=args.label_smoothing,
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
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
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
    parser.add_argument("--pointnext_radius", type=float, default=0.1)
    parser.add_argument("--pointnext_nsample", type=int, default=16)
    parser.add_argument("--pointnext_sa_layers", type=int, default=2)
    parser.add_argument(
        "--no_pointnext_sa_use_res",
        dest="pointnext_sa_use_res",
        action="store_false",
        help="Disable residual set-abstraction blocks in the PointNeXt-S wrapper",
    )
    parser.set_defaults(pointnext_sa_use_res=True)

    parser.add_argument("--features", default="intensity,confidence")
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--sampling_mode", default="video", choices=["video", "frame_window"])
    parser.add_argument("--frame_window_size", type=int, default=None)
    parser.add_argument("--exclude_ignore_in_sampling", action="store_true")
    parser.add_argument("--positive_oversample_ratio", type=float, default=0.0)
    parser.add_argument("--window_mode", default="none", choices=["none", "overlap"])
    parser.add_argument("--window_size_frames", type=int, default=12)
    parser.add_argument("--window_stride_frames", type=int, default=6)
    parser.add_argument(
        "--no_include_tail_window",
        dest="include_tail_window",
        action="store_false",
        help="Disable the final shorter frame-order window in --window_mode overlap",
    )
    parser.set_defaults(include_tail_window=True)
    parser.add_argument("--no_normalize_points", action="store_true")
    parser.add_argument("--cache_data", action="store_true")

    parser.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "ce"])
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument(
        "--class_weight",
        default=None,
        help="Comma-separated class weights, e.g. '1.0,4.0', or 'auto' for PointNeXt-style frequency weights",
    )
    parser.add_argument("--auto_class_weight_epsilon", type=float, default=0.02)
    parser.add_argument(
        "--no_normalize_auto_class_weight",
        dest="normalize_auto_class_weight",
        action="store_false",
        help="Disable PointNeXt-style normalization of auto class weights to mean 1",
    )
    parser.set_defaults(normalize_auto_class_weight=True)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--best_metric", default="iou_femur")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint_epoch_XXXX.pt every N epochs; 0 disables periodic checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
