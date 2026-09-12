from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch import nn

from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model
from stage5.training import build_loss
from stage5.utils.h5_io import read_path_list


@dataclass(frozen=True)
class SampleInfo:
    split: str
    dataset_index: int
    h5_index: int
    h5_alias: str
    window_id: int
    window_start: int
    window_end: int
    num_points: int
    num_valid: int
    num_positive: int

    @property
    def key(self) -> tuple[int, int]:
        return self.h5_index, self.window_id


@dataclass
class ForwardResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    loss: float | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_tokens(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(token.strip() for token in value.split(",") if token.strip())
    return tuple(str(token).strip() for token in value if str(token).strip())


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def build_model_kwargs(config: dict[str, Any], feature_dim: int) -> dict[str, Any]:
    model_name = str(config.get("model", "pointnext_s"))
    require(model_name.lower() == "pointnext_s", f"Padding check requires pointnext_s, got {model_name}")
    return {
        "name": model_name,
        "num_classes": int(config.get("num_classes", 2)),
        "feature_dim": int(feature_dim),
        "width": int(config.get("width", 32)),
        "depth": int(config.get("depth", 6)),
        "expansion": int(config.get("expansion", 4)),
        "dropout": float(config.get("dropout", 0.0)),
        "use_global_context": not bool(config.get("no_global_context", False)),
        "pointnext_radius": float(config.get("pointnext_radius", config.get("radius", 0.1))),
        "pointnext_nsample": int(config.get("pointnext_nsample", config.get("nsample", 16))),
        "pointnext_sa_layers": int(config.get("pointnext_sa_layers", config.get("sa_layers", 2))),
        "pointnext_sa_use_res": bool(
            config.get("pointnext_sa_use_res", config.get("sa_use_res", True))
        ),
    }


def build_dataset(paths: list[Path], config: dict[str, Any], features: tuple[str, ...]) -> Pseudo3DPointCloudDataset:
    require(config.get("window_mode", "overlap") == "overlap", "Run must use overlap windows")
    return Pseudo3DPointCloudDataset(
        paths,
        num_points=None,
        features=features,
        normalize_points=not bool(config.get("no_normalize_points", False)),
        window_mode="overlap",
        window_size_frames=int(config.get("window_size_frames", 12)),
        window_stride_frames=int(config.get("window_stride_frames", 6)),
        include_tail_window=bool(config.get("include_tail_window", True)),
        cache_data=False,
    )


def collect_sample_info(
    dataset: Pseudo3DPointCloudDataset,
    *,
    split: str,
) -> list[SampleInfo]:
    samples_by_h5: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for dataset_index, sample in enumerate(dataset.samples):
        samples_by_h5[int(sample["h5_index"])].append((dataset_index, sample["window"]))

    rows: list[SampleInfo] = []
    for h5_index, indexed_windows in sorted(samples_by_h5.items()):
        path = dataset.h5_paths[h5_index]
        with h5py.File(path, "r") as handle:
            frame_order = handle["point_cloud/frame_order"][:]
            labels = handle["annotation/point_label"][:].astype(np.int64)
            valid_mask = handle["annotation/valid_mask"][:].astype(bool)
        require(frame_order.shape == labels.shape == valid_mask.shape, f"Point arrays differ in {path}")

        for dataset_index, window in indexed_windows:
            require(window is not None, f"Dataset sample {dataset_index} has no window")
            selected = (frame_order >= window.start_frame) & (frame_order <= window.end_frame)
            selected_valid = valid_mask[selected]
            selected_labels = labels[selected]
            rows.append(
                SampleInfo(
                    split=split,
                    dataset_index=int(dataset_index),
                    h5_index=h5_index,
                    h5_alias=f"{split}_{h5_index:03d}",
                    window_id=int(window.window_id),
                    window_start=int(window.start_frame),
                    window_end=int(window.end_frame),
                    num_points=int(selected.sum()),
                    num_valid=int(selected_valid.sum()),
                    num_positive=int(np.sum(selected_labels[selected_valid] == 1)),
                )
            )
    rows.sort(key=lambda row: row.dataset_index)
    require(len(rows) == len(dataset), f"{split}: sample metadata count differs")
    return rows


def select_targets(rows: list[SampleInfo], modes: tuple[str, ...]) -> list[tuple[str, SampleInfo]]:
    require(bool(rows), "Cannot select targets from an empty split")
    by_points = sorted(rows, key=lambda row: (row.num_points, row.dataset_index))
    positive = [row for row in by_points if row.num_positive > 0]
    negative = [row for row in by_points if row.num_positive == 0]
    candidates: dict[str, SampleInfo] = {
        "min": by_points[0],
        "median": by_points[len(by_points) // 2],
        "max": by_points[-1],
    }
    if positive:
        candidates["positive_short"] = positive[0]
    if negative:
        candidates["negative_only"] = negative[0]

    selected: list[tuple[str, SampleInfo]] = []
    seen: set[int] = set()
    for mode in modes:
        require(mode in candidates, f"Target mode {mode!r} is unavailable")
        row = candidates[mode]
        if row.dataset_index in seen:
            continue
        seen.add(row.dataset_index)
        selected.append((mode, row))
    require(bool(selected), "No distinct parity targets were selected")
    return selected


def select_peers(target: SampleInfo, rows: list[SampleInfo]) -> dict[str, SampleInfo]:
    others = [row for row in rows if row.dataset_index != target.dataset_index]
    require(bool(others), "Padding parity requires at least two Dataset samples")
    preferably_other_h5 = [row for row in others if row.h5_index != target.h5_index] or others
    larger = [row for row in preferably_other_h5 if row.num_points >= target.num_points]
    near_pool = larger or preferably_other_h5
    near = min(near_pool, key=lambda row: (abs(row.num_points - target.num_points), row.dataset_index))

    sorted_rows = sorted(preferably_other_h5, key=lambda row: (row.num_points, row.dataset_index))
    median = sorted_rows[len(sorted_rows) // 2]
    maximum = sorted_rows[-1]
    return {
        "peer_near": near,
        "peer_median": median,
        "peer_max": maximum,
    }


def pad_batch_to_points(batch: dict[str, Any], target_points: int) -> dict[str, Any]:
    current_points = int(batch["points"].shape[1])
    require(target_points >= current_points, f"Cannot pad {current_points} points to {target_points}")
    if target_points == current_points:
        return batch

    fill_values: dict[str, float | int | bool] = {
        "points": 0.0,
        "features": 0.0,
        "labels": -1,
        "valid_mask": False,
        "frame_order": -1,
        "point_indices": -1,
    }
    padded = dict(batch)
    for field, fill_value in fill_values.items():
        value = batch[field]
        shape = (value.shape[0], target_points, *value.shape[2:])
        out = torch.full(shape, fill_value, dtype=value.dtype, device=value.device)
        out[:, :current_points] = value
        padded[field] = out
    return padded


def run_eval_forward(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    target_position: int,
    target_num_points: int,
    device: torch.device,
    seed: int,
) -> ForwardResult:
    set_seed(seed)
    model.eval()
    device_batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        output = model(device_batch)
    logits = output["logits"][target_position, :target_num_points].detach().cpu().float()
    require(torch.isfinite(logits).all().item(), "Eval logits contain NaN/Inf")
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return ForwardResult(logits, probabilities, predictions)


def comparison_metrics(
    baseline: ForwardResult,
    candidate: ForwardResult,
    *,
    probability_tolerance: float,
) -> dict[str, Any]:
    require(baseline.logits.shape == candidate.logits.shape, "Compared logits have different shapes")
    logit_delta = (candidate.logits - baseline.logits).abs()
    probability_delta = (candidate.probabilities - baseline.probabilities).abs()
    disagreements = candidate.predictions != baseline.predictions
    max_probability = float(probability_delta.max().item()) if probability_delta.numel() else 0.0
    return {
        "max_abs_logit_diff": float(logit_delta.max().item()) if logit_delta.numel() else 0.0,
        "mean_abs_logit_diff": float(logit_delta.mean().item()) if logit_delta.numel() else 0.0,
        "max_abs_probability_diff": max_probability,
        "mean_abs_probability_diff": float(probability_delta.mean().item())
        if probability_delta.numel()
        else 0.0,
        "prediction_disagreement_count": int(disagreements.sum().item()),
        "prediction_disagreement_rate": float(disagreements.float().mean().item())
        if disagreements.numel()
        else 0.0,
        "base_positive_count": int((baseline.predictions == 1).sum().item()),
        "case_positive_count": int((candidate.predictions == 1).sum().item()),
        "parity_pass": bool(max_probability <= probability_tolerance and not disagreements.any().item()),
    }


def make_eval_row(
    *,
    split: str,
    target_mode: str,
    target: SampleInfo,
    case: str,
    peer: SampleInfo | None,
    batch: dict[str, Any],
    target_position: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    padded_points = int(batch["points"].shape[1])
    padding_points = padded_points - target.num_points
    return {
        "split": split,
        "target_mode": target_mode,
        "target_alias": target.h5_alias,
        "target_window_id": target.window_id,
        "target_num_points": target.num_points,
        "target_num_valid": target.num_valid,
        "target_num_positive": target.num_positive,
        "case": case,
        "peer_alias": peer.h5_alias if peer is not None else "",
        "peer_window_id": peer.window_id if peer is not None else "",
        "peer_num_points": peer.num_points if peer is not None else "",
        "batch_size": int(batch["points"].shape[0]),
        "target_position": target_position,
        "padded_num_points": padded_points,
        "padding_points": padding_points,
        "padding_ratio": float(padding_points / padded_points),
        **metrics,
    }


def run_eval_cases(
    *,
    model: nn.Module,
    dataset: Pseudo3DPointCloudDataset,
    split: str,
    target_mode: str,
    target: SampleInfo,
    peers: dict[str, SampleInfo],
    sample_cache: dict[int, dict[str, Any]],
    device: torch.device,
    seed: int,
    probability_tolerance: float,
) -> list[dict[str, Any]]:
    def sample(info: SampleInfo) -> dict[str, Any]:
        if info.dataset_index not in sample_cache:
            sample_cache[info.dataset_index] = dataset[info.dataset_index]
        return sample_cache[info.dataset_index]

    target_sample = sample(target)
    single_batch = pad_point_window_collate([target_sample])
    baseline = run_eval_forward(
        model,
        single_batch,
        target_position=0,
        target_num_points=target.num_points,
        device=device,
        seed=seed,
    )

    cases: list[tuple[str, dict[str, Any], int, SampleInfo | None]] = []
    cases.append(("repeat_single", single_batch, 0, None))
    cases.append(
        (
            "duplicate_no_padding",
            pad_point_window_collate([target_sample, target_sample]),
            0,
            target,
        )
    )

    added_peer_cases: set[tuple[int, int]] = set()
    for peer_name in ("peer_near", "peer_median", "peer_max"):
        peer = peers[peer_name]
        if peer.key in added_peer_cases:
            continue
        added_peer_cases.add(peer.key)
        cases.append(
            (
                peer_name,
                pad_point_window_collate([target_sample, sample(peer)]),
                0,
                peer,
            )
        )

    max_peer = peers["peer_max"]
    cases.append(
        (
            "peer_max_target_last",
            pad_point_window_collate([sample(max_peer), target_sample]),
            1,
            max_peer,
        )
    )
    manual_target_points = max(target.num_points, max_peer.num_points)
    cases.append(
        (
            "manual_zero_padding",
            pad_batch_to_points(single_batch, manual_target_points),
            0,
            None,
        )
    )

    rows: list[dict[str, Any]] = []
    for case_name, case_batch, target_position, peer in cases:
        candidate = run_eval_forward(
            model,
            case_batch,
            target_position=target_position,
            target_num_points=target.num_points,
            device=device,
            seed=seed,
        )
        metrics = comparison_metrics(
            baseline,
            candidate,
            probability_tolerance=probability_tolerance,
        )
        rows.append(
            make_eval_row(
                split=split,
                target_mode=target_mode,
                target=target,
                case=case_name,
                peer=peer,
                batch=case_batch,
                target_position=target_position,
                metrics=metrics,
            )
        )
        print(
            f"  {split}/{target_mode}/{case_name}: padding={rows[-1]['padding_points']} "
            f"max_prob_diff={metrics['max_abs_probability_diff']:.6g} "
            f"disagree={metrics['prediction_disagreement_count']}"
        )
    return rows


def clone_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def capture_gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def capture_bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    buffers: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.running_mean is not None:
                buffers[f"{name}.running_mean"] = module.running_mean.detach().cpu().float().clone()
            if module.running_var is not None:
                buffers[f"{name}.running_var"] = module.running_var.detach().cpu().float().clone()
    return buffers


def gradient_comparison(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, float]:
    require(set(baseline) == set(candidate), "Compared gradient parameter sets differ")
    base_sq = 0.0
    candidate_sq = 0.0
    delta_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    for name in baseline:
        left = baseline[name].double().reshape(-1)
        right = candidate[name].double().reshape(-1)
        delta = right - left
        base_sq += float(torch.dot(left, left).item())
        candidate_sq += float(torch.dot(right, right).item())
        delta_sq += float(torch.dot(delta, delta).item())
        dot += float(torch.dot(left, right).item())
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max().item()))
    base_norm = math.sqrt(base_sq)
    candidate_norm = math.sqrt(candidate_sq)
    delta_norm = math.sqrt(delta_sq)
    cosine = dot / max(base_norm * candidate_norm, 1e-30)
    return {
        "base_gradient_norm": base_norm,
        "case_gradient_norm": candidate_norm,
        "gradient_delta_norm": delta_norm,
        "gradient_relative_l2_diff": delta_norm / max(base_norm, 1e-30),
        "gradient_cosine_similarity": cosine,
        "gradient_max_abs_diff": max_abs,
    }


def buffer_comparison(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, float]:
    require(set(baseline) == set(candidate), "Compared BatchNorm buffer sets differ")
    delta_sq = 0.0
    base_sq = 0.0
    max_abs = 0.0
    for name in baseline:
        left = baseline[name].double().reshape(-1)
        right = candidate[name].double().reshape(-1)
        delta = right - left
        delta_sq += float(torch.dot(delta, delta).item())
        base_sq += float(torch.dot(left, left).item())
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max().item()))
    delta_norm = math.sqrt(delta_sq)
    return {
        "bn_buffer_delta_norm": delta_norm,
        "bn_buffer_relative_l2_diff": delta_norm / max(math.sqrt(base_sq), 1e-30),
        "bn_buffer_max_abs_diff": max_abs,
    }


def run_train_once(
    model: nn.Module,
    initial_state: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    batch: dict[str, Any],
    *,
    target_num_points: int,
    device: torch.device,
    seed: int,
) -> tuple[ForwardResult, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.load_state_dict(initial_state, strict=True)
    model.train()
    model.zero_grad(set_to_none=True)
    set_seed(seed)
    device_batch = move_batch_to_device(batch, device)
    output = model(device_batch)
    loss_dict = loss_fn(output, device_batch)
    loss = loss_dict["loss"]
    require(torch.isfinite(loss).item(), "Train-mode loss is NaN/Inf")
    loss.backward()
    logits = output["logits"][0, :target_num_points].detach().cpu().float()
    require(torch.isfinite(logits).all().item(), "Train-mode logits contain NaN/Inf")
    result = ForwardResult(
        logits=logits,
        probabilities=torch.softmax(logits, dim=-1),
        predictions=logits.argmax(dim=-1),
        loss=float(loss.detach().cpu().item()),
    )
    return result, capture_gradients(model), capture_bn_buffers(model)


def run_train_padding_case(
    *,
    model: nn.Module,
    initial_state: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    target_mode: str,
    target: SampleInfo,
    target_sample: dict[str, Any],
    pad_to_points: int,
    repeat_count: int,
    device: torch.device,
    seed: int,
    probability_tolerance: float,
    gradient_tolerance: float,
) -> dict[str, Any] | None:
    if pad_to_points <= target.num_points:
        return None
    base_batch = pad_point_window_collate([target_sample for _ in range(repeat_count)])
    padded_batch = pad_batch_to_points(base_batch, pad_to_points)
    baseline, base_gradients, base_buffers = run_train_once(
        model,
        initial_state,
        loss_fn,
        base_batch,
        target_num_points=target.num_points,
        device=device,
        seed=seed,
    )
    candidate, case_gradients, case_buffers = run_train_once(
        model,
        initial_state,
        loss_fn,
        padded_batch,
        target_num_points=target.num_points,
        device=device,
        seed=seed,
    )
    prediction_metrics = comparison_metrics(
        baseline,
        candidate,
        probability_tolerance=probability_tolerance,
    )
    gradient_metrics = gradient_comparison(base_gradients, case_gradients)
    bn_metrics = buffer_comparison(base_buffers, case_buffers)
    train_parity_pass = bool(
        prediction_metrics["parity_pass"]
        and gradient_metrics["gradient_relative_l2_diff"] <= gradient_tolerance
        and bn_metrics["bn_buffer_max_abs_diff"] <= gradient_tolerance
    )
    return {
        "split": target.split,
        "target_mode": target_mode,
        "target_alias": target.h5_alias,
        "target_window_id": target.window_id,
        "target_num_points": target.num_points,
        "repeat_count": repeat_count,
        "padded_num_points": pad_to_points,
        "padding_points_per_sample": pad_to_points - target.num_points,
        "padding_ratio": float((pad_to_points - target.num_points) / pad_to_points),
        "base_loss": baseline.loss,
        "case_loss": candidate.loss,
        "loss_abs_diff": abs(float(candidate.loss) - float(baseline.loss)),
        **prediction_metrics,
        **gradient_metrics,
        **bn_metrics,
        "train_parity_pass": train_parity_pass,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PointNeXt-S output and gradient sensitivity to variable-length zero padding"
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--target_modes", default="min,positive_short")
    parser.add_argument("--train_mode_split", default="train", choices=("train", "val", "none"))
    parser.add_argument("--train_repeat_count", type=int, default=2)
    parser.add_argument("--probability_tolerance", type=float, default=1e-5)
    parser.add_argument("--gradient_tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(run_dir / "config.json")
    require(checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}")
    require(args.train_repeat_count > 0, "train_repeat_count must be positive")
    require(args.probability_tolerance >= 0.0, "probability_tolerance must be non-negative")
    require(args.gradient_tolerance >= 0.0, "gradient_tolerance must be non-negative")

    device = torch.device(args.device)
    require(device.type == "cuda", "PointNeXt-S parity check requires CUDA")
    require(torch.cuda.is_available(), "CUDA is not available")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seed(seed)

    features = parse_tokens(config.get("feature_names", config.get("features", "intensity,confidence")))
    splits = parse_tokens(args.splits)
    target_modes = parse_tokens(args.target_modes)
    require(bool(features), "Feature list is empty")
    require(bool(splits) and set(splits) <= {"train", "val"}, f"Unsupported splits: {splits}")

    datasets: dict[str, Pseudo3DPointCloudDataset] = {}
    sample_infos: dict[str, list[SampleInfo]] = {}
    selected_targets: dict[str, list[tuple[str, SampleInfo]]] = {}
    for split in splits:
        list_path = run_dir / f"{split}_files.txt"
        require(list_path.is_file(), f"Split list not found: {list_path}")
        paths = read_path_list(list_path)
        require(bool(paths), f"{split} split has no H5 files")
        missing = [path for path in paths if not path.is_file()]
        require(not missing, f"{split} split contains missing H5: {missing[0] if missing else ''}")
        dataset = build_dataset(paths, config, features)
        rows = collect_sample_info(dataset, split=split)
        datasets[split] = dataset
        sample_infos[split] = rows
        selected_targets[split] = select_targets(rows, target_modes)

    first_dataset = datasets[splits[0]]
    model_kwargs = build_model_kwargs(config, first_dataset.num_feature_channels)
    model_name = model_kwargs.pop("name")
    model = build_stage5_model(
        model_name,
        checkpoint=checkpoint_path,
        strict_checkpoint=True,
        **model_kwargs,
    ).to(device)
    initial_state = clone_state_to_cpu(model)

    class_weight = config.get("class_weight")
    if not isinstance(class_weight, (list, tuple)):
        class_weight = None
    loss_fn = build_loss(
        str(config.get("loss", "cross_entropy")),
        ignore_index=int(config.get("ignore_index", -1)),
        class_weight=class_weight,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    ).to(device)

    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    target_manifest: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "running",
        "run_name": run_dir.name,
        "checkpoint_name": checkpoint_path.name,
        "settings": {
            "splits": list(splits),
            "target_modes": list(target_modes),
            "train_mode_split": args.train_mode_split,
            "train_repeat_count": args.train_repeat_count,
            "probability_tolerance": args.probability_tolerance,
            "gradient_tolerance": args.gradient_tolerance,
            "seed": seed,
            "device": str(device),
            "features": list(features),
            "normalize_points": not bool(config.get("no_normalize_points", False)),
            "window_size_frames": int(config.get("window_size_frames", 12)),
            "window_stride_frames": int(config.get("window_stride_frames", 6)),
            "model": {"name": model_name, **model_kwargs},
        },
    }
    summary_path = output_dir / "padding_parity_summary.json"
    try:
        for split in splits:
            dataset = datasets[split]
            cache: dict[int, dict[str, Any]] = {}
            for target_offset, (target_mode, target) in enumerate(selected_targets[split]):
                peers = select_peers(target, sample_infos[split])
                target_manifest.append(
                    {
                        "split": split,
                        "target_mode": target_mode,
                        "target_alias": target.h5_alias,
                        "target_window_id": target.window_id,
                        "target_num_points": target.num_points,
                        "target_num_valid": target.num_valid,
                        "target_num_positive": target.num_positive,
                        "near_peer_num_points": peers["peer_near"].num_points,
                        "median_peer_num_points": peers["peer_median"].num_points,
                        "max_peer_num_points": peers["peer_max"].num_points,
                    }
                )
                print(
                    f"Checking {split}/{target_mode}: target={target.num_points}, "
                    f"max_peer={peers['peer_max'].num_points}"
                )
                eval_rows.extend(
                    run_eval_cases(
                        model=model,
                        dataset=dataset,
                        split=split,
                        target_mode=target_mode,
                        target=target,
                        peers=peers,
                        sample_cache=cache,
                        device=device,
                        seed=seed + target_offset,
                        probability_tolerance=args.probability_tolerance,
                    )
                )

                if split == args.train_mode_split:
                    if target.dataset_index not in cache:
                        cache[target.dataset_index] = dataset[target.dataset_index]
                    target_sample = cache[target.dataset_index]
                    train_row = run_train_padding_case(
                        model=model,
                        initial_state=initial_state,
                        loss_fn=loss_fn,
                        target_mode=target_mode,
                        target=target,
                        target_sample=target_sample,
                        pad_to_points=peers["peer_max"].num_points,
                        repeat_count=args.train_repeat_count,
                        device=device,
                        seed=seed + 10000 + target_offset,
                        probability_tolerance=args.probability_tolerance,
                        gradient_tolerance=args.gradient_tolerance,
                    )
                    if train_row is not None:
                        train_rows.append(train_row)
                        print(
                            f"  {split}/{target_mode}/train_manual_padding: "
                            f"max_prob_diff={train_row['max_abs_probability_diff']:.6g} "
                            f"gradient_rel_diff={train_row['gradient_relative_l2_diff']:.6g}"
                        )
                model.load_state_dict(initial_state, strict=True)
                model.eval()
                torch.cuda.empty_cache()

        reproducibility_cases = [row for row in eval_rows if row["case"] == "repeat_single"]
        batch_size_cases = [row for row in eval_rows if row["case"] == "duplicate_no_padding"]
        manual_padding_cases = [row for row in eval_rows if row["case"] == "manual_zero_padding"]
        padded_eval_cases = [
            row
            for row in eval_rows
            if int(row["padding_points"]) > 0 and row["case"] != "manual_zero_padding"
        ]
        reproducibility_controls_pass = bool(reproducibility_cases) and all(
            bool(row["parity_pass"]) for row in reproducibility_cases
        )
        batch_size_sensitivity_detected = any(
            not bool(row["parity_pass"]) for row in batch_size_cases
        )
        manual_padding_pass = (
            all(bool(row["parity_pass"]) for row in manual_padding_cases)
            if manual_padding_cases
            else None
        )
        peer_padding_pass = (
            all(bool(row["parity_pass"]) for row in padded_eval_cases) if padded_eval_cases else None
        )
        train_padding_pass = (
            all(bool(row["train_parity_pass"]) for row in train_rows) if train_rows else None
        )
        if not reproducibility_controls_pass:
            verdict = "inconclusive_no_padding_control_failed"
        elif manual_padding_pass is None:
            verdict = "inconclusive_no_padded_case"
        elif not manual_padding_pass or train_padding_pass is False:
            verdict = "padding_effect_detected"
        else:
            verdict = "no_material_padding_effect_detected"

        summary.update(
            {
                "status": "passed",
                "verdict": verdict,
                "num_targets": len(target_manifest),
                "num_eval_cases": len(eval_rows),
                "num_train_cases": len(train_rows),
                "reproducibility_controls_pass": reproducibility_controls_pass,
                "batch_size_sensitivity_detected": batch_size_sensitivity_detected,
                "manual_padding_parity_pass": manual_padding_pass,
                "peer_padding_parity_pass": peer_padding_pass,
                "train_padding_parity_pass": train_padding_pass,
                "max_eval_probability_diff": max(
                    (float(row["max_abs_probability_diff"]) for row in manual_padding_cases),
                    default=0.0,
                ),
                "max_eval_disagreement_rate": max(
                    (float(row["prediction_disagreement_rate"]) for row in manual_padding_cases),
                    default=0.0,
                ),
                "max_batch_size_probability_diff": max(
                    (float(row["max_abs_probability_diff"]) for row in batch_size_cases),
                    default=0.0,
                ),
                "max_train_gradient_relative_l2_diff": max(
                    (float(row["gradient_relative_l2_diff"]) for row in train_rows),
                    default=0.0,
                ),
                "max_train_bn_buffer_abs_diff": max(
                    (float(row["bn_buffer_max_abs_diff"]) for row in train_rows),
                    default=0.0,
                ),
            }
        )
        write_csv(output_dir / "padding_parity_targets.csv", target_manifest)
        write_csv(output_dir / "eval_padding_parity.csv", eval_rows)
        write_csv(output_dir / "train_padding_parity.csv", train_rows)
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        raise
    finally:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Stage5 PointNeXt-S padding parity check completed.")
    print(f"verdict: {summary['verdict']}")
    print(f"summary: {summary_path}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
