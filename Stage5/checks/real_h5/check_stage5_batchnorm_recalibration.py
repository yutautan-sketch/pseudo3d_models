from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from checks.real_h5.check_stage5_batch_integrity import array_sha256
from checks.real_h5.check_stage5_batchnorm_mode_parity import (
    run_window_forward,
    summarize_diff_group,
)
from checks.real_h5.check_stage5_overlap_aggregation import (
    check_mean_probability_parity,
    load_reference_h5_metrics,
)
from checks.real_h5.check_stage5_padding_parity import set_seed
from evaluate_stage5 import (
    MetricTotals,
    compute_metrics,
    model_from_checkpoint,
    path_video_name,
    write_csv,
)
from export_anonymized_stage5_metrics import assert_share_bundle_anonymous, build_aliases
from infer_stage5 import build_features, load_checkpoint
from stage5.utils.feature_normalization import normalize_xyz
from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list


MODEL_NAMES: tuple[str, ...] = ("original", "recalibrated")
GT_CLASS_IDS: dict[str, int] = {"valid_positive": 0, "valid_background": 1, "ignore": 2}
DIFF_ROW_NAMES: tuple[str, ...] = ("valid_positive", "valid_background", "ignore", "all")
BN_BUFFER_SUFFIXES: tuple[str, ...] = (".running_mean", ".running_var", ".num_batches_tracked")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_array_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator_safe = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, numerator / denominator_safe, 0.0)


class ParityMismatchError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Parameter / non-BatchNorm buffer invariance (requirement 2, 10)
# ----------------------------------------------------------------------------


def combined_hash(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(tensor.detach().cpu().numpy()).encode("ascii"))
    return digest.hexdigest()


def parameter_hash(model: nn.Module) -> str:
    return combined_hash(list(model.named_parameters()))


def non_bn_buffer_hash(model: nn.Module) -> str:
    buffers = [
        (name, tensor)
        for name, tensor in model.named_buffers()
        if not name.endswith(BN_BUFFER_SUFFIXES)
    ]
    return combined_hash(buffers)


# ----------------------------------------------------------------------------
# BatchNorm recalibration (requirements 1, 3-9)
# ----------------------------------------------------------------------------


def configure_bn_for_recalibration(model: nn.Module) -> tuple[dict[str, float | None], int]:
    """Reset BN running stats and switch only BN submodules to train() mode.

    The rest of the model (including any dropout) stays in eval() mode, so
    this is safe regardless of the checkpoint's configured dropout value.
    """
    model.eval()
    original_momentum: dict[str, float | None] = {}
    bn_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            original_momentum[name] = module.momentum
            module.reset_running_stats()
            module.momentum = None
            module.train()
            bn_count += 1
    require(bn_count > 0, "No BatchNorm modules found for recalibration")
    return original_momentum, bn_count


def run_recalibration_pass(
    model: nn.Module,
    paths: list[Path],
    *,
    features: tuple[str, ...],
    normalize_points: bool,
    device: torch.device,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    seed_base: int,
) -> dict[str, int]:
    processed_files = 0
    processed_windows = 0
    for path in paths:
        data = load_stage5_pointcloud_h5(path)
        raw_points = data["points"].astype(np.float32)
        points = normalize_xyz(raw_points) if normalize_points else raw_points
        feature_values = build_features(data, features)
        frame_order = data["frame_order"]
        windows = generate_frame_order_windows(
            frame_order,
            window_size_frames=window_size_frames,
            window_stride_frames=window_stride_frames,
            include_tail_window=include_tail_window,
        )
        require(bool(windows), f"No frame windows generated for {path}")
        for window_offset, window in enumerate(windows):
            indices = point_indices_for_window(frame_order, window)
            if indices.size == 0:
                continue
            seed = seed_base + window_offset
            run_window_forward(
                model,
                points[indices],
                feature_values[indices],
                device=device,
                seed=seed,
                h5_path=str(path),
            )
            processed_windows += 1
        processed_files += 1
    return {"processed_files": processed_files, "processed_windows": processed_windows}


def require_calibration_validation_disjoint(
    calibration_paths: list[Path], validation_paths: list[Path]
) -> None:
    overlap = set(calibration_paths) & set(validation_paths)
    require(not overlap, f"Calibration set overlaps validation set: {sorted(overlap)[:3]}")
    resolved_overlap = {p.resolve() for p in calibration_paths} & {p.resolve() for p in validation_paths}
    require(
        not resolved_overlap,
        f"Calibration set overlaps validation set after Path.resolve(): {sorted(resolved_overlap)[:3]}",
    )


def assert_bn_calibration_health(model: nn.Module, *, expected_num_batches_tracked: int) -> None:
    """Fail-fast health check for BN buffers right after recalibration (requirements 1-3)."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        running_mean = module.running_mean.detach().cpu().numpy()
        running_var = module.running_var.detach().cpu().numpy()
        require(
            bool(np.all(np.isfinite(running_mean))),
            f"BatchNorm module {name!r} running_mean is non-finite after recalibration",
        )
        require(
            bool(np.all(np.isfinite(running_var))),
            f"BatchNorm module {name!r} running_var is non-finite after recalibration",
        )
        require(
            bool(np.all(running_var >= 0)),
            f"BatchNorm module {name!r} running_var has negative values after recalibration",
        )
        actual_tracked = int(module.num_batches_tracked.item())
        require(
            actual_tracked == expected_num_batches_tracked,
            f"BatchNorm module {name!r} num_batches_tracked={actual_tracked} does not match "
            f"calibration_processed_windows={expected_num_batches_tracked}",
        )


def save_recalibrated_checkpoint(
    model: nn.Module,
    *,
    config: dict[str, Any],
    original_checkpoint_epoch: Any,
    calibration_stats: dict[str, int],
    original_momentum: dict[str, float | None],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": original_checkpoint_epoch,
            "recalibration_meta": {
                "policy": "momentum_none_cumulative_average",
                "processed_files": calibration_stats["processed_files"],
                "processed_windows": calibration_stats["processed_windows"],
                "original_momentum": original_momentum,
            },
        },
        output_path,
    )


def bn_buffer_diff_rows(model_original: nn.Module, model_recalibrated: nn.Module) -> list[dict[str, Any]]:
    original_bn = {
        name: module
        for name, module in model_original.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    recalibrated_bn = {
        name: module
        for name, module in model_recalibrated.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    require(
        set(original_bn) == set(recalibrated_bn),
        "BatchNorm module sets differ between original and recalibrated models",
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(original_bn):
        original_module = original_bn[name]
        recalibrated_module = recalibrated_bn[name]
        original_mean = original_module.running_mean.detach().cpu().numpy()
        recalibrated_mean = recalibrated_module.running_mean.detach().cpu().numpy()
        original_var = original_module.running_var.detach().cpu().numpy()
        recalibrated_var = recalibrated_module.running_var.detach().cpu().numpy()
        mean_diff = np.abs(original_mean - recalibrated_mean)
        var_diff = np.abs(original_var - recalibrated_var)
        rows.append(
            {
                "module_name": name,
                "num_features": int(original_module.num_features),
                "running_mean_abs_diff_mean": float(np.mean(mean_diff)),
                "running_mean_abs_diff_max": float(np.max(mean_diff)),
                "running_var_abs_diff_mean": float(np.mean(var_diff)),
                "running_var_abs_diff_max": float(np.max(var_diff)),
                "recalibrated_running_mean_finite": bool(np.all(np.isfinite(recalibrated_mean))),
                "recalibrated_running_var_finite": bool(np.all(np.isfinite(recalibrated_var))),
                "recalibrated_running_var_negative_count": int(np.sum(recalibrated_var < 0)),
                "original_num_batches_tracked": int(original_module.num_batches_tracked.item()),
                "recalibrated_num_batches_tracked": int(recalibrated_module.num_batches_tracked.item()),
            }
        )
    return rows


# ----------------------------------------------------------------------------
# Evaluation: original vs recalibrated, both in plain eval() mode.
# ----------------------------------------------------------------------------


def process_h5_pair(
    *,
    model_original: nn.Module,
    model_recalibrated: nn.Module,
    path: Path,
    split: str,
    alias: str,
    features: tuple[str, ...],
    normalize_points: bool,
    device: torch.device,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    ignore_index: int,
    seed_base: int,
    reference: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(path)
    video_name = path_video_name(path, data)
    labels = data["point_label"]
    valid_mask = data["valid_mask"]
    frame_order = data["frame_order"]
    raw_points = data["points"].astype(np.float32)
    points = normalize_xyz(raw_points) if normalize_points else raw_points
    feature_values = build_features(data, features)

    windows = generate_frame_order_windows(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    require(bool(windows), f"No frame windows generated for {path}")

    num_points = points.shape[0]
    models = {"original": model_original, "recalibrated": model_recalibrated}
    prob_sum = {name: np.zeros(num_points, dtype=np.float64) for name in MODEL_NAMES}
    vote_count = {name: np.zeros(num_points, dtype=np.int32) for name in MODEL_NAMES}
    window_totals = {name: MetricTotals.empty() for name in MODEL_NAMES}
    diff_chunks_abs: list[np.ndarray] = []
    diff_chunks_disagree: list[np.ndarray] = []
    diff_chunks_gt: list[np.ndarray] = []

    for window_offset, window in enumerate(windows):
        indices = point_indices_for_window(frame_order, window)
        if indices.size == 0:
            continue
        seed = seed_base + window_offset
        probability_by_model: dict[str, np.ndarray] = {}
        for name in MODEL_NAMES:
            probability_by_model[name] = run_window_forward(
                models[name],
                points[indices],
                feature_values[indices],
                device=device,
                seed=seed,
                h5_path=str(path),
            )

        window_labels = labels[indices]
        window_valid = valid_mask[indices]
        for name in MODEL_NAMES:
            p1 = probability_by_model[name][:, 1]
            pred = (p1 > 0.5).astype(np.uint8)
            _, counts = compute_metrics(window_labels, window_valid, pred, p1, ignore_index=ignore_index)
            window_totals[name].update(counts)
            prob_sum[name][indices] += p1
            vote_count[name][indices] += 1

        p1_original = probability_by_model["original"][:, 1]
        p1_recalibrated = probability_by_model["recalibrated"][:, 1]
        pred_original = p1_original > 0.5
        pred_recalibrated = p1_recalibrated > 0.5

        valid = window_valid & (window_labels != int(ignore_index))
        target_positive = window_labels == 1
        gt_class = np.full(indices.shape, GT_CLASS_IDS["ignore"], dtype=np.int8)
        gt_class[valid & target_positive] = GT_CLASS_IDS["valid_positive"]
        gt_class[valid & ~target_positive] = GT_CLASS_IDS["valid_background"]

        diff_chunks_abs.append(np.abs(p1_original - p1_recalibrated))
        diff_chunks_disagree.append(pred_original != pred_recalibrated)
        diff_chunks_gt.append(gt_class)

    for name in MODEL_NAMES:
        missing = vote_count[name] == 0
        require(
            not np.any(missing),
            f"{int(np.sum(missing))} points received no {name} prediction in {path}",
        )

    all_abs_diff = np.concatenate(diff_chunks_abs)
    all_disagreement = np.concatenate(diff_chunks_disagree)
    all_gt_class = np.concatenate(diff_chunks_gt)

    diff_rows: list[dict[str, Any]] = []
    for name in DIFF_ROW_NAMES:
        mask = (
            np.ones_like(all_gt_class, dtype=bool)
            if name == "all"
            else all_gt_class == GT_CLASS_IDS[name]
        )
        row = summarize_diff_group(mask, abs_diff=all_abs_diff, disagreement=all_disagreement)
        diff_rows.append({"split": split, "video_alias": alias, "gt_class": name, **row})

    aggregation_metrics: dict[str, dict[str, Any]] = {}
    aggregation_counts: dict[str, dict[str, Any]] = {}
    parity_mismatches: dict[str, Any] = {}
    for name in MODEL_NAMES:
        mean_probability = safe_array_divide(prob_sum[name], vote_count[name].astype(np.float64))
        pred_label = (mean_probability > 0.5).astype(np.uint8)
        metrics, counts = compute_metrics(
            labels, valid_mask, pred_label, mean_probability, ignore_index=ignore_index
        )
        aggregation_metrics[name] = metrics
        aggregation_counts[name] = counts
        if name == "original":
            parity_mismatches = check_mean_probability_parity(
                split=split, video_name=video_name, counts=counts, reference=reference
            )

    window_metrics = {name: window_totals[name].metrics() for name in MODEL_NAMES}
    window_counts = {name: dict(window_totals[name].values) for name in MODEL_NAMES}

    return {
        "video_name": video_name,
        "diff_rows": diff_rows,
        "window_metrics": window_metrics,
        "window_counts": window_counts,
        "aggregation_metrics": aggregation_metrics,
        "aggregation_counts": aggregation_counts,
        "parity_mismatches": parity_mismatches,
    }


# ----------------------------------------------------------------------------
# Split/checkpoint-level rollups
# ----------------------------------------------------------------------------


def build_checkpoint_summary(
    *,
    window_totals: dict[tuple[str, str], MetricTotals],
    aggregation_totals: dict[tuple[str, str], MetricTotals],
    aggregation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for granularity, totals_by_key in (("window", window_totals), ("aggregated", aggregation_totals)):
        for (split, model_name), totals in totals_by_key.items():
            metrics = totals.metrics()
            row: dict[str, Any] = {
                "split": split,
                "model": model_name,
                "granularity": granularity,
                **metrics,
            }
            if granularity == "aggregated":
                video_rows = [
                    r for r in aggregation_rows if r["split"] == split and r["model"] == model_name
                ]
                f1_values = [float(r["f1"]) for r in video_rows]
                iou_values = [float(r["iou_femur"]) for r in video_rows]
                tp_zero_video_count = sum(1 for r in video_rows if int(r["true_positive_count"]) == 0)
                row.update(
                    {
                        "num_videos": len(video_rows),
                        "video_mean_f1": float(np.mean(f1_values)) if f1_values else None,
                        "video_median_f1": float(np.median(f1_values)) if f1_values else None,
                        "video_mean_iou": float(np.mean(iou_values)) if iou_values else None,
                        "video_median_iou": float(np.median(iou_values)) if iou_values else None,
                        "tp_zero_video_count": tp_zero_video_count,
                    }
                )
            rows.append(row)
    return rows


def build_model_deltas(checkpoint_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["split"], row["granularity"], row["model"]): row for row in checkpoint_summary_rows}
    rows: list[dict[str, Any]] = []
    for (split, granularity, model_name), row in by_key.items():
        if model_name != "recalibrated":
            continue
        original_row = by_key.get((split, granularity, "original"))
        if original_row is None:
            continue
        delta: dict[str, Any] = {
            "split": split,
            "granularity": granularity,
            "f1_delta_recalibrated_minus_original": row["f1"] - original_row["f1"],
            "iou_delta_recalibrated_minus_original": row["iou_femur"] - original_row["iou_femur"],
            "recall_delta_recalibrated_minus_original": row["recall"] - original_row["recall"],
            "false_positive_count_delta_recalibrated_minus_original": row["false_positive_count"]
            - original_row["false_positive_count"],
        }
        if granularity == "aggregated":
            delta["tp_zero_video_count_delta_recalibrated_minus_original"] = (
                row["tp_zero_video_count"] - original_row["tp_zero_video_count"]
            )
        rows.append(delta)
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("Stage5 BatchNorm recalibration self-test passed.")
        return

    require(args.checkpoint is not None, "--checkpoint is required unless --self_test is set")
    require(args.train_list is not None, "--train_list is required (calibration set)")
    require(args.selected_train_list is not None, "--selected_train_list is required")
    require(args.val_list is not None, "--val_list is required")
    require(args.selection_summary is not None, "--selection_summary is required")
    require(args.reference_h5_metrics_csv is not None, "--reference_h5_metrics_csv is required")
    require(args.share_output_dir is not None, "--share_output_dir is required")
    require(args.private_output_dir is not None, "--private_output_dir is required")

    device = torch.device(args.device)
    require(device.type == "cuda", "BatchNorm recalibration check requires CUDA")
    require(torch.cuda.is_available(), "CUDA was requested but torch.cuda.is_available() is False")

    share_output_dir = Path(args.share_output_dir)
    private_output_dir = Path(args.private_output_dir)
    share_output_dir.mkdir(parents=True, exist_ok=True)
    private_output_dir.mkdir(parents=True, exist_ok=True)

    train_calibration_paths = read_path_list(args.train_list)
    selected_train_paths = read_path_list(args.selected_train_list)
    val_paths = read_path_list(args.val_list)
    require(bool(train_calibration_paths), "--train_list (calibration set) must not be empty")
    require(bool(selected_train_paths) or bool(val_paths), "No evaluation H5 files to process")
    for path in (*train_calibration_paths, *selected_train_paths, *val_paths):
        require(path.is_file(), f"H5 not found: {path}")
    duplicate_eval_paths = set(selected_train_paths) & set(val_paths)
    require(
        not duplicate_eval_paths,
        f"train_sanity and validation lists overlap: {sorted(duplicate_eval_paths)[:3]}",
    )
    require_calibration_validation_disjoint(train_calibration_paths, val_paths)

    selection_summary = json.loads(Path(args.selection_summary).read_text(encoding="utf-8"))
    reference = load_reference_h5_metrics(Path(args.reference_h5_metrics_csv))

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_epoch = checkpoint.get("epoch")
    model_original, config, features, normalize_points = model_from_checkpoint(
        checkpoint, device=device, strict=args.strict_checkpoint
    )
    model_recalibrated, config_recal, features_recal, normalize_points_recal = model_from_checkpoint(
        checkpoint, device=device, strict=args.strict_checkpoint
    )
    del checkpoint
    require(config == config_recal, "original/recalibrated model instances were built from different configs")
    require(features == features_recal, "original/recalibrated model instances resolved different features")
    require(
        normalize_points == normalize_points_recal,
        "original/recalibrated model instances resolved different normalize_points",
    )
    for param in model_original.parameters():
        param.requires_grad_(False)
    for param in model_recalibrated.parameters():
        param.requires_grad_(False)

    window_size_frames = int(
        args.window_size_frames if args.window_size_frames is not None else config.get("window_size_frames", 12)
    )
    window_stride_frames = int(
        args.window_stride_frames
        if args.window_stride_frames is not None
        else config.get("window_stride_frames", 6)
    )
    include_tail_window = (
        args.include_tail_window
        if args.include_tail_window is not None
        else bool(config.get("include_tail_window", True))
    )
    ignore_index = int(config.get("ignore_index", -1))

    param_hash_before = parameter_hash(model_recalibrated)
    buffer_hash_before = non_bn_buffer_hash(model_recalibrated)

    original_momentum, bn_module_count = configure_bn_for_recalibration(model_recalibrated)
    calibration_stats = run_recalibration_pass(
        model_recalibrated,
        train_calibration_paths,
        features=features,
        normalize_points=normalize_points,
        device=device,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=bool(include_tail_window),
        seed_base=int(args.seed),
    )
    assert_bn_calibration_health(
        model_recalibrated, expected_num_batches_tracked=calibration_stats["processed_windows"]
    )
    model_recalibrated.eval()

    param_hash_after = parameter_hash(model_recalibrated)
    buffer_hash_after = non_bn_buffer_hash(model_recalibrated)
    require(
        param_hash_before == param_hash_after,
        "Model parameters changed during recalibration (expected: frozen)",
    )
    require(
        buffer_hash_before == buffer_hash_after,
        "Non-BatchNorm buffers changed during recalibration (expected: unchanged)",
    )

    recalibrated_checkpoint_path = private_output_dir / "recalibrated_checkpoint_DO_NOT_SHARE.pt"
    save_recalibrated_checkpoint(
        model_recalibrated,
        config=config,
        original_checkpoint_epoch=checkpoint_epoch,
        calibration_stats=calibration_stats,
        original_momentum=original_momentum,
        output_path=recalibrated_checkpoint_path,
    )

    buffer_diff_rows = bn_buffer_diff_rows(model_original, model_recalibrated)

    split_paths = (("train_sanity", selected_train_paths), ("validation", val_paths))
    h5_manifest_rows: list[dict[str, str]] = []
    for split, paths in split_paths:
        for path in paths:
            with h5py.File(path, "r") as handle:
                file_attrs = dict(handle.attrs)
            video_name = path_video_name(path, {"file_attrs": file_attrs})
            h5_manifest_rows.append({"split": split, "video_name": video_name, "h5_path": str(path)})

    aliases, private_rows = build_aliases(h5_manifest_rows, selection_summary)
    write_csv(private_output_dir / "video_id_map_DO_NOT_SHARE.csv", private_rows)
    (private_output_dir / "DO_NOT_SHARE.txt").write_text(
        "This directory contains original video identifiers, H5 paths, and the recalibrated checkpoint.\n",
        encoding="utf-8",
    )

    diff_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    aggregation_metric_rows: list[dict[str, Any]] = []
    parity_failure_rows: list[dict[str, Any]] = []
    window_totals: dict[tuple[str, str], MetricTotals] = {
        (split, name): MetricTotals.empty() for split, _ in split_paths for name in MODEL_NAMES
    }
    aggregation_totals: dict[tuple[str, str], MetricTotals] = {
        (split, name): MetricTotals.empty() for split, _ in split_paths for name in MODEL_NAMES
    }

    summary: dict[str, Any] = {
        "status": "running",
        "run_name": Path(args.checkpoint).resolve().parent.name,
        "checkpoint_name": Path(args.checkpoint).name,
        "features": list(features),
        "normalize_points": normalize_points,
        "ignore_index": ignore_index,
        "window_size_frames": window_size_frames,
        "window_stride_frames": window_stride_frames,
        "include_tail_window": bool(include_tail_window),
        "models": list(MODEL_NAMES),
        "mean_baseline_parity_reference": Path(args.reference_h5_metrics_csv).name,
        "batchnorm_module_count": bn_module_count,
        "calibration_processed_files": calibration_stats["processed_files"],
        "calibration_processed_windows": calibration_stats["processed_windows"],
        "recalibration_policy": "momentum_none_cumulative_average",
        "parameter_hash_invariant": True,
        "non_bn_buffer_hash_invariant": True,
        "bn_calibration_health_passed": True,
    }
    summary_path = share_output_dir / "batchnorm_recalibration_summary.json"
    private_run_info_path = private_output_dir / "batchnorm_recalibration_run_info_DO_NOT_SHARE.json"
    with private_run_info_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "reference_h5_metrics_csv": str(Path(args.reference_h5_metrics_csv).resolve()),
                "train_list": str(Path(args.train_list).resolve()),
                "selected_train_list": str(Path(args.selected_train_list).resolve()),
                "val_list": str(Path(args.val_list).resolve()),
                "recalibrated_checkpoint": str(recalibrated_checkpoint_path.resolve()),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    try:
        seed_base = int(args.seed)
        for split, paths in split_paths:
            for path in tqdm(paths, desc=f"batchnorm recalibration {split}"):
                video_name = next(
                    (
                        row["video_name"]
                        for row in h5_manifest_rows
                        if row["split"] == split and row["h5_path"] == str(path)
                    ),
                    None,
                )
                require(video_name is not None, f"Missing manifest entry for {path}")
                alias = aliases[(split, video_name)]

                result = process_h5_pair(
                    model_original=model_original,
                    model_recalibrated=model_recalibrated,
                    path=path,
                    split=split,
                    alias=alias,
                    features=features,
                    normalize_points=normalize_points,
                    device=device,
                    window_size_frames=window_size_frames,
                    window_stride_frames=window_stride_frames,
                    include_tail_window=bool(include_tail_window),
                    ignore_index=ignore_index,
                    seed_base=seed_base,
                    reference=reference,
                )

                if result["parity_mismatches"]:
                    parity_failure_rows.append(
                        {
                            "split": split,
                            "video_name": result["video_name"],
                            "video_alias": alias,
                            "mismatches": json.dumps(result["parity_mismatches"], ensure_ascii=False),
                        }
                    )
                    write_csv(private_output_dir / "mean_baseline_parity_failure.csv", parity_failure_rows)
                    raise ParityMismatchError(
                        f"original-model mean_probability parity mismatch for split={split!r} "
                        f"video={result['video_name']!r}: {result['parity_mismatches']}"
                    )

                diff_rows.extend(result["diff_rows"])
                for name in MODEL_NAMES:
                    window_totals[(split, name)].update(result["window_counts"][name])
                    aggregation_totals[(split, name)].update(result["aggregation_counts"][name])
                    window_metric_rows.append(
                        {"split": split, "video_alias": alias, "model": name, **result["window_metrics"][name]}
                    )
                    aggregation_metric_rows.append(
                        {
                            "split": split,
                            "video_alias": alias,
                            "model": name,
                            **result["aggregation_metrics"][name],
                        }
                    )

        checkpoint_summary_rows = build_checkpoint_summary(
            window_totals=window_totals,
            aggregation_totals=aggregation_totals,
            aggregation_rows=aggregation_metric_rows,
        )
        model_delta_rows = build_model_deltas(checkpoint_summary_rows)

        write_csv(share_output_dir / "batchnorm_recalibration_probability_difference.csv", diff_rows)
        write_csv(share_output_dir / "batchnorm_recalibration_window_metrics.csv", window_metric_rows)
        write_csv(share_output_dir / "batchnorm_recalibration_aggregation_metrics.csv", aggregation_metric_rows)
        write_csv(share_output_dir / "batchnorm_recalibration_checkpoint_summary.csv", checkpoint_summary_rows)
        write_csv(share_output_dir / "batchnorm_recalibration_model_delta.csv", model_delta_rows)
        write_csv(share_output_dir / "batchnorm_recalibration_buffer_diff.csv", buffer_diff_rows)

        summary.update(
            {
                "status": "passed",
                "mean_baseline_parity": "passed",
                "num_videos": len(selected_train_paths) + len(val_paths),
                "checkpoint_summary": checkpoint_summary_rows,
                "model_delta": model_delta_rows,
            }
        )
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        if isinstance(error, ParityMismatchError):
            summary["mean_baseline_parity"] = "failed"
            summary["parity_failure_csv"] = "mean_baseline_parity_failure.csv (in private_output_dir)"
        raise
    finally:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    try:
        assert_share_bundle_anonymous(share_output_dir, private_rows=private_rows)
    except Exception as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["error"] = f"share bundle anonymization check failed: {error}"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        raise

    print("Stage5 BatchNorm recalibration checker passed.")
    print(f"checkpoint: {args.checkpoint}")
    print(f"calibration files/windows: {calibration_stats['processed_files']}/{calibration_stats['processed_windows']}")
    print(f"videos evaluated: {summary['num_videos']}")
    print(f"share output: {share_output_dir}")
    print(f"private output: {private_output_dir}")


# ----------------------------------------------------------------------------
# Synthetic self-test
# ----------------------------------------------------------------------------


class _DummyBufferModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("dummy_buffer", torch.zeros(2))
        self.linear = nn.Linear(3, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.linear(x))


def test_cumulative_update_matches_expected_value() -> None:
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(3)
    original_momentum, bn_count = configure_bn_for_recalibration(bn)
    require(bn_count == 1, f"Expected exactly one BatchNorm module, got {bn_count}")
    require(
        original_momentum[""] == 0.1,
        f"Default BatchNorm1d momentum must be 0.1 before override, got {original_momentum['']}",
    )
    require(bn.momentum is None, "momentum must be set to None for cumulative averaging")

    batches = [torch.randn(5, 3) for _ in range(4)]
    with torch.inference_mode():
        for batch in batches:
            bn(batch)

    expected_mean = torch.stack([b.mean(dim=0) for b in batches]).mean(dim=0)
    expected_var = torch.stack([b.var(dim=0, unbiased=True) for b in batches]).mean(dim=0)
    require(
        torch.allclose(bn.running_mean, expected_mean, atol=1e-5),
        "running_mean must equal the mean of per-batch means under momentum=None",
    )
    require(
        torch.allclose(bn.running_var, expected_var, atol=1e-5),
        "running_var must equal the mean of per-batch unbiased variances under momentum=None",
    )
    require(
        int(bn.num_batches_tracked.item()) == 4,
        "num_batches_tracked must equal the number of forward calls",
    )


def test_only_bn_state_changes_during_recalibration() -> None:
    torch.manual_seed(1)
    model = _DummyBufferModule()
    for param in model.parameters():
        param.requires_grad_(False)
    param_hash_before = parameter_hash(model)
    buffer_hash_before = non_bn_buffer_hash(model)

    configure_bn_for_recalibration(model)
    require(model.linear.training is False, "Non-BatchNorm submodules must remain in eval() mode")
    require(model.bn.training is True, "The BatchNorm submodule must be switched to train() mode")
    with torch.inference_mode():
        for _ in range(3):
            model(torch.randn(4, 3))
    model.eval()

    require(
        parameter_hash(model) == param_hash_before,
        "Parameters must not change during recalibration (frozen + no backward)",
    )
    require(
        non_bn_buffer_hash(model) == buffer_hash_before,
        "Non-BatchNorm buffers must not change during recalibration",
    )
    require(
        int(model.bn.num_batches_tracked.item()) == 3,
        "BatchNorm num_batches_tracked must reflect the number of forward calls",
    )

    # Negative control: non_bn_buffer_hash must actually detect a genuine change.
    model.dummy_buffer += 1.0
    require(
        non_bn_buffer_hash(model) != buffer_hash_before,
        "non_bn_buffer_hash must change when a non-BatchNorm buffer is mutated",
    )


def test_require_calibration_validation_disjoint_rejects_overlap() -> None:
    calibration_paths = [Path("a.h5"), Path("b.h5")]
    validation_paths = [Path("b.h5"), Path("c.h5")]
    raised = False
    try:
        require_calibration_validation_disjoint(calibration_paths, validation_paths)
    except AssertionError:
        raised = True
    require(raised, "Overlapping calibration/validation sets must be rejected")

    require_calibration_validation_disjoint(calibration_paths, [Path("c.h5")])


def test_assert_bn_calibration_health_passes_for_healthy_calibration() -> None:
    torch.manual_seed(5)
    bn = nn.BatchNorm1d(3)
    configure_bn_for_recalibration(bn)
    with torch.inference_mode():
        for _ in range(4):
            bn(torch.randn(5, 3))
    assert_bn_calibration_health(bn, expected_num_batches_tracked=4)


def test_assert_bn_calibration_health_rejects_num_batches_tracked_mismatch() -> None:
    torch.manual_seed(6)
    bn = nn.BatchNorm1d(3)
    configure_bn_for_recalibration(bn)
    with torch.inference_mode():
        for _ in range(4):
            bn(torch.randn(5, 3))
    raised = False
    try:
        assert_bn_calibration_health(bn, expected_num_batches_tracked=5)
    except AssertionError:
        raised = True
    require(raised, "assert_bn_calibration_health must reject a num_batches_tracked mismatch")


def test_assert_bn_calibration_health_rejects_non_finite_and_negative_stats() -> None:
    torch.manual_seed(7)
    bn = nn.BatchNorm1d(3)
    configure_bn_for_recalibration(bn)
    with torch.inference_mode():
        bn(torch.randn(5, 3))

    with torch.no_grad():
        bn.running_var[0] = -1.0
    raised = False
    try:
        assert_bn_calibration_health(bn, expected_num_batches_tracked=1)
    except AssertionError:
        raised = True
    require(raised, "assert_bn_calibration_health must reject negative running_var")

    with torch.no_grad():
        bn.running_var[0] = 1.0
        bn.running_mean[0] = float("nan")
    raised = False
    try:
        assert_bn_calibration_health(bn, expected_num_batches_tracked=1)
    except AssertionError:
        raised = True
    require(raised, "assert_bn_calibration_health must reject non-finite running_mean")


def test_require_calibration_validation_disjoint_uses_resolved_paths() -> None:
    real_path = Path(__file__).resolve()
    aliased_path = Path(__file__).resolve().parent / ".." / Path(__file__).resolve().parent.name / Path(__file__).name
    require(
        real_path != aliased_path,
        "test setup requires an unresolved alias path that differs textually from the resolved path",
    )
    require(
        aliased_path.resolve() == real_path,
        "test setup requires the alias path to resolve to the same file as real_path",
    )
    raised = False
    try:
        require_calibration_validation_disjoint([real_path], [aliased_path])
    except AssertionError:
        raised = True
    require(raised, "Calibration/validation overlap must be detected after Path.resolve()")


def test_recalibration_is_deterministic_across_reruns() -> None:
    def build_and_calibrate(seed_base: int) -> nn.BatchNorm1d:
        torch.manual_seed(123)
        bn = nn.BatchNorm1d(2)
        configure_bn_for_recalibration(bn)
        with torch.inference_mode():
            for offset in range(3):
                generator = torch.Generator().manual_seed(100 + offset)
                batch = torch.randn(4, 2, generator=generator)
                set_seed(seed_base + offset)
                bn(batch)
        return bn

    bn_a = build_and_calibrate(seed_base=7)
    bn_b = build_and_calibrate(seed_base=7)
    require(
        torch.allclose(bn_a.running_mean, bn_b.running_mean),
        "Re-running recalibration with the same inputs/seed must reproduce running_mean",
    )
    require(
        torch.allclose(bn_a.running_var, bn_b.running_var),
        "Re-running recalibration with the same inputs/seed must reproduce running_var",
    )
    require(
        int(bn_a.num_batches_tracked.item()) == int(bn_b.num_batches_tracked.item()),
        "Re-running recalibration must reproduce num_batches_tracked",
    )


def run_self_test() -> None:
    test_cumulative_update_matches_expected_value()
    test_only_bn_state_changes_during_recalibration()
    test_require_calibration_validation_disjoint_rejects_overlap()
    test_assert_bn_calibration_health_passes_for_healthy_calibration()
    test_assert_bn_calibration_health_rejects_num_batches_tracked_mismatch()
    test_assert_bn_calibration_health_rejects_non_finite_and_negative_stats()
    test_require_calibration_validation_disjoint_uses_resolved_paths()
    test_recalibration_is_deterministic_across_reruns()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalibrate PointNeXt-S BatchNorm running statistics on the training split "
        "and compare against the original checkpoint (diagnostic only; no production change)."
    )
    parser.add_argument("--self_test", action="store_true", help="Run synthetic tests and exit")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train_list", default=None, help="Calibration set (e.g. RUN_DIR/train_files.txt)")
    parser.add_argument("--selected_train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument(
        "--selection_summary",
        default=None,
        help="Path to evaluation_data/summary.json produced by prepare_stage5_evaluation_data.py",
    )
    parser.add_argument(
        "--reference_h5_metrics_csv",
        default=None,
        help="Existing evaluate_stage5.py h5_metrics.csv for the same checkpoint (original-model sanity parity)",
    )
    parser.add_argument("--share_output_dir", default=None)
    parser.add_argument("--private_output_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--window_size_frames", type=int, default=None)
    parser.add_argument("--window_stride_frames", type=int, default=None)
    parser.add_argument(
        "--include_tail_window",
        dest="include_tail_window",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no_include_tail_window",
        dest="include_tail_window",
        action="store_false",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
