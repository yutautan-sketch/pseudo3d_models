from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from infer_stage5 import build_features, checkpoint_config, load_checkpoint, parse_feature_list
from stage5.models import build_stage5_model
from stage5.models.model_factory import _extract_state_dict
from stage5.utils.feature_normalization import normalize_xyz
from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list
from stage5.utils.visualization_export import (
    DIAGNOSTIC_CATEGORY_COLORS,
    DIAGNOSTIC_CATEGORY_NAMES,
    write_diagnostic_segmentation_ply,
    write_probability_ply,
    write_selected_probability_ply,
)


DEFAULT_FIXED_TRAIN_VIDEO = "20250626_124212_7300"
COUNT_KEYS = (
    "true_positive_count",
    "false_positive_count",
    "true_negative_count",
    "false_negative_count",
    "total_point_count",
    "valid_point_count",
    "valid_positive_count",
    "valid_background_count",
    "ignore_point_count",
    "ignore_predicted_positive_count",
)


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def metrics_from_counts(counts: dict[str, float | int]) -> dict[str, float | int | None]:
    tp = int(counts["true_positive_count"])
    fp = int(counts["false_positive_count"])
    tn = int(counts["true_negative_count"])
    fn = int(counts["false_negative_count"])
    total = int(counts["total_point_count"])
    valid = int(counts["valid_point_count"])
    positive = int(counts["valid_positive_count"])
    background = int(counts["valid_background_count"])
    ignore = int(counts["ignore_point_count"])
    ignore_positive = int(counts["ignore_predicted_positive_count"])
    ignore_probability_sum = float(counts.get("ignore_probability_sum", 0.0))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    iou_femur = safe_divide(tp, tp + fp + fn)
    iou_background = safe_divide(tn, tn + fp + fn)
    return {
        "total_point_count": total,
        "valid_point_count": valid,
        "valid_positive_count": positive,
        "valid_background_count": background,
        "valid_ratio": safe_divide(valid, total),
        "valid_positive_ratio": safe_divide(positive, valid),
        "valid_background_ratio": safe_divide(background, valid),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "predicted_positive_count": tp + fp,
        "predicted_background_count": tn + fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2.0 * precision * recall, precision + recall),
        "iou_femur": iou_femur,
        "iou_background": iou_background,
        "mean_iou": 0.5 * (iou_femur + iou_background),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
        "ignore_point_count": ignore,
        "ignore_predicted_positive_count": ignore_positive,
        "ignore_predicted_positive_ratio": safe_divide(ignore_positive, ignore),
        "ignore_mean_prob_femur": (
            ignore_probability_sum / float(ignore) if ignore else None
        ),
    }


def compute_metrics(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    prob_femur: np.ndarray,
    *,
    ignore_index: int,
) -> tuple[dict[str, float | int | None], dict[str, float | int]]:
    labels = np.asarray(labels)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    pred_label = np.asarray(pred_label)
    prob_femur = np.asarray(prob_femur, dtype=np.float32)
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {labels.shape}")
    if any(value.shape != labels.shape for value in (valid_mask, pred_label, prob_femur)):
        raise ValueError("labels, valid_mask, pred_label, and prob_femur must share shape [N]")

    valid = valid_mask & (labels != int(ignore_index))
    unexpected = np.setdiff1d(np.unique(labels[valid]), np.asarray([0, 1], dtype=labels.dtype))
    if unexpected.size:
        raise ValueError(f"Valid points contain non-binary labels: {unexpected.tolist()}")

    target_positive = labels == 1
    predicted_positive = pred_label == 1
    ignore = ~valid
    counts: dict[str, float | int] = {
        "true_positive_count": int(np.sum(valid & target_positive & predicted_positive)),
        "false_positive_count": int(np.sum(valid & ~target_positive & predicted_positive)),
        "true_negative_count": int(np.sum(valid & ~target_positive & ~predicted_positive)),
        "false_negative_count": int(np.sum(valid & target_positive & ~predicted_positive)),
        "total_point_count": int(labels.size),
        "valid_point_count": int(np.sum(valid)),
        "valid_positive_count": int(np.sum(valid & target_positive)),
        "valid_background_count": int(np.sum(valid & ~target_positive)),
        "ignore_point_count": int(np.sum(ignore)),
        "ignore_predicted_positive_count": int(np.sum(ignore & predicted_positive)),
        "ignore_probability_sum": float(np.sum(prob_femur[ignore], dtype=np.float64)),
    }
    return metrics_from_counts(counts), counts


@dataclass
class MetricTotals:
    values: dict[str, float]

    @classmethod
    def empty(cls) -> "MetricTotals":
        return cls({key: 0.0 for key in (*COUNT_KEYS, "ignore_probability_sum")})

    def update(self, counts: dict[str, float | int]) -> None:
        for key in self.values:
            self.values[key] += float(counts[key])

    def metrics(self) -> dict[str, float | int | None]:
        return metrics_from_counts(self.values)


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    strict: bool,
) -> tuple[torch.nn.Module, dict[str, Any], tuple[str, ...], bool]:
    config = checkpoint_config(checkpoint)
    features = parse_feature_list(config.get("feature_names"))
    feature_dim = sum(2 if name == "pixel_xy" else 1 for name in features)
    configured_feature_dim = int(config.get("feature_dim", feature_dim))
    if configured_feature_dim != feature_dim:
        raise ValueError(
            f"Checkpoint feature_dim={configured_feature_dim} does not match features={features}"
        )

    model_name = str(config.get("model", "mlp_baseline"))
    kwargs: dict[str, Any] = {
        "num_classes": int(config.get("num_classes", 2)),
        "feature_dim": feature_dim,
        "width": int(config.get("width", 64)),
        "depth": int(config.get("depth", 6)),
        "expansion": int(config.get("expansion", 4)),
        "dropout": float(config.get("dropout", 0.1)),
        "use_global_context": not bool(config.get("no_global_context", False)),
    }
    if model_name.lower() == "pointnext_s":
        kwargs.update(
            {
                "pointnext_radius": float(
                    config.get("pointnext_radius", config.get("radius", 0.1))
                ),
                "pointnext_nsample": int(
                    config.get("pointnext_nsample", config.get("nsample", 16))
                ),
                "pointnext_sa_layers": int(
                    config.get("pointnext_sa_layers", config.get("sa_layers", 2))
                ),
                "pointnext_sa_use_res": bool(
                    config.get("pointnext_sa_use_res", config.get("sa_use_res", True))
                ),
            }
        )

    model = build_stage5_model(model_name, **kwargs).to(device)
    if int(model.num_classes) != 2:
        raise ValueError(
            f"Stage5 binary evaluation requires num_classes=2, got {model.num_classes}"
        )
    load_result = model.load_state_dict(_extract_state_dict(checkpoint), strict=strict)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"checkpoint missing keys: {load_result.missing_keys}")
        print(f"checkpoint unexpected keys: {load_result.unexpected_keys}")
    model.eval()
    normalize_points = not bool(config.get("no_normalize_points", False))
    return model, config, features, normalize_points


def path_video_name(path: Path, data: dict[str, Any]) -> str:
    value = data.get("file_attrs", {}).get("video_name", "")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip()
    if value:
        return value
    match = re.search(r"\d{8}_\d{6}_\d+", path.name)
    return match.group(0) if match else path.stem


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def select_train_paths(
    paths: list[Path],
    *,
    fixed_videos: Iterable[str],
    num_random: int,
    seed: int,
) -> list[Path]:
    selected: list[Path] = []
    for video in fixed_videos:
        matches = [path for path in paths if video in path.name]
        if len(matches) != 1:
            raise ValueError(
                f"Fixed train video {video!r} matched {len(matches)} paths in train list"
            )
        if matches[0] not in selected:
            selected.append(matches[0])

    candidates = [path for path in paths if path not in selected]
    if num_random < 0:
        raise ValueError("--num_random_train must be non-negative")
    if num_random > len(candidates):
        raise ValueError(
            f"Requested {num_random} random train files but only {len(candidates)} are available"
        )
    rng = random.Random(seed)
    selected.extend(rng.sample(candidates, num_random))
    return selected


def predict_h5(
    *,
    model: torch.nn.Module,
    data: dict[str, Any],
    features: tuple[str, ...],
    normalize_points: bool,
    device: torch.device,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    ignore_index: int,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    raw_points = data["points"].astype(np.float32)
    points = normalize_xyz(raw_points) if normalize_points else raw_points
    feature_values = build_features(data, features)
    frame_order = data["frame_order"]
    labels = data["point_label"].astype(np.int64)
    valid_mask = data["valid_mask"].astype(bool)
    windows = generate_frame_order_windows(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    if not windows:
        raise ValueError(f"No frame windows generated for {metadata['h5_path']}")

    num_classes = int(model.num_classes)
    probability_sum = np.zeros((points.shape[0], num_classes), dtype=np.float64)
    vote_count = np.zeros((points.shape[0],), dtype=np.int32)
    window_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for window in windows:
            indices = point_indices_for_window(frame_order, window)
            if indices.size == 0:
                continue
            batch = {
                "points": torch.from_numpy(points[indices][None]).float().to(device),
                "features": torch.from_numpy(feature_values[indices][None]).float().to(device),
            }
            logits = model(batch)["logits"][0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
            window_pred = np.argmax(probabilities, axis=1).astype(np.uint8)
            window_metrics, _ = compute_metrics(
                labels[indices],
                valid_mask[indices],
                window_pred,
                probabilities[:, 1],
                ignore_index=ignore_index,
            )
            window_rows.append(
                {
                    **metadata,
                    "window_id": int(window.window_id),
                    "window_start": int(window.start_frame),
                    "window_end": int(window.end_frame),
                    **window_metrics,
                }
            )
            probability_sum[indices] += probabilities.astype(np.float64)
            vote_count[indices] += 1

    missing = vote_count == 0
    if np.any(missing):
        raise RuntimeError(
            f"{int(np.sum(missing))} points received no prediction in {metadata['h5_path']}"
        )
    aggregated_probability = (
        probability_sum / vote_count[:, None].astype(np.float64)
    ).astype(np.float32)
    pred_label = np.argmax(aggregated_probability, axis=1).astype(np.uint8)
    return aggregated_probability[:, 1], pred_label, vote_count, window_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_path_list(path: Path, paths: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in paths), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = read_path_list(args.train_list) if args.train_list else []
    val_paths = read_path_list(args.val_list) if args.val_list else []
    if not train_paths and not val_paths:
        raise ValueError("Specify --train_list and/or --val_list")
    for path in (*train_paths, *val_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation H5 not found: {path}")
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(f"Train and validation lists overlap: {sorted(overlap)[:3]}")

    fixed_videos = args.fixed_train_video or [DEFAULT_FIXED_TRAIN_VIDEO]
    if args.selected_train_list:
        selected_train = read_path_list(args.selected_train_list)
        train_path_set = set(train_paths)
        unknown = [path for path in selected_train if path not in train_path_set]
        if unknown:
            raise ValueError(
                f"Selected train list contains paths outside --train_list: {unknown[:3]}"
            )
    else:
        selected_train = select_train_paths(
            train_paths,
            fixed_videos=fixed_videos,
            num_random=args.num_random_train,
            seed=args.selection_seed,
        ) if train_paths else []
    write_path_list(output_dir / "selected_train_files.txt", selected_train)
    write_path_list(output_dir / "validation_files.txt", val_paths)

    # Keep optimizer state and other checkpoint payloads off the GPU.
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_epoch = checkpoint.get("epoch")
    model, config, features, normalize_points = model_from_checkpoint(
        checkpoint,
        device=device,
        strict=args.strict_checkpoint,
    )
    del checkpoint
    window_size_frames = int(
        args.window_size_frames
        if args.window_size_frames is not None
        else config.get("window_size_frames", 12)
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

    checkpoint_name = Path(args.checkpoint).stem
    h5_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    seen_videos: set[tuple[str, str]] = set()

    split_paths = (("train_sanity", selected_train), ("validation", val_paths))
    for split, paths in split_paths:
        totals = MetricTotals.empty()
        for path in tqdm(paths, desc=f"evaluate {split}"):
            data = load_stage5_pointcloud_h5(path)
            video_name = path_video_name(path, data)
            if (split, video_name) in seen_videos:
                raise ValueError(f"Duplicate video name in {split}: {video_name}")
            seen_videos.add((split, video_name))
            metadata = {
                "checkpoint": checkpoint_name,
                "split": split,
                "video_name": video_name,
                "h5_path": str(path),
            }
            prob_femur, pred_label, vote_count, current_window_rows = predict_h5(
                model=model,
                data=data,
                features=features,
                normalize_points=normalize_points,
                device=device,
                window_size_frames=window_size_frames,
                window_stride_frames=window_stride_frames,
                include_tail_window=bool(include_tail_window),
                ignore_index=ignore_index,
                metadata=metadata,
            )
            window_rows.extend(current_window_rows)
            h5_metrics, counts = compute_metrics(
                data["point_label"],
                data["valid_mask"],
                pred_label,
                prob_femur,
                ignore_index=ignore_index,
            )
            totals.update(counts)
            h5_rows.append({**metadata, "num_windows": len(current_window_rows), **h5_metrics})

            name = safe_name(video_name)
            if not args.no_save_predictions:
                prediction_path = output_dir / "predictions" / split / f"{name}.npz"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    prediction_path,
                    point_indices=np.arange(pred_label.size, dtype=np.int64),
                    prob_femur=prob_femur.astype(np.float32),
                    pred_label=pred_label.astype(np.uint8),
                    vote_count=vote_count.astype(np.int32),
                )
            if not args.no_write_ply:
                ply_dir = output_dir / "ply" / split
                write_probability_ply(
                    ply_dir / f"{name}_probability.ply",
                    data["points"],
                    prob_femur,
                )
                write_diagnostic_segmentation_ply(
                    ply_dir / f"{name}_diagnostic.ply",
                    data["points"],
                    data["point_label"],
                    data["valid_mask"],
                    pred_label,
                    ignore_index=ignore_index,
                )
                write_selected_probability_ply(
                    ply_dir / f"{name}_predicted_positive.ply",
                    data["points"],
                    prob_femur,
                    pred_label == 1,
                )
        split_summaries[split] = {
            "num_files": len(paths),
            **totals.metrics(),
        }

    write_csv(output_dir / "h5_metrics.csv", h5_rows)
    write_csv(output_dir / "window_metrics.csv", window_rows)
    legend = {
        str(index): {"name": name, "rgb": DIAGNOSTIC_CATEGORY_COLORS[index].tolist()}
        for index, name in enumerate(DIAGNOSTIC_CATEGORY_NAMES)
    }
    with (output_dir / "diagnostic_ply_legend.json").open("w", encoding="utf-8") as handle:
        json.dump(legend, handle, indent=2, ensure_ascii=False)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "output_dir": str(output_dir.resolve()),
        "features": list(features),
        "normalize_points": normalize_points,
        "ignore_index": ignore_index,
        "aggregation": "mean_probability_by_original_point_index",
        "window_size_frames": window_size_frames,
        "window_stride_frames": window_stride_frames,
        "include_tail_window": bool(include_tail_window),
        "selection_seed": args.selection_seed,
        "fixed_train_videos": fixed_videos,
        "selected_train_files": [str(path) for path in selected_train],
        "validation_files": [str(path) for path in val_paths],
        "splits": split_summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Stage5 evaluation passed.")
    print(f"checkpoint: {args.checkpoint}")
    print(f"train sanity files: {len(selected_train)}")
    print(f"validation files: {len(val_paths)}")
    print(f"window metrics: {len(window_rows)}")
    print(f"output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage5 checkpoints on fixed train sanity and full validation splits"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument(
        "--selected_train_list",
        default=None,
        help="Optional preselected train-sanity paths; every path must be in --train_list",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--fixed_train_video",
        action="append",
        default=None,
        help="Train video substring to always include; repeat for multiple videos",
    )
    parser.add_argument("--num_random_train", type=int, default=2)
    parser.add_argument("--selection_seed", type=int, default=42)
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
    parser.add_argument("--no_write_ply", action="store_true")
    parser.add_argument("--no_save_predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
