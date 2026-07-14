from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from stage5.models import build_stage5_model
from stage5.models.model_factory import _extract_state_dict
from stage5.utils.feature_normalization import (
    normalize_frame_order,
    normalize_pixel_xy,
    normalize_uint8_feature,
    normalize_xyz,
)
from stage5.utils.h5_io import copy_h5_group, load_pointcloud_for_inference
from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window
from stage5.utils.visualization_export import write_probability_ply, write_selected_probability_ply


DEFAULT_FEATURES = ("intensity", "confidence")


def parse_feature_list(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_FEATURES
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_checkpoint(path: str | Path, *, map_location: str | torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict-like object: {path}")
    return checkpoint


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    return config if isinstance(config, dict) else {}


def build_features(data: dict[str, Any], feature_names: tuple[str, ...]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for name in feature_names:
        if name == "intensity":
            arrays.append(normalize_uint8_feature(data["intensity"])[:, None])
        elif name == "confidence":
            arrays.append(data["confidence"].astype(np.float32)[:, None])
        elif name == "alpha":
            arrays.append(normalize_uint8_feature(data["alpha"])[:, None])
        elif name == "frame_order":
            arrays.append(data["frame_order"].astype(np.float32)[:, None])
        elif name == "normalized_frame_order":
            arrays.append(normalize_frame_order(data["frame_order"])[:, None])
        elif name == "pixel_xy":
            arrays.append(normalize_pixel_xy(data["pixel_xy"]))
        else:
            raise ValueError(f"Unsupported Stage5 feature: {name}")

    if not arrays:
        return np.zeros((data["points"].shape[0], 0), dtype=np.float32)
    return np.concatenate(arrays, axis=1).astype(np.float32)


def resolve_model_kwargs(args: argparse.Namespace, config: dict[str, Any], feature_dim: int) -> dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        arg_value = getattr(args, name)
        if arg_value is not None:
            return arg_value
        return config.get(name, default)

    use_global_context = args.use_global_context
    if use_global_context is None:
        if "use_global_context" in config:
            use_global_context = bool(config["use_global_context"])
        else:
            use_global_context = not bool(config.get("no_global_context", False))

    model_name = value("model", "mlp_baseline")
    kwargs = {
        "name": model_name,
        "num_classes": int(value("num_classes", 2)),
        "feature_dim": int(feature_dim),
        "width": int(value("width", 64)),
        "depth": int(value("depth", 6)),
        "expansion": int(value("expansion", 4)),
        "dropout": float(value("dropout", 0.1)),
        "use_global_context": bool(use_global_context),
    }
    if str(model_name).lower() == "pointnext_s":
        kwargs.update(
            {
                "pointnext_radius": float(value("pointnext_radius", config.get("radius", 0.1))),
                "pointnext_nsample": int(value("pointnext_nsample", config.get("nsample", 16))),
                "pointnext_sa_layers": int(
                    value("pointnext_sa_layers", config.get("sa_layers", 2))
                ),
                "pointnext_sa_use_res": bool(
                    value("pointnext_sa_use_res", config.get("sa_use_res", True))
                ),
            }
        )
    return kwargs


def resolve_inference_mode(args: argparse.Namespace, config: dict[str, Any], model_name: str) -> str:
    if args.inference_mode != "auto":
        return args.inference_mode
    if str(model_name).lower() == "pointnext_s":
        return "window"
    if config.get("window_mode") == "overlap":
        return "window"
    return "chunk"


def resolve_window_settings(args: argparse.Namespace, config: dict[str, Any]) -> tuple[int, int, bool]:
    window_size_frames = (
        args.window_size_frames
        if args.window_size_frames is not None
        else int(config.get("window_size_frames", 12))
    )
    window_stride_frames = (
        args.window_stride_frames
        if args.window_stride_frames is not None
        else int(config.get("window_stride_frames", 6))
    )
    include_tail_window = args.include_tail_window
    if include_tail_window is None:
        include_tail_window = bool(config.get("include_tail_window", True))

    if window_size_frames <= 0:
        raise ValueError("--window_size_frames must be positive")
    if window_stride_frames <= 0:
        raise ValueError("--window_stride_frames must be positive")
    return int(window_size_frames), int(window_stride_frames), bool(include_tail_window)


def predict_all_points(
    *,
    model: torch.nn.Module,
    points: np.ndarray,
    features: np.ndarray,
    chunk_size: int,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")

    num_points = points.shape[0]
    logits_out = np.empty((num_points, num_classes), dtype=np.float32)
    prob_femur = np.empty((num_points,), dtype=np.float32)
    pred_label = np.empty((num_points,), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, num_points, chunk_size), desc="infer", leave=False):
            end = min(start + chunk_size, num_points)
            batch = {
                "points": torch.from_numpy(points[start:end][None]).float().to(device),
                "features": torch.from_numpy(features[start:end][None]).float().to(device),
            }
            output = model(batch)
            logits = output["logits"][0]
            prob = torch.softmax(logits, dim=-1)

            logits_np = logits.detach().cpu().numpy().astype(np.float32)
            prob_np = prob[:, 1].detach().cpu().numpy().astype(np.float32)
            pred_np = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.uint8)

            logits_out[start:end] = logits_np
            prob_femur[start:end] = prob_np
            pred_label[start:end] = pred_np

    return logits_out, prob_femur, pred_label


def predict_frame_windows(
    *,
    model: torch.nn.Module,
    points: np.ndarray,
    features: np.ndarray,
    frame_order: np.ndarray,
    window_size_frames: int,
    window_stride_frames: int,
    include_tail_window: bool,
    device: torch.device,
    num_classes: int,
    save_logits: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = generate_frame_order_windows(
        frame_order,
        window_size_frames=window_size_frames,
        window_stride_frames=window_stride_frames,
        include_tail_window=include_tail_window,
    )
    if not windows:
        raise ValueError("No frame_order windows were generated for inference")

    num_points = points.shape[0]
    prob_sum = np.zeros((num_points, num_classes), dtype=np.float64)
    logits_sum = np.zeros((num_points, num_classes), dtype=np.float64) if save_logits else None
    vote_count = np.zeros((num_points,), dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for window in tqdm(windows, desc="infer windows", leave=False):
            indices = point_indices_for_window(frame_order, window)
            if indices.size == 0:
                continue
            batch = {
                "points": torch.from_numpy(points[indices][None]).float().to(device),
                "features": torch.from_numpy(features[indices][None]).float().to(device),
            }
            output = model(batch)
            logits = output["logits"][0]
            prob = torch.softmax(logits, dim=-1)

            prob_sum[indices] += prob.detach().cpu().numpy().astype(np.float64)
            if logits_sum is not None:
                logits_sum[indices] += logits.detach().cpu().numpy().astype(np.float64)
            vote_count[indices] += 1

    missing = vote_count == 0
    if np.any(missing):
        missing_count = int(np.sum(missing))
        raise RuntimeError(f"{missing_count} point(s) received no window prediction")

    denominator = vote_count[:, None].astype(np.float64)
    prob_class = (prob_sum / denominator).astype(np.float32)
    logits_out = None
    if logits_sum is not None:
        logits_out = (logits_sum / denominator).astype(np.float32)
    prob_femur = prob_class[:, 1].astype(np.float32)
    pred_label = np.argmax(prob_class, axis=1).astype(np.uint8)

    window_summary = np.array(
        [
            (
                int(window.window_id),
                int(window.start_frame),
                int(window.end_frame),
                int(point_indices_for_window(frame_order, window).size),
            )
            for window in windows
        ],
        dtype=[
            ("window_id", "i4"),
            ("start_frame", "i4"),
            ("end_frame", "i4"),
            ("num_points", "i8"),
        ],
    )
    return logits_out, prob_class, prob_femur, pred_label, vote_count, window_summary


def write_output_h5(
    *,
    input_h5: Path,
    output_h5: Path,
    logits: np.ndarray | None,
    prob_class: np.ndarray,
    prob_femur: np.ndarray,
    pred_label: np.ndarray,
    vote_count: np.ndarray | None,
    window_summary: np.ndarray | None,
    config: dict[str, Any],
    copy_input: bool,
    save_logits: bool,
) -> None:
    if input_h5.resolve() == output_h5.resolve():
        raise ValueError("--output_h5 must be different from --input_h5")
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        if copy_input:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            if "point_cloud" in src:
                copy_h5_group(src, dst, "point_cloud")
            if "annotation" in src:
                copy_h5_group(src, dst, "annotation")
            if "frame_annotation" in src:
                copy_h5_group(src, dst, "frame_annotation")
            if "measurement" in src:
                copy_h5_group(src, dst, "measurement")

        pred_group = dst.create_group("prediction")
        if save_logits and logits is not None:
            pred_group.create_dataset("logits", data=logits, compression="gzip")
        pred_group.create_dataset("prob_class", data=prob_class, compression="gzip")
        pred_group.create_dataset("prob_femur", data=prob_femur, compression="gzip")
        pred_group.create_dataset("pred_label", data=pred_label, compression="gzip")
        if vote_count is not None:
            pred_group.create_dataset("vote_count", data=vote_count, compression="gzip")
        if window_summary is not None:
            pred_group.create_dataset("window_summary", data=window_summary, compression="gzip")
        pred_group.attrs["num_classes"] = int(prob_class.shape[1])
        pred_group.attrs["positive_class_index"] = 1
        pred_group.attrs["config_json"] = json.dumps(config, ensure_ascii=False)


def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint_config(checkpoint)

    feature_names = parse_feature_list(args.features if args.features is not None else config.get("feature_names"))
    normalize_points = args.normalize_points
    if normalize_points is None:
        normalize_points = not bool(config.get("no_normalize_points", False))

    data = load_pointcloud_for_inference(args.input_h5)
    raw_points = data["points"].astype(np.float32)
    points = normalize_xyz(raw_points) if normalize_points else raw_points
    features = build_features(data, feature_names)
    feature_dim = int(features.shape[1])

    model_kwargs = resolve_model_kwargs(args, config, feature_dim)
    model_name = model_kwargs.pop("name")
    inference_mode = resolve_inference_mode(args, config, model_name)
    window_size_frames, window_stride_frames, include_tail_window = resolve_window_settings(
        args,
        config,
    )
    model = build_stage5_model(
        model_name,
        checkpoint=None,
        **model_kwargs,
    ).to(device)
    state_dict = _extract_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=args.strict_checkpoint)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"checkpoint missing keys: {load_result.missing_keys}")
        print(f"checkpoint unexpected keys: {load_result.unexpected_keys}")

    inference_config = {
        "input_h5": str(args.input_h5),
        "checkpoint": str(args.checkpoint),
        "features": feature_names,
        "feature_dim": feature_dim,
        "normalize_points": bool(normalize_points),
        "inference_mode": inference_mode,
        "chunk_size": int(args.chunk_size),
        "save_logits": not args.no_save_logits,
        "window_size_frames": int(window_size_frames),
        "window_stride_frames": int(window_stride_frames),
        "include_tail_window": bool(include_tail_window),
        "aggregation": "mean_probability",
        "model": {"name": model_name, **model_kwargs},
    }

    if inference_mode == "chunk":
        logits, prob_femur, pred_label = predict_all_points(
            model=model,
            points=points,
            features=features,
            chunk_size=args.chunk_size,
            device=device,
            num_classes=int(model.num_classes),
        )
        prob_class = torch.softmax(torch.from_numpy(logits), dim=-1).numpy().astype(np.float32)
        vote_count = None
        window_summary = None
    elif inference_mode == "window":
        logits, prob_class, prob_femur, pred_label, vote_count, window_summary = predict_frame_windows(
            model=model,
            points=points,
            features=features,
            frame_order=data["frame_order"],
            window_size_frames=window_size_frames,
            window_stride_frames=window_stride_frames,
            include_tail_window=include_tail_window,
            device=device,
            num_classes=int(model.num_classes),
            save_logits=not args.no_save_logits,
        )
    else:
        raise ValueError(f"Unsupported inference_mode: {inference_mode}")

    write_output_h5(
        input_h5=Path(args.input_h5),
        output_h5=Path(args.output_h5),
        logits=logits,
        prob_class=prob_class,
        prob_femur=prob_femur,
        pred_label=pred_label,
        vote_count=vote_count,
        window_summary=window_summary,
        config=inference_config,
        copy_input=not args.no_copy_input,
        save_logits=not args.no_save_logits,
    )

    if args.output_ply is not None:
        write_probability_ply(args.output_ply, raw_points, prob_femur)

    if args.output_positive_ply is not None:
        selected_count = write_selected_probability_ply(
            args.output_positive_ply,
            raw_points,
            prob_femur,
            pred_label == 1,
        )
    else:
        selected_count = None

    print(f"wrote predictions: {args.output_h5}")
    if args.output_ply is not None:
        print(f"wrote PLY: {args.output_ply}")
    if args.output_positive_ply is not None:
        print(f"wrote positive-only PLY: {args.output_positive_ply} ({selected_count} points)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 pseudo-3D femur point-cloud inference")
    parser.add_argument("--input_h5", required=True, help="Annotated or unannotated point-cloud H5")
    parser.add_argument("--checkpoint", required=True, help="Stage5 checkpoint .pt")
    parser.add_argument("--output_h5", required=True, help="Output H5 with prediction group")
    parser.add_argument("--output_ply", default=None, help="Optional probability-colored PLY output")
    parser.add_argument(
        "--output_positive_ply",
        default=None,
        help="Optional PLY containing only points with predicted positive/femur label",
    )

    parser.add_argument("--features", default=None, help="Override comma-separated features from checkpoint config")
    parser.add_argument(
        "--inference_mode",
        choices=["auto", "window", "chunk"],
        default="auto",
        help="auto uses window inference for pointnext_s and chunk inference otherwise",
    )
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--no_copy_input", action="store_true", help="Only write prediction group")
    parser.add_argument("--no_save_logits", action="store_true", help="Do not save prediction/logits")

    parser.add_argument("--normalize_points", dest="normalize_points", action="store_true", default=None)
    parser.add_argument("--no_normalize_points", dest="normalize_points", action="store_false")

    parser.add_argument("--model", default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--expansion", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_global_context", action="store_true", default=None)
    parser.add_argument("--no_global_context", dest="use_global_context", action="store_false")
    parser.add_argument("--pointnext_radius", type=float, default=None)
    parser.add_argument("--pointnext_nsample", type=int, default=None)
    parser.add_argument("--pointnext_sa_layers", type=int, default=None)
    parser.add_argument("--pointnext_sa_use_res", dest="pointnext_sa_use_res", action="store_true", default=None)
    parser.add_argument("--no_pointnext_sa_use_res", dest="pointnext_sa_use_res", action="store_false")
    parser.add_argument("--window_size_frames", type=int, default=None)
    parser.add_argument("--window_stride_frames", type=int, default=None)
    parser.add_argument("--include_tail_window", dest="include_tail_window", action="store_true", default=None)
    parser.add_argument("--no_include_tail_window", dest="include_tail_window", action="store_false")

    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
