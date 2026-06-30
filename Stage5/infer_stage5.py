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
from stage5.utils.visualization_export import write_probability_ply


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

    return {
        "name": value("model", "mlp_baseline"),
        "num_classes": int(value("num_classes", 2)),
        "feature_dim": int(feature_dim),
        "width": int(value("width", 64)),
        "depth": int(value("depth", 6)),
        "expansion": int(value("expansion", 4)),
        "dropout": float(value("dropout", 0.1)),
        "use_global_context": bool(use_global_context),
    }


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


def write_output_h5(
    *,
    input_h5: Path,
    output_h5: Path,
    logits: np.ndarray,
    prob_femur: np.ndarray,
    pred_label: np.ndarray,
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
        if save_logits:
            pred_group.create_dataset("logits", data=logits, compression="gzip")
        pred_group.create_dataset("prob_femur", data=prob_femur, compression="gzip")
        pred_group.create_dataset("pred_label", data=pred_label, compression="gzip")
        pred_group.attrs["num_classes"] = int(logits.shape[1])
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
        "chunk_size": int(args.chunk_size),
        "save_logits": not args.no_save_logits,
        "model": {"name": model_name, **model_kwargs},
    }

    logits, prob_femur, pred_label = predict_all_points(
        model=model,
        points=points,
        features=features,
        chunk_size=args.chunk_size,
        device=device,
        num_classes=int(model.num_classes),
    )

    write_output_h5(
        input_h5=Path(args.input_h5),
        output_h5=Path(args.output_h5),
        logits=logits,
        prob_femur=prob_femur,
        pred_label=pred_label,
        config=inference_config,
        copy_input=not args.no_copy_input,
        save_logits=not args.no_save_logits,
    )

    if args.output_ply is not None:
        write_probability_ply(args.output_ply, raw_points, prob_femur)

    print(f"wrote predictions: {args.output_h5}")
    if args.output_ply is not None:
        print(f"wrote PLY: {args.output_ply}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 pseudo-3D femur point-cloud inference")
    parser.add_argument("--input_h5", required=True, help="Annotated or unannotated point-cloud H5")
    parser.add_argument("--checkpoint", required=True, help="Stage5 checkpoint .pt")
    parser.add_argument("--output_h5", required=True, help="Output H5 with prediction group")
    parser.add_argument("--output_ply", default=None, help="Optional probability-colored PLY output")

    parser.add_argument("--features", default=None, help="Override comma-separated features from checkpoint config")
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

    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
