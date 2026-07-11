from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from stage5.models.pointnext_s_segmentor import (
    _ensure_openpoints_importable,
    build_pointnext_s_config,
)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and all(torch.is_tensor(v) for v in value.values()):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a recognizable tensor state_dict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-load the official S3DIS PointNeXt-S checkpoint")
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_forward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    _ensure_openpoints_importable()
    from openpoints.models import build_model_from_cfg

    cfg = build_pointnext_s_config(
        num_classes=13,
        feature_dim=4,
        width=32,
        expansion=4,
        dropout=0.5,
        radius=0.1,
        nsample=32,
        sa_layers=2,
        sa_use_res=True,
    )
    model = build_model_from_cfg(cfg).to(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    result = model.load_state_dict(state_dict, strict=True)

    print("S3DIS PointNeXt-S strict load passed.")
    print(f"checkpoint: {checkpoint_path}")
    print(f"state_dict keys: {len(state_dict)}")
    print(f"missing keys: {len(result.missing_keys)}")
    print(f"unexpected keys: {len(result.unexpected_keys)}")
    if isinstance(checkpoint, dict):
        print(f"epoch: {checkpoint.get('epoch')}")
        print(f"best_val: {checkpoint.get('best_val')}")

    if args.skip_forward:
        return

    model.eval()
    data = {
        "pos": torch.randn(args.batch_size, args.num_points, 3, device=device).contiguous(),
        "x": torch.randn(args.batch_size, 4, args.num_points, device=device).contiguous(),
    }
    with torch.no_grad():
        logits = model(data)

    expected_shape = (args.batch_size, 13, args.num_points)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)} != {expected_shape}")
    if not torch.isfinite(logits).all():
        raise AssertionError("logits contains NaN/Inf")

    print("S3DIS PointNeXt-S forward after strict load passed.")
    print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
