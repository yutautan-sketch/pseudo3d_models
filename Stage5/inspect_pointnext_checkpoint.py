from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and all(torch.is_tensor(v) for v in value.values()):
                return value
        tensor_items = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
        if tensor_items and len(tensor_items) == len(checkpoint):
            return checkpoint
    raise ValueError("Could not find a tensor state_dict in this checkpoint")


def tensor_summary(tensor: torch.Tensor) -> str:
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch/OpenPoints checkpoint structure")
    parser.add_argument(
        "--checkpoint",
        default=(
            "Stage5/external/PointNeXt/weight/s3dis-seg/checkpoint/"
            "s3dis-train-pointnext-s-ngpus1-seed1742-20220525-162639-Gz4ViiL95b6KP9MMH2ytG3_ckpt_best.pth"
        ),
    )
    parser.add_argument("--max_keys", type=int, default=80)
    parser.add_argument("--prefix_depth", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint type: {type(checkpoint).__name__}")

    if isinstance(checkpoint, dict):
        print("top-level keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if torch.is_tensor(value):
                desc = tensor_summary(value)
            elif isinstance(value, dict):
                desc = f"dict(len={len(value)})"
            else:
                desc = type(value).__name__
            print(f"  {key}: {desc}")

    state_dict = extract_state_dict(checkpoint)
    print(f"state_dict keys: {len(state_dict)}")

    prefix_counts: Counter[str] = Counter()
    for key in state_dict.keys():
        parts = key.split(".")
        prefix = ".".join(parts[: args.prefix_depth])
        prefix_counts[prefix] += 1

    print(f"prefix counts depth={args.prefix_depth}:")
    for prefix, count in prefix_counts.most_common():
        print(f"  {prefix}: {count}")

    print(f"first {args.max_keys} state_dict entries:")
    for index, (key, value) in enumerate(state_dict.items()):
        if index >= args.max_keys:
            break
        print(f"  {key}: {tensor_summary(value)}")

    interesting_suffixes = (
        "encoder.0.0.convs.0.0.weight",
        "head.head.1.weight",
        "head.head.1.bias",
        "prediction.head.0.weight",
    )
    print("interesting matches:")
    for key, value in state_dict.items():
        if any(key.endswith(suffix) for suffix in interesting_suffixes):
            print(f"  {key}: {tensor_summary(value)}")


if __name__ == "__main__":
    main()
