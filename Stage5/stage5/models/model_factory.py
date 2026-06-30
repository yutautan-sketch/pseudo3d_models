from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .base_segmentor import BasePointSegmentor
from .mlp_baseline_segmentor import MLPBaselineSegmentor, mlp_baseline


ModelBuilder = Callable[..., BasePointSegmentor]


_MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "mlp_baseline": mlp_baseline,
}


def list_stage5_models() -> list[str]:
    return sorted(_MODEL_BUILDERS.keys())


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a recognizable model state dict")


def load_stage5_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=strict)
    return list(load_result.missing_keys), list(load_result.unexpected_keys)


def build_stage5_model(
    name: str = "mlp_baseline",
    *,
    num_classes: int = 2,
    feature_dim: int = 2,
    checkpoint: str | Path | None = None,
    strict_checkpoint: bool = False,
    **kwargs: Any,
) -> BasePointSegmentor:
    key = name.lower()
    if key not in _MODEL_BUILDERS:
        available = ", ".join(list_stage5_models())
        raise ValueError(f"Unknown Stage5 model '{name}'. Available models: {available}")

    model = _MODEL_BUILDERS[key](
        num_classes=num_classes,
        feature_dim=feature_dim,
        **kwargs,
    )

    if checkpoint is not None:
        load_stage5_checkpoint(
            model,
            checkpoint,
            strict=strict_checkpoint,
        )

    return model


__all__ = [
    "MLPBaselineSegmentor",
    "build_stage5_model",
    "list_stage5_models",
    "load_stage5_checkpoint",
]
