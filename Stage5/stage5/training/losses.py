from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn


LABEL_IGNORE = -1


def prepare_segmentation_targets(
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = LABEL_IGNORE,
) -> torch.Tensor:
    """Return labels with invalid points set to ignore_index."""
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, N], got {tuple(labels.shape)}")

    targets = labels.long()
    if valid_mask is not None:
        if valid_mask.shape != labels.shape:
            raise ValueError(
                f"valid_mask must have shape {tuple(labels.shape)}, got {tuple(valid_mask.shape)}"
            )
        targets = targets.masked_fill(~valid_mask.bool(), int(ignore_index))
    return targets


def flatten_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, N, C], got {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape [B, N], got {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logits and targets must share [B, N], got {tuple(logits.shape[:2])} "
            f"and {tuple(targets.shape)}"
        )
    return logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)


class BaseSegmentationLoss(nn.Module, ABC):
    """Base class for Stage 5 segmentation losses."""

    def __init__(self, *, ignore_index: int = LABEL_IGNORE) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    @abstractmethod
    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute loss and return a dictionary containing at least ``loss``."""

    def _get_logits_and_targets(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "logits" not in output:
            raise KeyError("Model output must contain 'logits'")
        if "labels" not in batch:
            raise KeyError("Batch must contain 'labels'")
        logits = output["logits"]
        labels = batch["labels"]
        valid_mask = batch.get("valid_mask")
        targets = prepare_segmentation_targets(
            labels,
            valid_mask,
            ignore_index=self.ignore_index,
        )
        return logits, targets


class CrossEntropySegmentationLoss(BaseSegmentationLoss):
    """Cross entropy for point-wise binary/multi-class segmentation."""

    def __init__(
        self,
        *,
        ignore_index: int = LABEL_IGNORE,
        class_weight: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(ignore_index=ignore_index)
        if class_weight is None:
            self.register_buffer("class_weight", None)
        else:
            weight = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("class_weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        logits, targets = self._get_logits_and_targets(output, batch)
        flat_logits, flat_targets = flatten_logits_and_targets(logits, targets)

        valid_count = (flat_targets != self.ignore_index).sum()
        if valid_count.item() == 0:
            loss = flat_logits.sum() * 0.0
        else:
            weight = self.class_weight
            if weight is not None:
                weight = weight.to(device=flat_logits.device, dtype=flat_logits.dtype)
            loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                weight=weight,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )

        return {
            "loss": loss,
            "ce_loss": loss.detach(),
            "valid_point_count": valid_count.detach(),
        }


LossBuilder = Callable[..., BaseSegmentationLoss]


_LOSS_BUILDERS: dict[str, LossBuilder] = {
    "cross_entropy": CrossEntropySegmentationLoss,
    "ce": CrossEntropySegmentationLoss,
}


def list_losses() -> list[str]:
    return sorted(_LOSS_BUILDERS.keys())


def build_loss(name: str = "cross_entropy", **kwargs: Any) -> BaseSegmentationLoss:
    key = name.lower()
    if key not in _LOSS_BUILDERS:
        available = ", ".join(list_losses())
        raise ValueError(f"Unknown Stage5 loss '{name}'. Available losses: {available}")
    return _LOSS_BUILDERS[key](**kwargs)

