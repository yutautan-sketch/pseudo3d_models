from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BasePointSegmentor(nn.Module, ABC):
    """Base interface for Stage 5 point-wise segmentation models."""

    def __init__(self, *, num_classes: int = 2, feature_dim: int = 2) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run point-wise segmentation.

        Expected input:
            points: Tensor[B, N, 3]
            features: Tensor[B, N, C]

        Expected output:
            logits: Tensor[B, N, num_classes]
        """

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
        }

