from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base_segmentor import BasePointSegmentor


class ConvBNAct1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: bool = True,
        activation: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm1d(out_channels))
        if activation:
            layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualPointMLPBlock(nn.Module):
    """Residual shared-MLP block for point-wise baseline segmentation."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.net = nn.Sequential(
            ConvBNAct1d(channels, hidden_channels),
            ConvBNAct1d(hidden_channels, hidden_channels, dropout=dropout),
            ConvBNAct1d(hidden_channels, channels, activation=False),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MLPBaselineSegmentor(BasePointSegmentor):
    """Lightweight pure-PyTorch shared-MLP segmentor for Stage 5.

    This baseline uses per-point 1x1 convolutions, residual MLP blocks, and an
    optional global max-pooled context feature. It provides a stable Stage 5
    wrapper interface for early pipeline validation before adding stronger
    point-cloud backends.
    """

    def __init__(
        self,
        *,
        num_classes: int = 2,
        feature_dim: int = 2,
        width: int = 64,
        depth: int = 6,
        expansion: int = 4,
        dropout: float = 0.1,
        use_global_context: bool = True,
    ) -> None:
        super().__init__(num_classes=num_classes, feature_dim=feature_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.expansion = int(expansion)
        self.dropout = float(dropout)
        self.use_global_context = bool(use_global_context)

        input_dim = 3 + self.feature_dim
        self.stem = nn.Sequential(
            ConvBNAct1d(input_dim, self.width),
            ConvBNAct1d(self.width, self.width),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualPointMLPBlock(
                    self.width,
                    expansion=self.expansion,
                    dropout=self.dropout,
                )
                for _ in range(self.depth)
            ]
        )

        head_input_dim = self.width * 2 if self.use_global_context else self.width
        self.head = nn.Sequential(
            ConvBNAct1d(head_input_dim, self.width),
            ConvBNAct1d(self.width, self.width // 2, dropout=self.dropout),
            nn.Conv1d(self.width // 2, self.num_classes, kernel_size=1),
        )

    def _prepare_inputs(self, batch: dict[str, Any]) -> torch.Tensor:
        if "points" not in batch:
            raise KeyError("MLPBaselineSegmentor expects batch['points']")
        points = batch["points"]
        features = batch.get("features")

        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"batch['points'] must have shape [B, N, 3], got {tuple(points.shape)}")

        if features is None:
            if self.feature_dim != 0:
                raise ValueError("batch['features'] is required when feature_dim > 0")
            features = points.new_zeros((*points.shape[:2], 0))
        elif features.ndim != 3:
            raise ValueError(
                f"batch['features'] must have shape [B, N, C], got {tuple(features.shape)}"
            )

        if features.shape[:2] != points.shape[:2]:
            raise ValueError(
                "batch['features'] and batch['points'] must share [B, N], "
                f"got {tuple(features.shape[:2])} and {tuple(points.shape[:2])}"
            )
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {features.shape[-1]}"
            )

        x = torch.cat([points, features], dim=-1)
        return x.transpose(1, 2).contiguous()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = self._prepare_inputs(batch)
        x = self.stem(x)
        x = self.blocks(x)

        if self.use_global_context:
            global_feature = torch.amax(x, dim=2, keepdim=True).expand_as(x)
            x = torch.cat([x, global_feature], dim=1)

        logits = self.head(x).transpose(1, 2).contiguous()
        return {"logits": logits}

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "depth": self.depth,
                "expansion": self.expansion,
                "dropout": self.dropout,
                "use_global_context": self.use_global_context,
            }
        )
        return config


def mlp_baseline(
    *,
    num_classes: int = 2,
    feature_dim: int = 2,
    dropout: float = 0.1,
    **kwargs: Any,
) -> MLPBaselineSegmentor:
    """Small Stage 5 shared-MLP segmentation baseline."""
    width = int(kwargs.pop("width", 64))
    depth = int(kwargs.pop("depth", 6))
    expansion = int(kwargs.pop("expansion", 4))
    use_global_context = bool(kwargs.pop("use_global_context", True))
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unknown mlp_baseline argument(s): {unknown}")

    return MLPBaselineSegmentor(
        num_classes=num_classes,
        feature_dim=feature_dim,
        width=width,
        depth=depth,
        expansion=expansion,
        dropout=dropout,
        use_global_context=use_global_context,
    )
