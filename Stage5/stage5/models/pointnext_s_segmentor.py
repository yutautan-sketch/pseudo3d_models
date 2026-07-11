from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .base_segmentor import BasePointSegmentor


class AttrDict(dict):
    """Small attribute-access dict compatible with OpenPoints configs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _attrdict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attrdict(item) for item in value]
    return value


def _ensure_openpoints_importable() -> None:
    stage5_root = Path(__file__).resolve().parents[2]
    pointnext_root = stage5_root / "external" / "PointNeXt"
    if not pointnext_root.exists():
        raise ImportError(
            f"PointNeXt repository not found: {pointnext_root}. "
            "Clone the official repository under Stage5/external/PointNeXt first."
        )
    pointnext_path = str(pointnext_root)
    if pointnext_path not in sys.path:
        sys.path.insert(0, pointnext_path)


def build_pointnext_s_config(
    *,
    num_classes: int,
    feature_dim: int,
    width: int = 32,
    expansion: int = 4,
    dropout: float = 0.1,
    radius: float = 0.1,
    nsample: int = 16,
    sa_layers: int = 2,
    sa_use_res: bool = True,
) -> AttrDict:
    return _attrdict(
        {
            "NAME": "BaseSeg",
            "encoder_args": {
                "NAME": "PointNextEncoder",
                "blocks": [1, 1, 1, 1, 1],
                "strides": [1, 4, 4, 4, 4],
                "sa_layers": int(sa_layers),
                "sa_use_res": bool(sa_use_res),
                "width": int(width),
                "in_channels": int(feature_dim),
                "expansion": int(expansion),
                "radius": float(radius),
                "nsample": int(nsample),
                "aggr_args": {
                    "feature_type": "dp_fj",
                    "reduction": "max",
                },
                "group_args": {
                    "NAME": "ballquery",
                    "normalize_dp": True,
                },
                "conv_args": {
                    "order": "conv-norm-act",
                },
                "act_args": {
                    "act": "relu",
                },
                "norm_args": {
                    "norm": "bn",
                },
            },
            "decoder_args": {
                "NAME": "PointNextDecoder",
            },
            "cls_args": {
                "NAME": "SegHead",
                "num_classes": int(num_classes),
                "in_channels": None,
                "dropout": float(dropout),
                "norm_args": {
                    "norm": "bn",
                },
            },
        }
    )


class PointNeXtSSegmentor(BasePointSegmentor):
    """Stage 5 wrapper around the official OpenPoints PointNeXt-S BaseSeg."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        feature_dim: int = 2,
        width: int = 32,
        expansion: int = 4,
        dropout: float = 0.1,
        radius: float = 0.1,
        nsample: int = 16,
        sa_layers: int = 2,
        sa_use_res: bool = True,
    ) -> None:
        super().__init__(num_classes=num_classes, feature_dim=feature_dim)
        self.width = int(width)
        self.expansion = int(expansion)
        self.dropout = float(dropout)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.sa_layers = int(sa_layers)
        self.sa_use_res = bool(sa_use_res)

        _ensure_openpoints_importable()
        try:
            from openpoints.models import build_model_from_cfg
        except ImportError as exc:
            raise ImportError(
                "Failed to import OpenPoints. Ensure PointNeXt/openpoints dependencies "
                "are installed in the active environment."
            ) from exc

        cfg = build_pointnext_s_config(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            width=self.width,
            expansion=self.expansion,
            dropout=self.dropout,
            radius=self.radius,
            nsample=self.nsample,
            sa_layers=self.sa_layers,
            sa_use_res=self.sa_use_res,
        )
        self.pointnext = build_model_from_cfg(cfg)

    def _prepare_openpoints_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "points" not in batch:
            raise KeyError("PointNeXtSSegmentor expects batch['points']")
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

        return {
            "pos": points.contiguous(),
            "x": features.transpose(1, 2).contiguous(),
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        openpoints_batch = self._prepare_openpoints_batch(batch)
        logits = self.pointnext(openpoints_batch)
        if logits.ndim != 3:
            raise ValueError(f"PointNeXt-S logits must have shape [B, C, N], got {tuple(logits.shape)}")
        return {"logits": logits.transpose(1, 2).contiguous()}

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "expansion": self.expansion,
                "dropout": self.dropout,
                "radius": self.radius,
                "nsample": self.nsample,
                "sa_layers": self.sa_layers,
                "sa_use_res": self.sa_use_res,
            }
        )
        return config


def pointnext_s(
    *,
    num_classes: int = 2,
    feature_dim: int = 2,
    dropout: float = 0.1,
    **kwargs: Any,
) -> PointNeXtSSegmentor:
    width = int(kwargs.pop("width", 32))
    expansion = int(kwargs.pop("expansion", 4))
    radius = float(kwargs.pop("radius", kwargs.pop("pointnext_radius", 0.1)))
    nsample = int(kwargs.pop("nsample", kwargs.pop("pointnext_nsample", 16)))
    sa_layers = int(kwargs.pop("sa_layers", kwargs.pop("pointnext_sa_layers", 2)))
    sa_use_res = bool(kwargs.pop("sa_use_res", kwargs.pop("pointnext_sa_use_res", True)))

    # Accepted for compatibility with train_stage5.py's shared model kwargs.
    kwargs.pop("depth", None)
    kwargs.pop("use_global_context", None)

    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unknown pointnext_s argument(s): {unknown}")

    return PointNeXtSSegmentor(
        num_classes=num_classes,
        feature_dim=feature_dim,
        width=width,
        expansion=expansion,
        dropout=dropout,
        radius=radius,
        nsample=nsample,
        sa_layers=sa_layers,
        sa_use_res=sa_use_res,
    )
