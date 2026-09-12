from __future__ import annotations

from typing import Any


def ensure_stage5_pointnext_decoder_registered() -> str:
    """Register Stage5PointNextDecoder with OpenPoints' MODELS registry (idempotent).

    The official OpenPoints `PointNextDecoder.__init__` accepts `**kwargs` but never
    reads `norm_args`/`act_args`, and its `_make_dec()` always builds
    `FeaturePropogation` without passing them through -- so `FeaturePropogation`'s own
    hardcoded `{'norm': 'bn1d'}` default is used unconditionally, regardless of what
    Stage5's `decoder_args` configures (discovered via a failing S5-11 GroupNorm
    structure check: 8 decoder BatchNorm modules remained after requesting
    `pointnext_norm=groupnorm`). This subclass only overrides `_make_dec()` to
    forward the configured `norm_args`/`act_args` into `FeaturePropogation`, without
    editing `external/PointNeXt`. It is registered under a new name, so
    `decoder_args={"NAME": "Stage5PointNextDecoder", ...}` resolves to it while the
    original `"PointNextDecoder"` registration is left untouched.

    Passing `norm_args={"norm": "bn"}` here reproduces `FeaturePropogation`'s
    original hardcoded default exactly (`create_norm` appends the '1d' dimension
    suffix internally), so switching `decoder_args["NAME"]` to this class does not
    change the batchnorm-default module graph or its state_dict keys.

    Returns the registered module name.
    """
    from openpoints.models.backbone.pointnext import FeaturePropogation, PointNextDecoder
    from openpoints.models.build import MODELS
    from torch import nn

    name = "Stage5PointNextDecoder"
    if MODELS.get(name) is not None:
        return name

    class Stage5PointNextDecoder(PointNextDecoder):
        def __init__(
            self,
            encoder_channel_list: list[int],
            decoder_layers: int = 2,
            decoder_stages: int = 4,
            **kwargs: Any,
        ) -> None:
            # Stored before super().__init__() (nn.Module.__init__ included), which
            # internally calls the overridden _make_dec() below via polymorphic
            # dispatch. Writing directly to __dict__ bypasses nn.Module.__setattr__,
            # so this is safe even though nn.Module.__init__() has not run yet.
            self.__dict__["norm_args"] = dict(kwargs.get("norm_args", {"norm": "bn1d"}))
            self.__dict__["act_args"] = dict(kwargs.get("act_args", {"act": "relu"}))
            super().__init__(
                encoder_channel_list,
                decoder_layers=decoder_layers,
                decoder_stages=decoder_stages,
                **kwargs,
            )

        def _make_dec(self, skip_channels: int, fp_channels: int) -> nn.Sequential:
            mlp = [skip_channels + self.in_channels] + [fp_channels] * self.decoder_layers
            layers = [FeaturePropogation(mlp, norm_args=self.norm_args, act_args=self.act_args)]
            self.in_channels = fp_channels
            return nn.Sequential(*layers)

    MODELS.register_module(name=name, module=Stage5PointNextDecoder)
    return name
