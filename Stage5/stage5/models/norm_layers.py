from __future__ import annotations

from typing import Any

from torch import nn

SUPPORTED_POINTNEXT_NORMS: tuple[str, ...] = ("batchnorm", "groupnorm")
BN_BUFFER_SUFFIXES: tuple[str, ...] = (".running_mean", ".running_var", ".num_batches_tracked")


class Stage5GroupNorm(nn.GroupNorm):
    """GroupNorm adapter for OpenPoints PointNeXt-S (S5-11).

    Subclassing nn.GroupNorm (rather than wrapping it) keeps `.weight`/`.bias`
    at the same dotted module path that BatchNorm occupied, so a BatchNorm
    checkpoint's norm affine parameters can be transferred by plain key match
    (see checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py).
    forward() is inherited unchanged and already supports both [B, C, N] and
    [B, C, *, *] inputs since nn.GroupNorm normalizes over any trailing dims.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 8,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Stage5GroupNorm received unsupported argument(s): {sorted(kwargs)}")
        if num_channels % num_groups != 0:
            raise ValueError(
                f"Stage5GroupNorm: num_channels={num_channels} is not divisible by "
                f"num_groups={num_groups}. Group count is never silently adjusted."
            )
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


def resolve_pointnext_norm_args(norm: str, norm_groups: int) -> dict[str, Any]:
    """Build the OpenPoints `norm_args` dict for a Stage5 PointNeXt-S normalization choice."""
    key = str(norm).lower()
    if key == "batchnorm":
        return {"norm": "bn"}
    if key == "groupnorm":
        if int(norm_groups) <= 0:
            raise ValueError(f"pointnext_norm_groups must be positive, got {norm_groups}")
        return {"norm": Stage5GroupNorm, "num_groups": int(norm_groups)}
    raise ValueError(
        f"Unsupported pointnext_norm={norm!r}. Supported values: {SUPPORTED_POINTNEXT_NORMS}"
    )
