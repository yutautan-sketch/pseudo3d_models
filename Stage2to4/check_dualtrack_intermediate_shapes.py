from pathlib import Path
from typing import Any

import torch
from torch import nn
from omegaconf import OmegaConf

from src.models import get_model
from src.utils.pseudo3d_processing import resolve_dualtrack_checkpoint


def tensor_summary(x: torch.Tensor) -> str:
    base = (
        f"Tensor(shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"device={x.device}"
    )

    if x.numel() == 0:
        return base + ", empty=True)"

    if torch.is_floating_point(x) or torch.is_complex(x):
        return (
            base
            + f", min={x.min().item():.4f}, "
              f"max={x.max().item():.4f}, "
              f"mean={x.mean().item():.4f})"
        )

    # Integer / bool tensors, e.g. sparse_indices
    return (
        base
        + f", min={x.min().item()}, "
          f"max={x.max().item()})"
    )


def summarize(obj: Any, depth: int = 0, max_depth: int = 2) -> str:
    """
    Robust summary for tensors, tuples, lists, dicts, and HuggingFace-style outputs.
    """
    if isinstance(obj, torch.Tensor):
        return tensor_summary(obj)

    if depth >= max_depth:
        return f"{type(obj).__name__}(...)"

    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {summarize(v, depth + 1, max_depth)}")
        return "{" + ", ".join(parts) + "}"

    if isinstance(obj, (list, tuple)):
        parts = [summarize(v, depth + 1, max_depth) for v in obj]
        return f"{type(obj).__name__}([" + ", ".join(parts) + "])"

    # HuggingFace ModelOutput often has .last_hidden_state
    if hasattr(obj, "last_hidden_state"):
        return (
            f"{type(obj).__name__}("
            f"last_hidden_state={summarize(obj.last_hidden_state, depth + 1, max_depth)})"
        )

    return repr(obj)


def make_forward_hook(name: str):
    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
        print(f"\n[HOOK] {name}")
        print(f"  module: {module.__class__.__name__}")
        print(f"  input : {summarize(inputs)}")
        print(f"  output: {summarize(output)}")

    return hook


class DebugSamplerWrapper(nn.Module):
    """
    Wraps RegularGridSampler-like object.

    The original RegularGridSampler implements __call__ directly, so standard
    forward hooks are not reliable for it. This wrapper prints the input/output
    explicitly and then returns the original sampler result.
    """

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    @property
    def grid_spacing(self):
        return getattr(self.sampler, "grid_spacing", None)

    def __call__(self, images, features):
        print("\n[DEBUG SAMPLER] RegularGridSampler")
        print(f"  grid_spacing: {self.grid_spacing}")
        print(f"  images  : {summarize(images)}")
        print(f"  features: {summarize(features)}")

        sparse_indices = self.sampler(images, features)

        print(f"  output sparse_indices: {summarize(sparse_indices)}")
        if isinstance(sparse_indices, torch.Tensor):
            print(f"  sparse_indices values: {sparse_indices.detach().cpu().tolist()}")

        return sparse_indices


def load_dualtrack_model(repo_root: Path, device: torch.device):
    cfg_path = repo_root / "configs" / "model" / "dualtrack.yaml"
    ckpt_path = resolve_dualtrack_checkpoint(repo_root)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Config:", cfg_path)
    print("Checkpoint:", ckpt_path)

    cfg = OmegaConf.load(cfg_path)
    cfg.checkpoint = str(ckpt_path)

    print("\nLoaded config:")
    print(OmegaConf.to_yaml(cfg))

    model = get_model(**cfg)
    model = model.to(device)
    model.eval()

    print("\nModel loaded successfully.")
    print("Model class:", model.__class__.__name__)
    print(f"num_params    : {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"num_trainable : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    return model


def register_shape_hooks(model: nn.Module):
    handles = []

    # Top-level branches
    handles.append(model.local_encoder.register_forward_hook(make_forward_hook("model.local_encoder")))
    handles.append(model.global_encoder.register_forward_hook(make_forward_hook("model.global_encoder")))
    handles.append(model.fusion_module.register_forward_hook(make_forward_hook("model.fusion_module")))
    handles.append(model.head.register_forward_hook(make_forward_hook("model.head")))

    # local_encoder is expected to be:
    #   nn.Sequential(local_encoder_backbone, Linear(64 -> 512))
    if isinstance(model.local_encoder, nn.Sequential):
        if len(model.local_encoder) >= 1:
            handles.append(
                model.local_encoder[0].register_forward_hook(
                    make_forward_hook("model.local_encoder[0] local_encoder_backbone")
                )
            )
        if len(model.local_encoder) >= 2:
            handles.append(
                model.local_encoder[1].register_forward_hook(
                    make_forward_hook("model.local_encoder[1] Linear(64 -> 512)")
                )
            )

    # Global encoder internals:
    # SimpleModelForSparseTrackingEstimation has backbone, proj, bert, fc.
    if hasattr(model.global_encoder, "backbone"):
        handles.append(
            model.global_encoder.backbone.register_forward_hook(
                make_forward_hook("model.global_encoder.backbone")
            )
        )

    if hasattr(model.global_encoder, "proj"):
        handles.append(
            model.global_encoder.proj.register_forward_hook(
                make_forward_hook("model.global_encoder.proj")
            )
        )

    if hasattr(model.global_encoder, "bert"):
        handles.append(
            model.global_encoder.bert.register_forward_hook(
                make_forward_hook("model.global_encoder.bert")
            )
        )

    return handles


def main():
    repo_root = Path(__file__).resolve().parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)

    model = load_dualtrack_model(repo_root, device)

    # Wrap sampler to explicitly print sparse_indices.
    model.sampler = DebugSamplerWrapper(model.sampler)

    handles = register_shape_hooks(model)

    # Official fusion_model_training.py-style dummy shapes:
    #   global_encoder_images: [B, N, 1, 224, 224]
    #   local_encoder_images : [B, N, 1, 256, 256]
    B = 1
    N = 32
    C = 1

    global_x = torch.rand(B, N, C, 224, 224, device=device)
    local_x = torch.rand(B, N, C, 256, 256, device=device)

    print("\nDummy inputs:")
    print(f"  global_x: {summarize(global_x)}")
    print(f"  local_x : {summarize(local_x)}")

    print("\nRunning forward with intermediate hooks...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.inference_mode():
            pred = model(global_x, local_x)
    finally:
        for h in handles:
            h.remove()

    print("\nForward completed.")
    print(f"Final pred: {summarize(pred)}")

    expected_shape = (B, N - 1, 6)
    print("\nExpected final output shape:", expected_shape)
    print("Actual final output shape  :", tuple(pred.shape))

    if tuple(pred.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected final prediction shape: got {tuple(pred.shape)}, "
            f"expected {expected_shape}"
        )

    if not torch.isfinite(pred).all():
        num_nan = torch.isnan(pred).sum().item()
        num_inf = torch.isinf(pred).sum().item()
        raise RuntimeError(
            f"Prediction contains non-finite values: NaN={num_nan}, Inf={num_inf}"
        )

    print("\nFinal shape check passed.")
    print("Finite-value check passed.")

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"CUDA peak memory allocated: {peak_allocated:.3f} GiB")
        print(f"CUDA peak memory reserved : {peak_reserved:.3f} GiB")

    print("\nIntermediate shape test finished successfully.")


if __name__ == "__main__":
    main()
