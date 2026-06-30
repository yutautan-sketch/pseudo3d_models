from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models import get_model
from src.utils.pseudo3d_processing import resolve_dualtrack_checkpoint


def describe_tensor(name: str, x: torch.Tensor):
    print(
        f"{name}: "
        f"shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"device={x.device}, "
        f"min={x.min().item():.4f}, "
        f"max={x.max().item():.4f}, "
        f"mean={x.mean().item():.4f}"
    )


def main():
    repo_root = Path(__file__).resolve().parent

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)

    # -----------------------------
    # 1. Load model
    # -----------------------------
    model = get_model(**cfg)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel loaded successfully.")
    print("Model class:", model.__class__.__name__)
    print(f"num_params    : {num_params:,}")
    print(f"num_trainable : {num_trainable:,}")

    # -----------------------------
    # 2. Create dummy inputs
    # -----------------------------
    # Based on the official fusion_model_training.py preprocessing:
    #   global_encoder_images: [B, N, 1, 224, 224]
    #   local_encoder_images : [B, N, 1, 256, 256]
    B = 1
    N = 32
    C = 1

    global_H, global_W = 224, 224
    local_H, local_W = 256, 256

    # Dummy images are sampled from [0, 1), matching FramesArrayToTensor() / 255.0 scale.
    global_x = torch.rand(B, N, C, global_H, global_W, device=device)
    local_x = torch.rand(B, N, C, local_H, local_W, device=device)

    print("\nDummy inputs:")
    describe_tensor("global_x", global_x)
    describe_tensor("local_x ", local_x)

    # -----------------------------
    # 3. Direct dummy forward
    # -----------------------------
    print("\nRunning direct forward: pred = model(global_x, local_x)")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        pred = model(global_x, local_x)

    print("\nForward completed.")
    describe_tensor("pred", pred)

    expected_shape = (B, N - 1, 6)
    print("\nExpected output shape:", expected_shape)
    print("Actual output shape  :", tuple(pred.shape))

    if tuple(pred.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected prediction shape: got {tuple(pred.shape)}, "
            f"expected {expected_shape}"
        )

    print("\nShape check passed.")

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"CUDA peak memory allocated: {peak_allocated:.3f} GiB")
        print(f"CUDA peak memory reserved : {peak_reserved:.3f} GiB")

    # -----------------------------
    # 4. Basic sanity check
    # -----------------------------
    if not torch.isfinite(pred).all():
        num_nan = torch.isnan(pred).sum().item()
        num_inf = torch.isinf(pred).sum().item()
        raise RuntimeError(
            f"Prediction contains non-finite values: "
            f"NaN={num_nan}, Inf={num_inf}"
        )

    print("Finite-value check passed.")
    print("\nDummy forward test finished successfully.")


if __name__ == "__main__":
    main()
