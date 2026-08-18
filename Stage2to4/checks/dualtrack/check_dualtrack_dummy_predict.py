from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def main():
    repo_root = REPO_ROOT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)

    # -----------------------------
    # 1. Load model
    # -----------------------------
    model = load_dualtrack_model(repo_root, device)

    # -----------------------------
    # 2. Create dummy batch
    # -----------------------------
    # Official fusion_model_training.py-style shapes:
    #   global_encoder_images: [B, N, 1, 224, 224]
    #   local_encoder_images : [B, N, 1, 256, 256]
    B = 1
    N = 32
    C = 1

    global_x = torch.rand(B, N, C, 224, 224, device=device)
    local_x = torch.rand(B, N, C, 256, 256, device=device)

    batch = {
        # forward_dict() uses sweep_id only to store current data IDs.
        # A list of length B matches DataLoader-like behavior.
        "sweep_id": ["dummy_sweep_000"],

        # Model inputs used by DualTrack.forward_dict()
        "global_encoder_images": global_x,
        "local_encoder_images": local_x,
    }

    print("\nDummy batch keys:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            describe_tensor(f"batch['{key}']", value)
        else:
            print(f"batch['{key}']: {value}")

    # -----------------------------
    # 3. Predict through official-like path
    # -----------------------------
    print("\nRunning official-like predict: pred = model.predict(batch)")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        pred = model.predict(batch)

    print("\nPredict completed.")
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

    # -----------------------------
    # 4. Confirm forward_dict side effect
    # -----------------------------
    print("\nChecking whether prediction was added to batch...")

    if "prediction" not in batch:
        raise RuntimeError(
            "batch['prediction'] was not added. "
            "This suggests model.predict(batch) did not call forward_dict() as expected."
        )

    describe_tensor("batch['prediction']", batch["prediction"])

    if not torch.equal(pred, batch["prediction"]):
        max_abs_diff = (pred - batch["prediction"]).abs().max().item()
        raise RuntimeError(
            f"pred and batch['prediction'] differ. max_abs_diff={max_abs_diff}"
        )

    print("batch['prediction'] check passed.")

    # -----------------------------
    # 5. Basic finite-value check
    # -----------------------------
    if not torch.isfinite(pred).all():
        num_nan = torch.isnan(pred).sum().item()
        num_inf = torch.isinf(pred).sum().item()
        raise RuntimeError(
            f"Prediction contains non-finite values: "
            f"NaN={num_nan}, Inf={num_inf}"
        )

    print("Finite-value check passed.")

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"CUDA peak memory allocated: {peak_allocated:.3f} GiB")
        print(f"CUDA peak memory reserved : {peak_reserved:.3f} GiB")

    print("\nDummy predict(batch) test finished successfully.")


if __name__ == "__main__":
    main()
