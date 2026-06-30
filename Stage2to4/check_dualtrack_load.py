from omegaconf import OmegaConf
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models import get_model
from src.utils.pseudo3d_processing import resolve_dualtrack_checkpoint


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
    print(cfg)

    model = get_model(**cfg)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel loaded successfully.")
    print("Model class:", model.__class__.__name__)
    print(f"num_params    : {num_params:,}")
    print(f"num_trainable : {num_trainable:,}")

    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)


if __name__ == "__main__":
    main()
