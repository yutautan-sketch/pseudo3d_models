from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch

from stage5.models import build_stage5_model


def make_dummy_pointcloud_h5(path: Path, *, num_points: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame_order = np.arange(num_points, dtype=np.int32) // 32
    pixel_xy = np.stack(
        [
            rng.integers(0, 256, size=num_points),
            rng.integers(0, 256, size=num_points),
        ],
        axis=1,
    ).astype(np.float32)

    with h5py.File(path, "w") as f:
        point_cloud = f.create_group("point_cloud")
        point_cloud.attrs["point_mode"] = "grid"
        point_cloud.create_dataset(
            "points",
            data=rng.normal(size=(num_points, 3)).astype(np.float32),
        )
        point_cloud.create_dataset(
            "intensity",
            data=rng.integers(0, 256, size=num_points, dtype=np.uint8),
        )
        point_cloud.create_dataset(
            "alpha",
            data=rng.integers(0, 256, size=num_points, dtype=np.uint8),
        )
        point_cloud.create_dataset(
            "confidence",
            data=rng.random(size=num_points).astype(np.float32),
        )
        point_cloud.create_dataset("frame_order", data=frame_order)
        point_cloud.create_dataset("pixel_xy", data=pixel_xy)
        point_cloud.create_dataset(
            "frame_index",
            data=(frame_order + 1).astype(np.int64),
        )
        point_cloud.create_dataset(
            "source_type",
            data=np.zeros(num_points, dtype=np.uint8),
        )

        labels = rng.choice([-1, 0, 1], size=num_points, p=[0.15, 0.65, 0.20]).astype(np.int8)
        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=labels)
        annotation.create_dataset("valid_mask", data=(labels != -1))


def make_dummy_checkpoint(path: Path, *, seed: int) -> None:
    torch.manual_seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "mlp_baseline",
        "num_classes": 2,
        "feature_names": ["intensity", "confidence"],
        "feature_dim": 2,
        "width": 16,
        "depth": 2,
        "expansion": 2,
        "dropout": 0.0,
        "no_global_context": False,
        "no_normalize_points": False,
    }
    model = build_stage5_model(
        "mlp_baseline",
        num_classes=config["num_classes"],
        feature_dim=config["feature_dim"],
        width=config["width"],
        depth=config["depth"],
        expansion=config["expansion"],
        dropout=config["dropout"],
        use_global_context=not config["no_global_context"],
    )
    torch.save(
        {
            "epoch": 0,
            "model": model.state_dict(),
            "config": config,
            "best_score": 0.0,
        },
        path,
    )


def run_inference(
    *,
    input_h5: Path,
    checkpoint: Path,
    output_h5: Path,
    output_ply: Path,
    chunk_size: int,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "infer_stage5.py"),
        "--input_h5",
        str(input_h5),
        "--checkpoint",
        str(checkpoint),
        "--output_h5",
        str(output_h5),
        "--output_ply",
        str(output_ply),
        "--chunk_size",
        str(chunk_size),
        "--device",
        "cpu",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_outputs(
    *,
    output_h5: Path,
    output_ply: Path,
    num_points: int,
    num_classes: int,
) -> None:
    if not output_h5.exists():
        raise AssertionError(f"Output H5 was not created: {output_h5}")
    if not output_ply.exists():
        raise AssertionError(f"Output PLY was not created: {output_ply}")

    with h5py.File(output_h5, "r") as f:
        if "point_cloud" not in f:
            raise AssertionError("Output H5 does not contain copied point_cloud group")
        if "prediction" not in f:
            raise AssertionError("Output H5 does not contain prediction group")

        prediction = f["prediction"]
        logits = prediction["logits"][:]
        prob_femur = prediction["prob_femur"][:]
        pred_label = prediction["pred_label"][:]

    if logits.shape != (num_points, num_classes):
        raise AssertionError(f"Unexpected logits shape: {logits.shape}")
    if prob_femur.shape != (num_points,):
        raise AssertionError(f"Unexpected prob_femur shape: {prob_femur.shape}")
    if pred_label.shape != (num_points,):
        raise AssertionError(f"Unexpected pred_label shape: {pred_label.shape}")
    if not np.isfinite(logits).all():
        raise AssertionError("logits contains NaN or inf")
    if not np.isfinite(prob_femur).all():
        raise AssertionError("prob_femur contains NaN or inf")
    if np.any((prob_femur < 0.0) | (prob_femur > 1.0)):
        raise AssertionError("prob_femur contains values outside [0, 1]")
    if not set(np.unique(pred_label).tolist()).issubset({0, 1}):
        raise AssertionError(f"pred_label has unexpected values: {np.unique(pred_label)}")

    with output_ply.open("r", encoding="utf-8") as f:
        header = [next(f).strip() for _ in range(10)]
    if header[0] != "ply" or f"element vertex {num_points}" not in header:
        raise AssertionError("PLY header does not match expected vertex count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dummy Stage5 inputs and validate infer_stage5.py")
    parser.add_argument(
        "--work_dir",
        default=str(REPO_ROOT / "_dummy_inference_check"),
        help="Directory for temporary dummy input/checkpoint/output files",
    )
    parser.add_argument("--num_points", type=int, default=257)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    input_h5 = work_dir / "dummy_pointcloud_annotated_grid.h5"
    checkpoint = work_dir / "dummy_mlp_baseline.pt"
    output_h5 = work_dir / "dummy_prediction.h5"
    output_ply = work_dir / "dummy_prediction.ply"

    make_dummy_pointcloud_h5(input_h5, num_points=args.num_points, seed=args.seed)
    make_dummy_checkpoint(checkpoint, seed=args.seed)
    run_inference(
        input_h5=input_h5,
        checkpoint=checkpoint,
        output_h5=output_h5,
        output_ply=output_ply,
        chunk_size=args.chunk_size,
    )
    validate_outputs(
        output_h5=output_h5,
        output_ply=output_ply,
        num_points=args.num_points,
        num_classes=2,
    )

    print("Dummy inference check passed.")
    print(f"work_dir: {work_dir}")
    print("Generated files are kept for inspection. Remove the work_dir manually when no longer needed.")


if __name__ == "__main__":
    main()
