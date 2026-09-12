from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from checks.dummy.check_dummy_pointnext_s_forward import make_window_dummy_h5, move_batch_to_device
from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate
from stage5.models import build_stage5_model
from stage5.models.model_factory import _extract_state_dict
from stage5.models.norm_layers import BN_BUFFER_SUFFIXES
from stage5.training import build_loss


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------
# S5-11 Step D5: transfer a BatchNorm Stage5 checkpoint's trainable parameters
# (convs/linear/head + norm affine weight/bias) into a freshly constructed
# GroupNorm model, excluding only the BatchNorm-specific running buffers,
# which GroupNorm does not have. Stage5GroupNorm subclasses nn.GroupNorm
# directly (see stage5/models/norm_layers.py), so a norm site's `.weight`/
# `.bias` sit at the exact same dotted key as they did under BatchNorm --
# no key remapping is required, only a shape-checked key intersection.
# ----------------------------------------------------------------------------


def transfer_batchnorm_to_groupnorm_state(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], list[str]]:
    """Return (filtered_state, source_unused_bn_buffer_keys, shape_mismatches, target_missing_keys)."""
    filtered: dict[str, Any] = {}
    target_missing: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []

    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            target_missing.append(key)
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": tuple(source_value.shape),
                    "target_shape": tuple(target_value.shape),
                }
            )
            continue
        filtered[key] = source_value

    source_unused = [key for key in source_state if key not in target_state]
    return filtered, source_unused, shape_mismatches, target_missing


def classify_source_unused(source_unused: list[str]) -> tuple[list[str], list[str]]:
    """Split source-only keys into expected BatchNorm buffer keys vs anything else."""
    expected_bn_buffers = [key for key in source_unused if key.endswith(BN_BUFFER_SUFFIXES)]
    unexpected = [key for key in source_unused if not key.endswith(BN_BUFFER_SUFFIXES)]
    return expected_bn_buffers, unexpected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-11 Step D5: transfer a BatchNorm Stage5 partial-init checkpoint's trainable "
        "parameters into a freshly constructed GroupNorm PointNeXt-S model."
    )
    parser.add_argument("--self_test", action="store_true", help="Run CPU-only synthetic tests and exit")
    parser.add_argument(
        "--source_checkpoint",
        default=str(
            REPO_ROOT
            / "work_dirs"
            / "_s3dis_to_stage5_pointnext_s_transfer"
            / "stage5_pointnext_s_s3dis_partial_init.pt"
        ),
        help="Existing Stage5 BatchNorm S3DIS partial-init checkpoint (the same one the "
        "teacher v6 BatchNorm control run initializes from)",
    )
    parser.add_argument(
        "--output_checkpoint",
        default=str(
            REPO_ROOT
            / "work_dirs"
            / "_s3dis_to_stage5_pointnext_s_transfer"
            / "stage5_pointnext_s_s3dis_partial_init_groupnorm.pt"
        ),
    )
    parser.add_argument("--report_path", default=None, help="Defaults to <output_checkpoint>.transfer_report.json")
    parser.add_argument("--norm_groups", type=int, default=8)
    parser.add_argument("--features", default="intensity,confidence")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--points_per_frame", type=int, default=80)
    parser.add_argument("--window_size_frames", type=int, default=8)
    parser.add_argument("--window_stride_frames", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_self_test() -> None:
    source_state = {
        "pointnext.encoder.encoder.1.0.convs.0.1.weight": torch.zeros(32),
        "pointnext.encoder.encoder.1.0.convs.0.1.bias": torch.zeros(32),
        "pointnext.encoder.encoder.1.0.convs.0.1.running_mean": torch.zeros(32),
        "pointnext.encoder.encoder.1.0.convs.0.1.running_var": torch.ones(32),
        "pointnext.encoder.encoder.1.0.convs.0.1.num_batches_tracked": torch.tensor(0),
        "pointnext.encoder.encoder.1.0.convs.0.0.weight": torch.zeros(32, 4, 1),
        "pointnext.head.head.0.1.weight": torch.zeros(16),
        "pointnext.head.head.0.1.bias": torch.zeros(16),
        "pointnext.head.head.0.1.running_mean": torch.zeros(16),
        "pointnext.head.head.0.1.running_var": torch.ones(16),
        "pointnext.head.head.0.1.num_batches_tracked": torch.tensor(0),
    }
    # Target (GroupNorm) has the same conv/affine keys, no running buffers, and
    # deliberately one shape-mismatched key to exercise the mismatch path.
    target_state = {
        "pointnext.encoder.encoder.1.0.convs.0.1.weight": torch.ones(32),
        "pointnext.encoder.encoder.1.0.convs.0.1.bias": torch.ones(32),
        "pointnext.encoder.encoder.1.0.convs.0.0.weight": torch.ones(32, 4, 1),
        "pointnext.head.head.0.1.weight": torch.ones(16),
        "pointnext.head.head.0.1.bias": torch.ones(16),
    }

    filtered, source_unused, shape_mismatches, target_missing = transfer_batchnorm_to_groupnorm_state(
        source_state, target_state
    )
    require(set(filtered.keys()) == set(target_state.keys()), "All target keys must be filled from a matching source")
    require(not target_missing, f"No target key should be missing in this fixture, got {target_missing}")
    require(not shape_mismatches, f"No shape mismatch expected in this fixture, got {shape_mismatches}")

    expected_bn_buffers, unexpected = classify_source_unused(source_unused)
    require(
        set(expected_bn_buffers)
        == {
            "pointnext.encoder.encoder.1.0.convs.0.1.running_mean",
            "pointnext.encoder.encoder.1.0.convs.0.1.running_var",
            "pointnext.encoder.encoder.1.0.convs.0.1.num_batches_tracked",
            "pointnext.head.head.0.1.running_mean",
            "pointnext.head.head.0.1.running_var",
            "pointnext.head.head.0.1.num_batches_tracked",
        },
        f"Unexpected set of classified BN buffer keys: {expected_bn_buffers}",
    )
    require(not unexpected, f"No unexpected source-only keys expected in this fixture, got {unexpected}")

    # Negative control: a genuinely unexpected source-only key (not a BN buffer)
    # must be classified as unexpected, not silently absorbed.
    source_state_with_stray_key = dict(source_state)
    source_state_with_stray_key["pointnext.some_removed_module.weight"] = torch.zeros(4)
    _, source_unused_2, _, _ = transfer_batchnorm_to_groupnorm_state(source_state_with_stray_key, target_state)
    _, unexpected_2 = classify_source_unused(source_unused_2)
    require(
        unexpected_2 == ["pointnext.some_removed_module.weight"],
        f"A non-BN-buffer source-only key must be classified as unexpected, got {unexpected_2}",
    )

    # Negative control: a missing target key (present in target, absent from source) must surface.
    target_state_with_extra_key = dict(target_state)
    target_state_with_extra_key["pointnext.new_head.weight"] = torch.zeros(2)
    _, _, _, target_missing_2 = transfer_batchnorm_to_groupnorm_state(source_state, target_state_with_extra_key)
    require(
        target_missing_2 == ["pointnext.new_head.weight"],
        f"An unmatched target key must be reported as missing, got {target_missing_2}",
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("Stage5 BatchNorm->GroupNorm transfer self-test passed.")
        return

    if args.device != "cuda":
        raise RuntimeError("PointNeXt-S CUDA ops require --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)

    source_path = Path(args.source_checkpoint)
    require(source_path.is_file(), f"Source (BatchNorm) checkpoint not found: {source_path}")
    source_checkpoint = torch.load(source_path, map_location="cpu")
    source_state = _extract_state_dict(source_checkpoint)
    source_config = source_checkpoint.get("config", {}) if isinstance(source_checkpoint, dict) else {}

    output_path = Path(args.output_checkpoint)
    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".transfer_report.json")

    work_dir = output_path.parent / "_batchnorm_to_groupnorm_transfer_dummy_data"
    work_dir.mkdir(parents=True, exist_ok=True)
    h5_path = work_dir / "dummy_transfer_check_grid.h5"
    make_window_dummy_h5(h5_path, num_frames=args.num_frames, points_per_frame=args.points_per_frame, seed=args.seed)
    dataset = Pseudo3DPointCloudDataset(
        [h5_path],
        num_points=None,
        features=args.features,
        window_mode="overlap",
        window_size_frames=args.window_size_frames,
        window_stride_frames=args.window_stride_frames,
        include_tail_window=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=pad_point_window_collate
    )
    batch = move_batch_to_device(next(iter(loader)), device)

    model_kwargs = dict(
        num_classes=int(source_config.get("num_classes", 2)),
        feature_dim=dataset.num_feature_channels,
        width=int(source_config.get("width", 32)),
        expansion=int(source_config.get("expansion", 4)),
        dropout=float(source_config.get("dropout", 0.0)),
        pointnext_radius=float(source_config.get("pointnext_radius", source_config.get("radius", 0.1))),
        pointnext_nsample=int(source_config.get("pointnext_nsample", source_config.get("nsample", 16))),
        pointnext_sa_layers=int(source_config.get("pointnext_sa_layers", source_config.get("sa_layers", 2))),
        pointnext_sa_use_res=bool(source_config.get("pointnext_sa_use_res", source_config.get("sa_use_res", True))),
        pointnext_norm="groupnorm",
        pointnext_norm_groups=int(args.norm_groups),
    )
    model = build_stage5_model("pointnext_s", **model_kwargs).to(device)
    target_state = model.state_dict()

    filtered, source_unused, shape_mismatches, target_missing = transfer_batchnorm_to_groupnorm_state(
        source_state, target_state
    )
    expected_bn_buffers, unexpected_source_unused = classify_source_unused(source_unused)

    require(not target_missing, f"GroupNorm model has keys with no BatchNorm-checkpoint counterpart: {target_missing}")
    require(not shape_mismatches, f"Shape mismatch(es) between BatchNorm source and GroupNorm target: {shape_mismatches}")
    require(
        not unexpected_source_unused,
        f"Unexpected unused BatchNorm-checkpoint key(s) beyond running buffers: {unexpected_source_unused}",
    )

    load_result = model.load_state_dict(filtered, strict=False)
    require(
        not load_result.missing_keys and not load_result.unexpected_keys,
        f"load_state_dict reported missing={load_result.missing_keys}, "
        f"unexpected={load_result.unexpected_keys} despite the pre-check finding none",
    )

    loss_fn = build_loss("cross_entropy", ignore_index=-1).to(device)
    model.eval()
    with torch.inference_mode():
        output = model(batch)
        loss_dict = loss_fn(output, batch)
    logits = output["logits"]
    require(bool(torch.isfinite(logits).all()), "GroupNorm transfer-initialized model produced non-finite logits")
    require(bool(torch.isfinite(loss_dict["loss"])), f"Loss is not finite: {loss_dict['loss']}")

    summary = {
        "source_checkpoint": str(source_path),
        "source_keys": len(source_state),
        "target_keys": len(target_state),
        "loaded_keys": len(filtered),
        "excluded_batchnorm_buffer_keys": len(expected_bn_buffers),
        "target_missing_keys": target_missing,
        "shape_mismatches": shape_mismatches,
        "unexpected_source_unused_keys": unexpected_source_unused,
        "norm_groups": int(args.norm_groups),
        "feature_dim": dataset.num_feature_channels,
        "logits_shape": tuple(logits.shape),
        "loss": float(loss_dict["loss"].item()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {**source_config, **model.get_config()},
            "transfer": summary,
        },
        output_path,
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 BatchNorm -> GroupNorm transfer check passed.")
    print(f"source checkpoint: {source_path}")
    print(f"source keys: {len(source_state)} / target keys: {len(target_state)}")
    print(f"loaded keys: {len(filtered)} (excluded BN buffer keys: {len(expected_bn_buffers)})")
    print(f"loss: {float(loss_dict['loss'].item()):.6f}")
    print(f"output checkpoint: {output_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
