from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.file_list_mode import read_file_list  # noqa: E402
from stage5.utils.split_contract import resolve_active_contract  # noqa: E402
from stage5.utils.rotation_augmentation import (  # noqa: E402
    MODE_NONE,
    MODE_RANDOM_Z_ROTATION,
    stable_video_id_from_path,
)

# ----------------------------------------------------------------------------
# S5-15 P2: limited GPU preflight (report section "GPU preflight実施手順").
#
# Stage B: dummy forward/backward on synthetic tensors -- finite loss and
#          gradients, and an optimizer step that actually moves parameters.
# Stage C: a few real-H5 steps over a fixed head-of-list subset, twice (none
#          and random_z_rotation), two epochs each, checking that one video
#          sees one angle per epoch, that the angle advances between epochs
#          through DataLoader workers, that nothing but XYZ changes, and that
#          the validation dataset is never augmented.
#
# This is a diagnostic checker in the S5-14 mould: it builds the model and
# iterates the Dataset itself. It is NOT a training path -- R0/R1 continue to
# run through train_stage5.sh. It writes only under its own output directory
# and never into stage5_runs/. Stage A (the bash/fixed-list path) is a
# PREFLIGHT_ONLY=1 run of train_stage5.sh, driven by the companion .sh.
# ----------------------------------------------------------------------------

DEFAULT_NUM_TRAIN_VIDEOS = 4
DEFAULT_NUM_VAL_VIDEOS = 1
DEFAULT_EPOCHS = 2
DEFAULT_NUM_WORKERS = 2
DEFAULT_ROTATION_DEGREES = 15.0
DEFAULT_CLASS_WEIGHT = (0.05963856, 1.94036150)
DISTANCE_TOLERANCE = 1e-4

TIMESTAMP_VIDEO_ID_PATTERN = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9]+")
ABSOLUTE_HOST_PATH_PATTERN = re.compile(r"(/mnt/data|/home/[A-Za-z0-9_.-]+)")


def select_subset(paths: list[Path], count: int, *, label: str) -> list[Path]:
    """Take the first `count` entries, in list order.

    Fixed head-of-list selection rather than sampling, so the subset is
    reproducible and not chosen to flatter the result.
    """
    if count <= 0:
        raise ValueError(f"{label} subset size must be positive, got {count}")
    if len(paths) < count:
        raise ValueError(f"{label} list holds {len(paths)} files, need at least {count}")
    return list(paths[:count])


def angle_violations(
    observations: list[dict[str, Any]],
    *,
    mode: str,
    epochs: tuple[int, ...],
    split: str,
) -> list[str]:
    """Check the per-(video, epoch) angle bookkeeping against the stop conditions.

    Torch-free so it can be exercised on a CPU host with synthetic records.
    """
    violations: list[str] = []
    by_key: dict[tuple[int, str], set[float | None]] = {}
    for record in observations:
        key = (int(record["epoch"]), str(record["video_alias"]))
        by_key.setdefault(key, set()).add(record["rotation_angle_degrees"])

    if mode == MODE_NONE or split == "validation":
        for (epoch, alias), angles in sorted(by_key.items()):
            if angles != {None}:
                violations.append(
                    f"{split} {alias} epoch {epoch}: expected no angle for mode={mode!r}, saw {sorted(angles, key=str)}"
                )
        return violations

    for (epoch, alias), angles in sorted(by_key.items()):
        if len(angles) != 1:
            violations.append(
                f"{split} {alias} epoch {epoch}: {len(angles)} distinct angles in one epoch ({sorted(angles)})"
            )
        elif None in angles:
            violations.append(f"{split} {alias} epoch {epoch}: augmentation is on but no angle was recorded")

    aliases = sorted({alias for _, alias in by_key})
    for alias in aliases:
        per_epoch = [
            next(iter(by_key[(epoch, alias)])) for epoch in epochs if (epoch, alias) in by_key
        ]
        if len(per_epoch) < 2:
            continue
        if len(set(per_epoch)) == 1:
            violations.append(
                f"{split} {alias}: the same angle {per_epoch[0]} in every epoch -- the epoch never reached the workers"
            )
    return violations


def correspondence_violations(reference: dict[str, Any], augmented: dict[str, Any], *, label: str) -> list[str]:
    """Everything except XYZ must survive the rotation untouched, and the
    rotation must preserve distances on real data."""
    import torch

    violations: list[str] = []
    for key in ("features", "labels", "valid_mask", "frame_order", "point_indices"):
        if not torch.equal(reference[key], augmented[key]):
            violations.append(f"{label}: {key} changed under augmentation")
    for key in ("window_start", "window_end"):
        if reference[key] != augmented[key]:
            violations.append(f"{label}: {key} changed under augmentation")
    if reference["meta"]["h5_index"] != augmented["meta"]["h5_index"]:
        violations.append(f"{label}: h5_index changed under augmentation")

    reference_points = reference["points"].to(torch.float64)
    augmented_points = augmented["points"].to(torch.float64)
    if reference_points.shape != augmented_points.shape:
        violations.append(f"{label}: point count changed under augmentation")
        return violations

    num_points = reference_points.shape[0]
    if num_points >= 2:
        generator = torch.Generator().manual_seed(20260916)
        left = torch.randint(0, num_points, (min(128, num_points),), generator=generator)
        right = torch.randint(0, num_points, (min(128, num_points),), generator=generator)
        before = (reference_points[left] - reference_points[right]).norm(dim=1)
        after = (augmented_points[left] - augmented_points[right]).norm(dim=1)
        max_drift = (before - after).abs().max().item()
        if max_drift > DISTANCE_TOLERANCE:
            violations.append(f"{label}: pairwise distances drifted by {max_drift} under rotation")
    return violations


def none_mode_regression_violations(reference: dict[str, Any], observed: dict[str, Any], *, label: str) -> list[str]:
    import torch

    violations: list[str] = []
    if not torch.equal(reference["points"], observed["points"]):
        violations.append(f"{label}: points differ from the unaugmented reference while mode=none")
    if observed["meta"]["rotation_angle_degrees"] is not None:
        violations.append(f"{label}: an angle was recorded while mode=none")
    return violations


def finite_violations(name: str, value: float) -> list[str]:
    if value is None or not math.isfinite(float(value)):
        return [f"{name} is not finite: {value}"]
    return []


def build_model_and_loss(*, checkpoint: str | None, device: str, class_weight: tuple[float, float]):
    """Build exactly what train_stage5.py builds.

    The model returns {"logits": [B, N, C]} and the loss is called as
    loss_fn(output, batch), returning a dict with loss/loss_sum/
    loss_normalizer -- the preflight exercises that real path rather than a
    hand-rolled cross entropy, so what it checks is what training will run.
    """
    from stage5.models import build_stage5_model
    from stage5.training import build_loss

    model = build_stage5_model(
        "pointnext_s",
        num_classes=2,
        feature_dim=2,
        checkpoint=checkpoint,
        strict_checkpoint=bool(checkpoint),
        width=32,
        depth=6,
        expansion=4,
        dropout=0.0,
        use_global_context=True,
        pointnext_radius=0.1,
        pointnext_nsample=32,
        pointnext_sa_layers=2,
        pointnext_sa_use_res=True,
        pointnext_norm="groupnorm",
        pointnext_norm_groups=8,
    ).to(device)
    loss_fn = build_loss(
        "cross_entropy",
        ignore_index=-1,
        class_weight=list(class_weight),
        label_smoothing=0.0,
    ).to(device)
    return model, loss_fn


def run_stage_b(*, checkpoint: str | None, device: str, num_points: int = 4096) -> dict[str, Any]:
    """Synthetic forward/backward: no H5, no Dataset, no DataLoader."""
    import torch

    violations: list[str] = []
    model, loss_fn = build_model_and_loss(
        checkpoint=checkpoint, device=device, class_weight=DEFAULT_CLASS_WEIGHT
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    generator = torch.Generator().manual_seed(20260916)
    batch = {
        "points": torch.randn(1, num_points, 3, generator=generator).to(device),
        "features": torch.rand(1, num_points, 2, generator=generator).to(device),
        "labels": (torch.rand(1, num_points, generator=generator) < 0.05).long().to(device),
        "valid_mask": (torch.rand(1, num_points, generator=generator) < 0.95).to(device),
    }

    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss_dict = loss_fn(output, batch)
    loss = loss_dict["loss"]
    loss_dict.get("loss_sum", loss).backward()

    loss_value = float(loss.detach().cpu())
    violations += finite_violations("stage_b.loss", loss_value)

    grad_norm = 0.0
    num_grads = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        num_grads += 1
        value = float(parameter.grad.detach().norm().cpu())
        violations += finite_violations("stage_b.gradient_norm", value)
        grad_norm += value**2
    grad_norm = math.sqrt(grad_norm)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    changed = sum(
        1
        for previous, parameter in zip(before, model.parameters())
        if not torch.equal(previous, parameter.detach())
    )
    if changed == 0:
        violations.append("stage_b: the optimizer step changed no parameter at all")

    return {
        "stage": "b_dummy_forward_backward",
        "device": device,
        "num_points": num_points,
        "loss": loss_value,
        "total_gradient_norm": grad_norm,
        "num_parameters_with_gradient": num_grads,
        "num_parameters_changed_by_step": changed,
        "optimizer_steps": 1,
        "violations": violations,
    }


def run_stage_c(
    *,
    train_paths: list[Path],
    val_paths: list[Path],
    mode: str,
    epochs: int,
    num_workers: int,
    rotation_degrees: float,
    seed: int,
    checkpoint: str | None,
    device: str,
    gradient_accumulation_steps: int,
    skip_optimization: bool,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate

    violations: list[str] = []
    dataset_kwargs: dict[str, Any] = {
        "num_points": None,
        "features": "intensity,confidence",
        "window_mode": "overlap",
        "window_size_frames": 16,
        "window_stride_frames": 8,
        "include_tail_window": True,
        "label_policy": "bbox_noncontour_ignore",
        "seed": seed,
    }
    train_dataset = Pseudo3DPointCloudDataset(
        train_paths,
        augmentation_mode=mode,
        augmentation_rotation_degrees=rotation_degrees,
        **dataset_kwargs,
    )
    reference_dataset = Pseudo3DPointCloudDataset(train_paths, augmentation_mode=MODE_NONE, **dataset_kwargs)
    val_dataset = Pseudo3DPointCloudDataset(val_paths, **dataset_kwargs)

    alias_by_index = {index: f"train_{index:03d}" for index in range(len(train_paths))}
    val_alias_by_index = {index: f"validation_{index:03d}" for index in range(len(val_paths))}

    model = loss_fn = optimizer = None
    if not skip_optimization:
        model, loss_fn = build_model_and_loss(
            checkpoint=checkpoint, device=device, class_weight=DEFAULT_CLASS_WEIGHT
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_point_window_collate,
    )

    epoch_range = tuple(range(1, epochs + 1))
    observations: list[dict[str, Any]] = []
    losses: list[float] = []
    optimizer_steps = 0
    windows_seen = 0

    for epoch in epoch_range:
        train_dataset.set_epoch(epoch)
        accumulated = 0
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            for meta in batch["meta"]:
                observations.append(
                    {
                        "epoch": epoch,
                        "video_alias": alias_by_index[int(meta["h5_index"])],
                        "window_id": int(meta["window_id"]),
                        "rotation_angle_degrees": meta["rotation_angle_degrees"],
                    }
                )
            windows_seen += len(batch["meta"])

            if optimizer is None:
                continue
            model.train()
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(device_batch)
            loss_dict = loss_fn(output, device_batch)
            loss = loss_dict["loss"]
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            violations += finite_violations(f"stage_c.epoch{epoch}.loss", loss_value)
            loss_dict.get("loss_sum", loss).backward()
            accumulated += 1
            if accumulated >= gradient_accumulation_steps or batch_index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        violations += finite_violations(
                            f"stage_c.epoch{epoch}.gradient_norm",
                            float(parameter.grad.detach().norm().cpu()),
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0
    del loader

    violations += angle_violations(observations, mode=mode, epochs=epoch_range, split="train")

    # Point correspondence against the unaugmented reference, for the same
    # window indices and the last epoch's angles.
    train_dataset.set_epoch(epoch_range[-1])
    for index in range(len(train_dataset)):
        reference_sample = reference_dataset[index]
        observed_sample = train_dataset[index]
        label = f"train sample {index}"
        if mode == MODE_NONE:
            violations += none_mode_regression_violations(reference_sample, observed_sample, label=label)
        else:
            violations += correspondence_violations(reference_sample, observed_sample, label=label)

    # The validation dataset must never be augmented, whatever the arm is.
    val_observations: list[dict[str, Any]] = []
    for index in range(len(val_dataset)):
        sample = val_dataset[index]
        val_observations.append(
            {
                "epoch": epoch_range[-1],
                "video_alias": val_alias_by_index[int(sample["meta"]["h5_index"])],
                "window_id": int(sample["meta"]["window_id"]),
                "rotation_angle_degrees": sample["meta"]["rotation_angle_degrees"],
            }
        )
    violations += angle_violations(
        val_observations, mode=mode, epochs=(epoch_range[-1],), split="validation"
    )

    angles_by_epoch: dict[str, dict[str, Any]] = {}
    for record in observations:
        angles_by_epoch.setdefault(str(record["epoch"]), {})[record["video_alias"]] = record[
            "rotation_angle_degrees"
        ]

    return {
        "stage": "c_real_h5_few_steps",
        "mode": mode,
        "device": device,
        "epochs": list(epoch_range),
        "num_train_videos": len(train_paths),
        "num_val_videos": len(val_paths),
        "num_train_windows_per_epoch": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "windows_seen_total": windows_seen,
        "optimizer_steps": optimizer_steps,
        "num_workers": num_workers,
        "losses": losses,
        "angles_by_epoch": angles_by_epoch,
        "skip_optimization": skip_optimization,
        "violations": violations,
    }


def privacy_self_check(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    return {
        "timestamp_like_video_ids_absent": not TIMESTAMP_VIDEO_ID_PATTERN.findall(text),
        "absolute_host_paths_absent": not ABSOLUTE_HOST_PATH_PATTERN.findall(text),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-15 limited GPU preflight: stage B (synthetic forward/backward) and "
        "stage C (a few real-H5 steps over a fixed subset). Writes only under --output_dir "
        "and never into stage5_runs/."
    )
    parser.add_argument("--stage", choices=["b", "c"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Initialization checkpoint")
    parser.add_argument(
        "--split_manifest",
        default=None,
        help="Name of an APPROVED pin in stage5/config/split_contract_pins.json (not a path). "
        "Falls back to $STAGE5_SPLIT_MANIFEST; absent or unapproved stops the run.",
    )
    parser.add_argument("--split_contract_pins", default=None)
    parser.add_argument("--artifact_coverage", default=None)
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument("--num_train_videos", type=int, default=DEFAULT_NUM_TRAIN_VIDEOS)
    parser.add_argument("--num_val_videos", type=int, default=DEFAULT_NUM_VAL_VIDEOS)
    parser.add_argument("--augmentation", choices=[MODE_NONE, MODE_RANDOM_Z_ROTATION], default=MODE_NONE)
    parser.add_argument("--augmentation_rotation_degrees", type=float, default=DEFAULT_ROTATION_DEGREES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip_optimization",
        action="store_true",
        help="Stage C data-path checks only, without building the model. Used to verify this "
        "checker itself on a CPU host; the real preflight does not pass it.",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if "stage5_runs" in output_dir.resolve().parts:
        print(f"FAIL: refusing to write under stage5_runs/: {output_dir}", file=sys.stderr)
        sys.exit(2)
    if output_dir.is_dir() and any(output_dir.iterdir()):
        print(f"FAIL: output directory already holds artifacts: {output_dir}", file=sys.stderr)
        sys.exit(2)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "b":
        result = run_stage_b(checkpoint=args.checkpoint, device=args.device)
        private_extra: dict[str, Any] = {}
    else:
        if not args.train_list or not args.val_list:
            print("FAIL: stage c needs --train_list and --val_list", file=sys.stderr)
            sys.exit(2)
        all_train = read_file_list(args.train_list, label="train")
        all_val = read_file_list(args.val_list, label="val")

        # S5-16 Step 0: the whole list is checked, not just the subset actually
        # loaded -- a sealed video that happens to fall outside the subset this
        # time is still a sealed video in the list this preflight is vouching for.
        contract = resolve_active_contract(
            manifest_key=args.split_manifest,
            pins_path=args.split_contract_pins,
            coverage_path=args.artifact_coverage,
        )
        contract.assert_paths_allowed(all_train, purpose="GPU preflight train inputs")
        contract.assert_paths_allowed(all_val, purpose="GPU preflight validation inputs")

        train_paths = select_subset(all_train, args.num_train_videos, label="train")
        val_paths = select_subset(all_val, args.num_val_videos, label="val")
        result = run_stage_c(
            train_paths=train_paths,
            val_paths=val_paths,
            mode=args.augmentation,
            epochs=args.epochs,
            num_workers=args.num_workers,
            rotation_degrees=args.augmentation_rotation_degrees,
            seed=args.seed,
            checkpoint=args.checkpoint,
            device=args.device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            skip_optimization=args.skip_optimization,
        )
        private_extra = {
            "train_video_ids": [stable_video_id_from_path(p) for p in train_paths],
            "val_video_ids": [stable_video_id_from_path(p) for p in val_paths],
        }

    violations = result["violations"]
    result["status"] = "passed" if not violations else "failed"

    private_summary = {**result, **private_extra}
    shareable_summary = {key: value for key, value in result.items()}
    shareable_summary["privacy_self_check"] = privacy_self_check(shareable_summary)

    (output_dir / "preflight_private.json").write_text(
        json.dumps(private_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "preflight_shareable.json").write_text(
        json.dumps(shareable_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Stage5 S5-15 GPU preflight stage {args.stage}")
    print(f"  status          : {result['status']}")
    if args.stage == "b":
        print(f"  loss            : {result['loss']}")
        print(f"  gradient norm   : {result['total_gradient_norm']}")
        print(f"  params changed  : {result['num_parameters_changed_by_step']}")
    else:
        print(f"  augmentation    : {result['mode']}")
        print(f"  windows/epoch   : {result['num_train_windows_per_epoch']} over {result['num_train_videos']} videos")
        print(f"  optimizer steps : {result['optimizer_steps']}")
        print(f"  angles by epoch : {json.dumps(result['angles_by_epoch'], ensure_ascii=False)}")
    if violations:
        print(f"  violations ({len(violations)}):")
        for violation in violations:
            print(f"    - {violation}")
    privacy = shareable_summary["privacy_self_check"]
    print(
        "  privacy         : "
        f"timestamp_like_video_ids_absent={privacy['timestamp_like_video_ids_absent']} "
        f"absolute_host_paths_absent={privacy['absolute_host_paths_absent']}"
    )
    print(f"  output          : {output_dir}")

    sys.exit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
