from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))
sys.path.insert(0, str(REPO_ROOT / "checks" / "dummy"))

from check_dummy_rotation_augmentation_dataset import build_fixture_videos  # noqa: E402
from check_stage5_s5_15_gpu_preflight import (  # noqa: E402
    angle_violations,
    correspondence_violations,
    finite_violations,
    none_mode_regression_violations,
    privacy_self_check,
    run_stage_c,
    select_subset,
)
from stage5.utils.rotation_augmentation import MODE_NONE, MODE_RANDOM_Z_ROTATION  # noqa: E402

# ----------------------------------------------------------------------------
# S5-15: CPU-side coverage for the limited GPU preflight.
#
# The stop-condition logic is torch-free and is exercised directly against
# constructed observation records, including every failure the preflight is
# supposed to halt on. Stage C's data path is then run for real on synthetic
# H5 fixtures with --skip_optimization, so everything except the GPU model
# step is verified here; stage B and the model step can only run on the GPU
# host.
# ----------------------------------------------------------------------------

CHECKER = REPO_ROOT / "checks" / "real_h5" / "check_stage5_s5_15_gpu_preflight.py"


def observation(epoch: int, alias: str, angle: float | None, window_id: int = 0) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "video_alias": alias,
        "window_id": window_id,
        "rotation_angle_degrees": angle,
    }


def expect_raises(exception_types, message: str, callable_, *args, **kwargs) -> None:
    try:
        callable_(*args, **kwargs)
    except exception_types:
        return
    raise AssertionError(f"expected {exception_types} for {message}, but no exception was raised")


def test_subset_selection_is_the_head_of_the_list_in_order() -> None:
    paths = [Path(f"/data/video_{i}.h5") for i in range(10)]
    assert select_subset(paths, 4, label="train") == paths[:4]
    assert select_subset(paths, 1, label="val") == paths[:1]
    expect_raises(ValueError, "zero videos", select_subset, paths, 0, label="train")
    expect_raises(ValueError, "more videos than available", select_subset, paths[:2], 4, label="train")
    print("  ok: the subset is the first N entries in list order, and an oversized request is refused")


def test_healthy_angle_observations_produce_no_violation() -> None:
    records = [
        observation(1, "train_000", 3.5, window_id=w) for w in range(4)
    ] + [
        observation(1, "train_001", -7.25, window_id=w) for w in range(3)
    ] + [
        observation(2, "train_000", -11.0, window_id=w) for w in range(4)
    ] + [
        observation(2, "train_001", 9.5, window_id=w) for w in range(3)
    ]
    assert angle_violations(records, mode=MODE_RANDOM_Z_ROTATION, epochs=(1, 2), split="train") == []
    print("  ok: one angle per (video, epoch) that advances between epochs raises nothing")


def test_two_angles_in_one_epoch_is_a_violation() -> None:
    records = [observation(1, "train_000", 3.5), observation(1, "train_000", 4.0)]
    violations = angle_violations(records, mode=MODE_RANDOM_Z_ROTATION, epochs=(1,), split="train")
    assert len(violations) == 1 and "2 distinct angles in one epoch" in violations[0]
    print("  ok: two angles for one video within one epoch is reported (stop condition 4)")


def test_angle_frozen_across_epochs_is_a_violation() -> None:
    """The specific failure the shared-epoch design exists to prevent: workers
    never see the new epoch, so every epoch reuses one angle."""
    records = [observation(epoch, "train_000", 3.5) for epoch in (1, 2)]
    violations = angle_violations(records, mode=MODE_RANDOM_Z_ROTATION, epochs=(1, 2), split="train")
    assert len(violations) == 1 and "the epoch never reached the workers" in violations[0]
    print("  ok: an angle identical in every epoch is reported (stop condition 5)")


def test_missing_angle_while_enabled_and_angle_while_disabled_are_both_violations() -> None:
    enabled_but_none = angle_violations(
        [observation(1, "train_000", None)], mode=MODE_RANDOM_Z_ROTATION, epochs=(1,), split="train"
    )
    assert len(enabled_but_none) == 1 and "no angle was recorded" in enabled_but_none[0]

    disabled_but_angle = angle_violations(
        [observation(1, "train_000", 3.5)], mode=MODE_NONE, epochs=(1,), split="train"
    )
    assert len(disabled_but_angle) == 1 and "expected no angle" in disabled_but_angle[0]
    assert angle_violations([observation(1, "train_000", None)], mode=MODE_NONE, epochs=(1,), split="train") == []
    print("  ok: a missing angle while enabled and an angle while disabled are both caught")


def test_validation_split_must_never_carry_an_angle() -> None:
    """Even in the rotation arm, the validation dataset is unaugmented."""
    violations = angle_violations(
        [observation(1, "validation_000", 5.0)],
        mode=MODE_RANDOM_Z_ROTATION,
        epochs=(1,),
        split="validation",
    )
    assert len(violations) == 1 and "expected no angle" in violations[0]
    assert (
        angle_violations(
            [observation(1, "validation_000", None)],
            mode=MODE_RANDOM_Z_ROTATION,
            epochs=(1,),
            split="validation",
        )
        == []
    )
    print("  ok: a validation sample carrying an angle is reported (stop condition 6)")


def test_finite_violations_catch_nan_and_inf() -> None:
    assert finite_violations("loss", 0.25) == []
    assert len(finite_violations("loss", float("nan"))) == 1
    assert len(finite_violations("loss", float("inf"))) == 1
    assert len(finite_violations("loss", None)) == 1
    print("  ok: NaN, Inf and a missing value are all reported (stop condition 1)")


def make_sample(points: np.ndarray, *, angle: float | None = None, **overrides: Any) -> dict[str, Any]:
    num_points = points.shape[0]
    sample = {
        "points": torch.from_numpy(points.astype(np.float32)),
        "features": torch.rand(num_points, 2, generator=torch.Generator().manual_seed(1)),
        "labels": torch.zeros(num_points, dtype=torch.long),
        "valid_mask": torch.ones(num_points, dtype=torch.bool),
        "frame_order": torch.arange(num_points, dtype=torch.long),
        "point_indices": torch.arange(num_points, dtype=torch.long),
        "window_start": 0,
        "window_end": 15,
        "meta": {"h5_index": 0, "rotation_angle_degrees": angle},
    }
    sample.update(overrides)
    return sample


def test_correspondence_accepts_a_real_rotation_and_rejects_tampering() -> None:
    from stage5.utils.rotation_augmentation import apply_z_rotation

    rng = np.random.default_rng(3)
    points = rng.normal(size=(256, 3))
    reference = make_sample(points)
    rotated = make_sample(apply_z_rotation(points, 12.5), angle=12.5)
    rotated["features"] = reference["features"]
    assert correspondence_violations(reference, rotated, label="sample 0") == []

    scaled = make_sample(points * 1.05, angle=12.5)
    scaled["features"] = reference["features"]
    scaled_violations = correspondence_violations(reference, scaled, label="sample 0")
    assert any("pairwise distances drifted" in v for v in scaled_violations)

    relabeled = make_sample(apply_z_rotation(points, 12.5), angle=12.5)
    relabeled["features"] = reference["features"]
    relabeled["labels"] = torch.ones(256, dtype=torch.long)
    relabeled_violations = correspondence_violations(reference, relabeled, label="sample 0")
    assert any("labels changed" in v for v in relabeled_violations)
    print("  ok: a true rotation passes; a scale change and an altered label are both reported")


def test_none_mode_regression_check_requires_exact_points() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(64, 3))
    reference = make_sample(points)
    identical = make_sample(points)
    assert none_mode_regression_violations(reference, identical, label="sample 0") == []

    nudged_points = points.copy()
    nudged_points[0, 0] += 1e-5
    nudged = make_sample(nudged_points)
    violations = none_mode_regression_violations(reference, nudged, label="sample 0")
    assert any("differ from the unaugmented reference" in v for v in violations)

    with_angle = make_sample(points, angle=1.0)
    assert any("angle was recorded" in v for v in none_mode_regression_violations(reference, with_angle, label="s"))
    print("  ok: none mode demands bit-identical points and no recorded angle (stop condition 3)")


def test_privacy_self_check_covers_the_shareable_payload() -> None:
    clean = {"angles_by_epoch": {"1": {"train_000": 3.5}}, "status": "passed"}
    assert privacy_self_check(clean)["timestamp_like_video_ids_absent"]
    assert privacy_self_check(clean)["absolute_host_paths_absent"]
    leaky = {"train_video_ids": ["20260711_120001_3_pointcloud.h5"], "path": "/mnt/data/x"}
    assert not privacy_self_check(leaky)["timestamp_like_video_ids_absent"]
    assert not privacy_self_check(leaky)["absolute_host_paths_absent"]
    print("  ok: the shareable payload's privacy patterns are checked, and a leak would be caught")


def test_stage_c_data_path_runs_end_to_end_on_synthetic_h5() -> None:
    """Everything stage C does except the GPU model step, on real Datasets and
    a real DataLoader with workers."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        videos = build_fixture_videos(root, count=5)
        train_paths, val_paths = videos[:4], videos[4:]

        rotated = run_stage_c(
            train_paths=train_paths,
            val_paths=val_paths,
            mode=MODE_RANDOM_Z_ROTATION,
            epochs=2,
            num_workers=2,
            rotation_degrees=15.0,
            seed=42,
            checkpoint=None,
            device="cpu",
            gradient_accumulation_steps=8,
            skip_optimization=True,
        )
        assert rotated["violations"] == [], rotated["violations"]
        assert rotated["num_train_videos"] == 4 and rotated["num_val_videos"] == 1
        assert set(rotated["angles_by_epoch"]) == {"1", "2"}
        epoch_1 = rotated["angles_by_epoch"]["1"]
        epoch_2 = rotated["angles_by_epoch"]["2"]
        assert set(epoch_1) == {f"train_{i:03d}" for i in range(4)}
        assert all(angle is not None for angle in epoch_1.values())
        assert all(epoch_1[alias] != epoch_2[alias] for alias in epoch_1)
        assert len(set(epoch_1.values())) == 4, "each video gets its own angle within an epoch"

        plain = run_stage_c(
            train_paths=train_paths,
            val_paths=val_paths,
            mode=MODE_NONE,
            epochs=2,
            num_workers=2,
            rotation_degrees=15.0,
            seed=42,
            checkpoint=None,
            device="cpu",
            gradient_accumulation_steps=8,
            skip_optimization=True,
        )
        assert plain["violations"] == [], plain["violations"]
        assert all(angle is None for angles in plain["angles_by_epoch"].values() for angle in angles.values())
        assert plain["num_train_windows_per_epoch"] == rotated["num_train_windows_per_epoch"]
        windows = rotated["num_train_windows_per_epoch"]
    print(
        f"  ok: stage C data path runs over 4+1 synthetic videos ({windows} windows/epoch), "
        "no violations in either mode"
    )


def test_model_and_loss_interface_contract_matches_how_the_preflight_calls_them() -> None:
    """Pins the contract the preflight's GPU steps rely on.

    A first run on the GPU host failed here: the segmentor returns
    {"logits": ...}, not a bare tensor, and the loss is called as
    loss_fn(output, batch). Both are checked on CPU with mlp_baseline (which
    needs no CUDA ops) and the real loss, so the same mistake cannot recur
    unnoticed.
    """
    from stage5.models import build_stage5_model
    from stage5.training import build_loss

    num_points = 64
    model = build_stage5_model("mlp_baseline", num_classes=2, feature_dim=2)
    batch = {
        "points": torch.randn(1, num_points, 3),
        "features": torch.rand(1, num_points, 2),
        "labels": (torch.rand(1, num_points) < 0.2).long(),
        "valid_mask": (torch.rand(1, num_points) < 0.9),
    }

    output = model(batch)
    assert isinstance(output, dict), "the segmentor returns a dict, not a tensor"
    assert "logits" in output, f"expected a 'logits' key, got {sorted(output)}"
    assert output["logits"].shape == (1, num_points, 2), output["logits"].shape

    loss_fn = build_loss(
        "cross_entropy", ignore_index=-1, class_weight=[0.05963856, 1.94036150], label_smoothing=0.0
    )
    loss_dict = loss_fn(output, batch)
    assert isinstance(loss_dict, dict) and "loss" in loss_dict
    loss = loss_dict["loss"]
    assert torch.isfinite(loss)

    # Exactly the call the preflight makes, including the loss_sum fallback.
    loss_dict.get("loss_sum", loss).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients"
    assert all(torch.isfinite(g).all() for g in grads)
    print("  ok: segmentor returns {'logits': [B, N, C]}, loss_fn(output, batch) -> dict, backward gives finite grads")


def test_checker_refuses_to_write_under_stage5_runs_or_into_an_occupied_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        forbidden = root / "stage5_runs" / "260916" / "preflight"
        forbidden.mkdir(parents=True)
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--stage", "b", "--output_dir", str(forbidden)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, result
        assert "refusing to write under stage5_runs/" in result.stderr

        occupied = root / "already_used"
        occupied.mkdir()
        (occupied / "previous.json").write_text("{}", encoding="utf-8")
        second = subprocess.run(
            [sys.executable, str(CHECKER), "--stage", "b", "--output_dir", str(occupied)],
            capture_output=True,
            text=True,
        )
        assert second.returncode == 2, second
        assert "already holds artifacts" in second.stderr
    print("  ok: writing under stage5_runs/ or into a non-empty output directory is refused before any work")


def test_stage_c_cli_requires_the_file_lists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--stage", "c", "--output_dir", str(Path(tmp) / "out")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "stage c needs --train_list and --val_list" in result.stderr
    print("  ok: stage C refuses to run without both file lists")


def test_stage_c_cli_writes_both_outputs_and_keeps_video_ids_private() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        videos = build_fixture_videos(root / "data", count=3)
        train_list = root / "train.txt"
        val_list = root / "val.txt"
        train_list.write_text("\n".join(str(p) for p in videos[:2]) + "\n", encoding="utf-8")
        val_list.write_text(str(videos[2]) + "\n", encoding="utf-8")
        output_dir = root / "out"

        result = subprocess.run(
            [
                sys.executable, str(CHECKER),
                "--stage", "c",
                "--train_list", str(train_list),
                "--val_list", str(val_list),
                "--num_train_videos", "2",
                "--num_val_videos", "1",
                "--augmentation", "random_z_rotation",
                "--epochs", "2",
                "--num_workers", "0",
                "--device", "cpu",
                "--skip_optimization",
                "--output_dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        private = json.loads((output_dir / "preflight_private.json").read_text(encoding="utf-8"))
        shareable = json.loads((output_dir / "preflight_shareable.json").read_text(encoding="utf-8"))
        assert private["status"] == "passed" and shareable["status"] == "passed"
        assert "train_video_ids" in private, "the private record keeps the real file names"
        assert "train_video_ids" not in shareable, "the shareable record must not"
        assert shareable["privacy_self_check"]["timestamp_like_video_ids_absent"]
        assert shareable["privacy_self_check"]["absolute_host_paths_absent"]
        assert set(shareable["angles_by_epoch"]["1"]) == {"train_000", "train_001"}
    print("  ok: the CLI writes a private and a shareable record, and only the private one names videos")


def main() -> None:
    tests = [
        test_subset_selection_is_the_head_of_the_list_in_order,
        test_healthy_angle_observations_produce_no_violation,
        test_two_angles_in_one_epoch_is_a_violation,
        test_angle_frozen_across_epochs_is_a_violation,
        test_missing_angle_while_enabled_and_angle_while_disabled_are_both_violations,
        test_validation_split_must_never_carry_an_angle,
        test_finite_violations_catch_nan_and_inf,
        test_correspondence_accepts_a_real_rotation_and_rejects_tampering,
        test_none_mode_regression_check_requires_exact_points,
        test_privacy_self_check_covers_the_shareable_payload,
        test_stage_c_data_path_runs_end_to_end_on_synthetic_h5,
        test_model_and_loss_interface_contract_matches_how_the_preflight_calls_them,
        test_checker_refuses_to_write_under_stage5_runs_or_into_an_occupied_directory,
        test_stage_c_cli_requires_the_file_lists,
        test_stage_c_cli_writes_both_outputs_and_keeps_video_ids_private,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 S5-15 GPU preflight CPU-side tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
