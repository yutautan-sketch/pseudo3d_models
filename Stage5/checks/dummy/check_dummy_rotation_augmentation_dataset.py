from __future__ import annotations

import multiprocessing
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.datasets import Pseudo3DPointCloudDataset, pad_point_window_collate  # noqa: E402
from stage5.utils.feature_normalization import normalize_xyz  # noqa: E402
from stage5.utils.rotation_augmentation import (  # noqa: E402
    MODE_NONE,
    MODE_RANDOM_Z_ROTATION,
    apply_z_rotation,
    derive_rotation_angle_degrees,
)

# ----------------------------------------------------------------------------
# S5-15 P2 Step 2: the Dataset side of the training-only Z-rotation.
#
# Covers the epoch/worker propagation contract that report section 8.3-1 made
# mandatory: real torch DataLoaders over synthetic H5 fixtures, across
# num_workers=0, non-persistent workers and persistent workers, over several
# epochs, checking that one video gets exactly one angle per epoch and that
# set_epoch() reaches already-started persistent workers. CPU only -- no CUDA,
# no real data, no training.
# ----------------------------------------------------------------------------

TRAIN_SEED = 42
EXPECTED_BASE_SEED = TRAIN_SEED + 500000
WINDOW_KWARGS: dict[str, Any] = {
    "window_mode": "overlap",
    "window_size_frames": 4,
    "window_stride_frames": 2,
    "include_tail_window": True,
}


def write_fixture_h5(path: Path, *, num_frames: int = 12, points_per_frame: int = 10, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    num_points = num_frames * points_per_frame
    frame_order = np.repeat(np.arange(num_frames, dtype=np.int64), points_per_frame)
    points = rng.normal(loc=3.0, scale=2.0, size=(num_points, 3)).astype(np.float32)
    labels = (rng.random(num_points) < 0.2).astype(np.int64)
    valid_mask = rng.random(num_points) < 0.95

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        group = f.create_group("point_cloud")
        group.create_dataset("points", data=points)
        group.create_dataset("intensity", data=rng.integers(0, 256, num_points, dtype=np.uint8))
        group.create_dataset("alpha", data=rng.integers(0, 256, num_points, dtype=np.uint8))
        group.create_dataset("confidence", data=rng.random(num_points).astype(np.float32))
        group.create_dataset("frame_order", data=frame_order)
        group.create_dataset("pixel_xy", data=rng.random((num_points, 2)).astype(np.float32) * 255.0)
        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=labels)
        annotation.create_dataset("valid_mask", data=valid_mask)
    return path


def build_fixture_videos(root: Path, *, count: int = 2) -> list[Path]:
    return [
        write_fixture_h5(root / f"fixture_video_{index}.h5", seed=100 + index, num_frames=12)
        for index in range(count)
    ]


def make_dataset(paths: list[Path], *, mode: str = MODE_NONE, **overrides: Any) -> Pseudo3DPointCloudDataset:
    kwargs: dict[str, Any] = {
        "num_points": None,
        "features": "intensity,confidence",
        "seed": TRAIN_SEED,
        "augmentation_mode": mode,
        **WINDOW_KWARGS,
    }
    kwargs.update(overrides)
    return Pseudo3DPointCloudDataset(paths, **kwargs)


def expected_angle(video_id: str, epoch: int) -> float:
    return derive_rotation_angle_degrees(
        base_seed=EXPECTED_BASE_SEED, epoch=epoch, video_id=video_id, max_abs_degrees=15.0
    )


def expect_raises(exception_types, message: str, callable_, *args, **kwargs) -> None:
    try:
        callable_(*args, **kwargs)
    except exception_types:
        return
    raise AssertionError(f"expected {exception_types} for {message}, but no exception was raised")


def angles_by_video(dataset: Pseudo3DPointCloudDataset) -> dict[int, set[float]]:
    grouped: dict[int, set[float]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        h5_index = int(sample["meta"]["h5_index"])
        grouped.setdefault(h5_index, set()).add(sample["meta"]["rotation_angle_degrees"])
    return grouped


def test_none_mode_output_is_bitwise_identical_to_the_pre_augmentation_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=1)
        dataset = make_dataset(paths, mode=MODE_NONE)
        with h5py.File(paths[0], "r") as f:
            raw_points = f["point_cloud/points"][:].astype(np.float32)
        reference = normalize_xyz(raw_points)

        for index in range(len(dataset)):
            sample = dataset[index]
            indices = sample["point_indices"].numpy()
            assert np.array_equal(sample["points"].numpy(), reference[indices]), index
            assert sample["meta"]["rotation_angle_degrees"] is None
    print(f"  ok: none mode reproduces normalize_xyz()+window slicing bit for bit, angle recorded as None")


def test_none_mode_consumes_no_randomness_and_leaves_source_points_untouched() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=1)
        dataset = make_dataset(paths, mode=MODE_NONE, cache_data=True)
        np.random.seed(4321)
        before = np.random.get_state()
        for index in range(len(dataset)):
            dataset[index]
        after = np.random.get_state()
        assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]

        with h5py.File(paths[0], "r") as f:
            on_disk = f["point_cloud/points"][:].astype(np.float32)
        assert np.array_equal(dataset._cache[0]["points"], on_disk)
    print("  ok: none mode leaves numpy's global RNG and the cached source points untouched")


def test_enabled_mode_applies_the_expected_angle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=1)
        dataset = make_dataset(paths, mode=MODE_RANDOM_Z_ROTATION)
        dataset.set_epoch(1)
        assert dataset.augmentation.base_seed == EXPECTED_BASE_SEED

        with h5py.File(paths[0], "r") as f:
            raw_points = f["point_cloud/points"][:].astype(np.float32)
        angle = expected_angle(paths[0].name, 1)
        reference = apply_z_rotation(normalize_xyz(raw_points), angle)

        sample = dataset[0]
        indices = sample["point_indices"].numpy()
        assert sample["meta"]["rotation_angle_degrees"] == angle
        assert np.array_equal(sample["points"].numpy(), reference[indices])
    print(f"  ok: enabled mode rotates the normalized video by its derived angle ({angle:.6f} degrees)")


def test_all_windows_of_a_video_share_one_angle_within_an_epoch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=2)
        dataset = make_dataset(paths, mode=MODE_RANDOM_Z_ROTATION)
        dataset.set_epoch(2)
        grouped = angles_by_video(dataset)

        assert len(grouped) == 2, grouped
        for h5_index, angles in grouped.items():
            assert len(angles) == 1, f"video {h5_index} saw {len(angles)} angles in one epoch: {angles}"
            assert next(iter(angles)) == expected_angle(paths[h5_index].name, 2)
        # Two different videos must not share one angle in the same epoch.
        video_angles = {expected_angle(path.name, 2) for path in paths}
        assert len(video_angles) == 2, video_angles
        window_count = len(dataset)
    print(f"  ok: {window_count} windows across 2 videos -> exactly 1 angle per video, differing between videos")


def test_set_epoch_changes_the_angle_without_accumulating_rotations() -> None:
    """cache_data=True keeps the source points in memory across epochs; each
    epoch must rotate the original points, never the previous epoch's output."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=1)
        dataset = make_dataset(paths, mode=MODE_RANDOM_Z_ROTATION, cache_data=True)
        with h5py.File(paths[0], "r") as f:
            normalized = normalize_xyz(f["point_cloud/points"][:].astype(np.float32))

        seen = []
        for epoch in (1, 2, 3, 1):
            dataset.set_epoch(epoch)
            sample = dataset[0]
            indices = sample["point_indices"].numpy()
            angle = expected_angle(paths[0].name, epoch)
            assert sample["meta"]["rotation_angle_degrees"] == angle
            assert np.array_equal(sample["points"].numpy(), apply_z_rotation(normalized, angle)[indices])
            seen.append(angle)

        assert len(set(seen[:3])) == 3, seen[:3]
        # Revisiting epoch 1 reproduces epoch 1 exactly: no drift accumulated.
        assert seen[3] == seen[0]
    print(f"  ok: epochs 1/2/3 give 3 distinct angles, and returning to epoch 1 reproduces it exactly")


def test_non_xyz_fields_are_unchanged_by_rotation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=1)
        plain = make_dataset(paths, mode=MODE_NONE)
        rotated = make_dataset(paths, mode=MODE_RANDOM_Z_ROTATION)
        rotated.set_epoch(3)

        for index in range(len(plain)):
            left, right = plain[index], rotated[index]
            for key in ("features", "labels", "valid_mask", "frame_order", "point_indices"):
                assert torch.equal(left[key], right[key]), f"{key} changed at sample {index}"
            assert left["window_start"] == right["window_start"]
            assert left["window_end"] == right["window_end"]
            assert left["meta"]["num_source_points"] == right["meta"]["num_source_points"]
            assert not torch.equal(left["points"], right["points"]), "rotation should move the XYZ"
    print("  ok: features/labels/valid_mask/frame_order/point_indices/window bounds identical; only XYZ moves")


def test_duplicate_file_names_are_rejected_only_when_augmentation_is_enabled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = write_fixture_h5(root / "mount_a" / "same_name.h5", seed=1)
        second = write_fixture_h5(root / "mount_b" / "same_name.h5", seed=2)
        expect_raises(
            ValueError,
            "duplicate file names with augmentation on",
            make_dataset,
            [first, second],
            mode=MODE_RANDOM_Z_ROTATION,
        )
        # Ordinary runs are unaffected: the collision only matters for angles.
        dataset = make_dataset([first, second], mode=MODE_NONE)
        assert len(dataset) > 0
    print("  ok: duplicate basenames raise with augmentation on, and are still allowed with it off")


def test_augmentation_requires_normalized_points_and_a_seed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=1)
        expect_raises(
            ValueError,
            "augmentation without normalize_points",
            make_dataset,
            paths,
            mode=MODE_RANDOM_Z_ROTATION,
            normalize_points=False,
        )
        expect_raises(
            ValueError,
            "augmentation without any seed",
            make_dataset,
            paths,
            mode=MODE_RANDOM_Z_ROTATION,
            seed=None,
        )
        # An explicit augmentation seed is enough even with no dataset seed.
        dataset = make_dataset(paths, mode=MODE_RANDOM_Z_ROTATION, seed=None, augmentation_seed=7)
        assert dataset.augmentation.base_seed == 7
        # none mode needs no seed at all.
        assert make_dataset(paths, mode=MODE_NONE, seed=None).augmentation.enabled is False
    print("  ok: augmentation demands normalized points and a seed; none mode demands neither")


def test_set_epoch_validation_and_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=1)
        dataset = make_dataset(paths, mode=MODE_RANDOM_Z_ROTATION)
        assert dataset.epoch == 0, "default epoch before any set_epoch() call"
        dataset.set_epoch(5)
        assert dataset.epoch == 5
        dataset.set_epoch(0)
        assert dataset.epoch == 0
        expect_raises(TypeError, "bool epoch", dataset.set_epoch, True)
        expect_raises(TypeError, "float epoch", dataset.set_epoch, 2.0)
        expect_raises(ValueError, "negative epoch", dataset.set_epoch, -1)
        expect_raises(ValueError, "C-int overflow epoch", dataset.set_epoch, 2**31)
    print("  ok: epoch defaults to 0, round-trips, and rejects bool/float/negative/oversized values")


def collect_loader_angles(
    dataset: Pseudo3DPointCloudDataset,
    *,
    num_workers: int,
    persistent_workers: bool,
    epochs: tuple[int, ...],
    multiprocessing_context: str | None = None,
) -> dict[int, dict[int, set[float]]]:
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_point_window_collate,
        persistent_workers=persistent_workers,
        multiprocessing_context=multiprocessing_context,
    )
    observed: dict[int, dict[int, set[float]]] = {}
    for epoch in epochs:
        dataset.set_epoch(epoch)
        per_video: dict[int, set[float]] = {}
        for batch in loader:
            for meta in batch["meta"]:
                per_video.setdefault(int(meta["h5_index"]), set()).add(meta["rotation_angle_degrees"])
        observed[epoch] = per_video
    del loader
    return observed


def test_dataloader_worker_patterns_all_agree_on_angles() -> None:
    """Report 8.3-1, made mandatory: the angle a window sees must not depend on
    worker count or on whether workers persist across epochs. Persistent
    workers are the case a plain attribute would silently break, since they are
    forked once and never see a later parent-side assignment."""
    start_method = multiprocessing.get_context().get_start_method()
    epochs = (1, 2, 3)
    # The last entry pins the start method explicitly: the shared epoch value
    # is deliberately lock-free so it also crosses into spawn-context workers,
    # rather than relying on this host happening to default to fork.
    configurations = [
        ("num_workers=0", 0, False, None),
        ("num_workers=2 persistent=False", 2, False, None),
        ("num_workers=2 persistent=True", 2, True, None),
        ("num_workers=2 persistent=True spawn", 2, True, "spawn"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=2)
        results: dict[str, dict[int, dict[int, set[float]]]] = {}
        for label, num_workers, persistent, context in configurations:
            dataset = make_dataset(paths, mode=MODE_RANDOM_Z_ROTATION)
            results[label] = collect_loader_angles(
                dataset,
                num_workers=num_workers,
                persistent_workers=persistent,
                epochs=epochs,
                multiprocessing_context=context,
            )

        for label, observed in results.items():
            for epoch in epochs:
                per_video = observed[epoch]
                assert set(per_video) == {0, 1}, f"{label} epoch {epoch}: videos {sorted(per_video)}"
                for h5_index, angles in per_video.items():
                    assert len(angles) == 1, f"{label} epoch {epoch} video {h5_index}: {angles}"
                    actual = next(iter(angles))
                    expected = expected_angle(paths[h5_index].name, epoch)
                    assert actual == expected, f"{label} epoch {epoch} video {h5_index}: {actual} != {expected}"

        # Across epochs the angles actually moved, so a persistent worker that
        # never received set_epoch() would have failed the check above.
        first = results["num_workers=2 persistent=True"]
        per_epoch_angles = [sorted(next(iter(first[epoch].values()))) for epoch in epochs]
        assert len({tuple(a) for a in per_epoch_angles}) == 3, per_epoch_angles

    print(
        f"  ok: {len(configurations)} loader configurations (incl. an explicit spawn context) "
        f"x 3 epochs x 2 videos agree exactly (host default start method: {start_method!r})"
    )


def test_none_mode_through_a_real_dataloader_records_no_angle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=2)
        dataset = make_dataset(paths, mode=MODE_NONE)
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=pad_point_window_collate
        )
        for epoch in (1, 2):
            dataset.set_epoch(epoch)
            for batch in loader:
                for meta in batch["meta"]:
                    assert meta["rotation_angle_degrees"] is None
        del loader
    print("  ok: none mode through 2 workers reports no angle on any sample, across epochs")


def test_train_cli_passes_augmentation_to_the_train_dataset_only() -> None:
    """Report 8.3-5: the train CLI's own validation dataset must be unaugmented,
    not merely the standalone evaluate_stage5.py path."""
    import argparse
    import inspect

    import train_stage5

    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture_videos(Path(tmp), count=2)
        args = argparse.Namespace(
            num_points=0,
            features="intensity,confidence",
            no_normalize_points=False,
            sampling_mode="video",
            exclude_ignore_in_sampling=False,
            positive_oversample_ratio=0.0,
            frame_window_size=None,
            label_policy="bbox_noncontour_ignore",
            seed=TRAIN_SEED,
            cache_data=False,
            augmentation=MODE_RANDOM_Z_ROTATION,
            augmentation_rotation_degrees=15.0,
            augmentation_seed=None,
            **WINDOW_KWARGS,
        )

        train_dataset = train_stage5.make_dataset(
            paths[:1], args=args, seed=args.seed, cache_data=False, augmentation_mode=args.augmentation
        )
        # Mirrors main(): augmentation_mode is simply not forwarded for val.
        val_dataset = train_stage5.make_dataset(
            paths[1:], args=args, seed=args.seed + 100000, cache_data=False
        )
        assert train_dataset.augmentation.enabled is True
        assert train_dataset.augmentation.base_seed == EXPECTED_BASE_SEED
        assert val_dataset.augmentation.enabled is False
        assert val_dataset[0]["meta"]["rotation_angle_degrees"] is None

        config = train_stage5.build_config(
            args,
            feature_dim=2,
            class_weight_info={"mode": "manual", "resolved": [0.05963856, 1.9403615]},
            augmentation=train_dataset.augmentation,
        )
        record = config["augmentation_info"]
        assert record["mode"] == MODE_RANDOM_Z_ROTATION
        assert record["base_seed"] == EXPECTED_BASE_SEED, "the resolved seed must be recorded, not None"
        assert record["seed_source"] == "train_seed_plus_500000"
        assert config["augmentation_seed"] is None, "the raw CLI value stays None alongside the resolved one"

        # Guards the specific failure 8.3-1 names: without this call every epoch
        # would reuse the default epoch 0 and so a single fixed angle.
        source = inspect.getsource(train_stage5.train)
        assert "train_dataset.set_epoch(epoch)" in source, "the epoch loop must publish the epoch"
        loop_index = source.index("for epoch in range(1, args.epochs + 1):")
        set_epoch_index = source.index("train_dataset.set_epoch(epoch)", loop_index)
        run_epoch_index = source.index("run_one_epoch(", set_epoch_index)
        assert set_epoch_index < run_epoch_index, "set_epoch() must run before the epoch's loader iteration"
    print("  ok: train dataset augmented, val dataset never; resolved seed recorded; set_epoch precedes the epoch")


def main() -> None:
    tests = [
        test_none_mode_output_is_bitwise_identical_to_the_pre_augmentation_path,
        test_none_mode_consumes_no_randomness_and_leaves_source_points_untouched,
        test_enabled_mode_applies_the_expected_angle,
        test_all_windows_of_a_video_share_one_angle_within_an_epoch,
        test_set_epoch_changes_the_angle_without_accumulating_rotations,
        test_non_xyz_fields_are_unchanged_by_rotation,
        test_duplicate_file_names_are_rejected_only_when_augmentation_is_enabled,
        test_augmentation_requires_normalized_points_and_a_seed,
        test_set_epoch_validation_and_default,
        test_dataloader_worker_patterns_all_agree_on_angles,
        test_none_mode_through_a_real_dataloader_records_no_angle,
        test_train_cli_passes_augmentation_to_the_train_dataset_only,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 S5-15 rotation augmentation dataset tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
