from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.rotation_augmentation import (  # noqa: E402
    ANGLE_DERIVATION_DESCRIPTION,
    AUGMENTATION_SEED_OFFSET,
    DEFAULT_MAX_ABS_DEGREES,
    MODE_NONE,
    MODE_RANDOM_Z_ROTATION,
    SEED_SOURCE_EXPLICIT,
    SEED_SOURCE_TRAIN_SEED_OFFSET,
    AugmentationConfig,
    apply_z_rotation,
    derive_rotation_angle_degrees,
    rotate_video_points,
    rotation_angle_key,
    rotation_matrix_z,
    stable_video_id_from_path,
)

# ----------------------------------------------------------------------------
# S5-15 P2 Step 1: synthetic coverage for the training-only Z-rotation helper.
# Pure numpy, no torch/CUDA/H5 -- every function under test is deterministic
# and CPU-only. The DataLoader/worker-level checks live in Step 2's test.
# ----------------------------------------------------------------------------

# Values recorded from this implementation (report section 8.3-5: verify the
# actual generated numbers rather than asserting a general "different inputs
# always differ" guarantee). A change to the key format, digest byte order or
# generator would break these.
FIXTURE_BASE_SEED = 500042
FIXTURE_ANGLES = {
    ("video_a.h5", 1): -1.7092337279270033,
    ("video_a.h5", 2): 12.698510851035216,
    ("video_a.h5", 3): -4.853117984718477,
    ("video_b.h5", 1): -1.111196395415817,
    ("video_b.h5", 2): -10.864198968684521,
    ("video_b.h5", 3): 10.067628023210851,
}


def fixed_point_cloud(*, num_points: int = 512, seed: int = 20260916) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_points, 3)).astype(np.float32)


def expect_raises(exception_types, message: str, callable_, *args, **kwargs) -> None:
    try:
        callable_(*args, **kwargs)
    except exception_types:
        return
    raise AssertionError(f"expected {exception_types} for {message}, but no exception was raised")


def test_rotation_matrix_is_orthogonal_with_determinant_one() -> None:
    for degrees in (-15.0, -0.5, 0.0, 15.0, 90.0, 180.0):
        matrix = rotation_matrix_z(degrees)
        assert matrix.shape == (3, 3) and matrix.dtype == np.float64
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12)
    assert np.allclose(rotation_matrix_z(0.0), np.eye(3), atol=1e-15)
    # +90 degrees is CCW about +Z: the x axis maps onto the y axis.
    assert np.allclose(rotation_matrix_z(90.0) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    # A rotation by -theta is the inverse (transpose) of +theta.
    assert np.allclose(rotation_matrix_z(-15.0), rotation_matrix_z(15.0).T, atol=1e-15)
    print("  ok: Z rotation matrix is orthogonal, det=1, CCW about +Z, and -theta is its inverse")


def test_rotation_matrix_matches_the_s5_14_diagnosis_convention() -> None:
    """The +-15 degree range inherits its meaning from the S5-14 supplement-2
    coordinate diagnosis, so the matrix convention must be identical. The
    checker is imported here only for comparison and is never modified."""
    checker_dir = REPO_ROOT / "checks" / "real_h5"
    sys.path.insert(0, str(checker_dir))
    from check_stage5_coordinate_transform_diagnostics import (  # noqa: E402
        rotation_matrix_z as diagnosis_rotation_matrix_z,
    )

    for degrees in (-15.0, 15.0, 37.5):
        ours = rotation_matrix_z(degrees)
        theirs = diagnosis_rotation_matrix_z(degrees)
        assert np.array_equal(ours, theirs), f"{degrees} degrees differs from the S5-14 convention"
    print("  ok: rotation matrices are bit-identical to the S5-14 diagnosis implementation")


def test_rotation_preserves_distances_and_z() -> None:
    points = fixed_point_cloud()
    rotated = apply_z_rotation(points, 15.0)
    assert rotated.shape == points.shape and rotated.dtype == np.float32

    # Pairwise distances over fixed sampled pairs, not a full NxN matrix.
    rng = np.random.default_rng(7)
    left = rng.integers(0, points.shape[0], size=256)
    right = rng.integers(0, points.shape[0], size=256)
    before = np.linalg.norm(points[left].astype(np.float64) - points[right].astype(np.float64), axis=1)
    after = np.linalg.norm(rotated[left].astype(np.float64) - rotated[right].astype(np.float64), axis=1)
    assert np.allclose(before, after, atol=1e-5), f"max distance drift {np.abs(before - after).max()}"

    # Norms from the origin (= the video centroid after normalize_xyz) and the
    # z coordinate are untouched by a rotation about +Z.
    assert np.allclose(
        np.linalg.norm(points.astype(np.float64), axis=1),
        np.linalg.norm(rotated.astype(np.float64), axis=1),
        atol=1e-5,
    )
    assert np.allclose(points[:, 2], rotated[:, 2], atol=1e-6)
    print("  ok: pairwise distances, origin norms and z are preserved (float32 tolerance 1e-5)")


def test_rotation_round_trip_returns_original() -> None:
    points = fixed_point_cloud()
    restored = apply_z_rotation(apply_z_rotation(points, 15.0), -15.0)
    assert np.allclose(points, restored, atol=1e-5), f"max drift {np.abs(points - restored).max()}"
    print("  ok: rotating by +15 then -15 degrees returns the original points")


def test_apply_z_rotation_never_writes_into_the_input() -> None:
    for dtype in (np.float32, np.float64):
        points = fixed_point_cloud().astype(dtype)
        snapshot = points.copy()
        rotated = apply_z_rotation(points, 15.0)
        assert np.array_equal(points, snapshot), f"input was modified in place for dtype {dtype}"
        assert rotated is not points
    empty = np.zeros((0, 3), dtype=np.float32)
    assert apply_z_rotation(empty, 15.0).shape == (0, 3)
    print("  ok: input array is never modified in place (float32/float64), empty [0, 3] handled")


def test_angle_is_within_range_for_many_combinations() -> None:
    for max_abs_degrees in (15.0, 1.0, 45.0):
        for base_seed in (0, 42, 500042):
            for epoch in range(0, 12):
                for video_id in ("a.h5", "b.h5", "long_name_video_000.h5"):
                    angle = derive_rotation_angle_degrees(
                        base_seed=base_seed,
                        epoch=epoch,
                        video_id=video_id,
                        max_abs_degrees=max_abs_degrees,
                    )
                    assert -max_abs_degrees <= angle <= max_abs_degrees, angle
    print("  ok: 324 combinations all fall inside [-max_abs_degrees, +max_abs_degrees]")


def test_angle_is_deterministic_for_the_same_triple() -> None:
    """Every window of a video within one epoch calls this independently --
    possibly from different worker processes -- and must get one angle."""
    angles = {
        derive_rotation_angle_degrees(base_seed=500042, epoch=3, video_id="video_a.h5")
        for _ in range(50)
    }
    assert len(angles) == 1, angles
    print(f"  ok: 50 repeated derivations of one (seed, epoch, video) give a single angle {angles.pop():.6f}")


def test_angle_changes_with_epoch_video_and_seed() -> None:
    base = dict(base_seed=500042, epoch=1, video_id="video_a.h5")
    reference = derive_rotation_angle_degrees(**base)
    # A fixed angle across epochs is the specific bug the epoch plumbing risks.
    per_epoch = [derive_rotation_angle_degrees(**{**base, "epoch": e}) for e in range(1, 6)]
    assert len(set(per_epoch)) == 5, per_epoch
    assert derive_rotation_angle_degrees(**{**base, "video_id": "video_b.h5"}) != reference
    assert derive_rotation_angle_degrees(**{**base, "base_seed": 500043}) != reference
    print("  ok: angle varies across epochs (5 distinct), videos and base seeds on these fixtures")


def test_angle_matches_the_recorded_fixture_values() -> None:
    for (video_id, epoch), expected in FIXTURE_ANGLES.items():
        actual = derive_rotation_angle_degrees(
            base_seed=FIXTURE_BASE_SEED,
            epoch=epoch,
            video_id=video_id,
            max_abs_degrees=DEFAULT_MAX_ABS_DEGREES,
        )
        assert actual == expected, f"({video_id}, {epoch}): {actual!r} != recorded {expected!r}"
    print(f"  ok: all {len(FIXTURE_ANGLES)} recorded fixture angles reproduce exactly")


def test_angle_is_independent_of_pythonhashseed() -> None:
    """The reason the derivation uses hashlib instead of Python's hash():
    PYTHONHASHSEED differs per process, and DataLoader workers are processes."""
    script = (
        "import sys; sys.path.insert(0, %r); "
        "from stage5.utils.rotation_augmentation import derive_rotation_angle_degrees as d; "
        "print(repr(d(base_seed=500042, epoch=2, video_id='video_a.h5')))" % str(REPO_ROOT)
    )
    values = []
    for hash_seed in ("0", "12345", "random"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        values.append(result.stdout.strip())
    assert len(set(values)) == 1, f"angle depends on PYTHONHASHSEED: {values}"
    assert float(values[0]) == FIXTURE_ANGLES[("video_a.h5", 2)]
    print(f"  ok: identical angle under PYTHONHASHSEED 0/12345/random ({values[0]})")


def test_angle_derivation_does_not_touch_the_global_numpy_rng() -> None:
    """RNG separation (report 4.4): shuffling and model init draw from numpy's
    global/legacy RNG, which the angle derivation must not consume."""
    np.random.seed(1234)
    before = np.random.get_state()
    for epoch in range(1, 20):
        derive_rotation_angle_degrees(base_seed=500042, epoch=epoch, video_id="video_a.h5")
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    print("  ok: 19 angle derivations leave numpy's global RNG state untouched")


def test_key_is_injective_for_ids_containing_the_separator() -> None:
    assert rotation_angle_key(base_seed=1, epoch=2, video_id="3|4") == "1|2|3|4"
    # Only video_id can contain '|', so no other triple produces the same key.
    distinct = {
        rotation_angle_key(base_seed=1, epoch=2, video_id="3|4"),
        rotation_angle_key(base_seed=1, epoch=2, video_id="3"),
        rotation_angle_key(base_seed=12, epoch=3, video_id="4"),
    }
    assert len(distinct) == 3, distinct
    angles = {
        derive_rotation_angle_degrees(base_seed=1, epoch=2, video_id="3|4"),
        derive_rotation_angle_degrees(base_seed=1, epoch=2, video_id="3"),
        derive_rotation_angle_degrees(base_seed=12, epoch=3, video_id="4"),
    }
    assert len(angles) == 3, angles
    print("  ok: keys and angles stay distinct even when a video id contains the '|' separator")


def test_stable_video_id_ignores_mount_prefix_and_relative_paths() -> None:
    """Report 8.3-2: the same data under a different mount prefix, or given as
    a relative path, must not change the angle."""
    name = "20260711_120001_3_pointcloud_annotated_bboxrank_v7.h5"
    variants = [
        f"/mnt/data/3d_projects/pseudo3d/annotated/{name}",
        f"/other/mount/point/annotated/{name}",
        f"./{name}",
        name,
    ]
    ids = {stable_video_id_from_path(path) for path in variants}
    assert ids == {name}, ids
    angles = {
        derive_rotation_angle_degrees(base_seed=500042, epoch=1, video_id=stable_video_id_from_path(path))
        for path in variants
    }
    assert len(angles) == 1, angles
    assert stable_video_id_from_path(Path(variants[0])) == name
    print("  ok: four path spellings of one video give one id and one angle")


def test_none_mode_derives_no_angle_and_returns_points_unchanged() -> None:
    config = AugmentationConfig.resolve(mode=MODE_NONE, train_seed=42)
    assert config.enabled is False
    expect_raises(RuntimeError, "angle_degrees_for() in none mode", config.angle_degrees_for, video_id="a.h5", epoch=1)

    points = fixed_point_cloud()
    snapshot = points.copy()
    np.random.seed(99)
    rng_before = np.random.get_state()
    rotated, angle = rotate_video_points(points, config=config, video_id="a.h5", epoch=1)
    rng_after = np.random.get_state()

    assert angle is None
    assert rotated is points, "none mode must return the very same array, not a copy"
    assert np.array_equal(points, snapshot)
    assert rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1])
    print("  ok: none mode returns the input array itself, derives no angle, consumes no randomness")


def test_enabled_mode_rotates_and_reports_its_angle() -> None:
    config = AugmentationConfig.resolve(mode=MODE_RANDOM_Z_ROTATION, train_seed=42)
    assert config.enabled is True
    points = fixed_point_cloud()
    rotated, angle = rotate_video_points(points, config=config, video_id="video_a.h5", epoch=2)
    assert angle == FIXTURE_ANGLES[("video_a.h5", 2)]
    assert np.allclose(rotated, apply_z_rotation(points, angle))
    assert not np.allclose(rotated, points), "a 12.7 degree rotation must actually move the points"
    print(f"  ok: enabled mode rotates by its reported angle ({angle:.6f} degrees)")


def test_rotate_video_points_gives_every_window_of_a_video_the_same_angle() -> None:
    """The S5-15 contract: rotate the whole video once, before window
    extraction, so slicing windows afterwards cannot introduce per-window
    angles or break the correspondence of overlapping points."""
    config = AugmentationConfig.resolve(mode=MODE_RANDOM_Z_ROTATION, train_seed=42)
    points = fixed_point_cloud(num_points=300)
    rotated_video, angle = rotate_video_points(points, config=config, video_id="video_a.h5", epoch=3)

    windows = [slice(0, 128), slice(64, 192), slice(128, 256), slice(192, 300)]
    for window in windows:
        from_video = rotated_video[window]
        from_window = apply_z_rotation(points[window], angle)
        assert np.array_equal(from_video, from_window), "window slice disagrees with the whole-video rotation"

    # Points shared by two overlapping windows stay identical.
    assert np.array_equal(rotated_video[64:128], rotated_video[windows[1]][:64])
    print(f"  ok: 4 overlapping windows share one angle ({angle:.6f}) and agree on overlapping points")


def test_resolve_prefers_explicit_seed_and_records_the_rule() -> None:
    derived = AugmentationConfig.resolve(mode=MODE_RANDOM_Z_ROTATION, train_seed=42)
    assert derived.base_seed == 42 + AUGMENTATION_SEED_OFFSET == 500042
    assert derived.seed_source == SEED_SOURCE_TRAIN_SEED_OFFSET

    explicit = AugmentationConfig.resolve(mode=MODE_RANDOM_Z_ROTATION, train_seed=42, augmentation_seed=7)
    assert explicit.base_seed == 7 and explicit.seed_source == SEED_SOURCE_EXPLICIT

    # An explicit 0 is a real value, not "unspecified".
    zero = AugmentationConfig.resolve(mode=MODE_RANDOM_Z_ROTATION, train_seed=42, augmentation_seed=0)
    assert zero.base_seed == 0 and zero.seed_source == SEED_SOURCE_EXPLICIT
    print("  ok: explicit seed wins (including 0); otherwise train_seed + 500000 with the rule recorded")


def test_to_config_dict_never_stores_a_null_seed() -> None:
    for mode in (MODE_NONE, MODE_RANDOM_Z_ROTATION):
        record = AugmentationConfig.resolve(mode=mode, train_seed=42).to_config_dict()
        assert record["mode"] == mode
        assert isinstance(record["base_seed"], int) and record["base_seed"] == 500042
        assert record["seed_source"] == SEED_SOURCE_TRAIN_SEED_OFFSET
        assert record["max_abs_degrees"] == DEFAULT_MAX_ABS_DEGREES
        assert record["angle_derivation"] == ANGLE_DERIVATION_DESCRIPTION
        assert None not in record.values()
    print("  ok: config record carries a concrete seed and the derivation rule for both modes")


def test_validation_rejects_bad_inputs() -> None:
    good = dict(base_seed=500042, epoch=1, video_id="a.h5")
    expect_raises(TypeError, "bool epoch", derive_rotation_angle_degrees, **{**good, "epoch": True})
    expect_raises(TypeError, "bool base_seed", derive_rotation_angle_degrees, **{**good, "base_seed": False})
    expect_raises(TypeError, "float epoch", derive_rotation_angle_degrees, **{**good, "epoch": 1.0})
    expect_raises(TypeError, "str base_seed", derive_rotation_angle_degrees, **{**good, "base_seed": "42"})
    expect_raises(ValueError, "negative epoch", derive_rotation_angle_degrees, **{**good, "epoch": -1})
    expect_raises(ValueError, "oversized seed", derive_rotation_angle_degrees, **{**good, "base_seed": 2**63})
    expect_raises(ValueError, "empty video id", derive_rotation_angle_degrees, **{**good, "video_id": ""})
    expect_raises(TypeError, "non-str video id", derive_rotation_angle_degrees, **{**good, "video_id": 3})
    expect_raises(
        ValueError, "zero degrees", derive_rotation_angle_degrees, **{**good, "max_abs_degrees": 0.0}
    )
    expect_raises(
        ValueError, "negative degrees", derive_rotation_angle_degrees, **{**good, "max_abs_degrees": -1.0}
    )
    expect_raises(
        ValueError, "nan degrees", derive_rotation_angle_degrees, **{**good, "max_abs_degrees": float("nan")}
    )
    expect_raises(ValueError, "unknown mode", AugmentationConfig, mode="random_xyz_rotation")
    expect_raises(ValueError, "[N, 2] points", apply_z_rotation, np.zeros((5, 2), dtype=np.float32), 15.0)
    expect_raises(ValueError, "1-D points", apply_z_rotation, np.zeros(9, dtype=np.float32), 15.0)
    print("  ok: 14 invalid inputs (bool/float/str seeds, negative epoch, empty id, bad degrees, bad shapes) rejected")


def main() -> None:
    tests = [
        test_rotation_matrix_is_orthogonal_with_determinant_one,
        test_rotation_matrix_matches_the_s5_14_diagnosis_convention,
        test_rotation_preserves_distances_and_z,
        test_rotation_round_trip_returns_original,
        test_apply_z_rotation_never_writes_into_the_input,
        test_angle_is_within_range_for_many_combinations,
        test_angle_is_deterministic_for_the_same_triple,
        test_angle_changes_with_epoch_video_and_seed,
        test_angle_matches_the_recorded_fixture_values,
        test_angle_is_independent_of_pythonhashseed,
        test_angle_derivation_does_not_touch_the_global_numpy_rng,
        test_key_is_injective_for_ids_containing_the_separator,
        test_stable_video_id_ignores_mount_prefix_and_relative_paths,
        test_none_mode_derives_no_angle_and_returns_points_unchanged,
        test_enabled_mode_rotates_and_reports_its_angle,
        test_rotate_video_points_gives_every_window_of_a_video_the_same_angle,
        test_resolve_prefers_explicit_seed_and_records_the_rule,
        test_to_config_dict_never_stores_a_null_seed,
        test_validation_rejects_bad_inputs,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 S5-15 rotation augmentation synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
