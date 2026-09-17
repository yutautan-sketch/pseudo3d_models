from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# S5-15 P2 Step 1: training-only random Z-rotation augmentation.
#
# Pure numpy, no torch: the angle for one (video, epoch) is a deterministic
# function of (base_seed, epoch, stable_video_id) alone, so every window of a
# video sees the same angle within an epoch no matter which DataLoader worker
# loads it, in which order, or how many workers there are.
#
# The rotation matrix convention matches the S5-14 supplement-2 coordinate
# diagnosis (right-handed CCW about +Z, float64 internally, float32 out), so
# the +-15 degree range carries the same meaning as it did there. That module
# is a finished S5-14 artifact and is deliberately neither imported nor
# modified here.
# ----------------------------------------------------------------------------

MODE_NONE = "none"
MODE_RANDOM_Z_ROTATION = "random_z_rotation"
SUPPORTED_AUGMENTATION_MODES = (MODE_NONE, MODE_RANDOM_Z_ROTATION)

DEFAULT_MAX_ABS_DEGREES = 15.0

# Fallback offset applied to the training seed when no explicit augmentation
# seed is given (policy-chat answer 2). An explicit seed always wins.
AUGMENTATION_SEED_OFFSET = 500000
SEED_SOURCE_EXPLICIT = "explicit"
SEED_SOURCE_TRAIN_SEED_OFFSET = f"train_seed_plus_{AUGMENTATION_SEED_OFFSET}"

# Fixed so the derivation can never drift: the first 8 bytes of the SHA-256
# digest, big-endian, seed a fresh numpy Generator. Python's built-in hash()
# is never used -- it is randomized per process via PYTHONHASHSEED and would
# give different angles in different DataLoader workers.
DIGEST_SEED_BYTES = 8
DIGEST_SEED_BYTE_ORDER = "big"
MAX_SEED_VALUE = 2**63 - 1

ANGLE_DERIVATION_DESCRIPTION = (
    "angle_degrees = default_rng("
    "int.from_bytes(sha256(f'{base_seed}|{epoch}|{video_id}'.encode('utf-8')).digest()[:8], 'big')"
    ").uniform(-max_abs_degrees, +max_abs_degrees)"
)


def _validate_non_negative_int(value: object, name: str) -> int:
    # bool is an int subclass; an epoch of True would silently mean 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value > MAX_SEED_VALUE:
        raise ValueError(f"{name} must be <= {MAX_SEED_VALUE}, got {value}")
    return int(value)


def _validate_video_id(video_id: object) -> str:
    if not isinstance(video_id, str):
        raise TypeError(f"video_id must be a str, got {type(video_id).__name__}: {video_id!r}")
    if not video_id:
        raise ValueError("video_id must not be empty")
    return video_id


def _validate_max_abs_degrees(max_abs_degrees: object) -> float:
    try:
        value = float(max_abs_degrees)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"max_abs_degrees must be a real number, got {max_abs_degrees!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"max_abs_degrees must be finite, got {value}")
    if value <= 0.0:
        raise ValueError(
            f"max_abs_degrees must be > 0, got {value}. Use mode={MODE_NONE!r} to disable augmentation."
        )
    return value


def stable_video_id_from_path(path: str | Path) -> str:
    """The angle-deriving identifier for one video.

    Deliberately the file name alone: the same H5 read through a different
    mount prefix, or as a relative instead of an absolute path, must produce
    the same angle. Callers are responsible for rejecting duplicate file names
    within one file list, which would otherwise collide here.
    """
    name = Path(path).name
    if not name:
        raise ValueError(f"cannot derive a stable video id from path: {path!r}")
    return name


def rotation_matrix_z(degrees: float) -> np.ndarray:
    theta = np.deg2rad(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_angle_key(*, base_seed: int, epoch: int, video_id: str) -> str:
    """The exact string hashed to derive an angle.

    base_seed and epoch are integers and cannot contain '|', so the first two
    separators are unambiguous and the trailing video_id -- whatever it
    contains -- cannot forge another triple's key.
    """
    return f"{base_seed}|{epoch}|{video_id}"


def derive_rotation_angle_degrees(
    *,
    base_seed: int,
    epoch: int,
    video_id: str,
    max_abs_degrees: float = DEFAULT_MAX_ABS_DEGREES,
) -> float:
    """Angle for one (video, epoch), reproducible across processes and runs.

    Uses its own Generator seeded from the digest, so it neither consumes nor
    perturbs numpy's global RNG (shared with shuffling and model init).
    """
    base_seed = _validate_non_negative_int(base_seed, "base_seed")
    epoch = _validate_non_negative_int(epoch, "epoch")
    video_id = _validate_video_id(video_id)
    max_abs_degrees = _validate_max_abs_degrees(max_abs_degrees)

    key = rotation_angle_key(base_seed=base_seed, epoch=epoch, video_id=video_id)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:DIGEST_SEED_BYTES], DIGEST_SEED_BYTE_ORDER)
    return float(np.random.default_rng(seed).uniform(-max_abs_degrees, max_abs_degrees))


def apply_z_rotation(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate [N, 3] points about the origin around +Z, returning a new array.

    The caller passes already-`normalize_xyz()`-ed points, whose centroid is
    the origin, so this is the rotation about the video centroid required by
    the S5-15 contract. Never re-normalizes, re-centers, scales or clips, and
    never writes into the input.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected [N, 3] points, got shape {values.shape}")
    matrix = rotation_matrix_z(angle_degrees)
    return (values @ matrix.T).astype(np.float32)


@dataclass(frozen=True)
class AugmentationConfig:
    """Resolved augmentation settings, as stored in config.json/checkpoints.

    base_seed is always a concrete int (never None) and seed_source records
    how it was obtained, so a run can be reproduced from its own config.
    """

    mode: str = MODE_NONE
    max_abs_degrees: float = DEFAULT_MAX_ABS_DEGREES
    base_seed: int = 0
    seed_source: str = SEED_SOURCE_EXPLICIT

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_AUGMENTATION_MODES:
            raise ValueError(
                f"unsupported augmentation mode: {self.mode!r}. "
                f"Supported modes are: {', '.join(SUPPORTED_AUGMENTATION_MODES)}"
            )
        object.__setattr__(self, "max_abs_degrees", _validate_max_abs_degrees(self.max_abs_degrees))
        object.__setattr__(self, "base_seed", _validate_non_negative_int(self.base_seed, "base_seed"))

    @classmethod
    def resolve(
        cls,
        *,
        mode: str,
        train_seed: int,
        augmentation_seed: int | None = None,
        max_abs_degrees: float = DEFAULT_MAX_ABS_DEGREES,
    ) -> AugmentationConfig:
        if augmentation_seed is None:
            train_seed = _validate_non_negative_int(train_seed, "train_seed")
            base_seed = train_seed + AUGMENTATION_SEED_OFFSET
            seed_source = SEED_SOURCE_TRAIN_SEED_OFFSET
        else:
            base_seed = _validate_non_negative_int(augmentation_seed, "augmentation_seed")
            seed_source = SEED_SOURCE_EXPLICIT
        return cls(mode=mode, max_abs_degrees=max_abs_degrees, base_seed=base_seed, seed_source=seed_source)

    @property
    def enabled(self) -> bool:
        return self.mode == MODE_RANDOM_Z_ROTATION

    def angle_degrees_for(self, *, video_id: str, epoch: int) -> float:
        if not self.enabled:
            raise RuntimeError(
                f"angle_degrees_for() called while mode={self.mode!r}; "
                "callers must check .enabled so the disabled path derives no angle at all"
            )
        return derive_rotation_angle_degrees(
            base_seed=self.base_seed,
            epoch=epoch,
            video_id=video_id,
            max_abs_degrees=self.max_abs_degrees,
        )

    def to_config_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "max_abs_degrees": self.max_abs_degrees,
            "base_seed": self.base_seed,
            "seed_source": self.seed_source,
            "angle_derivation": ANGLE_DERIVATION_DESCRIPTION,
        }


def rotate_video_points(
    points: np.ndarray,
    *,
    config: AugmentationConfig,
    video_id: str,
    epoch: int,
) -> tuple[np.ndarray, float | None]:
    """Apply the (video, epoch) rotation to a whole video's normalized points.

    Returns the points unchanged with a None angle when augmentation is off,
    so the disabled path stays free of any angle derivation. Applying this
    once per video before window extraction is what gives every window of that
    video the same angle within an epoch.
    """
    if not config.enabled:
        return points, None
    angle_degrees = config.angle_degrees_for(video_id=video_id, epoch=epoch)
    return apply_z_rotation(points, angle_degrees), angle_degrees
