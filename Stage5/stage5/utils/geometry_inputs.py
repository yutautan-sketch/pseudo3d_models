from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from stage5.utils.frame_windows import generate_frame_order_windows, point_indices_for_window
from stage5.utils.split_identity import extract_video_identities

# ----------------------------------------------------------------------------
# S5-17 S17-2: raw-array readers for the geometry CLI.
#
# Management document 3.3. Every reader checks the RAW arrays -- dtype, shape,
# row count, finiteness, integrality, value domain and the range of the
# integer type they are converted to -- and only then converts explicitly.
# `h5_io.load_stage5_pointcloud_h5` is not used here because it casts first
# (`astype`), which would hide a non-integral label, a non-0/1 valid mask or a
# float64 -> float32 loss.
#
# These functions never apply the seal contract themselves: the CLI calls the
# guard on every path BEFORE handing it to a reader. Error messages carry
# field names and counts, never a path or a video name, because they may be
# pasted into a shared log.
# ----------------------------------------------------------------------------

INT64_MAX = int(np.iinfo(np.int64).max)
INT64_MIN = int(np.iinfo(np.int64).min)
PROBABILITY_BAND = 1e-6          # management 3.3: |p1 - 0.5| <= 1e-6 accepts either label
NPZ_REQUIRED_KEYS = ("point_indices", "prob_femur", "pred_label", "vote_count")
CONFUSION_KEYS = ("true_positive_count", "false_positive_count", "true_negative_count", "false_negative_count")


class InputContractError(ValueError):
    """A raw input violates its contract. The run stops."""


# ----------------------------------------------------------------------------
# primitive checks
# ----------------------------------------------------------------------------


def integral_array(values: Any, *, name: str, lo: int | None = None, hi: int | None = None) -> np.ndarray:
    """int64 copy of an integral array, checked BEFORE conversion."""
    array = np.asarray(values)
    if array.dtype.kind == "b":
        raise InputContractError(f"{name}: boolean where integers were expected")
    if array.dtype.kind in "iu":
        if array.size and array.dtype.kind == "u" and int(array.max()) > INT64_MAX:
            raise InputContractError(f"{name}: values exceed the int64 range")
        out = array.astype(np.int64)
    elif array.dtype.kind == "f":
        if not np.all(np.isfinite(array)):
            raise InputContractError(f"{name}: non-finite values")
        if array.size and (float(array.min()) < INT64_MIN or float(array.max()) >= 2.0 ** 63):
            raise InputContractError(f"{name}: values exceed the int64 range")
        if not np.all(array == np.floor(array)):
            raise InputContractError(f"{name}: non-integral values")
        out = array.astype(np.int64)
    else:
        raise InputContractError(f"{name}: dtype {array.dtype} is not numeric")
    if lo is not None and out.size and int(out.min()) < lo:
        raise InputContractError(f"{name}: values below {lo}")
    if hi is not None and out.size and int(out.max()) > hi:
        raise InputContractError(f"{name}: values above {hi}")
    return out


def boolean_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return array.astype(bool)
    return integral_array(array, name=name, lo=0, hi=1).astype(bool)


def finite_float_array(values: Any, *, name: str, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise InputContractError(f"{name}: dtype {array.dtype} is not numeric")
    if array.ndim != 1 + len(shape_tail) or tuple(array.shape[1:]) != shape_tail:
        raise InputContractError(f"{name}: shape {array.shape}, expected [N, {', '.join(map(str, shape_tail))}]")
    out = array.astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise InputContractError(f"{name}: non-finite values")
    return out


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def single_identity(name: str, *, what: str) -> str:
    identities = extract_video_identities(name)
    if len(identities) != 1:
        raise InputContractError(f"{what}: expected exactly one video identity in the name, found {len(identities)}")
    return identities[0]


# ----------------------------------------------------------------------------
# teacher H5
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherArrays:
    pixel_xy: np.ndarray          # [N, 2] float64, LOCAL_CROP_PX
    frame_order: np.ndarray       # [N] int64
    point_label: np.ndarray       # [N] int64 in {-1, 0, 1}
    valid_mask: np.ndarray        # [N] bool, == (point_label != -1)
    points: np.ndarray            # [N, 3] float64, PSEUDO3D
    bbox_frame_order: np.ndarray | None
    bbox_local_xyxy: np.ndarray | None   # rows may be non-finite (Stage 4 skips those)

    @property
    def n_points(self) -> int:
        return int(self.frame_order.shape[0])

    @property
    def gt_positive(self) -> np.ndarray:
        return self.valid_mask & (self.point_label == 1)

    @property
    def valid_background(self) -> np.ndarray:
        return self.valid_mask & (self.point_label == 0)


def read_teacher_attrs(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        attrs = {key: _decode(value) for key, value in handle.attrs.items()}
    return {
        "video_name": str(attrs.get("video_name") or ""),
        "source_pseudo3d_h5": (str(attrs["source_pseudo3d_h5"]) if attrs.get("source_pseudo3d_h5") else None),
    }


def read_teacher_arrays(path: Path) -> TeacherArrays:
    with h5py.File(path, "r") as handle:
        for group, keys in (("point_cloud", ("pixel_xy", "frame_order", "points")),
                            ("annotation", ("point_label", "valid_mask"))):
            if group not in handle:
                raise InputContractError(f"teacher H5 lacks group {group!r}")
            missing = [key for key in keys if key not in handle[group]]
            if missing:
                raise InputContractError(f"teacher H5 lacks {group}/{missing}")
        pixel_xy = handle["point_cloud/pixel_xy"][()]
        frame_order = handle["point_cloud/frame_order"][()]
        points = handle["point_cloud/points"][()]
        point_label = handle["annotation/point_label"][()]
        valid_mask = handle["annotation/valid_mask"][()]
        bbox_frame_order = bbox_local_xyxy = None
        if "frame_annotation" in handle:
            group = handle["frame_annotation"]
            if "frame_order" in group and "bbox_local_xyxy" in group:
                bbox_frame_order = group["frame_order"][()]
                bbox_local_xyxy = group["bbox_local_xyxy"][()]

    xy = finite_float_array(pixel_xy, name="point_cloud/pixel_xy", shape_tail=(2,))
    n = xy.shape[0]
    xyz = finite_float_array(points, name="point_cloud/points", shape_tail=(3,))
    for name, array in (("point_cloud/frame_order", frame_order), ("annotation/point_label", point_label),
                        ("annotation/valid_mask", valid_mask)):
        if np.asarray(array).shape != (n,):
            raise InputContractError(f"{name}: shape {np.asarray(array).shape}, expected [{n}]")
    if xyz.shape[0] != n:
        raise InputContractError("point_cloud/points: row count differs from pixel_xy")
    frames = integral_array(frame_order, name="point_cloud/frame_order", lo=0)
    labels = integral_array(point_label, name="annotation/point_label", lo=-1, hi=1)
    valid = boolean_array(valid_mask, name="annotation/valid_mask")
    if not np.array_equal(valid, labels != -1):
        raise InputContractError(
            f"annotation: valid_mask disagrees with point_label != -1 at {int(np.sum(valid != (labels != -1)))} point(s)"
        )

    bbox_frames = bbox_xyxy = None
    if bbox_frame_order is not None:
        bbox_frames = integral_array(bbox_frame_order, name="frame_annotation/frame_order", lo=0)
        raw = np.asarray(bbox_local_xyxy)
        if raw.dtype.kind not in "fiu" or raw.shape != (bbox_frames.shape[0], 4):
            raise InputContractError("frame_annotation/bbox_local_xyxy: shape or dtype does not match frame_order")
        bbox_xyxy = raw.astype(np.float64)
    return TeacherArrays(xy, frames, labels, valid, xyz, bbox_frames, bbox_xyxy)


def h5_dataset_meta(path: Path) -> dict[str, Any]:
    """Shapes and dtypes only; no values are read (fail-fast audit)."""
    wanted = ("point_cloud/pixel_xy", "point_cloud/frame_order", "point_cloud/points",
              "annotation/point_label", "annotation/valid_mask",
              "frame_annotation/frame_order", "frame_annotation/bbox_local_xyxy")
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in wanted:
            out[key] = ({"shape": list(handle[key].shape), "dtype": str(handle[key].dtype)} if key in handle else None)
        out["attrs_present"] = sorted(k for k in ("video_name", "source_pseudo3d_h5") if k in handle.attrs)
    return out


# ----------------------------------------------------------------------------
# intermediate H5 (exact resolution only)
# ----------------------------------------------------------------------------

_SHAPE = re.compile(r"^\(\s*([\d,\s]+?)\s*,?\s*\)$")


def parse_shape(value: Any) -> tuple[int, ...] | None:
    match = _SHAPE.match(str(_decode(value)).strip())
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split(",") if part.strip())
    except ValueError:
        return None


def resolve_intermediate_exact(
    *,
    teacher_identity: str,
    video_name: str,
    recorded_source: str | None,
    intermediate_root: Path,
    suffix: str,
) -> tuple[Path, str]:
    """Recorded attr or exact basename, each with an exact identity match. Nothing else.

    No prefix glob and no ambiguity resolution: `..._091` must never resolve to
    `..._091_02` (report 1.3 F2).
    """
    if recorded_source:
        recorded = Path(recorded_source)
        if recorded.is_file():
            if extract_video_identities(recorded.name) != (teacher_identity,):
                raise InputContractError("the recorded intermediate H5 belongs to a different video identity")
            return recorded, "recorded_source_attr"
    if extract_video_identities(video_name) != (teacher_identity,):
        raise InputContractError("the teacher's video_name attr does not carry exactly the teacher's identity")
    candidate = intermediate_root / f"{video_name}{suffix}"
    if candidate.is_file():
        if extract_video_identities(candidate.name) != (teacher_identity,):
            raise InputContractError("the exact-basename intermediate H5 carries a different identity")
        return candidate, "basename_exact"
    raise InputContractError("the intermediate H5 could not be resolved by recorded attr or exact basename")


@dataclass(frozen=True)
class IntermediateMeta:
    attrs: dict[str, Any]
    local_shape_hw: tuple[int, int]
    n_local_frames: int


def read_intermediate_meta(path: Path) -> IntermediateMeta:
    with h5py.File(path, "r") as handle:
        attrs = {key: _decode(value) for key, value in handle.attrs.items()}
    shape = parse_shape(attrs.get("local_input_shape", ""))
    if shape is None or len(shape) != 4 or shape[2] <= 1 or shape[3] <= 1 or shape[0] <= 0:
        raise InputContractError("intermediate local_input_shape is missing or not (N, C, H, W)")
    return IntermediateMeta(attrs=attrs, local_shape_hw=(int(shape[2]), int(shape[3])), n_local_frames=int(shape[0]))


# ----------------------------------------------------------------------------
# prediction NPZ
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionArrays:
    pred_label: np.ndarray        # [N] bool
    prob_femur: np.ndarray        # [N] float64
    vote_count: np.ndarray        # [N] int64
    band_count: int               # points with |p1 - 0.5| <= band (either label accepted)


def check_label_probability(pred_label: np.ndarray, prob: np.ndarray, *, band: float = PROBABILITY_BAND) -> int:
    """Consistency of the saved label with the saved positive probability.

    The evaluation took argmax over float32 two-class means (ties -> background)
    and saved only p1, so p1 alone cannot reproduce labels inside a narrow band
    around 0.5. Outside the band the label must follow p1; inside, either label
    is accepted and counted.
    """
    above = prob > 0.5 + band
    below = prob < 0.5 - band
    if np.any(above & ~pred_label) or np.any(below & pred_label):
        raise InputContractError(
            f"pred_label disagrees with prob_femur outside the +-{band:g} band at "
            f"{int(np.sum(above & ~pred_label) + np.sum(below & pred_label))} point(s)"
        )
    return int(np.sum(~above & ~below))


def read_prediction_npz(path: Path, *, n_points: int) -> PredictionArrays:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in NPZ_REQUIRED_KEYS if key not in payload.files]
        if missing:
            raise InputContractError(f"prediction NPZ lacks {missing}")
        raw = {key: payload[key] for key in NPZ_REQUIRED_KEYS}
    for key, array in raw.items():
        if np.asarray(array).shape != (n_points,):
            raise InputContractError(f"prediction {key}: shape {np.asarray(array).shape}, expected [{n_points}]")
    indices = integral_array(raw["point_indices"], name="prediction point_indices", lo=0)
    if not np.array_equal(indices, np.arange(n_points, dtype=np.int64)):
        raise InputContractError("prediction point_indices is not arange(N) (the saved point-order convention)")
    labels = integral_array(raw["pred_label"], name="prediction pred_label", lo=0, hi=1).astype(bool)
    votes = integral_array(raw["vote_count"], name="prediction vote_count", lo=1)
    prob = np.asarray(raw["prob_femur"])
    if prob.dtype.kind != "f":
        raise InputContractError(f"prediction prob_femur: dtype {prob.dtype} is not floating")
    prob = prob.astype(np.float64)
    if not np.all(np.isfinite(prob)) or np.any(prob < 0.0) or np.any(prob > 1.0):
        raise InputContractError("prediction prob_femur: non-finite or outside [0, 1]")
    band = check_label_probability(labels, prob)
    return PredictionArrays(labels, prob, votes, band)


def npz_member_shapes(path: Path) -> dict[str, Any]:
    """Array shapes and dtypes from the .npy headers only (no array data is decoded)."""
    out: dict[str, Any] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            out[member[:-4]] = {"shape": list(shape), "dtype": str(dtype)}
    return out


def expected_vote_count(frame_order: np.ndarray, *, window_size: int, stride: int, include_tail: bool) -> np.ndarray:
    windows = generate_frame_order_windows(
        frame_order, window_size_frames=window_size, window_stride_frames=stride, include_tail_window=include_tail
    )
    counts = np.zeros(frame_order.shape[0], dtype=np.int64)
    for window in windows:
        counts[point_indices_for_window(frame_order, window)] += 1
    return counts


def confusion_counts(point_label: np.ndarray, valid_mask: np.ndarray, pred_label: np.ndarray) -> dict[str, int]:
    """Same definition as evaluate_stage5.compute_metrics (valid = valid_mask & label != -1)."""
    valid = valid_mask & (point_label != -1)
    positive = point_label == 1
    return {
        "true_positive_count": int(np.sum(valid & positive & pred_label)),
        "false_positive_count": int(np.sum(valid & ~positive & pred_label)),
        "true_negative_count": int(np.sum(valid & ~positive & ~pred_label)),
        "false_negative_count": int(np.sum(valid & positive & ~pred_label)),
    }


# ----------------------------------------------------------------------------
# evaluation-run artifacts (read only after the coverage check)
# ----------------------------------------------------------------------------


def read_summary_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputContractError(f"summary.json unreadable ({error.__class__.__name__})") from error
    for key in ("checkpoint", "checkpoint_epoch", "selected_train_files", "validation_files",
                "window_size_frames", "window_stride_frames", "include_tail_window"):
        if key not in payload:
            raise InputContractError(f"summary.json lacks {key!r}")
    return payload


def read_h5_metrics_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise InputContractError(f"h5_metrics.csv unreadable ({error.__class__.__name__})") from error
    required = ("split", "video_name", "h5_path", *CONFUSION_KEYS)
    if rows and any(key not in rows[0] for key in required):
        raise InputContractError("h5_metrics.csv lacks required columns")
    return rows
