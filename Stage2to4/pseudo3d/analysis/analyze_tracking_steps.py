from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "shape") and hasattr(value, "tolist"):
        if value.shape == ():
            return value.item()
        return value.tolist()

    if hasattr(value, "item") and not isinstance(value, (str, int, float, bool)):
        return value.item()

    return value


def parse_step_pairs(raw_pairs: list[str] | None) -> list[tuple[int, int]]:
    """
    Parse frame step pairs such as:
      --step_pairs 4-5,7-8
      --step_pairs 4:5 --step_pairs 7:8
    """
    if not raw_pairs:
        return []

    pairs = []
    for raw in raw_pairs:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                left, right = item.split("-", maxsplit=1)
            elif ":" in item:
                left, right = item.split(":", maxsplit=1)
            else:
                raise ValueError(
                    f"Invalid step pair '{item}'. Use forms like 4-5 or 4:5."
                )
            pairs.append((int(left), int(right)))

    return pairs


def normalize_vector(vector, *, eps: float):
    import numpy as np

    norm = float(np.linalg.norm(vector))
    if norm <= eps or not np.isfinite(norm):
        raise ValueError(f"Cannot normalize vector with norm={norm}")
    return vector / norm, norm


def estimate_progress_axis(
    pred_tracking,
    *,
    method: str,
    window: int,
    eps: float,
):
    """
    Estimate progress axis from prior tracking centers.
    """
    import numpy as np

    centers = pred_tracking[:, :3, 3].astype(np.float64)
    steps = np.diff(centers, axis=0)
    finite_steps = np.isfinite(steps).all(axis=1)
    step_lengths = np.linalg.norm(np.where(finite_steps[:, None], steps, 0.0), axis=1)
    valid_mask = finite_steps & (step_lengths > eps)
    valid_indices = np.flatnonzero(valid_mask)

    if valid_indices.size == 0:
        raise ValueError("No valid prior steps available for axis estimation.")

    if method == "first_valid":
        candidate_indices = valid_indices[:1]
    elif method == "early_mean":
        if window <= 0:
            raise ValueError(f"early_mean window must be > 0, got {window}")
        candidate_indices = valid_indices[:window]
    elif method == "global_mean":
        candidate_indices = valid_indices
    elif method == "start_to_end_prior":
        axis, norm = normalize_vector(centers[-1] - centers[0], eps=eps)
        return axis, {
            "axis_method_effective": method,
            "axis_num_candidates": int(len(centers)),
            "axis_raw_norm": norm,
        }
    else:
        raise ValueError(f"Unknown axis method: {method}")

    candidate_dirs = steps[candidate_indices] / step_lengths[candidate_indices, None]
    mean_dir = candidate_dirs.mean(axis=0)
    axis, norm = normalize_vector(mean_dir, eps=eps)

    return axis, {
        "axis_method_effective": method,
        "axis_num_candidates": int(candidate_indices.size),
        "axis_raw_norm": norm,
    }


def estimate_start_to_end_corr_axis(pred_tracking_corr, *, eps: float):
    centers = pred_tracking_corr[:, :3, 3].astype("float64")
    axis, norm = normalize_vector(centers[-1] - centers[0], eps=eps)
    return axis, {
        "axis_method_effective": "start_to_end_corr",
        "axis_num_candidates": int(len(centers)),
        "axis_raw_norm": norm,
    }


def compute_frame_normal_from_corners(frame_corners_world, frame_idx: int, *, eps: float):
    """
    Estimate frame plane normal from corner order:
      0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left
    """
    import numpy as np

    corners = frame_corners_world[frame_idx].astype(np.float64)
    x_axis = corners[1] - corners[0]
    y_axis = corners[3] - corners[0]
    normal = np.cross(x_axis, y_axis)
    normal, _ = normalize_vector(normal, eps=eps)
    return normal


def load_tracking_data(input_h5: Path, *, geometry_key: str):
    import h5py
    import numpy as np

    with h5py.File(input_h5, "r") as f:
        if "pred_tracking" not in f:
            raise KeyError("Required key 'pred_tracking' not found.")

        pred_tracking = f["pred_tracking"][:].astype(np.float32)

        corr_key = "pred_tracking_corr"
        if corr_key in f:
            pred_tracking_corr = f[corr_key][:].astype(np.float32)
        else:
            pred_tracking_corr = pred_tracking.copy()

        corners_key = (
            "frame_corners_world_corr"
            if geometry_key == "corr"
            else "frame_corners_world"
        )
        frame_corners_world = None
        if corners_key in f:
            frame_corners_world = f[corners_key][:].astype(np.float32)

        attrs = {key: _to_python_scalar(value) for key, value in f.attrs.items()}
        correction_attrs = {}
        if "tracking_correction" in f:
            correction_attrs = {
                key: _to_python_scalar(value)
                for key, value in f["tracking_correction"].attrs.items()
            }

    return {
        "pred_tracking": pred_tracking,
        "pred_tracking_corr": pred_tracking_corr,
        "frame_corners_world": frame_corners_world,
        "attrs": attrs,
        "tracking_correction_attrs": correction_attrs,
    }


def resolve_axis(data: dict[str, Any], args: argparse.Namespace):
    method = args.axis_method
    correction_attrs = data["tracking_correction_attrs"]

    if method == "attrs":
        method = str(correction_attrs.get("initial_direction_method", "global_mean"))

    if args.axis_window is not None:
        window = args.axis_window
    else:
        window = int(correction_attrs.get("initial_direction_window", 15))

    if method == "start_to_end_corr":
        return estimate_start_to_end_corr_axis(
            data["pred_tracking_corr"],
            eps=args.eps,
        )

    return estimate_progress_axis(
        data["pred_tracking"],
        method=method,
        window=window,
        eps=args.eps,
    )


def _xyz_columns(prefix: str, vector) -> dict[str, float]:
    return {
        f"{prefix}_x": float(vector[0]),
        f"{prefix}_y": float(vector[1]),
        f"{prefix}_z": float(vector[2]),
    }


def analyze_pair(
    *,
    input_h5: Path,
    data: dict[str, Any],
    axis,
    axis_meta: dict[str, Any],
    frame_i: int,
    frame_j: int,
    include_frame_normal: bool,
    eps: float,
) -> dict[str, Any]:
    import numpy as np

    pred_tracking = data["pred_tracking"]
    pred_tracking_corr = data["pred_tracking_corr"]

    n = pred_tracking.shape[0]
    if frame_i < 0 or frame_j < 0 or frame_i >= n or frame_j >= n:
        raise IndexError(f"Frame pair {frame_i}-{frame_j} out of range for N={n}")

    prior_i = pred_tracking[frame_i, :3, 3].astype(np.float64)
    prior_j = pred_tracking[frame_j, :3, 3].astype(np.float64)
    corr_i = pred_tracking_corr[frame_i, :3, 3].astype(np.float64)
    corr_j = pred_tracking_corr[frame_j, :3, 3].astype(np.float64)

    prior_step = prior_j - prior_i
    corr_step = corr_j - corr_i

    prior_axis_delta = float(np.dot(prior_step, axis))
    corr_axis_delta = float(np.dot(corr_step, axis))
    prior_step_len = float(np.linalg.norm(prior_step))
    corr_step_len = float(np.linalg.norm(corr_step))

    correction_attrs = data["tracking_correction_attrs"]
    min_step = float(correction_attrs.get("progress_monotonic_min_step", 0.0) or 0.0)

    row: dict[str, Any] = {
        "input_h5": str(input_h5),
        "frame_i": frame_i,
        "frame_j": frame_j,
        "axis_method": axis_meta["axis_method_effective"],
        "axis_num_candidates": axis_meta["axis_num_candidates"],
        "axis_raw_norm": axis_meta["axis_raw_norm"],
        "progress_monotonic_enabled": correction_attrs.get(
            "progress_monotonic_enabled",
            "",
        ),
        "progress_monotonic_min_step": min_step,
        "progress_monotonic_min_step_ratio": correction_attrs.get(
            "progress_monotonic_min_step_ratio",
            "",
        ),
        "progress_axis_scale": correction_attrs.get("progress_axis_scale", ""),
        "prior_step_len": prior_step_len,
        "corr_step_len": corr_step_len,
        "prior_axis_delta": prior_axis_delta,
        "corr_axis_delta": corr_axis_delta,
        "corr_axis_delta_minus_min_step": corr_axis_delta - min_step,
        "corr_axis_is_reverse": bool(corr_axis_delta < -eps),
        "corr_axis_violates_min_step": bool(corr_axis_delta + eps < min_step),
    }

    row.update(_xyz_columns("axis", axis))
    row.update(_xyz_columns("prior_center_i", prior_i))
    row.update(_xyz_columns("prior_center_j", prior_j))
    row.update(_xyz_columns("corr_center_i", corr_i))
    row.update(_xyz_columns("corr_center_j", corr_j))
    row.update(_xyz_columns("prior_step", prior_step))
    row.update(_xyz_columns("corr_step", corr_step))

    frame_corners_world = data.get("frame_corners_world")
    if include_frame_normal and frame_corners_world is not None:
        normal = compute_frame_normal_from_corners(
            frame_corners_world,
            frame_i,
            eps=eps,
        )
        normal_axis_dot = float(np.dot(normal, axis))
        if normal_axis_dot >= 0.0:
            aligned_normal = normal
            aligned_normal_sign = 1
        else:
            aligned_normal = -normal
            aligned_normal_sign = -1

        normal_prior_delta = float(np.dot(prior_step, normal))
        normal_corr_delta = float(np.dot(corr_step, normal))
        aligned_normal_prior_delta = float(np.dot(prior_step, aligned_normal))
        aligned_normal_corr_delta = float(np.dot(corr_step, aligned_normal))

        row.update(_xyz_columns("frame_i_normal", normal))
        row.update(_xyz_columns("frame_i_normal_axis_aligned", aligned_normal))
        row["frame_i_normal_axis_dot"] = normal_axis_dot
        row["frame_i_normal_axis_aligned_sign"] = aligned_normal_sign
        row["prior_frame_normal_delta"] = normal_prior_delta
        row["corr_frame_normal_delta"] = normal_corr_delta
        row["prior_axis_aligned_frame_normal_delta"] = aligned_normal_prior_delta
        row["corr_axis_aligned_frame_normal_delta"] = aligned_normal_corr_delta
        row["corr_axis_aligned_frame_normal_is_reverse"] = bool(
            aligned_normal_corr_delta < -eps
        )

    return row


def write_rows(rows: list[dict[str, Any]], output_csv: Path | None):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    if output_csv is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved tracking step analysis: {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze selected frame-to-frame tracking steps in a pseudo-3D h5."
        )
    )

    parser.add_argument(
        "--input_h5",
        type=Path,
        required=True,
        help="Input pseudo-3D h5 file.",
    )

    parser.add_argument(
        "--step_pairs",
        action="append",
        default=None,
        help=(
            "Frame step pairs such as '4-5,7-8'. "
            "May be specified multiple times. If omitted, all adjacent pairs are used."
        ),
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, CSV is printed to stdout.",
    )

    parser.add_argument(
        "--axis_method",
        type=str,
        default="attrs",
        choices=[
            "attrs",
            "first_valid",
            "early_mean",
            "global_mean",
            "start_to_end_prior",
            "start_to_end_corr",
        ],
        help=(
            "Progress axis estimation method. "
            "attrs uses tracking_correction.initial_direction_method."
        ),
    )

    parser.add_argument(
        "--axis_window",
        type=int,
        default=None,
        help="Override early_mean valid-step window.",
    )

    parser.add_argument(
        "--geometry_key",
        type=str,
        default="corr",
        choices=["prior", "corr"],
        help="Frame corners used for optional frame-normal analysis.",
    )

    parser.add_argument(
        "--include_frame_normal",
        action="store_true",
        help="Also report projection onto frame_i plane normal when corners exist.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for reverse/min-step checks.",
    )

    return parser


def main():
    args = build_parser().parse_args()
    data = load_tracking_data(args.input_h5, geometry_key=args.geometry_key)
    axis, axis_meta = resolve_axis(data, args)

    pairs = parse_step_pairs(args.step_pairs)
    if not pairs:
        n = data["pred_tracking"].shape[0]
        pairs = [(i, i + 1) for i in range(n - 1)]

    rows = [
        analyze_pair(
            input_h5=args.input_h5,
            data=data,
            axis=axis,
            axis_meta=axis_meta,
            frame_i=frame_i,
            frame_j=frame_j,
            include_frame_normal=args.include_frame_normal,
            eps=args.eps,
        )
        for frame_i, frame_j in pairs
    ]

    write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
