from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "h5_path",
    "video_name",
    "status",
    "source_video",
    "num_frames",
    "num_steps",
    "tracking_correction_preset",
    "tracking_correction_overrides",
    "correction_enabled",
    "median_step",
    "min_step",
    "max_step",
    "min_cos",
    "min_step_ratio",
    "max_step_ratio",
    "direction_update_alpha",
    "initial_direction_method",
    "initial_direction_window",
    "initial_direction_num_candidates",
    "initial_direction_norm",
    "initial_direction_fallback_to_first_valid",
    "progress_axis_scale",
    "progress_axis_scale_enabled",
    "progress_axis_scale_axis_source",
    "progress_axis_scale_axis_norm",
    "progress_axis_scale_extent_before",
    "progress_axis_scale_extent_after",
    "progress_axis_scale_displacement_mean",
    "progress_axis_scale_displacement_max",
    "progress_monotonic_enabled",
    "progress_monotonic_axis_source",
    "progress_monotonic_axis_norm",
    "progress_monotonic_min_step_ratio",
    "progress_monotonic_min_step",
    "num_progress_monotonic_adjusted",
    "progress_monotonic_displacement_mean",
    "progress_monotonic_displacement_max",
    "frame_normal_same_side_enabled",
    "frame_normal_same_side_min_step_ratio",
    "frame_normal_same_side_min_step",
    "frame_normal_same_side_boundary_enabled",
    "frame_normal_same_side_boundary_reference_norm",
    "num_frame_normal_same_side_boundary_detected",
    "num_frame_normal_same_side_boundary_corrected",
    "num_frame_normal_same_side_detected",
    "num_frame_normal_same_side_corrected",
    "frame_normal_same_side_displacement_mean",
    "frame_normal_same_side_displacement_max",
    "num_reverse_steps",
    "num_short_steps",
    "num_long_steps",
    "num_zero_or_invalid_steps",
    "reverse_step_ratio",
    "short_step_ratio",
    "long_step_ratio",
    "invalid_step_ratio",
    "step_lengths_prior_mean",
    "step_lengths_prior_median",
    "step_lengths_corr_mean",
    "step_lengths_corr_median",
]


def _to_python_scalar(value: Any) -> Any:
    """
    Convert h5/numpy attr values to CSV-friendly Python scalars.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "shape") and hasattr(value, "tolist"):
        if value.shape == ():
            return value.item()
        return value.tolist()

    if hasattr(value, "item") and not isinstance(value, (str, int, float, bool)):
        return value.item()

    return value


def _as_int(value: Any, default: int = 0) -> int:
    value = _to_python_scalar(value)
    if value in ("", None):
        return default
    return int(value)


def _safe_ratio(count: Any, num_steps: int) -> float:
    if num_steps <= 0:
        return 0.0
    return float(_as_int(count)) / float(num_steps)


def collect_h5_paths(
    *,
    input_h5: Path | None,
    input_dir: Path | None,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    """
    Collect h5 paths from either a single file or a directory.
    """
    if input_h5 is not None:
        input_h5 = Path(input_h5)
        if not input_h5.exists():
            raise FileNotFoundError(f"Input h5 not found: {input_h5}")
        return [input_h5]

    if input_dir is None:
        raise ValueError("Either input_h5 or input_dir must be specified.")

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    globber = input_dir.rglob if recursive else input_dir.glob
    paths = sorted(p for p in globber(pattern) if p.is_file())

    if not paths:
        raise FileNotFoundError(
            f"No h5 files matched pattern='{pattern}' in {input_dir}"
        )

    return paths


def summarize_one_h5(h5_path: Path, *, strict: bool = False) -> dict[str, Any]:
    """
    Read tracking_correction attrs from one pseudo-3D h5.
    """
    import h5py

    h5_path = Path(h5_path)

    row: dict[str, Any] = {
        "h5_path": str(h5_path),
        "video_name": h5_path.stem,
        "status": "ok",
    }

    with h5py.File(h5_path, "r") as f:
        root_attrs = {k: _to_python_scalar(v) for k, v in f.attrs.items()}

        row["source_video"] = root_attrs.get("source_video", "")

        if "num_frames" in root_attrs:
            num_frames = _as_int(root_attrs["num_frames"])
        elif "frame_indices" in f:
            num_frames = int(f["frame_indices"].shape[0])
        elif "pred_tracking" in f:
            num_frames = int(f["pred_tracking"].shape[0])
        else:
            num_frames = 0

        row["num_frames"] = num_frames
        row["num_steps"] = max(num_frames - 1, 0)

        if "tracking_correction" not in f:
            if strict:
                raise KeyError(f"tracking_correction group not found in {h5_path}")
            row["status"] = "missing_tracking_correction"
            return _fill_missing_columns(row)

        correction_attrs = {
            k: _to_python_scalar(v)
            for k, v in f["tracking_correction"].attrs.items()
        }

    row.update(correction_attrs)

    num_steps = _as_int(row.get("num_steps"))
    row["reverse_step_ratio"] = _safe_ratio(row.get("num_reverse_steps"), num_steps)
    row["short_step_ratio"] = _safe_ratio(row.get("num_short_steps"), num_steps)
    row["long_step_ratio"] = _safe_ratio(row.get("num_long_steps"), num_steps)
    row["invalid_step_ratio"] = _safe_ratio(
        row.get("num_zero_or_invalid_steps"),
        num_steps,
    )

    return _fill_missing_columns(row)


def _fill_missing_columns(row: dict[str, Any]) -> dict[str, Any]:
    for column in SUMMARY_COLUMNS:
        row.setdefault(column, "")
    return row


def write_summary_csv(rows: list[dict[str, Any]], output_csv: Path | None):
    """
    Write summary rows to a CSV file, or stdout when output_csv is None.
    """
    if output_csv is None:
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=SUMMARY_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        return

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved tracking correction summary: {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize tracking_correction metadata from pseudo-3D h5 files."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_h5",
        type=Path,
        default=None,
        help="Single pseudo-3D h5 file.",
    )
    input_group.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory containing pseudo-3D h5 files.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.h5",
        help="Glob pattern used with --input_dir. Default: *.h5.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input_dir recursively.",
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, CSV is printed to stdout.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if a file has no tracking_correction group.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    h5_paths = collect_h5_paths(
        input_h5=args.input_h5,
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    rows = [summarize_one_h5(path, strict=args.strict) for path in h5_paths]
    write_summary_csv(rows, args.output_csv)

    print(f"Processed h5 files: {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
