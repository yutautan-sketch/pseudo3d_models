from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from stage5.utils.h5_io import read_path_list

# ----------------------------------------------------------------------------
# S5-14 Step H2.5: Stage 4 dataset inventory and XY coordinate provenance
# audit. Determines whether cross-video XY comparison (Step H3.1) can use
# real per-video local-crop dimensions recovered from intermediate pseudo-3D
# H5 files (Step H2.5-B/C priority "option 3"), rather than a guessed
# constant or the observed-point-max normalization that H2.5/6 explicitly
# forbid for this purpose. h5py/numpy only, no torch/CUDA. Read-only: never
# writes to any teacher/source H5, never retrains, never changes production
# settings.
# ----------------------------------------------------------------------------


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------------
# Step H2.5-A: Stage 4 dataset inventory
# ----------------------------------------------------------------------------

MANIFEST_LIKE_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.csv",
    "*manifest*",
    "*summary*",
    "*README*",
    "*readme*",
)


def scan_dataset_inventory(stage4_training_ablation_root: Path) -> list[dict[str, Any]]:
    """Enumerate dataset/run directories under stage4_training_ablation/<date>/.

    Best-effort: records H5 counts under conventional `collected`/`annotated`
    subdirectories and any manifest/summary/log-like files found at shallow
    depth, without assuming a fixed schema for every dataset generation.
    """
    require(
        stage4_training_ablation_root.is_dir(),
        f"Dataset root not found: {stage4_training_ablation_root}",
    )
    entries: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in stage4_training_ablation_root.iterdir() if p.is_dir()):
        entry: dict[str, Any] = {"run_name": run_dir.name, "path": str(run_dir)}
        for subdir_name in ("collected", "annotated"):
            subdir = run_dir / subdir_name
            if subdir.is_dir():
                entry[f"{subdir_name}_h5_count"] = sum(1 for _ in subdir.glob("*.h5"))
                entry[f"{subdir_name}_dir"] = str(subdir)
            else:
                entry[f"{subdir_name}_h5_count"] = None
                entry[f"{subdir_name}_dir"] = None
        manifest_like: list[str] = []
        for pattern in MANIFEST_LIKE_PATTERNS:
            manifest_like.extend(sorted(p.name for p in run_dir.glob(pattern) if p.is_file()))
        entry["manifest_like_files"] = sorted(set(manifest_like))
        entries.append(entry)
    return entries


# ----------------------------------------------------------------------------
# Step H2.5-B: H5 attrs and source provenance
# ----------------------------------------------------------------------------


def parse_shape_string(value: str) -> tuple[int, ...] | None:
    """Parse a Python tuple-repr string like '(120, 1, 256, 256)' into ints.

    Returns None (not raises) if the string does not contain a clean
    comma-separated integer list -- callers treat this as an unresolved
    dimension for that video, not a fatal error.
    """
    match = re.match(r"^\(\s*([\d,\s]+?)\s*,?\s*\)$", value.strip())
    if not match:
        return None
    parts = [item.strip() for item in match.group(1).split(",") if item.strip()]
    try:
        return tuple(int(item) for item in parts)
    except ValueError:
        return None


def read_intermediate_local_dimensions(path: Path) -> dict[str, Any]:
    """Read crop/resize provenance attrs from an intermediate pseudo3d H5.

    Returns a dict always containing "status" ("ok" or a reason string) and,
    when status == "ok", "width"/"height" (the local-crop pixel space that
    `pixel_xy` in the final H5 is expressed in -- confirmed square per video
    since every preprocessing branch crops/resizes to `(local_crop_size,
    local_crop_size)`).
    """
    if not path.is_file():
        return {"status": "intermediate_h5_not_found"}
    try:
        with h5py.File(path, "r") as f:
            attrs = dict(f.attrs)
    except OSError as exc:
        return {"status": f"intermediate_h5_unreadable: {exc}"}

    raw_shape = attrs.get("local_input_shape")
    if raw_shape is None:
        return {"status": "local_input_shape_attr_missing"}
    if isinstance(raw_shape, bytes):
        raw_shape = raw_shape.decode("utf-8")
    shape = parse_shape_string(str(raw_shape))
    if shape is None or len(shape) != 4:
        return {"status": f"local_input_shape_unparseable: {raw_shape!r}"}

    _num_frames, _channels, height, width = shape
    if height <= 1 or width <= 1:
        return {"status": f"local_input_shape_degenerate: {shape}"}
    if height != width:
        return {"status": f"local_input_shape_not_square: height={height}, width={width}"}

    return {
        "status": "ok",
        "width": int(width),
        "height": int(height),
        "raw_width": attrs.get("raw_width"),
        "raw_height": attrs.get("raw_height"),
        "local_crop_top": attrs.get("local_crop_top"),
        "local_crop_left": attrs.get("local_crop_left"),
        "local_resize_scale": attrs.get("local_resize_scale"),
        "local_preprocess_effective": attrs.get("local_preprocess_effective"),
    }


def resolve_intermediate_h5(
    *,
    video_name: str,
    recorded_source_path: str | None,
    fallback_root: Path,
    fallback_suffix: str,
) -> dict[str, Any]:
    """Locate the intermediate pseudo3d H5 for one video.

    Tries the path recorded in the final H5's `source_pseudo3d_h5` attr
    first; falls back to a basename lookup under `fallback_root` if that
    path does not resolve. The two resolution methods are recorded
    separately (never silently treated as equivalent), per handoff H2.5-B.
    """
    if recorded_source_path:
        recorded = Path(recorded_source_path)
        if recorded.is_file():
            return {"method": "recorded_source_attr", "path": str(recorded)}

    exact_candidate = fallback_root / f"{video_name}{fallback_suffix}"
    if exact_candidate.is_file():
        return {"method": "basename_fallback_exact", "path": str(exact_candidate)}

    if fallback_root.is_dir():
        matches = sorted(fallback_root.glob(f"{video_name}*.h5"))
        if len(matches) == 1:
            return {"method": "basename_fallback_glob", "path": str(matches[0])}
        if len(matches) > 1:
            return {"method": "basename_fallback_ambiguous", "path": None, "candidates": [str(m) for m in matches]}

    return {"method": "unresolved", "path": None}


def audit_video_provenance(
    final_h5_path: Path,
    *,
    fallback_root: Path,
    fallback_suffix: str,
) -> dict[str, Any]:
    with h5py.File(final_h5_path, "r") as f:
        final_attrs = dict(f.attrs)
        require("point_cloud" in f and "pixel_xy" in f["point_cloud"], f"{final_h5_path}: missing point_cloud/pixel_xy")
        pixel_xy = f["point_cloud/pixel_xy"][:]

    video_name = final_attrs.get("video_name")
    if isinstance(video_name, bytes):
        video_name = video_name.decode("utf-8")
    require(bool(video_name), f"{final_h5_path}: missing video_name attr")

    recorded_source = final_attrs.get("source_pseudo3d_h5")
    if isinstance(recorded_source, bytes):
        recorded_source = recorded_source.decode("utf-8")

    resolution = resolve_intermediate_h5(
        video_name=str(video_name),
        recorded_source_path=str(recorded_source) if recorded_source else None,
        fallback_root=fallback_root,
        fallback_suffix=fallback_suffix,
    )

    dimensions: dict[str, Any] = {"status": "not_attempted"}
    bounds_ok: bool | None = None
    if resolution["path"]:
        dimensions = read_intermediate_local_dimensions(Path(resolution["path"]))
        if dimensions.get("status") == "ok":
            width, height = dimensions["width"], dimensions["height"]
            finite = bool(np.all(np.isfinite(pixel_xy)))
            in_bounds = finite and bool(
                np.all((pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < width))
                and np.all((pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < height))
            )
            bounds_ok = in_bounds

    return {
        "video_name": str(video_name),
        "has_recorded_source_attr": recorded_source is not None,
        "resolution_method": resolution["method"],
        "resolved_path": resolution["path"],
        "dimension_status": dimensions.get("status"),
        "width": dimensions.get("width"),
        "height": dimensions.get("height"),
        "pixel_xy_min_x": float(np.min(pixel_xy[:, 0])) if pixel_xy.size else None,
        "pixel_xy_max_x": float(np.max(pixel_xy[:, 0])) if pixel_xy.size else None,
        "pixel_xy_min_y": float(np.min(pixel_xy[:, 1])) if pixel_xy.size else None,
        "pixel_xy_max_y": float(np.max(pixel_xy[:, 1])) if pixel_xy.size else None,
        "pixel_xy_bounds_ok": bounds_ok,
    }


# ----------------------------------------------------------------------------
# Step H2.5-C: dimension decision priority order
# ----------------------------------------------------------------------------


def decide_dimension_policy(per_video: list[dict[str, Any]]) -> dict[str, Any]:
    # A resolved dimension is only usable if pixel_xy actually fits inside it.
    # A video with dimension_status == "ok" but pixel_xy_bounds_ok is False is
    # not "unresolved" (a normal, expected case handled by options 2/4) -- it
    # is a contradiction: our own per-video resolved dimension didn't fit that
    # video's own pixel_xy. That indicates a real data/logic problem and must
    # fail-fast rather than being silently folded into any option.
    bounds_violations = [
        row for row in per_video if row["dimension_status"] == "ok" and row["pixel_xy_bounds_ok"] is not True
    ]
    require(
        not bounds_violations,
        "pixel_xy out of the resolved intermediate-H5 dimension bounds for "
        f"{len(bounds_violations)} video(s): {[row['video_name'] for row in bounds_violations][:5]}",
    )

    resolved = [
        row for row in per_video if row["dimension_status"] == "ok" and row["pixel_xy_bounds_ok"] is True
    ]
    unresolved = [row for row in per_video if row not in resolved]

    if len(resolved) == len(per_video) and len(per_video) > 0:
        return {
            "option": 3,
            "description": "Per-video real dimensions resolved for every video from intermediate H5 provenance.",
            "resolved_count": len(resolved),
            "total_count": len(per_video),
            "common_dimension": None,
            "excluded_video_names": [],
        }

    resolved_dims = {(row["width"], row["height"]) for row in resolved}
    if len(resolved) > 0 and len(resolved_dims) == 1:
        common_width, common_height = next(iter(resolved_dims))
        bounds_checked_for_unresolved = [
            row
            for row in unresolved
            if row["pixel_xy_max_x"] is not None
            and row["pixel_xy_max_x"] < common_width
            and row["pixel_xy_min_x"] is not None
            and row["pixel_xy_min_x"] >= 0
            and row["pixel_xy_max_y"] is not None
            and row["pixel_xy_max_y"] < common_height
            and row["pixel_xy_min_y"] is not None
            and row["pixel_xy_min_y"] >= 0
        ]
        excluded = [row["video_name"] for row in unresolved if row not in bounds_checked_for_unresolved]
        if bounds_checked_for_unresolved:
            return {
                "option": 2,
                "description": (
                    "Common local-crop dimension confirmed across all individually-resolved videos; "
                    "extended to unresolved videos only where pixel_xy strictly fits the same bounds."
                ),
                "resolved_count": len(resolved),
                "bounds_confirmed_count": len(bounds_checked_for_unresolved),
                "total_count": len(per_video),
                "common_dimension": {"width": common_width, "height": common_height},
                "excluded_video_names": excluded,
            }
        return {
            "option": 4,
            "description": (
                "A common dimension is confirmed among resolved videos, but no unresolved video's "
                "pixel_xy could be bounds-verified against it; excluding all unresolved videos."
            ),
            "resolved_count": len(resolved),
            "total_count": len(per_video),
            "common_dimension": {"width": common_width, "height": common_height},
            "excluded_video_names": [row["video_name"] for row in unresolved],
        }

    return {
        "option": 4,
        "description": "No dimension could be resolved or corroborated for enough videos; cross-video XY analysis not adopted.",
        "resolved_count": len(resolved),
        "total_count": len(per_video),
        "common_dimension": None,
        "excluded_video_names": [row["video_name"] for row in per_video],
    }


# ----------------------------------------------------------------------------
# Step H2.5-D: normalization with fail-fast bounds checking
# ----------------------------------------------------------------------------


def normalize_pixel_xy_with_dimensions(pixel_xy: np.ndarray, *, width: int, height: int) -> np.ndarray:
    require(width > 1, f"width must be > 1, got {width}")
    require(height > 1, f"height must be > 1, got {height}")
    require(bool(np.all(np.isfinite(pixel_xy))), "pixel_xy contains non-finite value(s)")
    x = pixel_xy[:, 0]
    y = pixel_xy[:, 1]
    require(bool(np.all((x >= 0) & (x < width))), f"pixel_xy x out of bounds [0, {width})")
    require(bool(np.all((y >= 0) & (y < height))), f"pixel_xy y out of bounds [0, {height})")
    normalized_x = x / (width - 1)
    normalized_y = y / (height - 1)
    return np.stack([normalized_x, normalized_y], axis=1)


# ----------------------------------------------------------------------------
# Step H2.5-E: entry point and artifacts
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-14 Step H2.5: Stage 4 dataset inventory and XY coordinate provenance audit."
    )
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument(
        "--stage4_training_ablation_root",
        default=None,
        help="e.g. /mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711",
    )
    parser.add_argument(
        "--pseudo3d_outputs_root",
        default=None,
        help="Fallback directory for intermediate per-video H5s, e.g. "
        "/mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_outputs/260711",
    )
    parser.add_argument("--fallback_suffix", default="_ts448_oym96_corr.h5")
    parser.add_argument("--private_inventory_csv", default=None)
    parser.add_argument("--private_source_resolution_csv", default=None)
    parser.add_argument("--shareable_summary_json", default=None)
    parser.add_argument("--shareable_dimension_audit_csv", default=None)
    return parser.parse_args()


def anonymize_video_name(index: int, split: str) -> str:
    return f"{split}_{index:03d}"


def main() -> None:
    args = parse_args()
    require(bool(args.train_list) or bool(args.val_list), "Provide --train_list and/or --val_list")

    entries: list[tuple[str, Path]] = []
    for split, list_path in (("train", args.train_list), ("val", args.val_list)):
        if list_path:
            for path in read_path_list(list_path):
                entries.append((split, path))
    if args.max_files > 0:
        entries = entries[: args.max_files]
    require(bool(entries), "No H5 paths resolved from the given list(s)")

    inventory_rows: list[dict[str, Any]] = []
    if args.stage4_training_ablation_root:
        inventory_rows = scan_dataset_inventory(Path(args.stage4_training_ablation_root))

    fallback_root = Path(args.pseudo3d_outputs_root) if args.pseudo3d_outputs_root else Path(".")
    per_video: list[dict[str, Any]] = []
    for split, path in entries:
        require(path.is_file(), f"H5 not found: {path}")
        result = audit_video_provenance(path, fallback_root=fallback_root, fallback_suffix=args.fallback_suffix)
        result["split"] = split
        result["h5_path"] = str(path)
        per_video.append(result)

    policy = decide_dimension_policy(per_video)

    for index, row in enumerate(per_video):
        row["video_alias"] = anonymize_video_name(index, row["split"])

    resolved_methods = {row["resolution_method"] for row in per_video}
    summary = {
        "status": "passed",
        "num_videos": len(per_video),
        "dataset_inventory_run_count": len(inventory_rows),
        "resolution_methods_seen": sorted(resolved_methods),
        "resolved_dimension_count": sum(1 for row in per_video if row["dimension_status"] == "ok"),
        "policy": policy,
    }

    if args.private_inventory_csv and inventory_rows:
        output = Path(args.private_inventory_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(inventory_rows[0].keys()))
            writer.writeheader()
            writer.writerows(inventory_rows)

    if args.private_source_resolution_csv:
        output = Path(args.private_source_resolution_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(per_video[0].keys()) if per_video else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_video)

    if args.shareable_dimension_audit_csv:
        share_fields = [
            "video_alias",
            "split",
            "resolution_method",
            "dimension_status",
            "width",
            "height",
            "pixel_xy_bounds_ok",
        ]
        output = Path(args.shareable_dimension_audit_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=share_fields)
            writer.writeheader()
            for row in per_video:
                writer.writerow({key: row.get(key) for key in share_fields})

    if args.shareable_summary_json:
        output = Path(args.shareable_summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-14 Step H2.5 XY coordinate provenance audit passed.")
    print(f"videos audited: {summary['num_videos']}")
    print(f"resolved dimension count: {summary['resolved_dimension_count']} / {summary['num_videos']}")
    print(f"resolution methods seen: {summary['resolution_methods_seen']}")
    print(f"decision: option {policy['option']} -- {policy['description']}")
    print(f"excluded videos: {len(policy['excluded_video_names'])}")
    if args.shareable_summary_json:
        print(f"shareable summary: {args.shareable_summary_json}")
    if args.private_source_resolution_csv:
        print(f"private source resolution csv: {args.private_source_resolution_csv}")


if __name__ == "__main__":
    main()
