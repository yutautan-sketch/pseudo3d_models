from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from stage5.utils.visualization_export import write_probability_ply


DEFAULT_FIXED_TRAIN_VIDEO = "20250626_124212_7300"


def read_path_list(path: Path) -> list[Path]:
    paths: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            item = Path(token)
            paths.append(item if item.is_absolute() else path.parent / item)
    if not paths:
        raise ValueError(f"Path list is empty: {path}")
    return paths


def write_path_list(path: Path, paths: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in paths), encoding="utf-8")


def select_train_paths(
    paths: list[Path],
    *,
    fixed_videos: Iterable[str],
    num_random: int,
    seed: int,
) -> list[Path]:
    selected: list[Path] = []
    for video in fixed_videos:
        matches = [path for path in paths if video in path.name]
        if len(matches) != 1:
            raise ValueError(
                f"Fixed train video {video!r} matched {len(matches)} paths in train list"
            )
        if matches[0] not in selected:
            selected.append(matches[0])

    candidates = [path for path in paths if path not in selected]
    if num_random < 0:
        raise ValueError("--num_random_train must be non-negative")
    if num_random > len(candidates):
        raise ValueError(
            f"Requested {num_random} random train files but only {len(candidates)} are available"
        )
    selected.extend(random.Random(seed).sample(candidates, num_random))
    return selected


def video_name_from_path(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("video_name", "")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip()
    if value:
        return value

    match = re.search(r"\d{8}_\d{6}_\d+", path.name)
    if not match:
        raise ValueError(f"Could not extract video name from H5 filename: {path}")
    return match.group(0)


def _summary_reference_matches(root: Path, video_name: str) -> list[Path]:
    summary_path = root / "summary.csv"
    if not summary_path.is_file():
        return []

    matches: set[Path] = set()
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            input_h5 = str(row.get("input_h5", ""))
            output_ply = str(row.get("output_ply", ""))
            if video_name not in input_h5 and video_name not in output_ply:
                continue
            output_path = Path(output_ply)
            if not output_path.is_absolute():
                output_path = root / output_path
            if output_path.is_file():
                matches.add(output_path)
    return sorted(matches)


@lru_cache(maxsize=None)
def find_reference_ply(root: Path, *, video_name: str, annotated: bool) -> Path:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Reference PLY directory not found: {root}")

    summary_matches = _summary_reference_matches(root, video_name)
    if len(summary_matches) == 1:
        return summary_matches[0]
    if len(summary_matches) > 1:
        raise FileNotFoundError(
            f"Stage 4 summary.csv maps {video_name} to multiple PLY files: "
            f"{summary_matches}"
        )

    if annotated:
        pattern = f"{video_name}_pointcloud_annotated_foreground*.ply"
    else:
        pattern = f"{video_name}_pointcloud_foreground*.ply"
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) == 1:
        return matches[0]

    relative_matches = sorted(
        path
        for path in root.rglob("*.ply")
        if path.is_file() and video_name in str(path.relative_to(root))
    )
    if len(relative_matches) == 1:
        return relative_matches[0]

    kind = "annotated" if annotated else "raw"
    sample_paths = [str(path.relative_to(root)) for path in sorted(root.rglob("*.ply"))[:5]]
    raise FileNotFoundError(
        f"Expected exactly one {kind} reference PLY for {video_name} under {root}; "
        f"summary matches={len(summary_matches)}, filename matches={len(matches)}, "
        f"relative-path matches={len(relative_matches)}. Sample PLY paths={sample_paths}"
    )


def copy_reference_ply(
    source: Path,
    destination_dir: Path,
    *,
    video_name: str,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_name = (
        source.name if video_name in source.name else f"{video_name}_{source.name}"
    )
    destination = destination_dir / destination_name
    shutil.copy2(source, destination)
    return destination


def write_gt_positive_ply(h5_path: Path, output_path: Path) -> int:
    with h5py.File(h5_path, "r") as handle:
        required = ("point_cloud/points", "annotation/point_label", "annotation/valid_mask")
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"Missing GT positive PLY datasets in {h5_path}: {missing}")
        points = handle["point_cloud/points"][:].astype(np.float32)
        labels = handle["annotation/point_label"][:].astype(np.int64)
        valid_mask = handle["annotation/valid_mask"][:].astype(bool)
    if points.shape[0] != labels.shape[0] or labels.shape != valid_mask.shape:
        raise ValueError(f"Point/label/valid length mismatch in {h5_path}")

    selected = valid_mask & (labels == 1)
    selected_points = points[selected]
    write_probability_ply(
        output_path,
        selected_points,
        np.ones((selected_points.shape[0],), dtype=np.float32),
    )
    return int(selected_points.shape[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    reference_output_dir = Path(args.reference_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = read_path_list(Path(args.train_list))
    val_paths = read_path_list(Path(args.val_list))
    for path in (*train_paths, *val_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation H5 not found: {path}")
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(f"Train and validation lists overlap: {sorted(overlap)[:3]}")

    fixed_videos = args.fixed_train_video or [DEFAULT_FIXED_TRAIN_VIDEO]
    selected_train = select_train_paths(
        train_paths,
        fixed_videos=fixed_videos,
        num_random=args.num_random_train,
        seed=args.selection_seed,
    )
    selected_train_list = output_dir / "selected_train_files.txt"
    validation_list = output_dir / "validation_files.txt"
    write_path_list(selected_train_list, selected_train)
    write_path_list(validation_list, val_paths)

    rows: list[dict[str, object]] = []
    seen_videos: set[str] = set()
    split_paths = (("train_sanity", selected_train), ("validation", val_paths))
    for split, paths in split_paths:
        for h5_path in paths:
            video_name = video_name_from_path(h5_path)
            if video_name in seen_videos:
                raise ValueError(f"Duplicate evaluation video: {video_name}")
            seen_videos.add(video_name)

            raw_source = find_reference_ply(
                Path(args.raw_reference_ply_dir),
                video_name=video_name,
                annotated=False,
            )
            annotated_source = find_reference_ply(
                Path(args.annotated_reference_ply_dir),
                video_name=video_name,
                annotated=True,
            )
            raw_output = copy_reference_ply(
                raw_source,
                reference_output_dir / split / "pointcloud_foreground",
                video_name=video_name,
            )
            annotated_output = copy_reference_ply(
                annotated_source,
                reference_output_dir / split / "pointcloud_annotated_foreground",
                video_name=video_name,
            )
            gt_positive_output = (
                reference_output_dir
                / split
                / "ground_truth_positive"
                / f"{video_name}_ground_truth_positive.ply"
            )
            gt_positive_count = write_gt_positive_ply(h5_path, gt_positive_output)
            rows.append(
                {
                    "split": split,
                    "video_name": video_name,
                    "h5_path": str(h5_path),
                    "raw_ply_source": str(raw_source),
                    "raw_ply_output": str(raw_output),
                    "annotated_ply_source": str(annotated_source),
                    "annotated_ply_output": str(annotated_output),
                    "gt_positive_ply_output": str(gt_positive_output),
                    "gt_positive_point_count": gt_positive_count,
                }
            )

    manifest_csv = output_dir / "reference_ply_manifest.csv"
    write_csv(manifest_csv, rows)
    summary = {
        "train_list": str(Path(args.train_list)),
        "val_list": str(Path(args.val_list)),
        "selected_train_files": [str(path) for path in selected_train],
        "validation_files": [str(path) for path in val_paths],
        "fixed_train_videos": fixed_videos,
        "num_random_train": args.num_random_train,
        "selection_seed": args.selection_seed,
        "raw_reference_ply_dir": str(Path(args.raw_reference_ply_dir)),
        "annotated_reference_ply_dir": str(Path(args.annotated_reference_ply_dir)),
        "reference_output_dir": str(reference_output_dir),
        "num_reference_videos": len(rows),
        "manifest_csv": str(manifest_csv),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Stage5 evaluation data preparation passed.")
    print(f"train sanity files: {len(selected_train)}")
    print(f"validation files: {len(val_paths)}")
    print(f"reference PLY root: {reference_output_dir}")
    print(f"manifest: {manifest_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Stage5 evaluation data and collect corresponding reference PLY files"
    )
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--raw_reference_ply_dir", required=True)
    parser.add_argument("--annotated_reference_ply_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_output_dir", required=True)
    parser.add_argument("--fixed_train_video", action="append", default=None)
    parser.add_argument("--num_random_train", type=int, default=2)
    parser.add_argument("--selection_seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
