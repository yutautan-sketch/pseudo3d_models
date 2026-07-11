from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


S3DIS_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
)


def summarize_array(path: Path) -> dict:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] < 7:
        raise AssertionError(f"{path} must have shape [N, >=7], got {data.shape}")

    xyz = np.asarray(data[:, :3])
    rgb = np.asarray(data[:, 3:6])
    labels = np.asarray(data[:, 6]).astype(np.int64)
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    if unique_labels.size == 0:
        raise AssertionError(f"{path} has no labels")
    if unique_labels.min() < 0 or unique_labels.max() >= len(S3DIS_CLASSES):
        raise AssertionError(
            f"{path} labels outside 0..{len(S3DIS_CLASSES) - 1}: "
            f"min={unique_labels.min()} max={unique_labels.max()}"
        )
    if not np.isfinite(xyz).all():
        raise AssertionError(f"{path} xyz contains NaN/Inf")
    if not np.isfinite(rgb).all():
        raise AssertionError(f"{path} rgb contains NaN/Inf")

    return {
        "path": path,
        "shape": tuple(int(v) for v in data.shape),
        "xyz_min": xyz.min(axis=0),
        "xyz_max": xyz.max(axis=0),
        "rgb_min": rgb.min(axis=0),
        "rgb_max": rgb.max(axis=0),
        "labels": dict(zip(unique_labels.tolist(), label_counts.tolist())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect downloaded S3DIS presampled dataset layout")
    parser.add_argument(
        "--data_root",
        default="Stage5/external/PointNeXt/data/S3DIS/s3disfull",
        help="Path containing raw/ and processed/ directories",
    )
    parser.add_argument("--area", default="5", help="Area number to sample, e.g. 5")
    parser.add_argument("--max_files", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"

    if not data_root.is_dir():
        raise FileNotFoundError(f"S3DIS data root not found: {data_root}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"S3DIS raw directory not found: {raw_dir}")
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"S3DIS processed directory not found: {processed_dir}")

    all_files = sorted(raw_dir.glob("Area_*.npy"))
    area_files = sorted(raw_dir.glob(f"Area_{args.area}_*.npy"))
    processed_files = sorted(processed_dir.glob("*.pkl"))

    if not all_files:
        raise FileNotFoundError(f"No raw S3DIS .npy files found in {raw_dir}")
    if not area_files:
        raise FileNotFoundError(f"No Area_{args.area} .npy files found in {raw_dir}")

    print("S3DIS layout check")
    print(f"data_root       : {data_root}")
    print(f"raw files       : {len(all_files)}")
    print(f"Area_{args.area} files  : {len(area_files)}")
    print(f"processed files : {len(processed_files)}")
    for path in processed_files[:10]:
        print(f"  processed: {path.name}")

    area_counter = Counter(path.name.split("_")[1] for path in all_files if path.name.startswith("Area_"))
    print("area counts:")
    for area, count in sorted(area_counter.items(), key=lambda item: int(item[0])):
        print(f"  Area_{area}: {count}")

    sample_files = area_files[: max(1, args.max_files)]
    total_points = 0
    aggregate_labels: Counter[int] = Counter()

    print("sample raw files:")
    for path in sample_files:
        summary = summarize_array(path)
        total_points += summary["shape"][0]
        aggregate_labels.update(summary["labels"])
        xyz_min = ", ".join(f"{v:.3f}" for v in summary["xyz_min"])
        xyz_max = ", ".join(f"{v:.3f}" for v in summary["xyz_max"])
        rgb_min = ", ".join(f"{v:.1f}" for v in summary["rgb_min"])
        rgb_max = ", ".join(f"{v:.1f}" for v in summary["rgb_max"])
        print(f"  {path.name}: shape={summary['shape']}")
        print(f"    xyz min/max: [{xyz_min}] / [{xyz_max}]")
        print(f"    rgb min/max: [{rgb_min}] / [{rgb_max}]")
        label_text = ", ".join(
            f"{label}:{S3DIS_CLASSES[label]}={count}"
            for label, count in sorted(summary["labels"].items())
        )
        print(f"    labels: {label_text}")

    print(f"sampled points total: {total_points}")
    aggregate_text = ", ".join(
        f"{label}:{S3DIS_CLASSES[label]}={count}"
        for label, count in sorted(aggregate_labels.items())
    )
    print(f"sampled label aggregate: {aggregate_text}")
    print("S3DIS data layout check passed.")


if __name__ == "__main__":
    main()
