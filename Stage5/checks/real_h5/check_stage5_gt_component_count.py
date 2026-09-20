from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# S5-15: count how many separate GT femur regions appear per frame.
#
# The qualitative review of the S5-15 evaluation reported that videos showing
# the femur in "two places" are the ones the model fails on. That grouping was
# made by eye. This replaces it with a mechanical count: GT-positive points are
# clustered per frame in `pixel_xy` space by single linkage, so a "region" is a
# set of GT points no further than `link_distance` from another point in the
# same set -- the same thing an observer sees as one blob in the rendered frame.
#
# The count depends on `link_distance`, so a single value is never reported on
# its own: the CLI sweeps several radii and records how the classification
# changes, and the chosen threshold and minimum region size are written into
# the output. A finding that survives only one radius is not a finding.
#
# Reads the teacher H5s and the evaluation's own private alias map; it never
# needs the model, the predictions or the intermediate H5, and writes nothing
# outside its own output paths. numpy/h5py only: no torch, no CUDA.
# ----------------------------------------------------------------------------

DEFAULT_LINK_DISTANCES = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_MIN_COMPONENT_POINTS = 5
DEFAULT_MULTI_REGION_FRAME_FRACTION = 0.5
MAX_POINTS_PER_FRAME = 20000

TIMESTAMP_VIDEO_ID_PATTERN = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9]+")
ABSOLUTE_HOST_PATH_PATTERN = re.compile(r"(/mnt/data|/home/[A-Za-z0-9_.-]+)")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != root:
            self.parent[index], index = root, self.parent[index]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def component_sizes(points_xy: np.ndarray, *, link_distance: float) -> list[int]:
    """Single-linkage component sizes for one frame's GT points.

    Two points join the same component when their Euclidean distance in pixel
    space is <= link_distance; components are the transitive closure of that
    relation, which is what reads as one connected blob on the frame image.
    """
    count = int(points_xy.shape[0])
    if count == 0:
        return []
    if count == 1:
        return [1]
    if count > MAX_POINTS_PER_FRAME:
        raise ValueError(
            f"{count} GT points in one frame exceeds the {MAX_POINTS_PER_FRAME} supported here; "
            "raise MAX_POINTS_PER_FRAME deliberately rather than silently subsampling"
        )

    coords = points_xy.astype(np.float64)
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    close = distances <= float(link_distance)
    np.fill_diagonal(close, False)

    union_find = UnionFind(count)
    for i, j in zip(*np.nonzero(np.triu(close))):
        union_find.union(int(i), int(j))

    counts: dict[int, int] = {}
    for index in range(count):
        root = union_find.find(index)
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.values(), reverse=True)


def read_gt_points(h5_path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        pixel_xy = f["point_cloud/pixel_xy"][:].astype(np.float64)
        frame_order = f["point_cloud/frame_order"][:].astype(np.int64)
        point_label = f["annotation/point_label"][:].astype(np.int64)
        valid_mask = f["annotation/valid_mask"][:].astype(bool)
    for name, array in (("frame_order", frame_order), ("point_label", point_label), ("valid_mask", valid_mask)):
        if array.shape[0] != pixel_xy.shape[0]:
            raise ValueError(f"{h5_path.name}: {name} has {array.shape[0]} rows, pixel_xy has {pixel_xy.shape[0]}")
    return {
        "pixel_xy": pixel_xy,
        "frame_order": frame_order,
        "gt_positive": valid_mask & (point_label == 1),
    }


def analyze_video(
    data: dict[str, np.ndarray],
    *,
    link_distance: float,
    min_component_points: int,
) -> dict[str, Any]:
    """Per-frame region counts for one video at one link distance."""
    pixel_xy = data["pixel_xy"]
    frame_order = data["frame_order"]
    gt = data["gt_positive"]

    per_frame: list[dict[str, Any]] = []
    for frame in np.unique(frame_order[gt]):
        mask = gt & (frame_order == frame)
        sizes = component_sizes(pixel_xy[mask], link_distance=link_distance)
        kept = [size for size in sizes if size >= min_component_points]
        per_frame.append(
            {
                "frame_order": int(frame),
                "gt_points": int(mask.sum()),
                "num_regions_all": len(sizes),
                "num_regions": len(kept),
                "largest_region_points": sizes[0] if sizes else 0,
                "second_region_points": sizes[1] if len(sizes) > 1 else 0,
            }
        )

    counts = [row["num_regions"] for row in per_frame]
    multi = [c for c in counts if c >= 2]
    return {
        "link_distance": link_distance,
        "min_component_points": min_component_points,
        "num_gt_frames": len(per_frame),
        "median_regions_per_frame": statistics.median(counts) if counts else None,
        "max_regions_per_frame": max(counts) if counts else None,
        "multi_region_frames": len(multi),
        "multi_region_frame_fraction": (len(multi) / len(counts)) if counts else None,
        "per_frame": per_frame,
    }


def classify_video(sweep: list[dict[str, Any]], *, frame_fraction_threshold: float) -> dict[str, Any]:
    """Label a video multi-region, and say whether that survives every radius."""
    labels: dict[float, bool] = {}
    any_frame: dict[float, bool] = {}
    for entry in sweep:
        fraction = entry["multi_region_frame_fraction"]
        labels[entry["link_distance"]] = bool(fraction is not None and fraction >= frame_fraction_threshold)
        any_frame[entry["link_distance"]] = bool(fraction is not None and fraction > 0.0)
    values = set(labels.values())
    return {
        "multi_region_by_link_distance": {str(k): v for k, v in labels.items()},
        "stable_across_link_distances": len(values) == 1,
        # These two are about the RADIUS sweep: whether the fraction reached
        # frame_fraction_threshold at any / every link distance.
        "multi_region_any": any(labels.values()),
        "multi_region_all": all(labels.values()) if labels else False,
        # This one is about FRAMES: does the video contain even one frame with
        # two or more GT regions. Reported separately because the S5-15 data is
        # bimodal on exactly this -- videos sit at either 0% of frames or >=11%,
        # with nothing in between -- so it separates videos without depending on
        # where frame_fraction_threshold is placed inside that empty range.
        "multi_region_any_frame": any(any_frame.values()),
        "multi_region_any_frame_all_radii": all(any_frame.values()) if any_frame else False,
    }


def read_video_id_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"anonymous_id", "split", "original_h5_path"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{path} lacks column(s): {sorted(missing)}")
    return rows


def read_metrics_recall(path: Path, *, checkpoint: str) -> dict[str, dict[str, float]]:
    """alias -> {recall, f1, gt_positive} for one checkpoint."""
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("checkpoint") != checkpoint:
                continue
            out[row["video_name"]] = {
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "gt_positive": float(row["valid_positive_count"]),
            }
    return out


def group_comparison(
    videos: list[dict[str, Any]],
    metrics: dict[str, dict[str, float]],
    *,
    label_key: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, float]]] = {"multi_region": [], "single_region": []}
    for video in videos:
        entry = metrics.get(video["anonymous_id"])
        if entry is None:
            continue
        groups["multi_region" if video["classification"][label_key] else "single_region"].append(entry)

    summary: dict[str, Any] = {"label_key": label_key}
    for name, entries in groups.items():
        if not entries:
            summary[name] = {"num_videos": 0}
            continue
        summary[name] = {
            "num_videos": len(entries),
            "recall_median": statistics.median(e["recall"] for e in entries),
            "recall_mean": statistics.mean(e["recall"] for e in entries),
            "f1_median": statistics.median(e["f1"] for e in entries),
            "gt_positive_median": statistics.median(e["gt_positive"] for e in entries),
        }
    both = summary["multi_region"].get("num_videos", 0) and summary["single_region"].get("num_videos", 0)
    summary["recall_median_difference"] = (
        summary["multi_region"]["recall_median"] - summary["single_region"]["recall_median"] if both else None
    )
    summary["note"] = (
        "Grouping comes from the GT geometry alone; the recall figures come from the evaluation. "
        "A difference here is an association, not a demonstrated cause."
    )
    return summary


def privacy_self_check(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    return {
        "timestamp_like_video_ids_absent": not TIMESTAMP_VIDEO_ID_PATTERN.findall(text),
        "absolute_host_paths_absent": not ABSOLUTE_HOST_PATH_PATTERN.findall(text),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count GT femur regions per frame from the teacher H5s, replacing the by-eye "
        "'two places' grouping with a mechanical one. Sweeps the link distance so the result is "
        "not an artifact of a single radius. Read-only, CPU only."
    )
    parser.add_argument(
        "--video_id_map",
        required=True,
        help="video_id_map_DO_NOT_SHARE.csv from the evaluation's private directory "
        "(reused so no new numbering scheme is invented)",
    )
    parser.add_argument("--split", default="validation", help="Split to analyze; 'all' for every row")
    parser.add_argument(
        "--link_distances",
        default=",".join(str(d) for d in DEFAULT_LINK_DISTANCES),
        help="Comma-separated pixel distances to sweep",
    )
    parser.add_argument("--min_component_points", type=int, default=DEFAULT_MIN_COMPONENT_POINTS)
    parser.add_argument("--frame_fraction_threshold", type=float, default=DEFAULT_MULTI_REGION_FRAME_FRACTION)
    parser.add_argument("--metrics_csv", default=None, help="Anonymized h5_metrics CSV for the recall comparison")
    parser.add_argument("--metrics_checkpoint", default="best")
    parser.add_argument("--private_json", default=None)
    parser.add_argument("--shareable_json", default=None)
    parser.add_argument("--per_frame_csv", default=None, help="Optional per-frame detail (private: real names)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    link_distances = [float(x.strip()) for x in args.link_distances.split(",") if x.strip()]
    if not link_distances:
        print("FAIL: no link distances given", file=sys.stderr)
        sys.exit(2)

    rows = read_video_id_map(Path(args.video_id_map))
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        print(f"FAIL: no videos for split {args.split!r}", file=sys.stderr)
        sys.exit(2)

    videos: list[dict[str, Any]] = []
    per_frame_records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: r["anonymous_id"]):
        h5_path = Path(row["original_h5_path"])
        if not h5_path.is_file():
            print(f"FAIL: H5 not found for {row['anonymous_id']}: {h5_path}", file=sys.stderr)
            sys.exit(2)
        data = read_gt_points(h5_path)
        sweep = [
            analyze_video(data, link_distance=d, min_component_points=args.min_component_points)
            for d in link_distances
        ]
        classification = classify_video(sweep, frame_fraction_threshold=args.frame_fraction_threshold)
        videos.append(
            {
                "anonymous_id": row["anonymous_id"],
                "split": row["split"],
                "original_video_name": row.get("original_video_name", ""),
                "total_gt_points": int(data["gt_positive"].sum()),
                "sweep": [{k: v for k, v in entry.items() if k != "per_frame"} for entry in sweep],
                "classification": classification,
            }
        )
        for entry in sweep:
            for frame_row in entry["per_frame"]:
                per_frame_records.append(
                    {
                        "anonymous_id": row["anonymous_id"],
                        "original_video_name": row.get("original_video_name", ""),
                        "link_distance": entry["link_distance"],
                        **frame_row,
                    }
                )

    comparison = None
    if args.metrics_csv:
        metrics = read_metrics_recall(Path(args.metrics_csv), checkpoint=args.metrics_checkpoint)
        comparison = {
            key: group_comparison(videos, metrics, label_key=key)
            for key in ("multi_region_all", "multi_region_any", "multi_region_any_frame")
        }

    unstable = [v["anonymous_id"] for v in videos if not v["classification"]["stable_across_link_distances"]]
    summary = {
        "split": args.split,
        "link_distances": link_distances,
        "min_component_points": args.min_component_points,
        "frame_fraction_threshold": args.frame_fraction_threshold,
        "num_videos": len(videos),
        "num_unstable_classifications": len(unstable),
        "unstable_videos": unstable,
        "videos": videos,
        "recall_comparison": comparison,
        "metrics_checkpoint": args.metrics_checkpoint if args.metrics_csv else None,
    }

    shareable = json.loads(json.dumps(summary, ensure_ascii=False))
    for video in shareable["videos"]:
        video.pop("original_video_name", None)
    shareable["privacy_self_check"] = privacy_self_check(shareable)

    for path_str, payload in ((args.private_json, summary), (args.shareable_json, shareable)):
        if path_str:
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_frame_csv and per_frame_records:
        path = Path(args.per_frame_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_frame_records[0]))
            writer.writeheader()
            writer.writerows(per_frame_records)

    print("Stage5 GT region-count check")
    print(f"  split            : {args.split} ({len(videos)} videos)")
    print(f"  link distances   : {link_distances} px")
    print(f"  min region points: {args.min_component_points}")
    print(f"  multi-region rule: >= {args.frame_fraction_threshold:.0%} of GT frames have >= 2 regions")
    print(f"  unstable across radii: {len(unstable)}" + (f" {unstable}" if unstable else ""))
    print()
    header = "  ".join(f"d={d:g}" for d in link_distances)
    print(f"  {'alias':>24} {'GT frames':>9}  {header}   multi(all/any)")
    for video in videos:
        fractions = "  ".join(
            f"{(e['multi_region_frame_fraction'] or 0) * 100:5.0f}%" for e in video["sweep"]
        )
        cls = video["classification"]
        print(
            f"  {video['anonymous_id']:>24} {video['sweep'][0]['num_gt_frames']:>9}  {fractions}"
            f"   {'Y' if cls['multi_region_all'] else 'n'}/{'Y' if cls['multi_region_any'] else 'n'}"
        )
    if comparison:
        for key, group in comparison.items():
            multi, single = group["multi_region"], group["single_region"]
            print(f"\n  [{key}] recall median  multi-region: ", end="")
            print(
                f"{multi.get('recall_median', float('nan')) * 100:.2f}% (n={multi['num_videos']})"
                f"   single-region: {single.get('recall_median', float('nan')) * 100:.2f}% (n={single['num_videos']})"
                if multi["num_videos"] and single["num_videos"]
                else "n/a (one group is empty)"
            )
        print(f"\n  {comparison['multi_region_all']['note']}")
    if args.shareable_json:
        print(f"\n  privacy: {shareable['privacy_self_check']}")


if __name__ == "__main__":
    main()
