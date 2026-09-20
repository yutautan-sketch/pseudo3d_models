from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from checks.real_h5.check_stage5_class_weight_threshold_free import load_prediction, safe_name
from checks.real_h5.check_stage5_frame_xy_diagnostics import (
    frame_confusion_counts,
    frame_probability_stats,
)
from checks.real_h5.check_stage5_gt_component_count import (
    ABSOLUTE_HOST_PATH_PATTERN,
    TIMESTAMP_VIDEO_ID_PATTERN,
    component_sizes,
    read_gt_points,
    read_video_id_map,
)

# ----------------------------------------------------------------------------
# S5-15 verification (2): is the failure frame-localized, and is the
# multi-region effect a property of FRAMES or only of VIDEOS?
#
# 8.12.15 established that videos whose GT ever splits into two regions have
# much lower recall than videos whose GT never does (16.65% vs 56.52% median).
# That is a video-level association, and every video-level comparison is open
# to a video-level confound: the affected videos may simply be different
# recordings. It also cannot distinguish "two regions hurt" from "these videos
# are hard for some other reason that happens to co-occur".
#
# This checker moves the comparison inside the video. Every GT-bearing frame
# gets both its own region count and its own recall, so single-region and
# multi-region frames OF THE SAME VIDEO can be compared directly. A video
# contributes only when it contains both kinds of frame, which holds the
# recording, the patient, the crop and the model fixed. The across-video
# summary of those within-video differences is then a paired comparison, and
# an exact sign test says how easily it could have arisen by chance.
#
# Reads the teacher H5s and the evaluation's already-saved predictions/*.npz.
# No re-inference, no model, no torch, no CUDA. numpy/h5py only.
# ----------------------------------------------------------------------------

DEFAULT_LINK_DISTANCE = 4.0
DEFAULT_MIN_COMPONENT_POINTS = 5
NEAR_ZERO_RECALL = 0.05


def exact_sign_test(negative: int, positive: int) -> dict[str, Any]:
    """Two-sided exact sign test on paired differences, ties already excluded.

    Pure stdlib; scipy is not a dependency of this repository. The p-value is
    the exact binomial tail under p=0.5, so it makes no normality assumption
    and is valid at the sample sizes here (n <= 18).
    """
    n = int(negative) + int(positive)
    if n == 0:
        return {"num_pairs": 0, "p_value": None, "note": "no video contained both kinds of frame"}
    extreme = min(int(negative), int(positive))
    tail = sum(math.comb(n, k) for k in range(extreme + 1)) / (2.0**n)
    return {
        "num_pairs": n,
        "num_multi_region_worse": int(negative),
        "num_multi_region_better": int(positive),
        "p_value": float(min(1.0, 2.0 * tail)),
        "note": (
            "Exact two-sided sign test over within-video differences. It tests whether the "
            "direction is consistent, not how large the effect is, and it does not establish cause."
        ),
    }


def build_frame_table(
    *,
    gt_data: dict[str, np.ndarray],
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    prob_femur: np.ndarray,
    ignore_index: int,
    link_distance: float,
    min_component_points: int,
) -> list[dict[str, Any]]:
    """One row per frame that carries at least one valid GT-positive point."""
    frame_order = np.asarray(gt_data["frame_order"])
    pixel_xy = np.asarray(gt_data["pixel_xy"])
    gt_positive = np.asarray(gt_data["gt_positive"], dtype=bool)

    confusion = frame_confusion_counts(
        frame_order, point_label, valid_mask, pred_label, ignore_index=ignore_index
    )
    probability = frame_probability_stats(
        frame_order, point_label, valid_mask, prob_femur, ignore_index=ignore_index
    )

    rows: list[dict[str, Any]] = []
    for frame in np.unique(frame_order[gt_positive]).tolist():
        frame = int(frame)
        counts = confusion[frame]
        gt_count = counts["tp"] + counts["fn"]
        if gt_count == 0:
            # gt_positive said this frame has GT, so the two must agree; a
            # mismatch means the masks were built from different definitions.
            raise ValueError(
                f"frame {frame}: gt_positive selects points but tp+fn is 0; "
                "the GT masks disagree, which would silently corrupt the comparison"
            )
        sizes = component_sizes(pixel_xy[gt_positive & (frame_order == frame)], link_distance=link_distance)
        kept = [size for size in sizes if size >= min_component_points]
        stats = probability.get(frame, {})
        positive_stats = stats.get("positive", {})
        background_stats = stats.get("background", {})
        rows.append(
            {
                "frame_order": frame,
                "gt_positive_points": int(gt_count),
                "num_regions": len(kept),
                "num_regions_all": len(sizes),
                "largest_region_points": sizes[0] if sizes else 0,
                "true_positive": counts["tp"],
                "false_positive": counts["fp"],
                "false_negative": counts["fn"],
                "recall": counts["tp"] / gt_count,
                "mean_prob_on_gt_positive": positive_stats.get("mean"),
                "mean_prob_on_gt_background": background_stats.get("mean"),
            }
        )
    return rows


def summarize_frame_localization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Question 1: within a video, is recall all-or-nothing across frames?"""
    recalls = [row["recall"] for row in rows]
    if not recalls:
        return {"num_gt_frames": 0}
    zero = [r for r in recalls if r == 0.0]
    near_zero = [r for r in recalls if r <= NEAR_ZERO_RECALL]
    return {
        "num_gt_frames": len(recalls),
        "recall_median": statistics.median(recalls),
        "recall_min": min(recalls),
        "recall_max": max(recalls),
        "frames_with_zero_recall": len(zero),
        "fraction_frames_zero_recall": len(zero) / len(recalls),
        "frames_with_near_zero_recall": len(near_zero),
        "fraction_frames_near_zero_recall": len(near_zero) / len(recalls),
        "total_false_positive": sum(row["false_positive"] for row in rows),
        "frames_with_any_false_positive": sum(1 for row in rows if row["false_positive"] > 0),
    }


def within_video_region_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Question 2: inside one video, do multi-region frames do worse?

    Returns `comparable: False` when the video has only one kind of frame; such
    a video is excluded from the paired test rather than counted as a tie,
    because it carries no within-video evidence either way.
    """
    single = [row for row in rows if row["num_regions"] == 1]
    multi = [row for row in rows if row["num_regions"] >= 2]
    result: dict[str, Any] = {
        "num_single_region_frames": len(single),
        "num_multi_region_frames": len(multi),
        "comparable": bool(single and multi),
    }
    for name, subset in (("single_region", single), ("multi_region", multi)):
        if not subset:
            result[name] = None
            continue
        probs = [row["mean_prob_on_gt_positive"] for row in subset if row["mean_prob_on_gt_positive"] is not None]
        result[name] = {
            "recall_median": statistics.median(row["recall"] for row in subset),
            "recall_mean": statistics.mean(row["recall"] for row in subset),
            "gt_points_median": statistics.median(row["gt_positive_points"] for row in subset),
            "false_positive_median": statistics.median(row["false_positive"] for row in subset),
            # Threshold-free: unaffected by where the 0.5 cut falls.
            "mean_prob_on_gt_positive": statistics.mean(probs) if probs else None,
        }
    if result["comparable"]:
        result["recall_median_difference"] = (
            result["multi_region"]["recall_median"] - result["single_region"]["recall_median"]
        )
        multi_prob = result["multi_region"]["mean_prob_on_gt_positive"]
        single_prob = result["single_region"]["mean_prob_on_gt_positive"]
        result["mean_prob_difference"] = (
            multi_prob - single_prob if multi_prob is not None and single_prob is not None else None
        )
    else:
        result["recall_median_difference"] = None
        result["mean_prob_difference"] = None
    return result


def aggregate_paired(videos: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [v for v in videos if v["within_video"]["comparable"]]
    differences = [v["within_video"]["recall_median_difference"] for v in comparable]
    negative = sum(1 for d in differences if d < 0)
    positive = sum(1 for d in differences if d > 0)
    ties = sum(1 for d in differences if d == 0)

    prob_differences = [
        v["within_video"]["mean_prob_difference"]
        for v in comparable
        if v["within_video"]["mean_prob_difference"] is not None
    ]

    # Pooled across every GT frame of every video. Reported second and marked,
    # because pooling lets a video with many frames dominate and re-admits the
    # video-level confound the paired comparison exists to remove.
    pooled_single = [row for v in videos for row in v["frame_rows"] if row["num_regions"] == 1]
    pooled_multi = [row for v in videos for row in v["frame_rows"] if row["num_regions"] >= 2]

    def pooled_recall(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"num_frames": 0}
        tp = sum(row["true_positive"] for row in subset)
        gt = sum(row["gt_positive_points"] for row in subset)
        return {
            "num_frames": len(subset),
            "point_weighted_recall": tp / gt if gt else None,
            "frame_recall_median": statistics.median(row["recall"] for row in subset),
        }

    return {
        "videos_with_both_kinds_of_frame": len(comparable),
        "videos_excluded_as_single_kind": len(videos) - len(comparable),
        "within_video_recall_median_difference_median": (
            statistics.median(differences) if differences else None
        ),
        "num_ties": ties,
        "sign_test": exact_sign_test(negative, positive),
        "within_video_mean_prob_difference_median": (
            statistics.median(prob_differences) if prob_differences else None
        ),
        "pooled_single_region_frames": pooled_recall(pooled_single),
        "pooled_multi_region_frames": pooled_recall(pooled_multi),
        "pooled_note": (
            "Pooled figures ignore which video a frame came from, so they are NOT independent of the "
            "video-level confound. The within-video paired result above is the one that controls for it."
        ),
    }


def privacy_self_check(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    return {
        "timestamp_like_video_ids_absent": not TIMESTAMP_VIDEO_ID_PATTERN.findall(text),
        "absolute_host_paths_absent": not ABSOLUTE_HOST_PATH_PATTERN.findall(text),
    }


def resolve_checkpoint(evaluation_dir: Path, requested: str | None) -> str:
    """Take the checkpoint from the evaluation directory unless one is given.

    evaluate_stage5.py writes one directory per checkpoint, and that directory
    holds only its own rows. Asking for the checkpoint separately therefore
    invites naming one and pointing at the other, which is exactly what a
    default of "best" plus a `.../last` directory produces. So the directory
    name is the default, and an explicit value that disagrees with it stops the
    run rather than being trusted over the path.
    """
    from_dir = evaluation_dir.name
    if requested is None or requested == "":
        return from_dir
    if requested != from_dir:
        raise ValueError(
            f"--checkpoint is {requested!r} but --evaluation_dir points at {from_dir!r} "
            f"({evaluation_dir}). One of the two is wrong; refusing to guess which. "
            "Omit --checkpoint to take it from the directory."
        )
    return requested


def read_raw_h5_metrics(path: Path, *, checkpoint: str) -> dict[str, str]:
    """h5_path -> video_name, from the evaluation's raw (private) h5_metrics.csv.

    Needed only to resolve predictions/<split>/<name>.npz on disk; video_name is
    a real recording name and never reaches an output row.
    """
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("checkpoint") != checkpoint:
                continue
            mapping[row["h5_path"]] = row["video_name"]
    if not mapping:
        raise ValueError(f"{path}: no rows for checkpoint {checkpoint!r}")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the multi-region penalty is a property of frames or only of videos, "
        "by comparing single-region and multi-region frames WITHIN each video. Read-only, CPU only."
    )
    parser.add_argument("--video_id_map", required=True, help="video_id_map_DO_NOT_SHARE.csv (1-based aliases)")
    parser.add_argument("--evaluation_dir", required=True, help="evaluate_stage5.py per-checkpoint output_dir")
    parser.add_argument("--checkpoint", default=None, help="Defaults to the name of --evaluation_dir")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--link_distance", type=float, default=DEFAULT_LINK_DISTANCE)
    parser.add_argument("--min_component_points", type=int, default=DEFAULT_MIN_COMPONENT_POINTS)
    parser.add_argument("--private_json", default=None)
    parser.add_argument("--shareable_json", default=None)
    parser.add_argument("--per_frame_csv", default=None, help="Per-frame detail, aliases only (shareable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    try:
        checkpoint = resolve_checkpoint(evaluation_dir, args.checkpoint)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(2)
    rows = [r for r in read_video_id_map(Path(args.video_id_map)) if r["split"] == args.split]
    if not rows:
        print(f"FAIL: no videos for split {args.split!r}", file=sys.stderr)
        sys.exit(2)
    video_names = read_raw_h5_metrics(evaluation_dir / "h5_metrics.csv", checkpoint=checkpoint)

    from stage5.utils.h5_io import load_stage5_pointcloud_h5

    videos: list[dict[str, Any]] = []
    per_frame_records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: r["anonymous_id"]):
        alias = row["anonymous_id"]
        h5_path = Path(row["original_h5_path"])
        if not h5_path.is_file():
            print(f"FAIL: {alias}: missing teacher H5", file=sys.stderr)
            sys.exit(2)
        video_name = video_names.get(str(h5_path))
        if video_name is None:
            print(f"FAIL: {alias}: no h5_metrics row for this H5 at checkpoint {checkpoint!r}", file=sys.stderr)
            sys.exit(2)
        npz_path = evaluation_dir / "predictions" / args.split / f"{safe_name(video_name)}.npz"
        if not npz_path.is_file():
            print(f"FAIL: {alias}: missing prediction npz", file=sys.stderr)
            sys.exit(2)

        data = load_stage5_pointcloud_h5(h5_path)
        point_label = np.asarray(data["point_label"])
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        prob_femur, pred_label = load_prediction(npz_path)
        if prob_femur.shape != point_label.shape:
            print(
                f"FAIL: {alias}: prediction has {prob_femur.size} points, H5 has {point_label.size}",
                file=sys.stderr,
            )
            sys.exit(2)

        gt_data = read_gt_points(h5_path)
        frame_rows = build_frame_table(
            gt_data=gt_data,
            point_label=point_label,
            valid_mask=valid_mask,
            pred_label=pred_label,
            prob_femur=prob_femur,
            ignore_index=args.ignore_index,
            link_distance=args.link_distance,
            min_component_points=args.min_component_points,
        )
        videos.append(
            {
                "anonymous_id": alias,
                "frame_rows": frame_rows,
                "localization": summarize_frame_localization(frame_rows),
                "within_video": within_video_region_comparison(frame_rows),
            }
        )
        for frame_row in frame_rows:
            per_frame_records.append({"anonymous_id": alias, **frame_row})

    aggregate = aggregate_paired(videos)
    payload = {
        "split": args.split,
        "checkpoint": checkpoint,
        "link_distance": args.link_distance,
        "min_component_points": args.min_component_points,
        "num_videos": len(videos),
        "videos": [
            {k: v for k, v in video.items() if k != "frame_rows"} for video in videos
        ],
        "aggregate": aggregate,
        "interpretation_note": (
            "Frames are not independent observations: consecutive frames of one video overlap heavily "
            "in content. The paired within-video comparison controls for the video, not for that "
            "correlation, so the sign test's p-value should be read as a direction check."
        ),
        "requires_policy_chat_judgment": True,
    }
    payload["privacy_self_check"] = privacy_self_check(payload)

    print(f"Frame-level failure vs GT region count  [{checkpoint}, {args.split}]")
    print(f"  videos                         : {len(videos)}")
    print(f"  videos with both frame kinds   : {aggregate['videos_with_both_kinds_of_frame']}")
    print(f"  within-video recall difference : {aggregate['within_video_recall_median_difference_median']}")
    print(f"  sign test                      : {aggregate['sign_test']}")
    print(f"  pooled single-region frames    : {aggregate['pooled_single_region_frames']}")
    print(f"  pooled multi-region frames     : {aggregate['pooled_multi_region_frames']}")
    print("\n  per video (alias / GT frames / zero-recall frames / single vs multi recall median):")
    for video in videos:
        local = video["localization"]
        within = video["within_video"]
        single = within["single_region"]["recall_median"] if within["single_region"] else None
        multi = within["multi_region"]["recall_median"] if within["multi_region"] else None
        print(
            f"    {video['anonymous_id']:>18}  frames={local['num_gt_frames']:>4}  "
            f"zero={local['frames_with_zero_recall']:>4}  single={single}  multi={multi}"
        )
    print(f"\n  {aggregate['pooled_note']}")

    if args.per_frame_csv and per_frame_records:
        path = Path(args.per_frame_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_frame_records[0]))
            writer.writeheader()
            writer.writerows(per_frame_records)
        print(f"  per-frame CSV  : {path}")
    for key, target in (("private_json", args.private_json), ("shareable_json", args.shareable_json)):
        if not target:
            continue
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {key}   : {path}")

    if not all(payload["privacy_self_check"].values()):
        print("FAIL: privacy self-check found a real video id or host path", file=sys.stderr)
        sys.exit(2)
    print("\nDone. The grouping is mechanical; the association is not a demonstrated cause.")


if __name__ == "__main__":
    main()
