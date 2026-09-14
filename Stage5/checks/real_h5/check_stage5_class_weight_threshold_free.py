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

import numpy as np

from checks.real_h5.check_stage5_class_weight_ablation import (  # noqa: F401 (re-exported for callers)
    check_history,
    check_resolved_weight,
    compare_config,
    load_json,
)
from stage5.utils.h5_io import load_stage5_pointcloud_h5

# ----------------------------------------------------------------------------
# S5-13補足 Step 6.1/6.2: threshold-free (AUPRC/AUROC/PR curve) and
# threshold-sweep diagnostics for a single class-weight arm's checkpoint
# evaluation, reusing evaluate_stage5.py's already-saved per-H5 prediction
# .npz artifacts (prob_femur) and re-reading GT directly from the source H5
# instead of re-running the model. h5py/numpy only -- no torch/CUDA import
# anywhere in this module (evaluate_stage5.py is intentionally NOT imported,
# since it pulls in torch at module scope; safe_name() below duplicates its
# one-line filename rule instead).
# ----------------------------------------------------------------------------


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_name(value: str) -> str:
    # Mirrors evaluate_stage5.safe_name() exactly -- must stay in sync with
    # that function so predictions/<split>/<name>.npz resolves correctly.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def binary_counts(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    predicted_positive: np.ndarray,
    *,
    ignore_index: int,
) -> dict[str, int]:
    valid = valid_mask & (labels != int(ignore_index))
    target_positive = labels == 1
    return {
        "true_positive_count": int(np.sum(valid & target_positive & predicted_positive)),
        "false_positive_count": int(np.sum(valid & ~target_positive & predicted_positive)),
        "true_negative_count": int(np.sum(valid & ~target_positive & ~predicted_positive)),
        "false_negative_count": int(np.sum(valid & target_positive & ~predicted_positive)),
    }


def load_prediction(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    require(npz_path.is_file(), f"Missing prediction artifact: {npz_path}")
    with np.load(npz_path) as data:
        prob_femur = data["prob_femur"].astype(np.float64)
        pred_label = data["pred_label"].astype(np.uint8)
    require(bool(np.all(np.isfinite(prob_femur))), f"Non-finite probability in {npz_path}")
    require(bool(np.all((prob_femur >= 0.0) & (prob_femur <= 1.0))), f"Probability outside [0,1] in {npz_path}")
    return prob_femur, pred_label


def load_h5_metrics_rows(path: Path, *, checkpoint: str, split: str) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing h5_metrics.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["checkpoint"] == checkpoint and row["split"] == split]
    require(bool(rows), f"No rows for checkpoint={checkpoint!r} split={split!r} in {path}")
    return rows


def check_threshold_0p5_parity(
    *,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pred_label: np.ndarray,
    row: dict[str, str],
    ignore_index: int,
    context: str,
) -> None:
    counts = binary_counts(labels, valid_mask, pred_label == 1, ignore_index=ignore_index)
    for key in ("true_positive_count", "false_positive_count", "true_negative_count", "false_negative_count"):
        expected = int(row[key])
        actual = counts[key]
        require(
            actual == expected,
            f"{context}: threshold-0.5 parity mismatch for {key}: recomputed={actual} vs h5_metrics.csv={expected}",
        )


# ----------------------------------------------------------------------------
# Threshold-free curve construction. All functions here take a 1-D boolean/
# int y_true array and a 1-D float score array over the *valid* (non-ignore)
# points of one video or one pooled split.
# ----------------------------------------------------------------------------


def binary_clf_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative TP/FP counts at each distinct descending score threshold.

    Returns (thresholds, tps, fps) ordered from the highest score (most
    conservative threshold) to the lowest (threshold = min score, every
    point included).
    """
    require(y_true.size == scores.size, "y_true and scores must have the same length")
    require(y_true.size > 0, "binary_clf_curve requires at least one point")
    order = np.argsort(-scores, kind="mergesort")
    y_true_sorted = y_true[order].astype(np.int64)
    scores_sorted = scores[order]
    distinct = np.where(np.diff(scores_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct, y_true_sorted.size - 1]
    tps = np.cumsum(y_true_sorted)[threshold_idxs].astype(np.float64)
    counts = (threshold_idxs + 1).astype(np.float64)
    fps = counts - tps
    thresholds = scores_sorted[threshold_idxs]
    return thresholds, tps, fps


def average_precision(tps: np.ndarray, fps: np.ndarray, total_positive: float) -> float:
    if total_positive <= 0:
        return float("nan")
    precision = np.where(tps + fps > 0, tps / np.where(tps + fps > 0, tps + fps, 1.0), 1.0)
    recall = tps / total_positive
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def roc_auc(tps: np.ndarray, fps: np.ndarray, total_positive: float, total_negative: float) -> float:
    if total_positive <= 0 or total_negative <= 0:
        return float("nan")
    tpr = np.r_[0.0, tps / total_positive]
    fpr = np.r_[0.0, fps / total_negative]
    # Manual trapezoidal rule (not np.trapz/np.trapezoid): stays correct across
    # numpy versions regardless of which name (or neither) they expose.
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def max_f1_from_curve(tps: np.ndarray, fps: np.ndarray, total_positive: float) -> dict[str, float]:
    fn = total_positive - tps
    denom = 2 * tps + fps + fn
    f1 = np.where(denom > 0, 2 * tps / np.where(denom > 0, denom, 1.0), 0.0)
    idx = int(np.argmax(f1))
    return {"index": idx, "f1": float(f1[idx]), "tp": float(tps[idx]), "fp": float(fps[idx]), "fn": float(fn[idx])}


def recall_at_max_fpr(
    thresholds: np.ndarray, tps: np.ndarray, fps: np.ndarray, *, total_positive: float, total_negative: float, max_fpr: float
) -> dict[str, float] | None:
    if total_positive <= 0 or total_negative <= 0:
        return None
    fpr = fps / total_negative
    feasible = np.flatnonzero(fpr <= max_fpr)
    if feasible.size == 0:
        return None
    idx = int(feasible[-1])
    tp, fp = tps[idx], fps[idx]
    return {
        "threshold": float(thresholds[idx]),
        "recall": float(tp / total_positive),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        "fpr": float(fpr[idx]),
    }


def score_percentiles(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {key: float("nan") for key in ("mean", "median", "p10", "p25", "p75", "p90", "p95", "p99")}
    percentiles = np.percentile(scores, [10, 25, 50, 75, 90, 95, 99])
    return {
        "mean": float(np.mean(scores)),
        "median": float(percentiles[2]),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }


def threshold_grid_table(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(scores, kind="mergesort")
    scores_asc = scores[order]
    y_true_desc = y_true[order][::-1].astype(np.int64)
    cum_tp_desc = np.cumsum(y_true_desc)
    total_positive = int(cum_tp_desc[-1]) if cum_tp_desc.size else 0
    total_count = y_true.size
    total_negative = total_count - total_positive

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pos = scores_asc.size - int(np.searchsorted(scores_asc, threshold, side="left"))
        tp = int(cum_tp_desc[pos - 1]) if pos > 0 else 0
        fp = pos - tp
        fn = total_positive - tp
        tn = total_negative - fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / total_positive if total_positive > 0 else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        fpr = fp / total_negative if total_negative > 0 else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "true_positive_count": tp,
                "false_positive_count": fp,
                "true_negative_count": tn,
                "false_negative_count": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou_femur": iou,
                "false_positive_rate": fpr,
                "predicted_positive_count": pos,
                "predicted_positive_rate": pos / total_count if total_count > 0 else float("nan"),
            }
        )
    return rows


def summarize_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    reference_fpr: float | None,
    fixed_fpr_candidates: list[float],
    threshold_grid: np.ndarray,
) -> dict[str, Any]:
    thresholds, tps, fps = binary_clf_curve(y_true, scores)
    total_positive = float(tps[-1])
    total_negative = float(fps[-1])
    summary: dict[str, Any] = {
        "point_count": int(y_true.size),
        "positive_count": int(total_positive),
        "background_count": int(total_negative),
        "positive_ratio": total_positive / y_true.size if y_true.size else float("nan"),
        "average_precision": average_precision(tps, fps, total_positive),
        "roc_auc": roc_auc(tps, fps, total_positive, total_negative),
        "score_percentiles_positive": score_percentiles(scores[y_true == 1]),
        "score_percentiles_background": score_percentiles(scores[y_true == 0]),
        "max_f1": max_f1_from_curve(tps, fps, total_positive),
        "fixed_fpr": {
            f"{fpr:.4f}": recall_at_max_fpr(
                thresholds, tps, fps, total_positive=total_positive, total_negative=total_negative, max_fpr=fpr
            )
            for fpr in fixed_fpr_candidates
        },
        "threshold_grid": threshold_grid_table(y_true, scores, threshold_grid),
    }
    if reference_fpr is not None:
        summary["recall_at_reference_fpr"] = recall_at_max_fpr(
            thresholds, tps, fps, total_positive=total_positive, total_negative=total_negative, max_fpr=reference_fpr
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-13補足: threshold-free (AUPRC/AUROC/PR curve) diagnostics reusing "
        "evaluate_stage5.py's saved prediction .npz artifacts. No re-inference/torch/CUDA."
    )
    parser.add_argument("--evaluation_dir", required=True, help="evaluate_stage5.py per-checkpoint output_dir")
    parser.add_argument("--checkpoint", default="last", help="checkpoint value in h5_metrics.csv (e.g. 'last' or 'best')")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--threshold_grid_step", type=float, default=0.05)
    parser.add_argument("--reference_fpr", type=float, default=None)
    parser.add_argument("--fixed_fpr_candidates", default="0.01,0.05,0.10")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    h5_metrics_path = evaluation_dir / "h5_metrics.csv"
    threshold_grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.threshold_grid_step), 6)
    fixed_fpr_candidates = [float(item) for item in args.fixed_fpr_candidates.split(",") if item.strip()]

    result: dict[str, Any] = {"evaluation_dir": str(evaluation_dir), "checkpoint": args.checkpoint, "splits": {}}

    for split in ("train_sanity", "validation"):
        rows = load_h5_metrics_rows(h5_metrics_path, checkpoint=args.checkpoint, split=split)
        pooled_label: list[np.ndarray] = []
        pooled_prob: list[np.ndarray] = []
        per_video: list[dict[str, Any]] = []

        for row in rows:
            h5_path = Path(row["h5_path"])
            data = load_stage5_pointcloud_h5(h5_path)
            labels = np.asarray(data["point_label"])
            valid_mask = np.asarray(data["valid_mask"], dtype=bool)

            npz_path = evaluation_dir / "predictions" / split / f"{safe_name(row['video_name'])}.npz"
            prob_femur, pred_label = load_prediction(npz_path)
            require(
                prob_femur.shape == labels.shape,
                f"{row['video_name']}: prediction shape {prob_femur.shape} != H5 label shape {labels.shape}",
            )

            check_threshold_0p5_parity(
                labels=labels,
                valid_mask=valid_mask,
                pred_label=pred_label,
                row=row,
                ignore_index=args.ignore_index,
                context=f"{split}/{row['video_name']}",
            )

            valid = valid_mask & (labels != int(args.ignore_index))
            y_true = (labels[valid] == 1).astype(np.int64)
            scores = prob_femur[valid]
            pooled_label.append(y_true)
            pooled_prob.append(scores)

            video_summary = summarize_curve(
                y_true,
                scores,
                reference_fpr=args.reference_fpr,
                fixed_fpr_candidates=fixed_fpr_candidates,
                threshold_grid=threshold_grid,
            )
            video_summary["video_name"] = row["video_name"]
            per_video.append(video_summary)

        pooled_y_true = np.concatenate(pooled_label) if pooled_label else np.zeros((0,), dtype=np.int64)
        pooled_scores = np.concatenate(pooled_prob) if pooled_prob else np.zeros((0,), dtype=np.float64)
        aggregate_summary = summarize_curve(
            pooled_y_true,
            pooled_scores,
            reference_fpr=args.reference_fpr,
            fixed_fpr_candidates=fixed_fpr_candidates,
            threshold_grid=threshold_grid,
        )

        result["splits"][split] = {
            "num_videos": len(rows),
            "aggregate": aggregate_summary,
            "per_video": per_video,
        }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print("Stage5 class-weight threshold-free diagnostics passed.")
    for split, split_result in result["splits"].items():
        aggregate = split_result["aggregate"]
        print(
            f"{split}: videos={split_result['num_videos']} "
            f"AUPRC={aggregate['average_precision']:.4f} AUROC={aggregate['roc_auc']:.4f} "
            f"max_F1={aggregate['max_f1']['f1']:.4f}"
        )
    if args.output_json:
        print(f"output: {args.output_json}")


if __name__ == "__main__":
    main()
