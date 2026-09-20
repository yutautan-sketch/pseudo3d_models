from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from checks.real_h5.check_stage5_class_weight_threshold_free import load_prediction, safe_name
from checks.real_h5.check_stage5_frame_failure_vs_gt_regions import (
    privacy_self_check,
    read_raw_h5_metrics,
    resolve_checkpoint,
)
from checks.real_h5.check_stage5_gt_component_count import read_video_id_map
from stage5.utils.prediction_frame_render import validate_pixel_xy_bounds

# ----------------------------------------------------------------------------
# S5-15 verification (3): brightness and position versus what the model
# predicts.
#
# The qualitative review proposed two things the visualization could not
# settle (handoff 11.4/11.7):
#
#   (3) the model chases brightness that is high RELATIVE TO ITS OWN FRAME,
#       not brightness that is high for the video as a whole; and
#   (5) brightness alone cannot explain the false positives, because bright
#       edge artifacts are left alone while dimmer central structure is not.
#
# Both are testable from artifacts that already exist. Every point gets the
# grey value of the pixel it sits on, expressed three ways -- raw, as a
# percentile WITHIN its frame, and as a percentile across the WHOLE video --
# plus its distance from the crop edge. Probabilities are accumulated into a
# joint table over those axes, so each claim can be checked with the other
# axis HELD FIXED. A marginal association between brightness and probability
# would prove neither claim; the conditional tables are the point.
#
# `prob_femur` is used throughout, not the 0.5 decision, so nothing here
# depends on where the threshold falls.
#
# Reads the teacher H5s, the intermediate pseudo3d H5s (for the images) and
# the already-saved predictions/*.npz. No re-inference, no torch, no CUDA.
# ----------------------------------------------------------------------------

NUM_BRIGHTNESS_BINS = 10
NUM_EDGE_BINS = 5
GREY_LEVELS = 256


def brightness_percentile_lookup(counts: np.ndarray) -> np.ndarray:
    """Mid-rank percentile in [0,1] for each of the 256 grey levels.

    Mid-rank (half of the tied level counted below) rather than the cumulative
    upper edge, so that a flat image maps to 0.5 instead of 1.0 and a level's
    percentile does not depend on which side of the tie it is read from.
    """
    total = float(counts.sum())
    if total <= 0:
        return np.full(GREY_LEVELS, np.nan)
    below = np.concatenate([[0.0], np.cumsum(counts[:-1], dtype=np.float64)])
    return (below + 0.5 * counts) / total


def sample_brightness(gray: np.ndarray, pixel_xy: np.ndarray) -> np.ndarray:
    """Grey value of the pixel each point sits on (nearest pixel, no blending)."""
    height, width = int(gray.shape[0]), int(gray.shape[1])
    x = np.clip(np.floor(np.asarray(pixel_xy)[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.floor(np.asarray(pixel_xy)[:, 1]).astype(np.int64), 0, height - 1)
    return gray[y, x]


def edge_distance_normalized(pixel_xy: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Distance to the nearest crop border, scaled so the centre is 1.0.

    Reported instead of a radius from the centre because the observation was
    about the image BORDER (bright artifacts there are not predicted positive),
    and for a non-square crop those two are not the same ordering.
    """
    xy = np.asarray(pixel_xy, dtype=np.float64)
    to_edge = np.minimum(
        np.minimum(xy[:, 0], (width - 1) - xy[:, 0]),
        np.minimum(xy[:, 1], (height - 1) - xy[:, 1]),
    )
    scale = min(width - 1, height - 1) / 2.0
    return np.clip(to_edge / scale, 0.0, 1.0)


def to_bin(values: np.ndarray, *, num_bins: int) -> np.ndarray:
    """Map values in [0,1] to bin indices, with 1.0 landing in the last bin."""
    return np.clip((np.asarray(values) * num_bins).astype(np.int64), 0, num_bins - 1)


class Accumulator:
    """Streaming count / probability / predicted-positive sums over a bin grid.

    Streaming because the alternative -- holding every point of every video to
    rank them at the end -- would trade a bounded memory footprint for nothing:
    the percentiles are computed per video from the images themselves, so
    nothing here needs a global sort.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.count = np.zeros(shape, dtype=np.int64)
        self.prob_sum = np.zeros(shape, dtype=np.float64)
        self.predicted_positive = np.zeros(shape, dtype=np.int64)

    def add(self, indices: tuple[np.ndarray, ...], prob: np.ndarray, predicted: np.ndarray) -> None:
        np.add.at(self.count, indices, 1)
        np.add.at(self.prob_sum, indices, prob.astype(np.float64))
        np.add.at(self.predicted_positive, indices, predicted.astype(np.int64))

    def merge(self, other: "Accumulator") -> None:
        self.count += other.count
        self.prob_sum += other.prob_sum
        self.predicted_positive += other.predicted_positive

    def mean_prob(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.prob_sum / np.maximum(self.count, 1), np.nan)

    def positive_rate(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.predicted_positive / np.maximum(self.count, 1), np.nan)


def _cell(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def marginal_table(acc: Accumulator, axis_keep: int, *, axis_names: list[str]) -> list[dict[str, Any]]:
    """Collapse every axis except `axis_keep`."""
    other = tuple(i for i in range(acc.count.ndim) if i != axis_keep)
    count = acc.count.sum(axis=other)
    prob_sum = acc.prob_sum.sum(axis=other)
    predicted = acc.predicted_positive.sum(axis=other)
    rows = []
    for index in range(count.size):
        n = int(count[index])
        rows.append(
            {
                "axis": axis_names[axis_keep],
                "bin": index,
                "num_points": n,
                "mean_prob": (prob_sum[index] / n) if n else None,
                "predicted_positive_rate": (predicted[index] / n) if n else None,
            }
        )
    return rows


def conditional_span(acc: Accumulator, *, vary_axis: int, hold_axis: int, min_points: int) -> dict[str, Any]:
    """How much mean probability moves along `vary_axis` with `hold_axis` fixed.

    This is the test that separates the two brightness scales. If probability
    tracks within-frame brightness, then fixing the whole-video percentile and
    sweeping the within-frame percentile still moves it; if it tracks absolute
    brightness instead, that sweep flattens out. The reverse pairing is
    reported next to it, and whichever moves less is the weaker explanation.
    """
    mean = acc.mean_prob()
    other = tuple(i for i in range(acc.count.ndim) if i not in (vary_axis, hold_axis))
    if other:
        count = acc.count.sum(axis=other)
        prob_sum = acc.prob_sum.sum(axis=other)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, prob_sum / np.maximum(count, 1), np.nan)
        counts = count
    else:
        counts = acc.count
    if vary_axis > hold_axis:
        mean = mean.T
        counts = counts.T

    spans: list[float] = []
    per_stratum: list[dict[str, Any]] = []
    for hold_index in range(mean.shape[1]):
        column_counts = counts[:, hold_index]
        usable = column_counts >= min_points
        values = mean[:, hold_index][usable]
        if values.size < 2:
            per_stratum.append({"held_bin": hold_index, "num_usable_bins": int(values.size), "span": None})
            continue
        span = float(np.nanmax(values) - np.nanmin(values))
        spans.append(span)
        per_stratum.append({"held_bin": hold_index, "num_usable_bins": int(values.size), "span": span})
    # The median alone can hide an interaction: on the S5-15 best run the edge
    # effect is ~0.02 in every dim brightness stratum and 0.30 in the brightest
    # one, and a median of 0.023 reports that as "position barely matters". So
    # the extremes and the trend are reported next to it, and a run where the
    # span grows across strata is marked rather than summarized away.
    usable = [s for s in per_stratum if s["span"] is not None]
    ordered = [s["span"] for s in usable]
    return {
        "median_span": float(np.median(spans)) if spans else None,
        "max_span": max(ordered) if ordered else None,
        "max_span_at_held_bin": (
            usable[int(np.argmax(ordered))]["held_bin"] if ordered else None
        ),
        "span_at_lowest_held_bin": ordered[0] if ordered else None,
        "span_at_highest_held_bin": ordered[-1] if ordered else None,
        "span_grows_with_held_bin": (
            bool(ordered[-1] > 2.0 * ordered[0]) if len(ordered) >= 2 and ordered[0] > 0 else None
        ),
        "num_strata_used": len(spans),
        "per_stratum": per_stratum,
        "min_points_per_cell": min_points,
        "reading_note": (
            "Read max_span and the trend together with the median. When span_grows_with_held_bin is "
            "true the effect is an interaction, not a constant offset, and the median understates it "
            "because the strata where nothing happens outnumber the ones where it does."
        ),
    }


def intensity_brightness_agreement(intensity: np.ndarray, raw_brightness: np.ndarray) -> float | None:
    """Pearson correlation between the H5 `intensity` feature and the image grey.

    Reported because it decides how to read everything else: `intensity` is one
    of the two features the model actually receives, so if it already carries
    the grey value then brightness is an input, and if it does not then any
    brightness effect has to arrive through geometry instead.
    """
    a = np.asarray(intensity, dtype=np.float64)
    b = np.asarray(raw_brightness, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze_video(
    *,
    gray_images: Any,
    num_frames: int,
    width: int,
    height: int,
    pixel_xy: np.ndarray,
    frame_order: np.ndarray,
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    intensity: np.ndarray,
    prob_femur: np.ndarray,
    pred_label: np.ndarray,
    ignore_index: int,
    to_gray: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Per-point brightness percentiles and edge distance, binned into tables."""
    validate_pixel_xy_bounds(pixel_xy, width=width, height=height, context="brightness sampling")

    raw = np.zeros(pixel_xy.shape[0], dtype=np.int64)
    in_frame_percentile = np.full(pixel_xy.shape[0], np.nan)
    video_histogram = np.zeros(GREY_LEVELS, dtype=np.int64)

    frames_present = np.unique(frame_order)
    if int(frames_present.max(initial=-1)) >= num_frames:
        raise ValueError(
            f"frame_order reaches {int(frames_present.max())} but the intermediate H5 has "
            f"{num_frames} images; the points and the images are not the same video"
        )

    gray_by_frame: dict[int, np.ndarray] = {}
    for frame in frames_present.tolist():
        gray = np.asarray(to_gray(gray_images[int(frame)]))
        if gray.ndim != 2:
            raise ValueError(f"frame {frame}: expected a 2-D grey image, got shape {gray.shape}")
        gray_by_frame[int(frame)] = gray
        video_histogram += np.bincount(gray.ravel(), minlength=GREY_LEVELS)[:GREY_LEVELS]

    for frame, gray in gray_by_frame.items():
        mask = frame_order == frame
        values = sample_brightness(gray, pixel_xy[mask])
        raw[mask] = values
        lookup = brightness_percentile_lookup(np.bincount(gray.ravel(), minlength=GREY_LEVELS)[:GREY_LEVELS])
        in_frame_percentile[mask] = lookup[values]

    video_lookup = brightness_percentile_lookup(video_histogram)
    video_percentile = video_lookup[raw]
    edge = edge_distance_normalized(pixel_xy, width=width, height=height)

    valid = np.asarray(valid_mask, dtype=bool) & (point_label != int(ignore_index))
    gt_class = np.where(point_label == 1, 0, 1)  # 0 = GT positive, 1 = GT background
    prob = np.asarray(prob_femur, dtype=np.float64)
    predicted = (np.asarray(pred_label) == 1).astype(np.int64)

    brightness_bin = to_bin(in_frame_percentile, num_bins=NUM_BRIGHTNESS_BINS)
    video_bin = to_bin(video_percentile, num_bins=NUM_BRIGHTNESS_BINS)
    edge_bin = to_bin(edge, num_bins=NUM_EDGE_BINS)

    scale = Accumulator((NUM_BRIGHTNESS_BINS, NUM_BRIGHTNESS_BINS, 2))
    position = Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2))
    # Whole-video brightness against position as well: on the S5-15 best run the
    # whole-video percentile explained more of the group gap than the
    # within-frame one, so a joint adjustment built only from the within-frame
    # axis would understate how much the measured axes can account for.
    video_position = Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2))
    scale.add(
        (brightness_bin[valid], video_bin[valid], gt_class[valid]), prob[valid], predicted[valid]
    )
    position.add(
        (brightness_bin[valid], edge_bin[valid], gt_class[valid]), prob[valid], predicted[valid]
    )
    video_position.add(
        (video_bin[valid], edge_bin[valid], gt_class[valid]), prob[valid], predicted[valid]
    )

    return {
        "num_valid_points": int(valid.sum()),
        "intensity_vs_image_brightness_correlation": intensity_brightness_agreement(
            np.asarray(intensity)[valid], raw[valid]
        ),
        "num_frames_with_points": int(frames_present.size),
        "scale": scale,
        "position": position,
        "video_position": video_position,
        "brightness_raw_mean_on_gt_positive": float(raw[valid & (point_label == 1)].mean())
        if int((valid & (point_label == 1)).sum())
        else None,
        "brightness_raw_mean_on_gt_background": float(raw[valid & (point_label == 0)].mean())
        if int((valid & (point_label == 0)).sum())
        else None,
    }


def build_summary(scale: Accumulator, position: Accumulator, *, min_points: int) -> dict[str, Any]:
    axis_names = ["in_frame_brightness_percentile", "video_brightness_percentile", "gt_class"]
    position_names = ["in_frame_brightness_percentile", "edge_distance", "gt_class"]

    background = 1
    scale_bg = Accumulator(scale.count.shape[:2])
    scale_bg.count = scale.count[:, :, background]
    scale_bg.prob_sum = scale.prob_sum[:, :, background]
    scale_bg.predicted_positive = scale.predicted_positive[:, :, background]

    position_bg = Accumulator(position.count.shape[:2])
    position_bg.count = position.count[:, :, background]
    position_bg.prob_sum = position.prob_sum[:, :, background]
    position_bg.predicted_positive = position.predicted_positive[:, :, background]

    return {
        "marginals": {
            "in_frame_brightness": marginal_table(scale, 0, axis_names=axis_names),
            "video_brightness": marginal_table(scale, 1, axis_names=axis_names),
            "edge_distance": marginal_table(position, 1, axis_names=position_names),
        },
        "hypothesis_3_which_brightness_scale": {
            "question": (
                "With the whole-video brightness percentile held fixed, does the within-frame "
                "percentile still move the probability, and vice versa? Background points only, "
                "because these are the points a false positive can come from."
            ),
            "in_frame_varied_video_held": conditional_span(
                scale_bg, vary_axis=0, hold_axis=1, min_points=min_points
            ),
            "video_varied_in_frame_held": conditional_span(
                scale_bg, vary_axis=1, hold_axis=0, min_points=min_points
            ),
            "reading": (
                "The larger median span is the scale the probability follows more closely. The two "
                "percentiles are themselves correlated, so a small span on one side is evidence that "
                "it adds little ON TOP of the other, not that it is unrelated to the probability."
            ),
        },
        "hypothesis_5_position_beyond_brightness": {
            "question": (
                "Inside a single within-frame brightness bin, does the predicted-positive rate still "
                "depend on distance from the crop border? If it does, brightness alone cannot explain "
                "where the false positives land."
            ),
            "edge_varied_brightness_held": conditional_span(
                position_bg, vary_axis=1, hold_axis=0, min_points=min_points
            ),
            "predicted_positive_rate_by_brightness_and_edge": [
                {
                    "in_frame_brightness_bin": b,
                    "edge_distance_bin": e,
                    "num_points": int(position_bg.count[b, e]),
                    "mean_prob": _cell(position_bg.mean_prob()[b, e]),
                    "predicted_positive_rate": _cell(position_bg.positive_rate()[b, e]),
                }
                for b in range(NUM_BRIGHTNESS_BINS)
                for e in range(NUM_EDGE_BINS)
                if int(position_bg.count[b, e]) >= min_points
            ],
        },
        "bin_definitions": {
            "in_frame_brightness_percentile": f"{NUM_BRIGHTNESS_BINS} equal-width bins of the mid-rank "
            "percentile of the point grey value among ALL pixels of its own frame",
            "video_brightness_percentile": "same, ranked among all pixels of every frame of the video",
            "edge_distance": f"{NUM_EDGE_BINS} equal-width bins of the distance to the nearest crop "
            "border, scaled so the crop centre is 1.0",
            "gt_class": "0 = GT positive, 1 = GT background; ignore points are excluded entirely",
        },
    }


# ----------------------------------------------------------------------------
# Group comparison: does brightness or position explain the video-level gap?
# ----------------------------------------------------------------------------


def direct_standardization(
    *,
    reference_count: np.ndarray,
    reference_prob_sum: np.ndarray,
    target_count: np.ndarray,
    target_prob_sum: np.ndarray,
    min_relative_gap_for_ratio: float = 0.05,
) -> dict[str, Any]:
    """Re-weight the target group to the reference group's bin distribution.

    The video-level gap could be an exposure difference -- the affected videos
    simply having their GT in darker or more peripheral places -- or a response
    difference, the model treating the same kind of point differently. Direct
    standardization separates them: the target group's per-bin means are
    recombined using the REFERENCE group's weights, so the answer is what the
    target group would look like if its points were distributed like the
    reference group's. A gap that survives is not an exposure difference.

    Bins the target group does not populate cannot be standardized, so their
    weight is dropped and the dropped mass is reported. A large dropped mass
    makes the adjusted figure unreliable and must be read before it.
    """
    reference_count = np.asarray(reference_count, dtype=np.float64).ravel()
    reference_prob_sum = np.asarray(reference_prob_sum, dtype=np.float64).ravel()
    target_count = np.asarray(target_count, dtype=np.float64).ravel()
    target_prob_sum = np.asarray(target_prob_sum, dtype=np.float64).ravel()

    reference_total = reference_count.sum()
    target_total = target_count.sum()
    if reference_total <= 0 or target_total <= 0:
        return {"usable": False, "reason": "one of the groups has no points"}

    reference_mean = float(reference_prob_sum.sum() / reference_total)
    crude_target_mean = float(target_prob_sum.sum() / target_total)

    weights = reference_count / reference_total
    usable = target_count > 0
    dropped = float(weights[~usable].sum())
    if not usable.any() or weights[usable].sum() <= 0:
        return {"usable": False, "reason": "the target group populates none of the reference bins"}

    renormalized = weights[usable] / weights[usable].sum()
    target_bin_means = target_prob_sum[usable] / target_count[usable]
    adjusted_target_mean = float((renormalized * target_bin_means).sum())

    return {
        "usable": True,
        "reference_mean": reference_mean,
        "target_mean_crude": crude_target_mean,
        "target_mean_standardized_to_reference": adjusted_target_mean,
        "gap_crude": crude_target_mean - reference_mean,
        "gap_after_standardization": adjusted_target_mean - reference_mean,
        # The ratio is only meaningful when there is a gap to explain: dividing
        # a small adjusted gap by a near-zero crude one produces a large number
        # that says nothing (the first real run reported 2.98 on GT background
        # points whose crude gap was -0.005). Below the threshold it is None,
        # and the two gaps are read directly instead.
        "fraction_of_gap_explained_by_exposure": (
            1.0 - (adjusted_target_mean - reference_mean) / (crude_target_mean - reference_mean)
            if abs(crude_target_mean - reference_mean) >= min_relative_gap_for_ratio * abs(reference_mean)
            and abs(reference_mean) > 0
            else None
        ),
        "ratio_suppressed_as_gap_too_small": bool(
            abs(crude_target_mean - reference_mean) < min_relative_gap_for_ratio * abs(reference_mean)
        ),
        "reference_weight_dropped": dropped,
        "num_bins_used": int(usable.sum()),
        "note": (
            "A gap that barely moves under standardization is NOT explained by the binned axis. "
            "Read reference_weight_dropped first: if much of the reference distribution has no "
            "counterpart in the target group, the adjusted number rests on few bins."
        ),
    }


def group_exposure_and_response(
    groups: dict[str, dict[str, Accumulator]],
    *,
    reference: str,
    target: str,
    gt_class_index: int = 0,
) -> dict[str, Any]:
    """Compare two video groups at matched brightness and matched position.

    Defaults to GT-positive points, because the group difference under
    investigation is a recall difference: the question is whether the model
    assigns lower probability to the GT of one group once the brightness and
    position of that GT are accounted for.
    """
    if reference not in groups or target not in groups:
        return {"usable": False, "reason": "both groups must be present"}

    result: dict[str, Any] = {
        "reference_group": reference,
        "target_group": target,
        "gt_class": "gt_positive" if gt_class_index == 0 else "gt_background",
    }

    def slice_axis(name: str, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ref = groups[reference][name]
        tgt = groups[target][name]
        other = tuple(i for i in range(ref.count.ndim - 1) if i != axis)
        rc = ref.count[..., gt_class_index].sum(axis=other)
        rp = ref.prob_sum[..., gt_class_index].sum(axis=other)
        tc = tgt.count[..., gt_class_index].sum(axis=other)
        tp = tgt.prob_sum[..., gt_class_index].sum(axis=other)
        return rc, rp, tc, tp

    for label, (name, axis) in {
        "in_frame_brightness": ("scale", 0),
        "video_brightness": ("scale", 1),
        "edge_distance": ("position", 1),
    }.items():
        rc, rp, tc, tp = slice_axis(name, axis)
        result[f"exposure_{label}"] = {
            "reference_fraction": (rc / rc.sum()).tolist() if rc.sum() else None,
            "target_fraction": (tc / tc.sum()).tolist() if tc.sum() else None,
        }
        result[f"standardized_by_{label}"] = direct_standardization(
            reference_count=rc, reference_prob_sum=rp, target_count=tc, target_prob_sum=tp
        )

    # The joint grid is the strongest version: brightness AND position matched
    # together, so neither can absorb the other's effect.
    for key, name in (
        ("standardized_by_brightness_and_edge_jointly", "position"),
        ("standardized_by_video_brightness_and_edge_jointly", "video_position"),
        ("standardized_by_both_brightness_scales_jointly", "scale"),
    ):
        ref_joint = groups[reference][name]
        tgt_joint = groups[target][name]
        result[key] = direct_standardization(
            reference_count=ref_joint.count[..., gt_class_index],
            reference_prob_sum=ref_joint.prob_sum[..., gt_class_index],
            target_count=tgt_joint.count[..., gt_class_index],
            target_prob_sum=tgt_joint.prob_sum[..., gt_class_index],
        )
    result["reading"] = (
        "If the gap survives joint standardization, the affected videos are not merely darker or "
        "more peripheral: at the same brightness and the same distance from the border their GT "
        "still receives lower probability, and the axes measured here do not explain the group."
    )
    return result


def derive_any_frame_label(video: dict[str, Any]) -> bool | None:
    """`multi_region_any_frame` recomputed from the recorded sweep.

    The field was added to the GT region-count checker after its first real
    run, so an output produced before that does not carry it. The definition is
    exactly "some frame, at some link distance, held two or more regions",
    which the recorded per-radius frame fractions already determine -- so an
    older file is usable without re-running, and the result is identical rather
    than an approximation. Returns None when the sweep is absent.
    """
    sweep = video.get("sweep")
    if not sweep:
        return None
    fractions = [entry.get("multi_region_frame_fraction") for entry in sweep]
    if all(fraction is None for fraction in fractions):
        return None
    return any(fraction is not None and fraction > 0.0 for fraction in fractions)


def read_group_labels(path: Path, *, label_key: str) -> tuple[dict[str, str], str]:
    """alias -> "multi_region" / "single_region", from the GT region-count output.

    Also returns where the label came from, so a derived label is never
    presented as one the GT checker actually wrote.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload.get("videos")
    if not videos:
        raise ValueError(f"{path}: no 'videos' entries; is this the GT region-count shareable JSON?")
    labels: dict[str, str] = {}
    sources: set[str] = set()
    for video in videos:
        alias = video.get("anonymous_id")
        classification = video.get("classification", {})
        if label_key in classification:
            value = bool(classification[label_key])
            sources.add("classification_field")
        elif label_key == "multi_region_any_frame":
            derived = derive_any_frame_label(video)
            if derived is None:
                raise ValueError(
                    f"{path}: video {alias!r} has neither {label_key!r} nor a usable 'sweep' to "
                    "derive it from. Re-run check_stage5_gt_component_count.sh to regenerate it."
                )
            value = derived
            sources.add("derived_from_sweep")
        else:
            raise ValueError(
                f"{path}: video {alias!r} has no {label_key!r}; available keys are "
                f"{sorted(classification)}. Only 'multi_region_any_frame' can be derived from the "
                "recorded sweep; any other key needs check_stage5_gt_component_count.sh re-run."
            )
        labels[alias] = "multi_region" if value else "single_region"
    return labels, "+".join(sorted(sources))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the model follows within-frame brightness rather than absolute "
        "brightness, and whether position matters once brightness is held fixed. Read-only, CPU only."
    )
    parser.add_argument("--video_id_map", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Defaults to the name of --evaluation_dir")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument(
        "--pseudo3d_outputs_root",
        default=None,
        help="Fallback root holding the intermediate pseudo3d H5s. Only needed when a teacher H5 "
        "does not record a resolvable source_pseudo3d_h5 attr; the run says so if it is.",
    )
    parser.add_argument("--fallback_suffix", default="_pseudo3d.h5")
    parser.add_argument(
        "--stage2to4_root",
        required=True,
        help="Stage2to4 repository root; its image_to_uint8_gray is used so that the grey values "
        "match the ones the GT visualization and the frame export were read from",
    )
    parser.add_argument("--min_points_per_cell", type=int, default=200)
    parser.add_argument(
        "--gt_regions_json",
        default=None,
        help="gt_regions_shareable.json from check_stage5_gt_component_count.py. When given, the "
        "videos are split by its mechanical label and compared at matched brightness and position.",
    )
    parser.add_argument("--group_label_key", default="multi_region_any_frame")
    parser.add_argument("--private_json", default=None)
    parser.add_argument("--shareable_json", default=None)
    parser.add_argument("--per_video_csv", default=None, help="Per-video detail, aliases only (shareable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    try:
        checkpoint = resolve_checkpoint(evaluation_dir, args.checkpoint)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(2)
    stage2to4_root = Path(args.stage2to4_root)
    if not stage2to4_root.is_dir():
        print(f"FAIL: --stage2to4_root is not a directory: {stage2to4_root}", file=sys.stderr)
        sys.exit(2)
    sys.path.insert(0, str(stage2to4_root))

    import h5py

    from checks.real_h5.check_stage5_xy_coordinate_provenance import (
        read_intermediate_local_dimensions,
        resolve_intermediate_h5,
    )
    from stage5.utils.h5_io import load_stage5_pointcloud_h5
    from stage5.utils.prediction_frame_render import load_image_to_uint8_gray

    to_gray = load_image_to_uint8_gray()

    rows = [r for r in read_video_id_map(Path(args.video_id_map)) if r["split"] == args.split]
    if not rows:
        print(f"FAIL: no videos for split {args.split!r}", file=sys.stderr)
        sys.exit(2)
    video_names = read_raw_h5_metrics(evaluation_dir / "h5_metrics.csv", checkpoint=checkpoint)

    scale_total = Accumulator((NUM_BRIGHTNESS_BINS, NUM_BRIGHTNESS_BINS, 2))
    position_total = Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2))
    per_video: list[dict[str, Any]] = []

    group_labels: dict[str, str] = {}
    label_source = ""
    groups: dict[str, dict[str, Accumulator]] = {}
    if args.gt_regions_json:
        try:
            group_labels, label_source = read_group_labels(
                Path(args.gt_regions_json), label_key=args.group_label_key
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"FAIL: could not read --gt_regions_json: {error}", file=sys.stderr)
            sys.exit(2)
        missing = [r["anonymous_id"] for r in rows if r["anonymous_id"] not in group_labels]
        if missing:
            print(
                f"FAIL: --gt_regions_json has no label for {missing}. The two runs must cover the "
                "same videos, or the groups would be built from a different set.",
                file=sys.stderr,
            )
            sys.exit(2)
        for name in set(group_labels.values()):
            groups[name] = {
                "scale": Accumulator((NUM_BRIGHTNESS_BINS, NUM_BRIGHTNESS_BINS, 2)),
                "position": Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2)),
                "video_position": Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2)),
            }

    for row in sorted(rows, key=lambda r: r["anonymous_id"]):
        alias = row["anonymous_id"]
        h5_path = Path(row["original_h5_path"])
        video_name = video_names.get(str(h5_path))
        if video_name is None:
            print(f"FAIL: {alias}: no h5_metrics row at checkpoint {checkpoint!r}", file=sys.stderr)
            sys.exit(2)
        npz_path = evaluation_dir / "predictions" / args.split / f"{safe_name(video_name)}.npz"
        data = load_stage5_pointcloud_h5(h5_path)
        prob_femur, pred_label = load_prediction(npz_path)

        recorded = data["file_attrs"].get("source_pseudo3d_h5")
        if isinstance(recorded, bytes):
            recorded = recorded.decode("utf-8")
        if args.pseudo3d_outputs_root:
            resolution = resolve_intermediate_h5(
                video_name=video_name,
                recorded_source_path=str(recorded) if recorded else None,
                fallback_root=Path(args.pseudo3d_outputs_root),
                fallback_suffix=args.fallback_suffix,
            )
        elif recorded and Path(str(recorded)).is_file():
            # The teacher H5 names its own source, so no search root is needed.
            resolution = {"method": "recorded_source_attr", "path": str(recorded)}
        else:
            resolution = {"method": "unresolved_and_no_fallback_root", "path": None}
        if not resolution.get("path"):
            print(
                f"FAIL: {alias}: could not resolve the intermediate pseudo3d H5 "
                f"({resolution['method']}). The teacher H5 records "
                f"source_pseudo3d_h5={recorded!r}. If that path is stale, pass "
                "--pseudo3d_outputs_root (PSEUDO3D_OUTPUTS_ROOT) so the file can be found by name.",
                file=sys.stderr,
            )
            sys.exit(2)
        dimensions = read_intermediate_local_dimensions(Path(resolution["path"]))
        if dimensions.get("status") != "ok":
            print(f"FAIL: {alias}: local crop dimensions unavailable ({dimensions.get('status')})", file=sys.stderr)
            sys.exit(2)

        with h5py.File(resolution["path"], "r") as handle:
            if "local_encoder_images" not in handle:
                print(f"FAIL: {alias}: intermediate H5 has no local_encoder_images", file=sys.stderr)
                sys.exit(2)
            images = handle["local_encoder_images"]
            image_height, image_width = int(images.shape[-2]), int(images.shape[-1])
            if (image_height, image_width) != (int(dimensions["height"]), int(dimensions["width"])):
                print(
                    f"FAIL: {alias}: images are {image_height}x{image_width} but local_input_shape says "
                    f"{dimensions['height']}x{dimensions['width']}; the pixel space is ambiguous",
                    file=sys.stderr,
                )
                sys.exit(2)
            result = analyze_video(
                gray_images=images,
                num_frames=int(images.shape[0]),
                width=image_width,
                height=image_height,
                pixel_xy=np.asarray(data["pixel_xy"]),
                frame_order=np.asarray(data["frame_order"]).astype(np.int64),
                point_label=np.asarray(data["point_label"]),
                valid_mask=np.asarray(data["valid_mask"], dtype=bool),
                intensity=np.asarray(data["intensity"]),
                prob_femur=prob_femur,
                pred_label=pred_label,
                ignore_index=args.ignore_index,
                to_gray=to_gray,
            )

        scale_total.merge(result["scale"])
        position_total.merge(result["position"])
        if group_labels:
            group = groups[group_labels[alias]]
            group["scale"].merge(result["scale"])
            group["position"].merge(result["position"])
            group["video_position"].merge(result["video_position"])
        per_video.append(
            {
                "anonymous_id": alias,
                "num_valid_points": result["num_valid_points"],
                "num_frames_with_points": result["num_frames_with_points"],
                "resolution_method": resolution["method"],
                "intensity_vs_image_brightness_correlation": result[
                    "intensity_vs_image_brightness_correlation"
                ],
                "brightness_raw_mean_on_gt_positive": result["brightness_raw_mean_on_gt_positive"],
                "brightness_raw_mean_on_gt_background": result["brightness_raw_mean_on_gt_background"],
                "group": group_labels.get(alias, ""),
            }
        )

    summary = build_summary(scale_total, position_total, min_points=args.min_points_per_cell)
    if group_labels and {"multi_region", "single_region"} <= set(groups):
        summary["group_comparison"] = {
            "label_key": args.group_label_key,
            "label_source": label_source,
            "num_videos_per_group": {
                name: sum(1 for a in group_labels.values() if a == name) for name in sorted(groups)
            },
            "on_gt_positive_points": group_exposure_and_response(
                groups, reference="single_region", target="multi_region", gt_class_index=0
            ),
            "on_gt_background_points": group_exposure_and_response(
                groups, reference="single_region", target="multi_region", gt_class_index=1
            ),
            "caveat": (
                "18 videos split 10/8. Standardization matches the measured axes only; any other "
                "difference between the groups is untouched, so a surviving gap says these axes do "
                "not explain it, not that nothing does."
            ),
        }
    payload = {
        "split": args.split,
        "checkpoint": checkpoint,
        "num_videos": len(per_video),
        "per_video": per_video,
        **summary,
        "caveats": [
            "Brightness is read from the source image at each point's pixel. It is NOT necessarily "
            "what the model receives: the model is given `intensity` and `confidence`, plus "
            "mean-centred and max-scaled coordinates.",
            "The two brightness percentiles are strongly correlated with each other by construction, "
            "so the conditional spans compare their incremental contributions, not their total ones.",
            "Points within a frame are not independent observations, so bin counts are not sample "
            "sizes in the statistical sense and no p-value is reported here.",
            "Every number is an association measured on one split of one run; none of it identifies "
            "a cause or licenses a change to the production configuration.",
        ],
        "requires_policy_chat_judgment": True,
    }
    payload["privacy_self_check"] = privacy_self_check(payload)

    print(f"Brightness / position vs prediction  [{checkpoint}, {args.split}]")
    print(f"  videos : {len(per_video)}")
    correlations = [
        v["intensity_vs_image_brightness_correlation"]
        for v in per_video
        if v["intensity_vs_image_brightness_correlation"] is not None
    ]
    if correlations:
        print(
            f"  H5 `intensity` vs image grey, correlation per video: "
            f"min={min(correlations):.4f} median={float(np.median(correlations)):.4f} max={max(correlations):.4f}"
        )
        print("    (decides whether brightness is an input the model sees, or only a proxy)")
    print("\n  (3) which brightness scale does the probability follow, on GT background points")
    h3 = summary["hypothesis_3_which_brightness_scale"]
    print(f"      in-frame varied, video held : median span {h3['in_frame_varied_video_held']['median_span']}")
    print(f"      video varied, in-frame held : median span {h3['video_varied_in_frame_held']['median_span']}")
    print("\n  (5) does position matter with brightness held fixed")
    h5 = summary["hypothesis_5_position_beyond_brightness"]
    edge = h5["edge_varied_brightness_held"]
    print(f"      edge varied, brightness held: median span {edge['median_span']}")
    print(
        f"        max span {edge['max_span']} at brightness bin {edge['max_span_at_held_bin']}; "
        f"lowest bin {edge['span_at_lowest_held_bin']} -> highest bin {edge['span_at_highest_held_bin']}"
    )
    if edge["span_grows_with_held_bin"]:
        print("        the edge effect GROWS with brightness: this is an interaction, not an offset")
    print("\n  marginal mean probability by within-frame brightness bin (all classes):")
    for entry in summary["marginals"]["in_frame_brightness"]:
        print(
            f"      bin {entry['bin']}  n={entry['num_points']:>10}  "
            f"mean_prob={entry['mean_prob']}  pred_pos_rate={entry['predicted_positive_rate']}"
        )

    if "group_comparison" in summary:
        gc = summary["group_comparison"]
        print(f"\n  group comparison ({gc['label_key']}, source: {gc['label_source']}): {gc['num_videos_per_group']}")
        for class_name in ("on_gt_positive_points", "on_gt_background_points"):
            section = gc[class_name]
            print(f"    {class_name}:")
            for key in (
                "standardized_by_in_frame_brightness",
                "standardized_by_video_brightness",
                "standardized_by_edge_distance",
                "standardized_by_brightness_and_edge_jointly",
                "standardized_by_video_brightness_and_edge_jointly",
                "standardized_by_both_brightness_scales_jointly",
            ):
                s = section[key]
                if not s.get("usable"):
                    print(f"      {key:<48} unusable: {s.get('reason')}")
                    continue
                explained = s["fraction_of_gap_explained_by_exposure"]
                explained_text = "n/a (crude gap too small)" if explained is None else f"{explained:.3f}"
                print(
                    f"      {key:<48} gap {s['gap_crude']:+.5f} -> {s['gap_after_standardization']:+.5f}  "
                    f"explained {explained_text}  dropped {s['reference_weight_dropped']:.4f}"
                )
        positive = gc["on_gt_positive_points"]["standardized_by_brightness_and_edge_jointly"]
        background = gc["on_gt_background_points"]["standardized_by_brightness_and_edge_jointly"]
        if positive.get("usable") and background.get("usable"):
            print("    threshold-free discrimination (GT positive mean prob - GT background mean prob):")
            print(f"      single-region        {positive['reference_mean'] - background['reference_mean']:+.5f}")
            print(f"      multi-region crude   {positive['target_mean_crude'] - background['target_mean_crude']:+.5f}")
            print(
                f"      multi-region matched {positive['target_mean_standardized_to_reference'] - background['target_mean_standardized_to_reference']:+.5f}"
            )
        print(f"      {gc['caveat']}")

    if args.per_video_csv and per_video:
        path = Path(args.per_video_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_video[0]))
            writer.writeheader()
            writer.writerows(per_video)
        print(f"\n  per-video CSV : {path}")
    for key, target in (("private_json", args.private_json), ("shareable_json", args.shareable_json)):
        if not target:
            continue
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {key} : {path}")

    if not all(payload["privacy_self_check"].values()):
        print("FAIL: privacy self-check found a real video id or host path", file=sys.stderr)
        sys.exit(2)
    print("\nDone. These are associations measured on saved artifacts, not demonstrated causes.")


if __name__ == "__main__":
    main()
