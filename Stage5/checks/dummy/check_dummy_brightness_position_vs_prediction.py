from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))

from check_stage5_brightness_position_vs_prediction import (  # noqa: E402
    NUM_BRIGHTNESS_BINS,
    NUM_EDGE_BINS,
    Accumulator,
    analyze_video,
    brightness_percentile_lookup,
    build_summary,
    conditional_span,
    derive_any_frame_label,
    direct_standardization,
    edge_distance_normalized,
    group_exposure_and_response,
    intensity_brightness_agreement,
    marginal_table,
    read_group_labels,
    sample_brightness,
    to_bin,
)

# ----------------------------------------------------------------------------
# S5-15 verification (3): synthetic coverage for the brightness/position tables.
#
# The properties that matter: brightness is sampled at [y, x] and not [x, y]
# (a transposed lookup would still produce a plausible-looking table), the
# percentile is a mid-rank so a flat frame is 0.5 rather than 1.0, and above
# all the CONDITIONAL span really does hold the other axis fixed -- that
# function is the whole argument of the checker, so it is tested against
# tables built to have a known answer, including one where the marginal
# association points the opposite way from the conditional one.
# ----------------------------------------------------------------------------


def test_percentile_is_a_mid_rank() -> None:
    flat = np.zeros(256, dtype=np.int64)
    flat[100] = 50
    assert abs(brightness_percentile_lookup(flat)[100] - 0.5) < 1e-12, "a flat frame must be 0.5, not 1.0"

    two_levels = np.zeros(256, dtype=np.int64)
    two_levels[10] = 30
    two_levels[200] = 10
    lookup = brightness_percentile_lookup(two_levels)
    assert abs(lookup[10] - (0 + 15) / 40) < 1e-12
    assert abs(lookup[200] - (30 + 5) / 40) < 1e-12
    assert np.isnan(brightness_percentile_lookup(np.zeros(256, dtype=np.int64))[0])
    print("  ok: the percentile is a mid-rank, so ties do not push a level to 0.0 or 1.0")


def test_brightness_is_sampled_at_y_x_not_x_y() -> None:
    """A transposed lookup produces a table that still looks reasonable, so the
    orientation is pinned with an image whose two candidate pixels differ."""
    gray = np.zeros((8, 16), dtype=np.uint8)
    gray[2, 7] = 200  # row 2, column 7
    gray[7, 2] = 50
    values = sample_brightness(gray, np.array([[7.0, 2.0]]))  # pixel_xy is (x, y)
    assert int(values[0]) == 200, "pixel_xy is (x, y) and must index gray[y, x]"
    # Sub-pixel coordinates floor into their own pixel rather than rounding up.
    assert int(sample_brightness(gray, np.array([[7.9, 2.9]]))[0]) == 200
    print("  ok: a point at (x, y) reads gray[y, x], and sub-pixel positions floor")


def test_edge_distance_is_zero_at_the_border_and_one_at_the_centre() -> None:
    width, height = 65, 65
    corners = np.array([[0.0, 0.0], [64.0, 0.0], [0.0, 64.0], [64.0, 64.0]])
    assert np.allclose(edge_distance_normalized(corners, width=width, height=height), 0.0)
    centre = edge_distance_normalized(np.array([[32.0, 32.0]]), width=width, height=height)
    assert abs(centre[0] - 1.0) < 1e-12
    # A non-square crop: the short side sets the scale, and a point far along
    # the long axis is still at the border if it is near the short-axis edge.
    wide = edge_distance_normalized(np.array([[100.0, 0.0]]), width=201, height=21)
    assert abs(wide[0]) < 1e-12
    print("  ok: edge distance is 0 on the border and 1 at the centre, for non-square crops too")


def test_binning_puts_one_point_zero_in_the_last_bin() -> None:
    assert to_bin(np.array([0.0]), num_bins=10)[0] == 0
    assert to_bin(np.array([1.0]), num_bins=10)[0] == 9, "1.0 must not fall off the end"
    assert to_bin(np.array([0.999999]), num_bins=10)[0] == 9
    assert to_bin(np.array([0.1]), num_bins=10)[0] == 1
    print("  ok: a percentile of exactly 1.0 lands in the last bin instead of overflowing")


def build_table(mean_prob: np.ndarray, *, count: int = 1000) -> Accumulator:
    acc = Accumulator(mean_prob.shape)
    acc.count = np.full(mean_prob.shape, count, dtype=np.int64)
    acc.prob_sum = mean_prob * count
    acc.predicted_positive = (mean_prob * count).astype(np.int64)
    return acc


def test_conditional_span_holds_the_other_axis_fixed() -> None:
    # Probability depends on axis 0 only.
    axis0_only = np.tile(np.linspace(0.0, 0.9, 10)[:, None], (1, 10))
    acc = build_table(axis0_only)
    varied = conditional_span(acc, vary_axis=0, hold_axis=1, min_points=10)
    held = conditional_span(acc, vary_axis=1, hold_axis=0, min_points=10)
    assert abs(varied["median_span"] - 0.9) < 1e-12
    assert abs(held["median_span"]) < 1e-12, "sweeping the axis that carries no signal must be flat"
    assert varied["num_strata_used"] == 10

    # The transpose must be handled: the same table with the roles swapped.
    axis1_only = axis0_only.T
    acc_t = build_table(axis1_only)
    assert abs(conditional_span(acc_t, vary_axis=1, hold_axis=0, min_points=10)["median_span"] - 0.9) < 1e-12
    assert abs(conditional_span(acc_t, vary_axis=0, hold_axis=1, min_points=10)["median_span"]) < 1e-12
    print("  ok: the conditional span varies one axis with the other fixed, in both orientations")


def test_conditional_span_can_contradict_the_marginal() -> None:
    """The reason the checker reports conditionals at all: a marginal can point
    one way while every stratum points the other. If this ever agreed by
    construction, the conditional tables would be decoration."""
    mean = np.zeros((10, 10))
    counts = np.zeros((10, 10), dtype=np.int64)
    for i in range(10):
        for j in range(10):
            # Within every stratum j, probability FALLS with i ...
            mean[i, j] = 0.8 - 0.05 * i + 0.09 * j
            # ... but the high-i cells are populated only where j is also high.
            counts[i, j] = 1000 if abs(i - j) <= 1 else 0
    acc = Accumulator((10, 10))
    acc.count = counts
    acc.prob_sum = mean * counts
    acc.predicted_positive = (mean * counts).astype(np.int64)

    marginal = marginal_table(acc, 0, axis_names=["a", "b"])
    populated = [row for row in marginal if row["num_points"] > 0]
    assert populated[-1]["mean_prob"] > populated[0]["mean_prob"], "marginally, probability RISES with axis 0"

    span = conditional_span(acc, vary_axis=0, hold_axis=1, min_points=100)
    per_stratum = [s for s in span["per_stratum"] if s["span"] is not None]
    assert per_stratum, "some strata must have two usable bins"
    # Every stratum is a short descending run, so the conditional picture is the
    # opposite of the marginal one.
    assert all(s["span"] > 0 for s in per_stratum)
    assert span["median_span"] < 0.2, "within a stratum the movement is small and downward"
    print("  ok: the conditional result can and does disagree with the marginal one")


def test_span_extremes_and_trend_are_reported_next_to_the_median() -> None:
    """The median span hid a real interaction on the first real run: the edge
    effect was ~0.02 in every dim stratum and 0.30 in the brightest one, which
    a median of 0.023 reports as "position barely matters"."""
    mean = np.zeros((10, 5))
    for hold in range(5):
        # Nothing happens in the low strata; the top stratum moves a lot.
        amplitude = 0.02 if hold < 4 else 0.30
        mean[:, hold] = np.linspace(0.0, amplitude, 10)
    acc = build_table(mean)
    span = conditional_span(acc, vary_axis=0, hold_axis=1, min_points=10)
    assert abs(span["median_span"] - 0.02) < 1e-12, "the median reports the quiet strata"
    assert abs(span["max_span"] - 0.30) < 1e-12
    assert span["max_span_at_held_bin"] == 4
    assert abs(span["span_at_lowest_held_bin"] - 0.02) < 1e-12
    assert abs(span["span_at_highest_held_bin"] - 0.30) < 1e-12
    assert span["span_grows_with_held_bin"] is True
    assert "understates it" in span["reading_note"]

    # A genuinely constant effect must NOT be flagged as an interaction.
    flat = conditional_span(build_table(np.tile(np.linspace(0, 0.2, 10)[:, None], (1, 5))),
                            vary_axis=0, hold_axis=1, min_points=10)
    assert flat["span_grows_with_held_bin"] is False
    print("  ok: the max span and the trend are reported, so a median cannot hide an interaction")


def test_sparse_cells_are_excluded_not_averaged_in() -> None:
    mean = np.tile(np.linspace(0.0, 0.9, 10)[:, None], (1, 10))
    acc = build_table(mean, count=1000)
    acc.count[9, :] = 3  # a near-empty top bin
    acc.prob_sum[9, :] = 3 * 0.99
    strict = conditional_span(acc, vary_axis=0, hold_axis=1, min_points=100)
    loose = conditional_span(acc, vary_axis=0, hold_axis=1, min_points=1)
    assert abs(strict["median_span"] - 0.8) < 1e-12, "the 3-point bin must be dropped"
    assert abs(loose["median_span"] - 0.99) < 1e-12
    assert strict["min_points_per_cell"] == 100
    print("  ok: cells below min_points are dropped rather than contributing a noisy extreme")


def test_accumulator_merges_and_averages() -> None:
    a = Accumulator((2, 2))
    a.add((np.array([0, 0]), np.array([1, 1])), np.array([0.2, 0.4]), np.array([0, 1]))
    b = Accumulator((2, 2))
    b.add((np.array([0]), np.array([1])), np.array([0.9]), np.array([1]))
    a.merge(b)
    assert a.count[0, 1] == 3
    assert abs(a.mean_prob()[0, 1] - (0.2 + 0.4 + 0.9) / 3) < 1e-12
    assert abs(a.positive_rate()[0, 1] - 2 / 3) < 1e-12
    assert np.isnan(a.mean_prob()[1, 1]), "an empty cell is NaN, not 0.0"
    print("  ok: accumulators merge, and an empty cell stays NaN instead of reading as zero probability")


def test_intensity_correlation_reports_none_when_undefined() -> None:
    brightness = np.arange(100, dtype=np.float64)
    assert abs(intensity_brightness_agreement(brightness * 3 + 7, brightness) - 1.0) < 1e-12
    assert abs(intensity_brightness_agreement(-brightness, brightness) + 1.0) < 1e-12
    assert intensity_brightness_agreement(np.ones(100), brightness) is None, "a constant has no correlation"
    assert intensity_brightness_agreement(np.array([1.0]), np.array([1.0])) is None
    print("  ok: the intensity/brightness correlation is None when it is undefined, never 0.0")


def test_analyze_video_places_bright_points_in_the_top_bin() -> None:
    height = width = 32
    images = np.zeros((2, height, width), dtype=np.uint8)
    images[:, 16, 16] = 255  # one bright pixel per frame, at the centre
    num_points = 6
    pixel_xy = np.array(
        [[16.0, 16.0], [1.0, 1.0], [5.0, 5.0], [16.0, 16.0], [1.0, 1.0], [5.0, 5.0]]
    )
    frame_order = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    point_label = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)
    valid = np.ones(num_points, dtype=bool)
    prob = np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.1])
    pred = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)

    result = analyze_video(
        gray_images=images,
        num_frames=2,
        width=width,
        height=height,
        pixel_xy=pixel_xy,
        frame_order=frame_order,
        point_label=point_label,
        valid_mask=valid,
        intensity=np.array([255.0, 0.0, 0.0, 255.0, 0.0, 0.0]),
        prob_femur=prob,
        pred_label=pred,
        ignore_index=-1,
        to_gray=lambda image: image,
    )
    assert result["num_valid_points"] == 6
    scale = result["scale"]
    # The two bright points are GT positive (class 0) and sit in the top
    # within-frame brightness bin; the dark ones are background in the low bins.
    assert scale.count[NUM_BRIGHTNESS_BINS - 1, NUM_BRIGHTNESS_BINS - 1, 0] == 2
    assert scale.count[:, :, 1].sum() == 4
    assert result["brightness_raw_mean_on_gt_positive"] == 255.0
    assert result["brightness_raw_mean_on_gt_background"] == 0.0
    assert result["intensity_vs_image_brightness_correlation"] is not None
    assert result["position"].count.shape == (NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2)
    print("  ok: a point on the brightest pixel lands in the top within-frame brightness bin")


def test_analyze_video_refuses_a_frame_index_the_images_do_not_have() -> None:
    images = np.zeros((2, 16, 16), dtype=np.uint8)
    try:
        analyze_video(
            gray_images=images,
            num_frames=2,
            width=16,
            height=16,
            pixel_xy=np.array([[1.0, 1.0]]),
            frame_order=np.array([5], dtype=np.int64),
            point_label=np.array([1]),
            valid_mask=np.array([True]),
            intensity=np.array([1.0]),
            prob_femur=np.array([0.5]),
            pred_label=np.array([1], dtype=np.uint8),
            ignore_index=-1,
            to_gray=lambda image: image,
        )
    except ValueError as error:
        assert "not the same video" in str(error)
    else:
        raise AssertionError("a frame index beyond the image stack must stop the run")
    print("  ok: points whose frames the image stack does not contain stop the run")


def test_ignore_points_never_enter_the_tables() -> None:
    height = width = 16
    images = np.full((1, height, width), 10, dtype=np.uint8)
    result = analyze_video(
        gray_images=images,
        num_frames=1,
        width=width,
        height=height,
        pixel_xy=np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        frame_order=np.zeros(3, dtype=np.int64),
        point_label=np.array([1, 0, -1]),
        valid_mask=np.array([True, True, True]),
        intensity=np.array([1.0, 2.0, 3.0]),
        prob_femur=np.array([0.9, 0.2, 0.99]),
        pred_label=np.array([1, 0, 1], dtype=np.uint8),
        ignore_index=-1,
        to_gray=lambda image: image,
    )
    assert result["num_valid_points"] == 2
    assert int(result["scale"].count.sum()) == 2, "the ignore point must not be binned"
    print("  ok: ignore points are dropped before any table is filled")


def test_summary_labels_what_each_axis_means() -> None:
    scale = Accumulator((NUM_BRIGHTNESS_BINS, NUM_BRIGHTNESS_BINS, 2))
    position = Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2))
    scale.count[:] = 1000
    scale.prob_sum[:] = 300.0
    position.count[:] = 1000
    position.prob_sum[:] = 300.0
    summary = build_summary(scale, position, min_points=10)
    assert set(summary["bin_definitions"]) == {
        "in_frame_brightness_percentile",
        "video_brightness_percentile",
        "edge_distance",
        "gt_class",
    }
    assert "ignore points are excluded" in summary["bin_definitions"]["gt_class"]
    h3 = summary["hypothesis_3_which_brightness_scale"]
    assert abs(h3["in_frame_varied_video_held"]["median_span"]) < 1e-12
    assert "not that it is unrelated" in h3["reading"]
    print("  ok: the summary states what each bin means and how to read a small span")


def test_standardization_removes_a_pure_exposure_difference() -> None:
    """Two groups that respond identically bin for bin, but sit in different
    bins, must come out with a crude gap and no standardized gap."""
    # Response is the same function of the bin in both groups.
    bin_mean = np.array([0.1, 0.5, 0.9])
    reference_count = np.array([1000.0, 1000.0, 1000.0])
    target_count = np.array([2000.0, 800.0, 200.0])  # skewed towards the dim bin
    result = direct_standardization(
        reference_count=reference_count,
        reference_prob_sum=reference_count * bin_mean,
        target_count=target_count,
        target_prob_sum=target_count * bin_mean,
    )
    assert result["usable"] is True
    assert result["gap_crude"] < -0.1, "the crude comparison sees the exposure difference"
    assert abs(result["gap_after_standardization"]) < 1e-12, "matched bins must erase it"
    assert abs(result["fraction_of_gap_explained_by_exposure"] - 1.0) < 1e-9
    assert result["reference_weight_dropped"] == 0.0
    print("  ok: a pure exposure difference is fully explained away by standardization")


def test_standardization_keeps_a_pure_response_difference() -> None:
    """Same bins, different behaviour in them: standardization must NOT hide it."""
    counts = np.array([1000.0, 1000.0, 1000.0])
    reference_mean = np.array([0.4, 0.5, 0.6])
    target_mean = reference_mean - 0.2
    result = direct_standardization(
        reference_count=counts,
        reference_prob_sum=counts * reference_mean,
        target_count=counts,
        target_prob_sum=counts * target_mean,
    )
    assert abs(result["gap_crude"] + 0.2) < 1e-12
    assert abs(result["gap_after_standardization"] + 0.2) < 1e-12, "a response gap must survive"
    assert abs(result["fraction_of_gap_explained_by_exposure"]) < 1e-9
    print("  ok: a response difference at matched bins survives standardization untouched")


def test_standardization_reports_dropped_reference_weight() -> None:
    """Bins the target never populates cannot be standardized; their weight is
    dropped, and the size of that hole has to be visible."""
    result = direct_standardization(
        reference_count=np.array([500.0, 500.0]),
        reference_prob_sum=np.array([500.0 * 0.2, 500.0 * 0.8]),
        target_count=np.array([1000.0, 0.0]),
        target_prob_sum=np.array([1000.0 * 0.2, 0.0]),
    )
    assert result["usable"] is True
    assert abs(result["reference_weight_dropped"] - 0.5) < 1e-12
    assert result["num_bins_used"] == 1
    assert "rests on few bins" in result["note"]

    empty = direct_standardization(
        reference_count=np.array([100.0, 0.0]),
        reference_prob_sum=np.array([10.0, 0.0]),
        target_count=np.array([0.0, 100.0]),
        target_prob_sum=np.array([0.0, 10.0]),
    )
    assert empty["usable"] is False
    print("  ok: unmatched reference weight is reported, and a total mismatch is refused")


def test_group_comparison_uses_gt_positive_points_by_default() -> None:
    def group(positive_prob: float, background_prob: float) -> dict:
        scale = Accumulator((NUM_BRIGHTNESS_BINS, NUM_BRIGHTNESS_BINS, 2))
        position = Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2))
        video_position = Accumulator((NUM_BRIGHTNESS_BINS, NUM_EDGE_BINS, 2))
        for acc in (scale, position, video_position):
            acc.count[..., 0] = 100
            acc.prob_sum[..., 0] = 100 * positive_prob
            acc.count[..., 1] = 100
            acc.prob_sum[..., 1] = 100 * background_prob
        return {"scale": scale, "position": position, "video_position": video_position}

    groups = {"single_region": group(0.60, 0.10), "multi_region": group(0.30, 0.10)}
    positives = group_exposure_and_response(groups, reference="single_region", target="multi_region")
    joint = positives["standardized_by_brightness_and_edge_jointly"]
    assert positives["gt_class"] == "gt_positive"
    assert abs(joint["reference_mean"] - 0.60) < 1e-12
    assert abs(joint["gap_after_standardization"] + 0.30) < 1e-12, "the GT-positive gap must survive"
    for key in (
        "standardized_by_video_brightness_and_edge_jointly",
        "standardized_by_both_brightness_scales_jointly",
    ):
        assert positives[key]["usable"], f"{key} must be reported too"

    background = group_exposure_and_response(
        groups, reference="single_region", target="multi_region", gt_class_index=1
    )
    assert background["gt_class"] == "gt_background"
    assert abs(background["standardized_by_brightness_and_edge_jointly"]["gap_after_standardization"]) < 1e-12
    assert "do not explain the group" in positives["reading"]

    assert group_exposure_and_response({"single_region": groups["single_region"]},
                                       reference="single_region", target="multi_region")["usable"] is False
    print("  ok: the group comparison reads GT-positive points by default and keeps the classes apart")


def test_group_labels_come_from_the_gt_region_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gt_regions_shareable.json"
        path.write_text(
            json.dumps(
                {
                    "videos": [
                        {"anonymous_id": "validation_001", "classification": {"multi_region_any_frame": False}},
                        {"anonymous_id": "validation_002", "classification": {"multi_region_any_frame": True}},
                    ]
                }
            ),
            encoding="utf-8",
        )
        labels, source = read_group_labels(path, label_key="multi_region_any_frame")
        assert labels == {"validation_001": "single_region", "validation_002": "multi_region"}
        assert source == "classification_field"
        try:
            read_group_labels(path, label_key="not_a_key")
        except ValueError as error:
            assert "available keys are" in str(error)
        else:
            raise AssertionError("an unknown label key must stop the run, not default to one group")
    print("  ok: group labels are read from the mechanical GT output, and an unknown key stops the run")


def test_any_frame_label_is_derived_from_an_older_sweep_only_file() -> None:
    """The GT checker gained multi_region_any_frame after its first real run, so
    outputs from before it carry only the sweep. The label is recomputed from
    those recorded fractions -- identical by definition, not an approximation --
    and the result says it was derived rather than read."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old_gt_regions_shareable.json"
        path.write_text(
            json.dumps(
                {
                    "videos": [
                        {
                            "anonymous_id": "validation_001",
                            "classification": {"multi_region_all": False, "multi_region_any": False},
                            "sweep": [{"link_distance": d, "multi_region_frame_fraction": 0.0} for d in (2.0, 12.0)],
                        },
                        {
                            # 11% of frames: below the 50% threshold the old
                            # fields used, but still a multi-region video.
                            "anonymous_id": "validation_002",
                            "classification": {"multi_region_all": False, "multi_region_any": False},
                            "sweep": [{"link_distance": d, "multi_region_frame_fraction": 0.11} for d in (2.0, 12.0)],
                        },
                        {
                            "anonymous_id": "validation_003",
                            "classification": {"multi_region_all": False, "multi_region_any": False},
                            "sweep": [{"link_distance": 2.0, "multi_region_frame_fraction": None}],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert derive_any_frame_label(payload["videos"][0]) is False
        assert derive_any_frame_label(payload["videos"][1]) is True
        assert derive_any_frame_label(payload["videos"][2]) is None, "no usable sweep means no label"

        try:
            read_group_labels(path, label_key="multi_region_any_frame")
        except ValueError as error:
            assert "Re-run check_stage5_gt_component_count.sh" in str(error)
        else:
            raise AssertionError("a video with no derivable label must stop the run")

        payload["videos"] = payload["videos"][:2]
        path.write_text(json.dumps(payload), encoding="utf-8")
        labels, source = read_group_labels(path, label_key="multi_region_any_frame")
        assert labels == {"validation_001": "single_region", "validation_002": "multi_region"}
        assert source == "derived_from_sweep", "a derived label must not be reported as a recorded one"

        # Any other missing key still needs a re-run; only this one is derivable.
        try:
            read_group_labels(path, label_key="stable_across_link_distances")
        except ValueError as error:
            assert "Only 'multi_region_any_frame' can be derived" in str(error)
        else:
            raise AssertionError("an underivable key must stop the run")
    print("  ok: the any-frame label is recomputed from an older sweep, and says it was derived")


def test_explained_fraction_is_suppressed_when_there_is_no_gap_to_explain() -> None:
    """Dividing by a near-zero crude gap produces a large meaningless ratio: the
    first real run reported 2.98 for GT background points whose crude gap was
    -0.005. Below a relative floor the ratio is None and the gaps are read raw."""
    counts = np.array([1000.0, 1000.0])
    reference_mean = np.array([0.16, 0.17])
    # Practically identical groups: a crude gap far under 5% of the reference.
    target_mean = reference_mean + np.array([0.0, -0.01])
    tiny = direct_standardization(
        reference_count=counts,
        reference_prob_sum=counts * reference_mean,
        target_count=np.array([1500.0, 500.0]),
        target_prob_sum=np.array([1500.0, 500.0]) * target_mean,
    )
    assert abs(tiny["gap_crude"]) < 0.05 * tiny["reference_mean"]
    assert tiny["fraction_of_gap_explained_by_exposure"] is None
    assert tiny["ratio_suppressed_as_gap_too_small"] is True
    assert tiny["gap_crude"] is not None and tiny["gap_after_standardization"] is not None

    # A gap worth explaining still gets a ratio.
    real = direct_standardization(
        reference_count=counts,
        reference_prob_sum=counts * np.array([0.4, 0.6]),
        target_count=counts,
        target_prob_sum=counts * np.array([0.2, 0.4]),
    )
    assert real["ratio_suppressed_as_gap_too_small"] is False
    assert real["fraction_of_gap_explained_by_exposure"] is not None
    print("  ok: the explained-fraction ratio is suppressed when the crude gap is too small to divide by")


def main() -> None:
    tests = [
        test_percentile_is_a_mid_rank,
        test_brightness_is_sampled_at_y_x_not_x_y,
        test_edge_distance_is_zero_at_the_border_and_one_at_the_centre,
        test_binning_puts_one_point_zero_in_the_last_bin,
        test_conditional_span_holds_the_other_axis_fixed,
        test_conditional_span_can_contradict_the_marginal,
        test_span_extremes_and_trend_are_reported_next_to_the_median,
        test_sparse_cells_are_excluded_not_averaged_in,
        test_accumulator_merges_and_averages,
        test_intensity_correlation_reports_none_when_undefined,
        test_analyze_video_places_bright_points_in_the_top_bin,
        test_analyze_video_refuses_a_frame_index_the_images_do_not_have,
        test_ignore_points_never_enter_the_tables,
        test_summary_labels_what_each_axis_means,
        test_standardization_removes_a_pure_exposure_difference,
        test_standardization_keeps_a_pure_response_difference,
        test_standardization_reports_dropped_reference_weight,
        test_group_comparison_uses_gt_positive_points_by_default,
        test_explained_fraction_is_suppressed_when_there_is_no_gap_to_explain,
        test_group_labels_come_from_the_gt_region_output,
        test_any_frame_label_is_derived_from_an_older_sweep_only_file,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 brightness/position synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
