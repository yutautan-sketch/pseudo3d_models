from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_coordinate_transform_diagnostics import (
    PARITY_GATED_CONDITIONS,
    ROTATION_DEGREES,
    TRANSFORM_CONDITIONS,
    TRANSLATION_MAGNITUDE,
    aggregate_condition_summary,
    aggregate_hot_cold_condition_summary,
    apply_transform,
    build_video_hot_cold_masks,
    condition_hot_cold_rows,
    per_point_condition_metrics,
    rotation_matrix_z,
)
from checks.real_h5.check_stage5_s5_14_supplement import build_train_prior
from stage5.utils.feature_normalization import normalize_xyz

# ----------------------------------------------------------------------------
# S5-14補足2 Step T1: synthetic pass/fail coverage for
# check_stage5_coordinate_transform_diagnostics.py's torch-free logic --
# transform math (translation invertibility, rotation distance preservation
# and orthogonality, unit-ball exceedance bookkeeping), the
# pre-normalization-translation-is-cancelled-by-centering property that
# justifies transforming AFTER normalize_xyz(), per-point comparison
# metrics against an identity baseline, hot/cold-bin FPR reuse from the
# S5-14 supplement module, and condition-level aggregation. No CUDA/torch
# needed (run_condition_forward/process_video_coordinate_transform/main()
# are the only torch-touching pieces and are validated for real only via
# the T2 identity-preflight canonical-parity check on GPU, matching
# check_dummy_per_window_context_diagnostics.py's own convention).
# ----------------------------------------------------------------------------


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def test_transform_conditions_fixed_and_parity_gated_subset() -> None:
    assert TRANSFORM_CONDITIONS == (
        "identity",
        "repeat_identity",
        "translate_x_plus",
        "translate_x_minus",
        "translate_y_plus",
        "translate_y_minus",
        "rotate_z_plus",
        "rotate_z_minus",
    ), TRANSFORM_CONDITIONS
    assert PARITY_GATED_CONDITIONS == ("identity", "repeat_identity")
    assert TRANSLATION_MAGNITUDE == 0.1
    assert ROTATION_DEGREES == 15.0
    print("ok: the 8 conditions and the identity/repeat_identity parity-gated subset are fixed as specified")


def test_apply_transform_identity_and_repeat_identity_are_unmodified() -> None:
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.1]])
    for condition in ("identity", "repeat_identity"):
        transformed, meta = apply_transform(points, condition)
        assert np.allclose(transformed, points), (condition, transformed)
        assert meta["kind"] == "identity"
        assert meta["translation"] is None and meta["rotation_degrees"] is None
    print("ok: identity/repeat_identity leave the input XYZ unmodified")


def test_apply_transform_translation_is_invertible_and_axis_isolated() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(scale=0.3, size=(20, 3))
    plus, meta_plus = apply_transform(points, "translate_x_plus")
    minus, meta_minus = apply_transform(points, "translate_x_minus")
    # Applying the opposite-signed translation to the already-translated
    # points must recover the original within float32 round-trip tolerance.
    undone = plus.astype(np.float64) + np.array(meta_minus["translation"])
    assert np.allclose(undone, points, atol=1e-5), (undone - points)
    # X moves, Y/Z are untouched.
    assert np.allclose(plus[:, 1:], points[:, 1:].astype(np.float32), atol=1e-6)
    assert np.allclose(plus[:, 0] - points[:, 0], TRANSLATION_MAGNITUDE, atol=1e-6)
    assert np.allclose(minus[:, 0] - points[:, 0], -TRANSLATION_MAGNITUDE, atol=1e-6)
    print("ok: translate_x_plus/minus are inverses of each other and only move the X axis")

    plus_y, _ = apply_transform(points, "translate_y_plus")
    assert np.allclose(plus_y[:, 0], points[:, 0].astype(np.float32), atol=1e-6)
    assert np.allclose(plus_y[:, 2], points[:, 2].astype(np.float32), atol=1e-6)
    print("ok: translate_y_plus only moves the Y axis")


def test_apply_transform_rotation_preserves_pairwise_distance_and_is_orthogonal() -> None:
    matrix = rotation_matrix_z(ROTATION_DEGREES)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10), matrix
    assert abs(np.linalg.det(matrix) - 1.0) < 1e-10
    print(f"ok: rotation_matrix_z({ROTATION_DEGREES}) is orthogonal with determinant 1 (a proper rotation)")

    rng = np.random.default_rng(1)
    points = rng.normal(scale=0.3, size=(10, 3))
    rotated, meta = apply_transform(points, "rotate_z_plus")
    assert meta["rotation_degrees"] == ROTATION_DEGREES
    original_distance = float(np.linalg.norm(points[0] - points[1]))
    rotated_distance = float(np.linalg.norm(rotated[0].astype(np.float64) - rotated[1].astype(np.float64)))
    assert abs(original_distance - rotated_distance) < 1e-5, (original_distance, rotated_distance)
    print(f"ok: rotate_z_plus preserves pairwise point distance ({original_distance:.6f} vs {rotated_distance:.6f})")

    rotated_minus, meta_minus = apply_transform(points, "rotate_z_minus")
    assert meta_minus["rotation_degrees"] == -ROTATION_DEGREES
    # rotate_z_plus then rotate_z_minus (applied to the ALREADY-rotated points) must undo the rotation.
    matrix_minus = rotation_matrix_z(-ROTATION_DEGREES)
    undone = rotated.astype(np.float64) @ matrix_minus.T
    assert np.allclose(undone, points, atol=1e-4), (undone - points)
    print("ok: rotate_z_plus followed by rotate_z_minus (composed) recovers the original points")


def test_apply_transform_unit_norm_exceeded_rate() -> None:
    # A point already near the unit-ball surface: a +0.1 X translation should
    # push it outside; a point near the origin should stay inside.
    points = np.array([[0.95, 0.0, 0.0], [0.0, 0.0, 0.0]])
    transformed, meta = apply_transform(points, "translate_x_plus")
    assert meta["unit_norm_exceeded_rate"] == 0.5, meta
    print(f"ok: unit_norm_exceeded_rate correctly flags the point pushed outside the unit ball: {meta['unit_norm_exceeded_rate']}")

    expect_fail("apply_transform rejects an unknown condition", lambda: apply_transform(points, "scale_up"))
    expect_fail("apply_transform rejects non-[N,3] input", lambda: apply_transform(np.zeros((3, 2)), "identity"))


def test_pre_normalization_translation_is_cancelled_but_post_normalization_translation_is_not() -> None:
    """Two related but distinct facts, kept explicitly separate per the
    2026-09-15 policy-chat correction (report-to-policy-chat section 12.3):
    (1) normalize_xyz() cancels a uniform translation applied BEFORE it (its
    own mean-centering) -- this justifies injecting the transform AFTER
    normalize_xyz() instead, and is NOT an explanation for why
    translate_x/y_plus/minus (which ARE applied after normalization) showed
    low sensitivity in the real forward-pass results; that is a separate
    empirical finding about the model, not a normalization artifact.
    (2) A translation applied AFTER normalize_xyz() is NOT cancelled by
    anything -- apply_transform() actually moves the points.
    """
    rng = np.random.default_rng(2)
    raw_points = rng.normal(scale=1.0, size=(30, 3)).astype(np.float32)
    offset = np.array([5.0, -3.0, 2.0], dtype=np.float32)
    baseline = normalize_xyz(raw_points)
    shifted_before = normalize_xyz(raw_points + offset[None, :])
    assert np.allclose(baseline, shifted_before, atol=1e-4), np.abs(baseline - shifted_before).max()
    print(
        "ok: a uniform PRE-normalize_xyz() translation is fully cancelled by its own mean-centering "
        "(this justifies the injection point -- it does NOT explain the post-normalization translation results)"
    )

    shifted_after, _ = apply_transform(baseline, "translate_x_plus")
    assert not np.allclose(baseline, shifted_after, atol=1e-4)
    assert np.allclose(shifted_after[:, 0] - baseline[:, 0], TRANSLATION_MAGNITUDE, atol=1e-6)
    print(
        "ok: a translation applied AFTER normalize_xyz() (what this diagnosis actually runs) is NOT cancelled -- "
        "any low sensitivity seen in the real results is a separate empirical finding, not a normalization artifact"
    )


def test_per_point_condition_metrics_flip_and_confusion() -> None:
    labels = np.array([1, 1, 0, 0, -1], dtype=np.int64)  # pos, pos, bg, bg, ignore
    valid_mask = np.array([True, True, True, True, True])
    baseline_pred = np.array([1, 0, 0, 0, 0], dtype=np.uint8)  # TP, FN, TN, TN, (ignore)
    condition_pred = np.array([1, 1, 1, 0, 1], dtype=np.uint8)  # TP, TP(flip), FP(flip), TN, ignore->positive

    baseline_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.5])
    condition_prob = np.array([0.95, 0.6, 0.7, 0.2, 0.9])

    metrics = per_point_condition_metrics(
        baseline_prob_positive=baseline_prob,
        baseline_pred_label=baseline_pred,
        condition_prob_positive=condition_prob,
        condition_pred_label=condition_pred,
        labels=labels,
        valid_mask=valid_mask,
        ignore_index=-1,
    )
    # flip_rate/flip_to_*_count are whole-point metrics (not restricted to
    # valid points): index 1 (FN->TP), index 2 (TN->FP), and index 4 (the
    # ignore point itself, background->predicted-positive) all flip 0->1.
    assert metrics["flip_to_positive_count"] == 3, metrics
    assert metrics["flip_to_background_count"] == 0
    assert metrics["true_positive_count"] == 2  # index 0, 1
    assert metrics["false_positive_count"] == 1  # index 2
    assert metrics["true_negative_count"] == 1  # index 3
    assert metrics["false_negative_count"] == 0
    assert metrics["baseline_ignore_predicted_positive_rate"] == 0.0
    assert metrics["ignore_predicted_positive_rate"] == 1.0  # the ignore point flipped to predicted-positive
    assert abs(metrics["recall_diff"] - (1.0 - 0.5)) < 1e-9  # baseline recall 1/2, condition recall 2/2
    print(f"ok: per_point_condition_metrics flip/confusion/diff/ignore-rate bookkeeping: {metrics}")


def test_per_point_condition_metrics_undefined_denominators_stay_none() -> None:
    labels = np.array([0, 0], dtype=np.int64)  # no GT-positive points at all
    valid_mask = np.array([True, True])
    pred = np.array([0, 0], dtype=np.uint8)
    metrics = per_point_condition_metrics(
        baseline_prob_positive=np.array([0.1, 0.2]),
        baseline_pred_label=pred,
        condition_prob_positive=np.array([0.1, 0.2]),
        condition_pred_label=pred,
        labels=labels,
        valid_mask=valid_mask,
        ignore_index=-1,
    )
    assert metrics["recall"] is None and metrics["recall_diff"] is None
    assert metrics["true_positive_count_is_zero"] is True
    print("ok: an undefined recall denominator (no GT-positive points) stays None, not 0-filled")


def make_gt_only_h5(path: Path, *, frame_order: np.ndarray, point_label: np.ndarray, pixel_xy: np.ndarray, video_name: str) -> None:
    n = frame_order.shape[0]
    with h5py.File(path, "w") as f:
        pc = f.create_group("point_cloud")
        pc.attrs["point_mode"] = "grid"
        pc.create_dataset("points", data=np.zeros((n, 3), dtype=np.float32))
        pc.create_dataset("intensity", data=np.zeros(n, dtype=np.uint8))
        pc.create_dataset("alpha", data=np.zeros(n, dtype=np.uint8))
        pc.create_dataset("confidence", data=np.ones(n, dtype=np.float32))
        pc.create_dataset("frame_order", data=frame_order.astype(np.int64))
        pc.create_dataset("pixel_xy", data=pixel_xy.astype(np.float32))
        ann = f.create_group("annotation")
        ann.create_dataset("point_label", data=point_label.astype(np.int8))
        ann.create_dataset("valid_mask", data=np.ones(n, dtype=bool))
        f.attrs["video_name"] = video_name


def test_build_video_hot_cold_masks_self_exclusion() -> None:
    """2026-09-15 policy-chat correction: a video that is itself part of the
    train list (a train-sanity video) must get a leave-one-video-out
    (self-excluded) classification, reusing S5-14 supplement's own
    `prior_excluding_video()` unmodified; a video outside the train list
    (a validation video) must be unaffected."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        h5_a = root / "a.h5"
        make_gt_only_h5(
            h5_a, frame_order=np.arange(4, dtype=np.int64), point_label=np.array([1, 1, 1, 1]),
            pixel_xy=np.tile([10.0, 10.0], (4, 1)), video_name="a",
        )
        h5_b = root / "b.h5"
        make_gt_only_h5(
            h5_b, frame_order=np.arange(20, dtype=np.int64), point_label=np.array([1] * 10 + [0] * 10),
            pixel_xy=np.tile([200.0, 200.0], (20, 1)), video_name="b",
        )
        prior = build_train_prior([h5_a, h5_b], ignore_index=-1, pixel_width=256, pixel_height=256, grid_resolutions=[2])

        masks_a = build_video_hot_cold_masks(prior, h5_path=h5_a, grid_resolutions=(2,), definitions=("raw_count",))
        assert masks_a[(2, "raw_count")]["self_excluded"] is True
        print("ok: a train-list video (a.h5) gets self_excluded=True (leave-one-video-out)")

        # A video not in the train list at all (e.g. a validation video).
        h5_c = root / "c.h5"
        make_gt_only_h5(
            h5_c, frame_order=np.arange(1, dtype=np.int64), point_label=np.array([1]),
            pixel_xy=np.array([[10.0, 10.0]]), video_name="c",
        )
        masks_c = build_video_hot_cold_masks(prior, h5_path=h5_c, grid_resolutions=(2,), definitions=("raw_count",))
        assert masks_c[(2, "raw_count")]["self_excluded"] is False
        print("ok: a non-train-list video (c.h5) gets self_excluded=False (full 162-video-equivalent prior, unchanged)")


def test_condition_hot_cold_rows_with_per_video_masks_and_undefined_branch() -> None:
    hot_mask = np.array([[True, False], [False, False]])
    cold_mask = np.array([[False, False], [False, True]])
    video_hot_cold_masks = {
        (2, "rate"): {"hot_mask": hot_mask, "cold_mask": cold_mask, "status": "ok", "self_excluded": True},
        (2, "raw_count"): {"hot_mask": None, "cold_mask": None, "status": "undefined_boundary_collision", "self_excluded": True},
    }
    # Baseline (identity): no FP at all -> hot/cold FPR both 0. Condition: 2
    # new FPs appear, both in the hot bin (0,0) -- exactly the "coordinate
    # shift concentrates new FPs in the previously-hot region" signal this
    # diagnosis is built to detect.
    labels = np.zeros(8, dtype=np.int64)
    valid_mask = np.ones(8, dtype=bool)
    xy_normalized = np.array(
        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]]
    )
    baseline_pred = np.zeros(8, dtype=np.uint8)
    condition_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)

    rows = condition_hot_cold_rows(
        video_alias="train_sanity_000", split="train_sanity", condition="translate_x_plus", xy_normalized=xy_normalized,
        labels=labels, valid_mask=valid_mask, pred_label=condition_pred, baseline_pred_label=baseline_pred,
        ignore_index=-1, video_hot_cold_masks=video_hot_cold_masks,
    )
    assert len(rows) == 2
    by_definition = {r["definition"]: r for r in rows}

    ok_row = by_definition["rate"]
    assert abs(ok_row["hot_fpr"] - 0.5) < 1e-9, ok_row  # 2 FP / 4 background in the hot bin
    assert ok_row["cold_fpr"] == 0.0
    assert abs(ok_row["hot_fpr_diff_vs_identity"] - 0.5) < 1e-9
    assert ok_row["self_excluded_from_prior"] is True
    assert ok_row["quantile_status"] == "ok"
    print(f"ok: condition_hot_cold_rows (defined mask) detects new FPs concentrated in the hot bin and records self_excluded_from_prior: {ok_row}")

    undefined_row = by_definition["raw_count"]
    assert undefined_row["hot_fpr"] is None and undefined_row["hot_fpr_available"] is False
    assert undefined_row["hot_fpr_diff_vs_identity"] is None
    assert undefined_row["quantile_status"] == "undefined_boundary_collision"
    print(f"ok: condition_hot_cold_rows (undefined mask) reports None with the quantile_status reason, never a fabricated FPR: {undefined_row}")


def test_aggregate_condition_summary_and_hot_cold_summary() -> None:
    point_rows = [
        {"condition": "translate_x_plus", "flip_rate": 0.1, "prob_diff_abs_mean": 0.05, "false_positive_rate_diff": 0.2, "recall_diff": -0.1, "true_positive_count_is_zero": False},
        {"condition": "translate_x_plus", "flip_rate": 0.3, "prob_diff_abs_mean": 0.15, "false_positive_rate_diff": -0.05, "recall_diff": 0.02, "true_positive_count_is_zero": True},
        {"condition": "identity", "flip_rate": 0.0, "prob_diff_abs_mean": 0.0, "false_positive_rate_diff": 0.0, "recall_diff": 0.0, "true_positive_count_is_zero": False},
    ]
    summary = aggregate_condition_summary(point_rows)
    assert summary["translate_x_plus"]["num_videos"] == 2
    assert summary["translate_x_plus"]["num_videos_tp_zero"] == 1
    assert abs(summary["translate_x_plus"]["flip_rate"]["mean"] - 0.2) < 1e-9
    assert summary["translate_x_plus"]["false_positive_rate_diff"]["num_videos_increased"] == 1
    assert summary["translate_x_plus"]["false_positive_rate_diff"]["num_videos_decreased"] == 1
    print(f"ok: aggregate_condition_summary per-condition mean/median and increased/decreased counts: {summary['translate_x_plus']}")

    hot_cold_rows = [
        {"condition": "translate_x_plus", "grid_resolution": 16, "definition": "rate", "hot_fpr_diff_vs_identity": 0.3},
        {"condition": "translate_x_plus", "grid_resolution": 16, "definition": "rate", "hot_fpr_diff_vs_identity": -0.1},
        {"condition": "translate_x_plus", "grid_resolution": 16, "definition": "rate", "hot_fpr_diff_vs_identity": None},
    ]
    hc_summary = aggregate_hot_cold_condition_summary(hot_cold_rows)
    key = "translate_x_plus__grid16__rate"
    assert hc_summary[key]["num_videos_total"] == 3
    assert hc_summary[key]["num_videos_available"] == 2
    assert hc_summary[key]["num_videos_hot_fpr_increased"] == 1
    assert hc_summary[key]["num_videos_hot_fpr_decreased"] == 1
    print(f"ok: aggregate_hot_cold_condition_summary excludes unavailable (None) diffs from counts but keeps num_videos_total: {hc_summary[key]}")


def test_split_filtering_before_aggregation_separates_populations() -> None:
    """2026-09-15 policy-chat correction: condition_summary/hot_cold_summary
    must not mix train_sanity and validation. aggregate_condition_summary()
    itself stays split-agnostic; the fix is at the call site -- filter
    point_rows/hot_cold_rows by `split` before aggregating (once per split,
    plus an explicitly-labeled mixed reference), exactly as `main()` now
    does. This test documents and locks in that corrected usage pattern."""
    point_rows = [
        {
            "video_alias": "train_sanity_000", "split": "train_sanity", "condition": "rotate_z_plus",
            "flip_rate": 0.5, "prob_diff_abs_mean": 0.1, "false_positive_rate_diff": -0.3, "recall_diff": -0.2,
            "true_positive_count_is_zero": False,
        },
        {
            "video_alias": "validation_000", "split": "validation", "condition": "rotate_z_plus",
            "flip_rate": 0.05, "prob_diff_abs_mean": 0.01, "false_positive_rate_diff": 0.01, "recall_diff": 0.0,
            "true_positive_count_is_zero": False,
        },
    ]
    train_sanity_summary = aggregate_condition_summary([r for r in point_rows if r["split"] == "train_sanity"])
    validation_summary = aggregate_condition_summary([r for r in point_rows if r["split"] == "validation"])
    mixed_reference_summary = aggregate_condition_summary(point_rows)

    assert train_sanity_summary["rotate_z_plus"]["num_videos"] == 1
    assert validation_summary["rotate_z_plus"]["num_videos"] == 1
    assert mixed_reference_summary["rotate_z_plus"]["num_videos"] == 2
    assert train_sanity_summary["rotate_z_plus"]["flip_rate"]["mean"] != validation_summary["rotate_z_plus"]["flip_rate"]["mean"]
    print(
        "ok: filtering point_rows by split before aggregation keeps train_sanity/validation populations "
        f"separate (train_sanity flip_rate={train_sanity_summary['rotate_z_plus']['flip_rate']['mean']}, "
        f"validation flip_rate={validation_summary['rotate_z_plus']['flip_rate']['mean']}), distinct from the "
        f"mixed reference aggregate (num_videos={mixed_reference_summary['rotate_z_plus']['num_videos']})"
    )


def main() -> None:
    test_transform_conditions_fixed_and_parity_gated_subset()
    test_apply_transform_identity_and_repeat_identity_are_unmodified()
    test_apply_transform_translation_is_invertible_and_axis_isolated()
    test_apply_transform_rotation_preserves_pairwise_distance_and_is_orthogonal()
    test_apply_transform_unit_norm_exceeded_rate()
    test_pre_normalization_translation_is_cancelled_but_post_normalization_translation_is_not()
    test_per_point_condition_metrics_flip_and_confusion()
    test_per_point_condition_metrics_undefined_denominators_stay_none()
    test_build_video_hot_cold_masks_self_exclusion()
    test_condition_hot_cold_rows_with_per_video_masks_and_undefined_branch()
    test_aggregate_condition_summary_and_hot_cold_summary()
    test_split_filtering_before_aggregation_separates_populations()
    print("check_dummy_coordinate_transform_diagnostics: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
