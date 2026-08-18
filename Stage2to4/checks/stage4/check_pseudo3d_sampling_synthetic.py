from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.utils.alpha_texture_processing import AlphaTextureConfig, image_to_uint8_gray
from src.utils.pseudo3d_sampling import (
    SOURCE_CONTEXT_GRID,
    SOURCE_GLOBAL,
    SOURCE_LOCAL_PERCENTILE,
    SOURCE_TOPHAT,
    PointSamplingConfig,
    build_context_grid_mask,
    cleanup_evidence_mask,
    build_local_percentile_evidence,
    build_sampling_confidence,
    build_sampling_source_flags,
    build_tophat_evidence,
    selected_pixel_arrays,
    source_type_from_flags,
    summarize_per_frame_counts,
    summarize_source_flags,
)
from pseudo3d.export.export_pseudo3d_point_cloud import (
    build_frame_point_cloud,
    project_frame_points_to_world,
)


def _identity_tracking(num_frames: int) -> np.ndarray:
    return np.repeat(np.eye(4, dtype=np.float32)[None, ...], num_frames, axis=0)


def _translation(tx: float, ty: float, tz: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def _assert_raises_value_error(func: object) -> None:
    try:
        func()  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def _assert_point_cloud_array_lengths(
    point_cloud: dict[str, np.ndarray],
    *,
    num_frames: int,
) -> None:
    num_points = int(point_cloud["points"].shape[0])
    point_keys = (
        "points",
        "intensity",
        "alpha",
        "frame_index",
        "frame_order",
        "pixel_xy",
        "confidence",
        "source_type",
        "source_flags",
        "sampling_confidence",
    )
    frame_keys = (
        "per_frame_counts",
        "per_frame_global_counts",
        "per_frame_local_percentile_counts",
        "per_frame_tophat_counts",
        "per_frame_context_grid_counts",
        "per_frame_context_only_counts",
        "per_frame_evidence_counts",
        "per_frame_overlap_counts",
    )

    for key in point_keys:
        assert point_cloud[key].shape[0] == num_points, key
    for key in frame_keys:
        assert point_cloud[key].shape == (num_frames,), key
        assert point_cloud[key].dtype == np.int64, key


def test_context_grid_counts() -> None:
    shape = (16, 16)

    origin = build_context_grid_mask(shape, stride=4, phase="origin")
    centered = build_context_grid_mask(shape, stride=4, phase="centered")
    dual = build_context_grid_mask(shape, stride=4, phase="dual")

    assert int(origin.sum()) == 16
    assert int(centered.sum()) == 16
    assert int(dual.sum()) == 32
    assert not np.any(origin & centered)


def test_context_grid_boundaries_and_validation() -> None:
    shape = (15, 17)
    origin = build_context_grid_mask(shape, stride=4, phase="origin")
    centered = build_context_grid_mask(shape, stride=4, phase="centered")
    dual = build_context_grid_mask(shape, stride=4, phase="dual")

    assert origin.dtype == np.bool_
    assert int(origin.sum()) == 20
    assert int(centered.sum()) == 16
    assert int(dual.sum()) == 36
    assert origin[0, 0] and origin[12, 16]
    assert centered[2, 2] and centered[14, 14]
    assert np.array_equal(dual, origin | centered)

    stride_one = build_context_grid_mask((5, 7), stride=1, phase="dual")
    assert np.all(stride_one)

    large_stride_origin = build_context_grid_mask(
        (3, 4), stride=10, phase="origin"
    )
    large_stride_centered = build_context_grid_mask(
        (3, 4), stride=10, phase="centered"
    )
    large_stride_dual = build_context_grid_mask(
        (3, 4), stride=10, phase="dual"
    )
    assert int(large_stride_origin.sum()) == 1
    assert int(large_stride_centered.sum()) == 0
    assert np.array_equal(
        large_stride_dual,
        large_stride_origin | large_stride_centered,
    )

    _assert_raises_value_error(
        lambda: build_context_grid_mask((5, 7), stride=0, phase="origin")
    )
    _assert_raises_value_error(
        lambda: build_context_grid_mask((5, 7), stride=-1, phase="origin")
    )
    _assert_raises_value_error(
        lambda: build_context_grid_mask((5, 7), stride=4, phase="unknown")
    )
    _assert_raises_value_error(
        lambda: build_context_grid_mask((0, 7), stride=4, phase="origin")
    )
    _assert_raises_value_error(
        lambda: build_context_grid_mask((5, 7, 1), stride=4, phase="origin")
    )


def test_local_percentile_evidence() -> None:
    uniform = np.zeros((32, 32), dtype=np.uint8)
    uniform_result = build_local_percentile_evidence(
        uniform,
        window_size=9,
        percentile=80.0,
        min_contrast=4.0,
    )
    assert int(uniform_result["mask"].sum()) == 0

    image = np.zeros((32, 32), dtype=np.uint8)
    image[16, 8:24] = 160
    image[16, 16] = 220
    result = build_local_percentile_evidence(
        image,
        window_size=9,
        percentile=80.0,
        min_contrast=4.0,
    )
    assert int(result["mask"].sum()) > 0
    assert bool(result["mask"][16, 16])


def test_local_percentile_debug_boundaries_and_validation() -> None:
    for value in (20, 180):
        uniform = np.full((32, 32), value, dtype=np.uint8)
        result = build_local_percentile_evidence(
            uniform,
            window_size=9,
            percentile=80.0,
            min_contrast=4.0,
        )
        assert not np.any(result["mask"])

    image = np.full((64, 64), 20, dtype=np.uint8)
    targets = [(8, 8), (0, 63)]
    targets += [(18, x) for x in range(14, 19)]
    targets += [(y, 10) for y in range(30, 35)]
    targets += [(20, 38), (19, 38), (21, 38), (20, 37), (20, 39)]
    targets += [(42 + index, 38 + index) for index in range(5)]
    for y, x in targets:
        image[y, x] = 120

    debug = build_local_percentile_evidence(
        image,
        window_size=9,
        percentile=80.0,
        min_contrast=4.0,
        return_debug_maps=True,
    )
    for y, x in targets:
        assert debug["mask"][y, x], (y, x)

    assert set(debug) == {
        "mask",
        "threshold_map",
        "median_map",
        "contrast_map",
    }
    for key in ("threshold_map", "median_map", "contrast_map"):
        assert debug[key].shape == image.shape
        assert debug[key].dtype == np.float32

    expected_contrast = image.astype(np.float32) - debug["median_map"]
    np.testing.assert_allclose(debug["contrast_map"], expected_contrast)
    expected_mask = (
        (image.astype(np.float32) >= debug["threshold_map"])
        & (debug["contrast_map"] >= 4.0)
    )
    assert np.array_equal(debug["mask"], expected_mask)

    without_debug = build_local_percentile_evidence(
        image,
        window_size=9,
        percentile=80.0,
        min_contrast=4.0,
    )
    assert set(without_debug) == {"mask"}

    counts = []
    for contrast in (0.0, 4.0, 200.0):
        mask = build_local_percentile_evidence(
            image,
            window_size=9,
            percentile=80.0,
            min_contrast=contrast,
        )["mask"]
        counts.append(int(mask.sum()))
    assert counts[0] >= counts[1] >= counts[2] == 0

    window_one_all = build_local_percentile_evidence(
        image, window_size=1, percentile=80.0, min_contrast=0.0
    )["mask"]
    window_one_empty = build_local_percentile_evidence(
        image, window_size=1, percentile=80.0, min_contrast=4.0
    )["mask"]
    assert np.all(window_one_all)
    assert not np.any(window_one_empty)

    percentile_low = build_local_percentile_evidence(
        image, window_size=9, percentile=-10.0, min_contrast=4.0
    )["mask"]
    percentile_zero = build_local_percentile_evidence(
        image, window_size=9, percentile=0.0, min_contrast=4.0
    )["mask"]
    percentile_high = build_local_percentile_evidence(
        image, window_size=9, percentile=110.0, min_contrast=4.0
    )["mask"]
    percentile_hundred = build_local_percentile_evidence(
        image, window_size=9, percentile=100.0, min_contrast=4.0
    )["mask"]
    assert np.array_equal(percentile_low, percentile_zero)
    assert np.array_equal(percentile_high, percentile_hundred)

    repeat = build_local_percentile_evidence(
        image,
        window_size=9,
        percentile=80.0,
        min_contrast=4.0,
        return_debug_maps=True,
    )
    for key in debug:
        assert np.array_equal(debug[key], repeat[key]), key

    for invalid_window in (0, -3, 8):
        _assert_raises_value_error(
            lambda value=invalid_window: build_local_percentile_evidence(
                image, window_size=value
            )
        )


def test_tophat_evidence() -> None:
    uniform = np.full((32, 32), 30, dtype=np.uint8)
    uniform_result = build_tophat_evidence(
        uniform,
        kernel_size=9,
        percentile=85.0,
        min_response=2.0,
        morph_shape="ellipse",
    )
    assert int(uniform_result["mask"].sum()) == 0

    image = np.zeros((32, 32), dtype=np.uint8)
    image[14:18, 14:18] = 180
    result = build_tophat_evidence(
        image,
        kernel_size=9,
        percentile=80.0,
        min_response=2.0,
        morph_shape="ellipse",
    )
    assert int(result["mask"].sum()) > 0
    assert bool(result["mask"][16, 16])


def test_tophat_debug_shapes_and_validation() -> None:
    for value in (0, 20, 180, 255):
        uniform = np.full((32, 32), value, dtype=np.uint8)
        result = build_tophat_evidence(
            uniform,
            kernel_size=9,
            percentile=85.0,
            min_response=2.0,
            return_debug_maps=True,
        )
        assert not np.any(result["mask"])
        assert np.all(result["response_map"] == 0)
        assert np.isinf(float(result["threshold"]))

    image = np.full((64, 64), 20, dtype=np.uint8)
    targets = [(8, 8), (0, 63)]
    targets += [(18, x) for x in range(14, 19)]
    targets += [(y, 10) for y in range(30, 35)]
    targets += [(20, 38), (19, 38), (21, 38), (20, 37), (20, 39)]
    targets += [(42 + index, 38 + index) for index in range(5)]
    for y, x in targets:
        image[y, x] = 180

    shape_result = build_tophat_evidence(
        image,
        kernel_size=9,
        percentile=80.0,
        min_response=2.0,
    )
    for y, x in targets:
        assert shape_result["mask"][y, x], (y, x)

    percentile_image = np.full((21, 21), 20, dtype=np.uint8)
    responses = {(4, 4): 10, (4, 16): 20, (16, 4): 30, (16, 16): 40}
    for (y, x), response in responses.items():
        percentile_image[y, x] = 20 + response
    percentile_result = build_tophat_evidence(
        percentile_image,
        kernel_size=7,
        percentile=50.0,
        min_response=0.0,
        return_debug_maps=True,
    )
    positive = percentile_result["response_map"]
    positive = positive[positive > 0]
    np.testing.assert_allclose(
        np.sort(positive), np.asarray([10, 20, 30, 40], dtype=np.float32)
    )
    assert float(percentile_result["threshold"]) == 25.0
    assert int(percentile_result["mask"].sum()) == 2

    response_image = np.full((32, 32), 20, dtype=np.uint8)
    response_image[6, 6] = 30
    response_image[16, 16] = 60
    response_image[26, 26] = 120
    response_counts = []
    for minimum in (0.0, 20.0, 50.0, 200.0):
        mask = build_tophat_evidence(
            response_image,
            kernel_size=7,
            percentile=0.0,
            min_response=minimum,
        )["mask"]
        response_counts.append(int(mask.sum()))
    assert response_counts == [3, 2, 1, 0]

    morph_image = np.full((33, 33), 20, dtype=np.uint8)
    morph_image[15:18, 15:18] = 180
    for morph_shape in ("rect", "ellipse", "cross"):
        morph_result = build_tophat_evidence(
            morph_image,
            kernel_size=9,
            percentile=80.0,
            min_response=2.0,
            morph_shape=morph_shape,
        )
        assert morph_result["mask"][16, 16], morph_shape

    debug = build_tophat_evidence(
        morph_image,
        kernel_size=9,
        percentile=80.0,
        min_response=2.0,
        return_debug_maps=True,
    )
    assert set(debug) == {"mask", "response_map", "opened_map", "threshold"}
    assert debug["response_map"].dtype == np.float32
    assert debug["opened_map"].dtype == np.uint8
    expected_response = (
        morph_image.astype(np.float32) - debug["opened_map"].astype(np.float32)
    )
    np.testing.assert_allclose(debug["response_map"], expected_response)
    assert np.all(debug["response_map"] >= 0)

    positive = debug["response_map"][debug["response_map"] > 0]
    expected_threshold = np.percentile(positive, 80.0)
    assert float(debug["threshold"]) == float(expected_threshold)
    expected_mask = (
        (debug["response_map"] >= expected_threshold)
        & (debug["response_map"] >= 2.0)
    )
    assert np.array_equal(debug["mask"], expected_mask)

    kernel_one = build_tophat_evidence(
        morph_image,
        kernel_size=1,
        percentile=80.0,
        min_response=0.0,
        return_debug_maps=True,
    )
    assert not np.any(kernel_one["mask"])
    assert np.isinf(float(kernel_one["threshold"]))

    percentile_low = build_tophat_evidence(
        morph_image, kernel_size=9, percentile=-10.0, min_response=2.0
    )["mask"]
    percentile_zero = build_tophat_evidence(
        morph_image, kernel_size=9, percentile=0.0, min_response=2.0
    )["mask"]
    percentile_high = build_tophat_evidence(
        morph_image, kernel_size=9, percentile=110.0, min_response=2.0
    )["mask"]
    percentile_hundred = build_tophat_evidence(
        morph_image, kernel_size=9, percentile=100.0, min_response=2.0
    )["mask"]
    assert np.array_equal(percentile_low, percentile_zero)
    assert np.array_equal(percentile_high, percentile_hundred)

    repeat = build_tophat_evidence(
        morph_image,
        kernel_size=9,
        percentile=80.0,
        min_response=2.0,
        return_debug_maps=True,
    )
    for key in debug:
        assert np.array_equal(debug[key], repeat[key]), key

    for invalid_kernel in (0, -3, 8):
        _assert_raises_value_error(
            lambda value=invalid_kernel: build_tophat_evidence(
                morph_image, kernel_size=value
            )
        )
    _assert_raises_value_error(
        lambda: build_tophat_evidence(
            morph_image, kernel_size=9, morph_shape="unknown"
        )
    )


def test_source_flag_merge_and_confidence() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[16, 8:24] = 180
    image[16, 16] = 255
    global_mask = image > 220

    config = PointSamplingConfig(
        sampling_mode="combined_v2",
        include_context_grid=True,
        context_grid_stride=4,
        context_grid_phase="origin",
        enable_local_percentile=True,
        local_window_size=9,
        local_percentile=80.0,
        local_min_contrast=4.0,
        enable_tophat=True,
        tophat_kernel_size=9,
        tophat_percentile=80.0,
        tophat_min_response=2.0,
        tophat_morph_shape="ellipse",
    )
    sampling = build_sampling_source_flags(image, global_mask, config)
    flags = sampling["source_flags"]

    assert bool(flags[16, 16] & SOURCE_GLOBAL)
    assert bool(flags[16, 16] & SOURCE_LOCAL_PERCENTILE)
    assert bool(flags[16, 16] & SOURCE_TOPHAT)

    selected = selected_pixel_arrays(flags)
    assert int(sampling["selected_mask"].sum()) == selected["xs"].shape[0]
    assert selected["source_flags"].shape[0] <= (
        int(sampling["global_mask"].sum())
        + int(sampling["local_percentile_mask"].sum())
        + int(sampling["tophat_mask"].sum())
        + int(sampling["context_grid_mask"].sum())
    )

    confidence = build_sampling_confidence(
        flags,
        global_confidence=(image.astype(np.float32) / 255.0),
        local_confidence=0.7,
        tophat_confidence=0.7,
        context_confidence=0.0,
    )
    assert confidence.shape == image.shape
    assert float(confidence[16, 16]) == 1.0


def test_evidence_cleanup_and_source_isolation() -> None:
    raw_mask = np.zeros((32, 32), dtype=bool)
    raw_mask[8:24, 8:24] = True
    raw_mask[2, 2] = True
    raw_mask[28:30, 28:30] = True

    cleanup = cleanup_evidence_mask(
        raw_mask,
        open_ksize=3,
        close_ksize=5,
        morph_shape="ellipse",
        min_component_area=100,
        return_debug_maps=True,
    )
    assert cleanup["mask"].dtype == np.bool_
    assert cleanup["mask"].shape == raw_mask.shape
    assert cleanup["mask"][16, 16]
    assert not cleanup["mask"][2, 2]
    assert not cleanup["mask"][28, 28]
    assert np.array_equal(cleanup["raw_mask"], raw_mask)
    assert cleanup["morphology_mask"].shape == raw_mask.shape

    identity = cleanup_evidence_mask(
        raw_mask,
        open_ksize=0,
        close_ksize=0,
        min_component_area=0,
    )["mask"]
    assert np.array_equal(identity, raw_mask)

    rng = np.random.default_rng(123)
    image = rng.integers(0, 80, size=(64, 64), dtype=np.uint8)
    image[20:44, 20:44] = 220
    image[28:36, 28:36] = 255
    global_mask = np.zeros_like(image, dtype=bool)
    global_mask[0, 0] = True

    raw_config = PointSamplingConfig(
        sampling_mode="combined_v2",
        include_context_grid=True,
        context_grid_stride=4,
        enable_local_percentile=True,
        local_window_size=9,
        local_percentile=80.0,
        local_min_contrast=4.0,
        enable_tophat=True,
        tophat_kernel_size=9,
        tophat_percentile=80.0,
        tophat_min_response=2.0,
        enable_evidence_cleanup=False,
    )
    cleaned_config = replace(
        raw_config,
        enable_evidence_cleanup=True,
        evidence_open_ksize=3,
        evidence_close_ksize=5,
        evidence_morph_shape="ellipse",
        evidence_min_component_area=20,
    )
    raw = build_sampling_source_flags(image, global_mask, raw_config)
    cleaned = build_sampling_source_flags(
        image,
        global_mask,
        cleaned_config,
        return_debug_maps=True,
    )

    assert np.any(raw["local_percentile_mask"])
    assert np.any(raw["tophat_mask"])
    assert np.array_equal(cleaned["global_mask"], raw["global_mask"])
    assert np.array_equal(cleaned["context_grid_mask"], raw["context_grid_mask"])
    for source_name, prefix in (
        ("local_percentile_mask", "local"),
        ("tophat_mask", "tophat"),
    ):
        expected = cleanup_evidence_mask(
            raw[source_name],
            open_ksize=3,
            close_ksize=5,
            morph_shape="ellipse",
            min_component_area=20,
        )["mask"]
        assert np.array_equal(cleaned[source_name], expected), source_name
        assert np.array_equal(
            cleaned["debug_maps"][f"{prefix}_raw_evidence_mask"],
            raw[source_name],
        )

    defaults = PointSamplingConfig.combined_v2_defaults()
    assert defaults.enable_evidence_cleanup
    assert defaults.evidence_open_ksize == 3
    assert defaults.evidence_close_ksize == 5
    assert defaults.evidence_morph_shape == "ellipse"
    assert defaults.evidence_min_component_area == 100

    _assert_raises_value_error(
        lambda: cleanup_evidence_mask(np.zeros((2, 2, 1), dtype=bool))
    )
    _assert_raises_value_error(
        lambda: cleanup_evidence_mask(raw_mask, open_ksize=-1)
    )
    _assert_raises_value_error(
        lambda: cleanup_evidence_mask(raw_mask, close_ksize=-1)
    )
    _assert_raises_value_error(
        lambda: cleanup_evidence_mask(raw_mask, morph_shape="unknown")
    )
    _assert_raises_value_error(
        lambda: cleanup_evidence_mask(raw_mask, min_component_area=-1)
    )


def test_source_type_merge_and_statistics() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8, 8] = 255
    global_mask = np.zeros_like(image, dtype=bool)
    global_mask[8, 8] = True
    config = PointSamplingConfig(
        sampling_mode="combined_v2",
        include_context_grid=True,
        context_grid_stride=4,
        context_grid_phase="origin",
        enable_local_percentile=True,
        local_window_size=9,
        local_percentile=80.0,
        local_min_contrast=4.0,
        enable_tophat=True,
        tophat_kernel_size=9,
        tophat_percentile=80.0,
        tophat_min_response=2.0,
    )
    sampling = build_sampling_source_flags(image, global_mask, config)
    flags = sampling["source_flags"]
    all_sources = np.uint16(
        SOURCE_GLOBAL
        | SOURCE_LOCAL_PERCENTILE
        | SOURCE_TOPHAT
        | SOURCE_CONTEXT_GRID
    )
    assert flags[8, 8] == all_sources
    assert int(flags[8, 8]) == 15

    selected = selected_pixel_arrays(flags)
    coordinates = np.stack([selected["ys"], selected["xs"]], axis=1)
    assert np.unique(coordinates, axis=0).shape[0] == coordinates.shape[0]

    test_flags = np.asarray(
        [
            0,
            SOURCE_CONTEXT_GRID,
            SOURCE_TOPHAT | SOURCE_CONTEXT_GRID,
            SOURCE_LOCAL_PERCENTILE | SOURCE_TOPHAT | SOURCE_CONTEXT_GRID,
            all_sources,
        ],
        dtype=np.uint16,
    )
    expected_types = np.asarray([0, 4, 3, 2, 1], dtype=np.uint8)
    assert np.array_equal(source_type_from_flags(test_flags), expected_types)

    g = SOURCE_GLOBAL
    local = SOURCE_LOCAL_PERCENTILE
    top = SOURCE_TOPHAT
    context = SOURCE_CONTEXT_GRID
    manual_flags = np.asarray(
        [
            [0, g, local, top],
            [context, g | context, local | top, g | local | top | context],
            [g | local, top | context, context, 0],
        ],
        dtype=np.uint16,
    )
    stats = summarize_source_flags(manual_flags)
    expected_stats = {
        "total_selected_points": 10,
        "global_evidence_points": 4,
        "local_percentile_points": 4,
        "tophat_points": 4,
        "context_grid_points": 5,
        "context_only_points": 2,
        "evidence_points": 8,
        "overlap_points": 5,
        "legacy_point_count_multiplier": 2.5,
    }
    for key, expected in expected_stats.items():
        assert np.isclose(stats[key], expected), key

    no_global_stats = summarize_source_flags(
        np.asarray([context, local, top, local | context], dtype=np.uint16)
    )
    assert no_global_stats["legacy_point_count_multiplier"] == 0.0

    frame_stats = summarize_per_frame_counts(
        np.asarray([0, 2, 4, 10], dtype=np.int64)
    )
    assert frame_stats == {
        "points_per_frame_min": 0,
        "points_per_frame_mean": 4.0,
        "points_per_frame_median": 3.0,
        "points_per_frame_max": 10,
    }

    legacy_config = PointSamplingConfig(
        sampling_mode="legacy",
        include_context_grid=True,
        enable_local_percentile=True,
        enable_tophat=True,
    )
    legacy = build_sampling_source_flags(image, global_mask, legacy_config)
    assert int(legacy["selected_mask"].sum()) == 1
    assert legacy["source_flags"][8, 8] == SOURCE_GLOBAL
    assert not np.any(legacy["local_percentile_mask"])
    assert not np.any(legacy["tophat_mask"])
    assert not np.any(legacy["context_grid_mask"])

    _assert_raises_value_error(
        lambda: build_sampling_source_flags(
            image, np.zeros((16, 16), dtype=bool), config
        )
    )
    _assert_raises_value_error(
        lambda: build_sampling_source_flags(
            image,
            global_mask,
            PointSamplingConfig(sampling_mode="unknown"),
        )
    )


def test_sampling_confidence_alignment_and_validation() -> None:
    g = SOURCE_GLOBAL
    local = SOURCE_LOCAL_PERCENTILE
    top = SOURCE_TOPHAT
    context = SOURCE_CONTEXT_GRID
    flags = np.asarray(
        [
            [0, g, local, top],
            [context, g | local, g | top, g | context],
            [local | top, local | context, top | context, g | local | top | context],
        ],
        dtype=np.uint16,
    )
    global_confidence = np.asarray(
        [
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.50, 0.9, 0.4],
            [0.0, 0.00, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    confidence = build_sampling_confidence(
        flags,
        global_confidence=global_confidence,
        local_confidence=0.7,
        tophat_confidence=0.7,
        context_confidence=0.0,
    )
    expected = np.asarray(
        [
            [0.0, 0.25, 0.7, 0.7],
            [0.0, 0.70, 0.9, 0.4],
            [0.7, 0.70, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    assert confidence.dtype == np.float32
    np.testing.assert_allclose(confidence, expected)

    selected = selected_pixel_arrays(flags, sampling_confidence=confidence)
    ys, xs = np.nonzero(flags != 0)
    np.testing.assert_allclose(
        selected["sampling_confidence"], confidence[ys, xs]
    )

    repeat = build_sampling_confidence(
        flags,
        global_confidence=global_confidence,
        local_confidence=0.7,
        tophat_confidence=0.7,
        context_confidence=0.0,
    )
    assert np.array_equal(confidence, repeat)

    _assert_raises_value_error(
        lambda: build_sampling_confidence(
            flags, global_confidence=np.zeros((2, 2), dtype=np.float32)
        )
    )
    _assert_raises_value_error(
        lambda: selected_pixel_arrays(
            flags, sampling_confidence=np.zeros((2, 2), dtype=np.float32)
        )
    )


def test_build_frame_point_cloud_legacy_compatibility() -> None:
    images = np.zeros((2, 16, 16), dtype=np.float32)
    images[0, 4, 4] = 1.0
    images[0, 4, 8] = 0.5
    images[1, 8, 8] = 1.0

    point_cloud = build_frame_point_cloud(
        local_images=images,
        pred_tracking=_identity_tracking(2),
        pixel_to_image=np.eye(4, dtype=np.float32),
        frame_indices=np.asarray([10, 11], dtype=np.int64),
        texture_config=AlphaTextureConfig(texture_style="original"),
        min_alpha=1,
        min_intensity=1,
        original_foreground_mode="nonzero",
        point_mode="foreground",
        sampling_config=PointSamplingConfig(sampling_mode="legacy"),
        sample_stride=1,
        max_points_per_frame=0,
        random_seed=0,
    )

    assert point_cloud["points"].shape == (3, 3)
    assert point_cloud["per_frame_counts"].tolist() == [2, 1]
    assert np.all(point_cloud["source_flags"] == SOURCE_GLOBAL)
    assert np.all(point_cloud["source_type"] == 0)
    assert np.allclose(point_cloud["sampling_confidence"], 1.0)


def test_build_frame_point_cloud_combined_v2() -> None:
    images = np.zeros((1, 32, 32), dtype=np.float32)
    images[0, 16, 8:24] = 180.0 / 255.0
    images[0, 16, 16] = 1.0

    config = PointSamplingConfig(
        sampling_mode="combined_v2",
        include_context_grid=True,
        context_grid_stride=8,
        context_grid_phase="dual",
        enable_local_percentile=True,
        local_window_size=9,
        local_percentile=80.0,
        local_min_contrast=4.0,
        enable_tophat=True,
        tophat_kernel_size=9,
        tophat_percentile=80.0,
        tophat_min_response=2.0,
        tophat_morph_shape="ellipse",
    )

    point_cloud = build_frame_point_cloud(
        local_images=images,
        pred_tracking=_identity_tracking(1),
        pixel_to_image=np.eye(4, dtype=np.float32),
        frame_indices=np.asarray([20], dtype=np.int64),
        texture_config=AlphaTextureConfig(texture_style="original"),
        min_alpha=1,
        min_intensity=1,
        original_foreground_mode="nonzero",
        point_mode="foreground",
        sampling_config=config,
        sample_stride=1,
        max_points_per_frame=0,
        random_seed=0,
    )

    flags = point_cloud["source_flags"]
    assert flags.dtype == np.uint16
    assert int(point_cloud["per_frame_counts"][0]) == point_cloud["points"].shape[0]
    assert np.any((flags & SOURCE_GLOBAL) != 0)
    assert np.any((flags & SOURCE_CONTEXT_GRID) != 0)
    assert np.any((flags & SOURCE_LOCAL_PERCENTILE) != 0)
    assert np.any((flags & SOURCE_TOPHAT) != 0)
    assert np.all(point_cloud["sampling_confidence"] >= 0.0)
    assert np.all(point_cloud["sampling_confidence"] <= 1.0)


def test_projection_schema_empty_and_grid_modes() -> None:
    pixel_to_image = np.asarray(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    projected = project_frame_points_to_world(
        xs=np.asarray([2.0, 4.0], dtype=np.float32),
        ys=np.asarray([3.0, 5.0], dtype=np.float32),
        transform=_translation(100.0, 200.0, 300.0),
        pixel_to_image=pixel_to_image,
    )
    np.testing.assert_allclose(
        projected,
        np.asarray(
            [[114.0, 229.0, 330.0], [118.0, 235.0, 330.0]],
            dtype=np.float32,
        ),
    )

    images = np.zeros((3, 16, 16), dtype=np.float32)
    images[0, 3, 2] = 1.0
    images[0, 5, 4] = 1.0
    images[2, 7, 6] = 1.0
    tracking = np.stack(
        [
            _translation(100.0, 200.0, 300.0),
            np.eye(4, dtype=np.float32),
            _translation(-10.0, -20.0, -30.0),
        ]
    )
    legacy = build_frame_point_cloud(
        local_images=images,
        pred_tracking=tracking,
        pixel_to_image=pixel_to_image,
        frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
        texture_config=AlphaTextureConfig(texture_style="original"),
        min_alpha=1,
        min_intensity=1,
        original_foreground_mode="nonzero",
        point_mode="foreground",
        sampling_config=PointSamplingConfig(sampling_mode="legacy"),
        sample_stride=1,
        max_points_per_frame=0,
        random_seed=0,
    )
    np.testing.assert_allclose(
        legacy["pixel_xy"],
        np.asarray([[2, 3], [4, 5], [6, 7]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        legacy["points"],
        np.asarray(
            [[114, 229, 330], [118, 235, 330], [12, 21, 0]],
            dtype=np.float32,
        ),
    )
    assert legacy["per_frame_counts"].tolist() == [2, 0, 1]
    assert legacy["frame_index"].tolist() == [10, 10, 12]
    assert legacy["frame_order"].tolist() == [0, 0, 2]
    assert np.all(legacy["source_flags"] == SOURCE_GLOBAL)
    _assert_point_cloud_array_lengths(legacy, num_frames=3)

    empty = build_frame_point_cloud(
        local_images=np.zeros((2, 8, 8), dtype=np.float32),
        pred_tracking=_identity_tracking(2),
        pixel_to_image=np.eye(4, dtype=np.float32),
        frame_indices=np.asarray([20, 21], dtype=np.int64),
        texture_config=AlphaTextureConfig(texture_style="original"),
        min_alpha=1,
        min_intensity=1,
        original_foreground_mode="nonzero",
        point_mode="foreground",
        sampling_config=PointSamplingConfig(sampling_mode="legacy"),
        sample_stride=1,
        max_points_per_frame=0,
        random_seed=0,
    )
    assert empty["points"].shape == (0, 3)
    assert empty["pixel_xy"].shape == (0, 2)
    assert empty["per_frame_counts"].tolist() == [0, 0]
    _assert_point_cloud_array_lengths(empty, num_frames=2)

    grid = build_frame_point_cloud(
        local_images=np.zeros((1, 4, 5), dtype=np.float32),
        pred_tracking=_identity_tracking(1),
        pixel_to_image=np.eye(4, dtype=np.float32),
        frame_indices=np.asarray([30], dtype=np.int64),
        texture_config=AlphaTextureConfig(texture_style="original"),
        min_alpha=1,
        min_intensity=1,
        original_foreground_mode="nonzero",
        point_mode="grid",
        sampling_config=PointSamplingConfig(sampling_mode="legacy"),
        sample_stride=2,
        max_points_per_frame=0,
        random_seed=0,
    )
    np.testing.assert_allclose(
        grid["pixel_xy"],
        np.asarray(
            [[0, 0], [2, 0], [4, 0], [0, 2], [2, 2], [4, 2]],
            dtype=np.float32,
        ),
    )
    assert np.all(grid["source_flags"] == SOURCE_CONTEXT_GRID)
    assert grid["per_frame_context_only_counts"].tolist() == [6]

    base_kwargs = {
        "local_images": np.zeros((2, 8, 8), dtype=np.float32),
        "pred_tracking": _identity_tracking(2),
        "pixel_to_image": np.eye(4, dtype=np.float32),
        "frame_indices": np.asarray([0, 1], dtype=np.int64),
        "texture_config": AlphaTextureConfig(texture_style="original"),
        "min_alpha": 1,
        "min_intensity": 1,
        "original_foreground_mode": "nonzero",
        "point_mode": "foreground",
        "sampling_config": PointSamplingConfig(sampling_mode="legacy"),
        "sample_stride": 1,
        "max_points_per_frame": 0,
        "random_seed": 0,
    }
    _assert_raises_value_error(
        lambda: build_frame_point_cloud(
            **{**base_kwargs, "pred_tracking": _identity_tracking(1)}
        )
    )
    _assert_raises_value_error(
        lambda: build_frame_point_cloud(
            **{
                **base_kwargs,
                "frame_indices": np.asarray([0], dtype=np.int64),
            }
        )
    )
    _assert_raises_value_error(
        lambda: build_frame_point_cloud(**{**base_kwargs, "sample_stride": 0})
    )
    combined_config = PointSamplingConfig.combined_v2_defaults()
    for point_mode in ("grid", "dense"):
        _assert_raises_value_error(
            lambda mode=point_mode: build_frame_point_cloud(
                **{
                    **base_kwargs,
                    "point_mode": mode,
                    "sampling_config": combined_config,
                }
            )
        )


def test_subsampling_alignment_and_determinism() -> None:
    images = np.zeros((2, 32, 32), dtype=np.float32)
    for y, x in [(5, 5), (9, 13), (15, 21), (25, 27)]:
        images[0, y, x] = 1.0
    for y, x in [(3, 7), (11, 17), (19, 25), (27, 3)]:
        images[1, y, x] = 1.0

    config = PointSamplingConfig(
        sampling_mode="combined_v2",
        include_context_grid=True,
        context_grid_stride=2,
        context_grid_phase="origin",
        enable_local_percentile=True,
        local_window_size=9,
        local_percentile=80.0,
        local_min_contrast=4.0,
        enable_tophat=True,
        tophat_kernel_size=9,
        tophat_percentile=80.0,
        tophat_min_response=2.0,
    )

    def build(max_points: int, seed: int) -> dict[str, np.ndarray]:
        return build_frame_point_cloud(
            local_images=images,
            pred_tracking=_identity_tracking(2),
            pixel_to_image=np.eye(4, dtype=np.float32),
            frame_indices=np.asarray([101, 205], dtype=np.int64),
            texture_config=AlphaTextureConfig(texture_style="original"),
            min_alpha=1,
            min_intensity=1,
            original_foreground_mode="nonzero",
            point_mode="foreground",
            sampling_config=config,
            sample_stride=1,
            max_points_per_frame=max_points,
            random_seed=seed,
        )

    full = build(0, 123)
    assert full["per_frame_counts"].tolist() == [260, 260]
    assert full["per_frame_context_grid_counts"].tolist() == [256, 256]

    limited = build(37, 123)
    assert limited["per_frame_counts"].tolist() == [37, 37]
    _assert_point_cloud_array_lengths(limited, num_frames=2)

    for frame_order, frame_index in enumerate((101, 205)):
        frame_mask = limited["frame_order"] == frame_order
        coordinates = limited["pixel_xy"][frame_mask]
        assert coordinates.shape == (37, 2)
        assert np.unique(coordinates, axis=0).shape[0] == 37
        assert np.all(limited["frame_index"][frame_mask] == frame_index)

        stats = summarize_source_flags(limited["source_flags"][frame_mask])
        mappings = {
            "total_selected_points": "per_frame_counts",
            "global_evidence_points": "per_frame_global_counts",
            "local_percentile_points": "per_frame_local_percentile_counts",
            "tophat_points": "per_frame_tophat_counts",
            "context_grid_points": "per_frame_context_grid_counts",
            "context_only_points": "per_frame_context_only_counts",
            "evidence_points": "per_frame_evidence_counts",
            "overlap_points": "per_frame_overlap_counts",
        }
        for stat_key, array_key in mappings.items():
            assert limited[array_key][frame_order] == stats[stat_key], stat_key

    coordinates = limited["pixel_xy"]
    expected_points = np.concatenate(
        [coordinates, np.zeros((coordinates.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    np.testing.assert_allclose(limited["points"], expected_points)

    for point_index in range(limited["points"].shape[0]):
        frame_order = int(limited["frame_order"][point_index])
        x, y = limited["pixel_xy"][point_index].astype(np.int64)
        assert limited["intensity"][point_index] == image_to_uint8_gray(
            images[frame_order]
        )[y, x]

    assert np.array_equal(
        limited["source_type"], source_type_from_flags(limited["source_flags"])
    )
    for key in ("points", "confidence", "sampling_confidence"):
        assert np.all(np.isfinite(limited[key])), key

    repeat = build(37, 123)
    for key in limited:
        assert np.array_equal(limited[key], repeat[key], equal_nan=True), key

    different_seed = build(37, 124)
    assert not np.array_equal(limited["pixel_xy"], different_seed["pixel_xy"])

    large_limit = build(1000, 999)
    for key in full:
        assert np.array_equal(full[key], large_limit[key], equal_nan=True), key


def main() -> None:
    tests = [
        test_context_grid_counts,
        test_context_grid_boundaries_and_validation,
        test_local_percentile_evidence,
        test_local_percentile_debug_boundaries_and_validation,
        test_tophat_evidence,
        test_tophat_debug_shapes_and_validation,
        test_evidence_cleanup_and_source_isolation,
        test_source_flag_merge_and_confidence,
        test_source_type_merge_and_statistics,
        test_sampling_confidence_alignment_and_validation,
        test_build_frame_point_cloud_legacy_compatibility,
        test_build_frame_point_cloud_combined_v2,
        test_projection_schema_empty_and_grid_modes,
        test_subsampling_alignment_and_determinism,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("Synthetic pseudo3D sampling checks passed.")


if __name__ == "__main__":
    main()
