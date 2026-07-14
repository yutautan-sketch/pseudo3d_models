from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.utils.alpha_texture_processing import AlphaTextureConfig
from src.utils.pseudo3d_sampling import (
    SOURCE_CONTEXT_GRID,
    SOURCE_GLOBAL,
    SOURCE_LOCAL_PERCENTILE,
    SOURCE_TOPHAT,
    PointSamplingConfig,
    build_context_grid_mask,
    build_local_percentile_evidence,
    build_sampling_confidence,
    build_sampling_source_flags,
    build_tophat_evidence,
    selected_pixel_arrays,
)
from pseudo3d.export.export_pseudo3d_point_cloud import build_frame_point_cloud


def _identity_tracking(num_frames: int) -> np.ndarray:
    return np.repeat(np.eye(4, dtype=np.float32)[None, ...], num_frames, axis=0)


def test_context_grid_counts() -> None:
    shape = (16, 16)

    origin = build_context_grid_mask(shape, stride=4, phase="origin")
    centered = build_context_grid_mask(shape, stride=4, phase="centered")
    dual = build_context_grid_mask(shape, stride=4, phase="dual")

    assert int(origin.sum()) == 16
    assert int(centered.sum()) == 16
    assert int(dual.sum()) == 32
    assert not np.any(origin & centered)


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


def main() -> None:
    tests = [
        test_context_grid_counts,
        test_local_percentile_evidence,
        test_tophat_evidence,
        test_source_flag_merge_and_confidence,
        test_build_frame_point_cloud_legacy_compatibility,
        test_build_frame_point_cloud_combined_v2,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("Synthetic pseudo3D sampling checks passed.")


if __name__ == "__main__":
    main()
