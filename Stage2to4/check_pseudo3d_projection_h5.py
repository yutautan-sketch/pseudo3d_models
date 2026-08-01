from __future__ import annotations

import sys
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from src.utils.alpha_texture_processing import AlphaTextureConfig
from src.utils.pseudo3d_sampling import (
    SOURCE_CONTEXT_GRID,
    SOURCE_GLOBAL,
    SOURCE_LOCAL_PERCENTILE,
    SOURCE_TOPHAT,
    PointSamplingConfig,
    summarize_per_frame_counts,
    summarize_source_flags,
)
from pseudo3d.export.export_pseudo3d_point_cloud import (
    build_frame_point_cloud,
    load_point_cloud_inputs,
    save_point_cloud_h5,
)


NUM_FRAMES = 3
IMAGE_SHAPE = (16, 16)

POINT_LEVEL_KEYS = (
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

FRAME_LEVEL_KEYS = (
    "per_frame_counts",
    "per_frame_global_counts",
    "per_frame_local_percentile_counts",
    "per_frame_tophat_counts",
    "per_frame_context_grid_counts",
    "per_frame_context_only_counts",
    "per_frame_evidence_counts",
    "per_frame_overlap_counts",
)


def _translation(tx: float, ty: float, tz: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def _synthetic_arrays() -> dict[str, np.ndarray]:
    local_images = np.zeros((NUM_FRAMES, *IMAGE_SHAPE), dtype=np.float64)
    local_images[0, 3, 2] = 1.0
    local_images[0, 5, 4] = 0.5
    # Frame 1 intentionally remains empty.
    local_images[2, 7, 6] = 1.0

    pixel_to_image = np.asarray(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    prior = np.stack(
        [
            _translation(100.0, 200.0, 300.0),
            _translation(10.0, 20.0, 30.0),
            _translation(-10.0, -20.0, -30.0),
        ],
        axis=0,
    )
    correction_offset = _translation(5.0, -7.0, 11.0)
    corrected = np.stack(
        [correction_offset @ matrix for matrix in prior],
        axis=0,
    )

    return {
        "local_encoder_images": local_images,
        # Use int32 on disk to verify the loader's explicit int64 conversion.
        "frame_indices": np.asarray([10, 25, 41], dtype=np.int32),
        "pixel_to_image": pixel_to_image,
        "pred_tracking": prior,
        "pred_tracking_corr": corrected,
    }


def _write_synthetic_input_h5(
    path: Path,
    *,
    missing_key: str | None = None,
    overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    arrays = _synthetic_arrays()
    if overrides is not None:
        arrays.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["synthetic_marker"] = "stage4_projection_check"
        for key, value in arrays.items():
            if key != missing_key:
                handle.create_dataset(key, data=value)

        correction = handle.create_group("tracking_correction")
        correction.attrs["synthetic_correction"] = True
        correction.attrs["translation_offset_xyz"] = "5,-7,11"

    return arrays


def _expect_exception(
    exception_type: type[BaseException],
    func: Callable[[], Any],
    *,
    message_contains: str | None = None,
) -> None:
    try:
        func()
    except exception_type as exc:
        if message_contains is not None:
            assert message_contains in str(exc), (str(exc), message_contains)
        return
    except Exception as exc:  # pragma: no cover - improves failure diagnostics
        raise AssertionError(
            f"Expected {exception_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {exception_type.__name__}")


def test_synthetic_h5_creation(path: Path, expected: dict[str, np.ndarray]) -> None:
    assert path.is_file()
    with h5py.File(path, "r") as handle:
        assert handle.attrs["synthetic_marker"] == "stage4_projection_check"
        assert set(expected).issubset(handle.keys())
        assert "tracking_correction" in handle
        for key, value in expected.items():
            dataset = handle[key]
            assert dataset.shape == value.shape, key
            assert dataset.dtype == value.dtype, key
            assert np.array_equal(dataset[:], value), key


def _assert_common_loaded_fields(
    loaded: dict[str, Any],
    expected: dict[str, np.ndarray],
) -> None:
    assert loaded["local_encoder_images"].shape == (
        NUM_FRAMES,
        *IMAGE_SHAPE,
    )
    assert loaded["frame_indices"].shape == (NUM_FRAMES,)
    assert loaded["pixel_to_image"].shape == (4, 4)
    assert loaded["pred_tracking"].shape == (NUM_FRAMES, 4, 4)

    assert loaded["local_encoder_images"].dtype == np.float32
    assert loaded["frame_indices"].dtype == np.int64
    assert loaded["pixel_to_image"].dtype == np.float32
    assert loaded["pred_tracking"].dtype == np.float32

    np.testing.assert_allclose(
        loaded["local_encoder_images"],
        expected["local_encoder_images"].astype(np.float32),
    )
    assert np.array_equal(
        loaded["frame_indices"],
        expected["frame_indices"].astype(np.int64),
    )
    np.testing.assert_allclose(
        loaded["pixel_to_image"],
        expected["pixel_to_image"].astype(np.float32),
    )

    assert loaded["attrs"]["synthetic_marker"] == "stage4_projection_check"
    assert bool(
        loaded["tracking_correction_attrs"]["synthetic_correction"]
    )
    assert (
        loaded["tracking_correction_attrs"]["translation_offset_xyz"]
        == "5,-7,11"
    )


def test_prior_geometry_loading(
    path: Path,
    expected: dict[str, np.ndarray],
) -> None:
    loaded = load_point_cloud_inputs(path, geometry_key="prior")
    _assert_common_loaded_fields(loaded, expected)
    assert loaded["geometry_key"] == "prior"
    assert loaded["tracking_key"] == "pred_tracking"
    np.testing.assert_allclose(
        loaded["pred_tracking"],
        expected["pred_tracking"].astype(np.float32),
    )


def test_corrected_geometry_loading(
    path: Path,
    expected: dict[str, np.ndarray],
) -> None:
    loaded = load_point_cloud_inputs(path, geometry_key="corr")
    _assert_common_loaded_fields(loaded, expected)
    assert loaded["geometry_key"] == "corr"
    assert loaded["tracking_key"] == "pred_tracking_corr"
    np.testing.assert_allclose(
        loaded["pred_tracking"],
        expected["pred_tracking_corr"].astype(np.float32),
    )
    assert not np.array_equal(
        loaded["pred_tracking"],
        expected["pred_tracking"].astype(np.float32),
    )


def _build_legacy_point_cloud(
    path: Path,
    *,
    geometry_key: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    loaded = load_point_cloud_inputs(path, geometry_key=geometry_key)
    point_cloud = build_frame_point_cloud(
        local_images=loaded["local_encoder_images"],
        pred_tracking=loaded["pred_tracking"],
        pixel_to_image=loaded["pixel_to_image"],
        frame_indices=loaded["frame_indices"],
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
    return loaded, point_cloud


def test_prior_corrected_projection(path: Path) -> None:
    _, prior = _build_legacy_point_cloud(path, geometry_key="prior")
    _, corrected = _build_legacy_point_cloud(path, geometry_key="corr")

    assert prior["points"].shape == (3, 3)
    assert corrected["points"].shape == (3, 3)
    assert np.all(np.isfinite(prior["points"]))
    assert np.all(np.isfinite(corrected["points"]))

    # Tracking geometry must be the only cause of a prior/corr output change.
    for key in prior:
        if key == "points":
            continue
        assert np.array_equal(prior[key], corrected[key], equal_nan=True), key

    expected_offset = np.asarray([5.0, -7.0, 11.0], dtype=np.float32)
    actual_offsets = corrected["points"] - prior["points"]
    np.testing.assert_allclose(
        actual_offsets,
        np.broadcast_to(expected_offset, actual_offsets.shape),
    )


def _independently_reproject(
    loaded: dict[str, Any],
    point_cloud: dict[str, np.ndarray],
) -> np.ndarray:
    points = []
    for pixel_xy, frame_order in zip(
        point_cloud["pixel_xy"],
        point_cloud["frame_order"],
        strict=True,
    ):
        pixel_h = np.asarray(
            [pixel_xy[0], pixel_xy[1], 0.0, 1.0],
            dtype=np.float32,
        )
        image_h = loaded["pixel_to_image"] @ pixel_h
        world_h = loaded["pred_tracking"][int(frame_order)] @ image_h
        points.append(world_h[:3])
    return np.asarray(points, dtype=np.float32)


def test_independent_point_reprojection(path: Path) -> None:
    for geometry_key in ("prior", "corr"):
        loaded, point_cloud = _build_legacy_point_cloud(
            path,
            geometry_key=geometry_key,
        )
        independent = _independently_reproject(loaded, point_cloud)
        assert independent.shape == point_cloud["points"].shape
        np.testing.assert_allclose(independent, point_cloud["points"])


def test_empty_frame_and_frame_indices(path: Path) -> None:
    for geometry_key in ("prior", "corr"):
        loaded, point_cloud = _build_legacy_point_cloud(
            path,
            geometry_key=geometry_key,
        )
        assert loaded["frame_indices"].tolist() == [10, 25, 41]
        assert point_cloud["per_frame_counts"].tolist() == [2, 0, 1]
        assert point_cloud["frame_order"].tolist() == [0, 0, 2]
        assert point_cloud["frame_index"].tolist() == [10, 10, 41]
        np.testing.assert_allclose(
            point_cloud["pixel_xy"],
            np.asarray([[2, 3], [4, 5], [6, 7]], dtype=np.float32),
        )

        for frame_order, frame_index in zip(
            point_cloud["frame_order"],
            point_cloud["frame_index"],
            strict=True,
        ):
            assert frame_index == loaded["frame_indices"][frame_order]


def _combined_sampling_config() -> PointSamplingConfig:
    return PointSamplingConfig(
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
        local_source_confidence=0.7,
        tophat_source_confidence=0.7,
        context_source_confidence=0.0,
    )


def _build_combined_point_cloud(
    path: Path,
    *,
    geometry_key: str = "corr",
) -> tuple[dict[str, Any], PointSamplingConfig, dict[str, np.ndarray]]:
    loaded = load_point_cloud_inputs(path, geometry_key=geometry_key)
    config = _combined_sampling_config()
    point_cloud = build_frame_point_cloud(
        local_images=loaded["local_encoder_images"],
        pred_tracking=loaded["pred_tracking"],
        pixel_to_image=loaded["pixel_to_image"],
        frame_indices=loaded["frame_indices"],
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
    return loaded, config, point_cloud


def _point_cloud_metadata(
    *,
    source_h5: Path,
    loaded: dict[str, Any],
    config: PointSamplingConfig,
    point_cloud: dict[str, np.ndarray],
) -> dict[str, Any]:
    source_stats = summarize_source_flags(point_cloud["source_flags"])
    frame_stats = summarize_per_frame_counts(point_cloud["per_frame_counts"])
    return {
        "source_h5": str(source_h5),
        "geometry_key": loaded["geometry_key"],
        "tracking_key": loaded["tracking_key"],
        "point_mode": "foreground",
        "sampling_mode": config.sampling_mode,
        "include_context_grid": config.include_context_grid,
        "context_grid_stride": config.context_grid_stride,
        "context_grid_phase": config.context_grid_phase,
        "enable_local_percentile": config.enable_local_percentile,
        "local_window_size": config.local_window_size,
        "local_percentile": config.local_percentile,
        "local_min_contrast": config.local_min_contrast,
        "enable_tophat": config.enable_tophat,
        "tophat_kernel_size": config.tophat_kernel_size,
        "tophat_percentile": config.tophat_percentile,
        "tophat_min_response": config.tophat_min_response,
        "tophat_morph_shape": config.tophat_morph_shape,
        "enable_evidence_cleanup": config.enable_evidence_cleanup,
        "evidence_open_ksize": config.evidence_open_ksize,
        "evidence_close_ksize": config.evidence_close_ksize,
        "evidence_morph_shape": config.evidence_morph_shape,
        "evidence_min_component_area": config.evidence_min_component_area,
        "local_source_confidence": config.local_source_confidence,
        "tophat_source_confidence": config.tophat_source_confidence,
        "context_source_confidence": config.context_source_confidence,
        "sample_stride": 1,
        "max_points_per_frame": 0,
        "random_seed": 0,
        "num_frames": int(point_cloud["per_frame_counts"].shape[0]),
        "num_points": int(point_cloud["points"].shape[0]),
        **frame_stats,
        **source_stats,
        "source_type_frame": 0,
        "source_type_global": 1,
        "source_type_local_percentile": 2,
        "source_type_tophat": 3,
        "source_type_context_grid": 4,
        "source_flags_global": int(SOURCE_GLOBAL),
        "source_flags_local_percentile": int(SOURCE_LOCAL_PERCENTILE),
        "source_flags_tophat": int(SOURCE_TOPHAT),
        "source_flags_context_grid": int(SOURCE_CONTEXT_GRID),
        "optional_none_marker": None,
    }


def _assert_h5_attr_equal(actual: Any, expected: Any, key: str) -> None:
    if expected is None:
        assert actual == "None", key
    elif isinstance(expected, (float, np.floating)):
        assert np.isclose(actual, expected), key
    else:
        assert actual == expected, key


def _assert_round_trip_datasets(
    group: h5py.Group,
    point_cloud: dict[str, np.ndarray],
) -> None:
    assert set(group.keys()) == set(point_cloud)
    for key, expected in point_cloud.items():
        dataset = group[key]
        assert dataset.shape == expected.shape, key
        assert dataset.dtype == expected.dtype, key
        assert dataset.compression == "gzip", key
        assert np.array_equal(dataset[:], expected), key


def test_combined_v2_h5_round_trip(
    source_h5: Path,
    output_h5: Path,
) -> None:
    loaded, config, point_cloud = _build_combined_point_cloud(source_h5)
    metadata = _point_cloud_metadata(
        source_h5=source_h5,
        loaded=loaded,
        config=config,
        point_cloud=point_cloud,
    )
    save_point_cloud_h5(output_h5, point_cloud=point_cloud, meta=metadata)

    # 16x16 origin grid at stride 4 contributes 16 points to every frame.
    # The three evidence pixels are all off-grid.
    assert point_cloud["per_frame_counts"].tolist() == [18, 16, 17]
    assert point_cloud["per_frame_context_grid_counts"].tolist() == [16, 16, 16]
    assert point_cloud["per_frame_context_only_counts"].tolist() == [16, 16, 16]

    with h5py.File(output_h5, "r") as handle:
        assert set(handle.keys()) == {"point_cloud"}
        group = handle["point_cloud"]
        _assert_round_trip_datasets(group, point_cloud)

        for key, expected in metadata.items():
            assert key in group.attrs, key
            _assert_h5_attr_equal(group.attrs[key], expected, key)

        num_points = int(group["points"].shape[0])
        num_frames = int(group["per_frame_counts"].shape[0])
        assert num_points == int(group.attrs["num_points"])
        assert num_frames == int(group.attrs["num_frames"])
        assert int(group["per_frame_counts"][:].sum()) == num_points
        assert np.array_equal(
            np.bincount(group["frame_order"][:], minlength=num_frames),
            group["per_frame_counts"][:],
        )

        for key in POINT_LEVEL_KEYS:
            assert group[key].shape[0] == num_points, key
        for key in FRAME_LEVEL_KEYS:
            assert group[key].shape == (num_frames,), key
            assert group[key].dtype == np.int64, key

        source_stats = summarize_source_flags(group["source_flags"][:])
        for key, expected in source_stats.items():
            _assert_h5_attr_equal(group.attrs[key], expected, key)

        frame_stats = summarize_per_frame_counts(group["per_frame_counts"][:])
        for key, expected in frame_stats.items():
            _assert_h5_attr_equal(group.attrs[key], expected, key)

        stat_to_dataset = {
            "total_selected_points": "per_frame_counts",
            "global_evidence_points": "per_frame_global_counts",
            "local_percentile_points": "per_frame_local_percentile_counts",
            "tophat_points": "per_frame_tophat_counts",
            "context_grid_points": "per_frame_context_grid_counts",
            "context_only_points": "per_frame_context_only_counts",
            "evidence_points": "per_frame_evidence_counts",
            "overlap_points": "per_frame_overlap_counts",
        }
        frame_order = group["frame_order"][:]
        source_flags = group["source_flags"][:]
        for index in range(num_frames):
            recomputed = summarize_source_flags(source_flags[frame_order == index])
            for stat_key, dataset_key in stat_to_dataset.items():
                assert group[dataset_key][index] == recomputed[stat_key], (
                    index,
                    stat_key,
                )

        # The originally empty middle frame must contain context-only points.
        middle = frame_order == 1
        assert int(middle.sum()) == 16
        assert np.all(group["source_flags"][:][middle] == SOURCE_CONTEXT_GRID)
        assert np.all(group["sampling_confidence"][:][middle] == 0.0)
        assert np.all(group["confidence"][:][middle] == 1.0)


def test_empty_point_cloud_h5_round_trip(output_h5: Path) -> None:
    point_cloud = build_frame_point_cloud(
        local_images=np.zeros((2, 8, 8), dtype=np.float32),
        pred_tracking=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        pixel_to_image=np.eye(4, dtype=np.float32),
        frame_indices=np.asarray([3, 9], dtype=np.int64),
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
    metadata = {
        "sampling_mode": "legacy",
        "point_mode": "foreground",
        "num_frames": 2,
        "num_points": 0,
    }
    save_point_cloud_h5(output_h5, point_cloud=point_cloud, meta=metadata)

    with h5py.File(output_h5, "r") as handle:
        group = handle["point_cloud"]
        _assert_round_trip_datasets(group, point_cloud)
        assert group["points"].shape == (0, 3)
        assert group["pixel_xy"].shape == (0, 2)
        for key in POINT_LEVEL_KEYS:
            assert group[key].shape[0] == 0, key
        for key in FRAME_LEVEL_KEYS:
            assert group[key].shape == (2,), key
            assert np.all(group[key][:] == 0), key
        assert int(group.attrs["num_frames"]) == 2
        assert int(group.attrs["num_points"]) == 0


def _run_single_export_cli(
    *,
    source_h5: Path,
    output_h5: Path,
    geometry_key: str,
    sampling_mode: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "pseudo3d/export/export_pseudo3d_point_cloud.py"),
        "--input_h5",
        str(source_h5),
        "--geometry_key",
        geometry_key,
        "--preset",
        "original",
        "--output_h5",
        str(output_h5),
        "--point_mode",
        "foreground",
        "--sample_stride",
        "1",
        "--sampling_mode",
        sampling_mode,
        "--texture_style",
        "original",
    ]
    if sampling_mode == "combined_v2":
        command.extend(
            [
                "--include_context_grid",
                "--context_grid_stride",
                "4",
                "--context_grid_phase",
                "origin",
                "--enable_local_percentile",
                "--local_window_size",
                "9",
                "--local_percentile",
                "80",
                "--local_min_contrast",
                "4",
                "--enable_tophat",
                "--tophat_kernel_size",
                "9",
                "--tophat_percentile",
                "80",
                "--tophat_min_response",
                "2",
                "--tophat_morph_shape",
                "ellipse",
                "--no-enable_evidence_cleanup",
                "--local_source_confidence",
                "0.7",
                "--tophat_source_confidence",
                "0.7",
                "--context_source_confidence",
                "0.0",
            ]
        )

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Single export CLI failed\n"
            f"command: {' '.join(command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    assert output_h5.is_file(), output_h5
    return result


def _direct_point_cloud_for_case(
    source_h5: Path,
    *,
    geometry_key: str,
    sampling_mode: str,
) -> dict[str, np.ndarray]:
    if sampling_mode == "legacy":
        _, point_cloud = _build_legacy_point_cloud(
            source_h5,
            geometry_key=geometry_key,
        )
        return point_cloud
    if sampling_mode == "combined_v2":
        _, _, point_cloud = _build_combined_point_cloud(
            source_h5,
            geometry_key=geometry_key,
        )
        return point_cloud
    raise ValueError(f"Unknown sampling mode: {sampling_mode}")


def _read_point_cloud_file(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == {"point_cloud"}
        group = handle["point_cloud"]
        arrays = {key: group[key][:] for key in group.keys()}
        attrs = dict(group.attrs)
    return arrays, attrs


def test_single_cli_end_to_end(source_h5: Path, temp_dir: Path) -> None:
    cases = [
        ("legacy", "prior"),
        ("legacy", "corr"),
        ("combined_v2", "prior"),
        ("combined_v2", "corr"),
    ]
    outputs: dict[tuple[str, str], Path] = {}
    arrays_by_case: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    attrs_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    stdout_by_case: dict[tuple[str, str], str] = {}

    for sampling_mode, geometry_key in cases:
        case = (sampling_mode, geometry_key)
        output_h5 = temp_dir / f"cli_{sampling_mode}_{geometry_key}.h5"
        result = _run_single_export_cli(
            source_h5=source_h5,
            output_h5=output_h5,
            geometry_key=geometry_key,
            sampling_mode=sampling_mode,
        )
        outputs[case] = output_h5
        stdout_by_case[case] = result.stdout

        arrays, attrs = _read_point_cloud_file(output_h5)
        arrays_by_case[case] = arrays
        attrs_by_case[case] = attrs

        expected = _direct_point_cloud_for_case(
            source_h5,
            geometry_key=geometry_key,
            sampling_mode=sampling_mode,
        )
        assert set(arrays) == set(expected), case
        for key, expected_value in expected.items():
            assert arrays[key].shape == expected_value.shape, (case, key)
            assert arrays[key].dtype == expected_value.dtype, (case, key)
            assert np.array_equal(arrays[key], expected_value), (case, key)

        expected_tracking_key = (
            "pred_tracking" if geometry_key == "prior" else "pred_tracking_corr"
        )
        assert attrs["geometry_key"] == geometry_key
        assert attrs["tracking_key"] == expected_tracking_key
        assert attrs["point_mode"] == "foreground"
        assert attrs["sampling_mode"] == sampling_mode
        assert int(attrs["num_frames"]) == NUM_FRAMES
        assert int(attrs["num_points"]) == arrays["points"].shape[0]
        assert int(attrs["source_flags_global"]) == int(SOURCE_GLOBAL)
        assert int(attrs["source_flags_local_percentile"]) == int(
            SOURCE_LOCAL_PERCENTILE
        )
        assert int(attrs["source_flags_tophat"]) == int(SOURCE_TOPHAT)
        assert int(attrs["source_flags_context_grid"]) == int(SOURCE_CONTEXT_GRID)

        if sampling_mode == "legacy":
            assert not bool(attrs["include_context_grid"])
            assert not bool(attrs["enable_local_percentile"])
            assert not bool(attrs["enable_tophat"])
            assert not bool(attrs["enable_evidence_cleanup"])
        else:
            assert bool(attrs["include_context_grid"])
            assert int(attrs["context_grid_stride"]) == 4
            assert attrs["context_grid_phase"] == "origin"
            assert bool(attrs["enable_local_percentile"])
            assert int(attrs["local_window_size"]) == 9
            assert float(attrs["local_percentile"]) == 80.0
            assert float(attrs["local_min_contrast"]) == 4.0
            assert bool(attrs["enable_tophat"])
            assert int(attrs["tophat_kernel_size"]) == 9
            assert float(attrs["tophat_percentile"]) == 80.0
            assert float(attrs["tophat_min_response"]) == 2.0
            assert attrs["tophat_morph_shape"] == "ellipse"
            assert not bool(attrs["enable_evidence_cleanup"])
        assert int(attrs["evidence_open_ksize"]) == 3
        assert int(attrs["evidence_close_ksize"]) == 5
        assert attrs["evidence_morph_shape"] == "ellipse"
        assert int(attrs["evidence_min_component_area"]) == 100

        source_stats = summarize_source_flags(arrays["source_flags"])
        frame_stats = summarize_per_frame_counts(arrays["per_frame_counts"])
        for key, expected_value in {**source_stats, **frame_stats}.items():
            _assert_h5_attr_equal(attrs[key], expected_value, key)

        stdout = result.stdout
        required_stdout = (
            f"geometry_key: {geometry_key}",
            f"sampling_mode: {sampling_mode}",
            f"num_frames: {NUM_FRAMES}",
            f"num_points: {arrays['points'].shape[0]}",
            "global_evidence_points:",
            "local_percentile_points:",
            "tophat_points:",
            "context_grid_points:",
            "legacy_point_count_multiplier:",
            "Saved point cloud h5:",
        )
        for text in required_stdout:
            assert text in stdout, (case, text, stdout)

    assert len(set(outputs.values())) == len(cases)
    assert all(path.is_file() for path in outputs.values())

    expected_offset = np.asarray([5.0, -7.0, 11.0], dtype=np.float32)
    for sampling_mode in ("legacy", "combined_v2"):
        prior = arrays_by_case[(sampling_mode, "prior")]
        corrected = arrays_by_case[(sampling_mode, "corr")]
        for key in prior:
            if key == "points":
                continue
            assert np.array_equal(prior[key], corrected[key], equal_nan=True), (
                sampling_mode,
                key,
            )
        offsets = corrected["points"] - prior["points"]
        np.testing.assert_allclose(
            offsets,
            np.broadcast_to(expected_offset, offsets.shape),
        )

        assert attrs_by_case[(sampling_mode, "prior")]["geometry_key"] == "prior"
        assert attrs_by_case[(sampling_mode, "corr")]["geometry_key"] == "corr"
        assert stdout_by_case[(sampling_mode, "prior")]
        assert stdout_by_case[(sampling_mode, "corr")]


def _run_expected_cli_failure(
    *,
    source_h5: Path,
    output_h5: Path,
    geometry_key: str,
) -> str:
    command = [
        sys.executable,
        str(REPO_ROOT / "pseudo3d/export/export_pseudo3d_point_cloud.py"),
        "--input_h5",
        str(source_h5),
        "--geometry_key",
        geometry_key,
        "--preset",
        "original",
        "--output_h5",
        str(output_h5),
        "--point_mode",
        "foreground",
        "--sample_stride",
        "1",
        "--sampling_mode",
        "legacy",
        "--texture_style",
        "original",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(
            "Malformed input unexpectedly succeeded\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    assert not output_h5.exists(), output_h5
    diagnostics = f"{result.stdout}\n{result.stderr}"
    assert diagnostics.strip()
    return diagnostics


def test_invalid_input_shapes_rejected(temp_dir: Path) -> None:
    arrays = _synthetic_arrays()
    cases: list[tuple[str, str, dict[str, np.ndarray], str | None]] = [
        (
            "prior_tracking_frame_count",
            "prior",
            {"pred_tracking": arrays["pred_tracking"][:2]},
            "Number of tracking matrices must match local images",
        ),
        (
            "corrected_tracking_frame_count",
            "corr",
            {"pred_tracking_corr": arrays["pred_tracking_corr"][:2]},
            "Number of tracking matrices must match local images",
        ),
        (
            "frame_index_count",
            "prior",
            {"frame_indices": arrays["frame_indices"][:2]},
            "Number of frame_indices must match local images",
        ),
        (
            "pixel_to_image_shape",
            "prior",
            {"pixel_to_image": np.eye(3, dtype=np.float64)},
            None,
        ),
        (
            "tracking_matrix_shape",
            "prior",
            {
                "pred_tracking": np.repeat(
                    np.eye(3, dtype=np.float64)[None],
                    NUM_FRAMES,
                    axis=0,
                )
            },
            None,
        ),
    ]

    matrix_diagnostics: dict[str, str] = {}
    output_paths = []
    for name, geometry_key, overrides, expected_message in cases:
        input_h5 = temp_dir / f"invalid_{name}.h5"
        output_h5 = temp_dir / f"invalid_{name}_output.h5"
        _write_synthetic_input_h5(input_h5, overrides=overrides)
        diagnostics = _run_expected_cli_failure(
            source_h5=input_h5,
            output_h5=output_h5,
            geometry_key=geometry_key,
        )
        output_paths.append(output_h5)

        if expected_message is not None:
            assert expected_message in diagnostics, (name, diagnostics)
        else:
            matrix_diagnostics[name] = diagnostics
            assert any(
                token in diagnostics.lower()
                for token in ("matmul", "mismatch", "different from")
            ), (name, diagnostics)

    assert not any(path.exists() for path in output_paths)
    assert "pixel_to_image_shape" in matrix_diagnostics
    assert "tracking_matrix_shape" in matrix_diagnostics


def test_loader_validation(temp_dir: Path) -> None:
    _expect_exception(
        FileNotFoundError,
        lambda: load_point_cloud_inputs(
            temp_dir / "does_not_exist.h5",
            geometry_key="prior",
        ),
        message_contains="Input h5 not found",
    )
    print("[OK] validation: missing input H5")

    valid_path = temp_dir / "valid_for_unknown_geometry.h5"
    _write_synthetic_input_h5(valid_path)
    _expect_exception(
        ValueError,
        lambda: load_point_cloud_inputs(valid_path, geometry_key="unknown"),
        message_contains="Unknown geometry_key",
    )
    print("[OK] validation: unknown geometry key")

    cases = [
        ("local_encoder_images", "prior"),
        ("frame_indices", "prior"),
        ("pixel_to_image", "prior"),
        ("pred_tracking", "prior"),
        ("pred_tracking_corr", "corr"),
    ]
    for missing_key, geometry_key in cases:
        path = temp_dir / f"missing_{missing_key}.h5"
        _write_synthetic_input_h5(path, missing_key=missing_key)
        _expect_exception(
            KeyError,
            lambda candidate=path, geometry=geometry_key: load_point_cloud_inputs(
                candidate,
                geometry_key=geometry,
            ),
            message_contains=missing_key,
        )
        print(f"[OK] validation: missing {missing_key}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_projection_h5_") as temp:
        temp_dir = Path(temp)
        valid_path = temp_dir / "synthetic_pseudo3d.h5"
        expected = _write_synthetic_input_h5(valid_path)

        test_synthetic_h5_creation(valid_path, expected)
        print("[OK] synthetic pseudo3D H5 creation")

        test_prior_geometry_loading(valid_path, expected)
        print("[OK] prior geometry loading")

        test_corrected_geometry_loading(valid_path, expected)
        print("[OK] corrected geometry loading")

        print("[OK] loaded values, shapes, and dtypes")

        test_prior_corrected_projection(valid_path)
        print("[OK] prior point-cloud projection")
        print("[OK] corrected point-cloud projection")
        print("[OK] prior/corr sampling arrays are identical")
        print("[OK] prior/corr point offset")

        test_independent_point_reprojection(valid_path)
        print("[OK] independent point reprojection")

        test_empty_frame_and_frame_indices(valid_path)
        print("[OK] empty frame and non-contiguous frame indices")

        combined_output = temp_dir / "combined_v2_point_cloud.h5"
        test_combined_v2_h5_round_trip(valid_path, combined_output)
        print("[OK] combined_v2 point-cloud H5 round trip")
        print("[OK] point-level H5 schema and dtypes")
        print("[OK] frame-level H5 schema and dtypes")
        print("[OK] gzip dataset compression")
        print("[OK] H5 metadata round trip")
        print("[OK] aggregate source statistics")
        print("[OK] per-frame source statistics")
        print("[OK] points-per-frame statistics")

        empty_output = temp_dir / "empty_legacy_point_cloud.h5"
        test_empty_point_cloud_h5_round_trip(empty_output)
        print("[OK] empty point-cloud H5 round trip")

        test_single_cli_end_to_end(valid_path, temp_dir)
        print("[OK] CLI legacy prior export")
        print("[OK] CLI legacy corrected export")
        print("[OK] CLI combined_v2 prior export")
        print("[OK] CLI combined_v2 corrected export")
        print("[OK] CLI outputs match direct function results")
        print("[OK] CLI prior/corr sampling arrays are identical")
        print("[OK] CLI prior/corr point offset")
        print("[OK] CLI sampling and geometry metadata")
        print("[OK] CLI console statistics")

        test_invalid_input_shapes_rejected(temp_dir)
        print("[OK] validation: prior tracking frame count mismatch")
        print("[OK] validation: corrected tracking frame count mismatch")
        print("[OK] validation: frame index count mismatch")
        print("[OK] validation: invalid pixel_to_image shape")
        print("[OK] validation: invalid tracking matrix shape")
        print("[OK] failed CLI does not create output H5")
        print(
            "[INFO] invalid matrix shapes are currently rejected by "
            "NumPy matrix multiplication"
        )

        test_loader_validation(temp_dir)

    print("Synthetic pseudo3D H5 input checks passed.")


if __name__ == "__main__":
    main()
