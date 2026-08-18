from __future__ import annotations

import py_compile
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

STAGE2TO4_ROOT = Path(__file__).resolve().parents[2]
GIT_ROOT = STAGE2TO4_ROOT.parent
if str(STAGE2TO4_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2TO4_ROOT))
BASELINE_COMMIT = "65daf6d"
EXPORT_RELATIVE_PATH = Path("pseudo3d/export/export_pseudo3d_point_cloud.py")
ANNOTATE_RELATIVE_PATH = Path(
    "pseudo3d/annotation/annotate_pseudo3d_point_cloud.py"
)
ANNOTATED_PLY_RELATIVE_PATH = Path(
    "pseudo3d/export/export_annotated_point_cloud_ply.py"
)

import h5py
import numpy as np

from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    load_point_cloud_h5,
)
from src.utils.pseudo3d_sampling import (
    SOURCE_CONTEXT_GRID,
    SOURCE_GLOBAL,
    summarize_source_flags,
)


COMMON_POINT_CLOUD_DATASETS = (
    "points",
    "intensity",
    "alpha",
    "frame_index",
    "frame_order",
    "pixel_xy",
    "confidence",
    "source_type",
    "per_frame_counts",
)

COMMON_METADATA = (
    "source_h5",
    "geometry_key",
    "tracking_key",
    "preset",
    "point_mode",
    "texture_style",
    "texture_denoise",
    "texture_denoise_ksize",
    "texture_threshold_mode",
    "texture_percentile",
    "texture_min_component_area",
    "min_alpha",
    "min_intensity",
    "original_foreground_mode",
    "sample_stride",
    "max_points_per_frame",
    "random_seed",
    "num_frames",
    "num_points",
    "points_per_frame_min",
    "points_per_frame_max",
    "points_per_frame_mean",
    "source_type_frame",
)

CURRENT_LEGACY_DATASETS = (
    "source_flags",
    "sampling_confidence",
    "per_frame_global_counts",
    "per_frame_local_percentile_counts",
    "per_frame_tophat_counts",
    "per_frame_context_grid_counts",
    "per_frame_context_only_counts",
    "per_frame_evidence_counts",
    "per_frame_overlap_counts",
)


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Command failed\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {cwd}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def test_baseline_commit_available() -> None:
    _run(
        ["git", "cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}"],
        cwd=GIT_ROOT,
    )


def extract_baseline(temp_dir: Path) -> Path:
    archive_path = temp_dir / f"stage4_baseline_{BASELINE_COMMIT}.tar"
    extract_root = temp_dir / "baseline_checkout"
    extract_root.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            BASELINE_COMMIT,
            "Stage2to4",
        ],
        cwd=GIT_ROOT,
    )
    assert archive_path.is_file()
    assert archive_path.stat().st_size > 0

    # The archive is produced from a trusted local Git tree. Avoid filter="data"
    # here so this check also runs under the project's Python 3.11 environment.
    with tarfile.open(archive_path, mode="r") as archive:
        archive.extractall(extract_root)

    baseline_root = extract_root / "Stage2to4"
    baseline_export = baseline_root / EXPORT_RELATIVE_PATH
    baseline_annotate = baseline_root / ANNOTATE_RELATIVE_PATH
    assert baseline_export.is_file()
    assert baseline_annotate.is_file()
    py_compile.compile(str(baseline_export), doraise=True)
    py_compile.compile(str(baseline_annotate), doraise=True)
    py_compile.compile(
        str(STAGE2TO4_ROOT / EXPORT_RELATIVE_PATH),
        doraise=True,
    )
    py_compile.compile(
        str(STAGE2TO4_ROOT / ANNOTATE_RELATIVE_PATH),
        doraise=True,
    )
    return baseline_root


def test_cli_help(baseline_root: Path) -> None:
    baseline_help = _run(
        [sys.executable, str(baseline_root / EXPORT_RELATIVE_PATH), "--help"],
        cwd=baseline_root,
    ).stdout
    current_help = _run(
        [sys.executable, str(STAGE2TO4_ROOT / EXPORT_RELATIVE_PATH), "--help"],
        cwd=STAGE2TO4_ROOT,
    ).stdout
    baseline_annotation_help = _run(
        [sys.executable, str(baseline_root / ANNOTATE_RELATIVE_PATH), "--help"],
        cwd=baseline_root,
    ).stdout
    current_annotation_help = _run(
        [sys.executable, str(STAGE2TO4_ROOT / ANNOTATE_RELATIVE_PATH), "--help"],
        cwd=STAGE2TO4_ROOT,
    ).stdout

    assert "--sampling_mode" not in baseline_help
    assert "--sampling_mode" in current_help
    assert "legacy" in current_help
    assert "combined_v2" in current_help
    for help_text in (baseline_annotation_help, current_annotation_help):
        assert "--point_cloud_h5" in help_text
        assert "--pseudo3d_h5" in help_text
        assert "--voc_xml_root" in help_text


def _translation(tx: float, ty: float, tz: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def write_synthetic_legacy_input(path: Path) -> None:
    images = np.zeros((3, 16, 16), dtype=np.float32)
    images[0, 3, 2] = 1.0
    images[0, 5, 4] = 0.5
    # Frame 1 intentionally remains empty.
    images[2, 7, 6] = 1.0

    pixel_to_image = np.asarray(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    prior = np.stack(
        [
            _translation(100.0, 200.0, 300.0),
            _translation(10.0, 20.0, 30.0),
            _translation(-10.0, -20.0, -30.0),
        ]
    )
    correction = _translation(5.0, -7.0, 11.0)
    corrected = np.stack([correction @ matrix for matrix in prior])

    with h5py.File(path, "w") as handle:
        handle.attrs["synthetic_marker"] = "stage4_legacy_compatibility"
        handle.create_dataset("local_encoder_images", data=images)
        handle.create_dataset(
            "frame_indices",
            data=np.asarray([10, 25, 41], dtype=np.int64),
        )
        handle.create_dataset("pixel_to_image", data=pixel_to_image)
        handle.create_dataset("pred_tracking", data=prior)
        handle.create_dataset("pred_tracking_corr", data=corrected)


def write_synthetic_texture_input(path: Path) -> None:
    images = np.zeros((3, 64, 64), dtype=np.float32)

    images[0, 4:28, 4:28] = 1.0
    images[0, 36:56, 36:56] = 0.6
    images[0, 31, 12] = 0.8

    images[1, 10:40, 18:48] = 0.9
    images[1, 45:60, 5:25] = 0.5

    images[2, 5:25, 32:58] = 1.0
    images[2, 34:58, 6:30] = 0.7
    images[2, 30, 45] = 0.9

    prior = np.stack(
        [
            _translation(20.0, 40.0, 60.0),
            _translation(-5.0, 15.0, 25.0),
            _translation(100.0, -50.0, 10.0),
        ]
    )
    correction = _translation(3.0, 4.0, -2.0)
    corrected = np.stack([correction @ matrix for matrix in prior])

    with h5py.File(path, "w") as handle:
        handle.attrs["synthetic_marker"] = "stage4_legacy_texture_compatibility"
        handle.create_dataset("local_encoder_images", data=images)
        handle.create_dataset(
            "frame_indices",
            data=np.asarray([101, 205, 409], dtype=np.int64),
        )
        handle.create_dataset("pixel_to_image", data=np.eye(4, dtype=np.float32))
        handle.create_dataset("pred_tracking", data=prior)
        handle.create_dataset("pred_tracking_corr", data=corrected)


def write_synthetic_annotation_input(path: Path) -> None:
    images = np.zeros((3, 32, 32), dtype=np.float32)
    images[0, 4:12, 4:12] = 1.0
    images[0, 25, 2] = 0.8  # Foreground point outside the frame-0 VOC BBox.
    images[1, 10:18, 10:18] = 0.75  # This frame intentionally has no XML.
    images[2, 20:28, 20:28] = 0.5

    prior = np.stack(
        [
            _translation(1.0, 2.0, 3.0),
            _translation(4.0, 5.0, 6.0),
            _translation(7.0, 8.0, 9.0),
        ]
    )
    correction = _translation(0.5, -1.0, 2.0)
    corrected = np.stack([correction @ matrix for matrix in prior])

    with h5py.File(path, "w") as handle:
        handle.attrs["synthetic_marker"] = "stage4_legacy_annotation_compatibility"
        handle.create_dataset("local_encoder_images", data=images)
        handle.create_dataset(
            "frame_indices",
            data=np.asarray([110, 125, 141], dtype=np.int64),
        )
        handle.create_dataset("pixel_to_image", data=np.eye(4, dtype=np.float32))
        handle.create_dataset("pred_tracking", data=prior)
        handle.create_dataset("pred_tracking_corr", data=corrected)


def write_synthetic_voc_xml_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    def write_xml(frame_index: int, bbox: tuple[int, int, int, int]) -> None:
        xmin, ymin, xmax, ymax = bbox
        xml = f"""<annotation>
  <size><width>32</width><height>32</height><depth>1</depth></size>
  <object>
    <name>femur</name>
    <bndbox>
      <xmin>{xmin}</xmin><ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax><ymax>{ymax}</ymax>
    </bndbox>
  </object>
</annotation>
"""
        (root / f"{frame_index}.xml").write_text(xml)

    write_xml(110, (2, 2, 14, 14))
    write_xml(141, (18, 18, 30, 30))
    assert not (root / "125.xml").exists()


def _run_original_foreground_export(
    *,
    code_root: Path,
    input_h5: Path,
    output_h5: Path,
    current: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(code_root / EXPORT_RELATIVE_PATH),
        "--input_h5",
        str(input_h5),
        "--output_h5",
        str(output_h5),
        "--geometry_key",
        "corr",
        "--preset",
        "original",
        "--texture_style",
        "original",
        "--point_mode",
        "foreground",
        "--sample_stride",
        "1",
        "--max_points_per_frame",
        "0",
        "--random_seed",
        "0",
    ]
    if current:
        command.extend(["--sampling_mode", "legacy"])
    result = _run(command, cwd=code_root)
    assert output_h5.is_file()
    assert "num_frames: 3" in result.stdout
    assert "num_points: 3" in result.stdout
    return result


def _read_point_cloud_h5(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == {"point_cloud"}
        group = handle["point_cloud"]
        arrays = {key: group[key][:] for key in group.keys()}
        attrs = dict(group.attrs)
    return arrays, attrs


def _assert_attr_equal(actual: Any, expected: Any, key: str) -> None:
    if isinstance(expected, (float, np.floating)):
        assert np.isclose(actual, expected), (key, actual, expected)
    else:
        assert actual == expected, (key, actual, expected)


def test_common_legacy_output_equality(
    baseline_h5: Path,
    current_h5: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    baseline_arrays, baseline_attrs = _read_point_cloud_h5(baseline_h5)
    current_arrays, current_attrs = _read_point_cloud_h5(current_h5)

    assert set(baseline_arrays) == set(COMMON_POINT_CLOUD_DATASETS)
    for key in COMMON_POINT_CLOUD_DATASETS:
        assert key in current_arrays, key
        assert current_arrays[key].shape == baseline_arrays[key].shape, key
        assert current_arrays[key].dtype == baseline_arrays[key].dtype, key
        assert np.array_equal(current_arrays[key], baseline_arrays[key]), key

    for key in COMMON_METADATA:
        assert key in baseline_attrs, key
        assert key in current_attrs, key
        _assert_attr_equal(current_attrs[key], baseline_attrs[key], key)

    return baseline_arrays, baseline_attrs, current_arrays, current_attrs


def test_current_legacy_extension_fields(
    baseline_arrays: dict[str, np.ndarray],
    current_arrays: dict[str, np.ndarray],
    current_attrs: dict[str, Any],
    *,
    point_mode: str = "foreground",
) -> None:
    for key in CURRENT_LEGACY_DATASETS:
        assert key in current_arrays, key

    num_points = int(baseline_arrays["points"].shape[0])
    expected_flag = (
        SOURCE_GLOBAL if point_mode == "foreground" else SOURCE_CONTEXT_GRID
    )
    assert np.all(current_arrays["source_flags"] == expected_flag)
    assert current_arrays["source_flags"].dtype == np.uint16
    assert current_arrays["sampling_confidence"].dtype == np.float32
    assert np.array_equal(
        current_arrays["sampling_confidence"],
        baseline_arrays["confidence"],
    )

    per_frame_counts = baseline_arrays["per_frame_counts"]
    if point_mode == "foreground":
        assert np.array_equal(
            current_arrays["per_frame_global_counts"], per_frame_counts
        )
        assert np.array_equal(
            current_arrays["per_frame_evidence_counts"], per_frame_counts
        )
        zero_keys = (
            "per_frame_local_percentile_counts",
            "per_frame_tophat_counts",
            "per_frame_context_grid_counts",
            "per_frame_context_only_counts",
            "per_frame_overlap_counts",
        )
    else:
        assert np.array_equal(
            current_arrays["per_frame_context_grid_counts"], per_frame_counts
        )
        assert np.array_equal(
            current_arrays["per_frame_context_only_counts"], per_frame_counts
        )
        zero_keys = (
            "per_frame_global_counts",
            "per_frame_local_percentile_counts",
            "per_frame_tophat_counts",
            "per_frame_evidence_counts",
            "per_frame_overlap_counts",
        )
    for key in zero_keys:
        assert np.all(current_arrays[key] == 0), key

    assert current_attrs["sampling_mode"] == "legacy"
    assert not bool(current_attrs["include_context_grid"])
    assert not bool(current_attrs["enable_local_percentile"])
    assert not bool(current_attrs["enable_tophat"])
    assert not bool(current_attrs["enable_evidence_cleanup"])
    assert int(current_attrs["source_flags_global"]) == int(SOURCE_GLOBAL)

    source_stats = summarize_source_flags(current_arrays["source_flags"])
    for key, expected in source_stats.items():
        _assert_attr_equal(current_attrs[key], expected, key)
    assert source_stats["total_selected_points"] == num_points
    if point_mode == "foreground":
        assert source_stats["legacy_point_count_multiplier"] == 1.0
    else:
        assert source_stats["legacy_point_count_multiplier"] == 0.0


def _run_legacy_case_export(
    *,
    code_root: Path,
    input_h5: Path,
    output_h5: Path,
    output_ply: Path | None = None,
    geometry_key: str,
    preset: str,
    point_mode: str,
    sample_stride: int,
    max_points_per_frame: int,
    random_seed: int,
    extra_args: tuple[str, ...] = (),
    current: bool,
    enable_combined_options: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(code_root / EXPORT_RELATIVE_PATH),
        "--input_h5",
        str(input_h5),
        "--output_h5",
        str(output_h5),
        "--geometry_key",
        geometry_key,
        "--preset",
        preset,
        "--point_mode",
        point_mode,
        "--sample_stride",
        str(sample_stride),
        "--max_points_per_frame",
        str(max_points_per_frame),
        "--random_seed",
        str(random_seed),
        *extra_args,
    ]
    if output_ply is not None:
        command.extend(["--output_ply", str(output_ply)])
    if current:
        command.extend(["--sampling_mode", "legacy"])
        if enable_combined_options:
            command.extend(
                [
                    "--include_context_grid",
                    "--enable_local_percentile",
                    "--enable_tophat",
                    "--enable_evidence_cleanup",
                    "--evidence_open_ksize",
                    "7",
                    "--evidence_close_ksize",
                    "9",
                    "--evidence_morph_shape",
                    "cross",
                    "--evidence_min_component_area",
                    "17",
                ]
            )

    result = _run(command, cwd=code_root)
    assert output_h5.is_file()
    if output_ply is not None:
        assert output_ply.is_file()
    assert f"geometry_key: {geometry_key}" in result.stdout
    assert f"point_mode: {point_mode}" in result.stdout
    return result


def test_legacy_compatibility_matrix(
    *,
    baseline_root: Path,
    minimal_input_h5: Path,
    texture_input_h5: Path,
    temp_dir: Path,
) -> None:
    cases = [
        {
            "name": "prior_geometry",
            "input_h5": minimal_input_h5,
            "geometry_key": "prior",
            "preset": "original",
            "point_mode": "foreground",
            "sample_stride": 1,
            "max_points": 0,
            "seed": 0,
            "extra_args": ("--texture_style", "original"),
            "enable_combined_options": False,
        },
        {
            "name": "threshold_alpha_foreground",
            "input_h5": texture_input_h5,
            "geometry_key": "corr",
            "preset": "percentile85_area100",
            "point_mode": "foreground",
            "sample_stride": 1,
            "max_points": 0,
            "seed": 0,
            "extra_args": (
                "--texture_percentile",
                "85",
                "--texture_min_component_area",
                "100",
            ),
            "enable_combined_options": False,
        },
        {
            "name": "foreground_stride",
            "input_h5": texture_input_h5,
            "geometry_key": "corr",
            "preset": "original",
            "point_mode": "foreground",
            "sample_stride": 2,
            "max_points": 0,
            "seed": 0,
            "extra_args": ("--texture_style", "original"),
            "enable_combined_options": True,
        },
        {
            "name": "random_subsampling",
            "input_h5": texture_input_h5,
            "geometry_key": "corr",
            "preset": "original",
            "point_mode": "foreground",
            "sample_stride": 1,
            "max_points": 37,
            "seed": 123,
            "extra_args": ("--texture_style", "original"),
            "enable_combined_options": False,
        },
        {
            "name": "grid_mode",
            "input_h5": minimal_input_h5,
            "geometry_key": "corr",
            "preset": "original",
            "point_mode": "grid",
            "sample_stride": 4,
            "max_points": 0,
            "seed": 0,
            "extra_args": ("--texture_style", "original"),
            "enable_combined_options": False,
        },
        {
            "name": "dense_mode",
            "input_h5": minimal_input_h5,
            "geometry_key": "corr",
            "preset": "original",
            "point_mode": "dense",
            "sample_stride": 1,
            "max_points": 0,
            "seed": 0,
            "extra_args": ("--texture_style", "original"),
            "enable_combined_options": False,
        },
    ]

    output_paths: list[Path] = []
    for case in cases:
        name = str(case["name"])
        baseline_h5 = temp_dir / f"matrix_baseline_{name}.h5"
        current_h5 = temp_dir / f"matrix_current_{name}.h5"
        output_paths.extend([baseline_h5, current_h5])

        common_kwargs = {
            "input_h5": Path(case["input_h5"]),
            "geometry_key": str(case["geometry_key"]),
            "preset": str(case["preset"]),
            "point_mode": str(case["point_mode"]),
            "sample_stride": int(case["sample_stride"]),
            "max_points_per_frame": int(case["max_points"]),
            "random_seed": int(case["seed"]),
            "extra_args": tuple(case["extra_args"]),
        }
        _run_legacy_case_export(
            code_root=baseline_root,
            output_h5=baseline_h5,
            current=False,
            **common_kwargs,
        )
        _run_legacy_case_export(
            code_root=STAGE2TO4_ROOT,
            output_h5=current_h5,
            current=True,
            enable_combined_options=bool(case["enable_combined_options"]),
            **common_kwargs,
        )

        (
            baseline_arrays,
            _,
            current_arrays,
            current_attrs,
        ) = test_common_legacy_output_equality(baseline_h5, current_h5)
        assert baseline_arrays["points"].shape[0] > 0, name
        test_current_legacy_extension_fields(
            baseline_arrays,
            current_arrays,
            current_attrs,
            point_mode=str(case["point_mode"]),
        )

        if name == "random_subsampling":
            assert baseline_arrays["per_frame_counts"].tolist() == [37, 37, 37]
        if name == "grid_mode":
            assert baseline_arrays["per_frame_counts"].tolist() == [16, 16, 16]
        if name == "dense_mode":
            assert baseline_arrays["per_frame_counts"].tolist() == [256, 256, 256]
        if bool(case["enable_combined_options"]):
            assert not bool(current_attrs["include_context_grid"])
            assert not bool(current_attrs["enable_local_percentile"])
            assert not bool(current_attrs["enable_tophat"])
            assert not bool(current_attrs["enable_evidence_cleanup"])

    assert len(set(output_paths)) == len(output_paths)
    assert all(path.is_file() for path in output_paths)


def _parse_and_validate_ply(ply_path: Path, h5_path: Path) -> None:
    lines = ply_path.read_text().splitlines()
    assert lines[0] == "ply"
    assert lines[1] == "format ascii 1.0"
    end_header_index = lines.index("end_header")

    arrays, _ = _read_point_cloud_h5(h5_path)
    num_points = int(arrays["points"].shape[0])
    expected_header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        "property float confidence",
        "property int frame_order",
        "end_header",
    ]
    assert lines[: end_header_index + 1] == expected_header

    body = lines[end_header_index + 1 :]
    assert len(body) == num_points
    for index, line in enumerate(body):
        values = line.split()
        assert len(values) == 9, (index, values)
        xyz = np.asarray(values[:3], dtype=np.float64)
        rgb = np.asarray(values[3:6], dtype=np.int64)
        alpha = int(values[6])
        confidence = float(values[7])
        frame_order = int(values[8])

        np.testing.assert_allclose(
            xyz,
            arrays["points"][index],
            rtol=0.0,
            atol=1e-6,
        )
        expected_intensity = int(arrays["intensity"][index])
        assert np.array_equal(
            rgb,
            np.asarray([expected_intensity] * 3, dtype=np.int64),
        )
        assert alpha == int(arrays["alpha"][index])
        assert np.isclose(
            confidence,
            float(arrays["confidence"][index]),
            rtol=0.0,
            atol=1e-6,
        )
        assert frame_order == int(arrays["frame_order"][index])


def test_legacy_ply_compatibility(
    *,
    baseline_root: Path,
    minimal_input_h5: Path,
    texture_input_h5: Path,
    temp_dir: Path,
) -> None:
    cases = [
        {
            "name": "original_foreground",
            "input_h5": minimal_input_h5,
            "preset": "original",
            "point_mode": "foreground",
            "sample_stride": 1,
            "extra_args": ("--texture_style", "original"),
        },
        {
            "name": "threshold_alpha_foreground",
            "input_h5": texture_input_h5,
            "preset": "percentile85_area100",
            "point_mode": "foreground",
            "sample_stride": 1,
            "extra_args": (
                "--texture_percentile",
                "85",
                "--texture_min_component_area",
                "100",
            ),
        },
        {
            "name": "grid_mode",
            "input_h5": minimal_input_h5,
            "preset": "original",
            "point_mode": "grid",
            "sample_stride": 4,
            "extra_args": ("--texture_style", "original"),
        },
    ]

    output_paths: list[Path] = []
    for case in cases:
        name = str(case["name"])
        baseline_h5 = temp_dir / f"ply_baseline_{name}.h5"
        current_h5 = temp_dir / f"ply_current_{name}.h5"
        baseline_ply = temp_dir / f"ply_baseline_{name}.ply"
        current_ply = temp_dir / f"ply_current_{name}.ply"
        output_paths.extend([baseline_h5, current_h5, baseline_ply, current_ply])

        kwargs = {
            "input_h5": Path(case["input_h5"]),
            "geometry_key": "corr",
            "preset": str(case["preset"]),
            "point_mode": str(case["point_mode"]),
            "sample_stride": int(case["sample_stride"]),
            "max_points_per_frame": 0,
            "random_seed": 0,
            "extra_args": tuple(case["extra_args"]),
        }
        _run_legacy_case_export(
            code_root=baseline_root,
            output_h5=baseline_h5,
            output_ply=baseline_ply,
            current=False,
            **kwargs,
        )
        _run_legacy_case_export(
            code_root=STAGE2TO4_ROOT,
            output_h5=current_h5,
            output_ply=current_ply,
            current=True,
            **kwargs,
        )

        assert baseline_ply.read_bytes() == current_ply.read_bytes(), name
        _parse_and_validate_ply(baseline_ply, baseline_h5)
        _parse_and_validate_ply(current_ply, current_h5)

    assert len(set(output_paths)) == len(output_paths)
    assert all(path.is_file() for path in output_paths)


def _assert_loader_preserves_h5(
    raw_arrays: dict[str, np.ndarray],
    raw_attrs: dict[str, Any],
    loaded_arrays: dict[str, np.ndarray],
    loaded_attrs: dict[str, Any],
) -> None:
    for key, expected in raw_arrays.items():
        assert key in loaded_arrays, key
        actual = loaded_arrays[key]
        assert actual.shape == expected.shape, key
        assert actual.dtype == expected.dtype, key
        assert np.array_equal(actual, expected), key

    for key, expected in raw_attrs.items():
        assert key in loaded_attrs, key
        _assert_attr_equal(loaded_attrs[key], expected, key)


def test_old_h5_loader_compatibility(
    *,
    baseline_root: Path,
    input_h5: Path,
    temp_dir: Path,
) -> None:
    cases = (
        ("foreground", 1, SOURCE_GLOBAL),
        ("grid", 4, SOURCE_CONTEXT_GRID),
        ("dense", 1, SOURCE_CONTEXT_GRID),
    )

    output_paths: list[Path] = []
    for point_mode, sample_stride, expected_flag in cases:
        baseline_h5 = temp_dir / f"loader_baseline_{point_mode}.h5"
        output_paths.append(baseline_h5)
        _run_legacy_case_export(
            code_root=baseline_root,
            input_h5=input_h5,
            output_h5=baseline_h5,
            geometry_key="corr",
            preset="original",
            point_mode=point_mode,
            sample_stride=sample_stride,
            max_points_per_frame=0,
            random_seed=0,
            extra_args=("--texture_style", "original"),
            current=False,
        )

        raw_arrays, raw_attrs = _read_point_cloud_h5(baseline_h5)
        assert "source_flags" not in raw_arrays
        assert "sampling_confidence" not in raw_arrays

        loaded_arrays, loaded_attrs = load_point_cloud_h5(baseline_h5)
        _assert_loader_preserves_h5(
            raw_arrays,
            raw_attrs,
            loaded_arrays,
            loaded_attrs,
        )

        num_points = int(raw_arrays["points"].shape[0])
        assert loaded_arrays["source_flags"].shape == (num_points,)
        assert loaded_arrays["source_flags"].dtype == np.uint16
        assert np.all(loaded_arrays["source_flags"] == expected_flag)
        assert loaded_arrays["sampling_confidence"].shape == (num_points,)
        assert loaded_arrays["sampling_confidence"].dtype == np.float32
        assert np.array_equal(
            loaded_arrays["sampling_confidence"],
            raw_arrays["confidence"].astype(np.float32),
        )

        assert bool(loaded_attrs["source_flags_inferred"])
        assert bool(loaded_attrs["sampling_confidence_inferred"])
        assert loaded_attrs["source_flags_inferred_reason"] == (
            "source_flags missing in input point_cloud h5; inferred from point_mode"
        )
        assert loaded_attrs["sampling_confidence_inferred_reason"] == (
            "sampling_confidence missing in input point_cloud h5; copied confidence"
        )

    current_h5 = temp_dir / "loader_current_foreground.h5"
    output_paths.append(current_h5)
    _run_legacy_case_export(
        code_root=STAGE2TO4_ROOT,
        input_h5=input_h5,
        output_h5=current_h5,
        geometry_key="corr",
        preset="original",
        point_mode="foreground",
        sample_stride=1,
        max_points_per_frame=0,
        random_seed=0,
        extra_args=("--texture_style", "original"),
        current=True,
    )

    raw_arrays, raw_attrs = _read_point_cloud_h5(current_h5)
    loaded_arrays, loaded_attrs = load_point_cloud_h5(current_h5)
    _assert_loader_preserves_h5(
        raw_arrays,
        raw_attrs,
        loaded_arrays,
        loaded_attrs,
    )
    assert np.array_equal(
        loaded_arrays["source_flags"],
        raw_arrays["source_flags"],
    )
    assert np.array_equal(
        loaded_arrays["sampling_confidence"],
        raw_arrays["sampling_confidence"],
    )
    assert not bool(loaded_attrs["source_flags_inferred"])
    assert not bool(loaded_attrs["sampling_confidence_inferred"])
    assert "source_flags_inferred_reason" not in loaded_attrs
    assert "sampling_confidence_inferred_reason" not in loaded_attrs

    assert len(set(output_paths)) == len(output_paths)
    assert all(path.is_file() for path in output_paths)


def _run_annotation(
    *,
    code_root: Path,
    point_cloud_h5: Path,
    pseudo3d_h5: Path,
    voc_xml_root: Path,
    output_h5: Path,
) -> subprocess.CompletedProcess[str]:
    result = _run(
        [
            sys.executable,
            str(code_root / ANNOTATE_RELATIVE_PATH),
            "--point_cloud_h5",
            str(point_cloud_h5),
            "--pseudo3d_h5",
            str(pseudo3d_h5),
            "--voc_xml_root",
            str(voc_xml_root),
            "--video_name",
            "stage4_synthetic_annotation",
            "--output_h5",
            str(output_h5),
            "--geometry_key",
            "corr",
            "--xml_frame_number_offsets",
            "0",
            "--xml_frame_id_source",
            "frame_index",
            "--contour_preset",
            "original",
            "--contour_min_area",
            "0",
            "--contour_open_ksize",
            "1",
            "--contour_close_ksize",
            "1",
            "--contour_denoise",
            "none",
            "--bbox_inside_non_contour_label",
            "background",
            "--no_bbox_label",
            "-1",
        ],
        cwd=code_root,
    )
    assert output_h5.is_file()
    assert "num_points              : 193" in result.stdout
    assert "num_voc_bbox_frames     : 2" in result.stdout
    assert "num_voc_bboxes          : 2" in result.stdout
    assert "num_valid_contours      : 2" in result.stdout
    assert "num_labeled_points      : 128" in result.stdout
    return result


def _read_annotated_h5(
    path: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    group_names = ("point_cloud", "annotation", "frame_annotation", "measurement")
    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == set(group_names)
        arrays = {
            group_name: {
                key: np.asarray(handle[group_name][key][()])
                for key in handle[group_name].keys()
            }
            for group_name in group_names
        }
        group_attrs = {
            group_name: dict(handle[group_name].attrs)
            for group_name in group_names
        }
        root_attrs = dict(handle.attrs)
    return arrays, group_attrs, root_attrs


def _assert_array_identical(
    actual: np.ndarray,
    expected: np.ndarray,
    key: str,
) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, key
    assert actual.dtype == expected.dtype, key
    if expected.dtype.kind in {"f", "c"}:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            err_msg=key,
        )
    else:
        assert np.array_equal(actual, expected), key


def _assert_annotated_common_equal(
    reference_path: Path,
    actual_path: Path,
) -> None:
    reference_arrays, reference_group_attrs, reference_root_attrs = (
        _read_annotated_h5(reference_path)
    )
    actual_arrays, actual_group_attrs, actual_root_attrs = _read_annotated_h5(
        actual_path
    )

    for group_name, reference_group in reference_arrays.items():
        assert set(reference_group).issubset(actual_arrays[group_name]), group_name
        for key, expected in reference_group.items():
            _assert_array_identical(
                actual_arrays[group_name][key],
                expected,
                f"{group_name}/{key}",
            )

        for key, expected in reference_group_attrs[group_name].items():
            assert key in actual_group_attrs[group_name], (group_name, key)
            _assert_attr_equal(
                actual_group_attrs[group_name][key],
                expected,
                f"{group_name}:{key}",
            )

    for key, expected in reference_root_attrs.items():
        if key == "source_point_cloud_h5":
            continue
        assert key in actual_root_attrs, key
        _assert_attr_equal(actual_root_attrs[key], expected, key)


def _assert_annotation_labels_and_frames(annotated_h5: Path) -> None:
    arrays, _, root_attrs = _read_annotated_h5(annotated_h5)
    point_cloud = arrays["point_cloud"]
    annotation = arrays["annotation"]
    frame_annotation = arrays["frame_annotation"]

    labels = annotation["point_label"].astype(np.int8)
    valid_mask = annotation["valid_mask"].astype(bool)
    frame_order = point_cloud["frame_order"].astype(np.int32)
    pixel_xy = point_cloud["pixel_xy"].astype(np.float32)

    assert labels.shape == (193,)
    assert np.array_equal(valid_mask, labels != -1)
    assert np.all(labels[frame_order == 1] == -1)

    frame_zero = frame_order == 0
    outside_bbox = frame_zero & np.isclose(pixel_xy[:, 0], 2.0) & np.isclose(
        pixel_xy[:, 1], 25.0
    )
    assert int(outside_bbox.sum()) == 1
    assert labels[outside_bbox].tolist() == [0]
    assert np.all(labels[frame_zero & ~outside_bbox] == 1)
    assert np.all(labels[frame_order == 2] == 1)

    assert frame_annotation["frame_order"].tolist() == [0, 2]
    assert frame_annotation["frame_index"].tolist() == [110, 141]
    assert frame_annotation["valid_contour"].astype(bool).tolist() == [True, True]
    assert frame_annotation["num_frame_points"].tolist() == [65, 64]
    assert frame_annotation["num_labeled_points"].tolist() == [64, 64]
    assert frame_annotation["num_labeled_points_union"].tolist() == [64, 64]
    np.testing.assert_array_equal(
        frame_annotation["bbox_local_xyxy"],
        np.asarray([[2, 2, 14, 14], [18, 18, 30, 30]], dtype=np.float32),
    )
    assert int(root_attrs["num_points"]) == 193
    assert int(root_attrs["num_voc_bbox_frames"]) == 2
    assert int(root_attrs["num_voc_bboxes"]) == 2
    assert int(root_attrs["num_labeled_points"]) == 128


def _run_fallback_annotated_ply(
    *,
    annotated_h5: Path,
    output_ply: Path,
) -> None:
    result = _run(
        [
            sys.executable,
            str(STAGE2TO4_ROOT / ANNOTATED_PLY_RELATIVE_PATH),
            "--input_h5",
            str(annotated_h5),
            "--output_ply",
            str(output_ply),
            "--background_mode",
            "gray",
            "--color_mode",
            "annotation_source",
        ],
        cwd=STAGE2TO4_ROOT,
    )
    assert output_ply.is_file()
    assert f"Saved annotated PLY: {output_ply}" in result.stdout


def _assert_fallback_annotated_ply(
    *,
    annotated_h5: Path,
    output_ply: Path,
) -> None:
    arrays, group_attrs, root_attrs = _read_annotated_h5(annotated_h5)
    point_cloud = arrays["point_cloud"]
    annotation = arrays["annotation"]
    num_points = int(point_cloud["points"].shape[0])

    assert bool(group_attrs["point_cloud"]["source_flags_inferred"])
    assert bool(group_attrs["point_cloud"]["sampling_confidence_inferred"])
    assert bool(root_attrs["source_flags_inferred"])
    assert bool(root_attrs["sampling_confidence_inferred"])
    assert np.all(point_cloud["source_flags"] == SOURCE_GLOBAL)
    assert np.array_equal(
        point_cloud["sampling_confidence"],
        point_cloud["confidence"].astype(np.float32),
    )

    lines = output_ply.read_text().splitlines()
    assert lines[0] == "ply"
    assert lines[1] == "format ascii 1.0"
    end_header_index = lines.index("end_header")
    header = lines[: end_header_index + 1]
    assert f"element vertex {num_points}" in header
    assert "element edge" not in "\n".join(header)
    expected_properties = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        "property int label",
        "property int frame_order",
        "property float confidence",
        "property int source_flags",
        "property int source_type",
        "property float sampling_confidence",
    ]
    actual_properties = [line for line in header if line.startswith("property ")]
    assert actual_properties == expected_properties

    body = lines[end_header_index + 1 :]
    assert len(body) == num_points
    for index, line in enumerate(body):
        values = line.split()
        assert len(values) == 13, (index, values)
        xyz = np.asarray(values[0:3], dtype=np.float64)
        rgba = np.asarray(values[3:7], dtype=np.int64)
        label = int(values[7])
        frame_order = int(values[8])
        confidence = float(values[9])
        source_flags = int(values[10])
        source_type = int(values[11])
        sampling_confidence = float(values[12])

        np.testing.assert_allclose(
            xyz,
            point_cloud["points"][index],
            rtol=0.0,
            atol=1e-6,
        )
        assert np.all((rgba >= 0) & (rgba <= 255))
        assert label == int(annotation["point_label"][index])
        assert frame_order == int(point_cloud["frame_order"][index])
        assert np.isclose(
            confidence,
            float(point_cloud["confidence"][index]),
            rtol=0.0,
            atol=1e-6,
        )
        assert source_flags == int(SOURCE_GLOBAL)
        assert source_flags == int(point_cloud["source_flags"][index])
        assert source_type == int(point_cloud["source_type"][index])
        assert np.isclose(
            sampling_confidence,
            float(point_cloud["sampling_confidence"][index]),
            rtol=0.0,
            atol=1e-6,
        )


def test_legacy_annotation_compatibility(
    *,
    baseline_root: Path,
    temp_dir: Path,
) -> None:
    pseudo3d_h5 = temp_dir / "synthetic_legacy_annotation_input.h5"
    voc_xml_root = temp_dir / "synthetic_voc_xml"
    write_synthetic_annotation_input(pseudo3d_h5)
    write_synthetic_voc_xml_root(voc_xml_root)

    baseline_point_cloud_h5 = temp_dir / "annotation_baseline_point_cloud.h5"
    current_point_cloud_h5 = temp_dir / "annotation_current_point_cloud.h5"
    export_kwargs = {
        "input_h5": pseudo3d_h5,
        "geometry_key": "corr",
        "preset": "original",
        "point_mode": "foreground",
        "sample_stride": 1,
        "max_points_per_frame": 0,
        "random_seed": 0,
        "extra_args": ("--texture_style", "original"),
    }
    _run_legacy_case_export(
        code_root=baseline_root,
        output_h5=baseline_point_cloud_h5,
        current=False,
        **export_kwargs,
    )
    _run_legacy_case_export(
        code_root=STAGE2TO4_ROOT,
        output_h5=current_point_cloud_h5,
        current=True,
        **export_kwargs,
    )

    baseline_annotated_h5 = temp_dir / "annotation_baseline_reference.h5"
    current_annotated_h5 = temp_dir / "annotation_current_legacy.h5"
    fallback_annotated_h5 = temp_dir / "annotation_current_from_old_h5.h5"
    _run_annotation(
        code_root=baseline_root,
        point_cloud_h5=baseline_point_cloud_h5,
        pseudo3d_h5=pseudo3d_h5,
        voc_xml_root=voc_xml_root,
        output_h5=baseline_annotated_h5,
    )
    _run_annotation(
        code_root=STAGE2TO4_ROOT,
        point_cloud_h5=current_point_cloud_h5,
        pseudo3d_h5=pseudo3d_h5,
        voc_xml_root=voc_xml_root,
        output_h5=current_annotated_h5,
    )
    _run_annotation(
        code_root=STAGE2TO4_ROOT,
        point_cloud_h5=baseline_point_cloud_h5,
        pseudo3d_h5=pseudo3d_h5,
        voc_xml_root=voc_xml_root,
        output_h5=fallback_annotated_h5,
    )

    _assert_annotated_common_equal(baseline_annotated_h5, current_annotated_h5)
    _assert_annotated_common_equal(baseline_annotated_h5, fallback_annotated_h5)
    for path in (
        baseline_annotated_h5,
        current_annotated_h5,
        fallback_annotated_h5,
    ):
        _assert_annotation_labels_and_frames(path)

    current_raw_arrays, current_raw_attrs = _read_point_cloud_h5(
        current_point_cloud_h5
    )
    current_arrays, current_group_attrs, current_root_attrs = _read_annotated_h5(
        current_annotated_h5
    )
    for key, expected in current_raw_arrays.items():
        _assert_array_identical(
            current_arrays["point_cloud"][key],
            expected,
            f"current point_cloud/{key}",
        )
    for key, expected in current_raw_attrs.items():
        _assert_attr_equal(current_group_attrs["point_cloud"][key], expected, key)
    assert not bool(current_group_attrs["point_cloud"]["source_flags_inferred"])
    assert not bool(
        current_group_attrs["point_cloud"]["sampling_confidence_inferred"]
    )
    assert not bool(current_root_attrs["source_flags_inferred"])
    assert not bool(current_root_attrs["sampling_confidence_inferred"])
    assert bool(current_root_attrs["point_cloud_has_source_flags"])
    assert bool(current_root_attrs["point_cloud_has_sampling_confidence"])

    baseline_raw_arrays, baseline_raw_attrs = _read_point_cloud_h5(
        baseline_point_cloud_h5
    )
    fallback_arrays, fallback_group_attrs, fallback_root_attrs = _read_annotated_h5(
        fallback_annotated_h5
    )
    for key, expected in baseline_raw_arrays.items():
        _assert_array_identical(
            fallback_arrays["point_cloud"][key],
            expected,
            f"fallback point_cloud/{key}",
        )
    for key, expected in baseline_raw_attrs.items():
        _assert_attr_equal(fallback_group_attrs["point_cloud"][key], expected, key)

    fallback_point_cloud = fallback_arrays["point_cloud"]
    assert fallback_point_cloud["source_flags"].dtype == np.uint16
    assert np.all(fallback_point_cloud["source_flags"] == SOURCE_GLOBAL)
    assert fallback_point_cloud["sampling_confidence"].dtype == np.float32
    assert np.array_equal(
        fallback_point_cloud["sampling_confidence"],
        baseline_raw_arrays["confidence"].astype(np.float32),
    )
    assert bool(fallback_group_attrs["point_cloud"]["source_flags_inferred"])
    assert bool(
        fallback_group_attrs["point_cloud"]["sampling_confidence_inferred"]
    )
    assert "source_flags_inferred_reason" in fallback_group_attrs["point_cloud"]
    assert (
        "sampling_confidence_inferred_reason"
        in fallback_group_attrs["point_cloud"]
    )
    assert bool(fallback_root_attrs["source_flags_inferred"])
    assert bool(fallback_root_attrs["sampling_confidence_inferred"])
    assert not bool(fallback_root_attrs["point_cloud_has_source_flags"])
    assert not bool(fallback_root_attrs["point_cloud_has_sampling_confidence"])
    assert fallback_root_attrs["point_cloud_sampling_mode"] == "legacy"

    fallback_annotated_ply = temp_dir / "annotation_current_from_old_h5.ply"
    _run_fallback_annotated_ply(
        annotated_h5=fallback_annotated_h5,
        output_ply=fallback_annotated_ply,
    )
    _assert_fallback_annotated_ply(
        annotated_h5=fallback_annotated_h5,
        output_ply=fallback_annotated_ply,
    )

    output_paths = (
        pseudo3d_h5,
        baseline_point_cloud_h5,
        current_point_cloud_h5,
        baseline_annotated_h5,
        current_annotated_h5,
        fallback_annotated_h5,
        fallback_annotated_ply,
    )
    assert len(set(output_paths)) == len(output_paths)
    assert all(path.is_file() for path in output_paths)


def main() -> None:
    test_baseline_commit_available()
    print(f"[OK] baseline commit available: {BASELINE_COMMIT}")

    with tempfile.TemporaryDirectory(prefix="stage4_legacy_compatibility_") as temp:
        temp_dir = Path(temp)
        baseline_root = extract_baseline(temp_dir)
        print("[OK] baseline archive extraction")

        test_cli_help(baseline_root)
        print("[OK] baseline CLI help")
        print("[OK] current CLI help")

        input_h5 = temp_dir / "synthetic_legacy_input.h5"
        write_synthetic_legacy_input(input_h5)
        assert input_h5.is_file()
        print("[OK] synthetic legacy input H5 creation")

        baseline_h5 = temp_dir / "baseline_original_foreground_corr.h5"
        current_h5 = temp_dir / "current_original_foreground_corr.h5"
        _run_original_foreground_export(
            code_root=baseline_root,
            input_h5=input_h5,
            output_h5=baseline_h5,
            current=False,
        )
        print("[OK] baseline original foreground export")

        _run_original_foreground_export(
            code_root=STAGE2TO4_ROOT,
            input_h5=input_h5,
            output_h5=current_h5,
            current=True,
        )
        print("[OK] current legacy original foreground export")

        (
            baseline_arrays,
            _,
            current_arrays,
            current_attrs,
        ) = test_common_legacy_output_equality(baseline_h5, current_h5)
        print("[OK] common point-cloud datasets are identical")
        print("[OK] common metadata is identical")

        test_current_legacy_extension_fields(
            baseline_arrays,
            current_arrays,
            current_attrs,
        )
        print("[OK] current legacy source flags")
        print("[OK] current legacy confidence compatibility")
        print("[OK] current legacy frame statistics")

        texture_input_h5 = temp_dir / "synthetic_legacy_texture_input.h5"
        write_synthetic_texture_input(texture_input_h5)
        assert texture_input_h5.is_file()

        test_legacy_compatibility_matrix(
            baseline_root=baseline_root,
            minimal_input_h5=input_h5,
            texture_input_h5=texture_input_h5,
            temp_dir=temp_dir,
        )
        print("[OK] threshold-alpha foreground compatibility")
        print("[OK] foreground stride compatibility")
        print("[OK] random subsampling compatibility")
        print("[OK] grid mode compatibility")
        print("[OK] dense mode compatibility")
        print("[OK] prior geometry compatibility")
        print("[OK] legacy ignores combined_v2 enable options")
        print("[OK] all common datasets match across compatibility matrix")
        print("[OK] all common metadata matches across compatibility matrix")
        print("[OK] current legacy extension fields match point modes")

        test_legacy_ply_compatibility(
            baseline_root=baseline_root,
            minimal_input_h5=input_h5,
            texture_input_h5=texture_input_h5,
            temp_dir=temp_dir,
        )
        print("[OK] original foreground PLY compatibility")
        print("[OK] threshold-alpha foreground PLY compatibility")
        print("[OK] grid mode PLY compatibility")
        print("[OK] baseline/current PLY files are byte-identical")
        print("[OK] PLY header and vertex counts")
        print("[OK] PLY body matches H5 point data")
        print("[OK] PLY outputs are separated")

        test_old_h5_loader_compatibility(
            baseline_root=baseline_root,
            input_h5=input_h5,
            temp_dir=temp_dir,
        )
        print("[OK] old foreground H5 loader fallback")
        print("[OK] old grid/dense H5 loader fallback")
        print("[OK] old sampling confidence fallback")
        print("[OK] old H5 arrays and metadata are preserved")
        print("[OK] current H5 sampling fields are not inferred")

        test_legacy_annotation_compatibility(
            baseline_root=baseline_root,
            temp_dir=temp_dir,
        )
        print("[OK] baseline/current legacy annotation labels")
        print("[OK] old H5/current annotation fallback labels")
        print("[OK] annotation BBox and frame alignment")
        print("[OK] annotation point order and common schema")
        print("[OK] annotation sampling fields and inference metadata")
        print("[OK] old-schema fallback annotated PLY export")
        print("[OK] fallback PLY source flags and sampling confidence")
        print("[OK] annotation outputs are separated")

    print("Stage 4 legacy compatibility checks passed.")


if __name__ == "__main__":
    main()
