from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.pseudo3d_sampling import (
    summarize_per_frame_counts,
    summarize_source_flags,
)

DEFAULT_REBUILD_ROOT = Path(
    "/mnt/data/3d_projects/pseudo3d_dataset/stage4_rebuild/260722"
)
DEFAULT_REAL_VIDEO = "20250610_155045_7420"
DEFAULT_REAL_INPUT = (
    DEFAULT_REBUILD_ROOT
    / "pseudo3d_outputs"
    / f"{DEFAULT_REAL_VIDEO}_ts448_oym96_corr.h5"
)
DEFAULT_BATCH_ROOT = DEFAULT_REBUILD_ROOT / "combined_v2_default_corr"
DEFAULT_BATCH_SUMMARY = DEFAULT_BATCH_ROOT / "summary.csv"

SINGLE_EXPORT = Path("pseudo3d/export/export_pseudo3d_point_cloud.py")
BATCH_EXPORT = Path("pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py")

POINT_LEVEL_KEYS = (
    "points",
    "intensity",
    "alpha",
    "confidence",
    "frame_index",
    "frame_order",
    "pixel_xy",
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

SAMPLING_ARGUMENTS = (
    "--sampling_mode",
    "--include_context_grid",
    "--context_grid_stride",
    "--context_grid_phase",
    "--enable_local_percentile",
    "--local_window_size",
    "--local_percentile",
    "--local_min_contrast",
    "--enable_tophat",
    "--tophat_kernel_size",
    "--tophat_percentile",
    "--tophat_min_response",
    "--tophat_morph_shape",
    "--enable_evidence_cleanup",
    "--evidence_open_ksize",
    "--evidence_close_ksize",
    "--evidence_morph_shape",
    "--evidence_min_component_area",
)

SAMPLING_SHELLS = (
    Path("pseudo3d/export/export_pseudo3d_point_cloud.sh"),
    Path("pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.sh"),
    Path("pseudo3d/annotation/annotate_pseudo3d_point_cloud.sh"),
    Path("pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.sh"),
    Path("pseudo3d/export/export_annotated_point_cloud_ply.sh"),
    Path("pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.sh"),
    Path("pseudo3d/export/export_annotation_mask_visualization.sh"),
    Path("pseudo3d/batch/export/batch_export_annotation_mask_visualization.sh"),
    Path("pseudo3d/pipelines/batch_infer_and_export_pseudo3d.sh"),
)

CLI_FILES = (
    SINGLE_EXPORT,
    BATCH_EXPORT,
    Path("pseudo3d/annotation/annotate_pseudo3d_point_cloud.py"),
    Path("pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py"),
    Path("pseudo3d/export/export_annotated_point_cloud_ply.py"),
    Path("pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.py"),
    Path("pseudo3d/export/export_annotation_mask_visualization.py"),
    Path("pseudo3d/batch/export/batch_export_annotation_mask_visualization.py"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run(
    command: list[str],
    *,
    timeout: int,
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            "Command failed\n"
            f"command: {' '.join(command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(
            "Command unexpectedly succeeded\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _value_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind in {"S", "U", "O"} or right_array.dtype.kind in {
        "S",
        "U",
        "O",
    }:
        left_values = [_decode_text(value) for value in left_array.reshape(-1)]
        right_values = [_decode_text(value) for value in right_array.reshape(-1)]
        return left_values == right_values
    return bool(np.array_equal(left_array, right_array, equal_nan=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _real_single_command(
    *,
    input_h5: Path,
    output_h5: Path,
    output_ply: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / SINGLE_EXPORT),
        "--input_h5",
        str(input_h5),
        "--output_h5",
        str(output_h5),
        "--output_ply",
        str(output_ply),
        "--geometry_key",
        "corr",
        "--preset",
        "percentile85_area100",
        "--point_mode",
        "foreground",
        "--sample_stride",
        "2",
        "--sampling_mode",
        "combined_v2",
        "--include_context_grid",
        "--context_grid_stride",
        "4",
        "--context_grid_phase",
        "origin",
        "--enable_local_percentile",
        "--local_window_size",
        "41",
        "--local_percentile",
        "80",
        "--local_min_contrast",
        "4",
        "--enable_tophat",
        "--tophat_kernel_size",
        "21",
        "--tophat_percentile",
        "85",
        "--tophat_min_response",
        "2",
        "--tophat_morph_shape",
        "ellipse",
        "--enable_evidence_cleanup",
        "--evidence_open_ksize",
        "3",
        "--evidence_close_ksize",
        "5",
        "--evidence_morph_shape",
        "ellipse",
        "--evidence_min_component_area",
        "100",
        "--texture_percentile",
        "85",
        "--texture_min_component_area",
        "100",
    ]


def _assert_h5_single_batch_equal(single_h5: Path, batch_h5: Path) -> None:
    with h5py.File(single_h5, "r") as single_file, h5py.File(
        batch_h5, "r"
    ) as batch_file:
        _require(set(single_file.keys()) == {"point_cloud"}, "Unexpected single H5 groups")
        _require(set(batch_file.keys()) == {"point_cloud"}, "Unexpected batch H5 groups")
        single_group = single_file["point_cloud"]
        batch_group = batch_file["point_cloud"]
        _require(
            set(single_group.keys()) == set(batch_group.keys()),
            "Single/batch point_cloud dataset keys differ",
        )
        for key in sorted(single_group.keys()):
            single_dataset = single_group[key]
            batch_dataset = batch_group[key]
            _require(
                single_dataset.shape == batch_dataset.shape,
                f"Single/batch shape differs: {key}",
            )
            _require(
                single_dataset.dtype == batch_dataset.dtype,
                f"Single/batch dtype differs: {key}",
            )
            _require(
                _value_equal(single_dataset[...], batch_dataset[...]),
                f"Single/batch values or ordering differ: {key}",
            )
            for property_name in (
                "compression",
                "compression_opts",
                "shuffle",
                "fletcher32",
                "scaleoffset",
            ):
                _require(
                    getattr(single_dataset, property_name)
                    == getattr(batch_dataset, property_name),
                    f"Single/batch H5 filter differs: {key}/{property_name}",
                )

        single_attrs = dict(single_group.attrs)
        batch_attrs = dict(batch_group.attrs)
        _require(
            set(batch_attrs) - set(single_attrs) == {"batch_source"},
            "Expected batch_source to be the only batch-specific H5 attribute",
        )
        _require(
            _decode_text(batch_attrs["batch_source"])
            == "batch_export_pseudo3d_point_cloud.py",
            "Unexpected batch_source metadata",
        )
        for key, expected in single_attrs.items():
            _require(key in batch_attrs, f"Batch metadata missing: {key}")
            _require(
                _value_equal(batch_attrs[key], expected),
                f"Single/batch metadata differs: {key}",
            )


def _assert_real_single_batch_equivalence(
    args: argparse.Namespace,
    temp_dir: Path,
) -> None:
    _require(args.real_input.is_file(), f"Real pseudo3D H5 not found: {args.real_input}")
    batch_h5 = (
        Path(args.batch_root)
        / args.real_video
        / f"{args.real_video}_pointcloud_foreground_combined_v2.h5"
    )
    batch_ply = (
        Path(args.batch_root)
        / args.real_video
        / f"{args.real_video}_pointcloud_foreground_combined_v2.ply"
    )
    _require(batch_h5.is_file(), f"Batch H5 not found: {batch_h5}")
    _require(batch_ply.is_file(), f"Batch PLY not found: {batch_ply}")

    single_h5 = temp_dir / "single_combined_v2.h5"
    single_ply = temp_dir / "single_combined_v2.ply"
    result = _run(
        _real_single_command(
            input_h5=args.real_input,
            output_h5=single_h5,
            output_ply=single_ply,
        ),
        timeout=args.real_timeout,
        expect_success=True,
    )
    _require(single_h5.is_file(), "Single CLI did not create H5")
    _require(single_ply.is_file(), "Single CLI did not create PLY")
    _require("sampling_mode: combined_v2" in result.stdout, "Single CLI mode output differs")
    _assert_h5_single_batch_equal(single_h5, batch_h5)
    _require(
        single_ply.stat().st_size == batch_ply.stat().st_size,
        "Single/batch raw PLY sizes differ",
    )
    _require(_sha256(single_ply) == _sha256(batch_ply), "Single/batch raw PLY differs")


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _assert_float_text(actual: str, expected: float, label: str) -> None:
    _require(actual != "", f"Missing summary value: {label}")
    _require(
        np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-9),
        f"Summary {label}: {actual} != {expected}",
    )


def _assert_batch_summary(batch_summary: Path, expected_videos: int) -> None:
    rows = _read_csv(batch_summary)
    _require(len(rows) == expected_videos, f"Batch summary rows={len(rows)}")
    seen: set[str] = set()
    for row in rows:
        video_name = row.get("video_name", "")
        _require(video_name and video_name not in seen, f"Invalid summary video: {video_name}")
        seen.add(video_name)
        _require(row.get("status") == "ok", f"Batch summary status: {video_name}")
        _require(not row.get("error", "").strip(), f"Batch summary error: {video_name}")
        _require(
            row.get("sampling_mode") == "combined_v2",
            f"Batch summary sampling mode: {video_name}",
        )
        output_h5 = Path(row["output_h5"])
        output_ply = Path(row["output_ply"])
        _require(output_h5.is_file(), f"Summary H5 missing: {output_h5}")
        _require(output_ply.is_file(), f"Summary PLY missing: {output_ply}")
        _require(
            output_h5.name.endswith("_pointcloud_foreground_combined_v2.h5"),
            f"combined_v2 tag missing from H5: {output_h5.name}",
        )
        _require(
            output_ply.name.endswith("_pointcloud_foreground_combined_v2.ply"),
            f"combined_v2 tag missing from PLY: {output_ply.name}",
        )

        with h5py.File(output_h5, "r") as h5_file:
            group = h5_file["point_cloud"]
            attrs = dict(group.attrs)
            counts = group["per_frame_counts"][:]
            flags = group["source_flags"][:]
            num_points = int(group["points"].shape[0])
            num_frames = int(counts.shape[0])
            _require(int(counts.sum()) == num_points, f"Frame counts differ: {video_name}")
            for key in POINT_LEVEL_KEYS:
                _require(group[key].shape[0] == num_points, f"Point length: {key}")
            for key in FRAME_LEVEL_KEYS:
                _require(group[key].shape == (num_frames,), f"Frame shape: {key}")
            count_stats = summarize_per_frame_counts(counts)
            source_stats = summarize_source_flags(flags)

        _require(int(row["num_points"]) == num_points, f"Summary points: {video_name}")
        _require(int(row["num_frames"]) == num_frames, f"Summary frames: {video_name}")
        _require(int(attrs["sample_stride"]) == 2, f"Unexpected sample stride: {video_name}")
        for field in (
            "points_per_frame_min",
            "points_per_frame_median",
            "points_per_frame_max",
            "points_per_frame_mean",
        ):
            _assert_float_text(row[field], count_stats[field], f"{video_name}/{field}")
            _require(
                np.isclose(float(attrs[field]), float(count_stats[field]), rtol=0.0, atol=1e-9),
                f"H5 metadata {field}: {video_name}",
            )
        for field in (
            "global_evidence_points",
            "local_percentile_points",
            "tophat_points",
            "context_grid_points",
            "context_only_points",
            "evidence_points",
            "overlap_points",
            "legacy_point_count_multiplier",
        ):
            _assert_float_text(row[field], source_stats[field], f"{video_name}/{field}")
            _require(
                np.isclose(float(attrs[field]), float(source_stats[field]), rtol=0.0, atol=1e-9),
                f"H5 metadata {field}: {video_name}",
            )


def _translation(tx: float, ty: float, tz: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def _write_synthetic_pseudo3d(path: Path) -> None:
    images = np.zeros((3, 16, 16), dtype=np.float64)
    images[0, 2:6, 2:6] = 1.0
    images[1, 7:11, 4:12] = 0.7
    images[2, 4:13, 8:10] = 0.9
    tracking = np.stack(
        [_translation(0.0, 0.0, float(index)) for index in range(3)],
        axis=0,
    )
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("local_encoder_images", data=images)
        h5_file.create_dataset("frame_indices", data=np.asarray([3, 8, 15]))
        h5_file.create_dataset("pixel_to_image", data=np.eye(4, dtype=np.float64))
        h5_file.create_dataset("pred_tracking", data=tracking)
        h5_file.create_dataset("pred_tracking_corr", data=tracking)


def _batch_base_command(
    *,
    h5_list: Path,
    output_root: Path,
    summary_csv: Path,
    sampling_mode: str,
    filename_tag: str,
) -> list[str]:
    tag_suffix = f"_{filename_tag}" if filename_tag else ""
    command = [
        sys.executable,
        str(REPO_ROOT / BATCH_EXPORT),
        "--h5_list",
        str(h5_list),
        "--output_root",
        str(output_root),
        "--summary_csv",
        str(summary_csv),
        "--video_suffix_to_strip",
        "_synthetic",
        "--output_subdir_template",
        "{video_name}",
        "--output_h5_filename_template",
        f"{{video_name}}_pointcloud_foreground{tag_suffix}.h5",
        "--output_ply_filename_template",
        f"{{video_name}}_pointcloud_foreground{tag_suffix}.ply",
        "--geometry_key",
        "corr",
        "--preset",
        "original",
        "--point_mode",
        "foreground",
        "--sample_stride",
        "2",
        "--sampling_mode",
        sampling_mode,
        "--no_ply",
    ]
    return command


def _assert_skip_and_tag_isolation(temp_dir: Path) -> None:
    synthetic_h5 = temp_dir / "tag_case_synthetic.h5"
    _write_synthetic_pseudo3d(synthetic_h5)
    h5_list = temp_dir / "tag_case_list.txt"
    h5_list.write_text(f"{synthetic_h5}\n")
    output_root = temp_dir / "tag_outputs"

    legacy_summary = temp_dir / "legacy_summary.csv"
    legacy_command = _batch_base_command(
        h5_list=h5_list,
        output_root=output_root,
        summary_csv=legacy_summary,
        sampling_mode="legacy",
        filename_tag="",
    )
    _run(legacy_command, timeout=120, expect_success=True)

    combined_summary = temp_dir / "combined_summary.csv"
    combined_command = _batch_base_command(
        h5_list=h5_list,
        output_root=output_root,
        summary_csv=combined_summary,
        sampling_mode="combined_v2",
        filename_tag="combined_v2",
    )
    _run(combined_command + ["--skip_existing"], timeout=120, expect_success=True)
    first_combined_rows = _read_csv(combined_summary)
    _require(first_combined_rows[0]["status"] == "ok", "Legacy output caused false skip")

    second_summary = temp_dir / "combined_skip_summary.csv"
    second_command = _batch_base_command(
        h5_list=h5_list,
        output_root=output_root,
        summary_csv=second_summary,
        sampling_mode="combined_v2",
        filename_tag="combined_v2",
    )
    _run(second_command + ["--skip_existing"], timeout=120, expect_success=True)
    second_rows = _read_csv(second_summary)
    _require(second_rows[0]["status"] == "skipped", "Same-mode output was not skipped")

    video_dir = output_root / "tag_case"
    legacy_h5 = video_dir / "tag_case_pointcloud_foreground.h5"
    combined_h5 = video_dir / "tag_case_pointcloud_foreground_combined_v2.h5"
    _require(legacy_h5.is_file() and combined_h5.is_file(), "Sampling outputs separated")
    with h5py.File(legacy_h5, "r") as legacy_file, h5py.File(
        combined_h5, "r"
    ) as combined_file:
        _require(
            _decode_text(legacy_file["point_cloud"].attrs["sampling_mode"]) == "legacy",
            "Legacy tag contains non-legacy H5",
        )
        _require(
            _decode_text(combined_file["point_cloud"].attrs["sampling_mode"])
            == "combined_v2",
            "combined_v2 tag contains wrong H5",
        )


def _assert_batch_failure_modes(temp_dir: Path) -> None:
    valid_h5 = temp_dir / "valid_synthetic.h5"
    missing_h5 = temp_dir / "missing_synthetic.h5"
    _write_synthetic_pseudo3d(valid_h5)
    h5_list = temp_dir / "failure_list.txt"
    h5_list.write_text(f"{missing_h5}\n{valid_h5}\n")

    stop_root = temp_dir / "stop_outputs"
    stop_summary = temp_dir / "stop_summary.csv"
    stop_command = _batch_base_command(
        h5_list=h5_list,
        output_root=stop_root,
        summary_csv=stop_summary,
        sampling_mode="legacy",
        filename_tag="",
    )
    stop_result = _run(stop_command, timeout=120, expect_success=False)
    stop_rows = _read_csv(stop_summary)
    _require(len(stop_rows) == 1, "Batch did not stop after first failure")
    _require(stop_rows[0]["status"] == "failed", "Stop summary lacks failure")
    _require(stop_rows[0]["error"], "Stop summary lacks failure reason")
    _require("failed   : 1" in stop_result.stdout, "Stop console failure count differs")
    _require(
        not (stop_root / "valid" / "valid_pointcloud_foreground.h5").exists(),
        "Batch processed an item after stop-on-error",
    )

    continue_root = temp_dir / "continue_outputs"
    continue_summary = temp_dir / "continue_summary.csv"
    continue_command = _batch_base_command(
        h5_list=h5_list,
        output_root=continue_root,
        summary_csv=continue_summary,
        sampling_mode="legacy",
        filename_tag="",
    )
    continue_result = _run(
        continue_command + ["--continue_on_error"],
        timeout=120,
        expect_success=False,
    )
    continue_rows = _read_csv(continue_summary)
    _require(len(continue_rows) == 2, "continue_on_error did not visit all inputs")
    _require(
        [row["status"] for row in continue_rows] == ["failed", "ok"],
        "continue_on_error summary statuses differ",
    )
    _require(continue_rows[0]["error"], "Continue summary lacks failure reason")
    valid_output = (
        continue_root / "valid" / "valid_pointcloud_foreground.h5"
    )
    _require(valid_output.is_file(), "continue_on_error did not process valid input")
    _require("processed: 1" in continue_result.stdout, "Continue processed count differs")
    _require("failed   : 1" in continue_result.stdout, "Continue failure count differs")


def _assert_shell_and_pipeline_contracts() -> None:
    for relative_path in SAMPLING_SHELLS:
        path = REPO_ROOT / relative_path
        _require(path.is_file(), f"Shell script missing: {relative_path}")
        content = path.read_text()
        _require(
            re.search(r'^SAMPLING_MODE="legacy"', content, flags=re.MULTILINE) is not None,
            f"Pipeline default is not legacy: {relative_path}",
        )
        _require(
            'POINT_CLOUD_TAG="${MODE}"' in content,
            f"Legacy point-cloud tag missing: {relative_path}",
        )
        _require(
            'POINT_CLOUD_TAG="${MODE}_${SAMPLING_MODE}"' in content,
            f"combined_v2 point-cloud tag missing: {relative_path}",
        )
        _run(["bash", "-n", str(path)], timeout=30, expect_success=True)

    point_cloud_shells = (
        Path("pseudo3d/export/export_pseudo3d_point_cloud.sh"),
        Path("pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.sh"),
        Path("pseudo3d/pipelines/batch_infer_and_export_pseudo3d.sh"),
    )
    for relative_path in point_cloud_shells:
        content = (REPO_ROOT / relative_path).read_text()
        for argument in SAMPLING_ARGUMENTS:
            _require(argument in content, f"{relative_path} does not forward {argument}")

    downstream_contracts = {
        Path("pseudo3d/annotation/annotate_pseudo3d_point_cloud.sh"): (
            "_pointcloud_${POINT_CLOUD_TAG}.h5",
            "_pointcloud_annotated_${POINT_CLOUD_TAG}.h5",
        ),
        Path("pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.sh"): (
            "_pointcloud_${POINT_CLOUD_TAG}.h5",
            "_pointcloud_annotated_${POINT_CLOUD_TAG}.h5",
        ),
        Path("pseudo3d/export/export_annotated_point_cloud_ply.sh"): (
            "_pointcloud_annotated_${POINT_CLOUD_TAG}.h5",
            "_pointcloud_annotated_${POINT_CLOUD_TAG}.ply",
        ),
        Path("pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.sh"): (
            "*_pointcloud_annotated_${POINT_CLOUD_TAG}.h5",
        ),
        Path("pseudo3d/export/export_annotation_mask_visualization.sh"): (
            "_pointcloud_${POINT_CLOUD_TAG}.h5",
            "annotation_mask_textures_${POINT_CLOUD_TAG}",
        ),
        Path("pseudo3d/batch/export/batch_export_annotation_mask_visualization.sh"): (
            "*_pointcloud_annotated_${POINT_CLOUD_TAG}.h5",
        ),
    }
    for relative_path, tokens in downstream_contracts.items():
        content = (REPO_ROOT / relative_path).read_text()
        for token in tokens:
            _require(token in content, f"Tag propagation missing in {relative_path}: {token}")

    pipeline = (REPO_ROOT / "pseudo3d/pipelines/batch_infer_and_export_pseudo3d.sh").read_text()
    for token in (
        "{video_name}_pointcloud_${POINT_CLOUD_TAG}.h5",
        "{video_name}_pointcloud_annotated_${POINT_CLOUD_TAG}.h5",
        "*_pointcloud_annotated_${POINT_CLOUD_TAG}.h5",
        "--strict_xml_annotation_dir",
        "--xml_annotation_dir_name annotations_renamed",
    ):
        _require(token in pipeline, f"Batch pipeline propagation missing: {token}")


def _assert_compile_and_cli_help() -> None:
    for relative_path in CLI_FILES:
        path = REPO_ROOT / relative_path
        compile(path.read_text(), str(path), "exec")
        result = _run(
            [sys.executable, str(path), "--help"],
            timeout=60,
            expect_success=True,
        )
        _require("usage:" in result.stdout.lower(), f"CLI help missing: {relative_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4 final single/batch/shell/pipeline integration check. "
            "Real-data outputs are read-only; generated test outputs use a "
            "temporary directory."
        )
    )
    parser.add_argument("--real_video", default=DEFAULT_REAL_VIDEO)
    parser.add_argument("--real_input", type=Path, default=DEFAULT_REAL_INPUT)
    parser.add_argument("--batch_root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--batch_summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--expected_batch_videos", type=int, default=3)
    parser.add_argument("--real_timeout", type=int, default=900)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _require(args.real_timeout > 0, "real_timeout must be positive")
    print("Stage 4 final batch/pipeline integration check")
    print(f"real_video            : {args.real_video}")
    print(f"real_input            : {args.real_input}")
    print(f"batch_root            : {args.batch_root}")
    print(f"batch_summary         : {args.batch_summary}")
    print(f"expected_batch_videos : {args.expected_batch_videos}")

    with tempfile.TemporaryDirectory(prefix="stage4_batch_pipeline_") as temp:
        temp_dir = Path(temp)
        _assert_real_single_batch_equivalence(args, temp_dir)
        print("[OK] real-data single/batch H5 datasets and metadata")
        print("[OK] real-data single/batch raw PLY byte identity")

        _assert_batch_summary(args.batch_summary, args.expected_batch_videos)
        print("[OK] batch summary and saved H5 statistics")
        print("[OK] combined_v2 output filename isolation")

        _assert_skip_and_tag_isolation(temp_dir)
        print("[OK] skip_existing same-mode behavior")
        print("[OK] skip_existing does not confuse legacy and combined_v2")

        _assert_batch_failure_modes(temp_dir)
        print("[OK] batch stop-on-error behavior")
        print("[OK] batch continue_on_error behavior and failure reason")

        _assert_shell_and_pipeline_contracts()
        print("[OK] shell sampling argument and filename propagation")
        print("[OK] pipeline strict XML propagation")
        print("[OK] shell and pipeline defaults remain legacy")

        _assert_compile_and_cli_help()
        print("[OK] target Python compile and CLI help")
        print("[OK] target shell syntax")

    print("\nFinal integration summary")
    print("  real videos compared    : 1")
    print(f"  batch summaries checked : {args.expected_batch_videos}")
    print("  sampling modes isolated : legacy, combined_v2")
    print("  failure modes checked   : stop, continue")
    print("  persistent files written: 0")
    print("Stage 4 final batch/pipeline integration checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
