from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import yaml

from pseudo3d.analysis.stage4_sampling_evaluation import (
    SOURCE_BITS,
    DenseTeacherVideo,
    ablate_source,
    bbox_selected_counts,
    build_dense_teacher_video,
    label_counts,
    positive_extent_metrics,
    source_label_counts,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import (
    DenseTeacherConfig,
    Stage4SweepManifestItem,
    file_sha256,
    load_dense_teacher_config,
    load_stage4_sweep_manifest,
)
from src.utils.alpha_texture_processing import AlphaTextureConfig, make_texture_image
from src.utils.pseudo3d_sampling import (
    PointSamplingConfig,
    build_global_evidence_mask,
    build_sampling_source_flags,
)


@dataclass(frozen=True)
class EvaluationConfig:
    config_id: str
    config_name: str
    stage: str
    sampling: PointSamplingConfig
    sample_stride: int


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"search config '{name}' must be a mapping")
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sampling_from_mapping(raw: Mapping[str, Any]) -> PointSamplingConfig:
    allowed = {field.name for field in fields(PointSamplingConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown PointSamplingConfig fields: {unknown}")
    base = PointSamplingConfig.combined_v2_defaults()
    values = asdict(base)
    values.update(raw)
    config = PointSamplingConfig(**values)
    if config.sampling_mode != "combined_v2":
        raise ValueError("Parameter sweep configs must use sampling_mode=combined_v2")
    return config


def _expand_search_configs(
    raw: Mapping[str, Any],
    *,
    max_configs: int,
) -> tuple[list[EvaluationConfig], dict[str, Any]]:
    fixed = _require_mapping(raw.get("fixed", {}), "fixed")
    base_sampling_raw = _require_mapping(
        fixed.get("sampling", {}),
        "fixed.sampling",
    )
    base_sampling = _sampling_from_mapping(base_sampling_raw)
    base_sample_stride = int(fixed.get("sample_stride", 2))
    if base_sample_stride < 1:
        raise ValueError("fixed.sample_stride must be >= 1")

    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("search config 'stages' must be a non-empty list")

    expanded: list[tuple[str, str, dict[str, Any]]] = []
    for stage_raw in stages:
        stage_map = _require_mapping(stage_raw, "stages[]")
        stage_name = str(stage_map.get("name", "")).strip()
        if not stage_name:
            raise ValueError("Each search stage requires a name")
        stage_base = dict(
            _require_mapping(stage_map.get("base_overrides", {}), "base_overrides")
        )

        if bool(stage_map.get("include_base", False)):
            expanded.append((stage_name, f"{stage_name}_base", stage_base))

        explicit = stage_map.get("configs", [])
        if not isinstance(explicit, list):
            raise ValueError(f"stage {stage_name}: configs must be a list")
        for item_raw in explicit:
            item = _require_mapping(item_raw, f"stage {stage_name} config")
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError(f"stage {stage_name}: config requires a name")
            overrides = dict(stage_base)
            overrides.update(
                _require_mapping(item.get("overrides", {}), f"config {name} overrides")
            )
            expanded.append((stage_name, name, overrides))

        one_factor = stage_map.get("one_factor", {})
        one_factor_map = _require_mapping(one_factor, "one_factor")
        for parameter, values in one_factor_map.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"stage {stage_name}: one_factor {parameter} must be a list"
                )
            for value in values:
                overrides = dict(stage_base)
                overrides[str(parameter)] = value
                expanded.append(
                    (
                        stage_name,
                        f"{stage_name}_{parameter}_{value}",
                        overrides,
                    )
                )

        product = stage_map.get("product", {})
        product_map = _require_mapping(product, "product")
        if product_map:
            combinations: list[dict[str, Any]] = [dict(stage_base)]
            for parameter, values in product_map.items():
                if not isinstance(values, list) or not values:
                    raise ValueError(
                        f"stage {stage_name}: product {parameter} must be a list"
                    )
                combinations = [
                    {**combination, str(parameter): value}
                    for combination in combinations
                    for value in values
                ]
                if len(combinations) > max_configs:
                    raise ValueError(
                        f"stage {stage_name}: product exceeds max_configs={max_configs}"
                    )
            for index, overrides in enumerate(combinations):
                expanded.append(
                    (stage_name, f"{stage_name}_product_{index:04d}", overrides)
                )

    unique: dict[str, EvaluationConfig] = {}
    config_names: set[tuple[str, str]] = set()
    for stage, name, overrides_raw in expanded:
        name_key = (stage, name)
        if name_key in config_names:
            raise ValueError(f"Duplicate config name in stage: {stage}/{name}")
        config_names.add(name_key)
        overrides = dict(overrides_raw)
        sample_stride = int(overrides.pop("sample_stride", base_sample_stride))
        if sample_stride < 1:
            raise ValueError(f"config {name}: sample_stride must be >= 1")
        sampling = replace(base_sampling, **overrides)
        if sampling.sampling_mode != "combined_v2":
            raise ValueError(f"config {name}: sampling_mode must remain combined_v2")
        payload = {
            "sampling": asdict(sampling),
            "sample_stride": sample_stride,
            "global_texture": fixed.get("global_texture", {}),
        }
        config_id = f"cfg_{_canonical_hash(payload)[:12]}"
        if config_id in unique:
            continue
        unique[config_id] = EvaluationConfig(
            config_id=config_id,
            config_name=name,
            stage=stage,
            sampling=sampling,
            sample_stride=sample_stride,
        )

    configs = list(unique.values())
    if not configs:
        raise ValueError("Search config expands to zero unique configs")
    if len(configs) > max_configs:
        raise ValueError(
            f"Search config expands to {len(configs)} configs; max={max_configs}"
        )
    return configs, dict(fixed)


def _make_global_texture_config(fixed: Mapping[str, Any]) -> AlphaTextureConfig:
    raw = _require_mapping(fixed.get("global_texture", {}), "global_texture")
    allowed = {field.name for field in fields(AlphaTextureConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown AlphaTextureConfig fields: {unknown}")
    return AlphaTextureConfig(**dict(raw))


def _load_search_config(
    path: Path,
    *,
    max_configs: int,
) -> tuple[dict[str, Any], list[EvaluationConfig], dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Search config root must be a mapping")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported search config schema_version")
    configs, fixed = _expand_search_configs(raw, max_configs=max_configs)
    return dict(raw), configs, fixed


def _filter_evaluation_configs(
    configs: list[EvaluationConfig],
    *,
    stages: set[str] | None = None,
    config_names: set[str] | None = None,
) -> list[EvaluationConfig]:
    selected = list(configs)
    if stages:
        available_stages = {config.stage for config in configs}
        unknown = sorted(stages - available_stages)
        if unknown:
            raise ValueError(
                f"Unknown --stages values: {unknown}; "
                f"available={sorted(available_stages)}"
            )
        selected = [config for config in selected if config.stage in stages]
    if config_names:
        available_names = {config.config_name for config in selected}
        unknown = sorted(config_names - available_names)
        if unknown:
            raise ValueError(
                f"Unknown --config_names values after stage filtering: {unknown}; "
                f"available={sorted(available_names)}"
            )
        selected = [
            config for config in selected if config.config_name in config_names
        ]
    if not selected:
        raise ValueError("Stage/config filters selected zero evaluation configs")
    return selected


def _load_pseudo3d_for_sweep(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        local_images = handle["local_encoder_images"][:]
        frame_indices = handle["frame_indices"][:].astype(np.int64)
        attrs = dict(handle.attrs)
    if int(local_images.shape[0]) != int(frame_indices.size):
        raise ValueError(f"Frame count mismatch in {path}")
    return local_images, frame_indices, attrs


def _percentile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _aggregate_bbox_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["teacher_status"] == "teacher_valid"]
    invalid = [row for row in rows if row["teacher_status"] == "teacher_invalid"]
    count = len(valid)
    recalls = [float(row["positive_pixel_recall"]) for row in valid]
    teacher_positive = sum(int(row["teacher_positive_count"]) for row in valid)
    selected_positive = sum(int(row["selected_positive_count"]) for row in valid)
    result: dict[str, Any] = {
        "teacher_valid_bbox_count": count,
        "teacher_invalid_bbox_count": len(invalid),
        "zero_positive_bbox_count": sum(bool(row["zero_positive"]) for row in valid),
        "zero_positive_bbox_rate": (
            sum(bool(row["zero_positive"]) for row in valid) / count
            if count
            else float("nan")
        ),
        "positive_pixel_recall_micro": (
            selected_positive / teacher_positive
            if teacher_positive
            else float("nan")
        ),
        "positive_pixel_recall_mean": _mean(recalls),
        "positive_pixel_recall_median": _percentile(recalls, 50),
        "positive_pixel_recall_p10": _percentile(recalls, 10),
        "positive_pixel_recall_p05": _percentile(recalls, 5),
        "positive_pixel_recall_min": min(recalls) if recalls else float("nan"),
    }
    for threshold in (1, 8, 16, 32):
        hits = sum(
            int(row["selected_positive_count"]) >= threshold for row in valid
        )
        result[f"positive_bbox_hits_at_{threshold}"] = hits
        result[f"positive_bbox_recall_at_{threshold}"] = (
            hits / count if count else float("nan")
        )
    extent = [float(row["positive_extent_recall"]) for row in valid]
    result["positive_extent_recall_mean"] = _mean(extent)
    result["positive_extent_recall_p05"] = _percentile(extent, 5)
    return result


def _aggregate_frame_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = [int(row["selected_total_count"]) for row in rows]
    total = sum(points)
    legacy_total = sum(int(row["legacy_point_count"]) for row in rows)
    positive = sum(int(row["selected_positive_count"]) for row in rows)
    background = sum(int(row["selected_background_count"]) for row in rows)
    ignore = sum(int(row["selected_ignore_count"]) for row in rows)
    result: dict[str, Any] = {
        "num_frames": len(rows),
        "total_selected_points": total,
        "points_per_frame_mean": _mean(points),
        "points_per_frame_median": _percentile(points, 50),
        "points_per_frame_p95": _percentile(points, 95),
        "points_per_frame_max": max(points) if points else 0,
        "legacy_total_points": legacy_total,
        "legacy_point_count_multiplier": (
            total / legacy_total if legacy_total else float("nan")
        ),
        "legacy_zero_point_frames": sum(
            int(row["legacy_point_count"]) == 0 for row in rows
        ),
        "selected_positive_count": positive,
        "selected_background_count": background,
        "selected_ignore_count": ignore,
        "selected_positive_ratio": positive / total if total else 0.0,
        "selected_background_ratio": background / total if total else 0.0,
        "selected_ignore_ratio": ignore / total if total else 0.0,
        "useful_point_ratio": (positive + background) / total if total else 0.0,
        "sampling_time_seconds": sum(float(row["sampling_time_seconds"]) for row in rows),
        "sampling_time_per_frame_seconds": _mean(
            float(row["sampling_time_seconds"]) for row in rows
        ),
    }
    source_keys = [
        key
        for key in rows[0]
        if key.startswith("source_") or key.startswith("context_only_")
    ] if rows else []
    for key in source_keys:
        result[key] = sum(int(row[key]) for row in rows)
    context_total = int(result.get("context_only_selected", 0))
    result["context_background_yield"] = (
        int(result.get("context_only_background", 0)) / context_total
        if context_total
        else float("nan")
    )
    result["context_ignore_yield"] = (
        int(result.get("context_only_ignore", 0)) / context_total
        if context_total
        else float("nan")
    )
    return result


def _config_prefix(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "config_name": config.config_name,
        "stage": config.stage,
    }


def _teacher_quality_rows(
    item: Stage4SweepManifestItem,
    teacher: DenseTeacherVideo,
) -> list[dict[str, Any]]:
    rows = []
    for bbox in teacher.bboxes:
        rows.append(
            {
                "split": item.split,
                "video_name": item.video_name,
                "frame_order": bbox.frame_order,
                "frame_index": bbox.frame_index,
                "bbox_index": bbox.bbox_index,
                "object_index": bbox.object_index,
                "object_name": bbox.object_name,
                "xml_path": bbox.xml_path,
                "bbox_local_xyxy": json.dumps(bbox.local_xyxy),
                "teacher_status": bbox.teacher_status,
                "teacher_positive_count": bbox.teacher_positive_count,
                "annotation_reason": bbox.annotation_reason,
                "contour_area": bbox.contour_area,
                "binary_area": bbox.binary_area,
                "foreground_ratio_in_bbox": bbox.foreground_ratio_in_bbox,
            }
        )
    return rows


def _evaluate_video_config(
    *,
    item: Stage4SweepManifestItem,
    local_images: np.ndarray,
    teacher: DenseTeacherVideo,
    config: EvaluationConfig,
    texture_config: AlphaTextureConfig,
    min_alpha: int,
    min_intensity: int,
    original_foreground_mode: str,
    extent_min_elongation: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    frame_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    flags_by_order: list[np.ndarray] = []
    bboxes_by_order: dict[int, list[Any]] = defaultdict(list)
    for bbox in teacher.bboxes:
        bboxes_by_order[bbox.frame_order].append(bbox)

    for order in range(int(local_images.shape[0])):
        image = local_images[order]
        start = time.perf_counter()
        texture = make_texture_image(image, texture_config)
        global_mask, _ = build_global_evidence_mask(
            image,
            texture,
            min_alpha=min_alpha,
            min_intensity=min_intensity,
            original_foreground_mode=original_foreground_mode,
            sample_stride=config.sample_stride,
        )
        sampling = build_sampling_source_flags(
            image,
            global_mask,
            config.sampling,
        )
        elapsed = time.perf_counter() - start
        flags = np.asarray(sampling["source_flags"], dtype=np.uint16)
        flags_by_order.append(flags)
        selected = flags != 0
        counts = label_counts(selected, teacher.labels[order])
        sources = source_label_counts(flags, teacher.labels[order])
        frame_rows.append(
            {
                **_config_prefix(config),
                "split": item.split,
                "video_name": item.video_name,
                "frame_order": order,
                "frame_index": int(teacher.frame_indices[order]),
                "legacy_point_count": int(global_mask.sum()),
                "sampling_time_seconds": elapsed,
                **counts,
                **sources,
            }
        )

        for bbox in bboxes_by_order.get(order, []):
            bbox_counts = bbox_selected_counts(
                bbox,
                selected_mask=selected,
                labels=teacher.labels[order],
            )
            elongation, extent = positive_extent_metrics(
                bbox,
                selected_mask=selected,
                min_elongation=extent_min_elongation,
            )
            bbox_rows.append(
                {
                    **_config_prefix(config),
                    "split": item.split,
                    "video_name": item.video_name,
                    "frame_order": order,
                    "frame_index": int(teacher.frame_indices[order]),
                    "bbox_index": bbox.bbox_index,
                    "object_index": bbox.object_index,
                    "teacher_status": bbox.teacher_status,
                    "teacher_elongation": elongation,
                    "positive_extent_recall": extent,
                    **bbox_counts,
                }
            )

    ablation_rows: list[dict[str, Any]] = []
    full_bbox_selected = {
        (row["frame_order"], row["bbox_index"]): int(row["selected_positive_count"])
        for row in bbox_rows
    }
    for source_name in ("none", *SOURCE_BITS):
        ablation_name = "full" if source_name == "none" else f"without_{source_name}"
        total_points = 0
        valid_count = 0
        zero_count = 0
        hits = {threshold: 0 for threshold in (1, 8, 16, 32)}
        teacher_positive = 0
        selected_positive = 0
        marginal_rescued = 0
        for order, flags in enumerate(flags_by_order):
            selected = ablate_source(flags, source_name) != 0
            total_points += int(selected.sum())
            for bbox in bboxes_by_order.get(order, []):
                if not bbox.teacher_valid:
                    continue
                valid_count += 1
                selected_count = int(
                    np.count_nonzero(selected[bbox.positive_y, bbox.positive_x])
                )
                teacher_positive += bbox.teacher_positive_count
                selected_positive += selected_count
                zero_count += selected_count == 0
                for threshold in hits:
                    hits[threshold] += selected_count >= threshold
                full_count = full_bbox_selected[(order, bbox.bbox_index)]
                marginal_rescued += full_count > 0 and selected_count == 0
        ablation_rows.append(
            {
                **_config_prefix(config),
                "scope": "video",
                "split": item.split,
                "video_name": item.video_name,
                "ablation": ablation_name,
                "teacher_valid_bbox_count": valid_count,
                "zero_positive_bbox_count": zero_count,
                "teacher_positive_count": teacher_positive,
                "selected_positive_count": selected_positive,
                "total_selected_points": total_points,
                "marginal_rescued_bbox_count": marginal_rescued,
                **{
                    f"positive_bbox_hits_at_{threshold}": count
                    for threshold, count in hits.items()
                },
            }
        )
    return frame_rows, bbox_rows, ablation_rows


def _aggregate_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["config_id"]), str(row["ablation"]))].append(row)
    output = list(rows)
    for (_, ablation), group in grouped.items():
        first = group[0]
        valid = sum(int(row["teacher_valid_bbox_count"]) for row in group)
        zero = sum(int(row["zero_positive_bbox_count"]) for row in group)
        teacher_positive = sum(int(row["teacher_positive_count"]) for row in group)
        selected_positive = sum(int(row["selected_positive_count"]) for row in group)
        total = sum(int(row["total_selected_points"]) for row in group)
        global_row = {
            "config_id": first["config_id"],
            "config_name": first["config_name"],
            "stage": first["stage"],
            "scope": "global",
            "split": "all_selected",
            "video_name": "",
            "ablation": ablation,
            "teacher_valid_bbox_count": valid,
            "zero_positive_bbox_count": zero,
            "zero_positive_bbox_rate": zero / valid if valid else float("nan"),
            "teacher_positive_count": teacher_positive,
            "selected_positive_count": selected_positive,
            "positive_pixel_recall_micro": (
                selected_positive / teacher_positive
                if teacher_positive
                else float("nan")
            ),
            "total_selected_points": total,
            "marginal_rescued_bbox_count": sum(
                int(row["marginal_rescued_bbox_count"]) for row in group
            ),
        }
        for threshold in (1, 8, 16, 32):
            count = sum(int(row[f"positive_bbox_hits_at_{threshold}"]) for row in group)
            global_row[f"positive_bbox_hits_at_{threshold}"] = count
            global_row[f"positive_bbox_recall_at_{threshold}"] = (
                count / valid if valid else float("nan")
            )
        output.append(global_row)

    global_by_config: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in output:
        if row["scope"] == "global":
            global_by_config[str(row["config_id"])][str(row["ablation"])] = row
    for by_ablation in global_by_config.values():
        full = by_ablation.get("full")
        if full is None:
            continue
        for name, row in by_ablation.items():
            if name == "full":
                row["unique_added_points"] = 0
                row["marginal_rescued_per_1000_added_points"] = float("nan")
                continue
            added = int(full["total_selected_points"]) - int(row["total_selected_points"])
            if added < 0:
                raise AssertionError(
                    f"Source ablation increased selected points: {name}, delta={added}"
                )
            rescued = int(row["marginal_rescued_bbox_count"])
            row["unique_added_points"] = added
            row["marginal_rescued_per_1000_added_points"] = (
                rescued * 1000.0 / added if added > 0 else float("nan")
            )
    return output


def _pareto_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objectives = (
        ("zero_positive_bbox_rate", "min"),
        ("positive_bbox_recall_at_16", "max"),
        ("positive_pixel_recall_p05", "max"),
        ("selected_ignore_ratio", "min"),
        ("points_per_frame_mean", "min"),
    )

    def value(row: dict[str, Any], key: str, direction: str) -> float:
        raw = float(row[key])
        if np.isfinite(raw):
            return raw
        return float("inf") if direction == "min" else float("-inf")

    pareto = []
    for index, candidate in enumerate(summary_rows):
        dominated = False
        for other_index, other in enumerate(summary_rows):
            if index == other_index:
                continue
            no_worse = True
            strictly_better = False
            for key, direction in objectives:
                candidate_value = value(candidate, key, direction)
                other_value = value(other, key, direction)
                if direction == "min":
                    no_worse &= other_value <= candidate_value
                    strictly_better |= other_value < candidate_value
                else:
                    no_worse &= other_value >= candidate_value
                    strictly_better |= other_value > candidate_value
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append({**candidate, "pareto": True})
    return pareto


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage 4 sampling parameters in 2D against a dense, "
            "sampling-independent teacher and export Pareto-ready CSV files."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--search_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test", "smoke"],
        default=["smoke"],
    )
    parser.add_argument("--allow_test_split", action="store_true")
    parser.add_argument("--max_configs", type=int, default=100)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Run only the named stages from the expanded search config.",
    )
    parser.add_argument(
        "--config_names",
        nargs="+",
        default=None,
        help=(
            "Run only exact expanded config names, after optional --stages "
            "filtering."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_configs < 1:
        raise ValueError("max_configs must be >= 1")
    selected_splits = set(args.splits)
    if "test" in selected_splits and not args.allow_test_split:
        raise ValueError("Refusing to tune on test split without --allow_test_split")

    start_total = time.perf_counter()
    teacher_config = load_dense_teacher_config(args.teacher_config)
    manifest_items = [
        item
        for item in load_stage4_sweep_manifest(args.manifest)
        if item.enabled and item.split in selected_splits
    ]
    if not manifest_items:
        raise ValueError(f"No enabled manifest rows for splits={sorted(selected_splits)}")

    search_raw, configs, fixed = _load_search_config(
        args.search_config,
        max_configs=args.max_configs,
    )
    selected_stages = set(args.stages) if args.stages else None
    selected_config_names = set(args.config_names) if args.config_names else None
    configs = _filter_evaluation_configs(
        configs,
        stages=selected_stages,
        config_names=selected_config_names,
    )
    texture_config = _make_global_texture_config(fixed)
    min_alpha = int(fixed.get("min_alpha", 1))
    min_intensity = int(fixed.get("min_intensity", 1))
    original_foreground_mode = str(fixed.get("original_foreground_mode", "nonzero"))
    evaluation = _require_mapping(search_raw.get("evaluation", {}), "evaluation")
    extent_min_elongation = float(evaluation.get("extent_min_elongation", 2.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.search_config, args.output_dir / "search_space.yaml")

    all_teacher_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_bbox_rows: list[dict[str, Any]] = []
    all_ablation_rows: list[dict[str, Any]] = []
    teacher_generation_seconds = 0.0

    print("Stage 4 sampling parameter sweep")
    print(f"videos  : {len(manifest_items)}")
    print(f"configs : {len(configs)}")
    print(f"splits  : {sorted(selected_splits)}")
    print(
        "stages  : "
        f"{sorted(selected_stages) if selected_stages else 'all'}"
    )
    print(
        "names   : "
        f"{sorted(selected_config_names) if selected_config_names else 'all'}"
    )
    for video_index, item in enumerate(manifest_items, start=1):
        print(f"[{video_index}/{len(manifest_items)}] loading {item.video_name}")
        local_images, frame_indices, attrs = _load_pseudo3d_for_sweep(
            item.pseudo3d_h5
        )
        teacher_start = time.perf_counter()
        teacher = build_dense_teacher_video(
            video_name=item.video_name,
            local_images=local_images,
            frame_indices=frame_indices,
            pseudo3d_attrs=attrs,
            voc_xml_root=item.voc_xml_root,
            config=teacher_config,
        )
        teacher_generation_seconds += time.perf_counter() - teacher_start
        all_teacher_rows.extend(_teacher_quality_rows(item, teacher))

        for config_index, config in enumerate(configs, start=1):
            print(
                f"  [{config_index}/{len(configs)}] "
                f"{config.config_name} ({config.config_id})"
            )
            frame_rows, bbox_rows, ablation_rows = _evaluate_video_config(
                item=item,
                local_images=local_images,
                teacher=teacher,
                config=config,
                texture_config=texture_config,
                min_alpha=min_alpha,
                min_intensity=min_intensity,
                original_foreground_mode=original_foreground_mode,
                extent_min_elongation=extent_min_elongation,
            )
            all_frame_rows.extend(frame_rows)
            all_bbox_rows.extend(bbox_rows)
            all_ablation_rows.extend(ablation_rows)

    config_rows = []
    per_video_rows = []
    summary_rows = []
    teacher_valid_count = sum(
        row["teacher_status"] == "teacher_valid" for row in all_teacher_rows
    )
    if teacher_valid_count == 0:
        raise ValueError(
            "Selected manifest/splits contain no teacher-valid BBoxes; "
            "positive-retention comparison is not possible"
        )
    for config in configs:
        sampling_values = asdict(config.sampling)
        config_rows.append(
            {
                **_config_prefix(config),
                "sample_stride": config.sample_stride,
                **{f"sampling_{key}": value for key, value in sampling_values.items()},
            }
        )
        config_frames = [
            row for row in all_frame_rows if row["config_id"] == config.config_id
        ]
        config_bboxes = [
            row for row in all_bbox_rows if row["config_id"] == config.config_id
        ]
        config_video_rows = []
        for item in manifest_items:
            video_frames = [
                row for row in config_frames if row["video_name"] == item.video_name
            ]
            video_bboxes = [
                row for row in config_bboxes if row["video_name"] == item.video_name
            ]
            video_row = {
                **_config_prefix(config),
                "split": item.split,
                "video_name": item.video_name,
                **_aggregate_bbox_rows(video_bboxes),
                **_aggregate_frame_rows(video_frames),
            }
            config_video_rows.append(video_row)
            per_video_rows.append(video_row)
        summary_row = {
            **_config_prefix(config),
            "splits": ",".join(sorted(selected_splits)),
            "num_videos": len(manifest_items),
            **_aggregate_bbox_rows(config_bboxes),
            **_aggregate_frame_rows(config_frames),
        }
        for key in (
            "positive_bbox_recall_at_1",
            "positive_bbox_recall_at_16",
            "positive_pixel_recall_micro",
        ):
            video_values = [float(row[key]) for row in config_video_rows]
            summary_row[f"worst_video_{key}"] = (
                min(value for value in video_values if np.isfinite(value))
                if any(np.isfinite(value) for value in video_values)
                else float("nan")
            )
            summary_row[f"video_p05_{key}"] = _percentile(video_values, 5)
            summary_row[f"video_p10_{key}"] = _percentile(video_values, 10)
        summary_rows.append(summary_row)

    ablation_rows = _aggregate_ablation_rows(all_ablation_rows)
    pareto_rows = _pareto_rows(summary_rows)
    failure_rows = [
        row
        for row in all_bbox_rows
        if row["teacher_status"] == "teacher_valid" and bool(row["zero_positive"])
    ]

    _write_csv(args.output_dir / "configs.csv", config_rows)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "per_video.csv", per_video_rows)
    _write_csv(args.output_dir / "per_frame.csv", all_frame_rows)
    _write_csv(args.output_dir / "per_bbox.csv", all_bbox_rows)
    _write_csv(args.output_dir / "source_ablation.csv", ablation_rows)
    _write_csv(args.output_dir / "pareto_front.csv", pareto_rows)
    _write_csv(args.output_dir / "teacher_quality.csv", all_teacher_rows)
    _write_csv(args.output_dir / "failure_cases.csv", failure_rows)

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "teacher_config_path": str(args.teacher_config.resolve()),
        "teacher_config_fingerprint": teacher_config.fingerprint(),
        "search_config_path": str(args.search_config.resolve()),
        "search_config_sha256": file_sha256(args.search_config),
        "splits": sorted(selected_splits),
        "selected_stages": (
            sorted(selected_stages) if selected_stages else None
        ),
        "selected_config_names": (
            sorted(selected_config_names) if selected_config_names else None
        ),
        "num_videos": len(manifest_items),
        "num_configs": len(configs),
        "teacher_generation_seconds": teacher_generation_seconds,
        "total_runtime_seconds": time.perf_counter() - start_total,
        "bbox_rows": len(all_bbox_rows),
        "failure_rows": len(failure_rows),
        "pareto_configs": len(pareto_rows),
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print("\nSweep summary")
    print(f"  teacher BBoxes : {len(all_teacher_rows)}")
    print(f"  evaluated rows : {len(all_bbox_rows)}")
    print(f"  failures       : {len(failure_rows)}")
    print(f"  Pareto configs : {len(pareto_rows)}")
    print(f"  output_dir     : {args.output_dir}")
    print("Stage 4 sampling parameter sweep completed.")


if __name__ == "__main__":
    main()
