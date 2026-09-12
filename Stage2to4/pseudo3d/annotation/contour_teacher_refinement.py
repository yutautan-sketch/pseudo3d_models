from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from pseudo3d.annotation.annotate_pseudo3d_point_cloud import LocalBBox
from src.utils.alpha_texture_processing import image_to_uint8_gray


REFINEMENT_SCHEMA_VERSION = 3
ISSUE_CATEGORIES = {
    "overfilled",
    "border_contact",
    "ambiguous",
    "off_center",
    "invalid",
}
SUPPORTED_ADAPTIVE_METHODS = {"mean", "gaussian"}
SUPPORTED_MORPH_SHAPES = {"rect", "ellipse", "cross"}


@dataclass(frozen=True)
class CleanupVariant:
    name: str
    open_ksize: int
    close_ksize: int
    morph_shape: str
    min_component_area: int

    def validate(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("cleanup variant name must be non-empty without outer whitespace")
        if any(character in self.name for character in ("/", "\\", "\r", "\n", "\x00")):
            raise ValueError(f"unsafe cleanup variant name: {self.name!r}")
        for name, value in (
            ("open_ksize", self.open_ksize),
            ("close_ksize", self.close_ksize),
        ):
            if value < 0 or (value > 0 and value % 2 == 0):
                raise ValueError(f"cleanup {name} must be zero or a positive odd integer")
        if self.morph_shape not in SUPPORTED_MORPH_SHAPES:
            raise ValueError(
                f"cleanup morph_shape must be one of {sorted(SUPPORTED_MORPH_SHAPES)}"
            )
        if self.min_component_area < 0:
            raise ValueError("cleanup min_component_area must be >= 0")


@dataclass(frozen=True)
class TriggerConfig:
    max_selected_area_ratio: float
    max_foreground_area_ratio: float
    max_border_contact_ratio: float
    max_center_distance_norm: float
    min_baseline_score_margin: float


@dataclass(frozen=True)
class CandidateGenerationConfig:
    percentile_values: tuple[float, ...]
    otsu_enabled: bool
    adaptive_methods: tuple[str, ...]
    adaptive_block_sizes: tuple[int, ...]
    adaptive_c_values: tuple[float, ...]
    cleanup_variants: tuple[CleanupVariant, ...]


@dataclass(frozen=True)
class EligibilityConfig:
    min_contour_area: float
    min_filled_area: int
    min_area_ratio: float
    max_area_ratio: float
    max_center_distance_norm: float
    max_border_contact_ratio: float
    min_candidate_points: int


@dataclass(frozen=True)
class RankingConfig:
    preferred_area_ratio: float
    weights: Mapping[str, float]


@dataclass(frozen=True)
class DecisionConfig:
    production_thresholds_fixed: bool
    minimum_refine_score_gain: float
    minimum_score_margin: float
    minimum_stability_iou: float
    minimum_support_count: int
    equal_score_tolerance: float


@dataclass(frozen=True)
class ContourRefinementConfig:
    path: Path | None
    raw: Mapping[str, Any]
    schema_version: int
    config_name: str
    base_teacher_config_name: str
    base_teacher_fingerprint: str | None
    fingerprint: str
    trigger: TriggerConfig
    candidate_generation: CandidateGenerationConfig
    eligibility: EligibilityConfig
    ranking: RankingConfig
    decision: DecisionConfig

    def validate(self, *, require_production_thresholds: bool = False) -> None:
        if self.schema_version != REFINEMENT_SCHEMA_VERSION:
            raise ValueError(
                f"refinement schema_version must be {REFINEMENT_SCHEMA_VERSION}"
            )
        if not self.config_name or self.config_name != self.config_name.strip():
            raise ValueError("refinement config_name must not be empty")
        if not self.base_teacher_config_name or (
            self.base_teacher_config_name != self.base_teacher_config_name.strip()
        ):
            raise ValueError("base_teacher_config_name must not be empty")
        if self.base_teacher_fingerprint is not None and (
            len(self.base_teacher_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.base_teacher_fingerprint
            )
        ):
            raise ValueError(
                "base_teacher_fingerprint must be null or a lowercase SHA-256"
            )
        if require_production_thresholds and not self.decision.production_thresholds_fixed:
            raise ValueError(
                "Production thresholds are not fixed; use --candidate_generation_only"
            )

        finite_values = {
            "trigger.max_selected_area_ratio": self.trigger.max_selected_area_ratio,
            "trigger.max_foreground_area_ratio": self.trigger.max_foreground_area_ratio,
            "trigger.max_border_contact_ratio": self.trigger.max_border_contact_ratio,
            "trigger.max_center_distance_norm": self.trigger.max_center_distance_norm,
            "trigger.min_baseline_score_margin": self.trigger.min_baseline_score_margin,
            "eligibility.min_contour_area": self.eligibility.min_contour_area,
            "eligibility.min_area_ratio": self.eligibility.min_area_ratio,
            "eligibility.max_area_ratio": self.eligibility.max_area_ratio,
            "eligibility.max_center_distance_norm": self.eligibility.max_center_distance_norm,
            "eligibility.max_border_contact_ratio": self.eligibility.max_border_contact_ratio,
            "ranking.preferred_area_ratio": self.ranking.preferred_area_ratio,
            "decision.minimum_refine_score_gain": self.decision.minimum_refine_score_gain,
            "decision.minimum_score_margin": self.decision.minimum_score_margin,
            "decision.minimum_stability_iou": self.decision.minimum_stability_iou,
            "decision.equal_score_tolerance": self.decision.equal_score_tolerance,
        }
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name, value in (
            ("max_selected_area_ratio", self.trigger.max_selected_area_ratio),
            ("max_foreground_area_ratio", self.trigger.max_foreground_area_ratio),
            ("max_border_contact_ratio", self.trigger.max_border_contact_ratio),
            ("max_center_distance_norm", self.trigger.max_center_distance_norm),
            ("min_baseline_score_margin", self.trigger.min_baseline_score_margin),
            ("min_area_ratio", self.eligibility.min_area_ratio),
            ("max_area_ratio", self.eligibility.max_area_ratio),
            ("max_center_distance_norm", self.eligibility.max_center_distance_norm),
            ("max_border_contact_ratio", self.eligibility.max_border_contact_ratio),
            ("preferred_area_ratio", self.ranking.preferred_area_ratio),
            ("minimum_refine_score_gain", self.decision.minimum_refine_score_gain),
            ("minimum_score_margin", self.decision.minimum_score_margin),
            ("minimum_stability_iou", self.decision.minimum_stability_iou),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.eligibility.min_contour_area < 0.0:
            raise ValueError("eligibility min_contour_area must be >= 0")
        if self.eligibility.min_filled_area < 0:
            raise ValueError("eligibility min_filled_area must be >= 0")
        if self.eligibility.min_candidate_points < 0:
            raise ValueError("eligibility min_candidate_points must be >= 0")
        if self.eligibility.max_area_ratio < self.eligibility.min_area_ratio:
            raise ValueError("eligibility max_area_ratio must be >= min_area_ratio")
        if not (
            self.eligibility.min_area_ratio
            <= self.ranking.preferred_area_ratio
            <= self.eligibility.max_area_ratio
        ):
            raise ValueError("preferred_area_ratio must lie inside eligibility area range")
        if self.decision.minimum_support_count <= 0:
            raise ValueError("decision minimum_support_count must be > 0")
        if self.decision.equal_score_tolerance < 0.0:
            raise ValueError("decision equal_score_tolerance must be >= 0")

        generation = self.candidate_generation
        if not generation.percentile_values and not generation.otsu_enabled and not generation.adaptive_methods:
            raise ValueError("at least one refine candidate method must be enabled")
        if tuple(sorted(set(generation.percentile_values))) != generation.percentile_values:
            raise ValueError("percentile_values must be unique and sorted")
        if any(not 0.0 <= value <= 100.0 for value in generation.percentile_values):
            raise ValueError("percentile_values must be in [0, 100]")
        if any(method not in SUPPORTED_ADAPTIVE_METHODS for method in generation.adaptive_methods):
            raise ValueError(
                f"adaptive_methods must be a subset of {sorted(SUPPORTED_ADAPTIVE_METHODS)}"
            )
        if len(set(generation.adaptive_methods)) != len(generation.adaptive_methods):
            raise ValueError("adaptive_methods must not contain duplicates")
        if generation.adaptive_methods and (
            not generation.adaptive_block_sizes or not generation.adaptive_c_values
        ):
            raise ValueError(
                "enabled adaptive_methods require block sizes and C values"
            )
        for block_size in generation.adaptive_block_sizes:
            if block_size <= 1 or block_size % 2 == 0:
                raise ValueError("adaptive_block_sizes must contain odd integers > 1")
        if tuple(sorted(set(generation.adaptive_block_sizes))) != generation.adaptive_block_sizes:
            raise ValueError("adaptive_block_sizes must be unique and sorted")
        if any(not math.isfinite(value) for value in generation.adaptive_c_values):
            raise ValueError("adaptive_c_values must be finite")
        if tuple(sorted(set(generation.adaptive_c_values))) != generation.adaptive_c_values:
            raise ValueError("adaptive_c_values must be unique and sorted")
        if len(set(generation.cleanup_variants)) != len(generation.cleanup_variants):
            raise ValueError("cleanup_variants must not contain duplicates")
        names = [variant.name for variant in generation.cleanup_variants]
        if len(names) != len(set(names)):
            raise ValueError("cleanup variant names must be unique")
        if not generation.cleanup_variants:
            raise ValueError("cleanup_variants must not be empty")
        for variant in generation.cleanup_variants:
            variant.validate()

        required_weights = {
            "center",
            "area",
            "border",
            "solidity",
            "compactness",
            "extent",
            "stability",
            "support",
        }
        if set(self.ranking.weights) != required_weights:
            raise ValueError(
                "ranking weights must contain exactly " + ", ".join(sorted(required_weights))
            )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.ranking.weights.values()
        ):
            raise ValueError("ranking weights must be finite and >= 0")
        if sum(float(value) for value in self.ranking.weights.values()) <= 0.0:
            raise ValueError("ranking weights must have a positive sum")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"refinement config '{name}' must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"refinement config '{name}' must be a sequence")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"refinement config '{name}' must be true or false")
    return value


def _strict_keys(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown refinement config keys in {name}: {unknown}")


def _canonical_fingerprint(raw: Mapping[str, Any]) -> str:
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_contour_refinement_config(path: Path) -> ContourRefinementConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Refinement config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("Refinement config root must be a mapping")
    raw = dict(loaded)
    _strict_keys(
        raw,
        {
            "schema_version",
            "config_name",
            "base_teacher_config_name",
            "base_teacher_fingerprint",
            "trigger",
            "candidate_generation",
            "eligibility",
            "ranking",
            "decision",
        },
        "root",
    )
    trigger_raw = _mapping(raw.get("trigger"), "trigger")
    generation_raw = _mapping(raw.get("candidate_generation"), "candidate_generation")
    eligibility_raw = _mapping(raw.get("eligibility"), "eligibility")
    ranking_raw = _mapping(raw.get("ranking"), "ranking")
    weights_raw = _mapping(ranking_raw.get("weights"), "ranking.weights")
    decision_raw = _mapping(raw.get("decision"), "decision")
    _strict_keys(
        trigger_raw,
        {
            "max_selected_area_ratio",
            "max_foreground_area_ratio",
            "max_border_contact_ratio",
            "max_center_distance_norm",
            "min_baseline_score_margin",
        },
        "trigger",
    )
    _strict_keys(
        generation_raw,
        {
            "percentile_values",
            "otsu_enabled",
            "adaptive_methods",
            "adaptive_block_sizes",
            "adaptive_c_values",
            "cleanup_variants",
        },
        "candidate_generation",
    )
    _strict_keys(
        eligibility_raw,
        {
            "min_contour_area",
            "min_filled_area",
            "min_area_ratio",
            "max_area_ratio",
            "max_center_distance_norm",
            "max_border_contact_ratio",
            "min_candidate_points",
        },
        "eligibility",
    )
    _strict_keys(ranking_raw, {"preferred_area_ratio", "weights"}, "ranking")
    _strict_keys(
        decision_raw,
        {
            "production_thresholds_fixed",
            "minimum_refine_score_gain",
            "minimum_score_margin",
            "minimum_stability_iou",
            "minimum_support_count",
            "equal_score_tolerance",
        },
        "decision",
    )
    cleanup_variants = []
    for index, value in enumerate(
        _sequence(generation_raw.get("cleanup_variants"), "cleanup_variants")
    ):
        item = _mapping(value, f"cleanup_variants[{index}]")
        _strict_keys(
            item,
            {"name", "open_ksize", "close_ksize", "morph_shape", "min_component_area"},
            f"cleanup_variants[{index}]",
        )
        cleanup_variants.append(
            CleanupVariant(
                name=str(item.get("name", "")),
                open_ksize=int(item.get("open_ksize", 0)),
                close_ksize=int(item.get("close_ksize", 0)),
                morph_shape=str(item.get("morph_shape", "ellipse")),
                min_component_area=int(item.get("min_component_area", 0)),
            )
        )
    config = ContourRefinementConfig(
        path=path.resolve(),
        raw=raw,
        schema_version=int(raw.get("schema_version", 0)),
        config_name=str(raw.get("config_name", "")).strip(),
        base_teacher_config_name=str(raw.get("base_teacher_config_name", "")).strip(),
        base_teacher_fingerprint=(
            None
            if raw.get("base_teacher_fingerprint") is None
            else str(raw.get("base_teacher_fingerprint"))
        ),
        fingerprint=_canonical_fingerprint(raw),
        trigger=TriggerConfig(
            max_selected_area_ratio=float(trigger_raw.get("max_selected_area_ratio")),
            max_foreground_area_ratio=float(trigger_raw.get("max_foreground_area_ratio")),
            max_border_contact_ratio=float(trigger_raw.get("max_border_contact_ratio")),
            max_center_distance_norm=float(trigger_raw.get("max_center_distance_norm")),
            min_baseline_score_margin=float(trigger_raw.get("min_baseline_score_margin")),
        ),
        candidate_generation=CandidateGenerationConfig(
            percentile_values=tuple(
                float(value)
                for value in _sequence(
                    generation_raw.get("percentile_values"), "percentile_values"
                )
            ),
            otsu_enabled=_boolean(
                generation_raw.get("otsu_enabled", False),
                "candidate_generation.otsu_enabled",
            ),
            adaptive_methods=tuple(
                str(value)
                for value in _sequence(
                    generation_raw.get("adaptive_methods"), "adaptive_methods"
                )
            ),
            adaptive_block_sizes=tuple(
                int(value)
                for value in _sequence(
                    generation_raw.get("adaptive_block_sizes"), "adaptive_block_sizes"
                )
            ),
            adaptive_c_values=tuple(
                float(value)
                for value in _sequence(
                    generation_raw.get("adaptive_c_values"), "adaptive_c_values"
                )
            ),
            cleanup_variants=tuple(cleanup_variants),
        ),
        eligibility=EligibilityConfig(
            min_contour_area=float(eligibility_raw.get("min_contour_area")),
            min_filled_area=int(eligibility_raw.get("min_filled_area")),
            min_area_ratio=float(eligibility_raw.get("min_area_ratio")),
            max_area_ratio=float(eligibility_raw.get("max_area_ratio")),
            max_center_distance_norm=float(
                eligibility_raw.get("max_center_distance_norm")
            ),
            max_border_contact_ratio=float(
                eligibility_raw.get("max_border_contact_ratio")
            ),
            min_candidate_points=int(eligibility_raw.get("min_candidate_points")),
        ),
        ranking=RankingConfig(
            preferred_area_ratio=float(ranking_raw.get("preferred_area_ratio")),
            weights={str(key): float(value) for key, value in weights_raw.items()},
        ),
        decision=DecisionConfig(
            production_thresholds_fixed=_boolean(
                decision_raw.get("production_thresholds_fixed", False),
                "decision.production_thresholds_fixed",
            ),
            minimum_refine_score_gain=float(
                decision_raw.get("minimum_refine_score_gain")
            ),
            minimum_score_margin=float(decision_raw.get("minimum_score_margin")),
            minimum_stability_iou=float(
                decision_raw.get("minimum_stability_iou")
            ),
            minimum_support_count=int(decision_raw.get("minimum_support_count")),
            equal_score_tolerance=float(decision_raw.get("equal_score_tolerance")),
        ),
    )
    config.validate()
    return config


def bbox_roi_bounds(
    bbox: LocalBBox, image_shape_hw: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    if not bbox.valid:
        return None
    height, width = (int(value) for value in image_shape_hw)
    if height <= 0 or width <= 0:
        raise ValueError("image_shape_hw must be positive")
    x1, y1, x2, y2 = (float(value) for value in bbox.local_xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    left = int(math.floor(max(0.0, x1)))
    top = int(math.floor(max(0.0, y1)))
    right = int(math.ceil(min(float(width - 1), x2)))
    bottom = int(math.ceil(min(float(height - 1), y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def mask_sha256(mask: np.ndarray) -> str:
    mask_bool = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    digest = hashlib.sha256()
    digest.update(np.asarray(mask_bool.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(mask_bool, bitorder="little").tobytes())
    return digest.hexdigest()


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if first_bool.shape != second_bool.shape:
        raise ValueError("mask IoU requires identical shapes")
    union = int(np.sum(first_bool | second_bool))
    if union == 0:
        return math.nan
    return float(np.sum(first_bool & second_bool) / union)


def _morph_kernel(shape: str, size: int) -> np.ndarray:
    mapping = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(mapping[shape], (size, size))


def _cleanup_binary(binary: np.ndarray, variant: CleanupVariant) -> np.ndarray:
    cleaned = np.asarray(binary, dtype=np.uint8)
    if variant.open_ksize > 1:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            _morph_kernel(variant.morph_shape, variant.open_ksize),
        )
    if variant.close_ksize > 1:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            _morph_kernel(variant.morph_shape, variant.close_ksize),
        )
    cleaned = cleaned.astype(bool)
    if variant.min_component_area > 0 and np.any(cleaned):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned.astype(np.uint8), connectivity=8
        )
        keep = np.zeros(cleaned.shape, dtype=bool)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= variant.min_component_area:
                keep[labels == label] = True
        cleaned = keep
    return cleaned


def _threshold_binaries(
    crop: np.ndarray,
    config: CandidateGenerationConfig,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    outputs: list[tuple[np.ndarray, dict[str, Any]]] = []
    for percentile in config.percentile_values:
        threshold = float(np.percentile(crop, percentile))
        outputs.append(
            (
                crop >= threshold,
                {
                    "source": "refine_percentile",
                    "family": "percentile",
                    "threshold_method": "percentile",
                    "threshold_parameter": percentile,
                    "threshold_value": threshold,
                },
            )
        )
    if config.otsu_enabled:
        threshold, binary = cv2.threshold(
            crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        outputs.append(
            (
                binary.astype(bool),
                {
                    "source": "refine_otsu",
                    "family": "otsu",
                    "threshold_method": "otsu",
                    "threshold_parameter": threshold,
                    "threshold_value": threshold,
                },
            )
        )
    for method in config.adaptive_methods:
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if method == "mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        for block_size in config.adaptive_block_sizes:
            if block_size > min(crop.shape):
                continue
            for c_value in config.adaptive_c_values:
                binary = cv2.adaptiveThreshold(
                    crop,
                    255,
                    adaptive_method,
                    cv2.THRESH_BINARY,
                    block_size,
                    c_value,
                )
                outputs.append(
                    (
                        binary.astype(bool),
                        {
                            "source": f"refine_adaptive_{method}",
                            "family": f"adaptive_{method}_block{block_size}",
                            "threshold_method": f"adaptive_{method}",
                            "threshold_parameter": c_value,
                            "threshold_value": c_value,
                            "adaptive_block_size": block_size,
                        },
                    )
                )
    return outputs


def _points_in_mask(frame_xy: np.ndarray | None, mask: np.ndarray) -> int:
    if frame_xy is None:
        return -1
    xy = np.asarray(frame_xy)
    if xy.size == 0:
        return 0
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("frame_xy must have shape [N, 2]")
    rounded = np.rint(xy).astype(np.int64)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    if not np.any(valid):
        return 0
    selected = rounded[valid]
    return int(np.sum(mask[selected[:, 1], selected[:, 0]]))


def _candidate_metrics(
    mask: np.ndarray,
    bbox: LocalBBox,
    *,
    frame_xy: np.ndarray | None,
) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    bounds = bbox_roi_bounds(bbox, mask_bool.shape)
    if bounds is None:
        raise ValueError("candidate BBox is invalid")
    left, top, right, bottom = bounds
    roi = mask_bool[top : bottom + 1, left : right + 1].astype(np.uint8)
    bbox_area = int(roi.size)
    filled_area = int(roi.sum())
    if filled_area == 0:
        raise ValueError("candidate mask is empty")
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("candidate mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    x, y, width, height = cv2.boundingRect(contour)
    rect_area = int(width * height)
    moments = cv2.moments(roi, binaryImage=True)
    center_x = float(left + moments["m10"] / moments["m00"])
    center_y = float(top + moments["m01"] / moments["m00"])
    bbox_center_x = 0.5 * float(left + right)
    bbox_center_y = 0.5 * float(top + bottom)
    bbox_diagonal = float(math.hypot(right - left + 1, bottom - top + 1))
    center_distance = float(
        math.hypot(center_x - bbox_center_x, center_y - bbox_center_y)
        / bbox_diagonal
    )
    boundary = np.zeros_like(roi)
    cv2.drawContours(boundary, [contour], 0, color=1, thickness=1)
    edge = np.zeros_like(roi, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    boundary_bool = boundary.astype(bool)
    boundary_pixels = int(boundary_bool.sum())
    border_pixels = int(np.sum(boundary_bool & edge))
    return {
        "bbox_area": bbox_area,
        "filled_area": filled_area,
        "area_ratio": float(filled_area / bbox_area),
        "contour_area": contour_area,
        "perimeter": perimeter,
        "center_x": center_x,
        "center_y": center_y,
        "center_distance_norm": center_distance,
        "border_contact_pixels": border_pixels,
        "border_contact_ratio": (
            float(border_pixels / boundary_pixels) if boundary_pixels else math.nan
        ),
        "touch_top": bool(np.any(boundary_bool[0, :])),
        "touch_bottom": bool(np.any(boundary_bool[-1, :])),
        "touch_left": bool(np.any(boundary_bool[:, 0])),
        "touch_right": bool(np.any(boundary_bool[:, -1])),
        "solidity": float(contour_area / hull_area) if hull_area > 0.0 else math.nan,
        "extent": float(filled_area / rect_area) if rect_area > 0 else math.nan,
        "compactness": (
            float(4.0 * math.pi * contour_area / (perimeter * perimeter))
            if perimeter > 0.0
            else math.nan
        ),
        "aspect_ratio": float(width / height) if height > 0 else math.nan,
        "candidate_point_count": _points_in_mask(frame_xy, mask_bool),
    }


def _components_from_binary(
    binary_crop: np.ndarray,
    *,
    full_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
    provenance: Mapping[str, Any],
    bbox: LocalBBox,
    frame_xy: np.ndarray | None,
) -> list[dict[str, Any]]:
    left, top, right, bottom = bounds
    count, labels = cv2.connectedComponents(
        np.asarray(binary_crop, dtype=np.uint8), connectivity=8
    )
    candidates: list[dict[str, Any]] = []
    for component_label in range(1, count):
        component = (labels == component_label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        filled = np.zeros(component.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], 0, color=1, thickness=-1)
        full_mask = np.zeros(full_shape, dtype=bool)
        full_mask[top : bottom + 1, left : right + 1] = filled.astype(bool)
        item_provenance = dict(provenance)
        item_provenance["component_index"] = int(component_label - 1)
        item_provenance["parameter_key"] = json.dumps(
            item_provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        candidate = {
            "mask": full_mask,
            "mask_sha256": mask_sha256(full_mask),
            "provenance": [item_provenance],
            "sources": [str(item_provenance["source"])],
            "families": [str(item_provenance["family"])],
            "support_count": 1,
            "is_baseline_selected": False,
            "has_refine_source": True,
            **_candidate_metrics(full_mask, bbox, frame_xy=frame_xy),
        }
        candidates.append(candidate)
    return candidates


def build_refinement_candidates(
    *,
    image: np.ndarray,
    bbox: LocalBBox,
    baseline_candidates: Sequence[Mapping[str, Any]],
    baseline_result: Mapping[str, Any],
    config: ContourRefinementConfig,
    frame_xy: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    config.validate()
    gray = image_to_uint8_gray(image)
    bounds = bbox_roi_bounds(bbox, gray.shape)
    if bounds is None:
        return []
    left, top, right, bottom = bounds
    crop = np.ascontiguousarray(gray[top : bottom + 1, left : right + 1])
    raw_candidates: list[dict[str, Any]] = []

    baseline_valid = bool(baseline_result.get("valid", False))
    baseline_mask = (
        np.asarray(baseline_result["mask"], dtype=bool)
        if baseline_valid
        else np.zeros(gray.shape, dtype=bool)
    )
    for index, candidate_raw in enumerate(baseline_candidates):
        mask = np.asarray(candidate_raw["mask"], dtype=bool)
        if mask.shape != gray.shape or not np.any(mask):
            continue
        source = str(candidate_raw.get("source", "baseline"))
        provenance = {
            "source": source,
            "family": "baseline_v2",
            "threshold_method": "teacher_v2",
            "threshold_parameter": source,
            "threshold_value": math.nan,
            "component_index": int(candidate_raw.get("component_index", index)),
        }
        provenance["parameter_key"] = json.dumps(
            provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=True,
        )
        raw_candidates.append(
            {
                "mask": mask,
                "mask_sha256": mask_sha256(mask),
                "provenance": [provenance],
                "sources": [source],
                "families": ["baseline_v2"],
                "support_count": 1,
                "is_baseline_selected": bool(
                    baseline_valid and np.array_equal(mask, baseline_mask)
                ),
                "has_refine_source": False,
                "baseline_v2_score": float(candidate_raw.get("score", math.nan)),
                **_candidate_metrics(mask, bbox, frame_xy=frame_xy),
            }
        )

    for binary, threshold_provenance in _threshold_binaries(
        crop, config.candidate_generation
    ):
        for cleanup in config.candidate_generation.cleanup_variants:
            cleaned = _cleanup_binary(binary, cleanup)
            provenance = dict(threshold_provenance)
            provenance["cleanup_name"] = cleanup.name
            provenance["family"] = f"{provenance['family']}__{cleanup.name}"
            raw_candidates.extend(
                _components_from_binary(
                    cleaned,
                    full_shape=gray.shape,
                    bounds=bounds,
                    provenance=provenance,
                    bbox=bbox,
                    frame_xy=frame_xy,
                )
            )

    deduplicated: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates:
        key = str(candidate["mask_sha256"])
        if key not in deduplicated:
            deduplicated[key] = candidate
            continue
        existing = deduplicated[key]
        existing["provenance"].extend(candidate["provenance"])
        existing["sources"] = sorted(
            set(existing["sources"]) | set(candidate["sources"])
        )
        existing["families"] = sorted(
            set(existing["families"]) | set(candidate["families"])
        )
        existing["is_baseline_selected"] = bool(
            existing["is_baseline_selected"] or candidate["is_baseline_selected"]
        )
        existing["has_refine_source"] = bool(
            existing["has_refine_source"] or candidate["has_refine_source"]
        )
        if "baseline_v2_score" in candidate:
            current = float(existing.get("baseline_v2_score", -math.inf))
            existing["baseline_v2_score"] = max(
                current, float(candidate["baseline_v2_score"])
            )

    candidates = list(deduplicated.values())
    for candidate in candidates:
        candidate["provenance"] = sorted(
            candidate["provenance"],
            key=lambda value: str(value["parameter_key"]),
        )
        candidate["support_count"] = len(
            {str(value["parameter_key"]) for value in candidate["provenance"]}
        )
        candidate["primary_source"] = (
            str(baseline_result.get("selected_contour_source", "baseline_v2"))
            if candidate["is_baseline_selected"]
            else str(candidate["sources"][0])
        )
        candidate["is_refine_candidate"] = bool(
            candidate["has_refine_source"] and not candidate["is_baseline_selected"]
        )
        if baseline_valid:
            candidate["baseline_iou"] = mask_iou(candidate["mask"], baseline_mask)
            baseline_metrics = next(
                (
                    value
                    for value in candidates
                    if value.get("is_baseline_selected", False)
                ),
                None,
            )
            candidate["baseline_area_delta"] = (
                int(candidate["filled_area"] - baseline_metrics["filled_area"])
                if baseline_metrics is not None
                else 0
            )
            candidate["baseline_centroid_shift"] = (
                float(
                    math.hypot(
                        float(candidate["center_x"]) - float(baseline_metrics["center_x"]),
                        float(candidate["center_y"]) - float(baseline_metrics["center_y"]),
                    )
                )
                if baseline_metrics is not None
                else math.nan
            )
        else:
            candidate["baseline_iou"] = math.nan
            candidate["baseline_area_delta"] = int(candidate["filled_area"])
            candidate["baseline_centroid_shift"] = math.nan

    _add_stability_metrics(candidates)
    for candidate in candidates:
        _add_eligibility_and_score(candidate, config)
    candidates.sort(key=refinement_candidate_sort_key)
    return candidates


def _add_stability_metrics(candidates: Sequence[dict[str, Any]]) -> None:
    for candidate in candidates:
        own_keys_by_family: dict[str, set[str]] = {}
        for provenance in candidate["provenance"]:
            own_keys_by_family.setdefault(str(provenance["family"]), set()).add(
                str(provenance["parameter_key"])
            )
        # Exact masks produced by multiple parameters are perfect stability
        # observations even though mask-level deduplication has merged them.
        overlaps = [
            1.0
            for keys in own_keys_by_family.values()
            for _ in range(max(0, len(keys) - 1))
        ]
        for other in candidates:
            if other is candidate:
                continue
            other_by_family: dict[str, set[str]] = {}
            for provenance in other["provenance"]:
                other_by_family.setdefault(str(provenance["family"]), set()).add(
                    str(provenance["parameter_key"])
                )
            shared_families = set(own_keys_by_family) & set(other_by_family)
            if not any(
                own_keys_by_family[family] != other_by_family[family]
                for family in shared_families
            ):
                continue
            value = mask_iou(candidate["mask"], other["mask"])
            if math.isfinite(value):
                overlaps.append(value)
        candidate["stability_neighbor_count"] = len(overlaps)
        candidate["stability_iou_max"] = max(overlaps) if overlaps else 0.0
        candidate["stability_iou_median"] = (
            float(np.median(overlaps)) if overlaps else 0.0
        )


def _add_eligibility_and_score(
    candidate: dict[str, Any], config: ContourRefinementConfig
) -> None:
    eligibility = config.eligibility
    reasons = []
    finite_required = (
        "area_ratio",
        "contour_area",
        "center_distance_norm",
        "border_contact_ratio",
        "solidity",
        "extent",
        "compactness",
    )
    if any(not math.isfinite(float(candidate[name])) for name in finite_required):
        reasons.append("non_finite_metric")
    if float(candidate["contour_area"]) < eligibility.min_contour_area:
        reasons.append("contour_area_below_minimum")
    if int(candidate["filled_area"]) < eligibility.min_filled_area:
        reasons.append("filled_area_below_minimum")
    if float(candidate["area_ratio"]) < eligibility.min_area_ratio:
        reasons.append("area_ratio_below_minimum")
    if float(candidate["area_ratio"]) > eligibility.max_area_ratio:
        reasons.append("area_ratio_above_maximum")
    if float(candidate["center_distance_norm"]) > eligibility.max_center_distance_norm:
        reasons.append("center_distance_above_maximum")
    if float(candidate["border_contact_ratio"]) > eligibility.max_border_contact_ratio:
        reasons.append("border_contact_above_maximum")
    point_count = int(candidate["candidate_point_count"])
    if point_count >= 0 and point_count < eligibility.min_candidate_points:
        reasons.append("candidate_points_below_minimum")
    candidate["eligible"] = not reasons
    candidate["rejection_reasons"] = sorted(set(reasons))

    center_score = float(
        np.clip(
            1.0
            - float(candidate["center_distance_norm"])
            / max(eligibility.max_center_distance_norm, np.finfo(np.float64).eps),
            0.0,
            1.0,
        )
    )
    area_distance_scale = max(
        config.ranking.preferred_area_ratio - eligibility.min_area_ratio,
        eligibility.max_area_ratio - config.ranking.preferred_area_ratio,
        np.finfo(np.float64).eps,
    )
    area_score = float(
        np.clip(
            1.0
            - abs(float(candidate["area_ratio"]) - config.ranking.preferred_area_ratio)
            / area_distance_scale,
            0.0,
            1.0,
        )
    )
    score_parts = {
        "center": center_score,
        "area": area_score,
        "border": float(np.clip(1.0 - candidate["border_contact_ratio"], 0.0, 1.0)),
        "solidity": float(np.clip(candidate["solidity"], 0.0, 1.0)),
        "compactness": float(np.clip(candidate["compactness"], 0.0, 1.0)),
        "extent": float(np.clip(candidate["extent"], 0.0, 1.0)),
        "stability": float(np.clip(candidate["stability_iou_max"], 0.0, 1.0)),
        "support": float(
            np.clip(
                candidate["support_count"] / config.decision.minimum_support_count,
                0.0,
                1.0,
            )
        ),
    }
    weight_sum = sum(float(value) for value in config.ranking.weights.values())
    candidate["score_parts"] = score_parts
    candidate["score"] = float(
        sum(
            float(config.ranking.weights[name]) * score_parts[name]
            for name in score_parts
        )
        / weight_sum
    )


def refinement_candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(candidate["eligible"]),
        -float(candidate["score"]),
        float(candidate["center_distance_norm"]),
        abs(float(candidate["area_ratio"])),
        str(candidate["mask_sha256"]),
    )


def _baseline_score_margin(
    baseline_candidates: Sequence[Mapping[str, Any]],
) -> float:
    scores = sorted(
        (
            float(candidate["score"])
            for candidate in baseline_candidates
            if bool(candidate.get("eligible", False))
            and math.isfinite(float(candidate.get("score", math.nan)))
        ),
        reverse=True,
    )
    return float(scores[0] - scores[1]) if len(scores) >= 2 else math.inf


def evaluate_bbox_refinement(
    *,
    image: np.ndarray,
    bbox: LocalBBox,
    baseline_candidates: Sequence[Mapping[str, Any]],
    baseline_result: Mapping[str, Any],
    config: ContourRefinementConfig,
    frame_xy: np.ndarray | None = None,
    audit_categories: Sequence[str] = (),
    candidate_generation_only: bool = False,
) -> dict[str, Any]:
    config.validate(require_production_thresholds=not candidate_generation_only)
    candidates = build_refinement_candidates(
        image=image,
        bbox=bbox,
        baseline_candidates=baseline_candidates,
        baseline_result=baseline_result,
        config=config,
        frame_xy=frame_xy,
    )
    baseline_candidate = next(
        (candidate for candidate in candidates if candidate["is_baseline_selected"]),
        None,
    )
    trigger_reasons = []
    if not bool(baseline_result.get("valid", False)):
        trigger_reasons.append("baseline_invalid")
    if baseline_candidate is not None:
        if float(baseline_candidate["area_ratio"]) >= config.trigger.max_selected_area_ratio:
            trigger_reasons.append("selected_area_ratio_high")
        if (
            float(baseline_result.get("foreground_ratio_in_bbox", 0.0))
            >= config.trigger.max_foreground_area_ratio
        ):
            trigger_reasons.append("foreground_area_ratio_high")
        if (
            float(baseline_candidate["border_contact_ratio"])
            >= config.trigger.max_border_contact_ratio
        ):
            trigger_reasons.append("border_contact_high")
        if (
            float(baseline_candidate["center_distance_norm"])
            >= config.trigger.max_center_distance_norm
        ):
            trigger_reasons.append("center_distance_high")
    baseline_margin = _baseline_score_margin(baseline_candidates)
    if baseline_margin <= config.trigger.min_baseline_score_margin:
        trigger_reasons.append("baseline_score_margin_low")
    for category in sorted(set(str(value) for value in audit_categories) & ISSUE_CATEGORIES):
        trigger_reasons.append(f"phase1_category_{category}")
    trigger_reasons = sorted(set(trigger_reasons))

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    eligible.sort(key=refinement_candidate_sort_key)
    selected = eligible[0] if eligible else None
    second = eligible[1] if len(eligible) > 1 else None
    score_margin = (
        float(selected["score"] - second["score"])
        if selected is not None and second is not None
        else math.inf
    )
    score_gain = (
        float(selected["score"] - baseline_candidate["score"])
        if selected is not None and baseline_candidate is not None
        else math.inf
        if selected is not None
        else math.nan
    )
    ambiguous = bool(
        selected is not None
        and second is not None
        and selected["mask_sha256"] != second["mask_sha256"]
        and score_margin <= config.decision.equal_score_tolerance
    )

    if candidate_generation_only:
        decision = "candidate_generation_only"
        reason_codes = tuple(trigger_reasons or ["candidate_distribution_requested"])
    elif selected is None:
        decision = "manual_review"
        reason_codes = ("no_eligible_candidate",)
    elif ambiguous:
        decision = "manual_review"
        reason_codes = ("ambiguous_equal_score_different_masks",)
    elif not trigger_reasons and baseline_candidate is not None and baseline_candidate["eligible"]:
        selected = baseline_candidate
        decision = "auto_accept"
        reason_codes = ("baseline_quality_gate_passed",)
        score_gain = 0.0
    elif selected["is_baseline_selected"]:
        decision = "auto_accept"
        reason_codes = tuple(trigger_reasons + ["baseline_remains_best_candidate"])
        score_gain = 0.0
    elif not bool(selected["is_refine_candidate"]):
        decision = "manual_review"
        reason_codes = tuple(trigger_reasons + ["selected_candidate_is_not_refineable"])
    elif score_gain < config.decision.minimum_refine_score_gain:
        decision = "manual_review"
        reason_codes = tuple(trigger_reasons + ["refine_score_gain_below_minimum"])
    elif score_margin < config.decision.minimum_score_margin:
        decision = "manual_review"
        reason_codes = tuple(trigger_reasons + ["refine_score_margin_below_minimum"])
    elif float(selected["stability_iou_max"]) < config.decision.minimum_stability_iou:
        decision = "manual_review"
        reason_codes = tuple(trigger_reasons + ["refine_stability_below_minimum"])
    elif int(selected["support_count"]) < config.decision.minimum_support_count:
        decision = "manual_review"
        reason_codes = tuple(trigger_reasons + ["refine_support_below_minimum"])
    else:
        decision = "auto_refine"
        reason_codes = tuple(trigger_reasons + ["refine_candidate_improves_baseline"])

    proposal_mask = (
        np.asarray(selected["mask"], dtype=bool)
        if selected is not None
        else np.zeros(image_to_uint8_gray(image).shape, dtype=bool)
    )
    return {
        "proposed_decision": decision,
        "reason_codes": tuple(sorted(set(reason_codes))),
        "triggered": bool(trigger_reasons),
        "trigger_reasons": tuple(trigger_reasons),
        "baseline_score_margin": baseline_margin,
        "score_margin": score_margin,
        "score_gain": score_gain,
        "baseline_candidate": baseline_candidate,
        "selected_candidate": selected,
        "proposal_mask": proposal_mask,
        "candidates": candidates,
        "num_candidates": len(candidates),
        "num_eligible_candidates": len(eligible),
    }


def candidate_to_serializable(candidate: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"mask"}
    result = {key: value for key, value in candidate.items() if key not in excluded}
    result["provenance"] = json.dumps(
        result["provenance"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=True,
    )
    result["sources"] = json.dumps(result["sources"], ensure_ascii=False)
    result["families"] = json.dumps(result["families"], ensure_ascii=False)
    result["rejection_reasons"] = json.dumps(
        result["rejection_reasons"], ensure_ascii=False
    )
    score_parts = result.pop("score_parts")
    for name, value in score_parts.items():
        result[f"score_{name}"] = value
    return result
