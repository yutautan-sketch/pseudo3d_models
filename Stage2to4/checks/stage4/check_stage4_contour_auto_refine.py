from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from checks.stage4.check_stage4_contour_teacher_audit import (
    _args as phase1_args,
    _make_integration_fixture,
)
from pseudo3d.analysis.audit_stage4_contour_teacher import (
    load_audit_teacher_config,
    run_audit,
)
from pseudo3d.analysis.prototype_stage4_contour_auto_refine import run_prototype
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import LocalBBox
from pseudo3d.annotation.contour_teacher_refinement import (
    build_refinement_candidates,
    evaluate_bbox_refinement,
    load_contour_refinement_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(xyxy: tuple[int, int, int, int]) -> LocalBBox:
    values = np.asarray(xyxy, dtype=np.float32)
    return LocalBBox(values.copy(), values.copy(), values.copy(), True)


def _rectangle_mask(
    shape: tuple[int, int], xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = xyxy
    result[y1 : y2 + 1, x1 : x2 + 1] = True
    return result


def _config_mapping(
    base_teacher_fingerprint: str,
    *,
    base_teacher_config_name: str = "synthetic_bbox_ranked_teacher_v2",
    production_thresholds_fixed: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "config_name": "synthetic_contour_auto_refine_v1",
        "base_teacher_config_name": base_teacher_config_name,
        "base_teacher_fingerprint": base_teacher_fingerprint,
        "trigger": {
            "max_selected_area_ratio": 0.80,
            "max_foreground_area_ratio": 0.85,
            "max_border_contact_ratio": 0.50,
            "max_center_distance_norm": 0.35,
            "min_baseline_score_margin": 0.01,
        },
        "candidate_generation": {
            "percentile_values": [70.0, 85.0, 90.0, 95.0],
            "otsu_enabled": True,
            "adaptive_methods": [],
            "adaptive_block_sizes": [],
            "adaptive_c_values": [],
            "cleanup_variants": [
                {
                    "name": "none",
                    "open_ksize": 0,
                    "close_ksize": 0,
                    "morph_shape": "ellipse",
                    "min_component_area": 0,
                }
            ],
        },
        "eligibility": {
            "min_contour_area": 10.0,
            "min_filled_area": 20,
            "min_area_ratio": 0.02,
            "max_area_ratio": 0.70,
            "max_center_distance_norm": 0.45,
            "max_border_contact_ratio": 0.50,
            "min_candidate_points": 0,
        },
        "ranking": {
            "preferred_area_ratio": 0.15,
            "weights": {
                "center": 0.25,
                "area": 0.20,
                "border": 0.15,
                "solidity": 0.05,
                "compactness": 0.05,
                "extent": 0.05,
                "stability": 0.15,
                "support": 0.10,
            },
        },
        "decision": {
            "production_thresholds_fixed": production_thresholds_fixed,
            "minimum_refine_score_gain": 0.05,
            "minimum_score_margin": 0.02,
            "minimum_stability_iou": 0.50,
            "minimum_support_count": 2,
            "equal_score_tolerance": 1.0e-12,
        },
    }


def _write_config(
    path: Path,
    fingerprint: str = "0" * 64,
    *,
    production_thresholds_fixed: bool = True,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            _config_mapping(
                fingerprint,
                production_thresholds_fixed=production_thresholds_fixed,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _baseline(
    mask: np.ndarray | None,
    *,
    foreground_ratio: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mask is None:
        return [], {
            "valid": False,
            "reason": "no_eligible_ranked_contour",
            "mask": np.zeros((100, 100), dtype=bool),
            "selected_contour_source": "none",
            "foreground_ratio_in_bbox": foreground_ratio,
        }
    candidates = [
        {
            "source": "global",
            "component_index": 0,
            "mask": mask,
            "eligible": True,
            "score": 0.75,
        }
    ]
    result = {
        "valid": True,
        "reason": "bbox_ranked_selected_global",
        "mask": mask,
        "selected_contour_source": "global",
        "foreground_ratio_in_bbox": foreground_ratio,
    }
    return candidates, result


def test_auto_accept_and_refine(config_path: Path) -> None:
    config = load_contour_refinement_config(config_path)
    bbox = _bbox((10, 10, 90, 90))
    shape = (100, 100)
    center = _rectangle_mask(shape, (35, 35, 65, 65))
    image = np.full(shape, 40, dtype=np.uint8)
    image[center] = 230

    baseline_candidates, baseline_result = _baseline(center, foreground_ratio=0.15)
    accepted = evaluate_bbox_refinement(
        image=image,
        bbox=bbox,
        baseline_candidates=baseline_candidates,
        baseline_result=baseline_result,
        config=config,
    )
    assert accepted["proposed_decision"] == "auto_accept"
    np.testing.assert_array_equal(accepted["proposal_mask"], center)

    overfilled = _rectangle_mask(shape, (10, 10, 90, 90))
    baseline_candidates, baseline_result = _baseline(
        overfilled, foreground_ratio=1.0
    )
    refined = evaluate_bbox_refinement(
        image=image,
        bbox=bbox,
        baseline_candidates=baseline_candidates,
        baseline_result=baseline_result,
        config=config,
        audit_categories=("overfilled",),
    )
    assert refined["proposed_decision"] == "auto_refine", refined["reason_codes"]
    np.testing.assert_array_equal(refined["proposal_mask"], center)
    assert refined["selected_candidate"]["support_count"] >= 2
    assert refined["selected_candidate"]["stability_iou_max"] == 1.0
    assert refined["selected_candidate"]["area_ratio"] < 0.30
    assert refined["score_gain"] >= config.decision.minimum_refine_score_gain
    print("[OK] good-control auto-accept and stable overfilled auto-refine")


def test_rejection_and_ambiguity(config_path: Path) -> None:
    config = load_contour_refinement_config(config_path)
    bbox = _bbox((10, 10, 90, 90))
    shape = (100, 100)

    tiny_image = np.full(shape, 40, dtype=np.uint8)
    tiny_image[49:52, 49:52] = 240
    empty_candidates, empty_result = _baseline(None)
    tiny = evaluate_bbox_refinement(
        image=tiny_image,
        bbox=bbox,
        baseline_candidates=empty_candidates,
        baseline_result=empty_result,
        config=config,
        audit_categories=("invalid",),
    )
    assert tiny["proposed_decision"] == "manual_review"
    assert tiny["num_eligible_candidates"] == 0
    assert "no_eligible_candidate" in tiny["reason_codes"]

    symmetric = np.full(shape, 40, dtype=np.uint8)
    first = _rectangle_mask(shape, (25, 40, 39, 54))
    second = _rectangle_mask(shape, (61, 40, 75, 54))
    symmetric[first | second] = 230
    ambiguous = evaluate_bbox_refinement(
        image=symmetric,
        bbox=bbox,
        baseline_candidates=empty_candidates,
        baseline_result=empty_result,
        config=config,
        audit_categories=("invalid",),
    )
    assert ambiguous["proposed_decision"] == "manual_review"
    assert "ambiguous_equal_score_different_masks" in ambiguous["reason_codes"]
    assert ambiguous["num_eligible_candidates"] >= 2
    print("[OK] tiny-component rejection and equal-score ambiguity isolation")


def test_deduplication_and_determinism(config_path: Path) -> None:
    config = load_contour_refinement_config(config_path)
    bbox = _bbox((10, 10, 90, 90))
    image = np.full((100, 100), 40, dtype=np.uint8)
    image[35:66, 35:66] = 230
    first = build_refinement_candidates(
        image=image,
        bbox=bbox,
        baseline_candidates=[],
        baseline_result={
            "valid": False,
            "mask": np.zeros((100, 100), dtype=bool),
            "selected_contour_source": "none",
        },
        config=config,
    )
    second = build_refinement_candidates(
        image=image.copy(),
        bbox=bbox,
        baseline_candidates=[],
        baseline_result={
            "valid": False,
            "mask": np.zeros((100, 100), dtype=bool),
            "selected_contour_source": "none",
        },
        config=config,
    )
    assert [item["mask_sha256"] for item in first] == [
        item["mask_sha256"] for item in second
    ]
    assert [item["score"] for item in first] == [item["score"] for item in second]
    assert any(item["support_count"] >= 2 for item in first)
    assert len({item["mask_sha256"] for item in first}) == len(first)
    print("[OK] mask-level candidate deduplication and deterministic ranking")


def _phase3_args(
    paths: dict[str, Path],
    phase1_output: Path,
    refine_config: Path,
    output: Path,
    *,
    candidate_generation_only: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=paths["manifest"],
        annotated_root=paths["annotated_root"],
        phase1_audit_root=phase1_output,
        teacher_config=paths["teacher"],
        refine_config=refine_config,
        output_root=output,
        selection_csv=None,
        annotated_glob_template="{video_name}/*bboxrank_v2_nobbox_bg.h5",
        expected_videos=1,
        top_per_category=2,
        candidate_generation_only=candidate_generation_only,
        no_overlays=False,
        continue_on_error=False,
        overwrite=False,
    )


def test_end_to_end_read_only_prototype(root: Path) -> None:
    fixture_root = root / "integration"
    fixture_root.mkdir()
    paths = _make_integration_fixture(fixture_root)
    phase1_output = fixture_root / "phase1"
    run_audit(phase1_args(paths, phase1_output))
    teacher = load_audit_teacher_config(paths["teacher"])
    refine_path = _write_config(
        fixture_root / "refine.yaml",
        teacher.fingerprint,
        production_thresholds_fixed=False,
    )
    input_hashes = {
        name: _sha256(paths[name]) for name in ("pseudo", "xml", "annotated")
    }
    output_1 = fixture_root / "phase3_1"
    output_2 = fixture_root / "phase3_2"
    result_1 = run_prototype(
        _phase3_args(paths, phase1_output, refine_path, output_1)
    )
    result_2 = run_prototype(
        _phase3_args(paths, phase1_output, refine_path, output_2)
    )
    assert result_1["failures"] == []
    assert result_2["failures"] == []
    assert result_1["decision_rows"]
    assert all(
        row["proposed_decision"] == "candidate_generation_only"
        for row in result_1["decision_rows"]
    )
    for name, expected in input_hashes.items():
        assert _sha256(paths[name]) == expected
    for relative in (
        "selection_resolved.csv",
        "bbox_decisions.csv",
        "candidate_metrics.csv",
        "video_summary.csv",
        "category_summary.csv",
        "failures.csv",
        "run_config.yaml",
        "checksums.csv",
    ):
        assert (output_1 / relative).read_bytes() == (output_2 / relative).read_bytes()
    files_1 = sorted(
        path.relative_to(output_1)
        for path in output_1.rglob("*.png")
    )
    files_2 = sorted(
        path.relative_to(output_2)
        for path in output_2.rglob("*.png")
    )
    assert files_1 and files_1 == files_2
    for relative in files_1:
        assert (output_1 / relative).read_bytes() == (output_2 / relative).read_bytes()

    production_args = _phase3_args(
        paths,
        phase1_output,
        refine_path,
        fixture_root / "must_not_exist",
        candidate_generation_only=False,
    )
    try:
        run_prototype(production_args)
    except ValueError as exc:
        assert "Production thresholds are not fixed" in str(exc)
    else:
        raise AssertionError("unfixed production thresholds must be rejected")
    assert not production_args.output_root.exists()
    print("[OK] end-to-end Phase 1 contract, read-only inputs, and deterministic outputs")


def test_config_validation(root: Path) -> None:
    invalid = _config_mapping("0" * 64)
    invalid["candidate_generation"]["percentile_values"] = [85.0, 75.0]
    path = root / "invalid.yaml"
    path.write_text(yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8")
    try:
        load_contour_refinement_config(path)
    except ValueError as exc:
        assert "percentile_values" in str(exc)
    else:
        raise AssertionError("unsorted percentile values must be rejected")
    print("[OK] strict Phase 3 config validation")


def test_project_config_contract() -> None:
    teacher = load_audit_teacher_config(
        REPO_ROOT
        / "pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml"
    )
    refine = load_contour_refinement_config(
        REPO_ROOT
        / "pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3.yaml"
    )
    assert refine.base_teacher_config_name == teacher.config_name
    if refine.base_teacher_fingerprint is not None:
        assert refine.base_teacher_fingerprint == teacher.fingerprint
    assert refine.decision.production_thresholds_fixed is False
    try:
        refine.validate(require_production_thresholds=True)
    except ValueError as exc:
        assert "Production thresholds are not fixed" in str(exc)
    else:
        raise AssertionError("screening config must not enable production decisions")

    production = load_contour_refinement_config(
        REPO_ROOT
        / "pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml"
    )
    assert production.config_name == "stage4_contour_auto_refine_phase3_production_v1"
    assert production.base_teacher_config_name == teacher.config_name
    if production.base_teacher_fingerprint is not None:
        assert production.base_teacher_fingerprint == teacher.fingerprint
    production.validate(require_production_thresholds=True)
    assert production.trigger == refine.trigger
    assert production.candidate_generation == refine.candidate_generation
    assert production.eligibility == refine.eligibility
    assert production.ranking == refine.ranking
    assert production.decision.production_thresholds_fixed is True
    assert production.decision.minimum_refine_score_gain == 0.05
    assert production.decision.minimum_score_margin == 0.005
    assert production.decision.minimum_stability_iou == 0.70
    assert production.decision.minimum_support_count == 1
    assert production.decision.equal_score_tolerance == 1.0e-12
    print("[OK] project screening safety gate and balanced production config")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_contour_auto_refine_") as temp:
        root = Path(temp)
        config_path = _write_config(root / "refine.yaml")
        test_auto_accept_and_refine(config_path)
        test_rejection_and_ambiguity(config_path)
        test_deduplication_and_determinism(config_path)
        test_end_to_end_read_only_prototype(root)
        test_config_validation(root)
        test_project_config_contract()
    print("Stage 4 contour-teacher Phase 3 synthetic checks passed.")


if __name__ == "__main__":
    main()
