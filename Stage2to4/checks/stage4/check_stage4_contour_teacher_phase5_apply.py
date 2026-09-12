from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.stage4.check_stage4_contour_auto_refine import (
    _phase3_args,
    _write_config,
)
from checks.stage4.check_stage4_contour_teacher_audit import (
    _args as phase1_args,
    _make_integration_fixture,
)
from pseudo3d.analysis.audit_stage4_contour_teacher import (
    load_audit_teacher_config,
    mask_shape_metrics,
    run_audit,
)
from pseudo3d.analysis.prototype_stage4_contour_auto_refine import (
    _checksum_rows as phase3_checksum_rows,
    _write_csv as phase3_write_csv,
    run_prototype,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import LocalBBox
from pseudo3d.annotation.contour_teacher_refinement import mask_sha256
from pseudo3d.annotation.stage4_manual_review import file_sha256, read_csv_rows
from pseudo3d.batch.annotation.batch_apply_stage4_contour_refinement import (
    run_phase5,
)


def _force_one_auto_refine(phase3_root: Path, source_h5: Path) -> None:
    decisions = read_csv_rows(phase3_root / "bbox_decisions.csv")
    assert len(decisions) == 2
    target = decisions[0]
    proposal_path = phase3_root / target["proposal_mask_path"]
    proposal_u8 = cv2.imread(str(proposal_path), cv2.IMREAD_UNCHANGED)
    assert proposal_u8 is not None and proposal_u8.ndim == 2
    proposal = proposal_u8 > 0
    positive_y, positive_x = np.nonzero(proposal)
    assert positive_x.size > 4
    proposal[positive_y[::2], positive_x[::2]] = False
    assert np.any(proposal)
    assert cv2.imwrite(str(proposal_path), proposal.astype(np.uint8) * 255)

    with h5py.File(source_h5, "r") as handle:
        frame_group = handle["frame_annotation"]
        matches = np.flatnonzero(
            (frame_group["frame_order"][:].astype(np.int64) == int(target["frame_order"]))
            & (frame_group["frame_index"][:].astype(np.int64) == int(target["frame_index"]))
            & (frame_group["bbox_index"][:].astype(np.int64) == int(target["bbox_index"]))
        )
        assert matches.size == 1
        bbox_values = frame_group["bbox_local_xyxy"][int(matches[0])].astype(np.float32)
    bbox = LocalBBox(
        xml_xyxy=bbox_values.copy(),
        raw_xyxy=bbox_values.copy(),
        local_xyxy=bbox_values.copy(),
        valid=True,
    )
    metrics = mask_shape_metrics(proposal, bbox)
    target.update(
        {
            "proposed_decision": "auto_refine",
            "reason_codes": "synthetic_phase5_auto_refine",
            "proposal_source": "synthetic_refine",
            "proposal_mask_sha256": mask_sha256(proposal),
            "proposal_area_ratio": str(metrics["area_ratio"]),
            "proposal_center_distance_norm": str(metrics["center_distance_norm"]),
            "proposal_score": "0.95",
            "proposal_stability_iou": "1.0",
            "proposal_support_count": "2",
            "score_gain": "0.25",
            "score_margin": "0.10",
        }
    )
    decisions[1]["proposed_decision"] = "auto_accept"
    decisions[1]["reason_codes"] = "synthetic_preserve_v2"
    phase3_write_csv(phase3_root / "bbox_decisions.csv", decisions)

    summary_path = phase3_root / "refine_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision_counts"] = {"auto_accept": 1, "auto_refine": 1}
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase3_write_csv(
        phase3_root / "checksums.csv", phase3_checksum_rows(phase3_root)
    )


def _phase5_args(
    paths: dict[str, Path],
    phase1_root: Path,
    phase3_root: Path,
    refine_config: Path,
    output_root: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=paths["manifest"],
        source_annotated_root=paths["annotated_root"],
        phase1_audit_root=phase1_root,
        phase3_root=phase3_root,
        teacher_config=paths["teacher"],
        refine_config=refine_config,
        output_root=output_root,
        expected_videos=1,
        annotated_glob_template="{video_name}/*bboxrank_v2_nobbox_bg.h5",
        skip_existing=False,
        overwrite=False,
        continue_on_error=False,
    )


def _output_h5(root: Path) -> Path:
    matches = sorted(root.glob("*/*bboxrank_v3_refined_auto_v1.h5"))
    assert len(matches) == 1
    return matches[0]


def test_phase5_apply(root: Path) -> None:
    fixture = root / "fixture"
    fixture.mkdir()
    paths = _make_integration_fixture(fixture)
    source_hash = file_sha256(paths["annotated"])
    phase1_root = fixture / "phase1"
    run_audit(phase1_args(paths, phase1_root))
    teacher = load_audit_teacher_config(paths["teacher"])
    refine_config = _write_config(
        fixture / "refine.yaml",
        teacher.fingerprint,
        production_thresholds_fixed=True,
    )
    phase3_root = fixture / "phase3"
    result = run_prototype(
        _phase3_args(
            paths,
            phase1_root,
            refine_config,
            phase3_root,
            candidate_generation_only=False,
        )
    )
    assert not result["failures"]
    _force_one_auto_refine(phase3_root, paths["annotated"])

    output_1 = fixture / "phase5_1"
    output_2 = fixture / "phase5_2"
    result_1 = run_phase5(
        _phase5_args(paths, phase1_root, phase3_root, refine_config, output_1)
    )
    result_2 = run_phase5(
        _phase5_args(paths, phase1_root, phase3_root, refine_config, output_2)
    )
    assert result_1["summary"]["status"] == "ok"
    assert result_1["summary"]["refined_bboxes"] == 1
    assert result_2["summary"]["refined_bboxes"] == 1
    assert file_sha256(paths["annotated"]) == source_hash

    first_path = _output_h5(output_1)
    second_path = _output_h5(output_2)
    with h5py.File(paths["annotated"], "r") as source, h5py.File(
        first_path, "r"
    ) as first, h5py.File(second_path, "r") as second:
        for name in source["point_cloud"]:
            np.testing.assert_array_equal(
                source[f"point_cloud/{name}"][:], first[f"point_cloud/{name}"][:]
            )
        first_labels = first["annotation/point_label"][:]
        first_valid = first["annotation/valid_mask"][:]
        np.testing.assert_array_equal(first_valid, first_labels != -1)
        np.testing.assert_array_equal(
            first_labels, second["annotation/point_label"][:]
        )
        assert not np.array_equal(
            source["annotation/point_label"][:], first_labels
        )
        assert first.attrs["contour_teacher_schema"] == (
            "bboxrank_v3_refined_auto_v1"
        )
        assert int(first.attrs["contour_refined_bbox_count"]) == 1
        assert first["contour_refinement/frame_order"].shape == (1,)
        assert first["measurement/source_frame_order"][()] == source[
            "measurement/source_frame_order"
        ][()]
    print("[OK] Phase 5 auto-refine apply, v2 preservation, provenance, and determinism")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_phase5_apply_") as temporary:
        test_phase5_apply(Path(temporary))
    print("Stage 4 contour-teacher Phase 5 synthetic checks passed.")


if __name__ == "__main__":
    main()
