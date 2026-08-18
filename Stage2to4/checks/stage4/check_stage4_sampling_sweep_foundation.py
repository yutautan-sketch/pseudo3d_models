from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from pseudo3d.analysis.stage4_sampling_evaluation import (
    SOURCE_BITS,
    TeacherBBox,
    ablate_source,
    bbox_selected_counts,
    label_counts,
    source_label_counts,
)
from pseudo3d.analysis.sweep_stage4_sampling_parameters import (
    _aggregate_bbox_rows,
    _aggregate_frame_rows,
    _load_search_config,
    _pareto_rows,
)


SEARCH_CONFIG = (
    REPO_ROOT
    / "pseudo3d/analysis/configs/stage4_sampling_smoke_search.yaml"
)


def _bbox() -> TeacherBBox:
    positive_y = np.asarray([2, 2, 3, 3], dtype=np.int32)
    positive_x = np.asarray([2, 3, 2, 3], dtype=np.int32)
    return TeacherBBox(
        video_name="synthetic",
        frame_order=0,
        frame_index=10,
        bbox_index=0,
        object_index=0,
        object_name="leg",
        xml_path="synthetic.xml",
        local_xyxy=(1.0, 1.0, 4.0, 4.0),
        roi_bounds=(1, 1, 4, 4),
        positive_y=positive_y,
        positive_x=positive_x,
        teacher_status="teacher_valid",
        annotation_reason="ok",
        contour_area=4.0,
        binary_area=4,
        foreground_ratio_in_bbox=0.25,
    )


def test_known_teacher_and_sampling_counts() -> None:
    labels = np.zeros((6, 6), dtype=np.int8)
    labels[1:5, 1:5] = -1
    labels[2:4, 2:4] = 1
    bbox = _bbox()

    all_selected = np.ones_like(labels, dtype=bool)
    counts = label_counts(all_selected, labels)
    assert counts["selected_total_count"] == 36
    assert counts["selected_positive_count"] == 4
    assert counts["selected_ignore_count"] == 12
    assert counts["selected_background_count"] == 20
    bbox_counts = bbox_selected_counts(
        bbox,
        selected_mask=all_selected,
        labels=labels,
    )
    assert bbox_counts["positive_pixel_recall"] == 1.0
    assert not bbox_counts["zero_positive"]

    empty = np.zeros_like(labels, dtype=bool)
    empty_counts = bbox_selected_counts(bbox, selected_mask=empty, labels=labels)
    assert empty_counts["selected_positive_count"] == 0
    assert empty_counts["zero_positive"]


def test_source_ablation_and_overlap() -> None:
    flags = np.zeros((4, 4), dtype=np.uint16)
    flags[1, 1] = np.uint16(SOURCE_BITS["local"] | SOURCE_BITS["tophat"])
    flags[2, 2] = np.uint16(SOURCE_BITS["local"])
    flags[3, 3] = np.uint16(SOURCE_BITS["context"])
    labels = np.zeros((4, 4), dtype=np.int8)
    labels[1, 1] = 1
    labels[2, 2] = -1

    without_local = ablate_source(flags, "local")
    assert without_local[1, 1] == SOURCE_BITS["tophat"]
    assert without_local[2, 2] == 0
    assert without_local[3, 3] == SOURCE_BITS["context"]
    source_counts = source_label_counts(flags, labels)
    assert source_counts["source_local_selected"] == 2
    assert source_counts["source_tophat_selected"] == 1
    assert source_counts["source_local_positive"] == 1
    assert source_counts["source_tophat_positive"] == 1
    assert source_counts["context_only_selected"] == 1


def test_search_expansion_aggregation_and_pareto() -> None:
    _, configs, _ = _load_search_config(SEARCH_CONFIG, max_configs=10)
    assert len(configs) == 5
    assert len({config.config_id for config in configs}) == 5

    bbox_rows = [
        {
            "teacher_status": "teacher_valid",
            "zero_positive": False,
            "teacher_positive_count": 10,
            "selected_positive_count": 8,
            "positive_pixel_recall": 0.8,
            "positive_extent_recall": 0.75,
        },
        {
            "teacher_status": "teacher_invalid",
            "zero_positive": False,
            "teacher_positive_count": 0,
            "selected_positive_count": 0,
            "positive_pixel_recall": float("nan"),
            "positive_extent_recall": float("nan"),
        },
    ]
    bbox_summary = _aggregate_bbox_rows(bbox_rows)
    assert bbox_summary["teacher_valid_bbox_count"] == 1
    assert bbox_summary["teacher_invalid_bbox_count"] == 1
    assert bbox_summary["zero_positive_bbox_count"] == 0
    assert bbox_summary["positive_bbox_recall_at_8"] == 1.0

    frame_rows = [
        {
            "selected_total_count": 10,
            "legacy_point_count": 5,
            "selected_positive_count": 2,
            "selected_background_count": 5,
            "selected_ignore_count": 3,
            "sampling_time_seconds": 0.01,
            "context_only_selected": 2,
            "context_only_positive": 0,
            "context_only_background": 1,
            "context_only_ignore": 1,
        }
    ]
    frame_summary = _aggregate_frame_rows(frame_rows)
    assert frame_summary["legacy_point_count_multiplier"] == 2.0
    assert frame_summary["selected_ignore_ratio"] == 0.3
    assert frame_summary["context_background_yield"] == 0.5

    candidate_a = {
        "config_id": "a",
        "zero_positive_bbox_rate": 0.0,
        "positive_bbox_recall_at_16": 1.0,
        "positive_pixel_recall_p05": 0.8,
        "selected_ignore_ratio": 0.2,
        "points_per_frame_mean": 100.0,
    }
    candidate_b = {
        **candidate_a,
        "config_id": "b",
        "selected_ignore_ratio": 0.3,
        "points_per_frame_mean": 120.0,
    }
    pareto = _pareto_rows([candidate_a, candidate_b])
    assert [row["config_id"] for row in pareto] == ["a"]


def main() -> None:
    test_known_teacher_and_sampling_counts()
    print("[OK] known dense-teacher and sampling counts")
    test_source_ablation_and_overlap()
    print("[OK] source ablation and overlap handling")
    test_search_expansion_aggregation_and_pareto()
    print("[OK] search expansion, aggregation, and Pareto extraction")
    print("Stage 4 sampling sweep foundation checks passed.")


if __name__ == "__main__":
    main()
