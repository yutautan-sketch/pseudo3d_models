from __future__ import annotations

from pathlib import Path

import numpy as np

from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_BACKGROUND,
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
    LocalBBox,
    RankedContourConfig,
    VocBBox,
    build_bbox_ranked_contour_mask,
    build_contour_point_annotations,
)
from src.utils.alpha_texture_processing import AlphaTextureConfig


def make_bbox() -> LocalBBox:
    xyxy = np.asarray([10.0, 10.0, 90.0, 70.0], dtype=np.float32)
    return LocalBBox(
        xml_xyxy=xyxy.copy(),
        raw_xyxy=xyxy.copy(),
        local_xyxy=xyxy.copy(),
        valid=True,
    )


def make_config() -> RankedContourConfig:
    return RankedContourConfig(
        min_area_ratio=0.02,
        sufficient_area_ratio=0.10,
        max_center_distance_norm=0.50,
        center_weight=0.75,
        area_weight=0.25,
    )


def rectangle_mask(
    shape: tuple[int, int],
    xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = xyxy
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def test_local_beats_off_center_global() -> None:
    global_mask = rectangle_mask((100, 110), (12, 25, 30, 55))
    local_mask = rectangle_mask((100, 110), (38, 28, 62, 52))
    result = build_bbox_ranked_contour_mask(
        global_binary=global_mask,
        local_binary=local_mask,
        bbox=make_bbox(),
        min_contour_area=20.0,
        config=make_config(),
    )
    assert result["valid"]
    assert result["selected_contour_source"] == "local_percentile"
    assert result["local_candidate_score"] > result["global_candidate_score"]
    assert result["mask"][40, 50]
    assert not result["mask"][40, 20]
    print("[OK] centered local contour beats off-center global contour")


def test_global_can_win_without_source_bias() -> None:
    global_mask = rectangle_mask((100, 110), (38, 28, 62, 52))
    local_mask = rectangle_mask((100, 110), (70, 22, 86, 42))
    result = build_bbox_ranked_contour_mask(
        global_binary=global_mask,
        local_binary=local_mask,
        bbox=make_bbox(),
        min_contour_area=20.0,
        config=make_config(),
    )
    assert result["valid"]
    assert result["selected_contour_source"] == "global"
    assert result["global_candidate_score"] > result["local_candidate_score"]
    print("[OK] centered global contour wins when it is BBox-consistent")


def test_identical_sources_are_shared() -> None:
    mask = rectangle_mask((100, 110), (38, 28, 62, 52))
    result = build_bbox_ranked_contour_mask(
        global_binary=mask,
        local_binary=mask,
        bbox=make_bbox(),
        min_contour_area=20.0,
        config=make_config(),
    )
    assert result["valid"]
    assert result["selected_contour_source"] == "shared"
    assert result["reason"] == "bbox_ranked_identical_global_local"
    print("[OK] identical global/local contours have shared provenance")


def test_small_candidate_is_rejected() -> None:
    global_mask = rectangle_mask((100, 110), (38, 28, 62, 52))
    local_mask = rectangle_mask((100, 110), (49, 39, 51, 41))
    result = build_bbox_ranked_contour_mask(
        global_binary=global_mask,
        local_binary=local_mask,
        bbox=make_bbox(),
        min_contour_area=20.0,
        config=make_config(),
    )
    assert result["valid"]
    assert result["selected_contour_source"] == "global"
    assert result["num_local_candidates"] == 1
    assert result["num_local_eligible_candidates"] == 0
    print("[OK] too-small centered candidate is rejected")


def test_no_eligible_candidate() -> None:
    empty = np.zeros((100, 110), dtype=bool)
    tiny = rectangle_mask((100, 110), (49, 39, 51, 41))
    result = build_bbox_ranked_contour_mask(
        global_binary=empty,
        local_binary=tiny,
        bbox=make_bbox(),
        min_contour_area=20.0,
        config=make_config(),
    )
    assert not result["valid"]
    assert result["selected_contour_source"] == "none"
    assert not np.any(result["mask"])
    print("[OK] no eligible candidate produces an invalid empty mask")


def test_config_validation() -> None:
    try:
        RankedContourConfig(local_window_size=30).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("even local_window_size must be rejected")

    try:
        RankedContourConfig(center_weight=0.0, area_weight=0.0).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("zero score weights must be rejected")
    print("[OK] ranked contour configuration validation")


def test_stage5_label_policy() -> None:
    images = np.zeros((2, 10, 10), dtype=np.uint8)
    images[0, 3:7, 3:7] = 255
    bbox_xyxy = np.asarray([2.0, 2.0, 7.0, 7.0], dtype=np.float32)
    voc = VocBBox(
        xml_path=Path("synthetic_00001.xml"),
        object_index=0,
        object_name="leg",
        xml_xyxy=bbox_xyxy.copy(),
        image_size_wh=np.asarray([10.0, 10.0], dtype=np.float32),
    )
    local_bbox = LocalBBox(
        xml_xyxy=bbox_xyxy.copy(),
        raw_xyxy=bbox_xyxy.copy(),
        local_xyxy=bbox_xyxy.copy(),
        valid=True,
    )
    point_cloud = {
        "frame_order": np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        "pixel_xy": np.asarray(
            [[0, 0], [2, 2], [4, 4], [0, 0], [4, 4]],
            dtype=np.float32,
        ),
    }
    pseudo3d = {
        "local_encoder_images": images,
        "frame_indices": np.asarray([0, 1], dtype=np.int64),
    }
    labels, valid_mask, frame_annotation = build_contour_point_annotations(
        point_cloud=point_cloud,
        pseudo3d=pseudo3d,
        voc_bboxes={0: [voc]},
        local_bboxes={0: [local_bbox]},
        texture_config=AlphaTextureConfig(
            threshold_mode="fixed",
            fixed_threshold=180,
        ),
        min_alpha=1,
        min_contour_area=0.0,
        no_bbox_label=LABEL_BACKGROUND,
        bbox_inside_non_contour_label="ignore",
    )
    np.testing.assert_array_equal(
        labels,
        np.asarray(
            [
                LABEL_BACKGROUND,
                LABEL_IGNORE,
                LABEL_FEMUR_CANDIDATE,
                LABEL_BACKGROUND,
                LABEL_BACKGROUND,
            ],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(valid_mask, labels != LABEL_IGNORE)
    np.testing.assert_array_equal(
        np.unique(frame_annotation["frame_order"]),
        np.asarray([0], dtype=np.int32),
    )
    print("[OK] no-BBox background and BBox non-contour ignore policy")


def main() -> None:
    test_local_beats_off_center_global()
    test_global_can_win_without_source_bias()
    test_identical_sources_are_shared()
    test_small_candidate_is_rejected()
    test_no_eligible_candidate()
    test_config_validation()
    test_stage5_label_policy()
    print("Stage 4 BBox-ranked teacher synthetic checks passed.")


if __name__ == "__main__":
    main()
