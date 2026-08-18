from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import yaml

from pseudo3d.analysis.audit_stage4_contour_teacher import (
    candidate_comparison_metrics,
    load_audit_teacher_config,
    mask_shape_metrics,
    run_audit,
    select_category_rows,
)
from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_BACKGROUND,
    RankedContourConfig,
    build_bbox_ranked_contour_candidates,
    build_bbox_ranked_contour_mask,
    build_contour_point_annotations,
    load_voc_bboxes,
    ranked_contour_metadata,
    save_annotated_h5,
    xml_bbox_to_local,
)
from src.utils.alpha_texture_processing import AlphaTextureConfig, image_to_uint8_gray


VIDEO_NAME = "synthetic_contour_audit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rectangle_mask(
    shape: tuple[int, int], xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = xyxy
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def test_shape_metrics() -> None:
    from pseudo3d.annotation.annotate_pseudo3d_point_cloud import LocalBBox

    bbox_xyxy = np.asarray([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
    bbox = LocalBBox(bbox_xyxy.copy(), bbox_xyxy.copy(), bbox_xyxy.copy(), True)
    centered = _rectangle_mask((64, 64), (20, 20, 40, 40))
    metrics = mask_shape_metrics(centered, bbox)
    assert metrics["filled_area"] == 21 * 21
    assert metrics["bbox_area"] == 41 * 41
    assert metrics["component_count"] == 1
    assert metrics["border_contact_pixels"] == 0
    assert not metrics["touch_top"]
    assert not metrics["touch_bottom"]
    assert not metrics["touch_left"]
    assert not metrics["touch_right"]
    assert np.isfinite(metrics["solidity"])
    assert np.isfinite(metrics["extent"])
    assert np.isfinite(metrics["compactness"])
    np.testing.assert_allclose(metrics["center_distance_norm"], 0.0, atol=1e-12)

    touching = _rectangle_mask((64, 64), (10, 10, 30, 30))
    metrics_touching = mask_shape_metrics(touching, bbox)
    assert metrics_touching["touch_top"]
    assert metrics_touching["touch_left"]
    assert not metrics_touching["touch_bottom"]
    assert not metrics_touching["touch_right"]
    assert metrics_touching["border_contact_pixels"] > 0
    assert metrics_touching["border_contact_ratio"] > 0.0

    empty = mask_shape_metrics(np.zeros((64, 64), dtype=bool), bbox)
    assert empty["filled_area"] == 0
    assert empty["component_count"] == 0
    assert np.isnan(empty["solidity"])
    print("[OK] shape, center, and BBox-border metrics")


def test_candidate_metrics_and_selection_isolation() -> None:
    from pseudo3d.annotation.annotate_pseudo3d_point_cloud import LocalBBox

    bbox_xyxy = np.asarray([10.0, 10.0, 90.0, 70.0], dtype=np.float32)
    bbox = LocalBBox(bbox_xyxy.copy(), bbox_xyxy.copy(), bbox_xyxy.copy(), True)
    global_mask = _rectangle_mask((100, 110), (12, 25, 30, 55))
    local_mask = _rectangle_mask((100, 110), (38, 28, 62, 52))
    config = RankedContourConfig(
        min_area_ratio=0.02,
        sufficient_area_ratio=0.10,
        max_center_distance_norm=0.50,
        center_weight=0.75,
        area_weight=0.25,
    )
    before = build_bbox_ranked_contour_mask(
        global_binary=global_mask,
        local_binary=local_mask,
        bbox=bbox,
        min_contour_area=20.0,
        config=config,
    )
    candidates = build_bbox_ranked_contour_candidates(
        global_binary=global_mask,
        local_binary=local_mask,
        bbox=bbox,
        min_contour_area=20.0,
        config=config,
    )
    comparison = candidate_comparison_metrics(candidates)
    after = build_bbox_ranked_contour_mask(
        global_binary=global_mask,
        local_binary=local_mask,
        bbox=bbox,
        min_contour_area=20.0,
        config=config,
    )
    assert len(candidates) == 2
    assert comparison["top1_source"] == "local_percentile"
    assert comparison["top2_source"] == "global"
    assert comparison["score_margin"] > 0.0
    assert 0.0 <= comparison["top1_top2_iou"] <= 1.0
    assert before["selected_contour_source"] == after["selected_contour_source"]
    assert before["reason"] == after["reason"]
    np.testing.assert_array_equal(before["mask"], after["mask"])
    print("[OK] candidate comparison and teacher-selection isolation")


def test_category_ranking_is_deterministic() -> None:
    rows = []
    for index in range(5):
        rows.append(
            {
                "video_name": f"v{index}",
                "frame_order": index,
                "bbox_index": 0,
                "valid_contour": index != 4,
                "selected_area_ratio": 0.1 * index,
                "foreground_ratio_in_bbox": 0.12 * index,
                "border_contact_ratio": 0.05 * index,
                "border_contact_pixels": index,
                "num_global_eligible_candidates": 1,
                "num_local_eligible_candidates": 1,
                "score_margin": 0.01 * (5 - index),
                "top1_top2_iou": 0.1 * index,
                "center_distance_norm": 0.02 * index,
                "positive_point_count": 10 - index,
                "area_ratio": 0.1 * index,
            }
        )
    first = select_category_rows(rows, top_k=3)
    second = select_category_rows(list(reversed(rows)), top_k=3)
    first_ids = {
        key: [row["video_name"] for row in value] for key, value in first.items()
    }
    second_ids = {
        key: [row["video_name"] for row in value] for key, value in second.items()
    }
    assert first_ids == second_ids
    assert first_ids["overfilled"][0] == "v3"
    assert first_ids["invalid"] == ["v4"]
    print("[OK] deterministic audit-category ranking")


def _write_xml(path: Path, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    objects = "\n".join(
        f"  <object><name>{name}</name><bndbox>\n"
        f"    <xmin>{xyxy[0]}</xmin><ymin>{xyxy[1]}</ymin>\n"
        f"    <xmax>{xyxy[2]}</xmax><ymax>{xyxy[3]}</ymax>\n"
        f"  </bndbox></object>"
        for name, xyxy in boxes
    )
    text = f"""<annotation>
  <size><width>110</width><height>100</height><depth>1</depth></size>
{objects}
</annotation>
"""
    path.write_text(text, encoding="utf-8")


def _teacher_mapping() -> dict[str, object]:
    return {
        "schema_version": 2,
        "config_name": "synthetic_bbox_ranked_teacher_v2",
        "labels": {
            "ignore": -1,
            "background": 0,
            "positive": 1,
            "no_bbox_label": 0,
            "bbox_inside_non_contour_label": "ignore",
        },
        "xml": {
            "frame_number_offsets": [1],
            "frame_id_source": "frame_index",
            "annotation_dir_name": "annotations_renamed",
            "strict_annotation_dir": True,
        },
        "global_candidate": {
            "preset": "synthetic_fixed",
            "threshold_mode": "fixed",
            "fixed_threshold": 180,
            "percentile": 85.0,
            "min_component_area": 0,
            "min_area": 5.0,
            "min_alpha": 1,
            "open_ksize": 0,
            "close_ksize": 0,
            "morph_shape": "ellipse",
            "denoise": "none",
            "denoise_ksize": 3,
        },
        "local_candidate": {
            "window_size": 11,
            "percentile": 75.0,
            "min_contrast": 10.0,
            "cleanup_open_ksize": 0,
            "cleanup_close_ksize": 0,
            "cleanup_morph_shape": "ellipse",
            "cleanup_min_component_area": 0,
        },
        "ranking": {
            "min_area_ratio": 0.01,
            "sufficient_area_ratio": 0.10,
            "max_center_distance_norm": 0.50,
            "center_weight": 0.75,
            "area_weight": 0.25,
        },
        "selection": {
            "extract_all_external_components": True,
            "union_sources_before_ranking": False,
            "selected_filled_contour_is_positive": True,
            "equal_rank_different_masks": "reject",
        },
    }


def _make_integration_fixture(root: Path) -> dict[str, Path]:
    teacher_path = root / "teacher.yaml"
    teacher_path.write_text(
        yaml.safe_dump(_teacher_mapping(), sort_keys=False), encoding="utf-8"
    )
    teacher = load_audit_teacher_config(teacher_path)

    images = np.full((2, 100, 110), 40, dtype=np.uint8)
    images[0, 20:45, 20:45] = 230
    images[0, 35:65, 70:95] = 245
    frame_indices = np.asarray([10, 30], dtype=np.int64)
    pseudo_path = root / f"{VIDEO_NAME}_ts448_oym96_corr.h5"
    with h5py.File(pseudo_path, "w") as handle:
        handle.create_dataset("local_encoder_images", data=images)
        handle.create_dataset("frame_indices", data=frame_indices)
        handle.attrs["local_preprocess"] = "resize"
        handle.attrs["raw_width"] = 110.0
        handle.attrs["raw_height"] = 100.0

    voc_root = root / "voc"
    annotation_dir = voc_root / VIDEO_NAME / "annotations_renamed"
    annotation_dir.mkdir(parents=True)
    xml_path = annotation_dir / f"{VIDEO_NAME}_00011.xml"
    _write_xml(
        xml_path,
        [
            ("leg", (10, 10, 50, 55)),
            ("leg", (60, 20, 102, 75)),
        ],
    )
    voc_bboxes = load_voc_bboxes(
        voc_root,
        video_name=VIDEO_NAME,
        frame_indices=frame_indices,
        xml_frame_number_offsets=(1,),
        xml_frame_id_source="frame_index",
        xml_annotation_dir_name="annotations_renamed",
        strict_xml_annotation_dir=True,
    )
    shape = image_to_uint8_gray(images[0]).shape
    pseudo_attrs = {
        "local_preprocess": "resize",
        "raw_width": 110.0,
        "raw_height": 100.0,
    }
    local_bboxes = {
        order: [
            xml_bbox_to_local(
                voc.xml_xyxy,
                image_size_wh=voc.image_size_wh,
                attrs=pseudo_attrs,
                local_shape_hw=shape,
            )
            for voc in boxes
        ]
        for order, boxes in voc_bboxes.items()
    }

    yy, xx = np.mgrid[0:100:5, 0:110:5]
    xy_one = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    pixel_xy = np.concatenate([xy_one, xy_one], axis=0)
    frame_order = np.concatenate(
        [
            np.zeros(xy_one.shape[0], dtype=np.int32),
            np.ones(xy_one.shape[0], dtype=np.int32),
        ]
    )
    num_points = int(frame_order.size)
    point_cloud = {
        "points": np.zeros((num_points, 3), dtype=np.float32),
        "intensity": np.ones(num_points, dtype=np.float32),
        "alpha": np.ones(num_points, dtype=np.float32),
        "confidence": np.ones(num_points, dtype=np.float32),
        "frame_index": frame_indices[frame_order],
        "frame_order": frame_order,
        "pixel_xy": pixel_xy,
        "source_type": np.zeros(num_points, dtype=np.uint8),
        "source_flags": np.ones(num_points, dtype=np.uint16),
        "sampling_confidence": np.ones(num_points, dtype=np.float32),
    }
    labels, valid_mask, frame_annotation = build_contour_point_annotations(
        point_cloud=point_cloud,
        pseudo3d={"local_encoder_images": images, "frame_indices": frame_indices},
        voc_bboxes=voc_bboxes,
        local_bboxes=local_bboxes,
        texture_config=teacher.texture_config,
        min_alpha=teacher.min_alpha,
        min_contour_area=teacher.min_contour_area,
        no_bbox_label=LABEL_BACKGROUND,
        bbox_inside_non_contour_label="ignore",
        ranked_contour_config=teacher.ranked_config,
    )
    annotated_root = root / "annotated"
    annotated_path = (
        annotated_root
        / VIDEO_NAME
        / f"{VIDEO_NAME}_pointcloud_annotated_bboxrank_v2_nobbox_bg.h5"
    )
    meta = {
        "source_pseudo3d_h5": str(pseudo_path),
        "video_name": VIDEO_NAME,
        "label_mode": "bbox_ranked_global_local",
        "bbox_inside_non_contour_label": "ignore",
        "no_bbox_label": 0,
        "xml_frame_id_source": "frame_index",
        "xml_frame_number_offsets": "1",
        "xml_annotation_dir_name": "annotations_renamed",
        "strict_xml_annotation_dir": True,
        "contour_preset": "synthetic_fixed",
        "contour_threshold_mode": teacher.texture_config.threshold_mode,
        "contour_percentile": teacher.texture_config.percentile,
        "contour_fixed_threshold": teacher.texture_config.fixed_threshold,
        "contour_min_component_area": teacher.texture_config.min_component_area,
        "contour_min_area": teacher.min_contour_area,
        "contour_min_alpha": teacher.min_alpha,
        "contour_open_ksize": teacher.texture_config.open_ksize,
        "contour_close_ksize": teacher.texture_config.close_ksize,
        "contour_denoise": teacher.texture_config.denoise,
        "contour_denoise_ksize": teacher.texture_config.denoise_ksize,
        "num_labeled_points": int(np.sum(labels == 1)),
        **ranked_contour_metadata(teacher.ranked_config),
    }
    save_annotated_h5(
        annotated_path,
        point_cloud=point_cloud,
        point_cloud_attrs={"point_mode": "foreground", "sampling_mode": "combined_v2"},
        point_label=labels,
        valid_mask=valid_mask,
        frame_annotation=frame_annotation,
        measurement={
            "endpoint_1": np.full(3, np.nan, dtype=np.float32),
            "endpoint_2": np.full(3, np.nan, dtype=np.float32),
            "length": np.float32(np.nan),
            "source_frame_order": np.int32(-1),
            "source_frame_index": np.int64(-1),
            "source_method": "synthetic",
        },
        meta=meta,
    )

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video_name", "pseudo3d_h5", "voc_xml_root", "split", "enabled"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": VIDEO_NAME,
                "pseudo3d_h5": str(pseudo_path),
                "voc_xml_root": str(voc_root),
                "split": "train",
                "enabled": "true",
            }
        )
    return {
        "teacher": teacher_path,
        "manifest": manifest,
        "pseudo": pseudo_path,
        "xml": xml_path,
        "annotated_root": annotated_root,
        "annotated": annotated_path,
    }


def _args(paths: dict[str, Path], output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=paths["manifest"],
        annotated_root=paths["annotated_root"],
        teacher_config=paths["teacher"],
        output_root=output,
        annotated_glob_template="{video_name}/*bboxrank_v2_nobbox_bg.h5",
        expected_videos=1,
        top_k=2,
        preflight_only=False,
        no_overlays=False,
        continue_on_error=False,
        overwrite=False,
    )


def test_end_to_end_read_only_audit() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_contour_audit_") as temp:
        root = Path(temp)
        paths = _make_integration_fixture(root)
        input_hashes = {
            key: _sha256(paths[key]) for key in ("pseudo", "xml", "annotated")
        }
        result_1 = run_audit(_args(paths, root / "output_1"))
        result_2 = run_audit(_args(paths, root / "output_2"))
        assert len(result_1["bbox_rows"]) == 2
        assert len(result_2["bbox_rows"]) == 2
        assert result_1["failures"] == []
        assert result_2["failures"] == []
        for key, expected in input_hashes.items():
            assert _sha256(paths[key]) == expected

        for name in ("bbox_audit.csv", "video_summary.csv", "category_summary.csv"):
            assert (root / "output_1" / name).read_bytes() == (
                root / "output_2" / name
            ).read_bytes()
        summary = json.loads((root / "output_1" / "audit_summary.json").read_text())
        assert summary["videos_checked"] == 1
        assert summary["bbox_rows"] == 2
        assert summary["failure_rows"] == 0
        assert summary["input_files_unchanged"] is True

        overlays_1 = sorted((root / "output_1" / "overlays").rglob("*.png"))
        overlays_2 = sorted((root / "output_2" / "overlays").rglob("*.png"))
        assert overlays_1
        assert [path.relative_to(root / "output_1") for path in overlays_1] == [
            path.relative_to(root / "output_2") for path in overlays_2
        ]
        for first, second in zip(overlays_1, overlays_2, strict=True):
            assert first.read_bytes() == second.read_bytes()
    print("[OK] end-to-end read-only audit, schema, labels, and determinism")


def test_validation_failures() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_contour_audit_invalid_") as temp:
        root = Path(temp)
        paths = _make_integration_fixture(root)
        invalid_mapping = _teacher_mapping()
        invalid_mapping["labels"]["no_bbox_label"] = -1  # type: ignore[index]
        invalid_path = root / "invalid_teacher.yaml"
        invalid_path.write_text(yaml.safe_dump(invalid_mapping), encoding="utf-8")
        try:
            load_audit_teacher_config(invalid_path)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid no-BBox label must be rejected")

        args = _args(paths, root / "output")
        args.expected_videos = 2
        try:
            run_audit(args)
        except ValueError:
            pass
        else:
            raise AssertionError("enabled-video count mismatch must be rejected")
    print("[OK] teacher and manifest validation failures")


def main() -> None:
    test_shape_metrics()
    test_candidate_metrics_and_selection_isolation()
    test_category_ranking_is_deterministic()
    test_end_to_end_read_only_audit()
    test_validation_failures()
    print("Stage 4 contour-teacher Phase 1 synthetic checks passed.")


if __name__ == "__main__":
    main()
