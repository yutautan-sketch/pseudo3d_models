from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pseudo3d.analysis.audit_stage4_crop_quality_stray_ignore import (
    CropQualityAuditError,
    TargetSpec,
    run_audit,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256


TARGET_VIDEO = "synthetic_crop_target"
COMPARISON_VIDEO = "synthetic_crop_comparison"
OUTPUT_FILES = (
    "audit_summary.json",
    "bbox_geometry.csv",
    "frame_metrics.csv",
    "input_checksums.csv",
    "stray_points.csv",
    "video_summary.csv",
)


def _text(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _write_xml(path: Path, bbox: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "<annotation>",
                "  <size><width>12</width><height>8</height></size>",
                "  <object>",
                "    <name>leg</name>",
                "    <bndbox>",
                f"      <xmin>{x1}</xmin><ymin>{y1}</ymin>",
                f"      <xmax>{x2}</xmax><ymax>{y2}</ymax>",
                "    </bndbox>",
                "  </object>",
                "</annotation>",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_pseudo(path: Path, frame_indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "local_encoder_images",
            data=np.zeros((len(frame_indices), 8, 8), dtype=np.uint8),
        )
        handle.create_dataset("frame_indices", data=frame_indices)
        handle.attrs["local_preprocess"] = "resize_shorter_then_offset_crop"
        handle.attrs["local_resize_scale"] = 1.0
        handle.attrs["local_crop_left"] = 4.0
        handle.attrs["local_crop_top"] = 0.0
        handle.attrs["raw_width"] = 12.0
        handle.attrs["raw_height"] = 8.0


def _write_target_h5(
    path: Path,
    xml_paths: list[Path],
    *,
    schema: str,
    source_v5: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_order = np.asarray([0, 1, 1, 2], dtype=np.int32)
    pixel_xy = np.asarray([[2, 3], [0, 3], [1, 3], [1, 3]], dtype=np.float32)
    labels = np.asarray([-1, -1, 0, -1], dtype=np.int8)
    with h5py.File(path, "w") as handle:
        handle.attrs["contour_teacher_schema"] = schema
        handle.attrs["video_name"] = TARGET_VIDEO

        point = handle.create_group("point_cloud")
        point.create_dataset("frame_order", data=frame_order)
        point.create_dataset("pixel_xy", data=pixel_xy)

        annotation = handle.create_group("annotation")
        annotation.create_dataset("point_label", data=labels)
        annotation.create_dataset("valid_mask", data=labels != -1)

        frame = handle.create_group("frame_annotation")
        frame.create_dataset("frame_order", data=np.asarray([0, 1, 2], dtype=np.int32))
        frame.create_dataset("frame_index", data=np.asarray([10, 11, 12], dtype=np.int64))
        frame.create_dataset("bbox_index", data=np.zeros(3, dtype=np.int32))
        frame.create_dataset(
            "bbox_local_xyxy",
            data=np.asarray(
                [[1, 2, 4, 5], [0, 2, 0, 5], [0, 2, 2, 5]],
                dtype=np.float32,
            ),
        )
        frame.create_dataset("valid_contour", data=np.asarray([True, False, True]))
        frame.create_dataset(
            "selected_contour_source",
            data=_text(["global", "", "local_percentile"]),
        )
        frame.create_dataset(
            "annotation_reason",
            data=_text(["ok", "bbox_outside_crop", "ok"]),
        )
        frame.create_dataset("xml_path", data=_text([str(path) for path in xml_paths]))

        provenance = handle.create_group("manual_review_fullvideo_frames")
        provenance.create_dataset(
            "frame_order", data=np.asarray([0, 1, 2], dtype=np.int32)
        )
        provenance.create_dataset(
            "label_authority",
            data=_text(["cvat_snapshot", "cvat_snapshot", "cvat_snapshot"]),
        )
        provenance.create_dataset(
            "cvat_mask_status",
            data=_text(["positive", "empty", "positive"]),
        )
        if source_v5 is not None:
            invalidation = handle.create_group("xml_annotation_invalidation")
            invalidation.attrs["source_h5"] = str(source_v5.resolve())


def _write_manifest(
    path: Path,
    *,
    target_pseudo: Path,
    comparison_pseudo: Path,
    voc_root: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "video_name",
                "pseudo3d_h5",
                "voc_xml_root",
                "split",
                "enabled",
                "notes",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": TARGET_VIDEO,
                "pseudo3d_h5": target_pseudo,
                "voc_xml_root": voc_root,
                "split": "train",
                "enabled": "true",
                "notes": "synthetic target",
            }
        )
        writer.writerow(
            {
                "video_name": COMPARISON_VIDEO,
                "pseudo3d_h5": comparison_pseudo,
                "voc_xml_root": voc_root,
                "split": "train",
                "enabled": "true",
                "notes": "synthetic comparison",
            }
        )


def _fixture(root: Path) -> dict[str, Path]:
    voc_root = root / "voc"
    target_indices = np.asarray([10, 11, 12], dtype=np.int64)
    target_pseudo = root / f"{TARGET_VIDEO}.h5"
    _write_pseudo(target_pseudo, target_indices)
    target_xml = []
    for frame_index, bbox in zip(
        target_indices,
        ((5, 2, 8, 5), (1, 2, 3, 5), (3, 2, 6, 5)),
        strict=True,
    ):
        path = (
            voc_root
            / TARGET_VIDEO
            / "annotations_renamed"
            / f"{TARGET_VIDEO}_{int(frame_index) + 1:05d}.xml"
        )
        _write_xml(path, bbox)
        target_xml.append(path)

    comparison_pseudo = root / f"{COMPARISON_VIDEO}.h5"
    _write_pseudo(comparison_pseudo, np.asarray([20], dtype=np.int64))
    _write_xml(
        voc_root
        / COMPARISON_VIDEO
        / "annotations_renamed"
        / f"{COMPARISON_VIDEO}_00021.xml",
        (5, 1, 9, 6),
    )

    annotated_root = root / "annotated"
    source_v5 = root / f"{TARGET_VIDEO}_bboxrank_v5_cvat_authoritative_v1.h5"
    _write_target_h5(
        source_v5,
        target_xml,
        schema="bboxrank_v5_cvat_authoritative_v1",
    )
    target_v6 = (
        annotated_root
        / TARGET_VIDEO
        / f"{TARGET_VIDEO}_bboxrank_v6_cvat_authoritative_xml_invalidation_v1.h5"
    )
    _write_target_h5(
        target_v6,
        target_xml,
        schema="bboxrank_v6_cvat_authoritative_xml_invalidation_v1",
        source_v5=source_v5,
    )
    manifest = root / "manifest.csv"
    _write_manifest(
        manifest,
        target_pseudo=target_pseudo,
        comparison_pseudo=comparison_pseudo,
        voc_root=voc_root,
    )
    return {
        "manifest": manifest,
        "annotated_root": annotated_root,
        "target_pseudo": target_pseudo,
        "target_v6": target_v6,
        "source_v5": source_v5,
        "voc_root": voc_root,
    }


def _args(paths: dict[str, Path], output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=paths["manifest"],
        teacher_config=REPO_ROOT
        / "pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml",
        annotated_root=paths["annotated_root"],
        output_root=output,
        target=[TargetSpec(TARGET_VIDEO, (1,))],
        comparison_video=[COMPARISON_VIDEO],
        annotated_glob_template=(
            "{video_name}/*bboxrank_v6_cvat_authoritative_xml_invalidation_v1.h5"
        ),
        expected_stray_points=1,
        expected_stray_videos=1,
        expected_problem_frames=1,
        overwrite=False,
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_audit_and_determinism(root: Path) -> None:
    paths = _fixture(root)
    protected = {
        path: file_sha256(path)
        for path in (
            paths["manifest"],
            paths["target_pseudo"],
            paths["target_v6"],
            paths["source_v5"],
            *sorted(paths["voc_root"].rglob("*.xml")),
        )
    }
    first = root / "output_1"
    second = root / "output_2"
    result = run_audit(_args(paths, first))
    run_audit(_args(paths, second))

    assert result["summary"]["stray_ignore_points"] == 1
    assert result["summary"]["all_stray_explained_by_degenerate_bbox_semantics"]
    stray = _csv_rows(first / "stray_points.csv")
    assert len(stray) == 1
    assert stray[0]["frame_order"] == "1"
    assert stray[0]["rounded_x"] == "0"
    assert stray[0]["strict_bbox_inside"] == "False"
    assert stray[0]["permissive_bbox_inside"] == "True"
    assert stray[0]["source_v5_label"] == "-1"
    assert stray[0]["source_v5_stray_ignore"] == "True"

    bbox = _csv_rows(first / "bbox_geometry.csv")
    target = [row for row in bbox if row["video_name"] == TARGET_VIDEO]
    assert [row["crop_status"] for row in target] == [
        "fully_visible",
        "fully_outside_crop",
        "partially_clipped",
    ]
    assert target[1]["saved_bbox_degenerate"] == "True"
    assert target[1]["local_bbox_valid"] == "False"

    for name in OUTPUT_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert all(file_sha256(path) == digest for path, digest in protected.items())
    print(
        "[OK] degenerate-BBox stray reproduction, crop visibility, comparison, "
        "read-only inputs, and deterministic outputs"
    )


def test_fail_fast(root: Path) -> None:
    paths = _fixture(root / "invalid")
    args = _args(paths, root / "wrong_frames")
    args.target = [TargetSpec(TARGET_VIDEO, (0,))]
    try:
        run_audit(args)
    except CropQualityAuditError as exc:
        assert "Observed stray frame orders differ" in str(exc)
    else:
        raise AssertionError("Wrong expected problem frame was accepted")

    args = _args(paths, root / "wrong_count")
    args.expected_stray_points = 2
    try:
        run_audit(args)
    except CropQualityAuditError as exc:
        assert "Stray ignore count mismatch" in str(exc)
    else:
        raise AssertionError("Wrong expected stray count was accepted")
    assert not (root / "wrong_frames").exists()
    assert not (root / "wrong_count").exists()
    print("[OK] problem-frame and expected-count mismatches fail before output")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_crop_stray_ignore_") as directory:
        root = Path(directory)
        test_audit_and_determinism(root / "valid")
        test_fail_fast(root)
    print("Stage 4 crop-quality/stray-ignore synthetic checks passed.")


if __name__ == "__main__":
    main()
