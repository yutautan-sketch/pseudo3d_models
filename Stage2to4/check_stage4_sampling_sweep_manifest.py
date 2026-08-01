from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from pseudo3d.analysis.stage4_sampling_sweep_config import (
    DenseTeacherConfig,
    dense_teacher_config_from_mapping,
    load_dense_teacher_config,
    load_stage4_sweep_manifest,
)
from pseudo3d.analysis.validate_stage4_sampling_manifest import audit_manifest


TEACHER_CONFIG = (
    REPO_ROOT
    / "pseudo3d/analysis/configs/stage4_dense_teacher_v1.yaml"
)


def _assert_raises(expected: type[BaseException], function: object) -> None:
    try:
        function()  # type: ignore[operator]
    except expected:
        return
    raise AssertionError(f"Expected {expected.__name__}")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "video_name",
        "pseudo3d_h5",
        "voc_xml_root",
        "split",
        "enabled",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_pseudo3d_h5(path: Path, frame_indices: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "local_encoder_images",
            data=np.zeros((len(frame_indices), 1, 8, 8), dtype=np.float32),
        )
        handle.create_dataset(
            "frame_indices",
            data=np.asarray(frame_indices, dtype=np.int64),
        )


def _write_voc_xml(path: Path, *, frame_number: int) -> None:
    path.write_text(
        "\n".join(
            [
                "<annotation>",
                f"  <filename>frame_{frame_number:05d}.jpg</filename>",
                "  <size><width>16</width><height>16</height></size>",
                "  <object>",
                "    <name>leg</name>",
                "    <bndbox>",
                "      <xmin>2</xmin><ymin>3</ymin>",
                "      <xmax>10</xmax><ymax>12</ymax>",
                "    </bndbox>",
                "  </object>",
                "</annotation>",
            ]
        ),
        encoding="utf-8",
    )


def test_teacher_config_is_fixed_and_fingerprinted() -> None:
    config = load_dense_teacher_config(TEACHER_CONFIG)
    config.validate()
    assert config.xml_frame_number_offsets == (1,)
    assert config.xml_frame_id_source == "frame_index"
    assert config.xml_annotation_dir_name == "annotations_renamed"
    assert config.strict_xml_annotation_dir
    assert not config.enable_contour_fallback
    assert config.contour_preset == "percentile85_area100"
    assert config.contour_percentile == 85.0
    assert config.contour_min_component_area == 100
    assert len(config.fingerprint()) == 64
    assert config.fingerprint() == load_dense_teacher_config(
        TEACHER_CONFIG
    ).fingerprint()

    _assert_raises(
        ValueError,
        lambda: replace(config, strict_xml_annotation_dir=False).validate(),
    )
    _assert_raises(
        ValueError,
        lambda: replace(config, enable_contour_fallback=True).validate(),
    )


def test_manifest_parse_and_strict_preflight(temp_dir: Path) -> None:
    video_name = "synthetic_video"
    pseudo3d_h5 = temp_dir / "synthetic.h5"
    voc_root = temp_dir / "voc"
    annotation_dir = voc_root / video_name / "annotations_renamed"
    annotation_dir.mkdir(parents=True)
    _write_pseudo3d_h5(pseudo3d_h5, [10, 20])
    _write_voc_xml(
        annotation_dir / f"{video_name}_00011.xml",
        frame_number=11,
    )
    _write_voc_xml(
        annotation_dir / f"{video_name}_00021.xml",
        frame_number=21,
    )

    manifest = temp_dir / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "video_name": video_name,
                "pseudo3d_h5": pseudo3d_h5.name,
                "voc_xml_root": "voc",
                "split": "smoke",
                "enabled": "true",
                "notes": "relative paths",
            },
            {
                "video_name": "disabled_video",
                "pseudo3d_h5": "missing.h5",
                "voc_xml_root": "missing_voc",
                "split": "train",
                "enabled": "false",
                "notes": "disabled rows are recorded but not audited",
            },
        ],
    )

    teacher = load_dense_teacher_config(TEACHER_CONFIG)
    items = load_stage4_sweep_manifest(manifest)
    assert len(items) == 2
    assert items[0].pseudo3d_h5 == pseudo3d_h5.resolve()
    assert items[0].strict_annotation_dir(teacher) == annotation_dir.resolve()
    rows = audit_manifest(items, teacher)
    assert [row.status for row in rows] == ["ok", "disabled"]
    assert rows[0].num_frames == 2
    assert rows[0].num_xml_files == 2
    assert rows[0].num_matched_xml_frames == 2
    assert rows[0].num_matched_bboxes == 2


def test_manifest_validation_failures(temp_dir: Path) -> None:
    base = {
        "video_name": "video_a",
        "pseudo3d_h5": "a.h5",
        "voc_xml_root": "voc",
        "split": "train",
        "enabled": "true",
        "notes": "",
    }

    invalid_split = temp_dir / "invalid_split.csv"
    _write_manifest(invalid_split, [{**base, "split": "val"}])
    _assert_raises(ValueError, lambda: load_stage4_sweep_manifest(invalid_split))

    duplicate = temp_dir / "duplicate.csv"
    _write_manifest(
        duplicate,
        [base, {**base, "pseudo3d_h5": "b.h5", "split": "validation"}],
    )
    _assert_raises(ValueError, lambda: load_stage4_sweep_manifest(duplicate))

    no_enabled = temp_dir / "no_enabled.csv"
    _write_manifest(no_enabled, [{**base, "enabled": "false"}])
    _assert_raises(ValueError, lambda: load_stage4_sweep_manifest(no_enabled))

    malformed_teacher = DenseTeacherConfig().to_dict()
    malformed_teacher["fallback"]["enabled"] = True
    _assert_raises(
        ValueError,
        lambda: dense_teacher_config_from_mapping(malformed_teacher),
    )


def test_preflight_rejects_unmatched_strict_xml(temp_dir: Path) -> None:
    video_name = "unmatched_video"
    pseudo3d_h5 = temp_dir / "unmatched.h5"
    voc_root = temp_dir / "voc_unmatched"
    annotation_dir = voc_root / video_name / "annotations_renamed"
    annotation_dir.mkdir(parents=True)
    _write_pseudo3d_h5(pseudo3d_h5, [10])
    _write_voc_xml(annotation_dir / f"{video_name}_00999.xml", frame_number=999)

    manifest = temp_dir / "unmatched_manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "video_name": video_name,
                "pseudo3d_h5": str(pseudo3d_h5),
                "voc_xml_root": str(voc_root),
                "split": "smoke",
                "enabled": "true",
                "notes": "",
            }
        ],
    )
    rows = audit_manifest(
        load_stage4_sweep_manifest(manifest),
        load_dense_teacher_config(TEACHER_CONFIG),
    )
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "No strict VOC BBox matched" in rows[0].error


def main() -> None:
    test_teacher_config_is_fixed_and_fingerprinted()
    print("[OK] fixed teacher config and fingerprint")

    with tempfile.TemporaryDirectory(prefix="stage4_sweep_manifest_") as temp:
        temp_dir = Path(temp)
        test_manifest_parse_and_strict_preflight(temp_dir)
        print("[OK] manifest parsing and strict H5/XML preflight")
        test_manifest_validation_failures(temp_dir)
        print("[OK] manifest and teacher validation failures")
        test_preflight_rejects_unmatched_strict_xml(temp_dir)
        print("[OK] unmatched strict XML rejection")

    print("Stage 4 sampling sweep manifest checks passed.")


if __name__ == "__main__":
    main()

