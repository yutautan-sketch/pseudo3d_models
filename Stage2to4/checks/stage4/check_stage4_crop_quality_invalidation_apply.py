from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256
from pseudo3d.analysis.validate_stage4_crop_quality_manifests import (
    ACTION,
    MANIFEST_FIELDS,
    REASON_CODE,
    CropQualityInvalidation,
)
from pseudo3d.annotation.apply_crop_quality_invalidations import (
    FRAME_AUTHORITY,
    GROUP_NAME,
    OUTPUT_TOKEN,
    SOURCE_TOKEN,
    CropQualityApplyError,
    apply_crop_quality_invalidations_to_h5,
)


def _text(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _item(video: str = "synthetic_crop_quality", *, stray: int = 2) -> CropQualityInvalidation:
    return CropQualityInvalidation(
        schema_version=1,
        video_name=video,
        frame_order=1,
        frame_index=1,
        frame_stem=f"{video}__fo00001__fi00000001",
        expected_saved_bbox_rows=1,
        expected_stray_ignore_points=stray,
        expected_crop_status="fully_outside_crop",
        expected_visible_fraction=0.0,
        action=ACTION,
        reason_code=REASON_CODE,
        recoverable=True,
        notes="synthetic fully-outside crop",
    )


def _write_manifest(path: Path, item: CropQualityInvalidation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerow(item.to_row())


def _write_source(path: Path, xml_path: Path, *, video: str = "synthetic_crop_quality") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<annotation/>\n", encoding="utf-8")
    labels = np.asarray([1, -1, 0, 0, -1, -1], dtype=np.int8)
    with h5py.File(path, "w") as handle:
        handle.attrs["video_name"] = video
        handle.attrs["contour_teacher_schema"] = SOURCE_TOKEN
        handle.attrs["num_labeled_points"] = 1
        handle.attrs["num_voc_bboxes"] = 2
        handle.attrs["num_voc_bbox_frames"] = 2
        point = handle.create_group("point_cloud")
        point.create_dataset("points", data=np.arange(18, dtype=np.float32).reshape(6, 3))
        point.create_dataset("colors", data=np.arange(18, dtype=np.uint8).reshape(6, 3))
        point.create_dataset("frame_order", data=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32))
        annotation = handle.create_group("annotation")
        annotation.create_dataset("point_label", data=labels, compression="gzip")
        annotation.create_dataset("valid_mask", data=labels != -1, compression="gzip")
        annotation.attrs["label_source"] = SOURCE_TOKEN
        frame = handle.create_group("frame_annotation")
        frame.create_dataset("frame_order", data=np.asarray([0, 1], dtype=np.int32))
        frame.create_dataset("frame_index", data=np.asarray([0, 1], dtype=np.int64))
        frame.create_dataset("bbox_index", data=np.asarray([0, 0], dtype=np.int32))
        frame.create_dataset("xml_path", data=_text([str(xml_path), str(xml_path)]))
        frame.create_dataset(
            "bbox_local_xyxy",
            data=np.asarray([[1, 1, 5, 5], [255, 10, 255, 20]], dtype=np.float32),
        )
        frame.create_dataset("valid_contour", data=np.asarray([True, False]))
        frame.create_dataset(
            "selected_contour_source",
            data=_text(["manual_cvat_fullvideo_snapshot_authoritative_v1", "manual_cvat_fullvideo_snapshot_authoritative_empty_v1"]),
        )
        frame.create_dataset(
            "annotation_reason",
            data=_text(["cvat_snapshot_authoritative", "cvat_snapshot_authoritative_empty"]),
        )
        frame.create_dataset("fallback_used", data=np.asarray([False, False]))

        manual = handle.create_group("manual_review_fullvideo")
        manual.create_dataset("frame_order", data=np.asarray([1], dtype=np.int32))
        manual.create_dataset("frame_index", data=np.asarray([1], dtype=np.int64))
        manual.create_dataset("bbox_index", data=np.asarray([0], dtype=np.int32))
        manual.create_dataset("decision", data=_text(["manual_empty"]))
        manual.create_dataset("outside_bbox_positive_pixels", data=np.asarray([0], dtype=np.int64))

        frames = handle.create_group("manual_review_fullvideo_frames")
        frames.create_dataset("frame_order", data=np.asarray([0, 1], dtype=np.int32))
        frames.create_dataset("frame_index", data=np.asarray([0, 1], dtype=np.int64))
        frames.create_dataset(
            "frame_stem",
            data=_text([f"{video}__fo00000__fi00000000", f"{video}__fo00001__fi00000001"]),
        )
        frames.create_dataset("label_authority", data=_text(["cvat_snapshot", "cvat_snapshot"]))
        frames.create_dataset("cvat_mask_status", data=_text(["positive", "empty"]))
        frames.create_dataset("cvat_mask_positive_pixels", data=np.asarray([10, 0], dtype=np.int64))
        frames.create_dataset("positive_points", data=np.asarray([1, 0], dtype=np.int64))
        frames.create_dataset("ignore_points", data=np.asarray([1, 2], dtype=np.int64))
        frames.create_dataset("background_points", data=np.asarray([1, 1], dtype=np.int64))
        frames.create_dataset("positive_outside_cvat_mask_points", data=np.zeros(2, dtype=np.int64))

        previous = handle.create_group("xml_annotation_invalidation")
        previous.attrs["preserved"] = "v6-history"
        previous.create_dataset("frame_order", data=np.asarray([7], dtype=np.int32))
        payload = handle.create_group("preserved_payload")
        payload.create_dataset("values", data=np.asarray([3, 1, 4], dtype=np.int64))


def test_apply_resume_and_unaffected(root: Path) -> None:
    video = "synthetic_crop_quality"
    item = _item(video)
    manifest = root / "crop_invalidations.csv"
    source = root / f"source_{SOURCE_TOKEN}.h5"
    output = root / f"output_{OUTPUT_TOKEN}.h5"
    output_2 = root / f"output_2_{OUTPUT_TOKEN}.h5"
    xml_path = root / "voc" / f"{video}_00002.xml"
    _write_manifest(manifest, item)
    _write_source(source, xml_path, video=video)
    source_sha = file_sha256(source)
    result = apply_crop_quality_invalidations_to_h5(
        source_h5=source,
        output_h5=output,
        manifest_path=manifest,
        manifest_sha256=file_sha256(manifest),
        items=[item],
    )
    assert result["status"] == "processed"
    assert result["removed_ignore_points"] == 2
    assert result["removed_positive_points"] == 0
    assert result["removed_bbox_rows"] == 1
    assert file_sha256(source) == source_sha
    with h5py.File(source, "r") as before, h5py.File(output, "r") as after:
        orders = after["point_cloud/frame_order"][:]
        labels = after["annotation/point_label"][:]
        np.testing.assert_array_equal(labels[orders == 1], [0, 0, 0])
        np.testing.assert_array_equal(
            labels[orders == 0], before["annotation/point_label"][:][orders == 0]
        )
        assert np.all(after["annotation/valid_mask"][:][orders == 1])
        np.testing.assert_array_equal(after["frame_annotation/frame_order"][:], [0])
        assert len(after["manual_review_fullvideo/frame_order"]) == 0
        assert int(after["xml_annotation_invalidation/frame_order"][0]) == 7
        assert after["xml_annotation_invalidation"].attrs["preserved"] == "v6-history"
        group = after[GROUP_NAME]
        assert _decode(group["removed_frame_annotation/xml_path"][:]) == [str(xml_path)]
        assert len(group["removed_manual_review_fullvideo/frame_order"]) == 1
        assert len(group["source_manual_review_fullvideo_frames/frame_order"]) == 1
        frames = after["manual_review_fullvideo_frames"]
        assert _decode(frames["label_authority"][:])[1] == FRAME_AUTHORITY
        assert _decode(frames["crop_invalidation_action"][:])[1] == ACTION
        assert after.attrs["contour_teacher_schema"] == OUTPUT_TOKEN
        assert int(after.attrs["num_voc_bboxes"]) == 1

    first_hash = file_sha256(output)
    resumed = apply_crop_quality_invalidations_to_h5(
        source_h5=source,
        output_h5=output,
        manifest_path=manifest,
        manifest_sha256=file_sha256(manifest),
        items=[item],
        skip_existing=True,
    )
    assert resumed["status"] == "verified_existing"
    assert file_sha256(output) == first_hash
    apply_crop_quality_invalidations_to_h5(
        source_h5=source,
        output_h5=output_2,
        manifest_path=manifest,
        manifest_sha256=file_sha256(manifest),
        items=[item],
    )
    assert file_sha256(output_2) == first_hash

    unaffected_video = "synthetic_unaffected"
    unaffected_source = root / f"unaffected_{SOURCE_TOKEN}.h5"
    unaffected_output = root / f"unaffected_{OUTPUT_TOKEN}.h5"
    _write_source(unaffected_source, root / "voc" / "unaffected.xml", video=unaffected_video)
    apply_crop_quality_invalidations_to_h5(
        source_h5=unaffected_source,
        output_h5=unaffected_output,
        manifest_path=manifest,
        manifest_sha256=file_sha256(manifest),
        items=[],
    )
    with h5py.File(unaffected_source, "r") as before, h5py.File(unaffected_output, "r") as after:
        np.testing.assert_array_equal(before["annotation/point_label"][:], after["annotation/point_label"][:])
        np.testing.assert_array_equal(before["frame_annotation/frame_order"][:], after["frame_annotation/frame_order"][:])
        assert len(after[f"{GROUP_NAME}/frame_order"]) == 0
    print("[OK] crop tombstone, v6 preservation, provenance, deterministic resume, and unaffected propagation")


def test_rejections(root: Path) -> None:
    item = _item()
    manifest = root / "manifest.csv"
    source = root / f"source_{SOURCE_TOKEN}.h5"
    xml_path = root / "voc" / "source.xml"
    _write_manifest(manifest, item)
    _write_source(source, xml_path)
    bad_item = _item(stray=1)
    bad_manifest = root / "bad_manifest.csv"
    _write_manifest(bad_manifest, bad_item)
    cases = [
        (bad_manifest, [bad_item], "Stray ignore count mismatch"),
        (manifest, [item, item], "Supplied crop invalidations differ"),
    ]
    for index, (case_manifest, items, expected) in enumerate(cases):
        output = root / f"failed_{index}.h5"
        try:
            apply_crop_quality_invalidations_to_h5(
                source_h5=source,
                output_h5=output,
                manifest_path=case_manifest,
                manifest_sha256=file_sha256(case_manifest),
                items=items,
            )
        except CropQualityApplyError as exc:
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected failure containing {expected!r}")
        assert not output.exists()

    xml_path.unlink()
    output = root / "missing_xml.h5"
    try:
        apply_crop_quality_invalidations_to_h5(
            source_h5=source,
            output_h5=output,
            manifest_path=manifest,
            manifest_sha256=file_sha256(manifest),
            items=[item],
        )
    except CropQualityApplyError as exc:
        assert "must retain source XML" in str(exc)
    else:
        raise AssertionError("Missing XML must be rejected")
    assert not output.exists()
    print("[OK] stray-count, duplicate-row, and retained-XML safety gates")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_crop_quality_apply_") as directory:
        root = Path(directory)
        test_apply_resume_and_unaffected(root / "apply")
        test_rejections(root / "reject")
    print("Stage 4 crop-quality invalidation apply synthetic checks passed.")


if __name__ == "__main__":
    main()
