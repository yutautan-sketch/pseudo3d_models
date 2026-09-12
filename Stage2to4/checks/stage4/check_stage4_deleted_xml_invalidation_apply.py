from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pseudo3d.analysis.build_stage4_deleted_xml_invalidation_manifest import (
    ACTION,
    REASON_CODE,
    XmlFrameInvalidation,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256
from pseudo3d.annotation.apply_deleted_xml_invalidations import (
    FRAME_AUTHORITY,
    OUTPUT_TOKEN,
    SOURCE_TOKEN,
    XmlInvalidationApplyError,
    apply_invalidations_to_h5,
)


def _text(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _write_source(
    path: Path,
    missing_xml: Path,
    *,
    video: str = "synthetic_xml_invalidation",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    present_xml = missing_xml.parent / f"{video}_00001.xml"
    present_xml.parent.mkdir(parents=True, exist_ok=True)
    present_xml.write_text("<annotation/>\n", encoding="utf-8")
    labels = np.asarray([1, -1, 0, 1, 1, -1, 0, -1, 1], dtype=np.int8)
    with h5py.File(path, "w") as handle:
        handle.attrs["video_name"] = video
        handle.attrs["num_frames"] = 3
        handle.attrs["contour_teacher_schema"] = SOURCE_TOKEN
        handle.attrs["num_voc_bboxes"] = 3
        handle.attrs["num_voc_bbox_frames"] = 2
        handle.attrs["num_labeled_points"] = 4
        handle.attrs["manual_review_fullvideo_reviewed_video"] = True
        handle.attrs["manual_review_fullvideo_applied_bbox_count"] = 3

        point = handle.create_group("point_cloud")
        point.create_dataset("points", data=np.arange(27, dtype=np.float32).reshape(9, 3))
        point.create_dataset("colors", data=np.arange(27, dtype=np.uint8).reshape(9, 3))
        point.create_dataset("frame_order", data=np.repeat(np.arange(3), 3).astype(np.int32))
        point.create_dataset("source_flags", data=np.zeros(9, dtype=np.uint16))
        point.attrs["preserved"] = "point-cloud"

        annotation = handle.create_group("annotation")
        annotation.create_dataset("point_label", data=labels, compression="gzip")
        annotation.create_dataset("valid_mask", data=labels != -1, compression="gzip")
        annotation.attrs["label_source"] = SOURCE_TOKEN

        frame = handle.create_group("frame_annotation")
        frame.create_dataset("frame_order", data=np.asarray([0, 1, 1], dtype=np.int32))
        frame.create_dataset("frame_index", data=np.asarray([0, 1, 1], dtype=np.int64))
        frame.create_dataset("bbox_index", data=np.asarray([0, 0, 1], dtype=np.int32))
        frame.create_dataset(
            "xml_path",
            data=_text([str(present_xml), str(missing_xml), str(missing_xml)]),
        )
        frame.create_dataset("valid_contour", data=np.asarray([True, True, True]))
        frame.create_dataset(
            "selected_contour_source",
            data=_text(["global", "manual_cvat_fullvideo_authoritative", "local_percentile"]),
        )
        frame.create_dataset(
            "annotation_reason",
            data=_text(["ok", "cvat_snapshot_authoritative", "ok"]),
        )
        frame.create_dataset("fallback_used", data=np.asarray([False, False, True]))
        frame.create_dataset(
            "bbox_local_xyxy",
            data=np.asarray(
                [[1, 1, 5, 5], [2, 2, 6, 6], [7, 2, 10, 6]],
                dtype=np.float32,
            ),
        )
        frame.create_dataset("num_labeled_points", data=np.asarray([1, 1, 1], dtype=np.int32))
        frame.attrs["preserved"] = "frame-annotation"

        manual = handle.create_group("manual_review_fullvideo")
        manual.create_dataset(
            "frame_stem",
            data=_text(
                [
                    f"{video}__fo00000__fi00000000",
                    f"{video}__fo00001__fi00000001",
                    f"{video}__fo00001__fi00000001",
                ]
            ),
        )
        manual.create_dataset("frame_order", data=np.asarray([0, 1, 1], dtype=np.int32))
        manual.create_dataset("frame_index", data=np.asarray([0, 1, 1], dtype=np.int64))
        manual.create_dataset("bbox_index", data=np.asarray([0, 0, 1], dtype=np.int32))
        manual.create_dataset(
            "decision", data=_text(["manual_positive", "manual_positive", "manual_empty"])
        )
        manual.create_dataset("mask_sha256", data=_text(["a", "b", "c"]))
        manual.create_dataset(
            "before_positive_points", data=np.asarray([1, 2, 2], dtype=np.int64)
        )
        manual.create_dataset(
            "after_positive_points", data=np.asarray([1, 1, 1], dtype=np.int64)
        )
        manual.create_dataset(
            "outside_bbox_positive_pixels", data=np.asarray([0, 0, 0], dtype=np.int64)
        )
        manual.attrs["reviewed_video"] = True

        frames = handle.create_group("manual_review_fullvideo_frames")
        frames.create_dataset("frame_order", data=np.arange(3, dtype=np.int32))
        frames.create_dataset("frame_index", data=np.arange(3, dtype=np.int64))
        frames.create_dataset(
            "frame_stem",
            data=_text(
                [f"{video}__fo{value:05d}__fi{value:08d}" for value in range(3)]
            ),
        )
        frames.create_dataset("frame_role", data=_text(["context_only"] * 3))
        frames.create_dataset(
            "cvat_mask_status", data=_text(["positive", "positive", "empty"])
        )
        frames.create_dataset(
            "label_authority", data=_text(["cvat_snapshot"] * 3)
        )
        frames.create_dataset(
            "cvat_mask_positive_pixels", data=np.asarray([5, 23, 0], dtype=np.int64)
        )
        frames.create_dataset(
            "before_positive_points", data=np.asarray([1, 2, 1], dtype=np.int64)
        )
        frames.create_dataset(
            "positive_points", data=np.asarray([1, 2, 1], dtype=np.int64)
        )
        frames.create_dataset(
            "ignore_points", data=np.asarray([1, 1, 1], dtype=np.int64)
        )
        frames.create_dataset(
            "background_points", data=np.asarray([1, 0, 1], dtype=np.int64)
        )
        frames.create_dataset(
            "outside_bbox_positive_pixels", data=np.zeros(3, dtype=np.int64)
        )
        frames.create_dataset(
            "positive_outside_cvat_mask_points", data=np.zeros(3, dtype=np.int64)
        )
        frames.attrs["reviewed_video"] = True

        preserved = handle.create_group("preserved_payload")
        preserved.create_dataset("values", data=np.asarray([3, 1, 4], dtype=np.int64))
        preserved.attrs["meaning"] = "unchanged"


def _item(missing_xml: Path, *, bbox_rows: int = 2) -> XmlFrameInvalidation:
    video = "synthetic_xml_invalidation"
    return XmlFrameInvalidation(
        schema_version=1,
        video_name=video,
        frame_order=1,
        frame_index=1,
        frame_stem=f"{video}__fo00001__fi00000001",
        expected_xml_name=missing_xml.name,
        expected_saved_bbox_rows=bbox_rows,
        action=ACTION,
        reason_code=REASON_CODE,
        notes="synthetic intentional deletion",
    )


def _manifest(path: Path, item: XmlFrameInvalidation) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(item.to_row()))
        writer.writeheader()
        writer.writerow(item.to_row())


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def test_apply_and_resume(root: Path) -> None:
    missing_xml = root / "voc" / "annotations" / "synthetic_xml_invalidation_00002.xml"
    source = root / f"source_{SOURCE_TOKEN}.h5"
    output_1 = root / f"output_1_{OUTPUT_TOKEN}.h5"
    output_2 = root / f"output_2_{OUTPUT_TOKEN}.h5"
    manifest = root / "invalidations.csv"
    item = _item(missing_xml)
    _write_source(source, missing_xml)
    _manifest(manifest, item)
    source_before = file_sha256(source)
    manifest_sha = file_sha256(manifest)
    result = apply_invalidations_to_h5(
        source_h5=source,
        output_h5=output_1,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        items=[item],
    )
    assert result["status"] == "processed"
    assert result["removed_positive_points"] == 2
    assert result["removed_ignore_points"] == 1
    assert result["removed_bbox_rows"] == 2
    assert file_sha256(source) == source_before

    with h5py.File(source, "r") as before, h5py.File(output_1, "r") as after:
        np.testing.assert_array_equal(
            before["point_cloud/points"][:], after["point_cloud/points"][:]
        )
        np.testing.assert_array_equal(
            before["point_cloud/colors"][:], after["point_cloud/colors"][:]
        )
        np.testing.assert_array_equal(
            before["preserved_payload/values"][:],
            after["preserved_payload/values"][:],
        )
        before_labels = before["annotation/point_label"][:]
        after_labels = after["annotation/point_label"][:]
        orders = after["point_cloud/frame_order"][:]
        np.testing.assert_array_equal(after_labels[orders == 1], [0, 0, 0])
        np.testing.assert_array_equal(
            after_labels[orders != 1], before_labels[orders != 1]
        )
        assert np.all(after["annotation/valid_mask"][:][orders == 1])
        np.testing.assert_array_equal(
            after["frame_annotation/frame_order"][:], [0]
        )
        assert len(after["manual_review_fullvideo/frame_order"]) == 1
        archive = after[
            "xml_annotation_invalidation/removed_frame_annotation"
        ]
        np.testing.assert_array_equal(archive["bbox_index"][:], [0, 1])
        assert _decode(archive["xml_path"][:]) == [str(missing_xml)] * 2
        manual_archive = after[
            "xml_annotation_invalidation/removed_manual_review_fullvideo"
        ]
        np.testing.assert_array_equal(manual_archive["bbox_index"][:], [0, 1])
        provenance = after["manual_review_fullvideo_frames"]
        assert _decode(provenance["label_authority"][:])[1] == FRAME_AUTHORITY
        assert _decode(provenance["invalidation_action"][:])[1] == ACTION
        assert _decode(provenance["invalidation_reason_code"][:])[1] == REASON_CODE
        assert int(provenance["positive_points"][1]) == 0
        assert int(provenance["ignore_points"][1]) == 0
        assert int(provenance["background_points"][1]) == 3
        assert _decode(provenance["cvat_mask_status"][:])[1] == "positive"
        assert int(provenance["cvat_mask_positive_pixels"][1]) == 23
        assert after.attrs["contour_teacher_schema"] == OUTPUT_TOKEN
        assert int(after.attrs["num_voc_bboxes"]) == 1
        assert int(after.attrs["num_labeled_points"]) == 2

    first_hash = file_sha256(output_1)
    resumed = apply_invalidations_to_h5(
        source_h5=source,
        output_h5=output_1,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        items=[item],
        skip_existing=True,
    )
    assert resumed["status"] == "verified_existing"
    assert file_sha256(output_1) == first_hash
    apply_invalidations_to_h5(
        source_h5=source,
        output_h5=output_2,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        items=[item],
    )
    assert file_sha256(output_2) == first_hash
    print(
        "[OK] all-background tombstone, BBox removal/archive, provenance, "
        "non-target preservation, and deterministic resume"
    )

    unaffected_source = root / f"unaffected_source_{SOURCE_TOKEN}.h5"
    unaffected_output = root / f"unaffected_output_{OUTPUT_TOKEN}.h5"
    _write_source(
        unaffected_source,
        root / "unaffected" / "missing.xml",
        video="synthetic_unaffected",
    )
    apply_invalidations_to_h5(
        source_h5=unaffected_source,
        output_h5=unaffected_output,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        items=[],
    )
    with h5py.File(unaffected_source, "r") as before, h5py.File(
        unaffected_output, "r"
    ) as after:
        np.testing.assert_array_equal(
            before["annotation/point_label"][:], after["annotation/point_label"][:]
        )
        np.testing.assert_array_equal(
            before["annotation/valid_mask"][:], after["annotation/valid_mask"][:]
        )
        for name in before["frame_annotation"]:
            np.testing.assert_array_equal(
                before[f"frame_annotation/{name}"][:],
                after[f"frame_annotation/{name}"][:],
            )
        assert after.attrs["contour_teacher_schema"] == OUTPUT_TOKEN
        assert len(after["xml_annotation_invalidation/frame_order"]) == 0
    print("[OK] unaffected-video v5-to-v6 propagation without label/BBox changes")


def test_rejections_without_partial_output(root: Path) -> None:
    missing_xml = root / "voc" / "annotations" / "synthetic_xml_invalidation_00002.xml"
    source = root / f"source_{SOURCE_TOKEN}.h5"
    manifest = root / "invalidations.csv"
    item = _item(missing_xml)
    _write_source(source, missing_xml)
    _manifest(manifest, item)
    manifest_sha = file_sha256(manifest)

    bad_bbox = _item(missing_xml, bbox_rows=1)
    bad_bbox_manifest = root / "bad_bbox_manifest.csv"
    _manifest(bad_bbox_manifest, bad_bbox)
    cases = [
        (
            [bad_bbox],
            bad_bbox_manifest,
            file_sha256(bad_bbox_manifest),
            "Saved BBox count mismatch",
        ),
        (
            [item, item],
            manifest,
            manifest_sha,
            "Supplied invalidations differ",
        ),
    ]
    for index, (items, case_manifest, case_sha, expected) in enumerate(cases):
        output = root / f"failed_{index}.h5"
        try:
            apply_invalidations_to_h5(
                source_h5=source,
                output_h5=output,
                manifest_path=case_manifest,
                manifest_sha256=case_sha,
                items=items,
            )
        except XmlInvalidationApplyError as exc:
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected failure containing {expected!r}")
        assert not output.exists()

    missing_xml.parent.mkdir(parents=True, exist_ok=True)
    missing_xml.write_text("<annotation/>\n", encoding="utf-8")
    restored_output = root / "restored.h5"
    try:
        apply_invalidations_to_h5(
            source_h5=source,
            output_h5=restored_output,
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            items=[item],
        )
    except XmlInvalidationApplyError as exc:
        assert "exists again" in str(exc)
    else:
        raise AssertionError("Restored XML must be rejected")
    assert not restored_output.exists()
    print("[OK] BBox mismatch, duplicate frame, and restored XML rejection")


def test_existing_output_provenance_rejection(root: Path) -> None:
    missing_xml = root / "voc" / "annotations" / "synthetic_xml_invalidation_00002.xml"
    source = root / f"source_{SOURCE_TOKEN}.h5"
    output = root / f"output_{OUTPUT_TOKEN}.h5"
    manifest = root / "invalidations.csv"
    item = _item(missing_xml)
    _write_source(source, missing_xml)
    _manifest(manifest, item)
    manifest_sha = file_sha256(manifest)
    apply_invalidations_to_h5(
        source_h5=source,
        output_h5=output,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        items=[item],
    )
    with h5py.File(output, "r+") as handle:
        handle["xml_annotation_invalidation"].attrs["manifest_sha256"] = "wrong"
    try:
        apply_invalidations_to_h5(
            source_h5=source,
            output_h5=output,
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            items=[item],
            skip_existing=True,
        )
    except XmlInvalidationApplyError as exc:
        assert "provenance mismatch" in str(exc)
    else:
        raise AssertionError("Corrupt existing provenance must be rejected")
    print("[OK] corrupt verified-resume provenance rejection")


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="stage4_deleted_xml_invalidation_apply_"
    ) as directory:
        root = Path(directory)
        test_apply_and_resume(root / "apply")
        test_rejections_without_partial_output(root / "reject")
        test_existing_output_provenance_rejection(root / "resume")
    print("Stage 4 deleted-XML invalidation Step 3 synthetic checks passed.")


if __name__ == "__main__":
    main()
