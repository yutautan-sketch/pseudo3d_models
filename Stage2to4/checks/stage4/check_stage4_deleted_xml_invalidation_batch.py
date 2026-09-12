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
CHECK_ROOT = Path(__file__).resolve().parent
for candidate in (REPO_ROOT, CHECK_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


from check_stage4_deleted_xml_invalidation_apply import (  # noqa: E402
    _item,
    _manifest,
    _write_source,
)
from pseudo3d.analysis.build_stage4_deleted_xml_invalidation_manifest import (  # noqa: E402
    invalidation_fingerprint,
    load_xml_invalidation_manifest,
)
from pseudo3d.analysis.stage4_sampling_sweep_config import file_sha256  # noqa: E402
from pseudo3d.annotation.apply_deleted_xml_invalidations import (  # noqa: E402
    OUTPUT_TOKEN,
    SOURCE_TOKEN,
)
from pseudo3d.batch.annotation.batch_apply_stage4_deleted_xml_invalidations import (  # noqa: E402
    XmlInvalidationBatchError,
    run_batch,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> tuple[argparse.Namespace, dict[str, Path]]:
    affected = "synthetic_xml_invalidation"
    unaffected = "synthetic_unaffected"
    excluded = "synthetic_excluded"
    source_root = root / "v5" / "annotated"
    voc_root = root / "voc"
    affected_missing = voc_root / affected / "annotations" / f"{affected}_00002.xml"
    affected_source = (
        source_root
        / affected
        / f"{affected}_pointcloud_annotated_{SOURCE_TOKEN}.h5"
    )
    unaffected_source = (
        source_root
        / unaffected
        / f"{unaffected}_pointcloud_annotated_{SOURCE_TOKEN}.h5"
    )
    _write_source(affected_source, affected_missing, video=affected)
    _write_source(
        unaffected_source,
        voc_root / unaffected / "annotations" / f"{unaffected}_00002.xml",
        video=unaffected,
    )

    train_manifest = root / "train_manifest_cropclean.csv"
    manifest_fields = [
        "video_name",
        "pseudo3d_h5",
        "voc_xml_root",
        "split",
        "enabled",
        "notes",
    ]
    _write_csv(
        train_manifest,
        manifest_fields,
        [
            {
                "video_name": video,
                "pseudo3d_h5": str(root / "pseudo" / f"{video}.h5"),
                "voc_xml_root": str(voc_root),
                "split": "train",
                "enabled": "false" if video == excluded else "true",
                "notes": "synthetic fixture",
            }
            for video in (affected, unaffected, excluded)
        ],
    )
    exclusion_manifest = root / "exclusions.csv"
    _write_csv(
        exclusion_manifest,
        [
            "video_name",
            "reason_code",
            "evidence_frames",
            "scope",
            "recoverable",
            "notes",
        ],
        [
            {
                "video_name": excluded,
                "reason_code": "synthetic_exclusion",
                "evidence_frames": "0",
                "scope": "stage4_teacher",
                "recoverable": "true",
                "notes": "synthetic excluded video",
            }
        ],
    )
    invalidation_manifest = root / "invalidations.csv"
    _manifest(invalidation_manifest, _item(affected_missing))
    invalidations = load_xml_invalidation_manifest(invalidation_manifest)
    step2_summary = root / "step2_summary.json"
    step2_summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "teacher_version": SOURCE_TOKEN,
                "invalidations": 1,
                "videos": 1,
                "expected_positive_points_to_remove": 2,
                "mixed_xml_presence_frames": 0,
                "input_files_unchanged": True,
                "h5_files_written": 0,
                "train_manifest_sha256": file_sha256(train_manifest),
                "exclusion_manifest_sha256": file_sha256(exclusion_manifest),
                "output_manifest_sha256": file_sha256(invalidation_manifest),
                "invalidation_fingerprint": invalidation_fingerprint(invalidations),
                "v5_candidate_h5_sha256": {
                    affected_source.name: file_sha256(affected_source)
                },
                "failure_rows": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = root / "v6"
    args = argparse.Namespace(
        train_manifest=train_manifest,
        exclusion_manifest=exclusion_manifest,
        invalidation_manifest=invalidation_manifest,
        step2_summary=step2_summary,
        source_annotated_root=source_root,
        output_annotated_root=output_root / "annotated",
        output_collected_root=output_root / "collected",
        summary_csv=output_root / "invalidation_summary.csv",
        summary_json=output_root / "invalidation_summary.json",
        expected_videos=2,
        expected_excluded_videos=1,
        expected_invalidations=1,
        expected_affected_videos=1,
        expected_positive_points=2,
        expected_removed_bbox_rows=2,
        skip_existing=False,
        overwrite=False,
    )
    return args, {
        "affected_source": affected_source,
        "unaffected_source": unaffected_source,
        "output_root": output_root,
        "invalidation_manifest": invalidation_manifest,
    }


def test_full_batch_build_and_verified_resume(root: Path) -> None:
    args, paths = _fixture(root)
    source_hashes = {
        name: file_sha256(path)
        for name, path in paths.items()
        if name.endswith("_source")
    }
    summary = run_batch(args)
    assert summary["videos"] == 2
    assert summary["affected_videos"] == 1
    assert summary["invalidated_frames"] == 1
    assert summary["removed_positive_points"] == 2
    assert summary["removed_bbox_rows"] == 2
    assert summary["status_counts"] == {"processed": 2}
    assert summary["collected_status_counts"] == {"copied": 2}
    saved_summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    assert saved_summary == summary
    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [row["video_name"] for row in summary_rows] == [
        "synthetic_unaffected",
        "synthetic_xml_invalidation",
    ]
    assert all(
        row["output_h5_sha256"] == row["collected_h5_sha256"]
        for row in summary_rows
    )

    output_root = paths["output_root"]
    annotated = sorted((output_root / "annotated").rglob(f"*{OUTPUT_TOKEN}.h5"))
    collected = sorted((output_root / "collected").glob(f"*{OUTPUT_TOKEN}.h5"))
    assert len(annotated) == len(collected) == 2
    by_name = {path.name: path for path in collected}
    for path in annotated:
        assert file_sha256(path) == file_sha256(by_name[path.name])

    affected_h5 = next(path for path in annotated if "synthetic_xml_invalidation" in path.name)
    unaffected_h5 = next(path for path in annotated if "synthetic_unaffected" in path.name)
    with h5py.File(affected_h5, "r") as handle:
        orders = handle["point_cloud/frame_order"][:]
        labels = handle["annotation/point_label"][:]
        np.testing.assert_array_equal(labels[orders == 1], [0, 0, 0])
        assert not np.any(handle["frame_annotation/frame_order"][:] == 1)
    with h5py.File(paths["unaffected_source"], "r") as before, h5py.File(
        unaffected_h5, "r"
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
    for name, expected in source_hashes.items():
        assert file_sha256(paths[name]) == expected

    args.skip_existing = True
    resumed = run_batch(args)
    assert resumed["status_counts"] == {"verified_existing": 2}
    assert resumed["collected_status_counts"] == {"verified_existing": 2}
    print(
        "[OK] 2-video v5-to-v6 batch, affected tombstone, unaffected "
        "propagation, collection, and verified resume"
    )


def test_corrupt_collected_rejection(root: Path) -> None:
    args, paths = _fixture(root)
    run_batch(args)
    collected = sorted((paths["output_root"] / "collected").glob("*.h5"))
    assert collected
    with collected[0].open("ab") as handle:
        handle.write(b"corrupt")
    args.skip_existing = True
    try:
        run_batch(args)
    except XmlInvalidationBatchError as exc:
        assert "Collected H5 differs" in str(exc), str(exc)
    else:
        raise AssertionError("Corrupt collected H5 must be rejected")
    print("[OK] corrupt collected artifact rejection")


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="stage4_deleted_xml_invalidation_batch_"
    ) as directory:
        root = Path(directory)
        test_full_batch_build_and_verified_resume(root / "batch")
        test_corrupt_collected_rejection(root / "corrupt")
    print("Stage 4 deleted-XML invalidation Step 4 synthetic checks passed.")


if __name__ == "__main__":
    main()
