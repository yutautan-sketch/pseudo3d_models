from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.stage4_sampling_sweep_config import (
    DenseTeacherConfig,
    Stage4SweepManifestItem,
    file_sha256,
    load_dense_teacher_config,
    load_stage4_sweep_manifest,
)


REQUIRED_PSEUDO3D_DATASETS = (
    "local_encoder_images",
    "frame_indices",
)


@dataclass(frozen=True)
class ManifestAuditRow:
    row_number: int
    video_name: str
    split: str
    enabled: bool
    status: str
    pseudo3d_h5: str
    voc_xml_root: str
    strict_annotation_dir: str
    num_frames: int | None
    local_image_shape: str
    num_xml_files: int | None
    num_matched_xml_frames: int | None
    num_matched_bboxes: int | None
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_number": self.row_number,
            "video_name": self.video_name,
            "split": self.split,
            "enabled": self.enabled,
            "status": self.status,
            "pseudo3d_h5": self.pseudo3d_h5,
            "voc_xml_root": self.voc_xml_root,
            "strict_annotation_dir": self.strict_annotation_dir,
            "num_frames": self.num_frames,
            "local_image_shape": self.local_image_shape,
            "num_xml_files": self.num_xml_files,
            "num_matched_xml_frames": self.num_matched_xml_frames,
            "num_matched_bboxes": self.num_matched_bboxes,
            "error": self.error,
        }


def _audit_enabled_item(
    item: Stage4SweepManifestItem,
    teacher: DenseTeacherConfig,
) -> ManifestAuditRow:
    import h5py
    import numpy as np

    from pseudo3d.annotation.annotate_pseudo3d_point_cloud import load_voc_bboxes

    annotation_dir = item.strict_annotation_dir(teacher)
    num_frames: int | None = None
    local_shape = ""
    num_xml_files: int | None = None
    num_matched_xml_frames: int | None = None
    num_matched_bboxes: int | None = None

    try:
        if not item.pseudo3d_h5.is_file():
            raise FileNotFoundError(f"pseudo3d_h5 not found: {item.pseudo3d_h5}")
        if not item.voc_xml_root.is_dir():
            raise FileNotFoundError(f"voc_xml_root not found: {item.voc_xml_root}")
        if not annotation_dir.is_dir():
            raise FileNotFoundError(
                "strict annotation directory not found: "
                f"{annotation_dir}"
            )

        xml_files = sorted(annotation_dir.glob("*.xml"))
        num_xml_files = len(xml_files)
        if not xml_files:
            raise ValueError(f"strict annotation directory has no XML files: {annotation_dir}")

        with h5py.File(item.pseudo3d_h5, "r") as handle:
            missing = [
                key for key in REQUIRED_PSEUDO3D_DATASETS if key not in handle
            ]
            if missing:
                raise KeyError(f"pseudo3d H5 missing datasets: {missing}")
            local_dataset = handle["local_encoder_images"]
            frame_dataset = handle["frame_indices"]
            local_shape_tuple = tuple(int(value) for value in local_dataset.shape)
            local_shape = str(local_shape_tuple)
            if local_dataset.ndim not in {3, 4}:
                raise ValueError(
                    "local_encoder_images must have shape [N,H,W] or a "
                    f"supported 4D channel layout, got {local_shape_tuple}"
                )
            if not local_shape_tuple or local_shape_tuple[0] <= 0:
                raise ValueError("local_encoder_images contains no frames")
            if frame_dataset.ndim != 1:
                raise ValueError(
                    f"frame_indices must be 1D, got shape={frame_dataset.shape}"
                )
            if int(frame_dataset.shape[0]) != int(local_shape_tuple[0]):
                raise ValueError(
                    "frame count mismatch: local_encoder_images="
                    f"{local_shape_tuple[0]}, frame_indices={frame_dataset.shape[0]}"
                )
            frame_indices = frame_dataset[:].astype(np.int64)

        num_frames = int(frame_indices.size)
        if np.unique(frame_indices).size != frame_indices.size:
            raise ValueError("frame_indices contains duplicates")

        voc_bboxes = load_voc_bboxes(
            item.voc_xml_root,
            video_name=item.video_name,
            frame_indices=frame_indices,
            xml_frame_number_offsets=teacher.xml_frame_number_offsets,
            xml_frame_id_source=teacher.xml_frame_id_source,
            xml_annotation_dir_name=teacher.xml_annotation_dir_name,
            strict_xml_annotation_dir=teacher.strict_xml_annotation_dir,
        )
        num_matched_xml_frames = len(voc_bboxes)
        num_matched_bboxes = sum(len(boxes) for boxes in voc_bboxes.values())
        matched_paths = {
            str(box.xml_path)
            for boxes in voc_bboxes.values()
            for box in boxes
        }
        if len(matched_paths) != num_matched_xml_frames:
            raise ValueError(
                "One strict XML file was matched to multiple pseudo3D frames"
            )
        if num_matched_xml_frames == 0 or num_matched_bboxes == 0:
            raise ValueError(
                "No strict VOC BBox matched pseudo3D frame_indices using the "
                "fixed teacher XML rules"
            )

        return ManifestAuditRow(
            row_number=item.row_number,
            video_name=item.video_name,
            split=item.split,
            enabled=True,
            status="ok",
            pseudo3d_h5=str(item.pseudo3d_h5),
            voc_xml_root=str(item.voc_xml_root),
            strict_annotation_dir=str(annotation_dir),
            num_frames=num_frames,
            local_image_shape=local_shape,
            num_xml_files=num_xml_files,
            num_matched_xml_frames=num_matched_xml_frames,
            num_matched_bboxes=num_matched_bboxes,
            error="",
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        return ManifestAuditRow(
            row_number=item.row_number,
            video_name=item.video_name,
            split=item.split,
            enabled=True,
            status="failed",
            pseudo3d_h5=str(item.pseudo3d_h5),
            voc_xml_root=str(item.voc_xml_root),
            strict_annotation_dir=str(annotation_dir),
            num_frames=num_frames,
            local_image_shape=local_shape,
            num_xml_files=num_xml_files,
            num_matched_xml_frames=num_matched_xml_frames,
            num_matched_bboxes=num_matched_bboxes,
            error=f"{type(exc).__name__}: {exc}",
        )


def audit_manifest(
    items: list[Stage4SweepManifestItem],
    teacher: DenseTeacherConfig,
) -> list[ManifestAuditRow]:
    rows = []
    for item in items:
        if item.enabled:
            rows.append(_audit_enabled_item(item, teacher))
        else:
            rows.append(
                ManifestAuditRow(
                    row_number=item.row_number,
                    video_name=item.video_name,
                    split=item.split,
                    enabled=False,
                    status="disabled",
                    pseudo3d_h5=str(item.pseudo3d_h5),
                    voc_xml_root=str(item.voc_xml_root),
                    strict_annotation_dir=str(item.strict_annotation_dir(teacher)),
                    num_frames=None,
                    local_image_shape="",
                    num_xml_files=None,
                    num_matched_xml_frames=None,
                    num_matched_bboxes=None,
                    error="",
                )
            )
    return rows


def _write_audit_csv(path: Path, rows: list[ManifestAuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].to_dict())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def _write_metadata_json(
    path: Path,
    *,
    manifest_path: Path,
    teacher_path: Path,
    teacher: DenseTeacherConfig,
    rows: list[ManifestAuditRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(
        row.split for row in rows if row.enabled and row.status == "ok"
    )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "teacher_config_path": str(teacher_path.resolve()),
        "teacher_config_fingerprint": teacher.fingerprint(),
        "teacher_config": teacher.to_dict(),
        "rows_total": len(rows),
        "rows_enabled": sum(row.enabled for row in rows),
        "rows_ok": sum(row.status == "ok" for row in rows),
        "rows_failed": sum(row.status == "failed" for row in rows),
        "rows_disabled": sum(row.status == "disabled" for row in rows),
        "split_counts": dict(sorted(split_counts.items())),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Stage 4 sampling-sweep manifest, fixed dense-teacher "
            "configuration, pseudo3D H5 schema, and strict VOC XML mapping."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--audit_csv", type=Path, default=None)
    parser.add_argument("--metadata_json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    teacher = load_dense_teacher_config(args.teacher_config)
    items = load_stage4_sweep_manifest(args.manifest)
    rows = audit_manifest(items, teacher)

    if args.audit_csv is not None:
        _write_audit_csv(args.audit_csv, rows)
    if args.metadata_json is not None:
        _write_metadata_json(
            args.metadata_json,
            manifest_path=args.manifest,
            teacher_path=args.teacher_config,
            teacher=teacher,
            rows=rows,
        )

    print("Stage 4 sampling sweep manifest preflight")
    print(f"manifest                  : {args.manifest}")
    print(f"teacher_config            : {args.teacher_config}")
    print(f"teacher_fingerprint       : {teacher.fingerprint()}")
    print(f"rows                       : {len(rows)}")
    for row in rows:
        detail = (
            f"frames={row.num_frames}, matched_xml={row.num_matched_xml_frames}, "
            f"bboxes={row.num_matched_bboxes}"
            if row.status == "ok"
            else row.error
        )
        print(
            f"[{row.status.upper()}] row={row.row_number} "
            f"split={row.split} video={row.video_name} {detail}"
        )

    failed = [row for row in rows if row.status == "failed"]
    split_counts = Counter(row.split for row in rows if row.status == "ok")
    print("\nManifest summary")
    print(f"  enabled : {sum(row.enabled for row in rows)}")
    print(f"  ok      : {sum(row.status == 'ok' for row in rows)}")
    print(f"  failed  : {len(failed)}")
    print(f"  disabled: {sum(row.status == 'disabled' for row in rows)}")
    print(f"  splits  : {dict(sorted(split_counts.items()))}")
    if failed:
        raise SystemExit(1)
    print("Stage 4 sampling sweep manifest preflight passed.")


if __name__ == "__main__":
    main()

