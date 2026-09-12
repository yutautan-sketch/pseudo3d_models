from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py

from pseudo3d.export.export_stage4_point_label_visualization import (
    export_stage4_point_label_visualization,
    file_sha256,
    parse_color,
)


SUMMARY_FIELDS = (
    "status",
    "video_name",
    "annotated_h5",
    "annotated_h5_sha256",
    "output_dir",
    "automatic_contours_recomputed",
    "input_files_modified",
    "num_frames",
    "num_bbox_rows",
    "num_points",
    "positive_points",
    "ignore_points",
    "background_points",
    "cvat_mask_frames",
    "cvat_mask_pixels",
    "suppressed_cvat_mask_frames",
    "suppressed_cvat_mask_pixels",
    "cvat_label_authority",
    "cvat_authoritative_frames",
    "xml_invalidation_frames",
    "xml_invalidation_manifest_sha256",
    "xml_invalidation_fingerprint",
    "positive_outside_cvat_mask_points",
    "cvat_point_mask_mismatch_points",
    "source_counts_json",
    "error",
)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _video_name(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("video_name", path.parent.name)
    name = _decode(value).strip()
    if not name or name == "None":
        raise ValueError(f"Could not resolve video_name: {path}")
    return name


def collect_paths(
    *,
    annotated_dir: Path | None,
    h5_list: Path | None,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    if (annotated_dir is None) == (h5_list is None):
        raise ValueError("Specify exactly one of --annotated_dir and --h5_list")
    if h5_list is not None:
        if not Path(h5_list).is_file():
            raise FileNotFoundError(f"h5_list not found: {h5_list}")
        paths = []
        with Path(h5_list).open(encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value and not value.startswith("#"):
                    paths.append(Path(value).resolve())
    else:
        root = Path(annotated_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"annotated_dir not found: {root}")
        globber = root.rglob if recursive else root.glob
        paths = [path.resolve() for path in sorted(globber(pattern)) if path.is_file()]
    if not paths:
        raise FileNotFoundError("No annotated H5 files were found")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate annotated H5 paths")
    return paths


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _existing_summary(
    output_dir: Path,
    input_h5: Path,
    *,
    require_cvat_masks: bool,
    require_cvat_authoritative: bool,
    require_xml_invalidation: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Existing output lacks summary.json: {output_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        raise ValueError(f"Existing output status is not ok: {output_dir}")
    if summary.get("automatic_contours_recomputed") is not False:
        raise ValueError(f"Existing output is not saved-label visualization: {output_dir}")
    if require_cvat_masks and summary.get("cvat_masks_requested") is not True:
        raise ValueError(f"Existing output lacks requested CVAT mask overlay: {output_dir}")
    if require_cvat_authoritative:
        authority = summary.get("cvat_label_authority")
        if authority not in {
            "cvat_snapshot",
            "cvat_snapshot_with_xml_invalidation",
            "inherited",
            "inherited_with_xml_invalidation",
        }:
            raise ValueError(
                f"Existing output lacks authoritative label contract: {output_dir}"
            )
        if int(summary.get("positive_outside_cvat_mask_points", -1)) != 0:
            raise ValueError(
                f"Existing output has positive points outside CVAT mask: {output_dir}"
            )
        if int(summary.get("cvat_point_mask_mismatch_points", -1)) != 0:
            raise ValueError(
                f"Existing output has CVAT/point-label mismatches: {output_dir}"
            )
        authoritative_frames = int(
            summary.get("cvat_authoritative_frames", -1)
        )
        cvat_mask_frames = int(summary.get("cvat_mask_frames", -1))
        suppressed_frames = int(summary.get("suppressed_cvat_mask_frames", 0))
        if authority == "cvat_snapshot" and (
            authoritative_frames != int(summary["num_frames"])
            or cvat_mask_frames != authoritative_frames
        ):
            raise ValueError(
                f"Existing output does not cover every authoritative frame: {output_dir}"
            )
        if authority == "cvat_snapshot_with_xml_invalidation" and (
            authoritative_frames != int(summary["num_frames"])
            or cvat_mask_frames + suppressed_frames != authoritative_frames
            or suppressed_frames != int(summary["xml_invalidation_frames"])
        ):
            raise ValueError(
                "Existing output does not suppress exactly the invalidated "
                f"CVAT frames: {output_dir}"
            )
        if authority in {"inherited", "inherited_with_xml_invalidation"} and (
            authoritative_frames != 0 or cvat_mask_frames != 0
        ):
            raise ValueError(
                f"Inherited output unexpectedly contains CVAT frames: {output_dir}"
            )
    if require_xml_invalidation:
        if summary.get("contour_teacher_schema") != (
            "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
        ):
            raise ValueError(f"Existing output is not v6 XML invalidation: {output_dir}")
        for name in (
            "xml_invalidation_manifest_sha256",
            "xml_invalidation_fingerprint",
        ):
            if not str(summary.get(name, "")):
                raise ValueError(
                    f"Existing output lacks {name}: {output_dir}"
                )
    if summary.get("annotated_h5_sha256") != file_sha256(input_h5):
        raise ValueError(f"Existing output input checksum mismatch: {output_dir}")
    pseudo3d_h5 = Path(str(summary.get("pseudo3d_h5", "")))
    if not pseudo3d_h5.is_file():
        raise FileNotFoundError(
            f"Existing output source pseudo3D H5 is missing: {pseudo3d_h5}"
        )
    if summary.get("pseudo3d_h5_sha256") != file_sha256(pseudo3d_h5):
        raise ValueError(
            f"Existing output source pseudo3D checksum mismatch: {output_dir}"
        )
    cvat_source = summary.get("cvat_mask_source", {})
    cvat_zip_text = str(cvat_source.get("annotation_zip", ""))
    if cvat_zip_text:
        cvat_zip = Path(cvat_zip_text)
        if not cvat_zip.is_file():
            raise FileNotFoundError(
                f"Existing output CVAT annotation ZIP is missing: {cvat_zip}"
            )
        if cvat_source.get("annotation_zip_sha256") != file_sha256(cvat_zip):
            raise ValueError(f"Existing output CVAT ZIP checksum mismatch: {output_dir}")
    expected_frames = int(summary["num_frames"])
    actual_frames = len(list((output_dir / "frames").glob("annotation_frame_*.png")))
    if actual_frames != expected_frames:
        raise ValueError(
            f"Existing frame count mismatch: {actual_frames} != {expected_frames}"
        )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip_existing and --overwrite are mutually exclusive")
    if (args.cvat_review_root is None) != (args.cvat_snapshot_root is None):
        raise ValueError(
            "--cvat_review_root and --cvat_snapshot_root must be specified together"
        )
    for name in ("cvat_review_root", "cvat_snapshot_root"):
        value = getattr(args, name)
        if value is not None and not Path(value).is_dir():
            raise FileNotFoundError(f"{name} not found: {value}")
    paths = collect_paths(
        annotated_dir=args.annotated_dir,
        h5_list=args.h5_list,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    selected_names = {
        value.strip() for value in args.video_names.split(",") if value.strip()
    }
    resolved = [(path, _video_name(path)) for path in paths]
    if selected_names:
        found_names = {name for _, name in resolved}
        missing = selected_names - found_names
        if missing:
            raise FileNotFoundError(f"Requested videos not found: {sorted(missing)}")
        resolved = [(path, name) for path, name in resolved if name in selected_names]
    names = [name for _, name in resolved]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate video_name values in input H5 files")
    if args.expected_files >= 0 and len(resolved) != args.expected_files:
        raise ValueError(
            f"Input H5 count mismatch: {len(resolved)} != {args.expected_files}"
        )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures = 0
    print("Stage 4 saved point-label visualization batch")
    print(f"  inputs      : {len(resolved)}")
    print(f"  output root : {output_root}")
    print("  contours    : not recomputed")
    for index, (path, video_name) in enumerate(resolved, start=1):
        output_dir = output_root / video_name
        try:
            if output_dir.exists() and any(output_dir.iterdir()) and args.skip_existing:
                summary = _existing_summary(
                    output_dir,
                    path,
                    require_cvat_masks=args.cvat_review_root is not None,
                    require_cvat_authoritative=args.require_cvat_authoritative,
                    require_xml_invalidation=args.require_xml_invalidation,
                )
                status = "skipped_verified"
            else:
                summary = export_stage4_point_label_visualization(
                    path,
                    output_dir,
                    point_radius=args.point_radius,
                    point_alpha=args.point_alpha,
                    positive_color=parse_color(args.positive_color),
                    ignore_color=parse_color(args.ignore_color),
                    background_color=parse_color(args.background_color),
                    bbox_color=parse_color(args.bbox_color),
                    bbox_thickness=args.bbox_thickness,
                    cvat_review_root=args.cvat_review_root,
                    cvat_snapshot_root=args.cvat_snapshot_root,
                    cvat_mask_color=parse_color(args.cvat_mask_color),
                    cvat_mask_alpha=args.cvat_mask_alpha,
                    require_cvat_authoritative=args.require_cvat_authoritative,
                    overwrite=args.overwrite,
                )
                status = "processed"
            if args.require_xml_invalidation:
                if summary.get("contour_teacher_schema") != (
                    "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
                ):
                    raise ValueError(
                        f"Visualization input is not v6 XML invalidation: {path}"
                    )
                for name in (
                    "xml_invalidation_manifest_sha256",
                    "xml_invalidation_fingerprint",
                ):
                    if not str(summary.get(name, "")):
                        raise ValueError(
                            f"Visualization summary lacks {name}: {path}"
                        )
            row = {
                "status": status,
                "video_name": video_name,
                "annotated_h5": str(path),
                "annotated_h5_sha256": summary["annotated_h5_sha256"],
                "output_dir": str(output_dir),
                "automatic_contours_recomputed": False,
                "input_files_modified": False,
                "num_frames": summary["num_frames"],
                "num_bbox_rows": summary["num_bbox_rows"],
                "num_points": summary["num_points"],
                "positive_points": summary["positive_points"],
                "ignore_points": summary["ignore_points"],
                "background_points": summary["background_points"],
                "cvat_mask_frames": summary["cvat_mask_frames"],
                "cvat_mask_pixels": summary["cvat_mask_pixels"],
                "suppressed_cvat_mask_frames": summary.get(
                    "suppressed_cvat_mask_frames", 0
                ),
                "suppressed_cvat_mask_pixels": summary.get(
                    "suppressed_cvat_mask_pixels", 0
                ),
                "cvat_label_authority": summary.get(
                    "cvat_label_authority", "legacy"
                ),
                "cvat_authoritative_frames": summary.get(
                    "cvat_authoritative_frames", 0
                ),
                "xml_invalidation_frames": summary.get(
                    "xml_invalidation_frames", 0
                ),
                "xml_invalidation_manifest_sha256": summary.get(
                    "xml_invalidation_manifest_sha256", ""
                ),
                "xml_invalidation_fingerprint": summary.get(
                    "xml_invalidation_fingerprint", ""
                ),
                "positive_outside_cvat_mask_points": summary.get(
                    "positive_outside_cvat_mask_points", 0
                ),
                "cvat_point_mask_mismatch_points": summary.get(
                    "cvat_point_mask_mismatch_points", 0
                ),
                "source_counts_json": json.dumps(
                    summary["source_counts"], sort_keys=True, separators=(",", ":")
                ),
                "error": "",
            }
            print(
                f"  [{index}/{len(resolved)}] [OK {status}] {video_name}: "
                f"frames={summary['num_frames']}, points={summary['num_points']}, "
                f"positive={summary['positive_points']}"
            )
        except Exception as exc:
            failures += 1
            row = {
                "status": "failed",
                "video_name": video_name,
                "annotated_h5": str(path),
                "annotated_h5_sha256": "",
                "output_dir": str(output_dir),
                "automatic_contours_recomputed": "",
                "input_files_modified": "",
                "num_frames": "",
                "num_bbox_rows": "",
                "num_points": "",
                "positive_points": "",
                "ignore_points": "",
                "background_points": "",
                "cvat_mask_frames": "",
                "cvat_mask_pixels": "",
                "suppressed_cvat_mask_frames": "",
                "suppressed_cvat_mask_pixels": "",
                "cvat_label_authority": "",
                "cvat_authoritative_frames": "",
                "xml_invalidation_frames": "",
                "xml_invalidation_manifest_sha256": "",
                "xml_invalidation_fingerprint": "",
                "positive_outside_cvat_mask_points": "",
                "cvat_point_mask_mismatch_points": "",
                "source_counts_json": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  [{index}/{len(resolved)}] [FAILED] {video_name}: {row['error']}")
            if args.verbose_errors:
                traceback.print_exc()
            rows.append(row)
            _write_csv(args.summary_csv, rows)
            if not args.continue_on_error:
                raise
            continue
        rows.append(row)
        _write_csv(args.summary_csv, rows)
    result = {
        "inputs": len(resolved),
        "processed": sum(row["status"] == "processed" for row in rows),
        "skipped_verified": sum(
            row["status"] == "skipped_verified" for row in rows
        ),
        "failed": failures,
        "input_files_modified": 0,
        "xml_invalidation_frames": sum(
            int(row["xml_invalidation_frames"])
            for row in rows
            if row["status"] != "failed"
        ),
        "xml_invalidation_videos": sum(
            int(row["xml_invalidation_frames"]) > 0
            for row in rows
            if row["status"] != "failed"
        ),
        "suppressed_cvat_mask_frames": sum(
            int(row["suppressed_cvat_mask_frames"])
            for row in rows
            if row["status"] != "failed"
        ),
    }
    manifest_hashes = {
        str(row["xml_invalidation_manifest_sha256"])
        for row in rows
        if row["status"] != "failed"
        and str(row["xml_invalidation_manifest_sha256"])
    }
    fingerprints = {
        str(row["xml_invalidation_fingerprint"])
        for row in rows
        if row["status"] != "failed"
        and str(row["xml_invalidation_fingerprint"])
    }
    if args.require_xml_invalidation and (
        len(manifest_hashes) != 1 or len(fingerprints) != 1
    ):
        raise RuntimeError(
            "v6 visualization inputs do not share one invalidation manifest"
        )
    if (
        args.expected_xml_invalidated_frames >= 0
        and result["xml_invalidation_frames"]
        != args.expected_xml_invalidated_frames
    ):
        raise RuntimeError(
            "XML invalidation frame count mismatch: "
            f"{result['xml_invalidation_frames']} != "
            f"{args.expected_xml_invalidated_frames}"
        )
    if (
        args.expected_xml_invalidated_videos >= 0
        and result["xml_invalidation_videos"]
        != args.expected_xml_invalidated_videos
    ):
        raise RuntimeError(
            "XML invalidation video count mismatch: "
            f"{result['xml_invalidation_videos']} != "
            f"{args.expected_xml_invalidated_videos}"
        )
    if (
        args.expected_suppressed_cvat_frames >= 0
        and result["suppressed_cvat_mask_frames"]
        != args.expected_suppressed_cvat_frames
    ):
        raise RuntimeError(
            "Suppressed CVAT frame count mismatch: "
            f"{result['suppressed_cvat_mask_frames']} != "
            f"{args.expected_suppressed_cvat_frames}"
        )
    print("Stage 4 saved point-label visualization summary")
    for name, value in result.items():
        print(f"  {name:16s}: {value}")
    if failures:
        raise RuntimeError(f"Saved point-label visualization failures: {failures}")
    print("Stage 4 saved point-label visualization batch passed.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-render saved Stage 4 point labels without contour recomputation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotated_dir", type=Path)
    source.add_argument("--h5_list", type=Path)
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--video_names", default="")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--expected_files", type=int, default=-1)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--point_alpha", type=float, default=0.85)
    parser.add_argument("--positive_color", default="255,32,32")
    parser.add_argument("--ignore_color", default="255,210,0")
    parser.add_argument("--background_color", default="80,140,200")
    parser.add_argument("--bbox_color", default="160,255,80")
    parser.add_argument("--bbox_thickness", type=int, default=1)
    parser.add_argument("--cvat_review_root", type=Path, default=None)
    parser.add_argument("--cvat_snapshot_root", type=Path, default=None)
    parser.add_argument("--cvat_mask_color", default="0,255,255")
    parser.add_argument("--cvat_mask_alpha", type=float, default=0.35)
    parser.add_argument("--require_cvat_authoritative", action="store_true")
    parser.add_argument("--require_xml_invalidation", action="store_true")
    parser.add_argument("--expected_xml_invalidated_frames", type=int, default=-1)
    parser.add_argument("--expected_xml_invalidated_videos", type=int, default=-1)
    parser.add_argument("--expected_suppressed_cvat_frames", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--verbose_errors", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        raise SystemExit(
            "Stage 4 saved point-label visualization batch failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
