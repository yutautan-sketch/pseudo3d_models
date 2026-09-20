from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from checks.real_h5.check_stage5_xy_coordinate_provenance import (
    read_intermediate_local_dimensions,
    resolve_intermediate_h5,
)
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list
from stage5.utils.prediction_frame_render import (
    CATEGORY_IGNORE_OTHER,
    CATEGORY_TRUE_NEGATIVE,
    DEFAULT_DRAW_CATEGORIES,
    DIAGNOSTIC_CATEGORY_NAMES,
    count_categories,
    diagnostic_segmentation_categories,
    draw_bbox,
    frame_point_index_ranges,
    load_image_to_uint8_gray,
    render_diagnostic_categories,
    validate_pixel_xy_bounds,
)

# ----------------------------------------------------------------------------
# S5-15: render predicted-positive / GT-positive segmentation onto the original
# local-crop frame images, so that false positives can be read as anatomy or
# instruments rather than as points in a PLY (handoff sections 1 and 10).
#
# Inputs are existing evaluation artifacts only: the saved per-video
# predictions (`predictions/<split>/<video>.npz`), the annotated teacher H5,
# and the intermediate pseudo3d H5 that holds `local_encoder_images`. Nothing
# is retrained, no inference is run, and nothing under the evaluation
# directory is modified -- output goes to a new subdirectory. CPU only.
#
# PRIVACY: the frames are patient data. The output directory is DO_NOT_SHARE
# and file names contain real video names (handoff section 7).
# ----------------------------------------------------------------------------

DEFAULT_SPLITS = ("train_sanity", "validation")
DEFAULT_STAGE2TO4_ROOT = "/mnt/data/3d_projects/models/Stage2to4"
DEFAULT_FALLBACK_SUFFIX = "_ts448_oym96_corr.h5"
DEFAULT_OUTPUT_SUBDIR = "prediction_frames"
IGNORE_INDEX = -1

DO_NOT_SHARE_NOTICE = """This directory contains rendered patient frame images.

DO NOT SHARE. Do not copy any file from here into an anonymized or shareable
directory. File names, CSV rows and manifest.json contain real video names and
absolute paths. If these renderings need to be shared, anonymize them first
using the existing video_alias convention (validation_000, ...), and run the
privacy self-check regexes used by the Stage 5 checkers.
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_name(value: str) -> str:
    """Mirror of `evaluate_stage5.py:230`.

    Re-implemented rather than imported because importing `evaluate_stage5`
    would pull in torch for a CPU-only, torch-free export. The synthetic check
    asserts the two stay identical.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path, *, hash_file: bool) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": None,
    }
    if hash_file:
        record["sha256"] = file_sha256(path)
    return record


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------


def read_h5_metrics_rows(evaluation_dir: Path) -> list[dict[str, str]]:
    """Read split/video_name/h5_path from the evaluation's own h5_metrics.csv.

    This is preferred over a separate --h5_list because it names the H5 files
    that this evaluation actually used (`evaluate_stage5.py:437-442`).
    """
    path = evaluation_dir / "h5_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"h5_metrics.csv not found under {evaluation_dir}. Pass --h5_list to "
            "supply the video-to-H5 mapping instead."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"{path} has no rows")
    for column in ("split", "video_name", "h5_path"):
        require(
            column in rows[0],
            f"{path} is missing the required column {column!r}; columns are {list(rows[0])}",
        )
    return [
        {
            "split": str(row["split"]),
            "video_name": str(row["video_name"]),
            "h5_path": str(row["h5_path"]),
        }
        for row in rows
    ]


def read_h5_list_rows(h5_list: Path, splits: Sequence[str]) -> list[dict[str, str]]:
    """Fallback mapping: take video_name from each H5's own attrs."""
    rows: list[dict[str, str]] = []
    for path in read_path_list(h5_list):
        require(path.is_file(), f"H5 listed in {h5_list} not found: {path}")
        with h5py.File(path, "r") as handle:
            video_name = handle.attrs.get("video_name", "")
        if isinstance(video_name, bytes):
            video_name = video_name.decode("utf-8")
        video_name = str(video_name).strip()
        require(bool(video_name), f"{path}: missing video_name attr")
        for split in splits:
            rows.append({"split": split, "video_name": video_name, "h5_path": str(path)})
    return rows


def load_prediction_npz(path: Path, *, num_points: int, video_name: str) -> np.ndarray:
    """Load `pred_label` and fail loudly if it does not match the H5 point set."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction npz not found for {video_name}: {path}. Run the evaluation "
            "without --no_save_predictions first."
        )
    with np.load(path) as payload:
        require(
            "pred_label" in payload,
            f"{path}: missing 'pred_label'; keys are {list(payload.keys())}",
        )
        pred_label = payload["pred_label"][:]
    if int(pred_label.size) != int(num_points):
        raise ValueError(
            f"{video_name}: prediction/H5 point count mismatch -- "
            f"{path} has {int(pred_label.size)} points but the H5 has {int(num_points)}. "
            "These are not the same point set; refusing to draw."
        )
    return pred_label


def resolve_video_intermediate(
    *,
    video_name: str,
    file_attrs: dict[str, Any],
    pseudo3d_outputs_root: Path,
    fallback_suffix: str,
) -> dict[str, Any]:
    """Locate the intermediate pseudo3d H5 and read its local-crop dimensions.

    Reuses the S5-14 provenance audit helpers unchanged, and keeps the
    resolution method in the result so that a recorded-attr hit and a basename
    fallback are never conflated (handoff 4.4).
    """
    recorded = file_attrs.get("source_pseudo3d_h5")
    if isinstance(recorded, bytes):
        recorded = recorded.decode("utf-8")
    resolution = resolve_intermediate_h5(
        video_name=video_name,
        recorded_source_path=str(recorded) if recorded else None,
        fallback_root=pseudo3d_outputs_root,
        fallback_suffix=fallback_suffix,
    )
    result: dict[str, Any] = {
        "resolution_method": resolution["method"],
        "intermediate_h5": resolution.get("path"),
        "dimension_status": "not_attempted",
        "width": None,
        "height": None,
    }
    if not resolution.get("path"):
        return result
    dimensions = read_intermediate_local_dimensions(Path(resolution["path"]))
    result["dimension_status"] = dimensions.get("status")
    if dimensions.get("status") == "ok":
        result["width"] = int(dimensions["width"])
        result["height"] = int(dimensions["height"])
    return result


# ----------------------------------------------------------------------------
# Frame aggregation and selection
# ----------------------------------------------------------------------------


def per_frame_category_counts(
    categories: np.ndarray,
    frame_index: Any,
) -> list[dict[str, Any]]:
    """Count each diagnostic category per frame, in frame_order order."""
    rows: list[dict[str, Any]] = []
    for frame_order in range(frame_index.num_frames):
        point_indices = frame_index.indices(frame_order)
        counts = count_categories(categories[point_indices])
        rows.append(
            {
                "frame_order": int(frame_order),
                "num_points": int(point_indices.size),
                **counts,
            }
        )
    return rows


def select_frames(
    frame_rows: Sequence[dict[str, Any]],
    *,
    top_frames_per_video: int,
    all_frames: bool,
    zero_fp_samples: int,
    zero_tp_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Pick which frames to render, deterministically.

    Primary order is false_positive count descending; ties break on
    frame_order ascending, so the same inputs always produce the same set and
    the same ranking (handoff 10.3 Step 4). Optional zero-FP / zero-TP samples
    are drawn with a fixed seed from frames that actually carry points.
    """
    if top_frames_per_video < 0:
        raise ValueError(f"--top_frames_per_video must be >= 0, got {top_frames_per_video}")
    if zero_fp_samples < 0 or zero_tp_samples < 0:
        raise ValueError("sample counts must be >= 0")

    selected: dict[int, str] = {}
    if all_frames:
        for row in frame_rows:
            selected[int(row["frame_order"])] = "all_frames"
    else:
        ranked = sorted(
            frame_rows,
            key=lambda row: (-int(row["false_positive"]), int(row["frame_order"])),
        )
        for rank, row in enumerate(ranked[:top_frames_per_video]):
            selected[int(row["frame_order"])] = f"top_false_positive_rank_{rank + 1:03d}"

        populated = [row for row in frame_rows if int(row["num_points"]) > 0]
        for count, reason, predicate in (
            (zero_fp_samples, "zero_false_positive_sample", lambda r: int(r["false_positive"]) == 0),
            (zero_tp_samples, "zero_true_positive_sample", lambda r: int(r["true_positive"]) == 0),
        ):
            if count <= 0:
                continue
            candidates = [
                int(row["frame_order"])
                for row in populated
                if predicate(row) and int(row["frame_order"]) not in selected
            ]
            candidates.sort()
            if not candidates:
                continue
            rng = random.Random(seed)
            chosen = sorted(rng.sample(candidates, min(count, len(candidates))))
            for frame_order in chosen:
                selected[frame_order] = reason

    by_frame = {int(row["frame_order"]): row for row in frame_rows}
    return [
        {**by_frame[frame_order], "selection_reason": reason}
        for frame_order, reason in sorted(selected.items())
    ]


def frame_png_name(frame_order: int, false_positive: int) -> str:
    """`frame_00012_fp0431.png` -- sorts by frame and shows the FP count."""
    return f"frame_{int(frame_order):05d}_fp{int(false_positive):04d}.png"


# ----------------------------------------------------------------------------
# Per-video export
# ----------------------------------------------------------------------------


def export_video_frames(
    *,
    split: str,
    video_name: str,
    h5_path: Path,
    prediction_path: Path,
    output_root: Path,
    pseudo3d_outputs_root: Path,
    fallback_suffix: str,
    options: argparse.Namespace,
    hash_h5: bool,
) -> dict[str, Any]:
    """Render one video. Returns a record for the manifest and summary CSV."""
    data = load_stage5_pointcloud_h5(h5_path)
    num_points = int(data["points"].shape[0])
    pred_label = load_prediction_npz(
        prediction_path, num_points=num_points, video_name=video_name
    )

    resolved = resolve_video_intermediate(
        video_name=video_name,
        file_attrs=data["file_attrs"],
        pseudo3d_outputs_root=pseudo3d_outputs_root,
        fallback_suffix=fallback_suffix,
    )
    record: dict[str, Any] = {
        "split": split,
        "video_name": video_name,
        "h5_path": str(h5_path),
        "prediction_npz": str(prediction_path),
        "num_points": num_points,
        **resolved,
        "status": "ok",
        "reason": "",
        "num_frames": None,
        "num_frames_rendered": 0,
    }

    # A video whose intermediate H5 or dimensions cannot be resolved is
    # recorded as un-visualizable. Borrowing another video's dimensions would
    # be a guess (handoff 4.6), so it is never done.
    if not resolved["intermediate_h5"]:
        record["status"] = "unvisualized"
        record["reason"] = f"intermediate_h5_{resolved['resolution_method']}"
        return record
    if resolved["dimension_status"] != "ok":
        record["status"] = "unvisualized"
        record["reason"] = str(resolved["dimension_status"])
        return record

    width = int(resolved["width"])
    height = int(resolved["height"])
    categories = diagnostic_segmentation_categories(
        data["point_label"],
        data["valid_mask"],
        pred_label,
        ignore_index=IGNORE_INDEX,
    )

    intermediate_path = Path(resolved["intermediate_h5"])
    with h5py.File(intermediate_path, "r") as handle:
        require(
            "local_encoder_images" in handle,
            f"{intermediate_path}: missing 'local_encoder_images'",
        )
        images = handle["local_encoder_images"]
        num_frames = int(images.shape[0])
        image_height, image_width = int(images.shape[-2]), int(images.shape[-1])
        require(
            (image_height, image_width) == (height, width),
            f"{video_name}: local_encoder_images is {image_height}x{image_width} but "
            f"local_input_shape says {height}x{width}; pixel_xy's space is ambiguous",
        )

        # Stop conditions (handoff 10.3 Step 3): no silent skipping, no clipping.
        validate_pixel_xy_bounds(
            data["pixel_xy"], width=width, height=height, context=video_name
        )
        frame_index = frame_point_index_ranges(
            np.asarray(data["frame_order"]).astype(np.int64), num_frames
        )
        record["num_frames"] = num_frames

        frame_rows = per_frame_category_counts(categories, frame_index)
        chosen_rows = select_frames(
            frame_rows,
            top_frames_per_video=options.top_frames_per_video,
            all_frames=bool(options.all_frames),
            zero_fp_samples=int(options.zero_fp_samples),
            zero_tp_samples=int(options.zero_tp_samples),
            seed=int(options.seed),
        )

        safe_video = safe_name(video_name)
        split_dir = output_root / split
        video_dir = split_dir / safe_video
        selection_by_frame = {
            int(row["frame_order"]): str(row["selection_reason"]) for row in chosen_rows
        }
        for row in frame_rows:
            frame_order = int(row["frame_order"])
            row["selected"] = int(frame_order in selection_by_frame)
            row["selection_reason"] = selection_by_frame.get(frame_order, "")
            row["png"] = (
                frame_png_name(frame_order, int(row["false_positive"]))
                if frame_order in selection_by_frame
                else ""
            )

        if not options.dry_run and chosen_rows:
            import imageio.v2 as imageio

            image_to_uint8_gray = load_image_to_uint8_gray()
            bbox_rows_by_frame: dict[int, list[int]] = {}
            if options.draw_bbox and "bbox_frame_order" in data:
                for row_index, frame_order in enumerate(data["bbox_frame_order"]):
                    bbox_rows_by_frame.setdefault(int(frame_order), []).append(row_index)

            video_dir.mkdir(parents=True, exist_ok=True)
            for row in chosen_rows:
                frame_order = int(row["frame_order"])
                point_indices = frame_index.indices(frame_order)
                gray = image_to_uint8_gray(images[frame_order])
                rgb = render_diagnostic_categories(
                    gray,
                    pixel_xy=data["pixel_xy"][point_indices],
                    categories=categories[point_indices],
                    radius=int(options.radius),
                    alpha=float(options.alpha),
                    draw_categories=resolve_draw_categories(options),
                )
                for bbox_row in bbox_rows_by_frame.get(frame_order, []):
                    draw_bbox(rgb, data["bbox_local_xyxy"][bbox_row])
                imageio.imwrite(
                    video_dir / frame_png_name(frame_order, int(row["false_positive"])),
                    rgb,
                )
                record["num_frames_rendered"] += 1

    write_csv(split_dir / f"{safe_video}_frame_summary.csv", frame_rows)

    totals = count_categories(categories)
    record.update({f"total_{name}": int(value) for name, value in totals.items()})
    record["num_frames_selected"] = len(chosen_rows)
    record["inputs"] = {
        "h5": file_fingerprint(h5_path, hash_file=hash_h5),
        "intermediate_h5": file_fingerprint(intermediate_path, hash_file=hash_h5),
        "prediction_npz": file_fingerprint(prediction_path, hash_file=True),
    }
    return record


def resolve_draw_categories(options: argparse.Namespace) -> tuple[int, ...]:
    selected = list(DEFAULT_DRAW_CATEGORIES)
    if options.draw_true_negative:
        selected.append(CATEGORY_TRUE_NEGATIVE)
    if options.draw_ignore_other:
        selected.append(CATEGORY_IGNORE_OTHER)
    return tuple(sorted(set(selected)))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render Stage 5 predicted-positive vs GT segmentation onto the original "
            "local-crop frames of an existing evaluation output (CPU only)."
        )
    )
    parser.add_argument(
        "--evaluation_dir",
        required=True,
        help="<evaluation output>/<checkpoint name>, e.g. .../<experiment>/best",
    )
    parser.add_argument("--stage2to4_root", default=DEFAULT_STAGE2TO4_ROOT)
    parser.add_argument(
        "--pseudo3d_outputs_root",
        required=True,
        help="Fallback directory of intermediate per-video pseudo3d H5 files",
    )
    parser.add_argument("--fallback_suffix", default=DEFAULT_FALLBACK_SUFFIX)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--h5_list", default=None, help="Overrides h5_metrics.csv")
    parser.add_argument("--output_subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--top_frames_per_video", type=int, default=10)
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--zero_fp_samples", type=int, default=0)
    parser.add_argument("--zero_tp_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--draw_true_negative", action="store_true")
    parser.add_argument("--draw_ignore_other", action="store_true")
    parser.add_argument("--draw_bbox", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve every input and write the CSVs/manifest, but render no PNGs",
    )
    parser.add_argument(
        "--hash_h5",
        action="store_true",
        help=(
            "Also sha256 the (multi-GB) H5 inputs. Off by default: the prediction "
            "npz is always hashed, and H5s are recorded by path/size/mtime."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    evaluation_dir = Path(args.evaluation_dir).resolve()
    require(evaluation_dir.is_dir(), f"--evaluation_dir not found: {evaluation_dir}")
    pseudo3d_outputs_root = Path(args.pseudo3d_outputs_root)
    require(
        pseudo3d_outputs_root.is_dir(),
        f"--pseudo3d_outputs_root not found: {pseudo3d_outputs_root}",
    )

    # The only place Stage2to4 is put on sys.path (handoff 10.0/10.3 Step 2).
    stage2to4_root = Path(args.stage2to4_root)
    require(
        stage2to4_root.is_dir(),
        f"--stage2to4_root not found: {stage2to4_root}. It must contain "
        "src/utils/alpha_texture_processing.py.",
    )
    if str(stage2to4_root) not in sys.path:
        sys.path.insert(0, str(stage2to4_root))
    if not args.dry_run:
        load_image_to_uint8_gray()

    if args.h5_list:
        rows = read_h5_list_rows(Path(args.h5_list), args.splits)
    else:
        rows = read_h5_metrics_rows(evaluation_dir)
    rows = [row for row in rows if row["split"] in set(args.splits)]
    require(
        bool(rows),
        f"No videos found for splits {args.splits} under {evaluation_dir}",
    )

    output_root = evaluation_dir / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "DO_NOT_SHARE.txt").write_text(DO_NOT_SHARE_NOTICE, encoding="utf-8")

    records: list[dict[str, Any]] = []
    for row in rows:
        video_name = row["video_name"]
        split = row["split"]
        h5_path = Path(row["h5_path"])
        require(h5_path.is_file(), f"{video_name}: H5 not found: {h5_path}")
        prediction_path = (
            evaluation_dir / "predictions" / split / f"{safe_name(video_name)}.npz"
        )
        print(f"[{split}] {video_name}")
        record = export_video_frames(
            split=split,
            video_name=video_name,
            h5_path=h5_path,
            prediction_path=prediction_path,
            output_root=output_root,
            pseudo3d_outputs_root=pseudo3d_outputs_root,
            fallback_suffix=args.fallback_suffix,
            options=args,
            hash_h5=bool(args.hash_h5),
        )
        if record["status"] == "unvisualized":
            print(f"  UNVISUALIZED: {record['reason']}")
        else:
            print(
                f"  frames={record['num_frames']} "
                f"selected={record.get('num_frames_selected', 0)} "
                f"rendered={record['num_frames_rendered']} "
                f"FP={record.get('total_false_positive')} "
                f"source={record['resolution_method']}"
            )
        records.append(record)

    unvisualized = [row for row in records if row["status"] == "unvisualized"]
    require(
        len(unvisualized) < len(records),
        "Every video was un-visualizable; this is a setup problem, not per-video "
        "data loss. Check --pseudo3d_outputs_root and --fallback_suffix. "
        f"Reasons: {sorted({row['reason'] for row in unvisualized})}",
    )

    summary_rows = [
        {key: value for key, value in record.items() if key != "inputs"}
        for record in records
    ]
    write_csv(output_root / "summary.csv", summary_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_dir": str(evaluation_dir),
        "output_dir": str(output_root),
        "stage2to4_root": str(stage2to4_root),
        "pseudo3d_outputs_root": str(pseudo3d_outputs_root),
        "fallback_suffix": args.fallback_suffix,
        "splits": list(args.splits),
        "dry_run": bool(args.dry_run),
        "render_options": {
            "radius": int(args.radius),
            "alpha": float(args.alpha),
            "draw_categories": [
                DIAGNOSTIC_CATEGORY_NAMES[index] for index in resolve_draw_categories(args)
            ],
            "draw_bbox": bool(args.draw_bbox),
        },
        "frame_selection": {
            "top_frames_per_video": int(args.top_frames_per_video),
            "all_frames": bool(args.all_frames),
            "zero_fp_samples": int(args.zero_fp_samples),
            "zero_tp_samples": int(args.zero_tp_samples),
            "seed": int(args.seed),
            "order": "false_positive desc, then frame_order asc",
        },
        "hash_h5": bool(args.hash_h5),
        "num_videos": len(records),
        "num_frames_rendered": sum(int(row["num_frames_rendered"]) for row in records),
        "unvisualized_videos": [
            {
                "split": row["split"],
                "video_name": row["video_name"],
                "reason": row["reason"],
                "resolution_method": row["resolution_method"],
            }
            for row in unvisualized
        ],
        "videos": records,
        "privacy": "DO_NOT_SHARE: rendered patient frames and real video names",
    }
    write_json(output_root / "manifest.json", manifest)

    print("Done.")
    print(f"  output dir       : {output_root}")
    print(f"  videos           : {len(records)}")
    print(f"  frames rendered  : {manifest['num_frames_rendered']}")
    print(f"  unvisualized     : {len(unvisualized)}")
    for row in unvisualized:
        print(f"    {row['split']}/{row['video_name']}: {row['reason']}")


if __name__ == "__main__":
    main()
