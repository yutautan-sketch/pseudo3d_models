from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from export_anonymized_stage5_metrics import assert_share_bundle_anonymous
from stage5.utils.h5_io import load_stage5_pointcloud_h5, read_path_list
from stage5.utils.label_policy import (
    LABEL_BACKGROUND,
    LABEL_IGNORE,
    LABEL_POSITIVE,
    audit_bbox_noncontour_targets,
    compute_bbox_inside_mask,
)


# ----------------------------------------------------------------------------
# S5-12 Step E3: audit every teacher v6 H5 that will be used for the label
# policy ablation (train + validation lists) and prove, per file, that the
# bbox_noncontour_background contract holds:
#   1. valid_mask == (point_label != -1)
#   2. every ignore point lies inside a stored BBox (no stray ignore points)
#   3. no-BBox-frame points are background (or CVAT-authoritative positive,
#      never ignore)
# CPU/h5py only -- no CUDA or PointNeXt import anywhere in this checker.
# ----------------------------------------------------------------------------


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


SCHEMA_ATTR_KEYS = ("label_mode", "no_bbox_label", "bbox_inside_non_contour_label", "contour_teacher_schema")


def audit_file(path: Path) -> dict[str, Any]:
    data = load_stage5_pointcloud_h5(path)
    point_label = data["point_label"]
    valid_mask = data["valid_mask"]
    frame_order = data["frame_order"]

    require(
        np.array_equal(valid_mask, point_label != LABEL_IGNORE),
        f"{path}: valid_mask does not equal (point_label != -1)",
    )
    unexpected = np.setdiff1d(np.unique(point_label), np.asarray([LABEL_IGNORE, LABEL_BACKGROUND, LABEL_POSITIVE]))
    require(unexpected.size == 0, f"{path}: unexpected point labels {unexpected.tolist()}")

    has_bbox_data = "bbox_frame_order" in data and "bbox_local_xyxy" in data
    require(has_bbox_data, f"{path}: frame_annotation/frame_order or bbox_local_xyxy is missing")

    bbox_inside_mask = compute_bbox_inside_mask(
        frame_order=frame_order,
        pixel_xy=data["pixel_xy"],
        bbox_frame_order=data["bbox_frame_order"],
        bbox_local_xyxy=data["bbox_local_xyxy"],
    )
    target_stats = audit_bbox_noncontour_targets(point_label=point_label, bbox_inside_mask=bbox_inside_mask)

    bbox_frame_values = np.unique(data["bbox_frame_order"]) if data["bbox_frame_order"].size else np.array([], dtype=np.int64)
    no_bbox_mask = ~np.isin(frame_order, bbox_frame_values)
    no_bbox_background = int(np.sum(no_bbox_mask & (point_label == LABEL_BACKGROUND)))
    no_bbox_positive = int(np.sum(no_bbox_mask & (point_label == LABEL_POSITIVE)))
    no_bbox_ignore = int(np.sum(no_bbox_mask & (point_label == LABEL_IGNORE)))

    file_attrs = data.get("file_attrs", {})
    frame_annotation_attrs = data.get("frame_annotation_attrs", {})
    schema_attrs = {
        key: _decode(file_attrs.get(key, frame_annotation_attrs.get(key)))
        for key in SCHEMA_ATTR_KEYS
    }

    return {
        "point_count": int(point_label.size),
        "no_bbox_point_count": int(np.sum(no_bbox_mask)),
        "no_bbox_background_count": no_bbox_background,
        "no_bbox_positive_count": no_bbox_positive,
        "no_bbox_ignore_count": no_bbox_ignore,
        **target_stats,
        "schema_attrs": schema_attrs,
    }


def stray_point_details(path: Path) -> list[dict[str, Any]]:
    """Per-point detail for ignore points that fall outside every stored BBox in
    their frame -- frame_order and rounded pixel_xy only, plus the frame's own
    stored BBox rows (coords only) for direct comparison against a visualized
    frame image. No file path, video ID, or 3D coordinate is included."""
    data = load_stage5_pointcloud_h5(path)
    point_label = data["point_label"]
    frame_order = data["frame_order"]
    pixel_xy = data["pixel_xy"]
    bbox_frame_order = data["bbox_frame_order"]
    bbox_local_xyxy = data["bbox_local_xyxy"]

    bbox_inside_mask = compute_bbox_inside_mask(
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=bbox_frame_order,
        bbox_local_xyxy=bbox_local_xyxy,
    )
    stray_mask = (point_label == LABEL_IGNORE) & ~bbox_inside_mask
    stray_indices = np.flatnonzero(stray_mask)
    rounded_xy = np.rint(pixel_xy).astype(np.int64)

    rows: list[dict[str, Any]] = []
    for point_index in stray_indices:
        order = int(frame_order[point_index])
        frame_boxes = bbox_local_xyxy[bbox_frame_order == order]
        finite_boxes = [box for box in frame_boxes if np.isfinite(box).all()]
        rows.append(
            {
                "frame_order": order,
                "pixel_x": int(rounded_xy[point_index, 0]),
                "pixel_y": int(rounded_xy[point_index, 1]),
                "boxes_in_frame": len(frame_boxes),
                "finite_boxes_in_frame": len(finite_boxes),
                "frame_bbox_xyxy": "; ".join(
                    f"({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})" for b in finite_boxes
                ),
            }
        )
    return rows


def _decode(value: Any) -> str:
    if value is None:
        return ""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-12 Step E3: audit teacher v6 H5 files for the BBox non-contour "
        "label-policy contract before enabling --label_policy bbox_noncontour_background."
    )
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--share_output_dir", required=True)
    parser.add_argument("--private_output_dir", required=True)
    parser.add_argument(
        "--allow_stray_ignore",
        action="store_true",
        help="Report stray ignore points (outside any stored BBox) instead of failing fast. "
        "Never enable this to justify training with bbox_noncontour_background; use it only "
        "to inspect how many files/points are affected.",
    )
    parser.add_argument(
        "--dump_stray_alias",
        action="append",
        default=None,
        help="video_alias (e.g. train_068) to dump per-point stray-ignore detail for "
        "(frame_order, rounded pixel_xy, the frame's stored BBox coords). Repeatable. "
        "Implies --allow_stray_ignore for the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    share_output_dir = Path(args.share_output_dir)
    private_output_dir = Path(args.private_output_dir)
    share_output_dir.mkdir(parents=True, exist_ok=True)
    private_output_dir.mkdir(parents=True, exist_ok=True)

    dump_aliases = set(args.dump_stray_alias or [])
    allow_stray_ignore = args.allow_stray_ignore or bool(dump_aliases)

    splits = (("train", read_path_list(args.train_list)), ("val", read_path_list(args.val_list)))
    for split, paths in splits:
        require(bool(paths), f"--{split}_list must not be empty")

    share_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    violation_rows: list[dict[str, Any]] = []
    stray_detail_rows: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    schema_attr_values: dict[str, set[str]] = {key: set() for key in SCHEMA_ATTR_KEYS}
    found_dump_aliases: set[str] = set()

    for split, paths in splits:
        aliases = {index: f"{split}_{index:03d}" for index in range(len(paths))}
        for index, path in enumerate(paths):
            alias = aliases[index]
            require(path.is_file(), f"H5 not found: {path}")
            stats = audit_file(path)
            schema_attrs = stats.pop("schema_attrs")
            for key, value in schema_attrs.items():
                schema_attr_values[key].add(value)

            share_row = {"split": split, "video_alias": alias, **stats}
            share_rows.append(share_row)
            private_rows.append({"split": split, "video_alias": alias, "original_h5_path": str(path)})
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value

            if stats["stray_ignore_outside_bbox_count"] > 0:
                violation_rows.append(
                    {"split": split, "video_alias": alias, **stats}
                )

            if alias in dump_aliases:
                found_dump_aliases.add(alias)
                for row in stray_point_details(path):
                    stray_detail_rows.append({"split": split, "video_alias": alias, **row})

    write_csv(share_output_dir / "label_policy_bbox_preflight_per_file.csv", share_rows)
    write_csv(private_output_dir / "label_policy_bbox_preflight_video_id_map_DO_NOT_SHARE.csv", private_rows)
    if violation_rows:
        write_csv(
            private_output_dir / "label_policy_bbox_preflight_violations_DO_NOT_SHARE.csv", violation_rows
        )
    if dump_aliases:
        missing_dump_aliases = dump_aliases - found_dump_aliases
        require(not missing_dump_aliases, f"--dump_stray_alias not found in either list: {sorted(missing_dump_aliases)}")
        write_csv(share_output_dir / "label_policy_bbox_preflight_stray_point_detail.csv", stray_detail_rows)

    summary = {
        "status": "passed" if not violation_rows else "violations_found",
        "num_files": len(share_rows),
        "totals": totals,
        "violation_file_count": len(violation_rows),
        "schema_attrs_observed": {key: sorted(values) for key, values in schema_attr_values.items()},
    }
    with (share_output_dir / "label_policy_bbox_preflight_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    assert_share_bundle_anonymous(share_output_dir, private_rows=private_rows)

    if violation_rows and not allow_stray_ignore:
        raise SystemExit(
            f"[FAIL] {len(violation_rows)} file(s) contain ignore points outside any stored "
            "BBox. The bbox_noncontour_background label policy contract does not hold for "
            "this teacher v6 selection; see "
            f"{private_output_dir / 'label_policy_bbox_preflight_violations_DO_NOT_SHARE.csv'} "
            "before proceeding. Do not convert all-ignore to background as a workaround."
        )

    print("Stage5 label-policy BBox preflight passed." if not violation_rows else "Stage5 label-policy BBox preflight found violations (reported only).")
    print(f"files audited: {len(share_rows)}")
    print(f"total ignore points: {totals.get('ignore_point_count', 0)}")
    print(f"total bbox_noncontour target points: {totals.get('bbox_noncontour_target_point_count', 0)}")
    print(f"total stray ignore points (outside any BBox): {totals.get('stray_ignore_outside_bbox_count', 0)}")
    print(f"schema attrs observed: {summary['schema_attrs_observed']}")
    if dump_aliases:
        print(f"stray point detail dumped for: {sorted(found_dump_aliases)}")
        print(f"  -> {share_output_dir / 'label_policy_bbox_preflight_stray_point_detail.csv'}")
    print(f"share output: {share_output_dir}")
    print(f"private output: {private_output_dir}")


if __name__ == "__main__":
    main()
