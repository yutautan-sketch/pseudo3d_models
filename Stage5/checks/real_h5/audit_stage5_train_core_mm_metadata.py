from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from stage5.utils.file_list_mode import read_file_list  # noqa: E402
from stage5.utils.split_contract import resolve_active_contract  # noqa: E402
from stage5.utils.split_identity import file_sha256  # noqa: E402

# The intermediate-H5 resolution and attribute reading are REUSED from the
# S5-14 provenance checker rather than rewritten, so the two cannot drift.
from checks.real_h5.check_stage5_xy_coordinate_provenance import (  # noqa: E402
    read_intermediate_local_dimensions,
    resolve_intermediate_h5,
)

# ----------------------------------------------------------------------------
# S5-16 Step 0 工程4: what the train_core videos actually record about mm.
#
# Report 9.7-2 and handoff section 6. This establishes availability, NOT a
# scale. Three things are kept apart and reported apart:
#
#   * The crop/resize inverse-transform metadata (raw_width/raw_height,
#     local_crop_top/left, local_resize_scale). Present today via the
#     intermediate pseudo-3D H5.
#
#   * The `spacing` dataset. Present, but written from --spacing_x/--spacing_y,
#     whose default is 1.0 and whose own help calls it a "temporary pixel
#     spacing for visualization geometry". It is also expressed against the
#     LOCAL CROP plane, not the original frame. So a spacing of 1.0 is recorded
#     as "placeholder observed", never as a measured millimetre scale.
#
#   * The real mm scale, whose source the user has yet to confirm. Until that
#     confirmation this is reported as UNCONFIRMED -- not as "unavailable",
#     because the absence of a value in these files is not evidence that no
#     source exists (report 10.1, 9.7-2).
#
# No surrogate FL is computed and no T_FL is inferred. Values from one video
# are never borrowed for another. train_core only: validation clinical FL and
# the sealed internal_test are not read. CPU only, read-only.
# ----------------------------------------------------------------------------

PLACEHOLDER_SPACING = 1.0


def _json_default(value: Any) -> Any:
    """Coerce the types h5py hands back into JSON-writable ones.

    Attribute values arrive as numpy scalars (`raw_width` is an `int64`, not an
    `int`) and strings may arrive as bytes. Passing them straight to
    `json.dumps` raises, which is what happened on the first real run: the
    whole audit completed and then failed at the write.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def read_final_attrs(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return dict(handle.attrs)


def read_spacing(path: Path) -> dict[str, Any]:
    """The intermediate H5's spacing/dimensions, with its provenance caveat."""
    try:
        with h5py.File(path, "r") as handle:
            if "spacing" not in handle:
                return {"status": "spacing_dataset_absent"}
            spacing = [float(value) for value in handle["spacing"][:].tolist()]
            dimensions = (
                [float(value) for value in handle["dimensions"][:].tolist()]
                if "dimensions" in handle
                else None
            )
    except OSError as error:
        return {"status": f"intermediate_h5_unreadable: {error}"}

    x, y = (spacing + [None, None])[:2]
    placeholder = x == PLACEHOLDER_SPACING and y == PLACEHOLDER_SPACING
    return {
        "status": "ok",
        "spacing_x": x,
        "spacing_y": y,
        "x_equals_y": x == y,
        "dimensions": dimensions,
        "looks_like_placeholder_default": placeholder,
        "reference_plane": "local crop plane, not the original frame",
        "interpretation": (
            "placeholder default observed; this is not a measured mm scale"
            if placeholder
            else "non-default value observed; its unit and reference frame still need confirming"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the mm-related metadata available for train_core. Reports availability "
        "and gaps; computes no scale, no surrogate FL and no T_FL."
    )
    parser.add_argument("--train_core_list", required=True)
    parser.add_argument("--split_manifest", default=None, help="Approved pin name (not a path)")
    parser.add_argument("--split_contract_pins", default=None)
    parser.add_argument("--artifact_coverage", default=None)
    parser.add_argument("--intermediate_root", required=True)
    parser.add_argument("--intermediate_suffix", default="_pseudo3d.h5")
    parser.add_argument("--private_json", required=True)
    parser.add_argument("--shareable_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = read_file_list(args.train_core_list, label="train_core")

    contract = resolve_active_contract(
        manifest_key=args.split_manifest,
        pins_path=args.split_contract_pins,
        coverage_path=args.artifact_coverage,
    )
    contract.assert_paths_allowed(paths, purpose="mm metadata audit inputs")

    per_video: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        alias = f"train_core_{index:03d}"
        if not path.is_file():
            per_video.append({"alias": alias, "status": "final_h5_missing"})
            continue
        attrs = read_final_attrs(path)
        resolution = resolve_intermediate_h5(
            video_name=path.name.split("_pointcloud_")[0],
            recorded_source_path=(
                attrs.get("source_pseudo3d_h5").decode("utf-8")
                if isinstance(attrs.get("source_pseudo3d_h5"), bytes)
                else attrs.get("source_pseudo3d_h5")
            ),
            fallback_root=Path(args.intermediate_root),
            fallback_suffix=args.intermediate_suffix,
        )
        entry: dict[str, Any] = {
            "alias": alias,
            "intermediate_resolution_method": resolution.get("method"),
        }
        if not resolution.get("path"):
            entry["status"] = "intermediate_h5_unresolved"
            per_video.append(entry)
            continue
        intermediate = Path(resolution["path"])
        entry["crop_inverse_transform"] = read_intermediate_local_dimensions(intermediate)
        entry["spacing"] = read_spacing(intermediate)
        entry["status"] = "ok"
        per_video.append(entry)

    def tally(predicate) -> int:
        return sum(1 for entry in per_video if predicate(entry))

    crop_ok = tally(lambda e: (e.get("crop_inverse_transform") or {}).get("status") == "ok")
    crop_fields_complete = tally(
        lambda e: all(
            (e.get("crop_inverse_transform") or {}).get(field) is not None
            for field in ("raw_width", "raw_height", "local_crop_top", "local_crop_left", "local_resize_scale")
        )
    )
    spacing_ok = tally(lambda e: (e.get("spacing") or {}).get("status") == "ok")
    spacing_placeholder = tally(
        lambda e: (e.get("spacing") or {}).get("looks_like_placeholder_default") is True
    )
    spacing_xy_differ = tally(lambda e: (e.get("spacing") or {}).get("x_equals_y") is False)

    findings = {
        "num_train_core": len(paths),
        "intermediate_resolved": tally(lambda e: e.get("status") == "ok"),
        "intermediate_unresolved": tally(lambda e: e.get("status") == "intermediate_h5_unresolved"),
        "final_h5_missing": tally(lambda e: e.get("status") == "final_h5_missing"),
        "crop_inverse_transform_readable": crop_ok,
        "crop_inverse_transform_fields_complete": crop_fields_complete,
        "spacing_dataset_readable": spacing_ok,
        "spacing_is_placeholder_default": spacing_placeholder,
        "spacing_x_differs_from_y": spacing_xy_differ,
    }

    conclusion = {
        "crop_inverse_transform": (
            "available" if crop_fields_complete == len(paths) else "incomplete for some videos"
        ),
        "mm_scale": "UNCONFIRMED",
        "mm_scale_reason": (
            "The only scale recorded in these files is the intermediate H5's `spacing`, written from "
            "--spacing_x/--spacing_y (default 1.0, documented as a temporary visualization value) and "
            "expressed against the local crop plane rather than the original frame. That is not a "
            "measured millimetre scale. It is also NOT evidence that no source exists: the user "
            "reported that the real vertical and horizontal mm scales are obtainable, so the source "
            "and its format remain to be confirmed."
        ),
        "per_frame_vs_per_video": (
            "cannot be determined from these files: `spacing` is stored once per video, so a "
            "per-frame scale would have to come from whatever external source is confirmed"
        ),
        "not_done_here": [
            "no surrogate FL was computed or selected",
            "no T_FL was inferred",
            "no value was borrowed from another video",
            "pseudo-3D Z was not assumed to have an established mm scale",
        ],
        "blocks": "S5-17 mm evaluation and clinical-FL comparison until the source is confirmed",
    }

    private = {
        "schema": "stage5_train_core_mm_metadata_v1",
        "note": "DO_NOT_SHARE: carries per-video metadata and real paths.",
        "train_core_list": str(args.train_core_list),
        "tool_sha256": file_sha256(Path(__file__)),
        "split_manifest": contract.manifest.name,
        "findings": findings,
        "conclusion": conclusion,
        "per_video": per_video,
    }
    Path(args.private_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.private_json).write_text(
        json.dumps(private, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    if args.shareable_json:
        Path(args.shareable_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.shareable_json).write_text(
            json.dumps(
                {
                    "schema": "stage5_train_core_mm_metadata_shareable_v1",
                    "findings": findings,
                    "conclusion": conclusion,
                    "withheld": "per-video metadata, real paths and any scale values",
                },
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

    print("Stage5 train_core mm metadata audit (availability only)")
    for key, value in findings.items():
        print(f"  {key:42s}: {value}")
    print(f"  mm scale: {conclusion['mm_scale']} -- source still to be confirmed with the user.")
    print("  No surrogate FL and no T_FL were computed.")


if __name__ == "__main__":
    main()
