from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.file_list_mode import read_file_list  # noqa: E402
from stage5.utils.split_allocation import (  # noqa: E402
    SELECTION_METHOD,
    STRATIFICATION_CONDITION,
    STRATIFICATION_QUANTITY,
    STRATUM_GROUP_NAMES,
    SplitSpecError,
    assert_stratification_is_constructible,
    assert_supported_versions,
    assign_stratification_groups,
    count_boundary_tie_members,
    draw_internal_test,
)
from stage5.utils.split_contract import (  # noqa: E402
    SplitContractError,
    assert_bootstrap_allowed,
    assert_resume_consistent,
    load_bootstrap_state,
    load_input_audit,
    verify_bootstrap_inputs,
    verify_intermediate_fingerprint,
    verify_split_partition,
)
from stage5.utils.split_identity import extract_video_identities, file_sha256  # noqa: E402

# The GT region counting is REUSED, not reimplemented and not re-audited: the
# S5-15 checker's functions are imported and called as an internal step of the
# split build. Its CLI is not used, so no alias-map CSV has to be invented for
# the 159 candidates, and the candidate GT statistics never become a standalone
# diagnostic output that someone could read before the split is drawn
# (report 9.3).
# `read_gt_points` is deliberately NOT imported: it converts with `astype`
# before validating, so a non-integer frame number is truncated, a valid_mask
# of 2 or NaN becomes True, and a wrongly shaped pixel_xy passes. This module
# reads and validates the raw arrays itself (read_and_validate_gt_points) and
# hands the result to the audited clustering functions unchanged.
from checks.real_h5.check_stage5_gt_component_count import (  # noqa: E402
    analyze_video,
    classify_video,
)

# ----------------------------------------------------------------------------
# S5-16 Step 0 工程2: build the B' split, then seal it.
#
# Two phases on purpose (report 9.10, 10.10.1):
#
#   --dry_run_stratification  reads the GT and the clinical FL table, forms the
#       strata, and reports ONLY the FL marginal counts. It draws no random
#       numbers and confirms no split. Its intermediate result is written into
#       the sealed area and fingerprinted.
#
#   --confirm  re-uses that fingerprinted intermediate rather than reading the
#       H5 files a second time, draws the 18, runs the six partition checks,
#       writes the lists, the seal registry and the split manifest, and prints
#       the hashes the user must add to split_contract_pins.json.
#
# The output boundary is the point of the design. Per-video GT statistics, the
# clinical FL values, the tertile boundaries and the per-cell allocation are
# sealed: publishing stratum sizes together with the allocation rule would
# reconstruct internal_test's GT and FL composition, which is the thing the
# seal exists to prevent. The FL marginal alone does not, because the
# multi-region marginal is never published -- and that is a limit on what is
# disclosed, not a proof that nothing can be inferred (report 10.10, 4).
#
# CPU only, read-only on every existing artefact. Nothing outside the new
# output directory is created or modified.
# ----------------------------------------------------------------------------

DEFAULT_LINK_DISTANCES = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_MIN_COMPONENT_POINTS = 5
STRATIFICATION_FIELD = "multi_region_any_frame_all_radii"
INTERMEDIATE_SCHEMA = "stage5_bprime_stratification_v2"

SEALED_DIR_MODE = 0o700
SEALED_FILE_MODE = 0o600


def _write_sealed(path: Path, payload: Any) -> str:
    """Write into the sealed area directly -- never via a temp file elsewhere."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=SEALED_DIR_MODE)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    os.chmod(path, SEALED_FILE_MODE)
    return file_sha256(path)


def _write_open(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def provenance() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    gt_checker = REPO_ROOT / "checks" / "real_h5" / "check_stage5_gt_component_count.py"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": run(["rev-parse", "HEAD"]),
        "git_dirty": bool(run(["status", "--short"])),
        "python": platform.python_version(),
        "builder_sha256": file_sha256(Path(__file__)),
        "gt_component_checker_sha256": file_sha256(gt_checker),
        "split_allocation_sha256": file_sha256(REPO_ROOT / "stage5" / "utils" / "split_allocation.py"),
        "split_contract_sha256": file_sha256(REPO_ROOT / "stage5" / "utils" / "split_contract.py"),
    }


def identity_of(path: Path) -> str:
    found = extract_video_identities(path.name)
    if len(found) != 1:
        raise SplitSpecError(
            f"a listed file name yields {len(found)} video identities; exactly one is required"
        )
    return found[0]


def candidate_set(train_paths: list[Path], sanity_paths: list[Path]) -> list[Path]:
    """The 159: the old train list minus the train-sanity videos, in list order."""
    sanity_ids = {identity_of(path) for path in sanity_paths}
    missing = sanity_ids - {identity_of(path) for path in train_paths}
    if missing:
        raise SplitSpecError(
            f"{len(missing)} train-sanity video(s) are not in the old train list; the sanity set "
            "could not be confirmed against it"
        )
    return [path for path in train_paths if identity_of(path) not in sanity_ids]


def scan_candidate_gt(
    paths: list[Path], *, link_distances, min_component_points
) -> tuple[dict[str, bool], dict[str, int], dict[str, Any]]:
    """One GT pass per candidate: multi-region label AND the stratification value.

    Both axes come out of the same read, so the second axis costs no extra
    I/O. A video that fails validation stops the build; it is never given a
    zero or dropped into a missing group.
    """
    classification: dict[str, bool] = {}
    quantity: dict[str, int] = {}
    detail: dict[str, Any] = {}
    for index, path in enumerate(paths):
        identity = identity_of(path)
        try:
            data = read_and_validate_gt_points(path)
        except SplitSpecError as error:
            # Position and reason only: a real path or video ID is itself a
            # private identifier.
            raise SplitSpecError(
                f"candidate #{index} failed GT validation and the quantity cannot be computed "
                f"({error})"
            ) from error
        sweep = [
            analyze_video(data, link_distance=distance, min_component_points=min_component_points)
            for distance in link_distances
        ]
        result = classify_video(sweep, frame_fraction_threshold=0.5)
        classification[identity] = bool(result[STRATIFICATION_FIELD])
        quantity[identity] = gt_positive_frame_count(data)
        detail[identity] = {
            "classification": result,
            "gt_positive_frame_count": quantity[identity],
            "sweep": [{k: v for k, v in entry.items() if k != "per_frame"} for entry in sweep],
        }
    return classification, quantity, detail


INT64_MIN, INT64_MAX = -(2 ** 63), 2 ** 63 - 1


def read_and_validate_gt_points(h5_path: Path) -> dict[str, Any]:
    """Read the four GT arrays, validating the RAW values before converting.

    The S5-15 checker's own `read_gt_points` converts first
    (`.astype(np.int64)` / `.astype(bool)`), which silently truncates a
    non-integer frame number, turns a valid_mask of 2 -- or of NaN -- into
    True, and only ever checks `shape[0]`. None of those would be caught
    afterwards, so validation has to happen on the raw arrays (report 3.2.1).

    Returns the same dict shape the audited `analyze_video`/`classify_video`
    expect, so the clustering and classification logic is reused unchanged.
    One read per video; no additional I/O.
    """
    import h5py
    import numpy as np

    required = (
        "point_cloud/pixel_xy",
        "point_cloud/frame_order",
        "annotation/point_label",
        "annotation/valid_mask",
    )
    with h5py.File(h5_path, "r") as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise SplitSpecError(f"missing dataset(s) {missing}")
        raw_xy = f["point_cloud/pixel_xy"][:]
        raw_frame = f["point_cloud/frame_order"][:]
        raw_label = f["annotation/point_label"][:]
        raw_valid = f["annotation/valid_mask"][:]

    if raw_xy.ndim != 2 or raw_xy.shape[1] != 2:
        raise SplitSpecError(f"pixel_xy has shape {raw_xy.shape}, expected (N, 2)")
    for name, array in (("frame_order", raw_frame), ("point_label", raw_label),
                        ("valid_mask", raw_valid)):
        if array.ndim != 1:
            raise SplitSpecError(f"{name} has {array.ndim} dimensions, expected 1")
        if array.shape[0] != raw_xy.shape[0]:
            raise SplitSpecError(
                f"{name} has {array.shape[0]} rows, pixel_xy has {raw_xy.shape[0]}"
            )
    if raw_xy.shape[0] == 0:
        # Distinct from "this video has zero GT-positive frames", which is a
        # valid value. An empty point cloud cannot be measured at all.
        raise SplitSpecError("the point cloud is empty; the quantity cannot be computed")

    if not np.all(np.isfinite(raw_xy)):
        raise SplitSpecError("pixel_xy contains NaN or Inf")

    # frame_order: finite, integral, and inside the target integer range.
    # An integral-looking float can still be Inf or out of int64 range.
    if np.issubdtype(raw_frame.dtype, np.floating):
        if not np.all(np.isfinite(raw_frame)):
            raise SplitSpecError("frame_order contains NaN or Inf")
        if not np.all(raw_frame == np.floor(raw_frame)):
            raise SplitSpecError("frame_order holds non-integral values")
        if raw_frame.min() < INT64_MIN or raw_frame.max() > INT64_MAX:
            raise SplitSpecError("frame_order holds values outside the int64 range")
    elif not np.issubdtype(raw_frame.dtype, np.integer):
        raise SplitSpecError(f"frame_order has dtype {raw_frame.dtype}, expected integer")
    frame_order = raw_frame.astype(np.int64)
    if frame_order.min() < 0:
        raise SplitSpecError("frame_order holds negative values")

    if not np.issubdtype(raw_label.dtype, np.integer):
        raise SplitSpecError(f"point_label has dtype {raw_label.dtype}, expected integer")
    point_label = raw_label.astype(np.int64)
    bad = set(int(v) for v in np.unique(point_label)) - {-1, 0, 1}
    if bad:
        raise SplitSpecError(f"point_label holds value(s) outside {{-1, 0, 1}}: {sorted(bad)}")

    if raw_valid.dtype == bool:
        valid_mask = raw_valid
    elif np.issubdtype(raw_valid.dtype, np.integer):
        allowed = set(int(v) for v in np.unique(raw_valid)) <= {0, 1}
        if not allowed:
            raise SplitSpecError("valid_mask holds integer values other than 0/1")
        valid_mask = raw_valid.astype(bool)
    else:
        raise SplitSpecError(
            f"valid_mask has dtype {raw_valid.dtype}; only bool or 0/1 integers are accepted "
            "(a float mask would turn NaN into True)"
        )

    if not np.array_equal(valid_mask, point_label != -1):
        raise SplitSpecError("valid_mask differs from point_label != -1")

    return {
        "pixel_xy": raw_xy.astype(np.float64),
        "frame_order": frame_order,
        "gt_positive": valid_mask & (point_label == 1),
    }


def gt_positive_frame_count(data: dict[str, Any]) -> int:
    """The stratification quantity: distinct frames holding a GT-positive point.

    Identical to the audited checker's `num_gt_frames`, which is built from
    `np.unique(frame_order[gt])` and so does not depend on `link_distance`,
    `min_component_points`, or any pixel coordinate.
    """
    import numpy as np

    return int(np.unique(data["frame_order"][data["gt_positive"]]).size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and seal the S5-16 B' split (train_core 144 / validation 18 / "
        "internal_test 18). Two phases: --dry_run_stratification then --confirm."
    )
    parser.add_argument("--old_train_list", required=True, help="Saved train162 list (unchanged)")
    parser.add_argument("--old_val_list", required=True, help="Saved validation18 list (unchanged)")
    parser.add_argument("--sanity_list", required=True, help="Saved selected_train_files.txt (3 videos)")
    parser.add_argument("--output_dir", required=True, help="New private output directory")
    parser.add_argument(
        "--split_contract_pins",
        default=None,
        help="Pins file the initial-build check consults. Defaults to the repo file; a "
        "synthetic test points this at its own so it does not depend on production state.",
    )
    parser.add_argument(
        "--input_audit_json",
        required=True,
        help="S0-1 audit record fixing each input's expected SHA-256. Required: the inputs cannot "
        "vouch for themselves.",
    )
    parser.add_argument(
        "--expected_intermediate_sha256",
        default=None,
        help="With --confirm: the fingerprint --dry_run_stratification printed for the sealed "
        "stratification intermediate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--internal_test_size", type=int, default=18)
    parser.add_argument("--link_distances", default=",".join(str(d) for d in DEFAULT_LINK_DISTANCES))
    parser.add_argument("--min_component_points", type=int, default=DEFAULT_MIN_COMPONENT_POINTS)
    parser.add_argument("--bootstrap", action="store_true", required=True,
                        help="Explicit acknowledgement that this is the one-time initial build")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run_stratification", action="store_true")
    mode.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    sealed_dir = out_dir / "sealed"
    registry_path = out_dir / "sealed_registry.json"
    intermediate_path = sealed_dir / "stratification_DO_NOT_SHARE.json"

    # 10.10.1: the initial build is a narrow, explicitly-requested entry, not a
    # switch that turns the guard off. It closes as soon as a registry is
    # approved or already written, so it can never overwrite a confirmed split.
    assert_bootstrap_allowed(registry_out=registry_path, pins_path=args.split_contract_pins)

    # The expected hashes come from the S0-1 audit record, not from the inputs
    # themselves: a hash computed here and compared with itself would always
    # match and would verify nothing (report 11.7.3).
    inputs = {
        "old_train_list": args.old_train_list,
        "old_val_list": args.old_val_list,
        "sanity_list": args.sanity_list,
    }
    observed_inputs = verify_bootstrap_inputs(
        inputs, expected_sha256=load_input_audit(args.input_audit_json)
    )
    assert_resume_consistent(load_bootstrap_state(sealed_dir / "bootstrap_state.json"), observed_inputs)

    train_paths = read_file_list(args.old_train_list, label="old train")
    val_paths = read_file_list(args.old_val_list, label="old validation")
    sanity_paths = read_file_list(args.sanity_list, label="train sanity")
    if len(sanity_paths) != 3:
        raise SplitSpecError(f"expected 3 train-sanity videos, got {len(sanity_paths)}")

    candidates = candidate_set(train_paths, sanity_paths)
    candidate_ids = [identity_of(path) for path in candidates]
    order_index = {identity: index for index, identity in enumerate(candidate_ids)}
    # Aliases exist only for the private records and the refusal messages; the
    # existing validation_NNN / train_sanity_* namespaces are left alone.
    aliases = {identity: f"train_pool_{index:03d}" for index, identity in enumerate(candidate_ids)}

    link_distances = [float(value) for value in args.link_distances.split(",") if value.strip()]

    if args.dry_run_stratification:
        multi_region, quantity, gt_detail = scan_candidate_gt(
            candidates,
            link_distances=link_distances,
            min_component_points=args.min_component_points,
        )
        stratum_group = assign_stratification_groups(quantity, order_index=order_index)
        condition = assert_stratification_is_constructible(quantity, stratum_group)
        boundary_ties = count_boundary_tie_members(quantity, stratum_group)
        zero_valued = sum(1 for value in quantity.values() if value == 0)

        # Every setting the confirm step must match, fixed here. Schema alone
        # would not stop a confirm run using, say, a different
        # min_component_points (report 4.1.1).
        spec = {
            "stratification_field": STRATIFICATION_FIELD,
            "link_distances": link_distances,
            "min_component_points": args.min_component_points,
            "quantity": STRATIFICATION_QUANTITY,
            "quantity_version": STRATIFICATION_QUANTITY["version"],
            "condition": STRATIFICATION_CONDITION,
            "condition_version": STRATIFICATION_CONDITION["version"],
            "tertile_remainder_rule": "extra members go to the lower groups first",
            "tie_break": "equal values ordered by position in the saved train list",
            "seed": args.seed,
            "internal_test_size": args.internal_test_size,
            "selection_method": SELECTION_METHOD,
        }

        _write_sealed(
            sealed_dir / "bootstrap_state.json",
            {"schema": "stage5_split_bootstrap_state_v1", "input_sha256": observed_inputs},
        )
        digest = _write_sealed(
            intermediate_path,
            {
                "schema": INTERMEDIATE_SCHEMA,
                "note": "DO_NOT_SHARE: per-video GT statistics, stratification values and groups.",
                "candidate_identities": candidate_ids,
                "aliases": aliases,
                "multi_region": multi_region,
                "gt_detail": gt_detail,
                "stratification_values": quantity,
                "stratum_group": stratum_group,
                "condition_result": condition,
                "boundary_tie_members": boundary_ties,
                "zero_valued_candidates": zero_valued,
                "spec": spec,
                "input_sha256": observed_inputs,
                "provenance": provenance(),
            },
        )

        # Shared GT aggregates are the candidate count and the constructibility
        # boolean, and nothing else -- group sizes are a GT-derived frequency.
        # The fingerprint and the spec versions travel with it because the
        # confirm step needs them; they carry no GT value (report 4.4).
        _write_open(
            out_dir / "stratification_feasibility_SHARE.json",
            {
                "schema": "stage5_bprime_stratification_feasibility_v2",
                "num_candidates": len(candidate_ids),
                "strata_constructible": True,
                "intermediate_sha256": digest,
                "schema_versions": {
                    "intermediate": INTERMEDIATE_SCHEMA,
                    "quantity": STRATIFICATION_QUANTITY["version"],
                    "condition": STRATIFICATION_CONDITION["version"],
                    "selection_method": SELECTION_METHOD["name"],
                },
                "withheld": (
                    "group sizes, medians, the multi-region marginal, the cross table, "
                    "per-cell allocation, per-video values and extremes"
                ),
            },
        )
        print("Stage5 B' stratification dry run complete; NO split was drawn and none is confirmed.")
        print(f"  candidates            : {len(candidate_ids)}")
        print("  strata constructible  : True")
        print(f"  intermediate sha256   : {digest}")
        print("  Pass that fingerprint to --confirm as --expected_intermediate_sha256.")
        print("  Group sizes, medians, multi-region counts and per-video values are sealed")
        print("  and are not reported here.")
        return

    # ---------------- confirm ----------------
    if not intermediate_path.is_file():
        raise SplitContractError(
            "no sealed stratification intermediate found: run --dry_run_stratification first. "
            "The H5 files are not read a second time."
        )
    # The file's own fingerprint, against the value fixed at the dry run. The
    # content checks below would not notice an intermediate whose GT
    # classification or groups changed while its inputs and seed did not.
    intermediate_sha = verify_intermediate_fingerprint(
        intermediate_path, args.expected_intermediate_sha256
    )
    intermediate = json.loads(intermediate_path.read_text(encoding="utf-8"))
    if intermediate.get("schema") != INTERMEDIATE_SCHEMA:
        raise SplitContractError("the sealed intermediate has an unexpected schema")
    if intermediate.get("input_sha256") != observed_inputs:
        raise SplitContractError(
            "the inputs changed since the dry run; the stratification is not reused across "
            "different inputs"
        )
    if intermediate["candidate_identities"] != candidate_ids:
        raise SplitContractError("the candidate set changed since the dry run")

    spec = dict(intermediate["spec"])

    # The intermediate's settings are the single basis for the confirm run.
    # An unknown version is refused even when the corresponding CLI argument
    # was omitted: leaving an argument out is not a reason to accept a spec
    # this implementation does not know (report 4.1.1, correction 14).
    assert_supported_versions(
        str(spec.get("quantity_version", "")), str(spec.get("condition_version", ""))
    )
    if (spec.get("selection_method") or {}).get("name") != SELECTION_METHOD["name"]:
        raise SplitContractError(
            "the intermediate was drawn with a selection method this implementation does not use"
        )

    # Where a CLI argument was supplied and disagrees with the intermediate,
    # neither silently wins.
    for label, recorded, supplied in (
        ("seed", spec.get("seed"), args.seed),
        ("internal_test_size", spec.get("internal_test_size"), args.internal_test_size),
        ("min_component_points", spec.get("min_component_points"), args.min_component_points),
        ("link_distances", spec.get("link_distances"), link_distances),
    ):
        if recorded is not None and recorded != supplied:
            raise SplitContractError(
                f"{label} differs between the dry run and this confirm run "
                "(recorded in the sealed intermediate vs supplied now). Neither is preferred "
                "silently; re-run with the recorded settings or start a new dry run."
            )

    multi_region = {k: bool(v) for k, v in intermediate["multi_region"].items()}
    stratum_group = {k: int(v) for k, v in intermediate["stratum_group"].items()}

    picked_ids, alloc = draw_internal_test(
        candidate_ids,
        multi_region=multi_region,
        stratum_group=stratum_group,
        total=args.internal_test_size,
        seed=args.seed,
    )
    picked = set(picked_ids)
    internal_test = [path for path in candidates if identity_of(path) in picked]
    train_core = [path for path in train_paths if identity_of(path) not in picked]

    summary = verify_split_partition(
        train_core=train_core,
        validation=val_paths,
        internal_test=internal_test,
        old_train=train_paths,
        old_validation=val_paths,
        sanity=sanity_paths,
        expected_counts=(len(train_paths) - args.internal_test_size, len(val_paths), args.internal_test_size),
    )

    _write_open(out_dir / "train_core_144.txt", "\n".join(str(p) for p in train_core) + "\n")
    _write_open(out_dir / "validation_18.txt", "\n".join(str(p) for p in val_paths) + "\n")
    _write_sealed(sealed_dir / "internal_test_18.txt", "\n".join(str(p) for p in internal_test) + "\n")
    _write_sealed(
        sealed_dir / "allocation_DO_NOT_SHARE.json",
        {
            "note": "Per-cell allocation and stratum sizes. Sealed: together they reconstruct "
                    "internal_test's GT and FL composition.",
            "allocation": {
                f"{key[0]}|{STRATUM_GROUP_NAMES[key[1]]}": value for key, value in alloc.items()
            },
        },
    )

    registry_sha = _write_open(
        registry_path,
        {
            "schema": "stage5_seal_registry_v1",
            "sealed_video_identities": sorted(picked),
            # Binds the sealed set to the list it came from, so a manifest that
            # claims to hold this split is checked against the registry's own
            # record rather than on the strength of the split's name.
            "sealed_list_identity_sha256": summary["identity_sha256"]["internal_test"],
            "sealed_until": "S5-20c",
            "prohibited_checkpoint_rule": (
                "A checkpoint whose training set identity does not match train_core may never be "
                "evaluated on internal_test. This survives the S5-20c release: the old R0 50-epoch "
                "and W-A checkpoints trained on these videos and are permanently ineligible."
            ),
            "provenance": provenance(),
        },
    )
    manifest_sha = _write_open(
        out_dir / "s5_16_bprime.json",
        {
            "schema": "stage5_split_contract_v1",
            "name": "s5_16_bprime",
            "lists": {
                "train_core": {
                    "path": str(out_dir / "train_core_144.txt"),
                    "count": summary["counts"]["train_core"],
                    "content_sha256": summary["content_sha256"]["train_core"],
                    "identity_sha256": summary["identity_sha256"]["train_core"],
                },
                "validation": {
                    "path": str(out_dir / "validation_18.txt"),
                    "count": summary["counts"]["validation"],
                    "content_sha256": summary["content_sha256"]["validation"],
                    "identity_sha256": summary["identity_sha256"]["validation"],
                },
                "internal_test": {
                    "path": str(sealed_dir / "internal_test_18.txt"),
                    "count": summary["counts"]["internal_test"],
                    "content_sha256": summary["content_sha256"]["internal_test"],
                    "identity_sha256": summary["identity_sha256"]["internal_test"],
                },
            },
            "sealed_splits": ["internal_test"],
            "seal_registry_required": True,
            "sealed_until": "S5-20c",
            "allows_new_training": True,
            "allows_directory_mode": False,
            "teacher": "bboxrank_v7_cvat_authoritative_crop_quality_v1",
            "stratification": spec,
            "selection_method": SELECTION_METHOD,
            "intermediate_sha256": intermediate_sha,
            "input_sha256": observed_inputs,
            "provenance": provenance(),
        },
    )

    print("Stage5 B' split confirmed and sealed")
    print(f"  train_core     : {summary['counts']['train_core']}  "
          f"identity {summary['identity_sha256']['train_core'][:16]}")
    print(f"  validation     : {summary['counts']['validation']}  "
          f"identity {summary['identity_sha256']['validation'][:16]}")
    print(f"  internal_test  : {summary['counts']['internal_test']}  "
          f"identity {summary['identity_sha256']['internal_test'][:16]}")
    print(f"  validation path spelling unchanged: {summary['validation_representation_identical']}")
    print("  All six partition checks passed. Per-cell allocation and GT/FL statistics are sealed.")
    print("")
    print("  Next, to bring normal operation into force:")
    print(f"    seal_registry  {registry_path}  sha256 {registry_sha}")
    print(f"    s5_16_bprime   {out_dir / 's5_16_bprime.json'}  sha256 {manifest_sha}")
    print("  Add both to stage5/config/split_contract_pins.json, review the git diff and commit.")
    print("  Until those pins exist, training, evaluation and inference all refuse to start.")


if __name__ == "__main__":
    try:
        main()
    except (SplitSpecError, SplitContractError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(2)
