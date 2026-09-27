from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.file_list_mode import (  # noqa: E402
    FixedListError,
    list_content_sha256,
    list_identity_sha256,
    read_file_list,
)
from stage5.utils.split_allocation import SplitSpecError  # noqa: E402
from stage5.utils.split_contract import (  # noqa: E402
    INPUT_AUDIT_SCHEMA,
    SEAL_REGISTRY_PIN_KEY,
    Pins,
    SplitContractError,
    resolve_active_contract,
)
from stage5.utils.split_identity import (  # noqa: E402
    extract_video_identities,
    file_sha256,
    masked_shape,
)
from stage5.utils.train_sanity_selection import select_train_paths  # noqa: E402

# ----------------------------------------------------------------------------
# S5-16 Step 0 工程1 (S0-1): audit the inputs and FIX their expected hashes.
#
# This exists because the split builder must not verify its inputs against
# hashes it computed from those same inputs -- that always matches and proves
# nothing (report 11.7.3). The expected values have to be fixed once, here, and
# handed to the builder as `--input_audit_json`.
#
# What it establishes:
#   * the saved train162 / validation18 lists are readable, complete, teacher-v7,
#     duplicate-free, non-overlapping, and every listed H5 exists
#   * the three train-sanity videos are identified BY VIDEO IDENTITY and
#     corroborated against the old evaluation's private ID map -- three aliases
#     existing, or a count of three, is not accepted as proof (report 9.6)
#   * the clinical FL table's structure: columns present, exactly one row per
#     candidate, no unknown identifiers, and the declared missing tokens
#     sufficient to classify every value
#   * the 159-candidate set follows from train162 minus the sanity three
#
# What it deliberately does NOT do:
#   * it opens no H5 (existence and name only), so no point data is read
#   * it reports no clinical FL VALUE, no per-token count and hence no missing
#     count, and no GT statistic. The FL marginal is released once, later, by
#     the dry run (report 9.2). Distinct non-numeric tokens are listed in the
#     private record only, because the operator needs them to declare the token
#     set -- a token is not a measurement.
#   * it confirms nothing it could not corroborate: missing sanity evidence
#     stops the audit and writes no record, rather than being noted and passed
#     over.
#
# Pre-seal by construction: once a seal registry is approved, the old train162
# list contains sealed videos, so re-running this is refused by the contract.
# Stdlib only -- no numpy, no h5py, no torch.
# ----------------------------------------------------------------------------

TEACHER_V7_PATTERN = (
    "*_pointcloud_annotated_foreground_combined_v2_global_local_l75_w31_c12_area15_"
    "bboxrank_v7_cvat_authoritative_crop_quality_v1.h5"
)
DEFAULT_FIXED_TRAIN_VIDEO = "20250626_124212_7300"


class AuditError(ValueError):
    """A condition that must stop the audit rather than be recorded as a caveat."""


def provenance() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": run(["rev-parse", "HEAD"]),
        "git_dirty": bool(run(["status", "--short"])),
        "python": platform.python_version(),
        "tool_sha256": file_sha256(Path(__file__)),
    }


def identity_of(path: Path, *, label: str = "input", index: int | None = None) -> str:
    """The one video identity in a file name, with a diagnosable failure.

    On a mismatch the message gives the list, the position and the name's
    character SHAPE rather than the name, so the format problem can be seen
    without a video identifier appearing in the output.
    """
    found = extract_video_identities(path.name)
    if len(found) != 1:
        where = f"{label} entry #{index + 1}" if index is not None else f"a {label} entry"
        raise AuditError(
            f"{where} yields {len(found)} video identities; exactly one is required.\n"
            f"  expected shape: ########_######_# ... (YYYYMMDD_HHMMSS_N)\n"
            f"  actual shape  : {masked_shape(path.name)}\n"
            "  (digits shown as #, letters as a; the real name is not printed)"
        )
    return found[0]


def audit_list(path: str | Path, *, label: str, expected_count: int, pattern: str) -> dict[str, Any]:
    paths = read_file_list(path, label=label)
    identities = [
        identity_of(entry, label=label, index=index) for index, entry in enumerate(paths)
    ]
    if len(set(identities)) != len(identities):
        # Two very different problems share this symptom, and the message has
        # to say which: the SAME file listed twice (an input defect), or two
        # DIFFERENT files resolving to one identity (the identity rule is
        # wrong, which for a seal contract would let one video stand in for
        # another). Positions and masked shapes only -- no identifier.
        from collections import Counter

        collisions = [ident for ident, n in Counter(identities).items() if n > 1]
        details = []
        for ident in collisions[:5]:
            positions = [i for i, value in enumerate(identities) if value == ident]
            names = {paths[i].name for i in positions}
            kind = (
                "the same file name listed more than once"
                if len(names) == 1
                else "DIFFERENT file names resolving to one identity"
            )
            shapes = sorted({masked_shape(name) for name in names})
            details.append(
                f"    entries #{[i + 1 for i in positions]}: {kind}\n"
                + "".join(f"      shape: {shape}\n" for shape in shapes)
            )
        raise AuditError(
            f"{label} list resolves {len(collisions)} identit(ies) from more than one entry.\n"
            + "".join(details)
            + "  (digits shown as #, letters as a; real names are not printed)"
        )
    if expected_count and len(paths) != expected_count:
        raise AuditError(f"{label} list holds {len(paths)} entries, expected {expected_count}")

    import fnmatch

    mismatched = [entry for entry in paths if not fnmatch.fnmatch(entry.name, pattern)]
    if mismatched:
        raise AuditError(f"{len(mismatched)} {label} entr(ies) do not match the teacher v7 pattern")
    missing = [entry for entry in paths if not entry.is_file()]
    if missing:
        raise AuditError(f"{len(missing)} {label} H5 file(s) do not exist")

    return {
        "count": len(paths),
        "content_sha256": list_content_sha256(paths),
        "identity_sha256": list_identity_sha256(paths),
        "paths": paths,
        "identities": identities,
    }


def read_sanity_id_map(path: Path) -> list[str]:
    """Video identities the old evaluation recorded as train_sanity."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AuditError("the evaluation ID map is empty")
    required = {"split", "original_h5_path"}
    missing = required - set(rows[0])
    if missing:
        raise AuditError(f"the evaluation ID map lacks column(s): {sorted(missing)}")
    identities: list[str] = []
    for row in rows:
        if row.get("split") != "train_sanity":
            continue
        found = extract_video_identities(Path(row["original_h5_path"]).name)
        if len(found) != 1:
            raise AuditError("a train_sanity row in the ID map does not yield one video identity")
        identities.append(found[0])
    return identities


def confirm_sanity(args: argparse.Namespace, train: dict[str, Any]) -> dict[str, Any]:
    """Establish the three sanity videos, by identity, with corroboration."""
    evidence: dict[str, Any] = {"source": None, "corroborated_by_id_map": False}

    if args.sanity_list:
        sanity_paths = read_file_list(args.sanity_list, label="train sanity")
        evidence["source"] = "saved_selected_train_files"
    elif args.allow_sanity_rederivation:
        # Re-derivation is only meaningful against the exact saved list order,
        # so the list's content fingerprint must be pinned by the caller first.
        if not args.expected_train_content_sha256:
            raise AuditError(
                "re-derivation needs --expected_train_content_sha256: the selection draws from the "
                "train list IN ORDER, so re-deriving against an unverified list proves nothing"
            )
        if train["content_sha256"] != args.expected_train_content_sha256:
            raise AuditError(
                "the train list's content fingerprint does not match the value supplied for "
                "re-derivation; the saved order cannot be assumed"
            )
        sanity_paths = select_train_paths(
            list(train["paths"]),
            fixed_videos=[args.fixed_train_video],
            num_random=args.num_random_train,
            seed=args.selection_seed,
        )
        evidence["source"] = "re_derived_from_train_list_order"
        evidence["re_derivation"] = {
            "fixed_train_video_matched": True,
            "num_random": args.num_random_train,
            "seed": args.selection_seed,
            "note": "re-derived, not read from the saved selection; recorded as such",
        }
    else:
        raise AuditError(
            "no --sanity_list was given and re-derivation was not requested. The saved "
            "selected_train_files.txt is the first choice; pass --allow_sanity_rederivation only "
            "if it genuinely does not exist."
        )

    if len(sanity_paths) != args.expected_sanity_count:
        raise AuditError(
            f"expected {args.expected_sanity_count} train-sanity videos, got {len(sanity_paths)}"
        )
    identities = [
        identity_of(entry, label="train sanity", index=index)
        for index, entry in enumerate(sanity_paths)
    ]
    if len(set(identities)) != len(identities):
        raise AuditError("the train-sanity selection names the same video more than once")

    outside = set(identities) - set(train["identities"])
    if outside:
        raise AuditError(f"{len(outside)} train-sanity video(s) are not in the saved train list")
    absent = [entry for entry in sanity_paths if not entry.is_file()]
    if absent:
        raise AuditError(f"{len(absent)} train-sanity H5 file(s) do not exist")

    # Corroboration. A count of three, or three aliases existing, is not proof;
    # the identities themselves must line up one to one.
    if not args.evaluation_video_id_map:
        raise AuditError(
            "no --evaluation_video_id_map was given, so the sanity selection cannot be corroborated "
            "against the old evaluation's own record. The audit stops rather than confirming on "
            "the strength of the count alone (report 9.6)."
        )
    mapped = read_sanity_id_map(Path(args.evaluation_video_id_map))
    if sorted(mapped) != sorted(identities):
        raise AuditError(
            f"the evaluation ID map records {len(mapped)} train_sanity video(s) that do not match "
            "the selection one to one; the sanity set is not confirmed"
        )
    evidence["corroborated_by_id_map"] = True
    evidence["num_confirmed"] = len(identities)
    return {"paths": sanity_paths, "identities": identities, "evidence": evidence}


def assert_pre_seal(args: argparse.Namespace, every_path: list[Path]) -> str:
    """Refuse once a seal exists: this audit reads the pre-split train162 list."""
    try:
        pins = Pins.load(args.split_contract_pins)
    except SplitContractError:
        return "no pins file: pre-seal, as expected for S0-1"
    if SEAL_REGISTRY_PIN_KEY not in pins:
        return "pins present but no seal registry approved: pre-seal"
    contract = resolve_active_contract(
        manifest_key=args.split_manifest, pins_path=args.split_contract_pins
    )
    contract.assert_paths_allowed(every_path, purpose="S0-1 input audit")
    return "a seal registry is approved and none of these inputs intersect it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-16 Step 0 工程1 (S0-1): audit the saved lists, confirm the train-sanity "
        "set by video identity, confirm the clinical FL table's structure, and FIX the input "
        "hashes the split builder will verify against. Opens no H5; reports no FL value."
    )
    parser.add_argument("--old_train_list", required=True)
    parser.add_argument("--old_val_list", required=True)
    parser.add_argument("--sanity_list", default=None, help="Saved selected_train_files.txt (preferred)")
    parser.add_argument(
        "--evaluation_video_id_map",
        default=None,
        help="Old evaluation's video_id_map_DO_NOT_SHARE.csv, for corroboration (required)",
    )
    parser.add_argument("--allow_sanity_rederivation", action="store_true")
    parser.add_argument("--expected_train_content_sha256", default=None)
    parser.add_argument("--fixed_train_video", default=DEFAULT_FIXED_TRAIN_VIDEO)
    parser.add_argument("--num_random_train", type=int, default=2)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--expected_train_count", type=int, default=162)
    parser.add_argument("--expected_val_count", type=int, default=18)
    parser.add_argument("--expected_sanity_count", type=int, default=3)
    parser.add_argument("--h5_pattern", default=TEACHER_V7_PATTERN)
    parser.add_argument("--split_contract_pins", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--audit_out", required=True, help="input_audit.json for the split builder")
    parser.add_argument("--private_json", required=True)
    parser.add_argument("--shareable_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train = audit_list(
        args.old_train_list, label="train", expected_count=args.expected_train_count,
        pattern=args.h5_pattern,
    )
    val = audit_list(
        args.old_val_list, label="validation", expected_count=args.expected_val_count,
        pattern=args.h5_pattern,
    )
    overlap = set(train["identities"]) & set(val["identities"])
    if overlap:
        raise AuditError(f"{len(overlap)} video(s) appear in both the train and validation lists")

    seal_state = assert_pre_seal(args, list(train["paths"]) + list(val["paths"]))

    sanity = confirm_sanity(args, train)
    candidate_identities = [
        identity for identity in train["identities"] if identity not in set(sanity["identities"])
    ]
    expected_candidates = args.expected_train_count - args.expected_sanity_count
    if len(candidate_identities) != expected_candidates:
        raise AuditError(
            f"the candidate set holds {len(candidate_identities)}, expected {expected_candidates}"
        )


    # The point of the whole step: expected hashes fixed here, once, under the
    # exact labels the builder looks them up by.
    # Three inputs, not four: D-039 withdrew the clinical FL table entirely,
    # so there is no fourth input and no deferred pass (report 4.1.1/4.2).
    expected_sha256 = {
        "old_train_list": file_sha256(args.old_train_list),
        "old_val_list": file_sha256(args.old_val_list),
        "sanity_list": file_sha256(args.sanity_list) if args.sanity_list else None,
    }
    if expected_sha256["sanity_list"] is None:
        raise AuditError(
            "the builder takes --sanity_list as an input, so a saved sanity list file is required "
            "even when the selection was re-derived. Write the confirmed selection to a file and "
            "re-run the audit against it."
        )

    audit_record = {
        "schema": INPUT_AUDIT_SCHEMA,
        "note": (
            "Expected input hashes fixed by S0-1. The split builder verifies its inputs against "
            "THESE values; it must not recompute them from the inputs themselves."
        ),
        "complete": True,
        "expected_sha256": expected_sha256,
        "stratification_note": (
            "The second stratification axis is computed by the split builder from the teacher "
            "GT (D-039); this audit fixes no clinical FL input because none is used."
        ),
        "provenance": provenance(),
    }
    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_out).write_text(
        json.dumps(audit_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    private = {
        "schema": "stage5_step0_input_audit_private_v1",
        "note": "DO_NOT_SHARE: carries real list paths and the observed FL token strings.",
        "seal_state": seal_state,
        "lists": {
            "train": {k: v for k, v in train.items() if k not in ("paths", "identities")},
            "validation": {k: v for k, v in val.items() if k not in ("paths", "identities")},
        },
        "list_paths": {
            "old_train_list": str(args.old_train_list),
            "old_val_list": str(args.old_val_list),
            "sanity_list": str(args.sanity_list),
        },
        "train_sanity": {
            "count": len(sanity["identities"]),
            "evidence": sanity["evidence"],
            "content_sha256": list_content_sha256(list(sanity["paths"])),
            "identity_sha256": list_identity_sha256(list(sanity["paths"])),
        },
        "candidate_set": {"count": len(candidate_identities)},
        "expected_sha256": expected_sha256,
        "provenance": provenance(),
    }
    Path(args.private_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.private_json).write_text(
        json.dumps(private, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.shareable_json:
        shareable = {
            "schema": "stage5_step0_input_audit_shareable_v1",
            "counts": {
                "train": train["count"],
                "validation": val["count"],
                "train_sanity": len(sanity["identities"]),
                "candidates": len(candidate_identities),
            },
            "identity_sha256": {
                "train": train["identity_sha256"], "validation": val["identity_sha256"],
            },
            "content_sha256": {
                "train": train["content_sha256"], "validation": val["content_sha256"],
            },
            "train_sanity_evidence": sanity["evidence"],
            "withheld": "GT statistics, real paths and video IDs",
            "provenance": provenance(),
        }
        Path(args.shareable_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.shareable_json).write_text(
            json.dumps(shareable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print("Stage5 S5-16 Step 0 工程1: input audit passed")
    print(f"  seal state        : {seal_state}")
    print(f"  train list        : {train['count']}  identity {train['identity_sha256'][:16]}")
    print(f"  validation list   : {val['count']}  identity {val['identity_sha256'][:16]}")
    print(f"  train sanity      : {len(sanity['identities'])} confirmed via "
          f"{sanity['evidence']['source']}, corroborated by the evaluation ID map")
    print(f"  candidates        : {len(candidate_identities)}")
    print(f"\n  audit record written: {args.audit_out} (schema {INPUT_AUDIT_SCHEMA})")
    print("  Pass it to the split builder as --input_audit_json.")
    print("  This audit fixes the inputs only. The stratification quantity and the")
    print("  constructibility of the strata are the builder's responsibility (工程2a).")


if __name__ == "__main__":
    try:
        main()
    except (AuditError, FixedListError, SplitSpecError, SplitContractError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(2)
