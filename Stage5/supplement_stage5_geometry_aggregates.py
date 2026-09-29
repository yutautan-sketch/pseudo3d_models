from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_stage5_geometry as geo  # noqa: E402
from stage5.geometry import metrics as mt  # noqa: E402
from stage5.geometry.types import GeometryContractError  # noqa: E402
from stage5.utils.file_list_mode import FixedListError  # noqa: E402
from stage5.utils.geometry_inputs import InputContractError  # noqa: E402
from stage5.utils.split_contract import SplitContractError  # noqa: E402
from stage5.utils.split_identity import COVERAGE_SCHEMA, IdentityError, extract_video_identities, file_sha256  # noqa: E402

# ----------------------------------------------------------------------------
# S5-17 S17-6: supplementary shared aggregates from the S17-5 private record.
#
# Management decision S6-5 (report 14.10.5). The S17-5 `run` computed four
# quantities per video but did not aggregate them into the shared JSON:
#   (e) pseudo-3D reference values, H17-1b BBox consistency, H17-5 multiple
#   instances, H17-2 errors of M3-matched pairs.
# This tool aggregates the VALUES ALREADY SAVED in the private record. It reads
# no H5, no NPZ and no teacher generation CSV, recomputes no geometry, reselects
# nothing and is not a re-run of the diagnosis.
#
# Two subcommands, run separately:
#
#   register   Establishes the private record's provenance WITHOUT parsing it,
#              then writes a hash-bound ArtifactCoverage entry for it. The
#              primary evidence is the OPERATOR'S ATTESTATION (management
#              decision, report 14.13.3): the operator confirms that the file
#              was written directly by the approved S17-5 run and has not been
#              overwritten, edited or replaced since, and fixes its FULL
#              SHA-256 in the report. `--attested_private_sha256` must equal the
#              file's current hash (a short hash is refused). This is trust in
#              the operator's account of generation and custody, NOT a
#              generation-time hash. Supporting metadata checks:
#                * the shared run.json has the SHA-256 fixed in the report, and
#                  its (aggregate-only) content names this contract, a clean
#                  tree, pixel mode and no minimum override
#                * the run's resource record shows exit status 0 for MODE=run,
#                  COORDINATE_MODE=pixel
#                * the private record is 0600 in the same private directory,
#                  and its mtime falls inside the run window: written before
#                  the shared JSON, which precedes the resource record, within
#                  the measured elapsed time
#              The private record's bytes are hashed, not interpreted. The run
#              did not record the private record's hash at generation time;
#              without the attestation, metadata alone does not establish the
#              link (report 14.13.2).
#   aggregate  Reads the private record only after the coverage hash, the seal
#              and the allow-list have been checked; then re-verifies alias /
#              identity / split correspondence and RECOMPUTES shared run.json
#              aggregates (FP point totals, within-tau counts, held-out
#              coverage) from the private values, requiring exact agreement.
#              Only then are the four supplementary aggregates computed. Both
#              inputs are hashed again BEFORE the shared output is committed
#              (written to a temporary file and renamed only after that check),
#              so a changed input never leaves a new shared result behind.
#
# Scope of the guarantees, kept apart: BEFORE parsing, the private record is
# bound to the attested full hash and to the coverage record (seal and
# allow-list included). AFTER parsing, the alias / identity / path / split
# checks and the recomputed shared aggregates detect a record that does not
# belong to this run. The second is detection after parsing, not refusal before
# it.
#
# Output: a separate shared JSON (the private record and run.json are never
# overwritten), privacy-checked and finite, carrying the input hashes, the code
# revision and the aggregation specification. Train_core / validation / sanity,
# best / last and post-processes are never mixed; denominators are stated;
# undefined values are counted, never zero-filled.
# ----------------------------------------------------------------------------

PRIVATE_RECORD_NAME = "geometry_run_DO_NOT_SHARE.json"
PURPOSE_SUPPLEMENT = "supplement_private_record"
SUPPLEMENT_SCHEMA = "stage5_s5_17_geometry_supplement_v1"
PREDICTION_GROUPS = ("validation", "sanity_in_sample")
CHECKPOINTS = ("best", "last")
PAIR_METRICS = (
    ("center_error", "raw_px"),
    ("endpoint_error_mean", "raw_px"),
    ("endpoint_error_max", "raw_px"),
    ("axis_angle_error_deg", "deg"),
    ("length_relative_error", "ratio"),
)
MTIME_SLACK_S = 5.0
_ELAPSED = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")


def _elapsed_seconds(text: str) -> float:
    match = _ELAPSED.search(text)
    if not match:
        raise InputContractError("the resource record has no elapsed wall-clock time")
    hours, minutes, seconds = match.group(1), match.group(2), match.group(3)
    return (int(hours) if hours else 0) * 3600 + int(minutes) * 60 + float(seconds)


def _context(args: argparse.Namespace, *, coverage_path: str | None) -> geo.Context:
    ctx = geo.load_context(args, coverage_path=coverage_path)
    ctx.allow[PURPOSE_SUPPLEMENT] = ctx.allow[geo.PURPOSE_GT] | ctx.allow[geo.PURPOSE_PRED]
    return ctx


def _verify_shared_run(ctx: geo.Context, path: Path, expected_sha256: str) -> dict[str, Any]:
    """The shared run.json: fixed hash (from the report) and its aggregate-only content."""
    if not path.is_file():
        raise InputContractError("the shared run JSON is missing")
    if file_sha256(path) != expected_sha256.lower():
        raise InputContractError("the shared run JSON does not have the SHA-256 recorded in the report")
    shared = json.loads(path.read_text(encoding="utf-8"))
    geo.assert_shareable(shared)                       # aggregate-only: no identity, no path
    lists = ctx.contract.manifest.lists
    checks = (
        shared.get("schema") == geo.SHARED_SCHEMA,
        shared.get("coordinate_mode") == "pixel",
        (shared.get("git") or {}).get("dirty") is False,
        (shared.get("input_minimums") or {}).get("synthetic_override") is False,
        (shared.get("contract") or {}).get("train_core_identity16") == lists["train_core"]["identity_sha256"][:16],
        (shared.get("contract") or {}).get("validation_identity16") == lists["validation"]["identity_sha256"][:16],
    )
    if not all(checks):
        raise InputContractError("the shared run JSON is not a clean pixel-mode run of this contract")
    return shared


# ----------------------------------------------------------------------------
# register
# ----------------------------------------------------------------------------


def cmd_register(args: argparse.Namespace) -> int:
    ctx = _context(args, coverage_path=None)
    private = Path(args.private_run_json)
    shared_run = Path(args.shared_run_json)
    resource = Path(args.resource_usage)
    for path, what in ((private, "private record"), (resource, "resource record")):
        if not path.is_file():
            raise InputContractError(f"the {what} is missing")
    if private.name != PRIVATE_RECORD_NAME or private.parent != resource.parent \
            or shared_run.parent != private.parent / "shared":
        raise InputContractError("the private record, shared JSON and resource record are not one run's layout")
    if stat.S_IMODE(private.stat().st_mode) != 0o600:
        raise InputContractError("the private record is not 0600 as the run writes it")

    attested = str(args.attested_private_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", attested):
        raise InputContractError("the attested private-record hash must be a full 64-hex SHA-256")
    if not str(args.attestation_reference).strip():
        raise InputContractError("an attestation reference (where the operator's confirmation is recorded) is required")
    if file_sha256(private) != attested:
        raise InputContractError("the private record's current SHA-256 differs from the operator-attested value")

    _verify_shared_run(ctx, shared_run, args.expected_shared_run_sha256)
    text = resource.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"Exit status:\s*0\b", text) or "MODE=run" not in text or "COORDINATE_MODE=pixel" not in text:
        raise InputContractError("the resource record is not a successful MODE=run COORDINATE_MODE=pixel execution")
    elapsed = _elapsed_seconds(text)

    p_time, s_time, r_time = (path.stat().st_mtime for path in (private, shared_run, resource))
    if not (p_time <= s_time + 1.0 and s_time <= r_time + 1.0 and r_time - p_time <= elapsed + MTIME_SLACK_S):
        raise InputContractError("the private record's modification time is outside the run window")

    identities = sorted(ctx.allow[PURPOSE_SUPPLEMENT])
    digest = file_sha256(private)                      # bytes hashed, not interpreted
    evidence = {
        "shared_run_sha256": args.expected_shared_run_sha256.lower(),
        "resource_exit_status_0_mode_run_pixel": True,
        "elapsed_seconds": elapsed,
        "private_mode_0600": True,
        "mtime_order_private_shared_resource_within_window": True,
        "contents_parsed_before_registration": False,
        "operator_attestation": {
            "reference": str(args.attestation_reference).strip(),
            "attested_full_sha256_matches": True,
            "statements": ["written directly by the approved S17-5 run",
                           "not overwritten, edited or replaced since"],
        },
        "link_basis": ("operator attestation of generation and custody, supported by file metadata; "
                       "not a generation-time hash"),
        "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": geo.git_state(),
    }
    out = Path(args.coverage_out)
    entries: list[dict[str, Any]] = []
    if out.is_file():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing.get("schema") != COVERAGE_SCHEMA:
            raise InputContractError("the existing coverage file has a different schema")
        entries = [e for e in existing.get("entries", [])
                   if not (e["artifact_name"] == private.name and e["sha256"] == digest)]
    entries.append({"artifact_name": private.name, "sha256": digest, "video_identities": identities,
                    "provenance": evidence})
    geo.write_private_json(out.parent, out.name, {"schema": COVERAGE_SCHEMA, "entries": entries})
    if args.shared_json:
        geo.write_shared_json(Path(args.shared_json), {
            "schema": "stage5_s5_17_supplement_registration_v1",
            "private_record_sha256_16": digest[:16],
            "n_videos_covered": len(identities),
            "evidence": {k: v for k, v in evidence.items() if k != "git"},
            "git_dirty": evidence["git"]["dirty"],
        })
    print(f"private record registered: {len(identities)} videos")
    return 0


# ----------------------------------------------------------------------------
# aggregate: verification after parsing
# ----------------------------------------------------------------------------


def _verify_structure(ctx: geo.Context, record: dict[str, Any], shared: dict[str, Any]) -> None:
    for key in ("alias_map", "train_core_records", "prediction_records", "inputs"):
        if key not in record:
            raise SplitContractError(f"the private record lacks {key!r}")
    expected_alias = {alias: identity for identity, alias in ctx.alias_of.items()}
    alias_map = record["alias_map"]
    if set(alias_map) != set(expected_alias):
        raise SplitContractError("the private record's aliases are not exactly this contract's train_core + validation")
    for alias, entry in alias_map.items():
        identity = entry.get("identity")
        path = str(entry.get("path", ""))
        # Both the identity AND the exact listed path string: the run wrote ctx.path_of[identity] verbatim.
        if identity != expected_alias[alias] or extract_video_identities(Path(path).name) != (identity,) \
                or path != str(ctx.path_of[identity]):
            raise SplitContractError("an alias in the private record maps to a different video or path than the contract")

    core_aliases = {ctx.alias_of[i] for i in ctx.allow[geo.PURPOSE_GT]}
    inputs = record["inputs"]
    if set(inputs) != core_aliases:
        raise SplitContractError("the private record's input list is not exactly train_core")
    evaluable = {a for a, v in inputs.items() if not v.get("input_failure")}
    if set(record["train_core_records"]) != evaluable:
        raise SplitContractError("the train_core records are not exactly the input-evaluable train_core videos")

    expected_groups = {f"{g}/{c}" for g in PREDICTION_GROUPS for c in CHECKPOINTS}
    if set(record["prediction_records"]) != expected_groups:
        raise SplitContractError("the private record's prediction groups are not validation/sanity x best/last")
    val_aliases = {ctx.alias_of[i] for i in ctx.allow[geo.PURPOSE_PRED]}
    sanity_aliases = {ctx.alias_of[i] for i in ctx.sanity_ids}
    for group in expected_groups:
        keys = set(record["prediction_records"][group])
        allowed = val_aliases if group.startswith("validation/") else sanity_aliases
        table = shared["tables"]["validation" if group.startswith("validation/") else "sanity_in_sample"]
        if not keys <= allowed or len(keys) != int(table["input"]["evaluable"]):
            raise SplitContractError(f"the {group} records do not match the expected videos and count")


def _verify_against_shared_run(record: dict[str, Any], shared: dict[str, Any]) -> dict[str, int]:
    """Recompute shared run.json aggregates from the private values; require exact agreement."""
    checked = Counter()
    for rep, summary in shared["h17_1"]["per_representation"].items():
        values = [r["holdout"]["value"][rep] for r in record["train_core_records"].values()]
        recomputed = mt.summarize(values)
        expected = summary["holdout_coverage_descriptive"]
        if (recomputed["n_defined"], recomputed["median"]) != (expected["n_defined"], expected["median"]):
            raise SplitContractError("the private record does not reproduce the shared run's held-out coverage")
        checked["h17_1_holdout"] += 1
    for group in PREDICTION_GROUPS:
        for checkpoint in CHECKPOINTS:
            records = record["prediction_records"][f"{group}/{checkpoint}"]
            table = shared["tables"][group]["checkpoints"][checkpoint]["postprocess"]
            for name, entry in table.items():
                n_fp = sum(r["postprocess"][name]["fp_classes"]["n_fp"] for r in records.values())
                if n_fp != entry["fp_classes_pooled"]["n_fp"]:
                    raise SplitContractError("the private record does not reproduce the shared run's FP totals")
                for rep, rep_entry in entry["representations"].items():
                    within = sum(int(bool(r["postprocess"][name]["representations"][rep]["within_tau_rel_diagnostic"]))
                                 for r in records.values())
                    if within != rep_entry["within_tau_rel_diagnostic"]["count"]:
                        raise SplitContractError("the private record does not reproduce the shared run's counts")
                    checked["within_counts"] += 1
                checked["fp_totals"] += 1
    return dict(checked)


# ----------------------------------------------------------------------------
# aggregate: the four supplementary quantities
# ----------------------------------------------------------------------------


def _pseudo3d(records: dict[str, Any]) -> dict[str, Any]:
    values, failures, absent = [], Counter(), 0
    for record in records.values():
        reference = record.get("pseudo3d_reference")
        if reference is None:
            absent += 1
            continue
        if reference.get("failure"):
            failures[reference["failure"]] += 1
            values.append(None)
        else:
            values.append(reference["value"])
    return {
        "denominator": "input-evaluable train_core videos",
        "n_videos": len(records),
        "n_without_gt_points": absent,
        "failures_by_reason": dict(sorted(failures.items())),
        "value_pseudo3d_units": mt.summarize(values),
        "unit": "pseudo3d_units",
        "note": "reference only, never a candidate; not mm and not a bone length",
    }


def _bbox_consistency(records: dict[str, Any]) -> dict[str, Any]:
    iou, long_diff, frames, videos = [], [], 0, 0
    for record in records.values():
        entry = record.get("bbox_consistency") or {}
        n = int(entry.get("n_frames", 0))
        if n == 0:
            continue
        videos += 1
        frames += n
        iou.append((entry.get("iou") or {}).get("median"))
        long_diff.append((entry.get("long_side_rel_diff") or {}).get("median"))
    return {
        "denominator": "videos with >= 1 compared frame; each value is that video's per-frame median",
        "n_videos_compared": videos,
        "n_videos_without_compared_frames": len(records) - videos,
        "n_frames_compared_total": frames,
        "iou_video_median": mt.summarize(iou),
        "long_side_relative_difference_video_median": mt.summarize(long_diff),
        "note": ("region enclosing all BBoxes vs region enclosing all GT instances; descriptive, not per-instance "
                 "agreement. Frame-level values were saved only as per-video summaries, so pooled frame statistics "
                 "are not available."),
    }


def _multiple_instances(records: dict[str, Any]) -> dict[str, Any]:
    multi = large = videos_with_multi = 0
    for record in records.values():
        entry = record.get("h17_5_gt") or {}
        m = int(entry.get("multi_instance_frames", 0))
        multi += m
        large += int(entry.get("m1_m2_length_diff_over_10pct", 0))
        videos_with_multi += int(m > 0)
    return {
        "denominator": "GT frames with >= 2 instances (train_core, input-evaluable)",
        "n_videos": len(records),
        "n_videos_with_multi_instance_frames": videos_with_multi,
        "multi_instance_frames": multi,
        "m1_m2_length_diff_over_10pct_frames": large,
        "share_of_multi_instance_frames": mt.safe_ratio(large, multi),
        "note": "descriptive; M1 vs M2 (b) length difference > 10 %; neither rule is judged correct",
    }


def _matched_pairs(record: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in PREDICTION_GROUPS:
        out[group] = {}
        for checkpoint in CHECKPOINTS:
            records = record["prediction_records"][f"{group}/{checkpoint}"]
            names = list(shared["tables"][group]["checkpoints"][checkpoint]["postprocess"])
            out[group][checkpoint] = {}
            for name in names:
                pairs = [pair for r in records.values() for pair in r["postprocess"][name]["matched_pairs"]]
                entry: dict[str, Any] = {
                    "denominator": "M3-matched GT/prediction instance pairs (conditional on a match)",
                    "n_pairs": len(pairs),
                    "n_videos_with_pairs": sum(1 for r in records.values() if r["postprocess"][name]["matched_pairs"]),
                }
                for metric, unit in PAIR_METRICS:
                    entry[metric] = {"unit": unit, **mt.summarize([p.get(metric) for p in pairs])}
                entry["abs_length_relative_error"] = {
                    "unit": "ratio",
                    **mt.summarize([None if p.get("length_relative_error") is None else abs(p["length_relative_error"])
                                    for p in pairs]),
                }
                out[group][checkpoint][name] = entry
    out["note"] = ("errors of matched pairs only: unmatched GT (FN) and unmatched predictions (FP) are not included, "
                   "so this is not overall performance. center_error is the box centre; matching used centroids.")
    return out


def cmd_aggregate(args: argparse.Namespace) -> int:
    ctx = _context(args, coverage_path=args.artifact_coverage)
    if ctx.contract.coverage is None:
        raise SplitContractError("a coverage record for the private record is required")
    shared_run_path = Path(args.shared_run_json)
    shared = _verify_shared_run(ctx, shared_run_path, args.expected_shared_run_sha256)
    private = Path(args.private_run_json)
    if not private.is_file():
        raise InputContractError("the private record is missing")
    geo.guard(ctx, [private], PURPOSE_SUPPLEMENT)          # coverage hash, seal, allow-list -- before parsing
    private_sha = file_sha256(private)
    record = json.loads(private.read_text(encoding="utf-8"))
    _verify_structure(ctx, record, shared)
    checked = _verify_against_shared_run(record, shared)

    core = record["train_core_records"]
    payload = {
        "schema": SUPPLEMENT_SCHEMA,
        "supplements": "stage5_s5_17_geometry_summary_v1 (S17-5 run)",
        "inputs": {
            "shared_run_sha256": file_sha256(shared_run_path),
            "private_record_sha256_16": private_sha[:16],
            "reads": "private record values only; no H5, NPZ or teacher generation CSV; no recomputation",
        },
        "verification": {"alias_identity_split": "exact", "recomputed_shared_aggregates": checked},
        "code": {"git": {"dirty": geo.git_state()["dirty"], "head": geo.git_state()["head"]}},
        "coordinate_space": "raw_frame_px",
        "e_pseudo3d_reference": _pseudo3d(core),
        "h17_1b_bbox_consistency": _bbox_consistency(core),
        "h17_5_multiple_instances": _multiple_instances(core),
        "h17_2_matched_pairs": _matched_pairs(record, shared),
        "limits": list(geo.LIMITS) + [
            "The private record's link to the run rests on the operator's attestation of generation and custody "
            "with supporting file metadata, not on a generation-time hash.",
        ],
    }
    out = Path(args.shared_json)
    for path in (private, shared_run_path):
        if out.resolve() == path.resolve():
            raise InputContractError("the supplement must not overwrite its inputs")
    commit_shared_output(out, payload, checks=((private, private_sha),
                                              (shared_run_path, args.expected_shared_run_sha256.lower())))
    print("supplement complete: aggregates written to the shared JSON")
    return 0


def commit_shared_output(out: Path, payload: dict[str, Any], *, checks: Sequence[tuple[Path, str]]) -> None:
    """Privacy-check, write to a temporary file, re-hash every input, and only then rename into place.

    If any input changed since it was verified, the temporary file is removed and nothing new is
    published.
    """
    text = geo.assert_shareable(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".partial")
    temporary.write_text(text + "\n", encoding="utf-8")
    try:
        for path, expected in checks:
            if file_sha256(path) != expected:
                raise InputContractError("an input changed during aggregation; no new shared result was published")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, out)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5-17 supplementary aggregates from the S17-5 private record.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("register", cmd_register), ("aggregate", cmd_aggregate)):
        cmd = sub.add_parser(name)
        cmd.add_argument("--split_manifest", default=geo.DEFAULT_MANIFEST)
        cmd.add_argument("--split_contract_pins", default=None)
        cmd.add_argument("--train_core_list", required=True)
        cmd.add_argument("--validation_list", required=True)
        cmd.add_argument("--train_sanity_list", required=True)
        cmd.add_argument("--private_run_json", required=True)
        cmd.add_argument("--shared_run_json", required=True)
        cmd.add_argument("--expected_shared_run_sha256", required=True, help="the value fixed in the report")
        if name == "register":
            cmd.add_argument("--attested_private_sha256", required=True,
                             help="full SHA-256 of the private record fixed in the report by the operator's attestation")
            cmd.add_argument("--attestation_reference", required=True,
                             help="where the operator's confirmation is recorded (e.g. the report section)")
            cmd.add_argument("--resource_usage", required=True)
            cmd.add_argument("--coverage_out", required=True)
            cmd.add_argument("--shared_json", default=None)
        else:
            cmd.add_argument("--artifact_coverage", required=True)
            cmd.add_argument("--shared_json", required=True)
        cmd.set_defaults(func=func)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except (SplitContractError, InputContractError, GeometryContractError, IdentityError, FixedListError,
            geo.PrivacyError) as error:
        print(f"STOP ({error.__class__.__name__}): {geo.scrub(str(error))}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 -- the traceback may carry paths; keep it private
        directory = Path(args.private_run_json).parent
        try:
            target = directory / "supplement_error_traceback_DO_NOT_SHARE.txt"
            target.write_text(traceback.format_exc(), encoding="utf-8")
            os.chmod(target, 0o600)
            where = "written to the private directory"
        except OSError:
            where = "suppressed"
        print(f"STOP (unexpected {error.__class__.__name__}); traceback {where}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
