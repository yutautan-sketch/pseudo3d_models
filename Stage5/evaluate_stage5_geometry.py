from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from stage5.geometry import fl_estimate as fl  # noqa: E402
from stage5.geometry import metrics as mt  # noqa: E402
from stage5.geometry import postprocess as pp  # noqa: E402
from stage5.geometry import prior as pr  # noqa: E402
from stage5.geometry.frame_geometry import frame_universe_of, group_by_frame, video_frame_geometries  # noqa: E402
from stage5.geometry.transform import CropTransform  # noqa: E402
from stage5.geometry.types import (  # noqa: E402
    CoordSpace,
    FailureReason,
    GeometryContractError,
    InputUnevaluable,
    Points2D,
)
from stage5.utils.file_list_mode import (  # noqa: E402
    FixedListError,
    list_content_sha256,
    list_identity_sha256,
    read_file_list,
)
from stage5.utils.geometry_inputs import (  # noqa: E402
    CONFUSION_KEYS,
    InputContractError,
    TeacherArrays,
    confusion_counts,
    expected_vote_count,
    h5_dataset_meta,
    npz_member_shapes,
    read_h5_metrics_rows,
    read_intermediate_meta,
    read_prediction_npz,
    read_summary_json,
    read_teacher_arrays,
    read_teacher_attrs,
    resolve_intermediate_exact,
    single_identity,
)
from stage5.utils.split_contract import ActiveContract, SplitContractError, resolve_active_contract  # noqa: E402
from stage5.utils.split_identity import (  # noqa: E402
    COVERAGE_SCHEMA,
    VIDEO_ID_PATTERN,
    IdentityError,
    contains_video_identity,
    extract_video_identities,
    file_sha256,
    masked_shape,
    resolve_identities,
)

# ----------------------------------------------------------------------------
# S5-17 S17-2: geometry diagnosis CLI -- contract, inputs, provenance, outputs.
#
# Management document 3 / 5 / 7. Three subcommands, run separately:
#
#   register-coverage  Registers the multi-video artifacts of ONE evaluation
#                      checkpoint directory (`summary.json`, `h5_metrics.csv`)
#                      as a hash-bound ArtifactCoverage record. Provenance is
#                      established WITHOUT parsing either file: the directory
#                      the user names, the validation-list hash fixed in the
#                      git-tracked S5-15 launcher, and the FILE NAMES of the
#                      saved predictions (sanity3 + validation18 exactly). The
#                      bytes are hashed, not interpreted. Name listing is a
#                      provenance check only; it never selects inputs.
#   audit              S17-4 fail-fast, metadata only: dataset shapes/dtypes,
#                      attrs, intermediate resolution method, crop mode and
#                      invertibility, NPZ headers, coverage and hash states.
#   run                S17-5 diagnosis: H17-1 on train_core GT, prior and
#                      post-process constants from train_core, H17-2..H17-5
#                      on validation (best main, last auxiliary), sanity3
#                      in-sample in a separate table.
#
# Order of reading (every subcommand): pins -> registry -> manifest; explicit
# lists checked against the manifest's full identity SHA-256; purpose
# allow-lists; seal guard AND allow-list on every path BEFORE it is opened
# (teacher before its attrs, the resolved intermediate before it is opened,
# each NPZ built from the allow-list, never found by listing).
#
# Outputs: a private directory (0700, files 0600) with per-video records and
# the alias map, and a shared JSON of aggregates that must pass a privacy
# check (no video identity of either naming convention, no absolute path)
# before it is written. Error messages are scrubbed of identities and paths;
# tracebacks go to the private directory only.
#
# Not here: training, GPU, re-inference, production changes, mm results
# (`--coordinate_mode mm` stops until the mm source contract is fixed).
# ----------------------------------------------------------------------------

DEFAULT_MANIFEST = "s5_16_bprime"
PURPOSE_GT = "gt_prior"
PURPOSE_PRED = "prediction_diagnosis"
PURPOSE_IN_SAMPLE = "in_sample_diagnosis"
PURPOSE_RUN_ARTIFACT = "evaluation_run_artifact"
CHECKPOINT_EPOCHS = {"best": 6, "last": 50}
EXPECTED_WINDOW = (16, 8, True)
S5_15_LAUNCHER = REPO_ROOT / "checks" / "real_h5" / "run_stage5_s5_15_arm.sh"
TAU_REL_DIAGNOSTIC = 0.10
SHARED_SCHEMA = "stage5_s5_17_geometry_summary_v1"
INPUT_MIN_TRAIN_CORE = 130
INPUT_MIN_VALIDATION = 16
MODE_SHARE_LIMIT = 0.10
H17_5_REL_DIFF = 0.10
# Management decision S4-3 (report 12.9): when H17-1 yields no candidate, H17-2 is computed for
# exactly these three representations -- (a), (c) and (d); (b) shares (c)'s length and region.
# Diagnostic targets only, never adoption candidates, fixed before any real data is read.
FALLBACK_H17_2_REPRESENTATIONS = (fl.REP_AABB, fl.REP_AXIS, fl.REP_BEST_FRAME)
RAW = CoordSpace.RAW_FRAME_PX
LOCAL = CoordSpace.LOCAL_CROP_PX

P3_NOTE = (
    "P3 is applied only to frames in which the saved pred_label has at least one positive point; it compares "
    "point selection inside candidate-positive frames. It cannot recover FN frames without a predicted positive, "
    "and a P3 target frame is not necessarily a geometric present frame (a component of >= 5 points after "
    "post-processing)."
)
METRIC_NOTE = (
    "matching_centroid_distance is the M3 assignment distance between centroids; center_error is the error of the "
    "box centre (t_mid, s_mid). They are different quantities."
)
LIMITS = (
    "GT-derived quantities only; no clinical reference; not a clinical FL accuracy.",
    "Lengths are in original-frame pixels assuming isotropic x/y; mm is not established.",
    "Files are the unit; same-exam sibling files are present, so per-video statistics understate exam-level spread.",
    "Validation is a development set already used for method choices; the best post-process is chosen on it.",
    "Old train/validation share 4 exam groups (17 files in total across both); the validation-side count is unconfirmed.",
    "Single training run and seed.",
)


class PrivacyError(ValueError):
    """A shareable payload failed the privacy check; nothing is written."""


# ----------------------------------------------------------------------------
# output boundary
# ----------------------------------------------------------------------------

_PATH_LIKE = re.compile(r"(?:[A-Za-z]:\\|/)[^\s'\"]*/[^\s'\"]*")


def scrub(text: str) -> str:
    """Replace video identities by their masked shape and path-like tokens by <path>."""
    text = VIDEO_ID_PATTERN.sub(lambda m: masked_shape(m.group(1)), str(text))
    return _PATH_LIKE.sub("<path>", text)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def assert_shareable(payload: Any) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default,
                          allow_nan=False)
    except ValueError as error:
        raise PrivacyError("the shareable payload contains a non-finite number; nothing was written") from error
    if contains_video_identity(text):
        raise PrivacyError("the shareable payload contains a video identity; nothing was written")
    if re.search(r'"(?:[A-Za-z]:\\\\|/)[^"]*"', text) or "/mnt/" in text or "/home/" in text:
        raise PrivacyError("the shareable payload contains an absolute path; nothing was written")
    return text


def write_shared_json(path: Path, payload: Any) -> None:
    text = assert_shareable(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_private_json(directory: Path, name: str, payload: Any) -> Path:
    target = private_dir(directory) / name
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + "\n",
                      encoding="utf-8")
    os.chmod(target, 0o600)
    return target


def git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {"head": run(["rev-parse", "HEAD"]), "dirty": bool(run(["status", "--short"]))}


def safe_name(value: str) -> str:
    """The evaluation's NPZ naming (evaluate_stage5.safe_name), restated to avoid importing torch."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


# ----------------------------------------------------------------------------
# contract and allow-lists
# ----------------------------------------------------------------------------


@dataclass
class Context:
    contract: ActiveContract
    train_core: list[Path]
    validation: list[Path]
    sanity_ids: list[str]
    identity_of: dict[str, str]
    path_of: dict[str, Path]
    alias_of: dict[str, str]
    allow: dict[str, set[str]]


def _list_identities(paths: Sequence[Path], label: str) -> list[str]:
    ids = [single_identity(p.name, what=f"{label} #{i}") for i, p in enumerate(paths)]
    if len(set(ids)) != len(ids):
        raise SplitContractError(f"{label} names the same video identity more than once")
    return ids


def _assert_step0_sanity(contract: ActiveContract, sanity_list: Path) -> None:
    """The sanity list must be the file Step 0 used: its SHA-256 is in the pinned manifest's input record.

    The manifest was hash-verified against its pin when the contract resolved; it is re-hashed here
    before its raw payload is read, so a file swapped in between is not trusted.
    """
    source = contract.manifest.source_path
    if file_sha256(source) != contract.manifest.sha256:
        raise SplitContractError("the split manifest changed after it was verified against its pin")
    recorded = (json.loads(source.read_text(encoding="utf-8")).get("input_sha256") or {}).get("sanity_list")
    if not recorded:
        raise SplitContractError("the manifest records no Step 0 sanity-list hash, so sanity3 cannot be confirmed")
    if not sanity_list.is_file() or file_sha256(sanity_list) != recorded:
        raise SplitContractError(
            "the train_sanity list is not the Step 0 sanity list (its file SHA-256 differs from the manifest's input record)"
        )


def load_context(args: argparse.Namespace, *, coverage_path: str | None) -> Context:
    contract = resolve_active_contract(
        manifest_key=args.split_manifest, pins_path=args.split_contract_pins, coverage_path=coverage_path
    )
    train_core = read_file_list(args.train_core_list, label="train_core")
    validation = read_file_list(args.validation_list, label="validation")
    sanity = read_file_list(args.train_sanity_list, label="train_sanity")
    for key, paths in (("train_core", train_core), ("validation", validation)):
        expected = (contract.manifest.lists.get(key) or {}).get("identity_sha256")
        if not expected or list_identity_sha256(paths) != expected:
            raise SplitContractError(f"the {key} list is not the manifest's {key} (full identity SHA-256 differs)")
    _assert_step0_sanity(contract, Path(args.train_sanity_list))
    core_ids = _list_identities(train_core, "train_core")
    val_ids = _list_identities(validation, "validation")
    sanity_ids = _list_identities(sanity, "train_sanity")
    if len(sanity_ids) != 3 or not set(sanity_ids) <= set(core_ids):
        raise SplitContractError("train_sanity must be exactly 3 videos, all inside train_core")
    identity_of = {str(p): i for p, i in zip(train_core, core_ids)}
    identity_of.update({str(p): i for p, i in zip(validation, val_ids)})
    path_of = {i: p for p, i in zip(train_core, core_ids)}
    path_of.update({i: p for p, i in zip(validation, val_ids)})
    alias_of = {i: f"train_core_{k:03d}" for k, i in enumerate(core_ids)}
    alias_of.update({i: f"validation_{k:02d}" for k, i in enumerate(val_ids)})
    allow = {
        PURPOSE_GT: set(core_ids),
        PURPOSE_PRED: set(val_ids),
        PURPOSE_IN_SAMPLE: set(sanity_ids),
        PURPOSE_RUN_ARTIFACT: set(val_ids) | set(sanity_ids),
    }
    return Context(contract, train_core, validation, sanity_ids, identity_of, path_of, alias_of, allow)


def guard(ctx: Context, paths: Sequence[Path], purpose: str) -> None:
    """Seal refusal (registry first) and the purpose allow-list, before anything is opened."""
    ctx.contract.assert_paths_allowed(list(paths), purpose=purpose)
    allowed = ctx.allow[purpose]
    for index, path in enumerate(paths):
        identities = resolve_identities(Path(path), coverage=ctx.contract.coverage)
        outside = set(identities) - allowed
        if outside:
            raise SplitContractError(
                f"{purpose}: input #{index} covers {len(outside)} video(s) outside this purpose's allow-list"
            )


# ----------------------------------------------------------------------------
# per-video inputs
# ----------------------------------------------------------------------------


@dataclass
class VideoInput:
    identity: str
    alias: str
    video_name: str
    teacher: TeacherArrays | None = None
    transform: CropTransform | None = None
    input_failure: str | None = None
    preprocess_mode: str | None = None
    resolution_method: str | None = None
    local: Points2D | None = None
    raw: Points2D | None = None
    universe: tuple[int, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_wh(self) -> tuple[int, int]:
        assert self.transform is not None
        return (self.transform.raw_width, self.transform.raw_height)

    @property
    def local_wh(self) -> tuple[int, int]:
        assert self.transform is not None
        return (self.transform.local_width, self.transform.local_height)


def load_video(ctx: Context, identity: str, purpose: str, args: argparse.Namespace, *, read_arrays: bool = True
               ) -> VideoInput:
    path = ctx.path_of[identity]
    guard(ctx, [path], purpose)
    if not path.is_file():
        raise InputContractError(f"{ctx.alias_of[identity]}: listed teacher H5 is missing")
    attrs = read_teacher_attrs(path)
    video = VideoInput(identity=identity, alias=ctx.alias_of[identity], video_name=attrs["video_name"])
    intermediate, method = resolve_intermediate_exact(
        teacher_identity=identity,
        video_name=attrs["video_name"],
        recorded_source=attrs["source_pseudo3d_h5"],
        intermediate_root=Path(args.intermediate_root),
        suffix=args.intermediate_suffix,
    )
    guard(ctx, [intermediate], purpose)
    video.resolution_method = method
    meta = read_intermediate_meta(intermediate)
    video.preprocess_mode = str(meta.attrs.get("local_preprocess_effective", "<missing>"))
    try:
        video.transform = CropTransform.from_attrs(meta.attrs, local_shape_hw=meta.local_shape_hw)
    except InputUnevaluable as error:
        video.input_failure = error.reason
    if not read_arrays:
        return video

    teacher = read_teacher_arrays(path)
    if teacher.n_points and int(teacher.frame_order.max()) >= meta.n_local_frames:
        raise InputContractError(f"{video.alias}: frame_order reaches the number of local frames")
    if teacher.bbox_frame_order is not None and teacher.bbox_frame_order.size and \
            int(teacher.bbox_frame_order.max()) >= meta.n_local_frames:
        raise InputContractError(f"{video.alias}: frame_annotation frame_order reaches the number of local frames")
    height, width = meta.local_shape_hw
    xy = teacher.pixel_xy
    if xy.size and not (np.all((xy[:, 0] >= 0) & (xy[:, 0] < width)) and np.all((xy[:, 1] >= 0) & (xy[:, 1] < height))):
        raise InputContractError(f"{video.alias}: pixel_xy outside the local crop")
    video.teacher = teacher
    video.local = Points2D(xy, LOCAL)
    video.universe = frame_universe_of(teacher.frame_order)
    if video.transform is not None:
        video.raw = video.transform.local_to_raw(video.local)
    return video


# ----------------------------------------------------------------------------
# evaluation-run artifacts
# ----------------------------------------------------------------------------


@dataclass
class EvaluationRun:
    name: str
    directory: Path
    summary: dict[str, Any]
    rows: dict[tuple[str, str], dict[str, str]]
    hash_state: dict[str, str]


def _hash_state(actual_path: Path, expected: str | None) -> str:
    """'match' / 'unknown'; a recorded hash that does not match stops (never downgraded to unknown)."""
    if not expected:
        return "unknown"
    if not actual_path.is_file():
        raise InputContractError("a file with a recorded hash is missing")
    if file_sha256(actual_path) != expected.lower():
        raise InputContractError("a recorded hash does not match the current file")
    return "match"


def load_evaluation_run(ctx: Context, directory: Path, name: str, args: argparse.Namespace) -> EvaluationRun:
    if ctx.contract.coverage is None:
        raise SplitContractError("an artifact coverage record is required for summary.json / h5_metrics.csv")
    files = [directory / "summary.json", directory / "h5_metrics.csv"]
    for artifact in files:
        if not artifact.is_file():
            raise InputContractError(f"{name}: {artifact.name} is missing")
    guard(ctx, files, PURPOSE_RUN_ARTIFACT)          # coverage hash, seal, allow-list -- before parsing
    summary = read_summary_json(files[0])
    rows = read_h5_metrics_rows(files[1])

    expected = {"train_sanity": set(ctx.sanity_ids), "validation": ctx.allow[PURPOSE_PRED]}
    for split, key in (("train_sanity", "selected_train_files"), ("validation", "validation_files")):
        listed = [single_identity(Path(p).name, what=f"{name} summary {split} #{i}") for i, p in enumerate(summary[key])]
        if len(set(listed)) != len(listed) or set(listed) != expected[split]:
            raise SplitContractError(f"{name}: summary.json {split} list is not exactly the expected videos")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for index, row in enumerate(rows):
        split = row["split"]
        identity = single_identity(Path(row["h5_path"]).name, what=f"{name} h5_metrics row #{index}")
        if split not in expected or identity not in expected[split]:
            raise SplitContractError(f"{name}: h5_metrics row #{index} is outside the expected videos")
        if extract_video_identities(row["video_name"]) != (identity,) or (split, identity) in by_key:
            raise SplitContractError(f"{name}: h5_metrics row #{index} is inconsistent or duplicated")
        by_key[(split, identity)] = row
    for split, ids in expected.items():
        if {i for s, i in by_key if s == split} != ids:
            raise SplitContractError(f"{name}: h5_metrics does not cover exactly the expected {split} videos")

    if int(summary["checkpoint_epoch"]) != CHECKPOINT_EPOCHS[name] or Path(summary["checkpoint"]).name != f"{name}.pt":
        raise InputContractError(f"{name}: summary.json names a different checkpoint or epoch")
    window = (int(summary["window_size_frames"]), int(summary["window_stride_frames"]),
              bool(summary["include_tail_window"]))
    if window != EXPECTED_WINDOW:
        raise InputContractError(f"{name}: evaluation window {window} differs from {EXPECTED_WINDOW}")
    hash_state = {
        "checkpoint": _hash_state(Path(summary["checkpoint"]), getattr(args, f"checkpoint_sha256_{name}", None)),
        "evaluation_revision": "recorded" if getattr(args, f"evaluation_revision_{name}", None) else "unknown",
    }
    return EvaluationRun(name, directory, summary, by_key, hash_state)


def load_prediction(ctx: Context, run: EvaluationRun, video: VideoInput, split: str, purpose: str):
    npz = run.directory / "predictions" / split / f"{safe_name(video.video_name)}.npz"
    guard(ctx, [npz], purpose)
    if extract_video_identities(npz.name) != (video.identity,):
        raise SplitContractError(f"{video.alias}: the expected NPZ name does not carry exactly this video's identity")
    if not npz.is_file():
        raise InputContractError(f"{run.name}/{video.alias}: expected prediction NPZ is missing (no fallback)")
    assert video.teacher is not None
    pred = read_prediction_npz(npz, n_points=video.teacher.n_points)
    size, stride, tail = EXPECTED_WINDOW
    if not np.array_equal(pred.vote_count, expected_vote_count(video.teacher.frame_order, window_size=size,
                                                               stride=stride, include_tail=tail)):
        raise InputContractError(f"{run.name}/{video.alias}: vote_count differs from the window16/stride8/tail layout")
    recomputed = confusion_counts(video.teacher.point_label, video.teacher.valid_mask, pred.pred_label)
    row = run.rows[(split, video.identity)]
    if any(int(float(row[key])) != recomputed[key] for key in CONFUSION_KEYS):
        raise InputContractError(f"{run.name}/{video.alias}: recomputed confusion counts differ from h5_metrics.csv")
    return pred


# ----------------------------------------------------------------------------
# register-coverage
# ----------------------------------------------------------------------------


def _launcher_validation_sha256(launcher: Path) -> str:
    """The validation-list hash pinned in the S5-15 launcher, accepted only from a git-tracked, clean file.

    The git checks run in the launcher's own repository, so a synthetic test can supply its own.
    """
    text = launcher.read_text(encoding="utf-8")
    match = re.search(r'^EXPECTED_VAL_LIST_SHA256="([0-9a-f]{64})"', text, flags=re.MULTILINE)
    if not match:
        raise InputContractError("the S5-15 launcher does not carry EXPECTED_VAL_LIST_SHA256")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", launcher.name], cwd=launcher.parent,
                             capture_output=True, check=False).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", launcher.name], cwd=launcher.parent,
                           capture_output=True, check=False).returncode == 0
    if not (tracked and clean):
        raise InputContractError("the S5-15 launcher is not git-tracked and clean; its hash cannot serve as evidence")
    return match.group(1)


def cmd_register_coverage(args: argparse.Namespace) -> int:
    ctx = load_context(args, coverage_path=None)
    directory = Path(args.evaluation_checkpoint_dir)
    if directory.name != args.checkpoint_name:
        raise InputContractError("the checkpoint directory name does not match --checkpoint_name")

    # (ii) git-tracked evidence: the validation list is the one the S5-15 runs were pinned to.
    if list_content_sha256(ctx.validation) != _launcher_validation_sha256(Path(args.s5_15_launcher)):
        raise SplitContractError("the validation list differs from the S5-15 launcher's pinned validation list")

    # (iii) names only: the saved predictions are exactly sanity3 + validation18.
    predictions = directory / "predictions"
    if sorted(p.name for p in predictions.iterdir()) != ["train_sanity", "validation"]:
        raise SplitContractError("the predictions directory holds other entries than train_sanity/ and validation/")
    expected = {"train_sanity": set(ctx.sanity_ids), "validation": ctx.allow[PURPOSE_PRED]}
    for split, ids in expected.items():
        entries = sorted(p.name for p in (predictions / split).iterdir())
        if any(not name.endswith(".npz") for name in entries):
            raise SplitContractError(f"predictions/{split} holds non-NPZ entries")
        found = [single_identity(name, what=f"predictions/{split} #{i}") for i, name in enumerate(entries)]
        if len(set(found)) != len(found) or set(found) != ids:
            raise SplitContractError(f"predictions/{split} names are not exactly the expected videos")

    identities = sorted(expected["train_sanity"] | expected["validation"])
    evidence = {
        "checkpoint_name": args.checkpoint_name,
        "directory_named_by_user": True,
        "validation_list_matches_s5_15_launcher": True,
        "prediction_names_match_sanity3_validation18": True,
        "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "contents_parsed_before_registration": False,
    }
    out = Path(args.coverage_out)
    entries: list[dict[str, Any]] = []
    if out.is_file():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing.get("schema") != COVERAGE_SCHEMA:
            raise InputContractError("the existing coverage file has a different schema")
        entries = list(existing.get("entries", []))
    shared_hashes = {}
    for artifact in (directory / "summary.json", directory / "h5_metrics.csv"):
        if not artifact.is_file():
            raise InputContractError(f"{artifact.name} is missing")
        digest = file_sha256(artifact)
        entries = [e for e in entries if not (e["artifact_name"] == artifact.name and e["sha256"] == digest)]
        entries.append({"artifact_name": artifact.name, "sha256": digest, "video_identities": identities,
                        "provenance": evidence})
        shared_hashes[f"{args.checkpoint_name}/{artifact.name}"] = digest[:16]
    write_private_json(out.parent, out.name, {"schema": COVERAGE_SCHEMA, "entries": entries})
    shared = {"schema": "stage5_s5_17_coverage_registration_v1", "checkpoint_name": args.checkpoint_name,
              "n_videos_covered": len(identities), "sha256_16": shared_hashes,
              "evidence": {k: v for k, v in evidence.items() if k != "git"}, "git_dirty": evidence["git"]["dirty"]}
    if args.shared_json:
        write_shared_json(Path(args.shared_json), shared)
    print(f"coverage registered: {args.checkpoint_name}, 2 artifacts, {len(identities)} videos")
    return 0


# ----------------------------------------------------------------------------
# audit (S17-4, metadata only)
# ----------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    ctx = load_context(args, coverage_path=args.artifact_coverage)
    guard(ctx, ctx.train_core, PURPOSE_GT)
    guard(ctx, ctx.validation, PURPOSE_PRED)
    per_video: dict[str, Any] = {}
    tally: dict[str, Counter] = {k: Counter() for k in ("resolution", "mode", "input", "datasets")}
    groups = (("train_core", [ctx.identity_of[str(p)] for p in ctx.train_core], PURPOSE_GT),
              ("validation", [ctx.identity_of[str(p)] for p in ctx.validation], PURPOSE_PRED))
    for group, ids, purpose in groups:
        for identity in ids:
            video = load_video(ctx, identity, purpose, args, read_arrays=False)
            meta = h5_dataset_meta(ctx.path_of[identity])
            missing = [k for k, v in meta.items() if k != "attrs_present" and v is None
                       and not k.startswith("frame_annotation")]
            tally["resolution"][f"{group}:{video.resolution_method}"] += 1
            tally["mode"][f"{group}:{video.preprocess_mode}"] += 1
            tally["input"][f"{group}:{video.input_failure or 'evaluable'}"] += 1
            tally["datasets"][f"{group}:{'complete' if not missing else 'missing_required'}"] += 1
            per_video[video.alias] = {"resolution": video.resolution_method, "mode": video.preprocess_mode,
                                      "input_failure": video.input_failure, "datasets": meta}

    runs: dict[str, Any] = {}
    npz_tally = Counter()
    for name in ("best", "last"):
        directory = getattr(args, f"evaluation_dir_{name}")
        if not directory:
            runs[name] = {"status": "not_given"}
            continue
        run = load_evaluation_run(ctx, Path(directory), name, args)
        runs[name] = {"status": "coverage_ok", "hash_state": run.hash_state}
        for split, ids, purpose in (("validation", sorted(ctx.allow[PURPOSE_PRED]), PURPOSE_PRED),
                                    ("train_sanity", ctx.sanity_ids, PURPOSE_IN_SAMPLE)):
            for identity in ids:
                attrs = read_teacher_attrs(ctx.path_of[identity])
                npz = run.directory / "predictions" / split / f"{safe_name(attrs['video_name'])}.npz"
                guard(ctx, [npz], purpose)
                if not npz.is_file():
                    npz_tally[f"{name}:{split}:missing"] += 1
                    continue
                shapes = npz_member_shapes(npz)
                complete = all(k in shapes for k in ("point_indices", "prob_femur", "pred_label", "vote_count"))
                npz_tally[f"{name}:{split}:{'keys_complete' if complete else 'keys_missing'}"] += 1
    shared = {
        "schema": "stage5_s5_17_input_audit_v1",
        "counts": {k: dict(sorted(v.items())) for k, v in tally.items()},
        "npz": dict(sorted(npz_tally.items())),
        "evaluation_runs": runs,
        "metadata_only": True,
    }
    write_private_json(Path(args.private_out_dir), "input_audit_DO_NOT_SHARE.json",
                       {"per_video": per_video, "shared": shared, "git": git_state()})
    write_shared_json(Path(args.shared_json), shared)
    print("audit complete: counts written to the shared JSON")
    return 0


# ----------------------------------------------------------------------------
# run (S17-5)
# ----------------------------------------------------------------------------


def _estimates(frames, source: str, rule: str = "M1") -> dict[str, Any]:
    return {rep: fl.estimate_representation(frames, rep, instance_rule=rule, coord_space=RAW, source=source)
            for rep in fl.CANDIDATE_REPRESENTATIONS}


def _gt_frames(video: VideoInput, mask: np.ndarray, local: Points2D | None = None, raw: Points2D | None = None):
    assert video.teacher is not None
    return video_frame_geometries(video.teacher.frame_order, video.local if local is None else local,
                                  video.raw if raw is None else raw, mask,
                                  frame_universe=video.universe, local_bounds_wh=video.local_wh)


def analyze_gt_video(video: VideoInput) -> dict[str, Any]:
    teacher = video.teacher
    assert teacher is not None and video.raw is not None and video.transform is not None
    gt = teacher.gt_positive
    frames = _gt_frames(video, gt)
    base = _estimates(frames, "gt")
    record: dict[str, Any] = {
        "estimates_m1": {rep: e.as_record() for rep, e in base.items()},
        "estimates_m2": {rep: fl.estimate_quantile(frames, rep, instance_rule="M2", coord_space=RAW).as_record()
                         for rep in fl.FRAME_REPRESENTATIONS},
        "pseudo3d_reference": fl.estimate_pseudo3d_axis(teacher.points[gt]).as_record() if gt.any() else None,
        "holdout": mt.video_holdout(frames, video.raw, local_points=video.local, local_bounds_wh=video.local_wh),
    }

    inflation = {fl.REP_AABB: [], fl.REP_OBB: []}
    multi, multi_large = 0, 0
    lengths_b: list[float] = []
    lengths_undefined = 0
    # P3: frames with >= 1 valid GT-positive point (not the >= 5-point instance condition).
    counts = [int(m.size) for m in group_by_frame(teacher.frame_order, gt).values()]
    for frame, geometry in frames.items():
        if not geometry.present:
            continue
        m1 = geometry.instances[0]
        members = geometry.instance_point_indices[0]
        for rep in inflation:
            inflation[rep].append(mt.relative_inflation(m1, Points2D(video.raw.xy[members], RAW), rep))
        if m1.axis_usable:
            lengths_b.append(float(m1.length))
        else:
            lengths_undefined += 1
        if len(geometry.instances) > 1:
            multi += 1
            change = mt.relative_change(fl.frame_length(m1, fl.REP_OBB), fl.frame_length(geometry.merged, fl.REP_OBB))
            multi_large += int(change is not None and change > H17_5_REL_DIFF)
    record["inflation_median"] = {rep: mt.summarize(v)["median"] for rep, v in inflation.items()}
    record["h17_5_gt"] = {"multi_instance_frames": multi, "m1_m2_length_diff_over_10pct": multi_large}
    record["bbox_consistency"] = _bbox_consistency(video, frames)

    changes: dict[str, dict[str, list[float | None]]] = {rep: {} for rep in fl.CANDIDATE_REPRESENTATIONS}
    fp_stats = Counter()
    for kind, strength, seed in mt.PERTURBATION_CONDITIONS:
        if kind == "jitter":
            local2 = mt.perturb_local_jitter(video.local, gt, seed, frame_order=teacher.frame_order, sigma=strength)
            raw2 = video.transform.local_to_raw(local2, check_range=False)
            frames2 = _gt_frames(video, gt, local2, raw2)
        else:
            mask, stats = mt.perturb_gt_mask(kind, strength, seed, frame_order=teacher.frame_order, gt_mask=gt,
                                             background_mask=teacher.valid_background)
            if kind == "fp":
                fp_stats.update(stats)
            frames2 = _gt_frames(video, mask)
        perturbed = _estimates(frames2, "gt")
        cell = mt.cell_name(kind, strength)
        for rep in fl.CANDIDATE_REPRESENTATIONS:
            changes[rep].setdefault(cell, []).append(mt.relative_change(base[rep].value, perturbed[rep].value))
    record["perturbation_changes"] = changes
    record["fp_injection"] = dict(fp_stats)
    record["prior_summary"] = pr.summarize_video_for_prior(video.identity, frames, video.raw, raw_wh=video.raw_wh)
    record["constants_inputs"] = {"frame_lengths_b": lengths_b, "m1_frames_length_undefined": lengths_undefined,
                                  "gt_points_per_positive_frame": counts}
    return record


def _bbox_consistency(video: VideoInput, frames) -> dict[str, Any]:
    """H17-1b (descriptive): stored BBox vs point-GT M2 AABB, per frame, in raw px."""
    teacher = video.teacher
    if teacher is None or teacher.bbox_frame_order is None or video.transform is None:
        return {"n_frames": 0}
    ious, long_diff = [], []
    for frame in np.unique(teacher.bbox_frame_order):
        boxes = teacher.bbox_local_xyxy[teacher.bbox_frame_order == frame]
        boxes = boxes[np.all(np.isfinite(boxes), axis=1)]
        geometry = frames.get(int(frame))
        if boxes.size == 0 or geometry is None or not geometry.present:
            continue
        corners = np.concatenate([boxes[:, :2], boxes[:, 2:]])
        raw = video.transform.local_to_raw(Points2D(corners, LOCAL), check_range=False).xy
        bx0, by0 = raw.min(axis=0)
        bx1, by1 = raw.max(axis=0)
        gx0, gy0, gx1, gy1 = geometry.merged.aabb
        inter = max(0.0, min(bx1, gx1) - max(bx0, gx0)) * max(0.0, min(by1, gy1) - max(by0, gy0))
        union = (bx1 - bx0) * (by1 - by0) + (gx1 - gx0) * (gy1 - gy0) - inter
        ious.append(mt.safe_ratio(inter, union))
        long_diff.append(mt.signed_relative_error(max(bx1 - bx0, by1 - by0), geometry.merged.aabb_long_side))
    return {"n_frames": len(ious), "iou": mt.summarize(ious), "long_side_rel_diff": mt.summarize(long_diff),
            "note": "region enclosing all BBoxes vs region enclosing all GT instances; not per-instance agreement"}


def aggregate_h17_1(records: dict[str, dict[str, Any]], inputs: dict[str, VideoInput]) -> dict[str, Any]:
    evaluable = {a: r for a, r in records.items()}
    breakdown_input = Counter(v.input_failure for v in inputs.values() if v.input_failure)
    per_rep: dict[str, Any] = {}
    finals: dict[str, float] = {}
    statuses = {}
    for rep in fl.CANDIDATE_REPRESENTATIONS:
        denominators = mt.DenominatorBreakdown()
        for reason, count in breakdown_input.items():
            for _ in range(count):
                denominators.add_input_unevaluable(reason)
        for record in evaluable.values():
            denominators.add_result(record["estimates_m1"][rep]["failure"])
        cells = {}
        for kind, strength in mt.PERTURBATION_CELLS:
            name = mt.cell_name(kind, strength)
            cells[name] = mt.stability_cell_summary({a: r["perturbation_changes"][rep][name] for a, r in evaluable.items()})
        rule = mt.candidate_rule(cells, baseline_failures=sum(denominators.method_failure.values()))
        statuses[rep] = rule["status"]
        if rule["status"] == mt.RULE_PASS:
            finals[rep] = rule["max_p90"]
        per_rep[rep] = {
            "denominators": denominators.as_record(),
            "stability_cells": cells,
            "candidate_rule": rule,
            "holdout_coverage_descriptive": mt.summarize([r["holdout"]["value"][rep] for r in evaluable.values()]),
            "relative_inflation_descriptive": (mt.summarize([r["inflation_median"][rep] for r in evaluable.values()])
                                               if rep in (fl.REP_AABB, fl.REP_OBB) else None),
        }
    not_selectable = [rep for rep, s in statuses.items() if s == mt.RULE_NOT_SELECTABLE]
    excluded = {rep: {"status": s, "failing_cells": per_rep[rep]["candidate_rule"].get("failing_cells", [])}
                for rep, s in statuses.items() if s != mt.RULE_PASS}
    if finals:
        selection = {"status": "selected", **mt.select_primary(finals)}
    elif len(not_selectable) == len(statuses):
        selection = {"status": "not_selectable"}
    else:
        selection = {"status": "no_candidate"}
    selection["not_selectable_representations"] = not_selectable
    selection["excluded_representations"] = excluded
    selection["failure_policy"] = "F-A: failures and incomplete seeds count as worst; nearest-rank quantiles"
    return {"per_representation": per_rep, "selection": selection}


def _input_suspension(inputs: dict[str, VideoInput], minimum: int, total: int) -> dict[str, Any]:
    evaluable = sum(1 for v in inputs.values() if not v.input_failure)
    modes = Counter(v.preprocess_mode for v in inputs.values())
    usable_modes = Counter(v.preprocess_mode for v in inputs.values() if not v.input_failure)
    wiped = [m for m, n in modes.items() if n >= MODE_SHARE_LIMIT * total and usable_modes[m] == 0]
    return {"evaluable": evaluable, "minimum": minimum, "modes_entirely_excluded": len(wiped),
            "suspended": evaluable < minimum or bool(wiped)}


def analyze_prediction_video(video: VideoInput, pred, gt_frames, gt_estimates, reps: Sequence[str],
                             constants: pp.PostprocessConstants) -> dict[str, Any]:
    teacher = video.teacher
    assert teacher is not None and video.raw is not None
    originally_empty = not bool(pred.pred_label.any())
    out: dict[str, Any] = {"band_points": pred.band_count, "originally_empty": originally_empty, "postprocess": {}}
    for name in pp.POSTPROCESSES:
        mask = pp.apply_postprocess(name, frame_order=teacher.frame_order, local_points=video.local,
                                    raw_points=video.raw, pred_mask=pred.pred_label, probabilities=pred.prob_femur,
                                    constants=constants)
        frames = video_frame_geometries(teacher.frame_order, video.local, video.raw, mask, frame_universe=video.universe,
                                        local_bounds_wh=video.local_wh, probabilities=pred.prob_femur)
        detection = mt.instance_detection(gt_frames, frames)
        entry: dict[str, Any] = {
            "presence": mt.frame_presence_confusion(gt_frames, frames),
            "detection": {k: v for k, v in detection.items() if k != "matched_pairs"},
            "matched_pairs": detection["matched_pairs"],
            "fp_classes": mt.fp_spatial_classes(pred_mask=mask, point_label=teacher.point_label,
                                                valid_mask=teacher.valid_mask, frame_order=teacher.frame_order,
                                                raw_points=video.raw, gt_frames=gt_frames),
            "representations": {},
        }
        if name == "P3":
            entry["p3_target_frames"] = len(set(teacher.frame_order[pred.pred_label].tolist()))
            entry["p3_geometric_present_frames"] = sum(1 for g in frames.values() if g.present)
        for rep in reps:
            estimate = fl.estimate_representation(frames, rep, instance_rule="M1", coord_space=RAW, source="prediction")
            failure = estimate.failure
            if failure is not None:
                failure = (FailureReason.NO_PREDICTION if originally_empty
                           else FailureReason.REMOVED_BY_POSTPROCESS if not mask.any() else failure)
            reference = gt_estimates[rep].value
            if failure is None and reference is None:
                failure = FailureReason.GT_REFERENCE_UNDEFINED
            error = mt.signed_relative_error(reference, estimate.value) if failure is None else None
            item = {"estimate": estimate.as_record(), "failure": failure, "relative_error": error,
                    "within_tau_rel_diagnostic": None if error is None else abs(error) <= TAU_REL_DIAGNOSTIC}
            if rep == fl.REP_BEST_FRAME:
                oracle = fl.estimate_best_frame_oracle(gt_frames, frames, instance_rule="M1", coord_space=RAW)
                item["oracle"] = {"estimate": oracle.as_record(),
                                  "relative_error": mt.signed_relative_error(reference, oracle.value)}
            entry["representations"][rep] = item
        out["postprocess"][name] = entry
    return out


def aggregate_predictions(records: dict[str, dict[str, Any]], inputs: dict[str, VideoInput], reps: Sequence[str]
                          ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    unevaluable = Counter(v.input_failure for v in inputs.values() if v.input_failure)
    for name in pp.POSTPROCESSES:
        entry: dict[str, Any] = {"representations": {}}
        presence, detection, fp = Counter(), Counter(), Counter()
        for record in records.values():
            item = record["postprocess"][name]
            presence.update({k: item["presence"][k] for k in ("tp", "fp", "fn", "tn")})
            detection.update({k: item["detection"][k] for k in ("tp", "fp", "fn")})
            fp.update({k: v for k, v in item["fp_classes"].items() if k.startswith("n_")})
        entry["presence_pooled"] = {**presence, "precision": mt.safe_ratio(presence["tp"], presence["tp"] + presence["fp"]),
                                    "recall": mt.safe_ratio(presence["tp"], presence["tp"] + presence["fn"])}
        entry["detection_pooled"] = {**detection,
                                     "precision": mt.safe_ratio(detection["tp"], detection["tp"] + detection["fp"]),
                                     "recall": mt.safe_ratio(detection["tp"], detection["tp"] + detection["fn"])}
        entry["fp_classes_pooled"] = {
            **fp,
            "ignore_share_of_pred_all": mt.safe_ratio(fp["n_pred_ignore"], fp["n_pred_all"]),
            **{f"{c}_share_of_fp": mt.safe_ratio(fp[f"n_{c}"], fp["n_fp"])
               for c in ("gt_near", "gt_frame_far", "gt_absent_frame", "gt_scale_undefined")},
            **{f"{c}_share_of_pred_valid": mt.safe_ratio(fp[f"n_{c}"], fp["n_pred_valid"])
               for c in ("gt_near", "gt_frame_far", "gt_absent_frame", "gt_scale_undefined")},
        }
        if name == "P3":
            entry["p3_frames"] = {
                "target_frames": sum(r["postprocess"]["P3"]["p3_target_frames"] for r in records.values()),
                "geometric_present_frames": sum(r["postprocess"]["P3"]["p3_geometric_present_frames"]
                                                for r in records.values()),
                "note": P3_NOTE,
            }
        for rep in reps:
            denominators = mt.DenominatorBreakdown()
            for reason, count in unevaluable.items():
                for _ in range(count):
                    denominators.add_input_unevaluable(reason)
            errors, oracle_errors, within = [], [], 0
            for record in records.values():
                item = record["postprocess"][name]["representations"][rep]
                denominators.add_result(item["failure"])
                errors.append(item["relative_error"])
                within += int(bool(item["within_tau_rel_diagnostic"]))
                if "oracle" in item:
                    oracle_errors.append(item["oracle"]["relative_error"])
            entry["representations"][rep] = {
                "denominators": denominators.as_record(),
                "within_tau_rel_diagnostic": {"count": within, "of_evaluable": denominators.evaluable,
                                              "tau_rel": TAU_REL_DIAGNOSTIC, "note": "diagnostic, not T_FL"},
                "relative_error_of_successes": mt.summarize(errors),
                "abs_relative_error_of_successes": mt.summarize([None if e is None else abs(e) for e in errors]),
                "oracle_relative_error": mt.summarize(oracle_errors) if oracle_errors else None,
            }
        out[name] = entry
    return out


def _best_postprocess(aggregate: dict[str, Any], rep: str) -> str:
    counts = {name: aggregate[name]["representations"][rep]["within_tau_rel_diagnostic"]["count"]
              for name in pp.POSTPROCESSES}
    top = max(counts.values())
    return next(name for name in pp.POSTPROCESS_PRECEDENCE if counts[name] == top)


def cmd_run(args: argparse.Namespace) -> int:
    if args.coordinate_mode != "pixel":
        raise GeometryContractError(
            "mm mode stops: the mm/pixel source, key, granularity and frame mapping are not fixed (management 5.4)"
        )
    out_dir = private_dir(Path(args.private_out_dir))
    ctx = load_context(args, coverage_path=args.artifact_coverage)
    guard(ctx, ctx.train_core, PURPOSE_GT)
    guard(ctx, ctx.validation, PURPOSE_PRED)
    runs = {name: load_evaluation_run(ctx, Path(getattr(args, f"evaluation_dir_{name}")), name, args)
            for name in ("best", "last")}
    h5_hashes = json.loads(Path(args.h5_hash_record).read_text(encoding="utf-8")) if args.h5_hash_record else {}

    # --- H17-1 on train_core ------------------------------------------------
    core_inputs: dict[str, VideoInput] = {}
    core_records: dict[str, dict[str, Any]] = {}
    hash_states = Counter()
    for path in ctx.train_core:
        identity = ctx.identity_of[str(path)]
        hash_states[_hash_state(path, h5_hashes.get(identity))] += 1
        video = load_video(ctx, identity, PURPOSE_GT, args)
        core_inputs[video.alias] = video
        if video.input_failure is None:
            core_records[video.alias] = analyze_gt_video(video)
    h17_1 = aggregate_h17_1(core_records, core_inputs)
    h17_1["input"] = _input_suspension(core_inputs, args.min_train_core_evaluable, len(ctx.train_core))
    if h17_1["input"]["suspended"]:
        # The suspension is not bypassed: the rule result is kept for the record only.
        h17_1["selection"] = {"status": "suspended_input_insufficient",
                              "rule_result_not_used_while_suspended": h17_1["selection"]}

    constants_inputs = [r["constants_inputs"] for r in core_records.values()]
    constants = pp.PostprocessConstants.from_train_core(
        [x for c in constants_inputs for x in c["frame_lengths_b"]],
        [x for c in constants_inputs for x in c["gt_points_per_positive_frame"]],
    )
    summaries = [r["prior_summary"] for r in core_records.values()]
    prior = pr.build_prior(summaries)

    selection = h17_1["selection"]
    if selection.get("status") == "selected":
        reps = [selection["primary"], *selection["secondary"]]
        reps_basis = "h17_1_selection"
    else:
        reps = list(FALLBACK_H17_2_REPRESENTATIONS)
        reps_basis = "fixed_fallback_while_h17_1_undecided"
    reps_role = ("h17_1_candidates" if reps_basis == "h17_1_selection"
                 else "diagnostic_target_not_adoption_candidate" if reps else None)
    if len(reps) > 3 or not set(reps) <= set(fl.CANDIDATE_REPRESENTATIONS):
        raise GeometryContractError("H17-2 takes at most 3 representations from (a)-(d)")
    # The H17-1 judgement is never bypassed: a suspended H17-1 still reports as suspended, and the
    # fallback is computed only as a diagnostic.

    # --- H17-2..H17-5 on validation, sanity3 in-sample ------------------------
    tables: dict[str, Any] = {}
    private_rows: dict[str, Any] = {}
    for group, ids, split, purpose in (
        ("validation", [ctx.identity_of[str(p)] for p in ctx.validation], "validation", PURPOSE_PRED),
        ("sanity_in_sample", ctx.sanity_ids, "train_sanity", PURPOSE_IN_SAMPLE),
    ):
        inputs: dict[str, VideoInput] = {}
        gt_info: dict[str, Any] = {}
        for identity in ids:
            if group == "validation":
                hash_states[_hash_state(ctx.path_of[identity], h5_hashes.get(identity))] += 1
            video = load_video(ctx, identity, purpose, args)
            inputs[video.alias] = video
            if video.input_failure is None:
                frames = _gt_frames(video, video.teacher.gt_positive)
                # In-sample: exclude the video itself by identity. A sanity video that was
                # input-unevaluable in train_core is not in the prior, so nothing to exclude.
                in_prior = any(s.identity == identity for s in summaries)
                prior_source = (pr.build_prior(summaries, exclude_identity=identity)
                                if group != "validation" and in_prior else prior)
                prior_est = pr.prior_estimates(prior_source, frame_universe=video.universe, raw_wh=video.raw_wh)
                gt_info[video.alias] = (frames, _estimates(frames, "gt"), prior_est)
        minimum = args.min_validation_evaluable if group == "validation" else len(ids)
        group_out: dict[str, Any] = {"input": _input_suspension(inputs, minimum, len(ids)), "checkpoints": {}}
        group_out["prior_only"] = {}
        for rep in fl.CANDIDATE_REPRESENTATIONS:
            errors = [mt.signed_relative_error(g[1][rep].value, g[2][rep].value) for g in gt_info.values()]
            group_out["prior_only"][rep] = {
                "within_tau_rel_diagnostic": sum(1 for e in errors if e is not None and abs(e) <= TAU_REL_DIAGNOSTIC),
                "relative_error": mt.summarize(errors),
            }
        for name, run in runs.items():
            records = {}
            for alias, video in inputs.items():
                if video.input_failure is not None:
                    continue
                pred = load_prediction(ctx, run, video, split, purpose)
                frames, gt_est, _prior_est = gt_info[alias]
                if reps:
                    records[alias] = analyze_prediction_video(video, pred, frames, gt_est, reps, constants)
            aggregate = aggregate_predictions(records, inputs, reps) if reps else {}
            group_out["checkpoints"][name] = {
                "role": "main" if name == "best" else "auxiliary",
                "hash_state": run.hash_state,
                "postprocess": aggregate,
                "best_postprocess_by_representation": ({rep: _best_postprocess(aggregate, rep) for rep in reps}
                                                        if name == "best" and reps else None),
            }
            private_rows[f"{group}/{name}"] = records
        tables[group] = group_out

    shared = {
        "schema": SHARED_SCHEMA,
        "coordinate_mode": "pixel",
        "coordinate_space": RAW.value,
        "mm_available": False,
        "contract": {"manifest": args.split_manifest,
                     "train_core_identity16": ctx.contract.manifest.lists["train_core"]["identity_sha256"][:16],
                     "validation_identity16": ctx.contract.manifest.lists["validation"]["identity_sha256"][:16]},
        "provenance_hash_states": {"teacher_h5": dict(hash_states),
                                   **{f"{n}_run": r.hash_state for n, r in runs.items()}},
        "input_minimums": {
            "train_core": args.min_train_core_evaluable,
            "validation": args.min_validation_evaluable,
            "synthetic_override": (args.min_train_core_evaluable, args.min_validation_evaluable)
                                  != (INPUT_MIN_TRAIN_CORE, INPUT_MIN_VALIDATION),
        },
        "constants_from_train_core": {
            "p2_distance_raw_px": constants.p2_distance_raw_px,
            "p2_definition": "0.5 x median M1 (b) length over train_core frames where it is defined",
            "p2_source_frames": sum(len(c["frame_lengths_b"]) for c in constants_inputs),
            "p2_frames_length_undefined": sum(c["m1_frames_length_undefined"] for c in constants_inputs),
            "p3_top_k": constants.p3_top_k,
            "p3_definition": "median valid GT-positive points per frame with >= 1 valid GT-positive point",
            "p3_source_frames": sum(len(c["gt_points_per_positive_frame"]) for c in constants_inputs),
        },
        "h17_1": h17_1,
        "h17_2_representations": {"representations": reps, "basis": reps_basis, "role": reps_role,
                                  "note": ("fixed before real data; not respecified after seeing results"
                                           if reps_role == "diagnostic_target_not_adoption_candidate" else None)},
        "tables": tables,
        "notes": {"p3": P3_NOTE, "metrics": METRIC_NOTE,
                  "best_postprocess": "selected on validation (development choice; optimistic)"},
        "limits": list(LIMITS),
        "git": {"dirty": git_state()["dirty"]},
    }
    write_private_json(out_dir, "geometry_run_DO_NOT_SHARE.json", {
        "alias_map": {a: {"identity": ctx.identity_of[str(ctx.path_of[i])], "path": str(ctx.path_of[i])}
                      for i, a in ctx.alias_of.items()},
        "train_core_records": {a: _strip_frames(r) for a, r in core_records.items()},
        "prediction_records": private_rows,
        "inputs": {a: {"resolution": v.resolution_method, "mode": v.preprocess_mode, "input_failure": v.input_failure}
                   for group in (core_inputs,) for a, v in group.items()},
        "git": git_state(),
    })
    write_shared_json(Path(args.shared_json), shared)
    print("run complete: aggregates written to the shared JSON, per-video records to the private directory")
    return 0


def _strip_frames(record: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in record.items() if k != "prior_summary"}
    summary = record.get("prior_summary")
    if summary is not None:
        out["prior_summary"] = {k: getattr(summary, k) for k in ("center_norm", "axis_angle_norm", "length_norm",
                                                                  "width_norm", "presence_rate_by_bin")}
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split_manifest", default=DEFAULT_MANIFEST, help="approved pin name (not a path)")
    parser.add_argument("--split_contract_pins", default=None)
    parser.add_argument("--train_core_list", required=True)
    parser.add_argument("--validation_list", required=True)
    parser.add_argument("--train_sanity_list", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5-17 geometry diagnosis (training-free, CPU only).")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register-coverage", help="register summary.json / h5_metrics.csv of one checkpoint dir")
    _common(reg)
    reg.add_argument("--evaluation_checkpoint_dir", required=True)
    reg.add_argument("--checkpoint_name", required=True, choices=sorted(CHECKPOINT_EPOCHS))
    reg.add_argument("--coverage_out", required=True, help="private coverage JSON (appended if it exists)")
    reg.add_argument("--shared_json", default=None)
    reg.add_argument("--s5_15_launcher", default=str(S5_15_LAUNCHER),
                     help="git-tracked launcher pinning the validation list hash (default: the repository's)")
    reg.set_defaults(func=cmd_register_coverage)

    for name, func, help_text in (("audit", cmd_audit, "S17-4 metadata-only fail-fast"),
                                  ("run", cmd_run, "S17-5 diagnosis")):
        cmd = sub.add_parser(name, help=help_text)
        _common(cmd)
        cmd.add_argument("--intermediate_root", required=True)
        cmd.add_argument("--intermediate_suffix", default="_pseudo3d.h5")
        cmd.add_argument("--artifact_coverage", required=(name == "run"), default=None)
        cmd.add_argument("--evaluation_dir_best", required=(name == "run"), default=None)
        cmd.add_argument("--evaluation_dir_last", required=(name == "run"), default=None)
        cmd.add_argument("--checkpoint_sha256_best", default=None)
        cmd.add_argument("--checkpoint_sha256_last", default=None)
        cmd.add_argument("--evaluation_revision_best", default=None)
        cmd.add_argument("--evaluation_revision_last", default=None)
        cmd.add_argument("--private_out_dir", required=True)
        cmd.add_argument("--shared_json", required=True)
        if name == "run":
            cmd.add_argument("--coordinate_mode", choices=("pixel", "mm"), required=True)
            cmd.add_argument("--h5_hash_record", default=None, help="private JSON {identity: sha256}, if one exists")
            # Fixed by management document 5.1 (130 / 16). Overridden only by synthetic tests;
            # the launcher does not expose them and the shared JSON records any override.
            cmd.add_argument("--min_train_core_evaluable", type=int, default=INPUT_MIN_TRAIN_CORE,
                             help=argparse.SUPPRESS)
            cmd.add_argument("--min_validation_evaluable", type=int, default=INPUT_MIN_VALIDATION,
                             help=argparse.SUPPRESS)
        cmd.set_defaults(func=func)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except (SplitContractError, InputContractError, GeometryContractError, IdentityError, FixedListError,
            PrivacyError) as error:
        print(f"STOP ({error.__class__.__name__}): {scrub(str(error))}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 -- the traceback may carry paths; keep it private
        destination = getattr(args, "private_out_dir", None)
        if destination:
            try:
                target = private_dir(Path(destination)) / "error_traceback_DO_NOT_SHARE.txt"
                target.write_text(traceback.format_exc(), encoding="utf-8")
                os.chmod(target, 0o600)
                where = "written to the private directory"
            except OSError:
                where = "suppressed"
        else:
            where = "suppressed"
        print(f"STOP (unexpected {error.__class__.__name__}); traceback {where}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
