from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# S5-15 P1: single-run manifest audit of the W-A reference training run.
#
# This checker reads one train_stage5.py output_dir (config.json,
# train_files.txt, val_files.txt, history.json, checkpoint inventory) plus the
# initialization checkpoint named *inside that config.json* (never guessed
# from a basename), and reports whether the run matches the S5-15 P1 fixed
# conditions. It answers the four items that could not be settled from the
# anonymized artifacts available in the development container:
#
#   1. the initialization checkpoint's real path and SHA-256,
#   2. whether the referenced run directory actually exists,
#   3. whether the saved train/val file lists are the ones S5-15 will reuse,
#   4. the non-anonymized config values.
#
# Unlike check_stage5_class_weight_ablation.py this audit does NOT stop at the
# first mismatch: every check is recorded so a single real-machine run gives
# the complete picture. Statuses are PASS / FAIL / JUDGE (needs a human call,
# not automatically a defect) / UNKNOWN (not verifiable on this host).
#
# Plain JSON/text only -- no torch, h5py or numpy import is required, and no
# file inside the audited run directory is modified.
# ----------------------------------------------------------------------------

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_JUDGE = "JUDGE"
STATUS_UNKNOWN = "UNKNOWN"

# S5-15 P1 fixed conditions (handoff section 4 / revision record chapter 6 P1
# table). Every key here is compared against config.json exactly.
EXPECTED_CONFIG: dict[str, Any] = {
    "model": "pointnext_s",
    "num_classes": 2,
    "pointnext_norm": "groupnorm",
    "pointnext_norm_groups": 8,
    "features": "intensity,confidence",
    "label_policy": "bbox_noncontour_ignore",
    "loss": "cross_entropy",
    "label_smoothing": 0.0,
    "ignore_index": -1,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "grad_clip_norm": 10.0,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "epochs": 5,
    "seed": 42,
    "window_mode": "overlap",
    "window_size_frames": 16,
    "window_stride_frames": 8,
    "include_tail_window": True,
    "no_normalize_points": False,
    "num_points": 0,
    "sampling_mode": "video",
    "exclude_ignore_in_sampling": False,
    "positive_oversample_ratio": 0.0,
    "width": 32,
    "expansion": 4,
    "dropout": 0.0,
    "pointnext_radius": 0.1,
    "pointnext_nsample": 32,
    "pointnext_sa_layers": 2,
    "pointnext_sa_use_res": True,
    "strict_checkpoint": True,
    "num_train_files": 162,
    "num_val_files": 18,
}

# Recorded for the manifest but intentionally not enforced: train_stage5.sh
# documents that DEPTH and the global-context switch are ignored by the
# PointNeXt-S wrapper, so a difference there would not change training.
RECORDED_ONLY_CONFIG_KEYS = ("depth", "no_global_context", "val_every", "best_metric", "device")

# S5-13 recorded these for the W-A run (EX260914). They are identity checks:
# a mismatch means the audited directory is not the same run S5-13 audited.
EXPECTED_SAMPLE_COUNTS = {"num_train_samples": 715, "num_val_samples": 87}

EXPECTED_CLASS_WEIGHT = (0.05963856, 1.94036150)
CLASS_WEIGHT_TOLERANCE = 1e-6

EXPECTED_TEACHER_SUFFIX = "_bboxrank_v7_cvat_authoritative_crop_quality_v1.h5"

# Established privacy self-check patterns (S5-14): timestamp-style video IDs
# and absolute host paths must never appear in the shareable output.
TIMESTAMP_VIDEO_ID_PATTERN = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9]+")
ABSOLUTE_HOST_PATH_PATTERN = re.compile(r"(/mnt/data|/home/[A-Za-z0-9_.-]+)")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(
    results: list[dict[str, Any]],
    check: str,
    status: str,
    detail: str,
    **extra: Any,
) -> None:
    row: dict[str, Any] = {"check": check, "status": status, "detail": detail}
    row.update(extra)
    results.append(row)


def values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return expected == actual


def audit_config(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    for key in sorted(EXPECTED_CONFIG):
        expected = EXPECTED_CONFIG[key]
        if key not in config:
            record(results, f"config.{key}", STATUS_FAIL, f"key missing from config.json (expected {expected!r})")
            continue
        actual = config[key]
        status = STATUS_PASS if values_match(expected, actual) else STATUS_FAIL
        record(results, f"config.{key}", status, f"expected={expected!r} actual={actual!r}")

    for key, expected in EXPECTED_SAMPLE_COUNTS.items():
        actual = config.get(key)
        status = STATUS_PASS if values_match(expected, actual) else STATUS_FAIL
        record(
            results,
            f"identity.{key}",
            status,
            f"S5-13 recorded {expected} for W-A; this run reports {actual!r}",
        )

    return {key: config.get(key) for key in RECORDED_ONLY_CONFIG_KEYS}


def audit_class_weight(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    info = config.get("class_weight_info")
    if not isinstance(info, dict):
        record(results, "class_weight.info", STATUS_FAIL, "config.json has no class_weight_info object")
        return {}

    mode = info.get("mode")
    requested = info.get("requested")
    resolved = info.get("resolved")

    if isinstance(resolved, list) and len(resolved) == len(EXPECTED_CLASS_WEIGHT):
        matches = all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=CLASS_WEIGHT_TOLERANCE)
            for a, b in zip(resolved, EXPECTED_CLASS_WEIGHT)
        )
        record(
            results,
            "class_weight.resolved",
            STATUS_PASS if matches else STATUS_FAIL,
            f"expected={list(EXPECTED_CLASS_WEIGHT)} actual={resolved} (tolerance {CLASS_WEIGHT_TOLERANCE})",
        )
    else:
        record(results, "class_weight.resolved", STATUS_FAIL, f"class_weight_info.resolved is {resolved!r}")

    # Policy-chat answer 1: do not judge auto-vs-manual by the literal string.
    # The effective weight above is authoritative; the mode is reported so a
    # human can decide whether the run-directory name is merely stale.
    if mode == "manual":
        record(results, "class_weight.mode", STATUS_PASS, f"mode={mode!r} requested={requested!r}")
    else:
        record(
            results,
            "class_weight.mode",
            STATUS_JUDGE,
            f"mode={mode!r} requested={requested!r} -- effective weight is the authoritative check; "
            "confirm whether an auto-derived weight is acceptable as the S5-15 control",
        )

    for key in ("auto_class_weight_epsilon", "normalize_auto_class_weight"):
        record(results, f"class_weight.{key}", STATUS_PASS, f"recorded value: {config.get(key)!r}", enforced=False)

    record(results, "class_weight.top_level", STATUS_PASS, f"config.class_weight={config.get('class_weight')!r}", enforced=False)
    return {"mode": mode, "requested": requested, "resolved": resolved}


def audit_file_list(
    run_dir: Path,
    name: str,
    *,
    expected_count: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    path = run_dir / name
    if not path.is_file():
        record(results, f"file_list.{name}", STATUS_FAIL, f"missing: {name} not found in the run directory")
        return {"present": False}

    lines = read_lines(path)
    record(
        results,
        f"file_list.{name}.count",
        STATUS_PASS if len(lines) == expected_count else STATUS_FAIL,
        f"expected {expected_count} lines, found {len(lines)}",
    )

    fingerprint = sha256_text("\n".join(lines))
    basenames = [Path(line).name for line in lines]
    duplicates = sorted({name_ for name_ in basenames if basenames.count(name_) > 1})
    record(
        results,
        f"file_list.{name}.basename_unique",
        STATUS_PASS if not duplicates else STATUS_FAIL,
        (
            "all basenames unique (validates the S5-15 stable_video_id design, report section 4.3.1)"
            if not duplicates
            else f"{len(duplicates)} duplicated basename(s) -- stable_video_id would collide"
        ),
    )

    wrong_teacher = [name_ for name_ in basenames if not name_.endswith(EXPECTED_TEACHER_SUFFIX)]
    record(
        results,
        f"file_list.{name}.teacher_v7",
        STATUS_PASS if not wrong_teacher else STATUS_FAIL,
        (
            f"all {len(basenames)} entries end with {EXPECTED_TEACHER_SUFFIX}"
            if not wrong_teacher
            else f"{len(wrong_teacher)} entry/entries do not use the teacher v7 suffix"
        ),
    )

    missing_on_host = [line for line in lines if not Path(line).is_file()]
    record(
        results,
        f"file_list.{name}.files_present",
        STATUS_PASS if not missing_on_host else STATUS_UNKNOWN,
        (
            f"all {len(lines)} H5 files exist on this host"
            if not missing_on_host
            else f"{len(missing_on_host)} of {len(lines)} H5 files not found on this host"
        ),
    )

    return {
        "present": True,
        "count": len(lines),
        "content_sha256": fingerprint,
        "basenames_unique": not duplicates,
        "private_lines": lines,
    }


def audit_init_checkpoint(
    config: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    skip_hash: bool,
) -> dict[str, Any]:
    raw_path = config.get("checkpoint")
    if not raw_path:
        record(
            results,
            "init_checkpoint.path",
            STATUS_FAIL,
            "config.json has no 'checkpoint' value -- the run did not start from an initialization checkpoint",
        )
        return {"path": None}

    path = Path(str(raw_path))
    # Policy-chat answer 1: take the real path from the config, never infer it
    # from a basename. The basename is only reported as a readability hint.
    record(
        results,
        "init_checkpoint.path",
        STATUS_PASS,
        f"taken from config.json (basename: {path.name})",
    )

    groupnorm_named = "groupnorm" in path.name.lower()
    record(
        results,
        "init_checkpoint.groupnorm_basename_hint",
        STATUS_PASS if groupnorm_named else STATUS_JUDGE,
        (
            "basename mentions groupnorm, consistent with config.pointnext_norm"
            if groupnorm_named
            else "basename does not mention groupnorm -- confirm against config.pointnext_norm rather than the name"
        ),
        enforced=False,
    )

    report: dict[str, Any] = {"path": str(path), "basename": path.name, "exists": path.is_file()}
    if not path.is_file():
        record(results, "init_checkpoint.exists", STATUS_UNKNOWN, "file not found on this host")
        return report

    record(results, "init_checkpoint.exists", STATUS_PASS, f"{path.stat().st_size} bytes")
    if skip_hash:
        record(results, "init_checkpoint.sha256", STATUS_UNKNOWN, "hashing skipped via --skip_checkpoint_hash")
        return report

    digest = sha256_file(path)
    report["sha256"] = digest
    report["size_bytes"] = path.stat().st_size
    record(
        results,
        "init_checkpoint.sha256",
        STATUS_PASS,
        f"{digest} -- R0/R1 must start from this exact file",
    )
    return report


def audit_checkpoint_inventory(run_dir: Path, config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted(p.name for p in run_dir.glob("*.pt"))
    per_epoch = [name for name in present if name.startswith("checkpoint_epoch_")]
    save_every = config.get("save_every")

    for name in ("best.pt", "last.pt"):
        record(
            results,
            f"checkpoint.{name}",
            STATUS_PASS if name in present else STATUS_FAIL,
            "present" if name in present else "missing",
        )

    # S5-15 P1 requires per-epoch checkpoints. A W-A run made with the
    # train_stage5.sh default SAVE_EVERY=10 cannot have them for a 5-epoch run,
    # which is a real difference to weigh when deciding whether W-A can serve
    # as R0 rather than something to silently ignore.
    record(
        results,
        "checkpoint.per_epoch",
        STATUS_PASS if per_epoch else STATUS_JUDGE,
        (
            f"{len(per_epoch)} per-epoch checkpoint(s): {per_epoch}"
            if per_epoch
            else f"no checkpoint_epoch_*.pt found (config.save_every={save_every!r}); "
            "S5-15 P1 asks for per-epoch checkpoints, so reusing this run as R0 means accepting that gap"
        ),
    )
    return {"checkpoint_files": present, "per_epoch_checkpoints": per_epoch, "save_every": save_every}


def audit_history(run_dir: Path, results: list[dict[str, Any]], *, expected_epochs: int) -> dict[str, Any]:
    path = run_dir / "history.json"
    if not path.is_file():
        record(results, "history.present", STATUS_FAIL, "missing: history.json not found in the run directory")
        return {"present": False}

    history = load_json(path)
    if not isinstance(history, list):
        record(results, "history.present", STATUS_FAIL, "history.json is not a list")
        return {"present": False}

    record(
        results,
        "history.epochs",
        STATUS_PASS if len(history) == expected_epochs else STATUS_FAIL,
        f"expected {expected_epochs} epoch record(s), found {len(history)}",
    )

    nonfinite: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}", item)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                nonfinite.append(prefix)

    for entry in history:
        if not isinstance(entry, dict):
            continue
        walk(f"epoch{entry.get('epoch')}.train", entry.get("train"))
        if entry.get("val") is not None:
            walk(f"epoch{entry.get('epoch')}.val", entry.get("val"))

    record(
        results,
        "history.finite_metrics",
        STATUS_PASS if not nonfinite else STATUS_FAIL,
        "all recorded metrics finite" if not nonfinite else f"non-finite: {nonfinite[:10]}",
    )
    last_epoch = history[-1].get("epoch") if history and isinstance(history[-1], dict) else None
    return {"present": True, "epochs_recorded": len(history), "last_epoch": last_epoch}


def collect_environment() -> dict[str, Any]:
    """Environment facts the S5-15 manifest needs, including the actual
    multiprocessing start method (report section 4.3 must not assume fork)."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "multiprocessing_default_start_method": multiprocessing.get_context().get_start_method(),
        "multiprocessing_current_start_method": multiprocessing.get_start_method(allow_none=True),
        "multiprocessing_available_start_methods": multiprocessing.get_all_start_methods(),
    }
    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = None
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        info["torch_version"] = None
    return info


def privacy_self_check(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    timestamp_hits = TIMESTAMP_VIDEO_ID_PATTERN.findall(text)
    path_hits = ABSOLUTE_HOST_PATH_PATTERN.findall(text)
    return {
        "timestamp_like_video_ids_absent": not timestamp_hits,
        "absolute_host_paths_absent": not path_hits,
        "timestamp_like_video_id_hits": len(timestamp_hits),
        "absolute_host_path_hits": len(path_hits),
    }


def build_shareable_summary(private_summary: dict[str, Any]) -> dict[str, Any]:
    """Strip real paths and filenames, keeping counts, hashes and statuses."""
    shareable = {
        "status": private_summary["status"],
        "run_dir_alias": "wa_reference_run",
        "results": private_summary["results"],
        "recorded_only_config": private_summary["recorded_only_config"],
        "class_weight": private_summary["class_weight"],
        "checkpoint_inventory": private_summary["checkpoint_inventory"],
        "history": private_summary["history"],
        "environment": private_summary["environment"],
        "counts": private_summary["counts"],
    }
    init_checkpoint = dict(private_summary["init_checkpoint"])
    init_checkpoint.pop("path", None)
    shareable["init_checkpoint"] = init_checkpoint
    shareable["file_lists"] = {
        name: {key: value for key, value in info.items() if key != "private_lines"}
        for name, info in private_summary["file_lists"].items()
    }
    return shareable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-15 P1: audit one train_stage5.py run directory (the W-A reference run) "
        "against the S5-15 fixed conditions, reporting every check rather than stopping at the first mismatch."
    )
    parser.add_argument("--run_dir", required=True, help="W-A train_stage5.py output_dir to audit (read-only)")
    parser.add_argument("--expected_epochs", type=int, default=5)
    parser.add_argument("--expected_train_files", type=int, default=162)
    parser.add_argument("--expected_val_files", type=int, default=18)
    parser.add_argument("--skip_checkpoint_hash", action="store_true")
    parser.add_argument(
        "--private_json",
        default=None,
        help="Full audit output including real paths/filenames. Keep on the real machine, do not share.",
    )
    parser.add_argument(
        "--shareable_json",
        default=None,
        help="Anonymized audit output (counts, hashes, statuses only) safe to share with the policy chat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    results: list[dict[str, Any]] = []

    if not run_dir.is_dir():
        print(f"FAIL: run directory not found on this host: {run_dir}", file=sys.stderr)
        sys.exit(2)
    record(results, "run_dir.exists", STATUS_PASS, "run directory found on this host")

    config_path = run_dir / "config.json"
    if not config_path.is_file():
        print(f"FAIL: config.json not found: {config_path}", file=sys.stderr)
        sys.exit(2)
    config = load_json(config_path)
    record(results, "config.present", STATUS_PASS, "config.json loaded")

    recorded_only = audit_config(config, results)
    class_weight = audit_class_weight(config, results)
    train_list = audit_file_list(run_dir, "train_files.txt", expected_count=args.expected_train_files, results=results)
    val_list = audit_file_list(run_dir, "val_files.txt", expected_count=args.expected_val_files, results=results)
    init_checkpoint = audit_init_checkpoint(config, results, skip_hash=args.skip_checkpoint_hash)
    checkpoint_inventory = audit_checkpoint_inventory(run_dir, config, results)
    history = audit_history(run_dir, results, expected_epochs=args.expected_epochs)
    environment = collect_environment()

    counts = {
        status: sum(1 for row in results if row["status"] == status)
        for status in (STATUS_PASS, STATUS_FAIL, STATUS_JUDGE, STATUS_UNKNOWN)
    }
    status = "passed" if counts[STATUS_FAIL] == 0 else "failed"

    private_summary = {
        "status": status,
        "run_dir": str(run_dir),
        "results": results,
        "counts": counts,
        "recorded_only_config": recorded_only,
        "class_weight": class_weight,
        "file_lists": {"train_files.txt": train_list, "val_files.txt": val_list},
        "init_checkpoint": init_checkpoint,
        "checkpoint_inventory": checkpoint_inventory,
        "history": history,
        "environment": environment,
    }
    shareable_summary = build_shareable_summary(private_summary)
    shareable_summary["privacy_self_check"] = privacy_self_check(shareable_summary)

    if args.private_json:
        private_path = Path(args.private_json)
        private_path.parent.mkdir(parents=True, exist_ok=True)
        with private_path.open("w", encoding="utf-8") as f:
            json.dump(private_summary, f, indent=2, ensure_ascii=False)
    if args.shareable_json:
        shareable_path = Path(args.shareable_json)
        shareable_path.parent.mkdir(parents=True, exist_ok=True)
        with shareable_path.open("w", encoding="utf-8") as f:
            json.dump(shareable_summary, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-15 P1 manifest audit")
    print(f"  status : {status}")
    print(f"  counts : {counts}")
    for row in results:
        if row["status"] != STATUS_PASS:
            print(f"  [{row['status']}] {row['check']}: {row['detail']}")
    if init_checkpoint.get("sha256"):
        print(f"  init checkpoint sha256: {init_checkpoint['sha256']}")
    for name, info in private_summary["file_lists"].items():
        if info.get("present"):
            print(f"  {name}: {info['count']} lines, content sha256 {info['content_sha256']}")
    print(f"  multiprocessing default start method: {environment['multiprocessing_default_start_method']}")
    privacy = shareable_summary["privacy_self_check"]
    print(
        "  shareable privacy self-check: "
        f"timestamp_like_video_ids_absent={privacy['timestamp_like_video_ids_absent']} "
        f"absolute_host_paths_absent={privacy['absolute_host_paths_absent']}"
    )
    if args.private_json:
        print(f"  private output   : {args.private_json}")
    if args.shareable_json:
        print(f"  shareable output : {args.shareable_json}")

    sys.exit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
