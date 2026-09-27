from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.file_list_mode import list_content_sha256, list_identity_sha256, read_file_list  # noqa: E402
from stage5.utils.split_contract import resolve_active_contract  # noqa: E402

# ----------------------------------------------------------------------------
# S5-15 P3: the per-run manifest the policy chat requires at launch.
#
# A commit alone does not record which revision a run used, so this captures
# the real git HEAD and working-tree state at launch time together with the
# execution environment, the resolved inputs and the planned volume.
#
# It deliberately writes OUTSIDE the run directory: the launcher refuses to
# start into a non-empty run directory, and that protection must not be
# weakened just to drop a manifest in first. The manifest records the run
# directory path so the two can be associated afterwards.
# ----------------------------------------------------------------------------

TIMESTAMP_VIDEO_ID_PATTERN = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9]+")
ABSOLUTE_HOST_PATH_PATTERN = re.compile(r"(/mnt/data|/home/[A-Za-z0-9_.-]+)")


def run_git(args: list[str], *, cwd: Path) -> tuple[str | None, str | None]:
    """Return (stdout, error). A missing repo is recorded, never guessed at."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None, "git executable not found"
    if result.returncode != 0:
        return None, (result.stderr or "").strip() or f"git {' '.join(args)} exited {result.returncode}"
    return result.stdout.rstrip("\n"), None


def collect_git_state(repo_dir: Path) -> dict[str, Any]:
    head, head_error = run_git(["rev-parse", "HEAD"], cwd=repo_dir)
    status, status_error = run_git(["status", "--short"], cwd=repo_dir)
    state: dict[str, Any] = {
        "repo_dir": str(repo_dir),
        "head": head,
        "head_error": head_error,
        "status_short": status,
        "status_error": status_error,
    }
    if status is None:
        # Never claim a clean tree when the state could not be read.
        state["worktree_clean"] = None
        state["num_dirty_entries"] = None
        state["note"] = "working tree state unavailable; not recorded as clean"
    else:
        entries = [line for line in status.splitlines() if line.strip()]
        state["worktree_clean"] = not entries
        state["num_dirty_entries"] = len(entries)
        if entries:
            state["note"] = "non-ignored changes present at launch; this run is NOT from a clean tree"
    return state


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
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
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        info["torch_version"] = None

    driver, driver_error = None, None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            driver = result.stdout.strip()
        else:
            driver_error = (result.stderr or "").strip() or f"nvidia-smi exited {result.returncode}"
    except FileNotFoundError:
        driver_error = "nvidia-smi not found"
    info["gpu_driver"] = driver
    info["gpu_driver_error"] = driver_error
    return info


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()



_CONTRACT_ARGS: dict[str, Any] = {}


def _contract_for(label: str):
    """The active contract, resolved once from the CLI arguments."""
    if "resolved" not in _CONTRACT_ARGS:
        _CONTRACT_ARGS["resolved"] = resolve_active_contract(
            manifest_key=_CONTRACT_ARGS.get("split_manifest"),
            pins_path=_CONTRACT_ARGS.get("split_contract_pins"),
            coverage_path=_CONTRACT_ARGS.get("artifact_coverage"),
        )
    return _CONTRACT_ARGS["resolved"]


def collect_file_list(path: str | Path | None, *, label: str) -> dict[str, Any]:
    if not path:
        return {"path": None}
    entries = read_file_list(path, label=label)

    # S5-16 Step 0: a manifest must not vouch for a list containing sealed
    # videos, so the contract is consulted before the entries are fingerprinted.
    _contract_for(label).assert_paths_allowed(entries, purpose=f"{label} run-manifest inputs")
    return {
        "path": str(path),
        "count": len(entries),
        "content_sha256": list_content_sha256(entries),
        "identity_sha256": list_identity_sha256(entries),
    }


def anonymize(manifest: dict[str, Any]) -> dict[str, Any]:
    """Shareable copy: keep counts, hashes and settings, drop host paths."""
    shareable = json.loads(json.dumps(manifest, ensure_ascii=False))
    shareable.pop("command", None)
    shareable.pop("dry_run_log", None)
    git_state = shareable.get("git", {})
    git_state.pop("repo_dir", None)
    git_state.pop("status_short", None)
    for key in ("train_list", "val_list"):
        entry = shareable.get("inputs", {}).get(key)
        if isinstance(entry, dict):
            entry.pop("path", None)
    inputs = shareable.get("inputs", {})
    if isinstance(inputs.get("init_checkpoint"), dict):
        inputs["init_checkpoint"].pop("path", None)
    shareable.pop("output_dir", None)
    shareable.get("environment", {}).pop("python_executable", None)
    shareable["run_alias"] = f"s5_15_{manifest.get('arm', 'unknown')}"
    return shareable


def privacy_self_check(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    return {
        "timestamp_like_video_ids_absent": not TIMESTAMP_VIDEO_ID_PATTERN.findall(text),
        "absolute_host_paths_absent": not ABSOLUTE_HOST_PATH_PATTERN.findall(text),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the S5-15 P3 launch manifest for one arm, outside the run directory "
        "so the launcher's non-empty-run-dir protection stays intact."
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--mode", choices=["dry_run", "training"], required=True)
    parser.add_argument("--manifest_dir", required=True, help="Private directory the manifest is written to")
    parser.add_argument("--output_dir", required=True, help="Planned run directory (recorded, not written to)")
    parser.add_argument("--augmentation", required=True)
    parser.add_argument("--augmentation_rotation_degrees", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolved_augmentation_seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--class_weight", default=None)
    parser.add_argument("--pointnext_norm", default=None)
    parser.add_argument("--pointnext_norm_groups", type=int, default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument(
        "--split_manifest",
        default=None,
        help="Name of an APPROVED pin in stage5/config/split_contract_pins.json (not a path). "
        "Falls back to $STAGE5_SPLIT_MANIFEST; absent or unapproved stops the run.",
    )
    parser.add_argument("--split_contract_pins", default=None)
    parser.add_argument("--artifact_coverage", default=None)
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument("--command", default=None, help="Resolved launch command")
    parser.add_argument("--dry_run_log", default=None, help="File holding the dry-run output to embed")
    parser.add_argument("--planned_optimizer_steps", type=int, default=None)
    parser.add_argument("--repo_dir", default=str(REPO_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _CONTRACT_ARGS.update(
        {
            "split_manifest": args.split_manifest,
            "split_contract_pins": args.split_contract_pins,
            "artifact_coverage": args.artifact_coverage,
        }
    )
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).astimezone()
    init_checkpoint: dict[str, Any] = {"path": args.init_checkpoint}
    if args.init_checkpoint and Path(args.init_checkpoint).is_file():
        init_path = Path(args.init_checkpoint)
        init_checkpoint["sha256"] = sha256_file(init_path)
        init_checkpoint["size_bytes"] = init_path.stat().st_size
        init_checkpoint["basename"] = init_path.name

    dry_run_log = None
    if args.dry_run_log and Path(args.dry_run_log).is_file():
        dry_run_log = Path(args.dry_run_log).read_text(encoding="utf-8")

    manifest: dict[str, Any] = {
        "arm": args.arm,
        "mode": args.mode,
        "recorded_at": timestamp.isoformat(),
        "git": collect_git_state(Path(args.repo_dir)),
        "environment": collect_environment(),
        "settings": {
            "augmentation": args.augmentation,
            "augmentation_rotation_degrees": args.augmentation_rotation_degrees,
            "seed": args.seed,
            "resolved_augmentation_seed": args.resolved_augmentation_seed,
            "epochs": args.epochs,
            "save_every": args.save_every,
            "class_weight": args.class_weight,
            "pointnext_norm": args.pointnext_norm,
            "pointnext_norm_groups": args.pointnext_norm_groups,
        },
        "inputs": {
            "init_checkpoint": init_checkpoint,
            "train_list": collect_file_list(args.train_list, label="train"),
            "val_list": collect_file_list(args.val_list, label="val"),
        },
        "output_dir": args.output_dir,
        "planned": {
            "optimizer_steps": args.planned_optimizer_steps,
            "note": "planned values; the executed counts are reported separately and are not "
            "assumed to match",
        },
        "command": args.command,
        "dry_run_log": dry_run_log,
    }

    shareable = anonymize(manifest)
    shareable["privacy_self_check"] = privacy_self_check(shareable)

    stamp = timestamp.strftime("%Y%m%dT%H%M%S")
    private_path = manifest_dir / f"manifest_{args.arm}_{args.mode}_{stamp}.json"
    shareable_path = manifest_dir / f"manifest_{args.arm}_{args.mode}_{stamp}_shareable.json"
    private_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    shareable_path.write_text(json.dumps(shareable, indent=2, ensure_ascii=False), encoding="utf-8")

    git_state = manifest["git"]
    print(f"  manifest ({args.mode}) written")
    print(f"    git HEAD      : {git_state['head']}")
    if git_state["worktree_clean"] is None:
        print(f"    worktree      : UNKNOWN ({git_state.get('status_error')}) -- not recorded as clean")
    elif git_state["worktree_clean"]:
        print("    worktree      : clean (no non-ignored changes)")
    else:
        print(f"    worktree      : {git_state['num_dirty_entries']} non-ignored change(s) -- NOT clean")
    print(f"    private       : {private_path}")
    print(f"    shareable     : {shareable_path}")


if __name__ == "__main__":
    main()
