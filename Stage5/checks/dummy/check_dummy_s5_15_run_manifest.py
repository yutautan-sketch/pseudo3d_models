from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))

from write_stage5_s5_15_run_manifest import (  # noqa: E402
    anonymize,
    collect_environment,
    collect_file_list,
    collect_git_state,
    privacy_self_check,
)

# ----------------------------------------------------------------------------
# S5-15 P3: the launch manifest.
#
# The properties that matter are that a dirty or unreadable working tree is
# never recorded as clean, that the manifest lands OUTSIDE the run directory
# (whose non-empty guard must keep working), and that the shareable copy drops
# host paths and video identifiers.
# ----------------------------------------------------------------------------

WRITER = REPO_ROOT / "checks" / "real_h5" / "write_stage5_s5_15_run_manifest.py"


def init_repo(root: Path, *, dirty: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)
    if dirty:
        (root / "tracked.txt").write_text("modified\n", encoding="utf-8")
    return root


def video_name(index: int) -> str:
    return f"2026071{index % 10}_12{index:04d}_{index}_pointcloud_annotated.h5"


def write_lists(root: Path) -> tuple[Path, Path]:
    data = root / "collected"
    data.mkdir(parents=True, exist_ok=True)
    train_entries, val_entries = [], []
    for index in range(4):
        path = data / video_name(index)
        path.write_bytes(b"x")
        train_entries.append(path)
    for index in range(100, 101):
        path = data / video_name(index)
        path.write_bytes(b"x")
        val_entries.append(path)
    train_list = root / "train_files.txt"
    val_list = root / "val_files.txt"
    train_list.write_text("\n".join(str(p) for p in train_entries) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(str(p) for p in val_entries) + "\n", encoding="utf-8")
    return train_list, val_list


def run_writer(*, arm: str, mode: str, manifest_dir: Path, output_dir: Path, repo_dir: Path, **extra: Any):
    args = [
        sys.executable, str(WRITER),
        "--arm", arm,
        "--mode", mode,
        "--manifest_dir", str(manifest_dir),
        "--output_dir", str(output_dir),
        "--augmentation", extra.pop("augmentation", "none"),
        "--repo_dir", str(repo_dir),
    ]
    for key, value in extra.items():
        args += [f"--{key}", str(value)]
    return subprocess.run(args, capture_output=True, text=True)


def latest_manifests(manifest_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    private = sorted(p for p in manifest_dir.glob("manifest_*.json") if "shareable" not in p.name)[-1]
    shareable = sorted(manifest_dir.glob("manifest_*_shareable.json"))[-1]
    return (
        json.loads(private.read_text(encoding="utf-8")),
        json.loads(shareable.read_text(encoding="utf-8")),
    )


def test_clean_repository_is_recorded_as_clean_with_its_head() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = init_repo(Path(tmp) / "repo")
        state = collect_git_state(repo)
        assert state["head"] and len(state["head"]) == 40
        assert state["worktree_clean"] is True
        assert state["num_dirty_entries"] == 0
        assert "note" not in state
    print("  ok: a clean tree records its 40-character HEAD and worktree_clean=True")


def test_dirty_repository_is_never_recorded_as_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = init_repo(Path(tmp) / "repo", dirty=True)
        state = collect_git_state(repo)
        assert state["worktree_clean"] is False
        assert state["num_dirty_entries"] == 1
        assert "NOT from a clean tree" in state["note"]
        assert "tracked.txt" in state["status_short"]
    print("  ok: an uncommitted change is recorded as NOT clean, with the status text kept verbatim")


def test_missing_repository_is_unknown_rather_than_clean() -> None:
    """A checkout that is not a git repo must not silently look clean."""
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "not_a_repo"
        plain.mkdir()
        state = collect_git_state(plain)
        assert state["head"] is None
        assert state["worktree_clean"] is None, "unknown state must not collapse to clean"
        assert "not recorded as clean" in state["note"]
    print("  ok: a non-repository directory yields worktree_clean=None, never True")


def test_environment_capture_records_versions_and_gpu_fields() -> None:
    info = collect_environment()
    assert info["python_executable"] == sys.executable
    assert isinstance(info["python_version"], str)
    for key in ("numpy_version", "torch_version", "gpu_driver", "gpu_driver_error"):
        assert key in info, key
    print(
        "  ok: environment records python/numpy/torch and GPU fields "
        f"(torch={info['torch_version']}, cuda_available={info.get('cuda_available')})"
    )


def test_file_list_capture_reports_counts_and_both_fingerprints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train_list, val_list = write_lists(Path(tmp))
        train = collect_file_list(train_list, label="train")
        val = collect_file_list(val_list, label="val")
        assert train["count"] == 4 and val["count"] == 1
        assert len(train["content_sha256"]) == 64 and len(train["identity_sha256"]) == 64
        assert train["content_sha256"] != train["identity_sha256"]
        assert collect_file_list(None, label="train") == {"path": None}
    print("  ok: list capture records counts plus content and identity fingerprints")


def test_manifest_is_written_outside_the_run_directory() -> None:
    """The launcher refuses a non-empty run directory, so the manifest must not
    be placed there -- the protection is not weakened to make room for it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = init_repo(root / "repo")
        run_dir = root / "runs" / "s5_15_r0"
        manifest_dir = root / "manifests"
        result = run_writer(
            arm="r0", mode="dry_run", manifest_dir=manifest_dir, output_dir=run_dir, repo_dir=repo
        )
        assert result.returncode == 0, result.stderr
        assert not run_dir.exists(), "the run directory must not be created or touched"
        written = sorted(manifest_dir.glob("manifest_*.json"))
        assert len(written) == 2, [p.name for p in written]
        private, _ = latest_manifests(manifest_dir)
        assert private["output_dir"] == str(run_dir)
    print("  ok: the manifest lands in its own directory and the run directory is left untouched")


def test_dry_run_and_training_manifests_coexist_without_overwriting() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = init_repo(root / "repo")
        manifest_dir = root / "manifests"
        for mode in ("dry_run", "training"):
            result = run_writer(
                arm="r1",
                mode=mode,
                manifest_dir=manifest_dir,
                output_dir=root / "runs" / "r1",
                repo_dir=repo,
                augmentation="random_z_rotation",
            )
            assert result.returncode == 0, result.stderr
        names = sorted(p.name for p in manifest_dir.glob("manifest_*.json") if "shareable" not in p.name)
        assert len(names) == 2, names
        assert any("dry_run" in name for name in names) and any("training" in name for name in names)
    print("  ok: the dry-run and training manifests are kept as separate files")


def test_manifest_records_settings_inputs_and_planned_volume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = init_repo(root / "repo")
        train_list, val_list = write_lists(root)
        checkpoint = root / "init_groupnorm.pt"
        checkpoint.write_bytes(b"weights" * 100)
        manifest_dir = root / "manifests"

        result = run_writer(
            arm="r1",
            mode="training",
            manifest_dir=manifest_dir,
            output_dir=root / "runs" / "r1",
            repo_dir=repo,
            augmentation="random_z_rotation",
            augmentation_rotation_degrees=15.0,
            seed=42,
            resolved_augmentation_seed=500042,
            epochs=5,
            save_every=1,
            class_weight="0.05963856,1.94036150",
            pointnext_norm="groupnorm",
            pointnext_norm_groups=8,
            init_checkpoint=checkpoint,
            train_list=train_list,
            val_list=val_list,
            planned_optimizer_steps=450,
            command="TRAIN_LIST=... bash train_stage5.sh",
        )
        assert result.returncode == 0, result.stderr
        private, _ = latest_manifests(manifest_dir)

        settings = private["settings"]
        assert settings["augmentation"] == "random_z_rotation"
        assert settings["resolved_augmentation_seed"] == 500042
        assert settings["epochs"] == 5 and settings["save_every"] == 1
        assert settings["pointnext_norm"] == "groupnorm" and settings["pointnext_norm_groups"] == 8
        assert private["inputs"]["init_checkpoint"]["sha256"]
        assert private["inputs"]["train_list"]["count"] == 4
        assert private["planned"]["optimizer_steps"] == 450
        assert "not assumed to match" in private["planned"]["note"]
        assert private["command"].startswith("TRAIN_LIST=")
        assert private["recorded_at"]
    print("  ok: settings, resolved seed, input hashes, planned volume and the command are all recorded")


def test_shareable_copy_drops_host_paths_and_video_identifiers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = init_repo(root / "repo", dirty=True)
        train_list, val_list = write_lists(root)
        checkpoint = root / "init.pt"
        checkpoint.write_bytes(b"w")
        manifest_dir = root / "manifests"

        result = run_writer(
            arm="r0",
            mode="training",
            manifest_dir=manifest_dir,
            output_dir=Path("/mnt/data/3d_projects/stage5_runs/260917/run_r0"),
            repo_dir=repo,
            init_checkpoint=checkpoint,
            train_list=train_list,
            val_list=val_list,
            command="INIT_CHECKPOINT=/mnt/data/x bash train_stage5.sh",
        )
        assert result.returncode == 0, result.stderr
        private, shareable = latest_manifests(manifest_dir)

        assert private["output_dir"].startswith("/mnt/data")
        assert private["inputs"]["train_list"]["path"]
        assert private["git"]["status_short"]

        assert "output_dir" not in shareable
        assert "command" not in shareable
        assert "path" not in shareable["inputs"]["train_list"]
        assert "path" not in shareable["inputs"]["init_checkpoint"]
        assert "status_short" not in shareable["git"]
        assert shareable["run_alias"] == "s5_15_r0"
        # The dirtiness itself is still reported, only the file names are gone.
        assert shareable["git"]["worktree_clean"] is False
        assert shareable["inputs"]["train_list"]["content_sha256"]
        assert shareable["privacy_self_check"]["timestamp_like_video_ids_absent"]
        assert shareable["privacy_self_check"]["absolute_host_paths_absent"]
    print("  ok: the shareable copy keeps hashes and the dirty flag while dropping paths and video names")


def test_privacy_self_check_flags_a_leak() -> None:
    assert privacy_self_check({"a": "clean"})["absolute_host_paths_absent"]
    assert not privacy_self_check({"a": "/mnt/data/x"})["absolute_host_paths_absent"]
    assert not privacy_self_check({"a": "20260711_120001_3"})["timestamp_like_video_ids_absent"]
    print("  ok: the privacy self-check still flags host paths and timestamp-style video IDs")


def test_anonymize_does_not_mutate_the_private_manifest() -> None:
    manifest = {
        "arm": "r0",
        "git": {"repo_dir": "/mnt/data/repo", "status_short": " M a.py", "worktree_clean": False},
        "inputs": {"train_list": {"path": "/mnt/data/train.txt", "count": 162}, "init_checkpoint": {"path": "/x"}},
        "environment": {"python_executable": "/home/user/bin/python"},
        "output_dir": "/mnt/data/run",
        "command": "bash train.sh",
    }
    snapshot = json.loads(json.dumps(manifest))
    anonymize(manifest)
    assert manifest == snapshot, "anonymize() must not modify the private manifest in place"
    print("  ok: building the shareable copy leaves the private manifest untouched")


def main() -> None:
    tests = [
        test_clean_repository_is_recorded_as_clean_with_its_head,
        test_dirty_repository_is_never_recorded_as_clean,
        test_missing_repository_is_unknown_rather_than_clean,
        test_environment_capture_records_versions_and_gpu_fields,
        test_file_list_capture_reports_counts_and_both_fingerprints,
        test_manifest_is_written_outside_the_run_directory,
        test_dry_run_and_training_manifests_coexist_without_overwriting,
        test_manifest_records_settings_inputs_and_planned_volume,
        test_shareable_copy_drops_host_paths_and_video_identifiers,
        test_privacy_self_check_flags_a_leak,
        test_anonymize_does_not_mutate_the_private_manifest,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 S5-15 run manifest synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
