from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))

from check_stage5_s5_15_p1_manifest_audit import (  # noqa: E402
    EXPECTED_CLASS_WEIGHT,
    EXPECTED_CONFIG,
    EXPECTED_SAMPLE_COUNTS,
    EXPECTED_TEACHER_SUFFIX,
    STATUS_FAIL,
    STATUS_JUDGE,
    STATUS_PASS,
    STATUS_UNKNOWN,
    audit_checkpoint_inventory,
    audit_class_weight,
    audit_config,
    audit_file_list,
    audit_history,
    audit_init_checkpoint,
    build_shareable_summary,
    collect_environment,
    privacy_self_check,
    sha256_file,
    values_match,
)

# ----------------------------------------------------------------------------
# S5-15 P1: synthetic pass/fail coverage for the W-A manifest audit. Builds
# fake train_stage5.py run directories on a temp path -- realistic-looking
# absolute paths and timestamp-style video IDs included, so the anonymization
# and privacy self-check are exercised against the exact patterns they exist
# to catch. Pure stdlib: no numpy/h5py/torch/CUDA needed.
# ----------------------------------------------------------------------------

FAKE_DATA_ROOT = "/mnt/data/3d_projects/pseudo3d/annotated"


def status_of(results: list[dict[str, Any]], check: str) -> str:
    matches = [row["status"] for row in results if row["check"] == check]
    assert len(matches) == 1, f"expected exactly one result for {check!r}, found {len(matches)}"
    return matches[0]


def fake_video_path(index: int, *, teacher_suffix: str = EXPECTED_TEACHER_SUFFIX) -> str:
    stem = f"2026071{index % 10}_12{index:04d}_{index}_pointcloud_annotated_foreground_combined_v2"
    return f"{FAKE_DATA_ROOT}/{stem}{teacher_suffix}"


def build_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = dict(EXPECTED_CONFIG)
    config.update(EXPECTED_SAMPLE_COUNTS)
    config.update(
        {
            "checkpoint": "/mnt/data/3d_projects/models/Stage5/work_dirs/"
            "_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init_groupnorm.pt",
            "save_every": 10,
            "depth": 6,
            "no_global_context": False,
            "val_every": 1,
            "best_metric": "iou_femur",
            "device": "cuda",
            "class_weight": list(EXPECTED_CLASS_WEIGHT),
            "class_weight_info": {
                "mode": "manual",
                "requested": "0.05963856,1.94036150",
                "counts": None,
                "epsilon": None,
                "normalize": None,
                "resolved": list(EXPECTED_CLASS_WEIGHT),
            },
            "auto_class_weight_epsilon": 0.02,
            "normalize_auto_class_weight": True,
        }
    )
    config.update(overrides)
    return config


def build_history(*, epochs: int = 5, nonfinite: bool = False) -> list[dict[str, Any]]:
    history = []
    for epoch in range(1, epochs + 1):
        val_loss = float("nan") if (nonfinite and epoch == epochs) else 0.13 + 0.001 * epoch
        history.append(
            {
                "epoch": epoch,
                "train": {"loss": 0.3 - 0.01 * epoch, "f1": 0.02 * epoch, "fp": 1000 * epoch},
                "val": {"loss": val_loss, "f1": 0.01 * epoch, "iou_femur": 0.005 * epoch},
            }
        )
    return history


def write_run_dir(
    root: Path,
    *,
    config: dict[str, Any] | None = None,
    train_files: list[str] | None = None,
    val_files: list[str] | None = None,
    history: list[dict[str, Any]] | None = None,
    checkpoint_names: tuple[str, ...] = ("best.pt", "last.pt"),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = build_config() if config is None else config
    train_files = [fake_video_path(i) for i in range(162)] if train_files is None else train_files
    val_files = [fake_video_path(1000 + i) for i in range(18)] if val_files is None else val_files
    history = build_history() if history is None else history

    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "train_files.txt").write_text("\n".join(train_files) + "\n", encoding="utf-8")
    (root / "val_files.txt").write_text("\n".join(val_files) + "\n", encoding="utf-8")
    (root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    for name in checkpoint_names:
        (root / name).write_bytes(b"fake checkpoint bytes")
    return root


def test_values_match_distinguishes_bool_from_int() -> None:
    assert values_match(True, True)
    assert values_match(False, False)
    # A config carrying 1 where True is required is a real difference, not a match.
    assert not values_match(True, 1)
    assert not values_match(False, 0)
    assert values_match(0.001, 0.001)
    assert not values_match(0.001, 0.01)
    assert values_match("groupnorm", "groupnorm")
    assert not values_match("groupnorm", "batchnorm")
    print("  ok: values_match keeps bool/int and float comparisons strict")


def test_matching_config_passes_every_expected_key() -> None:
    results: list[dict[str, Any]] = []
    recorded_only = audit_config(build_config(), results)
    failures = [row for row in results if row["status"] != STATUS_PASS]
    assert not failures, f"unexpected non-PASS rows: {failures}"
    assert len(results) == len(EXPECTED_CONFIG) + len(EXPECTED_SAMPLE_COUNTS)
    assert recorded_only["depth"] == 6
    assert recorded_only["best_metric"] == "iou_femur"
    print(f"  ok: matching config passes all {len(results)} checks; depth/global-context only recorded")


def test_config_mismatches_are_all_reported_not_short_circuited() -> None:
    results: list[dict[str, Any]] = []
    audit_config(build_config(pointnext_norm="batchnorm", seed=7, window_stride_frames=4), results)
    assert status_of(results, "config.pointnext_norm") == STATUS_FAIL
    assert status_of(results, "config.seed") == STATUS_FAIL
    assert status_of(results, "config.window_stride_frames") == STATUS_FAIL
    # Everything else still evaluated -- the audit must not stop at the first miss.
    assert status_of(results, "config.label_policy") == STATUS_PASS
    assert status_of(results, "config.gradient_accumulation_steps") == STATUS_PASS
    print("  ok: three independent config mismatches all reported, remaining keys still checked")


def test_missing_config_key_is_failure_not_crash() -> None:
    config = build_config()
    del config["pointnext_norm_groups"]
    results: list[dict[str, Any]] = []
    audit_config(config, results)
    assert status_of(results, "config.pointnext_norm_groups") == STATUS_FAIL
    print("  ok: a config key missing entirely is reported as FAIL")


def test_sample_count_identity_check_detects_a_different_run() -> None:
    results: list[dict[str, Any]] = []
    audit_config(build_config(num_train_samples=999), results)
    assert status_of(results, "identity.num_train_samples") == STATUS_FAIL
    assert status_of(results, "identity.num_val_samples") == STATUS_PASS
    print("  ok: window-sample identity check flags a run that is not the S5-13 W-A")


def test_class_weight_auto_mode_is_judge_when_effective_weight_matches() -> None:
    """Policy-chat answer 1: do not call auto-vs-manual a mismatch on the string
    alone -- the effective weight decides, the mode is escalated for judgement."""
    results: list[dict[str, Any]] = []
    info = audit_class_weight(
        build_config(
            class_weight_info={
                "mode": "auto",
                "requested": "auto",
                "counts": [61192000, 688292],
                "epsilon": 0.02,
                "normalize": True,
                "resolved": list(EXPECTED_CLASS_WEIGHT),
            }
        ),
        results,
    )
    assert status_of(results, "class_weight.resolved") == STATUS_PASS
    assert status_of(results, "class_weight.mode") == STATUS_JUDGE
    assert info["mode"] == "auto"
    assert not [row for row in results if row["status"] == STATUS_FAIL]
    print("  ok: auto mode with the expected effective weight is JUDGE, not FAIL")


def test_class_weight_wrong_effective_value_fails() -> None:
    results: list[dict[str, Any]] = []
    audit_class_weight(
        build_config(
            class_weight_info={"mode": "manual", "requested": "0.5,1.5", "resolved": [0.5, 1.5]}
        ),
        results,
    )
    assert status_of(results, "class_weight.resolved") == STATUS_FAIL
    assert status_of(results, "class_weight.mode") == STATUS_PASS
    print("  ok: a different effective class weight fails even when mode is manual")


def test_file_list_checks_count_uniqueness_and_teacher_version() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(Path(tmp) / "run")
        results: list[dict[str, Any]] = []
        info = audit_file_list(run_dir, "train_files.txt", expected_count=162, results=results)
        assert info["count"] == 162
        assert info["basenames_unique"] is True
        assert status_of(results, "file_list.train_files.txt.count") == STATUS_PASS
        assert status_of(results, "file_list.train_files.txt.basename_unique") == STATUS_PASS
        assert status_of(results, "file_list.train_files.txt.teacher_v7") == STATUS_PASS
        # The fixture points at paths that do not exist here: UNKNOWN, not FAIL.
        assert status_of(results, "file_list.train_files.txt.files_present") == STATUS_UNKNOWN
        print(f"  ok: clean 162-entry list passes; missing H5 files on this host are UNKNOWN")


def test_duplicate_basename_fails_because_stable_video_id_would_collide() -> None:
    duplicated = [fake_video_path(i) for i in range(161)]
    duplicated.append(f"/mnt/data/other_mount/{Path(duplicated[0]).name}")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(Path(tmp) / "run", train_files=duplicated)
        results: list[dict[str, Any]] = []
        info = audit_file_list(run_dir, "train_files.txt", expected_count=162, results=results)
        assert info["basenames_unique"] is False
        assert status_of(results, "file_list.train_files.txt.basename_unique") == STATUS_FAIL
        assert status_of(results, "file_list.train_files.txt.count") == STATUS_PASS
        print("  ok: same basename under two mount prefixes fails the stable_video_id precondition")


def test_wrong_teacher_suffix_and_count_are_reported() -> None:
    files = [fake_video_path(i) for i in range(160)]
    files.append(fake_video_path(900, teacher_suffix="_bboxrank_v6_legacy.h5"))
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(Path(tmp) / "run", train_files=files)
        results: list[dict[str, Any]] = []
        audit_file_list(run_dir, "train_files.txt", expected_count=162, results=results)
        assert status_of(results, "file_list.train_files.txt.count") == STATUS_FAIL
        assert status_of(results, "file_list.train_files.txt.teacher_v7") == STATUS_FAIL
        print("  ok: 161-entry list with one teacher v6 file fails both count and teacher checks")


def test_file_list_fingerprint_is_stable_and_order_sensitive() -> None:
    base = [fake_video_path(i) for i in range(18)]
    reordered = list(reversed(base))
    with tempfile.TemporaryDirectory() as tmp:
        dir_a = write_run_dir(Path(tmp) / "a", val_files=base)
        dir_b = write_run_dir(Path(tmp) / "b", val_files=list(base))
        dir_c = write_run_dir(Path(tmp) / "c", val_files=reordered)
        results: list[dict[str, Any]] = []
        info_a = audit_file_list(dir_a, "val_files.txt", expected_count=18, results=results)
        info_b = audit_file_list(dir_b, "val_files.txt", expected_count=18, results=results)
        info_c = audit_file_list(dir_c, "val_files.txt", expected_count=18, results=results)
        assert info_a["content_sha256"] == info_b["content_sha256"]
        assert info_a["content_sha256"] != info_c["content_sha256"]
        print("  ok: file-list fingerprint is reproducible and changes when the order changes")


def test_init_checkpoint_hash_comes_from_the_config_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        init_path = Path(tmp) / "stage5_pointnext_s_s3dis_partial_init_groupnorm.pt"
        payload = b"groupnorm transfer init weights"
        init_path.write_bytes(payload)
        results: list[dict[str, Any]] = []
        report = audit_init_checkpoint(build_config(checkpoint=str(init_path)), results, skip_hash=False)
        assert report["sha256"] == hashlib.sha256(payload).hexdigest()
        assert report["sha256"] == sha256_file(init_path)
        assert report["size_bytes"] == len(payload)
        assert status_of(results, "init_checkpoint.exists") == STATUS_PASS
        assert status_of(results, "init_checkpoint.sha256") == STATUS_PASS
        assert status_of(results, "init_checkpoint.groupnorm_basename_hint") == STATUS_PASS
        print("  ok: init checkpoint is hashed from the path recorded in config.json")


def test_missing_init_checkpoint_is_unknown_and_batchnorm_name_is_judge() -> None:
    results: list[dict[str, Any]] = []
    report = audit_init_checkpoint(
        build_config(checkpoint="/mnt/data/does/not/exist/stage5_pointnext_s_s3dis_partial_init.pt"),
        results,
        skip_hash=False,
    )
    assert report["exists"] is False
    assert "sha256" not in report
    assert status_of(results, "init_checkpoint.exists") == STATUS_UNKNOWN
    assert status_of(results, "init_checkpoint.groupnorm_basename_hint") == STATUS_JUDGE
    assert not [row for row in results if row["status"] == STATUS_FAIL]
    print("  ok: unreachable checkpoint is UNKNOWN (not FAIL); a non-groupnorm basename is JUDGE only")


def test_absent_init_checkpoint_entry_is_failure() -> None:
    results: list[dict[str, Any]] = []
    audit_init_checkpoint(build_config(checkpoint=None), results, skip_hash=False)
    assert status_of(results, "init_checkpoint.path") == STATUS_FAIL
    print("  ok: a run with no initialization checkpoint at all fails")


def test_missing_per_epoch_checkpoints_is_judge_with_save_every_recorded() -> None:
    """SAVE_EVERY=10 on a 5-epoch run leaves no checkpoint_epoch_*.pt, which is a
    real gap against the S5-15 P1 save requirement -- surfaced, not hidden."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(Path(tmp) / "run")
        results: list[dict[str, Any]] = []
        inventory = audit_checkpoint_inventory(run_dir, build_config(), results)
        assert inventory["per_epoch_checkpoints"] == []
        assert inventory["save_every"] == 10
        assert status_of(results, "checkpoint.per_epoch") == STATUS_JUDGE
        assert status_of(results, "checkpoint.best.pt") == STATUS_PASS
        assert status_of(results, "checkpoint.last.pt") == STATUS_PASS

        run_dir_2 = write_run_dir(
            Path(tmp) / "run2",
            checkpoint_names=("best.pt", "last.pt", "checkpoint_epoch_1.pt", "checkpoint_epoch_5.pt"),
        )
        results_2: list[dict[str, Any]] = []
        inventory_2 = audit_checkpoint_inventory(run_dir_2, build_config(save_every=1), results_2)
        assert inventory_2["per_epoch_checkpoints"] == ["checkpoint_epoch_1.pt", "checkpoint_epoch_5.pt"]
        assert status_of(results_2, "checkpoint.per_epoch") == STATUS_PASS
        print("  ok: no per-epoch checkpoints -> JUDGE with save_every shown; present -> PASS")


def test_missing_best_checkpoint_is_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(Path(tmp) / "run", checkpoint_names=("last.pt",))
        results: list[dict[str, Any]] = []
        audit_checkpoint_inventory(run_dir, build_config(), results)
        assert status_of(results, "checkpoint.best.pt") == STATUS_FAIL
        assert status_of(results, "checkpoint.last.pt") == STATUS_PASS
        print("  ok: a missing best.pt is reported as FAIL")


def test_history_epoch_count_and_nonfinite_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = write_run_dir(Path(tmp) / "good")
        results: list[dict[str, Any]] = []
        summary = audit_history(good, results, expected_epochs=5)
        assert summary["epochs_recorded"] == 5 and summary["last_epoch"] == 5
        assert status_of(results, "history.epochs") == STATUS_PASS
        assert status_of(results, "history.finite_metrics") == STATUS_PASS

        short = write_run_dir(Path(tmp) / "short", history=build_history(epochs=3))
        results_short: list[dict[str, Any]] = []
        audit_history(short, results_short, expected_epochs=5)
        assert status_of(results_short, "history.epochs") == STATUS_FAIL

        nan_run = write_run_dir(Path(tmp) / "nan", history=build_history(nonfinite=True))
        results_nan: list[dict[str, Any]] = []
        audit_history(nan_run, results_nan, expected_epochs=5)
        assert status_of(results_nan, "history.finite_metrics") == STATUS_FAIL
        print("  ok: history epoch count and NaN val loss are both detected")


def test_environment_reports_a_real_multiprocessing_start_method() -> None:
    """Report section 4.3 must record the actual start method instead of
    assuming fork, so the audit has to read it rather than hard-code it."""
    env = collect_environment()
    assert env["multiprocessing_default_start_method"] in env["multiprocessing_available_start_methods"]
    assert isinstance(env["python_version"], str) and env["python_version"]
    assert "torch_version" in env and "numpy_version" in env
    print(
        "  ok: environment records start method "
        f"{env['multiprocessing_default_start_method']!r} of {env['multiprocessing_available_start_methods']}"
    )


def test_shareable_summary_drops_paths_and_passes_privacy_self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(Path(tmp) / "run")
        results: list[dict[str, Any]] = []
        train_info = audit_file_list(run_dir, "train_files.txt", expected_count=162, results=results)
        val_info = audit_file_list(run_dir, "val_files.txt", expected_count=18, results=results)
        init_report = audit_init_checkpoint(build_config(), results, skip_hash=False)
        private_summary = {
            "status": "passed",
            "run_dir": str(run_dir),
            "results": results,
            "counts": {STATUS_PASS: len(results), STATUS_FAIL: 0, STATUS_JUDGE: 0, STATUS_UNKNOWN: 0},
            "recorded_only_config": {"depth": 6},
            "class_weight": {"mode": "manual", "resolved": list(EXPECTED_CLASS_WEIGHT)},
            "file_lists": {"train_files.txt": train_info, "val_files.txt": val_info},
            "init_checkpoint": init_report,
            "checkpoint_inventory": {"checkpoint_files": ["best.pt", "last.pt"], "save_every": 10},
            "history": {"epochs_recorded": 5},
            "environment": collect_environment(),
        }
        private_before = copy.deepcopy(private_summary)
        shareable = build_shareable_summary(private_summary)

        assert "run_dir" not in shareable
        assert "path" not in shareable["init_checkpoint"]
        for name in ("train_files.txt", "val_files.txt"):
            assert "private_lines" not in shareable["file_lists"][name]
            assert shareable["file_lists"][name]["content_sha256"]

        privacy = privacy_self_check(shareable)
        assert privacy["timestamp_like_video_ids_absent"], privacy
        assert privacy["absolute_host_paths_absent"], privacy

        # The private copy still carries the real values, and is not mutated.
        assert private_summary == private_before
        assert private_summary["file_lists"]["train_files.txt"]["private_lines"][0].startswith(FAKE_DATA_ROOT)
        private_privacy = privacy_self_check(private_summary)
        assert not private_privacy["timestamp_like_video_ids_absent"]
        assert not private_privacy["absolute_host_paths_absent"]
        print("  ok: shareable output strips paths/filenames and passes both privacy patterns")


def test_privacy_self_check_detects_the_patterns_it_exists_for() -> None:
    leaked_id = {"note": "video 20260711_120001_3 regressed"}
    leaked_path = {"note": "/home/kodaira/anaconda3/envs/dualtrack311/bin/python"}
    clean = {"note": "validation_000 regressed", "sha256": "a" * 64}
    assert not privacy_self_check(leaked_id)["timestamp_like_video_ids_absent"]
    assert not privacy_self_check(leaked_path)["absolute_host_paths_absent"]
    assert privacy_self_check(clean)["timestamp_like_video_ids_absent"]
    assert privacy_self_check(clean)["absolute_host_paths_absent"]
    print("  ok: privacy self-check flags timestamp-style IDs and absolute host paths")


def main() -> None:
    tests = [
        test_values_match_distinguishes_bool_from_int,
        test_matching_config_passes_every_expected_key,
        test_config_mismatches_are_all_reported_not_short_circuited,
        test_missing_config_key_is_failure_not_crash,
        test_sample_count_identity_check_detects_a_different_run,
        test_class_weight_auto_mode_is_judge_when_effective_weight_matches,
        test_class_weight_wrong_effective_value_fails,
        test_file_list_checks_count_uniqueness_and_teacher_version,
        test_duplicate_basename_fails_because_stable_video_id_would_collide,
        test_wrong_teacher_suffix_and_count_are_reported,
        test_file_list_fingerprint_is_stable_and_order_sensitive,
        test_init_checkpoint_hash_comes_from_the_config_path,
        test_missing_init_checkpoint_is_unknown_and_batchnorm_name_is_judge,
        test_absent_init_checkpoint_entry_is_failure,
        test_missing_per_epoch_checkpoints_is_judge_with_save_every_recorded,
        test_missing_best_checkpoint_is_failure,
        test_history_epoch_count_and_nonfinite_metrics,
        test_environment_reports_a_real_multiprocessing_start_method,
        test_shareable_summary_drops_paths_and_passes_privacy_self_check,
        test_privacy_self_check_detects_the_patterns_it_exists_for,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 S5-15 P1 manifest audit synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
