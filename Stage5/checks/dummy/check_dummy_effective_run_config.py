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

from check_stage5_effective_run_config import (  # noqa: E402
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    check_checkpoint_present,
    check_settings,
    coerce_to,
    parse_expected,
)

# ----------------------------------------------------------------------------
# S5-15: the runtime counterpart to the static env-passthrough test.
#
# Its whole purpose is to catch, within the first few epochs, the class of
# deviation that went unnoticed for two full 5-epoch runs: an exported setting
# that never reached train_stage5.py, and periodic checkpoint saving silently
# doing nothing. Both cases are exercised here against real config.json files.
# ----------------------------------------------------------------------------

CHECKER = REPO_ROOT / "checks" / "real_h5" / "check_stage5_effective_run_config.py"


def write_run(root: Path, *, config: dict[str, Any], checkpoints: tuple[str, ...] = ()) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for name in checkpoints:
        (root / name).write_bytes(b"checkpoint bytes")
    return root


def long_run_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "epochs": 50,
        "save_every": 5,
        "augmentation": "none",
        "seed": 42,
        "pointnext_norm": "groupnorm",
        "pointnext_norm_groups": 8,
        "label_policy": "bbox_noncontour_ignore",
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "window_size_frames": 16,
        "window_stride_frames": 8,
        "include_tail_window": True,
        "class_weight": [0.05963856, 1.9403615],
    }
    config.update(overrides)
    return config


def status_of(results: list[dict[str, Any]], check: str) -> str:
    matches = [r["status"] for r in results if r["check"] == check]
    assert len(matches) == 1, f"expected one result for {check!r}, found {len(matches)}"
    return matches[0]


def test_expected_pairs_are_parsed_and_typed_from_the_effective_value() -> None:
    assert parse_expected(["epochs=50", "augmentation=none"]) == {"epochs": "50", "augmentation": "none"}
    # The expected text is read in whatever type config.json stored.
    assert coerce_to(50, "50") == 50
    assert coerce_to(0.5, "0.5") == 0.5
    assert coerce_to(True, "true") is True
    assert coerce_to(False, "0") is False
    assert coerce_to("none", "none") == "none"
    assert coerce_to([1, 2], "[1, 2]") == [1, 2]
    # A string "50" must not be mistaken for the integer 50.
    assert coerce_to("50", "50") == "50"
    print("  ok: KEY=VALUE pairs parse, and the expected value is typed like the effective one")


def test_matching_config_passes_every_expectation() -> None:
    results = check_settings(long_run_config(), parse_expected(["epochs=50", "save_every=5", "augmentation=none"]))
    assert all(r["status"] == STATUS_PASS for r in results), results
    print("  ok: a config matching every expectation passes")


def test_the_real_save_every_deviation_is_caught() -> None:
    """The S5-15 P3 case: SAVE_EVERY=1 was exported, config.json held 10.
    Run against a 50-epoch expectation, this fails immediately."""
    results = check_settings(long_run_config(save_every=10), parse_expected(["save_every=5"]))
    assert status_of(results, "config.save_every") == STATUS_FAIL
    assert "expected=5 effective=10" in [r["detail"] for r in results][0]
    print("  ok: an effective save_every that differs from the intended one is FAIL, not a warning")


def test_absent_key_and_wrong_values_are_failures() -> None:
    results = check_settings(long_run_config(), parse_expected(["not_a_key=1", "seed=7", "augmentation=random_z_rotation"]))
    assert status_of(results, "config.not_a_key") == STATUS_FAIL
    assert status_of(results, "config.seed") == STATUS_FAIL
    assert status_of(results, "config.augmentation") == STATUS_FAIL
    print("  ok: a missing key and wrong numeric/string values are all reported")


def test_missing_periodic_checkpoint_is_reported_with_what_is_there() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = write_run(Path(tmp) / "run", config=long_run_config(), checkpoints=("best.pt", "last.pt"))
        result = check_checkpoint_present(run, "checkpoint_epoch_0005.pt")
        assert result["status"] == STATUS_FAIL
        assert "best.pt" in result["detail"] and "last.pt" in result["detail"]
        assert "stop and report" in result["detail"]

        present = write_run(
            Path(tmp) / "run2",
            config=long_run_config(),
            checkpoints=("best.pt", "last.pt", "checkpoint_epoch_0005.pt"),
        )
        ok = check_checkpoint_present(present, "checkpoint_epoch_0005.pt")
        assert ok["status"] == STATUS_PASS and "periodic saving is working" in ok["detail"]

        skipped = check_checkpoint_present(present, None)
        assert skipped["status"] == STATUS_UNKNOWN
    print("  ok: a missing periodic checkpoint fails and lists what the run dir does hold")


def test_cli_exits_nonzero_on_deviation_and_zero_when_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = write_run(
            root / "good",
            config=long_run_config(),
            checkpoints=("best.pt", "last.pt", "checkpoint_epoch_0005.pt"),
        )
        clean = subprocess.run(
            [sys.executable, str(CHECKER), "--run_dir", str(good),
             "--expect", "epochs=50", "--expect", "save_every=5",
             "--expect_checkpoint", "checkpoint_epoch_0005.pt",
             "--json_out", str(root / "good.json")],
            capture_output=True, text=True,
        )
        assert clean.returncode == 0, clean.stderr
        assert json.loads((root / "good.json").read_text())["status"] == "passed"

        bad = write_run(root / "bad", config=long_run_config(save_every=10), checkpoints=("best.pt", "last.pt"))
        deviating = subprocess.run(
            [sys.executable, str(CHECKER), "--run_dir", str(bad),
             "--expect", "save_every=5", "--expect_checkpoint", "checkpoint_epoch_0005.pt",
             "--json_out", str(root / "bad.json")],
            capture_output=True, text=True,
        )
        assert deviating.returncode == 1, deviating
        summary = json.loads((root / "bad.json").read_text())
        assert summary["status"] == "failed"
        assert summary["counts"][STATUS_FAIL] == 2, summary["counts"]

        missing = subprocess.run(
            [sys.executable, str(CHECKER), "--run_dir", str(root / "absent")],
            capture_output=True, text=True,
        )
        assert missing.returncode == 2 and "config.json not found" in missing.stderr
    print("  ok: the CLI exits 1 on deviation, 0 when clean, and 2 when there is no config.json yet")


def test_run_directory_is_never_modified() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = write_run(
            Path(tmp) / "run",
            config=long_run_config(),
            checkpoints=("best.pt", "last.pt", "checkpoint_epoch_0005.pt"),
        )
        before = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
        subprocess.run(
            [sys.executable, str(CHECKER), "--run_dir", str(run), "--expect", "epochs=50"],
            capture_output=True, text=True, check=True,
        )
        after = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
        assert before == after, "the inspected run directory must be untouched"
    print("  ok: inspecting a run directory leaves every file untouched")


def main() -> None:
    tests = [
        test_expected_pairs_are_parsed_and_typed_from_the_effective_value,
        test_matching_config_passes_every_expectation,
        test_the_real_save_every_deviation_is_caught,
        test_absent_key_and_wrong_values_are_failures,
        test_missing_periodic_checkpoint_is_reported_with_what_is_there,
        test_cli_exits_nonzero_on_deviation_and_zero_when_clean,
        test_run_directory_is_never_modified,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 effective run-config tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
