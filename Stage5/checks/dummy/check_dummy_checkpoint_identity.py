from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))

from check_stage5_checkpoint_identity import (  # noqa: E402
    compare_state_dicts,
    load_checkpoint,
    sha256_file,
    summarize_metadata,
)

# ----------------------------------------------------------------------------
# S5-15: synthetic coverage for the checkpoint identity audit.
#
# The case that matters most is the middle one: two files that are NOT
# byte-identical but hold exactly the same model tensors. A hash mismatch alone
# must never be reported as "a different model".
# ----------------------------------------------------------------------------

CHECKER = REPO_ROOT / "checks" / "real_h5" / "check_stage5_checkpoint_identity.py"


def make_state_dict(*, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "encoder.weight": torch.randn(4, 3, generator=generator),
        "encoder.bias": torch.randn(4, generator=generator),
        "head.weight": torch.randn(2, 4, generator=generator),
    }


def write_checkpoint(
    path: Path,
    *,
    state: dict[str, torch.Tensor],
    epoch: int = 5,
    best_score: float = 0.12,
    config: dict[str, Any] | None = None,
    optimizer: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": state,
            "optimizer": optimizer if optimizer is not None else {"state": {}, "param_groups": [{"lr": 0.001}]},
            "config": config if config is not None else {"model": "pointnext_s", "seed": 42},
            "train_metrics": {"loss": 0.2},
            "val_metrics": {"loss": 0.25},
            "best_score": best_score,
        },
        path,
    )
    return path


def run_checker(run_dir: Path, json_out: Path, *extra: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--run_dir", str(run_dir), "--json_out", str(json_out), *extra],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"checker failed: {result.stderr}"
    return json.loads(json_out.read_text(encoding="utf-8"))


def test_byte_identical_checkpoints_need_no_load() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        state = make_state_dict()
        best = write_checkpoint(run_dir / "best.pt", state=state)
        last = run_dir / "last.pt"
        last.write_bytes(best.read_bytes())

        summary = run_checker(run_dir, Path(tmp) / "out.json")
        assert summary["file_identical"] is True
        assert summary["evaluation_model_identical"] is True
        assert summary["conclusion"] == "identical_file"
        assert "state_dict_comparison" not in summary, "an identical file needs no state_dict comparison"
    print("  ok: byte-identical best.pt/last.pt conclude 'identical_file' without loading either")


def test_different_files_with_identical_weights_are_not_called_a_different_model() -> None:
    """Policy-chat 8.7.3: a SHA-256 mismatch alone must not be read as a
    different model. Here only the optimizer state and best_score differ."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        state = make_state_dict()
        write_checkpoint(run_dir / "best.pt", state=state, best_score=0.12)
        write_checkpoint(
            run_dir / "last.pt",
            state={key: value.clone() for key, value in state.items()},
            best_score=0.10,
            optimizer={"state": {"step": 715}, "param_groups": [{"lr": 0.001}]},
        )

        summary = run_checker(run_dir, Path(tmp) / "out.json")
        assert summary["file_identical"] is False
        assert summary["evaluation_model_identical"] is True
        assert summary["conclusion"] == "different_file_same_evaluation_model"
        assert summary["state_dict_comparison"]["num_value_mismatches"] == 0
        assert summary["config_identical"] is True
        assert summary["metadata_a"]["best_score"] != summary["metadata_b"]["best_score"]
    print("  ok: differing files with identical tensors conclude 'different_file_same_evaluation_model'")


def test_genuinely_different_weights_are_reported_with_the_diff() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        write_checkpoint(run_dir / "best.pt", state=make_state_dict(seed=0), epoch=3)
        write_checkpoint(run_dir / "last.pt", state=make_state_dict(seed=1), epoch=5)

        summary = run_checker(run_dir, Path(tmp) / "out.json")
        assert summary["file_identical"] is False
        assert summary["evaluation_model_identical"] is False
        assert summary["conclusion"] == "different_evaluation_model"
        comparison = summary["state_dict_comparison"]
        assert comparison["num_value_mismatches"] == 3
        assert all(entry["max_abs_diff"] > 0 for entry in comparison["value_mismatches"])
        assert summary["metadata_a"]["epoch"] == 3 and summary["metadata_b"]["epoch"] == 5
    print("  ok: different tensors conclude 'different_evaluation_model' with per-key max_abs_diff")


def test_shape_and_key_differences_are_separated_from_value_differences() -> None:
    state_a = make_state_dict()
    state_b = {key: value.clone() for key, value in state_a.items()}
    state_b["head.weight"] = torch.randn(3, 4)
    state_b["extra.weight"] = torch.zeros(2)
    del state_b["encoder.bias"]

    comparison = compare_state_dicts(state_a, state_b)
    assert comparison["only_in_a"] == ["encoder.bias"]
    assert comparison["only_in_b"] == ["extra.weight"]
    assert comparison["num_shape_or_dtype_mismatches"] == 1
    assert comparison["identical"] is False

    dtype_shifted = {key: value.to(torch.float64) for key, value in state_a.items()}
    dtype_comparison = compare_state_dicts(state_a, dtype_shifted)
    assert dtype_comparison["num_shape_or_dtype_mismatches"] == 3
    assert dtype_comparison["num_value_mismatches"] == 0, "a dtype change is reported as such, not as a value diff"
    print("  ok: missing/extra keys, shape changes and dtype changes are reported apart from value differences")


def test_skip_load_reports_unknown_rather_than_guessing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        write_checkpoint(run_dir / "best.pt", state=make_state_dict(seed=0))
        write_checkpoint(run_dir / "last.pt", state=make_state_dict(seed=1))

        summary = run_checker(run_dir, Path(tmp) / "out.json", "--skip_load")
        assert summary["file_identical"] is False
        assert summary["evaluation_model_identical"] is None
        assert summary["conclusion"] == "file_differs_model_not_checked"
    print("  ok: --skip_load leaves the model question unanswered instead of inferring it from the hash")


def test_checker_reads_without_modifying_the_run_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        best = write_checkpoint(run_dir / "best.pt", state=make_state_dict(seed=0))
        last = write_checkpoint(run_dir / "last.pt", state=make_state_dict(seed=1))
        before = {p.name: (p.stat().st_mtime_ns, sha256_file(p)) for p in (best, last)}
        listing_before = sorted(p.name for p in run_dir.iterdir())

        run_checker(run_dir, Path(tmp) / "out.json")

        after = {p.name: (p.stat().st_mtime_ns, sha256_file(p)) for p in (best, last)}
        assert before == after, "the audited checkpoints must be untouched"
        assert sorted(p.name for p in run_dir.iterdir()) == listing_before, "no file may be added to the run dir"
    print("  ok: the audited checkpoints keep their bytes and mtimes, and no file is added to the run directory")


def test_load_and_metadata_helpers_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = write_checkpoint(Path(tmp) / "best.pt", state=make_state_dict(), epoch=5, best_score=0.42)
        payload, mode = load_checkpoint(path)
        metadata = summarize_metadata(payload)
        assert metadata["epoch"] == 5
        assert metadata["best_score"] == 0.42
        assert metadata["has_optimizer_state"] is True
        assert "model" in metadata["top_level_keys"] and "config" in metadata["top_level_keys"]
        assert isinstance(mode, str) and mode
    print(f"  ok: a saved checkpoint round-trips through the loader (load mode recorded)")


def main() -> None:
    tests = [
        test_byte_identical_checkpoints_need_no_load,
        test_different_files_with_identical_weights_are_not_called_a_different_model,
        test_genuinely_different_weights_are_reported_with_the_diff,
        test_shape_and_key_differences_are_separated_from_value_differences,
        test_skip_load_reports_unknown_rather_than_guessing,
        test_checker_reads_without_modifying_the_run_directory,
        test_load_and_metadata_helpers_round_trip,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 checkpoint identity synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
