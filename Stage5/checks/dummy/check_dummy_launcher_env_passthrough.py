from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# S5-15: every environment variable the arm launcher exports must actually be
# honoured by train_stage5.sh.
#
# This encodes the defect found after the P3 training runs: train_stage5.sh
# wrote SAVE_EVERY=10 as a plain assignment, which overwrote the SAVE_EVERY=1
# the launcher exported. Nothing failed and nothing warned -- the runs simply
# produced no per-epoch checkpoints, and the launch manifest recorded the
# intended 1 while config.json recorded the effective 10.
#
# Pure text analysis of the two shell scripts: no torch, no CUDA, no data.
# ----------------------------------------------------------------------------

TRAIN_SH = REPO_ROOT / "train_stage5.sh"
LAUNCHER_SH = REPO_ROOT / "checks" / "real_h5" / "run_stage5_s5_15_arm.sh"

# Assignment honouring an inherited value: VAR="${VAR:-default}" / ${VAR:?...}
OVERRIDABLE_PATTERN = re.compile(r'^([A-Z_][A-Z0-9_]*)=.*\$\{\1(:-|:\?|-)')
ANY_ASSIGNMENT_PATTERN = re.compile(r"^([A-Z_][A-Z0-9_]*)=")


def classify_assignments(script: Path) -> dict[str, str]:
    """Map variable name -> "overridable" | "hard_coded" (first assignment wins)."""
    classified: dict[str, str] = {}
    for raw_line in script.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = ANY_ASSIGNMENT_PATTERN.match(line)
        if not match:
            continue
        name = match.group(1)
        if name in classified:
            continue
        classified[name] = "overridable" if OVERRIDABLE_PATTERN.match(line) else "hard_coded"
    return classified


def launcher_exported_names() -> list[str]:
    text = LAUNCHER_SH.read_text(encoding="utf-8")
    block = re.search(r"^launch_env=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    assert block, "could not find the launch_env=( ... ) block in the launcher"
    names = re.findall(r'"([A-Z_][A-Z0-9_]*)=', block.group(1))
    assert names, "launch_env appears to export nothing"
    return names


def test_every_exported_variable_is_honoured_by_train_stage5() -> None:
    exported = launcher_exported_names()
    classified = classify_assignments(TRAIN_SH)

    ignored: list[str] = []
    hard_coded: list[str] = []
    for name in exported:
        state = classified.get(name)
        if state is None:
            # Not assigned in train_stage5.sh at all: the environment reaches
            # the CLI untouched, which is fine.
            ignored.append(name)
        elif state == "hard_coded":
            hard_coded.append(name)

    assert not hard_coded, (
        "train_stage5.sh overwrites these exported variables with plain assignments, so the "
        f"launcher's values are silently discarded: {hard_coded}. "
        'Write them as VAR="${VAR:-default}".'
    )
    print(
        f"  ok: all {len(exported)} exported variables are honoured "
        f"({len(exported) - len(ignored)} overridable in train_stage5.sh, {len(ignored)} not assigned there)"
    )


def test_the_four_previously_hard_coded_knobs_are_overridable_now() -> None:
    """The specific regression: these were plain assignments when the P3 runs
    were launched, and SAVE_EVERY=1 was lost because of it."""
    classified = classify_assignments(TRAIN_SH)
    for name, default in (("SAVE_EVERY", "10"), ("SEED", "42"), ("GRAD_CLIP_NORM", "10"), ("NUM_WORKERS", "4")):
        assert classified.get(name) == "overridable", f"{name} is {classified.get(name)!r}, expected overridable"
    text = TRAIN_SH.read_text(encoding="utf-8")
    for name, default in (("SAVE_EVERY", "10"), ("SEED", "42"), ("GRAD_CLIP_NORM", "10"), ("NUM_WORKERS", "4")):
        assert f'{name}="${{{name}:-{default}}}"' in text, f"{name}'s default changed; general use must be unaffected"
    print("  ok: SAVE_EVERY/SEED/GRAD_CLIP_NORM/NUM_WORKERS are overridable and keep their original defaults")


def test_classifier_actually_distinguishes_the_two_forms() -> None:
    """Guards the check itself: a classifier that called everything
    overridable would have passed the broken script too."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.sh"
        sample.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "# comment SHOULD_BE_IGNORED=1",
                    'GOOD="${GOOD:-5}"',
                    "BARE=10",
                    'REQUIRED="${REQUIRED:?must be set}"',
                    'DASH="${DASH-fallback}"',
                    'QUOTED_DEFAULT="${QUOTED_DEFAULT:-some value}"',
                    "OTHER_VAR=\"${UNRELATED:-x}\"",
                    'GOOD="${GOOD:-9}"',
                ]
            ),
            encoding="utf-8",
        )
        classified = classify_assignments(sample)
        assert classified["GOOD"] == "overridable"
        assert classified["BARE"] == "hard_coded"
        assert classified["REQUIRED"] == "overridable"
        assert classified["DASH"] == "overridable"
        assert classified["QUOTED_DEFAULT"] == "overridable"
        # Assigned from a *different* variable: its own value is not honoured.
        assert classified["OTHER_VAR"] == "hard_coded"
        assert "SHOULD_BE_IGNORED" not in classified
    print("  ok: the classifier separates VAR=\"${VAR:-d}\" from VAR=d, and ignores comments")


def test_the_broken_form_would_be_caught() -> None:
    """Re-runs the real check against a copy of train_stage5.sh with the old
    plain assignment restored, and requires it to fail."""
    import tempfile

    original = TRAIN_SH.read_text(encoding="utf-8")
    broken = original.replace('SAVE_EVERY="${SAVE_EVERY:-10}"', "SAVE_EVERY=10")
    assert broken != original, "could not construct the broken variant"

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "train_stage5.sh"
        sample.write_text(broken, encoding="utf-8")
        classified = classify_assignments(sample)
        assert classified["SAVE_EVERY"] == "hard_coded"
        assert "SAVE_EVERY" in launcher_exported_names()
    print("  ok: restoring the old SAVE_EVERY=10 form is classified hard_coded, so this test would fail")


def main() -> None:
    tests = [
        test_every_exported_variable_is_honoured_by_train_stage5,
        test_the_four_previously_hard_coded_knobs_are_overridable_now,
        test_classifier_actually_distinguishes_the_two_forms,
        test_the_broken_form_would_be_caught,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 launcher env passthrough tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
