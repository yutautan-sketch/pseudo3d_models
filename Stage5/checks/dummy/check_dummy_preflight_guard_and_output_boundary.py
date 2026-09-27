from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.split_identity import contains_video_identity  # noqa: E402

TRAIN_SH = REPO_ROOT / "train_stage5.sh"

# ----------------------------------------------------------------------------
# S5-16 Step 0: two properties of train_stage5.sh that live in bash, not Python.
#
# 1. ORDER (report 10.4). The teacher preflight opens every H5 in the list
#    manifest. A seal guard placed after it would be reading sealed videos in
#    order to decide whether it may read sealed videos. So the guard call must
#    appear -- and stop the script -- before the preflight runs.
#
# 2. OUTPUT BOUNDARY (report 10.10.3). The 180-file QC constants are already in
#    this script's git history. If the 162-file observations or expectations
#    reach shared output, on success OR on failure, the sealed videos' QC totals
#    follow by subtraction. So neither path may print them.
#
#    Note this is a different property from "no real paths in the output", and
#    it is checked separately: a message can be perfectly free of paths and
#    still hand over the statistics.
#
# The script cds into a hard-coded /mnt path, so it is examined structurally
# and its extracted pieces are executed, in the same spirit as
# check_dummy_fixed_list_mode.sh's sed extraction. Pure stdlib; no H5 is opened
# and no training is started.
# ----------------------------------------------------------------------------

QC_VARIABLES = (
    "cvat_videos", "cvat_frames", "invalidated_frames", "invalidated_videos",
    "removed_positive", "removed_bbox_rows", "crop_invalidated_frames",
    "crop_invalidated_videos", "crop_removed_positive", "crop_removed_ignore",
    "crop_removed_bbox_rows",
)

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def embedded_preflight(text: str) -> str:
    """The preflight heredoc specifically.

    train_stage5.sh has more than one `<<'PY'` block -- the torch-lib probe near
    the top uses the same delimiter -- so the block is selected by content
    rather than by being the first one found.
    """
    blocks = [chunk.split("\nPY\n", 1)[0] for chunk in text.split("<<'PY'")[1:]]
    matching = [block for block in blocks if "teacher v7 input preflight" in block]
    if len(matching) != 1:
        raise AssertionError(f"expected exactly one preflight heredoc, found {len(matching)}")
    return matching[0]


def test_guard_runs_before_the_preflight(text: str) -> None:
    print("\n[1] the seal guard runs before the teacher preflight")
    guard = text.find('split_contract.py" "${split_contract_args[@]}"')
    preflight = text.find('"${PYTHON}" - \\\n  "${INPUT_DIR}"')
    check(guard != -1, "train_stage5.sh calls the split contract guard")
    check(preflight != -1, "train_stage5.sh still runs the teacher preflight")
    check(guard < preflight, "the guard call precedes the preflight invocation")

    validation = text.find("stage5/utils/file_list_mode.py")
    check(validation < guard, "the guard runs after the list pair has been validated")
    check("set -euo pipefail" in text, "the script aborts on the guard's non-zero exit")

    manifest_write = text.find("--manifest_out")
    check(manifest_write < guard, "the guard checks the manifest the preflight will actually read")
    check(
        "--paths_from" in text and text.find("--paths_from") < preflight,
        "the guard is handed the resolved list, not a directory",
    )
    check(
        "--require_directory_mode" in text,
        "a directory-scan run asks the contract whether directory mode is permitted at all",
    )


def test_guard_stops_without_a_contract() -> None:
    print("\n[2] the guard stops when no contract is approved")
    module = REPO_ROOT / "stage5" / "utils" / "split_contract.py"
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "manifest.txt"
        listing.write_text("/nowhere/20250701_101010_101_x.h5\n", encoding="utf-8")
        missing_pins = Path(tmp) / "absent_pins.json"
        result = subprocess.run(
            [sys.executable, str(module), "assert", "--split_manifest", "s5_16_bprime",
             "--pins", str(missing_pins), "--paths_from", str(listing),
             "--purpose", "training preflight inputs", "--require_new_training"],
            capture_output=True, text=True,
        )
        check(result.returncode == 2, f"an absent pins file exits 2 (got {result.returncode})")
        check("preflight passed" not in result.stdout, "the preflight never ran")
        check(
            not contains_video_identity(result.stdout + result.stderr),
            "the refusal carries no video ID",
        )


def test_output_boundary(text: str) -> None:
    print("\n[3] QC totals stay out of shared output on success AND on failure")
    body = embedded_preflight(text)
    tree = ast.parse(body)

    shared_messages: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            shared_messages.append(ast.unparse(node))
        if isinstance(node, ast.Raise) and "SystemExit" in ast.unparse(node):
            shared_messages.append(ast.unparse(node))

    check(bool(shared_messages), "the preflight produces shared output to inspect")

    leaked = sorted(
        {
            variable
            for message in shared_messages
            for variable in QC_VARIABLES
            if re.search(rf"\{{[^}}]*\b{variable}\b", message)
        }
    )
    check(not leaked, f"no QC total is interpolated into shared output (leaked: {leaked})")

    success = [m for m in shared_messages if "preflight passed" in m]
    check(len(success) == 1, "there is exactly one success message")
    check(
        "files=" in success[0] and "videos=" in success[0],
        "the success message still reports the permitted counts",
    )
    check("withheld" in success[0].lower(), "the success message says the totals are withheld")

    failure = [m for m in shared_messages if "preflight failed" in m]
    check(len(failure) == 1, "there is exactly one failure message")
    check(
        "withheld" in failure[0].lower(),
        "the failure message says observed and expected values are withheld",
    )
    check(
        "failed" in failure[0] and "join" in failure[0],
        "the failure message names which check failed, without its numbers",
    )

    check("_write_private" in body, "a private log receives the numbers a human needs")
    check(
        body.count("_write_private(") >= 3,
        "the private log is written on the emit, pass and fail paths",
    )
    check(
        "preflight_private_log" in body,
        "the private log path is configurable rather than fixed to a shared location",
    )


def test_emit_mode_is_separate_from_verification(text: str) -> None:
    print("\n[4] fixing new expectations is separate from passing the checks")
    body = embedded_preflight(text)
    check("emit_teacher_expected" in body, "there is an explicit emit mode for a new target set")
    check(
        "teacher_expected_json" in body,
        "expectations can be loaded from a pinned private file rather than script literals",
    )
    check(
        "DO_NOT_SHARE" in body,
        "the emitted expectations are marked unshareable (they differ from the 180-file constants "
        "by the sealed videos' contribution)",
    )
    check(
        "change detection" in body.lower(),
        "the emitted file states that it is a change-detection baseline, not a quality verdict",
    )
    tree = ast.parse(body)
    emit_branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and "emit_teacher_expected" in ast.unparse(node.test)
    ]
    check(len(emit_branches) == 1, "emit and compare are mutually exclusive branches")
    compare_branch = emit_branches[0].orelse if emit_branches else []
    check(
        any(isinstance(node, ast.Raise) for stmt in compare_branch for node in ast.walk(stmt)),
        "the compare branch can still stop the run",
    )
    check(
        not any(isinstance(node, ast.Raise) for node in ast.walk(emit_branches[0]))
        if not compare_branch else
        not any(
            isinstance(node, ast.Raise)
            for stmt in emit_branches[0].body
            for node in ast.walk(stmt)
        ),
        "the emit branch records a baseline instead of asserting one, so it never doubles as a check",
    )

    # The historical constants must remain in the script as the default.
    check("EXPECTED_INPUT_FILES:-180" in text, "the 180-file default is unchanged for old conditions")
    check(
        'TEACHER_EXPECTED_JSON="${TEACHER_EXPECTED_JSON:-}"' in text,
        "the new expectations file is opt-in, so existing behaviour is the default",
    )


def test_private_work_area(text: str) -> None:
    print("\n[5] the list manifest goes to the private work area, not /tmp")
    check("${TMPDIR:-/tmp}/stage5_fixed_list_manifest" not in text, "the /tmp manifest path is gone")
    check(
        'mktemp "${PRIVATE_WORK_DIR}/stage5_fixed_list_manifest' in text,
        "the manifest is created in the private work area",
    )
    check("PRIVATE_WORK_DIR=" in text, "the private work area is configurable")
    check(
        "stage5_private_work" in text and "sealed" not in text.split("PRIVATE_WORK_DIR=")[1][:200],
        "the private work area is distinct from the sealed area",
    )
    check(
        "trap 'rm -f \"${LIST_MANIFEST}\"' EXIT" in text,
        "the manifest is removed on normal and abnormal exit alike",
    )


def test_out_of_use_routes_are_blocked() -> None:
    print("\n[6] routes not connected to the contract cannot be reached indirectly")
    evaluate = (REPO_ROOT / "evaluate_stage5.sh").read_text(encoding="utf-8")
    longrun = (REPO_ROOT / "checks" / "real_h5" / "run_stage5_s5_15_r0_longrun.sh").read_text(
        encoding="utf-8"
    )

    # evaluate_stage5.sh auto-runs the anonymized exporter, which is not
    # guarded. Under a seal contract it must refuse, and not merely default to
    # off -- a default is something a caller can quietly set back.
    refusal = evaluate.find("Refusing to run export_anonymized_stage5_metrics.py")
    invocation = evaluate.find('"${SCRIPT_DIR}/export_anonymized_stage5_metrics.py"')
    check(refusal != -1, "evaluate_stage5.sh carries an explicit refusal for the exporter")
    check(refusal < invocation, "the refusal is reached before the invocation")
    check(
        'EXPORT_ANONYMIZED_METRICS}" == "1" && -n "${SPLIT_MANIFEST}"' in evaluate,
        "the refusal triggers whenever a split contract is in force, whatever the default",
    )
    check("exit 2" in evaluate[refusal:invocation], "the refusal stops the launcher")
    check(
        'EXPORT_ANONYMIZED_METRICS="${EXPORT_ANONYMIZED_METRICS:-1}"' in evaluate,
        "the historical default is left alone, so non-sealed use is unchanged",
    )

    check(
        "bash checks/real_h5/check_stage5_effective_run_config.sh" not in longrun,
        "the longrun launcher no longer suggests an out-of-use checker",
    )
    check(
        "out of use while internal_test is sealed" in longrun,
        "it says why the command was withdrawn instead of silently dropping it",
    )


def main() -> None:
    print("Stage5 S5-16 Step 0: preflight guard order and output boundary checks")
    text = TRAIN_SH.read_text(encoding="utf-8")
    test_guard_runs_before_the_preflight(text)
    test_guard_stops_without_a_contract()
    test_output_boundary(text)
    test_emit_mode_is_separate_from_verification(text)
    test_private_work_area(text)
    test_out_of_use_routes_are_blocked()
    print(f"\nchecks run: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
