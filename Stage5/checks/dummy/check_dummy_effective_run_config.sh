#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: synthetic coverage for check_stage5_effective_run_config.py -- the
# runtime counterpart to the static env-passthrough test. Covers the real P3
# deviation (effective save_every=10 against an intended 5), missing keys and
# wrong values, a missing periodic checkpoint reported with what the run dir
# does hold, CLI exit codes, and the run directory staying untouched.
# Pure stdlib: no torch, no CUDA, no real run data.
#
#   bash checks/dummy/check_dummy_effective_run_config.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 effective run-config synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_effective_run_config.py"

# ------------------------------------------------------------
# Shell-level behaviour of check_stage5_effective_run_config.sh itself.
#
# The Python layer already treats an empty --expect_checkpoint as "skip", but
# the wrapper used "${VAR:-default}", which substitutes for an EMPTY value too,
# so EXPECT_CHECKPOINT="" silently fell back to the default and failed a
# healthy run that had not reached its first periodic save yet. That layer is
# exercised here through the real wrapper, with its hard-coded SCRIPT_DIR
# rewritten to this checkout so the test runs anywhere.
# ------------------------------------------------------------
fixture="$(mktemp -d)"
trap 'rm -rf "${fixture}"' EXIT
mkdir -p "${fixture}/run"
cat > "${fixture}/run/config.json" <<'JSON'
{"epochs": 50, "save_every": 5, "augmentation": "none", "seed": 42,
 "pointnext_norm": "groupnorm", "pointnext_norm_groups": 8,
 "label_policy": "bbox_noncontour_ignore", "batch_size": 1,
 "gradient_accumulation_steps": 8, "window_size_frames": 16, "window_stride_frames": 8}
JSON

sed -e "s#^SCRIPT_DIR=.*#SCRIPT_DIR=${SCRIPT_DIR}#" \
    "${SCRIPT_DIR}/checks/real_h5/check_stage5_effective_run_config.sh" > "${fixture}/wrapper.sh"

echo "[wrapper: EXPECT_CHECKPOINT= (empty) skips the checkpoint check]"
if RUN_DIR="${fixture}/run" EXPECT_CHECKPOINT="" JSON_OUT="${fixture}/skip.json" \
   PYTHON="${PYTHON}" bash "${fixture}/wrapper.sh" >/dev/null 2>&1; then
  echo "  ok: a run inspected before its first periodic save passes when the check is skipped"
else
  echo "FAIL: an empty EXPECT_CHECKPOINT still demanded a checkpoint" >&2
  exit 1
fi

echo "[wrapper: EXPECT_CHECKPOINT unset still defaults to checkpoint_epoch_0005.pt]"
if RUN_DIR="${fixture}/run" JSON_OUT="${fixture}/default.json" \
   PYTHON="${PYTHON}" bash "${fixture}/wrapper.sh" >/dev/null 2>&1; then
  echo "FAIL: the default checkpoint expectation was not applied" >&2
  exit 1
else
  echo "  ok: leaving it unset still requires checkpoint_epoch_0005.pt"
fi

echo "[wrapper: a real deviation still fails]"
sed -i 's/"save_every": 5/"save_every": 10/' "${fixture}/run/config.json"
if RUN_DIR="${fixture}/run" EXPECT_CHECKPOINT="" JSON_OUT="${fixture}/dev.json" \
   PYTHON="${PYTHON}" bash "${fixture}/wrapper.sh" >/dev/null 2>&1; then
  echo "FAIL: an effective save_every of 10 was not reported" >&2
  exit 1
else
  echo "  ok: skipping the checkpoint check does not mask a config deviation"
fi

echo
echo "Done."
