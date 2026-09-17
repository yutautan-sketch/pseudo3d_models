#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P2 Step 3: fixed train/val file-list inputs.
#
# Two parts, neither of which starts any training:
#   1. train_stage5.sh's input_source_args() is extracted via sed (same
#      approach as check_dummy_class_weight_tag.sh) and exercised directly, so
#      mode exclusivity, the one-sided-list rejection and the directory-mode
#      regression are checked against the real implementation.
#   2. the Python list validation/manifest logic in
#      stage5/utils/file_list_mode.py.
#
#   bash checks/dummy/check_dummy_fixed_list_mode.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
TRAIN_SH="${TRAIN_SH:-${SCRIPT_DIR}/train_stage5.sh}"

if [[ ! -f "${TRAIN_SH}" ]]; then
  echo "train_stage5.sh not found: ${TRAIN_SH}" >&2
  exit 1
fi

function_body="$(sed -n '/^input_source_args() {/,/^}/p' "${TRAIN_SH}")"
if [[ -z "${function_body}" ]]; then
  echo "Could not extract input_source_args() from ${TRAIN_SH}" >&2
  exit 1
fi
eval "${function_body}"

fail=0

expect_args() {
  local description="$1"
  local expected="$2"
  shift 2
  local actual
  if ! actual="$(input_source_args "$@" 2>/dev/null | tr '\n' ' ')"; then
    echo "FAIL: ${description}: input_source_args exited non-zero" >&2
    fail=1
    return
  fi
  actual="${actual% }"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "FAIL: ${description}" >&2
    echo "  expected: ${expected}" >&2
    echo "  actual  : ${actual}" >&2
    fail=1
  else
    echo "  ok: ${description}"
  fi
}

expect_rejected() {
  local description="$1"
  shift
  if input_source_args "$@" >/dev/null 2>&1; then
    echo "FAIL: ${description}: expected a non-zero exit, but the call succeeded" >&2
    fail=1
  else
    echo "  ok: ${description}"
  fi
}

echo "[input_source_args]"

# Fixed-list mode: --train_dir must be absent (train_stage5.py rejects it
# alongside --train_list), no --val_fraction can reach the seed-based re-split,
# and truncation is disabled so the lists are used whole.
expect_args \
  "both lists set -> --train_list/--val_list, no --train_dir, no --val_fraction, truncation off" \
  "--train_list /lists/train.txt --val_list /lists/val.txt --max_train_files 0 --max_val_files 0" \
  "/lists/train.txt" "/lists/val.txt" "/data/collected" "0.1" "0" "0"

expect_args \
  "fixed-list mode ignores non-zero MAX_*_FILES so the lists are never truncated" \
  "--train_list /lists/train.txt --val_list /lists/val.txt --max_train_files 0 --max_val_files 0" \
  "/lists/train.txt" "/lists/val.txt" "/data/collected" "0.1" "5" "3"

# Directory mode regression: unchanged historical behavior.
expect_args \
  "neither list set -> directory scan with val_fraction (historical default path)" \
  "--train_dir /data/collected --val_fraction 0.1 --max_train_files 0 --max_val_files 0" \
  "" "" "/data/collected" "0.1" "0" "0"

expect_args \
  "directory mode still honors MAX_*_FILES" \
  "--train_dir /data/collected --val_fraction 0.2 --max_train_files 7 --max_val_files 2" \
  "" "" "/data/collected" "0.2" "7" "2"

# One-sided configuration must stop instead of quietly splitting a directory.
expect_rejected "only TRAIN_LIST set is rejected" "/lists/train.txt" "" "/data/collected" "0.1" "0" "0"
expect_rejected "only VAL_LIST set is rejected" "" "/lists/val.txt" "/data/collected" "0.1" "0" "0"

if [[ "${fail}" -ne 0 ]]; then
  echo "input_source_args synthetic test FAILED" >&2
  exit 1
fi
echo "  input_source_args: 6 cases passed"
echo

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_fixed_list_mode.py"

echo "Done."
