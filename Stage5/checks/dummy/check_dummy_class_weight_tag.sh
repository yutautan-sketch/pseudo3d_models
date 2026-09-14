#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-13 Step F3: synthetic test of train_stage5.sh's class_weight_tag()
# function (auto/manual/none -> filesystem-safe run-name tag). The function
# definition is extracted directly from train_stage5.sh via sed so this test
# cannot silently drift from the real implementation; train_stage5.sh itself
# is not run here (it requires real Stage 4 H5 input data and CUDA).
#
#   bash checks/dummy/check_dummy_class_weight_tag.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
TRAIN_SH="${TRAIN_SH:-${SCRIPT_DIR}/train_stage5.sh}"

if [[ ! -f "${TRAIN_SH}" ]]; then
  echo "train_stage5.sh not found: ${TRAIN_SH}" >&2
  exit 1
fi

function_body="$(sed -n '/^class_weight_tag() {/,/^}/p' "${TRAIN_SH}")"
if [[ -z "${function_body}" ]]; then
  echo "Could not extract class_weight_tag() from ${TRAIN_SH}" >&2
  exit 1
fi
eval "${function_body}"

fail=0
check() {
  local input="$1"
  local expected="$2"
  local actual
  actual="$(class_weight_tag "${input}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "FAIL: class_weight_tag('${input}') = '${actual}', expected '${expected}'" >&2
    fail=1
  else
    echo "ok: class_weight_tag('${input}') = '${actual}'"
  fi
}

check "auto" "cw_auto"
check "Auto" "cw_auto"
check "pointnext_auto" "cw_auto"
check "" "cw_none"
check "0.5,1.5" "cw_manual_0p5_1p5"
check "0.05963856,1.94036150" "cw_manual_0p05963856_1p94036150"
check "1,1" "cw_manual_1_1"
check " 0.5 , 1.5 " "cw_manual_0p5_1p5"

if [[ "${fail}" -ne 0 ]]; then
  echo "class_weight_tag synthetic test FAILED" >&2
  exit 1
fi
echo "class_weight_tag synthetic test passed (8 cases)."
