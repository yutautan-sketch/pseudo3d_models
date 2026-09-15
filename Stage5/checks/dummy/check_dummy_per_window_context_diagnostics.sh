#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H4: synthetic pass/fail coverage for
# check_stage5_per_window_context_diagnostics.py's torch-free logic (parity
# gate, vote-count bucketing, stratified summaries, TP/FP/FN/TN
# classification, per-bin exposure/disagreement, recurrence cross-check
# join). No CUDA/torch needed.
#
#   bash checks/dummy/check_dummy_per_window_context_diagnostics.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 Step H4 synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_per_window_context_diagnostics.py"

echo "Done."
