#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14補足 Step S1-S5: synthetic pass/fail coverage for
# check_stage5_s5_14_supplement.py (denominator-carrying XY bin FPR/recall,
# train-only XY prior + self-exclusion + quantile tie handling, joint
# half x vote-count stratification, paired within-video temporal recall,
# and an end-to-end CLI privacy/alias check). No CUDA needed.
#
#   bash checks/dummy/check_dummy_s5_14_supplement.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 supplement synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_s5_14_supplement.py"

echo "Done."
