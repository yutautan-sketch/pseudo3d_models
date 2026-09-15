#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14補足3: synthetic pass/fail coverage for
# check_stage5_coordinate_transform_reconciliation.py -- point-weighted
# pooling vs. naive per-video-rate mean, median-of-diffs vs.
# diff-of-medians divergence, and the duplicate/missing-video and
# identity-self-diff fail-fast checks. No CUDA needed.
#
#   bash checks/dummy/check_dummy_coordinate_transform_reconciliation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 supplement3 reconciliation synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_coordinate_transform_reconciliation.py"

echo "Done."
