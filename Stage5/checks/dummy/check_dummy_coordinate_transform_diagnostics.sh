#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14補足2 Step T1: synthetic pass/fail coverage for
# check_stage5_coordinate_transform_diagnostics.py's torch-free logic
# (transform math, centering-cancels-pre-normalization-translation,
# per-point comparison metrics, hot/cold-bin FPR reuse, condition-level
# aggregation). No CUDA needed.
#
#   bash checks/dummy/check_dummy_coordinate_transform_diagnostics.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 supplement2 coordinate-transform synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_coordinate_transform_diagnostics.py"

echo "Done."
