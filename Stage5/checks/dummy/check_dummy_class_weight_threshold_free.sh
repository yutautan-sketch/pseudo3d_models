#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-13補足 Step 3: synthetic pass/fail coverage for the threshold-free
# (AUPRC/AUROC/PR curve) diagnostic math used by
# check_stage5_class_weight_threshold_free.py. Pure numpy, no CUDA/H5/real
# run data needed.
#
#   bash checks/dummy/check_dummy_class_weight_threshold_free.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 class-weight threshold-free diagnostics synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_class_weight_threshold_free.py"

echo "Done."
