#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-13 Step F3: synthetic pass/fail coverage for
# checks/real_h5/check_stage5_class_weight_ablation.py's comparison logic.
# Pure JSON/CSV fixtures in a temporary directory. No CUDA, H5, or real run
# data needed anywhere in this checker.
#
#   bash checks/dummy/check_dummy_class_weight_ablation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 class-weight ablation config-parity synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_class_weight_ablation.py"

echo "Done."
