#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P1: synthetic pass/fail coverage for
# check_stage5_s5_15_p1_manifest_audit.py -- config/class-weight/file-list/
# checkpoint/history checks, the auto-vs-manual JUDGE path, duplicate
# basenames, and the shareable-output anonymization plus privacy self-check.
# Pure stdlib: no numpy/h5py/torch, no CUDA.
#
#   bash checks/dummy/check_dummy_s5_15_p1_manifest_audit.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-15 P1 manifest audit synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_s5_15_p1_manifest_audit.py"

echo "Done."
