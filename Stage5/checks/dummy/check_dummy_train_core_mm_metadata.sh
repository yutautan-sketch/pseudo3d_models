#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-16 Step 0: coverage for the mm-metadata audit's restraint -- a spacing of
# 1.0 read as the documented placeholder rather than a measured scale, x and y
# compared separately, a missing or unreadable spacing reported rather than
# defaulted, the conclusion recorded as UNCONFIRMED rather than unavailable,
# unresolvable videos reported rather than filled in from a neighbour, and no
# surrogate FL or T_FL produced anywhere. numpy/h5py, no torch, no CUDA.
#
#   bash checks/dummy/check_dummy_train_core_mm_metadata.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-16 Step 0 train_core_mm_metadata synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_train_core_mm_metadata.py"

echo "Done."
