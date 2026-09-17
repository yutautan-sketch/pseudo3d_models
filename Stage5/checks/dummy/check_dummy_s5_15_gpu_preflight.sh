#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: CPU-side coverage for the limited GPU preflight
# (check_stage5_s5_15_gpu_preflight.py). Exercises every stop-condition rule
# against constructed records -- two angles in one epoch, an angle frozen
# across epochs, a missing angle while enabled, an angle while disabled, an
# augmented validation sample, NaN/Inf, a broken point correspondence and a
# none-mode regression -- then runs stage C's whole data path for real on
# synthetic H5 fixtures with --skip_optimization. Stage B and stage C's model
# step need the GPU host and are not covered here.
#
#   bash checks/dummy/check_dummy_s5_15_gpu_preflight.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-15 GPU preflight CPU-side test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_s5_15_gpu_preflight.py"

echo "Done."
