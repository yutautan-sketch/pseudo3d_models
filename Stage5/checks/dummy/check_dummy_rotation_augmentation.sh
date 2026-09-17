#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P2 Step 1: synthetic coverage for
# stage5/utils/rotation_augmentation.py -- rotation-matrix convention
# (cross-checked against the S5-14 supplement-2 diagnosis), distance/z
# preservation, input immutability, deterministic per-(video, epoch) angle
# derivation including PYTHONHASHSEED independence and global-RNG isolation,
# the none-mode no-op path, and seed resolution/config recording.
# Pure numpy: no torch, no CUDA, no H5.
#
#   bash checks/dummy/check_dummy_rotation_augmentation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-15 rotation augmentation synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_rotation_augmentation.py"

echo "Done."
