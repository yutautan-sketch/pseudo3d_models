#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P2 Step 2: Dataset-side coverage for the training-only Z-rotation --
# none-mode bit-for-bit equivalence with the pre-augmentation path, one angle
# per (video, epoch) shared by every window, no rotation accumulation across
# epochs with cache_data=True, non-XYZ fields untouched, duplicate file-name
# rejection, and the mandatory real-DataLoader epoch/worker matrix
# (num_workers=0, non-persistent workers, persistent workers, and an explicit
# spawn context). Uses torch on CPU with synthetic H5 fixtures: no CUDA, no
# real data, no training.
#
#   bash checks/dummy/check_dummy_rotation_augmentation_dataset.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-15 rotation augmentation dataset test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_rotation_augmentation_dataset.py"

echo "Done."
