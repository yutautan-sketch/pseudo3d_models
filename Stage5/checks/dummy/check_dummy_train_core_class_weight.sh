#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the train_core class weight -- the W-A
# formula surviving the move out of train_stage5.py unchanged, ignore and
# valid_mask=False points excluded, each video counted once from its original
# H5 rather than through the overlap windows, a neighbouring unlisted file not
# being counted, and empty/all-ignore/out-of-range targets stopping. Also that
# the CLI computes nothing without a confirmed target. numpy/h5py only, no
# torch, no CUDA, no training.
#
#   bash checks/dummy/check_dummy_train_core_class_weight.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-16 Step 0 train_core class weight synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_train_core_class_weight.py"

echo "Done."
