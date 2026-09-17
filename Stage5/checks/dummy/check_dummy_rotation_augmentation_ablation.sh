#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P2 Step 4: synthetic coverage for
# check_stage5_rotation_augmentation_ablation.py -- allowlisted config parity,
# augmentation-mode checks, file-list content vs representation differences,
# trained-checkpoint hash equality NOT being required, pooled vs per-video
# statistics, TP0 counts, and the selection criteria always deferring the
# adoption verdict. Also asserts that run_stage5_s5_15_arm.sh refuses to train
# without CONFIRM_TRAINING=1 and that the aggregation checker cannot launch a
# run at all. Pure stdlib fixtures: no torch, no CUDA, no training.
#
#   bash checks/dummy/check_dummy_rotation_augmentation_ablation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-15 rotation-augmentation ablation synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_rotation_augmentation_ablation.py"

echo "Done."
