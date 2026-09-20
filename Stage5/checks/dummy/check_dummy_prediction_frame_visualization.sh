#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: synthetic coverage for the prediction frame visualization
# (export_stage5_prediction_frames.py and
# stage5/utils/prediction_frame_render.py): per-category pixel colors, frame
# dispatch, deterministic top-FP frame selection, every stop condition, and
# the guarantee that the evaluation directory being read stays unchanged.
# Synthetic H5/npz only. No CUDA, no real data, no GPU env vars.
#
#   bash checks/dummy/check_dummy_prediction_frame_visualization.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

# Stage2to4 supplies image_to_uint8_gray. The check falls back to a stand-in
# if it is missing, and prints which one it used.
STAGE2TO4_ROOT="${STAGE2TO4_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
export STAGE2TO4_ROOT

echo "Stage5 S5-15 prediction frame visualization synthetic test"
echo "  python          : ${PYTHON}"
echo "  stage2to4 root  : ${STAGE2TO4_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_prediction_frame_visualization.py"

echo "Done."
