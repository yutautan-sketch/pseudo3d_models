#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H3/H3.1: synthetic pass/fail coverage for
# check_stage5_frame_xy_diagnostics.py (frame-level confusion/probability
# stats, XY centroid availability, GT-interval classification, XY grid
# binning boundaries, per-bin temporal recurrence, and a hand-built
# H5+prediction integration test). No CUDA needed.
#
#   bash checks/dummy/check_dummy_frame_xy_diagnostics.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 Step H3/H3.1 synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_frame_xy_diagnostics.py"

echo "Done."
