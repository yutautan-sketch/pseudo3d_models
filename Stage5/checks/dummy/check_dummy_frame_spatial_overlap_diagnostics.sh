#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H1: synthetic pass/fail coverage for
# check_stage5_frame_spatial_overlap_diagnostics.py's shared primitives, plus
# a hand-built H5 integration test for audit_h5() (Step H2). No CUDA needed.
#
#   bash checks/dummy/check_dummy_frame_spatial_overlap_diagnostics.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 Step H1 synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_frame_spatial_overlap_diagnostics.py"

echo "Done."
