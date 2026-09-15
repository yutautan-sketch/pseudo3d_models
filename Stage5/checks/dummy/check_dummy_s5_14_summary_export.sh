#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H5: synthetic pass/fail coverage for
# check_stage5_s5_14_summary_export.py (frame-position-decile aggregation,
# video_summary merge, vote-count-stratified error-rate aggregation, bundle
# privacy self-check). No CUDA needed.
#
#   bash checks/dummy/check_dummy_s5_14_summary_export.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 Step H5 synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_s5_14_summary_export.py"

echo "Done."
