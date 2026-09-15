#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H2.5: synthetic pass/fail coverage for
# check_stage5_xy_coordinate_provenance.py (shape-string parsing,
# intermediate-H5 dimension reading, source resolution, dimension-decision
# priority order, normalization fail-fast bounds checks). No CUDA needed.
#
#   bash checks/dummy/check_dummy_xy_coordinate_provenance.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-14 Step H2.5 synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_xy_coordinate_provenance.py"

echo "Done."
