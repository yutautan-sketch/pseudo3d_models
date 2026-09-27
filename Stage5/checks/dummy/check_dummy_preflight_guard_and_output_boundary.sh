#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-16 Step 0: the two train_stage5.sh properties that live in bash -- the
# seal guard being invoked before the teacher preflight (which opens every
# listed H5, so a guard after it would read sealed videos to decide whether it
# may read them), and the QC totals staying out of shared output on success as
# well as on failure (the 180-file constants are already in git, so publishing
# the 162-file numbers would give the sealed videos' totals by subtraction).
# Also that the emit mode for new expectations is separate from verification,
# and that the list manifest is written to the private work area rather than
# /tmp. Pure stdlib; no H5 is opened and no training is started.
#
#   bash checks/dummy/check_dummy_preflight_guard_and_output_boundary.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-16 Step 0 preflight guard / output boundary test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_preflight_guard_and_output_boundary.py"

echo "Done."
