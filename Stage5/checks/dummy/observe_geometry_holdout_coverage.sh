#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-17 S17-3: OBSERVATION (not a pass/fail test) of held-out coverage C_ho on
# synthetic shapes -- rectangle, 5:1 Gaussian ellipse, curved band, two
# regions -- at 20 / 100 / 1000 points, 50 frames x seeds 0/1/2 each, with the
# fixed held-out procedure. The table is reported to the management chat
# before real data so that the use of C_ho can be decided. numpy only.
#
#   bash checks/dummy/observe_geometry_holdout_coverage.sh
#   JSON_OUT=/path/observation.json bash checks/dummy/observe_geometry_holdout_coverage.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/observe_geometry_holdout_coverage.py" ${JSON_OUT:+--json "${JSON_OUT}"}

echo "Done."
