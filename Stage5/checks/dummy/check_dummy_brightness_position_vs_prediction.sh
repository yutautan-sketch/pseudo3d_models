#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 verification (3): synthetic coverage for
# check_stage5_brightness_position_vs_prediction.py -- mid-rank percentiles,
# brightness sampled at [y, x] rather than [x, y], edge distance on non-square
# crops, a percentile of exactly 1.0 landing in the last bin, the conditional
# span really holding the other axis fixed (including a table where the
# conditional result contradicts the marginal one), sparse cells being dropped,
# empty cells staying NaN, and ignore points never entering a table.
#
# numpy only. Does not need Stage2to4, the images or CUDA.
#
#   bash checks/dummy/check_dummy_brightness_position_vs_prediction.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 brightness/position synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_brightness_position_vs_prediction.py"

echo "Done."
