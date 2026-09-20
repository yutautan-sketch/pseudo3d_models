#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: synthetic coverage for check_stage5_gt_component_count.py -- single
# linkage on separated blobs, the count genuinely depending on the link
# distance, a small speck not becoming a second region, per-frame independence,
# ignore/background/invalidated points excluded, instability across radii being
# reported rather than resolved, the recall comparison disclaiming causation,
# and an end-to-end CLI run on synthetic H5s. numpy/h5py only, no CUDA.
#
#   bash checks/dummy/check_dummy_gt_component_count.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 GT region-count synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_gt_component_count.py"

echo "Done."
