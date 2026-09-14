#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-12 Step E2: synthetic BBox non-contour label-policy tests. Pure numpy
# array tests plus a Dataset-level integration test (Pseudo3DPointCloudDataset
# on a small hand-built H5). No CUDA or PointNeXt/OpenPoints import is needed
# anywhere in this checker.
#
#   bash checks/dummy/check_dummy_label_policy.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_dummy_label_policy_check}"

echo "Stage5 label-policy synthetic test"
echo "  python   : ${PYTHON}"
echo "  work dir : ${WORK_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_label_policy.py" --work_dir "${WORK_DIR}"

echo "Done."
