#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Inspect the downloaded S3DIS presampled dataset layout.
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"

DATA_ROOT="${SCRIPT_DIR}/external/PointNeXt/data/S3DIS/s3disfull"
AREA=5
MAX_FILES=5

echo "S3DIS data layout check"
echo "  python    : ${PYTHON}"
echo "  data root : ${DATA_ROOT}"
echo "  area      : ${AREA}"
echo "  max files : ${MAX_FILES}"

"${PYTHON}" "${SCRIPT_DIR}/checks/s3dis/check_s3dis_data_layout.py" \
  --data_root "${DATA_ROOT}" \
  --area "${AREA}" \
  --max_files "${MAX_FILES}"

echo "Done."
