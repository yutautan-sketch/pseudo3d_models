#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Validate Stage 5 PointNeXt-S wrapper with a dummy frame-window batch.
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"

TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${SCRIPT_DIR}/external/PointNeXt:${SCRIPT_DIR}/external/PointNeXt/openpoints:${PYTHONPATH:-}"

# ------------------------------------------------------------
# Dummy data / model settings
# ------------------------------------------------------------
WORK_DIR="${SCRIPT_DIR}/_dummy_pointnext_s_forward_check"
NUM_FRAMES=20
POINTS_PER_FRAME=80
WINDOW_SIZE_FRAMES=8
WINDOW_STRIDE_FRAMES=4
BATCH_SIZE=2

WIDTH=32
EXPANSION=4
DROPOUT=0.0
RADIUS=0.1
NSAMPLE=16
SEED=123
DEVICE="cuda"

echo "Stage5 PointNeXt-S dummy forward check"
echo "  python          : ${PYTHON}"
echo "  conda prefix    : ${CONDA_PREFIX}"
echo "  torch lib       : ${TORCH_LIB}"
echo "  work dir        : ${WORK_DIR}"
echo "  window frames   : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}"
echo "  batch size      : ${BATCH_SIZE}"
echo "  model width     : ${WIDTH}"
echo "  radius/nsample  : ${RADIUS}/${NSAMPLE}"

"${PYTHON}" "${SCRIPT_DIR}/check_dummy_pointnext_s_forward.py" \
  --work_dir "${WORK_DIR}" \
  --num_frames "${NUM_FRAMES}" \
  --points_per_frame "${POINTS_PER_FRAME}" \
  --window_size_frames "${WINDOW_SIZE_FRAMES}" \
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}" \
  --batch_size "${BATCH_SIZE}" \
  --width "${WIDTH}" \
  --expansion "${EXPANSION}" \
  --dropout "${DROPOUT}" \
  --radius "${RADIUS}" \
  --nsample "${NSAMPLE}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "Done."
