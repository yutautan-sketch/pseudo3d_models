#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Validate Stage 5 PointNeXt-S training through train_stage5.py.
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
# Dummy data / training settings
# ------------------------------------------------------------
WORK_DIR="${SCRIPT_DIR}/_dummy_pointnext_s_training_check"
NUM_TRAIN=2
NUM_VAL=1
NUM_FRAMES=16
POINTS_PER_FRAME=64
WINDOW_SIZE_FRAMES=8
WINDOW_STRIDE_FRAMES=4
BATCH_SIZE=1
EPOCHS=1

WIDTH=32
EXPANSION=4
DROPOUT=0.0
RADIUS=0.1
NSAMPLE=16
SA_LAYERS=2
SA_USE_RES=1

LR=1e-3
WEIGHT_DECAY=0.0
SEED=123
DEVICE="cuda"

echo "Stage5 PointNeXt-S dummy training CLI check"
echo "  python          : ${PYTHON}"
echo "  conda prefix    : ${CONDA_PREFIX}"
echo "  torch lib       : ${TORCH_LIB}"
echo "  work dir        : ${WORK_DIR}"
echo "  train/val files : ${NUM_TRAIN}/${NUM_VAL}"
echo "  frames/points   : ${NUM_FRAMES}/${POINTS_PER_FRAME}"
echo "  window frames   : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}"
echo "  batch/epochs    : ${BATCH_SIZE}/${EPOCHS}"
echo "  model width     : ${WIDTH}"
echo "  radius/nsample  : ${RADIUS}/${NSAMPLE}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/check_dummy_pointnext_s_training.py"
  --work_dir "${WORK_DIR}"
  --num_train "${NUM_TRAIN}"
  --num_val "${NUM_VAL}"
  --num_frames "${NUM_FRAMES}"
  --points_per_frame "${POINTS_PER_FRAME}"
  --window_size_frames "${WINDOW_SIZE_FRAMES}"
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --width "${WIDTH}"
  --expansion "${EXPANSION}"
  --dropout "${DROPOUT}"
  --radius "${RADIUS}"
  --nsample "${NSAMPLE}"
  --sa_layers "${SA_LAYERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ "${SA_USE_RES}" != "1" ]]; then
  cmd+=(--no_sa_use_res)
fi

"${cmd[@]}"

echo "Done."
