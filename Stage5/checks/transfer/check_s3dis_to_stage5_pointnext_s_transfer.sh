#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Partially transfer official S3DIS PointNeXt-S weights to Stage5 pointnext_s.
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

CHECKPOINT="${SCRIPT_DIR}/external/PointNeXt/weight/s3dis-seg/checkpoint/s3dis-train-pointnext-s-ngpus1-seed1742-20220525-162639-Gz4ViiL95b6KP9MMH2ytG3_ckpt_best.pth"
WORK_DIR="${SCRIPT_DIR}/work_dirs/_s3dis_to_stage5_pointnext_s_transfer"

FEATURES="intensity,confidence"
NUM_FRAMES=20
POINTS_PER_FRAME=80
WINDOW_SIZE_FRAMES=8
WINDOW_STRIDE_FRAMES=4
BATCH_SIZE=2

WIDTH=32
EXPANSION=4
DROPOUT=0.0
RADIUS=0.1
NSAMPLE=32
SA_LAYERS=2
SA_USE_RES=1

SEED=123
DEVICE="cuda"
SAVE_CHECKPOINT=1

echo "S3DIS -> Stage5 PointNeXt-S partial transfer check"
echo "  python          : ${PYTHON}"
echo "  source ckpt     : ${CHECKPOINT}"
echo "  work dir        : ${WORK_DIR}"
echo "  features        : ${FEATURES}"
echo "  window frames   : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}"
echo "  batch size      : ${BATCH_SIZE}"
echo "  model width     : ${WIDTH}"
echo "  radius/nsample  : ${RADIUS}/${NSAMPLE}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/checks/transfer/check_s3dis_to_stage5_pointnext_s_transfer.py"
  --checkpoint "${CHECKPOINT}"
  --work_dir "${WORK_DIR}"
  --features "${FEATURES}"
  --num_frames "${NUM_FRAMES}"
  --points_per_frame "${POINTS_PER_FRAME}"
  --window_size_frames "${WINDOW_SIZE_FRAMES}"
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}"
  --batch_size "${BATCH_SIZE}"
  --width "${WIDTH}"
  --expansion "${EXPANSION}"
  --dropout "${DROPOUT}"
  --radius "${RADIUS}"
  --nsample "${NSAMPLE}"
  --sa_layers "${SA_LAYERS}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ "${SA_USE_RES}" != "1" ]]; then
  cmd+=(--no_sa_use_res)
fi

if [[ "${SAVE_CHECKPOINT}" == "1" ]]; then
  cmd+=(--save_checkpoint)
fi

"${cmd[@]}"

echo "Done."
