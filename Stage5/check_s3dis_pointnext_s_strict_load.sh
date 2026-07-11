#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Strict-load the official S3DIS PointNeXt-S segmentation checkpoint.
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
BATCH_SIZE=1
NUM_POINTS=1024
DEVICE="cuda"
SKIP_FORWARD=0

echo "S3DIS PointNeXt-S strict-load check"
echo "  python     : ${PYTHON}"
echo "  checkpoint : ${CHECKPOINT}"
echo "  torch lib  : ${TORCH_LIB}"
echo "  batch/pts  : ${BATCH_SIZE}/${NUM_POINTS}"
echo "  device     : ${DEVICE}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/check_s3dis_pointnext_s_strict_load.py"
  --checkpoint "${CHECKPOINT}"
  --batch_size "${BATCH_SIZE}"
  --num_points "${NUM_POINTS}"
  --device "${DEVICE}"
)

if [[ "${SKIP_FORWARD}" == "1" ]]; then
  cmd+=(--skip_forward)
fi

"${cmd[@]}"

echo "Done."
