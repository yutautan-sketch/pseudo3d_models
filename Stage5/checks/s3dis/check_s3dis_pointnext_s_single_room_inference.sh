#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Run official S3DIS PointNeXt-S checkpoint inference on one S3DIS room.
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

DATA_ROOT="${SCRIPT_DIR}/external/PointNeXt/data/S3DIS/s3disfull"
CHECKPOINT="${SCRIPT_DIR}/external/PointNeXt/weight/s3dis-seg/checkpoint/s3dis-train-pointnext-s-ngpus1-seed1742-20220525-162639-Gz4ViiL95b6KP9MMH2ytG3_ckpt_best.pth"
OUTPUT_DIR="${SCRIPT_DIR}/work_dirs/_s3dis_pointnext_s_single_room_inference"

# Leave ROOM empty to use the smallest Area_5 room.
ROOM=""
MAX_POINTS=24000
SEED=123
DEVICE="cuda"

echo "S3DIS PointNeXt-S single-room inference"
echo "  python     : ${PYTHON}"
echo "  data root  : ${DATA_ROOT}"
echo "  checkpoint : ${CHECKPOINT}"
echo "  output dir : ${OUTPUT_DIR}"
echo "  room       : ${ROOM:-<smallest Area_5 room>}"
echo "  max points : ${MAX_POINTS}"
echo "  device     : ${DEVICE}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/checks/s3dis/check_s3dis_pointnext_s_single_room_inference.py"
  --data_root "${DATA_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --max_points "${MAX_POINTS}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ -n "${ROOM}" ]]; then
  cmd+=(--room "${ROOM}")
fi

"${cmd[@]}"

echo "Done."
