#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-11 Step D5: transfer the existing Stage5 BatchNorm S3DIS partial-init
# checkpoint's trainable parameters into a freshly constructed GroupNorm
# PointNeXt-S model (norm affine weight/bias included; BatchNorm running
# buffers excluded since GroupNorm has none). Fails fast on any other
# missing/unexpected/shape-mismatched key.
#
# Synthetic self-test (CPU only, no CUDA/checkpoint required):
#   SELF_TEST=1 bash checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.sh
#
# Full transfer (requires CUDA + the existing BatchNorm partial-init checkpoint):
#   bash checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SCRIPT_DIR}/external/PointNeXt:${SCRIPT_DIR}/external/PointNeXt/openpoints:${PYTHONPATH:-}"

SELF_TEST="${SELF_TEST:-0}"

if [[ "${SELF_TEST}" == "1" ]]; then
  echo "Stage5 BatchNorm -> GroupNorm transfer self-test"
  "${PYTHON}" "${SCRIPT_DIR}/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py" --self_test
  exit 0
fi

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SCRIPT_DIR}/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-${SCRIPT_DIR}/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init_groupnorm.pt}"
NORM_GROUPS="${NORM_GROUPS:-8}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Source BatchNorm checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  echo "Run checks/transfer/check_s3dis_to_stage5_pointnext_s_transfer.sh --save_checkpoint first." >&2
  exit 1
fi

echo "Stage5 BatchNorm -> GroupNorm transfer check"
echo "  python             : ${PYTHON}"
echo "  source checkpoint  : ${SOURCE_CHECKPOINT}"
echo "  output checkpoint  : ${OUTPUT_CHECKPOINT}"
echo "  norm_groups        : ${NORM_GROUPS}"
echo "  device             : ${DEVICE}"

"${PYTHON}" "${SCRIPT_DIR}/checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.py" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --output_checkpoint "${OUTPUT_CHECKPOINT}" \
  --norm_groups "${NORM_GROUPS}" \
  --device "${DEVICE}"

echo "Done."
echo "  output checkpoint: ${OUTPUT_CHECKPOINT}"
echo "  report           : ${OUTPUT_CHECKPOINT%.pt}.transfer_report.json"
