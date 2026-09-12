#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Measure PointNeXt-S sensitivity to variable-length zero padding.
#
# Quick eval-only smoke test:
#   SPLITS=train TARGET_MODES=min TRAIN_MODE_SPLIT=none \
#     bash checks/real_h5/check_stage5_padding_parity.sh
#
# Default run:
#   - train + validation targets
#   - single, duplicate, real-peer, reversed-order, manual-padding eval
#   - train-mode manual padding with loss, gradients, and BatchNorm buffers
# ------------------------------------------------------------

STAGE5_DIR="${STAGE5_DIR:-/mnt/data/3d_projects/models/Stage5}"
cd "${STAGE5_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

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
export PYTHONPATH="${STAGE5_DIR}/external/PointNeXt:${STAGE5_DIR}/external/PointNeXt/openpoints:${STAGE5_DIR}:${PYTHONPATH:-}"

# ------------------------------------------------------------
# Run, checkpoint, and output paths
# Keep the historical padded teacher-v2 run as the default so the report's
# padding diagnosis remains reproducible. Override RUN_DIR for another run.
# ------------------------------------------------------------
RUN_DIR="${RUN_DIR:-/mnt/data/3d_projects/stage5_runs/260801/pointnext_s_EX260801_260711_w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/best.pt}"
RUN_NAME="$(basename "${RUN_DIR}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/3d_projects/stage5_debug/padding_parity}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_$(basename "${CHECKPOINT}" .pt)}"

# ------------------------------------------------------------
# Check settings
# TARGET_MODES: min, median, max, positive_short, negative_only
# ------------------------------------------------------------
SPLITS="${SPLITS:-train,val}"
TARGET_MODES="${TARGET_MODES:-min,positive_short}"
TRAIN_MODE_SPLIT="${TRAIN_MODE_SPLIT:-train}"
TRAIN_REPEAT_COUNT="${TRAIN_REPEAT_COUNT:-2}"
PROBABILITY_TOLERANCE="${PROBABILITY_TOLERANCE:-1e-5}"
GRADIENT_TOLERANCE="${GRADIENT_TOLERANCE:-1e-5}"
DEVICE="${DEVICE:-cuda}"
SEED_OVERRIDE="${SEED_OVERRIDE:-}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/config.json" ]]; then
  echo "Run config not found: ${RUN_DIR}/config.json" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ "${SPLITS}" == *"train"* && ! -f "${RUN_DIR}/train_files.txt" ]]; then
  echo "Train split list not found: ${RUN_DIR}/train_files.txt" >&2
  exit 1
fi
if [[ "${SPLITS}" == *"val"* && ! -f "${RUN_DIR}/val_files.txt" ]]; then
  echo "Validation split list not found: ${RUN_DIR}/val_files.txt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Stage5 PointNeXt-S padding parity check"
echo "  python               : ${PYTHON}"
echo "  run dir              : ${RUN_DIR}"
echo "  checkpoint           : ${CHECKPOINT}"
echo "  output dir           : ${OUTPUT_DIR}"
echo "  splits               : ${SPLITS}"
echo "  target modes         : ${TARGET_MODES}"
echo "  train-mode split     : ${TRAIN_MODE_SPLIT}"
echo "  train repeat count   : ${TRAIN_REPEAT_COUNT}"
echo "  probability tolerance: ${PROBABILITY_TOLERANCE}"
echo "  gradient tolerance   : ${GRADIENT_TOLERANCE}"
echo "  device               : ${DEVICE}"

cmd=(
  "${PYTHON}" "${STAGE5_DIR}/checks/real_h5/check_stage5_padding_parity.py"
  --run_dir "${RUN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --splits "${SPLITS}"
  --target_modes "${TARGET_MODES}"
  --train_mode_split "${TRAIN_MODE_SPLIT}"
  --train_repeat_count "${TRAIN_REPEAT_COUNT}"
  --probability_tolerance "${PROBABILITY_TOLERANCE}"
  --gradient_tolerance "${GRADIENT_TOLERANCE}"
  --device "${DEVICE}"
)

if [[ -n "${SEED_OVERRIDE}" ]]; then
  cmd+=(--seed "${SEED_OVERRIDE}")
fi

"${cmd[@]}"

echo "Done."
echo "  summary      : ${OUTPUT_DIR}/padding_parity_summary.json"
echo "  targets      : ${OUTPUT_DIR}/padding_parity_targets.csv"
echo "  eval metrics : ${OUTPUT_DIR}/eval_padding_parity.csv"
if [[ "${TRAIN_MODE_SPLIT}" != "none" ]]; then
  echo "  train metrics: ${OUTPUT_DIR}/train_padding_parity.csv"
fi
