#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Diagnostic only (S5-10): recompute PointNeXt-S BatchNorm running statistics
# from the training split (momentum=None cumulative average), keeping all
# parameters frozen, then compare the recalibrated checkpoint's eval() output
# against the original checkpoint's eval() output. Does not change the
# original checkpoint or any production default.
#
# Synthetic self-test (no checkpoint/H5 required):
#   SELF_TEST=1 bash checks/real_h5/check_stage5_batchnorm_recalibration.sh
#
# Train-sanity-only smoke run (skip validation to save GPU time; calibration
# itself still runs over the full 163-file train_files.txt):
#   VAL_LIST=/tmp/stage5_empty_val_list.txt \
#     bash checks/real_h5/check_stage5_batchnorm_recalibration.sh
#
# Full run (requires GPU and CHECKPOINT/RUN_DIR/EVALUATION_DIR):
#   bash checks/real_h5/check_stage5_batchnorm_recalibration.sh
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
export PYTHONPATH="${STAGE5_DIR}/external/PointNeXt:${STAGE5_DIR}/external/PointNeXt/openpoints:${PYTHONPATH:-}"

SELF_TEST="${SELF_TEST:-0}"

if [[ "${SELF_TEST}" == "1" ]]; then
  echo "Stage5 BatchNorm recalibration self-test"
  "${PYTHON}" "${STAGE5_DIR}/checks/real_h5/check_stage5_batchnorm_recalibration.py" --self_test
  exit 0
fi

# ------------------------------------------------------------
# Run, checkpoint, and reference evaluation paths.
# S5-10: implemented after 実装事項A/B (S5-08/S5-09); best.pt only.
# ------------------------------------------------------------
RUN_DIR="${RUN_DIR:-/mnt/data/3d_projects/stage5_runs/260908/pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/best.pt}"
EVALUATION_DIR="${EVALUATION_DIR:-/mnt/data/3d_projects/stage5_evaluations/260908/pointnext_s_EX260908_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad}"

TRAIN_LIST="${TRAIN_LIST:-${RUN_DIR}/train_files.txt}"
SELECTED_TRAIN_LIST="${SELECTED_TRAIN_LIST:-${EVALUATION_DIR}/evaluation_data/selected_train_files.txt}"
VAL_LIST="${VAL_LIST:-${RUN_DIR}/val_files.txt}"
SELECTION_SUMMARY="${SELECTION_SUMMARY:-${EVALUATION_DIR}/evaluation_data/summary.json}"
REFERENCE_H5_METRICS_CSV="${REFERENCE_H5_METRICS_CSV:-${EVALUATION_DIR}/best/h5_metrics.csv}"

RUN_NAME="$(basename "${RUN_DIR}")"
CHECKPOINT_STEM="$(basename "${CHECKPOINT}" .pt)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/3d_projects/stage5_debug/batchnorm_recalibration}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_${CHECKPOINT_STEM}}"
SHARE_OUTPUT_DIR="${SHARE_OUTPUT_DIR:-${OUTPUT_DIR}/share_metrics}"
PRIVATE_OUTPUT_DIR="${PRIVATE_OUTPUT_DIR:-${OUTPUT_DIR}/private_DO_NOT_SHARE}"

DEVICE="${DEVICE:-cuda}"
STRICT_CHECKPOINT="${STRICT_CHECKPOINT:-1}"
WINDOW_SIZE_FRAMES="${WINDOW_SIZE_FRAMES:-}"
WINDOW_STRIDE_FRAMES="${WINDOW_STRIDE_FRAMES:-}"
SEED="${SEED:-42}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -s "${TRAIN_LIST}" ]]; then
  echo "Calibration train list not found or empty: ${TRAIN_LIST}" >&2
  exit 1
fi
if [[ ! -s "${SELECTED_TRAIN_LIST}" ]]; then
  echo "Selected train-sanity list not found or empty: ${SELECTED_TRAIN_LIST}" >&2
  exit 1
fi
if [[ ! -f "${VAL_LIST}" ]]; then
  echo "Validation list not found: ${VAL_LIST}" >&2
  exit 1
fi
if [[ ! -s "${VAL_LIST}" ]]; then
  echo "Validation list is empty: ${VAL_LIST} (train-sanity-only run, e.g. a smoke test)"
fi
if [[ ! -f "${SELECTION_SUMMARY}" ]]; then
  echo "Selection summary not found: ${SELECTION_SUMMARY}" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_H5_METRICS_CSV}" ]]; then
  echo "Reference h5_metrics.csv not found: ${REFERENCE_H5_METRICS_CSV}" >&2
  echo "Run evaluate_stage5.sh for this checkpoint first." >&2
  exit 1
fi

mkdir -p "${SHARE_OUTPUT_DIR}" "${PRIVATE_OUTPUT_DIR}"

echo "Stage5 BatchNorm recalibration checker"
echo "  python                  : ${PYTHON}"
echo "  checkpoint              : ${CHECKPOINT}"
echo "  calibration (train) list: ${TRAIN_LIST}"
echo "  selected train list     : ${SELECTED_TRAIN_LIST}"
echo "  validation list         : ${VAL_LIST}"
echo "  selection summary       : ${SELECTION_SUMMARY}"
echo "  reference h5_metrics.csv: ${REFERENCE_H5_METRICS_CSV}"
echo "  share output dir        : ${SHARE_OUTPUT_DIR}"
echo "  private output dir      : ${PRIVATE_OUTPUT_DIR}"
echo "  device                  : ${DEVICE}"
echo "  seed                    : ${SEED}"

cmd=(
  "${PYTHON}" "${STAGE5_DIR}/checks/real_h5/check_stage5_batchnorm_recalibration.py"
  --checkpoint "${CHECKPOINT}"
  --train_list "${TRAIN_LIST}"
  --selected_train_list "${SELECTED_TRAIN_LIST}"
  --val_list "${VAL_LIST}"
  --selection_summary "${SELECTION_SUMMARY}"
  --reference_h5_metrics_csv "${REFERENCE_H5_METRICS_CSV}"
  --share_output_dir "${SHARE_OUTPUT_DIR}"
  --private_output_dir "${PRIVATE_OUTPUT_DIR}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ "${STRICT_CHECKPOINT}" == "1" ]]; then
  cmd+=(--strict_checkpoint)
fi
if [[ -n "${WINDOW_SIZE_FRAMES}" ]]; then
  cmd+=(--window_size_frames "${WINDOW_SIZE_FRAMES}")
fi
if [[ -n "${WINDOW_STRIDE_FRAMES}" ]]; then
  cmd+=(--window_stride_frames "${WINDOW_STRIDE_FRAMES}")
fi

"${cmd[@]}"

echo "Done."
echo "  summary  : ${SHARE_OUTPUT_DIR}/batchnorm_recalibration_summary.json"
echo "  share    : ${SHARE_OUTPUT_DIR}"
echo "  private  : ${PRIVATE_OUTPUT_DIR} (includes the recalibrated diagnostic checkpoint; do not share)"
