#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-12 Step E7: common-target evaluation for one Run A/B label-policy
# checkpoint (canonical_v6, bbox_noncontour_as_background,
# bbox_noncontour_region). Run this once per checkpoint you want to compare
# (Run A epoch-5/best, Run B epoch-5/best).
#
# Prerequisite: run plain evaluate_stage5.sh for the SAME checkpoint first, so
# REFERENCE_H5_METRICS_CSV exists for the canonical_v6 parity gate.
#
#   CHECKPOINT=/path/to/run_a/last.pt \
#   REFERENCE_H5_METRICS_CSV=/path/to/run_a_eval/last/h5_metrics.csv \
#     bash checks/real_h5/check_stage5_label_policy_ablation_eval.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

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
export PYTHONPATH="${SCRIPT_DIR}/external/PointNeXt:${SCRIPT_DIR}/external/PointNeXt/openpoints:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the Run A/B checkpoint (e.g. .../last.pt or .../best.pt)}"
REFERENCE_H5_METRICS_CSV="${REFERENCE_H5_METRICS_CSV:?Set REFERENCE_H5_METRICS_CSV to the matching plain evaluate_stage5.py h5_metrics.csv}"

# Derived from the required CHECKPOINT/REFERENCE_H5_METRICS_CSV by default, so
# this always matches whichever run/checkpoint is actually being evaluated
# (Run A vs Run B, v6 vs v7, ...) instead of pointing at a stale hardcoded
# run. Override explicitly only if your layout differs from the standard
# train_stage5.sh / evaluate_stage5.sh output structure.
RUN_DIR="${RUN_DIR:-$(dirname "${CHECKPOINT}")}"
EVALUATION_DIR="${EVALUATION_DIR:-$(dirname "$(dirname "${REFERENCE_H5_METRICS_CSV}")")}"

SELECTED_TRAIN_LIST="${SELECTED_TRAIN_LIST:-${EVALUATION_DIR}/evaluation_data/selected_train_files.txt}"
VAL_LIST="${VAL_LIST:-${RUN_DIR}/val_files.txt}"
SELECTION_SUMMARY="${SELECTION_SUMMARY:-${EVALUATION_DIR}/evaluation_data/summary.json}"

CHECKPOINT_STEM="$(basename "${CHECKPOINT}" .pt)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/3d_projects/stage5_debug/label_policy_ablation}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(basename "$(dirname "${CHECKPOINT}")")_${CHECKPOINT_STEM}}"
SHARE_OUTPUT_DIR="${SHARE_OUTPUT_DIR:-${OUTPUT_DIR}/share_metrics}"
PRIVATE_OUTPUT_DIR="${PRIVATE_OUTPUT_DIR:-${OUTPUT_DIR}/private_DO_NOT_SHARE}"

DEVICE="${DEVICE:-cuda}"
STRICT_CHECKPOINT="${STRICT_CHECKPOINT:-1}"
EXPORT_PLY_ALIAS="${EXPORT_PLY_ALIAS:-}"
PLY_OUTPUT_DIR="${PLY_OUTPUT_DIR:-${PRIVATE_OUTPUT_DIR}/ply}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${REFERENCE_H5_METRICS_CSV}" ]]; then
  echo "Reference h5_metrics.csv not found: ${REFERENCE_H5_METRICS_CSV}" >&2
  echo "Run evaluate_stage5.sh for this checkpoint first." >&2
  exit 1
fi
if [[ ! -s "${SELECTED_TRAIN_LIST}" ]]; then
  echo "Selected train-sanity list not found or empty: ${SELECTED_TRAIN_LIST}" >&2
  exit 1
fi
if [[ ! -s "${VAL_LIST}" ]]; then
  echo "Validation list not found or empty: ${VAL_LIST}" >&2
  exit 1
fi

mkdir -p "${SHARE_OUTPUT_DIR}" "${PRIVATE_OUTPUT_DIR}"

echo "Stage5 label-policy ablation evaluator"
echo "  python                  : ${PYTHON}"
echo "  checkpoint              : ${CHECKPOINT}"
echo "  reference h5_metrics.csv: ${REFERENCE_H5_METRICS_CSV}"
echo "  selected train list     : ${SELECTED_TRAIN_LIST}"
echo "  validation list         : ${VAL_LIST}"
echo "  share output dir        : ${SHARE_OUTPUT_DIR}"
echo "  private output dir      : ${PRIVATE_OUTPUT_DIR}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_label_policy_ablation_eval.py"
  --checkpoint "${CHECKPOINT}"
  --selected_train_list "${SELECTED_TRAIN_LIST}"
  --val_list "${VAL_LIST}"
  --selection_summary "${SELECTION_SUMMARY}"
  --reference_h5_metrics_csv "${REFERENCE_H5_METRICS_CSV}"
  --share_output_dir "${SHARE_OUTPUT_DIR}"
  --private_output_dir "${PRIVATE_OUTPUT_DIR}"
  --device "${DEVICE}"
)
if [[ "${STRICT_CHECKPOINT}" == "1" ]]; then
  cmd+=(--strict_checkpoint)
fi
if [[ -n "${EXPORT_PLY_ALIAS}" ]]; then
  cmd+=(--ply_output_dir "${PLY_OUTPUT_DIR}")
  for alias in ${EXPORT_PLY_ALIAS}; do
    cmd+=(--export_ply_alias "${alias}")
  done
fi

"${cmd[@]}"

echo "Done."
echo "  summary  : ${SHARE_OUTPUT_DIR}/label_policy_ablation_summary.json"
echo "  share    : ${SHARE_OUTPUT_DIR}"
echo "  private  : ${PRIVATE_OUTPUT_DIR} (includes video ID map and any PLY exports; do not share)"
