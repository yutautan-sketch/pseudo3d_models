#!/usr/bin/env bash
set -euo pipefail

# Evaluate selected checkpoints on a fixed train-sanity subset and every file
# in the validation split recorded by train_stage5.py.

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
# Training run to evaluate
# ------------------------------------------------------------
DATE="${DATE:-260711}"
EX_DATE="${EX_DATE:-260817}"
RUN_ROOT="${RUN_ROOT:-/mnt/data/3d_projects/stage5_runs}"
PREFIX="${PREFIX:-w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8}"
DEFAULT_EXPERIMENT_NAME="pointnext_s_EX${EX_DATE}_${DATE}_${PREFIX}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${EX_DATE}/${DEFAULT_EXPERIMENT_NAME}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "${RUN_DIR}")}"

# Space-separated checkpoint filenames. Override this environment variable to
# evaluate a different set without editing the Python evaluator.
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-checkpoint_epoch_0030.pt checkpoint_epoch_0100.pt checkpoint_epoch_0150.pt best.pt}"

# ------------------------------------------------------------
# Evaluation selection and outputs
# ------------------------------------------------------------
FIXED_TRAIN_VIDEO="${FIXED_TRAIN_VIDEO:-20250626_124212_7300}"
NUM_RANDOM_TRAIN="${NUM_RANDOM_TRAIN:-2}"
SELECTION_SEED="${SELECTION_SEED:-42}"
DEVICE="${DEVICE:-cuda}"
STRICT_CHECKPOINT="${STRICT_CHECKPOINT:-1}"
WRITE_PLY="${WRITE_PLY:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
EXPORT_ANONYMIZED_METRICS="${EXPORT_ANONYMIZED_METRICS:-1}"

EVALUATION_ROOT="${EVALUATION_ROOT:-/mnt/data/3d_projects/stage5_evaluations}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVALUATION_ROOT}/${EX_DATE}/${EXPERIMENT_NAME}}"
TRAIN_LIST="${RUN_DIR}/train_files.txt"
VAL_LIST="${RUN_DIR}/val_files.txt"

# Reference PLY files exported by:
#   Stage2to4/pseudo3d/pipelines/export_stage4_bbox_ranked_pointcloud_visualizations.sh
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
STAGE4_SAMPLING_RUN="${STAGE4_SAMPLING_RUN:-global_local_l75_w31_c12_area15}"
STAGE4_TEACHER="${STAGE4_TEACHER:-bboxrank_v2_nobbox_bg}"
STAGE4_RUN_NAME="${STAGE4_RUN_NAME:-${STAGE4_SAMPLING_RUN}_${STAGE4_TEACHER}}"
STAGE4_RUN_ROOT="${STAGE4_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/${DATE}/${STAGE4_RUN_NAME}}"
RAW_REFERENCE_PLY_DIR="${RAW_REFERENCE_PLY_DIR:-${STAGE4_RUN_ROOT}/pointcloud_foreground}"
ANNOTATED_REFERENCE_PLY_DIR="${ANNOTATED_REFERENCE_PLY_DIR:-${STAGE4_RUN_ROOT}/pointcloud_annotated_foreground}"
EVALUATION_DATA_DIR="${OUTPUT_ROOT}/evaluation_data"
REFERENCE_OUTPUT_DIR="${OUTPUT_ROOT}/reference_ply"
SELECTED_TRAIN_LIST="${EVALUATION_DATA_DIR}/selected_train_files.txt"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Training run directory not found: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -s "${TRAIN_LIST}" ]]; then
  echo "Training file list not found or empty: ${TRAIN_LIST}" >&2
  exit 1
fi
if [[ ! -s "${VAL_LIST}" ]]; then
  echo "Validation file list not found or empty: ${VAL_LIST}" >&2
  exit 1
fi
if [[ ! -d "${RAW_REFERENCE_PLY_DIR}" || ! -d "${ANNOTATED_REFERENCE_PLY_DIR}" ]]; then
  echo "Stage 4 reference PLY directories are missing:" >&2
  echo "  raw       : ${RAW_REFERENCE_PLY_DIR}" >&2
  echo "  annotated : ${ANNOTATED_REFERENCE_PLY_DIR}" >&2
  echo "Run Stage2to4/pseudo3d/pipelines/export_stage4_bbox_ranked_pointcloud_visualizations.sh first." >&2
  exit 1
fi

read -r -a CHECKPOINT_ARRAY <<< "${CHECKPOINT_NAMES}"
if [[ "${#CHECKPOINT_ARRAY[@]}" -eq 0 ]]; then
  echo "CHECKPOINT_NAMES is empty" >&2
  exit 1
fi
for checkpoint_name in "${CHECKPOINT_ARRAY[@]}"; do
  if [[ ! -f "${RUN_DIR}/${checkpoint_name}" ]]; then
    echo "Checkpoint not found: ${RUN_DIR}/${checkpoint_name}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"

echo "Stage5 checkpoint evaluation pipeline"
echo "  run dir          : ${RUN_DIR}"
echo "  train list       : ${TRAIN_LIST}"
echo "  validation list  : ${VAL_LIST}"
echo "  fixed train video: ${FIXED_TRAIN_VIDEO}"
echo "  random train     : ${NUM_RANDOM_TRAIN} (seed=${SELECTION_SEED})"
echo "  checkpoints      : ${CHECKPOINT_NAMES}"
echo "  output root      : ${OUTPUT_ROOT}"
echo "  device           : ${DEVICE}"
echo "  write PLY        : ${WRITE_PLY}"
echo "  save predictions : ${SAVE_PREDICTIONS}"
echo "  raw reference PLY: ${RAW_REFERENCE_PLY_DIR}"
echo "  ann reference PLY: ${ANNOTATED_REFERENCE_PLY_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/prepare_stage5_evaluation_data.py" \
  --train_list "${TRAIN_LIST}" \
  --val_list "${VAL_LIST}" \
  --raw_reference_ply_dir "${RAW_REFERENCE_PLY_DIR}" \
  --annotated_reference_ply_dir "${ANNOTATED_REFERENCE_PLY_DIR}" \
  --output_dir "${EVALUATION_DATA_DIR}" \
  --reference_output_dir "${REFERENCE_OUTPUT_DIR}" \
  --fixed_train_video "${FIXED_TRAIN_VIDEO}" \
  --num_random_train "${NUM_RANDOM_TRAIN}" \
  --selection_seed "${SELECTION_SEED}"

EVALUATION_DIRS=()
for checkpoint_name in "${CHECKPOINT_ARRAY[@]}"; do
  checkpoint="${RUN_DIR}/${checkpoint_name}"
  checkpoint_stem="${checkpoint_name%.pt}"
  output_dir="${OUTPUT_ROOT}/${checkpoint_stem}"

  echo
  echo "Evaluating ${checkpoint_name}"
  cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/evaluate_stage5.py"
    --checkpoint "${checkpoint}"
    --train_list "${TRAIN_LIST}"
    --val_list "${VAL_LIST}"
    --selected_train_list "${SELECTED_TRAIN_LIST}"
    --output_dir "${output_dir}"
    --fixed_train_video "${FIXED_TRAIN_VIDEO}"
    --num_random_train "${NUM_RANDOM_TRAIN}"
    --selection_seed "${SELECTION_SEED}"
    --device "${DEVICE}"
  )
  if [[ "${STRICT_CHECKPOINT}" == "1" ]]; then
    cmd+=(--strict_checkpoint)
  fi
  if [[ "${WRITE_PLY}" != "1" ]]; then
    cmd+=(--no_write_ply)
  fi
  if [[ "${SAVE_PREDICTIONS}" != "1" ]]; then
    cmd+=(--no_save_predictions)
  fi
  "${cmd[@]}"
  EVALUATION_DIRS+=("${output_dir}")
done

summary_cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/summarize_stage5_evaluations.py"
  --evaluation_root "${OUTPUT_ROOT}"
)
for evaluation_dir in "${EVALUATION_DIRS[@]}"; do
  summary_cmd+=(--evaluation_dir "${evaluation_dir}")
done
"${summary_cmd[@]}"

if [[ "${EXPORT_ANONYMIZED_METRICS}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/export_anonymized_stage5_metrics.py" \
    --evaluation_root "${OUTPUT_ROOT}" \
    --share_output_dir "${OUTPUT_ROOT}/anonymized_metrics_SHARE_THIS" \
    --private_output_dir "${OUTPUT_ROOT}/anonymized_metrics_private_DO_NOT_SHARE" \
    --training_run_dir "${RUN_DIR}"
fi

echo
echo "Done. Evaluation outputs: ${OUTPUT_ROOT}"
