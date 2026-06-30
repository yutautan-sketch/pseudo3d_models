#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Run Stage 5 inference on collected annotated pseudo-3D point-cloud H5 files.
#
# This script expects annotated point-cloud H5 files already collected into one
# flat directory by:
#
#   Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh
#
# Edit the path/parameter blocks below, then run:
#
#   bash /mnt/data/3d_projects/models/Stage5/infer_stage5.sh
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run from Stage5 repository root
# ------------------------------------------------------------
SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="python"

DATE="260623"

# ------------------------------------------------------------
# input path info
# ------------------------------------------------------------
DATASET_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"

# Directory created by:
#   Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh
INPUT_DIR="${DATASET_ROOT}/${DATE}"

MODE="grid"
H5_PATTERN="*_annotated_${MODE}.h5"

# Optional single-video/file filter. Leave empty to process all matched H5s.
# Example:
#   VIDEO="20250627_104104_9320"
VIDEO="20250625_162948_0550"

# Optional quick subset. 0 means all files.
MAX_FILES=0

# ------------------------------------------------------------
# checkpoint path info
# ------------------------------------------------------------
RUN_ROOT="/mnt/data/3d_projects/stage5_runs"
EXPERIMENT_NAME="mlp_baseline_${DATE}_smoke"
CHECKPOINT_NAME="best.pt"
CHECKPOINT="${RUN_ROOT}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
PRED_ROOT="/mnt/data/3d_projects/stage5_predictions/${EXPERIMENT_NAME}"

# ------------------------------------------------------------
# inference parameters
# ------------------------------------------------------------
MODEL_NAME="mlp_baseline"
CHUNK_SIZE=131072
DEVICE="cuda"
WRITE_PLY=1
STRICT_CHECKPOINT=1

# Leave empty to use feature_names stored in checkpoint config.
FEATURES_OVERRIDE=""

# Keep logits for pipeline debugging. Set to 0 to save only prob_femur/pred_label.
SAVE_LOGITS=1

if [[ -n "${VIDEO}" ]]; then
  INPUT_PATTERN="${VIDEO}_pointcloud_annotated_${MODE}.h5"
else
  INPUT_PATTERN="${H5_PATTERN}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  echo "Run Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh first." >&2
  exit 1
fi

mapfile -t INPUT_H5_FILES < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name "${INPUT_PATTERN}" | sort)

if [[ "${#INPUT_H5_FILES[@]}" -eq 0 ]]; then
  echo "No input H5 files matched:" >&2
  echo "  INPUT_DIR=${INPUT_DIR}" >&2
  echo "  INPUT_PATTERN=${INPUT_PATTERN}" >&2
  exit 1
fi

mkdir -p "${PRED_ROOT}"

echo "Stage5 inference"
echo "  input dir  : ${INPUT_DIR}"
echo "  pattern    : ${INPUT_PATTERN}"
echo "  checkpoint : ${CHECKPOINT}"
echo "  output root: ${PRED_ROOT}"
echo "  num inputs : ${#INPUT_H5_FILES[@]}"
echo "  max files  : ${MAX_FILES}"
echo "  device     : ${DEVICE}"
echo "  save logits: ${SAVE_LOGITS}"

num_done=0
for input_h5 in "${INPUT_H5_FILES[@]}"; do
  if [[ "${MAX_FILES}" -gt 0 && "${num_done}" -ge "${MAX_FILES}" ]]; then
    break
  fi

  input_name="$(basename "${input_h5}")"
  video_name="${input_name%_pointcloud_annotated_${MODE}.h5}"
  video_name="${video_name%_annotated_${MODE}.h5}"

  output_h5="${PRED_ROOT}/${video_name}_stage5_pred_${MODEL_NAME}.h5"
  output_ply="${PRED_ROOT}/${video_name}_stage5_pred_${MODEL_NAME}.ply"

  echo "[$((num_done + 1))/${#INPUT_H5_FILES[@]}] ${input_h5}"

  cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/infer_stage5.py"
    --input_h5 "${input_h5}"
    --checkpoint "${CHECKPOINT}"
    --output_h5 "${output_h5}"
    --chunk_size "${CHUNK_SIZE}"
    --device "${DEVICE}"
  )

  if [[ "${STRICT_CHECKPOINT}" == "1" ]]; then
    cmd+=(--strict_checkpoint)
  fi

  if [[ -n "${FEATURES_OVERRIDE}" ]]; then
    cmd+=(--features "${FEATURES_OVERRIDE}")
  fi

  if [[ "${WRITE_PLY}" == "1" ]]; then
    cmd+=(--output_ply "${output_ply}")
  fi

  if [[ "${SAVE_LOGITS}" != "1" ]]; then
    cmd+=(--no_save_logits)
  fi

  "${cmd[@]}"
  num_done=$((num_done + 1))
done

echo "Done. Processed ${num_done} file(s)."
