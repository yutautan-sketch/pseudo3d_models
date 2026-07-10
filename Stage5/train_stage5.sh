#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Short Stage 5 training run on collected annotated pseudo-3D H5 files.
#
# This script expects annotated point-cloud H5 files already collected into one
# flat directory by:
#
#   Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh
#
# Edit the path/parameter blocks below, then run:
#
#   bash /mnt/data/3d_projects/models/Stage5/train_stage5.sh
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

# Optional quick subset. 0 means all files.
MAX_TRAIN_FILES=0

# Optional validation split from INPUT_DIR. 0.0 means use all files for train.
VAL_FRACTION=0.0
MAX_VAL_FILES=0

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/stage5_runs"
EXPERIMENT_NAME="mlp_baseline_${DATE}_smoke"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

# ------------------------------------------------------------
# Model / training parameters.
# ------------------------------------------------------------
MODEL_NAME="mlp_baseline"
FEATURES="intensity,confidence"
NUM_POINTS=0
BATCH_SIZE=4
EPOCHS=10
LR=1e-3
WEIGHT_DECAY=1e-4
NUM_WORKERS=4
DEVICE="cuda"
SEED=42

# Lightweight baseline size.
WIDTH=64
DEPTH=6
EXPANSION=4
DROPOUT=0.1

# Sampling knobs.
# In WINDOW_MODE="overlap", each frame-order window is one training sample and
# NUM_POINTS/SAMPLING_MODE are kept only for legacy non-window runs.
WINDOW_MODE="overlap"
WINDOW_SIZE_FRAMES=12
WINDOW_STRIDE_FRAMES=6
INCLUDE_TAIL_WINDOW=1
SAMPLING_MODE="video"
FRAME_WINDOW_SIZE=""
POSITIVE_OVERSAMPLE_RATIO=0.0
EXCLUDE_IGNORE_IN_SAMPLING=0

# Loss knobs.
CLASS_WEIGHT=""
LABEL_SMOOTHING=0.0

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  echo "Run Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh first, or set INPUT_DIR." >&2
  exit 1
fi

num_inputs="$(find "${INPUT_DIR}" -maxdepth 1 -type f -name "${H5_PATTERN}" | wc -l | tr -d ' ')"
if [[ "${num_inputs}" -eq 0 ]]; then
  echo "No input H5 files matched:" >&2
  echo "  INPUT_DIR=${INPUT_DIR}" >&2
  echo "  H5_PATTERN=${H5_PATTERN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Stage5 short training"
echo "  input dir      : ${INPUT_DIR}"
echo "  h5 pattern     : ${H5_PATTERN}"
echo "  matched files  : ${num_inputs}"
echo "  output dir     : ${OUTPUT_DIR}"
echo "  model          : ${MODEL_NAME}"
echo "  epochs         : ${EPOCHS}"
echo "  window mode    : ${WINDOW_MODE}"
echo "  window frames  : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}, include_tail=${INCLUDE_TAIL_WINDOW}"
echo "  num points     : ${NUM_POINTS} (ignored when window mode is overlap)"
echo "  batch size     : ${BATCH_SIZE}"
echo "  device         : ${DEVICE}"
echo "  val fraction   : ${VAL_FRACTION}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/train_stage5.py"
  --train_dir "${INPUT_DIR}"
  --h5_pattern "${H5_PATTERN}"
  --output_dir "${OUTPUT_DIR}"
  --model "${MODEL_NAME}"
  --features "${FEATURES}"
  --num_points "${NUM_POINTS}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --width "${WIDTH}"
  --depth "${DEPTH}"
  --expansion "${EXPANSION}"
  --dropout "${DROPOUT}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --sampling_mode "${SAMPLING_MODE}"
  --window_mode "${WINDOW_MODE}"
  --window_size_frames "${WINDOW_SIZE_FRAMES}"
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}"
  --positive_oversample_ratio "${POSITIVE_OVERSAMPLE_RATIO}"
  --val_fraction "${VAL_FRACTION}"
  --max_train_files "${MAX_TRAIN_FILES}"
  --max_val_files "${MAX_VAL_FILES}"
  --label_smoothing "${LABEL_SMOOTHING}"
)

if [[ -n "${FRAME_WINDOW_SIZE}" ]]; then
  cmd+=(--frame_window_size "${FRAME_WINDOW_SIZE}")
fi

if [[ "${EXCLUDE_IGNORE_IN_SAMPLING}" == "1" ]]; then
  cmd+=(--exclude_ignore_in_sampling)
fi

if [[ "${INCLUDE_TAIL_WINDOW}" != "1" ]]; then
  cmd+=(--no_include_tail_window)
fi

if [[ -n "${CLASS_WEIGHT}" ]]; then
  cmd+=(--class_weight "${CLASS_WEIGHT}")
fi

"${cmd[@]}"

echo "Done."
echo "  best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "  last checkpoint: ${OUTPUT_DIR}/last.pt"
