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

DATE="260711"
EX_DATE="260714"

# ------------------------------------------------------------
# input path info
# ------------------------------------------------------------
DATASET_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"

# Directory created by:
#   Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh
INPUT_DIR="${DATASET_ROOT}/${DATE}"

MODE="foreground"
H5_PATTERN="*_annotated_${MODE}.h5"

# Optional quick subset. 0 means all files.
MAX_TRAIN_FILES=0

# Optional validation split from INPUT_DIR. 0.0 means use all files for train.
VAL_FRACTION=0.1
MAX_VAL_FILES=0

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/stage5_runs/${EX_DATE}"
PREFIX="w20_s10_ce_smooth00_auto_weight_lr1e3_ep200"
EXPERIMENT_NAME="pointnext_s_EX${EX_DATE}_${DATE}_${PREFIX}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

# ------------------------------------------------------------
# Model / training parameters.
# ------------------------------------------------------------
MODEL_NAME="pointnext_s"
FEATURES="intensity,confidence"
NUM_POINTS=0
BATCH_SIZE=32
EPOCHS=200

# Conservative Stage5 fine-tuning LR. PointNeXt S3DIS uses 1e-2, but Stage5
# starts from weak labels and smaller batches, so keep 1e-3 for the smoke run.
LR=1e-3
WEIGHT_DECAY=1e-4
NUM_WORKERS=4
DEVICE="cuda"
SEED=42
GRAD_CLIP_NORM=10
SAVE_EVERY=10

# Lightweight baseline size.
# For MODEL_NAME="pointnext_s", WIDTH/EXPANSION/DROPOUT are used by the
# PointNeXt-S wrapper. DEPTH and global-context settings are ignored there.
WIDTH=32
DEPTH=6
EXPANSION=4
DROPOUT=0.0

# Optional Stage5 checkpoint initialization.
# Leave empty to train from scratch.
INIT_CHECKPOINT="/mnt/data/3d_projects/models/Stage5/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init.pt"
STRICT_CHECKPOINT=1

# PointNeXt-S knobs. Ignored by mlp_baseline.
POINTNEXT_RADIUS=0.1
POINTNEXT_NSAMPLE=32
POINTNEXT_SA_LAYERS=2
POINTNEXT_SA_USE_RES=1

# Sampling knobs.
# In WINDOW_MODE="overlap", each frame-order window is one training sample and
# NUM_POINTS/SAMPLING_MODE are kept only for legacy non-window runs.
WINDOW_MODE="overlap"
WINDOW_SIZE_FRAMES=20
WINDOW_STRIDE_FRAMES=10
INCLUDE_TAIL_WINDOW=1
SAMPLING_MODE="video"
FRAME_WINDOW_SIZE=""
POSITIVE_OVERSAMPLE_RATIO=0.0
EXCLUDE_IGNORE_IN_SAMPLING=0

# Loss knobs.
# CLASS_WEIGHT:
#   ""       : no class weight
#   "auto"   : PointNeXt-style 1 / (class_frequency + epsilon), normalized to mean 1
#   "1,4"    : manual class weights
CLASS_WEIGHT="auto"
AUTO_CLASS_WEIGHT_EPSILON=0.02
NORMALIZE_AUTO_CLASS_WEIGHT=1
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
echo "  lr/weight decay: ${LR}/${WEIGHT_DECAY}"
echo "  grad clip norm : ${GRAD_CLIP_NORM:-none}"
echo "  save every     : ${SAVE_EVERY}"
echo "  device         : ${DEVICE}"
echo "  val fraction   : ${VAL_FRACTION}"
echo "  class weight   : ${CLASS_WEIGHT:-none}"
echo "  label smoothing: ${LABEL_SMOOTHING}"
echo "  init checkpoint: ${INIT_CHECKPOINT:-none}"
if [[ "${CLASS_WEIGHT}" == "auto" || "${CLASS_WEIGHT}" == "pointnext_auto" ]]; then
  echo "  auto cls weight: epsilon=${AUTO_CLASS_WEIGHT_EPSILON}, normalize=${NORMALIZE_AUTO_CLASS_WEIGHT}"
fi
if [[ "${MODEL_NAME}" == "pointnext_s" ]]; then
  echo "  pointnext      : radius=${POINTNEXT_RADIUS}, nsample=${POINTNEXT_NSAMPLE}, sa_layers=${POINTNEXT_SA_LAYERS}, sa_use_res=${POINTNEXT_SA_USE_RES}"
fi

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
  --pointnext_radius "${POINTNEXT_RADIUS}"
  --pointnext_nsample "${POINTNEXT_NSAMPLE}"
  --pointnext_sa_layers "${POINTNEXT_SA_LAYERS}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --save_every "${SAVE_EVERY}"
  --sampling_mode "${SAMPLING_MODE}"
  --window_mode "${WINDOW_MODE}"
  --window_size_frames "${WINDOW_SIZE_FRAMES}"
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}"
  --positive_oversample_ratio "${POSITIVE_OVERSAMPLE_RATIO}"
  --val_fraction "${VAL_FRACTION}"
  --max_train_files "${MAX_TRAIN_FILES}"
  --max_val_files "${MAX_VAL_FILES}"
  --auto_class_weight_epsilon "${AUTO_CLASS_WEIGHT_EPSILON}"
  --label_smoothing "${LABEL_SMOOTHING}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
    echo "Initial checkpoint not found: ${INIT_CHECKPOINT}" >&2
    exit 1
  fi
  cmd+=(--checkpoint "${INIT_CHECKPOINT}")
fi

if [[ "${STRICT_CHECKPOINT}" == "1" ]]; then
  cmd+=(--strict_checkpoint)
fi

if [[ -n "${FRAME_WINDOW_SIZE}" ]]; then
  cmd+=(--frame_window_size "${FRAME_WINDOW_SIZE}")
fi

if [[ "${EXCLUDE_IGNORE_IN_SAMPLING}" == "1" ]]; then
  cmd+=(--exclude_ignore_in_sampling)
fi

if [[ "${INCLUDE_TAIL_WINDOW}" != "1" ]]; then
  cmd+=(--no_include_tail_window)
fi

if [[ "${POINTNEXT_SA_USE_RES}" != "1" ]]; then
  cmd+=(--no_pointnext_sa_use_res)
fi

if [[ -n "${CLASS_WEIGHT}" ]]; then
  cmd+=(--class_weight "${CLASS_WEIGHT}")
fi

if [[ "${NORMALIZE_AUTO_CLASS_WEIGHT}" != "1" ]]; then
  cmd+=(--no_normalize_auto_class_weight)
fi

"${cmd[@]}"

echo "Done."
echo "  best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "  last checkpoint: ${OUTPUT_DIR}/last.pt"
