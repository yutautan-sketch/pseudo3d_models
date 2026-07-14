#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Forward-check Stage 5 PointNeXt-S on real annotated H5 windows.
#
# This script checks:
#   - real H5 -> frame_order overlap window Dataset
#   - padded DataLoader batch -> PointNeXt-S wrapper
#   - S3DIS partially transferred Stage5 checkpoint load
#   - logits/loss sanity
#   - GT / prediction / femur-probability PLY export
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run from Stage5 repository root
# ------------------------------------------------------------
SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

# ------------------------------------------------------------
# Python / CUDA extension runtime paths
# ------------------------------------------------------------
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
# input path info
# ------------------------------------------------------------
# Leave VIDEO empty to use the first matching H5 in INPUT_DIR.
# Example:
#   VIDEO="20250627_104104_9320"
VIDEO="20250626_124212_7300"

DATASET_ROOT="/mnt/data/3d_projects/DualTrack_output/pseudo3d_visualizations"
DATE="stage2_v1_scale10.0/${VIDEO}_ts448_oym96_corr/percentile85_area100"
INPUT_DIR="${DATASET_ROOT}/${DATE}"

MODE="foreground"

if [[ -n "${VIDEO}" ]]; then
  INPUT_H5="${INPUT_DIR}/${VIDEO}_pointcloud_annotated_${MODE}.h5"
else
  INPUT_H5="$(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*_annotated_${MODE}.h5" | sort | head -n 1)"
fi

# ------------------------------------------------------------
# checkpoint / output path info
# ------------------------------------------------------------
CHECKPOINT="/mnt/data/3d_projects/models/Stage5/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init.pt"

OUTPUT_ROOT="/mnt/data/3d_projects/stage5_debug/pointnext_s_transfer_real_h5_forward"
RUN_NAME="${VIDEO:-first_h5}_${MODE}_w12_s6_s3dis_partial"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# ------------------------------------------------------------
# window / batch settings
# ------------------------------------------------------------
WINDOW_SIZE_FRAMES=12
WINDOW_STRIDE_FRAMES=6
BATCH_SIZE=2
MAX_BATCHES=2
MAX_EXPORT_WINDOWS=8

# ------------------------------------------------------------
# model settings
# These must match the Stage5 checkpoint configuration.
# ------------------------------------------------------------
FEATURES="intensity,confidence"
WIDTH=32
EXPANSION=4
DROPOUT=0.0
POINTNEXT_RADIUS=0.1
POINTNEXT_NSAMPLE=32
POINTNEXT_SA_LAYERS=2
POINTNEXT_SA_USE_RES=1
STRICT_CHECKPOINT=1

# 0: model receives raw pseudo-3D coordinates
# 1: model receives normalized xyz coordinates
NORMALIZE_POINTS=1

# 0: export the same coordinates as the model input
# 1: export original H5 xyz coordinates for Blender / CloudCompare inspection
EXPORT_RAW_POINTS=1

DEVICE="cuda"
NUM_WORKERS=0

# ------------------------------------------------------------
# validation
# ------------------------------------------------------------
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  echo "Run Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh first, or edit INPUT_DIR above." >&2
  exit 1
fi

if [[ -z "${INPUT_H5}" || ! -f "${INPUT_H5}" ]]; then
  echo "Input H5 not found: ${INPUT_H5}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Run check_s3dis_to_stage5_pointnext_s_transfer.sh with --save_checkpoint first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Stage5 real-H5 PointNeXt-S transfer forward check"
echo "  python          : ${PYTHON}"
echo "  conda prefix    : ${CONDA_PREFIX}"
echo "  torch lib       : ${TORCH_LIB}"
echo "  input h5        : ${INPUT_H5}"
echo "  checkpoint      : ${CHECKPOINT}"
echo "  output dir      : ${OUTPUT_DIR}"
echo "  window frames   : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}"
echo "  batch size      : ${BATCH_SIZE}"
echo "  max batches     : ${MAX_BATCHES}"
echo "  max exports     : ${MAX_EXPORT_WINDOWS}"
echo "  normalize input : ${NORMALIZE_POINTS}"
echo "  export raw xyz  : ${EXPORT_RAW_POINTS}"
echo "  model width     : ${WIDTH}"
echo "  radius/nsample  : ${POINTNEXT_RADIUS}/${POINTNEXT_NSAMPLE}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_real_h5_pointnext_s_transfer_forward.py"
  --input_h5 "${INPUT_H5}"
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --features "${FEATURES}"
  --window_size_frames "${WINDOW_SIZE_FRAMES}"
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}"
  --batch_size "${BATCH_SIZE}"
  --max_batches "${MAX_BATCHES}"
  --max_export_windows "${MAX_EXPORT_WINDOWS}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --width "${WIDTH}"
  --expansion "${EXPANSION}"
  --dropout "${DROPOUT}"
  --pointnext_radius "${POINTNEXT_RADIUS}"
  --pointnext_nsample "${POINTNEXT_NSAMPLE}"
  --pointnext_sa_layers "${POINTNEXT_SA_LAYERS}"
)

if [[ "${POINTNEXT_SA_USE_RES}" != "1" ]]; then
  cmd+=(--no_pointnext_sa_use_res)
fi

if [[ "${STRICT_CHECKPOINT}" == "1" ]]; then
  cmd+=(--strict_checkpoint)
fi

if [[ "${NORMALIZE_POINTS}" == "1" ]]; then
  cmd+=(--normalize_points)
fi

if [[ "${EXPORT_RAW_POINTS}" == "1" ]]; then
  cmd+=(--export_raw_points)
fi

"${cmd[@]}"

echo "Done."
echo "  PLY dir : ${OUTPUT_DIR}"
echo "  CSV     : ${OUTPUT_DIR}/real_h5_pointnext_s_forward_manifest.csv"
echo "  JSON    : ${OUTPUT_DIR}/real_h5_pointnext_s_forward_summary.json"
