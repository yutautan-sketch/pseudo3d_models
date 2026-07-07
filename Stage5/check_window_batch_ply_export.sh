#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Export Stage 5 frame-window DataLoader batches as individual PLY files.
#
# This is a debug/visualization script for checking:
#   - frame_order based overlap windows
#   - padding-aware DataLoader batching
#   - per-window point cloud geometry in Blender / CloudCompare / MeshLab
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run from Stage5 repository root
# ------------------------------------------------------------
cd /mnt/data/3d_projects/models/Stage5

# ------------------------------------------------------------
# input path info
# ------------------------------------------------------------
DATASET_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"
DATE="260623"
INPUT_DIR="${DATASET_ROOT}/${DATE}"

MODE="grid"

# Leave VIDEO empty to use the first matching H5 in INPUT_DIR.
# Example:
#   VIDEO="20250627_104104_9320"
VIDEO=""

if [[ -n "${VIDEO}" ]]; then
  INPUT_H5="${INPUT_DIR}/${VIDEO}_pointcloud_annotated_${MODE}.h5"
else
  INPUT_H5="$(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*_annotated_${MODE}.h5" | sort | head -n 1)"
fi

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/stage5_debug/window_batch_ply"
RUN_NAME="window${DATE}_w12_s6_raw_label"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# ------------------------------------------------------------
# window / batch settings
# ------------------------------------------------------------
WINDOW_SIZE_FRAMES=12
WINDOW_STRIDE_FRAMES=6
BATCH_SIZE=4
MAX_BATCHES=3

# label / frame_order
COLOR_BY="label"

# 0: export raw pseudo-3D coordinates
# 1: export normalized model-input coordinates
NORMALIZE_POINTS=0

# ------------------------------------------------------------
# validation
# ------------------------------------------------------------
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  echo "Run Stage2to4/scripts/utils/collect_annotated_pseudo3d_h5.sh first." >&2
  exit 1
fi

if [[ -z "${INPUT_H5}" || ! -f "${INPUT_H5}" ]]; then
  echo "Input H5 not found: ${INPUT_H5}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Stage5 window batch PLY export"
echo "  input h5     : ${INPUT_H5}"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  window size  : ${WINDOW_SIZE_FRAMES}"
echo "  window stride: ${WINDOW_STRIDE_FRAMES}"
echo "  batch size   : ${BATCH_SIZE}"
echo "  max batches  : ${MAX_BATCHES}"
echo "  color by     : ${COLOR_BY}"
echo "  normalize    : ${NORMALIZE_POINTS}"

cmd=(
  python check_window_batch_ply_export.py
  --input_h5 "${INPUT_H5}"
  --output_dir "${OUTPUT_DIR}"
  --window_size_frames "${WINDOW_SIZE_FRAMES}"
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}"
  --batch_size "${BATCH_SIZE}"
  --max_batches "${MAX_BATCHES}"
  --color_by "${COLOR_BY}"
)

if [[ "${NORMALIZE_POINTS}" == "1" ]]; then
  cmd+=(--normalize_points)
fi

"${cmd[@]}"

echo "Done."
echo "  PLY dir : ${OUTPUT_DIR}"
echo "  CSV     : ${OUTPUT_DIR}/batch_window_manifest.csv"
