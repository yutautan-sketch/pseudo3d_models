#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Run from DualTrack repository root
# ------------------------------------------------------------
cd /mnt/data/3d_projects/models/Stage2to4

# ------------------------------------------------------------
# input path info
# ------------------------------------------------------------
INPUT="/mnt/data/Data_hbl"
SUB_DIR="example_videos/for_anlyze/"
VIDEO="20250626_124212_7300"

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/DualTrack_output"
RUN_NAME="stage2_v1_scale10.0"
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile85_area100"
OUTPUT_SUFFIX="_ts448_oym96_corr"
MODE="foreground"  # grid / foreground / dense
SAMPLE_STRIDE="2"  # grid -> 4 / foreground -> 1
SAMPLING_MODE="legacy"  # legacy / combined_v2
CONTEXT_GRID_STRIDE="4"
CONTEXT_GRID_PHASE="origin"  # origin / centered / dual
LOCAL_WINDOW_SIZE="41"
LOCAL_PERCENTILE="80"
LOCAL_MIN_CONTRAST="4"
TOPHAT_KERNEL_SIZE="21"
TOPHAT_PERCENTILE="85"
TOPHAT_MIN_RESPONSE="2"
TOPHAT_MORPH_SHAPE="ellipse"  # rect / ellipse / cross

if [[ "${SAMPLING_MODE}" == "legacy" ]]; then
  POINT_CLOUD_TAG="${MODE}"
else
  POINT_CLOUD_TAG="${MODE}_${SAMPLING_MODE}"
fi

INPUT_VIDEO="${INPUT}/${SUB_DIR}/${VIDEO}.mp4"
OUTPUT_H5="${H5_DIR}/${VIDEO}${OUTPUT_SUFFIX}.h5"
OUTPUT_PCH5="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}/${VIDEO}_pointcloud_${POINT_CLOUD_TAG}.h5"
OUTPUT_PLY="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}/${VIDEO}_pointcloud_${POINT_CLOUD_TAG}.ply"

python pseudo3d/export/export_pseudo3d_point_cloud.py \
  --input_h5 "${OUTPUT_H5}" \
  --geometry_key corr \
  --preset "${DETAIL}" \
  --output_h5 "${OUTPUT_PCH5}" \
  --output_ply "${OUTPUT_PLY}" \
  --point_mode "${MODE}" \
  --sample_stride "${SAMPLE_STRIDE}" \
  --sampling_mode "${SAMPLING_MODE}" \
  --include_context_grid \
  --context_grid_stride "${CONTEXT_GRID_STRIDE}" \
  --context_grid_phase "${CONTEXT_GRID_PHASE}" \
  --enable_local_percentile \
  --local_window_size "${LOCAL_WINDOW_SIZE}" \
  --local_percentile "${LOCAL_PERCENTILE}" \
  --local_min_contrast "${LOCAL_MIN_CONTRAST}" \
  --enable_tophat \
  --tophat_kernel_size "${TOPHAT_KERNEL_SIZE}" \
  --tophat_percentile "${TOPHAT_PERCENTILE}" \
  --tophat_min_response "${TOPHAT_MIN_RESPONSE}" \
  --tophat_morph_shape "${TOPHAT_MORPH_SHAPE}"
