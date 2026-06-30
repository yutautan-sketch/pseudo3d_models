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
VIDEO="20250627_104104_9320"

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/DualTrack_output"
RUN_NAME="stage2_v1_scale10.0"
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile90_area100"
OUTPUT_SUFFIX="_ts448_oym96_corr"
MODE="grid"  # grid / foreground / dense
SAMPLE_STRIDE="4"  # grid -> 4 / foreground -> 1

INPUT_VIDEO="${INPUT}/${SUB_DIR}/${VIDEO}.mp4"
OUTPUT_H5="${H5_DIR}/${VIDEO}${OUTPUT_SUFFIX}.h5"
OUTPUT_PCH5="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}_corr/${VIDEO}_pointcloud_${MODE}.h5"
OUTPUT_PLY="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}_corr/${VIDEO}_pointcloud_${MODE}.ply"

python pseudo3d/export/export_pseudo3d_point_cloud.py \
  --input_h5 "${OUTPUT_H5}" \
  --geometry_key corr \
  --preset "${DETAIL}" \
  --output_h5 "${OUTPUT_PCH5}" \
  --output_ply "${OUTPUT_PLY}" \
  --point_mode "${MODE}" \
  --sample_stride "${SAMPLE_STRIDE}"
