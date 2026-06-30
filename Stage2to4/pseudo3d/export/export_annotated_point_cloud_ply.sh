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
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile90_area100"
OUTPUT_SUFFIX="_ts448_oym96_corr"
MODE="grid"  # grid / foreground / dense

POINT_CLOUD_DIR="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}_corr"
ANNOTATED_H5="${POINT_CLOUD_DIR}/${VIDEO}_pointcloud_annotated_${MODE}.h5"
OUTPUT_PLY="${POINT_CLOUD_DIR}/${VIDEO}_pointcloud_annotated_${MODE}.ply"
ANNOTATION_ONLY_OUTPUT_PLY="${POINT_CLOUD_DIR}/${VIDEO}_pointcloud_annotation_only_${MODE}.ply"

mkdir -p "${POINT_CLOUD_DIR}"

python pseudo3d/export/export_annotated_point_cloud_ply.py \
  --input_h5 "${ANNOTATED_H5}" \
  --output_ply "${OUTPUT_PLY}" \
  --annotation_only_output_ply "${ANNOTATION_ONLY_OUTPUT_PLY}" \
  --background_mode dim \
  --femur_color 255,64,32 \
  --background_alpha 90 \
  --ignore_alpha 35 \
  --include_annotation_markers \
  --annotation_marker_color 0,255,160 \
  --annotation_marker_size 3.0 \
  --annotation_marker_stride 8 \
  --max_annotation_markers 5000 \
  --point_mode "${MODE}"
