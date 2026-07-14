#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Run from DualTrack repository root
# ------------------------------------------------------------
cd /mnt/data/3d_projects/models/Stage2to4

# ------------------------------------------------------------
# input/output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"
RUN_NAME="260711"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile85_area100"
GEOMETRY_KEY="corr"
MODE="foreground"  # grid / foreground / dense

# annotated h5 files are expected as:
#   ${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}_${GEOMETRY_KEY}/${VIDEO}_pointcloud_annotated_${MODE}.h5
H5_PATTERN="*_pointcloud_annotated_${MODE}.h5"
SUMMARY_CSV="${VIS_DIR}/batch_annotated_point_cloud_ply_summary_${MODE}.csv"

mkdir -p "${VIS_DIR}"

python pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.py \
  --annotated_dir "${VIS_DIR}" \
  --pattern "${H5_PATTERN}" \
  --recursive \
  --summary_csv "${SUMMARY_CSV}" \
  --background_mode dim \
  --femur_color 255,64,32 \
  --background_alpha 85 \
  --ignore_alpha 35 \
  --include_annotation_markers \
  --annotation_marker_color 0,255,160 \
  --annotation_marker_size 3.0 \
  --annotation_marker_stride 8 \
  --max_annotation_markers 5000 \
  --skip_existing \
  --continue_on_error
