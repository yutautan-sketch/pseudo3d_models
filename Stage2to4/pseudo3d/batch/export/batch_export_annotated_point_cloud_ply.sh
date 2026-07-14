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
SAMPLING_MODE="legacy"  # legacy / combined_v2
COLOR_MODE="annotation_source"  # annotation / source / annotation_source

if [[ "${SAMPLING_MODE}" == "legacy" ]]; then
  POINT_CLOUD_TAG="${MODE}"
else
  POINT_CLOUD_TAG="${MODE}_${SAMPLING_MODE}"
fi

# annotated h5 files are expected as:
#   .../${VIDEO}_pointcloud_annotated_${POINT_CLOUD_TAG}.h5
H5_PATTERN="*_pointcloud_annotated_${POINT_CLOUD_TAG}.h5"
SUMMARY_CSV="${VIS_DIR}/batch_annotated_point_cloud_ply_summary_${POINT_CLOUD_TAG}.csv"

mkdir -p "${VIS_DIR}"

python pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.py \
  --annotated_dir "${VIS_DIR}" \
  --pattern "${H5_PATTERN}" \
  --recursive \
  --summary_csv "${SUMMARY_CSV}" \
  --background_mode dim \
  --color_mode "${COLOR_MODE}" \
  --femur_color 255,64,32 \
  --global_color 255,255,255 \
  --local_percentile_color 64,200,255 \
  --tophat_color 255,200,64 \
  --context_grid_color 120,120,120 \
  --background_alpha 85 \
  --ignore_alpha 35 \
  --include_annotation_markers \
  --annotation_marker_color 0,255,160 \
  --annotation_marker_size 3.0 \
  --annotation_marker_stride 8 \
  --max_annotation_markers 5000 \
  --skip_existing \
  --continue_on_error
