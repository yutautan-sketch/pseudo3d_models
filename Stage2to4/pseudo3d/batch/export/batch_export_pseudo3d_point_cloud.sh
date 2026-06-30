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
RUN_NAME="stage2_v1_scale10.0"
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile90_area100"
GEOMETRY_KEY="corr"
OUTPUT_SUFFIX="_ts448_oym96_corr"
MODE="grid"  # grid / foreground / dense
SAMPLE_STRIDE="4"  # grid -> e.g. 2/4, foreground -> usually 1, dense -> ignored
H5_PATTERN="*${OUTPUT_SUFFIX}.h5"

SUMMARY_CSV="${VIS_DIR}/batch_pointcloud_export_summary_${MODE}.csv"

mkdir -p "${VIS_DIR}"

python pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py \
  --input_dir "${H5_DIR}" \
  --pattern "${H5_PATTERN}" \
  --output_root "${VIS_DIR}" \
  --video_suffix_to_strip "${OUTPUT_SUFFIX}" \
  --preset "${DETAIL}" \
  --geometry_key "${GEOMETRY_KEY}" \
  --point_mode "${MODE}" \
  --sample_stride "${SAMPLE_STRIDE}" \
  --texture_percentile 90 \
  --texture_min_component_area 100 \
  --summary_csv "${SUMMARY_CSV}" \
  --skip_existing \
  --continue_on_error
