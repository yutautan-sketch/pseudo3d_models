#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Run from DualTrack repository root
# ------------------------------------------------------------
cd /mnt/data/3d_projects/models/Stage2to4

# ------------------------------------------------------------
# DualTrack checkpoint path info
# ------------------------------------------------------------
export DUALTRACK_CHECKPOINT_DIR="${DUALTRACK_CHECKPOINT_DIR:-/mnt/data/3d_projects/dualtrack_checkpoints}"
export DUALTRACK_CHECKPOINT="${DUALTRACK_CHECKPOINT:-${DUALTRACK_CHECKPOINT_DIR}/dualtrack_final.pt}"

# ------------------------------------------------------------
# input path info
# ------------------------------------------------------------
INPUT="/mnt/data/Data_hbl"
SUB_DIR="example_videos/"
VIDEO="20250710_111443_594_03"

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/DualTrack_output"
RUN_NAME="stage2_v1_scale10.0"
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile85_area100"
DETAIL_SUFFIX=""
OUTPUT_SUFFIX="_ts448_oym96_corr"

INPUT_VIDEO="${INPUT}/${SUB_DIR}/${VIDEO}.mp4"
OUTPUT_H5="${H5_DIR}/${VIDEO}${OUTPUT_SUFFIX}.h5"
OUTPUT_OBJ="${VIS_DIR}/${VIDEO}/${DETAIL}${DETAIL_SUFFIX}/${VIDEO}_textured.obj"
DEBUG_OBJ="${VIS_DIR}/${VIDEO}/${DETAIL}${DETAIL_SUFFIX}/${VIDEO}_tracking_debug.obj"

mkdir -p "${H5_DIR}"
mkdir -p "${VIS_DIR}/${VIDEO}/${DETAIL}${DETAIL_SUFFIX}"

# ------------------------------------------------------------
# Step 1: MP4 -> pseudo3D h5
# ------------------------------------------------------------
python pseudo3d/inference/infer_video_to_pseudo3d_h5.py \
  --input_mode video \
  --video_path "${INPUT_VIDEO}" \
  --output_h5 "${OUTPUT_H5}" \
  --stride 1 \
  --max_frames 128 \
  --local_preprocess resize_shorter_then_center_crop \
  --local_target_short_side 448 \
  --local_crop_offset_y -96 \
  --local_crop_offset_x 0 \
  --connect_slices \
  --tracking_correction_preset stage2_v1 \
  --tracking_progress_axis_scale 10.0 

# ------------------------------------------------------------
# Step 2: h5 -> textured OBJ / MTL / textures
# ------------------------------------------------------------
python pseudo3d/export/export_pseudo3d_visualization.py \
  --input_h5 "${OUTPUT_H5}" \
  --export_textured_obj "${OUTPUT_OBJ}" \
  --export_tracking_debug_obj "${DEBUG_OBJ}" \
  --preset "${DETAIL}" \
  --geometry_key corr \
  --texture_percentile 85 \
  --texture_min_component_area 100
