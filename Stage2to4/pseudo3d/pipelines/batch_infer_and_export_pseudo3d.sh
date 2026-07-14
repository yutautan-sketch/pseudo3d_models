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
SUB_DIR="predirectory_for_coco_annotation_leg/260630/260630_pseudo3d"
INPUT_DIR="${INPUT}/${SUB_DIR}"
VIDEO_PATTERN="*.mp4"

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"
RUN_NAME="260711"
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile85_area100"
OUTPUT_SUFFIX="_ts448_oym96_corr"

mkdir -p "${H5_DIR}"
mkdir -p "${VIS_DIR}"

# ------------------------------------------------------------
# Step 1: MP4 batch -> pseudo3D h5
# ------------------------------------------------------------
python pseudo3d/inference/batch_infer_video_to_pseudo3d_h5.py \
  --input_mode frame_dir \
  --input_dir "${INPUT_DIR}" \
  --pattern "${VIDEO_PATTERN}" \
  --output_dir "${H5_DIR}" \
  --output_suffix "${OUTPUT_SUFFIX}" \
  --stride 1 \
  --max_frames 128 \
  --local_preprocess resize_shorter_then_center_crop \
  --local_target_short_side 448 \
  --local_crop_offset_y -96 \
  --local_crop_offset_x 0 \
  --connect_slices \
  --tracking_correction_preset stage2_v1 \
  --tracking_progress_axis_scale 2.5 \
  --skip_existing \
  --continue_on_error

# ------------------------------------------------------------
# Step 2: h5 batch -> textured OBJ / MTL / textures
# ------------------------------------------------------------
python pseudo3d/batch/export/batch_export_pseudo3d_visualization.py \
  --input_dir "${H5_DIR}" \
  --pattern "*.h5" \
  --output_root "${VIS_DIR}" \
  --preset "${DETAIL}" \
  --geometry_key corr \
  --texture_percentile 85 \
  --texture_min_component_area 100 \
  --export_tracking_debug_obj \
  --skip_existing \
  --continue_on_error
