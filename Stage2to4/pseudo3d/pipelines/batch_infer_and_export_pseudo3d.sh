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

# ------------------------------------------------------------
# Step 3: h5 batch -> pseudo3D point-cloud h5 / PLY
# ------------------------------------------------------------
GEOMETRY_KEY="corr"
MODE="foreground"  # grid / foreground / dense
SAMPLE_STRIDE="2"  # grid -> e.g. 2/4, foreground -> usually 1, dense -> ignored
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
H5_PATTERN="*${OUTPUT_SUFFIX}.h5"

if [[ "${SAMPLING_MODE}" == "legacy" ]]; then
  POINT_CLOUD_TAG="${MODE}"
else
  POINT_CLOUD_TAG="${MODE}_${SAMPLING_MODE}"
fi

POINT_CLOUD_EXPORT_SUMMARY_CSV="${VIS_DIR}/batch_pointcloud_export_summary_${POINT_CLOUD_TAG}.csv"

python pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py \
  --input_dir "${H5_DIR}" \
  --pattern "${H5_PATTERN}" \
  --output_root "${VIS_DIR}" \
  --video_suffix_to_strip "${OUTPUT_SUFFIX}" \
  --preset "${DETAIL}" \
  --geometry_key "${GEOMETRY_KEY}" \
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
  --tophat_morph_shape "${TOPHAT_MORPH_SHAPE}" \
  --texture_percentile 85 \
  --texture_min_component_area 100 \
  --output_h5_filename_template "{video_name}_pointcloud_${POINT_CLOUD_TAG}.h5" \
  --output_ply_filename_template "{video_name}_pointcloud_${POINT_CLOUD_TAG}.ply" \
  --summary_csv "${POINT_CLOUD_EXPORT_SUMMARY_CSV}" \
  --skip_existing \
  --continue_on_error

# ------------------------------------------------------------
# Step 4: point-cloud h5 + VOC XML -> annotated point-cloud h5
# ------------------------------------------------------------
VOC_XML_ROOT="${INPUT}/predirectory_for_coco_annotation_leg/260630/260630_pseudo3d/"
POINT_CLOUD_ANNOTATION_SUMMARY_CSV="${VIS_DIR}/batch_pointcloud_annotation_summary_${POINT_CLOUD_TAG}.csv"

python pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py \
  --pseudo3d_dir "${H5_DIR}" \
  --pattern "${H5_PATTERN}" \
  --point_cloud_root "${VIS_DIR}" \
  --voc_xml_root "${VOC_XML_ROOT}" \
  --summary_csv "${POINT_CLOUD_ANNOTATION_SUMMARY_CSV}" \
  --video_suffix_to_strip "${OUTPUT_SUFFIX}" \
  --point_cloud_detail "${DETAIL}" \
  --point_cloud_filename_template "{video_name}_pointcloud_${POINT_CLOUD_TAG}.h5" \
  --output_filename_template "{video_name}_pointcloud_annotated_${POINT_CLOUD_TAG}.h5" \
  --geometry_key "${GEOMETRY_KEY}" \
  --label_mode contour_in_bbox \
  --bbox_ignore_margin 0 \
  --xml_frame_id_source frame_index \
  --xml_frame_number_offsets 1 \
  --contour_preset percentile85_area100 \
  --contour_percentile 85 \
  --contour_min_component_area 100 \
  --contour_min_area 20 \
  --contour_min_alpha 1 \
  --contour_open_ksize 3 \
  --contour_close_ksize 5 \
  --enable_contour_fallback \
  --fallback_percentiles 85,80,75,70 \
  --fallback_min_positive_points 5 \
  --bbox_inside_non_contour_label ignore \
  --skip_existing \
  --continue_on_error

# ------------------------------------------------------------
# Step 5: annotated point-cloud h5 -> colored PLY
# ------------------------------------------------------------
COLOR_MODE="annotation_source"  # annotation / source / annotation_source
ANNOTATED_H5_PATTERN="*_pointcloud_annotated_${POINT_CLOUD_TAG}.h5"
ANNOTATED_PLY_EXPORT_SUMMARY_CSV="${VIS_DIR}/batch_annotated_point_cloud_ply_summary_${POINT_CLOUD_TAG}.csv"

python pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.py \
  --annotated_dir "${VIS_DIR}" \
  --pattern "${ANNOTATED_H5_PATTERN}" \
  --recursive \
  --summary_csv "${ANNOTATED_PLY_EXPORT_SUMMARY_CSV}" \
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
