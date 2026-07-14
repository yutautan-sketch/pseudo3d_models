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
VOC_XML_ROOT="${INPUT}/predirectory_for_coco_annotation_leg/260630/260630_pseudo3d/"

# VOC XML layout expected by annotate_pseudo3d_point_cloud.py:
#   ${VOC_XML_ROOT}/${VIDEO}/annotations_renamed/${VIDEO}_${frame_number:05d}.xml
#
# frame_number is usually 1-based original video frame index, so this batch
# command uses:
#   --xml_frame_id_source frame_index
#   --xml_frame_number_offsets 1

# ------------------------------------------------------------
# pseudo3D / point-cloud path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"
RUN_NAME="260711"
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile85_area100"
GEOMETRY_KEY="corr"
OUTPUT_SUFFIX="_ts448_oym96_corr"
MODE="foreground"  # grid / foreground / dense
H5_PATTERN="*${OUTPUT_SUFFIX}.h5"

SUMMARY_CSV="${VIS_DIR}/batch_pointcloud_annotation_summary_${MODE}.csv"

mkdir -p "${VIS_DIR}"

python pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py \
  --pseudo3d_dir "${H5_DIR}" \
  --pattern "${H5_PATTERN}" \
  --point_cloud_root "${VIS_DIR}" \
  --voc_xml_root "${VOC_XML_ROOT}" \
  --summary_csv "${SUMMARY_CSV}" \
  --video_suffix_to_strip "${OUTPUT_SUFFIX}" \
  --point_cloud_detail "${DETAIL}" \
  --point_cloud_filename_template "{video_name}_pointcloud_${MODE}.h5" \
  --output_filename_template "{video_name}_pointcloud_annotated_${MODE}.h5" \
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
