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

# VOC XML layout expected by annotate_pseudo3d_point_cloud.py:
#   ${VOC_XML_ROOT}/${VIDEO}/annotations_renamed/${VIDEO}_${frame_number:05d}.xml
#
# frame_number is usually 1-based original video frame index, so the command
# below uses:
#   --xml_frame_id_source frame_index
#   --xml_frame_number_offsets 1
VOC_XML_ROOT="${INPUT}/predirectory_for_coco_annotation_leg/260630/260630_pseudo3d"

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
SAMPLING_MODE="legacy"  # legacy / combined_v2

if [[ "${SAMPLING_MODE}" == "legacy" ]]; then
  POINT_CLOUD_TAG="${MODE}"
else
  POINT_CLOUD_TAG="${MODE}_${SAMPLING_MODE}"
fi

PSEUDO3D_H5="${H5_DIR}/${VIDEO}${OUTPUT_SUFFIX}.h5"
POINT_CLOUD_DIR="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}"
POINT_CLOUD_H5="${POINT_CLOUD_DIR}/${VIDEO}_pointcloud_${POINT_CLOUD_TAG}.h5"
OUTPUT_ANNOTATED_H5="${POINT_CLOUD_DIR}/${VIDEO}_pointcloud_annotated_${POINT_CLOUD_TAG}.h5"

mkdir -p "${POINT_CLOUD_DIR}"

python pseudo3d/annotation/annotate_pseudo3d_point_cloud.py \
  --point_cloud_h5 "${POINT_CLOUD_H5}" \
  --pseudo3d_h5 "${PSEUDO3D_H5}" \
  --voc_xml_root "${VOC_XML_ROOT}" \
  --video_name "${VIDEO}" \
  --output_h5 "${OUTPUT_ANNOTATED_H5}" \
  --geometry_key corr \
  --label_mode contour_in_bbox \
  --bbox_ignore_margin 0 \
  --xml_frame_id_source frame_index \
  --xml_frame_number_offsets 1 \
  --xml_annotation_dir_name annotations_renamed \
  --strict_xml_annotation_dir \
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
  --bbox_inside_non_contour_label ignore
