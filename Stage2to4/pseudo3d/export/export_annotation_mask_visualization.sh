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

# VOC XML layout expected by export_annotation_mask_visualization.py:
#   ${VOC_XML_ROOT}/${VIDEO}/annotations/${VIDEO}_${frame_number:05d}.xml
VOC_XML_ROOT="${INPUT}/predirectory_for_coco_annotation_leg/260101_検証用/260101_val/"

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

PSEUDO3D_H5="${H5_DIR}/${VIDEO}${OUTPUT_SUFFIX}.h5"
POINT_CLOUD_DIR="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}_corr"
POINT_CLOUD_H5="${POINT_CLOUD_DIR}/${VIDEO}_pointcloud_${MODE}.h5"
OUTPUT_OBJ="${POINT_CLOUD_DIR}/${VIDEO}_annotation_mask_debug_${MODE}.obj"

mkdir -p "${POINT_CLOUD_DIR}"

python pseudo3d/export/export_annotation_mask_visualization.py \
  --pseudo3d_h5 "${PSEUDO3D_H5}" \
  --point_cloud_h5 "${POINT_CLOUD_H5}" \
  --voc_xml_root "${VOC_XML_ROOT}" \
  --video_name "${VIDEO}" \
  --output_obj "${OUTPUT_OBJ}" \
  --geometry_key corr \
  --texture_dir_name "annotation_mask_textures_${MODE}" \
  --xml_frame_id_source frame_index \
  --xml_frame_number_offsets 1 \
  --contour_preset percentile90_area100 \
  --contour_percentile 90 \
  --contour_min_component_area 100 \
  --contour_min_area 20 \
  --contour_min_alpha 1 \
  --contour_open_ksize 3 \
  --contour_close_ksize 5 \
  --enable_contour_fallback \
  --fallback_percentiles 85,80,75,70 \
  --fallback_min_positive_points 5 \
  --mask_color 255,32,32 \
  --mask_alpha 0.55 \
  --bbox_color 255,220,0 \
  --contour_color 0,255,255 \
  --fallback_contour_color 0,255,96 \
  --bbox_thickness 2 \
  --contour_thickness 2
