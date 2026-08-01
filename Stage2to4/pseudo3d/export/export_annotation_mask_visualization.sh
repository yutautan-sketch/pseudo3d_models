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
VIDEO="20250624_144848_7340_01"

# VOC XML layout expected by export_annotation_mask_visualization.py:
#   ${VOC_XML_ROOT}/${VIDEO}/annotations/${VIDEO}_${frame_number:05d}.xml
VOC_XML_ROOT="${INPUT}/predirectory_for_coco_annotation_leg/260630/260630_pseudo3d/"

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"
RUN_NAME="260711"
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile85_area100"
OUTPUT_SUFFIX="_ts448_oym96_corr"
MODE="foreground"  # grid / foreground / dense
SAMPLING_MODE="legacy"  # legacy / combined_v2
DRAW_POINT_CLOUD_SAMPLES="1"  # 1 / 0

if [[ "${SAMPLING_MODE}" == "legacy" ]]; then
  POINT_CLOUD_TAG="${MODE}"
else
  POINT_CLOUD_TAG="${MODE}_${SAMPLING_MODE}"
fi

PSEUDO3D_H5="${H5_DIR}/${VIDEO}${OUTPUT_SUFFIX}.h5"
POINT_CLOUD_DIR="${VIS_DIR}/${VIDEO}${OUTPUT_SUFFIX}/${DETAIL}_corr"
POINT_CLOUD_H5="${POINT_CLOUD_DIR}/${VIDEO}_pointcloud_${POINT_CLOUD_TAG}.h5"
OUTPUT_OBJ="${POINT_CLOUD_DIR}/${VIDEO}_annotation_mask_debug_${POINT_CLOUD_TAG}.obj"

mkdir -p "${POINT_CLOUD_DIR}"

python pseudo3d/export/export_annotation_mask_visualization.py \
  --pseudo3d_h5 "${PSEUDO3D_H5}" \
  --point_cloud_h5 "${POINT_CLOUD_H5}" \
  --voc_xml_root "${VOC_XML_ROOT}" \
  --video_name "${VIDEO}" \
  --output_obj "${OUTPUT_OBJ}" \
  --geometry_key corr \
  --texture_dir_name "annotation_mask_textures_${POINT_CLOUD_TAG}" \
  --xml_frame_id_source frame_index \
  --xml_frame_number_offsets 1 \
  --xml_annotation_dir_name annotations_renamed \
  --strict_xml_annotation_dir \
  --contour_preset "${DETAIL}" \
  --contour_percentile 85 \
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
  --contour_thickness 2 \
  $(if [[ "${DRAW_POINT_CLOUD_SAMPLES}" == "1" ]]; then echo "--draw_point_cloud_samples"; fi)
