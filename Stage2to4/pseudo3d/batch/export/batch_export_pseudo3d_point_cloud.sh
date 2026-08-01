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
H5_DIR="${OUTPUT_ROOT}/pseudo3d_outputs/${RUN_NAME}"
VIS_DIR="${OUTPUT_ROOT}/pseudo3d_visualizations/${RUN_NAME}"

DETAIL="percentile85_area100"
GEOMETRY_KEY="corr"
OUTPUT_SUFFIX="_ts448_oym96_corr"
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
EVIDENCE_OPEN_KSIZE="3"
EVIDENCE_CLOSE_KSIZE="5"
EVIDENCE_MORPH_SHAPE="ellipse"  # rect / ellipse / cross
EVIDENCE_MIN_COMPONENT_AREA="100"
H5_PATTERN="*${OUTPUT_SUFFIX}.h5"

if [[ "${SAMPLING_MODE}" == "legacy" ]]; then
  POINT_CLOUD_TAG="${MODE}"
else
  POINT_CLOUD_TAG="${MODE}_${SAMPLING_MODE}"
fi

SUMMARY_CSV="${VIS_DIR}/batch_pointcloud_export_summary_${POINT_CLOUD_TAG}.csv"

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
  --enable_evidence_cleanup \
  --evidence_open_ksize "${EVIDENCE_OPEN_KSIZE}" \
  --evidence_close_ksize "${EVIDENCE_CLOSE_KSIZE}" \
  --evidence_morph_shape "${EVIDENCE_MORPH_SHAPE}" \
  --evidence_min_component_area "${EVIDENCE_MIN_COMPONENT_AREA}" \
  --texture_percentile 85 \
  --texture_min_component_area 100 \
  --output_h5_filename_template "{video_name}_pointcloud_${POINT_CLOUD_TAG}.h5" \
  --output_ply_filename_template "{video_name}_pointcloud_${POINT_CLOUD_TAG}.ply" \
  --summary_csv "${SUMMARY_CSV}" \
  --skip_existing \
  --continue_on_error
