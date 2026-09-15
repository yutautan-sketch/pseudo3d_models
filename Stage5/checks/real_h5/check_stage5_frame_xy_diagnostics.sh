#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H3/H3.1: frame-level and normalized-XY-density/temporal-
# recurrence diagnostics for one class-weight arm's checkpoint evaluation.
# Reuses evaluate_stage5.py's saved prediction .npz artifacts and the
# Step H2.5-confirmed 256x256 local-crop dimension (verified for all 180
# teacher v7 videos, zero exclusions). No re-inference/torch/CUDA.
#
#   EVALUATION_DIR=/path/to/.../<experiment>/last \
#     bash checks/real_h5/check_stage5_frame_xy_diagnostics.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

EVALUATION_DIR="${EVALUATION_DIR:?Set EVALUATION_DIR to an evaluate_stage5.py per-checkpoint output_dir (contains h5_metrics.csv and predictions/)}"
CHECKPOINT="${CHECKPOINT:-last}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
PIXEL_WIDTH="${PIXEL_WIDTH:-256}"
PIXEL_HEIGHT="${PIXEL_HEIGHT:-256}"
GRID_RESOLUTIONS="${GRID_RESOLUTIONS:-16,8}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_frame_xy_diagnostics}"
FRAME_METRICS_CSV="${FRAME_METRICS_CSV:-${WORK_DIR}/frame_metrics.csv}"
VIDEO_SUMMARY_CSV="${VIDEO_SUMMARY_CSV:-${WORK_DIR}/video_summary.csv}"
XY_DENSITY_BINS_CSV="${XY_DENSITY_BINS_CSV:-${WORK_DIR}/xy_density_bins.csv}"
XY_TEMPORAL_RECURRENCE_CSV="${XY_TEMPORAL_RECURRENCE_CSV:-${WORK_DIR}/xy_temporal_recurrence.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORK_DIR}/stage5_s5_14_frame_xy_summary.json}"

mkdir -p "${WORK_DIR}"

echo "Stage5 S5-14 Step H3/H3.1 frame/XY diagnostics"
echo "  python          : ${PYTHON}"
echo "  evaluation dir  : ${EVALUATION_DIR}"
echo "  checkpoint      : ${CHECKPOINT}"
echo "  pixel dims      : ${PIXEL_WIDTH} x ${PIXEL_HEIGHT}"
echo "  grid resolutions: ${GRID_RESOLUTIONS}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_frame_xy_diagnostics.py" \
  --evaluation_dir "${EVALUATION_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --ignore_index "${IGNORE_INDEX}" \
  --pixel_width "${PIXEL_WIDTH}" \
  --pixel_height "${PIXEL_HEIGHT}" \
  --grid_resolutions "${GRID_RESOLUTIONS}" \
  --frame_metrics_csv "${FRAME_METRICS_CSV}" \
  --video_summary_csv "${VIDEO_SUMMARY_CSV}" \
  --xy_density_bins_csv "${XY_DENSITY_BINS_CSV}" \
  --xy_temporal_recurrence_csv "${XY_TEMPORAL_RECURRENCE_CSV}" \
  --summary_json "${SUMMARY_JSON}"

echo "Done."
echo "  frame metrics csv        : ${FRAME_METRICS_CSV}"
echo "  video summary csv        : ${VIDEO_SUMMARY_CSV}"
echo "  xy density bins csv      : ${XY_DENSITY_BINS_CSV}"
echo "  xy temporal recurrence csv: ${XY_TEMPORAL_RECURRENCE_CSV}"
echo "  summary json             : ${SUMMARY_JSON}"
