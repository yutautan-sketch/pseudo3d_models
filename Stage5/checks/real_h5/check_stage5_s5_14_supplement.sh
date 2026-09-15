#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14補足 Step S1-S5: denominator-carrying XY-bin FPR/recall, a
# train-GT-only XY prior with hot/cold FPR comparison, within-video paired
# temporal recall, and vote-count-joint exposure/FPR stratification.
# Reuses W-A's already-saved evaluate_stage5.py prediction .npz artifacts
# (pred_label, vote_count) and the existing Step H3 frame_metrics.csv.
# No re-training, no GPU re-inference, no torch/CUDA import.
#
#   EVALUATION_DIR=/path/to/.../<experiment>/last \
#   TRAIN_LIST=/path/to/train_file_list.txt \
#   FRAME_METRICS_CSV=/path/to/step_h3/frame_metrics.csv \
#     bash checks/real_h5/check_stage5_s5_14_supplement.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

EVALUATION_DIR="${EVALUATION_DIR:?Set EVALUATION_DIR to an evaluate_stage5.py per-checkpoint output_dir (contains h5_metrics.csv and predictions/)}"
TRAIN_LIST="${TRAIN_LIST:?Set TRAIN_LIST to the W-A train file list (162 H5 paths) for the Step S2 GT-only XY prior}"
FRAME_METRICS_CSV="${FRAME_METRICS_CSV:?Set FRAME_METRICS_CSV to the existing Step H3 frame_metrics.csv}"
CHECKPOINT="${CHECKPOINT:-last}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
PIXEL_WIDTH="${PIXEL_WIDTH:-256}"
PIXEL_HEIGHT="${PIXEL_HEIGHT:-256}"
GRID_RESOLUTIONS="${GRID_RESOLUTIONS:-16,8}"
WINDOW_SIZE_FRAMES="${WINDOW_SIZE_FRAMES:-16}"
WINDOW_STRIDE_FRAMES="${WINDOW_STRIDE_FRAMES:-8}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_s5_14_supplement}"
XY_BIN_DENOMINATORS_CSV="${XY_BIN_DENOMINATORS_CSV:-${WORK_DIR}/xy_bin_denominators.csv}"
TRAIN_XY_PRIOR_CSV="${TRAIN_XY_PRIOR_CSV:-${WORK_DIR}/train_xy_prior.csv}"
XY_HOT_COLD_VIDEO_METRICS_CSV="${XY_HOT_COLD_VIDEO_METRICS_CSV:-${WORK_DIR}/xy_hot_cold_video_metrics.csv}"
VIDEO_DECILE_COUNTS_CSV="${VIDEO_DECILE_COUNTS_CSV:-${WORK_DIR}/video_decile_counts.csv}"
PAIRED_TEMPORAL_METRICS_CSV="${PAIRED_TEMPORAL_METRICS_CSV:-${WORK_DIR}/paired_temporal_metrics.csv}"
VIDEO_TIME_EXPOSURE_METRICS_CSV="${VIDEO_TIME_EXPOSURE_METRICS_CSV:-${WORK_DIR}/video_time_exposure_metrics.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORK_DIR}/stage5_s5_14_supplement_summary.json}"

mkdir -p "${WORK_DIR}"

echo "Stage5 S5-14 supplement (Step S1-S5)"
echo "  python           : ${PYTHON}"
echo "  evaluation dir   : ${EVALUATION_DIR}"
echo "  train list       : ${TRAIN_LIST}"
echo "  frame metrics csv: ${FRAME_METRICS_CSV}"
echo "  checkpoint       : ${CHECKPOINT}"
echo "  grid resolutions : ${GRID_RESOLUTIONS}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_s5_14_supplement.py" \
  --evaluation_dir "${EVALUATION_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --train_list "${TRAIN_LIST}" \
  --frame_metrics_csv "${FRAME_METRICS_CSV}" \
  --ignore_index "${IGNORE_INDEX}" \
  --pixel_width "${PIXEL_WIDTH}" \
  --pixel_height "${PIXEL_HEIGHT}" \
  --grid_resolutions "${GRID_RESOLUTIONS}" \
  --window_size_frames "${WINDOW_SIZE_FRAMES}" \
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}" \
  --xy_bin_denominators_csv "${XY_BIN_DENOMINATORS_CSV}" \
  --train_xy_prior_csv "${TRAIN_XY_PRIOR_CSV}" \
  --xy_hot_cold_video_metrics_csv "${XY_HOT_COLD_VIDEO_METRICS_CSV}" \
  --video_decile_counts_csv "${VIDEO_DECILE_COUNTS_CSV}" \
  --paired_temporal_metrics_csv "${PAIRED_TEMPORAL_METRICS_CSV}" \
  --video_time_exposure_metrics_csv "${VIDEO_TIME_EXPOSURE_METRICS_CSV}" \
  --summary_json "${SUMMARY_JSON}"

echo "Done."
echo "  xy bin denominators csv        : ${XY_BIN_DENOMINATORS_CSV}"
echo "  train xy prior csv             : ${TRAIN_XY_PRIOR_CSV}"
echo "  xy hot/cold video metrics csv  : ${XY_HOT_COLD_VIDEO_METRICS_CSV}"
echo "  video decile counts csv        : ${VIDEO_DECILE_COUNTS_CSV}"
echo "  paired temporal metrics csv    : ${PAIRED_TEMPORAL_METRICS_CSV}"
echo "  video time exposure metrics csv: ${VIDEO_TIME_EXPOSURE_METRICS_CSV}"
echo "  summary json                   : ${SUMMARY_JSON}"
