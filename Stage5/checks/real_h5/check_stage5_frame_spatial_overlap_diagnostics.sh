#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H2: teacher v7 GT + structural window-exposure audit across all
# 180 H5 files. CPU/h5py only -- no GPU/CUDA/torch required, no predictions
# needed (pure H5 + window-geometry audit).
#
#   bash checks/real_h5/check_stage5_frame_spatial_overlap_diagnostics.sh
#
# Limit to a handful of files first for a quick check:
#   MAX_FILES=5 bash checks/real_h5/check_stage5_frame_spatial_overlap_diagnostics.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_DIR="${RUN_DIR:-/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad}"
TRAIN_LIST="${TRAIN_LIST:-${RUN_DIR}/train_files.txt}"
VAL_LIST="${VAL_LIST:-${RUN_DIR}/val_files.txt}"
MAX_FILES="${MAX_FILES:-0}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
WINDOW_SIZE_FRAMES="${WINDOW_SIZE_FRAMES:-16}"
WINDOW_STRIDE_FRAMES="${WINDOW_STRIDE_FRAMES:-8}"
OUTPUT_CSV="${OUTPUT_CSV:-${SCRIPT_DIR}/work_dirs/_frame_spatial_overlap_diagnostics/gt_overlap_exposure.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/work_dirs/_frame_spatial_overlap_diagnostics/gt_overlap_exposure.json}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

echo "Stage5 S5-14 Step H2 GT/exposure audit"
echo "  python       : ${PYTHON}"
echo "  train list   : ${TRAIN_LIST}"
echo "  val list     : ${VAL_LIST}"
echo "  max_files    : ${MAX_FILES} (0 = all)"
echo "  window       : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_frame_spatial_overlap_diagnostics.py" \
  --train_list "${TRAIN_LIST}" \
  --val_list "${VAL_LIST}" \
  --max_files "${MAX_FILES}" \
  --ignore_index "${IGNORE_INDEX}" \
  --window_size_frames "${WINDOW_SIZE_FRAMES}" \
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}" \
  --output_csv "${OUTPUT_CSV}" \
  --output_json "${OUTPUT_JSON}"

echo "Done."
echo "  output csv : ${OUTPUT_CSV}"
echo "  output json: ${OUTPUT_JSON}"
