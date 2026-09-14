#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-12 Step E4: Dataset parity between bbox_noncontour_ignore (Run A) and
# bbox_noncontour_background (Run B) on real teacher v6 H5s. CPU/h5py only --
# no GPU required. Only writes video_alias-labeled JSON locally; no path or
# video-name leakage.
#
#   bash checks/real_h5/check_stage5_label_policy_dataset_parity.sh
#
# Limit to a handful of files first for a quick check:
#   MAX_FILES=5 bash checks/real_h5/check_stage5_label_policy_dataset_parity.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_DIR="${RUN_DIR:-/mnt/data/3d_projects/stage5_runs/260912/pointnext_s_EX260912_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad}"
TRAIN_LIST="${TRAIN_LIST:-${RUN_DIR}/train_files.txt}"
VAL_LIST="${VAL_LIST:-${RUN_DIR}/val_files.txt}"
MAX_FILES="${MAX_FILES:-0}"
WINDOW_SIZE_FRAMES="${WINDOW_SIZE_FRAMES:-16}"
WINDOW_STRIDE_FRAMES="${WINDOW_STRIDE_FRAMES:-8}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/work_dirs/_label_policy_dataset_parity/summary.json}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "Stage5 label-policy Dataset parity check"
echo "  python       : ${PYTHON}"
echo "  train list   : ${TRAIN_LIST}"
echo "  val list     : ${VAL_LIST}"
echo "  max_files    : ${MAX_FILES} (0 = all)"
echo "  window       : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_label_policy_dataset_parity.py" \
  --train_list "${TRAIN_LIST}" \
  --val_list "${VAL_LIST}" \
  --max_files "${MAX_FILES}" \
  --window_size_frames "${WINDOW_SIZE_FRAMES}" \
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}" \
  --output_json "${OUTPUT_JSON}"

echo "Done."
echo "  output: ${OUTPUT_JSON}"
