#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-12 Step E3: audit every teacher v6 H5 in the train/val lists for the
# BBox non-contour label-policy contract (every ignore point lies inside a
# stored BBox) before enabling --label_policy bbox_noncontour_background.
# CPU/h5py only -- no GPU required.
#
#   bash checks/real_h5/check_stage5_label_policy_bbox_preflight.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_DIR="${RUN_DIR:-/mnt/data/3d_projects/stage5_runs/260912/pointnext_s_EX260912_260711_w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad}"
TRAIN_LIST="${TRAIN_LIST:-${RUN_DIR}/train_files.txt}"
VAL_LIST="${VAL_LIST:-${RUN_DIR}/val_files.txt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/3d_projects/stage5_debug/label_policy_bbox_preflight}"
SHARE_OUTPUT_DIR="${SHARE_OUTPUT_DIR:-${OUTPUT_ROOT}/share_metrics}"
PRIVATE_OUTPUT_DIR="${PRIVATE_OUTPUT_DIR:-${OUTPUT_ROOT}/private_DO_NOT_SHARE}"
ALLOW_STRAY_IGNORE="${ALLOW_STRAY_IGNORE:-0}"
# Space-separated video_alias values (e.g. "train_068 val_009") to dump
# per-point stray-ignore detail for (frame_order, rounded pixel_xy, the
# frame's stored BBox coords). Implies allow_stray_ignore for this run.
DUMP_STRAY_ALIAS="${DUMP_STRAY_ALIAS:-}"

if [[ ! -s "${TRAIN_LIST}" ]]; then
  echo "Train list not found or empty: ${TRAIN_LIST}" >&2
  exit 1
fi
if [[ ! -s "${VAL_LIST}" ]]; then
  echo "Validation list not found or empty: ${VAL_LIST}" >&2
  exit 1
fi

mkdir -p "${SHARE_OUTPUT_DIR}" "${PRIVATE_OUTPUT_DIR}"

echo "Stage5 label-policy BBox preflight"
echo "  python             : ${PYTHON}"
echo "  train list         : ${TRAIN_LIST}"
echo "  val list           : ${VAL_LIST}"
echo "  share output dir   : ${SHARE_OUTPUT_DIR}"
echo "  private output dir : ${PRIVATE_OUTPUT_DIR}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_label_policy_bbox_preflight.py"
  --train_list "${TRAIN_LIST}"
  --val_list "${VAL_LIST}"
  --share_output_dir "${SHARE_OUTPUT_DIR}"
  --private_output_dir "${PRIVATE_OUTPUT_DIR}"
)
if [[ "${ALLOW_STRAY_IGNORE}" == "1" ]]; then
  cmd+=(--allow_stray_ignore)
fi
if [[ -n "${DUMP_STRAY_ALIAS}" ]]; then
  for alias in ${DUMP_STRAY_ALIAS}; do
    cmd+=(--dump_stray_alias "${alias}")
  done
fi

"${cmd[@]}"

echo "Done."
echo "  summary : ${SHARE_OUTPUT_DIR}/label_policy_bbox_preflight_summary.json"
