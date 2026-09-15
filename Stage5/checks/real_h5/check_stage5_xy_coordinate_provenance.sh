#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H2.5: Stage 4 dataset inventory and XY coordinate provenance
# audit. Determines whether teacher v7's crop/image dimensions can be
# recovered from intermediate pseudo-3D H5 provenance, for Step H3.1's
# cross-video XY comparison. CPU/h5py only, no CUDA. Read-only: never
# writes to any teacher/source H5.
#
#   bash checks/real_h5/check_stage5_xy_coordinate_provenance.sh
#
# Limit to a handful of files first for a quick check:
#   MAX_FILES=5 bash checks/real_h5/check_stage5_xy_coordinate_provenance.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_DIR="${RUN_DIR:-/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad}"
TRAIN_LIST="${TRAIN_LIST:-${RUN_DIR}/train_files.txt}"
VAL_LIST="${VAL_LIST:-${RUN_DIR}/val_files.txt}"
MAX_FILES="${MAX_FILES:-0}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
DATE="${DATE:-260711}"
STAGE4_TRAINING_ABLATION_ROOT="${STAGE4_TRAINING_ABLATION_ROOT:-${DATASET_ROOT}/stage4_training_ablation/${DATE}}"
PSEUDO3D_OUTPUTS_ROOT="${PSEUDO3D_OUTPUTS_ROOT:-${DATASET_ROOT}/pseudo3d_outputs/${DATE}}"
FALLBACK_SUFFIX="${FALLBACK_SUFFIX:-_ts448_oym96_corr.h5}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_xy_coordinate_provenance}"
PRIVATE_INVENTORY_CSV="${PRIVATE_INVENTORY_CSV:-${WORK_DIR}/private_DO_NOT_SHARE/stage4_dataset_coordinate_inventory.csv}"
PRIVATE_SOURCE_RESOLUTION_CSV="${PRIVATE_SOURCE_RESOLUTION_CSV:-${WORK_DIR}/private_DO_NOT_SHARE/stage4_xy_source_resolution.csv}"
SHAREABLE_SUMMARY_JSON="${SHAREABLE_SUMMARY_JSON:-${WORK_DIR}/SHARE_THIS/stage4_xy_coordinate_provenance_summary.json}"
SHAREABLE_DIMENSION_AUDIT_CSV="${SHAREABLE_DIMENSION_AUDIT_CSV:-${WORK_DIR}/SHARE_THIS/stage4_xy_dimension_audit.csv}"

mkdir -p "$(dirname "${PRIVATE_INVENTORY_CSV}")" "$(dirname "${SHAREABLE_SUMMARY_JSON}")"

echo "Stage5 S5-14 Step H2.5 XY coordinate provenance audit"
echo "  python                        : ${PYTHON}"
echo "  train list                    : ${TRAIN_LIST}"
echo "  val list                      : ${VAL_LIST}"
echo "  max_files                     : ${MAX_FILES} (0 = all)"
echo "  stage4_training_ablation_root : ${STAGE4_TRAINING_ABLATION_ROOT}"
echo "  pseudo3d_outputs_root         : ${PSEUDO3D_OUTPUTS_ROOT}"
echo "  fallback_suffix               : ${FALLBACK_SUFFIX}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_xy_coordinate_provenance.py" \
  --train_list "${TRAIN_LIST}" \
  --val_list "${VAL_LIST}" \
  --max_files "${MAX_FILES}" \
  --stage4_training_ablation_root "${STAGE4_TRAINING_ABLATION_ROOT}" \
  --pseudo3d_outputs_root "${PSEUDO3D_OUTPUTS_ROOT}" \
  --fallback_suffix "${FALLBACK_SUFFIX}" \
  --private_inventory_csv "${PRIVATE_INVENTORY_CSV}" \
  --private_source_resolution_csv "${PRIVATE_SOURCE_RESOLUTION_CSV}" \
  --shareable_summary_json "${SHAREABLE_SUMMARY_JSON}" \
  --shareable_dimension_audit_csv "${SHAREABLE_DIMENSION_AUDIT_CSV}"

echo "Done."
echo "  private inventory csv        : ${PRIVATE_INVENTORY_CSV}"
echo "  private source resolution csv: ${PRIVATE_SOURCE_RESOLUTION_CSV}"
echo "  shareable summary json       : ${SHAREABLE_SUMMARY_JSON}"
echo "  shareable dimension audit csv: ${SHAREABLE_DIMENSION_AUDIT_CSV}"
