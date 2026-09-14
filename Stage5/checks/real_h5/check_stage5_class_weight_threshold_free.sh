#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-13補足 Step 6: threshold-free (AUPRC/AUROC/PR curve) diagnostics for one
# class-weight arm's checkpoint evaluation. Reuses evaluate_stage5.py's saved
# prediction .npz artifacts (SAVE_PREDICTIONS=1, the evaluate_stage5.sh
# default) and reads GT directly from the source H5 -- no model re-inference,
# no CUDA/torch. Fail-fasts if the recomputed threshold-0.5 TP/FP/TN/FN
# disagree with the existing h5_metrics.csv (parity gate).
#
#   EVALUATION_DIR=/path/to/.../<experiment>/last \
#     bash checks/real_h5/check_stage5_class_weight_threshold_free.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

EVALUATION_DIR="${EVALUATION_DIR:?Set EVALUATION_DIR to an evaluate_stage5.py per-checkpoint output_dir (contains h5_metrics.csv and predictions/)}"
CHECKPOINT="${CHECKPOINT:-last}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
THRESHOLD_GRID_STEP="${THRESHOLD_GRID_STEP:-0.05}"
REFERENCE_FPR="${REFERENCE_FPR:-}"
FIXED_FPR_CANDIDATES="${FIXED_FPR_CANDIDATES:-0.01,0.05,0.10}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/work_dirs/_class_weight_threshold_free/$(basename "$(dirname "${EVALUATION_DIR}")")_$(basename "${EVALUATION_DIR}")_${CHECKPOINT}.json}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "Stage5 class-weight threshold-free diagnostics"
echo "  python          : ${PYTHON}"
echo "  evaluation dir  : ${EVALUATION_DIR}"
echo "  checkpoint      : ${CHECKPOINT}"
echo "  threshold step  : ${THRESHOLD_GRID_STEP}"
echo "  fixed FPR cands : ${FIXED_FPR_CANDIDATES}"

args=(
  "${SCRIPT_DIR}/checks/real_h5/check_stage5_class_weight_threshold_free.py"
  --evaluation_dir "${EVALUATION_DIR}"
  --checkpoint "${CHECKPOINT}"
  --ignore_index "${IGNORE_INDEX}"
  --threshold_grid_step "${THRESHOLD_GRID_STEP}"
  --fixed_fpr_candidates "${FIXED_FPR_CANDIDATES}"
  --output_json "${OUTPUT_JSON}"
)
if [[ -n "${REFERENCE_FPR}" ]]; then
  args+=(--reference_fpr "${REFERENCE_FPR}")
fi

"${PYTHON}" "${args[@]}"

echo "Done."
echo "  output: ${OUTPUT_JSON}"
