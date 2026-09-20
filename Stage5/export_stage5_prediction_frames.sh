#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: render predicted-positive / GT-positive segmentation onto the
# original local-crop frames of an existing evaluation output, so that false
# positives can be read as anatomy or instruments instead of as PLY points.
#
# Reads existing evaluation artifacts only. No training, no inference, no CUDA.
# Never modifies the evaluation output it reads -- it adds one subdirectory.
#
# PRIVACY: the output contains rendered patient frames and real video names.
# It is DO_NOT_SHARE. Anonymize separately before sharing anything from it.
#
#   bash export_stage5_prediction_frames.sh
#
# Resolve inputs and write CSVs/manifest without rendering any PNG first:
#   DRY_RUN=1 bash export_stage5_prediction_frames.sh
#
# One checkpoint only, more frames per video:
#   CHECKPOINT_NAMES="best" TOP_FRAMES_PER_VIDEO=30 \
#     bash export_stage5_prediction_frames.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

EVALUATION_ROOT="${EVALUATION_ROOT:-/mnt/data/3d_projects/stage5_evaluations}"
EX_DATE="${EX_DATE:-260919}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVALUATION_ROOT}/${EX_DATE}/${EXPERIMENT_NAME}}"

# best = epoch 6, last = epoch 50. Both get the same layout so that the same
# video and frame number can be compared side by side.
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-best last}"
SPLITS="${SPLITS:-train_sanity validation}"

STAGE2TO4_ROOT="${STAGE2TO4_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
DATE="${DATE:-260711}"
PSEUDO3D_OUTPUTS_ROOT="${PSEUDO3D_OUTPUTS_ROOT:-${DATASET_ROOT}/pseudo3d_outputs/${DATE}}"
FALLBACK_SUFFIX="${FALLBACK_SUFFIX:-_ts448_oym96_corr.h5}"

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-prediction_frames}"
TOP_FRAMES_PER_VIDEO="${TOP_FRAMES_PER_VIDEO:-10}"
ZERO_FP_SAMPLES="${ZERO_FP_SAMPLES:-2}"
ZERO_TP_SAMPLES="${ZERO_TP_SAMPLES:-2}"
SEED="${SEED:-0}"
RADIUS="${RADIUS:-2}"
ALPHA="${ALPHA:-0.7}"

ALL_FRAMES="${ALL_FRAMES:-0}"
DRAW_TRUE_NEGATIVE="${DRAW_TRUE_NEGATIVE:-0}"
DRAW_IGNORE_OTHER="${DRAW_IGNORE_OTHER:-0}"
DRAW_BBOX="${DRAW_BBOX:-0}"
DRY_RUN="${DRY_RUN:-0}"
HASH_H5="${HASH_H5:-0}"

read -r -a CHECKPOINT_ARRAY <<< "${CHECKPOINT_NAMES}"
read -r -a SPLIT_ARRAY <<< "${SPLITS}"
if [[ "${#CHECKPOINT_ARRAY[@]}" -eq 0 ]]; then
  echo "CHECKPOINT_NAMES is empty" >&2
  exit 1
fi

OPTIONAL_ARGS=()
[[ "${ALL_FRAMES}" == "1" ]] && OPTIONAL_ARGS+=(--all_frames)
[[ "${DRAW_TRUE_NEGATIVE}" == "1" ]] && OPTIONAL_ARGS+=(--draw_true_negative)
[[ "${DRAW_IGNORE_OTHER}" == "1" ]] && OPTIONAL_ARGS+=(--draw_ignore_other)
[[ "${DRAW_BBOX}" == "1" ]] && OPTIONAL_ARGS+=(--draw_bbox)
[[ "${DRY_RUN}" == "1" ]] && OPTIONAL_ARGS+=(--dry_run)
[[ "${HASH_H5}" == "1" ]] && OPTIONAL_ARGS+=(--hash_h5)

echo "Stage5 S5-15 prediction frame visualization"
echo "  python                : ${PYTHON}"
echo "  evaluation output root: ${OUTPUT_ROOT}"
echo "  checkpoints           : ${CHECKPOINT_NAMES}"
echo "  splits                : ${SPLITS}"
echo "  stage2to4 root        : ${STAGE2TO4_ROOT}"
echo "  pseudo3d outputs root : ${PSEUDO3D_OUTPUTS_ROOT}"
echo "  fallback suffix       : ${FALLBACK_SUFFIX}"
echo "  top frames per video  : ${TOP_FRAMES_PER_VIDEO} (all_frames=${ALL_FRAMES})"
echo "  radius / alpha        : ${RADIUS} / ${ALPHA}"
echo "  dry run               : ${DRY_RUN}"

for checkpoint_name in "${CHECKPOINT_ARRAY[@]}"; do
  evaluation_dir="${OUTPUT_ROOT}/${checkpoint_name}"
  if [[ ! -d "${evaluation_dir}" ]]; then
    echo "Evaluation directory not found: ${evaluation_dir}" >&2
    echo "Run evaluate_stage5.sh first, or set CHECKPOINT_NAMES." >&2
    exit 1
  fi

  echo ""
  echo "=== ${checkpoint_name} ==="
  "${PYTHON}" "${SCRIPT_DIR}/export_stage5_prediction_frames.py" \
    --evaluation_dir "${evaluation_dir}" \
    --stage2to4_root "${STAGE2TO4_ROOT}" \
    --pseudo3d_outputs_root "${PSEUDO3D_OUTPUTS_ROOT}" \
    --fallback_suffix "${FALLBACK_SUFFIX}" \
    --splits "${SPLIT_ARRAY[@]}" \
    --output_subdir "${OUTPUT_SUBDIR}" \
    --top_frames_per_video "${TOP_FRAMES_PER_VIDEO}" \
    --zero_fp_samples "${ZERO_FP_SAMPLES}" \
    --zero_tp_samples "${ZERO_TP_SAMPLES}" \
    --seed "${SEED}" \
    --radius "${RADIUS}" \
    --alpha "${ALPHA}" \
    ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}
done

echo ""
echo "Done."
for checkpoint_name in "${CHECKPOINT_ARRAY[@]}"; do
  echo "  ${OUTPUT_ROOT}/${checkpoint_name}/${OUTPUT_SUBDIR}"
done
echo "  DO_NOT_SHARE: rendered patient frames and real video names."
