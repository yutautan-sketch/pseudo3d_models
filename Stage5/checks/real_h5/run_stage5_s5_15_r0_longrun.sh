#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: the 50-epoch R0 long run.
#
# Deliberately a SEPARATE script from run_stage5_s5_15_arm.sh, which stays
# fixed at the 5-epoch R0/R1 comparison. That comparison is finished history
# and must not be silently re-pointed at 50 epochs; this run gets its own
# launcher, its own experiment name and its own output directory.
#
# Conditions are identical to P3's R0 (augmentation=none) and start from the
# SAME GroupNorm transfer checkpoint. It does NOT resume from P3's last.pt:
# `train_stage5.py --checkpoint` restores weights only, so a fresh 50-epoch run
# from the audited init gives one consistent history instead of a stitched one.
# The repeated first five epochs are accepted for that reason; this is not a
# re-run of P3 and does not regenerate P3's missing per-epoch checkpoints.
#
# SAVE_EVERY=5 keeps epoch 5 (the first save, checked early), every tenth epoch,
# and epochs 25 and 50, which the interim and final reviews need.
#
# Defaults to a dry run. Only CONFIRM_TRAINING=1 starts training.
#
#   ARM_LABEL=r0long50 bash checks/real_h5/run_stage5_s5_15_r0_longrun.sh
#   CONFIRM_TRAINING=1 bash checks/real_h5/run_stage5_s5_15_r0_longrun.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

# ------------------------------------------------------------
# Fixed conditions: identical to P3's R0 (S5-15 P1 table, audited 2026-09-16).
# ------------------------------------------------------------
INIT_CHECKPOINT="/mnt/data/3d_projects/models/Stage5/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init_groupnorm.pt"
EXPECTED_INIT_SHA256="55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b"
WA_RUN_DIR="/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad"
TRAIN_LIST="${TRAIN_LIST:-${WA_RUN_DIR}/train_files.txt}"
VAL_LIST="${VAL_LIST:-${WA_RUN_DIR}/val_files.txt}"
EXPECTED_TRAIN_LIST_SHA256="582579833f345b77d111f9f02a0606a7423d4d5161dd0cb6436c994fcf353ab0"
EXPECTED_VAL_LIST_SHA256="0c251380e40f0def3a76bdedd74572ac0ba22e799bf0287c8fee3f14ca59e83f"

POINTNEXT_NORM="groupnorm"
POINTNEXT_NORM_GROUPS="8"
CLASS_WEIGHT="0.05963856,1.94036150"
LABEL_POLICY="bbox_noncontour_ignore"
AUGMENTATION="none"
SEED="42"

# Long-run specific.
EPOCHS="50"
SAVE_EVERY="5"
FIRST_PERIODIC_CHECKPOINT="checkpoint_epoch_0005.pt"
PLANNED_OPTIMIZER_STEPS="4500"

EX_DATE="${EX_DATE:-260919}"
ARM_LABEL="${ARM_LABEL:-r0long50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/3d_projects/stage5_runs/${EX_DATE}}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRIPT_DIR}/work_dirs/_s5_15_longrun_manifests}"

EXPERIMENT_NAME="pointnext_s_EX${EX_DATE}_s5_15_${ARM_LABEL}_none_gn8_cwfixed_lr1e3_ep${EPOCHS}_bs1_acc8_nopad"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"

CONFIRM_TRAINING="${CONFIRM_TRAINING:-0}"
if [[ "${CONFIRM_TRAINING}" != "0" && "${CONFIRM_TRAINING}" != "1" ]]; then
  echo "CONFIRM_TRAINING must be 0 or 1; got ${CONFIRM_TRAINING}" >&2
  exit 1
fi

# ------------------------------------------------------------
# Pre-launch verification (read-only).
# ------------------------------------------------------------
for path in "${INIT_CHECKPOINT}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required input not found: ${path}" >&2
    exit 1
  fi
done

actual_init_sha256="$(sha256sum "${INIT_CHECKPOINT}" | cut -d' ' -f1)"
if [[ "${actual_init_sha256}" != "${EXPECTED_INIT_SHA256}" ]]; then
  echo "Initialization checkpoint SHA-256 mismatch; the long run must start from the audited weights." >&2
  echo "  expected: ${EXPECTED_INIT_SHA256}" >&2
  echo "  actual  : ${actual_init_sha256}" >&2
  exit 1
fi

list_summary="$("${PYTHON}" "${SCRIPT_DIR}/stage5/utils/file_list_mode.py" \
  --train_list "${TRAIN_LIST}" \
  --val_list "${VAL_LIST}" \
  --expected_total 180 \
  --expected_train_sha256 "${EXPECTED_TRAIN_LIST_SHA256}" \
  --expected_val_sha256 "${EXPECTED_VAL_LIST_SHA256}")"

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output directory already exists and is not empty: ${OUTPUT_DIR}" >&2
  echo "Refusing to mix runs. Pick a different EX_DATE, or move the existing run aside." >&2
  exit 1
fi

echo "Stage5 S5-15 R0 long run (${EPOCHS} epochs)"
echo "  relation       : fresh run from the audited init; NOT a resume of the P3 R0 run,"
echo "                 : and NOT a re-run to regenerate P3's missing per-epoch checkpoints"
echo "  augmentation   : ${AUGMENTATION} (R0 conditions, unchanged from P3)"
echo "  init ckpt      : ${INIT_CHECKPOINT}"
echo "  init sha256    : ${actual_init_sha256} (matches the audited value)"
echo "  train list     : ${TRAIN_LIST}"
echo "  val list       : ${VAL_LIST}"
echo "  list check     : ${list_summary}"
echo "  norm           : ${POINTNEXT_NORM} (${POINTNEXT_NORM_GROUPS} groups)"
echo "  class weight   : ${CLASS_WEIGHT}"
echo "  label policy   : ${LABEL_POLICY}"
echo "  seed           : ${SEED}"
echo "  epochs         : ${EPOCHS}"
echo "  save_every     : ${SAVE_EVERY} -> epochs 5,10,15,20,25,30,35,40,45,50 plus best.pt/last.pt"
echo "  first check    : ${FIRST_PERIODIC_CHECKPOINT} must exist once epoch ${SAVE_EVERY} finishes"
echo "  planned steps  : ${PLANNED_OPTIMIZER_STEPS} optimizer updates (90/epoch x ${EPOCHS})"
echo "  output dir     : ${OUTPUT_DIR}"

launch_env=(
  "TRAIN_LIST=${TRAIN_LIST}"
  "VAL_LIST=${VAL_LIST}"
  "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
  "POINTNEXT_NORM=${POINTNEXT_NORM}"
  "POINTNEXT_NORM_GROUPS=${POINTNEXT_NORM_GROUPS}"
  "CLASS_WEIGHT=${CLASS_WEIGHT}"
  "LABEL_POLICY=${LABEL_POLICY}"
  "AUGMENTATION=${AUGMENTATION}"
  "EPOCHS=${EPOCHS}"
  "SAVE_EVERY=${SAVE_EVERY}"
  "SEED=${SEED}"
  "EX_DATE=${EX_DATE}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  "PYTHON=${PYTHON}"
)

launch_command="$(printf '%s ' "${launch_env[@]}")bash ${SCRIPT_DIR}/train_stage5.sh"

echo
echo "Command:"
printf '  %s \\\n' "${launch_env[@]}"
echo "    bash ${SCRIPT_DIR}/train_stage5.sh"

write_manifest() {
  "${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/write_stage5_s5_15_run_manifest.py" \
    --arm "${ARM_LABEL}" \
    --mode "$1" \
    --manifest_dir "${MANIFEST_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --augmentation "${AUGMENTATION}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --save_every "${SAVE_EVERY}" \
    --class_weight "${CLASS_WEIGHT}" \
    --pointnext_norm "${POINTNEXT_NORM}" \
    --pointnext_norm_groups "${POINTNEXT_NORM_GROUPS}" \
    --init_checkpoint "${INIT_CHECKPOINT}" \
    --train_list "${TRAIN_LIST}" \
    --val_list "${VAL_LIST}" \
    --command "${launch_command}" \
    --planned_optimizer_steps "${PLANNED_OPTIMIZER_STEPS}" \
    --repo_dir "${SCRIPT_DIR}"
}

echo
if [[ "${CONFIRM_TRAINING}" != "1" ]]; then
  write_manifest dry_run
  echo
  echo "Dry run: training was NOT started."
  echo "Re-run with CONFIRM_TRAINING=1 once this long run has been approved."
  echo
  echo "After epoch ${SAVE_EVERY} completes, verify the effective settings and periodic saving with:"
  echo "  RUN_DIR=${OUTPUT_DIR} bash checks/real_h5/check_stage5_effective_run_config.sh"
  exit 0
fi

write_manifest training

echo
echo "CONFIRM_TRAINING=1: starting the ${EPOCHS}-epoch R0 long run."
env "${launch_env[@]}" bash "${SCRIPT_DIR}/train_stage5.sh"

echo "Done."
echo "  run dir: ${OUTPUT_DIR}"
