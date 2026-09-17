#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P3: launch one comparison arm.
#
#   ARM=r0  augmentation=none        (control)
#   ARM=r1  augmentation=random_z_rotation, +-15 degrees
#
# Both arms start from the SAME GroupNorm transfer checkpoint and the SAME
# saved train/val lists; augmentation is the only intended difference. Every
# experiment condition below is an internal constant rather than something the
# caller types, so the two arms cannot drift apart by a mistyped override.
#
# THIS SCRIPT CAN START TRAINING, which needs its own approval. It therefore
# defaults to a dry run: it validates the conditions and prints the exact
# command, then stops. Only CONFIRM_TRAINING=1 actually launches, and the run
# is named after the arm so the two never share an output directory.
#
#   ARM=r0 bash checks/real_h5/run_stage5_s5_15_arm.sh                  # dry run
#   ARM=r0 CONFIRM_TRAINING=1 bash checks/real_h5/run_stage5_s5_15_arm.sh
#
# The comparison itself lives in check_stage5_rotation_augmentation_ablation.sh,
# which never trains.
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

# ------------------------------------------------------------
# Fixed experiment conditions (S5-15 P1 table, audited 2026-09-16).
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
EPOCHS="5"
SAVE_EVERY="1"
SEED="42"
AUGMENTATION_ROTATION_DEGREES="15.0"

EX_DATE="${EX_DATE:-260916}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/3d_projects/stage5_runs/${EX_DATE}}"

# train_stage5.py derives the augmentation base seed as SEED + 500000 when no
# explicit one is given. Recomputed here so the dry run can show the value that
# will be recorded, and so the manifest carries it.
AUGMENTATION_SEED_OFFSET="500000"
RESOLVED_AUGMENTATION_SEED="$((SEED + AUGMENTATION_SEED_OFFSET))"

# Manifests are written here, NOT into the run directory: the launcher refuses
# to start into a non-empty run directory and that protection is not weakened
# to make room for a manifest. Each manifest records its run directory path.
MANIFEST_DIR="${MANIFEST_DIR:-${SCRIPT_DIR}/work_dirs/_s5_15_p3_manifests}"

ARM="${ARM:?Set ARM to r0 (augmentation=none) or r1 (random_z_rotation)}"
case "${ARM}" in
  r0) AUGMENTATION="none" ;;
  r1) AUGMENTATION="random_z_rotation" ;;
  *)
    echo "ARM must be r0 or r1; got ${ARM}" >&2
    exit 1
    ;;
esac

CONFIRM_TRAINING="${CONFIRM_TRAINING:-0}"
if [[ "${CONFIRM_TRAINING}" != "0" && "${CONFIRM_TRAINING}" != "1" ]]; then
  echo "CONFIRM_TRAINING must be 0 or 1; got ${CONFIRM_TRAINING}" >&2
  exit 1
fi

EXPERIMENT_NAME="pointnext_s_EX${EX_DATE}_s5_15_${ARM}_${AUGMENTATION}_gn8_cwfixed_lr1e3_ep${EPOCHS}_bs1_acc8_nopad"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

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
  echo "Initialization checkpoint SHA-256 mismatch; both arms must start from the audited weights." >&2
  echo "  expected: ${EXPECTED_INIT_SHA256}" >&2
  echo "  actual  : ${actual_init_sha256}" >&2
  exit 1
fi

# Validate the list pair up front with the same module train_stage5.sh uses, so
# a duplicate, a missing H5 or a stale teacher version stops the arm before any
# GPU time is spent rather than midway through the preflight. The fingerprints
# are verified here, not merely displayed: a list whose content or order has
# drifted from the audited one must stop the arm.
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

echo "Stage5 S5-15 arm ${ARM}"
echo "  augmentation   : ${AUGMENTATION} (train only; eval/inference never augmented)"
echo "  rotation range : +-${AUGMENTATION_ROTATION_DEGREES} degrees (ignored when augmentation=none)"
echo "  init ckpt      : ${INIT_CHECKPOINT}"
echo "  init sha256    : ${actual_init_sha256} (matches the audited value)"
echo "  train list     : ${TRAIN_LIST}"
echo "  val list       : ${VAL_LIST}"
echo "  list check     : ${list_summary}"
echo "  expected list  : train ${EXPECTED_TRAIN_LIST_SHA256}"
echo "                 : val   ${EXPECTED_VAL_LIST_SHA256}"
echo "  note           : the run's own saved train_files.txt/val_files.txt are compared against these"
echo "                 : by check_stage5_rotation_augmentation_ablation.sh after training"
echo "  norm           : ${POINTNEXT_NORM} (${POINTNEXT_NORM_GROUPS} groups)"
echo "  class weight   : ${CLASS_WEIGHT}"
echo "  label policy   : ${LABEL_POLICY}"
echo "  epochs         : ${EPOCHS} (save_every=${SAVE_EVERY}, every epoch kept)"
echo "  seed           : ${SEED} (augmentation base seed resolves to ${RESOLVED_AUGMENTATION_SEED})"
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
  "AUGMENTATION_ROTATION_DEGREES=${AUGMENTATION_ROTATION_DEGREES}"
  "EPOCHS=${EPOCHS}"
  "SAVE_EVERY=${SAVE_EVERY}"
  "SEED=${SEED}"
  "EX_DATE=${EX_DATE}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  "PYTHON=${PYTHON}"
)

echo
echo "Command:"
launch_command="$(printf '%s ' "${launch_env[@]}")bash ${SCRIPT_DIR}/train_stage5.sh"
printf '  %s \\\n' "${launch_env[@]}"
echo "    bash ${SCRIPT_DIR}/train_stage5.sh"

# Planned optimizer updates: ceil(windows / accumulation) per epoch. The
# window count comes from the data, so this stays a planned figure; the
# executed count is reported separately and is never assumed to match.
PLANNED_OPTIMIZER_STEPS="${PLANNED_OPTIMIZER_STEPS:-450}"

write_manifest() {
  local mode="$1"
  "${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/write_stage5_s5_15_run_manifest.py" \
    --arm "${ARM}" \
    --mode "${mode}" \
    --manifest_dir "${MANIFEST_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --augmentation "${AUGMENTATION}" \
    --augmentation_rotation_degrees "${AUGMENTATION_ROTATION_DEGREES}" \
    --seed "${SEED}" \
    --resolved_augmentation_seed "${RESOLVED_AUGMENTATION_SEED}" \
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
  echo "Re-run with CONFIRM_TRAINING=1 once this arm has been approved."
  exit 0
fi

write_manifest training

echo
echo "CONFIRM_TRAINING=1: starting ${ARM} training."
env "${launch_env[@]}" bash "${SCRIPT_DIR}/train_stage5.sh"

echo "Done."
echo "  run dir: ${OUTPUT_DIR}"
