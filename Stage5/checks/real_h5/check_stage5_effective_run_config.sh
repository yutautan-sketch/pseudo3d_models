#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: verify a training run's EFFECTIVE settings and that periodic
# checkpoint saving works, early enough to stop a run that is deviating.
#
# The env-passthrough test is a static analysis of the shell scripts; this is
# its runtime counterpart, reading what train_stage5.py actually wrote to
# config.json. Defaults match the 50-epoch R0 long run.
#
# Read-only, CPU only. Run it once epoch 5 has finished.
#
#   RUN_DIR=/path/to/run bash checks/real_h5/check_stage5_effective_run_config.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the train_stage5.py output_dir to inspect}"
# "-" rather than ":-": an explicitly empty EXPECT_CHECKPOINT must mean "skip the
# checkpoint check", which is what a run inspected before its first periodic save
# needs. With ":-" an empty value fell back to the default and reported a
# missing checkpoint as a failure on a perfectly healthy run.
EXPECT_CHECKPOINT="${EXPECT_CHECKPOINT-checkpoint_epoch_0005.pt}"
EPOCHS="${EPOCHS:-50}"
SAVE_EVERY="${SAVE_EVERY:-5}"
AUGMENTATION="${AUGMENTATION:-none}"
SEED="${SEED:-42}"
JSON_OUT="${JSON_OUT:-${SCRIPT_DIR}/work_dirs/_s5_15_longrun_manifests/effective_config_check.json}"

echo "Stage5 effective run-config check (read-only)"
echo "  python  : ${PYTHON}"
echo "  run dir : ${RUN_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_effective_run_config.py" \
  --run_dir "${RUN_DIR}" \
  --expect "epochs=${EPOCHS}" \
  --expect "save_every=${SAVE_EVERY}" \
  --expect "augmentation=${AUGMENTATION}" \
  --expect "seed=${SEED}" \
  --expect "pointnext_norm=groupnorm" \
  --expect "pointnext_norm_groups=8" \
  --expect "label_policy=bbox_noncontour_ignore" \
  --expect "batch_size=1" \
  --expect "gradient_accumulation_steps=8" \
  --expect "window_size_frames=16" \
  --expect "window_stride_frames=8" \
  --expect_checkpoint "${EXPECT_CHECKPOINT}" \
  --json_out "${JSON_OUT}"

echo "Done."
echo "  output: ${JSON_OUT}"
