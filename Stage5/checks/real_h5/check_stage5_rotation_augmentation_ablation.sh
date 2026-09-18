#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P3: compare the R0 (augmentation=none) and R1 (random_z_rotation)
# training runs. Aggregation and verification ONLY -- this script never starts
# training and never modifies a run directory. Use
# run_stage5_s5_15_arm.sh for the (separately approved) training runs.
#
# CPU/JSON/CSV only: no torch, no CUDA. Trained-checkpoint hash equality is
# deliberately not required; the arms are supposed to diverge.
#
#   R0_DIR=/path/to/r0 R1_DIR=/path/to/r1 \
#     bash checks/real_h5/check_stage5_rotation_augmentation_ablation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

R0_DIR="${R0_DIR:?Set R0_DIR to the R0 (augmentation=none) train_stage5.py output_dir}"
R1_DIR="${R1_DIR:?Set R1_DIR to the R1 (random_z_rotation) train_stage5.py output_dir}"
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-5}"
EXPECTED_ROTATION_DEGREES="${EXPECTED_ROTATION_DEGREES:-15.0}"
EXPECTED_INIT_SHA256="${EXPECTED_INIT_SHA256:-55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b}"
REFERENCE_RUN_DIR="${REFERENCE_RUN_DIR:-}"
EVAL_CSV_R0="${EVAL_CSV_R0:-}"
EVAL_CSV_R1="${EVAL_CSV_R1:-}"
# Launch manifests. Supplying them compares what the launcher intended to pass
# with what each run actually used; leaving them unset reports that comparison
# as UNKNOWN rather than silently skipping it.
MANIFEST_R0="${MANIFEST_R0:-}"
MANIFEST_R1="${MANIFEST_R1:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/work_dirs/_s5_15_rotation_ablation}"
PRIVATE_JSON="${PRIVATE_JSON:-${OUTPUT_ROOT}/rotation_ablation_private.json}"
SHAREABLE_JSON="${SHAREABLE_JSON:-${OUTPUT_ROOT}/rotation_ablation_shareable.json}"

mkdir -p "${OUTPUT_ROOT}"

echo "Stage5 S5-15 rotation-augmentation ablation check (no training is started)"
echo "  python   : ${PYTHON}"
echo "  R0 dir   : ${R0_DIR}"
echo "  R1 dir   : ${R1_DIR}"
echo "  epochs   : ${EXPECTED_EPOCHS}"

args=(
  "${SCRIPT_DIR}/checks/real_h5/check_stage5_rotation_augmentation_ablation.py"
  --r0_dir "${R0_DIR}"
  --r1_dir "${R1_DIR}"
  --expected_epochs "${EXPECTED_EPOCHS}"
  --expected_rotation_degrees "${EXPECTED_ROTATION_DEGREES}"
  --private_json "${PRIVATE_JSON}"
  --shareable_json "${SHAREABLE_JSON}"
)

if [[ -n "${EXPECTED_INIT_SHA256}" ]]; then
  args+=(--expected_init_sha256 "${EXPECTED_INIT_SHA256}")
fi
if [[ -n "${REFERENCE_RUN_DIR}" ]]; then
  args+=(--reference_run_dir "${REFERENCE_RUN_DIR}")
fi
if [[ -n "${MANIFEST_R0}" ]]; then
  args+=(--manifest_r0 "${MANIFEST_R0}")
fi
if [[ -n "${MANIFEST_R1}" ]]; then
  args+=(--manifest_r1 "${MANIFEST_R1}")
fi
if [[ -n "${EVAL_CSV_R0}" && -n "${EVAL_CSV_R1}" ]]; then
  args+=(--eval_csv_r0 "${EVAL_CSV_R0}" --eval_csv_r1 "${EVAL_CSV_R1}")
elif [[ -n "${EVAL_CSV_R0}" || -n "${EVAL_CSV_R1}" ]]; then
  echo "Set EVAL_CSV_R0 and EVAL_CSV_R1 together, or neither." >&2
  exit 1
fi

"${PYTHON}" "${args[@]}"

echo "Done."
echo "  private   : ${PRIVATE_JSON}"
echo "  shareable : ${SHAREABLE_JSON}"
