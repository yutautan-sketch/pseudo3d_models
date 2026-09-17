#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: compare two checkpoints in one run directory -- first as files
# (SHA-256), then, only if they differ, as evaluation models (state_dict
# keys/shapes/dtypes/values). Read-only and CPU-only: no CUDA, no training,
# and nothing in RUN_DIR is modified.
#
# Written for the approved one-off audit of the historical W-A run's
# best.pt vs last.pt.
#
#   RUN_DIR=/path/to/run bash checks/real_h5/check_stage5_checkpoint_identity.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the train_stage5.py output_dir holding the two checkpoints}"
CHECKPOINT_A="${CHECKPOINT_A:-best.pt}"
CHECKPOINT_B="${CHECKPOINT_B:-last.pt}"
SKIP_LOAD="${SKIP_LOAD:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/work_dirs/_s5_15_p1_audit}"
JSON_OUT="${JSON_OUT:-${OUTPUT_ROOT}/wa_checkpoint_identity.json}"

mkdir -p "${OUTPUT_ROOT}"

echo "Stage5 checkpoint identity check (read-only, CPU only)"
echo "  python : ${PYTHON}"
echo "  run dir: ${RUN_DIR}"
echo "  compare: ${CHECKPOINT_A} vs ${CHECKPOINT_B}"

args=(
  "${SCRIPT_DIR}/checks/real_h5/check_stage5_checkpoint_identity.py"
  --run_dir "${RUN_DIR}"
  --checkpoint_a "${CHECKPOINT_A}"
  --checkpoint_b "${CHECKPOINT_B}"
  --json_out "${JSON_OUT}"
)

if [[ "${SKIP_LOAD}" == "1" ]]; then
  args+=(--skip_load)
fi

"${PYTHON}" "${args[@]}"

echo "Done."
echo "  output: ${JSON_OUT}"
