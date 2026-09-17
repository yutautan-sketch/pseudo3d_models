#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 P1: audit the W-A reference training run against the S5-15 fixed
# conditions. CPU/JSON only -- no GPU, no torch/h5py required, and nothing
# inside RUN_DIR is modified. No training is started by this script.
#
# Writes two outputs:
#   PRIVATE_JSON   full audit incl. real paths/filenames -- keep on this host.
#   SHAREABLE_JSON anonymized (counts, hashes, statuses) -- safe to share.
#
#   RUN_DIR=/path/to/wa/run bash checks/real_h5/check_stage5_s5_15_p1_manifest_audit.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the W-A train_stage5.py output_dir to audit}"
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-5}"
EXPECTED_TRAIN_FILES="${EXPECTED_TRAIN_FILES:-162}"
EXPECTED_VAL_FILES="${EXPECTED_VAL_FILES:-18}"
SKIP_CHECKPOINT_HASH="${SKIP_CHECKPOINT_HASH:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/work_dirs/_s5_15_p1_audit}"
PRIVATE_JSON="${PRIVATE_JSON:-${OUTPUT_ROOT}/p1_manifest_audit_private.json}"
SHAREABLE_JSON="${SHAREABLE_JSON:-${OUTPUT_ROOT}/p1_manifest_audit_shareable.json}"

mkdir -p "${OUTPUT_ROOT}"

echo "Stage5 S5-15 P1 manifest audit"
echo "  python    : ${PYTHON}"
echo "  run dir   : ${RUN_DIR}"
echo "  epochs    : ${EXPECTED_EPOCHS}"
echo "  files     : train=${EXPECTED_TRAIN_FILES} val=${EXPECTED_VAL_FILES}"

args=(
  "${SCRIPT_DIR}/checks/real_h5/check_stage5_s5_15_p1_manifest_audit.py"
  --run_dir "${RUN_DIR}"
  --expected_epochs "${EXPECTED_EPOCHS}"
  --expected_train_files "${EXPECTED_TRAIN_FILES}"
  --expected_val_files "${EXPECTED_VAL_FILES}"
  --private_json "${PRIVATE_JSON}"
  --shareable_json "${SHAREABLE_JSON}"
)

if [[ "${SKIP_CHECKPOINT_HASH}" == "1" ]]; then
  args+=(--skip_checkpoint_hash)
fi

"${PYTHON}" "${args[@]}"

echo "Done."
echo "  private   : ${PRIVATE_JSON}"
echo "  shareable : ${SHAREABLE_JSON}"
