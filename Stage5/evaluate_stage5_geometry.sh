#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-17: training-free geometry diagnosis (CPU only). No training, no GPU,
# no re-inference, no production change. Reads the saved S5-15 R0 long-run
# evaluation (best = epoch 6 main, last = epoch 50 auxiliary), teacher v7 H5s
# and intermediate H5 attrs, under the B' split contract (pin s5_16_bprime).
#
# APPROVAL: the real-data steps are gated by the S5-17 management document.
#   MODE=register-coverage and MODE=audit belong to S17-4,
#   MODE=run belongs to S17-5. Do not run them before that step is approved.
#
# PRIVACY: PRIVATE_OUT_DIR receives per-video records and the alias map and is
# DO_NOT_SHARE. Only SHARED_JSON (aggregates, privacy-checked before writing)
# may be shared.
#
#   MODE=register-coverage CHECKPOINT_NAME=best TRAIN_SANITY_LIST=... \
#     PRIVATE_OUT_DIR=... bash evaluate_stage5_geometry.sh
#   MODE=audit TRAIN_SANITY_LIST=... PRIVATE_OUT_DIR=... SHARED_JSON=... bash evaluate_stage5_geometry.sh
#   MODE=run   TRAIN_SANITY_LIST=... PRIVATE_OUT_DIR=... SHARED_JSON=... bash evaluate_stage5_geometry.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

MODE="${MODE:?set MODE to register-coverage, audit or run}"

SPLIT_MANIFEST="${SPLIT_MANIFEST:-s5_16_bprime}"
SPLIT_DIR="${SPLIT_DIR:-/mnt/data/3d_projects/stage5_splits/s5_16_step0_bprime}"
TRAIN_CORE_LIST="${TRAIN_CORE_LIST:-${SPLIT_DIR}/train_core_144.txt}"
VALIDATION_LIST="${VALIDATION_LIST:-${SPLIT_DIR}/validation_18.txt}"
TRAIN_SANITY_LIST="${TRAIN_SANITY_LIST:?set TRAIN_SANITY_LIST to the explicit 3-video sanity list}"

EVALUATION_ROOT="${EVALUATION_ROOT:-/mnt/data/3d_projects/stage5_evaluations}"
EX_DATE="${EX_DATE:-260919}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pointnext_s_EX260919_s5_15_r0long50_none_gn8_cwfixed_lr1e3_ep50_bs1_acc8_nopad}"
EVALUATION_OUTPUT="${EVALUATION_OUTPUT:-${EVALUATION_ROOT}/${EX_DATE}/${EXPERIMENT_NAME}}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
DATE="${DATE:-260711}"
INTERMEDIATE_ROOT="${INTERMEDIATE_ROOT:-${DATASET_ROOT}/pseudo3d_outputs/${DATE}}"
INTERMEDIATE_SUFFIX="${INTERMEDIATE_SUFFIX:-_ts448_oym96_corr.h5}"

PRIVATE_OUT_DIR="${PRIVATE_OUT_DIR:?set PRIVATE_OUT_DIR (DO_NOT_SHARE)}"
ARTIFACT_COVERAGE="${ARTIFACT_COVERAGE:-${PRIVATE_OUT_DIR}/artifact_coverage_DO_NOT_SHARE.json}"

COMMON=(
  --split_manifest "${SPLIT_MANIFEST}"
  --train_core_list "${TRAIN_CORE_LIST}"
  --validation_list "${VALIDATION_LIST}"
  --train_sanity_list "${TRAIN_SANITY_LIST}"
)

echo "Stage5 S5-17 geometry diagnosis: ${MODE}"
echo "  python : ${PYTHON}"

case "${MODE}" in
  register-coverage)
    CHECKPOINT_NAME="${CHECKPOINT_NAME:?set CHECKPOINT_NAME to best or last}"
    "${PYTHON}" "${SCRIPT_DIR}/evaluate_stage5_geometry.py" register-coverage "${COMMON[@]}" \
      --evaluation_checkpoint_dir "${EVALUATION_OUTPUT}/${CHECKPOINT_NAME}" \
      --checkpoint_name "${CHECKPOINT_NAME}" \
      --coverage_out "${ARTIFACT_COVERAGE}" \
      ${SHARED_JSON:+--shared_json "${SHARED_JSON}"}
    ;;
  audit|run)
    SHARED_JSON="${SHARED_JSON:?set SHARED_JSON (aggregates only)}"
    EXTRA=()
    [[ -n "${CHECKPOINT_SHA256_BEST:-}" ]] && EXTRA+=(--checkpoint_sha256_best "${CHECKPOINT_SHA256_BEST}")
    [[ -n "${CHECKPOINT_SHA256_LAST:-}" ]] && EXTRA+=(--checkpoint_sha256_last "${CHECKPOINT_SHA256_LAST}")
    [[ -n "${EVALUATION_REVISION_BEST:-}" ]] && EXTRA+=(--evaluation_revision_best "${EVALUATION_REVISION_BEST}")
    [[ -n "${EVALUATION_REVISION_LAST:-}" ]] && EXTRA+=(--evaluation_revision_last "${EVALUATION_REVISION_LAST}")
    if [[ "${MODE}" == "run" ]]; then
      EXTRA+=(--coordinate_mode "${COORDINATE_MODE:-pixel}")
      [[ -n "${H5_HASH_RECORD:-}" ]] && EXTRA+=(--h5_hash_record "${H5_HASH_RECORD}")
    fi
    "${PYTHON}" "${SCRIPT_DIR}/evaluate_stage5_geometry.py" "${MODE}" "${COMMON[@]}" \
      --intermediate_root "${INTERMEDIATE_ROOT}" \
      --intermediate_suffix "${INTERMEDIATE_SUFFIX}" \
      --artifact_coverage "${ARTIFACT_COVERAGE}" \
      --evaluation_dir_best "${EVALUATION_OUTPUT}/best" \
      --evaluation_dir_last "${EVALUATION_OUTPUT}/last" \
      --private_out_dir "${PRIVATE_OUT_DIR}" \
      --shared_json "${SHARED_JSON}" \
      ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  *)
    echo "unknown MODE: ${MODE}" >&2
    exit 2
    ;;
esac

echo "Done."
