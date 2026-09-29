#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-17 S17-6: supplementary shared aggregates from the S17-5 private record
# ((e) pseudo-3D reference, H17-1b BBox consistency, H17-5 multiple instances,
# H17-2 matched-pair errors). Reads ONLY the saved private record of the S17-5
# run (after a hash-bound coverage check); no H5, NPZ or teacher generation CSV,
# no recomputation, not a re-run of the diagnosis.
#
#   MODE=register  -- operator attestation (full hash fixed in the report) plus
#                     metadata checks, then a coverage entry
#   MODE=aggregate -- aggregates into a NEW shared JSON (inputs untouched)
#
#   EXPECTED_SHARED_RUN_SHA256=<value fixed in the S5-17 report> \
#   TRAIN_SANITY_LIST=... PRIVATE_OUT_DIR=... MODE=register bash supplement_stage5_geometry_aggregates.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

MODE="${MODE:?set MODE to register or aggregate}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-s5_16_bprime}"
SPLIT_DIR="${SPLIT_DIR:-/mnt/data/3d_projects/stage5_splits/s5_16_step0_bprime}"
TRAIN_CORE_LIST="${TRAIN_CORE_LIST:-${SPLIT_DIR}/train_core_144.txt}"
VALIDATION_LIST="${VALIDATION_LIST:-${SPLIT_DIR}/validation_18.txt}"
TRAIN_SANITY_LIST="${TRAIN_SANITY_LIST:?set TRAIN_SANITY_LIST (the Step 0 sanity list)}"
PRIVATE_OUT_DIR="${PRIVATE_OUT_DIR:?set PRIVATE_OUT_DIR (the S17-5 private directory)}"
EXPECTED_SHARED_RUN_SHA256="${EXPECTED_SHARED_RUN_SHA256:?set the shared run.json SHA-256 fixed in the report}"
ARTIFACT_COVERAGE="${ARTIFACT_COVERAGE:-${PRIVATE_OUT_DIR}/artifact_coverage_DO_NOT_SHARE.json}"

COMMON=(
  --split_manifest "${SPLIT_MANIFEST}"
  --train_core_list "${TRAIN_CORE_LIST}"
  --validation_list "${VALIDATION_LIST}"
  --train_sanity_list "${TRAIN_SANITY_LIST}"
  --private_run_json "${PRIVATE_OUT_DIR}/geometry_run_DO_NOT_SHARE.json"
  --shared_run_json "${PRIVATE_OUT_DIR}/shared/run.json"
  --expected_shared_run_sha256 "${EXPECTED_SHARED_RUN_SHA256}"
)

echo "Stage5 S5-17 geometry supplement: ${MODE}"
echo "  python : ${PYTHON}"

case "${MODE}" in
  register)
    ATTESTED_PRIVATE_SHA256="${ATTESTED_PRIVATE_SHA256:?set the operator-attested full SHA-256 fixed in the report}"
    ATTESTATION_REFERENCE="${ATTESTATION_REFERENCE:?set where the operator attestation is recorded}"
    "${PYTHON}" "${SCRIPT_DIR}/supplement_stage5_geometry_aggregates.py" register "${COMMON[@]}" \
      --attested_private_sha256 "${ATTESTED_PRIVATE_SHA256}" \
      --attestation_reference "${ATTESTATION_REFERENCE}" \
      --resource_usage "${PRIVATE_OUT_DIR}/run_resource_usage.txt" \
      --coverage_out "${ARTIFACT_COVERAGE}" \
      --shared_json "${PRIVATE_OUT_DIR}/shared/supplement_registration.json"
    ;;
  aggregate)
    "${PYTHON}" "${SCRIPT_DIR}/supplement_stage5_geometry_aggregates.py" aggregate "${COMMON[@]}" \
      --artifact_coverage "${ARTIFACT_COVERAGE}" \
      --shared_json "${PRIVATE_OUT_DIR}/shared/supplement.json"
    ;;
  *)
    echo "unknown MODE: ${MODE}" >&2
    exit 2
    ;;
esac

echo "Done."
