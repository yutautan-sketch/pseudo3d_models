#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

RUN_NAME="${RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v5_cvat_authoritative_v1}"
RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${RUN_NAME}}"
ANNOTATED_ROOT="${ANNOTATED_ROOT:-${RUN_ROOT}/annotated}"
AUDIT_ROOT="${AUDIT_ROOT:-${RUN_ROOT}/deleted_xml_annotation_audit_step1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/deleted_xml_annotation_invalidation_step2}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest_cropclean_v1.csv}"
EXCLUSION_MANIFEST="${EXCLUSION_MANIFEST:-pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-pseudo3d/analysis/configs/stage4_deleted_xml_invalidations_v1.csv}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-${OUTPUT_ROOT}/manifest_summary.json}"

EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-181}"
EXPECTED_INVALIDATIONS="${EXPECTED_INVALIDATIONS:-7}"
EXPECTED_POSITIVE_POINTS="${EXPECTED_POSITIVE_POINTS:-2124}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1, got: ${OVERWRITE}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
for required_file in "${TRAIN_MANIFEST}" "${EXCLUSION_MANIFEST}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
for audit_file in \
  audit_summary.json \
  xml_inventory.csv \
  missing_xml_candidates.csv \
  video_summary.csv \
  input_checksums.csv; do
  if [[ ! -f "${AUDIT_ROOT}/${audit_file}" ]]; then
    echo "Step 1 audit artifact not found: ${AUDIT_ROOT}/${audit_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${ANNOTATED_ROOT}" ]]; then
  echo "v5 annotated root not found: ${ANNOTATED_ROOT}" >&2
  exit 1
fi

mode_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  mode_args+=(--overwrite)
fi

"${PYTHON}" \
  pseudo3d/analysis/build_stage4_deleted_xml_invalidation_manifest.py \
  --audit_root "${AUDIT_ROOT}" \
  --train_manifest "${TRAIN_MANIFEST}" \
  --exclusion_manifest "${EXCLUSION_MANIFEST}" \
  --annotated_root "${ANNOTATED_ROOT}" \
  --output_manifest "${OUTPUT_MANIFEST}" \
  --output_summary "${OUTPUT_SUMMARY}" \
  --expected_videos "${EXPECTED_VIDEOS}" \
  --expected_invalidations "${EXPECTED_INVALIDATIONS}" \
  --expected_positive_points "${EXPECTED_POSITIVE_POINTS}" \
  "${mode_args[@]}"

echo
echo "Stage 4 deleted-XML Step 2 manifest fixation complete."
echo "  manifest   : ${OUTPUT_MANIFEST}"
echo "  summary    : ${OUTPUT_SUMMARY}"
echo "  audit root : ${AUDIT_ROOT}"
