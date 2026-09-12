#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

RUN_NAME="${RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v5_cvat_authoritative_v1}"
RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${RUN_NAME}}"
ANNOTATED_ROOT="${ANNOTATED_ROOT:-${RUN_ROOT}/annotated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/deleted_xml_annotation_audit_step1}"

MANIFEST="${MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest_cropclean_v1.csv}"
EXCLUSION_MANIFEST="${EXCLUSION_MANIFEST:-pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv}"

EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-181}"
EXPECTED_EXCLUDED_VIDEOS="${EXPECTED_EXCLUDED_VIDEOS:-1}"
EXPECTED_MISSING_FRAMES="${EXPECTED_MISSING_FRAMES:--1}"
EXPECTED_MISSING_XML_ENTRIES="${EXPECTED_MISSING_XML_ENTRIES:--1}"
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
for required_file in "${MANIFEST}" "${EXCLUSION_MANIFEST}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
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
  pseudo3d/analysis/audit_stage4_deleted_xml_annotations.py \
  --manifest "${MANIFEST}" \
  --exclusion_manifest "${EXCLUSION_MANIFEST}" \
  --annotated_root "${ANNOTATED_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --expected_videos "${EXPECTED_VIDEOS}" \
  --expected_excluded_videos "${EXPECTED_EXCLUDED_VIDEOS}" \
  --expected_missing_frames "${EXPECTED_MISSING_FRAMES}" \
  --expected_missing_xml_entries "${EXPECTED_MISSING_XML_ENTRIES}" \
  "${mode_args[@]}"

echo
echo "Stage 4 deleted-XML Step 1 audit complete."
echo "  summary   : ${OUTPUT_ROOT}/audit_summary.json"
echo "  candidates: ${OUTPUT_ROOT}/missing_xml_candidates.csv"
echo "  inventory : ${OUTPUT_ROOT}/xml_inventory.csv"
echo "  videos    : ${OUTPUT_ROOT}/video_summary.csv"
echo "  checksums : ${OUTPUT_ROOT}/input_checksums.csv"
