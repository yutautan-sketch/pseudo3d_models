#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

SOURCE_TOKEN="bboxrank_v5_cvat_authoritative_v1"
OUTPUT_TOKEN="bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-global_local_l75_w31_c12_area15_${SOURCE_TOKEN}}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-global_local_l75_w31_c12_area15_${OUTPUT_TOKEN}}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SOURCE_RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${OUTPUT_RUN_NAME}}"
SOURCE_ANNOTATED_ROOT="${SOURCE_ANNOTATED_ROOT:-${SOURCE_RUN_ROOT}/annotated}"
ANNOTATED_ROOT="${ANNOTATED_ROOT:-${OUTPUT_ROOT}/annotated}"
COLLECTED_ROOT="${COLLECTED_ROOT:-${OUTPUT_ROOT}/collected}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/invalidation_summary.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_ROOT}/invalidation_summary.json}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest_cropclean_v1.csv}"
EXCLUSION_MANIFEST="${EXCLUSION_MANIFEST:-pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv}"
INVALIDATION_MANIFEST="${INVALIDATION_MANIFEST:-pseudo3d/analysis/configs/stage4_deleted_xml_invalidations_v1.csv}"
STEP2_SUMMARY="${STEP2_SUMMARY:-${SOURCE_RUN_ROOT}/deleted_xml_annotation_invalidation_step2/manifest_summary.json}"

EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-181}"
EXPECTED_EXCLUDED_VIDEOS="${EXPECTED_EXCLUDED_VIDEOS:-1}"
EXPECTED_INVALIDATIONS="${EXPECTED_INVALIDATIONS:-7}"
EXPECTED_AFFECTED_VIDEOS="${EXPECTED_AFFECTED_VIDEOS:-2}"
EXPECTED_POSITIVE_POINTS="${EXPECTED_POSITIVE_POINTS:-2124}"
EXPECTED_REMOVED_BBOX_ROWS="${EXPECTED_REMOVED_BBOX_ROWS:-7}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ "${SKIP_EXISTING}" != "0" && "${SKIP_EXISTING}" != "1" ]]; then
  echo "SKIP_EXISTING must be 0 or 1, got: ${SKIP_EXISTING}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
for required_file in \
  "${TRAIN_MANIFEST}" \
  "${EXCLUSION_MANIFEST}" \
  "${INVALIDATION_MANIFEST}" \
  "${STEP2_SUMMARY}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${SOURCE_ANNOTATED_ROOT}" ]]; then
  echo "v5 annotated source root not found: ${SOURCE_ANNOTATED_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
mode_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  mode_args+=(--skip_existing)
else
  mode_args+=(--overwrite)
fi

echo "Stage 4 teacher v6 deleted-XML invalidation build"
echo "  source run        : ${SOURCE_RUN_ROOT}"
echo "  output run        : ${OUTPUT_ROOT}"
echo "  expected videos   : ${EXPECTED_VIDEOS}"
echo "  excluded videos   : ${EXPECTED_EXCLUDED_VIDEOS}"
echo "  invalidated frames: ${EXPECTED_INVALIDATIONS}"
echo "  affected videos   : ${EXPECTED_AFFECTED_VIDEOS}"
echo "  label policy      : deleted XML frame -> all background"

"${PYTHON}" \
  pseudo3d/batch/annotation/batch_apply_stage4_deleted_xml_invalidations.py \
  --train_manifest "${TRAIN_MANIFEST}" \
  --exclusion_manifest "${EXCLUSION_MANIFEST}" \
  --invalidation_manifest "${INVALIDATION_MANIFEST}" \
  --step2_summary "${STEP2_SUMMARY}" \
  --source_annotated_root "${SOURCE_ANNOTATED_ROOT}" \
  --output_annotated_root "${ANNOTATED_ROOT}" \
  --output_collected_root "${COLLECTED_ROOT}" \
  --summary_csv "${SUMMARY_CSV}" \
  --summary_json "${SUMMARY_JSON}" \
  --expected_videos "${EXPECTED_VIDEOS}" \
  --expected_excluded_videos "${EXPECTED_EXCLUDED_VIDEOS}" \
  --expected_invalidations "${EXPECTED_INVALIDATIONS}" \
  --expected_affected_videos "${EXPECTED_AFFECTED_VIDEOS}" \
  --expected_positive_points "${EXPECTED_POSITIVE_POINTS}" \
  --expected_removed_bbox_rows "${EXPECTED_REMOVED_BBOX_ROWS}" \
  "${mode_args[@]}" \
  2>&1 | tee "${LOG_ROOT}/01_deleted_xml_invalidation_apply.log"

annotated_count="$(find "${ANNOTATED_ROOT}" -type f -name "*${OUTPUT_TOKEN}.h5" | wc -l | tr -d ' ')"
collected_count="$(find "${COLLECTED_ROOT}" -maxdepth 1 -type f -name "*${OUTPUT_TOKEN}.h5" | wc -l | tr -d ' ')"
if [[ "${annotated_count}" -ne "${EXPECTED_VIDEOS}" || "${collected_count}" -ne "${EXPECTED_VIDEOS}" ]]; then
  echo "v6 H5 count mismatch: annotated=${annotated_count}, collected=${collected_count}" >&2
  exit 1
fi

echo
echo "Stage 4 teacher v6 deleted-XML invalidation build complete."
echo "  annotated : ${ANNOTATED_ROOT}"
echo "  collected : ${COLLECTED_ROOT}"
echo "  summary   : ${SUMMARY_JSON}"
echo "  files     : annotated=${annotated_count}, collected=${collected_count}"
