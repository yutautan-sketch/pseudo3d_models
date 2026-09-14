#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
PYTHON_BIN="${PYTHON_BIN:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
V6_RUN_ROOT="${V6_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1}"

SOURCE_MANIFEST="${SOURCE_MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv}"
PREVIOUS_EXCLUSIONS="${PREVIOUS_EXCLUSIONS:-${REPO_ROOT}/pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv}"
VIDEO_EXCLUSIONS="${VIDEO_EXCLUSIONS:-${REPO_ROOT}/pseudo3d/analysis/configs/stage4_video_exclusions_v2.csv}"
CROP_INVALIDATIONS="${CROP_INVALIDATIONS:-${REPO_ROOT}/pseudo3d/analysis/configs/stage4_crop_quality_invalidations_v1.csv}"
AUDIT_ROOT="${AUDIT_ROOT:-${V6_RUN_ROOT}/crop_quality_stray_ignore_audit_step1}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest_cropclean_v2.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${V6_RUN_ROOT}/crop_quality_manifest_step2}"
OVERWRITE="${OVERWRITE:-0}"

cd "${REPO_ROOT}"

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

echo "Stage 4 crop-quality manifest v2 build"
echo "  audit root        : ${AUDIT_ROOT}"
echo "  source manifest   : ${SOURCE_MANIFEST}"
echo "  video exclusions  : ${VIDEO_EXCLUSIONS}"
echo "  crop invalidations: ${CROP_INVALIDATIONS}"
echo "  output manifest   : ${OUTPUT_MANIFEST}"

"${PYTHON_BIN}" pseudo3d/analysis/validate_stage4_crop_quality_manifests.py \
  --audit_root "${AUDIT_ROOT}" \
  --source_manifest "${SOURCE_MANIFEST}" \
  --previous_exclusions "${PREVIOUS_EXCLUSIONS}" \
  --video_exclusions "${VIDEO_EXCLUSIONS}" \
  --crop_invalidations "${CROP_INVALIDATIONS}" \
  --output_summary "${OUTPUT_ROOT}/manifest_validation_summary.json" \
  --expected_excluded_video 20250626_090758_8000 \
  --expected_excluded_video 20250626_090652_6340 \
  --expected_invalidations 1 \
  "${overwrite_args[@]}"

"${PYTHON_BIN}" pseudo3d/analysis/build_stage4_exclusion_manifest.py \
  --source_manifest "${SOURCE_MANIFEST}" \
  --exclusions "${VIDEO_EXCLUSIONS}" \
  --output_manifest "${OUTPUT_MANIFEST}" \
  --summary_json "${OUTPUT_ROOT}/excluded_manifest_summary.json" \
  --expected_source_rows 182 \
  --expected_excluded 2 \
  "${overwrite_args[@]}"

echo
echo "Stage 4 crop-quality manifest v2 build complete."
echo "  train manifest    : ${OUTPUT_MANIFEST}"
echo "  exclusion summary : ${OUTPUT_ROOT}/excluded_manifest_summary.json"
echo "  validation summary: ${OUTPUT_ROOT}/manifest_validation_summary.json"
echo "  frame invalidation: ${CROP_INVALIDATIONS}"
