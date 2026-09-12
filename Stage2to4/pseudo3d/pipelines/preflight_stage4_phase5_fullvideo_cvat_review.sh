#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
MANIFEST="${MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv}"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg}"
AUTO_RUN_NAME="${AUTO_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SOURCE_RUN_NAME}}"
AUTO_RUN_ROOT="${AUTO_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${AUTO_RUN_NAME}}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-${AUTO_RUN_ROOT}/manual_review_cvat_phase5_fullvideo_v2_preflight}"

SOURCE_ANNOTATED_ROOT="${SOURCE_ANNOTATED_ROOT:-${SOURCE_RUN_ROOT}/annotated}"
AUTO_ANNOTATED_ROOT="${AUTO_ANNOTATED_ROOT:-${AUTO_RUN_ROOT}/annotated}"
PHASE1_ROOT="${PHASE1_ROOT:-${SOURCE_RUN_ROOT}/contour_teacher_audit_phase1}"
PHASE3_ROOT="${PHASE3_ROOT:-${SOURCE_RUN_ROOT}/contour_auto_refine_phase3_production_v1_decisions}"
TEACHER_CONFIG="${TEACHER_CONFIG:-pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml}"
REFINE_CONFIG="${REFINE_CONFIG:-pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml}"
MANUAL_CONFIG="${MANUAL_CONFIG:-pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml}"

EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-182}"
EXPECTED_SELECTED_VIDEOS="${EXPECTED_SELECTED_VIDEOS:-60}"
EXPECTED_REVIEW_BBOXES="${EXPECTED_REVIEW_BBOXES:-129}"
EXPECTED_AUTO_ACCEPT="${EXPECTED_AUTO_ACCEPT:-31}"
EXPECTED_AUTO_REFINE="${EXPECTED_AUTO_REFINE:-35}"
EXPECTED_MANUAL_REVIEW="${EXPECTED_MANUAL_REVIEW:-63}"
VERIFY_SOURCE_CHECKSUMS="${VERIFY_SOURCE_CHECKSUMS:-}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1, got: ${OVERWRITE}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
for required_file in "${MANIFEST}" "${TEACHER_CONFIG}" "${REFINE_CONFIG}" "${MANUAL_CONFIG}"; do
  if [[ ! -f "${required_file}" && ! -f "${REPO_ROOT}/${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
for required_dir in "${SOURCE_ANNOTATED_ROOT}" "${AUTO_ANNOTATED_ROOT}" "${PHASE1_ROOT}" "${PHASE3_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory not found: ${required_dir}" >&2
    exit 1
  fi
done
if [[ -n "${VERIFY_SOURCE_CHECKSUMS}" && ! -f "${VERIFY_SOURCE_CHECKSUMS}" ]]; then
  echo "VERIFY_SOURCE_CHECKSUMS not found: ${VERIFY_SOURCE_CHECKSUMS}" >&2
  exit 1
fi

optional_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  optional_args+=(--overwrite)
fi
if [[ -n "${VERIFY_SOURCE_CHECKSUMS}" ]]; then
  optional_args+=(--verify_source_checksums "${VERIFY_SOURCE_CHECKSUMS}")
fi

cd "${REPO_ROOT}"

echo "Stage 4 Phase 5 full-video real-data preflight"
echo "  Phase 3 source : ${SOURCE_ANNOTATED_ROOT}"
echo "  correction base: ${AUTO_ANNOTATED_ROOT}"
echo "  report root    : ${PREFLIGHT_ROOT}"
echo "  selected videos: ${EXPECTED_SELECTED_VIDEOS}"
echo "  target BBoxes  : ${EXPECTED_REVIEW_BBOXES}"

"${PYTHON}" \
  pseudo3d/analysis/preflight_stage4_phase5_fullvideo_cvat_review.py \
  --manifest "${MANIFEST}" \
  --annotated_root "${SOURCE_ANNOTATED_ROOT}" \
  --source_annotated_root "${AUTO_ANNOTATED_ROOT}" \
  --phase1_audit_root "${PHASE1_ROOT}" \
  --phase3_root "${PHASE3_ROOT}" \
  --teacher_config "${TEACHER_CONFIG}" \
  --refine_config "${REFINE_CONFIG}" \
  --manual_review_config "${MANUAL_CONFIG}" \
  --output_root "${PREFLIGHT_ROOT}" \
  --annotated_glob_template "{video_name}/*bboxrank_v2_nobbox_bg.h5" \
  --source_annotated_glob_template "{video_name}/*bboxrank_v3_refined_auto_v1.h5" \
  --expected_videos "${EXPECTED_VIDEOS}" \
  --expected_selected_videos "${EXPECTED_SELECTED_VIDEOS}" \
  --expected_review_bboxes "${EXPECTED_REVIEW_BBOXES}" \
  --expected_auto_accept "${EXPECTED_AUTO_ACCEPT}" \
  --expected_auto_refine "${EXPECTED_AUTO_REFINE}" \
  --expected_manual_review "${EXPECTED_MANUAL_REVIEW}" \
  "${optional_args[@]}"

echo
echo "Stage 4 Phase 5 full-video preflight complete."
echo "  summary         : ${PREFLIGHT_ROOT}/preflight_summary.json"
echo "  video summary   : ${PREFLIGHT_ROOT}/video_summary.csv"
echo "  target BBoxes   : ${PREFLIGHT_ROOT}/target_bboxes.csv"
echo "  source checksums: ${PREFLIGHT_ROOT}/source_checksums.csv"
