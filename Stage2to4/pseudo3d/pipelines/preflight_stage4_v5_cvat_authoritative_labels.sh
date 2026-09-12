#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v5_cvat_authoritative_v1}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SOURCE_RUN_NAME}}"
OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${OUTPUT_RUN_NAME}}"
SOURCE_ANNOTATED_ROOT="${SOURCE_ANNOTATED_ROOT:-${SOURCE_RUN_ROOT}/annotated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_RUN_ROOT}/cvat_authoritative_preflight_step5}"

MANIFEST="${MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest_cropclean_v1.csv}"
EXCLUSION_MANIFEST="${EXCLUSION_MANIFEST:-pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv}"
MANUAL_REVIEW_CONFIG="${MANUAL_REVIEW_CONFIG:-pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml}"

PHASE5_RETURN_ROOT="${PHASE5_RETURN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/cvat_returns/phase5_fullvideo_v3_textfree_reviewed_snapshot_59_v2/manual_review_cvat_phase5_fullvideo_v3_textfree}"
REVIEW_ROOT="${REVIEW_ROOT:-${SOURCE_RUN_ROOT}/manual_review_cvat_phase5_fullvideo_v3_textfree}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${PHASE5_RETURN_ROOT}/cvat_exports_after_textfree_review_v2}"
TASK_MAP_CSV="${TASK_MAP_CSV:-${PHASE5_RETURN_ROOT}/task_management/task_map.csv}"

EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-181}"
EXPECTED_REVIEWED_VIDEOS="${EXPECTED_REVIEWED_VIDEOS:-59}"
EXPECTED_ACTIONABLE_BBOXES="${EXPECTED_ACTIONABLE_BBOXES:-112}"
EXPECTED_TASK_FRAMES="${EXPECTED_TASK_FRAMES:-3014}"
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
for required_file in "${MANIFEST}" "${EXCLUSION_MANIFEST}" "${MANUAL_REVIEW_CONFIG}" "${TASK_MAP_CSV}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
for required_dir in "${SOURCE_ANNOTATED_ROOT}" "${REVIEW_ROOT}" "${SNAPSHOT_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory not found: ${required_dir}" >&2
    exit 1
  fi
done

mode_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  mode_args+=(--overwrite)
fi

"${PYTHON}" \
  pseudo3d/analysis/preflight_stage4_cvat_authoritative_labels.py \
  --manifest "${MANIFEST}" \
  --exclusion_manifest "${EXCLUSION_MANIFEST}" \
  --source_annotated_root "${SOURCE_ANNOTATED_ROOT}" \
  --review_root "${REVIEW_ROOT}" \
  --snapshot_root "${SNAPSHOT_ROOT}" \
  --task_map_csv "${TASK_MAP_CSV}" \
  --manual_review_config "${MANUAL_REVIEW_CONFIG}" \
  --output_root "${OUTPUT_ROOT}" \
  --expected_videos "${EXPECTED_VIDEOS}" \
  --expected_reviewed_videos "${EXPECTED_REVIEWED_VIDEOS}" \
  --expected_actionable_bboxes "${EXPECTED_ACTIONABLE_BBOXES}" \
  --expected_task_frames "${EXPECTED_TASK_FRAMES}" \
  "${mode_args[@]}"

echo
echo "Stage 4 CVAT-authoritative Step 5 preflight complete."
echo "  summary      : ${OUTPUT_ROOT}/preflight_summary.json"
echo "  frame metrics: ${OUTPUT_ROOT}/frame_metrics.csv"
echo "  video summary: ${OUTPUT_ROOT}/video_summary.csv"
