#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
AUTO_RUN_NAME="${AUTO_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1}"
AUTO_RUN_ROOT="${AUTO_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${AUTO_RUN_NAME}}"
PHASE5_REVIEW_ROOT="${PHASE5_REVIEW_ROOT:-${AUTO_RUN_ROOT}/manual_review_cvat_phase5_all_phase3_v1}"

CVAT_HOST="${CVAT_HOST:-http://localhost:8080}"
CVAT_USERNAME="${CVAT_USERNAME:-}"
CVAT_ORGANIZATION_SLUG="${CVAT_ORGANIZATION_SLUG:-}"
REFERENCE_TASK_ID="${REFERENCE_TASK_ID:-5}"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-1-3_14}"
REFERENCE_IS_MASTER="${REFERENCE_IS_MASTER:-1}"
EXPECTED_VIDEO_PACKAGES="${EXPECTED_VIDEO_PACKAGES:-60}"
MAX_NEW_TASKS="${MAX_NEW_TASKS:-0}"
APPLY="${APPLY:-0}"

if [[ "${APPLY}" != "0" && "${APPLY}" != "1" ]]; then
  echo "APPLY must be 0 or 1, got: ${APPLY}" >&2
  exit 1
fi
if [[ "${REFERENCE_IS_MASTER}" != "0" && "${REFERENCE_IS_MASTER}" != "1" ]]; then
  echo "REFERENCE_IS_MASTER must be 0 or 1, got: ${REFERENCE_IS_MASTER}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${PHASE5_REVIEW_ROOT}" ]]; then
  echo "Phase 5 review root not found: ${PHASE5_REVIEW_ROOT}" >&2
  exit 1
fi

apply_args=()
if [[ "${APPLY}" == "1" ]]; then
  apply_args+=(--apply)
fi
username_args=()
if [[ -n "${CVAT_USERNAME}" ]]; then
  username_args+=(--username "${CVAT_USERNAME}")
fi
organization_args=()
if [[ -n "${CVAT_ORGANIZATION_SLUG}" ]]; then
  organization_args+=(--organization_slug "${CVAT_ORGANIZATION_SLUG}")
fi
reference_scope_args=()
if [[ "${REFERENCE_IS_MASTER}" == "1" ]]; then
  reference_scope_args+=(--reference_is_master)
fi

cd "${REPO_ROOT}"

echo "Stage 4 Phase 5-B CVAT task launcher"
echo "  review root      : ${PHASE5_REVIEW_ROOT}"
echo "  CVAT host        : ${CVAT_HOST}"
echo "  reference task   : ${REFERENCE_TASK_ID} (${REFERENCE_VIDEO})"
echo "  reference master : ${REFERENCE_IS_MASTER}"
echo "  expected packages: ${EXPECTED_VIDEO_PACKAGES}"
echo "  max new tasks    : ${MAX_NEW_TASKS}"
echo "  apply            : ${APPLY}"

"${PYTHON}" \
  pseudo3d/batch/export/batch_create_stage4_phase5_cvat_tasks.py \
  --review_root "${PHASE5_REVIEW_ROOT}" \
  --server_host "${CVAT_HOST}" \
  --reference_task_id "${REFERENCE_TASK_ID}" \
  --reference_video "${REFERENCE_VIDEO}" \
  "${reference_scope_args[@]}" \
  --expected_video_packages "${EXPECTED_VIDEO_PACKAGES}" \
  --max_new_tasks "${MAX_NEW_TASKS}" \
  "${username_args[@]}" \
  "${organization_args[@]}" \
  "${apply_args[@]}"
