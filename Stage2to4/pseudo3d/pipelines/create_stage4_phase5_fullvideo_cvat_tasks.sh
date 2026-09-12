#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"

PHASE5_FULLVIDEO_REVIEW_ROOT="${PHASE5_FULLVIDEO_REVIEW_ROOT:-/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v2}"
CVAT_HOST="${CVAT_HOST:-http://localhost:8080}"
CVAT_USERNAME="${CVAT_USERNAME:-}"
CVAT_ORGANIZATION_SLUG="${CVAT_ORGANIZATION_SLUG:-}"
TASK_NAME_PREFIX="${TASK_NAME_PREFIX:-stage4_phase5_fullvideo__}"
CVAT_ANNOTATION_FORMAT="${CVAT_ANNOTATION_FORMAT:-Segmentation mask 1.1}"
EXPECTED_VIDEO_PACKAGES="${EXPECTED_VIDEO_PACKAGES:-60}"
EXPECTED_REVIEW_BBOXES="${EXPECTED_REVIEW_BBOXES:-129}"

SMOKE_ONLY="${SMOKE_ONLY:-1}"
SMOKE_VIDEO="${SMOKE_VIDEO:-1-3_14}"
MAX_NEW_TASKS="${MAX_NEW_TASKS:-0}"
APPLY="${APPLY:-0}"

TASK_MANAGEMENT_ROOT="${TASK_MANAGEMENT_ROOT:-${PHASE5_FULLVIDEO_REVIEW_ROOT}/task_management}"
STATE_JSON="${STATE_JSON:-${TASK_MANAGEMENT_ROOT}/task_creation_state.json}"
TASK_MAP_CSV="${TASK_MAP_CSV:-${TASK_MANAGEMENT_ROOT}/task_map.csv}"

for value_name in APPLY SMOKE_ONLY; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
done
if [[ "${PYTHON}" == */* ]]; then
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON}" >&2
    exit 1
  fi
elif ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python command not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Repository root not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${PHASE5_FULLVIDEO_REVIEW_ROOT}" ]]; then
  echo "Full-video review root not found: ${PHASE5_FULLVIDEO_REVIEW_ROOT}" >&2
  exit 1
fi
for required_file in \
  "${PHASE5_FULLVIDEO_REVIEW_ROOT}/partition_summary.json" \
  "${PHASE5_FULLVIDEO_REVIEW_ROOT}/video_progress.csv" \
  "${PHASE5_FULLVIDEO_REVIEW_ROOT}/mac_transfer_manifest.csv" \
  "${PHASE5_FULLVIDEO_REVIEW_ROOT}/mac_transfer_summary.json" \
  "${PHASE5_FULLVIDEO_REVIEW_ROOT}/package_validation_summary.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required Step 4 artifact not found: ${required_file}" >&2
    exit 1
  fi
done

command_args=(
  --review_root "${PHASE5_FULLVIDEO_REVIEW_ROOT}"
  --server_host "${CVAT_HOST}"
  --reference_task_id 0
  --reference_video "${SMOKE_VIDEO}"
  --full_video_standalone
  --task_name_prefix "${TASK_NAME_PREFIX}"
  --annotation_format "${CVAT_ANNOTATION_FORMAT}"
  --expected_video_packages "${EXPECTED_VIDEO_PACKAGES}"
  --expected_review_bboxes "${EXPECTED_REVIEW_BBOXES}"
  --max_new_tasks "${MAX_NEW_TASKS}"
  --state_json "${STATE_JSON}"
  --task_map_csv "${TASK_MAP_CSV}"
)
if [[ "${APPLY}" == "1" ]]; then
  command_args+=(--apply)
fi
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  command_args+=(--only_video "${SMOKE_VIDEO}")
fi
if [[ -n "${CVAT_USERNAME}" ]]; then
  command_args+=(--username "${CVAT_USERNAME}")
fi
if [[ -n "${CVAT_ORGANIZATION_SLUG}" ]]; then
  command_args+=(--organization_slug "${CVAT_ORGANIZATION_SLUG}")
fi

cd "${REPO_ROOT}"

echo "Stage 4 Phase 5 full-video standalone CVAT task launcher"
echo "  review root      : ${PHASE5_FULLVIDEO_REVIEW_ROOT}"
echo "  CVAT host        : ${CVAT_HOST}"
echo "  task prefix      : ${TASK_NAME_PREFIX}"
echo "  annotation format: ${CVAT_ANNOTATION_FORMAT}"
echo "  expected packages: ${EXPECTED_VIDEO_PACKAGES}"
echo "  expected BBoxes  : ${EXPECTED_REVIEW_BBOXES}"
echo "  smoke only       : ${SMOKE_ONLY}"
echo "  smoke video      : ${SMOKE_VIDEO}"
echo "  max new tasks    : ${MAX_NEW_TASKS}"
echo "  apply            : ${APPLY}"
echo "  state            : ${STATE_JSON}"

"${PYTHON}" \
  pseudo3d/batch/export/batch_create_stage4_phase5_cvat_tasks.py \
  "${command_args[@]}"
