#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"

PHASE5_TEXTFREE_REVIEW_ROOT="${PHASE5_TEXTFREE_REVIEW_ROOT:-/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v3_textfree}"
CORRECTED_SNAPSHOT_ROOT="${CORRECTED_SNAPSHOT_ROOT:-/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v2/cvat_exports_before_textfree_v1}"
EXCLUSION_MANIFEST="${EXCLUSION_MANIFEST:-${REPO_ROOT}/pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv}"
CVAT_MASK_MODULE="${REPO_ROOT}/pseudo3d/export/convert_masks_to_cvat_segmentation_mask_1_1.py"

CVAT_HOST="${CVAT_HOST:-http://localhost:8080}"
CVAT_USERNAME="${CVAT_USERNAME:-}"
CVAT_ORGANIZATION_SLUG="${CVAT_ORGANIZATION_SLUG:-}"
TASK_NAME_PREFIX="${TASK_NAME_PREFIX:-stage4_phase5_fullvideo_v3_textfree__}"
CVAT_ANNOTATION_FORMAT="${CVAT_ANNOTATION_FORMAT:-Segmentation mask 1.1}"
EXPECTED_VIDEO_PACKAGES="${EXPECTED_VIDEO_PACKAGES:-60}"
EXPECTED_REVIEW_BBOXES="${EXPECTED_REVIEW_BBOXES:-129}"

SMOKE_ONLY="${SMOKE_ONLY:-1}"
SMOKE_VIDEO="${SMOKE_VIDEO:-1-3_14}"
MAX_NEW_TASKS="${MAX_NEW_TASKS:-0}"
APPLY="${APPLY:-0}"

TASK_MANAGEMENT_ROOT="${TASK_MANAGEMENT_ROOT:-${PHASE5_TEXTFREE_REVIEW_ROOT}/task_management}"
STATE_JSON="${STATE_JSON:-${TASK_MANAGEMENT_ROOT}/task_creation_state.json}"
TASK_MAP_CSV="${TASK_MAP_CSV:-${TASK_MANAGEMENT_ROOT}/task_map.csv}"

for value_name in APPLY SMOKE_ONLY; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
done
for required_dir in "${PHASE5_TEXTFREE_REVIEW_ROOT}" "${CORRECTED_SNAPSHOT_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory not found: ${required_dir}" >&2
    exit 1
  fi
done
if [[ ! -f "${EXCLUSION_MANIFEST}" ]]; then
  echo "Exclusion manifest not found: ${EXCLUSION_MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${CVAT_MASK_MODULE}" ]]; then
  echo "Required Stage2to4 module not found: ${CVAT_MASK_MODULE}" >&2
  echo "Copy pseudo3d/export/__init__.py and convert_masks_to_cvat_segmentation_mask_1_1.py to this checkout." >&2
  exit 1
fi

set -- \
  --review_root "${PHASE5_TEXTFREE_REVIEW_ROOT}" \
  --server_host "${CVAT_HOST}" \
  --reference_task_id 0 \
  --reference_video "${SMOKE_VIDEO}" \
  --full_video_standalone \
  --task_name_prefix "${TASK_NAME_PREFIX}" \
  --annotation_format "${CVAT_ANNOTATION_FORMAT}" \
  --annotation_snapshot_root "${CORRECTED_SNAPSHOT_ROOT}" \
  --exclusion_manifest "${EXCLUSION_MANIFEST}" \
  --expected_video_packages "${EXPECTED_VIDEO_PACKAGES}" \
  --expected_review_bboxes "${EXPECTED_REVIEW_BBOXES}" \
  --max_new_tasks "${MAX_NEW_TASKS}" \
  --state_json "${STATE_JSON}" \
  --task_map_csv "${TASK_MAP_CSV}"
if [[ "${APPLY}" == "1" ]]; then
  set -- "$@" --apply
fi
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  set -- "$@" --only_video "${SMOKE_VIDEO}"
fi
if [[ -n "${CVAT_USERNAME}" ]]; then
  set -- "$@" --username "${CVAT_USERNAME}"
fi
if [[ -n "${CVAT_ORGANIZATION_SLUG}" ]]; then
  set -- "$@" --organization_slug "${CVAT_ORGANIZATION_SLUG}"
fi

cd "${REPO_ROOT}"

echo "Stage 4 Phase 5 text-free CVAT Task launcher"
echo "  review root       : ${PHASE5_TEXTFREE_REVIEW_ROOT}"
echo "  corrected snapshot: ${CORRECTED_SNAPSHOT_ROOT}"
echo "  exclusion manifest: ${EXCLUSION_MANIFEST}"
echo "  CVAT host         : ${CVAT_HOST}"
echo "  task prefix       : ${TASK_NAME_PREFIX}"
echo "  package videos    : ${EXPECTED_VIDEO_PACKAGES}"
echo "  package BBoxes    : ${EXPECTED_REVIEW_BBOXES}"
echo "  smoke only        : ${SMOKE_ONLY}"
echo "  smoke video       : ${SMOKE_VIDEO}"
echo "  apply             : ${APPLY}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON}" -m pseudo3d.batch.export.batch_create_stage4_phase5_cvat_tasks \
  "$@"
