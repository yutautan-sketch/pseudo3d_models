#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"

PHASE5_FULLVIDEO_REVIEW_ROOT="${PHASE5_FULLVIDEO_REVIEW_ROOT:-/Users/yutakodaira/Desktop/manual_review_cvat_phase5_fullvideo_v2}"
CVAT_HOST="${CVAT_HOST:-http://localhost:8080}"
CVAT_USERNAME="${CVAT_USERNAME:-}"
CVAT_ORGANIZATION_SLUG="${CVAT_ORGANIZATION_SLUG:-}"
CVAT_ANNOTATION_FORMAT="${CVAT_ANNOTATION_FORMAT:-Segmentation mask 1.1}"
EXPECTED_TASKS="${EXPECTED_TASKS:-60}"
ONLY_VIDEO="${ONLY_VIDEO:-}"
MAX_TASKS="${MAX_TASKS:-0}"
APPLY="${APPLY:-0}"

TASK_MAP_CSV="${TASK_MAP_CSV:-${PHASE5_FULLVIDEO_REVIEW_ROOT}/task_management/task_map.csv}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${PHASE5_FULLVIDEO_REVIEW_ROOT}/cvat_exports_before_textfree_v1}"

if [[ "${APPLY}" != "0" && "${APPLY}" != "1" ]]; then
  echo "APPLY must be 0 or 1, got: ${APPLY}" >&2
  exit 1
fi
if [[ ! -d "${PHASE5_FULLVIDEO_REVIEW_ROOT}" ]]; then
  echo "Full-video review root not found: ${PHASE5_FULLVIDEO_REVIEW_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${TASK_MAP_CSV}" ]]; then
  echo "Task map not found: ${TASK_MAP_CSV}" >&2
  exit 1
fi

set -- \
  --review_root "${PHASE5_FULLVIDEO_REVIEW_ROOT}" \
  --server_host "${CVAT_HOST}" \
  --task_map_csv "${TASK_MAP_CSV}" \
  --output_root "${SNAPSHOT_ROOT}" \
  --annotation_format "${CVAT_ANNOTATION_FORMAT}" \
  --expected_tasks "${EXPECTED_TASKS}" \
  --max_tasks "${MAX_TASKS}"
if [[ -n "${CVAT_USERNAME}" ]]; then
  set -- "$@" --username "${CVAT_USERNAME}"
fi
if [[ -n "${CVAT_ORGANIZATION_SLUG}" ]]; then
  set -- "$@" --organization_slug "${CVAT_ORGANIZATION_SLUG}"
fi
if [[ -n "${ONLY_VIDEO}" ]]; then
  set -- "$@" --only_video "${ONLY_VIDEO}"
fi
if [[ "${APPLY}" == "1" ]]; then
  set -- "$@" --apply
fi

cd "${REPO_ROOT}"

echo "Stage 4 Phase 5 corrected CVAT snapshot launcher"
echo "  review root      : ${PHASE5_FULLVIDEO_REVIEW_ROOT}"
echo "  CVAT host        : ${CVAT_HOST}"
echo "  annotation format: ${CVAT_ANNOTATION_FORMAT}"
echo "  expected tasks   : ${EXPECTED_TASKS}"
echo "  only video       : ${ONLY_VIDEO:-all}"
echo "  max tasks        : ${MAX_TASKS}"
echo "  apply            : ${APPLY}"
echo "  task map         : ${TASK_MAP_CSV}"
echo "  snapshot root    : ${SNAPSHOT_ROOT}"

"${PYTHON}" \
  pseudo3d/batch/export/batch_export_stage4_phase5_cvat_task_snapshots.py \
  "$@"
