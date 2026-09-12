#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

AUTO_RUN_NAME="${AUTO_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1}"
AUTO_RUN_ROOT="${AUTO_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${AUTO_RUN_NAME}}"
SOURCE_REVIEW_ROOT="${SOURCE_REVIEW_ROOT:-${AUTO_RUN_ROOT}/manual_review_cvat_phase5_fullvideo_v2}"
TEXTFREE_REVIEW_ROOT="${TEXTFREE_REVIEW_ROOT:-${AUTO_RUN_ROOT}/manual_review_cvat_phase5_fullvideo_v3_textfree}"

EXPECTED_VIDEO_PACKAGES="${EXPECTED_VIDEO_PACKAGES:-60}"
EXPECTED_REVIEW_BBOXES="${EXPECTED_REVIEW_BBOXES:-129}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1, got: ${OVERWRITE}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_REVIEW_ROOT}" ]]; then
  echo "Source full-video review root not found: ${SOURCE_REVIEW_ROOT}" >&2
  exit 1
fi

set -- \
  --source_review_root "${SOURCE_REVIEW_ROOT}" \
  --output_root "${TEXTFREE_REVIEW_ROOT}" \
  --expected_video_packages "${EXPECTED_VIDEO_PACKAGES}" \
  --expected_review_bboxes "${EXPECTED_REVIEW_BBOXES}"
if [[ "${OVERWRITE}" == "1" ]]; then
  set -- "$@" --overwrite
fi

cd "${REPO_ROOT}"

echo "Stage 4 Phase 5 full-video text-free review migration"
echo "  source review root : ${SOURCE_REVIEW_ROOT}"
echo "  output review root : ${TEXTFREE_REVIEW_ROOT}"
echo "  expected videos    : ${EXPECTED_VIDEO_PACKAGES}"
echo "  expected BBoxes    : ${EXPECTED_REVIEW_BBOXES}"
echo "  overwrite          : ${OVERWRITE}"

"${PYTHON}" \
  pseudo3d/batch/export/rebuild_stage4_phase5_textfree_review_package.py \
  "$@"

echo
echo "Stage 4 Phase 5 text-free package complete."
echo "  summary      : ${TEXTFREE_REVIEW_ROOT}/textfree_migration_summary.json"
echo "  video tasks  : ${TEXTFREE_REVIEW_ROOT}/videos"
echo "  Mac manifest : ${TEXTFREE_REVIEW_ROOT}/mac_transfer_manifest.csv"
