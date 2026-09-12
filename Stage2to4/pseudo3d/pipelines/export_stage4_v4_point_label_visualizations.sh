#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711}"
RUN_NAME="${RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v4_manual_fullvideo_v1}"
RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/${RUN_NAME}}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1}"
PHASE5_RETURN_ROOT="${PHASE5_RETURN_ROOT:-${DATASET_ROOT}/cvat_returns/phase5_fullvideo_v3_textfree_reviewed_snapshot_59_v2/manual_review_cvat_phase5_fullvideo_v3_textfree}"
CVAT_REVIEW_ROOT="${CVAT_REVIEW_ROOT:-${SOURCE_RUN_ROOT}/manual_review_cvat_phase5_fullvideo_v3_textfree}"
CVAT_SNAPSHOT_ROOT="${CVAT_SNAPSHOT_ROOT:-${PHASE5_RETURN_ROOT}/cvat_exports_after_textfree_review_v2}"
ANNOTATED_DIR="${ANNOTATED_DIR:-${RUN_ROOT}/annotated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/annotation_textures_v4_labels}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/summary.csv}"
ANNOTATED_PATTERN="${ANNOTATED_PATTERN:-*_pointcloud_annotated_*bboxrank_v4_manual_fullvideo_v1.h5}"
EXPECTED_FILES="${EXPECTED_FILES:-181}"
VIDEO_NAMES="${VIDEO_NAMES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${ANNOTATED_DIR}" ]]; then
  echo "Annotated H5 directory not found: ${ANNOTATED_DIR}" >&2
  exit 1
fi
for required_dir in "${CVAT_REVIEW_ROOT}" "${CVAT_SNAPSHOT_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required CVAT directory not found: ${required_dir}" >&2
    exit 1
  fi
done
if [[ "${SKIP_EXISTING}" != "0" && "${SKIP_EXISTING}" != "1" ]]; then
  echo "SKIP_EXISTING must be 0 or 1, got: ${SKIP_EXISTING}" >&2
  exit 1
fi

mode_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  mode_args+=(--skip_existing)
else
  mode_args+=(--overwrite)
fi
selection_args=()
count_args=(--expected_files "${EXPECTED_FILES}")
if [[ -n "${VIDEO_NAMES}" ]]; then
  selection_args+=(--video_names "${VIDEO_NAMES}")
  count_args=()
fi

cd "${REPO_ROOT}"

"${PYTHON}" \
  pseudo3d/batch/export/batch_export_stage4_point_label_visualization.py \
  --annotated_dir "${ANNOTATED_DIR}" \
  --pattern "${ANNOTATED_PATTERN}" \
  --recursive \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${SUMMARY_CSV}" \
  --point_radius 1 \
  --point_alpha 0.85 \
  --positive_color "255,32,32" \
  --ignore_color "255,210,0" \
  --background_color "80,140,200" \
  --bbox_color "160,255,80" \
  --bbox_thickness 1 \
  --cvat_review_root "${CVAT_REVIEW_ROOT}" \
  --cvat_snapshot_root "${CVAT_SNAPSHOT_ROOT}" \
  --cvat_mask_color "0,255,255" \
  --cvat_mask_alpha 0.35 \
  "${selection_args[@]}" \
  "${count_args[@]}" \
  "${mode_args[@]}" \
  --continue_on_error

echo
echo "Stage 4 teacher v4 saved point-label visualization complete."
echo "  input  : ${ANNOTATED_DIR}"
echo "  output : ${OUTPUT_ROOT}"
echo "  summary: ${SUMMARY_CSV}"
