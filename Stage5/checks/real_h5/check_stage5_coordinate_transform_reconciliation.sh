#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14補足3: CPU-only reconciliation of check_stage5_coordinate_transform_
# diagnostics.py's already-saved coordinate_transform_point_metrics.csv.
# No re-inference, no new training, no new conditions/videos. Produces
# split-pooled (point-weighted) confusion-derived precision/recall/FPR/
# F1/IoU, and separates median-of-per-video-diffs from diff-of-split-
# medians for F1/IoU. Pure CSV/JSON, no CUDA.
#
#   POINT_METRICS_CSV=/path/to/coordinate_transform_point_metrics.csv \
#     bash checks/real_h5/check_stage5_coordinate_transform_reconciliation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

POINT_METRICS_CSV="${POINT_METRICS_CSV:?Set POINT_METRICS_CSV to the coordinate_transform_point_metrics.csv output of check_stage5_coordinate_transform_diagnostics.py}"
EXPECTED_TRAIN_SANITY_VIDEOS="${EXPECTED_TRAIN_SANITY_VIDEOS:-3}"
EXPECTED_VALIDATION_VIDEOS="${EXPECTED_VALIDATION_VIDEOS:-18}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_coordinate_transform_diagnostics}"
SPLIT_POOLED_METRICS_CSV="${SPLIT_POOLED_METRICS_CSV:-${WORK_DIR}/coordinate_transform_split_pooled_metrics.csv}"
MEDIAN_DIFF_METRICS_CSV="${MEDIAN_DIFF_METRICS_CSV:-${WORK_DIR}/coordinate_transform_median_diff_metrics.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORK_DIR}/stage5_coordinate_transform_reconciliation_summary.json}"

mkdir -p "${WORK_DIR}"

echo "Stage5 S5-14 supplement3 coordinate-transform reconciliation"
echo "  python             : ${PYTHON}"
echo "  point metrics csv  : ${POINT_METRICS_CSV}"
echo "  expected videos    : train_sanity=${EXPECTED_TRAIN_SANITY_VIDEOS} validation=${EXPECTED_VALIDATION_VIDEOS}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_coordinate_transform_reconciliation.py" \
  --point_metrics_csv "${POINT_METRICS_CSV}" \
  --expected_train_sanity_videos "${EXPECTED_TRAIN_SANITY_VIDEOS}" \
  --expected_validation_videos "${EXPECTED_VALIDATION_VIDEOS}" \
  --split_pooled_metrics_csv "${SPLIT_POOLED_METRICS_CSV}" \
  --median_diff_metrics_csv "${MEDIAN_DIFF_METRICS_CSV}" \
  --summary_json "${SUMMARY_JSON}"

echo "Done."
echo "  split pooled metrics csv : ${SPLIT_POOLED_METRICS_CSV}"
echo "  median diff metrics csv  : ${MEDIAN_DIFF_METRICS_CSV}"
echo "  summary json              : ${SUMMARY_JSON}"
