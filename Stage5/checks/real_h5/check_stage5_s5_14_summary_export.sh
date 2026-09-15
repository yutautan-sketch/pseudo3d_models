#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H5: aggregation, shareable-bundle export, and completion
# summary. Consumes already-generated Step H2/H2.5/H3/H3.1/H4 outputs (all
# already video_alias-only) and writes frame_position_deciles.csv,
# stage5_s5_14_summary.json, a merged video_summary.csv, and a
# vote_count_error_rates.csv, then runs a privacy self-check over the whole
# bundle. CPU/CSV/JSON only, no CUDA.
#
#   bash checks/real_h5/check_stage5_s5_14_summary_export.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

GT_OVERLAP_EXPOSURE_CSV="${GT_OVERLAP_EXPOSURE_CSV:-${SCRIPT_DIR}/work_dirs/_frame_spatial_overlap_diagnostics/gt_overlap_exposure.csv}"
FRAME_METRICS_CSV="${FRAME_METRICS_CSV:-${SCRIPT_DIR}/work_dirs/_frame_xy_diagnostics/frame_metrics.csv}"
H3_VIDEO_SUMMARY_CSV="${H3_VIDEO_SUMMARY_CSV:-${SCRIPT_DIR}/work_dirs/_frame_xy_diagnostics/video_summary.csv}"
XY_DENSITY_BINS_CSV="${XY_DENSITY_BINS_CSV:-${SCRIPT_DIR}/work_dirs/_frame_xy_diagnostics/xy_density_bins.csv}"
XY_TEMPORAL_RECURRENCE_CSV="${XY_TEMPORAL_RECURRENCE_CSV:-${SCRIPT_DIR}/work_dirs/_frame_xy_diagnostics/xy_temporal_recurrence.csv}"
POINT_OVERLAP_ERROR_STATISTICS_CSV="${POINT_OVERLAP_ERROR_STATISTICS_CSV:-${SCRIPT_DIR}/work_dirs/_per_window_context_diagnostics/point_overlap_error_statistics.csv}"
H4_VIDEO_SUMMARY_CSV="${H4_VIDEO_SUMMARY_CSV:-${SCRIPT_DIR}/work_dirs/_per_window_context_diagnostics/video_summary.csv}"
H4_SUMMARY_JSON="${H4_SUMMARY_JSON:-${SCRIPT_DIR}/work_dirs/_per_window_context_diagnostics/stage5_s5_14_h4_summary.json}"
XY_COORDINATE_PROVENANCE_SUMMARY_JSON="${XY_COORDINATE_PROVENANCE_SUMMARY_JSON:-${SCRIPT_DIR}/work_dirs/_xy_coordinate_provenance/SHARE_THIS/stage4_xy_coordinate_provenance_summary.json}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/work_dirs/_s5_14_summary/SHARE_THIS}"

echo "Stage5 S5-14 Step H5 summary export"
echo "  python     : ${PYTHON}"
echo "  output dir : ${OUTPUT_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_s5_14_summary_export.py" \
  --gt_overlap_exposure_csv "${GT_OVERLAP_EXPOSURE_CSV}" \
  --frame_metrics_csv "${FRAME_METRICS_CSV}" \
  --h3_video_summary_csv "${H3_VIDEO_SUMMARY_CSV}" \
  --xy_density_bins_csv "${XY_DENSITY_BINS_CSV}" \
  --xy_temporal_recurrence_csv "${XY_TEMPORAL_RECURRENCE_CSV}" \
  --point_overlap_error_statistics_csv "${POINT_OVERLAP_ERROR_STATISTICS_CSV}" \
  --h4_video_summary_csv "${H4_VIDEO_SUMMARY_CSV}" \
  --h4_summary_json "${H4_SUMMARY_JSON}" \
  --xy_coordinate_provenance_summary_json "${XY_COORDINATE_PROVENANCE_SUMMARY_JSON}" \
  --output_dir "${OUTPUT_DIR}"

echo "Done."
echo "  shareable bundle: ${OUTPUT_DIR}"
