#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14 Step H4: per-window context diagnostics for the fixed 21 videos.
# Performs the ONE re-inference this stage allows (per-window forward via
# the S5-08 OverlapAccumulator) and verifies it against the saved W-A
# aggregate prediction with NO tolerance -- any mismatch (including CUDA
# non-determinism) stops the run rather than being silently accepted.
# Requires CUDA/torch and the W-A checkpoint.
#
#   bash checks/real_h5/check_stage5_per_window_context_diagnostics.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"
TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SCRIPT_DIR}/external/PointNeXt:${SCRIPT_DIR}/external/PointNeXt/openpoints:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the W-A epoch-5 checkpoint (e.g. .../EX260914_..._nopad/last.pt)}"
EVALUATION_DIR="${EVALUATION_DIR:?Set EVALUATION_DIR to an evaluate_stage5.py per-checkpoint output_dir (contains h5_metrics.csv and predictions/)}"
CHECKPOINT_ROW_NAME="${CHECKPOINT_ROW_NAME:-last}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
PIXEL_WIDTH="${PIXEL_WIDTH:-256}"
PIXEL_HEIGHT="${PIXEL_HEIGHT:-256}"
GRID_RESOLUTION="${GRID_RESOLUTION:-16}"
DEVICE="${DEVICE:-cuda}"
XY_TEMPORAL_RECURRENCE_CSV="${XY_TEMPORAL_RECURRENCE_CSV:-}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_per_window_context_diagnostics}"
STRATIFIED_SUMMARY_CSV="${STRATIFIED_SUMMARY_CSV:-${WORK_DIR}/point_overlap_error_statistics.csv}"
BIN_EXPOSURE_CSV="${BIN_EXPOSURE_CSV:-${WORK_DIR}/bin_exposure_disagreement.csv}"
VIDEO_SUMMARY_CSV="${VIDEO_SUMMARY_CSV:-${WORK_DIR}/video_summary.csv}"
RECURRENCE_CROSS_CHECK_JSON="${RECURRENCE_CROSS_CHECK_JSON:-${WORK_DIR}/recurrence_exposure_cross_check.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORK_DIR}/stage5_s5_14_h4_summary.json}"
DIFF_ARTIFACT_DIR="${DIFF_ARTIFACT_DIR:-${WORK_DIR}/private_DO_NOT_SHARE}"

mkdir -p "${WORK_DIR}"

echo "Stage5 S5-14 Step H4 per-window context diagnostics"
echo "  python                    : ${PYTHON}"
echo "  checkpoint                : ${CHECKPOINT}"
echo "  evaluation dir            : ${EVALUATION_DIR}"
echo "  pixel dims                : ${PIXEL_WIDTH} x ${PIXEL_HEIGHT}"
echo "  grid resolution           : ${GRID_RESOLUTION}"
echo "  device                    : ${DEVICE}"
echo "  xy temporal recurrence csv: ${XY_TEMPORAL_RECURRENCE_CSV:-<not provided, cross-check skipped>}"

args=(
  "${SCRIPT_DIR}/checks/real_h5/check_stage5_per_window_context_diagnostics.py"
  --checkpoint "${CHECKPOINT}"
  --evaluation_dir "${EVALUATION_DIR}"
  --checkpoint_row_name "${CHECKPOINT_ROW_NAME}"
  --ignore_index "${IGNORE_INDEX}"
  --pixel_width "${PIXEL_WIDTH}"
  --pixel_height "${PIXEL_HEIGHT}"
  --grid_resolution "${GRID_RESOLUTION}"
  --device "${DEVICE}"
  --strict_checkpoint
  --stratified_summary_csv "${STRATIFIED_SUMMARY_CSV}"
  --bin_exposure_csv "${BIN_EXPOSURE_CSV}"
  --video_summary_csv "${VIDEO_SUMMARY_CSV}"
  --recurrence_cross_check_json "${RECURRENCE_CROSS_CHECK_JSON}"
  --summary_json "${SUMMARY_JSON}"
  --diff_artifact_dir "${DIFF_ARTIFACT_DIR}"
)
if [[ -n "${XY_TEMPORAL_RECURRENCE_CSV}" ]]; then
  args+=(--xy_temporal_recurrence_csv "${XY_TEMPORAL_RECURRENCE_CSV}")
fi

"${PYTHON}" "${args[@]}"

echo "Done."
echo "  stratified summary csv : ${STRATIFIED_SUMMARY_CSV}"
echo "  bin exposure csv       : ${BIN_EXPOSURE_CSV}"
echo "  video summary csv      : ${VIDEO_SUMMARY_CSV}"
echo "  summary json           : ${SUMMARY_JSON}"
echo "  diff artifact dir      : ${DIFF_ARTIFACT_DIR} (only written if a parity mismatch occurs)"
