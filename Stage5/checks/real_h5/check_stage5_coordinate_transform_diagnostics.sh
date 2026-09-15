#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-14補足2: coordinate-transform (prediction equivariance) diagnosis for
# the fixed 21 videos. 8 fixed conditions (identity/repeat_identity/±0.1 X
# or Y translation/±15deg Z rotation) applied to the model's own
# normalize_xyz() input XYZ, pre-window-extraction. identity/repeat_identity
# are verified against the saved W-A aggregate prediction with NO
# tolerance. No retraining. Requires CUDA/torch and the W-A checkpoint
# (21 videos x 8 conditions = 168 video-forward-equivalents).
#
#   CHECKPOINT=/path/to/w-a/last.pt \
#   EVALUATION_DIR=/path/to/.../<experiment>/last \
#   TRAIN_LIST=/path/to/w-a/train_files.txt \
#     bash checks/real_h5/check_stage5_coordinate_transform_diagnostics.sh
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
TRAIN_LIST="${TRAIN_LIST:?Set TRAIN_LIST to the W-A train file list (162 H5 paths), for the leave-one-video-out hot/cold prior}"
CHECKPOINT_ROW_NAME="${CHECKPOINT_ROW_NAME:-last}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
PIXEL_WIDTH="${PIXEL_WIDTH:-256}"
PIXEL_HEIGHT="${PIXEL_HEIGHT:-256}"
DEVICE="${DEVICE:-cuda}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_coordinate_transform_diagnostics}"
POINT_METRICS_CSV="${POINT_METRICS_CSV:-${WORK_DIR}/coordinate_transform_point_metrics.csv}"
HOT_COLD_METRICS_CSV="${HOT_COLD_METRICS_CSV:-${WORK_DIR}/coordinate_transform_hot_cold_metrics.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${WORK_DIR}/stage5_coordinate_transform_summary.json}"
DIFF_ARTIFACT_DIR="${DIFF_ARTIFACT_DIR:-${WORK_DIR}/private_DO_NOT_SHARE}"

mkdir -p "${WORK_DIR}"

echo "Stage5 S5-14 supplement2 coordinate-transform diagnosis"
echo "  python              : ${PYTHON}"
echo "  checkpoint          : ${CHECKPOINT}"
echo "  evaluation dir      : ${EVALUATION_DIR}"
echo "  train list          : ${TRAIN_LIST}"
echo "  pixel dims          : ${PIXEL_WIDTH} x ${PIXEL_HEIGHT}"
echo "  device              : ${DEVICE}"
echo "  conditions          : identity, repeat_identity, translate_x(+/-0.1), translate_y(+/-0.1), rotate_z(+/-15deg)"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_coordinate_transform_diagnostics.py" \
  --checkpoint "${CHECKPOINT}" \
  --evaluation_dir "${EVALUATION_DIR}" \
  --checkpoint_row_name "${CHECKPOINT_ROW_NAME}" \
  --train_list "${TRAIN_LIST}" \
  --ignore_index "${IGNORE_INDEX}" \
  --pixel_width "${PIXEL_WIDTH}" \
  --pixel_height "${PIXEL_HEIGHT}" \
  --device "${DEVICE}" \
  --strict_checkpoint \
  --point_metrics_csv "${POINT_METRICS_CSV}" \
  --hot_cold_metrics_csv "${HOT_COLD_METRICS_CSV}" \
  --summary_json "${SUMMARY_JSON}" \
  --diff_artifact_dir "${DIFF_ARTIFACT_DIR}"

echo "Done."
echo "  point metrics csv    : ${POINT_METRICS_CSV}"
echo "  hot/cold metrics csv : ${HOT_COLD_METRICS_CSV}"
echo "  summary json         : ${SUMMARY_JSON}"
echo "  diff artifact dir    : ${DIFF_ARTIFACT_DIR} (only written if an identity/repeat_identity parity mismatch occurs)"
