#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Verify real-H5 Dataset / overlap-window / batch integrity.
#
# The Python checker reads the saved run config and split lists, then checks:
#   - every window against the source H5 arrays
#   - point_indices, labels, masks, features, and xyz point order
#   - collate padding and sample separation
#   - one shuffled DataLoader epoch with the training worker count
#   - representative same-H5, cross-H5, and min/max-size batches
#
# Outputs contain train_000 / val_000 aliases and hashes, not source names.
# ------------------------------------------------------------

STAGE5_DIR="${STAGE5_DIR:-/mnt/data/3d_projects/models/Stage5}"
cd "${STAGE5_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${STAGE5_DIR}:${PYTHONPATH:-}"

# ------------------------------------------------------------
# Run and output paths
# Keep the historical teacher-v2 run as the default so the report's integrity
# result remains reproducible. Override RUN_DIR to inspect a teacher-v6 run.
# ------------------------------------------------------------
RUN_DIR="${RUN_DIR:-/mnt/data/3d_projects/stage5_runs/260801/pointnext_s_EX260801_260711_w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8}"
RUN_NAME="$(basename "${RUN_DIR}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/3d_projects/stage5_debug/batch_integrity}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

# ------------------------------------------------------------
# Check settings
# Empty overrides use the values saved in RUN_DIR/config.json.
# ------------------------------------------------------------
SPLITS="${SPLITS:-train,val}"
MAX_FILES_PER_SPLIT="${MAX_FILES_PER_SPLIT:-0}"
RAW_POINT_CHECKS="${RAW_POINT_CHECKS:-3}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/config.json" ]]; then
  echo "Run config not found: ${RUN_DIR}/config.json" >&2
  exit 1
fi
if [[ "${SPLITS}" == *"train"* && ! -f "${RUN_DIR}/train_files.txt" ]]; then
  echo "Train split list not found: ${RUN_DIR}/train_files.txt" >&2
  exit 1
fi
if [[ "${SPLITS}" == *"val"* && ! -f "${RUN_DIR}/val_files.txt" ]]; then
  echo "Validation split list not found: ${RUN_DIR}/val_files.txt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Stage5 real-H5 batch integrity check"
echo "  python              : ${PYTHON}"
echo "  run dir             : ${RUN_DIR}"
echo "  splits              : ${SPLITS}"
echo "  output dir          : ${OUTPUT_DIR}"
echo "  max files/split     : ${MAX_FILES_PER_SPLIT} (0 means all)"
echo "  raw point checks    : ${RAW_POINT_CHECKS}"
echo "  batch override      : ${BATCH_SIZE_OVERRIDE:-saved run config}"
echo "  workers override    : ${NUM_WORKERS_OVERRIDE:-saved run config}"

cmd=(
  "${PYTHON}" "${STAGE5_DIR}/checks/real_h5/check_stage5_batch_integrity.py"
  --run_dir "${RUN_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --splits "${SPLITS}"
  --max_files_per_split "${MAX_FILES_PER_SPLIT}"
  --raw_point_checks "${RAW_POINT_CHECKS}"
)

if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
  cmd+=(--batch_size "${BATCH_SIZE_OVERRIDE}")
fi
if [[ -n "${NUM_WORKERS_OVERRIDE}" ]]; then
  cmd+=(--num_workers "${NUM_WORKERS_OVERRIDE}")
fi

"${cmd[@]}"

echo "Done."
echo "  summary              : ${OUTPUT_DIR}/integrity_summary.json"
echo "  train sample hashes  : ${OUTPUT_DIR}/train_sample_integrity.csv"
echo "  train batch report   : ${OUTPUT_DIR}/train_batch_integrity.csv"
echo "  val sample hashes    : ${OUTPUT_DIR}/val_sample_integrity.csv"
echo "  val batch report     : ${OUTPUT_DIR}/val_batch_integrity.csv"
