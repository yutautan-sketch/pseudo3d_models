#!/usr/bin/env bash
set -euo pipefail

# Create a shareable metrics bundle from an existing Stage5 evaluation run.

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

DATE="${DATE:-260711}"
EX_DATE="${EX_DATE:-260801}"
RUN_ROOT="${RUN_ROOT:-/mnt/data/3d_projects/stage5_runs}"
EVALUATION_ROOT="${EVALUATION_ROOT:-/mnt/data/3d_projects/stage5_evaluations}"
PREFIX="${PREFIX:-w16_s8_bboxrankv2_nobboxbg_glocal_ce_smooth00_auto_weight_lr1e3_ep150_bs8}"
DEFAULT_EXPERIMENT_NAME="pointnext_s_EX${EX_DATE}_${DATE}_${PREFIX}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${EX_DATE}/${DEFAULT_EXPERIMENT_NAME}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "${RUN_DIR}")}"
EVALUATION_OUTPUT_ROOT="${EVALUATION_OUTPUT_ROOT:-${EVALUATION_ROOT}/${EX_DATE}/${EXPERIMENT_NAME}}"

SHARE_OUTPUT_DIR="${SHARE_OUTPUT_DIR:-${EVALUATION_OUTPUT_ROOT}/anonymized_metrics_SHARE_THIS}"
PRIVATE_OUTPUT_DIR="${PRIVATE_OUTPUT_DIR:-${EVALUATION_OUTPUT_ROOT}/anonymized_metrics_private_DO_NOT_SHARE}"

if [[ ! -d "${EVALUATION_OUTPUT_ROOT}" ]]; then
  echo "Stage5 evaluation output directory not found: ${EVALUATION_OUTPUT_ROOT}" >&2
  exit 1
fi

echo "Stage5 anonymized metrics export"
echo "  evaluation root : ${EVALUATION_OUTPUT_ROOT}"
echo "  training run    : ${RUN_DIR}"
echo "  share output    : ${SHARE_OUTPUT_DIR}"
echo "  private mapping : ${PRIVATE_OUTPUT_DIR}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/export_anonymized_stage5_metrics.py"
  --evaluation_root "${EVALUATION_OUTPUT_ROOT}"
  --share_output_dir "${SHARE_OUTPUT_DIR}"
  --private_output_dir "${PRIVATE_OUTPUT_DIR}"
)
if [[ -d "${RUN_DIR}" ]]; then
  cmd+=(--training_run_dir "${RUN_DIR}")
else
  echo "Warning: training run not found; config.json/metrics.jsonl will be omitted." >&2
fi

"${cmd[@]}"

echo "Done. Share only: ${SHARE_OUTPUT_DIR}"
