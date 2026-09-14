#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-13: config-parity and completion checks between the W-A (control,
# strong auto-derived weight) and W-B (moderate fixed [0.5, 1.5]) class
# weight ablation training runs. CPU/JSON only -- no GPU required, reads
# each run directory's config.json/train_files.txt/val_files.txt/
# history.json (and, optionally, the two evaluate_stage5.py h5_metrics.csv
# outputs). Only writes machine-readable JSON locally; no path or
# video-name leakage beyond what the run directories themselves already
# contain (private, not for sharing as-is).
#
#   RUN_A_DIR=/path/to/w-a/run RUN_B_DIR=/path/to/w-b/run \
#     bash checks/real_h5/check_stage5_class_weight_ablation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

RUN_A_DIR="${RUN_A_DIR:?Set RUN_A_DIR to the W-A (control) train_stage5.py output_dir}"
RUN_B_DIR="${RUN_B_DIR:?Set RUN_B_DIR to the W-B train_stage5.py output_dir}"
EXPECTED_WEIGHT_A="${EXPECTED_WEIGHT_A:-0.05963856,1.94036150}"
EXPECTED_WEIGHT_B="${EXPECTED_WEIGHT_B:-0.5,1.5}"
EPOCHS="${EPOCHS:-5}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-last.pt}"
SKIP_CHECKPOINT_HASH="${SKIP_CHECKPOINT_HASH:-0}"
EVAL_CSV_A="${EVAL_CSV_A:-}"
EVAL_CSV_B="${EVAL_CSV_B:-}"
OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/work_dirs/_class_weight_ablation/config_parity.json}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "Stage5 class-weight ablation config-parity check"
echo "  python         : ${PYTHON}"
echo "  run A (W-A)    : ${RUN_A_DIR}"
echo "  run B (W-B)    : ${RUN_B_DIR}"
echo "  expected wt A  : ${EXPECTED_WEIGHT_A}"
echo "  expected wt B  : ${EXPECTED_WEIGHT_B}"
echo "  epochs         : ${EPOCHS}"

args=(
  "${SCRIPT_DIR}/checks/real_h5/check_stage5_class_weight_ablation.py"
  --run_a_dir "${RUN_A_DIR}"
  --run_b_dir "${RUN_B_DIR}"
  --expected_weight_a "${EXPECTED_WEIGHT_A}"
  --expected_weight_b "${EXPECTED_WEIGHT_B}"
  --epochs "${EPOCHS}"
  --checkpoint_filename "${CHECKPOINT_FILENAME}"
  --output_json "${OUTPUT_JSON}"
)

if [[ "${SKIP_CHECKPOINT_HASH}" == "1" ]]; then
  args+=(--skip_checkpoint_hash)
fi
if [[ -n "${EVAL_CSV_A}" && -n "${EVAL_CSV_B}" ]]; then
  args+=(--eval_csv_a "${EVAL_CSV_A}" --eval_csv_b "${EVAL_CSV_B}")
fi

"${PYTHON}" "${args[@]}"

echo "Done."
echo "  output: ${OUTPUT_JSON}"
