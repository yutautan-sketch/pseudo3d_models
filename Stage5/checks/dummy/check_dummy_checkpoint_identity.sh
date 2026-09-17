#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: synthetic coverage for check_stage5_checkpoint_identity.py --
# byte-identical checkpoints concluding without a load, differing files whose
# tensors match NOT being called a different model, genuinely different
# weights reported with per-key max_abs_diff, key/shape/dtype differences kept
# apart from value differences, --skip_load leaving the model question
# unanswered, and the audited run directory staying untouched.
# Uses torch on CPU with tiny synthetic checkpoints: no CUDA, no real run data.
#
#   bash checks/dummy/check_dummy_checkpoint_identity.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 checkpoint identity synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_checkpoint_identity.py"

echo "Done."
