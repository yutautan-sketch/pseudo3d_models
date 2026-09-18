#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: synthetic coverage for check_stage5_effective_run_config.py -- the
# runtime counterpart to the static env-passthrough test. Covers the real P3
# deviation (effective save_every=10 against an intended 5), missing keys and
# wrong values, a missing periodic checkpoint reported with what the run dir
# does hold, CLI exit codes, and the run directory staying untouched.
# Pure stdlib: no torch, no CUDA, no real run data.
#
#   bash checks/dummy/check_dummy_effective_run_config.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 effective run-config synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_effective_run_config.py"

echo "Done."
