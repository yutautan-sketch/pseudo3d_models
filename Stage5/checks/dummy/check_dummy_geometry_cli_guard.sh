#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-17 S17-3: synthetic coverage of the geometry CLI (evaluate_stage5_geometry)
# -- a synthetic contract (pins, registry, manifest), synthetic teacher and
# intermediate H5s, NPZs, summary.json / h5_metrics.csv and a git-tracked
# synthetic launcher. Covers raw-array checks, exact intermediate resolution
# (no prefix match), purpose allow-lists, a sealed video refused before it is
# opened, coverage registration from names and git evidence only, hash
# tampering, missing coverage, missing NPZ without fallback, point/label/vote/
# confusion inconsistencies, mm mode refusal, and register -> audit -> run
# completing with a privacy-checked shared JSON and 0600 private records.
# numpy/h5py and git. No real data, no production pins, no torch, no CUDA.
#
#   bash checks/dummy/check_dummy_geometry_cli_guard.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-17 geometry CLI synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_geometry_cli_guard.py"

echo "Done."
