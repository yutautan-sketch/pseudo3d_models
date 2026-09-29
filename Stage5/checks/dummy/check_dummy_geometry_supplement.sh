#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-17 S17-6: synthetic coverage of supplement_stage5_geometry_aggregates.py
# -- registration of the S17-5 private record from metadata only (fixed shared
# run hash, clean pixel run, exit status, 0600, mtime window) and each of its
# refusals; aggregation only after the coverage check; exact re-verification
# of alias / identity / split and of recomputed shared aggregates; known values
# of the four supplementary aggregates, denominators, undefined counts and
# group separation; inputs left unchanged; privacy-checked finite output.
# Synthetic data only. No H5, no real data, no production pins.
#
#   bash checks/dummy/check_dummy_geometry_supplement.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-17 geometry supplement synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_geometry_supplement.py"

echo "Done."
