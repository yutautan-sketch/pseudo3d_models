#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-17 S17-1/S17-3: deterministic synthetic coverage of the geometry core --
# crop inverse against hand-computed coordinates and the Stage 4 forward
# formula, anisotropic mm conversion, known rectangle/line geometry, degenerate
# and ambiguous instances, sign- and endpoint-swap invariance, the (b) box and
# (c) endpoint band agreeing on an asymmetric minor axis, component parity with
# the S5-15 implementation, (d) GT / practical / oracle selection kept apart,
# P1-P3, gated Hungarian matching, frame presence including TN, undefined vs 0,
# FP classes with explicit denominators, the 15 perturbations, the stability
# and primary-selection rules, held-out split conditions, the prior built and
# mapped back in normalised coordinates across frame sizes and aspect ratios, prior
# self-exclusion and the input/method denominator breakdown.
# numpy only, no H5, no pins, no real data, no torch, no CUDA.
#
#   bash checks/dummy/check_dummy_geometry_core.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-17 geometry core synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_geometry_core.py"

echo "Done."
