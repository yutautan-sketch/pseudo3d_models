#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 verification (2): synthetic coverage for
# check_stage5_frame_failure_vs_gt_regions.py -- the exact binomial sign test,
# per-frame region counts and per-frame recall being computed per frame, ignore
# points leaving both sides of recall, a single-kind video being excluded from
# the paired test rather than counted as a tie, the pooled view staying labelled
# as confounded, point-weighted pooling, and an end-to-end CLI run that keeps
# the 1-based aliases and leaks no real video name. numpy/h5py only, no CUDA.
#
#   bash checks/dummy/check_dummy_frame_failure_vs_gt_regions.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 frame-failure-vs-GT-regions synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_frame_failure_vs_gt_regions.py"

echo "Done."
