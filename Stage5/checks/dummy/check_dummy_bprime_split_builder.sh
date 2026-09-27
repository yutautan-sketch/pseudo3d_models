#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-16 Step 0: end-to-end coverage for the tool that performs the irreversible
# split. Over synthetic H5s: the dry run releasing the clinical-FL marginal and
# NOTHING else (no multi-region marginal, no cross table, no per-cell
# allocation -- together those reconstruct internal_test's composition); sealed
# material written into the sealed directory at mode 0600 under a 0700 dir;
# --confirm refusing an unverified or tampered intermediate and an input that
# drifted from the audit record; the six partition checks holding on real
# output; pinning the result and watching the ordinary guard deny the sealed
# videos; and the initial build path refusing to run twice. numpy/h5py, no
# torch, no CUDA, no real data.
#
#   bash checks/dummy/check_dummy_bprime_split_builder.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-16 Step 0 bprime_split_builder synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_bprime_split_builder.py"

echo "Done."
