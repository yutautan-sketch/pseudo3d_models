#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the B' seal contract -- a missing,
# unapproved or hash-mismatched contract stopping instead of allowing, the
# seal registry denying before any manifest allows (legacy included), the pin
# outside the artifact being the expected hash, artifacts whose video
# membership cannot be established being refused, the initial build path being
# closed once a registry is approved, refusals carrying no video ID or path,
# and the six partition checks. Pure stdlib -- no numpy/h5py/torch, and no H5
# is opened anywhere: sealed reads are proven to stop before the open by
# refusing paths that do not exist.
#
#   bash checks/dummy/check_dummy_split_contract.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-16 Step 0 split contract synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_split_contract.py"

echo "Done."
