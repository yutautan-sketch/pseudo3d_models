#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the S0-1 input audit -- the expected
# input hashes being fixed under the labels the split builder looks them up by,
# every unconfirmed input stopping AND writing no record (so nothing reaches
# the builder wearing a fixed hash), three matching aliases with the wrong
# video identities being rejected, re-derivation refusing without a pinned list
# fingerprint, the FL table's structure confirmed without any value being read
# out, and the stdlib re-derivation agreeing with evaluate_stage5.py's own copy
# (extracted by text so the two cannot drift). Pure stdlib; synthetic
# H5-named files only, none of them opened.
#
#   bash checks/dummy/check_dummy_step0_input_audit.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-16 Step 0 S0-1 input audit synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_step0_input_audit.py"

echo "Done."
