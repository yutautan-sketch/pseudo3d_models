#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the deterministic half of the B' split
# -- clinical FL intake where only a matched row carrying a declared token is
# missing (an absent, duplicated or unmatched row is an input defect that
# stops), the three FL groups with the remainder rule fixed in advance, ties
# broken by the saved input order, largest-remainder allocation with capacity
# and no randomness, and a seeded draw derived from sha256(seed|identity) so it
# does not depend on an RNG implementation. Pure stdlib, no real FL values.
#
#   bash checks/dummy/check_dummy_bprime_split_allocation.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

echo "Stage5 S5-16 Step 0 B' split allocation synthetic test"
echo "  python : ${PYTHON}"

"${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_bprime_split_allocation.py"

echo "Done."
