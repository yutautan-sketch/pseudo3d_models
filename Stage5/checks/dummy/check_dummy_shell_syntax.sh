#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Runs `bash -n` over every shell script under Stage5, so a syntax error
# cannot reach the real machine.
#
# This exists because one specific mistake has now been made twice: an
# apostrophe inside a ${VAR:?message} expansion. Bash parses the message as
# shell text even inside double quotes, so the apostrophe opens a quote that
# is never closed and the whole file fails to parse -- with an error pointing
# at the end of the file rather than at the apostrophe. It was first hit in
# S5-14 (check_stage5_coordinate_transform_reconciliation.sh) and again in
# S5-15 (check_stage5_gt_component_count.sh). Writing "the X of the Y" instead
# of "Y's X" avoids it.
#
# Pure bash: no python, no data, no CUDA.
#
#   bash checks/dummy/check_dummy_shell_syntax.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

failures=0
checked=0

while IFS= read -r script; do
  checked=$((checked + 1))
  if ! error="$(bash -n "${script}" 2>&1)"; then
    echo "FAIL: ${script}" >&2
    echo "  ${error}" >&2
    failures=$((failures + 1))
  fi
done < <(find . -name "*.sh" -type f -not -path "./external/*" | sort)

echo "Stage5 shell syntax check"
echo "  scripts checked : ${checked}"
echo "  failures        : ${failures}"

if [[ "${failures}" -ne 0 ]]; then
  echo "bash -n failed for ${failures} script(s)." >&2
  echo "If the error points at the end of the file, look for an apostrophe inside a \${VAR:?message}." >&2
  exit 1
fi

# Guard the specific pattern directly: an apostrophe inside ${...:?...} or
# ${...:-...} is legal in some positions but has bitten twice, so flag it.
suspicious="$(grep -rn "\${[A-Z_][A-Z0-9_]*:[?-][^}]*'" --include="*.sh" . | grep -v "^./external/" || true)"
if [[ -n "${suspicious}" ]]; then
  echo "FAIL: apostrophe inside a \${VAR:?...} / \${VAR:-...} expansion:" >&2
  echo "${suspicious}" >&2
  exit 1
fi
echo "  apostrophe-in-expansion: none found"
echo "Done."
