#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-11 Step D2/D3: verify that the default (batchnorm) PointNeXt-S wrapper
# path is unchanged, and that pointnext_norm=groupnorm structurally replaces
# every normalization site 1:1 with the Stage5GroupNorm adapter, with
# identical train()/eval() behavior and a clean checkpoint round-trip.
#
# Synthetic self-test (CPU only, no CUDA/checkpoint/H5 required):
#   SELF_TEST=1 bash checks/dummy/check_dummy_pointnext_s_groupnorm.sh
#
# Full structural check (requires CUDA; dummy H5 data only):
#   bash checks/dummy/check_dummy_pointnext_s_groupnorm.sh
#
# Optional: also strict-load a real production BatchNorm checkpoint into the
# default wrapper path as an extra backward-compat check:
#   LEGACY_CHECKPOINT=/mnt/data/.../best.pt bash checks/dummy/check_dummy_pointnext_s_groupnorm.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SCRIPT_DIR}/external/PointNeXt:${SCRIPT_DIR}/external/PointNeXt/openpoints:${PYTHONPATH:-}"

SELF_TEST="${SELF_TEST:-0}"

if [[ "${SELF_TEST}" == "1" ]]; then
  echo "Stage5 PointNeXt-S GroupNorm self-test"
  "${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_pointnext_s_groupnorm.py" --self_test
  exit 0
fi

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/_dummy_pointnext_s_groupnorm_check}"
NORM_GROUPS="${NORM_GROUPS:-8}"
LEGACY_CHECKPOINT="${LEGACY_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"

echo "Stage5 PointNeXt-S GroupNorm structure check"
echo "  python            : ${PYTHON}"
echo "  work dir          : ${WORK_DIR}"
echo "  norm_groups       : ${NORM_GROUPS}"
echo "  legacy checkpoint : ${LEGACY_CHECKPOINT:-<none>}"
echo "  device            : ${DEVICE}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/checks/dummy/check_dummy_pointnext_s_groupnorm.py"
  --work_dir "${WORK_DIR}"
  --norm_groups "${NORM_GROUPS}"
  --device "${DEVICE}"
  --seed "${SEED}"
)
if [[ -n "${LEGACY_CHECKPOINT}" ]]; then
  cmd+=(--legacy_checkpoint "${LEGACY_CHECKPOINT}")
fi

"${cmd[@]}"

echo "Done."
echo "  summary: ${WORK_DIR}/groupnorm_structure_summary.json"
