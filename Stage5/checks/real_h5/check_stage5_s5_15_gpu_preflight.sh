#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 limited GPU preflight. Three stages, selected with STAGE:
#
#   STAGE=a  fixed-list/bash path. Runs train_stage5.sh with PREFLIGHT_ONLY=1
#            over the real 180-file lists: no GPU, no training. This is the
#            only way to exercise train_stage5.sh's real path, because its
#            teacher preflight asserts dataset-wide totals that a small subset
#            cannot satisfy.
#   STAGE=b  synthetic forward/backward on the GPU. No H5, no Dataset.
#   STAGE=c  a few real-H5 steps over a fixed head-of-list subset, run twice
#            (augmentation=none, then random_z_rotation), 2 epochs each.
#
# Total when all stages run: 1 CPU pass + 3 GPU invocations, roughly 12
# optimizer steps of training-equivalent compute. Nothing is written under
# stage5_runs/, and no arm training is started.
#
#   STAGE=a bash checks/real_h5/check_stage5_s5_15_gpu_preflight.sh
#   STAGE=b bash checks/real_h5/check_stage5_s5_15_gpu_preflight.sh
#   STAGE=c bash checks/real_h5/check_stage5_s5_15_gpu_preflight.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
CONDA_PREFIX="${CONDA_PREFIX:-/home/kodaira/anaconda3/envs/dualtrack311}"
export CONDA_PREFIX
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"

# ------------------------------------------------------------
# Fixed conditions (S5-15 P1 table, audited 2026-09-16).
# ------------------------------------------------------------
INIT_CHECKPOINT="/mnt/data/3d_projects/models/Stage5/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init_groupnorm.pt"
EXPECTED_INIT_SHA256="55ec6e6bcb39d58f398719b33826e80715a94bc6e7670d5b88623cd7c668438b"
WA_RUN_DIR="/mnt/data/3d_projects/stage5_runs/260914/pointnext_s_EX260914_260711_w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_auto_weight_lr1e3_ep5_bs1_acc8_nopad"
TRAIN_LIST="${TRAIN_LIST:-${WA_RUN_DIR}/train_files.txt}"
VAL_LIST="${VAL_LIST:-${WA_RUN_DIR}/val_files.txt}"

NUM_TRAIN_VIDEOS="4"
NUM_VAL_VIDEOS="1"
EPOCHS="2"
NUM_WORKERS="2"
ROTATION_DEGREES="15.0"
SEED="42"
DEVICE="${DEVICE:-cuda}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/work_dirs/_s5_15_gpu_preflight}"

STAGE="${STAGE:?Set STAGE to a (fixed-list path, no GPU), b (dummy forward/backward) or c (real H5 few steps)}"

if [[ "${STAGE}" != "a" ]]; then
  TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
  export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Initialization checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 1
fi
actual_init_sha256="$(sha256sum "${INIT_CHECKPOINT}" | cut -d' ' -f1)"
if [[ "${actual_init_sha256}" != "${EXPECTED_INIT_SHA256}" ]]; then
  echo "Initialization checkpoint SHA-256 mismatch:" >&2
  echo "  expected: ${EXPECTED_INIT_SHA256}" >&2
  echo "  actual  : ${actual_init_sha256}" >&2
  exit 1
fi

echo "Stage5 S5-15 GPU preflight (stage ${STAGE})"
echo "  python    : ${PYTHON}"
echo "  init ckpt : ${actual_init_sha256} (matches the audited value)"

case "${STAGE}" in
  a)
    # PREFLIGHT_ONLY=1 makes train_stage5.sh stop before building any model.
    echo "  mode      : fixed-list path check via train_stage5.sh PREFLIGHT_ONLY=1 (no GPU, no training)"
    echo "  train list: ${TRAIN_LIST}"
    echo "  val list  : ${VAL_LIST}"
    echo
    TRAIN_LIST="${TRAIN_LIST}" \
    VAL_LIST="${VAL_LIST}" \
    PREFLIGHT_ONLY=1 \
    PYTHON="${PYTHON}" \
      bash "${SCRIPT_DIR}/train_stage5.sh"
    ;;
  b)
    OUTPUT_DIR="${OUTPUT_ROOT}/stage_b_dummy_forward_backward"
    echo "  mode      : synthetic forward/backward on ${DEVICE}"
    echo "  output    : ${OUTPUT_DIR}"
    "${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_s5_15_gpu_preflight.py" \
      --stage b \
      --checkpoint "${INIT_CHECKPOINT}" \
      --device "${DEVICE}" \
      --output_dir "${OUTPUT_DIR}"
    ;;
  c)
    echo "  mode      : real-H5 few steps on ${DEVICE}"
    echo "  subset    : first ${NUM_TRAIN_VIDEOS} train + ${NUM_VAL_VIDEOS} validation video(s), in list order"
    echo "  epochs    : ${EPOCHS} per augmentation mode"
    for augmentation in none random_z_rotation; do
      OUTPUT_DIR="${OUTPUT_ROOT}/stage_c_${augmentation}"
      echo
      echo "--- augmentation=${augmentation} -> ${OUTPUT_DIR}"
      "${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_s5_15_gpu_preflight.py" \
        --stage c \
        --checkpoint "${INIT_CHECKPOINT}" \
        --train_list "${TRAIN_LIST}" \
        --val_list "${VAL_LIST}" \
        --num_train_videos "${NUM_TRAIN_VIDEOS}" \
        --num_val_videos "${NUM_VAL_VIDEOS}" \
        --augmentation "${augmentation}" \
        --augmentation_rotation_degrees "${ROTATION_DEGREES}" \
        --epochs "${EPOCHS}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}"
    done
    ;;
  *)
    echo "STAGE must be a, b or c; got ${STAGE}" >&2
    exit 1
    ;;
esac

echo
echo "Done (stage ${STAGE}). No arm training was started."
