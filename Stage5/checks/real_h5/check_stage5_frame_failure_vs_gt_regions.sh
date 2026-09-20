#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 verification (2): is the multi-region penalty a property of FRAMES or
# only of VIDEOS?
#
# 8.12.15 compared whole videos, so a video-level confound could explain it.
# This compares single-region and multi-region frames inside the same video,
# which holds the recording, the patient and the crop fixed, and reports an
# exact sign test over those within-video differences.
#
# Reads the teacher H5s and the already-saved predictions/*.npz. No
# re-inference: no model, no checkpoint, no CUDA.
#
# The per-frame CSV uses aliases only and is shareable; the private JSON is not.
#
# CHECKPOINT defaults to the name of EVALUATION_DIR, because that directory
# holds only its own checkpoint's rows; setting the two to different values
# stops the run instead of being resolved one way or the other.
#
#   VIDEO_ID_MAP=/path/to/video_id_map_DO_NOT_SHARE.csv \
#   EVALUATION_DIR=/path/to/<eval_output>/best \
#     bash checks/real_h5/check_stage5_frame_failure_vs_gt_regions.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

VIDEO_ID_MAP="${VIDEO_ID_MAP:?Set VIDEO_ID_MAP to the video_id_map_DO_NOT_SHARE.csv of the evaluation}"
EVALUATION_DIR="${EVALUATION_DIR:?Set EVALUATION_DIR to the per-checkpoint evaluation output directory}"
CHECKPOINT="${CHECKPOINT:-$(basename "${EVALUATION_DIR}")}"  # the evaluation directory names its own checkpoint
SPLIT="${SPLIT:-validation}"
LINK_DISTANCE="${LINK_DISTANCE:-4.0}"
MIN_COMPONENT_POINTS="${MIN_COMPONENT_POINTS:-5}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/work_dirs/_s5_15_frame_failure}"
PRIVATE_JSON="${PRIVATE_JSON:-${OUTPUT_ROOT}/frame_failure_${CHECKPOINT}_private_DO_NOT_SHARE.json}"
SHAREABLE_JSON="${SHAREABLE_JSON:-${OUTPUT_ROOT}/frame_failure_${CHECKPOINT}_shareable.json}"
PER_FRAME_CSV="${PER_FRAME_CSV:-${OUTPUT_ROOT}/frame_failure_${CHECKPOINT}_per_frame_shareable.csv}"

mkdir -p "${OUTPUT_ROOT}"

echo "Stage5 frame-level failure vs GT regions (read-only, CPU only)"
echo "  python         : ${PYTHON}"
echo "  evaluation dir : ${EVALUATION_DIR}"
echo "  checkpoint     : ${CHECKPOINT}"
echo "  split          : ${SPLIT}"
echo "  link distance  : ${LINK_DISTANCE}"

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_frame_failure_vs_gt_regions.py" \
  --video_id_map "${VIDEO_ID_MAP}" \
  --evaluation_dir "${EVALUATION_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --link_distance "${LINK_DISTANCE}" \
  --min_component_points "${MIN_COMPONENT_POINTS}" \
  --ignore_index "${IGNORE_INDEX}" \
  --private_json "${PRIVATE_JSON}" \
  --shareable_json "${SHAREABLE_JSON}" \
  --per_frame_csv "${PER_FRAME_CSV}"

echo "Done."
echo "  shareable (share this) : ${SHAREABLE_JSON}"
echo "  per-frame (share this) : ${PER_FRAME_CSV}"
echo "  private   (DO NOT share): ${PRIVATE_JSON}"
