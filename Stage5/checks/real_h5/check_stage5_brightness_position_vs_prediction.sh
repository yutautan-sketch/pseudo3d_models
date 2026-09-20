#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15 verification (3): brightness and position versus what the model
# predicts.
#
# Settles two things the frame images could only suggest (handoff 11.4):
#   (3) whether the model follows brightness relative to its own frame rather
#       than absolute brightness, and
#   (5) whether position still matters once brightness is held fixed.
#
# Both are answered from conditional tables, so each axis is examined with the
# other one fixed. Uses prob_femur, so no result depends on the 0.5 threshold.
#
# Reads the teacher H5s, the intermediate pseudo3d H5s and the already-saved
# predictions/*.npz. No re-inference: no model, no checkpoint, no CUDA.
#
# PSEUDO3D_OUTPUTS_ROOT is optional. The intermediate H5 is normally found from
# the source_pseudo3d_h5 attr the teacher H5 records; the root is only a
# fallback for when that path no longer resolves, and the run says so if it is
# needed.
#
# GT_REGIONS_JSON is optional. Pointing it at gt_regions_shareable.json adds the
# group comparison: the two video groups are compared at MATCHED brightness and
# MATCHED distance from the border, so an exposure difference and a response
# difference cannot be mistaken for one another.
#
#   VIDEO_ID_MAP=/path/to/video_id_map_DO_NOT_SHARE.csv \
#   EVALUATION_DIR=/path/to/<eval_output>/best \
#     bash checks/real_h5/check_stage5_brightness_position_vs_prediction.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

VIDEO_ID_MAP="${VIDEO_ID_MAP:?Set VIDEO_ID_MAP to the video_id_map_DO_NOT_SHARE.csv of the evaluation}"
EVALUATION_DIR="${EVALUATION_DIR:?Set EVALUATION_DIR to the per-checkpoint evaluation output directory}"
PSEUDO3D_OUTPUTS_ROOT="${PSEUDO3D_OUTPUTS_ROOT:-}"  # optional: only a fallback when source_pseudo3d_h5 does not resolve
STAGE2TO4_ROOT="${STAGE2TO4_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
CHECKPOINT="${CHECKPOINT:-$(basename "${EVALUATION_DIR}")}"  # the evaluation directory names its own checkpoint
SPLIT="${SPLIT:-validation}"
FALLBACK_SUFFIX="${FALLBACK_SUFFIX:-_pseudo3d.h5}"
IGNORE_INDEX="${IGNORE_INDEX:--1}"
MIN_POINTS_PER_CELL="${MIN_POINTS_PER_CELL:-200}"
GT_REGIONS_JSON="${GT_REGIONS_JSON:-}"       # optional: gt_regions_shareable.json enables the group comparison
GROUP_LABEL_KEY="${GROUP_LABEL_KEY:-multi_region_any_frame}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/work_dirs/_s5_15_brightness_position}"
PRIVATE_JSON="${PRIVATE_JSON:-${OUTPUT_ROOT}/brightness_position_${CHECKPOINT}_private_DO_NOT_SHARE.json}"
SHAREABLE_JSON="${SHAREABLE_JSON:-${OUTPUT_ROOT}/brightness_position_${CHECKPOINT}_shareable.json}"
PER_VIDEO_CSV="${PER_VIDEO_CSV:-${OUTPUT_ROOT}/brightness_position_${CHECKPOINT}_per_video_shareable.csv}"

mkdir -p "${OUTPUT_ROOT}"

echo "Stage5 brightness / position vs prediction (read-only, CPU only)"
echo "  python          : ${PYTHON}"
echo "  evaluation dir  : ${EVALUATION_DIR}"
echo "  checkpoint      : ${CHECKPOINT}"
echo "  split           : ${SPLIT}"
echo "  stage2to4 root  : ${STAGE2TO4_ROOT}"
echo "  min points/cell : ${MIN_POINTS_PER_CELL}"
echo "  group labels    : ${GT_REGIONS_JSON:-<none: group comparison skipped>}"

args=()
if [[ -n "${PSEUDO3D_OUTPUTS_ROOT}" ]]; then
  args+=(--pseudo3d_outputs_root "${PSEUDO3D_OUTPUTS_ROOT}")
fi
if [[ -n "${GT_REGIONS_JSON}" ]]; then
  args+=(--gt_regions_json "${GT_REGIONS_JSON}" --group_label_key "${GROUP_LABEL_KEY}")
fi

"${PYTHON}" "${SCRIPT_DIR}/checks/real_h5/check_stage5_brightness_position_vs_prediction.py" \
  "${args[@]+"${args[@]}"}" \
  --video_id_map "${VIDEO_ID_MAP}" \
  --evaluation_dir "${EVALUATION_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --stage2to4_root "${STAGE2TO4_ROOT}" \
  --fallback_suffix "${FALLBACK_SUFFIX}" \
  --ignore_index "${IGNORE_INDEX}" \
  --min_points_per_cell "${MIN_POINTS_PER_CELL}" \
  --private_json "${PRIVATE_JSON}" \
  --shareable_json "${SHAREABLE_JSON}" \
  --per_video_csv "${PER_VIDEO_CSV}"

echo "Done."
echo "  shareable (share this) : ${SHAREABLE_JSON}"
echo "  per-video (share this) : ${PER_VIDEO_CSV}"
echo "  private   (DO NOT share): ${PRIVATE_JSON}"
