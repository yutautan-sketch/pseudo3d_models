#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# S5-15: count GT femur regions per frame, mechanically, to replace the by-eye
# "the femur appears in two places" grouping from the qualitative review.
#
# Reuses the evaluation's own private alias map so no new numbering scheme is
# invented (the review already hit an off-by-one between two schemes). Sweeps
# the link distance, because the region count depends on it and a finding that
# survives only one radius is not a finding.
#
# Read-only, CPU only: reads the teacher H5s, writes only its own outputs.
# The private JSON and per-frame CSV carry real video names -- DO NOT SHARE
# them; share the shareable JSON.
#
#   VIDEO_ID_MAP=/path/to/video_id_map_DO_NOT_SHARE.csv \
#     bash checks/real_h5/check_stage5_gt_component_count.sh
# ------------------------------------------------------------

SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"

VIDEO_ID_MAP="${VIDEO_ID_MAP:?Set VIDEO_ID_MAP to the video_id_map_DO_NOT_SHARE.csv of the evaluation}"
SPLIT="${SPLIT:-validation}"
LINK_DISTANCES="${LINK_DISTANCES:-2,3,4,6,8,12}"
MIN_COMPONENT_POINTS="${MIN_COMPONENT_POINTS:-5}"
FRAME_FRACTION_THRESHOLD="${FRAME_FRACTION_THRESHOLD:-0.5}"
METRICS_CSV="${METRICS_CSV:-}"
METRICS_CHECKPOINT="${METRICS_CHECKPOINT:-best}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/work_dirs/_s5_15_gt_regions}"
PRIVATE_JSON="${PRIVATE_JSON:-${OUTPUT_ROOT}/gt_regions_private_DO_NOT_SHARE.json}"
SHAREABLE_JSON="${SHAREABLE_JSON:-${OUTPUT_ROOT}/gt_regions_shareable.json}"
PER_FRAME_CSV="${PER_FRAME_CSV:-${OUTPUT_ROOT}/gt_regions_per_frame_DO_NOT_SHARE.csv}"

mkdir -p "${OUTPUT_ROOT}"

echo "Stage5 GT region-count check (read-only, CPU only)"
echo "  python         : ${PYTHON}"
echo "  split          : ${SPLIT}"
echo "  link distances : ${LINK_DISTANCES}"

args=(
  "${SCRIPT_DIR}/checks/real_h5/check_stage5_gt_component_count.py"
  --video_id_map "${VIDEO_ID_MAP}"
  --split "${SPLIT}"
  --link_distances "${LINK_DISTANCES}"
  --min_component_points "${MIN_COMPONENT_POINTS}"
  --frame_fraction_threshold "${FRAME_FRACTION_THRESHOLD}"
  --private_json "${PRIVATE_JSON}"
  --shareable_json "${SHAREABLE_JSON}"
  --per_frame_csv "${PER_FRAME_CSV}"
)

if [[ -n "${METRICS_CSV}" ]]; then
  args+=(--metrics_csv "${METRICS_CSV}" --metrics_checkpoint "${METRICS_CHECKPOINT}")
fi

"${PYTHON}" "${args[@]}"

echo "Done."
echo "  shareable (share this) : ${SHAREABLE_JSON}"
echo "  private   (DO NOT share): ${PRIVATE_JSON}"
echo "  per-frame (DO NOT share): ${PER_FRAME_CSV}"
