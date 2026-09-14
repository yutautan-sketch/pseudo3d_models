#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
PYTHON_BIN="${PYTHON_BIN:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

V6_RUN_ROOT="${V6_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/global_local_l75_w31_c12_area15_bboxrank_v6_cvat_authoritative_xml_invalidation_v1}"
MANIFEST="${MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv}"
TEACHER_CONFIG="${TEACHER_CONFIG:-${REPO_ROOT}/pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${V6_RUN_ROOT}/crop_quality_stray_ignore_audit_step1}"
OVERWRITE="${OVERWRITE:-0}"

TRAIN_VIDEO="${TRAIN_VIDEO:-20250626_090652_6340}"
TRAIN_FRAME_ORDERS="${TRAIN_FRAME_ORDERS:-38,43,44,46}"
VALIDATION_VIDEO="${VALIDATION_VIDEO:-20250625_161030_0550}"
VALIDATION_FRAME_ORDERS="${VALIDATION_FRAME_ORDERS:-51}"
COMPARISON_VIDEO="${COMPARISON_VIDEO:-20250626_090758_8000}"

cd "${REPO_ROOT}"

args=(
  pseudo3d/analysis/audit_stage4_crop_quality_stray_ignore.py
  --manifest "${MANIFEST}"
  --teacher_config "${TEACHER_CONFIG}"
  --annotated_root "${V6_RUN_ROOT}/annotated"
  --output_root "${OUTPUT_ROOT}"
  --target "${TRAIN_VIDEO}:${TRAIN_FRAME_ORDERS}"
  --target "${VALIDATION_VIDEO}:${VALIDATION_FRAME_ORDERS}"
  --comparison_video "${COMPARISON_VIDEO}"
  --expected_stray_points 16
  --expected_stray_videos 2
  --expected_problem_frames 5
)

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

echo "Stage 4 crop-quality/stray-ignore read-only audit"
echo "  v6 root          : ${V6_RUN_ROOT}"
echo "  train target     : ${TRAIN_VIDEO}:${TRAIN_FRAME_ORDERS}"
echo "  validation target: ${VALIDATION_VIDEO}:${VALIDATION_FRAME_ORDERS}"
echo "  comparison       : ${COMPARISON_VIDEO}"
echo "  output root      : ${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${args[@]}"

echo
echo "Stage 4 crop-quality/stray-ignore audit complete."
echo "  summary     : ${OUTPUT_ROOT}/audit_summary.json"
echo "  stray points: ${OUTPUT_ROOT}/stray_points.csv"
echo "  frames      : ${OUTPUT_ROOT}/frame_metrics.csv"
echo "  BBoxes      : ${OUTPUT_ROOT}/bbox_geometry.csv"
echo "  videos      : ${OUTPUT_ROOT}/video_summary.csv"
echo "  checksums   : ${OUTPUT_ROOT}/input_checksums.csv"
