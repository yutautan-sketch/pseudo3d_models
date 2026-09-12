#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
MANIFEST="${MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv}"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v2_nobbox_bg}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-global_local_l75_w31_c12_area15_bboxrank_v3_refined_auto_v1}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SOURCE_RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${OUTPUT_RUN_NAME}}"
SOURCE_ANNOTATED_ROOT="${SOURCE_ANNOTATED_ROOT:-${SOURCE_RUN_ROOT}/annotated}"
ANNOTATED_ROOT="${OUTPUT_ROOT}/annotated"
COLLECTED_ROOT="${OUTPUT_ROOT}/collected"
VISUALIZATION_ROOT="${OUTPUT_ROOT}/annotation_textures"
LOG_ROOT="${OUTPUT_ROOT}/logs"

PHASE1_ROOT="${PHASE1_ROOT:-${SOURCE_RUN_ROOT}/contour_teacher_audit_phase1}"
PHASE3_ROOT="${PHASE3_ROOT:-${SOURCE_RUN_ROOT}/contour_auto_refine_phase3_production_v1_decisions}"
TEACHER_CONFIG="${TEACHER_CONFIG:-pseudo3d/analysis/configs/stage4_bbox_ranked_teacher_v2.yaml}"
REFINE_CONFIG="${REFINE_CONFIG:-pseudo3d/analysis/configs/stage4_contour_auto_refine_phase3_production_v1.yaml}"

EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-182}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXPORT_VISUALIZATIONS="${EXPORT_VISUALIZATIONS:-1}"
ANNOTATED_PATTERN="*_pointcloud_annotated_*bboxrank_v3_refined_auto_v1.h5"

for value_name in SKIP_EXISTING EXPORT_VISUALIZATIONS; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
for required_file in "${MANIFEST}" "${TEACHER_CONFIG}" "${REFINE_CONFIG}"; do
  if [[ ! -f "${required_file}" && ! -f "${REPO_ROOT}/${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
for required_dir in "${SOURCE_ANNOTATED_ROOT}" "${PHASE1_ROOT}" "${PHASE3_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory not found: ${required_dir}" >&2
    exit 1
  fi
done

cd "${REPO_ROOT}"
mkdir -p "${ANNOTATED_ROOT}" "${COLLECTED_ROOT}" "${LOG_ROOT}"

skip_args=()
collect_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_args+=(--skip_existing)
else
  collect_args+=(--overwrite)
fi

echo "Stage 4 teacher v3 refined-auto full build"
echo "  source run     : ${SOURCE_RUN_ROOT}"
echo "  phase3 decisions: ${PHASE3_ROOT}"
echo "  output run     : ${OUTPUT_ROOT}"
echo "  expected videos: ${EXPECTED_VIDEOS}"
echo "  policy         : auto_refine only; other BBoxes preserve v2"

echo
echo "[1/4] Applying Phase 3 auto-refine decisions"
"${PYTHON}" \
  pseudo3d/batch/annotation/batch_apply_stage4_contour_refinement.py \
  --manifest "${MANIFEST}" \
  --source_annotated_root "${SOURCE_ANNOTATED_ROOT}" \
  --phase1_audit_root "${PHASE1_ROOT}" \
  --phase3_root "${PHASE3_ROOT}" \
  --teacher_config "${TEACHER_CONFIG}" \
  --refine_config "${REFINE_CONFIG}" \
  --output_root "${ANNOTATED_ROOT}" \
  --expected_videos "${EXPECTED_VIDEOS}" \
  "${skip_args[@]}" \
  2>&1 | tee "${LOG_ROOT}/01_phase5_apply.log"

echo
echo "[2/4] Collecting teacher v3 H5 files"
"${PYTHON}" \
  scripts/utils/collect_annotated_pseudo3d_h5.py \
  --source_root "${ANNOTATED_ROOT}" \
  --output_dir "${COLLECTED_ROOT}" \
  --mode foreground \
  --pattern "${ANNOTATED_PATTERN}" \
  --manifest_csv "${COLLECTED_ROOT}/manifest.csv" \
  "${collect_args[@]}" \
  2>&1 | tee "${LOG_ROOT}/02_collect.log"

annotated_count="$(find "${ANNOTATED_ROOT}" -type f -name "${ANNOTATED_PATTERN}" | wc -l | tr -d ' ')"
collected_count="$(find "${COLLECTED_ROOT}" -maxdepth 1 -type f -name "${ANNOTATED_PATTERN}" | wc -l | tr -d ' ')"
if [[ "${annotated_count}" -ne "${EXPECTED_VIDEOS}" || "${collected_count}" -ne "${EXPECTED_VIDEOS}" ]]; then
  echo "Phase 5 H5 count mismatch: annotated=${annotated_count}, collected=${collected_count}" >&2
  exit 1
fi

echo
echo "[3/4] Auditing the saved three-label policy"
"${PYTHON}" \
  checks/stage4/check_stage4_bbox_ranked_label_policy.py \
  --input_dir "${COLLECTED_ROOT}" \
  --pattern "${ANNOTATED_PATTERN}" \
  --expected_files "${EXPECTED_VIDEOS}" \
  2>&1 | tee "${LOG_ROOT}/03_label_policy.log"

echo
if [[ "${EXPORT_VISUALIZATIONS}" == "1" ]]; then
  echo "[4/4] Exporting annotation textures"
  PYTHON="${PYTHON}" \
  REPO_ROOT="${REPO_ROOT}" \
  RUN_ROOT="${OUTPUT_ROOT}" \
  ANNOTATED_DIR="${ANNOTATED_ROOT}" \
  OUTPUT_ROOT="${VISUALIZATION_ROOT}" \
  SAMPLING_RUN_NAME="global_local_l75_w31_c12_area15" \
  TEACHER_RUN_NAME="bboxrank_v3_refined_auto_v1" \
  ANNOTATED_PATTERN="${ANNOTATED_PATTERN}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
    bash pseudo3d/batch/export/batch_export_annotation_mask_visualization.sh \
    2>&1 | tee "${LOG_ROOT}/04_visualization.log"
else
  echo "[4/4] Annotation texture export skipped (EXPORT_VISUALIZATIONS=0)"
fi

echo
echo "Stage 4 teacher v3 refined-auto build complete."
echo "  annotated     : ${ANNOTATED_ROOT}"
echo "  collected     : ${COLLECTED_ROOT}"
echo "  visualizations: ${VISUALIZATION_ROOT}"
echo "  files          : annotated=${annotated_count}, collected=${collected_count}"
