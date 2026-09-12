#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

SOURCE_TOKEN="bboxrank_v3_refined_auto_v1"
OUTPUT_TOKEN="bboxrank_v5_cvat_authoritative_v1"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-global_local_l75_w31_c12_area15_${SOURCE_TOKEN}}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-global_local_l75_w31_c12_area15_${OUTPUT_TOKEN}}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SOURCE_RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${OUTPUT_RUN_NAME}}"
SOURCE_ANNOTATED_ROOT="${SOURCE_ANNOTATED_ROOT:-${SOURCE_RUN_ROOT}/annotated}"
ANNOTATED_ROOT="${OUTPUT_ROOT}/annotated"
COLLECTED_ROOT="${OUTPUT_ROOT}/collected"
VISUALIZATION_ROOT="${VISUALIZATION_ROOT:-${OUTPUT_ROOT}/annotation_textures_v5_cvat_authoritative_labels}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-${OUTPUT_ROOT}/cvat_authoritative_preflight_step5}"
ACCEPTANCE_JSON="${ACCEPTANCE_JSON:-${OUTPUT_ROOT}/cvat_authoritative_acceptance_step6.json}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

MANIFEST="${MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest_cropclean_v1.csv}"
EXCLUSION_MANIFEST="${EXCLUSION_MANIFEST:-pseudo3d/analysis/configs/stage4_video_exclusions_v1.csv}"
MANUAL_REVIEW_CONFIG="${MANUAL_REVIEW_CONFIG:-pseudo3d/analysis/configs/stage4_manual_review_cvat_v1.yaml}"

PHASE5_RETURN_ROOT="${PHASE5_RETURN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/cvat_returns/phase5_fullvideo_v3_textfree_reviewed_snapshot_59_v2/manual_review_cvat_phase5_fullvideo_v3_textfree}"
PHASE5_FULLVIDEO_REVIEW_ROOT="${PHASE5_FULLVIDEO_REVIEW_ROOT:-${SOURCE_RUN_ROOT}/manual_review_cvat_phase5_fullvideo_v3_textfree}"
PHASE5_FULLVIDEO_SNAPSHOT_ROOT="${PHASE5_FULLVIDEO_SNAPSHOT_ROOT:-${PHASE5_RETURN_ROOT}/cvat_exports_after_textfree_review_v2}"
PHASE5_FULLVIDEO_TASK_MAP="${PHASE5_FULLVIDEO_TASK_MAP:-${PHASE5_RETURN_ROOT}/task_management/task_map.csv}"

EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-181}"
EXPECTED_REVIEWED_VIDEOS="${EXPECTED_REVIEWED_VIDEOS:-59}"
EXPECTED_INHERITED_VIDEOS="${EXPECTED_INHERITED_VIDEOS:-122}"
EXPECTED_EXCLUDED_VIDEOS="${EXPECTED_EXCLUDED_VIDEOS:-1}"
EXPECTED_AUTHORITATIVE_FRAMES="${EXPECTED_AUTHORITATIVE_FRAMES:-3014}"
EXPECTED_ACTIONABLE_BBOXES="${EXPECTED_ACTIONABLE_BBOXES:-112}"
EXPECTED_BBOX_OVERRIDE_FRAMES="${EXPECTED_BBOX_OVERRIDE_FRAMES:--1}"
EXPECTED_BBOX_OVERRIDE_PIXELS="${EXPECTED_BBOX_OVERRIDE_PIXELS:--1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ANNOTATED_PATTERN="*_pointcloud_annotated_*${OUTPUT_TOKEN}.h5"

if [[ "${SKIP_EXISTING}" != "0" && "${SKIP_EXISTING}" != "1" ]]; then
  echo "SKIP_EXISTING must be 0 or 1, got: ${SKIP_EXISTING}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
for required_file in "${MANIFEST}" "${EXCLUSION_MANIFEST}" "${MANUAL_REVIEW_CONFIG}" "${PHASE5_FULLVIDEO_TASK_MAP}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
for required_file in "${PREFLIGHT_ROOT}/preflight_summary.json" "${PREFLIGHT_ROOT}/input_checksums.csv"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Step 5 preflight artifact not found: ${required_file}" >&2
    exit 1
  fi
done
for required_dir in "${SOURCE_ANNOTATED_ROOT}" "${PHASE5_FULLVIDEO_REVIEW_ROOT}" "${PHASE5_FULLVIDEO_SNAPSHOT_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory not found: ${required_dir}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_ROOT}"
common_args=(
  --manifest "${MANIFEST}"
  --exclusion_manifest "${EXCLUSION_MANIFEST}"
  --source_annotated_root "${SOURCE_ANNOTATED_ROOT}"
  --review_root "${PHASE5_FULLVIDEO_REVIEW_ROOT}"
  --snapshot_root "${PHASE5_FULLVIDEO_SNAPSHOT_ROOT}"
  --task_map_csv "${PHASE5_FULLVIDEO_TASK_MAP}"
  --manual_review_config "${MANUAL_REVIEW_CONFIG}"
  --output_root "${ANNOTATED_ROOT}"
  --output_teacher_token "${OUTPUT_TOKEN}"
  --expected_videos "${EXPECTED_VIDEOS}"
  --expected_reviewed_videos "${EXPECTED_REVIEWED_VIDEOS}"
  --expected_actionable_bboxes "${EXPECTED_ACTIONABLE_BBOXES}"
  --expected_bbox_override_frames "${EXPECTED_BBOX_OVERRIDE_FRAMES}"
  --expected_bbox_override_pixels "${EXPECTED_BBOX_OVERRIDE_PIXELS}"
)
import_mode_args=()
collect_mode_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  import_mode_args+=(--skip_existing)
else
  import_mode_args+=(--overwrite)
  collect_mode_args+=(--overwrite)
fi

echo "Stage 4 teacher v5 CVAT-authoritative build"
echo "  source run      : ${SOURCE_RUN_ROOT}"
echo "  CVAT return root: ${PHASE5_RETURN_ROOT}"
echo "  output run      : ${OUTPUT_ROOT}"
echo "  reviewed videos : ${EXPECTED_REVIEWED_VIDEOS}"
echo "  inherited videos: ${EXPECTED_INHERITED_VIDEOS}"
echo "  excluded videos : ${EXPECTED_EXCLUDED_VIDEOS}"
echo "  Task frames     : ${EXPECTED_AUTHORITATIVE_FRAMES}"
echo "  label authority : CVAT snapshot on every Task frame"

echo
echo "[1/5] Building CVAT-authoritative annotated H5 files"
"${PYTHON}" \
  pseudo3d/batch/annotation/batch_import_stage4_phase5_fullvideo_cvat.py \
  "${common_args[@]}" \
  "${import_mode_args[@]}" \
  2>&1 | tee "${LOG_ROOT}/01_authoritative_import.log"

echo
echo "[2/5] Collecting final teacher H5 files"
mkdir -p "${COLLECTED_ROOT}"
"${PYTHON}" \
  scripts/utils/collect_annotated_pseudo3d_h5.py \
  --source_root "${ANNOTATED_ROOT}" \
  --output_dir "${COLLECTED_ROOT}" \
  --mode foreground \
  --pattern "${ANNOTATED_PATTERN}" \
  --manifest_csv "${COLLECTED_ROOT}/manifest.csv" \
  "${collect_mode_args[@]}" \
  2>&1 | tee "${LOG_ROOT}/02_collect.log"

annotated_count="$(find "${ANNOTATED_ROOT}" -type f -name "${ANNOTATED_PATTERN}" | wc -l | tr -d ' ')"
collected_count="$(find "${COLLECTED_ROOT}" -maxdepth 1 -type f -name "${ANNOTATED_PATTERN}" | wc -l | tr -d ' ')"
if [[ "${annotated_count}" -ne "${EXPECTED_VIDEOS}" || "${collected_count}" -ne "${EXPECTED_VIDEOS}" ]]; then
  echo "Final H5 count mismatch: annotated=${annotated_count}, collected=${collected_count}" >&2
  exit 1
fi

echo
echo "[3/5] Auditing the saved three-label and CVAT-authority policy"
"${PYTHON}" \
  checks/stage4/check_stage4_bbox_ranked_label_policy.py \
  --input_dir "${COLLECTED_ROOT}" \
  --pattern "${ANNOTATED_PATTERN}" \
  --expected_files "${EXPECTED_VIDEOS}" \
  --allow_cvat_authoritative_no_bbox_positive \
  2>&1 | tee "${LOG_ROOT}/03_label_policy.log"

echo
echo "[4/5] Rendering saved labels with authoritative CVAT masks"
PYTHON="${PYTHON}" \
REPO_ROOT="${REPO_ROOT}" \
RUN_ROOT="${OUTPUT_ROOT}" \
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT}" \
PHASE5_RETURN_ROOT="${PHASE5_RETURN_ROOT}" \
CVAT_REVIEW_ROOT="${PHASE5_FULLVIDEO_REVIEW_ROOT}" \
CVAT_SNAPSHOT_ROOT="${PHASE5_FULLVIDEO_SNAPSHOT_ROOT}" \
ANNOTATED_DIR="${ANNOTATED_ROOT}" \
OUTPUT_ROOT="${VISUALIZATION_ROOT}" \
SUMMARY_CSV="${VISUALIZATION_ROOT}/summary.csv" \
EXPECTED_FILES="${EXPECTED_VIDEOS}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
bash pseudo3d/pipelines/export_stage4_v5_cvat_authoritative_point_label_visualizations.sh \
  2>&1 | tee "${LOG_ROOT}/04_visualization.log"

echo
echo "[5/5] Checking the versioned v5 build against Step 5 projections"
"${PYTHON}" \
  checks/stage4/check_stage4_v5_cvat_authoritative_acceptance.py \
  --manifest "${MANIFEST}" \
  --exclusion_manifest "${EXCLUSION_MANIFEST}" \
  --source_annotated_root "${SOURCE_ANNOTATED_ROOT}" \
  --annotated_root "${ANNOTATED_ROOT}" \
  --collected_root "${COLLECTED_ROOT}" \
  --visualization_root "${VISUALIZATION_ROOT}" \
  --snapshot_root "${PHASE5_FULLVIDEO_SNAPSHOT_ROOT}" \
  --preflight_root "${PREFLIGHT_ROOT}" \
  --output_json "${ACCEPTANCE_JSON}" \
  --expected_videos "${EXPECTED_VIDEOS}" \
  --expected_reviewed_videos "${EXPECTED_REVIEWED_VIDEOS}" \
  --expected_inherited_videos "${EXPECTED_INHERITED_VIDEOS}" \
  --expected_excluded_videos "${EXPECTED_EXCLUDED_VIDEOS}" \
  --expected_authoritative_frames "${EXPECTED_AUTHORITATIVE_FRAMES}" \
  2>&1 | tee "${LOG_ROOT}/05_acceptance.log"

echo
echo "Stage 4 teacher v5 CVAT-authoritative build complete."
echo "  annotated     : ${ANNOTATED_ROOT}"
echo "  collected     : ${COLLECTED_ROOT}"
echo "  visualizations: ${VISUALIZATION_ROOT}"
echo "  acceptance    : ${ACCEPTANCE_JSON}"
echo "  files         : annotated=${annotated_count}, collected=${collected_count}"
