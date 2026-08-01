#!/usr/bin/env bash
set -euo pipefail

# Export both raw and annotation-colored PLY files for the complete 260711
# BBox-ranked v2 dataset without rebuilding point-cloud H5 files.

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
SAMPLING_RUN_NAME="${SAMPLING_RUN_NAME:-global_local_l75_w31_c12_area15}"
TEACHER_RUN_NAME="${TEACHER_RUN_NAME:-bboxrank_v2_nobbox_bg}"
V2_RUN_NAME="${V2_RUN_NAME:-${SAMPLING_RUN_NAME}_${TEACHER_RUN_NAME}}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SAMPLING_RUN_NAME}}"
V2_RUN_ROOT="${V2_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${V2_RUN_NAME}}"
RAW_H5_ROOT="${RAW_H5_ROOT:-${SOURCE_RUN_ROOT}/point_cloud}"
ANNOTATED_H5_ROOT="${ANNOTATED_H5_ROOT:-${V2_RUN_ROOT}/annotated}"
RAW_PLY_ROOT="${RAW_PLY_ROOT:-${V2_RUN_ROOT}/pointcloud_foreground}"
ANNOTATED_PLY_ROOT="${ANNOTATED_PLY_ROOT:-${V2_RUN_ROOT}/pointcloud_annotated_foreground}"

POINT_CLOUD_TAG="foreground_combined_v2_${SAMPLING_RUN_NAME}"
ANNOTATED_TAG="${POINT_CLOUD_TAG}_${TEACHER_RUN_NAME}"
RAW_H5_PATTERN="*_pointcloud_${POINT_CLOUD_TAG}.h5"
ANNOTATED_H5_PATTERN="*_pointcloud_annotated_${ANNOTATED_TAG}.h5"
RAW_PLY_PATTERN="*_pointcloud_${POINT_CLOUD_TAG}.ply"
ANNOTATED_PLY_PATTERN="*_pointcloud_annotated_${ANNOTATED_TAG}.ply"

EXPECTED_FILES="${EXPECTED_FILES:-182}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${RAW_H5_ROOT}" ]]; then
  echo "Raw point-cloud H5 root not found: ${RAW_H5_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${ANNOTATED_H5_ROOT}" ]]; then
  echo "Annotated point-cloud H5 root not found: ${ANNOTATED_H5_ROOT}" >&2
  exit 1
fi
if [[ "${SKIP_EXISTING}" != "0" && "${SKIP_EXISTING}" != "1" ]]; then
  echo "SKIP_EXISTING must be 0 or 1, got: ${SKIP_EXISTING}" >&2
  exit 1
fi

raw_h5_count="$(
  find "${RAW_H5_ROOT}" -type f -name "${RAW_H5_PATTERN}" |
    wc -l |
    tr -d ' '
)"
annotated_h5_count="$(
  find "${ANNOTATED_H5_ROOT}" -type f -name "${ANNOTATED_H5_PATTERN}" |
    wc -l |
    tr -d ' '
)"
if [[ "${raw_h5_count}" -ne "${EXPECTED_FILES}" ]]; then
  echo "Raw H5 count mismatch: expected=${EXPECTED_FILES}, actual=${raw_h5_count}" >&2
  exit 1
fi
if [[ "${annotated_h5_count}" -ne "${EXPECTED_FILES}" ]]; then
  echo "Annotated H5 count mismatch: expected=${EXPECTED_FILES}, actual=${annotated_h5_count}" >&2
  exit 1
fi

mkdir -p "${RAW_PLY_ROOT}" "${ANNOTATED_PLY_ROOT}"

skip_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_args+=(--skip_existing)
fi

echo "Stage 4 BBox-ranked point-cloud visualization export"
echo "  raw H5 root       : ${RAW_H5_ROOT}"
echo "  annotated H5 root : ${ANNOTATED_H5_ROOT}"
echo "  raw PLY root      : ${RAW_PLY_ROOT}"
echo "  annotated PLY root: ${ANNOTATED_PLY_ROOT}"
echo "  input files       : raw=${raw_h5_count}, annotated=${annotated_h5_count}"
echo "  skip existing     : ${SKIP_EXISTING}"

echo
echo "[1/2] Exporting pointcloud_foreground PLY files"
"${PYTHON}" \
  pseudo3d/batch/export/batch_export_point_cloud_ply.py \
  --input_dir "${RAW_H5_ROOT}" \
  --pattern "${RAW_H5_PATTERN}" \
  --recursive \
  --output_root "${RAW_PLY_ROOT}" \
  --output_subdir_template "{parent_name}" \
  --output_filename_template "{input_stem}.ply" \
  --summary_csv "${RAW_PLY_ROOT}/summary.csv" \
  "${skip_args[@]}"

echo
echo "[2/2] Exporting pointcloud_annotated_foreground PLY files"
"${PYTHON}" \
  pseudo3d/batch/export/batch_export_annotated_point_cloud_ply.py \
  --annotated_dir "${ANNOTATED_H5_ROOT}" \
  --pattern "${ANNOTATED_H5_PATTERN}" \
  --recursive \
  --output_root "${ANNOTATED_PLY_ROOT}" \
  --output_subdir_template "{parent_name}" \
  --output_filename_template "{input_stem}.ply" \
  --summary_csv "${ANNOTATED_PLY_ROOT}/summary.csv" \
  --no_annotation_only_output \
  --background_mode dim \
  --color_mode annotation_source \
  --femur_color "255,64,32" \
  --global_color "255,255,255" \
  --local_percentile_color "64,200,255" \
  --tophat_color "255,200,64" \
  --context_grid_color "120,120,120" \
  --background_alpha 85 \
  --ignore_alpha 35 \
  "${skip_args[@]}"

raw_ply_count="$(
  find "${RAW_PLY_ROOT}" -type f -name "${RAW_PLY_PATTERN}" |
    wc -l |
    tr -d ' '
)"
annotated_ply_count="$(
  find "${ANNOTATED_PLY_ROOT}" -type f -name "${ANNOTATED_PLY_PATTERN}" |
    wc -l |
    tr -d ' '
)"
if [[ "${raw_ply_count}" -ne "${EXPECTED_FILES}" ]]; then
  echo "Raw PLY count mismatch: expected=${EXPECTED_FILES}, actual=${raw_ply_count}" >&2
  exit 1
fi
if [[ "${annotated_ply_count}" -ne "${EXPECTED_FILES}" ]]; then
  echo "Annotated PLY count mismatch: expected=${EXPECTED_FILES}, actual=${annotated_ply_count}" >&2
  exit 1
fi

echo
echo "Stage 4 point-cloud visualization export complete."
echo "  pointcloud_foreground           : ${raw_ply_count}"
echo "  pointcloud_annotated_foreground : ${annotated_ply_count}"
