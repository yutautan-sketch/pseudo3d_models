#!/usr/bin/env bash
set -euo pipefail

# Build the first Stage 5 sampling ablation dataset:
#   legacy global evidence + tuned local-percentile evidence
#
# Deliberately disabled:
#   top-hat evidence
#   context grid
#
# Environment overrides:
#   PYTHON, REPO_ROOT, DATASET_ROOT, SOURCE_MANIFEST, VOC_XML_ROOT,
#   OUTPUT_ROOT, SKIP_EXISTING

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv}"
VOC_XML_ROOT="${VOC_XML_ROOT:-/mnt/data/Data_hbl/predirectory_for_coco_annotation_leg/260630/260630_pseudo3d}"

RUN_NAME="global_local_l75_w31_c12_area15"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${RUN_NAME}}"
POINT_CLOUD_ROOT="${OUTPUT_ROOT}/point_cloud"
ANNOTATED_ROOT="${OUTPUT_ROOT}/annotated"
COLLECTED_ROOT="${OUTPUT_ROOT}/collected"
LOG_ROOT="${OUTPUT_ROOT}/logs"

POINT_CLOUD_TAG="foreground_combined_v2_${RUN_NAME}"
RAW_H5_TEMPLATE="{video_name}_pointcloud_${POINT_CLOUD_TAG}.h5"
ANNOTATED_H5_TEMPLATE="{video_name}_pointcloud_annotated_${POINT_CLOUD_TAG}.h5"
ANNOTATED_PATTERN="*_pointcloud_annotated_${POINT_CLOUD_TAG}.h5"

SKIP_EXISTING="${SKIP_EXISTING:-0}"

cd "${REPO_ROOT}"
mkdir -p \
  "${POINT_CLOUD_ROOT}" \
  "${ANNOTATED_ROOT}" \
  "${COLLECTED_ROOT}" \
  "${LOG_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "Source manifest not found: ${SOURCE_MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${VOC_XML_ROOT}" ]]; then
  echo "VOC XML root not found: ${VOC_XML_ROOT}" >&2
  exit 1
fi
if [[ "${SKIP_EXISTING}" != "0" && "${SKIP_EXISTING}" != "1" ]]; then
  echo "SKIP_EXISTING must be 0 or 1, got: ${SKIP_EXISTING}" >&2
  exit 1
fi

skip_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_args+=(--skip_existing)
fi

echo "Stage 4 global+local training ablation dataset"
echo "  source manifest : ${SOURCE_MANIFEST}"
echo "  VOC XML root    : ${VOC_XML_ROOT}"
echo "  output root     : ${OUTPUT_ROOT}"
echo "  point-cloud tag : ${POINT_CLOUD_TAG}"
echo "  skip existing   : ${SKIP_EXISTING}"

export_cmd=(
  "${PYTHON}"
  pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py
  --h5_list "${SOURCE_MANIFEST}"
  --output_root "${POINT_CLOUD_ROOT}"
  --summary_csv "${POINT_CLOUD_ROOT}/summary.csv"
  --video_suffix_to_strip "_ts448_oym96_corr"
  --output_subdir_template "{video_name}"
  --output_h5_filename_template "${RAW_H5_TEMPLATE}"
  --output_ply_filename_template "{video_name}_pointcloud_${POINT_CLOUD_TAG}.ply"
  --geometry_key corr
  --preset percentile85_area100
  --point_mode foreground
  --sample_stride 2
  --sampling_mode combined_v2
  --no-include_context_grid
  --enable_local_percentile
  --local_window_size 31
  --local_percentile 75
  --local_min_contrast 12
  --no-enable_tophat
  --enable_evidence_cleanup
  --evidence_open_ksize 3
  --evidence_close_ksize 5
  --evidence_morph_shape ellipse
  --evidence_min_component_area 15
  --local_source_confidence 0.7
  --texture_style threshold_alpha
  --texture_denoise median
  --texture_denoise_ksize 3
  --texture_threshold_mode percentile
  --texture_percentile 85
  --texture_open_ksize 3
  --texture_close_ksize 5
  --texture_morph_shape ellipse
  --texture_min_component_area 100
  --no_ply
  "${skip_args[@]}"
)

echo
echo "[1/3] Exporting global+local point clouds"
"${export_cmd[@]}" 2>&1 | tee "${LOG_ROOT}/01_export.log"

annotate_cmd=(
  "${PYTHON}"
  pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py
  --h5_list "${SOURCE_MANIFEST}"
  --point_cloud_root "${POINT_CLOUD_ROOT}"
  --voc_xml_root "${VOC_XML_ROOT}"
  --output_root "${ANNOTATED_ROOT}"
  --summary_csv "${ANNOTATED_ROOT}/summary.csv"
  --video_suffix_to_strip "_ts448_oym96_corr"
  --point_cloud_detail "${RUN_NAME}"
  --point_cloud_subdir_template "{video_name}"
  --point_cloud_filename_template "${RAW_H5_TEMPLATE}"
  --output_subdir_template "{video_name}"
  --output_filename_template "${ANNOTATED_H5_TEMPLATE}"
  --geometry_key corr
  --xml_frame_number_offsets "1"
  --xml_frame_id_source frame_index
  --xml_annotation_dir_name annotations_renamed
  --strict_xml_annotation_dir
  --contour_preset percentile85_area100
  --contour_min_area 20
  --bbox_inside_non_contour_label ignore
  "${skip_args[@]}"
)

echo
echo "[2/3] Annotating point clouds with strict dense-teacher conditions"
"${annotate_cmd[@]}" 2>&1 | tee "${LOG_ROOT}/02_annotation.log"

collect_cmd=(
  "${PYTHON}"
  scripts/utils/collect_annotated_pseudo3d_h5.py
  --source_root "${ANNOTATED_ROOT}"
  --output_dir "${COLLECTED_ROOT}"
  --mode foreground
  --pattern "${ANNOTATED_PATTERN}"
  --manifest_csv "${COLLECTED_ROOT}/manifest.csv"
)

if [[ "${SKIP_EXISTING}" == "0" ]]; then
  existing_count="$(
    find "${COLLECTED_ROOT}" -maxdepth 1 -type f -name "${ANNOTATED_PATTERN}" |
      wc -l |
      tr -d ' '
  )"
  if [[ "${existing_count}" -gt 0 ]]; then
    echo "Collected outputs already exist (${existing_count})." >&2
    echo "Set SKIP_EXISTING=1 to resume without overwriting." >&2
    exit 1
  fi
fi

echo
echo "[3/3] Collecting annotated H5 files for Stage 5"
"${collect_cmd[@]}" 2>&1 | tee "${LOG_ROOT}/03_collect.log"

expected_count="$(
  "${PYTHON}" - "${SOURCE_MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
print(sum((row.get("enabled") or "").strip().lower() in {"1", "true", "yes"} for row in rows))
PY
)"
collected_count="$(
  find "${COLLECTED_ROOT}" -maxdepth 1 -type f -name "${ANNOTATED_PATTERN}" |
    wc -l |
    tr -d ' '
)"

echo
echo "Dataset build summary"
echo "  expected videos : ${expected_count}"
echo "  collected videos: ${collected_count}"
echo "  collected root  : ${COLLECTED_ROOT}"
echo "  H5 pattern      : ${ANNOTATED_PATTERN}"

if [[ "${collected_count}" -ne "${expected_count}" ]]; then
  echo "Collected H5 count does not match the enabled manifest count." >&2
  exit 1
fi

echo "Stage 4 global+local ablation dataset build completed."
