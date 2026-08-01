#!/usr/bin/env bash
set -euo pipefail

# Reuse the existing global+local point clouds and rebuild only annotation and
# collected H5 files with the BBox-ranked global/local teacher v2.

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${DATASET_ROOT}/stage4_sampling_parameter_sweep/260711/manifests/train_manifest.csv}"
VOC_XML_ROOT="${VOC_XML_ROOT:-/mnt/data/Data_hbl/predirectory_for_coco_annotation_leg/260630/260630_pseudo3d}"

SAMPLING_RUN_NAME="${SAMPLING_RUN_NAME:-global_local_l75_w31_c12_area15}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SAMPLING_RUN_NAME}}"
POINT_CLOUD_ROOT="${POINT_CLOUD_ROOT:-${SOURCE_RUN_ROOT}/point_cloud}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/stage4_training_ablation/260711/${SAMPLING_RUN_NAME}_bboxrank_v2}"
ANNOTATED_ROOT="${OUTPUT_ROOT}/annotated"
COLLECTED_ROOT="${OUTPUT_ROOT}/collected"
LOG_ROOT="${OUTPUT_ROOT}/logs"

POINT_CLOUD_TAG="foreground_combined_v2_${SAMPLING_RUN_NAME}"
RAW_H5_TEMPLATE="{video_name}_pointcloud_${POINT_CLOUD_TAG}.h5"
ANNOTATED_TAG="${POINT_CLOUD_TAG}_bboxrank_v2"
ANNOTATED_H5_TEMPLATE="{video_name}_pointcloud_annotated_${ANNOTATED_TAG}.h5"
ANNOTATED_PATTERN="*_pointcloud_annotated_${ANNOTATED_TAG}.h5"

# Optional comma-separated subset. Empty means every enabled manifest row.
VIDEO_NAMES="${VIDEO_NAMES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REQUIRE_FULL_DATASET="${REQUIRE_FULL_DATASET:-1}"
EXPECTED_FULL_DATASET_FILES="${EXPECTED_FULL_DATASET_FILES:-182}"

cd "${REPO_ROOT}"
mkdir -p "${ANNOTATED_ROOT}" "${COLLECTED_ROOT}" "${LOG_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "Source manifest not found: ${SOURCE_MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${POINT_CLOUD_ROOT}" ]]; then
  echo "Point-cloud root not found: ${POINT_CLOUD_ROOT}" >&2
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
if [[ "${REQUIRE_FULL_DATASET}" != "0" && "${REQUIRE_FULL_DATASET}" != "1" ]]; then
  echo "REQUIRE_FULL_DATASET must be 0 or 1, got: ${REQUIRE_FULL_DATASET}" >&2
  exit 1
fi
if [[ "${REQUIRE_FULL_DATASET}" == "1" && -n "${VIDEO_NAMES}" ]]; then
  echo "VIDEO_NAMES cannot be used when REQUIRE_FULL_DATASET=1." >&2
  echo "Unset VIDEO_NAMES for the 260711 full-dataset build." >&2
  exit 1
fi

effective_manifest="${SOURCE_MANIFEST}"
if [[ -n "${VIDEO_NAMES}" ]]; then
  effective_manifest="${OUTPUT_ROOT}/selected_manifest.csv"
  "${PYTHON}" - \
    "${SOURCE_MANIFEST}" \
    "${VIDEO_NAMES}" \
    "${effective_manifest}" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
requested = [value.strip() for value in sys.argv[2].split(",") if value.strip()]
output = Path(sys.argv[3])

with source.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise SystemExit("Source manifest has no header")
    rows = list(reader)
    fieldnames = list(reader.fieldnames)

by_name = {str(row.get("video_name", "")).strip(): row for row in rows}
missing = [name for name in requested if name not in by_name]
if missing:
    raise SystemExit(f"VIDEO_NAMES not found in manifest: {missing}")

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for name in requested:
        writer.writerow(by_name[name])
print(f"Selected manifest rows: {len(requested)}")
PY
fi

manifest_enabled_count="$(
  "${PYTHON}" - "${effective_manifest}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))

def enabled(row):
    value = str(row.get("enabled", "true")).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}

print(sum(enabled(row) for row in rows))
PY
)"

if [[ "${REQUIRE_FULL_DATASET}" == "1" && "${manifest_enabled_count}" -ne "${EXPECTED_FULL_DATASET_FILES}" ]]; then
  echo "Full-dataset manifest count mismatch:" >&2
  echo "  expected=${EXPECTED_FULL_DATASET_FILES}" >&2
  echo "  actual=${manifest_enabled_count}" >&2
  echo "  manifest=${effective_manifest}" >&2
  exit 1
fi

skip_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_args+=(--skip_existing)
fi

echo "Stage 4 BBox-ranked annotation rebuild"
echo "  source manifest : ${effective_manifest}"
echo "  point clouds    : ${POINT_CLOUD_ROOT}"
echo "  VOC XML root    : ${VOC_XML_ROOT}"
echo "  output root     : ${OUTPUT_ROOT}"
echo "  skip existing   : ${SKIP_EXISTING}"
echo "  enabled videos  : ${manifest_enabled_count}"
echo "  require full    : ${REQUIRE_FULL_DATASET}"

annotate_cmd=(
  "${PYTHON}"
  pseudo3d/batch/annotation/batch_annotate_pseudo3d_point_cloud.py
  --h5_list "${effective_manifest}"
  --point_cloud_root "${POINT_CLOUD_ROOT}"
  --voc_xml_root "${VOC_XML_ROOT}"
  --output_root "${ANNOTATED_ROOT}"
  --summary_csv "${ANNOTATED_ROOT}/summary.csv"
  --video_suffix_to_strip "_ts448_oym96_corr"
  --point_cloud_detail "${SAMPLING_RUN_NAME}"
  --point_cloud_subdir_template "{video_name}"
  --point_cloud_filename_template "${RAW_H5_TEMPLATE}"
  --output_subdir_template "{video_name}"
  --output_filename_template "${ANNOTATED_H5_TEMPLATE}"
  --geometry_key corr
  --label_mode bbox_ranked_global_local
  --xml_frame_number_offsets "1"
  --xml_frame_id_source frame_index
  --xml_annotation_dir_name annotations_renamed
  --strict_xml_annotation_dir
  --contour_preset percentile85_area100
  --contour_min_area 20
  --bbox_inside_non_contour_label ignore
  --ranked_local_window_size 31
  --ranked_local_percentile 75
  --ranked_local_min_contrast 12
  --ranked_local_cleanup_open_ksize 3
  --ranked_local_cleanup_close_ksize 5
  --ranked_local_cleanup_morph_shape ellipse
  --ranked_local_cleanup_min_component_area 15
  --ranked_min_area_ratio 0.02
  --ranked_sufficient_area_ratio 0.10
  --ranked_max_center_distance_norm 0.50
  --ranked_center_weight 0.75
  --ranked_area_weight 0.25
  "${skip_args[@]}"
)

echo
echo "[1/2] Building BBox-ranked annotation H5 files"
"${annotate_cmd[@]}" 2>&1 | tee "${LOG_ROOT}/01_annotation.log"

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
  collect_cmd+=(--overwrite)
fi

echo
echo "[2/2] Collecting BBox-ranked annotation H5 files"
"${collect_cmd[@]}" 2>&1 | tee "${LOG_ROOT}/02_collect.log"

annotated_count="$(
  find "${ANNOTATED_ROOT}" -type f -name "${ANNOTATED_PATTERN}" |
    wc -l |
    tr -d ' '
)"
collected_count="$(
  find "${COLLECTED_ROOT}" -maxdepth 1 -type f -name "${ANNOTATED_PATTERN}" |
    wc -l |
    tr -d ' '
)"
if [[ "${REQUIRE_FULL_DATASET}" == "1" ]]; then
  if [[ "${annotated_count}" -ne "${EXPECTED_FULL_DATASET_FILES}" ]]; then
    echo "Annotated full-dataset count mismatch: ${annotated_count}" >&2
    exit 1
  fi
  if [[ "${collected_count}" -ne "${EXPECTED_FULL_DATASET_FILES}" ]]; then
    echo "Collected full-dataset count mismatch: ${collected_count}" >&2
    exit 1
  fi
fi

echo
echo "BBox-ranked annotation rebuild complete."
echo "  annotated : ${ANNOTATED_ROOT}"
echo "  collected : ${COLLECTED_ROOT}"
echo "  summary   : ${ANNOTATED_ROOT}/summary.csv"
echo "  files     : annotated=${annotated_count}, collected=${collected_count}"
