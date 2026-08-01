#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kodaira/anaconda3/envs/dualtrack311/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/data/3d_projects/models/Stage2to4}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset/stage4_training_ablation/260711}"
SAMPLING_RUN_NAME="${SAMPLING_RUN_NAME:-global_local_l75_w31_c12_area15}"
RUN_NAME="${RUN_NAME:-${SAMPLING_RUN_NAME}_bboxrank_v2}"
RUN_ROOT="${RUN_ROOT:-${DATASET_ROOT}/${RUN_NAME}}"
ANNOTATED_DIR="${ANNOTATED_DIR:-${RUN_ROOT}/annotated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/annotation_textures}"

POINT_CLOUD_TAG="${POINT_CLOUD_TAG:-foreground_combined_v2_${SAMPLING_RUN_NAME}_bboxrank_v2}"
ANNOTATED_PATTERN="${ANNOTATED_PATTERN:-*_pointcloud_annotated_${POINT_CLOUD_TAG}.h5}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/summary.csv}"

# Comma-separated video names. Empty means all annotated H5 files.
# Example:
# VIDEO_NAMES="20250403_114201_376,20250624_132811_8530"
VIDEO_NAMES="${VIDEO_NAMES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ ! -d "${ANNOTATED_DIR}" ]]; then
  echo "Annotated H5 directory not found: ${ANNOTATED_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

input_args=(
  --annotated_dir "${ANNOTATED_DIR}"
  --pattern "${ANNOTATED_PATTERN}"
  --recursive
)

if [[ -n "${VIDEO_NAMES}" ]]; then
  selection_list="${OUTPUT_ROOT}/selected_annotated_h5.txt"

  "${PYTHON}" - \
    "${ANNOTATED_DIR}" \
    "${ANNOTATED_PATTERN}" \
    "${VIDEO_NAMES}" \
    "${selection_list}" <<'PY'
from pathlib import Path
import sys

annotated_dir = Path(sys.argv[1])
pattern = sys.argv[2]
video_names = [value.strip() for value in sys.argv[3].split(",") if value.strip()]
selection_list = Path(sys.argv[4])

all_paths = sorted(path.resolve() for path in annotated_dir.rglob(pattern))
selected = []
missing = []

for video_name in video_names:
    matches = [
        path
        for path in all_paths
        if path.parent.name == video_name
        or path.name.startswith(f"{video_name}_")
    ]
    if len(matches) != 1:
        missing.append((video_name, len(matches)))
        continue
    selected.append(matches[0])

if missing:
    details = ", ".join(f"{name} ({count} matches)" for name, count in missing)
    raise SystemExit(f"Could not resolve exactly one annotated H5: {details}")

selection_list.write_text(
    "".join(f"{path}\n" for path in selected),
    encoding="utf-8",
)
print(f"Selected annotated H5 files: {len(selected)}")
for path in selected:
    print(f"  {path}")
PY

  input_args=(--h5_list "${selection_list}")
fi

skip_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_args=(--skip_existing)
fi

cd "${REPO_ROOT}"

"${PYTHON}" \
  pseudo3d/batch/export/batch_export_annotation_mask_visualization.py \
  "${input_args[@]}" \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${SUMMARY_CSV}" \
  --output_subdir_template "{parent_name}" \
  --output_filename_template "{visualization_stem}.obj" \
  --texture_dir_name_template "frames" \
  --xml_frame_number_offsets "1" \
  --xml_frame_id_source frame_index \
  --xml_annotation_dir_name annotations_renamed \
  --strict_xml_annotation_dir \
  --contour_preset percentile85_area100 \
  --contour_percentile 85 \
  --contour_min_component_area 100 \
  --contour_min_area 20 \
  --contour_min_alpha 1 \
  --contour_open_ksize 3 \
  --contour_close_ksize 5 \
  --contour_denoise median \
  --contour_denoise_ksize 3 \
  --draw_point_cloud_samples \
  --sample_point_radius 0 \
  --sample_point_alpha 0.9 \
  --global_sample_color "255,255,255" \
  --local_percentile_sample_color "64,200,255" \
  --tophat_sample_color "0,140,255" \
  --context_grid_sample_color "160,160,160" \
  --bbox_color "255,220,0" \
  --bbox_thickness 1 \
  --contour_color "0,255,0" \
  --ranked_global_contour_color "0,255,0" \
  --ranked_local_contour_color "255,0,255" \
  --ranked_shared_contour_color "255,255,0" \
  --contour_thickness 1 \
  --mask_color "255,32,32" \
  --mask_alpha 0.35 \
  "${skip_args[@]}" \
  --continue_on_error

echo
echo "Annotation texture visualization complete."
echo "  input : ${ANNOTATED_DIR}"
echo "  output: ${OUTPUT_ROOT}"
echo "  images: ${OUTPUT_ROOT}/<video_name>/frames/annotation_frame_*.png"
echo "  summary: ${SUMMARY_CSV}"
