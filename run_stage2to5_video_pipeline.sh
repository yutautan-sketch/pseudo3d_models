#!/usr/bin/env bash
set -euo pipefail

# Edit the paths below, then run:
#   bash /workspace/run_stage2to5_video_pipeline.sh

# ------------------------------------------------------------
# Path settings (edit these for your environment)
# ------------------------------------------------------------
INPUT_DIR="/mnt/data/Data_hbl/example_videos/t/1/crop"

STAGE2TO4_ROOT="/mnt/data/3d_projects/models/Stage2to4"
STAGE5_ROOT="/mnt/data/3d_projects/models/Stage5"
PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"
DATASET_NAME="t_1_crop"

DATE="260711"
EX_DATE="260714"

# Same checkpoint convention as the other Stage2to4 pipeline scripts.
# batch_infer_video_to_pseudo3d_h5.py resolves this environment variable.
export DUALTRACK_CHECKPOINT_DIR="${DUALTRACK_CHECKPOINT_DIR:-/mnt/data/3d_projects/dualtrack_checkpoints}"
export DUALTRACK_CHECKPOINT="${DUALTRACK_CHECKPOINT:-${DUALTRACK_CHECKPOINT_DIR}/dualtrack_final.pt}"

# ------------------------------------------------------------
# Stage2to4 output paths
# ------------------------------------------------------------
DATASET_ROOT="/mnt/data/3d_projects/pseudo3d_dataset"
PSEUDO3D_H5_DIR="${DATASET_ROOT}/pseudo3d_outputs/${DATASET_NAME}"
POINT_CLOUD_ROOT="${DATASET_ROOT}/pseudo3d_visualizations/${DATASET_NAME}"
VIDEO_LIST="${PSEUDO3D_H5_DIR}/input_videos.txt"

# ------------------------------------------------------------
# Stage5 inference parameters
# ------------------------------------------------------------
MODEL_NAME="pointnext_s"
INFERENCE_MODE="window"
WINDOW_SIZE_FRAMES=20
WINDOW_STRIDE_FRAMES=10
CHUNK_SIZE=131072
DEVICE="cuda"

# ------------------------------------------------------------
# Stage5 checkpoint/output paths (same convention as infer_stage5.sh)
# ------------------------------------------------------------
RUN_ROOT="/mnt/data/3d_projects/stage5_runs"
PREFIX="w${WINDOW_SIZE_FRAMES}_s${WINDOW_STRIDE_FRAMES}_ce_smooth00_auto_weight_lr1e3_ep200"
EXPERIMENT_NAME="${EX_DATE}/${MODEL_NAME}_EX${EX_DATE}_${DATE}_${PREFIX}"
CHECKPOINT_NAME="checkpoint_epoch_0200.pt"
STAGE5_CHECKPOINT="${RUN_ROOT}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"
STAGE5_OUTPUT_ROOT="/mnt/data/3d_projects/stage5_predictions/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"
STAGE5_OUTPUT_DIR="${STAGE5_OUTPUT_ROOT}/${DATASET_NAME}"

OUTPUT_SUFFIX="_ts448_oym96_corr"
POINT_MODE="foreground"
POINT_CLOUD_PATTERN="*_pointcloud_${POINT_MODE}.h5"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: input directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${DUALTRACK_CHECKPOINT}" ]]; then
  echo "ERROR: DualTrack checkpoint not found: ${DUALTRACK_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${STAGE5_CHECKPOINT}" ]]; then
  echo "ERROR: Stage5 checkpoint not found: ${STAGE5_CHECKPOINT}" >&2
  exit 1
fi

mapfile -d '' -t INPUT_VIDEOS < <(
  find "${INPUT_DIR}" -type f \( -iname '*.mp4' \) -print0 | sort -z
)
if [[ "${#INPUT_VIDEOS[@]}" -eq 0 ]]; then
  echo "ERROR: no MP4 videos found under: ${INPUT_DIR}" >&2
  exit 1
fi

# Stage2to4 currently writes H5 files by video stem into one directory. Refuse
# ambiguous inputs instead of silently overwriting videos with identical stems.
declare -A SEEN_VIDEO_STEMS=()
for video_path in "${INPUT_VIDEOS[@]}"; do
  video_file="$(basename "${video_path}")"
  video_stem="${video_file%.*}"
  if [[ -n "${SEEN_VIDEO_STEMS[${video_stem}]:-}" ]]; then
    echo "ERROR: duplicate video stem '${video_stem}':" >&2
    echo "  ${SEEN_VIDEO_STEMS[${video_stem}]}" >&2
    echo "  ${video_path}" >&2
    exit 1
  fi
  SEEN_VIDEO_STEMS["${video_stem}"]="${video_path}"
done

mkdir -p "${PSEUDO3D_H5_DIR}" "${POINT_CLOUD_ROOT}" "${STAGE5_OUTPUT_DIR}"

: > "${VIDEO_LIST}"
for video_path in "${INPUT_VIDEOS[@]}"; do
  printf '%s\n' "${video_path}" >> "${VIDEO_LIST}"
done

echo "[1/4] Stage2to4: ${#INPUT_VIDEOS[@]} MP4 video(s) -> pseudo3D H5"
cd "${STAGE2TO4_ROOT}"
"${PYTHON}" pseudo3d/inference/batch_infer_video_to_pseudo3d_h5.py \
  --input_mode video \
  --video_list "${VIDEO_LIST}" \
  --output_dir "${PSEUDO3D_H5_DIR}" \
  --output_suffix "${OUTPUT_SUFFIX}" \
  --stride 2 \
  --max_frames 512 \
  --local_preprocess resize_shorter_then_center_crop \
  --local_target_short_side 448 \
  --local_crop_offset_y -96 \
  --local_crop_offset_x 0 \
  --connect_slices \
  --tracking_correction_preset stage2_v1 \
  --tracking_progress_axis_scale 2.5 \
  --skip_existing \
  --continue_on_error

echo "[2/4] Stage2to4: pseudo3D H5 -> textured OBJ / MTL / textures"
"${PYTHON}" pseudo3d/batch/export/batch_export_pseudo3d_visualization.py \
  --input_dir "${PSEUDO3D_H5_DIR}" \
  --pattern "*${OUTPUT_SUFFIX}.h5" \
  --output_root "${POINT_CLOUD_ROOT}" \
  --preset percentile85_area100 \
  --geometry_key corr \
  --texture_percentile 85 \
  --texture_min_component_area 100 \
  --export_tracking_debug_obj \
  --skip_existing \
  --continue_on_error

echo "[3/4] Stage2to4: pseudo3D H5 -> point-cloud H5"
"${PYTHON}" pseudo3d/batch/export/batch_export_pseudo3d_point_cloud.py \
  --input_dir "${PSEUDO3D_H5_DIR}" \
  --pattern "*${OUTPUT_SUFFIX}.h5" \
  --output_root "${POINT_CLOUD_ROOT}" \
  --video_suffix_to_strip "${OUTPUT_SUFFIX}" \
  --preset percentile85_area100 \
  --geometry_key corr \
  --point_mode "${POINT_MODE}" \
  --sample_stride 2 \
  --sampling_mode legacy \
  --include_context_grid \
  --context_grid_stride 4 \
  --context_grid_phase origin \
  --enable_local_percentile \
  --local_window_size 41 \
  --local_percentile 80 \
  --local_min_contrast 4 \
  --enable_tophat \
  --tophat_kernel_size 21 \
  --tophat_percentile 85 \
  --tophat_min_response 2 \
  --tophat_morph_shape ellipse \
  --texture_percentile 85 \
  --texture_min_component_area 100 \
  --output_h5_filename_template "{video_name}_pointcloud_${POINT_MODE}.h5" \
  --output_ply_filename_template "{video_name}_pointcloud_${POINT_MODE}.ply" \
  --summary_csv "${POINT_CLOUD_ROOT}/batch_export_summary.csv" \
  --skip_existing \
  --continue_on_error

mapfile -d '' -t POINT_CLOUD_H5_FILES < <(
  find "${POINT_CLOUD_ROOT}" -type f -name "${POINT_CLOUD_PATTERN}" -print0 | sort -z
)
if [[ "${#POINT_CLOUD_H5_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no point-cloud H5 files were generated." >&2
  exit 1
fi

echo "[4/4] Stage5: ${#POINT_CLOUD_H5_FILES[@]} point-cloud H5 file(s) -> predictions"
cd "${STAGE5_ROOT}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"
TORCH_LIB="$("${PYTHON}" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${STAGE5_ROOT}/external/PointNeXt:${STAGE5_ROOT}/external/PointNeXt/openpoints:${PYTHONPATH:-}"

for input_h5 in "${POINT_CLOUD_H5_FILES[@]}"; do
  input_name="$(basename "${input_h5}")"
  video_name="${input_name%_pointcloud_${POINT_MODE}.h5}"
  video_output_dir="${STAGE5_OUTPUT_DIR}/${video_name}"
  output_prefix="${video_output_dir}/${video_name}_stage5_pred_${MODEL_NAME}"
  mkdir -p "${video_output_dir}"

  "${PYTHON}" infer_stage5.py \
    --input_h5 "${input_h5}" \
    --checkpoint "${STAGE5_CHECKPOINT}" \
    --output_h5 "${output_prefix}.h5" \
    --output_ply "${output_prefix}.ply" \
    --output_positive_ply "${output_prefix}_positive.ply" \
    --model "${MODEL_NAME}" \
    --inference_mode "${INFERENCE_MODE}" \
    --window_size_frames "${WINDOW_SIZE_FRAMES}" \
    --window_stride_frames "${WINDOW_STRIDE_FRAMES}" \
    --chunk_size "${CHUNK_SIZE}" \
    --device "${DEVICE}" \
    --pointnext_radius 0.1 \
    --pointnext_nsample 32 \
    --pointnext_sa_layers 2 \
    --strict_checkpoint
done

echo
echo "Completed."
echo "  pseudo3D H5 : ${PSEUDO3D_H5_DIR}"
echo "  OBJ/point clouds: ${POINT_CLOUD_ROOT}"
echo "  Stage5 output: ${STAGE5_OUTPUT_DIR}"
echo "  OBJ files:"
find "${POINT_CLOUD_ROOT}" -type f -name '*.obj' -print | sort | sed 's/^/    /'
echo "  positive PLY files:"
find "${STAGE5_OUTPUT_DIR}" -type f -name '*_positive.ply' -print | sort | sed 's/^/    /'
