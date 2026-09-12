#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Padding-free Stage 5 training on collected annotated pseudo-3D H5 files.
#
# This script expects the accepted Stage 4 teacher v6 H5 files collected into
# one flat directory by:
#
#   Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v6_xml_invalidation.sh
#
# Edit the path/parameter blocks below, then run:
#
#   bash /mnt/data/3d_projects/models/Stage5/train_stage5.sh
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run from Stage5 repository root
# ------------------------------------------------------------
SCRIPT_DIR="/mnt/data/3d_projects/models/Stage5"
cd "${SCRIPT_DIR}"

PYTHON="/home/kodaira/anaconda3/envs/dualtrack311/bin/python"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  CONDA_PREFIX="$(dirname "$(dirname "${PYTHON}")")"
fi

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"

TORCH_LIB="$("${PYTHON}" - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"

DATE="${DATE:-260711}"
EX_DATE="${EX_DATE:-260908}"

# ------------------------------------------------------------
# input path info
# ------------------------------------------------------------
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/3d_projects/pseudo3d_dataset}"

STAGE4_SAMPLING_RUN="${STAGE4_SAMPLING_RUN:-global_local_l75_w31_c12_area15}"
STAGE4_TEACHER="${STAGE4_TEACHER:-bboxrank_v6_cvat_authoritative_xml_invalidation_v1}"
STAGE4_RUN_NAME="${STAGE4_RUN_NAME:-${STAGE4_SAMPLING_RUN}_${STAGE4_TEACHER}}"
INPUT_DIR="${INPUT_DIR:-${DATASET_ROOT}/stage4_training_ablation/${DATE}/${STAGE4_RUN_NAME}/collected}"

MODE="${MODE:-foreground}"
H5_PATTERN="${H5_PATTERN:-*_pointcloud_annotated_${MODE}_combined_v2_${STAGE4_RUN_NAME}.h5}"
EXPECTED_INPUT_FILES="${EXPECTED_INPUT_FILES:-181}"
EXPECTED_CVAT_VIDEOS="${EXPECTED_CVAT_VIDEOS:-59}"
EXPECTED_CVAT_FRAMES="${EXPECTED_CVAT_FRAMES:-3014}"
EXPECTED_XML_INVALIDATED_FRAMES="${EXPECTED_XML_INVALIDATED_FRAMES:-7}"
EXPECTED_XML_INVALIDATED_VIDEOS="${EXPECTED_XML_INVALIDATED_VIDEOS:-2}"
EXPECTED_XML_REMOVED_POSITIVE_POINTS="${EXPECTED_XML_REMOVED_POSITIVE_POINTS:-2124}"
EXPECTED_XML_REMOVED_BBOX_ROWS="${EXPECTED_XML_REMOVED_BBOX_ROWS:-7}"
EXPECTED_EXCLUDED_VIDEO="${EXPECTED_EXCLUDED_VIDEO:-20250626_090758_8000}"

# Optional quick subset. 0 means all files.
MAX_TRAIN_FILES=0

# Optional validation split from INPUT_DIR. 0.0 means use all files for train.
VAL_FRACTION=0.1
MAX_VAL_FILES=0

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/stage5_runs/${EX_DATE}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
PREFIX="w16_s8_bboxrankv6_cvatxmlinv_glocal_ce_smooth00_auto_weight_lr1e3_ep${EPOCHS}_bs${BATCH_SIZE}_acc${GRADIENT_ACCUMULATION_STEPS}_nopad"
EXPERIMENT_NAME="pointnext_s_EX${EX_DATE}_${DATE}_${PREFIX}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

# ------------------------------------------------------------
# Model / training parameters.
# ------------------------------------------------------------
MODEL_NAME="pointnext_s"
FEATURES="intensity,confidence"
NUM_POINTS=0

# Conservative Stage5 fine-tuning LR. PointNeXt S3DIS uses 1e-2, but Stage5
# starts from weak labels and smaller batches, so keep 1e-3 for the smoke run.
LR=1e-3
WEIGHT_DECAY=1e-4
NUM_WORKERS=4
DEVICE="cuda"
SEED=42
GRAD_CLIP_NORM=10
SAVE_EVERY=10

# Lightweight baseline size.
# For MODEL_NAME="pointnext_s", WIDTH/EXPANSION/DROPOUT are used by the
# PointNeXt-S wrapper. DEPTH and global-context settings are ignored there.
WIDTH=32
DEPTH=6
EXPANSION=4
DROPOUT=0.0

# Optional Stage5 checkpoint initialization.
# Leave empty to train from scratch.
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data/3d_projects/models/Stage5/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init.pt}"
STRICT_CHECKPOINT=1

# PointNeXt-S knobs. Ignored by mlp_baseline.
POINTNEXT_RADIUS=0.1
POINTNEXT_NSAMPLE=32
POINTNEXT_SA_LAYERS=2
POINTNEXT_SA_USE_RES=1

# Normalization used throughout the PointNeXt-S encoder/decoder/head (S5-11).
# "batchnorm" is the historical default. For a GroupNorm pilot, set this to
# "groupnorm" and point INIT_CHECKPOINT at the GroupNorm-transferred init
# checkpoint produced by checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.sh.
POINTNEXT_NORM="${POINTNEXT_NORM:-batchnorm}"
POINTNEXT_NORM_GROUPS="${POINTNEXT_NORM_GROUPS:-8}"

# Sampling knobs.
# In WINDOW_MODE="overlap", each frame-order window is one training sample and
# NUM_POINTS/SAMPLING_MODE are kept only for legacy non-window runs.
WINDOW_MODE="overlap"
WINDOW_SIZE_FRAMES=16
WINDOW_STRIDE_FRAMES=8
INCLUDE_TAIL_WINDOW=1
SAMPLING_MODE="video"
FRAME_WINDOW_SIZE=""
POSITIVE_OVERSAMPLE_RATIO=0.0
EXCLUDE_IGNORE_IN_SAMPLING=0

# Loss knobs.
# CLASS_WEIGHT:
#   ""       : no class weight
#   "auto"   : PointNeXt-style 1 / (class_frequency + epsilon), normalized to mean 1
#   "1,4"    : manual class weights
CLASS_WEIGHT="auto"
AUTO_CLASS_WEIGHT_EPSILON=0.02
NORMALIZE_AUTO_CLASS_WEIGHT=1
LABEL_SMOOTHING=0.0

if [[ "${BATCH_SIZE}" -ne 1 ]]; then
  echo "Padding-free PointNeXt baseline requires BATCH_SIZE=1; got ${BATCH_SIZE}" >&2
  exit 1
fi
if [[ "${GRADIENT_ACCUMULATION_STEPS}" -le 0 ]]; then
  echo "GRADIENT_ACCUMULATION_STEPS must be positive" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  echo "Run Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v6_xml_invalidation.sh first, or set INPUT_DIR." >&2
  exit 1
fi

num_inputs="$(find "${INPUT_DIR}" -maxdepth 1 -type f -name "${H5_PATTERN}" | wc -l | tr -d ' ')"
if [[ "${num_inputs}" -eq 0 ]]; then
  echo "No input H5 files matched:" >&2
  echo "  INPUT_DIR=${INPUT_DIR}" >&2
  echo "  H5_PATTERN=${H5_PATTERN}" >&2
  exit 1
fi
if [[ "${num_inputs}" -ne "${EXPECTED_INPUT_FILES}" ]]; then
  echo "Stage 5 requires the complete ${DATE} teacher v6 dataset:" >&2
  echo "  expected files=${EXPECTED_INPUT_FILES}" >&2
  echo "  matched files=${num_inputs}" >&2
  echo "  INPUT_DIR=${INPUT_DIR}" >&2
  echo "  H5_PATTERN=${H5_PATTERN}" >&2
  exit 1
fi

"${PYTHON}" - \
  "${INPUT_DIR}" \
  "${H5_PATTERN}" \
  "${EXPECTED_INPUT_FILES}" \
  "${STAGE4_TEACHER}" \
  "${EXPECTED_CVAT_VIDEOS}" \
  "${EXPECTED_CVAT_FRAMES}" \
  "${EXPECTED_XML_INVALIDATED_FRAMES}" \
  "${EXPECTED_XML_INVALIDATED_VIDEOS}" \
  "${EXPECTED_XML_REMOVED_POSITIVE_POINTS}" \
  "${EXPECTED_XML_REMOVED_BBOX_ROWS}" \
  "${EXPECTED_EXCLUDED_VIDEO}" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

input_dir = Path(sys.argv[1])
pattern = sys.argv[2]
expected = int(sys.argv[3])
expected_teacher = sys.argv[4]
expected_cvat_videos = int(sys.argv[5])
expected_cvat_frames = int(sys.argv[6])
expected_invalidated_frames = int(sys.argv[7])
expected_invalidated_videos = int(sys.argv[8])
expected_removed_positive = int(sys.argv[9])
expected_removed_bbox_rows = int(sys.argv[10])
expected_excluded_video = sys.argv[11]
paths = sorted(input_dir.glob(pattern))
if len(paths) != expected:
    raise SystemExit(f"H5 preflight count mismatch: {len(paths)} != {expected}")

def text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


videos = set()
cvat_videos = 0
cvat_frames = 0
invalidated_videos = 0
invalidated_frames = 0
removed_positive = 0
removed_bbox_rows = 0
manifest_hashes = set()
manifest_fingerprints = set()
for path in paths:
    with h5py.File(path, "r") as handle:
        label_mode = text(handle.attrs.get("label_mode", ""))
        teacher_schema = text(handle.attrs.get("contour_teacher_schema", ""))
        label_source = text(handle.attrs.get("label_source", ""))
        video_name = text(handle.attrs.get("video_name", ""))
        if label_mode != "bbox_ranked_global_local":
            raise SystemExit(f"Unexpected label_mode in {path}: {label_mode}")
        if teacher_schema != expected_teacher:
            raise SystemExit(
                f"Unexpected contour_teacher_schema in {path}: {teacher_schema}"
            )
        if label_source != expected_teacher:
            raise SystemExit(
                f"Unexpected label_source in {path}: {label_source}"
            )
        required = (
            "point_cloud/frame_order",
            "annotation/point_label",
            "annotation/valid_mask",
            "frame_annotation/frame_order",
            "manual_review_fullvideo_frames/frame_order",
            "manual_review_fullvideo_frames/positive_outside_cvat_mask_points",
            "xml_annotation_invalidation/frame_order",
            "xml_annotation_invalidation/after_positive_points",
            "xml_annotation_invalidation/after_ignore_points",
            "xml_annotation_invalidation/before_positive_points",
            "xml_annotation_invalidation/removed_frame_annotation_rows",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise SystemExit(f"Missing label-policy datasets in {path}: {missing}")
        annotation_label_source = text(
            handle["annotation"].attrs.get("label_source", "")
        )
        if annotation_label_source != expected_teacher:
            raise SystemExit(
                f"Unexpected annotation label_source in {path}: "
                f"{annotation_label_source}"
            )

        frame_order = handle["point_cloud/frame_order"][:].astype(np.int64)
        point_label = handle["annotation/point_label"][:].astype(np.int8)
        valid_mask = handle["annotation/valid_mask"][:].astype(bool)
        if not (frame_order.shape == point_label.shape == valid_mask.shape):
            raise SystemExit(
                f"point annotation shape mismatch in {path}: "
                f"{frame_order.shape}, {point_label.shape}, {valid_mask.shape}"
            )
        labels_found = set(int(value) for value in np.unique(point_label))
        if not labels_found <= {-1, 0, 1}:
            raise SystemExit(
                f"Unexpected labels in {path}: {sorted(labels_found)}"
            )
        if not np.array_equal(valid_mask, point_label != -1):
            raise SystemExit(f"valid_mask differs from point_label != -1 in {path}")

        review = handle["manual_review_fullvideo_frames"]
        review_count = len(review["frame_order"])
        if np.any(
            review["positive_outside_cvat_mask_points"][:].astype(np.int64) != 0
        ):
            raise SystemExit(f"CVAT-mask-external positive remains in {path}")
        cvat_frames += review_count
        cvat_videos += int(review_count > 0)

        invalidation = handle["xml_annotation_invalidation"]
        invalidation_count = len(invalidation["frame_order"])
        invalidated_frames += invalidation_count
        invalidated_videos += int(invalidation_count > 0)
        manifest_hash = text(invalidation.attrs.get("manifest_sha256", ""))
        manifest_fingerprint = text(
            invalidation.attrs.get("manifest_fingerprint", "")
        )
        if not manifest_hash or not manifest_fingerprint:
            raise SystemExit(f"Missing invalidation manifest identity in {path}")
        manifest_hashes.add(manifest_hash)
        manifest_fingerprints.add(manifest_fingerprint)
        expected_group_attrs = {
            "teacher_version": expected_teacher,
            "source_teacher_version": "bboxrank_v5_cvat_authoritative_v1",
            "policy": "explicit_deleted_xml_frame_tombstone",
            "frame_authority": "xml_deletion_manifest",
        }
        for name, expected_value in expected_group_attrs.items():
            actual = text(invalidation.attrs.get(name, ""))
            if actual != expected_value:
                raise SystemExit(
                    f"Unexpected xml invalidation {name} in {path}: {actual!r}"
                )

        if invalidation_count:
            after_positive = invalidation["after_positive_points"][:].astype(np.int64)
            after_ignore = invalidation["after_ignore_points"][:].astype(np.int64)
            if np.any(after_positive != 0) or np.any(after_ignore != 0):
                raise SystemExit(f"Invalidated frame retains non-background labels in {path}")
            for order in invalidation["frame_order"][:].astype(np.int64):
                selected = frame_order == order
                if not np.any(selected) or not np.all(point_label[selected] == 0):
                    raise SystemExit(
                        f"Invalidated frame_order={order} is not all background in {path}"
                    )
                if not np.all(valid_mask[selected]):
                    raise SystemExit(
                        f"Invalidated frame_order={order} is not fully valid in {path}"
                    )
            removed_positive += int(
                invalidation["before_positive_points"][:].astype(np.int64).sum()
            )
            removed_bbox_rows += int(
                invalidation["removed_frame_annotation_rows"][:]
                .astype(np.int64)
                .sum()
            )
        if not video_name:
            raise SystemExit(f"Missing video_name in {path}")
        if video_name in videos:
            raise SystemExit(f"Duplicate video_name: {video_name}")
        videos.add(video_name)
if expected_excluded_video in videos:
    raise SystemExit(f"Excluded crop-failure video is present: {expected_excluded_video}")
if len(manifest_hashes) != 1 or len(manifest_fingerprints) != 1:
    raise SystemExit(
        "v6 files do not share one invalidation manifest identity: "
        f"sha={len(manifest_hashes)}, fingerprint={len(manifest_fingerprints)}"
    )
expected_counts = {
    "cvat_videos": (cvat_videos, expected_cvat_videos),
    "cvat_frames": (cvat_frames, expected_cvat_frames),
    "invalidated_frames": (invalidated_frames, expected_invalidated_frames),
    "invalidated_videos": (invalidated_videos, expected_invalidated_videos),
    "removed_positive": (removed_positive, expected_removed_positive),
    "removed_bbox_rows": (removed_bbox_rows, expected_removed_bbox_rows),
}
for name, (actual, expected_value) in expected_counts.items():
    if actual != expected_value:
        raise SystemExit(f"v6 {name} mismatch: {actual} != {expected_value}")
print(
    "Stage 5 teacher v6 input preflight passed: "
    f"files={len(paths)}, videos={len(videos)}, "
    f"cvat_videos={cvat_videos}, cvat_frames={cvat_frames}, "
    f"invalidated_videos={invalidated_videos}, "
    f"invalidated_frames={invalidated_frames}, "
    f"removed_positive={removed_positive}, removed_bbox_rows={removed_bbox_rows}"
)
PY

mkdir -p "${OUTPUT_DIR}"

echo "Stage5 padding-free training"
echo "  input dir      : ${INPUT_DIR}"
echo "  h5 pattern     : ${H5_PATTERN}"
echo "  matched files  : ${num_inputs}"
echo "  teacher        : ${STAGE4_TEACHER}"
echo "  label policy   : CVAT-reviewed=authoritative; unreviewed=inherited; XML-invalidated=background"
echo "  output dir     : ${OUTPUT_DIR}"
echo "  model          : ${MODEL_NAME}"
echo "  epochs         : ${EPOCHS}"
echo "  window mode    : ${WINDOW_MODE}"
echo "  window frames  : size=${WINDOW_SIZE_FRAMES}, stride=${WINDOW_STRIDE_FRAMES}, include_tail=${INCLUDE_TAIL_WINDOW}"
echo "  num points     : ${NUM_POINTS} (ignored when window mode is overlap)"
echo "  physical batch : ${BATCH_SIZE}"
echo "  grad accum     : ${GRADIENT_ACCUMULATION_STEPS}"
echo "  effective batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) samples"
echo "  padding policy : none (physical batch size must remain 1)"
echo "  lr/weight decay: ${LR}/${WEIGHT_DECAY}"
echo "  grad clip norm : ${GRAD_CLIP_NORM:-none}"
echo "  save every     : ${SAVE_EVERY}"
echo "  device         : ${DEVICE}"
echo "  val fraction   : ${VAL_FRACTION}"
echo "  class weight   : ${CLASS_WEIGHT:-none}"
echo "  label smoothing: ${LABEL_SMOOTHING}"
echo "  init checkpoint: ${INIT_CHECKPOINT:-none}"
if [[ "${CLASS_WEIGHT}" == "auto" || "${CLASS_WEIGHT}" == "pointnext_auto" ]]; then
  echo "  auto cls weight: epsilon=${AUTO_CLASS_WEIGHT_EPSILON}, normalize=${NORMALIZE_AUTO_CLASS_WEIGHT}"
fi
if [[ "${MODEL_NAME}" == "pointnext_s" ]]; then
  echo "  pointnext      : radius=${POINTNEXT_RADIUS}, nsample=${POINTNEXT_NSAMPLE}, sa_layers=${POINTNEXT_SA_LAYERS}, sa_use_res=${POINTNEXT_SA_USE_RES}, norm=${POINTNEXT_NORM}, norm_groups=${POINTNEXT_NORM_GROUPS}"
fi

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/train_stage5.py"
  --train_dir "${INPUT_DIR}"
  --h5_pattern "${H5_PATTERN}"
  --output_dir "${OUTPUT_DIR}"
  --model "${MODEL_NAME}"
  --features "${FEATURES}"
  --num_points "${NUM_POINTS}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --width "${WIDTH}"
  --depth "${DEPTH}"
  --expansion "${EXPANSION}"
  --dropout "${DROPOUT}"
  --pointnext_radius "${POINTNEXT_RADIUS}"
  --pointnext_nsample "${POINTNEXT_NSAMPLE}"
  --pointnext_sa_layers "${POINTNEXT_SA_LAYERS}"
  --pointnext_norm "${POINTNEXT_NORM}"
  --pointnext_norm_groups "${POINTNEXT_NORM_GROUPS}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --save_every "${SAVE_EVERY}"
  --sampling_mode "${SAMPLING_MODE}"
  --window_mode "${WINDOW_MODE}"
  --window_size_frames "${WINDOW_SIZE_FRAMES}"
  --window_stride_frames "${WINDOW_STRIDE_FRAMES}"
  --positive_oversample_ratio "${POSITIVE_OVERSAMPLE_RATIO}"
  --val_fraction "${VAL_FRACTION}"
  --max_train_files "${MAX_TRAIN_FILES}"
  --max_val_files "${MAX_VAL_FILES}"
  --auto_class_weight_epsilon "${AUTO_CLASS_WEIGHT_EPSILON}"
  --label_smoothing "${LABEL_SMOOTHING}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
    echo "Initial checkpoint not found: ${INIT_CHECKPOINT}" >&2
    exit 1
  fi
  cmd+=(--checkpoint "${INIT_CHECKPOINT}")
fi

if [[ "${STRICT_CHECKPOINT}" == "1" ]]; then
  cmd+=(--strict_checkpoint)
fi

if [[ -n "${FRAME_WINDOW_SIZE}" ]]; then
  cmd+=(--frame_window_size "${FRAME_WINDOW_SIZE}")
fi

if [[ "${EXCLUDE_IGNORE_IN_SAMPLING}" == "1" ]]; then
  cmd+=(--exclude_ignore_in_sampling)
fi

if [[ "${INCLUDE_TAIL_WINDOW}" != "1" ]]; then
  cmd+=(--no_include_tail_window)
fi

if [[ "${POINTNEXT_SA_USE_RES}" != "1" ]]; then
  cmd+=(--no_pointnext_sa_use_res)
fi

if [[ -n "${CLASS_WEIGHT}" ]]; then
  cmd+=(--class_weight "${CLASS_WEIGHT}")
fi

if [[ "${NORMALIZE_AUTO_CLASS_WEIGHT}" != "1" ]]; then
  cmd+=(--no_normalize_auto_class_weight)
fi

"${cmd[@]}"

echo "Done."
echo "  best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "  last checkpoint: ${OUTPUT_DIR}/last.pt"
