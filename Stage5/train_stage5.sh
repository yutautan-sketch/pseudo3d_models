#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Padding-free Stage 5 training on collected annotated pseudo-3D H5 files.
#
# This script expects the accepted Stage 4 teacher v7 H5 files collected into
# one flat directory by:
#
#   Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v7_crop_quality.sh
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
STAGE4_TEACHER="${STAGE4_TEACHER:-bboxrank_v7_cvat_authoritative_crop_quality_v1}"
STAGE4_RUN_NAME="${STAGE4_RUN_NAME:-${STAGE4_SAMPLING_RUN}_${STAGE4_TEACHER}}"
INPUT_DIR="${INPUT_DIR:-${DATASET_ROOT}/stage4_training_ablation/${DATE}/${STAGE4_RUN_NAME}/collected}"

MODE="${MODE:-foreground}"
H5_PATTERN="${H5_PATTERN:-*_pointcloud_annotated_${MODE}_combined_v2_${STAGE4_RUN_NAME}.h5}"
EXPECTED_INPUT_FILES="${EXPECTED_INPUT_FILES:-180}"
EXPECTED_CVAT_VIDEOS="${EXPECTED_CVAT_VIDEOS:-58}"
EXPECTED_CVAT_FRAMES="${EXPECTED_CVAT_FRAMES:-2960}"
EXPECTED_XML_INVALIDATED_FRAMES="${EXPECTED_XML_INVALIDATED_FRAMES:-7}"
EXPECTED_XML_INVALIDATED_VIDEOS="${EXPECTED_XML_INVALIDATED_VIDEOS:-2}"
EXPECTED_XML_REMOVED_POSITIVE_POINTS="${EXPECTED_XML_REMOVED_POSITIVE_POINTS:-2124}"
EXPECTED_XML_REMOVED_BBOX_ROWS="${EXPECTED_XML_REMOVED_BBOX_ROWS:-7}"
EXPECTED_CROP_INVALIDATED_FRAMES="${EXPECTED_CROP_INVALIDATED_FRAMES:-1}"
EXPECTED_CROP_INVALIDATED_VIDEOS="${EXPECTED_CROP_INVALIDATED_VIDEOS:-1}"
EXPECTED_CROP_REMOVED_POSITIVE_POINTS="${EXPECTED_CROP_REMOVED_POSITIVE_POINTS:-0}"
EXPECTED_CROP_REMOVED_IGNORE_POINTS="${EXPECTED_CROP_REMOVED_IGNORE_POINTS:-6}"
EXPECTED_CROP_REMOVED_BBOX_ROWS="${EXPECTED_CROP_REMOVED_BBOX_ROWS:-1}"
EXPECTED_EXCLUDED_VIDEOS="${EXPECTED_EXCLUDED_VIDEOS:-20250626_090758_8000,20250626_090652_6340}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

# Optional quick subset. 0 means all files.
MAX_TRAIN_FILES=0

# Optional validation split from INPUT_DIR. 0.0 means use all files for train.
VAL_FRACTION=0.1
MAX_VAL_FILES=0

# Fixed-list mode (S5-15). Set BOTH to reuse saved train/val file lists exactly
# as they are, instead of scanning INPUT_DIR and re-splitting it by
# VAL_FRACTION/SEED. Leaving both empty keeps the historical directory mode
# unchanged. Setting only one is an error: there is deliberately no fallback to
# directory splitting, because a silently different split is exactly what this
# mode exists to prevent.
TRAIN_LIST="${TRAIN_LIST:-}"
VAL_LIST="${VAL_LIST:-}"

# Emits the train_stage5.py arguments selecting the input source, one per line.
# Kept as a function so checks/dummy/check_dummy_fixed_list_mode.sh can extract
# and exercise it without running any training.
input_source_args() {
  local train_list="$1"
  local val_list="$2"
  local input_dir="$3"
  local val_fraction="$4"
  local max_train_files="$5"
  local max_val_files="$6"

  if [[ -n "${train_list}" || -n "${val_list}" ]]; then
    if [[ -z "${train_list}" || -z "${val_list}" ]]; then
      echo "TRAIN_LIST and VAL_LIST must be set together for fixed-list mode (no directory-split fallback)." >&2
      return 1
    fi
    # --train_dir is deliberately absent: train_stage5.py rejects both at once,
    # and passing neither --val_dir nor --val_fraction means the seed-based
    # re-split is never reached. Truncation is disabled so the lists are used
    # whole, in their own order.
    printf '%s\n' --train_list "${train_list}" --val_list "${val_list}" \
      --max_train_files 0 --max_val_files 0
    return 0
  fi

  printf '%s\n' --train_dir "${input_dir}" --val_fraction "${val_fraction}" \
    --max_train_files "${max_train_files}" --max_val_files "${max_val_files}"
  return 0
}

# ------------------------------------------------------------
# output path info
# ------------------------------------------------------------
OUTPUT_ROOT="/mnt/data/3d_projects/stage5_runs/${EX_DATE}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

# Loss knobs.
# CLASS_WEIGHT:
#   ""       : no class weight
#   "auto"   : PointNeXt-style 1 / (class_frequency + epsilon), normalized to mean 1
#   "1,4"    : manual class weights
CLASS_WEIGHT="${CLASS_WEIGHT:-auto}"

# Filesystem-safe tag distinguishing auto/manual/none class weight in the run
# name and startup log (S5-13). "auto" intentionally stays unresolved here
# (no numeric suffix): the resolved values depend on the train split computed
# inside train_stage5.py (seed-dependent), so duplicating that split/count
# logic here would require re-reading every H5 file a second time. The
# resolved auto weight is already recorded in config.json/checkpoint config
# (class_weight_info) and in label_policy_diagnostics.csv.
class_weight_tag() {
  local value="$1"
  local lower
  lower="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  if [[ -z "${lower}" ]]; then
    echo "cw_none"
    return
  fi
  if [[ "${lower}" == "auto" || "${lower}" == "pointnext_auto" ]]; then
    echo "cw_auto"
    return
  fi
  local sanitized
  sanitized="$(printf '%s' "${value}" | tr -d '[:space:]' | tr ',' '_' | tr '.' 'p' | tr -c 'A-Za-z0-9_' '_')"
  echo "cw_manual_${sanitized}"
}
CLASS_WEIGHT_TAG="$(class_weight_tag "${CLASS_WEIGHT}")"

PREFIX="${PREFIX:-w16_s8_bboxrankv7_cvatcropq_glocal_ce_smooth00_${CLASS_WEIGHT_TAG}_lr1e3_ep${EPOCHS}_bs${BATCH_SIZE}_acc${GRADIENT_ACCUMULATION_STEPS}_nopad}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pointnext_s_EX${EX_DATE}_${DATE}_${PREFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}"
ALLOW_EXISTING_OUTPUT_DIR="${ALLOW_EXISTING_OUTPUT_DIR:-0}"

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

# Effective training label for BBox non-contour points (S5-12).
# "bbox_noncontour_ignore" is the historical default (no-op). Set to
# "bbox_noncontour_background" for the Run B label-policy pilot.
LABEL_POLICY="${LABEL_POLICY:-bbox_noncontour_ignore}"

# Training-only input augmentation (S5-15). "none" is the production default.
# "random_z_rotation" rotates each training video about its centroid by one
# angle per epoch; validation, train-sanity evaluation and inference are never
# augmented. AUGMENTATION_SEED is optional: left empty, train_stage5.py derives
# it as SEED + 500000 and records the resolved value in config.json.
AUGMENTATION="${AUGMENTATION:-none}"
AUGMENTATION_ROTATION_DEGREES="${AUGMENTATION_ROTATION_DEGREES:-15.0}"
AUGMENTATION_SEED="${AUGMENTATION_SEED:-}"

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
# CLASS_WEIGHT/CLASS_WEIGHT_TAG are resolved earlier (output path info block)
# because PREFIX/EXPERIMENT_NAME/OUTPUT_DIR depend on the tag.
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
if [[ "${PREFLIGHT_ONLY}" != "0" && "${PREFLIGHT_ONLY}" != "1" ]]; then
  echo "PREFLIGHT_ONLY must be 0 or 1; got ${PREFLIGHT_ONLY}" >&2
  exit 1
fi
if [[ "${ALLOW_EXISTING_OUTPUT_DIR}" != "0" && "${ALLOW_EXISTING_OUTPUT_DIR}" != "1" ]]; then
  echo "ALLOW_EXISTING_OUTPUT_DIR must be 0 or 1; got ${ALLOW_EXISTING_OUTPUT_DIR}" >&2
  exit 1
fi

LIST_MANIFEST=""
if [[ -n "${TRAIN_LIST}" || -n "${VAL_LIST}" ]]; then
  # Fixed-list mode: the teacher preflight below must inspect exactly the files
  # that will be trained on, so INPUT_DIR is never scanned here. Adding or
  # removing files in INPUT_DIR therefore cannot change this run's inputs.
  if [[ -z "${TRAIN_LIST}" || -z "${VAL_LIST}" ]]; then
    echo "TRAIN_LIST and VAL_LIST must be set together for fixed-list mode (no directory-split fallback)." >&2
    exit 1
  fi
  LIST_MANIFEST="$(mktemp "${TMPDIR:-/tmp}/stage5_fixed_list_manifest.XXXXXX")"
  trap 'rm -f "${LIST_MANIFEST}"' EXIT
  "${PYTHON}" "${SCRIPT_DIR}/stage5/utils/file_list_mode.py" \
    --train_list "${TRAIN_LIST}" \
    --val_list "${VAL_LIST}" \
    --h5_pattern "${H5_PATTERN}" \
    --expected_total "${EXPECTED_INPUT_FILES}" \
    --manifest_out "${LIST_MANIFEST}"
  num_inputs="$(grep -c . "${LIST_MANIFEST}" | tr -d ' ')"
else
  if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Input directory not found: ${INPUT_DIR}" >&2
    echo "Run Stage2to4/pseudo3d/pipelines/build_stage4_bbox_ranked_v7_crop_quality.sh first, or set INPUT_DIR." >&2
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
    echo "Stage 5 requires the complete ${DATE} teacher v7 dataset:" >&2
    echo "  expected files=${EXPECTED_INPUT_FILES}" >&2
    echo "  matched files=${num_inputs}" >&2
    echo "  INPUT_DIR=${INPUT_DIR}" >&2
    echo "  H5_PATTERN=${H5_PATTERN}" >&2
    exit 1
  fi
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
  "${EXPECTED_CROP_INVALIDATED_FRAMES}" \
  "${EXPECTED_CROP_INVALIDATED_VIDEOS}" \
  "${EXPECTED_CROP_REMOVED_POSITIVE_POINTS}" \
  "${EXPECTED_CROP_REMOVED_IGNORE_POINTS}" \
  "${EXPECTED_CROP_REMOVED_BBOX_ROWS}" \
  "${EXPECTED_EXCLUDED_VIDEOS}" \
  "${LIST_MANIFEST}" <<'PY'
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
expected_crop_invalidated_frames = int(sys.argv[11])
expected_crop_invalidated_videos = int(sys.argv[12])
expected_crop_removed_positive = int(sys.argv[13])
expected_crop_removed_ignore = int(sys.argv[14])
expected_crop_removed_bbox_rows = int(sys.argv[15])
expected_excluded_videos = {
    value.strip() for value in sys.argv[16].split(",") if value.strip()
}
list_manifest = sys.argv[17] if len(sys.argv) > 17 else ""
if list_manifest:
    # Fixed-list mode: inspect the listed files themselves, so the preflight and
    # the training run can never look at different file sets.
    paths = [
        Path(line.strip())
        for line in Path(list_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
else:
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
crop_invalidated_videos = 0
crop_invalidated_frames = 0
crop_removed_positive = 0
crop_removed_ignore = 0
crop_removed_bbox_rows = 0
crop_manifest_hashes = set()
crop_manifest_fingerprints = set()
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
            "crop_quality_invalidation/frame_order",
            "crop_quality_invalidation/after_positive_points",
            "crop_quality_invalidation/after_ignore_points",
            "crop_quality_invalidation/before_positive_points",
            "crop_quality_invalidation/before_ignore_points",
            "crop_quality_invalidation/removed_frame_annotation_rows",
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
            "teacher_version": "bboxrank_v6_cvat_authoritative_xml_invalidation_v1",
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

        crop = handle["crop_quality_invalidation"]
        crop_count = len(crop["frame_order"])
        crop_invalidated_frames += crop_count
        crop_invalidated_videos += int(crop_count > 0)
        crop_manifest_hash = text(crop.attrs.get("manifest_sha256", ""))
        crop_manifest_fingerprint = text(
            crop.attrs.get("manifest_fingerprint", "")
        )
        if not crop_manifest_hash or not crop_manifest_fingerprint:
            raise SystemExit(f"Missing crop invalidation manifest identity in {path}")
        crop_manifest_hashes.add(crop_manifest_hash)
        crop_manifest_fingerprints.add(crop_manifest_fingerprint)
        expected_crop_attrs = {
            "teacher_version": expected_teacher,
            "source_teacher_version": (
                "bboxrank_v6_cvat_authoritative_xml_invalidation_v1"
            ),
            "policy": "fully-outside local-crop frame -> all background",
            "frame_authority": "crop_quality_invalidation_manifest",
        }
        for name, expected_value in expected_crop_attrs.items():
            actual = text(crop.attrs.get(name, ""))
            if actual != expected_value:
                raise SystemExit(
                    f"Unexpected crop invalidation {name} in {path}: {actual!r}"
                )
        if crop_count:
            after_positive = crop["after_positive_points"][:].astype(np.int64)
            after_ignore = crop["after_ignore_points"][:].astype(np.int64)
            if np.any(after_positive != 0) or np.any(after_ignore != 0):
                raise SystemExit(
                    f"Crop-invalidated frame retains non-background labels in {path}"
                )
            for order in crop["frame_order"][:].astype(np.int64):
                selected = frame_order == order
                if not np.any(selected) or not np.all(point_label[selected] == 0):
                    raise SystemExit(
                        f"Crop-invalidated frame_order={order} is not all background "
                        f"in {path}"
                    )
                if not np.all(valid_mask[selected]):
                    raise SystemExit(
                        f"Crop-invalidated frame_order={order} is not fully valid "
                        f"in {path}"
                    )
            crop_removed_positive += int(
                crop["before_positive_points"][:].astype(np.int64).sum()
            )
            crop_removed_ignore += int(
                crop["before_ignore_points"][:].astype(np.int64).sum()
            )
            crop_removed_bbox_rows += int(
                crop["removed_frame_annotation_rows"][:].astype(np.int64).sum()
            )
        if not video_name:
            raise SystemExit(f"Missing video_name in {path}")
        if video_name in videos:
            raise SystemExit(f"Duplicate video_name: {video_name}")
        videos.add(video_name)
unexpected_excluded = videos & expected_excluded_videos
if unexpected_excluded:
    raise SystemExit(
        f"Excluded crop-failure videos are present: {sorted(unexpected_excluded)}"
    )
if len(manifest_hashes) != 1 or len(manifest_fingerprints) != 1:
    raise SystemExit(
        "v7 files do not share one XML invalidation manifest identity: "
        f"sha={len(manifest_hashes)}, fingerprint={len(manifest_fingerprints)}"
    )
if len(crop_manifest_hashes) != 1 or len(crop_manifest_fingerprints) != 1:
    raise SystemExit(
        "v7 files do not share one crop invalidation manifest identity: "
        f"sha={len(crop_manifest_hashes)}, "
        f"fingerprint={len(crop_manifest_fingerprints)}"
    )
expected_counts = {
    "cvat_videos": (cvat_videos, expected_cvat_videos),
    "cvat_frames": (cvat_frames, expected_cvat_frames),
    "invalidated_frames": (invalidated_frames, expected_invalidated_frames),
    "invalidated_videos": (invalidated_videos, expected_invalidated_videos),
    "removed_positive": (removed_positive, expected_removed_positive),
    "removed_bbox_rows": (removed_bbox_rows, expected_removed_bbox_rows),
    "crop_invalidated_frames": (
        crop_invalidated_frames,
        expected_crop_invalidated_frames,
    ),
    "crop_invalidated_videos": (
        crop_invalidated_videos,
        expected_crop_invalidated_videos,
    ),
    "crop_removed_positive": (
        crop_removed_positive,
        expected_crop_removed_positive,
    ),
    "crop_removed_ignore": (
        crop_removed_ignore,
        expected_crop_removed_ignore,
    ),
    "crop_removed_bbox_rows": (
        crop_removed_bbox_rows,
        expected_crop_removed_bbox_rows,
    ),
}
for name, (actual, expected_value) in expected_counts.items():
    if actual != expected_value:
        raise SystemExit(f"v7 {name} mismatch: {actual} != {expected_value}")
print(
    "Stage 5 teacher v7 input preflight passed: "
    f"files={len(paths)}, videos={len(videos)}, "
    f"cvat_videos={cvat_videos}, cvat_frames={cvat_frames}, "
    f"invalidated_videos={invalidated_videos}, "
    f"invalidated_frames={invalidated_frames}, "
    f"removed_positive={removed_positive}, removed_bbox_rows={removed_bbox_rows}, "
    f"crop_invalidated_videos={crop_invalidated_videos}, "
    f"crop_invalidated_frames={crop_invalidated_frames}, "
    f"crop_removed_ignore={crop_removed_ignore}, "
    f"crop_removed_bbox_rows={crop_removed_bbox_rows}"
)
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Stage 5 teacher v7 input preflight-only run complete; training was not started."
  if [[ -n "${LIST_MANIFEST}" ]]; then
    echo "  input source  : fixed lists (INPUT_DIR not scanned)"
    echo "  train list    : ${TRAIN_LIST}"
    echo "  val list      : ${VAL_LIST}"
  else
    echo "  input source  : directory scan"
    echo "  input dir     : ${INPUT_DIR}"
  fi
  echo "  h5 pattern    : ${H5_PATTERN}"
  echo "  matched files : ${num_inputs}"
  echo "  teacher       : ${STAGE4_TEACHER}"
  exit 0
fi

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ "${ALLOW_EXISTING_OUTPUT_DIR}" != "1" ]]; then
    echo "Output directory already exists and is not empty: ${OUTPUT_DIR}" >&2
    echo "Set ALLOW_EXISTING_OUTPUT_DIR=1 to explicitly resume/overwrite it, or pick a different EX_DATE/OUTPUT_DIR/EXPERIMENT_NAME." >&2
    exit 1
  fi
fi
mkdir -p "${OUTPUT_DIR}"

echo "Stage5 padding-free training"
if [[ -n "${LIST_MANIFEST}" ]]; then
  echo "  input source   : fixed lists (INPUT_DIR not scanned, no fraction re-split)"
  echo "  train list     : ${TRAIN_LIST}"
  echo "  val list       : ${VAL_LIST}"
else
  echo "  input source   : directory scan + val_fraction=${VAL_FRACTION}"
  echo "  input dir      : ${INPUT_DIR}"
fi
echo "  h5 pattern     : ${H5_PATTERN}"
echo "  matched files  : ${num_inputs}"
echo "  teacher        : ${STAGE4_TEACHER}"
echo "  augmentation   : ${AUGMENTATION} (train only; eval/inference never augmented)"
echo "  label policy   : CVAT-reviewed=authoritative; unreviewed=inherited; XML/crop-invalidated=background"
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
echo "  class weight   : ${CLASS_WEIGHT:-none} (tag=${CLASS_WEIGHT_TAG})"
echo "  label smoothing: ${LABEL_SMOOTHING}"
echo "  init checkpoint: ${INIT_CHECKPOINT:-none}"
if [[ "${CLASS_WEIGHT}" == "auto" || "${CLASS_WEIGHT}" == "pointnext_auto" ]]; then
  echo "  auto cls weight: epsilon=${AUTO_CLASS_WEIGHT_EPSILON}, normalize=${NORMALIZE_AUTO_CLASS_WEIGHT}"
fi
if [[ "${MODEL_NAME}" == "pointnext_s" ]]; then
  echo "  pointnext      : radius=${POINTNEXT_RADIUS}, nsample=${POINTNEXT_NSAMPLE}, sa_layers=${POINTNEXT_SA_LAYERS}, sa_use_res=${POINTNEXT_SA_USE_RES}, norm=${POINTNEXT_NORM}, norm_groups=${POINTNEXT_NORM_GROUPS}"
  echo "  label_policy   : ${LABEL_POLICY}"
fi

input_source_output="$(input_source_args \
  "${TRAIN_LIST}" "${VAL_LIST}" "${INPUT_DIR}" \
  "${VAL_FRACTION}" "${MAX_TRAIN_FILES}" "${MAX_VAL_FILES}")" || exit 1
mapfile -t INPUT_SOURCE_ARGS <<< "${input_source_output}"

cmd=(
  "${PYTHON}" "${SCRIPT_DIR}/train_stage5.py"
  "${INPUT_SOURCE_ARGS[@]}"
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
  --label_policy "${LABEL_POLICY}"
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
  --augmentation "${AUGMENTATION}"
  --augmentation_rotation_degrees "${AUGMENTATION_ROTATION_DEGREES}"
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

if [[ -n "${AUGMENTATION_SEED}" ]]; then
  cmd+=(--augmentation_seed "${AUGMENTATION_SEED}")
fi

"${cmd[@]}"

echo "Done."
echo "  best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "  last checkpoint: ${OUTPUT_DIR}/last.pt"
