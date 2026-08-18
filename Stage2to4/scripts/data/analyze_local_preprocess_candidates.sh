#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/analyze_local_preprocess_candidates.py" \
  --video_dir /mnt/data/Data_hbl/example_videos/for_anlyze \
  --annotation_root /mnt/data/Data_hbl/predirectory_for_coco_annotation_hbl/251117_検証用/anno \
  --output_dir preprocess_reports_bbox \
  --stride 1 \
  --max_frames 128 \
  --target_short_sides 448 480 \
  --crop_offsets_y -96 -64 \
  --crop_offsets_x 0 \
  --save_bbox_montage
