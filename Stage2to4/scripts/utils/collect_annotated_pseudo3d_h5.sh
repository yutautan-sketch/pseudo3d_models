# ------------------------------------------------------------
# Run from DualTrack repository root
# ------------------------------------------------------------
cd /mnt/data/3d_projects/models/Stage2to4

python scripts/utils/collect_annotated_pseudo3d_h5.py \
  --source_root /mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_visualizations/stage2_v1_scale10.0 \
  --output_dir /mnt/data/3d_projects/pseudo3d_dataset/260623 \
  --mode grid \
  --manifest_csv /mnt/data/3d_projects/pseudo3d_dataset/260623/manifest.csv