# ------------------------------------------------------------
# Run from DualTrack repository root
# ------------------------------------------------------------
cd /mnt/data/3d_projects/models/Stage2to4

DATE="260711"

python scripts/utils/collect_annotated_pseudo3d_h5.py \
  --source_root "/mnt/data/3d_projects/pseudo3d_dataset/pseudo3d_visualizations/${DATE}" \
  --output_dir "/mnt/data/3d_projects/pseudo3d_dataset/${DATE}" \
  --mode foreground \
  --manifest_csv "/mnt/data/3d_projects/pseudo3d_dataset/${DATE}/manifest.csv"