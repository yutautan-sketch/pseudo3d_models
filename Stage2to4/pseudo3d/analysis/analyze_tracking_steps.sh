VIDEO="20250627_104104_9320"

python pseudo3d/analysis/analyze_tracking_steps.py \
  --input_h5 /mnt/data/3d_projects/DualTrack_output/pseudo3d_outputs/stage2_v1_scale10.0_test/${VIDEO}_ts448_oym96_corr.h5 \
  --output_csv /mnt/data/3d_projects/DualTrack_output/pseudo3d_outputs/stage2_v1_scale10.0_test/step_analysis_${VIDEO}.csv \
  --axis_method attrs \
  --geometry_key corr \
  --include_frame_normal