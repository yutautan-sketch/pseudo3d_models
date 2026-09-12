# Stage5 File Layout

## Current implementation handoff

- `stage5_revision_management_record.md`: authoritative Stage 5 overview,
  chronological revision record, accepted decisions, current status, and next
  implementation order. Start here when deciding what work comes next.
- `stage5_overlap_aggregation_handoff_prompt.md`: current teacher v6 state,
  overlap aggregation diagnostics, BatchNorm parity steps, privacy rules, and
  completion criteria for the next implementation chat.
- `stage5_overlap_aggregation_implementation_policy.md`: implementation-chat
  rationale behind the handoff prompt's overlap aggregation checker design
  (module reuse, accumulator math, mean baseline parity method, output
  layout, and the F1-F3 design decisions). The handoff prompt is authoritative
  if the two disagree.
- `stage5_pointnext_s_training_evaluation_report.md`:
  authoritative investigation history and current findings.
- `TRAINING_IMPROVEMENT_PLAN.md`: earlier planning history; use the evaluation
  report for detailed evidence and `stage5_revision_management_record.md` for
  the current priority order.

## Core code

Paths below are relative to `Stage5/` (this file lives in `docs/stage5/`).

- `train_stage5.py`: training CLI.
- `infer_stage5.py`: inference CLI.
- `evaluate_stage5.py`: annotated H5 evaluation CLI with window and aggregated metrics.
- `prepare_stage5_evaluation_data.py`: selects evaluation H5 files and collects reference PLY files.
- `summarize_stage5_evaluations.py`: combines metrics from multiple checkpoints.
- `export_anonymized_stage5_metrics.py`: creates shareable split-labeled metrics and a separate private ID map.
- `stage5/`: dataset, model wrappers, losses, metrics, and utilities.
- `train_stage5.sh`: editable training launcher with external data paths.
- `infer_stage5.sh`: editable inference launcher with external data paths.
- `evaluate_stage5.sh`: fixed train-sanity/full-validation checkpoint evaluation pipeline.
- `export_anonymized_stage5_metrics.sh`: standalone anonymized-metrics exporter for an existing evaluation.

## Check and debug scripts

`check_*.py` and `check_*.sh` are smoke tests or visualization helpers. They are
grouped under `checks/` so they stay separate from the regular training and
inference launchers.

Main groups:

- `checks/dummy/check_dummy_*`: synthetic H5/model checks.
- `checks/visualization/check_window_batch_ply_export.*`: frame-window DataLoader PLY export.
- `checks/s3dis/check_s3dis_*`: official PointNeXt/S3DIS checkpoint and data checks.
- `checks/transfer/check_s3dis_to_stage5_pointnext_s_transfer.*`: partial S3DIS-to-Stage5 weight transfer.
- `checks/real_h5/check_real_h5_pointnext_s_transfer_forward.*`: real Stage5 H5 forward/PLY check.
- `checks/real_h5/check_stage5_batch_integrity.*`: Dataset/window/point_indices audit against source H5 (S5-04).
- `checks/real_h5/check_stage5_padding_parity.*`: zero-padding sensitivity for logits/gradient/BatchNorm (S5-05).
- `checks/real_h5/check_stage5_overlap_aggregation.*`: overlap-window probability/aggregation comparison (S5-08, 実装事項A).
- `checks/real_h5/check_stage5_batchnorm_mode_parity.*`: eval()/train() BatchNorm probability parity (S5-09, 実装事項B).
- `checks/real_h5/check_stage5_batchnorm_recalibration.*`: diagnostic BatchNorm running-statistics recalibration from the training split, compared against the original checkpoint (S5-10). Saves a diagnostic checkpoint under `stage5_debug/`; never overwrites the original and is never shared.
- `checks/dummy/check_dummy_pointnext_s_groupnorm.*`: BatchNorm-default backward-compatibility and GroupNorm structural/train-eval-parity checks for the PointNeXt-S wrapper's `pointnext_norm` option (S5-11). `--self_test` is CPU-only; the full run needs CUDA and dummy H5 data only.
- `checks/transfer/check_stage5_batchnorm_to_groupnorm_transfer.*`: transfers the existing Stage5 BatchNorm S3DIS partial-init checkpoint's trainable parameters (conv/linear/head/norm affine) into a freshly constructed GroupNorm model, excluding only BatchNorm running buffers (S5-11). Saves `stage5_pointnext_s_s3dis_partial_init_groupnorm.pt` alongside the existing BatchNorm partial-init checkpoint.
- `checks/inspection/inspect_pointnext_checkpoint.py`: checkpoint structure inspection.

## Generated local work directories

Generated outputs from check scripts are stored under:

- `work_dirs/_dummy_inference_check`
- `work_dirs/_dummy_training_check`
- `work_dirs/_dummy_pointnext_s_forward_check`
- `work_dirs/_dummy_pointnext_s_training_check`
- `work_dirs/_dummy_window_check`
- `work_dirs/_s3dis_pointnext_s_single_room_inference`
- `work_dirs/_s3dis_to_stage5_pointnext_s_transfer`

These are ignored by git. The current Stage5 PointNeXt-S partial initialization
checkpoint is:

`work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init.pt`

## External repositories and weights

- `external/PointNeXt/`: local clone of the official PointNeXt/OpenPoints code
  and downloaded weights/datasets. This directory is ignored by git.

## Analysis assets

- `assets/`: copied metrics/config/history snapshots used for discussion and
  comparison. Keep only selected lightweight analysis artifacts here.
