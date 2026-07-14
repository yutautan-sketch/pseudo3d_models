# Stage5 File Layout

## Core code

- `train_stage5.py`: training CLI.
- `infer_stage5.py`: inference CLI.
- `stage5/`: dataset, model wrappers, losses, metrics, and utilities.
- `train_stage5.sh`: editable training launcher with external data paths.
- `infer_stage5.sh`: editable inference launcher with external data paths.

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
