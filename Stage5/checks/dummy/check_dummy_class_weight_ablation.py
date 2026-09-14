from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_class_weight_ablation import (
    check_file_lists,
    check_history,
    check_resolved_weight,
    compare_config,
    compare_eval_targets,
)

# ----------------------------------------------------------------------------
# S5-13 Step F3: synthetic pass/fail coverage for
# checks/real_h5/check_stage5_class_weight_ablation.py's comparison logic,
# using hand-built config.json/train_files.txt/val_files.txt/history.json/
# h5_metrics.csv fixtures. No GPU/H5/CUDA/real run data required.
# ----------------------------------------------------------------------------


def base_config(*, output_dir: str, class_weight: list[float], mode: str, requested: str) -> dict[str, Any]:
    return {
        "output_dir": output_dir,
        "train_list": None,
        "train_dir": "/mnt/data/3d_projects/pseudo3d_dataset/collected",
        "val_list": None,
        "val_dir": None,
        "max_train_files": 0,
        "max_val_files": 0,
        "val_fraction": 0.1,
        "checkpoint": "/mnt/data/3d_projects/models/Stage5/work_dirs/_s3dis_to_stage5_pointnext_s_transfer/stage5_pointnext_s_s3dis_partial_init.pt",
        "pointnext_norm": "groupnorm",
        "pointnext_norm_groups": 8,
        "label_policy": "bbox_noncontour_ignore",
        "seed": 42,
        "window_size_frames": 16,
        "window_stride_frames": 8,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.0,
        "epochs": 5,
        "features": "intensity,confidence",
        "class_weight": class_weight,
        "class_weight_info": {
            "mode": mode,
            "requested": requested,
            "resolved": class_weight,
            "counts": None,
            "epsilon": None,
            "normalize": None,
        },
    }


def write_run_dir(
    root: Path,
    name: str,
    *,
    class_weight: list[float],
    mode: str,
    requested: str,
    train_files: list[str],
    val_files: list[str],
    epochs: int = 5,
    nonfinite_epoch: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = base_config(output_dir=str(run_dir), class_weight=class_weight, mode=mode, requested=requested)
    if config_overrides:
        config.update(config_overrides)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f)

    (run_dir / "train_files.txt").write_text("".join(f"{p}\n" for p in train_files), encoding="utf-8")
    (run_dir / "val_files.txt").write_text("".join(f"{p}\n" for p in val_files), encoding="utf-8")

    history = []
    for epoch in range(1, epochs + 1):
        loss_value = 0.5 if epoch != nonfinite_epoch else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train": {"loss": loss_value, "f1": 0.1, "iou_femur": 0.05},
                "val": {"loss": 0.6, "f1": 0.08, "iou_femur": 0.04},
            }
        )
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f)

    (run_dir / "last.pt").write_bytes(b"dummy-checkpoint-bytes")
    return run_dir


def write_eval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_no_config_diff(config_a: dict[str, Any], config_b: dict[str, Any]) -> None:
    diffs = compare_config(config_a, config_b)
    if diffs:
        raise AssertionError(diffs)


def expect_pass(label: str, fn) -> None:
    fn()
    print(f"ok (expected pass): {label}")


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_files = [f"/data/train_{i:03d}.h5" for i in range(3)]
        val_files = [f"/data/val_{i:03d}.h5" for i in range(2)]

        run_a = write_run_dir(
            root,
            "run_a",
            class_weight=[0.05963856, 1.94036150],
            mode="manual",
            requested="0.05963856,1.94036150",
            train_files=train_files,
            val_files=val_files,
        )
        run_b_ok = write_run_dir(
            root,
            "run_b_ok",
            class_weight=[0.5, 1.5],
            mode="manual",
            requested="0.5,1.5",
            train_files=train_files,
            val_files=val_files,
        )

        expect_pass(
            "matching config aside from class_weight/output_dir",
            lambda: assert_no_config_diff(
                json.loads((run_a / "config.json").read_text()),
                json.loads((run_b_ok / "config.json").read_text()),
            ),
        )

        run_b_bad_norm = write_run_dir(
            root,
            "run_b_bad_norm",
            class_weight=[0.5, 1.5],
            mode="manual",
            requested="0.5,1.5",
            train_files=train_files,
            val_files=val_files,
            config_overrides={"pointnext_norm": "batchnorm"},
        )
        expect_fail(
            "unexpected config diff (pointnext_norm changed) is rejected",
            lambda: assert_no_config_diff(
                json.loads((run_a / "config.json").read_text()),
                json.loads((run_b_bad_norm / "config.json").read_text()),
            ),
        )

        expect_pass(
            "identical train_files.txt/val_files.txt",
            lambda: check_file_lists(run_a, run_b_ok, name="train_files.txt"),
        )

        run_b_diff_files = write_run_dir(
            root,
            "run_b_diff_files",
            class_weight=[0.5, 1.5],
            mode="manual",
            requested="0.5,1.5",
            train_files=[*train_files[:-1], "/data/train_999.h5"],
            val_files=val_files,
        )
        expect_fail(
            "mismatched train_files.txt content is rejected",
            lambda: check_file_lists(run_a, run_b_diff_files, name="train_files.txt"),
        )

        expect_pass(
            "resolved weight matches expected value",
            lambda: check_resolved_weight(
                json.loads((run_b_ok / "config.json").read_text()),
                expected=[0.5, 1.5],
                label="W-B",
            ),
        )
        expect_fail(
            "resolved weight not matching expected value is rejected",
            lambda: check_resolved_weight(
                json.loads((run_b_ok / "config.json").read_text()),
                expected=[0.25, 1.75],
                label="W-B",
            ),
        )

        expect_pass(
            "5-epoch finite history passes",
            lambda: check_history(run_a, expected_epochs=5, label="W-A"),
        )

        run_nonfinite = write_run_dir(
            root,
            "run_nonfinite",
            class_weight=[0.5, 1.5],
            mode="manual",
            requested="0.5,1.5",
            train_files=train_files,
            val_files=val_files,
            nonfinite_epoch=3,
        )
        expect_fail(
            "non-finite train loss at epoch 3 is rejected",
            lambda: check_history(run_nonfinite, expected_epochs=5, label="run_nonfinite"),
        )

        run_short = write_run_dir(
            root,
            "run_short",
            class_weight=[0.5, 1.5],
            mode="manual",
            requested="0.5,1.5",
            train_files=train_files,
            val_files=val_files,
            epochs=3,
        )
        expect_fail(
            "history.json with fewer than expected epochs is rejected",
            lambda: check_history(run_short, expected_epochs=5, label="run_short"),
        )

        eval_csv_a = root / "eval_a" / "h5_metrics.csv"
        eval_csv_b_ok = root / "eval_b_ok" / "h5_metrics.csv"
        eval_csv_b_bad = root / "eval_b_bad" / "h5_metrics.csv"
        rows_a = [
            {
                "split": "validation",
                "video_name": "video_alpha",
                "total_point_count": "1000",
                "valid_point_count": "900",
                "valid_positive_count": "100",
                "valid_background_count": "800",
            },
            {
                "split": "validation",
                "video_name": "video_beta",
                "total_point_count": "2000",
                "valid_point_count": "1800",
                "valid_positive_count": "200",
                "valid_background_count": "1600",
            },
        ]
        write_eval_csv(eval_csv_a, rows_a)
        write_eval_csv(eval_csv_b_ok, rows_a)
        rows_b_bad = [dict(rows_a[0]), dict(rows_a[1])]
        rows_b_bad[1]["valid_positive_count"] = "199"
        write_eval_csv(eval_csv_b_bad, rows_b_bad)

        expect_pass(
            "identical evaluation target point sets",
            lambda: compare_eval_targets(eval_csv_a, eval_csv_b_ok),
        )
        expect_fail(
            "differing evaluation target point counts are rejected",
            lambda: compare_eval_targets(eval_csv_a, eval_csv_b_bad),
        )

    print("check_dummy_class_weight_ablation: all synthetic pass/fail cases behaved as expected.")


if __name__ == "__main__":
    main()
