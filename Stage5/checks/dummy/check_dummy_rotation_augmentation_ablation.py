from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "checks" / "real_h5"))

from check_stage5_rotation_augmentation_ablation import (  # noqa: E402
    INTENDED_DIFFERENCE_KEYS,
    STATUS_FAIL,
    STATUS_JUDGE,
    STATUS_PASS,
    STATUS_UNKNOWN,
    build_shareable_summary,
    check_augmentation_settings,
    check_manifest_against_config,
    check_checkpoints,
    check_file_lists,
    compare_configs,
    evaluate_selection_criteria,
    per_video_diff_comparison,
    privacy_self_check,
    read_h5_metrics_csv,
    split_pooled_comparison,
)

# ----------------------------------------------------------------------------
# S5-15 P2 Step 4: synthetic coverage for the R0/R1 comparison checker and for
# the launcher's refusal to train without explicit confirmation. Pure stdlib
# fixtures: no torch, no CUDA, no real run directories, and nothing here starts
# training.
# ----------------------------------------------------------------------------

METRIC_FIELDS = [
    "checkpoint",
    "split",
    "video_name",
    "h5_path",
    "precision",
    "recall",
    "f1",
    "iou_femur",
    "false_positive_rate",
    "true_positive_count",
    "false_positive_count",
    "true_negative_count",
    "false_negative_count",
    "ignore_point_count",
    "ignore_predicted_positive_count",
]


def base_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "train_list": "/mnt/data/lists/train_files.txt",
        "val_list": "/mnt/data/lists/val_files.txt",
        "train_dir": None,
        "val_fraction": 0.0,
        "checkpoint": "/mnt/data/init/stage5_pointnext_s_s3dis_partial_init_groupnorm.pt",
        "pointnext_norm": "groupnorm",
        "pointnext_norm_groups": 8,
        "seed": 42,
        "epochs": 5,
        "label_policy": "bbox_noncontour_ignore",
        "class_weight": [0.05963856, 1.9403615],
        "augmentation": "none",
        "augmentation_rotation_degrees": 15.0,
        "augmentation_seed": None,
        "augmentation_info": {
            "mode": "none",
            "max_abs_degrees": 15.0,
            "base_seed": 500042,
            "seed_source": "train_seed_plus_500000",
            "angle_derivation": "sha256...",
        },
        "output_dir": "/mnt/data/runs/r0",
    }
    config.update(overrides)
    return config


def r1_config(**overrides: Any) -> dict[str, Any]:
    config = base_config(
        augmentation="random_z_rotation",
        output_dir="/mnt/data/runs/r1",
        augmentation_info={
            "mode": "random_z_rotation",
            "max_abs_degrees": 15.0,
            "base_seed": 500042,
            "seed_source": "train_seed_plus_500000",
            "angle_derivation": "sha256...",
        },
    )
    config.update(overrides)
    return config


def video_name(index: int) -> str:
    return f"2026071{index % 10}_12{index:04d}_{index}"


def write_run_dir(
    root: Path,
    *,
    config: dict[str, Any],
    train_entries: list[str] | None = None,
    val_entries: list[str] | None = None,
    epochs: int = 5,
    checkpoints: tuple[str, ...] = ("best.pt", "last.pt", *[f"checkpoint_epoch_{i}.pt" for i in range(1, 6)]),
    last_payload: bytes = b"r0 weights",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_entries = train_entries or [f"/mnt/data/collected/{video_name(i)}.h5" for i in range(6)]
    val_entries = val_entries or [f"/mnt/data/collected/{video_name(1000 + i)}.h5" for i in range(2)]
    (root / "train_files.txt").write_text("\n".join(train_entries) + "\n", encoding="utf-8")
    (root / "val_files.txt").write_text("\n".join(val_entries) + "\n", encoding="utf-8")
    history = [
        {"epoch": e, "train": {"loss": 0.3 - 0.01 * e, "f1": 0.01 * e}, "val": {"loss": 0.2, "f1": 0.01 * e}}
        for e in range(1, epochs + 1)
    ]
    (root / "history.json").write_text(json.dumps(history), encoding="utf-8")
    for name in checkpoints:
        (root / name).write_bytes(last_payload if name == "last.pt" else b"other")
    return root


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
    return path


def metrics_row(
    *,
    split: str,
    index: int,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    ignore_points: int = 1000,
    ignore_positive: int = 10,
) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "checkpoint": "last.pt",
        "split": split,
        "video_name": video_name(index),
        "h5_path": f"/mnt/data/collected/{video_name(index)}.h5",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_femur": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "ignore_point_count": ignore_points,
        "ignore_predicted_positive_count": ignore_positive,
    }


def status_of(results: list[dict[str, Any]], check: str) -> str:
    matches = [row["status"] for row in results if row["check"] == check]
    assert len(matches) == 1, f"expected exactly one result for {check!r}, found {len(matches)}"
    return matches[0]


def test_identical_configs_except_augmentation_pass() -> None:
    results: list[dict[str, Any]] = []
    diffs = compare_configs(base_config(), r1_config(), results)
    assert diffs == [], diffs
    assert status_of(results, "config.parity_outside_allowlist") == STATUS_PASS
    assert set(INTENDED_DIFFERENCE_KEYS) == {"output_dir", "augmentation", "augmentation_info"}
    print("  ok: only output_dir/augmentation/augmentation_info may differ, and that pair passes")


def test_any_other_config_difference_is_reported() -> None:
    """Report 8.7.1/answer 3: intended differences are an allowlist, so a real
    setting difference cannot hide behind a broad exclusion."""
    for key, value in (
        ("seed", 7),
        ("class_weight", [0.5, 1.5]),
        ("pointnext_norm", "batchnorm"),
        ("train_list", "/mnt/data/lists/other_train.txt"),
        ("epochs", 10),
        ("augmentation_rotation_degrees", 30.0),
    ):
        results: list[dict[str, Any]] = []
        diffs = compare_configs(base_config(), r1_config(**{key: value}), results)
        assert len(diffs) == 1 and diffs[0].startswith(f"{key}:"), diffs
        assert status_of(results, "config.parity_outside_allowlist") == STATUS_FAIL
    print("  ok: seed, class weight, norm, train_list, epochs and rotation range differences all caught")


def test_augmentation_modes_must_be_none_and_random_z_rotation() -> None:
    results: list[dict[str, Any]] = []
    check_augmentation_settings(base_config(), r1_config(), expected_degrees=15.0, results=results)
    assert status_of(results, "augmentation.r0_is_none") == STATUS_PASS
    assert status_of(results, "augmentation.r1_is_random_z_rotation") == STATUS_PASS
    assert status_of(results, "augmentation.r1_degrees") == STATUS_PASS
    assert status_of(results, "augmentation.r1_seed_recorded") == STATUS_PASS

    swapped: list[dict[str, Any]] = []
    check_augmentation_settings(r1_config(), base_config(), expected_degrees=15.0, results=swapped)
    assert status_of(swapped, "augmentation.r0_is_none") == STATUS_FAIL
    assert status_of(swapped, "augmentation.r1_is_random_z_rotation") == STATUS_FAIL

    wrong_degrees: list[dict[str, Any]] = []
    info = dict(r1_config()["augmentation_info"])
    info["max_abs_degrees"] = 30.0
    check_augmentation_settings(
        base_config(), r1_config(augmentation_info=info), expected_degrees=15.0, results=wrong_degrees
    )
    assert status_of(wrong_degrees, "augmentation.r1_degrees") == STATUS_FAIL
    print("  ok: swapped arms and a wrong rotation range are both reported")


def test_file_list_representation_difference_is_judge_not_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        entries = [f"/mnt/data/collected/{video_name(i)}.h5" for i in range(6)]
        other_prefix = [f"/other/mount/collected/{video_name(i)}.h5" for i in range(6)]
        r0 = write_run_dir(root / "r0", config=base_config(), train_entries=entries)
        r1 = write_run_dir(root / "r1", config=r1_config(), train_entries=other_prefix)

        results: list[dict[str, Any]] = []
        check_file_lists(r0, r1, reference_dir=None, results=results)
        assert status_of(results, "file_list.train_files.txt.arms_match") == STATUS_JUDGE
        assert status_of(results, "file_list.val_files.txt.arms_match") == STATUS_PASS

        different = write_run_dir(
            root / "r1b",
            config=r1_config(),
            train_entries=entries[:5] + [f"/mnt/data/collected/{video_name(99)}.h5"],
        )
        content_results: list[dict[str, Any]] = []
        check_file_lists(r0, different, reference_dir=None, results=content_results)
        assert status_of(content_results, "file_list.train_files.txt.arms_match") == STATUS_FAIL
    print("  ok: a prefix-only difference is JUDGE while a different video set is FAIL")


def test_reference_list_comparison_is_reported_separately() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        entries = [f"/mnt/data/collected/{video_name(i)}.h5" for i in range(6)]
        r0 = write_run_dir(root / "r0", config=base_config(), train_entries=entries)
        r1 = write_run_dir(root / "r1", config=r1_config(), train_entries=entries)
        reference = write_run_dir(root / "wa", config=base_config(), train_entries=entries)

        results: list[dict[str, Any]] = []
        check_file_lists(r0, r1, reference_dir=reference, results=results)
        assert status_of(results, "file_list.train_files.txt.vs_reference") == STATUS_PASS

        missing_reference: list[dict[str, Any]] = []
        check_file_lists(r0, r1, reference_dir=root / "absent", results=missing_reference)
        assert status_of(missing_reference, "file_list.train_files.txt.vs_reference") == STATUS_UNKNOWN
    print("  ok: the reference comparison passes on matching lists and is UNKNOWN when absent")


def test_trained_checkpoint_equality_is_not_required_but_identity_is_flagged() -> None:
    """Answer 3 explicitly does not require the trained checkpoints to match.
    Identical weights would instead mean augmentation never reached training."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r0 = write_run_dir(root / "r0", config=base_config(), last_payload=b"r0 weights")
        r1 = write_run_dir(root / "r1", config=r1_config(), last_payload=b"r1 weights")
        results: list[dict[str, Any]] = []
        check_checkpoints(r0, r1, expected_epochs=5, results=results)
        assert status_of(results, "trained_checkpoints.differ") == STATUS_PASS
        assert status_of(results, "r0.checkpoint.per_epoch") == STATUS_PASS

        same = write_run_dir(root / "r1same", config=r1_config(), last_payload=b"r0 weights")
        identical: list[dict[str, Any]] = []
        check_checkpoints(r0, same, expected_epochs=5, results=identical)
        assert status_of(identical, "trained_checkpoints.differ") == STATUS_JUDGE

        missing_epochs = write_run_dir(
            root / "r1few", config=r1_config(), checkpoints=("best.pt", "last.pt")
        )
        sparse: list[dict[str, Any]] = []
        check_checkpoints(r0, missing_epochs, expected_epochs=5, results=sparse)
        assert status_of(sparse, "r1.checkpoint.per_epoch") == STATUS_JUDGE
    print("  ok: differing weights PASS, identical weights JUDGE, missing per-epoch checkpoints JUDGE")


def build_eval_pair(root: Path, *, r1_better: bool) -> tuple[Path, Path]:
    r0_rows = [metrics_row(split="train_sanity", index=i, tp=100, fp=50, tn=9000, fn=40) for i in range(3)]
    r0_rows += [metrics_row(split="validation", index=1000 + i, tp=100, fp=60, tn=9000, fn=50) for i in range(18)]
    if r1_better:
        r1_rows = [metrics_row(split="train_sanity", index=i, tp=110, fp=45, tn=9005, fn=30) for i in range(3)]
        r1_rows += [
            metrics_row(split="validation", index=1000 + i, tp=115, fp=55, tn=9005, fn=35) for i in range(18)
        ]
    else:
        r1_rows = [metrics_row(split="train_sanity", index=i, tp=80, fp=70, tn=8980, fn=60) for i in range(3)]
        r1_rows += [
            metrics_row(split="validation", index=1000 + i, tp=70, fp=90, tn=8970, fn=80) for i in range(18)
        ]
    return (
        write_metrics_csv(root / "r0_metrics.csv", r0_rows),
        write_metrics_csv(root / "r1_metrics.csv", r1_rows),
    )


def test_pooled_and_per_video_statistics_are_kept_separate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_r0, csv_r1 = build_eval_pair(root, r1_better=True)
        rows_r0, rows_r1 = read_h5_metrics_csv(csv_r0), read_h5_metrics_csv(csv_r1)

        pooled = {row["split"]: row for row in split_pooled_comparison(rows_r0, rows_r1)}
        assert set(pooled) == {"train_sanity", "validation"}
        validation = pooled["validation"]
        assert validation["num_videos_pooled"] == 18
        # Pooled values come from summed counts, not from averaging rates.
        assert validation["recall_r0"] == (100 * 18) / ((100 + 50) * 18)
        assert validation["f1_diff"] > 0
        assert validation["ignore_positive_rate_r0"] == 10 / 1000

        per_video = {row["split"]: row for row in per_video_diff_comparison(rows_r0, rows_r1)}
        entry = per_video["validation"]
        assert entry["num_videos_paired"] == 18
        assert "f1_median_of_per_video_diffs" in entry and "f1_diff_of_split_medians" in entry
        assert entry["num_videos_improved"] == 18
        assert entry["num_videos_worsened"] == 0
        assert entry["num_tp_zero_videos_r0"] == 0
    print("  ok: pooled metrics derive from summed counts; median-of-diffs and diff-of-medians both reported")


def test_tp_zero_videos_are_counted_per_arm() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r0_rows = [metrics_row(split="validation", index=1000 + i, tp=100, fp=10, tn=900, fn=10) for i in range(4)]
        r1_rows = [metrics_row(split="validation", index=1000 + i, tp=100, fp=10, tn=900, fn=10) for i in range(2)]
        r1_rows += [metrics_row(split="validation", index=1000 + i, tp=0, fp=10, tn=900, fn=110) for i in range(2, 4)]
        csv_r0 = write_metrics_csv(root / "r0.csv", r0_rows)
        csv_r1 = write_metrics_csv(root / "r1.csv", r1_rows)
        entry = per_video_diff_comparison(read_h5_metrics_csv(csv_r0), read_h5_metrics_csv(csv_r1))[0]
        assert entry["num_tp_zero_videos_r0"] == 0
        assert entry["num_tp_zero_videos_r1"] == 2
    print("  ok: TP0 videos are counted separately for each arm")


def test_selection_criteria_never_emit_an_adoption_verdict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_r0, csv_r1 = build_eval_pair(root, r1_better=True)
        rows_r0, rows_r1 = read_h5_metrics_csv(csv_r0), read_h5_metrics_csv(csv_r1)
        pooled = split_pooled_comparison(rows_r0, rows_r1)
        per_video = per_video_diff_comparison(rows_r0, rows_r1)

        selection = evaluate_selection_criteria(pooled, per_video, marginal_threshold=0.005)
        statuses = [entry["status"] for entry in selection["criteria"]]
        assert statuses[0] == "satisfied" and statuses[1] == "satisfied" and statuses[3] == "satisfied"
        # The recall/FPR trade-off is never auto-decided.
        assert statuses[2] == "requires_judgment"
        assert selection["overall"] == "requires_policy_chat_judgment"
        assert "does not decide adoption" in selection["note"]

        worse_r0, worse_r1 = build_eval_pair(root / "worse", r1_better=False)
        worse = evaluate_selection_criteria(
            split_pooled_comparison(read_h5_metrics_csv(worse_r0), read_h5_metrics_csv(worse_r1)),
            per_video_diff_comparison(read_h5_metrics_csv(worse_r0), read_h5_metrics_csv(worse_r1)),
            marginal_threshold=0.005,
        )
        worse_statuses = [entry["status"] for entry in worse["criteria"]]
        assert worse_statuses[0] == "not_satisfied" and worse_statuses[1] == "not_satisfied"
        assert worse["overall"] == "requires_policy_chat_judgment"
    print("  ok: criteria report satisfied/not_satisfied/requires_judgment; the overall verdict is always deferred")


def test_marginal_differences_are_annotated_not_silently_accepted() -> None:
    pooled = [{"split": "validation", "f1_r0": 0.5, "f1_r1": 0.5001, "recall_diff": 0.0, "false_positive_rate_diff": 0.0}]
    per_video = [
        {
            "split": "validation",
            "f1_median_of_per_video_diffs": 0.0004,
            "iou_femur_median_r0": 0.4,
            "iou_femur_median_r1": 0.4001,
            "num_tp_zero_videos_r0": 0,
            "num_tp_zero_videos_r1": 0,
        }
    ]
    selection = evaluate_selection_criteria(pooled, per_video, marginal_threshold=0.005)
    first = selection["criteria"][0]
    assert first["status"] == "satisfied", "a positive median still counts as satisfied by the written rule"
    assert first["marginal"] is True, "but the magnitude must be flagged for a human to weigh"
    print("  ok: a +0.04pt median is 'satisfied' by the rule yet explicitly flagged as marginal")


def test_missing_split_data_is_unknown_rather_than_assumed() -> None:
    selection = evaluate_selection_criteria([], [], marginal_threshold=0.005)
    statuses = [entry["status"] for entry in selection["criteria"]]
    assert statuses[0] == STATUS_UNKNOWN and statuses[1] == STATUS_UNKNOWN and statuses[3] == STATUS_UNKNOWN
    assert statuses[2] == "requires_judgment"
    print("  ok: with no evaluation data the computable criteria are UNKNOWN, never assumed satisfied")


def test_shareable_summary_strips_paths_and_passes_the_privacy_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_r0, csv_r1 = build_eval_pair(root, r1_better=True)
        rows_r0, rows_r1 = read_h5_metrics_csv(csv_r0), read_h5_metrics_csv(csv_r1)
        private = {
            "status": "passed",
            "counts": {STATUS_PASS: 10, STATUS_FAIL: 0, STATUS_JUDGE: 0, STATUS_UNKNOWN: 0},
            "results": [{"check": "config.parity_outside_allowlist", "status": STATUS_PASS, "detail": "ok"}],
            "r0_dir": "/mnt/data/runs/r0",
            "r1_dir": "/mnt/data/runs/r1",
            "augmentation": {"r0_mode": "none", "r1_mode": "random_z_rotation"},
            "init_checkpoint": {
                "path": "/mnt/data/init/stage5_pointnext_s_s3dis_partial_init_groupnorm.pt",
                "basename": "stage5_pointnext_s_s3dis_partial_init_groupnorm.pt",
                "sha256": "55ec" + "0" * 60,
            },
            "file_lists": {"train_files.txt": {"count": 162, "content_sha256": "a" * 64}},
            "history": {"r0": {"epochs_recorded": 5}, "r1": {"epochs_recorded": 5}},
            "checkpoint_inventory": {"r0": {"checkpoint_files": ["best.pt"]}},
            "split_pooled_comparison": split_pooled_comparison(rows_r0, rows_r1),
            "per_video_comparison": per_video_diff_comparison(rows_r0, rows_r1),
            "selection_criteria": {"overall": "requires_policy_chat_judgment", "criteria": []},
        }
        shareable = build_shareable_summary(private)
        assert "r0_dir" not in shareable and "r1_dir" not in shareable
        assert "path" not in shareable["init_checkpoint"]
        assert shareable["init_checkpoint"]["sha256"].startswith("55ec")

        privacy = privacy_self_check(shareable)
        assert privacy["timestamp_like_video_ids_absent"], privacy
        assert privacy["absolute_host_paths_absent"], privacy

        # The per-video aggregation must not carry individual video names.
        assert not privacy_self_check(private)["absolute_host_paths_absent"]
    print("  ok: shareable output drops run paths and video identifiers, and passes both privacy patterns")


def write_manifest(path: Path, **settings: Any) -> Path:
    payload = {
        "arm": "r0",
        "mode": "training",
        "git": {"head": "646cb52b3507e174c996c11d6889d6d0f549dfe5", "worktree_clean": True},
        "settings": {
            "augmentation": "none",
            "augmentation_rotation_degrees": 15.0,
            "seed": 42,
            "epochs": 5,
            "pointnext_norm": "groupnorm",
            "pointnext_norm_groups": 8,
            "save_every": 1,
            **settings,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_metrics_can_be_read_from_one_csv_per_split() -> None:
    """The anonymized export writes train_sanity and validation metrics to
    separate files, so the checker must accept several CSVs per arm."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sanity = write_metrics_csv(root / "train_sanity.csv",
            [metrics_row(split="train_sanity", index=i, tp=100, fp=50, tn=9000, fn=40) for i in range(3)])
        validation = write_metrics_csv(root / "validation.csv",
            [metrics_row(split="validation", index=1000 + i, tp=100, fp=60, tn=9000, fn=50) for i in range(18)])

        combined = read_h5_metrics_csv(f"{sanity},{validation}")
        assert len(combined) == 21
        assert {k[0] for k in combined} == {"train_sanity", "validation"}
        assert len(read_h5_metrics_csv(str(validation))) == 18

        expect_duplicate = False
        try:
            read_h5_metrics_csv(f"{validation},{validation}")
        except ValueError as error:
            expect_duplicate = "duplicate" in str(error)
        assert expect_duplicate, "the same CSV passed twice must be rejected, not silently merged"
    print("  ok: one CSV per split is accepted, a single CSV still works, and a repeated file is rejected")


def test_manifest_and_config_agreeing_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = write_manifest(Path(tmp) / "m.json", save_every=10)
        results: list[dict[str, Any]] = []
        report = check_manifest_against_config(
            manifest, base_config(save_every=10), arm="r0", results=results
        )
        assert report["mismatches"] == []
        assert status_of(results, "r0.manifest.settings_match_config") == STATUS_PASS
        assert status_of(results, "r0.manifest.worktree_clean") == STATUS_PASS
    print("  ok: a manifest whose intended settings all reached the run passes")


def test_the_real_save_every_discrepancy_is_reported_as_judge_not_fail() -> None:
    """The actual S5-15 P3 case: the launcher exported SAVE_EVERY=1 but
    train_stage5.sh overwrote it, so config.json recorded 10. It changes which
    artifacts were written, not the comparison, so it is a judgment call."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest = write_manifest(Path(tmp) / "m.json", save_every=1)
        results: list[dict[str, Any]] = []
        report = check_manifest_against_config(
            manifest, base_config(save_every=10), arm="r0", results=results
        )
        assert len(report["mismatches"]) == 1
        mismatch = report["mismatches"][0]
        assert mismatch == {"setting": "save_every", "intended": 1, "effective": 10, "critical": False}
        assert status_of(results, "r0.manifest.save_every") == STATUS_JUDGE
        assert "not the comparison itself" in [r["detail"] for r in results if r["check"].endswith("save_every")][0]
    print("  ok: the save_every intended=1 / effective=10 discrepancy is caught and rated JUDGE")


def test_a_setting_that_changes_comparability_is_fatal() -> None:
    for setting, effective in (("seed", 7), ("epochs", 3), ("augmentation", "random_z_rotation"),
                               ("pointnext_norm", "batchnorm"), ("augmentation_rotation_degrees", 30.0)):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_manifest(Path(tmp) / "m.json")
            results: list[dict[str, Any]] = []
            report = check_manifest_against_config(
                manifest, base_config(**{setting: effective}), arm="r0", results=results
            )
            assert any(m["setting"] == setting and m["critical"] for m in report["mismatches"]), setting
            assert status_of(results, f"r0.manifest.{setting}") == STATUS_FAIL, setting
    print("  ok: seed/epochs/augmentation/norm/rotation-range discrepancies are all FAIL")


def test_a_dirty_or_unknown_worktree_at_launch_is_surfaced() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        payload = json.loads(write_manifest(path).read_text())
        payload["git"]["worktree_clean"] = False
        path.write_text(json.dumps(payload), encoding="utf-8")
        results: list[dict[str, Any]] = []
        check_manifest_against_config(path, base_config(save_every=1), arm="r1", results=results)
        assert status_of(results, "r1.manifest.worktree_clean") == STATUS_JUDGE

        payload["git"]["worktree_clean"] = None
        path.write_text(json.dumps(payload), encoding="utf-8")
        unknown: list[dict[str, Any]] = []
        check_manifest_against_config(path, base_config(save_every=1), arm="r1", results=unknown)
        assert status_of(unknown, "r1.manifest.worktree_clean") == STATUS_UNKNOWN
    print("  ok: a dirty tree at launch is JUDGE and an unreadable one is UNKNOWN, never silently PASS")


def test_launcher_refuses_to_train_without_explicit_confirmation() -> None:
    """The launcher can start training, so it must not do so by default."""
    launcher = REPO_ROOT / "checks" / "real_h5" / "run_stage5_s5_15_arm.sh"
    text = launcher.read_text(encoding="utf-8")
    assert 'CONFIRM_TRAINING="${CONFIRM_TRAINING:-0}"' in text, "confirmation must default to off"
    dry_run_index = text.index('if [[ "${CONFIRM_TRAINING}" != "1" ]]; then')
    launch_index = text.index('bash "${SCRIPT_DIR}/train_stage5.sh"')
    assert dry_run_index < launch_index, "the dry-run exit must come before the launch"
    assert "exit 0" in text[dry_run_index:launch_index], "the dry-run branch must exit before training"
    assert "Dry run: training was NOT started." in text

    # The comparison checker must not be able to launch training at all.
    checker = (REPO_ROOT / "checks" / "real_h5" / "check_stage5_rotation_augmentation_ablation.sh").read_text()
    assert "train_stage5.sh" not in checker.replace(
        "run_stage5_s5_15_arm.sh for the (separately approved) training runs.", ""
    ), "the aggregation checker must never invoke the training launcher"
    result = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print("  ok: training needs CONFIRM_TRAINING=1; the aggregation checker cannot start a run")


def main() -> None:
    tests = [
        test_identical_configs_except_augmentation_pass,
        test_any_other_config_difference_is_reported,
        test_augmentation_modes_must_be_none_and_random_z_rotation,
        test_file_list_representation_difference_is_judge_not_fail,
        test_reference_list_comparison_is_reported_separately,
        test_trained_checkpoint_equality_is_not_required_but_identity_is_flagged,
        test_pooled_and_per_video_statistics_are_kept_separate,
        test_tp_zero_videos_are_counted_per_arm,
        test_metrics_can_be_read_from_one_csv_per_split,
        test_manifest_and_config_agreeing_passes,
        test_the_real_save_every_discrepancy_is_reported_as_judge_not_fail,
        test_a_setting_that_changes_comparability_is_fatal,
        test_a_dirty_or_unknown_worktree_at_launch_is_surfaced,
        test_selection_criteria_never_emit_an_adoption_verdict,
        test_marginal_differences_are_annotated_not_silently_accepted,
        test_missing_split_data_is_unknown_rather_than_assumed,
        test_shareable_summary_strips_paths_and_passes_the_privacy_check,
        test_launcher_refuses_to_train_without_explicit_confirmation,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 S5-15 rotation-augmentation ablation synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
