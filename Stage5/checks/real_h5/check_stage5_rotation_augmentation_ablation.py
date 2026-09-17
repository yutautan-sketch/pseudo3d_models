from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CHECKS_REAL_H5 = Path(__file__).resolve().parent
if str(CHECKS_REAL_H5) not in sys.path:
    sys.path.insert(0, str(CHECKS_REAL_H5))

from stage5.utils.file_list_mode import compare_file_lists  # noqa: E402

# Reused as-is because evaluate_stage5.py's h5_metrics.csv already carries the
# same true/false_positive/negative_count columns this expects. The S5-14
# module is not modified, and its condition-vs-identity median helper is NOT
# reused: that one is keyed by transform condition, while two training arms
# need a different pairing, so it is reimplemented here rather than bent.
from check_stage5_coordinate_transform_reconciliation import pooled_confusion_metrics  # noqa: E402

# ----------------------------------------------------------------------------
# S5-15 P3: R0 (augmentation=none) vs R1 (random_z_rotation) comparison.
#
# Aggregation and verification only -- this checker never trains and never
# touches a checkpoint's weights. It confirms the two arms differ *only* in
# augmentation, then reports pooled and per-video evaluation differences plus
# the pre-registered selection criteria as satisfied / not satisfied /
# requires_judgment. It deliberately does not emit an adoption verdict.
# ----------------------------------------------------------------------------

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_JUDGE = "JUDGE"
STATUS_UNKNOWN = "UNKNOWN"

# The only config keys allowed to differ between the arms. Everything else --
# including train_list/val_list, seed, class weight and every model knob -- must
# match exactly, so a stray setting difference cannot hide behind a broad
# exclusion list.
INTENDED_DIFFERENCE_KEYS = ("output_dir", "augmentation", "augmentation_info")

EVALUATED_SPLITS = ("train_sanity", "validation")
PER_VIDEO_DIFF_METRICS = ("f1", "iou_femur", "precision", "recall", "false_positive_rate")

TIMESTAMP_VIDEO_ID_PATTERN = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9]+")
ABSOLUTE_HOST_PATH_PATTERN = re.compile(r"(/mnt/data|/home/[A-Za-z0-9_.-]+)")


def record(results: list[dict[str, Any]], check: str, status: str, detail: str, **extra: Any) -> None:
    row: dict[str, Any] = {"check": check, "status": status, "detail": detail}
    row.update(extra)
    results.append(row)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def read_h5_metrics_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["split"], row["video_name"])
            if key in rows:
                raise ValueError(f"duplicate (split, video_name) in {path.name}: {key}")
            rows[key] = row
    return rows


def compare_configs(config_r0: dict[str, Any], config_r1: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    diffs: list[str] = []
    for key in sorted(set(config_r0) | set(config_r1)):
        if key in INTENDED_DIFFERENCE_KEYS:
            continue
        value_r0 = config_r0.get(key, "<<missing>>")
        value_r1 = config_r1.get(key, "<<missing>>")
        if value_r0 != value_r1:
            diffs.append(f"{key}: R0={value_r0!r} R1={value_r1!r}")
    record(
        results,
        "config.parity_outside_allowlist",
        STATUS_PASS if not diffs else STATUS_FAIL,
        (
            f"every config key outside {list(INTENDED_DIFFERENCE_KEYS)} matches"
            if not diffs
            else f"{len(diffs)} unintended difference(s): {diffs[:5]}"
        ),
    )
    return diffs


def check_fixed_list_mode(config: dict[str, Any], *, arm: str, results: list[dict[str, Any]]) -> None:
    used_lists = bool(config.get("train_list")) and bool(config.get("val_list"))
    record(
        results,
        f"{arm}.fixed_list_mode",
        STATUS_PASS if used_lists else STATUS_FAIL,
        (
            "trained from explicit train_list/val_list"
            if used_lists
            else f"train_list={config.get('train_list')!r} val_list={config.get('val_list')!r}: "
            "the run did not use the saved lists"
        ),
    )
    if used_lists and config.get("train_dir"):
        record(
            results,
            f"{arm}.no_train_dir",
            STATUS_FAIL,
            f"train_dir is also set ({config.get('train_dir')!r})",
        )


def check_augmentation_settings(
    config_r0: dict[str, Any],
    config_r1: dict[str, Any],
    *,
    expected_degrees: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_r0 = config_r0.get("augmentation")
    mode_r1 = config_r1.get("augmentation")
    record(
        results,
        "augmentation.r0_is_none",
        STATUS_PASS if mode_r0 == "none" else STATUS_FAIL,
        f"R0 augmentation={mode_r0!r}",
    )
    record(
        results,
        "augmentation.r1_is_random_z_rotation",
        STATUS_PASS if mode_r1 == "random_z_rotation" else STATUS_FAIL,
        f"R1 augmentation={mode_r1!r}",
    )

    info_r1 = config_r1.get("augmentation_info") or {}
    degrees = info_r1.get("max_abs_degrees")
    record(
        results,
        "augmentation.r1_degrees",
        STATUS_PASS if degrees == expected_degrees else STATUS_FAIL,
        f"R1 max_abs_degrees={degrees!r}, expected {expected_degrees}",
    )
    base_seed = info_r1.get("base_seed")
    record(
        results,
        "augmentation.r1_seed_recorded",
        STATUS_PASS if isinstance(base_seed, int) else STATUS_FAIL,
        f"R1 base_seed={base_seed!r} seed_source={info_r1.get('seed_source')!r}",
    )
    return {"r0_mode": mode_r0, "r1_mode": mode_r1, "r1_augmentation_info": info_r1}


def check_init_checkpoint(
    config_r0: dict[str, Any],
    config_r1: dict[str, Any],
    *,
    expected_sha256: str | None,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    path_r0 = config_r0.get("checkpoint")
    path_r1 = config_r1.get("checkpoint")
    same_path = bool(path_r0) and path_r0 == path_r1
    record(
        results,
        "init_checkpoint.same_path",
        STATUS_PASS if same_path else STATUS_FAIL,
        "both arms record the same initialization checkpoint"
        if same_path
        else f"paths differ (R0 basename={Path(str(path_r0)).name!r}, R1 basename={Path(str(path_r1)).name!r})",
    )

    report: dict[str, Any] = {"basename": Path(str(path_r0)).name if path_r0 else None}
    if not path_r0:
        return report

    path = Path(str(path_r0))
    if not path.is_file():
        record(results, "init_checkpoint.sha256", STATUS_UNKNOWN, "file not present on this host")
        return report

    digest = sha256_file(path)
    report["sha256"] = digest
    if expected_sha256:
        matches = digest == expected_sha256
        record(
            results,
            "init_checkpoint.sha256",
            STATUS_PASS if matches else STATUS_FAIL,
            f"{digest}" + ("" if matches else f" != expected {expected_sha256}"),
        )
    else:
        record(results, "init_checkpoint.sha256", STATUS_PASS, f"{digest} (no expected value given)")
    return report


def check_file_lists(
    run_dir_r0: Path,
    run_dir_r1: Path,
    *,
    reference_dir: Path | None,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in ("train_files.txt", "val_files.txt"):
        path_r0, path_r1 = run_dir_r0 / name, run_dir_r1 / name
        if not path_r0.is_file() or not path_r1.is_file():
            record(results, f"file_list.{name}.present", STATUS_FAIL, "missing from one or both run directories")
            continue
        lines_r0 = [Path(line) for line in read_lines(path_r0)]
        lines_r1 = [Path(line) for line in read_lines(path_r1)]
        comparison = compare_file_lists(lines_r1, lines_r0)
        status = STATUS_PASS if comparison["representation_identical"] else (
            STATUS_JUDGE if comparison["representation_differs_only"] else STATUS_FAIL
        )
        record(
            results,
            f"file_list.{name}.arms_match",
            status,
            (
                f"both arms list the same {comparison['actual_count']} files identically"
                if status == STATUS_PASS
                else "same videos in the same order but written differently (path representation only)"
                if status == STATUS_JUDGE
                else f"content differs: only_in_R1={comparison['only_in_actual']} only_in_R0={comparison['only_in_reference']}"
            ),
        )
        entry = {
            "count": len(lines_r0),
            "content_sha256": comparison["content_sha256_reference"],
            "identity_sha256": comparison["identity_sha256_reference"],
            "arms_content_identical": comparison["content_identical"],
        }

        if reference_dir is not None:
            reference_path = reference_dir / name
            if reference_path.is_file():
                reference_lines = [Path(line) for line in read_lines(reference_path)]
                against_reference = compare_file_lists(lines_r0, reference_lines)
                if against_reference["representation_identical"]:
                    reference_status, detail = STATUS_PASS, "identical to the reference list"
                elif against_reference["representation_differs_only"]:
                    reference_status, detail = (
                        STATUS_JUDGE,
                        "same videos in the same order, different path representation "
                        "(a representation difference, not a content one)",
                    )
                else:
                    reference_status, detail = (
                        STATUS_FAIL,
                        f"different videos or order: only_here={against_reference['only_in_actual']} "
                        f"only_in_reference={against_reference['only_in_reference']}",
                    )
                record(results, f"file_list.{name}.vs_reference", reference_status, detail)
                entry["reference_content_identical"] = against_reference["content_identical"]
                entry["reference_identity_sha256"] = against_reference["identity_sha256_reference"]
            else:
                record(results, f"file_list.{name}.vs_reference", STATUS_UNKNOWN, "reference list not found")
        report[name] = entry
    return report


def check_history(run_dir: Path, *, arm: str, expected_epochs: int, results: list[dict[str, Any]]) -> dict[str, Any]:
    path = run_dir / "history.json"
    if not path.is_file():
        record(results, f"{arm}.history", STATUS_FAIL, "history.json not found in the run directory")
        return {}
    history = load_json(path)
    if not isinstance(history, list):
        record(results, f"{arm}.history", STATUS_FAIL, "history.json is not a list")
        return {}
    record(
        results,
        f"{arm}.history.epochs",
        STATUS_PASS if len(history) == expected_epochs else STATUS_FAIL,
        f"{len(history)} epoch record(s), expected {expected_epochs}",
    )

    nonfinite: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}", item)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            nonfinite.append(prefix)

    for entry in history:
        if isinstance(entry, dict):
            walk(f"epoch{entry.get('epoch')}.train", entry.get("train"))
            if entry.get("val") is not None:
                walk(f"epoch{entry.get('epoch')}.val", entry.get("val"))
    record(
        results,
        f"{arm}.history.finite_metrics",
        STATUS_PASS if not nonfinite else STATUS_FAIL,
        "all recorded metrics finite" if not nonfinite else f"non-finite: {nonfinite[:5]}",
    )
    return {"epochs_recorded": len(history)}


def check_checkpoints(
    run_dir_r0: Path,
    run_dir_r1: Path,
    *,
    expected_epochs: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for arm, run_dir in (("r0", run_dir_r0), ("r1", run_dir_r1)):
        present = sorted(p.name for p in run_dir.glob("*.pt"))
        per_epoch = [name for name in present if name.startswith("checkpoint_epoch_")]
        inventory[arm] = {"checkpoint_files": present, "per_epoch_checkpoints": per_epoch}
        for name in ("best.pt", "last.pt"):
            record(
                results,
                f"{arm}.checkpoint.{name}",
                STATUS_PASS if name in present else STATUS_FAIL,
                "present" if name in present else "missing",
            )
        record(
            results,
            f"{arm}.checkpoint.per_epoch",
            STATUS_PASS if len(per_epoch) >= expected_epochs else STATUS_JUDGE,
            f"{len(per_epoch)} per-epoch checkpoint(s) for {expected_epochs} epochs (SAVE_EVERY=1 expected)",
        )

    # Deliberately NOT a requirement that the trained weights match: the arms
    # are supposed to diverge. Identical files would instead suggest the
    # augmentation never reached training, so that is what gets flagged.
    last_r0, last_r1 = run_dir_r0 / "last.pt", run_dir_r1 / "last.pt"
    if last_r0.is_file() and last_r1.is_file():
        identical = sha256_file(last_r0) == sha256_file(last_r1)
        record(
            results,
            "trained_checkpoints.differ",
            STATUS_PASS if not identical else STATUS_JUDGE,
            "R0 and R1 last.pt differ, as expected for two arms (hash equality is not required)"
            if not identical
            else "R0 and R1 last.pt are byte-identical -- check that augmentation actually reached training",
        )
    else:
        record(results, "trained_checkpoints.differ", STATUS_UNKNOWN, "last.pt missing on this host")
    return inventory


def split_pooled_comparison(
    rows_r0: dict[tuple[str, str], dict[str, str]],
    rows_r1: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in EVALUATED_SPLITS:
        split_r0 = [row for (s, _), row in rows_r0.items() if s == split]
        split_r1 = [row for (s, _), row in rows_r1.items() if s == split]
        if not split_r0 or not split_r1:
            continue
        pooled_r0 = pooled_confusion_metrics(split_r0)
        pooled_r1 = pooled_confusion_metrics(split_r1)
        entry: dict[str, Any] = {"split": split}
        for metric in ("precision", "recall", "false_positive_rate", "f1", "iou_femur"):
            value_r0, value_r1 = pooled_r0[metric], pooled_r1[metric]
            entry[f"{metric}_r0"] = value_r0
            entry[f"{metric}_r1"] = value_r1
            entry[f"{metric}_diff"] = (
                None if value_r0 is None or value_r1 is None else value_r1 - value_r0
            )
        entry["num_videos_pooled"] = pooled_r0["num_videos_pooled"]
        # Ignore-region positives are counted apart from FP on valid GT.
        for arm, split_rows in (("r0", split_r0), ("r1", split_r1)):
            ignore_points = sum(int(r.get("ignore_point_count", 0)) for r in split_rows)
            ignore_positive = sum(int(r.get("ignore_predicted_positive_count", 0)) for r in split_rows)
            entry[f"ignore_point_count_{arm}"] = ignore_points
            entry[f"ignore_predicted_positive_count_{arm}"] = ignore_positive
            entry[f"ignore_positive_rate_{arm}"] = (
                ignore_positive / ignore_points if ignore_points > 0 else None
            )
        out.append(entry)
    return out


def per_video_diff_comparison(
    rows_r0: dict[tuple[str, str], dict[str, str]],
    rows_r1: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    """Paired per-video differences, keeping the median of per-video diffs and
    the difference of the split medians as separate statistics (they can
    diverge, as S5-14 supplement 3 found)."""
    out: list[dict[str, Any]] = []
    for split in EVALUATED_SPLITS:
        keys = sorted(
            key for key in set(rows_r0) & set(rows_r1) if key[0] == split
        )
        if not keys:
            continue
        entry: dict[str, Any] = {"split": split, "num_videos_paired": len(keys)}
        for metric in PER_VIDEO_DIFF_METRICS:
            values_r0 = [parse_optional_float(rows_r0[key].get(metric)) for key in keys]
            values_r1 = [parse_optional_float(rows_r1[key].get(metric)) for key in keys]
            paired = [
                (a, b) for a, b in zip(values_r0, values_r1) if a is not None and b is not None
            ]
            if not paired:
                entry[f"{metric}_median_of_per_video_diffs"] = None
                entry[f"{metric}_diff_of_split_medians"] = None
                continue
            diffs = [b - a for a, b in paired]
            entry[f"{metric}_median_of_per_video_diffs"] = statistics.median(diffs)
            entry[f"{metric}_diff_of_split_medians"] = statistics.median(
                [b for _, b in paired]
            ) - statistics.median([a for a, _ in paired])
            entry[f"{metric}_median_r0"] = statistics.median([a for a, _ in paired])
            entry[f"{metric}_median_r1"] = statistics.median([b for _, b in paired])
            if metric == "f1":
                entry["num_videos_improved"] = sum(1 for d in diffs if d > 0)
                entry["num_videos_worsened"] = sum(1 for d in diffs if d < 0)
                entry["num_videos_unchanged"] = sum(1 for d in diffs if d == 0)
        for arm, rows in (("r0", rows_r0), ("r1", rows_r1)):
            entry[f"num_tp_zero_videos_{arm}"] = sum(
                1 for key in keys if int(rows[key].get("true_positive_count", 0)) == 0
            )
        out.append(entry)
    return out


def evaluate_selection_criteria(
    pooled_rows: list[dict[str, Any]],
    per_video_rows: list[dict[str, Any]],
    *,
    marginal_threshold: float,
) -> dict[str, Any]:
    """The pre-registered P3 criteria, reported as satisfied / not_satisfied /
    requires_judgment. Never an adoption verdict: a trade-off or a marginal
    difference is for the policy chat to weigh, not for this script."""
    pooled = {row["split"]: row for row in pooled_rows}
    per_video = {row["split"]: row for row in per_video_rows}
    criteria: list[dict[str, Any]] = []

    def status_of(value: bool | None) -> str:
        if value is None:
            return STATUS_UNKNOWN
        return "satisfied" if value else "not_satisfied"

    validation = per_video.get("validation")
    validation_pooled = pooled.get("validation")
    f1_median_diff = validation.get("f1_median_of_per_video_diffs") if validation else None
    criteria.append(
        {
            "criterion": "1. validation paired F1 diff median is positive",
            "status": status_of(None if f1_median_diff is None else f1_median_diff > 0),
            "value": f1_median_diff,
            "marginal": (
                None if f1_median_diff is None else abs(f1_median_diff) < marginal_threshold
            ),
        }
    )

    iou_r0 = validation.get("iou_femur_median_r0") if validation else None
    iou_r1 = validation.get("iou_femur_median_r1") if validation else None
    pooled_f1_r0 = validation_pooled.get("f1_r0") if validation_pooled else None
    pooled_f1_r1 = validation_pooled.get("f1_r1") if validation_pooled else None
    tp0_r0 = validation.get("num_tp_zero_videos_r0") if validation else None
    tp0_r1 = validation.get("num_tp_zero_videos_r1") if validation else None
    condition_2 = (
        None
        if None in (iou_r0, iou_r1, pooled_f1_r0, pooled_f1_r1, tp0_r0, tp0_r1)
        else (iou_r1 >= iou_r0 and pooled_f1_r1 >= pooled_f1_r0 and tp0_r1 <= tp0_r0)
    )
    criteria.append(
        {
            "criterion": "2. validation split median IoU and pooled F1 not below R0, TP0 not increased",
            "status": status_of(condition_2),
            "value": {
                "median_iou_r0": iou_r0,
                "median_iou_r1": iou_r1,
                "pooled_f1_r0": pooled_f1_r0,
                "pooled_f1_r1": pooled_f1_r1,
                "tp_zero_videos_r0": tp0_r0,
                "tp_zero_videos_r1": tp0_r1,
            },
        }
    )

    criteria.append(
        {
            "criterion": "3. no worsening explained by recall drop or FP increase alone; trade-off stated",
            "status": "requires_judgment",
            "value": {
                "validation_pooled_recall_diff": validation_pooled.get("recall_diff") if validation_pooled else None,
                "validation_pooled_fpr_diff": (
                    validation_pooled.get("false_positive_rate_diff") if validation_pooled else None
                ),
                "validation_recall_median_of_per_video_diffs": (
                    validation.get("recall_median_of_per_video_diffs") if validation else None
                ),
                "validation_fpr_median_of_per_video_diffs": (
                    validation.get("false_positive_rate_median_of_per_video_diffs") if validation else None
                ),
            },
            "note": "a recall/FPR trade-off is a judgment for the policy chat, not a computed verdict",
        }
    )

    sanity = per_video.get("train_sanity")
    sanity_f1_diff = sanity.get("f1_median_of_per_video_diffs") if sanity else None
    sanity_iou_diff = sanity.get("iou_femur_median_of_per_video_diffs") if sanity else None
    sanity_tp0_r0 = sanity.get("num_tp_zero_videos_r0") if sanity else None
    sanity_tp0_r1 = sanity.get("num_tp_zero_videos_r1") if sanity else None
    condition_4 = (
        None
        if None in (sanity_f1_diff, sanity_iou_diff, sanity_tp0_r0, sanity_tp0_r1)
        else (sanity_f1_diff >= 0 and sanity_iou_diff >= 0 and sanity_tp0_r1 <= sanity_tp0_r0)
    )
    criteria.append(
        {
            "criterion": "4. train sanity median F1/IoU and TP0 do not worsen",
            "status": status_of(condition_4),
            "value": {
                "f1_median_of_per_video_diffs": sanity_f1_diff,
                "iou_median_of_per_video_diffs": sanity_iou_diff,
                "tp_zero_videos_r0": sanity_tp0_r0,
                "tp_zero_videos_r1": sanity_tp0_r1,
            },
            "note": "train sanity is 3 videos and is a safety check, not a representative sample",
        }
    )

    return {
        "criteria": criteria,
        "overall": "requires_policy_chat_judgment",
        "note": (
            "Even with every computable criterion satisfied this stays a single-seed provisional "
            "result. This script does not decide adoption."
        ),
    }


def privacy_self_check(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    timestamp_hits = TIMESTAMP_VIDEO_ID_PATTERN.findall(text)
    path_hits = ABSOLUTE_HOST_PATH_PATTERN.findall(text)
    return {
        "timestamp_like_video_ids_absent": not timestamp_hits,
        "absolute_host_paths_absent": not path_hits,
        "timestamp_like_video_id_hits": len(timestamp_hits),
        "absolute_host_path_hits": len(path_hits),
    }


def build_shareable_summary(private_summary: dict[str, Any]) -> dict[str, Any]:
    shareable = {
        key: private_summary[key]
        for key in (
            "status",
            "counts",
            "results",
            "augmentation",
            "history",
            "checkpoint_inventory",
            "split_pooled_comparison",
            "per_video_comparison",
            "selection_criteria",
        )
        if key in private_summary
    }
    init_checkpoint = dict(private_summary.get("init_checkpoint", {}))
    init_checkpoint.pop("path", None)
    shareable["init_checkpoint"] = init_checkpoint
    shareable["file_lists"] = private_summary.get("file_lists", {})
    shareable["run_dir_aliases"] = {"r0": "r0_control", "r1": "r1_random_z_rotation"}
    return shareable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-15 P3: verify that the R0 (none) and R1 (random_z_rotation) training runs "
        "differ only in augmentation, then compare their fixed-21-video evaluations. "
        "Aggregation only: this never trains and never modifies a run directory."
    )
    parser.add_argument("--r0_dir", required=True, help="R0 (control, augmentation=none) run directory")
    parser.add_argument("--r1_dir", required=True, help="R1 (random_z_rotation) run directory")
    parser.add_argument("--expected_epochs", type=int, default=5)
    parser.add_argument("--expected_rotation_degrees", type=float, default=15.0)
    parser.add_argument("--expected_init_sha256", default=None)
    parser.add_argument("--reference_run_dir", default=None, help="Run dir whose saved lists are the reference")
    parser.add_argument("--eval_csv_r0", default=None, help="R0 evaluate_stage5.py h5_metrics.csv")
    parser.add_argument("--eval_csv_r1", default=None, help="R1 evaluate_stage5.py h5_metrics.csv")
    parser.add_argument(
        "--marginal_threshold",
        type=float,
        default=0.005,
        help="Annotates a criterion-1 result as marginal; never changes satisfied/not_satisfied",
    )
    parser.add_argument("--private_json", default=None)
    parser.add_argument("--shareable_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir_r0, run_dir_r1 = Path(args.r0_dir), Path(args.r1_dir)
    results: list[dict[str, Any]] = []

    for arm, run_dir in (("r0", run_dir_r0), ("r1", run_dir_r1)):
        if not run_dir.is_dir():
            print(f"FAIL: {arm} run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(2)
        if not (run_dir / "config.json").is_file():
            print(f"FAIL: {arm} config.json not found", file=sys.stderr)
            sys.exit(2)
    if run_dir_r0.resolve() == run_dir_r1.resolve():
        print("FAIL: --r0_dir and --r1_dir point at the same run directory", file=sys.stderr)
        sys.exit(2)

    config_r0 = load_json(run_dir_r0 / "config.json")
    config_r1 = load_json(run_dir_r1 / "config.json")

    compare_configs(config_r0, config_r1, results)
    check_fixed_list_mode(config_r0, arm="r0", results=results)
    check_fixed_list_mode(config_r1, arm="r1", results=results)
    augmentation = check_augmentation_settings(
        config_r0, config_r1, expected_degrees=args.expected_rotation_degrees, results=results
    )
    init_checkpoint = check_init_checkpoint(
        config_r0, config_r1, expected_sha256=args.expected_init_sha256, results=results
    )
    file_lists = check_file_lists(
        run_dir_r0,
        run_dir_r1,
        reference_dir=Path(args.reference_run_dir) if args.reference_run_dir else None,
        results=results,
    )
    history = {
        "r0": check_history(run_dir_r0, arm="r0", expected_epochs=args.expected_epochs, results=results),
        "r1": check_history(run_dir_r1, arm="r1", expected_epochs=args.expected_epochs, results=results),
    }
    checkpoint_inventory = check_checkpoints(
        run_dir_r0, run_dir_r1, expected_epochs=args.expected_epochs, results=results
    )

    pooled_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    if args.eval_csv_r0 and args.eval_csv_r1:
        rows_r0 = read_h5_metrics_csv(Path(args.eval_csv_r0))
        rows_r1 = read_h5_metrics_csv(Path(args.eval_csv_r1))
        only_r0 = sorted(set(rows_r0) - set(rows_r1))
        only_r1 = sorted(set(rows_r1) - set(rows_r0))
        record(
            results,
            "evaluation.same_targets",
            STATUS_PASS if not only_r0 and not only_r1 else STATUS_FAIL,
            f"{len(rows_r0)} evaluated (split, video) pairs match"
            if not only_r0 and not only_r1
            else f"{len(only_r0)} only in R0, {len(only_r1)} only in R1",
        )
        pooled_rows = split_pooled_comparison(rows_r0, rows_r1)
        per_video_rows = per_video_diff_comparison(rows_r0, rows_r1)
        selection = evaluate_selection_criteria(
            pooled_rows, per_video_rows, marginal_threshold=args.marginal_threshold
        )
    else:
        record(
            results,
            "evaluation.same_targets",
            STATUS_UNKNOWN,
            "evaluation CSVs not supplied; ran config/parity checks only",
        )

    counts = {
        status: sum(1 for row in results if row["status"] == status)
        for status in (STATUS_PASS, STATUS_FAIL, STATUS_JUDGE, STATUS_UNKNOWN)
    }
    status = "passed" if counts[STATUS_FAIL] == 0 else "failed"

    private_summary = {
        "status": status,
        "counts": counts,
        "results": results,
        "r0_dir": str(run_dir_r0),
        "r1_dir": str(run_dir_r1),
        "augmentation": augmentation,
        "init_checkpoint": init_checkpoint,
        "file_lists": file_lists,
        "history": history,
        "checkpoint_inventory": checkpoint_inventory,
        "split_pooled_comparison": pooled_rows,
        "per_video_comparison": per_video_rows,
        "selection_criteria": selection,
    }
    shareable_summary = build_shareable_summary(private_summary)
    shareable_summary["privacy_self_check"] = privacy_self_check(shareable_summary)

    for path_str, payload in ((args.private_json, private_summary), (args.shareable_json, shareable_summary)):
        if path_str:
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Stage5 S5-15 rotation-augmentation ablation check")
    print(f"  status : {status}")
    print(f"  counts : {counts}")
    for row in results:
        if row["status"] != STATUS_PASS:
            print(f"  [{row['status']}] {row['check']}: {row['detail']}")
    for row in pooled_rows:
        print(
            f"  pooled {row['split']}: f1 {row['f1_r0']} -> {row['f1_r1']} "
            f"(diff {row['f1_diff']}), recall diff {row['recall_diff']}, FPR diff {row['false_positive_rate_diff']}"
        )
    for row in per_video_rows:
        print(
            f"  per-video {row['split']}: F1 median-of-diffs "
            f"{row.get('f1_median_of_per_video_diffs')}, diff-of-medians {row.get('f1_diff_of_split_medians')}, "
            f"TP0 {row.get('num_tp_zero_videos_r0')} -> {row.get('num_tp_zero_videos_r1')}"
        )
    if selection:
        for entry in selection["criteria"]:
            print(f"  criterion {entry['status']}: {entry['criterion']}")
        print(f"  overall: {selection['overall']}")
    if args.shareable_json:
        privacy = shareable_summary["privacy_self_check"]
        print(
            "  shareable privacy self-check: "
            f"timestamp_like_video_ids_absent={privacy['timestamp_like_video_ids_absent']} "
            f"absolute_host_paths_absent={privacy['absolute_host_paths_absent']}"
        )

    sys.exit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
