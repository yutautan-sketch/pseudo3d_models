from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# S5-13 Step F3/F4/F7: config-parity and completion checks for the W-A
# (strong auto-derived control, reused from S5-12 Run A) vs W-B (moderate
# fixed [0.5, 1.5]) class-weight ablation.
#
# This checker only reads each training run directory's config.json /
# train_files.txt / val_files.txt / history.json, plus (optionally) the
# initial checkpoint file's byte-level SHA-256 and evaluate_stage5.py's
# h5_metrics.csv. It does not import torch/h5py/numpy and does not touch CUDA
# -- everything here is plain JSON/CSV/text, so it can run identically on the
# CPU-only development container and the GPU training machine.
# ----------------------------------------------------------------------------


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# Config keys allowed to differ between W-A and W-B: the ablation's intended
# variable (class weight) and run/output plumbing that does not affect
# training behavior (input-list mechanism, output path). W-B may reuse W-A's
# saved train_files.txt/val_files.txt via --train_list/--val_list instead of
# re-deriving the split from --train_dir/--val_fraction, so those args are
# allowed to differ too -- the authoritative check is that the *resolved*
# train_files.txt/val_files.txt content is identical (checked separately).
# Every other config.json key must match exactly, which is what enforces
# "the only intended difference is class weight" (request doc 7.2 item 4).
CLASS_WEIGHT_CONFIG_KEYS = ("class_weight", "class_weight_info")
IGNORED_CONFIG_KEYS = (
    "output_dir",
    "train_list",
    "train_dir",
    "val_list",
    "val_dir",
    "max_train_files",
    "max_val_files",
    "val_fraction",
)

POINT_SET_FIELDS = (
    "total_point_count",
    "valid_point_count",
    "valid_positive_count",
    "valid_background_count",
)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: Path) -> list[str]:
    require(path.is_file(), f"Missing file: {path}")
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


def parse_weight_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def compare_config(config_a: dict[str, Any], config_b: dict[str, Any]) -> list[str]:
    keys = sorted(set(config_a) | set(config_b))
    diffs: list[str] = []
    for key in keys:
        if key in CLASS_WEIGHT_CONFIG_KEYS or key in IGNORED_CONFIG_KEYS:
            continue
        value_a = config_a.get(key, "<<missing>>")
        value_b = config_b.get(key, "<<missing>>")
        if value_a != value_b:
            diffs.append(f"{key}: {value_a!r} != {value_b!r}")
    return diffs


def check_file_lists(run_dir_a: Path, run_dir_b: Path, *, name: str) -> int:
    path_a = run_dir_a / name
    path_b = run_dir_b / name
    lines_a = read_lines(path_a)
    lines_b = read_lines(path_b)
    if lines_a != lines_b:
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(lines_a, lines_b)) if x != y),
            min(len(lines_a), len(lines_b)),
        )
        raise AssertionError(
            f"{name} differs between W-A ({path_a}) and W-B ({path_b}): "
            f"{len(lines_a)} vs {len(lines_b)} lines, first mismatch at index {first_diff}"
        )
    return len(lines_a)


def check_resolved_weight(
    config: dict[str, Any],
    *,
    expected: list[float] | None,
    label: str,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    info = config.get("class_weight_info")
    require(info is not None, f"{label}: config.json has no class_weight_info")
    resolved = info.get("resolved")
    if expected is not None:
        require(resolved is not None, f"{label}: class_weight_info.resolved is null")
        require(
            len(resolved) == len(expected)
            and all(math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance) for a, b in zip(resolved, expected)),
            f"{label}: resolved class weight {resolved} does not match expected {expected}",
        )
    return {"mode": info.get("mode"), "requested": info.get("requested"), "resolved": resolved}


def check_history(run_dir: Path, *, expected_epochs: int, label: str) -> dict[str, Any]:
    history = load_json(run_dir / "history.json")
    require(
        isinstance(history, list) and len(history) == expected_epochs,
        f"{label}: history.json has "
        f"{len(history) if isinstance(history, list) else 'invalid'} epoch(s), expected {expected_epochs}",
    )
    require(
        history[-1]["epoch"] == expected_epochs,
        f"{label}: last recorded epoch is {history[-1]['epoch']}, expected {expected_epochs}",
    )

    nonfinite: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}", item)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                nonfinite.append(prefix)

    for record in history:
        walk(f"epoch{record.get('epoch')}.train", record.get("train"))
        if record.get("val") is not None:
            walk(f"epoch{record.get('epoch')}.val", record.get("val"))
    require(not nonfinite, f"{label}: non-finite metric value(s) in history.json: {nonfinite[:10]}")
    return {"epochs_recorded": len(history), "last_epoch": history[-1]["epoch"]}


def check_checkpoint_present(run_dir: Path, *, filename: str, label: str) -> Path:
    path = run_dir / filename
    require(path.is_file(), f"{label}: missing checkpoint {path}")
    return path


def read_h5_metrics_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    require(path.is_file(), f"Missing evaluation metrics CSV: {path}")
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["split"], row["video_name"])
            require(key not in rows, f"Duplicate (split, video_name) in {path}: {key}")
            rows[key] = row
    return rows


def compare_eval_targets(csv_a: Path, csv_b: Path) -> dict[str, Any]:
    rows_a = read_h5_metrics_csv(csv_a)
    rows_b = read_h5_metrics_csv(csv_b)
    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    require(
        not only_a and not only_b,
        f"Evaluated (split, video_name) sets differ: only_in_A={only_a[:5]}, only_in_B={only_b[:5]}",
    )
    mismatches: list[str] = []
    for key in sorted(rows_a):
        row_a, row_b = rows_a[key], rows_b[key]
        for field in POINT_SET_FIELDS:
            if row_a.get(field) != row_b.get(field):
                mismatches.append(f"{key} {field}: {row_a.get(field)} != {row_b.get(field)}")
    require(
        not mismatches,
        f"Evaluation target point sets differ between W-A/W-B despite identical label policy/teacher: {mismatches[:10]}",
    )
    return {"videos_compared": len(rows_a), "fields_checked": list(POINT_SET_FIELDS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-13: config-parity and completion checks between the W-A (control) and "
        "W-B (moderate fixed weight) class-weight ablation training runs."
    )
    parser.add_argument("--run_a_dir", required=True, help="W-A (control) train_stage5.py output_dir")
    parser.add_argument("--run_b_dir", required=True, help="W-B train_stage5.py output_dir")
    parser.add_argument("--expected_weight_a", default="0.05963856,1.94036150")
    parser.add_argument("--expected_weight_b", default="0.5,1.5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--checkpoint_filename", default="last.pt", help="Primary checkpoint compared across arms")
    parser.add_argument("--skip_checkpoint_hash", action="store_true", help="Skip hashing --checkpoint of each config.json")
    parser.add_argument("--eval_csv_a", default=None, help="Optional W-A evaluate_stage5.py h5_metrics.csv (Step F7 item 8)")
    parser.add_argument("--eval_csv_b", default=None, help="Optional W-B evaluate_stage5.py h5_metrics.csv (Step F7 item 8)")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir_a = Path(args.run_a_dir)
    run_dir_b = Path(args.run_b_dir)
    expected_weight_a = parse_weight_list(args.expected_weight_a) if args.expected_weight_a else None
    expected_weight_b = parse_weight_list(args.expected_weight_b) if args.expected_weight_b else None

    config_a = load_json(run_dir_a / "config.json")
    config_b = load_json(run_dir_b / "config.json")

    config_diffs = compare_config(config_a, config_b)
    require(
        not config_diffs,
        "Unexpected config differences beyond class weight/run metadata "
        f"(request doc 7.2 item 4): {config_diffs}",
    )

    train_files_count = check_file_lists(run_dir_a, run_dir_b, name="train_files.txt")
    val_files_count = check_file_lists(run_dir_a, run_dir_b, name="val_files.txt")

    weight_a = check_resolved_weight(config_a, expected=expected_weight_a, label="W-A")
    weight_b = check_resolved_weight(config_b, expected=expected_weight_b, label="W-B")
    require(
        weight_a["resolved"] != weight_b["resolved"],
        "W-A and W-B resolved to the same class weight -- ablation has no effective difference",
    )

    history_a = check_history(run_dir_a, expected_epochs=args.epochs, label="W-A")
    history_b = check_history(run_dir_b, expected_epochs=args.epochs, label="W-B")

    checkpoint_a = check_checkpoint_present(run_dir_a, filename=args.checkpoint_filename, label="W-A")
    checkpoint_b = check_checkpoint_present(run_dir_b, filename=args.checkpoint_filename, label="W-B")

    init_checkpoint_report: dict[str, Any] = {"path_a": config_a.get("checkpoint"), "path_b": config_b.get("checkpoint")}
    require(
        config_a.get("checkpoint") == config_b.get("checkpoint"),
        f"Initialization checkpoint path differs: {config_a.get('checkpoint')} != {config_b.get('checkpoint')}",
    )
    if not args.skip_checkpoint_hash and config_a.get("checkpoint"):
        init_path = Path(config_a["checkpoint"])
        if init_path.is_file():
            init_checkpoint_report["sha256"] = sha256_file(init_path)
        else:
            init_checkpoint_report["sha256"] = None
            init_checkpoint_report["hash_skipped_reason"] = f"file not found on this host: {init_path}"

    eval_target_report: dict[str, Any] | None = None
    if args.eval_csv_a and args.eval_csv_b:
        eval_target_report = compare_eval_targets(Path(args.eval_csv_a), Path(args.eval_csv_b))

    summary = {
        "status": "passed",
        "run_a_dir": str(run_dir_a),
        "run_b_dir": str(run_dir_b),
        "config_diffs": config_diffs,
        "train_files_count": train_files_count,
        "val_files_count": val_files_count,
        "weight_a": weight_a,
        "weight_b": weight_b,
        "history_a": history_a,
        "history_b": history_b,
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "init_checkpoint": init_checkpoint_report,
        "eval_target_parity": eval_target_report,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 class-weight ablation config-parity check passed.")
    print(f"train_files.txt / val_files.txt: {train_files_count} / {val_files_count} lines, identical")
    print(f"W-A resolved weight: {weight_a['resolved']} (mode={weight_a['mode']})")
    print(f"W-B resolved weight: {weight_b['resolved']} (mode={weight_b['mode']})")
    print(f"W-A epochs recorded: {history_a['epochs_recorded']}, W-B epochs recorded: {history_b['epochs_recorded']}")
    if eval_target_report is not None:
        print(f"evaluation target parity: {eval_target_report['videos_compared']} videos compared, fields identical")
    if args.output_json:
        print(f"output: {args.output_json}")


if __name__ == "__main__":
    main()
