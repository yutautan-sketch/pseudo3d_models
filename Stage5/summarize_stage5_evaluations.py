from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(root: Path, evaluation_dirs: list[Path] | None = None) -> None:
    if evaluation_dirs:
        summary_paths = [path / "summary.json" for path in evaluation_dirs]
    else:
        summary_paths = sorted(root.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No checkpoint summary.json files found under {root}")
    missing = [path for path in summary_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint evaluation summary: {missing[0]}")

    summaries: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    h5_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    reference_selection: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        selection = (
            tuple(summary["selected_train_files"]),
            tuple(summary["validation_files"]),
        )
        if reference_selection is None:
            reference_selection = selection
        elif selection != reference_selection:
            raise ValueError(
                f"Evaluation file selection differs in {summary_path}; "
                "checkpoint comparison is invalid"
            )

        checkpoint_name = Path(summary["checkpoint"]).name
        for split, metrics in summary["splits"].items():
            checkpoint_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "checkpoint_epoch": summary.get("checkpoint_epoch"),
                    "split": split,
                    **metrics,
                }
            )
        h5_rows.extend(read_csv(summary_path.parent / "h5_metrics.csv"))
        window_rows.extend(read_csv(summary_path.parent / "window_metrics.csv"))

    write_csv(root / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(root / "h5_metrics_all_checkpoints.csv", h5_rows)
    write_csv(root / "window_metrics_all_checkpoints.csv", window_rows)
    comparison = {
        "num_checkpoints": len(summaries),
        "checkpoints": [summary["checkpoint"] for summary in summaries],
        "selected_train_files": list(reference_selection[0]) if reference_selection else [],
        "validation_files": list(reference_selection[1]) if reference_selection else [],
        "outputs": {
            "checkpoint_summary": str(root / "checkpoint_summary.csv"),
            "h5_metrics": str(root / "h5_metrics_all_checkpoints.csv"),
            "window_metrics": str(root / "window_metrics_all_checkpoints.csv"),
        },
    }
    with (root / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)

    print(f"Summarized {len(summaries)} checkpoint evaluations: {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Stage5 checkpoint evaluation outputs")
    parser.add_argument("--evaluation_root", required=True)
    parser.add_argument(
        "--evaluation_dir",
        action="append",
        default=None,
        help="Checkpoint evaluation directory to include; repeat for multiple checkpoints",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluation_dirs = [Path(path) for path in args.evaluation_dir] if args.evaluation_dir else None
    summarize(Path(args.evaluation_root), evaluation_dirs)
