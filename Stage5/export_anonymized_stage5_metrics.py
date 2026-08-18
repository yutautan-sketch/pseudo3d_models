from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


VIDEO_ID_PATTERN = re.compile(r"\d{8}_\d{6}_\d+")
PATH_KEY_TOKENS = ("path", "dir", "root", "file", "checkpoint")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def video_token_from_path(value: str) -> str | None:
    match = VIDEO_ID_PATTERN.search(value)
    return match.group(0) if match else None


def ordered_videos_from_paths(
    paths: Iterable[str],
    *,
    path_to_video: dict[str, str],
) -> list[str]:
    videos: list[str] = []
    for value in paths:
        video = path_to_video.get(str(value)) or video_token_from_path(str(value))
        if video and video not in videos:
            videos.append(video)
    return videos


def build_aliases(
    h5_rows: list[dict[str, str]],
    selection_summary: dict[str, Any],
) -> tuple[dict[tuple[str, str], str], list[dict[str, str]]]:
    videos_by_split: dict[str, list[str]] = {"train_sanity": [], "validation": []}
    h5_path_by_key: dict[tuple[str, str], str] = {}
    path_to_video: dict[str, str] = {}
    for row in h5_rows:
        split = row.get("split", "")
        video = row.get("video_name", "")
        h5_path = row.get("h5_path", "")
        if split not in videos_by_split or not video:
            raise ValueError(f"Unexpected split/video_name in H5 metrics row: {row}")
        if video not in videos_by_split[split]:
            videos_by_split[split].append(video)
        h5_path_by_key[(split, video)] = h5_path
        if h5_path:
            path_to_video[h5_path] = video

    fixed_videos = [str(value) for value in selection_summary.get("fixed_train_videos", [])]
    selected_train_order = ordered_videos_from_paths(
        selection_summary.get("selected_train_files", []),
        path_to_video=path_to_video,
    )
    validation_order = ordered_videos_from_paths(
        selection_summary.get("validation_files", []),
        path_to_video=path_to_video,
    )

    train_videos = videos_by_split["train_sanity"]
    selected_train_order.extend(video for video in train_videos if video not in selected_train_order)
    validation_order.extend(
        video for video in videos_by_split["validation"] if video not in validation_order
    )

    aliases: dict[tuple[str, str], str] = {}
    private_rows: list[dict[str, str]] = []
    fixed_index = 0
    random_index = 0
    for video in selected_train_order:
        if video not in train_videos:
            continue
        if video in fixed_videos:
            fixed_index += 1
            alias = f"train_sanity_fixed_{fixed_index:03d}"
            role = "fixed"
        else:
            random_index += 1
            alias = f"train_sanity_random_{random_index:03d}"
            role = "random"
        aliases[("train_sanity", video)] = alias
        private_rows.append(
            {
                "anonymous_id": alias,
                "split": "train_sanity",
                "selection_role": role,
                "original_video_name": video,
                "original_h5_path": h5_path_by_key.get(("train_sanity", video), ""),
            }
        )

    validation_index = 0
    for video in validation_order:
        if video not in videos_by_split["validation"]:
            continue
        validation_index += 1
        alias = f"validation_{validation_index:03d}"
        aliases[("validation", video)] = alias
        private_rows.append(
            {
                "anonymous_id": alias,
                "split": "validation",
                "selection_role": "validation",
                "original_video_name": video,
                "original_h5_path": h5_path_by_key.get(("validation", video), ""),
            }
        )

    expected = {
        (split, video)
        for split, videos in videos_by_split.items()
        for video in videos
    }
    missing = sorted(expected - set(aliases))
    if missing:
        raise ValueError(f"Failed to assign anonymous IDs: {missing}")
    return aliases, private_rows


def anonymize_metric_rows(
    rows: list[dict[str, str]],
    aliases: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    anonymized: list[dict[str, str]] = []
    for row in rows:
        split = row.get("split", "")
        video = row.get("video_name", "")
        alias = aliases.get((split, video))
        if alias is None:
            raise ValueError(f"No anonymous ID for split={split!r}, video={video!r}")
        output = dict(row)
        output["video_name"] = alias
        if "h5_path" in output:
            output["h5_path"] = f"{alias}.h5"
        if "checkpoint" in output:
            output["checkpoint"] = Path(output["checkpoint"]).name
        anonymized.append(output)
    return anonymized


def split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == split]


def sanitize_json_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_json_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json_value(item, key=key) for item in value]
    if isinstance(value, str):
        lower_key = key.lower()
        if value.startswith("/") or any(token in lower_key for token in PATH_KEY_TOKENS):
            return "REDACTED_PATH"
        return VIDEO_ID_PATTERN.sub("REDACTED_VIDEO", value)
    return value


def copy_anonymized_training_files(training_run_dir: Path, output_dir: Path) -> list[str]:
    outputs: list[str] = []
    config_path = training_run_dir / "config.json"
    if config_path.is_file():
        target = output_dir / "training_config_anonymized.json"
        write_json(target, sanitize_json_value(read_json(config_path)))
        outputs.append(str(target.name))

    metrics_path = training_run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        target = output_dir / "training_epoch_metrics.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("r", encoding="utf-8") as source, target.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in {metrics_path} at line {line_number}"
                    ) from exc
                destination.write(
                    json.dumps(sanitize_json_value(record), ensure_ascii=False) + "\n"
                )
        outputs.append(str(target.name))
    return outputs


def assert_share_bundle_anonymous(
    share_output_dir: Path,
    *,
    private_rows: list[dict[str, str]],
) -> None:
    forbidden = {
        value
        for row in private_rows
        for key, value in row.items()
        if key.startswith("original_") and value
    }
    absolute_path_pattern = re.compile(r"(?:^|[\s\"'])/(?:mnt|home|workspace|data)/")
    failures: list[str] = []
    for path in sorted(share_output_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        leaked = [value for value in forbidden if value in text]
        if leaked:
            failures.append(f"{path}: contains original identifier/path")
        if VIDEO_ID_PATTERN.search(text):
            failures.append(f"{path}: contains timestamp-like video ID")
        if absolute_path_pattern.search(text):
            failures.append(f"{path}: contains absolute host path")
    if failures:
        raise RuntimeError("Share bundle anonymization check failed:\n" + "\n".join(failures))


def export(args: argparse.Namespace) -> None:
    evaluation_root = Path(args.evaluation_root)
    share_output_dir = Path(args.share_output_dir)
    private_output_dir = Path(args.private_output_dir)
    share_output_dir.mkdir(parents=True, exist_ok=True)
    private_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = read_csv(evaluation_root / "checkpoint_summary.csv")
    h5_rows = read_csv(evaluation_root / "h5_metrics_all_checkpoints.csv")
    window_rows = read_csv(evaluation_root / "window_metrics_all_checkpoints.csv")
    selection_summary = read_json(evaluation_root / "evaluation_data" / "summary.json")
    comparison_manifest = read_json(evaluation_root / "comparison_manifest.json")

    aliases, private_rows = build_aliases(h5_rows, selection_summary)
    anonymized_h5 = anonymize_metric_rows(h5_rows, aliases)
    anonymized_windows = anonymize_metric_rows(window_rows, aliases)
    anonymized_checkpoints = [
        {
            **row,
            "checkpoint": Path(row.get("checkpoint", "")).name,
        }
        for row in checkpoint_rows
    ]

    share_files: list[str] = []
    for split, directory_name in (
        ("train_sanity", "train_sanity_evaluation"),
        ("validation", "validation_accuracy"),
    ):
        split_dir = share_output_dir / directory_name
        h5_output = split_dir / f"{split}_h5_metrics.csv"
        window_output = split_dir / f"{split}_window_metrics.csv"
        checkpoint_output = split_dir / f"{split}_checkpoint_summary.csv"
        write_csv(h5_output, split_rows(anonymized_h5, split))
        write_csv(window_output, split_rows(anonymized_windows, split))
        write_csv(checkpoint_output, split_rows(anonymized_checkpoints, split))
        share_files.extend(
            str(path.relative_to(share_output_dir))
            for path in (h5_output, window_output, checkpoint_output)
        )

    checkpoints = [Path(str(value)).name for value in comparison_manifest.get("checkpoints", [])]
    public_manifest = {
        "schema": "stage5_anonymized_metrics_v1",
        "checkpoints": checkpoints,
        "train_sanity_ids": [
            row["anonymous_id"] for row in private_rows if row["split"] == "train_sanity"
        ],
        "validation_ids": [
            row["anonymous_id"] for row in private_rows if row["split"] == "validation"
        ],
        "alias_meaning": {
            "train_sanity_fixed_NNN": "fixed training-data sanity sample",
            "train_sanity_random_NNN": "seeded random training-data sanity sample",
            "validation_NNN": "held-out validation sample",
        },
        "included_files": share_files,
    }
    if args.training_run_dir:
        training_outputs = copy_anonymized_training_files(
            Path(args.training_run_dir),
            share_output_dir / "training_history",
        )
        share_files.extend(f"training_history/{name}" for name in training_outputs)

    public_manifest["included_files"] = share_files
    public_manifest_path = share_output_dir / "anonymized_metrics_manifest.json"
    write_json(public_manifest_path, public_manifest)
    share_files.append(str(public_manifest_path.relative_to(share_output_dir)))

    private_map_path = private_output_dir / "video_id_map_DO_NOT_SHARE.csv"
    write_csv(private_map_path, private_rows)
    (private_output_dir / "DO_NOT_SHARE.txt").write_text(
        "This directory contains original video identifiers and H5 paths.\n",
        encoding="utf-8",
    )

    readme = share_output_dir / "SHARE_THIS_DIRECTORY.txt"
    readme.write_text(
        "This directory contains anonymized Stage5 metrics only.\n"
        "train_sanity_evaluation: behavior on selected training samples.\n"
        "validation_accuracy: metrics on held-out validation samples.\n"
        "The private video-ID mapping is intentionally stored outside this directory.\n",
        encoding="utf-8",
    )
    share_files.append(str(readme.relative_to(share_output_dir)))

    assert_share_bundle_anonymous(share_output_dir, private_rows=private_rows)
    report = {
        "status": "passed",
        "schema": "stage5_anonymized_metrics_v1",
        "num_train_sanity_videos": sum(
            row["split"] == "train_sanity" for row in private_rows
        ),
        "num_validation_videos": sum(row["split"] == "validation" for row in private_rows),
        "num_checkpoints": len(checkpoints),
        "share_files": share_files,
        "privacy_checks": {
            "original_identifiers_absent": True,
            "timestamp_like_video_ids_absent": True,
            "absolute_host_paths_absent": True,
        },
    }
    write_json(share_output_dir / "anonymization_report.json", report)

    print("Stage5 anonymized metrics export passed.")
    print(f"share this directory : {share_output_dir}")
    print(f"do not share         : {private_output_dir}")
    print(f"train sanity videos  : {report['num_train_sanity_videos']}")
    print(f"validation videos    : {report['num_validation_videos']}")
    print(f"checkpoints          : {report['num_checkpoints']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export anonymized Stage5 metrics for sharing")
    parser.add_argument("--evaluation_root", required=True)
    parser.add_argument("--share_output_dir", required=True)
    parser.add_argument("--private_output_dir", required=True)
    parser.add_argument("--training_run_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
