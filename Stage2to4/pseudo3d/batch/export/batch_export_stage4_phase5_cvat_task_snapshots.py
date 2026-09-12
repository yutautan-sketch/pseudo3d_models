from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.batch.export import batch_create_stage4_phase5_cvat_tasks as task_batch


SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_ANNOTATION_FORMAT = "Segmentation mask 1.1"


class CvatTaskSnapshotError(RuntimeError):
    """Raised when a CVAT task snapshot violates the fixed Phase 5 contract."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise CvatTaskSnapshotError(f"CSV contains no rows: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CvatTaskSnapshotError(f"JSON root must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _integer(row: Mapping[str, str], key: str, video_name: str) -> int:
    try:
        return int(str(row.get(key, "")).strip())
    except ValueError as exc:
        raise CvatTaskSnapshotError(
            f"Invalid {key} for {video_name}: {row.get(key)!r}"
        ) from exc


def load_task_map(
    review_root: Path, task_map_csv: Path, *, expected_tasks: int
) -> list[dict[str, Any]]:
    rows = _read_csv(task_map_csv)
    if len(rows) != expected_tasks:
        raise CvatTaskSnapshotError(
            f"task_map row count mismatch: {len(rows)} != {expected_tasks}"
        )
    required = {
        "video_name",
        "task_id",
        "task_name",
        "status",
        "image_count",
        "review_bboxes",
    }
    if not required.issubset(rows[0]):
        raise CvatTaskSnapshotError(
            f"task_map lacks required columns: {sorted(required - set(rows[0]))}"
        )

    normalized: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    seen_task_ids: set[int] = set()
    for row in rows:
        video_name = task_batch._safe_video_name(row.get("video_name", ""))
        task_id = _integer(row, "task_id", video_name)
        task_name = str(row.get("task_name", "")).strip()
        if task_id <= 0 or not task_name:
            raise CvatTaskSnapshotError(
                f"Task is not fully registered for {video_name}: id={task_id}, name={task_name!r}"
            )
        if video_name in seen_videos or task_id in seen_task_ids:
            raise CvatTaskSnapshotError(
                f"Duplicate video/task in task_map: video={video_name}, task_id={task_id}"
            )
        seen_videos.add(video_name)
        seen_task_ids.add(task_id)

        images_dir = review_root / "videos" / video_name / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)
        images = sorted(images_dir.glob("*.png"), key=lambda path: path.name)
        image_count = _integer(row, "image_count", video_name)
        if len(images) != image_count:
            raise CvatTaskSnapshotError(
                f"Local image count mismatch for {video_name}: {len(images)} != {image_count}"
            )
        normalized.append(
            {
                "video_name": video_name,
                "task_id": task_id,
                "task_name": task_name,
                "image_count": image_count,
                "review_bboxes": _integer(row, "review_bboxes", video_name),
                "source_task_status": str(row.get("status", "")),
                "expected_frame_names": [path.name for path in images],
            }
        )
    normalized.sort(key=lambda row: row["video_name"])
    return normalized


def _validate_annotation_zip(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise CvatTaskSnapshotError(f"CVAT produced an empty/missing annotation ZIP: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            corrupt = archive.testzip()
            if corrupt is not None:
                raise CvatTaskSnapshotError(
                    f"CVAT annotation ZIP contains a corrupt member {corrupt!r}: {path}"
                )
            basenames = {Path(name).name for name in members}
            if "labelmap.txt" not in basenames or "default.txt" not in basenames:
                raise CvatTaskSnapshotError(
                    f"CVAT annotation ZIP lacks labelmap.txt/default.txt: {path}"
                )
    except zipfile.BadZipFile as exc:
        raise CvatTaskSnapshotError(f"CVAT annotation export is not a valid ZIP: {path}") from exc
    return int(path.stat().st_size), len(members)


def _verify_task(task: Any, row: Mapping[str, Any]) -> None:
    task.fetch()
    if int(task.id) != int(row["task_id"]):
        raise CvatTaskSnapshotError(
            f"Retrieved Task ID mismatch: {task.id} != {row['task_id']}"
        )
    if str(task.name) != str(row["task_name"]):
        raise CvatTaskSnapshotError(
            f"Task #{task.id} name mismatch: {task.name!r} != {row['task_name']!r}"
        )
    if int(task.size) != int(row["image_count"]):
        raise CvatTaskSnapshotError(
            f"Task #{task.id} image count mismatch: {task.size} != {row['image_count']}"
        )
    actual_frames = task_batch._task_frame_names(task)
    if actual_frames != list(row["expected_frame_names"]):
        raise CvatTaskSnapshotError(
            f"Task #{task.id} frame names/order differ from the transferred package"
        )
    label_names = {str(label.name) for label in task.get_labels()}
    if label_names != {"femur"}:
        raise CvatTaskSnapshotError(
            f"Task #{task.id} must expose only the 'femur' label: {sorted(label_names)}"
        )


def _new_state(args: argparse.Namespace, task_map_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "in_progress",
        "server_host": args.server_host,
        "organization_slug": args.organization_slug,
        "annotation_format": args.annotation_format,
        "expected_tasks": int(args.expected_tasks),
        "task_map_sha256": task_map_sha256,
        "created_utc": task_batch._utc_now(),
        "updated_utc": task_batch._utc_now(),
        "tasks": {},
    }


def _load_state(
    path: Path, args: argparse.Namespace, task_map_sha256: str
) -> dict[str, Any]:
    if not path.exists():
        return _new_state(args, task_map_sha256)
    state = _read_json(path)
    expected = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "server_host": args.server_host,
        "organization_slug": args.organization_slug,
        "annotation_format": args.annotation_format,
        "expected_tasks": int(args.expected_tasks),
        "task_map_sha256": task_map_sha256,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise CvatTaskSnapshotError(
                f"Existing export state differs for {key}: {state.get(key)!r} != {value!r}"
            )
    if not isinstance(state.get("tasks"), dict):
        raise CvatTaskSnapshotError("Existing export state tasks must be an object")
    return state


def _snapshot_paths(output_root: Path, row: Mapping[str, Any]) -> tuple[Path, Path]:
    video_name = str(row["video_name"])
    task_id = int(row["task_id"])
    annotations = (
        output_root
        / "annotations"
        / video_name
        / f"{video_name}__task{task_id}__reviewed_segmentation_mask_1_1.zip"
    )
    backup = (
        output_root
        / "task_backups"
        / video_name
        / f"{video_name}__task{task_id}__post_review_backup.zip"
    )
    return annotations, backup


def _verify_recorded_file(path: Path, sha256: str, kind: str) -> None:
    if not path.is_file() or not sha256 or task_batch._file_sha256(path) != sha256:
        raise CvatTaskSnapshotError(f"Recorded {kind} is missing or changed: {path}")


def _write_outputs(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    state: dict[str, Any],
) -> None:
    state["updated_utc"] = task_batch._utc_now()
    _atomic_write_json(output_root / "export_state.json", state)
    records = state["tasks"]
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        record = dict(records.get(str(row["video_name"]), {}))
        manifest_rows.append(
            {
                "video_name": row["video_name"],
                "task_id": row["task_id"],
                "task_name": row["task_name"],
                "status": record.get("status", "pending"),
                "image_count": row["image_count"],
                "review_bboxes": row["review_bboxes"],
                "annotation_zip": record.get("annotation_zip", ""),
                "annotation_bytes": record.get("annotation_bytes", ""),
                "annotation_sha256": record.get("annotation_sha256", ""),
                "backup_zip": record.get("backup_zip", ""),
                "backup_bytes": record.get("backup_bytes", ""),
                "backup_sha256": record.get("backup_sha256", ""),
            }
        )
    _atomic_write_csv(
        output_root / "export_manifest.csv",
        tuple(manifest_rows[0]),
        manifest_rows,
    )


def run(args: argparse.Namespace, *, client: Any | None = None) -> None:
    review_root = Path(args.review_root).resolve()
    task_map_csv = Path(args.task_map_csv).resolve()
    output_root = Path(args.output_root).resolve()
    rows = load_task_map(
        review_root, task_map_csv, expected_tasks=int(args.expected_tasks)
    )
    if args.only_video:
        selected = [row for row in rows if row["video_name"] == args.only_video]
        if not selected:
            raise CvatTaskSnapshotError(
                f"--only_video is absent from task_map: {args.only_video}"
            )
    else:
        selected = list(rows)
    if args.max_tasks:
        selected = selected[: int(args.max_tasks)]

    print("Stage 4 Phase 5 CVAT corrected-task snapshot batch")
    print(f"  review_root       : {review_root}")
    print(f"  task_map_csv      : {task_map_csv}")
    print(f"  output_root       : {output_root}")
    print(f"  server_host       : {args.server_host}")
    print(f"  annotation_format : {args.annotation_format}")
    print(f"  registered_tasks  : {len(rows)}")
    print(f"  selected_now      : {len(selected)}")
    print(f"  apply             : {bool(args.apply)}")

    task_map_sha256 = task_batch._file_sha256(task_map_csv)
    state_path = output_root / "export_state.json"
    if args.apply:
        state = _load_state(state_path, args, task_map_sha256)
    else:
        state = _new_state(args, task_map_sha256)

    own_client = client is None
    if own_client:
        client = task_batch._connect(args)
    if args.organization_slug:
        client.organization_slug = args.organization_slug

    try:
        records: dict[str, Any] = state["tasks"]
        completed_now = 0
        for index, row in enumerate(selected, start=1):
            video_name = str(row["video_name"])
            task = client.tasks.retrieve(int(row["task_id"]))
            _verify_task(task, row)
            annotation_path, backup_path = _snapshot_paths(output_root, row)
            record = dict(records.get(video_name, {}))
            if record:
                if int(record.get("task_id", -1)) != int(row["task_id"]):
                    raise CvatTaskSnapshotError(
                        f"Stored task ID differs for {video_name}"
                    )
                if str(record.get("status")) == "complete":
                    _verify_recorded_file(
                        annotation_path, str(record.get("annotation_sha256", "")), "annotation ZIP"
                    )
                    _verify_recorded_file(
                        backup_path, str(record.get("backup_sha256", "")), "task backup"
                    )
                    _validate_annotation_zip(annotation_path)
                    task_batch._validate_task_backup(backup_path)
                    print(f"  [{index}/{len(selected)}] [SKIP complete] {video_name}")
                    continue

            if not args.apply:
                print(
                    f"  [{index}/{len(selected)}] [PLAN] {video_name}: "
                    f"task=#{row['task_id']}, frames={row['image_count']}"
                )
                continue

            record.setdefault("video_name", video_name)
            record.setdefault("task_id", int(row["task_id"]))
            record.setdefault("task_name", str(row["task_name"]))
            status = str(record.get("status", "pending"))
            if status == "annotation_exported":
                _verify_recorded_file(
                    annotation_path, str(record.get("annotation_sha256", "")), "annotation ZIP"
                )
                _validate_annotation_zip(annotation_path)
            else:
                if annotation_path.exists():
                    raise FileExistsError(
                        f"Untracked annotation snapshot already exists: {annotation_path}"
                    )
                annotation_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = annotation_path.with_name(
                    f".{annotation_path.stem}.partial.zip"
                )
                if temporary.exists():
                    temporary.unlink()
                print(f"  [{index}/{len(selected)}] exporting annotations: {video_name}")
                task.export_dataset(
                    args.annotation_format, temporary, include_images=False
                )
                annotation_bytes, annotation_members = _validate_annotation_zip(temporary)
                os.replace(temporary, annotation_path)
                record.update(
                    {
                        "status": "annotation_exported",
                        "annotation_zip": str(annotation_path),
                        "annotation_bytes": annotation_bytes,
                        "annotation_members": annotation_members,
                        "annotation_sha256": task_batch._file_sha256(annotation_path),
                        "updated_utc": task_batch._utc_now(),
                    }
                )
                records[video_name] = record
                _write_outputs(output_root, rows, state)

            if backup_path.exists():
                raise FileExistsError(
                    f"Untracked post-review backup already exists: {backup_path}"
                )
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = backup_path.with_name(f".{backup_path.stem}.partial.zip")
            if temporary.exists():
                temporary.unlink()
            print(f"  [{index}/{len(selected)}] downloading task backup: {video_name}")
            task.download_backup(temporary, lightweight=False)
            backup_bytes, backup_members = task_batch._validate_task_backup(temporary)
            os.replace(temporary, backup_path)
            record.update(
                {
                    "status": "complete",
                    "backup_zip": str(backup_path),
                    "backup_bytes": backup_bytes,
                    "backup_members": backup_members,
                    "backup_sha256": task_batch._file_sha256(backup_path),
                    "completed_utc": task_batch._utc_now(),
                    "updated_utc": task_batch._utc_now(),
                }
            )
            records[video_name] = record
            completed_now += 1
            _write_outputs(output_root, rows, state)
    finally:
        if own_client:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    if not args.apply:
        print("Stage 4 Phase 5 CVAT snapshot dry-run passed; no files were written.")
        return

    completed_total = sum(
        str(record.get("status")) == "complete"
        for record in state["tasks"].values()
    )
    state["completed_tasks"] = completed_total
    state["status"] = "complete" if completed_total == len(rows) else "in_progress"
    _write_outputs(output_root, rows, state)
    summary = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": state["status"],
        "registered_tasks": len(rows),
        "completed_tasks": completed_total,
        "annotation_format": args.annotation_format,
        "task_map_sha256": task_map_sha256,
        "export_state": "export_state.json",
        "export_manifest": "export_manifest.csv",
        "updated_utc": state["updated_utc"],
    }
    _atomic_write_json(output_root / "export_summary.json", summary)
    print("Stage 4 Phase 5 CVAT corrected-task snapshot batch finished.")
    print(f"  completed_this_run : {completed_now}")
    print(f"  completed_total    : {completed_total}/{len(rows)}")
    print(f"  export_state       : {output_root / 'export_state.json'}")
    print(f"  export_manifest    : {output_root / 'export_manifest.csv'}")
    print(f"  export_summary     : {output_root / 'export_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export corrected Segmentation mask 1.1 annotations and a post-review "
            "backup from every registered Stage 4 Phase 5 CVAT task."
        )
    )
    parser.add_argument("--review_root", type=Path, required=True)
    parser.add_argument("--server_host", default="http://localhost:8080")
    parser.add_argument("--username", default="")
    parser.add_argument("--organization_slug", default="")
    parser.add_argument("--task_map_csv", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--annotation_format", default=DEFAULT_ANNOTATION_FORMAT)
    parser.add_argument("--expected_tasks", type=int, default=60)
    parser.add_argument("--only_video", default="")
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write annotation snapshots and backups; otherwise perform a read-only preflight.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.review_root = Path(args.review_root).resolve()
    args.task_map_csv = Path(
        args.task_map_csv
        or args.review_root / "task_management" / "task_map.csv"
    ).resolve()
    args.output_root = Path(
        args.output_root
        or args.review_root / "cvat_exports_before_textfree_v1"
    ).resolve()
    if args.expected_tasks <= 0:
        raise SystemExit("--expected_tasks must be > 0")
    if args.max_tasks < 0:
        raise SystemExit("--max_tasks must be >= 0")
    if args.only_video:
        args.only_video = task_batch._safe_video_name(args.only_video)
    if not args.annotation_format or any(
        character in args.annotation_format for character in ("\r", "\n", "\0")
    ):
        raise SystemExit("--annotation_format must be non-empty and printable")
    args.server_host = str(args.server_host).rstrip("/")
    parsed = urlparse(args.server_host)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise SystemExit(
            "--server_host must point explicitly to localhost/loopback; external CVAT hosts are forbidden"
        )
    try:
        run(args)
    except Exception as exc:
        raise SystemExit(
            f"Stage 4 Phase 5 CVAT task snapshot batch failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
