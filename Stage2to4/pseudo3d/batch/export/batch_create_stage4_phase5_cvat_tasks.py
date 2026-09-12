from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root_text = str(REPO_ROOT)
# A copied review workspace can already contain REPO_ROOT later in sys.path.
# Always move it to the front so an unrelated installed ``pseudo3d`` namespace
# cannot shadow this checkout when the script is launched by file path.
sys.path[:] = [entry for entry in sys.path if entry != repo_root_text]
sys.path.insert(0, repo_root_text)

from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    validate_cvat_segmentation_mask_zip,
)


TASK_STATE_SCHEMA_VERSION = 1
TASK_NAME_PREFIX = "stage4_phase5__"
FULLVIDEO_TASK_NAME_PREFIX = "stage4_phase5_fullvideo__"
ANNOTATION_FORMAT = "Segmentation mask 1.1"


class CvatTaskBatchError(RuntimeError):
    """Raised when a Phase 5 CVAT task batch violates its local/server contract."""


@dataclass(frozen=True)
class ReviewPackage:
    video_name: str
    root: Path
    images: tuple[Path, ...]
    annotation_zip: Path
    image_count: int
    annotation_zip_sha256: str
    review_frames: int
    review_bboxes: int
    actionable_bboxes: int
    context_only_frames: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CvatTaskBatchError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise CvatTaskBatchError(f"CSV contains no rows: {path}")
    return rows


def _safe_relative_path(value: str, name: str) -> Path:
    path = Path(str(value))
    if not str(value).strip() or path.is_absolute() or ".." in path.parts:
        raise CvatTaskBatchError(f"Unsafe {name}: {value!r}")
    return path


def _safe_video_name(value: str) -> str:
    value = str(value).strip()
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
        or "\n" in value
    ):
        raise CvatTaskBatchError(f"Unsafe video_name: {value!r}")
    return value


def _is_ignored_platform_metadata(path: Path) -> bool:
    """Ignore metadata files added by Finder without weakening image validation."""
    return path.name == ".DS_Store" or path.name.startswith("._")


def load_review_packages(
    review_root: Path,
    *,
    expected_video_packages: int,
    expected_export_mode: str | None = None,
) -> list[ReviewPackage]:
    review_root = Path(review_root).resolve()
    partition = _read_json(review_root / "partition_summary.json")
    if str(partition.get("status", "")) != "ok" or int(
        partition.get("failure_rows", -1)
    ) != 0:
        raise CvatTaskBatchError("Phase 5 review export is not complete and failure-free")
    stored_count = int(partition.get("video_packages", -1))
    if stored_count != expected_video_packages:
        raise CvatTaskBatchError(
            "partition_summary video count mismatch: "
            f"stored={stored_count}, expected={expected_video_packages}"
        )

    progress_filename = str(partition.get("progress_csv", "progress.csv"))
    if progress_filename not in {"progress.csv", "video_progress.csv"}:
        raise CvatTaskBatchError(
            f"Unsupported partition progress CSV: {progress_filename!r}"
        )
    if expected_export_mode == "full_video" and progress_filename != "video_progress.csv":
        raise CvatTaskBatchError(
            "Full-video standalone mode requires partition progress_csv=video_progress.csv"
        )
    progress_rows = _read_csv(review_root / progress_filename)
    if len(progress_rows) != stored_count:
        raise CvatTaskBatchError(
            f"progress.csv row count mismatch: rows={len(progress_rows)}, stored={stored_count}"
        )

    packages: list[ReviewPackage] = []
    seen_videos: set[str] = set()
    for row in progress_rows:
        video_name = _safe_video_name(row.get("video_name", ""))
        if video_name in seen_videos:
            raise CvatTaskBatchError(f"Duplicate progress video_name: {video_name}")
        seen_videos.add(video_name)
        package_relative = _safe_relative_path(row.get("package_root", ""), "package_root")
        package_root = (review_root / package_relative).resolve()
        try:
            package_root.relative_to(review_root)
        except ValueError as exc:
            raise CvatTaskBatchError(
                f"package_root escapes review_root: {package_relative}"
            ) from exc
        if package_root.name != video_name or not package_root.is_dir():
            raise CvatTaskBatchError(
                f"Video package directory mismatch/not found: {video_name}: {package_root}"
            )

        images_dir = package_root / "images"
        images = tuple(sorted(images_dir.glob("*.png"), key=lambda path: path.name))
        unexpected_images = (
            sorted(
                path.name
                for path in images_dir.iterdir()
                if not _is_ignored_platform_metadata(path)
                and (not path.is_file() or path.suffix != ".png")
            )
            if images_dir.is_dir()
            else []
        )
        if not images_dir.is_dir() or not images or unexpected_images:
            raise CvatTaskBatchError(
                f"Invalid images directory for {video_name}: images={len(images)}, "
                f"unexpected={unexpected_images}"
            )
        if len({path.name for path in images}) != len(images):
            raise CvatTaskBatchError(f"Duplicate image basename for {video_name}")

        zip_relative = _safe_relative_path(row.get("cvat_import_zip", ""), "cvat_import_zip")
        annotation_zip = (review_root / zip_relative).resolve()
        try:
            annotation_zip.relative_to(package_root)
        except ValueError as exc:
            raise CvatTaskBatchError(
                f"CVAT ZIP is outside its video package: {video_name}: {annotation_zip}"
            ) from exc
        if not annotation_zip.is_file():
            raise FileNotFoundError(annotation_zip)

        summary = _read_json(package_root / "export_summary.json")
        if (
            str(summary.get("status", "")) != "ok"
            or str(summary.get("partition_video_name", "")) != video_name
            or int(summary.get("failure_rows", -1)) != 0
        ):
            raise CvatTaskBatchError(f"Invalid video export summary: {video_name}")
        if expected_export_mode is not None and str(
            summary.get("export_mode", "")
        ) != expected_export_mode:
            raise CvatTaskBatchError(
                f"Unexpected export mode for {video_name}: "
                f"{summary.get('export_mode')!r} != {expected_export_mode!r}"
            )
        review_frames = int(row.get("review_frames", -1))
        if review_frames != len(images) or int(summary.get("review_frames", -1)) != len(images):
            raise CvatTaskBatchError(
                f"Image/review-frame count mismatch for {video_name}: "
                f"images={len(images)}, progress={review_frames}, "
                f"summary={summary.get('review_frames')}"
            )
        zip_sha256 = _file_sha256(annotation_zip)
        if zip_sha256 != str(summary.get("cvat_import_zip_sha256", "")):
            raise CvatTaskBatchError(f"CVAT ZIP checksum mismatch: {video_name}")

        packages.append(
            ReviewPackage(
                video_name=video_name,
                root=package_root,
                images=images,
                annotation_zip=annotation_zip,
                image_count=len(images),
                annotation_zip_sha256=zip_sha256,
                review_frames=review_frames,
                review_bboxes=int(row.get("review_bboxes", -1)),
                actionable_bboxes=int(row.get("actionable_bboxes", -1)),
                context_only_frames=int(row.get("context_only_frames", -1)),
            )
        )
    packages.sort(key=lambda item: item.video_name)
    return packages


def _load_excluded_videos(path: Path | None, available: set[str]) -> tuple[set[str], str]:
    if path is None:
        return set(), ""
    path = Path(path).resolve()
    rows = _read_csv(path)
    if "video_name" not in rows[0]:
        raise CvatTaskBatchError("Exclusion manifest lacks video_name")
    excluded = {_safe_video_name(row.get("video_name", "")) for row in rows}
    if len(excluded) != len(rows):
        raise CvatTaskBatchError("Exclusion manifest contains duplicate videos")
    unknown = sorted(excluded - available)
    if unknown:
        raise CvatTaskBatchError(f"Exclusion manifest videos are absent from package: {unknown}")
    return excluded, _file_sha256(path)


def _apply_annotation_snapshot_overrides(
    packages: Sequence[ReviewPackage], snapshot_root: Path | None
) -> tuple[list[ReviewPackage], str]:
    if snapshot_root is None:
        return list(packages), ""
    snapshot_root = Path(snapshot_root).resolve()
    summary = _read_json(snapshot_root / "export_summary.json")
    manifest_path = snapshot_root / "export_manifest.csv"
    rows = _read_csv(manifest_path)
    if (
        str(summary.get("status", "")) != "complete"
        or int(summary.get("completed_tasks", -1)) != len(packages)
        or int(summary.get("registered_tasks", -1)) != len(packages)
        or len(rows) != len(packages)
    ):
        raise CvatTaskBatchError("Corrected snapshot export is not complete for every package")
    by_video: dict[str, Mapping[str, str]] = {}
    for row in rows:
        video_name = _safe_video_name(row.get("video_name", ""))
        if video_name in by_video or str(row.get("status", "")) != "complete":
            raise CvatTaskBatchError(
                f"Invalid/duplicate corrected snapshot manifest row: {video_name}"
            )
        by_video[video_name] = row
    expected = {package.video_name for package in packages}
    if set(by_video) != expected:
        raise CvatTaskBatchError("Corrected snapshot/package video sets differ")

    overridden: list[ReviewPackage] = []
    for package in packages:
        row = by_video[package.video_name]
        annotation_zip = Path(str(row.get("annotation_zip", ""))).resolve()
        try:
            annotation_zip.relative_to(snapshot_root)
        except ValueError as exc:
            raise CvatTaskBatchError(
                f"Corrected snapshot path escapes snapshot_root: {annotation_zip}"
            ) from exc
        expected_sha = str(row.get("annotation_sha256", ""))
        if not annotation_zip.is_file() or not expected_sha or _file_sha256(annotation_zip) != expected_sha:
            raise CvatTaskBatchError(
                f"Corrected snapshot is missing or changed: {package.video_name}"
            )
        # A CVAT re-export omits both PNG members when a reviewer clears a
        # frame completely.  For corrected snapshots, any known image stem may
        # therefore be an intentional empty mask.  The validator still rejects
        # partial pairs, unknown/default.txt stems, extra members, bad shapes,
        # labels, and undeclared colors.
        allowed_missing = sorted(image.stem for image in package.images)
        validation = validate_cvat_segmentation_mask_zip(
            annotation_zip,
            label_name="femur",
            images_dir=package.root / "images",
            allowed_missing_mask_stems=allowed_missing,
        )
        if validation.images != package.image_count:
            raise CvatTaskBatchError(
                f"Corrected snapshot frame count mismatch: {package.video_name}"
            )
        overridden.append(
            replace(
                package,
                annotation_zip=annotation_zip,
                annotation_zip_sha256=expected_sha,
            )
        )
    return overridden, _file_sha256(manifest_path)


def verify_fullvideo_transfer_package(
    review_root: Path,
    *,
    allow_mutable_progress: bool,
    expected_video_packages: int,
    expected_review_bboxes: int,
) -> dict[str, Any]:
    """Verify the Step 4 transfer manifest before touching local CVAT state."""
    review_root = Path(review_root).resolve()
    manifest_path = review_root / "mac_transfer_manifest.csv"
    transfer_summary = _read_json(review_root / "mac_transfer_summary.json")
    validation_summary = _read_json(review_root / "package_validation_summary.json")
    if str(transfer_summary.get("status", "")) != "ok":
        raise CvatTaskBatchError("Mac transfer summary is not successful")
    if str(validation_summary.get("status", "")) != "ok" or int(
        validation_summary.get("failure_rows", -1)
    ) != 0:
        raise CvatTaskBatchError("Step 4 package validation summary is not successful")
    if int(validation_summary.get("videos_checked", -1)) != int(
        expected_video_packages
    ) or int(validation_summary.get("target_bboxes_checked", -1)) != int(
        expected_review_bboxes
    ):
        raise CvatTaskBatchError(
            "Step 4 package validation counts differ from the fixed Task contract"
        )
    manifest_sha256 = _file_sha256(manifest_path)
    if manifest_sha256 != str(transfer_summary.get("manifest_sha256", "")):
        raise CvatTaskBatchError("Mac transfer manifest SHA-256 mismatch")

    rows = _read_csv(manifest_path)
    if len(rows) != int(transfer_summary.get("transfer_files", -1)):
        raise CvatTaskBatchError("Mac transfer manifest row count mismatch")
    seen: set[str] = set()
    original_progress_sha256 = ""
    original_transfer_bytes = 0
    mutable = {"video_progress.csv"} if allow_mutable_progress else set()
    for row in rows:
        relative_text = str(row.get("relative_path", ""))
        relative = _safe_relative_path(relative_text, "transfer relative_path")
        normalized = relative.as_posix()
        if normalized in seen:
            raise CvatTaskBatchError(
                f"Duplicate Mac transfer manifest path: {normalized}"
            )
        seen.add(normalized)
        path = (review_root / relative).resolve()
        try:
            path.relative_to(review_root)
        except ValueError as exc:
            raise CvatTaskBatchError(
                f"Transfer manifest path escapes review root: {normalized}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_bytes = int(row.get("bytes", -1))
        expected_sha256 = str(row.get("sha256", ""))
        original_transfer_bytes += expected_bytes
        if normalized == "video_progress.csv":
            original_progress_sha256 = expected_sha256
        if normalized in mutable:
            continue
        if path.stat().st_size != expected_bytes or _file_sha256(path) != expected_sha256:
            raise CvatTaskBatchError(
                f"Transferred package file differs from Step 4 manifest: {normalized}"
            )
    if original_transfer_bytes != int(transfer_summary.get("transfer_bytes", -1)):
        raise CvatTaskBatchError("Mac transfer byte total mismatch")
    if "video_progress.csv" not in seen or not original_progress_sha256:
        raise CvatTaskBatchError("Mac transfer manifest lacks video_progress.csv")
    if not any(value.startswith("videos/") for value in seen):
        raise CvatTaskBatchError("Mac transfer manifest contains no video package files")
    return {
        "manifest_sha256": manifest_sha256,
        "original_video_progress_sha256": original_progress_sha256,
        "transfer_files": len(rows),
        "transfer_bytes": original_transfer_bytes,
    }


def load_master_review_package(
    review_root: Path, video_packages: Sequence[ReviewPackage]
) -> ReviewPackage:
    """Load the root package that combines every Phase 3 case in one CVAT task."""
    review_root = Path(review_root).resolve()
    images_dir = review_root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    images = tuple(sorted(images_dir.glob("*.png"), key=lambda path: path.name))
    unexpected = sorted(
        path.name
        for path in images_dir.iterdir()
        if not _is_ignored_platform_metadata(path)
        and (not path.is_file() or path.suffix != ".png")
    )
    if not images or unexpected:
        raise CvatTaskBatchError(
            f"Invalid master images directory: images={len(images)}, unexpected={unexpected}"
        )
    expected_names = sorted(
        image.name for package in video_packages for image in package.images
    )
    actual_names = [image.name for image in images]
    if len(expected_names) != len(set(expected_names)):
        raise CvatTaskBatchError("Per-video packages contain duplicate master image names")
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise CvatTaskBatchError(
            f"Master/per-video image membership differs: missing={missing}, extra={extra}"
        )

    annotation_zip = review_root / "cvat" / "annotations_segmentation_mask_1_1.zip"
    if not annotation_zip.is_file():
        raise FileNotFoundError(annotation_zip)
    summary = _read_json(review_root / "export_summary.json")
    if str(summary.get("status", "")) != "ok" or int(summary.get("failure_rows", -1)) != 0:
        raise CvatTaskBatchError("Master review export is not complete and failure-free")
    if int(summary.get("review_frames", -1)) != len(images):
        raise CvatTaskBatchError(
            "Master image/review-frame count mismatch: "
            f"images={len(images)}, summary={summary.get('review_frames')}"
        )
    zip_sha256 = _file_sha256(annotation_zip)
    if zip_sha256 != str(summary.get("cvat_import_zip_sha256", "")):
        raise CvatTaskBatchError("Master CVAT ZIP checksum mismatch")
    return ReviewPackage(
        video_name="__all_phase3_master__",
        root=review_root,
        images=images,
        annotation_zip=annotation_zip,
        image_count=len(images),
        annotation_zip_sha256=zip_sha256,
        review_frames=len(images),
        review_bboxes=int(summary.get("selected_review_bboxes", -1)),
        actionable_bboxes=int(summary.get("actionable_review_bboxes", -1)),
        context_only_frames=int(summary.get("context_only_frames", -1)),
    )


def _empty_state(args: argparse.Namespace, project_id: int | None) -> dict[str, Any]:
    return {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "status": "in_progress",
        "server_host": args.server_host,
        "organization_slug": args.organization_slug,
        "project_id": project_id,
        "reference_task_id": args.reference_task_id,
        "reference_video": args.reference_video,
        "task_name_prefix": args.task_name_prefix,
        "annotation_format": args.annotation_format,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "tasks": {},
    }


def _load_state(
    path: Path, args: argparse.Namespace, project_id: int | None
) -> dict[str, Any]:
    if not path.exists():
        return _empty_state(args, project_id)
    state = dict(_read_json(path))
    expected = {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "server_host": args.server_host,
        "organization_slug": args.organization_slug,
        "project_id": project_id,
        "reference_task_id": args.reference_task_id,
        "reference_video": args.reference_video,
        "task_name_prefix": args.task_name_prefix,
        "annotation_format": args.annotation_format,
    }
    stored_format = str(state.get("annotation_format", ""))
    if stored_format.casefold() == str(args.annotation_format).casefold():
        state["annotation_format"] = args.annotation_format
    for key, value in expected.items():
        if state.get(key) != value:
            raise CvatTaskBatchError(
                f"Existing state differs for {key}: stored={state.get(key)!r}, expected={value!r}"
            )
    if not isinstance(state.get("tasks"), Mapping):
        raise CvatTaskBatchError("Existing state tasks must be an object")
    state["tasks"] = dict(state["tasks"])
    return state


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_task_map(path: Path, packages: Sequence[ReviewPackage], state: Mapping[str, Any]) -> None:
    fieldnames = (
        "video_name",
        "task_id",
        "task_name",
        "task_url",
        "status",
        "image_count",
        "review_bboxes",
        "actionable_bboxes",
        "context_only_frames",
        "annotation_zip_sha256",
        "backup_path",
        "backup_sha256",
    )
    rows_by_video = state.get("tasks", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for package in packages:
            row = dict(rows_by_video.get(package.video_name, {}))
            writer.writerow(
                {
                    "video_name": package.video_name,
                    "task_id": row.get("task_id", ""),
                    "task_name": row.get("task_name", f"{state['task_name_prefix']}{package.video_name}"),
                    "task_url": row.get("task_url", ""),
                    "status": row.get("status", "pending"),
                    "image_count": package.image_count,
                    "review_bboxes": package.review_bboxes,
                    "actionable_bboxes": package.actionable_bboxes,
                    "context_only_frames": package.context_only_frames,
                    "annotation_zip_sha256": package.annotation_zip_sha256,
                    "backup_path": row.get("backup_path", ""),
                    "backup_sha256": row.get("backup_sha256", ""),
                }
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _save_state(
    *,
    state_path: Path,
    task_map_csv: Path,
    packages: Sequence[ReviewPackage],
    state: dict[str, Any],
) -> None:
    state["updated_utc"] = _utc_now()
    _atomic_write_json(state_path, state)
    _write_task_map(task_map_csv, packages, state)


def _task_project_id(task: Any) -> int | None:
    value = getattr(task, "project_id", None)
    return None if value is None else int(value)


def _task_frame_names(task: Any) -> list[str]:
    return [Path(str(frame.name)).name for frame in task.get_frames_info()]


def _verify_task(
    task: Any,
    package: ReviewPackage,
    *,
    expected_project_id: int | None,
    require_name: str | None,
    require_exact_femur_label: bool = False,
) -> None:
    task.fetch()
    if require_name is not None and str(task.name) != require_name:
        raise CvatTaskBatchError(
            f"Task #{task.id} name mismatch: {task.name!r} != {require_name!r}"
        )
    if _task_project_id(task) != expected_project_id:
        raise CvatTaskBatchError(
            f"Task #{task.id} project mismatch: {_task_project_id(task)} != {expected_project_id}"
        )
    if int(task.size) != package.image_count:
        raise CvatTaskBatchError(
            f"Task #{task.id} image count mismatch: {task.size} != {package.image_count}"
        )
    expected_names = [path.name for path in package.images]
    actual_names = _task_frame_names(task)
    if actual_names != expected_names:
        raise CvatTaskBatchError(
            f"Task #{task.id} frame names/order differ from package: "
            f"actual={actual_names}, expected={expected_names}"
        )
    label_names = {str(label.name) for label in task.get_labels()}
    if "femur" not in label_names:
        raise CvatTaskBatchError(
            f"Task #{task.id} does not expose the required 'femur' label: {sorted(label_names)}"
        )
    if require_exact_femur_label and label_names != {"femur"}:
        raise CvatTaskBatchError(
            f"Task #{task.id} must expose only the 'femur' label: {sorted(label_names)}"
        )


def _credentials(args: argparse.Namespace) -> tuple[str | None, str | None]:
    token = os.environ.get("CVAT_ACCESS_TOKEN", "").strip()
    if token:
        return token, None
    username = (args.username or os.environ.get("CVAT_USERNAME", "")).strip()
    if not username:
        username = input("CVAT username: ").strip()
    if not username:
        raise CvatTaskBatchError("CVAT username must not be empty")
    password = os.environ.get("CVAT_PASSWORD")
    if password is None:
        password = getpass.getpass("CVAT password: ")
    if not password:
        raise CvatTaskBatchError("CVAT password must not be empty")
    return username, password


def _import_sdk() -> tuple[Any, Any]:
    try:
        from cvat_sdk import make_client
        from cvat_sdk.core.proxies.tasks import ResourceType
    except ImportError as exc:
        raise CvatTaskBatchError(
            "cvat-sdk is not installed. Install the client version matching CVAT 2.73 "
            "(for this server: python -m pip install 'cvat-sdk==2.73.0')."
        ) from exc
    return make_client, ResourceType


def _connect(args: argparse.Namespace) -> Any:
    make_client, _ = _import_sdk()
    first, password = _credentials(args)
    if password is None:
        return make_client(args.server_host, access_token=first)
    return make_client(args.server_host, credentials=(first, password))


def _task_url(server_host: str, task_id: int) -> str:
    return f"{server_host.rstrip('/')}/tasks/{int(task_id)}"


def _backup_path(package: ReviewPackage, task_name: str) -> Path:
    return package.root / "cvat" / "task_backups" / f"{task_name}__pre_review.zip"


def _state_row(
    *,
    args: argparse.Namespace,
    package: ReviewPackage,
    task: Any,
    status: str,
    reference_task: bool,
) -> dict[str, Any]:
    return {
        "video_name": package.video_name,
        "task_id": int(task.id),
        "task_name": str(task.name),
        "task_url": _task_url(args.server_host, int(task.id)),
        "status": status,
        "reference_task": bool(reference_task),
        "project_id": _task_project_id(task),
        "image_count": package.image_count,
        "annotation_zip": str(package.annotation_zip),
        "annotation_zip_sha256": package.annotation_zip_sha256,
        "backup_path": "",
        "backup_sha256": "",
        "updated_utc": _utc_now(),
    }


def _validate_task_backup(path: Path) -> tuple[int, int]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise CvatTaskBatchError(f"CVAT produced an empty/missing backup: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.namelist()
            if not members or archive.testzip() is not None:
                raise CvatTaskBatchError(f"CVAT backup ZIP is corrupt: {path}")
            basenames = {Path(name).name for name in members if not name.endswith("/")}
            if not ({"task.json", "annotations.json"} & basenames):
                raise CvatTaskBatchError(
                    f"CVAT backup ZIP lacks task/annotation metadata: {path}"
                )
    except zipfile.BadZipFile as exc:
        raise CvatTaskBatchError(f"CVAT backup is not a valid ZIP: {path}") from exc
    return int(path.stat().st_size), len(members)


def _empty_fullvideo_state(
    args: argparse.Namespace, transfer_contract: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "status": "in_progress",
        "task_mode": "full_video_standalone",
        "server_host": args.server_host,
        "organization_slug": args.organization_slug,
        "project_id": None,
        "task_name_prefix": args.task_name_prefix,
        "annotation_format": args.annotation_format,
        "expected_tasks": int(args.active_video_packages),
        "expected_review_bboxes": int(args.active_review_bboxes),
        "package_video_count": int(args.expected_video_packages),
        "package_review_bboxes": int(args.expected_review_bboxes),
        "excluded_videos": list(args.excluded_videos),
        "exclusion_manifest_sha256": args.exclusion_manifest_sha256,
        "annotation_snapshot_manifest_sha256": args.annotation_snapshot_manifest_sha256,
        "transfer_manifest_sha256": transfer_contract["manifest_sha256"],
        "original_video_progress_sha256": transfer_contract[
            "original_video_progress_sha256"
        ],
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "tasks": {},
    }


def _load_fullvideo_state(
    path: Path,
    args: argparse.Namespace,
    transfer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return _empty_fullvideo_state(args, transfer_contract)
    state = dict(_read_json(path))
    expected = {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "task_mode": "full_video_standalone",
        "server_host": args.server_host,
        "organization_slug": args.organization_slug,
        "project_id": None,
        "task_name_prefix": args.task_name_prefix,
        "annotation_format": args.annotation_format,
        "expected_tasks": int(args.active_video_packages),
        "expected_review_bboxes": int(args.active_review_bboxes),
        "package_video_count": int(args.expected_video_packages),
        "package_review_bboxes": int(args.expected_review_bboxes),
        "excluded_videos": list(args.excluded_videos),
        "exclusion_manifest_sha256": args.exclusion_manifest_sha256,
        "annotation_snapshot_manifest_sha256": args.annotation_snapshot_manifest_sha256,
        "transfer_manifest_sha256": transfer_contract["manifest_sha256"],
        "original_video_progress_sha256": transfer_contract[
            "original_video_progress_sha256"
        ],
    }
    stored_format = str(state.get("annotation_format", ""))
    if stored_format.casefold() == str(args.annotation_format).casefold():
        state["annotation_format"] = args.annotation_format
    for key, value in expected.items():
        if state.get(key) != value:
            raise CvatTaskBatchError(
                f"Existing full-video state differs for {key}: "
                f"stored={state.get(key)!r}, expected={value!r}"
            )
    if not isinstance(state.get("tasks"), Mapping):
        raise CvatTaskBatchError("Existing full-video state tasks must be an object")
    state["tasks"] = dict(state["tasks"])
    return state


def _write_fullvideo_progress(path: Path, state: Mapping[str, Any]) -> None:
    rows = _read_csv(path)
    tasks = state.get("tasks", {})
    excluded = set(str(value) for value in state.get("excluded_videos", []))
    if not isinstance(tasks, Mapping):
        raise CvatTaskBatchError("Full-video state tasks must be an object")
    original_fields = list(rows[0].keys())
    task_fields = [
        "cvat_task_id",
        "cvat_task_name",
        "cvat_task_url",
        "cvat_task_creation_status",
        "cvat_backup_path",
        "cvat_backup_sha256",
        "cvat_task_updated_utc",
    ]
    fieldnames = original_fields + [
        value for value in task_fields if value not in original_fields
    ]
    seen_videos: set[str] = set()
    for row in rows:
        video_name = _safe_video_name(row.get("video_name", ""))
        if video_name in seen_videos:
            raise CvatTaskBatchError(
                f"Duplicate video in full-video progress CSV: {video_name}"
            )
        seen_videos.add(video_name)
        task = dict(tasks.get(video_name, {}))
        default_status = "excluded" if video_name in excluded else "pending"
        row.update(
            {
                "cvat_task_id": task.get("task_id", ""),
                "cvat_task_name": task.get("task_name", ""),
                "cvat_task_url": task.get("task_url", ""),
                "cvat_task_creation_status": task.get("status", default_status),
                "cvat_backup_path": task.get("backup_path", ""),
                "cvat_backup_sha256": task.get("backup_sha256", ""),
                "cvat_task_updated_utc": task.get("updated_utc", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _save_fullvideo_state(
    *,
    state_path: Path,
    task_map_csv: Path,
    progress_csv: Path,
    packages: Sequence[ReviewPackage],
    state: dict[str, Any],
) -> None:
    state["updated_utc"] = _utc_now()
    _atomic_write_json(state_path, state)
    _write_task_map(task_map_csv, packages, state)
    _write_fullvideo_progress(progress_csv, state)


def _fullvideo_state_row(
    *, args: argparse.Namespace, package: ReviewPackage, task: Any, status: str
) -> dict[str, Any]:
    row = _state_row(
        args=args,
        package=package,
        task=task,
        status=status,
        reference_task=False,
    )
    row["project_id"] = None
    row["backup_bytes"] = 0
    row["backup_members"] = 0
    return row


def run_fullvideo_standalone(args: argparse.Namespace) -> None:
    review_root = Path(args.review_root).resolve()
    state_path = Path(args.state_json).resolve()
    task_map_csv = Path(args.task_map_csv).resolve()
    progress_csv = review_root / "video_progress.csv"
    transfer_contract = verify_fullvideo_transfer_package(
        review_root,
        allow_mutable_progress=state_path.exists(),
        expected_video_packages=args.expected_video_packages,
        expected_review_bboxes=args.expected_review_bboxes,
    )
    packages = load_review_packages(
        review_root,
        expected_video_packages=args.expected_video_packages,
        expected_export_mode="full_video",
    )
    packages, snapshot_manifest_sha256 = _apply_annotation_snapshot_overrides(
        packages, getattr(args, "annotation_snapshot_root", None)
    )
    all_by_video = {package.video_name: package for package in packages}
    excluded_videos, exclusion_manifest_sha256 = _load_excluded_videos(
        getattr(args, "exclusion_manifest", None), set(all_by_video)
    )
    actual_review_bboxes = sum(package.review_bboxes for package in packages)
    if actual_review_bboxes != int(args.expected_review_bboxes):
        raise CvatTaskBatchError(
            "Full-video package target BBox count mismatch: "
            f"{actual_review_bboxes} != {args.expected_review_bboxes}"
        )
    active_packages = [
        package for package in packages if package.video_name not in excluded_videos
    ]
    by_video = {package.video_name: package for package in active_packages}
    active_review_bboxes = sum(package.review_bboxes for package in active_packages)
    args.active_video_packages = len(active_packages)
    args.active_review_bboxes = active_review_bboxes
    args.excluded_videos = sorted(excluded_videos)
    args.exclusion_manifest_sha256 = exclusion_manifest_sha256
    args.annotation_snapshot_manifest_sha256 = snapshot_manifest_sha256
    if args.only_video and args.only_video in excluded_videos:
        raise CvatTaskBatchError(f"--only_video is excluded by manifest: {args.only_video}")
    if args.only_video and args.only_video not in by_video:
        raise CvatTaskBatchError(
            f"--only_video is not in the full-video package: {args.only_video}"
        )
    selected_packages = [
        package
        for package in active_packages
        if not args.only_video or package.video_name == args.only_video
    ]

    print("Stage 4 Phase 5 full-video standalone CVAT task batch")
    print(f"  review_root       : {review_root}")
    print(f"  server_host       : {args.server_host}")
    print(f"  video_packages    : {len(packages)}")
    print(f"  target_bboxes     : {actual_review_bboxes}")
    print(f"  active_tasks      : {len(active_packages)}")
    print(f"  active_bboxes     : {active_review_bboxes}")
    print(f"  excluded_videos   : {sorted(excluded_videos)}")
    print(
        "  annotation_source: "
        + ("corrected snapshots" if snapshot_manifest_sha256 else "package initial ZIPs")
    )
    print(f"  selected_now      : {len(selected_packages)}")
    print(f"  only_video        : {args.only_video or 'all'}")
    print(f"  task_name_prefix  : {args.task_name_prefix}")
    print(f"  transfer_files    : {transfer_contract['transfer_files']}")
    print(f"  apply             : {args.apply}")

    with _connect(args) as client:
        client.organization_slug = args.organization_slug
        state = _load_fullvideo_state(state_path, args, transfer_contract)
        state_tasks: dict[str, Any] = state["tasks"]
        unknown_state_videos = sorted(set(state_tasks) - set(by_video))
        if unknown_state_videos:
            raise CvatTaskBatchError(
                f"State contains videos outside the package: {unknown_state_videos}"
            )

        visible_tasks = list(client.tasks.list())
        tasks_by_name: dict[str, list[Any]] = {}
        for task in visible_tasks:
            tasks_by_name.setdefault(str(task.name), []).append(task)

        planned_new = 0
        for package in selected_packages:
            expected_name = f"{args.task_name_prefix}{package.video_name}"
            stored = dict(state_tasks.get(package.video_name, {}))
            if stored:
                action = (
                    f"resume task #{stored.get('task_id')} "
                    f"from {stored.get('status')}"
                )
            else:
                matches = tasks_by_name.get(expected_name, [])
                if matches:
                    raise CvatTaskBatchError(
                        f"Untracked existing task name collision: {expected_name!r}, "
                        f"ids={[int(task.id) for task in matches]}. "
                        "Refusing to adopt it implicitly."
                    )
                action = "create + import + pre-review backup"
                planned_new += 1
            print(
                f"  [PLAN] {package.video_name}: frames={package.image_count}, "
                f"targets={package.review_bboxes}, action={action}"
            )
        print(f"  new_tasks_planned : {planned_new}")
        if not args.apply:
            print(
                "Stage 4 Phase 5 full-video CVAT task dry-run passed; "
                "no server or local state was changed."
            )
            return

        _, ResourceType = _import_sdk()
        created_this_run = 0
        backed_up_this_run = 0
        for index, package in enumerate(selected_packages, start=1):
            expected_name = f"{args.task_name_prefix}{package.video_name}"
            stored = dict(state_tasks.get(package.video_name, {}))
            if stored:
                task_id = int(stored.get("task_id", -1))
                if task_id <= 0:
                    raise CvatTaskBatchError(
                        f"Stored task ID is invalid: {package.video_name}: {task_id}"
                    )
                task = client.tasks.retrieve(task_id)
                _verify_task(
                    task,
                    package,
                    expected_project_id=None,
                    require_name=expected_name,
                    require_exact_femur_label=True,
                )
                if str(stored.get("annotation_zip_sha256", "")) != package.annotation_zip_sha256:
                    raise CvatTaskBatchError(
                        f"Package ZIP changed after task creation: {package.video_name}"
                    )
            else:
                if args.max_new_tasks and created_this_run >= args.max_new_tasks:
                    print(
                        f"  [{index}/{len(selected_packages)}] [LIMIT] "
                        f"{package.video_name}"
                    )
                    continue
                if tasks_by_name.get(expected_name):
                    raise CvatTaskBatchError(
                        f"Untracked task appeared before creation: {expected_name!r}"
                    )
                print(
                    f"  [{index}/{len(selected_packages)}] creating {expected_name}"
                )
                task = client.tasks.create_from_data(
                    spec={
                        "name": expected_name,
                        "labels": [{"name": "femur", "color": "#ff0000"}],
                    },
                    resources=list(package.images),
                    resource_type=ResourceType.LOCAL,
                    data_params={
                        "image_quality": 100,
                        "sorting_method": "lexicographical",
                    },
                )
                created_this_run += 1
                _verify_task(
                    task,
                    package,
                    expected_project_id=None,
                    require_name=expected_name,
                    require_exact_femur_label=True,
                )
                stored = _fullvideo_state_row(
                    args=args,
                    package=package,
                    task=task,
                    status="task_created",
                )
                state_tasks[package.video_name] = stored
                _save_fullvideo_state(
                    state_path=state_path,
                    task_map_csv=task_map_csv,
                    progress_csv=progress_csv,
                    packages=active_packages,
                    state=state,
                )

            if str(stored.get("status")) == "task_created":
                print(
                    f"  [{index}/{len(selected_packages)}] importing annotations: "
                    f"{package.video_name}"
                )
                task.import_annotations(
                    args.annotation_format,
                    package.annotation_zip,
                    conv_mask_to_poly=False,
                )
                _verify_task(
                    task,
                    package,
                    expected_project_id=None,
                    require_name=expected_name,
                    require_exact_femur_label=True,
                )
                stored["status"] = "annotations_imported"
                stored["updated_utc"] = _utc_now()
                state_tasks[package.video_name] = stored
                _save_fullvideo_state(
                    state_path=state_path,
                    task_map_csv=task_map_csv,
                    progress_csv=progress_csv,
                    packages=active_packages,
                    state=state,
                )

            backup_path = _backup_path(package, expected_name)
            if str(stored.get("status")) == "backup_verified":
                backup_bytes, backup_members = _validate_task_backup(backup_path)
                if (
                    str(stored.get("backup_sha256", "")) != _file_sha256(backup_path)
                    or int(stored.get("backup_bytes", -1)) != backup_bytes
                    or int(stored.get("backup_members", -1)) != backup_members
                ):
                    raise CvatTaskBatchError(
                        f"Verified pre-review backup changed: {package.video_name}"
                    )
                print(
                    f"  [{index}/{len(selected_packages)}] [SKIP backup_verified] "
                    f"{package.video_name}"
                )
                continue
            if str(stored.get("status")) != "annotations_imported":
                raise CvatTaskBatchError(
                    f"Unsupported task creation state for {package.video_name}: "
                    f"{stored.get('status')!r}"
                )
            if backup_path.exists():
                raise FileExistsError(
                    f"Untracked pre-review backup already exists: {backup_path}"
                )
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"  [{index}/{len(selected_packages)}] downloading pre-review backup: "
                f"{package.video_name}"
            )
            task.download_backup(backup_path, lightweight=False)
            backup_bytes, backup_members = _validate_task_backup(backup_path)
            stored["status"] = "backup_verified"
            stored["backup_path"] = str(backup_path)
            stored["backup_sha256"] = _file_sha256(backup_path)
            stored["backup_bytes"] = backup_bytes
            stored["backup_members"] = backup_members
            stored["updated_utc"] = _utc_now()
            state_tasks[package.video_name] = stored
            backed_up_this_run += 1
            _save_fullvideo_state(
                state_path=state_path,
                task_map_csv=task_map_csv,
                progress_csv=progress_csv,
                packages=active_packages,
                state=state,
            )

        completed_total = sum(
            str(row.get("status")) == "backup_verified"
            for row in state_tasks.values()
        )
        state["status"] = (
            "complete" if completed_total == len(active_packages) else "in_progress"
        )
        state["completed_tasks"] = completed_total
        state["expected_tasks"] = len(active_packages)
        _save_fullvideo_state(
            state_path=state_path,
            task_map_csv=task_map_csv,
            progress_csv=progress_csv,
            packages=active_packages,
            state=state,
        )
        print("Stage 4 Phase 5 full-video CVAT task batch finished.")
        print(f"  tasks_created_this_run : {created_this_run}")
        print(f"  backups_verified_now   : {backed_up_this_run}")
        print(f"  backups_verified_total : {completed_total}/{len(active_packages)}")
        print(f"  state_json              : {state_path}")
        print(f"  task_map_csv            : {task_map_csv}")
        print(f"  video_progress_csv      : {progress_csv}")


def run(args: argparse.Namespace) -> None:
    packages = load_review_packages(
        args.review_root, expected_video_packages=args.expected_video_packages
    )
    by_video = {package.video_name: package for package in packages}
    master_package = None
    if args.reference_is_master:
        master_package = load_master_review_package(args.review_root, packages)
    elif args.reference_video not in by_video:
        raise CvatTaskBatchError(
            f"Reference video is not in the review batch: {args.reference_video}"
        )

    print("Stage 4 Phase 5-B CVAT task batch")
    print(f"  review_root       : {Path(args.review_root).resolve()}")
    print(f"  server_host       : {args.server_host}")
    print(f"  reference_task_id : {args.reference_task_id}")
    print(
        "  reference_scope   : "
        + ("all_phase3_master" if args.reference_is_master else args.reference_video)
    )
    print(f"  video_packages    : {len(packages)}")
    if master_package is not None:
        print(f"  master_frames     : {master_package.image_count}")
        print(f"  master_bboxes     : {master_package.review_bboxes}")
    print(f"  apply             : {args.apply}")

    with _connect(args) as client:
        reference_task = client.tasks.retrieve(args.reference_task_id)
        reference_project_id = _task_project_id(reference_task)
        reference_organization = getattr(reference_task, "organization", None)
        if reference_organization is not None and not args.organization_slug:
            raise CvatTaskBatchError(
                "Reference task belongs to an organization; pass --organization_slug "
                "with that organization's slug"
            )
        client.organization_slug = args.organization_slug
        reference_task = client.tasks.retrieve(args.reference_task_id)
        reference_package = (
            master_package
            if master_package is not None
            else by_video[args.reference_video]
        )
        _verify_task(
            reference_task,
            reference_package,
            expected_project_id=reference_project_id,
            require_name=None,
        )
        print(
            f"  inferred_project  : {reference_project_id if reference_project_id is not None else 'standalone'}"
        )
        print(f"  reference_task    : #{reference_task.id} {reference_task.name!r} [OK]")

        if args.reference_is_master:
            if args.apply:
                raise CvatTaskBatchError(
                    "Task #{} is already the complete 129-frame master task; "
                    "--apply/per-video task creation is unnecessary".format(
                        args.reference_task_id
                    )
                )
            print(
                "Stage 4 Phase 5-B master CVAT task preflight passed; "
                "all root and per-video package images are accounted for."
            )
            return

        state_path = Path(args.state_json).resolve()
        task_map_csv = Path(args.task_map_csv).resolve()
        state = _load_state(state_path, args, reference_project_id)
        state_tasks: dict[str, Any] = state["tasks"]

        visible_tasks = list(client.tasks.list())
        tasks_by_name: dict[str, list[Any]] = {}
        for task in visible_tasks:
            tasks_by_name.setdefault(str(task.name), []).append(task)

        planned_new = 0
        for package in packages:
            if package.video_name == args.reference_video:
                action = f"reuse reference task #{args.reference_task_id}"
            elif package.video_name in state_tasks:
                action = f"resume task #{state_tasks[package.video_name].get('task_id')}"
            else:
                task_name = f"{args.task_name_prefix}{package.video_name}"
                matches = tasks_by_name.get(task_name, [])
                if matches:
                    raise CvatTaskBatchError(
                        f"Untracked existing task name collision: {task_name!r}, "
                        f"ids={[int(task.id) for task in matches]}. Refusing to adopt it implicitly."
                    )
                action = "create + import + backup"
                planned_new += 1
            print(
                f"  [PLAN] {package.video_name}: frames={package.image_count}, "
                f"bboxes={package.review_bboxes}, action={action}"
            )
        print(f"  new_tasks_planned : {planned_new}")
        if not args.apply:
            print("Stage 4 Phase 5-B CVAT task dry-run passed; no server or local state was changed.")
            return

        _, ResourceType = _import_sdk()
        created_this_run = 0
        completed_this_run = 0
        for index, package in enumerate(packages, start=1):
            expected_name = f"{args.task_name_prefix}{package.video_name}"
            stored = dict(state_tasks.get(package.video_name, {}))
            is_reference = package.video_name == args.reference_video
            if is_reference:
                task = reference_task
                if stored and int(stored.get("task_id", -1)) != int(task.id):
                    raise CvatTaskBatchError(
                        f"Stored reference task ID mismatch for {package.video_name}"
                    )
                if not stored:
                    stored = _state_row(
                        args=args,
                        package=package,
                        task=task,
                        status="annotations_imported",
                        reference_task=True,
                    )
                    state_tasks[package.video_name] = stored
                    _save_state(
                        state_path=state_path,
                        task_map_csv=task_map_csv,
                        packages=packages,
                        state=state,
                    )
            elif stored:
                task = client.tasks.retrieve(int(stored.get("task_id", -1)))
                _verify_task(
                    task,
                    package,
                    expected_project_id=reference_project_id,
                    require_name=expected_name,
                )
                if str(stored.get("annotation_zip_sha256", "")) != package.annotation_zip_sha256:
                    raise CvatTaskBatchError(
                        f"Package ZIP changed after task creation: {package.video_name}"
                    )
            else:
                if args.max_new_tasks and created_this_run >= args.max_new_tasks:
                    print(f"  [{index}/{len(packages)}] [LIMIT] {package.video_name}")
                    continue
                spec: dict[str, Any] = {"name": expected_name}
                if reference_project_id is None:
                    spec["labels"] = [{"name": "femur", "color": "#ff0000"}]
                else:
                    spec["project_id"] = reference_project_id
                print(f"  [{index}/{len(packages)}] creating {expected_name}")
                task = client.tasks.create_from_data(
                    spec=spec,
                    resources=list(package.images),
                    resource_type=ResourceType.LOCAL,
                    data_params={
                        "image_quality": 100,
                        "sorting_method": "lexicographical",
                    },
                )
                created_this_run += 1
                _verify_task(
                    task,
                    package,
                    expected_project_id=reference_project_id,
                    require_name=expected_name,
                )
                stored = _state_row(
                    args=args,
                    package=package,
                    task=task,
                    status="task_created",
                    reference_task=False,
                )
                state_tasks[package.video_name] = stored
                _save_state(
                    state_path=state_path,
                    task_map_csv=task_map_csv,
                    packages=packages,
                    state=state,
                )

            if not is_reference and str(stored.get("status")) == "task_created":
                print(f"  [{index}/{len(packages)}] importing annotations: {package.video_name}")
                task.import_annotations(
                    args.annotation_format,
                    package.annotation_zip,
                    conv_mask_to_poly=False,
                )
                stored["status"] = "annotations_imported"
                stored["updated_utc"] = _utc_now()
                state_tasks[package.video_name] = stored
                _save_state(
                    state_path=state_path,
                    task_map_csv=task_map_csv,
                    packages=packages,
                    state=state,
                )

            backup_path = _backup_path(package, expected_name)
            if str(stored.get("status")) == "completed":
                expected_backup_sha = str(stored.get("backup_sha256", ""))
                if (
                    not backup_path.is_file()
                    or not expected_backup_sha
                    or _file_sha256(backup_path) != expected_backup_sha
                ):
                    raise CvatTaskBatchError(
                        f"Completed task backup is missing or changed: {package.video_name}"
                    )
                print(f"  [{index}/{len(packages)}] [SKIP completed] {package.video_name}")
                continue
            if backup_path.exists():
                raise FileExistsError(
                    f"Untracked pre-review backup already exists: {backup_path}"
                )
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [{index}/{len(packages)}] downloading pre-review backup: {package.video_name}")
            task.download_backup(backup_path, lightweight=False)
            if not backup_path.is_file() or backup_path.stat().st_size == 0:
                raise CvatTaskBatchError(f"CVAT produced an empty/missing backup: {backup_path}")
            stored["status"] = "completed"
            stored["backup_path"] = str(backup_path)
            stored["backup_sha256"] = _file_sha256(backup_path)
            stored["updated_utc"] = _utc_now()
            state_tasks[package.video_name] = stored
            completed_this_run += 1
            _save_state(
                state_path=state_path,
                task_map_csv=task_map_csv,
                packages=packages,
                state=state,
            )

        completed_total = sum(
            str(row.get("status")) == "completed" for row in state_tasks.values()
        )
        state["status"] = (
            "complete" if completed_total == len(packages) else "in_progress"
        )
        state["completed_tasks"] = completed_total
        state["expected_tasks"] = len(packages)
        _save_state(
            state_path=state_path,
            task_map_csv=task_map_csv,
            packages=packages,
            state=state,
        )
        print("Stage 4 Phase 5-B CVAT task batch finished.")
        print(f"  tasks_created_this_run : {created_this_run}")
        print(f"  tasks_completed_now    : {completed_this_run}")
        print(f"  tasks_completed_total  : {completed_total}/{len(packages)}")
        print(f"  state_json              : {state_path}")
        print(f"  task_map_csv            : {task_map_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one local CVAT task per Phase 5 video package, import the fixed "
            "Segmentation Mask 1.1 ZIP, and save a pre-review task backup."
        )
    )
    parser.add_argument("--review_root", type=Path, required=True)
    parser.add_argument("--server_host", default="http://localhost:8080")
    parser.add_argument("--username", default="")
    parser.add_argument(
        "--organization_slug",
        default="",
        help="CVAT organization slug; leave empty for the personal workspace.",
    )
    parser.add_argument("--reference_task_id", type=int, default=5)
    parser.add_argument("--reference_video", default="1-3_14")
    parser.add_argument(
        "--reference_is_master",
        action="store_true",
        help=(
            "Validate the reference task against review_root/images (all Phase 3 "
            "cases) and do not create per-video tasks."
        ),
    )
    parser.add_argument(
        "--full_video_standalone",
        action="store_true",
        help=(
            "Create standalone tasks from the Step 4 full-video package without "
            "using or modifying the selected-frame reference task."
        ),
    )
    parser.add_argument(
        "--only_video",
        default="",
        help="Limit this invocation to one video (used for the first-task smoke test).",
    )
    parser.add_argument("--task_name_prefix", default=TASK_NAME_PREFIX)
    parser.add_argument("--annotation_format", default=ANNOTATION_FORMAT)
    parser.add_argument(
        "--annotation_snapshot_root",
        type=Path,
        default=None,
        help=(
            "Optional completed corrected-task snapshot root. Its per-video "
            "Segmentation mask ZIPs replace the package initial ZIPs."
        ),
    )
    parser.add_argument(
        "--exclusion_manifest",
        type=Path,
        default=None,
        help="Optional CSV with video_name rows to omit from new Task creation.",
    )
    parser.add_argument("--expected_video_packages", type=int, default=60)
    parser.add_argument("--expected_review_bboxes", type=int, default=129)
    parser.add_argument(
        "--max_new_tasks",
        type=int,
        default=0,
        help="Maximum tasks to create in this invocation; 0 means unlimited.",
    )
    parser.add_argument("--state_json", type=Path, default=None)
    parser.add_argument("--task_map_csv", type=Path, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create/import/backup tasks. Without this flag, run read-only preflight.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.review_root = Path(args.review_root).resolve()
    if args.annotation_snapshot_root is not None:
        args.annotation_snapshot_root = Path(args.annotation_snapshot_root).resolve()
    if args.exclusion_manifest is not None:
        args.exclusion_manifest = Path(args.exclusion_manifest).resolve()
    if args.full_video_standalone:
        if args.reference_is_master:
            raise SystemExit(
                "--full_video_standalone and --reference_is_master are mutually exclusive"
            )
        if args.reference_task_id < 0:
            raise SystemExit("--reference_task_id must be >= 0 in standalone mode")
    elif args.reference_task_id <= 0:
        raise SystemExit("--reference_task_id must be > 0")
    if args.expected_video_packages <= 0:
        raise SystemExit("--expected_video_packages must be > 0")
    if args.expected_review_bboxes <= 0:
        raise SystemExit("--expected_review_bboxes must be > 0")
    if args.max_new_tasks < 0:
        raise SystemExit("--max_new_tasks must be >= 0")
    if not args.task_name_prefix or any(value in args.task_name_prefix for value in ("/", "\\", "\n")):
        raise SystemExit("--task_name_prefix must be non-empty and path-safe")
    if not args.annotation_format or any(
        value in args.annotation_format for value in ("\r", "\n", "\0")
    ):
        raise SystemExit("--annotation_format must be non-empty and printable")
    args.reference_video = _safe_video_name(args.reference_video)
    if args.only_video:
        args.only_video = _safe_video_name(args.only_video)
    if args.only_video and not args.full_video_standalone:
        raise SystemExit("--only_video requires --full_video_standalone")
    args.server_host = str(args.server_host).rstrip("/")
    if not args.server_host.startswith(("http://", "https://")):
        raise SystemExit("--server_host must begin with http:// or https://")
    parsed_host = urlparse(args.server_host)
    if parsed_host.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise SystemExit(
            "--server_host must resolve explicitly to localhost/loopback; "
            "external CVAT hosts are forbidden for this dataset"
        )
    cvat_state_root = args.review_root / "cvat"
    args.state_json = Path(args.state_json or cvat_state_root / "task_creation_state.json")
    args.task_map_csv = Path(args.task_map_csv or cvat_state_root / "task_map.csv")
    try:
        if args.full_video_standalone:
            run_fullvideo_standalone(args)
        else:
            run(args)
    except Exception as exc:
        operation = (
            "Stage 4 Phase 5 full-video CVAT task batch"
            if args.full_video_standalone
            else "Stage 4 Phase 5-B CVAT task batch"
        )
        raise SystemExit(
            f"{operation} failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
