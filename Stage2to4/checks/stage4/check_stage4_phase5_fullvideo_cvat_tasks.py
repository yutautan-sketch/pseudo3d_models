from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.batch.export import batch_create_stage4_phase5_cvat_tasks as tasks


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_zip(path: Path, member: str = "labelmap.txt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, b"background:0,0,0::\nfemur:255,0,0::\n")


def _build_review_root(root: Path) -> Path:
    review_root = root / "manual_review_cvat_phase5_fullvideo_v2"
    videos = (("video_a", 2, 1), ("video_b", 3, 2))
    progress_rows: list[dict[str, object]] = []
    for video_name, frames, targets in videos:
        video_root = review_root / "videos" / video_name
        for frame_order in range(frames):
            image = video_root / "images" / (
                f"{video_name}__fo{frame_order:05d}__fi{frame_order:08d}.png"
            )
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image:{video_name}:{frame_order}".encode("utf-8"))
        annotation_zip = (
            video_root / "cvat" / "annotations_segmentation_mask_1_1.zip"
        )
        _write_zip(annotation_zip)
        _write_json(
            video_root / "export_summary.json",
            {
                "status": "ok",
                "failure_rows": 0,
                "export_mode": "full_video",
                "partition_video_name": video_name,
                "review_frames": frames,
                "cvat_import_zip_sha256": tasks._file_sha256(annotation_zip),
            },
        )
        progress_rows.append(
            {
                "video_name": video_name,
                "export_mode": "full_video",
                "review_frames": frames,
                "target_review_frames": targets,
                "review_bboxes": targets,
                "actionable_bboxes": targets,
                "unreviewable_bboxes": 0,
                "auto_accept": 0,
                "auto_refine": targets,
                "manual_review": 0,
                "context_only_frames": frames - targets,
                "package_root": f"videos/{video_name}",
                "cvat_import_zip": (
                    f"videos/{video_name}/cvat/"
                    "annotations_segmentation_mask_1_1.zip"
                ),
                "review_status": "pending",
                "correction_status": "pending",
            }
        )
    _write_csv(review_root / "video_progress.csv", progress_rows)
    _write_json(
        review_root / "partition_summary.json",
        {
            "status": "ok",
            "failure_rows": 0,
            "video_packages": 2,
            "progress_csv": "video_progress.csv",
        },
    )
    _write_json(
        review_root / "package_validation_summary.json",
        {
            "status": "ok",
            "failure_rows": 0,
            "videos_checked": 2,
            "target_bboxes_checked": 3,
        },
    )

    manifest_rows: list[dict[str, object]] = []
    transfer_paths = [
        review_root / "partition_summary.json",
        review_root / "video_progress.csv",
    ] + sorted(path for path in (review_root / "videos").rglob("*") if path.is_file())
    for path in transfer_paths:
        manifest_rows.append(
            {
                "relative_path": path.relative_to(review_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": tasks._file_sha256(path),
            }
        )
    _write_csv(review_root / "mac_transfer_manifest.csv", manifest_rows)
    _write_json(
        review_root / "mac_transfer_summary.json",
        {
            "status": "ok",
            "transfer_files": len(manifest_rows),
            "transfer_bytes": sum(int(row["bytes"]) for row in manifest_rows),
            "manifest_sha256": tasks._file_sha256(
                review_root / "mac_transfer_manifest.csv"
            ),
        },
    )
    return review_root


class _FakeTask:
    def __init__(self, task_id: int, name: str, resources: list[Path]):
        self.id = task_id
        self.name = name
        self.project_id = None
        self.organization = None
        self.size = len(resources)
        self._frames = [SimpleNamespace(name=path.name) for path in resources]
        self._labels = [SimpleNamespace(name="femur")]
        self.annotation_imported = False
        self.annotation_format = ""
        self.annotation_path = ""

    def fetch(self) -> None:
        return None

    def get_frames_info(self) -> list[SimpleNamespace]:
        return self._frames

    def get_labels(self) -> list[SimpleNamespace]:
        return self._labels

    def import_annotations(self, annotation_format: str, *args: object, **_kwargs: object) -> None:
        self.annotation_imported = True
        self.annotation_format = annotation_format
        self.annotation_path = str(args[0]) if args else ""

    def download_backup(self, path: Path, *, lightweight: bool) -> None:
        assert lightweight is False
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("task.json", json.dumps({"id": self.id, "name": self.name}))
            archive.writestr("annotations.json", "{}")


class _FakeTasks:
    def __init__(self) -> None:
        self.values: dict[int, _FakeTask] = {}
        self.next_id = 10

    def list(self) -> list[_FakeTask]:
        return list(self.values.values())

    def retrieve(self, task_id: int) -> _FakeTask:
        return self.values[int(task_id)]

    def create_from_data(
        self,
        *,
        spec: dict[str, object],
        resources: list[Path],
        resource_type: object,
        data_params: dict[str, object],
    ) -> _FakeTask:
        assert resource_type == "LOCAL"
        assert data_params["sorting_method"] == "lexicographical"
        task = _FakeTask(self.next_id, str(spec["name"]), resources)
        self.values[task.id] = task
        self.next_id += 1
        return task


class _FakeClient:
    def __init__(self) -> None:
        self.tasks = _FakeTasks()
        self.organization_slug = ""

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _args(review_root: Path, *, apply: bool, only_video: str) -> argparse.Namespace:
    management = review_root / "task_management"
    return argparse.Namespace(
        review_root=review_root,
        expected_video_packages=2,
        expected_review_bboxes=3,
        only_video=only_video,
        server_host="http://localhost:8080",
        organization_slug="",
        task_name_prefix=tasks.FULLVIDEO_TASK_NAME_PREFIX,
        annotation_format=tasks.ANNOTATION_FORMAT,
        apply=apply,
        max_new_tasks=0,
        state_json=management / "task_creation_state.json",
        task_map_csv=management / "task_map.csv",
        username="test",
    )


def test_smoke_resume_and_backup(root: Path) -> None:
    review_root = _build_review_root(root)
    client = _FakeClient()
    original_connect = tasks._connect
    original_import_sdk = tasks._import_sdk
    tasks._connect = lambda _args: client
    tasks._import_sdk = lambda: (None, SimpleNamespace(LOCAL="LOCAL"))
    try:
        tasks.run_fullvideo_standalone(
            _args(review_root, apply=False, only_video="video_a")
        )
        assert not (review_root / "task_management").exists()

        tasks.run_fullvideo_standalone(
            _args(review_root, apply=True, only_video="video_a")
        )
        assert len(client.tasks.values) == 1
        assert next(iter(client.tasks.values.values())).annotation_format == (
            "Segmentation mask 1.1"
        )
        state_path = review_root / "task_management" / "task_creation_state.json"
        legacy_state = dict(tasks._read_json(state_path))
        legacy_state["annotation_format"] = "Segmentation Mask 1.1"
        _write_json(state_path, legacy_state)
        tasks.run_fullvideo_standalone(_args(review_root, apply=True, only_video=""))
        assert len(client.tasks.values) == 2
        tasks.run_fullvideo_standalone(_args(review_root, apply=True, only_video=""))
        assert len(client.tasks.values) == 2
    finally:
        tasks._connect = original_connect
        tasks._import_sdk = original_import_sdk

    state = tasks._read_json(
        review_root / "task_management" / "task_creation_state.json"
    )
    assert state["status"] == "complete"
    assert int(state["completed_tasks"]) == 2
    for row in state["tasks"].values():
        assert row["status"] == "backup_verified"
        backup = Path(str(row["backup_path"]))
        assert tasks._file_sha256(backup) == row["backup_sha256"]
    progress = tasks._read_csv(review_root / "video_progress.csv")
    assert {row["cvat_task_creation_status"] for row in progress} == {
        "backup_verified"
    }
    print("[OK] standalone dry-run, one-video smoke, resume-all, and idempotent backup verification")


def test_corrected_snapshot_override_and_exclusion(root: Path) -> None:
    review_root = _build_review_root(root / "snapshot_override")
    snapshot_root = root / "corrected_snapshots"
    manifest_rows: list[dict[str, object]] = []
    for task_id, video_name in ((20, "video_a"), (21, "video_b")):
        archive = (
            snapshot_root
            / "annotations"
            / video_name
            / f"{video_name}__task{task_id}__reviewed_segmentation_mask_1_1.zip"
        )
        _write_zip(archive)
        manifest_rows.append(
            {
                "video_name": video_name,
                "task_id": task_id,
                "task_name": f"old__{video_name}",
                "status": "complete",
                "image_count": 0,
                "review_bboxes": 0,
                "annotation_zip": str(archive.resolve()),
                "annotation_bytes": archive.stat().st_size,
                "annotation_sha256": tasks._file_sha256(archive),
                "backup_zip": "unused",
                "backup_bytes": 0,
                "backup_sha256": "unused",
            }
        )
    _write_csv(snapshot_root / "export_manifest.csv", manifest_rows)
    _write_json(
        snapshot_root / "export_summary.json",
        {
            "status": "complete",
            "registered_tasks": 2,
            "completed_tasks": 2,
        },
    )
    exclusions = root / "exclusions.csv"
    _write_csv(
        exclusions,
        [{"video_name": "video_b", "reason_code": "synthetic_exclusion"}],
    )
    args = _args(review_root, apply=True, only_video="")
    args.annotation_snapshot_root = snapshot_root
    args.exclusion_manifest = exclusions
    client = _FakeClient()
    original_connect = tasks._connect
    original_import_sdk = tasks._import_sdk
    original_validate = tasks.validate_cvat_segmentation_mask_zip
    validation_calls: list[tuple[str, set[str]]] = []

    def _fake_validate(path: Path, **kwargs: object) -> SimpleNamespace:
        validation_calls.append(
            (
                str(path),
                set(str(value) for value in kwargs["allowed_missing_mask_stems"]),
            )
        )
        return SimpleNamespace(images=2 if "video_a" in str(path) else 3)

    tasks._connect = lambda _args: client
    tasks._import_sdk = lambda: (None, SimpleNamespace(LOCAL="LOCAL"))
    tasks.validate_cvat_segmentation_mask_zip = _fake_validate
    try:
        tasks.run_fullvideo_standalone(args)
    finally:
        tasks._connect = original_connect
        tasks._import_sdk = original_import_sdk
        tasks.validate_cvat_segmentation_mask_zip = original_validate
    assert len(client.tasks.values) == 1
    assert len(validation_calls) == 2
    for path, allowed_missing in validation_calls:
        video_name = "video_a" if "video_a" in path else "video_b"
        expected_frames = 2 if video_name == "video_a" else 3
        assert allowed_missing == {
            f"{video_name}__fo{index:05d}__fi{index:08d}"
            for index in range(expected_frames)
        }
    task = next(iter(client.tasks.values.values()))
    assert "corrected_snapshots" in task.annotation_path
    state = tasks._read_json(review_root / "task_management" / "task_creation_state.json")
    assert state["status"] == "complete"
    assert state["expected_tasks"] == 1
    assert state["excluded_videos"] == ["video_b"]
    assert set(state["tasks"]) == {"video_a"}
    task_map = tasks._read_csv(review_root / "task_management" / "task_map.csv")
    assert [row["video_name"] for row in task_map] == ["video_a"]
    progress = tasks._read_csv(review_root / "video_progress.csv")
    excluded = next(row for row in progress if row["video_name"] == "video_b")
    assert excluded["cvat_task_creation_status"] == "excluded"
    print("[OK] corrected snapshot override and explicit video exclusion")


def test_transfer_tamper_rejection(root: Path) -> None:
    review_root = _build_review_root(root)
    image = next((review_root / "videos").rglob("*.png"))
    image.write_bytes(image.read_bytes() + b"tampered")
    try:
        tasks.verify_fullvideo_transfer_package(
            review_root,
            allow_mutable_progress=False,
            expected_video_packages=2,
            expected_review_bboxes=3,
        )
    except tasks.CvatTaskBatchError as exc:
        assert "differs from Step 4 manifest" in str(exc)
    else:
        raise AssertionError("Transferred-file tampering was not rejected")
    print("[OK] transferred package tampering rejection")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_fullvideo_cvat_tasks_") as value:
        root = Path(value)
        test_smoke_resume_and_backup(root / "roundtrip")
        test_corrected_snapshot_override_and_exclusion(root / "snapshot")
        test_transfer_tamper_rejection(root / "tamper")
    print("Stage 4 Phase 5 full-video CVAT task synthetic checks passed.")


if __name__ == "__main__":
    main()
