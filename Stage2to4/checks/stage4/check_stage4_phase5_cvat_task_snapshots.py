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

from pseudo3d.batch.export import batch_export_stage4_phase5_cvat_task_snapshots as snapshots


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FakeTask:
    def __init__(self, task_id: int, name: str, frames: list[str]) -> None:
        self.id = task_id
        self.name = name
        self.size = len(frames)
        self._frames = [SimpleNamespace(name=value) for value in frames]

    def fetch(self) -> None:
        return None

    def get_frames_info(self) -> list[SimpleNamespace]:
        return self._frames

    def get_labels(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="femur")]

    def export_dataset(self, format_name: str, path: Path, *, include_images: bool) -> None:
        assert format_name == "Segmentation mask 1.1"
        assert include_images is False
        frame_names = [str(frame.name) for frame in self._frames]
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("labelmap.txt", "# label:color_rgb:parts:actions\nbackground:0,0,0::\nfemur:255,0,0::\n")
            archive.writestr("ImageSets/Segmentation/default.txt", "\n".join(frame_names) + "\n")
            archive.writestr(f"SegmentationClass/{Path(frame_names[0]).stem}.png", b"png")

    def download_backup(self, path: Path, *, lightweight: bool) -> None:
        assert lightweight is False
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("task.json", json.dumps({"id": self.id}))
            archive.writestr("annotations.json", "{}")


class FakeClient:
    def __init__(self, tasks: list[FakeTask]) -> None:
        self.tasks = SimpleNamespace(retrieve=lambda task_id: {
            task.id: task for task in tasks
        }[int(task_id)])
        self.organization_slug = ""


def _args(review_root: Path, *, apply: bool) -> argparse.Namespace:
    return argparse.Namespace(
        review_root=review_root,
        server_host="http://localhost:8080",
        username="test",
        organization_slug="",
        task_map_csv=review_root / "task_management" / "task_map.csv",
        output_root=review_root / "snapshots",
        annotation_format="Segmentation mask 1.1",
        expected_tasks=2,
        only_video="",
        max_tasks=0,
        apply=apply,
    )


def test_dry_run_apply_and_resume(root: Path) -> None:
    review_root = root / "review"
    rows: list[dict[str, object]] = []
    fake_tasks: list[FakeTask] = []
    for task_id, video_name, frame_count in ((6, "video_a", 2), (7, "video_b", 3)):
        frames = [f"{video_name}__fo{index:05d}__fi{index:08d}.png" for index in range(frame_count)]
        images = review_root / "videos" / video_name / "images"
        images.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            (images / frame).write_bytes(b"image")
        rows.append(
            {
                "video_name": video_name,
                "task_id": task_id,
                "task_name": f"stage4_phase5_fullvideo__{video_name}",
                "task_url": f"http://localhost:8080/tasks/{task_id}",
                "status": "completed",
                "image_count": frame_count,
                "review_bboxes": 1,
                "actionable_bboxes": 1,
                "context_only_frames": frame_count - 1,
                "annotation_zip_sha256": "unused",
                "backup_path": "unused",
                "backup_sha256": "unused",
            }
        )
        fake_tasks.append(FakeTask(task_id, str(rows[-1]["task_name"]), frames))
    _write_csv(review_root / "task_management" / "task_map.csv", rows)
    client = FakeClient(fake_tasks)

    snapshots.run(_args(review_root, apply=False), client=client)
    assert not (review_root / "snapshots").exists()

    snapshots.run(_args(review_root, apply=True), client=client)
    summary = json.loads((review_root / "snapshots" / "export_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["completed_tasks"] == 2
    manifest = snapshots._read_csv(review_root / "snapshots" / "export_manifest.csv")
    assert len(manifest) == 2
    assert {row["status"] for row in manifest} == {"complete"}
    for row in manifest:
        snapshots._validate_annotation_zip(Path(row["annotation_zip"]))
        snapshots.task_batch._validate_task_backup(Path(row["backup_zip"]))

    before = {
        path: snapshots.task_batch._file_sha256(path)
        for path in (review_root / "snapshots").rglob("*.zip")
    }
    snapshots.run(_args(review_root, apply=True), client=client)
    after = {path: snapshots.task_batch._file_sha256(path) for path in before}
    assert before == after
    print("[OK] dry-run, corrected annotation export, task backup, manifest, and resume")


def test_contract_failures(root: Path) -> None:
    review_root = root / "invalid"
    images = review_root / "videos" / "video_a" / "images"
    images.mkdir(parents=True)
    (images / "frame.png").write_bytes(b"image")
    _write_csv(
        review_root / "task_management" / "task_map.csv",
        [{
            "video_name": "video_a",
            "task_id": 6,
            "task_name": "task",
            "status": "completed",
            "image_count": 2,
            "review_bboxes": 1,
        }],
    )
    try:
        snapshots.load_task_map(
            review_root,
            review_root / "task_management" / "task_map.csv",
            expected_tasks=1,
        )
    except snapshots.CvatTaskSnapshotError as exc:
        assert "Local image count mismatch" in str(exc)
    else:
        raise AssertionError("image count mismatch was accepted")
    print("[OK] strict task-map and local-frame validation")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_cvat_task_snapshots_") as temporary:
        root = Path(temporary)
        test_dry_run_apply_and_resume(root)
        test_contract_failures(root)
    print("Stage 4 Phase 5 CVAT corrected-task snapshot synthetic checks passed.")


if __name__ == "__main__":
    main()
