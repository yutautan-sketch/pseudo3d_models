from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.file_list_mode import (  # noqa: E402
    FixedListError,
    compare_file_lists,
    list_content_sha256,
    list_identity_sha256,
    read_file_list,
    validate_fixed_lists,
    write_combined_manifest,
)

# ----------------------------------------------------------------------------
# S5-15 P2 Step 3: fixed train/val list inputs.
#
# Report section 7.1.1: the lists must be used whole and in order, never
# silently replaced by a directory scan that a seed re-splits, and every
# failure must stop rather than fall back. Pure stdlib -- no numpy/h5py/torch,
# and no training is started anywhere in this file.
# ----------------------------------------------------------------------------

H5_PATTERN = "*_pointcloud_annotated_foreground_combined_v2_global_local_l75_w31_c12_area15_bboxrank_v7_cvat_authoritative_crop_quality_v1.h5"
MODULE_PATH = REPO_ROOT / "stage5" / "utils" / "file_list_mode.py"


def video_name(index: int, *, teacher: str = "bboxrank_v7_cvat_authoritative_crop_quality_v1") -> str:
    return (
        f"2026071{index % 10}_12{index:04d}_{index}_pointcloud_annotated_foreground_combined_v2_"
        f"global_local_l75_w31_c12_area15_{teacher}.h5"
    )


def make_h5_files(root: Path, indices: range | list[int], **kwargs) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in indices:
        path = root / video_name(index, **kwargs)
        path.write_bytes(b"not a real h5, only the name and existence matter here")
        paths.append(path)
    return paths


def write_list(path: Path, paths: list[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return path


def expect_raises(exception_types, message: str, callable_, *args, **kwargs) -> None:
    try:
        callable_(*args, **kwargs)
    except exception_types:
        return
    raise AssertionError(f"expected {exception_types} for {message}, but no exception was raised")


def build_valid_pair(root: Path, *, train_count: int = 5, val_count: int = 2) -> tuple[Path, Path, list[Path], list[Path]]:
    data_dir = root / "collected"
    train_files = make_h5_files(data_dir, range(train_count))
    val_files = make_h5_files(data_dir, range(1000, 1000 + val_count))
    train_list = write_list(root / "train_files.txt", train_files)
    val_list = write_list(root / "val_files.txt", val_files)
    return train_list, val_list, train_files, val_files


def test_valid_pair_is_accepted_in_list_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, train_files, val_files = build_valid_pair(root)
        # Deliberately unsorted on disk order: the list order is authoritative.
        shuffled = [train_files[2], train_files[0], train_files[4], train_files[1], train_files[3]]
        write_list(train_list, shuffled)

        inputs = validate_fixed_lists(train_list, val_list, h5_pattern=H5_PATTERN)
        assert list(inputs.train_paths) == shuffled, "train order was not preserved"
        assert list(inputs.val_paths) == val_files
        assert inputs.all_paths == tuple(shuffled) + tuple(val_files)
    print("  ok: a valid pair is accepted and both lists keep their own order (not re-sorted)")


def test_one_sided_or_empty_or_unreadable_lists_stop_without_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, _, _ = build_valid_pair(root)

        missing = root / "does_not_exist.txt"
        expect_raises(FixedListError, "missing train list", validate_fixed_lists, missing, val_list)
        expect_raises(FixedListError, "missing val list", validate_fixed_lists, train_list, missing)

        empty = write_list(root / "empty.txt", [])
        empty.write_text("", encoding="utf-8")
        expect_raises(FixedListError, "empty train list", validate_fixed_lists, empty, val_list)

        comments_only = root / "comments.txt"
        comments_only.write_text("# nothing but a comment\n\n", encoding="utf-8")
        expect_raises(FixedListError, "comment-only list", validate_fixed_lists, comments_only, val_list)
    print("  ok: missing, empty and comment-only lists raise instead of falling back to directory splitting")


def test_duplicates_and_overlap_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, train_files, val_files = build_valid_pair(root)

        write_list(train_list, train_files + [train_files[0]])
        expect_raises(FixedListError, "duplicate inside the train list", validate_fixed_lists, train_list, val_list)

        write_list(train_list, train_files)
        write_list(val_list, val_files + [train_files[0]])
        expect_raises(FixedListError, "same file in both lists", validate_fixed_lists, train_list, val_list)

        # Same file name under two different directories: distinct paths, but
        # one video identity for augmentation angles and for evaluation.
        other_dir = root / "other_mount"
        twin = make_h5_files(other_dir, [0])
        write_list(val_list, val_files + twin)
        expect_raises(FixedListError, "same file name via another mount", validate_fixed_lists, train_list, val_list)
    print("  ok: in-list duplicates, train/val overlap and same-name-different-mount entries all rejected")


def test_missing_files_and_wrong_teacher_pattern_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, train_files, val_files = build_valid_pair(root)

        write_list(train_list, train_files + [root / "collected" / video_name(77)])
        expect_raises(FixedListError, "listed file that does not exist", validate_fixed_lists, train_list, val_list)

        old_teacher = make_h5_files(root / "collected", [78], teacher="bboxrank_v6_legacy")
        write_list(train_list, train_files + old_teacher)
        expect_raises(
            FixedListError,
            "teacher v6 file in the list",
            validate_fixed_lists,
            train_list,
            val_list,
            h5_pattern=H5_PATTERN,
        )
        # Without a pattern the same list passes the teacher check, so the
        # rejection above really came from the pattern and not from something else.
        validate_fixed_lists(train_list, val_list)
    print("  ok: non-existent entries and non-teacher-v7 file names are rejected")


def test_expected_total_is_enforced() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, _, _ = build_valid_pair(root, train_count=5, val_count=2)
        validate_fixed_lists(train_list, val_list, expected_total=7)
        expect_raises(
            FixedListError,
            "wrong total file count",
            validate_fixed_lists,
            train_list,
            val_list,
            expected_total=180,
        )
    print("  ok: the total file count is enforced when an expected total is given")


def test_extra_files_in_the_directory_do_not_change_the_target_set() -> None:
    """Report 8.7.1: the preflight must inspect the listed files, so files
    appearing in INPUT_DIR must not silently join the run."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, _, _ = build_valid_pair(root)
        before = validate_fixed_lists(train_list, val_list, h5_pattern=H5_PATTERN)
        manifest_before = write_combined_manifest(before, root / "manifest_before.txt")

        make_h5_files(root / "collected", [900, 901, 902])
        after = validate_fixed_lists(train_list, val_list, h5_pattern=H5_PATTERN)
        manifest_after = write_combined_manifest(after, root / "manifest_after.txt")

        assert before.all_paths == after.all_paths
        assert manifest_before.read_text() == manifest_after.read_text()
        assert len(after.all_paths) == 7
    print("  ok: adding 3 unlisted H5 files to the directory leaves the resolved target set identical")


def test_manifest_holds_train_then_val_in_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, train_files, val_files = build_valid_pair(root)
        inputs = validate_fixed_lists(train_list, val_list, h5_pattern=H5_PATTERN)
        manifest = write_combined_manifest(inputs, root / "manifest.txt")
        lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines == [str(p) for p in train_files + val_files]
    print(f"  ok: the preflight manifest lists all 7 files, train first, in list order")


def test_fingerprints_separate_content_from_path_representation() -> None:
    """Report 8.7.1: a list naming the same videos under a different prefix is
    not the same finding as a list naming different videos."""
    reference = [Path(f"/mnt/data/collected/{video_name(i)}") for i in range(4)]
    same_videos_other_prefix = [Path(f"/other/mount/collected/{video_name(i)}") for i in range(4)]
    reordered = [reference[1], reference[0], reference[2], reference[3]]
    different_videos = reference[:3] + [Path(f"/mnt/data/collected/{video_name(99)}")]

    identical = compare_file_lists(reference, reference)
    assert identical["content_identical"] and identical["representation_identical"]
    assert not identical["representation_differs_only"]

    prefixed = compare_file_lists(same_videos_other_prefix, reference)
    assert prefixed["content_identical"], "same videos in the same order must count as identical content"
    assert not prefixed["representation_identical"]
    assert prefixed["representation_differs_only"], "this is a representation difference, not a content one"
    assert prefixed["identity_sha256_actual"] == prefixed["identity_sha256_reference"]
    assert prefixed["content_sha256_actual"] != prefixed["content_sha256_reference"]

    swapped = compare_file_lists(reordered, reference)
    assert not swapped["content_identical"], "a different order is a content difference"

    changed = compare_file_lists(different_videos, reference)
    assert not changed["content_identical"]
    assert not changed["representation_differs_only"]
    assert changed["only_in_actual"] == [video_name(99)]
    assert changed["only_in_reference"] == [video_name(3)]

    assert list_content_sha256(reference) != list_identity_sha256(reference)
    print("  ok: prefix-only differences, reordering and changed videos are told apart from each other")


def test_comments_and_blank_lines_are_ignored_when_reading() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = make_h5_files(root / "collected", range(3))
        listing = root / "with_comments.txt"
        listing.write_text(
            "# saved by train_stage5.py\n"
            f"{files[0]}\n"
            "\n"
            f"  {files[1]}  \n"
            f"{files[2]}\n",
            encoding="utf-8",
        )
        assert read_file_list(listing, label="train") == files
    print("  ok: comments, blank lines and surrounding whitespace are ignored when reading a list")


def test_cli_writes_a_manifest_and_rejects_bad_input_with_exit_code_2() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, train_files, val_files = build_valid_pair(root)
        manifest = root / "cli_manifest.txt"
        result = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--train_list", str(train_list),
                "--val_list", str(val_list),
                "--h5_pattern", H5_PATTERN,
                "--expected_total", "7",
                "--manifest_out", str(manifest),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "train=5 val=2 total=7" in result.stdout
        assert manifest.read_text().splitlines() == [str(p) for p in train_files + val_files]

        write_list(train_list, train_files + [train_files[0]])
        rejected = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--train_list", str(train_list),
                "--val_list", str(val_list),
                "--manifest_out", str(root / "never_written.txt"),
            ],
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2, rejected
        assert "Fixed-list input rejected" in rejected.stderr
        assert not (root / "never_written.txt").exists(), "a rejected run must not leave a manifest behind"
    print("  ok: the CLI writes the manifest on success and exits 2 without a manifest on rejection")


def test_cli_verifies_expected_fingerprints_and_separates_the_two_failure_kinds() -> None:
    """P3 approval: the dry run must confirm the lists' content, order and hash,
    not merely display them."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_list, val_list, train_files, val_files = build_valid_pair(root)
        train_sha = list_content_sha256(train_files)
        val_sha = list_content_sha256(val_files)

        def run(extra: list[str], manifest_name: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [
                    sys.executable, str(MODULE_PATH),
                    "--train_list", str(train_list),
                    "--val_list", str(val_list),
                    "--manifest_out", str(root / manifest_name),
                    *extra,
                ],
                capture_output=True,
                text=True,
            )

        matched = run(["--expected_train_sha256", train_sha, "--expected_val_sha256", val_sha], "ok.txt")
        assert matched.returncode == 0, matched.stderr
        assert "fingerprints verified against the expected values" in matched.stdout
        assert (root / "ok.txt").exists()

        # Different videos: content differs, and the identity hash differs too.
        wrong = run(["--expected_train_sha256", "0" * 64], "never.txt")
        assert wrong.returncode == 2
        assert "train list fingerprint mismatch" in wrong.stderr
        assert not (root / "never.txt").exists(), "a rejected list must not leave a manifest behind"

        # Same videos, same order, different path spelling: the content hash
        # differs while the identity hash does not, and the message says so.
        # The files have to exist under the new prefix, otherwise the earlier
        # existence check fires first and this case is never reached.
        moved = make_h5_files(root / "other_mount", range(len(train_files)))
        assert [p.name for p in moved] == [p.name for p in train_files]
        assert list_identity_sha256(moved) == list_identity_sha256(train_files)
        write_list(train_list, moved)
        representation = subprocess.run(
            [
                sys.executable, str(MODULE_PATH),
                "--train_list", str(train_list),
                "--val_list", str(val_list),
                "--expected_train_sha256", train_sha,
            ],
            capture_output=True,
            text=True,
        )
        assert representation.returncode == 2
        assert "identity (file names only" in representation.stderr
    print("  ok: the CLI verifies fingerprints, refuses on mismatch, and names the identity hash for triage")


def main() -> None:
    tests = [
        test_valid_pair_is_accepted_in_list_order,
        test_cli_verifies_expected_fingerprints_and_separates_the_two_failure_kinds,
        test_one_sided_or_empty_or_unreadable_lists_stop_without_fallback,
        test_duplicates_and_overlap_are_rejected,
        test_missing_files_and_wrong_teacher_pattern_are_rejected,
        test_expected_total_is_enforced,
        test_extra_files_in_the_directory_do_not_change_the_target_set,
        test_manifest_holds_train_then_val_in_order,
        test_fingerprints_separate_content_from_path_representation,
        test_comments_and_blank_lines_are_ignored_when_reading,
        test_cli_writes_a_manifest_and_rejects_bad_input_with_exit_code_2,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nStage5 S5-15 fixed-list mode synthetic tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
