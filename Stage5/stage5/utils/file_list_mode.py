from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# S5-15 P2 Step 3: fixed train/val file-list inputs.
#
# S5-15 must train on the exact saved W-A lists, never on a directory scan that
# is re-split by fraction and seed. This module validates a (train_list,
# val_list) pair and emits the combined manifest that train_stage5.sh's teacher
# preflight then inspects, so the preflight examines precisely the files that
# will be trained on rather than whatever a glob happens to match.
#
# Deliberately stdlib-only (no numpy/h5py/torch): it runs before any training
# process starts, and its whole job is to stop early and loudly rather than
# fall back to directory splitting.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedListInputs:
    train_paths: tuple[Path, ...]
    val_paths: tuple[Path, ...]

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return self.train_paths + self.val_paths


class FixedListError(ValueError):
    """Raised for any condition that must stop the run instead of falling back."""


def read_file_list(path: str | Path, *, label: str) -> list[Path]:
    list_path = Path(path)
    if not list_path.is_file():
        raise FixedListError(f"{label} file list not found: {list_path}")
    try:
        text = list_path.read_text(encoding="utf-8")
    except OSError as error:
        raise FixedListError(f"{label} file list could not be read: {list_path} ({error})") from error

    entries: list[Path] = []
    for line in text.splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        entries.append(Path(token))
    if not entries:
        raise FixedListError(f"{label} file list is empty: {list_path}")
    return entries


def _find_duplicates(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def validate_fixed_lists(
    train_list: str | Path,
    val_list: str | Path,
    *,
    h5_pattern: str | None = None,
    require_existing: bool = True,
    expected_total: int | None = None,
) -> FixedListInputs:
    """Validate the pair of lists, preserving their order exactly.

    Every failure raises instead of falling back to directory splitting: a
    silently different split is the failure mode this whole mode exists to
    prevent.
    """
    train_paths = read_file_list(train_list, label="train")
    val_paths = read_file_list(val_list, label="val")

    for label, paths in (("train", train_paths), ("val", val_paths)):
        duplicates = _find_duplicates(paths)
        if duplicates:
            raise FixedListError(f"duplicate entries in the {label} file list: {duplicates}")

    overlap = sorted({str(p) for p in train_paths} & {str(p) for p in val_paths})
    if overlap:
        raise FixedListError(f"the same H5 appears in both the train and val file lists: {overlap}")

    # File names must be unique across the union too: the augmentation angle is
    # derived from the file name, and evaluation identifies videos by name.
    name_duplicates = _find_duplicates([Path(path.name) for path in train_paths + val_paths])
    if name_duplicates:
        raise FixedListError(f"duplicate H5 file name(s) across the train/val lists: {name_duplicates}")

    if h5_pattern:
        mismatched = [str(p) for p in train_paths + val_paths if not fnmatch.fnmatch(p.name, h5_pattern)]
        if mismatched:
            raise FixedListError(
                f"{len(mismatched)} listed file(s) do not match the expected teacher pattern "
                f"{h5_pattern!r}; first: {mismatched[:3]}"
            )

    if require_existing:
        missing = [str(p) for p in train_paths + val_paths if not p.is_file()]
        if missing:
            raise FixedListError(f"{len(missing)} listed H5 file(s) do not exist; first: {missing[:3]}")

    if expected_total is not None and len(train_paths) + len(val_paths) != expected_total:
        raise FixedListError(
            f"file lists hold {len(train_paths)} + {len(val_paths)} = "
            f"{len(train_paths) + len(val_paths)} files, expected {expected_total}"
        )

    return FixedListInputs(train_paths=tuple(train_paths), val_paths=tuple(val_paths))


def write_combined_manifest(inputs: FixedListInputs, path: str | Path) -> Path:
    """Write train entries then val entries, in list order, for the preflight."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(str(p) for p in inputs.all_paths) + "\n", encoding="utf-8")
    return manifest_path


def list_content_sha256(paths: list[Path] | tuple[Path, ...]) -> str:
    """Fingerprint of the list exactly as written, including path spelling."""
    return hashlib.sha256("\n".join(str(p) for p in paths).encode("utf-8")).hexdigest()


def list_identity_sha256(paths: list[Path] | tuple[Path, ...]) -> str:
    """Fingerprint of which videos are listed, in order, ignoring path spelling."""
    return hashlib.sha256("\n".join(Path(p).name for p in paths).encode("utf-8")).hexdigest()


def compare_file_lists(actual: list[Path] | tuple[Path, ...], reference: list[Path] | tuple[Path, ...]) -> dict:
    """Separate a content difference from a mere path-representation difference.

    A run whose saved list names the same videos in the same order but with a
    different prefix is not the same thing as a run trained on different files,
    and report section 8.7.1 requires the two to be reported apart rather than
    the check being relaxed until it passes.
    """
    actual_names = [Path(p).name for p in actual]
    reference_names = [Path(p).name for p in reference]
    content_identical = actual_names == reference_names
    representation_identical = [str(p) for p in actual] == [str(p) for p in reference]
    return {
        "content_identical": content_identical,
        "representation_identical": representation_identical,
        "representation_differs_only": content_identical and not representation_identical,
        "actual_count": len(actual_names),
        "reference_count": len(reference_names),
        "content_sha256_actual": list_content_sha256(list(actual)),
        "content_sha256_reference": list_content_sha256(list(reference)),
        "identity_sha256_actual": list_identity_sha256(list(actual)),
        "identity_sha256_reference": list_identity_sha256(list(reference)),
        "only_in_actual": sorted(set(actual_names) - set(reference_names))[:5],
        "only_in_reference": sorted(set(reference_names) - set(actual_names))[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an S5-15 fixed train/val H5 file-list pair and emit the combined "
        "manifest that train_stage5.sh's teacher preflight inspects."
    )
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--h5_pattern", default=None, help="Expected teacher file-name pattern")
    parser.add_argument("--expected_total", type=int, default=None)
    parser.add_argument("--manifest_out", default=None, help="Where to write the combined manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        inputs = validate_fixed_lists(
            args.train_list,
            args.val_list,
            h5_pattern=args.h5_pattern,
            expected_total=args.expected_total,
        )
    except FixedListError as error:
        print(f"Fixed-list input rejected: {error}", file=sys.stderr)
        raise SystemExit(2)

    if args.manifest_out:
        write_combined_manifest(inputs, args.manifest_out)

    print(
        "Fixed-list inputs validated: "
        f"train={len(inputs.train_paths)} val={len(inputs.val_paths)} "
        f"total={len(inputs.all_paths)} "
        f"train_sha256={list_content_sha256(list(inputs.train_paths))[:16]} "
        f"val_sha256={list_content_sha256(list(inputs.val_paths))[:16]}"
    )


if __name__ == "__main__":
    main()
