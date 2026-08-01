from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ALLOWED_SPLITS = ("train", "validation", "test", "smoke")
FIELDNAMES = (
    "video_name",
    "pseudo3d_h5",
    "voc_xml_root",
    "split",
    "enabled",
    "notes",
)


@dataclass(frozen=True)
class ManifestBuildRow:
    video_name: str
    pseudo3d_h5: Path
    voc_xml_root: Path
    split: str
    enabled: bool
    notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "video_name": self.video_name,
            "pseudo3d_h5": str(self.pseudo3d_h5.resolve()),
            "voc_xml_root": str(self.voc_xml_root.resolve()),
            "split": self.split,
            "enabled": "true" if self.enabled else "false",
            "notes": self.notes,
        }


def _video_name_from_h5(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix:
        if not stem.endswith(suffix):
            raise ValueError(
                f"H5 stem does not end with --video_suffix_to_strip={suffix!r}: "
                f"{path.name}"
            )
        stem = stem[: -len(suffix)]
    if not stem:
        raise ValueError(f"Empty video_name derived from H5: {path}")
    return stem


def build_manifest_rows(
    *,
    h5_dir: Path,
    h5_pattern: str,
    video_suffix_to_strip: str,
    voc_xml_root: Path,
    annotation_dir_name: str,
    split: str,
) -> list[ManifestBuildRow]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unknown split={split!r}; expected one of {ALLOWED_SPLITS}")
    if not h5_dir.is_dir():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    if not voc_xml_root.is_dir():
        raise FileNotFoundError(f"VOC XML root not found: {voc_xml_root}")
    annotation_part = Path(annotation_dir_name)
    if (
        not annotation_dir_name
        or annotation_part.is_absolute()
        or len(annotation_part.parts) != 1
        or annotation_part.name in {".", ".."}
    ):
        raise ValueError("annotation_dir_name must be one relative directory name")

    h5_paths = sorted(path for path in h5_dir.glob(h5_pattern) if path.is_file())
    if not h5_paths:
        raise ValueError(f"No H5 files matched {h5_dir / h5_pattern}")

    rows: list[ManifestBuildRow] = []
    seen: dict[str, Path] = {}
    for h5_path in h5_paths:
        video_name = _video_name_from_h5(h5_path, video_suffix_to_strip)
        if video_name in seen:
            raise ValueError(
                f"Multiple H5 files map to video_name={video_name!r}: "
                f"{seen[video_name]} and {h5_path}"
            )
        seen[video_name] = h5_path

        annotation_dir = voc_xml_root / video_name / annotation_dir_name
        xml_count = (
            sum(1 for path in annotation_dir.glob("*.xml") if path.is_file())
            if annotation_dir.is_dir()
            else 0
        )
        enabled = xml_count > 0
        notes = (
            f"strict_xml_files={xml_count}"
            if enabled
            else f"disabled: no strict XML in {annotation_dir}"
        )
        rows.append(
            ManifestBuildRow(
                video_name=video_name,
                pseudo3d_h5=h5_path,
                voc_xml_root=voc_xml_root,
                split=split,
                enabled=enabled,
                notes=notes,
            )
        )

    if not any(row.enabled for row in rows):
        raise ValueError(
            "No H5 video has a non-empty strict annotation directory under "
            f"{voc_xml_root}"
        )
    return rows


def write_manifest(path: Path, rows: list[ManifestBuildRow], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output manifest already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an explicit Stage 4 sampling-sweep manifest from a pseudo3D "
            "H5 directory. The split must be supplied; it is never inferred."
        )
    )
    parser.add_argument("--h5_dir", type=Path, required=True)
    parser.add_argument(
        "--h5_pattern",
        default="*_ts448_oym96_corr.h5",
    )
    parser.add_argument(
        "--video_suffix_to_strip",
        default="_ts448_oym96_corr",
    )
    parser.add_argument("--voc_xml_root", type=Path, required=True)
    parser.add_argument(
        "--annotation_dir_name",
        default="annotations_renamed",
    )
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_manifest_rows(
        h5_dir=args.h5_dir,
        h5_pattern=args.h5_pattern,
        video_suffix_to_strip=args.video_suffix_to_strip,
        voc_xml_root=args.voc_xml_root,
        annotation_dir_name=args.annotation_dir_name,
        split=args.split,
    )
    write_manifest(args.output_manifest, rows, overwrite=args.overwrite)

    enabled = sum(row.enabled for row in rows)
    print("Stage 4 sampling sweep manifest built")
    print(f"h5_dir          : {args.h5_dir}")
    print(f"h5_pattern      : {args.h5_pattern}")
    print(f"voc_xml_root    : {args.voc_xml_root}")
    print(f"split           : {args.split}")
    print(f"output_manifest : {args.output_manifest}")
    print(f"rows            : {len(rows)}")
    print(f"enabled         : {enabled}")
    print(f"disabled        : {len(rows) - enabled}")
    print(
        "Run validate_stage4_sampling_manifest.py before starting the "
        "parameter sweep."
    )


if __name__ == "__main__":
    main()
