from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_SCHEMA_VERSION = 1
TEACHER_CONFIG_SCHEMA_VERSION = 1
ALLOWED_SPLITS = ("train", "validation", "test", "smoke")
REQUIRED_MANIFEST_COLUMNS = (
    "video_name",
    "pseudo3d_h5",
    "voc_xml_root",
    "split",
    "enabled",
)
OPTIONAL_MANIFEST_COLUMNS = ("notes",)
_VIDEO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Stage4SweepManifestItem:
    row_number: int
    video_name: str
    pseudo3d_h5: Path
    voc_xml_root: Path
    split: str
    enabled: bool
    notes: str = ""

    def strict_annotation_dir(self, teacher: "DenseTeacherConfig") -> Path:
        return (
            self.voc_xml_root
            / self.video_name
            / teacher.xml_annotation_dir_name
        )


@dataclass(frozen=True)
class DenseTeacherConfig:
    schema_version: int = TEACHER_CONFIG_SCHEMA_VERSION
    config_name: str = "stage4_dense_teacher_v1"

    label_ignore: int = -1
    label_background: int = 0
    label_positive: int = 1
    no_bbox_label: int = -1
    bbox_inside_non_contour_label: str = "ignore"

    xml_frame_number_offsets: tuple[int, ...] = (1,)
    xml_frame_id_source: str = "frame_index"
    xml_annotation_dir_name: str = "annotations_renamed"
    strict_xml_annotation_dir: bool = True

    contour_preset: str = "percentile85_area100"
    contour_threshold_mode: str = "percentile"
    contour_percentile: float = 85.0
    contour_min_component_area: int = 100
    contour_min_area: float = 20.0
    contour_min_alpha: int = 1
    contour_open_ksize: int = 3
    contour_close_ksize: int = 5
    contour_morph_shape: str = "ellipse"
    contour_denoise: str = "median"
    contour_denoise_ksize: int = 3

    enable_contour_fallback: bool = False

    def validate(self) -> None:
        if self.schema_version != TEACHER_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported teacher config schema_version: "
                f"{self.schema_version}"
            )
        if not self.config_name.strip():
            raise ValueError("teacher config_name must not be empty")
        if (
            self.label_ignore,
            self.label_background,
            self.label_positive,
        ) != (-1, 0, 1):
            raise ValueError("Teacher labels must be ignore=-1, background=0, positive=1")
        if self.no_bbox_label != self.label_ignore:
            raise ValueError("no_bbox_label must be the ignore label")
        if self.bbox_inside_non_contour_label != "ignore":
            raise ValueError("BBox pixels outside the teacher contour must be ignore")
        if not self.xml_frame_number_offsets:
            raise ValueError("xml_frame_number_offsets must not be empty")
        if len(set(self.xml_frame_number_offsets)) != len(
            self.xml_frame_number_offsets
        ):
            raise ValueError("xml_frame_number_offsets must not contain duplicates")
        if self.xml_frame_id_source not in {"frame_index", "frame_order", "both"}:
            raise ValueError(
                "xml_frame_id_source must be frame_index, frame_order, or both"
            )
        annotation_dir = Path(self.xml_annotation_dir_name)
        if (
            not self.xml_annotation_dir_name
            or annotation_dir.is_absolute()
            or len(annotation_dir.parts) != 1
            or annotation_dir.name in {".", ".."}
        ):
            raise ValueError(
                "xml_annotation_dir_name must be one relative directory name"
            )
        if not self.strict_xml_annotation_dir:
            raise ValueError(
                "Dense teacher generation requires strict_xml_annotation_dir=true"
            )
        if self.enable_contour_fallback:
            raise ValueError(
                "Dense teacher generation must not enable sampling-dependent "
                "contour fallback"
            )
        if self.contour_threshold_mode not in {
            "fixed",
            "otsu",
            "percentile",
            "adaptive",
        }:
            raise ValueError(
                f"Unknown contour_threshold_mode: {self.contour_threshold_mode}"
            )
        if not 0.0 <= float(self.contour_percentile) <= 100.0:
            raise ValueError("contour_percentile must be within [0, 100]")
        if int(self.contour_min_component_area) < 0:
            raise ValueError("contour_min_component_area must be >= 0")
        if float(self.contour_min_area) < 0.0:
            raise ValueError("contour_min_area must be >= 0")
        if not 0 <= int(self.contour_min_alpha) <= 255:
            raise ValueError("contour_min_alpha must be within [0, 255]")
        for name, value in (
            ("contour_open_ksize", self.contour_open_ksize),
            ("contour_close_ksize", self.contour_close_ksize),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.contour_morph_shape not in {"rect", "ellipse", "cross"}:
            raise ValueError(
                f"Unknown contour_morph_shape: {self.contour_morph_shape}"
            )
        if self.contour_denoise not in {
            "none",
            "median",
            "gaussian",
            "bilateral",
        }:
            raise ValueError(f"Unknown contour_denoise: {self.contour_denoise}")
        if int(self.contour_denoise_ksize) <= 0:
            raise ValueError("contour_denoise_ksize must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "config_name": self.config_name,
            "labels": {
                "ignore": int(self.label_ignore),
                "background": int(self.label_background),
                "positive": int(self.label_positive),
                "no_bbox_label": int(self.no_bbox_label),
                "bbox_inside_non_contour_label": (
                    self.bbox_inside_non_contour_label
                ),
            },
            "xml": {
                "frame_number_offsets": [
                    int(value) for value in self.xml_frame_number_offsets
                ],
                "frame_id_source": self.xml_frame_id_source,
                "annotation_dir_name": self.xml_annotation_dir_name,
                "strict_annotation_dir": bool(self.strict_xml_annotation_dir),
            },
            "contour": {
                "preset": self.contour_preset,
                "threshold_mode": self.contour_threshold_mode,
                "percentile": float(self.contour_percentile),
                "min_component_area": int(self.contour_min_component_area),
                "min_area": float(self.contour_min_area),
                "min_alpha": int(self.contour_min_alpha),
                "open_ksize": int(self.contour_open_ksize),
                "close_ksize": int(self.contour_close_ksize),
                "morph_shape": self.contour_morph_shape,
                "denoise": self.contour_denoise,
                "denoise_ksize": int(self.contour_denoise_ksize),
            },
            "fallback": {
                "enabled": bool(self.enable_contour_fallback),
            },
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"teacher config '{name}' must be a mapping")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"teacher config '{name}' must be true or false")
    return value


def dense_teacher_config_from_mapping(
    raw: Mapping[str, Any],
) -> DenseTeacherConfig:
    labels = _require_mapping(raw.get("labels", {}), "labels")
    xml = _require_mapping(raw.get("xml", {}), "xml")
    contour = _require_mapping(raw.get("contour", {}), "contour")
    fallback = _require_mapping(raw.get("fallback", {}), "fallback")

    offsets_raw = xml.get("frame_number_offsets", [1])
    if not isinstance(offsets_raw, (list, tuple)):
        raise ValueError("xml.frame_number_offsets must be a list")

    config = DenseTeacherConfig(
        schema_version=int(raw.get("schema_version", 0)),
        config_name=str(raw.get("config_name", "")),
        label_ignore=int(labels.get("ignore", -1)),
        label_background=int(labels.get("background", 0)),
        label_positive=int(labels.get("positive", 1)),
        no_bbox_label=int(labels.get("no_bbox_label", -1)),
        bbox_inside_non_contour_label=str(
            labels.get("bbox_inside_non_contour_label", "ignore")
        ),
        xml_frame_number_offsets=tuple(int(value) for value in offsets_raw),
        xml_frame_id_source=str(xml.get("frame_id_source", "frame_index")),
        xml_annotation_dir_name=str(
            xml.get("annotation_dir_name", "annotations_renamed")
        ),
        strict_xml_annotation_dir=_require_bool(
            xml.get("strict_annotation_dir", True),
            "xml.strict_annotation_dir",
        ),
        contour_preset=str(contour.get("preset", "percentile85_area100")),
        contour_threshold_mode=str(contour.get("threshold_mode", "percentile")),
        contour_percentile=float(contour.get("percentile", 85.0)),
        contour_min_component_area=int(contour.get("min_component_area", 100)),
        contour_min_area=float(contour.get("min_area", 20.0)),
        contour_min_alpha=int(contour.get("min_alpha", 1)),
        contour_open_ksize=int(contour.get("open_ksize", 3)),
        contour_close_ksize=int(contour.get("close_ksize", 5)),
        contour_morph_shape=str(contour.get("morph_shape", "ellipse")),
        contour_denoise=str(contour.get("denoise", "median")),
        contour_denoise_ksize=int(contour.get("denoise_ksize", 3)),
        enable_contour_fallback=_require_bool(
            fallback.get("enabled", False),
            "fallback.enabled",
        ),
    )
    config.validate()
    return config


def load_dense_teacher_config(path: Path) -> DenseTeacherConfig:
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Teacher config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Teacher config root must be a mapping: {path}")
    return dense_teacher_config_from_mapping(raw)


def _parse_enabled(value: str, *, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Manifest row {row_number}: enabled must be true/false, got {value!r}"
    )


def _resolve_manifest_path(value: str, *, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve(strict=False)


def load_stage4_sweep_manifest(path: Path) -> list[Stage4SweepManifestItem]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sweep manifest not found: {path}")

    items: list[Stage4SweepManifestItem] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Sweep manifest has no header: {path}")
        missing = [
            column
            for column in REQUIRED_MANIFEST_COLUMNS
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"Sweep manifest missing required columns {missing}: {path}"
            )

        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            video_name = (row.get("video_name") or "").strip()
            pseudo3d_h5 = (row.get("pseudo3d_h5") or "").strip()
            voc_xml_root = (row.get("voc_xml_root") or "").strip()
            split = (row.get("split") or "").strip().lower()
            enabled_raw = (row.get("enabled") or "").strip()
            notes = (row.get("notes") or "").strip()

            if not video_name or not _VIDEO_NAME_PATTERN.fullmatch(video_name):
                raise ValueError(
                    f"Manifest row {row_number}: invalid video_name {video_name!r}"
                )
            if not pseudo3d_h5:
                raise ValueError(
                    f"Manifest row {row_number}: pseudo3d_h5 must not be empty"
                )
            if not voc_xml_root:
                raise ValueError(
                    f"Manifest row {row_number}: voc_xml_root must not be empty"
                )
            if split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"Manifest row {row_number}: split must be one of "
                    f"{ALLOWED_SPLITS}, got {split!r}"
                )
            enabled = _parse_enabled(enabled_raw, row_number=row_number)
            items.append(
                Stage4SweepManifestItem(
                    row_number=row_number,
                    video_name=video_name,
                    pseudo3d_h5=_resolve_manifest_path(
                        pseudo3d_h5,
                        manifest_dir=path.parent,
                    ),
                    voc_xml_root=_resolve_manifest_path(
                        voc_xml_root,
                        manifest_dir=path.parent,
                    ),
                    split=split,
                    enabled=enabled,
                    notes=notes,
                )
            )

    if not items:
        raise ValueError(f"Sweep manifest contains no data rows: {path}")

    duplicate_videos = _duplicates(item.video_name for item in items)
    if duplicate_videos:
        raise ValueError(
            "Sweep manifest contains duplicate video_name values: "
            f"{sorted(duplicate_videos)}"
        )
    duplicate_h5 = _duplicates(str(item.pseudo3d_h5) for item in items)
    if duplicate_h5:
        raise ValueError(
            "Sweep manifest contains duplicate pseudo3d_h5 values: "
            f"{sorted(duplicate_h5)}"
        )
    if not any(item.enabled for item in items):
        raise ValueError("Sweep manifest has no enabled rows")
    return items


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
