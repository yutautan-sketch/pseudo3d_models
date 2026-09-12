from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FORMAT_NAME = "Segmentation Mask 1.1"
FORMAT_SCHEMA_VERSION = 1
BACKGROUND_INDEX = 0
FOREGROUND_INDEX = 1
DEFAULT_BACKGROUND_VALUE = 0
DEFAULT_FOREGROUND_VALUE = 255
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_MASK_EXTENSIONS = {".png"}
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_FILE_MODE = 0o100644
ALLOWED_DIRECTORY_MEMBERS = {
    "ImageSets/",
    "ImageSets/Segmentation/",
    "SegmentationClass/",
    "SegmentationClass/images/",
    "SegmentationObject/",
    "SegmentationObject/images/",
}
LABELMAP_PATH = "labelmap.txt"
CVAT_LABELMAP_HEADER = "# label:color_rgb:parts:actions"
DEFAULT_SET_PATH = "ImageSets/Segmentation/default.txt"
CLASS_PREFIX = "SegmentationClass/"
OBJECT_PREFIX = "SegmentationObject/"
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class CvatMaskError(ValueError):
    """Raised when input files or a Segmentation Mask ZIP violate the contract."""


@dataclass(frozen=True)
class InputPair:
    stem: str
    image_path: Path
    mask_path: Path
    image_shape_hw: tuple[int, int]
    normalized_mask: np.ndarray


@dataclass(frozen=True)
class ZipValidationSummary:
    schema_version: int
    format_name: str
    status: str
    input_zip: str
    zip_sha256: str
    zip_members: int
    images: int
    masks: int
    classes: int
    label_name: str
    background_index: int
    foreground_index: int
    unique_indices: list[int]
    stems: list[str]
    size_mismatches: int
    missing_masks: int
    missing_mask_stems: list[str]
    extra_masks: int
    reference_masks_compared: int
    reference_pixel_mismatches: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label_name(label_name: str) -> str:
    value = str(label_name)
    if value != value.strip() or not value:
        raise CvatMaskError("label_name must be non-empty without outer whitespace")
    if len(value) > 128:
        raise CvatMaskError("label_name must contain at most 128 characters")
    if value.casefold() == "background":
        raise CvatMaskError("foreground label_name must not be 'background'")
    if any(character in value for character in (":", "\r", "\n", "/", "\\", "\x00")):
        raise CvatMaskError(
            "label_name must not contain colon, newline, path separators, or NUL"
        )
    if not value.isprintable():
        raise CvatMaskError("label_name must contain printable characters only")
    return value


def make_labelmap_text(label_name: str) -> str:
    label_name = validate_label_name(label_name)
    return f"background:0,0,0::\n{label_name}:255,0,0::\n"


def _parse_labelmap_line(line: str, expected_name: str) -> tuple[int, int, int]:
    fields = line.split(":")
    if len(fields) != 4 or fields[0] != expected_name or fields[2:] != ["", ""]:
        raise CvatMaskError(
            "labelmap.txt differs from the fixed background/femur contract"
        )
    components = fields[1].split(",")
    if len(components) != 3:
        raise CvatMaskError("labelmap.txt contains an invalid RGB color")
    try:
        rgb = tuple(int(component) for component in components)
    except ValueError as exc:
        raise CvatMaskError("labelmap.txt contains an invalid RGB color") from exc
    if any(component < 0 or component > 255 for component in rgb):
        raise CvatMaskError("labelmap.txt RGB color is outside [0, 255]")
    return rgb  # type: ignore[return-value]


def validate_labelmap_payload(
    payload: bytes, label_name: str
) -> tuple[int, int, int]:
    """Validate the two-label contract and return CVAT's femur RGB color.

    CVAT preserves label names when importing a Segmentation Mask archive, but
    a task created without an explicit color can be exported with a different
    foreground RGB value. The semantic contract therefore fixes the two label
    names/order and black background while accepting one non-black femur color.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CvatMaskError("labelmap.txt must be UTF-8") from exc
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise CvatMaskError("labelmap.txt contains unsupported line endings")
    lines = normalized.splitlines()
    if lines and lines[0] == CVAT_LABELMAP_HEADER:
        lines = lines[1:]
    if len(lines) != 2:
        raise CvatMaskError(
            "labelmap.txt differs from the fixed background/femur contract"
        )
    background_rgb = _parse_labelmap_line(lines[0], "background")
    foreground_rgb = _parse_labelmap_line(lines[1], validate_label_name(label_name))
    if background_rgb != (0, 0, 0) or foreground_rgb == background_rgb:
        raise CvatMaskError(
            "labelmap.txt differs from the fixed background/femur contract"
        )
    return foreground_rgb


def _ensure_plain_directory(path: Path, name: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise CvatMaskError(f"{name} must not be a symlink: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} not found or not a directory: {path}")
    return path.resolve()


def _scan_flat_files(
    directory: Path,
    *,
    allowed_extensions: set[str],
    kind: str,
) -> dict[str, Path]:
    directory = _ensure_plain_directory(directory, f"{kind}_dir")
    result: dict[str, Path] = {}
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    if not entries:
        raise CvatMaskError(f"{kind}_dir is empty: {directory}")
    for entry in entries:
        if entry.is_symlink():
            raise CvatMaskError(f"{kind} symlink is not supported: {entry}")
        if entry.is_dir():
            raise CvatMaskError(f"Nested {kind} directory is not supported: {entry}")
        if not entry.is_file():
            raise CvatMaskError(f"Unsupported {kind} entry: {entry}")
        suffix = entry.suffix.lower()
        if suffix not in allowed_extensions:
            raise CvatMaskError(
                f"Unsupported {kind} extension {entry.suffix!r}: {entry}"
            )
        stem = entry.stem
        if not stem or stem in {".", ".."}:
            raise CvatMaskError(f"Invalid empty {kind} stem: {entry}")
        if any(character in stem for character in (":", "/", "\\", "\r", "\n", "\x00")):
            raise CvatMaskError(f"Unsafe {kind} stem {stem!r}: {entry}")
        if stem in result:
            raise CvatMaskError(
                f"Duplicate {kind} stem {stem!r}: {result[stem]} and {entry}"
            )
        result[stem] = entry.resolve()
    return result


def scan_images(images_dir: Path) -> dict[str, Path]:
    return _scan_flat_files(
        images_dir,
        allowed_extensions=SUPPORTED_IMAGE_EXTENSIONS,
        kind="image",
    )


def scan_masks(masks_dir: Path) -> dict[str, Path]:
    return _scan_flat_files(
        masks_dir,
        allowed_extensions=SUPPORTED_MASK_EXTENSIONS,
        kind="mask",
    )


def _read_image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise CvatMaskError(f"Failed to decode image: {path}")
    if image.ndim not in {2, 3}:
        raise CvatMaskError(f"Unsupported image dimensionality {image.shape}: {path}")
    height, width = int(image.shape[0]), int(image.shape[1])
    if height <= 0 or width <= 0:
        raise CvatMaskError(f"Image has an empty spatial dimension: {path}")
    return height, width


def normalize_binary_mask(
    mask: np.ndarray,
    *,
    background_value: int = DEFAULT_BACKGROUND_VALUE,
    foreground_value: int = DEFAULT_FOREGROUND_VALUE,
    source: str = "mask",
) -> np.ndarray:
    if not 0 <= int(background_value) <= 255:
        raise CvatMaskError("background_value must be in [0, 255]")
    if not 0 <= int(foreground_value) <= 255:
        raise CvatMaskError("foreground_value must be in [0, 255]")
    if int(background_value) == int(foreground_value):
        raise CvatMaskError("background_value and foreground_value must differ")
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise CvatMaskError(f"Mask must be 2-D grayscale, got {mask.shape}: {source}")
    if mask.dtype != np.uint8:
        raise CvatMaskError(f"Mask must have dtype uint8, got {mask.dtype}: {source}")
    unique = set(int(value) for value in np.unique(mask))
    allowed = {int(background_value), int(foreground_value)}
    unexpected = sorted(unique - allowed)
    if unexpected:
        raise CvatMaskError(
            f"Mask contains values outside {sorted(allowed)}: {unexpected}: {source}"
        )
    normalized = np.zeros(mask.shape, dtype=np.uint8)
    normalized[mask == int(foreground_value)] = FOREGROUND_INDEX
    return normalized


def _read_normalized_mask(
    path: Path,
    *,
    background_value: int,
    foreground_value: int,
) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise CvatMaskError(f"Failed to decode mask: {path}")
    return normalize_binary_mask(
        mask,
        background_value=background_value,
        foreground_value=foreground_value,
        source=str(path),
    )


def collect_input_pairs(
    *,
    images_dir: Path,
    masks_dir: Path,
    background_value: int = DEFAULT_BACKGROUND_VALUE,
    foreground_value: int = DEFAULT_FOREGROUND_VALUE,
) -> list[InputPair]:
    images = scan_images(images_dir)
    masks = scan_masks(masks_dir)
    image_stems = set(images)
    mask_stems = set(masks)
    missing = sorted(image_stems - mask_stems)
    extra = sorted(mask_stems - image_stems)
    if missing or extra:
        raise CvatMaskError(
            f"Image/mask stem mismatch: missing_masks={missing}, extra_masks={extra}"
        )
    pairs: list[InputPair] = []
    for stem in sorted(image_stems):
        image_shape = _read_image_shape(images[stem])
        mask = _read_normalized_mask(
            masks[stem],
            background_value=background_value,
            foreground_value=foreground_value,
        )
        if mask.shape != image_shape:
            raise CvatMaskError(
                f"Image/mask size mismatch for {stem}: image={image_shape}, mask={mask.shape}"
            )
        pairs.append(
            InputPair(
                stem=stem,
                image_path=images[stem],
                mask_path=masks[stem],
                image_shape_hw=image_shape,
                normalized_mask=mask,
            )
        )
    return pairs


def encode_indexed_png(mask: np.ndarray) -> bytes:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise CvatMaskError(
            f"Indexed output mask must be 2-D uint8, got {mask.shape} {mask.dtype}"
        )
    success, encoded = cv2.imencode(
        ".png",
        mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    if not success:
        raise CvatMaskError("OpenCV failed to encode indexed PNG")
    return encoded.tobytes()


def _member_payloads(
    pairs: Sequence[InputPair],
    *,
    label_name: str,
) -> list[tuple[str, bytes]]:
    stems = [pair.stem for pair in pairs]
    payloads: list[tuple[str, bytes]] = [
        (LABELMAP_PATH, make_labelmap_text(label_name).encode("utf-8")),
        (DEFAULT_SET_PATH, ("\n".join(stems) + "\n").encode("utf-8")),
    ]
    png_by_stem = {pair.stem: encode_indexed_png(pair.normalized_mask) for pair in pairs}
    payloads.extend(
        (f"{CLASS_PREFIX}{stem}.png", png_by_stem[stem]) for stem in stems
    )
    payloads.extend(
        (f"{OBJECT_PREFIX}{stem}.png", png_by_stem[stem]) for stem in stems
    )
    return payloads


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = FIXED_FILE_MODE << 16
    info.flag_bits = 0
    return info


def write_deterministic_zip(path: Path, payloads: Sequence[tuple[str, bytes]]) -> None:
    names = [name for name, _ in payloads]
    if len(names) != len(set(names)):
        raise CvatMaskError("Internal error: duplicate output ZIP member")
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        for name, payload in payloads:
            archive.writestr(
                _zip_info(name),
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _safe_member_name(name: str) -> PurePosixPath:
    raw_parts = name.split("/")
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or _WINDOWS_DRIVE.match(name)
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise CvatMaskError(f"Unsafe ZIP member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CvatMaskError(f"Unsafe ZIP member path: {name!r}")
    return path


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_member_bytes: int,
) -> bytes:
    if info.file_size > max_member_bytes:
        raise CvatMaskError(
            f"ZIP member exceeds size limit {max_member_bytes}: {info.filename}"
        )
    payload = archive.read(info)
    if len(payload) != info.file_size:
        raise CvatMaskError(f"ZIP member size changed while reading: {info.filename}")
    return payload


def _decode_indexed_png(
    payload: bytes,
    source: str,
    *,
    foreground_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise CvatMaskError(f"Failed to decode ZIP mask PNG: {source}")
    if mask.dtype != np.uint8 or mask.ndim not in {2, 3}:
        raise CvatMaskError(
            f"ZIP class mask must be 2-D or 3-channel uint8, "
            f"got {mask.shape} {mask.dtype}: {source}"
        )
    if mask.ndim == 3:
        if mask.shape[2] != 3:
            raise CvatMaskError(
                f"ZIP RGB class mask must have 3 channels, got {mask.shape}: {source}"
            )
        background_bgr = np.array([0, 0, 0], dtype=np.uint8)
        foreground_bgr = np.array(foreground_rgb[::-1], dtype=np.uint8)
        is_background = np.all(mask == background_bgr, axis=2)
        is_foreground = np.all(mask == foreground_bgr, axis=2)
        if not np.all(is_background | is_foreground):
            colors_bgr = np.unique(mask.reshape(-1, 3), axis=0)
            unexpected_rgb = [
                tuple(int(value) for value in color[::-1])
                for color in colors_bgr
                if not (
                    np.array_equal(color, background_bgr)
                    or np.array_equal(color, foreground_bgr)
                )
            ]
            raise CvatMaskError(
                f"ZIP RGB class mask contains colors outside labelmap "
                f"{unexpected_rgb}: {source}"
            )
        return is_foreground.astype(np.uint8)
    unexpected = sorted(
        set(int(value) for value in np.unique(mask))
        - {BACKGROUND_INDEX, FOREGROUND_INDEX}
    )
    if unexpected:
        raise CvatMaskError(f"ZIP mask contains unknown indices {unexpected}: {source}")
    return mask


def _decode_object_support_png(payload: bytes, source: str) -> np.ndarray:
    """Decode an instance mask to binary support without interpreting instance IDs."""
    encoded = np.frombuffer(payload, dtype=np.uint8)
    mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise CvatMaskError(f"Failed to decode ZIP object mask PNG: {source}")
    if mask.dtype != np.uint8 or mask.ndim not in {2, 3}:
        raise CvatMaskError(
            f"ZIP object mask must be 2-D or 3-channel uint8, "
            f"got {mask.shape} {mask.dtype}: {source}"
        )
    if mask.ndim == 3:
        if mask.shape[2] != 3:
            raise CvatMaskError(
                f"ZIP RGB object mask must have 3 channels, got {mask.shape}: {source}"
            )
        return np.any(mask != 0, axis=2).astype(np.uint8)
    return (mask != 0).astype(np.uint8)


def _parse_default_stems(payload: bytes) -> tuple[list[str], str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CvatMaskError("default.txt is not valid UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise CvatMaskError("default.txt contains no stems")
    if any(not line or line != line.strip() for line in lines):
        raise CvatMaskError("default.txt contains an empty or whitespace-padded stem")
    if len(lines) != len(set(lines)):
        raise CvatMaskError("default.txt contains duplicate stems")
    if lines != sorted(lines):
        raise CvatMaskError("default.txt stems are not lexicographically sorted")
    prefixed = [line.startswith("images/") for line in lines]
    if any(prefixed) and not all(prefixed):
        raise CvatMaskError("default.txt mixes flat and images/-prefixed stems")
    member_prefix = "images/" if all(prefixed) else ""
    stems = [line[len(member_prefix) :] for line in lines]
    if len(stems) != len(set(stems)):
        raise CvatMaskError("default.txt contains duplicate normalized stems")
    for stem in stems:
        if (
            Path(stem).name != stem
            or Path(stem).suffix
            or any(character in stem for character in (":", "/", "\\", "\x00"))
        ):
            raise CvatMaskError(f"default.txt stem must be flat and extension-free: {stem}")
    return stems, member_prefix


def _expected_file_members(
    stems: Sequence[str], *, member_prefix: str = ""
) -> set[str]:
    return {
        LABELMAP_PATH,
        DEFAULT_SET_PATH,
        *(f"{CLASS_PREFIX}{member_prefix}{stem}.png" for stem in stems),
        *(f"{OBJECT_PREFIX}{member_prefix}{stem}.png" for stem in stems),
    }


def _normalize_allowed_missing_stems(
    stems: Sequence[str] | None,
) -> set[str]:
    if stems is None:
        return set()
    normalized = [str(stem) for stem in stems]
    if len(normalized) != len(set(normalized)):
        raise CvatMaskError("allowed missing mask stems contain duplicates")
    for stem in normalized:
        if (
            not stem
            or stem != stem.strip()
            or Path(stem).name != stem
            or Path(stem).suffix
            or any(character in stem for character in (":", "/", "\\", "\x00"))
        ):
            raise CvatMaskError(
                f"allowed missing mask stem must be flat and extension-free: {stem!r}"
            )
    return set(normalized)


def load_allowed_missing_stems_file(path: Path) -> list[str]:
    """Load the explicit context-only stem allowlist used for CVAT exports."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Allowed-missing stems file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CvatMaskError(
            f"Allowed-missing stems file is not valid UTF-8: {path}"
        ) from exc
    lines = text.splitlines()
    allowed = _normalize_allowed_missing_stems(lines)
    if lines != sorted(lines):
        raise CvatMaskError("allowed missing mask stems must be lexicographically sorted")
    return sorted(allowed)


def validate_cvat_segmentation_mask_zip(
    input_zip: Path,
    *,
    label_name: str = "femur",
    images_dir: Path | None = None,
    reference_masks_dir: Path | None = None,
    background_value: int = DEFAULT_BACKGROUND_VALUE,
    foreground_value: int = DEFAULT_FOREGROUND_VALUE,
    allowed_missing_mask_stems: Sequence[str] | None = None,
    max_member_bytes: int = 256 * 1024 * 1024,
    max_total_bytes: int = 2 * 1024 * 1024 * 1024,
) -> ZipValidationSummary:
    input_zip = Path(input_zip)
    label_name = validate_label_name(label_name)
    if not input_zip.is_file():
        raise FileNotFoundError(f"Input ZIP not found: {input_zip}")
    if input_zip.suffix.lower() != ".zip":
        raise CvatMaskError(f"Input must have .zip extension: {input_zip}")
    if max_member_bytes <= 0 or max_total_bytes <= 0:
        raise CvatMaskError("ZIP size limits must be positive")
    allowed_missing = _normalize_allowed_missing_stems(allowed_missing_mask_stems)

    image_shapes: dict[str, tuple[int, int]] | None = None
    if images_dir is not None:
        images = scan_images(images_dir)
        image_shapes = {stem: _read_image_shape(path) for stem, path in images.items()}
    reference_masks: dict[str, np.ndarray] | None = None
    if reference_masks_dir is not None:
        mask_paths = scan_masks(reference_masks_dir)
        reference_masks = {
            stem: _read_normalized_mask(
                path,
                background_value=background_value,
                foreground_value=foreground_value,
            )
            for stem, path in mask_paths.items()
        }

    with zipfile.ZipFile(input_zip, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise CvatMaskError("ZIP contains duplicate member names")
        total_size = sum(int(info.file_size) for info in infos)
        if total_size > max_total_bytes:
            raise CvatMaskError(f"ZIP uncompressed size exceeds limit: {total_size}")
        regular: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            _safe_member_name(info.filename.rstrip("/") if info.is_dir() else info.filename)
            if _is_zip_symlink(info):
                raise CvatMaskError(f"ZIP symlink member is forbidden: {info.filename}")
            if info.is_dir():
                if info.filename not in ALLOWED_DIRECTORY_MEMBERS:
                    raise CvatMaskError(f"Unknown ZIP directory member: {info.filename}")
                continue
            regular[info.filename] = info
        for required in (LABELMAP_PATH, DEFAULT_SET_PATH):
            if required not in regular:
                raise CvatMaskError(f"ZIP missing required member: {required}")

        labelmap = _read_zip_member(
            archive, regular[LABELMAP_PATH], max_member_bytes=max_member_bytes
        )
        foreground_rgb = validate_labelmap_payload(labelmap, label_name)
        stems, member_prefix = _parse_default_stems(
            _read_zip_member(
                archive, regular[DEFAULT_SET_PATH], max_member_bytes=max_member_bytes
            )
        )
        expected_files = _expected_file_members(stems, member_prefix=member_prefix)
        actual_files = set(regular)
        extra_files = sorted(actual_files - expected_files)
        unknown_allowed = sorted(allowed_missing - set(stems))
        if unknown_allowed:
            raise CvatMaskError(
                f"Allowed-missing stems are absent from default.txt: {unknown_allowed}"
            )
        missing_pairs: set[str] = set()
        partial_pairs: list[str] = []
        required_missing: list[str] = []
        for stem in stems:
            class_name = f"{CLASS_PREFIX}{member_prefix}{stem}.png"
            object_name = f"{OBJECT_PREFIX}{member_prefix}{stem}.png"
            class_present = class_name in actual_files
            object_present = object_name in actual_files
            if class_present != object_present:
                partial_pairs.append(stem)
            elif not class_present:
                if stem in allowed_missing:
                    missing_pairs.add(stem)
                else:
                    required_missing.extend((class_name, object_name))
        if partial_pairs:
            raise CvatMaskError(
                "ZIP contains a partial class/object mask pair for stems: "
                f"{sorted(partial_pairs)}"
            )
        if required_missing or extra_files:
            raise CvatMaskError(
                f"ZIP member mismatch: missing={sorted(required_missing)}, "
                f"extra={extra_files}"
            )
        if missing_pairs and image_shapes is None:
            raise CvatMaskError(
                "images_dir is required when context-only mask pairs are absent"
            )

        if image_shapes is not None and set(image_shapes) != set(stems):
            raise CvatMaskError(
                "ZIP/image stem mismatch: "
                f"missing_images={sorted(set(stems) - set(image_shapes))}, "
                f"extra_images={sorted(set(image_shapes) - set(stems))}"
            )
        if reference_masks is not None and set(reference_masks) != set(stems):
            raise CvatMaskError(
                "ZIP/reference mask stem mismatch: "
                f"missing_reference={sorted(set(stems) - set(reference_masks))}, "
                f"extra_reference={sorted(set(reference_masks) - set(stems))}"
            )

        unique_indices: set[int] = set()
        pixel_mismatches = 0
        for stem in stems:
            class_name = f"{CLASS_PREFIX}{member_prefix}{stem}.png"
            object_name = f"{OBJECT_PREFIX}{member_prefix}{stem}.png"
            if stem in missing_pairs:
                assert image_shapes is not None
                class_mask = np.zeros(image_shapes[stem], dtype=np.uint8)
                object_mask = np.zeros(image_shapes[stem], dtype=np.uint8)
            else:
                class_mask = _decode_indexed_png(
                    _read_zip_member(
                        archive, regular[class_name], max_member_bytes=max_member_bytes
                    ),
                    class_name,
                    foreground_rgb=foreground_rgb,
                )
                object_mask = _decode_object_support_png(
                    _read_zip_member(
                        archive, regular[object_name], max_member_bytes=max_member_bytes
                    ),
                    object_name,
                )
            unique_indices.update(int(value) for value in np.unique(class_mask))
            unique_indices.update(int(value) for value in np.unique(object_mask))
            if class_mask.shape != object_mask.shape:
                raise CvatMaskError(
                    f"Class/object mask shape mismatch for {stem}: "
                    f"{class_mask.shape} vs {object_mask.shape}"
                )
            if not np.array_equal(class_mask, object_mask):
                raise CvatMaskError(
                    f"Class mask and object-mask foreground support differ for {stem}"
                )
            if image_shapes is not None and class_mask.shape != image_shapes[stem]:
                raise CvatMaskError(
                    f"ZIP mask/image size mismatch for {stem}: "
                    f"mask={class_mask.shape}, image={image_shapes[stem]}"
                )
            if reference_masks is not None:
                reference = reference_masks[stem]
                if class_mask.shape != reference.shape:
                    raise CvatMaskError(
                        f"ZIP/reference mask size mismatch for {stem}: "
                        f"{class_mask.shape} vs {reference.shape}"
                    )
                pixel_mismatches += int(np.sum(class_mask != reference))

    if reference_masks is not None and pixel_mismatches:
        raise CvatMaskError(
            f"ZIP class masks differ from normalized references: pixels={pixel_mismatches}"
        )
    return ZipValidationSummary(
        schema_version=FORMAT_SCHEMA_VERSION,
        format_name=FORMAT_NAME,
        status="ok",
        input_zip=str(input_zip.resolve()),
        zip_sha256=file_sha256(input_zip),
        zip_members=len(infos),
        images=len(stems),
        masks=len(stems) - len(missing_pairs),
        classes=2,
        label_name=label_name,
        background_index=BACKGROUND_INDEX,
        foreground_index=FOREGROUND_INDEX,
        unique_indices=sorted(unique_indices),
        stems=list(stems),
        size_mismatches=0,
        missing_masks=len(missing_pairs),
        missing_mask_stems=sorted(missing_pairs),
        extra_masks=0,
        reference_masks_compared=(len(reference_masks) if reference_masks is not None else 0),
        reference_pixel_mismatches=pixel_mismatches,
    )


def read_cvat_segmentation_class_masks(
    input_zip: Path,
    *,
    label_name: str = "femur",
    images_dir: Path | None = None,
    allowed_missing_mask_stems: Sequence[str] | None = None,
    max_member_bytes: int = 256 * 1024 * 1024,
    max_total_bytes: int = 2 * 1024 * 1024 * 1024,
) -> tuple[ZipValidationSummary, dict[str, np.ndarray]]:
    """Validate a CVAT ZIP and return its class masks without extracting files.

    The strict Phase 2 class/object equality, label map, stem, path, type, index,
    and size checks remain the single source of truth.  Reference-mask equality
    is intentionally omitted because a Phase 4 export is expected to be edited.
    """
    summary = validate_cvat_segmentation_mask_zip(
        input_zip,
        label_name=label_name,
        images_dir=images_dir,
        allowed_missing_mask_stems=allowed_missing_mask_stems,
        reference_masks_dir=None,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
    )
    masks: dict[str, np.ndarray] = {}
    allowed_missing = _normalize_allowed_missing_stems(allowed_missing_mask_stems)
    image_shapes = (
        {
            stem: _read_image_shape(path)
            for stem, path in scan_images(Path(images_dir)).items()
        }
        if images_dir is not None
        else {}
    )
    with zipfile.ZipFile(Path(input_zip), "r") as archive:
        info_by_name = {info.filename: info for info in archive.infolist()}
        foreground_rgb = validate_labelmap_payload(
            _read_zip_member(
                archive,
                info_by_name[LABELMAP_PATH],
                max_member_bytes=max_member_bytes,
            ),
            label_name,
        )
        for stem in summary.stems:
            flat_member = f"{CLASS_PREFIX}{stem}.png"
            prefixed_member = f"{CLASS_PREFIX}images/{stem}.png"
            member = (
                flat_member
                if flat_member in info_by_name
                else prefixed_member if prefixed_member in info_by_name else None
            )
            if member is None:
                if stem not in allowed_missing or stem not in image_shapes:
                    raise CvatMaskError(f"Validated class mask is unavailable: {stem}")
                masks[stem] = np.zeros(image_shapes[stem], dtype=np.uint8)
                continue
            masks[stem] = _decode_indexed_png(
                _read_zip_member(
                    archive,
                    info_by_name[member],
                    max_member_bytes=max_member_bytes,
                ),
                member,
                foreground_rgb=foreground_rgb,
            )
    return summary, masks


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_unpacked_reference(
    output_dir: Path,
    payloads: Sequence[tuple[str, bytes]],
    *,
    overwrite: bool,
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if output_dir.is_symlink():
            raise CvatMaskError(f"keep_unpacked_dir must not be a symlink: {output_dir}")
        if not output_dir.is_dir():
            raise CvatMaskError(
                f"keep_unpacked_dir exists but is not a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"keep_unpacked_dir is not empty; pass --overwrite: {output_dir}"
                )
            import shutil

            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for member, payload in payloads:
        destination = output_dir.joinpath(*PurePosixPath(member).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def convert_masks_to_cvat_zip(
    *,
    images_dir: Path,
    masks_dir: Path,
    output_zip: Path,
    label_name: str = "femur",
    background_value: int = DEFAULT_BACKGROUND_VALUE,
    foreground_value: int = DEFAULT_FOREGROUND_VALUE,
    summary_json: Path | None = None,
    keep_unpacked_dir: Path | None = None,
    overwrite: bool = False,
) -> ZipValidationSummary:
    images_dir = _ensure_plain_directory(images_dir, "images_dir")
    masks_dir = _ensure_plain_directory(masks_dir, "masks_dir")
    output_zip = Path(output_zip).expanduser().resolve(strict=False)
    label_name = validate_label_name(label_name)
    if output_zip.suffix.lower() != ".zip":
        raise CvatMaskError(f"output_zip must have .zip extension: {output_zip}")
    for source_dir in (images_dir, masks_dir):
        if output_zip == source_dir or source_dir in output_zip.parents:
            raise CvatMaskError("output_zip must be outside images_dir and masks_dir")
    if output_zip.exists() and not overwrite:
        raise FileExistsError(f"Output ZIP exists; pass --overwrite: {output_zip}")
    if output_zip.exists() and output_zip.is_symlink():
        raise CvatMaskError(f"Output ZIP must not be a symlink: {output_zip}")
    if summary_json is not None:
        summary_json = Path(summary_json).expanduser().resolve(strict=False)
        if summary_json == output_zip:
            raise CvatMaskError("summary_json and output_zip must be different paths")
        if summary_json.exists() and not overwrite:
            raise FileExistsError(
                f"summary_json exists; pass --overwrite: {summary_json}"
            )
        for source_dir in (images_dir, masks_dir):
            if summary_json == source_dir or source_dir in summary_json.parents:
                raise CvatMaskError(
                    "summary_json must be outside images_dir and masks_dir"
                )
    if keep_unpacked_dir is not None:
        keep_unpacked_dir = Path(keep_unpacked_dir).expanduser().resolve(strict=False)
        if keep_unpacked_dir == output_zip or keep_unpacked_dir in output_zip.parents:
            raise CvatMaskError(
                "output_zip must not be placed inside keep_unpacked_dir"
            )
        for source_dir in (images_dir, masks_dir):
            if keep_unpacked_dir == source_dir or source_dir in keep_unpacked_dir.parents:
                raise CvatMaskError(
                    "keep_unpacked_dir must be outside images_dir and masks_dir"
                )

    pairs = collect_input_pairs(
        images_dir=images_dir,
        masks_dir=masks_dir,
        background_value=background_value,
        foreground_value=foreground_value,
    )
    payloads = _member_payloads(pairs, label_name=label_name)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_zip.name}.", suffix=".tmp", dir=output_zip.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_deterministic_zip(temporary, payloads)
        # Validator requires a .zip suffix; the temporary name is private but valid.
        validation_temporary = temporary.with_suffix(".zip")
        os.replace(temporary, validation_temporary)
        temporary = validation_temporary
        validate_cvat_segmentation_mask_zip(
            temporary,
            label_name=label_name,
            images_dir=images_dir,
            reference_masks_dir=masks_dir,
            background_value=background_value,
            foreground_value=foreground_value,
        )
        if keep_unpacked_dir is not None:
            _write_unpacked_reference(
                keep_unpacked_dir, payloads, overwrite=overwrite
            )
        os.replace(temporary, output_zip)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    summary = validate_cvat_segmentation_mask_zip(
        output_zip,
        label_name=label_name,
        images_dir=images_dir,
        reference_masks_dir=masks_dir,
        background_value=background_value,
        foreground_value=foreground_value,
    )
    if summary_json is not None:
        _atomic_write_json(summary_json, asdict(summary))
    return summary


def _print_summary(summary: ZipValidationSummary) -> None:
    print("CVAT Segmentation Mask 1.1 validation")
    print(f"  status                    : {summary.status}")
    print(f"  input_zip                 : {summary.input_zip}")
    print(f"  images                    : {summary.images}")
    print(f"  masks                     : {summary.masks}")
    print(f"  missing_masks             : {summary.missing_masks}")
    print(f"  classes                   : {summary.classes}")
    print(f"  label_name                : {summary.label_name}")
    print(f"  unique_indices            : {summary.unique_indices}")
    print(f"  reference_masks_compared  : {summary.reference_masks_compared}")
    print(f"  reference_pixel_mismatches: {summary.reference_pixel_mismatches}")
    print(f"  zip_members               : {summary.zip_members}")
    print(f"  zip_sha256                : {summary.zip_sha256}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or validate a local CVAT Segmentation Mask 1.1 ZIP. "
            "No network or CVAT API access is performed."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser(
        "convert", help="Convert flat 0/255 binary masks to a deterministic 0/1 ZIP."
    )
    convert.add_argument("--images_dir", type=Path, required=True)
    convert.add_argument("--masks_dir", type=Path, required=True)
    convert.add_argument("--output_zip", type=Path, required=True)
    convert.add_argument("--label_name", type=str, default="femur")
    convert.add_argument("--background_value", type=int, default=0)
    convert.add_argument("--foreground_value", type=int, default=255)
    convert.add_argument("--summary_json", type=Path, default=None)
    convert.add_argument("--keep_unpacked_dir", type=Path, default=None)
    convert.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="Read-only validation of a generated or CVAT-exported ZIP."
    )
    validate.add_argument("--input_zip", type=Path, required=True)
    validate.add_argument("--images_dir", type=Path, default=None)
    validate.add_argument("--reference_masks_dir", type=Path, default=None)
    validate.add_argument("--label_name", type=str, default="femur")
    validate.add_argument("--background_value", type=int, default=0)
    validate.add_argument("--foreground_value", type=int, default=255)
    validate.add_argument(
        "--allowed_missing_stems_file",
        type=Path,
        default=None,
        help=(
            "Optional sorted context-only stem list. Both class/object PNGs may "
            "be absent only for these stems and are interpreted as empty masks."
        ),
    )
    validate.add_argument("--summary_json", type=Path, default=None)
    validate.add_argument("--max_member_bytes", type=int, default=256 * 1024 * 1024)
    validate.add_argument("--max_total_bytes", type=int, default=2 * 1024 * 1024 * 1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "convert":
            summary = convert_masks_to_cvat_zip(
                images_dir=args.images_dir,
                masks_dir=args.masks_dir,
                output_zip=args.output_zip,
                label_name=args.label_name,
                background_value=args.background_value,
                foreground_value=args.foreground_value,
                summary_json=args.summary_json,
                keep_unpacked_dir=args.keep_unpacked_dir,
                overwrite=args.overwrite,
            )
        else:
            allowed_missing = (
                load_allowed_missing_stems_file(args.allowed_missing_stems_file)
                if args.allowed_missing_stems_file is not None
                else None
            )
            summary = validate_cvat_segmentation_mask_zip(
                args.input_zip,
                label_name=args.label_name,
                images_dir=args.images_dir,
                reference_masks_dir=args.reference_masks_dir,
                background_value=args.background_value,
                foreground_value=args.foreground_value,
                allowed_missing_mask_stems=allowed_missing,
                max_member_bytes=args.max_member_bytes,
                max_total_bytes=args.max_total_bytes,
            )
            if args.summary_json is not None:
                _atomic_write_json(args.summary_json, asdict(summary))
        _print_summary(summary)
        print("CVAT Segmentation Mask 1.1 check passed.")
    except Exception as exc:
        raise SystemExit(f"CVAT mask operation failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
