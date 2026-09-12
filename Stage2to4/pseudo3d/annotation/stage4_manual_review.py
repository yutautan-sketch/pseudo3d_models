from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from pseudo3d.annotation.annotate_pseudo3d_point_cloud import (
    LABEL_BACKGROUND,
    LABEL_FEMUR_CANDIDATE,
    LABEL_IGNORE,
    LocalBBox,
)


MANUAL_REVIEW_SCHEMA_VERSION = 1
FRAME_STEM_SEPARATOR = "__"
REVIEW_DECISIONS = {"auto_accept", "auto_refine", "manual_review"}


class ManualReviewError(ValueError):
    """Raised when a Phase 4 review package or correction violates its contract."""


@dataclass(frozen=True)
class ManualReviewConfig:
    path: Path
    raw: dict[str, Any]
    config_name: str
    fingerprint: str
    label_name: str
    require_nonempty_manual_bbox: bool
    reject_bbox_union_outside_pixels: bool
    no_bbox_label: int
    bbox_inside_non_contour_label: str
    output_suffix: str


def _canonical_fingerprint(raw: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(raw), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_manual_review_config(path: Path) -> ManualReviewConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manual-review config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ManualReviewError("Manual-review config root must be a mapping")
    raw = dict(loaded)
    if int(raw.get("schema_version", 0)) != MANUAL_REVIEW_SCHEMA_VERSION:
        raise ManualReviewError("Unsupported manual-review schema_version")
    config_name = str(raw.get("config_name", "")).strip()
    if not config_name:
        raise ManualReviewError("manual-review config_name must not be empty")
    cvat = raw.get("cvat", {})
    labels = raw.get("labels", {})
    validation = raw.get("validation", {})
    output = raw.get("output", {})
    for value, name in (
        (cvat, "cvat"),
        (labels, "labels"),
        (validation, "validation"),
        (output, "output"),
    ):
        if not isinstance(value, Mapping):
            raise ManualReviewError(f"manual-review config {name} must be a mapping")
    label_name = str(cvat.get("label_name", "")).strip()
    if str(cvat.get("format_name", "")) != "Segmentation Mask 1.1":
        raise ManualReviewError("Phase 4 fixed CVAT format must be Segmentation Mask 1.1")
    if label_name != "femur":
        raise ManualReviewError("Phase 4 fixed CVAT label_name must be 'femur'")
    if (
        int(cvat.get("background_index", -1)),
        int(cvat.get("foreground_index", -1)),
    ) != (0, 1):
        raise ManualReviewError("Phase 4 CVAT indices must be background=0/femur=1")
    convert_to_polygons = cvat.get("convert_masks_to_polygons", None)
    if convert_to_polygons is not False:
        raise ManualReviewError("Phase 4 requires Convert masks to polygons=false")
    fixed_labels = (
        int(labels.get("ignore", -999)),
        int(labels.get("background", -999)),
        int(labels.get("positive", -999)),
    )
    if fixed_labels != (-1, 0, 1):
        raise ManualReviewError("Phase 4 labels must be ignore=-1/background=0/positive=1")
    no_bbox = int(labels.get("no_bbox_label", -999))
    bbox_policy = str(labels.get("bbox_inside_non_contour_label", ""))
    if no_bbox != 0 or bbox_policy != "ignore":
        raise ManualReviewError(
            "Phase 4 requires no-BBox=background and BBox non-contour=ignore"
        )
    require_nonempty = validation.get("require_nonempty_manual_bbox", True)
    reject_outside = validation.get("reject_bbox_union_outside_pixels", True)
    if not isinstance(require_nonempty, bool) or not isinstance(reject_outside, bool):
        raise ManualReviewError("manual-review validation switches must be booleans")
    if not reject_outside:
        raise ManualReviewError("Phase 4 must reject BBox-union outside pixels")
    suffix = str(
        output.get(
            "annotated_h5_suffix",
            "_pointcloud_annotated_bboxrank_v2_phase4_manual_v1.h5",
        )
    )
    if not suffix.startswith("_") or not suffix.endswith(".h5") or "/" in suffix:
        raise ManualReviewError("output.annotated_h5_suffix must be a flat _*.h5 suffix")
    return ManualReviewConfig(
        path=path.resolve(),
        raw=raw,
        config_name=config_name,
        fingerprint=_canonical_fingerprint(raw),
        label_name=label_name,
        require_nonempty_manual_bbox=require_nonempty,
        reject_bbox_union_outside_pixels=reject_outside,
        no_bbox_label=no_bbox,
        bbox_inside_non_contour_label=bbox_policy,
        output_suffix=suffix,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_stem(video_name: str, frame_order: int, frame_index: int) -> str:
    video_name = str(video_name)
    if not video_name or any(value in video_name for value in ("/", "\\", "\0", "\n")):
        raise ManualReviewError(f"Unsafe video_name: {video_name!r}")
    if int(frame_order) < 0 or int(frame_index) < 0:
        raise ManualReviewError("frame_order and frame_index must be non-negative")
    return f"{video_name}__fo{int(frame_order):05d}__fi{int(frame_index):08d}"


def bbox_key(video_name: str, frame_order: int, frame_index: int, bbox_index: int) -> str:
    if int(bbox_index) < 0:
        raise ManualReviewError("bbox_index must be non-negative")
    return f"{frame_stem(video_name, frame_order, frame_index)}__bbox{int(bbox_index):03d}"


def bbox_bounds(
    bbox: LocalBBox | Sequence[float], shape_hw: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    if isinstance(bbox, LocalBBox):
        if not bbox.valid:
            return None
        values = bbox.local_xyxy
    else:
        values = bbox
    if len(values) != 4:
        raise ManualReviewError("BBox must contain four coordinates")
    height, width = (int(value) for value in shape_hw)
    x1, y1, x2, y2 = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    left = int(math.floor(max(0.0, x1)))
    top = int(math.floor(max(0.0, y1)))
    right = int(math.ceil(min(float(width - 1), x2)))
    bottom = int(math.ceil(min(float(height - 1), y2)))
    if right < left or bottom < top:
        return None
    return left, top, right, bottom


def bbox_union_mask(
    shape_hw: tuple[int, int], bboxes: Sequence[LocalBBox | Sequence[float]]
) -> np.ndarray:
    result = np.zeros(shape_hw, dtype=bool)
    for bbox in bboxes:
        bounds = bbox_bounds(bbox, shape_hw)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        result[top : bottom + 1, left : right + 1] = True
    return result


def clip_mask_to_bbox(
    mask: np.ndarray,
    bbox: LocalBBox | Sequence[float],
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros(mask.shape, dtype=bool)
    bounds = bbox_bounds(bbox, mask.shape)
    if bounds is None:
        return result
    left, top, right, bottom = bounds
    result[top : bottom + 1, left : right + 1] = mask[
        top : bottom + 1, left : right + 1
    ]
    return result


def compose_initial_frame_mask(
    *,
    shape_hw: tuple[int, int],
    bboxes: Sequence[LocalBBox | Sequence[float]],
    bbox_masks: Sequence[np.ndarray],
) -> np.ndarray:
    if len(bboxes) != len(bbox_masks):
        raise ManualReviewError("bboxes and bbox_masks length mismatch")
    union = np.zeros(shape_hw, dtype=bool)
    for bbox, mask in zip(bboxes, bbox_masks, strict=True):
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape != shape_hw:
            raise ManualReviewError(
                f"BBox proposal mask shape mismatch: {candidate.shape} != {shape_hw}"
            )
        union |= clip_mask_to_bbox(candidate, bbox)
    allowed = bbox_union_mask(shape_hw, bboxes)
    if np.any(union & ~allowed):
        raise AssertionError("Internal error: composed mask escaped BBox union")
    return union


def apply_corrected_frame_mask(
    *,
    point_labels: np.ndarray,
    frame_orders: np.ndarray,
    pixel_xy: np.ndarray,
    target_frame_order: int,
    corrected_mask: np.ndarray,
    bboxes: Sequence[LocalBBox | Sequence[float]],
    manual_bbox_indices: Sequence[int],
    require_nonempty_manual_bbox: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.asarray(point_labels, dtype=np.int8).copy()
    orders = np.asarray(frame_orders)
    xy_all = np.asarray(pixel_xy)
    corrected = np.asarray(corrected_mask)
    if corrected.ndim != 2 or corrected.dtype not in (np.bool_, np.uint8):
        raise ManualReviewError("Corrected mask must be 2-D bool/uint8")
    corrected = corrected.astype(bool)
    if labels.shape != orders.shape or xy_all.shape != (labels.shape[0], 2):
        raise ManualReviewError("point label/frame order/pixel_xy arrays are misaligned")
    allowed = bbox_union_mask(corrected.shape, bboxes)
    outside_mask = corrected & ~allowed
    outside = int(np.sum(outside_mask))
    if outside:
        outside_y, outside_x = np.nonzero(outside_mask)
        raise ManualReviewError(
            f"Corrected mask has {outside} pixel(s) outside strict BBox union; "
            f"x_range={int(outside_x.min())}..{int(outside_x.max())}, "
            f"y_range={int(outside_y.min())}..{int(outside_y.max())}"
        )
    for bbox_index in manual_bbox_indices:
        if bbox_index < 0 or bbox_index >= len(bboxes):
            raise ManualReviewError(f"Unknown manual-review bbox_index: {bbox_index}")
        if require_nonempty_manual_bbox:
            one_bbox = bbox_union_mask(corrected.shape, [bboxes[bbox_index]])
            if not np.any(corrected & one_bbox):
                raise ManualReviewError(
                    f"Manual-review bbox_index={bbox_index} has an empty correction"
                )
    frame_selector = orders.astype(np.int64) == int(target_frame_order)
    frame_xy = np.rint(xy_all[frame_selector]).astype(np.int64)
    if frame_xy.size:
        height, width = corrected.shape
        if np.any(frame_xy[:, 0] < 0) or np.any(frame_xy[:, 0] >= width):
            raise ManualReviewError("point_cloud/pixel_xy x coordinate is outside image")
        if np.any(frame_xy[:, 1] < 0) or np.any(frame_xy[:, 1] >= height):
            raise ManualReviewError("point_cloud/pixel_xy y coordinate is outside image")
        rebuilt = np.full(frame_xy.shape[0], LABEL_BACKGROUND, dtype=np.int8)
        inside_bbox = allowed[frame_xy[:, 1], frame_xy[:, 0]]
        inside_positive = corrected[frame_xy[:, 1], frame_xy[:, 0]]
        rebuilt[inside_bbox] = LABEL_IGNORE
        rebuilt[inside_positive] = LABEL_FEMUR_CANDIDATE
        labels[frame_selector] = rebuilt
    return labels, {
        "frame_points": int(frame_selector.sum()),
        "positive_points": int(np.sum(labels[frame_selector] == LABEL_FEMUR_CANDIDATE)),
        "ignore_points": int(np.sum(labels[frame_selector] == LABEL_IGNORE)),
        "background_points": int(np.sum(labels[frame_selector] == LABEL_BACKGROUND)),
        "corrected_positive_pixels": int(corrected.sum()),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManualReviewError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_png(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".png", np.asarray(image), [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not success:
        raise OSError(f"OpenCV failed to encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def prepare_output_root(path: Path, *, overwrite: bool) -> None:
    path = Path(path)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ManualReviewError(f"output_root must be a plain directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"output_root is not empty; pass --overwrite: {path}")
            import shutil

            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def atomic_destination(path: Path) -> tuple[int, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
