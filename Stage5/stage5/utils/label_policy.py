from __future__ import annotations

from typing import Any

import numpy as np

LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_POSITIVE = 1

SUPPORTED_LABEL_POLICIES: tuple[str, ...] = (
    "bbox_noncontour_ignore",
    "bbox_noncontour_background",
)
DEFAULT_LABEL_POLICY = "bbox_noncontour_ignore"


def compute_bbox_inside_mask(
    *,
    frame_order: np.ndarray,
    pixel_xy: np.ndarray,
    bbox_frame_order: np.ndarray,
    bbox_local_xyxy: np.ndarray,
) -> np.ndarray:
    """Boolean mask over points: True where the point falls inside at least one
    stored BBox for its frame.

    Mirrors the geometry in
    Stage2to4/checks/stage4/check_stage4_bbox_ranked_label_policy.py (rounded
    pixel_xy, per-frame union of BBoxes) so Stage 4 and Stage 5 agree on what
    "inside a BBox" means. Frames with no stored BBox contribute no True
    entries.
    """
    frame_order = np.asarray(frame_order)
    pixel_xy = np.asarray(pixel_xy)
    num_points = frame_order.shape[0]
    inside = np.zeros(num_points, dtype=bool)
    if bbox_frame_order is None or bbox_local_xyxy is None:
        return inside
    bbox_frame_order = np.asarray(bbox_frame_order).astype(np.int64)
    bbox_local_xyxy = np.asarray(bbox_local_xyxy).astype(np.float64)
    if bbox_frame_order.size == 0:
        return inside
    if bbox_local_xyxy.shape != (bbox_frame_order.size, 4):
        raise ValueError(
            f"bbox_local_xyxy must have shape [{bbox_frame_order.size}, 4], "
            f"got {bbox_local_xyxy.shape}"
        )

    rounded_xy = np.rint(pixel_xy).astype(np.int64)
    for order in np.unique(bbox_frame_order):
        point_idx = np.flatnonzero(frame_order == order)
        if point_idx.size == 0:
            continue
        xy = rounded_xy[point_idx]
        inside_union = np.zeros(point_idx.size, dtype=bool)
        for bbox in bbox_local_xyxy[bbox_frame_order == order]:
            if not np.isfinite(bbox).all():
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            left = int(np.floor(max(0.0, x1)))
            top = int(np.floor(max(0.0, y1)))
            right = int(np.ceil(x2))
            bottom = int(np.ceil(y2))
            if right <= left or bottom <= top:
                continue
            inside_union |= (
                (xy[:, 0] >= left)
                & (xy[:, 0] <= right)
                & (xy[:, 1] >= top)
                & (xy[:, 1] <= bottom)
            )
        inside[point_idx] = inside_union
    return inside


def audit_bbox_noncontour_targets(
    *, point_label: np.ndarray, bbox_inside_mask: np.ndarray
) -> dict[str, int]:
    """Count how source `point_label` relates to the reconstructed BBox membership.

    `stray_ignore_outside_bbox_count` is the key safety signal: under the
    teacher v6 bbox-ranked label contract, every ignore point is expected to
    lie inside some stored BBox (BBox non-contour). A nonzero count here means
    that assumption does not hold for this H5, and the background conversion
    must not proceed (see `require_no_stray_ignore_outside_bbox`).
    """
    point_label = np.asarray(point_label)
    bbox_inside_mask = np.asarray(bbox_inside_mask, dtype=bool)
    ignore_mask = point_label == LABEL_IGNORE
    target_mask = ignore_mask & bbox_inside_mask
    stray_mask = ignore_mask & ~bbox_inside_mask
    return {
        "point_count": int(point_label.size),
        "ignore_point_count": int(np.sum(ignore_mask)),
        "bbox_inside_point_count": int(np.sum(bbox_inside_mask)),
        "bbox_noncontour_target_point_count": int(np.sum(target_mask)),
        "stray_ignore_outside_bbox_count": int(np.sum(stray_mask)),
    }


def require_no_stray_ignore_outside_bbox(stats: dict[str, int], *, context: str) -> None:
    stray = stats["stray_ignore_outside_bbox_count"]
    if stray != 0:
        raise AssertionError(
            f"{context}: found {stray} ignore point(s) outside any stored BBox. "
            "The bbox_noncontour label-policy contract (every ignore point is BBox "
            "non-contour) does not hold for this H5; refusing to convert to "
            "background. Run the S5-12 Step E3 preflight to locate the affected "
            "file(s) before training with bbox_noncontour_background."
        )


def apply_bbox_noncontour_label_policy(
    policy: str,
    *,
    point_label: np.ndarray,
    valid_mask: np.ndarray,
    frame_order: np.ndarray | None = None,
    pixel_xy: np.ndarray | None = None,
    bbox_frame_order: np.ndarray | None = None,
    bbox_local_xyxy: np.ndarray | None = None,
    context: str = "apply_bbox_noncontour_label_policy",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive effective (training) label/valid_mask arrays from source teacher v6 labels.

    `policy == "bbox_noncontour_ignore"` is a strict no-op (the historical
    default) and never touches `frame_order`/`pixel_xy`/BBox arrays, so it
    works even for H5s or callers that do not provide them.

    `policy == "bbox_noncontour_background"` reconstructs which ignore points
    lie inside a stored BBox and converts only those to background
    (`point_label=0`, `valid_mask=True`); everything else -- including
    positive points and any CVAT-authoritative no-BBox label -- is left
    byte-identical to the source. Returned arrays are always copies; inputs
    are never mutated in place.
    """
    if policy not in SUPPORTED_LABEL_POLICIES:
        raise ValueError(
            f"Unsupported label_policy={policy!r}. Supported: {SUPPORTED_LABEL_POLICIES}"
        )

    point_label = np.asarray(point_label)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if policy == "bbox_noncontour_ignore":
        return (
            point_label.copy(),
            valid_mask.copy(),
            {"policy": policy, "converted_point_count": 0},
        )

    # policy == "bbox_noncontour_background"
    missing = [
        name
        for name, value in (
            ("frame_order", frame_order),
            ("pixel_xy", pixel_xy),
            ("bbox_frame_order", bbox_frame_order),
            ("bbox_local_xyxy", bbox_local_xyxy),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"{context}: policy={policy!r} requires {missing} (frame_annotation "
            "BBox data not available for this H5)"
        )

    bbox_inside_mask = compute_bbox_inside_mask(
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=bbox_frame_order,
        bbox_local_xyxy=bbox_local_xyxy,
    )
    stats = audit_bbox_noncontour_targets(point_label=point_label, bbox_inside_mask=bbox_inside_mask)
    require_no_stray_ignore_outside_bbox(stats, context=context)

    target_mask = (point_label == LABEL_IGNORE) & bbox_inside_mask
    effective_label = point_label.copy()
    effective_valid_mask = valid_mask.copy()
    effective_label[target_mask] = LABEL_BACKGROUND
    effective_valid_mask[target_mask] = True

    stats["policy"] = policy
    stats["converted_point_count"] = int(np.sum(target_mask))
    return effective_label, effective_valid_mask, stats


def summarize_label_policy_for_arrays(
    policy: str,
    *,
    source_point_label: np.ndarray,
    source_valid_mask: np.ndarray,
    frame_order: np.ndarray | None = None,
    pixel_xy: np.ndarray | None = None,
    bbox_frame_order: np.ndarray | None = None,
    bbox_local_xyxy: np.ndarray | None = None,
    context: str = "summarize_label_policy_for_arrays",
) -> dict[str, Any]:
    """One H5's worth of source/effective class counts and conversion stats."""
    effective_label, effective_valid_mask, stats = apply_bbox_noncontour_label_policy(
        policy,
        point_label=source_point_label,
        valid_mask=source_valid_mask,
        frame_order=frame_order,
        pixel_xy=pixel_xy,
        bbox_frame_order=bbox_frame_order,
        bbox_local_xyxy=bbox_local_xyxy,
        context=context,
    )
    source_valid = np.asarray(source_valid_mask, dtype=bool)
    effective_valid = effective_valid_mask
    source_label = np.asarray(source_point_label)
    return {
        **stats,
        "source_positive_count": int(np.sum(source_valid & (source_label == LABEL_POSITIVE))),
        "source_background_count": int(np.sum(source_valid & (source_label == LABEL_BACKGROUND))),
        "source_ignore_count": int(np.sum(~source_valid)),
        "effective_positive_count": int(
            np.sum(effective_valid & (effective_label == LABEL_POSITIVE))
        ),
        "effective_background_count": int(
            np.sum(effective_valid & (effective_label == LABEL_BACKGROUND))
        ),
        "effective_ignore_count": int(np.sum(~effective_valid)),
    }
