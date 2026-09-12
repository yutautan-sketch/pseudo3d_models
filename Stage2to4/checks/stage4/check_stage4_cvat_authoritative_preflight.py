from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.analysis.preflight_stage4_cvat_authoritative_labels import (
    AuthoritativePreflightError,
    summarize_authoritative_frame_points,
)


def test_point_projection_metrics() -> None:
    labels = np.asarray([1, 1, 0, -1, 1, 0], dtype=np.int8)
    xy = np.asarray(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        dtype=np.float32,
    )
    mask = np.zeros((2, 6), dtype=np.uint8)
    mask[0, [0, 2, 3]] = 1
    expected = {
        "frame_points": 6,
        "source_positive_points": 3,
        "source_positive_inside_cvat_points": 1,
        "source_positive_outside_cvat_points": 2,
        "projected_positive_points": 3,
        "retained_positive_points": 1,
        "added_positive_points": 2,
        "removed_positive_points": 2,
    }
    first = summarize_authoritative_frame_points(
        point_labels=labels,
        pixel_xy=xy,
        cvat_mask=mask,
    )
    second = summarize_authoritative_frame_points(
        point_labels=labels,
        pixel_xy=xy,
        cvat_mask=mask,
    )
    assert first == expected
    assert second == expected
    print("[OK] deterministic old/retained/added/removed point partitions")


def test_empty_and_context_projection() -> None:
    labels = np.asarray([1, -1, 0], dtype=np.int8)
    xy = np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    empty = summarize_authoritative_frame_points(
        point_labels=labels,
        pixel_xy=xy,
        cvat_mask=np.zeros((1, 3), dtype=np.uint8),
    )
    assert empty["projected_positive_points"] == 0
    assert empty["removed_positive_points"] == 1

    context_mask = np.zeros((1, 3), dtype=np.uint8)
    context_mask[0, 2] = 1
    context = summarize_authoritative_frame_points(
        point_labels=labels,
        pixel_xy=xy,
        cvat_mask=context_mask,
    )
    assert context["projected_positive_points"] == 1
    assert context["added_positive_points"] == 1
    assert context["source_positive_outside_cvat_points"] == 1
    print("[OK] empty-mask removal and context-mask salvage estimates")


def test_geometry_rejection() -> None:
    try:
        summarize_authoritative_frame_points(
            point_labels=np.asarray([1], dtype=np.int8),
            pixel_xy=np.asarray([[3, 0]], dtype=np.float32),
            cvat_mask=np.zeros((1, 3), dtype=np.uint8),
        )
    except AuthoritativePreflightError as exc:
        assert "outside CVAT mask" in str(exc)
    else:
        raise AssertionError("Out-of-bounds point must be rejected")
    print("[OK] strict point/mask geometry rejection")


def test_read_only_source_contract() -> None:
    source = (
        REPO_ROOT
        / "pseudo3d/analysis/preflight_stage4_cvat_authoritative_labels.py"
    ).read_text(encoding="utf-8")
    assert 'h5py.File(source_h5, "r")' in source
    assert 'h5py.File(source_h5, "r+")' not in source
    assert "h5_files_written\": 0" in source
    print("[OK] read-only source H5 contract")


def main() -> None:
    test_point_projection_metrics()
    test_empty_and_context_projection()
    test_geometry_rejection()
    test_read_only_source_contract()
    print("Stage 4 CVAT-authoritative Step 5 synthetic checks passed.")


if __name__ == "__main__":
    main()
