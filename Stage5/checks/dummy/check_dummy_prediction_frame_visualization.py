from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.prediction_frame_render import (
    CATEGORY_FALSE_NEGATIVE,
    CATEGORY_FALSE_POSITIVE,
    CATEGORY_IGNORE_OTHER,
    CATEGORY_IGNORE_PREDICTED_POSITIVE,
    CATEGORY_TRUE_NEGATIVE,
    CATEGORY_TRUE_POSITIVE,
    DIAGNOSTIC_CATEGORY_COLORS,
    DIAGNOSTIC_CATEGORY_NAMES,
    count_categories,
    diagnostic_segmentation_categories,
    frame_point_index_ranges,
    render_diagnostic_categories,
    validate_pixel_xy_bounds,
)

# ----------------------------------------------------------------------------
# S5-15: synthetic coverage for the prediction frame visualization
# (export_stage5_prediction_frames.py + stage5/utils/prediction_frame_render.py).
#
# Verifies that each of the six diagnostic categories lands on the expected
# pixel in the expected color, that points never leak across frames, that the
# top-FP frame selection is deterministic (including ties), that every stop
# condition actually stops, and that the evaluation directory being read is
# left byte-for-byte unchanged. No CUDA, no real data.
#
# Stage2to4 supplies `image_to_uint8_gray`. If it is not importable, a minimal
# stand-in is installed so this check still runs anywhere; the check reports
# which one it used, and the real module is always preferred.
# ----------------------------------------------------------------------------

IMAGE_SIZE = 16
NUM_FRAMES = 3
BACKGROUND_GRAY = 0


def expect_error(label: str, error_types, fn) -> None:
    try:
        fn()
    except error_types as exc:
        print(f"ok (expected failure): {label} -> {type(exc).__name__}: {exc}")
        return
    raise AssertionError(f"{label}: expected {error_types} but nothing was raised")


def install_image_to_uint8_gray(stage2to4_root: str | None) -> str:
    """Prefer Stage2to4's real converter; fall back to a documented stand-in."""
    if stage2to4_root and Path(stage2to4_root).is_dir():
        if stage2to4_root not in sys.path:
            sys.path.insert(0, stage2to4_root)
    try:
        from src.utils.alpha_texture_processing import image_to_uint8_gray  # noqa: F401

        return "stage2to4"
    except ImportError:
        pass

    def image_to_uint8_gray(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.ndim == 3:
            img = img[0] if img.shape[0] == 1 else img[..., 0] if img.shape[-1] == 1 else img[0]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    module = types.ModuleType("src.utils.alpha_texture_processing")
    module.image_to_uint8_gray = image_to_uint8_gray
    src = sys.modules.setdefault("src", types.ModuleType("src"))
    utils = sys.modules.setdefault("src.utils", types.ModuleType("src.utils"))
    src.utils = utils
    utils.alpha_texture_processing = module
    sys.modules["src.utils.alpha_texture_processing"] = module
    return "stand-in"


# ----------------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------------


def build_points(entries: list[tuple[int, int, int, int, bool, int]]) -> dict[str, np.ndarray]:
    """entries: (frame_order, x, y, point_label, valid, pred_label)."""
    frame_order = np.asarray([item[0] for item in entries], dtype=np.int64)
    pixel_xy = np.asarray([[item[1], item[2]] for item in entries], dtype=np.float32)
    point_label = np.asarray([item[3] for item in entries], dtype=np.int64)
    valid_mask = np.asarray([item[4] for item in entries], dtype=bool)
    pred_label = np.asarray([item[5] for item in entries], dtype=np.uint8)
    num = len(entries)
    return {
        "frame_order": frame_order,
        "pixel_xy": pixel_xy,
        "point_label": point_label,
        "valid_mask": valid_mask,
        "pred_label": pred_label,
        "points": np.zeros((num, 3), dtype=np.float32),
        "intensity": np.zeros(num, dtype=np.float32),
        "alpha": np.zeros(num, dtype=np.float32),
        "confidence": np.zeros(num, dtype=np.float32),
    }


def write_final_h5(path: Path, arrays: dict[str, np.ndarray], *, video_name: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        group = f.create_group("point_cloud")
        for key in ("points", "intensity", "alpha", "confidence", "frame_order", "pixel_xy"):
            group.create_dataset(key, data=arrays[key])
        annotation = f.create_group("annotation")
        annotation.create_dataset("point_label", data=arrays["point_label"])
        annotation.create_dataset("valid_mask", data=arrays["valid_mask"])
        f.attrs["video_name"] = video_name
        if source:
            f.attrs["source_pseudo3d_h5"] = source


def write_intermediate_h5(path: Path, *, num_frames: int = NUM_FRAMES, size: int = IMAGE_SIZE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.full((num_frames, 1, size, size), BACKGROUND_GRAY, dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.create_dataset("local_encoder_images", data=images)
        f.attrs["local_input_shape"] = f"({num_frames}, 1, {size}, {size})"
        f.attrs["raw_width"] = 1920
        f.attrs["raw_height"] = 1080
        f.attrs["local_crop_top"] = 10
        f.attrs["local_crop_left"] = 20
        f.attrs["local_resize_scale"] = 0.5


def write_prediction_npz(path: Path, pred_label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        point_indices=np.arange(pred_label.size, dtype=np.int64),
        prob_femur=pred_label.astype(np.float32),
        pred_label=pred_label.astype(np.uint8),
        vote_count=np.ones(pred_label.size, dtype=np.int32),
    )


def write_h5_metrics_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["split,video_name,h5_path"]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_options(**overrides) -> argparse.Namespace:
    values = {
        "top_frames_per_video": 10,
        "all_frames": False,
        "zero_fp_samples": 0,
        "zero_tp_samples": 0,
        "seed": 0,
        "radius": 0,
        "alpha": 1.0,
        "draw_true_negative": False,
        "draw_ignore_other": False,
        "draw_bbox": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def snapshot_tree(root: Path) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns, digest)
    return snapshot


# ----------------------------------------------------------------------------
# Unit-level checks
# ----------------------------------------------------------------------------


def test_category_colors_match_ply() -> None:
    expected = {
        "true_positive": (30, 180, 70),
        "false_positive": (230, 45, 45),
        "false_negative": (245, 190, 35),
        "true_negative": (65, 105, 180),
        "ignore_predicted_positive": (200, 65, 180),
        "ignore_other": (145, 145, 145),
    }
    assert tuple(DIAGNOSTIC_CATEGORY_NAMES) == tuple(expected), DIAGNOSTIC_CATEGORY_NAMES
    for index, name in enumerate(DIAGNOSTIC_CATEGORY_NAMES):
        assert tuple(int(v) for v in DIAGNOSTIC_CATEGORY_COLORS[index]) == expected[name], name
    print("ok: diagnostic colors match the handoff table and the diagnostic PLY")


def test_safe_name_matches_evaluate_stage5() -> None:
    """The npz file names come from evaluate_stage5.safe_name; guard the copy."""
    from export_stage5_prediction_frames import safe_name

    source = (REPO_ROOT / "evaluate_stage5.py").read_text(encoding="utf-8")
    expected_body = 'return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")'
    assert expected_body in source, (
        "evaluate_stage5.safe_name changed; update the copy in "
        "export_stage5_prediction_frames.py"
    )
    for value, expected in (
        ("20240101_120000_1", "20240101_120000_1"),
        ("video name/with:chars", "video_name_with_chars"),
        ("__leading_trailing__", "leading_trailing"),
    ):
        assert safe_name(value) == expected, (value, safe_name(value))
    print("ok: safe_name matches evaluate_stage5.py and strips as expected")


def test_render_each_category_pixel() -> None:
    """Every category must land on its own pixel in its own color."""
    placements = [
        (2, 2, CATEGORY_TRUE_POSITIVE, 1, True, 1),
        (4, 2, CATEGORY_FALSE_POSITIVE, 0, True, 1),
        (6, 2, CATEGORY_FALSE_NEGATIVE, 1, True, 0),
        (8, 2, CATEGORY_TRUE_NEGATIVE, 0, True, 0),
        (10, 2, CATEGORY_IGNORE_PREDICTED_POSITIVE, 0, False, 1),
        (12, 2, CATEGORY_IGNORE_OTHER, 0, False, 0),
    ]
    entries = [(0, x, y, label, valid, pred) for x, y, _, label, valid, pred in placements]
    arrays = build_points(entries)
    categories = diagnostic_segmentation_categories(
        arrays["point_label"], arrays["valid_mask"], arrays["pred_label"], ignore_index=-1
    )
    expected_categories = [item[2] for item in placements]
    assert list(categories) == expected_categories, (list(categories), expected_categories)

    gray = np.full((IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_GRAY, dtype=np.uint8)

    rgb = render_diagnostic_categories(
        gray, pixel_xy=arrays["pixel_xy"], categories=categories, radius=0, alpha=1.0
    )
    for x, y, category, *_ in placements:
        actual = tuple(int(v) for v in rgb[y, x])
        if category in (CATEGORY_TRUE_NEGATIVE, CATEGORY_IGNORE_OTHER):
            assert actual == (BACKGROUND_GRAY,) * 3, (
                f"{DIAGNOSTIC_CATEGORY_NAMES[category]} must not be drawn by default, "
                f"got {actual}"
            )
        else:
            expected = tuple(int(v) for v in DIAGNOSTIC_CATEGORY_COLORS[category])
            assert actual == expected, (DIAGNOSTIC_CATEGORY_NAMES[category], actual, expected)
    print("ok: TP/FP/FN/ignore_predicted_positive drawn; TN/ignore_other off by default")

    rgb_all = render_diagnostic_categories(
        gray,
        pixel_xy=arrays["pixel_xy"],
        categories=categories,
        radius=0,
        alpha=1.0,
        draw_categories=range(len(DIAGNOSTIC_CATEGORY_NAMES)),
    )
    for x, y, category, *_ in placements:
        expected = tuple(int(v) for v in DIAGNOSTIC_CATEGORY_COLORS[category])
        actual = tuple(int(v) for v in rgb_all[y, x])
        assert actual == expected, (DIAGNOSTIC_CATEGORY_NAMES[category], actual, expected)
    print("ok: all six categories drawn when explicitly requested")

    assert gray.max() == BACKGROUND_GRAY, "render must not modify the input image"
    print("ok: the source grayscale image is left unmodified")


def test_false_positive_drawn_on_top() -> None:
    """FP is the subject of this export; it must never be hidden."""
    entries = [
        (0, 5, 5, 1, True, 1),   # true_positive
        (0, 5, 5, 0, True, 1),   # false_positive on the same pixel
    ]
    arrays = build_points(entries)
    categories = diagnostic_segmentation_categories(
        arrays["point_label"], arrays["valid_mask"], arrays["pred_label"], ignore_index=-1
    )
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_GRAY, dtype=np.uint8)
    rgb = render_diagnostic_categories(
        gray, pixel_xy=arrays["pixel_xy"], categories=categories, radius=0, alpha=1.0
    )
    expected = tuple(int(v) for v in DIAGNOSTIC_CATEGORY_COLORS[CATEGORY_FALSE_POSITIVE])
    actual = tuple(int(v) for v in rgb[5, 5])
    assert actual == expected, (actual, expected)
    print("ok: false_positive paints over true_positive on a shared pixel")


def test_radius_and_alpha() -> None:
    entries = [(0, 8, 8, 0, True, 1)]
    arrays = build_points(entries)
    categories = diagnostic_segmentation_categories(
        arrays["point_label"], arrays["valid_mask"], arrays["pred_label"], ignore_index=-1
    )
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE), 100, dtype=np.uint8)
    rgb = render_diagnostic_categories(
        gray, pixel_xy=arrays["pixel_xy"], categories=categories, radius=1, alpha=1.0
    )
    color = tuple(int(v) for v in DIAGNOSTIC_CATEGORY_COLORS[CATEGORY_FALSE_POSITIVE])
    assert tuple(int(v) for v in rgb[8, 8]) == color
    assert tuple(int(v) for v in rgb[7, 8]) == color, "radius=1 must cover the 4-neighbourhood"
    assert tuple(int(v) for v in rgb[5, 5]) == (100, 100, 100), "radius=1 must stay local"

    half = render_diagnostic_categories(
        gray, pixel_xy=arrays["pixel_xy"], categories=categories, radius=0, alpha=0.5
    )
    # `blend_mask` clips then casts to uint8 (truncation), exactly as the
    # Stage 4 exporter does; the expectation must truncate too, not round.
    blended = tuple(int(0.5 * 100 + 0.5 * c) for c in color)
    assert tuple(int(v) for v in half[8, 8]) == blended, (half[8, 8], blended)
    print("ok: radius dilates locally and alpha blends against the frame")


def test_frame_dispatch() -> None:
    """A point must be drawn on its own frame and on no other."""
    entries = [
        (0, 2, 2, 0, True, 1),
        (1, 3, 3, 0, True, 1),
        (2, 4, 4, 0, True, 1),
        (1, 5, 5, 1, True, 1),
    ]
    arrays = build_points(entries)
    categories = diagnostic_segmentation_categories(
        arrays["point_label"], arrays["valid_mask"], arrays["pred_label"], ignore_index=-1
    )
    frame_index = frame_point_index_ranges(arrays["frame_order"], NUM_FRAMES)
    assert list(frame_index.counts) == [1, 2, 1], list(frame_index.counts)

    gray = np.full((IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_GRAY, dtype=np.uint8)
    drawn: dict[int, set[tuple[int, int]]] = {}
    for frame_order in range(NUM_FRAMES):
        point_indices = frame_index.indices(frame_order)
        rgb = render_diagnostic_categories(
            gray,
            pixel_xy=arrays["pixel_xy"][point_indices],
            categories=categories[point_indices],
            radius=0,
            alpha=1.0,
        )
        ys, xs = np.nonzero(np.any(rgb != BACKGROUND_GRAY, axis=-1))
        drawn[frame_order] = {(int(x), int(y)) for x, y in zip(xs, ys)}

    assert drawn[0] == {(2, 2)}, drawn[0]
    assert drawn[1] == {(3, 3), (5, 5)}, drawn[1]
    assert drawn[2] == {(4, 4)}, drawn[2]
    print("ok: frame_order dispatch keeps every point on its own frame")


def test_frame_index_stop_conditions() -> None:
    expect_error(
        "frame_order beyond the available images",
        ValueError,
        lambda: frame_point_index_ranges(np.asarray([0, 1, 9], dtype=np.int64), NUM_FRAMES),
    )
    expect_error(
        "negative frame_order",
        ValueError,
        lambda: frame_point_index_ranges(np.asarray([0, -1], dtype=np.int64), NUM_FRAMES),
    )
    expect_error(
        "pixel_xy outside the local-crop image",
        ValueError,
        lambda: validate_pixel_xy_bounds(
            np.asarray([[1.0, 1.0], [IMAGE_SIZE + 3.0, 1.0]], dtype=np.float32),
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            context="synthetic",
        ),
    )
    expect_error(
        "non-finite pixel_xy",
        ValueError,
        lambda: validate_pixel_xy_bounds(
            np.asarray([[np.nan, 1.0]], dtype=np.float32),
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        ),
    )
    validate_pixel_xy_bounds(
        np.asarray([[IMAGE_SIZE - 0.4, 0.0]], dtype=np.float32),
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
    )
    print("ok: coordinate and frame stop conditions raise instead of skipping")


def test_frame_selection_order() -> None:
    from export_stage5_prediction_frames import select_frames

    frame_rows = [
        {"frame_order": 0, "num_points": 10, "false_positive": 5, "true_positive": 1},
        {"frame_order": 1, "num_points": 10, "false_positive": 9, "true_positive": 0},
        {"frame_order": 2, "num_points": 10, "false_positive": 5, "true_positive": 2},
        {"frame_order": 3, "num_points": 10, "false_positive": 0, "true_positive": 3},
        {"frame_order": 4, "num_points": 10, "false_positive": 7, "true_positive": 0},
    ]
    chosen = select_frames(
        frame_rows,
        top_frames_per_video=3,
        all_frames=False,
        zero_fp_samples=0,
        zero_tp_samples=0,
        seed=0,
    )
    assert [row["frame_order"] for row in chosen] == [0, 1, 4], chosen
    reasons = {row["frame_order"]: row["selection_reason"] for row in chosen}
    assert reasons[1] == "top_false_positive_rank_001", reasons
    assert reasons[4] == "top_false_positive_rank_002", reasons
    # frames 0 and 2 tie at FP=5; the lower frame_order must win, every time.
    assert reasons[0] == "top_false_positive_rank_003", reasons
    for _ in range(5):
        repeat = select_frames(
            frame_rows,
            top_frames_per_video=3,
            all_frames=False,
            zero_fp_samples=0,
            zero_tp_samples=0,
            seed=0,
        )
        assert repeat == chosen, "selection must be deterministic across runs"
    print("ok: top-FP selection ranks by FP desc, breaks ties by frame_order asc")

    sampled = select_frames(
        frame_rows,
        top_frames_per_video=1,
        all_frames=False,
        zero_fp_samples=1,
        zero_tp_samples=1,
        seed=0,
    )
    orders = [row["frame_order"] for row in sampled]
    assert 1 in orders, orders
    assert 3 in orders, "the only zero-FP frame must be the zero-FP sample"
    assert any(int(frame_rows[o]["true_positive"]) == 0 for o in orders if o != 1), orders
    print("ok: zero-FP / zero-TP samples are added deterministically")

    every = select_frames(
        frame_rows,
        top_frames_per_video=1,
        all_frames=True,
        zero_fp_samples=0,
        zero_tp_samples=0,
        seed=0,
    )
    assert [row["frame_order"] for row in every] == [0, 1, 2, 3, 4], every
    print("ok: --all_frames selects every frame in frame_order")


# ----------------------------------------------------------------------------
# End-to-end export
# ----------------------------------------------------------------------------


def build_export_fixture(root: Path) -> dict[str, object]:
    """Build one evaluation directory with one video, as evaluate_stage5 leaves it."""
    entries = [
        (0, 2, 2, 1, True, 1),    # TP
        (0, 4, 2, 0, True, 1),    # FP
        (0, 6, 2, 0, True, 1),    # FP
        (0, 8, 2, 1, True, 0),    # FN
        (0, 10, 2, 0, True, 0),   # TN
        (0, 12, 2, 0, False, 1),  # ignore_predicted_positive
        (1, 3, 3, 0, True, 1),    # FP
        (2, 5, 5, 1, True, 1),    # TP only; zero FP
    ]
    arrays = build_points(entries)
    video_name = "20240101_120000_1"
    data_dir = root / "data"
    intermediate = data_dir / f"{video_name}_ts448_oym96_corr.h5"
    write_intermediate_h5(intermediate)
    final_h5 = data_dir / f"{video_name}_final.h5"
    write_final_h5(final_h5, arrays, video_name=video_name, source=str(intermediate))

    evaluation_dir = root / "evaluation" / "best"
    write_prediction_npz(
        evaluation_dir / "predictions" / "validation" / f"{video_name}.npz",
        arrays["pred_label"],
    )
    write_h5_metrics_csv(
        evaluation_dir / "h5_metrics.csv",
        [("validation", video_name, str(final_h5))],
    )
    return {
        "arrays": arrays,
        "video_name": video_name,
        "final_h5": final_h5,
        "intermediate": intermediate,
        "evaluation_dir": evaluation_dir,
        "data_dir": data_dir,
        "prediction": evaluation_dir / "predictions" / "validation" / f"{video_name}.npz",
    }


def test_export_end_to_end(root: Path) -> None:
    import imageio.v2 as imageio

    from export_stage5_prediction_frames import export_video_frames

    fixture = build_export_fixture(root)
    evaluation_dir = fixture["evaluation_dir"]
    output_root = evaluation_dir / "prediction_frames"
    before = snapshot_tree(evaluation_dir)

    record = export_video_frames(
        split="validation",
        video_name=fixture["video_name"],
        h5_path=fixture["final_h5"],
        prediction_path=fixture["prediction"],
        output_root=output_root,
        pseudo3d_outputs_root=fixture["data_dir"],
        fallback_suffix="_ts448_oym96_corr.h5",
        options=default_options(top_frames_per_video=2),
        hash_h5=False,
    )

    assert record["status"] == "ok", record
    assert record["resolution_method"] == "recorded_source_attr", record
    assert record["num_frames"] == NUM_FRAMES, record
    assert record["total_false_positive"] == 3, record
    assert record["total_true_positive"] == 2, record
    assert record["num_frames_rendered"] == 2, record

    video_dir = output_root / "validation" / fixture["video_name"]
    pngs = sorted(p.name for p in video_dir.glob("*.png"))
    assert pngs == ["frame_00000_fp0002.png", "frame_00001_fp0001.png"], pngs
    print(f"ok: exported {pngs} (top-2 by FP, frame number and FP count in the name)")

    frame0 = np.asarray(imageio.imread(video_dir / "frame_00000_fp0002.png"))
    fp_color = tuple(int(v) for v in DIAGNOSTIC_CATEGORY_COLORS[CATEGORY_FALSE_POSITIVE])
    tp_color = tuple(int(v) for v in DIAGNOSTIC_CATEGORY_COLORS[CATEGORY_TRUE_POSITIVE])
    assert tuple(int(v) for v in frame0[2, 4]) == fp_color, frame0[2, 4]
    assert tuple(int(v) for v in frame0[2, 2]) == tp_color, frame0[2, 2]
    assert tuple(int(v) for v in frame0[2, 10]) == (BACKGROUND_GRAY,) * 3, "TN must stay off"
    assert tuple(int(v) for v in frame0[3, 3]) == (BACKGROUND_GRAY,) * 3, (
        "frame 1's point must not appear on frame 0"
    )
    print("ok: written PNG carries the expected colors at the expected pixels")

    summary_csv = output_root / "validation" / f"{fixture['video_name']}_frame_summary.csv"
    text = summary_csv.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert len(lines) == NUM_FRAMES + 1, lines
    header = lines[0].split(",")
    for column in ("frame_order", "false_positive", "true_positive", "selected", "png"):
        assert column in header, (column, header)
    print("ok: per-frame summary CSV has one row per frame with the category counts")

    after = snapshot_tree(evaluation_dir)
    unchanged = {k: v for k, v in after.items() if not k.startswith("prediction_frames")}
    assert unchanged == before, (
        "the existing evaluation output must be unchanged; differences: "
        f"{set(unchanged.items()) ^ set(before.items())}"
    )
    print("ok: existing evaluation files unchanged (size, mtime and sha256)")


def test_export_stop_conditions(root: Path) -> None:
    from export_stage5_prediction_frames import export_video_frames

    fixture = build_export_fixture(root)
    evaluation_dir = fixture["evaluation_dir"]
    output_root = evaluation_dir / "prediction_frames"

    def run(**overrides):
        kwargs = {
            "split": "validation",
            "video_name": fixture["video_name"],
            "h5_path": fixture["final_h5"],
            "prediction_path": fixture["prediction"],
            "output_root": output_root,
            "pseudo3d_outputs_root": fixture["data_dir"],
            "fallback_suffix": "_ts448_oym96_corr.h5",
            "options": default_options(),
            "hash_h5": False,
        }
        kwargs.update(overrides)
        return export_video_frames(**kwargs)

    truncated = root / "truncated.npz"
    write_prediction_npz(truncated, fixture["arrays"]["pred_label"][:-1])
    expect_error(
        "prediction npz point count != H5 point count",
        ValueError,
        lambda: run(prediction_path=truncated),
    )

    expect_error(
        "prediction npz missing",
        FileNotFoundError,
        lambda: run(prediction_path=root / "absent.npz"),
    )

    out_of_bounds = dict(fixture["arrays"])
    pixel_xy = out_of_bounds["pixel_xy"].copy()
    pixel_xy[0, 0] = IMAGE_SIZE + 5.0
    out_of_bounds["pixel_xy"] = pixel_xy
    oob_h5 = root / "out_of_bounds.h5"
    write_final_h5(
        oob_h5,
        out_of_bounds,
        video_name=fixture["video_name"],
        source=str(fixture["intermediate"]),
    )
    expect_error(
        "pixel_xy outside the local-crop image",
        ValueError,
        lambda: run(h5_path=oob_h5),
    )

    bad_frames = dict(fixture["arrays"])
    frame_order = bad_frames["frame_order"].copy()
    frame_order[0] = NUM_FRAMES + 2
    bad_frames["frame_order"] = frame_order
    frame_h5 = root / "bad_frame_order.h5"
    write_final_h5(
        frame_h5,
        bad_frames,
        video_name=fixture["video_name"],
        source=str(fixture["intermediate"]),
    )
    expect_error(
        "frame_order beyond local_encoder_images",
        ValueError,
        lambda: run(h5_path=frame_h5),
    )

    # An unresolvable intermediate H5 is recorded, not guessed around and not
    # silently skipped: no other video's dimensions are borrowed.
    orphan_h5 = root / "orphan.h5"
    write_final_h5(orphan_h5, fixture["arrays"], video_name="20240101_120000_9", source="")
    record = run(h5_path=orphan_h5, video_name="20240101_120000_9")
    assert record["status"] == "unvisualized", record
    assert record["resolution_method"] == "unresolved", record
    assert record["width"] is None and record["height"] is None, record
    print(f"ok (expected failure): unresolvable intermediate H5 -> {record['reason']}")

    print("ok: every stop condition stops and reports")


def test_dry_run_writes_no_png(root: Path) -> None:
    from export_stage5_prediction_frames import export_video_frames

    fixture = build_export_fixture(root)
    output_root = fixture["evaluation_dir"] / "prediction_frames"
    record = export_video_frames(
        split="validation",
        video_name=fixture["video_name"],
        h5_path=fixture["final_h5"],
        prediction_path=fixture["prediction"],
        output_root=output_root,
        pseudo3d_outputs_root=fixture["data_dir"],
        fallback_suffix="_ts448_oym96_corr.h5",
        options=default_options(dry_run=True),
        hash_h5=False,
    )
    assert record["status"] == "ok", record
    assert record["num_frames_rendered"] == 0, record
    assert not list(output_root.rglob("*.png")), list(output_root.rglob("*.png"))
    assert (output_root / "validation" / f"{fixture['video_name']}_frame_summary.csv").is_file()
    print("ok: --dry_run resolves inputs and writes the CSV but renders nothing")


def test_count_categories() -> None:
    categories = np.asarray([0, 0, 1, 2, 5], dtype=np.uint8)
    counts = count_categories(categories)
    assert counts["true_positive"] == 2, counts
    assert counts["false_positive"] == 1, counts
    assert counts["true_negative"] == 0, counts
    assert sum(counts.values()) == categories.size, counts
    print("ok: count_categories totals every category")


def main() -> None:
    source = install_image_to_uint8_gray(os.environ.get("STAGE2TO4_ROOT"))
    print(f"image_to_uint8_gray source: {source}")

    test_category_colors_match_ply()
    test_safe_name_matches_evaluate_stage5()
    test_render_each_category_pixel()
    test_false_positive_drawn_on_top()
    test_radius_and_alpha()
    test_frame_dispatch()
    test_frame_index_stop_conditions()
    test_frame_selection_order()
    test_count_categories()

    with tempfile.TemporaryDirectory() as tmp:
        test_export_end_to_end(Path(tmp) / "e2e")
    with tempfile.TemporaryDirectory() as tmp:
        test_export_stop_conditions(Path(tmp) / "stop")
    with tempfile.TemporaryDirectory() as tmp:
        test_dry_run_writes_no_png(Path(tmp) / "dry")

    print("check_dummy_prediction_frame_visualization: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
