from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checks.real_h5.check_stage5_xy_coordinate_provenance import (
    audit_video_provenance,
    decide_dimension_policy,
    normalize_pixel_xy_with_dimensions,
    parse_shape_string,
    read_intermediate_local_dimensions,
    resolve_intermediate_h5,
)

# ----------------------------------------------------------------------------
# S5-14 Step H2.5: synthetic pass/fail coverage for
# check_stage5_xy_coordinate_provenance.py -- shape-string parsing,
# intermediate-H5 dimension reading, source resolution (recorded attr vs.
# basename fallback vs. unresolved), the dimension-decision priority order
# (option 3/2/4), normalization fail-fast bounds checks, and a hand-built
# final+intermediate H5 pair integration test. No CUDA needed.
# ----------------------------------------------------------------------------


def expect_pass(label: str, fn) -> None:
    fn()
    print(f"ok (expected pass): {label}")


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"ok (expected fail): {label}")
        return
    raise AssertionError(f"{label}: expected AssertionError but none was raised")


def test_parse_shape_string() -> None:
    assert parse_shape_string("(120, 1, 256, 256)") == (120, 1, 256, 256)
    assert parse_shape_string("(1,)") == (1,)
    assert parse_shape_string("not a tuple") is None
    assert parse_shape_string("(1, 2, three)") is None
    assert parse_shape_string("") is None
    print("ok: parse_shape_string valid/invalid cases")


def write_intermediate_h5(path: Path, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in attrs.items():
            f.attrs[key] = value


def test_read_intermediate_local_dimensions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        ok_path = root / "ok.h5"
        write_intermediate_h5(
            ok_path,
            {
                "local_input_shape": "(120, 1, 256, 256)",
                "raw_width": 1920,
                "raw_height": 1080,
                "local_crop_top": 10,
                "local_crop_left": 20,
                "local_resize_scale": 0.5,
                "local_preprocess_effective": "resize_shorter_then_center_crop",
            },
        )
        result = read_intermediate_local_dimensions(ok_path)
        assert result["status"] == "ok", result
        assert result["width"] == 256 and result["height"] == 256, result
        print(f"ok: read_intermediate_local_dimensions well-formed case = {result}")

        missing_path = root / "missing_attr.h5"
        write_intermediate_h5(missing_path, {"raw_width": 100})
        result = read_intermediate_local_dimensions(missing_path)
        assert result["status"] == "local_input_shape_attr_missing", result
        print("ok: read_intermediate_local_dimensions missing attr detected")

        unparseable_path = root / "unparseable.h5"
        write_intermediate_h5(unparseable_path, {"local_input_shape": "garbage"})
        result = read_intermediate_local_dimensions(unparseable_path)
        assert result["status"].startswith("local_input_shape_unparseable"), result
        print("ok: read_intermediate_local_dimensions unparseable shape detected")

        nonsquare_path = root / "nonsquare.h5"
        write_intermediate_h5(nonsquare_path, {"local_input_shape": "(10, 1, 200, 300)"})
        result = read_intermediate_local_dimensions(nonsquare_path)
        assert result["status"].startswith("local_input_shape_not_square"), result
        print("ok: read_intermediate_local_dimensions non-square shape detected (unexpected per pipeline research)")

        missing_file_result = read_intermediate_local_dimensions(root / "does_not_exist.h5")
        assert missing_file_result["status"] == "intermediate_h5_not_found", missing_file_result
        print("ok: read_intermediate_local_dimensions missing file detected")


def test_resolve_intermediate_h5() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fallback_root = root / "pseudo3d_outputs"
        fallback_root.mkdir()

        recorded = root / "recorded_source.h5"
        recorded.write_bytes(b"x")
        result = resolve_intermediate_h5(
            video_name="video_a", recorded_source_path=str(recorded), fallback_root=fallback_root, fallback_suffix="_corr.h5"
        )
        assert result["method"] == "recorded_source_attr" and result["path"] == str(recorded), result
        print("ok: resolve_intermediate_h5 uses recorded source attr when it exists")

        exact = fallback_root / "video_b_corr.h5"
        exact.write_bytes(b"x")
        result = resolve_intermediate_h5(
            video_name="video_b", recorded_source_path="/does/not/exist.h5", fallback_root=fallback_root, fallback_suffix="_corr.h5"
        )
        assert result["method"] == "basename_fallback_exact" and result["path"] == str(exact), result
        print("ok: resolve_intermediate_h5 falls back to exact basename match when recorded path is stale")

        glob_only = fallback_root / "video_c_something_else.h5"
        glob_only.write_bytes(b"x")
        result = resolve_intermediate_h5(
            video_name="video_c", recorded_source_path=None, fallback_root=fallback_root, fallback_suffix="_corr.h5"
        )
        assert result["method"] == "basename_fallback_glob" and result["path"] == str(glob_only), result
        print("ok: resolve_intermediate_h5 falls back to a single glob match")

        (fallback_root / "video_d_one.h5").write_bytes(b"x")
        (fallback_root / "video_d_two.h5").write_bytes(b"x")
        result = resolve_intermediate_h5(
            video_name="video_d", recorded_source_path=None, fallback_root=fallback_root, fallback_suffix="_corr.h5"
        )
        assert result["method"] == "basename_fallback_ambiguous" and result["path"] is None, result
        print("ok: resolve_intermediate_h5 refuses to silently pick among multiple glob matches")

        result = resolve_intermediate_h5(
            video_name="video_e", recorded_source_path=None, fallback_root=fallback_root, fallback_suffix="_corr.h5"
        )
        assert result["method"] == "unresolved" and result["path"] is None, result
        print("ok: resolve_intermediate_h5 reports unresolved when nothing matches")


def test_normalize_pixel_xy_with_dimensions() -> None:
    pixel_xy = np.array([[0.0, 0.0], [255.0, 255.0], [128.0, 64.0]], dtype=np.float32)
    normalized = normalize_pixel_xy_with_dimensions(pixel_xy, width=256, height=256)
    assert np.allclose(normalized[0], [0.0, 0.0])
    assert np.allclose(normalized[1], [1.0, 1.0])
    print(f"ok: normalize_pixel_xy_with_dimensions corner cases = {normalized.tolist()}")

    expect_fail(
        "normalize_pixel_xy_with_dimensions rejects width<=1",
        lambda: normalize_pixel_xy_with_dimensions(pixel_xy, width=1, height=256),
    )
    expect_fail(
        "normalize_pixel_xy_with_dimensions rejects a NaN",
        lambda: normalize_pixel_xy_with_dimensions(
            np.array([[float("nan"), 0.0]], dtype=np.float32), width=256, height=256
        ),
    )
    expect_fail(
        "normalize_pixel_xy_with_dimensions rejects out-of-bounds x (no silent clipping)",
        lambda: normalize_pixel_xy_with_dimensions(np.array([[256.0, 0.0]], dtype=np.float32), width=256, height=256),
    )
    expect_fail(
        "normalize_pixel_xy_with_dimensions rejects negative y",
        lambda: normalize_pixel_xy_with_dimensions(np.array([[0.0, -1.0]], dtype=np.float32), width=256, height=256),
    )


def test_decide_dimension_policy() -> None:
    all_resolved = [
        {"video_name": f"v{i}", "dimension_status": "ok", "width": 256, "height": 256,
         "pixel_xy_min_x": 0.0, "pixel_xy_max_x": 255.0, "pixel_xy_min_y": 0.0, "pixel_xy_max_y": 255.0,
         "pixel_xy_bounds_ok": True}
        for i in range(3)
    ]
    policy = decide_dimension_policy(all_resolved)
    assert policy["option"] == 3 and policy["excluded_video_names"] == [], policy
    print(f"ok: decide_dimension_policy option 3 (all resolved) = {policy}")

    resolved_but_out_of_bounds = all_resolved[:-1] + [
        {"video_name": "v_bounds_violation", "dimension_status": "ok", "width": 256, "height": 256,
         "pixel_xy_min_x": 0.0, "pixel_xy_max_x": 999.0, "pixel_xy_min_y": 0.0, "pixel_xy_max_y": 255.0,
         "pixel_xy_bounds_ok": False}
    ]
    expect_fail(
        "decide_dimension_policy fail-fasts when a resolved-dimension video's own pixel_xy violates bounds "
        "(never silently included in option 3, per policy-level review)",
        lambda: decide_dimension_policy(resolved_but_out_of_bounds),
    )

    mixed_in_bounds = all_resolved + [
        {"video_name": "v_unresolved_in_bounds", "dimension_status": "intermediate_h5_not_found",
         "width": None, "height": None, "pixel_xy_min_x": 0.0, "pixel_xy_max_x": 200.0,
         "pixel_xy_min_y": 0.0, "pixel_xy_max_y": 200.0}
    ]
    policy = decide_dimension_policy(mixed_in_bounds)
    assert policy["option"] == 2, policy
    assert policy["excluded_video_names"] == [], policy
    print(f"ok: decide_dimension_policy option 2 (common dim + bounds-verified unresolved) = {policy}")

    mixed_both = all_resolved + [
        {"video_name": "v_unresolved_in_bounds_2", "dimension_status": "intermediate_h5_not_found",
         "width": None, "height": None, "pixel_xy_min_x": 0.0, "pixel_xy_max_x": 200.0,
         "pixel_xy_min_y": 0.0, "pixel_xy_max_y": 200.0},
        {"video_name": "v_unresolved_out_of_bounds", "dimension_status": "intermediate_h5_not_found",
         "width": None, "height": None, "pixel_xy_min_x": 0.0, "pixel_xy_max_x": 999.0,
         "pixel_xy_min_y": 0.0, "pixel_xy_max_y": 200.0},
    ]
    policy = decide_dimension_policy(mixed_both)
    assert policy["option"] == 2, policy
    assert policy["excluded_video_names"] == ["v_unresolved_out_of_bounds"], policy
    print(f"ok: decide_dimension_policy option 2 excludes only the out-of-bounds video, "
          f"keeps the in-bounds one = {policy}")

    single_out_of_bounds = all_resolved + [
        {"video_name": "v_unresolved_out_of_bounds", "dimension_status": "intermediate_h5_not_found",
         "width": None, "height": None, "pixel_xy_min_x": 0.0, "pixel_xy_max_x": 999.0,
         "pixel_xy_min_y": 0.0, "pixel_xy_max_y": 200.0}
    ]
    policy = decide_dimension_policy(single_out_of_bounds)
    assert policy["option"] == 4, policy
    assert policy["excluded_video_names"] == ["v_unresolved_out_of_bounds"], policy
    print(f"ok: decide_dimension_policy falls to option 4 when zero unresolved videos are bounds-verified = {policy}")

    none_resolved = [
        {"video_name": f"v{i}", "dimension_status": "intermediate_h5_not_found", "width": None, "height": None,
         "pixel_xy_min_x": 0.0, "pixel_xy_max_x": 200.0, "pixel_xy_min_y": 0.0, "pixel_xy_max_y": 200.0}
        for i in range(3)
    ]
    policy = decide_dimension_policy(none_resolved)
    assert policy["option"] == 4 and len(policy["excluded_video_names"]) == 3, policy
    print(f"ok: decide_dimension_policy option 4 (nothing resolved -> full exclusion) = {policy}")


def test_audit_video_provenance_integration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        intermediate_path = root / "intermediate.h5"
        write_intermediate_h5(intermediate_path, {"local_input_shape": "(50, 1, 256, 256)"})

        final_path = root / "final.h5"
        n = 10
        with h5py.File(final_path, "w") as f:
            pc = f.create_group("point_cloud")
            pc.create_dataset("pixel_xy", data=np.array([[i * 20.0, i * 15.0] for i in range(n)], dtype=np.float32))
            f.attrs["video_name"] = "video_x"
            f.attrs["source_pseudo3d_h5"] = str(intermediate_path)

        result = audit_video_provenance(final_path, fallback_root=root / "nonexistent_fallback", fallback_suffix="_corr.h5")
        assert result["resolution_method"] == "recorded_source_attr", result
        assert result["dimension_status"] == "ok", result
        assert result["width"] == 256 and result["height"] == 256, result
        assert result["pixel_xy_bounds_ok"] is True, result
        print(f"ok: audit_video_provenance integration (recorded source resolves) = {result}")

        final_path_stale = root / "final_stale_source.h5"
        with h5py.File(final_path_stale, "w") as f:
            pc = f.create_group("point_cloud")
            pc.create_dataset("pixel_xy", data=np.array([[10.0, 10.0]], dtype=np.float32))
            f.attrs["video_name"] = "video_y"
            f.attrs["source_pseudo3d_h5"] = "/this/path/does/not/exist.h5"

        fallback_root = root / "fallback"
        fallback_root.mkdir()
        write_intermediate_h5(fallback_root / "video_y_corr.h5", {"local_input_shape": "(10, 1, 128, 128)"})

        result = audit_video_provenance(final_path_stale, fallback_root=fallback_root, fallback_suffix="_corr.h5")
        assert result["resolution_method"] == "basename_fallback_exact", result
        assert result["width"] == 128, result
        print(f"ok: audit_video_provenance integration (stale recorded source falls back to basename) = {result}")


def main() -> None:
    test_parse_shape_string()
    test_read_intermediate_local_dimensions()
    test_resolve_intermediate_h5()
    test_normalize_pixel_xy_with_dimensions()
    test_decide_dimension_policy()
    test_audit_video_provenance_integration()
    print("check_dummy_xy_coordinate_provenance: all synthetic checks behaved as expected.")


if __name__ == "__main__":
    main()
