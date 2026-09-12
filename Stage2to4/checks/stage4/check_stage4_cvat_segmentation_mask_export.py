from __future__ import annotations

import json
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from pseudo3d.export.convert_masks_to_cvat_segmentation_mask_1_1 import (
    CLASS_PREFIX,
    CVAT_LABELMAP_HEADER,
    DEFAULT_SET_PATH,
    LABELMAP_PATH,
    OBJECT_PREFIX,
    CvatMaskError,
    collect_input_pairs,
    convert_masks_to_cvat_zip,
    file_sha256,
    load_allowed_missing_stems_file,
    normalize_binary_mask,
    read_cvat_segmentation_class_masks,
    validate_cvat_segmentation_mask_zip,
    validate_label_name,
)


def _require_raises(
    exception_type: type[BaseException],
    callback: Callable[[], object],
) -> BaseException:
    try:
        callback()
    except exception_type as exc:
        return exc
    raise AssertionError(f"Expected {exception_type.__name__} was not raised")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write test image: {path}")


def _make_fixture(root: Path, *, creation_order: tuple[str, ...] = ("frame_b", "frame_a", "frame_c")) -> tuple[Path, Path, dict[str, np.ndarray]]:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    shapes = {
        "frame_a": (17, 23),
        "frame_b": (19, 21),
        "frame_c": (16, 20),
    }
    binary_masks: dict[str, np.ndarray] = {}
    for index, stem in enumerate(creation_order):
        height, width = shapes[stem]
        image = np.full((height, width), 30 + index * 20, dtype=np.uint8)
        image[:, width // 2 :] += 40
        image_suffix = ".jpg" if stem == "frame_b" else ".png"
        _write_image(images / f"{stem}{image_suffix}", image)

        mask = np.zeros((height, width), dtype=np.uint8)
        if stem == "frame_b":
            mask[3:-3, 4:-4] = 255
        elif stem == "frame_c":
            diagonal = np.arange(min(height, width))
            mask[diagonal, diagonal] = 255
        binary_masks[stem] = mask
        _write_image(masks / f"{stem}.png", mask)
    return images, masks, binary_masks


def _read_zip_mask(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    encoded = np.frombuffer(archive.read(member), dtype=np.uint8)
    mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    assert mask is not None
    return mask


def test_normal_conversion_and_determinism(root: Path) -> None:
    fixture_1 = root / "fixture_1"
    fixture_2 = root / "fixture_2"
    images_1, masks_1, source_masks_1 = _make_fixture(fixture_1)
    images_2, masks_2, _ = _make_fixture(
        fixture_2, creation_order=("frame_c", "frame_b", "frame_a")
    )
    output_1 = root / "first.zip"
    output_2 = root / "second.zip"
    summary_json = root / "first.summary.json"
    unpacked = root / "unpacked"
    input_hashes = {
        path: file_sha256(path)
        for path in sorted([*images_1.iterdir(), *masks_1.iterdir()])
    }

    summary_1 = convert_masks_to_cvat_zip(
        images_dir=images_1,
        masks_dir=masks_1,
        output_zip=output_1,
        summary_json=summary_json,
        keep_unpacked_dir=unpacked,
    )
    summary_2 = convert_masks_to_cvat_zip(
        images_dir=images_2,
        masks_dir=masks_2,
        output_zip=output_2,
    )
    assert output_1.read_bytes() == output_2.read_bytes()
    assert summary_1.zip_sha256 == summary_2.zip_sha256
    assert summary_1.stems == ["frame_a", "frame_b", "frame_c"]
    assert summary_1.unique_indices == [0, 1]
    assert summary_1.reference_pixel_mismatches == 0
    assert summary_1.reference_masks_compared == 3
    assert json.loads(summary_json.read_text(encoding="utf-8"))["status"] == "ok"
    for path, expected_hash in input_hashes.items():
        assert file_sha256(path) == expected_hash

    expected_members = [
        LABELMAP_PATH,
        DEFAULT_SET_PATH,
        *(f"{CLASS_PREFIX}{stem}.png" for stem in summary_1.stems),
        *(f"{OBJECT_PREFIX}{stem}.png" for stem in summary_1.stems),
    ]
    unpacked_payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(output_1, "r") as archive:
        assert archive.namelist() == expected_members
        assert archive.read(LABELMAP_PATH) == (
            b"background:0,0,0::\nfemur:255,0,0::\n"
        )
        assert archive.read(DEFAULT_SET_PATH) == b"frame_a\nframe_b\nframe_c\n"
        assert not any(name.startswith("images/") for name in archive.namelist())
        unpacked_payloads = {name: archive.read(name) for name in expected_members}
        for stem in summary_1.stems:
            class_mask = _read_zip_mask(archive, f"{CLASS_PREFIX}{stem}.png")
            object_mask = _read_zip_mask(archive, f"{OBJECT_PREFIX}{stem}.png")
            expected = normalize_binary_mask(source_masks_1[stem])
            assert class_mask.dtype == np.uint8 and class_mask.ndim == 2
            np.testing.assert_array_equal(class_mask, expected)
            np.testing.assert_array_equal(object_mask, expected)

    for member in expected_members:
        assert (unpacked / member).read_bytes() == unpacked_payloads[member]
    validation = validate_cvat_segmentation_mask_zip(
        output_1,
        images_dir=images_1,
        reference_masks_dir=masks_1,
    )
    assert validation.status == "ok"
    print("[OK] deterministic 0/255 -> 0/1 conversion and exact ZIP contract")


def test_pairing_and_filesystem_validation(root: Path) -> None:
    images, masks, _ = _make_fixture(root / "pairing")
    (masks / "frame_a.png").unlink()
    _require_raises(
        CvatMaskError,
        lambda: collect_input_pairs(images_dir=images, masks_dir=masks),
    )
    _write_image(masks / "frame_a.png", np.zeros((17, 23), dtype=np.uint8))
    _write_image(masks / "extra.png", np.zeros((5, 5), dtype=np.uint8))
    _require_raises(
        CvatMaskError,
        lambda: collect_input_pairs(images_dir=images, masks_dir=masks),
    )
    (masks / "extra.png").unlink()

    _write_image(images / "frame_a.jpg", np.zeros((17, 23), dtype=np.uint8))
    _require_raises(
        CvatMaskError,
        lambda: collect_input_pairs(images_dir=images, masks_dir=masks),
    )
    (images / "frame_a.jpg").unlink()
    (images / "unsupported.txt").write_text("x", encoding="utf-8")
    _require_raises(
        CvatMaskError,
        lambda: collect_input_pairs(images_dir=images, masks_dir=masks),
    )
    (images / "unsupported.txt").unlink()
    (masks / "nested").mkdir()
    _require_raises(
        CvatMaskError,
        lambda: collect_input_pairs(images_dir=images, masks_dir=masks),
    )
    (masks / "nested").rmdir()

    symlink = masks / "linked.png"
    try:
        symlink.symlink_to(masks / "frame_a.png")
    except OSError:
        pass
    else:
        _require_raises(
            CvatMaskError,
            lambda: collect_input_pairs(images_dir=images, masks_dir=masks),
        )
        symlink.unlink()
    print("[OK] missing/extra/duplicate/unsupported/nested/symlink rejection")


def test_mask_and_shape_validation(root: Path) -> None:
    images, masks, _ = _make_fixture(root / "mask_validation")
    output = root / "must_not_exist.zip"

    _write_image(masks / "frame_a.png", np.zeros((18, 23), dtype=np.uint8))
    _require_raises(
        CvatMaskError,
        lambda: convert_masks_to_cvat_zip(
            images_dir=images, masks_dir=masks, output_zip=output
        ),
    )
    assert not output.exists()

    rgb = np.zeros((17, 23, 3), dtype=np.uint8)
    _write_image(masks / "frame_a.png", rgb)
    _require_raises(
        CvatMaskError,
        lambda: convert_masks_to_cvat_zip(
            images_dir=images, masks_dir=masks, output_zip=output
        ),
    )
    assert not output.exists()

    uint16 = np.zeros((17, 23), dtype=np.uint16)
    uint16[2:4, 2:4] = 255
    _write_image(masks / "frame_a.png", uint16)
    _require_raises(
        CvatMaskError,
        lambda: convert_masks_to_cvat_zip(
            images_dir=images, masks_dir=masks, output_zip=output
        ),
    )
    assert not output.exists()

    unknown = np.zeros((17, 23), dtype=np.uint8)
    unknown[2, 2] = 128
    _write_image(masks / "frame_a.png", unknown)
    _require_raises(
        CvatMaskError,
        lambda: convert_masks_to_cvat_zip(
            images_dir=images, masks_dir=masks, output_zip=output
        ),
    )
    assert not output.exists()
    print("[OK] shape/RGB/uint16/unknown-value rejection without partial ZIP")


def _copy_zip_with_extra(
    source: Path,
    destination: Path,
    member_name: str,
    payload: bytes,
) -> None:
    with zipfile.ZipFile(source, "r") as input_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            output_archive.writestr(info, input_archive.read(info.filename))
        output_archive.writestr(member_name, payload)


def _copy_zip_replacing_member(
    source: Path,
    destination: Path,
    member_name: str,
    payload: bytes,
) -> None:
    with zipfile.ZipFile(source, "r") as input_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            output_archive.writestr(
                info,
                payload if info.filename == member_name else input_archive.read(info.filename),
            )


def _copy_zip_omitting_members(
    source: Path,
    destination: Path,
    omitted: set[str],
) -> None:
    with zipfile.ZipFile(source, "r") as input_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            if info.filename not in omitted:
                output_archive.writestr(info, input_archive.read(info.filename))


def test_cvat_export_labelmap_header(root: Path) -> None:
    images, masks, _ = _make_fixture(root / "labelmap_header")
    valid = root / "labelmap_header_input.zip"
    convert_masks_to_cvat_zip(images_dir=images, masks_dir=masks, output_zip=valid)

    with zipfile.ZipFile(valid, "r") as archive:
        base_labelmap = archive.read(LABELMAP_PATH)

    headered = root / "labelmap_header_cvat.zip"
    _copy_zip_replacing_member(
        valid,
        headered,
        LABELMAP_PATH,
        f"{CVAT_LABELMAP_HEADER}\n".encode("utf-8") + base_labelmap,
    )
    summary = validate_cvat_segmentation_mask_zip(headered, images_dir=images)
    assert summary.status == "ok"

    unknown_comment = root / "labelmap_header_unknown.zip"
    _copy_zip_replacing_member(
        valid,
        unknown_comment,
        LABELMAP_PATH,
        b"# unexpected header\n" + base_labelmap,
    )
    _require_raises(
        CvatMaskError,
        lambda: validate_cvat_segmentation_mask_zip(unknown_comment),
    )
    print("[OK] CVAT optional labelmap header and strict label contract")


def test_cvat_images_prefix_round_trip(root: Path) -> None:
    images, masks, source_masks = _make_fixture(root / "images_prefix")
    valid = root / "images_prefix_input.zip"
    exported = root / "images_prefix_cvat.zip"
    convert_masks_to_cvat_zip(images_dir=images, masks_dir=masks, output_zip=valid)

    with zipfile.ZipFile(valid, "r") as input_archive, zipfile.ZipFile(
        exported, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            name = info.filename
            payload = input_archive.read(info)
            if name == DEFAULT_SET_PATH:
                stems = payload.decode("utf-8").splitlines()
                payload = "".join(f"images/{stem}\n" for stem in stems).encode("utf-8")
            elif name.startswith(CLASS_PREFIX):
                name = f"{CLASS_PREFIX}images/{name[len(CLASS_PREFIX):]}"
            elif name.startswith(OBJECT_PREFIX):
                name = f"{OBJECT_PREFIX}images/{name[len(OBJECT_PREFIX):]}"
            output_archive.writestr(name, payload)

    summary, loaded = read_cvat_segmentation_class_masks(
        exported,
        images_dir=images,
    )
    assert summary.stems == ["frame_a", "frame_b", "frame_c"]
    assert sorted(loaded) == summary.stems
    for stem in summary.stems:
        np.testing.assert_array_equal(loaded[stem], normalize_binary_mask(source_masks[stem]))
    print("[OK] CVAT images/ stem prefix normalization and mask loading")


def test_cvat_rgb_mask_round_trip(root: Path) -> None:
    images, masks, source_masks = _make_fixture(root / "rgb_export")
    valid = root / "rgb_export_input.zip"
    exported = root / "rgb_export_cvat.zip"
    cvat_assigned_rgb = (17, 91, 203)
    convert_masks_to_cvat_zip(images_dir=images, masks_dir=masks, output_zip=valid)

    with zipfile.ZipFile(valid, "r") as input_archive, zipfile.ZipFile(
        exported, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            name = info.filename
            payload = input_archive.read(info)
            if name == LABELMAP_PATH:
                payload = (
                    f"{CVAT_LABELMAP_HEADER}\n"
                    "background:0,0,0::\n"
                    f"femur:{','.join(str(value) for value in cvat_assigned_rgb)}::\n"
                ).encode("utf-8")
            elif name.startswith((CLASS_PREFIX, OBJECT_PREFIX)):
                stem = Path(name).stem
                foreground = normalize_binary_mask(source_masks[stem]).astype(bool)
                bgr = np.zeros((*foreground.shape, 3), dtype=np.uint8)
                bgr[foreground] = (
                    cvat_assigned_rgb[::-1]
                    if name.startswith(CLASS_PREFIX)
                    else (0, 255, 0)
                )
                success, encoded = cv2.imencode(".png", bgr)
                assert success
                payload = encoded.tobytes()
            output_archive.writestr(name, payload)

    summary, loaded = read_cvat_segmentation_class_masks(exported, images_dir=images)
    assert summary.unique_indices == [0, 1]
    for stem in summary.stems:
        np.testing.assert_array_equal(loaded[stem], normalize_binary_mask(source_masks[stem]))

    unknown_color = root / "rgb_export_unknown_color.zip"
    bad = np.zeros((17, 23, 3), dtype=np.uint8)
    bad[2:6, 3:8] = (255, 0, 0)  # RGB blue, absent from the fixed labelmap.
    success, encoded = cv2.imencode(".png", bad)
    assert success
    _copy_zip_replacing_member(
        valid,
        unknown_color,
        f"{CLASS_PREFIX}frame_a.png",
        encoded.tobytes(),
    )
    _require_raises(
        CvatMaskError,
        lambda: validate_cvat_segmentation_mask_zip(unknown_color),
    )
    print("[OK] CVAT-assigned RGB class colors and object support normalization")


def test_explicit_context_only_missing_mask_pair(root: Path) -> None:
    images, masks, _ = _make_fixture(root / "context_only_missing")
    valid = root / "context_only_complete.zip"
    missing_pair = root / "context_only_missing_pair.zip"
    partial_pair = root / "context_only_partial_pair.zip"
    convert_masks_to_cvat_zip(images_dir=images, masks_dir=masks, output_zip=valid)
    pair = {
        f"{CLASS_PREFIX}frame_a.png",
        f"{OBJECT_PREFIX}frame_a.png",
    }
    _copy_zip_omitting_members(valid, missing_pair, pair)

    _require_raises(
        CvatMaskError,
        lambda: validate_cvat_segmentation_mask_zip(missing_pair, images_dir=images),
    )
    summary, loaded = read_cvat_segmentation_class_masks(
        missing_pair,
        images_dir=images,
        allowed_missing_mask_stems=["frame_a"],
    )
    assert summary.images == 3
    assert summary.masks == 2
    assert summary.missing_masks == 1
    assert summary.missing_mask_stems == ["frame_a"]
    assert sorted(loaded) == ["frame_a", "frame_b", "frame_c"]
    assert not np.any(loaded["frame_a"])

    allowlist = root / "context_only_stems.txt"
    allowlist.write_text("frame_a\n", encoding="utf-8")
    assert load_allowed_missing_stems_file(allowlist) == ["frame_a"]
    _require_raises(
        CvatMaskError,
        lambda: validate_cvat_segmentation_mask_zip(
            missing_pair,
            images_dir=images,
            allowed_missing_mask_stems=["unknown"],
        ),
    )
    _copy_zip_omitting_members(
        valid,
        partial_pair,
        {f"{OBJECT_PREFIX}frame_a.png"},
    )
    _require_raises(
        CvatMaskError,
        lambda: validate_cvat_segmentation_mask_zip(
            partial_pair,
            images_dir=images,
            allowed_missing_mask_stems=["frame_a"],
        ),
    )
    print("[OK] explicit context-only missing pair and zero-mask synthesis")


def test_zip_validator_failures(root: Path) -> None:
    images, masks, _ = _make_fixture(root / "zip_validation")
    valid = root / "valid.zip"
    convert_masks_to_cvat_zip(images_dir=images, masks_dir=masks, output_zip=valid)

    unknown = root / "unknown_member.zip"
    _copy_zip_with_extra(valid, unknown, "unexpected.txt", b"x")
    _require_raises(
        CvatMaskError, lambda: validate_cvat_segmentation_mask_zip(unknown)
    )

    unsafe = root / "unsafe.zip"
    _copy_zip_with_extra(valid, unsafe, "../escape.txt", b"x")
    _require_raises(
        CvatMaskError, lambda: validate_cvat_segmentation_mask_zip(unsafe)
    )

    duplicate = root / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(valid, "r") as input_archive, zipfile.ZipFile(
            duplicate, "w", compression=zipfile.ZIP_DEFLATED
        ) as output_archive:
            for info in input_archive.infolist():
                output_archive.writestr(info, input_archive.read(info.filename))
            output_archive.writestr(LABELMAP_PATH, b"duplicate")
    _require_raises(
        CvatMaskError, lambda: validate_cvat_segmentation_mask_zip(duplicate)
    )

    unknown_index = root / "unknown_index.zip"
    with zipfile.ZipFile(valid, "r") as input_archive, zipfile.ZipFile(
        unknown_index, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        target = f"{CLASS_PREFIX}frame_a.png"
        bad_mask = np.full((17, 23), 2, dtype=np.uint8)
        success, encoded = cv2.imencode(".png", bad_mask)
        assert success
        for info in input_archive.infolist():
            payload = encoded.tobytes() if info.filename == target else input_archive.read(info.filename)
            output_archive.writestr(info, payload)
    _require_raises(
        CvatMaskError,
        lambda: validate_cvat_segmentation_mask_zip(unknown_index),
    )
    print("[OK] unknown/unsafe/duplicate/unknown-index ZIP rejection")


def test_label_and_overwrite_validation(root: Path) -> None:
    for invalid in ("", " femur", "background", "bad:name", "a/b", "a\nb"):
        _require_raises(CvatMaskError, lambda value=invalid: validate_label_name(value))

    images, masks, _ = _make_fixture(root / "overwrite")
    output = root / "existing.zip"
    first = convert_masks_to_cvat_zip(
        images_dir=images, masks_dir=masks, output_zip=output
    )
    _require_raises(
        FileExistsError,
        lambda: convert_masks_to_cvat_zip(
            images_dir=images, masks_dir=masks, output_zip=output
        ),
    )
    second = convert_masks_to_cvat_zip(
        images_dir=images,
        masks_dir=masks,
        output_zip=output,
        overwrite=True,
    )
    assert first.zip_sha256 == second.zip_sha256
    print("[OK] label validation and explicit overwrite behavior")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stage4_cvat_mask_export_") as temporary:
        root = Path(temporary)
        test_normal_conversion_and_determinism(root)
        test_pairing_and_filesystem_validation(root)
        test_mask_and_shape_validation(root)
        test_cvat_export_labelmap_header(root)
        test_cvat_images_prefix_round_trip(root)
        test_cvat_rgb_mask_round_trip(root)
        test_explicit_context_only_missing_mask_pair(root)
        test_zip_validator_failures(root)
        test_label_and_overwrite_validation(root)
    print("Stage 4 CVAT Segmentation Mask 1.1 synthetic checks passed.")


if __name__ == "__main__":
    main()
