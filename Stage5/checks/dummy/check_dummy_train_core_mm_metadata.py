from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from stage5.utils.split_identity import file_sha256  # noqa: E402
from checks.real_h5.audit_stage5_train_core_mm_metadata import read_spacing  # noqa: E402

# ----------------------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the train_core mm-metadata audit.
#
# Report 9.7-2, 10.1. The thing worth testing is the audit's RESTRAINT. It
# establishes availability, and it must not:
#   * read a spacing of 1.0 as a measured millimetre scale (it is the
#     --spacing_x/--spacing_y default, documented as a temporary visualization
#     value, and it is expressed against the local crop plane, not the frame)
#   * report the mm scale as "unavailable" when what is actually true is
#     "unconfirmed" -- the user reported the real scales are obtainable
#   * compute a surrogate FL or infer T_FL
#   * borrow one video's metadata for another when resolution fails
#
# numpy/h5py; no torch, no CUDA. Synthetic H5s only.
# ----------------------------------------------------------------------------

TOOL = REPO_ROOT / "checks" / "real_h5" / "audit_stage5_train_core_mm_metadata.py"
SUFFIX = (
    "_pointcloud_annotated_foreground_combined_v2_global_local_l75_w31_c12_area15_"
    "bboxrank_v7_cvat_authoritative_crop_quality_v1.h5"
)

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def write_intermediate(path: Path, *, spacing=(1.0, 1.0, 1.0), square: bool = True,
                       complete: bool = True, with_spacing: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        shape = "(40, 1, 256, 256)" if square else "(40, 1, 256, 128)"
        f.attrs["local_input_shape"] = shape
        if complete:
            f.attrs["raw_width"] = 640
            f.attrs["raw_height"] = 480
            f.attrs["local_crop_top"] = 10
            f.attrs["local_crop_left"] = 20
            f.attrs["local_resize_scale"] = 0.5
        if with_spacing:
            f.create_dataset("spacing", data=np.asarray(spacing, dtype=np.float32))
            f.create_dataset("dimensions", data=np.asarray([256, 256, 1], dtype=np.float32))
    return path


def write_final(path: Path, source: Path | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if source is not None:
            f.attrs["source_pseudo3d_h5"] = str(source)
    return path


def test_spacing_interpretation(root: Path) -> None:
    print("\n[1] a spacing of 1.0 is a placeholder, never a measured scale")
    default = write_intermediate(root / "default.h5")
    got = read_spacing(default)
    check(got["status"] == "ok", "the spacing dataset is read")
    check(got["looks_like_placeholder_default"] is True, "1.0/1.0 is flagged as the default")
    check("not a measured mm scale" in got["interpretation"], "and is explicitly not called a scale")
    check(
        "local crop plane" in got["reference_plane"],
        "the reference plane is recorded as the crop, not the original frame",
    )

    real = write_intermediate(root / "real.h5", spacing=(0.12, 0.15, 1.0))
    got = read_spacing(real)
    check(got["looks_like_placeholder_default"] is False, "a non-default value is not flagged as default")
    check(got["x_equals_y"] is False, "x and y are compared separately")
    check("still need confirming" in got["interpretation"], "its unit and frame still need confirming")

    anisotropic = read_spacing(write_intermediate(root / "aniso.h5", spacing=(0.12, 0.12, 1.0)))
    check(anisotropic["x_equals_y"] is True, "equal x/y is reported as such rather than assumed")

    absent = read_spacing(write_intermediate(root / "nospacing.h5", with_spacing=False))
    check(absent["status"] == "spacing_dataset_absent", "a missing spacing dataset is reported, not defaulted")

    check(
        read_spacing(root / "does_not_exist.h5")["status"].startswith("intermediate_h5_unreadable"),
        "an unreadable intermediate is reported rather than skipped",
    )


def run_tool(root: Path, finals: list[Path]) -> subprocess.CompletedProcess:
    listing = root / "train_core.txt"
    listing.write_text("\n".join(str(p) for p in finals) + "\n", encoding="utf-8")
    private = root / "mm_private.json"
    share = root / "mm_share.json"
    result = subprocess.run(
        [sys.executable, str(TOOL),
         "--train_core_list", str(listing),
         "--intermediate_root", str(root / "inter"),
         "--private_json", str(private), "--shareable_json", str(share)],
        capture_output=True, text=True,
    )
    result.private = private  # type: ignore[attr-defined]
    result.share = share  # type: ignore[attr-defined]
    return result


def test_requires_a_contract(root: Path) -> None:
    print("\n[2] the audit refuses to read train_core without a contract")
    inter = write_intermediate(root / "inter" / "20250701_101010_101_pseudo3d.h5")
    final = write_final(root / f"20250701_101010_101{SUFFIX}", inter)
    result = run_tool(root, [final])
    check(result.returncode != 0, "no approved split contract means no audit")
    check(not Path(result.private).is_file(), "and nothing is written")


def test_conclusion_is_unconfirmed_not_unavailable() -> None:
    print("\n[3] the mm scale is reported UNCONFIRMED, and nothing is inferred")
    source = TOOL.read_text(encoding="utf-8")
    check('"mm_scale": "UNCONFIRMED"' in source, "the conclusion is UNCONFIRMED")
    check(
        "NOT evidence that no source exists" in source,
        "it records that absence here is not evidence that no source exists",
    )
    check(
        "no surrogate FL was computed or selected" in source
        and "no T_FL was inferred" in source,
        "it states that no surrogate FL and no T_FL were produced",
    )
    check(
        "no value was borrowed from another video" in source,
        "it states that no value is borrowed between videos",
    )
    check(
        "temporary visualization value" in source or "temporary visualization" in source,
        "it records why `spacing` is not a measurement",
    )
    check(
        not re.search(r"\bT_FL\s*=", source) and "surrogate_fl" not in source.lower(),
        "the tool computes no T_FL or surrogate FL anywhere",
    )
    check(
        "validation" not in source.split("def parse_args")[0].split("PLACEHOLDER_SPACING")[0].lower()
        or "train_core only" in source,
        "its scope is stated as train_core only",
    )


def test_unresolved_videos_are_reported_not_filled_in() -> None:
    print("\n[4] an unresolvable video is reported, not filled in from a neighbour")
    source = TOOL.read_text(encoding="utf-8")
    check("intermediate_h5_unresolved" in source, "unresolved videos get their own status")
    check("final_h5_missing" in source, "a missing final H5 gets its own status")
    check(
        "intermediate_unresolved" in source and "crop_inverse_transform_fields_complete" in source,
        "the summary counts gaps separately from successes",
    )
    check(
        "incomplete for some videos" in source,
        "a partially complete crop transform is not rounded up to 'available'",
    )


def test_end_to_end_writes_both_records(root: Path) -> None:
    print("\n[5] the audit runs to completion and writes both records")
    # The first real run completed the whole audit and then failed at the
    # JSON write, because h5py hands attributes back as numpy scalars
    # (raw_width is an int64, not an int). No test reached the write path --
    # the only end-to-end case asserted a REFUSAL. This one runs it through.
    import json

    data = root / "data"
    finals, ids = [], []
    for index in range(3):
        ident = f"2025070{index}_12000{index}_{index}"
        ids.append(ident)
        inter = write_intermediate(root / "inter" / f"{ident}_pseudo3d.h5")
        finals.append(write_final(data / f"{ident}{SUFFIX}", inter))

    listing = root / "train_core.txt"
    listing.write_text("\n".join(str(p) for p in finals) + "\n", encoding="utf-8")

    # A synthetic contract: the tool refuses to read anything without one.
    from stage5.utils.file_list_mode import list_identity_sha256

    identity = list_identity_sha256(finals)
    registry = root / "registry.json"
    registry.write_text(json.dumps({
        "schema": "stage5_seal_registry_v1",
        "sealed_video_identities": ["20259999_999999_9"],
        "sealed_list_identity_sha256": "sealed-list-hash",
        "sealed_until": "S5-20c",
    }, indent=2), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "stage5_split_contract_v1",
        "name": "synthetic",
        "lists": {
            "train_core": {"count": len(finals), "identity_sha256": identity, "content_sha256": "c"},
            "internal_test": {"count": 1, "identity_sha256": "sealed-list-hash", "content_sha256": "c"},
        },
        "sealed_splits": ["internal_test"],
        "seal_registry_required": True,
        "allows_new_training": True,
        "allows_directory_mode": False,
    }, indent=2), encoding="utf-8")
    pins = root / "pins.json"
    pins.write_text(json.dumps({
        "schema": "stage5_split_contract_pins_v1",
        "pins": {
            "seal_registry": {"path": str(registry), "expected_sha256": file_sha256(registry)},
            "synthetic": {"path": str(manifest), "expected_sha256": file_sha256(manifest)},
        },
    }, indent=2), encoding="utf-8")

    private, share = root / "mm_private.json", root / "mm_share.json"
    result = subprocess.run(
        [sys.executable, str(TOOL),
         "--train_core_list", str(listing),
         "--split_manifest", "synthetic", "--split_contract_pins", str(pins),
         "--intermediate_root", str(root / "inter"),
         "--private_json", str(private), "--shareable_json", str(share)],
        capture_output=True, text=True,
    )
    check(result.returncode == 0, f"the audit completes (stderr: {result.stderr.strip()[-300:]})")
    if result.returncode != 0:
        return

    check(private.is_file() and share.is_file(), "both records are written")
    priv = json.loads(private.read_text(encoding="utf-8"))
    shared = json.loads(share.read_text(encoding="utf-8"))
    check(priv["findings"]["num_train_core"] == 3, "the private record counts the inputs")
    check(
        priv["findings"]["crop_inverse_transform_fields_complete"] == 3,
        "the numpy-typed crop attrs survived serialisation with their values",
    )
    check(
        priv["per_video"][0]["crop_inverse_transform"]["raw_width"] == 640,
        "an int64 attribute is written as a number, not dropped or stringified",
    )
    check(
        priv["findings"]["spacing_is_placeholder_default"] == 3,
        "the placeholder spacing is recognised end to end",
    )
    check(shared["conclusion"]["mm_scale"] == "UNCONFIRMED", "the shared record says UNCONFIRMED")
    check("per_video" not in shared, "the shared record carries no per-video detail")
    check(
        not re.search(r"[0-9]{8}_[0-9]{6}_[0-9]+", json.dumps(shared)),
        "and no video identifier",
    )


def main() -> None:
    print("Stage5 S5-16 Step 0: train_core mm metadata synthetic checks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_spacing_interpretation(root / "spacing")
        test_requires_a_contract(root / "contract")
        test_conclusion_is_unconfirmed_not_unavailable()
        test_unresolved_videos_are_reported_not_filled_in()
        test_end_to_end_writes_both_records(root / "e2e")
    print(f"\nchecks run: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
