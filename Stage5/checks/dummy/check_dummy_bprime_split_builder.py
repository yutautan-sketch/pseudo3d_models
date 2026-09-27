from __future__ import annotations

import json
import os
from collections import Counter
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from stage5.utils.split_contract import SplitContractError, resolve_active_contract  # noqa: E402
from stage5.utils.split_identity import contains_video_identity, file_sha256  # noqa: E402

# ----------------------------------------------------------------------------
# S5-16 Step 0: end-to-end synthetic coverage for the split BUILDER.
#
# The allocation maths and the contract primitives are covered elsewhere; this
# exercises the tool that will actually perform the irreversible split, over
# synthetic H5s, from dry run through to a pinned contract that denies the
# sealed videos.
#
# What it pins down:
#   * the dry run releases the clinical-FL marginal and NOTHING else -- no
#     multi-region marginal, no cross table, no per-cell allocation. Those
#     three together would reconstruct internal_test's composition.
#   * sealed material is written into the sealed directory, with restrictive
#     modes, and never to a temp location first
#   * --confirm re-uses the fingerprinted intermediate instead of re-reading
#     the H5s, and refuses a tampered one
#   * the six partition checks hold on real output: 144/18/18 shape, sanity
#     kept in train_core, validation untouched
#   * the loop closes: pin the registry and manifest, and the contract then
#     denies the sealed videos through the ordinary guard
#   * the initial build path will not run twice
#
# numpy/h5py (the builder imports the GT region counter); no torch, no CUDA.
# ----------------------------------------------------------------------------

TOOL = REPO_ROOT / "checks" / "real_h5" / "build_stage5_bprime_split.py"
SUFFIX = (
    "_pointcloud_annotated_foreground_combined_v2_global_local_l75_w31_c12_area15_"
    "bboxrank_v7_cvat_authoritative_crop_quality_v1.h5"
)
NUM_TRAIN = 24
NUM_VAL = 4
NUM_SANITY = 3
NUM_INTERNAL = 4

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def identity(index: int) -> str:
    return f"2025070{index % 10}_12{index:04d}_{index}"


TOTAL_FRAMES = 12


def write_h5(path: Path, *, two_regions: bool, num_gt_frames: int = 6) -> Path:
    """Background over TOTAL_FRAMES; GT positives over the first num_gt_frames.

    `num_gt_frames` is varied across the fixture on purpose: it IS the
    stratification quantity, so a fixture that gave every video the same count
    would be degenerate and the constructibility condition would (correctly)
    refuse to split it. `two_regions` puts a second blob far away.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    xs, ys, frames, labels = [], [], [], []
    for frame in range(TOTAL_FRAMES):
        if frame < num_gt_frames:
            for i in range(8):
                xs.append(10.0 + i)
                ys.append(10.0)
                frames.append(frame)
                labels.append(1)
            if two_regions:
                for i in range(8):
                    xs.append(200.0 + i)
                    ys.append(200.0)
                    frames.append(frame)
                    labels.append(1)
        for i in range(10):
            xs.append(50.0 + i)
            ys.append(80.0)
            frames.append(frame)
            labels.append(0)
    with h5py.File(path, "w") as f:
        pc = f.create_group("point_cloud")
        pc.create_dataset("pixel_xy", data=np.stack([xs, ys], axis=1).astype(np.float32))
        pc.create_dataset("frame_order", data=np.asarray(frames, dtype=np.int64))
        ann = f.create_group("annotation")
        ann.create_dataset("point_label", data=np.asarray(labels, dtype=np.int64))
        ann.create_dataset("valid_mask", data=np.ones(len(labels), dtype=bool))
    return path


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        data = root / "data"
        # The GT frame count varies across videos: it is the stratification
        # quantity, so a constant fixture would be degenerate by construction.
        self.gt_frames = {i: 3 + (i % 9) for i in range(NUM_TRAIN)}
        self.train_paths = [
            write_h5(data / f"{identity(i)}{SUFFIX}", two_regions=(i % 3 == 0),
                     num_gt_frames=self.gt_frames[i])
            for i in range(NUM_TRAIN)
        ]
        self.val_paths = [
            write_h5(data / f"{identity(500 + i)}{SUFFIX}", two_regions=False, num_gt_frames=5)
            for i in range(NUM_VAL)
        ]
        self.sanity_paths = self.train_paths[:NUM_SANITY]
        self.candidates = [
            p.name.split("_pointcloud")[0] for p in self.train_paths[NUM_SANITY:]
        ]
        self.train_list = self._list("train.txt", self.train_paths)
        self.val_list = self._list("val.txt", self.val_paths)
        self.sanity_list = self._list("sanity.txt", self.sanity_paths)
        self.audit = root / "input_audit.json"
        self.audit.write_text(
            json.dumps(
                {
                    "schema": "stage5_step0_input_audit_v2",
                    "complete": True,
                    "expected_sha256": {
                        "old_train_list": file_sha256(self.train_list),
                        "old_val_list": file_sha256(self.val_list),
                        "sanity_list": file_sha256(self.sanity_list),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.out = root / "split_out"

    def _list(self, name: str, paths: list[Path]) -> Path:
        path = self.root / name
        path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
        return path

    def _empty_pins(self) -> Path:
        """A pins file of this fixture's own.

        Without this the tool falls back to the repo's real pins file, so the
        test would pass or fail depending on whether a split has been sealed
        in production. A synthetic test must not read production state.
        """
        path = self.root / "empty_pins.json"
        if not path.is_file():
            path.write_text(
                json.dumps({"schema": "stage5_split_contract_pins_v1", "pins": {}}, indent=2),
                encoding="utf-8",
            )
        return path

    def run(self, *extra: str, out_dir: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL), "--bootstrap",
             "--old_train_list", str(self.train_list),
             "--old_val_list", str(self.val_list),
             "--sanity_list", str(self.sanity_list),
             "--input_audit_json", str(self.audit),
             "--split_contract_pins", str(self._empty_pins()),
             "--output_dir", str(out_dir or self.out),
             "--internal_test_size", str(NUM_INTERNAL),
             *extra],
            capture_output=True, text=True,
        )


def test_dry_run_releases_only_the_allowed_aggregate(fx: Fixture) -> str:
    print("\n[1] the dry run releases the candidate count and the boolean, and nothing else")
    result = fx.run("--dry_run_stratification")
    check(result.returncode == 0, f"the dry run succeeds (stderr: {result.stderr.strip()[:300]})")
    if result.returncode != 0:
        return ""
    check("NO split was drawn" in result.stdout, "it states that no split was drawn")

    share = json.loads((fx.out / "stratification_feasibility_SHARE.json").read_text(encoding="utf-8"))
    check(share["num_candidates"] == len(fx.candidates), "the candidate count is released")
    check(share["strata_constructible"] is True, "the constructibility boolean is released")

    # The shared GT aggregate is the count and the boolean, and nothing else:
    # group sizes plus the allocation rule would reconstruct the composition.
    #
    # This is checked on the STRUCTURE, not by word-matching the file. The
    # `withheld` notice deliberately names the things it withholds ("group
    # sizes, medians, ... per-cell allocation"), so a substring search for
    # those words finds the notice rather than a leak.
    check(
        set(share) == {"schema", "num_candidates", "strata_constructible",
                       "intermediate_sha256", "schema_versions", "withheld"},
        f"the shared file releases exactly the allowed keys (got {sorted(share)})",
    )
    numeric = {
        key: value for key, value in share.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    check(
        set(numeric) == {"num_candidates"},
        f"the only number released is the candidate count (got {sorted(numeric)})",
    )
    for absent in ("multi_region", "allocation", "group_sizes", "medians",
                   "stratification_values", "stratum_group", "boundary_tie_members",
                   "zero_valued_candidates"):
        check(absent not in share, f"{absent} is not a released key")
    check(
        "medians" in share["withheld"] and "allocation" in share["withheld"],
        "the notice names what is being withheld",
    )
    # The linking information is a different kind and must survive.
    check("intermediate_sha256" in share, "the fingerprint the confirm step needs is kept")
    check(
        share["schema_versions"]["quantity"] == "gt_positive_frame_count_v1"
        and share["schema_versions"]["condition"] == "strictly_increasing_group_medians_v1",
        "the spec versions travel with it",
    )
    # A video ID must not appear ANYWHERE in the file, notice included, so
    # this one is checked against the whole serialised form.
    check(
        not contains_video_identity(json.dumps(share)),
        "the shared file carries no video ID",
    )
    check(
        not contains_video_identity(result.stdout),
        "the dry run's stdout carries no video ID",
    )
    check(
        "sealed" in result.stdout,
        "stdout says the remaining figures are sealed",
    )

    sealed = fx.out / "sealed"
    intermediate = sealed / "stratification_DO_NOT_SHARE.json"
    check(intermediate.is_file(), "the intermediate is written into the sealed directory")
    mode = stat.S_IMODE(os.stat(intermediate).st_mode)
    check(mode == 0o600, f"the sealed intermediate is owner-only (mode {oct(mode)})")
    check(
        stat.S_IMODE(os.stat(sealed).st_mode) == 0o700,
        "the sealed directory itself is owner-only",
    )

    payload = json.loads(intermediate.read_text(encoding="utf-8"))
    check(payload["schema"] == "stage5_bprime_stratification_v2", "the intermediate is v2")
    check("multi_region" in payload and "stratification_values" in payload,
          "the sealed side does hold the detail")
    check("boundary_tie_members" in payload,
          "boundary ties are recorded, so the median condition's limit is visible")
    check("zero_valued_candidates" in payload, "genuine zeros are counted")
    check(payload["condition_result"]["constructible"] is True, "the condition result is recorded")

    # The decisive check: the numbers the sealed side actually computed must
    # not appear anywhere in the shared file's values or in stdout. The
    # fingerprint is hex, so it is excluded from the search to avoid an
    # incidental digit match.
    sealed_numbers = set(payload["condition_result"]["medians"])
    sealed_numbers |= set(Counter(payload["stratum_group"].values()).values())
    sealed_numbers |= set(payload["stratification_values"].values())
    sealed_numbers.discard(len(fx.candidates))  # the candidate count is allowed out
    # Only the data-bearing fields: the schema strings contain digits of their
    # own ("stage5", "v2") and would produce spurious matches.
    released = json.dumps({k: v for k, v in share.items()
                           if k in ("num_candidates", "strata_constructible")})
    stdout_without_fingerprint = "\n".join(
        line for line in result.stdout.splitlines() if "sha256" not in line
    )
    leaked = sorted(
        n for n in sealed_numbers
        if str(n) in released or f": {n}" in stdout_without_fingerprint
    )
    check(not leaked, f"no sealed median, group size or per-video value is released (leaked: {leaked})")
    check(
        payload["spec"]["selection_method"]["name"] == "sha256_keyed_sort_v1",
        "the selection method is recorded in the intermediate",
    )

    printed = re.search(r"intermediate sha256\s*:\s*([0-9a-f]{64})", result.stdout)
    check(printed is not None, "the dry run prints the full intermediate fingerprint")
    check(
        printed is not None and printed.group(1) == file_sha256(intermediate),
        "and the printed fingerprint is the file's real hash",
    )
    return printed.group(1) if printed else ""


def test_confirm_requires_the_fingerprint(fx: Fixture, fingerprint: str) -> None:
    print("\n[2] --confirm will not proceed on an unverified intermediate")
    result = fx.run("--confirm")
    check(result.returncode == 2, "confirming without the fingerprint stops")
    check(not (fx.out / "train_core_144.txt").is_file(), "and writes no lists")

    intermediate = fx.out / "sealed" / "stratification_DO_NOT_SHARE.json"
    original = intermediate.read_text(encoding="utf-8")
    payload = json.loads(original)
    flipped = {k: (not v) for k, v in payload["multi_region"].items()}
    payload["multi_region"] = flipped
    intermediate.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = fx.run("--confirm", "--expected_intermediate_sha256", fingerprint)
    check(result.returncode == 2, "a tampered intermediate is refused")
    check("fingerprint" in result.stderr.lower(), "and the refusal names the fingerprint")
    intermediate.write_text(original, encoding="utf-8")

    original_list = fx.train_list.read_text(encoding="utf-8")
    fx.train_list.write_text(original_list + "# a trailing comment changes the file\n", encoding="utf-8")
    result = fx.run("--confirm", "--expected_intermediate_sha256", fingerprint)
    check(result.returncode == 2, "an input that no longer matches the audit record is refused")
    fx.train_list.write_text(original_list, encoding="utf-8")


def test_confirm_writes_a_valid_split(fx: Fixture, fingerprint: str) -> None:
    print("\n[4] --confirm draws the split and passes the six partition checks")
    result = fx.run("--confirm", "--expected_intermediate_sha256", fingerprint)
    check(result.returncode == 0, f"confirm succeeds (stderr: {result.stderr.strip()[:300]})")
    check("All six partition checks passed" in result.stdout, "it reports the partition checks")

    train_core = (fx.out / "train_core_144.txt").read_text(encoding="utf-8").split()
    validation = (fx.out / "validation_18.txt").read_text(encoding="utf-8").split()
    internal = (fx.out / "sealed" / "internal_test_18.txt").read_text(encoding="utf-8").split()

    check(len(internal) == NUM_INTERNAL, f"internal_test holds {NUM_INTERNAL}")
    check(len(train_core) == NUM_TRAIN - NUM_INTERNAL, "train_core is the rest of the old train list")
    check(validation == [str(p) for p in fx.val_paths], "validation is unchanged, in content and order")
    check(not (set(train_core) & set(internal)), "train_core and internal_test are disjoint")
    check(
        set(train_core) | set(internal) == {str(p) for p in fx.train_paths},
        "the two reconstruct the old train list exactly",
    )
    check(
        all(str(p) in train_core for p in fx.sanity_paths),
        "every train-sanity video stayed in train_core",
    )

    check((fx.out / "sealed" / "internal_test_18.txt").is_file(), "internal_test lives in the sealed dir")
    check(not (fx.out / "internal_test_18.txt").exists(), "and not outside it")
    alloc = fx.out / "sealed" / "allocation_DO_NOT_SHARE.json"
    check(alloc.is_file(), "the per-cell allocation is sealed")
    check(stat.S_IMODE(os.stat(alloc).st_mode) == 0o600, "and owner-only")

    check(
        not contains_video_identity(result.stdout),
        "confirm's stdout carries no video ID",
    )

    manifest = json.loads((fx.out / "s5_16_bprime.json").read_text(encoding="utf-8"))
    check(manifest["sealed_splits"] == ["internal_test"], "the manifest declares the sealed split")
    check(manifest["seal_registry_required"] is True, "and acknowledges the registry")
    check(manifest["allows_directory_mode"] is False, "directory mode is refused by the manifest")
    check(
        manifest["selection_method"]["name"] == "sha256_keyed_sort_v1",
        "the manifest records the selection method",
    )
    check(manifest["intermediate_sha256"] == fingerprint, "and the intermediate it was drawn from")

    registry = json.loads((fx.out / "sealed_registry.json").read_text(encoding="utf-8"))
    check(
        registry["sealed_list_identity_sha256"] == manifest["lists"]["internal_test"]["identity_sha256"],
        "the registry binds the sealed set to the list it came from",
    )
    check(len(registry["sealed_video_identities"]) == NUM_INTERNAL, "it seals the drawn videos")
    check(
        "permanently ineligible" in registry["prohibited_checkpoint_rule"],
        "the old-checkpoint prohibition survives the release",
    )


def test_pinning_closes_the_loop(fx: Fixture) -> None:
    print("\n[5] once pinned, the ordinary guard denies the sealed videos")
    registry = fx.out / "sealed_registry.json"
    manifest = fx.out / "s5_16_bprime.json"
    pins = fx.root / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "schema": "stage5_split_contract_pins_v1",
                "pins": {
                    "seal_registry": {
                        "path": str(registry), "expected_sha256": file_sha256(registry),
                        "approved_on": "2026-09-26", "approved_scope": "synthetic",
                    },
                    "s5_16_bprime": {
                        "path": str(manifest), "expected_sha256": file_sha256(manifest),
                        "approved_on": "2026-09-26", "approved_scope": "synthetic",
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    contract = resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins)
    check(contract.manifest.name == "s5_16_bprime", "the freshly built contract loads")

    internal = (fx.out / "sealed" / "internal_test_18.txt").read_text(encoding="utf-8").split()
    train_core = (fx.out / "train_core_144.txt").read_text(encoding="utf-8").split()
    try:
        contract.assert_paths_allowed(internal, purpose="synthetic")
        check(False, "a sealed video was allowed through")
    except SplitContractError:
        check(True, "the sealed videos are denied through the ordinary guard")
    contract.assert_paths_allowed(train_core, purpose="synthetic")
    check(True, "train_core passes")
    contract.assert_new_training_allowed()
    check(True, "new training on this split is permitted")


def test_initial_build_will_not_run_twice(fx: Fixture, fingerprint: str) -> None:
    print("\n[6] the initial build path will not overwrite a confirmed split")
    result = fx.run("--confirm", "--expected_intermediate_sha256", fingerprint)
    check(result.returncode == 2, "re-running confirm after a registry exists is refused")
    check("already exists" in result.stderr or "already approved" in result.stderr,
          "and says the registry is already there")
    result = fx.run("--dry_run_stratification")
    check(result.returncode == 2, "so is re-running the dry run")


def test_raw_value_validation(root: Path) -> None:
    print("\n[7] invalid inputs are caught BEFORE the astype that would hide them")
    import numpy as np

    from checks.real_h5.build_stage5_bprime_split import (
        SplitSpecError,
        gt_positive_frame_count,
        read_and_validate_gt_points,
    )

    def make(path: Path, *, xy=None, frame=None, label=None, valid=None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 4
        xy = np.zeros((n, 2), dtype=np.float32) if xy is None else xy
        frame = np.zeros(n, dtype=np.int64) if frame is None else frame
        label = np.array([1, 1, 0, 0], dtype=np.int64) if label is None else label
        valid = (label != -1) if valid is None else valid
        with h5py.File(path, "w") as f:
            g = f.create_group("point_cloud")
            g.create_dataset("pixel_xy", data=xy)
            g.create_dataset("frame_order", data=frame)
            a = f.create_group("annotation")
            a.create_dataset("point_label", data=label)
            a.create_dataset("valid_mask", data=valid)
        return path

    def expect_stop(path: Path, label: str) -> None:
        try:
            read_and_validate_gt_points(path)
            check(False, f"{label} (was accepted)")
        except SplitSpecError:
            check(True, label)

    good = make(root / "good.h5")
    data = read_and_validate_gt_points(good)
    check(gt_positive_frame_count(data) == 1, "a valid file yields the expected frame count")

    # Each of these is silently corrupted by the checker's astype-first reader.
    expect_stop(
        make(root / "frac_frame.h5", frame=np.array([0.0, 3.7, 3.7, 1.0], dtype=np.float64)),
        "a non-integral frame_order stops instead of truncating to another frame",
    )
    expect_stop(
        make(root / "inf_frame.h5", frame=np.array([0.0, np.inf, 1.0, 2.0], dtype=np.float64)),
        "an Inf frame_order stops",
    )
    expect_stop(
        make(root / "nan_frame.h5", frame=np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float64)),
        "a NaN frame_order stops",
    )
    expect_stop(
        make(root / "huge_frame.h5", frame=np.array([0.0, 1e30, 1.0, 2.0], dtype=np.float64)),
        "an integral float beyond the int64 range stops",
    )
    expect_stop(
        make(root / "neg_frame.h5", frame=np.array([0, -1, 1, 2], dtype=np.int64)),
        "a negative frame_order stops",
    )
    expect_stop(
        make(root / "mask_two.h5", valid=np.array([2, 1, 1, 1], dtype=np.int64)),
        "a valid_mask of 2 stops instead of becoming True",
    )
    expect_stop(
        make(root / "mask_nan.h5", valid=np.array([np.nan, 1.0, 1.0, 1.0], dtype=np.float64)),
        "a NaN valid_mask stops instead of becoming True",
    )
    expect_stop(
        make(root / "xy3.h5", xy=np.zeros((4, 3), dtype=np.float32)),
        "a pixel_xy of shape (N, 3) stops instead of passing the row check",
    )
    expect_stop(
        make(root / "xy_nan.h5", xy=np.full((4, 2), np.nan, dtype=np.float32)),
        "a pixel_xy holding NaN stops (it would break the clustering)",
    )
    expect_stop(
        make(root / "bad_label.h5", label=np.array([1, 1, 5, 0], dtype=np.int64),
             valid=np.ones(4, dtype=bool)),
        "a label outside {-1, 0, 1} stops",
    )
    expect_stop(
        make(root / "mask_mismatch.h5", label=np.array([1, 1, -1, 0], dtype=np.int64),
             valid=np.ones(4, dtype=bool)),
        "valid_mask disagreeing with point_label != -1 stops",
    )
    expect_stop(
        make(root / "empty.h5", xy=np.zeros((0, 2), dtype=np.float32),
             frame=np.zeros(0, dtype=np.int64), label=np.zeros(0, dtype=np.int64),
             valid=np.zeros(0, dtype=bool)),
        "an empty point cloud stops (distinct from a genuine zero frame count)",
    )

    # A genuine zero is a value, not a failure.
    all_background = make(root / "zero.h5", label=np.zeros(4, dtype=np.int64),
                          valid=np.ones(4, dtype=bool))
    check(
        gt_positive_frame_count(read_and_validate_gt_points(all_background)) == 0,
        "a video with no GT-positive point yields 0 and is NOT treated as a failure",
    )


def test_settings_mismatch_is_refused(fx: Fixture, fingerprint: str) -> None:
    print("\n[3] confirming with different settings than the dry run is refused")
    for extra, label in (
        (["--min_component_points", "3"], "a different min_component_points"),
        (["--seed", "43"], "a different seed"),
        (["--internal_test_size", "5"], "a different internal_test_size"),
        (["--link_distances", "2,4"], "different link distances"),
    ):
        result = fx.run("--confirm", "--expected_intermediate_sha256", fingerprint, *extra)
        check(result.returncode == 2, f"{label} stops the confirm run")
        check(
            "differs between the dry run and this confirm run" in result.stderr
            or "recorded" in result.stderr,
            f"{label}: the refusal says the two disagree rather than preferring one",
        )

    # An unknown spec version is refused even with no CLI argument supplied.
    intermediate = fx.out / "sealed" / "stratification_DO_NOT_SHARE.json"
    original = intermediate.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["spec"]["quantity_version"] = "gt_positive_frame_count_v99"
    intermediate.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = fx.run("--confirm", "--expected_intermediate_sha256", file_sha256(intermediate))
    check(result.returncode == 2, "an unsupported quantity version stops")
    check("not supported" in result.stderr, "and says the implementation does not support it")
    intermediate.write_text(original, encoding="utf-8")

    payload = json.loads(original)
    payload["schema"] = "stage5_bprime_stratification_v1"
    intermediate.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = fx.run("--confirm", "--expected_intermediate_sha256", file_sha256(intermediate))
    check(result.returncode == 2, "a v1 intermediate is refused")
    intermediate.write_text(original, encoding="utf-8")


def test_degenerate_quantity_stops(root: Path) -> None:
    print("\n[8] a quantity that does not separate the groups stops the dry run")
    # Every video with the same GT frame count. The rank split still forms
    # three groups, so "three groups received members" would wave this
    # through; the median condition is what catches it. (The first version of
    # this fixture was accidentally like this, which is how the gap showed.)
    fx = Fixture(root)
    data = root / "data"
    for index in range(NUM_TRAIN):
        write_h5(data / f"{identity(index)}{SUFFIX}", two_regions=(index % 3 == 0),
                 num_gt_frames=6)
    # the lists still point at those paths, but their contents changed
    fx.audit.write_text(
        json.dumps(
            {
                "schema": "stage5_step0_input_audit_v2",
                "complete": True,
                "expected_sha256": {
                    "old_train_list": file_sha256(fx.train_list),
                    "old_val_list": file_sha256(fx.val_list),
                    "sanity_list": file_sha256(fx.sanity_list),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = fx.run("--dry_run_stratification", out_dir=root / "degenerate_out")
    check(result.returncode == 2, f"the dry run stops (got {result.returncode})")
    check(
        "not strictly increasing" in result.stderr,
        "and says the groups do not differ in the quantity",
    )
    check(
        "does not fall back" in result.stderr,
        "and states that it does not fall back to another stratification",
    )
    out = root / "degenerate_out"
    check(
        not (out / "stratification_feasibility_SHARE.json").exists(),
        "no shared output is written",
    )
    check(
        not (out / "sealed" / "stratification_DO_NOT_SHARE.json").exists(),
        "no intermediate is written either: the condition is checked before anything is saved",
    )
    check(
        not contains_video_identity(result.stdout + result.stderr),
        "the refusal carries no video ID",
    )


def main() -> None:
    print("Stage5 S5-16 Step 0: B' split builder end-to-end synthetic checks")
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(Path(tmp))
        fingerprint = test_dry_run_releases_only_the_allowed_aggregate(fx)
        if fingerprint:
            test_confirm_requires_the_fingerprint(fx, fingerprint)
            test_settings_mismatch_is_refused(fx, fingerprint)
            test_confirm_writes_a_valid_split(fx, fingerprint)
            test_pinning_closes_the_loop(fx)
            test_initial_build_will_not_run_twice(fx, fingerprint)
        else:
            check(False, "no fingerprint was printed; the later stages could not run")
        test_raw_value_validation(Path(tmp) / "validation")
        test_degenerate_quantity_stops(Path(tmp) / "degenerate")
    print(f"\nchecks run: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
