from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.geometry import metrics as mt  # noqa: E402
from stage5.utils.file_list_mode import list_identity_sha256  # noqa: E402
from stage5.utils.split_identity import contains_video_identity, file_sha256  # noqa: E402

# ----------------------------------------------------------------------------
# S5-17 S17-6: synthetic coverage of supplement_stage5_geometry_aggregates.
#
# Management decisions S6-5 and 14.13. A synthetic contract, a synthetic
# private record with known values and a consistent synthetic shared run.json.
#
# What the checks establish, kept apart (report 14.13.2):
#   * BEFORE parsing: registration requires the operator-attested FULL hash to
#     equal the file (plus the supporting metadata checks), and aggregation
#     parses only after the coverage hash, seal and allow-list checks.
#   * AFTER parsing: swapped aliases, an extra sealed alias, a different path,
#     an input-unevaluable record, a missing group and values that do not
#     reproduce the shared run are DETECTED and stop the run. The altered
#     records in these cases register successfully on purpose -- registration
#     does not parse -- so these cases test detection after parsing, not
#     refusal before it. Passing them does not prove the registered set.
#   * Output: a changed input stops the run before the shared result is
#     committed, so no new shared result is published.
# numpy only; no H5, no real data, no production pins.
# ----------------------------------------------------------------------------

TOOL = REPO_ROOT / "supplement_stage5_geometry_aggregates.py"
SUFFIX = "_pointcloud_annotated_foreground_combined_v2_bboxrank_v7.h5"
TRAIN_IDS = ["20250101_100000_1", "20250101_100000_1_02", "1-2_34_56", "20250102_110000_3"]
FALLBACK_ID = "20250102_110000_3"
SANITY_IDS = ["20250101_100000_1", "1-2_34_56", "20250101_100000_1_02"]
VAL_IDS = ["20250201_090000_7", "3-4_56"]
SEALED_IDS = ["20250301_080000_9"]
RESOURCE_TEXT = (
    '\tCommand being timed: "env MODE=run COORDINATE_MODE=pixel SHARED_JSON=/x/shared/run.json '
    'bash evaluate_stage5_geometry.sh"\n'
    "\tElapsed (wall clock) time (h:mm:ss or m:ss): 2:53.03\n"
    "\tMaximum resident set size (kbytes): 4584052\n"
    "\tExit status: 0\n"
)
REPS = ("a_aabb", "b_obb", "c_axis", "d_bestframe")

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def pair(center, endpoint_mean, endpoint_max, angle, length):
    return {"frame_order": 0, "matching_centroid_distance": center, "center_error": center,
            "endpoint_error_mean": endpoint_mean, "endpoint_error_max": endpoint_max,
            "axis_angle_error_deg": angle, "length_relative_error": length}


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        teacher = {i: root / "teacher" / f"{i}{SUFFIX}" for i in TRAIN_IDS + VAL_IDS}
        self.teacher = teacher
        self.train_list = self._list("train_core.txt", [teacher[i] for i in TRAIN_IDS])
        self.val_list = self._list("validation.txt", [teacher[i] for i in VAL_IDS])
        self.sanity_list = self._list("sanity.txt", [teacher[i] for i in SANITY_IDS])
        self.pins = self._contract()
        self.alias = {i: f"train_core_{k:03d}" for k, i in enumerate(TRAIN_IDS)}
        self.alias.update({i: f"validation_{k:02d}" for k, i in enumerate(VAL_IDS)})
        self.private_dir = root / "private"
        self.private_dir.mkdir(mode=0o700)
        self.private = self.private_dir / "geometry_run_DO_NOT_SHARE.json"
        self.shared_run = self.private_dir / "shared" / "run.json"
        self.resource = self.private_dir / "run_resource_usage.txt"
        self.coverage = self.private_dir / "artifact_coverage_DO_NOT_SHARE.json"
        self.record = self._record()
        self.write_all(self.record, self.shared_from(self.record))

    def _list(self, name: str, paths: list[Path]) -> Path:
        path = self.root / "lists" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
        return path

    def _contract(self) -> Path:
        directory = self.root / "contract"
        directory.mkdir()
        registry = directory / "registry.json"
        registry.write_text(json.dumps({"schema": "stage5_seal_registry_v1", "sealed_video_identities": SEALED_IDS,
                                        "sealed_list_identity_sha256": "sealed-list-hash"}), encoding="utf-8")
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "stage5_split_contract_v1", "name": "synthetic",
            "lists": {"train_core": {"identity_sha256": list_identity_sha256([self.teacher[i] for i in TRAIN_IDS])},
                      "validation": {"identity_sha256": list_identity_sha256([self.teacher[i] for i in VAL_IDS])},
                      "internal_test": {"identity_sha256": "sealed-list-hash"}},
            "sealed_splits": ["internal_test"], "seal_registry_required": True,
            "input_sha256": {"sanity_list": file_sha256(self.sanity_list)},
        }), encoding="utf-8")
        pins = directory / "pins.json"
        pins.write_text(json.dumps({"schema": "stage5_split_contract_pins_v1", "pins": {
            "seal_registry": {"path": str(registry), "expected_sha256": file_sha256(registry)},
            "synthetic": {"path": str(manifest), "expected_sha256": file_sha256(manifest)},
        }}), encoding="utf-8")
        self.manifest_lists = json.loads(manifest.read_text())["lists"]
        return pins

    def _record(self) -> dict:
        a = self.alias
        core = {
            a[TRAIN_IDS[0]]: {"holdout": {"value": dict(zip(REPS, (0.99, 0.90, 0.90, 0.91)))},
                              "pseudo3d_reference": {"value": 10.0, "failure": None},
                              "bbox_consistency": {"n_frames": 4, "iou": {"median": 0.8}, "long_side_rel_diff": {"median": 0.1}},
                              "h17_5_gt": {"multi_instance_frames": 3, "m1_m2_length_diff_over_10pct": 1}},
            a[TRAIN_IDS[1]]: {"holdout": {"value": dict(zip(REPS, (0.97, 0.88, 0.88, None)))},
                              "pseudo3d_reference": {"value": 30.0, "failure": None},
                              "bbox_consistency": {"n_frames": 0},
                              "h17_5_gt": {"multi_instance_frames": 0, "m1_m2_length_diff_over_10pct": 0}},
            a[TRAIN_IDS[2]]: {"holdout": {"value": dict(zip(REPS, (1.0, 0.92, 0.92, 0.93)))},
                              "pseudo3d_reference": {"value": None, "failure": "axis_ambiguous"},
                              "bbox_consistency": {"n_frames": 6, "iou": {"median": 0.6}, "long_side_rel_diff": {"median": -0.2}},
                              "h17_5_gt": {"multi_instance_frames": 5, "m1_m2_length_diff_over_10pct": 2}},
        }

        def video(n_fp, within, pairs):
            return {"postprocess": {"P0": {"fp_classes": {"n_fp": n_fp},
                                           "representations": {"c_axis": {"within_tau_rel_diagnostic": within}},
                                           "matched_pairs": pairs}}}

        val_a, val_b = a[VAL_IDS[0]], a[VAL_IDS[1]]
        sanity = [a[i] for i in SANITY_IDS]
        predictions = {
            "validation/best": {val_a: video(5, True, [pair(2.0, 3.0, 4.0, 5.0, -0.1), pair(4.0, 5.0, 6.0, 7.0, None)]),
                                val_b: video(7, False, [])},
            "validation/last": {val_a: video(1, False, []), val_b: video(2, True, [pair(10.0, 11.0, 12.0, 13.0, 0.3)])},
            "sanity_in_sample/best": {s: video(4, True, [pair(1.0, 1.0, 1.0, 1.0, 0.0)]) for s in sanity},
            "sanity_in_sample/last": {s: video(3, False, []) for s in sanity},
        }
        return {
            "alias_map": {a[i]: {"identity": i, "path": str(self.teacher[i])} for i in TRAIN_IDS + VAL_IDS},
            "train_core_records": core,
            "prediction_records": predictions,
            "inputs": {a[i]: {"resolution": "recorded_source_attr", "mode": "resize_shorter_then_offset_crop",
                              "input_failure": "crop_fallback_resize" if i == FALLBACK_ID else None}
                       for i in TRAIN_IDS},
            "git": {"head": "synthetic", "dirty": False},
        }

    def shared_from(self, record: dict) -> dict:
        def table(group: str) -> dict:
            out = {"input": {"evaluable": len(record["prediction_records"][f"{group}/best"])}, "checkpoints": {}}
            for checkpoint in ("best", "last"):
                videos = record["prediction_records"][f"{group}/{checkpoint}"].values()
                out["checkpoints"][checkpoint] = {"postprocess": {"P0": {
                    "fp_classes_pooled": {"n_fp": sum(v["postprocess"]["P0"]["fp_classes"]["n_fp"] for v in videos)},
                    "representations": {"c_axis": {"within_tau_rel_diagnostic": {
                        "count": sum(int(v["postprocess"]["P0"]["representations"]["c_axis"]["within_tau_rel_diagnostic"])
                                     for v in videos)}}},
                }}}
            return out

        per_rep = {rep: {"holdout_coverage_descriptive": mt.summarize(
            [r["holdout"]["value"][rep] for r in record["train_core_records"].values()])} for rep in REPS}
        return {
            "schema": "stage5_s5_17_geometry_summary_v1", "coordinate_mode": "pixel", "git": {"dirty": False},
            "input_minimums": {"synthetic_override": False, "train_core": 130, "validation": 16},
            "contract": {"manifest": "synthetic",
                         "train_core_identity16": self.manifest_lists["train_core"]["identity_sha256"][:16],
                         "validation_identity16": self.manifest_lists["validation"]["identity_sha256"][:16]},
            "h17_1": {"per_representation": per_rep},
            "tables": {"validation": table("validation"), "sanity_in_sample": table("sanity_in_sample")},
        }

    def write_all(self, record: dict, shared: dict, *, resource: str = RESOURCE_TEXT) -> None:
        self.shared_run.parent.mkdir(parents=True, exist_ok=True)
        self.private.write_text(json.dumps(record), encoding="utf-8")
        os.chmod(self.private, 0o600)
        self.shared_run.write_text(json.dumps(shared), encoding="utf-8")
        self.resource.write_text(resource, encoding="utf-8")
        self.touch()

    def touch(self, private_offset: float = 0.0) -> None:
        base = 1_900_000_000.0
        os.utime(self.private, (base + private_offset, base + private_offset))
        os.utime(self.shared_run, (base + 1, base + 1))
        os.utime(self.resource, (base + 2, base + 2))

    def common(self) -> list[str]:
        return ["--split_manifest", "synthetic", "--split_contract_pins", str(self.pins),
                "--train_core_list", str(self.train_list), "--validation_list", str(self.val_list),
                "--train_sanity_list", str(self.sanity_list), "--private_run_json", str(self.private),
                "--shared_run_json", str(self.shared_run),
                "--expected_shared_run_sha256", file_sha256(self.shared_run)]

    def register(self, *extra: str, attested: str | None = None) -> subprocess.CompletedProcess:
        return run("register", *self.common(), "--resource_usage", str(self.resource),
                   "--coverage_out", str(self.coverage),
                   "--attested_private_sha256", attested or file_sha256(self.private),
                   "--attestation_reference", "synthetic attestation", *extra)

    def aggregate(self, out: Path, *extra: str) -> subprocess.CompletedProcess:
        return run("aggregate", *self.common(), "--artifact_coverage", str(self.coverage),
                   "--shared_json", str(out), *extra)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def stops(world: World, result: subprocess.CompletedProcess, label: str, expect: str) -> None:
    check(result.returncode == 2, f"{label}: stops with exit 2 (got {result.returncode})")
    check(expect in result.stderr, f"{label}: names the reason ({expect!r})")
    text = result.stdout + result.stderr
    check(not contains_video_identity(text) and str(world.root) not in text and "Traceback" not in text,
          f"{label}: no identity, path or traceback on the console")


# ----------------------------------------------------------------------------


def test_register(world: World) -> None:
    print("[1] register: provenance from metadata only, hash-bound coverage")
    shared = world.private_dir / "shared" / "registration.json"
    result = world.register("--shared_json", str(shared))
    check(result.returncode == 0, f"registration succeeds (stderr: {result.stderr[-200:]})")
    payload = json.loads(world.coverage.read_text(encoding="utf-8"))
    entry = payload["entries"][-1]
    check(entry["artifact_name"] == "geometry_run_DO_NOT_SHARE.json" and entry["sha256"] == file_sha256(world.private),
          "the entry is bound to the private record's hash")
    check(sorted(entry["video_identities"]) == sorted(TRAIN_IDS + VAL_IDS), "covers train_core + validation")
    check(entry["provenance"]["contents_parsed_before_registration"] is False, "contents were not parsed")
    check(entry["provenance"]["operator_attestation"]["reference"] == "synthetic attestation"
          and "not a generation-time hash" in entry["provenance"]["link_basis"],
          "the operator attestation and its nature are recorded")
    check(stat.S_IMODE(world.coverage.stat().st_mode) == 0o600, "coverage record is 0600")
    check(not contains_video_identity(shared.read_text(encoding="utf-8")), "registration summary carries no identity")


def test_register_refusals(world: World) -> None:
    print("[2] register refusals: shared hash, 0600, run window, exit status, clean run")
    args = world.common()
    args[args.index("--expected_shared_run_sha256") + 1] = "0" * 64
    stops(world, run("register", *args, "--resource_usage", str(world.resource), "--coverage_out",
                     str(world.root / "x.json"), "--attested_private_sha256", file_sha256(world.private),
                     "--attestation_reference", "synthetic attestation"),
          "a shared run JSON with another hash", "SHA-256 recorded in the report")

    stops(world, world.register(attested="0" * 64), "an attested hash that differs from the file", "operator-attested")
    stops(world, world.register(attested=file_sha256(world.private)[:16]), "a short attested hash", "full 64-hex")

    os.chmod(world.private, 0o644)
    stops(world, world.register(), "a private record that is not 0600", "0600")
    os.chmod(world.private, 0o600)

    world.touch(private_offset=10.0)
    stops(world, world.register(), "a private record modified after the run", "outside the run window")
    world.touch(private_offset=-400.0)
    stops(world, world.register(), "a private record older than the run", "outside the run window")
    world.touch()

    world.resource.write_text(RESOURCE_TEXT.replace("Exit status: 0", "Exit status: 1"), encoding="utf-8")
    world.touch()
    stops(world, world.register(), "a failed run", "not a successful")
    world.resource.write_text(RESOURCE_TEXT, encoding="utf-8")
    world.touch()

    dirty = world.shared_from(world.record)
    dirty["git"]["dirty"] = True
    original = world.shared_run.read_bytes()
    world.shared_run.write_text(json.dumps(dirty), encoding="utf-8")
    world.touch()
    stops(world, world.register(), "a run from a dirty tree", "clean pixel-mode run")
    world.shared_run.write_bytes(original)
    world.touch()


def test_aggregate(world: World) -> None:
    print("[3] aggregate: known values, denominators, undefined counts, separation, boundary")
    before = (file_sha256(world.private), file_sha256(world.shared_run))
    out = world.private_dir / "shared" / "supplement.json"
    result = world.aggregate(out)
    check(result.returncode == 0, f"aggregation succeeds (stderr: {result.stderr[-300:]})")
    if result.returncode != 0:
        return
    text = out.read_text(encoding="utf-8")
    payload = json.loads(text)
    check(not contains_video_identity(text) and str(world.root) not in text
          and "Infinity" not in text and "NaN" not in text, "shared supplement: no identity, path or non-finite value")
    check((file_sha256(world.private), file_sha256(world.shared_run)) == before, "inputs are left unchanged")
    check(payload["verification"]["recomputed_shared_aggregates"]
          == {"h17_1_holdout": 4, "fp_totals": 4, "within_counts": 4}, "shared aggregates were recomputed and matched")

    e = payload["e_pseudo3d_reference"]
    check(e["value_pseudo3d_units"]["median"] == 20.0 and e["value_pseudo3d_units"]["n_undefined"] == 1
          and e["failures_by_reason"] == {"axis_ambiguous": 1} and e["n_videos"] == 3, "(e): median 20, one failure")
    b = payload["h17_1b_bbox_consistency"]
    check(b["n_videos_compared"] == 2 and b["n_videos_without_compared_frames"] == 1 and b["n_frames_compared_total"] == 10
          and abs(b["iou_video_median"]["median"] - 0.7) < 1e-12, "H17-1b: 2 videos, 10 frames, median of medians 0.7")
    m = payload["h17_5_multiple_instances"]
    check((m["multi_instance_frames"], m["m1_m2_length_diff_over_10pct_frames"], m["n_videos_with_multi_instance_frames"])
          == (8, 3, 2) and m["share_of_multi_instance_frames"] == 0.375, "H17-5: 3 of 8 multi-instance frames")
    pairs = payload["h17_2_matched_pairs"]
    vb, vl = pairs["validation"]["best"]["P0"], pairs["validation"]["last"]["P0"]
    check(vb["n_pairs"] == 2 and vb["n_videos_with_pairs"] == 1 and vb["center_error"]["median"] == 3.0,
          "validation/best: 2 pairs from 1 video, centre error median 3")
    check(vb["length_relative_error"]["n_defined"] == 1 and vb["length_relative_error"]["n_undefined"] == 1
          and vb["abs_length_relative_error"]["median"] == 0.1, "an undefined length error is counted, not zero-filled")
    check(vl["n_pairs"] == 1 and vl["center_error"]["median"] == 10.0, "best and last are kept apart")
    check(pairs["sanity_in_sample"]["best"]["P0"]["n_pairs"] == 3 and pairs["sanity_in_sample"]["last"]["P0"]["n_pairs"] == 0
          and pairs["sanity_in_sample"]["last"]["P0"]["center_error"]["median"] is None,
          "sanity is a separate group; no pairs gives undefined, not 0")
    check("not overall performance" in pairs["note"] and payload["inputs"]["shared_run_sha256"] == before[1],
          "the conditional nature and the input hash are recorded")

    stops(world, world.aggregate(world.shared_run), "an output path equal to an input", "must not overwrite")


def _re_register(world: World, record: dict, shared: dict | None = None) -> None:
    world.write_all(record, shared if shared is not None else world.shared_from(world.record))
    result = world.register()
    check(result.returncode == 0, "the altered record registers (registration does not parse contents)")


def test_aggregate_refusals(world: World) -> None:
    print("[4] aggregate refusals: coverage, hash, alias/identity, split, recomputed totals")
    out = world.root / "refused.json"
    args = world.common()
    stops(world, run("aggregate", *args, "--artifact_coverage", str(world.root / "absent.json"), "--shared_json", str(out)),
          "no coverage record", "coverage could not be read")

    original = world.private.read_bytes()
    world.private.write_bytes(original + b" ")
    stops(world, world.aggregate(out), "the private record changed after registration", "could not be tied to a video")
    world.private.write_bytes(original)
    world.touch()

    cases = []
    swapped = copy.deepcopy(world.record)
    a0, a1 = world.alias[TRAIN_IDS[0]], world.alias[TRAIN_IDS[1]]
    swapped["alias_map"][a0]["identity"], swapped["alias_map"][a1]["identity"] = TRAIN_IDS[1], TRAIN_IDS[0]
    cases.append(("aliases swapped between two videos", swapped, None, "maps to a different video"))
    moved = copy.deepcopy(world.record)
    moved["alias_map"][a0]["path"] = str(world.root / "elsewhere" / Path(moved["alias_map"][a0]["path"]).name)
    cases.append(("the same video under a different path", moved, None, "different video or path"))
    extra = copy.deepcopy(world.record)
    extra["alias_map"]["train_core_999"] = {"identity": SEALED_IDS[0], "path": "x"}
    cases.append(("an extra alias for a sealed video", extra, None, "not exactly this contract"))
    fallback = copy.deepcopy(world.record)
    fallback["train_core_records"][world.alias[FALLBACK_ID]] = fallback["train_core_records"][a0]
    cases.append(("a record for an input-unevaluable video", fallback, None, "input-evaluable"))
    missing = copy.deepcopy(world.record)
    del missing["prediction_records"]["sanity_in_sample/last"]
    cases.append(("a missing prediction group", missing, None, "prediction groups"))
    fp = copy.deepcopy(world.record)
    fp["prediction_records"]["validation/best"][world.alias[VAL_IDS[0]]]["postprocess"]["P0"]["fp_classes"]["n_fp"] = 99
    cases.append(("values that do not reproduce the shared run", fp, world.shared_from(world.record), "FP totals"))
    for label, record, shared, expect in cases:
        _re_register(world, record, shared)
        stops(world, world.aggregate(out), label, expect)
    check(not out.exists(), "no supplement is written by a refused aggregation")
    world.write_all(world.record, world.shared_from(world.record))


def test_change_during_aggregation(world: World) -> None:
    print("[5] an input changed during aggregation: no new shared result is published")
    import argparse
    import supplement_stage5_geometry_aggregates as sup

    out = world.private_dir / "shared" / "supplement_midway.json"
    args = argparse.Namespace(
        split_manifest="synthetic", split_contract_pins=str(world.pins), train_core_list=str(world.train_list),
        validation_list=str(world.val_list), train_sanity_list=str(world.sanity_list),
        private_run_json=str(world.private), shared_run_json=str(world.shared_run),
        expected_shared_run_sha256=file_sha256(world.shared_run), artifact_coverage=str(world.coverage),
        shared_json=str(out),
    )
    original_bytes = world.private.read_bytes()
    original = sup._matched_pairs

    def tamper(record, shared):
        world.private.write_bytes(original_bytes + b" ")     # the input changes after it was verified
        return original(record, shared)

    sup._matched_pairs = tamper
    try:
        raised = False
        try:
            sup.cmd_aggregate(args)
        except sup.InputContractError as error:
            raised = "no new shared result" in str(error)
        check(raised, "the change is detected before the shared output is committed")
    finally:
        sup._matched_pairs = original
        world.private.write_bytes(original_bytes)
        world.touch()
    check(not out.exists(), "no new shared result is published")
    check(not out.with_name(out.name + ".partial").exists(), "the temporary output is removed")


def main() -> int:
    print("Stage5 S5-17 geometry supplement synthetic test")
    with tempfile.TemporaryDirectory(prefix="s5_17_supplement_") as tmp:
        world = World(Path(tmp))
        test_register(world)
        test_register_refusals(world)
        test_aggregate(world)
        test_aggregate_refusals(world)
        test_change_during_aggregation(world)
    print(f"checks: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        return 1
    print("All geometry supplement checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
