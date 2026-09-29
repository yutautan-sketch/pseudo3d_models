from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
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

import evaluate_stage5_geometry as cli  # noqa: E402
from stage5.utils import geometry_inputs as gi  # noqa: E402
from stage5.utils.file_list_mode import list_content_sha256, list_identity_sha256  # noqa: E402
from stage5.utils.split_contract import SplitContractError  # noqa: E402
from stage5.utils.split_identity import contains_video_identity, file_sha256  # noqa: E402

# ----------------------------------------------------------------------------
# S5-17 S17-3: synthetic coverage of the geometry CLI (evaluate_stage5_geometry).
#
# Management document 7: synthetic contract, synthetic H5/NPZ, no production
# pins, no real data. Success paths run to completion and write their JSON;
# refusals are checked to happen BEFORE a read (a sealed video is given a path
# that does not exist, so a guard that fires late would report "missing"
# instead). Identities cover the timestamp form with and without a fourth
# segment and the case form; `..._1` and `..._1_02` coexist as a prefix trap.
#
# numpy/h5py, git (for the launcher evidence). No torch, no CUDA.
# ----------------------------------------------------------------------------

TOOL = REPO_ROOT / "evaluate_stage5_geometry.py"
TEACHER_SUFFIX = "_pointcloud_annotated_foreground_combined_v2_bboxrank_v7.h5"
INTER_SUFFIX = "_pseudo3d.h5"
TRAIN_IDS = ["20250101_100000_1", "20250101_100000_1_02", "1-2_34_56",
             "20250102_110000_3", "20250103_120000_4", "20250104_130000_5"]
FALLBACK_ID = "20250104_130000_5"
SANITY_IDS = ["20250101_100000_1", "1-2_34_56", "20250102_110000_3"]
VAL_IDS = ["20250201_090000_7", "20250201_090000_7_03", "3-4_56"]
SEALED_IDS = ["20250301_080000_9", "5-6_78"]
N_FRAMES = 13

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def rotated_grid(length, width, angle_deg, center, nt=31, ns=5) -> np.ndarray:
    t, s = np.meshgrid(np.linspace(-length / 2, length / 2, nt), np.linspace(-width / 2, width / 2, ns))
    a = math.radians(angle_deg)
    u, p = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    return np.asarray(center) + t.reshape(-1, 1) * u + s.reshape(-1, 1) * p


# ----------------------------------------------------------------------------
# synthetic world
# ----------------------------------------------------------------------------


def teacher_arrays() -> dict[str, np.ndarray]:
    xy, frames, labels, bbox_frames, bboxes = [], [], [], [], []
    background = np.stack(np.meshgrid(np.arange(8.0, 256.0, 16.0), np.arange(8.0, 256.0, 16.0)), -1).reshape(-1, 2)
    for f in range(N_FRAMES):
        xy.append(background)
        frames += [f] * len(background)
        labels += [0] * len(background)
        if f < N_FRAMES - 1:                                   # the last frame has no GT
            gt = rotated_grid(60.0 + 2 * f, 6.0, 20.0, (128.0 + 0.5 * f, 128.0))
            ignore = gt[:10] + np.array([0.0, 12.0])            # a 10-point chain 12 px away
            xy += [gt, ignore]
            frames += [f] * (len(gt) + len(ignore))
            labels += [1] * len(gt) + [-1] * len(ignore)
            bbox_frames.append(f)
            bboxes.append([gt[:, 0].min(), gt[:, 1].min(), gt[:, 0].max(), gt[:, 1].max()])
    pixel_xy = np.concatenate(xy)
    frame_order = np.asarray(frames)
    label = np.asarray(labels, dtype=np.int8)
    return {
        "pixel_xy": pixel_xy,
        "frame_order": frame_order,
        "points": np.column_stack([pixel_xy, frame_order * 2.0]),
        "point_label": label,
        "valid_mask": label != -1,
        "bbox_frame_order": np.asarray(bbox_frames),
        "bbox_local_xyxy": np.asarray(bboxes),
    }


def write_teacher(path: Path, arrays: dict[str, np.ndarray], identity: str, source: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        pc = f.create_group("point_cloud")
        pc["pixel_xy"] = arrays["pixel_xy"].astype(np.float32)
        pc["frame_order"] = arrays["frame_order"].astype(np.int32)
        pc["points"] = arrays["points"].astype(np.float32)
        an = f.create_group("annotation")
        an["point_label"] = arrays["point_label"]
        an["valid_mask"] = arrays["valid_mask"]
        fa = f.create_group("frame_annotation")
        fa["frame_order"] = arrays["bbox_frame_order"].astype(np.int32)
        fa["bbox_local_xyxy"] = arrays["bbox_local_xyxy"].astype(np.float32)
        f.attrs["video_name"] = identity
        f.attrs["source_pseudo3d_h5"] = str(source)
    return path


def write_intermediate(path: Path, *, mode: str = "resize_shorter_then_offset_crop") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["local_input_shape"] = f"({N_FRAMES}, 1, 256, 256)"
        f.attrs["num_frames"] = np.int64(N_FRAMES)
        f.attrs["raw_width"] = np.int64(640)
        f.attrs["raw_height"] = np.int64(480)
        f.attrs["local_preprocess_effective"] = mode
        f.attrs["local_resize_scale"] = float("nan") if mode == "offset_crop_fallback_resize" else 448.0 / 480.0
        f.attrs["local_crop_left"] = np.int64(170)
        f.attrs["local_crop_top"] = np.int64(0)
    return path


def prediction(arrays: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    label = arrays["point_label"]
    prob = np.where(label == 1, 0.8, np.where(label == -1, 0.7, 0.1)) + rng.uniform(-0.05, 0.05, label.shape)
    frames = arrays["frame_order"]
    fp = np.flatnonzero((frames == 5) & (label == 0))[:3]      # three isolated FP points
    prob[fp] = 0.9
    pred = prob > 0.5
    return {
        "point_indices": np.arange(label.size, dtype=np.int64),
        "prob_femur": prob.astype(np.float32),
        "pred_label": pred.astype(np.uint8),
        "vote_count": gi.expected_vote_count(frames, window_size=16, stride=8, include_tail=True).astype(np.int32),
    }


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.teacher: dict[str, Path] = {}
        self.arrays = teacher_arrays()
        for identity in TRAIN_IDS + VAL_IDS:
            inter = write_intermediate(
                root / "inter" / f"{identity}{INTER_SUFFIX}",
                mode="offset_crop_fallback_resize" if identity == FALLBACK_ID else "resize_shorter_then_offset_crop",
            )
            self.teacher[identity] = write_teacher(root / "teacher" / f"{identity}{TEACHER_SUFFIX}", self.arrays,
                                                   identity, inter)
        self.train_list = self._list("train_core.txt", [self.teacher[i] for i in TRAIN_IDS])
        self.val_list = self._list("validation.txt", [self.teacher[i] for i in VAL_IDS])
        self.sanity_list = self._list("sanity.txt", [self.teacher[i] for i in SANITY_IDS])
        self.pins = self.contract("pins.json", SEALED_IDS, [self.teacher[i] for i in VAL_IDS])
        self.eval_root = root / "evaluation"
        for name, epoch in (("best", 6), ("last", 50)):
            self._evaluation(name, epoch)
        self.launcher = self._launcher()
        self.private = root / "private"
        self.shared = root / "shared"
        self.coverage = self.private / "coverage.json"

    def _list(self, name: str, paths: list[Path]) -> Path:
        path = self.root / "lists" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
        return path

    def contract(self, name: str, sealed: list[str], validation_paths: list[Path]) -> Path:
        directory = self.root / "contract" / name.replace(".json", "")
        directory.mkdir(parents=True, exist_ok=True)
        registry = directory / "registry.json"
        registry.write_text(json.dumps({"schema": "stage5_seal_registry_v1", "sealed_video_identities": sealed,
                                        "sealed_list_identity_sha256": "sealed-list-hash",
                                        "sealed_until": "S5-20c"}), encoding="utf-8")
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "stage5_split_contract_v1", "name": "synthetic",
            "lists": {
                "train_core": {"identity_sha256": list_identity_sha256([self.teacher[i] for i in TRAIN_IDS])},
                "validation": {"identity_sha256": list_identity_sha256(validation_paths)},
                "internal_test": {"identity_sha256": "sealed-list-hash"},
            },
            "sealed_splits": ["internal_test"], "seal_registry_required": True,
            "allows_new_training": False, "allows_directory_mode": False,
            "input_sha256": {"sanity_list": file_sha256(self.sanity_list)},   # Step 0 input record
        }), encoding="utf-8")
        pins = directory / name
        pins.write_text(json.dumps({"schema": "stage5_split_contract_pins_v1", "pins": {
            "seal_registry": {"path": str(registry), "expected_sha256": file_sha256(registry)},
            "synthetic": {"path": str(manifest), "expected_sha256": file_sha256(manifest)},
        }}), encoding="utf-8")
        return pins

    def _evaluation(self, name: str, epoch: int) -> None:
        directory = self.eval_root / name
        checkpoint = self.eval_root / "checkpoints" / f"{name}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"synthetic checkpoint {name}".encode())
        rows = []
        for split, ids in (("train_sanity", SANITY_IDS), ("validation", VAL_IDS)):
            for k, identity in enumerate(ids):
                pred = prediction(self.arrays, seed=epoch * 100 + k)
                target = directory / "predictions" / split / f"{identity}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **pred)
                counts = gi.confusion_counts(self.arrays["point_label"].astype(np.int64), self.arrays["valid_mask"],
                                             pred["pred_label"].astype(bool))
                rows.append({"checkpoint": f"{name}.pt", "split": split, "video_name": identity,
                             "h5_path": str(self.teacher[identity]), **counts})
        (directory / "summary.json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "checkpoint_epoch": epoch,
            "selected_train_files": [str(self.teacher[i]) for i in SANITY_IDS],
            "validation_files": [str(self.teacher[i]) for i in VAL_IDS],
            "window_size_frames": 16, "window_stride_frames": 8, "include_tail_window": True,
        }), encoding="utf-8")
        with (directory / "h5_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _launcher(self) -> Path:
        repo = self.root / "launcher_repo"
        repo.mkdir()
        launcher = repo / "run_stage5_s5_15_arm.sh"
        paths = [self.teacher[i] for i in VAL_IDS]
        launcher.write_text(f'#!/usr/bin/env bash\nEXPECTED_VAL_LIST_SHA256="{list_content_sha256(paths)}"\n',
                            encoding="utf-8")
        for args in (["init", "-q"], ["add", launcher.name],
                     ["-c", "user.email=synthetic@example.invalid", "-c", "user.name=synthetic",
                      "commit", "-qm", "synthetic launcher"]):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        return launcher

    def common(self, *, pins: Path | None = None, validation: Path | None = None, sanity: Path | None = None
               ) -> list[str]:
        return ["--split_manifest", "synthetic", "--split_contract_pins", str(pins or self.pins),
                "--train_core_list", str(self.train_list), "--validation_list", str(validation or self.val_list),
                "--train_sanity_list", str(sanity or self.sanity_list)]

    def inputs(self) -> list[str]:
        return ["--intermediate_root", str(self.root / "inter"), "--intermediate_suffix", INTER_SUFFIX,
                "--evaluation_dir_best", str(self.eval_root / "best"),
                "--evaluation_dir_last", str(self.eval_root / "last")]


def cli_run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def clean_output(world: World, result: subprocess.CompletedProcess, label: str) -> None:
    text = result.stdout + result.stderr
    check(not contains_video_identity(text), f"{label}: stdout/stderr carry no video identity")
    check(str(world.root) not in text, f"{label}: stdout/stderr carry no absolute path")
    check("Traceback" not in text, f"{label}: no traceback on the console")


def stops(world: World, result: subprocess.CompletedProcess, label: str, *, expect: str | None = None) -> None:
    check(result.returncode == 2, f"{label}: stops with exit 2 (got {result.returncode})")
    if expect is not None:
        check(expect in result.stderr, f"{label}: the stop names the reason ({expect!r})")
    clean_output(world, result, label)


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------


def test_raw_readers(root: Path) -> None:
    print("[1] raw-array readers: integrality, int64 range, domains, band, exact intermediate resolution")
    for values, label in ((np.array([1.0, 2.5]), "non-integral"), (np.array([1.0, np.nan]), "non-finite"),
                          (np.array([1e19]), "beyond int64"), (np.array([2 ** 63], dtype=np.uint64), "uint64 overflow")):
        try:
            gi.integral_array(values, name="x")
            check(False, f"integral_array refuses {label}")
        except gi.InputContractError:
            check(True, f"integral_array refuses {label}")
    check(gi.integral_array(np.array([3.0, 4.0]), name="x").dtype == np.int64, "integral floats become int64")

    arrays = teacher_arrays()
    good = write_teacher(root / "readers" / f"20250901_000000_1{TEACHER_SUFFIX}", arrays, "20250901_000000_1", root)
    read = gi.read_teacher_arrays(good)
    check(read.pixel_xy.dtype == np.float64 and read.n_points == arrays["frame_order"].size, "teacher arrays read")
    bad_label = dict(arrays, point_label=np.where(arrays["point_label"] == 0, 2, arrays["point_label"]).astype(np.int8))
    bad = write_teacher(root / "readers" / f"20250901_000000_2{TEACHER_SUFFIX}", bad_label, "20250901_000000_2", root)
    check(_raises(gi.InputContractError, gi.read_teacher_arrays, bad), "point_label outside {-1,0,1} stops")
    inconsistent = dict(arrays, valid_mask=np.ones_like(arrays["valid_mask"]))
    bad = write_teacher(root / "readers" / f"20250901_000000_3{TEACHER_SUFFIX}", inconsistent, "20250901_000000_3", root)
    check(_raises(gi.InputContractError, gi.read_teacher_arrays, bad), "valid_mask != (label != -1) stops")

    pred = np.array([False, True, False, True])
    prob = np.array([0.5 + 5e-7, 0.5 - 5e-7, 0.1, 0.9])
    check(gi.check_label_probability(pred, prob) == 2, "labels inside the +-1e-6 band are accepted and counted")
    check(_raises(gi.InputContractError, gi.check_label_probability, np.array([False]), np.array([0.6])),
          "a label contradicting p1 outside the band stops")

    inter = root / "readers" / "inter"
    inter.mkdir(parents=True)
    write_intermediate(inter / f"20250901_000000_1_02{INTER_SUFFIX}")        # only the longer identity exists
    check(_raises(gi.InputContractError, gi.resolve_intermediate_exact, teacher_identity="20250901_000000_1",
                  video_name="20250901_000000_1", recorded_source=None, intermediate_root=inter, suffix=INTER_SUFFIX),
          "..._1 does not resolve to ..._1_02 (no prefix match)")
    other = write_intermediate(inter / f"20250901_000000_9{INTER_SUFFIX}")
    check(_raises(gi.InputContractError, gi.resolve_intermediate_exact, teacher_identity="20250901_000000_1",
                  video_name="20250901_000000_1", recorded_source=str(other), intermediate_root=inter,
                  suffix=INTER_SUFFIX), "a recorded intermediate of another identity stops")
    npz = root / "readers" / "p.npz"
    np.savez_compressed(npz, a=np.zeros((5, 2), dtype=np.float32))
    check(gi.npz_member_shapes(npz) == {"a": {"shape": [5, 2], "dtype": "float32"}}, "NPZ headers read without data")


def _raises(exc, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def test_privacy_helpers() -> None:
    print("[2] output boundary helpers")
    for value in ("20250101_100000_1_02", "1-2_34", {"p": "/mnt/data/x.h5"}):
        check(_raises(cli.PrivacyError, cli.assert_shareable, {"x": value}), f"shareable check refuses {value!r}")
    text = cli.scrub("stop at /tmp/a/20250101_100000_1_02.h5 and 1-2_34_56")
    check(not contains_video_identity(text) and "/tmp/" not in text, "scrub masks identities and paths")


def test_guard_in_process(world: World) -> None:
    print("[3] purpose allow-lists in process")
    args = argparse.Namespace(split_manifest="synthetic", split_contract_pins=str(world.pins),
                              train_core_list=str(world.train_list), validation_list=str(world.val_list),
                              train_sanity_list=str(world.sanity_list))
    ctx = cli.load_context(args, coverage_path=None)
    check(_raises(SplitContractError, cli.guard, ctx, [world.teacher[VAL_IDS[0]]], cli.PURPOSE_GT),
          "a validation video is refused for GT/prior building")
    check(_raises(SplitContractError, cli.guard, ctx, [world.teacher["20250103_120000_4"]], cli.PURPOSE_IN_SAMPLE),
          "a non-sanity train_core video is refused for in-sample diagnosis")
    check(_raises(SplitContractError, cli.guard, ctx, [world.teacher[TRAIN_IDS[0]]], cli.PURPOSE_PRED),
          "a train_core video is refused for prediction diagnosis")
    cli.guard(ctx, [world.teacher[VAL_IDS[1]]], cli.PURPOSE_PRED)
    check(True, "a validation video is allowed for prediction diagnosis")


def test_register_coverage(world: World) -> None:
    print("[4] register-coverage: provenance without parsing, then hash-bound records")
    for name in ("best", "last"):
        result = cli_run("register-coverage", *world.common(), "--evaluation_checkpoint_dir",
                         str(world.eval_root / name), "--checkpoint_name", name, "--coverage_out", str(world.coverage),
                         "--shared_json", str(world.shared / f"coverage_{name}.json"),
                         "--s5_15_launcher", str(world.launcher))
        check(result.returncode == 0, f"{name}: registered (stderr: {cli.scrub(result.stderr)[-200:]})")
        clean_output(world, result, f"register {name}")
    payload = json.loads(world.coverage.read_text(encoding="utf-8"))
    check(len(payload["entries"]) == 4, "two artifacts x two checkpoints registered")
    check(all(len(e["video_identities"]) == 6 for e in payload["entries"]), "each covers sanity3 + validation3")
    check(stat.S_IMODE(world.coverage.stat().st_mode) == 0o600, "coverage record is 0600")
    shared = (world.shared / "coverage_best.json").read_text(encoding="utf-8")
    check(not contains_video_identity(shared), "registration shared JSON carries no identity")

    extra = world.eval_root / "best" / "predictions" / "validation" / f"{SEALED_IDS[0]}.npz"
    shutil.copy(world.eval_root / "best" / "predictions" / "validation" / f"{VAL_IDS[0]}.npz", extra)
    result = cli_run("register-coverage", *world.common(), "--evaluation_checkpoint_dir", str(world.eval_root / "best"),
                     "--checkpoint_name", "best", "--coverage_out", str(world.root / "tmp_cov.json"),
                     "--s5_15_launcher", str(world.launcher))
    stops(world, result, "an extra prediction name at registration", expect="not exactly the expected videos")
    extra.unlink()

    world.launcher.write_text(world.launcher.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    result = cli_run("register-coverage", *world.common(), "--evaluation_checkpoint_dir", str(world.eval_root / "best"),
                     "--checkpoint_name", "best", "--coverage_out", str(world.root / "tmp_cov.json"),
                     "--s5_15_launcher", str(world.launcher))
    stops(world, result, "an uncommitted launcher edit", expect="git-tracked and clean")
    subprocess.run(["git", "checkout", "--", world.launcher.name], cwd=world.launcher.parent, check=True)

    result = cli_run("register-coverage", *world.common(), "--evaluation_checkpoint_dir", str(world.eval_root / "best"),
                     "--checkpoint_name", "last", "--coverage_out", str(world.root / "tmp_cov.json"),
                     "--s5_15_launcher", str(world.launcher))
    stops(world, result, "a directory that is not the named checkpoint")


def test_audit(world: World) -> None:
    print("[5] audit: metadata only, coverage and hash states")
    shared_path = world.shared / "audit.json"
    best_sha = file_sha256(world.eval_root / "checkpoints" / "best.pt")
    result = cli_run("audit", *world.common(), *world.inputs(), "--artifact_coverage", str(world.coverage),
                     "--checkpoint_sha256_best", best_sha,
                     "--private_out_dir", str(world.private), "--shared_json", str(shared_path))
    check(result.returncode == 0, f"audit completes (stderr: {cli.scrub(result.stderr)[-200:]})")
    clean_output(world, result, "audit")
    if result.returncode != 0:
        return
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    check(shared["counts"]["input"].get("train_core:crop_fallback_resize") == 1, "fallback resize counted as input-side")
    check(shared["counts"]["resolution"].get("train_core:recorded_source_attr") == 6, "resolution method counted")
    check(shared["npz"].get("best:validation:keys_complete") == 3, "NPZ headers checked")
    check(shared["evaluation_runs"]["best"]["hash_state"]["checkpoint"] == "match"
          and shared["evaluation_runs"]["last"]["hash_state"]["checkpoint"] == "unknown", "hash states match / unknown")
    check(not contains_video_identity(shared_path.read_text(encoding="utf-8")), "audit shared JSON has no identity")

    result = cli_run("audit", *world.common(), *world.inputs(), "--artifact_coverage", str(world.coverage),
                     "--checkpoint_sha256_best", "0" * 64,
                     "--private_out_dir", str(world.private), "--shared_json", str(world.root / "x.json"))
    stops(world, result, "a recorded checkpoint hash that does not match", expect="recorded hash does not match")

    result = cli_run("audit", *world.common(), *world.inputs(),
                     "--private_out_dir", str(world.private), "--shared_json", str(world.root / "x.json"))
    stops(world, result, "evaluation artifacts without a coverage record", expect="coverage record is required")


def test_sealed_before_open(world: World) -> None:
    print("[6] a sealed video is refused before it is opened")
    ghost = world.root / "does_not_exist" / f"{VAL_IDS[2]}{TEACHER_SUFFIX}"
    paths = [world.teacher[VAL_IDS[0]], world.teacher[VAL_IDS[1]], ghost]
    validation = world._list("validation_ghost.txt", paths)
    pins = world.contract("pins_sealed.json", SEALED_IDS + [VAL_IDS[2]], paths)
    result = cli_run("audit", *world.common(pins=pins, validation=validation), *world.inputs()[:4],
                     "--private_out_dir", str(world.private), "--shared_json", str(world.root / "x.json"))
    stops(world, result, "sealed validation video with a non-existent path", expect="sealed video")
    check("missing" not in result.stderr, "the refusal is the seal, not a missing file")

    swapped = world._list("validation_swapped.txt", [world.teacher[i] for i in reversed(VAL_IDS)])
    result = cli_run("audit", *world.common(validation=swapped), *world.inputs()[:4],
                     "--private_out_dir", str(world.private), "--shared_json", str(world.root / "x.json"))
    stops(world, result, "a validation list in a different order", expect="full identity SHA-256 differs")
    wrong_sanity = world._list("sanity_wrong.txt", [world.teacher[i] for i in SANITY_IDS[:2]] + [world.teacher[VAL_IDS[0]]])
    result = cli_run("audit", *world.common(sanity=wrong_sanity), *world.inputs()[:4],
                     "--private_out_dir", str(world.private), "--shared_json", str(world.root / "x.json"))
    stops(world, result, "a sanity list with a validation video", expect="Step 0 sanity list")
    reordered = world._list("sanity_reordered.txt", [world.teacher[i] for i in reversed(SANITY_IDS)])
    result = cli_run("audit", *world.common(sanity=reordered), *world.inputs()[:4],
                     "--private_out_dir", str(world.private), "--shared_json", str(world.root / "x.json"))
    stops(world, result, "the same three videos in another file than Step 0's", expect="Step 0 sanity list")


def _run_args(world: World, shared: Path) -> list[str]:
    return ["run", *world.common(), *world.inputs(), "--artifact_coverage", str(world.coverage),
            "--private_out_dir", str(world.private), "--shared_json", str(shared), "--coordinate_mode", "pixel",
            "--min_train_core_evaluable", "4", "--min_validation_evaluable", "3"]


def test_run_success(world: World) -> None:
    print("[7] run: completes, writes private records and an aggregate-only shared JSON")
    shared_path = world.shared / "run.json"
    result = cli_run(*_run_args(world, shared_path))
    check(result.returncode == 0, f"run completes (stderr: {cli.scrub(result.stderr)[-300:]})")
    clean_output(world, result, "run")
    if result.returncode != 0:
        return
    text = shared_path.read_text(encoding="utf-8")
    shared = json.loads(text)
    check(not contains_video_identity(text) and str(world.root) not in text, "shared JSON: no identity, no path")
    h17 = shared["h17_1"]
    check(h17["per_representation"]["a_aabb"]["denominators"]["total"] == 6
          and h17["per_representation"]["a_aabb"]["denominators"]["input_unevaluable"] == 1,
          "H17-1 denominators: 6 = 1 input-unevaluable + 5 evaluable")
    check(h17["input"]["modes_entirely_excluded"] == 1 and h17["selection"]["status"] == "suspended_input_insufficient",
          "a wholly excluded mode above 10 % suspends the H17-1 judgement")
    check(shared["h17_2_representations"]["representations"] == ["a_aabb", "c_axis", "d_bestframe"]
          and shared["h17_2_representations"]["basis"] == "fixed_fallback_while_h17_1_undecided"
          and shared["h17_2_representations"]["role"] == "diagnostic_target_not_adoption_candidate",
          "while H17-1 is undecided, the fixed fallback (a), (c), (d) is computed as a diagnostic only")
    kept = h17["selection"].get("rule_result_not_used_while_suspended", {})
    check("excluded_representations" in kept and "not_selectable_representations" in kept,
          "the suspended selection keeps the rule result, with excluded representations, for the record")
    check("Infinity" not in text and "NaN" not in text, "no Infinity or NaN in the shared JSON")
    check(shared["input_minimums"]["synthetic_override"] is True, "the minimum override is recorded")
    best = shared["tables"]["validation"]["checkpoints"]["best"]
    c_axis = best["postprocess"]["P0"]["representations"]["c_axis"]
    check(c_axis["denominators"]["total"] == 3 and c_axis["denominators"]["success"] == 3,
          "validation denominators: 3 evaluable, 3 successes")
    check(c_axis["within_tau_rel_diagnostic"]["note"] == "diagnostic, not T_FL", "tau_rel is labelled as not T_FL")
    fp = best["postprocess"]["P0"]["fp_classes_pooled"]
    check(fp["n_fp"] == 9 and fp["n_pred_ignore"] > 0, "FP and ignore predictions pooled with their denominators")
    check("note" in best["postprocess"]["P3"]["p3_frames"], "P3 carries its target-frame note")
    check(best["best_postprocess_by_representation"] is not None
          and shared["tables"]["validation"]["checkpoints"]["last"]["best_postprocess_by_representation"] is None,
          "the best post-process is chosen on best only")
    check("sanity_in_sample" in shared["tables"], "sanity3 is a separate in-sample table")
    private = world.private / "geometry_run_DO_NOT_SHARE.json"
    check(private.is_file() and stat.S_IMODE(private.stat().st_mode) == 0o600
          and stat.S_IMODE(world.private.stat().st_mode) == 0o700, "private record 0600 in a 0700 directory")


def _mutate_npz(path: Path, **changes) -> bytes:
    original = path.read_bytes()
    with np.load(path) as payload:
        data = {k: payload[k].copy() for k in payload.files}
    for key, fn in changes.items():
        data[key] = fn(data[key])
    np.savez_compressed(path, **data)
    return original


def test_run_refusals(world: World) -> None:
    print("[8] run refusals: coverage, hash, NPZ absence, point consistency, mm")
    shared = world.root / "refused.json"
    result = cli_run(*[a if a != str(world.coverage) else str(world.root / "absent.json")
                       for a in _run_args(world, shared)])
    stops(world, result, "run without the coverage record")

    metrics = world.eval_root / "best" / "h5_metrics.csv"
    original = metrics.read_bytes()
    metrics.write_bytes(original + b"\n")
    stops(world, cli_run(*_run_args(world, shared)), "h5_metrics.csv changed after registration")
    metrics.write_bytes(original)

    npz_dir = world.eval_root / "best" / "predictions" / "validation"
    target = npz_dir / f"{VAL_IDS[0]}.npz"
    decoy = npz_dir / f"{VAL_IDS[0]}_09.npz"                     # a longer identity sharing the prefix
    target.rename(decoy)
    stops(world, cli_run(*_run_args(world, shared)), "the expected NPZ is missing", expect="no fallback")
    decoy.rename(target)

    n = teacher_arrays()["frame_order"].size
    background = int(np.flatnonzero(teacher_arrays()["point_label"] == 0)[0])
    cases = (
        ("one point fewer", dict(point_indices=lambda a: a[:-1], prob_femur=lambda a: a[:-1],
                                 pred_label=lambda a: a[:-1], vote_count=lambda a: a[:-1]), "shape"),
        ("a label contradicting p1", dict(pred_label=lambda a: np.where(np.arange(n) == background, 1, a)), "band"),
        ("an extra window vote", dict(vote_count=lambda a: np.where(np.arange(n) == background, a + 1, a)), "vote_count"),
        ("a consistent flip that changes the confusion counts",
         dict(prob_femur=lambda a: np.where(np.arange(n) == background, np.float32(0.95), a),
              pred_label=lambda a: np.where(np.arange(n) == background, 1, a)), "confusion counts"),
    )
    for label, changes, expect in cases:
        saved = _mutate_npz(target, **changes)
        stops(world, cli_run(*_run_args(world, shared)), label, expect=expect)
        target.write_bytes(saved)

    args = _run_args(world, shared)
    args[args.index("--coordinate_mode") + 1] = "mm"
    stops(world, cli_run(*args), "mm mode before the mm contract", expect="mm mode stops")
    check(not shared.exists(), "no shared JSON is written by a stopped run")


def main() -> int:
    print("Stage5 S5-17 geometry CLI synthetic test")
    if shutil.which("git") is None:
        print("  FAIL: git is required for the launcher-evidence checks")
        return 1
    with tempfile.TemporaryDirectory(prefix="s5_17_cli_") as tmp:
        root = Path(tmp)
        os.umask(0o022)
        test_raw_readers(root)
        test_privacy_helpers()
        world = World(root / "world")
        test_guard_in_process(world)
        test_register_coverage(world)
        test_audit(world)
        test_sealed_before_open(world)
        test_run_success(world)
        test_run_refusals(world)
    print(f"checks: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        return 1
    print("All geometry CLI checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
