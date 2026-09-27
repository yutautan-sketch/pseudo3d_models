from __future__ import annotations

import csv
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.split_contract import (  # noqa: E402
    SplitContractError,
    load_input_audit,
    verify_bootstrap_inputs,
)
from stage5.utils.split_identity import contains_video_identity, file_sha256, masked_shape  # noqa: E402
from stage5.utils.train_sanity_selection import select_train_paths  # noqa: E402

# ----------------------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the S0-1 input audit.
#
# Report 9.6, 11.7.3, 12.5. The audit's job is to FIX the expected input hashes
# so the builder has something external to verify against, and to confirm the
# train-sanity three by video identity rather than by counting aliases.
#
# The properties under test:
#   * every confirmation failure STOPS and writes no audit record -- an
#     unconfirmed input must not reach the builder wearing a fixed hash
#   * three aliases existing, or a count of three, is not proof
#   * the stdlib re-derivation agrees with evaluate_stage5.py's own copy
#     (extracted by text, so the two cannot drift apart unnoticed)
#   * no clinical FL value, and no per-token count, leaves the tool
#
# Pure stdlib; synthetic H5-named empty files only, and nothing is opened.
# ----------------------------------------------------------------------------

TOOL = REPO_ROOT / "checks" / "real_h5" / "audit_stage5_step0_inputs.py"
SUFFIX = (
    "_pointcloud_annotated_foreground_combined_v2_global_local_l75_w31_c12_area15_"
    "bboxrank_v7_cvat_authoritative_crop_quality_v1.h5"
)
FIXED_VIDEO = "20250626_124212_7300"

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def empty_pins(root: Path) -> Path:
    """A pins file local to the test; see Fixture._empty_pins."""
    path = root / "empty_pins_local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(
            json.dumps({"schema": "stage5_split_contract_pins_v1", "pins": {}}, indent=2),
            encoding="utf-8",
        )
    return path


def identity(index: int) -> str:
    return f"2025070{index % 10}_12{index:04d}_{index}"


def make_h5(root: Path, ident: str) -> Path:
    path = root / f"{ident}{SUFFIX}"
    path.write_bytes(b"synthetic; only the name and existence matter to S0-1")
    return path


def write_list(path: Path, paths: list[Path]) -> Path:
    path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return path


def write_id_map(path: Path, sanity: list[Path]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anonymous_id", "split", "original_h5_path"])
        writer.writeheader()
        for index, entry in enumerate(sanity):
            writer.writerow(
                {
                    "anonymous_id": f"train_sanity_{index:03d}",
                    "split": "train_sanity",
                    "original_h5_path": str(entry),
                }
            )
    return path




class Fixture:
    """A consistent synthetic S0-1 input set, 20 train + 4 validation."""

    def __init__(self, root: Path, *, num_train: int = 20, num_val: int = 4) -> None:
        self.root = root
        data = root / "data"
        data.mkdir(parents=True, exist_ok=True)
        self.train_paths = [make_h5(data, FIXED_VIDEO)]
        self.train_paths += [make_h5(data, identity(i)) for i in range(1, num_train)]
        self.val_paths = [make_h5(data, identity(500 + i)) for i in range(num_val)]
        self.sanity_paths = select_train_paths(
            list(self.train_paths), fixed_videos=[FIXED_VIDEO], num_random=2, seed=42
        )
        sanity_ids = {p.name.split("_pointcloud")[0] for p in self.sanity_paths}
        self.candidates = [
            p.name.split("_pointcloud")[0]
            for p in self.train_paths
            if p.name.split("_pointcloud")[0] not in sanity_ids
        ]
        self.train_list = write_list(root / "train.txt", self.train_paths)
        self.val_list = write_list(root / "val.txt", self.val_paths)
        self.sanity_list = write_list(root / "sanity.txt", self.sanity_paths)
        self.id_map = write_id_map(root / "id_map.csv", self.sanity_paths)
        self.num_train = num_train
        self.num_val = num_val

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

    def run(self, *extra: str, out_name: str = "audit") -> subprocess.CompletedProcess:
        audit_out = self.root / f"{out_name}_input_audit.json"
        private = self.root / f"{out_name}_private.json"
        cmd = [
            sys.executable, str(TOOL),
            "--old_train_list", str(self.train_list),
            "--old_val_list", str(self.val_list),
            "--sanity_list", str(self.sanity_list),
            "--evaluation_video_id_map", str(self.id_map),
            "--expected_train_count", str(self.num_train),
            "--expected_val_count", str(self.num_val),
            "--split_contract_pins", str(self._empty_pins()),
            "--audit_out", str(audit_out),
            "--private_json", str(private),
            "--shareable_json", str(self.root / f"{out_name}_share.json"),
            *extra,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        result.audit_out = audit_out  # type: ignore[attr-defined]
        result.private = private  # type: ignore[attr-defined]
        return result


def test_happy_path(root: Path) -> None:
    print("\n[1] a consistent input set is audited and its hashes fixed")
    fx = Fixture(root)
    result = fx.run()
    check(result.returncode == 0, f"the audit passes (stderr: {result.stderr.strip()[:200]})")
    check(Path(result.audit_out).is_file(), "the audit record is written")

    record = json.loads(Path(result.audit_out).read_text(encoding="utf-8"))
    check(record["schema"] == "stage5_step0_input_audit_v2", "the record carries the builder's schema")
    expected = record["expected_sha256"]
    check(
        set(expected) == {"old_train_list", "old_val_list", "sanity_list"},
        "exactly the three input labels the builder looks up; there is no fourth input",
    )
    check(
        expected["old_train_list"] == file_sha256(fx.train_list)
        and expected["sanity_list"] == file_sha256(fx.sanity_list),
        "the fixed hashes are the real ones at audit time",
    )
    check(record["complete"] is True, "the record is complete; there is no deferred pass")
    check(
        record["schema"] == "stage5_step0_input_audit_v2",
        "the record is issued under the revised schema",
    )
    check("17 candidates" not in result.stdout, "stdout does not enumerate candidates")
    check(f"candidates        : {len(fx.candidates)}" in result.stdout, "the candidate count is reported")


def test_failures_write_nothing(root: Path) -> None:
    print("\n[2] every unconfirmed input stops, and writes no record")
    fx = Fixture(root)

    def expect_stop(result, label: str) -> None:
        check(result.returncode == 2, f"{label} (exit {result.returncode})")
        check(not Path(result.audit_out).is_file(), f"{label}: no audit record is written")

    # A missing H5 that the list still names.
    victim = fx.train_paths[5]
    victim.unlink()
    expect_stop(fx.run(out_name="missing_h5"), "a listed H5 that does not exist stops")
    victim.write_bytes(b"restored")

    expect_stop(
        fx.run("--expected_train_count", str(fx.num_train + 1), out_name="wrong_count"),
        "a train list of the wrong length stops",
    )

    # No corroboration available.
    fx2 = Fixture(root / "b")
    result = subprocess.run(
        [sys.executable, str(TOOL),
         "--old_train_list", str(fx2.train_list), "--old_val_list", str(fx2.val_list),
         "--sanity_list", str(fx2.sanity_list),
         "--expected_train_count", str(fx2.num_train), "--expected_val_count", str(fx2.num_val),
         "--audit_out", str(root / "b" / "no_map.json"),
         "--split_contract_pins", str(empty_pins(root)),
         "--private_json", str(root / "b" / "no_map_private.json")],
        capture_output=True, text=True,
    )
    check(result.returncode == 2, "no evaluation ID map stops the audit")
    check(not (root / "b" / "no_map.json").is_file(), "and writes no record")

    # The map names three train_sanity videos, but not the same three: a count
    # of three and three aliases are both satisfied here.
    fx3 = Fixture(root / "c")
    wrong = [fx3.train_paths[1], fx3.train_paths[2], fx3.train_paths[3]]
    check(
        len(wrong) == 3 and set(wrong) != set(fx3.sanity_paths),
        "the decoy map has three rows that are the wrong three videos",
    )
    write_id_map(fx3.id_map, wrong)
    expect_stop(fx3.run(out_name="wrong_sanity"), "three matching aliases with the wrong identities stops")

    # Re-derivation without the list fingerprint pinned.
    fx4 = Fixture(root / "d")
    result = subprocess.run(
        [sys.executable, str(TOOL),
         "--old_train_list", str(fx4.train_list), "--old_val_list", str(fx4.val_list),
         "--evaluation_video_id_map", str(fx4.id_map), "--allow_sanity_rederivation",
         "--expected_train_count", str(fx4.num_train), "--expected_val_count", str(fx4.num_val),
         "--audit_out", str(root / "d" / "rederive.json"),
         "--split_contract_pins", str(empty_pins(root)),
         "--private_json", str(root / "d" / "rederive_private.json")],
        capture_output=True, text=True,
    )
    check(result.returncode == 2, "re-derivation without a pinned list fingerprint stops")
    check(
        "IN ORDER" in result.stderr or "order" in result.stderr,
        "and says why: the selection depends on the saved list order",
    )


def test_output_boundary(root: Path) -> None:
    print("\n[3] no GT statistic and no video ID leaves the tool")
    fx = Fixture(root)
    result = fx.run(out_name="boundary")
    check(result.returncode == 0, "the audit passes")

    combined = result.stdout + result.stderr
    check(
        not contains_video_identity(combined),
        "stdout/stderr carry no video ID",
    )
    check(
        "withheld" in json.dumps(json.loads((root / "boundary_share.json").read_text(
            encoding="utf-8"))).lower(),
        "the shareable record states what is withheld",
    )

    share = json.loads((root / "boundary_share.json").read_text(encoding="utf-8"))
    share_text = json.dumps(share)
    check(
        not contains_video_identity(share_text),
        "the shareable record carries no video ID",
    )
    check("distinct_non_numeric_tokens" not in share_text,
          "the shareable record does not even carry the token list")

    private = json.loads(Path(result.private).read_text(encoding="utf-8"))
    check("list_paths" in private, "the private record keeps the real paths")
    check(
        private["candidate_set"]["count"] == len(fx.candidates),
        "the private record carries the candidate count",
    )


def test_rederivation_matches_the_evaluation_copy(root: Path) -> None:
    print("\n[4] the stdlib re-derivation agrees with evaluate_stage5.py's own copy")
    source = (REPO_ROOT / "evaluate_stage5.py").read_text(encoding="utf-8")
    start = source.index("def select_train_paths(")
    end = source.index("\ndef ", start + 10)
    extracted = source[start:end]
    check("random.Random(" in extracted, "the evaluation copy was located")

    namespace: dict = {"random": random, "Path": Path, "Iterable": list}
    exec(compile(extracted, "<evaluate_stage5.select_train_paths>", "exec"), namespace)
    their = namespace["select_train_paths"]

    fx = Fixture(root)
    for seed in (42, 7, 1234):
        mine = select_train_paths(
            list(fx.train_paths), fixed_videos=[FIXED_VIDEO], num_random=2, seed=seed
        )
        theirs = their(
            list(fx.train_paths), fixed_videos=[FIXED_VIDEO], num_random=2, seed=seed
        )
        check(mine == theirs, f"both copies select the same videos at seed {seed}")

    # The property that makes re-derivation order-dependent, and so unsafe to
    # re-run against train_core: dropping one earlier entry changes the draw.
    shortened = [p for p in fx.train_paths if p != fx.train_paths[1]]
    reordered = select_train_paths(
        shortened, fixed_videos=[FIXED_VIDEO], num_random=2, seed=42
    )
    baseline = select_train_paths(
        list(fx.train_paths), fixed_videos=[FIXED_VIDEO], num_random=2, seed=42
    )
    check(
        reordered != baseline,
        "changing the list changes the selection, which is why the saved list is preferred",
    )


def test_previous_spec_records_are_refused(root: Path) -> None:
    print("\n[5] records written under the previous specification are refused")

    def write(name: str, payload: dict) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def refused(path: Path, label: str) -> str:
        try:
            load_input_audit(path)
        except SplitContractError as error:
            check(True, label)
            return str(error)
        check(False, f"{label} (was accepted)")
        return ""

    # A v1 record, including the clinical-FL-era partial audit with
    # complete:false, must not be reinterpreted under the revised spec. The
    # schema bump is what makes that mechanical (D-039, report 4.1).
    refused(
        write("v1_partial.json", {
            "schema": "stage5_step0_input_audit_v1",
            "complete": False,
            "expected_sha256": {"old_train_list": "a" * 64, "old_val_list": "b" * 64,
                                "sanity_list": "c" * 64},
        }),
        "a v1 partial audit record is refused",
    )
    message = refused(
        write("v1_complete.json", {
            "schema": "stage5_step0_input_audit_v1",
            "complete": True,
            "expected_sha256": {"old_train_list": "a" * 64, "old_val_list": "b" * 64,
                                "sanity_list": "c" * 64, "clinical_fl_csv": "d" * 64},
        }),
        "a v1 record that called itself complete is refused too",
    )
    check("re-run S0-1" in message, "the refusal says to re-run S0-1 rather than leaving it ambiguous")

    refused(
        write("no_schema.json", {"expected_sha256": {"old_train_list": "a" * 64}}),
        "a record with no schema at all is refused",
    )

    # And the current one is accepted, with exactly three inputs.
    fx = Fixture(root / "current")
    result = fx.run(out_name="current")
    check(result.returncode == 0, "the revised audit runs")
    loaded = load_input_audit(Path(result.audit_out))
    check(
        set(loaded) == {"old_train_list", "old_val_list", "sanity_list"},
        "and its record loads with exactly the three inputs",
    )


def test_naming_mismatch_is_diagnosable(root: Path) -> None:
    print("\n[6] a name that yields no video identity says which entry, without naming it")
    check(masked_shape("20250403_103433_091_x.h5") == "########_######_###_a.a#",
          "digits mask to # and letters to a")
    check("20250403" not in masked_shape("20250403_103433_091_x.h5"),
          "the masked shape does not reproduce the identifier")

    fx = Fixture(root)
    odd = root / "data" / f"case_17_left{SUFFIX}"
    odd.write_bytes(b"synthetic")
    broken = fx.train_paths[:-1] + [odd]
    write_list(fx.train_list, broken)
    result = fx.run(out_name="badname")
    check(result.returncode == 2, "an unparseable entry stops the audit")
    check("train entry #" in result.stderr, "the refusal names the list and the position")
    check("actual shape" in result.stderr, "and shows the name's shape")
    check(
        not contains_video_identity(result.stderr),
        "while printing no video identifier",
    )
    check("case_17_left" not in result.stderr, "and not the offending name either")


def test_second_naming_convention(root: Path) -> None:
    print("\n[7] both teacher naming conventions are recognised")
    from stage5.utils.split_identity import NAMING_CONVENTIONS, extract_video_identities

    check(
        {name for name, _ in NAMING_CONVENTIONS} == {"timestamp", "case"},
        "the enumerated conventions are timestamp and case",
    )
    check(
        extract_video_identities(f"20250403_103433_091{SUFFIX}") == ("20250403_103433_091",),
        "the timestamp convention resolves",
    )
    # The optional fourth segment. Without it the three-segment pattern every
    # other Stage 5 tool uses collapses distinct files onto one identity.
    check(
        extract_video_identities(f"20250403_103433_091_02{SUFFIX}") == ("20250403_103433_091_02",),
        "a trailing fourth segment is part of the identity",
    )
    check(
        len({extract_video_identities(f"20250403_103433_091_{s}{SUFFIX}")[0]
             for s in ("01", "02", "03")}) == 3,
        "files differing only in the fourth segment get three distinct identities",
    )
    check(
        extract_video_identities(f"20250403_103433_091_02{SUFFIX}")[0]
        != extract_video_identities(f"20250403_103433_091{SUFFIX}")[0],
        "and none of them collides with the three-segment form",
    )
    check(
        extract_video_identities(f"1-2_34_56{SUFFIX}") == ("1-2_34_56",),
        "the case convention with two trailing groups resolves",
    )
    check(
        extract_video_identities(f"1-2_34{SUFFIX}") == ("1-2_34",),
        "the case convention with one trailing group resolves",
    )
    check(
        extract_video_identities(f"12-345_67_89{SUFFIX}") == ("12-345_67_89",),
        "multi-digit case numbers resolve",
    )
    # The two must not cross-match, or a video could get two identities.
    for name in (f"20250403_103433_091{SUFFIX}", f"1-2_34_56{SUFFIX}"):
        check(len(extract_video_identities(name)) == 1, f"exactly one identity for {masked_shape(name)[:20]}")
    # No length limit on the case form's trailing groups, so it never
    # truncates. An earlier draft capped them at two digits and collapsed
    # `1-2_34_567` and `1-2_34_891` onto `1-2_34` -- one identity for two
    # videos, which is the one failure a seal contract cannot have.
    check(
        extract_video_identities(f"1-2_34_567{SUFFIX}") == ("1-2_34_567",),
        "a three-digit trailing group is not truncated",
    )
    check(
        len({extract_video_identities(f"1-2_34_{n}{SUFFIX}")[0] for n in ("56", "567", "891")}) == 3,
        "trailing groups of different lengths stay distinct identities",
    )
    # The accepted cost of removing that limit: a name carrying BOTH forms
    # resolves to one combined identity rather than the timestamp alone. No
    # such name exists in the 180-video set (checked 2026-09-27), and the
    # result is still a single stable identity, not a collision.
    combined = extract_video_identities(f"1-2_20250403_103433_091{SUFFIX}")
    check(len(combined) == 1, "a name carrying both forms still yields exactly one identity")
    check(
        combined[0] == "1-2_20250403_103433_091",
        "and that identity is the whole token, not a silent truncation",
    )
    # Still an enumerated set: an unknown shape is refused, not guessed.
    check(extract_video_identities(f"case_17_left{SUFFIX}") == (), "an unknown convention yields nothing")

    # And the audit accepts a mixed-convention list end to end.
    fx = Fixture(root)
    data = root / "data"
    mixed = list(fx.train_paths)
    for offset, stem in enumerate(("1-2_34_56", "3-4_78", "20250403_103433_091_02")):
        path = data / f"{stem}{SUFFIX}"
        path.write_bytes(b"synthetic")
        mixed[-(offset + 1)] = path
    write_list(fx.train_list, mixed)
    fx.train_paths = mixed
    fx.sanity_paths = select_train_paths(
        list(mixed), fixed_videos=[FIXED_VIDEO], num_random=2, seed=42
    )
    write_list(fx.sanity_list, fx.sanity_paths)
    write_id_map(fx.id_map, fx.sanity_paths)
    result = fx.run(out_name="mixed")
    check(result.returncode == 0,
          f"a list mixing both conventions is audited (stderr: {result.stderr.strip()[:200]})")
    check(
        not contains_video_identity(result.stdout),
        "and no identifier of either convention reaches stdout",
    )


def test_duplicate_identity_says_which_kind(root: Path) -> None:
    print("\n[8] a repeated identity says whether it is a duplicate entry or a collapse")
    fx = Fixture(root)
    doubled = list(fx.train_paths) + [fx.train_paths[0]]
    write_list(fx.train_list, doubled)
    result = fx.run("--expected_train_count", str(len(doubled)), out_name="dup")
    check(result.returncode == 2, "a repeated identity stops the audit")
    check("resolves" in result.stderr and "more than one entry" in result.stderr,
          "the refusal reports how many identities are affected")
    check("the same file name listed more than once" in result.stderr,
          "and identifies this case as a duplicated entry, not an identity collapse")
    check("entries #" in result.stderr, "with the positions")
    check(not contains_video_identity(result.stderr), "and no video identifier")
    check("shape:" in result.stderr, "and the masked shape")


def main() -> None:
    print("Stage5 S5-16 Step 0: S0-1 input audit synthetic checks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for test in (
            test_happy_path,
            test_failures_write_nothing,
            test_output_boundary,
            test_rederivation_matches_the_evaluation_copy,
            test_previous_spec_records_are_refused,
            test_naming_mismatch_is_diagnosable,
            test_second_naming_convention,
            test_duplicate_identity_says_which_kind,
        ):
            sub = root / test.__name__
            sub.mkdir(parents=True, exist_ok=True)
            test(sub)

    print(f"\nchecks run: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
