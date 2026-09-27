from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.split_contract import (  # noqa: E402
    ActiveContract,
    Pins,
    SealRegistry,
    SplitContractError,
    SplitManifest,
    assert_bootstrap_allowed,
    assert_resume_consistent,
    load_input_audit,
    resolve_active_contract,
    verify_intermediate_fingerprint,
    verify_split_partition,
    verify_bootstrap_inputs,
    write_legacy_manifest,
)
from stage5.utils.split_identity import (  # noqa: E402
    ArtifactCoverage,
    IdentityError,
    contains_video_identity,
    extract_video_identities,
)

# ----------------------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the B' seal contract.
#
# Report 9.9, 10.8, 10.10. No real data, no real manifest, no H5 is opened
# anywhere here -- the sealed set is checked by refusing PATHS THAT DO NOT
# EXIST, which is how a guard that fires before the open is told apart from one
# that fires when the read fails. Pure stdlib.
#
# The properties under test, in one place:
#   * a missing/unapproved/mismatched contract STOPS; it never means allowed
#   * the registry denies before any manifest allows, legacy included
#   * a pin outside the artifact is the expected hash, not the artifact's own
#   * an artifact whose video membership cannot be established is refused
#   * refusals carry no video ID, no path, and no statistic
# ----------------------------------------------------------------------------

SEALED = ["20250701_101010_101", "20250702_111111_202"]
OPEN_VIDEOS = ["20250703_121212_303", "20250704_131313_404"]
TEACHER_SUFFIX = (
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


def expect_stop(fn, label: str, *, exc=(SplitContractError, IdentityError)) -> Exception | None:
    """The call must raise. Returning normally is the failure being hunted."""
    try:
        fn()
    except exc as error:
        check(True, label)
        return error
    except Exception as error:  # noqa: BLE001
        check(False, f"{label} (raised {type(error).__name__} instead: {error})")
        return None
    check(False, f"{label} (returned normally instead of stopping)")
    return None


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher_path(root: Path, identity: str) -> Path:
    return root / f"{identity}{TEACHER_SUFFIX}"


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


SEALED_LIST_IDENTITY = "sealed-list-identity-hash"


def write_registry(root: Path, sealed=SEALED, *, sealed_list_identity=SEALED_LIST_IDENTITY) -> Path:
    return write_json(
        root / "sealed_registry.json",
        {
            "schema": "stage5_seal_registry_v1",
            "sealed_video_identities": list(sealed),
            "sealed_list_identity_sha256": sealed_list_identity,
            "sealed_until": "S5-20c",
            "prohibited_checkpoint_rule": (
                "A checkpoint whose training set does not match train_core may never be scored "
                "on internal_test, before or after the seal is lifted."
            ),
        },
    )


def write_manifest(root: Path, *, name="s5_16_bprime", sealed=("internal_test",),
                   allows_new_training=True, allows_directory_mode=False,
                   train_identity="train-core-identity-hash", seal_registry_required=True,
                   internal_identity=SEALED_LIST_IDENTITY) -> Path:
    return write_json(
        root / f"{name}.json",
        {
            "schema": "stage5_split_contract_v1",
            "name": name,
            "lists": {
                "train_core": {"count": 144, "identity_sha256": train_identity, "content_sha256": "c1"},
                "validation": {"count": 18, "identity_sha256": "v", "content_sha256": "c2"},
                "internal_test": {
                    "count": 18, "identity_sha256": internal_identity, "content_sha256": "c3",
                },
            },
            "sealed_splits": list(sealed),
            "seal_registry_required": seal_registry_required,
            "allows_new_training": allows_new_training,
            "allows_directory_mode": allows_directory_mode,
        },
    )


def write_pins(root: Path, entries: dict) -> Path:
    return write_json(
        root / "pins.json",
        {
            "schema": "stage5_split_contract_pins_v1",
            "pins": {
                key: {
                    "path": str(value["path"]),
                    "expected_sha256": value["expected_sha256"],
                    "approved_on": "2026-09-22",
                    "approved_scope": "synthetic test",
                }
                for key, value in entries.items()
            },
        },
    )


def standard_setup(root: Path, **manifest_kwargs) -> tuple[Path, Path, Path]:
    registry = write_registry(root)
    manifest = write_manifest(root, **manifest_kwargs)
    pins = write_pins(
        root,
        {
            "seal_registry": {"path": registry, "expected_sha256": sha256_of(registry)},
            manifest.stem: {"path": manifest, "expected_sha256": sha256_of(manifest)},
        },
    )
    return registry, manifest, pins


# ---------------------------------------------------------------------------


def test_missing_and_unapproved_contract(root: Path) -> None:
    print("\n[1] a missing or unapproved contract stops; it is never an allow")
    expect_stop(
        lambda: resolve_active_contract(manifest_key="anything", pins_path=root / "does_not_exist.json"),
        "absent pins file stops the run",
    )

    empty_pins = write_json(root / "empty_pins.json", {"schema": "stage5_split_contract_pins_v1", "pins": {}})
    expect_stop(
        lambda: resolve_active_contract(manifest_key="s5_16_bprime", pins_path=empty_pins),
        "pins without a seal_registry entry stops the normal path",
    )

    registry, manifest, _ = standard_setup(root)
    pins_no_manifest = write_pins(
        root, {"seal_registry": {"path": registry, "expected_sha256": sha256_of(registry)}}
    )
    expect_stop(
        lambda: resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins_no_manifest),
        "an unpinned manifest is refused rather than used unverified",
    )
    expect_stop(
        lambda: resolve_active_contract(manifest_key=None, pins_path=pins_no_manifest),
        "no manifest selected stops the run",
    )


def test_pin_is_the_authority(root: Path) -> None:
    print("\n[2] the expected hash comes from the pins file, not from the artifact")
    registry, manifest, pins = standard_setup(root)

    original = manifest.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["allows_directory_mode"] = True
    # The tamperer also writes a correct self-stated hash. If the code ever
    # trusted that field, this would sail through.
    tampered["sha256"] = "placeholder"
    manifest.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
    tampered["sha256"] = sha256_of(manifest)
    manifest.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    error = expect_stop(
        lambda: resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins),
        "a manifest edited together with its own self-stated hash still stops",
    )
    check(
        error is not None and "pin" in str(error).lower(),
        "the refusal names the pin as the authority",
    )
    manifest.write_text(original, encoding="utf-8")

    registry_text = registry.read_text(encoding="utf-8")
    registry.write_text(registry_text.replace("S5-20c", "S5-19"), encoding="utf-8")
    expect_stop(
        lambda: resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins),
        "an edited seal registry stops on its pin",
    )
    registry.write_text(registry_text, encoding="utf-8")

    contract = resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins)
    check(contract.registry.sealed_until == "S5-20c", "the restored, pinned contract loads")


def test_manifest_claims_are_checked_against_the_registry(root: Path) -> None:
    print("\n[3] a manifest's claims about the seal are checked, not taken on the split's name")
    registry = write_registry(root)

    def pins_for(manifest: Path) -> Path:
        return write_pins(
            root,
            {
                "seal_registry": {"path": registry, "expected_sha256": sha256_of(registry)},
                manifest.stem: {"path": manifest, "expected_sha256": sha256_of(manifest)},
            },
        )

    # Every manifest must acknowledge that the current registry applies to it.
    silent = write_manifest(root, name="silent_about_registry", sealed=(), seal_registry_required=False)
    expect_stop(
        lambda: resolve_active_contract(manifest_key="silent_about_registry", pins_path=pins_for(silent)),
        "a manifest that does not set seal_registry_required=true is rejected",
    )

    # Declaring no sealed split of its own is legitimate, and is NOT the same
    # as claiming nothing is sealed: the registry still decides every deny.
    defers = write_manifest(root, name="defers_to_registry", sealed=())
    contract = resolve_active_contract(manifest_key="defers_to_registry", pins_path=pins_for(defers))
    check(contract.manifest.sealed == (), "a manifest may define no sealed split of its own")
    check(contract.manifest.seal_registry_required, "it still states that the registry applies")
    expect_stop(
        lambda: contract.assert_paths_allowed([teacher_path(root, SEALED[0])], purpose="probe"),
        "and the registry still denies the sealed video through it",
    )

    bad = write_manifest(root, name="seals_unlisted", sealed=("no_such_split",))
    expect_stop(
        lambda: resolve_active_contract(manifest_key="seals_unlisted", pins_path=pins_for(bad)),
        "a manifest sealing a split it does not list is rejected",
    )

    # Naming the sealed split is not the check: its fingerprint must match the
    # registry's record of the list the sealed videos came from.
    wrong = write_manifest(root, name="wrong_fingerprint", internal_identity="some-other-hash")
    expect_stop(
        lambda: resolve_active_contract(manifest_key="wrong_fingerprint", pins_path=pins_for(wrong)),
        "claiming to hold the sealed split with a mismatched fingerprint is rejected",
    )

    unbound = write_registry(root / "unbound", sealed_list_identity=None)
    claims = write_manifest(root / "unbound", name="claims_sealed")
    pins3 = write_pins(
        root / "unbound",
        {
            "seal_registry": {"path": unbound, "expected_sha256": sha256_of(unbound)},
            "claims_sealed": {"path": claims, "expected_sha256": sha256_of(claims)},
        },
    )
    expect_stop(
        lambda: resolve_active_contract(manifest_key="claims_sealed", pins_path=pins3),
        "a claim to hold the sealed split is rejected when the registry records nothing to check it against",
    )


def test_deny_before_open(root: Path) -> None:
    print("\n[4] sealed videos are refused before anything is opened")
    _registry, _manifest, pins = standard_setup(root)
    contract = resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins)

    absent = root / "nowhere" / f"{SEALED[0]}{TEACHER_SUFFIX}"
    check(not absent.exists(), "the sealed probe path genuinely does not exist")
    error = expect_stop(
        lambda: contract.assert_paths_allowed([absent], purpose="probe"),
        "a sealed path is refused even though the file does not exist (so the open never happened)",
    )
    check(
        error is not None and not isinstance(error, FileNotFoundError),
        "the refusal is a contract stop, not a failed read",
    )

    contract.assert_paths_allowed(
        [root / f"{OPEN_VIDEOS[0]}{TEACHER_SUFFIX}"], purpose="probe"
    )
    check(True, "a non-sealed path passes")

    expect_stop(
        lambda: contract.assert_paths_allowed(
            [
                root / f"{OPEN_VIDEOS[0]}{TEACHER_SUFFIX}",
                root / f"{SEALED[1]}{TEACHER_SUFFIX}",
            ],
            purpose="probe",
        ),
        "one sealed entry among allowed ones stops the whole batch",
    )

    # Same video, other artifact kinds and other path spellings.
    for label, name in (
        ("intermediate pseudo-3D H5", f"{SEALED[0]}_pseudo3d.h5"),
        ("prediction NPZ", f"{SEALED[0]}.npz"),
        ("per-video JSON", f"gt_regions_{SEALED[0]}.json"),
        ("a different mount prefix", f"/other/mount/{SEALED[0]}{TEACHER_SUFFIX}"),
    ):
        expect_stop(
            lambda name=name: contract.assert_paths_allowed([Path(name)], purpose="probe"),
            f"the seal also covers the {label}",
        )


def test_identity_resolution(root: Path) -> None:
    print("\n[5] an artifact whose video membership is unknown is refused")
    _registry, _manifest, pins = standard_setup(root)
    contract = resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins)

    check(extract_video_identities("summary.json") == (), "a shared name yields no identity")
    check(
        extract_video_identities(f"x_{SEALED[0]}_y_{OPEN_VIDEOS[0]}.npz") == (SEALED[0], OPEN_VIDEOS[0]),
        "several identities in one name are all found",
    )
    check(
        extract_video_identities(f"1234{SEALED[0]}.h5") == (),
        "a longer digit run does not yield a shifted false identity",
    )

    expect_stop(
        lambda: contract.assert_paths_allowed([root / "aggregate_summary.json"], purpose="probe"),
        "an unnamed artifact with no coverage is refused, not waved through",
    )

    aggregate = root / "aggregate_summary.json"
    aggregate.write_text('{"videos": 3}', encoding="utf-8")

    covering_sealed = write_json(
        root / "coverage_sealed.json",
        {
            "schema": "stage5_artifact_coverage_v1",
            "entries": [
                {
                    "artifact_name": "aggregate_summary.json",
                    "sha256": sha256_of(aggregate),
                    "video_identities": [OPEN_VIDEOS[0], SEALED[0]],
                }
            ],
        },
    )
    sealed_contract = resolve_active_contract(
        manifest_key="s5_16_bprime", pins_path=pins, coverage_path=covering_sealed
    )
    expect_stop(
        lambda: sealed_contract.assert_paths_allowed([aggregate], purpose="probe"),
        "one sealed video inside a multi-video aggregate refuses the aggregate",
    )

    covering_open = write_json(
        root / "coverage_open.json",
        {
            "schema": "stage5_artifact_coverage_v1",
            "entries": [
                {
                    "artifact_name": "aggregate_summary.json",
                    "sha256": sha256_of(aggregate),
                    "video_identities": list(OPEN_VIDEOS),
                }
            ],
        },
    )
    open_contract = resolve_active_contract(
        manifest_key="s5_16_bprime", pins_path=pins, coverage_path=covering_open
    )
    open_contract.assert_paths_allowed([aggregate], purpose="probe")
    check(True, "an aggregate covering only open videos passes")

    aggregate.write_text('{"videos": 4}', encoding="utf-8")
    expect_stop(
        lambda: open_contract.assert_paths_allowed([aggregate], purpose="probe"),
        "a coverage record whose hash no longer matches the artifact stops",
    )

    empty_cov = write_json(
        root / "coverage_empty.json",
        {
            "schema": "stage5_artifact_coverage_v1",
            "entries": [
                {
                    "artifact_name": "aggregate_summary.json",
                    "sha256": sha256_of(aggregate),
                    "video_identities": [],
                }
            ],
        },
    )
    empty_contract = resolve_active_contract(
        manifest_key="s5_16_bprime", pins_path=pins, coverage_path=empty_cov
    )
    expect_stop(
        lambda: empty_contract.assert_paths_allowed([aggregate], purpose="probe"),
        "an empty coverage record is not a statement that nothing is sealed",
    )

    expect_stop(
        lambda: ArtifactCoverage.from_json(root / "coverage_open.json").lookup(root / "vanished.json"),
        "a coverage lookup for an artifact with no record stops",
    )


def test_legacy_cannot_bypass(root: Path) -> None:
    print("\n[6] a legacy manifest reproduces a condition; it does not lift the seal")
    registry = write_registry(root)
    old_train = root / "legacy_train.txt"
    old_val = root / "legacy_val.txt"
    # The old train list still names the now-sealed videos, as the real one does.
    old_train.write_text(
        "\n".join(str(teacher_path(root, v)) for v in (OPEN_VIDEOS[0], *SEALED)) + "\n", encoding="utf-8"
    )
    old_val.write_text(str(teacher_path(root, OPEN_VIDEOS[1])) + "\n", encoding="utf-8")

    legacy = write_legacy_manifest(
        out_path=root / "legacy_s5_15.json", name="legacy_s5_15", train_list=old_train, val_list=old_val
    )
    pins = write_pins(
        root,
        {
            "seal_registry": {"path": registry, "expected_sha256": sha256_of(registry)},
            "legacy_s5_15": {"path": legacy, "expected_sha256": sha256_of(legacy)},
        },
    )
    contract = resolve_active_contract(manifest_key="legacy_s5_15", pins_path=pins)

    check(
        contract.manifest.sealed == (),
        "legacy declares no sealed split of its own: its 162-video list is not the 18 sealed videos",
    )
    check(
        contract.manifest.seal_registry_required,
        "legacy still states that the current registry applies to it in full",
    )
    expect_stop(contract.assert_new_training_allowed, "legacy refuses new training")
    expect_stop(contract.assert_directory_mode_allowed, "legacy refuses directory mode")

    for purpose in ("evaluation", "inference", "checker read"):
        expect_stop(
            lambda p=purpose: contract.assert_paths_allowed(
                [teacher_path(root, SEALED[0])], purpose=p
            ),
            f"legacy still refuses a sealed video for {purpose}",
        )

    contract.assert_paths_allowed([teacher_path(root, OPEN_VIDEOS[0])], purpose="evaluation")
    check(True, "legacy passes for targets that do not intersect the seal")


def test_checkpoint_rule(root: Path) -> None:
    print("\n[7] a checkpoint trained on sealed videos may not be scored on them")
    _registry, _manifest, pins = standard_setup(root)
    contract = resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins)

    expect_stop(
        lambda: contract.assert_checkpoint_allowed(
            train_identity_sha256="old-162-video-run", split_name="internal_test"
        ),
        "an old checkpoint is refused on the sealed split",
    )
    contract.assert_checkpoint_allowed(
        train_identity_sha256="train-core-identity-hash", split_name="internal_test"
    )
    check(True, "a train_core checkpoint is accepted on the sealed split")
    contract.assert_checkpoint_allowed(train_identity_sha256="anything", split_name="validation")
    check(True, "the rule does not fire on a non-sealed split")


def test_bootstrap(root: Path) -> None:
    print("\n[8] the initial build path cannot be re-entered or widened")
    boot = root / "boot"
    boot.mkdir(parents=True, exist_ok=True)
    empty_pins = write_json(boot / "pins.json", {"schema": "stage5_split_contract_pins_v1", "pins": {}})

    assert_bootstrap_allowed(registry_out=boot / "sealed_registry.json", pins_path=empty_pins)
    check(True, "the initial build runs while no registry is approved and none exists")

    audited = boot / "train162.txt"
    audited.write_text("a\nb\n", encoding="utf-8")
    inputs = {"train162": str(audited)}

    # The expected hash arrives from the S0-1 audit record. A hash recomputed
    # from the input and compared with itself would always match.
    audit = write_json(
        boot / "input_audit.json",
        {
            "schema": "stage5_step0_input_audit_v2",
            "expected_sha256": {"train162": sha256_of(audited)},
        },
    )
    expected = load_input_audit(audit)
    observed = verify_bootstrap_inputs(inputs, expected_sha256=expected)
    check(observed["train162"] == sha256_of(audited), "an input matching the audited hash is accepted")

    audited.write_text("a\nb\nc\n", encoding="utf-8")
    expect_stop(
        lambda: verify_bootstrap_inputs(inputs, expected_sha256=expected),
        "an input that no longer matches the audited hash stops the build",
    )
    expect_stop(
        lambda: verify_bootstrap_inputs(inputs, expected_sha256={}),
        "no audited hashes at all stops, rather than verifying nothing",
    )
    expect_stop(
        lambda: verify_bootstrap_inputs(
            {"train162": str(audited), "extra": str(audited)}, expected_sha256=expected
        ),
        "an input the audit record does not cover is not read",
    )
    expect_stop(
        lambda: load_input_audit(boot / "no_such_audit.json"),
        "a missing audit record stops",
    )
    stale = write_json(
        boot / "v1_audit.json",
        {
            "schema": "stage5_step0_input_audit_v1",
            "expected_sha256": {"train162": sha256_of(audited)},
        },
    )
    expect_stop(
        lambda: load_input_audit(stale),
        "an audit record from the previous specification is refused, not reinterpreted",
    )

    assert_resume_consistent({"schema": "x", "input_sha256": observed}, observed)
    check(True, "resuming on the same inputs is allowed")
    expect_stop(
        lambda: assert_resume_consistent(
            {"schema": "x", "input_sha256": observed}, {"train162": "different"}
        ),
        "resuming with changed inputs stops rather than mixing two input sets",
    )

    existing = write_registry(boot)
    expect_stop(
        lambda: assert_bootstrap_allowed(registry_out=existing, pins_path=empty_pins),
        "the initial build refuses to overwrite an existing registry",
    )

    approved_pins = write_pins(
        boot, {"seal_registry": {"path": existing, "expected_sha256": sha256_of(existing)}}
    )
    expect_stop(
        lambda: assert_bootstrap_allowed(
            registry_out=boot / "another_registry.json", pins_path=approved_pins
        ),
        "once a registry is approved the initial build path is closed",
    )


def test_refusals_leak_nothing(root: Path) -> None:
    print("\n[9] refusals carry no video ID, no path and no statistic")
    _registry, _manifest, pins = standard_setup(root)
    contract = resolve_active_contract(manifest_key="s5_16_bprime", pins_path=pins)
    try:
        contract.assert_paths_allowed([teacher_path(root, SEALED[0])], purpose="probe")
        message = ""
    except SplitContractError as error:
        message = str(error)
    check(bool(message), "the sealed read produced a refusal message")
    check(SEALED[0] not in message, "the refusal does not repeat the video ID")
    check(str(root) not in message, "the refusal does not repeat the real path")
    check(
        not contains_video_identity(message),
        "no timestamp-like video ID appears anywhere in the refusal",
    )


def test_cli(root: Path) -> None:
    print("\n[10] the CLI that bash calls before the preflight")
    _registry, _manifest, pins = standard_setup(root)
    module = REPO_ROOT / "stage5" / "utils" / "split_contract.py"

    allowed_list = root / "allowed.txt"
    allowed_list.write_text(
        "\n".join(str(teacher_path(root, v)) for v in OPEN_VIDEOS) + "\n", encoding="utf-8"
    )
    sealed_list = root / "sealed_input.txt"
    sealed_list.write_text(str(teacher_path(root, SEALED[0])) + "\n", encoding="utf-8")

    ok = subprocess.run(
        [sys.executable, str(module), "assert", "--split_manifest", "s5_16_bprime",
         "--pins", str(pins), "--paths_from", str(allowed_list), "--purpose", "synthetic"],
        capture_output=True, text=True,
    )
    check(ok.returncode == 0, f"allowed inputs exit 0 (got {ok.returncode}: {ok.stderr.strip()[:200]})")
    check("Split contract satisfied" in ok.stdout, "success prints the contract summary")
    check(
        not contains_video_identity(ok.stdout + ok.stderr),
        "success output carries no video ID",
    )

    denied = subprocess.run(
        [sys.executable, str(module), "assert", "--split_manifest", "s5_16_bprime",
         "--pins", str(pins), "--paths_from", str(sealed_list), "--purpose", "synthetic"],
        capture_output=True, text=True,
    )
    check(denied.returncode == 2, f"a sealed input exits 2 (got {denied.returncode})")
    check(
        not contains_video_identity(denied.stdout + denied.stderr),
        "the refusal output carries no video ID",
    )

    unpinned = subprocess.run(
        [sys.executable, str(module), "assert", "--split_manifest", "not_approved",
         "--pins", str(pins), "--paths_from", str(allowed_list)],
        capture_output=True, text=True,
    )
    check(unpinned.returncode == 2, "an unapproved manifest name exits 2")

    no_selection = subprocess.run(
        [sys.executable, str(module), "assert", "--pins", str(pins), "--paths_from", str(allowed_list)],
        capture_output=True, text=True,
    )
    check(no_selection.returncode == 2, "omitting the manifest selection exits 2")


def test_partition_checks(root: Path) -> None:
    print("\n[11] the six partition checks a new split must pass before it is written")
    old_train = [teacher_path(root, f"2025070{i % 10}_120000_{i}") for i in range(162)]
    old_val = [teacher_path(root, f"2025080{i % 10}_120000_{i}") for i in range(18)]
    sanity = [old_train[0], old_train[7], old_train[19]]
    internal_test = [p for p in old_train if p not in sanity][:18]
    train_core = [p for p in old_train if p not in internal_test]

    def partition(**overrides):
        kwargs = dict(
            train_core=train_core, validation=old_val, internal_test=internal_test,
            old_train=old_train, old_validation=old_val, sanity=sanity,
        )
        kwargs.update(overrides)
        return lambda: verify_split_partition(**kwargs)

    summary = partition()()
    check(summary["all_checks_passed"], "a correct partition passes all six checks")
    check(summary["counts"] == {"train_core": 144, "validation": 18, "internal_test": 18},
          "the counts are 144/18/18")

    expect_stop(partition(train_core=train_core + [internal_test[0]]),
                "overlap between train_core and internal_test stops")
    expect_stop(partition(train_core=train_core[:-1]), "a wrong train_core count stops")
    expect_stop(partition(internal_test=internal_test[:-1]), "a wrong internal_test count stops")
    expect_stop(partition(train_core=train_core + [train_core[0]]),
                "a duplicate inside a split stops")

    moved_sanity = [p for p in train_core if p != sanity[1]]
    leaked = [p for p in internal_test if p != internal_test[0]] + [sanity[1]]
    expect_stop(partition(train_core=moved_sanity + [internal_test[0]], internal_test=leaked),
                "a train-sanity video leaving train_core stops")

    expect_stop(partition(validation=list(reversed(old_val))),
                "reordering validation stops: its order is part of the contract")
    outsider = teacher_path(root, "20259999_120000_999")
    expect_stop(partition(validation=old_val[:-1] + [outsider]),
                "changing a validation entry stops")
    expect_stop(partition(old_train=old_train[:-1]),
                "a partition that does not reconstruct the old train list stops")

    # Same videos under another mount prefix: content identical, spelling not.
    remounted = [Path("/other/mount") / p.name for p in old_val]
    moved = verify_split_partition(
        train_core=train_core, validation=remounted, internal_test=internal_test,
        old_train=old_train, old_validation=old_val, sanity=sanity,
    )
    check(moved["all_checks_passed"], "a pure path-prefix change still satisfies the video-identity checks")
    check(not moved["validation_representation_identical"],
          "the path-spelling difference is reported rather than hidden")
    check(moved["identity_sha256"]["validation"] != moved["content_sha256"]["validation"],
          "identity and content fingerprints are kept apart")


def test_intermediate_fingerprint(root: Path) -> None:
    print("\n[12] the dry run's intermediate is checked by its own fingerprint")
    intermediate = root / "stratification.json"
    payload = {
        "schema": "stage5_bprime_stratification_v1",
        "candidate_identities": ["a", "b", "c"],
        "multi_region": {"a": True, "b": False, "c": False},
        "fl_group": {"a": 0, "b": 1, "c": 2},
        "spec": {"seed": 42},
    }
    write_json(intermediate, payload)
    fingerprint = sha256_of(intermediate)

    check(
        verify_intermediate_fingerprint(intermediate, fingerprint) == fingerprint,
        "an unchanged intermediate is accepted",
    )
    expect_stop(
        lambda: verify_intermediate_fingerprint(intermediate, None),
        "no expected fingerprint stops: a file cannot vouch for itself",
    )
    expect_stop(
        lambda: verify_intermediate_fingerprint(root / "absent.json", fingerprint),
        "a missing intermediate stops",
    )

    # The case the content checks alone would miss: same inputs, same candidate
    # set, same seed -- only the GT classification and FL groups altered.
    tampered = dict(payload)
    tampered["multi_region"] = {"a": False, "b": True, "c": False}
    tampered["fl_group"] = {"a": 2, "b": 1, "c": 0}
    write_json(intermediate, tampered)
    check(
        json.loads(intermediate.read_text(encoding="utf-8"))["spec"]["seed"] == 42
        and json.loads(intermediate.read_text(encoding="utf-8"))["candidate_identities"]
        == payload["candidate_identities"],
        "the tampered intermediate still passes every content check the builder makes",
    )
    expect_stop(
        lambda: verify_intermediate_fingerprint(intermediate, fingerprint),
        "but its altered GT classification and FL groups are caught by the fingerprint",
    )


def main() -> None:
    print("Stage5 S5-16 Step 0: split contract synthetic checks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for test in (
            test_missing_and_unapproved_contract,
            test_pin_is_the_authority,
            test_manifest_claims_are_checked_against_the_registry,
            test_deny_before_open,
            test_identity_resolution,
            test_legacy_cannot_bypass,
            test_checkpoint_rule,
            test_bootstrap,
            test_refusals_leak_nothing,
            test_cli,
            test_partition_checks,
            test_intermediate_fingerprint,
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
