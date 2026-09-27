from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.split_identity import (  # noqa: E402
    ArtifactCoverage,
    IdentityError,
    file_sha256,
    resolve_identities,
)

# ----------------------------------------------------------------------------
# S5-16 Step 0: the B' seal contract.
#
# Report sections 10.2/10.3 and 10.10.1. Three ideas that must stay separate:
#
#   1. The SEAL REGISTRY is the single authority for which videos are sealed.
#      It is resolved independently of whichever split manifest is in use, and
#      the deny it produces is evaluated BEFORE any manifest's allow. No
#      manifest -- legacy included -- can erase, replace or shrink it. This is
#      what stops an evaluation over the old 162-video train list from reaching
#      the current internal_test.
#
#      A manifest's own `sealed_splits` is a separate thing: the splits THAT
#      MANIFEST defines as sealed sets. A legacy manifest defines none -- its
#      162-video list is not the 18 sealed videos, and calling it so would
#      conflate the two -- but it must still set `seal_registry_required`, so
#      an absent or empty seal field is never read as "nothing is sealed". A
#      manifest that does claim to hold the sealed split is checked against the
#      registry's fingerprint for it; the split's name is not the check.
#
#   2. The PINS file is the repo-tracked, git-reviewed record of which registry
#      and which manifests are approved, and what their SHA-256 must be. A hash
#      a file carries about itself proves nothing -- a file and its self-stated
#      hash can be changed together -- so the expected value is fixed HERE,
#      outside the artifact. This prevents misdirection and unapproved
#      substitution; it does not stop the one user who can edit both, and that
#      limit is stated rather than papered over.
#
#   3. BOOTSTRAP is the one narrow path that builds the registry in the first
#      place. Requiring a registry to read the inputs that produce the registry
#      would deadlock, so bootstrap is a separate, explicitly-requested entry
#      with pre-audited inputs -- never a flag that switches the guard off.
#
# Deliberately stdlib-only, and every failure raises: this runs before the
# teacher preflight, before numpy/h5py/torch, before any H5 is opened.
# ----------------------------------------------------------------------------

PINS_SCHEMA = "stage5_split_contract_pins_v1"
REGISTRY_SCHEMA = "stage5_seal_registry_v1"
MANIFEST_SCHEMA = "stage5_split_contract_v1"
BOOTSTRAP_STATE_SCHEMA = "stage5_split_bootstrap_state_v1"
INPUT_AUDIT_SCHEMA = "stage5_step0_input_audit_v2"

DEFAULT_PINS_PATH = REPO_ROOT / "stage5" / "config" / "split_contract_pins.json"
MANIFEST_ENV_VAR = "STAGE5_SPLIT_MANIFEST"
PINS_ENV_VAR = "STAGE5_SPLIT_CONTRACT_PINS"
SEAL_REGISTRY_PIN_KEY = "seal_registry"


class SplitContractError(ValueError):
    """Any condition that must stop the run before an input is opened."""


def _read_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise SplitContractError(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SplitContractError(f"{label} could not be read: {path} ({error})") from error
    except json.JSONDecodeError as error:
        raise SplitContractError(f"{label} is not valid JSON: {path} ({error})") from error


# ---------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinEntry:
    path: Path
    expected_sha256: str
    approved_on: str
    approved_scope: str


class Pins:
    """Repo-tracked approvals. Changing one is a git diff the user reviews."""

    def __init__(self, entries: dict[str, PinEntry], *, source: Path) -> None:
        self._entries = entries
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Pins":
        pins_path = Path(path) if path else Path(os.environ.get(PINS_ENV_VAR, DEFAULT_PINS_PATH))
        payload = _read_json(pins_path, label="split-contract pins")
        if payload.get("schema") != PINS_SCHEMA:
            raise SplitContractError(
                f"pins schema is {payload.get('schema')!r}, expected {PINS_SCHEMA!r}: {pins_path}"
            )
        entries: dict[str, PinEntry] = {}
        for key, row in (payload.get("pins") or {}).items():
            missing = [field for field in ("path", "expected_sha256") if not row.get(field)]
            if missing:
                raise SplitContractError(f"pin {key!r} lacks {missing} in {pins_path}")
            entries[str(key)] = PinEntry(
                path=Path(str(row["path"])),
                expected_sha256=str(row["expected_sha256"]).lower(),
                approved_on=str(row.get("approved_on", "")),
                approved_scope=str(row.get("approved_scope", "")),
            )
        return cls(entries, source=pins_path)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def keys(self) -> Iterable[str]:
        return self._entries.keys()

    def get(self, key: str) -> PinEntry:
        if key not in self._entries:
            known = sorted(self._entries)
            raise SplitContractError(
                f"no approved pin named {key!r} in {self.source}; approved pins: {known}. "
                "An unpinned manifest is refused, never used unverified."
            )
        return self._entries[key]

    def resolve_verified(self, key: str, *, label: str) -> tuple[Path, str]:
        """Path and hash of a pinned artifact, verified against the pin itself.

        The expected value comes from the pins file, never from a field inside
        the artifact being checked.
        """
        entry = self.get(key)
        if not entry.path.is_file():
            raise SplitContractError(f"{label} pinned as {key!r} not found: {entry.path}")
        actual = file_sha256(entry.path)
        if actual != entry.expected_sha256:
            raise SplitContractError(
                f"{label} hash does not match its approved pin {key!r}\n"
                f"  expected (pins file): {entry.expected_sha256}\n"
                f"  actual   (on disk)  : {actual}\n"
                "  The pin is the authority; a matching hash written inside the artifact is not."
            )
        return entry.path, actual


# ---------------------------------------------------------------------------
# seal registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealRegistry:
    sealed_video_identities: frozenset[str]
    sealed_until: str
    prohibited_checkpoint_rule: str
    source_path: Path
    sha256: str
    # Fingerprint of the list the sealed videos came from. A manifest that
    # claims to hold that sealed split must match this, so the claim is checked
    # against the registry's own record rather than accepted on the split's
    # name (report 11.7.1, 3).
    sealed_list_identity_sha256: str | None = None

    @classmethod
    def load_verified(cls, pins: Pins) -> "SealRegistry":
        path, digest = pins.resolve_verified(SEAL_REGISTRY_PIN_KEY, label="seal registry")
        payload = _read_json(path, label="seal registry")
        if payload.get("schema") != REGISTRY_SCHEMA:
            raise SplitContractError(
                f"seal registry schema is {payload.get('schema')!r}, expected {REGISTRY_SCHEMA!r}"
            )
        identities = payload.get("sealed_video_identities")
        if not isinstance(identities, list):
            raise SplitContractError("seal registry lacks a sealed_video_identities list")
        return cls(
            sealed_video_identities=frozenset(str(item) for item in identities),
            sealed_until=str(payload.get("sealed_until", "")),
            prohibited_checkpoint_rule=str(payload.get("prohibited_checkpoint_rule", "")),
            source_path=path,
            sha256=digest,
            sealed_list_identity_sha256=(
                str(payload["sealed_list_identity_sha256"])
                if payload.get("sealed_list_identity_sha256")
                else None
            ),
        )


# ---------------------------------------------------------------------------
# split manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitManifest:
    name: str
    lists: dict
    # Splits THIS manifest itself defines as sealed sets. A manifest that
    # defines none says so explicitly via seal_registry_required; it does not
    # get to leave the field empty and mean nothing in particular.
    sealed: tuple[str, ...]
    seal_registry_required: bool
    allows_new_training: bool
    allows_directory_mode: bool
    stratification: dict
    source_path: Path
    sha256: str

    @classmethod
    def load_verified(cls, pin_key: str, *, pins: Pins, registry: SealRegistry) -> "SplitManifest":
        path, digest = pins.resolve_verified(pin_key, label="split manifest")
        payload = _read_json(path, label="split manifest")
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise SplitContractError(
                f"split manifest schema is {payload.get('schema')!r}, expected {MANIFEST_SCHEMA!r}"
            )
        manifest = cls(
            name=str(payload.get("name", pin_key)),
            lists=dict(payload.get("lists") or {}),
            sealed=tuple(str(item) for item in (payload.get("sealed_splits") or [])),
            seal_registry_required=bool(payload.get("seal_registry_required", False)),
            allows_new_training=bool(payload.get("allows_new_training", False)),
            allows_directory_mode=bool(payload.get("allows_directory_mode", False)),
            stratification=dict(payload.get("stratification") or {}),
            source_path=path,
            sha256=digest,
        )
        manifest._assert_consistent_with(registry)
        return manifest

    def _assert_consistent_with(self, registry: SealRegistry) -> None:
        """Check the manifest's claims about the seal against the registry.

        Two separate things, kept separate (report 11.7.1, 3):

        * Every manifest must acknowledge that the CURRENT registry applies to
          it, by setting `seal_registry_required: true`. That is what a
          manifest with no sealed split of its own says -- an absent or empty
          field is not read as "nothing is sealed", it is rejected.

        * A manifest that claims to HOLD the sealed split must match the
          registry's fingerprint for it. Declaring a split's name proves
          nothing on its own, so the name is not treated as the check.

        A legacy manifest therefore declares no sealed split. Its old train
        list does name videos that are now sealed, but the list as a whole is
        not the sealed set, and calling it one would conflate 162 videos with
        18. Deny is decided by the registry in either case.
        """
        if not self.seal_registry_required:
            raise SplitContractError(
                f"manifest {self.name!r} does not set seal_registry_required=true. Every manifest "
                "must state that the current seal registry applies to it; an unstated or empty "
                "seal field is never read as 'nothing is sealed'."
            )

        listed = {str(entry) for entry in self.lists}
        unknown = sorted(set(self.sealed) - listed)
        if unknown:
            raise SplitContractError(
                f"manifest {self.name!r} declares sealed split(s) {unknown} that it does not list"
            )

        for split_name in self.sealed:
            claimed = (self.lists.get(split_name) or {}).get("identity_sha256")
            if registry.sealed_list_identity_sha256 is None:
                raise SplitContractError(
                    f"manifest {self.name!r} claims to hold the sealed split {split_name!r}, but the "
                    "registry records no fingerprint to check that claim against"
                )
            if claimed != registry.sealed_list_identity_sha256:
                raise SplitContractError(
                    f"manifest {self.name!r} claims to hold the sealed split {split_name!r}, but its "
                    "fingerprint does not match the registry's. The split's name is not the check."
                )


# ---------------------------------------------------------------------------
# active contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveContract:
    registry: SealRegistry
    manifest: SplitManifest
    coverage: ArtifactCoverage | None

    def assert_paths_allowed(self, paths: Sequence[str | Path], *, purpose: str) -> None:
        """Stop before opening anything that belongs to a sealed video.

        Deny is evaluated first and on its own: the registry decides, not the
        manifest. Messages carry positions and counts only -- a real path or
        file name would itself be a private identifier (report 9.3).
        """
        sealed = self.registry.sealed_video_identities
        for index, raw in enumerate(paths):
            path = Path(raw)
            try:
                identities = resolve_identities(path, coverage=self.coverage)
            except IdentityError as error:
                raise SplitContractError(
                    f"{purpose}: input #{index} could not be tied to a video, so it is refused "
                    f"({error.__class__.__name__}). An artifact whose membership is unknown is never "
                    "allowed through. Detail is in the private log."
                ) from error
            hits = sealed & set(identities)
            if hits:
                raise SplitContractError(
                    f"{purpose}: input #{index} covers {len(hits)} sealed video(s) and is refused "
                    f"before being opened (sealed until {self.registry.sealed_until or 'further notice'}). "
                    "This applies to every manifest, legacy included."
                )

    def assert_new_training_allowed(self) -> None:
        if not self.manifest.allows_new_training:
            raise SplitContractError(
                f"manifest {self.manifest.name!r} has allows_new_training=false; it exists to reproduce "
                "an earlier condition, not to start new training."
            )

    def assert_directory_mode_allowed(self) -> None:
        if not self.manifest.allows_directory_mode:
            raise SplitContractError(
                f"manifest {self.manifest.name!r} has allows_directory_mode=false: a directory scan "
                "cannot be checked against an ordered contract and could pull in sealed videos. "
                "Use explicit train/val file lists."
            )

    def assert_checkpoint_allowed(self, *, train_identity_sha256: str | None, split_name: str) -> None:
        """Old checkpoints trained on sealed videos may not be scored on them."""
        if split_name not in self.manifest.sealed:
            return
        expected = (self.manifest.lists.get("train_core") or {}).get("identity_sha256")
        if not expected:
            raise SplitContractError(
                "the manifest does not record train_core's identity hash, so a checkpoint's training "
                "set cannot be verified against it"
            )
        if train_identity_sha256 != expected:
            raise SplitContractError(
                f"this checkpoint's training set does not match train_core, so it may not be evaluated "
                f"on {split_name!r}. {self.registry.prohibited_checkpoint_rule or ''}".strip()
            )


def resolve_active_contract(
    *,
    manifest_key: str | None = None,
    pins_path: str | Path | None = None,
    coverage_path: str | Path | None = None,
) -> ActiveContract:
    """Resolve pins -> registry -> manifest. Anything missing stops the run.

    `manifest_key` names an APPROVED PIN, not a path: a run cannot be pointed
    at an arbitrary file, and an unapproved manifest has no name to give.
    """
    pins = Pins.load(pins_path)
    key = manifest_key or os.environ.get(MANIFEST_ENV_VAR) or ""
    if not key:
        raise SplitContractError(
            "no split manifest selected: pass --split_manifest <pin name> or set "
            f"{MANIFEST_ENV_VAR}. A missing selection stops the run; it never means 'unrestricted'."
        )
    registry = SealRegistry.load_verified(pins)
    manifest = SplitManifest.load_verified(key, pins=pins, registry=registry)
    coverage = ArtifactCoverage.from_json(coverage_path) if coverage_path else None
    return ActiveContract(registry=registry, manifest=manifest, coverage=coverage)


# ---------------------------------------------------------------------------
# bootstrap (report 10.10.1)
# ---------------------------------------------------------------------------


def assert_bootstrap_allowed(*, registry_out: str | Path, pins_path: str | Path | None = None) -> None:
    """The one-time build of the registry, guarded against re-entry.

    Refuses once a registry is approved or already written, so the initial path
    can never be used to overwrite, shrink or regenerate a confirmed split.
    """
    try:
        pins = Pins.load(pins_path)
    except SplitContractError:
        pins = None
    if pins is not None and SEAL_REGISTRY_PIN_KEY in pins:
        raise SplitContractError(
            "a seal registry is already approved in the pins file: normal operation is in force and "
            "the initial build path is closed. Sealed outputs are never overwritten or regenerated here."
        )
    if Path(registry_out).exists():
        raise SplitContractError(
            f"seal registry already exists: {registry_out}. The initial build path does not overwrite it."
        )


def load_input_audit(path: str | Path) -> dict[str, str]:
    """Expected input hashes fixed during S0-1, read from the audit record.

    These have to arrive from somewhere other than the files being checked. A
    hash computed from an input and then compared with itself always matches
    and verifies nothing (report 11.7.3).
    """
    audit_path = Path(path)
    payload = _read_json(audit_path, label="input audit record")
    if payload.get("schema") != INPUT_AUDIT_SCHEMA:
        # v1 records were written against the clinical-FL premise that D-039
        # withdrew, including the `complete: false` partial audits. They are
        # refused rather than reinterpreted under the new spec.
        raise SplitContractError(
            f"input audit schema is {payload.get('schema')!r}, expected {INPUT_AUDIT_SCHEMA!r}. "
            "A record written under the previous specification is not accepted; re-run S0-1."
        )
    expected = payload.get("expected_sha256")
    if not isinstance(expected, dict) or not expected:
        raise SplitContractError(f"input audit record holds no expected_sha256 map: {audit_path}")
    return {str(key): str(value).lower() for key, value in expected.items()}


def verify_bootstrap_inputs(
    inputs: dict[str, str | Path], *, expected_sha256: dict[str, str]
) -> dict[str, str]:
    """Check each input against the hash the S0-1 audit fixed for it.

    `expected_sha256` must come from the audit record, never from the inputs
    themselves. Bootstrap reads only what was audited; it is not a mode in
    which arbitrary inputs become readable.
    """
    if not expected_sha256:
        raise SplitContractError(
            "no audited input hashes were supplied, so nothing can be verified. Pass the S0-1 "
            "input audit record; recomputing the hashes here would only compare them with "
            "themselves."
        )
    missing_expectation = sorted(set(inputs) - set(expected_sha256))
    if missing_expectation:
        raise SplitContractError(
            f"the input audit record fixes no hash for {missing_expectation}; an unaudited input "
            "is not read"
        )
    observed: dict[str, str] = {}
    for label, raw in inputs.items():
        path = Path(raw)
        if not path.is_file():
            raise SplitContractError(f"bootstrap input {label!r} not found: {path}")
        actual = file_sha256(path)
        if actual != expected_sha256[label]:
            raise SplitContractError(
                f"bootstrap input {label!r} does not match its audited hash\n"
                f"  expected (audit record): {expected_sha256[label]}\n"
                f"  actual   (on disk)     : {actual}"
            )
        observed[label] = actual
    return observed


def verify_intermediate_fingerprint(path: str | Path, expected_sha256: str | None) -> str:
    """Check the dry run's own output file against the hash fixed at that time.

    Checking only the JSON's declared contents would miss an intermediate whose
    GT classification or FL groups were altered while its input list, candidate
    set and seed stayed the same (report 11.7.3).
    """
    if not expected_sha256:
        raise SplitContractError(
            "no expected fingerprint was given for the stratification intermediate. The value "
            "printed by --dry_run_stratification must be supplied here; a file cannot vouch for "
            "itself."
        )
    intermediate = Path(path)
    if not intermediate.is_file():
        raise SplitContractError(f"stratification intermediate not found: {intermediate}")
    actual = file_sha256(intermediate)
    if actual != str(expected_sha256).lower():
        raise SplitContractError(
            "the stratification intermediate does not match the fingerprint fixed at the dry run\n"
            f"  expected: {expected_sha256}\n  actual  : {actual}\n"
            "  Its GT classification, FL groups or specification changed; it is not used."
        )
    return actual


def load_bootstrap_state(path: str | Path) -> dict:
    state_path = Path(path)
    if not state_path.is_file():
        return {}
    payload = _read_json(state_path, label="bootstrap state")
    if payload.get("schema") != BOOTSTRAP_STATE_SCHEMA:
        raise SplitContractError(
            f"bootstrap state schema is {payload.get('schema')!r}, expected {BOOTSTRAP_STATE_SCHEMA!r}"
        )
    return payload


def assert_resume_consistent(state: dict, observed_inputs: dict[str, str]) -> None:
    """A resumed build must be the same build, on the same inputs."""
    if not state:
        return
    recorded = dict(state.get("input_sha256") or {})
    if recorded and recorded != observed_inputs:
        changed = sorted(
            label for label in set(recorded) | set(observed_inputs)
            if recorded.get(label) != observed_inputs.get(label)
        )
        raise SplitContractError(
            f"resuming with different inputs than the interrupted build: {changed}. "
            "Restart from a clean state rather than mixing two input sets."
        )


def write_legacy_manifest(
    *,
    out_path: str | Path,
    name: str,
    train_list: str | Path,
    val_list: str | Path,
) -> Path:
    """A manifest that reproduces an earlier condition and nothing more.

    It cannot enable new training, cannot enable directory mode, and -- because
    the registry is consulted first and independently -- cannot reach a sealed
    video even though the old train list nominally contains those videos.
    """
    from stage5.utils.file_list_mode import (
        list_content_sha256,
        list_identity_sha256,
        read_file_list,
    )

    train_paths = read_file_list(train_list, label="legacy train")
    val_paths = read_file_list(val_list, label="legacy val")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "name": name,
        "purpose": "reproduce a pre-S5-16 run condition; not usable for new training",
        "lists": {
            "train_core": {
                "path": str(train_list),
                "count": len(train_paths),
                "content_sha256": list_content_sha256(train_paths),
                "identity_sha256": list_identity_sha256(train_paths),
            },
            "validation": {
                "path": str(val_list),
                "count": len(val_paths),
                "content_sha256": list_content_sha256(val_paths),
                "identity_sha256": list_identity_sha256(val_paths),
            },
        },
        # This manifest defines NO sealed split of its own. The old train list
        # names 162 videos, 18 of which are now sealed; calling that list "the
        # sealed split" would conflate the two. What it does declare is that
        # the current registry applies to it in full.
        "sealed_splits": [],
        "seal_registry_required": True,
        "allows_new_training": False,
        "allows_directory_mode": False,
        "legacy_note": (
            "The old train list still names videos that are now sealed. This manifest does not "
            "unseal them and does not claim to hold them: the current seal registry is consulted "
            "independently and decides every deny, so any target intersecting it stops -- for "
            "evaluation, inference and checkers as well as training. Full reproduction of the old "
            "condition is given up rather than lifting the seal."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI (called from bash before any preflight)
# ---------------------------------------------------------------------------


def _cmd_assert(args: argparse.Namespace) -> None:
    contract = resolve_active_contract(
        manifest_key=args.split_manifest,
        pins_path=args.pins,
        coverage_path=args.coverage,
    )
    paths: list[str] = list(args.path or [])
    for list_file in args.paths_from or []:
        from stage5.utils.file_list_mode import read_file_list

        paths.extend(str(entry) for entry in read_file_list(list_file, label="guarded"))
    contract.assert_paths_allowed(paths, purpose=args.purpose)
    if args.require_new_training:
        contract.assert_new_training_allowed()
    if args.require_directory_mode:
        contract.assert_directory_mode_allowed()
    # Counts and contract hashes only: observed statistics never reach stdout.
    print(
        "Split contract satisfied: "
        f"manifest={contract.manifest.name} "
        f"manifest_sha256={contract.manifest.sha256[:16]} "
        f"registry_sha256={contract.registry.sha256[:16]} "
        f"checked_inputs={len(paths)} "
        f"sealed_videos={len(contract.registry.sealed_video_identities)}"
    )


def _cmd_emit_legacy(args: argparse.Namespace) -> None:
    out = write_legacy_manifest(
        out_path=args.out,
        name=args.name,
        train_list=args.train_list,
        val_list=args.val_list,
    )
    print(f"Legacy manifest written: {out}")
    print(f"  sha256: {file_sha256(out)}")
    print("  Add this path and hash to stage5/config/split_contract_pins.json, review the git diff,")
    print("  and commit it. Until the pin exists, this manifest cannot be selected.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage5 B' seal contract: refuse sealed videos before any input is opened."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("assert", help="Verify inputs against the contract, then exit 0 or 2")
    check.add_argument("--split_manifest", default=None, help="Approved pin name (not a path)")
    check.add_argument("--pins", default=None)
    check.add_argument("--coverage", default=None, help="Artifact coverage JSON for unnamed artifacts")
    check.add_argument("--path", action="append", default=None)
    check.add_argument("--paths_from", action="append", default=None, dest="paths_from")
    check.add_argument("--purpose", default="guarded read")
    check.add_argument("--require_new_training", action="store_true")
    check.add_argument("--require_directory_mode", action="store_true")
    check.set_defaults(func=_cmd_assert)

    legacy = sub.add_parser("emit-legacy-manifest", help="Write a reproduce-only manifest")
    legacy.add_argument("--train_list", required=True)
    legacy.add_argument("--val_list", required=True)
    legacy.add_argument("--out", required=True)
    legacy.add_argument("--name", default="legacy_s5_15")
    legacy.set_defaults(func=_cmd_emit_legacy)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        args.func(args)
    except (SplitContractError, IdentityError) as error:
        print(f"Split contract refused the run: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# split partition checks (report 8.3-4)
# ---------------------------------------------------------------------------


def verify_split_partition(
    *,
    train_core: Sequence[str | Path],
    validation: Sequence[str | Path],
    internal_test: Sequence[str | Path],
    old_train: Sequence[str | Path],
    old_validation: Sequence[str | Path],
    sanity: Sequence[str | Path],
    expected_counts: tuple[int, int, int] = (144, 18, 18),
) -> dict:
    """The six checks the new split must pass before it is confirmed.

    Comparison is by video identity, so the same video under a different mount
    prefix still counts as the same video; the path spelling is reported
    separately rather than the comparison being loosened until it passes
    (`compare_file_lists`, file_list_mode).

    Returns a summary of hashes and counts. Raises on the first failure: a
    partition that fails any of these is not written.
    """
    from stage5.utils.file_list_mode import list_content_sha256, list_identity_sha256

    def identities(paths: Sequence[str | Path]) -> list[str]:
        return [Path(p).name for p in paths]

    core, val, test = identities(train_core), identities(validation), identities(internal_test)
    old_core, old_val, sanity_names = identities(old_train), identities(old_validation), identities(sanity)

    for label, names in (("train_core", core), ("validation", val), ("internal_test", test)):
        if len(set(names)) != len(names):
            raise SplitContractError(f"{label} contains the same video more than once")

    # 1. mutually disjoint
    for left, right, label in (
        (core, val, "train_core/validation"),
        (core, test, "train_core/internal_test"),
        (val, test, "validation/internal_test"),
    ):
        shared = set(left) & set(right)
        if shared:
            raise SplitContractError(f"{label} overlap in {len(shared)} video(s)")

    # 2. counts
    counts = (len(core), len(val), len(test))
    if counts != tuple(expected_counts):
        raise SplitContractError(
            f"split sizes are {counts}, expected {tuple(expected_counts)}"
        )

    # 3. train_core + internal_test reconstructs the old train list exactly
    if set(core) | set(test) != set(old_core):
        raise SplitContractError(
            "train_core + internal_test does not reconstruct the old train list"
        )

    # 4. the union is the old 180 videos
    if set(core) | set(test) | set(val) != set(old_core) | set(old_val):
        raise SplitContractError("the three splits do not cover exactly the old video set")

    # 5. every train-sanity video stays in train_core
    stray = set(sanity_names) - set(core)
    if stray:
        raise SplitContractError(f"{len(stray)} train-sanity video(s) did not stay in train_core")

    # 6. validation is untouched, in content AND in order
    if val != old_val:
        raise SplitContractError(
            "validation changed: its contents or its order differ from the saved list"
        )

    return {
        "counts": {"train_core": counts[0], "validation": counts[1], "internal_test": counts[2]},
        "identity_sha256": {
            "train_core": list_identity_sha256(list(train_core)),
            "validation": list_identity_sha256(list(validation)),
            "internal_test": list_identity_sha256(list(internal_test)),
        },
        "content_sha256": {
            "train_core": list_content_sha256(list(train_core)),
            "validation": list_content_sha256(list(validation)),
            "internal_test": list_content_sha256(list(internal_test)),
        },
        "validation_representation_identical": [str(p) for p in validation] == [str(p) for p in old_validation],
        "all_checks_passed": True,
    }
