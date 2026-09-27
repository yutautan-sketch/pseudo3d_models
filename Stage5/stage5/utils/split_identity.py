from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# S5-16 Step 0: resolving which videos an artifact belongs to.
#
# The seal contract is enforced per VIDEO, not per path: the same video appears
# as a teacher H5, an intermediate pseudo-3D H5, a prediction NPZ, a GT-region
# JSON and a row in several CSVs, and a guard that only matched one spelling of
# one of those would not be a guard at all.
#
# Report section 10.10.2: a file name that does not match the video-ID pattern
# is NOT evidence that the file is free of sealed videos. Shared-name JSON/CSV
# outputs, aggregates over many videos and alias-only artifacts cannot be
# judged from the name. Those must be covered by a verified coverage record
# bound to the artifact's own content hash, or the read stops. "No identity
# found" is never an allow.
#
# Deliberately stdlib-only: this runs before numpy/h5py/torch are imported, and
# before any H5 is opened.
# ----------------------------------------------------------------------------

# The 180-video teacher set uses TWO naming conventions, confirmed against the
# saved lists on 2026-09-27:
#
#   timestamp  YYYYMMDD_HHMMSS_N[_NN]   e.g. 20250403_103433_091
#                                             20250403_103433_091_02
#   case       D-D_NN[_NN]              e.g. 1-2_34_56, 1-2_34
#
# The case form was missed initially because every other Stage 5 tool matches
# the timestamp form only. A guard that recognised just that form would fail to
# identify 10 of the 180 videos, which is exactly the set the seal has to be
# able to name.
#
# The timestamp form's OPTIONAL FOURTH segment was missed too, and mattered
# more: the three-segment pattern every existing tool uses stops before it, so
# `..._091_01`, `..._091_02` and `..._091_03` all resolve to `..._091`. In the
# saved train list that collapses 14 groups of files onto 14 identities --
# different files sharing one identity, which is the failure a seal contract
# cannot tolerate. All trailing numeric segments are therefore part of the
# identity.
#
# NOTE (open question, report 21): whether a trailing `_NN` marks an
# INDEPENDENT video or a clip of one acquisition is not established here. This
# module only guarantees that each FILE gets one unambiguous identity. If the
# segments turn out to be clips of a single acquisition, the unit the split
# draws on is a separate decision and is not made here.
#
# The two alternatives cannot cross-match on the names that exist: the
# timestamp form contains no hyphen, and no name carries both forms. The
# surrounding lookarounds stop a longer digit run from yielding a shifted
# match.
#
# The case form's trailing groups are `_[0-9]+`, deliberately NOT length
# limited. An earlier draft capped them at two digits, which silently
# TRUNCATED: `1-2_34_567` and `1-2_34_891` both collapsed to `1-2_34`, giving
# two different videos one identity. For a seal contract that is the worst
# possible failure -- it would let one video stand in for another -- so the
# groups match whatever digits are there and a collision cannot be introduced
# by the pattern itself.
#
# This is an ENUMERATED set of known conventions, not a permissive pattern: a
# name matching neither is still refused, with a masked shape in the message so
# a new convention can be recognised rather than guessed at.
NAMING_CONVENTIONS = (
    ("timestamp", r"[0-9]{8}_[0-9]{6}_[0-9]+(?:_[0-9]+)*"),
    ("case", r"[0-9]+-[0-9]+(?:_[0-9]+)+"),
)

VIDEO_ID_PATTERN = re.compile(
    r"(?<![0-9])(" + "|".join(pattern for _name, pattern in NAMING_CONVENTIONS) + r")(?![0-9])"
)

COVERAGE_SCHEMA = "stage5_artifact_coverage_v1"


class IdentityError(ValueError):
    """Raised when an artifact's video membership cannot be established."""


def extract_video_identities(name: str) -> tuple[str, ...]:
    """Video identities readable from a file name, in order of appearance.

    Returns an empty tuple when the name carries none; the caller must then
    fall back to a coverage record rather than treating the file as unrelated.
    """
    seen: list[str] = []
    for match in VIDEO_ID_PATTERN.finditer(str(name)):
        identity = match.group(1)
        if identity not in seen:
            seen.append(identity)
    return tuple(seen)


def contains_video_identity(text: str) -> bool:
    """Whether any known video identity appears anywhere in `text`.

    Used by the privacy assertions. It covers BOTH naming conventions, which a
    timestamp-only search does not -- a check written against one convention
    would quietly pass a leaked name from the other.
    """
    return bool(VIDEO_ID_PATTERN.search(str(text)))


def masked_shape(name: str) -> str:
    """A name's character shape, with the characters themselves removed.

    Digits become '#' and letters 'a'; separators and the extension survive.
    This makes a naming mismatch diagnosable ("########_######_#### ..." vs
    something else) without reproducing a video identifier, so it is safe to
    put in an error message that may be pasted into a shared log.
    """
    out = []
    for char in str(name):
        if char.isdigit():
            out.append("#")
        elif char.isalpha():
            out.append("a")
        else:
            out.append(char)
    return "".join(out)


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CoverageEntry:
    artifact_name: str
    sha256: str
    video_identities: tuple[str, ...]


class ArtifactCoverage:
    """Verified artifact -> videos records, bound to each artifact's own hash.

    The hash binding is what stops one artifact's coverage record from being
    reused to vouch for a different artifact that happens to share a name.
    """

    def __init__(self, entries: tuple[CoverageEntry, ...], *, source: str | None = None) -> None:
        self._by_name: dict[str, list[CoverageEntry]] = {}
        for entry in entries:
            self._by_name.setdefault(entry.artifact_name, []).append(entry)
        self.source = source

    @classmethod
    def from_json(cls, path: str | Path) -> "ArtifactCoverage":
        coverage_path = Path(path)
        try:
            payload = json.loads(coverage_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise IdentityError(f"artifact coverage could not be read: {coverage_path} ({error})") from error
        except json.JSONDecodeError as error:
            raise IdentityError(f"artifact coverage is not valid JSON: {coverage_path} ({error})") from error
        if payload.get("schema") != COVERAGE_SCHEMA:
            raise IdentityError(
                f"artifact coverage schema is {payload.get('schema')!r}, expected {COVERAGE_SCHEMA!r}"
            )
        entries: list[CoverageEntry] = []
        for index, row in enumerate(payload.get("entries", [])):
            missing = [key for key in ("artifact_name", "sha256", "video_identities") if key not in row]
            if missing:
                raise IdentityError(f"artifact coverage entry {index} lacks {missing}")
            entries.append(
                CoverageEntry(
                    artifact_name=str(row["artifact_name"]),
                    sha256=str(row["sha256"]).lower(),
                    video_identities=tuple(str(item) for item in row["video_identities"]),
                )
            )
        return cls(tuple(entries), source=str(coverage_path))

    @classmethod
    def empty(cls) -> "ArtifactCoverage":
        return cls(())

    def lookup(self, path: Path) -> tuple[str, ...]:
        """Identities for `path`, verified against its content hash.

        A name match with a hash mismatch raises rather than falling through to
        "unknown": a changed artifact under a known name is exactly the case
        where a stale coverage record would wave the wrong file through.
        """
        candidates = self._by_name.get(path.name)
        if not candidates:
            raise IdentityError(
                f"no coverage record names {path.name!r}; its video membership cannot be established"
            )
        if not path.is_file():
            raise IdentityError(f"coverage record exists for {path.name!r} but the file is missing: {path}")
        actual = file_sha256(path)
        for entry in candidates:
            if entry.sha256 == actual:
                if not entry.video_identities:
                    raise IdentityError(
                        f"coverage record for {path.name!r} lists no videos; an empty record is not a "
                        "statement that the artifact is free of sealed videos"
                    )
                return entry.video_identities
        raise IdentityError(
            f"coverage record for {path.name!r} does not match the file's content hash "
            f"(actual {actual[:16]}...); a coverage record from another artifact is not reused"
        )


def resolve_identities(path: str | Path, *, coverage: ArtifactCoverage | None = None) -> tuple[str, ...]:
    """Every video `path` may contain. Raises instead of returning nothing.

    Name-derived identities are accepted directly; anything else must be
    covered by a hash-verified record.
    """
    artifact = Path(path)
    from_name = extract_video_identities(artifact.name)
    if from_name:
        return from_name
    if coverage is None:
        raise IdentityError(
            f"{artifact.name!r} carries no video ID and no artifact coverage was supplied; "
            "an unidentifiable artifact is refused rather than allowed"
        )
    return coverage.lookup(artifact)
