from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

# ----------------------------------------------------------------------------
# S5-16 Step 0: the deterministic half of the B' split.
#
# Report 9.2, decision D-039. Everything here is fixed BEFORE any result is
# seen: the stratification quantity, the rank split into three groups, the
# largest-remainder allocation of the 18 internal_test videos across strata,
# and the within-stratum draw.
#
# The second stratification axis is NOT a clinical FL tertile. The 180
# annotated videos have no clinical measurement; the FL figures that exist are
# derived from the BBox annotations, so stratifying on them would have meant
# stratifying on a second derivation of the same labels under a misleading
# name. D-039 replaced that axis with `gt_positive_frame_count`.
#
# What that quantity is, and is not:
#   * it is the number of distinct frames holding at least one valid
#     GT-positive point, in the SAVED teacher point cloud
#   * it is NOT a femur length, NOT a size, and NOT an observation time --
#     fps and stride are unconfirmed across the 159 candidates, so a slow
#     sweep over a short femur and a fast sweep over a long one are not
#     distinguished
#   * it balances only the draw of internal_test 18 out of the 159 candidates.
#     validation 18 is fixed and not stratified, and train_core 144 carries
#     the fixed sanity 3, so this is not a balance "across the three splits"
#
# Two further choices worth stating plainly:
#
#   * Anything ambiguous stops. A value that cannot be computed is never
#     folded into a zero or a missing group; a genuine zero is a valid value
#     and is ranked with the rest.
#
#   * The draw is a SHA-256 keyed sort, not an RNG call -- an implementation
#     choice made to reduce library- and language-version dependence, taken
#     before any real data was split, and not a finding that an RNG draw would
#     have been invalid. See SELECTION_METHOD.
#
# Deliberately stdlib-only: no numpy, no h5py. The whole module is exercised by
# checks/dummy/check_dummy_bprime_split_allocation.py without real data.
# ----------------------------------------------------------------------------

# Fixed and recorded in the split manifest so the draw is reproducible from a
# written specification rather than from this file's current contents.
SELECTION_METHOD = {
    "name": "sha256_keyed_sort_v1",
    "key": "sha256(f'{seed}|{video_identity}').hexdigest()",
    "encoding": "utf-8",
    "separator": "|",
    "seed_serialization": "decimal integer, str(seed)",
    "order": "key ascending; equal keys ordered by video identity ascending",
    "take": "the first `allocation` entries of that order",
    "output_order": "returned in candidate input order, not in key order",
    "note": (
        "An implementation choice for version independence, fixed before any real split. "
        "Not equivalent to, and not compared against, an RNG-based draw."
    ),
}

STRATUM_GROUP_LOW = 0
STRATUM_GROUP_MID = 1
STRATUM_GROUP_HIGH = 2
STRATUM_GROUP_NAMES = {
    STRATUM_GROUP_LOW: "gtframes_low",
    STRATUM_GROUP_MID: "gtframes_mid",
    STRATUM_GROUP_HIGH: "gtframes_high",
}

# Recorded in the sealed intermediate and in the split manifest, and verified
# at confirm time. The implementation refuses a version it does not know, even
# when the corresponding CLI argument is omitted: omitting an argument is not a
# reason to accept an unknown version (report 4.1.1, correction 14).
STRATIFICATION_QUANTITY = {
    "name": "gt_positive_frame_count",
    "version": "gt_positive_frame_count_v1",
    "definition": (
        "number of distinct frame_order values holding at least one point with "
        "valid_mask AND point_label == 1, in the saved teacher point cloud"
    ),
    "unit": "frames (count)",
    "aggregation": "none; the quantity is per-video by construction",
    "reflects": "the teacher state AFTER xml/crop-quality invalidation",
    "is_not": (
        "a femur length, a physical size, an observation time, or a candidate "
        "surrogate FL definition for S5-17"
    ),
}

STRATIFICATION_CONDITION = {
    "name": "strictly_increasing_group_medians",
    "version": "strictly_increasing_group_medians_v1",
    "rule": "median(G1) < median(G2) < median(G3)",
    "purpose": "stop a degenerate three-way split before it is used",
    "does_not_guarantee": (
        "that the groups' value ranges do not overlap, nor that the "
        "stratification is effective"
    ),
}

SUPPORTED_QUANTITY_VERSIONS = (STRATIFICATION_QUANTITY["version"],)
SUPPORTED_CONDITION_VERSIONS = (STRATIFICATION_CONDITION["version"],)


class SplitSpecError(ValueError):
    """Any input condition that must stop the split instead of being absorbed."""


# ---------------------------------------------------------------------------
# clinical FL intake
# ---------------------------------------------------------------------------
# stratification quantity
# ---------------------------------------------------------------------------


def stratification_group_medians(
    values: Mapping[str, int], groups: Mapping[str, int]
) -> tuple[float, float, float]:
    """Median of each of the three groups, low to high."""
    medians: list[float] = []
    for group in (STRATUM_GROUP_LOW, STRATUM_GROUP_MID, STRATUM_GROUP_HIGH):
        members = sorted(values[identity] for identity, g in groups.items() if g == group)
        if not members:
            raise SplitSpecError(f"stratification group {group} is empty")
        middle = len(members) // 2
        medians.append(
            float(members[middle])
            if len(members) % 2
            else (members[middle - 1] + members[middle]) / 2.0
        )
    return medians[0], medians[1], medians[2]


def assert_stratification_is_constructible(
    values: Mapping[str, int], groups: Mapping[str, int]
) -> dict:
    """Stop a degenerate three-way split before it is used.

    The rank split will happily produce three groups even when every value is
    identical, so "three groups received members" is not the check. The check
    is that the groups differ IN THE QUANTITY: strictly increasing medians.

    This is a pre-fixed rule against degeneracy. It does NOT guarantee that the
    groups' value ranges are disjoint -- a tie straddling a boundary still
    splits across groups -- and it does not establish that the stratification
    is effective (report 3.3).
    """
    low, mid, high = stratification_group_medians(values, groups)
    constructible = low < mid < high
    if not constructible:
        raise SplitSpecError(
            "the three stratification groups do not differ in the quantity "
            f"(medians {low}, {mid}, {high} are not strictly increasing). The split stops for "
            "re-judgement; it does not fall back to another stratification."
        )
    return {
        "condition": STRATIFICATION_CONDITION["name"],
        "condition_version": STRATIFICATION_CONDITION["version"],
        "medians": [low, mid, high],
        "constructible": True,
    }


def assert_supported_versions(quantity_version: str, condition_version: str) -> None:
    """Refuse a spec version this implementation does not know.

    Called at confirm time even when the corresponding CLI arguments were
    omitted: leaving an argument out is not a reason to accept an unknown
    version recorded in the intermediate (report 4.1.1, correction 14).
    """
    if quantity_version not in SUPPORTED_QUANTITY_VERSIONS:
        raise SplitSpecError(
            f"stratification quantity version {quantity_version!r} is not supported by this "
            f"implementation (supported: {list(SUPPORTED_QUANTITY_VERSIONS)})"
        )
    if condition_version not in SUPPORTED_CONDITION_VERSIONS:
        raise SplitSpecError(
            f"stratification condition version {condition_version!r} is not supported by this "
            f"implementation (supported: {list(SUPPORTED_CONDITION_VERSIONS)})"
        )


def assign_stratification_groups(
    values: Mapping[str, int], *, order_index: Mapping[str, int]
) -> dict[str, int]:
    """Rank the candidates by the quantity and cut them into three groups.

    Equal values are ordered by the video's position in the saved train162
    list, so a tie is resolved by a fixed input order and never by chance. The
    quantity is an integer, so ties are expected to be common; equal values CAN
    land in different groups when a tie straddles a boundary, and the number of
    candidates sitting in such a block is recorded rather than hidden.
    """
    missing_order = [identity for identity in values if identity not in order_index]
    if missing_order:
        raise SplitSpecError(f"{len(missing_order)} value(s) have no position in the input order")
    for identity, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise SplitSpecError("stratification values must be plain integers")
        if value < 0:
            raise SplitSpecError("a stratification value is negative")

    ordered = sorted(values, key=lambda identity: (values[identity], order_index[identity]))
    low, mid, _high = tertile_group_sizes(len(ordered))

    groups: dict[str, int] = {}
    for position, identity in enumerate(ordered):
        if position < low:
            groups[identity] = STRATUM_GROUP_LOW
        elif position < low + mid:
            groups[identity] = STRATUM_GROUP_MID
        else:
            groups[identity] = STRATUM_GROUP_HIGH
    return groups


def count_boundary_tie_members(
    values: Mapping[str, int], groups: Mapping[str, int]
) -> int:
    """Candidates whose value also occurs in a neighbouring group.

    Recorded (sealed) so that the limit of the median condition is visible:
    strictly increasing medians do not mean the groups are disjoint in value.
    """
    by_group: dict[int, set[int]] = {}
    for identity, group in groups.items():
        by_group.setdefault(group, set()).add(values[identity])
    shared: set[int] = set()
    for left, right in ((STRATUM_GROUP_LOW, STRATUM_GROUP_MID),
                        (STRATUM_GROUP_MID, STRATUM_GROUP_HIGH)):
        shared |= by_group.get(left, set()) & by_group.get(right, set())
    return sum(1 for identity in groups if values[identity] in shared)




# ---------------------------------------------------------------------------
# rank split into three groups
# ---------------------------------------------------------------------------


def tertile_group_sizes(num_candidates: int) -> tuple[int, int, int]:
    """Sizes low, mid, high. The remainder goes to the LOWER groups first.

    Fixed before any data is seen so the rule cannot be chosen to suit a result.
    """
    if num_candidates < 3:
        raise SplitSpecError(
            f"only {num_candidates} candidate(s); three groups cannot be formed. This stops "
            "rather than falling back to another stratification"
        )
    base, remainder = divmod(num_candidates, 3)
    sizes = [base + 1] * remainder + [base] * (3 - remainder)
    if min(sizes) < 1:
        raise SplitSpecError(f"a clinical FL group would be empty (sizes={sizes}); the split stops")
    return sizes[0], sizes[1], sizes[2]




# ---------------------------------------------------------------------------
# strata and allocation
# ---------------------------------------------------------------------------


def build_strata(
    candidate_identities: Sequence[str],
    *,
    multi_region: Mapping[str, bool],
    stratum_group: Mapping[str, int],
) -> dict[tuple[int, int], list[str]]:
    """Cells keyed (multi_region, stratum_group), members in input order."""
    cells: dict[tuple[int, int], list[str]] = {}
    for identity in candidate_identities:
        if identity not in multi_region:
            raise SplitSpecError("a candidate has no multi-region classification")
        if identity not in stratum_group:
            raise SplitSpecError("a candidate has no stratification group")
        key = (int(bool(multi_region[identity])), int(stratum_group[identity]))
        cells.setdefault(key, []).append(identity)
    return cells


def allocate_largest_remainder(
    cells: Mapping[tuple[int, int], Sequence[str]],
    *,
    total: int,
) -> dict[tuple[int, int], int]:
    """Spread `total` picks across cells proportionally, with no randomness.

    Integer parts first, then the remaining seats by largest fractional part;
    ties on the fraction go to the lower cell key. Capacity is never exceeded,
    and seats freed by a capped cell are redistributed in the same fixed order.
    """
    population = sum(len(members) for members in cells.values())
    if total > population:
        raise SplitSpecError(f"cannot allocate {total} videos from a population of {population}")

    quotas = {key: total * len(members) / population for key, members in cells.items()}
    alloc = {key: min(int(quotas[key]), len(cells[key])) for key in cells}

    remaining = total - sum(alloc.values())
    by_fraction = sorted(cells, key=lambda key: (-(quotas[key] - int(quotas[key])), key))
    for key in by_fraction:
        if remaining <= 0:
            break
        if alloc[key] < len(cells[key]):
            alloc[key] += 1
            remaining -= 1

    # Any seat still unplaced went to a cell already at capacity; hand it on in
    # the same fixed cell order rather than leaving the allocation short.
    while remaining > 0:
        placed = False
        for key in sorted(cells):
            if alloc[key] < len(cells[key]):
                alloc[key] += 1
                remaining -= 1
                placed = True
                if remaining <= 0:
                    break
        if not placed:
            raise SplitSpecError(
                f"{remaining} video(s) could not be allocated: every stratum is at capacity"
            )
    return alloc


def selection_key(identity: str, *, seed: int) -> str:
    """Per-video sort key: sha256 of "<seed>|<identity>", UTF-8, hex digest.

    Spelled out in SELECTION_METHOD and written into the split manifest, so the
    draw can be reproduced from the recorded specification rather than from
    whatever this file happens to contain later.
    """
    return hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).hexdigest()


def select_within_cell(members: Sequence[str], count: int, *, seed: int) -> list[str]:
    """The `count` members with the lowest derived key, returned in input order."""
    if count > len(members):
        raise SplitSpecError(f"cannot select {count} from a stratum of {len(members)}")
    if count == 0:
        return []
    ranked = sorted(members, key=lambda identity: (selection_key(identity, seed=seed), identity))
    chosen = set(ranked[:count])
    return [identity for identity in members if identity in chosen]


def draw_internal_test(
    candidate_identities: Sequence[str],
    *,
    multi_region: Mapping[str, bool],
    stratum_group: Mapping[str, int],
    total: int,
    seed: int,
) -> tuple[list[str], dict[tuple[int, int], int]]:
    """The whole draw: strata -> allocation -> per-stratum selection.

    Returns the picks in candidate input order, plus the per-cell allocation.
    The allocation is a sealed quantity: publishing it alongside stratum sizes
    would reconstruct internal_test's composition (report 4.4).
    """
    cells = build_strata(
        candidate_identities, multi_region=multi_region, stratum_group=stratum_group
    )
    alloc = allocate_largest_remainder(cells, total=total)
    picked: set[str] = set()
    for key in sorted(cells):
        picked.update(select_within_cell(cells[key], alloc[key], seed=seed))
    if len(picked) != total:
        raise SplitSpecError(f"selected {len(picked)} videos, expected {total}")
    return [identity for identity in candidate_identities if identity in picked], alloc
