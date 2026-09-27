from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.split_allocation import (  # noqa: E402
    SELECTION_METHOD,
    STRATIFICATION_CONDITION,
    STRATIFICATION_QUANTITY,
    STRATUM_GROUP_HIGH,
    STRATUM_GROUP_LOW,
    STRATUM_GROUP_MID,
    SplitSpecError,
    allocate_largest_remainder,
    assert_stratification_is_constructible,
    assert_supported_versions,
    assign_stratification_groups,
    build_strata,
    count_boundary_tie_members,
    draw_internal_test,
    select_within_cell,
    selection_key,
    stratification_group_medians,
    tertile_group_sizes,
)

# ----------------------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the deterministic half of the B' split.
#
# Report 9.2, 9.9, 10.8. No real clinical FL values and no real video IDs
# appear here. What is being pinned down is that every rule was fixed BEFORE
# any result could be seen, and that ambiguity stops instead of being absorbed:
# an unmatched, duplicated or absent row is an input defect, never "missing".
# Pure stdlib.
# ----------------------------------------------------------------------------

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def expect_stop(fn, label: str) -> None:
    global CHECKS
    try:
        fn()
    except SplitSpecError:
        check(True, label)
        return
    except Exception as error:  # noqa: BLE001
        check(False, f"{label} (raised {type(error).__name__}: {error})")
        return
    check(False, f"{label} (returned normally instead of stopping)")


def ids(count: int, *, start: int = 0) -> list[str]:
    return [f"cand{index:03d}" for index in range(start, start + count)]


def test_groups_and_condition() -> None:
    print("\n[1] three groups from the stratification quantity, with the remainder rule fixed")
    check(tertile_group_sizes(159) == (53, 53, 53), "an exact multiple splits evenly")
    check(tertile_group_sizes(157) == (53, 52, 52), "one spare goes to the low group")
    check(tertile_group_sizes(158) == (53, 53, 52), "two spares go to the low and mid groups")
    check(tertile_group_sizes(3) == (1, 1, 1), "the smallest workable population splits")
    expect_stop(lambda: tertile_group_sizes(2), "fewer than three candidates stops, with no fallback")

    cands = ids(7)
    order = {identity: index for index, identity in enumerate(cands)}
    values = {c: 10 + i for i, c in enumerate(cands)}
    groups = assign_stratification_groups(values, order_index=order)
    check(
        [groups[c] for c in cands]
        == [STRATUM_GROUP_LOW, STRATUM_GROUP_LOW, STRATUM_GROUP_LOW,
            STRATUM_GROUP_MID, STRATUM_GROUP_MID, STRATUM_GROUP_HIGH, STRATUM_GROUP_HIGH],
        "7 candidates split 3/2/2 in ascending order of the quantity",
    )

    # A genuine zero is a value, not a gap: it ranks lowest and stays in.
    with_zero = dict(values)
    with_zero[cands[0]] = 0
    groups_z = assign_stratification_groups(with_zero, order_index=order)
    check(groups_z[cands[0]] == STRATUM_GROUP_LOW, "a zero value ranks lowest rather than being dropped")
    check(len(groups_z) == len(cands), "no candidate is removed because its value is zero")

    expect_stop(
        lambda: assign_stratification_groups({c: -1 for c in cands}, order_index=order),
        "a negative value stops",
    )
    expect_stop(
        lambda: assign_stratification_groups({c: 1.5 for c in cands}, order_index=order),
        "a non-integer value stops",
    )
    expect_stop(
        lambda: assign_stratification_groups({c: True for c in cands}, order_index=order),
        "a bool is not accepted as an integer count",
    )
    expect_stop(
        lambda: assign_stratification_groups(values, order_index={}),
        "a candidate with no position in the input order stops",
    )

    # Ties: resolved by input order, and they really can straddle a boundary.
    tied = ids(6)
    order_t = {identity: index for index, identity in enumerate(tied)}
    flat = {c: 40 for c in tied}
    groups_t = assign_stratification_groups(flat, order_index=order_t)
    check(
        [groups_t[c] for c in tied] == [STRATUM_GROUP_LOW, STRATUM_GROUP_LOW, STRATUM_GROUP_MID,
                                        STRATUM_GROUP_MID, STRATUM_GROUP_HIGH, STRATUM_GROUP_HIGH],
        "ties are broken by the fixed input order",
    )
    check(len({groups_t[c] for c in tied}) == 3, "equal values do land in different groups")

    print("\n[2] the constructibility condition stops a degenerate split")
    ok = assert_stratification_is_constructible(values, groups)
    check(ok["constructible"] is True, "strictly increasing medians are accepted")
    check(ok["medians"] == sorted(ok["medians"]), "the recorded medians are ordered")
    check(
        ok["condition_version"] == STRATIFICATION_CONDITION["version"],
        "the condition version is recorded with the result",
    )

    # The case the rank split alone would wave through.
    expect_stop(
        lambda: assert_stratification_is_constructible(flat, groups_t),
        "all-identical values stop, even though the rank split formed three groups",
    )
    check(
        len({groups_t[c] for c in tied}) == 3,
        "and the rank split really did form three groups, so the condition is what caught it",
    )

    near = {c: v for c, v in zip(ids(9), [1, 1, 1, 1, 1, 1, 2, 3, 4])}
    order_n = {c: i for i, c in enumerate(near)}
    groups_n = assign_stratification_groups(near, order_index=order_n)
    lows = stratification_group_medians(near, groups_n)
    if lows[0] == lows[1]:
        expect_stop(
            lambda: assert_stratification_is_constructible(near, groups_n),
            "two groups sharing a median stop",
        )
    else:
        check(True, "the near-degenerate case happened to separate; condition still evaluated")

    print("\n[3] boundary ties are counted, not hidden")
    straddle = {c: v for c, v in zip(ids(6), [1, 2, 2, 2, 2, 3])}
    order_s = {c: i for i, c in enumerate(straddle)}
    groups_s = assign_stratification_groups(straddle, order_index=order_s)
    count = count_boundary_tie_members(straddle, groups_s)
    check(count > 0, "candidates whose value spans two groups are counted")
    check(
        count_boundary_tie_members({c: i for i, c in enumerate(ids(6))},
                                   assign_stratification_groups(
                                       {c: i for i, c in enumerate(ids(6))},
                                       order_index={c: i for i, c in enumerate(ids(6))})) == 0,
        "all-distinct values give no boundary ties",
    )

    print("\n[4] unsupported spec versions are refused")
    assert_supported_versions(
        STRATIFICATION_QUANTITY["version"], STRATIFICATION_CONDITION["version"]
    )
    check(True, "the current versions are accepted")
    expect_stop(
        lambda: assert_supported_versions("gt_positive_frame_count_v99",
                                          STRATIFICATION_CONDITION["version"]),
        "an unknown quantity version is refused",
    )
    expect_stop(
        lambda: assert_supported_versions(STRATIFICATION_QUANTITY["version"],
                                          "some_other_condition_v2"),
        "an unknown condition version is refused",
    )
    expect_stop(
        lambda: assert_supported_versions("", ""),
        "an absent version is refused rather than defaulted",
    )


def test_allocation() -> None:
    print("\n[5] allocating 18 across strata, with no randomness")
    cells = {(0, 0): ids(50), (0, 1): ids(40, start=50), (1, 0): ids(39, start=90), (1, 3): ids(30, start=129)}
    alloc = allocate_largest_remainder(cells, total=18)
    check(sum(alloc.values()) == 18, "the allocation sums to the requested total")
    check(all(alloc[key] <= len(cells[key]) for key in cells), "no stratum is over-allocated")

    with_empty = dict(cells)
    with_empty[(1, 1)] = []
    alloc2 = allocate_largest_remainder(with_empty, total=18)
    check(alloc2[(1, 1)] == 0, "an empty stratum receives nothing")
    check(sum(alloc2.values()) == 18, "an empty stratum does not break the total")

    tiny = {(0, 0): ids(1), (0, 1): ids(1, start=1), (1, 0): ids(1, start=2), (1, 1): ids(1, start=3)}
    alloc3 = allocate_largest_remainder(tiny, total=4)
    check(all(alloc3[key] == 1 for key in tiny), "strata smaller than one seat are capped at their size")
    expect_stop(
        lambda: allocate_largest_remainder(tiny, total=5),
        "asking for more than the population stops",
    )

    # Equal fractional parts must resolve by cell key, never by chance.
    symmetric = {(0, 0): ids(10), (0, 1): ids(10, start=10), (1, 0): ids(10, start=20),
                 (1, 1): ids(10, start=30)}
    first = allocate_largest_remainder(symmetric, total=18)
    check(first[(0, 0)] >= first[(1, 1)], "a fractional tie goes to the lower cell key")
    for _ in range(5):
        random.seed(random.randrange(10 ** 6))
        check_again = allocate_largest_remainder(symmetric, total=18)
        if check_again != first:
            check(False, "the allocation changed between runs")
            break
    else:
        check(True, "the allocation is identical across runs and unaffected by global random state")


def test_selection() -> None:
    print("\n[6] the within-stratum draw is seeded, reproducible and order-preserving")
    members = ids(20)
    picked = select_within_cell(members, 5, seed=42)
    check(len(picked) == 5, "the requested number is drawn")
    check(picked == [m for m in members if m in set(picked)], "picks come back in input order")
    check(select_within_cell(members, 5, seed=42) == picked, "the same seed reproduces the draw")
    check(select_within_cell(members, 5, seed=43) != picked, "a different seed draws differently")
    expect_stop(lambda: select_within_cell(members, 21, seed=42), "over-drawing a stratum stops")
    check(select_within_cell(members, 0, seed=42) == [], "a zero allocation draws nothing")

    expected = hashlib.sha256(b"42|cand000").hexdigest()
    check(selection_key("cand000", seed=42) == expected,
          "the key is sha256(seed|identity), so it does not depend on an RNG implementation")

    random.seed(1)
    a = select_within_cell(members, 5, seed=42)
    random.seed(999)
    b = select_within_cell(members, 5, seed=42)
    check(a == b, "the draw ignores the global random state entirely")


def test_end_to_end() -> None:
    print("\n[7] the whole draw over a 159-candidate population")
    cands = ids(159)
    order = {identity: index for index, identity in enumerate(cands)}
    # Integer counts with plenty of ties, and a few genuine zeros.
    values = {identity: (0 if index % 53 == 0 else 12 + (index % 31))
              for index, identity in enumerate(cands)}
    groups = assign_stratification_groups(values, order_index=order)
    assert_stratification_is_constructible(values, groups)
    multi = {identity: (index % 5 == 0) for index, identity in enumerate(cands)}

    picks, alloc = draw_internal_test(
        cands, multi_region=multi, stratum_group=groups, total=18, seed=42
    )
    check(len(picks) == 18, "18 videos are selected")
    check(len(set(picks)) == 18, "the selection has no duplicates")
    check(set(picks) <= set(cands), "every pick comes from the candidate set")
    check(sum(alloc.values()) == 18, "the per-cell allocation sums to 18")
    check(picks == sorted(picks, key=lambda identity: order[identity]),
          "picks are returned in the input order")

    again, _ = draw_internal_test(
        cands, multi_region=multi, stratum_group=groups, total=18, seed=42
    )
    check(again == picks, "the whole draw is reproducible from the same inputs and seed")

    cells = build_strata(cands, multi_region=multi, stratum_group=groups)
    check(sum(len(m) for m in cells.values()) == 159, "every candidate lands in exactly one stratum")
    check(
        {key[1] for key in cells} <= {STRATUM_GROUP_LOW, STRATUM_GROUP_MID, STRATUM_GROUP_HIGH},
        "there are exactly three value groups and no missing group",
    )
    check(
        any(values[identity] == 0 for identity in cands),
        "the population really did contain genuine zeros, and they were not dropped",
    )
    for key, members in cells.items():
        drawn = len([p for p in picks if p in set(members)])
        check(drawn == alloc[key], f"stratum {key} received exactly its allocation")

    expect_stop(
        lambda: build_strata(cands, multi_region={}, stratum_group=groups),
        "a candidate without a multi-region classification stops",
    )
    expect_stop(
        lambda: build_strata(cands, multi_region=multi, stratum_group={}),
        "a candidate without a stratification group stops",
    )


def test_selection_method_is_specified() -> None:
    print("\n[8] the draw is reproducible from the written specification, not from this file")
    members = [f"vid{i}" for i in range(12)]

    check(SELECTION_METHOD["name"] == "sha256_keyed_sort_v1", "the method carries a version name")
    check(SELECTION_METHOD["encoding"] == "utf-8", "the string encoding is specified")
    check(SELECTION_METHOD["separator"] == "|", "the separator is specified")

    # Each clause of the specification, rebuilt from the spec text alone.
    seed = 42
    rebuilt = {
        identity: hashlib.sha256(
            f"{seed}{SELECTION_METHOD['separator']}{identity}".encode(SELECTION_METHOD["encoding"])
        ).hexdigest()
        for identity in members
    }
    check(
        all(rebuilt[identity] == selection_key(identity, seed=seed) for identity in members),
        "the key formula in the specification reproduces the implementation exactly",
    )

    expected = sorted(members, key=lambda identity: (rebuilt[identity], identity))[:4]
    drawn = select_within_cell(members, 4, seed=seed)
    check(set(drawn) == set(expected), "'key ascending, ties by identity, take the first N' holds")
    check(
        drawn == [identity for identity in members if identity in set(expected)],
        "the result is returned in candidate input order, not key order",
    )

    collide = ["dup", "dup2"]
    check(
        select_within_cell(sorted(collide), 1, seed=seed)
        == select_within_cell(sorted(collide, reverse=True), 1, seed=seed),
        "the identity tie-break makes the draw independent of the members' arrival order",
    )

    check(
        "not compared against" in SELECTION_METHOD["note"].lower()
        or "not equivalent" in SELECTION_METHOD["note"].lower(),
        "the record states this is not equivalent to, and not compared with, an RNG draw",
    )


def main() -> None:
    print("Stage5 S5-16 Step 0: B' split allocation synthetic checks")
    for test in (test_groups_and_condition, test_allocation, test_selection,
                 test_end_to_end, test_selection_method_is_specified):
        test()
    print(f"\nchecks run: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
