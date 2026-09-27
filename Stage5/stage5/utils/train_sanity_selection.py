from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

# ----------------------------------------------------------------------------
# S5-16 Step 0: the train-sanity selection, as a stdlib-only reference.
#
# `evaluate_stage5.py` and `prepare_stage5_evaluation_data.py` each carry their
# own copy of this function (identical apart from a temporary variable). Those
# copies are deliberately left alone here -- changing the behaviour of the
# evaluation path is not part of Step 0 -- but the S0-1 audit needs to be able
# to re-derive the selection WITHOUT importing torch or h5py, so the same
# algorithm lives here too.
#
# check_dummy_step0_input_audit.py extracts the copy out of `evaluate_stage5.py`
# by text and asserts it agrees with this one, the same anti-drift trick
# check_dummy_fixed_list_mode.sh already uses on `train_stage5.sh`.
#
# Why this matters for Step 0: the selection draws from `paths` IN LIST ORDER,
# so the identity of the three sanity videos is defined only relative to the
# saved train162 list. Re-running it against train_core would pick different
# videos. The audit therefore prefers the saved selection and treats
# re-derivation as a fallback that must still be corroborated.
# ----------------------------------------------------------------------------


def select_train_paths(
    paths: list[Path],
    *,
    fixed_videos: Iterable[str],
    num_random: int,
    seed: int,
) -> list[Path]:
    selected: list[Path] = []
    for video in fixed_videos:
        matches = [path for path in paths if video in path.name]
        if len(matches) != 1:
            raise ValueError(
                f"Fixed train video {video!r} matched {len(matches)} paths in train list"
            )
        if matches[0] not in selected:
            selected.append(matches[0])

    candidates = [path for path in paths if path not in selected]
    if num_random < 0:
        raise ValueError("--num_random_train must be non-negative")
    if num_random > len(candidates):
        raise ValueError(
            f"Requested {num_random} random train files but only {len(candidates)} are available"
        )
    rng = random.Random(seed)
    selected.extend(rng.sample(candidates, num_random))
    return selected
