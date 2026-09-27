from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

# ----------------------------------------------------------------------------
# S5-16 Step 0: the class-weight computation, moved out of train_stage5.py.
#
# Both functions are the W-A logic unchanged -- same formula, same defaults,
# same return types. They moved so the train_core-only recomputation can run on
# CPU without importing torch, and so one implementation serves both the
# training run and the Step 0 tool. Report 9.8: reuse in the smallest scope
# that avoids a second copy of the formula.
#
# Counting reads each video's original H5 once and uses valid, non-ignore
# points only. It deliberately does NOT go through the overlap windows, which
# would count points in the overlap twice.
# ----------------------------------------------------------------------------


def compute_class_counts_from_h5(
    h5_paths: Iterable[Path],
    *,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    counts = np.zeros((int(num_classes),), dtype=np.int64)
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            if "annotation" not in f:
                raise KeyError(f"'annotation' group not found in {h5_path}")
            annotation = f["annotation"]
            if "point_label" not in annotation:
                raise KeyError(f"annotation/point_label not found in {h5_path}")
            labels = annotation["point_label"][:].astype(np.int64)
            if "valid_mask" in annotation:
                valid_mask = annotation["valid_mask"][:].astype(bool)
            else:
                valid_mask = np.ones_like(labels, dtype=bool)

        if labels.shape != valid_mask.shape:
            raise ValueError(
                f"annotation length mismatch in {h5_path}: "
                f"point_label={labels.shape}, valid_mask={valid_mask.shape}"
            )

        mask = valid_mask & (labels != int(ignore_index))
        selected = labels[mask]
        if selected.size == 0:
            continue
        invalid = selected[(selected < 0) | (selected >= int(num_classes))]
        if invalid.size > 0:
            unique_invalid = sorted(set(int(item) for item in invalid.tolist()))
            raise ValueError(
                f"Found label(s) outside [0, {num_classes - 1}] in {h5_path}: {unique_invalid}"
            )
        counts += np.bincount(selected, minlength=int(num_classes))[: int(num_classes)]
    return counts


def pointnext_class_weights_from_counts(
    counts: np.ndarray,
    *,
    epsilon: float = 0.02,
    normalize: bool = True,
) -> list[float]:
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute class weights because no valid labeled points were found")
    frequency = counts.astype(np.float64) / float(total)
    weights = 1.0 / (frequency + float(epsilon))
    if normalize:
        weights = weights * len(weights) / weights.sum()
    return weights.astype(np.float32).tolist()
