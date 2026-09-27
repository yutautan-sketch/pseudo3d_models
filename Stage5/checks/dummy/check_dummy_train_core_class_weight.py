from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from stage5.utils.class_weights import (  # noqa: E402
    compute_class_counts_from_h5,
    pointnext_class_weights_from_counts,
)

# ----------------------------------------------------------------------------
# S5-16 Step 0: synthetic coverage for the train_core class weight.
#
# Report 9.9. The formula is W-A's and moved unchanged out of train_stage5.py,
# so what is checked here is that the move preserved it exactly, that the
# counting rules are the intended ones (valid, non-ignore, once per video, not
# through the overlap windows), and that an empty or unusable target stops.
#
# Needs numpy and h5py. No torch, no CUDA, and the training main is never
# started.
# ----------------------------------------------------------------------------

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL: {label}")


def expect_raises(fn, label: str, exc=Exception) -> None:
    global CHECKS
    try:
        fn()
    except exc:
        check(True, label)
        return
    check(False, f"{label} (returned normally instead of raising)")


def write_h5(path: Path, labels: list[int], valid: list[bool] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        group = f.create_group("annotation")
        group.create_dataset("point_label", data=np.asarray(labels, dtype=np.int64))
        if valid is not None:
            group.create_dataset("valid_mask", data=np.asarray(valid, dtype=bool))
    return path


def reference_weights(counts, *, epsilon=0.02, normalize=True) -> list[float]:
    """The formula written out independently, to catch a silent change."""
    total = float(sum(counts))
    weights = [1.0 / (value / total + epsilon) for value in counts]
    if normalize:
        scale = len(weights) / sum(weights)
        weights = [value * scale for value in weights]
    return [float(np.float32(value)) for value in weights]


def test_formula() -> None:
    print("\n[1] the formula, epsilon and normalisation survived the move unchanged")
    counts = np.asarray([61192000, 688292], dtype=np.int64)
    got = pointnext_class_weights_from_counts(counts)
    check(np.allclose(got, reference_weights([61192000, 688292])),
          "weights match an independently written 1/(freq+eps), mean-normalised")
    check(abs(sum(got) / len(got) - 1.0) < 1e-5, "the normalised weights have mean 1")
    check(all(isinstance(value, float) for value in got), "the result is a plain float list")
    check(np.asarray(got, dtype=np.float32).tolist() == got, "the values are float32-representable")

    unnormalised = pointnext_class_weights_from_counts(counts, normalize=False)
    check(np.allclose(unnormalised, reference_weights([61192000, 688292], normalize=False)),
          "normalisation can be switched off without changing the base formula")
    bigger_eps = pointnext_class_weights_from_counts(counts, epsilon=0.1)
    check(bigger_eps[1] < got[1], "a larger epsilon damps the rare class's weight")

    expect_raises(lambda: pointnext_class_weights_from_counts(np.zeros((2,), dtype=np.int64)),
                  "an all-zero count stops rather than dividing by zero", ValueError)


def test_counting_rules(root: Path) -> None:
    print("\n[2] counting: valid, non-ignore, once per video")
    a = write_h5(root / "a.h5", [0, 0, 1, 1, -1], [True, True, True, True, True])
    counts = compute_class_counts_from_h5([a], num_classes=2, ignore_index=-1)
    check(counts.tolist() == [2, 2], "ignore-labelled points are excluded")

    b = write_h5(root / "b.h5", [0, 0, 1, 1], [True, False, True, False])
    counts = compute_class_counts_from_h5([b], num_classes=2, ignore_index=-1)
    check(counts.tolist() == [1, 1], "valid_mask=False points are excluded")

    counts = compute_class_counts_from_h5([a, b], num_classes=2, ignore_index=-1)
    check(counts.tolist() == [3, 3], "counts add across videos")

    # The overlap windows would see points in the overlap twice; reading the
    # original H5 once per video is what keeps the count honest.
    counts_once = compute_class_counts_from_h5([a], num_classes=2, ignore_index=-1)
    counts_twice = compute_class_counts_from_h5([a, a], num_classes=2, ignore_index=-1)
    check((counts_twice == counts_once * 2).all(),
          "listing a video twice doubles its contribution, so each video must be listed once")

    no_mask = write_h5(root / "c.h5", [0, 1, 1])
    counts = compute_class_counts_from_h5([no_mask], num_classes=2, ignore_index=-1)
    check(counts.tolist() == [1, 2], "a file without valid_mask treats every point as valid")

    all_ignored = write_h5(root / "d.h5", [-1, -1], [True, True])
    counts = compute_class_counts_from_h5([all_ignored], num_classes=2, ignore_index=-1)
    check(counts.tolist() == [0, 0], "an all-ignore video contributes nothing")
    expect_raises(lambda: pointnext_class_weights_from_counts(counts),
                  "an all-ignore target then stops at the weight step", ValueError)

    out_of_range = write_h5(root / "e.h5", [0, 1, 5], [True, True, True])
    expect_raises(lambda: compute_class_counts_from_h5([out_of_range], num_classes=2, ignore_index=-1),
                  "a label outside [0, num_classes-1] stops", ValueError)

    mismatched = root / "f.h5"
    mismatched.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(mismatched, "w") as f:
        group = f.create_group("annotation")
        group.create_dataset("point_label", data=np.zeros((4,), dtype=np.int64))
        group.create_dataset("valid_mask", data=np.ones((3,), dtype=bool))
    expect_raises(lambda: compute_class_counts_from_h5([mismatched], num_classes=2, ignore_index=-1),
                  "a length mismatch between labels and valid_mask stops", ValueError)

    empty = root / "g.h5"
    with h5py.File(empty, "w") as f:
        f.create_group("other")
    expect_raises(lambda: compute_class_counts_from_h5([empty], num_classes=2, ignore_index=-1),
                  "a file without an annotation group stops", KeyError)


def test_target_restriction(root: Path) -> None:
    print("\n[3] only the listed videos are counted")
    listed = write_h5(root / "listed.h5", [0, 0, 0, 1], [True] * 4)
    write_h5(root / "not_listed.h5", [1] * 100, [True] * 100)
    counts = compute_class_counts_from_h5([listed], num_classes=2, ignore_index=-1)
    check(counts.tolist() == [3, 1],
          "a file sitting beside the listed one is not counted: the list is the target, not the directory")


def test_cli_refuses_without_contract(root: Path) -> None:
    print("\n[4] the CLI refuses to compute without a confirmed target")
    tool = REPO_ROOT / "checks" / "real_h5" / "compute_stage5_train_core_class_weight.py"
    listing = root / "list.txt"
    listing.write_text(str(write_h5(root / "x.h5", [0, 1], [True, True])) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tool), "--train_core_list", str(listing),
         "--private_json", str(root / "out.json")],
        capture_output=True, text=True,
    )
    check(result.returncode != 0, "no approved split contract means no computation")
    check(not (root / "out.json").exists(), "nothing is written when the target is unconfirmed")


def main() -> None:
    print("Stage5 S5-16 Step 0: train_core class weight synthetic checks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_formula()
        test_counting_rules(root / "counting")
        test_target_restriction(root / "restriction")
        test_cli_refuses_without_contract(root / "cli")
    print(f"\nchecks run: {CHECKS}, failures: {len(FAILURES)}")
    if FAILURES:
        for label in FAILURES:
            print(f"  - {label}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
