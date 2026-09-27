from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage5.utils.class_weights import (  # noqa: E402
    compute_class_counts_from_h5,
    pointnext_class_weights_from_counts,
)
from stage5.utils.file_list_mode import (  # noqa: E402
    list_content_sha256,
    list_identity_sha256,
    read_file_list,
)
from stage5.utils.split_contract import resolve_active_contract  # noqa: E402
from stage5.utils.split_identity import file_sha256  # noqa: E402

# ----------------------------------------------------------------------------
# S5-16 Step 0 工程3: the train_core-only class weight, computed once.
#
# Report 9.10 and handoff section 5. The formula, epsilon and normalisation are
# W-A's, reused from stage5/utils/class_weights.py rather than restated here --
# a second copy of the formula is exactly how two "identical" computations stop
# being identical. Counting reads each video's original H5 once; it does not go
# through the overlap windows, which would double-count the overlap.
#
# The result is FIXED at this point and reused unchanged by S5-18..S5-20. The
# old W-A value is reported for history only: the difference is not a reason to
# adjust or re-search the new one (handoff section 5).
#
# Only train_core is read. validation and internal_test are never counted, and
# the seal contract refuses them before any H5 is opened. CPU only; the
# training main is never started.
# ----------------------------------------------------------------------------

OLD_W_A_CLASS_WEIGHT = [0.05963856, 1.94036150]


def git_revision(repo_dir: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {"head": run(["rev-parse", "HEAD"]), "dirty": bool(run(["status", "--short"]))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the S5-17+ class weight from train_core only, once, on CPU. "
        "Records the inputs, the code revision and the settings so the value can be fixed."
    )
    parser.add_argument("--train_core_list", required=True, help="Explicit train_core file list")
    parser.add_argument("--split_manifest", default=None, help="Approved pin name (not a path)")
    parser.add_argument("--split_contract_pins", default=None)
    parser.add_argument("--artifact_coverage", default=None)
    parser.add_argument(
        "--expected_identity_sha256",
        default=None,
        help="Identity fingerprint the list must have; normally the manifest's train_core value",
    )
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--private_json", required=True, help="DO_NOT_SHARE output with the counts")
    parser.add_argument("--shareable_json", default=None, help="Hashes and settings only, no counts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = read_file_list(args.train_core_list, label="train_core")

    # The contract decides before a single H5 is opened. A list that reached
    # here with a sealed video in it stops the tool rather than being filtered.
    contract = resolve_active_contract(
        manifest_key=args.split_manifest,
        pins_path=args.split_contract_pins,
        coverage_path=args.artifact_coverage,
    )
    contract.assert_paths_allowed(paths, purpose="class-weight counting inputs")

    identity = list_identity_sha256(paths)
    content = list_content_sha256(paths)

    expected = args.expected_identity_sha256
    if expected is None:
        expected = (contract.manifest.lists.get("train_core") or {}).get("identity_sha256")
    if not expected:
        print(
            "FAIL: no expected train_core identity fingerprint was given and the manifest records "
            "none; the target set cannot be confirmed, so nothing is computed.",
            file=sys.stderr,
        )
        sys.exit(2)
    if identity != expected:
        print(
            "FAIL: the supplied list is not the confirmed train_core set.\n"
            f"  expected identity: {expected}\n  actual identity  : {identity}",
            file=sys.stderr,
        )
        sys.exit(2)

    missing = [path for path in paths if not path.is_file()]
    if missing:
        print(f"FAIL: {len(missing)} listed H5 file(s) do not exist", file=sys.stderr)
        sys.exit(2)

    counts = compute_class_counts_from_h5(
        paths, num_classes=args.num_classes, ignore_index=args.ignore_index
    )
    weights = pointnext_class_weights_from_counts(
        counts, epsilon=args.epsilon, normalize=args.normalize
    )

    total = int(counts.sum())
    settings = {
        "num_classes": int(args.num_classes),
        "ignore_index": int(args.ignore_index),
        "auto_class_weight_epsilon": float(args.epsilon),
        "normalize_auto_class_weight": bool(args.normalize),
        "formula": "w_c = 1 / (count_c / total + epsilon); normalized to mean 1; cast to float32",
        "counted_from": "original H5 annotation/point_label with valid_mask, once per video",
        "not_counted_via": "overlap windows (would double-count points in the overlap)",
    }
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(REPO_ROOT),
        "class_weights_module_sha256": file_sha256(REPO_ROOT / "stage5" / "utils" / "class_weights.py"),
        "tool_sha256": file_sha256(Path(__file__)),
        "python": platform.python_version(),
        "split_manifest": contract.manifest.name,
        "split_manifest_sha256": contract.manifest.sha256,
        "seal_registry_sha256": contract.registry.sha256,
    }

    private = {
        "schema": "stage5_train_core_class_weight_v1",
        "note": "DO_NOT_SHARE: carries label counts and the real list path.",
        "train_core_list": str(args.train_core_list),
        "num_files": len(paths),
        "identity_sha256": identity,
        "content_sha256": content,
        "counts": [int(value) for value in counts.tolist()],
        "total_valid_points": total,
        "class_weight": weights,
        "settings": settings,
        "provenance": provenance,
        "history_only_old_w_a": {
            "value": OLD_W_A_CLASS_WEIGHT,
            "note": (
                "Reported for history only. The difference is NOT a reason to adjust or re-search "
                "this value (handoff section 5). The old value came from the 162-video era, which "
                "included internal_test candidates."
            ),
        },
    }
    Path(args.private_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.private_json).write_text(
        json.dumps(private, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.shareable_json:
        shareable = {
            "schema": "stage5_train_core_class_weight_shareable_v1",
            "num_files": len(paths),
            "identity_sha256": identity,
            "content_sha256": content,
            "class_weight": weights,
            "settings": settings,
            "provenance": provenance,
            "withheld": "per-class counts and the total point count are not shared",
        }
        Path(args.shareable_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.shareable_json).write_text(
            json.dumps(shareable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # Counts stay out of stdout; the resolved weight is the thing being fixed.
    print("Stage5 train_core class weight computed once")
    print(f"  files           : {len(paths)}")
    print(f"  identity sha256 : {identity[:16]}")
    print(f"  epsilon         : {args.epsilon}  normalize: {args.normalize}")
    print(f"  class weight    : {weights}")
    print("  Per-class counts are in the private JSON only.")
    print("  Fix this value for S5-18..S5-20; do not recompute per stage.")


if __name__ == "__main__":
    main()
