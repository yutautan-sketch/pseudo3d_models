from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# S5-15: are two checkpoints in one run directory the same thing?
#
# Written for the approved one-off audit of the historical W-A run, whose
# best.pt and last.pt were never compared (handoff 6.2 asks that a run where
# epoch 5 and best coincide be recorded as such, so the evaluation is not done
# twice for nothing).
#
# Read-only, CPU-only, and it never writes into the audited run directory. The
# order matters: SHA-256 first, and a mismatch is NOT by itself evidence of a
# different model -- serialization order or optimizer/metadata can differ while
# the evaluated weights are identical, so the two questions are answered and
# reported separately.
# ----------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    """Load with the same call the production loader uses.

    torch >= 2.6 defaults to weights_only=True, which can reject a checkpoint
    that also carries its config. The fallback is recorded in the output rather
    than applied silently; these are the project's own trusted artifacts.
    """
    import torch

    try:
        return torch.load(path, map_location="cpu"), "default"
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload, f"weights_only=False after: {type(error).__name__}"


def compare_state_dicts(state_a: dict[str, Any], state_b: dict[str, Any]) -> dict[str, Any]:
    import torch

    keys_a, keys_b = set(state_a), set(state_b)
    only_a, only_b = sorted(keys_a - keys_b), sorted(keys_b - keys_a)
    shared = sorted(keys_a & keys_b)

    shape_dtype_mismatches: list[str] = []
    value_mismatches: list[dict[str, Any]] = []
    for key in shared:
        tensor_a, tensor_b = state_a[key], state_b[key]
        if not torch.is_tensor(tensor_a) or not torch.is_tensor(tensor_b):
            if tensor_a != tensor_b:
                value_mismatches.append({"key": key, "note": "non-tensor entries differ"})
            continue
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            shape_dtype_mismatches.append(
                f"{key}: {tuple(tensor_a.shape)}/{tensor_a.dtype} vs {tuple(tensor_b.shape)}/{tensor_b.dtype}"
            )
            continue
        if not torch.equal(tensor_a, tensor_b):
            difference = (tensor_a.to(torch.float64) - tensor_b.to(torch.float64)).abs().max().item()
            value_mismatches.append({"key": key, "max_abs_diff": difference})

    return {
        "num_keys_a": len(keys_a),
        "num_keys_b": len(keys_b),
        "only_in_a": only_a[:10],
        "only_in_b": only_b[:10],
        "num_shape_or_dtype_mismatches": len(shape_dtype_mismatches),
        "shape_or_dtype_mismatches": shape_dtype_mismatches[:10],
        "num_value_mismatches": len(value_mismatches),
        "value_mismatches": value_mismatches[:10],
        "identical": not only_a and not only_b and not shape_dtype_mismatches and not value_mismatches,
    }


def summarize_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": payload.get("epoch"),
        "best_score": payload.get("best_score"),
        "has_optimizer_state": "optimizer" in payload,
        "top_level_keys": sorted(payload.keys()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two checkpoints in one run directory: first as files (SHA-256), then, "
        "only if they differ, as evaluation models (state_dict keys/shapes/dtypes/values). "
        "Read-only and CPU-only; the run directory is never modified."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint_a", default="best.pt")
    parser.add_argument("--checkpoint_b", default="last.pt")
    parser.add_argument(
        "--skip_load",
        action="store_true",
        help="Compare files only; do not load either checkpoint even if the hashes differ",
    )
    parser.add_argument("--json_out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    path_a, path_b = run_dir / args.checkpoint_a, run_dir / args.checkpoint_b
    for path in (path_a, path_b):
        if not path.is_file():
            print(f"FAIL: checkpoint not found: {path}", file=sys.stderr)
            sys.exit(2)

    sha_a, sha_b = sha256_file(path_a), sha256_file(path_b)
    file_identical = sha_a == sha_b

    summary: dict[str, Any] = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "sha256_a": sha_a,
        "sha256_b": sha_b,
        "size_bytes_a": path_a.stat().st_size,
        "size_bytes_b": path_b.stat().st_size,
        "file_identical": file_identical,
    }

    if file_identical:
        summary["evaluation_model_identical"] = True
        summary["conclusion"] = "identical_file"
        summary["note"] = "byte-identical, so no load was needed and the two names denote one checkpoint"
    elif args.skip_load:
        summary["evaluation_model_identical"] = None
        summary["conclusion"] = "file_differs_model_not_checked"
        summary["note"] = "--skip_load was given; a hash mismatch alone does not establish a different model"
    else:
        payload_a, load_mode_a = load_checkpoint(path_a)
        payload_b, load_mode_b = load_checkpoint(path_b)
        summary["load_mode_a"] = load_mode_a
        summary["load_mode_b"] = load_mode_b
        summary["metadata_a"] = summarize_metadata(payload_a)
        summary["metadata_b"] = summarize_metadata(payload_b)
        summary["config_identical"] = payload_a.get("config") == payload_b.get("config")

        state_a = payload_a.get("model", payload_a)
        state_b = payload_b.get("model", payload_b)
        state_comparison = compare_state_dicts(state_a, state_b)
        summary["state_dict_comparison"] = state_comparison
        summary["evaluation_model_identical"] = state_comparison["identical"]
        if state_comparison["identical"]:
            summary["conclusion"] = "different_file_same_evaluation_model"
            summary["note"] = (
                "the files differ but every model tensor matches exactly; the difference lives in "
                "metadata or optimizer state, which evaluation does not use"
            )
        else:
            summary["conclusion"] = "different_evaluation_model"
            summary["note"] = "model tensors differ, so these are genuinely two different evaluation models"

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 checkpoint identity check")
    print(f"  {args.checkpoint_a}: {sha_a} ({summary['size_bytes_a']} bytes)")
    print(f"  {args.checkpoint_b}: {sha_b} ({summary['size_bytes_b']} bytes)")
    print(f"  file identical            : {summary['file_identical']}")
    print(f"  evaluation model identical: {summary['evaluation_model_identical']}")
    print(f"  conclusion                : {summary['conclusion']}")
    print(f"  note                      : {summary['note']}")
    if "metadata_a" in summary:
        print(f"  epoch {args.checkpoint_a}: {summary['metadata_a']['epoch']}, "
              f"{args.checkpoint_b}: {summary['metadata_b']['epoch']}")
        print(f"  config identical          : {summary['config_identical']}")
        comparison = summary["state_dict_comparison"]
        print(
            f"  state_dict                : {comparison['num_keys_a']} vs {comparison['num_keys_b']} keys, "
            f"{comparison['num_value_mismatches']} value mismatch(es)"
        )
    if args.json_out:
        print(f"  output: {args.json_out}")


if __name__ == "__main__":
    main()
