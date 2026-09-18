from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# S5-15: confirm early that a running/finished training run is using the
# settings it was supposed to, and that periodic checkpoint saving works.
#
# The S5-15 P3 runs exposed the gap this closes: `SAVE_EVERY=1` was exported but
# a plain assignment in train_stage5.sh discarded it, so `config.json` recorded
# `save_every: 10` and no per-epoch checkpoint was ever written. Nothing failed,
# and the deviation surfaced only after both 5-epoch runs had finished.
#
# The env-passthrough test is a STATIC analysis of the shell scripts. This is
# the runtime counterpart: it reads what train_stage5.py actually wrote, so a
# value that fails to reach Python for any reason is caught while the run is
# still young rather than at the end.
#
# Read-only, CPU-only, stdlib-only: it never modifies the run directory.
# ----------------------------------------------------------------------------

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_UNKNOWN = "UNKNOWN"


def parse_expected(pairs: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        expected[key.strip()] = value.strip()
    return expected


def coerce_to(actual: Any, text: str) -> Any:
    """Read the expected value in the type config.json actually stored."""
    if isinstance(actual, bool):
        lowered = text.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return text
    if isinstance(actual, int):
        try:
            return int(text)
        except ValueError:
            return text
    if isinstance(actual, float):
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(actual, list):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def check_settings(config: dict[str, Any], expected: dict[str, str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for key, raw in expected.items():
        if key not in config:
            results.append(
                {"check": f"config.{key}", "status": STATUS_FAIL, "detail": f"key absent from config.json (expected {raw!r})"}
            )
            continue
        actual = config[key]
        wanted = coerce_to(actual, raw)
        matches = actual == wanted
        results.append(
            {
                "check": f"config.{key}",
                "status": STATUS_PASS if matches else STATUS_FAIL,
                "detail": f"expected={wanted!r} effective={actual!r}",
            }
        )
    return results


def check_checkpoint_present(run_dir: Path, name: str | None) -> dict[str, Any]:
    if not name:
        return {
            "check": "checkpoint.first_periodic",
            "status": STATUS_UNKNOWN,
            "detail": "no --expect_checkpoint given; periodic saving was not verified",
        }
    path = run_dir / name
    if path.is_file():
        return {
            "check": "checkpoint.first_periodic",
            "status": STATUS_PASS,
            "detail": f"{name} exists ({path.stat().st_size} bytes): periodic saving is working",
        }
    present = sorted(p.name for p in run_dir.glob("*.pt"))
    return {
        "check": "checkpoint.first_periodic",
        "status": STATUS_FAIL,
        "detail": f"{name} is missing; the run directory holds {present}. "
        "If the epoch has completed, periodic saving is not working -- stop and report.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a training run's EFFECTIVE config.json against expected values, and "
        "optionally that a specific periodic checkpoint exists. Read-only and CPU-only."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config.json key expected to hold VALUE; repeatable",
    )
    parser.add_argument(
        "--expect_checkpoint",
        default=None,
        help="Checkpoint file that must exist by now, e.g. checkpoint_epoch_0005.pt",
    )
    parser.add_argument("--json_out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        print(f"FAIL: config.json not found under {run_dir}", file=sys.stderr)
        sys.exit(2)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    results = check_settings(config, parse_expected(args.expect))
    results.append(check_checkpoint_present(run_dir, args.expect_checkpoint))

    counts = {
        status: sum(1 for r in results if r["status"] == status)
        for status in (STATUS_PASS, STATUS_FAIL, STATUS_UNKNOWN)
    }
    status = "passed" if counts[STATUS_FAIL] == 0 else "failed"
    summary = {"status": status, "counts": counts, "results": results, "run_dir": str(run_dir)}

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Stage5 effective run-config check")
    print(f"  status : {status}")
    print(f"  counts : {counts}")
    for row in results:
        if row["status"] != STATUS_PASS:
            print(f"  [{row['status']}] {row['check']}: {row['detail']}")
        else:
            print(f"  [PASS] {row['check']}: {row['detail']}")
    if args.json_out:
        print(f"  output: {args.json_out}")

    sys.exit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
