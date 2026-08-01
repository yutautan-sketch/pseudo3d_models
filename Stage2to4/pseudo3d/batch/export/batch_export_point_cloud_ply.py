from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudo3d.annotation.annotate_pseudo3d_point_cloud import load_point_cloud_h5
from pseudo3d.export.export_pseudo3d_point_cloud import save_point_cloud_ply


def collect_inputs(
    input_dir: Path,
    *,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Point-cloud input directory not found: {input_dir}")
    globber = input_dir.rglob if recursive else input_dir.glob
    paths = sorted(path for path in globber(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No point-cloud H5 files matched {pattern!r} in {input_dir}"
        )
    return paths


def format_template(template: str, context: dict[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        raise KeyError(
            f"Unknown template field {exc} in {template!r}; "
            f"available={sorted(context)}"
        ) from exc


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "input_h5",
        "output_ply",
        "num_points",
        "point_mode",
        "sampling_mode",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def attr_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert existing raw point-cloud H5 files to PLY."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--output_subdir_template", default="{parent_name}")
    parser.add_argument("--output_filename_template", default="{input_stem}.ply")
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    inputs = collect_inputs(
        args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    print(f"Found raw point-cloud H5 files: {len(inputs)}")
    print(f"Output root                    : {args.output_root}")

    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    failed = 0
    for index, input_h5 in enumerate(inputs, start=1):
        context = {
            "input_stem": input_h5.stem,
            "input_name": input_h5.name,
            "parent_name": input_h5.parent.name,
        }
        output_ply = (
            args.output_root
            / format_template(args.output_subdir_template, context)
            / format_template(args.output_filename_template, context)
        )
        print("\n" + "=" * 72)
        print(f"[{index}/{len(inputs)}] input : {input_h5}")
        print(f"output: {output_ply}")

        if args.skip_existing and output_ply.is_file():
            rows.append(
                {
                    "status": "skipped",
                    "input_h5": str(input_h5),
                    "output_ply": str(output_ply),
                }
            )
            skipped += 1
            print("Skipping existing output.")
            continue

        try:
            point_cloud, attrs = load_point_cloud_h5(input_h5)
            save_point_cloud_ply(output_ply, point_cloud=point_cloud)
            row = {
                "status": "ok",
                "input_h5": str(input_h5),
                "output_ply": str(output_ply),
                "num_points": int(point_cloud["points"].shape[0]),
                "point_mode": attr_text(attrs.get("point_mode"), "foreground"),
                "sampling_mode": attr_text(attrs.get("sampling_mode"), "legacy"),
                "error": "",
            }
            rows.append(row)
            processed += 1
            print(f"Exported: points={row['num_points']}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "status": "failed",
                    "input_h5": str(input_h5),
                    "output_ply": str(output_ply),
                    "error": message,
                }
            )
            failed += 1
            print(f"ERROR: {message}", file=sys.stderr)
            traceback.print_exc()
            if not args.continue_on_error:
                break

    write_summary(args.summary_csv, rows)
    print("\nRaw point-cloud PLY export summary:")
    print(f"  processed: {processed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {failed}")
    print(f"  summary  : {args.summary_csv}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
