from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def decode(values: np.ndarray) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only audit of Stage 4 BBox-ranked teacher output."
    )
    parser.add_argument("--annotated_h5", type=Path, required=True)
    parser.add_argument("--require_local_selection", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.annotated_h5, "r") as handle:
        label_mode = str(handle.attrs.get("label_mode", ""))
        if label_mode != "bbox_ranked_global_local":
            raise AssertionError(f"Unexpected label_mode: {label_mode}")
        if "frame_annotation" not in handle:
            raise KeyError("frame_annotation group is missing")
        group = handle["frame_annotation"]
        required = {
            "frame_order",
            "frame_index",
            "bbox_index",
            "valid_contour",
            "selected_contour_source",
            "contour_selection_score",
            "center_distance_norm",
            "selected_area_ratio",
            "global_candidate_score",
            "local_candidate_score",
            "num_global_candidates",
            "num_local_candidates",
            "annotation_reason",
            "xml_path",
        }
        missing = required - set(group.keys())
        if missing:
            raise KeyError(f"Missing frame_annotation datasets: {sorted(missing)}")

        rows = {key: group[key][:] for key in required}
        sources = decode(rows["selected_contour_source"])
        reasons = decode(rows["annotation_reason"])
        xml_paths = decode(rows["xml_path"])
        valid = rows["valid_contour"].astype(bool)
        global_score = rows["global_candidate_score"].astype(np.float64)
        local_score = rows["local_candidate_score"].astype(np.float64)

        if len(sources) != int(valid.size):
            raise AssertionError("frame annotation row count mismatch")
        for index, source in enumerate(sources):
            if valid[index] and source not in {
                "global",
                "local_percentile",
                "shared",
            }:
                raise AssertionError(
                    f"Valid row {index} has invalid selected source: {source}"
                )
            if not valid[index] and source != "none":
                raise AssertionError(
                    f"Invalid row {index} has selected source: {source}"
                )
            if (
                source == "local_percentile"
                and np.isfinite(global_score[index])
                and local_score[index] + 1e-7 < global_score[index]
            ):
                raise AssertionError(
                    f"Local row {index} has a lower score than global"
                )

        local_count = int(sum(source == "local_percentile" for source in sources))
        global_count = int(sum(source == "global" for source in sources))
        shared_count = int(sum(source == "shared" for source in sources))
        if args.require_local_selection and local_count == 0:
            raise AssertionError("No local-percentile contour was selected")

        labels = handle["annotation/point_label"][:].astype(np.int8)
        positive_count = int(np.count_nonzero(labels == 1))
        print("Stage 4 BBox-ranked teacher real-data audit (read-only)")
        print(f"annotated_h5 : {args.annotated_h5}")
        print(f"video_name   : {handle.attrs.get('video_name', '')}")
        print(f"bbox rows    : {len(sources)}")
        print(
            "selected     : "
            f"global={global_count}, local={local_count}, shared={shared_count}"
        )
        print(f"positive pts : {positive_count}")

        print("\nPer-BBox selection")
        for index, source in enumerate(sources):
            print(
                f"row={index} "
                f"frame={int(rows['frame_order'][index])}/"
                f"{int(rows['frame_index'][index])} "
                f"bbox={int(rows['bbox_index'][index])} "
                f"valid={bool(valid[index])} "
                f"source={source} "
                f"score={float(rows['contour_selection_score'][index]):.4f} "
                f"global={float(global_score[index]):.4f} "
                f"local={float(local_score[index]):.4f} "
                f"center={float(rows['center_distance_norm'][index]):.4f} "
                f"area_ratio={float(rows['selected_area_ratio'][index]):.4f} "
                f"candidates={int(rows['num_global_candidates'][index])}/"
                f"{int(rows['num_local_candidates'][index])} "
                f"reason={reasons[index]} "
                f"xml={xml_paths[index]}"
            )

    print("Stage 4 BBox-ranked teacher real-data audit passed.")


if __name__ == "__main__":
    main()
