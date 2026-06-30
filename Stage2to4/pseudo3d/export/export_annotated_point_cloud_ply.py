from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_FEMUR_CANDIDATE = 1
LABEL_FRAME_ENDPOINT = 2
LABEL_MEASUREMENT_ENDPOINT = 3
LABEL_ANNOTATION_MARKER = 4
np: Any = None


def ensure_numpy() -> Any:
    global np
    if np is None:
        import numpy as _np

        np = _np
    return np


def parse_color(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Color must be R,G,B, got: {text}")
    rgb = tuple(int(part) for part in parts)
    if any(value < 0 or value > 255 for value in rgb):
        raise ValueError(f"Color values must be in [0,255], got: {text}")
    return rgb


def load_annotated_h5(path: Path) -> dict[str, Any]:
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotated h5 not found: {path}")

    with h5py.File(path, "r") as f:
        if "point_cloud" not in f:
            raise KeyError(f"'point_cloud' group not found in {path}")
        if "annotation" not in f:
            raise KeyError(f"'annotation' group not found in {path}")

        pc = f["point_cloud"]
        ann = f["annotation"]
        required_pc = [
            "points",
            "intensity",
            "alpha",
            "confidence",
            "frame_order",
        ]
        required_ann = ["point_label", "valid_mask"]
        for key in required_pc:
            if key not in pc:
                raise KeyError(f"Required point_cloud/{key} not found in {path}")
        for key in required_ann:
            if key not in ann:
                raise KeyError(f"Required annotation/{key} not found in {path}")

        data: dict[str, Any] = {
            "points": pc["points"][:].astype(np.float32),
            "intensity": pc["intensity"][:].astype(np.uint8),
            "alpha": pc["alpha"][:].astype(np.uint8),
            "confidence": pc["confidence"][:].astype(np.float32),
            "frame_order": pc["frame_order"][:].astype(np.int32),
            "point_label": ann["point_label"][:].astype(np.int8),
            "valid_mask": ann["valid_mask"][:].astype(bool),
            "annotation_attrs": dict(ann.attrs),
            "root_attrs": dict(f.attrs),
        }

        if "frame_index" in pc:
            data["frame_index"] = pc["frame_index"][:].astype(np.int64)
        if "pixel_xy" in pc:
            data["pixel_xy"] = pc["pixel_xy"][:].astype(np.float32)
        if "source_type" in pc:
            data["source_type"] = pc["source_type"][:].astype(np.uint8)

        if "frame_annotation" in f:
            frame_group = f["frame_annotation"]
            frame_annotation: dict[str, Any] = {}
            for key in frame_group.keys():
                value = frame_group[key][:]
                if value.dtype.kind == "S":
                    value = value.astype(str)
                frame_annotation[key] = value
            data["frame_annotation"] = frame_annotation

            if "endpoint_3d" in frame_group and "valid_endpoint" in frame_group:
                data["frame_endpoint_3d"] = frame_group["endpoint_3d"][:].astype(
                    np.float32
                )
                data["frame_valid_endpoint"] = frame_group["valid_endpoint"][:].astype(
                    bool
                )
                if "frame_order" in frame_group:
                    data["frame_endpoint_order"] = frame_group["frame_order"][:].astype(
                        np.int32
                    )

        if "measurement" in f:
            meas = f["measurement"]
            if "endpoint_1" in meas and "endpoint_2" in meas:
                endpoint_1 = meas["endpoint_1"][:].astype(np.float32)
                endpoint_2 = meas["endpoint_2"][:].astype(np.float32)
                data["measurement_endpoint_3d"] = np.stack(
                    [endpoint_1, endpoint_2],
                    axis=0,
                )

    return data


def make_point_colors(
    *,
    intensity: np.ndarray,
    alpha: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    femur_color: tuple[int, int, int],
    background_mode: str,
    background_alpha: int,
    ignore_alpha: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = intensity.astype(np.uint8)
    colors = np.stack([gray, gray, gray], axis=1)
    out_alpha = alpha.astype(np.uint8).copy()

    if background_mode == "dim":
        colors = (colors.astype(np.float32) * 0.35).astype(np.uint8)
        out_alpha[:] = np.uint8(background_alpha)
    elif background_mode == "gray":
        out_alpha[:] = np.uint8(background_alpha)
    elif background_mode == "hidden":
        pass
    else:
        raise ValueError(f"Unknown background_mode: {background_mode}")

    ignore = labels == LABEL_IGNORE
    colors[ignore] = (60, 60, 60)
    out_alpha[ignore] = np.uint8(ignore_alpha)

    annotated = valid_mask & (labels == LABEL_FEMUR_CANDIDATE)
    colors[annotated] = np.asarray(femur_color, dtype=np.uint8)
    out_alpha[annotated] = np.uint8(255)

    return colors.astype(np.uint8), out_alpha


def select_points(
    data: dict[str, Any],
    *,
    background_mode: str,
    label_values: set[int],
) -> np.ndarray:
    labels = data["point_label"]
    valid_mask = data["valid_mask"]
    if background_mode == "hidden":
        return valid_mask & np.isin(labels, list(label_values))
    return np.ones(labels.shape[0], dtype=bool)


def select_annotation_points(data: dict[str, Any]) -> np.ndarray:
    labels = data["point_label"]
    valid_mask = data["valid_mask"]
    return valid_mask & (labels == LABEL_FEMUR_CANDIDATE)


def build_endpoint_vertices_and_edges(
    data: dict[str, Any],
    *,
    start_vertex_index: int,
    include_measurement_endpoint: bool,
    include_frame_endpoints: bool,
    measurement_color: tuple[int, int, int],
    frame_endpoint_color: tuple[int, int, int],
) -> tuple[list[tuple[float, float, float, int, int, int, int, int, int, float]], list[tuple[int, int, int, int, int, int, int]]]:
    vertices = []
    edges = []

    def add_endpoint_pair(
        endpoint_pair: np.ndarray,
        *,
        color: tuple[int, int, int],
        label: int,
        frame_order: int,
        confidence: float,
    ) -> None:
        if endpoint_pair.shape != (2, 3):
            return
        if not np.isfinite(endpoint_pair).all():
            return
        i0 = start_vertex_index + len(vertices)
        for point in endpoint_pair:
            vertices.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    255,
                    int(label),
                    int(frame_order),
                    float(confidence),
                )
            )
        edges.append((i0, i0 + 1, int(color[0]), int(color[1]), int(color[2]), 255, label))

    if include_frame_endpoints and "frame_endpoint_3d" in data:
        endpoints = data["frame_endpoint_3d"]
        valid = data.get(
            "frame_valid_endpoint",
            np.ones(endpoints.shape[0], dtype=bool),
        )
        orders = data.get(
            "frame_endpoint_order",
            np.arange(endpoints.shape[0], dtype=np.int32),
        )
        for endpoint_pair, is_valid, order in zip(endpoints, valid, orders, strict=True):
            if not bool(is_valid):
                continue
            add_endpoint_pair(
                endpoint_pair,
                color=frame_endpoint_color,
                label=2,
                frame_order=int(order),
                confidence=1.0,
            )

    if include_measurement_endpoint and "measurement_endpoint_3d" in data:
        add_endpoint_pair(
            data["measurement_endpoint_3d"],
            color=measurement_color,
            label=3,
            frame_order=-1,
            confidence=1.0,
        )

    return vertices, edges


def build_annotation_marker_vertices_and_edges(
    data: dict[str, Any],
    *,
    start_vertex_index: int,
    marker_color: tuple[int, int, int],
    marker_size: float,
    marker_stride: int,
    max_markers: int,
) -> tuple[list[tuple[float, float, float, int, int, int, int, int, int, float]], list[tuple[int, int, int, int, int, int, int]]]:
    labels = data["point_label"]
    valid_mask = data["valid_mask"]
    annotated_idx = np.flatnonzero(valid_mask & (labels == LABEL_FEMUR_CANDIDATE))

    marker_stride = max(1, int(marker_stride))
    annotated_idx = annotated_idx[::marker_stride]
    if max_markers > 0 and annotated_idx.size > max_markers:
        annotated_idx = annotated_idx[:max_markers]

    vertices = []
    edges = []
    half = float(marker_size) * 0.5
    if half <= 0.0:
        return vertices, edges

    offsets = np.asarray(
        [
            [-half, 0.0, 0.0],
            [half, 0.0, 0.0],
            [0.0, -half, 0.0],
            [0.0, half, 0.0],
            [0.0, 0.0, -half],
            [0.0, 0.0, half],
        ],
        dtype=np.float32,
    )
    edge_pairs = [(0, 1), (2, 3), (4, 5)]
    points = data["points"]
    frame_order = data["frame_order"]
    confidence = data["confidence"]

    for point_idx in annotated_idx:
        base = start_vertex_index + len(vertices)
        center = points[int(point_idx)].astype(np.float32)
        for offset in offsets:
            point = center + offset
            vertices.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(marker_color[0]),
                    int(marker_color[1]),
                    int(marker_color[2]),
                    255,
                    LABEL_ANNOTATION_MARKER,
                    int(frame_order[int(point_idx)]),
                    float(confidence[int(point_idx)]),
                )
            )
        for a, b in edge_pairs:
            edges.append(
                (
                    base + a,
                    base + b,
                    int(marker_color[0]),
                    int(marker_color[1]),
                    int(marker_color[2]),
                    255,
                    LABEL_ANNOTATION_MARKER,
                )
            )

    return vertices, edges


def save_ply(
    output_ply: Path,
    *,
    data: dict[str, Any],
    keep: np.ndarray,
    colors: np.ndarray,
    out_alpha: np.ndarray,
    include_measurement_endpoint: bool,
    include_frame_endpoints: bool,
    measurement_color: tuple[int, int, int],
    frame_endpoint_color: tuple[int, int, int],
    include_annotation_markers: bool,
    annotation_marker_color: tuple[int, int, int],
    annotation_marker_size: float,
    annotation_marker_stride: int,
    max_annotation_markers: int,
) -> None:
    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    points = data["points"][keep]
    labels = data["point_label"][keep]
    frame_order = data["frame_order"][keep]
    confidence = data["confidence"][keep]
    colors = colors[keep]
    out_alpha = out_alpha[keep]

    endpoint_vertices, edges = build_endpoint_vertices_and_edges(
        data,
        start_vertex_index=int(points.shape[0]),
        include_measurement_endpoint=include_measurement_endpoint,
        include_frame_endpoints=include_frame_endpoints,
        measurement_color=measurement_color,
        frame_endpoint_color=frame_endpoint_color,
    )
    marker_vertices = []
    if include_annotation_markers:
        marker_vertices, marker_edges = build_annotation_marker_vertices_and_edges(
            data,
            start_vertex_index=int(points.shape[0] + len(endpoint_vertices)),
            marker_color=annotation_marker_color,
            marker_size=annotation_marker_size,
            marker_stride=annotation_marker_stride,
            max_markers=max_annotation_markers,
        )
        edges.extend(marker_edges)

    vertex_count = int(points.shape[0] + len(endpoint_vertices) + len(marker_vertices))
    with output_ply.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment generated_by export_annotated_point_cloud_ply.py\n")
        f.write("comment label -1=ignore 0=background 1=femur_candidate 2=frame_endpoint 3=measurement_endpoint 4=annotation_marker\n")
        f.write(f"element vertex {vertex_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("property int label\n")
        f.write("property int frame_order\n")
        f.write("property float confidence\n")
        if edges:
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property uchar alpha\n")
            f.write("property int label\n")
        f.write("end_header\n")

        for p, rgb, a, label, order, conf in zip(
            points,
            colors,
            out_alpha,
            labels,
            frame_order,
            confidence,
            strict=True,
        ):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {int(a)} "
                f"{int(label)} {int(order)} {float(conf):.6f}\n"
            )

        for vertex in endpoint_vertices:
            f.write(
                f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                f"{vertex[3]} {vertex[4]} {vertex[5]} {vertex[6]} "
                f"{vertex[7]} {vertex[8]} {vertex[9]:.6f}\n"
            )

        for vertex in marker_vertices:
            f.write(
                f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                f"{vertex[3]} {vertex[4]} {vertex[5]} {vertex[6]} "
                f"{vertex[7]} {vertex[8]} {vertex[9]:.6f}\n"
            )

        for edge in edges:
            f.write(
                f"{edge[0]} {edge[1]} {edge[2]} {edge[3]} {edge[4]} "
                f"{edge[5]} {edge[6]}\n"
            )

    print(f"Saved annotated PLY: {output_ply}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export annotated pseudo-3D point cloud h5 to PLY. "
            "Only annotated femur-candidate points receive label color by default; "
            "other points remain grayscale context."
        )
    )
    parser.add_argument("--input_h5", type=Path, required=True)
    parser.add_argument("--output_ply", type=Path, required=True)
    parser.add_argument(
        "--annotation_only_output_ply",
        type=Path,
        default=None,
        help=(
            "Optional second PLY containing only point_label=1 points. "
            "This is useful for checking the exact 3D locations produced by "
            "the contour-in-BBox annotation mask."
        ),
    )
    parser.add_argument(
        "--background_mode",
        type=str,
        default="dim",
        choices=["dim", "gray", "hidden"],
        help=(
            "dim/gray keep all points with grayscale context. "
            "hidden exports only selected label values."
        ),
    )
    parser.add_argument(
        "--label_values",
        type=str,
        default="1",
        help="Comma-separated labels to export when --background_mode hidden.",
    )
    parser.add_argument(
        "--femur_color",
        type=str,
        default="255,64,32",
        help="RGB color for point_label=1 femur candidate points.",
    )
    parser.add_argument(
        "--background_alpha",
        type=int,
        default=90,
        help="Alpha for non-annotated context points.",
    )
    parser.add_argument(
        "--ignore_alpha",
        type=int,
        default=35,
        help="Alpha for point_label=-1 ignore points.",
    )
    parser.add_argument(
        "--include_measurement_endpoint",
        action="store_true",
        help="Append representative measurement endpoint vertices and an edge.",
    )
    parser.add_argument(
        "--include_frame_endpoints",
        action="store_true",
        help="Append all valid frame-wise endpoint vertices and edges.",
    )
    parser.add_argument(
        "--measurement_color",
        type=str,
        default="32,144,255",
        help="RGB color for representative measurement endpoint edge.",
    )
    parser.add_argument(
        "--frame_endpoint_color",
        type=str,
        default="255,220,32",
        help="RGB color for frame-wise endpoint edges.",
    )
    parser.add_argument(
        "--include_annotation_markers",
        action="store_true",
        help=(
            "Append small cross markers at point_label=1 positions so the "
            "annotated point locations are visible even when point rendering is small."
        ),
    )
    parser.add_argument(
        "--annotation_marker_color",
        type=str,
        default="0,255,160",
        help="RGB color for annotation position markers.",
    )
    parser.add_argument(
        "--annotation_marker_size",
        type=float,
        default=3.0,
        help="World-unit size of each annotation cross marker.",
    )
    parser.add_argument(
        "--annotation_marker_stride",
        type=int,
        default=8,
        help="Use every Nth annotated point for cross markers.",
    )
    parser.add_argument(
        "--max_annotation_markers",
        type=int,
        default=5000,
        help="Maximum number of annotation cross markers. Use 0 for no limit.",
    )
    parser.add_argument(
        "--point_mode",
        type=str,
        default=None,
        choices=["foreground", "grid", "dense"],
        help=argparse.SUPPRESS,
    )
    return parser


def parse_label_values(text: str) -> set[int]:
    values = set()
    for token in text.replace(" ", "").split(","):
        if token:
            values.add(int(token))
    if not values:
        values.add(LABEL_FEMUR_CANDIDATE)
    return values


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not (0 <= args.background_alpha <= 255):
        raise ValueError("--background_alpha must be in [0,255]")
    if not (0 <= args.ignore_alpha <= 255):
        raise ValueError("--ignore_alpha must be in [0,255]")
    if args.annotation_marker_size <= 0:
        raise ValueError("--annotation_marker_size must be > 0")
    if args.annotation_marker_stride < 1:
        raise ValueError("--annotation_marker_stride must be >= 1")
    if args.max_annotation_markers < 0:
        raise ValueError("--max_annotation_markers must be >= 0")

    ensure_numpy()
    data = load_annotated_h5(args.input_h5)
    femur_color = parse_color(args.femur_color)
    measurement_color = parse_color(args.measurement_color)
    frame_endpoint_color = parse_color(args.frame_endpoint_color)
    annotation_marker_color = parse_color(args.annotation_marker_color)
    label_values = parse_label_values(args.label_values)

    colors, out_alpha = make_point_colors(
        intensity=data["intensity"],
        alpha=data["alpha"],
        labels=data["point_label"],
        valid_mask=data["valid_mask"],
        femur_color=femur_color,
        background_mode=args.background_mode,
        background_alpha=args.background_alpha,
        ignore_alpha=args.ignore_alpha,
    )
    keep = select_points(
        data,
        background_mode=args.background_mode,
        label_values=label_values,
    )

    save_ply(
        args.output_ply,
        data=data,
        keep=keep,
        colors=colors,
        out_alpha=out_alpha,
        include_measurement_endpoint=args.include_measurement_endpoint,
        include_frame_endpoints=args.include_frame_endpoints,
        measurement_color=measurement_color,
        frame_endpoint_color=frame_endpoint_color,
        include_annotation_markers=args.include_annotation_markers,
        annotation_marker_color=annotation_marker_color,
        annotation_marker_size=args.annotation_marker_size,
        annotation_marker_stride=args.annotation_marker_stride,
        max_annotation_markers=args.max_annotation_markers,
    )

    annotation_keep = select_annotation_points(data)
    if args.annotation_only_output_ply is not None:
        annotation_colors, annotation_alpha = make_point_colors(
            intensity=data["intensity"],
            alpha=data["alpha"],
            labels=data["point_label"],
            valid_mask=data["valid_mask"],
            femur_color=femur_color,
            background_mode="hidden",
            background_alpha=args.background_alpha,
            ignore_alpha=args.ignore_alpha,
        )
        save_ply(
            args.annotation_only_output_ply,
            data=data,
            keep=annotation_keep,
            colors=annotation_colors,
            out_alpha=annotation_alpha,
            include_measurement_endpoint=False,
            include_frame_endpoints=False,
            measurement_color=measurement_color,
            frame_endpoint_color=frame_endpoint_color,
            include_annotation_markers=False,
            annotation_marker_color=annotation_marker_color,
            annotation_marker_size=args.annotation_marker_size,
            annotation_marker_stride=args.annotation_marker_stride,
            max_annotation_markers=args.max_annotation_markers,
        )

    labels = data["point_label"]
    frame_annotation = data.get("frame_annotation", {})
    valid_contours = frame_annotation.get("valid_contour")
    num_valid_contours = (
        int(np.sum(valid_contours.astype(bool)))
        if valid_contours is not None
        else None
    )
    print("Annotated point cloud PLY export:")
    print(f"  input_h5          : {args.input_h5}")
    print(f"  output_ply        : {args.output_ply}")
    if args.annotation_only_output_ply is not None:
        print(f"  annotation_only   : {args.annotation_only_output_ply}")
    print(f"  total_points      : {labels.shape[0]}")
    print(f"  exported_points   : {int(keep.sum())}")
    print(f"  femur_candidates  : {int(np.sum(labels == LABEL_FEMUR_CANDIDATE))}")
    print(f"  background_points : {int(np.sum(labels == LABEL_BACKGROUND))}")
    print(f"  ignore_points     : {int(np.sum(labels == LABEL_IGNORE))}")
    if num_valid_contours is not None:
        print(f"  valid_contours    : {num_valid_contours}")
    if args.include_annotation_markers:
        marker_source = int(np.sum(data["valid_mask"] & (labels == LABEL_FEMUR_CANDIDATE)))
        marker_count = max(0, (marker_source + max(1, args.annotation_marker_stride) - 1) // max(1, args.annotation_marker_stride))
        if args.max_annotation_markers > 0:
            marker_count = min(marker_count, args.max_annotation_markers)
        print(f"  annotation_markers: {marker_count}")


if __name__ == "__main__":
    main()
