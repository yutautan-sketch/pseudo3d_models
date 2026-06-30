from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import xml.etree.ElementTree as ET


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def read_video_frames_gray(
    video_path: Path,
    stride: int = 1,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    indices = []
    frame_idx = 0
    kept = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx >= start_frame and (frame_idx - start_frame) % stride == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            indices.append(frame_idx)
            kept += 1

            if max_frames is not None and kept >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return (
        np.stack(frames, axis=0).astype(np.uint8),
        np.asarray(indices, dtype=np.int64),
        fps,
    )


def center_crop_frames(
    frames: np.ndarray,
    crop_size: int = 256,
    offset_y: int = 0,
    offset_x: int = 0,
) -> tuple[np.ndarray, dict]:
    return crop_frames_with_offset(
        frames,
        crop_size=crop_size,
        offset_y=offset_y,
        offset_x=offset_x,
    )


def crop_frames_with_offset(
    frames: np.ndarray,
    crop_size: int = 256,
    offset_y: int = 0,
    offset_x: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Crop crop_size x crop_size from frames with an offset from the image center.

    offset_y:
        Negative -> crop upward
        Positive -> crop downward

    offset_x:
        Negative -> crop left
        Positive -> crop right

    Offsets are in pixels of the current frame resolution.
    """
    n, h, w = frames.shape

    if h < crop_size or w < crop_size:
        out = resize_frames(frames, (crop_size, crop_size))
        return out, {
            "mode_effective": "offset_crop_fallback_resize",
            "resized_h": crop_size,
            "resized_w": crop_size,
            "crop_top": 0,
            "crop_left": 0,
            "crop_center_y": crop_size / 2.0,
            "crop_center_x": crop_size / 2.0,
            "offset_y": offset_y,
            "offset_x": offset_x,
            "scale": np.nan,
            "crop_clipped": True,
        }

    center_y = h / 2.0 + offset_y
    center_x = w / 2.0 + offset_x

    top = int(round(center_y - crop_size / 2.0))
    left = int(round(center_x - crop_size / 2.0))

    clipped = False

    if top < 0:
        top = 0
        clipped = True
    if left < 0:
        left = 0
        clipped = True
    if top + crop_size > h:
        top = h - crop_size
        clipped = True
    if left + crop_size > w:
        left = w - crop_size
        clipped = True

    out = frames[:, top:top + crop_size, left:left + crop_size]

    return out, {
        "mode_effective": "offset_crop",
        "resized_h": h,
        "resized_w": w,
        "crop_top": top,
        "crop_left": left,
        "crop_center_y": top + crop_size / 2.0,
        "crop_center_x": left + crop_size / 2.0,
        "offset_y": offset_y,
        "offset_x": offset_x,
        "scale": 1.0,
        "crop_clipped": clipped,
    }


def resize_frames(frames: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = size_hw
    resized = [
        cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return np.stack(resized, axis=0).astype(np.uint8)


def resize_whole_to_square(
    frames: np.ndarray,
    output_size: int = 256,
) -> tuple[np.ndarray, dict]:
    out = resize_frames(frames, (output_size, output_size))
    return out, {
        "mode_effective": "resize",
        "resized_h": output_size,
        "resized_w": output_size,
        "crop_top": 0,
        "crop_left": 0,
        "scale": np.nan,
    }


def resize_shorter_then_center_crop(
    frames: np.ndarray,
    target_short_side: int,
    crop_size: int = 256,
    offset_y: int = 0,
    offset_x: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Aspect-ratio preserving resize:
        min(H, W) -> target_short_side
    then crop crop_size x crop_size with an offset from the resized image center.
    """
    if target_short_side < crop_size:
        raise ValueError(
            f"target_short_side must be >= crop_size. "
            f"got target_short_side={target_short_side}, crop_size={crop_size}"
        )

    n, h, w = frames.shape
    short = min(h, w)
    scale = target_short_side / float(short)

    resized_h = int(round(h * scale))
    resized_w = int(round(w * scale))

    resized = resize_frames(frames, (resized_h, resized_w))

    cropped, meta = crop_frames_with_offset(
        resized,
        crop_size=crop_size,
        offset_y=offset_y,
        offset_x=offset_x,
    )

    meta.update(
        {
            "mode_effective": "resize_shorter_then_offset_crop",
            "resized_h": resized_h,
            "resized_w": resized_w,
            "scale": scale,
            "offset_y": offset_y,
            "offset_x": offset_x,
        }
    )

    return cropped, meta


def preprocess_candidate(
    frames: np.ndarray,
    mode: str,
    crop_size: int,
    target_short_side: Optional[int] = None,
    offset_y: int = 0,
    offset_x: int = 0,
) -> tuple[np.ndarray, dict]:
    if mode == "center_crop":
        return center_crop_frames(
            frames,
            crop_size=crop_size,
            offset_y=offset_y,
            offset_x=offset_x,
        )

    if mode == "resize":
        return resize_whole_to_square(frames, output_size=crop_size)

    if mode == "resize_shorter_then_center_crop":
        if target_short_side is None:
            raise ValueError(
                "target_short_side is required for resize_shorter_then_center_crop"
            )
        return resize_shorter_then_center_crop(
            frames,
            target_short_side=target_short_side,
            crop_size=crop_size,
            offset_y=offset_y,
            offset_x=offset_x,
        )

    raise ValueError(f"Unknown mode: {mode}")


def canny_edge_density(frame: np.ndarray) -> float:
    # Slight blur reduces isolated noise.
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return float((edges > 0).mean())


def otsu_foreground_ratio(frame: np.ndarray) -> float:
    # Otsu can fail on nearly flat images; still useful as a rough indicator.
    _, th = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((th > 0).mean())


def high_intensity_ratio(frame: np.ndarray, threshold: int = 180) -> float:
    return float((frame >= threshold).mean())


def non_dark_ratio(frame: np.ndarray, threshold: int = 20) -> float:
    return float((frame >= threshold).mean())


def parse_pascal_voc_xml(xml_path: Path, class_filter: Optional[set[str]] = None) -> list[dict]:
    """
    Parse Pascal VOC XML.

    Returns:
        [
          {
            "name": str,
            "bbox": np.ndarray [xmin, ymin, xmax, ymax], float32,
            "xml_width": int | None,
            "xml_height": int | None,
          },
          ...
        ]
    """
    if not xml_path.exists():
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    xml_width = None
    xml_height = None

    if size_node is not None:
        w_node = size_node.find("width")
        h_node = size_node.find("height")
        if w_node is not None and h_node is not None:
            xml_width = int(float(w_node.text))
            xml_height = int(float(h_node.text))

    objects = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        bbox_node = obj.find("bndbox")

        if name_node is None or bbox_node is None:
            continue

        name = name_node.text.strip() if name_node.text else ""

        if class_filter is not None and name not in class_filter:
            continue

        def get_float(tag: str) -> float:
            node = bbox_node.find(tag)
            if node is None or node.text is None:
                raise ValueError(f"Missing {tag} in {xml_path}")
            return float(node.text)

        xmin = get_float("xmin")
        ymin = get_float("ymin")
        xmax = get_float("xmax")
        ymax = get_float("ymax")

        # 保険：座標順序を正規化
        x0, x1 = sorted([xmin, xmax])
        y0, y1 = sorted([ymin, ymax])

        objects.append(
            {
                "name": name,
                "bbox": np.asarray([x0, y0, x1, y1], dtype=np.float32),
                "xml_width": xml_width,
                "xml_height": xml_height,
            }
        )

    return objects


def load_bboxes_for_sampled_frames(
    *,
    annotation_root: Path,
    video_path: Path,
    frame_indices: np.ndarray,
    class_filter: Optional[set[str]] = None,
) -> dict[int, list[dict]]:
    """
    Load BBoxes for sampled frame indices.

    XML file naming:
        {video_name}_{1_based_frame_number:05d}.xml

    Returns:
        dict:
          frame_idx_0based -> list of parsed objects
    """
    video_name = video_path.stem
    video_ann_dir = annotation_root / video_name

    out: dict[int, list[dict]] = {}

    if not video_ann_dir.exists():
        return out

    for frame_idx in frame_indices:
        xml_frame_num = int(frame_idx) + 1
        xml_path = video_ann_dir / f"{video_name}_{xml_frame_num:05d}.xml"

        objs = parse_pascal_voc_xml(xml_path, class_filter=class_filter)
        if objs:
            out[int(frame_idx)] = objs

    return out


def transform_bbox_to_crop_space(
    bbox_raw: np.ndarray,
    *,
    mode: str,
    raw_h: int,
    raw_w: int,
    crop_size: int,
    meta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform raw bbox to candidate crop coordinate.

    Returns:
        bbox_before_clip:
            transformed bbox in crop coordinate, before clipping.
        bbox_clipped:
            clipped bbox in [0, crop_size].
    """
    xmin, ymin, xmax, ymax = bbox_raw.astype(np.float32)

    if mode == "resize":
        sx = crop_size / float(raw_w)
        sy = crop_size / float(raw_h)

        bbox = np.asarray(
            [
                xmin * sx,
                ymin * sy,
                xmax * sx,
                ymax * sy,
            ],
            dtype=np.float32,
        )

    else:
        scale = float(meta.get("scale", 1.0))
        if not np.isfinite(scale):
            scale = 1.0

        crop_left = float(meta.get("crop_left", 0.0))
        crop_top = float(meta.get("crop_top", 0.0))

        bbox = np.asarray(
            [
                xmin * scale - crop_left,
                ymin * scale - crop_top,
                xmax * scale - crop_left,
                ymax * scale - crop_top,
            ],
            dtype=np.float32,
        )

    clipped = bbox.copy()
    clipped[0] = np.clip(clipped[0], 0, crop_size)
    clipped[2] = np.clip(clipped[2], 0, crop_size)
    clipped[1] = np.clip(clipped[1], 0, crop_size)
    clipped[3] = np.clip(clipped[3], 0, crop_size)

    # clip後に幅/高さが負にならないように正規化
    if clipped[2] < clipped[0]:
        clipped[2] = clipped[0]
    if clipped[3] < clipped[1]:
        clipped[3] = clipped[1]

    return bbox, clipped


def bbox_area(b: np.ndarray) -> float:
    w = max(float(b[2] - b[0]), 0.0)
    h = max(float(b[3] - b[1]), 0.0)
    return w * h


def compute_bbox_metrics_for_candidate(
    *,
    bboxes_by_frame: dict[int, list[dict]],
    frame_indices: np.ndarray,
    mode: str,
    raw_h: int,
    raw_w: int,
    crop_size: int,
    meta: dict,
) -> dict:
    """
    Evaluate how annotated BBoxes are represented after preprocessing.

    Metrics:
        bbox_visible_ratio:
            fraction of transformed bbox area remaining inside crop.

        bbox_crop_area_ratio:
            visible bbox area / crop area.

        bbox_crop_diag_ratio:
            visible bbox diagonal / crop diagonal.

        bbox_score_area:
            visible_ratio * sqrt(area_ratio)

        bbox_score_diag:
            visible_ratio * diag_ratio
    """
    visible_ratios = []
    area_ratios = []
    width_ratios = []
    height_ratios = []
    diag_ratios = []
    score_area = []
    score_diag = []
    fully_inside = []
    partially_visible = []
    missing = []

    num_annotated_frames = 0
    num_bboxes = 0

    crop_area = float(crop_size * crop_size)
    crop_diag = float(np.sqrt(crop_size**2 + crop_size**2))

    for frame_idx in frame_indices:
        objs = bboxes_by_frame.get(int(frame_idx), [])
        if not objs:
            continue

        num_annotated_frames += 1

        for obj in objs:
            bbox_raw = obj["bbox"]
            bbox_before_clip, bbox_clipped = transform_bbox_to_crop_space(
                bbox_raw,
                mode=mode,
                raw_h=raw_h,
                raw_w=raw_w,
                crop_size=crop_size,
                meta=meta,
            )

            original_area = bbox_area(bbox_before_clip)
            visible_area = bbox_area(bbox_clipped)

            if original_area <= 1e-6:
                continue

            num_bboxes += 1

            vr = visible_area / original_area
            cw = max(float(bbox_clipped[2] - bbox_clipped[0]), 0.0)
            ch = max(float(bbox_clipped[3] - bbox_clipped[1]), 0.0)
            diag = float(np.sqrt(cw * cw + ch * ch))

            ar = visible_area / crop_area
            wr = cw / crop_size
            hr = ch / crop_size
            dr = diag / crop_diag

            visible_ratios.append(vr)
            area_ratios.append(ar)
            width_ratios.append(wr)
            height_ratios.append(hr)
            diag_ratios.append(dr)

            score_area.append(vr * np.sqrt(max(ar, 0.0)))
            score_diag.append(vr * dr)

            fully_inside.append(vr >= 0.95)
            partially_visible.append(vr > 0.0)
            missing.append(vr <= 0.0)

    if num_bboxes == 0:
        return {
            "num_annotated_frames": num_annotated_frames,
            "num_bboxes": 0,
            "bbox_visible_ratio_mean": np.nan,
            "bbox_visible_ratio_min": np.nan,
            "bbox_crop_area_ratio_mean": np.nan,
            "bbox_crop_width_ratio_mean": np.nan,
            "bbox_crop_height_ratio_mean": np.nan,
            "bbox_crop_diag_ratio_mean": np.nan,
            "bbox_score_area_mean": np.nan,
            "bbox_score_diag_mean": np.nan,
            "bbox_fully_inside_rate": np.nan,
            "bbox_partially_visible_rate": np.nan,
            "bbox_missing_rate": np.nan,
        }

    visible_ratios = np.asarray(visible_ratios, dtype=np.float32)
    area_ratios = np.asarray(area_ratios, dtype=np.float32)
    width_ratios = np.asarray(width_ratios, dtype=np.float32)
    height_ratios = np.asarray(height_ratios, dtype=np.float32)
    diag_ratios = np.asarray(diag_ratios, dtype=np.float32)
    score_area = np.asarray(score_area, dtype=np.float32)
    score_diag = np.asarray(score_diag, dtype=np.float32)
    fully_inside = np.asarray(fully_inside, dtype=bool)
    partially_visible = np.asarray(partially_visible, dtype=bool)
    missing = np.asarray(missing, dtype=bool)

    return {
        "num_annotated_frames": num_annotated_frames,
        "num_bboxes": num_bboxes,
        "bbox_visible_ratio_mean": float(np.mean(visible_ratios)),
        "bbox_visible_ratio_min": float(np.min(visible_ratios)),
        "bbox_crop_area_ratio_mean": float(np.mean(area_ratios)),
        "bbox_crop_width_ratio_mean": float(np.mean(width_ratios)),
        "bbox_crop_height_ratio_mean": float(np.mean(height_ratios)),
        "bbox_crop_diag_ratio_mean": float(np.mean(diag_ratios)),
        "bbox_score_area_mean": float(np.mean(score_area)),
        "bbox_score_diag_mean": float(np.mean(score_diag)),
        "bbox_fully_inside_rate": float(np.mean(fully_inside)),
        "bbox_partially_visible_rate": float(np.mean(partially_visible)),
        "bbox_missing_rate": float(np.mean(missing)),
    }


def draw_bbox_on_frame(frame: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    """
    Draw clipped BBoxes on a grayscale frame.

    frame:
        uint8 [H,W]
    boxes:
        list of [xmin,ymin,xmax,ymax] in frame coordinate
    """
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

    return img


def make_bbox_montage(
    *,
    frames: np.ndarray,
    frame_indices: np.ndarray,
    bboxes_by_frame: dict[int, list[dict]],
    mode: str,
    raw_h: int,
    raw_w: int,
    crop_size: int,
    meta: dict,
    output_path: Path,
    title: str,
    num_images: int = 16,
    cols: int = 4,
):
    """
    Create montage with transformed/clipped BBoxes overlaid.
    Only sampled frames are shown; if frame has no annotation, no box is drawn.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idxs = select_montage_indices(len(frames), num_images)

    h, w = frames.shape[1:]
    rows = int(np.ceil(len(idxs) / cols))

    pad = 8
    label_h = 24
    title_h = 42

    canvas_h = title_h + rows * (h + label_h + pad) + pad
    canvas_w = cols * (w + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        title[:160],
        (pad, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for k, local_i in enumerate(idxs):
        frame = frames[local_i]
        raw_frame_idx = int(frame_indices[local_i])

        boxes_to_draw = []
        for obj in bboxes_by_frame.get(raw_frame_idx, []):
            _, clipped = transform_bbox_to_crop_space(
                obj["bbox"],
                mode=mode,
                raw_h=raw_h,
                raw_w=raw_w,
                crop_size=crop_size,
                meta=meta,
            )
            boxes_to_draw.append(clipped)

        img_bgr = draw_bbox_on_frame(frame, boxes_to_draw)

        r = k // cols
        c = k % cols

        y0 = title_h + pad + r * (h + label_h + pad)
        x0 = pad + c * (w + pad)

        canvas[y0:y0 + h, x0:x0 + w] = img_bgr

        label = f"idx={raw_frame_idx}, boxes={len(boxes_to_draw)}"
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def compute_candidate_metrics(frames: np.ndarray) -> dict:
    """
    frames: uint8 [N,256,256]
    Returns aggregate metrics over frames.
    """
    x = frames.astype(np.float32)

    means = x.mean(axis=(1, 2))
    stds = x.std(axis=(1, 2))
    p5 = np.percentile(x, 5, axis=(1, 2))
    p95 = np.percentile(x, 95, axis=(1, 2))
    contrasts = p95 - p5

    edge_densities = np.array([canny_edge_density(f) for f in frames], dtype=np.float32)
    otsu_ratios = np.array([otsu_foreground_ratio(f) for f in frames], dtype=np.float32)
    non_dark_ratios = np.array([non_dark_ratio(f) for f in frames], dtype=np.float32)
    high_ratios = np.array([high_intensity_ratio(f) for f in frames], dtype=np.float32)

    if len(frames) >= 2:
        diffs = np.abs(x[1:] - x[:-1]).mean(axis=(1, 2))
        frame_diff_mean = float(diffs.mean())
        frame_diff_std = float(diffs.std())
    else:
        frame_diff_mean = np.nan
        frame_diff_std = np.nan

    return {
        "frame_mean_mean": float(means.mean()),
        "frame_mean_std": float(means.std()),
        "frame_std_mean": float(stds.mean()),
        "contrast_p95_p5_mean": float(contrasts.mean()),
        "edge_density_mean": float(edge_densities.mean()),
        "edge_density_std": float(edge_densities.std()),
        "otsu_foreground_ratio_mean": float(otsu_ratios.mean()),
        "non_dark_ratio_mean": float(non_dark_ratios.mean()),
        "high_intensity_ratio_mean": float(high_ratios.mean()),
        "frame_diff_mean": frame_diff_mean,
        "frame_diff_std": frame_diff_std,
    }


def select_montage_indices(n: int, num_images: int) -> np.ndarray:
    if n <= num_images:
        return np.arange(n)
    return np.linspace(0, n - 1, num_images).round().astype(np.int64)


def make_montage(
    frames: np.ndarray,
    frame_indices: np.ndarray,
    output_path: Path,
    title: str,
    num_images: int = 16,
    cols: int = 4,
):
    """
    Create montage PNG using OpenCV only.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idxs = select_montage_indices(len(frames), num_images)
    selected = frames[idxs]
    selected_frame_indices = frame_indices[idxs]

    h, w = selected.shape[1:]
    rows = int(np.ceil(len(selected) / cols))

    pad = 8
    label_h = 24
    title_h = 36

    canvas_h = title_h + rows * (h + label_h + pad) + pad
    canvas_w = cols * (w + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        title,
        (pad, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for k, frame in enumerate(selected):
        r = k // cols
        c = k % cols

        y0 = title_h + pad + r * (h + label_h + pad)
        x0 = pad + c * (w + pad)

        img_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        canvas[y0:y0 + h, x0:x0 + w] = img_bgr

        label = f"idx={int(selected_frame_indices[k])}"
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def write_metrics_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def find_videos(video_dir: Path, recursive: bool = False) -> list[Path]:
    if video_dir.is_file() and video_dir.suffix.lower() in VIDEO_EXTS:
        return [video_dir]

    pattern = "**/*" if recursive else "*"
    videos = [
        p for p in video_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze local input preprocessing candidates for DualTrack. "
            "Creates montages and CSV metrics for center_crop, "
            "resize_shorter_then_center_crop, and resize."
        )
    )

    parser.add_argument(
        "--video_dir",
        type=Path,
        required=True,
        help="Input video file or directory containing videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output report directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search videos under video_dir.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Frame sampling stride.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First frame index.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
        help="Maximum sampled frames per video. Use -1 for no limit.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="Final local input crop/resize size.",
    )
    parser.add_argument(
        "--target_short_sides",
        type=int,
        nargs="+",
        default=[256, 320, 384, 448, 480, 512, 640],
        help="Candidate short-side sizes for resize_shorter_then_center_crop.",
    )
    parser.add_argument(
        "--include_center_crop",
        action="store_true",
        help="Also evaluate direct center_crop from raw frames.",
    )
    parser.add_argument(
        "--include_resize",
        action="store_true",
        help="Also evaluate direct resize whole frame to crop_size x crop_size.",
    )
    parser.add_argument(
        "--montage_frames",
        type=int,
        default=16,
        help="Number of frames to show in montage.",
    )
    parser.add_argument(
        "--montage_cols",
        type=int,
        default=4,
        help="Number of columns in montage.",
    )
    parser.add_argument(
        "--crop_offsets_y",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Candidate crop offsets in y direction after resize. "
            "Negative values move the crop upward. Unit: pixels."
        ),
    )
    parser.add_argument(
        "--crop_offsets_x",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Candidate crop offsets in x direction after resize. "
            "Negative values move the crop left. Unit: pixels."
        ),
    )
    parser.add_argument(
        "--annotation_root",
        type=Path,
        default=None,
        help=(
            "Root directory of Pascal VOC XML annotations. "
            "Expected: annotation_root/video_name/video_name_00001.xml"
        ),
    )
    parser.add_argument(
        "--annotation_classes",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional class names to evaluate. "
            "If omitted, all objects in XML are used."
        ),
    )
    parser.add_argument(
        "--save_bbox_montage",
        action="store_true",
        help="Save montage images with transformed BBoxes overlaid.",
    )

    args = parser.parse_args()

    max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames

    videos = find_videos(args.video_dir, recursive=args.recursive)
    if not videos:
        raise RuntimeError(f"No videos found under {args.video_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"Found {len(videos)} videos.")
    for video_path in videos:
        print(f"\nProcessing: {video_path}")

        frames, frame_indices, fps = read_video_frames_gray(
            video_path,
            stride=args.stride,
            start_frame=args.start_frame,
            max_frames=max_frames,
        )

        n, raw_h, raw_w = frames.shape
        video_stem = video_path.stem
        video_out_dir = args.output_dir / video_stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  raw shape: {frames.shape}, fps={fps}")
        
        class_filter = set(args.annotation_classes) if args.annotation_classes else None

        bboxes_by_frame = {}
        if args.annotation_root is not None:
            bboxes_by_frame = load_bboxes_for_sampled_frames(
                annotation_root=args.annotation_root,
                video_path=video_path,
                frame_indices=frame_indices,
                class_filter=class_filter,
            )

            num_annotated_frames_raw = len(bboxes_by_frame)
            num_boxes_raw = sum(len(v) for v in bboxes_by_frame.values())

            print(
                f"  annotations: frames={num_annotated_frames_raw}, "
                f"boxes={num_boxes_raw}"
            )

        candidates = []

        # Direct center crop from raw frames, with offset candidates.
        if args.include_center_crop:
            for oy in args.crop_offsets_y:
                for ox in args.crop_offsets_x:
                    candidates.append(
                        {
                            "mode": "center_crop",
                            "target_short_side": None,
                            "offset_y": oy,
                            "offset_x": ox,
                        }
                    )

        # Resize shorter side, then offset crop.
        for s in args.target_short_sides:
            for oy in args.crop_offsets_y:
                for ox in args.crop_offsets_x:
                    candidates.append(
                        {
                            "mode": "resize_shorter_then_center_crop",
                            "target_short_side": s,
                            "offset_y": oy,
                            "offset_x": ox,
                        }
                    )

        # Whole-frame resize does not use crop offsets.
        if args.include_resize:
            candidates.append(
                {
                    "mode": "resize",
                    "target_short_side": None,
                    "offset_y": 0,
                    "offset_x": 0,
                }
            )

        for cand in candidates:
            mode = cand["mode"]
            target_short_side = cand["target_short_side"]
            offset_y = cand["offset_y"]
            offset_x = cand["offset_x"]

            if mode == "resize_shorter_then_center_crop":
                candidate_name = (
                    f"target_short_{target_short_side}"
                    f"_oy{offset_y:+d}_ox{offset_x:+d}"
                )
            elif mode == "center_crop":
                candidate_name = f"center_crop_oy{offset_y:+d}_ox{offset_x:+d}"
            else:
                candidate_name = mode

            # Make directory names filesystem-friendly.
            candidate_name = candidate_name.replace("+", "p").replace("-", "m")

            candidate_dir = video_out_dir / candidate_name

            try:
                processed, meta = preprocess_candidate(
                    frames,
                    mode=mode,
                    crop_size=args.crop_size,
                    target_short_side=target_short_side,
                    offset_y=offset_y,
                    offset_x=offset_x,
                )
            except Exception as e:
                print(f"  [SKIP] {candidate_name}: {e}")
                continue

            metrics = compute_candidate_metrics(processed)
            bbox_metrics = {}
            if args.annotation_root is not None:
                bbox_metrics = compute_bbox_metrics_for_candidate(
                    bboxes_by_frame=bboxes_by_frame,
                    frame_indices=frame_indices,
                    mode=mode,
                    raw_h=raw_h,
                    raw_w=raw_w,
                    crop_size=args.crop_size,
                    meta=meta,
                )

            row = {
                "video": video_path.name,
                "video_path": str(video_path),
                "candidate": candidate_name,
                "mode": mode,
                "target_short_side": target_short_side if target_short_side is not None else "",
                "offset_y": offset_y,
                "offset_x": offset_x,
                "raw_num_frames": n,
                "raw_height": raw_h,
                "raw_width": raw_w,
                "fps": fps,
                "stride": args.stride,
                "crop_size": args.crop_size,
                **meta,
                **metrics,
                **bbox_metrics,
            }

            summary_rows.append(row)

            # Save per-candidate metrics
            write_metrics_csv(candidate_dir / "metrics.csv", row)

            # Save montage
            montage_title = (
                f"{video_path.name} | {candidate_name} | "
                f"raw={raw_h}x{raw_w} -> local={processed.shape[1]}x{processed.shape[2]}"
            )
            make_montage(
                processed,
                frame_indices=frame_indices,
                output_path=candidate_dir / "montage.png",
                title=montage_title,
                num_images=args.montage_frames,
                cols=args.montage_cols,
            )

            msg = (
                f"  {candidate_name}: "
                f"edge={metrics['edge_density_mean']:.4f}, "
                f"std={metrics['frame_std_mean']:.2f}, "
                f"contrast={metrics['contrast_p95_p5_mean']:.2f}, "
                f"non_dark={metrics['non_dark_ratio_mean']:.3f}"
            )

            if bbox_metrics:
                msg += (
                    f", bbox_vis={bbox_metrics['bbox_visible_ratio_mean']:.3f}, "
                    f"bbox_diag={bbox_metrics['bbox_crop_diag_ratio_mean']:.3f}, "
                    f"bbox_score={bbox_metrics['bbox_score_diag_mean']:.3f}, "
                    f"inside={bbox_metrics['bbox_fully_inside_rate']:.3f}"
                )

            print(msg)

            if args.save_bbox_montage and args.annotation_root is not None:
                bbox_title = (
                    f"{video_path.name} | {candidate_name} | "
                    f"BBox overlay | raw={raw_h}x{raw_w}"
                )
                make_bbox_montage(
                    frames=processed,
                    frame_indices=frame_indices,
                    bboxes_by_frame=bboxes_by_frame,
                    mode=mode,
                    raw_h=raw_h,
                    raw_w=raw_w,
                    crop_size=args.crop_size,
                    meta=meta,
                    output_path=candidate_dir / "montage_bbox.png",
                    title=bbox_title,
                    num_images=args.montage_frames,
                    cols=args.montage_cols,
                )

    # Write summary CSV
    if summary_rows:
        summary_path = args.output_dir / "summary.csv"
        fieldnames = list(summary_rows[0].keys())

        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nSaved summary CSV: {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()