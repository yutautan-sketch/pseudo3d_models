from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


def load_pseudo3d_h5(h5_path: Path):
    if not h5_path.exists():
        raise FileNotFoundError(f"Input h5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        required_keys = [
            "frame_mesh_vertices",
            "frame_mesh_faces",
            "frame_corners_world",
            "pred_tracking",
            "frame_indices",
        ]

        for key in required_keys:
            if key not in f:
                raise KeyError(f"Required key '{key}' not found in {h5_path}")

        vertices = f["frame_mesh_vertices"][:].astype(np.float32)
        faces = f["frame_mesh_faces"][:].astype(np.int64)
        frame_corners_world = f["frame_corners_world"][:].astype(np.float32)
        pred_tracking = f["pred_tracking"][:].astype(np.float32)
        frame_indices = f["frame_indices"][:].astype(np.int64)

        images = {}
        for key in ["raw_images", "local_encoder_images", "global_encoder_images"]:
            if key in f:
                images[key] = f[key][:]

        attrs = dict(f.attrs)

    return {
        "vertices": vertices,
        "faces": faces,
        "frame_corners_world": frame_corners_world,
        "pred_tracking": pred_tracking,
        "frame_indices": frame_indices,
        "images": images,
        "attrs": attrs,
    }


def make_pyvista_quad_mesh(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """
    Convert vertices [V,3] and quad faces [F,4] to PyVista PolyData.

    PyVista face format:
        [4, v0, v1, v2, v3, 4, v0, v1, v2, v3, ...]
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be [V,3], got {vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 4:
        raise ValueError(f"faces must be [F,4], got {faces.shape}")

    faces_pv = np.hstack(
        [
            np.full((faces.shape[0], 1), 4, dtype=np.int64),
            faces.astype(np.int64),
        ]
    ).ravel()

    mesh = pv.PolyData(vertices, faces_pv)
    return mesh


def normalize_image_for_texture(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 [H,W] for PyVista texture.

    Accepts:
      raw_images            : [H,W], uint8
      local/global inputs   : [1,H,W] or [H,W], float in [0,1]
    """
    img = np.asarray(image)

    if img.ndim == 3:
        # [1,H,W] or [C,H,W]
        if img.shape[0] == 1:
            img = img[0]
        else:
            # fallback: use first channel
            img = img[0]

    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image after squeeze, got {img.shape}")

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    img = np.nan_to_num(img)

    # Most saved model inputs are already [0,1].
    # If not, robustly normalize.
    min_v = float(img.min())
    max_v = float(img.max())

    if max_v <= 1.5 and min_v >= -0.5:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        denom = max(max_v - min_v, 1e-6)
        img = (img - min_v) / denom * 255.0

    return img.astype(np.uint8)


def make_textured_quad(corners: np.ndarray) -> pv.PolyData:
    """
    Create one quad mesh from four 3D corners and assign UV coordinates.

    corners:
      [4,3] ordered as:
        0: top-left
        1: top-right
        2: bottom-right
        3: bottom-left
    """
    if corners.shape != (4, 3):
        raise ValueError(f"corners must be [4,3], got {corners.shape}")

    faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    quad = pv.PolyData(corners.astype(np.float32), faces)

    # UV coordinates. Depending on image orientation, V may need flipping.
    # This mapping is a reasonable first convention:
    # top-left, top-right, bottom-right, bottom-left
    tcoords = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    quad.active_texture_coordinates = tcoords

    return quad


def add_textured_slices(
    plotter: pv.Plotter,
    frame_corners_world: np.ndarray,
    images: np.ndarray,
    texture_every: int = 1,
    texture_opacity: float = 1.0,
):
    """
    Add each frame as an image-textured quad.

    images can be:
      raw_images             [N,H,W]
      local_encoder_images   [N,1,H,W]
      global_encoder_images  [N,1,H,W]
    """
    if texture_every < 1:
        raise ValueError(f"texture_every must be >= 1, got {texture_every}")

    n_frames = frame_corners_world.shape[0]

    if len(images) != n_frames:
        raise ValueError(
            f"Number of images and frame corners differ: "
            f"len(images)={len(images)}, n_frames={n_frames}"
        )

    for i in range(0, n_frames, texture_every):
        img_uint8 = normalize_image_for_texture(images[i])
        texture = pv.Texture(img_uint8)

        quad = make_textured_quad(frame_corners_world[i])

        plotter.add_mesh(
            quad,
            texture=texture,
            opacity=texture_opacity,
            show_edges=False,
        )


def make_slice_only_mesh(frame_corners_world: np.ndarray) -> pv.PolyData:
    """
    Rebuild a mesh containing only one quad per frame.
    This ignores side faces between slices.
    """
    if frame_corners_world.ndim != 3 or frame_corners_world.shape[1:] != (4, 3):
        raise ValueError(
            f"frame_corners_world must be [N,4,3], got {frame_corners_world.shape}"
        )

    n = frame_corners_world.shape[0]
    vertices = frame_corners_world.reshape(n * 4, 3).astype(np.float32)

    faces = []
    for i in range(n):
        base = i * 4
        faces.append([base + 0, base + 1, base + 2, base + 3])

    faces = np.asarray(faces, dtype=np.int64)
    return make_pyvista_quad_mesh(vertices, faces)


def add_trajectory(plotter: pv.Plotter, pred_tracking: np.ndarray):
    """
    Add frame-center trajectory from pred_tracking[:, :3, 3].
    """
    trajectory = pred_tracking[:, :3, 3].astype(np.float32)

    if len(trajectory) >= 2:
        line = pv.lines_from_points(trajectory)
        plotter.add_mesh(line, line_width=4, label="trajectory")

    # Start/end markers
    start = pv.PolyData(trajectory[[0]])
    end = pv.PolyData(trajectory[[-1]])

    plotter.add_mesh(start, point_size=14, render_points_as_spheres=True, label="start")
    plotter.add_mesh(end, point_size=14, render_points_as_spheres=True, label="end")


def add_frame_corner_points(plotter: pv.Plotter, frame_corners_world: np.ndarray):
    points = frame_corners_world.reshape(-1, 3).astype(np.float32)
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(
        point_cloud,
        point_size=4,
        render_points_as_spheres=True,
        opacity=0.7,
        label="frame corners",
    )


def add_frame_labels(
    plotter: pv.Plotter,
    frame_corners_world: np.ndarray,
    frame_indices: np.ndarray,
    label_every: int,
):
    if label_every <= 0:
        return

    centers = frame_corners_world.mean(axis=1)

    label_points = []
    labels = []

    for i in range(0, len(centers), label_every):
        label_points.append(centers[i])
        labels.append(str(int(frame_indices[i])))

    if not label_points:
        return

    label_points = np.asarray(label_points, dtype=np.float32)
    plotter.add_point_labels(
        label_points,
        labels,
        font_size=12,
        point_size=6,
        shape_opacity=0.3,
        always_visible=True,
    )


def export_mesh(mesh: pv.PolyData, ply_path: Path | None, obj_path: Path | None):
    if ply_path is not None:
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(str(ply_path))
        print(f"Saved PLY: {ply_path}")

    if obj_path is not None:
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(str(obj_path))
        print(f"Saved OBJ: {obj_path}")


def visualize_and_save(
    *,
    mesh: pv.PolyData,
    frame_corners_world: np.ndarray,
    pred_tracking: np.ndarray,
    frame_indices: np.ndarray,
    screenshot_path: Path | None,
    show_edges: bool,
    show_trajectory: bool,
    show_corner_points: bool,
    label_every: int,
    off_screen: bool,
    window_size: tuple[int, int],
    show_textures: bool,
    texture_images: np.ndarray | None,
    texture_every: int,
    texture_opacity: float,
    hide_base_mesh: bool,
):
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)

    plotter.set_background("white")

    if not hide_base_mesh:
        plotter.add_mesh(
            mesh,
            opacity=0.25 if show_textures else 0.35,
            show_edges=show_edges,
            edge_color="black",
            label="slice mesh",
        )

    if show_textures:
        if texture_images is None:
            raise ValueError("--show_textures was specified, but texture_images is None")

        add_textured_slices(
            plotter=plotter,
            frame_corners_world=frame_corners_world,
            images=texture_images,
            texture_every=texture_every,
            texture_opacity=texture_opacity,
        )

    if show_trajectory:
        add_trajectory(plotter, pred_tracking)

    if show_corner_points:
        add_frame_corner_points(plotter, frame_corners_world)

    if label_every > 0:
        add_frame_labels(
            plotter,
            frame_corners_world=frame_corners_world,
            frame_indices=frame_indices,
            label_every=label_every,
        )

    plotter.add_axes()
    if not hide_base_mesh or show_trajectory or show_corner_points:
        plotter.add_legend()

    plotter.camera_position = "iso"
    plotter.reset_camera()

    if screenshot_path is not None:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path), auto_close=True)
        print(f"Saved screenshot: {screenshot_path}")
    else:
        plotter.show(auto_close=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize pseudo-3D frame polygon data from h5 and export PNG/PLY/OBJ."
        )
    )

    parser.add_argument(
        "--input_h5",
        type=Path,
        required=True,
        help="Path to pseudo-3D h5 file.",
    )

    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Output PNG screenshot path. If omitted, opens an interactive window if possible.",
    )

    parser.add_argument(
        "--export_ply",
        type=Path,
        default=None,
        help="Output PLY mesh path.",
    )

    parser.add_argument(
        "--export_obj",
        type=Path,
        default=None,
        help="Output OBJ mesh path.",
    )

    parser.add_argument(
        "--show_only_slices",
        action="store_true",
        help=(
            "Ignore saved connected side faces and rebuild mesh with one quad per frame. "
            "Useful to check slice placement only."
        ),
    )

    parser.add_argument(
        "--show_edges",
        action="store_true",
        help="Show mesh edges.",
    )

    parser.add_argument(
        "--show_trajectory",
        action="store_true",
        help="Show trajectory from pred_tracking[:, :3, 3].",
    )

    parser.add_argument(
        "--show_corner_points",
        action="store_true",
        help="Show four corner points of each frame.",
    )

    parser.add_argument(
        "--label_every",
        type=int,
        default=0,
        help=(
            "Show frame index labels every N frames. "
            "0 disables labels. Labels use original video frame indices."
        ),
    )

    parser.add_argument(
        "--off_screen",
        action="store_true",
        help=(
            "Use off-screen rendering. Recommended for SSH/server environment "
            "when saving screenshots."
        ),
    )

    parser.add_argument(
        "--window_width",
        type=int,
        default=1400,
        help="PyVista window width.",
    )

    parser.add_argument(
        "--window_height",
        type=int,
        default=1000,
        help="PyVista window height.",
    )
    
    parser.add_argument(
        "--show_textures",
        action="store_true",
        help="Show each frame image as a texture on its corresponding 3D slice.",
    )

    parser.add_argument(
        "--texture_key",
        type=str,
        default="local_encoder_images",
        choices=["local_encoder_images", "global_encoder_images", "raw_images"],
        help=(
            "Image dataset in h5 to use as slice texture. "
            "Recommended: local_encoder_images."
        ),
    )

    parser.add_argument(
        "--texture_every",
        type=int,
        default=1,
        help="Show texture every N frames. Use >1 to reduce visual clutter.",
    )

    parser.add_argument(
        "--texture_opacity",
        type=float,
        default=1.0,
        help="Opacity of textured slice images.",
    )

    parser.add_argument(
        "--hide_base_mesh",
        action="store_true",
        help="Hide the semi-transparent polygon mesh and show only textured slices.",
    )

    args = parser.parse_args()

    data = load_pseudo3d_h5(args.input_h5)

    vertices = data["vertices"]
    faces = data["faces"]
    frame_corners_world = data["frame_corners_world"]
    pred_tracking = data["pred_tracking"]
    frame_indices = data["frame_indices"]
    images_dict = data["images"]

    print("Loaded pseudo-3D h5:")
    print(f"  input_h5           : {args.input_h5}")
    print(f"  vertices           : {vertices.shape}")
    print(f"  faces              : {faces.shape}")
    print(f"  frame_corners_world: {frame_corners_world.shape}")
    print(f"  pred_tracking      : {pred_tracking.shape}")
    print(f"  frame_indices      : {frame_indices.shape}")
    if len(frame_indices) > 0:
        print(f"  frame_indices[:10] : {frame_indices[:10].tolist()}")

    if args.show_only_slices:
        mesh = make_slice_only_mesh(frame_corners_world)
        print("Using slice-only mesh rebuilt from frame_corners_world.")
    else:
        mesh = make_pyvista_quad_mesh(vertices, faces)
        print("Using saved frame_mesh_vertices / frame_mesh_faces.")

    print("Mesh summary:")
    print(f"  n_points: {mesh.n_points}")
    print(f"  n_cells : {mesh.n_cells}")
    print(f"  bounds  : {mesh.bounds}")

    export_mesh(
        mesh,
        ply_path=args.export_ply,
        obj_path=args.export_obj,
    )

    # If screenshot is specified, off-screen is usually preferred in SSH environments.
    off_screen = args.off_screen or args.screenshot is not None

    texture_images = None
    if args.show_textures:
        if args.texture_key not in images_dict:
            available = list(images_dict.keys())
            raise KeyError(
                f"texture_key='{args.texture_key}' not found in h5. "
                f"Available image keys: {available}"
            )

        texture_images = images_dict[args.texture_key]
        print("Texture images:")
        print(f"  texture_key: {args.texture_key}")
        print(f"  shape      : {texture_images.shape}")
        print(f"  dtype      : {texture_images.dtype}")

    visualize_and_save(
        mesh=mesh,
        frame_corners_world=frame_corners_world,
        pred_tracking=pred_tracking,
        frame_indices=frame_indices,
        screenshot_path=args.screenshot,
        show_edges=args.show_edges,
        show_trajectory=args.show_trajectory,
        show_corner_points=args.show_corner_points,
        label_every=args.label_every,
        off_screen=off_screen,
        window_size=(args.window_width, args.window_height),
        show_textures=args.show_textures,
        texture_images=texture_images,
        texture_every=args.texture_every,
        texture_opacity=args.texture_opacity,
        hide_base_mesh=args.hide_base_mesh,
    )


if __name__ == "__main__":
    main()
