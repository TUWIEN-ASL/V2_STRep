"""Overlay SAM3D axes and an optional reconstructed mesh silhouette on RGB."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def quaternion_matrix(quaternion):
    """Match PyTorch3D quaternion_to_matrix for a wxyz quaternion."""
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Pose contains an invalid quaternion.")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def draw_pose(image, rotation, translation, intrinsics, axis_length):
    # Upstream SAM3D applies rotation to row vectors. Flip X/Y into OpenCV.
    local_points = np.vstack([np.zeros(3), np.eye(3) * axis_length])
    camera_points = (local_points @ rotation + translation) * [-1, -1, 1]
    # Axis length is already in pose units; do not apply the object's scale again.
    if not np.isfinite(camera_points).all() or camera_points[0, 2] <= 0:
        raise ValueError("Object origin must be finite and in front of the camera.")
    visible = camera_points[:, 2] > 1e-6
    pixels = np.full((4, 2), np.nan)
    projected = camera_points[visible] @ intrinsics.T
    pixels[visible] = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(pixels[0]).all():
        raise ValueError("Cannot project the object origin.")
    origin = tuple(np.rint(pixels[0]).astype(int))
    height, width = image.shape[:2]
    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]  # OpenCV BGR
    for index, (label, color) in enumerate(zip("XYZ", colors), start=1):
        if not np.isfinite(pixels[index]).all():
            continue
        endpoint = tuple(np.rint(pixels[index]).astype(int))
        intersects, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
        if intersects:
            cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.12)
        if 0 <= endpoint[0] < width and 0 <= endpoint[1] < height:
            label_position = (endpoint[0] + 5, endpoint[1] - 5)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
    cv2.circle(image, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)
    return image


def mesh_silhouette(mesh_path, rotation, translation, scale, intrinsics, image_shape):
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    # Undo SAM3D's GLB-only Z-up to Y-up export rotation.
    export_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    vertices = np.asarray(mesh.vertices) @ export_rotation.T
    points = ((vertices * scale) @ rotation + translation) * [-1, -1, 1]
    if len(mesh.faces) == 0 or not np.isfinite(points).all():
        raise ValueError("Mesh must contain faces and finite camera-frame vertices.")
    if np.any(points[:, 2] <= 1e-6):
        raise ValueError("Mesh reaches behind the camera; check the mesh/pose pairing.")
    projected = points @ intrinsics.T
    pixels = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e7:
        raise ValueError("Mesh projects outside the supported range; check pose/intrinsics.")
    triangles = np.rint(pixels[mesh.faces]).astype(np.int32)
    height, width = image_shape[:2]
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    intersects = ((high[:, 0] >= 0) & (low[:, 0] < width)
                  & (high[:, 1] >= 0) & (low[:, 1] < height))
    silhouette = np.zeros((height, width), dtype=np.uint8)
    # Union individual triangles; this is a full-object silhouette, without scene occlusion.
    for triangle in triangles[intersects]:
        cv2.fillConvexPoly(silhouette, triangle, 255)
    return silhouette


def draw_silhouette(image, silhouette, opacity, reference_mask=None):
    color = (255, 255, 0)  # Cyan in BGR.
    foreground = silhouette > 0
    image[foreground] = np.rint(
        (1 - opacity) * image[foreground] + opacity * np.array(color)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    cv2.putText(image, "Cyan: reconstructed mesh", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if reference_mask is not None:
        contours, _ = cv2.findContours(reference_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        reference = reference_mask > 0
        union = np.count_nonzero(foreground | reference)
        iou = np.count_nonzero(foreground & reference) / union if union else 0.0
        cv2.putText(image, f"Yellow: input mask | silhouette IoU: {iou:.3f}", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Original RGB image")
    parser.add_argument("--pose", type=Path, required=True, help="pose.npz with its saved camera matrix")
    parser.add_argument("--output", type=Path, help="Default: pose_overlay.png, or mesh_overlay.png with --mesh")
    parser.add_argument("--axis-length", type=float, default=0.05,
                        help="Axis length in pose units: meters for RGB-D, relative for RGB-only")
    parser.add_argument("--mesh", type=Path, help="SAM3D mesh.glb from the same run as --pose")
    parser.add_argument("--mask", type=Path, help="Original binary object mask for comparison")
    parser.add_argument("--opacity", type=float, default=0.35, help="Mesh silhouette opacity, 0 to 1")
    args = parser.parse_args()
    if not np.isfinite(args.axis_length) or args.axis_length <= 0:
        parser.error("--axis-length must be positive and finite.")
    if not 0 <= args.opacity <= 1:
        parser.error("--opacity must be between 0 and 1.")
    if args.mask is not None and args.mesh is None:
        parser.error("--mask requires --mesh.")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    with np.load(args.pose, allow_pickle=False) as pose:
        if "intrinsics_pixels" not in pose:
            parser.error("This pose has no saved camera. Rerun the updated demo_full.py.")
        if "image_size_hw" in pose and tuple(pose["image_size_hw"]) != image.shape[:2]:
            parser.error("Use the original image resolution saved with this pose.")
        units = str(pose.get("translation_units", "unknown units"))
        camera_source = str(pose.get("intrinsics_source", pose.get("depth_source", "unknown")))
        rotation = quaternion_matrix(pose["rotation"])
        translation = np.asarray(pose["translation"], dtype=float).reshape(3)
        intrinsics = np.asarray(pose["intrinsics_pixels"], dtype=float).reshape(3, 3)
        scale = np.asarray(pose["scale"], dtype=float).reshape(-1) if args.mesh else None
    silhouette = None
    if args.mesh is not None:
        reference_mask = None
        if args.mask is not None:
            reference_mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
            if reference_mask is None or reference_mask.shape != image.shape[:2]:
                parser.error("--mask must be readable and match the original image dimensions.")
            reference_mask = (reference_mask > 0).astype(np.uint8) * 255
        silhouette = mesh_silhouette(args.mesh, rotation, translation, scale,
                                     intrinsics, image.shape)
        image = draw_silhouette(image, silhouette, args.opacity, reference_mask)
    overlay = draw_pose(image, rotation, translation, intrinsics, args.axis_length)
    cv2.putText(overlay, f"Camera: {camera_source} | axes: {args.axis_length:g} {units}",
                (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    default_name = "mesh_overlay.png" if args.mesh else "pose_overlay.png"
    output_path = args.output or args.pose.parent / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"Could not save overlay: {output_path}")
    if silhouette is not None:
        mask_path = output_path.with_name(output_path.stem + "_mask.png")
        if not cv2.imwrite(str(mask_path), silhouette):
            raise OSError(f"Could not save reconstructed mask: {mask_path}")
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
