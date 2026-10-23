"""Align saved RGB-only SAM3D scene points to measured RGB-D, then correct its pose.

CPU only; same image pixels provide correspondences. This fits scene geometry,
not mesh geometry. Inspect both scene residuals and the corrected mesh overlay.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation
from visualize_pose import draw_pose, draw_silhouette, quaternion_matrix


def similarity(source, target, weights=None):
    weights = np.ones(len(source)) if weights is None else weights
    if weights.sum() <= 0:
        return None
    weights = weights / weights.sum()
    cx, cy = weights @ source, weights @ target
    x, y = source - cx, target - cy
    u, singular, vt = np.linalg.svd((x * weights[:, None]).T @ y)
    if singular[0] <= 0 or singular[1] < 1e-6 * singular[0]:
        return None  # Reject collinear points; a noncollinear plane is sufficient.
    sign = np.array([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = (u * sign) @ vt
    scale = (singular * sign).sum() / (weights * (x*x).sum(1)).sum()
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale, rotation, cy - scale * cx @ rotation


def residuals(fit, source, target):
    scale, rotation, shift = fit
    return np.linalg.norm(scale * source @ rotation + shift - target, axis=1)


def robust_similarity(source, target, threshold, rng):
    best, score = None, (-1, -np.inf)
    for iteration in range(501):
        ids = np.arange(len(source)) if iteration == 0 else rng.choice(len(source), 3, replace=False)
        fit = similarity(source[ids], target[ids])
        if fit is None:
            continue
        errors = residuals(fit, source, target)
        inside = errors < threshold
        candidate = (int(inside.sum()), -float(np.median(errors[inside])) if inside.any() else -np.inf)
        if candidate > score:
            best, score = fit, candidate
    if best is None or score[0] < max(12, 0.25 * len(source)):
        raise ValueError("No supported similarity: fewer than 25% inliers (minimum 12).")
    for _ in range(8):
        errors = residuals(best, source, target)
        weights = np.minimum(1, threshold / np.maximum(errors, 1e-12)) * (errors < 3 * threshold)
        fit = similarity(source, target, weights)
        if fit is None:
            break
        best = fit
    if (residuals(best, source, target) < threshold).sum() < max(12, 0.25 * len(source)):
        raise ValueError("Refined similarity lost its inlier support.")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-run", type=Path, required=True)
    parser.add_argument("--rgbd-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--inlier-m", type=float, default=0.01)
    parser.add_argument("--max-points", type=int, default=5000)
    args = parser.parse_args()
    cameras = [json.loads((run / "camera.json").read_text()) for run in (args.rgb_run, args.rgbd_run)]
    predicted, measured_camera = cameras
    if predicted["source"] != "MoGe" or measured_camera["source"] != "calibrated" or measured_camera["pointmap_units"] != "meters":
        parser.error("Use an RGB-only MoGe run and a calibrated measured RGB-D run.")
    if predicted["image"] != measured_camera["image"] or predicted["mask"] != measured_camera["mask"]:
        parser.error("Runs must use the same original image and object mask.")
    if predicted["pointmap_frame"] != measured_camera["pointmap_frame"]:
        parser.error("Pointmaps must use the same coordinate convention.")
    if not np.isfinite(args.inlier_m) or args.inlier_m <= 0 or args.max_points < 12:
        parser.error("Use a positive metre threshold and at least 12 points.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory.")
    image = cv2.imread(predicted["image"])
    mask_image = cv2.imread(predicted["mask"], cv2.IMREAD_GRAYSCALE)
    source_map, target_map = [np.load(run / "pointmap.npy", allow_pickle=False) for run in (args.rgb_run, args.rgbd_run)]
    if image is None or mask_image is None or image.shape[:2] != mask_image.shape or source_map.shape != target_map.shape or source_map.shape != (*mask_image.shape, 3):
        parser.error("Both pointmaps, image and mask must have matching dimensions.")
    observed = mask_image > 0
    valid_target = np.isfinite(target_map).all(2) & (target_map[..., 2] > 0)
    valid = valid_target & np.isfinite(source_map).all(2) & (source_map[..., 2] > 0)
    interior = cv2.erode(observed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    pixels = np.argwhere(interior & valid)
    source, target = source_map[interior & valid].astype(float), target_map[interior & valid].astype(float)
    if len(source) < 12:
        parser.error("Too few valid shared object pixels after mask erosion.")
    rng = np.random.default_rng(42)
    ids = rng.choice(len(source), min(args.max_points, len(source)), replace=False)
    print("Aligning saved scene pointmaps on CPU...", flush=True)
    scale, correction, shift = robust_similarity(source[ids], target[ids], args.inlier_m, rng)
    errors = residuals((scale, correction, shift), source, target)
    inliers = errors < args.inlier_m
    inlier_hull = np.zeros(observed.shape, dtype=np.uint8)
    if inliers.sum() >= 3:
        cv2.fillConvexPoly(inlier_hull, cv2.convexHull(pixels[inliers, ::-1].astype(np.int32)), 1)
    baseline_scale = float(np.median(target[:, 2] / source[:, 2]))
    baseline_errors = np.linalg.norm(baseline_scale * source - target, axis=1)
    with np.load(args.rgb_run / "pose.npz", allow_pickle=False) as saved:
        original = dict(saved)
    r0, t0, s0 = quaternion_matrix(original["rotation"]), original["translation"].reshape(3), original["scale"].reshape(-1)
    rotation, translation, size = r0 @ correction, scale * t0 @ correction + shift, scale * s0
    k = np.asarray(measured_camera["intrinsics_pixels"], dtype=float)
    mesh = trimesh.load(str(args.rgb_run / "mesh.glb"), force="mesh", process=False)
    export_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    native_vertices = np.asarray(mesh.vertices) @ export_rotation.T
    args.output.mkdir(parents=True, exist_ok=True)

    def render_result(name, r, t, s, intrinsics, metric):
        vertices = ((native_vertices * s) @ r + t) * [-1, -1, 1]
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
        v, u = np.indices(observed.shape)
        directions = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(intrinsics).T
        rays = np.concatenate((np.zeros_like(directions), directions), axis=-1).astype(np.float32)
        depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()  # Ray Z=1, so t_hit is optical depth.
        foreground = np.isfinite(depth) & (depth > 0)
        overlap = foreground & observed & valid_target
        metrics = {"mask_iou": float((foreground & observed).sum() / (foreground | observed).sum()),
                   "mask_coverage": float((foreground & observed).sum() / observed.sum()),
                   "mesh_depth_error_m": float(np.median(np.abs(depth[overlap] - target_map[..., 2][overlap]))) if metric and overlap.any() else None}
        overlay = draw_silhouette(image.copy(), foreground.astype(np.uint8) * 255, 0.35, observed.astype(np.uint8) * 255)
        overlay = draw_pose(overlay, r, t, intrinsics, 0.05 if metric else 0.05 / scale)
        cv2.putText(overlay, name, (15, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imwrite(str(args.output / f"{name}.png"), overlay)
        return metrics

    report = {"rgb_run": str(args.rgb_run.resolve()), "rgbd_run": str(args.rgbd_run.resolve()),
              "threshold_m": args.inlier_m, "seed": 42, "valid_pairs": len(source), "fit_pairs": len(ids),
              "scale": float(scale), "rotation_row": correction.tolist(), "translation_m": shift.tolist(),
              "rotation_deg": float(np.rad2deg(Rotation.from_matrix(correction).magnitude())),
              "inlier_fraction": float(inliers.mean()),
              "inlier_hull_mask_coverage": float(((inlier_hull > 0) & observed).sum() / observed.sum()),
              "scene_error_m": {"scale_only_median": float(np.median(baseline_errors)), "aligned_median": float(np.median(errors)),
                                "aligned_p90": float(np.percentile(errors, 90)), "inlier_median": float(np.median(errors[inliers])) if inliers.any() else None},
              "scale_only_factor": baseline_scale,
              "notes": ["Fits scene points, not mesh points; all valid shared object pixels used for reported errors.",
                        "Original RGB-only depth has relative units; no original metric mesh error is reported.",
                        "Scale-only baseline uses median measured/predicted Z and calibrated intrinsics.",
                        "Mesh IoU ignores scene occlusion; mesh depth error uses overlapping valid object pixels."]}
    print("Rendering comparison on CPU...", flush=True)
    report["original"] = render_result("original", r0, t0, s0, np.asarray(predicted["intrinsics_pixels"]), False)
    report["scale_only"] = render_result("scale_only", r0, baseline_scale * t0, baseline_scale * s0, k, True)
    report["aligned"] = render_result("aligned", rotation, translation, size, k, True)
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    np.savez(args.output / "pose.npz", rotation=quaternion[None], translation=translation[None], scale=size.reshape(original["scale"].shape),
             intrinsics_pixels=k, image_size_hw=np.array(observed.shape), translation_units="meters", intrinsics_source="calibrated",
             pose_method="RGB-only scene-depth similarity correction")
    shutil.copy2(args.rgb_run / "mesh.glb", args.output / "mesh.glb")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Scene fit: {inliers.mean():.1%} inliers, median error {np.median(errors)*1000:.1f} mm; scale={scale:.4g}, rotation={report['rotation_deg']:.1f} deg")
    print(f"Mesh IoU: {report['original']['mask_iou']:.3f} original -> {report['aligned']['mask_iou']:.3f} aligned")
    print(f"Mesh depth error (m): {report['scale_only']['mesh_depth_error_m']} scale-only -> "
          f"{report['aligned']['mesh_depth_error_m']} aligned; saved {args.output}")


if __name__ == "__main__":
    main()
