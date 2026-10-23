"""Fit a metric SAM3D mesh pose directly to measured depth and the object mask.

CPU render-and-compare with local normals, contours, pose and bounded size.
The saved result is an experimental candidate, not a validated semantic frame.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from visualize_pose import draw_pose, draw_silhouette, quaternion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Metric mesh.glb and pose.npz")
    parser.add_argument("--rgbd-run", type=Path, required=True, help="Measured camera.json and pointmap.npy")
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--fit-size", type=int, default=192, help="Approximate longest side of fitting crop")
    parser.add_argument("--depth-sigma", type=float, default=0.01, help="Robust depth scale in metres")
    parser.add_argument("--mask-weight", type=float, default=4.0)
    parser.add_argument("--max-angle", type=float, default=45, help="Bound for each rotation-vector component, degrees")
    parser.add_argument("--max-translation", type=float, default=0.15, help="Mesh-centre shift bound per camera axis, metres")
    parser.add_argument("--max-evals", type=int, default=2000, help="Evaluation budget per optimized start")
    parser.add_argument("--starts", type=int, default=4, help="Number of distinct starts to optimize")
    parser.add_argument("--scale-bound", type=float, default=1.6, help="Scale multiplier range: 1/bound to bound")
    parser.add_argument("--scale-mode", choices=("uniform", "axes"), default="uniform")
    parser.add_argument("--shape-weight", type=float, default=2.0, help="Penalty for changing proportions in axes mode")
    parser.add_argument("--normal-radius", type=int, default=6, help="Measured normal neighborhood radius in original pixels")
    parser.add_argument("--normal-sigma-deg", type=float, default=15.0)
    parser.add_argument("--normal-weight", type=float, default=2.0)
    parser.add_argument("--contour-sigma-px", type=float, default=5.0, help="Contour distance scale in original pixels")
    parser.add_argument("--contour-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory.")
    settings = [args.fit_size, args.depth_sigma, args.mask_weight, args.max_angle, args.max_translation, args.max_evals]
    if not np.isfinite(settings).all() or min(settings) <= 0 or args.fit_size < 32:
        parser.error("Use positive settings and a fitting size of at least 32.")
    settings = [args.starts, args.scale_bound-1, args.normal_radius, args.normal_sigma_deg, args.normal_weight,
                args.contour_sigma_px, args.contour_weight, args.shape_weight]
    if not np.isfinite(settings).all() or min(settings) <= 0:
        parser.error("Use positive refinement settings and a scale bound above one.")
    camera = json.loads((args.rgbd_run / "camera.json").read_text())
    with np.load(args.run / "pose.npz", allow_pickle=False) as saved:
        pose = dict(saved)
    if camera["source"] != "calibrated" or camera["pointmap_units"] != "meters" or str(pose.get("translation_units")) != "meters":
        parser.error("Both the input pose and measured pointmap must use metres.")
    image = cv2.imread(camera["image"])
    mask_image = cv2.imread(camera["mask"], cv2.IMREAD_GRAYSCALE)
    points = np.load(args.rgbd_run / "pointmap.npy", allow_pickle=False)
    if image is None or mask_image is None or image.shape[:2] != mask_image.shape or points.shape != (*mask_image.shape, 3):
        parser.error("Image, object mask and measured pointmap must have matching dimensions.")
    observed = mask_image > 0
    valid = np.isfinite(points).all(2) & (points[..., 2] > 0)
    if (observed & valid).sum() < 100:
        parser.error("Too few measured object pixels.")
    depth = np.where(valid, points[..., 2], 0)
    k = np.asarray(camera["intrinsics_pixels"], dtype=float)
    if not np.allclose(pose["intrinsics_pixels"], k) or tuple(pose["image_size_hw"]) != observed.shape:
        parser.error("Use a pose saved with this calibrated camera and image resolution.")
    flip = np.diag([-1, -1, 1])
    r0, t0 = flip @ quaternion_matrix(pose["rotation"]).T, pose["translation"].reshape(3) @ flip
    scale = pose["scale"].reshape(-1)
    mesh = trimesh.load(str(args.run / "mesh.glb"), force="mesh", process=False)
    export_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    vertices = (np.asarray(mesh.vertices) @ export_rotation.T) * scale
    pivot = (vertices.min(0) + vertices.max(0)) / 2
    pivot_camera = r0 @ pivot + t0
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    v, u = np.indices(observed.shape)
    directions = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(k).T
    # Local depth neighborhoods supply surface normals, never an object/world frame.
    radius = min(args.normal_radius, (min(observed.shape)-1)//2)
    kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
    xyz = depth[..., None] * directions
    patch_normals = np.cross(xyz[radius:-radius, 2*radius:] - xyz[radius:-radius, :-2*radius],
                             xyz[2*radius:, radius:-radius] - xyz[:-2*radius, radius:-radius])
    lengths = np.linalg.norm(patch_normals, axis=-1)
    normals = np.zeros_like(xyz)
    normals[radius:-radius, radius:-radius] = patch_normals / np.maximum(lengths[..., None], 1e-12)
    normal_valid = cv2.erode((observed & valid).astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    normal_valid &= (cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)) < 4*args.depth_sigma
    normal_valid &= np.linalg.norm(normals, axis=-1) > 0.5
    y, x = np.where(observed)
    margin = int(0.6 * max(np.ptp(x) + 1, np.ptp(y) + 1))
    x0, x1 = max(0, x.min()-margin), min(observed.shape[1], x.max()+margin+1)
    y0, y1 = max(0, y.min()-margin), min(observed.shape[0], y.max()+margin+1)
    stride = max(1, int(np.ceil(max(x1-x0, y1-y0) / args.fit_size)))
    crop = np.s_[y0:y1:stride, x0:x1:stride]
    fit_mask, fit_valid, fit_depth, fit_rays = observed[crop], valid[crop], depth[crop], directions[crop]
    fit_normals, normal_samples = normals[crop], normal_valid[crop]
    interior = cv2.erode(observed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    samples = interior[crop] & fit_valid
    if samples.sum() < 30:
        parser.error("Too few fitting pixels; increase --fit-size.")
    tolerance = args.depth_sigma

    def contour(mask):
        return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    fit_edge = contour(fit_mask)
    edge_distance = cv2.distanceTransform((~fit_edge).astype(np.uint8), cv2.DIST_L2, 3)

    def contour_error(exposed):
        edge = contour(exposed)
        if not edge.any():
            return 10.0
        distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
        return float(0.5 * (np.minimum(edge_distance[edge]*stride/args.contour_sigma_px, 10).mean()
                            + np.minimum(distance[fit_edge]*stride/args.contour_sigma_px, 10).mean()))

    def unpack(parameters):
        # Keep scale/rotation centred on the mesh; export translation of its native origin.
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix() @ r0
        multiplier = np.exp(parameters[6:])
        return rotation, pivot_camera + 0.1*parameters[3:6] - rotation @ (multiplier*pivot), multiplier

    def render(rotation, translation, multiplier, rays_camera):
        directions_object = (rays_camera @ rotation) / multiplier
        origins = np.broadcast_to((-translation @ rotation) / multiplier, directions_object.shape)
        rays = np.concatenate((origins, directions_object), axis=-1).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays))
        rendered_normals = (hits["primitive_normals"].numpy() / multiplier) @ rotation.T
        rendered_normals /= np.maximum(np.linalg.norm(rendered_normals, axis=-1, keepdims=True), 1e-12)
        return hits["t_hit"].numpy(), rendered_normals

    def visible(rendered_depth, mask, valid_depth, measured_depth):
        hit = np.isfinite(rendered_depth) & (rendered_depth > 0)
        # Keep all inside-mask hits: a bad pose must not hide behind its own target.
        occluded = (~mask) & valid_depth & (rendered_depth > measured_depth + tolerance)
        return hit, hit & ~occluded

    size_parameters = 1 if args.scale_mode == "uniform" else 3
    evaluations, best_cost, best_parameters = 0, np.inf, np.zeros(6+size_parameters)

    def objective(parameters, details=False):
        nonlocal evaluations, best_cost, best_parameters
        evaluations += 1
        rendered, rendered_normals = render(*unpack(parameters), fit_rays)
        hit, exposed = visible(rendered, fit_mask, fit_valid, fit_depth)
        # Fixed observed pixels: missing rays pay 10 sigma instead of disappearing.
        error = np.full(fit_mask.shape, 10.0)
        shared = samples & hit
        error[shared] = np.minimum(np.abs(rendered[shared] - fit_depth[shared]) / args.depth_sigma, 10)
        depth_loss = np.mean(2 * (np.sqrt(1 + error[samples]**2) - 1))
        # Fixed normal samples prevent bad candidates from hiding difficult surfaces.
        normal_error = np.full(fit_mask.shape, 90.0 / args.normal_sigma_deg)
        paired = normal_samples & hit
        cosine = np.clip(np.abs(np.sum(rendered_normals[paired]*fit_normals[paired], axis=-1)), 0, 1)
        normal_error[paired] = np.rad2deg(np.arccos(cosine)) / args.normal_sigma_deg
        normal_loss = np.mean(2*(np.sqrt(1 + normal_error[normal_samples]**2)-1)) if normal_samples.any() else 0.0
        iou = (exposed & fit_mask).sum() / (exposed | fit_mask).sum()
        contour_loss = contour_error(exposed)
        cost = float(depth_loss + args.normal_weight*normal_loss + args.mask_weight*(1-iou)
                     + args.contour_weight*contour_loss + 0.05*np.mean(parameters[6:]**2)
                     + args.shape_weight*np.sum((parameters[6:]-parameters[6:].mean())**2)
                     + 0.001*np.dot(parameters[:6], parameters[:6]))
        if cost < best_cost:
            best_cost, best_parameters = cost, parameters.copy()
        return {"total": cost, "depth": float(depth_loss), "normal": float(normal_loss), "mask": float(1-iou),
                "contour": contour_loss} if details else cost

    print(f"Fitting pose and {args.scale_mode} scale on CPU; {int(normal_samples.sum())} reliable normal pixels...", flush=True)
    initial = np.zeros(6+size_parameters)
    initial_cost = objective(initial)
    initial_terms = objective(initial, details=True)
    rendered, rendered_normals = render(r0, t0, 1.0, fit_rays)
    shared = samples & np.isfinite(rendered) & (rendered > 0)
    seeded = initial.copy()
    if shared.any():
        delta = np.median((fit_depth[shared] - rendered[shared])[:, None] * fit_rays[shared], axis=0)
        seeded[3:6] = np.clip(delta, -args.max_translation, args.max_translation) / 0.1
    angle = np.deg2rad(args.max_angle)
    starts = [initial, seeded]
    for axis in np.eye(3):
        for sign in (-1, 1):
            start = seeded.copy()
            start[:3] = sign * min(np.deg2rad(35), angle) * axis
            starts.append(start)
    # A data-derived normal rotation is only another hypothesis; no semantic axis is prescribed.
    paired = normal_samples & np.isfinite(rendered) & (rendered > 0)
    if paired.sum() >= 30:
        a, b = rendered_normals[paired].copy(), fit_normals[paired].copy()
        for n in (a, b):
            n[np.sum(n*fit_rays[paired], axis=-1) > 0] *= -1
        a, b = np.median(a, axis=0), np.median(b, axis=0)
        if min(np.linalg.norm(a), np.linalg.norm(b)) > 0.5:
            delta_rotation, _ = Rotation.align_vectors([b/np.linalg.norm(b)], [a/np.linalg.norm(a)])
            start = seeded.copy()
            start[:3] = np.clip(delta_rotation.as_rotvec(), -angle, angle)
            starts.append(start)
    candidates = [np.r_[start[:6], np.full(size_parameters, np.log(s))] for start in starts
                  for s in (1/np.sqrt(args.scale_bound), 1.0, np.sqrt(args.scale_bound))]
    for start in list(candidates):
        rendered, _ = render(*unpack(start), fit_rays)
        shared = samples & np.isfinite(rendered) & (rendered > 0)
        if shared.any():
            delta = np.median((fit_depth[shared]-rendered[shared])[:, None]*fit_rays[shared], axis=0)
            start = start.copy()
            start[3:6] = np.clip(start[3:6] + delta/0.1, -args.max_translation/0.1, args.max_translation/0.1)
            candidates.append(start)
    candidates.append(initial.copy())
    ranked = sorted(((objective(start), start) for start in candidates), key=lambda item: item[0])
    selected = []
    for _, start in ranked:
        if all(np.linalg.norm(start[:3]-old[:3]) > np.deg2rad(10) for old in selected):
            selected.append(start)
        if len(selected) == args.starts:
            break
    bounds = ([(-angle, angle)] * 3 + [(-args.max_translation/0.1, args.max_translation/0.1)] * 3
              + [(-np.log(args.scale_bound), np.log(args.scale_bound))] * size_parameters)
    # Derivative-free optimization avoids float32 raycast/visibility Jacobian errors.
    trials = []
    for index, start in enumerate(selected):
        print(f"Refining start {index+1}/{len(selected)}...", flush=True)
        result = minimize(objective, start, method="Powell", bounds=bounds,
                          options={"maxfev": args.max_evals, "xtol": 1e-3, "ftol": 1e-4})
        trials.append({"start": start.tolist(), "final_cost": float(result.fun), "evaluations": int(result.nfev),
                       "success": bool(result.success), "message": str(result.message)})
    best = best_parameters.copy()
    rotation, translation, multiplier = unpack(best)
    final_terms = objective(best, details=True)
    args.output.mkdir(parents=True, exist_ok=True)

    def save_overlay(name, r, t, multiplier):
        rendered, rendered_normals = render(r, t, multiplier, directions)
        hit, exposed = visible(rendered, observed, valid, depth)
        shared = hit & observed & valid
        overlay = draw_silhouette(image.copy(), exposed.astype(np.uint8) * 255, 0.35, observed.astype(np.uint8) * 255)
        overlay = draw_pose(overlay, r.T @ flip, t @ flip, k, 0.05)
        cv2.putText(overlay, name + ": scene-occluded mesh pixels hidden", (15, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(str(args.output / f"{name}.png"), overlay)
        paired = normal_valid & hit
        angles = np.rad2deg(np.arccos(np.clip(np.abs(np.sum(normals[paired]*rendered_normals[paired], axis=-1)), 0, 1)))
        return {"visible_mask_iou": float((exposed & observed).sum() / (exposed | observed).sum()),
                "full_mesh_mask_iou": float((hit & observed).sum() / (hit | observed).sum()),
                "mask_coverage": float((hit & observed).sum() / observed.sum()),
                "depth_error_m": float(np.median(np.abs(rendered[shared] - depth[shared]))) if shared.any() else None,
                "normal_error_median_deg": float(np.median(angles)) if len(angles) else None,
                "normal_error_p90_deg": float(np.percentile(angles, 90)) if len(angles) else None,
                "normal_coverage": float(paired.sum()/max(1, normal_valid.sum()))}

    report = {"run": str(args.run.resolve()), "rgbd_run": str(args.rgbd_run.resolve()),
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "fit_crop_xyxy": list(map(int, (x0, y0, x1, y1))), "fit_stride": stride,
              "fit_pixels": int(samples.sum()), "evaluations": evaluations, "initial_cost": initial_cost,
              "fit_normal_pixels": int(normal_samples.sum()), "full_normal_pixels": int(normal_valid.sum()),
              "candidate_cost": best_cost, "optimizer_trials": trials, "initial_terms": initial_terms, "candidate_terms": final_terms,
              "scale_multipliers_xyz": np.broadcast_to(multiplier, (3,)).tolist(), "pivot_shift_m": (0.1*best[3:6]).tolist(),
              "scale_at_bound": bool(np.any(np.abs(best[6:]) > 0.98*np.log(args.scale_bound))),
              "rotation_change_deg": float(np.rad2deg(np.linalg.norm(best[:3]))), "translation_change_m": (translation-t0).tolist(),
              "before": save_overlay("before", r0, t0, 1.0), "after": save_overlay("after", rotation, translation, multiplier),
              "notes": ["Pose and bounded size fit; native axes retained. Candidate needs visual review.",
                        "Uniform mode retains proportions; axes mode regularizes them. Hidden dimensions are not measured ground truth.",
                        "Best visited candidate across all starts, including initialization; optimizer success is not pose accuracy.",
                        "Normal angles are unsigned and use reliable local depth neighborhoods; inspect normal coverage too.",
                        "Depth error uses overlapping object pixels; inspect coverage alongside error.",
                        "Fitting uses an expanded object crop; reported before/after metrics use the full image.",
                        "Only outside-mask surfaces behind measured scene depth are treated as occluded.",
                        "Limited search around the initial pose; flat or occluded surfaces can leave pose ambiguous."]}
    quaternion = Rotation.from_matrix(rotation.T @ flip).as_quat()[[3, 0, 1, 2]]
    corrected = dict(pose, rotation=quaternion[None], translation=(translation @ flip)[None],
                     scale=np.broadcast_to(scale*multiplier, (3,)).copy().reshape(1, 3),
                     pose_method="Mesh pose and size fit with depth, normals and contours")
    np.savez(args.output / "pose.npz", **corrected)
    shutil.copy2(args.run / "mesh.glb", args.output / "mesh.glb")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    for name in ("before", "after"):
        metrics = report[name]
        print(f"{name}: IoU={metrics['visible_mask_iou']:.3f}, depth(m)={metrics['depth_error_m']}, normal(deg)={metrics['normal_error_median_deg']}")
    print(f"Saved candidate to {args.output}; XYZ scale multipliers={np.round(np.broadcast_to(multiplier, (3,)), 3)}")


if __name__ == "__main__":
    main()
