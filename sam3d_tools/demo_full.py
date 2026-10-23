"""Full SAM3D demo with sequential GPU residency and an external checkout.

Uses the upstream loader (its initial memory peak is unchanged), then offloads
finished stages to CPU. Sampling settings and output resolution are unchanged.
"""
import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
root_parser = argparse.ArgumentParser(add_help=False)
root_parser.add_argument("--sam3d-root", type=Path,
                         default=Path(__file__).resolve().parents[1] / "external/sam-3d-objects",
                         help="Official SAM3D checkout (default: external/sam-3d-objects)")
ROOT = root_parser.parse_known_args()[0].sam3d_root.expanduser().resolve()
sys.path[:0] = [str(ROOT / "notebook"), str(ROOT)]

import numpy as np
import torch
import trimesh
from loguru import logger
from inference import Inference, load_image, load_mask
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp


def move_tensor_attributes(obj, device):
    """Move tensors held by plain geometry objects, outside nn.Module buffers."""
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            setattr(obj, name, value.to(device))
    if hasattr(obj, "device"):
        obj.device = torch.device(device)


def move_model(model, device):
    if model is None:
        return
    model.to(device)
    # Some position encoders keep tensors outside registered buffers.
    for child in model.modules():
        for name, value in vars(child).items():
            if isinstance(value, torch.Tensor):
                setattr(child, name, value.to(device))
    extractor = getattr(model, "mesh_extractor", None)
    if extractor is not None:
        move_tensor_attributes(extractor, device)
        move_tensor_attributes(extractor.mesh_extractor, device)


def release_cache():
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def gpu_stage(name, models, device):
    print(f"{name}...", flush=True)
    try:
        for model in models:
            move_model(model, device)
        yield
    finally:
        for model in models:
            move_model(model, "cpu")
        release_cache()


def staged_call(function, name, models, device):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with gpu_stage(name, models, device):
            return function(*args, **kwargs)
    return wrapped


def enable_offloading(pipeline):
    """Wrap this pipeline instance without modifying the upstream source."""
    models = pipeline.models
    embedders = pipeline.condition_embedders
    depth = pipeline.depth_model.model
    for model in [*models.values(), *embedders.values(), depth]:
        move_model(model, "cpu")
    release_cache()

    stages = (
        ("compute_pointmap", "Depth estimation", [depth]),
        ("sample_sparse_structure", "Coarse shape and pose", [
            models["ss_generator"], models["ss_decoder"],
            embedders["ss_condition_embedder"],
        ]),
        ("sample_slat", "Detailed latent generation", [
            models["slat_generator"], embedders["slat_condition_embedder"],
        ]),
    )
    for method, label, required in stages:
        setattr(pipeline, method, staged_call(
            getattr(pipeline, method), label, required, pipeline.device,
        ))

    def decode_slat(slat, formats=("mesh", "gaussian")):
        outputs = {}
        decoders = {"mesh": "slat_decoder_mesh", "gaussian": "slat_decoder_gs",
                    "gaussian_4": "slat_decoder_gs_4"}
        for name, key in decoders.items():
            if name not in formats:
                continue
            with gpu_stage(f"Decoding {name}", [models[key]], pipeline.device):
                # Fresh wrappers prevent sparse convolution caches being retained
                # by the shared latent between the mesh and Gaussian decoders.
                decoder_input = sp.SparseTensor(feats=slat.feats, coords=slat.coords)
                try:
                    with torch.no_grad():
                        outputs[name] = models[key](decoder_input)
                finally:
                    del decoder_input
                if name == "mesh":
                    for mesh in outputs[name]:
                        move_tensor_attributes(mesh, "cpu")
        # Gaussian tensors remain on GPU for optional texture/layout rendering.
        return outputs

    pipeline.decode_slat = decode_slat


def load_rgbd_inputs(depth_path, camera_path, camera_name, depth_scale, image, mask):
    """Backproject registered optical-axis depth in meters using pixel intrinsics."""
    with open(camera_path) as stream:
        camera = json.load(stream)[camera_name]
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Camera intrinsics must be a finite 3x3 matrix.")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or depth_scale <= 0:
        raise ValueError("Focal lengths and depth scale must be positive.")
    depth = np.load(depth_path, allow_pickle=False).astype(np.float32) * depth_scale
    if depth.shape != image.shape[:2] or mask.shape != depth.shape:
        raise ValueError("RGB, mask and registered depth must have the same resolution.")
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid & mask):
        raise ValueError("The object mask contains no valid measured depth.")
    v, u = np.indices(depth.shape, dtype=np.float32)
    pixels = np.stack((u, v, np.ones_like(depth)), axis=-1)
    points = (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]
    points[~valid] = np.nan
    # OpenCV (+X right, +Y down) -> SAM3D/PyTorch3D (+X left, +Y up).
    points[..., :2] *= -1
    return points, intrinsics


def use_rgbd_inputs(pipeline, points, intrinsics):
    """Use measured geometry and calibrated intrinsics without running MoGe."""
    height, width = points.shape[:2]
    normalized_k = intrinsics.copy()
    normalized_k[0] /= width
    normalized_k[1] /= height

    def compute_pointmap(image, pointmap=None):
        print("Using measured depth and camera intrinsics...", flush=True)
        rgb = torch.from_numpy(image[..., :3].astype(np.float32) / 255).to(pipeline.device)
        mask = torch.from_numpy(image[..., -1].astype(np.float32) / 255).to(pipeline.device)
        xyz = torch.from_numpy(points).to(pipeline.device).permute(2, 0, 1)
        return {"pointmap": pipeline._clip_pointmap(xyz, mask),
                "pts_color": rgb.permute(2, 0, 1),
                "intrinsics": torch.from_numpy(normalized_k).to(pipeline.device)}

    @torch.inference_mode(False)
    def refine_layout(gs, intrinsics, pose, inputs, backend="gsplat"):
        # Match the default preprocessor's centered square padding and resize.
        # Keep the measured focal lengths and off-center principal point.
        side = max(height, width)
        render_h, render_w = inputs["rgb_image"].shape[-2:]
        intrinsics = intrinsics.clone()
        intrinsics[0] *= width
        intrinsics[1] *= height
        intrinsics[0, 2] += (side - width) // 2 + 0.5
        intrinsics[1, 2] += (side - height) // 2 + 0.5
        intrinsics[:2] /= side
        intrinsics[0, 2] -= 0.5 / render_w
        intrinsics[1, 2] -= 0.5 / render_h
        rotation, translation, scale, iou, initial_iou, _, accepted = (
            pipeline.layout_post_optimization_method_GS(
                gs, pose["rotation"], pose["translation"], pose["scale"],
                inputs["rgb_image_mask"][0, 0], inputs["rgb_image"][0],
                inputs["rgb_pointmap_unnorm"][0].permute(1, 2, 0), intrinsics,
                Enable_occlusion_check=False, Enable_manual_alignment=False,
                Enable_shape_ICP=False, Enable_rendering_optimization=True,
                min_size=518, device=pipeline.device, backend=backend,
            )
        )
        return {"rotation": rotation, "translation": translation,
                "scale": pipeline.refine_scale(scale), "iou": iou,
                "iou_before_optim": initial_iou, "optim_accepted": accepted}

    # Instance overrides keep the upstream files and the RGB-only path intact.
    pipeline.compute_pointmap = compute_pointmap
    pipeline.run_post_optimization_GS = refine_layout


def observe_layout_refinement(pipeline, requested):
    """Record refinement outcomes without changing SAM3D's optimization."""
    report = {"requested": requested, "started": False, "completed": False,
              "accepted": None, "error": None, "gaussian_iou_before": None,
              "gaussian_iou_retained_pose": None, "messages": []}
    prefixes = ("INITIAL IoU:", "Step 3 Render-Compare:",
                "Error during layout post optimization:")

    def capture_message(message):
        text = message.record["message"]
        report["messages"].append(text)
        if text.startswith(prefixes[2]):
            report["error"] = report["error"] or text
        elif text.startswith(prefixes[1]):
            # Includes the candidate IoU even if SAM3D rejects that pose.
            print(f"Layout refinement: {text}", flush=True)

    sink = logger.add(capture_message, level="INFO", filter=lambda record: (
        record["name"] in ("sam3d_objects.pipeline.inference_utils",
                           "sam3d_objects.pipeline.inference_pipeline_pointmap")
        and record["message"].startswith(prefixes)
    ))
    refine = pipeline.run_post_optimization_GS

    @wraps(refine)
    def observed(*args, **kwargs):
        report["started"] = True
        print("Layout refinement: running...", flush=True)
        try:
            result = refine(*args, **kwargs)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise  # Preserve upstream error handling and fallback behavior.
        report.update(completed=True, accepted=bool(result["optim_accepted"]),
                      gaussian_iou_before=float(result["iou_before_optim"]),
                      gaussian_iou_retained_pose=float(result["iou"]))
        return result

    pipeline.run_post_optimization_GS = observed
    return report, sink


def observe_camera_inputs(pipeline):
    """Copy camera/geometry inputs to CPU without changing model inputs."""
    captured = {"refinement": None}
    compute = pipeline.compute_pointmap

    @wraps(compute)
    def observed_compute(*args, **kwargs):
        result = compute(*args, **kwargs)
        captured["intrinsics_normalized"] = result["intrinsics"].detach().float().cpu().numpy().copy()
        captured["pointmap"] = result["pointmap"].detach().permute(1, 2, 0).float().cpu().numpy().copy()
        return result

    pipeline.compute_pointmap = observed_compute
    refine = pipeline.layout_post_optimization_method_GS
    if refine is not None:
        @wraps(refine)
        def observed_refine(gaussian, quaternion, translation, scale, mask,
                            rgb_gt, point_map, intrinsics, *args, **kwargs):
            captured["refinement"] = {
                "intrinsics_normalized": intrinsics.detach().float().cpu().tolist(),
                "input_mask_size_hw": list(mask.shape[-2:]),
                "note": "Unmodified camera passed to the official GS refinement method.",
            }
            return refine(gaussian, quaternion, translation, scale, mask,
                          rgb_gt, point_map, intrinsics, *args, **kwargs)

        pipeline.layout_post_optimization_method_GS = observed_refine
    return captured


def save_camera_inputs(output_dir, captured, args, image_shape, rgbd):
    height, width = image_shape[:2]
    normalized_k = captured["intrinsics_normalized"]
    renderer_k = normalized_k.copy()
    renderer_k[0] *= width
    renderer_k[1] *= height
    if rgbd is None:
        pixel_k = renderer_k.copy()
        # MoGe UV centers are (pixel + 0.5) / size; OpenCV indexes pixel centers by integers.
        pixel_k[:2, 2] -= 0.5
    else:
        pixel_k = rgbd[1].copy()
    source = "calibrated" if rgbd is not None else "MoGe"
    camera = {
        "source": source, "image_size_hw": [height, width], "seed": args.seed,
        "image": str(args.image.resolve()), "mask": str(args.mask.resolve()),
        "intrinsics_normalized": normalized_k.tolist(),
        "intrinsics_pixels": pixel_k.tolist(),
        "intrinsics_pixels_note": "OpenCV integer pixel centers; original RGB resolution.",
        "upstream_renderer_intrinsics_pixels": renderer_k.tolist(),
        "upstream_renderer_note": "Original-resolution gsplat conversion: normalized rows multiplied by width/height.",
        "pointmap_file": "pointmap.npy", "pointmap_layout": "HWC XYZ",
        "pointmap_frame": "PyTorch3D: +X left, +Y up, +Z forward",
        "pointmap_units": "meters" if rgbd is not None else "relative",
        "pointmap_stage": "After compute_pointmap clipping, before pose preprocessing.",
        "refinement": captured["refinement"],
    }
    np.save(output_dir / "pointmap.npy", captured["pointmap"], allow_pickle=False)
    with open(output_dir / "camera.json", "w") as stream:
        json.dump(camera, stream, indent=2)
        stream.write("\n")
    print(f"Camera ({source}): fx={pixel_k[0, 0]:.2f}, fy={pixel_k[1, 1]:.2f}, "
          f"cx={pixel_k[0, 2]:.2f}, cy={pixel_k[1, 2]:.2f}")
    return pixel_k


def main():
    example = ROOT / "notebook/images/shutterstock_stylish_kidsroom_1640806567"
    parser = argparse.ArgumentParser(description=__doc__, parents=[root_parser])
    parser.add_argument("--config", type=Path, default=ROOT / "checkpoints/hf/pipeline.yaml")
    parser.add_argument("--image", type=Path, default=example / "image.png")
    parser.add_argument("--mask", type=Path, default=example / "14.png")
    parser.add_argument("--depth", type=Path, help="RGB-aligned optical-axis depth .npy")
    parser.add_argument("--camera-json", type=Path, help="Camera calibration JSON")
    parser.add_argument("--camera-name", default="external_sensor0")
    parser.add_argument("--depth-scale", type=float, default=1.0,
                        help="Multiplier to meters: 1 for meters, 0.001 for millimeters")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mesh-postprocess", action="store_true")
    parser.add_argument("--texture-baking", action="store_true")
    parser.add_argument("--layout-refinement", action="store_true")
    args = parser.parse_args()
    if (args.depth is None) != (args.camera_json is None):
        parser.error("Provide --depth and --camera-json together.")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    rgb, mask = load_image(args.image), load_mask(args.mask)
    rgbd = None
    if args.depth is not None:
        rgbd = load_rgbd_inputs(args.depth, args.camera_json, args.camera_name,
                               args.depth_scale, rgb, mask)
    print("Loading models...", flush=True)
    inference = Inference(str(args.config.resolve()), compile=False)
    pipeline = inference._pipeline
    enable_offloading(pipeline)
    if rgbd is not None:
        use_rgbd_inputs(pipeline, *rgbd)
    camera_inputs = observe_camera_inputs(pipeline)
    image = inference.merge_mask_to_rgba(rgb, mask)
    refinement, log_sink = observe_layout_refinement(pipeline, args.layout_refinement)
    try:
        output = pipeline.run(
            image, seed=args.seed, stage1_only=False,
            decode_formats=["mesh", "gaussian"],
            with_mesh_postprocess=args.mesh_postprocess,
            with_texture_baking=args.texture_baking,
            with_layout_postprocess=args.layout_refinement,
            use_vertex_color=True, stage1_inference_steps=None,
        )
    finally:
        logger.remove(log_sink)

    args.output.mkdir(parents=True, exist_ok=True)
    pixel_k = save_camera_inputs(args.output, camera_inputs, args, rgb.shape, rgbd)
    with open(args.output / "refinement.json", "w") as stream:
        json.dump(refinement, stream, indent=2)
        stream.write("\n")
    if refinement["completed"]:
        print(f"Layout refinement: completed; accepted={refinement['accepted']}")
    elif refinement["error"]:
        print("Layout refinement: failed; see refinement.json.")
    else:
        state = "requested but not started" if args.layout_refinement else "disabled"
        print(f"Layout refinement: {state}")
    output["glb"].export(str(args.output / "mesh.glb"))
    move_tensor_attributes(output["gs"], "cpu")
    output["gs"].save_ply(str(args.output / "splat.ply"))
    voxels = output["coords"][:, 1:].detach().cpu().numpy() / 64 - 0.5
    trimesh.points.PointCloud(voxels).export(str(args.output / "coarse.ply"))
    pose = {name: output[name].detach().float().cpu().numpy()
            for name in ("rotation", "translation", "scale")}
    pose["intrinsics_pixels"] = pixel_k
    np.savez(args.output / "pose.npz", **pose, voxel=voxels,
             image_size_hw=np.asarray(rgb.shape[:2]),
             intrinsics_source="calibrated" if rgbd is not None else "MoGe",
             rotation_format="quaternion_wxyz",
             depth_source="measured" if rgbd is not None else "MoGe",
             translation_units="meters" if rgbd is not None else "relative",
             camera_frame="PyTorch3D: +X left, +Y up, +Z forward",
             frame_note="Native SAM3D pose; GLB geometry is separately rotated Z-up to Y-up.")
    del output
    release_cache()
    print(f"Saved reconstruction, pose and camera diagnostics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
