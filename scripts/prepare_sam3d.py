"""Prepare or reuse a refined SAM3D demo frame for a temporary planning config."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Temporary launcher config to update")
    parser.add_argument("--prepare-inputs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--entity", choices=("controlled", "target"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config.resolve()
    base = config_path.parent
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(config_path.read_text())
    reconstruction = config.get("reconstruction", {})
    methods = {entity: reconstruction.get(f"{entity}_object_frame_method", "pca")
               for entity in ("controlled", "target")}
    if any(method not in ("pca", "sam3d") for method in methods.values()):
        parser.error("controlled_object_frame_method and target_object_frame_method must be pca or sam3d")
    if args.entity is None:
        # Separate workers release each object's GPU memory before the next starts.
        for entity, method in methods.items():
            if method == "sam3d":
                subprocess.run([sys.executable, str(Path(__file__).resolve()), str(config_path),
                                "--entity", entity], check=True)
        return
    demo = config["demo"]
    settings = config["sam3d"]
    camera = {**config.get("camera", {}), **config.get("demo_camera", {})}
    resolve = lambda value: (base / Path(value).expanduser()).resolve()
    if args.entity == "target":
        analysis = json.loads(resolve(demo.get("analysis") or demo["video_grounding"]).read_text())
        if analysis.get("video_structure", analysis)["target_representation"] != "object_pose_6d":
            return
    image = resolve(demo.get("rgb") or demo["initial_rgb"])
    depth = resolve(demo["initial_depth"])
    grounding = resolve(demo.get("scene_grounding") or demo.get("analysis") or demo["video_grounding"])
    model = config.get("target_pose_registration", {}).get("sam2_model_path", "facebook/sam2-hiera-large")

    if args.prepare_inputs:
        import numpy as np
        from PIL import Image
        from motion_planning.full_pipeline import load_camera_bundle
        from motion_planning.target_pose_registration import Sam2BoxSegmenter

        work = args.prepare_inputs
        calibration = load_camera_bundle(camera, base)
        payload = json.loads(grounding.read_text())
        scene = payload.get("demo_initial_grounding", payload)
        bbox = (scene["controlled_entity"]["bbox"] if args.entity == "controlled"
                else scene["target_geometry"]["target_object_bbox"])
        rgb = np.asarray(Image.open(work / "image.png").convert("RGB"))
        mask = Sam2BoxSegmenter(model, "cuda").segment(rgb, bbox)
        Image.fromarray(mask.astype(np.uint8) * 255).save(work / "mask.png")
        (work / "camera.json").write_text(json.dumps({"calibrated": {
            "intrinsics": calibration.intrinsic_matrix.tolist(),
            "extrinsics": calibration.transform_world_camera.tolist(),
        }}, indent=2))
        return

    root = resolve(settings["root"])
    tools_root = project_root / "sam3d_tools"
    scripts = ["demo_full.py", "align_scene_depth.py", "fit_mesh_depth.py", "visualize_pose.py"]
    sources = [image, depth, grounding, Path(__file__),
               project_root / "motion_planning/full_pipeline.py",
               project_root / "motion_planning/target_pose_registration.py"]
    sources += [resolve(camera[key]) for key in ("bundle", "intrinsics", "T_world_camera") if camera.get(key)]
    sources += [tools_root / name for name in scripts] + [root / "checkpoints/hf/pipeline.yaml"]
    signature = {"files": [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources],
                 "settings": settings, "camera": camera, "sam2_model": model, "entity": args.entity}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = project_root / "sam3d_cache" / key
    complete = cache / "complete.json"
    cached = json.loads(complete.read_text()) if complete.exists() else None
    work = cache / cached["directory"] if cached else None
    ready = work is not None and all((work / "fit" / name).is_file() for name in ("pose.npz", "mesh.glb", "report.json"))
    if ready:
        print(f"[SAM3D] Reusing {args.entity} pose: {work / 'fit/pose.npz'}", flush=True)
    else:
        cache.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="attempt-", dir=cache))
        shutil.copy2(image, work / "image.png")
        shutil.copy2(depth, work / "depth.npy")
        print(f"[SAM3D] Preparing demo {args.entity} frame: {work}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(config_path),
                        "--entity", args.entity, "--prepare-inputs", str(work)], check=True)
        command = ["conda", "run", "--no-capture-output", "-n", settings["environment"], "python"]

        def run(script, *arguments):
            subprocess.run(command + [str(tools_root / script), *map(str, arguments)], cwd=root, check=True)

        common = ["--sam3d-root", root, "--image", work / "image.png",
                  "--mask", work / "mask.png", "--seed", settings["seed"]]
        run("demo_full.py", *common, "--output", work / "rgb")
        run("demo_full.py", *common, "--depth", work / "depth.npy", "--camera-json", work / "camera.json",
            "--camera-name", "calibrated", "--depth-scale", camera.get("depth_scale_to_m", 1.0), "--output", work / "rgbd")
        run("align_scene_depth.py", "--rgb-run", work / "rgb", "--rgbd-run", work / "rgbd", "--output", work / "metric")
        run("fit_mesh_depth.py", "--run", work / "metric", "--rgbd-run", work / "rgbd", "--output", work / "fit",
            "--scale-mode", settings["scale_mode"], "--scale-bound", settings["scale_bound"], "--starts", settings["starts"])
        pending = work / "complete.json"
        pending.write_text(json.dumps({"directory": work.name, "signature": signature}, indent=2))
        pending.replace(complete)
    pose_key = "sam3d_pose" if args.entity == "controlled" else "sam3d_target_pose"
    config["demo"][pose_key] = str(work / "fit/pose.npz")
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
