#!/usr/bin/env python3
"""Inspect a saved Panda plan and its RGB-D scene in a Viser browser viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_planning import CameraModel, load_optimization_result
from motion_planning.full_pipeline import load_camera_bundle, load_json, merge_mappings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Plan directory or optimization_result.npz.")
    parser.add_argument("--host", default="0.0.0.0", help="Address on which to serve the viewer.")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed multiplier.")
    parser.add_argument("--point-stride", type=int, default=4, help="RGB-D sampling stride in pixels.")
    parser.add_argument("--no-scene", action="store_true", help="Show only the robot, paths and frames.")
    return parser.parse_args()


def load_scene_cloud(plan_dir: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Backproject referenced current RGB-D inputs, with legacy snapshot support."""
    scene_path = plan_dir / "replay_scene.json"
    if scene_path.exists():
        scene = load_json(scene_path)

        def scene_input(value: str) -> Path:
            source = Path(value).expanduser()
            return (source if source.is_absolute() else scene_path.parent / source).resolve()

        rgb_path = scene_input(scene["rgb"])
        depth_path = scene_input(scene["depth"])
        camera = CameraModel(
            intrinsic_matrix=scene["intrinsic_matrix"],
            transform_world_camera=scene["T_world_camera"],
            depth_scale_to_m=float(scene["depth_scale_to_m"]),
        )
    else:
        run_dir = plan_dir.parent
        inputs = run_dir / "00_inputs"
        config = load_json(inputs / "config_used.json")
        config_dir = Path(load_json(run_dir / "run_manifest.json")["config"]).parent
        records = {item["source"]: item for item in load_json(inputs / "input_manifest.json")}

        def saved_input(value: str) -> Path:
            source = Path(value).expanduser()
            source = (source if source.is_absolute() else config_dir / source).resolve()
            record = records.get(str(source))
            if record and record.get("snapshot"):
                return inputs / "files" / Path(record["snapshot"]).name
            return source

        camera_config = merge_mappings(config.get("camera", {}), config.get("current_camera"))
        for key in ("bundle", "intrinsics", "T_world_camera"):
            if camera_config.get(key):
                camera_config[key] = str(saved_input(camera_config[key]))
        camera = load_camera_bundle(camera_config, config_dir)
        current = config["current"]
        rgb_path = saved_input(current.get("rgb") or current["initial_rgb"])
        depth_path = saved_input(current["initial_depth"])

    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float64).squeeze()
    if rgb.shape[:2] != depth.shape:
        raise ValueError("Saved current RGB and depth image sizes differ.")

    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[rows, cols] * camera.depth_scale_to_m
    valid = np.isfinite(z) & (z > 0.0)
    k = camera.intrinsic_matrix
    points = np.stack(((cols - k[0, 2]) * z / k[0, 0],
                       (rows - k[1, 2]) * z / k[1, 1], z), axis=-1)[valid]
    pose = camera.transform_world_camera
    points = points @ pose[:3, :3].T + pose[:3, 3]
    return points.astype(np.float32), rgb[rows, cols][valid]


def main() -> None:
    args = parse_args()
    if not 0.1 <= args.speed <= 4.0 or args.point_stride < 1:
        raise ValueError("Use --speed between 0.1 and 4.0 and --point-stride at least 1.")

    import viser
    from viser.extras import ViserUrdf
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    plan = args.plan.resolve()
    plan_dir = plan if plan.is_dir() else plan.parent
    result = load_optimization_result(plan)
    joints = result.planned_joints
    frame_count = len(joints)
    if frame_count < 2:
        raise ValueError("Replay requires a trajectory with at least two frames.")

    urdf = load_robot_description(result.urdf_description)
    urdf.joint_map["panda_hand_tcp_joint"].origin[2, 3] = result.config.tcp_offset_m
    # Refresh yourdfpy's cached fixed-joint transform before Viser reads it.
    urdf.update_cfg({"panda_hand_tcp_joint": 0.0})
    if joints.shape[1] != len(urdf.actuated_joint_names):
        raise ValueError("Saved joint count differs from the replay robot model.")

    cloud = None
    if not args.no_scene:
        try:
            cloud = load_scene_cloud(plan_dir, args.point_stride)
        except (OSError, ValueError, KeyError) as error:
            print(f"WARNING: scene cloud unavailable: {error}; showing robot and paths.")

    trajectories = {
        "TCP": result.achieved_hand_poses_world,
        "Planned task frame": result.achieved_effect_poses_world,
    }
    desired_path = plan_dir / "desired_effect_poses_world.npy"
    if desired_path.exists():
        trajectories["Desired task frame"] = np.load(desired_path, allow_pickle=False)
    quaternions = {
        name: Rotation.from_matrix(poses[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        for name, poses in trajectories.items()
    }

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground", plane="xy", width=2.0, height=2.0,
        position=(0.0, 0.0, result.config.table_z_world),
    )
    robot = ViserUrdf(server, urdf, root_node_name="/robot")
    scene_cloud = None
    if cloud is not None:
        scene_cloud = server.scene.add_point_cloud(
            "/scene", points=cloud[0], colors=cloud[1], point_size=0.003,
        )
    paths_root = server.scene.add_frame("/paths", show_axes=False)
    frames_root = server.scene.add_frame("/frames", show_axes=False)
    frame_handles = {}
    for name, poses in trajectories.items():
        frame_handles[name] = server.scene.add_frame(
            f"/frames/{name}", position=poses[0, :3, 3], wxyz=quaternions[name][0],
            axes_length=0.06, axes_radius=0.002,
        )
        server.scene.add_label(f"/frames/{name}/label", name, position=(0.0, 0.0, 0.07))
        if name != "TCP":
            segments = np.stack((poses[:-1, :3, 3], poses[1:, :3, 3]), axis=1).astype(np.float32)
            color = (30, 210, 60) if name == "Desired task frame" else (50, 100, 255)
            server.scene.add_line_segments(
                f"/paths/{name}", points=segments,
                colors=np.broadcast_to(np.asarray(color, dtype=np.uint8), segments.shape).copy(),
                line_width=3.0,
            )

    server.gui.add_markdown(
        f"**{plan_dir.parent.name}** — {frame_count} frames, "
        f"{(frame_count - 1) * result.config.dt:.2f} s\n\n"
        "Paths: **green = desired**, **blue = planned**. "
        "Axes: X red, Y green, Z blue. The RGB-D cloud shows the initial scene."
    )
    frame = server.gui.add_slider("Frame", min=0, max=frame_count - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    loop = server.gui.add_checkbox("Loop", initial_value=True)
    speed = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=args.speed)
    show_scene = server.gui.add_checkbox("Scene cloud", initial_value=cloud is not None, disabled=cloud is None)
    show_paths = server.gui.add_checkbox("Paths", initial_value=True)
    show_frames = server.gui.add_checkbox("Coordinate frames", initial_value=True)

    @server.on_client_connect
    def set_initial_view(client) -> None:
        center = np.asarray((0.45, 0.0, 0.3))
        client.camera.look_at = center
        client.camera.position = center + np.asarray((1.0, -1.0, 0.75))
        client.camera.up_direction = (0.0, 0.0, 1.0)

    port = server.get_port()
    print(f"Viser: http://localhost:{port} (use the remote PC address when viewing from another computer)")
    print("Use Play or drag Frame. Press Ctrl+C here to close the viewer.")
    previous_time = time.monotonic()
    frame_progress = 0.0
    previous_state = None
    try:
        while True:
            now = time.monotonic()
            elapsed = now - previous_time
            previous_time = now
            if playing.value:
                frame_progress += elapsed * speed.value / result.config.dt
                advance = int(frame_progress)
                if advance:
                    frame_progress -= advance
                    index = frame.value + advance
                    if index >= frame_count and not loop.value:
                        index = frame_count - 1
                        playing.value = False
                    frame.value = index % frame_count
            else:
                frame_progress = 0.0

            state = (frame.value, show_scene.value, show_paths.value, show_frames.value)
            if state != previous_state:
                index = int(frame.value)
                # Only this loop mutates robot configuration; GUI events change controls.
                with server.atomic():
                    robot.update_cfg(joints[index])
                    for name, poses in trajectories.items():
                        frame_handles[name].position = poses[index, :3, 3]
                        frame_handles[name].wxyz = quaternions[name][index]
                    if scene_cloud is not None:
                        scene_cloud.visible = show_scene.value
                    paths_root.visible = show_paths.value
                    frames_root.visible = show_frames.value
                previous_state = state
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
