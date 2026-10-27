# V2-STRep

Transfer demonstrated object motion to a current scene and optimize a Panda robot trajectory. The pipeline combines Dream2Flow object tracking, geometric task references, object registration, grasp candidates, and trajectory optimization. Object frames can be estimated with PCA or refined SAM3D reconstructions.

[Project page](PROJECT_PAGE_URL) · [arXiv](ARXIV_URL)

<!-- Replace PROJECT_PAGE_URL and ARXIV_URL above with the public links. -->

## Installation

Requires Linux, Conda, and an NVIDIA GPU with a compatible driver. Run the following commands from the same terminal unless otherwise noted.

### Main environment

```bash
git clone https://github.com/TUWIEN-ASL/V2_STRep.git
cd V2_STRep

conda create -n v2_strep python=3.10 pip -y
conda activate v2_strep
bash setup_main.sh
```

`setup_main.sh` installs the bundled Dream2Flow version, its pinned dependencies, the SpaTracker memory patch, and the planning/visualization packages. A separate Dream2Flow environment is not needed. Keep the complete `Dream2Flow/` directory, including its package metadata and README.

### SAM3D environment

The supplied configuration uses SAM3D for object frames. Clone the official repository inside the project root:

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
```

Follow the [official SAM3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) to create its separate `sam3d-objects` environment and download the model checkpoints. This includes requesting the required Hugging Face access. If `mamba` is unavailable, the official instructions allow using `conda` instead. Activate the SAM3D environment before installing its Python packages.

After setup, `sam-3d-objects/checkpoints/hf/pipeline.yaml` should exist relative to the V2-STRep root. Return to the main environment before launching V2-STRep:

```bash
cd ..
conda activate v2_strep
```

An existing SAM3D installation elsewhere can also be used: set `sam3d.root` and `sam3d.environment` in the [configuration](config/full_pipeline_config.json). V2-STRep invokes that environment automatically. Its reconstruction and refinement wrappers live in `sam3d_tools/`; the official SAM3D core does not need to be edited.

To use only PCA, set both object-frame methods to `"pca"` as described below; SAM3D installation is then unnecessary.

## Quick example

From the project root:

```bash
conda activate v2_strep
bash run_full_pipeline.sh bottle bottle_example grasp different_scene
```

This uses `data/bottle/`. The supplied annotations are already saved, so no VLM API key is needed. Additional model weights are downloaded on first use.

The flow viewer uses port **8080**. Press **Enter in the terminal** when finished inspecting the flow to continue planning. The final trajectory opens in Viser on port **8081**, and outputs are saved under `runs/bottle_example/`. Use the viewer URL printed in the terminal; for a remote machine, access or forward its viewer ports.

To try the same sample using its supplied flow, without regenerating depth and flow:

```bash
bash run_scene_transfer.sh bottle bottle_transfer
```

Both launchers generate and visualize a trajectory; they do not execute it on the physical robot.

## Repository layout

```text
V2_STRep/
├── config/                 # Planning configuration and Panda collision spheres
├── data/                   # Task inputs and camera calibration
├── prompts/                # Templates for preparing task and grasp annotations
├── motion_planning/        # Reconstruction, registration, references, optimization
├── scripts/                # Planning entry points, grasp processing, Viser replay
├── sam3d_tools/             # SAM3D reconstruction and scene-fitting wrappers
├── Dream2Flow/              # Bundled modified Dream2Flow, with installed deps/
├── sam-3d-objects/          # Official external checkout; installed separately
├── input/                  # Temporary shared inputs staged by the launchers
├── sam3d_cache/             # Reusable reconstruction/refinement results
├── runs/                   # Saved trajectories and visualizations
├── setup_main.sh
├── run_full_pipeline.sh
└── run_scene_transfer.sh
```

The checkout can have any directory name; it does not have to be named `V2_STRep`. Commands in this README assume the project root is the working directory. If you move an already installed checkout, rerun `setup_main.sh` from its new location to refresh the editable package paths.

## Task data

The supplied task folders are `apple`, `bottle`, `drawer`, `micro` (microwave), `sponge`, and `tap`. Use the same layout for a new task:

```text
data/
├── camera_calibration_info.json
└── my_task/
    ├── demo_rgb.png
    ├── demo_initial_depth.npy
    ├── demo.json
    ├── current_rgb.png
    ├── current_initial_depth.npy
    ├── current.json
    ├── grasp_spec.json
    ├── rgb_original.mp4
    ├── object_flow_result.pt
    └── current_q.npy            # Add when using given_q; optional in grasp mode
```

| File | Purpose / requirement |
| --- | --- |
| `demo_rgb.png` | Initial demonstration image used for object and target grounding. |
| `demo_initial_depth.npy` | Measured depth aligned with the demonstration image. |
| `demo.json` | Demonstration instruction, phase structure, rotation policies, and initial grounding. |
| `current_rgb.png` | Image of the scene where the robot will perform the task. |
| `current_initial_depth.npy` | Measured depth aligned with the current image. |
| `current.json` | Current-scene grounding and per-phase rotation policies corresponding to `demo.json`. |
| `grasp_spec.json` | Image-space grasp candidates; required for `grasp` mode and scene transfer. |
| `rgb_original.mp4` | Demonstration video for the full launcher. `rgb.mp4` is accepted if this file is absent. |
| `object_flow_result.pt` | Precomputed Dream2Flow result for the scene-transfer launcher. |
| `current_q.npy` | Initial robot joints for `given_q`; optional optimization seed in `grasp` mode. |

The supplied examples include grasp specifications and saved flow, but do not include `current_q.npy`. Use `grasp` mode for those examples.

### Images, depth, and calibration

- Each depth file must be a numeric NumPy array of shape `(H, W)`, registered to its RGB image at the same resolution. Store depth in **metres**.
- Pixels use original-image coordinates `[u, v] = [column, row]`, with the origin at the top left. Bounding boxes use `[x_min, y_min, x_max, y_max]`. Do not use normalized coordinates or coordinates from a resized preview.
- `demo_rgb.png` and its measured depth should correspond to the initial demonstration frame. Phase indices in `demo.json` are **zero-based and inclusive**, and refer to the video/flow frame sequence. If you trim or resample a video, update the annotations and flow together.
- Put `camera_calibration_info.json` in the **parent of the task folder**, including when supplying an external task path. See the [included calibration file](data/camera_calibration_info.json) for its structure.
- The default workflow expects one camera entry named `external_sensor0`, containing a 3×3 `intrinsics` matrix and a 4×4 `extrinsics` matrix. The latter is **camera optical frame → robot base/world**, `T_world_camera`, not its inverse. Replace the example calibration with your own measurements.

Keep `camera.depth_scale_to_m` at `1.0` for this workflow. Convert millimetre depth arrays to metres before saving them: changing that configuration field alone does not scale every launcher stage, particularly grasp reconstruction.

For multiple cameras, selection must be consistent across stages. The full launcher accepts `CAMERA_NAME`, but the scene-transfer grasp stage currently uses `external_sensor0`. The planner's `camera.intrinsics_key` and `camera.T_world_camera_key` can select nested fields, for example `external_sensor0.intrinsics` and `external_sensor0.extrinsics`. A single calibrated `external_sensor0` entry is the simplest supported layout.

### Preparing the annotations

The launchers read existing JSON annotations; they do not call a VLM to generate them. Use the supplied examples as schema references and the following prompts to prepare a new task:

| Output | Prompt | Information to provide |
| --- | --- | --- |
| `demo.json` | [demo_prompt.txt](prompts/demo_prompt.txt) | Demonstration video, initial image, instruction, frame count, and image dimensions. |
| `current.json` | [current_prompt.txt](prompts/current_prompt.txt) | Current image, current instruction, and the demonstration JSON. |
| `grasp_spec.json` | [grasp_prompt.txt](prompts/grasp_prompt.txt) | Current image and manipulation instruction. |

`demo.json` contains `task_instruction`, `video_structure`, and `demo_initial_grounding`. `current.json` keeps the demonstration's phase IDs and supplies `phase_rotation_policies` and `current_grounding`; it does not redefine the phase sequence. `grasp_spec.json` contains a nonempty `grasps` list with pixel anchors and opening-axis information. The grasp script converts these annotations and measured depth into metric TCP poses.

Use the existing prompt conventions with either frame method. SAM3D object axes are reordered and signed to match the convention used by the task specification.

## Running the launchers

### Full pipeline: regenerate flow and plan

```bash
bash run_full_pipeline.sh TASK_OR_DIR [RUN_NAME] [given_q|grasp] [same_scene|different_scene]
```

| Argument | Meaning | Default |
| --- | --- | --- |
| `TASK_OR_DIR` | A name under `data/`, or a task-directory path such as `./data/my_task`. | Required |
| `RUN_NAME` | Output folder name under `runs/`. | Task folder name |
| Planning mode | `given_q` starts from supplied robot joints; `grasp` plans from image-derived grasp candidates. | `given_q` |
| Scene mode | `different_scene` registers demo objects to current objects; `same_scene` uses the demo scene directly. | `different_scene` |

Examples:

```bash
# Supplied examples, using their current-scene grasp annotations.
bash run_full_pipeline.sh apple apple_example grasp different_scene
bash run_full_pipeline.sh tap tap_example grasp different_scene

# Your task, with current_q.npy specifying the starting robot configuration.
bash run_full_pipeline.sh ./data/my_task my_task_given_q given_q different_scene
```

For `given_q`, save a finite one-dimensional joint vector in `current_q.npy`; the usual Panda arm input is seven joint angles in radians. Full-model vectors follow the robot's actuated-joint order and joint units. In `grasp` mode, the file is optional and only supplies an optimization seed; the selected grasp constrains the starting TCP pose.

**`same_scene` changes the task inputs:** it replaces `current_rgb.png` and `current_initial_depth.npy` with their demo counterparts and regenerates `current.json` from `demo.json`. Registration then uses the identity transform. It does not update `grasp_spec.json` or `current_q.npy`; those must already match the demo scene. Use a separate task copy if you need to preserve the original current scene.

The full launcher generates depth and flow, prepares object frames and grasps, optimizes the trajectory, and opens Viser. It recomputes flow even if the task folder already contains `object_flow_result.pt`. Large depth tensors are removed after flow generation, and `runs/<run>/dream2flow/` is removed after successful planning. The generated flow is **not** automatically saved back into the task folder.

### Scene transfer: reuse saved demonstration flow

```bash
bash run_scene_transfer.sh TASK_OR_DIR [RUN_NAME]
```

```bash
bash run_scene_transfer.sh bottle bottle_transfer
bash run_scene_transfer.sh ./data/my_task my_task_transfer
```

This launcher always uses **`grasp` + `different_scene`**. It requires a compatible `object_flow_result.pt` in the task folder, together with demo/current RGB-D, both task JSON files, and `grasp_spec.json`. It skips video depth/flow generation and the flow viewer, then performs registration, grasp generation, planning, and final visualization. The original saved flow is retained.

Use it to evaluate a new current scene while keeping the same demonstration and its flow. Update `current_rgb.png`, `current_initial_depth.npy`, `current.json`, and `grasp_spec.json` together. A completed full-pipeline run does not automatically create the task-local flow file required here; the supplied examples already include one.

Both launchers overwrite outputs with the same run name and share the project's `input/` staging directory. Use distinct run names to preserve comparisons, and run one launcher at a time per checkout.

## Configuration

Edit [config/full_pipeline_config.json](config/full_pipeline_config.json). Relative paths in this file are resolved from **`config/`**, so `../sam-3d-objects` points to the checkout in the project root.

The tables below describe the **shipped configuration**, rather than every fallback used when a key is omitted. Keep the remaining settings initially and adjust only what your data or experiment requires.

### Object frames and registration

| Setting | Shipped value | When to change it |
| --- | --- | --- |
| `reconstruction.controlled_object_frame_method` | `"sam3d"` | Use `"pca"` for a frame estimated directly from the observed point cloud. |
| `reconstruction.target_object_frame_method` | `"sam3d"` | Same choice for targets represented as `object_pose_6d`. Plane, point-normal, and axis-line targets use their geometric representations. |
| `target_pose_registration.controlled_entity_rotation_constraint` | `"full_6d"` | Use `"yaw_only"` if demo-to-current object changes should be limited to rotation about world up, plus translation. |
| `target_pose_registration.target_object_rotation_constraint` | `"full_6d"` | Apply the same restriction to an `object_pose_6d` target. |
| `reconstruction.world_up` | `[0, 0, 1]` | Up direction for object frames and yaw registration. The grasp/table workflow still assumes a Z-up world; this is not a global coordinate-frame switch. |

For PCA-only operation, change these two existing entries:

```json
"controlled_object_frame_method": "pca",
"target_object_frame_method": "pca"
```

These belong inside `reconstruction`; keep its other entries. Mixed PCA/SAM3D operation is also supported.

SAM3D supplies refined **object-frame orientation**; demonstrated controlled-object translations still come from the tracked reconstruction, and the target retains its reconstructed origin. Demo-to-current matching still uses LoFTR with geometric registration. `yaw_only` constrains that initial registration, not the robot's full trajectory or its phase-specific rotation policy, and does not force an initially tilted object upright.

### SAM3D location and refinement

| `sam3d` setting | Shipped value | Meaning |
| --- | --- | --- |
| `root` | `"../sam-3d-objects"` | Official checkout path; an absolute path to an existing installation also works. |
| `environment` | `"sam3d-objects"` | Conda environment used for reconstruction/refinement. |
| `seed` | `42` | Reconstruction seed. |
| `scale_mode` | `"axes"` | Independent XYZ size refinement; `"uniform"` uses one scale multiplier. |
| `scale_bound` | `2.0` | Allows scale multipliers between `0.5` and `2.0`; must exceed `1`. |
| `starts` | `6` | Maximum number of distinct refinement starts; fewer may reduce runtime. |

Matching completed results are reused from `sam3d_cache/`. The cache depends on the demonstration inputs, grounding, calibration, SAM3D settings, and relevant implementation/configuration files. Changing only the current scene can reuse the demo reconstruction; current-scene registration still runs. Changing reconstruction inputs or settings may trigger a rebuild.

### Contact, timing, and robot setup

| Setting | Shipped value | Meaning |
| --- | --- | --- |
| `motion_reference.plane_press_down_m` | `0.01` | For each learned plane-contact phase, retain its demonstrated entry distance, then shift 1 cm along negative plane Z. Use `0.0` for no additional press-down. |
| `optimizer.dt` | `0.05` | Trajectory sample interval in seconds; affects optimization and playback. It is not inferred automatically from video FPS. |
| `optimizer.tcp_offset_m` | `0.1234` | Panda TCP offset along its fixed joint's Z direction; match the installed tool. |
| `optimizer.table_z_world` | `0.0` | World height of the horizontal collision plane. |
| `robot.sphere_json` | `"panda_spheres.json"` | Collision-sphere model, relative to `config/`; match the robot/tool geometry. |

Plane contact holds the adjusted entry distance throughout the phase. Setting press-down to zero does not restore variations in demonstrated normal distance, and the offset follows the plane normal rather than necessarily world Z.

Registration thresholds, track filtering, smoothing, and optimizer weights are grouped in their corresponding sections. Keep them unchanged for an initial comparison. If adjusting `multi_grasp` error thresholds, note that they prefer candidates meeting the limits: if none pass, the planner warns and can still export the best successfully optimized candidate. Inspect the saved metrics rather than assuming that an exported trajectory passed every threshold.

The launchers prepare temporary configuration copies and fill in task inputs, grasp/joint mode, and generated SAM3D pose paths. Editing those runtime-input fields in the base JSON may be overridden by the selected launcher. Depth-generation window/overlap and some flow settings are currently shell arguments, not JSON settings.

## Outputs and replay

Successful runs keep three result folders:

```text
runs/bottle_example/
├── trajectory/                 # Final plan, metrics, configuration, replay metadata
├── object_pose_registration/   # Object frames, registration diagnostics and overlays
└── vlm_grasps/                 # Grasp poses/overlay in grasp mode, candidate comparison
```

Important files inside `trajectory/`:

| File | Contents |
| --- | --- |
| `joint_trajectory.npy` | Planned joint samples in the robot model's actuated-joint order. |
| `ee_trajectory.npy` / `.csv` | Achieved TCP poses in world coordinates: `x, y, z, qx, qy, qz, qw`; translation in metres. |
| `trajectory_summary.json` | Array shapes, sample interval, quaternion order, and endpoint poses. |
| `optimization_result.npz` | Full saved optimization result used by the viewer. |
| `desired_effect_poses_world.npy` | Desired task-frame trajectory for comparison with the optimized motion. |
| `optimization_metrics.json` | Optimization and task-error diagnostics. |
| `config_used.json` / `run_manifest.json` | Effective configuration and run provenance. |
| `replay_scene.json` | Calibration and references to the original current-scene RGB-D files. |

Inspect `object_pose_registration/poses.json` and its controlled-object/target visualizations to check object frames before interpreting the trajectory. `vlm_grasps/` contains reconstructed grasp candidates; despite the folder name, replaying the supplied annotations requires no VLM request.

Reopen a saved plan from the project root:

```bash
conda activate v2_strep
python scripts/replay_optimized_trajectory_viser.py \
  --plan runs/bottle_example/trajectory \
  --port 8081
```

Keep the saved trajectory directory and the original task RGB-D files available and unchanged: the viewer references those scene inputs rather than saving another copy. Add `--no-scene` to display the robot, paths, and frames without the scene point cloud. Set `VISER_PORT=8082` before either launcher if you need a different final-viewer port.
