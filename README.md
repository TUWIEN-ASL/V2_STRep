# V2-STRep


[Project page](https://tuwien-asl.github.io/V2-STRep/) · [arXiv](https://arxiv.org/abs/2609.20582)

## Installation

Requires Linux, Conda, and an NVIDIA GPU. Our environment builds on [Dream2Flow](https://dream2flow.github.io/), with dependency versions pinned for V2-STRep and modifications for GPU memory use compatible with RTX 4000 Ada and pipeline integration. The setup script installs our bundled Dream2Flow version and the additional planning dependencies.

```bash
git clone https://github.com/TUWIEN-ASL/V2_STRep.git
cd V2_STRep
conda create -n v2_strep python=3.10 pip -y
conda activate v2_strep
bash setup_main.sh
```

### Optional: SAM3D

Install SAM3D only if an object-frame method in the configuration is set to `"sam3d"`. For PCA-only operation, set both methods to `"pca"`; no SAM3D installation is needed. The supplied configuration currently selects PCA.

From the project root:

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
```

Follow the [official SAM3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) to create its separate `sam3d-objects` environment and download the checkpoints. You can use `conda` instead of `mamba`. An installation elsewhere is also supported by setting `sam3d.root` below.

After installation, return to the project root and run V2-STRep from its main environment:

```bash
cd ..
conda activate v2_strep
```

## Data preparation

Example tasks are provided in `data/`. Prepare your own task using this structure:

```text
data/
├── camera_calibration_info.json
└── my_task/
    ├── rgb_original.mp4          # Demonstration video
    ├── demo_rgb.png              # Initial demonstration frame
    ├── demo_initial_depth.npy    # Depth aligned with demo_rgb.png
    ├── demo.json                 # Demonstration task annotation
    ├── current_rgb.png           # Current scene
    ├── current_initial_depth.npy # Depth aligned with current_rgb.png
    ├── current.json              # Current-scene task annotation
    ├── grasp_spec.json           # Required for grasp mode
    ├── current_q.npy             # Required for given_q mode
    └── object_flow_result.pt     # Required only for scene transfer
```

1. Capture the demonstration video and its initial RGB-D frame, plus an RGB-D image of the current scene. Depth must be a numeric `(H, W)` NumPy array in **metres**, aligned with the corresponding RGB image.
2. Provide your camera calibration using the [example structure](data/camera_calibration_info.json). The `external_sensor0` entry contains 3×3 `intrinsics` and 4×4 `extrinsics`, where extrinsics map **camera coordinates to robot base/world coordinates**. Keep this file in the task folder's parent directory.
3. Generate the three task JSON files using the prompts below. Our supplied task JSON annotations were generated with **GPT-6 Astra Max**. The launchers read these saved files; they do not make VLM API calls.

| Output | Prompt | Inputs |
| --- | --- | --- |
| `demo.json` | [demo_prompt.txt](prompts/demo_prompt.txt) | Demo video, initial image, instruction, frame count, and image dimensions. |
| `current.json` | [current_prompt.txt](prompts/current_prompt.txt) | Current image, instruction, and `demo.json`. |
| `grasp_spec.json` | [grasp_prompt.txt](prompts/grasp_prompt.txt) | Current image and manipulation instruction. |

Use original-image pixel coordinates `[u, v]`. Phase frame indices are zero-based and inclusive; keep them consistent with the video and saved flow. For `given_q`, save the robot's initial joint vector in `current_q.npy` (normally seven Panda arm angles in radians). The supplied examples use `grasp` and do not include this joint file.

## Running

Run commands from the project root with `v2_strep` active. A task name selects `data/<name>`; an explicit task-directory path also works.

### Full pipeline

Generates depth and object flow from the demonstration video, registers the objects, plans the motion, and opens Viser.

```bash
# Arguments: task, run name, planning mode, scene mode
bash run_full_pipeline.sh bottle bottle_example grasp different_scene
```

- Planning mode: `grasp` uses `grasp_spec.json`; `given_q` uses `current_q.npy`.
- Scene mode: `different_scene` uses your current observation; `same_scene` copies demo RGB-D into the current inputs and regenerates `current.json`. It overwrites those files, so use a task copy if needed; grasp/joint inputs must match that scene.
- If omitted, the run name defaults to the task name, and the modes default to `given_q different_scene`.

When the flow viewer opens on port **8080**, press **Enter in the terminal** to continue. The final trajectory viewer uses port **8081**.

### Scene transfer with saved flow

Skips depth/flow generation and always uses `grasp different_scene`:

```bash
# Arguments: task, run name
bash run_scene_transfer.sh bottle bottle_transfer
```

This requires `object_flow_result.pt` in the task folder; the supplied examples include it. For another current scene, update its RGB-D, `current.json`, and `grasp_spec.json`, keeping the demo inputs and flow unchanged. The full launcher does not automatically save its generated flow back into the task folder.

Results go to `runs/<run_name>/`. Use distinct run names to keep previous results, and run one launcher at a time per checkout because they share staged inputs.

## Configuration

Edit [config/full_pipeline_config.json](config/full_pipeline_config.json):

| Setting | Current value | Options / purpose |
| --- | --- | --- |
| `reconstruction.controlled_object_frame_method` | `"sam3d"` | `"pca"` or `"sam3d"` for the controlled object's frame. |
| `reconstruction.target_object_frame_method` | `"sam3d"` | `"pca"` or `"sam3d"` for targets represented as `object_pose_6d`. |
| `target_pose_registration.controlled_entity_rotation_constraint` | `"full_6d"` | Full rotation and translation, or `"yaw_only"` for rotation about world up plus translation. |
| `target_pose_registration.target_object_rotation_constraint` | `"full_6d"` | The same choice for an `object_pose_6d` target. |
| `sam3d.root` | `"../sam-3d-objects"` | SAM3D checkout path, relative to `config/` or absolute. |

Set both frame methods to `"pca"` to skip SAM3D, or enable `"sam3d"` independently for either object. SAM3D supplies object-frame orientation; demo-to-current matching still uses LoFTR and geometric registration. The rotation constraints apply to that registration, not the entire robot trajectory.

## Replay

```bash
conda activate v2_strep
python scripts/replay_optimized_trajectory_viser.py \
  --plan runs/bottle_example/trajectory \
  --port 8081
```

Keep the original task RGB-D files available: the viewer uses them to display the scene.
