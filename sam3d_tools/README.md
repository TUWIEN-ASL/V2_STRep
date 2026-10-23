# SAM3D integration

These project-owned scripts use an external official SAM3D checkout:

- `demo_full.py`: RGB or RGB-D reconstruction with sequential GPU offloading.
- `align_scene_depth.py`: align predicted scene depth to measured depth.
- `fit_mesh_depth.py`: refine mesh pose and size using depth, normals and silhouettes.
- `visualize_pose.py`: shared pose/mesh overlays and a standalone visualization command.

The normal pipeline launches them through `../scripts/prepare_sam3d.py`. In
`../config/full_pipeline_config.json`, `sam3d.root` locates the official checkout
relative to that config file; `sam3d.environment` selects its Conda environment.
The launcher uses this checkout as its working directory and passes it explicitly
to reconstruction.
Neither the project folder name nor the working directory selects the wrappers.

For standalone reconstruction, run from the project root in the SAM3D environment:

```bash
python sam3d_tools/demo_full.py --sam3d-root ../../sam-3d-objects \
  --image /path/to/image.png --mask /path/to/mask.png --output /path/to/new-output
```

Without `--sam3d-root`, reconstruction expects `external/sam-3d-objects` inside
this project. The existing configuration still selects your current checkout.
Checkpoint and example-image defaults remain relative to the official checkout.

`dependencies.json` records the upstream revision used when separating these
scripts. It does not install or enforce that revision. Install the official SAM3D
environment and checkpoints separately.
The old custom scripts in the external checkout are no longer used by the launcher.
