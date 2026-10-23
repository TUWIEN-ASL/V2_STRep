# V2-STRep

## Quick Start

Requires Linux, Conda, and an NVIDIA GPU with a compatible driver.

### 1. Install the main environment

```bash
git clone https://github.com/TUWIEN-ASL/V2_STRep.git
cd V2_STRep

conda create -n v2_strep python=3.10 pip -y
conda activate v2_strep
bash setup_main.sh
```

### 2. Install SAM3D

Clone the official repository inside the V2-STRep project folder:

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
```

Follow the [official SAM3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) to create its separate `sam3d-objects` environment and download the model checkpoints, including the required Hugging Face access approval.

The default configuration already uses this location. After setup, `sam-3d-objects/checkpoints/hf/pipeline.yaml` should exist relative to the project root.

### 3. Run the bottle example

Return to the project root and activate the main environment:

```bash
cd ..
conda activate v2_strep
bash run_full_pipeline.sh bottle bottle_example grasp different_scene
```

This uses the supplied data in `data/bottle/`; no VLM API key is needed. Additional model weights are downloaded on first use.

When the flow viewer opens, press **Enter in the terminal** to continue. The final trajectory opens in Viser on port **8081**, and results are saved under `runs/bottle_example/`.
