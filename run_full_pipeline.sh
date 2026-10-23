#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Usage:
#   ./run_full_pipeline.sh \
#       TASK_OR_DIR [RUN_NAME] [MODE] [SCENE_MODE]
# A task name selects data/<name>; an explicit directory path is also accepted.
#
# MODE:
#   given_q  - use current_q.npy as the fixed initial robot configuration
#   grasp    - estimate grasp pose from current RGB-D; current_q.npy is optional
#              and, when present, is used only as an optimization warm-start.
#
# SCENE_MODE:
#   different_scene - use the provided current_rgb.png, current_initial_depth.npy,
#                     and current.json, and run normal demo->current registration.
#   same_scene      - overwrite/create:
#                       current_rgb.png           <- demo_rgb.png
#                       current_initial_depth.npy <- demo_initial_depth.npy
#                       current.json              <- generate_current.py(demo.json)
#                     and request identity demo->current registration.
#
# Examples:
#   ./run_full_pipeline.sh \
#       bottle bottle_demo grasp same_scene
#
#   ./run_full_pipeline.sh \
#       bottle bottle_transfer grasp different_scene

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${1:?Usage: $0 TASK_OR_DIR [RUN_NAME] [given_q|grasp] [same_scene|different_scene]}"
if [[ "${TASK_DIR}" != */* ]]; then
    TASK_DIR="${SCRIPT_DIR}/data/${TASK_DIR}"
fi
TASK_DIR="$(realpath "${TASK_DIR}")"
RUN_NAME="${2:-$(basename "$TASK_DIR")}"

VISER_PORT="${VISER_PORT:-8081}"
PLANNING_MODE="${3:-${PLANNING_MODE:-given_q}}"
SCENE_MODE="${4:-${SCENE_MODE:-different_scene}}"
GENERATE_CURRENT="${GENERATE_CURRENT:-${SCRIPT_DIR}/scripts/generate_current.py}"

case "${PLANNING_MODE}" in
    given_q|given-q|q)
        PLANNING_MODE="given_q"
        ;;
    grasp|grasp_pose|grasp-pose)
        PLANNING_MODE="grasp"
        ;;
    *)
        echo "[ERROR] MODE must be 'given_q' or 'grasp'; got: ${PLANNING_MODE}" >&2
        exit 2
        ;;
esac

case "${SCENE_MODE}" in
    same_scene|same-scene|same|true|1|yes|y)
        SCENE_MODE="same_scene"
        ;;
    different_scene|different-scene|different|false|0|no|n)
        SCENE_MODE="different_scene"
        ;;
    *)
        echo "[ERROR] SCENE_MODE must be 'same_scene' or 'different_scene'; got: ${SCENE_MODE}" >&2
        exit 2
        ;;
esac

# Shared files live one level above the individual task.
TASKS_ROOT="$(dirname "$TASK_DIR")"
CAMERA_CALIBRATION="${TASKS_ROOT}/camera_calibration_info.json"

# Resolve the project from this launcher, independent of the checkout location.
PIPELINE_ROOT="${SCRIPT_DIR}"

# Use the bundled backend unless an external checkout is explicitly selected.
DREAM2FLOW_ROOT="${DREAM2FLOW_ROOT:-${PIPELINE_ROOT}/Dream2Flow}"
if [[ ! -d "${DREAM2FLOW_ROOT}" ]]; then
    echo "[ERROR] Dream2Flow directory not found: ${DREAM2FLOW_ROOT}" >&2
    echo "[ERROR] Set DREAM2FLOW_ROOT to your patched Dream2Flow checkout." >&2
    exit 1
fi
DREAM2FLOW_ROOT="$(cd "${DREAM2FLOW_ROOT}" && pwd)"
# Saved flow objects must load from the same backend during planning.
export PYTHONPATH="${DREAM2FLOW_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_DIR="runs/${RUN_NAME}"

# Used only in PLANNING_MODE=grasp.
GRASP_ESTIMATOR="${GRASP_ESTIMATOR:-${PIPELINE_ROOT}/scripts/vlm_grasp_pose_multi.py}"
GRASP_POSES_FILE="${GRASP_POSES_FILE:-${PIPELINE_ROOT}/${RUN_DIR}/vlm_grasps/grasp_poses_world.npy}"

SCENE_DIR="$(realpath -m "${PIPELINE_ROOT}/${RUN_DIR}/dream2flow")"
case "${TASK_DIR}/" in
    "${SCENE_DIR}/"*)
        echo "[ERROR] Task inputs must be outside the temporary Dream2Flow run folder." >&2
        exit 2
        ;;
esac
PIPELINE_INPUT="${PIPELINE_ROOT}/input"
DEPTH_FRAMES="${SCENE_DIR}/depth_frames.pt"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Missing file: $1" >&2
        exit 1
    fi
}

echo "[Run] ${RUN_NAME}: ${PLANNING_MODE}, ${SCENE_MODE}, SpaTracker depth"
echo "[Input] ${TASK_DIR}"

# Prepare current observation for same-scene mode.
#
# This intentionally overwrites current_* files when SCENE_MODE=same_scene.
# generate_current.py lives in the project scripts directory.

require_file "${CAMERA_CALIBRATION}"
require_file "${TASK_DIR}/demo_rgb.png"
require_file "${TASK_DIR}/demo_initial_depth.npy"
require_file "${TASK_DIR}/demo.json"

if [[ "${SCENE_MODE}" == "same_scene" ]]; then
    require_file "${GENERATE_CURRENT}"

    echo "[Scene] Copying demo to current observation (overwrites current_* files)"

    cp -f "${TASK_DIR}/demo_rgb.png" \
          "${TASK_DIR}/current_rgb.png"

    cp -f "${TASK_DIR}/demo_initial_depth.npy" \
          "${TASK_DIR}/current_initial_depth.npy"

    python3 "${GENERATE_CURRENT}" \
        "${TASK_DIR}/demo.json" \
        "${TASK_DIR}/current.json"
fi

# Check required files

for f in current_rgb.png current.json current_initial_depth.npy; do
    require_file "${TASK_DIR}/${f}"
done

if [[ "${PLANNING_MODE}" == "given_q" ]]; then
    require_file "${TASK_DIR}/current_q.npy"
else
    require_file "${TASK_DIR}/grasp_spec.json"
    require_file "${GRASP_ESTIMATOR}"
fi

if [[ ! -f "${TASK_DIR}/rgb.mp4" && ! -f "${TASK_DIR}/rgb_original.mp4" ]]; then
    echo "[ERROR] Need either ${TASK_DIR}/rgb.mp4 or ${TASK_DIR}/rgb_original.mp4" >&2
    exit 1
fi

mkdir -p "${SCENE_DIR}" "${PIPELINE_INPUT}"

# Prepare Dream2Flow scene

echo "[Prepare] Dream2Flow scene"

cp "${CAMERA_CALIBRATION}" \
   "${SCENE_DIR}/camera_calibration_info.json"

cp "${TASK_DIR}/demo_rgb.png" \
   "${SCENE_DIR}/camera_rgb.png"

# Keep the full video. Prefer rgb_original.mp4 when it exists so an older
# downsampled rgb.mp4 cannot silently reduce the temporal resolution.
if [[ -f "${TASK_DIR}/rgb_original.mp4" ]]; then

    echo "[Video] Using full rgb_original.mp4 (no temporal downsampling)"

    cp "${TASK_DIR}/rgb_original.mp4" \
       "${SCENE_DIR}/rgb.mp4"

elif [[ -f "${TASK_DIR}/rgb.mp4" ]]; then

    echo "[Video] rgb_original.mp4 not found; using existing rgb.mp4"

    cp "${TASK_DIR}/rgb.mp4" \
       "${SCENE_DIR}/rgb.mp4"

fi

# Convert demo depth .npy -> .pt for Dream2Flow

python3 - \
    "${TASK_DIR}/demo_initial_depth.npy" \
    "${SCENE_DIR}/initial_depth.pt" <<'PY'

import sys
import numpy as np
import torch

depth = np.load(sys.argv[1])
torch.save(torch.from_numpy(depth), sys.argv[2])
PY

# Read controlled entity bbox from demo.json

read -r BBOX_X1 BBOX_Y1 BBOX_X2 BBOX_Y2 < <(

python3 - "${TASK_DIR}/demo.json" <<'PY'

import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

bbox = data["demo_initial_grounding"]["controlled_entity"]["bbox"]

if not isinstance(bbox, list) or len(bbox) != 4:
    raise ValueError(
        "demo_initial_grounding.controlled_entity.bbox "
        "must be [x_min, y_min, x_max, y_max]"
    )

print(*bbox)
PY

)

# Write Dream2Flow scene metadata.

python3 - \
    "${TASK_DIR}/demo.json" \
    "${SCENE_DIR}/scene_data.yaml" <<'PY'

import json
import sys
import yaml

demo_json = sys.argv[1]
output_yaml = sys.argv[2]

with open(demo_json, "r", encoding="utf-8") as f:
    data = json.load(f)

instruction = str(data.get("task_instruction", "manipulate the object")).strip()
object_name = str(
    data.get("video_structure", {}).get("controlled_entity", {}).get("description", "object")
).strip()

scene = {
    "instruction": instruction or "manipulate the object",
    "object_name": object_name or "object",
}

with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(scene, f, sort_keys=False)

PY

# Detect camera name automatically

CAMERA_NAME="${CAMERA_NAME:-$(

python3 - "${CAMERA_CALIBRATION}" <<'PY'

import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict) or not data:
    raise ValueError(
        "camera_calibration_info.json must contain at least one camera"
    )

print(next(iter(data.keys())))

PY

)}"

echo "[Camera] ${CAMERA_NAME} (${CAMERA_CALIBRATION})"

# Release depth-generation GPU memory before starting flow tracking.
echo "[Depth] Generating full-video depth"

cd "${DREAM2FLOW_ROOT}"

python -m dream2flow.scripts.generate_depth \
    --scene-dir "${SCENE_DIR}" \
    --video-path "${SCENE_DIR}/rgb.mp4" \
    --output-path "${DEPTH_FRAMES}" \
    --window 48 \
    --overlap 8 \
    --device cuda

require_file "${DEPTH_FRAMES}"

# Generate 3D object flow in a fresh Python process using saved depth

echo "[Flow] Generating 3D object flow from saved depth"

python -m dream2flow.scripts.create_3d_flow \
    --scene-dir "${SCENE_DIR}" \
    --video-generation-method "local file" \
    --video-path "${SCENE_DIR}/rgb.mp4" \
    --depth-mode "playback" \
    --depth-frames "${DEPTH_FRAMES}" \
    --initial-depth "${SCENE_DIR}/initial_depth.pt" \
    --camera-calibration "${SCENE_DIR}/camera_calibration_info.json" \
    --camera-name "${CAMERA_NAME}" \
    --start-image "${SCENE_DIR}/camera_rgb.png" \
    --scene-data "${SCENE_DIR}/scene_data.yaml" \
    --output-path "${SCENE_DIR}/object_flow_result.pt" \
    --object-bbox \
        "${BBOX_X1}" \
        "${BBOX_Y1}" \
        "${BBOX_X2}" \
        "${BBOX_Y2}" \
    --device cuda

# create_3d_flow.py waits for Enter to close its viewer before planning continues.

require_file "${SCENE_DIR}/object_flow_result.pt"
# Video depth is only needed between depth generation and flow tracking.
rm -f -- "${DEPTH_FRAMES}"

# Select planning start mode

if [[ "${PLANNING_MODE}" == "grasp" ]]; then
    echo "[Grasp] Estimating from current RGB-D"

    # Never let a failed grasp-estimation run silently reuse an old pose file.
    rm -f "${GRASP_POSES_FILE}"

    cd "$(dirname "${GRASP_ESTIMATOR}")"

    python3 "$(basename "${GRASP_ESTIMATOR}")" \
        --rgb "${TASK_DIR}/current_rgb.png" \
        --depth "${TASK_DIR}/current_initial_depth.npy" \
        --json "${TASK_DIR}/grasp_spec.json" \
        --camera-json "${CAMERA_CALIBRATION}" \
        --camera-name "${CAMERA_NAME}" \
        --output-dir "$(dirname "${GRASP_POSES_FILE}")"

    require_file "${GRASP_POSES_FILE}"

    if [[ -f "${TASK_DIR}/current_q.npy" ]]; then
        echo "[Grasp] Warm-start: current_q.npy"
    else
        echo "[Grasp] Warm-start: nominal robot configuration"
    fi

    cd "${DREAM2FLOW_ROOT}"
else
    echo "[Grasp] Fixed start: ${TASK_DIR}/current_q.npy"
fi

# Prepare motion-planning input directory

echo "[Prepare] Planning inputs"

cp "${CAMERA_CALIBRATION}" \
   "${PIPELINE_INPUT}/camera_calibration_info.json"

for f in demo_initial_depth.npy current_initial_depth.npy demo_rgb.png current_rgb.png; do
    cp "${TASK_DIR}/${f}" "${PIPELINE_INPUT}/${f}"
done

# Avoid stale q from an older task/run.
rm -f "${PIPELINE_INPUT}/current_q.npy"

if [[ -f "${TASK_DIR}/current_q.npy" ]]; then
    cp "${TASK_DIR}/current_q.npy" \
       "${PIPELINE_INPUT}/current_q.npy"
fi

for f in demo.json current.json; do
    cp "${TASK_DIR}/${f}" "${PIPELINE_INPUT}/${f}"
done

if [[ "${PLANNING_MODE}" == "grasp" ]]; then
    cp "${TASK_DIR}/grasp_spec.json" \
       "${PIPELINE_INPUT}/grasp_spec.json"
else
    rm -f "${PIPELINE_INPUT}/grasp_spec.json"
fi

# Keep the runtime config beside its source so relative paths stay consistent.

BASE_CONFIG="${PIPELINE_ROOT}/config/full_pipeline_config.json"
require_file "${BASE_CONFIG}"

RUNTIME_CONFIG="$(mktemp "${PIPELINE_ROOT}/config/full_pipeline_spatracker_${PLANNING_MODE}.XXXXXX.json")"
cleanup_runtime_config() {
    rm -f "${RUNTIME_CONFIG}"
}
trap cleanup_runtime_config EXIT

HAS_CURRENT_Q=0
if [[ -f "${PIPELINE_INPUT}/current_q.npy" ]]; then
    HAS_CURRENT_Q=1
fi

python3 - \
    "${BASE_CONFIG}" \
    "${RUNTIME_CONFIG}" \
    "${PLANNING_MODE}" \
    "${SCENE_MODE}" \
    "${HAS_CURRENT_Q}" \
    "${GRASP_POSES_FILE}" \
    "${SCENE_DIR}/object_flow_result.pt" \
    "${TASK_DIR}" <<'PY'
import json
from pathlib import Path
import sys

src, dst, mode, scene_mode, has_q_text, grasp_poses_file, object_flow_file, task_dir = sys.argv[1:]
has_q = bool(int(has_q_text))

with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["replay_scene"] = {
    "rgb": str(Path(task_dir) / "current_rgb.png"),
    "depth": str(Path(task_dir) / "current_initial_depth.npy"),
}

cfg["demo"]["object_flow"] = object_flow_file
current = cfg.setdefault("current", {})
# Clear optional inputs from earlier runs in either planning mode.
for key in ("T_world_grasp", "grasp_pose", "q_candidates", "grasp_scores"):
    current[key] = None

if mode == "given_q":
    current["use_given_q"] = True
    current["q"] = "../input/current_q.npy"

    current["grasp_poses"] = None

elif mode == "grasp":
    current["use_given_q"] = False

    current["grasp_poses"] = grasp_poses_file
    current["grasp_pose_frame"] = "world"

    # If measured q is available, keep it only as a numerical warm-start.
    current["q"] = "../input/current_q.npy" if has_q else None

else:
    raise ValueError(f"Unsupported planning mode: {mode}")

reconstruction = cfg.setdefault("reconstruction", {})

if scene_mode == "same_scene":
    # Reuse demo poses without estimating a demo-to-current transform.
    reconstruction["same_scene_identity_registration"] = True
elif scene_mode == "different_scene":
    reconstruction["same_scene_identity_registration"] = False
else:
    raise ValueError(f"Unsupported scene mode: {scene_mode}")

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

PY

# Run full planning pipeline

echo "[Plan] Preparing object poses and optimizing trajectory"

cd "${PIPELINE_ROOT}"

python3 scripts/prepare_sam3d.py "${RUNTIME_CONFIG}"

python3 scripts/run_full_pipeline.py \
    --config "${RUNTIME_CONFIG}" \
    --output-dir "${RUN_DIR}" \
    --overwrite

# Flow staging is no longer needed after planning; Viser uses the task RGB-D.
rm -rf -- "${SCENE_DIR}"

echo "[Done] ${RUN_NAME}"
echo "Plan: ${PIPELINE_ROOT}/${RUN_DIR}/trajectory"

# Inspect the saved optimized trajectory in Viser.
python3 scripts/replay_optimized_trajectory_viser.py \
    --plan "${RUN_DIR}/trajectory" \
    --port "${VISER_PORT}"
