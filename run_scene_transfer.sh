#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Usage:
#   ./run_scene_transfer.sh TASK_OR_DIR [RUN_NAME]
#   ./run_scene_transfer.sh bottle bottle_transfer
# A task name selects data/<name>; an explicit directory path is also accepted.
#
# Folder layout for EACH pose:
#   data/bottle/
#     demo_rgb.png
#     demo_initial_depth.npy
#     demo.json
#     object_flow_result.pt
#     current_rgb.png
#     current_initial_depth.npy
#     current.json
#     grasp_spec.json
#     current_q.npy              # optional
#
# The demo_* files and object_flow_result.pt can be identical across pose
# folders. Only current_* and grasp_spec.json need to change.
#
# Uses precomputed flow, estimates the current grasp, plans and opens Viser.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${1:?Usage: $0 TASK_OR_DIR [RUN_NAME]}"
if [[ "${TASK_DIR}" != */* ]]; then
    TASK_DIR="${SCRIPT_DIR}/data/${TASK_DIR}"
fi
TASK_DIR="$(realpath "${TASK_DIR}")"
RUN_NAME="${2:-$(basename "$TASK_DIR")}"

VISER_PORT="${VISER_PORT:-8081}"

# Shared calibration stays one level above the individual pose folders.
TASKS_ROOT="$(dirname "$TASK_DIR")"
CAMERA_CALIBRATION="${CAMERA_CALIBRATION:-${TASKS_ROOT}/camera_calibration_info.json}"

# Resolve the project from this launcher, independent of the checkout location.
PIPELINE_ROOT="${SCRIPT_DIR}"

# Load saved Dream2Flow objects from the selected backend, even after a move.
DREAM2FLOW_ROOT="${DREAM2FLOW_ROOT:-${PIPELINE_ROOT}/Dream2Flow}"
DREAM2FLOW_ROOT="$(cd "${DREAM2FLOW_ROOT}" && pwd)"
export PYTHONPATH="${DREAM2FLOW_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PIPELINE_INPUT="${PIPELINE_ROOT}/input"
RUN_DIR="runs/${RUN_NAME}"

# Grasp-only, different-scene mode.
GRASP_ESTIMATOR="${GRASP_ESTIMATOR:-${PIPELINE_ROOT}/scripts/vlm_grasp_pose_multi.py}"
GRASP_POSES_FILE="${GRASP_POSES_FILE:-${PIPELINE_ROOT}/${RUN_DIR}/vlm_grasps/grasp_poses_world.npy}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Missing file: $1" >&2
        exit 1
    fi
}

echo "[Run] ${RUN_NAME}: grasp, different_scene, precomputed flow"
echo "[Input] ${TASK_DIR}"
echo "[Camera] ${CAMERA_CALIBRATION}"

# Check only files needed AFTER flow generation

require_file "${CAMERA_CALIBRATION}"
require_file "${GRASP_ESTIMATOR}"

for f in \
    demo_rgb.png \
    demo_initial_depth.npy \
    demo.json \
    object_flow_result.pt \
    current_rgb.png \
    current_initial_depth.npy \
    current.json \
    grasp_spec.json
do
    require_file "${TASK_DIR}/${f}"
done

mkdir -p "${PIPELINE_INPUT}"

# Estimate grasp pose from the CURRENT RGB-D observation

echo "[Grasp] Estimating from current RGB-D"

# Never allow a failed grasp-estimation run to reuse poses from an older task.
rm -f "${GRASP_POSES_FILE}"

cd "$(dirname "${GRASP_ESTIMATOR}")"

python3 "$(basename "${GRASP_ESTIMATOR}")" \
    --rgb "${TASK_DIR}/current_rgb.png" \
    --depth "${TASK_DIR}/current_initial_depth.npy" \
    --json "${TASK_DIR}/grasp_spec.json" \
    --camera-json "${CAMERA_CALIBRATION}" \
    --output-dir "$(dirname "${GRASP_POSES_FILE}")"

require_file "${GRASP_POSES_FILE}"

# Prepare motion-planning input directory

echo "[Prepare] Planning inputs"

cp "${CAMERA_CALIBRATION}" \
   "${PIPELINE_INPUT}/camera_calibration_info.json"

for f in \
    demo_initial_depth.npy demo_rgb.png demo.json object_flow_result.pt \
    current_initial_depth.npy current_rgb.png current.json grasp_spec.json
do
    cp "${TASK_DIR}/${f}" "${PIPELINE_INPUT}/${f}"
done

if [[ -f "${TASK_DIR}/current_q.npy" ]]; then
    cp "${TASK_DIR}/current_q.npy" \
       "${PIPELINE_INPUT}/current_q.npy"
else
    # Do not accidentally reuse current_q.npy from a previous pose run.
    rm -f "${PIPELINE_INPUT}/current_q.npy"
fi

# Build a temporary runtime config:
#   - grasp mode ONLY
#   - different scene ONLY
#   - current_q.npy, if present, is only a warm-start seed

BASE_CONFIG="${PIPELINE_ROOT}/config/full_pipeline_config.json"
require_file "${BASE_CONFIG}"

RUNTIME_CONFIG="$(mktemp "${PIPELINE_ROOT}/config/full_pipeline_grasp_different_scene.XXXXXX.json")"
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
    "${HAS_CURRENT_Q}" \
    "${GRASP_POSES_FILE}" \
    "${TASK_DIR}" <<'PYCFG'
import json
from pathlib import Path
import sys

src, dst, has_q_text, grasp_poses_file, task_dir = sys.argv[1:]
has_q = bool(int(has_q_text))

with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["replay_scene"] = {
    "rgb": str(Path(task_dir) / "current_rgb.png"),
    "depth": str(Path(task_dir) / "current_initial_depth.npy"),
}

current = cfg.setdefault("current", {})
reconstruction = cfg.setdefault("reconstruction", {})

# Grasp mode ONLY.
current["use_given_q"] = False
current["T_world_grasp"] = None
current["grasp_pose"] = None
current["grasp_poses"] = grasp_poses_file
current["grasp_pose_frame"] = "world"
current["q_candidates"] = None
current["grasp_scores"] = None

# Optional measured q is only a numerical warm-start.
current["q"] = "../input/current_q.npy" if has_q else None

# Different-scene mode ONLY.
reconstruction["same_scene_identity_registration"] = False

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print("[Grasp] Warm-start:", current["q"] or "nominal robot configuration")
PYCFG

# Run full planning / optimization pipeline

echo "[Plan] Preparing object poses and optimizing trajectory"

cd "${PIPELINE_ROOT}"

python3 scripts/prepare_sam3d.py "${RUNTIME_CONFIG}"

python3 scripts/run_full_pipeline.py \
    --config "${RUNTIME_CONFIG}" \
    --output-dir "${RUN_DIR}" \
    --overwrite

echo "[Done] ${RUN_NAME}"
echo "Flow: ${TASK_DIR}/object_flow_result.pt"
echo "Plan: ${PIPELINE_ROOT}/${RUN_DIR}/trajectory"

# Inspect the saved optimized trajectory in Viser.
python3 scripts/replay_optimized_trajectory_viser.py \
    --plan "${RUN_DIR}/trajectory" \
    --port "${VISER_PORT}"
