#!/usr/bin/env bash
# From the project root:
#   conda create -n v2_strep python=3.10 pip -y
#   conda activate v2_strep
#   bash setup_main.sh
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    echo "Activate a dedicated Conda environment first (recommended: v2_strep)." >&2
    exit 1
fi
if [[ "$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "3.10" ]]; then
    echo "The main environment requires Python 3.10." >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAM_ROOT="${PROJECT_ROOT}/Dream2Flow"
cd "${PROJECT_ROOT}"

# Limit package-index overrides to this script and its child processes.
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple

checkout() {
    local path="$1" url="$2" revision="$3" actual
    if [[ ! -e "${path}" ]]; then
        git clone "${url}" "${path}"
        git -C "${path}" checkout --detach "${revision}"
    else
        actual="$(git -C "${path}" rev-parse HEAD)"
        if [[ "${actual}" != "${revision}" ]]; then
            echo "Expected ${revision} at ${path}; existing checkout was left unchanged." >&2
            exit 1
        fi
    fi
}

echo "Preparing Dream2Flow dependencies..."
mkdir -p "${DREAM_ROOT}/deps"
# Revisions are recorded in Dream2Flow/dependencies.json.
checkout "${DREAM_ROOT}/deps/co-tracker" \
    https://github.com/facebookresearch/co-tracker.git \
    82e02e8029753ad4ef13cf06be7f4fc5facdda4d
checkout "${DREAM_ROOT}/deps/SpaTrackerV2" \
    https://github.com/henry123-boy/SpaTrackerV2.git \
    7e12274c52077860cebfe007a6290777db43b63c

SPA_ROOT="${DREAM_ROOT}/deps/SpaTrackerV2"
SPA_PATCH="${DREAM_ROOT}/patches/spatracker_memory.patch"
if git -C "${SPA_ROOT}" apply --reverse --check "${SPA_PATCH}" 2>/dev/null; then
    echo "SpaTracker memory patch is already present."
else
    git -C "${SPA_ROOT}" apply "${SPA_PATCH}"
fi

echo "Installing the main environment..."
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install "numpy==1.26.4" "scipy==1.15.3"
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.4.1+cu118" "torchvision==0.19.1+cu118" "xformers==0.0.28"
python -m pip install --no-deps --only-binary=:all: \
    "torch-cluster==1.6.3+pt24cu118" \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
python -m pip install -r "${PROJECT_ROOT}/requirements-main.txt"

# Its other dependencies are installed above; avoid PyRoKi's unpinned JAXLS.
python -m pip install --no-deps \
    "pyroki @ git+https://github.com/chungmin99/pyroki.git@388e43e1fc0d0ee382968d3dd72970fd62a0450c"
# Preserve the tested Torch stack despite SAM2's declared newer minimum.
# SAM2 may skip its optional CUDA extension if no matching compiler is available.
python -m pip install --no-deps --no-build-isolation \
    "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
python -m pip install --no-deps \
    -e "${DREAM_ROOT}" -e "${DREAM_ROOT}/deps/co-tracker"

echo "Main environment installed. SAM3D uses its separate sam3d-objects environment."
