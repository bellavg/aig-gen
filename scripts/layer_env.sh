#!/bin/bash
#SBATCH --job-name=layerdag_uv_job
#SBATCH --partition=gpu_h100      # Or your cluster's relevant GPU partition
#SBATCH --gpus=1                  # Request 1 GPU if you intend to use it at runtime
#SBATCH --time=02:00:00           # Adjust time as needed
#SBATCH --output=slurm_logs/layerdag_uv_%j.out # Relative to where sbatch is called
#SBATCH --error=slurm_logs/layerdag_uv_%j.err  # Relative to where sbatch is called

# --- Environment Setup ---
echo "Loading base environment module..."
module load 2024 # This module provides Python and potentially CUDA runtime
echo "Modules loaded."
echo "Python from module: $(which python)"
python --version

# --- Path Setup ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Assume the project root is one level above the scripts directory
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# Define the LayerDAG project directory name
LAYERDAG_DIR_NAME="LayerDAG"
# Construct the full absolute path to the LayerDAG project directory
FULL_LAYERDAG_PATH="${PROJECT_ROOT}/${LAYERDAG_DIR_NAME}"
# Define the virtual environment directory path (inside LayerDAG project)
VENV_DIR="${FULL_LAYERDAG_PATH}/.venv_layerdag_uv"

echo "Script directory: ${SCRIPT_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Full LayerDAG project path: ${FULL_LAYERDAG_PATH}"
echo "Virtual environment path: ${VENV_DIR}"

# --- Project and Virtual Environment Setup ---
# Navigate to the LayerDAG project directory
cd "${FULL_LAYERDAG_PATH}" || { echo "ERROR: Failed to cd to ${FULL_LAYERDAG_PATH}"; exit 1; }
echo "Current working directory: $(pwd)"

echo "Setting up Python virtual environment with uv..."
# Use uv directly as an executable. The venv will be named .venv_layerdag_uv
# and created in the current directory (which is FULL_LAYERDAG_PATH).
# We use the python from the loaded module as the base for the venv.
if [ ! -d ".venv_layerdag_uv" ]; then # Check for venv in current directory
    uv venv .venv_layerdag_uv --python $(which python)
    if [ $? -ne 0 ]; then
        echo "ERROR: uv venv command failed. Exiting."
        exit 1
    fi
    echo "uv virtual environment created at $(pwd)/.venv_layerdag_uv"
else
    echo "uv virtual environment already exists at $(pwd)/.venv_layerdag_uv"
fi

echo "Activating Python environment..."
# Activate using path relative to current directory (FULL_LAYERDAG_PATH)
source ".venv_layerdag_uv/bin/activate" || { echo "ERROR: Failed to activate uv environment"; exit 1; }
echo "Python environment activated. Current python from venv: $(which python)"
python --version

# Optional: For runtime linking issues within the venv (less common for uv venvs than conda)
# export LD_LIBRARY_PATH="$(pwd)/.venv_layerdag_uv/lib:$LD_LIBRARY_PATH"

echo "Installing/checking project dependencies with uv..."
# Install project in editable mode and its dependencies from pyproject.toml
# This assumes pyproject.toml is in the current directory (FULL_LAYERDAG_PATH)
uv pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: uv pip install -e . failed. Exiting."
    exit 1
fi
echo "Dependencies installed."

# --- Run LayerDAG ---
# The current working directory is already FULL_LAYERDAG_PATH
echo "Running LayerDAG script..."
CONFIG_FILE="configs/LayerDAG/aig.yaml" # Path relative to FULL_LAYERDAG_PATH [cite: uploaded:LayerDAG/configs/LayerDAG/aig.yaml]

# Use srun to execute the python script within the Slurm allocation
# Paths to train.py and config file are now relative to the CWD (FULL_LAYERDAG_PATH)
srun python -u train.py --config_file "${CONFIG_FILE}" --num_threads 4 --seed 0 [cite: uploaded:LayerDAG/train.py]

echo "--- Environment Verification (Post-run) ---"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") and torch.version.cuda else \"N/A (or CPU build)\"}')"

echo "LayerDAG job finished."