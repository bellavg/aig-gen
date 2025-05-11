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
echo "Initial Python from module: $(which python)"
python --version

# --- Path Setup ---
# SLURM_SUBMIT_DIR is the directory from which sbatch was invoked.
# If you run 'sbatch layerdag.sh' from your 'scripts' directory:
# SLURM_SUBMIT_DIR will be /path/to/project_root/scripts
# We then get the parent of SLURM_SUBMIT_DIR to find the actual project root.
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set. Are you running this via sbatch?"
    # As a fallback for local testing, you could use SCRIPT_DIR, but this path logic
    # would need to assume the script is always in a 'scripts' subdir of the project root.
    # For Slurm, SLURM_SUBMIT_DIR is the way to go.
    SCRIPT_DIR_FALLBACK=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
    PROJECT_ROOT=$(dirname "$SCRIPT_DIR_FALLBACK")
else
    PROJECT_ROOT=$(dirname "$SLURM_SUBMIT_DIR")
fi

LAYERDAG_DIR_NAME="LayerDAG"
FULL_LAYERDAG_PATH="${PROJECT_ROOT}/${LAYERDAG_DIR_NAME}"
VENV_DIR_NAME=".venv_layerdag_uv" # Name of the venv directory within FULL_LAYERDAG_PATH

echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-"Not set (maybe not a Slurm job?)"}"
echo "Effective Project Root: ${PROJECT_ROOT}"
echo "Full LayerDAG project path: ${FULL_LAYERDAG_PATH}"
echo "Virtual environment will be in: ${FULL_LAYERDAG_PATH}/${VENV_DIR_NAME}"

# --- Project and Virtual Environment Setup ---
# Navigate to the LayerDAG project directory
cd "${FULL_LAYERDAG_PATH}" || { echo "ERROR: Failed to cd to ${FULL_LAYERDAG_PATH}. Check path and permissions."; exit 1; }
echo "Current working directory: $(pwd)"

echo "Setting up Python virtual environment with uv..."
# Use uv directly as an executable. The venv will be named $VENV_DIR_NAME
# and created in the current directory (which is FULL_LAYERDAG_PATH).
if [ ! -d "${VENV_DIR_NAME}" ]; then
    uv venv "${VENV_DIR_NAME}" --python $(which python) # Use python from loaded module
    if [ $? -ne 0 ]; then
        echo "ERROR: uv venv command failed. Exiting."
        exit 1
    fi
    echo "uv virtual environment created at $(pwd)/${VENV_DIR_NAME}"
else
    echo "uv virtual environment already exists at $(pwd)/${VENV_DIR_NAME}"
fi

echo "Activating Python environment..."
# Activate using path relative to current directory (FULL_LAYERDAG_PATH)
source "${VENV_DIR_NAME}/bin/activate" || { echo "ERROR: Failed to activate uv environment"; exit 1; }
echo "Python environment activated. Current python from venv: $(which python)"
python --version

# Optional: For runtime linking issues (usually not needed for Python venvs created by uv)
# export LD_LIBRARY_PATH="$(pwd)/${VENV_DIR_NAME}/lib:$LD_LIBRARY_PATH"

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
echo "Running LayerDAG script..."
CONFIG_FILE="configs/LayerDAG/aig.yaml" # Path relative to FULL_LAYERDAG_PATH

# Use srun to execute the python script within the Slurm allocation
# Paths to train.py and config file are now relative to the CWD (FULL_LAYERDAG_PATH)
srun python -u train.py --config_file "${CONFIG_FILE}" --num_threads 4 --seed 0

echo "--- Environment Verification (Post-run) ---"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") and torch.version.cuda else \"N/A (or CPU build)\"}')"

echo "LayerDAG job finished."