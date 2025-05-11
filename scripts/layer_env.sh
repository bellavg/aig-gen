#!/bin/bash
#SBATCH --job-name=layerdag_uv_job
#SBATCH --partition=gpu_h100  # Or your cluster's relevant GPU partition
#SBATCH --gpus=1              # Request 1 GPU if you intend to use it at runtime
#SBATCH --time=02:00:00       # Adjust time as needed
#SBATCH --output=slurm_logs/layerdag_uv_%j.out
#SBATCH --error=slurm_logs/layerdag_uv_%j.err

# --- Environment Setup ---
echo "Loading base environment module..."
module load 2024 # Or your specific Anaconda3/XXXX module that might include Python/CUDA
# module load python/3.10.x # This might now be redundant if 'module load 2024' provides Python
# module load cuda/12.1     # We are now assuming 'module load 2024' handles this.
                           # Commented out based on your observation.
echo "Modules loaded."

# --- Project and uv Setup ---
# (Ensure uv is accessible in your PATH as discussed before)
PROJECT_DIR="./LayerDAG" # IMPORTANT: Change this path
ENV_DIR="${PROJECT_DIR}/.venv_layerdag_uv"

cd "${PROJECT_DIR}" || { echo "ERROR: Failed to cd to ${PROJECT_DIR}"; exit 1; }
echo "Current directory: $(pwd)"

echo "Setting up Python virtual environment with uv..."
# Ensure the python used by uv venv is the one from your loaded module environment
# If 'module load 2024' makes a suitable python available, uv should pick it up.
# You can be explicit: uv venv "${ENV_DIR}" --python $(which python)
if [ ! -d "${ENV_DIR}" ]; then
    python -m uv venv "${ENV_DIR}"
    echo "uv environment created."
else
    echo "uv environment already exists."
fi

echo "Activating Python environment..."
source "${ENV_DIR}/bin/activate" || { echo "ERROR: Failed to activate uv environment"; exit 1; }
echo "Python environment activated. Current python: $(which python)"
python --version
# Inside your Slurm script, after 'source "${ENV_DIR}/bin/activate"'
# and before running your python command:
export LD_LIBRARY_PATH="${ENV_DIR}/lib:$LD_LIBRARY_PATH"
echo "Installing/checking project dependencies with uv..."
# Using your CUDA-agnostic pyproject.toml
python -m uv pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: uv failed to install dependencies. Exiting."
    exit 1
fi
echo "Dependencies installed."

cd ..
# --- Run LayerDAG ---
echo "Running LayerDAG script..."
CONFIG_FILE="LayerDAG/configs/LayerDAG/aig.yaml" #

srun python -u LayerDAG/train.py --config_file "${CONFIG_FILE}" --num_threads 4 --seed 0

echo "--- Environment Verification (Post-run) ---"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") and torch.version.cuda else \"N/A (or CPU build)\"}')"

echo "LayerDAG job finished."