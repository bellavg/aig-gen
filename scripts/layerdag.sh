#!/bin/bash
#SBATCH --job-name=layerdag_train_h100
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/layerdag_train_%j.out


# --- User Configuration ---
CONDA_ENV_NAME="layerdag_h100_env"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.3.0"
TORCHVISION_VERSION="0.18.0"
TORCHAUDIO_VERSION="2.3.0"
DGL_TORCH_URL_SUFFIX="torch-2.3" # Corresponds to PyTorch 2.3.x for DGL
# Replace "YOUR_WANDB_API_KEY_HERE" with your actual key if not set globally.
export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"

# --- Path Setup ---
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set. This script is intended to be run via sbatch."
    exit 1
fi
# Assuming this script is in a 'scripts' or similar subdirectory of the project root
PROJECT_ROOT=$(dirname "$SLURM_SUBMIT_DIR") # Or adjust if script is at project root
LAYERDAG_PROJECT_DIR="${PROJECT_ROOT}/LayerDAG" # Assuming LayerDAG is a subdir of PROJECT_ROOT
SETUP_DONE_FLAG="${PROJECT_ROOT}/${CONDA_ENV_NAME}_setup_done.flag"

echo "--- Job Configuration ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Project Root: ${PROJECT_ROOT}"
echo "LayerDAG Project Dir: ${LAYERDAG_PROJECT_DIR}"
echo "Conda Env Name: ${CONDA_ENV_NAME}"
echo "Setup Done Flag Path: ${SETUP_DONE_FLAG}"
echo "-------------------------"

# --- Module Loading ---
echo "Loading modules..."
module purge # Start with a clean environment
module load 2024 # Or your cluster's default environment module
module load Anaconda3/2024.06-1 # Or the specific Anaconda version available
# ** IMPORTANT: Commented out system CUDA to prioritize Conda's CUDA. **
# ** If issues persist, you might need to coordinate this with your cluster admin **
# ** or find the exact system CUDA module that is compatible and ONLY load that if Conda's isn't sufficient. **
# module load cuda/12.1 # Or the specific CUDA version on your cluster that matches pytorch-cuda=12.1

echo "Modules loaded."
echo "Initial LD_LIBRARY_PATH (after module loads): ${LD_LIBRARY_PATH}"

# --- Conda Environment Setup ---
if [ ! -f "${SETUP_DONE_FLAG}" ]; then
    echo "Performing first-time setup for Conda environment: ${CONDA_ENV_NAME}"
    echo "IMPORTANT: If this setup fails, delete the flag file '${SETUP_DONE_FLAG}' and the Conda environment '${CONDA_ENV_NAME}' before retrying."

    # Check if environment already exists from a partial previous run
    if conda env list | grep -q "${CONDA_ENV_NAME}"; then
        echo "Warning: Conda environment '${CONDA_ENV_NAME}' already exists. Removing it for a clean setup."
        conda env remove -n "${CONDA_ENV_NAME}" -y
    fi

    conda create -n "${CONDA_ENV_NAME}" python=${PYTHON_VERSION} -y
    if [ $? -ne 0 ]; then echo "ERROR: Failed to create Conda environment. Exiting."; exit 1; fi

    # Activate the environment for installation
    # Using 'conda activate' is preferred over 'source activate' for newer Conda versions
    eval "$(conda shell.bash hook)" # Ensures 'conda activate' is available
    conda activate "${CONDA_ENV_NAME}"
    if [ $? -ne 0 ]; then echo "ERROR: Failed to activate Conda environment for installation. Exiting."; exit 1; fi
    echo "Conda environment activated for setup. CONDA_PREFIX: ${CONDA_PREFIX}"

    echo "Installing PyTorch CUDA toolkit (pytorch-cuda=12.1) from Conda..."
    # This package provides the CUDA runtime libraries like libnvrtc.so.12
    conda install pytorch-cuda=12.1 -c pytorch -c nvidia -y
    if [ $? -ne 0 ]; then echo "ERROR: Failed to install pytorch-cuda=12.1. Exiting."; conda deactivate; exit 1; fi

    echo "Installing PyTorch ${PYTORCH_VERSION} (for CUDA 12.1) via pip..."
    # Using pip for PyTorch to get the exact version compiled for cu121
    pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then echo "ERROR: Failed to install PyTorch. Exiting."; conda deactivate; exit 1; fi

    echo "Installing DGL (for PyTorch ${DGL_TORCH_URL_SUFFIX} and CUDA 12.1) via pip..."
    pip install dgl -f https://data.dgl.ai/wheels/${DGL_TORCH_URL_SUFFIX}/cu121/repo.html
    if [ $? -ne 0 ]; then echo "ERROR: Failed to install DGL. Exiting."; conda deactivate; exit 1; fi

    echo "Installing other dependencies..."
    pip install numpy==1.26.3 tqdm einops wandb pydantic pandas
    if [ $? -ne 0 ]; then echo "ERROR: Failed to install other dependencies. Exiting."; conda deactivate; exit 1; fi

    echo "Conda environment setup and package installation complete."
    touch "${SETUP_DONE_FLAG}"
    conda deactivate
    echo "Deactivated environment after setup."
else
    echo "Conda environment '${CONDA_ENV_NAME}' setup previously done. Skipping setup."
fi

# --- Activate Conda Environment for the Job ---
echo "Activating Conda environment: ${CONDA_ENV_NAME} for the job..."
eval "$(conda shell.bash hook)" # Ensure 'conda activate' is available
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate Conda environment for the job. Exiting."; exit 1; fi
echo "Conda environment activated. CONDA_PREFIX: ${CONDA_PREFIX}"

# --- ** CRITICAL: Set LD_LIBRARY_PATH for Conda's CUDA libraries ** ---
# The pytorch-cuda package installs libraries (like libnvrtc.so.12) into $CONDA_PREFIX/lib.
# This ensures that these libraries are found by PyTorch and DGL.
if [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "Prepended Conda lib directory to LD_LIBRARY_PATH."
else
    echo "WARNING: ${CONDA_PREFIX}/lib does not exist. CUDA libraries might not be found."
fi
# You can also be more specific if you know the exact subfolder, e.g., for NVIDIA libraries:
# NVIDIA_LIB_DIR=$(find "${CONDA_PREFIX}/lib" -type d -name 'nvidia*' -print -quit)
# if [ -n "$NVIDIA_LIB_DIR" ]; then
# export LD_LIBRARY_PATH="${NVIDIA_LIB_DIR}:${LD_LIBRARY_PATH}"
# fi
# Or directly find the directory containing libnvrtc.so.12
CONDA_NVRTC_LIB_DIR=$(find "${CONDA_PREFIX}/lib" -name "libnvrtc.so.12" -printf "%h" -quit 2>/dev/null)
if [ -n "$CONDA_NVRTC_LIB_DIR" ] && [[ ":$LD_LIBRARY_PATH:" != *":${CONDA_NVRTC_LIB_DIR}:"* ]]; then
    export LD_LIBRARY_PATH="${CONDA_NVRTC_LIB_DIR}:${LD_LIBRARY_PATH}"
    echo "Specifically prepended directory of libnvrtc.so.12 (${CONDA_NVRTC_LIB_DIR}) to LD_LIBRARY_PATH."
elif [ -z "$CONDA_NVRTC_LIB_DIR" ]; then
     echo "WARNING: libnvrtc.so.12 not found in ${CONDA_PREFIX}/lib. This is likely the cause of the error."
fi

echo "LD_LIBRARY_PATH after conda activation and manual update: ${LD_LIBRARY_PATH}"
echo "Which nvcc: $(which nvcc || echo 'nvcc not in PATH')"
echo "Which python: $(which python)"
echo "Python version: $(python --version)"

# --- Verify Library Paths and CUDA Availability ---
python -c "import os; print(f'PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"Not set\")}')"
python -c "import sys; print(f'sys.path: {sys.path}')"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'Torch CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") and torch.version.cuda else \"N/A\"}'); print(f'Torch lib path: {torch.__file__}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
python -c "import dgl; print(f'DGL: {dgl.__version__}'); print(f'DGL lib path: {dgl.__file__}')"
# Attempt to import the problematic module directly to see if LD_LIBRARY_PATH helps
python -c "import dgl.sparse; print('Successfully imported dgl.sparse')"


# --- Navigate to Project Directory ---
cd "${LAYERDAG_PROJECT_DIR}"
if [ $? -ne 0 ]; then echo "ERROR: Failed to cd to ${LAYERDAG_PROJECT_DIR}. Exiting."; exit 1; fi
echo "Current working directory: $(pwd)"

# --- Training Command ---
echo "Starting LayerDAG Training (Config: configs/LayerDAG/aig.yaml)"
srun python -u train.py \
    --config_file "configs/LayerDAG/aig.yaml" \
    --num_threads 4 \
    --seed 0

if [ $? -ne 0 ]; then
    echo "ERROR: Training script exited with an error."
    # Add any cleanup or error reporting here
fi

# --- Completion ---
echo "LayerDAG Training Script Finished (Job ID: $SLURM_JOB_ID)"
conda deactivate
echo "Conda environment deactivated."
