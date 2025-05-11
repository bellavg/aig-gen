#!/bin/bash
#SBATCH --job-name=layerdag_train_h100
#SBATCH --partition=gpu_h100          # Adjust to your H100 partition
#SBATCH --gpus=1
#SBATCH --time=16:00:00               # Adjust training time as needed
#SBATCH --output=slurm_logs/layerdag_train_%j.out # Ensure slurm_logs dir exists


# --- User Configuration ---
CONDA_ENV_NAME="layerdag_h100_env"
PYTHON_VERSION="3.10"
# For H100, target CUDA 12.1. Check PyTorch/DGL websites for latest stable if needed.
PYTORCH_VERSION="2.3.0" # Example, ensure this version has a cu121 build
TORCHVISION_VERSION="0.18.0" # Corresponds to PyTorch 2.3.0
TORCHAUDIO_VERSION="2.3.0" # Corresponds to PyTorch 2.3.0
DGL_TORCH_URL_SUFFIX="torch-2.3" # For DGL wheel URL, matches PyTorch major.minor

export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83" # Replace with your actual key

# --- Path Setup ---
# Assumes this script is in a 'scripts' directory, and 'LayerDAG' is a sibling.
# SLURM_SUBMIT_DIR is the directory from which sbatch was invoked.
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set. This script is intended to be run via sbatch."
    exit 1
fi
# Assuming you run 'sbatch your_script_name.sh' from the 'scripts' directory
PROJECT_ROOT=$(dirname "$SLURM_SUBMIT_DIR") # This will be parent of 'scripts' dir
LAYERDAG_PROJECT_DIR="${PROJECT_ROOT}/LayerDAG" # Path to your LayerDAG project code

# Flag file to indicate if setup has been completed
SETUP_DONE_FLAG="${PROJECT_ROOT}/${CONDA_ENV_NAME}_setup_done.flag"

echo "--- Job Configuration ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)" # Should be 'scripts' directory if sbatched from there
echo "Project Root: ${PROJECT_ROOT}"
echo "LayerDAG Project Dir: ${LAYERDAG_PROJECT_DIR}"
echo "Conda Env Name: ${CONDA_ENV_NAME}"
echo "Setup Done Flag: ${SETUP_DONE_FLAG}"
echo "-------------------------"

# --- Module Loading ---
echo "Loading modules..."
module load 2024                # Or your cluster's base environment module
module load Anaconda3/2024.06-1 # Or your specific Anaconda module
echo "Modules loaded."

# --- Conda Environment Setup ---
# Check if setup has been done previously
if [ ! -f "${SETUP_DONE_FLAG}" ]; then
    echo "Performing first-time setup for Conda environment: ${CONDA_ENV_NAME}"

    # Create Conda environment
    conda create -n "${CONDA_ENV_NAME}" python=${PYTHON_VERSION} -y
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create Conda environment. Exiting."
        exit 1
    fi

    echo "Activating new Conda environment: ${CONDA_ENV_NAME} for installation..."
    source activate "${CONDA_ENV_NAME}" # Use 'source activate' for older conda or 'conda activate'
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate Conda environment for installation. Exiting."
        exit 1
    fi

    echo "Installing PyTorch for CUDA 12.1..."
    pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install PyTorch. Exiting."
        conda deactivate
        exit 1
    fi

    echo "Installing DGL for PyTorch ${PYTORCH_VERSION} and CUDA 12.1..."
    pip install dgl -f https://data.dgl.ai/wheels/${DGL_TORCH_URL_SUFFIX}/cu121/repo.html
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install DGL. Exiting."
        conda deactivate
        exit 1
    fi

    echo "Installing other dependencies (numpy, tqdm, etc.)..."
    pip install numpy==1.26.3 tqdm einops wandb pydantic pandas
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install other dependencies. Exiting."
        conda deactivate
        exit 1
    fi

    echo "Conda environment setup and package installation complete."
    # Create the flag file to indicate setup is done
    touch "${SETUP_DONE_FLAG}"
    conda deactivate
    echo "Deactivated environment after setup."
else
    echo "Conda environment '${CONDA_ENV_NAME}' setup previously done (flag file found)."
fi

# --- Activate Conda Environment for the Job ---
echo "Activating Conda environment: ${CONDA_ENV_NAME} for the job..."
source activate "${CONDA_ENV_NAME}" # Use 'source activate' or 'conda activate'
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment for the job. Exiting."
    exit 1
fi
echo "Conda environment activated. Python version: $(python --version)"
echo "PyTorch version and CUDA status:"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'Torch CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") and torch.version.cuda else \"N/A\"}')"
echo "DGL version:"
python -c "import dgl; print(f'DGL: {dgl.__version__}')"


# --- Set WANDB API Key ---
if [ -z "${WANDB_API_KEY}" ] || [ "${WANDB_API_KEY}" == "YOUR_WANDB_API_KEY_HERE" ]; then
    echo "WARNING: WANDB_API_KEY is not set or is set to the placeholder. WandB logging might fail or be anonymous."
else
    export WANDB_API_KEY="${WANDB_API_KEY}"
    echo "WANDB_API_KEY has been set."
fi

# --- Navigate to Project Directory ---
cd "${LAYERDAG_PROJECT_DIR}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change directory to ${LAYERDAG_PROJECT_DIR}. Exiting."
    exit 1
fi
echo "Current working directory: $(pwd)"

# --- Training Command ---
echo ""
echo "========================================"
echo "Starting LayerDAG Training"
echo "Config: configs/LayerDAG/aig.yaml" # [cite: uploaded:LayerDAG/configs/LayerDAG/aig.yaml]
echo "========================================"

# The -u flag is for unbuffered Python output, good for logs
# train.py is expected to be in the LAYERDAG_PROJECT_DIR
srun python -u train.py \
    --config_file "configs/LayerDAG/aig.yaml" \
    --num_threads 4 \
    --seed 0

# --- Completion ---
echo ""
echo "========================================"
echo "LayerDAG Training Script Finished"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Deactivate conda environment
conda deactivate
echo "Conda environment deactivated."
