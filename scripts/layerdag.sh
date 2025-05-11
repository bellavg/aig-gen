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
DGL_TORCH_URL_SUFFIX="torch-2.3"
# Replace "YOUR_WANDB_API_KEY_HERE" with your actual key if not set globally.
export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"

# --- Path Setup ---
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set. This script is intended to be run via sbatch."
    exit 1
fi
PROJECT_ROOT=$(dirname "$SLURM_SUBMIT_DIR")
LAYERDAG_PROJECT_DIR="${PROJECT_ROOT}/LayerDAG"
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
module load 2024
module load Anaconda3/2024.06-1
# ** VERIFY AND CHANGE 'cuda/12.1' TO THE EXACT MODULE NAME ON YOUR CLUSTER **
module load cuda/12.1
echo "Modules loaded."
echo "LD_LIBRARY_PATH after module loads: ${LD_LIBRARY_PATH}"

# --- Conda Environment Setup ---
if [ ! -f "${SETUP_DONE_FLAG}" ]; then
    echo "Performing first-time setup for Conda environment: ${CONDA_ENV_NAME}"
    echo "IMPORTANT: If this setup fails, delete the flag file '${SETUP_DONE_FLAG}' before retrying."

    conda create -n "${CONDA_ENV_NAME}" python=${PYTHON_VERSION} -y
    if [ $? -ne 0 ]; then echo "ERROR: Failed to create Conda environment. Exiting."; exit 1; fi

    source activate "${CONDA_ENV_NAME}"
    if [ $? -ne 0 ]; then echo "ERROR: Failed to activate Conda environment for installation. Exiting."; exit 1; fi

    echo "Installing PyTorch CUDA toolkit (pytorch-cuda=12.1) from Conda..."
    conda install pytorch-cuda=12.1 -c pytorch -c nvidia -y
    if [ $? -ne 0 ]; then echo "ERROR: Failed to install pytorch-cuda=12.1. Exiting."; conda deactivate; exit 1; fi

    echo "Installing PyTorch ${PYTORCH_VERSION} (for CUDA 12.1) via pip..."
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
    echo "Conda environment '${CONDA_ENV_NAME}' setup previously done."
fi

# --- Activate Conda Environment for the Job ---
echo "Activating Conda environment: ${CONDA_ENV_NAME} for the job..."
source activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate Conda environment for the job. Exiting."; exit 1; fi
echo "LD_LIBRARY_PATH after conda activation: ${LD_LIBRARY_PATH}"
echo "Python version: $(python --version)"
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}'); print(f'Torch CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") and torch.version.cuda else \"N/A\"}')"
python -c "import dgl; print(f'DGL: {dgl.__version__}')"

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

# --- Completion ---
echo "LayerDAG Training Script Finished (Job ID: $SLURM_JOB_ID)"
conda deactivate
echo "Conda environment deactivated."
