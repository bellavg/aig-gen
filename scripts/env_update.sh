#!/bin/bash
#SBATCH --job-name=env_digress
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=../slurm_logs/digress_env_%j.out

# --- Configuration ---
ENV_NAME="digress"
PYTHON_VERSION="3.9"
PYTORCH_VERSION="2.2.1"
CUDA_VERSION="11.8" # Ensure this CUDA version is compatible with your node's H100 driver

echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

source activate digress

echo "Installing other dependencies into 'digress' environment..."
echo "Installing other dependencies into 'digress' environment (attempt 2)..."
conda install \
    hydra-core \
    imageio \
    matplotlib \
    networkx \
    numpy \
    omegaconf \
    overrides \
    pandas \
    pyemd \
    pygsp \
    pytorch_lightning \
    scipy \
    setuptools \
    pytorch-geometric \
    torchmetrics \
    tqdm \
    wandb \
    -c conda-forge -y

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install one or more of the 'other dependencies' with corrected names. Please check the output above."
else
    echo "Successfully installed 'other dependencies' (or updated existing ones)."
fi

# 5. Configure channel priorities
# Order of addition matters: conda-forge -> pytorch -> nvidia will result in
# nvidia (top), pytorch, conda-forge in the list.
# With channel_priority strict, this means for a package:
# 1. Try nvidia. 2. Try pytorch. 3. Try conda-forge.
# For graph-tool (primarily in conda-forge), this is fine.
# For PyTorch, it ensures pytorch/nvidia channels are preferred.
#echo "Setting Conda channel priorities..."
#conda config --env --add channels conda-forge
#conda config --env --add channels pytorch
#conda config --env --add channels nvidia
#conda config --env --set channel_priority strict
#echo "Channel priorities set. Current channels:"
#conda config --show channels # Shows effective list; nvidia should be at the top
#
## 6. Install graph-tool
#echo "Installing graph-tool from conda-forge..."
#conda install graph-tool -c conda-forge -y
#if [ $? -ne 0 ]; then echo "ERROR: Failed to install graph-tool. Exiting."; exit 1; fi
#echo "graph-tool Conda installation successful. Verifying Python import..."
#
## Simplified graph-tool import check
#python -c "import graph_tool.all"
#if [ $? -ne 0 ]; then
#    echo "ERROR: Failed to import graph_tool.all in Python. Check traceback above. Exiting."
#    exit 1
#fi
#echo "graph-tool imported successfully in Python."
#
## 7. Install PyTorch and CUDA toolkit
#echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}..."
#conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y
#if [ $? -ne 0 ]; then echo "ERROR: Failed to install PyTorch. Exiting."; exit 1; fi
#echo "PyTorch and CUDA toolkit installed."
#
## 8. Install other dependencies
#echo "Installing other dependencies from conda-forge..."
#conda install hydra-core imageio matplotlib networkx numpy omegaconf overrides pandas pyemd pygsp pytorch_lightning scipy setuptools torch_geometric torchmetrics tqdm wandb -c conda-forge -y
#if [ $? -ne 0 ]; then echo "ERROR: Failed to install other dependencies. Exiting."; exit 1; fi
#echo "Other dependencies installed."
#
## 9. Install DiGress project
#echo "Navigating to DiGress project directory for installation..."
## IMPORTANT: Adjust this path to your DiGress project location
#TARGET_DIGRESS_DIR="../DiGress" # Adjust this path relative to where the SBATCH script is run
#if [ -d "$TARGET_DIGRESS_DIR" ]; then
#    cd "$TARGET_DIGRESS_DIR"
#    if [ ! -f "setup.py" ]; then
#        echo "ERROR: setup.py not found in $(pwd). Current PWD: $(pwd). Cannot install DiGress. Exiting."
#        exit 1
#    fi
#    echo "Installing DiGress project in editable mode from $(pwd)..."
#    pip install -e .
#    if [ $? -ne 0 ]; then echo "ERROR: Failed to install DiGress project. Exiting."; exit 1; fi
#    echo "DiGress project installed."
#else
#    echo "ERROR: DiGress project directory '$TARGET_DIGRESS_DIR' not found from initial PWD. Current PWD: $(pwd). Cannot install DiGress. Exiting."
#    exit 1
#fi
#
#echo "--- Environment Verification ---"
#echo "PYTHONPATH: $PYTHONPATH"
#echo "CONDA_PREFIX: $CONDA_PREFIX"
#echo "WHICH PYTHON: $(which python)"
#echo "PYTHON VERSION in this environment:"
#python --version
#echo "TORCH VERSION and CUDA status:"
#python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.version.cuda else \"N/A\"}')"
#echo "GRAPH-TOOL check (final):"
#python -c "try: import graph_tool.all as gt; print('graph-tool imported successfully!') except Exception as e: print(f'Error importing graph-tool: {e}')"
#
#echo "Script finished."