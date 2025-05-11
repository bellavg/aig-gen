#!/bin/bash
#SBATCH --job-name=env_digress
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=01:30:00              # Increased time slightly for robust installation
#SBATCH --output=../slurm_logs/digress_env_%j.out

# --- Configuration ---
ENV_NAME="digress"
PYTHON_VERSION="3.9"
PYTORCH_VERSION="2.2.1"
CUDA_VERSION="11.8"

echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

# --- Environment Creation and Setup ---

# 1. Deactivate any active environment
echo "Deactivating current conda environment if any..."
conda deactivate

# 2. Remove the environment if it already exists
echo "Checking for and removing existing Conda environment '$ENV_NAME'..."
if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Environment '$ENV_NAME' found. Removing..."
    conda env remove -n "$ENV_NAME" -y
    if [ $? -ne 0 ]; then echo "ERROR: Failed to remove '$ENV_NAME'. Exiting."; exit 1; fi
    echo "Environment '$ENV_NAME' removed."
else
    echo "Environment '$ENV_NAME' not found. Proceeding to create it."
fi

# 3. Create the new Conda environment
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
if [ $? -ne 0 ]; then echo "ERROR: Failed to create '$ENV_NAME'. Exiting."; exit 1; fi
echo "Conda environment '$ENV_NAME' created."

# 4. Activate the new environment
echo "Activating Conda environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate '$ENV_NAME'. Exiting."; exit 1; fi
echo "Conda environment '$ENV_NAME' activated."

# 5. Configure channel priorities FIRST
echo "Setting Conda channel priorities..."
conda config --env --add channels conda-forge
conda config --env --add channels pytorch
conda config --env --add channels nvidia
conda config --env --set channel_priority strict # Crucial for graph-tool from conda-forge
echo "Channel priorities set. Current channels:"
conda config --show channels

# OPTIONAL: If libmamba warnings are excessive or lead to failures, try classic solver
# echo "Setting Conda solver to 'classic'..."
# conda config --env --set solver classic
# conda config --show solver

# 6. Install graph-tool (and its specific dependencies like a compatible libgomp)
# Let conda choose the best graph-tool version compatible with Python 3.9 from conda-forge
# Also explicitly ask for conda-forge's libgomp if issues persist, though usually handled by deps.
echo "Installing graph-tool from conda-forge..."
conda install graph-tool -c conda-forge -y
# If the above fails, or you get libgomp errors on import, you could be more specific:
# conda install 'graph-tool' 'libgomp<2' -c conda-forge -y # To ensure conda-forge's libgomp if there's a system conflict
# Or try a specific recent version known to work with Python 3.9, e.g., graph-tool=2.57

if [ $? -ne 0 ]; then echo "ERROR: Failed to install graph-tool. Exiting."; exit 1; fi
echo "graph-tool installed. Verifying import..."
python -c "try: import graph_tool.all as gt; print('graph-tool imported successfully!') except Exception as e: print(f'Error importing graph-tool: {e}'); import sys; sys.exit(1)"
if [ $? -ne 0 ]; then echo "ERROR: graph-tool import failed. Exiting."; exit 1; fi
echo "graph-tool imported successfully."

# 7. Install PyTorch and CUDA toolkit
echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}..."
# The -c pytorch and -c nvidia flags should not be strictly needed here if channels are configured
# and channel_priority is strict, but explicit can sometimes help if conda gets confused.
conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y
if [ $? -ne 0 ]; then echo "ERROR: Failed to install PyTorch. Exiting."; exit 1; fi
echo "PyTorch and CUDA toolkit installed."

# 8. Install other dependencies
echo "Installing other dependencies from conda-forge..."
conda install hydra-core imageio matplotlib networkx numpy omegaconf overrides pandas pyemd pygsp pytorch_lightning scipy setuptools torch_geometric torchmetrics tqdm wandb -c conda-forge -y
if [ $? -ne 0 ]; then echo "ERROR: Failed to install other dependencies. Exiting."; exit 1; fi
echo "Other dependencies installed."

# 9. Install DiGress project
echo "Navigating to DiGress project directory for installation..."
# IMPORTANT: Adjust this path to your DiGress project location
# This assumes your script is in a subdir (e.g., 'scripts') and DiGress is one level up from *that*.
# Or, if slurm_logs is ../slurm_logs, then DiGress is perhaps also a sibling to the script's dir.
# cd ../DiGress # Example: If script is in 'project_root/scripts' and DiGress is 'project_root/DiGress'
# A safer bet if your output log is ../slurm_logs/ is that DiGress is in the same dir as 'slurm_logs' parent
TARGET_DIGRESS_DIR="../DiGress" # Adjust this path
if [ -d "$TARGET_DIGRESS_DIR" ]; then
    cd "$TARGET_DIGRESS_DIR"
    if [ ! -f "setup.py" ]; then
        echo "ERROR: setup.py not found in $(pwd). Cannot install DiGress. Exiting."
        exit 1
    fi
    echo "Installing DiGress project in editable mode from $(pwd)..."
    pip install -e .
    if [ $? -ne 0 ]; then echo "ERROR: Failed to install DiGress project. Exiting."; exit 1; fi
    echo "DiGress project installed."
else
    echo "ERROR: DiGress project directory '$TARGET_DIGRESS_DIR' not found from $(pwd). Cannot install DiGress. Exiting."
    exit 1
fi

echo "--- Environment Verification ---"
echo "PYTHONPATH: $PYTHONPATH"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "WHICH PYTHON: $(which python)"
echo "PYTHON VERSION in this environment:"
python --version
echo "TORCH VERSION and CUDA status:"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.version.cuda else \"N/A\"}')"
echo "GRAPH-TOOL check (final):"
python -c "try: import graph_tool.all as gt; print('graph-tool imported successfully!') except Exception as e: print(f'Error importing graph-tool: {e}')"

echo "Script finished."