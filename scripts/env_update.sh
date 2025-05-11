#!/bin/bash
#SBATCH --job-name=env
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=00:59:00              # Initial requested time, adjust as needed
#SBATCH --output=../slurm_logs/digress_env_%j.out

cd ..
cd DiGress

# --- Configuration ---
CONDA_ENV_NAME="digress" # CHANGE THIS to your Conda environment name



echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."


#conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9

source activate digress

conda remove --all --keep-env

# 2. IMPORTANT: Configure channel priorities for this environment
# This tells conda to prefer conda-forge for general packages, then pytorch.
echo "Setting Conda channel priorities..."
conda config --env --add channels conda-forge
conda config --env --add channels pytorch
conda config --env --add channels nvidia # For CUDA parts needed by PyTorch from conda
conda config --env --set channel_priority strict

# 3. Install graph-tool from conda-forge
# (Using a specific version like 2.45 if you are sure, otherwise let conda pick a compatible one)
echo "Installing graph-tool..."
conda install graph-tool=2.45 -c conda-forge -y
# Or for a generally compatible version: conda install graph-tool -c conda-forge -y

# 4. Install PyTorch, torchvision, torchaudio, AND the CUDA toolkit using Conda
# This command is for PyTorch 2.2.1 and CUDA 11.8.
# **** ADJUST THIS COMMAND BASED ON PYTORCH.ORG FOR YOUR NEEDS ****
echo "Installing PyTorch and CUDA toolkit..."
conda install pytorch torchvision torchaudio pytorch-cuda -y
# The -c pytorch and -c nvidia flags should not be needed here if channels are configured as above.
# Conda will use the channel priority.

# 5. Install other dependencies from your requirements.txt
# Preferably using Conda from conda-forge where possible.
echo "Installing other dependencies from requirements.txt..."
conda install hydra-core imageio matplotlib networkx numpy omegaconf overrides pandas pyemd pygsp pytorch_lightning scipy setuptools torch_geometric torchmetrics tqdm wandb -c conda-forge -y

# 6. If there are any packages from requirements.txt not found on conda-forge or
#    if you need very specific versions available only on pip:
#    pip install <package_name>
#    For now, let's assume most are covered by the conda install above.

# 7. Install your DiGress project in editable mode
# Navigate to your DiGress project root directory first if you're not already there.
# Assuming your DiGress project has a setup.py file.
echo "Installing DiGress project in editable mode..."
cd DiGress # Change this to the actual path of your DiGress project
pip install -e .

echo "Conda environment 'digress' has been recreated and packages installed."
echo "PYTHON VERSION in this environment:"
python --version
echo "TORCH VERSION and CUDA status:"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
echo "GRAPH-TOOL check (this might fail if libgomp is still an issue, but we hope not):"
python -c "try: import graph_tool.all as gt; print('graph-tool imported successfully!') except Exception as e: print(f'Error importing graph-tool: {e}')"



