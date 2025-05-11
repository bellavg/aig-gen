#!/bin/bash
#SBATCH --job-name=env
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=00:59:00              # Initial requested time, adjust as needed
#SBATCH --output=../slurm_logs/digress_env_%j.out

cd ..
cd Digress

# --- Configuration ---
CONDA_ENV_NAME="digress" # CHANGE THIS to your Conda environment name



echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."


conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9

conda activate digress

conda install -c conda-forge graph-tool=2.45

conda install -c "nvidia/label/cuda-11.8.0" cuda

pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip install -e .





