#!/bin/bash
#SBATCH --job-name=env
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=00:59:00              # Initial requested time, adjust as needed
#SBATCH --output=../slurm_logs/digress_aig_train_%j.out

cd ..

# --- Configuration ---
CONDA_ENV_NAME="g2pt-aig" # CHANGE THIS to your Conda environment name



echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate "${CONDA_ENV_NAME}"

conda install conda-forge::graph-tool

pip install -r DiGress/requirements.txt

# Deactivate conda environment
conda deactivate
