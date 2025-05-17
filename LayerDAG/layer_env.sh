#!/bin/bash
#SBATCH --job-name=layerdag_uv_job
#SBATCH --partition=gpu_a100      # Or your cluster's relevant GPU partition
#SBATCH --gpus=1                  # Request 1 GPU if you intend to use it at runtime
#SBATCH --time=02:00:00           # Adjust time as needed
#SBATCH --output=slurm_logs/layerdag_%j.out # Relative to where sbatch is called
#SBATCH --error=slurm_logs/layerdag_%j.err  # Relative to where sbatch is called

# --- Environment Setup ---
echo "Loading base environment module..."
module load 2022 # This module provides Python and potentially CUDA runtime
module load CUDA/11.6.0
conda create -n LayerDAG python=3.10 -y
conda activate LayerDAG
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c conda-forge cudatoolkit=11.6
conda clean --all -y
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install tqdm einops wandb pydantic pandas
pip install numpy==1.26.3

echo "Done"