#!/bin/bash
#SBATCH --job-name=layerdag_train
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/layerdag_train_%j.out


echo "Loading base environment module..."
module load 2022 # This module provides Python and potentially CUDA runtime
module load CUDA/11.6.0

source activate LayerDAG
# Clear pip's cache
export CUDA_LAUNCH_BLOCKING=1

# --- Training Command ---
echo "Starting LayerDAG Training (Config: configs/LayerDAG/aig.yaml)"
srun python -u train.py \
    --config_file "configs/LayerDAG/aig.yaml" \
    --seed 0

if [ $? -ne 0 ]; then
    echo "ERROR: Training script exited with an error."
    # Add any cleanup or error reporting here
fi

# --- Completion ---
echo "LayerDAG Training Script Finished (Job ID: $SLURM_JOB_ID)"
conda deactivate
echo "Conda environment deactivated."
