#!/bin/bash
#SBATCH --job-name=g2pt_base
#SBATCH --partition=gpu_h100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=../slurm_logs/g2pt_base_topo_3_%j.out

export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"


cd ..

# Create log directories if they don't exist
mkdir -p slurm_logs
mkdir -p results/aig-base-topo-3 # Ensure output dir exists

module load 2024
module load Anaconda3/2024.06-1

source activate g2pt-aig

echo "Starting training script..."
srun python -u train.py configs/datasets/aig.py configs/networks/base.py \
    --dataset=aig \
    --ordering=topo \
    --wandb_log=True \
    --num_augmentations=3


echo "Training script finished."


