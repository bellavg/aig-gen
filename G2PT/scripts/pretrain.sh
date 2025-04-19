#!/bin/bash
#SBATCH --job-name=g2pt_base
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=../slurm_logs/g2pt_large_%j.out

export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"


cd ..

# Create log directories if they don't exist
mkdir -p slurm_logs
mkdir -p results/aig-small-topo # Ensure output dir exists

module load 2024
module load Anaconda3/2024.06-1

source activate g2pt-aig

echo "Starting training script..."
srun python -u train.py configs/datasets/aig.py configs/networks/large.py \
    --dataset=aig \
    --ordering=topo \
    --batch_size=32 \
    --out_dir=results/aig-small-topo \
    --wandb_log=True

echo "Training script finished."


