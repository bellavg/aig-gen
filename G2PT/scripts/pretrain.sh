#!/bin/bash
#SBATCH --job-name=g2pt_base
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/g2pt_base_%j.out


cd ..

# Create log directories if they don't exist
mkdir -p slurm_logs


module load 2024
module load Anaconda3/2024.06-1

source activate g2pt-aig


srun python -u train.py configs/datasets/aig.py configs/networks/small.py \
    --dataset=aig \
    --ordering=topo \
    --batch_size=32 \


   
