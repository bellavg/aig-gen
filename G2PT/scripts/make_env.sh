#!/bin/bash
#SBATCH --job-name=env
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=00:59:00
#SBATCH --output=slurm_logs/env_%j.out

# Create log directories if they don't exist
cd ..

mkdir -p slurm_logs

module purge

module load 2024
module load Anaconda3/2024.06-1

conda env create -f env.yml
source activate g2pt-aig

echo "Environment created and activated"

