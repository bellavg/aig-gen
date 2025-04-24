#!/bin/bash
#SBATCH --job-name=g2pt_base_topo_cond
#SBATCH --partition=gpu_h100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=../slurm_logs/g2pt_base_topo_cond_%j.out

export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"


cd ..

# Create log directories if they don't exist
mkdir -p slurm_logs
mkdir -p results/aig-base-topo-5 # Ensure output dir exists

module load 2024
module load Anaconda3/2024.06-1

source activate g2pt-aig
# f"{dataset}-{model_name}-{ordering}-{num_augmentations}"
echo "Starting training script..."
srun python -u train.py configs/aig.py configs/networks/base.py\
    --dataset=aig \
    --wandb_log=True \
    --ordering=topo \
    --num_augmentations=5

echo "Training script finished."


