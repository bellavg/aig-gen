#!/bin/bash
#SBATCH --job-name=g2pt_base_deg
#SBATCH --partition=gpu_h100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=00:59:00
#SBATCH --output=./slurm_logs/g2pt_base_real_%j.out

export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"



# Create log directories if they don't exist
mkdir -p slurm_logs


module load 2024
module load Anaconda3/2024.06-1

source activate g2pt-aig

# f"{dataset}-{model_name}-{ordering}-{num_augmentations}"
echo "Starting training script..."
srun python -u G2PT/train.py G2PT/configs/aig.py G2PT/configs/base.py G2PT/configs/train_aig.py G2PT/configs/sample_aig.py



echo "Training script finished."


