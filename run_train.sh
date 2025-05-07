#!/bin/bash
#SBATCH --job-name=graphebm  # Changed job name for GraphAF
#SBATCH --partition=gpu_h100       # Specify the appropriate partition here (adjust if needed)
#SBATCH --gpus=1
#SBATCH --time=12:00:00            # Adjust time limit if needed
#SBATCH --output=./slurm_logs/graphebm_aig_train_%j.out # Changed log file name for GraphAF


# --- Configuration ---
# Choose the model to train
MODEL_NAME="GraphEBM" # Set to GraphAF

# Choose the device ('cuda' or 'cpu')
# The script will try to use cuda if requested and available, otherwise fallback to cpu
REQUESTED_DEVICE="cuda"

# Conda environment name (using the one from your example script)
CONDA_ENV_NAME="g2pt-aig"

# WANDB API Key (optional, keep if you use Weights & Biases)
export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"
# --- End Configuration ---


# Create log directories if they don't exist
mkdir -p slurm_logs
echo "Log directory ensured."

# Load necessary environment modules (adjust based on your cluster)
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1
echo "Modules loaded."

# Activate the conda environment
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
conda_status=$?
if [ $conda_status -ne 0 ]; then
    echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'. Exiting."
    exit 1
fi
echo "Conda environment activated."


# Run the training script using srun
# -u ensures unbuffered output, which is good for logs
echo "Starting training script for model: ${MODEL_NAME} on device: ${REQUESTED_DEVICE}..."
srun python -u train_graphs.py \
    --model ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE}

train_status=$? # Capture the exit status of the python script

if [ $train_status -ne 0 ]; then
    echo "Error: Training script failed with exit code ${train_status}."
    exit $train_status
fi

echo "Training script finished successfully."
