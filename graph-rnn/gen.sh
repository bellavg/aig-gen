#!/bin/bash
#SBATCH --job-name=rnn
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=08:10:00              # Initial requested time, adjust as needed
#SBATCH --output=slurm_logs/rnn_%j.out

# Ensure WANDB_API_KEY is set in your environment or you have logged in via `wandb login`
export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83" # Added your W&B API Key

export HYDRA_FULL_ERROR=1

# --- Configuration ---
CONDA_ENV_NAME="aig-rnn" # CHANGE THIS to your Conda environment name
PROJECT_ROOT="./" # IMPORTANT: SET THIS!


echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate "${CONDA_ENV_NAME}"


# Weights and bias key: 725d958326cb39d0ba89d73b557c294f85ecbf83
# --- Training Command ---
echo ""
echo "========================================"
echo "Starting RNN AIG Training"
echo "========================================"
echo "Running command:"
echo "----------------------------------------"

# Execute the training script
# The -u flag is for unbuffered Python output, good for logs
# Execute the training script
# The -u flag is for unbuffered Python output, good for logs

python generate_and_evaluate_aigs.py configs/checkpoints_aig_typed_baseline/checkpoint_step_30000.pth \
    --num_graphs 10 \
    --max_nodes_generate 64 \
    --min_nodes_generate 5 \
    --train_set_pkl  \
    --results_file my_aig_evaluation.txt \
    --output_generated_pkl generated_aigs_sample.pkl

# --- Completion ---

# Deactivate conda environment
conda deactivate
