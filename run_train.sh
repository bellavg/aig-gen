#!/bin/bash
#SBATCH --job-name=graphebm_tuned_v4 # Incremented version
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # Adjust time limit as needed
#SBATCH --output=./slurm_logs/graphebm_aig_train_tuned_v4_%j.out


# --- Configuration ---
MODEL_NAME="GraphEBM"
REQUESTED_DEVICE="cuda"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name

# --- End Configuration ---


# Create log directories if they don't exist
mkdir -p slurm_logs
echo "Log directory ensured: ./slurm_logs"

# Load necessary environment modules (adjust based on your cluster's setup)
echo "Loading modules..."
module load 2024 # Consider specifying more precise modules like cuda/version if known
module load Anaconda3/2024.06-1 # Or your specific Anaconda/Miniconda module
echo "Modules loaded."

# Activate the conda environment
# Note: 'conda activate' is preferred in modern conda, but 'source activate' might be needed on some HPCs.
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
conda_status=$?
if [ $conda_status -ne 0 ]; then
    echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'. Exiting."
    exit 1
fi
echo "Conda environment activated."


# Run the training script using srun
echo "Starting training script for model: ${MODEL_NAME} on device: ${REQUESTED_DEVICE} with tuned hyperparameters..."
# Ensure train_graphs.py correctly parses these arguments and passes them to the GraphEBM model/training functions.
# IMPORTANT: All arguments below MUST be on lines that are NOT fully commented out,
# and line continuations `\` must be the last character on the line.

srun python -u train_graphs.py \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root "./data/" \
    `# Learning Rate (Paper Appendix D: 0.0001)` \
    --lr 0.0001 \
    `# Energy Regularizer Alpha (Paper Appendix D: alpha=1)` \
    --ebm_alpha 1.0 \
    `# Langevin Dynamics Step Size (lambda/2 in paper Eq.7)` \
    `# Paper Appendix D: lambda/2 in [10, 50]. This range is unusually large for typical EBMs.` \
    `# Using 0.1 as a more conventional starting point.` \
    --ebm_ld_step_size 0.1 \
    `# Noise level for Langevin Dynamics (sigma in paper Eq.7) (Paper Appendix D: sigma=0.005)` \
    --ebm_ld_noise 0.005 \
    `# Number of Langevin Dynamics steps (K in paper) (Paper Appendix D: K in [30, 300])` \
    --ebm_ld_step 100 \
    `# Dequantization coefficient (t in paper Eq.8) (Paper Appendix D: t in [0,1])` \
    --ebm_c 0.01 \
    `# Langevin Dynamics Gradient Clipping (Paper Appendix D: clip grad magnitude < 0.01)` \
    `# This flag should be parsed by train_graphs.py and passed as 'clamp_lgd_grad' (boolean) to GraphEBM.train_rand_gen` \
    --ebm_clamp_lgd_grad \
    `# Weight decay (L2 regularization on model parameters) (Paper does not specify, common practice)` \
    --weight_decay 1e-5 \
    `# Batch size (Paper Appendix D: 128)` \
    --batch_size 64 \
    `# Maximum number of training epochs (Paper Appendix D: up to 20)` \
    --max_epochs 100 \
    `# Save interval for model checkpoints (User preference)` \
    --save_interval 5
    # Note: If you have other arguments like --ebm_hidden, ensure they are also passed correctly.

train_status=$?

if [ $train_status -ne 0 ]; then
    echo "Error: Training script failed with exit code ${train_status}."
    # Optional: print last few lines of the output log for quick diagnosis
    echo "Last 50 lines of output:"
    tail -n 50 ./slurm_logs/graphebm_aig_train_tuned_v4_${SLURM_JOB_ID}.out
    exit $train_status
fi

echo "Training script finished successfully."
