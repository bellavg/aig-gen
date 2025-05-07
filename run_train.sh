#!/bin/bash
#SBATCH --job-name=graphebm_tuned_v3 # Incremented version
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # Adjust time limit as needed
#SBATCH --output=./slurm_logs/graphebm_aig_train_tuned_v3_%j.out


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
srun python -u train_graphs.py \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root "./data/" \
    \
    # --- Hyperparameters Aligned with or Considered from GraphEBM Paper (Liu et al., 2021) ---
    \
    # Learning Rate (Paper Appendix D: 0.0001)
    --lr 0.0001 \
    \
    # Energy Regularizer Alpha (Paper Appendix D: alpha=1)
    # This was 0.01 in your script, changing to 1.0 to match paper.
    --ebm_alpha 1.0 \
    # Alternatives for ebm_alpha if 1.0 is unstable: 0.1, 0.05, 0.01
    \
    # Langevin Dynamics Step Size (lambda/2 in paper Eq.7)
    # Paper Appendix D: lambda/2 in [10, 50]. This range is unusually large for typical EBMs.
    # Your script's 0.1 is a more conventional starting point.
    # The heuristic ld_step_size approx 2 * (ld_noise^2) for ld_noise=0.005 gives 0.00005.
    # Sticking with 0.1 as a tunable value. If using very small step sizes, ebm_ld_step might need to be increased.
    --ebm_ld_step_size 0.1 \
    # Alternatives for ebm_ld_step_size: 0.01, 0.001, (or 1.0, 10.0 if exploring paper's range with caution)
    \
    # Noise level for Langevin Dynamics (sigma in paper Eq.7) (Paper Appendix D: sigma=0.005)
    --ebm_ld_noise 0.005 \
    # Alternatives for ebm_ld_noise: 0.001, 0.01
    \
    # Number of Langevin Dynamics steps (K in paper) (Paper Appendix D: K in [30, 300])
    --ebm_ld_step 100 \
    # Alternatives for ebm_ld_step: 60 (common for EBMs), 200 (slower, potentially better samples)
    \
    # Dequantization coefficient (t in paper Eq.8) (Paper Appendix D: t in [0,1])
    --ebm_c 0.01 \
    # Alternatives for ebm_c: 0.0 (no dequantization), 0.05
    \
    # Langevin Dynamics Gradient Clipping (Paper Appendix D: clip grad magnitude < 0.01)
    # Ensure your train_graphs.py script implements this.
    # This might be a boolean flag like --ebm_clamp_lgd_grad or handled internally.
    # Add the flag if your script expects it, e.g.:
    # --ebm_clamp_lgd_grad \
    \
    # --- General training parameters ---
    # Weight decay (L2 regularization on model parameters) (Paper does not specify, common practice)
    --weight_decay 1e-5 \
    # Alternatives for weight_decay: 0 (no decay), 1e-4
    \
    # Batch size (Paper Appendix D: 128)
    --batch_size 64 \
    # Alternatives for batch_size: 128 (paper's value, if memory allows), 32
    \
    # Maximum number of training epochs (Paper Appendix D: up to 20)
    --max_epochs 100 \
    # Alternatives for max_epochs: 20 (paper's baseline), 50, 200
    \
    # Save interval for model checkpoints (User preference)
    --save_interval 5 \
    \
    # Hidden dimension for EBM networks (d in paper) (Paper Appendix D: d=64)
    # Assuming this is set in the train_graphs.py or EnergyFunc default if not passed.
    # --ebm_hidden 64 \

train_status=$?

if [ $train_status -ne 0 ]; then
    echo "Error: Training script failed with exit code ${train_status}."
    # Optional: print last few lines of the output log for quick diagnosis
    echo "Last 50 lines of output:"
    tail -n 50 ./slurm_logs/graphebm_aig_train_tuned_v3_${SLURM_JOB_ID}.out
    exit $train_status
fi

echo "Training script finished successfully."
