#!/bin/bash
#SBATCH --job-name=graphebm_stable_v1 # New name for stability test
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=./slurm_logs/graphebm_aig_train_stable_v1_%j.out


# --- Configuration ---
MODEL_NAME="GraphEBM"
REQUESTED_DEVICE="cuda"
CONDA_ENV_NAME="g2pt-aig"

# --- End Configuration ---


mkdir -p slurm_logs
echo "Log directory ensured: ./slurm_logs"

echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
conda_status=$?
if [ $conda_status -ne 0 ]; then
    echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'. Exiting."
    exit 1
fi
echo "Conda environment activated."

echo "Starting training script for model: ${MODEL_NAME} on device: ${REQUESTED_DEVICE} with stability adjustments..."

srun python -u train_graphs.py \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root "./data/" \
    `# --- Stability Adjustments ---` \
    `# Reduced Learning Rate` \
    --lr 5e-5 \
    `# Alternatives for lr: 1e-5 (if still unstable), 0.0001 (if 5e-5 is too slow AND stable)` \
    \
    `# Reduced Langevin Dynamics Step Size` \
    --ebm_ld_step_size 0.01 \
    `# Alternatives for ebm_ld_step_size: 0.001 (if 0.01 still unstable), 0.1 (original value if stable)` \
    \
    `# Gradient Clipping for Model Parameters (ensure train_graphs.py uses this)` \
    --grad_clip_value 1.0 \
    `# Alternatives for grad_clip_value: 5.0, or remove if not needed/causing issues` \
    \
    `# --- Parameters Aligned with GraphEBM Paper ---` \
    --ebm_alpha 1.0 \
    --ebm_ld_noise 0.005 \
    --ebm_ld_step 100 \
    `# Alternatives for ebm_ld_step: 150 or 200 if ld_step_size is very small` \
    --ebm_c 0.01 \
    --ebm_clamp_lgd_grad \
    \
    `# --- General training parameters ---` \
    --weight_decay 1e-5 \
    --batch_size 64 \
    `# Paper uses 128, can try if memory allows and training is stable` \
    --max_epochs 100 \
    `# Paper trains for "up to 20 epochs", can reduce if convergence is fast & stable` \
    --save_interval 5 \
    \
    `# --- EBM Model Structure parameters (ensure these match your EnergyFunc defaults or are desired) ---` \
    `# These are passed to GraphEBM constructor in train_graphs.py` \
    `# --ebm_hidden 64 ` \
    `# --ebm_depth 2 ` \
    `# --ebm_swish_act ` \
    `# --ebm_add_self ` \
    `# --ebm_dropout 0.0 ` \
    `# --ebm_n_power_iterations 1`

train_status=$?

if [ $train_status -ne 0 ]; then
    echo "Error: Training script failed with exit code ${train_status}."
    echo "Last 50 lines of output:"
    tail -n 50 ./slurm_logs/graphebm_aig_train_stable_v1_${SLURM_JOB_ID}.out
    exit $train_status
fi

echo "Training script finished successfully."
