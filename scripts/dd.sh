#!/bin/bash
#SBATCH --job-name=digress_aig_train
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=12:00:00              # Initial requested time, adjust as needed
#SBATCH --output=slurm_logs/digress_aig_train_%j.out


# --- Configuration ---
CONDA_ENV_NAME="digress_aig_env" # CHANGE THIS to your Conda environment name
PROJECT_ROOT="./DiGress" # IMPORTANT: SET THIS!

# --- End Configuration ---

# --- Setup ---
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "Project Root: ${PROJECT_ROOT}"
echo "Current working directory: $(pwd)"

# Create slurm_logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/slurm_logs"
echo "Log and error directories ensured."

echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment ${CONDA_ENV_NAME}."
    exit 1
fi
echo "Conda environment activated: $(which python)"
# --- End Setup ---

# --- Navigate to Project Directory ---
cd "${PROJECT_ROOT}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to ${PROJECT_ROOT}."
    exit 1
fi
echo "Changed working directory to: $(pwd)"

# --- Training Command ---
echo ""
echo "========================================"
echo "Starting DiGress AIG Training"
echo "========================================"
echo "Running command:"
echo "srun python -u src/main.py experiment=aig dataset=aig general.abs_path_to_project_root=${PROJECT_ROOT}"
echo "----------------------------------------"

# Execute the training script
# The -u flag is for unbuffered Python output, good for logs
srun python -u src/main.py \
    experiment=aig \
    dataset=aig \
    general.abs_path_to_project_root="${PROJECT_ROOT}" \
    # Add any other specific Hydra overrides here if needed, e.g.:
    # general.resume=outputs/YYYY-MM-DD/HH-MM-SS-aig_resume/checkpoints/last-epoch=XXXX.ckpt
    # train.n_epochs=6000 # To override the config file value

# --- Completion ---
echo ""
echo "========================================"
echo "DiGress AIG Training Script Finished"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Deactivate conda environment
conda deactivate
