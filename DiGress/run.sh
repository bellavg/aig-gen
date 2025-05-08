#!/bin/bash
#SBATCH --job-name=digress_aig_train # Job name for DiGress AIG training
#SBATCH --partition=gpu_h100       # Specify your GPU partition
#SBATCH --gpus=1                   # Number of GPUs requested (matches config)
#SBATCH --time=12:00:00            # Adjust time based on expected training duration
#SBATCH --output=./slurm_logs/digress_aig_train_%j.out # SLURM output log file



# --- Configuration ---
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name (update if different)
# Path to the DiGress project directory (where src/, configs/, etc. are located)
# Assumes you submit the job from this directory, otherwise provide an absolute path
PROJECT_DIR=$(pwd)
# Path to the main script within the project directory
MAIN_SCRIPT="src/main.py"
# Name of the experiment config file (relative to configs/experiment/)
EXPERIMENT_CONFIG="aig_full" # Corresponds to configs/experiment/aig_full.yaml

# WandB API Key (Optional but recommended if using 'online' mode)
# Best practice: Set this as an environment variable before submitting the job
 export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83"
# Or uncomment and set it here (less secure):
# WANDB_API_KEY="YOUR_API_KEY"

# --- End Configuration ---

# --- Helper Function for Error Checking ---
check_exit_code() {
  exit_code=$1
  step_name=$2
  if [ $exit_code -ne 0 ]; then
    echo "Error: Step '${step_name}' failed with exit code ${exit_code}."
    echo "Job failed at $(date)"
    # Optional: Add more debugging info if needed
    # echo "Last 50 lines of output from slurm log:"
    # tail -n 50 "./slurm_logs/digress_aig_train_${SLURM_JOB_ID}.out"
    exit $exit_code
  fi
  echo "Step '${step_name}' completed successfully."
}
# --- End Helper Function ---

# --- Setup ---
echo "Job started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Create log directory if it doesn't exist
mkdir -p ./slurm_logs
echo "SLURM log directory ensured."

echo "Loading modules..."
# Add module load commands specific to your cluster environment if needed
# module load 2024 # Example
# module load Anaconda3/2024.06-1 # Example
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
check_exit_code $? "Activate Conda Env"
echo "Conda environment activated: $(which python)"

# Set WandB API Key if defined in the script (use environment variable preferably)
# if [ -n "${WANDB_API_KEY}" ]; then
#   export WANDB_API_KEY=${WANDB_API_KEY}
#   echo "WANDB_API_KEY set."
# fi

# Check if main script exists
if [ ! -f "${MAIN_SCRIPT}" ]; then
    echo "Error: Main script not found at ${MAIN_SCRIPT}"
    exit 1
fi
# --- End Setup ---

# === Run DiGress Training & Evaluation ===
echo "========================================"
echo "Starting DiGress Training for AIG"
echo "Using Experiment Config: ${EXPERIMENT_CONFIG}"
echo "========================================"

# The main.py script handles training and the final evaluation internally now.
# We use Hydra to specify the experiment config.
# Hydra automatically finds configs relative to the main script's location
# if config_path is set correctly in @hydra.main decorator.
# Ensure config_path in main.py is correct (e.g., '../configs')

# Use srun if required by your cluster setup, otherwise just python
# Using python directly is often sufficient if SLURM allocates the GPU correctly.
# Add '-u' for unbuffered output to see logs in real-time
python -u ${MAIN_SCRIPT} +experiment=${EXPERIMENT_CONFIG}

check_exit_code $? "Run DiGress main.py"

# === End Training & Evaluation ===


echo "Job finished successfully at $(date)."

exit 0
