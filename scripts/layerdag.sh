#!/bin/bash
#SBATCH --job-name=layerdag_aig_train   # Updated job name
#SBATCH --partition=gpu_h100            # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=16:00:00                # Initial requested time, adjust as needed
#SBATCH --output=../slurm_logs/layerdag_aig_%j.out # Updated output log file name

# Change to the LayerDAG project root directory.
# This assumes your SLURM script is in a subdirectory (e.g., 'slurm_scripts')
# of the main LayerDAG project directory. If train.py is in the same directory
# as this script, you can remove or comment out the 'cd ..' line.
cd ..

# --- Configuration ---
CONDA_ENV_NAME="LayerDAG" # Name of your existing Conda environment

echo "Current working directory: $(pwd)"

echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

# --- Activate Existing Conda Environment ---
echo ""
echo "========================================"
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
echo "========================================"
source activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: ${CONDA_ENV_NAME}. Please ensure it exists and is correctly set up."
    exit 1
fi
echo "Conda environment ${CONDA_ENV_NAME} activated."

# --- Set LD_LIBRARY_PATH ---
# Get the absolute path to the Conda environment
# CONDA_PREFIX is set when an environment is activated
CONDA_ENV_PATH=$(conda info --envs | grep "${CONDA_ENV_NAME}" | awk '{print $NF}') # $NF gets the last field (path)
if [ -z "${CONDA_ENV_PATH}" ] || [ ! -d "${CONDA_ENV_PATH}/lib" ]; then
    echo "Could not reliably determine Conda environment path or its lib directory via 'conda info --envs'."
    echo "Attempting to use CONDA_PREFIX environment variable."
    if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
        CONDA_ENV_LIB_PATH="$CONDA_PREFIX/lib"
        echo "Using CONDA_PREFIX for lib directory: ${CONDA_ENV_LIB_PATH}"
    else
        echo "CONDA_PREFIX is not set or its lib directory does not exist."
        # Fallback using the path from the error message if needed, but CONDA_PREFIX is better
        # Assuming your home directory is /home/igardner1, otherwise this needs adjustment
        CONDA_ENV_LIB_PATH_FALLBACK="/home/igardner1/.conda/envs/${CONDA_ENV_NAME}/lib"
        if [ -d "${CONDA_ENV_LIB_PATH_FALLBACK}" ]; then
            echo "Using fallback path for Conda lib directory: ${CONDA_ENV_LIB_PATH_FALLBACK}"
            CONDA_ENV_LIB_PATH="${CONDA_ENV_LIB_PATH_FALLBACK}"
        else
            echo "Fallback path also not found: ${CONDA_ENV_LIB_PATH_FALLBACK}. Exiting."
            exit 1
        fi
    fi
else
    CONDA_ENV_LIB_PATH="${CONDA_ENV_PATH}/lib"
    echo "Determined Conda lib directory: ${CONDA_ENV_LIB_PATH}"
fi

echo "Adding ${CONDA_ENV_LIB_PATH} to LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB_PATH}:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH is now: ${LD_LIBRARY_PATH}"
# --- End Set LD_LIBRARY_PATH ---


# Ensure WANDB_API_KEY is set in your environment or you have logged in via `wandb login`
export WANDB_API_KEY="725d958326cb39d0ba89d73b557c294f85ecbf83" # Added your W&B API Key

# --- Training Command for LayerDAG ---
echo ""
echo "========================================"
echo "Starting LayerDAG AIG Training"
echo "========================================"
echo "Running command:"
echo "srun python -u train.py --config_file configs/LayerDAG/aig.yaml"
echo "----------------------------------------"

# Execute the LayerDAG training script
# The -u flag is for unbuffered Python output, good for logs
srun python -u LayerDAG/src/train.py \
    --config_file configs/LayerDAG/aig.yaml \
    # You can add other arguments for train.py here if needed, e.g.:
    # --num_threads 16 \
    # --seed 42

# --- Completion ---
echo ""
echo "========================================"
echo "LayerDAG AIG Training Script Finished"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Deactivate conda environment
conda deactivate
echo "Conda environment deactivated."
