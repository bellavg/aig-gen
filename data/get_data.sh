#!/bin/bash
#SBATCH --job-name=process_aig_pyg    # Job name for processing
#SBATCH --partition=gpu_h100         # Or a CPU partition if preferred/available
#SBATCH --gpus=1                     # Processing might not need GPU, but keeps env consistent
#SBATCH --time=02:00:99              # Adjust time as needed (processing can take time)
#SBATCH --output=./slurm_logs/process_aig_pyg_%j.out # Log file

# --- Configuration ---
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name

# --- Script Name ---
# *** Ensure this is the correct name of your processing script ***
PROCESS_SCRIPT="digress_data_processing.py"

# --- End Configuration ---


# --- Setup ---
mkdir -p slurm_logs
# The Python script will create the output directories (output_root/dataset_name/processed)
echo "Log directory ensured."

echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1 # Or your specific Anaconda module
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
conda_status=$?
if [ $conda_status -ne 0 ]; then
    echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'. Exiting."
    exit 1
fi
echo "Conda environment activated."
# --- End Setup ---



# Check if the script exists
if [ ! -f "${PROCESS_SCRIPT}" ]; then
    echo "Error: Processing script '${PROCESS_SCRIPT}' not found."
    exit 1
fi

# Execute the processing script
srun python -u ${PROCESS_SCRIPT}


# Deactivate environment (optional)
# conda deactivate

exit 0