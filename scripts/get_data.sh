#!/bin/bash
#SBATCH --job-name=process_aig_pyg    # Job name for processing
#SBATCH --partition=gpu_h100         # Or a CPU partition if preferred/available
#SBATCH --gpus=1                     # Processing might not need GPU, but keeps env consistent
#SBATCH --time=03:00:99              # Adjust time as needed (processing can take time)
#SBATCH --output=../slurm_logs/process_aig_pyg_%j.out # Log file

cd ..

# --- Configuration ---
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name

# --- Data Paths ---
RAW_PKL_DIR="./data/aigs"       # *** Directory containing input PKL files ***
OUTPUT_ROOT_DIR="./ggraph/data/aigs_pyg" # *** Root directory for processed output ***
FILE_PREFIX="real_aigs_part_"      # *** Prefix of your PKL files ***

# --- File Allocation ---
# Define how many PKL files go into each split
NUM_TRAIN_FILES=4
NUM_VAL_FILES=1
NUM_TEST_FILES=1

# --- Script Name ---
# *** Ensure this is the correct name of your processing script ***
PROCESS_SCRIPT="dataset.py"


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


# === Run Processing Script ===
echo "========================================"
echo "Starting AIG PKL to PyG Processing"
echo "========================================"
echo " - Raw PKL Directory : ${RAW_PKL_DIR}"
echo " - Output Root       : ${OUTPUT_ROOT_DIR}"
echo " - File Prefix       : ${FILE_PREFIX}"
echo " - File Splits (T/V/T): ${NUM_TRAIN_FILES}/${NUM_VAL_FILES}/${NUM_TEST_FILES}"
echo "----------------------------------------"

# Execute the processing script
srun python -u  ggraph/dataset.py \
    --raw_dir "${RAW_PKL_DIR}" \
    --output_root "${OUTPUT_ROOT_DIR}" \
    --file_prefix "${FILE_PREFIX}" \
    --num_train_files ${NUM_TRAIN_FILES} \
    --run_conversion_test


# Deactivate environment (optional)
# conda deactivate

exit 0