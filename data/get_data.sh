#!/bin/bash
#SBATCH --job-name=pyg    # Job name for processing
#SBATCH --partition=gpu_h100         # Or a CPU partition if preferred/available
#SBATCH --gpus=1                     # Processing might not need GPU, but keeps env consistent
#SBATCH --time=03:00:99              # Adjust time as needed (processing can take time)
#SBATCH --output=../slurm_logs/to_pyg2_%j.out # Log file


# --- Configuration ---
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name


echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1 # Or your specific Anaconda module
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}

echo "Conda environment activated."
# --- End Setup ---


# Execute the processing script
srun python -u make_raw_data.py

# Deactivate environment (optional)
# conda deactivate
