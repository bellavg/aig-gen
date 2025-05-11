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

echo "Log and output directories ensured."
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1
echo "Modules loaded."


source activate LayerDAG
echo "Conda environment activated."

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
srun python -u LayerDAG/train.py \
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
