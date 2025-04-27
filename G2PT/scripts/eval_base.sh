#!/bin/bash
#SBATCH --job-name=eval_base
#SBATCH --partition=gpu_h100     # Or another suitable GPU partition
#SBATCH --gpus=1
#SBATCH --time=00:59:00          # Adjust time as needed for sampling
#SBATCH --output=../slurm_logs/eval_thorough_%j.out

# Optional: Load API key if needed by sample/eval scripts (usually not)
# export WANDB_API_KEY="YOUR_API_KEY"

# Navigate to the base directory (G2PT/)
cd ..

# Ensure output directories exist (sample.py saves inside --out_dir)
# The evaluation script reads from this directory.
MODEL_OUT_DIR="results/aig-base-topo"
mkdir -p slurm_logs
mkdir -p $MODEL_OUT_DIR # Ensure the model output dir exists

# Load necessary modules
module load 2024
module load Anaconda3/2024.06-1

# Activate Conda environment
source activate g2pt-aig # Use the correct environment name

echo "--- Starting Evaluation Script ---"
# Define the path to the generated pickle file
GENERATED_PICKLE_PATH="$MODEL_OUT_DIR/cond_generated_uniform.pkl"

# Check if the generated file exists before running evaluation
if [ -f "$GENERATED_PICKLE_PATH" ]; then
    # Run the evaluation script on the generated pickle file
    srun python -u thorough_eval.py "$GENERATED_PICKLE_PATH"
else
    echo "Error: Generated pickle file not found at $GENERATED_PICKLE_PATH. Skipping evaluation."
fi

echo "--- Evaluation Finished ---"
echo "Script complete."
