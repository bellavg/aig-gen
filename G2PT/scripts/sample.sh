#!/bin/bash
#SBATCH --job-name=sample_eval_bfs
#SBATCH --partition=gpu_h100     # Or another suitable GPU partition
#SBATCH --gpus=1
#SBATCH --time=04:00:00          # Adjust time as needed for sampling
#SBATCH --output=../slurm_logs/sample_eval_bfs_%j.out

# Optional: Load API key if needed by sample/eval scripts (usually not)
# export WANDB_API_KEY="YOUR_API_KEY"

# Navigate to the base directory (G2PT/)
cd ..

# Ensure output directories exist (sample.py saves inside --out_dir)
# The evaluation script reads from this directory.
MODEL_OUT_DIR="results/aig-base-bfs"
mkdir -p slurm_logs
mkdir -p $MODEL_OUT_DIR # Ensure the model output dir exists

# Load necessary modules
module load 2024
module load Anaconda3/2024.06-1

# Activate Conda environment
source activate g2pt-aig # Use the correct environment name

echo "--- Starting Sampling Script ---"
# Run the sampling script
# - Use the output directory from training as --out_dir to load the checkpoint
# - Use the correct tokenizer path for AIG
# - Adjust --num_samples and --batch_size as needed
# - The output pickle file (default: generated_aigs.pkl) will be saved inside MODEL_OUT_DIR
srun python -u sample.py \
    --out_dir $MODEL_OUT_DIR \
    --tokenizer_path tokenizers/aig/ \
    --num_samples 25000 \
    --temperature 0.8 \
    --parsing_mode='robust' \
    --batch_size 256 \
    --seed 1337 \
    --output_filename generated_base_bfs_aigs.pkl # Optional: give specific name

echo "--- Sampling Finished ---"

echo "--- Starting Evaluation Script ---"
# Define the path to the generated pickle file
GENERATED_PICKLE_PATH="$MODEL_OUT_DIR/generated_base_bfs_aigs.pkl"

# Check if the generated file exists before running evaluation
if [ -f "$GENERATED_PICKLE_PATH" ]; then
    # Run the evaluation script on the generated pickle file
    srun python -u evaluate_aigs.py "$GENERATED_PICKLE_PATH"
else
    echo "Error: Generated pickle file not found at $GENERATED_PICKLE_PATH. Skipping evaluation."
fi

echo "--- Evaluation Finished ---"
echo "Script complete."
