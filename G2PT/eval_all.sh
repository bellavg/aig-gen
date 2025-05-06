#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu_h100     # Or a cpu partition if evaluate_aigs.py doesn't need GPU
#SBATCH --gpus=1                 # Adjust if needed, maybe 0 if using CPU partition
#SBATCH --time=08:00:00          # Adjust time estimate based on number of files/graphs
#SBATCH --output=./slurm_logs/eval_all_%j.out

# Navigate to the base directory (G2PT/) relative to the script location (scripts/)

# Ensure log directory exists
mkdir -p slurm_logs

# Load necessary modules (same as your other scripts)
module load 2024
module load Anaconda3/2024.06-1

# Activate Conda environment (ensure evaluate_aigs dependencies are met)
source activate g2pt-aig # Use the correct environment name

echo "--- Starting Re-evaluation Script ---"
echo "Using updated evaluate_aigs.py"

# --- Find and Evaluate All .pkl Files in results/ ---
# Use find to locate all .pkl files within the results directory and its subdirs
# -print0 and read -d $'\0' handle filenames with spaces/special chars safely

find ./sampling_outputs -name '*.pkl' -print0 | while IFS= read -r -d $'\0' pkl_file; do
    echo "-----------------------------------------------------"
    echo "Evaluating file: $pkl_file"
    echo "-----------------------------------------------------"

    # Check if the file exists (redundant with find, but safe)
    if [ -f "$pkl_file" ]; then
        # Run the evaluation script on the found pickle file
        # Use srun if needed within the loop for Slurm resource tracking per file,
        # or run directly if the main job allocation is sufficient.
        # Running directly might be simpler here.
        python -u evaluate_aigs.py "$pkl_file" --train_data_dir "./datasets/aig/"
    else
        echo "Warning: File '$pkl_file' found by 'find' but seems not accessible?"
    fi
    echo # Add a newline for better log readability
done

echo "--- Re-evaluation Finished ---"
echo "Script complete."