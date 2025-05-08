#!/bin/bash
#SBATCH --job-name=file_investigator
#SBATCH --partition=gpu_h100 # Or your specific GPU partition
#SBATCH --gpus=1             # Requesting a GPU node to mimic training environment
#SBATCH --time=00:10:00      # Short time, only for investigation
#SBATCH --output=./slurm_logs/file_investigator_%j.out

# --- Configuration ---
CONDA_ENV_NAME="g2pt-aig"
FILE_TO_INVESTIGATE="/gpfs/home6/igardner1/aig-gen/data/aigs_pyg/aig/processed/train_processed_data.pt"
PARENT_DIR_PROCESSED=$(dirname "${FILE_TO_INVESTIGATE}")
PARENT_DIR_AIG=$(dirname "${PARENT_DIR_PROCESSED}")
PARENT_DIR_AIGS_PYG=$(dirname "${PARENT_DIR_AIG}")
PARENT_DIR_DATA=$(dirname "${PARENT_DIR_AIGS_PYG}")
PROJECT_ROOT=$(dirname "${PARENT_DIR_DATA}") # Should be /gpfs/home6/igardner1/aig-gen

echo "Investigation Script for File: ${FILE_TO_INVESTIGATE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on Host: $(hostname)"
echo "SLURM Submit Directory: ${SLURM_SUBMIT_DIR}"
echo "Current Working Directory: $(pwd)"
echo "Date: $(date)"
echo "--------------------------------------------------"

# --- Shell-Level Diagnostics ---
echo "--- Shell-Level Diagnostics ---"
echo "1. Checking Project Root:"
ls -ld "${PROJECT_ROOT}" || echo "Error listing Project Root: ${PROJECT_ROOT}"

echo -e "\n2. Checking .../data/ directory:"
ls -ld "${PARENT_DIR_DATA}" || echo "Error listing .../data/ directory: ${PARENT_DIR_DATA}"
ls -lA "${PARENT_DIR_DATA}" # List contents

echo -e "\n3. Checking .../data/aigs_pyg/ directory:"
ls -ld "${PARENT_DIR_AIGS_PYG}" || echo "Error listing .../data/aigs_pyg/ directory: ${PARENT_DIR_AIGS_PYG}"
ls -lA "${PARENT_DIR_AIGS_PYG}" # List contents

echo -e "\n4. Checking .../data/aigs_pyg/aig/ directory:"
ls -ld "${PARENT_DIR_AIG}" || echo "Error listing .../data/aigs_pyg/aig/ directory: ${PARENT_DIR_AIG}"
ls -lA "${PARENT_DIR_AIG}" # List contents

echo -e "\n5. Checking .../data/aigs_pyg/aig/processed/ directory:"
ls -ld "${PARENT_DIR_PROCESSED}" || echo "Error listing .../data/aigs_pyg/aig/processed/ directory: ${PARENT_DIR_PROCESSED}"
ls -lA "${PARENT_DIR_PROCESSED}" # List contents

echo -e "\n6. Checking the target file itself: ${FILE_TO_INVESTIGATE}"
ls -l "${FILE_TO_INVESTIGATE}" || echo "ls -l: File not found or inaccessible: ${FILE_TO_INVESTIGATE}"
stat "${FILE_TO_INVESTIGATE}" || echo "stat: File not found or inaccessible: ${FILE_TO_INVESTIGATE}"

echo "--------------------------------------------------"

# --- Python-Level Diagnostics ---
echo -e "\n--- Python-Level Diagnostics ---"
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1 # Adjust if your Anaconda module is different
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
if [ $? -ne 0 ]; then
    echo "FATAL: Failed to activate Conda environment ${CONDA_ENV_NAME}. Exiting."
    exit 1
fi
echo "Conda environment activated."
echo "Python version: $(python --version)"
echo "Python executable: $(which python)"

# Create a temporary Python script to perform checks
# This uses a heredoc to write the Python script content
cat > investigate_file.py << EOF
import os
import os.path as osp

print(f"Python: Running checks from {osp.abspath('.')}")

exact_path_to_check = "${FILE_TO_INVESTIGATE}"
print(f"Python: Hardcoded exact path to check: {exact_path_to_check}")

print(f"Python: Checking existence with os.path.exists('{exact_path_to_check}'): {osp.exists(exact_path_to_check)}")
print(f"Python: Checking if it's a file with os.path.isfile('{exact_path_to_check}'): {osp.isfile(exact_path_to_check)}")
print(f"Python: Checking if it's a directory with os.path.isdir('{exact_path_to_check}'): {osp.isdir(exact_path_to_check)}")

if osp.exists(exact_path_to_check):
    print(f"Python: Attempting to get file size with os.path.getsize...")
    try:
        file_size = osp.getsize(exact_path_to_check)
        print(f"Python: File size via os.path.getsize: {file_size} bytes")
    except Exception as e_size:
        print(f"Python: Error getting file size with os.path.getsize: {e_size}")

    print(f"Python: Attempting to get file stats with os.stat...")
    try:
        file_stat = os.stat(exact_path_to_check)
        print(f"Python: os.stat result: {file_stat}")
    except Exception as e_stat:
        print(f"Python: Error getting os.stat: {e_stat}")

    print(f"Python: Attempting to open the file for reading (binary)...")
    try:
        with open(exact_path_to_check, 'rb') as f_test:
            print(f"Python: Successfully opened '{exact_path_to_check}' for binary reading.")
            first_bytes = f_test.read(32) # Read first 32 bytes
            print(f"Python: First 32 bytes (if any): {first_bytes}")
        print(f"Python: Successfully closed the file.")
    except Exception as e_open:
        print(f"Python: Error opening file '{exact_path_to_check}': {e_open}")
else:
    print(f"Python: File '{exact_path_to_check}' does not exist according to os.path.exists.")

    # Check parent directories from Python's perspective
    current_path = exact_path_to_check
    for i in range(4): # Check up to 4 levels of parent directories
        parent_dir = osp.dirname(current_path)
        if parent_dir == current_path: # Reached root or an unresolvable path
            break
        print(f"Python: Checking parent directory '{parent_dir}':")
        print(f"  Exists: {osp.exists(parent_dir)}")
        print(f"  Is directory: {osp.isdir(parent_dir)}")
        if osp.exists(parent_dir) and osp.isdir(parent_dir):
            try:
                print(f"  Listing contents of '{parent_dir}': {os.listdir(parent_dir)}")
            except Exception as e_ls_py:
                print(f"  Error listing directory '{parent_dir}': {e_ls_py}")
        current_path = parent_dir
        if parent_dir == "/": # Stop if we reach the filesystem root
            break

print(f"Python: End of Python checks.")
EOF

# Execute the temporary Python script
echo -e "\nExecuting Python investigation script (investigate_file.py)..."
python investigate_file.py

# Clean up the temporary Python script
rm investigate_file.py

echo "--------------------------------------------------"
echo "Investigation script finished."
