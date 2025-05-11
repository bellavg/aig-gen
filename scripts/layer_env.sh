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
NEW_CONDA_ENV_NAME="LayerDAG" # Name for the new Conda environment

echo "Current working directory: $(pwd)"

echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

# --- Create and Set Up New Conda Environment ---
echo ""
echo "========================================"
echo "Setting up Conda environment: ${NEW_CONDA_ENV_NAME}"
echo "========================================"

# Check if the environment already exists
if conda env list | grep -q "${NEW_CONDA_ENV_NAME}"; then
    echo "Conda environment '${NEW_CONDA_ENV_NAME}' already exists. Activating it."
else
    echo "Creating new Conda environment: ${NEW_CONDA_ENV_NAME}..."
    conda create -n "${NEW_CONDA_ENV_NAME}" python=3.10 -y
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment: ${NEW_CONDA_ENV_NAME}"
        exit 1
    fi
    echo "Conda environment created. Activating and installing packages..."

    # Activate the new environment for subsequent pip/conda install commands
    source activate "${NEW_CONDA_ENV_NAME}"
    if [ $? -ne 0 ]; then
        echo "Failed to activate newly created conda environment: ${NEW_CONDA_ENV_NAME}"
        exit 1
    fi

    echo "Installing PyTorch 1.12.0 (attempting to resolve CUDA automatically)..."
    # Removed +cu116 suffix, relying on the extra-index-url or pip's default behavior
    pip install torch==1.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
    if [ $? -ne 0 ]; then echo "Failed to install PyTorch."; exit 1; fi

    # Removed explicit conda install cudatoolkit=11.6
    # echo "Installing CUDA Toolkit 11.6 via Conda..."
    # conda install -c conda-forge cudatoolkit=11.6 -y
    # if [ $? -ne 0 ]; then echo "Failed to install CUDA Toolkit."; exit 1; fi

    echo "Cleaning Conda cache..."
    conda clean --all -y

    echo "Installing DGL 1.1.0 (attempting to resolve CUDA automatically)..."
    # Removed +cu116 suffix, relying on the find-links URL or pip's default behavior
    pip install dgl==1.1.0 -f https://data.dgl.ai/wheels/cu116/repo.html
    if [ $? -ne 0 ]; then echo "Failed to install DGL."; exit 1; fi

    echo "Installing other Python packages (tqdm, einops, wandb, pydantic, pandas, numpy)..."
    pip install tqdm einops wandb pydantic pandas
    if [ $? -ne 0 ]; then echo "Failed to install base packages."; exit 1; fi
    pip install numpy==1.26.3
    if [ $? -ne 0 ]; then echo "Failed to install numpy."; exit 1; fi

    echo "Package installation complete for ${NEW_CONDA_ENV_NAME}."
fi

# Activate the target environment (either pre-existing or newly created and set up)
echo "Activating conda environment: ${NEW_CONDA_ENV_NAME} for the job..."
source activate "${NEW_CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: ${NEW_CONDA_ENV_NAME} for the job."
    exit 1
fi
echo "Conda environment ${NEW_CONDA_ENV_NAME} activated."


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
srun python -u train.py \
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
