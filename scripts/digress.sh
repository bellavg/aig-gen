#!/bin/bash
#SBATCH --job-name=digress_aig_train
#SBATCH --partition=gpu_h100          # Or your specific H100 partition
#SBATCH --gpus=1
#SBATCH --time=16:00:00              # Initial requested time, adjust as needed
#SBATCH --output=../slurm_logs/digress_%j.out

cd ..

#!/bin/bash

# Script to set up the DiGress environment on a GPU cluster
# This version excludes RDKit installation.

# --- Configuration ---
CONDA_ENV_NAME="digress_gpu"
PYTHON_VERSION="3.9"
GRAPH_TOOL_VERSION="2.45"

# --- CUDA and PyTorch Version Configuration ---
# Defaulting to CUDA 11.8 as it's a common version compatible with PyTorch 2.0.1.
# You can change these if your cluster requires a different CUDA version
# or if you want to use a different PyTorch build.
# Ensure TARGET_CUDA_VERSION and PYTORCH_CUDA_SUFFIX are compatible.
TARGET_CUDA_VERSION="11.8.0" # Defaulting to CUDA 11.8.0
PYTORCH_CUDA_SUFFIX="cu118"   # Corresponding suffix for PyTorch with CUDA 11.8

PYTORCH_VERSION="2.0.1" # As specified in the DiGress README

# Get the directory where the script is located, assuming DiGress project root
# If you place this script in the root of the DiGress project, this should work.
# Otherwise, adjust SCRIPT_DIR or manually set PROJECT_ROOT.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}" # Assuming script is in DiGress project root

# --- Helper Functions ---
echo_step() {
    echo "-----------------------------------------------------"
    echo "STEP: $1"
    echo "-----------------------------------------------------"
}

echo_info() {
    echo "INFO: $1"
}

echo_warning() {
    echo "WARNING: $1"
}

echo_error() {
    echo "ERROR: $1" >&2
}

check_command_success() {
    if [ $? -ne 0 ]; then
        echo_error "Previous command failed. Exiting."
        exit 1
    fi
}

check_python_import() {
    echo_info "Checking Python import: $1"
    python3 -c "import $1"
    if [ $? -ne 0 ]; then
        echo_error "Failed to import '$1' in the '$CONDA_ENV_NAME' environment."
        echo_info "Please check the installation steps and ensure the environment is activated."
        # exit 1 # Optionally exit on failed check
    else
        echo_info "Successfully imported '$1'."
    fi
}


# --- Main Setup ---

echo_step "Starting DiGress Environment Setup (No RDKit)"
echo_info "Using default CUDA Version: $TARGET_CUDA_VERSION and PyTorch CUDA Suffix: $PYTORCH_CUDA_SUFFIX"
echo_info "You can edit this script to change these versions if needed."

# 1. Create Conda Environment (without RDKit)
echo_step "Creating Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION"
conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
check_command_success

# 2. Activate Conda Environment
echo_step "Activating Conda environment '$CONDA_ENV_NAME'"
# Note: `conda activate` might not work directly in scripts in all shells.
# This attempts to initialize conda for bash if not already done.
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
check_command_success
echo_info "Environment '$CONDA_ENV_NAME' activated."
echo_info "Python version: $(python --version)"

# 3. Install graph-tool
echo_step "Installing graph-tool version $GRAPH_TOOL_VERSION"
conda install -y -c conda-forge graph-tool="$GRAPH_TOOL_VERSION"
check_command_success
check_python_import "graph_tool"

# 4. Install NVCC drivers for your CUDA version
echo_step "Installing CUDA toolkit (nvcc drivers) version $TARGET_CUDA_VERSION"
echo_info "This step assumes your cluster nodes have the necessary NVIDIA drivers installed."
echo_info "We are installing the CUDA toolkit within the conda environment."
conda install -y -c "nvidia/label/cuda-${TARGET_CUDA_VERSION}" cuda
check_command_success
echo_info "CUDA toolkit installation command executed. Verify nvcc is available if needed:"
echo_info "  (After script finishes, in activated env: nvcc --version)"


# 5. Install PyTorch with corresponding CUDA version
echo_step "Installing PyTorch $PYTORCH_VERSION for CUDA suffix $PYTORCH_CUDA_SUFFIX"
pip3 install torch=="$PYTORCH_VERSION" --index-url https://download.pytorch.org/whl/"$PYTORCH_CUDA_SUFFIX"
check_command_success
check_python_import "torch"
echo_info "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo_info "CUDA available to PyTorch: $(python3 -c 'import torch; print(torch.cuda.is_available())')"


# 6. Install other packages from requirements.txt
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo_step "Installing packages from $REQUIREMENTS_FILE"
    pip install -r "$REQUIREMENTS_FILE"
    check_command_success
else
    echo_warning "$REQUIREMENTS_FILE not found. Skipping pip install from requirements file."
    echo_warning "Please ensure you are running this script from the DiGress project root or have requirements.txt available."
fi

# 7. Install DiGress project in editable mode
echo_step "Installing DiGress in editable mode"
pip install -e "${PROJECT_ROOT}"
check_command_success

# 8. Compile orca.cpp
ORCA_DIR="${PROJECT_ROOT}/src/analysis/orca"
ORCA_EXEC="${ORCA_DIR}/orca"
if [ -d "$ORCA_DIR" ] && [ -f "${ORCA_DIR}/orca.cpp" ]; then
    echo_step "Compiling orca.cpp"
    cd "$ORCA_DIR" || { echo_error "Failed to cd into $ORCA_DIR"; exit 1; }
    g++ -O2 -std=c++11 -o orca orca.cpp
    check_command_success
    if [ -f "$ORCA_EXEC" ]; then
        echo_info "orca compiled successfully at $ORCA_EXEC"
    else
        echo_warning "orca compilation command executed, but $ORCA_EXEC not found. Check compilation output."
    fi
    cd "$PROJECT_ROOT" || { echo_error "Failed to cd back to $PROJECT_ROOT"; exit 1; }
else
    echo_warning "orca.cpp or its directory not found at $ORCA_DIR. Skipping orca compilation."
    echo_warning "If orca is needed, please ensure the path is correct and files exist."
fi

echo_step "DiGress Environment Setup (No RDKit) Complete!"
echo_info "To use the environment, activate it with: conda activate $CONDA_ENV_NAME"
echo_info "Remember to set general.abs_path_to_project_root in your Hydra configs or pass it via command line when running main.py."
echo_info "Example run command from project root:"
echo_info "  conda activate $CONDA_ENV_NAME"
echo_info "  python src/main.py experiment=aig dataset=aig general.abs_path_to_project_root=\$(pwd)"
echo "-----------------------------------------------------"


cd DiGress
pip install -e .
cd ..
# --- Configuration ---
CONDA_ENV_NAME="digress_gpu" # CHANGE THIS to your Conda environment name
PROJECT_ROOT="./Digress" # IMPORTANT: SET THIS!


echo "Loading modules..."
module load 2024 # Or your specific module environment
module load Anaconda3/2024.06-1 # Or your Anaconda module
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate "${CONDA_ENV_NAME}"

# Weights and bias key: 725d958326cb39d0ba89d73b557c294f85ecbf83
# --- Training Command ---
echo ""
echo "========================================"
echo "Starting DiGress AIG Training"
echo "========================================"
echo "Running command:"
echo "srun python -u src/main.py experiment=aig dataset=aig general.abs_path_to_project_root=${PROJECT_ROOT}"
echo "----------------------------------------"

# Execute the training script
# The -u flag is for unbuffered Python output, good for logs
srun python -u DiGress/src/main.py \
    +experiment=aig.yaml \
    dataset=aig \
    +general.abs_path_to_project_root="${PROJECT_ROOT}" \
    # Add any other specific Hydra overrides here if needed, e.g.:
    # general.resume=outputs/YYYY-MM-DD/HH-MM-SS-aig_resume/checkpoints/last-epoch=XXXX.ckpt
    # train.n_epochs=6000 # To override the config file value

# --- Completion ---
echo ""
echo "========================================"
echo "DiGress AIG Training Script Finished"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Deactivate conda environment
conda deactivate
