#!/bin/bash
#SBATCH --job-name=aig_sample_eval # Job name
#SBATCH --partition=gpu_h100       # Specify partition (adjust if needed)
#SBATCH --gpus=1                   # Request 1 GPU
#SBATCH --time=02:00:00            # Adjust time limit (e.g., 2 hours)
#SBATCH --output=./slurm_logs/aig_sample_eval_%j.out # Log file pattern

# --- Configuration ---

# --- Model Selection ---
MODEL_NAME="GraphAF" # Choose: "GraphDF", "GraphAF", or "GraphEBM"

# --- Paths ---
# CHECKPOINT_PATH="GraphDF/rand_gen_aig_ckpts/graphdf_rand_gen_ckpt_epoch_50.pth" # Example for GraphDF
# CHECKPOINT_PATH="GraphAF/rand_gen_aig_ckpts/rand_gen_ckpt_epoch_50.pth" # Example for GraphAF
CHECKPOINT_PATH="GraphAF/rand_gen_aig_ckpts/rand_gen_ckpt_50.pth" # Example for GraphEBM
# Ensure the path above points to your actual trained model checkpoint

SAMPLING_OUTPUT_DIR="generated_aigs" # Directory to save the generated pickle file
SAMPLING_OUTPUT_FILENAME="${MODEL_NAME}_samples.pkl" # Name of the output pickle file
SAMPLING_OUTPUT_PATH="${SAMPLING_OUTPUT_DIR}/${SAMPLING_OUTPUT_FILENAME}" # Full path

# (Optional) Path to training data directory for novelty calculation in evaluate_aigs.py
# Needs to contain data_meta.json and train/ subdir with bin files
TRAIN_DATA_DIR="./G2PT/datasets/aig" # Set to your training data dir or leave empty ""

# --- Environment ---
CONDA_ENV_NAME="g2pt-aig" # Your conda environment name
REQUESTED_DEVICE="cuda"   # "cuda" or "cpu"

# --- Sampling Parameters ---
NUM_SAMPLES=1000     # Number of AIGs to generate
MIN_NODES=5          # Min nodes for filtering (GraphDF/GraphAF)

# Model-specific sampling parameters (only relevant ones are used based on MODEL_NAME)
TEMP_DF_NODE=0.3     # GraphDF node temperature
TEMP_DF_EDGE=0.3     # GraphDF edge temperature
TEMP_AF=0.6          # GraphAF temperature
EBM_C=0.0            # GraphEBM dequantization factor
EBM_LD_STEP=150      # GraphEBM Langevin steps
EBM_LD_NOISE=0.005   # GraphEBM Langevin noise
EBM_LD_STEP_SIZE=30  # GraphEBM Langevin step size
EBM_CLAMP_FLAG="--ebm_clamp" # Use "--ebm_clamp" to enable, "--no-ebm_clamp" to disable

# --- Scripts to Run ---
SAMPLING_SCRIPT="sample_graphs.py" # Your sampling script (from Canvas)
EVALUATION_SCRIPT="evaluate_aigs.py" # Your evaluation script

# --- WANDB (Optional) ---
# export WANDB_API_KEY="YOUR_API_KEY_HERE"

# --- End Configuration ---


# --- Script Execution ---

# Create directories
mkdir -p slurm_logs
mkdir -p "${SAMPLING_OUTPUT_DIR}"
echo "Log and output directories ensured."

# Load necessary environment modules (adjust based on your cluster)
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1 # Or your specific Anaconda/Python module
echo "Modules loaded."

# Activate the conda environment
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
conda_status=$?
if [ $conda_status -ne 0 ]; then
    echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'. Exiting."
    exit 1
fi
echo "Conda environment activated."


# --- Step 1: Run Sampling ---
echo "Starting AIG sampling script: ${SAMPLING_SCRIPT}..."
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Outputting to: ${SAMPLING_OUTPUT_PATH}"

# Construct model-specific arguments
MODEL_ARGS=""
if [ "${MODEL_NAME}" == "GraphDF" ]; then
    MODEL_ARGS="--temperature_df ${TEMP_DF_NODE} ${TEMP_DF_EDGE}"
elif [ "${MODEL_NAME}" == "GraphAF" ]; then
    MODEL_ARGS="--temperature_af ${TEMP_AF}"
elif [ "${MODEL_NAME}" == "GraphEBM" ]; then
    MODEL_ARGS="--ebm_c ${EBM_C} --ebm_ld_step ${EBM_LD_STEP} --ebm_ld_noise ${EBM_LD_NOISE} --ebm_ld_step_size ${EBM_LD_STEP_SIZE} ${EBM_CLAMP_FLAG}"
fi

# Run the sampling script using srun
# -u ensures unbuffered output
srun python -u ${SAMPLING_SCRIPT} \
    --model ${MODEL_NAME} \
    --checkpoint "${CHECKPOINT_PATH}" \
    --num_samples ${NUM_SAMPLES} \
    --output_file "${SAMPLING_OUTPUT_PATH}" \
    --min_nodes ${MIN_NODES} \
    --device ${REQUESTED_DEVICE} \
    ${MODEL_ARGS}

sample_status=$? # Capture the exit status

if [ $sample_status -ne 0 ]; then
    echo "Error: Sampling script (${SAMPLING_SCRIPT}) failed with exit code ${sample_status}. Evaluation will not run."
    exit $sample_status
fi

echo "Sampling script finished successfully."
echo "Generated samples saved to: ${SAMPLING_OUTPUT_PATH}"


# --- Step 2: Run Evaluation ---
echo "Starting AIG evaluation script: ${EVALUATION_SCRIPT}..."

# Construct evaluation arguments
EVAL_ARGS=""
if [ -n "${TRAIN_DATA_DIR}" ] && [ -d "${TRAIN_DATA_DIR}" ]; then
    EVAL_ARGS="--train_data_dir ${TRAIN_DATA_DIR}"
    echo "Using training data from ${TRAIN_DATA_DIR} for novelty calculation."
else
    echo "Training data directory not specified or not found. Novelty will not be calculated."
fi

# Check if sampling output file exists before running evaluation
if [ ! -f "${SAMPLING_OUTPUT_PATH}" ]; then
    echo "Error: Sampling output file (${SAMPLING_OUTPUT_PATH}) not found. Cannot run evaluation."
    exit 1
fi

# Run the evaluation script using srun
srun python -u ${EVALUATION_SCRIPT} \
    "${SAMPLING_OUTPUT_PATH}" \
    ${EVAL_ARGS}

eval_status=$? # Capture the exit status

if [ $eval_status -ne 0 ]; then
    echo "Error: Evaluation script (${EVALUATION_SCRIPT}) failed with exit code ${eval_status}."
    exit $eval_status
fi

echo "Evaluation script finished successfully."
echo "--- Job Complete ---"

