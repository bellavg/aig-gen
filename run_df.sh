#!/bin/bash
#SBATCH --job-name=graphdf_aig_full_v1 # Updated job name for full pipeline
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=08:00:00 # Increased time for gen + eval
#SBATCH --output=./slurm_logs/graphdf_aig_full_tuned_v1_%j.out # Updated log file name


# --- Configuration ---
MODEL_NAME="GraphDF"
REQUESTED_DEVICE="cuda"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name
DATA_ROOT="./data/"
# Directory containing training data (e.g., data/aig/train/xs.bin) needed for novelty check
TRAIN_DATA_DIR="${DATA_ROOT}aig"

# Training Hyperparameters
LR=0.001
WEIGHT_DECAY=0
BATCH_SIZE=32      # Reduced batch size (closer to paper)
MAX_EPOCHS=20      # Reduced epochs (closer to paper)
SAVE_INTERVAL=5    # Save every 5 epochs
SAVE_DIR="${MODEL_NAME}/rand_gen_aig_ckpts_tuned_v1" # Specific save dir for this run

# Generation Parameters
NUM_SAMPLES=1000 # Number of graphs to generate (adjust as needed)
TEMP_NODE=0.6    # Generation temperature for nodes (tune this)
TEMP_EDGE=0.6    # Generation temperature for edges (tune this)
MIN_NODES=5      # Minimum nodes for generated graphs
GEN_OUTPUT_DIR="./generated_graphs"
# Include key generation parameters in the filename for easy identification
GEN_PICKLE_FILENAME="${MODEL_NAME}_generated_${NUM_SAMPLES}_temp${TEMP_NODE}_${TEMP_EDGE}_epoch${MAX_EPOCHS}_${SLURM_JOB_ID}.pkl"
GEN_PICKLE_PATH="${GEN_OUTPUT_DIR}/${GEN_PICKLE_FILENAME}"

# Script Names
TRAIN_SCRIPT="train_graphs.py"
GEN_SCRIPT="sample_graphs.py" # Use the correct script name
EVAL_SCRIPT="evaluate_aigs.py"

# --- End Configuration ---


# --- Setup ---
# Create directories if they don't exist
mkdir -p slurm_logs
mkdir -p ${SAVE_DIR}
mkdir -p ${GEN_OUTPUT_DIR}
echo "Log and output directories ensured."

# Load necessary environment modules
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1
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
# --- End Setup ---


# === Step 1: Training ===
echo "========================================"
echo "Starting Training: ${MODEL_NAME}"
echo "========================================"
echo " - Learning Rate: ${LR}"
echo " - Batch Size: ${BATCH_SIZE}"
echo " - Max Epochs: ${MAX_EPOCHS}"
echo " - Save Directory: ${SAVE_DIR}"
echo "----------------------------------------"

srun python -u ${TRAIN_SCRIPT} \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root ${DATA_ROOT} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --save_interval ${SAVE_INTERVAL} \
    --save_dir ${SAVE_DIR} \
    `# Model architecture params will use defaults from train_graphs.py base_conf`

train_status=$?

if [ $train_status -ne 0 ]; then
    echo "Error: Training script failed with exit code ${train_status}."
    echo "Last 50 lines of training output:"
    tail -n 50 ./slurm_logs/graphdf_aig_full_tuned_v1_${SLURM_JOB_ID}.out
    exit $train_status
fi
echo "----------------------------------------"
echo "Training script finished successfully."
echo "========================================"


# === Step 2: Generation ===
echo ""
echo "========================================"
echo "Starting Generation: ${MODEL_NAME}"
echo "========================================"
# Assuming the last checkpoint saved is the one to use
CHECKPOINT_FILENAME="graphdf_rand_gen_ckpt_epoch_${MAX_EPOCHS}.pth"
CHECKPOINT_PATH="${SAVE_DIR}/${CHECKPOINT_FILENAME}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Warning: Final checkpoint '${CHECKPOINT_PATH}' not found after training."
    # Attempt to find the latest saved checkpoint as a fallback
    LATEST_CHECKPOINT=$(ls -t "${SAVE_DIR}"/graphdf_rand_gen_ckpt_epoch_*.pth 2>/dev/null | head -n 1)
    if [ -f "${LATEST_CHECKPOINT}" ]; then
        echo "Warning: Using latest found checkpoint instead: ${LATEST_CHECKPOINT}"
        CHECKPOINT_PATH=${LATEST_CHECKPOINT}
    else
        echo "Error: No checkpoints found in ${SAVE_DIR}. Cannot generate. Exiting."
        exit 1
    fi
fi

echo " - Using Checkpoint: ${CHECKPOINT_PATH}"
echo " - Number of Samples: ${NUM_SAMPLES}"
echo " - Temperature (Node, Edge): ${TEMP_NODE}, ${TEMP_EDGE}"
echo " - Min Nodes: ${MIN_NODES}"
echo " - Output File: ${GEN_PICKLE_PATH}"
echo "----------------------------------------"

# Execute the generation script (sample_graphs.py)
srun python -u ${GEN_SCRIPT} \
    --model ${MODEL_NAME} \
    --checkpoint ${CHECKPOINT_PATH} \
    --output_file ${GEN_PICKLE_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --temperature_df ${TEMP_NODE} ${TEMP_EDGE} \
    --min_nodes ${MIN_NODES} \
    --device ${REQUESTED_DEVICE} \
    `# Add any other necessary args required by sample_graphs.py`
    `# e.g., model config params if not loaded from checkpoint or hardcoded`
    # --max_size 64 --node_dim 4 --bond_dim 3 --edge_unroll 12 ...

gen_status=$?

if [ $gen_status -ne 0 ]; then
    echo "Error: Generation script (${GEN_SCRIPT}) failed with exit code ${gen_status}."
    echo "Last 50 lines of output (includes training and generation):"
    tail -n 50 ./slurm_logs/graphdf_aig_full_tuned_v1_${SLURM_JOB_ID}.out
    exit $gen_status
fi

if [ ! -f "${GEN_PICKLE_PATH}" ]; then
    echo "Error: Generated pickle file '${GEN_PICKLE_PATH}' not found after generation script ran. Cannot evaluate."
    exit 1
fi
echo "----------------------------------------"
echo "Generation script finished successfully."
echo "========================================"


# === Step 3: Evaluation ===
echo ""
echo "========================================"
echo "Starting Evaluation"
echo "========================================"
echo " - Evaluating File: ${GEN_PICKLE_PATH}"
echo " - Training Data Dir (for Novelty): ${TRAIN_DATA_DIR}"
echo "----------------------------------------"

# Execute the evaluation script
srun python -u ${EVAL_SCRIPT} \
    ${GEN_PICKLE_PATH} \
    --train_data_dir ${TRAIN_DATA_DIR} \
    `# Add any other necessary args for evaluate_aigs.py`

eval_status=$?

if [ $eval_status -ne 0 ]; then
    echo "Error: Evaluation script (${EVAL_SCRIPT}) failed with exit code ${eval_status}."
    echo "Last 50 lines of output (includes training, generation, and evaluation):"
    tail -n 50 ./slurm_logs/graphdf_aig_full_tuned_v1_${SLURM_JOB_ID}.out
    exit $eval_status
fi
echo "----------------------------------------"
echo "Evaluation script finished successfully."
echo "========================================"
echo "Full pipeline completed."

