#!/bin/bash
#SBATCH --job-name=graphdf_aig_full_v3 # Updated job name for tuned run v3
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # Keep increased time
#SBATCH --output=./slurm_logs/graphdf_aig_full_%j.out # Updated log file name


# --- Configuration ---
MODEL_NAME="GraphDF"
REQUESTED_DEVICE="cuda"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name

# --- Data Configuration ---
# *** IMPORTANT: Set these to match your processed augmented dataset location ***
DATA_ROOT_PROCESSED="./data_pyg_augmented/"   # Root dir where the 'aig_augmented_ds' folder lives
DATASET_NAME="aig_ds" # The name used in the processing script

# Directory containing ORIGINAL training data (e.g., .bin files) needed ONLY for novelty check in evaluation
TRAIN_DATA_DIR_FOR_NOVELTY="./data/aigs" # Example: Adjust if needed

# --- Training Hyperparameters (Updated based on suggestions & analysis) ---
LR=0.0005                 # Suggested lower LR
WEIGHT_DECAY=1e-5           # Suggested weight decay
BATCH_SIZE=64             # Suggested batch size
MAX_EPOCHS=50             # Suggested increased epochs
SAVE_INTERVAL=5           # Save every 5 epochs
EDGE_UNROLL=25            # *** Updated based on analysis (99th=21, 99.9th=29) ***
GRAD_CLIP=1.0             # Suggested gradient clipping
LR_PATIENCE=5             # Suggested patience for LR scheduler
SAVE_DIR="${MODEL_NAME}/rand_gen_${DATASET_NAME}_ckpts" # Specific save dir for this run

# --- Validation/Generation Check Parameters ---
NUM_VAL_GEN=100           # Suggested samples for validation check

# --- Generation Parameters (for final sampling after training) ---
NUM_SAMPLES=1000          # Number of graphs to generate
TEMP_NODE=0.7             # Suggested generation temperature for nodes
TEMP_EDGE=0.7             # Suggested generation temperature for edges
MIN_NODES=5               # Minimum nodes for generated graphs
GEN_OUTPUT_DIR="./generated_graphs" # Separate output dir
# Include key generation parameters in the filename
GEN_PICKLE_FILENAME="${MODEL_NAME}_generated_${NUM_SAMPLES}_temp${TEMP_NODE}_${TEMP_EDGE}_epoch${MAX_EPOCHS}_${SLURM_JOB_ID}.pkl"
GEN_PICKLE_PATH="${GEN_OUTPUT_DIR}/${GEN_PICKLE_FILENAME}"

# --- Script Names ---
# *** Ensure these point to the CORRECT versions of your scripts ***
TRAIN_SCRIPT="train_graphs.py" # The script updated in the previous step
GEN_SCRIPT="sample_graphs.py"
EVAL_SCRIPT="evaluate_aigs.py"

# --- End Configuration ---


# --- Setup ---
mkdir -p slurm_logs
mkdir -p ${SAVE_DIR}
mkdir -p ${GEN_OUTPUT_DIR}
echo "Log and output directories ensured."
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1
echo "Modules loaded."
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
conda_status=$?
if [ $conda_status -ne 0 ]; then    echo "Error: Failed to activate conda environment '${CONDA_ENV_NAME}'. Exiting."    exit 1; fi
echo "Conda environment activated."
# --- End Setup ---


# === Step 1: Training ===
echo "========================================"
echo "Starting Training: ${MODEL_NAME} on ${DATASET_NAME} (Tuned v3)"
echo "========================================"
echo " - Data Root (Processed): ${DATA_ROOT_PROCESSED}"
echo " - Dataset Name: ${DATASET_NAME}"
echo " - Learning Rate: ${LR}"
echo " - Weight Decay: ${WEIGHT_DECAY}"
echo " - Batch Size: ${BATCH_SIZE}"
echo " - Max Epochs: ${MAX_EPOCHS}"
echo " - Edge Unroll: ${EDGE_UNROLL}" # Now set based on analysis
echo " - Grad Clip: ${GRAD_CLIP}"
echo " - LR Patience: ${LR_PATIENCE}"
echo " - Save Directory: ${SAVE_DIR}"
echo "----------------------------------------"

# Pass the updated and new arguments to the training script
srun python -u ${TRAIN_SCRIPT} \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root ${DATA_ROOT_PROCESSED} \
    --dataset_name ${DATASET_NAME} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --save_interval ${SAVE_INTERVAL} \
    --save_dir ${SAVE_DIR} \
    --edge_unroll ${EDGE_UNROLL} \
    --grad_clip_value ${GRAD_CLIP} \
    --use_lr_scheduler \
    --lr_scheduler_patience ${LR_PATIENCE} \
    --validate_generation \
    --num_val_gen_samples ${NUM_VAL_GEN} \
    --temperature_df ${TEMP_NODE} ${TEMP_EDGE} \
    `# Add other model architecture params if they differ from defaults`
    # --num_flow_layer 12 --num_rgcn_layer 3 --gaf_nhid 128 --gaf_nout 128 ...

train_status=$?
# (Error handling remains the same)
if [ $train_status -ne 0 ]; then    echo "Error: Training script failed with exit code ${train_status}." ; tail -n 50 ./slurm_logs/graphdf_aig_full_tuned_v3_${SLURM_JOB_ID}.out ; exit $train_status ; fi
echo "----------------------------------------"; echo "Training script finished successfully."; echo "========================================"


# === Step 2: Generation ===
echo ""; echo "========================================"; echo "Starting Generation: ${MODEL_NAME}"; echo "========================================"
CHECKPOINT_FILENAME="${MODEL_NAME,,}_ckpt_epoch_${MAX_EPOCHS}.pth" # Use lowercase model name
CHECKPOINT_PATH="${SAVE_DIR}/${CHECKPOINT_FILENAME}"
# (Checkpoint finding logic remains the same)
if [ ! -f "${CHECKPOINT_PATH}" ]; then echo "Warning: Final checkpoint '${CHECKPOINT_PATH}' not found." ; LATEST_CHECKPOINT=$(ls -t "${SAVE_DIR}"/${MODEL_NAME,,}_ckpt_epoch_*.pth 2>/dev/null | head -n 1); if [ -f "${LATEST_CHECKPOINT}" ]; then echo "Warning: Using latest found checkpoint instead: ${LATEST_CHECKPOINT}"; CHECKPOINT_PATH=${LATEST_CHECKPOINT}; else echo "Error: No checkpoints found in ${SAVE_DIR}. Cannot generate."; exit 1; fi; fi

echo " - Using Checkpoint: ${CHECKPOINT_PATH}"
echo " - Number of Samples: ${NUM_SAMPLES}"
echo " - Temperature (Node, Edge): ${TEMP_NODE}, ${TEMP_EDGE}"
echo " - Min Nodes: ${MIN_NODES}"
echo " - Output File: ${GEN_PICKLE_PATH}"
echo "----------------------------------------"

# Execute the generation script
srun python -u ${GEN_SCRIPT} \
    --model ${MODEL_NAME} \
    --checkpoint ${CHECKPOINT_PATH} \
    --output_file ${GEN_PICKLE_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --temperature_df ${TEMP_NODE} ${TEMP_EDGE} \
    --min_nodes ${MIN_NODES} \
    --device ${REQUESTED_DEVICE} \
    `# Pass necessary model config to generation script if needed`
    `# These should match the trained model!`
    --edge_unroll ${EDGE_UNROLL} \
    # --max_size 64 --node_dim 4 --bond_dim 3 ... (Add if sample_graphs.py needs them)

gen_status=$?
# (Error handling remains the same)
if [ $gen_status -ne 0 ]; then echo "Error: Generation script (${GEN_SCRIPT}) failed with exit code ${gen_status}." ; tail -n 50 ./slurm_logs/graphdf_aig_full_tuned_v3_${SLURM_JOB_ID}.out ; exit $gen_status ; fi
if [ ! -f "${GEN_PICKLE_PATH}" ]; then echo "Error: Generated pickle file '${GEN_PICKLE_PATH}' not found. Cannot evaluate." ; exit 1 ; fi
echo "----------------------------------------"; echo "Generation script finished successfully."; echo "========================================"


# === Step 3: Evaluation ===
echo ""; echo "========================================"; echo "Starting Evaluation"; echo "========================================"
echo " - Evaluating File: ${GEN_PICKLE_PATH}"
echo " - Training Data Dir (for Novelty): ${TRAIN_DATA_DIR_FOR_NOVELTY}" # Use the specific variable
echo "----------------------------------------"

# Execute the evaluation script
srun python -u ${EVAL_SCRIPT} \
    ${GEN_PICKLE_PATH} \
    --train_data_dir ${TRAIN_DATA_DIR_FOR_NOVELTY} \
    `# Add any other necessary args for evaluate_aigs.py`

eval_status=$?
# (Error handling remains the same)
if [ $eval_status -ne 0 ]; then echo "Error: Evaluation script (${EVAL_SCRIPT}) failed with exit code ${eval_status}." ; tail -n 50 ./slurm_logs/graphdf_aig_full_tuned_v3_${SLURM_JOB_ID}.out ; exit $eval_status ; fi
echo "----------------------------------------"; echo "Evaluation script finished successfully."; echo "========================================"
echo "Full pipeline completed."