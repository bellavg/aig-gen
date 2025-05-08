#!/bin/bash
#SBATCH --job-name=graphdf_aig_train_gen_v3 # Updated job name
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # Keep increased time
#SBATCH --output=./slurm_logs/graphdf_aig_train_gen_%j.out # Updated log file name


# --- Configuration ---
MODEL_NAME="GraphDF"
REQUESTED_DEVICE="cuda"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name

# --- Data Configuration ---
# Processed Data Location (for loading)
DATA_ROOT_PROCESSED="./aigs_pyg/"   # Root dir containing the 'aig' folder
DATASET_NAME="aig"              # The dataset subfolder name (contains 'processed')
# Raw Data Location (for dataset context during loading)
RAW_DATA_DIR="./data/aigs"              # *** ADDED: Directory containing original PKL files ***
RAW_FILE_PREFIX="real_aigs_part_" # *** ADDED: Prefix for original PKL files ***
NUM_TRAIN_FILES=4                 # *** ADDED: Number of PKL files for train split ***
NUM_VAL_FILES=1                   # *** ADDED: Number of PKL files for val split ***
NUM_TEST_FILES=1                  # *** ADDED: Number of PKL files for test split ***
# Training Data Location (for novelty evaluation)
TRAIN_DATA_DIR_FOR_NOVELTY="./data/aigs" # Example: Adjust if needed

# --- Training Hyperparameters ---
LR=0.0005
WEIGHT_DECAY=1e-5
BATCH_SIZE=64
MAX_EPOCHS=50
SAVE_INTERVAL=5
EDGE_UNROLL=25            # *** Crucial: MUST match the intended model architecture ***
GRAD_CLIP=1.0
NUM_AUGMENTATIONS=5       # Number of augmentations used for training data
SAVE_DIR="${MODEL_NAME}/rand_gen_${DATASET_NAME}_ckpts" # Specific save dir for this run

# Architecture Defaults (Ensure these match training if not overridden)
MAX_SIZE=64 # Corresponds to MAX_NODES_PAD
NODE_DIM=4  # Corresponds to NUM_NODE_FEATURES
BOND_DIM=3  # Corresponds to NUM_ADJ_CHANNELS
NUM_FLOW_LAYER=12
NUM_RGCN_LAYER=3
GAF_NHID=128
GAF_NOUT=128

# --- Generation Parameters (for final sampling after training) ---
NUM_SAMPLES=1000          # Number of graphs to generate
TEMP_NODE=0.7             # Generation temperature for nodes (GraphDF)
TEMP_EDGE=0.7             # Generation temperature for edges (GraphDF)
MIN_NODES=5               # Minimum nodes for generated graphs
GEN_OUTPUT_DIR="./generated_graphs" # Separate output dir
# Include key generation parameters in the filename
GEN_PICKLE_FILENAME="${MODEL_NAME}_generated_${NUM_SAMPLES}_temp${TEMP_NODE}_${TEMP_EDGE}_epoch${MAX_EPOCHS}_${SLURM_JOB_ID}.pkl"
GEN_PICKLE_PATH="${GEN_OUTPUT_DIR}/${GEN_PICKLE_FILENAME}"

# --- Script Names ---
# *** Ensure these point to the CORRECT versions of your scripts ***
TRAIN_SCRIPT="train_graphs.py" # The simplified script that delegates training
GEN_SCRIPT="sample_graphs.py"
EVAL_SCRIPT="evaluate_aigs.py"
# --- End Configuration ---


# --- Helper Function for Error Checking ---
check_exit_code() {
  exit_code=$1
  step_name=$2
  if [ $exit_code -ne 0 ]; then
    echo "Error: Step '${step_name}' failed with exit code ${exit_code}."
    echo "Last 50 lines of output from slurm log:"
    tail -n 50 "./slurm_logs/graphdf_aig_train_gen_${SLURM_JOB_ID}.out" # Match log file name
    exit $exit_code
  fi
  echo "Step '${step_name}' completed successfully."
}
# --- End Helper Function ---


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
check_exit_code $? "Activate Conda Env"
echo "Conda environment activated."
# --- End Setup ---


# === Step 1: Training ===
echo "========================================"
echo "Starting Training: ${MODEL_NAME} on ${DATASET_NAME}"
echo "========================================"
echo " - Data Root (Processed): ${DATA_ROOT_PROCESSED}"
echo " - Dataset Name: ${DATASET_NAME}"
echo " - Raw Data Dir: ${RAW_DATA_DIR}"
echo " - Learning Rate: ${LR}"
echo " - Weight Decay: ${WEIGHT_DECAY}"
echo " - Batch Size: ${BATCH_SIZE}"
echo " - Max Epochs: ${MAX_EPOCHS}"
echo " - Edge Unroll: ${EDGE_UNROLL}"
echo " - Grad Clip: ${GRAD_CLIP}"
echo " - Save Directory: ${SAVE_DIR}"
echo "----------------------------------------"

# Call the simplified train_graphs.py script, ensuring all required args are present
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
    --num_augmentations ${NUM_AUGMENTATIONS} \
    --raw_data_dir ${RAW_DATA_DIR} \
    --raw_file_prefix ${RAW_FILE_PREFIX} \
    --num_train_files ${NUM_TRAIN_FILES} \
    --num_val_files ${NUM_VAL_FILES} \
    --num_test_files ${NUM_TEST_FILES} \
    --num_flow_layer ${NUM_FLOW_LAYER} \
    --num_rgcn_layer ${NUM_RGCN_LAYER} \
    --gaf_nhid ${GAF_NHID} \
    --gaf_nout ${GAF_NOUT} \
    # Add --st_type, --deq_coeff if needed by train_graphs.py



# === Step 2: Generation ===
echo ""; echo "========================================"; echo "Starting Generation: ${MODEL_NAME}"; echo "========================================"
CHECKPOINT_FILENAME="${MODEL_NAME,,}_ckpt_epoch_${MAX_EPOCHS}.pth" # Use lowercase model name
CHECKPOINT_PATH="${SAVE_DIR}/${CHECKPOINT_FILENAME}"
# Checkpoint finding logic
if [ ! -f "${CHECKPOINT_PATH}" ]; then echo "Warning: Final checkpoint '${CHECKPOINT_PATH}' not found." ; LATEST_CHECKPOINT=$(ls -t "${SAVE_DIR}"/${MODEL_NAME,,}_ckpt_epoch_*.pth 2>/dev/null | head -n 1); if [ -f "${LATEST_CHECKPOINT}" ]; then echo "Warning: Using latest found checkpoint instead: ${LATEST_CHECKPOINT}"; CHECKPOINT_PATH=${LATEST_CHECKPOINT}; else echo "Error: No checkpoints found in ${SAVE_DIR}. Cannot generate."; exit 1; fi; fi

echo " - Using Checkpoint: ${CHECKPOINT_PATH}"
echo " - Number of Samples: ${NUM_SAMPLES}"
echo " - Temperature (Node, Edge): ${TEMP_NODE}, ${TEMP_EDGE}" # GraphDF uses two temps
echo " - Min Nodes: ${MIN_NODES}"
echo " - Output File: ${GEN_PICKLE_PATH}"
echo "----------------------------------------"

# Execute the generation script
# *** Use --temperature_df for GraphDF ***
srun python -u ${GEN_SCRIPT} \
    --model ${MODEL_NAME} \
    --checkpoint ${CHECKPOINT_PATH} \
    --output_file ${GEN_PICKLE_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --temperature_df ${TEMP_NODE} ${TEMP_EDGE} \
    --min_nodes ${MIN_NODES} \
    --device ${REQUESTED_DEVICE} \
    `# *** Added necessary ARCHITECTURE args for model instantiation ***` \
    `# These MUST match the parameters used during training!` \
    --edge_unroll ${EDGE_UNROLL} \
    --max_size ${MAX_SIZE} \
    --node_dim ${NODE_DIM} \
    --bond_dim ${BOND_DIM} \
    --num_flow_layer ${NUM_FLOW_LAYER} \
    --num_rgcn_layer ${NUM_RGCN_LAYER} \
    --gaf_nhid ${GAF_NHID} \
    --gaf_nout ${GAF_NOUT} \
    # Add --st_type, --deq_coeff if needed by sample_graphs.py


# === Step 3: Evaluation ===
echo ""; echo "========================================"; echo "Starting Evaluation"; echo "========================================"
echo " - Evaluating File: ${GEN_PICKLE_PATH}"
echo " - Training Data Dir (for Novelty): ${TRAIN_DATA_DIR_FOR_NOVELTY}"
echo "----------------------------------------"

# Execute the evaluation script
srun python -u ${EVAL_SCRIPT} \
    ${GEN_PICKLE_PATH} \
    --train_data_dir ${TRAIN_DATA_DIR_FOR_NOVELTY} \


