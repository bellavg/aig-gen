#!/bin/bash
#SBATCH --job-name=graphaf_aig_train_gen # Job name for GraphAF
#SBATCH --partition=gpu_h100 # Or your specific GPU partition
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # Adjust time as needed for GraphAF
#SBATCH --output=./slurm_logs/graphaf_aig_train_gen_%j.out # Log file specific to GraphAF


# --- Configuration ---
MODEL_NAME="GraphAF" # ****** Set Model Name ******
REQUESTED_DEVICE="cuda"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name

# --- Data Configuration ---
# Processed Data Location (for loading)
DATA_ROOT_PROCESSED_REL="./data/aigs_pyg"    # Relative root dir containing the 'aig' folder
DATA_ROOT_PROCESSED=$(realpath -m "${DATA_ROOT_PROCESSED_REL}") # Absolute path
DATASET_NAME="aig"              # The dataset subfolder name (contains 'processed')

# Raw Data Location (for dataset context during loading)
RAW_DATA_DIR_REL="./data/aigs"              # Relative directory containing original PKL files
RAW_DATA_DIR=$(realpath -m "${RAW_DATA_DIR_REL}") # Absolute path
RAW_FILE_PREFIX="real_aigs_part_" # Prefix for original PKL files
NUM_TRAIN_FILES=4                 # Number of PKL files for train split
NUM_VAL_FILES=1                   # Number of PKL files for val split
NUM_TEST_FILES=1                  # Number of PKL files for test split

# Training Data Location (for novelty evaluation)
TRAIN_DATA_DIR_FOR_NOVELTY_REL="./data/aigs" # Relative path for novelty evaluation
TRAIN_DATA_DIR_FOR_NOVELTY=$(realpath -m "${TRAIN_DATA_DIR_FOR_NOVELTY_REL}") # Absolute path

# --- Training Hyperparameters ---
LR=0.0005
WEIGHT_DECAY=1e-5
BATCH_SIZE=64
MAX_EPOCHS=50
SAVE_INTERVAL=5
EDGE_UNROLL=25
GRAD_CLIP=1.0
NUM_AUGMENTATIONS=5

# Save Directory (made absolute)
SAVE_DIR_REL="${MODEL_NAME}/rand_gen_${DATASET_NAME}_ckpts" # Relative specific save dir for GraphAF run
SAVE_DIR=$(realpath -m "${SAVE_DIR_REL}") # Absolute path

# Architecture Defaults (Ensure these match training if not overridden)
MAX_SIZE=64
NODE_DIM=4
BOND_DIM=3
NUM_FLOW_LAYER=12
NUM_RGCN_LAYER=3
GAF_NHID=128
GAF_NOUT=128

# --- Generation Parameters (for final sampling after training) ---
NUM_SAMPLES=1000
TEMPERATURE_AF=0.75
MIN_NODES=5

# Generation Output Directory (made absolute)
GEN_OUTPUT_DIR_REL="./generated_graphs" # Relative separate output dir
GEN_OUTPUT_DIR=$(realpath -m "${GEN_OUTPUT_DIR_REL}") # Absolute path

# Include key generation parameters in the filename
# GEN_PICKLE_PATH will be absolute because GEN_OUTPUT_DIR is now absolute
GEN_PICKLE_FILENAME="${MODEL_NAME}_generated_${NUM_SAMPLES}_temp${TEMPERATURE_AF}_epoch${MAX_EPOCHS}_${SLURM_JOB_ID}.pkl"
GEN_PICKLE_PATH="${GEN_OUTPUT_DIR}/${GEN_PICKLE_FILENAME}"

# --- Script Names ---
TRAIN_SCRIPT="train_graphs.py"
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
    # Ensure this path is correct, relative to job's CWD or make it absolute too if needed
    tail -n 50 "./slurm_logs/graphaf_aig_train_gen_${SLURM_JOB_ID}.out"
    exit $exit_code
  fi
  echo "Step '${step_name}' completed successfully."
}
# --- End Helper Function ---


# --- Setup ---
# The mkdir -p commands will now use absolute paths for SAVE_DIR and GEN_OUTPUT_DIR
# slurm_logs is still relative here; SBATCH --output handles its main log file path.
mkdir -p ./slurm_logs # This creates slurm_logs relative to the job's CWD
mkdir -p ${SAVE_DIR}
mkdir -p ${GEN_OUTPUT_DIR}
echo "Log and output directories ensured."
echo "DATA_ROOT_PROCESSED set to: ${DATA_ROOT_PROCESSED}"
echo "SAVE_DIR set to: ${SAVE_DIR}"
echo "GEN_OUTPUT_DIR set to: ${GEN_OUTPUT_DIR}"

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

srun python -u ${TRAIN_SCRIPT} \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root "${DATA_ROOT_PROCESSED}" \
    --dataset_name "${DATASET_NAME}" \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --save_interval ${SAVE_INTERVAL} \
    --save_dir "${SAVE_DIR}" \
    --edge_unroll ${EDGE_UNROLL} \
    --grad_clip_value ${GRAD_CLIP} \
    --num_flow_layer ${NUM_FLOW_LAYER} \
    --num_rgcn_layer ${NUM_RGCN_LAYER} \
    --gaf_nhid ${GAF_NHID} \
    --gaf_nout ${GAF_NOUT}


# === Step 2: Generation ===
echo ""; echo "========================================"; echo "Starting Generation: ${MODEL_NAME}"; echo "========================================"
# CHECKPOINT_PATH will be absolute because SAVE_DIR is now absolute
CHECKPOINT_FILENAME="${MODEL_NAME,,}_ckpt_epoch_${MAX_EPOCHS}.pth"
CHECKPOINT_PATH="${SAVE_DIR}/${CHECKPOINT_FILENAME}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Warning: Final checkpoint '${CHECKPOINT_PATH}' not found."
    LATEST_CHECKPOINT=$(ls -t "${SAVE_DIR}"/${MODEL_NAME,,}_ckpt_epoch_*.pth 2>/dev/null | head -n 1)
    if [ -f "${LATEST_CHECKPOINT}" ]; then
        echo "Warning: Using latest found checkpoint instead: ${LATEST_CHECKPOINT}"
        CHECKPOINT_PATH=${LATEST_CHECKPOINT}
    else
        echo "Error: No checkpoints found in ${SAVE_DIR}. Cannot generate."
        # exit 1 # Commenting out exit 1 to allow evaluation step to potentially run if needed for debugging
                   # or if generation is optional. Re-enable if training must succeed for generation.
    fi
fi

# Proceed with generation only if a checkpoint is found
if [ -f "${CHECKPOINT_PATH}" ]; then
    echo " - Using Checkpoint: ${CHECKPOINT_PATH}"
    echo " - Number of Samples: ${NUM_SAMPLES}"
    echo " - Temperature: ${TEMPERATURE_AF}"
    echo " - Min Nodes: ${MIN_NODES}"
    echo " - Output File: ${GEN_PICKLE_PATH}"
    echo "----------------------------------------"

    srun python -u ${GEN_SCRIPT} \
        --model ${MODEL_NAME} \
        --checkpoint "${CHECKPOINT_PATH}" \
        --output_file "${GEN_PICKLE_PATH}" \
        --num_samples ${NUM_SAMPLES} \
        --temperature_af ${TEMPERATURE_AF} \
        --min_nodes ${MIN_NODES} \
        --device ${REQUESTED_DEVICE} \
        --edge_unroll ${EDGE_UNROLL} \
        --max_size ${MAX_SIZE} \
        --node_dim ${NODE_DIM} \
        --bond_dim ${BOND_DIM} \
        --num_flow_layer ${NUM_FLOW_LAYER} \
        --num_rgcn_layer ${NUM_RGCN_LAYER} \
        --gaf_nhid ${GAF_NHID} \
        --gaf_nout ${GAF_NOUT}
else
    echo "Skipping generation step as no valid checkpoint was found."
fi


# === Step 3: Evaluation ===
echo ""; echo "========================================"; echo "Starting Evaluation"; echo "========================================"
# Evaluate only if the generated file exists
if [ -f "${GEN_PICKLE_PATH}" ]; then
    echo " - Evaluating File: ${GEN_PICKLE_PATH}"
    echo " - Training Data Dir (for Novelty): ${TRAIN_DATA_DIR_FOR_NOVELTY}"
    echo "----------------------------------------"

    srun python -u ${EVAL_SCRIPT} \
        "${GEN_PICKLE_PATH}" \
        --train_data_dir "${TRAIN_DATA_DIR_FOR_NOVELTY}"
else
    echo "Skipping evaluation step as generated pickle file '${GEN_PICKLE_PATH}' not found."
fi

echo "Job finished."
