#!/bin/bash
#SBATCH --job-name=graphebm_retrain_pipe # New job name
#SBATCH --partition=gpu_h100 # Or your specific GPU partition
#SBATCH --gpus=1
#SBATCH --time=24:00:00 # Increased time for full train + sample + eval
#SBATCH --output=./slurm_logs/graphebm_aig_retrain_pipe_%j.out

# --- Configuration ---
MODEL_NAME="GraphEBM"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name
# *** Updated Data Paths ***
DATA_ROOT="./aigs_pyg/"       # Root directory containing the 'aig' folder
DATASET_NAME="aig"          # Dataset subfolder name (contains 'processed')
TRAIN_DATA_DIR="./data/aig" # Base directory for training data (for novelty eval)
REQUESTED_DEVICE="cuda"

# Training Hyperparameters (Using stable settings)
LR=5e-5
EBM_ALPHA=1.0
EBM_LD_STEP_SIZE=0.01
EBM_LD_NOISE=0.005
EBM_LD_STEP=100
EBM_C=0.01
EBM_CLAMP_LGD_GRAD="--ebm_clamp_lgd_grad" # Pass the flag to enable
WEIGHT_DECAY=1e-5
BATCH_SIZE=64
MAX_EPOCHS=100 # Number of epochs to train for
SAVE_INTERVAL=10 # Save checkpoint frequency
GRAD_CLIP_VALUE=1.0
# *** Added Edge Unroll (needed by train_graphs.py model config setup) ***
EDGE_UNROLL=15 # Default, adjust if a specific structure is intended/required
# *** Added Num Augmentations (needed by train_graphs.py dataset loading) ***
NUM_AUGMENTATIONS=5 # Default, adjust to match your dataset processing

# Generation Parameters
NUM_SAMPLES=1000
NUM_MIN_NODES=5 # Minimum nodes for generated graphs
# *** Updated Save Dir based on DATASET_NAME ***
CKPT_SAVE_DIR="${MODEL_NAME}/rand_gen_${DATASET_NAME}_ckpts" # Directory where checkpoints are saved by train_graphs.py
# Use the checkpoint saved at the *end* of the training run
FINAL_CKPT_PATH="${CKPT_SAVE_DIR}/epoch_${MAX_EPOCHS}.pt"
GEN_OUTPUT_DIR="./generated_graphs" # Separate output dir
GEN_PICKLE_FILENAME="${MODEL_NAME}_retrained_generated_aigs_${NUM_SAMPLES}_epoch${MAX_EPOCHS}_${SLURM_JOB_ID}.pkl" # Added epoch
GEN_PICKLE_PATH="${GEN_OUTPUT_DIR}/${GEN_PICKLE_FILENAME}"

# --- Script Names ---
TRAIN_SCRIPT="train_graphs.py" # Simplified script that delegates training
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
    tail -n 50 "./slurm_logs/graphebm_aig_retrain_pipe_${SLURM_JOB_ID}.out" # Match output file name
    exit $exit_code
  fi
  echo "Step '${step_name}' completed successfully."
}
# --- End Helper Function ---


# --- Setup ---
mkdir -p slurm_logs
mkdir -p ${GEN_OUTPUT_DIR} # Ensure generation output dir exists
# Note: CKPT_SAVE_DIR will be created by train_graphs.py or its delegated runner method
echo "Log and output directories ensured: ./slurm_logs, ${GEN_OUTPUT_DIR}"
# Clean previous checkpoints for a true scratch run? Optional.
# echo "Removing previous checkpoints in ${CKPT_SAVE_DIR}..."
# rm -rf "${CKPT_SAVE_DIR}" # Use with caution!

echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1
echo "Modules loaded."

echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
check_exit_code $? "Activate Conda Env"
echo "Conda environment activated."
# --- End Setup ---


# === STEP 1: Training ===
echo "--- Starting Step 1: Training ${MODEL_NAME} from scratch ---"
# Note: Save directory creation is handled within train_graphs.py or the runner

srun python -u ${TRAIN_SCRIPT} \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root ${DATA_ROOT} \
    `# *** Added required arguments ***` \
    --dataset_name ${DATASET_NAME} \
    --edge_unroll ${EDGE_UNROLL} \
    --save_dir "${CKPT_SAVE_DIR}" \
    --num_augmentations ${NUM_AUGMENTATIONS} \
    `# --- Training Hyperparameters ---` \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --save_interval ${SAVE_INTERVAL} \
    --grad_clip_value ${GRAD_CLIP_VALUE} \
    `# --- EBM Specific Training Hyperparameters ---` \
    --ebm_alpha ${EBM_ALPHA} \
    --ebm_ld_step_size ${EBM_LD_STEP_SIZE} \
    --ebm_ld_noise ${EBM_LD_NOISE} \
    --ebm_ld_step ${EBM_LD_STEP} \
    --ebm_c ${EBM_C} \
    ${EBM_CLAMP_LGD_GRAD} \
    `# --- Optional: Pass EBM architecture params if different from defaults ---` \
    # --ebm_hidden 64 --ebm_depth 2 ...

check_exit_code $? "Training"
echo "--- Finished Step 1: Training ---"


# === STEP 2: Generation ===
echo "--- Starting Step 2: Generating ${NUM_SAMPLES} Samples using checkpoint ${FINAL_CKPT_PATH} ---"
if [ ! -f "${FINAL_CKPT_PATH}" ]; then
    echo "Error: Expected checkpoint file not found after training: ${FINAL_CKPT_PATH}"
    echo "Check if MAX_EPOCHS (${MAX_EPOCHS}) matches the final saved epoch or if training failed."
    exit 1
fi

# Using corrected argument names for sample_graphs.py
srun python -u ${GEN_SCRIPT} \
    --model ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --checkpoint "${FINAL_CKPT_PATH}" \
    --num_samples ${NUM_SAMPLES} \
    --output_file "${GEN_PICKLE_PATH}" \
    --min_nodes ${NUM_MIN_NODES} \
    `# *** Added necessary ARCHITECTURE args for model instantiation ***` \
    `# These MUST match the parameters used during training!` \
    --edge_unroll ${EDGE_UNROLL} \
    `# Add max_size, node_dim, bond_dim if sample_graphs.py needs them` \
    # --max_size 64 --node_dim 4 --bond_dim 3 \
    `# Add EBM architecture params` \
    # --ebm_hidden 64 --ebm_depth 2 ... \
    `# --- Pass necessary EBM GENERATION params ---` \
    --ebm_c ${EBM_C} \
    --ebm_ld_step ${EBM_LD_STEP} \
    --ebm_ld_noise ${EBM_LD_NOISE} \
    --ebm_ld_step_size ${EBM_LD_STEP_SIZE} \
    ${EBM_CLAMP_LGD_GRAD}

check_exit_code $? "Generation"
echo "--- Finished Step 2: Generation (Output: ${GEN_PICKLE_PATH}) ---"


# === STEP 3: Evaluation ===
echo "--- Starting Step 3: Evaluating Generated Samples ---"
if [ ! -f "${GEN_PICKLE_PATH}" ]; then
    echo "Error: Expected generated pickle file not found: ${GEN_OUTPUT_PKL}"
    exit 1
fi

# The first argument to evaluate_aigs.py is positional
srun python -u ${EVAL_SCRIPT} \
    "${GEN_PICKLE_PATH}" \
    --train_data_dir "${TRAIN_DATA_DIR}" \
    `# Add other necessary args for evaluate_aigs.py if needed`

check_exit_code $? "Evaluation"
echo "--- Finished Step 3: Evaluation ---"


echo "--- Full Pipeline Completed Successfully ---"
