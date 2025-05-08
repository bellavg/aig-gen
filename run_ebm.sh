#!/bin/bash
#SBATCH --job-name=graphebm_retrain_pipe # New job name
#SBATCH --partition=gpu_h100 # Or your specific GPU partition
#SBATCH --gpus=1
#SBATCH --time=24:00:00 # Increased time for full train + sample + eval
#SBATCH --output=./slurm_logs/graphebm_aig_retrain_pipe_%j.out

# --- Configuration ---
MODEL_NAME="GraphEBM"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name
DATA_ROOT="./data/"       # Root directory for processed data (e.g., data/aig/processed/train)
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

# Generation Parameters
NUM_SAMPLES=1000
NUM_MIN_NODES=5 # Minimum nodes for generated graphs
CKPT_SAVE_DIR="${MODEL_NAME}/rand_gen_aig_ckpts" # Directory where checkpoints are saved by train_graphs.py
# Use the checkpoint saved at the *end* of the training run
FINAL_CKPT_PATH="${CKPT_SAVE_DIR}/epoch_${MAX_EPOCHS}.pt"
GEN_OUTPUT_PKL="${MODEL_NAME}_retrained_generated_aigs_${NUM_SAMPLES}.pkl" # New output name

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
echo "Log directory ensured: ./slurm_logs"
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
# Ensure the save directory exists before training starts
mkdir -p "${CKPT_SAVE_DIR}"

srun python -u train_graphs.py \
    --model_type ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --data_root ${DATA_ROOT} \
    --lr ${LR} \
    --ebm_alpha ${EBM_ALPHA} \
    --ebm_ld_step_size ${EBM_LD_STEP_SIZE} \
    --ebm_ld_noise ${EBM_LD_NOISE} \
    --ebm_ld_step ${EBM_LD_STEP} \
    --ebm_c ${EBM_C} \
    ${EBM_CLAMP_LGD_GRAD} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --save_interval ${SAVE_INTERVAL} \
    --grad_clip_value ${GRAD_CLIP_VALUE} \
    `# Pass --save_dir explicitly to train_graphs.py if it accepts it` \
    `# --save_dir "${CKPT_SAVE_DIR}" ` \
    `# Add other necessary args for train_graphs.py if needed (e.g., --ebm_hidden)`

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
srun python -u sample_graphs.py \
    --model ${MODEL_NAME} \
    --device ${REQUESTED_DEVICE} \
    --checkpoint "${FINAL_CKPT_PATH}" \
    --num_samples ${NUM_SAMPLES} \
    --output_file "${GEN_OUTPUT_PKL}" \
    --min_nodes ${NUM_MIN_NODES} \
    `# Pass necessary EBM generation params matching sample_graphs.py parser` \
    --ebm_c ${EBM_C} \
    --ebm_ld_step ${EBM_LD_STEP} \
    --ebm_ld_noise ${EBM_LD_NOISE} \
    --ebm_ld_step_size ${EBM_LD_STEP_SIZE} \
    ${EBM_CLAMP_LGD_GRAD} \
    `# Add other necessary args for sample_graphs.py if needed (e.g., --ebm_hidden)`

check_exit_code $? "Generation"
echo "--- Finished Step 2: Generation (Output: ${GEN_OUTPUT_PKL}) ---"


# === STEP 3: Evaluation ===
echo "--- Starting Step 3: Evaluating Generated Samples ---"
if [ ! -f "${GEN_OUTPUT_PKL}" ]; then
    echo "Error: Expected generated pickle file not found: ${GEN_OUTPUT_PKL}"
    exit 1
fi

# The first argument to evaluate_aigs.py is positional
srun python -u evaluate_aigs.py \
    "${GEN_OUTPUT_PKL}" \
    --train_data_dir "${TRAIN_DATA_DIR}" \
    `# Add other necessary args for evaluate_aigs.py if needed`

check_exit_code $? "Evaluation"
echo "--- Finished Step 3: Evaluation ---"


echo "--- Full Pipeline Completed Successfully ---"

