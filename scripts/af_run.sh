#!/bin/bash
#SBATCH --job-name=af # Updated job name
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # Keep increased time
#SBATCH --output=../slurm_logs/sample_af_%j.out


# --- Configuration ---
MODEL_NAME="GraphAF"
REQUESTED_DEVICE="cuda"
CONDA_ENV_NAME="g2pt-aig" # Your Conda environment name

# --- Script Names ---
# *** Ensure these point to the CORRECT versions of your scripts ***
TRAIN_SCRIPT="train_graphs.py" # The simplified script that delegates training

# --- End Configuration ---

cd ..

# --- Setup ---
mkdir -p slurm_logs


echo "Log and output directories ensured."
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1
echo "Modules loaded."
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate ${CONDA_ENV_NAME}
echo "Conda environment activated."
# --- End Setup ---

#
### Call the simplified train_graphs.py script, ensuring all required args are present
#srun python -u ggraph/train_graphs.py \
#    --model 'GraphAF'
#


srun python -u ggraph/sample_graphs.py \
    --model 'GraphAF' \
    --checkpoint "./ggraph/checkpoints/GraphAF/rand_gen_ckpt_24.pth" \
    --evaluate \
    --save


#
## === Step 2: Generation ===
#echo ""; echo "========================================"; echo "Starting Generation: ${MODEL_NAME}"; echo "========================================"
#CHECKPOINT_FILENAME="${MODEL_NAME,,}_ckpt_epoch_${MAX_EPOCHS}.pth" # Use lowercase model name
#CHECKPOINT_PATH="${SAVE_DIR}/${CHECKPOINT_FILENAME}"
## Checkpoint finding logic
#if [ ! -f "${CHECKPOINT_PATH}" ]; then echo "Warning: Final checkpoint '${CHECKPOINT_PATH}' not found." ; LATEST_CHECKPOINT=$(ls -t "${SAVE_DIR}"/${MODEL_NAME,,}_ckpt_epoch_*.pth 2>/dev/null | head -n 1); if [ -f "${LATEST_CHECKPOINT}" ]; then echo "Warning: Using latest found checkpoint instead: ${LATEST_CHECKPOINT}"; CHECKPOINT_PATH=${LATEST_CHECKPOINT}; else echo "Error: No checkpoints found in ${SAVE_DIR}. Cannot generate."; exit 1; fi; fi
#
#echo " - Using Checkpoint: ${CHECKPOINT_PATH}"
#echo " - Number of Samples: ${NUM_SAMPLES}"
#echo " - Temperature (Node, Edge): ${TEMP_NODE}, ${TEMP_EDGE}" # GraphDF uses two temps
#echo " - Min Nodes: ${MIN_NODES}"
#echo " - Output File: ${GEN_PICKLE_PATH}"
#echo "----------------------------------------"

# Execute the generation script
# *** Use --temperature_df for GraphDF ***

#
#
## Execute the evaluation script
#srun python -u ${EVAL_SCRIPT} \
#    "./GraphDF/generated_graphs.pkl"


