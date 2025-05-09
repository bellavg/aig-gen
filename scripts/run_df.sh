#!/bin/bash
#SBATCH --job-name=df # Updated job name
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # Keep increased time
#SBATCH --output=../slurm_logs/df_%j.out


# --- Configuration ---
MODEL_NAME="GraphDF"
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
check_exit_code $? "Activate Conda Env"
echo "Conda environment activated."
# --- End Setup ---


## Call the simplified train_graphs.py script, ensuring all required args are present
srun python -u ggraph/train_graphs.py

#    --model_type ${MODEL_NAME} \
#    --device ${REQUESTED_DEVICE} \
#    --data_root ${DATA_ROOT_PROCESSED} \
#    --dataset_name ${DATASET_NAME} \
#    --lr ${LR} \
#    --weight_decay ${WEIGHT_DECAY} \
#    --batch_size ${BATCH_SIZE} \
#    --max_epochs ${MAX_EPOCHS} \
#    --save_interval ${SAVE_INTERVAL} \
#    --save_dir ${SAVE_DIR} \
#    --edge_unroll ${EDGE_UNROLL} \
#    --grad_clip_value ${GRAD_CLIP} \
#    --num_flow_layer ${NUM_FLOW_LAYER} \
#    --num_rgcn_layer ${NUM_RGCN_LAYER} \
#    --gaf_nhid ${GAF_NHID} \
#    --gaf_nout ${GAF_NOUT} \
#    # Add --st_type, --deq_coeff if needed by train_graphs.py


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
#srun python -u sample_df.py \
#    --checkpoint "./GraphDF/rand_gen_aig_ckpts_tuned_v1/graphdf_rand_gen_ckpt_epoch_20.pth" \
#    --output_file "./GraphDF/generated_graphs.pkl"
#
#
## Execute the evaluation script
#srun python -u ${EVAL_SCRIPT} \
#    "./GraphDF/generated_graphs.pkl"


