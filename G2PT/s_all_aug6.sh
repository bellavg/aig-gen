#!/bin/bash
#SBATCH --job-name=aig_sample
#SBATCH --partition=gpu_h100     # Specify the appropriate partition
#SBATCH --gpus=1                 # Requesting one GPU
#SBATCH --time=08:00:00          # Adjust time as needed (sampling can be faster than training)
#SBATCH --output=./slurm_logs/aig_sample_exp_%j.out

# --- Configuration ---
# !!! IMPORTANT: Set these paths correctly !!!
# Path to the specific checkpoint file (e.g., best.pt or ckpt.pt)
MODEL_CHECKPOINT_PATH="./results/aig-base-topo-aug6/best.pt"
# Path to the AIG tokenizer directory
TOKENIZER_PATH="./datasets/aig/tokenizer"
# Base directory where the output .pkl files will be saved (within this script's execution dir)
# The script will save files like 'sampling_outputs/generated_aigs_beam_normal_k5.pkl'
BASE_OUTPUT_DIR="sampling_outputs"
# General sampling parameters
NUM_SAMPLES=1000 # Number of sequences to generate per configuration
MAX_NEW_TOKENS=768
# --- End Configuration ---

# Navigate to the base directory (G2PT/) assuming the script is in a 'scripts' subdir
cd "$(dirname "$0")/.." # Go up one level from the script's directory

# Ensure output directories exist
mkdir -p slurm_logs
mkdir -p $BASE_OUTPUT_DIR

# Load necessary modules (adjust based on your HPC environment)
echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1

# Activate Conda environment
echo "Activating Conda environment..."
source activate g2pt-aig # Use the correct environment name

# Check if checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT_PATH" ]; then
    echo "ERROR: Model checkpoint not found at $MODEL_CHECKPOINT_PATH"
    exit 1
fi

# Check if tokenizer exists
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer directory not found at $TOKENIZER_PATH"
    exit 1
fi

echo "Starting AIG sampling experiments..."
echo "Model Checkpoint: $MODEL_CHECKPOINT_PATH"
echo "Tokenizer Path: $TOKENIZER_PATH"
echo "Output Directory: $BASE_OUTPUT_DIR"
echo "Samples per run: $NUM_SAMPLES"

# --- Experiment Runs ---

# 1. Standard Beam Search (k=5)
echo "--- Running: Standard Beam Search (k=5) ---"
srun python -u beam_search_sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --search_type normal \
    --beam_size 5 \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_filename "generated_aigs_beam.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR # Save inside the base output dir

# 2. Beam Search + Sampling (k=5, T=0.6)
echo "--- Running: Beam Search + Sampling (k=5, T=0.6) ---"
srun python -u beam_search_sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --search_type normal \
    --beam_size 5 \
    --do_sample_beam \
    --temperature 0.6 \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_filename "generated_aigs_beam.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR

# 3. Beam Search + Sampling (k=5, T=1.0)
echo "--- Running: Beam Search + Sampling (k=5, T=1.0) ---"
srun python -u beam_search_sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --search_type normal \
    --beam_size 5 \
    --do_sample_beam \
    --temperature 1.0 \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_filename "generated_aigs_beam.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR

# 4. Diverse Beam Search (k=6, g=3, p=1.0)
echo "--- Running: Diverse Beam Search (k=6, g=3, p=1.0) ---"
srun python -u beam_search_sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --search_type normal \
    --beam_size 6 \
    --num_beam_groups 3 \
    --diversity_penalty 1.0 \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_filename "generated_aigs_beam.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR



# --- MODIFIED: Add Multinomial Sampling (from sample.py) ---
echo "--- Running: Multinomial Sampling (T=0.8) ---"
srun python -u sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --num_samples $NUM_SAMPLES \
    --temperature 0.8 \
    --output_filename "generated_aigs_multinomial_t0.8.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR # sample.py also saves inside --out_dir

# --- MODIFIED: Add Multinomial Sampling (from sample.py) ---
echo "--- Running: Multinomial Sampling (T=0.6) ---"
srun python -u sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --num_samples $NUM_SAMPLES \
    --temperature 0.6 \
    --output_filename "generated_aigs_multinomial_t0.6.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR # sample.py also saves inside --out_dir


# 5. Constrained Beam Search (k=5, T=0.6)
echo "--- Running: Constrained Beam Search (k=5, T=0.6) ---"
srun python -u beam_search_sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --search_type constrained \
    --beam_size 5 \
    --temperature 0.6 \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_filename "generated_aigs_beam.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR


echo "--- All Sampling Experiments Finished ---"

