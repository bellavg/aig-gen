#!/bin/bash
#SBATCH --job-name=aig_sample_1k # Changed to reflect 1k samples aim
#SBATCH --partition=gpu_h100     # Specify the appropriate partition
#SBATCH --gpus=1                 # Requesting one GPU
#SBATCH --time=08:00:00          # Adjust time as needed
#SBATCH --output=./slurm_logs/aig_sample_1k_%j.out # Updated log name

# --- Configuration ---
MODEL_CHECKPOINT_PATH="results/aig-base-topo-aug6/best.pt"
TOKENIZER_PATH="datasets/aig/tokenizer"
BASE_OUTPUT_DIR="sampling_outputs_6" # Consider a new output dir for 1k experiments
NUM_SAMPLES=1000
MAX_NEW_TOKENS=768
# --- End Configuration ---

mkdir -p slurm_logs
mkdir -p $BASE_OUTPUT_DIR

echo "Loading modules..."
module load 2024
module load Anaconda3/2024.06-1

echo "Activating Conda environment..."
source activate g2pt-aig

if [ ! -f "$MODEL_CHECKPOINT_PATH" ]; then
    echo "ERROR: Model checkpoint not found at $MODEL_CHECKPOINT_PATH"
    exit 1
fi
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer directory not found at $TOKENIZER_PATH"
    exit 1
fi

echo "Starting AIG sampling experiments (Target: $NUM_SAMPLES samples each)..."
echo "Model Checkpoint: $MODEL_CHECKPOINT_PATH"
echo "Tokenizer Path: $TOKENIZER_PATH"
echo "Output Directory: $BASE_OUTPUT_DIR"

# 1. Standard Beam Search (Python script uses k=1000 internally)
echo "--- Running: Standard Beam Search (Effective k=1000) ---"
srun python -u beam_search_sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --search_type normal \
    --beam_size 1000 \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_filename "generated_aigs_beam.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR

# 2. Beam Search + Sampling (k=5 per call, T=0.6)
echo "--- Running: Beam Search + Sampling (k=5 per call, T=0.6) ---"
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

# 3. Beam Search + Sampling (k=5 per call, T=1.0)
echo "--- Running: Beam Search + Sampling (k=5 per call, T=1.0) ---"
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

# 4. Diverse Beam Search (Python script uses k=1000 internally, e.g., g=20)
#    The --beam_size here is mainly for filename if Python didnt override,
#    but Python's effective_beam_size_for_run will be 1000 for diverse.
echo "--- Running: Diverse Beam Search (Effective k=1000, g=20, p=1.0) ---"
srun python -u beam_search_sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --search_type normal \
    --beam_size 1000 \
    --num_beam_groups 20 \
    --diversity_penalty 1.0 \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_filename "generated_aigs_beam.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR


# --- Multinomial Sampling (using sample.py script) ---
echo "--- Running: Multinomial Sampling (T=0.8) ---"
srun python -u sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --num_samples $NUM_SAMPLES \
    --temperature 0.8 \
    --output_filename "generated_aigs_multinomial_t0.8.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR

echo "--- Running: Multinomial Sampling (T=0.6) ---"
srun python -u sample.py \
    --input_checkpoint $MODEL_CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --num_samples $NUM_SAMPLES \
    --temperature 0.6 \
    --output_filename "generated_aigs_multinomial_t0.6.pkl" \
    --parsing_mode robust \
    --out_dir $BASE_OUTPUT_DIR


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