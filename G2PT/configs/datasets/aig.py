# File: G2PT/configs/datasets/aig.py

# --- AIG Dataset Configuration ---

# Dataset name (used to identify which data/tokenizer to load)
dataset = 'aig'

# Vocabulary Size
# Calculated from vocab.json (0-102 = 103 tokens) + added special tokens (UNK, PAD, MASK = 3 tokens) = 106
# Rounding up slightly (e.g., to 112) is also common. Let's use 112 for potential padding/efficiency.
vocab_size = 112 # Or use 106 if you prefer the exact count

# Block Size / Max Sequence Length
# Should be >= the maximum sequence length in your prepared AIG dataset.
# We used model_max_length=1024 in tokenizer_config.json as a placeholder.
# VERIFY THIS against your actual data from prepare_aig.py output (e.g., data_meta.json shapes).
block_size = 1024 # ADJUST AS NEEDED

# --- Paths (These might be automatically inferred by the main script, but good to define) ---

# Path to the directory containing the processed .bin files (relative to project root)
# Matches the output_dir used in prepare_aig.py
data_dir = '../../datasets/aig/'

# Path to the tokenizer directory (relative to project root)
tokenizer_path = '../../tokenizers/aig/'

# Add any other dataset-specific flags or settings if needed by your setup
# Example:
# pin_memory = False # If relevant for your dataloader

print(f"Loaded AIG dataset configuration: vocab_size={vocab_size}, block_size={block_size}")