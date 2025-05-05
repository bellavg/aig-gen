# G2PT/configs/train.py

# --- Data Settings ---
# Dataset name - this will likely be overridden by command-line args or configurator
# Set a default or leave as None if it *must* be specified externally.
dataset = 'aig'
# Number of data sequences generated per graph (if applicable to dataset)
num_augmentations = 6
#TODO check augment number

# Sequence ordering method (e.g., 'bfs', 'dfs') - specific to your data generation
ordering = 'topo'

# --- I/O Settings ---
# Base directory for saving outputs (logs, checkpoints)
# This will often be combined with run-specific names.
out_dir_base = 'results'
# Interval (in iterations) for running evaluation
eval_interval = 1000
# Interval (in iterations) for logging training progress
log_interval = 10
# Number of iterations to run for evaluation to estimate loss
eval_iters = 200
# If True, save a checkpoint after every evaluation, not just on improvement
always_save_checkpoint = False
# Initialization mode: 'scratch', 'resume', or 'gpt2*' (for pretrained weights)
init_from = 'scratch'

# --- Weights & Biases Logging ---
# Enable wandb logging if True
wandb_log = True
# Wandb project name
wandb_project = 'real-g2pt'
# Default Wandb run name (can be None, will be constructed later if logging)
# Example format: f"{dataset}-{model_name}-{ordering}-{num_augmentations}"
wandb_run_name = None

# --- Training Hyperparameters ---
# Gradient accumulation steps to simulate larger batch sizes
#TODO think about this?

# total batch size = gradient_accumulation_steps * ddp_world_size * batch_size
gradient_accumulation_steps = 15
# Micro-batch size (batch size per GPU per forward/backward pass)
batch_size = 32
# Total number of training iterations
max_iters = 60000
# Early stopping patience (number of evaluations without improvement before stopping)
# Set to 0 or a very large number to disable early stopping.
patience = 7

# --- Optimizer Settings (AdamW) ---
# Maximum learning rate
learning_rate = 5e-5
# Weight decay
weight_decay = 0.1
# AdamW beta1 parameter
beta1 = 0.9
# AdamW beta2 parameter
beta2 = 0.95
# Gradient clipping value (0.0 to disable)
grad_clip = 1.0

# --- Learning Rate Decay Settings ---
# Enable learning rate decay if True
decay_lr = True
# Number of warmup iterations where LR increases linearly
warmup_iters = 2000
# Number of iterations over which LR decays (typically max_iters or slightly less)
lr_decay_iters = 50000
# Minimum learning rate after decay
min_lr = 1e-5

# --- DDP Settings ---
# Backend for Distributed Data Parallel ('nccl', 'gloo', etc.)
backend = 'nccl'

# --- System Settings ---
# Compile model with PyTorch 2.0 for potential speedup (requires PyTorch 2.0+)
compile = False
# Number of worker processes for DataLoader
# Set to 0 for debugging or if issues arise. Can be increased for performance.
# Often set dynamically based on DDP status in the main script.
num_loader_workers = 0 # Default to 0, will be updated in train script if DDP

