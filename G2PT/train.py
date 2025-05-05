# G2PT/train.py
import os
import time
import math
from contextlib import nullcontext
import sys # For potentially adding config paths
import logging # Use logging
from typing import List, Optional # For type hints
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, default_data_collator
import networkx as nx # For type hint

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("G2PT_Trainer")

# --- Project Imports ---
import G2PT.configs.aig as aig_cfg
import G2PT.configs.base as net_cfg
import G2PT.configs.train_aig as train_cfg
from G2PT.datasets_utils import get_datasets, seq_to_nxgraph
from G2PT.model import GPT, GPTConfig
from G2PT.evaluate_aigs import validate_aig_structures # Import the validation function
from G2PT.sample import generate_and_parse_aigs # Import the generation+parsing function

# --- Environment Setup ---
# Recommended for Dynamo + DDP, but might cause issues, monitor performance/errors
# torch._dynamo.config.optimize_ddp = False
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid tokenizer parallelism conflicts


# -----------------------------------------------------------------------------
# Configuration Loading & Overrides
# -----------------------------------------------------------------------------
# Start with base configurations loaded from files
config = {}
config.update({k: v for k, v in vars(aig_cfg).items() if not k.startswith('_')})
config.update({k: v for k, v in vars(net_cfg).items() if not k.startswith('_')})
config.update({k: v for k, v in vars(train_cfg).items() if not k.startswith('_')})

# Apply overrides from configurator.py (command line args)
# This modifies the 'config' dictionary directly.
try:
    # Make the config dict available for configurator.py to modify
    global_config_for_exec = config
    # Ensure configurator.py can be found relative to train.py or in PYTHONPATH
    configurator_path = os.path.join(os.path.dirname(__file__), 'configurator.py')
    if not os.path.exists(configurator_path):
         # Fallback if not in the same directory (e.g., running from project root)
         configurator_path = 'configurator.py'

    if os.path.exists(configurator_path):
        logger.info(f"Loading overrides from {configurator_path}")
        with open(configurator_path) as f:
            exec(f.read(), {'config': global_config_for_exec})
        config = global_config_for_exec # Get potentially modified config back
        logger.info("Configuration potentially overridden by configurator.py")
    else:
        logger.warning("configurator.py not found, using default configurations.")

except Exception as e:
    logger.error(f"Error executing configurator.py: {e}", exc_info=True)
    sys.exit(1)

# --- Extract final config values after potential overrides ---
# Training settings
num_augmentations = config['num_augmentations']
out_dir_base = config['out_dir_base']
eval_interval = config['eval_interval']
log_interval = config['log_interval']
eval_iters = config['eval_iters']
always_save_checkpoint = config['always_save_checkpoint']
init_from = config['init_from']
wandb_log = config['wandb_log']
wandb_project = config['wandb_project']
wandb_run_name_base = config['wandb_run_name'] # Base name
gradient_accumulation_steps = config['gradient_accumulation_steps']
batch_size = config['batch_size'] # Per GPU batch size
max_iters = config['max_iters']
patience = config['patience']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
beta1 = config['beta1']
beta2 = config['beta2']
grad_clip = config['grad_clip']
decay_lr = config['decay_lr']
warmup_iters = config['warmup_iters']
lr_decay_iters = config['lr_decay_iters']
min_lr = config['min_lr']
backend = config['backend'] # DDP backend
compile_model = config['compile'] # Renamed to avoid conflict
num_loader_workers_cfg = config['num_loader_workers']

# Data settings from aig_cfg (might be overridden in config dict)
dataset = config['dataset']
block_size = config['block_size']
vocab_size = config.get('vocab_size', None) # Allow override, default to None initially
ordering = config['ordering']
# Use path from aig_cfg, ensure it's accessible
tokenizer_base_path = os.path.join(os.path.dirname(__file__), aig_cfg.tokenizer_path)
# Fallback if train.py is not in G2PT/
if not os.path.exists(os.path.join(tokenizer_base_path)):
     tokenizer_base_path = aig_cfg.tokenizer_path # Assume it's relative to execution dir

# Network settings from net_cfg (might be overridden in config dict)
n_layer = config['n_layer']
n_head = config['n_head']
n_embd = config['n_embd']
dropout = config['dropout']
bias = config['bias']
model_name = config['model_name']

# --- NEW: Validity Check Configuration ---
# How often to run the validity check (e.g., 2 means every 2nd eval interval)
validity_check_interval_multiplier = config.get('validity_check_interval_multiplier', 2)
# Number of graphs to sample for validity check
validity_num_samples = config.get('validity_num_samples', 100)
# Max new tokens for generation during validity check (use a reasonable default)
validity_max_new_tokens = config.get('validity_max_new_tokens', block_size) # Default to model's block size
# Temperature for sampling during validity check
validity_temperature = config.get('validity_temperature', 0.8)
# Top-k for sampling during validity check (optional)
validity_top_k = config.get('validity_top_k', None)
# Batch size for generation during validity check (can be different from training batch size)
validity_gen_batch_size = config.get('validity_gen_batch_size', batch_size * 2) # Example: larger batch size for faster sampling
# Parsing mode for generated sequences
validity_parsing_mode = config.get('validity_parsing_mode', 'strict')
# Save checkpoint based on validity improvement?
save_on_validity_improve = config.get('save_on_validity_improve', True)
# --- End Validity Check Configuration ---

# -----------------------------------------------------------------------------

# --- DDP Setup ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # Check if already initialized (e.g., by torchrun)
    if not is_initialized():
        init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    # Adjust gradient accumulation steps for DDP if needed (ensure total batch size is reasonable)
    # Effective batch size = batch_size * ddp_world_size * gradient_accumulation_steps
    # No adjustment needed here, accumulation happens locally first.
    logger.info(f"DDP active. Rank {ddp_rank}/{ddp_world_size}. Local Rank {ddp_local_rank}. Device: {device}.")
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0 # Set rank to 0 for non-DDP runs
    # --- Non-DDP Device Detection ---
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.warning("MPS device selected. Performance and support may vary.")
    else:
        device = 'cpu'
    logger.info(f"DDP not active. Running on device: {device}")
# --- End DDP / Device Setup ---

# Construct dynamic names/paths (only after DDP setup for rank info)
if wandb_log and master_process:
    wandb_run_name = f"{wandb_run_name_base}-{dataset}-{model_name}-{ordering}-aug{num_augmentations}"
else:
    # Include rank in local run name if DDP is active but wandb is off
    rank_suffix = f"-rank{ddp_rank}" if ddp else ""
    wandb_run_name = f"{dataset}-{model_name}-{ordering}-aug{num_augmentations}-local{rank_suffix}"

# Output directory specific to this run
out_dir = os.path.join(out_dir_base, wandb_run_name)
logger.info(f"Output directory: {out_dir}")

# Calculate total tokens processed per optimizer step (global batch size)
global_batch_size = batch_size * ddp_world_size * gradient_accumulation_steps
tokens_per_iter = global_batch_size * block_size # Approximate, depends on actual sequence lengths
logger.info(f"Global Batch Size: {global_batch_size} sequences")
logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
logger.info(f"Tokens per Optimizer Step (approx): {tokens_per_iter:,}")


# --- Dtype and Autocast Setup ---
if 'cuda' in device:
    if torch.cuda.is_bf16_supported():
        final_dtype_str = 'bfloat16'
    else:
        final_dtype_str = 'float16'
    # Enable TF32 for matmuls and cuDNN if using CUDA
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda'
elif device == 'mps':
    final_dtype_str = 'float32' # MPS typically uses float32
    device_type = 'mps'
else: # CPU
    final_dtype_str = 'float32'
    device_type = 'cpu'

logger.info(f"Selected dtype: {final_dtype_str}")
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[final_dtype_str]
# Enable autocast only for CUDA/MPS, disable for CPU
autocast_enabled = device_type in ('cuda', 'mps')
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype, enabled=autocast_enabled)
logger.info(f"Autocast context: enabled={autocast_enabled}, device_type='{device_type}', dtype='{ptdtype}'")
# --- End Dtype / Autocast Setup ---

# Create output directory if it doesn't exist (only on master process)
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Seed setting (important for reproducibility)
torch.manual_seed(1337 + seed_offset)
np.random.seed(1337 + seed_offset) # Seed numpy if used elsewhere
if 'cuda' in device:
     torch.cuda.manual_seed_all(1337 + seed_offset) # Seed all GPUs

# --- Data Loading ---

logger.info(f"Loading tokenizer from: {tokenizer_base_path}")
try:
    # Use legacy=False if your tokenizer files support it
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_base_path)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Max length: {tokenizer.model_max_length}")
    # --- Vocab Size Check ---
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
         logger.warning(f"Config vocab size ({vocab_size}) does not match tokenizer's ({tokenizer.vocab_size}). Using tokenizer's size.")
         vocab_size = tokenizer.vocab_size # Prioritize tokenizer's actual size
    elif vocab_size is None:
         logger.info(f"Setting vocab_size from tokenizer: {tokenizer.vocab_size}")
         vocab_size = tokenizer.vocab_size # Set vocab size if not in config
except Exception as e:
    logger.error(f"Error loading tokenizer from {tokenizer_base_path}: {e}", exc_info=True)
    sys.exit(1)


logger.info(f"Loading datasets for '{dataset}' (ordering: {ordering}, augmentations: {num_augmentations})...")
try:
    train_dataset, eval_dataset = get_datasets(dataset, tokenizer, ordering, num_augmentations)
    logger.info(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")
except Exception as e:
     logger.error(f"Failed to load datasets: {e}", exc_info=True)
     sys.exit(1)

def data_collate_fn(features):
    """Collator that pads to the max length in the batch, not block_size."""
    # Use HuggingFace default collator first (handles dict structure, converts to tensors)
    features = default_data_collator(features)
    # Find max length based on attention mask in the current batch
    max_len_in_batch = int(features['attention_mask'].sum(-1).max())
    features = {
        k: (v[:, :max_len_in_batch] if v.ndim > 1 else v)
        for k, v in features.items()
    }
    return features


# Adjust num_workers based on DDP status and config
# Often 0 is safest for DDP to avoid potential issues, but can be > 0 if carefully managed
num_workers = num_loader_workers_cfg if not ddp else 0 # Default to 0 for DDP
logger.info(f"DataLoader num_workers: {num_workers}")

# Create DataLoaders
train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, drop_last=False) if ddp else None
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None), # Shuffle only if not using DDP sampler
    pin_memory=True, # May speed up CPU-to-GPU transfers if using CUDA
    drop_last=False, # Keep the last incomplete batch
    num_workers=num_workers,
    collate_fn=data_collate_fn # Use custom collator
)

# Eval loader sampler should not shuffle
eval_sampler = DistributedSampler(eval_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=False) if ddp else None
eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size, # Can potentially use a larger batch size for eval
    sampler=eval_sampler,
    shuffle=False, # No need to shuffle evaluation data
    pin_memory=True,
    drop_last=False,
    num_workers=num_workers,
    collate_fn=data_collate_fn # Use custom collator
)
# --- End Data Loading ---

# --- Model Initialization ---
iter_num = 0
best_val_loss = 1e9 # Initialize with a large value
best_val_validity = -1.0 # Initialize validity score (0.0 to 1.0)
eval_counter = 0 # Counter for total evaluations performed
evals_no_improve = 0 # Counter for early stopping (based on val_loss)

# Consolidate model arguments from network config and data config
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, # Use block_size from data config
    bias=bias,
    vocab_size=vocab_size, # Use vocab_size determined from tokenizer/data config
    dropout=dropout
)

if init_from == 'scratch':
    logger.info("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    logger.info(f"Model initialized with {model.get_num_params()/1e6:.2f}M parameters.")
elif init_from == 'resume':
    logger.info(f"Resuming training from {out_dir}")
    # Look for ckpt.pt or best.pt
    ckpt_path = None
    for fname in ['ckpt.pt', 'best.pt']:
         fpath = os.path.join(out_dir, fname)
         if os.path.exists(fpath):
              ckpt_path = fpath
              break
    if ckpt_path is None:
        logger.error(f"Checkpoint path (ckpt.pt or best.pt) does not exist in {out_dir}. Cannot resume.")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device) # Load to target device directly

    # Load model args from checkpoint, overriding config if necessary
    if 'model_args' in checkpoint:
        checkpoint_model_args = checkpoint['model_args']
    elif 'config' in checkpoint: # Handle older checkpoint format
         logger.warning("Loading model args from 'config' dict in checkpoint.")
         cfg = checkpoint['config']
         checkpoint_model_args = {
             'n_layer': cfg.get('n_layer'), 'n_head': cfg.get('n_head'), 'n_embd': cfg.get('n_embd'),
             'block_size': cfg.get('block_size'), 'bias': cfg.get('bias', False),
             'vocab_size': cfg.get('vocab_size'), 'dropout': cfg.get('dropout', 0.0)
         }
    else:
         logger.error("Cannot find 'model_args' or 'config' in checkpoint. Cannot resume.")
         sys.exit(1)


    # Force essential config attributes to match checkpoint values
    essential_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    for k in essential_keys:
        if k in model_args and k in checkpoint_model_args and model_args[k] != checkpoint_model_args[k]:
             logger.warning(f"Overriding config '{k}' ({model_args[k]}) with checkpoint value ({checkpoint_model_args[k]})")
        if k in checkpoint_model_args:
             model_args[k] = checkpoint_model_args[k] # Use checkpoint value
        elif k not in model_args:
             logger.error(f"Essential model argument '{k}' missing from both config and checkpoint!")
             sys.exit(1)


    # Create the model structure
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    logger.info(f"Model structure created with {model.get_num_params()/1e6:.2f}M parameters.")

    # Load the model state dict
    state_dict = checkpoint['model']
    # Fix potential prefix issues in state dict keys (DDP, compile)
    unwanted_prefixes = ['_orig_mod.', '_module.', 'module.']
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        original_k = k
        for prefix in unwanted_prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned_state_dict[k] = v

    # Load the state dict (allow missing/unexpected keys initially)
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading state_dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
    logger.info("Model state loaded.")

    # Load training state
    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', 1e9)
    best_val_validity = checkpoint.get('best_val_validity', -1.0) # Load best validity
    eval_counter = checkpoint.get('eval_counter', iter_num // eval_interval if eval_interval > 0 else 0) # Estimate eval counter
    evals_no_improve = checkpoint.get('evals_no_improve', 0) # Load patience counter
    logger.info(f"Resuming from iteration {iter_num}, best_val_loss: {best_val_loss:.4f}, best_val_validity: {best_val_validity:.3f}")

# TODO: Add handling for init_from = 'gpt2*' if needed (loading pretrained weights)
elif init_from != 'scratch':
     logger.error(f"Unknown init_from value: {init_from}. Use 'scratch' or 'resume'.")
     sys.exit(1)


# Crop model block size if requested config value is smaller than loaded model's
if model.config.block_size > block_size:
    logger.info(f"Cropping model block size from {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # Update args dict to reflect change
elif model.config.block_size < block_size:
     logger.warning(f"Model block size ({model.config.block_size}) is smaller than config ({block_size}). Using model's block size.")
     block_size = model.config.block_size # Use the actual model block size


model.to(device)
# --- End Model Initialization ---

# --- Optimizer and Scaler Setup ---
# Initialize GradScaler for mixed precision training (enabled only if dtype is float16)
scaler = torch.amp.GradScaler(enabled=(final_dtype_str == 'float16'))
logger.info(f"Gradient Scaler enabled: {scaler.is_enabled()}")

# Initialize optimizer
# Ensure model.configure_optimizers exists and works correctly
try:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    logger.info(f"Optimizer configured: {type(optimizer)}")
except AttributeError:
     logger.error("Model does not have a 'configure_optimizers' method. Cannot proceed.")
     # Fallback: Create a standard AdamW optimizer if method is missing
     # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
     # logger.warning("Using standard AdamW optimizer as fallback.")
     sys.exit(1)
except Exception as e:
     logger.error(f"Error configuring optimizer: {e}", exc_info=True)
     sys.exit(1)


if init_from == 'resume' and 'optimizer' in checkpoint:
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Optimizer state loaded from checkpoint.")
    except Exception as e:
        logger.error(f"Failed to load optimizer state: {e}. Starting with fresh optimizer state.", exc_info=True)

checkpoint = None # Free up memory after loading
# --- End Optimizer and Scaler Setup ---

# --- Model Compilation (Optional) ---
if compile_model:
    if hasattr(torch, 'compile'):
        logger.info("Compiling the model... (this may take a minute)")
        try:
            # Use a recommended mode if applicable, otherwise default
            # model = torch.compile(model, mode="reduce-overhead") # Example mode
            model = torch.compile(model)
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Continuing without compilation.", exc_info=True)
            compile_model = False # Disable if compilation fails
    else:
        logger.warning("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
        compile_model = False
# --- End Model Compilation ---

# --- DDP Wrapping ---
# Wrap the model *after* compilation and moving to device
if ddp:
    # find_unused_parameters can be needed if some outputs aren't used in loss
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False) # Set find_unused_parameters=True if getting errors
    logger.info("Model wrapped with DistributedDataParallel.")
# --- End DDP Wrapping ---

# --- Evaluation Function ---
@torch.no_grad()
def estimate_loss(data_loader, split_name):
    """ Estimates loss over a given data loader using eval_iters batches. """
    model.eval() # Set model to evaluation mode
    losses = torch.zeros(eval_iters, device=device) # Use tensor for potential GPU ops
    batches_processed = 0
    loader_iter = iter(data_loader)

    for k in range(eval_iters):
        try:
            data = next(loader_iter)
            # Prepare batch data (move to device)
            # Ensure labels are correctly shifted and use attention mask
            X = data['input_ids'][:, :-1].to(device, non_blocking=True) # Input context
            Y = data['labels'][:, 1:].to(device, non_blocking=True)     # Target tokens (shifted)
            # Use attention mask for loss calculation if model supports it
            # Assume model handles masking internally if Y_mask is not passed or needed
            Y_mask = data.get('attention_mask', None) # Get mask if present
            if Y_mask is not None:
                 Y_mask = Y_mask[:, 1:].to(device, non_blocking=True) # Mask for targets

            # Forward pass under autocast context
            with ctx:
                # Pass mask only if the model signature expects it
                # Adjust this call based on your model's forward method
                try:
                     logits, loss = model(X, targets=Y, target_mask=Y_mask) # Example signature
                except TypeError:
                     # Fallback if model doesn't accept target_mask
                     logger.debug("Model forward doesn't accept target_mask, calling without it.")
                     logits, loss = model(X, targets=Y)

            losses[k] = loss.item()
            batches_processed += 1
        except StopIteration:
            logger.warning(f"DataLoader for split '{split_name}' exhausted before reaching eval_iters ({eval_iters}). Processed {batches_processed} batches.")
            losses = losses[:batches_processed] # Use only the losses we got
            break
        except Exception as e:
             logger.error(f"Error during loss estimation for split '{split_name}', batch {k}: {e}", exc_info=True)
             # Decide how to handle errors: skip batch, return NaN, etc.
             losses[k] = float('nan') # Mark error for this batch


    model.train() # Set model back to training mode

    if batches_processed == 0:
        return float('nan') # Return NaN if no batches were processed
    else:
        # Filter out potential NaNs before calculating mean
        valid_losses = losses[~torch.isnan(losses)]
        if valid_losses.numel() == 0:
             return float('nan') # Return NaN if all batches resulted in errors
        else:
             return valid_losses.mean().item() # Calculate mean on device, then move to CPU
# --- End Evaluation Function ---


# --- NEW: Graph Validity Check Function ---
@torch.no_grad()
def check_graph_validity(model_to_check, tokenizer_for_check, device_for_check):
    """ Samples graphs using refactored function and checks their validity. """

    logger.info(f"\n--- Starting Validity Check ---")
    logger.info(f"Sampling {validity_num_samples} graphs...")
    logger.info(f"Generation params: temp={validity_temperature}, top_k={validity_top_k}, max_new_tokens={validity_max_new_tokens}, batch_size={validity_gen_batch_size}")

    model_to_check.eval() # Ensure model is in eval mode

    # Use the refactored generation and parsing function
    # Ensure the model passed is the raw model if DDP is used
    raw_model_for_check = model_to_check.module if isinstance(model_to_check, DDP) else model_to_check

    # Generate and parse graphs
    # Need to ensure the device passed is the correct one (e.g., cuda:0 for master)
    generated_graphs: List[nx.DiGraph] = generate_and_parse_aigs(
        model=raw_model_for_check,
        tokenizer=tokenizer_for_check,
        device=device_for_check, # Pass the device where the model resides
        num_samples=validity_num_samples,
        batch_size=validity_gen_batch_size,
        temperature=validity_temperature,
        top_k=validity_top_k,
        max_new_tokens=validity_max_new_tokens,
        parsing_mode=validity_parsing_mode,
        seed=1337 + iter_num # Use a seed based on iteration for consistency across checks
    )

    num_generated = len(generated_graphs)
    logger.info(f"Successfully generated and parsed {num_generated} graphs.")

    if num_generated == 0:
        logger.warning("No graphs were successfully parsed for validity check.")
        validity_score = 0.0 # Treat as 0% valid if none could be parsed
    else:
        # Validate the parsed graphs using the refactored validation function
        logger.info(f"Evaluating validity of {num_generated} generated graphs...")
        try:
            # This function returns the fraction (0.0 to 1.0) of valid graphs
            validity_score = validate_aig_structures(generated_graphs)
            logger.info(f"--- Validity Score: {validity_score:.4f} ---")
        except Exception as e:
            logger.error(f"Error during validity evaluation: {e}", exc_info=True)
            validity_score = -1.0 # Indicate error

    model_to_check.train() # Set model back to train mode (important!)
    return validity_score
# --- End Graph Validity Check Function ---


# --- Learning Rate Scheduler ---
def get_lr(it):
    """ Learning rate decay scheduler (cosine with warmup). """
    # 1) Linear warmup
    if it < warmup_iters:
        # Ensure warmup_iters is not zero to avoid division by zero
        return learning_rate * it / max(1, warmup_iters)
    # 2) Constant minimum LR after decay phase
    if lr_decay_iters <= 0 or it > lr_decay_iters: # Handle case where decay is disabled or finished
        return min_lr
    # 3) Cosine decay in between
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1, f"Decay ratio out of bounds: {decay_ratio} (it={it})"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (learning_rate - min_lr)
# --- End Learning Rate Scheduler ---

# --- Main Training Loop ---
if __name__ == '__main__':
    # --- Logging Setup ---
    if wandb_log and master_process:
        import wandb
        logger.info("Initializing wandb...")
        try:
            # Pass essential config parameters to wandb
            wandb_config = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
            wandb_config.update({ # Add specific calculated values if needed
                 'global_batch_size': global_batch_size,
                 'tokens_per_iter': tokens_per_iter,
            })
            wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)
            logger.info("Wandb initialized.")
        except Exception as e:
            logger.error(f"Error initializing wandb: {e}", exc_info=True)
            wandb_log = False # Disable logging if init fails
    # --- End Logging Setup ---

    t0 = time.time() # Start timer for the first iteration
    local_iter_num = 0 # Iterations run in this specific process since last resume/start
    # Get the underlying model if using DDP for saving/MFU estimation
    raw_model = model.module if ddp else model
    running_mfu = -1.0 # For tracking Model Flops Utilization (MFU)

    logger.info(f"\n--- Starting Training Loop ---")
    logger.info(f"Max iterations: {max_iters}")
    logger.info(f"Logging interval: {log_interval}")
    logger.info(f"Evaluation interval: {eval_interval}")
    logger.info(f"Initial learning rate: {learning_rate}")
    logger.info(f"Decay LR: {decay_lr}, Warmup Iters: {warmup_iters}, Decay Iters: {lr_decay_iters}, Min LR: {min_lr}")
    logger.info(f"Gradient Clipping: {grad_clip}")
    logger.info(f"Patience for early stopping: {patience}")
    logger.info(f"Saving checkpoints to: {out_dir}\n")
    logger.info(f"Validity Check: Enabled (every {validity_check_interval_multiplier} evals, {validity_num_samples} samples)")


    # Fetch the first batch to start the loop
    train_iter = iter(train_loader)
    current_data = next(train_iter)

    # Main loop structure using train_loader iterator
    while True:
        # Set epoch for distributed sampler (needed for shuffling each epoch)
        # Calculate approximate epoch based on iterations and loader length
        if ddp and train_sampler is not None:
             # Ensure len(train_loader) is not zero
             if len(train_loader) > 0:
                  current_epoch = iter_num // len(train_loader)
                  train_sampler.set_epoch(current_epoch)
             # else: handle empty loader case if necessary


        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- Evaluation and Checkpointing ---
        # Run evaluation *before* the training step for the current iteration number
        if iter_num % eval_interval == 0 and master_process:
            eval_counter += 1 # Increment evaluation counter
            logger.info(f"--- Starting Eval Iter {iter_num} (Eval #{eval_counter}) ---")
            # Estimate loss on train and val splits
            train_loss = estimate_loss(train_loader, 'train')
            val_loss = estimate_loss(eval_loader, 'val')
            logger.info(f"Eval Losses -> Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            current_validity = -1.0 # Default if check is skipped or fails
            run_validity_check = (
                validity_check_interval_multiplier > 0 and
                eval_counter % validity_check_interval_multiplier == 0
            )

            if run_validity_check:
                 # Run validity check using the current model state
                 # Pass the master process device (e.g., 'cuda:0' or 'cpu')
                 master_device = device if not ddp else f'cuda:{ddp_local_rank}' # Should be rank 0's device
                 current_validity = check_graph_validity(
                      model_to_check=model, # Pass the potentially DDP-wrapped model
                      tokenizer_for_check=tokenizer,
                      device_for_check=master_device
                 )
                 logger.info(f"--- Validity Check Result: {current_validity:.4f} (Best: {best_val_validity:.4f}) ---")


            # Log metrics (master process only)
            if wandb_log:
                 log_data = {
                      "iter": iter_num,
                      "eval/train_loss": train_loss,
                      "eval/val_loss": val_loss,
                      "eval/best_val_loss": best_val_loss,
                      "eval/patience_counter": evals_no_improve,
                      "eval/eval_counter": eval_counter,
                      "lr": lr, # Log LR at eval time as well
                 }
                 # Log validity only if it was run and successful
                 if run_validity_check and current_validity >= 0.0:
                      log_data["eval/validity_score"] = current_validity
                      log_data["eval/best_validity_score"] = best_val_validity
                 try:
                      wandb.log(log_data)
                 except Exception as e:
                      logger.error(f"Wandb logging failed during eval: {e}", exc_info=True)

            # --- Checkpointing logic (master process only) ---
            # Check for improvement
            val_loss_improved = not np.isnan(val_loss) and val_loss < best_val_loss
            # Check validity improvement only if the check ran successfully (>= 0.0)
            validity_improved = run_validity_check and current_validity >= 0.0 and current_validity > best_val_validity

            # Decide whether to save based on configuration and improvements
            should_save = False
            save_reason = []

            if val_loss_improved:
                save_reason.append(f"Val loss improved ({best_val_loss:.4f} -> {val_loss:.4f})")
                best_val_loss = val_loss
                evals_no_improve = 0 # Reset patience on loss improvement
                should_save = True # Always save on val loss improvement

            # Consider saving on validity improvement only if configured and loss didn't improve
            if not val_loss_improved and save_on_validity_improve and validity_improved:
                save_reason.append(f"Validity improved ({best_val_validity:.4f} -> {current_validity:.4f})")
                best_val_validity = current_validity
                # Optional: Reset patience also on validity improvement?
                # evals_no_improve = 0
                should_save = True # Save if validity improved and loss didn't

            if always_save_checkpoint and not should_save: # Save if forced, unless already saving for other reasons
                 save_reason.append("always_save_checkpoint=True")
                 should_save = True

            # Save the checkpoint if needed
            if should_save and iter_num > 0: # Avoid saving at iteration 0
                logger.info(f"Saving checkpoint to {out_dir}. Reason: {', '.join(save_reason)}.")
                checkpoint_data = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args, # Save args used for this model
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'best_val_validity': best_val_validity, # Save best validity
                    'evals_no_improve': evals_no_improve,
                    'eval_counter': eval_counter, # Save eval counter
                    'config': config, # Save the final config dictionary used for the run
                }
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                best_ckpt_path = os.path.join(out_dir, 'best.pt') # Separate file for best loss
                temp_ckpt_path = ckpt_path + ".tmp" # Save to temp file first

                try:
                    torch.save(checkpoint_data, temp_ckpt_path)
                    os.rename(temp_ckpt_path, ckpt_path) # Atomic rename for latest checkpoint
                    logger.info(f"Checkpoint saved to {ckpt_path}")
                    # Also save to best.pt if val_loss improved
                    if val_loss_improved:
                         temp_best_ckpt_path = best_ckpt_path + ".tmp"
                         torch.save(checkpoint_data, temp_best_ckpt_path)
                         os.rename(temp_best_ckpt_path, best_ckpt_path)
                         logger.info(f"Best validation loss checkpoint saved to {best_ckpt_path}")

                except Exception as e:
                     logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

            # Update patience counter only if validation loss did not improve
            if not val_loss_improved:
                 evals_no_improve += 1
                 logger.info(f"Validation loss did not improve. Evals without improvement: {evals_no_improve}/{patience if patience > 0 else 'inf'}")
            logger.info(f"--- Finished Eval Iter {iter_num} ---")
        # --- End Evaluation and Checkpointing ---


        # --- Termination Conditions ---
        if iter_num >= max_iters:
            logger.info(f"Reached max_iters ({max_iters}). Stopping training.")
            break
        # Check patience only if it's enabled (patience > 0)
        if patience > 0 and evals_no_improve >= patience:
            logger.info(f"Validation loss did not improve for {patience} evaluations. Stopping training (early stopping).")
            break
        # --- End Termination Conditions ---


        # --- Training Step ---
        model.train() # Ensure model is in training mode
        # Loop over gradient accumulation steps
        for micro_step in range(gradient_accumulation_steps):
            # Prepare batch data from the pre-fetched data
            # Need to handle the case where current_data might be from a previous iteration
            # Better: fetch data inside the accumulation loop if iterator allows multiple fetches per step
            try:
                 # If it's the first micro_step, use the data fetched outside the loop
                 # Otherwise, fetch new data. This assumes one batch per train_iter.next()
                 if micro_step > 0:
                      current_data = next(train_iter)

                 X = current_data['input_ids'][:, :-1].to(device, non_blocking=True)
                 Y = current_data['labels'][:, 1:].to(device, non_blocking=True)
                 Y_mask = current_data.get('attention_mask', None)
                 if Y_mask is not None:
                      Y_mask = Y_mask[:, 1:].to(device, non_blocking=True)

                 # For DDP, sync gradients only on the last micro-step
                 # Use context manager for potentially cleaner code if model is DDP-wrapped
                 sync_context = model.no_sync() if (ddp and (micro_step < gradient_accumulation_steps - 1)) else nullcontext()

                 with sync_context:
                      with ctx: # Autocast context
                           # Adjust forward call based on model signature
                           try:
                                logits, loss = model(X, targets=Y, target_mask=Y_mask)
                           except TypeError:
                                logits, loss = model(X, targets=Y)

                           # Scale loss for gradient accumulation
                           loss = loss / gradient_accumulation_steps

                      # Backward pass: scaler automatically handles scaling/unscaling
                      # The backward pass should be inside the sync_context for DDP
                      scaler.scale(loss).backward()

            except StopIteration:
                 logger.info(f"Training data loader exhausted at iteration {iter_num}, micro_step {micro_step}. Resetting iterator.")
                 # Reset iterator and fetch next batch for the *next* outer loop iteration
                 train_iter = iter(train_loader)
                 # We might miss some accumulation steps here, decide how to handle
                 # Option 1: Break inner loop and proceed to optimizer step with partial accumulation
                 # Option 2: Fetch next data immediately (might be complex with loop structure)
                 # Let's break and step with whatever gradients we have accumulated
                 break # Break micro_step loop
            except Exception as e:
                 logger.error(f"Error during training step iter={iter_num}, micro_step={micro_step}: {e}", exc_info=True)
                 # Decide how to handle: skip batch, stop training?
                 # For now, let's try to continue, but this might lead to issues.
                 # Consider adding a counter for consecutive errors and stopping if too many occur.
                 loss = None # Mark loss as None to skip logging if error occurred


        # --- Optimizer Step ---
        # Perform optimizer step only after accumulating gradients for all micro_steps
        # Unscale gradients and clip (if enabled) before optimizer step
        if grad_clip > 0.0:
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            # Need to clip gradients for the raw model parameters if using DDP
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

        # Optimizer step (scaler handles checks for inf/NaN gradients)
        scaler.step(optimizer)
        # Update the scaler for next iteration
        scaler.update()
        # Flush gradients
        optimizer.zero_grad(set_to_none=True)
        # --- End Optimizer Step ---

        # --- Timing and Logging ---
        t1 = time.time()
        dt = t1 - t0 # Time delta for one optimizer step (incl. accumulation)
        t0 = t1
        # Log training loss and MFU periodically (master process only)
        if iter_num % log_interval == 0 and master_process:
            # loss is potentially scaled by grad_accum steps and might be from last micro step
            # Recompute loss on last batch for logging? Or use the potentially scaled value?
            # Let's log the loss from the last micro_step (multiplied back)
            if loss is not None: # Check if loss calculation succeeded
                 lossf = loss.item() * gradient_accumulation_steps # Approximate total loss for logging
                 if local_iter_num >= 5: # Calculate MFU after a few iterations to stabilize
                      # Estimate MFU using the raw model and global batch size
                      mfu = raw_model.estimate_mfu(global_batch_size, dt)
                      if mfu is not None: # MFU calculation might fail
                           running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                 logger.info(f"Iter {iter_num}/{max_iters} | Train Loss: {lossf:.4f} | LR: {lr:.2e} | Time/Iter: {dt*1000:.2f}ms | MFU: {running_mfu*100:.2f}%")
                 if wandb_log:
                      try:
                           wandb.log({
                                "iter": iter_num,
                                "train/loss": lossf,
                                # "lr": lr, # Already logged during eval
                                "timing/iter_time_ms": dt * 1000,
                                "perf/mfu_perc": running_mfu * 100,
                                "scaler_scale": scaler.get_scale() # Log grad scaler state
                           })
                      except Exception as e:
                           logger.error(f"Wandb logging failed during train step: {e}", exc_info=True)
            else:
                 logger.warning(f"Iter {iter_num}: Skipping train loss logging due to error in micro_step.")

        iter_num += 1
        local_iter_num += 1

        # Fetch the next batch *after* completing the current iteration's work
        try:
             current_data = next(train_iter)
        except StopIteration:
             logger.info(f"Training data loader exhausted at end of iteration {iter_num-1}. Resetting iterator.")
             train_iter = iter(train_loader)
             current_data = next(train_iter) # Fetch the first batch of the new epoch/pass
        # --- End Training Step Logic ---


    # --- End Training Loop ---

    # --- Cleanup ---
    if ddp:
        destroy_process_group()
    logger.info("Training finished.")
    if wandb_log and master_process:
        wandb.finish()
    # --- End Cleanup ---

