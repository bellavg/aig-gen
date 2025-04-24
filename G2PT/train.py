# G2PT/train.py
import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
# Make sure torch.utils.data.DataLoader is imported if needed elsewhere, though not directly used here after imports
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# Ensure datasets_utils can be found (adjust path if necessary)
from datasets_utils import get_datasets
from transformers import AutoTokenizer
# from tokenization_g2pt_fast import G2PTTokenizerFast # Keep commented unless used
from torch.utils.data.distributed import DistributedSampler
# Ensure model can be found (adjust path if necessary)
from model import GPTConfig, GPT
from transformers import default_data_collator

# Optimize DDP setting (consider True for potential speedups in newer PyTorch versions, but False is safer for compatibility)
# torch._dynamo.config.optimize_ddp = False # Keep as is if working, or experiment with True

# Disable tokenizer parallelism to avoid potential issues with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# -----------------------------------------------------------------------------
# Default configuration values (Keep these or load from config file)
# ... (All your default config values like num_augmentations, out_dir, etc.) ...
num_augmentations = 5
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 200
always_save_checkpoint = False
init_from = 'scratch'
wandb_log = False
wandb_project = 'g2pt'
wandb_run_name = None # Will be set later if wandb_log is True
dataset = 'aig' # Specify your dataset name here or via config
gradient_accumulation_steps = 5 * 8
batch_size = 32
block_size = 1024 # Ensure this is >= max sequence length found by check_seq_length.py
vocab_size = None # Will be determined from tokenizer
ordering = 'topo' # Or 'topo', 'deg'
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = False
model_name = 'base'
learning_rate = 5e-5
max_iters = 300000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 30000 # Should likely be <= max_iters
min_lr = 1e-5
backend = 'nccl'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # Set to True to try compile (requires PyTorch 2.0+)
# -----------------------------------------------------------------------------

# --- Configuration Loading (Keep as is) ---
# Load additional configuration from external file if needed
try:
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))] # Include NoneType
    # Ensure configurator.py exists or handle FileNotFoundError
    exec(open('configurator.py').read()) # Overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # Configuration dictionary for logging
    print("Configuration loaded/updated from configurator.py")
except FileNotFoundError:
    config = {k: globals()[k] for k in config_keys} # Use defaults if no config file
    print("configurator.py not found, using default settings.")
except Exception as e:
     config = {k: globals()[k] for k in config_keys} # Use defaults on error
     print(f"Error loading configurator.py: {e}. Using default settings.")
# -----------------------------------------------------------------------------

# --- Set Run Name and Output Dir (after config loading) ---
if wandb_log:
    # Use dataset name from config if available, default otherwise
    effective_dataset_name = config.get('dataset', 'default_dataset')
    effective_model_name = config.get('model_name', 'default_model')
    effective_ordering = config.get('ordering', 'default_order')
    effective_num_augmentations = config.get('num_augmentations', 1)
    wandb_run_name = config.get('wandb_run_name') or f"{effective_dataset_name}-{effective_model_name}-{effective_ordering}-{effective_num_augmentations}"

# Ensure out_dir uses the potentially updated wandb_run_name or a default
effective_out_dir_name = wandb_run_name if wandb_run_name else f"{config.get('dataset', 'aig')}-{config.get('model_name', 'base')}-run"
out_dir = os.path.join('results', effective_out_dir_name) # Place results in a 'results' subdirectory
os.makedirs(out_dir, exist_ok=True) # Ensure output directory exists
print(f"Output directory: {out_dir}")
# -----------------------------------------------------------------------------

# --- DDP Setup (Keep as is) ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=config['backend']) # Use backend from config
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    # Adjust gradient accumulation steps for DDP
    assert config['gradient_accumulation_steps'] % ddp_world_size == 0
    gradient_accumulation_steps = config['gradient_accumulation_steps'] // ddp_world_size
    print(f"DDP active. Rank {ddp_rank}/{ddp_world_size}. Using device: {device}")
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    gradient_accumulation_steps = config['gradient_accumulation_steps'] # Use original value
    # Simpler device logic
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'
         print("Warning: MPS support is experimental.")
    else:
         device = 'cpu'
    print(f"Running on device: {device}")
# -----------------------------------------------------------------------------

# --- Tokens Per Iteration Calculation (Keep as is) ---
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config['batch_size'] * config['block_size']
print(f"Tokens per iteration: {tokens_per_iter:,}")
print(f"Effective batch size (gradient_accumulation_steps * ddp_world_size * batch_size): {gradient_accumulation_steps * ddp_world_size * config['batch_size']}")
# -----------------------------------------------------------------------------

# --- Dtype and Autocast Setup (Keep as is) ---
# Determine device type ('cuda' or 'cpu') based on the final device string
device_type = 'cuda' if 'cuda' in device else 'cpu'
# Use dtype from config
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
print(f"Using dtype: {config['dtype']}")
print(f"Autocast context: device_type={device_type}, dtype={ptdtype}")
# Enable TF32 for CUDA acceleration if using CUDA
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# -----------------------------------------------------------------------------

# --- Data Preparation ---
# Load tokenizer
tokenizer_path = os.path.join('tokenizers', config['dataset']) # Construct path relative to script location
print(f"Loading tokenizer from: {tokenizer_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    # Set pad token if not already set (important for default_data_collator)
    if tokenizer.pad_token is None:
        if '[PAD]' in tokenizer.vocab:
             tokenizer.pad_token = '[PAD]'
             print("Set tokenizer pad_token to '[PAD]'")
        else:
             # Handle case where PAD token is missing, maybe add it or use EOS
             # For now, raise an error as padding is usually required.
             raise ValueError("Tokenizer loaded successfully but is missing a PAD token ('[PAD]').")
except Exception as e:
    print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
    exit(1) # Exit if tokenizer fails to load

# Load datasets using the updated get_datasets function
print(f"Loading datasets for '{config['dataset']}' with ordering '{config['ordering']}'...")
try:
    # Pass the loaded tokenizer to get_datasets
    train_dataset, eval_dataset = get_datasets(
        config['dataset'],
        tokenizer,
        config['ordering'],
        config['num_augmentations']
    )
    print(f"Datasets loaded. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
except Exception as e:
    print(f"Failed to load datasets: {e}")
    exit(1) # Exit if datasets fail to load
# -----------------------------------------------------------------------------

# --- Data Collator (Keep as is, default_data_collator handles padding with tokenizer.pad_token_id) ---
# This collate function truncates sequences *after* batching and padding.
# Ensure block_size is large enough for the longest sequence + context tokens.
def data_collate_fn(features):
    # Default collator handles padding based on tokenizer.pad_token_id
    # It will pad 'input_ids', 'attention_mask', and 'labels' to the max length in the batch.
    collated_features = default_data_collator(features)

    # Optional: Truncate to block_size *after* padding if necessary,
    # though usually truncation happens *during* tokenization if `truncation=True`.
    # If sequences might exceed block_size after prepending context, truncation here is needed.
    max_len_in_batch = collated_features['input_ids'].shape[1]
    target_len = min(max_len_in_batch, config['block_size']) # Ensure not exceeding block_size

    # Truncate all relevant keys
    for k in collated_features.keys():
        if isinstance(collated_features[k], torch.Tensor) and collated_features[k].ndim > 1:
             collated_features[k] = collated_features[k][:, :target_len]

    return collated_features
# -----------------------------------------------------------------------------

# --- Data Loaders (Keep as is) ---
num_loader_workers = 8 if ddp else 0 # Use more workers for DDP
print(f"DataLoader num_workers: {num_loader_workers}")

train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42+seed_offset) if ddp else None # Add seed for reproducibility
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    sampler=train_sampler,
    shuffle=(train_sampler is None), # Shuffle only if not using DDP sampler
    pin_memory=True, # Enable pin_memory for faster GPU transfer if possible
    num_workers=num_loader_workers,
    collate_fn=data_collate_fn,
    drop_last=False # Keep drop_last=False unless strictly necessary
)

eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if ddp else None # No shuffling for eval
eval_loader = DataLoader(
    eval_dataset,
    batch_size=config['batch_size'], # Use same batch size for eval or adjust if needed
    sampler=eval_sampler,
    shuffle=False,
    pin_memory=True,
    num_workers=num_loader_workers,
    collate_fn=data_collate_fn,
    drop_last=False
)
# -----------------------------------------------------------------------------

# --- Initialization Variables (Keep as is) ---
patience = 7 # Define patience for early stopping (optional)
iter_num = 0
best_val_loss = 1e9 # Use a large number for initial best loss
evals_no_improve = 0 # Counter for early stopping
# -----------------------------------------------------------------------------

# --- Model Initialization ---
# Use parameters from config
model_args = dict(
    n_layer=config['n_layer'],
    n_head=config['n_head'],
    n_embd=config['n_embd'],
    block_size=config['block_size'],
    bias=config['bias'],
    vocab_size=None, # Set dynamically below
    dropout=config['dropout']
)

if config['init_from'] == 'scratch':
    print("Initializing a new model from scratch")
    # *** IMPORTANT: Set vocab_size from the loaded tokenizer ***
    model_args['vocab_size'] = tokenizer.vocab_size
    print(f"Model vocab size set to: {model_args['vocab_size']}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif config['init_from'] == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # Ensure essential config matches, but allow overriding others like dropout
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            # Check if the arg exists in the checkpoint, otherwise use current config
            if k in checkpoint_model_args:
                 model_args[k] = checkpoint_model_args[k]
            else:
                 print(f"Warning: Checkpoint missing argument '{k}'. Using value from current config: {model_args.get(k)}")

        # *** Set vocab_size from checkpoint if available, otherwise from tokenizer ***
        if 'vocab_size' in checkpoint_model_args:
             model_args['vocab_size'] = checkpoint_model_args['vocab_size']
             print(f"Model vocab size loaded from checkpoint: {model_args['vocab_size']}")
             # Verify checkpoint vocab size matches current tokenizer
             if model_args['vocab_size'] != tokenizer.vocab_size:
                  print(f"Warning: Checkpoint vocab size ({model_args['vocab_size']}) differs from current tokenizer vocab size ({tokenizer.vocab_size}). Ensure this is intended.")
        else:
             model_args['vocab_size'] = tokenizer.vocab_size
             print(f"Warning: Checkpoint missing 'vocab_size'. Using size from current tokenizer: {model_args['vocab_size']}")


        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # Fix state dict keys if needed (common issue with DDP/compile)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint.get('iter_num', 0) # Use get with default
        best_val_loss = checkpoint.get('best_val_loss', 1e9) # Use get with default
        print(f"Resumed from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")
        # Load optimizer state if resuming
        if 'optimizer' in checkpoint:
             # Optimizer needs to be created before loading state
             optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
             optimizer.load_state_dict(checkpoint['optimizer'])
             print("Optimizer state loaded from checkpoint.")
             checkpoint = None # Free up memory
        else:
             optimizer = None # Mark optimizer as not loaded yet
             print("Warning: Optimizer state not found in checkpoint.")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}. Starting from scratch.")
        config['init_from'] = 'scratch' # Fallback to scratch
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        config['init_from'] = 'scratch' # Fallback to scratch

    # Handle the case where init_from was changed back to 'scratch'
    if config['init_from'] == 'scratch':
        print("Initializing a new model from scratch (fallback from resume)")
        model_args['vocab_size'] = tokenizer.vocab_size
        print(f"Model vocab size set to: {model_args['vocab_size']}")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        optimizer = None # Ensure optimizer is reset if falling back

# --- Model Surgery (Keep as is) ---
# Crop block size if needed (e.g., fine-tuning with smaller context)
if config['block_size'] < model.config.block_size:
    print(f"Cropping model block size from {model.config.block_size} to {config['block_size']}")
    model.crop_block_size(config['block_size'])
    model_args['block_size'] = config['block_size'] # Update args to reflect change

# Move model to device
model.to(device)
# -----------------------------------------------------------------------------

# --- Optimizer ---
# Create optimizer if not loaded from checkpoint
if 'optimizer' not in locals() or optimizer is None:
     optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
     print("Optimizer created.")
# -----------------------------------------------------------------------------

# --- GradScaler (Keep as is) ---
scaler = torch.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
# -----------------------------------------------------------------------------

# --- Compile Model (Optional) ---
if config['compile']:
    if hasattr(torch, 'compile'):
        print("Compiling the model... (takes a ~minute)")
        try:
            # Ensure model is unwrapped before compiling if DDP was used in resume
            unoptimized_model = model.module if isinstance(model, DDP) else model
            model = torch.compile(unoptimized_model) # Requires PyTorch 2.0+
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}. Continuing without compilation.")
            # Fallback to the uncompiled model
            model = unoptimized_model if 'unoptimized_model' in locals() else model
    else:
        print("torch.compile not available. Continuing without compilation.")
# -----------------------------------------------------------------------------

# --- DDP Wrapping ---
# Wrap model *after* potential compilation and *before* starting training loop
if ddp:
    # Ensure model is on the correct device before wrapping
    model.to(device)
    model = DDP(model, device_ids=[ddp_local_rank])
    print("Model wrapped in DDP.")
# -----------------------------------------------------------------------------

# --- Loss Estimation Function ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split, loader in [('train', train_loader), ('val', eval_loader)]:
        # Use a list to store losses and calculate mean at the end for flexibility
        losses = []
        num_eval_batches = 0
        # Iterate through the loader until enough iterations are done or loader is exhausted
        for data in loader:
            # Unpack data (already includes context tokens in input_ids/labels/mask)
            # Slice to get X (input) and Y (target). Y should be shifted.
            # Input: token 0 to n-1. Target: token 1 to n.
            X = data['input_ids'][:, :-1].to(device)
            Y = data['labels'][:, 1:].to(device) # Targets are shifted, ignore_index handles context tokens
            # Attention mask should also be sliced if used directly by model (though standard GPT doesn't need it explicitly)
            # Y_mask = data['attention_mask'][:, 1:].to(device) # Slice mask if needed

            # Forward pass with autocast context
            with ctx:
                # Pass only X and Y (targets) to the model.
                # The model's forward should handle causal masking internally.
                # Ensure model's forward signature matches: model(idx, targets=None, ...)
                logits, loss = model(X, targets=Y) # Pass targets directly for loss calculation

                # Handle potential DDP loss gathering if needed (usually loss is averaged across devices by DDP)
                # loss = loss.mean() # Ensure loss is scalar if coming from multiple devices without reduction

            losses.append(loss.item())
            num_eval_batches += 1
            if num_eval_batches >= eval_iters:
                break # Stop after eval_iters batches

        # Calculate mean loss, handle case where loader had fewer batches than eval_iters
        if losses:
            out[split] = np.mean(losses)
        else:
            out[split] = float('nan') # Indicate if no batches were processed

    model.train() # Set model back to training mode
    return out
# -----------------------------------------------------------------------------

# --- Learning Rate Scheduler Function (Keep as is) ---
def get_lr(it):
    lr = config['learning_rate']
    min_lr = config['min_lr']
    warmup_iters = config['warmup_iters']
    lr_decay_iters = config['lr_decay_iters']

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)
# -----------------------------------------------------------------------------


# --- Training Loop ---
if __name__ == '__main__':
    # Setup logging (e.g., wandb)
    if config['wandb_log'] and master_process:
        import wandb
        print("Initializing wandb...")
        try:
            wandb.init(project=config['wandb_project'], name=wandb_run_name, config=config)
            print("wandb initialized.")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            config['wandb_log'] = False # Disable wandb if init fails

    t0 = time.time()
    local_iter_num = 0 # Iterations since this process started
    # Get the raw model reference for saving/estimation (needed if using DDP or compile)
    raw_model = model.module if ddp or config['compile'] else model
    running_mfu = -1.0

    print("Starting training loop...")
    # Use iter_num from checkpoint if resumed
    current_iter = iter_num

    # Main loop
    while True:
        # Set epoch for distributed sampler (ensures shuffling changes each epoch)
        if ddp and hasattr(train_sampler, 'set_epoch'):
             train_sampler.set_epoch(current_iter // len(train_loader)) # Approximate epoch

        # --- Batch Iteration ---
        for micro_step, data in enumerate(train_loader):
            # Determine learning rate for this iteration
            lr = get_lr(current_iter) if config['decay_lr'] else config['learning_rate']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Prepare batch data
            # Input: token 0 to n-1. Target: token 1 to n.
            X = data['input_ids'][:, :-1].to(device)
            Y = data['labels'][:, 1:].to(device) # Targets are shifted

            # Forward/Backward pass
            # Set sync for DDP only on the last micro-step
            if ddp:
                # Ensure requires_grad is set correctly if using torch.compile with DDP
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            # Use autocast context
            with ctx:
                # Pass only X and Y (targets) to the model
                logits, loss = model(X, targets=Y)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Backward pass, letting scaler handle dtype checks
            scaler.scale(loss).backward()

            # --- Gradient Accumulation Step ---
            if (micro_step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                if config['grad_clip'] > 0.0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

                # Optimizer step
                scaler.step(optimizer)
                scaler.update() # Update scaler for next iteration

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)

                # --- Logging and Timing ---
                t1 = time.time()
                dt = t1 - t0
                t0 = t1 # Reset timer

                if current_iter % config['log_interval'] == 0 and master_process:
                    lossf = loss.item() * gradient_accumulation_steps # Approximate loss for this step
                    if local_iter_num >= 5: # Calculate MFU after a few steps
                         mfu = raw_model.estimate_mfu(gradient_accumulation_steps * ddp_world_size * config['batch_size'], dt)
                         running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                    print(f"iter {current_iter}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}, mfu {running_mfu*100:.2f}%")

                    # Log to wandb if enabled
                    if config['wandb_log']:
                         try:
                              wandb.log({
                                   "iter": current_iter,
                                   "train/loss": lossf,
                                   "lr": lr,
                                   "mfu": running_mfu * 100, # Log MFU percentage
                              })
                         except Exception as e:
                              print(f"wandb logging failed: {e}")

                # Increment iteration counters
                current_iter += 1
                local_iter_num += 1

                # --- Evaluation and Checkpointing ---
                if current_iter % config['eval_interval'] == 0 and master_process:
                    losses = estimate_loss()
                    print(f"step {current_iter}: train loss {losses.get('train', float('nan')):.4f}, val loss {losses.get('val', float('nan')):.4f}")

                    # Log validation loss to wandb
                    if config['wandb_log']:
                         try:
                              wandb.log({
                                   "iter": current_iter, # Log iter again for step alignment
                                   "val/loss": losses.get('val', float('nan')),
                                   "train/loss_eval": losses.get('train', float('nan')), # Log train loss estimated during eval phase
                              })
                         except Exception as e:
                              print(f"wandb logging failed: {e}")

                    # Save checkpoint if validation loss improved or always saving
                    if losses.get('val', float('inf')) < best_val_loss or config['always_save_checkpoint']:
                        if losses.get('val', float('inf')) < best_val_loss:
                             best_val_loss = losses['val']
                             evals_no_improve = 0 # Reset patience counter
                             print(f"New best validation loss: {best_val_loss:.4f}")
                        else:
                             evals_no_improve += 1 # Increment patience counter if not improving but saving anyway

                        # Prepare checkpoint data
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args, # Save model args used
                            'iter_num': current_iter,
                            'best_val_loss': best_val_loss,
                            'config': config, # Save the run config
                        }
                        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                        print(f"Saving checkpoint to {ckpt_path}")
                        torch.save(checkpoint, ckpt_path)
                        # Save a separate 'best.pt' as well
                        if evals_no_improve == 0: # Only save best.pt if loss actually improved
                             best_ckpt_path = os.path.join(out_dir, 'best.pt')
                             print(f"Saving best checkpoint to {best_ckpt_path}")
                             torch.save(checkpoint, best_ckpt_path)

                    else:
                         evals_no_improve += 1 # Increment patience if validation loss did not improve

                    # Early stopping check
                    if patience > 0 and evals_no_improve >= patience:
                         print(f"Stopping early after {patience} evaluations without improvement.")
                         break # Break inner loop (batch iteration)

            # --- Check Termination Conditions ---
            if current_iter > config['max_iters']:
                break # Break inner loop (batch iteration)
            if patience > 0 and evals_no_improve >= patience:
                 break # Break inner loop again if early stopping triggered during eval

        # --- Check Termination Conditions (Outer Loop) ---
        if current_iter > config['max_iters']:
            print(f"Reached max iterations ({config['max_iters']}).")
            break # Break outer loop (while True)
        if patience > 0 and evals_no_improve >= patience:
             break # Break outer loop if early stopping triggered

    # --- End of Training ---
    print("Training finished.")
    if ddp:
        destroy_process_group()

    # Final wandb finish
    if config['wandb_log'] and master_process:
        wandb.finish()

