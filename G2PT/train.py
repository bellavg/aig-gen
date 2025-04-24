import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets_utils import get_datasets
from transformers import AutoTokenizer
# from tokenization_g2pt_fast import G2PTTokenizerFast
from torch.utils.data.distributed import DistributedSampler
from model import GPTConfig, GPT
from transformers import default_data_collator

torch._dynamo.config.optimize_ddp = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Default configuration values for training a GPT-2 model on OpenWebText

# I/O settings
out_dir = 'out'  # Directory to save outputs
eval_interval = 500  # Interval for evaluation
log_interval = 100  # Interval for logging
eval_iters = 200  # Number of iterations for evaluation
always_save_checkpoint = False  # Save checkpoint after each eval if True
init_from = 'scratch'  # Options: 'scratch', 'resume', 'gpt2*'

# Weights & Biases (wandb) logging settings
wandb_log = False  # Enable wandb logging if True
wandb_project = 'g2pt'  # Wandb project name
wandb_run_name = None  # Wandb run name

# Data settings
dataset = None  # Dataset to be used
gradient_accumulation_steps = 5 * 8  # Simulate larger batch sizes
batch_size = 12  # Micro-batch size if gradient_accumulation_steps > 1
block_size = 1024  # Block size for model input
vocab_size = None  # Vocabulary size
ordering = 'topo'
num_augmentations = 5
patience = 7

# Model architecture settings
n_layer = 12  # Number of layers
n_head = 12  # Number of attention heads
n_embd = 768  # Embedding size
dropout = 0.0  # Dropout rate; 0 for pretraining, 0.1+ for finetuning
bias = False  # Use bias in LayerNorm and Linear layers if True
model_name = 'base'

# AdamW optimizer settings
learning_rate = 5e-5  # Maximum learning rate
max_iters = 30000  # Total number of training iterations
weight_decay = 1e-1  # Weight decay for optimizer
beta1 = 0.9  # Beta1 for AdamW
beta2 = 0.95  # Beta2 for AdamW
grad_clip = 1.0  # Gradient clipping value; disable if 0.0

# Learning rate decay settings
decay_lr = True  # Enable learning rate decay if True
warmup_iters = 2000  # Number of warmup iterations
lr_decay_iters = 30000  # Iterations for learning rate decay
min_lr = 1e-5  # Minimum learning rate

# Distributed Data Parallel (DDP) settings
backend = 'nccl'  # Backend for DDP; options: 'nccl', 'gloo', etc.

# System settings
device = 'cuda'  # Device for training; options: 'cpu', 'cuda', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # Data type
compile = False  # Compile model with PyTorch 2.0 for speed if True

# -----------------------------------------------------------------------------
# Load additional configuration from external file
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # Override settings from command line or config file
config = {k: globals()[k] for k in config_keys}  # Configuration dictionary for logging
# -----------------------------------------------------------------------------
if wandb_log:
    wandb_run_name = f"{dataset}-{model_name}-{ordering}"
out_dir = f'results/{wandb_run_name}'
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter},{gradient_accumulation_steps},{ddp_world_size}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data preparation
tokenizer = AutoTokenizer.from_pretrained(f'tokenizers/{dataset}')
train_dataset, eval_dataset = get_datasets(dataset, tokenizer, ordering, num_augmentations)


def data_collate_fn(features):
    # datasets output datapoint with max length, we need to truncate to the max length of the batch (by checking the attention mask)
    features = default_data_collator(features)
    seq_len = features['attention_mask'].sum(-1)
    max_len = seq_len.max()
    features = {k: v[..., :max_len] for k, v in features.items()}
    return features


train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    pin_memory=True,
    drop_last=False,
    num_workers=8,
    collate_fn=data_collate_fn
)

eval_sampler = DistributedSampler(eval_dataset) if ddp else None
eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=batch_size,
    sampler=eval_sampler,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=8,
    collate_fn=data_collate_fn
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
evals_no_improve = 0

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', eval_loader)]:
        losses = torch.zeros(eval_iters)
        num_eval_iters = 0
        while True:
            for data in loader:
                X, Y, Y_mask = data['input_ids'][:, :-1], data['labels'][:, 1:], data['attention_mask'][:, 1:]
                X = X.to(device)
                Y = Y.to(device)
                Y_mask = Y_mask.to(device)
                with ctx:
                    logits, loss = model(X, targets=Y, target_masks=Y_mask)
                losses[num_eval_iters] = loss.item()
                num_eval_iters += 1
                if num_eval_iters >= eval_iters:
                    break

            if num_eval_iters >= eval_iters:
                break

        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# --- Training Loop ---
if __name__ == '__main__':
    # Setup logging (wandb)
    if config['wandb_log'] and master_process:
        import wandb
        print("Initializing wandb...")
        try:
            wandb.init(project=config['wandb_project'], name=wandb_run_name, config=config)
            print("wandb initialized.")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            config['wandb_log'] = False

    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if isinstance(model, DDP) else model  # Get raw model

    running_mfu = -1.0
    current_iter = iter_num # Start from loaded iter_num

    print("Starting training loop...")
    while True:
        # Set epoch for distributed sampler
        if ddp and hasattr(train_sampler, 'set_epoch'):
             epoch = current_iter // len(train_loader) if len(train_loader) > 0 else 0
             train_sampler.set_epoch(epoch)

        # --- Batch Iteration ---
        for micro_step_in_epoch, data in enumerate(train_loader):
            lr = get_lr(current_iter) if config['decay_lr'] else config['learning_rate']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            X = data['input_ids'][:, :-1].to(device)
            Y = data['labels'][:, 1:].to(device)
            Y_mask = data['attention_mask'][:, 1:].to(device)

            is_last_micro_step = (micro_step_in_epoch + 1) % gradient_accumulation_steps == 0
            sync_context = model.no_sync() if ddp and not is_last_micro_step else nullcontext()

            with ctx, sync_context:
                # *** Verify this model call signature matches your model.py ***
                logits, loss = model(X, targets=Y, target_masks=Y_mask)
                if loss is not None:
                     loss = loss / gradient_accumulation_steps
                else:
                     print(f"Warning: Loss is None for iter {current_iter}. Skipping backward pass.")
                     continue # Skip micro-step if loss is None

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected ({loss.item()}) at iter {current_iter}. Skipping step.")
                # Skip optimizer step but ensure accumulation cycle completes if needed
                # If it's the last micro step, need to increment counters etc.
                if is_last_micro_step:
                     optimizer.zero_grad(set_to_none=True) # Still zero grads
                     current_iter += 1
                     local_iter_num += 1
                continue # Skip backward/step

            scaler.scale(loss).backward()

            # --- Gradient Accumulation Step ---
            if is_last_micro_step:
                if config['grad_clip'] > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # --- Logging and Timing ---
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                if current_iter % config['log_interval'] == 0 and master_process:
                    lossf = loss.item() * gradient_accumulation_steps
                    if local_iter_num >= 5:
                         current_raw_model = model.module if isinstance(model, DDP) else model
                         try: # Add try-except for MFU estimation
                              mfu = current_raw_model.estimate_mfu(gradient_accumulation_steps * ddp_world_size * config['batch_size'], dt)
                              running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                         except AttributeError: # Handle if estimate_mfu is not implemented
                              running_mfu = -1.0 # Indicate MFU not available
                              print("Warning: estimate_mfu method not found on model.")
                         except Exception as e_mfu: # Catch other potential errors
                              running_mfu = -1.0
                              print(f"Warning: MFU estimation failed: {e_mfu}")

                    print(f"iter {current_iter}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}, mfu {running_mfu*100:.2f}%")

                    if config['wandb_log']:
                         try:
                              wandb.log({
                                   "iter": current_iter, "train/loss": lossf, "lr": lr,
                                   "mfu": running_mfu * 100,
                              }, step=current_iter)
                         except Exception as e: print(f"wandb logging failed: {e}")

                # Increment iteration counter
                current_iter += 1
                local_iter_num += 1

                # --- Evaluation and Checkpointing ---
                if current_iter % config['eval_interval'] == 0 and master_process:
                    losses = estimate_loss()
                    val_loss = losses.get('val', float('inf'))
                    print(f"step {current_iter}: train loss {losses.get('train', float('nan')):.4f}, val loss {val_loss:.4f}")

                    if config['wandb_log']:
                         try:
                              wandb.log({
                                   "val/loss": val_loss,
                                   "train/loss_eval": losses.get('train', float('nan')),
                              }, step=current_iter)
                         except Exception as e: print(f"wandb logging failed: {e}")

                    if val_loss < best_val_loss or config['always_save_checkpoint']:
                        if val_loss < best_val_loss:
                             best_val_loss = val_loss
                             evals_no_improve = 0
                             print(f"New best validation loss: {best_val_loss:.4f}")
                        elif config['always_save_checkpoint']:
                             evals_no_improve += 1

                        current_raw_model = model.module if isinstance(model, DDP) else model
                        checkpoint = {
                            'model': current_raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'model_args': model_args, 'iter_num': current_iter,
                            'best_val_loss': best_val_loss, 'config': config,
                        }
                        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                        print(f"Saving checkpoint to {ckpt_path}")
                        torch.save(checkpoint, ckpt_path)
                        if evals_no_improve == 0:
                             best_ckpt_path = os.path.join(out_dir, 'best.pt')
                             print(f"Saving best checkpoint to {best_ckpt_path}")
                             torch.save(checkpoint, best_ckpt_path)
                    else:
                         evals_no_improve += 1

                    if config['patience'] > 0 and evals_no_improve >= config['patience']:
                         print(f"Stopping early after {config['patience']} evaluations without improvement.")
                         break # Break inner loop

            # --- Check Termination Conditions (Inside Batch Loop) ---
            if current_iter > config['max_iters']: break
            if config['patience'] > 0 and evals_no_improve >= config['patience']: break

        # --- Check Termination Conditions (After Batch Loop) ---
        if current_iter > config['max_iters']:
            print(f"Reached max iterations ({config['max_iters']}).")
            break
        if config['patience'] > 0 and evals_no_improve >= config['patience']:
             break

    # --- End of Training ---
    print("Training finished.")
    if ddp:
        destroy_process_group()
    if config['wandb_log'] and master_process:
        wandb.finish()