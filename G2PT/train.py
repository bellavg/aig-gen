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
# Ensure these config files exist and are importable
try:
    import configs.aig as aig_cfg
    import configs.base as net_cfg
    import configs.train_aig as train_cfg
    import configs.sample_aig as sample_cfg
    from datasets_utils import get_datasets, seq_to_nxgraph
    from model import GPT, GPTConfig
    from evaluate_aigs import validate_aig_structures # Import the validation function
    from sample_in_train import get_graphs # Import the generation+parsing function
except ImportError as e:
     logger.error(f"Failed to import required modules or configs: {e}")
     logger.error("Ensure all config files (aig.py, base.py, train_aig.py, sample_aig.py) exist in G2PT/configs/")
     logger.error("Ensure datasets_utils.py, model.py, evaluate_aigs.py, sample.py exist and G2PT is in PYTHONPATH.")
     sys.exit(1)

# --- Environment Setup ---
# torch._dynamo.config.optimize_ddp = False
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid tokenizer parallelism conflicts

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main():
    # Start with base configurations loaded from files
    config = {}
    config.update({k: v for k, v in vars(aig_cfg).items() if not k.startswith('_')})
    config.update({k: v for k, v in vars(net_cfg).items() if not k.startswith('_')})
    config.update({k: v for k, v in vars(train_cfg).items() if not k.startswith('_')})
    config.update({k: v for k, v in vars(sample_cfg).items() if not k.startswith('_')})

    # Apply overrides from configurator.py (command line args)
    try:
        global_config_for_exec = config # Make config dict accessible to exec
        configurator_path = os.path.join(os.path.dirname(__file__), 'configurator.py')
        if not os.path.exists(configurator_path):
             configurator_path = 'configurator.py' # Fallback relative to execution

        if os.path.exists(configurator_path):
            logger.info(f"Loading overrides from {configurator_path}")
            with open(configurator_path) as f:
                # Pass config directly to the execution context
                exec(f.read(), {'config': global_config_for_exec})
            config = global_config_for_exec # Use the potentially modified config
            logger.info("Configuration potentially overridden by configurator.py")
        else:
            logger.warning("configurator.py not found, using default configurations.")

    except Exception as e:
        logger.error(f"Error executing configurator.py: {e}", exc_info=True)
        sys.exit(1)

    # --- Extract final config values after potential overrides ---
    # Use config.get() for flexibility and to avoid KeyErrors if keys are missing
    num_augmentations = config.get('num_augmentations', 1)
    out_dir_base = config.get('out_dir_base', 'results')
    eval_interval = config.get('eval_interval', 1000)
    log_interval = config.get('log_interval', 10)
    eval_iters = config.get('eval_iters', 200)
    always_save_checkpoint = config.get('always_save_checkpoint', False)
    init_from = config.get('init_from', 'scratch')
    wandb_log = config.get('wandb_log', False)
    wandb_project = config.get('wandb_project', 'g2pt')
    wandb_run_name_base = config.get('wandb_run_name', None)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    batch_size = config.get('batch_size', 12)
    max_iters = config.get('max_iters', 600000)
    patience = config.get('patience', 0) # Default to 0 (disabled)
    learning_rate = config.get('learning_rate', 6e-4)
    weight_decay = config.get('weight_decay', 1e-1)
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.95)
    grad_clip = config.get('grad_clip', 1.0)
    decay_lr = config.get('decay_lr', True)
    warmup_iters = config.get('warmup_iters', 2000)
    lr_decay_iters = config.get('lr_decay_iters', 600000)
    min_lr = config.get('min_lr', 6e-5)
    backend = config.get('backend', 'nccl')
    compile_model = config.get('compile', False)
    num_loader_workers_cfg = config.get('num_loader_workers', 0)

    dataset = config.get('dataset')
    if dataset is None: logger.error("Dataset name ('dataset') must be specified in config."); sys.exit(1)
    block_size = config.get('block_size', 1024)
    vocab_size = config.get('vocab_size', None) # Will be set from tokenizer
    ordering = config.get('ordering', 'topo')
    tokenizer_path_cfg = config.get('tokenizer_path', f'./datasets/{dataset}/tokenizer')

    n_layer = config.get('n_layer', 12)
    n_head = config.get('n_head', 12)
    n_embd = config.get('n_embd', 768)
    dropout = config.get('dropout', 0.0)
    bias = config.get('bias', False)
    model_name = config.get('model_name', 'base')

    validity_check_interval_multiplier = config.get('validity_check_interval_multiplier', 2)
    validity_num_samples = config.get('validity_num_samples', 100)
    validity_max_new_tokens = config.get('validity_max_new_tokens', block_size)
    validity_temperature = config.get('validity_temperature', 0.8)
    validity_top_k = config.get('validity_top_k', None)
    validity_gen_batch_size = config.get('validity_gen_batch_size', batch_size * 2)
    validity_parsing_mode = config.get('validity_parsing_mode', 'strict')
    save_on_validity_improve = config.get('save_on_validity_improve', True)
    # --- End Config Extraction ---

    # --- DDP Setup ---
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        if not is_initialized(): init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        logger.info(f"DDP active. Rank {ddp_rank}/{ddp_world_size}. Local Rank {ddp_local_rank}. Device: {device}.")
    else:
        master_process = True; seed_offset = 0; ddp_world_size = 1; ddp_rank = 0
        if torch.cuda.is_available(): device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = 'mps'; logger.warning("MPS device selected.")
        else: device = 'cpu'
        logger.info(f"DDP not active. Running on device: {device}")
    # --- End DDP Setup ---

    # Construct dynamic names/paths
    if wandb_log and master_process:
        wandb_run_name = f"{wandb_run_name_base or f'{dataset}-{model_name}'}-{ordering}-aug{num_augmentations}"
    else:
        rank_suffix = f"-rank{ddp_rank}" if ddp else ""
        wandb_run_name = f"{dataset}-{model_name}-{ordering}-aug{num_augmentations}-local{rank_suffix}"

    out_dir = os.path.join(out_dir_base, wandb_run_name)
    logger.info(f"Output directory: {out_dir}")

    global_batch_size = batch_size * ddp_world_size * gradient_accumulation_steps
    tokens_per_iter = global_batch_size * block_size
    logger.info(f"Global Batch Size: {global_batch_size} sequences")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logger.info(f"Tokens per Optimizer Step (approx): {tokens_per_iter:,}")

    # --- Dtype and Autocast Setup ---
    if 'cuda' in device:
        final_dtype_str = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        device_type = 'cuda'
    else: final_dtype_str = 'float32'; device_type = 'mps' if device == 'mps' else 'cpu'

    logger.info(f"Selected dtype: {final_dtype_str}")
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[final_dtype_str]
    autocast_enabled = device_type in ('cuda', 'mps')
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype, enabled=autocast_enabled)
    logger.info(f"Autocast context: enabled={autocast_enabled}, device_type='{device_type}', dtype='{ptdtype}'")
    # --- End Dtype / Autocast Setup ---

    if master_process: os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset); np.random.seed(1337 + seed_offset)
    if 'cuda' in device: torch.cuda.manual_seed_all(1337 + seed_offset)

    # --- Data Loading ---
    tokenizer_load_path = os.path.join(os.path.dirname(__file__), tokenizer_path_cfg)
    if not os.path.exists(tokenizer_load_path):
         tokenizer_load_path = tokenizer_path_cfg # Fallback relative to execution

    logger.info(f"Loading tokenizer from: {tokenizer_load_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
        # Set vocab_size from tokenizer if not already set or mismatch
        logger.info(f"Final vocab size: {vocab_size}. Tokenizer Max length: {tokenizer.model_max_length}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {tokenizer_load_path}: {e}", exc_info=True)
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
        features = default_data_collator(features)
        if 'attention_mask' not in features: return features # Skip if no mask
        try: # Add try-except for safety
             max_len_in_batch = int(features['attention_mask'].sum(-1).max())
             # Ensure max_len is at least 1
             max_len_in_batch = max(1, max_len_in_batch)
             features = {
                 k: (v[:, :max_len_in_batch] if v.ndim > 1 and v.shape[1] > 0 else v)
                 for k, v in features.items()
             }
        except Exception as e:
             logger.error(f"Error in data_collate_fn: {e}")
             # Optionally return original features or raise error
        return features

    num_workers = num_loader_workers_cfg if not ddp else 0
    logger.info(f"DataLoader num_workers: {num_workers}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, drop_last=False) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
                              pin_memory=True, drop_last=False, num_workers=num_workers, collate_fn=data_collate_fn)

    eval_sampler = DistributedSampler(eval_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=False) if ddp else None
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler, shuffle=False,
                             pin_memory=True, drop_last=False, num_workers=num_workers, collate_fn=data_collate_fn)
    # --- End Data Loading ---

    # --- Model Initialization ---
    iter_num = 0; best_val_loss = 1e9; best_val_validity = -1.0; eval_counter = 0; evals_no_improve = 0

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=vocab_size, dropout=dropout)

    model = None # Initialize model variable
    checkpoint = None # Initialize checkpoint variable
    if init_from == 'scratch':
        logger.info("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        logger.info(f"Model initialized with {model.get_num_params()/1e6:.2f}M parameters.")
    elif init_from == 'resume':
        logger.info(f"Resuming training from {out_dir}")
        ckpt_path = None
        for fname in ['ckpt.pt', 'best.pt']:
             fpath = os.path.join(out_dir, fname)
             if os.path.exists(fpath): ckpt_path = fpath; break
        if ckpt_path is None: logger.error(f"Checkpoint not found in {out_dir}."); sys.exit(1)

        logger.info(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Load model args from checkpoint
        if 'model_args' in checkpoint: checkpoint_model_args = checkpoint['model_args']
        elif 'config' in checkpoint:
             cfg = config
             checkpoint_model_args = {'n_layer': cfg.get('n_layer'), 'n_head': cfg.get('n_head'), 'n_embd': cfg.get('n_embd'),
                                       'block_size': cfg.get('block_size'), 'bias': cfg.get('bias', False),
                                       'vocab_size': cfg.get('vocab_size'), 'dropout': cfg.get('dropout', 0.0)}
        else: logger.error("Cannot find 'model_args' or 'config' in checkpoint."); sys.exit(1)

        # Force essential keys and update model_args
        essential_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
        for k in essential_keys:
            if k in checkpoint_model_args:
                if k in model_args and model_args[k] != checkpoint_model_args[k]:
                     logger.warning(f"Overriding config '{k}' ({model_args[k]}) with checkpoint value ({checkpoint_model_args[k]})")
                model_args[k] = checkpoint_model_args[k]
            elif k not in model_args: logger.error(f"Essential arg '{k}' missing."); sys.exit(1)

        # Create model and load state
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # Fix potential prefix issues in state dict keys (DDP, compile)
        unwanted_prefixes = ['_orig_mod.', '_module.', 'module.']
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            original_k = k
            key_modified = False
            for prefix in unwanted_prefixes:
                if k.startswith(prefix):
                    cleaned_state_dict[k[len(prefix):]] = v
                    key_modified = True
                    break
            if not key_modified:
                cleaned_state_dict[k] = v


        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys: logger.warning(f"Missing keys loading state_dict: {missing_keys}")
        if unexpected_keys: logger.warning(f"Unexpected keys loading state_dict: {unexpected_keys}")
        logger.info("Model state loaded.")

        # Load training state
        iter_num = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', 1e9)
        best_val_validity = checkpoint.get('best_val_validity', -1.0)
        eval_counter = checkpoint.get('eval_counter', iter_num // eval_interval if eval_interval > 0 else 0)
        evals_no_improve = checkpoint.get('evals_no_improve', 0)
        logger.info(f"Resuming from iter {iter_num}, best_val_loss: {best_val_loss:.4f}, best_val_validity: {best_val_validity:.3f}")
    else: logger.error(f"Unknown init_from: {init_from}"); sys.exit(1)

    if model is None: logger.error("Model was not initialized."); sys.exit(1)

    # Crop block size if needed
    if model.config.block_size > block_size:
        logger.info(f"Cropping model block size from {model.config.block_size} to {block_size}")
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size
    elif model.config.block_size < block_size:
        logger.warning(f"Model block size ({model.config.block_size}) < config ({block_size}). Using model's size.")
        block_size = model.config.block_size

    model.to(device)
    # --- End Model Initialization ---

    # --- Optimizer and Scaler Setup ---
    scaler = torch.amp.GradScaler(enabled=(final_dtype_str == 'float16'))
    logger.info(f"Gradient Scaler enabled: {scaler.is_enabled()}")
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    logger.info(f"Optimizer configured: {type(optimizer)}")
    if init_from == 'resume' and checkpoint is not None and 'optimizer' in checkpoint: # Check if checkpoint exists
        try: optimizer.load_state_dict(checkpoint['optimizer']); logger.info("Optimizer state loaded.")
        except Exception as e: logger.error(f"Failed to load optimizer state: {e}.", exc_info=True)
    checkpoint = None # Free memory
    # --- End Optimizer and Scaler Setup ---

    # --- Model Compilation ---
    if compile_model:
        if hasattr(torch, 'compile'):
            logger.info("Compiling model...")
            try: model = torch.compile(model); logger.info("Model compiled.")
            except Exception as e: logger.warning(f"Compilation failed: {e}. Continuing without.", exc_info=True); compile_model = False
        else: logger.warning("torch.compile unavailable."); compile_model = False
    # --- End Model Compilation ---

    # --- DDP Wrapping ---
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
        logger.info("Model wrapped with DDP.")
    # --- End DDP Wrapping ---

    # --- Helper Functions (Defined inside main) ---
    @torch.no_grad()
    def estimate_loss(data_loader, split_name):
        """ Estimates loss over a given data loader using eval_iters batches. """
        model.eval()
        losses = torch.zeros(eval_iters, device=device)
        batches_processed = 0
        loader_iter = iter(data_loader)
        for k in range(eval_iters):
            try:
                data = next(loader_iter)
                X = data['input_ids'][:, :-1].to(device, non_blocking=True)
                Y = data['labels'][:, 1:].to(device, non_blocking=True)
                Y_mask = data.get('attention_mask', None)
                if Y_mask is not None: Y_mask = Y_mask[:, 1:].to(device, non_blocking=True)

                with ctx:
                    try:
                        logits, loss = model(X, targets=Y, target_masks=Y_mask)
                    except TypeError: # Fallback if model doesn't accept mask
                         logger.debug(f"Eval {split_name}: model forward doesn't accept target_mask.")
                         logits, loss = model(X, targets=Y)
                    # --- End Model Call ---
                losses[k] = loss.item()
                batches_processed += 1
            except StopIteration:
                logger.warning(f"Eval '{split_name}' loader exhausted before {eval_iters} iters.")
                losses = losses[:batches_processed]; break
            except Exception as e:
                logger.error(f"Error during loss estimation ({split_name}, batch {k}): {e}", exc_info=True)
                losses[k] = float('nan')
        model.train()
        if batches_processed == 0: return float('nan')
        valid_losses = losses[~torch.isnan(losses)]
        return valid_losses.mean().item() if valid_losses.numel() > 0 else float('nan')

    @torch.no_grad()
    def check_graph_validity():
        """ Samples graphs and checks validity. """
        logger.info(f"\n--- Starting Validity Check ---")
        # Use non-local variables defined earlier in main()
        logger.info(f"Sampling {validity_num_samples} graphs...")
        logger.info(f"Generation params: temp={validity_temperature}, top_k={validity_top_k}, max_new_tokens={validity_max_new_tokens}, batch_size={validity_gen_batch_size}")

        model.eval() # Ensure eval mode
        raw_model_for_check = model.module if ddp else model # Use raw model for generation
        master_device = device if not ddp else f'cuda:{ddp_local_rank}' # Ensure generation on correct device

        generated_graphs = get_graphs(model=raw_model_for_check, tokenizer=tokenizer,
            num_samples=validity_num_samples, batch_size=validity_gen_batch_size,
            temperature=validity_temperature,
           parsing_mode=validity_parsing_mode,
            seed=1337 + iter_num) # Consistent seed based on iteration)

        num_generated = len(generated_graphs)
        logger.info(f"Generated and parsed {num_generated} graphs.")
        if num_generated == 0: validity_score = 0.0
        else:
            logger.info(f"Evaluating validity of {num_generated} graphs...")
            try: validity_score = validate_aig_structures(generated_graphs)
            except Exception as e: logger.error(f"Validity evaluation error: {e}", exc_info=True); validity_score = -1.0
        logger.info(f"--- Validity Score: {validity_score:.4f} ---")
        model.train() # Set back to train mode
        return validity_score

    def get_lr(it):
        if it < warmup_iters: return learning_rate * it / max(1, warmup_iters)
        if lr_decay_iters <= 0 or it > lr_decay_iters: return min_lr
        decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    # --- End Helper Functions ---

    # --- Logging Setup ---
    if wandb_log and master_process:
        import wandb
        logger.info("Initializing wandb...")
        try:
            wandb_config = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
            wandb_config.update({'global_batch_size': global_batch_size, 'tokens_per_iter': tokens_per_iter})
            wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)
            logger.info("Wandb initialized.")
        except Exception as e: logger.error(f"Wandb init error: {e}", exc_info=True); wandb_log = False
    # --- End Logging Setup ---

    # --- Training Loop ---
    t0 = time.time(); local_iter_num = 0
    raw_model = model.module if ddp else model; running_mfu = -1.0
    logger.info(f"\n--- Starting Training Loop (Max Iter: {max_iters}) ---")

    train_iter = iter(train_loader) # Initialize iterator
    current_data = None # Initialize current_data to None
    try:
        current_data = next(train_iter) # Fetch the first batch
    except StopIteration:
        logger.error("Training data loader is empty! Cannot start training.")
        if ddp: destroy_process_group()
        sys.exit(1)

    while True:
        # Set sampler epoch
        if ddp and train_sampler is not None:
             if len(train_loader) > 0: train_sampler.set_epoch(iter_num // len(train_loader))

        # Set learning rate
        lr = get_lr(iter_num);
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        # Evaluation phase
        if iter_num % eval_interval == 0 and master_process:
            eval_counter += 1
            logger.info(f"--- Starting Eval Iter {iter_num} (Eval #{eval_counter}) ---")
            train_loss = estimate_loss(train_loader, 'train')
            val_loss = estimate_loss(eval_loader, 'val')
            logger.info(f"Eval Losses -> Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            current_validity = -1.0
            run_validity_check = (validity_check_interval_multiplier > 0 and eval_counter % validity_check_interval_multiplier == 0)
            if run_validity_check: current_validity = check_graph_validity()

            if wandb_log:
                 log_data = {"iter": iter_num, "eval/train_loss": train_loss, "eval/val_loss": val_loss,
                              "eval/best_val_loss": best_val_loss, "eval/patience_counter": evals_no_improve,
                              "eval/eval_counter": eval_counter, "lr": lr}
                 if run_validity_check and current_validity >= 0.0:
                      log_data["eval/validity_score"] = current_validity; log_data["eval/best_validity_score"] = best_val_validity
                 try: wandb.log(log_data)
                 except Exception as e: logger.error(f"Wandb eval log failed: {e}", exc_info=True)

            # Checkpointing logic
            val_loss_improved = not np.isnan(val_loss) and val_loss < best_val_loss
            validity_improved = run_validity_check and current_validity >= 0.0 and current_validity > best_val_validity
            should_save = False; save_reason = []
            if val_loss_improved and save_on_validity_improve and validity_improved:
                save_reason.append(f"Loss improved ({best_val_loss:.4f}->{val_loss:.4f})"); best_val_loss = val_loss; evals_no_improve = 0; should_save = True
                save_reason.append(f"Validity improved ({best_val_validity:.4f}->{current_validity:.4f})"); best_val_validity = current_validity; should_save = True
            elif not val_loss_improved and save_on_validity_improve and validity_improved:
                save_reason.append(f"Validity improved ({best_val_validity:.4f}->{current_validity:.4f})"); best_val_validity = current_validity; should_save = True
            elif val_loss_improved:
                logger.info(f"Loss improved ({best_val_loss:.4f}->{val_loss:.4f})")
            if always_save_checkpoint and not should_save: save_reason.append("Always save"); should_save = True

            if should_save and iter_num > 0:
                logger.info(f"Saving checkpoint to {out_dir}. Reason: {', '.join(save_reason)}.")
                checkpoint_data = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                                   'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss,
                                   'best_val_validity': best_val_validity, 'evals_no_improve': evals_no_improve,
                                   'eval_counter': eval_counter}
                ckpt_path = os.path.join(out_dir, 'ckpt.pt'); best_ckpt_path = os.path.join(out_dir, 'best.pt')
                temp_ckpt_path = ckpt_path + ".tmp"; temp_best_ckpt_path = best_ckpt_path + ".tmp"
                try:
                    torch.save(checkpoint_data, temp_ckpt_path); os.rename(temp_ckpt_path, ckpt_path)
                    logger.info(f"Checkpoint saved to {ckpt_path}")
                    if val_loss_improved:
                         torch.save(checkpoint_data, temp_best_ckpt_path); os.rename(temp_best_ckpt_path, best_ckpt_path)
                         logger.info(f"Best val loss checkpoint saved to {best_ckpt_path}")
                except Exception as e: logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

            if not val_loss_improved: evals_no_improve += 1; logger.info(f"Val loss no improve: {evals_no_improve}/{patience or 'inf'}")
            logger.info(f"--- Finished Eval Iter {iter_num} ---")

        # Termination conditions
        if iter_num >= max_iters: logger.info("Reached max_iters."); break
        if patience > 0 and evals_no_improve >= patience: logger.info("Early stopping."); break

        # Training step
        model.train()
        optimizer.zero_grad(set_to_none=True) # Zero gradients at the start of the accumulation cycle
        for micro_step in range(gradient_accumulation_steps):
            # Need to handle the case where current_data is None (e.g., after StopIteration)
            if current_data is None:
                 try:
                      current_data = next(train_iter)
                 except StopIteration:
                      logger.info(f"Train loader exhausted mid-accumulation at iter {iter_num}, micro {micro_step}. Resetting.")
                      train_iter = iter(train_loader)
                      try: current_data = next(train_iter)
                      except StopIteration: logger.error("Train loader empty after reset!"); break # Break inner loop
                      # If we reset mid-accumulation, should we break the inner loop?
                      # Yes, proceed to optimizer step with potentially fewer grads.
                      break

            # Prepare data for this micro-step
            X = current_data['input_ids'][:, :-1].to(device, non_blocking=True)
            Y = current_data['labels'][:, 1:].to(device, non_blocking=True)
            Y_mask = current_data.get('attention_mask', None)
            if Y_mask is not None: Y_mask = Y_mask[:, 1:].to(device, non_blocking=True)

            # DDP sync context
            is_last_micro_step = (micro_step == gradient_accumulation_steps - 1)
            sync_context = model.no_sync() if (ddp and not is_last_micro_step) else nullcontext()

            with sync_context:
                with ctx:
                    # --- IMPORTANT: Ensure model.py's forward accepts target_mask ---
                    try:
                        logits, loss = model(X, targets=Y, target_masks=Y_mask)
                        # Scale loss *before* backward pass
                        loss = loss / gradient_accumulation_steps
                    except TypeError:
                        logger.debug(f"Train iter {iter_num}: model forward doesn't accept target_mask.")
                        logits, loss = model(X, targets=Y)
                        loss = loss / gradient_accumulation_steps
                    except Exception as e:
                         logger.error(f"Error in model forward/loss iter={iter_num}, micro={micro_step}: {e}", exc_info=True)
                         loss = None # Mark loss as None on error
                         break # Break micro-step loop on forward error
                    # --- End Model Call ---

            # Backward pass only if loss calculation was successful
            if loss is not None:
                # scaler handles scaling/unscaling
                scaler.scale(loss).backward()
            else:
                 # If loss is None due to error, break accumulation loop
                 break

            # Fetch next data *for the next micro_step* if not the last one
            # And reset current_data to None so it's fetched next time if needed
            current_data = None
            if not is_last_micro_step:
                 try:
                      current_data = next(train_iter)
                 except StopIteration:
                      # This break will lead to optimizer step with accumulated grads so far
                      logger.info(f"Train loader exhausted mid-accumulation at iter {iter_num}, micro {micro_step}. Resetting.")
                      train_iter = iter(train_loader)
                      # Don't fetch here, let the outer loop handle it if needed
                      break # Break micro_step loop


        # Optimizer Step (after accumulation loop or break)
        if grad_clip > 0.0:
            scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer) # Step optimizer
        scaler.update() # Update scaler
        # Gradients are zeroed at the beginning of the next accumulation cycle

        # Timing and Logging
        t1 = time.time(); dt = t1 - t0; t0 = t1
        if iter_num % log_interval == 0 and master_process:
            if loss is not None: # Use loss from the last successful micro_step
                 lossf = loss.item() * gradient_accumulation_steps # Rescale for logging
                 if local_iter_num >= 5:
                      mfu = raw_model.estimate_mfu(global_batch_size, dt)
                      if mfu is not None: running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                 logger.info(f"Iter {iter_num}/{max_iters} | Loss: {lossf:.4f} | LR: {lr:.2e} | Time/Iter: {dt*1000:.2f}ms | MFU: {running_mfu*100:.2f}%")
                 if wandb_log:
                      try: wandb.log({"iter": iter_num, "train/loss": lossf, "timing/iter_time_ms": dt * 1000,
                                      "perf/mfu_perc": running_mfu * 100, "scaler_scale": scaler.get_scale()})
                      except Exception as e: logger.error(f"Wandb train log failed: {e}", exc_info=True)
            else: logger.warning(f"Iter {iter_num}: Skipping train loss logging due to error in micro_step.")

        iter_num += 1; local_iter_num += 1

        # Fetch next batch for the start of the *next* iteration's accumulation cycle
        # This handles the case where the micro-step loop finished normally
        if current_data is None: # Only fetch if not already fetched due to StopIteration break
             try:
                  current_data = next(train_iter)
             except StopIteration:
                  logger.info(f"Train loader exhausted at end of iter {iter_num-1}. Resetting.")
                  train_iter = iter(train_loader)
                  try: current_data = next(train_iter)
                  except StopIteration: logger.error("Train loader empty after reset!"); break # Break while loop
    # --- End Training Loop ---

    # --- Cleanup ---
    if ddp: destroy_process_group()
    logger.info("Training finished.")
    if wandb_log and master_process: wandb.finish()
    # --- End Cleanup ---

# --- Script Entry Point ---
if __name__ == '__main__':
    main()
# --- End Script Entry Point ---

