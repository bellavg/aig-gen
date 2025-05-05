"""
Sample from a trained model - AIG Generation Only Version
Refactored for use in training script.
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
import networkx as nx # Needed for type hint and checking return type
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase # For type hints
from typing import List, Optional # For type hints

# --- Module Imports ---
# Assume model.py and datasets_utils.py are in the same G2PT directory or accessible via PYTHONPATH
try:
    from model import GPTConfig, GPT
    # Ensure seq_to_nxgraph is available (it creates DiGraphs)
    from datasets_utils import seq_to_nxgraph
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"Error importing local modules (GPT, seq_to_nxgraph): {e}")
    print("Ensure sample.py is run from the G2PT directory or PYTHONPATH is set correctly.")
    MODULE_IMPORTS_OK = False
# --- End Module Imports ---

import argparse
import pickle
import logging # Use logging for better feedback

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("g2pt_sampler")


def parse_args():
    """Parses command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description='Sample from a trained G2PT model for AIGs')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing model checkpoint (e.g., results/aig-small-topo/ckpt.pt)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to the trained tokenizer directory (e.g., G2PT/datasets/aig/tokenizer)')
    parser.add_argument('--batch_size', type=int, default=64, # Smaller default might be safer for varied GPUs
                        help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=100, # Smaller default for quicker testing
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.8, # Often better than 1.0 for structured data
                        help='Sampling temperature (e.g., 0.8 for less randomness, 1.0 for standard)')
    parser.add_argument('--top_k', type=int, default=None, # Add top_k option
                        help='Top-k sampling parameter (optional)')
    parser.add_argument('--max_new_tokens', type=int, default=512, # Control generation length
                        help='Maximum number of new tokens to generate per sequence')
    parser.add_argument('--output_filename', type=str, default='generated_aigs.pkl',
                        help='Name for the output pickle file (saved in out_dir)')
    parser.add_argument('--parsing_mode', type=str, default='strict', choices=['strict', 'robust'],
                        help='Edge sequence parsing mode for seq_to_nxgraph: strict or robust')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cpu", "cuda", "cuda:0"). Auto-detects if None.')

    return parser.parse_args()

def setup_device_and_ctx(seed, requested_device=None):
    """Sets up device, dtype, and autocast context."""
    if requested_device:
        device = requested_device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'
         logger.warning("MPS device selected. Performance and compatibility may vary.")
    else:
         device = 'cpu'
    logger.info(f"Using device: {device}")

    # Determine appropriate dtype based on chosen device
    if 'cuda' in device:
        # Note: Autocast context handles dtype selection based on device capability
        # We still set torch defaults for potential non-autocast operations
        if torch.cuda.is_bf16_supported():
            dtype_str = 'bfloat16'
            torch.set_default_dtype(torch.bfloat16)
        else:
            dtype_str = 'float16'
            torch.set_default_dtype(torch.float16)
        # Enable TF32 for CUDA acceleration if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype_str = 'float32'
        torch.set_default_dtype(torch.float32) # Set default for CPU/MPS

    logger.info(f"Selected precision mode: {dtype_str}")

    # Seed setting
    torch.manual_seed(seed)
    np.random.seed(seed) # Seed numpy if used indirectly
    if 'cuda' in device:
        torch.cuda.manual_seed_all(seed) # Seed all GPUs if applicable

    # Autocast context
    device_type = 'cuda' if 'cuda' in device else ('mps' if device == 'mps' else 'cpu')
    # Use the determined dtype for autocast on GPU, float32 otherwise
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype, enabled=(device_type != 'cpu'))
    logger.info(f"Autocast context: enabled={device_type != 'cpu'}, device_type='{device_type}', dtype='{ptdtype}'")

    return device, ctx

def load_model_and_tokenizer(out_dir, tokenizer_path, device):
    """Loads the model checkpoint and tokenizer."""
    # --- Load Tokenizer ---
    try:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        # Use legacy=False if your tokenizer files support it and were created with it
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, legacy=False)
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Max length: {tokenizer.model_max_length}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {tokenizer_path}: {e}", exc_info=True)
        return None, None

    # --- Load Model Checkpoint ---
    # Allow loading from either ckpt.pt or best.pt for flexibility
    ckpt_path_options = [os.path.join(out_dir, 'ckpt.pt'), os.path.join(out_dir, 'best.pt')]
    ckpt_path = None
    for path in ckpt_path_options:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        logger.error(f"Error: No checkpoint file (ckpt.pt or best.pt) found in {out_dir}")
        return None, tokenizer # Return tokenizer if loaded

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError: # Should be caught above, but double-check
        logger.error(f"Error: Checkpoint file not found at {ckpt_path}")
        return None, tokenizer
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)
        return None, tokenizer

    # --- Initialize Model ---
    try:
        # Ensure config from checkpoint is used
        if 'config' in checkpoint and 'model_args' not in checkpoint:
             # Handle checkpoints saved with older format (config dict)
             logger.warning("Loading model args from 'config' dictionary in checkpoint.")
             cfg = checkpoint['config']
             # Map relevant keys from config dict to model_args dict
             model_args = {
                 'n_layer': cfg.get('n_layer'), 'n_head': cfg.get('n_head'), 'n_embd': cfg.get('n_embd'),
                 'block_size': cfg.get('block_size'), 'bias': cfg.get('bias', False), # Add default for bias
                 'vocab_size': cfg.get('vocab_size'), 'dropout': cfg.get('dropout', 0.0) # Add default for dropout
             }
             # Ensure critical args are present
             if None in model_args.values():
                 raise ValueError(f"Missing critical model args in checkpoint 'config': {model_args}")
        elif 'model_args' in checkpoint:
             logger.info("Loading model args from 'model_args' dictionary in checkpoint.")
             model_args = checkpoint['model_args']
        else:
             raise KeyError("'model_args' or 'config' dictionary not found in checkpoint.")

        # --- Vocab Size Check ---
        # Ensure model's vocab size matches tokenizer's
        if model_args.get('vocab_size') != tokenizer.vocab_size:
             logger.warning(f"Checkpoint vocab size ({model_args.get('vocab_size')}) differs from tokenizer ({tokenizer.vocab_size}). Using tokenizer's size.")
             model_args['vocab_size'] = tokenizer.vocab_size

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        logger.info(f"Model structure created with {model.get_num_params()/1e6:.2f}M parameters.")

    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error recreating model structure from checkpoint args: {e}", exc_info=True)
        return None, tokenizer
    except Exception as e:
        logger.error(f"Unexpected error creating model structure: {e}", exc_info=True)
        return None, tokenizer


    # --- Load Model State Dict ---
    try:
        state_dict = checkpoint['model']
        # Clean up potential DDP/compile prefixes
        unwanted_prefixes = ['_orig_mod.', '_module.', 'module.']
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            original_k = k
            for prefix in unwanted_prefixes:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned_state_dict[k] = v
            # if original_k != k:
            #     logger.debug(f"Removed prefix from state dict key: {original_k} -> {k}")

        # Adjust for potential size mismatches (e.g., vocab expansion)
        model.load_state_dict(cleaned_state_dict, strict=False) # Use strict=False initially
        logger.info("Model state_dict loaded successfully (strict=False).")

        # Perform a stricter check afterwards if needed, logging mismatches
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=True)
        if missing_keys:
             logger.warning(f"State dict check: Missing keys: {missing_keys}")
        if unexpected_keys:
             logger.warning(f"State dict check: Unexpected keys: {unexpected_keys}")


    except KeyError:
        logger.error("Error: 'model' state dict not found in checkpoint.")
        return None, tokenizer
    except Exception as e:
        logger.error(f"Error loading model state_dict: {e}", exc_info=True)
        return None, tokenizer

    model.eval() # Set to evaluation mode
    model.to(device) # Move model to the target device
    logger.info(f"Model loaded to {device} and set to eval mode.")

    # Compile the model (optional, PyTorch 2.0+)
    # Note: Compilation might add overhead for single/few batch generation
    # Consider disabling compilation if sampling speed is critical and batches are small
    # if torch.__version__.startswith("2."):
    #     logger.info("Compiling the model...")
    #     try:
    #         model = torch.compile(model)
    #         logger.info("Model compiled.")
    #     except Exception as e:
    #         logger.warning(f"Model compilation failed: {e}. Proceeding without compilation.")
    # else:
    #     logger.info("Torch version < 2.0, skipping model compilation.")


    # Note: The original script converted to Hugging Face format.
    # This is generally NOT needed if using the original model's generate method.
    # If the model class itself doesn't have a .generate(), you might need a HF wrapper.
    # Assuming the provided GPT model class *does* have a .generate() method compatible
    # with the arguments used below.

    return model, tokenizer


# --- REFACTORED FUNCTION for Train Script ---
@torch.no_grad() # Ensure no gradients are calculated during generation/parsing
def generate_and_parse_aigs(
    model: torch.nn.Module, # Use base Module type hint
    tokenizer: PreTrainedTokenizerBase, # Use base Tokenizer type hint
    device: torch.device,
    num_samples: int,
    batch_size: int,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    max_new_tokens: int = 512,
    parsing_mode: str = 'strict',
    seed: int = 1337 # Allow passing seed for reproducibility within the function
    ) -> List[nx.DiGraph]:
    """
    Generates sequences using the model and parses them into AIG NetworkX graphs.

    Args:
        model: The trained PyTorch model (must have a .generate method).
        tokenizer: The tokenizer.
        device: The torch device to run generation on.
        num_samples: Total number of AIGs to attempt generating and parsing.
        batch_size: Batch size for the generation process.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter (optional).
        max_new_tokens: Maximum number of tokens to generate *after* the start token.
        parsing_mode: 'strict' or 'robust' for seq_to_nxgraph.
        seed: Random seed for generation reproducibility.

    Returns:
        A list of successfully parsed NetworkX DiGraph objects.
        The list might be shorter than num_samples if parsing errors occur.
    """
    if not MODULE_IMPORTS_OK:
        logger.error("Cannot generate AIGs because module imports failed.")
        return []
    if not hasattr(model, 'generate'):
         logger.error("The provided model object does not have a 'generate' method.")
         return []

    # Set seed for this specific generation call
    torch.manual_seed(seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed_all(seed)

    model.eval() # Ensure model is in eval mode

    # --- Determine Start and End Tokens ---
    start_token = "<boc>" # Begin of Circuit token
    eos_token = "<eog>"   # End of Graph token

    try:
        start_token_id = tokenizer.convert_tokens_to_ids(start_token)
        input_ids_start = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    except KeyError:
        logger.error(f"Start token '{start_token}' not found in tokenizer vocab! Cannot generate.")
        return []

    try:
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        logger.info(f"Using EOS token '{eos_token}' (ID: {eos_token_id}).")
    except KeyError:
        logger.warning(f"EOS token '{eos_token}' not found in tokenizer. Using tokenizer's default EOS or PAD.")
        # Fallback to tokenizer's default eos_token_id if available, otherwise pad_token_id
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        if eos_token_id is None:
             logger.error("Neither specific EOS token nor tokenizer default EOS/PAD token found. Generation might not terminate.")
             # Consider raising an error or returning empty list if EOS is critical
             return []
        logger.warning(f"Using fallback EOS/PAD token ID: {eos_token_id}")


    # --- Generation Loop ---
    generated_sequences_text = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    logger.info(f"Starting generation for {num_samples} samples in {num_batches} batches (batch size: {batch_size})...")

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(generated_sequences_text))
        if current_batch_size <= 0:
            break # Should not happen with correct loop logic, but safe check

        logger.info(f"Generating batch {i+1}/{num_batches} (size {current_batch_size})...")
        batch_start_ids = input_ids_start.repeat(current_batch_size, 1)

        # Generate sequence IDs
        # Note: model.generate needs to handle attention mask implicitly or explicitly if needed
        # The base GPT model's generate might not use attention_mask in the same way as HF's
        # Ensure the model's generate method signature matches these arguments.
        output_ids = model.generate(
            batch_start_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id
            # Removed pad_token_id, let generate handle padding/stopping via eos_token_id
            # Removed do_sample=True, assuming model.generate does sampling by default if temp > 0
        ) # Shape: (batch_size, sequence_length)

        # Decode the generated IDs into strings
        # Slice generated IDs to remove the input start token ID: output_ids[:, input_ids_start.shape[1]:]
        # Decode, skipping special tokens like padding *during decoding* if desired,
        # but keep BOC/EOG for parsing if seq_to_nxgraph expects them.
        # Let's keep special tokens for now as seq_to_nxgraph might need them.
        seq_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        generated_sequences_text.extend(seq_strs)
        logger.info(f"Batch {i+1} generated {len(seq_strs)} sequences.")

    logger.info(f"Finished generating {len(generated_sequences_text)} raw sequences.")

    # --- Parse Sequences to Graphs ---
    logger.info(f"Parsing {len(generated_sequences_text)} sequences into AIG DiGraphs (mode: {parsing_mode})...")
    generated_graphs: List[nx.DiGraph] = []
    num_parsing_errors = 0

    for i, seq_str in enumerate(generated_sequences_text):
        try:
            # Remove potential padding tokens before parsing, but keep BOC/EOG etc.
            # This depends heavily on what seq_to_nxgraph expects.
            # Assuming seq_to_nxgraph handles the full string including special tokens.
            # clean_seq = seq_str.replace(tokenizer.pad_token, "").strip() # Example cleaning

            graph = seq_to_nxgraph(seq_str, parsing_mode=parsing_mode)

            if isinstance(graph, nx.DiGraph): # Check specifically for DiGraph
                generated_graphs.append(graph)
            else:
                logger.warning(f"seq_to_nxgraph did not return a NetworkX DiGraph for sequence {i}. Got {type(graph)}. Sequence (start): {seq_str[:100]}...")
                num_parsing_errors += 1
        except Exception as e:
            logger.error(f"Error parsing sequence {i} to AIG: {e}\nSequence (start): {seq_str[:100]}...", exc_info=False) # Set exc_info=True for full traceback
            num_parsing_errors += 1

    logger.info(f"Successfully parsed {len(generated_graphs)} graphs.")
    if num_parsing_errors > 0:
        logger.warning(f"Encountered {num_parsing_errors} errors during parsing.")

    return generated_graphs
# --- END REFACTORED FUNCTION ---


# --- Main Execution Block (for standalone script) ---
if __name__ == '__main__':
    if not MODULE_IMPORTS_OK:
        exit(1) # Exit if imports failed

    args = parse_args()
    logger.info("--- Starting AIG Sampling Script (Standalone) ---")
    logger.info(f"Arguments: {vars(args)}") # Log arguments as dict

    # Ensure the output directory exists (relative to the script's execution path)
    # If out_dir is intended to be relative to the script location, fine.
    # If it's absolute, this is also fine.
    # If it's relative to *data*, adjust path construction.
    abs_out_dir = os.path.abspath(args.out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)
    logger.info(f"Output directory: {abs_out_dir}")


    device, ctx = setup_device_and_ctx(args.seed, args.device)

    # Load Tokenizer and Model
    model, tokenizer = load_model_and_tokenizer(args.out_dir, args.tokenizer_path, device)

    if model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer. Exiting.")
        exit(1)

    # --- Generate and Parse ---
    with ctx: # Use autocast context for generation
        generated_graphs = generate_and_parse_aigs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            parsing_mode=args.parsing_mode,
            seed=args.seed
        )

    # --- Reporting ---
    print("\n--- AIG Generation & Parsing Summary ---")
    print(f"Target samples            : {args.num_samples}")
    print(f"Successfully parsed graphs: {len(generated_graphs)}")
    print("--------------------------------------")

    # --- Saving Results ---
    output_file_path = os.path.join(abs_out_dir, args.output_filename)
    logger.info(f"Saving {len(generated_graphs)} generated AIG DiGraphs to {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(generated_graphs, f)
        logger.info("Graphs saved successfully.")
    except Exception as e:
        logger.error(f"Error saving pickle file: {e}", exc_info=True)

    logger.info("--- AIG Sampling Script Finished ---")
