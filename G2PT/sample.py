"""
Sample from a trained model - AIG Generation Only Version
Uses Hugging Face generate method after converting the custom model.
Refactored for use in training script.
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
import networkx as nx # Needed for type hint and checking return type
# Use HF types for model and tokenizer
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from typing import List, Optional, Tuple # For type hints

# --- Module Imports ---
# Assume model.py and datasets_utils.py are in the same G2PT directory or accessible via PYTHONPATH
try:
    from model import GPTConfig, GPT # Still needed to load the original model config/weights
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
logger = logging.getLogger("g2pt_sampler_hf_reverted") # Updated logger name


def parse_args():
    """Parses command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description='Sample from a trained G2PT model for AIGs using HF generate')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing model checkpoint (e.g., results/aig-small-topo/ckpt.pt)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to the trained tokenizer directory (e.g., G2PT/datasets/aig/tokenizer)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (e.g., 0.8 for less randomness, 1.0 for standard)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling parameter (optional)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate per sequence')
    parser.add_argument('--output_filename', type=str, default='generated_aigs_hf_reverted.pkl', # Changed default name
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
        if torch.cuda.is_bf16_supported(): dtype_str = 'bfloat16'
        else: dtype_str = 'float16'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype_str = 'float32'
    logger.info(f"Selected precision mode: {dtype_str}")

    # Seed setting
    torch.manual_seed(seed)
    np.random.seed(seed)
    if 'cuda' in device: torch.cuda.manual_seed_all(seed)

    # Autocast context
    device_type = 'cuda' if 'cuda' in device else ('mps' if device == 'mps' else 'cpu')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype, enabled=(device_type != 'cpu'))
    logger.info(f"Autocast context: enabled={device_type != 'cpu'}, device_type='{device_type}', dtype='{ptdtype}'")

    return device, ctx

def load_model_and_tokenizer(out_dir, tokenizer_path, device):
    """Loads the model checkpoint, tokenizer, and converts model to HF format."""
    # --- Load Tokenizer ---
    try:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, legacy=False)
        # --- Add PAD token if missing ---
        if tokenizer.pad_token is None:
             logger.warning("Tokenizer missing PAD token. Adding '<|pad|>' as pad_token.")
             tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, PAD ID: {tokenizer.pad_token_id}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {tokenizer_path}: {e}", exc_info=True)
        return None, None

    # --- Load Model Checkpoint ---
    ckpt_path_options = [os.path.join(out_dir, 'ckpt.pt'), os.path.join(out_dir, 'best.pt')]
    ckpt_path = next((path for path in ckpt_path_options if os.path.exists(path)), None)

    if ckpt_path is None:
        logger.error(f"Error: No checkpoint file (ckpt.pt or best.pt) found in {out_dir}")
        return None, tokenizer

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)
        return None, tokenizer

    # --- Initialize Original Model ---
    try:
        if 'model_args' in checkpoint: model_args = checkpoint['model_args']
        elif 'config' in checkpoint:
             logger.warning("Loading model args from 'config' dictionary in checkpoint.")
             cfg_dict = vars(checkpoint['config']) if not isinstance(checkpoint['config'], dict) else checkpoint['config']
             model_args = {'n_layer': cfg_dict.get('n_layer'), 'n_head': cfg_dict.get('n_head'), 'n_embd': cfg_dict.get('n_embd'),
                           'block_size': cfg_dict.get('block_size'), 'bias': cfg_dict.get('bias', False),
                           'vocab_size': cfg_dict.get('vocab_size'), 'dropout': cfg_dict.get('dropout', 0.0)}
             if None in [model_args['n_layer'], model_args['n_head'], model_args['n_embd'], model_args['block_size'], model_args['vocab_size']]:
                 raise ValueError(f"Missing critical model args in checkpoint 'config': {model_args}")
        else: raise KeyError("'model_args' or 'config' dictionary not found in checkpoint.")

        # --- Vocab Size Check/Update ---
        # Use tokenizer vocab size for model init if different from checkpoint
        if model_args.get('vocab_size') != tokenizer.vocab_size:
             logger.warning(f"Checkpoint vocab size ({model_args.get('vocab_size')}) differs from tokenizer ({tokenizer.vocab_size}). Initializing model with tokenizer's size.")
             model_args['vocab_size'] = tokenizer.vocab_size

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf) # Initialize original model structure

    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error recreating model structure from checkpoint args: {e}", exc_info=True)
        return None, tokenizer
    except Exception as e:
        logger.error(f"Unexpected error creating model structure: {e}", exc_info=True)
        return None, tokenizer

    # --- Load State Dict into Original Model ---
    try:
        state_dict = checkpoint['model']
        unwanted_prefixes = ['_orig_mod.', '_module.', 'module.']
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            original_k = k; key_modified = False
            for prefix in unwanted_prefixes:
                if k.startswith(prefix): cleaned_state_dict[k[len(prefix):]] = v; key_modified = True; break
            if not key_modified: cleaned_state_dict[k] = v

        # Load cleaned state dict into the original model structure
        load_result = model.load_state_dict(cleaned_state_dict, strict=False)
        logger.info("Original model state_dict loaded.")
        if load_result.missing_keys: logger.warning(f"State dict check (Original Model): Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys: logger.warning(f"State dict check (Original Model): Unexpected keys: {load_result.unexpected_keys}")

    except Exception as e:
        logger.error(f"Error loading state_dict into original model: {e}", exc_info=True)
        return None, tokenizer

    # --- Convert to Hugging Face Model ---
    try:
        logger.info("Converting model to Hugging Face format...")
        hf_model = model.to_hf()
        # --- Resize Embeddings if PAD token was added ---
        if hf_model.config.vocab_size != tokenizer.vocab_size:
             logger.warning(f"Resizing HF model embeddings from {hf_model.config.vocab_size} to {tokenizer.vocab_size} due to added PAD token.")
             hf_model.resize_token_embeddings(len(tokenizer))
             # Ensure the underlying config also reflects this
             hf_model.config.vocab_size = len(tokenizer)

        logger.info("Successfully converted model to Hugging Face format.")
    except Exception as e:
        logger.error(f"Error converting model to Hugging Face format: {e}", exc_info=True)
        return None, tokenizer # Return None for model if conversion fails

    hf_model.eval() # Set HF model to evaluation mode
    hf_model.to(device) # Move HF model to the target device
    logger.info(f"HF Model loaded to {device} and set to eval mode.")

    # Return the Hugging Face model and the tokenizer
    return hf_model, tokenizer


# --- REFACTORED FUNCTION for Train Script ---
@torch.no_grad() # Ensure no gradients are calculated during generation/parsing
def generate_and_parse_aigs(
    model: PreTrainedModel, # Expect HF PreTrainedModel
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    max_new_tokens: int = 512,
    parsing_mode: str = 'strict',
    seed: int = 1337
    ) -> List[nx.DiGraph]:
    """
    Generates sequences using the HF model and parses them into AIG NetworkX graphs.

    Args:
        model: The trained model in Hugging Face format.
        tokenizer: The tokenizer.
        device: The torch device to run generation on.
        num_samples: Total number of AIGs to attempt generating and parsing.
        batch_size: Batch size for the generation process.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter (optional).
        max_new_tokens: Maximum number of new tokens to generate *after* the start token.
        parsing_mode: 'strict' or 'robust' for seq_to_nxgraph.
        seed: Random seed for generation reproducibility.

    Returns:
        A list of successfully parsed NetworkX DiGraph objects.
    """
    if not MODULE_IMPORTS_OK:
        logger.error("Cannot generate AIGs because module imports failed.")
        return []
    # No need to check for model.generate, HF models have it

    # Set seed for this specific generation call using a Generator object
    generator = torch.Generator(device=device).manual_seed(seed)

    model.eval() # Ensure model is in eval mode

    # --- Determine Start and End Tokens ---
    start_token = "<boc>" # Begin of Circuit token
    eos_token_str = "<eog>"   # End of Graph token string

    try:
        start_token_id = tokenizer.convert_tokens_to_ids(start_token)
        # Prepare inputs for HF model generate
        inputs = tokenizer([start_token], return_tensors="pt").to(device) # Prepare prompt for one sample
        input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"] # Usually inferred by HF generate
    except KeyError:
        logger.error(f"Start token '{start_token}' not found in tokenizer vocab! Cannot generate.")
        return []
    except Exception as e:
         logger.error(f"Error preparing generation inputs: {e}")
         return []

    # Get EOS token ID for stopping criteria
    try:
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_str)
        logger.info(f"Using EOS token '{eos_token_str}' (ID: {eos_token_id}) for generation stop.")
    except KeyError:
        logger.warning(f"EOS token '{eos_token_str}' not found in tokenizer. Relying on max_new_tokens or PAD.")
        eos_token_id = tokenizer.pad_token_id # Use pad as fallback stop ID

    if tokenizer.pad_token_id is None:
         logger.error("Tokenizer does not have a PAD token ID, which is required by HF generate.")
         return []


    # --- Generation Loop ---
    generated_sequences_text = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    logger.info(f"Starting generation for {num_samples} samples in {num_batches} batches (batch size: {batch_size})...")

    # Calculate max_length for HF generate
    prompt_length = input_ids.shape[1]
    max_length = prompt_length + max_new_tokens

    # Configure generation parameters using GenerationConfig for clarity
    generation_config = GenerationConfig(
        max_length=max_length,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True, # Enable sampling
        temperature=temperature,
        top_k=top_k,
        # top_p=None, # Add top_p if needed
        # num_return_sequences=1, # Default is 1
    )

    for i in range(num_batches):
        current_batch_target = min(batch_size, num_samples - len(generated_sequences_text))
        if current_batch_target <= 0: break

        logger.info(f"Generating batch {i+1}/{num_batches} (target size {current_batch_target})...")

        # Repeat the prompt for the current batch size
        batch_input_ids = input_ids.repeat(current_batch_target, 1)
        # batch_attention_mask = attention_mask.repeat(current_batch_target, 1) # If needed

        # --- Call Hugging Face model's generate method ---
        # Pass the generator object for reproducibility
        output_ids = model.generate(
            inputs=batch_input_ids, # Use 'inputs' argument
            # attention_mask=batch_attention_mask, # Can often be omitted
            generation_config=generation_config,
            generator=generator # Pass generator for reproducibility
        ) # Shape: (current_batch_target, sequence_length)
        # --- End HF Call ---

        # Decode the generated IDs into strings
        # Keep special tokens as seq_to_nxgraph might need them
        # Skip prompt tokens during decoding
        seq_strs = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=False)
        # Prepend the start token back manually if seq_to_nxgraph needs it
        full_seq_strs = [start_token + " " + s for s in seq_strs]

        generated_sequences_text.extend(full_seq_strs)
        logger.info(f"Batch {i+1} generated {len(seq_strs)} sequences.")

    logger.info(f"Finished generating {len(generated_sequences_text)} raw sequences.")

    # --- Parse Sequences to Graphs ---
    logger.info(f"Parsing {len(generated_sequences_text)} sequences into AIG DiGraphs (mode: {parsing_mode})...")
    generated_graphs: List[nx.DiGraph] = []
    num_parsing_errors = 0

    for i, seq_str in enumerate(generated_sequences_text):
        try:
            # Assuming seq_to_nxgraph handles the full string including special tokens.
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
    logger.info("--- Starting AIG Sampling Script (Using HF Generate) ---")
    logger.info(f"Arguments: {vars(args)}")

    # Ensure the output directory exists
    abs_out_dir = os.path.abspath(args.out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)
    logger.info(f"Output directory: {abs_out_dir}")

    device, ctx = setup_device_and_ctx(args.seed, args.device)

    # Load Tokenizer and HF Model
    hf_model, tokenizer = load_model_and_tokenizer(args.out_dir, args.tokenizer_path, device)

    if hf_model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer. Exiting.")
        exit(1)

    # --- Generate and Parse ---
    with ctx: # Use autocast context if on GPU
        generated_graphs = generate_and_parse_aigs(
            model=hf_model, # Pass the HF model
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
