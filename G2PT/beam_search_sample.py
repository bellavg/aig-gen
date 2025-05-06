# G2PT/beam_search_sample.py
# MODIFIED: Added support for both normal (HF) and constrained beam search.
# MODIFIED: Added options for multinomial sampling and diversity penalty to normal beam search.

"""
Sample from a trained AIG model using Beam Search.
Supports both standard beam search (via Hugging Face) and
a custom constrained beam search enforcing AIG validity rules.
Normal beam search can optionally use multinomial sampling or diversity penalties.
"""
import os
import argparse
import json
import heapq
import re
import pickle
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any, Optional
from contextlib import nullcontext
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from constrained_beam import generate_constrained_beam
# Local imports (ensure these are accessible)
try:
    # Need original model definition to load checkpoint before HF conversion
    from model import GPTConfig, GPT
    # Need the sequence-to-graph parser for AIGs
    from datasets_utils import seq_to_nxgraph
    from configs.aig import vocab_size
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"Error importing local modules (GPT, seq_to_nxgraph): {e}")
    print("Ensure sample.py is run from the G2PT directory or PYTHONPATH is set correctly.")
    MODULE_IMPORTS_OK = False

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("G2PT_AIG_BeamSampler")



# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained AIG model using Beam Search (Normal or Constrained)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing model checkpoint (e.g., results/aig-small-topo)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to tokenizer (e.g., datasets/aig/tokenizer)')
    parser.add_argument('--input_checkpoint', type=str, required=True,
                        help='Name/path for the input model checkpoint')
    parser.add_argument('--search_type', type=str, default='normal', choices=['normal', 'constrained'],
                        help='Type of beam search to perform: "normal" (Hugging Face default) or "constrained" (AIG rules)')
    parser.add_argument('--num_samples', type=int, default=10, # Default for beam search
                        help='Number of valid samples to generate (or sequences for normal beam search)')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Number of beams to maintain during search (num_beams)')
    parser.add_argument('--max_new_tokens', type=int, default=768, # Max length of generated part
                        help='Maximum number of new tokens to generate per sequence')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (used if do_sample_beam is True, or for constrained search)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--output_filename', type=str, default='generated_aigs_beam.pkl',
                        help='Name for the output pickle file')
    parser.add_argument('--parsing_mode', type=str, default='robust', choices=['strict', 'robust'],
                        help='Final sequence parsing mode for seq_to_nxgraph')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cpu", "cuda", "cuda:0"). Auto-detects if None.')

    # --- NEW ARGUMENTS for Normal Beam Search Options ---
    parser.add_argument('--do_sample_beam', action='store_true', # Use flag to enable
                        help='Enable multinomial sampling within the normal beam search (do_sample=True). Default is False.')
    parser.add_argument('--num_beam_groups', type=int, default=1,
                        help='Number of groups for diverse beam search (for normal search type). Default is 1 (no grouping). beam_size must be divisible by this.')
    parser.add_argument('--diversity_penalty', type=float, default=0.0,
                        help='Penalty for similar beams in diverse beam search (for normal search type). Default is 0.0 (no penalty).')
    # ----------------------------------------------------

    return parser.parse_args()

# --- Device Setup ---
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

# --- Model Loading ---
def load_model_and_tokenizer(ckpt_path, tokenizer_path, device):
    """
    Loads the tokenizer and the model checkpoint.
    Converts the loaded model to Hugging Face format for generation.
    """
    # --- Load Tokenizer ---
    try:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, legacy=False)
        # Ensure PAD token exists (HF generate requires it)
        if tokenizer.pad_token is None:
             logger.warning("Tokenizer missing PAD token. Adding '<|pad|>' as pad_token.")
             tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}, PAD ID: {tokenizer.pad_token_id}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {tokenizer_path}: {e}", exc_info=True)
        return None, None


    logger.info(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)
        return None, tokenizer

    # --- Initialize Original Model Structure ---
    try:
        if 'model_args' in checkpoint: model_args = checkpoint['model_args']
        elif 'config' in checkpoint:
             logger.warning("Loading model args from 'config' dictionary/object in checkpoint.")
             cfg_dict = vars(checkpoint['config']) if not isinstance(checkpoint['config'], dict) else checkpoint['config']
             model_args = {'n_layer': cfg_dict.get('n_layer'), 'n_head': cfg_dict.get('n_head'), 'n_embd': cfg_dict.get('n_embd'),
                           'block_size': cfg_dict.get('block_size'), 'bias': cfg_dict.get('bias', False),
                           'vocab_size': cfg_dict.get('vocab_size'), 'dropout': cfg_dict.get('dropout', 0.0)}
             if None in [model_args['n_layer'], model_args['n_head'], model_args['n_embd'], model_args['block_size'], model_args['vocab_size']]:
                 raise ValueError(f"Missing critical model args in checkpoint 'config': {model_args}")
        else: raise KeyError("'model_args' or 'config' dictionary/object not found in checkpoint.")

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

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
        logger.info("Successfully converted model to Hugging Face format.")
    except Exception as e:
        logger.error(f"Error converting model to Hugging Face format: {e}", exc_info=True)
        return None, tokenizer

    # --- Final Setup ---
    hf_model.eval()
    hf_model.to(device)
    logger.info(f"HF Model loaded to {device} and set to eval mode.")
    logger.info(f"Model parameter count: {sum(p.numel() for p in hf_model.parameters())/1e6:.2f}M")

    return hf_model, tokenizer

# --- Token ID Mappings (for Constrained Search) ---
def get_token_ids(tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """Loads vocab and extracts relevant token IDs needed for constrained search."""
    vocab = tokenizer.get_vocab()
    token_ids = {}
    try:
        # --- Essential Structure Tokens ---
        token_ids['boc_id'] = vocab["<boc>"]
        token_ids['eoc_id'] = vocab["<eoc>"]
        token_ids['bog_id'] = vocab["<bog>"]
        token_ids['eog_id'] = vocab["<eog>"]
        token_ids['sepc_id'] = vocab["<sepc>"]
        token_ids['sepg_id'] = vocab["<sepg>"]
        token_ids['pad_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1 # Handle padding

        # --- Node/Edge/Index Types (Sets for faster lookups) ---
        token_ids['node_type_ids'] = {vocab[tok]: tok for tok in vocab if tok.startswith("NODE_")}
        token_ids['edge_type_ids'] = {vocab[tok]: tok for tok in vocab if tok.startswith("EDGE_")}
        token_ids['idx_token_ids'] = {vocab[tok]: tok for tok in vocab if tok.startswith("IDX_")}

        token_ids['node_type_id_set'] = set(token_ids['node_type_ids'].keys())
        token_ids['edge_type_id_set'] = set(token_ids['edge_type_ids'].keys())
        token_ids['idx_token_id_set'] = set(token_ids['idx_token_ids'].keys())

        # --- Specific Node Types for Constraints ---
        token_ids['NODE_PI_ID'] = vocab.get("NODE_PI")
        token_ids['NODE_AND_ID'] = vocab.get("NODE_AND")
        token_ids['NODE_CONST0_ID'] = vocab.get("NODE_CONST0")
        token_ids['NODE_PO_ID'] = vocab.get("NODE_PO")
        # Validate essential node types
        if None in [token_ids['NODE_PI_ID'], token_ids['NODE_AND_ID'], token_ids['NODE_CONST0_ID'], token_ids['NODE_PO_ID']]:
             raise KeyError("One or more required NODE types (PI, AND, CONST0, PO) not found in vocab.")

        # --- Combined Sets for Masking ---
        token_ids['special_token_ids'] = {
            token_ids['boc_id'], token_ids['eoc_id'], token_ids['bog_id'],
            token_ids['eog_id'], token_ids['sepc_id'], token_ids['sepg_id']
        }
        token_ids['valid_token_ids'] = (token_ids['special_token_ids'] |
                                       token_ids['node_type_id_set'] |
                                       token_ids['edge_type_id_set'] |
                                       token_ids['idx_token_id_set'])
        if token_ids['pad_id'] != -1: # Add pad_id if it's valid
             token_ids['valid_token_ids'].add(token_ids['pad_id'])

        # --- Constraint Maps & Patterns ---
        token_ids['max_in_degree_map'] = {
            token_ids['NODE_PI_ID']: 0,
            token_ids['NODE_CONST0_ID']: 0,
            token_ids['NODE_AND_ID']: 2,
            token_ids['NODE_PO_ID']: 1 # PO must have exactly one input
        }
        token_ids['max_out_degree_map'] = {
            token_ids['NODE_PO_ID']: 0 # PO cannot have outputs
        }
        token_ids['idx_pattern'] = re.compile(r'IDX_(\d+)') # Regex for parsing IDX tokens

    except KeyError as e:
        logger.error(f"Error: Token '{e}' not found in tokenizer vocabulary. Check vocab.json.")
        return None # Indicate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing token IDs: {e}", exc_info=True)
        return None

    return token_ids



# --- Normal Beam Search Generation Function (using HF) ---
@torch.no_grad()
def generate_normal_beam(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    prompt: str = "<boc>",
    beam_size: int = 5,
    num_return_sequences: int = 1,
    max_new_tokens: int = 768,
    temperature: float = 1.0,
    do_sample: bool = False, # NEW parameter
    num_beam_groups: int = 1, # NEW parameter
    diversity_penalty: float = 0.0 # NEW parameter
) -> List[List[int]]:
    """
    Generates sequences using standard beam search via Hugging Face `generate`.
    Can optionally use multinomial sampling or diverse beam search options.
    Returns a list of token ID lists.
    """
    model.eval()
    search_mode = "sampling" if do_sample else "deterministic"
    diversity_mode = "diverse" if num_beam_groups > 1 and diversity_penalty > 0.0 else "standard"
    logger.info(f"Starting normal beam search (mode: {search_mode}, diversity: {diversity_mode}, num_beams={beam_size}, num_return={num_return_sequences})...")

    # --- Input Preparation ---
    try:
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_length = input_ids.shape[1]
        max_length = prompt_length + max_new_tokens
    except Exception as e:
        logger.error(f"Error preparing inputs for normal beam search: {e}", exc_info=True)
        return []

    # --- Get EOS and PAD IDs ---
    eos_token_str = "<eog>"
    try: eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_str)
    except KeyError: eos_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is None:
         logger.error("Tokenizer needs PAD token for HF generate.")
         return []

    # --- Validate Diverse Beam Search Params ---
    use_diversity = num_beam_groups > 1 and diversity_penalty > 0.0
    if use_diversity and beam_size % num_beam_groups != 0:
        logger.error(f"For diverse beam search, num_beams ({beam_size}) must be divisible by num_beam_groups ({num_beam_groups}).")
        return []

    # --- Configure Generation ---
    generation_config = GenerationConfig(
        max_length=max_length,
        num_beams=beam_size,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample, # Use the passed parameter
        num_return_sequences=min(num_return_sequences, beam_size),
        early_stopping=True, # Generally good for beam search
        temperature=temperature, # Pass temperature
        # --- Add Diversity Params ---
        num_beam_groups=num_beam_groups if use_diversity else 1,
        diversity_penalty=diversity_penalty if use_diversity else 0.0,
    )

    # --- Generate ---
    try:
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        # Output shape: (num_return_sequences, sequence_length)
        logger.info(f"Normal beam search generated {output_ids.shape[0]} sequences.")
        # Return list of lists (token IDs)
        return output_ids.tolist()
    except Exception as e:
        logger.error(f"Error during normal beam search generation: {e}", exc_info=True)
        return []


# --- Main Execution Block ---
if __name__ == '__main__':
    if not MODULE_IMPORTS_OK:
        logger.error("Exiting due to failed local imports.")
        exit(1)

    args = parse_args()
    logger.info(f"--- Starting AIG Beam Search Sampling Script ({args.search_type}) ---")
    logger.info(f"Arguments: {vars(args)}")

    # --- Setup ---
    device, ctx = setup_device_and_ctx(args.seed, args.device)
    abs_out_dir = os.path.abspath(args.out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)
    logger.info(f"Output directory: {abs_out_dir}")

    # --- Load Model and Tokenizer ---
    model, tokenizer = load_model_and_tokenizer(args.input_checkpoint, args.tokenizer_path, device)
    if model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer. Exiting.")
        exit(1)

    # --- Select and Run Beam Search ---
    generated_sequences_text = []
    if args.search_type == 'normal':
        logger.info("Running NORMAL beam search...")
        with ctx:
            output_token_ids_list = generate_normal_beam(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt="<boc>",
                beam_size=args.beam_size,
                num_return_sequences=args.num_samples, # Ask for num_samples sequences
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                # --- Pass new args ---
                do_sample=args.do_sample_beam,
                num_beam_groups=args.num_beam_groups,
                diversity_penalty=args.diversity_penalty
                # ---------------------
            )
        # Decode the list of token ID lists
        generated_sequences_text = tokenizer.batch_decode(output_token_ids_list, skip_special_tokens=False)

    elif args.search_type == 'constrained':
        logger.info("Running CONSTRAINED beam search...")
        # Load token IDs needed for constraints
        token_ids = get_token_ids(tokenizer)
        if token_ids is None:
            logger.error("Failed to get token IDs for constrained search. Exiting.")
            exit(1)

        # Constrained search generates one sequence at a time, loop if needed
        all_beams = []
        num_generated_ok = 0
        attempts = 0
        max_attempts = args.num_samples * max(5, args.beam_size*2) # Limit attempts

        while num_generated_ok < args.num_samples and attempts < max_attempts:
             attempts += 1
             logger.info(f"Constrained search attempt {attempts}/{max_attempts} for sample {num_generated_ok+1}/{args.num_samples}")
             with ctx:
                  # Generate beams (returns list of (log_prob, sequence_ids))
                  beams = generate_constrained_beam(
                      model=model,
                      tokenizer=tokenizer,
                      token_ids=token_ids,
                      device=device,
                      prompt="<boc>",
                      beam_size=args.beam_size,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                  )
             if beams:
                  # Take the best beam found in this attempt
                  best_log_prob, best_sequence_ids = beams[0]
                  seq_str = tokenizer.decode(best_sequence_ids, skip_special_tokens=False)
                  # Basic check for completeness before adding
                  if seq_str.endswith("<eog>") and "<bog>" in seq_str and "<eoc>" in seq_str:
                       generated_sequences_text.append(seq_str)
                       num_generated_ok += 1
                       logger.info(f"Found valid sequence {num_generated_ok}/{args.num_samples}. LogProb: {best_log_prob:.2f}")
                  # else: logger.debug("Constrained search result did not seem complete.")
             # else: logger.debug("Constrained search attempt yielded no beams.")
             # Add slight variation to seed for next attempt if needed?
             # torch.manual_seed(args.seed + attempts) # Optional: vary seed

        if num_generated_ok < args.num_samples:
             logger.warning(f"Constrained search finished early. Found {num_generated_ok}/{args.num_samples} sequences after {attempts} attempts.")

    else:
        logger.error(f"Unknown search_type: {args.search_type}")
        exit(1)

    logger.info(f"Finished generation. Total sequences obtained: {len(generated_sequences_text)}")

    # --- Convert Sequences to AIG DiGraphs ---
    logger.info(f"Parsing {len(generated_sequences_text)} sequences into AIG DiGraphs (mode: {args.parsing_mode})...")
    generated_graphs = []
    num_parsing_errors = 0

    for i, seq_str in enumerate(generated_sequences_text):
        try:
            graph = seq_to_nxgraph(seq_str, parsing_mode=args.parsing_mode)
            if isinstance(graph, nx.DiGraph):
                generated_graphs.append(graph)
            else:
                logger.warning(f"seq_to_nxgraph did not return a DiGraph for sequence {i}. Got {type(graph)}. Seq: {seq_str[:100]}...")
                num_parsing_errors += 1
        except Exception as e:
            logger.error(f"Error parsing sequence {i} to AIG: {e}\nSequence: {seq_str[:150]}...", exc_info=False)
            num_parsing_errors += 1

    # --- Reporting ---
    print(f"\n--- AIG Beam Search ({args.search_type}) Generation Summary ---")
    print(f"Target samples            : {args.num_samples}")
    print(f"Sequences generated       : {len(generated_sequences_text)}")
    print(f"Successfully parsed graphs: {len(generated_graphs)}")
    print(f"Errors during parsing     : {num_parsing_errors}")
    print("----------------------------------------------------")

    # --- Saving Results ---
    # --- MODIFIED: Construct dynamic filename ---
    base_name, _ = os.path.splitext(args.output_filename)
    search_tag = args.search_type
    beam_tag = f"_k{args.beam_size}"
    sampling_tag = "_sample" if args.search_type == 'normal' and args.do_sample_beam else ""
    diversity_tag = f"_diverseG{args.num_beam_groups}P{args.diversity_penalty:.1f}" if args.search_type == 'normal' and args.num_beam_groups > 1 and args.diversity_penalty > 0.0 else ""
    temp_tag = f"_t{args.temperature:.1f}" if (args.search_type == 'constrained' or (
                args.search_type == 'normal' and args.do_sample_beam)) else ""  # Add temp if sampling or constrained

    final_filename = f"{base_name}_{search_tag}{beam_tag}{sampling_tag}{diversity_tag}{temp_tag}.pkl"
    output_file_path = os.path.join(abs_out_dir, final_filename)
    # --- End Filename Modification ---

    logger.info(f"Saving {len(generated_graphs)} generated AIG DiGraphs to {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(generated_graphs, f)
        logger.info("Graphs saved successfully.")
    except Exception as e:
        logger.error(f"Error saving pickle file: {e}", exc_info=True)

    logger.info(f"--- AIG Beam Search ({args.search_type}) Sampling Script Finished ---")
