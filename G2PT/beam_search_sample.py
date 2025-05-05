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

# Local imports (ensure these are accessible)
try:
    # Need original model definition to load checkpoint before HF conversion
    from model import GPTConfig, GPT
    # Need the sequence-to-graph parser for AIGs
    from datasets_utils import seq_to_nxgraph
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"Error importing local modules (GPT, seq_to_nxgraph): {e}")
    print("Ensure sample.py is run from the G2PT directory or PYTHONPATH is set correctly.")
    MODULE_IMPORTS_OK = False

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("G2PT_AIG_BeamSampler")


# --- Beam State (for Constrained Search) ---
@dataclass(order=True)
class BeamState:
    """Represents one beam in the constrained search."""
    log_prob: float # Total log probability (used for sorting/heapq)
    # --- Fields not used for ordering ---
    sequence: List[int] = field(compare=False) # List of token IDs
    graph_state: Dict[str, Any] = field(compare=False) # Dictionary tracking graph structure
    kv_cache: Optional[Tuple[torch.Tensor]] = field(compare=False) # Model's KV cache
    is_finished: bool = field(default=False, compare=False) # Flag if <eog> was generated

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained AIG model using Beam Search (Normal or Constrained)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing model checkpoint (e.g., results/aig-small-topo)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to tokenizer (e.g., datasets/aig/tokenizer)')
    parser.add_argument('--search_type', type=str, default='constrained', choices=['normal', 'constrained'],
                        help='Type of beam search to perform: "normal" (Hugging Face default) or "constrained" (AIG rules)')
    parser.add_argument('--num_samples', type=int, default=10, # Default for beam search
                        help='Number of valid samples to generate (or sequences for normal beam search)')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Number of beams to maintain during search (num_beams)')
    parser.add_argument('--max_new_tokens', type=int, default=768, # Max length of generated part
                        help='Maximum number of new tokens to generate per sequence')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (used if do_sample_beam is True, or for constrained search)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--output_filename', type=str, default='generated_aigs_beam.pkl',
                        help='Name for the output pickle file')
    parser.add_argument('--parsing_mode', type=str, default='strict', choices=['strict', 'robust'],
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
def load_model_and_tokenizer(out_dir, tokenizer_path, device):
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
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, PAD ID: {tokenizer.pad_token_id}")
    except Exception as e:
        logger.error(f"Error loading tokenizer from {tokenizer_path}: {e}", exc_info=True)
        return None, None

    # --- Load Model Checkpoint ---
    ckpt_path_options = [os.path.join(out_dir, 'best.pt'), os.path.join(out_dir, 'ckpt.pt')]
    ckpt_path = next((path for path in ckpt_path_options if os.path.exists(path)), None)

    if ckpt_path is None:
        logger.error(f"Error: No checkpoint file (best.pt or ckpt.pt) found in {out_dir}")
        return None, tokenizer

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

        # --- Vocab Size Check/Update ---
        if model_args.get('vocab_size') != tokenizer.vocab_size:
             logger.warning(f"Checkpoint vocab size ({model_args.get('vocab_size')}) differs from tokenizer ({tokenizer.vocab_size}). Initializing model with tokenizer's size.")
             model_args['vocab_size'] = tokenizer.vocab_size

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
        if hf_model.config.vocab_size != tokenizer.vocab_size:
             logger.warning(f"Resizing HF model embeddings from {hf_model.config.vocab_size} to {tokenizer.vocab_size}.")
             hf_model.resize_token_embeddings(len(tokenizer))
             hf_model.config.vocab_size = len(tokenizer)
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

# --- Helper: DAG Check (for Constrained Search) ---
def has_path(start_node: int, end_node: int, edges: Set[Tuple[int, int]]) -> bool:
    """Checks if a path exists from start_node to end_node using BFS."""
    if start_node == end_node: return True # Path to self exists if edge is added
    q = deque([start_node])
    visited = {start_node}
    adj = defaultdict(list)
    for u, v in edges: adj[u].append(v)

    while q:
        curr = q.popleft()
        for neighbor in adj.get(curr, []):
            if neighbor == end_node: return True
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return False

# --- Helper: Update Graph State (for Constrained Search) ---
def update_graph_state(current_state: Dict[str, Any],
                       token_id: int,
                       token_ids: Dict[str, Any],
                       tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """
    Updates the lightweight graph state based on the newly added token.
    Returns the *new* state dictionary. Does not modify the input state.
    """
    # Create a deep copy to avoid modifying the original state
    new_state = {
        'node_types': current_state['node_types'].copy(),
        'node_indices': current_state['node_indices'].copy(),
        'in_degree': current_state['in_degree'].copy(),
        'out_degree': current_state['out_degree'].copy(),
        'edges': current_state['edges'].copy(),
        'context': current_state['context'],
        'last_idx_token': current_state.get('last_idx_token'),
        'last_node_type_token': current_state.get('last_node_type_token'),
        'is_complete': current_state.get('is_complete', False)
    }
    current_context = new_state['context']
    token_str = tokenizer.decode([token_id]) # Get string representation

    # State machine logic (same as previous version)
    if current_context == 'EXPECTING_NODE_TYPE':
        if token_id in token_ids['node_type_id_set']:
            new_state['last_node_type_token'] = token_id
            new_state['context'] = 'EXPECTING_NODE_IDX'
        elif token_id == token_ids['eoc_id']:
            new_state['context'] = 'EXPECTING_BOG'
        else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_NODE_IDX':
        idx_match = token_ids['idx_pattern'].match(token_str)
        if idx_match and token_id in token_ids['idx_token_id_set']:
            node_index = int(idx_match.group(1))
            last_node_type = new_state.get('last_node_type_token')
            if last_node_type is not None:
                 if node_index not in new_state['node_indices']:
                      new_state['node_indices'].add(node_index)
                      new_state['node_types'][node_index] = token_ids['node_type_ids'][last_node_type]
                      new_state['in_degree'][node_index] = 0
                      new_state['out_degree'][node_index] = 0
            else: new_state['context'] = 'ERROR'; logger.debug("Constraint Violation: IDX token found without preceding NODE type.")
            new_state['last_idx_token'] = token_id
            new_state['context'] = 'EXPECTING_SEPARATOR_OR_EOC'
        else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_SEPARATOR_OR_EOC':
        if token_id == token_ids['sepc_id']: new_state['context'] = 'EXPECTING_NODE_TYPE'
        elif token_id == token_ids['eoc_id']: new_state['context'] = 'EXPECTING_BOG'
        else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_BOG':
        if token_id == token_ids['bog_id']: new_state['context'] = 'EXPECTING_EDGE_SRC_OR_EOG'
        else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_EDGE_SRC_OR_EOG':
         idx_match = token_ids['idx_pattern'].match(token_str)
         if idx_match and token_id in token_ids['idx_token_id_set']:
             new_state['last_idx_token'] = token_id; new_state['context'] = 'EXPECTING_EDGE_DST'
         elif token_id == token_ids['eog_id']:
             new_state['context'] = 'FINISHED'; new_state['is_complete'] = check_graph_completeness(new_state, token_ids)
         elif token_id == token_ids['sepg_id']: new_state['context'] = 'EXPECTING_EDGE_SRC'
         else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_EDGE_SRC':
         idx_match = token_ids['idx_pattern'].match(token_str)
         if idx_match and token_id in token_ids['idx_token_id_set']:
             new_state['last_idx_token'] = token_id; new_state['context'] = 'EXPECTING_EDGE_DST'
         else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_EDGE_DST':
        idx_match_dst = token_ids['idx_pattern'].match(token_str)
        if idx_match_dst and token_id in token_ids['idx_token_id_set']:
            new_state['last_dst_idx_token'] = token_id; new_state['context'] = 'EXPECTING_EDGE_TYPE'
        else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_EDGE_TYPE':
        if token_id in token_ids['edge_type_id_set']:
            src_token_id = new_state.get('last_idx_token'); dst_token_id = new_state.get('last_dst_idx_token')
            src_token_str = tokenizer.decode([src_token_id]) if src_token_id else None
            dst_token_str = tokenizer.decode([dst_token_id]) if dst_token_id else None
            src_match = token_ids['idx_pattern'].match(src_token_str) if src_token_str else None
            dst_match = token_ids['idx_pattern'].match(dst_token_str) if dst_token_str else None
            if src_match and dst_match:
                u = int(src_match.group(1)); v = int(dst_match.group(1))
                if u in new_state['node_indices'] and v in new_state['node_indices']:
                     new_state['edges'].add((u, v)); new_state['out_degree'][u] += 1; new_state['in_degree'][v] += 1
                     new_state['context'] = 'EXPECTING_SEPARATOR_OR_EOG'
                else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Edge nodes {u} or {v} not defined.")
            else: new_state['context'] = 'ERROR'; logger.debug("Constraint Violation: Could not parse source/dest tokens for edge.")
        else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context == 'EXPECTING_SEPARATOR_OR_EOG':
        if token_id == token_ids['sepg_id']: new_state['context'] = 'EXPECTING_EDGE_SRC'
        elif token_id == token_ids['eog_id']:
            new_state['context'] = 'FINISHED'; new_state['is_complete'] = check_graph_completeness(new_state, token_ids)
        else: new_state['context'] = 'ERROR'; logger.debug(f"Constraint Violation: Unexpected token {token_str} in context {current_context}")

    elif current_context in ['FINISHED', 'ERROR']: pass # No state change
    else: new_state['context'] = 'ERROR'; logger.error(f"Internal Error: Unknown context '{current_context}'")

    # Clean up temporary state vars
    new_state.pop('last_dst_idx_token', None)
    if new_state['context'] != 'EXPECTING_EDGE_DST': new_state.pop('last_idx_token', None)
    if new_state['context'] != 'EXPECTING_NODE_IDX': new_state.pop('last_node_type_token', None)

    return new_state

# --- Helper: Check Graph Completeness (for Constrained Search) ---
def check_graph_completeness(graph_state: Dict[str, Any], token_ids: Dict[str, Any]) -> bool:
    """Checks if the graph state meets basic AIG structural completeness criteria."""
    node_types = graph_state['node_types']
    in_degree = graph_state['in_degree']
    # Get node type strings from IDs for comparison
    NODE_AND_STR = token_ids['node_type_ids'].get(token_ids['NODE_AND_ID'])
    NODE_PO_STR = token_ids['node_type_ids'].get(token_ids['NODE_PO_ID'])

    # Check if minimum node counts are met (optional, but good for quality)
    # if len(node_types) < MIN_NODES_REQUIRED: return False

    for node_idx, node_type_str in node_types.items():
        # Check AND gate in-degree
        if node_type_str == NODE_AND_STR and in_degree.get(node_idx, 0) != 2:
            # logger.debug(f"Completeness Check Fail: AND node {node_idx} has in-degree {in_degree.get(node_idx, 0)} != 2")
            return False
        # Check PO gate in-degree (must be exactly 1)
        if node_type_str == NODE_PO_STR and in_degree.get(node_idx, 0) != 1:
            # logger.debug(f"Completeness Check Fail: PO node {node_idx} has in-degree {in_degree.get(node_idx, 0)} != 1")
            return False
    # Add more checks if needed (e.g., all PIs used? All nodes connected?)
    return True


# --- Helper: Calculate Constraint Mask (for Constrained Search) ---
def calculate_constraint_mask(beam_state: BeamState,
                              token_ids: Dict[str, Any],
                              vocab_size: int,
                              tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """
    Calculates the mask for invalid next tokens based on the current beam state.
    Returns a boolean tensor where True indicates an invalid token.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    graph_state = beam_state.graph_state
    context = graph_state['context']

    # 1. Mask fundamentally invalid tokens (not part of AIG vocab structure)
    all_indices = torch.arange(vocab_size)
    valid_mask = torch.tensor([idx in token_ids['valid_token_ids'] for idx in all_indices.tolist()], dtype=torch.bool)
    mask[~valid_mask] = True

    # 2. Grammar Constraints (allowed tokens based on current context)
    allowed_token_types = set()
    # --- Populate allowed_token_types based on context (same logic as previous version) ---
    if context == 'EXPECTING_NODE_TYPE':
        allowed_token_types.update(token_ids['node_type_id_set'])
        allowed_token_types.add(token_ids['eoc_id'])
    elif context == 'EXPECTING_NODE_IDX':
        allowed_token_types.update(token_ids['idx_token_id_set'])
    elif context == 'EXPECTING_SEPARATOR_OR_EOC':
        allowed_token_types.add(token_ids['sepc_id'])
        allowed_token_types.add(token_ids['eoc_id'])
    elif context == 'EXPECTING_BOG':
        allowed_token_types.add(token_ids['bog_id'])
    elif context == 'EXPECTING_EDGE_SRC_OR_EOG':
        allowed_token_types.update(token_ids['idx_token_id_set'])
        allowed_token_types.add(token_ids['eog_id'])
        allowed_token_types.add(token_ids['sepg_id'])
    elif context == 'EXPECTING_EDGE_SRC':
         allowed_token_types.update(token_ids['idx_token_id_set'])
    elif context == 'EXPECTING_EDGE_DST':
        allowed_token_types.update(token_ids['idx_token_id_set'])
    elif context == 'EXPECTING_EDGE_TYPE':
        allowed_token_types.update(token_ids['edge_type_id_set'])
    elif context == 'EXPECTING_SEPARATOR_OR_EOG':
        allowed_token_types.add(token_ids['sepg_id'])
        allowed_token_types.add(token_ids['eog_id'])
    elif context == 'FINISHED' or context == 'ERROR':
         if token_ids['pad_id'] != -1: allowed_token_types.add(token_ids['pad_id'])
    # --- End context-based allowed types ---

    # Apply grammar mask (mask everything NOT allowed by context)
    grammar_mask = torch.tensor([idx not in allowed_token_types for idx in all_indices.tolist()], dtype=torch.bool)
    mask[grammar_mask] = True

    # --- 3. AIG Specific Structural Constraints (Refine allowed tokens) ---
    node_types = graph_state['node_types']
    node_indices = graph_state['node_indices']
    in_degree = graph_state['in_degree']
    out_degree = graph_state['out_degree']
    edges = graph_state['edges']

    # --- Apply structural constraints based on context ---
    if context == 'EXPECTING_NODE_IDX':
        # Constraint: Prevent defining more nodes than MAX_NODE_COUNT allows (if IDX tokens go up that high)
        # This depends on how IDX tokens are defined. Assuming IDX_0 to IDX_{MAX-1}
        max_node_idx_allowed = -1
        for idx_id in token_ids['idx_token_id_set']:
             idx_str = token_ids['idx_token_ids'].get(idx_id)
             if idx_str:
                  match = token_ids['idx_pattern'].match(idx_str)
                  if match: max_node_idx_allowed = max(max_node_idx_allowed, int(match.group(1)))

        # Mask IDX tokens that represent nodes already defined
        for node_idx_val in node_indices:
             # Find the corresponding token ID (this could be slow, maybe precompute map)
             target_idx_str = f"IDX_{node_idx_val}"
             target_idx_id = None
             for idx_id, idx_str in token_ids['idx_token_ids'].items():
                  if idx_str == target_idx_str: target_idx_id = idx_id; break
             if target_idx_id is not None: mask[target_idx_id] = True

    elif context == 'EXPECTING_EDGE_DST':
        src_token_id = graph_state.get('last_idx_token')
        src_token_str = tokenizer.decode([src_token_id]) if src_token_id else None
        src_match = token_ids['idx_pattern'].match(src_token_str) if src_token_str else None
        if src_match:
            u = int(src_match.group(1))
            u_type_str = node_types.get(u) # Get type string (e.g., "NODE_PO")
            u_type_id = None
            for tid, tstr in token_ids['node_type_ids'].items():
                 if tstr == u_type_str: u_type_id = tid; break

            # PO Out-Degree Constraint: If source is PO, mask all destinations
            if u_type_id == token_ids['NODE_PO_ID']:
                 mask[list(token_ids['idx_token_id_set'])] = True
            else:
                # Apply constraints to potential destination nodes (v)
                for v_token_id, v_token_str in token_ids['idx_token_ids'].items():
                    if mask[v_token_id]: continue # Skip if already masked

                    v_match = token_ids['idx_pattern'].match(v_token_str)
                    if v_match:
                        v = int(v_match.group(1))
                        if v not in node_indices: mask[v_token_id] = True; continue # Dest must exist

                        v_type_str = node_types.get(v)
                        v_type_id = None
                        for tid, tstr in token_ids['node_type_ids'].items():
                             if tstr == v_type_str: v_type_id = tid; break

                        # PI/CONST0 In-Degree Constraint
                        if v_type_id in [token_ids['NODE_PI_ID'], token_ids['NODE_CONST0_ID']]:
                            mask[v_token_id] = True; continue
                        # AND In-Degree Constraint
                        if v_type_id == token_ids['NODE_AND_ID'] and in_degree.get(v, 0) >= 2:
                            mask[v_token_id] = True; continue
                        # PO In-Degree Constraint
                        if v_type_id == token_ids['NODE_PO_ID'] and in_degree.get(v, 0) >= 1:
                            mask[v_token_id] = True; continue
                        # DAG Constraint: Check if adding edge u->v creates path v->u
                        if has_path(v, u, edges):
                            mask[v_token_id] = True; continue
        else: mask[list(token_ids['idx_token_id_set'])] = True # Mask all dest if source is invalid

    elif context in ['EXPECTING_EDGE_SRC', 'EXPECTING_EDGE_SRC_OR_EOG']:
         # Constraint: Source node must exist and cannot be PO
         for u_token_id, u_token_str in token_ids['idx_token_ids'].items():
              if mask[u_token_id]: continue
              u_match = token_ids['idx_pattern'].match(u_token_str)
              if u_match:
                   u = int(u_match.group(1))
                   if u not in node_indices: mask[u_token_id] = True; continue
                   u_type_str = node_types.get(u)
                   u_type_id = None
                   for tid, tstr in token_ids['node_type_ids'].items():
                        if tstr == u_type_str: u_type_id = tid; break
                   if u_type_id == token_ids['NODE_PO_ID']: mask[u_token_id] = True; continue
              else: mask[u_token_id] = True # Invalid IDX token format

    # Termination Constraint: Mask EOG if graph is incomplete
    if context in ['EXPECTING_EDGE_SRC_OR_EOG', 'EXPECTING_SEPARATOR_OR_EOG']:
         if not check_graph_completeness(graph_state, token_ids):
              mask[token_ids['eog_id']] = True

    # --- End Structural Constraints ---

    return mask


# --- Constrained Beam Search Generation Function ---
@torch.no_grad()
def generate_constrained_beam(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Dict[str, Any],
    device: torch.device,
    prompt: str = "<boc>",
    beam_size: int = 5,
    max_new_tokens: int = 768,
    temperature: float = 1.0,
) -> List[Tuple[float, List[int]]]:
    """
    Generates sequences using constrained beam search enforcing AIG rules.
    Returns a list of (log_probability, sequence_token_ids) for completed beams.
    """
    model.eval()
    vocab_size = tokenizer.vocab_size

    # --- Initial state ---
    try:
        start_token_id = token_ids['boc_id']
    except KeyError:
        logger.error("Start token '<boc>' not found in token_ids.")
        return []
    initial_sequence = [start_token_id]
    initial_graph_state = {
        'node_types': {}, 'node_indices': set(), 'in_degree': defaultdict(int),
        'out_degree': defaultdict(int), 'edges': set(),
        'context': 'EXPECTING_NODE_TYPE', 'is_complete': False
    }
    initial_beam = BeamState(log_prob=0.0, sequence=initial_sequence, graph_state=initial_graph_state, kv_cache=None)
    active_beams = [initial_beam] # List of active BeamState objects
    finished_beams = [] # List to store finished BeamState objects

    # --- Beam Search Loop ---
    for step in range(max_new_tokens):
        if not active_beams: break # Stop if no active beams

        candidates = [] # Stores (-log_prob, beam_state) for heapq

        for beam in active_beams:
            if beam.is_finished:
                # If a beam finished previously, keep it around but don't expand
                heapq.heappush(candidates, (-beam.log_prob, beam))
                continue

            # Prepare input for this beam (only the last token)
            current_token_id = torch.tensor([[beam.sequence[-1]]], device=device)

            # --- Model Forward Pass ---
            try:
                 outputs = model(
                    input_ids=current_token_id,
                    past_key_values=beam.kv_cache,
                    use_cache=True,
                 )
                 logits = outputs.logits[:, -1, :] # Logits for the next token
                 next_kv_cache = outputs.past_key_values
            except Exception as e:
                 logger.error(f"Error during model forward pass: {e}", exc_info=True)
                 logger.error(f"Beam sequence causing error: {tokenizer.decode(beam.sequence)}")
                 continue # Skip this beam if model fails

            # Apply temperature
            logits = logits / temperature

            # --- Calculate and Apply Constraint Mask ---
            try:
                constraint_mask = calculate_constraint_mask(beam, token_ids, vocab_size, tokenizer).to(device)
                logits[constraint_mask] = -float('inf') # Mask invalid tokens
            except Exception as e:
                 logger.error(f"Error calculating constraint mask: {e}", exc_info=True)
                 logger.error(f"Beam sequence causing error: {tokenizer.decode(beam.sequence)}")
                 continue # Skip this beam if masking fails

            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # --- Select Top-k Successors ---
            # Ensure there are valid next steps
            finite_log_probs_mask = log_probs > -float('inf')
            if not finite_log_probs_mask.any():
                 # logger.debug(f"Beam reached dead end (all next tokens masked). Seq: {tokenizer.decode(beam.sequence)}")
                 continue # No valid next steps

            # Consider only beam_size successors
            num_successors = min(beam_size, int(finite_log_probs_mask.sum().item()))
            if num_successors <= 0: continue # Should not happen if check above passed

            top_log_probs, top_indices = torch.topk(log_probs, num_successors, dim=-1)

            # --- Expand Beam ---
            for j in range(num_successors):
                next_token_id = top_indices[0, j].item()
                next_log_prob = top_log_probs[0, j].item()

                # Create new sequence and calculate total log prob
                new_sequence = beam.sequence + [next_token_id]
                new_total_log_prob = beam.log_prob + next_log_prob

                # Update graph state based on the new token
                try:
                    new_graph_state = update_graph_state(beam.graph_state, next_token_id, token_ids, tokenizer)
                    # Check for immediate error state after update
                    if new_graph_state['context'] == 'ERROR':
                         # logger.debug(f"Candidate beam resulted in ERROR state. Seq: {tokenizer.decode(new_sequence)}")
                         continue # Do not add beams that immediately enter error state
                except Exception as e:
                     logger.error(f"Error updating graph state: {e}", exc_info=True)
                     logger.error(f"Sequence causing error: {tokenizer.decode(new_sequence)}")
                     continue # Skip this candidate if state update fails

                # Determine if the new beam is finished
                is_finished = (next_token_id == token_ids['eog_id'] and new_graph_state.get('is_complete', False))

                # Create the new beam state
                new_beam = BeamState(
                    log_prob=new_total_log_prob,
                    sequence=new_sequence,
                    graph_state=new_graph_state,
                    kv_cache=next_kv_cache,
                    is_finished=is_finished
                )

                # Add to candidates list (using negative log_prob for min-heap)
                heapq.heappush(candidates, (-new_beam.log_prob, new_beam))
            # --- End Beam Expansion Loop ---
        # --- End Loop Through Active Beams ---

        # --- Prune Candidates to Keep Top `beam_size` ---
        active_beams = [] # Reset active beams for this step
        seen_sequences = set() # Prevent duplicates

        processed_candidates = 0
        while candidates and processed_candidates < beam_size:
             # Pop the best candidate (lowest negative log_prob = highest log_prob)
             neg_log_prob, current_beam = heapq.heappop(candidates)

             # Check for duplicates based on sequence
             seq_tuple = tuple(current_beam.sequence)
             if seq_tuple in seen_sequences: continue
             seen_sequences.add(seq_tuple)

             # Add to finished or active list
             if current_beam.is_finished:
                  finished_beams.append(current_beam)
             else:
                  active_beams.append(current_beam)
             processed_candidates += 1

        # Check termination conditions
        if not active_beams: break # Stop if no more active beams
        # Stop if max length reached for all active beams
        # Add 1 to initial_sequence length because prompt is included in beam.sequence
        if len(active_beams[0].sequence) >= max_new_tokens + 1:
             logger.info("Max generation length reached for active beams.")
             finished_beams.extend(active_beams) # Consider unfinished beams as finished
             active_beams = [] # Stop the loop
    # --- End Beam Search Step Loop ---

    # Add any remaining active beams (that didn't finish naturally)
    finished_beams.extend(active_beams)

    # Sort finished beams by log probability (descending)
    finished_beams.sort(key=lambda b: b.log_prob, reverse=True)

    logger.info(f"Constrained beam search finished. Found {len(finished_beams)} potential sequences.")
    # Return list of (log_prob, sequence)
    return [(beam.log_prob, beam.sequence) for beam in finished_beams]

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
    model, tokenizer = load_model_and_tokenizer(args.out_dir, args.tokenizer_path, device)
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
    output_file_path = os.path.join(abs_out_dir, args.output_filename)
    logger.info(f"Saving {len(generated_graphs)} generated AIG DiGraphs to {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(generated_graphs, f)
        logger.info("Graphs saved successfully.")
    except Exception as e:
        logger.error(f"Error saving pickle file: {e}", exc_info=True)

    logger.info(f"--- AIG Beam Search ({args.search_type}) Sampling Script Finished ---")
