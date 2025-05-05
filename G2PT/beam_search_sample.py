"""
Sample from a trained AIG model using Constrained Beam Search.
"""
import os
import argparse
import json
import heapq
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any, Optional
from contextlib import nullcontext
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel # Use HF types
from model import GPTConfig, GPT # Keep for loading config, but use HF model for generation

# Assuming datasets_utils.py is in the same directory or accessible
from datasets_utils import seq_to_nxgraph

# --- Beam State ---
@dataclass(order=True)
class BeamState:
    """Represents one beam in the search."""
    log_prob: float # Total log probability (used for sorting/heapq)
    # --- Fields not used for ordering ---
    sequence: List[int] = field(compare=False) # List of token IDs
    graph_state: Dict[str, Any] = field(compare=False) # Dictionary tracking graph structure
    kv_cache: Optional[Tuple[torch.Tensor]] = field(compare=False) # Model's KV cache
    is_finished: bool = field(default=False, compare=False) # Flag if <eog> was generated

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Sample from a trained AIG model using Constrained Beam Search')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing model checkpoint (e.g., results/aig-small-topo)')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to tokenizer (e.g., datasets/aig/tokenizer)')
    parser.add_argument('--num_samples', type=int, default=10, # Smaller default for beam search
                        help='Number of valid samples to generate')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Number of beams to maintain during search')
    parser.add_argument('--max_new_tokens', type=int, default=250, # Max length of generated part
                        help='Maximum number of tokens to generate per sequence')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (1.0 = standard)')
    # parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling threshold') # Optional
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--output_filename', type=str, default='generated_aigs_beam.pkl',
                        help='Name for the output pickle file')
    parser.add_argument('--parsing_mode', type=str, default='strict', choices=['strict', 'robust'],
                        help='Final sequence parsing mode for seq_to_nxgraph')

    return parser.parse_args()

# --- Device Setup (reuse from sample.py) ---
def setup_device(seed):
    # Automatically detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'
    else:
         device = 'cpu'
    print(f"Using device: {device}")

    # Determine appropriate dtype based on chosen device
    if device == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        # Enable TF32 for CUDA acceleration if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = 'float32'
    print(f"Using dtype: {dtype}")

    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return device, ctx

# --- Model Loading (reuse from sample.py, ensures hf_model) ---
def load_model(out_dir, device) -> PreTrainedModel: # Return type hint for HF model
    ckpt_path = os.path.join(out_dir, 'best.pt')
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        print("Ensure --out_dir points to the correct directory containing ckpt.pt")
        exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    # Load model configuration from checkpoint
    try:
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf) # Load original model first
    except KeyError:
        print("Error: 'model_args' not found in checkpoint. Cannot recreate model.")
        exit(1)
    except Exception as e:
        print(f"Error recreating model from checkpoint args: {e}")
        exit(1)

    # Load model state dict
    try:
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    except KeyError:
        print("Error: 'model' state dict not found in checkpoint.")
        exit(1)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit(1)

    # Convert to Hugging Face format - THIS IS IMPORTANT for KV cache handling
    try:
        hf_model = model.to_hf()
        print("Successfully converted model to Hugging Face format.")
    except Exception as e:
        print(f"Error: Failed to convert model to Hugging Face format: {e}.")
        print("Constrained beam search relies on HF model structure for KV cache.")
        exit(1)

    hf_model.eval()
    hf_model.to(device)
    print(f"Model loaded successfully with {sum(p.numel() for p in hf_model.parameters())/1e6:.2f}M parameters.")
    return hf_model

# --- Token ID Mappings ---
def get_token_ids(tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """Loads vocab and extracts relevant token IDs."""
    vocab = tokenizer.get_vocab()
    token_ids = {}
    try:
        token_ids['boc_id'] = vocab["<boc>"]
        token_ids['eoc_id'] = vocab["<eoc>"]
        token_ids['bog_id'] = vocab["<bog>"]
        token_ids['eog_id'] = vocab["<eog>"]
        token_ids['sepc_id'] = vocab["<sepc>"]
        token_ids['sepg_id'] = vocab["<sepg>"]
        token_ids['pad_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1 # Handle padding

        # Node/Edge Types
        token_ids['node_type_ids'] = {vocab[tok]: tok for tok in vocab if tok.startswith("NODE_")}
        token_ids['edge_type_ids'] = {vocab[tok]: tok for tok in vocab if tok.startswith("EDGE_")}
        token_ids['idx_token_ids'] = {vocab[tok]: tok for tok in vocab if tok.startswith("IDX_")}

        # Specific Node Types for Constraints
        token_ids['NODE_PI_ID'] = vocab.get("NODE_PI")
        token_ids['NODE_AND_ID'] = vocab.get("NODE_AND")
        token_ids['NODE_CONST0_ID'] = vocab.get("NODE_CONST0")
        token_ids['NODE_PO_ID'] = vocab.get("NODE_PO")

        # Precompute sets for faster lookups
        token_ids['node_type_id_set'] = set(token_ids['node_type_ids'].keys())
        token_ids['edge_type_id_set'] = set(token_ids['edge_type_ids'].keys())
        token_ids['idx_token_id_set'] = set(token_ids['idx_token_ids'].keys())
        token_ids['special_token_ids'] = {
            token_ids['boc_id'], token_ids['eoc_id'], token_ids['bog_id'],
            token_ids['eog_id'], token_ids['sepc_id'], token_ids['sepg_id']
        }
        token_ids['valid_token_ids'] = (token_ids['special_token_ids'] |
                                       token_ids['node_type_id_set'] |
                                       token_ids['edge_type_id_set'] |
                                       token_ids['idx_token_id_set'])
        # Add pad_id if it's valid
        if token_ids['pad_id'] != -1:
             token_ids['valid_token_ids'].add(token_ids['pad_id'])


        # Constraint Maps
        token_ids['max_in_degree_map'] = {
            token_ids['NODE_PI_ID']: 0,
            token_ids['NODE_CONST0_ID']: 0,
            token_ids['NODE_AND_ID']: 2,
            token_ids['NODE_PO_ID']: 1
        }
        token_ids['max_out_degree_map'] = {
            token_ids['NODE_PO_ID']: 0
        }
        # Regex for parsing IDX tokens
        token_ids['idx_pattern'] = re.compile(r'IDX_(\d+)')

    except KeyError as e:
        print(f"Error: Token '{e}' not found in tokenizer vocabulary. Check vocab.json.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while processing token IDs: {e}")
        exit(1)

    # Validate that essential node types were found
    if None in [token_ids['NODE_PI_ID'], token_ids['NODE_AND_ID'], token_ids['NODE_CONST0_ID'], token_ids['NODE_PO_ID']]:
         print("Error: One or more required NODE types (PI, AND, CONST0, PO) not found in vocab.")
         exit(1)

    return token_ids

# --- Helper: DAG Check ---
def has_path(start_node: int, end_node: int, edges: Set[Tuple[int, int]]) -> bool:
    """Checks if a path exists from start_node to end_node using BFS."""
    if start_node == end_node: # Path to self exists if edge is added
        return True
    q = deque([start_node])
    visited = {start_node}
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    while q:
        curr = q.popleft()
        for neighbor in adj.get(curr, []):
            if neighbor == end_node:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return False

# --- Helper: Update Graph State ---
def update_graph_state(current_state: Dict[str, Any],
                       token_id: int,
                       token_ids: Dict[str, Any],
                       tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """
    Updates the lightweight graph state based on the newly added token.
    Returns the *new* state dictionary.
    """
    new_state = {
        'node_types': current_state['node_types'].copy(),
        'node_indices': current_state['node_indices'].copy(),
        'in_degree': current_state['in_degree'].copy(),
        'out_degree': current_state['out_degree'].copy(),
        'edges': current_state['edges'].copy(),
        'context': current_state['context'],
        'last_idx_token': current_state.get('last_idx_token'), # Store last IDX token for edge context
        'last_node_type_token': current_state.get('last_node_type_token'), # Store last NODE type token
        'is_complete': current_state.get('is_complete', False) # Track if graph seems complete
    }
    current_context = new_state['context']
    token_str = tokenizer.decode([token_id]) # Get the string representation

    # --- State Machine based on Context and Token ---
    if current_context == 'EXPECTING_NODE_TYPE':
        if token_id in token_ids['node_type_id_set']:
            new_state['last_node_type_token'] = token_id
            new_state['context'] = 'EXPECTING_NODE_IDX'
        elif token_id == token_ids['eoc_id']:
            new_state['context'] = 'EXPECTING_BOG' # End of node definitions
        else:
             # Invalid token for this context (should be masked, but handle defensively)
             print(f"Warning: Unexpected token {token_str} in context {current_context}")
             new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_NODE_IDX':
        idx_match = token_ids['idx_pattern'].match(token_str)
        if idx_match and token_id in token_ids['idx_token_id_set']:
            node_index = int(idx_match.group(1))
            last_node_type = new_state.get('last_node_type_token')
            if last_node_type is not None:
                 # Add node if not already present (can happen with robust parsing/generation)
                 if node_index not in new_state['node_indices']:
                      new_state['node_indices'].add(node_index)
                      new_state['node_types'][node_index] = token_ids['node_type_ids'][last_node_type]
                      # Initialize degrees (important!)
                      new_state['in_degree'][node_index] = 0
                      new_state['out_degree'][node_index] = 0
                 else:
                      # Node already exists, potentially update type? Or warn?
                      # For now, assume first definition wins or it's consistent.
                      pass
            else:
                 print(f"Warning: IDX token {token_str} found without preceding NODE type.")
                 new_state['context'] = 'ERROR'

            new_state['last_idx_token'] = token_id # Store for potential edge source
            new_state['context'] = 'EXPECTING_SEPARATOR_OR_EOC' # Expect <sepc> or <eoc>
        else:
            print(f"Warning: Unexpected token {token_str} in context {current_context}")
            new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_SEPARATOR_OR_EOC':
        if token_id == token_ids['sepc_id']:
            new_state['context'] = 'EXPECTING_NODE_TYPE' # Start next node def
        elif token_id == token_ids['eoc_id']:
            new_state['context'] = 'EXPECTING_BOG' # End node definitions
        else:
            print(f"Warning: Unexpected token {token_str} in context {current_context}")
            new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_BOG':
        if token_id == token_ids['bog_id']:
            new_state['context'] = 'EXPECTING_EDGE_SRC_OR_EOG' # Start of edge definitions
        else:
            print(f"Warning: Unexpected token {token_str} in context {current_context}")
            new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_EDGE_SRC_OR_EOG':
         idx_match = token_ids['idx_pattern'].match(token_str)
         if idx_match and token_id in token_ids['idx_token_id_set']:
             new_state['last_idx_token'] = token_id # Store potential source
             new_state['context'] = 'EXPECTING_EDGE_DST'
         elif token_id == token_ids['eog_id']:
             new_state['context'] = 'FINISHED' # End of graph definition
             # Check final graph completeness here
             new_state['is_complete'] = check_graph_completeness(new_state, token_ids)
         elif token_id == token_ids['sepg_id']: # Allow starting with separator
              new_state['context'] = 'EXPECTING_EDGE_SRC'
         else:
             print(f"Warning: Unexpected token {token_str} in context {current_context}")
             new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_EDGE_SRC': # After <sepg>
         idx_match = token_ids['idx_pattern'].match(token_str)
         if idx_match and token_id in token_ids['idx_token_id_set']:
             new_state['last_idx_token'] = token_id # Store source
             new_state['context'] = 'EXPECTING_EDGE_DST'
         else:
             print(f"Warning: Unexpected token {token_str} in context {current_context}")
             new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_EDGE_DST':
        idx_match_dst = token_ids['idx_pattern'].match(token_str)
        if idx_match_dst and token_id in token_ids['idx_token_id_set']:
            # Store destination temporarily (need edge type next)
            new_state['last_dst_idx_token'] = token_id
            new_state['context'] = 'EXPECTING_EDGE_TYPE'
        else:
            print(f"Warning: Unexpected token {token_str} in context {current_context}")
            new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_EDGE_TYPE':
        if token_id in token_ids['edge_type_id_set']:
            src_token_id = new_state.get('last_idx_token')
            dst_token_id = new_state.get('last_dst_idx_token')
            src_token_str = tokenizer.decode([src_token_id]) if src_token_id else None
            dst_token_str = tokenizer.decode([dst_token_id]) if dst_token_id else None

            src_match = token_ids['idx_pattern'].match(src_token_str) if src_token_str else None
            dst_match = token_ids['idx_pattern'].match(dst_token_str) if dst_token_str else None

            if src_match and dst_match:
                u = int(src_match.group(1))
                v = int(dst_match.group(1))
                # --- Update graph state with the new edge ---
                if u in new_state['node_indices'] and v in new_state['node_indices']:
                     new_state['edges'].add((u, v))
                     new_state['out_degree'][u] = new_state['out_degree'].get(u, 0) + 1
                     new_state['in_degree'][v] = new_state['in_degree'].get(v, 0) + 1
                     new_state['context'] = 'EXPECTING_SEPARATOR_OR_EOG' # Expect <sepg> or <eog>
                else:
                     print(f"Warning: Edge nodes {u} or {v} not defined.")
                     new_state['context'] = 'ERROR'
            else:
                print(f"Warning: Could not parse source/dest tokens for edge.")
                new_state['context'] = 'ERROR'
        else:
            print(f"Warning: Unexpected token {token_str} in context {current_context}")
            new_state['context'] = 'ERROR'

    elif current_context == 'EXPECTING_SEPARATOR_OR_EOG': # After edge triplet
        if token_id == token_ids['sepg_id']:
            new_state['context'] = 'EXPECTING_EDGE_SRC' # Start next edge
        elif token_id == token_ids['eog_id']:
            new_state['context'] = 'FINISHED' # End of graph
            new_state['is_complete'] = check_graph_completeness(new_state, token_ids)
        else:
            print(f"Warning: Unexpected token {token_str} in context {current_context}")
            new_state['context'] = 'ERROR'

    # Handle FINISHED and ERROR states (no further updates)
    elif current_context in ['FINISHED', 'ERROR']:
        pass

    else: # Should not happen
        print(f"Error: Unknown context '{current_context}'")
        new_state['context'] = 'ERROR'

    # Clean up temporary state vars if they exist
    new_state.pop('last_dst_idx_token', None)
    if new_state['context'] != 'EXPECTING_EDGE_DST':
         new_state.pop('last_idx_token', None) # Clear if not expecting dst
    if new_state['context'] != 'EXPECTING_NODE_IDX':
         new_state.pop('last_node_type_token', None) # Clear if not expecting idx

    return new_state

# --- Helper: Check Graph Completeness ---
def check_graph_completeness(graph_state: Dict[str, Any], token_ids: Dict[str, Any]) -> bool:
    """Checks if the graph state meets basic completeness criteria."""
    # Example checks (adjust as needed):
    # 1. All AND gates must have in-degree 2
    # 2. All PO gates must have in-degree 1
    node_types = graph_state['node_types']
    in_degree = graph_state['in_degree']
    NODE_AND_STR = token_ids['node_type_ids'].get(token_ids['NODE_AND_ID'])
    NODE_PO_STR = token_ids['node_type_ids'].get(token_ids['NODE_PO_ID'])

    for node_idx, node_type_str in node_types.items():
        if node_type_str == NODE_AND_STR and in_degree.get(node_idx, 0) != 2:
            # print(f"Debug: AND node {node_idx} incomplete (in-degree {in_degree.get(node_idx, 0)})")
            return False
        if node_type_str == NODE_PO_STR and in_degree.get(node_idx, 0) != 1:
            # print(f"Debug: PO node {node_idx} incomplete (in-degree {in_degree.get(node_idx, 0)})")
            return False
    # Add other checks if necessary (e.g., minimum number of nodes/edges?)
    return True


# --- Helper: Calculate Constraint Mask ---
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

    # 1. Mask all tokens that are fundamentally invalid (not in vocab structure)
    # Create a mask for all possible token indices
    all_indices = torch.arange(vocab_size)
    # Assume `token_ids['valid_token_ids']` is a set of valid integer IDs
    valid_mask = torch.tensor([idx in token_ids['valid_token_ids'] for idx in all_indices.tolist()], dtype=torch.bool)
    mask[~valid_mask] = True # Mask tokens not in our defined valid set

    # 2. Grammar Constraints (based on current context)
    allowed_token_types = set()
    if context == 'EXPECTING_NODE_TYPE':
        allowed_token_types.update(token_ids['node_type_id_set'])
        allowed_token_types.add(token_ids['eoc_id'])
    elif context == 'EXPECTING_NODE_IDX':
        allowed_token_types.update(token_ids['idx_token_id_set']) # Further checks needed for *which* idx
    elif context == 'EXPECTING_SEPARATOR_OR_EOC':
        allowed_token_types.add(token_ids['sepc_id'])
        allowed_token_types.add(token_ids['eoc_id'])
    elif context == 'EXPECTING_BOG':
        allowed_token_types.add(token_ids['bog_id'])
    elif context == 'EXPECTING_EDGE_SRC_OR_EOG':
        allowed_token_types.update(token_ids['idx_token_id_set'])
        allowed_token_types.add(token_ids['eog_id'])
        allowed_token_types.add(token_ids['sepg_id']) # Allow <sepg> first
    elif context == 'EXPECTING_EDGE_SRC': # After <sepg>
         allowed_token_types.update(token_ids['idx_token_id_set'])
    elif context == 'EXPECTING_EDGE_DST':
        allowed_token_types.update(token_ids['idx_token_id_set'])
    elif context == 'EXPECTING_EDGE_TYPE':
        allowed_token_types.update(token_ids['edge_type_id_set'])
    elif context == 'EXPECTING_SEPARATOR_OR_EOG': # After edge triplet
        allowed_token_types.add(token_ids['sepg_id'])
        allowed_token_types.add(token_ids['eog_id'])
    elif context == 'FINISHED' or context == 'ERROR':
         allowed_token_types.add(token_ids['pad_id']) # Only allow padding if finished/error

    # Apply grammar mask (mask everything NOT allowed by the current context)
    grammar_mask = torch.tensor([idx not in allowed_token_types for idx in all_indices.tolist()], dtype=torch.bool)
    mask[grammar_mask] = True

    # --- 3. AIG Specific Structural Constraints ---
    # These often refine the grammar constraints (e.g., which IDX tokens are allowed)

    # Get current graph properties
    node_types = graph_state['node_types']
    node_indices = graph_state['node_indices']
    in_degree = graph_state['in_degree']
    out_degree = graph_state['out_degree']
    edges = graph_state['edges']

    # Constraint logic based on context
    if context == 'EXPECTING_NODE_IDX':
         # Prevent redefining existing nodes? Or just allow? For now, allow.
         # Prevent defining too many nodes? (Add if needed)
         pass # No additional structural constraints here yet

    elif context == 'EXPECTING_EDGE_DST':
        src_token_id = graph_state.get('last_idx_token')
        src_token_str = tokenizer.decode([src_token_id]) if src_token_id else None
        src_match = token_ids['idx_pattern'].match(src_token_str) if src_token_str else None
        if src_match:
            u = int(src_match.group(1))
            u_type = node_types.get(u)

            # PO Out-Degree Constraint
            if u_type == token_ids['node_type_ids'].get(token_ids['NODE_PO_ID']):
                 mask[list(token_ids['idx_token_id_set'])] = True # Mask ALL destination indices if source is PO

            else: # Apply other constraints only if source is not PO
                for v_token_id, v_token_str in token_ids['idx_token_ids'].items():
                    if mask[v_token_id]: continue # Skip if already masked by grammar

                    v_match = token_ids['idx_pattern'].match(v_token_str)
                    if v_match:
                        v = int(v_match.group(1))

                        # Check if destination node exists (it should have been defined)
                        if v not in node_indices:
                            mask[v_token_id] = True; continue

                        v_type_str = node_types.get(v)
                        v_type_id = None
                        for tid, tstr in token_ids['node_type_ids'].items():
                             if tstr == v_type_str: v_type_id = tid; break

                        # PI/CONST0 In-Degree Constraint
                        if v_type_id == token_ids['NODE_PI_ID'] or v_type_id == token_ids['NODE_CONST0_ID']:
                            mask[v_token_id] = True; continue

                        # AND In-Degree Constraint
                        if v_type_id == token_ids['NODE_AND_ID'] and in_degree.get(v, 0) >= 2:
                            mask[v_token_id] = True; continue

                        # PO In-Degree Constraint
                        if v_type_id == token_ids['NODE_PO_ID'] and in_degree.get(v, 0) >= 1:
                            mask[v_token_id] = True; continue

                        # DAG Constraint
                        if has_path(v, u, edges):
                            mask[v_token_id] = True; continue
        else:
             # If source token is invalid, mask all destinations
             mask[list(token_ids['idx_token_id_set'])] = True


    elif context == 'EXPECTING_EDGE_SRC' or context == 'EXPECTING_EDGE_SRC_OR_EOG':
         # Constraint: Source node must exist
         for u_token_id, u_token_str in token_ids['idx_token_ids'].items():
              if mask[u_token_id]: continue
              u_match = token_ids['idx_pattern'].match(u_token_str)
              if u_match:
                   u = int(u_match.group(1))
                   if u not in node_indices:
                        mask[u_token_id] = True; continue
                   # Constraint: PO cannot be a source (redundant with check in EXPECTING_EDGE_DST?)
                   u_type_str = node_types.get(u)
                   if u_type_str == token_ids['node_type_ids'].get(token_ids['NODE_PO_ID']):
                       mask[u_token_id] = True; continue
              else: # Should not happen if vocab is correct
                   mask[u_token_id] = True


    # Termination Constraint (applied when EOG is a possibility)
    if context in ['EXPECTING_EDGE_SRC_OR_EOG', 'EXPECTING_SEPARATOR_OR_EOG']:
         if not check_graph_completeness(graph_state, token_ids):
              mask[token_ids['eog_id']] = True # Mask EOG if graph is incomplete

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
    max_new_tokens: int = 250,
    temperature: float = 1.0,
    # top_k: Optional[int] = None, # Optional: Add top-k logic if needed
) -> List[Tuple[float, List[int]]]:
    """
    Generates sequences using constrained beam search.
    Returns a list of (log_probability, sequence_token_ids) for completed beams.
    """
    model.eval()
    vocab_size = tokenizer.vocab_size

    # Initial state
    start_token_id = token_ids['boc_id']
    initial_sequence = [start_token_id]
    initial_graph_state = {
        'node_types': {}, 'node_indices': set(), 'in_degree': defaultdict(int),
        'out_degree': defaultdict(int), 'edges': set(),
        'context': 'EXPECTING_NODE_TYPE', # Start by expecting a node type or EOC
        'is_complete': False
    }
    # Use negative log_prob because heapq is a min-heap
    initial_beam = BeamState(log_prob=0.0, sequence=initial_sequence, graph_state=initial_graph_state, kv_cache=None)

    # Active beams (min-heap based on negative log_prob)
    active_beams = [initial_beam]
    finished_beams = []

    for step in range(max_new_tokens):
        if not active_beams: break # Stop if no active beams left

        candidates = [] # Candidates for the next step (log_prob, sequence, kv_cache, graph_state)

        for beam in active_beams:
            if beam.is_finished:
                # Add finished beams directly to candidates to preserve them,
                # but they won't be expanded further.
                # Use negative log_prob for heapq consistency
                heapq.heappush(candidates, (-beam.log_prob, beam))
                continue

            # Prepare input for this beam
            current_token_id = torch.tensor([[beam.sequence[-1]]], device=device)

            # Get logits and KV cache
            try:
                 outputs = model(
                    input_ids=current_token_id,
                    past_key_values=beam.kv_cache,
                    use_cache=True,
                 )
                 logits = outputs.logits[:, -1, :] # Logits for the next token (Batch=1, Seq=1, Vocab)
                 next_kv_cache = outputs.past_key_values
            except Exception as e:
                 print(f"Error during model forward pass: {e}")
                 print(f"Beam sequence causing error: {tokenizer.decode(beam.sequence)}")
                 continue # Skip this beam if model fails

            # Apply temperature
            logits = logits / temperature

            # Calculate constraint mask
            constraint_mask = calculate_constraint_mask(beam, token_ids, vocab_size, tokenizer).to(device)

            # Apply mask
            logits[constraint_mask] = -float('inf')

            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get top-k candidates (k = beam_size)
            # Ensure we don't select tokens with -inf log_prob
            finite_log_probs = log_probs[log_probs > -float('inf')]
            if finite_log_probs.numel() == 0:
                 # print(f"Warning: Beam reached dead end (all next tokens masked). Seq: {tokenizer.decode(beam.sequence)}")
                 continue # No valid next steps for this beam

            # Consider only beam_size successors even if more are valid
            num_successors = min(beam_size, finite_log_probs.numel())
            top_log_probs, top_indices = torch.topk(log_probs, num_successors, dim=-1)

            # Expand the beam
            for j in range(num_successors):
                next_token_id = top_indices[0, j].item()
                next_log_prob = top_log_probs[0, j].item()

                new_sequence = beam.sequence + [next_token_id]
                new_total_log_prob = beam.log_prob + next_log_prob

                # Update graph state (potentially expensive)
                try:
                    new_graph_state = update_graph_state(beam.graph_state, next_token_id, token_ids, tokenizer)
                except Exception as e:
                     print(f"Error updating graph state: {e}")
                     print(f"Sequence causing error: {tokenizer.decode(new_sequence)}")
                     continue # Skip this candidate if state update fails

                new_beam = BeamState(
                    log_prob=new_total_log_prob,
                    sequence=new_sequence,
                    graph_state=new_graph_state,
                    kv_cache=next_kv_cache,
                    is_finished=(next_token_id == token_ids['eog_id'] and new_graph_state.get('is_complete', False))
                )

                # Add to candidates list (using negative log_prob for min-heap)
                heapq.heappush(candidates, (-new_beam.log_prob, new_beam))


        # Prune candidates to keep only the top `beam_size`
        active_beams = []
        seen_sequences = set() # Prevent duplicates if multiple paths lead to same sequence

        # Use heapq to efficiently get the top k *unique* beams
        # We store (-log_prob, beam) in the heap
        count = 0
        processed_candidates = set() # Keep track of processed candidate tuples to avoid duplicates from heap
        while candidates and count < beam_size:
             neg_log_prob, current_beam = heapq.heappop(candidates)

             # Create a hashable representation of the beam state for duplicate checking
             # Using sequence tuple is a reasonable proxy here
             seq_tuple = tuple(current_beam.sequence)
             if seq_tuple in seen_sequences:
                  continue # Skip duplicate sequence

             seen_sequences.add(seq_tuple)

             if current_beam.is_finished:
                  finished_beams.append(current_beam)
             else:
                  active_beams.append(current_beam)
             count += 1


        # Optional: Add finished beams back to active if fewer than beam_size active beams remain?
        # This can sometimes help explore alternatives if main paths finish early.
        # However, it complicates length normalization if comparing finished/unfinished.
        # For now, keep finished separate.

        # Stop if max length reached for all active beams
        if active_beams and len(active_beams[0].sequence) >= max_new_tokens + len(initial_sequence):
             print("Max length reached for active beams.")
             # Move all remaining active beams to finished (they might be incomplete)
             finished_beams.extend(active_beams)
             active_beams = [] # Stop the loop


    # Add any remaining active beams (that didn't finish) to the finished list
    finished_beams.extend(active_beams)

    # Sort finished beams by log probability (descending)
    finished_beams.sort(key=lambda b: b.log_prob, reverse=True)

    # Return list of (log_prob, sequence)
    return [(beam.log_prob, beam.sequence) for beam in finished_beams]


# --- Main Execution ---
if __name__ == '__main__':
    args = parse_args()
    print("--- Starting AIG Constrained Beam Search Sampling Script ---")
    print(f"Arguments: {args}")

    device, ctx = setup_device(args.seed)

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if tokenizer.pad_token is None:
             print("Warning: Tokenizer does not have a pad token defined. Adding '<pad>'.")
             # Add a pad token if missing. Adjust vocab size in model config if necessary.
             tokenizer.add_special_tokens({'pad_token': '<pad>'})
             # NOTE: If adding tokens, the model's embedding layer size might need adjustment
             # if loading a checkpoint trained without this token.
    except Exception as e:
        print(f"Error loading tokenizer from {args.tokenizer_path}: {e}")
        exit(1)

    # Get token ID mappings
    token_ids = get_token_ids(tokenizer)

    # Load Model (ensure it's the HF version)
    model = load_model(args.out_dir, device)
    # Resize model embeddings if pad token was added *after* training
    # model.resize_token_embeddings(len(tokenizer)) # Uncomment if pad token was added

    print(f"Generating {args.num_samples} samples using beam search (k={args.beam_size})...")

    generated_sequences = []
    attempts = 0
    max_attempts = args.num_samples * max(5, args.beam_size * 2) # Heuristic limit

    while len(generated_sequences) < args.num_samples and attempts < max_attempts:
        attempts += 1
        if attempts % 10 == 0:
             print(f"Attempt {attempts}/{max_attempts}, Found {len(generated_sequences)}/{args.num_samples} valid sequences...")

        # Generate one sequence using beam search
        with ctx: # Autocast context
             # Note: This generates only *one* best sequence per call currently.
             # To get more diverse samples, you might need to run this multiple times
             # or modify generate_constrained_beam to return multiple beams.
             best_beams = generate_constrained_beam(
                 model=model,
                 tokenizer=tokenizer,
                 token_ids=token_ids,
                 device=device,
                 prompt="<boc>",
                 beam_size=args.beam_size,
                 max_new_tokens=args.max_new_tokens,
                 temperature=args.temperature,
             )

        if not best_beams:
             print("Warning: Beam search returned no sequences.")
             continue

        # Select the top sequence (or potentially sample from top beams)
        # For now, just take the single best one
        best_log_prob, best_sequence_ids = best_beams[0]

        # Decode
        seq_str = tokenizer.decode(best_sequence_ids, skip_special_tokens=False)

        # --- Optional: Add basic check if sequence seems valid before full parsing ---
        if not seq_str.endswith("<eog>"):
             # print(f"Skipping sequence (attempt {attempts}): Did not end with <eog>. LogProb: {best_log_prob:.2f}")
             # print(f"Sequence sample: {seq_str[-50:]}")
             continue
        if "<bog>" not in seq_str or "<eoc>" not in seq_str:
             # print(f"Skipping sequence (attempt {attempts}): Missing <bog> or <eoc>. LogProb: {best_log_prob:.2f}")
             continue

        # Add the successfully generated sequence string
        generated_sequences.append(seq_str)
        print(f"Generated sequence {len(generated_sequences)}/{args.num_samples}. LogProb: {best_log_prob:.2f}")


    if len(generated_sequences) < args.num_samples:
         print(f"\nWarning: Only generated {len(generated_sequences)} sequences after {max_attempts} attempts.")
    else:
         print(f"\nSuccessfully generated {len(generated_sequences)} sequences.")


    # --- Convert Sequences to AIG DiGraphs ---
    print("\nConverting generated sequences to AIG DiGraphs...")
    generated_graphs = []
    num_processed = 0
    num_errors = 0

    for i, seq_str in enumerate(generated_sequences):
        try:
            graph = seq_to_nxgraph(seq_str, parsing_mode=args.parsing_mode)
            if isinstance(graph, nx.Graph):
                # --- Optional: Add final validity check using evaluate_aigs.py logic ---
                # from evaluate_aigs import calculate_structural_aig_metrics
                # metrics = calculate_structural_aig_metrics(graph)
                # if metrics.get('is_structurally_valid', False):
                #      generated_graphs.append(graph)
                #      num_processed += 1
                # else:
                #      print(f"Warning: Sequence {i} produced a structurally INVALID graph post-beam search.")
                #      print(f"Reasons: {metrics.get('constraints_failed')}")
                #      num_errors += 1
                # --- End Optional Check ---
                # If not doing final check, just add the graph:
                generated_graphs.append(graph)
                num_processed += 1
            else:
                print(f"Warning: seq_to_nxgraph did not return a NetworkX graph for sequence {i}. Got {type(graph)}.")
                num_errors += 1
        except Exception as e:
            print(f"Error processing sequence {i} to AIG: {e}\nSequence sample: {seq_str[:150]}...")
            num_errors += 1

    # --- Reporting ---
    print("\n--- AIG Beam Search Generation Summary ---")
    print(f"Total sequences generated   : {len(generated_sequences)}")
    print(f"Successfully converted    : {num_processed}")
    print(f"Errors during conversion  : {num_errors}") # Or validity failures if final check added
    print("---------------------------------------")

    # --- Saving Results ---
    output_file_path = os.path.join(args.out_dir, args.output_filename)
    print(f"Saving {len(generated_graphs)} generated AIG DiGraphs to {output_file_path}")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'wb') as f:
            import pickle # Ensure pickle is imported
            pickle.dump(generated_graphs, f)
    except Exception as e:
        print(f"Error saving pickle file: {e}")

    print("\n--- AIG Constrained Beam Search Sampling Script Finished ---")

