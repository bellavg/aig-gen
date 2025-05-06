

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