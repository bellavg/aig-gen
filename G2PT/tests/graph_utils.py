# G2PT/tests/graph_utils.py
# Utility functions for converting between graph formats for testing.

import os
import sys
import torch
import networkx as nx
import numpy as np
import logging
import warnings # Added warnings



import G2PT.configs.aig as aig_cfg
print("graph_utils.py: Imported G2PT.configs.aig successfully.")
config_loaded = True


# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')
logger = logging.getLogger("graph_utils")

# --- Conversion Functions ---

def pyg_data_to_nx(data):
    """
    Converts a PyG Data object to a NetworkX DiGraph.
    Handles different formats of node features.
    Adds edge types based on edge_attr (vocab IDs or feature indices).
    """
    if not hasattr(data, 'x') or data.x is None:
        logger.warning("Input PyG data missing 'x' attribute.")
        return None
    if not hasattr(data, 'edge_index'):
        logger.warning("Input PyG data missing 'edge_index' attribute. Assuming no edges.")
        data.edge_index = torch.empty((2, 0), dtype=torch.long)

    G = nx.DiGraph()
    num_nodes = data.x.size(0)

    # Node Mappings
    node_type_list = list(aig_cfg.NODE_TYPE_KEYS)
    node_vocab_offset = aig_cfg.NODE_VOCAB_OFFSET
    vocab_id_to_node_feature_index = {aig_cfg.NODE_FEATURE_INDEX_TO_VOCAB[i]: i for i in range(aig_cfg.NUM_NODE_FEATURES)}

    # --- Node Type Determination (same as before) ---
    type_indices = []
    if data.x.numel() == 0: return G
    if data.x.dim() > 1 and data.x.shape[1] > 1 and data.x.dtype == torch.float:
        type_indices = torch.argmax(data.x, dim=1).tolist()
    elif data.x.dim() == 1 and data.x.dtype in [torch.long, torch.int, torch.int16]:
        is_vocab_id = torch.all((data.x >= node_vocab_offset) & (data.x < node_vocab_offset + len(node_type_list))).item()
        is_feature_idx = torch.all((data.x >= 0) & (data.x < len(node_type_list))).item()
        if is_vocab_id: type_indices = [vocab_id_to_node_feature_index.get(vid.item(), -1) for vid in data.x]
        elif is_feature_idx: type_indices = data.x.tolist()
        else: logger.error(f"Unexpected integer values in data.x: {data.x.unique().tolist()}"); return None
    else: logger.error(f"Unexpected shape/type for data.x: {data.x.shape}, dtype: {data.x.dtype}."); return None

    # --- Add Nodes (same as before) ---
    for i in range(num_nodes):
        if i < len(type_indices):
            node_type_idx = int(type_indices[i])
            node_type_str = node_type_list[node_type_idx] if 0 <= node_type_idx < len(node_type_list) else 'UNKNOWN'
            if node_type_str == 'UNKNOWN': logger.warning(f"Node {i}: Unknown node type index {node_type_idx} encountered.")
            G.add_node(i, type=node_type_str)
        else: logger.error(f"Node index mismatch: Graph claims {num_nodes} nodes, but only {len(type_indices)} type indices found."); G.add_node(i, type='UNKNOWN_ERROR')

    # --- Add Edges WITH Types ---
    edge_type_list = list(aig_cfg.EDGE_TYPE_KEYS)
    edge_vocab_offset = aig_cfg.EDGE_VOCAB_OFFSET
    # <<< CHANGE HERE: Create reverse map: vocab_id -> edge_type_string >>>
    edge_vocab_id_to_type_str = {v: k for k, v in aig_cfg.EDGE_TYPE_VOCAB.items()}
    # <<< CHANGE HERE: Create map: feature_idx -> edge_type_string >>>
    edge_feature_idx_to_type_str = {i: k for i, k in enumerate(edge_type_list)}

    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
             logger.error(f"Invalid edge_index shape: {data.edge_index.shape}. Expected [2, num_edges].")
        else:
            edge_index = data.edge_index.cpu().numpy()
            num_edges = edge_index.shape[1]
            edge_types = ['UNKNOWN_EDGE'] * num_edges # Default edge type

            # Determine edge types from edge_attr if available
            if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.numel() > 0:
                if data.edge_attr.shape[0] != num_edges:
                     logger.warning(f"Edge attribute shape mismatch ({data.edge_attr.shape[0]}) with edge index ({num_edges}). Edge types set to UNKNOWN.")
                else:
                    edge_attr = data.edge_attr
                    edge_type_strings = []
                    # Determine edge type strings based on edge_attr format
                    if edge_attr.dim() > 1 and edge_attr.shape[1] > 1 and edge_attr.dtype == torch.float: # One-hot float
                        edge_type_indices = torch.argmax(edge_attr, dim=1).tolist()
                        edge_type_strings = [edge_feature_idx_to_type_str.get(idx, 'UNKNOWN_EDGE') for idx in edge_type_indices]
                    elif edge_attr.dim() == 1 and edge_attr.dtype in [torch.long, torch.int, torch.int16]: # Int indices or vocab IDs
                        is_edge_vocab_id = torch.all((edge_attr >= edge_vocab_offset) & (edge_attr < edge_vocab_offset + len(edge_type_list))).item()
                        is_edge_feature_idx = torch.all((edge_attr >= 0) & (edge_attr < len(edge_type_list))).item()
                        if is_edge_vocab_id:
                             edge_type_strings = [edge_vocab_id_to_type_str.get(vid.item(), 'UNKNOWN_EDGE') for vid in edge_attr]
                        elif is_edge_feature_idx:
                             edge_type_strings = [edge_feature_idx_to_type_str.get(idx.item(), 'UNKNOWN_EDGE') for idx in edge_attr]
                        else: logger.warning(f"Unexpected integer values in edge_attr: {edge_attr.unique().tolist()}. Edge types set to UNKNOWN.")
                    else: logger.warning(f"Unexpected shape/type for edge_attr: {edge_attr.shape}, dtype: {edge_attr.dtype}. Edge types set to UNKNOWN.")

                    # Assign determined types if found
                    if edge_type_strings:
                        edge_types = edge_type_strings

            # Add edges to graph
            for i in range(num_edges):
                u, v = int(edge_index[0, i]), int(edge_index[1, i])
                if u in G and v in G:
                    G.add_edge(u, v, type=edge_types[i]) # Add edge WITH type
                else:
                    logger.warning(f"Invalid edge index found: ({u}, {v}) for graph with {num_nodes} nodes. Skipping edge.")
    return G


# --- bin_data_to_nx (No changes needed) ---
def bin_data_to_nx(node_vocab_ids, edge_index, edge_vocab_ids):
    """Converts unpadded data (from .bin files) with VOCAB IDs into a NetworkX DiGraph."""
    if node_vocab_ids is None: logger.warning("Input node_vocab_ids is None."); return None
    G = nx.DiGraph(); num_nodes = node_vocab_ids.size(0)
    try:
        node_type_list = list(aig_cfg.NODE_TYPE_KEYS); edge_type_list = list(aig_cfg.EDGE_TYPE_KEYS)
        node_vocab_offset = aig_cfg.NODE_VOCAB_OFFSET; edge_vocab_offset = aig_cfg.EDGE_VOCAB_OFFSET
        if node_vocab_offset is None or edge_vocab_offset is None: raise ValueError("Vocab offsets not found in aig_cfg")
    except (AttributeError, ValueError) as e: logger.error(f"bin_data_to_nx: Could not get config values: {e}. Returning None."); return None
    for i in range(num_nodes):
        node_vocab_id = node_vocab_ids[i].item(); node_type_idx = node_vocab_id - node_vocab_offset
        node_type_str = node_type_list[node_type_idx] if 0 <= node_type_idx < len(node_type_list) else 'UNKNOWN_NODE'
        if node_type_str == 'UNKNOWN_NODE': logger.warning(f"Node {i}: Unknown node vocab ID {node_vocab_id} encountered.")
        G.add_node(i, type=node_type_str)
    if edge_index is not None and edge_index.numel() > 0:
        if edge_vocab_ids is None or edge_vocab_ids.numel() == 0 or edge_vocab_ids.shape[0] != edge_index.shape[1]:
             logger.error(f"Edge attributes (vocab IDs) missing or shape mismatch with edge_index. Cannot add edge types.")
             num_edges = edge_index.shape[1]
             for i in range(num_edges):
                 u, v = int(edge_index[0, i]), int(edge_index[1, i])
                 if u in G and v in G: G.add_edge(u, v, type='UNKNOWN_EDGE_ATTR')
                 else: logger.warning(f"Invalid edge index ({u}, {v}) found when adding edges without attrs.")
        else:
            edge_index_np = edge_index.cpu().numpy(); num_edges = edge_index_np.shape[1]
            for i in range(num_edges):
                u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                edge_vocab_id = edge_vocab_ids[i].item(); edge_type_idx = edge_vocab_id - edge_vocab_offset
                edge_type_str = edge_type_list[edge_type_idx] if 0 <= edge_type_idx < len(edge_type_list) else 'UNKNOWN_EDGE'
                if edge_type_str == 'UNKNOWN_EDGE': logger.warning(f"Edge ({u}, {v}): Unknown edge vocab ID {edge_vocab_id} encountered.")
                if u in G and v in G: G.add_edge(u, v, type=edge_type_str)
                else: logger.warning(f"Invalid edge index found: ({u}, {v}) for graph with {num_nodes} nodes. Skipping edge.")
    return G
