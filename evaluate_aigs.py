# evaluate_aigs.py
import pickle
import networkx as nx
import argparse
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np
import sys
from tqdm import tqdm
import json # Added for loading metadata
import torch # Added for tensor operations during unpadding

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluate_g2pt_aigs")

# --- Import the AIG configuration ---
try:
    from data import aig_config as aig_config
except ImportError:
    logger.error("Failed to import AIG configuration from 'configs.aig'. Ensure it's accessible.")
    # Provide default values or exit if config is critical
    # Fallback values might not be accurate, exiting is safer
    sys.exit(1)

# Derive constants from config
VALID_AIG_NODE_TYPES = set(aig_config.NODE_TYPE_KEYS)
VALID_AIG_EDGE_TYPES = set(aig_config.EDGE_TYPE_KEYS)
NODE_CONST0 = "NODE_CONST0"
NODE_PI = "NODE_PI"
NODE_AND = "NODE_AND"
NODE_PO = "NODE_PO"
MIN_AND_COUNT_CONFIG = aig_config.MIN_AND_COUNT
MIN_PO_COUNT_CONFIG = aig_config.MIN_PO_COUNT
PAD_VALUE = aig_config.PAD_VALUE # Needed for unpadding

# --- Existing Functions (calculate_structural_aig_metrics, count_pi_po_paths, validate_aig_structures) ---
# ... (Keep your existing functions here - unchanged) ...
def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculates structural AIG validity metrics based on assigned types.
    Counts violations across the graph instead of breaking early.
    Uses constants from aig_config.
    Returns a dictionary of detailed metrics and violation counts.
    """
    metrics = {
        'num_nodes': 0,
        'is_dag': False,
        'num_pi': 0, 'num_po': 0, 'num_and': 0, 'num_const0': 0,
        'num_unknown_nodes': 0,
        'num_unknown_edges': 0,
        'pi_indegree_violations': 0,
        'const0_indegree_violations': 0,
        'and_indegree_violations': 0,
        'po_outdegree_violations': 0,
        'po_indegree_violations': 0,
        'isolated_nodes': 0, # Still counts relevant isolates for reporting
        'is_structurally_valid': False, # The key flag indicating validity
        'constraints_failed': [] # List to store reasons for failure
    }

    num_nodes = G.number_of_nodes()
    metrics['num_nodes'] = num_nodes

    if not isinstance(G, nx.DiGraph) or num_nodes == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object")
        metrics['is_structurally_valid'] = False # Explicitly set invalid
        return metrics # Return early for invalid input

    # 1. Check DAG property (Critical)
    try: # Add try-except for robustness
        metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
        if not metrics['is_dag']:
            metrics['constraints_failed'].append("Not a DAG")
    except Exception as e:
         logger.warning(f"DAG check failed for a graph: {e}")
         metrics['is_dag'] = False # Assume not DAG if check fails
         metrics['constraints_failed'].append(f"DAG Check Error: {e}")


    # 2. Check Node Types and Basic Degrees
    node_type_counts = Counter()
    unknown_node_indices = []
    for node, data in G.nodes(data=True):
        # Handle cases where node data might not be a dictionary
        if isinstance(data, dict):
             node_type = data.get('type')
        else:
             node_type = "Error: Node data not dict"
             logger.warning(f"Node {node} data is not a dictionary: {data}")

        node_type_counts[node_type] += 1

        # Use VALID_AIG_NODE_TYPES derived from config
        if node_type not in VALID_AIG_NODE_TYPES:
            metrics['num_unknown_nodes'] += 1
            unknown_node_indices.append(node)
            continue # Skip degree checks for unknown nodes

        # Check degrees - Add try-except for robustness
        try:
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
        except Exception as e:
             logger.warning(f"Could not get degree for node {node}: {e}")
             # Mark as violation? Or just skip checks? Let's add a general violation count later if invalid.
             continue

        # Check degrees based on assigned type (using defined type strings)
        if node_type == NODE_CONST0:
            metrics['num_const0'] += 1
            if in_deg != 0: metrics['const0_indegree_violations'] += 1
        elif node_type == NODE_PI:
            metrics['num_pi'] += 1
            if in_deg != 0: metrics['pi_indegree_violations'] += 1
        elif node_type == NODE_AND:
            metrics['num_and'] += 1
            if in_deg != 2: metrics['and_indegree_violations'] += 1
        elif node_type == NODE_PO:
            metrics['num_po'] += 1
            if out_deg != 0: metrics['po_outdegree_violations'] += 1
            # Changed check: PO *can* have 0 inputs if directly connected to PI/Const0 in minimal valid cases?
            # Original paper/common definition: PO must have inputs. Let's stick to that.
            if in_deg == 0: metrics['po_indegree_violations'] += 1 # Keep check for in_degree == 0


    # Add failure reasons based on type/degree checks to the list
    if metrics['num_unknown_nodes'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_nodes']} unknown node types")
    if metrics['const0_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['const0_indegree_violations']} CONST0 nodes with incorrect in-degree")
    if metrics['pi_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['pi_indegree_violations']} PI nodes with incorrect in-degree")
    if metrics['and_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['and_indegree_violations']} AND nodes with incorrect in-degree")
    if metrics['po_outdegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_outdegree_violations']} PO nodes with incorrect out-degree")
    if metrics['po_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_indegree_violations']} PO nodes with incorrect in-degree (0)")


    # 3. Check Edge Types
    for u, v, data in G.edges(data=True):
        # Handle cases where edge data might not be a dictionary
        if isinstance(data, dict):
             edge_type = data.get('type')
        else:
             edge_type = "Error: Edge data not dict"
             logger.warning(f"Edge ({u},{v}) data is not a dictionary: {data}")

        # Use VALID_AIG_EDGE_TYPES derived from config
        if edge_type is not None and edge_type not in VALID_AIG_EDGE_TYPES:
            metrics['num_unknown_edges'] += 1
    if metrics['num_unknown_edges'] > 0:
        # Add failure reason to the list
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_edges']} unknown edge types")

    # 4. Check Basic AIG Requirements (Using config values)
    if metrics['num_pi'] == 0 and metrics['num_const0'] == 0 :
         metrics['constraints_failed'].append("No Primary Inputs or Const0 found")
    # Use MIN_AND_COUNT_CONFIG and MIN_PO_COUNT_CONFIG safely
    min_and = MIN_AND_COUNT_CONFIG if 'MIN_AND_COUNT_CONFIG' in globals() else 1
    min_po = MIN_PO_COUNT_CONFIG if 'MIN_PO_COUNT_CONFIG' in globals() else 1
    if metrics['num_and'] < min_and :
        metrics['constraints_failed'].append(f"Insufficient AND gates ({metrics['num_and']} < {min_and})")
    if metrics['num_po'] < min_po:
        metrics['constraints_failed'].append(f"Insufficient POs ({metrics['num_po']} < {min_po})")

    # --- 5. Check isolated nodes (Keep calculation, but don't add to constraints_failed) ---
    try: # Add try-except for isolates calculation
        all_isolates = list(nx.isolates(G))
        relevant_isolates = []
        for node_idx in all_isolates:
             if node_idx not in G: continue # Check if node exists
             node_data = G.nodes[node_idx]
             isolated_node_type = node_data.get('type', None) if isinstance(node_data, dict) else None
             if isolated_node_type != NODE_CONST0: # Only count non-CONST0 nodes
                 relevant_isolates.append(node_idx)
        metrics['isolated_nodes'] = len(relevant_isolates) # Count relevant isolates for reporting
    except Exception as e:
         logger.warning(f"Isolate check failed for a graph: {e}")
         metrics['isolated_nodes'] = -1 # Indicate error


    # --- Final Validity Check ---
    # A graph is structurally valid IFF it passes ALL *critical* checks:
    is_valid = (
        metrics['is_dag'] and
        metrics['num_unknown_nodes'] == 0 and
        metrics['num_unknown_edges'] == 0 and
        metrics['const0_indegree_violations'] == 0 and
        metrics['pi_indegree_violations'] == 0 and
        metrics['and_indegree_violations'] == 0 and
        metrics['po_outdegree_violations'] == 0 and
        metrics['po_indegree_violations'] == 0 and
        (metrics['num_pi'] > 0 or metrics['num_const0'] > 0) and # At least one input source
        metrics['num_and'] >= min_and and # Use safe min_and
        metrics['num_po'] >= min_po # Use safe min_po
    )
    metrics['is_structurally_valid'] = is_valid
    # If invalid, ensure constraints_failed has at least one entry
    if not is_valid and not metrics['constraints_failed']:
         metrics['constraints_failed'].append("General Validity Check Failed")
    # --- End Final Validity Check ---

    return metrics


def count_pi_po_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Counts PIs reaching POs and POs reachable from PIs based on reachability.
    Uses assigned node types (defined globally from config). Assumes graph object is valid.
    """
    results = {
        'num_pi': 0, 'num_po': 0, 'num_const0': 0,
        'num_pis_reaching_po': 0, 'num_pos_reachable_from_pi': 0,
        'fraction_pis_connected': 0.0, 'fraction_pos_connected': 0.0,
        'error': None
    }
    if G.number_of_nodes() == 0:
        return results

    try:
        # Get nodes by assigned type (using defined type strings)
        pis = set()
        pos = set()
        const0_nodes = set()
        for node, data in G.nodes(data=True):
             # Ensure data is a dictionary before using .get()
             if isinstance(data, dict):
                  node_type = data.get('type')
                  if node_type == NODE_PI: pis.add(node)
                  elif node_type == NODE_PO: pos.add(node)
                  elif node_type == NODE_CONST0: const0_nodes.add(node)
             # else: logger.warning(f"Node {node} data is not a dictionary in count_pi_po_paths.")

        # Source nodes for path checking are PIs and Const0
        source_nodes = pis.union(const0_nodes)

        results['num_pi'] = len(pis) # Report actual PIs separately
        results['num_po'] = len(pos)
        results['num_const0'] = len(const0_nodes)

        if not source_nodes or not pos: # No paths possible if no sources or no POs
            return results

        connected_sources = set()
        connected_pos = set()

        # --- Perform Reachability Checks ---
        # Optimization: Precompute reachability from all sources and to all POs if needed frequently
        # For now, calculate individually

        # Find sources that can reach at least one PO
        for source_node in source_nodes:
             try:
                 if source_node not in G: continue
                 # Check reachability to ANY PO node
                 for po_node in pos:
                      if po_node not in G: continue
                      if nx.has_path(G, source_node, po_node):
                           connected_sources.add(source_node)
                           break # Source is connected, no need to check other POs for this source
             except nx.NodeNotFound: continue # Should not happen due to 'in G' check, but safe
             except Exception as e:
                  logger.warning(f"Path check failed for source {source_node}: {e}")
                  results['error'] = "Path check error" # Flag error


        # Find POs that are reachable from at least one source
        for po_node in pos:
             try:
                 if po_node not in G: continue
                 # Check reachability from ANY source node
                 for source_node in source_nodes:
                      if source_node not in G: continue
                      if nx.has_path(G, source_node, po_node):
                           connected_pos.add(po_node)
                           break # PO is connected, no need to check other sources for this PO
             except nx.NodeNotFound: continue
             except Exception as e:
                  logger.warning(f"Path check failed for PO {po_node}: {e}")
                  results['error'] = "Path check error" # Flag error

        # --- End Reachability Checks ---


        num_sources_total = len(source_nodes)
        results['num_pis_reaching_po'] = len(connected_sources) # Count includes const0 if connected
        results['num_pos_reachable_from_pi'] = len(connected_pos)

        # Calculate fractions
        if num_sources_total > 0: results['fraction_pis_connected'] = results['num_pis_reaching_po'] / num_sources_total
        if results['num_po'] > 0: results['fraction_pos_connected'] = results['num_pos_reachable_from_pi'] / results['num_po']

    except Exception as e:
        logger.error(f"Unexpected error during count_pi_po_paths execution: {e}", exc_info=True)
        results['error'] = f"Processing error: {e}"

    return results

def validate_aig_structures(graphs: List[nx.DiGraph]) -> float:
    """
    Validates a list of NetworkX DiGraphs based on structural AIG rules.

    Args:
        graphs: A list of NetworkX DiGraph objects representing AIGs.

    Returns:
        The fraction (0.0 to 1.0) of graphs that are structurally valid.
        Returns 0.0 if the input list is empty.
    """
    num_total = len(graphs)
    if num_total == 0:
        logger.warning("validate_aig_structures received an empty list of graphs.")
        return 0.0

    num_valid_structurally = 0
    for i, graph in enumerate(graphs):
        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} in list is not a NetworkX DiGraph, counting as invalid.")
            continue
        struct_metrics = calculate_structural_aig_metrics(graph)
        if struct_metrics.get('is_structurally_valid', False):
            num_valid_structurally += 1
        # else: logger.debug(f"Graph {i} failed validation: {struct_metrics.get('constraints_failed', [])}")

    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    logger.info(f"Validated {num_total} graphs. Structurally Valid: {num_valid_structurally} ({validity_fraction*100:.2f}%)")
    return validity_fraction

# --- Isomorphism Helper ---
node_matcher = nx.isomorphism.categorical_node_match('type', 'UNKNOWN')
edge_matcher = nx.isomorphism.categorical_edge_match('type', 'UNKNOWN_EDGE')

def are_graphs_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    """Checks isomorphism considering node/edge 'type' attributes."""
    try:
        return nx.is_isomorphic(G1, G2, node_match=node_matcher, edge_match=edge_matcher)
    except Exception as e:
        logger.warning(f"Isomorphism check failed between two graphs: {e}")
        return False

# --- Uniqueness Calculation ---
def calculate_uniqueness(valid_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    """Calculates uniqueness among valid graphs."""
    num_valid = len(valid_graphs)
    if num_valid <= 1: return (1.0, num_valid)

    unique_graph_indices = []
    logger.info(f"Calculating uniqueness for {num_valid} valid graphs...")
    for i in tqdm(range(num_valid), desc="Checking Uniqueness", leave=False):
        is_unique = True
        G1 = valid_graphs[i]
        for unique_idx in unique_graph_indices:
            G2 = valid_graphs[unique_idx]
            if are_graphs_isomorphic(G1, G2):
                is_unique = False; break
        if is_unique: unique_graph_indices.append(i)

    num_unique = len(unique_graph_indices)
    uniqueness_score = num_unique / num_valid
    logger.info(f"Found {num_unique} unique graphs out of {num_valid} valid graphs.")
    return uniqueness_score, num_unique

# --- Novelty Calculation ---
def calculate_novelty(valid_graphs: List[nx.DiGraph], train_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    """Calculates novelty against a training set."""
    num_valid = len(valid_graphs)
    num_train = len(train_graphs)
    if num_valid == 0: return (0.0, 0)
    if num_train == 0:
        logger.warning("Training set is empty, novelty will be 100%.")
        return (1.0, num_valid)

    num_novel = 0
    logger.info(f"Calculating novelty for {num_valid} valid graphs against {num_train} training graphs...")
    for gen_graph in tqdm(valid_graphs, desc="Checking Novelty", leave=False):
        is_novel = True
        for train_graph in train_graphs:
            if are_graphs_isomorphic(gen_graph, train_graph):
                is_novel = False; break
        if is_novel: num_novel += 1

    novelty_score = num_novel / num_valid
    logger.info(f"Found {num_novel} novel graphs out of {num_valid} valid graphs.")
    return novelty_score, num_novel

# --- NEW: Bin Data to NX Conversion Function ---
# (Copied and adapted from tests/graph_utils.py or datasets_utils.py logic)
def bin_data_to_nx(node_vocab_ids, edge_index, edge_vocab_ids):
    """Converts unpadded data (VOCAB IDs) from bin files into a NetworkX DiGraph."""
    if node_vocab_ids is None or node_vocab_ids.numel() == 0:
        # Return empty graph if no nodes
        return nx.DiGraph()

    G = nx.DiGraph()
    num_nodes = node_vocab_ids.size(0)

    # Precompute mappings from vocab IDs back to string names
    # These assume aig_config was imported successfully
    node_id_to_type_str = {v: k for k, v in aig_config.NODE_TYPE_VOCAB.items()}
    edge_id_to_type_str = {v: k for k, v in aig_config.EDGE_TYPE_VOCAB.items()}

    # Add nodes with string types
    for i in range(num_nodes):
        node_vocab_id = node_vocab_ids[i].item()
        node_type_str = node_id_to_type_str.get(node_vocab_id, 'UNKNOWN_NODE')
        if node_type_str == 'UNKNOWN_NODE':
             logger.warning(f"Node {i}: Unknown node vocab ID {node_vocab_id} encountered during bin->nx conversion.")
        G.add_node(i, type=node_type_str)

    # Add edges with string types
    if edge_index is not None and edge_index.numel() > 0:
        if edge_vocab_ids is None or edge_vocab_ids.numel() == 0 or edge_vocab_ids.shape[0] != edge_index.shape[1]:
             logger.error(f"Bin->NX Error: Edge attributes (vocab IDs) missing or shape mismatch with edge_index. Cannot add typed edges.")
             # Optionally add untyped edges or just skip
        else:
            edge_index_np = edge_index.cpu().numpy()
            num_edges = edge_index_np.shape[1]
            for i in range(num_edges):
                u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                edge_vocab_id = edge_vocab_ids[i].item()
                edge_type_str = edge_id_to_type_str.get(edge_vocab_id, 'UNKNOWN_EDGE')
                if edge_type_str == 'UNKNOWN_EDGE':
                    logger.warning(f"Edge ({u}, {v}): Unknown edge vocab ID {edge_vocab_id} encountered during bin->nx conversion.")
                # Ensure nodes exist before adding edge (should always be true if unpadding worked)
                if u in G and v in G:
                    G.add_edge(u, v, type=edge_type_str)
                else:
                    logger.warning(f"Bin->NX Error: Node {u} or {v} not found when adding edge. Skipping edge.")
    return G


# --- NEW: Training Graph Loader for Bin Files ---
def load_training_graphs_from_bin(train_data_dir: str, train_split_name: str = 'train') -> Optional[List[nx.DiGraph]]:
    """
    Loads training graphs from memmap bin files (xs.bin, etc.) located in a specified directory.

    Args:
        train_data_dir: The base directory containing data_meta.json and the split subfolder (e.g., './datasets/aig').
        train_split_name: The name of the subdirectory containing the bin files (default: 'train').

    Returns:
        A list of NetworkX DiGraphs, or None if loading fails.
    """
    meta_path = os.path.join(train_data_dir, 'data_meta.json')
    split_path = os.path.join(train_data_dir, train_split_name)
    logger.info(f"Attempting to load training graphs from bin files in: {split_path}")
    logger.info(f"Using metadata from: {meta_path}")

    if not os.path.isdir(split_path):
        logger.error(f"Training data split directory not found: {split_path}")
        return None
    if not os.path.exists(meta_path):
        logger.error(f"Metadata file not found: {meta_path}")
        return None

    # Load metadata to get shapes
    try:
        with open(meta_path, 'r') as f:
            data_meta = json.load(f)
        shape_key = f"{train_split_name}_shape"
        if shape_key not in data_meta:
            raise KeyError(f"Shape information for split '{train_split_name}' not found in {meta_path}")
        shape = data_meta[shape_key]
        num_train_graphs = shape['xs'][0]
        if num_train_graphs == 0:
             logger.warning(f"Metadata indicates 0 graphs in training split '{train_split_name}'.")
             return []
    except Exception as e:
        logger.error(f"Error loading or parsing metadata {meta_path}: {e}")
        return None

    # Open memmap files
    memmap_files = {}
    try:
        memmap_files['xs'] = np.memmap(os.path.join(split_path, 'xs.bin'), dtype=np.int16, mode='r', shape=tuple(shape['xs']))
        memmap_files['edge_indices'] = np.memmap(os.path.join(split_path, 'edge_indices.bin'), dtype=np.int16, mode='r', shape=tuple(shape['edge_indices']))
        memmap_files['edge_attrs'] = np.memmap(os.path.join(split_path, 'edge_attrs.bin'), dtype=np.int16, mode='r', shape=tuple(shape['edge_attrs']))
        # num_inputs/outputs not needed for novelty check itself
    except Exception as e:
        logger.error(f"Error opening memmap files in {split_path}: {e}")
        # Clean up any opened files
        for f in memmap_files.values(): del f
        return None

    train_graphs_nx = []
    logger.info(f"Loading and converting {num_train_graphs} training graphs from bin files...")

    # Iterate and unpad each graph
    for idx in tqdm(range(num_train_graphs), desc="Loading Train Graphs", leave=False):
        try:
            # 1. Read raw data for this index
            raw_x = np.array(memmap_files['xs'][idx]).astype(np.int64)
            raw_edge_index = np.array(memmap_files['edge_indices'][idx]).astype(np.int64) # Shape [2, max_E]
            raw_edge_attr = np.array(memmap_files['edge_attrs'][idx]).astype(np.int64) # Shape [max_E]

            # 2. Unpad nodes
            node_padding_mask = raw_x != PAD_VALUE
            x_ids = torch.from_numpy(raw_x[node_padding_mask])
            num_valid_nodes = len(x_ids)
            if num_valid_nodes == 0: continue # Skip empty graphs

            # 3. Create node index map
            old_indices = np.arange(len(raw_x))
            new_indices_map = -np.ones_like(old_indices, dtype=np.int64)
            new_indices_map[node_padding_mask] = np.arange(num_valid_nodes)

            # 4. Unpad edges
            if raw_edge_attr.ndim > 1: raw_edge_attr = raw_edge_attr.flatten()
            edge_padding_mask = raw_edge_attr != PAD_VALUE

            edge_index_final = torch.tensor([[], []], dtype=torch.long)
            edge_attr_final = torch.tensor([], dtype=torch.long)

            if edge_padding_mask.shape[0] != raw_edge_index.shape[1]:
                 # Shape mismatch, likely due to padding issues or data corruption
                 # Log warning but proceed to create graph with potentially missing edges
                 logger.warning(f"Train Graph {idx}: Shape mismatch edge_attr ({edge_padding_mask.shape[0]}) vs edge_index ({raw_edge_index.shape[1]}) after filtering.")
                 # Attempt to filter based on the shorter length if possible? Or just skip edges?
                 # Safest might be to create graph with only nodes here.
                 # Let's try filtering based on available attributes.
                 min_len = min(edge_padding_mask.shape[0], raw_edge_index.shape[1])
                 edge_padding_mask = edge_padding_mask[:min_len]
                 edge_attr_ids_filtered = torch.from_numpy(raw_edge_attr[:min_len][edge_padding_mask])
                 edge_index_filtered = torch.from_numpy(raw_edge_index[:, :min_len][:, edge_padding_mask])

                 # Proceed with remapping using filtered data
                 if edge_index_filtered.numel() > 0:
                    src_nodes_old = edge_index_filtered[0, :].numpy()
                    dst_nodes_old = edge_index_filtered[1, :].numpy()
                    # Clip indices to be safe before looking up in map
                    src_nodes_old = np.clip(src_nodes_old, 0, len(new_indices_map) - 1)
                    dst_nodes_old = np.clip(dst_nodes_old, 0, len(new_indices_map) - 1)
                    src_nodes_new = new_indices_map[src_nodes_old]
                    dst_nodes_new = new_indices_map[dst_nodes_old]
                    valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)
                    edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask], dst_nodes_new[valid_edge_mask]], dtype=torch.long)
                    edge_attr_final = edge_attr_ids_filtered[valid_edge_mask]

            else: # Shapes match
                 edge_attr_ids_filtered = torch.from_numpy(raw_edge_attr[edge_padding_mask])
                 edge_index_filtered = torch.from_numpy(raw_edge_index[:, edge_padding_mask])

                 # 5. Remap edge indices
                 if edge_index_filtered.numel() > 0:
                    src_nodes_old = edge_index_filtered[0, :].numpy()
                    dst_nodes_old = edge_index_filtered[1, :].numpy()
                    # Clip indices before mapping
                    src_nodes_old = np.clip(src_nodes_old, 0, len(new_indices_map) - 1)
                    dst_nodes_old = np.clip(dst_nodes_old, 0, len(new_indices_map) - 1)
                    src_nodes_new = new_indices_map[src_nodes_old]
                    dst_nodes_new = new_indices_map[dst_nodes_old]
                    valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)
                    edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask], dst_nodes_new[valid_edge_mask]], dtype=torch.long)
                    edge_attr_final = edge_attr_ids_filtered[valid_edge_mask]

            # 6. Convert unpadded data to NetworkX graph
            nx_graph = bin_data_to_nx(x_ids, edge_index_final, edge_attr_final)
            if nx_graph is not None:
                train_graphs_nx.append(nx_graph)
            else:
                 logger.warning(f"Failed to convert training graph index {idx} from bin data to NetworkX.")

        except Exception as e:
            logger.error(f"Error processing training graph index {idx} from bin files: {e}", exc_info=True)
            # Continue processing other graphs

    # Clean up memmap file handles
    for f in memmap_files.values(): del f

    logger.info(f"Finished loading and converting training graphs. Successfully processed {len(train_graphs_nx)} graphs.")
    return train_graphs_nx


# --- MODIFIED: Main Evaluation Logic ---
def run_standalone_evaluation(args):
    """Runs the evaluation including Validity, Uniqueness, and Novelty."""
    logger.info(f"Loading generated AIGs from: {args.input_pickle_file}")
    try:
        with open(args.input_pickle_file, 'rb') as f: generated_graphs = pickle.load(f)
        if not isinstance(generated_graphs, list): logger.error("Pickle file does not contain a list."); return
        logger.info(f"Loaded {len(generated_graphs)} generated graphs.")
    except FileNotFoundError: logger.error(f"Input pickle file not found: {args.input_pickle_file}"); return
    except Exception as e: logger.error(f"Error loading generated graphs pickle file: {e}"); return

    if not generated_graphs: logger.warning("No generated graphs found in the pickle file. Exiting."); return

    # --- Load Training Data (if path provided, now expects directory) ---
    train_graphs = None
    if args.train_data_dir:
        train_graphs = load_training_graphs_from_bin(args.train_data_dir)
    if args.train_data_dir and train_graphs is None:
         logger.warning(f"Could not load training graphs from {args.train_data_dir}. Novelty will not be calculated.")
    # --- End Load Training Data ---

    num_total = len(generated_graphs)
    valid_graphs = [] # Store the actual valid graph objects
    aggregate_metrics = defaultdict(list)
    aggregate_path_metrics = defaultdict(list)
    failed_constraints_summary = Counter()

    logger.info("Starting evaluation (Pass 1: Validity and Metrics)...")
    for i, graph in enumerate(tqdm(generated_graphs, desc="Evaluating Validity")):
        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph, skipping.")
            failed_constraints_summary["Invalid Graph Object"] += 1; continue

        struct_metrics = calculate_structural_aig_metrics(graph)

        # Aggregate structural metrics for all graphs
        for key, value in struct_metrics.items():
             # Ensure value is serializable and correct type before appending
             if isinstance(value, (int, float, bool)):
                aggregate_metrics[key].append(float(value))
             elif isinstance(value, list) and key == 'constraints_failed':
                 # Don't aggregate the list itself, handle summary later
                 pass

        # Store valid graphs and calculate path metrics for them
        if struct_metrics.get('is_structurally_valid', False): # Use .get for safety
            valid_graphs.append(graph) # Store the valid graph object
            try: # Add try-except for path metrics
                 path_metrics = count_pi_po_paths(graph)
                 if path_metrics.get('error') is None:
                    for key, value in path_metrics.items():
                        if isinstance(value, (int, float)): aggregate_path_metrics[key].append(value)
                 else: logger.warning(f"Skipping path metrics for valid graph {i} due to error: {path_metrics['error']}")
            except Exception as e:
                 logger.error(f"Error calculating path metrics for valid graph {i}: {e}")
        else:
            # Collect failure reasons only for invalid graphs
            for reason in struct_metrics.get('constraints_failed', ["Unknown Failure"]):
                failed_constraints_summary[reason] += 1

    logger.info("Evaluation (Pass 1) finished.")

    num_valid_structurally = len(valid_graphs)

    # --- Calculate Uniqueness ---
    uniqueness_score, num_unique = calculate_uniqueness(valid_graphs)
    # --- End Uniqueness ---

    # --- Calculate Novelty (if training data loaded) ---
    novelty_score, num_novel = (-1.0, -1) # Default values if not calculated
    if train_graphs is not None:
        novelty_score, num_novel = calculate_novelty(valid_graphs, train_graphs)
    # --- End Novelty ---

    # --- Reporting ---
    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    validity_percentage = validity_fraction * 100

    print("\n--- G2PT AIG V.U.N. Evaluation Summary ---")
    print(f"Total Graphs Loaded             : {num_total}")
    print(f"Structurally Valid AIGs (V)     : {num_valid_structurally} ({validity_percentage:.2f}%)")
    if num_valid_structurally > 0:
         print(f"Unique Valid AIGs             : {num_unique}")
         print(f"Uniqueness (U) among valid    : {uniqueness_score:.4f} ({uniqueness_score*100:.2f}%)")
         if train_graphs is not None:
             print(f"Novel Valid AIGs vs Train Set : {num_novel}")
             print(f"Novelty (N) among valid       : {novelty_score:.4f} ({novelty_score*100:.2f}%)")
         else:
             print(f"Novelty (N) among valid       : Not calculated (no training set provided)")
    else:
         print(f"Uniqueness (U) among valid    : N/A (0 valid graphs)")
         print(f"Novelty (N) among valid       : N/A (0 valid graphs)")

    print("\n--- Average Structural Metrics (All Generated Graphs) ---")
    for key, values in sorted(aggregate_metrics.items()):
        if key == 'is_structurally_valid': continue
        if not values: continue
        avg_value = np.mean(values)
        if key == 'is_dag': print(f"  - Avg {key:<27}: {avg_value*100:.2f}%")
        else: print(f"  - Avg {key:<27}: {avg_value:.3f}")

    print("\n--- Constraint Violation Summary (Across Invalid Graphs) ---")
    num_invalid_graphs = num_total - num_valid_structurally
    if num_invalid_graphs == 0: print("  No structural violations detected.")
    else:
        sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        print(f"  (Violations summarized across {num_invalid_graphs} invalid graphs)")
        total_violation_instances = sum(failed_constraints_summary.values())
        print(f"  (Total violation instances logged: {total_violation_instances})")
        for reason, count in sorted_reasons:
            reason_percentage_of_invalid = (count / num_invalid_graphs) * 100 if num_invalid_graphs > 0 else 0
            print(f"  - {reason:<45}: {count:<6} graphs ({reason_percentage_of_invalid:.1f}% of invalid)")

    print("\n--- Average Path Connectivity Metrics (Valid Graphs Only) ---")
    num_graphs_for_path_metrics = len(aggregate_path_metrics.get('num_pi', []))
    if num_graphs_for_path_metrics == 0: print("  No structurally valid graphs to calculate path metrics for.")
    else:
        print(f"  (Based on {num_graphs_for_path_metrics} structurally valid graphs)")
        for key, values in sorted(aggregate_path_metrics.items()):
             if key == 'error' or not values: continue
             avg_value = np.mean(values)
             print(f"  - Avg {key:<27}: {avg_value:.3f}")

    print("------------------------------------")


# --- MODIFIED: Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate generated AIGs for Validity, Uniqueness, and Novelty.')
    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs (e.g., generated_aigs.pkl).')
    # Modified argument to expect the directory containing train/ and data_meta.json
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='(Optional) Path to the training dataset directory (e.g., ./datasets/aig) for Novelty calculation. Expects data_meta.json and train/ subdir with bin files.')

    parsed_args = parser.parse_args()
    run_standalone_evaluation(parsed_args)