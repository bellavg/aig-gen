"""
Evaluate performance of the graph generation model through visualization,
MMD metrics, and AIG-specific validity, uniqueness, and novelty metrics.
"""

import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle # For loading training graphs
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple
import sys
import io # For capturing print output
import warnings
from tqdm import tqdm

# GraphRNN specific imports
import mmd
# Ensure data.py and extension_data.py are compatible or adapt if used for loading test sets
# For AIG evaluation, we'll primarily focus on evaluating sets of generated graphs.
# import data
# import extension_data
# from data import GraphDataSet # If test_graphs are loaded this way

# Ensure generate.py is the one that outputs (adj_matrix, node_attributes)
# and its load_model_from_config returns all necessary new params.
import generate # From graphrnn_generate_with_node_attrs artifact
# Ensure model.py is the one with node attribute prediction
import model # From graphrnn_model_with_node_attrs artifact

import graph_metrics # Original GraphRNN graph metrics
import mmd_stanford_impl # Original GraphRNN MMD implementation
import orbit_stats # Original GraphRNN orbit statistics

# --- AIG-specific imports (from your evaluate_aigs.py and aig_config.py) ---
# Make sure aig_config.py is in the Python path or adjust the import.
# For this example, assuming it can be imported directly.
try:
    from aig_config import (
        NODE_TYPE_KEYS, EDGE_TYPE_KEYS, MIN_AND_COUNT, MIN_PO_COUNT,
        DECODING_NODE_TYPE_NX, DECODING_EDGE_TYPE_NX, # For decoding one-hot vectors if needed by AIG funcs
        NODE_TYPE_ENCODING, EDGE_LABEL_ENCODING # For encoding during graph construction
    )
    AIG_CONFIG_LOADED = True
except ImportError:
    AIG_CONFIG_LOADED = False
    print("WARNING: Could not import from aig_config.py. AIG-specific evaluations will likely fail.")
    # Define placeholders if not loaded, so the script doesn't crash immediately
    NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"] # Example
    EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"] # Example
    MIN_AND_COUNT = 1
    MIN_PO_COUNT = 1
    # These are crucial for the AIG evaluation functions
    DECODING_NODE_TYPE_NX = {}
    DECODING_EDGE_TYPE_NX = {}
    NODE_TYPE_ENCODING = {k: [0.0]*len(NODE_TYPE_KEYS) for k in NODE_TYPE_KEYS}
    EDGE_LABEL_ENCODING = {k: [0.0]*len(EDGE_TYPE_KEYS) for k in EDGE_TYPE_KEYS}


# --- Logger Setup (from evaluate_aigs.py) ---
# Using GraphRNN's print statements for now, can integrate logging if preferred
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("graphrnn_evaluate_aigs")


# === Helper functions from evaluate_aigs.py (adapted for direct use) ===
# Define node type constants directly from NODE_TYPE_KEYS for clarity
_NODE_CONST0_STR = NODE_TYPE_KEYS[0] if NODE_TYPE_KEYS else "NODE_CONST0"
_NODE_PI_STR = NODE_TYPE_KEYS[1] if len(NODE_TYPE_KEYS) > 1 else "NODE_PI"
_NODE_AND_STR = NODE_TYPE_KEYS[2] if len(NODE_TYPE_KEYS) > 2 else "NODE_AND"
_NODE_PO_STR = NODE_TYPE_KEYS[3] if len(NODE_TYPE_KEYS) > 3 else "NODE_PO"

def get_node_type_str_from_attrs(attrs: dict) -> str:
    raw_type = attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    if isinstance(raw_type, str): return raw_type
    if isinstance(raw_type, (list, np.ndarray)):
        try:
            type_tuple = tuple(float(x) for x in raw_type)
            return DECODING_NODE_TYPE_NX.get(type_tuple, "UNKNOWN_ENCODING")
        except Exception: return "UNKNOWN_VECTOR_CONVERSION_ERROR"
    return f"UNKNOWN_TYPE_FORMAT_{type(raw_type).__name__}"

def get_edge_type_str_from_attrs(attrs: dict) -> str:
    raw_type = attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    if isinstance(raw_type, str): return raw_type
    if isinstance(raw_type, (list, np.ndarray)):
        try:
            type_tuple = tuple(float(x) for x in raw_type)
            return DECODING_EDGE_TYPE_NX.get(type_tuple, "UNKNOWN_ENCODING")
        except Exception: return "UNKNOWN_VECTOR_CONVERSION_ERROR"
    return f"UNKNOWN_TYPE_FORMAT_{type(raw_type).__name__}"

def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    if not AIG_CONFIG_LOADED: return {'is_structurally_valid': 0.0, 'constraints_failed': ['AIG_CONFIG_NOT_LOADED']}
    metrics = defaultdict(float)
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['constraints_failed'] = []

    if not isinstance(G, nx.DiGraph) or metrics['num_nodes'] == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object")
        metrics['is_structurally_valid'] = False
        return dict(metrics)

    try:
        metrics['is_dag'] = float(nx.is_directed_acyclic_graph(G))
        if not metrics['is_dag']: metrics['constraints_failed'].append("Not a DAG")
    except Exception: metrics['is_dag'] = 0.0; metrics['constraints_failed'].append("DAG Check Error")

    node_type_counts = Counter()
    for node, data in G.nodes(data=True):
        node_type_str = get_node_type_str_from_attrs(data)
        node_type_counts[node_type_str] += 1
        if node_type_str not in NODE_TYPE_KEYS:
            metrics['num_unknown_nodes'] += 1
            if "Unknown node types" not in metrics['constraints_failed']: metrics['constraints_failed'].append("Unknown node types")
            continue
        try:
            in_deg, out_deg = G.in_degree(node), G.out_degree(node)
        except Exception: metrics['degree_check_errors'] += 1; continue

        if node_type_str == _NODE_CONST0_STR:
            if in_deg != 0: metrics['const0_indegree_violations'] += 1
        elif node_type_str == _NODE_PI_STR:
            if in_deg != 0: metrics['pi_indegree_violations'] += 1
        elif node_type_str == _NODE_AND_STR:
            if in_deg != 2: metrics['and_indegree_violations'] += 1 # Strict check
        elif node_type_str == _NODE_PO_STR:
            if out_deg != 0: metrics['po_outdegree_violations'] += 1
            if in_deg == 0: metrics['po_indegree_violations'] += 1 # PO must have input

    metrics['num_pi'] = float(node_type_counts.get(_NODE_PI_STR, 0))
    metrics['num_po'] = float(node_type_counts.get(_NODE_PO_STR, 0))
    metrics['num_and'] = float(node_type_counts.get(_NODE_AND_STR, 0))

    if metrics['const0_indegree_violations'] > 0: metrics['constraints_failed'].append("CONST0 in-degree != 0")
    if metrics['pi_indegree_violations'] > 0: metrics['constraints_failed'].append("PI in-degree != 0")
    if metrics['and_indegree_violations'] > 0: metrics['constraints_failed'].append("AND in-degree != 2")
    if metrics['po_outdegree_violations'] > 0: metrics['constraints_failed'].append("PO out-degree != 0")
    if metrics['po_indegree_violations'] > 0: metrics['constraints_failed'].append("PO in-degree == 0")

    for u, v, data in G.edges(data=True):
        edge_type_str = get_edge_type_str_from_attrs(data)
        if edge_type_str not in EDGE_TYPE_KEYS:
            metrics['num_unknown_edges'] += 1
            if "Unknown edge types" not in metrics['constraints_failed']: metrics['constraints_failed'].append("Unknown edge types")

    if metrics['num_pi'] == 0 and metrics.get('num_const0', 0.0) == 0: metrics['constraints_failed'].append("No PIs or Const0")
    if metrics['num_and'] < MIN_AND_COUNT: metrics['constraints_failed'].append(f"AND gates < {MIN_AND_COUNT}")
    if metrics['num_po'] < MIN_PO_COUNT: metrics['constraints_failed'].append(f"POs < {MIN_PO_COUNT}")

    is_valid = (
            metrics['is_dag'] > 0.5 and metrics['num_unknown_nodes'] == 0 and
            metrics['num_unknown_edges'] == 0 and metrics['const0_indegree_violations'] == 0 and
            metrics['pi_indegree_violations'] == 0 and metrics['and_indegree_violations'] == 0 and
            metrics['po_outdegree_violations'] == 0 and metrics['po_indegree_violations'] == 0 and
            (metrics['num_pi'] > 0 or metrics.get('num_const0', 0.0) > 0) and
            metrics['num_and'] >= MIN_AND_COUNT and metrics['num_po'] >= MIN_PO_COUNT
    )
    metrics['is_structurally_valid'] = float(is_valid)
    if not is_valid and not metrics['constraints_failed']: metrics['constraints_failed'].append("General Validity Rules Failed")
    return dict(metrics)

def node_match_aig_flexible(node1_attrs, node2_attrs):
    if not AIG_CONFIG_LOADED: return False # Cannot match if config is missing
    type1_str = get_node_type_str_from_attrs(node1_attrs)
    type2_str = get_node_type_str_from_attrs(node2_attrs)
    if "UNKNOWN" in type1_str and "UNKNOWN" not in type2_str: return False
    if "UNKNOWN" in type2_str and "UNKNOWN" not in type1_str: return False
    return type1_str == type2_str

def edge_match_aig_flexible(edge1_attrs, edge2_attrs):
    if not AIG_CONFIG_LOADED: return False
    type1_str = get_edge_type_str_from_attrs(edge1_attrs)
    type2_str = get_edge_type_str_from_attrs(edge2_attrs)
    if "UNKNOWN" in type1_str and "UNKNOWN" not in type2_str: return False
    if "UNKNOWN" in type2_str and "UNKNOWN" not in type1_str: return False
    return type1_str == type2_str

def are_graphs_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    if G1 is None or G2 is None: return False
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False
    try:
        return nx.is_isomorphic(G1, G2, node_match=node_match_aig_flexible, edge_match=edge_match_aig_flexible)
    except Exception: return False

def calculate_uniqueness(valid_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs)
    if num_valid <= 1: return (1.0, num_valid)
    unique_graph_representatives = []
    for i in tqdm(range(num_valid), desc="Checking Uniqueness", leave=False, ncols=80):
        current_graph = valid_graphs[i]
        is_unique_to_list = True
        for representative_graph in unique_graph_representatives:
            if are_graphs_isomorphic(current_graph, representative_graph):
                is_unique_to_list = False; break
        if is_unique_to_list: unique_graph_representatives.append(current_graph)
    num_unique = len(unique_graph_representatives)
    return (num_unique / num_valid if num_valid > 0 else 0.0), num_unique

def calculate_novelty(valid_graphs: List[nx.DiGraph], train_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs)
    if num_valid == 0: return (0.0, 0)
    if not train_graphs: return (1.0, num_valid) # All are novel if no training set
    num_novel = 0
    for gen_graph in tqdm(valid_graphs, desc="Checking Novelty", leave=False, ncols=80):
        is_novel_to_train_set = True
        for train_graph in train_graphs:
            if are_graphs_isomorphic(gen_graph, train_graph):
                is_novel_to_train_set = False; break
        if is_novel_to_train_set: num_novel += 1
    return (num_novel / num_valid if num_valid > 0 else 0.0), num_novel

def load_graphs_from_pkl(pkl_file_path: str) -> Optional[List[nx.DiGraph]]:
    if not os.path.exists(pkl_file_path):
        print(f"Error: PKL file not found: {pkl_file_path}"); return None
    try:
        with open(pkl_file_path, 'rb') as f:
            graphs = pickle.load(f)
        if isinstance(graphs, list) and all(isinstance(g, nx.DiGraph) for g in graphs):
            return graphs
        else:
            print(f"Error: Content of {pkl_file_path} is not a list of NetworkX DiGraphs.")
            return None
    except Exception as e:
        print(f"Error loading graphs from {pkl_file_path}: {e}"); return None

# === GraphRNN's original evaluation functions (some might be adapted/reused) ===

def generated_output_to_aig_networkx(
    adj_matrix_scalar_edges: np.ndarray,
    node_attrs_one_hot: np.ndarray,
    mode: str # To interpret adj_matrix_scalar_edges correctly
    ) -> nx.DiGraph:
    """
    Converts raw output from GraphRNN's generate function (adj matrix with scalar edge classes
    and one-hot node attributes) into a NetworkX DiGraph with 'type' attributes
    set as one-hot vectors for AIG evaluation.
    """
    if not AIG_CONFIG_LOADED:
        warnings.warn("AIG_CONFIG not loaded, cannot properly create AIG NetworkX graph.")
        return nx.DiGraph() # Return empty graph

    G = nx.DiGraph()
    num_nodes = node_attrs_one_hot.shape[0]

    # Add nodes with their one-hot 'type' attributes
    for i in range(num_nodes):
        # Ensure node_type_vector is a list for NetworkX attribute
        node_type_vector = node_attrs_one_hot[i, :].tolist()
        G.add_node(i, type=node_type_vector)

    # Add edges with one-hot 'type' attributes
    # adj_matrix_scalar_edges[i, j] is the scalar class of edge i -> j
    # For AIGs (mode='aig-custom-topsort'), class 0 might be EDGE_REG, class 1 EDGE_INV.
    # A "no edge" class isn't explicitly output by generate.py's adj_matrix construction.
    # The adj_matrix is constructed based on S_i vectors which imply connections.
    # We need to map these scalar classes back to the one-hot EDGE_LABEL_ENCODING.

    # Create a reverse map from scalar index to one-hot vector for edges
    # This assumes EDGE_TYPE_KEYS order matches the scalar classes (0, 1, ...)
    scalar_to_one_hot_edge = {}
    for idx, key in enumerate(EDGE_TYPE_KEYS):
        scalar_to_one_hot_edge[idx] = EDGE_LABEL_ENCODING[key]

    # If the adj_matrix_scalar_edges is for the original 'directed-multiclass' (4 classes)
    # this mapping will be different. For AIGs, we assume it's simpler.

    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_scalar_class = int(adj_matrix_scalar_edges[i, j])

            if mode == 'aig-custom-topsort': # Or your specific AIG mode
                # Assume non-zero scalar class means an edge exists.
                # Class 0 could be EDGE_REG, Class 1 EDGE_INV.
                # If adj_matrix_scalar_edges was built from S_i that are binary (0=no, 1=edge),
                # then this needs a different interpretation.
                # The generate.py for graphrnn_generate_with_node_attrs stores argmax for multiclass edges.
                if edge_scalar_class in scalar_to_one_hot_edge: # Check if it's a valid edge class index
                    # This implies that the adj_matrix_scalar_edges contains meaningful class indices
                    # where an edge exists. If it can contain a "no edge" class, that needs handling.
                    # The m_seq_to_adj_mat fills with 0s by default.
                    # If 0 is a valid edge class (e.g. EDGE_REG), this is fine.
                    # If 0 means "no edge" for your scalar classes, then skip G.add_edge.
                    # Let's assume generate.py's adj_matrix_final has non-zero for actual edges
                    # and the value is the class index.
                    # For AIG, if 0=REG, 1=INV, and m_seq_to_adj_mat puts these values,
                    # we need to check if the value means an edge.
                    # The `list_adj_vecs_for_matrix` in generate.py stores 0 for no edge within M-window.
                    # So, `adj_matrix_scalar_edges` from `m_seq_to_adj_mat` will have 0 for no edge.
                    # We need a convention: does class 0 (e.g. EDGE_REG) mean an edge, or is there a separate "no edge" indicator?
                    # Assuming for AIG, if adj_matrix_scalar_edges[i,j] is a valid key in scalar_to_one_hot_edge, it's an edge.
                    # This means your scalar edge classes should not include a "no edge" type if using this logic.
                    # Or, if 0 can be EDGE_REG, and also "no edge", that's ambiguous.

                    # Let's assume the adj_matrix from generate.py has 0 for no edge,
                    # and 1, 2, ... for edge types (or 0, 1 for your 2 types if re-indexed).
                    # The `processed_slice_for_storage` in generate.py takes argmax.
                    # So, it will be 0 for first class, 1 for second.
                    # If an edge slot in S_i was all zeros, argmax might give 0.
                    # This needs careful alignment with how S_i is stored and m_seq_to_adj_mat works.

                    # Safest: Assume adj_matrix_scalar_edges[i,j] > 0 means an edge of class (value-1) if classes are 1-indexed.
                    # Or if it's 0-indexed (0=REG, 1=INV), then any of these values means an edge.
                    # The current `m_seq_to_adj_mat` directly places the scalar values.
                    # If `list_adj_vecs_for_matrix` has 0 for EDGE_REG and 1 for EDGE_INV,
                    # then `adj_matrix_scalar_edges[i,j]` will be 0 or 1.
                    # We need to distinguish "no edge" from "edge of type 0".
                    # This is a common issue. GraphRNN's original S_i was binary.
                    # For multiclass, the `list_adj_vecs_for_matrix` stores argmax.
                    # If the one-hot for "no edge" was `[1,0,0,..]` and argmax is 0,
                    # then class 0 could mean "no edge".

                    # Let's assume for 'aig-custom-topsort', the adj_matrix_scalar_edges
                    # ALREADY represents the structure correctly, and values are class indices
                    # for EXISTING edges. And 0 means no edge if it's not a valid class index.
                    # This is hard to reconcile perfectly without seeing the exact values in adj_matrix_scalar_edges.

                    # Simplification: If adj_matrix_scalar_edges[i,j] is a key in scalar_to_one_hot_edge,
                    # it implies an edge of that type. This means your "no edge" should not be a key.
                    # This is problematic if class 0 (e.g. EDGE_REG) is a valid edge type.

                    # Alternative: assume m_seq_to_adj_mat produces a matrix where non-zero entries are edge class indices.
                    # This is not how it's written. It just places the numbers.

                    # Let's assume the output of `generate.py`'s `adj_matrix_final` for `aig-custom-topsort`
                    # is such that `adj_matrix_final[i,j]` gives the scalar class (0 for REG, 1 for INV)
                    # IF an edge exists, and some other value (or it's implicit from structure for DAGs)
                    # if no edge. This is the trickiest part.
                    # The `m_seq_to_adj_mat` fills with 0s. If 0 is EDGE_REG, we can't tell "no edge" from "EDGE_REG".

                    # Revisit generate.py: `final_slice_for_storage` is `np.zeros(m_window_size, dtype=int)`.
                    # `processed_slice_for_storage` (argmax result) is put into it.
                    # So, if no edge was predicted for a slot in M-window, it remains 0.
                    # If EDGE_REG is class 0, then `adj_matrix_scalar_edges[i,j] == 0` could be EDGE_REG or no edge.

                    # **NECESSARY CONVENTION:**
                    # For 'aig-custom-topsort', let's assume generate.py's `adj_matrix_final`
                    # needs one more transformation: if it contains class indices (e.g., 0 for REG, 1 for INV),
                    # we need a way to know if an edge *exists*.
                    # The original GraphRNN `adj_matrix` was binary.
                    # The `directed-multiclass` (4-class) had class 0 for "no edge".
                    # For your AIG, you have 2 edge types. You need to decide if your model
                    # predicts "no edge" as a third class, or if presence is implicit.
                    # If presence is implicit (any valid class index means edge), then how does `m_seq_to_adj_mat`
                    # represent "no edge" if its default fill is 0, and 0 is EDGE_REG?

                    # Easiest Path: Modify `generate.py` so that for `aig-custom-topsort`,
                    # `list_adj_vecs_for_matrix` stores, e.g., -1 for no edge, and 0/1 for edge types.
                    # Then `m_seq_to_adj_mat` would produce a matrix with -1 for no edge.
                    # Here, we'd check `if edge_scalar_class != -1:`
                    # For now, assuming current `adj_matrix_scalar_edges` has 0/1 for edge types and
                    # we need external info or a different matrix for structure.

                    # This function will assume adj_matrix_scalar_edges[i,j] contains the scalar class
                    # AND that this implies an edge exists. This is a strong assumption.
                    if edge_scalar_class in scalar_to_one_hot_edge: # If it's a valid type index
                         edge_type_one_hot = scalar_to_one_hot_edge[edge_scalar_class]
                         G.add_edge(i, j, type=edge_type_one_hot)

            elif mode == 'directed-multiclass': # Original GraphRNN 4-class
                # class 0 = no edge, 1 = fwd, 2 = bwd, 3 = bi
                if edge_scalar_class == 1: # forward i->j
                    G.add_edge(i, j, type=EDGE_LABEL_ENCODING.get(EDGE_TYPE_KEYS[0],[1.0,0.0])) # Default to first AIG type
                elif edge_scalar_class == 2: # backward j->i
                    G.add_edge(j, i, type=EDGE_LABEL_ENCODING.get(EDGE_TYPE_KEYS[0],[1.0,0.0]))
                elif edge_scalar_class == 3: # bidirectional
                    G.add_edge(i, j, type=EDGE_LABEL_ENCODING.get(EDGE_TYPE_KEYS[0],[1.0,0.0]))
                    G.add_edge(j, i, type=EDGE_LABEL_ENCODING.get(EDGE_TYPE_KEYS[0],[1.0,0.0]))
            # For 'undirected' or 'directed-topsort' (binary), adj_matrix_scalar_edges is binary
            elif edge_scalar_class == 1: # Edge exists
                 # Assign a default AIG edge type if mode is not AIG-specific
                 default_aig_edge_type = EDGE_LABEL_ENCODING.get(EDGE_TYPE_KEYS[0], [1.0,0.0])
                 G.add_edge(i, j, type=default_aig_edge_type)
    return G


# --- Original GraphRNN MMD functions (largely unchanged) ---
def original_generated_graph_to_networkx(adj_matrix, directed=True):
    # This was GraphRNN's original converter for binary adj matrices
    # or matrices where value indicates existence.
    return nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph if directed else None)

def draw_generated_graph(graph_nx, file_name="graph_drawing", title="Generated Graph"):
    """Draws a networkx graph."""
    if graph_nx.number_of_nodes() == 0:
        print(f"Skipping drawing for empty graph: {file_name}")
        return
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(graph_nx, k=1 / (np.sqrt(graph_nx.number_of_nodes()) + 1e-6), iterations=50)

    # Simple drawing for now, can be enhanced to show node/edge types
    nx.draw(graph_nx, pos=pos, with_labels=True, node_size=300, font_size=8)
    plt.title(title)
    plt.savefig(f"{file_name}.png")
    plt.close()

# ... (Keep other MMD helper functions from original GraphRNN evaluate.py) ...
# _diff_func, compare_graphs_avg_degree, etc.
# get_orbit_stats, compare_graphs_mmd_orbit_stats, etc.
# These will operate on the NetworkX graphs, which now have AIG types.
# Their interpretation might change slightly (e.g. degree is just degree, not type-aware).

def compare_graphs_avg_clustering_coeff(graph1, graph2):
    if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0: return 1.0 # Max diff if one is empty
    with warnings.catch_warnings(): # NetworkX can issue warnings for clustering on DiGraphs
        warnings.simplefilter("ignore", category=UserWarning)
        avg1 = np.mean(np.array(list(nx.clustering(graph1.to_undirected()).values()))) if graph1.number_of_nodes() > 0 else 0
        avg2 = np.mean(np.array(list(nx.clustering(graph2.to_undirected()).values()))) if graph2.number_of_nodes() > 0 else 0
    return abs(avg1 - avg2)

# --- Main Evaluation Orchestration ---
def run_evaluation_suite(
    generated_graph_outputs: List[Tuple[np.ndarray, np.ndarray]], # List of (adj_matrix, node_attrs)
    generation_mode: str, # Mode used by generate.py (e.g. 'aig-custom-topsort')
    training_graphs_nx: Optional[List[nx.DiGraph]] = None, # Loaded training AIGs
    config: Optional[Dict] = None # Model config, if needed for params
    ):
    """
    Runs the full evaluation suite: AIG V.U.N. and GraphRNN MMD metrics.
    """
    if not AIG_CONFIG_LOADED and generation_mode == 'aig-custom-topsort':
        print("Critical: AIG_CONFIG not loaded. AIG-specific evaluation cannot proceed meaningfully.")
        # return {} # Or some error state

    print(f"Converting {len(generated_graph_outputs)} raw generated outputs to NetworkX AIGs...")
    generated_aigs_nx = []
    for adj_matrix, node_attrs in tqdm(generated_graph_outputs, desc="Converting to NX", ncols=80):
        if adj_matrix.size == 0 and node_attrs.size == 0 : # Skip if generate produced empty
            print("Skipping an empty graph output from generation.")
            continue
        nx_aig = generated_output_to_aig_networkx(adj_matrix, node_attrs, generation_mode)
        generated_aigs_nx.append(nx_aig)

    if not generated_aigs_nx:
        print("No graphs were successfully converted to NetworkX format. Evaluation aborted.")
        return {}

    print(f"\n--- Starting AIG V.U.N. Evaluation for {len(generated_aigs_nx)} graphs ---")
    results = {}

    # 1. AIG Validity
    num_total_generated = len(generated_aigs_nx)
    structurally_valid_aigs = []
    aig_metric_sums = defaultdict(float)
    constraints_failed_summary = Counter()

    for G_idx, G_aig in enumerate(tqdm(generated_aigs_nx, desc="AIG Validity Check", ncols=80)):
        metrics = calculate_structural_aig_metrics(G_aig)
        for k, v in metrics.items():
            if isinstance(v, (float, int)): aig_metric_sums[k] += v
        if metrics.get('is_structurally_valid', 0.0) > 0.5:
            structurally_valid_aigs.append(G_aig)
        elif 'constraints_failed' in metrics:
            for reason in metrics['constraints_failed']:
                constraints_failed_summary[reason] += 1

    results['aig_num_generated'] = num_total_generated
    results['aig_num_structurally_valid'] = len(structurally_valid_aigs)
    results['aig_validity_fraction'] = (len(structurally_valid_aigs) / num_total_generated) if num_total_generated > 0 else 0.0

    print(f"AIG Structural Validity: {results['aig_num_structurally_valid']}/{results['aig_num_generated']} ({results['aig_validity_fraction']*100:.2f}%)")
    if constraints_failed_summary:
        print("AIG Constraint Failures:")
        for reason, count in constraints_failed_summary.items():
            print(f"  - {reason}: {count}")

    # 2. AIG Uniqueness (on structurally valid AIGs)
    if structurally_valid_aigs:
        uniqueness_score, num_unique = calculate_uniqueness(structurally_valid_aigs)
        results['aig_uniqueness_score'] = uniqueness_score
        results['aig_num_unique_valid'] = num_unique
        print(f"AIG Uniqueness (among valid): {num_unique}/{len(structurally_valid_aigs)} ({uniqueness_score*100:.2f}%)")
    else:
        results['aig_uniqueness_score'] = 0.0
        results['aig_num_unique_valid'] = 0
        print("AIG Uniqueness: N/A (0 structurally valid AIGs)")

    # 3. AIG Novelty (on structurally valid AIGs, if training data provided)
    if structurally_valid_aigs and training_graphs_nx:
        novelty_score, num_novel = calculate_novelty(structurally_valid_aigs, training_graphs_nx)
        results['aig_novelty_score'] = novelty_score
        results['aig_num_novel_valid'] = num_novel
        print(f"AIG Novelty (vs {len(training_graphs_nx)} train graphs): {num_novel}/{len(structurally_valid_aigs)} ({novelty_score*100:.2f}%)")
    elif structurally_valid_aigs:
        results['aig_novelty_score'] = 1.0 # All are novel if no training set
        results['aig_num_novel_valid'] = len(structurally_valid_aigs)
        print(f"AIG Novelty: 100% (No training set provided for comparison)")
    else:
        results['aig_novelty_score'] = 0.0
        results['aig_num_novel_valid'] = 0
        print("AIG Novelty: N/A (0 structurally valid AIGs)")

    # --- GraphRNN MMD Metrics (can be run on `generated_aigs_nx` or a subset) ---
    # For MMD, we typically compare against a test set.
    # If `test_graphs_nx` (a list of NetworkX graphs from a test dataset) is available:
    # test_graphs_nx = ... # Load your test set NX graphs here

    # For now, let's assume MMDs are calculated if a test set is provided as an argument.
    # The original GraphRNN evaluate.py loads test graphs using its `data.GraphDataSet`.
    # This part needs to be adapted based on how you want to provide the "ground truth" distribution.

    # Placeholder for MMDs - you'd integrate the original MMD calculation loop here,
    # using `generated_aigs_nx` as one set and a loaded test set of NX graphs as the other.
    # Example:
    # if test_graphs_nx:
    #     print("\n--- Starting GraphRNN MMD Evaluation ---")
    #     # ... (use mmd_stanford_fn_... and compare_graphs_mmd_... functions) ...
    #     # results['mmd_degree'] = compare_graphs_mmd_degree(test_graphs_nx, generated_aigs_nx, mmd_func)
    # else:
    #     print("\n--- GraphRNN MMD Evaluation: Skipped (No test set provided) ---")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate generated graphs from GraphRNN.")
    parser.add_argument('model_path', help='Path of the trained GraphRNN model checkpoint (.pth file)')
    parser.add_argument('--num_graphs_to_generate', type=int, default=100, help='Number of graphs to generate for evaluation.')
    parser.add_argument('--num_nodes_per_graph', type=int, default=None,
                        help='Target number of nodes for each generated graph. If None, uses a small default or varies.')
    # AIG specific args
    parser.add_argument('--aig_train_set_pkl', type=str, default=None,
                        help='(Optional) Path to PKL file containing a list of training NetworkX AIGs for novelty calculation.')
    parser.add_argument('--results_file', type=str, default="graphrnn_aig_eval_results.txt",
                        help="Filename to save the evaluation summary.")

    args = parser.parse_args()

    if not AIG_CONFIG_LOADED:
        sys.exit("Error: aig_config.py could not be loaded. Please ensure it's in the Python path. Exiting.")

    # 1. Load GraphRNN Model
    print(f"Loading model from: {args.model_path}")
    (node_model, edge_model, m_param,
     num_node_classes, edge_feat_len,
     edge_gen_func, model_mode) = generate.load_model_from_config(args.model_path)

    print(f"Model loaded. Mode: {model_mode}, M: {m_param}, NodeClasses: {num_node_classes}, EdgeFeatLen: {edge_feat_len}")

    # 2. Generate Graphs
    print(f"\nGenerating {args.num_graphs_to_generate} graphs...")
    raw_generated_outputs = []

    # Determine num_nodes for generation
    # GraphRNN's original generate_new_graphs matches node counts from a test set.
    # Here, we generate with a fixed or default num_nodes_per_graph.
    nodes_to_generate = args.num_nodes_per_graph if args.num_nodes_per_graph is not None else np.random.randint(10, 21) # Example default range

    for i in tqdm(range(args.num_graphs_to_generate), desc="Generating Graphs", ncols=80):
        if args.num_nodes_per_graph is None: # Vary node count if not specified
            nodes_to_generate = np.random.randint(max(2, MIN_PI_COUNT + MIN_PO_COUNT),
                                                  config.get('data',{}).get('max_node_count', 31) if 'config' in locals() else 31)


        adj_matrix, node_attrs = generate.generate(
            num_total_nodes=nodes_to_generate,
            node_model=node_model,
            edge_model=edge_model,
            m_window_size=m_param,
            num_node_classes=num_node_classes,
            edge_feature_len_model=edge_feat_len,
            edge_gen_function=edge_gen_func,
            mode=model_mode
        )
        if adj_matrix.size > 0 : # Only add if something was generated
             raw_generated_outputs.append((adj_matrix, node_attrs))
        elif i < 5 : # Print warning for first few empty generations
             print(f"Warning: Graph {i+1} generation resulted in an empty graph (0 nodes).")


    if not raw_generated_outputs:
        print("No graphs were generated. Exiting evaluation.")
        sys.exit(0)
    print(f"Successfully generated {len(raw_generated_outputs)} non-empty raw graph outputs.")

    # 3. Load Training AIGs (if provided for novelty)
    training_aigs_nx = None
    if args.aig_train_set_pkl:
        print(f"\nLoading training AIGs from: {args.aig_train_set_pkl}")
        training_aigs_nx = load_graphs_from_pkl(args.aig_train_set_pkl)
        if training_aigs_nx:
            print(f"Loaded {len(training_aigs_nx)} training AIGs.")
        else:
            print(f"Warning: Could not load training AIGs from {args.aig_train_set_pkl}.")

    # 4. Run Evaluation Suite
    eval_results = run_evaluation_suite(
        raw_generated_outputs,
        generation_mode=model_mode, # Pass the model's generation mode
        training_graphs_nx=training_aigs_nx,
        config=None # Pass loaded config if run_evaluation_suite needs it
    )

    # 5. Report Results
    print("\n--- Overall Evaluation Summary ---")
    output_buffer = io.StringIO()
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}", file=output_buffer)
        else:
            print(f"{key}: {value}", file=output_buffer)

    results_string = output_buffer.getvalue()
    print(results_string)

    with open(args.results_file, 'w') as f:
        f.write(f"Evaluation Results for Model: {args.model_path}\n")
        f.write(f"Number of Graphs Generated for Eval: {len(raw_generated_outputs)} (Target: {args.num_graphs_to_generate})\n")
        f.write(f"Target Nodes Per Graph: {'Varied' if args.num_nodes_per_graph is None else args.num_nodes_per_graph}\n")
        if training_aigs_nx:
            f.write(f"Training AIGs for Novelty: {len(training_aigs_nx)} from {args.aig_train_set_pkl}\n")
        else:
            f.write("Training AIGs for Novelty: Not loaded.\n")
        f.write("-------------------------------------\n")
        f.write(results_string)
    print(f"Evaluation summary saved to: {args.results_file}")

    # Optional: Draw some of the first few valid AIGs
    # num_to_draw = 5
    # valid_drawn_count = 0
    # if 'structurally_valid_aigs' in eval_results and eval_results['structurally_valid_aigs']:
    #     print(f"\nDrawing up to {num_to_draw} valid generated AIGs...")
    #     for i, nx_graph in enumerate(eval_results['structurally_valid_aigs']):
    #         if i >= num_to_draw: break
    #         draw_generated_graph(nx_graph, file_name=f"valid_aig_sample_{i}", title=f"Valid AIG Sample {i}")
    #         valid_drawn_count +=1
    #     if valid_drawn_count > 0: print(f"Saved drawings for {valid_drawn_count} valid AIGs.")

    print("\nEvaluation finished.")
