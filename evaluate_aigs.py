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
# Removed json, torch imports as they were primarily for bin loading

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluate_aigs_pkl")

# --- Import the AIG configuration ---
try:
    # Assuming aig_config.py is accessible
    import data.aig_config as aig_config
except ImportError:
    import G2PT.configs.aig as aig_config

# --- Constants from Config or Fallbacks ---
if aig_config:
    NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
    EDGE_TYPE_KEYS = getattr(aig_config, 'EDGE_TYPE_KEYS', ["EDGE_REG", "EDGE_INV"]) # Needed for isomorphism check maybe?
    VALID_AIG_NODE_TYPES = set(NODE_TYPE_KEYS)
    VALID_AIG_EDGE_TYPES = set(EDGE_TYPE_KEYS)
    NODE_CONST0 = getattr(aig_config, 'NODE_CONST0_KEY', "NODE_CONST0")
    NODE_PI = getattr(aig_config, 'NODE_PI_KEY', "NODE_PI")
    NODE_AND = getattr(aig_config, 'NODE_AND_KEY', "NODE_AND")
    NODE_PO = getattr(aig_config, 'NODE_PO_KEY', "NODE_PO")
    MIN_AND_COUNT_CONFIG = getattr(aig_config, 'MIN_AND_COUNT', 1)
    MIN_PO_COUNT_CONFIG = getattr(aig_config, 'MIN_PO_COUNT', 1)
    NODE_TYPE_ENCODING = getattr(aig_config, 'NODE_TYPE_ENCODING', {
        "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], "NODE_PI": [0.0, 1.0, 0.0, 0.0],
        "NODE_AND": [0.0, 0.0, 1.0, 0.0], "NODE_PO": [0.0, 0.0, 0.0, 1.0]
    })
    EDGE_LABEL_ENCODING = getattr(aig_config, 'EDGE_LABEL_ENCODING', {
         "EDGE_REG": [1.0, 0.0], "EDGE_INV": [0.0, 1.0]
    })
    # Create reverse mappings if needed for isomorphism check or validation
    ONE_HOT_TO_NODE_TYPE_STR = {tuple(float(x) for x in v): k for k, v in NODE_TYPE_ENCODING.items()}
    ONE_HOT_TO_EDGE_TYPE_STR = {tuple(float(x) for x in v): k for k, v in EDGE_LABEL_ENCODING.items()}


else: # Fallbacks if aig_config is not loaded
    NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
    EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]
    VALID_AIG_NODE_TYPES = set(NODE_TYPE_KEYS)
    VALID_AIG_EDGE_TYPES = set(EDGE_TYPE_KEYS)
    NODE_CONST0 = "NODE_CONST0"; NODE_PI = "NODE_PI"; NODE_AND = "NODE_AND"; NODE_PO = "NODE_PO"
    MIN_AND_COUNT_CONFIG = 1; MIN_PO_COUNT_CONFIG = 1
    ONE_HOT_TO_NODE_TYPE_STR = {
        (1.0, 0.0, 0.0, 0.0): "NODE_CONST0", (0.0, 1.0, 0.0, 0.0): "NODE_PI",
        (0.0, 0.0, 1.0, 0.0): "NODE_AND", (0.0, 0.0, 0.0, 1.0): "NODE_PO"
    }
    ONE_HOT_TO_EDGE_TYPE_STR = { (1.0, 0.0): "EDGE_REG", (0.0, 1.0): "EDGE_INV" }

# Helper to get type string from node attributes (list/array)
def get_node_type_from_attrs(node_attrs: dict) -> str:
    raw_type = node_attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    try:
        type_tuple = tuple(float(x) for x in raw_type)
        return ONE_HOT_TO_NODE_TYPE_STR.get(type_tuple, "UNKNOWN_ENCODING")
    except Exception: return "UNKNOWN_CONVERSION_ERROR"

# Helper to get type string from edge attributes (list/array)
def get_edge_type_from_attrs(edge_attrs: dict) -> str:
    raw_type = edge_attrs.get('type')
    if raw_type is None: return "UNKNOWN_MISSING_ATTR"
    try:
        type_tuple = tuple(float(x) for x in raw_type)
        return ONE_HOT_TO_EDGE_TYPE_STR.get(type_tuple, "UNKNOWN_ENCODING")
    except Exception: return "UNKNOWN_CONVERSION_ERROR"


# --- Existing Functions (calculate_structural_aig_metrics, count_pi_po_paths, validate_aig_structures) ---
def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """Calculates structural AIG validity metrics based on assigned types."""
    # (This function remains largely the same, but ensure it uses the globally defined
    # NODE_TYPE constants and VALID_AIG_NODE_TYPES/VALID_AIG_EDGE_TYPES correctly)
    metrics = defaultdict(float) # Use float for easier averaging later
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['constraints_failed'] = [] # Store reasons for failure

    if not isinstance(G, nx.DiGraph) or metrics['num_nodes'] == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object")
        metrics['is_structurally_valid'] = False
        return dict(metrics)

    # 1. Check DAG
    try:
        metrics['is_dag'] = float(nx.is_directed_acyclic_graph(G))
        if not metrics['is_dag']: metrics['constraints_failed'].append("Not a DAG")
    except Exception as e:
         logger.warning(f"DAG check failed: {e}")
         metrics['is_dag'] = 0.0; metrics['constraints_failed'].append("DAG Check Error")

    # 2. Check Node Types and Degrees
    node_type_counts = Counter()
    for node, data in G.nodes(data=True):
        node_type = get_node_type_from_attrs(data)
        node_type_counts[node_type] += 1

        if node_type not in VALID_AIG_NODE_TYPES:
            metrics['num_unknown_nodes'] += 1; continue

        try: in_deg = G.in_degree(node); out_deg = G.out_degree(node)
        except Exception: metrics['degree_check_errors'] += 1; continue # Count errors

        if node_type == NODE_CONST0:
            if in_deg != 0: metrics['const0_indegree_violations'] += 1
        elif node_type == NODE_PI:
            if in_deg != 0: metrics['pi_indegree_violations'] += 1
        elif node_type == NODE_AND:
            if in_deg != 2: metrics['and_indegree_violations'] += 1
        elif node_type == NODE_PO:
            if out_deg != 0: metrics['po_outdegree_violations'] += 1
            if in_deg == 0: metrics['po_indegree_violations'] += 1 # PO needs input

    # Add node counts to metrics
    metrics['num_pi'] = float(node_type_counts.get(NODE_PI, 0))
    metrics['num_po'] = float(node_type_counts.get(NODE_PO, 0))
    metrics['num_and'] = float(node_type_counts.get(NODE_AND, 0))
    metrics['num_const0'] = float(node_type_counts.get(NODE_CONST0, 0))
    metrics['num_unknown_nodes'] = float(sum(v for k, v in node_type_counts.items() if k not in VALID_AIG_NODE_TYPES))

    # Add failure reasons based on type/degree checks
    if metrics['num_unknown_nodes'] > 0: metrics['constraints_failed'].append("Unknown node types")
    if metrics['const0_indegree_violations'] > 0: metrics['constraints_failed'].append("CONST0 in-degree != 0")
    if metrics['pi_indegree_violations'] > 0: metrics['constraints_failed'].append("PI in-degree != 0")
    if metrics['and_indegree_violations'] > 0: metrics['constraints_failed'].append("AND in-degree != 2")
    if metrics['po_outdegree_violations'] > 0: metrics['constraints_failed'].append("PO out-degree != 0")
    if metrics['po_indegree_violations'] > 0: metrics['constraints_failed'].append("PO in-degree == 0")

    # 3. Check Edge Types
    for u, v, data in G.edges(data=True):
        edge_type = get_edge_type_from_attrs(data)
        if edge_type not in VALID_AIG_EDGE_TYPES:
            metrics['num_unknown_edges'] += 1
    if metrics['num_unknown_edges'] > 0:
        metrics['constraints_failed'].append("Unknown edge types")

    # 4. Check Basic AIG Requirements
    if metrics['num_pi'] == 0 and metrics['num_const0'] == 0 :
         metrics['constraints_failed'].append("No PIs or Const0")
    if metrics['num_and'] < MIN_AND_COUNT_CONFIG :
        metrics['constraints_failed'].append(f"AND gates < {MIN_AND_COUNT_CONFIG}")
    if metrics['num_po'] < MIN_PO_COUNT_CONFIG:
        metrics['constraints_failed'].append(f"POs < {MIN_PO_COUNT_CONFIG}")

    # 5. Check isolated nodes (excluding CONST0)
    try:
        all_isolates = list(nx.isolates(G))
        relevant_isolates = [n for n in all_isolates if get_node_type_from_attrs(G.nodes[n]) != NODE_CONST0]
        metrics['isolated_nodes'] = float(len(relevant_isolates))
        # Do not add isolated nodes to constraints_failed for validity, but report it
    except Exception as e:
         logger.warning(f"Isolate check failed: {e}")
         metrics['isolated_nodes'] = -1.0 # Indicate error

    # --- Final Validity Check ---
    is_valid = (
        metrics['is_dag'] > 0.5 and # Check if DAG check passed
        metrics['num_unknown_nodes'] == 0 and
        metrics['num_unknown_edges'] == 0 and
        metrics['const0_indegree_violations'] == 0 and
        metrics['pi_indegree_violations'] == 0 and
        metrics['and_indegree_violations'] == 0 and
        metrics['po_outdegree_violations'] == 0 and
        metrics['po_indegree_violations'] == 0 and
        (metrics['num_pi'] > 0 or metrics['num_const0'] > 0) and
        metrics['num_and'] >= MIN_AND_COUNT_CONFIG and
        metrics['num_po'] >= MIN_PO_COUNT_CONFIG
    )
    metrics['is_structurally_valid'] = float(is_valid) # Store as float
    if not is_valid and not metrics['constraints_failed']:
         metrics['constraints_failed'].append("General Validity Check Failed")

    return dict(metrics) # Convert back to dict


def count_pi_po_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """Counts PIs reaching POs and POs reachable from PIs based on reachability."""
    # (This function remains the same, it operates on the NetworkX graph structure)
    results = defaultdict(float)
    results['error'] = None
    if G.number_of_nodes() == 0: return dict(results)

    try:
        pis = set(n for n, d in G.nodes(data=True) if get_node_type_from_attrs(d) == NODE_PI)
        pos = set(n for n, d in G.nodes(data=True) if get_node_type_from_attrs(d) == NODE_PO)
        const0_nodes = set(n for n, d in G.nodes(data=True) if get_node_type_from_attrs(d) == NODE_CONST0)
        source_nodes = pis.union(const0_nodes)

        results['num_pi'] = float(len(pis))
        results['num_po'] = float(len(pos))
        results['num_const0'] = float(len(const0_nodes))

        if not source_nodes or not pos: return dict(results)

        connected_sources = set()
        connected_pos = set()

        for source_node in source_nodes:
             if source_node not in G: continue
             for po_node in pos:
                  if po_node not in G: continue
                  try:
                      if nx.has_path(G, source_node, po_node):
                           connected_sources.add(source_node); break
                  except Exception as e: logger.warning(f"Path check {source_node}->{po_node} failed: {e}"); results['error'] = "Path check error"; break # Break inner loop on error

        for po_node in pos:
             if po_node not in G: continue
             for source_node in source_nodes:
                  if source_node not in G: continue
                  try:
                      if nx.has_path(G, source_node, po_node):
                           connected_pos.add(po_node); break
                  except Exception as e: logger.warning(f"Path check {source_node}->{po_node} failed: {e}"); results['error'] = "Path check error"; break # Break inner loop on error


        num_sources_total = len(source_nodes)
        results['num_pis_reaching_po'] = float(len(connected_sources))
        results['num_pos_reachable_from_pi'] = float(len(connected_pos))

        if num_sources_total > 0: results['fraction_pis_connected'] = results['num_pis_reaching_po'] / num_sources_total
        if results['num_po'] > 0: results['fraction_pos_connected'] = results['num_pos_reachable_from_pi'] / results['num_po']

    except Exception as e:
        logger.error(f"Error during count_pi_po_paths: {e}", exc_info=True)
        results['error'] = f"Processing error: {e}"

    return dict(results)


def validate_aig_structures(graphs: List[nx.DiGraph]) -> float:
    """Validates a list of NetworkX DiGraphs based on structural AIG rules."""
    # (This function remains the same)
    num_total = len(graphs)
    if num_total == 0: return 0.0
    num_valid_structurally = 0
    for i, graph in enumerate(graphs):
        if not isinstance(graph, nx.DiGraph): continue
        struct_metrics = calculate_structural_aig_metrics(graph)
        if struct_metrics.get('is_structurally_valid', 0.0) > 0.5:
            num_valid_structurally += 1
    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    logger.info(f"Validated {num_total} graphs. Structurally Valid: {num_valid_structurally} ({validity_fraction*100:.2f}%)")
    return validity_fraction

# --- Isomorphism Helper (Updated to use string types directly) ---
def node_match_aig(node1_attrs, node2_attrs):
    """Node matcher for isomorphism check based on 'type' string."""
    return node1_attrs.get('type') == node2_attrs.get('type')

def edge_match_aig(edge1_attrs, edge2_attrs):
    """Edge matcher for isomorphism check based on 'type' string."""
    return edge1_attrs.get('type') == edge2_attrs.get('type')


def are_graphs_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    """Checks isomorphism considering node/edge 'type' string attributes."""
    try:
        # Use the string-based matchers
        return nx.is_isomorphic(G1, G2, node_match=node_match_aig, edge_match=edge_match_aig)
    except Exception as e:
        logger.warning(f"Isomorphism check failed between two graphs: {e}")
        return False

# --- Uniqueness Calculation ---
def calculate_uniqueness(valid_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    """Calculates uniqueness among valid graphs."""
    # (This function remains the same, uses are_graphs_isomorphic)
    num_valid = len(valid_graphs)
    if num_valid <= 1: return (1.0, num_valid)
    unique_graph_indices = []
    logger.info(f"Calculating uniqueness for {num_valid} valid graphs...")
    for i in tqdm(range(num_valid), desc="Checking Uniqueness", leave=False):
        is_unique = True; G1 = valid_graphs[i]
        for unique_idx in unique_graph_indices:
            G2 = valid_graphs[unique_idx]
            if are_graphs_isomorphic(G1, G2): is_unique = False; break
        if is_unique: unique_graph_indices.append(i)
    num_unique = len(unique_graph_indices); uniqueness_score = num_unique / num_valid
    logger.info(f"Found {num_unique} unique graphs out of {num_valid} valid graphs.")
    return uniqueness_score, num_unique

# --- Novelty Calculation ---
def calculate_novelty(valid_graphs: List[nx.DiGraph], train_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    """Calculates novelty against a training set."""
    # (This function remains the same, uses are_graphs_isomorphic)
    num_valid = len(valid_graphs); num_train = len(train_graphs)
    if num_valid == 0: return (0.0, 0)
    if num_train == 0: logger.warning("Training set empty, novelty is 100%."); return (1.0, num_valid)
    num_novel = 0
    logger.info(f"Calculating novelty for {num_valid} valid graphs against {num_train} training graphs...")
    for gen_graph in tqdm(valid_graphs, desc="Checking Novelty", leave=False):
        is_novel = True
        for train_graph in train_graphs:
            if are_graphs_isomorphic(gen_graph, train_graph): is_novel = False; break
        if is_novel: num_novel += 1
    novelty_score = num_novel / num_valid
    logger.info(f"Found {num_novel} novel graphs out of {num_valid} valid graphs.")
    return novelty_score, num_novel


# --- NEW: Training Graph Loader for PKL Files ---
def load_training_graphs_from_pkl(train_pkl_dir: str, train_pkl_prefix: str, num_files: int) -> Optional[List[nx.DiGraph]]:
    """
    Loads training graphs from the first N PKL files matching a prefix in a directory.

    Args:
        train_pkl_dir (str): Directory containing the training PKL files.
        train_pkl_prefix (str): Prefix of the training PKL files (e.g., "real_aigs_part_").
        num_files (int): The number of PKL files to load (e.g., 4).

    Returns:
        A list of NetworkX DiGraphs, or None if loading fails.
    """
    if not os.path.isdir(train_pkl_dir):
        logger.error(f"Training PKL directory not found: {train_pkl_dir}")
        return None

    try:
        # Find files matching the prefix and sort them (to ensure consistent loading order)
        matching_files = sorted([
            f for f in os.listdir(train_pkl_dir)
            if f.startswith(train_pkl_prefix) and f.endswith(".pkl")
        ])
    except OSError as e:
        logger.error(f"Error listing files in directory {train_pkl_dir}: {e}")
        return None

    if not matching_files:
        logger.error(f"No PKL files found in {train_pkl_dir} with prefix '{train_pkl_prefix}'.")
        return None

    # Select the first 'num_files' files
    files_to_load = matching_files[:num_files]
    if len(files_to_load) < num_files:
        logger.warning(f"Found only {len(files_to_load)} files matching prefix, expected {num_files}.")
    if not files_to_load:
        logger.error("No training PKL files selected to load.")
        return None

    logger.info(f"Loading training graphs from {len(files_to_load)} PKL files in {train_pkl_dir}:")
    for fname in files_to_load: logger.info(f" - {fname}")

    all_train_graphs = []
    for filename in tqdm(files_to_load, desc="Loading Training PKLs"):
        file_path = os.path.join(train_pkl_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                graphs_chunk = pickle.load(f)
            if isinstance(graphs_chunk, list):
                # Filter out any non-DiGraph items just in case
                valid_graphs_in_chunk = [g for g in graphs_chunk if isinstance(g, nx.DiGraph)]
                all_train_graphs.extend(valid_graphs_in_chunk)
                if len(valid_graphs_in_chunk) != len(graphs_chunk):
                    logger.warning(f"File {filename} contained non-DiGraph items.")
            else:
                logger.warning(f"File {filename} did not contain a list. Skipping.")
        except Exception as e:
            logger.error(f"Error loading or processing PKL file {filename}: {e}")
            # Optionally decide whether to continue or return None based on severity
            # return None # Stop if any file fails

    logger.info(f"Finished loading training graphs. Total graphs loaded: {len(all_train_graphs)}")
    return all_train_graphs


# --- Main Evaluation Logic ---
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

    # --- Load Training Data (Using new PKL method) ---
    train_graphs = None
    if args.train_pkl_dir and args.train_pkl_prefix:
        train_graphs = load_training_graphs_from_pkl(
            args.train_pkl_dir,
            args.train_pkl_prefix,
            args.num_train_pkl_files
        )
        if train_graphs is None:
             logger.warning(f"Could not load training graphs from PKL files in {args.train_pkl_dir}. Novelty will not be calculated.")
        elif not train_graphs:
             logger.warning(f"Loaded 0 training graphs from PKL files in {args.train_pkl_dir}. Novelty will be 100%.")

    elif args.train_pkl_dir or args.train_pkl_prefix:
         logger.warning("Both --train_pkl_dir and --train_pkl_prefix must be provided to load training graphs for novelty.")
    # --- End Load Training Data ---

    num_total = len(generated_graphs)
    valid_graphs = []
    aggregate_metrics = defaultdict(list)
    aggregate_path_metrics = defaultdict(list)
    failed_constraints_summary = Counter()

    logger.info("Starting evaluation (Pass 1: Validity and Metrics)...")
    for i, graph in enumerate(tqdm(generated_graphs, desc="Evaluating Validity")):
        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph, skipping.")
            failed_constraints_summary["Invalid Graph Object"] += 1; continue

        struct_metrics = calculate_structural_aig_metrics(graph)
        for key, value in struct_metrics.items():
             if isinstance(value, (int, float, bool)): aggregate_metrics[key].append(float(value))
        if struct_metrics.get('is_structurally_valid', 0.0) > 0.5:
            valid_graphs.append(graph)
            try:
                 path_metrics = count_pi_po_paths(graph)
                 if path_metrics.get('error') is None:
                    for key, value in path_metrics.items():
                        if isinstance(value, (int, float)): aggregate_path_metrics[key].append(value)
                 else: logger.warning(f"Skipping path metrics for valid graph {i} due to error: {path_metrics['error']}")
            except Exception as e: logger.error(f"Error calculating path metrics for valid graph {i}: {e}")
        else:
            for reason in struct_metrics.get('constraints_failed', ["Unknown Failure"]):
                failed_constraints_summary[reason] += 1
    logger.info("Evaluation (Pass 1) finished.")

    num_valid_structurally = len(valid_graphs)
    uniqueness_score, num_unique = calculate_uniqueness(valid_graphs)
    novelty_score, num_novel = (-1.0, -1) # Default if not calculated
    if train_graphs is not None: # Check if training graphs were loaded
        novelty_score, num_novel = calculate_novelty(valid_graphs, train_graphs)

    # --- Reporting (remains the same) ---
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
             print(f"Novelty (N) among valid       : Not calculated (training set PKL files not provided/loaded)")
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


# --- Main Execution Block (Updated Arguments) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate generated AIGs for Validity, Uniqueness, and Novelty.')
    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs.')
    # Arguments for loading training graphs from PKL files
    parser.add_argument('--train_pkl_dir', type=str, default="./data/aigs/",
                        help='(Optional) Path to the directory containing training PKL files (e.g., ./raw_aigs_pkl/) for Novelty calculation.')
    parser.add_argument('--train_pkl_prefix', type=str, default="real_aigs_part_",
                        help='(Optional) Prefix of the training PKL files (used with --train_pkl_dir).')
    parser.add_argument('--num_train_pkl_files', type=int, default=4,
                        help='(Optional) Number of training PKL files to load (used with --train_pkl_dir).')

    parsed_args = parser.parse_args()
    run_standalone_evaluation(parsed_args)