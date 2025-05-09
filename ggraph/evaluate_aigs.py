# evaluate_aigs.py
import pickle
import networkx as nx
import argparse
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np  # For checking np.ndarray type
import sys
import io
import warnings  # For warning about unknown type formats
from tqdm import tqdm

# Import necessary items from aig_config
from aig_config import (
    NODE_TYPE_KEYS, EDGE_TYPE_KEYS, MIN_AND_COUNT, MIN_PO_COUNT,
    DECODING_NODE_TYPE_NX, DECODING_EDGE_TYPE_NX  # For decoding one-hot vectors
)

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluate_aigs_pkl")

# Define node type constants directly from NODE_TYPE_KEYS for clarity
NODE_CONST0 = NODE_TYPE_KEYS[0]
NODE_PI = NODE_TYPE_KEYS[1]
NODE_AND = NODE_TYPE_KEYS[2]
NODE_PO = NODE_TYPE_KEYS[3]


# --- Helper functions to get type string from node/edge attributes ---
def get_node_type_str_from_attrs(attrs: dict) -> str:
    """
    Retrieves the node 'type' attribute as a string.
    Handles cases where 'type' is already a string or a one-hot vector (list/ndarray).
    """
    raw_type = attrs.get('type')
    if raw_type is None:
        return "UNKNOWN_MISSING_ATTR"

    if isinstance(raw_type, str):
        return raw_type

    if isinstance(raw_type, (list, np.ndarray)):
        try:
            # Convert to tuple of floats, as DECODING_NODE_TYPE_NX uses float tuples as keys
            type_tuple = tuple(float(x) for x in raw_type)
            return DECODING_NODE_TYPE_NX.get(type_tuple, "UNKNOWN_ENCODING")
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"Could not convert node type vector {raw_type} to tuple of floats: {e}. Treating as UNKNOWN.")
            return "UNKNOWN_VECTOR_CONVERSION_ERROR"

    warnings.warn(
        f"Node type attribute has unexpected format: {raw_type} (type: {type(raw_type)}). Treating as UNKNOWN.")
    return f"UNKNOWN_TYPE_FORMAT_{type(raw_type).__name__}"


def get_edge_type_str_from_attrs(attrs: dict) -> str:
    """
    Retrieves the edge 'type' attribute as a string.
    Handles cases where 'type' is already a string or a one-hot vector (list/ndarray).
    """
    raw_type = attrs.get('type')
    if raw_type is None:
        return "UNKNOWN_MISSING_ATTR"

    if isinstance(raw_type, str):
        return raw_type

    if isinstance(raw_type, (list, np.ndarray)):
        try:
            # Convert to tuple of floats, as DECODING_EDGE_TYPE_NX uses float tuples as keys
            type_tuple = tuple(float(x) for x in raw_type)
            return DECODING_EDGE_TYPE_NX.get(type_tuple, "UNKNOWN_ENCODING")
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"Could not convert edge type vector {raw_type} to tuple of floats: {e}. Treating as UNKNOWN.")
            return "UNKNOWN_VECTOR_CONVERSION_ERROR"

    warnings.warn(
        f"Edge type attribute has unexpected format: {raw_type} (type: {type(raw_type)}). Treating as UNKNOWN.")
    return f"UNKNOWN_TYPE_FORMAT_{type(raw_type).__name__}"


# --- Structural Metrics Calculation (Updated to use new type helpers) ---
def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """Calculates structural AIG validity metrics based on assigned types (string or vector)."""
    metrics = defaultdict(float)
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['constraints_failed'] = []

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
        metrics['is_dag'] = 0.0;
        metrics['constraints_failed'].append("DAG Check Error")

    # 2. Check Node Types and Degrees
    node_type_counts = Counter()
    for node, data in G.nodes(data=True):
        node_type_str = get_node_type_str_from_attrs(data)  # Use new helper
        node_type_counts[node_type_str] += 1

        if node_type_str not in NODE_TYPE_KEYS:
            metrics['num_unknown_nodes'] += 1
            if "Unknown node types" not in metrics['constraints_failed']:
                metrics['constraints_failed'].append("Unknown node types")
            continue

        try:
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
        except Exception:
            metrics['degree_check_errors'] += 1
            if "Degree Check Error" not in metrics['constraints_failed']:
                metrics['constraints_failed'].append("Degree Check Error")
            continue

        if node_type_str == NODE_CONST0:
            if in_deg != 0: metrics['const0_indegree_violations'] += 1
        elif node_type_str == NODE_PI:
            if in_deg != 0: metrics['pi_indegree_violations'] += 1
        elif node_type_str == NODE_AND:
            if in_deg != 2: metrics['and_indegree_violations'] += 1
        elif node_type_str == NODE_PO:
            if out_deg != 0: metrics['po_outdegree_violations'] += 1
            if in_deg == 0: metrics['po_indegree_violations'] += 1

    metrics['num_pi'] = float(node_type_counts.get(NODE_PI, 0))
    metrics['num_po'] = float(node_type_counts.get(NODE_PO, 0))
    metrics['num_and'] = float(node_type_counts.get(NODE_AND, 0))
    metrics['num_const0'] = float(node_type_counts.get(NODE_CONST0, 0))

    if metrics['const0_indegree_violations'] > 0 and "CONST0 in-degree != 0" not in metrics['constraints_failed']:
        metrics['constraints_failed'].append("CONST0 in-degree != 0")
    if metrics['pi_indegree_violations'] > 0 and "PI in-degree != 0" not in metrics['constraints_failed']:
        metrics['constraints_failed'].append("PI in-degree != 0")
    if metrics['and_indegree_violations'] > 0 and "AND in-degree != 2" not in metrics['constraints_failed']:
        metrics['constraints_failed'].append("AND in-degree != 2")
    if metrics['po_outdegree_violations'] > 0 and "PO out-degree != 0" not in metrics['constraints_failed']:
        metrics['constraints_failed'].append("PO out-degree != 0")
    if metrics['po_indegree_violations'] > 0 and "PO in-degree == 0" not in metrics['constraints_failed']:
        metrics['constraints_failed'].append("PO in-degree == 0")

    # 3. Check Edge Types
    for u, v, data in G.edges(data=True):
        edge_type_str = get_edge_type_str_from_attrs(data)  # Use new helper
        if edge_type_str not in EDGE_TYPE_KEYS:
            metrics['num_unknown_edges'] += 1
            if "Unknown edge types" not in metrics['constraints_failed']:
                metrics['constraints_failed'].append("Unknown edge types")

    # 4. Check Basic AIG Requirements
    if metrics['num_pi'] == 0 and metrics['num_const0'] == 0:
        if "No PIs or Const0" not in metrics['constraints_failed']:
            metrics['constraints_failed'].append("No PIs or Const0")
    if metrics['num_and'] < MIN_AND_COUNT:
        if f"AND gates < {MIN_AND_COUNT}" not in metrics['constraints_failed']:
            metrics['constraints_failed'].append(f"AND gates < {MIN_AND_COUNT}")
    if metrics['num_po'] < MIN_PO_COUNT:
        if f"POs < {MIN_PO_COUNT}" not in metrics['constraints_failed']:
            metrics['constraints_failed'].append(f"POs < {MIN_PO_COUNT}")

    # 5. Check isolated nodes (excluding CONST0)
    try:
        all_isolates = list(nx.isolates(G))
        # Use the helper for checking type of isolated nodes as well
        relevant_isolates = [n for n in all_isolates if get_node_type_str_from_attrs(G.nodes[n]) != NODE_CONST0]
        metrics['isolated_nodes'] = float(len(relevant_isolates))
    except Exception as e:
        logger.warning(f"Isolate check failed: {e}")
        metrics['isolated_nodes'] = -1.0

    is_valid = (
            metrics['is_dag'] > 0.5 and
            metrics['num_unknown_nodes'] == 0 and
            metrics['num_unknown_edges'] == 0 and
            metrics['const0_indegree_violations'] == 0 and
            metrics['pi_indegree_violations'] == 0 and
            metrics['and_indegree_violations'] == 0 and
            metrics['po_outdegree_violations'] == 0 and
            metrics['po_indegree_violations'] == 0 and
            (metrics['num_pi'] > 0 or metrics['num_const0'] > 0) and
            metrics['num_and'] >= MIN_AND_COUNT and
            metrics['num_po'] >= MIN_PO_COUNT
    )
    metrics['is_structurally_valid'] = float(is_valid)
    if not is_valid and not metrics['constraints_failed']:
        metrics['constraints_failed'].append("General Validity Check Failed (one or more rules violated)")

    return dict(metrics)


def count_pi_po_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """Counts PIs reaching POs and POs reachable from PIs."""
    results = defaultdict(float)
    results['error'] = None
    if G.number_of_nodes() == 0: return dict(results)

    try:
        pis = set(n for n, d in G.nodes(data=True) if get_node_type_str_from_attrs(d) == NODE_PI)
        pos = set(n for n, d in G.nodes(data=True) if get_node_type_str_from_attrs(d) == NODE_PO)
        const0_nodes = set(n for n, d in G.nodes(data=True) if get_node_type_str_from_attrs(d) == NODE_CONST0)
        source_nodes = pis.union(const0_nodes)

        results['num_pi'] = float(len(pis))
        results['num_po'] = float(len(pos))
        results['num_const0'] = float(len(const0_nodes))

        if not source_nodes or not pos:
            results['num_pis_reaching_po'] = 0.0
            results['num_pos_reachable_from_pi'] = 0.0
            results['fraction_pis_connected'] = 0.0
            results['fraction_pos_connected'] = 0.0
            return dict(results)

        connected_sources = set()
        connected_pos = set()

        for source_node in source_nodes:
            if source_node not in G: continue
            for po_node in pos:
                if po_node not in G: continue
                try:
                    if nx.has_path(G, source_node, po_node):
                        connected_sources.add(source_node)
                        break
                except Exception as e:
                    logger.warning(f"Path check {source_node}->{po_node} failed: {e}")
                    results['error'] = "Path check error";
                    break
            if results['error']: break

        if not results['error']:
            for po_node in pos:
                if po_node not in G: continue
                for source_node in source_nodes:
                    if source_node not in G: continue
                    try:
                        if nx.has_path(G, source_node, po_node):
                            connected_pos.add(po_node)
                            break
                    except Exception as e:
                        logger.warning(f"Path check {source_node}->{po_node} failed: {e}")
                        results['error'] = "Path check error";
                        break
                if results['error']: break

        num_sources_total = len(source_nodes)
        results['num_pis_reaching_po'] = float(len(connected_sources))
        results['num_pos_reachable_from_pi'] = float(len(connected_pos))

        results['fraction_pis_connected'] = results[
                                                'num_pis_reaching_po'] / num_sources_total if num_sources_total > 0 else 0.0
        results['fraction_pos_connected'] = results['num_pos_reachable_from_pi'] / results['num_po'] if results[
                                                                                                            'num_po'] > 0 else 0.0

    except Exception as e:
        logger.error(f"Error during count_pi_po_paths: {e}", exc_info=True)
        results['error'] = f"Processing error: {e}"
    return dict(results)


def validate_aig_structures(graphs: List[nx.DiGraph]) -> float:
    """Validates a list of NetworkX DiGraphs."""
    num_total = len(graphs)
    if num_total == 0: return 0.0
    num_valid_structurally = 0
    for i, graph in enumerate(graphs):
        if not isinstance(graph, nx.DiGraph): continue
        struct_metrics = calculate_structural_aig_metrics(graph)
        if struct_metrics.get('is_structurally_valid', 0.0) > 0.5:
            num_valid_structurally += 1
    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    logger.info(
        f"Validated {num_total} graphs. Structurally Valid: {num_valid_structurally} ({validity_fraction * 100:.2f}%)")
    return validity_fraction


# --- Isomorphism Helper (Updated to use flexible type helpers) ---
def node_match_aig_flexible(node1_attrs, node2_attrs):
    """Node matcher for isomorphism, handles string or vector types."""
    type1_str = get_node_type_str_from_attrs(node1_attrs)
    type2_str = get_node_type_str_from_attrs(node2_attrs)
    # Ensure we don't match if one type is unknown and the other is known
    if "UNKNOWN" in type1_str and "UNKNOWN" not in type2_str: return False
    if "UNKNOWN" in type2_str and "UNKNOWN" not in type1_str: return False
    return type1_str == type2_str


def edge_match_aig_flexible(edge1_attrs, edge2_attrs):
    """Edge matcher for isomorphism, handles string or vector types."""
    type1_str = get_edge_type_str_from_attrs(edge1_attrs)
    type2_str = get_edge_type_str_from_attrs(edge2_attrs)
    if "UNKNOWN" in type1_str and "UNKNOWN" not in type2_str: return False
    if "UNKNOWN" in type2_str and "UNKNOWN" not in type1_str: return False
    return type1_str == type2_str


def are_graphs_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    """Checks isomorphism considering node/edge types (string or vector)."""
    if G1 is None or G2 is None: return False
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False
    try:
        # Use the flexible matchers
        return nx.is_isomorphic(G1, G2, node_match=node_match_aig_flexible, edge_match=edge_match_aig_flexible)
    except Exception as e:
        logger.warning(f"Isomorphism check failed between two graphs: {e}")
        return False


# --- Uniqueness Calculation (no change needed here, uses are_graphs_isomorphic) ---
def calculate_uniqueness(valid_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs)
    if num_valid <= 1: return (1.0, num_valid)
    unique_graph_representatives = []
    logger.info(f"Calculating uniqueness for {num_valid} valid graphs...")
    for i in tqdm(range(num_valid), desc="Checking Uniqueness", leave=False):
        current_graph = valid_graphs[i]
        if current_graph is None: continue
        is_unique_to_list = True
        for representative_graph in unique_graph_representatives:
            if are_graphs_isomorphic(current_graph, representative_graph):
                is_unique_to_list = False;
                break
        if is_unique_to_list:
            unique_graph_representatives.append(current_graph)
    num_unique = len(unique_graph_representatives)
    uniqueness_score = num_unique / num_valid if num_valid > 0 else 0.0
    logger.info(f"Found {num_unique} unique graphs out of {num_valid} valid graphs.")
    return uniqueness_score, num_unique


# --- Novelty Calculation (no change needed here, uses are_graphs_isomorphic) ---
def calculate_novelty(valid_graphs: List[nx.DiGraph], train_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    num_valid = len(valid_graphs)
    if num_valid == 0: return (0.0, 0)
    if not train_graphs:
        logger.warning("Training set empty or not loaded, novelty is 100% by definition.")
        return (1.0, num_valid) if num_valid > 0 else (0.0, 0)
    num_train = len(train_graphs)
    num_novel = 0
    logger.info(f"Calculating novelty for {num_valid} valid graphs against {num_train} training graphs...")
    for gen_graph in tqdm(valid_graphs, desc="Checking Novelty", leave=False):
        if gen_graph is None: continue
        is_novel_to_train_set = True
        for train_graph in train_graphs:
            if train_graph is None: continue
            if are_graphs_isomorphic(gen_graph, train_graph):
                is_novel_to_train_set = False;
                break
        if is_novel_to_train_set:
            num_novel += 1
    novelty_score = num_novel / num_valid if num_valid > 0 else 0.0
    logger.info(f"Found {num_novel} novel graphs out of {num_valid} valid graphs.")
    return novelty_score, num_novel


# --- Training Graph Loader for PKL Files (no change needed here) ---
def load_training_graphs_from_pkl(train_pkl_dir: str, train_pkl_prefix: str, num_files_to_load: int) -> Optional[
    List[nx.DiGraph]]:
    if not os.path.isdir(train_pkl_dir):
        logger.error(f"Training PKL directory not found: {train_pkl_dir}");
        return None
    try:
        all_pkl_files = sorted(
            [f for f in os.listdir(train_pkl_dir) if f.startswith(train_pkl_prefix) and f.endswith(".pkl")])
    except OSError as e:
        logger.error(f"Error listing files in directory {train_pkl_dir}: {e}");
        return None
    if not all_pkl_files:
        logger.error(f"No PKL files found in {train_pkl_dir} with prefix '{train_pkl_prefix}'.");
        return None
    files_to_process = all_pkl_files[:num_files_to_load]
    if len(files_to_process) < num_files_to_load:
        logger.warning(f"Found only {len(files_to_process)} files matching prefix, requested {num_files_to_load}.")
    if not files_to_process:
        logger.error("No training PKL files selected to load.");
        return None
    logger.info(f"Loading training graphs from {len(files_to_process)} PKL files in {train_pkl_dir}:")
    for fname in files_to_process: logger.info(f" - {fname}")
    all_train_graphs = []
    for filename in tqdm(files_to_process, desc="Loading Training PKLs"):
        file_path = os.path.join(train_pkl_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                graphs_chunk = pickle.load(f)
            if isinstance(graphs_chunk, list):
                valid_graphs_in_chunk = [g for g in graphs_chunk if isinstance(g, nx.DiGraph)]
                all_train_graphs.extend(valid_graphs_in_chunk)
                if len(valid_graphs_in_chunk) != len(graphs_chunk):
                    logger.warning(f"File {filename} contained non-DiGraph items that were skipped.")
            else:
                logger.warning(f"File {filename} did not contain a list. Skipping.")
        except Exception as e:
            logger.error(f"Error loading or processing PKL file {filename}: {e}")
    logger.info(f"Finished loading training graphs. Total graphs loaded: {len(all_train_graphs)}")
    return all_train_graphs


# --- Main Evaluation Logic (no change needed in the main loop itself) ---
def run_standalone_evaluation(input_source, results_filename="evaluation_results.txt",
                              train_pkl_dir=None, train_pkl_prefix=None, num_train_pkl_files=0):
    """Runs the evaluation, prints results to console, AND saves them to a file."""
    generated_graphs = []
    if isinstance(input_source, str):  # Assume it's a file path
        try:
            with open(input_source, 'rb') as f:
                generated_graphs = pickle.load(f)
            if not isinstance(generated_graphs, list):
                logger.error(f"Content of {input_source} is not a list. Evaluation cannot proceed.")
                generated_graphs = []  # Ensure it's a list to prevent downstream errors
        except FileNotFoundError:
            err_msg = f"Error: Input file {input_source} not found.\n"
            logger.error(err_msg.strip())
            with open(results_filename, 'w') as res_file:
                res_file.write(err_msg)
            print(err_msg, file=sys.stderr);
            return
        except Exception as e:
            err_msg = f"Error: Could not load graphs from {input_source}. Reason: {e}\n"
            logger.error(err_msg.strip())
            with open(results_filename, 'w') as res_file:
                res_file.write(err_msg)
            print(err_msg, file=sys.stderr);
            return
    elif isinstance(input_source, list):
        generated_graphs = input_source
    else:
        err_msg = "Error: input_source must be a file path (str) or a list of graphs.\n"
        logger.error(err_msg.strip())
        with open(results_filename, 'w') as res_file:
            res_file.write(err_msg)
        print(err_msg, file=sys.stderr);
        return

    if not generated_graphs:
        warn_msg = "No generated graphs provided or loaded to evaluate.\n"
        logger.warning(warn_msg.strip())
        with open(results_filename, 'w') as res_file: res_file.write(warn_msg)
        print(warn_msg);
        return

    train_graphs = None
    if train_pkl_dir and train_pkl_prefix and num_train_pkl_files > 0:
        train_graphs = load_training_graphs_from_pkl(train_pkl_dir, train_pkl_prefix, num_train_pkl_files)
        if train_graphs is None:
            logger.warning(f"Could not load training graphs from {train_pkl_dir}. Novelty will not be calculated.")
    else:
        logger.info("Training data not specified or num_files is 0. Novelty will not be calculated.")

    num_total = 0
    valid_graphs = []
    aggregate_metrics = defaultdict(list)
    aggregate_path_metrics = defaultdict(list)
    failed_constraints_summary = Counter()
    graphs_actually_processed = 0

    logger.info("Starting evaluation (Pass 1: Validity and Metrics)...")
    for i, graph_candidate in enumerate(tqdm(generated_graphs, desc="Evaluating Validity")):
        graphs_actually_processed += 1
        if not isinstance(graph_candidate, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph (type: {type(graph_candidate)}), skipping.")
            failed_constraints_summary["Invalid Graph Object Type"] += 1
            aggregate_metrics['is_structurally_valid'].append(0.0)  # Count as invalid
            num_total += 1  # Still count towards total loaded/attempted
            continue

        num_total += 1  # It's a graph, count it.
        struct_metrics = calculate_structural_aig_metrics(graph_candidate)
        for key, value in struct_metrics.items():
            if isinstance(value, (int, float, bool)): aggregate_metrics[key].append(float(value))

        if struct_metrics.get('is_structurally_valid', 0.0) > 0.5:
            valid_graphs.append(graph_candidate)
            try:
                path_metrics = count_pi_po_paths(graph_candidate)
                if path_metrics and path_metrics.get('error') is None:
                    for key, value in path_metrics.items():
                        if isinstance(value, (int, float)): aggregate_path_metrics[key].append(value)
                elif path_metrics:
                    logger.warning(f"Skipping path metrics for valid graph {i} due to error: {path_metrics['error']}")
            except Exception as e:
                logger.error(f"Error calculating path metrics for valid graph {i}: {e}", exc_info=True)
        else:
            reasons = struct_metrics.get('constraints_failed', ["Unknown Failure"])
            if not isinstance(reasons, list): reasons = [str(reasons)]
            for reason in reasons:
                failed_constraints_summary[reason] += 1
    logger.info(f"Evaluation (Pass 1) finished. Processed {graphs_actually_processed} items from input.")

    num_valid_structurally = len(valid_graphs)
    uniqueness_score, num_unique = calculate_uniqueness(valid_graphs)
    novelty_score, num_novel = (-1.0, -1)  # Default if not calculated
    if train_graphs is not None:  # Check if train_graphs were successfully loaded
        novelty_score, num_novel = calculate_novelty(valid_graphs, train_graphs)

    # --- Reporting ---
    original_stdout = sys.stdout
    string_buffer = io.StringIO()
    sys.stdout = string_buffer

    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    validity_percentage = validity_fraction * 100
    print("\n--- G2PT AIG V.U.N. Evaluation Summary ---")
    print(f"Total Graphs Attempted          : {num_total}")
    print(f"Structurally Valid AIGs (V)     : {num_valid_structurally} ({validity_percentage:.2f}%)")
    if num_valid_structurally > 0:
        print(f"Unique Valid AIGs             : {num_unique}")
        print(f"Uniqueness (U) among valid    : {uniqueness_score:.4f} ({uniqueness_score * 100:.2f}%)")
        if novelty_score != -1.0:
            print(f"Novel Valid AIGs vs Train Set : {num_novel}")
            print(f"Novelty (N) among valid       : {novelty_score:.4f} ({novelty_score * 100:.2f}%)")
        else:
            print(f"Novelty (N) among valid       : Not calculated")
    else:
        print(f"Uniqueness (U) among valid    : N/A (0 valid graphs)")
        print(f"Novelty (N) among valid       : N/A (0 valid graphs)")

    print("\n--- Average Structural Metrics (All Processed Graphs) ---")
    if aggregate_metrics and num_total > 0:  # Ensure num_total is not zero for averaging
        for key, values in sorted(aggregate_metrics.items()):
            if key == 'is_structurally_valid': continue
            if not values:
                print(f"  - Avg {key:<27}: N/A (no data)")
                continue
            avg_value = np.mean(values)
            std_value = np.std(values) if len(values) > 1 else 0
            if key == 'is_dag':
                print(f"  - Percentage {key:<22}: {avg_value * 100:.2f}%")
            else:
                print(f"  - Avg {key:<27}: {avg_value:.3f} (Std: {std_value:.3f})")
    else:
        print("  No structural metrics data collected or no graphs processed.")

    print("\n--- Constraint Violation Summary (Across Invalid Graphs) ---")
    num_actually_invalid_structurally = num_total - num_valid_structurally
    if not failed_constraints_summary and num_actually_invalid_structurally == 0:
        print("  No structural violations detected among processed graphs.")
    elif not failed_constraints_summary and num_actually_invalid_structurally > 0:
        print(
            f"  {num_actually_invalid_structurally} graphs were structurally invalid, but no specific reasons were logged by metrics.")
    else:
        sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        print(f"  (Violations summarized across {num_actually_invalid_structurally} structurally invalid graphs)")
        total_violation_instances = sum(failed_constraints_summary.values())
        print(f"  (Total violation instances logged: {total_violation_instances})")  # Sum of counts for each reason
        for reason, count in sorted_reasons:
            reason_percentage_of_invalid = (
                                                       count / num_actually_invalid_structurally) * 100 if num_actually_invalid_structurally > 0 else 0
            print(
                f"  - {reason:<45}: {count:<6} occurrences ({reason_percentage_of_invalid:.1f}% of invalid graphs had this)")

    print("\n--- Average Path Connectivity Metrics (Structurally Valid Graphs Only) ---")
    num_graphs_for_path_metrics = len(aggregate_path_metrics.get('num_pi', []))
    if num_graphs_for_path_metrics == 0:
        print("  No structurally valid graphs had path metrics calculated (or no valid graphs).")
    else:
        print(f"  (Based on {num_graphs_for_path_metrics} structurally valid graphs with path data)")
        for key, values in sorted(aggregate_path_metrics.items()):
            if key == 'error' or not values:
                if key != 'error': print(f"  - Avg {key:<27}: N/A (no data)")
                continue
            avg_value = np.mean(values)
            std_value = np.std(values) if len(values) > 1 else 0.0
            print(f"  - Avg {key:<27}: {avg_value:.3f} (Std: {std_value:.3f})")
    print("------------------------------------")

    results_output = string_buffer.getvalue()
    sys.stdout = original_stdout
    string_buffer.close()
    print(results_output)
    try:
        with open(results_filename, 'w') as res_file:
            res_file.write(results_output)
        logger.info(f"Evaluation results also saved to {results_filename}")
    except IOError as e:
        logger.error(f"Failed to write results to file {results_filename}: {e}")
        print(f"Error: Failed to write results to file {results_filename}: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate generated AIGs for Validity, Uniqueness, and Novelty.')
    parser.add_argument('input_source', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs, or the list itself.')
    parser.add_argument('--results_file', type=str, default="evaluation_results.txt",
                        help="Filename to save the evaluation summary.")
    parser.add_argument('--train_pkl_dir', type=str, default="./data/aigs",
                        help='(Optional) Path to the directory containing training PKL files.')
    parser.add_argument('--train_pkl_prefix', type=str, default="real_aigs_par_",
                        help='(Optional) Prefix of the training PKL files.')
    parser.add_argument('--num_train_pkl_files', type=int, default=4,
                        help='(Optional) Number of training PKL files to load.')

    parsed_args = parser.parse_args()
    run_standalone_evaluation(
        parsed_args.input_source,
        parsed_args.results_file,
        parsed_args.train_pkl_dir,
        parsed_args.train_pkl_prefix,
        parsed_args.num_train_pkl_files
    )
