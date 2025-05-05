# evaluate_aigs.py
import pickle
import networkx as nx
import argparse
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np # For calculating averages
import sys # Added for path adjustment

# --- Logger Setup ---
# Configure logger only once at the top level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluate_g2pt_aigs")


# --- Import the AIG configuration ---

from configs import aig as aig_config

# Derive valid node/edge types from config
VALID_AIG_NODE_TYPES = set(aig_config.NODE_TYPE_KEYS)
VALID_AIG_EDGE_TYPES = set(aig_config.EDGE_TYPE_KEYS)
# Extract specific node type keys for checks (optional, but clearer)
NODE_CONST0 = "NODE_CONST0" # Or directly use aig_config.NODE_TYPE_KEYS[0] if preferred
NODE_PI = "NODE_PI"
NODE_AND = "NODE_AND"
NODE_PO = "NODE_PO"
# Use counts from config if import was successful
MIN_AND_COUNT_CONFIG = aig_config.MIN_AND_COUNT
MIN_PO_COUNT_CONFIG = aig_config.MIN_PO_COUNT


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
    metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
    if not metrics['is_dag']:
        metrics['constraints_failed'].append("Not a DAG")

    # 2. Check Node Types and Basic Degrees
    node_type_counts = Counter()
    unknown_node_indices = []
    for node, data in G.nodes(data=True):
        node_type = data.get('type')
        node_type_counts[node_type] += 1

        # Use VALID_AIG_NODE_TYPES derived from config
        if node_type not in VALID_AIG_NODE_TYPES:
            metrics['num_unknown_nodes'] += 1
            unknown_node_indices.append(node)
            continue # Skip degree checks for unknown nodes

        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

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
            if in_deg == 0: metrics['po_indegree_violations'] += 1

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
        edge_type = data.get('type')
        # Use VALID_AIG_EDGE_TYPES derived from config
        if edge_type is not None and edge_type not in VALID_AIG_EDGE_TYPES:
            metrics['num_unknown_edges'] += 1
    if metrics['num_unknown_edges'] > 0:
        # Add failure reason to the list
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_edges']} unknown edge types")

    # 4. Check Basic AIG Requirements (Using config values)
    if metrics['num_pi'] == 0 and metrics['num_const0'] == 0 :
         metrics['constraints_failed'].append("No Primary Inputs or Const0 found")
    if metrics['num_and'] < MIN_AND_COUNT_CONFIG :
        metrics['constraints_failed'].append(f"Insufficient AND gates ({metrics['num_and']} < {MIN_AND_COUNT_CONFIG})")
    if metrics['num_po'] < MIN_PO_COUNT_CONFIG:
        metrics['constraints_failed'].append(f"Insufficient POs ({metrics['num_po']} < {MIN_PO_COUNT_CONFIG})")

    # --- 5. Check isolated nodes (Keep calculation, but don't add to constraints_failed) ---
    all_isolates = list(nx.isolates(G))
    relevant_isolates = []
    for node_idx in all_isolates:
        isolated_node_type = G.nodes[node_idx].get('type', None)
        if isolated_node_type != NODE_CONST0: # Only count non-CONST0 nodes
            relevant_isolates.append(node_idx)
    metrics['isolated_nodes'] = len(relevant_isolates) # Count relevant isolates for reporting


    # --- Final Validity Check ---
    # A graph is structurally valid IFF it passes ALL checks:
    # DAG, Known Node Types, Known Edge Types, Correct Degrees, Min PI/Const0, Min AND, Min PO
    is_valid = (
        metrics['is_dag'] and
        metrics['num_unknown_nodes'] == 0 and
        metrics['num_unknown_edges'] == 0 and # <-- Added this check
        metrics['const0_indegree_violations'] == 0 and
        metrics['pi_indegree_violations'] == 0 and
        metrics['and_indegree_violations'] == 0 and
        metrics['po_outdegree_violations'] == 0 and
        metrics['po_indegree_violations'] == 0 and
        (metrics['num_pi'] > 0 or metrics['num_const0'] > 0) and # At least one input source
        metrics['num_and'] >= MIN_AND_COUNT_CONFIG and
        metrics['num_po'] >= MIN_PO_COUNT_CONFIG
    )
    metrics['is_structurally_valid'] = is_valid
    # The constraints_failed list still contains all individual reasons if is_valid is False.
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
             node_type = data.get('type')
             if node_type == NODE_PI: pis.add(node)
             elif node_type == NODE_PO: pos.add(node)
             elif node_type == NODE_CONST0: const0_nodes.add(node)

        # Source nodes for path checking are PIs and Const0
        source_nodes = pis.union(const0_nodes)

        results['num_pi'] = len(pis) # Report actual PIs separately
        results['num_po'] = len(pos)
        results['num_const0'] = len(const0_nodes)

        if not source_nodes or not pos: # No paths possible if no sources or no POs
            return results

        connected_sources = set()
        connected_pos = set()

        # Find all nodes reachable from any source (PI or CONST0)
        all_reachable_from_sources = set()
        for source_node in source_nodes:
            try:
                if source_node in G:
                    desc = nx.descendants(G, source_node)
                    all_reachable_from_sources.update(desc)
                    all_reachable_from_sources.add(source_node)
                else: logger.warning(f"Source node {source_node} not found in graph during path check.")
            except Exception as e: logger.warning(f"Error finding descendants from source {source_node}: {e}")

        # Find all nodes that can reach any PO
        all_ancestors_of_pos = set()
        for po_node in pos:
            try:
                if po_node in G:
                    anc = nx.ancestors(G, po_node)
                    all_ancestors_of_pos.update(anc)
                    all_ancestors_of_pos.add(po_node)
                else: logger.warning(f"PO node {po_node} not found in graph during path check.")
            except Exception as e: logger.warning(f"Error finding ancestors for PO {po_node}: {e}")

        # Intersect the sets
        connected_sources = source_nodes.intersection(all_ancestors_of_pos)
        connected_pos = pos.intersection(all_reachable_from_sources)

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


# --- Function for Train Script ---
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
# --- END FUNCTION ---


# --- Main Evaluation Logic (for standalone script execution) ---
def run_standalone_evaluation(args):
    """Runs the evaluation when the script is executed directly."""
    logger.info(f"Loading generated AIGs from: {args.input_pickle_file}")
    try:
        with open(args.input_pickle_file, 'rb') as f: generated_graphs = pickle.load(f)
        if not isinstance(generated_graphs, list): logger.error("Pickle file does not contain a list."); return
        logger.info(f"Loaded {len(generated_graphs)} graphs.")
    except FileNotFoundError: logger.error(f"Input pickle file not found: {args.input_pickle_file}"); return
    except Exception as e: logger.error(f"Error loading pickle file: {e}"); return

    if not generated_graphs: logger.warning("No graphs found in the pickle file. Exiting."); return

    num_total = len(generated_graphs)
    num_valid_structurally = 0
    aggregate_metrics = defaultdict(list)
    aggregate_path_metrics = defaultdict(list)
    failed_constraints_summary = Counter()

    logger.info("Starting evaluation...")
    for i, graph in enumerate(generated_graphs):
        if i % 100 == 0 and i > 0: logger.info(f"Processed {i}/{num_total} graphs...")
        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph, skipping.")
            failed_constraints_summary["Invalid Graph Object"] += 1; continue

        struct_metrics = calculate_structural_aig_metrics(graph)
        if struct_metrics['is_structurally_valid']: num_valid_structurally += 1
        else:
            for reason in struct_metrics.get('constraints_failed', []): failed_constraints_summary[reason] += 1

        for key, value in struct_metrics.items():
            if isinstance(value, (int, float, bool)): aggregate_metrics[key].append(float(value))

        if struct_metrics['is_structurally_valid']:
             path_metrics = count_pi_po_paths(graph)
             if path_metrics.get('error') is None:
                  for key, value in path_metrics.items():
                     if isinstance(value, (int, float)): aggregate_path_metrics[key].append(value)
             else: logger.warning(f"Skipping path metrics for graph {i} due to error: {path_metrics['error']}")

    logger.info("Evaluation finished.")

    # --- Reporting ---
    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    validity_percentage = validity_fraction * 100

    print("\n--- G2PT AIG Evaluation Summary ---")
    print(f"Total Graphs Loaded             : {num_total}")
    print(f"Structurally Valid AIGs         : {num_valid_structurally} ({validity_percentage:.2f}%)")

    print("\n--- Average Structural Metrics (All Graphs) ---")
    for key, values in sorted(aggregate_metrics.items()):
        if key == 'is_structurally_valid': continue
        if not values: continue
        avg_value = np.mean(values)
        if key == 'is_dag': print(f"  - Avg {key:<27}: {avg_value*100:.2f}%")
        else: print(f"  - Avg {key:<27}: {avg_value:.3f}")

    print("\n--- Constraint Violation Summary (Across All Graphs) ---")
    total_graphs_with_violations = num_total - num_valid_structurally
    if not failed_constraints_summary: print("  No constraint violations detected.")
    else:
        sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        print(f"  (Violations counted across {num_total} graphs; {total_graphs_with_violations} graphs failed)")
        for reason, count in sorted_reasons:
            reason_percentage_of_graphs = (count / num_total) * 100 if num_total > 0 else 0
            print(f"  - {reason:<45}: {count:<6} occurrences ({reason_percentage_of_graphs:.1f}% of graphs)")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate structural validity and path connectivity of generated AIGs using config.')
    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs (e.g., generated_aigs.pkl).')

    parsed_args = parser.parse_args()
    run_standalone_evaluation(parsed_args)
