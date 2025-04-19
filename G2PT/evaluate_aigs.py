# evaluate_g2pt_aigs.py
import pickle
import networkx as nx
import argparse
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np # For calculating averages

# --- Constants ---
# Node types expected in the 'type' attribute from seq_to_nxgraph
VALID_AIG_NODE_TYPES = {'NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'}
# Edge types expected in the 'type' attribute from seq_to_nxgraph
VALID_AIG_EDGE_TYPES = {'EDGE_INV', 'EDGE_REG'}

# AIG constraint constants (from your original script)
MAX_PI_COUNT = 14
MIN_PI_COUNT = 2
MIN_AND_COUNT = 1
MIN_PO_COUNT = 1
MAX_PO_COUNT = 30

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluate_g2pt_aigs")

# --- Adapted Evaluation Functions ---

def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculates structural AIG validity metrics based on assigned types.
    Counts violations across the graph instead of breaking early.
    Returns a dictionary of detailed metrics and violation counts.
    """
    metrics = {
        'num_nodes': 0,
        'is_dag': False,
        'num_pi': 0, 'num_po': 0, 'num_and': 0, 'num_const0': 0,
        'num_unknown_nodes': 0,
        'num_unknown_edges': 0,
        'pi_indegree_violations': 0,      # PIs (not node 0) with in-degree != 0
        'const0_indegree_violations': 0,  # Node 0 (if exists) with in-degree != 0
        'and_indegree_violations': 0,     # ANDs with in-degree != 2
        'po_outdegree_violations': 0,     # POs with out-degree != 0
        'po_indegree_violations': 0,      # POs with in-degree == 0
        'isolated_nodes': 0,
        'is_structurally_valid': False, # Overall flag based on critical checks
        'constraints_failed': [] # List of reasons for failure
    }

    num_nodes = G.number_of_nodes()
    metrics['num_nodes'] = num_nodes

    if not isinstance(G, nx.DiGraph) or num_nodes == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object")
        return metrics # Return early for invalid input

    # 1. Check DAG property (Critical)
    metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
    if not metrics['is_dag']:
        metrics['constraints_failed'].append("Not a DAG")
        # Don't return early, continue checking other properties if possible

    # 2. Check Node Types and Basic Degrees
    node_type_counts = Counter()
    unknown_node_indices = []
    for node, data in G.nodes(data=True):
        node_type = data.get('type')
        node_type_counts[node_type] += 1

        if node_type not in VALID_AIG_NODE_TYPES:
            metrics['num_unknown_nodes'] += 1
            unknown_node_indices.append(node)
            continue # Skip degree checks for unknown nodes

        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        # Check degrees based on assigned type
        if node_type == 'NODE_CONST0':
            metrics['num_const0'] += 1
            if in_deg != 0:
                metrics['const0_indegree_violations'] += 1
        elif node_type == 'NODE_PI':
            metrics['num_pi'] += 1
            if in_deg != 0:
                 metrics['pi_indegree_violations'] += 1
        elif node_type == 'NODE_AND':
            metrics['num_and'] += 1
            if in_deg != 2:
                metrics['and_indegree_violations'] += 1
        elif node_type == 'NODE_PO':
            metrics['num_po'] += 1
            if out_deg != 0:
                 metrics['po_outdegree_violations'] += 1
            if in_deg == 0:
                 metrics['po_indegree_violations'] += 1

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
        if edge_type not in VALID_AIG_EDGE_TYPES:
            metrics['num_unknown_edges'] += 1

    if metrics['num_unknown_edges'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_edges']} unknown edge types")

    # 4. Check Basic AIG Requirements
    # Note: num_pi here includes node 0 if it was typed as PI
    if metrics['num_pi'] == 0 and metrics['num_const0'] == 0 :
         metrics['constraints_failed'].append("No Primary Inputs or Const0 found")
    if metrics['num_and'] < MIN_AND_COUNT :
        metrics['constraints_failed'].append(f"Insufficient AND gates ({metrics['num_and']} < {MIN_AND_COUNT})")
    if metrics['num_po'] < MIN_PO_COUNT:
        metrics['constraints_failed'].append(f"Insufficient POs ({metrics['num_po']} < {MIN_PO_COUNT})")

    # 5. Check isolated nodes
    isolates = list(nx.isolates(G))
    metrics['isolated_nodes'] = len(isolates)
    if metrics['isolated_nodes'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['isolated_nodes']} isolated nodes")

    # Determine overall structural validity based on critical failures
    # Requires DAG, no unknown types, and no critical degree violations.
    critical_degree_violations = (
        metrics['const0_indegree_violations'] +
        metrics['pi_indegree_violations'] +
        metrics['and_indegree_violations'] +
        metrics['po_outdegree_violations'] +
        metrics['po_indegree_violations']
    ) > 0

    if (metrics['is_dag'] and
        metrics['num_unknown_nodes'] == 0 and
        metrics['num_unknown_edges'] == 0 and
        not critical_degree_violations and
        metrics['num_pi'] + metrics['num_const0'] > 0 and # Must have at least one input source
        metrics['num_and'] >= MIN_AND_COUNT and
        metrics['num_po'] >= MIN_PO_COUNT and
        metrics['isolated_nodes'] == 0):
        metrics['is_structurally_valid'] = True

    return metrics


def count_pi_po_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Counts PIs reaching POs and POs reachable from PIs based on reachability.
    Uses assigned node types. Assumes graph object is valid.
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
        # Get nodes by assigned type
        pis = set()
        pos = set()
        const0_nodes = set()
        for node, data in G.nodes(data=True):
             node_type = data.get('type')
             if node_type == "NODE_PI":
                 pis.add(node)
             elif node_type == "NODE_PO":
                 pos.add(node)
             elif node_type == "NODE_CONST0":
                 const0_nodes.add(node)
                 # Treat const0 as a potential starting point for paths, like PIs
                 pis.add(node) # Add const0 to the set of sources

        results['num_pi'] = len(pis - const0_nodes) # Report actual PIs separately
        results['num_po'] = len(pos)
        results['num_const0'] = len(const0_nodes)

        # Use the combined set 'pis' (including const0) as sources for path checking
        source_nodes = pis

        if not source_nodes or not pos:
            return results # No paths possible if no sources or no destinations

        connected_sources = set()
        connected_pos = set()

        # Find all nodes reachable from any source (PI or CONST0)
        all_reachable_from_sources = set()
        for source_node in source_nodes:
            try:
                all_reachable_from_sources.update(nx.descendants(G, source_node))
                all_reachable_from_sources.add(source_node)
            except nx.NodeNotFound:
                logger.warning(f"Source node {source_node} not found during descendant search.")
            except Exception as e:
                logger.warning(f"Error finding descendants from source {source_node}: {e}")

        # Find all nodes that can reach any PO
        all_ancestors_of_pos = set()
        for po_node in pos:
            try:
                all_ancestors_of_pos.update(nx.ancestors(G, po_node))
                all_ancestors_of_pos.add(po_node)
            except nx.NodeNotFound:
                logger.warning(f"PO node {po_node} not found during ancestor search.")
            except Exception as e:
                logger.warning(f"Error finding ancestors for PO {po_node}: {e}")

        # Find sources that can reach at least one PO
        connected_sources = source_nodes.intersection(all_ancestors_of_pos)
        # Find POs that are reachable from at least one source
        connected_pos = pos.intersection(all_reachable_from_sources)

        # Report counts relative to actual PIs (excluding const0) if needed,
        # but connectivity fraction often includes const0 as a functional input
        num_sources_total = len(source_nodes) # PI + Const0
        results['num_pis_reaching_po'] = len(connected_sources) # Count includes const0 if connected
        results['num_pos_reachable_from_pi'] = len(connected_pos)

        # Calculate fractions based on sources (PI+Const0) and POs
        if num_sources_total > 0:
            # Fraction of PIs+Const0 that can reach a PO
            results['fraction_pis_connected'] = results['num_pis_reaching_po'] / num_sources_total
        if results['num_po'] > 0:
            results['fraction_pos_connected'] = results['num_pos_reachable_from_pi'] / results['num_po']

    except Exception as e:
        logger.error(f"Unexpected error during count_pi_po_paths execution: {e}", exc_info=True)
        results['error'] = f"Processing error: {e}"

    return results

# --- Main Evaluation Logic ---

def main(args):
    logger.info(f"Loading generated AIGs from: {args.input_pickle_file}")
    try:
        with open(args.input_pickle_file, 'rb') as f:
            generated_graphs = pickle.load(f)
        if not isinstance(generated_graphs, list):
            logger.error("Pickle file does not contain a list of graphs.")
            return
        logger.info(f"Loaded {len(generated_graphs)} graphs.")
    except FileNotFoundError:
        logger.error(f"Input pickle file not found: {args.input_pickle_file}")
        return
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        return

    if not generated_graphs:
        logger.warning("No graphs found in the pickle file. Exiting.")
        return

    num_total = len(generated_graphs)
    num_valid_structurally = 0
    aggregate_metrics = defaultdict(list)
    aggregate_path_metrics = defaultdict(list)
    failed_constraints_summary = Counter()

    logger.info("Starting evaluation...")
    for i, graph in enumerate(generated_graphs):
        if i % 100 == 0 and i > 0:
             logger.info(f"Processed {i}/{num_total} graphs...")

        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph, skipping.")
            failed_constraints_summary["Invalid Graph Object"] += 1
            continue

        # Calculate structural metrics
        struct_metrics = calculate_structural_aig_metrics(graph)
        if struct_metrics['is_structurally_valid']:
            num_valid_structurally += 1
        else:
            # Increment constraint failure counts
            for reason in struct_metrics.get('constraints_failed', []):
                failed_constraints_summary[reason] += 1

        # Store numerical metrics for averaging later
        for key, value in struct_metrics.items():
            if isinstance(value, (int, float, bool)):
                 aggregate_metrics[key].append(float(value)) # Convert bools to float for averaging


        # Calculate path metrics only if graph is structurally valid (or always, if desired)
        # If calculated always, might get skewed results for invalid graphs
        if struct_metrics['is_structurally_valid']:
             path_metrics = count_pi_po_paths(graph)
             if path_metrics.get('error') is None:
                  for key, value in path_metrics.items():
                     if isinstance(value, (int, float)):
                           aggregate_path_metrics[key].append(value)
             else:
                 logger.warning(f"Skipping path metrics for graph {i} due to error: {path_metrics['error']}")


    logger.info("Evaluation finished.")

    # --- Reporting ---
    validity_percentage = (num_valid_structurally / num_total) * 100 if num_total > 0 else 0

    print("\n--- G2PT AIG Evaluation Summary ---")
    print(f"Total Graphs Loaded             : {num_total}")
    print(f"Structurally Valid AIGs         : {num_valid_structurally} ({validity_percentage:.2f}%)")

    print("\n--- Average Structural Metrics (All Graphs) ---")
    for key, values in sorted(aggregate_metrics.items()):
        if key == 'is_structurally_valid': continue # Already reported percentage
        avg_value = np.mean(values) if values else 0
        # Format booleans nicely
        if key == 'is_dag':
             print(f"  - Avg {key:<27}: {avg_value*100:.2f}%")
        else:
             print(f"  - Avg {key:<27}: {avg_value:.3f}")


    print("\n--- Constraint Violation Summary (Across All Graphs) ---")
    total_violations = sum(failed_constraints_summary.values())
    if total_violations == 0:
        print("  No constraint violations detected.")
    else:
        # Sort reasons by count for clarity
        sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        for reason, count in sorted_reasons:
            reason_percentage = (count / total_violations) * 100
            print(f"  - {reason:<40}: {count:<6} ({reason_percentage:.1f}% of violations)")


    print("\n--- Average Path Connectivity Metrics (Valid Graphs Only) ---")
    if not aggregate_path_metrics:
         print("  No valid graphs to calculate path metrics.")
    else:
        num_graphs_for_path_metrics = len(aggregate_path_metrics.get('num_pi', [])) # Use any key's list length
        print(f"  (Based on {num_graphs_for_path_metrics} structurally valid graphs)")
        for key, values in sorted(aggregate_path_metrics.items()):
             if key == 'error': continue
             avg_value = np.mean(values) if values else 0.0
             print(f"  - Avg {key:<27}: {avg_value:.3f}")


    print("------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate structural validity and path connectivity of generated AIGs.')
    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs (e.g., generated_aigs.pkl).')

    parsed_args = parser.parse_args()
    main(parsed_args)