# evaluate_g2pt_aigs.py
import pickle
import networkx as nx
import argparse
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np # For calculating averages and other stats
import math # For histogram bins

# --- Constants ---
# (Keep existing constants)
VALID_AIG_NODE_TYPES = {'NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'}
VALID_AIG_EDGE_TYPES = {'EDGE_INV', 'EDGE_REG'}
MAX_PI_COUNT = 14
MIN_PI_COUNT = 2
MIN_AND_COUNT = 1
MIN_PO_COUNT = 1
MAX_PO_COUNT = 30

# --- Logger Setup ---
# (Keep existing logger setup)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluate_g2pt_aigs")

# --- Adapted Evaluation Functions ---
# (Keep calculate_structural_aig_metrics and count_pi_po_paths as they are)
def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculates structural AIG validity metrics based on assigned types.
    Counts violations across the graph instead of breaking early.
    *** MODIFIED: Does NOT consider isolated nodes (other than CONST0) as a validity failure. ***
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
        'is_structurally_valid': False,
        'constraints_failed': [],
        'total_violation_count': 0 # ADDED: To quantify the degree of invalidity
    }
    node_0_is_const0 = False

    num_nodes = G.number_of_nodes()
    metrics['num_nodes'] = num_nodes

    if not isinstance(G, nx.DiGraph) or num_nodes == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object")
        # Assign a high violation count for fundamentally broken input
        metrics['total_violation_count'] = 1 # Or more, depending on how you want to weight this
        return metrics

    # 1. Check DAG property
    metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
    if not metrics['is_dag']:
        metrics['constraints_failed'].append("Not a DAG")
        metrics['total_violation_count'] += 1 # Count non-DAG as one major violation

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

        if node_type == 'NODE_CONST0':
            metrics['num_const0'] += 1
            if node == 0: node_0_is_const0 = True
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
                 metrics['po_indegree_violations'] += 1 # PO with 0 in-degree is problematic

    # Add failure reasons based on type/degree checks & update violation count
    if metrics['num_unknown_nodes'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_nodes']} unknown node types")
        metrics['total_violation_count'] += metrics['num_unknown_nodes'] # Count each unknown node
    if metrics['const0_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['const0_indegree_violations']} CONST0 nodes with incorrect in-degree")
        metrics['total_violation_count'] += metrics['const0_indegree_violations']
    if metrics['pi_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['pi_indegree_violations']} PI nodes with incorrect in-degree")
        metrics['total_violation_count'] += metrics['pi_indegree_violations']
    if metrics['and_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['and_indegree_violations']} AND nodes with incorrect in-degree")
        metrics['total_violation_count'] += metrics['and_indegree_violations']
    if metrics['po_outdegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_outdegree_violations']} PO nodes with incorrect out-degree")
        metrics['total_violation_count'] += metrics['po_outdegree_violations']
    if metrics['po_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_indegree_violations']} PO nodes with incorrect in-degree (0)")
        metrics['total_violation_count'] += metrics['po_indegree_violations']


    # 3. Check Edge Types
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type')
        if edge_type is not None and edge_type not in VALID_AIG_EDGE_TYPES:
            metrics['num_unknown_edges'] += 1
    if metrics['num_unknown_edges'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_edges']} unknown edge types")
        metrics['total_violation_count'] += metrics['num_unknown_edges'] # Count each unknown edge


    # 4. Check Basic AIG Requirements
    has_inputs = metrics['num_pi'] > 0 or metrics['num_const0'] > 0
    has_enough_ands = metrics['num_and'] >= MIN_AND_COUNT
    has_enough_pos = metrics['num_po'] >= MIN_PO_COUNT

    if not has_inputs :
         metrics['constraints_failed'].append("No Primary Inputs or Const0 found")
         metrics['total_violation_count'] += 1 # Count as one violation
    if not has_enough_ands :
        metrics['constraints_failed'].append(f"Insufficient AND gates ({metrics['num_and']} < {MIN_AND_COUNT})")
        metrics['total_violation_count'] += 1 # Count as one violation
    if not has_enough_pos:
        metrics['constraints_failed'].append(f"Insufficient POs ({metrics['num_po']} < {MIN_PO_COUNT})")
        metrics['total_violation_count'] += 1 # Count as one violation

    # --- 5. Check isolated nodes ---
    all_isolates = list(nx.isolates(G))
    relevant_isolates = []
    for node_idx in all_isolates:
        # Check if node exists and has data before accessing type
        if node_idx in G and G.nodes[node_idx]:
            isolated_node_type = G.nodes[node_idx].get('type')
            # Only count non-CONST0 isolates as relevant for this metric
            if isolated_node_type != 'NODE_CONST0':
                relevant_isolates.append(node_idx)
        # else: # Handle case where isolate node_idx might not be fully defined (less likely but safe)
        #    relevant_isolates.append(node_idx) # Or log a warning

    metrics['isolated_nodes'] = len(relevant_isolates)
    if metrics['isolated_nodes'] > 0:
       # Add to constraints_failed for reporting purposes, even if not affecting validity flag
       metrics['constraints_failed'].append(f"Found {metrics['isolated_nodes']} relevant isolated nodes (non-CONST0)")
       metrics['total_violation_count'] += metrics['isolated_nodes'] # Count each isolated node as a violation


    # Determine overall structural validity based on critical failures
    critical_degree_violations = (
        metrics['const0_indegree_violations'] +
        metrics['pi_indegree_violations'] +
        metrics['and_indegree_violations'] +
        metrics['po_outdegree_violations'] +
        metrics['po_indegree_violations'] # PO in-degree 0 is critical
    ) > 0

    # Simplified validity check based on the presence of critical failures
    # (Matching the logic from the original code regarding which checks determine validity)
    if (metrics['is_dag'] and
        metrics['num_unknown_nodes'] == 0 and
        metrics['num_unknown_edges'] == 0 and
        not critical_degree_violations and
        has_inputs and # Check the boolean flag directly
        has_enough_ands and
        has_enough_pos):
        metrics['is_structurally_valid'] = True
    # else: The flag remains False (invalid)

    # If the graph was determined invalid, ensure violation count is at least 1
    if not metrics['is_structurally_valid'] and metrics['total_violation_count'] == 0:
         # This might happen if validity criteria include checks not explicitly counted above
         # For instance, if the combination matters but individual counts were zero.
         # Add a generic violation if it's invalid but no specific counts were triggered.
         metrics['total_violation_count'] = 1
         # Optionally add a generic failure reason too:
         if not metrics['constraints_failed']: # Only if no specific reason was logged
              metrics['constraints_failed'].append("Failed validity criteria (unspecified combination)")


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
        pis_only = set() # Actual PIs
        pos = set()
        const0_nodes = set()
        for node, data in G.nodes(data=True):
             node_type = data.get('type')
             if node_type == "NODE_PI":
                 pis_only.add(node)
             elif node_type == "NODE_PO":
                 pos.add(node)
             elif node_type == "NODE_CONST0":
                 const0_nodes.add(node)

        # Combine PIs and Const0 for source node analysis
        source_nodes = pis_only.union(const0_nodes)

        results['num_pi'] = len(pis_only) # Report actual PIs separately
        results['num_po'] = len(pos)
        results['num_const0'] = len(const0_nodes)


        if not source_nodes or not pos:
             # If no PIs/Const0 or no POs, connectivity is trivially 0
            results['num_pis_reaching_po'] = 0
            results['num_pos_reachable_from_pi'] = 0
            results['fraction_pis_connected'] = 0.0
            results['fraction_pos_connected'] = 0.0
            return results

        connected_sources = set()
        connected_pos = set()

        # Find all nodes reachable from any source (PI or CONST0)
        all_reachable_from_sources = set()
        for source_node in source_nodes:
            try:
                desc = nx.descendants(G, source_node)
                all_reachable_from_sources.update(desc)
                all_reachable_from_sources.add(source_node) # Include the source itself
            except nx.NodeNotFound:
                logger.warning(f"Source node {source_node} not found during descendant search.")
            except Exception as e:
                # If graph is not a DAG, descendants/ancestors can fail
                if isinstance(e, nx.NetworkXUnfeasible):
                     logger.warning(f"Graph is not a DAG, cannot reliably compute reachability from source {source_node}.")
                     results['error'] = "Non-DAG graph, reachability unreliable."
                     # Optionally try to compute anyway, or just return/break
                     return results # Stop path calculation for non-DAGs
                else:
                    logger.warning(f"Error finding descendants from source {source_node}: {e}")


        # Find all nodes that can reach any PO
        all_ancestors_of_pos = set()
        for po_node in pos:
            try:
                anc = nx.ancestors(G, po_node)
                all_ancestors_of_pos.update(anc)
                all_ancestors_of_pos.add(po_node) # Include the PO itself
            except nx.NodeNotFound:
                logger.warning(f"PO node {po_node} not found during ancestor search.")
            except Exception as e:
                 if isinstance(e, nx.NetworkXUnfeasible):
                     logger.warning(f"Graph is not a DAG, cannot reliably compute reachability to PO {po_node}.")
                     results['error'] = "Non-DAG graph, reachability unreliable."
                     return results # Stop path calculation for non-DAGs
                 else:
                    logger.warning(f"Error finding ancestors for PO {po_node}: {e}")


        # Find sources that can reach at least one PO
        # A source is connected if it's an ancestor of *any* PO
        connected_sources = source_nodes.intersection(all_ancestors_of_pos)

        # Find POs that are reachable from at least one source
        # A PO is connected if it's reachable from *any* source
        connected_pos = pos.intersection(all_reachable_from_sources)


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

    # Aggregate metrics for ALL graphs (as before)
    aggregate_metrics = defaultdict(list)
    # Path metrics ONLY for VALID graphs (as before)
    aggregate_path_metrics = defaultdict(list)
    # Summary of constraint failure *types* across ALL graphs (as before)
    failed_constraints_summary = Counter()

    # --- NEW: Data collection specifically for INVALID graphs ---
    invalid_graph_indices = []
    invalid_graph_violation_counts = [] # Stores the 'total_violation_count' for each invalid graph
    invalid_graph_failure_reasons = Counter() # Counts failure reasons *only* for invalid graphs
    invalid_graph_metrics_aggregated = defaultdict(list) # Aggregate metrics *only* for invalid graphs

    logger.info("Starting evaluation...")
    for i, graph in enumerate(generated_graphs):
        if i % 100 == 0 and i > 0:
             logger.info(f"Processed {i}/{num_total} graphs...")

        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph, skipping.")
            failed_constraints_summary["Invalid Graph Object"] += 1
            # --- NEW: Also track this as an invalid graph ---
            invalid_graph_indices.append(i)
            invalid_graph_violation_counts.append(1) # Assign 1 violation for invalid object type
            invalid_graph_failure_reasons["Invalid Graph Object"] += 1
            continue

        # Calculate structural metrics (includes 'total_violation_count')
        struct_metrics = calculate_structural_aig_metrics(graph)

        # --- Store metrics for ALL graphs ---
        for key, value in struct_metrics.items():
             # Store numerical metrics for averaging later across all graphs
            if isinstance(value, (int, float, bool)) and key != 'constraints_failed':
                 aggregate_metrics[key].append(float(value)) # Convert bools to float for averaging

        # --- Process based on validity ---
        if struct_metrics['is_structurally_valid']:
            num_valid_structurally += 1
            # Calculate path metrics only if graph is structurally valid
            path_metrics = count_pi_po_paths(graph)
            if path_metrics.get('error') is None:
                  for key, value in path_metrics.items():
                     if isinstance(value, (int, float)):
                           aggregate_path_metrics[key].append(value)
            else:
                 logger.warning(f"Skipping path metrics for structurally valid graph {i} due to path calculation error: {path_metrics['error']}")
        else:
            # --- This graph is INVALID ---
            invalid_graph_indices.append(i)
            violation_count = struct_metrics.get('total_violation_count', 1) # Default to 1 if key missing
            invalid_graph_violation_counts.append(violation_count)

            # Increment overall constraint failure counts (as before)
            failure_reasons = struct_metrics.get('constraints_failed', ['Unknown Validity Failure'])
            if not failure_reasons: failure_reasons = ['Unknown Validity Failure'] # Ensure list isn't empty

            for reason in failure_reasons:
                failed_constraints_summary[reason] += 1
                # --- NEW: Also count reasons specifically for invalid graphs ---
                invalid_graph_failure_reasons[reason] += 1

            # --- NEW: Aggregate metrics specifically for invalid graphs ---
            for key, value in struct_metrics.items():
                 if isinstance(value, (int, float, bool)) and key != 'constraints_failed':
                      invalid_graph_metrics_aggregated[key].append(float(value))


    logger.info("Evaluation finished.")
    num_invalid = len(invalid_graph_indices)

    # --- Reporting ---
    validity_percentage = (num_valid_structurally / num_total) * 100 if num_total > 0 else 0
    invalidity_percentage = (num_invalid / num_total) * 100 if num_total > 0 else 0

    print("\n--- G2PT AIG Evaluation Summary ---")
    print(f"Total Graphs Loaded             : {num_total}")
    print(f"Structurally Valid AIGs         : {num_valid_structurally} ({validity_percentage:.2f}%)")
    print(f"Structurally Invalid AIGs       : {num_invalid} ({invalidity_percentage:.2f}%)") # Added

    print("\n--- Average Structural Metrics (All Graphs) ---")
    # (Keep this section as is - provides overall context)
    for key, values in sorted(aggregate_metrics.items()):
        if key == 'is_structurally_valid' or key == 'constraints_failed': continue
        avg_value = np.mean(values) if values else 0
        if key == 'is_dag':
             print(f"  - Avg {key:<27}: {avg_value*100:.2f}%")
        else:
             print(f"  - Avg {key:<27}: {avg_value:.3f}")

    print("\n--- Constraint Violation Summary (Across ALL Graphs) ---")
    # (Keep this section as is - shows prevalence of issues overall)
    total_violations_all = sum(failed_constraints_summary.values())
    if total_violations_all == 0:
        print("  No constraint violations detected across all graphs.")
    else:
        print(f"  (Total types of violations logged: {total_violations_all})")
        sorted_reasons_all = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        for reason, count in sorted_reasons_all:
            # Calculate percentage relative to the number of graphs where *any* violation could occur
            # Or relative to total violations logged, depending on interpretation.
            # Let's use percentage of total logged violation *instances*
            reason_percentage = (count / total_violations_all) * 100 if total_violations_all > 0 else 0
            # Alternative: percentage relative to *total graphs*
            # reason_percentage_vs_graphs = (count / num_total) * 100 if num_total > 0 else 0
            # print(f"  - {reason:<45}: {count:<6} ({reason_percentage:.1f}% of total violations, {reason_percentage_vs_graphs:.1f}% of graphs)")
            print(f"  - {reason:<50}: {count:<6} ({reason_percentage:.1f}% of logged violations)")


    print("\n--- Average Path Connectivity Metrics (Valid Graphs Only) ---")
    # (Keep this section as is)
    if not aggregate_path_metrics or not aggregate_path_metrics.get('num_po'): # Check if lists are populated
         print(f"  No valid graphs found or path metrics could not be calculated for valid graphs.")
    else:
        num_graphs_for_path_metrics = len(aggregate_path_metrics.get('num_po', [])) # Use a key guaranteed to exist if populated
        print(f"  (Based on {num_graphs_for_path_metrics} structurally valid graphs with successful path calculation)")
        for key, values in sorted(aggregate_path_metrics.items()):
             if key == 'error': continue
             avg_value = np.mean(values) if values else 0.0
             print(f"  - Avg {key:<27}: {avg_value:.3f}")

    # --- NEW: Detailed Analysis of INVALID Graphs ---
    print("\n--- Analysis of Invalid Graphs ---")
    if num_invalid == 0:
        print("  No invalid graphs found to analyze.")
    else:
        print(f"  Number of Invalid Graphs      : {num_invalid}")

        # 1. Degree of Invalidity (Number of Violations per Invalid Graph)
        print("\n  --- Degree of Invalidity (Violation Count per Invalid Graph) ---")
        if invalid_graph_violation_counts:
            avg_violations = np.mean(invalid_graph_violation_counts)
            median_violations = np.median(invalid_graph_violation_counts)
            min_violations = np.min(invalid_graph_violation_counts)
            max_violations = np.max(invalid_graph_violation_counts)
            std_dev_violations = np.std(invalid_graph_violation_counts)

            print(f"    - Average Violations per Invalid Graph : {avg_violations:.2f}")
            print(f"    - Median Violations per Invalid Graph  : {median_violations:.2f}")
            print(f"    - Min/Max Violations per Invalid Graph : {min_violations}/{max_violations}")
            print(f"    - Std Dev of Violations              : {std_dev_violations:.2f}")

            # Simple Histogram / Distribution
            print("\n    --- Distribution of Violation Counts ---")
            # Use numpy histogram to get counts and bin edges
            counts, bin_edges = np.histogram(invalid_graph_violation_counts, bins=max(1, min(10, int(max_violations)))) # Auto-binning up to 10 bins
            for i in range(len(counts)):
                 bin_lower = bin_edges[i]
                 bin_upper = bin_edges[i+1]
                 # Make bin labels more intuitive
                 if bin_upper - bin_lower < 1.1 : # Likely integer counts
                      if int(bin_lower) == int(bin_upper - 1):
                          label = f"Count = {int(bin_lower)}"
                      else:
                          label = f"Count = {int(bin_lower)} to {int(bin_upper-1)}"
                 else: # Floating point bins (less likely here, but good practice)
                      label = f"Count = {bin_lower:.1f} to <{bin_upper:.1f}"
                 percentage = (counts[i] / num_invalid) * 100
                 print(f"      - {label:<25}: {counts[i]:<5} graphs ({percentage:.1f}%)")

        else:
            print("    - No violation counts recorded for invalid graphs.")

        # 2. Character of Invalidity (Common Failure Reasons in Invalid Graphs)
        print("\n  --- Character of Invalidity (Common Failure Reasons within Invalid Graphs) ---")
        total_violations_invalid = sum(invalid_graph_failure_reasons.values())
        if total_violations_invalid == 0:
             print("    - No specific failure reasons logged for invalid graphs (might be 'Unknown Validity Failure').")
        else:
            print(f"    (Total types of violations logged across {num_invalid} invalid graphs: {total_violations_invalid})")
            sorted_reasons_invalid = sorted(invalid_graph_failure_reasons.items(), key=lambda item: item[1], reverse=True)
            for reason, count in sorted_reasons_invalid:
                 # Percentage relative to the number of invalid graphs
                 percentage_vs_invalid_graphs = (count / num_invalid) * 100
                 # Percentage relative to the total number of violation instances within invalid graphs
                 percentage_vs_total_violations = (count / total_violations_invalid) * 100
                 print(f"      - {reason:<50}: {count:<6} ({percentage_vs_invalid_graphs:.1f}% of invalid graphs had this issue; {percentage_vs_total_violations:.1f}% of total violations in invalid graphs)")

        # 3. Average Metrics for Invalid Graphs
        print("\n  --- Average Structural Metrics (Invalid Graphs Only) ---")
        if not invalid_graph_metrics_aggregated:
            print("    - No metrics collected for invalid graphs.")
        else:
            for key, values in sorted(invalid_graph_metrics_aggregated.items()):
                 if key == 'is_structurally_valid' or key == 'constraints_failed': continue # Skip non-numeric or boolean flag
                 avg_value = np.mean(values) if values else 0
                 if key == 'is_dag':
                     print(f"    - Avg {key:<27}: {avg_value*100:.2f}%")
                 else:
                     print(f"    - Avg {key:<27}: {avg_value:.3f}")


    print("\n------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate structural validity and path connectivity of generated AIGs, with detailed analysis of invalid graphs.')
    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs (e.g., generated_aigs.pkl).')

    parsed_args = parser.parse_args()
    main(parsed_args)