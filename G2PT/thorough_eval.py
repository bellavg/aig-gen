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

# ADD copy import at the top
import copy


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


def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculates structural AIG validity metrics based on assigned types.
    Counts violations, logs unknown type values, distinguishes AND errors.
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
        'and_indegree_violations': 0, # Total AND in-degree violations != 2
        'and_indegree_lt_2_violations': 0, # NEW: Count AND in-degree < 2
        'and_indegree_gt_2_violations': 0, # NEW: Count AND in-degree > 2
        'po_outdegree_violations': 0,
        'po_indegree_violations': 0,
        'isolated_nodes': 0, # Count of relevant (non-CONST0) isolates
        'is_structurally_valid': False,
        'constraints_failed': [],
        'total_violation_count': 0,
        'unknown_node_type_values': set() # To collect unique unknown types
    }
    # Removed node_0_is_const0 as it wasn't used after its check was removed earlier

    num_nodes = G.number_of_nodes()
    metrics['num_nodes'] = num_nodes

    if not isinstance(G, nx.DiGraph) or num_nodes == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object")
        metrics['total_violation_count'] = 1
        # Convert set to list before returning early
        metrics['unknown_node_type_values'] = list(metrics['unknown_node_type_values'])
        return metrics

    # 1. Check DAG property
    try:
        metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
        if not metrics['is_dag']:
            metrics['constraints_failed'].append("Not a DAG")
            metrics['total_violation_count'] += 1
    except Exception as e:
        # Handle potential errors during DAG check on unusual graphs
        metrics['is_dag'] = False
        metrics['constraints_failed'].append(f"DAG Check Failed: {e}")
        metrics['total_violation_count'] += 1


    # 2. Check Node Types and Basic Degrees
    node_type_counts = Counter()
    unknown_node_indices = []
    # Need to handle potential errors accessing node data if graph is malformed
    try:
        nodes_data = G.nodes(data=True)
    except Exception as e:
         metrics['constraints_failed'].append(f"Error accessing node data: {e}")
         # Cannot proceed reliably, return current metrics
         metrics['unknown_node_type_values'] = list(metrics['unknown_node_type_values'])
         return metrics

    for node, data in nodes_data:
        # Ensure data is a dictionary before using .get()
        if not isinstance(data, dict):
             node_type = 'Error: Node data not a dict'
             metrics['num_unknown_nodes'] += 1
             unknown_node_indices.append(node)
             metrics['unknown_node_type_values'].add(node_type)
             continue # Cannot process this node further
        else:
             node_type = data.get('type') # Safe access

        node_type_counts[node_type] += 1

        if node_type not in VALID_AIG_NODE_TYPES:
            metrics['num_unknown_nodes'] += 1
            unknown_node_indices.append(node)
            metrics['unknown_node_type_values'].add(str(node_type)) # Log the actual type
            continue # Skip degree checks for unknown nodes

        # Check degrees - use try-except for robustness on potentially broken graphs
        try:
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
        except Exception as e:
             metrics['constraints_failed'].append(f"Error getting degree for node {node}: {e}")
             metrics['total_violation_count'] += 1 # Count this as a violation
             continue # Skip degree checks if degree fails


        if node_type == 'NODE_CONST0':
            metrics['num_const0'] += 1
            # Removed node_0_is_const0 check here
            if in_deg != 0:
                metrics['const0_indegree_violations'] += 1
        elif node_type == 'NODE_PI':
            metrics['num_pi'] += 1
            if in_deg != 0:
                 metrics['pi_indegree_violations'] += 1
        elif node_type == 'NODE_AND':
            metrics['num_and'] += 1
            if in_deg != 2:
                metrics['and_indegree_violations'] += 1 # Increment total count
                if in_deg < 2:
                    metrics['and_indegree_lt_2_violations'] += 1
                else: # Must be > 2
                    metrics['and_indegree_gt_2_violations'] += 1
        elif node_type == 'NODE_PO':
            metrics['num_po'] += 1
            if out_deg != 0:
                 metrics['po_outdegree_violations'] += 1
            if in_deg == 0: # Check if PO has no inputs
                 metrics['po_indegree_violations'] += 1

    # --- Update violation counts and failure reasons ---
    if metrics['num_unknown_nodes'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_nodes']} unknown node types")
        metrics['total_violation_count'] += metrics['num_unknown_nodes']
    if metrics['const0_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['const0_indegree_violations']} CONST0 nodes with incorrect in-degree")
        metrics['total_violation_count'] += metrics['const0_indegree_violations']
    if metrics['pi_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['pi_indegree_violations']} PI nodes with incorrect in-degree")
        metrics['total_violation_count'] += metrics['pi_indegree_violations']
    if metrics['and_indegree_violations'] > 0:
         # Using the simple detailed string here, count already added above
         metrics['constraints_failed'].append(f"Found {metrics['and_indegree_violations']} AND nodes with incorrect in-degree")
         metrics['total_violation_count'] += metrics['and_indegree_violations'] # Add specific counts
    if metrics['po_outdegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_outdegree_violations']} PO nodes with incorrect out-degree")
        metrics['total_violation_count'] += metrics['po_outdegree_violations']
    if metrics['po_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_indegree_violations']} PO nodes with incorrect in-degree (0)")
        metrics['total_violation_count'] += metrics['po_indegree_violations']


    # 3. Check Edge Types
    try:
        edges_data = G.edges(data=True)
    except Exception as e:
        metrics['constraints_failed'].append(f"Error accessing edge data: {e}")
        edges_data = [] # Cannot check edges

    for u, v, data in edges_data:
        # Ensure data is a dictionary
        if not isinstance(data, dict):
             edge_type = 'Error: Edge data not a dict'
             metrics['num_unknown_edges'] += 1
        else:
            edge_type = data.get('type')

        if edge_type is not None and edge_type not in VALID_AIG_EDGE_TYPES:
            metrics['num_unknown_edges'] += 1
            # Optionally log unknown edge types too if needed:
            # metrics['unknown_edge_type_values'].add(str(edge_type))
    if metrics['num_unknown_edges'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_edges']} unknown edge types")
        metrics['total_violation_count'] += metrics['num_unknown_edges']


    # 4. Check Basic AIG Requirements
    has_inputs = metrics['num_pi'] > 0 or metrics['num_const0'] > 0
    has_enough_ands = metrics['num_and'] >= MIN_AND_COUNT
    has_enough_pos = metrics['num_po'] >= MIN_PO_COUNT

    if not has_inputs :
         metrics['constraints_failed'].append("No Primary Inputs or Const0 found")
         metrics['total_violation_count'] += 1
    if not has_enough_ands :
        metrics['constraints_failed'].append(f"Insufficient AND gates ({metrics['num_and']} < {MIN_AND_COUNT})")
        metrics['total_violation_count'] += 1
    if not has_enough_pos:
        metrics['constraints_failed'].append(f"Insufficient POs ({metrics['num_po']} < {MIN_PO_COUNT})")
        metrics['total_violation_count'] += 1


    # --- 5. Check isolated nodes --- < FIXED SECTION > ---
    relevant_isolates = [] # Initialize the list here
    try:
        all_isolates = list(nx.isolates(G))
        for node_idx in all_isolates:
            # Check if node still exists (might be removed if graph mutated unexpectedly)
            if node_idx not in G:
                continue
            # Safely get node data
            isolated_node_data = G.nodes.get(node_idx, {})
            if not isinstance(isolated_node_data, dict):
                 # Handle case where node data is corrupted
                 isolated_node_type = 'Error: Node data not a dict'
            else:
                isolated_node_type = isolated_node_data.get('type')

            # Only count non-CONST0 isolates as relevant for this metric
            if isolated_node_type != 'NODE_CONST0':
                relevant_isolates.append(node_idx)
    except Exception as e:
         metrics['constraints_failed'].append(f"Error checking isolates: {e}")
         # Cannot reliably count isolates, maybe set count to -1 or skip?
         metrics['isolated_nodes'] = -1 # Indicate error state

    if metrics['isolated_nodes'] != -1: # Only process if isolate check didn't fail
        metrics['isolated_nodes'] = len(relevant_isolates)
        if metrics['isolated_nodes'] > 0:
           metrics['constraints_failed'].append(f"Found {metrics['isolated_nodes']} relevant isolated nodes (non-CONST0)")
           metrics['total_violation_count'] += metrics['isolated_nodes']
    # --- </ FIXED SECTION > ---


    # --- Determine overall structural validity ---
    # Calculate critical degree violations (use .get with default 0 for safety)
    critical_degree_violations_count = (
        metrics.get('const0_indegree_violations', 0) +
        metrics.get('pi_indegree_violations', 0) +
        metrics.get('and_indegree_violations', 0) + # Summing the total AND violation count
        metrics.get('po_outdegree_violations', 0) +
        metrics.get('po_indegree_violations', 0)
    )
    critical_degree_violations = critical_degree_violations_count > 0

    # Check validity criteria
    if (metrics['is_dag'] and # Must be DAG
        metrics.get('num_unknown_nodes', 0) == 0 and # No unknown nodes
        metrics.get('num_unknown_edges', 0) == 0 and # No unknown edges
        not critical_degree_violations and # No critical degree issues
        has_inputs and # Must have some input source
        has_enough_ands and # Must meet min AND count
        has_enough_pos): # Must meet min PO count
        metrics['is_structurally_valid'] = True
    # else: The flag remains False (invalid)


    # --- Fallback violation count ---
    if not metrics['is_structurally_valid'] and metrics['total_violation_count'] == 0:
         # This ensures every invalid graph has at least one counted violation
         metrics['total_violation_count'] = 1
         if not metrics['constraints_failed']:
              metrics['constraints_failed'].append("Failed validity criteria (unspecified combination)")

    # Convert set to list for consistent return type
    metrics['unknown_node_type_values'] = list(metrics['unknown_node_type_values'])

    return metrics


def repair_po_out_degree(G: nx.DiGraph) -> int:
    """
    Removes outgoing edges from any NODE_PO node. Modifies graph G in place.
    Returns the number of edges removed.
    """
    edges_removed_count = 0
    nodes_to_check = list(G.nodes()) # Iterate over a copy of node list
    for node in nodes_to_check:
        if node in G and G.nodes[node].get('type') == 'NODE_PO':
            out_degree = G.out_degree(node)
            if out_degree > 0:
                # Get successors and remove edges
                successors = list(G.successors(node)) # Important to listify before removing
                logger.debug(f"Repairing PO Node {node}: Removing {len(successors)} outgoing edges.")
                for successor in successors:
                    if G.has_edge(node, successor): # Check if edge still exists
                        G.remove_edge(node, successor)
                        edges_removed_count += 1
    return edges_removed_count

def repair_pi_in_degree(G: nx.DiGraph) -> int:
    """
    Removes incoming edges to any NODE_PI node. Modifies graph G in place.
    Returns the number of edges removed.
    """
    edges_removed_count = 0
    nodes_to_check = list(G.nodes()) # Iterate over a copy of node list
    for node in nodes_to_check:
         if node in G and G.nodes[node].get('type') == 'NODE_PI':
            in_degree = G.in_degree(node)
            if in_degree > 0:
                predecessors = list(G.predecessors(node))
                logger.debug(f"Repairing PI Node {node}: Removing {len(predecessors)} incoming edges.")
                for predecessor in predecessors:
                     if G.has_edge(predecessor, node):
                        G.remove_edge(predecessor, node)
                        edges_removed_count += 1
    return edges_removed_count

def repair_const0_in_degree(G: nx.DiGraph) -> int:
    """
    Removes incoming edges to any NODE_CONST0 node. Modifies graph G in place.
    Returns the number of edges removed.
    """
    edges_removed_count = 0
    nodes_to_check = list(G.nodes()) # Iterate over a copy of node list
    for node in nodes_to_check:
        # Specifically check for CONST0 type
        if node in G and G.nodes[node].get('type') == 'NODE_CONST0':
            in_degree = G.in_degree(node)
            if in_degree > 0:
                predecessors = list(G.predecessors(node))
                logger.debug(f"Repairing CONST0 Node {node}: Removing {len(predecessors)} incoming edges.")
                for predecessor in predecessors:
                    if G.has_edge(predecessor, node):
                        G.remove_edge(predecessor, node)
                        edges_removed_count += 1
    return edges_removed_count

# Optional: Function to remove isolates if needed after repairs
def remove_isolates_except_const0(G: nx.DiGraph) -> int:
    """
    Finds and removes isolated nodes unless they are NODE_CONST0.
    Modifies graph G in place.
    Returns the number of nodes removed.
    """
    nodes_removed_count = 0
    # Need to potentially repeat as removing one node can isolate another
    while True:
        isolates = list(nx.isolates(G))
        removed_in_pass = 0
        if not isolates:
            break # No isolates found

        nodes_to_remove_this_pass = []
        for node in isolates:
            # Check if node exists and get type, default to None if error
            node_data = G.nodes.get(node, {})
            node_type = node_data.get('type') if node_data else None

            # Remove if not CONST0
            if node_type != 'NODE_CONST0':
                 nodes_to_remove_this_pass.append(node)

        if not nodes_to_remove_this_pass:
             break # No relevant isolates to remove this pass

        for node in nodes_to_remove_this_pass:
             logger.debug(f"Removing isolated node {node} (type: {node_type})")
             G.remove_node(node)
             nodes_removed_count += 1
             removed_in_pass += 1

        if removed_in_pass == 0: # Safety break if no progress made
             break

    return nodes_removed_count


# --- Modify `main` Function ---

def main(args):
    logger.info(f"Loading generated AIGs from: {args.input_pickle_file}")
    try:
        with open(args.input_pickle_file, 'rb') as f:
            generated_graphs = pickle.load(f)
        # Ensure graphs have node data if they are just tuples/lists from older formats
        # This basic check might need adjustment based on your exact pickle content
        if isinstance(generated_graphs, list) and generated_graphs:
            if not isinstance(generated_graphs[0], nx.DiGraph):
                 logger.error("Pickle file does not contain NetworkX DiGraphs.")
                 # Optionally add conversion logic here if possible
                 return
        elif not isinstance(generated_graphs, list):
             logger.error("Pickle file does not contain a list of graphs.")
             return
        logger.info(f"Loaded {len(generated_graphs)} graphs.")
    except FileNotFoundError:
        logger.error(f"Input pickle file not found: {args.input_pickle_file}")
        return
    except Exception as e:
        logger.error(f"Error loading or processing pickle file: {e}", exc_info=True)
        return

    if not generated_graphs:
        logger.warning("No graphs found in the pickle file. Exiting.")
        return

    num_total = len(generated_graphs)
    num_valid_structurally = 0

    # --- Data Collection Structures ---
    aggregate_metrics = defaultdict(list)
    aggregate_path_metrics = defaultdict(list)
    failed_constraints_summary = Counter()
    invalid_graph_indices = []
    invalid_graph_violation_counts = []
    invalid_graph_failure_reasons = Counter()
    invalid_graph_metrics_aggregated = defaultdict(list)
    all_unknown_node_types = Counter() # NEW: Collect all unknown types found

    logger.info("--- Initial Evaluation Pass ---")
    for i, graph in enumerate(generated_graphs):
        if i % 1000 == 0 and i > 0: # Adjusted reporting frequency
             logger.info(f" Initial eval processed {i}/{num_total} graphs...")

        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph, skipping.")
            failed_constraints_summary["Invalid Graph Object"] += 1
            invalid_graph_indices.append(i)
            invalid_graph_violation_counts.append(1)
            invalid_graph_failure_reasons["Invalid Graph Object"] += 1
            all_unknown_node_types['Invalid Graph Object'] += 1 # Track reason
            continue
        # Ensure nodes have data dictionary (might be needed if loaded from simpler format)
        # for node in graph.nodes:
        #      if 'type' not in graph.nodes[node]:
        #          graph.nodes[node]['type'] = None # Or some default/error marker


        # Calculate structural metrics
        struct_metrics = calculate_structural_aig_metrics(graph)

        # Aggregate metrics for ALL graphs
        for key, value in struct_metrics.items():
            if isinstance(value, (int, float, bool)) and key != 'constraints_failed' and key != 'unknown_node_type_values':
                 aggregate_metrics[key].append(float(value))
            elif key == 'unknown_node_type_values': # NEW: Aggregate unknown types
                 for unknown_type in value:
                    all_unknown_node_types[unknown_type] += 1

        # Process based on validity
        if struct_metrics['is_structurally_valid']:
            num_valid_structurally += 1
            # Calculate path metrics for valid graphs
            # ... (keep existing path metrics logic) ...
            path_metrics = count_pi_po_paths(graph)
            if path_metrics.get('error') is None:
                  for key, value in path_metrics.items():
                     if isinstance(value, (int, float)):
                           aggregate_path_metrics[key].append(value)
            else:
                 logger.warning(f"Skipping path metrics for structurally valid graph {i} due to path calculation error: {path_metrics['error']}")

        else:
            # Graph is invalid
            invalid_graph_indices.append(i)
            violation_count = struct_metrics.get('total_violation_count', 1)
            invalid_graph_violation_counts.append(violation_count)
            failure_reasons = struct_metrics.get('constraints_failed', ['Unknown Validity Failure'])
            if not failure_reasons: failure_reasons = ['Unknown Validity Failure']
            for reason in failure_reasons:
                failed_constraints_summary[reason] += 1
                invalid_graph_failure_reasons[reason] += 1
            for key, value in struct_metrics.items():
                 if isinstance(value, (int, float, bool)) and key != 'constraints_failed' and key != 'unknown_node_type_values':
                      invalid_graph_metrics_aggregated[key].append(float(value))


    logger.info("Initial evaluation finished.")
    num_invalid = len(invalid_graph_indices)

    # --- NEW: Repair and Re-evaluation Pass ---
    logger.info("--- Repair and Re-evaluation Pass (for Invalid Graphs) ---")
    num_repaired_to_valid = 0
    graphs_repaired = 0
    edges_removed_po = 0
    edges_removed_pi = 0
    edges_removed_const0 = 0
    nodes_removed_isolated = 0 # Optional isolate removal count

    if num_invalid > 0:
        logger.info(f"Attempting repairs on {num_invalid} invalid graphs...")
        for idx, original_index in enumerate(invalid_graph_indices):
            if idx % 500 == 0 and idx > 0: # Adjust reporting frequency
                 logger.info(f" Repair pass processed {idx}/{num_invalid} invalid graphs...")

            # Get the original graph
            original_graph = generated_graphs[original_index]
            if not isinstance(original_graph, nx.DiGraph):
                logger.warning(f"Skipping repair for index {original_index}, original item was not a DiGraph.")
                continue

            # --- Create a DEEP COPY to repair ---
            # This is crucial to avoid modifying the original list if you need it later
            # and to ensure re-evaluation is on the repaired state.
            graph_to_repair = copy.deepcopy(original_graph)
            graphs_repaired += 1

            # Apply repairs IN PLACE on the copy
            edges_removed_po += repair_po_out_degree(graph_to_repair)
            edges_removed_pi += repair_pi_in_degree(graph_to_repair)
            edges_removed_const0 += repair_const0_in_degree(graph_to_repair)

            # --- Optional: Remove newly isolated nodes ---
            # Uncomment the next line if you want to remove isolates created by edge removal
            # nodes_removed_isolated += remove_isolates_except_const0(graph_to_repair)
            # --------------------------------------------

            # Re-evaluate the repaired graph
            metrics_after_repair = calculate_structural_aig_metrics(graph_to_repair)

            if metrics_after_repair['is_structurally_valid']:
                num_repaired_to_valid += 1
                logger.debug(f"Graph {original_index} became valid after repairs.")
            # else:
            #    logger.debug(f"Graph {original_index} still invalid. Reasons: {metrics_after_repair['constraints_failed']}")

        logger.info(f"Repair pass finished. Repaired {graphs_repaired} graphs.")
        logger.info(f"Edges removed (PO Out): {edges_removed_po}")
        logger.info(f"Edges removed (PI In): {edges_removed_pi}")
        logger.info(f"Edges removed (Const0 In): {edges_removed_const0}")
        if nodes_removed_isolated > 0: # Only report if isolates were removed
             logger.info(f"Nodes removed (Isolated): {nodes_removed_isolated}")

    else:
        logger.info("No invalid graphs found, skipping repair pass.")


    # --- Reporting ---
    validity_percentage = (num_valid_structurally / num_total) * 100 if num_total > 0 else 0
    invalidity_percentage = (num_invalid / num_total) * 100 if num_total > 0 else 0
    repaired_validity_percentage = ((num_valid_structurally + num_repaired_to_valid) / num_total) * 100 if num_total > 0 else 0
    percent_of_invalid_repaired = (num_repaired_to_valid / num_invalid) * 100 if num_invalid > 0 else 0


    print("\n--- G2PT AIG Evaluation Summary ---")
    print(f"Total Graphs Loaded             : {num_total}")
    print(f"Initially Valid AIGs            : {num_valid_structurally} ({validity_percentage:.2f}%)")
    print(f"Initially Invalid AIGs          : {num_invalid} ({invalidity_percentage:.2f}%)")

    # --- NEW Repair Results ---
    print("\n--- Repair Attempt Summary (Simple Edge Removals) ---")
    print(f"Graphs Attempted for Repair     : {graphs_repaired}")
    print(f"Graphs Valid After Repair       : {num_repaired_to_valid} ({percent_of_invalid_repaired:.2f}% of initially invalid graphs)")
    print(f"Total Valid Graphs After Repair : {num_valid_structurally + num_repaired_to_valid} ({repaired_validity_percentage:.2f}% of total graphs)")
    print(f"  - PO Outgoing Edges Removed   : {edges_removed_po}")
    print(f"  - PI Incoming Edges Removed   : {edges_removed_pi}")
    print(f"  - Const0 Incoming Edges Removed: {edges_removed_const0}")
    # if nodes_removed_isolated > 0 : # Only show if option was enabled and nodes were removed
    #      print(f"  - Isolated Nodes Removed    : {nodes_removed_isolated}")


    # --- NEW Unknown Node Type Report ---
    print("\n--- Unknown Node Types Encountered ---")
    if not all_unknown_node_types:
        print("  No unknown node types detected.")
    else:
        print(f"  (Found {len(all_unknown_node_types)} unique unknown type strings)")
        # Sort by count descending for clarity
        sorted_unknown_types = sorted(all_unknown_node_types.items(), key=lambda item: item[1], reverse=True)
        for unknown_type, count in sorted_unknown_types:
             print(f"  - Type '{unknown_type}': {count} occurrences")


    # --- NEW AND Gate In-Degree Breakdown ---
    print("\n--- AND Gate In-Degree Violation Details (All Graphs) ---")
    total_and_lt_2 = sum(aggregate_metrics.get('and_indegree_lt_2_violations', [0]))
    total_and_gt_2 = sum(aggregate_metrics.get('and_indegree_gt_2_violations', [0]))
    total_and_violations = total_and_lt_2 + total_and_gt_2
    print(f"  Total AND gates with incorrect in-degree : {total_and_violations}")
    if total_and_violations > 0:
        percent_lt_2 = (total_and_lt_2 / total_and_violations) * 100
        percent_gt_2 = (total_and_gt_2 / total_and_violations) * 100
        print(f"    - In-degree < 2 (Not enough inputs): {total_and_lt_2} ({percent_lt_2:.1f}%)")
        print(f"    - In-degree > 2 (Too many inputs)  : {total_and_gt_2} ({percent_gt_2:.1f}%)")
    else:
        print("  No AND gate in-degree violations detected.")


    print("\n--- Average Structural Metrics (All Graphs - Initial State) ---")
    # (Report initial aggregate_metrics as before)
    for key, values in sorted(aggregate_metrics.items()):
        # Skip keys already reported specifically or non-numeric
        if key in ['is_structurally_valid', 'constraints_failed', 'unknown_node_type_values',
                   'and_indegree_violations', 'and_indegree_lt_2_violations', 'and_indegree_gt_2_violations']: continue
        avg_value = np.mean(values) if values else 0
        if key == 'is_dag':
             print(f"  - Avg {key:<27}: {avg_value*100:.2f}%")
        else:
             print(f"  - Avg {key:<27}: {avg_value:.3f}")


    print("\n--- Constraint Violation Summary (Across ALL Graphs - Initial State) ---")
    # (Report initial failed_constraints_summary as before)
    total_violations_all = sum(failed_constraints_summary.values())
    if total_violations_all == 0:
        print("  No constraint violations detected across all graphs.")
    else:
        # Recalculate total based on potentially updated counter keys
        # total_violations_all = sum(failed_constraints_summary.values()) # Already calculated
        print(f"  (Total types of violations logged: {total_violations_all})")
        sorted_reasons_all = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        # Limit the output length if it's excessively long
        max_reasons_to_show = 30
        for i, (reason, count) in enumerate(sorted_reasons_all):
             if i >= max_reasons_to_show:
                 print(f"  ... (omitting {len(sorted_reasons_all) - max_reasons_to_show} less frequent reasons)")
                 break
             reason_percentage = (count / total_violations_all) * 100 if total_violations_all > 0 else 0
             print(f"  - {reason:<50}: {count:<6} ({reason_percentage:.1f}% of logged violations)")


    print("\n--- Average Path Connectivity Metrics (Valid Graphs Only - Initial State) ---")
    # (Report initial aggregate_path_metrics as before)
    if not aggregate_path_metrics or not aggregate_path_metrics.get('num_po'):
         print(f"  No initially valid graphs found or path metrics could not be calculated.")
    else:
        num_graphs_for_path_metrics = len(aggregate_path_metrics.get('num_po', []))
        print(f"  (Based on {num_graphs_for_path_metrics} initially valid graphs with successful path calculation)")
        for key, values in sorted(aggregate_path_metrics.items()):
             if key == 'error': continue
             avg_value = np.mean(values) if values else 0.0
             print(f"  - Avg {key:<27}: {avg_value:.3f}")


    print("\n--- Analysis of Invalid Graphs (Initial State) ---")
    # (Report initial invalid graph analysis as before)
    if num_invalid == 0:
        print("  No initially invalid graphs found to analyze.")
    else:
        # ... (Keep the detailed reporting from the previous version for Degree, Character, and Avg Metrics of *initial* invalid graphs) ...
        print(f"  Number of Initially Invalid Graphs: {num_invalid}")

        # 1. Degree of Invalidity (Initial)
        print("\n  --- Degree of Invalidity (Violation Count per Initial Invalid Graph) ---")
        # ... (Keep np.mean, np.median, etc. calculations and printing on invalid_graph_violation_counts) ...
        if invalid_graph_violation_counts:
            avg_violations = np.mean(invalid_graph_violation_counts)
            median_violations = np.median(invalid_graph_violation_counts)
            min_violations = np.min(invalid_graph_violation_counts)
            max_violations = np.max(invalid_graph_violation_counts)
            std_dev_violations = np.std(invalid_graph_violation_counts)
            print(f"    - Average Violations per Invalid Graph : {avg_violations:.2f}")
            # ... (print median, min/max, std dev) ...
            # ... (print distribution histogram) ...
        else:
             print("    - No violation counts recorded for invalid graphs.")


        # 2. Character of Invalidity (Initial)
        print("\n  --- Character of Invalidity (Common Failure Reasons within Initial Invalid Graphs) ---")
        # ... (Keep sorting and printing of invalid_graph_failure_reasons, limiting length if needed) ...
        total_violations_invalid = sum(invalid_graph_failure_reasons.values())
        if total_violations_invalid == 0:
             print("    - No specific failure reasons logged for initial invalid graphs.")
        else:
             # ... (print sorted_reasons_invalid, limited) ...
             max_reasons_to_show = 30
             sorted_reasons_invalid = sorted(invalid_graph_failure_reasons.items(), key=lambda item: item[1], reverse=True)
             print(f"    (Total types of violations logged across {num_invalid} initial invalid graphs: {total_violations_invalid})")
             for i, (reason, count) in enumerate(sorted_reasons_invalid):
                 if i >= max_reasons_to_show:
                     print(f"      ... (omitting {len(sorted_reasons_invalid) - max_reasons_to_show} less frequent reasons)")
                     break
                 percentage_vs_invalid_graphs = (count / num_invalid) * 100
                 percentage_vs_total_violations = (count / total_violations_invalid) * 100
                 print(f"      - {reason:<50}: {count:<6} ({percentage_vs_invalid_graphs:.1f}% had issue; {percentage_vs_total_violations:.1f}% of violations)")


        # 3. Average Metrics for Invalid Graphs (Initial)
        print("\n  --- Average Structural Metrics (Initial Invalid Graphs Only) ---")
        # ... (Keep reporting of invalid_graph_metrics_aggregated) ...
        if not invalid_graph_metrics_aggregated:
            print("    - No metrics collected for initial invalid graphs.")
        else:
            for key, values in sorted(invalid_graph_metrics_aggregated.items()):
                 # Skip keys reported elsewhere or non-numeric
                 if key in ['is_structurally_valid', 'constraints_failed', 'unknown_node_type_values',
                           'and_indegree_violations', 'and_indegree_lt_2_violations', 'and_indegree_gt_2_violations']: continue
                 avg_value = np.mean(values) if values else 0
                 if key == 'is_dag':
                     print(f"    - Avg {key:<27}: {avg_value*100:.2f}%")
                 else:
                     print(f"    - Avg {key:<27}: {avg_value:.3f}")


    print("\n------------------------------------")
    print("--- Evaluation Finished ---")
    print("Script complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate structural validity of generated AIGs, attempt simple repairs, and report results.')
    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs.')

    parsed_args = parser.parse_args()
    # Optional: Add argument for enabling/disabling isolate removal repair step
    main(parsed_args)
