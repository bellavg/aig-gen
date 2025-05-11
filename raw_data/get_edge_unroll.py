#!/usr/bin/env python3
import os
import pickle  # Changed from torch
import warnings
import os.path as osp
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm
import networkx as nx  # For loading nx.DiGraph from PKL
from typing import List
# --- Assumed Imports (Make sure these work in your environment) ---
try:
    # Import config to get node type keys/indices
    # This assumes aig_config.py is in the same directory or accessible in PYTHONPATH
    import aig_config
except ImportError as e:
    # Try a relative import if this script is part of a package structure
    try:
        from . import aig_config  # If analyze_aigs.py is in the same package as aig_config.py
    except ImportError:
        print(f"Error importing aig_config: {e}")
        print("Please ensure aig_config.py is accessible (e.g., in the same directory or PYTHONPATH).")
        print("Using fallback node type keys if config is not found.")
        aig_config = None  # Fallback
# --- End Imports ---

# --- Constants from Config or Fallbacks ---
if aig_config:
    NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
    NODE_CONST0_STR = getattr(aig_config, 'NODE_CONST0_KEY', "NODE_CONST0")  # Assuming key names if they exist
    NODE_PI_STR = getattr(aig_config, 'NODE_PI_KEY', "NODE_PI")
    NODE_AND_STR = getattr(aig_config, 'NODE_AND_KEY', "NODE_AND")
    NODE_PO_STR = getattr(aig_config, 'NODE_PO_KEY', "NODE_PO")
    # One-hot encodings from your PKL generation script
    # These are used to map node 'type' attribute (a list) back to a string
    NODE_TYPE_ENCODING = getattr(aig_config, 'NODE_TYPE_ENCODING', {
        "NODE_CONST0": [1.0, 0.0, 0.0, 0.0],
        "NODE_PI": [0.0, 1.0, 0.0, 0.0],
        "NODE_AND": [0.0, 0.0, 1.0, 0.0],
        "NODE_PO": [0.0, 0.0, 0.0, 1.0]
    })
    # Create a reverse mapping from tuple(one_hot_vector) to type_string
    ONE_HOT_TO_NODE_TYPE_STR = {tuple(v): k for k, v in NODE_TYPE_ENCODING.items()}

else:  # Fallbacks if aig_config is not loaded
    NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
    NODE_CONST0_STR = "NODE_CONST0"
    NODE_PI_STR = "NODE_PI"
    NODE_AND_STR = "NODE_AND"
    NODE_PO_STR = "NODE_PO"
    ONE_HOT_TO_NODE_TYPE_STR = {
        (1.0, 0.0, 0.0, 0.0): "NODE_CONST0",
        (0.0, 1.0, 0.0, 0.0): "NODE_PI",
        (0.0, 0.0, 1.0, 0.0): "NODE_AND",
        (0.0, 0.0, 0.0, 1.0): "NODE_PO"
    }
NUM_NODE_FEATURES = len(NODE_TYPE_KEYS)  # Should be 4 based on typical AIG setup


# --- End Constants ---


def get_node_type_from_attrs(node_attrs: dict) -> str | None:
    """
    Determines the node type string from its attributes dictionary.
    Assumes the 'type' attribute is a one-hot encoded list/numpy array.
    """
    raw_type = node_attrs.get('type')
    if raw_type is None:
        warnings.warn(f"Node missing 'type' attribute: {node_attrs}")
        return "UNKNOWN"
    try:
        # Convert to tuple of floats for dict lookup
        type_tuple = tuple(float(x) for x in raw_type)
        return ONE_HOT_TO_NODE_TYPE_STR.get(type_tuple, "UNKNOWN_ENCODING")
    except Exception as e:
        warnings.warn(f"Error converting node type {raw_type} to tuple: {e}")
        return "UNKNOWN_CONVERSION_ERROR"


def analyze_degrees_and_distances_from_nx(graphs: List[nx.DiGraph]):
    """
    Analyzes in-degree, out-degree, and edge connection distances for different node types
    from a list of NetworkX DiGraphs.

    Args:
        graphs (List[nx.DiGraph]): A list of NetworkX DiGraph objects.

    Returns:
        tuple: (degree_data, distance_data)
            degree_data (dict): Contains in/out-degree stats per node type.
            distance_data (list): List of all calculated edge distances (target_idx - source_idx).
    """
    degree_data = defaultdict(lambda: defaultdict(list))
    all_edge_distances = []

    print(f"Analyzing {len(graphs)} graphs from PKL file...")
    for i, graph in enumerate(tqdm(graphs, desc="Analyzing Graphs")):
        if not isinstance(graph, nx.DiGraph):
            warnings.warn(f"Item {i} is not a NetworkX DiGraph. Skipping.")
            continue

        num_nodes = graph.number_of_nodes()
        if num_nodes == 0:
            continue

        # --- Node Type and Degree Analysis ---
        # Sort nodes to get a consistent ordering for this graph
        # This helps if original node IDs are not contiguous or not in generation order
        try:
            # Attempt to sort, assuming node IDs are sortable (e.g., integers)
            sorted_node_ids = sorted(list(graph.nodes()))
        except TypeError:
            # Fallback if node IDs are not sortable (e.g., mixed types, though unlikely for AIGs from your generator)
            warnings.warn(
                f"Graph {i}: Node IDs are not sortable. Using arbitrary order from graph.nodes(). This might affect distance interpretation if order matters.")
            sorted_node_ids = list(graph.nodes())

        node_to_ordered_idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}

        for original_node_id in sorted_node_ids:
            node_attrs = graph.nodes[original_node_id]
            node_type_str = get_node_type_from_attrs(node_attrs)

            if node_type_str and not node_type_str.startswith("UNKNOWN"):
                degree_data[node_type_str]['in'].append(graph.in_degree(original_node_id))
                degree_data[node_type_str]['out'].append(graph.out_degree(original_node_id))
            elif node_type_str:
                warnings.warn(
                    f"Graph {i}, Node {original_node_id}: Type '{node_type_str}' from attributes {node_attrs.get('type')}")

        # --- Edge Distance Analysis (Edge Unroll Clue) ---
        for u_old, v_old in graph.edges():
            # Get ordered indices
            u_new_idx = node_to_ordered_idx.get(u_old)
            v_new_idx = node_to_ordered_idx.get(v_old)

            if u_new_idx is None or v_new_idx is None:
                warnings.warn(
                    f"Graph {i}: Edge ({u_old}, {v_old}) contains node not in sorted_node_ids map. Skipping edge distance.")
                continue

            # We are interested in how "far back" a connection is made.
            # If target_idx > source_idx, this is a forward connection in the ordered list.
            # The distance is target_idx - source_idx.
            # This value indicates how many steps "back" the source node was, relative to the target.
            # The edge_unroll parameter in GraphDF typically defines how many previous nodes
            # the current node (target) considers connecting to.
            if v_new_idx > u_new_idx:  # Standard forward edge in the ordering
                distance = v_new_idx - u_new_idx
                all_edge_distances.append(distance)
            # else: # Edge goes "backwards" in the sorted list or is a self-loop (if u_new_idx == v_new_idx)
            # These are less typical for generative models that build sequentially,
            # but can exist in arbitrary AIGs.
            # For edge_unroll, we are primarily interested in connections to *previous* nodes.
            # If you want to capture all spans, you could use abs(v_new_idx - u_new_idx).
            # However, for `edge_unroll`, `target - source` (for target > source) is more direct.
            # warnings.warn(f"Graph {i}: Edge ({u_old}, {v_old}) -> ({u_new_idx}, {v_new_idx}) is not strictly forward in sorted order. Distance calculation might need adjustment based on interpretation.")

    return degree_data, all_edge_distances


def print_degree_stats(degree_data):
    """Prints formatted statistics for the collected degree data."""
    print("\n--- Node Degree Distribution Analysis ---")
    for node_type_key in NODE_TYPE_KEYS:  # Iterate in defined order
        print(f"\nNode Type: {node_type_key}")
        if node_type_key not in degree_data:
            print("  No nodes of this type found.")
            continue

        for degree_type in ['in', 'out']:
            degrees = degree_data[node_type_key][degree_type]
            if not degrees:
                print(f"  {degree_type.capitalize()}-Degrees: No data")
                continue

            degrees_np = np.array(degrees)
            count = len(degrees_np)
            min_deg = np.min(degrees_np)
            max_deg = np.max(degrees_np)
            mean_deg = np.mean(degrees_np)
            median_deg = np.median(degrees_np)
            p25 = np.percentile(degrees_np, 25)
            p75 = np.percentile(degrees_np, 75)
            p95 = np.percentile(degrees_np, 95)
            p99 = np.percentile(degrees_np, 99)

            print(f"  {degree_type.capitalize()}-Degrees ({count} nodes):")
            print(f"    Min: {min_deg:.0f}, Max: {max_deg:.0f}")
            print(f"    Mean: {mean_deg:.2f}, Median: {median_deg:.0f}")
            print(f"    Percentiles: 25th={p25:.0f}, 75th={p75:.0f}, 95th={p95:.0f}, 99th={p99:.0f}")

            if node_type_key == NODE_AND_STR and degree_type == 'in':
                non_two_count = np.sum(degrees_np != 2)
                if count > 0:
                    print(f"    AND nodes NOT having in-degree 2: {non_two_count} ({non_two_count / count:.1%})")


def print_distance_stats(distance_data):
    """Prints formatted statistics for edge connection distances."""
    print("\n--- Edge Connection Distance Analysis (for Edge Unroll) ---")
    if not distance_data:
        print("  No edge distance data collected.")
        return

    distances_np = np.array(distance_data)
    count = len(distances_np)
    min_dist = np.min(distances_np)
    max_dist = np.max(distances_np)
    mean_dist = np.mean(distances_np)
    median_dist = np.median(distances_np)

    print(f"  Analyzed {count} edges (where target_ordered_idx > source_ordered_idx).")
    print(f"  Min Distance: {min_dist:.0f}")
    print(f"  Max Distance: {max_dist:.0f}")
    print(f"  Mean Distance: {mean_dist:.2f}")
    print(f"  Median Distance: {median_dist:.0f}")

    percentiles_to_show = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
    print("  Percentiles of Edge Distances:")
    for p_val in percentiles_to_show:
        p = np.percentile(distances_np, p_val)
        print(f"    {p_val}th: {p:.0f}")
    print("\n  Interpretation for `edge_unroll`:")
    print("  The `edge_unroll` parameter in GraphDF determines how many previous nodes")
    print("  a newly generated node considers connecting to. The percentiles above")
    print("  (e.g., 95th, 99th) can give an empirical upper bound for this value.")
    print("  For example, if the 99th percentile is 15, an `edge_unroll` of 15-20")
    print("  might be sufficient to capture most connections in your dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze node degrees and edge connection distances from AIGs in a PKL file.")
    parser.add_argument('--pkl_file', type=str, required=True,
                        help="Path to the single PKL file containing a list of NetworkX DiGraphs.")

    args = parser.parse_args()

    if not osp.exists(args.pkl_file):
        print(f"Error: PKL file not found at {args.pkl_file}")
        exit(1)

    print(f"Loading NetworkX graphs from: {args.pkl_file}")
    try:
        with open(args.pkl_file, 'rb') as f:
            nx_graphs = pickle.load(f)
        if not isinstance(nx_graphs, list):
            print(f"Error: Expected a list of graphs in {args.pkl_file}, got {type(nx_graphs)}.")
            exit(1)
        if not all(isinstance(g, nx.DiGraph) for g in nx_graphs if
                   g is not None):  # Allow for None if some graphs failed generation
            print(f"Error: Not all items in the PKL file are NetworkX DiGraphs.")
            # Optionally print more details about non-DiGraph items
            for idx, item_in_list in enumerate(nx_graphs):
                if not isinstance(item_in_list, nx.DiGraph) and item_in_list is not None:
                    print(f" Item at index {idx} is of type: {type(item_in_list)}")
            # exit(1) # Decide if this is a fatal error or a warning
            warnings.warn("Some items in the PKL file are not NetworkX DiGraphs. They will be skipped.")
            nx_graphs = [g for g in nx_graphs if isinstance(g, nx.DiGraph)]


    except Exception as e:
        print(f"\nError loading or validating PKL file: {e}")
        exit(1)

    if not nx_graphs:
        print("No valid graphs found in the PKL file. No analysis to perform.")
        exit()

    print(f"Successfully loaded {len(nx_graphs)} graphs.")

    # Perform analysis
    degree_data, edge_distances = analyze_degrees_and_distances_from_nx(nx_graphs)

    # Print results
    print_degree_stats(degree_data)
    print_distance_stats(edge_distances)

    print("\nAnalysis finished.")
