#!/usr/bin/env python3
"""
Find Maximum Node Count in AIG Graph Dataset

This script loads an AIG graph dataset and analyzes the node count distribution,
with a focus on finding the maximum number of nodes in any graph.
"""

import os
import pickle
import networkx as nx
from typing import List, Dict, Any, Tuple
import time
from collections import Counter


def load_graphs(pickle_path: str) -> List[nx.DiGraph]:
    """
    Load graphs from the pickle file.

    Args:
        pickle_path: Path to the pickle file containing the graph dataset

    Returns:
        List of NetworkX DiGraph objects
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")

    print(f"Loading graphs from {pickle_path}...")
    start_time = time.time()

    with open(pickle_path, "rb") as f:
        graphs = pickle.load(f)

    load_time = time.time() - start_time
    print(f"Loaded {len(graphs)} graphs in {load_time:.2f} seconds.")

    return graphs


def analyze_node_counts(graphs: List[nx.DiGraph]) -> Dict[str, Any]:
    """
    Analyze the node count distribution in the graph dataset.

    Args:
        graphs: List of NetworkX DiGraph objects

    Returns:
        Dictionary with analysis results
    """
    start_time = time.time()

    results = {
        "total_graphs": len(graphs),
        "node_counts": [],
        "min_nodes": float('inf'),
        "max_nodes": 0,
        "avg_nodes": 0,
        "median_nodes": 0,
        "graphs_by_size": Counter(),
        "largest_graphs": []
    }

    # Count nodes in each graph
    for i, G in enumerate(graphs):
        node_count = G.number_of_nodes()
        results["node_counts"].append(node_count)
        results["graphs_by_size"][node_count] += 1

        # Update min and max
        if node_count < results["min_nodes"]:
            results["min_nodes"] = node_count
        if node_count > results["max_nodes"]:
            results["max_nodes"] = node_count
            results["largest_graphs"] = [(i, node_count)]
        elif node_count == results["max_nodes"]:
            results["largest_graphs"].append((i, node_count))

    # Calculate average
    if results["node_counts"]:
        results["avg_nodes"] = sum(results["node_counts"]) / len(results["node_counts"])

    # Calculate median
    if results["node_counts"]:
        sorted_counts = sorted(results["node_counts"])
        mid = len(sorted_counts) // 2
        if len(sorted_counts) % 2 == 0:
            results["median_nodes"] = (sorted_counts[mid - 1] + sorted_counts[mid]) / 2
        else:
            results["median_nodes"] = sorted_counts[mid]

    # Get top 20 most common node counts
    results["most_common_sizes"] = results["graphs_by_size"].most_common(20)

    # Get histogram data
    histogram_data = []
    min_size = max(results["min_nodes"], 0)
    max_size = results["max_nodes"] + 1
    bin_size = max(1, (max_size - min_size) // 20)  # Aim for ~20 bins

    for bin_start in range(min_size, max_size, bin_size):
        bin_end = min(bin_start + bin_size, max_size)
        bin_count = sum(results["graphs_by_size"][size] for size in range(bin_start, bin_end))
        histogram_data.append((f"{bin_start}-{bin_end - 1}", bin_count))

    results["histogram"] = histogram_data

    # Time tracking
    analysis_time = time.time() - start_time
    results["analysis_time"] = analysis_time

    return results


def get_largest_graph_details(graphs: List[nx.DiGraph], graph_indices: List[Tuple[int, int]],
                              max_examples: int = 5) -> List[Dict[str, Any]]:
    """
    Get detailed information about the largest graphs.

    Args:
        graphs: List of NetworkX DiGraph objects
        graph_indices: List of tuples (index, node_count) for the largest graphs
        max_examples: Maximum number of examples to include

    Returns:
        List of dictionaries with details about the largest graphs
    """
    largest_graph_details = []

    for i, (graph_idx, node_count) in enumerate(graph_indices[:max_examples]):
        G = graphs[graph_idx]

        # Count node types
        node_types = {}
        type_counts = {"constant-0": 0, "PI": 0, "AND": 0, "PO": 0, "unknown": 0}

        for _, data in G.nodes(data=True):
            if "type" in data:
                node_type = data["type"]
                type_str = str(node_type)

                # Identify node type based on pattern
                if type_str == "[0, 0, 0]" or type_str == "[1, 0, 0, 0]":  # Original or new encoding
                    type_counts["constant-0"] += 1
                elif type_str == "[1, 0, 0]" or type_str == "[0, 1, 0, 0]":
                    type_counts["PI"] += 1
                elif type_str == "[0, 1, 0]" or type_str == "[0, 0, 1, 0]":
                    type_counts["AND"] += 1
                elif type_str == "[0, 0, 1]" or type_str == "[0, 0, 0, 1]":
                    type_counts["PO"] += 1
                else:
                    type_counts["unknown"] += 1

        # Get graph attributes
        details = {
            "graph_index": graph_idx,
            "node_count": node_count,
            "edge_count": G.number_of_edges(),
            "inputs": G.graph.get("inputs", "unknown"),
            "outputs": G.graph.get("outputs", "unknown"),
            "node_type_counts": type_counts
        }

        largest_graph_details.append(details)

    return largest_graph_details


def main():
    """Main function to find the maximum node count."""
    # Try to find the pickle file
    home_dir = os.path.expanduser("~")
    potential_paths = [
        "final_data.pkl",
        "all_rand_aigs_data.pkl",
        "updated_final_data.pkl",
        "updated_all_rand_aigs_data.pkl",
        os.path.join(home_dir, "Downloads", "final_data.pkl"),
        os.path.join(home_dir, "Downloads", "all_rand_aigs_data.pkl"),
        os.path.join(home_dir, "Downloads", "updated_final_data.pkl"),
        os.path.join(home_dir, "Downloads", "updated_all_rand_aigs_data.pkl")
    ]

    pickle_path = None
    for path in potential_paths:
        if os.path.exists(path):
            pickle_path = path
            break

    if not pickle_path:
        pickle_path = input("Please enter the path to your pickle file: ").strip()

    # Load the graphs
    try:
        graphs = load_graphs(pickle_path)
    except Exception as e:
        print(f"Error loading graphs: {e}")
        return

    # Analyze node counts
    print("\nAnalyzing node count distribution...")
    results = analyze_node_counts(graphs)

    # Print results
    print("\n=== NODE COUNT ANALYSIS ===")
    print(f"Total graphs: {results['total_graphs']}")
    print(f"Minimum node count: {results['min_nodes']}")
    print(f"Maximum node count: {results['max_nodes']}")
    print(f"Average node count: {results['avg_nodes']:.2f}")
    print(f"Median node count: {results['median_nodes']}")

    # Print histogram
    print("\nNode count distribution:")
    for bin_range, count in results["histogram"]:
        percentage = (count / results["total_graphs"]) * 100
        print(f"  {bin_range}: {count} graphs ({percentage:.1f}%)")

    # Print most common sizes
    print("\nMost common node counts:")
    for size, count in results["most_common_sizes"][:10]:
        percentage = (count / results["total_graphs"]) * 100
        print(f"  {size} nodes: {count} graphs ({percentage:.1f}%)")

    # Get details about the largest graphs
    print(f"\nFound {len(results['largest_graphs'])} graphs with {results['max_nodes']} nodes (the maximum)")
    largest_graph_details = get_largest_graph_details(graphs, results["largest_graphs"])

    print("\nLargest graph details:")
    for i, details in enumerate(largest_graph_details):
        print(f"\nLarge Graph {i + 1} (Index {details['graph_index']}):")
        print(f"  Nodes: {details['node_count']}")
        print(f"  Edges: {details['edge_count']}")
        print(f"  Inputs: {details['inputs']}")
        print(f"  Outputs: {details['outputs']}")
        print("  Node type distribution:")
        for node_type, count in details['node_type_counts'].items():
            if count > 0:
                percentage = (count / details['node_count']) * 100
                print(f"    {node_type}: {count} ({percentage:.1f}%)")

    print(f"\nAnalysis completed in {results['analysis_time']:.2f} seconds.")


if __name__ == "__main__":
    main()