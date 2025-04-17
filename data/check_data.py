#!/usr/bin/env python3
"""
Update Node Type Encodings in AIG Graph Dataset

This script loads an AIG graph dataset, updates the node type encodings to use
a true one-hot encoding format, and saves the modified dataset to a new file.
"""

import os
import pickle
import networkx as nx
from typing import List, Dict, Any, Tuple
import numpy as np
import time

# Original node type encoding
ORIGINAL_NODE_TYPE_ENCODING = {
    "0": [0, 0, 0],
    "PI": [1, 0, 0],
    "AND": [0, 1, 0],
    "PO": [0, 0, 1]
}

# New true one-hot node type encoding
NEW_NODE_TYPE_ENCODING = {
    "0": [1, 0, 0, 0],
    "PI": [0, 1, 0, 0],
    "AND": [0, 0, 1, 0],
    "PO": [0, 0, 0, 1]
}

# Reverse mapping from encoding to node type name
ORIGINAL_ENCODING_TO_TYPE = {str(v): k for k, v in ORIGINAL_NODE_TYPE_ENCODING.items()}


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


def update_node_encodings(graphs: List[nx.DiGraph],
                          verbose: bool = True,
                          sample_interval: int = 1000) -> Tuple[List[nx.DiGraph], Dict[str, int]]:
    """
    Update node type encodings in all graphs from original to new format.

    Args:
        graphs: List of NetworkX DiGraph objects
        verbose: Whether to print progress updates
        sample_interval: How often to print progress updates

    Returns:
        Tuple of (updated graphs, statistics dictionary)
    """
    stats = {
        "total_graphs": len(graphs),
        "total_nodes": 0,
        "nodes_updated": 0,
        "graphs_with_no_node_types": 0,
        "type_counts": {
            "0": 0,
            "PI": 0,
            "AND": 0,
            "PO": 0,
            "unknown": 0
        }
    }

    start_time = time.time()

    for i, G in enumerate(graphs):
        if verbose and (i % sample_interval == 0 or i == len(graphs) - 1):
            elapsed = time.time() - start_time
            graphs_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"Processing graph {i + 1}/{len(graphs)} ({graphs_per_sec:.2f} graphs/s)")

        graph_has_node_types = False

        # Update node encodings
        for node, data in G.nodes(data=True):
            stats["total_nodes"] += 1

            if "type" in data:
                graph_has_node_types = True
                node_type = data["type"]

                # Convert numpy arrays to lists for comparison
                if isinstance(node_type, np.ndarray):
                    node_type = node_type.tolist()

                # Convert to string for dictionary lookup
                node_type_str = str(node_type)

                # Map the original encoding to node type name
                if node_type_str in ORIGINAL_ENCODING_TO_TYPE:
                    type_name = ORIGINAL_ENCODING_TO_TYPE[node_type_str]
                    stats["type_counts"][type_name] += 1

                    # Update to new encoding
                    data["type"] = NEW_NODE_TYPE_ENCODING[type_name]
                    stats["nodes_updated"] += 1
                else:
                    stats["type_counts"]["unknown"] += 1

        if not graph_has_node_types:
            stats["graphs_with_no_node_types"] += 1

    total_time = time.time() - start_time
    stats["processing_time"] = total_time
    stats["nodes_per_second"] = stats["total_nodes"] / total_time if total_time > 0 else 0

    return graphs, stats


def save_graphs(graphs: List[nx.DiGraph], output_path: str) -> None:
    """
    Save the updated graphs to a pickle file.

    Args:
        graphs: List of NetworkX DiGraph objects
        output_path: Path to save the pickle file
    """
    print(f"Saving {len(graphs)} graphs to {output_path}...")
    start_time = time.time()

    with open(output_path, "wb") as f:
        pickle.dump(graphs, f)

    save_time = time.time() - start_time
    print(f"Saved successfully in {save_time:.2f} seconds.")


def verify_update(original_graphs: List[nx.DiGraph],
                  updated_graphs: List[nx.DiGraph],
                  num_to_check: int = 5) -> None:
    """
    Verify that the update was performed correctly by comparing
    a sample of original and updated graphs.

    Args:
        original_graphs: List of original NetworkX DiGraph objects
        updated_graphs: List of updated NetworkX DiGraph objects
        num_to_check: Number of graphs to check
    """
    print(f"\nVerifying update on {num_to_check} sample graphs:")

    for i in range(min(num_to_check, len(original_graphs))):
        original_G = original_graphs[i]
        updated_G = updated_graphs[i]

        print(f"\nGraph {i + 1}:")

        # Check node count
        print(f"  Node count: Original={original_G.number_of_nodes()}, Updated={updated_G.number_of_nodes()}")

        # Check a sample node
        sample_nodes = list(original_G.nodes(data=True))[:3]
        for node_id, orig_data in sample_nodes:
            if "type" in orig_data:
                # Get original type
                orig_type = orig_data["type"]
                if isinstance(orig_type, np.ndarray):
                    orig_type = orig_type.tolist()

                # Get updated type
                updated_type = updated_G.nodes[node_id]["type"]
                if isinstance(updated_type, np.ndarray):
                    updated_type = updated_type.tolist()

                # Map to type names
                orig_type_str = str(orig_type)
                orig_type_name = ORIGINAL_ENCODING_TO_TYPE.get(orig_type_str, "unknown")

                print(f"  Node {node_id}:")
                print(f"    Original type: {orig_type} ({orig_type_name})")
                print(f"    Updated type: {updated_type}")

                # Check if update is correct
                expected_type = NEW_NODE_TYPE_ENCODING.get(orig_type_name, None)
                if expected_type is not None:
                    is_correct = updated_type == expected_type
                    print(f"    Correct update: {is_correct}")
                else:
                    print(f"    Unknown original type, can't verify")


def main():
    """Main function to update node type encodings."""
    # Try to find the pickle file
    home_dir = os.path.expanduser("~")
    potential_paths = [
        "final_data.pkl",
        "all_rand_aigs_data.pkl",
        os.path.join(home_dir, "Downloads", "final_data.pkl"),
        os.path.join(home_dir, "Downloads", "all_rand_aigs_data.pkl")
    ]

    pickle_path = None
    for path in potential_paths:
        if os.path.exists(path):
            pickle_path = path
            break

    if not pickle_path:
        pickle_path = input("Please enter the path to your pickle file: ").strip()

    # Ask for output path
    output_dir = os.path.dirname(pickle_path) or '.'
    default_output = os.path.join(output_dir, "updated_" + os.path.basename(pickle_path))
    output_path = input(f"Enter output path or press Enter for default [{default_output}]: ").strip()
    if not output_path:
        output_path = default_output

    # Load the graphs
    try:
        original_graphs = load_graphs(pickle_path)
    except Exception as e:
        print(f"Error loading graphs: {e}")
        return

    # Create a copy of the original graphs
    print("Creating a copy of the original graphs...")
    graphs_copy = pickle.loads(pickle.dumps(original_graphs))

    # Update node encodings
    print("\nUpdating node type encodings...")
    updated_graphs, stats = update_node_encodings(graphs_copy)

    # Print statistics
    print("\n=== UPDATE STATISTICS ===")
    print(f"Total graphs: {stats['total_graphs']}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Nodes updated: {stats['nodes_updated']} ({stats['nodes_updated'] / stats['total_nodes'] * 100:.2f}%)")
    print(f"Graphs with no node types: {stats['graphs_with_no_node_types']}")
    print("\nNode type counts:")
    for type_name, count in stats['type_counts'].items():
        print(f"  {type_name}: {count}")
    print(f"\nProcessing time: {stats['processing_time']:.2f} seconds")
    print(f"Processing speed: {stats['nodes_per_second']:.2f} nodes/second")

    # Verify the update
    verify_update(original_graphs, updated_graphs)

    # Confirm save
    should_save = input("\nSave updated graphs? (y/n): ").strip().lower()
    if should_save == 'y':
        save_graphs(updated_graphs, output_path)
        print(f"Updated graphs saved to {output_path}")
    else:
        print("Update canceled. No changes were saved.")


if __name__ == "__main__":
    main()