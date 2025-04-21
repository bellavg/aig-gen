#!/usr/bin/env python3
"""
Simple PI and PO Count Analysis for AIG Graph Dataset

This script loads a single AIG graph dataset and reports the minimum and maximum
counts for Primary Inputs (PIs) and Primary Outputs (POs).
"""

import os
import pickle
import networkx as nx
from collections import Counter


def main():
    """Main function to analyze PI and PO counts in a single pickle file."""
    # Get the pickle file path from user
    pickle_path = input("Enter the path to your pickle file: ").strip()

    if not os.path.exists(pickle_path):
        print(f"Error: File not found at {pickle_path}")
        return

    # Load the graphs
    print(f"Loading graphs from {pickle_path}...")
    try:
        with open(pickle_path, "rb") as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs successfully.")
    except Exception as e:
        print(f"Error loading graphs: {e}")
        return

    # Analyze PI/PO counts
    pi_counts = []
    po_counts = []
    pi_distribution = Counter()
    po_distribution = Counter()

    print("Analyzing PI and PO counts...")
    for G in graphs:
        # Get PI and PO counts from graph attributes (preferred method)
        if "inputs" in G.graph and "outputs" in G.graph:
            pi_count = G.graph["inputs"]
            po_count = G.graph["outputs"]
        else:
            # Count from node types if attributes not available
            pi_count = 0
            po_count = 0
            for _, data in G.nodes(data=True):
                if "type" not in data:
                    continue

                node_type = data["type"]
                type_str = str(node_type)

                # Check for PI node (both original and one-hot encoding)
                if type_str == "[1, 0, 0]" or type_str == "[0, 1, 0, 0]":
                    pi_count += 1
                # Check for PO node (both original and one-hot encoding)
                elif type_str == "[0, 0, 1]" or type_str == "[0, 0, 0, 1]":
                    po_count += 1

        pi_counts.append(pi_count)
        po_counts.append(po_count)
        pi_distribution[pi_count] += 1
        po_distribution[po_count] += 1

    # Calculate statistics
    min_pi = min(pi_counts) if pi_counts else 0
    max_pi = max(pi_counts) if pi_counts else 0
    avg_pi = sum(pi_counts) / len(pi_counts) if pi_counts else 0

    min_po = min(po_counts) if po_counts else 0
    max_po = max(po_counts) if po_counts else 0
    avg_po = sum(po_counts) / len(po_counts) if po_counts else 0

    # Print results
    print("\n=== PI/PO COUNT ANALYSIS ===")
    print(f"Total graphs analyzed: {len(graphs)}")

    print("\nPrimary Inputs (PI):")
    print(f"  Minimum: {min_pi}")
    print(f"  Maximum: {max_pi}")
    print(f"  Average: {avg_pi:.2f}")

    print("\nPI distribution:")
    for count, num_graphs in sorted(pi_distribution.items()):
        percentage = (num_graphs / len(graphs)) * 100
        print(f"  {count} inputs: {num_graphs} graphs ({percentage:.1f}%)")

    print("\nPrimary Outputs (PO):")
    print(f"  Minimum: {min_po}")
    print(f"  Maximum: {max_po}")
    print(f"  Average: {avg_po:.2f}")

    print("\nPO distribution:")
    for count, num_graphs in sorted(po_distribution.items()):
        percentage = (num_graphs / len(graphs)) * 100
        print(f"  {count} outputs: {num_graphs} graphs ({percentage:.1f}%)")


if __name__ == "__main__":
    main()