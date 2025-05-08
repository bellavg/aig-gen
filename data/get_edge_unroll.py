# analyze_aig_degrees.py
import os
import torch
import warnings
import os.path as osp
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm

# --- Assumed Imports (Make sure these work in your environment) ---
try:
    # Import only the base loader, not the augmented one for analysis
    from aig_dataset import AIGDatasetLoader
    # Import config to get node type keys/indices
    import aig_config
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure aig_dataset.py and aig_config.py are accessible.")
    exit(1)
# --- End Imports ---

# --- Constants from Config ---
NODE_TYPE_KEYS = getattr(aig_config, 'NODE_TYPE_KEYS', ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"])
NODE_CONST0_STR = "NODE_CONST0"
NODE_PI_STR = "NODE_PI"
NODE_AND_STR = "NODE_AND"
NODE_PO_STR = "NODE_PO"
NUM_NODE_FEATURES = len(NODE_TYPE_KEYS)
# --- End Constants ---


def analyze_degrees(dataset_loader):
    """
    Analyzes in-degree and out-degree distributions for different node types
    in a PyTorch Geometric InMemoryDataset.

    Args:
        dataset_loader: An instance of AIGDatasetLoader (or similar)
                        containing the graph data.

    Returns:
        dict: A dictionary containing degree statistics for each node type.
              e.g., {'NODE_PI': {'in': [...], 'out': [...]}, ...}
    """
    degree_data = defaultdict(lambda: defaultdict(list))
    # { 'NODE_PI': {'in': [0, 0, ...], 'out': [1, 2, ...]}, ... }

    print(f"Analyzing {len(dataset_loader)} graphs...")
    for i in tqdm(range(len(dataset_loader)), desc="Analyzing Graphs"):
        try:
            data = dataset_loader.get(i)
        except Exception as e:
            warnings.warn(f"Could not load graph {i}: {e}")
            continue

        num_nodes = data.num_atom.item()
        if num_nodes == 0:
            continue

        # Ensure features and adjacency matrix are on CPU for numpy conversion
        node_features = data.x[:num_nodes].cpu() # Shape: [num_nodes, node_dim]
        adj = data.adj[:, :num_nodes, :num_nodes].cpu() # Shape: [bond_dim, num_nodes, num_nodes]

        # Determine node types (find index of '1' in one-hot features)
        try:
            node_type_indices = torch.argmax(node_features, dim=1).numpy()
        except Exception as e:
            warnings.warn(f"Could not determine node types for graph {i}: {e}")
            continue

        # Calculate degrees from the adjacency tensor (summing REG and INV channels)
        # In-degree: Sum over source nodes (dim 2) for each target node (dim 1)
        in_degrees = adj[:2, :, :].sum(dim=(0, 2)).numpy() # Sum channels 0,1 ; Sum sources (dim 2) -> [num_nodes]
        # Out-degree: Sum over target nodes (dim 1) for each source node (dim 2)
        out_degrees = adj[:2, :, :].sum(dim=(0, 1)).numpy() # Sum channels 0,1 ; Sum targets (dim 1) -> [num_nodes]

        # Store degrees based on node type
        for node_idx in range(num_nodes):
            type_idx = node_type_indices[node_idx]
            if 0 <= type_idx < len(NODE_TYPE_KEYS):
                type_str = NODE_TYPE_KEYS[type_idx]
                degree_data[type_str]['in'].append(in_degrees[node_idx])
                degree_data[type_str]['out'].append(out_degrees[node_idx])
            else:
                warnings.warn(f"Graph {i}, Node {node_idx}: Invalid type index {type_idx}")

    return degree_data

def print_degree_stats(degree_data):
    """Prints formatted statistics for the collected degree data."""
    print("\n--- Degree Distribution Analysis ---")
    for node_type in NODE_TYPE_KEYS:
        print(f"\nNode Type: {node_type}")
        if node_type not in degree_data:
            print("  No nodes of this type found.")
            continue

        for degree_type in ['in', 'out']:
            degrees = degree_data[node_type][degree_type]
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

            print(f"  {degree_type.capitalize()}-Degrees ({count} nodes):")
            print(f"    Min: {min_deg:.0f}, Max: {max_deg:.0f}")
            print(f"    Mean: {mean_deg:.2f}, Median: {median_deg:.0f}")
            print(f"    Percentiles: 25th={p25:.0f}, 75th={p75:.0f}, 95th={p95:.0f}")

            # Specific check for AND in-degree
            if node_type == NODE_AND_STR and degree_type == 'in':
                non_two_count = np.sum(degrees_np != 2)
                if count > 0:
                    print(f"    Nodes NOT having in-degree 2: {non_two_count} ({non_two_count/count:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze node degree distribution in pre-processed AIG dataset.")
    parser.add_argument('--data_root', type=str, default="./",
                        help="Root directory containing the dataset (e.g., './data/').")
    parser.add_argument('--dataset_name', type=str, default='aig',
                        help="Name of the dataset subdirectory (e.g., 'aig').")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help="Dataset split to analyze (train, val, or test).")

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}, Split: {args.split}")
    try:
        # Load without augmentation wrapper
        dataset = AIGDatasetLoader(
            root=args.data_root,
            name=args.dataset_name,
            dataset_type=args.split
        )
    except FileNotFoundError as e:
        print(f"\nError loading dataset: {e}")
        print("Please ensure the processed data file exists for the specified split.")
        exit(1)
    except Exception as e:
        print(f"\nError initializing dataset loader: {e}")
        exit(1)

    if len(dataset) == 0:
        print("Dataset is empty. No analysis to perform.")
        exit()

    # Perform analysis
    degree_stats = analyze_degrees(dataset)

    # Print results
    print_degree_stats(degree_stats)

    print("\nAnalysis finished.")