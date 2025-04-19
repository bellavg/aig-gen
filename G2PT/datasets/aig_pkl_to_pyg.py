# G2PT/datasets/prepare_aig_pyg.py
# Stage 1: Convert NetworkX AIGs from .pkl to PyG Data objects saved as .pt files.

import os
import pickle
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
# from torch_geometric.utils import from_networkx # We'll manually convert for directed graphs
from tqdm import tqdm
import warnings
from types import SimpleNamespace

# --- Configuration ---
CFG = SimpleNamespace(
    # --- Paths ---
    input_pickle_path='aig_data.pkl',     # <--- SET THIS: Path to your input AIG pickle file
    output_pyg_dir='aig/',             # <--- Base directory for PyG dataset output
                                             #      Raw files will be saved in output_pyg_dir/raw/

    # --- Data Handling ---
    split_ratios=(0.7, 0.15, 0.15),          # Train/Val/Test split ratios

    # --- Feature Mappings (Node: Assumes input type is a tuple like (1,0,0,0)) ---
    # Maps the original tuple representation to the desired one-hot feature vector
    node_type_map={
        tuple([0, 0, 0]): [1.0, 0.0, 0.0, 0.0], # CONST0 -> Index 0
        tuple([1, 0, 0]): [0.0, 1.0, 0.0, 0.0], # PI -> Index 1
        tuple([0, 1, 0]): [0.0, 0.0, 1.0, 0.0], # AND -> Index 2
        tuple([0, 0, 1]): [0.0, 0.0, 0.0, 1.0], # PO -> Index 3
    },
    # Default feature if node type in .pkl is unknown or missing 'type' attribute
    default_node_feature=[0.0, 0.0, 0.0, 0.0], # Represents an 'unknown' type (all zeros)
    num_node_features=4, # Should match the length of the feature vectors above

    # --- Feature Mappings (Edge: Assumes input type is a tuple like (1,0)) ---
    edge_type_map={
        tuple([1, 0]): [1.0, 0.0], # INV -> Index 0
        tuple([0, 1]): [0.0, 1.0], # REG -> Index 1
    },
    # Default feature if edge type in .pkl is unknown or missing 'type' attribute (using REG)
    default_edge_feature=[0.0, 1.0], # Default to REG
    num_edge_features=2, # Should match the length of the feature vectors above
)

# --- Main Conversion Logic ---
if __name__ == '__main__':
    print(f"--- Stage 1: AIG .pkl to PyG .pt Conversion ---")
    print(f"Input : {CFG.input_pickle_path}")
    print(f"Output Dir: {CFG.output_pyg_dir}")

    # --- Load Data ---
    try:
        with open(CFG.input_pickle_path, 'rb') as f:
            all_nx_graphs = pickle.load(f)
        print(f"Loaded {len(all_nx_graphs)} graphs from pickle.")
    except FileNotFoundError:
        print(f"Error: Input pickle file not found at {CFG.input_pickle_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading pickle file '{CFG.input_pickle_path}': {e}")
        exit(1)

    if not isinstance(all_nx_graphs, list):
        print(f"Error: Expected a list of graphs, got {type(all_nx_graphs)}")
        exit(1)

    # Filter and Convert to PyG Data objects
    all_pyg_data = []
    skipped_graphs = 0
    print("Converting NetworkX graphs to PyG Data objects...")

    for nx_graph in tqdm(all_nx_graphs, desc="Converting"):
        # Basic validation
        if not isinstance(nx_graph, nx.DiGraph):
            warnings.warn(f"Skipping item - not a NetworkX DiGraph.")
            skipped_graphs += 1
            continue
        if nx_graph.number_of_nodes() == 0:
            warnings.warn(f"Skipping item - graph has no nodes.")
            skipped_graphs += 1
            continue

        # --- Manual Conversion from NetworkX DiGraph to PyG Components ---
        node_list = list(nx_graph.nodes()) # Preserve node order
        node_map = {node_id: i for i, node_id in enumerate(node_list)} # Map NX id to 0-based index

        # 1. Node Features (x)
        x_features = []
        valid_graph = True
        for node_id in node_list:
            nx_attrs = nx_graph.nodes[node_id]
            # Ensure 'type' exists and convert to tuple for dict key lookup
            type_tuple = tuple(nx_attrs.get('type', []))
            feature = CFG.node_type_map.get(type_tuple)
            if feature is None: # Handle missing or unknown types
                 warnings.warn(f"Graph skipped: Node {node_id} has unknown or missing 'type' attribute (value: {nx_attrs.get('type')}).")
                 valid_graph = False
                 break
            x_features.append(feature)
        if not valid_graph:
            skipped_graphs += 1
            continue

        # 2. Edge Index and Edge Attributes (edge_index, edge_attr)
        edge_indices_list = []
        edge_attrs_list = []
        for u, v, nx_edge_data in nx_graph.edges(data=True):
            src_idx = node_map.get(u)
            dst_idx = node_map.get(v)
            # This check should ideally not be needed if graph integrity is good, but safety first
            if src_idx is None or dst_idx is None:
                 warnings.warn(f"Graph skipped: Edge ({u},{v}) refers to a node not in node_list.")
                 valid_graph = False
                 break

            # Get edge type, default if missing
            edge_type_tuple = tuple(nx_edge_data.get('type', CFG.default_edge_feature))
            edge_feature = CFG.edge_type_map.get(edge_type_tuple)
            if edge_feature is None:
                warnings.warn(f"Edge ({u},{v}) has unknown 'type' {edge_type_tuple}. Using default feature.")
                edge_feature = CFG.default_edge_feature # Use default

            edge_indices_list.append([src_idx, dst_idx])
            edge_attrs_list.append(edge_feature)

        if not valid_graph:
            skipped_graphs += 1
            continue

        # --- Create PyG Data object ---
        data = Data(
            # Node features (one-hot)
            x=torch.tensor(x_features, dtype=torch.float).reshape(-1, CFG.num_node_features),
            # Edge index [2, num_edges]
            edge_index=torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous() if edge_indices_list else torch.empty((2,0), dtype=torch.long),
            # Edge attributes (one-hot)
            edge_attr=torch.tensor(edge_attrs_list, dtype=torch.float).reshape(-1, CFG.num_edge_features) if edge_attrs_list else torch.empty((0, CFG.num_edge_features), dtype=torch.float),
            # Standard empty graph label for generation
            y=torch.zeros((1, 0), dtype=torch.float)
        )
        all_pyg_data.append(data)
    # --- End Conversion Loop ---

    print(f"Successfully converted {len(all_pyg_data)} graphs.")
    if skipped_graphs > 0:
        print(f"Skipped {skipped_graphs} graphs due to errors or invalid format.")
    if not all_pyg_data:
        print("Error: No graphs were converted successfully. Exiting.")
        exit(1)

    # --- Shuffle and Split ---
    print(f"Shuffling {len(all_pyg_data)} converted graphs...")
    np.random.shuffle(all_pyg_data)
    num_graphs = len(all_pyg_data)
    num_train = int(num_graphs * CFG.split_ratios[0])
    num_val = int(num_graphs * CFG.split_ratios[1])
    # Ensure robust split calculation for small datasets
    if num_train + num_val >= num_graphs:
        num_val = max(0, num_graphs - num_train) # Val gets remainder up to N
        num_test = 0
    else:
        num_test = num_graphs - num_train - num_val

    datasets = {
        'train': all_pyg_data[:num_train],
        'val': all_pyg_data[num_train : num_train + num_val],
        'test': all_pyg_data[num_train + num_val :]
    }
    print(f"Split sizes - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

    # --- Save Splits ---
    output_raw_dir = os.path.join(CFG.output_pyg_dir, 'raw')
    os.makedirs(output_raw_dir, exist_ok=True)
    print(f"Saving splits to {output_raw_dir}...")

    for split_name, data_list in datasets.items():
        if not data_list and split_name != 'test': # Allow empty test set
             print(f"Warning: Split '{split_name}' is empty.")
        # Only save if list is not empty or if it's the test set (to ensure file exists)
        # if data_list or split_name == 'test':
        save_path = os.path.join(output_raw_dir, f'{split_name}.pt')
        try:
            torch.save(data_list, save_path)
            print(f"Saved {split_name} split ({len(data_list)} graphs) to {save_path}")
        except Exception as e:
            print(f"Error saving {split_name} split to {save_path}: {e}")

    print("--- Stage 1: Conversion to PyG .pt files finished. ---")
    print(f"PyG dataset raw files saved in: {output_raw_dir}")
    print("Next steps:")
    print("1. Implement AIGPygDataset and AIGPygDataModule (like in aig_pyg_dataset.py).")
    print("2. Run prepare_aig.py script using the new DataModule to create final .bin files.")