import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
# We need the collate function to mimic InMemoryDataset's saving format
from torch_geometric.data.collate import collate
import warnings
import os.path as osp
import argparse
import gc

# --- Hardcoded AIG Parameters ---
# These should match your model's expectations and the data generation
MAX_NODES = 64             # Max nodes for padding
NUM_NODE_FEATURES = 4      # Node types: CONST0, PI, AND, PO (one-hot)
NUM_EXPLICIT_EDGE_TYPES = 2 # Edge types: REG, INV
NUM_ADJ_CHANNELS = NUM_EXPLICIT_EDGE_TYPES + 1 # +1 for no-edge
# --- End Hardcoding ---

def _process_graph(graph, max_nodes, num_node_features, num_adj_channels, num_explicit_edge_types):
    """
    Converts a single NetworkX graph to a PyTorch Geometric Data object
    with padding for dense processing. Matches the logic from the user's
    working simpler dataset, including node mapping and 3-channel adjacency.

    Returns None if the graph is invalid or cannot be processed.
    """
    num_nodes_in_graph = graph.number_of_nodes()

    # Basic checks
    if num_nodes_in_graph == 0:
        warnings.warn("Skipping graph with 0 nodes.")
        return None
    if num_nodes_in_graph > max_nodes:
        warnings.warn(f"Skipping graph with {num_nodes_in_graph} nodes (max allowed: {max_nodes}).")
        return None

    # --- Node ID Mapping ---
    node_list = list(graph.nodes())
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    # --- Node Features ---
    node_features_list = []
    valid_nodes = True
    for old_node_id in node_list:
        attrs = graph.nodes[old_node_id]
        if 'type' not in attrs or not isinstance(attrs['type'], (list, np.ndarray)) or len(attrs['type']) < num_node_features:
            warnings.warn(f"Node {old_node_id} missing 'type' or 'type' has < {num_node_features} elements. Skipping graph.")
            valid_nodes = False; break
        # Ensure it's a numpy array before converting to tensor for consistency
        node_type_vec = np.asarray(attrs['type'][:num_node_features])
        node_feature = torch.tensor(node_type_vec, dtype=torch.float)
        node_features_list.append(node_feature)
    if not valid_nodes: return None

    # Stack and Pad Node Features
    if not node_features_list: return None
    x_stacked = torch.stack(node_features_list)
    pad_size = max_nodes - num_nodes_in_graph
    if pad_size < 0: return None
    x_padded = F.pad(x_stacked, (0, 0, 0, pad_size), "constant", 0)

    # --- Adjacency Matrix ---
    adj_matrix = torch.zeros((num_adj_channels, max_nodes, max_nodes), dtype=torch.float)
    adj_matrix[num_explicit_edge_types, :, :] = 1.0 # Initialize no-edge channel

    valid_edges = True
    for u_old, v_old, attrs in graph.edges(data=True):
        if 'type' not in attrs or not isinstance(attrs['type'], (list, np.ndarray)) or len(attrs['type']) < num_explicit_edge_types:
             warnings.warn(f"Edge ({u_old}-{v_old}) missing 'type' or 'type' has < {num_explicit_edge_types} elements. Skipping graph.")
             valid_edges = False; break

        u = node_id_map.get(u_old)
        v = node_id_map.get(v_old)
        if u is None or v is None:
             warnings.warn(f"Edge ({u_old}-{v_old}) involves node not in map. Skipping graph.")
             valid_edges = False; break

        # Ensure edge type is a numpy array for consistent processing
        edge_type_vec = np.asarray(attrs['type'])

        # Check length before using argmax
        if len(edge_type_vec) != num_explicit_edge_types:
            warnings.warn(f"Edge ({u_old}-{v_old}) has 'type' of unexpected length {len(edge_type_vec)} (expected {num_explicit_edge_types}). Skipping graph.")
            valid_edges = False; break

        try:
            # ***** CORRECTED LINE *****
            # Use np.argmax for numpy arrays to find the index of the '1'
            edge_type_index = np.argmax(edge_type_vec).item() # .item() converts numpy int to python int
            # ***** END CORRECTION *****

            # Check if the argmax result makes sense (should be 0 or 1 for one-hot of length 2)
            # This also implicitly checks if the max value was indeed 1.0 (or close to it)
            # If the vector was all zeros, argmax returns 0, which might be misleading.
            # A stricter check could be added if needed: if edge_type_vec[edge_type_index] != 1.0: raise ValueError
            if edge_type_index >= num_explicit_edge_types:
                 warnings.warn(f"Edge ({u_old}-{v_old}) 'type' {edge_type_vec} resulted in unexpected index {edge_type_index}. Skipping graph.")
                 valid_edges = False; break

        except Exception as e: # Catch potential errors during argmax or conversion
             warnings.warn(f"Edge ({u_old}-{v_old}) encountered error processing 'type' {edge_type_vec}: {e}. Skipping graph.")
             valid_edges = False; break

        adj_matrix[edge_type_index, u, v] = 1.0
        adj_matrix[num_explicit_edge_types, u, v] = 0.0

    if not valid_edges: return None

    for k_node_diag in range(max_nodes):
        adj_matrix[num_explicit_edge_types, k_node_diag, k_node_diag] = 0.0

    # --- Create Data Object ---
    data = Data(
        x=x_padded,
        adj=adj_matrix,
        num_atom=torch.tensor(num_nodes_in_graph, dtype=torch.long)
    )
    return data

def process_split(raw_dir, raw_files, output_path):
    """Processes a list of raw PKL files for a specific split and saves the collated PT file."""
    data_list = []
    print(f"\nProcessing split using raw files: {raw_files}")
    if not raw_files:
        print("No raw files specified for this split. Skipping.")
        return

    for raw_path_name in raw_files:
        raw_path = osp.join(raw_dir, raw_path_name)
        print(f" Reading raw file: {raw_path}")
        if not osp.exists(raw_path):
             warnings.warn(f" Raw file not found: {raw_path}. Skipping.")
             continue
        try:
            # Load the whole chunk into memory first
            with open(raw_path, 'rb') as f:
                graphs_chunk = pickle.load(f)
            print(f"  -> Loaded {len(graphs_chunk)} graphs from {raw_path_name}.")
        except Exception as e:
            warnings.warn(f" Could not load {raw_path}: {e}. Skipping chunk.")
            continue

        print(f"  -> Converting graphs...")
        processed_count_in_chunk = 0
        # Process graphs from the loaded chunk
        # Using tqdm here might be helpful for large chunks
        # for i, nx_graph in tqdm(enumerate(graphs_chunk), total=len(graphs_chunk), desc=f"Converting {raw_path_name}"):
        for i, nx_graph in enumerate(graphs_chunk):
            processed_data = _process_graph(nx_graph, MAX_NODES, NUM_NODE_FEATURES, NUM_ADJ_CHANNELS, NUM_EXPLICIT_EDGE_TYPES)
            if processed_data:
                data_list.append(processed_data)
                processed_count_in_chunk += 1

        print(f"  -> Converted {processed_count_in_chunk} valid graphs from this chunk.")
        del graphs_chunk # Free memory from the loaded chunk
        gc.collect()

    if not data_list :
        print(f"Warning: No valid graphs processed for split. Saving a placeholder to {output_path}.")
        dummy_x = torch.zeros((MAX_NODES, NUM_NODE_FEATURES), dtype=torch.float)
        dummy_adj = torch.zeros((NUM_ADJ_CHANNELS, MAX_NODES, MAX_NODES), dtype=torch.float)
        dummy_data = Data(x=dummy_x, adj=dummy_adj, num_atom=torch.tensor(0, dtype=torch.long))
        data_list = [dummy_data]

    print(f"Collating {len(data_list)} graphs for the split...")
    # Use PyG's collate function directly
    print(f"Collating {len(data_list)} graphs for the split...")
    # ***** MODIFIED LINE *****
    # Use the collate method from the InMemoryDataset class itself
    # We need a dummy instance to call this method.
    # Alternatively, ensure the standalone collate function is used correctly,
    # but using the class method might be more robust to version changes.
    try:
        # Create a temporary dummy dataset instance to access the class's collate method
        # This feels slightly hacky but ensures we use the method tied to the expected format
        dummy_dataset = InMemoryDataset(root=None)  # root=None avoids directory creation
        data, slices = dummy_dataset.collate(data_list)
    except Exception as e:
        print(f"Error during collation: {e}")
        print("Failed to collate data. Cannot save processed file.")
        return  # Exit the function if collation fails
    # ***** END MODIFICATION *****

    print(f"Saving processed split data to {output_path}")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    torch.save((data, slices), output_path)
    print(f"Successfully saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AIG PKL files into PyG InMemoryDataset format.")
    parser.add_argument('--raw_dir', type=str, default="/Users/bellavg/aig-gen/data/raw_aigs", help="Directory containing the raw PKL files (e.g., /path/to/raw_aigs).")
    parser.add_argument('--output_root', type=str, default="./", help="Root directory where the processed 'aig/processed/{split}/data.pt' files will be saved.")
    parser.add_argument('--num_train', type=int, default=4, help="Number of PKL files to use for training.")
    parser.add_argument('--num_total', type=int, default=6, help="Total number of PKL files available.")
    parser.add_argument('--name', type=str, default='aig', help="Name of the dataset (used for subdirectories).")

    args = parser.parse_args()

    # --- Determine file splits ---
    if not osp.isdir(args.raw_dir):
        print(f"Error: Raw directory not found: {args.raw_dir}")
        exit()

    all_pkl_files = sorted([f for f in os.listdir(args.raw_dir) if f.startswith("real_aigs_part_") and f.endswith(".pkl")])
    effective_total_files = len(all_pkl_files)
    print(f"Found {effective_total_files} PKL files in {args.raw_dir}.")

    if effective_total_files < args.num_total:
         warnings.warn(f"Expected {args.num_total} total files, but found {effective_total_files}.")

    # Define file lists for each split
    train_files = all_pkl_files[:min(args.num_train, effective_total_files)]
    test_files = []
    if args.num_train < effective_total_files:
        test_files = [all_pkl_files[args.num_train]]
    val_files = []
    if args.num_train + 1 < effective_total_files:
        val_files = [all_pkl_files[args.num_train + 1]]

    # --- Define output paths ---
    train_output_path = osp.join(args.output_root, args.name, 'processed', 'train', 'data.pt')
    val_output_path = osp.join(args.output_root, args.name, 'processed', 'val', 'data.pt')
    test_output_path = osp.join(args.output_root, args.name, 'processed', 'test', 'data.pt')

    # --- Process each split ---
    process_split(args.raw_dir, train_files, train_output_path)
    process_split(args.raw_dir, val_files, val_output_path)
    process_split(args.raw_dir, test_files, test_output_path)

    print("\nProcessing finished.")

