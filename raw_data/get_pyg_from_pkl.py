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
from tqdm import tqdm # Added for progress bar

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
    with padding for dense processing. Adjacency matrix follows PyG convention
    [Channel, Target, Source].

    Returns None if the graph is invalid or cannot be processed.
    """
    num_nodes_in_graph = graph.number_of_nodes()

    # Basic checks
    if num_nodes_in_graph == 0:
        # warnings.warn("Skipping graph with 0 nodes.") # Can be noisy
        return None
    if num_nodes_in_graph > max_nodes:
        warnings.warn(f"Skipping graph with {num_nodes_in_graph} nodes (max allowed: {max_nodes}).")
        return None

    # --- Node ID Mapping (Ensure consistent order if needed) ---
    # Sorting nodes ensures a consistent mapping order, though topological sort
    # during training is the primary mechanism for sequence consistency.
    try:
        # Attempt to sort nodes numerically if they are integers/sortable
        node_list = sorted(list(graph.nodes()))
    except TypeError:
        # Fallback if nodes are not sortable (e.g., mixed types)
        node_list = list(graph.nodes())
        warnings.warn("Graph nodes were not sortable, using arbitrary order for mapping.")

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    # --- Node Features ---
    node_features_list = []
    valid_nodes = True
    for old_node_id in node_list: # Iterate in the mapped order
        attrs = graph.nodes[old_node_id]
        # Check if 'type' attribute exists and is a list/array of expected length
        node_type_vec_raw = attrs.get('type') # Use .get for safety
        if node_type_vec_raw is None or not isinstance(node_type_vec_raw, (list, np.ndarray)) or len(node_type_vec_raw) < num_node_features:
            warnings.warn(f"Node {old_node_id} missing 'type' or 'type' attribute is invalid/short. Skipping graph.")
            valid_nodes = False; break
        try:
            # Ensure it's a numpy array before converting to tensor
            node_type_vec = np.asarray(node_type_vec_raw[:num_node_features])
            # Check if it's a valid one-hot encoding (exactly one '1')
            if not (np.sum(node_type_vec) == 1 and np.all((node_type_vec == 0) | (node_type_vec == 1))):
                 warnings.warn(f"Node {old_node_id} 'type' attribute {node_type_vec_raw} is not a valid one-hot vector. Skipping graph.")
                 valid_nodes = False; break
            node_feature = torch.tensor(node_type_vec, dtype=torch.float)
            node_features_list.append(node_feature)
        except Exception as e:
            warnings.warn(f"Error processing 'type' for node {old_node_id}: {e}. Skipping graph.")
            valid_nodes = False; break
    if not valid_nodes: return None

    # Stack and Pad Node Features
    if not node_features_list: return None # Should not happen if valid_nodes is True
    x_stacked = torch.stack(node_features_list)
    pad_size = max_nodes - num_nodes_in_graph
    # This check should be redundant now due to the initial check, but keep for safety
    if pad_size < 0: return None
    x_padded = F.pad(x_stacked, (0, 0, 0, pad_size), "constant", 0.0) # Pad with float 0.0

    # --- Adjacency Matrix ---
    # Initialize with zeros, shape [Channel, Target, Source]
    adj_matrix = torch.zeros((num_adj_channels, max_nodes, max_nodes), dtype=torch.float)
    # Initialize the last channel (NO_EDGE) to 1 everywhere
    adj_matrix[num_explicit_edge_types, :, :] = 1.0

    valid_edges = True
    for u_old, v_old, attrs in graph.edges(data=True):
        edge_type_vec_raw = attrs.get('type') # Use .get for safety

        # Check edge type attribute
        if edge_type_vec_raw is None or not isinstance(edge_type_vec_raw, (list, np.ndarray)) or len(edge_type_vec_raw) < num_explicit_edge_types:
             warnings.warn(f"Edge ({u_old}-{v_old}) missing 'type' or 'type' attribute is invalid/short. Skipping graph.")
             valid_edges = False; break

        # Map old node IDs to new sequential IDs
        u = node_id_map.get(u_old) # Source index
        v = node_id_map.get(v_old) # Target index
        if u is None or v is None:
             warnings.warn(f"Edge ({u_old}-{v_old}) involves node not in map. Skipping graph.")
             valid_edges = False; break

        try:
            # Ensure edge type is a numpy array
            edge_type_vec = np.asarray(edge_type_vec_raw[:num_explicit_edge_types])
            # Check if it's a valid one-hot encoding for the explicit types
            if not (len(edge_type_vec) == num_explicit_edge_types and np.sum(edge_type_vec) == 1 and np.all((edge_type_vec == 0) | (edge_type_vec == 1))):
                 warnings.warn(f"Edge ({u_old}-{v_old}) 'type' attribute {edge_type_vec_raw} is not a valid one-hot vector for {num_explicit_edge_types} explicit types. Skipping graph.")
                 valid_edges = False; break

            # Get the index (0 or 1) corresponding to REG or INV
            edge_type_index = np.argmax(edge_type_vec).item()

            # --- *** THE FIX *** ---
            # Assign 1.0 at [channel, target_index, source_index]
            adj_matrix[edge_type_index, v, u] = 1.0
            # --- *** END FIX *** ---

            # Mark this entry as NOT being "no-edge"
            adj_matrix[num_explicit_edge_types, v, u] = 0.0

        except Exception as e: # Catch potential errors during numpy conversion or argmax
             warnings.warn(f"Error processing 'type' for edge ({u_old}-{v_old}): {e}. Skipping graph.")
             valid_edges = False; break

    if not valid_edges: return None

    # Ensure diagonal of the NO_EDGE channel is 0 (no self-loops in NO_EDGE)
    # Nodes are connected to themselves only via explicit edge types if needed (unlikely for AIG)
    # Or potentially keep as 1 if self-connection in NO_EDGE channel is desired representation?
    # Setting to 0 aligns with original GraphDF code's adj += np.eye(N) applied *before* setting NO_EDGE channel.
    # Let's set diagonal to 0 for clarity, assuming no self-loops in the NO_EDGE sense.
    for k_node_diag in range(max_nodes):
        adj_matrix[num_explicit_edge_types, k_node_diag, k_node_diag] = 0.0

    # --- Create Data Object ---
    data = Data(
        x=x_padded,
        adj=adj_matrix,
        num_atom=torch.tensor(num_nodes_in_graph, dtype=torch.long) # Store actual node count
    )
    return data

def process_split(raw_dir, raw_files, output_path):
    """Processes a list of raw PKL files for a specific split and saves the collated PT file."""
    data_list = []
    print(f"\nProcessing split using raw files: {raw_files}")
    if not raw_files:
        print("No raw files specified for this split. Skipping.")
        return

    total_graphs_in_files = 0
    graphs_processed_successfully = 0

    for raw_path_name in raw_files:
        raw_path = osp.join(raw_dir, raw_path_name)
        print(f" Reading raw file: {raw_path}")
        if not osp.exists(raw_path):
             warnings.warn(f" Raw file not found: {raw_path}. Skipping.")
             continue
        try:
            with open(raw_path, 'rb') as f:
                # Load graphs carefully, handle potential large files
                graphs_chunk = pickle.load(f)
            if not isinstance(graphs_chunk, list):
                 warnings.warn(f"File {raw_path} does not contain a list. Skipping.")
                 continue
            num_in_chunk = len(graphs_chunk)
            total_graphs_in_files += num_in_chunk
            print(f"  -> Loaded {num_in_chunk} graphs from {raw_path_name}.")
        except Exception as e:
            warnings.warn(f" Could not load {raw_path}: {e}. Skipping chunk.")
            continue

        print(f"  -> Converting graphs...")
        processed_count_in_chunk = 0
        # Use tqdm for progress within a chunk
        for nx_graph in tqdm(graphs_chunk, desc=f"  Converting {raw_path_name}", leave=False):
            processed_data = _process_graph(nx_graph, MAX_NODES, NUM_NODE_FEATURES, NUM_ADJ_CHANNELS, NUM_EXPLICIT_EDGE_TYPES)
            if processed_data:
                data_list.append(processed_data)
                processed_count_in_chunk += 1

        graphs_processed_successfully += processed_count_in_chunk
        print(f"  -> Converted {processed_count_in_chunk}/{num_in_chunk} valid graphs from this chunk.")
        del graphs_chunk # Free memory
        gc.collect()

    print(f"\nFinished reading files. Total graphs loaded: {total_graphs_in_files}. Total successfully converted: {graphs_processed_successfully}.")

    if not data_list :
        print(f"Warning: No valid graphs processed for split '{osp.basename(osp.dirname(output_path))}'. Cannot save data.pt.")
        # Optionally save an empty placeholder if downstream code requires the file
        # placeholder_path = output_path
        # os.makedirs(osp.dirname(placeholder_path), exist_ok=True)
        # torch.save((Data(), {}), placeholder_path) # Save empty Data and slices
        # print(f"Saved empty placeholder to {placeholder_path}")
        return # Stop if no data

    print(f"Collating {len(data_list)} graphs for the split...")
    try:
        # Use PyG's collate function
        # Create a dummy dataset instance to access the class's collate method
        dummy_dataset = InMemoryDataset(root=None)
        data, slices = dummy_dataset.collate(data_list)
    except Exception as e:
        print(f"Error during collation: {e}")
        print("Failed to collate data. Cannot save processed file.")
        return

    print(f"Saving processed split data to {output_path}")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    # Save the collated data and slices tuple
    torch.save((data, slices), output_path)
    print(f"Successfully saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AIG PKL files into PyG InMemoryDataset format.")
    parser.add_argument('--raw_dir', type=str, default="./raw_aigs", help="Directory containing the raw PKL files (e.g., /path/to/raw_aigs).")
    parser.add_argument('--output_root', type=str, default="./", help="Root directory where the processed 'aig/processed/{split}/data.pt' files will be saved.")
    parser.add_argument('--num_train_files', type=int, default=4, help="Number of PKL files to use for the training set.")
    parser.add_argument('--num_val_files', type=int, default=1, help="Number of PKL files to use for the validation set.")
    parser.add_argument('--num_test_files', type=int, default=1, help="Number of PKL files to use for the test set.")
    parser.add_argument('--file_prefix', type=str, default="real_aigs_part_", help="Prefix of the raw PKL files.")
    parser.add_argument('--dataset_name', type=str, default='aig', help="Name of the dataset (used for subdirectories).")

    args = parser.parse_args()

    # --- Determine file splits ---
    if not osp.isdir(args.raw_dir):
        print(f"Error: Raw directory not found: {args.raw_dir}")
        exit(1)

    # Find and sort all relevant files
    try:
        all_pkl_files = sorted([
            f for f in os.listdir(args.raw_dir)
            if f.startswith(args.file_prefix) and f.endswith(".pkl")
        ])
    except OSError as e:
        print(f"Error listing files in directory {args.raw_dir}: {e}")
        exit(1)

    effective_total_files = len(all_pkl_files)
    print(f"Found {effective_total_files} PKL files matching prefix '{args.file_prefix}' in {args.raw_dir}.")

    # Allocate files to splits sequentially
    num_train = min(args.num_train_files, effective_total_files)
    num_val = min(args.num_val_files, max(0, effective_total_files - num_train))
    num_test = min(args.num_test_files, max(0, effective_total_files - num_train - num_val))

    train_files = all_pkl_files[:num_train]
    val_files = all_pkl_files[num_train : num_train + num_val]
    test_files = all_pkl_files[num_train + num_val : num_train + num_val + num_test]

    print(f"Allocating files: Train={len(train_files)}, Validation={len(val_files)}, Test={len(test_files)}")
    if len(train_files) + len(val_files) + len(test_files) < effective_total_files:
        warnings.warn("Some PKL files were not allocated to any split.")

    # --- Define output paths ---
    train_output_path = osp.join(args.output_root, args.dataset_name, 'processed', 'train', 'data.pt')
    val_output_path = osp.join(args.output_root, args.dataset_name, 'processed', 'val', 'data.pt')
    test_output_path = osp.join(args.output_root, args.dataset_name, 'processed', 'test', 'data.pt')

    # --- Process each split ---
    process_split(args.raw_dir, train_files, train_output_path)
    process_split(args.raw_dir, val_files, val_output_path)
    process_split(args.raw_dir, test_files, test_output_path)

    print("\nProcessing finished.")
