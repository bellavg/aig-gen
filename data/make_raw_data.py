#!/usr/bin/env python3
import os
import os.path as osp
import pickle
import warnings  # Kept for explicit warnings, not for exceptions
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import gc

# Directly import from aig_config.py.
# If this file is missing or variables are not defined, the script will raise an error.
from aig_config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES


def convert_nx_to_custom_pyg(
        nx_graph: nx.DiGraph,
        num_node_features: int,
        num_explicit_edge_types: int
) -> Data | None:  # Updated type hint for clarity
    """
    Converts a NetworkX DiGraph to a PyTorch Geometric Data object with
    a custom adjacency tensor format (N_src, N_tgt, N_edge_types_plus_no_edge_channel).
    Nodes are ordered topologically. Node features are not padded.
    Returns None if the graph cannot be processed due to data validation issues.
    Will raise an error if topological sort fails (e.g., graph has cycles).
    """
    num_nodes_in_graph = nx_graph.number_of_nodes()

    if num_nodes_in_graph == 0:
        warnings.warn("Skipping empty graph.")
        return None

    # Perform topological sort.
    # If the graph has cycles, networkx.topological_sort will raise an error
    # (e.g., NetworkXUnfeasible or NetworkXHasCycle), and the script will halt.
    node_list = list(nx.topological_sort(nx_graph))

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    # --- Node Features ---
    node_features_list = []
    for old_node_id in node_list:  # Iterating in topologically sorted order
        attrs = nx_graph.nodes[old_node_id]
        node_type_vec_raw = attrs.get('type')

        # Validate node 'type' attribute
        if node_type_vec_raw is None or \
                not isinstance(node_type_vec_raw, (list, np.ndarray)) or \
                len(node_type_vec_raw) != num_node_features:
            warnings.warn(
                f"Node {old_node_id} has invalid 'type' attribute (length or existence). "
                f"Expected {num_node_features}-len vector. Got: {node_type_vec_raw}. Skipping graph."
            )
            return None

        node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
        # Check if it's one-hot (sum is 1, all elements are 0 or 1)
        if not (np.isclose(np.sum(node_type_vec), 1.0) and \
                np.all((np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
            warnings.warn(
                f"Node {old_node_id} 'type' vector is not one-hot. Sum: {np.sum(node_type_vec)}. "
                f"Values: {node_type_vec}. Skipping graph."
            )
            return None
        node_features_list.append(torch.tensor(node_type_vec, dtype=torch.float))

    if not node_features_list:  # Should not be reached if num_nodes_in_graph > 0 and checks pass
        warnings.warn("Node feature list is empty despite non-zero nodes. Skipping graph.")
        return None

    x_tensor = torch.stack(node_features_list)  # Shape: (N, num_node_features)

    # --- Adjacency Tensor ---
    # Shape: (N_source, N_target, num_explicit_edge_types + 1 for no-edge channel)
    adj_tensor = torch.zeros(
        (num_nodes_in_graph, num_nodes_in_graph, num_explicit_edge_types + 1),
        dtype=torch.float
    )

    no_edge_channel_idx = num_explicit_edge_types  # Index for the "no edge" channel
    adj_tensor[:, :, no_edge_channel_idx] = 1.0  # Initialize all as "no edge"

    for u_old, v_old, edge_attrs in nx_graph.edges(data=True):
        edge_type_vec_raw = edge_attrs.get('type')

        # Validate edge 'type' attribute
        if edge_type_vec_raw is None or \
                not isinstance(edge_type_vec_raw, (list, np.ndarray)) or \
                len(edge_type_vec_raw) != num_explicit_edge_types:
            warnings.warn(
                f"Edge ({u_old}-{v_old}) has invalid 'type' attribute. "
                f"Expected {num_explicit_edge_types}-len vector. Got: {edge_type_vec_raw}. Skipping graph."
            )
            return None

        u_new, v_new = node_id_map.get(u_old), node_id_map.get(v_old)
        # u_new and v_new should always be found if node_list was derived from nx_graph.nodes()

        edge_type_vec = np.asarray(edge_type_vec_raw, dtype=np.float32)
        if not (np.isclose(np.sum(edge_type_vec), 1.0) and \
                np.all((np.isclose(edge_type_vec, 0.0)) | (np.isclose(edge_type_vec, 1.0)))):
            warnings.warn(
                f"Edge ({u_old}-{v_old}) 'type' vector is not one-hot. "
                f"Sum: {np.sum(edge_type_vec)}. Values: {edge_type_vec}. Skipping graph."
            )
            return None

        edge_channel_index = np.argmax(edge_type_vec).item()
        if not (0 <= edge_channel_index < num_explicit_edge_types):
            warnings.warn(
                f"Edge ({u_old}-{v_old}) 'type' vector resulted in invalid channel index: {edge_channel_index} "
                f"(max is {num_explicit_edge_types - 1}). Skipping graph."
            )
            return None

        # Set the specific edge type channel to 1
        adj_tensor[u_new, v_new, edge_channel_index] = 1.0
        # Set the "no edge" channel to 0 for this pair, as an explicit edge exists
        adj_tensor[u_new, v_new, no_edge_channel_idx] = 0.0

    # Create PyG Data object
    pyg_data = Data(x=x_tensor, adj=adj_tensor, num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long))

    # Add optional graph-level attributes if they exist in the NetworkX graph
    if 'inputs' in nx_graph.graph:
        pyg_data.num_inputs = torch.tensor(nx_graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in nx_graph.graph:
        pyg_data.num_outputs = torch.tensor(nx_graph.graph['outputs'], dtype=torch.long)
    if 'gates' in nx_graph.graph:
        pyg_data.num_gates = torch.tensor(nx_graph.graph['gates'], dtype=torch.long)

    return pyg_data


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace these with the actual paths to your .pkl files
    pkl_file_paths = [
        "../raw_data/networkx_aigs/real_aigs_part_1_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_2_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_3_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_4_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_5_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_6_of_6.pkl",
    ]

    # Define the output directory for the processed .pt files
    output_pyg_dir = "./raw_pyg"
    # --- End Configuration ---

    os.makedirs(output_pyg_dir, exist_ok=True)

    print(f"--- AIG to Raw PyG Conversion (Topological Sort) ---")
    print(f"Using NUM_NODE_FEATURES = {NUM_NODE_FEATURES}")
    print(f"Using NUM_EXPLICIT_EDGE_TYPES = {NUM_EDGE_FEATURES}")
    print(f"Total edge channels in 'adj' tensor will be: {NUM_EDGE_FEATURES + 1}")
    print(f"Output directory: {osp.abspath(output_pyg_dir)}\n")

    total_files_processed = 0
    total_graphs_successfully_converted_across_all_files = 0

    for pkl_file_idx, pkl_file_path in enumerate(pkl_file_paths):
        if not osp.exists(pkl_file_path):
            warnings.warn(f"Pickle file not found: {pkl_file_path}. Skipping.")
            continue

        print(f"Processing PKL file ({pkl_file_idx + 1}/{len(pkl_file_paths)}): {pkl_file_path}")

        # If open or pickle.load fails, the script will raise an error
        # (e.g., FileNotFoundError, pickle.UnpicklingError)
        with open(pkl_file_path, 'rb') as f:
            nx_graphs_in_chunk = pickle.load(f)

        if not isinstance(nx_graphs_in_chunk, list):
            warnings.warn(f"Content of {pkl_file_path} is not a list. Skipping this file.")
            del nx_graphs_in_chunk  # Clean up loaded data
            gc.collect()
            continue

        processed_pyg_data_list = []
        num_original_graphs_in_chunk = len(nx_graphs_in_chunk)
        num_successfully_converted_in_chunk = 0

        # Iterate through graphs in the current chunk
        for nx_graph_idx, nx_graph in enumerate(
                tqdm(nx_graphs_in_chunk, desc=f"  Converting graphs from {osp.basename(pkl_file_path)}", unit="graph")):
            if not isinstance(nx_graph, nx.DiGraph):
                warnings.warn(
                    f"Item at index {nx_graph_idx} in {pkl_file_path} is not a NetworkX DiGraph. Skipping."
                )
                continue

            # Convert the NetworkX graph to a PyG Data object
            # This step will raise an error if topological sort fails (e.g. due to cycles)
            pyg_data_item = convert_nx_to_custom_pyg(
                nx_graph,
                NUM_NODE_FEATURES,
                NUM_EDGE_FEATURES
            )

            if pyg_data_item is not None:
                processed_pyg_data_list.append(pyg_data_item)
                num_successfully_converted_in_chunk += 1

        print(f"  Finished converting graphs from {osp.basename(pkl_file_path)}.")
        print(
            f"  Successfully converted {num_successfully_converted_in_chunk}/{num_original_graphs_in_chunk} graphs in this file.")
        total_graphs_successfully_converted_across_all_files += num_successfully_converted_in_chunk

        # Save the list of processed PyG Data objects for the current file
        if processed_pyg_data_list:
            base_name = osp.basename(pkl_file_path)
            # Create a .pt filename based on the input .pkl filename
            output_pt_filename = osp.splitext(base_name)[0] + "_raw_pyg_topo.pt"
            output_pt_path = osp.join(output_pyg_dir, output_pt_filename)

            # If torch.save fails (e.g. disk full), script will raise an error
            torch.save(processed_pyg_data_list, output_pt_path)
            print(f"  Saved {len(processed_pyg_data_list)} processed PyG Data objects to: {output_pt_path}\n")
        else:
            print(
                f"  No graphs from {osp.basename(pkl_file_path)} were successfully converted or the file was empty.\n")

        total_files_processed += 1
        # Clean up memory before processing the next file
        del nx_graphs_in_chunk
        del processed_pyg_data_list
        gc.collect()

    print(f"--- Processing Summary ---")
    print(f"Total PKL files processed: {total_files_processed}/{len(pkl_file_paths)}")
    print(
        f"Total graphs successfully converted across all files: {total_graphs_successfully_converted_across_all_files}")
    print("Processing finished.")

