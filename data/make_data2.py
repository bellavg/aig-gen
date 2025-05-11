#!/usr/bin/env python3
import os
import os.path as osp
import pickle  # Kept if convert_nx_to_custom_pyg is used elsewhere, not directly by main
import warnings
import torch
import networkx as nx  # Kept if convert_nx_to_custom_pyg is used elsewhere
import numpy as np  # Kept if convert_nx_to_custom_pyg is used elsewhere
from torch_geometric.data import Data  # Kept if convert_nx_to_custom_pyg is used elsewhere
from tqdm import tqdm
import gc

# Attempt to import config, though not strictly needed by the main combiner logic
# It's good practice if convert_nx_to_custom_pyg is kept as a utility
from aig_config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES


def convert_nx_to_custom_pyg(
        nx_graph: nx.DiGraph,
        num_node_features: int,
        NUM_EDGE_FEATURES: int
) -> Data | None:
    """
    Converts a NetworkX DiGraph to a PyTorch Geometric Data object with
    a custom adjacency tensor format (N_src, N_tgt, N_edge_types_plus_no_edge_channel).
    Nodes are ordered topologically. Node features are not padded.
    Returns None if the graph cannot be processed due to data validation issues.
    Will raise an error if topological sort fails (e.g., graph has cycles).
    (This function is kept for utility but not directly called by the main combiner logic below)
    """
    num_nodes_in_graph = nx_graph.number_of_nodes()

    if num_nodes_in_graph == 0:
        warnings.warn("Skipping empty graph.")
        return None

    node_list = list(nx.topological_sort(nx_graph))
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    node_features_list = []
    for old_node_id in node_list:
        attrs = nx_graph.nodes[old_node_id]
        node_type_vec_raw = attrs.get('type')
        if node_type_vec_raw is None or \
                not isinstance(node_type_vec_raw, (list, np.ndarray)) or \
                len(node_type_vec_raw) != num_node_features:
            warnings.warn(
                f"Node {old_node_id} has invalid 'type' attribute. Expected {num_node_features}-len. Skipping."
            )
            return None
        node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
        if not (np.isclose(np.sum(node_type_vec), 1.0) and \
                np.all((np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
            warnings.warn(f"Node {old_node_id} 'type' vector not one-hot. Skipping.")
            return None
        node_features_list.append(torch.tensor(node_type_vec, dtype=torch.float))

    if not node_features_list:
        warnings.warn("Node feature list empty. Skipping.")
        return None
    x_tensor = torch.stack(node_features_list)

    adj_tensor = torch.zeros(
        (num_nodes_in_graph, num_nodes_in_graph, NUM_EDGE_FEATURES + 1),
        dtype=torch.float
    )
    no_edge_channel_idx = NUM_EDGE_FEATURES
    adj_tensor[:, :, no_edge_channel_idx] = 1.0

    for u_old, v_old, edge_attrs in nx_graph.edges(data=True):
        edge_type_vec_raw = edge_attrs.get('type')
        if edge_type_vec_raw is None or \
                not isinstance(edge_type_vec_raw, (list, np.ndarray)) or \
                len(edge_type_vec_raw) != NUM_EDGE_FEATURES:
            warnings.warn(f"Edge ({u_old}-{v_old}) invalid 'type'. Expected {NUM_EDGE_FEATURES}-len. Skipping.")
            return None
        u_new, v_new = node_id_map.get(u_old), node_id_map.get(v_old)
        edge_type_vec = np.asarray(edge_type_vec_raw, dtype=np.float32)
        if not (np.isclose(np.sum(edge_type_vec), 1.0) and \
                np.all((np.isclose(edge_type_vec, 0.0)) | (np.isclose(edge_type_vec, 1.0)))):
            warnings.warn(f"Edge ({u_old}-{v_old}) 'type' not one-hot. Skipping.")
            return None
        edge_channel_index = np.argmax(edge_type_vec).item()
        if not (0 <= edge_channel_index < NUM_EDGE_FEATURES):
            warnings.warn(f"Edge ({u_old}-{v_old}) invalid channel index. Skipping.")
            return None
        adj_tensor[u_new, v_new, edge_channel_index] = 1.0
        adj_tensor[u_new, v_new, no_edge_channel_idx] = 0.0

    pyg_data = Data(x=x_tensor, adj=adj_tensor, num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long))
    if 'inputs' in nx_graph.graph: pyg_data.num_inputs = torch.tensor(nx_graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in nx_graph.graph: pyg_data.num_outputs = torch.tensor(nx_graph.graph['outputs'], dtype=torch.long)
    if 'gates' in nx_graph.graph: pyg_data.num_gates = torch.tensor(nx_graph.graph['gates'], dtype=torch.long)
    return pyg_data


if __name__ == "__main__":
    # --- Configuration ---
    # Base names of the original PKL files, used to construct the expected .pt file names
    # (as if they were processed by the previous version of the script)
    original_pkl_basenames = [
        "real_aigs_part_1_of_6.pkl",
        "real_aigs_part_2_of_6.pkl",
        "real_aigs_part_3_of_6.pkl",
        "real_aigs_part_4_of_6.pkl",
        "real_aigs_part_5_of_6.pkl",
        "real_aigs_part_6_of_6.pkl",
    ]

    # Directory where intermediate .pt files are expected to be, and where the combined file will be saved.
    processed_data_dir = "./raw_pyg"  # Make sure this matches where your .pt files are

    # Suffix used for the intermediate processed files
    intermediate_file_suffix = "_raw_pyg_topo.pt"
    combined_output_filename = "all_raw_graphs.pt"
    # --- End Configuration ---

    os.makedirs(processed_data_dir, exist_ok=True)  # Ensure output directory exists

    print(f"--- Raw PyG Combiner Script ---")
    print(f"Expecting intermediate .pt files in: {osp.abspath(processed_data_dir)}")
    print(f"Combined file will be saved as: {combined_output_filename} in the same directory.\n")

    expected_intermediate_pt_files = []
    all_intermediate_files_exist = True

    for pkl_basename in original_pkl_basenames:
        base, _ = osp.splitext(pkl_basename)  # Get filename without .pkl
        pt_filename = base + intermediate_file_suffix
        pt_filepath = osp.join(processed_data_dir, pt_filename)
        expected_intermediate_pt_files.append(pt_filepath)

        if not osp.exists(pt_filepath):
            all_intermediate_files_exist = False
            warnings.warn(f"MISSING intermediate file: {pt_filepath}")

    if not all_intermediate_files_exist:
        print("\nOne or more required intermediate .pt files are missing. Cannot proceed with combination.")
        print("Please ensure all the following files exist in the target directory:")
        for f_path in expected_intermediate_pt_files:
            print(f"  - {f_path} {'(Found)' if osp.exists(f_path) else '(Missing!)'}")
        exit(1)  # Exit if files are missing

    print("All required intermediate .pt files found. Proceeding with combination.")

    # --- Combine all processed intermediate .pt files ---
    all_pyg_data_objects_combined = []
    total_estimated_bytes = 0

    for pt_file_path in tqdm(expected_intermediate_pt_files, desc="Loading intermediate .pt files for combination",
                             unit="file"):
        intermediate_list = torch.load(pt_file_path)  # Assumes torch.load is safe
        all_pyg_data_objects_combined.extend(intermediate_list)

        # Estimate size of tensor data in newly loaded objects
        for data_obj in intermediate_list:
            if hasattr(data_obj, 'x') and data_obj.x is not None:
                total_estimated_bytes += data_obj.x.nbytes
            if hasattr(data_obj, 'adj') and data_obj.adj is not None:
                total_estimated_bytes += data_obj.adj.nbytes
            if hasattr(data_obj, 'num_nodes') and data_obj.num_nodes is not None:
                total_estimated_bytes += data_obj.num_nodes.nbytes
            # Add other tensor attributes if they exist and you want to count them
            for attr_name in ['num_inputs', 'num_outputs', 'num_gates']:
                if hasattr(data_obj, attr_name) and getattr(data_obj, attr_name) is not None:
                    total_estimated_bytes += getattr(data_obj, attr_name).nbytes
        del intermediate_list
        gc.collect()

    print(f"\nTotal graphs combined from intermediate files: {len(all_pyg_data_objects_combined)}")

    total_estimated_gb = total_estimated_bytes / (1024 ** 3)  # Convert bytes to GB
    print(f"Estimated total tensor data size for combined list: {total_estimated_gb:.2f} GB")

    size_limit_gb = 32.0  # Define the size limit in GB
    # size_limit_bytes = size_limit_gb * (1024**3) # Not strictly needed if comparing GB to GB

    if total_estimated_gb > size_limit_gb:
        warnings.warn(
            f"WARNING: Estimated tensor data size ({total_estimated_gb:.2f} GB) exceeds the {size_limit_gb:.1f}GB limit. "
            "The combined file will NOT be saved to prevent potential issues."
        )
    elif not all_pyg_data_objects_combined:
        print("No data objects were loaded. Nothing to save for the combined file.")
    else:
        combined_output_path = osp.join(processed_data_dir, combined_output_filename)
        print(f"Estimated size is within limits. Saving combined data to: {combined_output_path}")

        torch.save(all_pyg_data_objects_combined, combined_output_path)
        print(f"Successfully saved {len(all_pyg_data_objects_combined)} combined graphs to {combined_output_path}.")

    del all_pyg_data_objects_combined
    gc.collect()

    print("\nProcessing finished.")
