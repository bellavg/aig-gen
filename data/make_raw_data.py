#!/usr/bin/env python3
import os
import os.path as osp
import pickle
import warnings
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import gc

# Attempt to import configuration constants
# User has directly imported, assuming aig_config.py is correct
from aig_config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES


def convert_nx_to_pyg_with_edge_index_directed(
        nx_graph: nx.DiGraph,
        num_node_features: int,
        NUM_EDGE_FEATURES: int
) -> Data | None:
    """
    Converts a NetworkX DiGraph to a PyTorch Geometric Data object
    for DIRECTED graphs.
    Includes:
    - x: Node features (unpadded).
    - adj: Custom adjacency tensor (N_src, N_tgt, NUM_EDGE_FEATURES), representing directed edges.
    - edge_index: Standard PyG edge index (2, Num_edges), representing directed edges.
    - edge_attr: Edge attributes for edge_index (Num_edges, NUM_EDGE_FEATURES).
    Nodes are ordered based on their original node IDs (sorted numerically/lexicographically).
    Returns None if the graph cannot be processed or has validation issues.
    """
    num_nodes_in_graph = nx_graph.number_of_nodes()

    if num_nodes_in_graph == 0:
        warnings.warn("Skipping empty graph.")
        return None

    # --- Node Ordering based on Node ID ---
    # MODIFICATION: Sort nodes intrinsically
    try:
        # Attempt to sort, assuming node IDs are sortable (e.g., integers or strings)
        node_list = sorted(list(nx_graph.nodes()))
    except TypeError:
        warnings.warn(
            f"Node IDs are not sortable (e.g. mixed types). Skipping graph. Node IDs: {list(nx_graph.nodes())[:5]}..."
        )
        return None

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    # --- Node Features (x) ---
    node_features_list = []
    for old_node_id in node_list: # Iterate in the new sorted order
        attrs = nx_graph.nodes[old_node_id]
        node_type_vec_raw = attrs.get('type')
        if node_type_vec_raw is None or \
                not isinstance(node_type_vec_raw, (list, np.ndarray)) or \
                len(node_type_vec_raw) != num_node_features:
            warnings.warn(
                f"Node {old_node_id} has invalid 'type' attribute. Expected {num_node_features}-len. Skipping graph."
            )
            return None
        node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
        if not (np.isclose(np.sum(node_type_vec), 1.0) and \
                np.all((np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
            warnings.warn(
                f"Node {old_node_id} 'type' vector not one-hot. Sum: {np.sum(node_type_vec)}. Values: {node_type_vec}. Skipping graph.")
            return None
        node_features_list.append(torch.tensor(node_type_vec, dtype=torch.float))

    if not node_features_list:
        warnings.warn("Node feature list empty despite non-zero nodes. Skipping graph.")
        return None
    x_tensor = torch.stack(node_features_list)

    # --- Custom Adjacency Tensor (adj) for DIRECTED graph ---
    adj_tensor = torch.zeros(
        (num_nodes_in_graph, num_nodes_in_graph, NUM_EDGE_FEATURES),
        dtype=torch.float
    )

    # --- Edge Index and Edge Attributes for DIRECTED graph ---
    edge_index_sources = []
    edge_index_targets = []
    edge_attributes_list = []

    for u_old, v_old, edge_attrs in nx_graph.edges(data=True):
        edge_type_vec_raw = edge_attrs.get('type')
        if edge_type_vec_raw is None or \
                not isinstance(edge_type_vec_raw, (list, np.ndarray)) or \
                len(edge_type_vec_raw) != NUM_EDGE_FEATURES:
            warnings.warn(
                f"Edge ({u_old}-{v_old}) invalid 'type' attribute. Expected {NUM_EDGE_FEATURES}-len. Skipping graph.")
            return None

        u_new, v_new = node_id_map.get(u_old), node_id_map.get(v_old)
        # Check if mapping was successful (node_id might not be in node_list if graph changed during iteration - though unlikely here)
        if u_new is None or v_new is None:
            warnings.warn(f"Edge ({u_old}-{v_old}) contains nodes not in the mapped node list. Skipping graph.")
            return None


        edge_type_vec = np.asarray(edge_type_vec_raw, dtype=np.float32)
        if not (np.isclose(np.sum(edge_type_vec), 1.0) and np.all((np.isclose(edge_type_vec, 0.0)) | (np.isclose(edge_type_vec, 1.0)))):
            warnings.warn(
                f"Edge ({u_old}-{v_old}) 'type' vector not one-hot. Sum: {np.sum(edge_type_vec)}. Values: {edge_type_vec}. Skipping graph.")
            return None

        edge_channel_index = np.argmax(edge_type_vec).item()
        if not (0 <= edge_channel_index < NUM_EDGE_FEATURES):
            warnings.warn(
                f"Edge ({u_old}-{v_old}) 'type' vector resulted in invalid channel index: {edge_channel_index}. Max is {NUM_EDGE_FEATURES - 1}. Skipping graph.")
            return None

        # Populate custom adj tensor (DIRECTED)
        adj_tensor[u_new, v_new, edge_channel_index] = 1.0
        # MODIFICATION: Removed symmetrization: adj_tensor[v_new, u_new, edge_channel_index] = 1.0

        # Populate edge_index and edge_attr lists (DIRECTED)
        current_edge_attr_as_tensor = torch.tensor(edge_type_vec, dtype=torch.float)

        edge_index_sources.append(u_new)
        edge_index_targets.append(v_new)
        edge_attributes_list.append(current_edge_attr_as_tensor)

        # MODIFICATION: Removed adding the reverse edge for undirected representation

    edge_index_tensor = torch.tensor([edge_index_sources, edge_index_targets], dtype=torch.long)

    if edge_attributes_list:
        edge_attr_tensor = torch.stack(edge_attributes_list)
    else:
        edge_attr_tensor = torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float)

    pyg_data = Data(
        x=x_tensor,
        adj=adj_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long)
    )

    if 'inputs' in nx_graph.graph: pyg_data.num_inputs = torch.tensor(nx_graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in nx_graph.graph: pyg_data.num_outputs = torch.tensor(nx_graph.graph['outputs'], dtype=torch.long)
    if 'gates' in nx_graph.graph: pyg_data.num_gates = torch.tensor(nx_graph.graph['gates'], dtype=torch.long)

    return pyg_data


if __name__ == "__main__":
    # --- Configuration ---
    pkl_file_paths = [
        "../raw_data/networkx_aigs/real_aigs_part_1_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_2_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_3_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_4_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_5_of_6.pkl",
        "../raw_data/networkx_aigs/real_aigs_part_6_of_6.pkl",
    ]

    # MODIFICATION: Changed filenames to indicate "directed"
    output_pyg_dir = "./data/pyg_full_directed" # MODIFIED
    intermediate_file_suffix = "_pyg_full_directed.pt" # MODIFIED
    combined_output_filename = "aig_directed.pt" # MODIFIED
    # --- End Configuration ---

    os.makedirs(output_pyg_dir, exist_ok=True)

    print(f"--- AIG to Full PyG Conversion (DIRECTED GRAPHS) ---") # MODIFIED
    print(f"Using NUM_NODE_FEATURES = {NUM_NODE_FEATURES}")
    print(f"Using NUM_EDGE_FEATURES = {NUM_EDGE_FEATURES} (for edge_attr and adj channels)")
    print(f"Output directory: {osp.abspath(output_pyg_dir)}\n")

    total_files_processed = 0
    total_graphs_successfully_converted_across_all_files = 0
    successfully_saved_intermediate_pt_files = []

    for pkl_file_idx, pkl_file_path in enumerate(pkl_file_paths):
        if not osp.exists(pkl_file_path):
            warnings.warn(f"Pickle file not found: {pkl_file_path}. Skipping.")
            continue

        print(f"Processing PKL file ({pkl_file_idx + 1}/{len(pkl_file_paths)}): {pkl_file_path}")

        try:
            with open(pkl_file_path, 'rb') as f:
                # Ensure compatibility with files pickled with different protocol versions
                nx_graphs_in_chunk = pickle.load(f, encoding='latin1') # Added encoding for broader compatibility
        except Exception as e:
            warnings.warn(f"Could not load pickle file {pkl_file_path}: {e}. Skipping.")
            continue


        if not isinstance(nx_graphs_in_chunk, list):
            warnings.warn(f"Content of {pkl_file_path} is not a list. Skipping this file.")
            if 'nx_graphs_in_chunk' in locals(): del nx_graphs_in_chunk
            gc.collect()
            continue

        processed_pyg_data_list = []
        num_original_graphs_in_chunk = len(nx_graphs_in_chunk)
        num_successfully_converted_in_chunk = 0

        for nx_graph_idx, nx_graph in enumerate(
                tqdm(nx_graphs_in_chunk, desc=f"  Converting graphs from {osp.basename(pkl_file_path)}", unit="graph")):
            if not isinstance(nx_graph, nx.DiGraph):
                warnings.warn(f"Item at index {nx_graph_idx} in {pkl_file_path} is not a NetworkX DiGraph. Skipping.")
                continue

            # MODIFICATION: Call the new directed conversion function
            pyg_data_item = convert_nx_to_pyg_with_edge_index_directed(
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

        if processed_pyg_data_list:
            base_name = osp.basename(pkl_file_path)
            output_pt_filename = osp.splitext(base_name)[0] + intermediate_file_suffix
            output_pt_path = osp.join(output_pyg_dir, output_pt_filename)

            try:
                torch.save(processed_pyg_data_list, output_pt_path)
                print(f"  Saved {len(processed_pyg_data_list)} processed PyG Data objects to: {output_pt_path}\n")
                successfully_saved_intermediate_pt_files.append(output_pt_path)
            except Exception as e:
                warnings.warn(f"Could not save intermediate file {output_pt_path}: {e}")

        else:
            print(
                f"  No graphs from {osp.basename(pkl_file_path)} were successfully converted or the file was empty.\n")

        total_files_processed += 1
        del nx_graphs_in_chunk # Ensure memory is freed
        del processed_pyg_data_list # Ensure memory is freed
        gc.collect()

    print(f"\n--- Intermediate Processing Summary ---")
    print(f"Total PKL files processed: {total_files_processed}/{len(pkl_file_paths)}")
    print(f"Total graphs successfully converted: {total_graphs_successfully_converted_across_all_files}")
    print(f"Number of intermediate .pt files created: {len(successfully_saved_intermediate_pt_files)}")

    if successfully_saved_intermediate_pt_files:
        print(f"\n--- Combining All Processed Intermediate .pt Files ---")
        all_pyg_data_objects_combined = []
        total_estimated_bytes = 0

        for pt_file_path in tqdm(successfully_saved_intermediate_pt_files, desc="Loading intermediate .pt files",
                                 unit="file"):
            try:
                intermediate_list = torch.load(pt_file_path, weights_only=False) # Preserved weights_only=False
                all_pyg_data_objects_combined.extend(intermediate_list)

                for data_obj in intermediate_list:
                    if hasattr(data_obj, 'x') and data_obj.x is not None: total_estimated_bytes += data_obj.x.nbytes
                    if hasattr(data_obj, 'adj') and data_obj.adj is not None: total_estimated_bytes += data_obj.adj.nbytes
                    if hasattr(data_obj, 'edge_index') and data_obj.edge_index is not None: total_estimated_bytes += data_obj.edge_index.nbytes
                    if hasattr(data_obj, 'edge_attr') and data_obj.edge_attr is not None: total_estimated_bytes += data_obj.edge_attr.nbytes
                    if hasattr(data_obj, 'num_nodes') and data_obj.num_nodes is not None: total_estimated_bytes += data_obj.num_nodes.nbytes
                    for attr_name in ['num_inputs', 'num_outputs', 'num_gates']:
                        if hasattr(data_obj, attr_name) and getattr(data_obj, attr_name) is not None:
                            total_estimated_bytes += getattr(data_obj, attr_name).nbytes
                del intermediate_list # Ensure memory is freed
                gc.collect()
            except Exception as e:
                warnings.warn(f"Could not load or process intermediate file {pt_file_path}: {e}. Skipping this file for combination.")


        print(f"Total graphs combined: {len(all_pyg_data_objects_combined)}")
        total_estimated_gb = total_estimated_bytes / (1024 ** 3)
        print(f"Estimated total tensor data size for combined list: {total_estimated_gb:.2f} GB")

        size_limit_gb = 32.0

        if total_estimated_gb > size_limit_gb:
            warnings.warn(
                f"WARNING: Estimated size ({total_estimated_gb:.2f} GB) exceeds {size_limit_gb:.1f}GB limit. Combined file NOT saved."
            )
        elif not all_pyg_data_objects_combined:
            print("No data objects loaded/processed successfully. Nothing to save for the combined file.")
        else:
            combined_output_path = osp.join(output_pyg_dir, combined_output_filename)
            print(f"Estimated size within limits. Saving combined data to: {combined_output_path}")
            try:
                torch.save(all_pyg_data_objects_combined, combined_output_path)
                print(f"Successfully saved {len(all_pyg_data_objects_combined)} combined graphs to {combined_output_path}.")
            except Exception as e:
                warnings.warn(f"Could not save combined file {combined_output_path}: {e}")


        del all_pyg_data_objects_combined # Ensure memory is freed
        gc.collect()
    elif total_files_processed > 0:
        print("\nNo intermediate .pt files were successfully created or loaded. No combined file generated.")
    else:
        print("\nNo PKL files processed. No combined file generated.")

    print("\nProcessing finished.")