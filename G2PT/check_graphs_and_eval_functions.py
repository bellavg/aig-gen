# TODO check my graphs in my data file against my eval functions should be 100 percent!!!
# check pyg and check bin


# G2PT/validate_input_data.py
import os
import argparse
import torch
import networkx as nx
import numpy as np
import logging
import json
from tqdm import tqdm
import time

# --- Import necessary functions and classes ---
try:
    # Need the dataset class to load .pt files
    from datasets.aig_dataset import AIGPygDataset
except ImportError:
     try: from aig_dataset import AIGPygDataset
     except ImportError: print("Error: Could not find AIGPygDataset."); exit(1)

try:
    # Need the dataset class to load .bin files
    # Ensure this is the version that loads num_inputs/num_outputs
    from datasets_utils import NumpyBinDataset
except ImportError:
     print("Error importing NumpyBinDataset from datasets_utils.")
     print("Make sure datasets_utils.py is in the correct path.")
     exit(1)

try:
    from evaluate_aigs import calculate_structural_aig_metrics, VALID_AIG_NODE_TYPES
except ImportError:
    print("Error: Could not import evaluate_aigs.py. Cannot perform validation.")
    exit(1)

# --- Logger Setup ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
# logger = logging.getLogger("validate_input")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s') # <-- CHANGE HERE
logger = logging.getLogger("validate_input")

# --- Helper Function (pyg_data_to_nx) ---
# Converts PyG Data object (using feature indices) to NX DiGraph
def pyg_data_to_nx(data):
    if not hasattr(data, 'x') or data.x is None: return None
    G = nx.DiGraph(); num_nodes = data.x.size(0)
    # Assumes data.x stores feature indices 0-3 for node types
    node_feature_idx_to_type = {0: 'NODE_CONST0', 1: 'NODE_PI', 2: 'NODE_AND', 3: 'NODE_PO'}

    # --- Node Processing (Remains the same) ---
    if data.x.numel() == 0: return G # Handle empty node features case

    if data.x.dim() > 1 and data.x.shape[1] > 1 and data.x.dtype == torch.float:
        type_indices = torch.argmax(data.x, dim=1).tolist()
    elif data.x.dim() == 1 and data.x.dtype == torch.long:
        type_indices = data.x.tolist()
    else:
        logger.error(f"Unexpected shape/type for data.x: {data.x.shape}, dtype: {data.x.dtype}.")
        return None

    for i in range(num_nodes):
        if i < len(type_indices):
            node_type_idx = int(type_indices[i])
            node_type_str = node_feature_idx_to_type.get(node_type_idx, 'UNKNOWN')
            if node_type_str == 'UNKNOWN': logger.warning(f"Unknown node feature index {node_type_idx}.")
            G.add_node(i, type=node_type_str)
        else:
            logger.error(f"Node index mismatch: {num_nodes} vs {len(type_indices)}")
            G.add_node(i, type='UNKNOWN')
    # --- End Node Processing ---


    # --- Edge Processing MODIFIED ---
    # Check if edge_index and edge_attr exist and are not empty
    if (hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0 and
        hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.numel() > 0):

        edge_index = data.edge_index.cpu() # Keep as tensor
        edge_attr = data.edge_attr.cpu()   # Keep as tensor
        num_edges = edge_index.shape[1]

        # Verify that the number of edges matches the number of attributes
        if num_edges != edge_attr.shape[0]:
            logger.warning(f"PyG Graph: Mismatch between edge_index ({num_edges} edges) and edge_attr ({edge_attr.shape[0]} attrs). Skipping edge processing.")
        else:
            # Define mapping from edge feature index to edge type string
            # Assumes index 0 is INV, index 1 is REG based on aig_pkl_to_pyg.py logic
            edge_feature_idx_to_type = {0: 'EDGE_INV', 1: 'EDGE_REG'}

            # Determine edge types from edge_attr (expecting one-hot vectors)
            try:
                if edge_attr.dim() > 1 and edge_attr.shape[1] > 1:
                    # If edge_attr is one-hot (e.g., [1.0, 0.0] or [0.0, 1.0])
                    edge_type_indices = torch.argmax(edge_attr, dim=1).tolist()
                elif edge_attr.dim() == 1:
                     # If edge_attr already contains indices (0 or 1)
                     edge_type_indices = edge_attr.long().tolist()
                else:
                     logger.warning(f"PyG Graph: Unexpected edge_attr shape {edge_attr.shape}. Cannot determine edge types.")
                     edge_type_indices = [] # Fallback to empty list

            except Exception as e:
                 logger.error(f"PyG Graph: Error processing edge_attr to get indices: {e}")
                 edge_type_indices = [] # Fallback

            if len(edge_type_indices) == num_edges:
                edge_index_np = edge_index.numpy() # Convert index to numpy for iteration
                for i in range(num_edges):
                    u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                    edge_type_idx = edge_type_indices[i]
                    # Map index to string type
                    edge_type_str = edge_feature_idx_to_type.get(edge_type_idx, 'UNKNOWN_EDGE')

                    if edge_type_str == 'UNKNOWN_EDGE':
                        logger.warning(f"PyG Graph: Unknown edge feature index {edge_type_idx} for edge ({u}, {v}).")

                    # Add edge WITH the type attribute
                    if u in G and v in G:
                        G.add_edge(u, v, type=edge_type_str) # Assign type here
                    else:
                        logger.warning(f"Invalid edge index in PyG data: ({u}, {v}) for max node {num_nodes-1}. Skipping edge.")
            else:
                 logger.warning(f"PyG Graph: Could not determine edge type indices correctly. Skipping edge attribute assignment.")
                 # Fallback: Add edges without type if index determination failed
                 edge_index_np = edge_index.numpy()
                 for i in range(num_edges):
                     u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                     if u in G and v in G: G.add_edge(u, v) # No type attribute
                     else: logger.warning(f"Invalid edge index in PyG data: ({u}, {v}). Skipping edge.")

    elif hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
         # Case: Edges exist, but edge_attr is missing or empty
         logger.warning("PyG Graph: edge_index present but edge_attr missing or empty. Adding edges without type attributes.")
         edge_index_np = data.edge_index.cpu().numpy()
         num_edges = edge_index_np.shape[1]
         for i in range(num_edges):
             u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
             if u in G and v in G: G.add_edge(u, v) # No type attribute
             else: logger.warning(f"Invalid edge index in PyG data: ({u}, {v}). Skipping edge.")
    # --- End Edge Processing MODIFIED ---

    return G

# --- Helper Function (bin_data_to_nx) ---
# Converts unpadded data loaded from .bin files (using vocab IDs) to NX DiGraph
def bin_data_to_nx(x_ids, edge_index, edge_attr_ids):
    """
    Args:
        x_ids (torch.Tensor): Tensor of node vocab IDs (e.g., 71-74).
        edge_index (torch.Tensor): Tensor of edge indices [2, num_edges].
        edge_attr_ids (torch.Tensor): Tensor of edge vocab IDs (e.g., 75-76).
    """
    G = nx.DiGraph()
    num_nodes = x_ids.shape[0]
    # Assumes x_ids contains vocab IDs 71-74
    vocab_id_to_type = {71: 'NODE_CONST0', 72: 'NODE_PI', 73: 'NODE_AND', 74: 'NODE_PO'}
    edge_vocab_id_to_type = {75: 'EDGE_INV', 76: 'EDGE_REG'}

    if num_nodes == 0: return G

    for i in range(num_nodes):
        node_id_val = x_ids[i].item()
        node_type_str = vocab_id_to_type.get(node_id_val, 'UNKNOWN_ID')
        if node_type_str == 'UNKNOWN_ID': logger.warning(f"Unknown node vocab ID {node_id_val} from bin data.")
        G.add_node(i, type=node_type_str) # Node indices 0 to N-1

    if edge_index.numel() > 0:
        num_edges = edge_index.shape[1]
        if edge_attr_ids.shape[0] != num_edges:
             logger.warning(f"Bin data edge index/attr mismatch: {num_edges} vs {edge_attr_ids.shape[0]}. Skipping edges.")
        else:
            for i in range(num_edges):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                edge_id_val = edge_attr_ids[i].item()
                edge_type_str = edge_vocab_id_to_type.get(edge_id_val, 'UNKNOWN_EDGE_ID')
                if edge_type_str == 'UNKNOWN_EDGE_ID': logger.warning(f"Unknown edge vocab ID {edge_id_val} from bin data.")

                # Check if node indices from edge_index are valid for the graph G
                if u in G and v in G:
                    G.add_edge(u, v, type=edge_type_str)
                else:
                    logger.warning(f"Invalid edge index in bin data: ({u}, {v}) for max node {num_nodes-1}. Skipping.")
    return G


# --- Main Validation Function ---
def main(args):
    start_time = time.time()

    # --- 1. Validate PyG Data ---
    logger.info(f"--- Phase 1: Validating PyG Data from {args.pyg_data_dir} ---")
    try:
        pyg_dataset = AIGPygDataset(root=args.pyg_data_dir, split=args.split)
    except Exception as e:
        logger.error(f"Failed to load PyG dataset from {args.pyg_data_dir} for split '{args.split}': {e}")
        return

    num_pyg_to_check = min(args.num_graphs, len(pyg_dataset))
    if num_pyg_to_check <= 0: logger.warning(f"No graphs in PyG split '{args.split}'."); pyg_success_rate = 0.0
    else:
        pyg_valid_count = 0
        pyg_processed_count = 0
        for i in tqdm(range(num_pyg_to_check), desc=f"Validating PyG {args.split}"):
            try:
                data = pyg_dataset[i]
                pyg_processed_count += 1
                nx_graph = pyg_data_to_nx(data)
                if nx_graph is None:
                    logger.warning(f"PyG Graph {i}: Failed conversion to NX.")
                    continue

                validity_metrics = calculate_structural_aig_metrics(nx_graph)
                if validity_metrics.get('is_structurally_valid', False):
                    pyg_valid_count += 1
                else:
                    logger.warning(f"Bin Graph {i}: FAILED structural validity.")
                    reasons = validity_metrics.get('constraints_failed', [])
                    print(f"  Bin Graph {i} Failure Reasons: {reasons}")
                    # --- ADD THIS CHECK ---
                    if any("isolated nodes" in reason for reason in reasons):
                        isolates = list(nx.isolates(nx_graph))
                        print(f"    Bin Graph {i} Isolated Node Indices: {isolates}")
                    # --- END ADD ---
                    logger.debug(f"  -> Reasons: {reasons}")

            except Exception as e:
                logger.error(f"Error processing PyG graph index {i}: {e}", exc_info=True)

        pyg_success_rate = (pyg_valid_count / pyg_processed_count) * 100 if pyg_processed_count > 0 else 0
        logger.info(f"PyG Validation Summary: {pyg_valid_count} / {pyg_processed_count} ({pyg_success_rate:.2f}%) passed.")

    # --- 2. Validate Bin Data ---
    logger.info(f"--- Phase 2: Validating Bin Data from {args.bin_data_dir} ---")
    bin_split_path = os.path.join(args.bin_data_dir, args.split)
    meta_path = os.path.join(args.bin_data_dir, 'data_meta.json')

    if not os.path.isdir(bin_split_path) or not os.path.exists(meta_path):
        logger.error(f"Bin data path or meta file not found in {args.bin_data_dir}. Cannot validate bin data.")
        return

    try:
        with open(meta_path, 'r') as f: data_meta = json.load(f)
        split_shape_key = f"{args.split}_shape"
        if split_shape_key not in data_meta:
             raise KeyError(f"Shape information for split '{args.split}' not found in {meta_path}")
        bin_shape = data_meta[split_shape_key]
        # Use dummy tokenizer/process_fn as we manually reconstruct graph
        dummy_tokenizer = None; dummy_process_fn = None
        # Ensure num_inputs/outputs shapes exist in metadata
        if 'num_inputs' not in bin_shape or 'num_outputs' not in bin_shape:
             raise KeyError("Shapes for num_inputs/num_outputs missing from data_meta.json")

        bin_dataset = NumpyBinDataset(
            path=bin_split_path,
            num_data=bin_shape['xs'][0], # Use original number of graphs
            num_node_class=4, num_edge_class=2, # Placeholder values
            shape=bin_shape,
            process_fn=dummy_process_fn,
            tokenizer=dummy_tokenizer,
            num_augmentations=1 # We only access original data
        )
    except Exception as e:
        logger.error(f"Failed to load Bin dataset from {bin_split_path} using meta {meta_path}: {e}")
        return

    num_bin_to_check = min(args.num_graphs, len(bin_dataset.xs)) # Check against actual loaded data size
    if num_bin_to_check <= 0: logger.warning(f"No graphs loaded from bin split '{args.split}'."); bin_success_rate = 0.0
    else:
        bin_valid_count = 0
        bin_processed_count = 0
        for i in tqdm(range(num_bin_to_check), desc=f"Validating Bin {args.split}"):
            try:
                # Manually extract and unpad data for graph i
                raw_x = np.array(bin_dataset.xs[i]).astype(np.int64)
                raw_edge_index = np.array(bin_dataset.edge_indices[i]).astype(np.int64)
                raw_edge_attr = np.array(bin_dataset.edge_attrs[i]).astype(np.int64)

                node_padding_mask = raw_x != -100
                x_ids = torch.from_numpy(raw_x[node_padding_mask])
                num_valid_nodes = len(x_ids)
                if num_valid_nodes == 0: continue # Skip empty graphs

                old_indices = np.arange(len(raw_x)); new_indices_map = -np.ones_like(old_indices)
                new_indices_map[node_padding_mask] = np.arange(num_valid_nodes)

                if raw_edge_attr.ndim > 1: raw_edge_attr = raw_edge_attr.flatten()
                edge_padding_mask = raw_edge_attr != -100
                edge_attr_ids_filtered_by_attr = torch.from_numpy(raw_edge_attr[edge_padding_mask])

                if edge_padding_mask.shape[0] != raw_edge_index.shape[1]:
                    logger.warning(f"Bin Graph {i}: Edge index/attr shape mismatch after unpadding. Skipping edge processing.")
                    edge_index_filtered_by_attr = torch.tensor([[], []], dtype=torch.long)
                    edge_attr_ids_filtered_by_attr = torch.tensor([], dtype=torch.long)
                else:
                    edge_index_filtered_by_attr = torch.from_numpy(raw_edge_index[:, edge_padding_mask])

                if edge_index_filtered_by_attr.numel() > 0:
                    src_nodes_old = edge_index_filtered_by_attr[0, :].numpy(); dst_nodes_old = edge_index_filtered_by_attr[1, :].numpy()
                    src_nodes_new = new_indices_map[src_nodes_old]; dst_nodes_new = new_indices_map[dst_nodes_old]
                    valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)
                    edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask], dst_nodes_new[valid_edge_mask]], dtype=torch.long)
                    edge_attr_final = edge_attr_ids_filtered_by_attr[valid_edge_mask]
                else:
                    edge_index_final = torch.tensor([[], []], dtype=torch.long); edge_attr_final = torch.tensor([], dtype=torch.long)

                bin_processed_count += 1
                # Reconstruct NX graph from the unpadded bin data
                nx_graph = bin_data_to_nx(x_ids, edge_index_final, edge_attr_final)
                if nx_graph is None:
                    logger.warning(f"Bin Graph {i}: Failed conversion to NX.")
                    continue

                validity_metrics = calculate_structural_aig_metrics(nx_graph)
                if validity_metrics.get('is_structurally_valid', False):
                    bin_valid_count += 1
                else:
                    logger.warning(f"Bin Graph {i}: FAILED structural validity.")
                    reasons = validity_metrics.get('constraints_failed', [])
                    print(f"  Bin Graph {i} Failure Reasons: {reasons}")
                    # --- ADD THIS CHECK ---
                    if any("isolated nodes" in reason for reason in reasons):
                        isolates = list(nx.isolates(nx_graph))
                        print(f"    Bin Graph {i} Isolated Node Indices: {isolates}")
                    # --- END ADD ---
                    logger.debug(f"  -> Reasons: {reasons}")

            except Exception as e:
                logger.error(f"Error processing Bin graph index {i}: {e}", exc_info=True)

        bin_success_rate = (bin_valid_count / bin_processed_count) * 100 if bin_processed_count > 0 else 0
        logger.info(f"Bin Validation Summary: {bin_valid_count} / {bin_processed_count} ({bin_success_rate:.2f}%) passed.")

    # --- Final Summary ---
    end_time = time.time()
    print("\n--- Overall Input Data Validation Summary ---")
    print(f"Checked {args.split} split.")
    print(f"Graphs checked per source: {args.num_graphs}")
    print(f"PyG Data Success Rate:     {pyg_success_rate:.2f}%")
    print(f".bin Data Success Rate:    {bin_success_rate:.2f}%")
    print(f"Total Time:                {end_time - start_time:.2f} seconds")
    print("---------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate original PyG and final Bin data against evaluation functions.')
    parser.add_argument('--pyg_data_dir', type=str, default="./datasets/aig",
                        help='Directory containing the AIG PyG data (root with raw/processed subdirs)')
    parser.add_argument('--bin_data_dir', type=str, default="./datasets/aig",
                        help='Directory containing the final .bin data and data_meta.json')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Which data split to check')
    parser.add_argument('--num_graphs', type=int, default=100,
                        help='Max number of graphs to check from the dataset')

    parsed_args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(parsed_args.pyg_data_dir): logger.error(f"PyG data directory not found: {parsed_args.pyg_data_dir}"); exit(1)
    if not os.path.isdir(parsed_args.bin_data_dir): logger.error(f"Bin data directory not found: {parsed_args.bin_data_dir}"); exit(1)
    if not os.path.exists(os.path.join(parsed_args.bin_data_dir, 'data_meta.json')): logger.error(f"data_meta.json not found in {parsed_args.bin_data_dir}"); exit(1)

    main(parsed_args)