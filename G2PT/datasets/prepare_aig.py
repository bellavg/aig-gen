# File: G2PT/datasets/prepare_aig.py

import os
import pickle
import json
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import networkx as nx
import warnings

# --- Configuration ---
# Adjust these paths and settings as needed
CFG = SimpleNamespace(
    dataset=SimpleNamespace(
        # Path to your input pickle file containing NetworkX DiGraphs
        pickle_path='current_data.pkl',  # Make sure this path is correct
        # Output directory for processed data (relative to where script is run)
        output_dir='./aig/',
        # Train/Val/Test split ratios
        split_ratios=(0.8, 0.1, 0.1)  # Includes test split
    ),
    general=SimpleNamespace(
        # Padding value used in G2PT examples
        padding_value=-100,
        # --- Expected vocab IDs (THESE MUST MATCH YOUR vocab.json) ---
        node_const0_id=97,
        node_pi_id=98,
        node_and_id=99,
        node_po_id=100,
        edge_inv_id=101,
        edge_reg_id=102,
        # --- Corresponding one-hot encodings from your data ---
        node_const0_enc=tuple([1, 0, 0, 0]),
        node_pi_enc=tuple([0, 1, 0, 0]),
        node_and_enc=tuple([0, 0, 1, 0]),
        node_po_enc=tuple([0, 0, 0, 1]),
        edge_inv_enc=tuple([1, 0]),
        edge_reg_enc=tuple([0, 1]),
    )
)

# --- Vocabulary Mapping (using IDs from CFG) ---
# Maps the tuple representation of one-hot encodings to the correct integer IDs
NODE_TYPE_MAP = {
    CFG.general.node_const0_enc: CFG.general.node_const0_id,
    CFG.general.node_pi_enc: CFG.general.node_pi_id,
    CFG.general.node_and_enc: CFG.general.node_and_id,
    CFG.general.node_po_enc: CFG.general.node_po_id,
}

EDGE_TYPE_MAP = {
    CFG.general.edge_inv_enc: CFG.general.edge_inv_id,
    CFG.general.edge_reg_enc: CFG.general.edge_reg_id,
}


# --- Main Processing Logic ---
def prepare_aig_dataset():
    print(f"Starting AIG dataset preparation...")
    print(f"Loading AIG dataset from: {CFG.dataset.pickle_path}")
    try:
        with open(CFG.dataset.pickle_path, 'rb') as f:
            all_graphs = pickle.load(f)
        print(f"Loaded {len(all_graphs)} graphs.")
    except Exception as e:
        print(f"Error loading pickle file '{CFG.dataset.pickle_path}': {e}")
        return

    if not isinstance(all_graphs, list):
        print(f"Error: Expected a list of graphs from pickle file, got {type(all_graphs)}")
        return

    # Filter out non-DiGraph items just in case
    num_original = len(all_graphs)
    all_graphs = [g for g in all_graphs if isinstance(g, nx.DiGraph)]
    if len(all_graphs) < num_original:
        print(f"Warning: Filtered out {num_original - len(all_graphs)} non-DiGraph items.")
    if not all_graphs:
        print("Error: No valid DiGraphs found in the pickle file.")
        return

    # --- Shuffle and Split Data ---
    print("Shuffling and splitting graphs...")
    np.random.shuffle(all_graphs)
    num_graphs = len(all_graphs)
    num_train = int(num_graphs * CFG.dataset.split_ratios[0])
    num_val = int(num_graphs * CFG.dataset.split_ratios[1])
    num_test = num_graphs - num_train - num_val  # Remainder goes to test

    # Ensure splits are valid
    if num_train <= 0 or num_val <= 0 or num_test <= 0:
        print(f"Warning: Dataset size ({num_graphs}) is too small for 80/10/10 split. Adjusting.")
        if num_graphs >= 3:
            num_val = max(1, int(num_graphs * CFG.dataset.split_ratios[1]))
            num_test = max(1, int(num_graphs * CFG.dataset.split_ratios[2]))
            num_train = num_graphs - num_val - num_test
            if num_train <= 0:  # Handle edge case where val/test take everything
                num_train = 1
                num_val = max(1, num_graphs - num_train - num_test)
                if num_val <= 0: num_val = 1
                num_test = num_graphs - num_train - num_val
        elif num_graphs == 2:
            num_train, num_val, num_test = 1, 1, 0
        elif num_graphs == 1:
            num_train, num_val, num_test = 1, 0, 0
        else:  # num_graphs == 0
            print("Error: No graphs to process after filtering.")
            return

    dataset_splits = {
        'train': all_graphs[:num_train],
        'eval': all_graphs[num_train:num_train + num_val],
        'test': all_graphs[num_train + num_val:]
    }
    print(
        f"Final Split - Train: {len(dataset_splits['train'])}, Val: {len(dataset_splits['eval'])}, Test: {len(dataset_splits['test'])}")

    data_meta = {}

    # --- Process Each Split ---
    for split_name, graph_list in dataset_splits.items():
        if not graph_list:  # Skip empty splits
            print(f"Skipping empty split: {split_name}")
            continue

        print(f"\nProcessing split: {split_name}")
        output_split_dir = os.path.join(CFG.dataset.output_dir, split_name)
        os.makedirs(output_split_dir, exist_ok=True)

        all_xs = []
        all_edge_indices = []  # Will store tensors of shape [2, num_edges]
        all_edge_attrs = []

        skipped_graphs = 0
        for graph in tqdm(graph_list, desc=f"  Processing {split_name} graphs"):
            valid_graph = True

            # --- 1. Node Processing ---
            node_list = list(graph.nodes())
            if not node_list:  # Skip empty graphs
                skipped_graphs += 1
                continue
            # Create mapping from original node ID to 0-based index
            node_map = {node_id: i for i, node_id in enumerate(node_list)}

            xs_for_graph = []
            for node_id in node_list:
                try:
                    # Ensure 'type' attribute exists
                    if 'type' not in graph.nodes[node_id]:
                        raise KeyError(f"Node {node_id} missing 'type' attribute.")

                    node_type_one_hot = tuple(graph.nodes[node_id]['type'])
                    node_type_id = NODE_TYPE_MAP.get(node_type_one_hot)  # Map to vocab ID
                    if node_type_id is None:
                        warnings.warn(f"Graph skipped: Unknown node type {node_type_one_hot} for node {node_id}.")
                        valid_graph = False
                        break
                    xs_for_graph.append(node_type_id)
                except Exception as e:
                    warnings.warn(f"Graph skipped: Error processing node {node_id}: {e}")
                    valid_graph = False
                    break

            if not valid_graph:
                skipped_graphs += 1
                continue

            # --- 2. Edge Processing ---
            edge_indices_for_graph = []  # List of [src_idx, dst_idx] pairs
            edge_attrs_for_graph = []  # List of edge type IDs
            for u, v, edge_data in graph.edges(data=True):
                try:
                    # Map original NetworkX node IDs to 0-based indices
                    src_idx = node_map[u]
                    dst_idx = node_map[v]

                    # Default to REG edge type if 'type' attribute is missing
                    edge_type_enc = tuple(edge_data.get('type', CFG.general.edge_reg_enc))
                    edge_type_id = EDGE_TYPE_MAP.get(edge_type_enc)  # Map to vocab ID

                    if edge_type_id is None:
                        warnings.warn(
                            f"Graph skipped: Unknown edge type {edge_type_enc} for edge ({u},{v}). Defaulting to REG.")
                        # Default to REG ID if type is unknown, or skip graph if preferred
                        edge_type_id = CFG.general.edge_reg_id  # Or set valid_graph=False and break

                    edge_indices_for_graph.append([src_idx, dst_idx])
                    edge_attrs_for_graph.append(edge_type_id)

                except KeyError as e:
                    warnings.warn(f"Graph skipped: Node {e} in edge ({u},{v}) not found in node_map.")
                    valid_graph = False
                    break
                except Exception as e:
                    warnings.warn(f"Graph skipped: Error processing edge ({u},{v}): {e}")
                    valid_graph = False
                    break

            if not valid_graph:
                skipped_graphs += 1
                continue

            # Append tensors for the valid graph
            all_xs.append(torch.tensor(xs_for_graph, dtype=torch.long))
            all_edge_attrs.append(torch.tensor(edge_attrs_for_graph, dtype=torch.long))

            # Convert edge list to tensor [2, num_edges]
            if edge_indices_for_graph:
                # Shape [num_edges, 2] -> transpose to [2, num_edges]
                edge_indices_tensor = torch.tensor(edge_indices_for_graph, dtype=torch.long).t().contiguous()
                all_edge_indices.append(edge_indices_tensor)
            else:
                all_edge_indices.append(torch.empty((2, 0), dtype=torch.long))

        if skipped_graphs > 0:
            print(f"  Skipped {skipped_graphs} invalid or empty graphs in {split_name} split.")
        if not all_xs:
            print(f"Error: No valid graphs processed for {split_name} split. Cannot save .bin files.")
            continue

        # --- 3. Padding ---
        print(f"  Padding {split_name} data...")
        xs_padded = pad_sequence(all_xs, batch_first=True, padding_value=float(CFG.general.padding_value))
        xs_np = xs_padded.numpy().astype(np.int16)

        edge_attrs_padded = pad_sequence(all_edge_attrs, batch_first=True,
                                         padding_value=float(CFG.general.padding_value))
        edge_attrs_np = edge_attrs_padded.numpy().astype(np.int16)

        # Pad the list of [2, num_edges] tensors. Transpose each to [num_edges, 2] before padding.
        edge_indices_to_pad = [ei.t() for ei in all_edge_indices]
        edge_indices_padded = pad_sequence(edge_indices_to_pad, batch_first=True,
                                           padding_value=float(CFG.general.padding_value))  # Pads to [N, max_edges, 2]
        # Transpose final array to [N, 2, max_edges]
        edge_indices_np = edge_indices_padded.numpy().astype(np.int16).transpose(0, 2, 1)

        print(
            f"    Final shapes - xs: {xs_np.shape}, edge_indices: {edge_indices_np.shape}, edge_attrs: {edge_attrs_np.shape}")

        # --- 4. Saving with Memmap ---
        print(f"  Saving {split_name} data using memmap...")
        xs_path = os.path.join(output_split_dir, 'xs.bin')
        edge_indices_path = os.path.join(output_split_dir, 'edge_indices.bin')
        edge_attrs_path = os.path.join(output_split_dir, 'edge_attrs.bin')

        # Save arrays
        xs_data = np.memmap(xs_path, dtype=np.int16, mode='w+', shape=xs_np.shape)
        xs_data[:] = xs_np
        xs_data.flush()

        edge_indices_data = np.memmap(edge_indices_path, dtype=np.int16, mode='w+', shape=edge_indices_np.shape)
        edge_indices_data[:] = edge_indices_np
        edge_indices_data.flush()

        edge_attrs_data = np.memmap(edge_attrs_path, dtype=np.int16, mode='w+', shape=edge_attrs_np.shape)
        edge_attrs_data[:] = edge_attrs_np
        edge_attrs_data.flush()
        print(f"    Saved data to {output_split_dir}")

        # Store shapes in metadata (convert numpy shapes to lists for JSON)
        data_meta[f'{split_name}_shape'] = {
            'xs': list(xs_np.shape),
            'edge_indices': list(edge_indices_np.shape),
            'edge_attrs': list(edge_attrs_np.shape)
        }

    # --- Save Metadata ---
    meta_path = os.path.join(CFG.dataset.output_dir, 'data_meta.json')
    print(f"\nSaving metadata to: {meta_path}")
    try:
        with open(meta_path, 'w') as f:
            json.dump(data_meta, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

    print("\nDataset preparation finished.")


if __name__ == '__main__':
    prepare_aig_dataset()