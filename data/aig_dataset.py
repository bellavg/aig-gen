#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
import warnings
import os.path as osp
import argparse
import gc
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Deque, Optional
from collections import Counter, deque
import random

# --- AIG Model/Data Parameters ---
MAX_NODES_PAD = 64
NUM_NODE_FEATURES = 4
NUM_EXPLICIT_EDGE_TYPES = 2
NUM_ADJ_CHANNELS = NUM_EXPLICIT_EDGE_TYPES + 1
# --- End AIG Parameters ---

# --- User's Custom Randomized Topological Sort ---
def custom_randomized_topological_sort(G: nx.DiGraph, random_generator: random.Random) -> List[int]:
    """ Performs a randomized topological sort. """
    # (Implementation remains the same as before)
    if not G.is_directed(): raise nx.NetworkXError("Undirected graph.")
    in_degree_map = {n: d for n, d in G.in_degree()}
    zero_in_degree_nodes = [n for n, d in in_degree_map.items() if d == 0]
    if len(zero_in_degree_nodes) > 1: random_generator.shuffle(zero_in_degree_nodes)
    queue: Deque[int] = deque(zero_in_degree_nodes)
    result_order: List[int] = []
    while queue:
        u = queue.popleft(); result_order.append(u)
        newly_zero_in_degree: List[int] = []
        successors_of_u = [succ for succ in G.successors(u) if succ in G]
        for v in sorted(list(successors_of_u)):
            if v in in_degree_map:
                in_degree_map[v] -= 1
                if in_degree_map[v] == 0: newly_zero_in_degree.append(v)
                elif in_degree_map[v] < 0: raise RuntimeError(f"Negative in-degree for {v}")
            else: warnings.warn(f"Node {v} (successor of {u}) not found in in_degree_map.")
        if len(newly_zero_in_degree) > 1: random_generator.shuffle(newly_zero_in_degree)
        for node in newly_zero_in_degree: queue.append(node)
    if len(result_order) != G.number_of_nodes():
        missing_nodes = set(G.nodes()) - set(result_order)
        cycle_nodes = {node for node, degree in in_degree_map.items() if degree > 0 and node in G}
        raise nx.NetworkXUnfeasible(f"Graph contains cycle/is disconnected. Length: {len(result_order)}/{G.number_of_nodes()}. Missing: {missing_nodes}. Cycle: {cycle_nodes}")
    return result_order
# --- End Custom Sort ---


def _convert_nx_to_pyg_data(graph: nx.DiGraph,
                            max_nodes_pad: int,
                            num_node_features: int,
                            num_adj_channels: int,
                            num_explicit_edge_types: int) -> Data | None:
    """ Converts NetworkX graph from PKL to PyG Data object with padding. """
    # (Implementation remains the same as before)
    num_nodes_in_graph = graph.number_of_nodes()
    if num_nodes_in_graph == 0: return None
    if num_nodes_in_graph > max_nodes_pad:
        warnings.warn(f"Skipping graph with {num_nodes_in_graph} nodes (> {max_nodes_pad}).")
        return None

    try: node_list = sorted(list(graph.nodes()))
    except TypeError: node_list = list(graph.nodes())
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    node_features_list = []
    for old_node_id in node_list:
        attrs = graph.nodes[old_node_id]; node_type_vec_raw = attrs.get('type')
        if node_type_vec_raw is None or not isinstance(node_type_vec_raw, (list, np.ndarray)) or len(node_type_vec_raw) != num_node_features:
             warnings.warn(f"Node {old_node_id} invalid type. Skipping graph.")
             return None
        try:
            node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
            if not (np.isclose(np.sum(node_type_vec), 1.0) and np.all((np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
                 warnings.warn(f"Node {old_node_id} type not one-hot. Skipping graph.")
                 return None
            node_features_list.append(torch.tensor(node_type_vec, dtype=torch.float))
        except Exception: warnings.warn(f"Node {old_node_id} type conversion error. Skipping graph."); return None
    if not node_features_list: return None

    x_stacked = torch.stack(node_features_list)
    num_padding_nodes = max_nodes_pad - num_nodes_in_graph
    x_padded = F.pad(x_stacked, (0, 0, 0, num_padding_nodes), "constant", 0.0)

    adj_matrix = torch.zeros((num_adj_channels, max_nodes_pad, max_nodes_pad), dtype=torch.float)
    adj_matrix[num_explicit_edge_types, :, :] = 1.0

    for u_old, v_old, attrs in graph.edges(data=True):
        edge_type_vec_raw = attrs.get('type')
        if edge_type_vec_raw is None or not isinstance(edge_type_vec_raw, (list, np.ndarray)) or len(edge_type_vec_raw) != num_explicit_edge_types:
             warnings.warn(f"Edge ({u_old}-{v_old}) invalid type. Skipping graph.")
             return None
        u_new, v_new = node_id_map.get(u_old), node_id_map.get(v_old)
        if u_new is None or v_new is None or not (0 <= u_new < num_nodes_in_graph and 0 <= v_new < num_nodes_in_graph): continue
        try:
            edge_type_vec = np.asarray(edge_type_vec_raw, dtype=np.float32)
            if not (np.isclose(np.sum(edge_type_vec), 1.0) and np.all((np.isclose(edge_type_vec, 0.0)) | (np.isclose(edge_type_vec, 1.0)))):
                 warnings.warn(f"Edge ({u_old}-{v_old}) type not one-hot. Skipping graph.")
                 return None
            edge_channel_index = np.argmax(edge_type_vec).item()
            if not (0 <= edge_channel_index < num_explicit_edge_types):
                 warnings.warn(f"Edge ({u_old}-{v_old}) invalid channel index. Skipping graph.")
                 return None
            adj_matrix[edge_channel_index, v_new, u_new] = 1.0
            adj_matrix[num_explicit_edge_types, v_new, u_new] = 0.0
        except Exception: warnings.warn(f"Edge ({u_old}-{v_old}) type conversion error. Skipping graph."); return None

    no_edge_channel_idx = num_explicit_edge_types
    for k_node_diag in range(max_nodes_pad):
        adj_matrix[no_edge_channel_idx, k_node_diag, k_node_diag] = 0.0
        for ch in range(num_explicit_edge_types): adj_matrix[ch, k_node_diag, k_node_diag] = 0.0

    data = Data(x=x_padded, adj=adj_matrix, num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long))
    if 'inputs' in graph.graph: data.num_inputs = torch.tensor(graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in graph.graph: data.num_outputs = torch.tensor(graph.graph['outputs'], dtype=torch.long)
    if 'gates' in graph.graph: data.num_gates = torch.tensor(graph.graph['gates'], dtype=torch.long)
    return data


class AIGProcessedAugmentedDataset(InMemoryDataset):
    """
    Processes raw PKL AIG graphs, optionally augments them, saves/loads PyG data.
    Handles OOM by processing PKL files sequentially and saving intermediate chunks.
    """
    def __init__(self, root: str, dataset_name: str, split: str,
                 raw_dir: Optional[str] = None,
                 file_prefix: Optional[str] = None,
                 pkl_file_names_for_split: Optional[List[str]] = None,
                 num_augmentations: int = 5,
                 transform=None, pre_transform=None, pre_filter=None):
        """ Initializes the dataset, handling processing or loading. """
        self._temp_raw_dir = raw_dir
        self._temp_file_prefix = file_prefix
        self._temp_pkl_files = pkl_file_names_for_split if pkl_file_names_for_split is not None else []
        self.dataset_name = dataset_name
        self.split = split
        self.num_augmentations = num_augmentations
        processed_root = osp.join(root, dataset_name)
        super().__init__(processed_root, transform, pre_transform, pre_filter)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Dataset '{self.dataset_name}' split '{self.split}' initialized. Samples: {len(self)}")
        except FileNotFoundError:
             raise FileNotFoundError(f"Final processed file not found: {self.processed_paths[0]}. Run processing first.")
        except Exception as e:
            # If loading fails, it might be due to the weights_only issue during load itself
            # Try loading with weights_only=False as a fallback during initialization
            try:
                print(f"Initial load failed for {self.processed_paths[0]}. Attempting load with weights_only=False...")
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
                print(f"Successfully loaded with weights_only=False.")
                print(f"Dataset '{self.dataset_name}' split '{self.split}' initialized. Samples: {len(self)}")
            except Exception as e_fallback:
                 raise RuntimeError(f"Failed to load processed data from {self.processed_paths[0]} even with weights_only=False: {e_fallback}")

        # Clean up temporary attributes if they exist
        if hasattr(self, '_temp_raw_dir'): del self._temp_raw_dir
        if hasattr(self, '_temp_file_prefix'): del self._temp_file_prefix
        if hasattr(self, '_temp_pkl_files'): del self._temp_pkl_files

    @property
    def raw_file_names(self) -> List[str]:
        return getattr(self, '_temp_pkl_files', [])

    @property
    def raw_paths(self) -> List[str]:
        if not hasattr(self, '_temp_raw_dir') or self._temp_raw_dir is None: return []
        return [osp.join(self._temp_raw_dir, name) for name in self.raw_file_names]

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return [f'{self.split}_augmented_data.pt']

    def download(self):
        pass

    def process(self):
        """ Processes raw PKL files sequentially and saves PyG data. """
        if self._temp_raw_dir is None or self._temp_file_prefix is None or not self._temp_pkl_files:
            raise ValueError("Raw directory, file prefix, and file list are required for processing.")

        print(f"Processing raw PKL files sequentially and augmenting for split: {self.split}...")
        intermediate_files = []
        total_original_graphs, total_successful_conversions, total_augmentations_created = 0, 0, 0
        os.makedirs(self.processed_dir, exist_ok=True)

        for pkl_file_idx, raw_path in enumerate(tqdm(self.raw_paths, desc=f"Processing PKL Chunks ({self.split})")):
            if not osp.exists(raw_path): continue
            current_chunk_data_list = []
            try:
                with open(raw_path, 'rb') as f: nx_graphs_chunk = pickle.load(f)
                if not isinstance(nx_graphs_chunk, list): continue
            except Exception as e: warnings.warn(f"Load error {raw_path}: {e}"); continue

            for graph_idx_in_chunk, nx_graph in enumerate(nx_graphs_chunk):
                if not isinstance(nx_graph, nx.DiGraph): continue
                total_original_graphs += 1
                base_pyg_data = _convert_nx_to_pyg_data(
                    nx_graph, MAX_NODES_PAD, NUM_NODE_FEATURES,
                    NUM_ADJ_CHANNELS, NUM_EXPLICIT_EDGE_TYPES)
                if base_pyg_data is not None:
                    total_successful_conversions += 1
                    current_chunk_data_list.append(base_pyg_data.clone())
                    num_actual_nodes = base_pyg_data.num_nodes.item()
                    if num_actual_nodes > 0 and self.num_augmentations > 0:
                        try:
                            temp_nx_graph = nx.DiGraph()
                            for i in range(num_actual_nodes): temp_nx_graph.add_node(i)
                            for ch in range(NUM_EXPLICIT_EDGE_TYPES):
                                adj_channel = base_pyg_data.adj[ch, :num_actual_nodes, :num_actual_nodes]
                                sources, targets = adj_channel.nonzero(as_tuple=True)
                                for src, tgt in zip(sources.tolist(), targets.tolist()): temp_nx_graph.add_edge(src, tgt)
                            for aug_idx in range(self.num_augmentations):
                                try:
                                    aug_seed = (pkl_file_idx * len(nx_graphs_chunk) + graph_idx_in_chunk) * (self.num_augmentations + 1) + (aug_idx + 1)
                                    local_random = random.Random(aug_seed)
                                    ordered_nodes = custom_randomized_topological_sort(temp_nx_graph, local_random)
                                    if len(ordered_nodes) == num_actual_nodes:
                                        augmented_data = base_pyg_data.clone()
                                        current_order = np.array(ordered_nodes, dtype=np.int64)
                                        padding_order = np.arange(num_actual_nodes, MAX_NODES_PAD, dtype=np.int64)
                                        full_perm = np.concatenate([current_order, padding_order])
                                        full_perm_tensor = torch.from_numpy(full_perm).long()
                                        augmented_data.x = augmented_data.x[full_perm_tensor]
                                        augmented_data.adj = augmented_data.adj[:, full_perm_tensor][:, :, full_perm_tensor]
                                        current_chunk_data_list.append(augmented_data)
                                        total_augmentations_created += 1
                                except nx.NetworkXUnfeasible: pass
                                except Exception as e: warnings.warn(f"Aug error: {e}")
                        except Exception as recon_e: warnings.warn(f"Recon error: {recon_e}")

            if current_chunk_data_list:
                intermediate_path = osp.join(self.processed_dir, f'{self.split}_temp_part_{pkl_file_idx}.pt')
                try:
                    # Save intermediate list directly
                    torch.save(current_chunk_data_list, intermediate_path)
                    intermediate_files.append(intermediate_path)
                    # print(f"  Saved intermediate chunk: {osp.basename(intermediate_path)} ({len(current_chunk_data_list)} graphs)") # Can be verbose
                except Exception as e: warnings.warn(f"Failed to save intermediate chunk {intermediate_path}: {e}")
            del current_chunk_data_list, nx_graphs_chunk; gc.collect()

        print(f"\nCombining {len(intermediate_files)} intermediate chunks for split '{self.split}'...")
        final_data_list = []
        for intermediate_path in tqdm(intermediate_files, desc="Combining Chunks"):
            try:
                # *** FIX: Load with weights_only=False ***
                chunk_data = torch.load(intermediate_path, weights_only=False)
                final_data_list.extend(chunk_data)
                os.remove(intermediate_path)
            except Exception as e:
                warnings.warn(f"Failed to load or delete intermediate chunk {intermediate_path}: {e}")

        print(f"\nFinished processing for split '{self.split}'.")
        print(f"Total original graphs considered: {total_original_graphs}")
        print(f"Successfully converted (before aug): {total_successful_conversions}")
        print(f"Total augmentations created: {total_augmentations_created}")
        print(f"Total samples combined: {len(final_data_list)}")

        # --- FIX: Handle empty list before collate ---
        if not final_data_list:
            warnings.warn(f"No data to collate for split '{self.split}'. Saving empty dataset.")
            # Create placeholder empty data and slices
            # Need a dummy Data object to get the class for collate
            # Or handle saving manually
            # Let's create dummy data manually to avoid collate error
            data = Data() # Create an empty Data object
            slices = {key: torch.tensor([0, 0]) for key in data.keys} # Basic slices for empty data
        else:
            # Apply pre_filter and pre_transform if specified
            if self.pre_filter is not None:
                 print("Applying pre_filter...")
                 final_data_list = [d for d in final_data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                 print("Applying pre_transform...")
                 final_data_list = [self.pre_transform(d) for d in final_data_list]

            if not final_data_list: # Check again after filtering
                 warnings.warn(f"No data remaining after pre-filtering for split '{self.split}'. Saving empty dataset.")
                 data = Data()
                 slices = {key: torch.tensor([0, 0]) for key in data.keys}
            else:
                 print(f"Collating final data list ({len(final_data_list)} samples)...")
                 data, slices = self.collate(final_data_list) # Now safe to call collate

        final_save_path = self.processed_paths[0]
        torch.save((data, slices), final_save_path)
        print(f"Saved final processed data for split '{self.split}' to: {final_save_path}")


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AIG PKL files into augmented PyG InMemoryDataset format.")
    parser.add_argument('--raw_dir', type=str, required=True, help="Dir containing raw PKL files.")
    parser.add_argument('--output_root', type=str, required=True, help="Root dir for dataset output.")
    parser.add_argument('--dataset_name', type=str, default='aig_ds', help="Name for dataset subfolder.")
    parser.add_argument('--file_prefix', type=str, default="real_aigs_part_", help="Prefix of raw PKL files.")
    parser.add_argument('--num_augmentations', type=int, default=5, help="Num augmentations for training set.")
    parser.add_argument('--num_train_files', type=int, default=4, help="Num PKL files for training.")
    parser.add_argument('--num_val_files', type=int, default=1, help="Num PKL files for validation.")
    parser.add_argument('--num_test_files', type=int, default=1, help="Num PKL files for test.")
    args = parser.parse_args()

    if not osp.isdir(args.raw_dir): print(f"Error: Raw dir not found: {args.raw_dir}"); exit(1)
    try:
        all_pkl_files_names = sorted([f for f in os.listdir(args.raw_dir) if f.startswith(args.file_prefix) and f.endswith(".pkl")])
    except OSError as e: print(f"Error listing files: {e}"); exit(1)
    if not all_pkl_files_names: print(f"No PKL files found."); exit(1)

    effective_total_files = len(all_pkl_files_names)
    print(f"Found {effective_total_files} PKL files matching prefix '{args.file_prefix}' in {args.raw_dir}.")

    current_idx = 0; train_files_list, val_files_list, test_files_list = [], [], []
    if args.num_train_files > 0: take_n = min(args.num_train_files, effective_total_files - current_idx); train_files_list = all_pkl_files_names[current_idx : current_idx + take_n]; current_idx += take_n
    if args.num_val_files > 0: take_n = min(args.num_val_files, effective_total_files - current_idx); val_files_list = all_pkl_files_names[current_idx : current_idx + take_n]; current_idx += take_n
    if args.num_test_files > 0: take_n = min(args.num_test_files, effective_total_files - current_idx); test_files_list = all_pkl_files_names[current_idx : current_idx + take_n]; current_idx += take_n
    print(f"\nFile Allocation:\n  Train: {train_files_list}\n  Val:   {val_files_list}\n  Test:  {test_files_list}")
    if current_idx < effective_total_files: warnings.warn(f"{effective_total_files - current_idx} PKL files unallocated.")

    # --- Process Splits by Instantiating Dataset ---
    if train_files_list:
        print(f"\n--- Initializing/Processing Training Set ({args.num_augmentations} augs/graph) ---")
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='train',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=train_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=args.num_augmentations)
    if val_files_list:
        print(f"\n--- Initializing/Processing Validation Set (0 augs/graph) ---")
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='val',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=val_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=0)
    if test_files_list:
        print(f"\n--- Initializing/Processing Test Set (0 augs/graph) ---")
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='test',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=test_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=0)

    print("\nAll specified dataset processing finished.")
    print(f"Processed data should be in subdirs under: {osp.join(args.output_root, args.dataset_name, 'processed')}")
