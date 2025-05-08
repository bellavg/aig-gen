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
            # Ensure v exists in the map before decrementing
            if v in in_degree_map:
                in_degree_map[v] -= 1
                if in_degree_map[v] == 0: newly_zero_in_degree.append(v)
                elif in_degree_map[v] < 0: raise RuntimeError(f"Negative in-degree for {v}")
            else:
                 warnings.warn(f"Node {v} (successor of {u}) not found in in_degree_map during topological sort. Graph might be inconsistent.")

        if len(newly_zero_in_degree) > 1: random_generator.shuffle(newly_zero_in_degree)
        for node in newly_zero_in_degree: queue.append(node)
    if len(result_order) != G.number_of_nodes():
        missing_nodes = set(G.nodes()) - set(result_order)
        cycle_nodes = {node for node, degree in in_degree_map.items() if degree > 0 and node in G}
        raise nx.NetworkXUnfeasible(f"Graph contains a cycle or is disconnected. Topological sort cannot proceed. "
                                    f"Result length: {len(result_order)}, Expected: {G.number_of_nodes()}. "
                                    f"Missing nodes: {missing_nodes}. Cycle nodes (approx): {cycle_nodes}")
    return result_order
# --- End Custom Sort ---


def _convert_nx_to_pyg_data(graph: nx.DiGraph,
                            max_nodes_pad: int,
                            num_node_features: int,
                            num_adj_channels: int,
                            num_explicit_edge_types: int) -> Data | None:
    """ Converts NetworkX graph from PKL to PyG Data object with padding. """
    # (Keep this function exactly as before)
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
    Processes raw PKL AIG graphs, optionally augments them using randomized
    topological sort, and saves/loads them as a PyTorch Geometric InMemoryDataset.
    Handles both initial processing and subsequent loading of processed files.
    """
    def __init__(self, root: str, dataset_name: str, split: str,
                 # Make raw data args optional for loading
                 raw_dir: Optional[str] = None,
                 file_prefix: Optional[str] = None,
                 pkl_file_names_for_split: Optional[List[str]] = None,
                 num_augmentations: int = 5, # Default augmentations for training
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Initializes the dataset, handling processing or loading.
        """
        # Store raw info temporarily, needed if super calls process()
        # These are NOT instance attributes after __init__ finishes unless explicitly set later
        self._temp_raw_dir = raw_dir
        self._temp_file_prefix = file_prefix
        self._temp_pkl_files = pkl_file_names_for_split if pkl_file_names_for_split is not None else []

        # Set essential attributes needed by properties before calling super()
        self.dataset_name = dataset_name
        self.split = split
        self.num_augmentations = num_augmentations # Store augmentation info

        # The root directory for InMemoryDataset should contain the 'processed' folder
        processed_root = osp.join(root, dataset_name)

        # === Call super().__init__ EARLY ===
        # This call checks if processed_paths exist. If not, it calls self.process().
        # self.process() will need the temporary raw info variables set above.
        super().__init__(processed_root, transform, pre_transform, pre_filter)
        # === super().__init__() finished ===

        # Now load data. super() either loaded it or called process() which saved it.
        try:
            # self.processed_paths is now defined by the superclass constructor
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Dataset '{self.dataset_name}' split '{self.split}' initialized. Samples: {len(self)}")
        except FileNotFoundError:
             raise FileNotFoundError(f"Processed file not found at {self.processed_paths[0]}. "
                                     "Ensure processing completed successfully or provide raw data args.")
        except Exception as e:
            raise RuntimeError(f"Failed to load processed data from {self.processed_paths[0]}: {e}")

        # Clean up temporary attributes if they are no longer needed
        del self._temp_raw_dir
        del self._temp_file_prefix
        del self._temp_pkl_files


    @property
    def raw_file_names(self) -> List[str]:
        """ Returns the list of raw filenames relative to raw_dir. Used by InMemoryDataset if processing. """
        # This property might be called by super().__init__ BEFORE the temp vars are deleted,
        # so it needs to access the temporary list.
        # If loading, this list might be empty, which is fine.
        return getattr(self, '_temp_pkl_files', []) # Access temp list safely

    # raw_paths property is used by self.process()
    @property
    def raw_paths(self) -> List[str]:
        """ Returns a list of absolute paths to the raw files for this split. """
        # This property is called BY self.process(), so temp vars MUST exist if process runs.
        if not hasattr(self, '_temp_raw_dir') or self._temp_raw_dir is None: return []
        return [osp.join(self._temp_raw_dir, name) for name in self.raw_file_names]

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        """ Returns the name(s) of the processed file(s) relative to processed_dir. """
        return [f'{self.split}_augmented_data.pt']

    def download(self):
        """ Download dataset. Not needed here as PKL files are local. """
        pass

    def process(self):
        """ Processes raw PKL files, applies augmentation, and saves PyG data. """
        # Check if necessary raw data information was provided via temp vars
        # These checks run only if super().__init__ determined processing is needed
        if self._temp_raw_dir is None or self._temp_file_prefix is None or not self._temp_pkl_files:
            raise ValueError("Raw directory, file prefix, and file list are required for processing, but not provided to __init__.")

        print(f"Processing raw PKL files and augmenting for split: {self.split}...")
        all_data_for_split = []
        original_graphs_processed = 0
        successful_conversions = 0
        augmentations_created = 0

        # Use self.raw_paths which correctly uses the temp vars
        for pkl_file_idx, raw_path in enumerate(tqdm(self.raw_paths, desc=f"Processing PKL files for {self.split}")):
            # --- Load Chunk ---
            if not osp.exists(raw_path):
                warnings.warn(f"Raw file not found: {raw_path}. Skipping.")
                continue
            try:
                with open(raw_path, 'rb') as f: nx_graphs_chunk = pickle.load(f)
                if not isinstance(nx_graphs_chunk, list):
                    warnings.warn(f"File {raw_path} did not contain a list. Skipping."); continue
            except Exception as e:
                warnings.warn(f"Could not load {raw_path}: {e}. Skipping file."); continue

            # --- Process Graphs in Chunk ---
            for graph_idx_in_chunk, nx_graph in enumerate(
                    tqdm(nx_graphs_chunk, desc=f"  Graphs in {osp.basename(raw_path)}", leave=False)):
                if not isinstance(nx_graph, nx.DiGraph):
                    warnings.warn("Item is not a NetworkX DiGraph. Skipping."); continue

                original_graphs_processed += 1
                base_pyg_data = _convert_nx_to_pyg_data(
                    nx_graph, MAX_NODES_PAD, NUM_NODE_FEATURES,
                    NUM_ADJ_CHANNELS, NUM_EXPLICIT_EDGE_TYPES)

                if base_pyg_data is not None:
                    successful_conversions += 1
                    all_data_for_split.append(base_pyg_data.clone()) # Add original

                    # --- Apply Augmentations (if training split) ---
                    num_actual_nodes = base_pyg_data.num_nodes.item()
                    # Use self.num_augmentations set in __init__
                    if num_actual_nodes > 0 and self.num_augmentations > 0:
                        try:
                            temp_nx_graph = nx.DiGraph() # Reconstruct temp graph
                            for i in range(num_actual_nodes): temp_nx_graph.add_node(i)
                            for ch in range(NUM_EXPLICIT_EDGE_TYPES):
                                adj_channel = base_pyg_data.adj[ch, :num_actual_nodes, :num_actual_nodes]
                                sources, targets = adj_channel.nonzero(as_tuple=True)
                                for src, tgt in zip(sources.tolist(), targets.tolist()):
                                    temp_nx_graph.add_edge(src, tgt)

                            # Apply augmentations
                            for aug_idx in range(self.num_augmentations):
                                try:
                                    # Consistent seeding
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
                                        all_data_for_split.append(augmented_data)
                                        augmentations_created += 1
                                    else: pass # warnings.warn("Topo sort mismatch. Skipping aug.")
                                except nx.NetworkXUnfeasible: pass # warnings.warn("Cycle detected. Skipping aug.")
                                except Exception as e: warnings.warn(f"Augmentation error: {e}")
                        except Exception as recon_e:
                             warnings.warn(f"Graph reconstruction error: {recon_e}")

            del nx_graphs_chunk; gc.collect()

        # --- Final Logging and Saving ---
        print(f"\nFinished PKL processing and augmentation for split '{self.split}'.")
        print(f"Total original graphs considered: {original_graphs_processed}")
        print(f"Successfully converted to PyG Data (before augmentation): {successful_conversions}")
        print(f"Total augmentations created: {augmentations_created}")
        print(f"Total samples saved for this split (original + augmentations): {len(all_data_for_split)}")

        if not all_data_for_split:
            warnings.warn(f"No data processed for split '{self.split}'. Saving empty file.")
            data, slices = self.collate([])
        else:
            data, slices = self.collate(all_data_for_split)

        # Ensure processed directory exists before saving
        # self.processed_dir is defined by the superclass based on the root path
        os.makedirs(self.processed_dir, exist_ok=True)
        # self.processed_paths[0] is also defined by the superclass
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved processed data for split '{self.split}' to: {self.processed_paths[0]}")

    # get() and len() are inherited from InMemoryDataset and work correctly


# --- Command Line Interface for Processing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AIG PKL files into augmented PyG InMemoryDataset format.")
    parser.add_argument('--raw_dir', type=str, required=True,
                        help="Directory containing the raw PKL files.")
    parser.add_argument('--output_root', type=str, required=True,
                        help="Root directory where dataset subfolders (e.g., 'aig_ds/processed/') will be created.")
    parser.add_argument('--dataset_name', type=str, default='aig_ds',
                        help="Name of the dataset (used for subdirectories under output_root).")
    parser.add_argument('--file_prefix', type=str, default="real_aigs_part_",
                        help="Prefix of the raw PKL files.")
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help="Number of randomized topological sort augmentations per graph (for training set).")

    parser.add_argument('--num_train_files', type=int, default=4, help="Num PKL files for training.")
    parser.add_argument('--num_val_files', type=int, default=1, help="Num PKL files for validation.")
    parser.add_argument('--num_test_files', type=int, default=1, help="Num PKL files for test.")
    args = parser.parse_args()

    if not osp.isdir(args.raw_dir):
        print(f"Error: Raw directory not found: {args.raw_dir}"); exit(1)

    try:
        all_pkl_files_names = sorted([
            f for f in os.listdir(args.raw_dir)
            if f.startswith(args.file_prefix) and f.endswith(".pkl") ])
    except OSError as e:
        print(f"Error listing files in {args.raw_dir}: {e}"); exit(1)

    if not all_pkl_files_names:
        print(f"No PKL files found in {args.raw_dir} with prefix '{args.file_prefix}'. Exiting."); exit(1)

    effective_total_files = len(all_pkl_files_names)
    print(f"Found {effective_total_files} PKL files matching prefix '{args.file_prefix}' in {args.raw_dir}.")

    # --- Allocate files to splits ---
    current_idx = 0
    train_files_list, val_files_list, test_files_list = [], [], []
    if args.num_train_files > 0:
        take_n = min(args.num_train_files, effective_total_files - current_idx)
        train_files_list = all_pkl_files_names[current_idx : current_idx + take_n]; current_idx += take_n
    if args.num_val_files > 0:
        take_n = min(args.num_val_files, effective_total_files - current_idx)
        val_files_list = all_pkl_files_names[current_idx : current_idx + take_n]; current_idx += take_n
    if args.num_test_files > 0:
        take_n = min(args.num_test_files, effective_total_files - current_idx)
        test_files_list = all_pkl_files_names[current_idx : current_idx + take_n]; current_idx += take_n

    print(f"\nFile Allocation:\n  Train: {train_files_list}\n  Val:   {val_files_list}\n  Test:  {test_files_list}")
    if current_idx < effective_total_files:
        warnings.warn(f"{effective_total_files - current_idx} PKL files unallocated: {all_pkl_files_names[current_idx:]}")

    # --- Process Splits by Instantiating Dataset ---
    # This triggers the .process() method if the processed file doesn't exist
    if train_files_list:
        print(f"\n--- Initializing/Processing Training Set ({args.num_augmentations} augs/graph) ---")
        # Pass all required args for processing
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='train',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=train_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=args.num_augmentations)
    if val_files_list:
        print(f"\n--- Initializing/Processing Validation Set (0 augs/graph) ---")
        # Pass all required args for processing, but set num_augmentations=0
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='val',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=val_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=0)
    if test_files_list:
        print(f"\n--- Initializing/Processing Test Set (0 augs/graph) ---")
        # Pass all required args for processing, but set num_augmentations=0
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='test',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=test_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=0)

    print("\nAll specified dataset processing finished.")
    print(f"Processed data should be in subdirs under: {osp.join(args.output_root, args.dataset_name, 'processed')}")
