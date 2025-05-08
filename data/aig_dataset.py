#!/usr/bin/env python3
import os
import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
from typing import List, Tuple # Optional if you remove all type hints for unused parts

# --- AIG Model/Data Parameters (Constants used if data has specific structure) ---
# These might still be relevant if your model relies on knowing these dimensions,
# even if the dataset loader itself doesn't use them for processing.
# If they are ONLY for the _convert_nx_to_pyg_data function which is removed,
# then these can also be removed from this file. For now, kept for context.
MAX_NODES_PAD = 64
NUM_NODE_FEATURES = 4
NUM_EXPLICIT_EDGE_TYPES = 2
NUM_ADJ_CHANNELS = NUM_EXPLICIT_EDGE_TYPES + 1
# --- End AIG Parameters ---

class AIGPreprocessedDatasetLoader(InMemoryDataset):
    def __init__(self, root: str, dataset_name: str, split: str,
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Loads a pre-processed AIG dataset.

        Args:
            root (str): Root directory where the dataset folder `dataset_name` is located.
            dataset_name (str): Name of the dataset folder.
            split (str): The name of the split (e.g., 'train', 'val', 'test').
                         This determines the filename (e.g., 'train_processed_data.pt').
            transform: PyG transforms.
            pre_transform: PyG pre-transforms (not typically used for loader-only).
            pre_filter: PyG pre-filter (not typically used for loader-only).
        """
        self.dataset_name = dataset_name
        self.split = split

        # The root directory for this specific dataset (e.g., ./my_output_root/aig_ds)
        # InMemoryDataset will look for a 'processed' subfolder within this path.
        dataset_processed_root_path = osp.join(root, dataset_name)

        # Call super().__init__
        # This will:
        # 1. Check if self.processed_paths[0] exists.
        # 2. If yes, it loads the data.
        # 3. If no, it will call self.download() because self.raw_file_names is empty.
        super().__init__(dataset_processed_root_path, transform, pre_transform, pre_filter)

        # After super().__init__(), self.data and self.slices should be populated
        # if loading was successful from self.processed_paths[0].
        # If self.download() was called (because the file didn't exist),
        # it would have raised an error as per our implementation below.
        if self.data is None or self.slices is None:
            # This state implies that super().__init__ failed to load the processed file.
            # Our download() method below should have already raised an error.
            # This is a fallback check.
            raise FileNotFoundError(
                f"Pre-processed file not found or failed to load: {self.processed_paths[0]}. "
                f"This dataset loader expects .pt files to already exist. "
                f"Please ensure 'root' ('{root}'), 'dataset_name' ('{dataset_name}'), "
                f"and 'split' ('{split}') arguments correctly point to an existing file."
            )

        print(f"Dataset '{self.dataset_name}' split '{self.split}' successfully loaded from {self.processed_paths[0]}. Samples: {len(self)}")

    @property
    def raw_file_names(self) -> List[str]:
        # No raw files are used by this loader.
        return []

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        # Defines the name of the pre-processed file to load.
        return [f'{self.split}_processed_data.pt']

    def download(self):
        # This method is called by InMemoryDataset if processed_paths[0] is not found
        # AND raw_file_names is empty.
        # Since we require pre-processed files, we raise an error.
        raise FileNotFoundError(
            f"Pre-processed file not found: {self.processed_paths[0]}. "
            f"This loader expects this file to exist and does not support downloading or on-the-fly processing. "
            f"Please ensure the file is present at the correct path."
        )

    def process(self):
        # This method should not be called if raw_file_names is empty and
        # download() is implemented as above.
        # It's included as a safeguard.
        raise NotImplementedError(
            "process() should not be called. This dataset loader is for pre-processed files only. "
            f"Ensure {self.processed_paths[0]} exists."
        )

# The following functions from your original script are related to processing raw data.
# Since this version of the dataset class ONLY loads pre-processed .pt files,
# these functions (_convert_nx_to_pyg_data, custom_randomized_topological_sort)
# are no longer used by AIGPreprocessedDatasetLoader and can be removed from this file
# if it's solely dedicated to this loader.
# If you still use the `if __name__ == "__main__":` block from the original
# aig_dataset.py to *generate* the .pt files, you'd keep them there or in a separate processing script.

# (Optional: Remove _convert_nx_to_pyg_data and custom_randomized_topological_sort
#  and the `if __name__ == "__main__":` block from your original aig_dataset.py
#  if this file will now *only* contain the AIGPreprocessedDatasetLoader class)


#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
# Import BaseData explicitly if needed for type hints or checks
from torch_geometric.data.data import BaseData  # For type checking loaded data
import warnings
import os.path as osp
import argparse
import gc
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Deque, Optional
from collections import Counter, deque
import random
import shutil  # For removing temporary directories

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
    in_degree_map = {n: d for n, d in G.in_degree() if n in G}  # Ensure node is in G

    # Handle nodes that might be in in_degree but not in G.nodes (if G was modified)
    # Or nodes in G.nodes() but not in in_degree (isolated nodes)
    for node in list(G.nodes()):  # Use list to avoid issues if G is modified
        if node not in in_degree_map:
            # This case should ideally not happen if in_degree is called on the current G
            # but as a safeguard, or if G can have isolated nodes not caught by in_degree() if it returns only >0
            if G.in_degree(node) == 0:
                in_degree_map[node] = 0
            else:
                # This would be an inconsistent state
                warnings.warn(
                    f"Node {node} in G.nodes() but not in initial in_degree_map and has in-degree > 0. Graph state might be inconsistent.")

    zero_in_degree_nodes = [n for n, d in in_degree_map.items() if d == 0]
    if len(zero_in_degree_nodes) > 1: random_generator.shuffle(zero_in_degree_nodes)

    queue: Deque[int] = deque(zero_in_degree_nodes)
    result_order: List[int] = []

    processed_nodes_count = 0

    while queue:
        u = queue.popleft()
        result_order.append(u)
        processed_nodes_count += 1

        # Sort successors for deterministic behavior when not shuffling, and to process them in a consistent order
        # Ensure successors are actually in the graph and in the in_degree_map
        successors_of_u = sorted([succ for succ in G.successors(u) if succ in G and succ in in_degree_map])

        newly_zero_in_degree: List[int] = []
        for v in successors_of_u:
            in_degree_map[v] -= 1
            if in_degree_map[v] == 0:
                newly_zero_in_degree.append(v)
            elif in_degree_map[v] < 0:
                # This indicates a graph structure problem or an error in the algorithm
                raise RuntimeError(
                    f"Negative in-degree for node {v} encountered during topological sort. Current order: {result_order}")

        if len(newly_zero_in_degree) > 1: random_generator.shuffle(newly_zero_in_degree)
        for node in newly_zero_in_degree: queue.append(node)

    if processed_nodes_count != G.number_of_nodes():
        # More detailed cycle detection or disconnected component identification
        remaining_nodes = set(G.nodes()) - set(result_order)
        cycle_nodes_or_unreachable = {node for node, degree in in_degree_map.items() if
                                      degree > 0 and node in G and node in remaining_nodes}

        # Try to find a cycle explicitly for better error message
        try:
            cycle = nx.find_cycle(G.subgraph(remaining_nodes))  # Check for cycle in remaining part
            raise nx.NetworkXUnfeasible(
                f"Graph contains a cycle. Example cycle: {cycle}. Processed: {len(result_order)}/{G.number_of_nodes()}. Remaining nodes possibly in cycle: {cycle_nodes_or_unreachable}")
        except nx.NetworkXNoCycle:
            # If no cycle found, it might be disconnected or other issues
            raise nx.NetworkXUnfeasible(
                f"Topological sort failed. Graph might be disconnected or have other structural issues. Processed: {len(result_order)}/{G.number_of_nodes()}. Remaining nodes: {remaining_nodes}. Nodes with positive in-degree in remainder: {cycle_nodes_or_unreachable}")

    return result_order


# --- End Custom Sort ---


def _convert_nx_to_pyg_data(graph: nx.DiGraph,
                            max_nodes_pad: int,
                            num_node_features: int,
                            num_adj_channels: int,
                            num_explicit_edge_types: int) -> Optional[Data]:
    """ Converts NetworkX graph from PKL to PyG Data object with padding. """
    num_nodes_in_graph = graph.number_of_nodes()
    if num_nodes_in_graph == 0:
        # warnings.warn("Skipping empty graph.") # Can be too verbose
        return None
    if num_nodes_in_graph > max_nodes_pad:
        warnings.warn(f"Skipping graph with {num_nodes_in_graph} nodes (> {max_nodes_pad}).")
        return None

    try:
        # Ensure node list is consistently ordered for mapping.
        # Sorting is important if node IDs are not already 0...N-1 integers.
        node_list = sorted(list(graph.nodes()))
    except TypeError:
        # Fallback if nodes are not sortable (e.g., mixed types, though usually not for nx graphs)
        node_list = list(graph.nodes())
        warnings.warn(
            "Nodes could not be sorted, using arbitrary order. This might lead to inconsistencies if node IDs are not integers.")

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    node_features_list = []
    for old_node_id in node_list:  # Iterate in the sorted order
        attrs = graph.nodes[old_node_id]
        node_type_vec_raw = attrs.get('type')
        if node_type_vec_raw is None or not isinstance(node_type_vec_raw, (list, np.ndarray)) or len(
                node_type_vec_raw) != num_node_features:
            warnings.warn(f"Node {old_node_id} has invalid type attribute (length or existence). Skipping graph.")
            return None
        try:
            node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
            # Check for one-hot encoding properties
            if not (np.isclose(np.sum(node_type_vec), 1.0) and np.all(
                    (np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
                warnings.warn(
                    f"Node {old_node_id} type vector is not one-hot. Sum: {np.sum(node_type_vec)}. Values: {node_type_vec}. Skipping graph.")
                return None
            node_features_list.append(torch.tensor(node_type_vec, dtype=torch.float))
        except Exception as e:
            warnings.warn(f"Node {old_node_id} type conversion to tensor failed: {e}. Skipping graph.")
            return None

    if not node_features_list:  # Should be caught by num_nodes_in_graph == 0, but as a safeguard
        return None

    x_stacked = torch.stack(node_features_list)
    num_padding_nodes = max_nodes_pad - num_nodes_in_graph
    # Pad node features: (padding_left, padding_right, padding_top, padding_bottom)
    x_padded = F.pad(x_stacked, (0, 0, 0, num_padding_nodes), "constant", 0.0)

    # Adjacency matrix: (Channels, MaxNodes, MaxNodes)
    # Using float for adj as it's common, can be bool if memory is critical and PyG handles it.
    adj_matrix = torch.zeros((num_adj_channels, max_nodes_pad, max_nodes_pad), dtype=torch.float)

    # Initialize the "no_edge_channel" (last channel) with self-loops for existing nodes,
    # and all-to-all for padding nodes (or handle as per model needs).
    # A common strategy for the "no edge" or "complement" channel is to mark where explicit edges ARE NOT.
    # Let's assume the last channel indicates "no explicit typed edge"
    # For actual nodes, if there's no explicit edge, this channel will be 1.
    # For padding nodes, they don't have connections.
    # Initialize last channel to 1 for all possible connections, then set to 0 where explicit edges exist.
    adj_matrix[num_explicit_edge_types, :num_nodes_in_graph, :num_nodes_in_graph] = 1.0

    for u_old, v_old, attrs in graph.edges(data=True):
        edge_type_vec_raw = attrs.get('type')
        if edge_type_vec_raw is None or not isinstance(edge_type_vec_raw, (list, np.ndarray)) or len(
                edge_type_vec_raw) != num_explicit_edge_types:
            warnings.warn(f"Edge ({u_old}-{v_old}) has invalid type attribute. Skipping graph.")
            return None

        u_new, v_new = node_id_map.get(u_old), node_id_map.get(v_old)
        # Ensure nodes are part of the mapped nodes and within the actual graph size (before padding)
        if u_new is None or v_new is None or not (0 <= u_new < num_nodes_in_graph and 0 <= v_new < num_nodes_in_graph):
            warnings.warn(
                f"Edge ({u_old}-{v_old}) connects to unknown or out-of-bounds new node IDs ({u_new}, {v_new}). Skipping this edge.")
            continue

        try:
            edge_type_vec = np.asarray(edge_type_vec_raw, dtype=np.float32)
            if not (np.isclose(np.sum(edge_type_vec), 1.0) and np.all(
                    (np.isclose(edge_type_vec, 0.0)) | (np.isclose(edge_type_vec, 1.0)))):
                warnings.warn(
                    f"Edge ({u_old}-{v_old}) type vector is not one-hot. Sum: {np.sum(edge_type_vec)}. Values: {edge_type_vec}. Skipping graph.")
                return None

            edge_channel_index = np.argmax(edge_type_vec).item()  # .item() to get Python scalar
            if not (0 <= edge_channel_index < num_explicit_edge_types):
                warnings.warn(
                    f"Edge ({u_old}-{v_old}) type vector resulted in invalid channel index: {edge_channel_index}. Skipping graph.")
                return None

            # Adjacency matrix convention: adj[channel, target_node, source_node] = 1
            adj_matrix[edge_channel_index, v_new, u_new] = 1.0
            # If an explicit edge exists, set the "no_edge_channel" to 0 for this connection
            adj_matrix[num_explicit_edge_types, v_new, u_new] = 0.0

        except Exception as e:
            warnings.warn(f"Edge ({u_old}-{v_old}) type conversion or assignment failed: {e}. Skipping graph.")
            return None

    # Finalize "no_edge_channel": self-loops in this channel are typically 0.
    # If a node has no incoming typed edges from another node j, then adj[no_edge_channel, i, j] remains 1.
    # If it has an incoming typed edge, it was set to 0.
    # Set diagonal of all channels to 0 (no self-loops of explicit types, and no self-loop in "no_edge_channel")
    for k_node_diag in range(max_nodes_pad):  # Iterate over the full padded dimension
        for ch in range(num_adj_channels):  # All channels including the "no_edge_channel"
            adj_matrix[ch, k_node_diag, k_node_diag] = 0.0

    # Create PyG Data object
    data = Data(x=x_padded, adj=adj_matrix, num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long))

    # Add graph-level attributes if they exist
    if 'inputs' in graph.graph: data.num_inputs = torch.tensor(graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in graph.graph: data.num_outputs = torch.tensor(graph.graph['outputs'], dtype=torch.long)
    if 'gates' in graph.graph: data.num_gates = torch.tensor(graph.graph['gates'], dtype=torch.long)

    return data


class AIGProcessedAugmentedDataset(InMemoryDataset):
    def __init__(self, root: str, dataset_name: str, split: str,
                 raw_dir: Optional[str] = None,
                 file_prefix: Optional[str] = None,
                 pkl_file_names_for_split: Optional[List[str]] = None,
                 num_augmentations: int = 5,
                 transform=None, pre_transform=None, pre_filter=None):

        self._temp_raw_dir = raw_dir
        self._temp_file_prefix = file_prefix
        self._temp_pkl_files = pkl_file_names_for_split if pkl_file_names_for_split is not None else []
        self.dataset_name = dataset_name
        self.split = split
        self.num_augmentations = num_augmentations if self.split == 'train' else 0  # Aug only for train

        # The root directory for this specific dataset (e.g., ./my_output_root/aig_ds)
        processed_dataset_root = osp.join(root, dataset_name)

        # Call super().__init__
        # This will trigger self.process() if processed_paths[0] is not found,
        # or load from processed_paths[0] if it exists.
        super().__init__(processed_dataset_root, transform, pre_transform, pre_filter)

        # Load the data and slices
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            print(f"Dataset '{self.dataset_name}' split '{self.split}' successfully loaded. Samples: {len(self)}")
        except FileNotFoundError:
            # This case should ideally be handled by InMemoryDataset calling process(),
            # but if process() fails to create the file, this could be reached.
            raise FileNotFoundError(
                f"Processed file not found: {self.processed_paths[0]}. Processing might have failed or was skipped.")
        except Exception as e:
            raise RuntimeError(f"Failed to load processed data from {self.processed_paths[0]}: {e}")

        # Clean up temporary attributes if they were set for processing
        if hasattr(self, '_temp_raw_dir'): del self._temp_raw_dir
        if hasattr(self, '_temp_file_prefix'): del self._temp_file_prefix
        if hasattr(self, '_temp_pkl_files'): del self._temp_pkl_files

    @property
    def raw_file_names(self) -> List[str]:
        # These are used by InMemoryDataset to check if raw files exist
        # and decide whether to call download() or process().
        # We need them to point to the actual PKL files for the current split.
        return getattr(self, '_temp_pkl_files', [])

    @property
    def raw_paths(self) -> List[str]:
        # Full paths to the raw PKL files for this split.
        if not hasattr(self, '_temp_raw_dir') or self._temp_raw_dir is None: return []
        return [osp.join(self._temp_raw_dir, name) for name in self.raw_file_names]

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        # The final, single file where the collated dataset for this split will be stored.
        return [f'{self.split}_processed_data.pt']  # Changed name slightly for clarity

    def download(self):
        # This method is called if raw_paths files are not found.
        # We assume PKL files are already present, so no download step is needed.
        pass

    def process(self):
        """
        Processes raw PKL files.
        Converts each NetworkX graph to a PyG Data object.
        Applies augmentations if specified for the 'train' split.
        Saves each original graph + its augmentations as a list to a temporary .pt file.
        Then, loads all these temporary .pt files, collates them, and saves the final dataset.
        """
        if self._temp_raw_dir is None or not self._temp_pkl_files:
            if not self.raw_paths:  # If raw_paths is empty, it means no files to process
                print(f"No raw files specified for split '{self.split}'. Skipping processing.")
                # Create an empty processed file to satisfy InMemoryDataset
                empty_data_list = []
                if self.pre_filter is not None: empty_data_list = [d for d in empty_data_list if self.pre_filter(d)]
                if self.pre_transform is not None: empty_data_list = [self.pre_transform(d) for d in empty_data_list]

                # Collate the empty list to get the correct structure for data and slices
                data, slices = self.collate(empty_data_list)
                torch.save((data, slices), self.processed_paths[0])
                print(f"Saved empty dataset for split '{self.split}' to: {self.processed_paths[0]}")
                return
            raise ValueError("Raw directory and file list are required for processing if raw_paths is not empty.")

        print(f"Processing raw PKL files for split: {self.split}...")

        # Temporary directory to store .pt files for each original graph + its augmentations
        # This directory will be within the self.processed_dir (e.g., .../my_dataset/processed/train_temp_pts)
        temp_individual_graph_pts_dir = osp.join(self.processed_dir, f"{self.split}_temp_individual_pts")
        os.makedirs(temp_individual_graph_pts_dir, exist_ok=True)

        paths_to_temp_pt_files = []  # Stores paths to the temporary .pt files created

        total_original_graphs_considered = 0
        total_successful_conversions = 0
        total_augmentations_created = 0
        temp_file_counter = 0  # For naming temporary .pt files

        # --- Loop 1: Process each PKL file, then each graph within it ---
        for pkl_file_idx, raw_pkl_path in enumerate(tqdm(self.raw_paths, desc=f"Processing PKL Chunks ({self.split})")):
            if not osp.exists(raw_pkl_path):
                warnings.warn(f"Raw PKL file not found: {raw_pkl_path}. Skipping.")
                continue

            try:
                with open(raw_pkl_path, 'rb') as f:
                    nx_graphs_in_chunk = pickle.load(f)
                if not isinstance(nx_graphs_in_chunk, list):
                    warnings.warn(f"Content of {raw_pkl_path} is not a list. Skipping.")
                    continue
            except Exception as e:
                warnings.warn(f"Error loading or reading PKL file {raw_pkl_path}: {e}. Skipping.")
                continue

            # Process each NetworkX graph from the current PKL chunk
            for nx_graph_idx, nx_graph in enumerate(nx_graphs_in_chunk):
                if not isinstance(nx_graph, nx.DiGraph):
                    warnings.warn(
                        f"Item at index {nx_graph_idx} in {raw_pkl_path} is not a NetworkX DiGraph. Skipping.")
                    continue

                total_original_graphs_considered += 1

                # This list will hold the base_pyg_data and its augmentations (if any)
                # for the CURRENT nx_graph only.
                pyg_data_objects_for_current_nx_graph = []

                base_pyg_data = _convert_nx_to_pyg_data(
                    nx_graph, MAX_NODES_PAD, NUM_NODE_FEATURES,
                    NUM_ADJ_CHANNELS, NUM_EXPLICIT_EDGE_TYPES)

                if base_pyg_data is not None:
                    total_successful_conversions += 1
                    pyg_data_objects_for_current_nx_graph.append(base_pyg_data)  # Add the original converted graph

                    num_actual_nodes = base_pyg_data.num_nodes.item()

                    # Perform augmentations only if conditions are met
                    if num_actual_nodes > 0 and self.num_augmentations > 0 and self.split == 'train':
                        try:
                            # Reconstruct a temporary NetworkX graph from base_pyg_data for topological sort
                            # This graph should only contain the actual nodes, not padded ones.
                            temp_nx_for_aug = nx.DiGraph()
                            # Nodes in temp_nx_for_aug will be 0 to num_actual_nodes-1
                            for i in range(num_actual_nodes): temp_nx_for_aug.add_node(i)

                            for ch_idx in range(NUM_EXPLICIT_EDGE_TYPES):
                                # Extract the relevant part of the adjacency matrix
                                adj_channel_for_actual_nodes = base_pyg_data.adj[ch_idx, :num_actual_nodes,
                                                               :num_actual_nodes]
                                # Get source and target indices where edges exist
                                source_indices, target_indices = adj_channel_for_actual_nodes.nonzero(as_tuple=True)
                                for src_node_idx, tgt_node_idx in zip(source_indices.tolist(), target_indices.tolist()):
                                    # Edges in PyG adj are (target, source), so reverse for NetworkX (source, target)
                                    temp_nx_for_aug.add_edge(tgt_node_idx,
                                                             src_node_idx)  # Corrected: adj is [channel, v, u]

                            for aug_idx in range(self.num_augmentations):
                                try:
                                    # Unique seed for each augmentation attempt
                                    aug_seed = (pkl_file_idx * len(nx_graphs_in_chunk) * (self.num_augmentations + 1)) + \
                                               (nx_graph_idx * (self.num_augmentations + 1)) + \
                                               (aug_idx + 1)
                                    local_random_generator = random.Random(aug_seed)

                                    # Perform randomized topological sort on the reconstructed graph
                                    ordered_node_indices = custom_randomized_topological_sort(temp_nx_for_aug,
                                                                                              local_random_generator)

                                    if len(ordered_node_indices) == num_actual_nodes:
                                        augmented_data_item = base_pyg_data.clone()  # Clone the original PyG data

                                        # Create the permutation map for actual nodes
                                        permutation_for_actual_nodes = np.array(ordered_node_indices, dtype=np.int64)
                                        # Create the identity map for padding nodes
                                        permutation_for_padding_nodes = np.arange(num_actual_nodes, MAX_NODES_PAD,
                                                                                  dtype=np.int64)

                                        # Combine to get the full permutation array for padded size
                                        full_permutation = np.concatenate(
                                            [permutation_for_actual_nodes, permutation_for_padding_nodes])
                                        full_permutation_tensor = torch.from_numpy(full_permutation).long()

                                        # Apply permutation
                                        augmented_data_item.x = augmented_data_item.x[full_permutation_tensor]
                                        augmented_data_item.adj = augmented_data_item.adj[:, full_permutation_tensor][:,
                                                                  :, full_permutation_tensor]

                                        pyg_data_objects_for_current_nx_graph.append(augmented_data_item)
                                        total_augmentations_created += 1
                                except nx.NetworkXUnfeasible as e:
                                    # This is expected if the graph has cycles, common in AIGs if not careful with interpretation
                                    # warnings.warn(f"Augmentation skipped for a graph due to cycle (or unfeasible sort): {e}")
                                    pass  # Don't be too verbose for expected failures
                                except Exception as e_aug:
                                    warnings.warn(f"Error during a single augmentation: {e_aug}")
                        except Exception as e_recon:
                            warnings.warn(f"Error reconstructing graph for augmentation: {e_recon}")

                # After processing one nx_graph and its augmentations, save them to a temp file
                if pyg_data_objects_for_current_nx_graph:
                    temp_pt_file_path = osp.join(temp_individual_graph_pts_dir, f"graph_batch_{temp_file_counter}.pt")
                    try:
                        torch.save(pyg_data_objects_for_current_nx_graph, temp_pt_file_path)
                        paths_to_temp_pt_files.append(temp_pt_file_path)
                        temp_file_counter += 1
                    except Exception as e_save:
                        warnings.warn(f"Could not save temporary batch file {temp_pt_file_path}: {e_save}")

                del pyg_data_objects_for_current_nx_graph, base_pyg_data  # Free memory
                if (nx_graph_idx + 1) % 200 == 0:  # Periodically collect garbage
                    gc.collect()

            del nx_graphs_in_chunk  # Free memory from the loaded PKL chunk
            gc.collect()
        # --- End Loop 1 ---

        print(f"\nFinished converting and augmenting. Processed {total_original_graphs_considered} original graphs.")
        print(f"Successfully converted to PyG (before augmentation): {total_successful_conversions} graphs.")
        print(f"Total augmentations created for '{self.split}' split: {total_augmentations_created}.")
        print(f"Total temporary .pt files created: {len(paths_to_temp_pt_files)}")

        # --- Loop 2: Load all temporary .pt files and collate ---
        all_pyg_data_objects_list = []  # This will store ALL Data objects from all temp files

        if paths_to_temp_pt_files:
            print(f"\nLoading data from {len(paths_to_temp_pt_files)} temporary .pt files...")
            for temp_pt_file_path in tqdm(paths_to_temp_pt_files, desc="Loading temp .pt files"):
                try:
                    # Each .pt file contains a list of Data objects
                    list_of_data_objects = torch.load(temp_pt_file_path, weights_only=False)  # Must be False
                    all_pyg_data_objects_list.extend(list_of_data_objects)
                    os.remove(temp_pt_file_path)  # Delete the temporary file after loading
                except Exception as e_load:
                    warnings.warn(f"Could not load or delete temporary file {temp_pt_file_path}: {e_load}")

            try:  # Attempt to remove the temporary directory
                shutil.rmtree(temp_individual_graph_pts_dir)
            except OSError as e_rmdir:
                warnings.warn(
                    f"Could not remove temporary directory {temp_individual_graph_pts_dir}: {e_rmdir}. You may need to remove it manually.")
        else:
            print(f"No temporary .pt files were created for split '{self.split}'. The resulting dataset will be empty.")

        del paths_to_temp_pt_files  # Free memory
        gc.collect()

        print(
            f"\nTotal PyG Data objects (including originals and augmentations) to collate: {len(all_pyg_data_objects_list)}")

        # Apply pre_filter and pre_transform if they are defined
        # These are standard hooks from InMemoryDataset
        if self.pre_filter is not None:
            # Filter the list of Data objects
            all_pyg_data_objects_list = [data for data in all_pyg_data_objects_list if self.pre_filter(data)]
            print(f"Data objects after pre_filter: {len(all_pyg_data_objects_list)}")

        if self.pre_transform is not None:
            # Apply transformation to each Data object
            # This can be memory intensive if pre_transform creates large objects or many copies
            processed_list_for_transform = []
            for data_idx, data_obj in enumerate(tqdm(all_pyg_data_objects_list, desc="Applying pre_transform")):
                processed_list_for_transform.append(self.pre_transform(data_obj))
                if (data_idx + 1) % 500 == 0:  # Manage memory during transform
                    gc.collect()
            all_pyg_data_objects_list = processed_list_for_transform
            del processed_list_for_transform
            gc.collect()
            print(f"Data objects after pre_transform: {len(all_pyg_data_objects_list)}")

        # Collate all Data objects into a single large Data object and corresponding slices
        # This is the standard InMemoryDataset step.
        print(f"Collating {len(all_pyg_data_objects_list)} PyG Data objects...")
        if not all_pyg_data_objects_list:
            warnings.warn(f"No data to collate for split '{self.split}'. Saving an empty dataset structure.")

        # The self.collate method handles the creation of the large 'data' object and 'slices' dictionary
        data, slices = self.collate(all_pyg_data_objects_list)

        del all_pyg_data_objects_list  # Free memory
        gc.collect()

        # Save the final collated data and slices to the path specified by self.processed_paths[0]
        final_save_path = self.processed_paths[0]
        print(f"Saving final processed data for split '{self.split}' to: {final_save_path}")
        torch.save((data, slices), final_save_path)
        print(f"Successfully saved final data for split '{self.split}'.")


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AIG PKL files into augmented PyG InMemoryDataset format.")
    parser.add_argument('--raw_dir', type=str, required=True, help="Directory containing raw PKL files.")
    parser.add_argument('--output_root', type=str, required=True,
                        help="Root directory for dataset output (e.g., ./datasets_output).")
    parser.add_argument('--dataset_name', type=str, default='aig_pyg_dataset',
                        help="Name for the dataset subfolder within output_root (e.g., my_aig_graphs).")
    parser.add_argument('--file_prefix', type=str, default="real_aigs_part_",
                        help="Prefix of raw PKL files to look for.")
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help="Number of augmentations for each graph in the training set. Set to 0 for no augmentations.")
    parser.add_argument('--num_train_files', type=int, default=4,
                        help="Number of PKL files to allocate for the training set.")
    parser.add_argument('--num_val_files', type=int, default=1,
                        help="Number of PKL files to allocate for the validation set.")
    parser.add_argument('--num_test_files', type=int, default=1,
                        help="Number of PKL files to allocate for the test set.")
    args = parser.parse_args()

    # Validate raw directory
    if not osp.isdir(args.raw_dir):
        print(f"Error: Raw directory not found: {args.raw_dir}")
        exit(1)

    # Create output directory if it doesn't exist
    # The InMemoryDataset will create processed_dir inside dataset_name,
    # so we just need to ensure output_root exists.
    os.makedirs(args.output_root, exist_ok=True)
    # The full path for the dataset will be osp.join(args.output_root, args.dataset_name)

    # List and sort PKL files
    try:
        all_available_pkl_files = sorted([
            f for f in os.listdir(args.raw_dir)
            if f.startswith(args.file_prefix) and f.endswith(".pkl")
        ])
    except OSError as e:
        print(f"Error listing files in raw directory '{args.raw_dir}': {e}")
        exit(1)

    if not all_available_pkl_files:
        print(f"No PKL files found in '{args.raw_dir}' with prefix '{args.file_prefix}'.")
        exit(1)

    print(f"Found {len(all_available_pkl_files)} PKL files matching prefix '{args.file_prefix}' in {args.raw_dir}:")
    # for fname in all_available_pkl_files: print(f"  - {fname}")

    # Allocate files to splits
    current_file_idx = 0
    train_files_list, val_files_list, test_files_list = [], [], []

    # Training files
    if args.num_train_files > 0:
        take_n = min(args.num_train_files, len(all_available_pkl_files) - current_file_idx)
        if take_n > 0:
            train_files_list = all_available_pkl_files[current_file_idx: current_file_idx + take_n]
            current_file_idx += take_n
        else:
            warnings.warn("Not enough files to satisfy requested number of training files, or num_train_files is 0.")

    # Validation files
    if args.num_val_files > 0:
        take_n = min(args.num_val_files, len(all_available_pkl_files) - current_file_idx)
        if take_n > 0:
            val_files_list = all_available_pkl_files[current_file_idx: current_file_idx + take_n]
            current_file_idx += take_n
        else:
            warnings.warn(
                "Not enough files to satisfy requested number of validation files after allocating training, or num_val_files is 0.")

    # Test files
    if args.num_test_files > 0:
        take_n = min(args.num_test_files, len(all_available_pkl_files) - current_file_idx)
        if take_n > 0:
            test_files_list = all_available_pkl_files[current_file_idx: current_file_idx + take_n]
            current_file_idx += take_n
        else:
            warnings.warn(
                "Not enough files to satisfy requested number of test files after allocating train/val, or num_test_files is 0.")

    print(f"\nFile Allocation:")
    print(f"  Train ({len(train_files_list)} files): {train_files_list if train_files_list else 'None'}")
    print(f"  Val   ({len(val_files_list)} files): {val_files_list if val_files_list else 'None'}")
    print(f"  Test  ({len(test_files_list)} files): {test_files_list if test_files_list else 'None'}")

    if current_file_idx < len(all_available_pkl_files):
        warnings.warn(f"{len(all_available_pkl_files) - current_file_idx} PKL files remain unallocated.")

    # --- Process Splits by Instantiating Dataset ---
    # The __init__ method of AIGProcessedAugmentedDataset will trigger processing if needed.

    if train_files_list:
        print(
            f"\n--- Initializing/Processing Training Set ({args.num_augmentations} augs/graph if split is 'train') ---")
        # For training, pass the specified number of augmentations
        train_dataset = AIGProcessedAugmentedDataset(
            root=args.output_root,
            raw_dir=args.raw_dir,
            split='train',
            file_prefix=args.file_prefix,  # Needed for context if process is called
            pkl_file_names_for_split=train_files_list,
            dataset_name=args.dataset_name,
            num_augmentations=args.num_augmentations
        )
        del train_dataset;
        gc.collect()  # Release memory if dataset object is not needed further
    else:
        print("\nSkipping Training Set processing as no files were allocated.")

    if val_files_list:
        print(f"\n--- Initializing/Processing Validation Set (0 augs/graph) ---")
        # For validation, augmentations are typically 0
        val_dataset = AIGProcessedAugmentedDataset(
            root=args.output_root,
            raw_dir=args.raw_dir,
            split='val',
            file_prefix=args.file_prefix,
            pkl_file_names_for_split=val_files_list,
            dataset_name=args.dataset_name,
            num_augmentations=0  # No augmentations for validation
        )
        del val_dataset;
        gc.collect()
    else:
        print("\nSkipping Validation Set processing as no files were allocated.")

    if test_files_list:
        print(f"\n--- Initializing/Processing Test Set (0 augs/graph) ---")
        # For test, augmentations are typically 0
        test_dataset = AIGProcessedAugmentedDataset(
            root=args.output_root,
            raw_dir=args.raw_dir,
            split='test',
            file_prefix=args.file_prefix,
            pkl_file_names_for_split=test_files_list,
            dataset_name=args.dataset_name,
            num_augmentations=0  # No augmentations for test
        )
        del test_dataset;
        gc.collect()
    else:
        print("\nSkipping Test Set processing as no files were allocated.")

    print("\nAll specified dataset processing finished (or skipped if no files).")
    print(
        f"Processed data (if any) should be in subdirectories under: {osp.join(args.output_root, args.dataset_name, 'processed')}")
    print("Example: training data at:",
          osp.join(args.output_root, args.dataset_name, 'processed', 'train_processed_data.pt'))
