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
from typing import List, Tuple, Dict, Any, Deque, Optional # Added Optional
from collections import Counter, deque
import random

# --- AIG Model/Data Parameters ---
# Ensure these match your GraphDF/GraphAF/GraphEBM model expectations.
MAX_NODES_PAD = 64             # Max nodes for padding (model.max_size)
NUM_NODE_FEATURES = 4      # Node types: CONST0, PI, AND, PO (one-hot) (model.node_dim)
NUM_EXPLICIT_EDGE_TYPES = 2 # Edge types: REG, INV
NUM_ADJ_CHANNELS = NUM_EXPLICIT_EDGE_TYPES + 1 # model.bond_dim
# --- End AIG Parameters ---

# --- User's Custom Randomized Topological Sort ---
def custom_randomized_topological_sort(G: nx.DiGraph, random_generator: random.Random) -> List[int]:
    """
    Performs a topological sort, randomizing the order of nodes
    that have the same in-degree at each step.
    Uses the provided random_generator instance.
    Raises NetworkXUnfeasible if a cycle is detected.
    """
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined for undirected graphs.")

    in_degree_map = {node: degree for node, degree in G.in_degree()}
    zero_in_degree_nodes = [node for node, degree in in_degree_map.items() if degree == 0]

    if len(zero_in_degree_nodes) > 1:
        random_generator.shuffle(zero_in_degree_nodes)

    queue: Deque[int] = deque(zero_in_degree_nodes)
    result_order: List[int] = []

    while queue:
        u = queue.popleft()
        result_order.append(u)

        newly_zero_in_degree: List[int] = []
        successors_of_u = [succ for succ in G.successors(u) if succ in G]
        for v in sorted(list(successors_of_u)):
            in_degree_map[v] -= 1
            if in_degree_map[v] == 0:
                newly_zero_in_degree.append(v)
            elif in_degree_map[v] < 0:
                raise RuntimeError(f"In-degree became negative for node {v} during topological sort.")

        if len(newly_zero_in_degree) > 1:
            random_generator.shuffle(newly_zero_in_degree)

        for node in newly_zero_in_degree:
            queue.append(node)

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
    """
    Converts a single NetworkX graph (from PKL) to a PyTorch Geometric Data object
    with padding for dense processing. Adjacency matrix follows PyG convention
    [Channel, Target, Source].
    """
    num_nodes_in_graph = graph.number_of_nodes()

    if num_nodes_in_graph == 0:
        # warnings.warn("Skipping graph with 0 nodes during PyG conversion.")
        return None
    if num_nodes_in_graph > max_nodes_pad:
        warnings.warn(f"Skipping graph with {num_nodes_in_graph} nodes (max allowed: {max_nodes_pad}). Graph rejected.")
        return None

    try:
        node_list = sorted(list(graph.nodes()))
    except TypeError:
        node_list = list(graph.nodes())
        warnings.warn("Graph nodes were not sortable, using arbitrary order for mapping to PyG.")

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}

    node_features_list = []
    valid_nodes = True
    for old_node_id in node_list:
        attrs = graph.nodes[old_node_id]
        node_type_vec_raw = attrs.get('type')

        if node_type_vec_raw is None or not isinstance(node_type_vec_raw, (list, np.ndarray)) or len(
                node_type_vec_raw) != num_node_features:
            warnings.warn(
                f"Node {old_node_id} has invalid 'type' attribute (val: {node_type_vec_raw}, len: {len(node_type_vec_raw) if hasattr(node_type_vec_raw, '__len__') else 'N/A'}, expected_len: {num_node_features}). Skipping graph.")
            valid_nodes = False;
            break
        try:
            node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
            # Use isclose for float comparisons
            if not (np.isclose(np.sum(node_type_vec), 1.0) and np.all(
                    (np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
                warnings.warn(
                    f"Node {old_node_id} 'type' attribute {node_type_vec_raw} is not a valid one-hot vector. Sum: {np.sum(node_type_vec)}. Skipping graph.")
                valid_nodes = False;
                break
            node_feature_tensor = torch.tensor(node_type_vec, dtype=torch.float)
            node_features_list.append(node_feature_tensor)
        except Exception as e:
            warnings.warn(
                f"Error processing 'type' for node {old_node_id} ('{node_type_vec_raw}'): {e}. Skipping graph.")
            valid_nodes = False;
            break
    if not valid_nodes or not node_features_list: return None

    x_stacked = torch.stack(node_features_list)
    num_padding_nodes = max_nodes_pad - num_nodes_in_graph
    x_padded = F.pad(x_stacked, (0, 0, 0, num_padding_nodes), "constant", 0.0)

    adj_matrix = torch.zeros((num_adj_channels, max_nodes_pad, max_nodes_pad), dtype=torch.float)
    adj_matrix[num_explicit_edge_types, :, :] = 1.0  # Initialize NO_EDGE channel

    valid_edges = True
    for u_old, v_old, attrs in graph.edges(data=True):
        edge_type_vec_raw = attrs.get('type')
        if edge_type_vec_raw is None or not isinstance(edge_type_vec_raw, (list, np.ndarray)) or len(
                edge_type_vec_raw) != num_explicit_edge_types:
            warnings.warn(f"Edge ({u_old}-{v_old}) has invalid 'type' attribute. Skipping graph.")
            valid_edges = False;
            break

        u_new, v_new = node_id_map.get(u_old), node_id_map.get(v_old)
        if u_new is None or v_new is None or not (0 <= u_new < num_nodes_in_graph and 0 <= v_new < num_nodes_in_graph):
            # warnings.warn(f"Edge ({u_old}->{v_old} mapped to {u_new}->{v_new}) has out-of-bounds index. Skipping edge.")
            continue # Just skip the edge if nodes aren't valid

        try:
            edge_type_vec = np.asarray(edge_type_vec_raw, dtype=np.float32)
            if not (np.isclose(np.sum(edge_type_vec), 1.0) and np.all(
                    (np.isclose(edge_type_vec, 0.0)) | (np.isclose(edge_type_vec, 1.0)))):
                warnings.warn(f"Edge ({u_old}-{v_old}) 'type' {edge_type_vec_raw} not valid one-hot. Skipping graph.")
                valid_edges = False;
                break
            edge_channel_index = np.argmax(edge_type_vec).item()
            if not (0 <= edge_channel_index < num_explicit_edge_types):
                warnings.warn(
                    f"Edge ({u_old}-{v_old}) type {edge_type_vec_raw} invalid channel index {edge_channel_index}. Skipping graph.")
                valid_edges = False;
                break

            # Adjacency: [Channel, Target, Source]
            adj_matrix[edge_channel_index, v_new, u_new] = 1.0
            adj_matrix[num_explicit_edge_types, v_new, u_new] = 0.0 # Mark as not NO_EDGE
        except Exception as e:
            warnings.warn(f"Error processing 'type' for edge ({u_old}-{v_old}): {e}. Skipping graph.")
            valid_edges = False;
            break
    if not valid_edges: return None

    # Ensure no self-loops in any channel
    no_edge_channel_idx = num_explicit_edge_types
    for k_node_diag in range(max_nodes_pad):
        adj_matrix[no_edge_channel_idx, k_node_diag, k_node_diag] = 0.0
        for ch in range(num_explicit_edge_types):
            adj_matrix[ch, k_node_diag, k_node_diag] = 0.0

    # Create Data object
    data = Data(x=x_padded, adj=adj_matrix, num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long))

    # Add graph-level attributes if they exist
    if 'inputs' in graph.graph: data.num_inputs = torch.tensor(graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in graph.graph: data.num_outputs = torch.tensor(graph.graph['outputs'], dtype=torch.long)
    if 'gates' in graph.graph: data.num_gates = torch.tensor(graph.graph['gates'], dtype=torch.long)
    # Handle output_patterns carefully - only add if convertible
    if 'output_patterns' in graph.graph and isinstance(graph.graph['output_patterns'], list):
        try:
            # Attempt conversion, assuming patterns are padded lists of numbers
            patterns_tensor = torch.tensor(graph.graph['output_patterns'], dtype=torch.float) # Or long if ints
            data.output_patterns = patterns_tensor
        except Exception as e:
            warnings.warn(f"Could not convert 'output_patterns' to tensor: {e}")

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

        Args:
            root (str): Root directory where the dataset folder (`dataset_name`) resides
                        or will be created.
            dataset_name (str): Name of the dataset folder (e.g., 'aig_ds').
            split (str): Data split: 'train', 'val', or 'test'.
            raw_dir (Optional[str]): Directory containing raw PKL files.
                                     Required only if processed data doesn't exist.
            file_prefix (Optional[str]): Prefix of raw PKL files.
                                         Required only if processed data doesn't exist.
            pkl_file_names_for_split (Optional[List[str]]): Specific PKL filenames for this split.
                                                             Required only if processed data doesn't exist.
            num_augmentations (int): Number of augmentations to apply during processing.
                                     Set to 0 for validation/test sets typically.
            transform: PyG transforms applied after loading.
            pre_transform: PyG transforms applied before saving processed data.
            pre_filter: PyG pre-filtering applied before saving processed data.
        """
        self.dataset_name = dataset_name
        self.split = split
        # Store raw info - only used if self.process() is called
        self.raw_dir = raw_dir
        self.file_prefix = file_prefix
        self._raw_file_names_for_this_split = pkl_file_names_for_split if pkl_file_names_for_split is not None else []
        self.num_augmentations = num_augmentations

        # The root directory for InMemoryDataset should contain the 'processed' folder
        processed_root = osp.join(root, dataset_name)

        # Call super().__init__ FIRST. It checks processed_paths and calls self.process() if needed.
        super().__init__(processed_root, transform, pre_transform, pre_filter)

        # After super().__init__, if processed files existed, they are loaded.
        # If not, self.process() was called, and then files are loaded.
        # We load data/slices here regardless, assuming super() handled it.
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Dataset '{self.dataset_name}' split '{self.split}' initialized. Samples: {len(self)}")
        except FileNotFoundError:
             # This happens if process() was needed but failed to create the file,
             # or if the file got deleted after processing.
             raise FileNotFoundError(f"Processed file not found at {self.processed_paths[0]}. "
                                     "Ensure processing completed successfully or provide raw data args.")
        except Exception as e:
            raise RuntimeError(f"Failed to load processed data from {self.processed_paths[0]}: {e}")

    @property
    def raw_file_names(self) -> List[str]:
        """ Returns the list of raw filenames for this split. Used by InMemoryDataset if processing. """
        # Returns the base filenames, not full paths
        return self._raw_file_names_for_this_split

    # raw_paths property is used by self.process()
    @property
    def raw_paths(self) -> List[str]:
        """ Returns a list of absolute paths to the raw files for this split. """
        if not self.raw_dir: return [] # Needed if called before process() checks args
        return [osp.join(self.raw_dir, name) for name in self.raw_file_names]

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        """ Returns the name(s) of the processed file(s). """
        # Defines the filename saved within self.processed_dir
        return [f'{self.split}_augmented_data.pt']

    def download(self):
        """ Download dataset. Not needed here as PKL files are local. """
        pass

    def process(self):
        """ Processes raw PKL files, applies augmentation, and saves PyG data. """
        # Check if necessary raw data information was provided
        if self.raw_dir is None or self.file_prefix is None or not self._raw_file_names_for_this_split:
            raise ValueError("Raw directory, file prefix, and file list are required for processing, but not provided.")

        print(f"Processing raw PKL files and augmenting for split: {self.split}...")
        all_data_for_split = []
        original_graphs_processed = 0
        successful_conversions = 0
        augmentations_created = 0

        # Iterate through the PKL files assigned to this split
        for pkl_file_idx, raw_path in enumerate(tqdm(self.raw_paths, desc=f"Processing PKL files for {self.split}")):
            if not osp.exists(raw_path):
                warnings.warn(f"Raw file not found: {raw_path}. Skipping.")
                continue
            try:
                with open(raw_path, 'rb') as f: nx_graphs_chunk = pickle.load(f)
                if not isinstance(nx_graphs_chunk, list):
                    warnings.warn(f"File {raw_path} does not contain a list. Skipping."); continue
            except Exception as e:
                warnings.warn(f"Could not load {raw_path}: {e}. Skipping file."); continue

            # Process each NetworkX graph in the chunk
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
                    # Add the original (unaugmented) graph
                    all_data_for_split.append(base_pyg_data.clone())

                    # --- Apply Augmentations ---
                    num_actual_nodes = base_pyg_data.num_nodes.item()
                    if num_actual_nodes > 0 and self.num_augmentations > 0:
                        try: # Wrap graph reconstruction and augmentation in try-except
                            # Reconstruct temporary nx graph for topo sort
                            temp_nx_graph = nx.DiGraph()
                            for i in range(num_actual_nodes): temp_nx_graph.add_node(i)
                            for ch in range(NUM_EXPLICIT_EDGE_TYPES):
                                adj_channel = base_pyg_data.adj[ch, :num_actual_nodes, :num_actual_nodes]
                                sources, targets = adj_channel.nonzero(as_tuple=True)
                                for src, tgt in zip(sources.tolist(), targets.tolist()):
                                    # Adj is [Channel, Target, Source], so edge is src -> tgt
                                    temp_nx_graph.add_edge(src, tgt)

                            # Apply augmentations
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

                                        # Permute features and adjacency matrix
                                        augmented_data.x = augmented_data.x[full_perm_tensor]
                                        # Permute rows (target) then columns (source) for each channel
                                        augmented_data.adj = augmented_data.adj[:, full_perm_tensor][:, :, full_perm_tensor]

                                        all_data_for_split.append(augmented_data)
                                        augmentations_created += 1
                                    else:
                                        warnings.warn(f"Topo sort mismatch (Aug {aug_idx}, Graph {original_graphs_processed - 1}). Skipping aug.")
                                except nx.NetworkXUnfeasible:
                                    warnings.warn(f"Cycle detected (Aug {aug_idx}, Graph {original_graphs_processed - 1}). Skipping aug.")
                                except Exception as e:
                                    warnings.warn(f"Augmentation error (Aug {aug_idx}, Graph {original_graphs_processed - 1}): {e}")
                        except Exception as recon_e:
                             warnings.warn(f"Error reconstructing graph {original_graphs_processed - 1} for augmentation: {recon_e}")


            del nx_graphs_chunk; gc.collect() # Clean up memory

        # --- Final Logging and Saving ---
        print(f"\nFinished PKL processing and augmentation for split '{self.split}'.")
        print(f"Total original graphs considered: {original_graphs_processed}")
        print(f"Successfully converted to PyG Data (before augmentation): {successful_conversions}")
        print(f"Total augmentations created: {augmentations_created}")
        print(f"Total samples saved for this split (original + augmentations): {len(all_data_for_split)}")

        if not all_data_for_split:
            warnings.warn(f"No data processed for split '{self.split}'. Saving empty file.")
            # Create empty data structure to avoid errors on load
            data, slices = self.collate([])
        else:
            data, slices = self.collate(all_data_for_split)

        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        save_path = self.processed_paths[0]
        torch.save((data, slices), save_path)
        print(f"Saved processed data for split '{self.split}' to: {save_path}")

    # get() and len() are inherited from InMemoryDataset and work correctly
    # after self.data and self.slices are loaded/created.


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
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='train',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=train_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=args.num_augmentations)
    if val_files_list:
        print(f"\n--- Initializing/Processing Validation Set (0 augs/graph) ---")
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='val',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=val_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=0) # No augmentation for validation
    if test_files_list:
        print(f"\n--- Initializing/Processing Test Set (0 augs/graph) ---")
        AIGProcessedAugmentedDataset(root=args.output_root, raw_dir=args.raw_dir, split='test',
                                     file_prefix=args.file_prefix, pkl_file_names_for_split=test_files_list,
                                     dataset_name=args.dataset_name, num_augmentations=0) # No augmentation for test

    print("\nAll specified dataset processing finished.")
    print(f"Processed data should be in subdirs under: {osp.join(args.output_root, args.dataset_name, 'processed')}")
