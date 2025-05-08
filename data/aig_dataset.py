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
import time  # For detailed timing

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
    in_degree_map = {n: d for n, d in G.in_degree() if n in G}

    for node in list(G.nodes()):
        if node not in in_degree_map:
            if G.in_degree(node) == 0:
                in_degree_map[node] = 0
            else:
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

        successors_of_u = sorted([succ for succ in G.successors(u) if succ in G and succ in in_degree_map])
        newly_zero_in_degree: List[int] = []
        for v in successors_of_u:
            in_degree_map[v] -= 1
            if in_degree_map[v] == 0:
                newly_zero_in_degree.append(v)
            elif in_degree_map[v] < 0:
                raise RuntimeError(
                    f"Negative in-degree for node {v} encountered during topological sort. Current order: {result_order}")

        if len(newly_zero_in_degree) > 1: random_generator.shuffle(newly_zero_in_degree)
        for node in newly_zero_in_degree: queue.append(node)

    if processed_nodes_count != G.number_of_nodes():
        remaining_nodes = set(G.nodes()) - set(result_order)
        cycle_nodes_or_unreachable = {node for node, degree in in_degree_map.items() if
                                      degree > 0 and node in G and node in remaining_nodes}
        try:
            # Check for cycle in the subgraph of remaining nodes to give a more specific error
            subgraph_of_remaining = G.subgraph(remaining_nodes)
            cycle = nx.find_cycle(subgraph_of_remaining)
            raise nx.NetworkXUnfeasible(
                f"Graph contains a cycle. Example cycle in remaining nodes: {cycle}. Processed: {len(result_order)}/{G.number_of_nodes()}. Remaining nodes possibly in cycle: {cycle_nodes_or_unreachable}")
        except nx.NetworkXNoCycle:
            # If no cycle is found in the remaining subgraph, it might be a disconnected component issue
            # or an issue with nodes not being processed correctly.
            raise nx.NetworkXUnfeasible(
                f"Topological sort failed. Graph might be disconnected or have other structural issues not forming a simple cycle in the remainder. Processed: {len(result_order)}/{G.number_of_nodes()}. Remaining nodes: {remaining_nodes}. Nodes with positive in-degree in remainder: {cycle_nodes_or_unreachable}")
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
        return None
    if num_nodes_in_graph > max_nodes_pad:
        warnings.warn(f"Skipping graph with {num_nodes_in_graph} nodes (> {max_nodes_pad}).")
        return None

    try:
        node_list = sorted(list(graph.nodes()))
    except TypeError:
        node_list = list(graph.nodes())
        warnings.warn(
            "Nodes could not be sorted, using arbitrary order. This might lead to inconsistencies if node IDs are not integers.")

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}
    node_features_list = []
    for old_node_id in node_list:
        attrs = graph.nodes[old_node_id]
        node_type_vec_raw = attrs.get('type')
        if node_type_vec_raw is None or not isinstance(node_type_vec_raw, (list, np.ndarray)) or len(
                node_type_vec_raw) != num_node_features:
            warnings.warn(f"Node {old_node_id} has invalid type attribute (length or existence). Skipping graph.")
            return None
        try:
            node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
            if not (np.isclose(np.sum(node_type_vec), 1.0) and np.all(
                    (np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
                warnings.warn(
                    f"Node {old_node_id} type vector is not one-hot. Sum: {np.sum(node_type_vec)}. Values: {node_type_vec}. Skipping graph.")
                return None
            node_features_list.append(torch.tensor(node_type_vec, dtype=torch.float))
        except Exception as e:
            warnings.warn(f"Node {old_node_id} type conversion to tensor failed: {e}. Skipping graph.")
            return None

    if not node_features_list:
        return None

    x_stacked = torch.stack(node_features_list)
    num_padding_nodes = max_nodes_pad - num_nodes_in_graph
    x_padded = F.pad(x_stacked, (0, 0, 0, num_padding_nodes), "constant", 0.0)
    adj_matrix = torch.zeros((num_adj_channels, max_nodes_pad, max_nodes_pad), dtype=torch.float)
    adj_matrix[num_explicit_edge_types, :num_nodes_in_graph, :num_nodes_in_graph] = 1.0

    for u_old, v_old, attrs in graph.edges(data=True):
        edge_type_vec_raw = attrs.get('type')
        if edge_type_vec_raw is None or not isinstance(edge_type_vec_raw, (list, np.ndarray)) or len(
                edge_type_vec_raw) != num_explicit_edge_types:
            warnings.warn(f"Edge ({u_old}-{v_old}) has invalid type attribute. Skipping graph.")
            return None

        u_new, v_new = node_id_map.get(u_old), node_id_map.get(v_old)
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

            edge_channel_index = np.argmax(edge_type_vec).item()
            if not (0 <= edge_channel_index < num_explicit_edge_types):
                warnings.warn(
                    f"Edge ({u_old}-{v_old}) type vector resulted in invalid channel index: {edge_channel_index}. Skipping graph.")
                return None

            adj_matrix[edge_channel_index, v_new, u_new] = 1.0
            adj_matrix[num_explicit_edge_types, v_new, u_new] = 0.0
        except Exception as e:
            warnings.warn(f"Edge ({u_old}-{v_old}) type conversion or assignment failed: {e}. Skipping graph.")
            return None

    for k_node_diag in range(max_nodes_pad):
        for ch in range(num_adj_channels):
            adj_matrix[ch, k_node_diag, k_node_diag] = 0.0

    data = Data(x=x_padded, adj=adj_matrix, num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long))
    if 'inputs' in graph.graph: data.num_inputs = torch.tensor(graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in graph.graph: data.num_outputs = torch.tensor(graph.graph['outputs'], dtype=torch.long)
    if 'gates' in graph.graph: data.num_gates = torch.tensor(graph.graph['gates'], dtype=torch.long)
    return data


class AIGProcessedAugmentedDataset(InMemoryDataset):
    def __init__(self, root: str, dataset_name: str, split: str,
                 raw_dir: Optional[str] = None,
                 file_prefix: Optional[str] = None,
                 pkl_file_names_for_split: Optional[List[str]] = None,
                 # num_augmentations parameter removed
                 transform=None, pre_transform=None, pre_filter=None):

        self._temp_raw_dir = raw_dir
        self._temp_file_prefix = file_prefix
        self._temp_pkl_files = pkl_file_names_for_split if pkl_file_names_for_split is not None else []
        self.dataset_name = dataset_name
        self.split = split
        # self.num_augmentations removed

        processed_dataset_root = osp.join(root, dataset_name)
        super().__init__(processed_dataset_root, transform, pre_transform, pre_filter)

        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            print(f"Dataset '{self.dataset_name}' split '{self.split}' successfully loaded. Samples: {len(self)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Processed file not found: {self.processed_paths[0]}. Processing might have failed or was skipped.")
        except Exception as e:
            raise RuntimeError(f"Failed to load processed data from {self.processed_paths[0]}: {e}")

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
        return [f'{self.split}_processed_data.pt']

    def download(self):
        pass

    def process(self):
        """ Processes graphs with a single topological sort and reordering. """
        if self._temp_raw_dir is None or not self._temp_pkl_files:
            if not self.raw_paths:
                print(
                    f"No raw files specified for split '{self.split}'. Skipping processing and creating empty dataset file.")
                empty_data_list = []
                if self.pre_filter is not None: empty_data_list = [d for d in empty_data_list if self.pre_filter(d)]
                if self.pre_transform is not None: empty_data_list = [self.pre_transform(d) for d in empty_data_list]
                data, slices = self.collate(empty_data_list)
                torch.save((data, slices), self.processed_paths[0])
                print(f"Saved empty dataset for split '{self.split}' to: {self.processed_paths[0]}")
                return
            raise ValueError("Raw directory and file list are required for processing if raw_paths is not empty.")

        print(
            f"Starting processing for split: {self.split} (each graph will undergo one topological sort and reordering)...")

        temp_individual_graph_pts_dir = osp.join(self.processed_dir, f"{self.split}_temp_individual_pts")
        os.makedirs(temp_individual_graph_pts_dir, exist_ok=True)
        print(f"Temporary storage for individual graph .pt files: {temp_individual_graph_pts_dir}")

        paths_to_temp_pt_files = []
        total_original_graphs_considered = 0
        total_successful_conversions = 0
        # total_augmentations_created removed
        temp_file_counter = 0

        print(f"Beginning to iterate through {len(self.raw_paths)} PKL files for split '{self.split}'.")
        for pkl_file_idx, raw_pkl_path in enumerate(self.raw_paths):
            pkl_start_time = time.time()
            print(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing PKL file {pkl_file_idx + 1}/{len(self.raw_paths)}: {raw_pkl_path}")

            if not osp.exists(raw_pkl_path):
                warnings.warn(f"Raw PKL file not found: {raw_pkl_path}. Skipping.")
                continue

            nx_graphs_in_chunk = []
            try:
                print(f"  Loading PKL file: {raw_pkl_path}...")
                load_start_time = time.time()
                with open(raw_pkl_path, 'rb') as f:
                    nx_graphs_in_chunk = pickle.load(f)
                load_end_time = time.time()
                print(f"  Successfully loaded PKL file in {load_end_time - load_start_time:.2f} seconds.")

                if not isinstance(nx_graphs_in_chunk, list):
                    warnings.warn(f"  Content of {raw_pkl_path} is not a list. Skipping this file.")
                    continue
                print(f"  Found {len(nx_graphs_in_chunk)} graphs in {raw_pkl_path}.")
            except Exception as e:
                warnings.warn(f"  Error loading or reading PKL file {raw_pkl_path}: {e}. Skipping this file.")
                continue

            for nx_graph_idx, nx_graph in enumerate(
                    tqdm(nx_graphs_in_chunk, desc=f"  Graphs in {osp.basename(raw_pkl_path)}", unit="graph")):
                # graph_process_start_time = time.time() # Uncomment for detailed timing per graph
                if not isinstance(nx_graph, nx.DiGraph):
                    warnings.warn(
                        f"    Item at index {nx_graph_idx} in {raw_pkl_path} is not a NetworkX DiGraph. Skipping.")
                    continue

                total_original_graphs_considered += 1
                pyg_data_objects_for_current_nx_graph = []

                base_pyg_data = _convert_nx_to_pyg_data(
                    nx_graph, MAX_NODES_PAD, NUM_NODE_FEATURES,
                    NUM_ADJ_CHANNELS, NUM_EXPLICIT_EDGE_TYPES)

                if base_pyg_data is not None:
                    total_successful_conversions += 1
                    num_actual_nodes = base_pyg_data.num_nodes.item()

                    if num_actual_nodes > 0:
                        # print(f"      Attempting topological sort and reorder for graph {nx_graph_idx+1}...") # Detailed log
                        # reorder_start_time = time.time() # Detailed timing
                        try:
                            temp_nx_for_sort = nx.DiGraph()
                            for i in range(num_actual_nodes): temp_nx_for_sort.add_node(i)
                            for ch_idx in range(NUM_EXPLICIT_EDGE_TYPES):
                                adj_channel_for_actual_nodes = base_pyg_data.adj[ch_idx, :num_actual_nodes,
                                                               :num_actual_nodes]
                                source_indices, target_indices = adj_channel_for_actual_nodes.nonzero(as_tuple=True)
                                for src_node_idx, tgt_node_idx in zip(source_indices.tolist(), target_indices.tolist()):
                                    temp_nx_for_sort.add_edge(tgt_node_idx, src_node_idx)  # adj is [ch, v, u] for u->v

                            # Seed for deterministic sort for this specific graph
                            sort_seed = (pkl_file_idx * len(nx_graphs_in_chunk)) + nx_graph_idx
                            local_random_generator = random.Random(sort_seed)

                            ordered_node_indices = custom_randomized_topological_sort(temp_nx_for_sort,
                                                                                      local_random_generator)

                            if len(ordered_node_indices) == num_actual_nodes:
                                permutation_for_actual_nodes = np.array(ordered_node_indices, dtype=np.int64)
                                permutation_for_padding_nodes = np.arange(num_actual_nodes, MAX_NODES_PAD,
                                                                          dtype=np.int64)
                                full_permutation = np.concatenate(
                                    [permutation_for_actual_nodes, permutation_for_padding_nodes])
                                full_permutation_tensor = torch.from_numpy(full_permutation).long()

                                # Apply permutation directly to base_pyg_data's tensors
                                base_pyg_data.x = base_pyg_data.x[full_permutation_tensor]
                                base_pyg_data.adj = base_pyg_data.adj[:, full_permutation_tensor][:, :,
                                                    full_permutation_tensor]
                                # pyg_data_objects_for_current_nx_graph.append(base_pyg_data) # Will be added below
                            else:
                                warnings.warn(
                                    f"      Topological sort for graph {nx_graph_idx} resulted in {len(ordered_node_indices)} nodes, expected {num_actual_nodes}. Using original order.")

                        except nx.NetworkXUnfeasible as e_unfeasible:
                            warnings.warn(
                                f"      Topological sort unfeasible for graph {nx_graph_idx} (e.g., cycle: {e_unfeasible}). Using original order.")
                        except Exception as e_sort:
                            warnings.warn(
                                f"      Error during topological sort/reordering for graph {nx_graph_idx}: {e_sort}. Using original order.")
                        # print(f"      Sort/reorder attempt took {time.time() - reorder_start_time:.4f}s") # Detailed timing

                    # Add the (potentially reordered) base_pyg_data
                    pyg_data_objects_for_current_nx_graph.append(base_pyg_data)

                if pyg_data_objects_for_current_nx_graph:  # Should always be true if base_pyg_data was not None
                    temp_pt_file_path = osp.join(temp_individual_graph_pts_dir, f"graph_batch_{temp_file_counter}.pt")
                    try:
                        torch.save(pyg_data_objects_for_current_nx_graph,
                                   temp_pt_file_path)  # Saves a list with one item
                        paths_to_temp_pt_files.append(temp_pt_file_path)
                        temp_file_counter += 1
                    except Exception as e_save:
                        warnings.warn(f"      Could not save temporary batch file {temp_pt_file_path}: {e_save}")

                del pyg_data_objects_for_current_nx_graph, base_pyg_data
                # if (nx_graph_idx + 1) % 50 == 0: gc.collect() # Periodic GC

            del nx_graphs_in_chunk
            gc.collect()
            pkl_end_time = time.time()
            print(
                f"  Finished processing PKL file {osp.basename(raw_pkl_path)} in {pkl_end_time - pkl_start_time:.2f} seconds.")
        # --- End Loop 1 ---

        print(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished converting and reordering all PKL files for split '{self.split}'.")
        print(f"Total original graphs considered: {total_original_graphs_considered}")
        print(f"Successfully converted to PyG (and potentially reordered): {total_successful_conversions}")
        # total_augmentations_created print removed
        print(f"Total temporary .pt files created: {len(paths_to_temp_pt_files)}")

        all_pyg_data_objects_list = []
        if paths_to_temp_pt_files:
            print(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {len(paths_to_temp_pt_files)} temporary .pt files...")
            for temp_pt_file_path in tqdm(paths_to_temp_pt_files, desc="Loading temp .pt files"):
                try:
                    # Each .pt file contains a list of Data objects (in this case, a list with one Data object)
                    list_of_data_objects = torch.load(temp_pt_file_path, weights_only=False)
                    all_pyg_data_objects_list.extend(list_of_data_objects)
                    os.remove(temp_pt_file_path)
                except Exception as e_load:
                    warnings.warn(f"Could not load or delete temporary file {temp_pt_file_path}: {e_load}")

            try:
                shutil.rmtree(temp_individual_graph_pts_dir)
                print(f"Successfully removed temporary directory: {temp_individual_graph_pts_dir}")
            except OSError as e_rmdir:
                warnings.warn(
                    f"Could not remove temporary directory {temp_individual_graph_pts_dir}: {e_rmdir}. You may need to remove it manually.")
        else:
            print(f"No temporary .pt files were created for split '{self.split}'. The resulting dataset will be empty.")

        del paths_to_temp_pt_files
        gc.collect()

        print(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total PyG Data objects to collate: {len(all_pyg_data_objects_list)}")

        if self.pre_filter is not None:
            filter_start_time = time.time()
            all_pyg_data_objects_list = [data for data in all_pyg_data_objects_list if self.pre_filter(data)]
            print(
                f"Data objects after pre_filter: {len(all_pyg_data_objects_list)}. Took {time.time() - filter_start_time:.2f}s.")

        if self.pre_transform is not None:
            transform_start_time = time.time()
            processed_list_for_transform = []
            for data_idx, data_obj in enumerate(tqdm(all_pyg_data_objects_list, desc="Applying pre_transform")):
                processed_list_for_transform.append(self.pre_transform(data_obj))
                if (data_idx + 1) % 500 == 0: gc.collect()
            all_pyg_data_objects_list = processed_list_for_transform
            del processed_list_for_transform
            gc.collect()
            print(
                f"Data objects after pre_transform: {len(all_pyg_data_objects_list)}. Took {time.time() - transform_start_time:.2f}s.")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Collating {len(all_pyg_data_objects_list)} PyG Data objects...")
        collate_start_time = time.time()
        if not all_pyg_data_objects_list:
            warnings.warn(f"No data to collate for split '{self.split}'. Saving an empty dataset structure.")

        data, slices = self.collate(all_pyg_data_objects_list)
        print(f"Collation finished in {time.time() - collate_start_time:.2f}s.")

        del all_pyg_data_objects_list
        gc.collect()

        final_save_path = self.processed_paths[0]
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving final processed data for split '{self.split}' to: {final_save_path}")
        save_final_start_time = time.time()
        torch.save((data, slices), final_save_path)
        print(
            f"Successfully saved final data for split '{self.split}'. Took {time.time() - save_final_start_time:.2f}s.")


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process AIG PKL files into PyG InMemoryDataset format with topological sort reordering.")
    parser.add_argument('--raw_dir', type=str, required=True, help="Directory containing raw PKL files.")
    parser.add_argument('--output_root', type=str, required=True,
                        help="Root directory for dataset output (e.g., ./datasets_output).")
    parser.add_argument('--dataset_name', type=str, default='aig_pyg_dataset',
                        help="Name for the dataset subfolder within output_root (e.g., my_aig_graphs).")
    parser.add_argument('--file_prefix', type=str, default="real_aigs_part_",
                        help="Prefix of raw PKL files to look for.")
    # num_augmentations argument removed
    parser.add_argument('--num_train_files', type=int, default=4,
                        help="Number of PKL files to allocate for the training set.")
    parser.add_argument('--num_val_files', type=int, default=1,
                        help="Number of PKL files to allocate for the validation set.")
    parser.add_argument('--num_test_files', type=int, default=1,
                        help="Number of PKL files to allocate for the test set.")
    args = parser.parse_args()

    if not osp.isdir(args.raw_dir):
        print(f"Error: Raw directory not found: {args.raw_dir}")
        exit(1)
    os.makedirs(args.output_root, exist_ok=True)

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

    current_file_idx = 0
    train_files_list, val_files_list, test_files_list = [], [], []

    if args.num_train_files > 0:
        take_n = min(args.num_train_files, len(all_available_pkl_files) - current_file_idx)
        if take_n > 0:
            train_files_list = all_available_pkl_files[
                               current_file_idx: current_file_idx + take_n]; current_file_idx += take_n
        else:
            warnings.warn("Not enough files for training set or num_train_files is 0.")

    if args.num_val_files > 0:
        take_n = min(args.num_val_files, len(all_available_pkl_files) - current_file_idx)
        if take_n > 0:
            val_files_list = all_available_pkl_files[
                             current_file_idx: current_file_idx + take_n]; current_file_idx += take_n
        else:
            warnings.warn("Not enough files for validation set or num_val_files is 0.")

    if args.num_test_files > 0:
        take_n = min(args.num_test_files, len(all_available_pkl_files) - current_file_idx)
        if take_n > 0:
            test_files_list = all_available_pkl_files[
                              current_file_idx: current_file_idx + take_n]; current_file_idx += take_n
        else:
            warnings.warn("Not enough files for test set or num_test_files is 0.")

    print(f"\nFile Allocation:")
    print(f"  Train ({len(train_files_list)} files): {train_files_list if train_files_list else 'None'}")
    print(f"  Val   ({len(val_files_list)} files): {val_files_list if val_files_list else 'None'}")
    print(f"  Test  ({len(test_files_list)} files): {test_files_list if test_files_list else 'None'}")
    if current_file_idx < len(all_available_pkl_files):
        warnings.warn(f"{len(all_available_pkl_files) - current_file_idx} PKL files remain unallocated.")

    if train_files_list:
        print(f"\n--- Initializing/Processing Training Set (single topological sort per graph) ---")
        train_dataset = AIGProcessedAugmentedDataset(
            root=args.output_root, raw_dir=args.raw_dir, split='train',
            file_prefix=args.file_prefix, pkl_file_names_for_split=train_files_list,
            dataset_name=args.dataset_name
            # num_augmentations argument removed
        )
        del train_dataset;
        gc.collect()
    else:
        print("\nSkipping Training Set processing (no files allocated).")

    if val_files_list:
        print(f"\n--- Initializing/Processing Validation Set (single topological sort per graph) ---")
        val_dataset = AIGProcessedAugmentedDataset(
            root=args.output_root, raw_dir=args.raw_dir, split='val',
            file_prefix=args.file_prefix, pkl_file_names_for_split=val_files_list,
            dataset_name=args.dataset_name
        )
        del val_dataset;
        gc.collect()
    else:
        print("\nSkipping Validation Set processing (no files allocated).")

    if test_files_list:
        print(f"\n--- Initializing/Processing Test Set (single topological sort per graph) ---")
        test_dataset = AIGProcessedAugmentedDataset(
            root=args.output_root, raw_dir=args.raw_dir, split='test',
            file_prefix=args.file_prefix, pkl_file_names_for_split=test_files_list,
            dataset_name=args.dataset_name
        )
        del test_dataset;
        gc.collect()
    else:
        print("\nSkipping Test Set processing (no files allocated).")

    print("\nAll specified dataset processing finished (or skipped).")
    print(
        f"Processed data (if any) should be in subdirectories under: {osp.join(args.output_root, args.dataset_name, 'processed')}")
