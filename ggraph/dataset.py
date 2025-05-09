#!/usr/bin/env python3
import os
import torch
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.data.data import BaseData  # For type hinting
import os.path as osp
from typing import List, Tuple, Dict, Optional, Any
import torch_geometric.io.fs as pyg_fs  # Import PyG's filesystem utilities
import pickle
import warnings
import argparse
import gc
import shutil
import networkx as nx
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
from aig_config import (
        MAX_NODE_COUNT, NUM_NODE_FEATURES, NUM_ADJ_CHANNELS,
        NUM_EXPLICIT_EDGE_TYPES, NODE_TYPE_KEYS, EDGE_TYPE_KEYS,
        NODE_TYPE_ENCODING_NX, EDGE_TYPE_ENCODING_NX
    )


class AIGPreprocessedDatasetLoader(Dataset):
    def __init__(self, root: str,  split: str,
                 transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None,
                 pre_filter: Optional[callable] = None):

        self.split = split
        self.dataset_specific_root = root

        super().__init__(root=self.dataset_specific_root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        self._processed_file_path = osp.join(self.processed_dir, self.processed_file_names[0])

        osp_exists = osp.exists(self._processed_file_path)
        pygfs_exists = pyg_fs.exists(self._processed_file_path)

        if osp_exists and pygfs_exists:
            try:
                loaded_content = torch.load(self._processed_file_path, map_location='cpu', weights_only=False)

                if not isinstance(loaded_content, tuple) or len(loaded_content) != 2:
                    raise TypeError(
                        f"Loaded data is not a tuple of 2 elements (data, slices). Got {type(loaded_content)}")

                self._data_list = None
                self.data, self.slices = loaded_content

            except Exception as e:
                print(f"  ERROR during direct torch.load or data assignment: {e}")
                raise RuntimeError(f"Failed to load pre-processed data from {self._processed_file_path}: {e}")
        else:
            raise FileNotFoundError(
                f"Processed file not found at {self._processed_file_path}. "
                f"Ensure the file exists and paths are correct. "
                f"osp.exists: {osp_exists}, pyg_fs.exists: {pygfs_exists}"
            )

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return [f'{self.split}_processed_data.pt']

    def download(self):
        pass

    def process(self):
        print(f"AIGPreprocessedDatasetLoader.process() called for split '{self.split}'. "
              f"This loader expects data to be already processed by AIGProcessedDataset.")
        if not pyg_fs.exists(self._processed_file_path):
            raise FileNotFoundError(f"Processed file not found: {self._processed_file_path}. "
                                    f"Please run the AIG data generation and processing script first.")

    def len(self) -> int:
        if self.slices is None:
            return 0
        for _, value in self.slices.items():
            if isinstance(value, torch.Tensor):  # Check if it's a tensor
                if value.ndim > 0:  # Check if it's not an empty tensor
                    return value.size(0) - 1
                else:  # Handle scalar tensor case if necessary, though slices are usually 1D
                    return 0
        return 0

    def get(self, idx: int) -> BaseData:
        if not hasattr(self, '_data_list') or self._data_list is None:
            if self.data is None or self.slices is None:
                raise IndexError(f"Dataset not loaded properly (data or slices is None), cannot get item {idx}")
            if not isinstance(self.data, BaseData):
                raise TypeError(f"self.data is not a PyG BaseData object, cannot get item {idx}")

            data = self.data.__class__()
            if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None:
                # Ensure __num_nodes__ itself is not None and idx is within bounds
                if idx < len(self.data.__num_nodes__):
                    data.num_nodes = self.data.__num_nodes__[idx]
                else:
                    # This case should ideally not happen if len() is correct
                    warnings.warn(
                        f"Index {idx} out of bounds for self.data.__num_nodes__ (len {len(self.data.__num_nodes__)}).")
                    # Fallback or raise error, depending on desired strictness
                    # For now, let it proceed and potentially fail on slicing if idx is truly out of overall bounds

            # Check overall index validity against dataset length
            num_items = self.len()
            if idx < 0 or idx >= num_items:
                raise IndexError(f"Index {idx} out of bounds for dataset with length {num_items}")

            for key in self.data.keys:
                item, slices_for_key = self.data[key], self.slices[key]
                # Ensure slices_for_key is a tensor and idx+1 is within its bounds
                if not isinstance(slices_for_key, torch.Tensor) or idx + 1 >= slices_for_key.size(0):
                    warnings.warn(
                        f"Slices for key '{key}' are invalid or index {idx} is out of bounds for slices. Skipping key.")
                    data[key] = None  # Or handle appropriately
                    continue

                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices_for_key[idx], slices_for_key[idx + 1])
                data[key] = item[s]
            return data
        else:
            # This case would be for a list of Data objects (not typical for InMemoryDataset after load)
            if idx < 0 or idx >= len(self._data_list):
                raise IndexError(f"Index {idx} out of bounds for self._data_list with length {len(self._data_list)}")
            return self._data_list[idx]


def _convert_nx_to_pyg_data(graph: nx.DiGraph,
                            max_nodes_pad: int,
                            num_node_features: int,
                            num_adj_channels: int,
                            num_explicit_edge_types: int) -> Optional[Data]:
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
            warnings.warn(
                f"Node {old_node_id} has invalid 'type' attribute (length or existence). Expected {num_node_features}-len one-hot vector. Got: {node_type_vec_raw}. Skipping graph.")
            return None
        try:
            node_type_vec = np.asarray(node_type_vec_raw, dtype=np.float32)
            if not (np.isclose(np.sum(node_type_vec), 1.0) and np.all(
                    (np.isclose(node_type_vec, 0.0)) | (np.isclose(node_type_vec, 1.0)))):
                warnings.warn(
                    f"Node {old_node_id} 'type' vector is not one-hot. Sum: {np.sum(node_type_vec)}. Values: {node_type_vec}. Skipping graph.")
                return None
            node_features_list.append(torch.tensor(node_type_vec, dtype=torch.float))
        except Exception as e:
            warnings.warn(f"Node {old_node_id} 'type' conversion to tensor failed: {e}. Skipping graph.")
            return None

    if not node_features_list:
        return None

    x_stacked = torch.stack(node_features_list)
    num_padding_nodes = max_nodes_pad - num_nodes_in_graph
    x_padded = F.pad(x_stacked, (0, 0, 0, num_padding_nodes), "constant", 0.0)

    adj_matrix = torch.zeros((num_adj_channels, max_nodes_pad, max_nodes_pad), dtype=torch.float)

    # The "no edge" channel is assumed to be the last one.
    # Its index is num_explicit_edge_types.
    no_edge_channel_idx = num_explicit_edge_types
    if no_edge_channel_idx < num_adj_channels:
        adj_matrix[no_edge_channel_idx, :num_nodes_in_graph, :num_nodes_in_graph] = 1.0
    else:
        warnings.warn(
            f"no_edge_channel_idx ({no_edge_channel_idx}) is out of bounds for num_adj_channels ({num_adj_channels}). No-edge channel might not be set correctly.")

    for u_old, v_old, attrs in graph.edges(data=True):
        edge_type_vec_raw = attrs.get('type')
        if edge_type_vec_raw is None or not isinstance(edge_type_vec_raw, (list, np.ndarray)) or len(
                edge_type_vec_raw) != num_explicit_edge_types:
            warnings.warn(
                f"Edge ({u_old}-{v_old}) has invalid 'type' attribute. Expected {num_explicit_edge_types}-len one-hot vector. Got: {edge_type_vec_raw}. Skipping graph.")
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
                    f"Edge ({u_old}-{v_old}) 'type' vector is not one-hot. Sum: {np.sum(edge_type_vec)}. Values: {edge_type_vec}. Skipping graph.")
                return None

            edge_channel_index = np.argmax(edge_type_vec).item()
            if not (0 <= edge_channel_index < num_explicit_edge_types):
                warnings.warn(
                    f"Edge ({u_old}-{v_old}) 'type' vector resulted in invalid channel index: {edge_channel_index} (max is {num_explicit_edge_types - 1}). Skipping graph.")
                return None

            adj_matrix[edge_channel_index, v_new, u_new] = 1.0
            if no_edge_channel_idx < num_adj_channels:
                adj_matrix[no_edge_channel_idx, v_new, u_new] = 0.0

        except Exception as e:
            warnings.warn(f"Edge ({u_old}-{v_old}) 'type' conversion or assignment failed: {e}. Skipping graph.")
            return None

    for k_node_diag in range(max_nodes_pad):
        for ch in range(num_adj_channels):
            adj_matrix[ch, k_node_diag, k_node_diag] = 0.0

    data = Data(x=x_padded, adj=adj_matrix, num_nodes=torch.tensor(num_nodes_in_graph, dtype=torch.long))

    if 'inputs' in graph.graph: data.num_inputs = torch.tensor(graph.graph['inputs'], dtype=torch.long)
    if 'outputs' in graph.graph: data.num_outputs = torch.tensor(graph.graph['outputs'], dtype=torch.long)
    if 'gates' in graph.graph: data.num_gates = torch.tensor(graph.graph['gates'], dtype=torch.long)

    return data


def _convert_pyg_to_nx_data(pyg_data: Data,
                            num_node_features: int,
                            num_explicit_edge_types: int
                            ) -> Optional[nx.DiGraph]:
    if not isinstance(pyg_data, Data):
        warnings.warn("Input is not a PyG Data object. Cannot convert.")
        return None

    reconstructed_graph = nx.DiGraph()
    num_actual_nodes = pyg_data.num_nodes.item()

    node_features_actual = pyg_data.x[:num_actual_nodes].cpu().numpy()

    for i in range(num_actual_nodes):
        node_type_vector = node_features_actual[i]
        if len(node_type_vector) != num_node_features:
            warnings.warn(
                f"Node {i} feature vector length mismatch. Expected {num_node_features}, got {len(node_type_vector)}")
            reconstructed_graph.add_node(i)
            continue
        reconstructed_graph.add_node(i, type=list(node_type_vector))

    adj_matrix_actual = pyg_data.adj[:, :num_actual_nodes, :num_actual_nodes].cpu().numpy()

    for channel_idx in range(num_explicit_edge_types):
        # Adjacency matrix stores adj[channel, target, source]
        # np.where on .T gives (source, target)
        source_indices, target_indices = np.where(adj_matrix_actual[channel_idx].T == 1.0)

        for src_node, tgt_node in zip(source_indices, target_indices):
            if reconstructed_graph.has_node(src_node) and reconstructed_graph.has_node(tgt_node):
                edge_type_vector = np.zeros(num_explicit_edge_types, dtype=np.float32)
                edge_type_vector[channel_idx] = 1.0
                reconstructed_graph.add_edge(src_node, tgt_node, type=list(edge_type_vector))
            else:
                warnings.warn(
                    f"Edge ({src_node} -> {tgt_node}) from channel {channel_idx} refers to non-existent nodes. Skipping.")

    if hasattr(pyg_data, 'num_inputs') and pyg_data.num_inputs is not None:
        reconstructed_graph.graph['inputs'] = pyg_data.num_inputs.item()
    if hasattr(pyg_data, 'num_outputs') and pyg_data.num_outputs is not None:
        reconstructed_graph.graph['outputs'] = pyg_data.num_outputs.item()
    if hasattr(pyg_data, 'num_gates') and pyg_data.num_gates is not None:
        reconstructed_graph.graph['gates'] = pyg_data.num_gates.item()

    return reconstructed_graph


def test_conversion_fidelity(original_nx_graph: nx.DiGraph,
                             max_nodes_pad: int,
                             num_node_feat: int,
                             num_adj_ch: int,
                             num_expl_edge_types: int
                             ) -> bool:
    print(f"\n--- Testing Conversion Fidelity for Graph (Nodes, Edges, and Types Only) ---")
    if not isinstance(original_nx_graph, nx.DiGraph):
        print("ERROR: Original graph is not a NetworkX DiGraph.")
        return False

    print("Step 1: Converting Original NX Graph to PyG Data...")
    pyg_data = _convert_nx_to_pyg_data(original_nx_graph, max_nodes_pad, num_node_feat, num_adj_ch, num_expl_edge_types)

    if pyg_data is None:
        if original_nx_graph.number_of_nodes() > max_nodes_pad:
            print(
                f"Original graph has {original_nx_graph.number_of_nodes()} nodes, which exceeds max_nodes_pad {max_nodes_pad}. "
                "Conversion to PyG was skipped as expected. Test PASSES for this specific case.")
            return True
        print("ERROR: NX to PyG conversion failed unexpectedly (returned None).")
        return False
    print("NX to PyG conversion successful.")

    print("\nStep 2: Converting PyG Data back to Reconstructed NX Graph...")
    reconstructed_nx_graph = _convert_pyg_to_nx_data(pyg_data, num_node_feat, num_expl_edge_types)

    if reconstructed_nx_graph is None:
        print("ERROR: PyG to NX conversion failed (returned None).")
        return False
    print("PyG to NX conversion successful.")

    print("\nStep 3: Comparing Original and Reconstructed NX Graphs...")
    match = True
    error_messages = []

    original_node_list_sorted = sorted(list(original_nx_graph.nodes()))

    num_nodes_orig = original_nx_graph.number_of_nodes()
    num_nodes_reco = reconstructed_nx_graph.number_of_nodes()
    if num_nodes_orig != num_nodes_reco:
        error_messages.append(f"Node count mismatch: Original={num_nodes_orig}, Reconstructed={num_nodes_reco}")
        match = False
    else:
        print(f"Node count matches: {num_nodes_orig}")
        for i in range(num_nodes_reco):
            original_node_id = original_node_list_sorted[i]
            orig_attrs = original_nx_graph.nodes[original_node_id]
            reco_attrs = reconstructed_nx_graph.nodes[i]

            orig_type = orig_attrs.get('type')
            reco_type = reco_attrs.get('type')

            if orig_type is None and reco_type is None:
                continue
            if orig_type is None or reco_type is None:
                error_messages.append(
                    f"Node 'type' attribute missing for one graph: Original node {original_node_id} (reco index {i}) -> Orig_has_type={orig_type is not None}, Reco_has_type={reco_type is not None}")
                match = False
                continue
            orig_type_np = np.asarray(orig_type, dtype=np.float32)
            reco_type_np = np.asarray(reco_type, dtype=np.float32)
            if not np.allclose(orig_type_np, reco_type_np):
                error_messages.append(f"Node 'type' mismatch for original node {original_node_id} (reco index {i}): "
                                      f"Original={orig_type}, Reconstructed={reco_type}")
                match = False
    if not error_messages or all(
            "Node count matches" in msg for msg in error_messages):  # Check if node count matched before saying this
        print("Node 'type' attributes comparison completed.")

    num_edges_orig = original_nx_graph.number_of_edges()
    num_edges_reco = reconstructed_nx_graph.number_of_edges()
    if num_edges_orig != num_edges_reco:
        error_messages.append(f"Edge count mismatch: Original={num_edges_orig}, Reconstructed={num_edges_reco}")
        match = False
    else:
        print(f"Edge count matches: {num_edges_orig}")
        original_edges_dict = {}
        for u_orig, v_orig, data_orig in original_nx_graph.edges(data=True):
            original_edges_dict[(u_orig, v_orig)] = data_orig.get('type')

        reconstructed_edges_found_in_original_count = 0
        edges_in_reco_not_in_orig_mapped = []

        for u_reco, v_reco, data_reco in reconstructed_nx_graph.edges(data=True):
            if u_reco >= len(original_node_list_sorted) or v_reco >= len(original_node_list_sorted):
                error_messages.append(
                    f"Edge ({u_reco}->{v_reco}) in reconstructed graph has out-of-bounds node index for mapping.")
                match = False
                continue

            original_src_node_id = original_node_list_sorted[u_reco]
            original_tgt_node_id = original_node_list_sorted[v_reco]

            reco_edge_type = data_reco.get('type')

            if (original_src_node_id, original_tgt_node_id) in original_edges_dict:
                original_edge_type = original_edges_dict[(original_src_node_id, original_tgt_node_id)]
                reconstructed_edges_found_in_original_count += 1

                if original_edge_type is None and reco_edge_type is None:
                    continue
                if original_edge_type is None or reco_edge_type is None:
                    error_messages.append(
                        f"Edge 'type' attribute missing for one graph for edge ({original_src_node_id}->{original_tgt_node_id})")
                    match = False
                    continue

                orig_edge_type_np = np.asarray(original_edge_type, dtype=np.float32)
                reco_edge_type_np = np.asarray(reco_edge_type, dtype=np.float32)
                if not np.allclose(orig_edge_type_np, reco_edge_type_np):
                    error_messages.append(f"Edge 'type' mismatch for ({original_src_node_id}->{original_tgt_node_id}): "
                                          f"Original={original_edge_type}, Reconstructed={reco_edge_type}")
                    match = False
            else:
                edges_in_reco_not_in_orig_mapped.append((original_src_node_id, original_tgt_node_id))

        if edges_in_reco_not_in_orig_mapped:
            error_messages.append(
                f"{len(edges_in_reco_not_in_orig_mapped)} Edges found in reconstructed graph but not in original (after mapping IDs): {edges_in_reco_not_in_orig_mapped[:5]}...")
            match = False

        if reconstructed_edges_found_in_original_count != num_edges_orig:
            missing_orig_edges = []
            for (u_orig, v_orig), type_orig in original_edges_dict.items():
                try:
                    # Map original node IDs to their new 0-indexed IDs used in reconstructed_nx_graph
                    u_reco_mapped_idx = original_node_list_sorted.index(u_orig)
                    v_reco_mapped_idx = original_node_list_sorted.index(v_orig)
                    if not reconstructed_nx_graph.has_edge(u_reco_mapped_idx, v_reco_mapped_idx):
                        missing_orig_edges.append((u_orig, v_orig))
                except ValueError:
                    missing_orig_edges.append((u_orig, v_orig))
            if missing_orig_edges:
                error_messages.append(
                    f"{len(missing_orig_edges)} Original edges not found in reconstructed graph (after mapping IDs): {missing_orig_edges[:5]}...")
                match = False

    if not error_messages or all("Edge count matches" in msg for msg in error_messages):  # Check if edge count matched
        print("Edge 'type' attributes comparison completed.")

    if match:
        print("SUCCESS: Reconstructed graph matches original graph for nodes, edges, and their 'type' attributes.")
    else:
        print("FAILURE: Graph mismatch detected for nodes/edges/types. Errors:")
        for err in error_messages:
            print(f"  - {err}")
    return match


class AIGProcessedDataset(InMemoryDataset):
    def __init__(self, root: str, split: str,
                 raw_dir: Optional[str] = None,
                 file_prefix: Optional[str] = None,
                 pkl_file_names_for_split: Optional[List[str]] = None,
                 transform=None, pre_transform=None, pre_filter=None):

        self._temp_raw_dir = raw_dir
        self._temp_file_prefix = file_prefix
        self._temp_pkl_files = pkl_file_names_for_split if pkl_file_names_for_split is not None else []
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)

        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
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
        if self._temp_raw_dir is None or not self._temp_pkl_files:
            if not self.raw_paths:
                print(f"No raw files specified for split '{self.split}'. Skipping processing.")
                empty_data_list = []
                if self.pre_filter is not None: empty_data_list = [d for d in empty_data_list if self.pre_filter(d)]
                if self.pre_transform is not None: empty_data_list = [self.pre_transform(d) for d in empty_data_list]
                data, slices = self.collate(empty_data_list)
                torch.save((data, slices), self.processed_paths[0])
                print(f"Saved empty dataset for split '{self.split}' to: {self.processed_paths[0]}")
                return
            raise ValueError("Raw directory and file list are required for processing if raw_paths is not empty.")

        print(f"Processing raw PKL files for split: {self.split}...")

        temp_individual_graph_pts_dir = osp.join(self.processed_dir, f"{self.split}_temp_individual_pts")
        os.makedirs(temp_individual_graph_pts_dir, exist_ok=True)
        paths_to_temp_pt_files = []

        total_original_graphs_considered = 0
        total_successful_conversions = 0
        temp_file_counter = 0

        current_max_nodes_pad = MAX_NODE_COUNT
        current_num_node_features = NUM_NODE_FEATURES
        current_num_adj_channels = NUM_ADJ_CHANNELS
        current_num_explicit_edge_types = NUM_EXPLICIT_EDGE_TYPES

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

            for nx_graph_idx, nx_graph in enumerate(nx_graphs_in_chunk):
                if not isinstance(nx_graph, nx.DiGraph):
                    warnings.warn(
                        f"Item at index {nx_graph_idx} in {raw_pkl_path} is not a NetworkX DiGraph. Skipping.")
                    continue

                total_original_graphs_considered += 1
                pyg_data_objects_for_current_nx_graph = []

                base_pyg_data = _convert_nx_to_pyg_data(
                    nx_graph,
                    current_max_nodes_pad,
                    current_num_node_features,
                    current_num_adj_channels,
                    current_num_explicit_edge_types
                )

                if base_pyg_data is not None:
                    total_successful_conversions += 1
                    pyg_data_objects_for_current_nx_graph.append(base_pyg_data)

                if pyg_data_objects_for_current_nx_graph:
                    temp_pt_file_path = osp.join(temp_individual_graph_pts_dir, f"graph_batch_{temp_file_counter}.pt")
                    try:
                        torch.save(pyg_data_objects_for_current_nx_graph, temp_pt_file_path)
                        paths_to_temp_pt_files.append(temp_pt_file_path)
                        temp_file_counter += 1
                    except Exception as e_save:
                        warnings.warn(f"Could not save temporary batch file {temp_pt_file_path}: {e_save}")

                del pyg_data_objects_for_current_nx_graph, base_pyg_data
                if (nx_graph_idx + 1) % 200 == 0:
                    gc.collect()

            del nx_graphs_in_chunk
            gc.collect()

        print(f"\nFinished converting graphs. Processed {total_original_graphs_considered} original graphs.")
        print(f"Successfully converted to PyG: {total_successful_conversions} graphs.")
        print(f"Total temporary .pt files created: {len(paths_to_temp_pt_files)}")

        all_pyg_data_objects_list = []
        if paths_to_temp_pt_files:
            print(f"\nLoading data from {len(paths_to_temp_pt_files)} temporary .pt files...")
            for temp_pt_file_path in tqdm(paths_to_temp_pt_files, desc="Loading temp .pt files"):
                try:
                    list_of_data_objects = torch.load(temp_pt_file_path, weights_only=False)
                    all_pyg_data_objects_list.extend(list_of_data_objects)
                    os.remove(temp_pt_file_path)
                except Exception as e_load:
                    warnings.warn(f"Could not load or delete temporary file {temp_pt_file_path}: {e_load}")
            try:
                shutil.rmtree(temp_individual_graph_pts_dir)
            except OSError as e_rmdir:
                warnings.warn(
                    f"Could not remove temporary directory {temp_individual_graph_pts_dir}: {e_rmdir}.")
        else:
            print(f"No temporary .pt files were created for split '{self.split}'. The resulting dataset will be empty.")

        del paths_to_temp_pt_files
        gc.collect()

        print(f"\nTotal PyG Data objects to collate: {len(all_pyg_data_objects_list)}")

        if self.pre_filter is not None:
            all_pyg_data_objects_list = [data for data in all_pyg_data_objects_list if self.pre_filter(data)]
            print(f"Data objects after pre_filter: {len(all_pyg_data_objects_list)}")

        if self.pre_transform is not None:
            processed_list_for_transform = []
            for data_idx, data_obj in enumerate(tqdm(all_pyg_data_objects_list, desc="Applying pre_transform")):
                processed_list_for_transform.append(self.pre_transform(data_obj))
                if (data_idx + 1) % 500 == 0:
                    gc.collect()
            all_pyg_data_objects_list = processed_list_for_transform
            del processed_list_for_transform
            gc.collect()
            print(f"Data objects after pre_transform: {len(all_pyg_data_objects_list)}")

        print(f"Collating {len(all_pyg_data_objects_list)} PyG Data objects...")
        if not all_pyg_data_objects_list:
            warnings.warn(f"No data to collate for split '{self.split}'. Saving an empty dataset structure.")

        data, slices = self.collate(all_pyg_data_objects_list)
        del all_pyg_data_objects_list
        gc.collect()

        final_save_path = self.processed_paths[0]
        print(f"Saving final processed data for split '{self.split}' to: {final_save_path}")
        torch.save((data, slices), final_save_path)
        print(f"Successfully saved final data for split '{self.split}'.")


from itertools import repeat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process AIG PKL files into PyG InMemoryDataset format and optionally run tests.")
    parser.add_argument('--raw_dir', type=str, required=True,
                        help="Directory containing raw PKL files (output of AIG generator).")
    parser.add_argument('--output_root', type=str, required=True,
                        help="Root directory for dataset output (e.g., ./datasets_output).")
    parser.add_argument('--file_prefix', type=str, default="real_aigs_part_",
                        help="Prefix of raw PKL files to look for.")
    parser.add_argument('--num_train_files', type=int, default=4,
                        help="Number of PKL files to allocate for the training set. More files might be needed if testing many graphs.")
    parser.add_argument('--num_val_files', type=int, default=0,
                        help="Number of PKL files to allocate for the validation set.")
    parser.add_argument('--num_test_files', type=int, default=0,
                        help="Number of PKL files to allocate for the test set.")
    parser.add_argument('--run_conversion_test', action='store_true',
                        help="If set, run the NX <-> PyG conversion fidelity test on sample graphs.")
    parser.add_argument('--num_graphs_to_test', type=int, default=10,
                        help="Number of graphs to sample and test for conversion fidelity (if --run_conversion_test is set).")
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
        if args.run_conversion_test:
            print("Cannot run conversion test as no PKL files are available.")
        exit(1)

    print(f"Found {len(all_available_pkl_files)} PKL files matching prefix '{args.file_prefix}' in {args.raw_dir}:")

    current_file_idx = 0
    train_files_list, val_files_list, test_files_list = [], [], []  # Keep val/test lists for completeness

    # Allocate files for training (these will be used to source test graphs if needed)
    if args.num_train_files > 0:
        take_n = min(args.num_train_files, len(all_available_pkl_files) - current_file_idx)
        if take_n > 0:
            train_files_list = all_available_pkl_files[current_file_idx: current_file_idx + take_n]
            current_file_idx += take_n
    # (Add similar logic for val_files_list and test_files_list if they are to be processed later)

    print(f"\nFile Allocation for Processing/Testing:")
    print(
        f"  Train files considered for tests ({len(train_files_list)} files): {train_files_list if train_files_list else 'None'}")

    if args.run_conversion_test:
        if not train_files_list:
            print("Conversion test requested, but no training files were allocated or found. Skipping test.")
        else:
            print(f"\n--- Running Conversion Fidelity Test for up to {args.num_graphs_to_test} graphs ---")
            graphs_to_test_conversion = []
            for pkl_file_name in train_files_list:
                if len(graphs_to_test_conversion) >= args.num_graphs_to_test:
                    break
                test_pkl_file_path = osp.join(args.raw_dir, pkl_file_name)
                print(f"Loading sample graphs from: {test_pkl_file_path}")
                try:
                    with open(test_pkl_file_path, 'rb') as f:
                        sample_nx_graphs_from_file = pickle.load(f)
                    if sample_nx_graphs_from_file and isinstance(sample_nx_graphs_from_file, list):
                        # Add graphs from this file until we reach num_graphs_to_test or run out
                        needed = args.num_graphs_to_test - len(graphs_to_test_conversion)
                        graphs_to_test_conversion.extend(sample_nx_graphs_from_file[:needed])
                except Exception as e:
                    print(f"Warning: Could not load or process PKL file {test_pkl_file_path} for testing: {e}")

            if not graphs_to_test_conversion:
                print("ERROR: Could not load any valid NetworkX graphs from the specified PKL files for testing.")
            else:
                num_actually_testing = min(args.num_graphs_to_test, len(graphs_to_test_conversion))
                print(f"Proceeding to test {num_actually_testing} loaded graphs.")

                passed_tests = 0
                failed_tests = 0
                for i, sample_graph in enumerate(graphs_to_test_conversion[:num_actually_testing]):
                    if not isinstance(sample_graph, nx.DiGraph):
                        print(
                            f"Warning: Item at index {i} from loaded graphs is not a NetworkX DiGraph. Skipping this item.")
                        failed_tests += 1  # Count as a failure to load a valid graph for test
                        continue

                    print(
                        f"\nTesting graph {i + 1}/{num_actually_testing} (Nodes: {sample_graph.number_of_nodes()}, Edges: {sample_graph.number_of_edges()})...")
                    test_result = test_conversion_fidelity(
                        sample_graph,
                        MAX_NODE_COUNT,
                        NUM_NODE_FEATURES,
                        NUM_ADJ_CHANNELS,
                        NUM_EXPLICIT_EDGE_TYPES
                    )
                    if test_result:
                        passed_tests += 1
                    else:
                        failed_tests += 1

                print(f"\nConversion Fidelity Test Summary:")
                print(f"  Graphs Tested: {num_actually_testing}")
                print(f"  Tests Passed : {passed_tests}")
                print(f"  Tests Failed : {failed_tests}")

        print("--- End of Conversion Fidelity Test ---\n")

    # --- Dataset Processing (AIGProcessedDataset instantiation) ---
    # This part will create/load the .pt files for train, val, test splits as configured
    if train_files_list:  # Check if there are files allocated for actual training set processing
        print(f"\n--- Initializing/Processing Training Set (from {len(train_files_list)} PKL files) ---")
        train_dataset = AIGProcessedDataset(
            root=args.output_root,
            raw_dir=args.raw_dir,
            split='train',
            file_prefix=args.file_prefix,
            pkl_file_names_for_split=train_files_list,  # Use the allocated list
        )
        del train_dataset;
        gc.collect()
    else:
        print("\nSkipping Training Set processing as no files were allocated for it.")

    # Add similar blocks for val_files_list and test_files_list if you intend to process them
    # For example:
    # if val_files_list:
    #     print(f"\n--- Initializing/Processing Validation Set ---")
    #     val_dataset = AIGProcessedDataset(...)
    #     del val_dataset; gc.collect()
    # else:
    #     print("\nSkipping Validation Set processing...")

    print("\nAll specified dataset processing finished (or skipped if no files).")
