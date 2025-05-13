#!/usr/bin/env python3
import os
import os.path as osp
import pathlib
import math  # For precise splitting
import warnings  # For explicit warnings
import shutil  # For copying files in download()

import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader  # Used by AbstractDataModule
import pytorch_lightning as pl  # Used by AbstractDataModule
# Ensure torch_geometric.utils is available for dense_to_sparse if needed by AbstractDataModule's utils
import torch_geometric.utils

from tqdm import tqdm
import numpy as np

# DiGress imports (ensure these paths are correct for your project structure)
# Assuming AbstractDataModule and AbstractDatasetInfos are correctly defined
# For this example, I'll include the AbstractDataModule and AbstractDatasetInfos
# definitions directly if they are not complex, or assume they are importable.
# If src.datasets.abstract_dataset is not found, these placeholders will be used.
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.diffusion.distributions import DistributionNodes
import src.utils as utils  # For to_dense

# Your AIG specific imports
from src.aig_config import (
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES,  # NUM_EDGE_FEATURES is num *actual* AIG types
    NODE_TYPE_KEYS, EDGE_TYPE_KEYS,  # EDGE_TYPE_KEYS length is NUM_EDGE_FEATURES
    check_validity
)
from typing import Union, List


# --- Helper Function: PyG Data to NetworkX DiGraph (MODIFIED) ---
def convert_pyg_to_nx_for_aig_validation(pyg_data: Data) -> Union[nx.DiGraph, None]:
    """
    Converts a PyTorch Geometric Data object back to a NetworkX DiGraph
    for AIG-specific validation.
    Assumes pyg_data.edge_attr has NUM_EDGE_FEATURES + 1 columns,
    where index 0 is for "no edge" (and should not be set for actual edges)
    and indices 1 to NUM_EDGE_FEATURES+1 map to actual AIG edge types.
    """
    if not all(hasattr(pyg_data, attr) for attr in ['x', 'edge_index', 'edge_attr', 'num_nodes']):
        warnings.warn(
            "PyG to NX: pyg_data object is missing required attributes (x, edge_index, edge_attr, num_nodes).")
        return None

    nx_graph = nx.DiGraph()

    if pyg_data.num_nodes is None:
        warnings.warn("PyG to NX: pyg_data.num_nodes is None.")
        return None
    num_nodes = pyg_data.num_nodes.item()

    if pyg_data.x is None:
        warnings.warn("PyG to NX: pyg_data.x is None.")
        return None

    # Validate NODE_TYPE_KEYS and NUM_NODE_FEATURES consistency
    if len(NODE_TYPE_KEYS) != NUM_NODE_FEATURES:
        warnings.warn(f"PyG to NX: NODE_TYPE_KEYS length ({len(NODE_TYPE_KEYS)}) "
                      f"mismatches NUM_NODE_FEATURES ({NUM_NODE_FEATURES}).")
        return None
    # Validate EDGE_TYPE_KEYS and NUM_EDGE_FEATURES consistency
    if len(EDGE_TYPE_KEYS) != NUM_EDGE_FEATURES:  # NUM_EDGE_FEATURES is num *actual* types
        warnings.warn(f"PyG to NX: EDGE_TYPE_KEYS length ({len(EDGE_TYPE_KEYS)}) "
                      f"mismatches NUM_EDGE_FEATURES ({NUM_EDGE_FEATURES}).")
        return None

    # Validate node features shape
    if pyg_data.x.shape[0] != num_nodes or pyg_data.x.shape[1] != NUM_NODE_FEATURES:
        warnings.warn(
            f"PyG to NX: Node feature tensor x has incorrect shape (got {pyg_data.x.shape}, expected ({num_nodes}, {NUM_NODE_FEATURES})).")
        return None

    # Process nodes
    for i in range(num_nodes):
        node_feature_vector = pyg_data.x[i].cpu().numpy()
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_ATTRIBUTE"  # As per aig_config.check_validity
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)):  # type_index is for actual node types
                node_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    # Process edges
    if pyg_data.edge_index is not None and pyg_data.edge_index.shape[1] > 0:
        if pyg_data.edge_attr is None:
            warnings.warn("PyG to NX: pyg_data.edge_attr is None but there are edges.")
            return None
        # MODIFIED: Validate edge_attr shape (now NUM_EDGE_FEATURES + 1 columns)
        expected_edge_attr_cols = NUM_EDGE_FEATURES + 1
        if pyg_data.edge_attr.shape[0] != pyg_data.edge_index.shape[1] or \
                pyg_data.edge_attr.shape[1] != expected_edge_attr_cols:
            warnings.warn(f"PyG to NX: Edge attribute tensor edge_attr has incorrect shape "
                          f"(got {pyg_data.edge_attr.shape}, expected ({pyg_data.edge_index.shape[1]}, {expected_edge_attr_cols})).")
            return None

        for i in range(pyg_data.edge_index.shape[1]):
            src_node = pyg_data.edge_index[0, i].item()
            tgt_node = pyg_data.edge_index[1, i].item()
            edge_feature_vector = pyg_data.edge_attr[i].cpu().numpy()  # Length NUM_EDGE_FEATURES + 1

            # Check if one-hot
            if not (np.isclose(np.sum(edge_feature_vector), 1.0) and
                    np.all((np.isclose(edge_feature_vector, 0.0)) | (np.isclose(edge_feature_vector, 1.0)))):
                edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
            else:
                shifted_type_index = np.argmax(edge_feature_vector)
                if shifted_type_index == 0:  # Index 0 is for "no edge", should not be set for an existing edge's attr
                    warnings.warn(
                        f"PyG to NX: Edge ({src_node}-{tgt_node}) attribute implies 'no edge' (index 0 is 1). Treating as UNKNOWN.")
                    edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
                else:
                    actual_aig_type_index = shifted_type_index - 1  # Convert back to 0-indexed actual AIG type
                    if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):  # Check against actual AIG types
                        edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
                    else:
                        edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]
            nx_graph.add_edge(src_node, tgt_node, type=edge_type_str)

    # Add graph-level attributes if they exist
    for attr_name in ['inputs', 'outputs', 'gates']:
        if hasattr(pyg_data, f'num_{attr_name}') and getattr(pyg_data, f'num_{attr_name}') is not None:
            nx_graph.graph[attr_name] = getattr(pyg_data, f'num_{attr_name}').item()
    return nx_graph


# --- End Helper Function ---


class AIGCustomDataset(InMemoryDataset):
    def __init__(self, root: str, split: str, cfg, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.cfg = cfg
        if self.cfg is None:
            raise ValueError("cfg must be provided to AIGCustomDataset.")

        base_path = pathlib.Path(self.cfg.general.abs_path_to_project_root)
        self.path_to_source_combined_raw_file = base_path / self.cfg.dataset.datadir_for_all_raw / self.cfg.dataset.all_raw_graphs_filename

        super().__init__(root, transform, pre_transform, pre_filter)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Loaded AIGCustomDataset '{split}' with {len(self)} graphs from {self.processed_paths[0]}")
        except FileNotFoundError:
            warnings.warn(
                f"Processed file not found for {split} at {self.processed_paths[0]}. Will be created by process().")
        except Exception as e:
            warnings.warn(f"Error loading processed file for {split} at {self.processed_paths[0]}: {e}")

    @property
    def raw_file_names(self) -> List[str]:
        return [osp.basename(self.path_to_source_combined_raw_file)]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.split}.pt']

    def download(self):
        dest_raw_file_path = osp.join(self.raw_dir, self.raw_file_names[0])
        if osp.exists(dest_raw_file_path): return

        if not osp.exists(self.path_to_source_combined_raw_file):
            raise FileNotFoundError(f"Source raw AIG data not found: {self.path_to_source_combined_raw_file}")

        os.makedirs(self.raw_dir, exist_ok=True)
        shutil.copy(self.path_to_source_combined_raw_file, dest_raw_file_path)
        print(f"Copied source AIG raw data to {dest_raw_file_path}")

    def process(self):
        path_to_combined_raw_in_raw_dir = osp.join(self.raw_dir, self.raw_file_names[0])
        print(f"AIGCustomDataset ({self.split}): Processing raw data from: {path_to_combined_raw_in_raw_dir}")

        all_graphs_list_from_raw_file = torch.load(path_to_combined_raw_in_raw_dir)
        num_total_graphs = len(all_graphs_list_from_raw_file)

        if num_total_graphs == 0:
            warnings.warn(f"Raw AIG data file {path_to_combined_raw_in_raw_dir} is empty.")
            torch.save(self.collate([]), self.processed_paths[0])
            return

        # Splitting logic
        idx_train_end = math.floor(num_total_graphs * 4 / 6)
        idx_val_end = math.floor(num_total_graphs * 5 / 6)
        if num_total_graphs >= 3:
            if idx_train_end == num_total_graphs:
                idx_train_end = max(0, num_total_graphs - 2)
                idx_val_end = max(idx_train_end, num_total_graphs - 1)
            elif idx_val_end == num_total_graphs:
                if idx_train_end < num_total_graphs - 1:
                    idx_val_end = max(idx_train_end, num_total_graphs - 1)
        elif num_total_graphs == 2:
            idx_train_end, idx_val_end = 1, 1
        elif num_total_graphs == 1:
            idx_train_end, idx_val_end = 1, 1

        current_split_graphs_raw = []
        if self.split == 'train':
            current_split_graphs_raw = all_graphs_list_from_raw_file[:idx_train_end]
        elif self.split == 'val':
            current_split_graphs_raw = all_graphs_list_from_raw_file[idx_train_end:idx_val_end]
        elif self.split == 'test':
            current_split_graphs_raw = all_graphs_list_from_raw_file[idx_val_end:]

        print(
            f"AIGDataset ({self.split}): Num graphs for this split (before processing): {len(current_split_graphs_raw)}")

        processed_pyg_data_list = []
        valid_count, invalid_count = 0, 0

        # NUM_EDGE_FEATURES from aig_config is the number of *actual* AIG edge types
        # The new edge_attr dimension will be NUM_EDGE_FEATURES + 1
        new_edge_attr_dim = NUM_EDGE_FEATURES + 1

        for i, data_item_raw in enumerate(
                tqdm(current_split_graphs_raw, desc=f"Processing/Validating for {self.split}", unit="graph")):
            # Create a new Data object to avoid modifying the one in all_graphs_list_from_raw_file if it's referenced elsewhere
            data_item = data_item_raw.clone()  # Ensure we work on a copy

            # --- Standardize and Validate Basic Attributes ---
            if not all(hasattr(data_item, attr) for attr in ['x', 'edge_index', 'num_nodes']):
                warnings.warn(
                    f"Graph {i} in {self.split} missing basic attributes (x, edge_index, num_nodes). Skipping.")
                invalid_count += 1
                continue

            if not hasattr(data_item, 'adj'):  # adj was part of the raw data
                warnings.warn(
                    f"Graph {i} in {self.split} missing 'adj' attribute. Skipping if critical, or proceeding without.")
                # Depending on whether 'adj' is strictly needed later, you might skip or create a dummy one.
                # For now, we assume 'adj' from raw data is carried over as is.

            if not hasattr(data_item, 'y'):
                data_item.y = torch.zeros([1, 0], dtype=data_item.x.dtype if data_item.x is not None else torch.float)

            if not torch.is_tensor(data_item.num_nodes) or data_item.num_nodes.numel() != 1:
                try:
                    data_item.num_nodes = torch.tensor(int(data_item.num_nodes), dtype=torch.long)
                except:
                    warnings.warn(f"Graph {i} num_nodes invalid. Skipping.")
                    invalid_count += 1;
                    continue

            if data_item.x is None or data_item.x.shape[0] != data_item.num_nodes.item():
                warnings.warn(f"Graph {i} num_nodes vs x.shape[0] mismatch. Skipping.")
                invalid_count += 1;
                continue

            # --- Transform edge_attr ---
            # The raw data_item.edge_attr is (NumEdges, NUM_EDGE_FEATURES)
            # We need to convert it to (NumEdges, NUM_EDGE_FEATURES + 1)
            if data_item.edge_index.shape[1] > 0:  # If there are edges
                if not hasattr(data_item, 'edge_attr') or data_item.edge_attr is None:
                    warnings.warn(f"Graph {i} has edges but no edge_attr. Skipping.")
                    invalid_count += 1;
                    continue

                original_edge_attr = data_item.edge_attr
                if original_edge_attr.shape[1] != NUM_EDGE_FEATURES:
                    warnings.warn(
                        f"Graph {i} original edge_attr dim is {original_edge_attr.shape[1]}, expected {NUM_EDGE_FEATURES}. Skipping.")
                    invalid_count += 1;
                    continue

                num_edges = original_edge_attr.shape[0]
                new_ea = torch.zeros((num_edges, new_edge_attr_dim), dtype=original_edge_attr.dtype,
                                     device=original_edge_attr.device)

                # Find indices of actual types (0 to NUM_EDGE_FEATURES-1)
                actual_type_indices = torch.argmax(original_edge_attr, dim=1)
                # Shift these indices by +1 for the new_ea
                shifted_indices = (actual_type_indices + 1).unsqueeze(1)
                new_ea.scatter_(1, shifted_indices, 1.0)
                data_item.edge_attr = new_ea
            else:  # No edges
                data_item.edge_attr = torch.empty((0, new_edge_attr_dim),
                                                  dtype=data_item.x.dtype if data_item.x is not None else torch.float,
                                                  device=data_item.x.device if data_item.x is not None else 'cpu')

            # --- AIG-specific validation using the (now modified) data_item ---
            nx_graph_for_validation = convert_pyg_to_nx_for_aig_validation(data_item)
            if nx_graph_for_validation is None:
                # convert_pyg_to_nx_for_aig_validation already prints warnings
                invalid_count += 1
                continue

            if check_validity(nx_graph_for_validation):
                processed_pyg_data_list.append(data_item)  # Add the modified data_item
                valid_count += 1
            else:
                # Optionally log which graph failed or specific reasons if check_validity provides them
                # warnings.warn(f"Graph {i} in {self.split} failed AIG check_validity. Skipping.")
                invalid_count += 1

        print(f"AIGDataset ({self.split}): Validation done. Valid: {valid_count}, Invalid/Skipped: {invalid_count}")

        if self.pre_filter is not None:
            processed_pyg_data_list = [d for d in processed_pyg_data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            processed_pyg_data_list = [self.pre_transform(d) for d in processed_pyg_data_list]

        if not processed_pyg_data_list:
            warnings.warn(f"AIGDataset ({self.split}): No data to save after processing and validation.")

        print(f"AIGDataset ({self.split}): Saving {len(processed_pyg_data_list)} graphs to {self.processed_paths[0]}")
        torch.save(self.collate(processed_pyg_data_list), self.processed_paths[0])


class AIGCustomDataModule(AbstractDataModule):
    def __init__(self, cfg):
        base_path = pathlib.Path(cfg.general.abs_path_to_project_root)
        dataset_instance_root = base_path / cfg.dataset.datadir / cfg.dataset.name

        train_dataset = AIGCustomDataset(root=str(dataset_instance_root), split='train', cfg=cfg)
        val_dataset = AIGCustomDataset(root=str(dataset_instance_root), split='val', cfg=cfg)
        test_dataset = AIGCustomDataset(root=str(dataset_instance_root), split='test', cfg=cfg)

        super().__init__(cfg, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
        print(f"AIGCustomDataModule initialized for '{cfg.dataset.name}'. Root: {dataset_instance_root}")
        print(f"  Train graphs: {len(self.train_dataset)}")
        print(f"  Val graphs  : {len(self.val_dataset)}")
        print(f"  Test graphs : {len(self.test_dataset)}")


class AIGCustomDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: AIGCustomDataModule, dataset_config):
        super().__init__(datamodule, dataset_config)

        self.n_nodes_dist_stats = self.datamodule.node_counts()
        self.node_type_dist_stats = self.datamodule.node_types()
        # This will now return a distribution of length NUM_EDGE_FEATURES + 1
        self.edge_type_dist_stats = self.datamodule.edge_counts()

        self.complete_infos(n_nodes=self.n_nodes_dist_stats, node_types=self.node_type_dist_stats)

        # Assuming extra_features_fn and domain_features_fn are None or defined in cfg
        extra_features_fn = getattr(datamodule.cfg.dataset, "extra_features_method", None)
        domain_features_fn = getattr(datamodule.cfg.dataset, "domain_features_method", None)

        self.compute_input_output_dims(datamodule, extra_features_fn, domain_features_fn)

        print(f"AIGCustomDatasetInfos for '{self.name}':")
        print(f"  Max nodes: {self.max_n_nodes}")
        print(f"  Node classes (num_node_features): {self.num_classes}")
        print(
            f"  Node type dist len: {self.node_type_dist_stats.size(0) if self.node_type_dist_stats is not None else 'N/A'}")
        # The edge_type_dist_stats will now have NUM_EDGE_FEATURES + 1 elements
        print(
            f"  Edge type dist len (NUM_ACTUAL_EDGE_FEATURES + 1 for no-edge): {self.edge_type_dist_stats.size(0) if self.edge_type_dist_stats is not None else 'N/A'}")
        print(f"  Input Dims: {self.input_dims}")
        print(f"  Output Dims: {self.output_dims}")

