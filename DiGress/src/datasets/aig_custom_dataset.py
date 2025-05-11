#!/usr/bin/env python3
import os
import os.path as osp
import pathlib
import math  # For precise splitting
import warnings  # For explicit warnings

import torch
import networkx as nx  # For the validity check conversion
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import numpy as np

# DiGress imports (ensure these paths are correct for your project structure)
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.diffusion.distributions import DistributionNodes

# from src.utils import compute_input_output_dims # Called within AbstractDatasetInfos

from src.aig_config import (
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
    NODE_TYPE_KEYS, EDGE_TYPE_KEYS,  # Needed for PyG to NX conversion
    check_validity  # Your AIG validity function
)

from typing import Union



# --- Helper Function: PyG Data to NetworkX DiGraph ---
# This function is based on the content of pyg_to_nx_conversion_v2
# It's included here for completeness if this script is run standalone,
# or it could be imported if it's in a separate utility file.

def convert_pyg_to_nx_for_aig_validation(pyg_data: Data) -> Union[nx.DiGraph, None]:
    """
    Converts a PyTorch Geometric Data object back to a NetworkX DiGraph,
    setting node and edge 'type' attributes as strings based on NODE_TYPE_KEYS
    and EDGE_TYPE_KEYS from aig_config.py.
    """
    if not all(hasattr(pyg_data, attr) for attr in ['x', 'edge_index', 'edge_attr', 'num_nodes']):
        print("Error converting PyG to NX: pyg_data object is missing required attributes.")
        return None

    nx_graph = nx.DiGraph()
    num_nodes = pyg_data.num_nodes.item()

    if len(NODE_TYPE_KEYS) != NUM_NODE_FEATURES:
        print(f"Error converting PyG to NX: NODE_TYPE_KEYS length ({len(NODE_TYPE_KEYS)}) "
              f"mismatches NUM_NODE_FEATURES ({NUM_NODE_FEATURES}).")
        return None
    if len(EDGE_TYPE_KEYS) != NUM_EDGE_FEATURES:
        print(f"Error converting PyG to NX: EDGE_TYPE_KEYS length ({len(EDGE_TYPE_KEYS)}) "
              f"mismatches NUM_EDGE_FEATURES ({NUM_EDGE_FEATURES}).")
        return None

    if pyg_data.x.shape[0] != num_nodes or pyg_data.x.shape[1] != NUM_NODE_FEATURES:
        print(f"Error converting PyG to NX: Node feature tensor x has incorrect shape.")
        return None

    for i in range(num_nodes):
        node_feature_vector = pyg_data.x[i].cpu().numpy()
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_ATTRIBUTE"  # Consistent with check_validity
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)):
                node_type_str = "UNKNOWN_TYPE_ATTRIBUTE"  # Index out of bounds
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    if pyg_data.edge_index.shape[1] > 0:
        if pyg_data.edge_attr.shape[0] != pyg_data.edge_index.shape[1] or \
                pyg_data.edge_attr.shape[1] != NUM_EDGE_FEATURES:
            print(f"Error converting PyG to NX: Edge attribute tensor edge_attr has incorrect shape.")
            return None  # Or handle more gracefully if partial graphs without valid edge attrs are okay

        for i in range(pyg_data.edge_index.shape[1]):
            src_node = pyg_data.edge_index[0, i].item()
            tgt_node = pyg_data.edge_index[1, i].item()
            edge_feature_vector = pyg_data.edge_attr[i].cpu().numpy()
            if not (np.isclose(np.sum(edge_feature_vector), 1.0) and
                    np.all((np.isclose(edge_feature_vector, 0.0)) | (np.isclose(edge_feature_vector, 1.0)))):
                edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
            else:
                type_index = np.argmax(edge_feature_vector)
                if not (0 <= type_index < len(EDGE_TYPE_KEYS)):
                    edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE"
                else:
                    edge_type_str = EDGE_TYPE_KEYS[type_index]
            nx_graph.add_edge(src_node, tgt_node, type=edge_type_str)

    if hasattr(pyg_data, 'num_inputs') and pyg_data.num_inputs is not None:
        nx_graph.graph['inputs'] = pyg_data.num_inputs.item()
    if hasattr(pyg_data, 'num_outputs') and pyg_data.num_outputs is not None:
        nx_graph.graph['outputs'] = pyg_data.num_outputs.item()
    if hasattr(pyg_data, 'num_gates') and pyg_data.num_gates is not None:
        nx_graph.graph['gates'] = pyg_data.num_gates.item()

    return nx_graph


# --- End Helper Function ---


class AIGCustomDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None,
                 raw_files_for_split=None):
        self.split = split
        self._raw_files_for_split = raw_files_for_split if raw_files_for_split else []
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"Successfully loaded AIGCustomDataset '{split}' with {len(self)} graphs from {self.processed_paths[0]}")

    @property
    def raw_file_names(self):
        return self._raw_files_for_split

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def download(self):
        print(f"AIGCustomDataset ({self.split}): Checking for its specific intermediate raw file in {self.raw_dir}")
        if not self.raw_file_names:
            warnings.warn(f"AIGCustomDataset ({self.split}): No raw files specified for this split by the DataModule.")
            return
        for name in self.raw_file_names:
            raw_file_path = osp.join(self.raw_dir, name)
            if not osp.exists(raw_file_path):
                raise FileNotFoundError(
                    f"Intermediate split raw file {name} (expected at {raw_file_path}) not found for split {self.split}. "
                    f"This file should have been prepared by AIGCustomDataModule."
                )
            print(f"Found intermediate split raw file for {self.split}: {raw_file_path}")

    def process(self):
        if not self.raw_file_names:
            warnings.warn(f"AIGCustomDataset ({self.split}): No raw files to process for this split.")
            torch.save(self.collate([]), self.processed_paths[0])
            return

        path_to_intermediate_split_data = osp.join(self.raw_dir, self.raw_file_names[0])
        print(
            f"AIGCustomDataset ({self.split}): Processing intermediate split data from: {path_to_intermediate_split_data}")

        data_list_for_this_split = torch.load(path_to_intermediate_split_data)

        processed_pyg_data_list = []
        valid_count = 0
        invalid_count = 0

        for i, data_item in enumerate(
                tqdm(data_list_for_this_split, desc=f"Validating and Verifying Data for {self.split} split",
                     unit="graph")):
            required_attrs = ['x', 'edge_index', 'edge_attr', 'num_nodes', 'adj']
            if not all(hasattr(data_item, attr) for attr in required_attrs):
                warnings.warn(f"Graph {i} in {self.split} is missing attributes. Skipping.")
                invalid_count += 1
                continue

            if not hasattr(data_item, 'y'):
                data_item.y = torch.zeros([1, 0]).float()

            if not torch.is_tensor(data_item.num_nodes):
                try:
                    data_item.num_nodes = torch.tensor(int(data_item.num_nodes), dtype=torch.long)
                except ValueError:
                    warnings.warn(f"Graph {i} in {self.split} has non-integer num_nodes. Skipping.")
                    invalid_count += 1
                    continue

            if data_item.num_nodes.item() != data_item.x.shape[0]:
                warnings.warn(f"Graph {i} in {self.split} has inconsistent num_nodes. Correcting.")
                data_item.num_nodes = torch.tensor(data_item.x.shape[0], dtype=torch.long)

            if data_item.edge_index.shape[1] > 0:
                if not (data_item.edge_attr.ndim == 2 and
                        data_item.edge_attr.shape[0] == data_item.edge_index.shape[1] and
                        data_item.edge_attr.shape[1] == NUM_EDGE_FEATURES):
                    warnings.warn(f"Graph {i} in {self.split} has inconsistent edge_attr shape. Skipping.")
                    invalid_count += 1
                    continue
            elif not (data_item.edge_attr.ndim == 2 and
                      data_item.edge_attr.shape[0] == 0 and
                      data_item.edge_attr.shape[1] == NUM_EDGE_FEATURES):
                warnings.warn(f"Graph {i} in {self.split} with no edges has inconsistent edge_attr shape. Correcting.")
                data_item.edge_attr = torch.empty((0, NUM_EDGE_FEATURES), dtype=data_item.x.dtype,
                                                  device=data_item.x.device)

            # Convert to NetworkX for AIG-specific validation
            nx_graph_for_validation = convert_pyg_to_nx_for_aig_validation(data_item)

            if nx_graph_for_validation is None:
                warnings.warn(f"Graph {i} in {self.split} failed PyG to NX conversion. Skipping.")
                invalid_count += 1
                continue

            # Perform the AIG validity check
            if check_validity(nx_graph_for_validation):
                processed_pyg_data_list.append(data_item)
                valid_count += 1
            else:
                # Optionally log which graph failed or specific reasons if check_validity provides them
                # print(f"Graph {i} in {self.split} failed AIG check_validity. Skipping.")
                invalid_count += 1

        print(
            f"AIGCustomDataset ({self.split}): Validation complete. Valid graphs: {valid_count}, Invalid/Skipped graphs: {invalid_count}")

        if self.pre_filter is not None:
            # Note: pre_filter will run on already AIG-validated graphs
            processed_pyg_data_list = [data for data in processed_pyg_data_list if self.pre_filter(data)]
            print(
                f"AIGCustomDataset ({self.split}): After pre_filter, {len(processed_pyg_data_list)} graphs remaining.")
        if self.pre_transform is not None:
            processed_pyg_data_list = [self.pre_transform(data) for data in processed_pyg_data_list]
            print(
                f"AIGCustomDataset ({self.split}): After pre_transform, {len(processed_pyg_data_list)} graphs remaining.")

        num_output_graphs = len(processed_pyg_data_list)

        if num_output_graphs == 0 and (
                valid_count > 0 or invalid_count > 0):  # Had graphs, but all filtered out by pre_filter/transform
            warnings.warn(
                f"AIGCustomDataset ({self.split}): All {valid_count} valid graphs were filtered out by pre_filter/pre_transform.")
        elif num_output_graphs == 0:
            warnings.warn(f"AIGCustomDataset ({self.split}): No data to save after validation and processing.")

        print(
            f"AIGCustomDataset ({self.split}): Saving {num_output_graphs} processed and AIG-valid graphs to {self.processed_paths[0]}...")
        torch.save(self.collate(processed_pyg_data_list), self.processed_paths[0])


class AIGCustomDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        base_path = pathlib.Path(cfg.general.abs_path_to_project_root)
        dataset_instance_root = base_path / cfg.dataset.datadir / cfg.dataset.name

        path_to_combined_raw_dir = base_path / cfg.dataset.datadir_for_all_raw
        combined_raw_filename = cfg.dataset.all_raw_graphs_filename
        path_to_combined_raw_file = path_to_combined_raw_dir / combined_raw_filename
        import os
        # This is the path string that your error message says it's looking for
        relative_path_in_error = "data/pyg_full/all_graphs_pyg.pt"

        # Let's see what the CWD is
        script_cwd = os.getcwd()
        print(f"DEBUG: Script's Current Working Directory is: {script_cwd}")

        # Let's see what the absolute path being tried is
        # The variable path_to_combined_raw_file holds the relative path.
        # Let's assume its value is exactly what the error shows.
        # When open() or similar functions get a relative path, they resolve it against CWD.
        absolute_path_being_attempted = os.path.abspath(
            relative_path_in_error)  # or os.path.abspath(path_to_combined_raw_file)
        print(f"DEBUG: Absolute path being attempted by script: {absolute_path_being_attempted}")
        print(f"DEBUG: Does this absolute path exist? {os.path.exists(absolute_path_being_attempted)}")
        print(f"DEBUG: Is it a file? {os.path.isfile(absolute_path_being_attempted)}")

        intermediate_splits_raw_dir = dataset_instance_root / "raw"
        final_processed_dir = dataset_instance_root / "processed"
        os.makedirs(intermediate_splits_raw_dir, exist_ok=True)
        os.makedirs(final_processed_dir, exist_ok=True)

        train_split_intermediate_basename = "train_split_intermediate.pt"
        val_split_intermediate_basename = "val_split_intermediate.pt"
        test_split_intermediate_basename = "test_split_intermediate.pt"

        path_train_split_intermediate = intermediate_splits_raw_dir / train_split_intermediate_basename
        path_val_split_intermediate = intermediate_splits_raw_dir / val_split_intermediate_basename
        path_test_split_intermediate = intermediate_splits_raw_dir / test_split_intermediate_basename

        force_resplit = cfg.dataset.get('force_resplit_raw', False)
        if force_resplit or not (osp.exists(path_train_split_intermediate) and
                                 osp.exists(path_val_split_intermediate) and
                                 osp.exists(path_test_split_intermediate)):
            if not osp.exists(path_to_combined_raw_file):
                raise FileNotFoundError(f"The combined raw AIG data file was not found: {path_to_combined_raw_file}")

            print(f"AIGCustomDataModule: Loading all raw graphs from: {path_to_combined_raw_file} for splitting.")
            all_graphs_list = torch.load(path_to_combined_raw_file)

            num_total_graphs = len(all_graphs_list)
            if num_total_graphs < 3:
                raise ValueError(f"Too few graphs ({num_total_graphs}) to split. Need at least 3.")

            idx_train_end = math.floor(num_total_graphs * 4 / 6)
            idx_val_end = math.floor(num_total_graphs * 5 / 6)

            if idx_train_end == num_total_graphs:
                idx_train_end = num_total_graphs - 2
                idx_val_end = num_total_graphs - 1
            elif idx_val_end == num_total_graphs:
                if idx_train_end < num_total_graphs - 1:
                    idx_val_end = num_total_graphs - 1

            train_graphs = all_graphs_list[:idx_train_end]
            val_graphs = all_graphs_list[idx_train_end:idx_val_end]
            test_graphs = all_graphs_list[idx_val_end:]

            if not test_graphs and val_graphs and idx_val_end == num_total_graphs and len(val_graphs) > 1:
                test_graphs = [val_graphs.pop(-1)]

            if not train_graphs or not val_graphs or not test_graphs:
                warnings.warn(
                    f"One or more splits are empty. Total: {num_total_graphs}, Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}.")

            print(
                f"AIGCustomDataModule: Splitting data (ordered): Train {len(train_graphs)}, Val {len(val_graphs)}, Test {len(test_graphs)}")

            torch.save(train_graphs, path_train_split_intermediate)
            print(f"AIGCustomDataModule: Saved training intermediate data to {path_train_split_intermediate}")
            torch.save(val_graphs, path_val_split_intermediate)
            print(f"AIGCustomDataModule: Saved validation intermediate data to {path_val_split_intermediate}")
            torch.save(test_graphs, path_test_split_intermediate)
            print(f"AIGCustomDataModule: Saved test intermediate data to {path_test_split_intermediate}")
        else:
            print(
                f"AIGCustomDataModule: Intermediate split files found in {intermediate_splits_raw_dir}. Using existing splits.")

        datasets = {
            'train': AIGCustomDataset(root=str(dataset_instance_root), split='train',
                                      raw_files_for_split=[train_split_intermediate_basename]),
            'val': AIGCustomDataset(root=str(dataset_instance_root), split='val',
                                    raw_files_for_split=[val_split_intermediate_basename]),
            'test': AIGCustomDataset(root=str(dataset_instance_root), split='test',
                                     raw_files_for_split=[test_split_intermediate_basename])
        }

        super().__init__(cfg, datasets)


class AIGCustomDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: AIGCustomDataModule, dataset_config):
        super().__init__(datamodule, dataset_config)

        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()

        super().complete_infos(self.n_nodes, self.node_types)

        extra_features_cfg = datamodule.cfg.dataset.get('extra_features', None)
        domain_features_cfg = datamodule.cfg.dataset.get('domain_features', None)

        self.compute_input_output_dims(datamodule,
                                       extra_features_cfg,
                                       domain_features_cfg)
        print(f"AIGCustomDatasetInfos initialized for '{self.name}'. Max nodes: {self.max_n_nodes}.")
        print(f"  Node type distribution size: {self.node_types.size(0) if self.node_types is not None else 'N/A'}")
        print(f"  Edge type distribution size: {self.edge_types.size(0) if self.edge_types is not None else 'N/A'}")


#
