#!/usr/bin/env python3
import os
import os.path as osp
import pathlib
import warnings  # For explicit warnings
import math
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils # Keep this for potential future use, though not directly used in snippet

from tqdm import tqdm
import numpy as np

# DiGress imports
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
# from src.diffusion.distributions import DistributionNodes # Not directly used in this file

# Your AIG specific imports
from src.aig_config import (
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
    NODE_TYPE_KEYS, EDGE_TYPE_KEYS,
    check_validity  # AIG-specific structural validation
)
from typing import Union, List


def convert_pyg_to_nx_for_aig_validation(pyg_data: Data) -> Union[nx.DiGraph, None]:  # Changed DiGraph to Graph
    """
    Converts a PyTorch Geometric Data object, specifically formatted for AIGs
    (with edge_attr having NUM_EDGE_FEATURES + 1 dimensions), to a NetworkX Graph (undirected)
    for validation purposes.
    """
    # Basic attribute checks
    if not all(hasattr(pyg_data, attr) for attr in ['x', 'edge_index', 'edge_attr']):
        warnings.warn(
            "convert_pyg_to_nx: pyg_data object missing one or more core attributes (x, edge_index, edge_attr).")
        return None
    if pyg_data.num_nodes is None:  # num_nodes can be explicitly set or inferred by PyG
        warnings.warn("convert_pyg_to_nx: pyg_data.num_nodes is None.")
        return None

    num_nodes = pyg_data.num_nodes  # This should be an int after .item() if it was a tensor

    if pyg_data.x is None or pyg_data.x.shape[0] != num_nodes or pyg_data.x.shape[1] != NUM_NODE_FEATURES:
        warnings.warn(
            f"convert_pyg_to_nx: Node feature tensor 'x' mismatch. Shape: {pyg_data.x.shape if pyg_data.x is not None else 'None'}, Expected nodes: {num_nodes}, Expected features: {NUM_NODE_FEATURES}")
        return None

    nx_graph = nx.DiGraph()  # MODIFICATION: Changed DiGraph to Graph

    # Process nodes
    for i in range(num_nodes):
        node_feature_vector = pyg_data.x[i].cpu().numpy()
        # Check if one-hot encoded
        if not (np.isclose(np.sum(node_feature_vector), 1.0) and
                np.all((np.isclose(node_feature_vector, 0.0)) | (np.isclose(node_feature_vector, 1.0)))):
            node_type_str = "UNKNOWN_TYPE_ATTRIBUTE_NON_ONE_HOT"
            warnings.warn(f"Node {i} features not one-hot: {node_feature_vector}")
        else:
            type_index = np.argmax(node_feature_vector)
            if not (0 <= type_index < len(NODE_TYPE_KEYS)):
                node_type_str = "UNKNOWN_TYPE_ATTRIBUTE_BAD_INDEX"
                warnings.warn(
                    f"Node {i} type index {type_index} out of bounds for NODE_TYPE_KEYS (len {len(NODE_TYPE_KEYS)}).")
            else:
                node_type_str = NODE_TYPE_KEYS[type_index]
        nx_graph.add_node(i, type=node_type_str)

    # Process edges
    if pyg_data.edge_index is not None and pyg_data.edge_index.shape[1] > 0:
        if pyg_data.edge_attr is None:
            warnings.warn("convert_pyg_to_nx: Graph has edges but 'edge_attr' is None.")
            return None

        expected_edge_attr_dim = NUM_EDGE_FEATURES + 1
        if pyg_data.edge_attr.shape[0] != pyg_data.edge_index.shape[1] or \
                pyg_data.edge_attr.shape[1] != expected_edge_attr_dim:
            warnings.warn(f"convert_pyg_to_nx: Edge attribute tensor 'edge_attr' has incorrect shape. "
                          f"Got {pyg_data.edge_attr.shape}, expected ({pyg_data.edge_index.shape[1]}, {expected_edge_attr_dim}). "
                          "This implies the input pyg_data might not have been processed by AIGDataset.process correctly.")
            return None

        for i in range(pyg_data.edge_index.shape[1]):
            src, tgt = pyg_data.edge_index[0, i].item(), pyg_data.edge_index[1, i].item()
            edge_feature_vector_processed = pyg_data.edge_attr[i].cpu().numpy()

            if not (np.isclose(np.sum(edge_feature_vector_processed), 1.0) and
                    np.all((np.isclose(edge_feature_vector_processed, 0.0)) | (
                    np.isclose(edge_feature_vector_processed, 1.0)))):
                edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE_NON_ONE_HOT"
                warnings.warn(f"Edge ({src}-{tgt}) processed features not one-hot: {edge_feature_vector_processed}")
            else:
                shifted_type_index = np.argmax(edge_feature_vector_processed)
                if shifted_type_index == 0:
                    edge_type_str = "NO_SPECIFIC_AIG_TYPE_OR_ABSENT_EDGE"
                    # This warning might be frequent if index 0 is a common "generic" edge type post-processing
                    # warnings.warn(f"Edge ({src}-{tgt}) has type index 0 in its {expected_edge_attr_dim}-dim 'edge_attr'. This implies no specific AIG type.")
                else:
                    actual_aig_type_index = shifted_type_index - 1
                    if not (0 <= actual_aig_type_index < len(EDGE_TYPE_KEYS)):
                        edge_type_str = "UNKNOWN_TYPE_ATTRIBUTE_BAD_INDEX"
                        warnings.warn(
                            f"Edge ({src}-{tgt}) decoded to invalid actual_aig_type_index {actual_aig_type_index} from shifted index {shifted_type_index}.")
                    else:
                        edge_type_str = EDGE_TYPE_KEYS[actual_aig_type_index]


            nx_graph.add_edge(src, tgt, type=edge_type_str)

    # Add graph-level attributes
    for attr_name in ['inputs', 'outputs', 'gates', 'name', 'filename']:
        if hasattr(pyg_data, attr_name):
            attr_val = getattr(pyg_data, attr_name)
            if isinstance(attr_val, torch.Tensor) and attr_val.numel() == 1:
                nx_graph.graph[attr_name] = attr_val.item()
            elif isinstance(attr_val, (str, int, float, list, dict)):
                nx_graph.graph[attr_name] = attr_val
    return nx_graph

class AIGDataset(InMemoryDataset):
    def __init__(self, split: str, root: str, cfg, dataset_name: str = 'aig', transform=None, pre_transform=None,
                 pre_filter=None):
        self.dataset_name = cfg.dataset.name
        self.split = split
        self.cfg = cfg # Hydra config
        if self.cfg is None:
            raise ValueError("cfg (Hydra config) must be provided to AIGDataset.")

        # Determine file_idx based on split, used for accessing raw_paths
        if self.split == 'train':
            self.file_idx = 0
        elif self.split == 'val':
            self.file_idx = 1
        elif self.split == 'test':
            self.file_idx = 2
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")

        super().__init__(root, transform, pre_transform, pre_filter)
        # Load processed data for the current split
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Successfully loaded processed AIGDataset for split '{split}' with {len(self)} graphs from {self.processed_paths[0]}")
        except FileNotFoundError:
            warnings.warn(
                f"Processed file for AIGDataset split '{split}' not found at {self.processed_paths[0]}. "
                f"It will be created if `process()` is called (e.g., on first instantiation if raw files exist).")
        except Exception as e:
            warnings.warn(f"Error loading processed file for AIGDataset split '{split}' at {self.processed_paths[0]}: {e}")

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files specific to each split, expected in self.raw_dir."""
        return [f'{s}_split_raw.pt' for s in ['train', 'val', 'test']]

    @property
    def processed_file_names(self) -> List[str]:
        """Name of the processed file for the *current* split, saved in self.processed_dir."""
        return [f'{self.split}_processed.pt']

    def download(self):
        """
        Ensures that the raw data for each split (train_split_raw.pt, etc.) exists.
        If not, it loads a single source AIG file (specified in cfg) and splits it.
        This method is called by InMemoryDataset if raw files are not found.
        """
        all_individual_raw_splits_exist = all(osp.exists(p) for p in self.raw_paths)

        if all_individual_raw_splits_exist and not self.cfg.dataset.get('force_resplit_raw', False):
            print(f"All per-split raw files found in {self.raw_dir}. Skipping source file splitting.")
            return

        print(f"One or more per-split raw files missing or force_resplit_raw=True. "
              f"Attempting to split from source AIG file specified in config...")

        source_file_path = pathlib.Path(
            self.cfg.general.abs_path_to_project_root
        ) / self.cfg.dataset.datadir_for_all_raw / self.cfg.dataset.all_raw_graphs_filename

        if not osp.exists(source_file_path):
            raise FileNotFoundError(
                f"Source raw AIG data file not found: {source_file_path}. "
                "This file is expected to be a .pt file containing a list of PyG Data objects, "
                "created by your AIG processing scripts (e.g., create_dataset.py)."
            )

        print(f"Loading all graphs from source file: {source_file_path}")
        all_graphs_list_from_source = torch.load(source_file_path)
        num_total_graphs = len(all_graphs_list_from_source)

        if num_total_graphs == 0:
            warnings.warn(f"Source raw AIG data file {source_file_path} is empty. Creating empty raw split files.")
            os.makedirs(self.raw_dir, exist_ok=True)
            for raw_path in self.raw_paths:
                torch.save([], raw_path)
            return

        num_train_default = math.floor(num_total_graphs * (4 / 6))
        num_val_default = math.floor(num_total_graphs * (1 / 6))
        train_len_cfg = self.cfg.dataset.get('num_train_graphs', num_train_default)
        val_len_cfg = self.cfg.dataset.get('num_val_graphs', num_val_default)
        train_len = min(train_len_cfg, num_total_graphs)
        val_len = min(val_len_cfg, num_total_graphs - train_len)
        test_len = num_total_graphs - train_len - val_len

        if test_len < 0:
            warnings.warn(f"Specified train/val lengths ({train_len}, {val_len}) exceed total graphs ({num_total_graphs}). Adjusting test_len.")
            test_len = 0
            val_len = num_total_graphs - train_len
            if val_len < 0:
                train_len = num_total_graphs
                val_len = 0

        print(f"Total graphs from source: {num_total_graphs}. "
              f"Splitting into: Train={train_len}, Val={val_len}, Test={test_len}")

        train_data = all_graphs_list_from_source[:train_len]
        val_data = all_graphs_list_from_source[train_len : train_len + val_len]
        test_data = all_graphs_list_from_source[train_len + val_len:]

        os.makedirs(self.raw_dir, exist_ok=True)
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])
        print(f"Saved raw splits to {self.raw_dir}: "
              f"Train ({len(train_data)} to {self.raw_paths[0]}), "
              f"Val ({len(val_data)} to {self.raw_paths[1]}), "
              f"Test ({len(test_data)} to {self.raw_paths[2]})")

    def process(self):
        current_split_raw_data_path = self.raw_paths[self.file_idx]
        print(f"AIGDataset (split: '{self.split}'): Processing raw data from: {current_split_raw_data_path}")

        if not osp.exists(current_split_raw_data_path):
            raise FileNotFoundError(
                f"Raw data file for split '{self.split}' not found at {current_split_raw_data_path}. "
                "Ensure `download()` method has run successfully or the file exists."
            )

        current_split_graphs_raw = torch.load(current_split_raw_data_path)
        if not isinstance(current_split_graphs_raw, list):
            raise TypeError(f"Raw data file {current_split_raw_data_path} did not contain a list of graphs.")

        processed_pyg_data_list = []
        valid_graph_count, invalid_graph_count = 0, 0
        new_edge_attr_dim = NUM_EDGE_FEATURES + 1

        for i, data_item_raw in enumerate(
                tqdm(current_split_graphs_raw, desc=f"Processing and Validating '{self.split}' split", unit="graph")):
            data_item = data_item_raw.clone()

            if not all(hasattr(data_item, attr) for attr in ['x', 'edge_index']):
                warnings.warn(f"Graph {i} in '{self.split}' split is missing basic attributes (x, edge_index). Skipping.")
                invalid_graph_count += 1
                continue

            if not hasattr(data_item, 'y') or data_item.y is None:
                data_item.y = torch.zeros([1, 0], dtype=data_item.x.dtype if data_item.x is not None else torch.float)

            if not hasattr(data_item, 'num_nodes') or data_item.num_nodes is None:
                if data_item.x is not None:
                    data_item.num_nodes = data_item.x.shape[0]
                else:
                    warnings.warn(f"Graph {i} in '{self.split}' missing 'num_nodes' and 'x'. Skipping.")
                    invalid_graph_count += 1
                    continue
            elif torch.is_tensor(data_item.num_nodes):
                 if data_item.num_nodes.numel() != 1:
                    warnings.warn(f"Graph {i} 'num_nodes' tensor has multiple elements. Skipping.")
                    invalid_graph_count += 1; continue
                 data_item.num_nodes = data_item.num_nodes.item()
            elif not isinstance(data_item.num_nodes, int):
                try:
                    data_item.num_nodes = int(data_item.num_nodes)
                except ValueError:
                    warnings.warn(f"Graph {i} 'num_nodes' is not a convertible integer. Skipping.")
                    invalid_graph_count += 1; continue

            if data_item.x is None or data_item.x.shape[0] != data_item.num_nodes:
                warnings.warn(
                    f"Graph {i} num_nodes ({data_item.num_nodes}) vs x.shape[0] "
                    f"({data_item.x.shape[0] if data_item.x is not None else 'None'}) mismatch. Skipping."
                )
                invalid_graph_count += 1
                continue
            if data_item.x.shape[1] != NUM_NODE_FEATURES:
                warnings.warn(
                    f"Graph {i} node features dim is {data_item.x.shape[1]}, expected {NUM_NODE_FEATURES}. Skipping."
                )
                invalid_graph_count +=1; continue

            if data_item.edge_index.shape[1] > 0:
                if not hasattr(data_item, 'edge_attr') or data_item.edge_attr is None:
                    warnings.warn(f"Graph {i} in '{self.split}' has edges but no 'edge_attr'. Skipping.")
                    invalid_graph_count += 1
                    continue

                original_edge_attr = data_item.edge_attr
                if original_edge_attr.shape[1] != NUM_EDGE_FEATURES:
                    warnings.warn(
                        f"Graph {i} in '{self.split}' has original edge_attr dim {original_edge_attr.shape[1]}, "
                        f"but expected {NUM_EDGE_FEATURES}. Skipping."
                    )
                    invalid_graph_count += 1
                    continue

                num_edges = original_edge_attr.shape[0]
                # Create a column of zeros for the "no specific AIG type" channel
                zeros_column = torch.zeros((num_edges, 1),
                                           dtype=original_edge_attr.dtype,
                                           device=original_edge_attr.device)
                # Concatenate the zeros column with the original edge attributes
                # original_edge_attr has actual AIG types (e.g., REG, INV)
                # new_ea will have shape (num_edges, NUM_EDGE_FEATURES + 1)
                # where index 0 is for "no specific type", and indices 1..NUM_EDGE_FEATURES are for actual types
                new_ea = torch.cat((zeros_column, original_edge_attr), dim=1)
                data_item.edge_attr = new_ea
            else:
                data_item.edge_attr = torch.empty((0, new_edge_attr_dim),
                                                  dtype=data_item.x.dtype if data_item.x is not None else torch.float, # Match node feature dtype if x exists
                                                  device=data_item.x.device if data_item.x is not None else torch.device('cpu')) # Match node feature device or default to cpu

            nx_graph_for_validation = convert_pyg_to_nx_for_aig_validation(data_item)
            if nx_graph_for_validation is None:
                invalid_graph_count += 1
                continue

            if check_validity(nx_graph_for_validation):
                if hasattr(data_item, 'adj'):
                    delattr(data_item, 'adj')
                processed_pyg_data_list.append(data_item)
                valid_graph_count += 1
            else:
                invalid_graph_count += 1

        print(f"AIGDataset (split: '{self.split}'): Processing complete. "
              f"Valid AIGs kept: {valid_graph_count}, Invalid/Skipped: {invalid_graph_count}")

        if self.pre_filter is not None:
            processed_pyg_data_list = [d for d in processed_pyg_data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            processed_pyg_data_list = [self.pre_transform(d) for d in processed_pyg_data_list]

        if not processed_pyg_data_list:
            warnings.warn(f"AIGDataset (split: '{self.split}'): No data to save after processing and validation. "
                          f"The file {self.processed_paths[0]} will contain an empty list.")

        torch.save(self.collate(processed_pyg_data_list), self.processed_paths[0])
        print(f"AIGDataset (split: '{self.split}'): Saved {len(processed_pyg_data_list)} processed graphs to {self.processed_paths[0]}")


class AIGDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(cfg.general.abs_path_to_project_root)
        dataset_instance_root = base_path / self.datadir / cfg.dataset.name

        print(f"AIGDataModule: Root directory for AIGDataset instance '{cfg.dataset.name}' will be: {dataset_instance_root}")

        datasets = {
            'train': AIGDataset(split='train', root=str(dataset_instance_root), cfg=cfg),
            'val': AIGDataset(split='val', root=str(dataset_instance_root), cfg=cfg),
            'test': AIGDataset(split='test', root=str(dataset_instance_root), cfg=cfg)
        }
        super().__init__(cfg, datasets)


class AIGDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: AIGDataModule, dataset_config: dict):
        self.name = datamodule.cfg.dataset.name
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        from src.diffusion.extra_features import DummyExtraFeatures

        extra_features_name = getattr(datamodule.cfg.model, "extra_features", None)
        domain_features_name = getattr(datamodule.cfg.model, "domain_features", None)

        if extra_features_name is None:
            extra_features_fn = DummyExtraFeatures()
        else:
            warnings.warn(f"AIG model has extra_features '{extra_features_name}' specified, but only DummyExtraFeatures is used by default here. Ensure this is intended.")
            extra_features_fn = DummyExtraFeatures()

        if domain_features_name is None:
            domain_features_fn = DummyExtraFeatures()
        else:
            warnings.warn(f"AIG model has domain_features '{domain_features_name}' specified, but only DummyExtraFeatures is used by default here. Ensure this is intended.")
            domain_features_fn = DummyExtraFeatures()

        self.compute_input_output_dims(datamodule, extra_features_fn, domain_features_fn)

        print(f"Initialized AIGDatasetInfos for dataset: '{self.name}'")
        print(f"  Max nodes in dataset: {self.max_n_nodes}")
        print(f"  Number of node types (classes for prediction): {self.num_classes} (should be {NUM_NODE_FEATURES})")
        if self.node_types is not None:
            print(f"  Node type distribution tensor length: {self.node_types.size(0)}")
        else:
            print("  Node type distribution not available.")
        if self.edge_types is not None:
            print(f"  Edge type distribution tensor length: {self.edge_types.size(0)} (should be {NUM_EDGE_FEATURES + 1})")
        else:
            print("  Edge type distribution not available.")
        print(f"  Input Dims for model: {self.input_dims}")
        print(f"  Output Dims for model: {self.output_dims}")

        # --- Corrected Sanity Checks for AIG ---
        # self.input_dims has been computed by the superclass method,
        # incorporating NUM_NODE_FEATURES and any extra feature dimensions.
        # For AIG with DummyExtraFeatures, extra feature dimensions are 0.

        # Check for X input dimension
        # Base input dim for X is NUM_NODE_FEATURES. Dummy features add 0.
        expected_x_input_dim_base = NUM_NODE_FEATURES
        if self.input_dims['X'] != expected_x_input_dim_base:
             warnings.warn(
                 f"Input X dim ({self.input_dims['X']}) mismatch. Expected base: {expected_x_input_dim_base} (NUM_NODE_FEATURES). "
                 f"This implies extra_features might not be dummy or there's an issue in input_dims calculation."
             )

        # Check for X output dimension
        if self.output_dims['X'] != NUM_NODE_FEATURES:
             warnings.warn(f"Output X dim ({self.output_dims['X']}) mismatch with NUM_NODE_FEATURES ({NUM_NODE_FEATURES}).")

        # Check for E input dimension
        # Base input dim for E is NUM_EDGE_FEATURES + 1 (due to the added 'no specific type' channel). Dummy features add 0.
        expected_e_input_dim_base = NUM_EDGE_FEATURES + 1
        if self.input_dims['E'] != expected_e_input_dim_base:
             warnings.warn(
                 f"Input E dim ({self.input_dims['E']}) mismatch. Expected base: {expected_e_input_dim_base} (NUM_EDGE_FEATURES + 1). "
                 f"This implies extra_features might not be dummy or there's an issue in input_dims calculation."
             )

        # Check for E output dimension
        if self.output_dims['E'] != (NUM_EDGE_FEATURES + 1):
             warnings.warn(f"Output E dim ({self.output_dims['E']}) mismatch with NUM_EDGE_FEATURES+1 ({NUM_EDGE_FEATURES+1}).")
