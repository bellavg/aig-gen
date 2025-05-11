#!/usr/bin/env python3
import os
import os.path as osp
import pathlib  # For robust path manipulation
import random  # For splitting data

import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

# DiGress imports (adjust paths if your file is located differently)
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.diffusion.distributions import DistributionNodes
# from src.utils import compute_input_output_dims # If you need to call this directly,
# otherwise it's called within AbstractDatasetInfos

# Assuming your aig_config.py is in a place Python can find it,
# or adjust the import path. For example, if aig_config.py is in the project root:
# import sys
# sys.path.append(str(pathlib.Path(__file__).resolve().parents[2])) # Add project root to path
from aig_config import NUM_NODE_FEATURES, NUM_EXPLICIT_EDGE_TYPES


class AIGCustomDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None,
                 raw_files_for_split=None):  # Renamed for clarity
        self.split = split
        # This list should contain the filenames (not full paths) of the raw files
        # that are relevant for THIS specific split (e.g., ['train_aig_data.pt'])
        # These files are expected to be in self.raw_dir.
        self._raw_files_for_split = raw_files_for_split if raw_files_for_split else []

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"Loaded AIGCustomDataset '{split}' with {len(self)} graphs from {self.processed_paths[0]}")

    @property
    def raw_file_names(self):
        # Returns the list of raw file names specific to this split.
        # These files should be located by the download() method in self.raw_dir.
        return self._raw_files_for_split

    @property
    def processed_file_names(self):
        # This is the single file this InMemoryDataset will save after its processing.
        return [f'aig_processed_{self.split}.pt']

    def download(self):
        # This method ensures that the files listed in self.raw_file_names()
        # are present in self.raw_dir.
        # Your data conversion script creates files like "..._raw_pyg_topo.pt".
        # You need a strategy to make these available for each split.

        # Example Strategy:
        # Your Hydra config for the dataset might specify:
        # dataset:
        #   name: my_aig_dataset
        #   datadir: data/my_aig_dataset_root
        #   train_input_raw_files: ["path/to/your/project/data/raw_pyg/train_part1_raw_pyg_topo.pt", ...]
        #   val_input_raw_files: ["path/to/your/project/data/raw_pyg/val_part1_raw_pyg_topo.pt"]
        #   test_input_raw_files: ["path/to/your/project/data/raw_pyg/test_part1_raw_pyg_topo.pt"]

        # The AIGCustomDataModule would then pass the BASENAMES of these files
        # (after potentially copying them or ensuring they are symlinked to self.raw_dir)
        # via the `raw_files_for_split` argument to this Dataset.

        # For this example, we'll assume the files passed in _raw_files_for_split
        # are already just the basenames and are expected to be in self.raw_dir.
        # If not, this download() method would need to copy them from their original location
        # (e.g., specified in cfg) to self.raw_dir.

        print(f"AIGCustomDataset ({self.split}): Checking for raw files in {self.raw_dir}")
        if not self._raw_files_for_split:
            print(f"Warning ({self.split}): No raw files specified for this split.")
            return

        for name in self.raw_file_names:  # These are now the basenames from _raw_files_for_split
            raw_file_path = osp.join(self.raw_dir, name)
            if not osp.exists(raw_file_path):
                # This is where you might copy files from a central location
                # (e.g., ./data/raw_pyg/ from your script's output) to self.raw_dir
                # For now, it raises an error if not found directly.
                raise FileNotFoundError(
                    f"Raw file {name} (expected at {raw_file_path}) not found for split {self.split}. "
                    f"Ensure files are correctly placed in the 'raw' directory of the dataset root, "
                    f"or adjust the logic in AIGCustomDataModule to prepare them."
                )
            print(f"Found raw file for {self.split}: {raw_file_path}")

    def process(self):
        all_data_objects_for_split = []

        if not self.raw_file_names:  # Should be self._raw_files_for_split
            print(f"Warning ({self.split}): No raw files to process for this split.")
            # Save an empty collated list if PyG requires a processed file.
            torch.save(self.collate([]), self.processed_paths[0])
            return

        # Iterate through all raw files designated for this split
        for raw_filename in self.raw_file_names:  # Uses the property, which uses _raw_files_for_split
            path_to_raw_list_of_data = osp.join(self.raw_dir, raw_filename)
            print(f"Processing raw data from: {path_to_raw_list_of_data} for split {self.split}")

            # This file contains a list of Data(x, adj) objects from your script
            raw_pyg_data_list_from_one_file = torch.load(path_to_raw_list_of_data)
            all_data_objects_for_split.extend(raw_pyg_data_list_from_one_file)

        processed_pyg_data_list = []
        for i, raw_data_item in enumerate(
                tqdm(all_data_objects_for_split, desc=f"Converting dense adj for {self.split} split")):
            if not hasattr(raw_data_item, 'x') or not hasattr(raw_data_item, 'adj'):
                print(f"Warning: Skipping item {i} in {self.split} due to missing 'x' or 'adj' attribute.")
                continue

            x = raw_data_item.x
            dense_adj_tensor = raw_data_item.adj  # Shape (N, N, num_explicit_edge_types + 1)

            num_nodes = x.shape[0]

            current_edge_indices = []
            current_edge_attrs = []

            for src_node_idx in range(num_nodes):
                for tgt_node_idx in range(num_nodes):
                    # Iterate over explicit edge type channels
                    for edge_type_channel in range(NUM_EXPLICIT_EDGE_TYPES):
                        if dense_adj_tensor[src_node_idx, tgt_node_idx, edge_type_channel].item() == 1.0:
                            current_edge_indices.append([src_node_idx, tgt_node_idx])
                            attr = torch.zeros(NUM_EXPLICIT_EDGE_TYPES, dtype=torch.float)
                            attr[edge_type_channel] = 1.0
                            current_edge_attrs.append(attr)

            if current_edge_indices:
                edge_index = torch.tensor(current_edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.stack(current_edge_attrs)
            else:  # Handle graphs with no edges
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, NUM_EXPLICIT_EDGE_TYPES), dtype=torch.float)

            pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            pyg_data.num_nodes = torch.tensor(num_nodes, dtype=torch.long)

            if hasattr(raw_data_item, 'num_inputs'): pyg_data.num_inputs = raw_data_item.num_inputs
            if hasattr(raw_data_item, 'num_outputs'): pyg_data.num_outputs = raw_data_item.num_outputs
            if hasattr(raw_data_item, 'num_gates'): pyg_data.num_gates = raw_data_item.num_gates
            pyg_data.y = getattr(raw_data_item, 'y', torch.zeros([1, 0]).float())

            processed_pyg_data_list.append(pyg_data)

        if self.pre_filter is not None:
            processed_pyg_data_list = [data for data in processed_pyg_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            processed_pyg_data_list = [self.pre_transform(data) for data in processed_pyg_data_list]

        if not processed_pyg_data_list and len(all_data_objects_for_split) > 0:  # All items were filtered out
            print(f"Warning: All data items were filtered out for split {self.split}.")
        elif not processed_pyg_data_list:  # No items to begin with or all filtered
            print(f"Warning: No data to save for split {self.split} after processing/filtering.")

        print(
            f"Saving {len(processed_pyg_data_list)} processed graphs for {self.split} to {self.processed_paths[0]}...")
        torch.save(self.collate(processed_pyg_data_list), self.processed_paths[0])


class AIGCustomDataModule(AbstractDataModule):  # Inherit from DiGress's AbstractDataModule
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir

        # project_root/data_cfg.dataset.datadir/my_aig_dataset_name
        # This is where the InMemoryDataset will create/look for processed/ and raw/
        base_path = pathlib.Path(cfg.general.abs_path_to_project_root)
        dataset_root_path = base_path / self.datadir / cfg.dataset.name
        os.makedirs(dataset_root_path / "raw", exist_ok=True)  # Ensure raw dir exists for dataset
        os.makedirs(dataset_root_path / "processed", exist_ok=True)  # Ensure processed dir exists

        # The raw files (output of your script) need to be copied or symlinked
        # into dataset_root_path / "raw" / <file_name_as_in_cfg> by your setup.
        # AIGCustomDataset.download() will then look for them there.
        # For example, if cfg.dataset.train_raw_filenames_basenames = ['train_aigs.pt']
        # then 'train_aigs.pt' must exist in dataset_root_path / "raw" / 'train_aigs.pt'
        # The list itself comes from the config.

        datasets = {
            'train': AIGCustomDataset(root=str(dataset_root_path), split='train',
                                      raw_files_for_split=cfg.dataset.train_raw_filenames_basenames),
            'val': AIGCustomDataset(root=str(dataset_root_path), split='val',
                                    raw_files_for_split=cfg.dataset.val_raw_filenames_basenames),
            'test': AIGCustomDataset(root=str(dataset_root_path), split='test',
                                     raw_files_for_split=cfg.dataset.test_raw_filenames_basenames)
        }

        super().__init__(cfg, datasets)


class AIGCustomDatasetInfos(AbstractDatasetInfos):  # Inherit from DiGress's AbstractDatasetInfos
    def __init__(self, datamodule: AIGCustomDataModule, dataset_config, extra_features_cfg=None,
                 domain_features_cfg=None):
        super().__init__(datamodule, dataset_config)  # Basic init from parent

        # These methods are from DiGress's AbstractDataModule
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()  # Computes dist over data.x
        self.edge_types = self.datamodule.edge_counts()  # Computes dist over data.edge_attr (incl. no-edge)

        # This completes additional info like max_n_nodes, nodes_dist
        super().complete_infos(self.n_nodes, self.node_types)

        # This computes input/output dimensions for the model based on example batches
        # and extra features. If you don't use extra features, pass None or ensure
        # the ExtraFeatures class handles None config.
        # from src.diffusion.extra_features import ExtraFeatures # Example import
        # from src.diffusion.extra_features_molecular import ExtraMolecularFeatures # Example import
        # extra_features_processor = ExtraFeatures(extra_features_cfg) if extra_features_cfg else None
        # domain_features_processor = ExtraMolecularFeatures(domain_features_cfg) if domain_features_cfg else None
        # self.compute_input_output_dims(datamodule, extra_features_processor, domain_features_processor)

        # Simpler call if you've set up extra features elsewhere or don't use them:
        # Assuming datamodule.cfg has 'extra_features' and 'domain_features_edge_attr_kind' sections
        self.compute_input_output_dims(datamodule,
                                       datamodule.cfg.dataset.extra_features,
                                       datamodule.cfg.dataset.domain_features)

# Example Hydra Config (configs/dataset/aig_custom.yaml)
# name: my_aig_dataset_name_in_cfg  # Used by AIGCustomDataModule to build root path
# datadir: data                       # Base for dataset roots, e.g. project_root/data/
#                                     # Full root for InMemoryDataset: project_root/data/my_aig_dataset_name_in_cfg
#
# # Basenames of the files (output of your script, containing List[Data(x,adj)])
# # These files MUST be placed by you (or a script) into:
# # project_root/data/my_aig_dataset_name_in_cfg/raw/
# train_raw_filenames_basenames: ['train_aigs_raw_pyg_topo.pt']
# val_raw_filenames_basenames: ['val_aigs_raw_pyg_topo.pt']
# test_raw_filenames_basenames: ['test_aigs_raw_pyg_topo.pt']
#
# node_feature_dim: ${aig_config.NUM_NODE_FEATURES} # If aig_config is accessible to Hydra
# edge_feature_dim: ${aig_config.NUM_EXPLICIT_EDGE_TYPES}
#
# _target_: src.datasets.aig_custom_dataset.AIGCustomDataModule
#
# dataset_infos_target: src.datasets.aig_custom_dataset.AIGCustomDatasetInfos # To be called by main script
#
# extra_features: null # Or configure if you use them
# domain_features: null # Or configure
#
# # Other params from DiGress's default dataset configs:
# remove_h: False
# cleanup: False
# use_graphs: True
# batch_size: 64 # Example
# num_workers: 0 # Example
# pin_memory: True # Example
#
# # In your main experiment config (e.g., configs/experiment/my_aig_experiment.yaml)
# # defaults:
# #   - dataset: aig_custom
# #   - ... other defaults ...
# # general:
# #   abs_path_to_project_root: /path/to/your/DiGress_project_clone