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
from itertools import repeat


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
