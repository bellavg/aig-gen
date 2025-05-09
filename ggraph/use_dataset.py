#!/usr/bin/env python3
import os.path as osp
import warnings
from itertools import repeat
from typing import List, Tuple, Optional

import torch
import torch_geometric.io.fs as pyg_fs
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData


class AIGPreprocessedDatasetLoader(Dataset):
    def __init__(self, root: str, split: str,
                 transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None,
                 pre_filter: Optional[callable] = None):

        self.split = split
        self.dataset_specific_root = root
        super().__init__(root=self.dataset_specific_root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        current_processed_file_path = self.processed_paths[0]
        osp_exists = osp.exists(current_processed_file_path)
        pygfs_exists = pyg_fs.exists(current_processed_file_path)

        if osp_exists and pygfs_exists:
            try:
                loaded_content = torch.load(current_processed_file_path, map_location='cpu', weights_only=False)
                if not isinstance(loaded_content, tuple) or len(loaded_content) != 2:
                    raise TypeError(
                        f"Loaded data is not a tuple of 2 elements (data, slices). Got {type(loaded_content)}")
                self._data_list = None
                self.data, self.slices = loaded_content
            except Exception as e:
                print(f"  ERROR during direct torch.load or data assignment: {e}")
                raise RuntimeError(f"Failed to load pre-processed data from {current_processed_file_path}: {e}")
        else:
            raise FileNotFoundError(
                f"Processed file not found at {current_processed_file_path} during __init__ data loading. "
                f"osp.exists: {osp_exists}, pyg_fs.exists: {pygfs_exists}"
            )
        print(f"Dataset loader for split '{self.split}' initialization complete. Samples: {len(self)}")

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
        current_processed_file_path = self.processed_paths[0]
        if not pyg_fs.exists(current_processed_file_path):
            raise FileNotFoundError(
                f"Processed file not found at {current_processed_file_path} during process() check. "
                f"Please run the AIG data generation and processing script first.")

    def len(self) -> int:
        if self.slices is None:
            return 0
        for _, value in self.slices.items():  # Assumes slices is not empty if len > 0
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.size(0) - 1
        return 0  # Should ideally not be reached if dataset has items

    def get(self, idx: int) -> BaseData:
        if not hasattr(self, '_data_list') or self._data_list is None:
            if self.data is None or self.slices is None:
                raise IndexError(
                    f"Dataset not loaded properly (self.data or self.slices is None), cannot get item {idx}")
            if not isinstance(self.data, BaseData):
                raise TypeError(f"self.data is not a PyG BaseData object, cannot get item {idx}")

            num_items = self.len()
            if not (0 <= idx < num_items):  # Check if idx is valid
                raise IndexError(f"Index {idx} out of bounds for dataset with length {num_items}")

            data_obj = self.data.__class__()

            # Store the true number of nodes if available from __num_nodes__
            # This value will be used by PyG's num_nodes property if __num_nodes__ is set on data_obj
            true_num_nodes_for_this_item = None
            if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None:
                if idx < len(self.data.__num_nodes__):
                    val = self.data.__num_nodes__[idx]
                    if torch.is_tensor(val) and val.numel() == 1:
                        true_num_nodes_for_this_item = val.item()
                    elif isinstance(val, int):
                        true_num_nodes_for_this_item = val
                    if true_num_nodes_for_this_item is not None:
                        # Set the special __num_nodes__ attribute.
                        # This is the canonical way PyG stores this for collated graphs.
                        data_obj.__num_nodes__ = true_num_nodes_for_this_item
                else:
                    warnings.warn(
                        f"Index {idx} out of bounds for self.data.__num_nodes__ (len {len(self.data.__num_nodes__)}). "
                        f"Actual num_nodes for item {idx} might be inferred incorrectly."
                    )

            for key in self.data.keys():
                # If the key is 'num_nodes' and we've already set data_obj.__num_nodes__,
                # we can potentially skip direct assignment for 'num_nodes' key,
                # as the property data_obj.num_nodes will use data_obj.__num_nodes__.
                if key == 'num_nodes' and hasattr(data_obj, '__num_nodes__'):
                    # The num_nodes property will correctly use data_obj.__num_nodes__
                    # No need to slice and assign self.data['num_nodes'] if __num_nodes__ is set
                    continue

                if key not in self.slices:
                    warnings.warn(
                        f"Key '{key}' found in self.data but not in self.slices. "
                        f"Skipping attribute '{key}' for reconstructed Data object at index {idx}."
                    )
                    continue

                item = self.data[key]
                slices_for_key = self.slices[key]

                if not (isinstance(slices_for_key, torch.Tensor) and \
                        slices_for_key.ndim > 0 and \
                        idx < slices_for_key.size(0) and \
                        (idx + 1) < slices_for_key.size(0) + 1):  # Check idx+1 validity for slicing
                    warnings.warn(
                        f"Slices for key '{key}' are invalid or index {idx} is out of bounds. "
                        f"Slices type: {type(slices_for_key)}, ndim: {slices_for_key.ndim if isinstance(slices_for_key, torch.Tensor) else 'N/A'}, "
                        f"size: {slices_for_key.size(0) if isinstance(slices_for_key, torch.Tensor) else 'N/A'}. Skipping key '{key}'.")
                    continue

                if not torch.is_tensor(item):
                    warnings.warn(
                        f"Item for key '{key}' is not a tensor (type: {type(item)}). "
                        f"Skipping slicing for this attribute on data_obj at index {idx}."
                    )
                    continue

                s = list(repeat(slice(None), item.dim()))
                cat_dim = self.data.__cat_dim__(key, item)

                if cat_dim is None or not (0 <= cat_dim < item.dim()):
                    warnings.warn(
                        f"cat_dim {cat_dim} is invalid for item '{key}' with {item.dim()} dimensions. Skipping key '{key}'.")
                    continue

                try:
                    start_slice = slices_for_key[idx]
                    end_slice = slices_for_key[idx + 1]
                    s[cat_dim] = slice(start_slice, end_slice)
                    data_obj[key] = item[s]
                except Exception as e:  # Catch any slicing error
                    warnings.warn(
                        f"Error slicing item '{key}' for index {idx} with slice ({start_slice}:{end_slice}) on dim {cat_dim}. "
                        f"Error: {e}. Item shape: {item.shape if torch.is_tensor(item) else 'N/A'}. "
                        f"Skipping attribute '{key}'."
                    )
            return data_obj
        else:  # Should not happen with current __init__
            if idx < 0 or idx >= len(self._data_list):
                raise IndexError(f"Index {idx} out of bounds for self._data_list with length {len(self._data_list)}")
            return self._data_list[idx]