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
        self._batch_has_valid_dunder_num_nodes = False  # Initialize flag

        super().__init__(root=self.dataset_specific_root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        current_processed_file_path = self.processed_paths[0]

        if pyg_fs.exists(current_processed_file_path):
            try:
                loaded_content = torch.load(current_processed_file_path, map_location='cpu', weights_only=False)
                if not isinstance(loaded_content, tuple) or len(loaded_content) != 2:
                    raise TypeError(
                        f"Loaded data from {current_processed_file_path} is not a tuple of 2 elements (data, slices). Got {type(loaded_content)}")
                self._data_list = None
                self.data, self.slices = loaded_content

                # One-time check for num_nodes inconsistency and set flag
                has_num_nodes_key_in_data = 'num_nodes' in self.data
                has_num_nodes_key_in_slices = 'num_nodes' in self.slices
                has_batch_dunder_num_nodes_attr = hasattr(self.data, '__num_nodes__') and \
                                                  self.data.__num_nodes__ is not None and \
                                                  torch.is_tensor(self.data.__num_nodes__) and \
                                                  self.data.__num_nodes__.ndim > 0  # Ensure it's a non-empty tensor

                # Calculate initial length based on slices for comparison
                initial_len_from_slices = 0
                if self.slices is not None:
                    for _, value in self.slices.items():
                        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.numel() > 0:
                            cat_dim_check_key = next(iter(self.slices))  # get a key to check cat_dim
                            if self.data[cat_dim_check_key].size(
                                    self.data.__cat_dim__(cat_dim_check_key, self.data[cat_dim_check_key])) > 0:
                                initial_len_from_slices = value.size(0) - 1 if value.size(0) > 0 else 0
                                break

                num_graphs_from_dunder_nodes = 0
                if has_batch_dunder_num_nodes_attr:
                    num_graphs_from_dunder_nodes = self.data.__num_nodes__.size(0)

                if has_num_nodes_key_in_data and not has_num_nodes_key_in_slices:
                    if has_batch_dunder_num_nodes_attr:
                        self._batch_has_valid_dunder_num_nodes = (
                                    num_graphs_from_dunder_nodes == initial_len_from_slices and num_graphs_from_dunder_nodes > 0)
                        if self._batch_has_valid_dunder_num_nodes:
                            warnings.warn(
                                f"Dataset Inconsistency for split '{self.split}': "
                                f"'num_nodes' key exists in self.data but is missing from 'self.slices'. "
                                f"However, 'self.data.__num_nodes__' is present and appears consistent with dataset length ({num_graphs_from_dunder_nodes} items). "
                                f"The loader will use 'self.data.__num_nodes__'. This is usually acceptable.",
                                UserWarning
                            )
                        else:
                            warnings.warn(
                                f"Dataset Inconsistency for split '{self.split}': "
                                f"'num_nodes' key exists in self.data but is missing from 'self.slices'. "
                                f"'self.data.__num_nodes__' is present but its length ({num_graphs_from_dunder_nodes}) "
                                f"might be inconsistent with length from slices ({initial_len_from_slices}) or is zero. "
                                f"Review dataset integrity.",
                                UserWarning
                            )
                    else:  # No __num_nodes__ fallback
                        warnings.warn(
                            f"CRITICAL Dataset Inconsistency for split '{self.split}': "
                            f"'num_nodes' key exists in self.data but is missing from 'self.slices', AND "
                            f"the batch-level 'self.data.__num_nodes__' attribute is also missing, None, or not a valid tensor. "
                            f"Number of nodes for individual graphs may be inferred incorrectly by PyG.",
                            UserWarning
                        )
                elif not has_num_nodes_key_in_data and not has_batch_dunder_num_nodes_attr:
                    warnings.warn(
                        f"Note for split '{self.split}': Neither 'num_nodes' key in self.data nor "
                        f"'self.data.__num_nodes__' attribute found. PyG will infer num_nodes from other attributes (e.g., x, edge_index).",
                        UserWarning
                    )

            except Exception as e:
                print(f"  ERROR during direct torch.load or data assignment from {current_processed_file_path}: {e}")
                raise RuntimeError(f"Failed to load pre-processed data: {e}")
        else:
            raise FileNotFoundError(
                f"Processed file not found at {current_processed_file_path} during __init__ data loading. "
                f"Please ensure the dataset is correctly processed by 'make_dataset.py'."
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
              f"This loader expects data to be already processed by AIGProcessedDataset script.")
        current_processed_file_path = self.processed_paths[0]
        if not pyg_fs.exists(current_processed_file_path):
            raise FileNotFoundError(
                f"Processed file not found at {current_processed_file_path} during process() check. "
                f"Please run the AIG data generation and processing script (e.g., make_dataset.py) first.")

    def len(self) -> int:
        if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None and \
                torch.is_tensor(self.data.__num_nodes__):
            return self.data.__num_nodes__.size(0)
        elif hasattr(self.data, '__num_nodes__') and isinstance(self.data.__num_nodes__, (list, tuple)):
            return len(self.data.__num_nodes__)

        if self.slices is not None:
            for key_for_len, slice_tensor_for_len in self.slices.items():
                if isinstance(slice_tensor_for_len,
                              torch.Tensor) and slice_tensor_for_len.ndim > 0 and slice_tensor_for_len.numel() > 0:
                    # Ensure the key exists in data and is a tensor before checking its cat_dim size
                    if key_for_len in self.data and torch.is_tensor(self.data[key_for_len]):
                        cat_dim = self.data.__cat_dim__(key_for_len, self.data[key_for_len])
                        if cat_dim is not None and self.data[key_for_len].size(cat_dim) > 0:
                            return slice_tensor_for_len.size(0) - 1 if slice_tensor_for_len.size(0) > 0 else 0
            return 0
        return 0

    def get(self, idx: int) -> BaseData:
        if not hasattr(self, '_data_list') or self._data_list is None:
            if self.data is None or self.slices is None:
                raise IndexError(
                    f"Dataset not loaded properly (self.data or self.slices is None), cannot get item {idx}")
            if not isinstance(self.data, BaseData):
                raise TypeError(f"self.data is not a PyG BaseData object, cannot get item {idx}")

            num_items = self.len()
            if not (0 <= idx < num_items):
                raise IndexError(f"Index {idx} out of bounds for dataset with length {num_items}")

            data_obj = self.data.__class__()

            data_obj_dunder_num_nodes_was_set = False
            if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None:
                if idx < len(self.data.__num_nodes__):
                    val = self.data.__num_nodes__[idx]
                    true_num_nodes_for_this_item = None
                    if torch.is_tensor(val) and val.numel() == 1:
                        true_num_nodes_for_this_item = val.item()
                    elif isinstance(val, int):
                        true_num_nodes_for_this_item = val

                    if true_num_nodes_for_this_item is not None:
                        data_obj.__num_nodes__ = true_num_nodes_for_this_item
                        data_obj_dunder_num_nodes_was_set = True
                    else:
                        warnings.warn(
                            f"Value for num_nodes from self.data.__num_nodes__[{idx}] is unexpected (type: {type(val)}). "
                            f"data_obj.__num_nodes__ not set from this source for item {idx}.", UserWarning)
                else:
                    warnings.warn(
                        f"Index {idx} out of bounds for self.data.__num_nodes__ (len {len(self.data.__num_nodes__)}). "
                        f"data_obj.__num_nodes__ not set for item {idx}. PyG will infer num_nodes.", UserWarning
                    )

            for key in self.data.keys():
                if key == 'num_nodes' and data_obj_dunder_num_nodes_was_set:
                    continue  # Already handled by setting data_obj.__num_nodes__

                if key not in self.slices:
                    # Conditional warning for 'num_nodes' if it's missing from slices
                    if key == 'num_nodes':  # Implies data_obj_dunder_num_nodes_was_set is False
                        if self._batch_has_valid_dunder_num_nodes:  # Batch was supposed to have good __num_nodes__
                            warnings.warn(
                                f"Note: 'num_nodes' is missing from slices for index {idx}, and data_obj.__num_nodes__ "
                                f"could not be set for this specific item (e.g. bad value in self.data.__num_nodes__[{idx}]). "
                                f"PyG will infer num_nodes for this item, which might be from padded 'x'.", UserWarning)
                        else:  # Batch __num_nodes__ was already problematic or missing
                            warnings.warn(
                                f"Warning: Key 'num_nodes' is missing from slices for index {idx}, and batch-level "
                                f"'self.data.__num_nodes__' was not valid/available. "
                                f"PyG will infer num_nodes for this item.", UserWarning)
                    else:  # For keys other than 'num_nodes'
                        warnings.warn(
                            f"Key '{key}' found in self.data but not in self.slices. "
                            f"Skipping attribute '{key}' for reconstructed Data object at index {idx}.",
                            UserWarning
                        )
                    continue

                item = self.data[key]
                slices_for_key = self.slices[key]

                if not (isinstance(slices_for_key, torch.Tensor) and \
                        slices_for_key.ndim > 0 and \
                        idx < slices_for_key.size(0) and \
                        (idx + 1) < (slices_for_key.size(0) + 1)):
                    warnings.warn(
                        f"Slices for key '{key}' are invalid or index {idx} is out of bounds. "
                        f"Skipping key '{key}'.", UserWarning)
                    continue

                if not torch.is_tensor(item):
                    warnings.warn(
                        f"Item for key '{key}' is not a tensor (type: {type(item)}). "
                        f"Skipping slicing for this attribute on data_obj at index {idx}.", UserWarning
                    )
                    continue

                s = list(repeat(slice(None), item.dim()))
                cat_dim = self.data.__cat_dim__(key, item)

                if cat_dim is None or not (0 <= cat_dim < item.dim()):
                    warnings.warn(
                        f"cat_dim {cat_dim} is invalid for item '{key}' with {item.dim()} dimensions. "
                        f"Skipping key '{key}'.", UserWarning)
                    continue

                try:
                    start_slice = slices_for_key[idx]
                    end_slice = slices_for_key[idx + 1]
                    s[cat_dim] = slice(start_slice, end_slice)
                    data_obj[key] = item[s]
                except Exception as e:
                    warnings.warn(
                        f"Error slicing item '{key}' for index {idx}. Error: {e}. Skipping attribute '{key}'.",
                        UserWarning
                    )
            return data_obj
        else:
            if idx < 0 or idx >= len(self._data_list):
                raise IndexError(f"Index {idx} out of bounds for self._data_list with length {len(self._data_list)}")
            return self._data_list[idx]

