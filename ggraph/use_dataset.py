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

        if pyg_fs.exists(current_processed_file_path):
            try:
                loaded_content = torch.load(current_processed_file_path, map_location='cpu', weights_only=False)
                if not isinstance(loaded_content, tuple) or len(loaded_content) != 2:
                    raise TypeError(
                        f"Loaded data from {current_processed_file_path} is not a tuple of 2 elements (data, slices). Got {type(loaded_content)}")
                self._data_list = None  # Standard for InMemoryDataset-like behavior
                self.data, self.slices = loaded_content

                # One-time check for num_nodes inconsistency
                has_num_nodes_key_in_data = 'num_nodes' in self.data  # Checks self.data.keys()
                has_num_nodes_key_in_slices = 'num_nodes' in self.slices
                has_batch_num_nodes_attr = hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None

                if has_num_nodes_key_in_data and not has_num_nodes_key_in_slices:
                    if has_batch_num_nodes_attr:
                        warnings.warn(
                            f"Dataset Inconsistency for split '{self.split}': "
                            f"'num_nodes' key exists in the collated data object (self.data) but is missing from 'self.slices'. "
                            f"The loader will attempt to use the batch-level 'self.data.__num_nodes__' to determine "
                            f"node counts for individual graphs. This is usually acceptable.",
                            UserWarning
                        )
                        if len(self.data.__num_nodes__) != self.len():  # Check consistency of __num_nodes__ length with dataset length
                            warnings.warn(
                                f"Potential Issue: Length of self.data.__num_nodes__ ({len(self.data.__num_nodes__)}) "
                                f"does not match the calculated dataset length ({self.len()}). "
                                f"This might lead to errors if __num_nodes__ is not correctly representing all items.",
                                UserWarning
                            )
                    else:
                        warnings.warn(
                            f"CRITICAL Dataset Inconsistency for split '{self.split}': "
                            f"'num_nodes' key exists in self.data but is missing from 'self.slices', AND "
                            f"the batch-level 'self.data.__num_nodes__' attribute is also missing or None. "
                            f"The number of nodes for individual graphs may be inferred incorrectly by PyG, "
                            f"potentially leading to errors in model processing.",
                            UserWarning
                        )
                elif not has_num_nodes_key_in_data and not has_batch_num_nodes_attr:
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
        # This loader does not handle raw files directly; assumes data is pre-processed.
        return []

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        # Defines the name of the processed file relative to self.processed_dir
        return [f'{self.split}_processed_data.pt']

    def download(self):
        # This loader assumes data is already generated and processed by AIGProcessedDataset.
        pass

    def process(self):
        # This method is called by the Dataset superclass if it determines
        # that processed files (based on self.processed_paths) do not exist.
        print(f"AIGPreprocessedDatasetLoader.process() called for split '{self.split}'. "
              f"This loader expects data to be already processed by AIGProcessedDataset script.")
        current_processed_file_path = self.processed_paths[0]
        if not pyg_fs.exists(current_processed_file_path):
            raise FileNotFoundError(
                f"Processed file not found at {current_processed_file_path} during process() check. "
                f"Please run the AIG data generation and processing script (e.g., make_dataset.py) first.")
        # If the file *does* exist but process() was still called (e.g. force_reload=True in superclass),
        # this loader doesn't re-process; it relies on the existing file.
        # The __init__ method will handle loading it.

    def len(self) -> int:
        # Prioritize __num_nodes__ for counting graphs in the batch if available
        if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None:
            if torch.is_tensor(self.data.__num_nodes__):
                return self.data.__num_nodes__.size(0)  # Length of the tensor
            elif isinstance(self.data.__num_nodes__, (list, tuple)):
                return len(self.data.__num_nodes__)

        # Fallback to using slices if __num_nodes__ is not available or not a sequence
        if self.slices is not None:
            for key, value in self.slices.items():  # Iterate to find a valid slice tensor
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.numel() > 0:
                    # For attributes concatenated along a new batch dimension (dim 0),
                    # slices usually has N+1 entries where N is number of graphs.
                    # For attributes that are per-graph (like num_nodes if it *were* sliced),
                    # it would be a direct index.
                    # Assuming standard PyG InMemoryDataset slicing for node/edge features:
                    if self.data[key].size(self.data.__cat_dim__(key, self.data[
                        key])) > 0:  # Check if the concatenated dim is not empty
                        return value.size(0) - 1 if value.size(0) > 0 else 0
            return 0  # If slices is empty or contains no valid slice tensors
        return 0

    def get(self, idx: int) -> BaseData:
        if not hasattr(self,
                       '_data_list') or self._data_list is None:  # Standard check for InMemoryDataset-like loading
            if self.data is None or self.slices is None:
                raise IndexError(
                    f"Dataset not loaded properly (self.data or self.slices is None), cannot get item {idx}")
            if not isinstance(self.data, BaseData):
                raise TypeError(f"self.data is not a PyG BaseData object, cannot get item {idx}")

            num_items = self.len()
            if not (0 <= idx < num_items):  # Check if idx is valid using the potentially revised len()
                raise IndexError(f"Index {idx} out of bounds for dataset with length {num_items}")

            data_obj = self.data.__class__()  # Create a new empty Data object of the same type as the batch

            # Attempt to set the specific __num_nodes__ for the individual data_obj
            # This is the most reliable way for PyG to know the true node count.
            true_num_nodes_for_this_item = None
            if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None:
                if idx < len(self.data.__num_nodes__):  # Check bounds for __num_nodes__ list/tensor
                    val = self.data.__num_nodes__[idx]
                    if torch.is_tensor(val) and val.numel() == 1:  # Scalar tensor
                        true_num_nodes_for_this_item = val.item()
                    elif isinstance(val, int):  # Already an int
                        true_num_nodes_for_this_item = val

                    if true_num_nodes_for_this_item is not None:
                        data_obj.__num_nodes__ = true_num_nodes_for_this_item
                    else:  # Value from __num_nodes__[idx] was not a scalar tensor or int
                        warnings.warn(
                            f"Value for num_nodes from self.data.__num_nodes__[{idx}] is unexpected (type: {type(val)}). "
                            f"data_obj.__num_nodes__ not set from this source for item {idx}.", UserWarning)
                else:  # idx out of bounds for __num_nodes__
                    warnings.warn(
                        f"Index {idx} out of bounds for self.data.__num_nodes__ (len {len(self.data.__num_nodes__)}). "
                        f"data_obj.__num_nodes__ not set for item {idx}. PyG will infer num_nodes.", UserWarning
                    )

            # Iterate through all keys in the batched data to reconstruct the individual Data object
            for key in self.data.keys():
                # If 'num_nodes' key is encountered AND data_obj.__num_nodes__ was successfully set,
                # we can skip trying to slice 'num_nodes'. The data_obj.num_nodes property will use __num_nodes__.
                if key == 'num_nodes' and hasattr(data_obj, '__num_nodes__') and data_obj.__num_nodes__ is not None:
                    continue

                # Check if slices exist for this key
                if key not in self.slices:
                    # This warning is now more targeted due to the __init__ check and the 'num_nodes' specific skip above.
                    # It will primarily show for keys other than 'num_nodes' if their slices are missing.
                    # Or for 'num_nodes' if data_obj.__num_nodes__ could NOT be set.
                    warnings.warn(
                        f"Key '{key}' found in self.data but not in self.slices. "
                        f"Skipping attribute '{key}' for reconstructed Data object at index {idx}.",
                        UserWarning
                    )
                    continue

                item = self.data[key]
                slices_for_key = self.slices[key]

                # Validate the slices_for_key tensor and idx bounds for slicing
                if not (isinstance(slices_for_key, torch.Tensor) and \
                        slices_for_key.ndim > 0 and \
                        idx < slices_for_key.size(0) and \
                        (idx + 1) < (slices_for_key.size(0) + 1)):  # Ensure slices_for_key[idx+1] is a valid access
                    warnings.warn(
                        f"Slices for key '{key}' are invalid or index {idx} is out of bounds. "
                        f"Slices type: {type(slices_for_key)}, "
                        f"ndim: {slices_for_key.ndim if isinstance(slices_for_key, torch.Tensor) else 'N/A'}, "
                        f"size: {slices_for_key.size(0) if isinstance(slices_for_key, torch.Tensor) else 'N/A'}. "
                        f"Skipping key '{key}'.", UserWarning)
                    continue

                # Ensure the item to be sliced is a tensor
                if not torch.is_tensor(item):
                    warnings.warn(
                        f"Item for key '{key}' is not a tensor (type: {type(item)}). "
                        f"Skipping slicing for this attribute on data_obj at index {idx}.", UserWarning
                    )
                    continue

                s = list(repeat(slice(None), item.dim()))  # Prepare for multi-dimensional slicing
                cat_dim = self.data.__cat_dim__(key, item)  # Get the dimension along which this item was concatenated

                # Validate cat_dim
                if cat_dim is None or not (0 <= cat_dim < item.dim()):
                    warnings.warn(
                        f"cat_dim {cat_dim} is invalid for item '{key}' which has {item.dim()} dimensions. "
                        f"Skipping key '{key}'.", UserWarning)
                    continue

                # Perform the slicing
                try:
                    start_slice = slices_for_key[idx]
                    end_slice = slices_for_key[idx + 1]
                    s[cat_dim] = slice(start_slice, end_slice)
                    data_obj[key] = item[s]
                except Exception as e:  # Catch any slicing error
                    warnings.warn(
                        f"Error slicing item '{key}' for index {idx} with slice ({start_slice}:{end_slice}) on dim {cat_dim}. "
                        f"Error: {e}. Item shape: {item.shape if torch.is_tensor(item) else 'N/A'}. "
                        f"Skipping attribute '{key}'.", UserWarning
                    )
            return data_obj
        else:
            # This branch is for when self._data_list is a list of Data objects (not typical after __init__ logic)
            if idx < 0 or idx >= len(self._data_list):  # Basic bounds check
                raise IndexError(f"Index {idx} out of bounds for self._data_list with length {len(self._data_list)}")
            return self._data_list[idx]

