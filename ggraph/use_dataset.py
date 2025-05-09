#!/usr/bin/env python3
import os.path as osp
import warnings
from itertools import repeat
from typing import List, Tuple, Optional  # Any removed as it was not used

import torch
import torch_geometric.io.fs as pyg_fs  # Import PyG's filesystem utilities
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData  # For type hinting


class AIGPreprocessedDatasetLoader(Dataset):
    def __init__(self, root: str, split: str,
                 transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None,
                 pre_filter: Optional[callable] = None):

        self.split = split
        # Assuming 'root' is the path to the specific dataset directory,
        # e.g., './ggraph/data/aigs_pyg/my_dataset_name'
        # The Dataset superclass will use this to create self.root, self.processed_dir, etc.
        self.dataset_specific_root = root

        super().__init__(root=self.dataset_specific_root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        # self.processed_paths[0] is now available and gives the full path
        # to the processed file, e.g., <root>/processed/<split>_processed_data.pt

        current_processed_file_path = self.processed_paths[0]  # Use the property

        osp_exists = osp.exists(current_processed_file_path)
        pygfs_exists = pyg_fs.exists(current_processed_file_path)

        if osp_exists and pygfs_exists:
            try:
                loaded_content = torch.load(current_processed_file_path, map_location='cpu', weights_only=False)

                if not isinstance(loaded_content, tuple) or len(loaded_content) != 2:
                    raise TypeError(
                        f"Loaded data is not a tuple of 2 elements (data, slices). Got {type(loaded_content)}")

                self._data_list = None  # For PyG Dataset compatibility
                self.data, self.slices = loaded_content

            except Exception as e:
                print(f"  ERROR during direct torch.load or data assignment: {e}")
                raise RuntimeError(f"Failed to load pre-processed data from {current_processed_file_path}: {e}")
        else:
            # This path should ideally not be hit if process() is correctly guarded,
            # or if the files are guaranteed to exist by the AIGProcessedDataset script.
            # If process() was called by super().__init__(), it would have already raised an error.
            raise FileNotFoundError(
                f"Processed file not found at {current_processed_file_path} during __init__ data loading. "
                f"This is unexpected if process() did not run or did not error. "
                f"osp.exists: {osp_exists}, pyg_fs.exists: {pygfs_exists}"
            )

        # dataset_name is not an attribute of this class as per its __init__ signature.
        # If you need it for the print statement, it should be passed to __init__ and set.
        # For now, removing it from the print statement to avoid another AttributeError.
        print(f"Dataset loader for split '{self.split}' initialization complete. Samples: {len(self)}")

    @property
    def raw_file_names(self) -> List[str]:
        # This loader does not handle raw files directly.
        return []

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        # Defines the name of the processed file relative to self.processed_dir
        return [f'{self.split}_processed_data.pt']

    def download(self):
        # This loader assumes data is already generated and processed.
        pass

    def process(self):
        # This method is called by the Dataset superclass if it determines
        # that processed files (based on self.processed_paths) do not exist.
        # For this loader, this means the prerequisite processed file is missing.
        print(f"AIGPreprocessedDatasetLoader.process() called for split '{self.split}'. "
              f"This loader expects data to be already processed by AIGProcessedDataset.")

        # Use self.processed_paths[0] which is guaranteed to be set up by super().__init__()
        # before process() is called.
        current_processed_file_path = self.processed_paths[0]

        if not pyg_fs.exists(current_processed_file_path):
            raise FileNotFoundError(
                f"Processed file not found at {current_processed_file_path} during process() check. "
                f"Please run the AIG data generation and processing script (e.g., the one that creates AIGProcessedDataset) first.")
        # If the file *does* exist but process() was still called (e.g. force_reload=True),
        # this loader doesn't re-process, it relies on existing files.
        # The __init__ method will handle loading it.

    def len(self) -> int:
        if self.slices is None:
            return 0
        # Iterate over the keys in self.slices and get the size of the first tensor found.
        # This is a common way InMemoryDataset determines length from collated data.
        for _, value in self.slices.items():
            if isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    return value.size(0) - 1
                else:
                    return 0
        return 0

    def get(self, idx: int) -> BaseData:
        # This method retrieves a single Data object from the collated self.data
        if not hasattr(self, '_data_list') or self._data_list is None:  # Standard for InMemoryDataset-like loading
            if self.data is None or self.slices is None:
                raise IndexError(
                    f"Dataset not loaded properly (self.data or self.slices is None), cannot get item {idx}")
            if not isinstance(self.data, BaseData):  # self.data should be a single large Data object
                raise TypeError(f"self.data is not a PyG BaseData object, cannot get item {idx}")

            num_items = self.len()
            if idx < 0 or idx >= num_items:
                raise IndexError(f"Index {idx} out of bounds for dataset with length {num_items}")

            data_obj = self.data.__class__()  # Create a new empty Data object of the same type

            if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None:
                if idx < len(self.data.__num_nodes__):
                    data_obj.num_nodes = self.data.__num_nodes__[idx]
                else:
                    warnings.warn(
                        f"Index {idx} out of bounds for self.data.__num_nodes__ (len {len(self.data.__num_nodes__)}). num_nodes might not be set correctly.")

            for key in self.data.keys:
                item, slices_for_key = self.data[key], self.slices[key]

                if not isinstance(slices_for_key,
                                  torch.Tensor) or slices_for_key.ndim == 0 or idx + 1 >= slices_for_key.size(0):
                    warnings.warn(
                        f"Slices for key '{key}' are invalid or index {idx} (slice end {idx + 1}) is out of bounds for slices tensor of size {slices_for_key.size(0) if isinstance(slices_for_key, torch.Tensor) else 'N/A'}. Skipping key.")
                    data_obj[key] = None
                    continue

                s = list(repeat(slice(None), item.dim()))
                cat_dim = self.data.__cat_dim__(key, item)

                if cat_dim >= item.dim():
                    warnings.warn(
                        f"cat_dim {cat_dim} is out of bounds for item '{key}' with {item.dim()} dimensions. Skipping key.")
                    data_obj[key] = None
                    continue

                try:
                    s[cat_dim] = slice(slices_for_key[idx], slices_for_key[idx + 1])
                    data_obj[key] = item[s]
                except IndexError as e:
                    warnings.warn(
                        f"IndexError while slicing item '{key}' with slice {s}: {e}. Item shape: {item.shape}, slices_for_key[idx]: {slices_for_key[idx]}, slices_for_key[idx+1]: {slices_for_key[idx + 1]}")
                    data_obj[key] = None  # Set to None or handle error appropriately
            return data_obj
        else:
            # This branch would be for when self._data_list is a list of Data objects.
            if idx < 0 or idx >= len(self._data_list):
                raise IndexError(f"Index {idx} out of bounds for self._data_list with length {len(self._data_list)}")
            return self._data_list[idx]
