#!/usr/bin/env python3
import os
import torch
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.data.data import BaseData  # For type hinting
import os.path as osp
from typing import List, Tuple, Dict, Optional
import torch_geometric.io.fs as pyg_fs  # Import PyG's filesystem utilities


class AIGPreprocessedDatasetLoader(Dataset):  # Changed parent to Dataset for more control
    def __init__(self, root: str, dataset_name: str, split: str,
                 transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None,
                 pre_filter: Optional[callable] = None):
        """
        Loads a pre-processed AIG dataset more directly.
        Bypasses some of InMemoryDataset's initial checks if file exists.
        """
        self.dataset_name = dataset_name
        self.split = split

        # Construct the root directory for this specific dataset,
        # where the 'processed' folder is expected by PyG's conventions.
        # e.g., /path/to/your_data_root/your_dataset_name
        self.dataset_specific_root = osp.join(root, dataset_name)

        # Call super().__init__ for the base Dataset class.
        # The `root` argument for Dataset is the directory where the dataset should be stored/found.
        super().__init__(root=self.dataset_specific_root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        # Manually construct the path to the processed file
        # This is what InMemoryDataset would typically do with self.processed_paths[0]
        self._processed_file_path = osp.join(self.processed_dir, self.processed_file_names[0])

        print(f"\n--- [AIGLoader Custom - In __init__] ---")
        print(f"  Dataset specific root (self.root): {self.root}")  # self.root is now dataset_specific_root
        print(f"  Processed directory (self.processed_dir): {self.processed_dir}")
        print(f"  Processed file name (self.processed_file_names[0]): {self.processed_file_names[0]}")
        print(f"  Full path to processed file (self._processed_file_path): {self._processed_file_path}")

        # Check existence using both os.path and pyg_fs for thoroughness
        osp_exists = osp.exists(self._processed_file_path)
        pygfs_exists = pyg_fs.exists(self._processed_file_path)
        print(f"  osp.exists on processed file: {osp_exists}")
        print(f"  pyg_fs.exists on processed file: {pygfs_exists}")

        if osp_exists and pygfs_exists:
            print(f"  File confirmed to exist. Attempting direct torch.load...")
            try:
                # Directly load the data since we know it's pre-processed
                # InMemoryDataset typically stores data as a tuple (data_obj, slices_obj)
                loaded_content = torch.load(self._processed_file_path, map_location='cpu', weights_only=False)

                if not isinstance(loaded_content, tuple) or len(loaded_content) != 2:
                    raise TypeError(
                        f"Loaded data is not a tuple of 2 elements (data, slices). Got {type(loaded_content)}")

                self._data_list = None  # For list-based datasets, not used here
                self.data, self.slices = loaded_content
                print(f"  Successfully loaded data and slices via torch.load.")
                print(f"  Type of self.data: {type(self.data)}")
                print(f"  Type of self.slices: {type(self.slices)}")

            except Exception as e:
                print(f"  ERROR during direct torch.load or data assignment: {e}")
                # If loading fails here, it's a more fundamental issue with the file or torch.load
                raise RuntimeError(f"Failed to load pre-processed data from {self._processed_file_path}: {e}")
        else:
            # This case should ideally not be reached if the file was created correctly
            # and our previous diagnostics were accurate.
            print(
                f"  ERROR: Processed file not found at {self._processed_file_path} despite earlier checks. This is unexpected.")
            raise FileNotFoundError(
                f"Processed file not found at {self._processed_file_path}. "
                f"Ensure the file exists and paths are correct. "
                f"osp.exists: {osp_exists}, pyg_fs.exists: {pygfs_exists}"
            )

        print(f"Dataset '{self.dataset_name}' split '{self.split}' initialization complete. Samples: {len(self)}")

    @property
    def raw_file_names(self) -> List[str]:
        # This dataset loader does not use raw files.
        return []

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        # Defines the name of the pre-processed file to load.
        return [f'{self.split}_processed_data.pt']

    # We don't need download() or process() as we load directly in __init__
    # If these were called, it would mean an issue with the Dataset base class logic,
    # which is less likely for this scenario.

    def len(self) -> int:
        if isinstance(self.slices, dict):
            for _, Slices in self.slices.items():
                return Slices.size(0) - 1
        elif isinstance(self.slices, torch.Tensor):  # Should not happen for dict slices
            return self.slices.size(0) - 1
        return 0  # Fallback

    def get(self, idx: int) -> BaseData:
        if self.data is None or self.slices is None:
            raise IndexError("Dataset not loaded properly")

        # Simplified get, assuming self.data is a single large BaseData object
        # and self.slices is a dictionary of tensors.
        # This mimics how InMemoryDataset.get works.
        if not isinstance(self.data, BaseData):
            raise TypeError("self.data is not a PyG BaseData object")

        data = self.data.__class__()  # Create a new empty Data object of the same type
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()

            if isinstance(item, torch.Tensor):
                s = list(item.shape)
                s[self.data.__cat_dim__(key, item)] = end - start
                data[key] = item[start:end]
            elif isinstance(item, list) and torch.is_tensor(item[0]):  # Handle list of tensors
                data[key] = item[start:end]
            else:  # For other types, just copy (though not typical for graph attributes)
                data[key] = item[start:end] if isinstance(item, (list, tuple)) else item
        return data

