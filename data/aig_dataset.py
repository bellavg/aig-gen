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

    def len(self) -> int:
        # Determine length from the slices dictionary
        if isinstance(self.slices, dict):
            # Find a key in slices that corresponds to a tensor attribute in data
            # that indicates the number of graphs (e.g., 'x', 'adj', 'num_nodes')
            # Often, any key in slices will work if data is consistent.
            slice_key = next(iter(self.slices))  # Get the first key
            if slice_key is not None:
                return self.slices[slice_key].size(0) - 1
        # Fallback or alternative if slices structure is different
        if hasattr(self.data, 'num_nodes') and isinstance(self.data.num_nodes, torch.Tensor):
            # If num_nodes is stored per graph, its length might represent the number of graphs
            # This might not be correct if num_nodes is collated differently.
            # The slices method is generally more reliable for InMemoryDataset style.
            pass  # Avoid using len(self.data.num_nodes) unless sure about collation
        return 0  # Default if length cannot be determined

    def get(self, idx: int) -> BaseData:
        # Retrieve a single graph Data object from the collated self.data
        if self.data is None or self.slices is None:
            raise IndexError(f"Dataset not loaded properly, cannot get item {idx}")

        if not isinstance(self.data, BaseData):
            raise TypeError(f"self.data is not a PyG BaseData object, cannot get item {idx}")

        data = self.data.__class__()  # Create a new empty Data object of the same type
        if hasattr(self.data, '__num_nodes__') and self.data.__num_nodes__ is not None:
            # Check if __num_nodes__ exists and is not None before accessing
            if isinstance(self.data.__num_nodes__, (list, torch.Tensor)) and idx < len(self.data.__num_nodes__):
                data.num_nodes = self.data.__num_nodes__[idx]
            else:
                # Handle cases where num_nodes might be stored differently or index is out of bounds
                pass  # Or set a default, or raise an error depending on expected structure

        # *** CORRECTED LINE: Added () to keys method call ***
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]

            # Check if idx is valid for the slices tensor
            if not isinstance(slices, torch.Tensor) or idx + 1 >= slices.size(0):
                raise IndexError(
                    f"Index {idx} out of bounds for slices of key '{key}' (size: {slices.size(0) if isinstance(slices, torch.Tensor) else 'N/A'})")

            start, end = slices[idx].item(), slices[idx + 1].item()

            # Handle slice assignment based on item type
            if torch.is_tensor(item):
                # Determine the dimension to slice along
                cat_dim = self.data.__cat_dim__(key, item)  # Default is usually 0
                if cat_dim is None:
                    # If __cat_dim__ returns None, it might be a graph-level attribute.
                    # Check if slices indicate it should be treated as such (start=idx, end=idx+1)
                    # This logic might need adjustment based on how graph-level attributes are stored/sliced.
                    if end - start == 1:  # Likely graph-level attribute
                        data[key] = item[start]  # Get the single value for this graph
                    else:  # Unclear how to handle this slice
                        # Default to original slicing logic, might error if cat_dim is None
                        data[key] = item[start:end]
                else:
                    # Standard slicing along the concatenation dimension
                    data[key] = item.narrow(cat_dim, start, end - start)
            elif isinstance(item, list) and item and torch.is_tensor(item[0]):  # Handle list of tensors
                data[key] = item[start:end]
            elif isinstance(item, list) or isinstance(item, tuple):  # Handle list/tuple of non-tensors
                data[key] = item[start:end]  # Slice the list/tuple
            else:
                # For scalar or other non-tensor/non-list types (uncommon for collated data)
                # This likely represents a graph-level attribute repeated across batches.
                # If slices indicate a single item (end-start == 1), take the item.
                if end - start == 1:
                    # Need to determine how to get the single item corresponding to idx.
                    # If 'item' itself is the value repeated, this might work:
                    data[key] = item
                    # If 'item' is a list/tensor containing these values:
                    # data[key] = item[start] # Or item[idx] if not collated
                else:
                    # Fallback if unsure how item is structured
                    data[key] = item  # Assign the whole item, might be incorrect

        return data

