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

        self.dataset_specific_root = osp.join(root, dataset_name)

        super().__init__(root=self.dataset_specific_root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

        self._processed_file_path = osp.join(self.processed_dir, self.processed_file_names[0])

        print(f"\n--- [AIGLoader Custom - In __init__] ---")
        print(f"  Dataset specific root (self.root): {self.root}")
        print(f"  Processed directory (self.processed_dir): {self.processed_dir}")
        print(f"  Processed file name (self.processed_file_names[0]): {self.processed_file_names[0]}")
        print(f"  Full path to processed file (self._processed_file_path): {self._processed_file_path}")

        osp_exists = osp.exists(self._processed_file_path)
        pygfs_exists = pyg_fs.exists(self._processed_file_path)
        print(f"  osp.exists on processed file: {osp_exists}")
        print(f"  pyg_fs.exists on processed file: {pygfs_exists}")

        if osp_exists and pygfs_exists:
            print(f"  File confirmed to exist. Attempting direct torch.load...")
            try:
                loaded_content = torch.load(self._processed_file_path, map_location='cpu', weights_only=False)

                if not isinstance(loaded_content, tuple) or len(loaded_content) != 2:
                    raise TypeError(
                        f"Loaded data is not a tuple of 2 elements (data, slices). Got {type(loaded_content)}")

                self._data_list = None
                self.data, self.slices = loaded_content
                print(f"  Successfully loaded data and slices via torch.load.")
                print(f"  Type of self.data: {type(self.data)}")
                print(f"  Type of self.slices: {type(self.slices)}")

            except Exception as e:
                print(f"  ERROR during direct torch.load or data assignment: {e}")
                raise RuntimeError(f"Failed to load pre-processed data from {self._processed_file_path}: {e}")
        else:
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
        return []

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return [f'{self.split}_processed_data.pt']

    def len(self) -> int:
        if isinstance(self.slices, dict):
            slice_key = next(iter(self.slices), None)  # Get first key or None
            if slice_key is not None:
                # Ensure the slice tensor is valid before accessing size
                if isinstance(self.slices[slice_key], torch.Tensor):
                    return self.slices[slice_key].size(0) - 1
                else:
                    print(f"Warning: Slice for key '{slice_key}' is not a Tensor. Cannot determine length accurately.")
                    return 0
            else:
                print("Warning: self.slices is an empty dictionary. Cannot determine length.")
                return 0  # Slices dictionary is empty
        return 0  # Default if length cannot be determined

    def get(self, idx: int) -> BaseData:
        if self.data is None or self.slices is None:
            raise IndexError(f"Dataset not loaded properly, cannot get item {idx}")

        if not isinstance(self.data, BaseData):
            raise TypeError(f"self.data is not a PyG BaseData object, cannot get item {idx}")

        # Check if index is valid based on dataset length
        num_items = self.len()
        if idx < 0 or idx >= num_items:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {num_items}")

        data = self.data.__class__()  # Create a new empty Data object of the same type

        # Try to set num_nodes specifically IF it exists in self.data but not necessarily slices
        # This assumes self.data.num_nodes might store per-graph node counts directly if not collated.
        # This part might need adjustment based on how num_nodes was actually saved.
        if hasattr(self.data, 'num_nodes') and 'num_nodes' not in self.slices:
            if isinstance(self.data.num_nodes,
                          torch.Tensor) and self.data.num_nodes.dim() > 0 and idx < self.data.num_nodes.size(0):
                data.num_nodes = self.data.num_nodes[idx].item()  # Get scalar value
            # Add other potential ways num_nodes might be stored if not a simple tensor list
            # else:
            #     print(f"Warning: Cannot determine num_nodes for idx {idx} from self.data.num_nodes")

        # Iterate through keys found in the main data object
        for key in self.data.keys():
            # *** CORRECTED LOGIC: Check if key exists in slices before accessing ***
            if key not in self.slices:
                # If the key isn't in slices, it wasn't collated.
                # It might be a graph-level attribute or something else.
                # We might want to copy it directly if it applies to the whole graph.
                # Example: If 'graph_id' was stored but not sliced.
                # For 'num_nodes', we handled it specifically above, so we can skip here.
                if key != 'num_nodes':  # Avoid handling num_nodes twice
                    # Decide how to handle non-sliced attributes. Copying might be appropriate for some.
                    # data[key] = self.data[key] # Uncomment cautiously if needed
                    pass  # Skip keys not in slices for now
                continue  # Move to the next key

            # --- Original slicing logic for keys that ARE in self.slices ---
            item, slices_tensor = self.data[key], self.slices[key]

            # Check if idx is valid for this specific slice tensor
            if not isinstance(slices_tensor, torch.Tensor) or idx + 1 >= slices_tensor.size(0):
                # This should ideally not happen if idx is valid for len(), but added as safeguard
                print(
                    f"Warning: Index {idx} seems out of bounds for slices of key '{key}' (size: {slices_tensor.size(0) if isinstance(slices_tensor, torch.Tensor) else 'N/A'}). Skipping key.")
                continue
                # raise IndexError(f"Index {idx} out of bounds for slices of key '{key}' (size: {slices_tensor.size(0) if isinstance(slices_tensor, torch.Tensor) else 'N/A'})")

            start, end = slices_tensor[idx].item(), slices_tensor[idx + 1].item()

            # Handle slice assignment based on item type
            if torch.is_tensor(item):
                cat_dim = self.data.__cat_dim__(key, item)
                if cat_dim is None:
                    # Graph-level tensor attribute (e.g., a single tensor for the whole graph)
                    if end - start == 1:
                        data[key] = item[start]
                    else:
                        # This case is ambiguous - multiple entries for a non-concatenated tensor?
                        print(
                            f"Warning: Ambiguous slicing for key '{key}' (cat_dim=None, slice size={end - start}). Assigning slice.")
                        data[key] = item[start:end]  # Fallback slicing
                else:
                    # Standard slicing along the concatenation dimension
                    data[key] = item.narrow(cat_dim, start, end - start)
            # Handle lists (less common for collated data, but possible)
            elif isinstance(item, list):
                data[key] = item[start:end]
            # Handle tuples (even less common)
            elif isinstance(item, tuple):
                data[key] = item[start:end]
            else:
                # Scalar or other types - likely graph-level attributes
                if end - start == 1:
                    # If item is a list/tensor containing these values:
                    if isinstance(item, (list, tuple, torch.Tensor)) and start < len(item):
                        data[key] = item[start]
                    else:  # Assume 'item' itself is the value (repeated across graphs)
                        data[key] = item
                else:
                    # Fallback if unsure how item is structured
                    print(
                        f"Warning: Ambiguous slicing for non-tensor/non-list key '{key}' (slice size={end - start}). Assigning whole item.")
                    data[key] = item

                    # After loop, ensure data.num_nodes is set if it wasn't handled above
        # It might be calculable from other attributes like 'x'
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            if hasattr(data, 'x') and torch.is_tensor(data.x):
                data.num_nodes = data.x.size(0)  # Infer from node features
            # Add other ways to infer num_nodes if necessary

        return data
