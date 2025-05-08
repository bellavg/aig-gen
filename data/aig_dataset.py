#!/usr/bin/env python3
import os
import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
from typing import List, Tuple  # Optional if you remove all type hints for unused parts
import torch_geometric.io.fs as pyg_fs  # Import PyG's filesystem utilities

# --- AIG Model/Data Parameters (Constants used if data has specific structure) ---
MAX_NODES_PAD = 64  # Kept for context if your model might still use it
NUM_NODE_FEATURES = 4
NUM_EXPLICIT_EDGE_TYPES = 2
NUM_ADJ_CHANNELS = NUM_EXPLICIT_EDGE_TYPES + 1


# --- End AIG Parameters ---

class AIGPreprocessedDatasetLoader(InMemoryDataset):
    def __init__(self, root: str, dataset_name: str, split: str,
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Loads a pre-processed AIG dataset.
        """
        self.dataset_name = dataset_name
        self.split = split

        dataset_processed_root_path = osp.join(root, dataset_name)

        # --- DEBUG PRINTS ---
        print(f"\n--- [AIGLoader DEBUG - Before super().__init__] ---")
        print(f"Passed 'root' to AIGLoader: {root}")
        print(f"Passed 'dataset_name' to AIGLoader: {dataset_name}")
        print(f"Passed 'split' to AIGLoader: {split}")
        print(
            f"'dataset_processed_root_path' (this will be 'self.root' for InMemoryDataset): {dataset_processed_root_path}")

        _debug_expected_processed_dir = osp.join(dataset_processed_root_path, "processed")
        _debug_expected_processed_file = f'{self.split}_processed_data.pt'
        _debug_full_path_to_check = osp.join(_debug_expected_processed_dir, _debug_expected_processed_file)

        print(f"Manually constructed full path to .pt file (for check): {_debug_full_path_to_check}")
        print(f"  osp.exists on this manually constructed path: {osp.exists(_debug_full_path_to_check)}")
        # --- ADDED PYG FS CHECK ---
        print(
            f"  torch_geometric.io.fs.exists on this manually constructed path: {pyg_fs.exists(_debug_full_path_to_check)}")
        # --- END ADDED PYG FS CHECK ---
        if osp.exists(_debug_full_path_to_check):
            try:
                print(f"    Size of this file (os.path.getsize): {osp.getsize(_debug_full_path_to_check)}")
            except Exception as e_size_debug:
                print(f"    Error getting size: {e_size_debug}")
        print(f"--- [AIGLoader DEBUG - Calling super().__init__ now...] ---")
        # --- END DEBUG PRINTS ---

        super().__init__(dataset_processed_root_path, transform, pre_transform, pre_filter)

        # --- DEBUG PRINTS AFTER super().__init__ ---
        print(f"\n--- [AIGLoader DEBUG - After super().__init__] ---")
        print(f"self.root (from InMemoryDataset): {self.root}")
        print(f"self.processed_dir (from InMemoryDataset): {self.processed_dir}")
        print(f"self.processed_file_names (from InMemoryDataset via AIGLoader): {self.processed_file_names}")
        if hasattr(self, 'processed_paths') and self.processed_paths:
            print(f"self.processed_paths[0] (from InMemoryDataset): {self.processed_paths[0]}")
            print(f"  osp.exists(self.processed_paths[0]) after super init: {osp.exists(self.processed_paths[0])}")
            # --- ADDED PYG FS CHECK ---
            print(
                f"  torch_geometric.io.fs.exists(self.processed_paths[0]) after super init: {pyg_fs.exists(self.processed_paths[0])}")
            # --- END ADDED PYG FS CHECK ---
        else:
            print(f"self.processed_paths is not set or is empty after super init.")
        # --- END DEBUG PRINTS AFTER super().__init__ ---

        if self.data is None or self.slices is None:
            expected_file_path_in_error = "Unknown"
            if hasattr(self, 'processed_paths') and self.processed_paths:
                expected_file_path_in_error = self.processed_paths[0]
            else:
                _fallback_processed_dir = osp.join(dataset_processed_root_path, "processed")
                _fallback_processed_file = f'{self.split}_processed_data.pt'
                expected_file_path_in_error = osp.join(_fallback_processed_dir, _fallback_processed_file)

            print(f"[AIGLoader DEBUG] self.data or self.slices is None. Raising FileNotFoundError.")
            raise FileNotFoundError(
                f"Pre-processed file not found or failed to load: {expected_file_path_in_error}. "
                f"This dataset loader expects .pt files to already exist. "
                f"Please ensure 'root' ('{root}'), 'dataset_name' ('{dataset_name}'), "
                f"and 'split' ('{split}') arguments correctly point to an existing file."
            )

        print(
            f"Dataset '{self.dataset_name}' split '{self.split}' successfully loaded from {self.processed_paths[0]}. Samples: {len(self)}")

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return [f'{self.split}_processed_data.pt']

    def download(self):
        _path_download_tried = "UnknownPathInDownload"
        if hasattr(self, 'processed_paths') and self.processed_paths:
            _path_download_tried = self.processed_paths[0]
        else:
            if hasattr(self, 'root') and self.root:
                _fallback_proc_dir_in_download = osp.join(self.root, "processed")
                _fallback_proc_file_in_download = f'{self.split}_processed_data.pt'
                _path_download_tried = osp.join(_fallback_proc_dir_in_download, _fallback_proc_file_in_download)
            else:
                _path_download_tried = f"PathConstructionFailedInDownload(root_unknown)/processed/{self.split}_processed_data.pt"

        print(
            f"[AIGLoader DEBUG] download() method called! This means InMemoryDataset thinks '{_path_download_tried}' does not exist (likely via pyg_fs.exists).")

        raise FileNotFoundError(
            f"Pre-processed file not found: {_path_download_tried}. "
            f"This loader expects this file to exist and does not support downloading or on-the-fly processing. "
            f"Please ensure the file is present at the correct path."
        )

    def process(self):
        raise NotImplementedError(
            "process() should not be called. This dataset loader is for pre-processed files only. "
            f"Ensure {self.processed_paths[0] if hasattr(self, 'processed_paths') and self.processed_paths else 'UNKNOWN_PROCESSED_PATH'} exists."
        )
