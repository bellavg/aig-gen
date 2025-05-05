# G2PT/datasets/aig_dataset.py
# Defines Dataset and DataModule for loading pre-processed AIG PyG .pt files.
# Expects single raw files (train.pt, val.pt, test.pt) per split.
# MODIFIED: Only initializes datasets for splits where the raw file exists,
#           assigns None to missing splits before passing to the base class.

import os
import pathlib
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import sys # Added sys import for path adjustment if needed
import gc # Added gc import
from G2PT.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# Import the AIG configuration directly (needed for stats info and vocab details)
try:
    import G2PT.configs.aig as aig_cfg
    print(f"Successfully imported G2PT.configs.aig.")
except ImportError as e:
    print(f"Error importing G2PT.configs.aig: {e}")
    print("Ensure G2PT/configs/aig.py exists and G2PT is in the Python path.")
    sys.exit(1)


class AIGPygDataset(InMemoryDataset):
    """
    Loads pre-processed PyTorch Geometric Data objects for AIGs
    from saved .pt files (train.pt, val.pt, test.pt).
    Assumes raw files are located in root/raw/.
    Processed data is saved in root/processed/.
    """
    def __init__(self, split, root, transform=None, pre_transform=None, pre_filter=None):
        """
        Initializes the dataset for a specific split.

        Args:
            split (str): The dataset split ('train', 'val', or 'test').
            root (str): The root directory where 'raw/' and 'processed/' subfolders reside or will be created.
            transform: PyG transforms applied after loading.
            pre_transform: PyG transforms applied before saving processed data.
            pre_filter: PyG pre-filtering applied before saving processed data.
        """
        self.split = split
        print(f"AIGPygDataset '{split}': Initializing with root='{root}'")
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load the single processed data file (if it exists)
        try:
            # self.processed_paths[0] points to root/processed/aig_processed_{split}.pt
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Loaded processed PyG data for split '{split}' from {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"Processed file not found for split '{split}' at {self.processed_paths[0]}. Will run process().")
            # PyG's InMemoryDataset handles calling self.process() automatically.
        except Exception as e:
            print(f"Error loading processed file for split '{split}': {e}")
            raise e # Re-raise other unexpected errors

    @property
    def raw_file_names(self):
        """Specifies the name of the single raw file expected for this split."""
        # Expects root/raw/train.pt, root/raw/val.pt, etc.
        # NOTE: If you created train2.pt, this needs adjustment to handle it during process().
        #       Keeping it simple for now, assuming only train.pt is used for 'train'.
        return [f'{self.split}.pt']

    @property
    def processed_file_names(self):
        """Specifies the name of the single processed file for the current split."""
        # Saves to root/processed/aig_processed_train.pt, etc.
        return [f'aig_processed_{self.split}.pt']

    def download(self):
        """Checks if the raw file exists. No actual download."""
        print(f"AIGPygDataset: Download step called for split '{self.split}'. Checking for raw file.")
        raw_path = self.raw_paths[0] # Path to the single raw file (e.g., root/raw/train.pt)
        print(f"AIGPygDataset '{self.split}': Checking for raw file at '{raw_path}'")
        if not os.path.exists(raw_path):
             raise FileNotFoundError(
                 f"Raw file not found: {raw_path}. "
                 f"This file ({self.raw_file_names[0]}) should be created by the 'aig_pkl_to_pyg.py' script "
                 f"(the version that saves combined train.pt/val.pt). "
                 f"Ensure that script ran successfully and the 'root' path ('{self.root}') is correct."
             )
        print(f"AIGPygDataset '{self.split}': Raw file found at '{raw_path}'")


    def process(self):
        """
        Loads raw data from the single raw file (e.g., train.pt), applies transformations/filters,
        and saves a single processed data file (e.g., aig_processed_train.pt).
        """
        raw_path = self.raw_paths[0]
        print(f"AIGPygDataset '{self.split}': Processing raw PyG data from: {raw_path}")
        try:
            # Load the list of Data objects from the raw .pt file
            data_list = torch.load(raw_path, weights_only=False) # Use weights_only=False for Data objects
        except FileNotFoundError:
             print(f"Raw file not found during process(): {raw_path}")
             raise
        except Exception as e:
             print(f"Error loading raw file {raw_path} during process(): {e}")
             raise

        if not isinstance(data_list, list):
             print(f"Error: Expected a list of Data objects from {raw_path}, got {type(data_list)}. Processing empty dataset.")
             data_list = []

        print(f"Total graphs loaded from raw file for split '{self.split}': {len(data_list)}")

        if self.pre_filter is not None:
             num_before_filter = len(data_list)
             print(f"Applying pre-filter to {num_before_filter} graphs...")
             data_list = [data for data in data_list if self.pre_filter(data)]
             print(f"Data count after filter: {len(data_list)}")

        if self.pre_transform is not None:
             print(f"Applying pre-transform...")
             data_list = [self.pre_transform(data) for data in data_list]
             print(f"Data transformed.")

        # This step might still require significant memory if the combined data_list is huge.
        print(f"Collating {len(data_list)} graphs for split '{self.split}'...")
        data, slices = self.collate(data_list)
        print("Collation complete.")

        save_path = self.processed_paths[0]
        print(f"Saving processed {self.split} data ({len(data_list)} graphs) to {save_path}...")
        try:
            # Use _use_new_zipfile_serialization=False if you encounter issues saving large files
            torch.save((data, slices), save_path, _use_new_zipfile_serialization=False)
            print(f"AIGPygDataset '{self.split}': Processing complete.")
        except Exception as e:
             print(f"Error saving processed file {save_path}: {e}")
             raise e # Re-raise the error


class AIGPygDataModule(AbstractDataModule):
    """
    DataModule for loading the AIG dataset stored as PyG .pt files.
    Uses the root path provided via cfg.dataset.datadir.
    Initializes the standard AIGPygDataset (expecting single raw files).
    MODIFIED: Only initializes datasets for splits where raw files exist,
              and assigns None to missing splits before passing to the base class.
    """
    def __init__(self, cfg): # cfg should contain dataset.datadir and train params like batch_size
        self.train_cfg = cfg
        self.aig_cfg = aig_cfg

        # --- Get the dataset root path ---
        try:
            root_path = cfg.dataset.datadir
            if not root_path or not isinstance(root_path, str): raise ValueError("datadir invalid")
            root_path = os.path.abspath(root_path)
            print(f"AIGPygDataModule: Using root path from cfg: {root_path}")
        except (AttributeError, ValueError) as e:
            print(f"Error getting root path from cfg: {e}")
            raise
        # --- Root path determined ---

        # --- Ensure directory structure exists ---
        print(f"AIGPygDataModule: Final root path: {root_path}")
        processed_path = os.path.join(root_path, 'processed')
        raw_path_dir = os.path.join(root_path, 'raw')
        try:
            os.makedirs(root_path, exist_ok=True)
            os.makedirs(raw_path_dir, exist_ok=True)
            os.makedirs(processed_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory structure at {root_path}: {e}")
            raise
        # --- Directories ensured ---

        # --- Initialize only EXISTING splits, assign None otherwise ---
        # *** FIX: Ensure all keys ('train', 'val', 'test') exist in the dict ***
        initialized_datasets = {}
        for split in ['train', 'val', 'test']: # Iterate through ALL expected splits
            raw_file_path = os.path.join(raw_path_dir, f"{split}.pt")
            if os.path.exists(raw_file_path):
                print(f"Found raw file for split '{split}', initializing dataset...")
                try:
                    # Initialize the dataset for this existing split
                    initialized_datasets[split] = AIGPygDataset(split=split, root=root_path)
                except Exception as e:
                    # Catch errors during dataset initialization itself (e.g., loading processed file)
                    print(f"Error initializing AIGPygDataset for existing split '{split}': {e}")
                    # Assign None if initialization fails for an existing raw file
                    initialized_datasets[split] = None
                    # Optionally re-raise if this failure is critical
                    # raise
            else:
                # If the raw file doesn't exist, add None to the dictionary for this key
                print(f"Raw file for split '{split}' not found at '{raw_file_path}'. Setting dataset to None.")
                initialized_datasets[split] = None # <-- Assign None for missing splits
        # --- End split initialization loop ---

        # Check if at least train or val was initialized (optional, but good practice)
        if initialized_datasets.get('train') is None and initialized_datasets.get('val') is None:
             print("Warning: Neither train nor val datasets could be initialized. Check raw file paths.")
             # Depending on downstream code, this might be acceptable or an error.

        # Print final initialized dataset sizes (using len on the actual dataset object or 0 if None)
        print(f"Final Initialized Dataset sizes: Train={len(initialized_datasets.get('train',[]))}, Val={len(initialized_datasets.get('val',[]))}, Test=0")

        # --- Initialize the AbstractDataModule base class ---
        # Pass the dictionary which now *always* contains 'train', 'val', 'test' keys (value might be None)
        super().__init__(self.train_cfg, initialized_datasets)
        print("AIGPygDataModule initialization complete.")


class AIGDatasetInfos(AbstractDatasetInfos):
    """
    Provides metadata and statistics for the AIG dataset,
    using definitions from the imported aig_cfg.
    (No changes needed in this class)
    """
    def __init__(self, datamodule: AbstractDataModule, cfg=None, recompute_statistics=False):
        self.datamodule = datamodule
        self.name = aig_cfg.dataset
        self.input_dims = None; self.output_dims = None
        self.atom_decoder = list(aig_cfg.NODE_TYPE_KEYS)
        self.atom_encoder = {name: i for i, name in enumerate(self.atom_decoder)}
        self.num_atom_types = len(self.atom_decoder)
        self.feature_index_to_vocab_id = aig_cfg.NODE_FEATURE_INDEX_TO_VOCAB
        self.vocab_id_to_feature_index = {v: k for k, v in self.feature_index_to_vocab_id.items()}
        self.bond_decoder = list(aig_cfg.EDGE_TYPE_KEYS)
        self.bond_encoder = {name: i for i, name in enumerate(self.bond_decoder)}
        self.num_bond_types = len(self.bond_decoder)
        self.edge_feature_index_to_vocab_id = aig_cfg.EDGE_FEATURE_INDEX_TO_VOCAB
        self.n_nodes = None; self.node_types = None; self.edge_types = None
        self.max_n_nodes = aig_cfg.MAX_NODE_COUNT
        try:
            # Determine stats dir based on available datasets in datamodule
            # Check train, then val, then test for a valid root path
            # Use getattr for safer access to potentially missing dataset attributes
            if hasattr(datamodule, 'train_dataset') and datamodule.train_dataset and hasattr(datamodule.train_dataset, 'root'):
                 abs_data_dir = datamodule.train_dataset.root
            elif hasattr(datamodule, 'val_dataset') and datamodule.val_dataset and hasattr(datamodule.val_dataset, 'root'):
                 abs_data_dir = datamodule.val_dataset.root
            # elif hasattr(datamodule, 'test_dataset') and datamodule.test_dataset and hasattr(datamodule.test_dataset, 'root'): # Check test too
            #      abs_data_dir = datamodule.test_dataset.root
            else:
                 # Fallback: Try getting from module directly (requires DataModule to store root_path)
                 if hasattr(datamodule, 'root_path'): abs_data_dir = datamodule.root_path
                 else: raise ValueError("Could not determine data directory from datamodule.")
            if not abs_data_dir: raise ValueError("Determined data directory is empty or None.")
        except (AttributeError, ValueError) as e:
             print(f"Warning: Could not reliably get root path from datamodule: {e}. Falling back.")
             script_dir = os.path.dirname(os.path.realpath(__file__)); g2pt_root = os.path.dirname(script_dir)
             abs_data_dir = os.path.abspath(os.path.join(g2pt_root, os.path.normpath(aig_cfg.data_dir)))
        stats_dir = os.path.join(abs_data_dir, 'stats')
        print(f"AIGDatasetInfos: Looking for/saving statistics files in: {stats_dir}")
        os.makedirs(stats_dir, exist_ok=True)
        meta_files = { "n_nodes": os.path.join(stats_dir, f'{self.name}_n_counts.txt'),
                       "node_types": os.path.join(stats_dir, f'{self.name}_node_types_vocab_dist.txt'),
                       "edge_types": os.path.join(stats_dir, f'{self.name}_edge_types_vocab_dist.txt'), }
        stats_loaded = False
        if not recompute_statistics and all(os.path.exists(p) for p in meta_files.values()):
             try:
                 self.n_nodes = torch.from_numpy(np.loadtxt(meta_files["n_nodes"])).float()
                 self.node_types = torch.from_numpy(np.loadtxt(meta_files["node_types"])).float()
                 self.edge_types = torch.from_numpy(np.loadtxt(meta_files["edge_types"])).float()
                 print(f"Loaded pre-computed statistics for AIG from {stats_dir}"); stats_loaded = True
             except Exception as e: print(f"Warning: Failed to load statistics files from {stats_dir}: {e}."); stats_loaded = False
        if not stats_loaded: print("Statistics files not found or failed to load. Using default placeholder statistics."); self._load_default_stats()
        if self.n_nodes is not None and len(self.n_nodes) > 1:
            try:
                nz_indices = torch.nonzero(self.n_nodes).squeeze()
                if nz_indices.numel() > 0:
                    derived_max_nodes = nz_indices.max().item()
                    if derived_max_nodes > self.max_n_nodes: print(f"Updating max_n_nodes based on loaded distribution ({self.max_n_nodes} -> {derived_max_nodes})"); self.max_n_nodes = derived_max_nodes
                    elif derived_max_nodes < self.max_n_nodes: print(f"Note: Max node count from distribution ({derived_max_nodes}) is less than config ({self.max_n_nodes}). Using config value.")
                else: print(f"Warning: n_nodes distribution is all zeros. Using config max_n_nodes: {self.max_n_nodes}")
            except Exception as e: print(f"Warning: Could not determine max_n_nodes from distribution: {e}.")
        else: print(f"Using max_n_nodes from config: {self.max_n_nodes}")
        if self.n_nodes is None or self.node_types is None: raise ValueError("Failed to load or generate n_nodes/node_types statistics.")
        expected_vocab_size = aig_cfg.vocab_size
        if self.node_types.shape[0] < expected_vocab_size:
             print(f"Warning: Loaded/default node_types distribution length ({self.node_types.shape[0]}) < expected vocab size ({expected_vocab_size}). Padding.")
             padding_size = expected_vocab_size - self.node_types.shape[0]; self.node_types = torch.cat((self.node_types, torch.zeros(padding_size, dtype=self.node_types.dtype)), dim=0)
        elif self.node_types.shape[0] > expected_vocab_size:
             print(f"Warning: Loaded/default node_types distribution length ({self.node_types.shape[0]}) > expected vocab size ({expected_vocab_size}). Truncating."); self.node_types = self.node_types[:expected_vocab_size]
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        print(f"AIGDatasetInfos initialized: max_n_nodes={self.max_n_nodes}, num_node_types={self.num_atom_types}")

    def _load_default_stats(self):
        print("Loading default placeholder statistics for AIG.")
        max_nodes = self.max_n_nodes; self.n_nodes = torch.zeros(max_nodes + 1, dtype=torch.float)
        if max_nodes > 0: self.n_nodes[1:].fill_(1.0 / max_nodes)
        self.node_types = torch.zeros(aig_cfg.vocab_size, dtype=torch.float)
        num_aig_node_types = len(aig_cfg.NODE_TYPE_VOCAB)
        if num_aig_node_types > 0:
             uniform_prob = 1.0 / num_aig_node_types; valid_node_vocab_ids = [v for v in aig_cfg.NODE_TYPE_VOCAB.values() if v < aig_cfg.vocab_size]
             if valid_node_vocab_ids: self.node_types[valid_node_vocab_ids] = uniform_prob
             else: print("Warning: No valid node vocab IDs found in config within vocab size range.")
        self.edge_types = torch.zeros(aig_cfg.vocab_size, dtype=torch.float)
        num_aig_edge_types = len(aig_cfg.EDGE_TYPE_VOCAB)
        if num_aig_edge_types > 0:
            uniform_prob = 1.0 / num_aig_edge_types; valid_edge_vocab_ids = [v for v in aig_cfg.EDGE_TYPE_VOCAB.values() if v < aig_cfg.vocab_size]
            if valid_edge_vocab_ids: self.edge_types[valid_edge_vocab_ids] = uniform_prob
            else: print("Warning: No valid edge vocab IDs found in config within vocab size range.")

