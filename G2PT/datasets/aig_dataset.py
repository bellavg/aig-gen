# File: G2PT/datasets/digress_datasets/aig_dataset.py

import os
import torch
import numpy as np
import json
from pathlib import Path
from .digress_datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
# Note: Changed import path assuming aig_dataset.py is in digress_datasets

# Note: The actual AIGDataset class that *reads* the .bin files is omitted here.
# G2PT's training pipeline might handle loading directly from the .bin files
# using the paths and shapes defined in data_meta.json, potentially via a
# generic dataloader configured by the DataModule. If explicit loading is needed,
# a Dataset class reading the memmap files would need to be implemented.

class AIGDataModule(AbstractDataModule):
    """
    DataModule for the AIG dataset.
    Assumes that the data (.bin files) has already been processed by prepare_aig.py.
    The actual loading of .bin files is likely handled by the main training script
    or a lower-level dataloader configured by this module and data_meta.json.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        # Should point to the directory containing train/val/test subdirs with .bin files
        self.datadir = cfg.dataset.datadir

        # We initialize with None datasets, as the actual data loading from .bin
        # might be handled elsewhere based on data_meta.json.
        # If the framework requires an explicit Dataset instance here,
        # you'd need one that reads the memmap files.
        datasets = {'train': None, 'val': None, 'test': None}
        print(f"Initializing AIGDataModule. Data loading from {self.datadir} based on .bin files and data_meta.json is expected.")
        super().__init__(cfg, datasets)

        # Placeholder for dataset info - will be properly initialized later
        self.dataset_infos = None

    # You might need to override setup() or prepare_data() if specific
    # actions are needed before training, e.g., ensuring statistics are computed.
    # def setup(self, stage=None):
    #     super().setup(stage)
    #     # Example: initialize dataset_infos after setup
    #     if self.dataset_infos is None:
    #          self.dataset_infos = AIGinfos(self, self.cfg)


class AIGinfos(AbstractDatasetInfos):
    """Holds statistics and metadata for the AIG dataset."""
    def __init__(self, datamodule: AIGDataModule, cfg, recompute_statistics=False):
        self.name = 'AIG'
        self.input_dims = None # Will be set by compute_input_output_dims if called
        self.output_dims = None # Will be set by compute_input_output_dims if called

        # --- Node Types ---
        # Must match the order implicitly defined by IDs 97, 98, 99, 100
        self.node_types_enum = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
        self.num_node_types = len(self.node_types_enum)
        # Vocab IDs for node types (consistent with prepare_aig.py and vocab.json)
        self.node_type_ids = {name: i + 97 for i, name in enumerate(self.node_types_enum)}

        # --- Edge Types ---
        # Must match the order implicitly defined by IDs 101, 102
        self.edge_types_enum = ["EDGE_INV", "EDGE_REG"]
        self.num_edge_types = len(self.edge_types_enum)
        # Vocab IDs for edge types (consistent with prepare_aig.py and vocab.json)
        # Note: edge_counts method usually returns stats including a "no edge" type at index 0
        self.edge_type_ids = {name: i + 101 for i, name in enumerate(self.edge_types_enum)}

        # --- Dataset Statistics ---
        self.n_nodes = None      # Distribution of node counts
        self.node_types = None   # Distribution of node type IDs
        self.edge_types = None   # Distribution of edge type IDs (incl. no-edge)
        self.max_n_nodes = 256   # Default max number of nodes, updated from stats

        # --- Load or Compute Statistics ---
        stats_dir = os.path.join(cfg.dataset.datadir, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        meta_files = {
            "n_nodes": os.path.join(stats_dir, f'{self.name}_n_counts.txt'),
            "node_types": os.path.join(stats_dir, f'{self.name}_node_types.txt'),
            "edge_types": os.path.join(stats_dir, f'{self.name}_edge_types.txt'),
        }

        # Determine if stats can be computed (dataloaders must be available)
        # This check might be better placed within the training script logic
        # before initializing Infos, or requires datamodule setup to be called first.
        can_compute_stats = hasattr(datamodule, 'dataloaders') and datamodule.dataloaders

        if recompute_statistics or not all(os.path.exists(p) for p in meta_files.values()):
            if not can_compute_stats:
                 print(f"Warning: Statistics cannot be computed for {self.name} because dataloaders are not ready "
                       f"(run prepare_dataloader() on DataModule first?). Attempting to load from file or using defaults.")
                 self._load_stats_from_files_or_defaults(meta_files)
            else:
                 print(f"Recomputing statistics for {self.name} dataset using provided DataModule...")
                 try:
                     # Compute stats using methods from AbstractDataModule
                     self.n_nodes = datamodule.node_counts(max_nodes_possible=1024) # Check more nodes
                     self.node_types = datamodule.node_types() # Should yield counts for IDs 97-100
                     self.edge_types = datamodule.edge_counts() # Should yield counts for ID 101, 102 (+ no-edge)

                     # Basic validation of computed stats shapes
                     if self.node_types.size(0) != self.num_node_types:
                         print(f"Warning: Computed node types dist size ({self.node_types.size(0)}) "
                               f"doesn't match expected ({self.num_node_types}). Check data processing/vocab.")
                     # edge_counts typically includes a "no edge" count at index 0
                     # Need to confirm how AbstractDataModule calculates this for ID-based attributes
                     # Assuming it returns counts for IDs 101, 102 and maybe others? Let's adjust expected size based on max ID.
                     # max_edge_id = max(self.edge_type_ids.values())
                     # expected_edge_stat_size = max_edge_id + 1 # Assuming counts up to max ID
                     # if self.edge_types.size(0) != expected_edge_stat_size:
                     #    print(f"Warning: Computed edge types dist size ({self.edge_types.size(0)}) "
                     #          f"doesn't match expected size based on max ID ({expected_edge_stat_size}).")
                     # Safer: Just print shape for now
                     print(f"Computed node type distribution shape: {self.node_types.shape}")
                     print(f"Computed edge type distribution shape: {self.edge_types.shape}")


                     print("Saving computed statistics...")
                     np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
                     np.savetxt(meta_files["node_types"], self.node_types.numpy())
                     np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
                     print(f"Saved statistics to {stats_dir}")

                 except Exception as e:
                     print(f"Error computing statistics: {e}. Using default placeholders.")
                     self._load_default_stats()
        else:
             # Load stats from files if they exist
             self._load_stats_from_files_or_defaults(meta_files)

        # Update max_n_nodes based on the distribution
        if self.n_nodes is not None and len(self.n_nodes) > 1:
            try:
                 # Find the last index with a non-zero count
                 nz_indices = torch.nonzero(self.n_nodes).squeeze()
                 if nz_indices.numel() > 0:
                     self.max_n_nodes = max(nz_indices.max().item(), 1)
                 else: # Handle all-zero case
                     self.max_n_nodes = 32 # Fallback if distribution is empty/zero
                 print(f"Determined max_n_nodes from distribution: {self.max_n_nodes}")
            except Exception as e:
                 print(f"Warning: Could not determine max_n_nodes from distribution: {e}. Using default: {self.max_n_nodes}")
                 self.max_n_nodes = 256 # Fallback
        else:
            print(f"Warning: n_nodes distribution not available. Using default max_n_nodes: {self.max_n_nodes}")
            self.max_n_nodes = 256 # Fallback

        # Finalize - crucial step from AbstractDatasetInfos
        # Needs self.n_nodes and self.node_types to be tensors
        if self.n_nodes is None or self.node_types is None:
             print("Warning: Cannot complete infos as node distributions are missing. Loading defaults.")
             self._load_default_stats() # Ensure they are tensors before calling complete_infos

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        print(f"AIGinfos initialized: max_n_nodes={self.max_n_nodes}, num_node_types={self.num_node_types}")


    def _load_stats_from_files_or_defaults(self, meta_files):
        """Loads statistics from files or uses defaults if files are missing/invalid."""
        try:
            self.n_nodes = torch.from_numpy(np.loadtxt(meta_files["n_nodes"])).float()
            self.node_types = torch.from_numpy(np.loadtxt(meta_files["node_types"])).float()
            self.edge_types = torch.from_numpy(np.loadtxt(meta_files["edge_types"])).float()
            print(f"Loaded existing statistics from {os.path.dirname(meta_files['n_nodes'])}")
        except Exception as e:
            print(f"Warning: Failed to load statistics files: {e}. Using default placeholders.")
            self._load_default_stats()

    def _load_default_stats(self):
        """Load placeholder statistics."""
        print("Loading default placeholder statistics.")
        self.max_n_nodes = 256
        self.n_nodes = torch.zeros(self.max_n_nodes + 1, dtype=torch.float)
        # Simple uniform distribution over assumed typical sizes
        if self.max_n_nodes >= 20:
            self.n_nodes[16:21].fill_(1.0 / 5.0)
        else:
             self.n_nodes[1:self.max_n_nodes+1].fill_(1.0 / self.max_n_nodes) # Uniform if max nodes < 16

        # Uniform distribution over node types
        self.node_types = torch.ones(self.num_node_types, dtype=torch.float) / self.num_node_types
        # Uniform distribution over edge types (including potential "no edge" type at index 0)
        # Adjust size based on how AbstractDataModule.edge_counts works. Assume it includes 0 + max ID.
        max_edge_id = max(self.edge_type_ids.values())
        self.edge_types = torch.ones(max_edge_id + 1, dtype=torch.float) / (max_edge_id + 1)