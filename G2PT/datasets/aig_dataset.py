# G2PT/datasets/aig_pyg_dataset.py
# Defines Dataset and DataModule for loading pre-processed AIG PyG .pt files

import os
import pathlib
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
# Adjust import path based on your directory structure
# If aig_pyg_dataset.py is inside 'datasets', this should work:
from .abstract_dataset import AbstractDataModule, AbstractDatasetInfos

class AIGPygDataset(InMemoryDataset):
    """
    Loads pre-processed PyTorch Geometric Data objects for AIGs
    from saved .pt files (train.pt, val.pt, test.pt).
    Assumes files are located in root/raw/.
    """
    def __init__(self, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        # root should be './aig_pyg/' (the directory containing 'raw' and 'processed')
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load processed data
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(f"Loaded processed PyG data for split '{split}' from {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"Processed file not found for split '{split}'. Need to run process() first.")
            # If you want it to automatically process if file not found:
            # print("Processing raw data...")
            # self.process()
            # self.data, self.slices = torch.load(self.processed_paths[0])
            # print(f"Loaded processed PyG data for split '{split}' after processing.")
            raise # Or raise error requiring explicit processing call
        except Exception as e:
            print(f"Error loading processed file for split '{split}': {e}")
            raise e

    @property
    def raw_file_names(self):
        # Files expected in root/raw/ relative to the root specified in DataModule
        return [f'{self.split}.pt']

    @property
    def processed_file_names(self):
        # Files saved in root/processed/ relative to the root specified in DataModule
        # Using a different name to distinguish from raw files
        return [f'aig_processed_{self.split}.pt']

    def download(self):
        # No download needed, assumes raw files are created by prepare_aig_pyg.py
        print(f"AIGPygDataset: Download step skipped for split '{self.split}', expects raw .pt files.")
        raw_path = os.path.join(self.raw_dir, f'{self.split}.pt')
        if not os.path.exists(raw_path):
             # Provide a more helpful error message if the file is missing.
             raise FileNotFoundError(
                 f"Raw file not found: {raw_path}. "
                 f"Ensure you have run the script to generate PyG .pt files first (e.g., prepare_aig_pyg.py) "
                 f"and that the 'datadir' in CFG points to the correct base directory ('./aig_pyg/')."
             )

    def process(self):
        # Read data from raw files
        raw_path = self.raw_paths[0] # Will be root/raw/train.pt etc.
        print(f"Processing raw PyG data from: {raw_path}")
        try:
            data_list = torch.load(raw_path)
        except FileNotFoundError:
             print(f"Raw file not found during process(): {raw_path}")
             raise
        except Exception as e:
             print(f"Error loading raw file {raw_path} during process(): {e}")
             raise

        if not isinstance(data_list, list):
             print(f"Error: Expected a list of Data objects from {raw_path}, got {type(data_list)}. Creating empty dataset.")
             data_list = [] # Process empty list

        # Apply pre-filtering and pre-transformations if specified
        if self.pre_filter is not None:
             print(f"Applying pre-filter...")
             data_list = [data for data in data_list if self.pre_filter(data)]
             print(f"Data count after filter: {len(data_list)}")


        if self.pre_transform is not None:
             print(f"Applying pre-transform...")
             data_list = [self.pre_transform(data) for data in data_list]
             print(f"Data transformed.")


        # Collate the list of Data objects into a single large Data object
        data, slices = self.collate(data_list)
        save_path = self.processed_paths[0] # Will be root/processed/aig_processed_train.pt etc.
        print(f"Saving processed {self.split} data ({len(data_list)} graphs) to {save_path}...")
        torch.save((data, slices), save_path)
        print("Processing complete.")


class AIGPygDataModule(AbstractDataModule):
    """
    DataModule for loading the AIG dataset stored as PyG .pt files.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        # datadir in CFG should point to the base directory where 'raw' and 'processed' live
        # e.g., './aig_pyg/'
        self.datadir = cfg.dataset.datadir
        # Assumes this script is in 'datasets', so parent is 'G2PT/'
        base_path = pathlib.Path(os.path.realpath(__file__)).parent.parent
        root_path = os.path.join(base_path, 'datasets', self.datadir) # Construct full path

        print(f"Initializing AIGPygDataModule with root path: {root_path}")
        if not os.path.exists(root_path):
            print(f"Warning: Root path does not exist: {root_path}")
            # Decide if you want to create it or raise error
            # os.makedirs(root_path)

        datasets = {}
        for split in ['train', 'val', 'test']:
            try:
                # Initialize the dataset for each split
                # This will trigger download (checking raw files) and process (if processed not found)
                datasets[split] = AIGPygDataset(split=split, root=root_path)
            except FileNotFoundError as e:
                print(f"Error initializing AIGPygDataset for split '{split}': {e}")
                print(f"Make sure the raw file '{split}.pt' exists in '{os.path.join(root_path, 'raw')}'")
                raise # Re-raise the error to stop execution if raw files are missing
            except Exception as e:
                print(f"Unexpected error initializing AIGPygDataset for split '{split}': {e}")
                raise

        print(f"Dataset sizes: Train={len(datasets.get('train',[]))}, Val={len(datasets.get('val',[]))}, Test={len(datasets.get('test',[]))}")
        # Initialize the AbstractDataModule with the PyG datasets
        # Note: batch_size, num_workers from cfg.train are used by the DataLoader later
        super().__init__(cfg, datasets)

# G2PT/datasets/aig_pyg_dataset.py (Add this class)


# (Keep AIGPygDataset and AIGPygDataModule definitions from the previous step here)


class AIGDatasetInfos(AbstractDatasetInfos):
    """
    Provides metadata and statistics for the AIG dataset,
    assuming data might be loaded via AIGPygDataModule.
    Statistics (like n_nodes, node_types, edge_types distributions)
    are expected to be loaded from files or computed elsewhere,
    similar to the non-Pyg version.
    """
    def __init__(self, datamodule: AbstractDataModule, cfg, recompute_statistics=False):
        # datamodule is passed but might not be directly used if loading pre-computed stats
        self.datamodule = datamodule
        self.name = 'aig'
        self.input_dims = None # Will be set by compute_input_output_dims if called later
        self.output_dims = None # Will be set by compute_input_output_dims if called later

        # --- AIG Specific Type Definitions ---
        # Node types based on vocabulary
        self.atom_decoder = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'] # Order matters if mapping index->name
        self.atom_encoder = {name: i for i, name in enumerate(self.atom_decoder)} # Maps name->index (0-3)
        self.num_atom_types = len(self.atom_decoder)

        # Corresponding Vocabulary IDs (Important!) - Matches vocab.json
        # This maps the index (0-3) used internally above to the actual vocab ID
        self.feature_index_to_vocab_id = {
             0: 71, # CONST0
             1: 72, # PI
             2: 73, # AND
             3: 74 # PO
        }
        # Reverse map might be useful sometimes
        self.vocab_id_to_feature_index = {v: k for k, v in self.feature_index_to_vocab_id.items()}

        # Edge types based on vocabulary
        self.bond_decoder = ['EDGE_INV', 'EDGE_REG'] # Order matters
        self.bond_encoder = {name: i for i, name in enumerate(self.bond_decoder)} # Maps name->index (0-1)
        self.num_bond_types = len(self.bond_decoder)
         # Maps edge feature index (0-1) to edge vocab ID
        self.edge_feature_index_to_vocab_id = {
             0: 75, # INV
             1: 76  # REG
        }

        # --- Statistics ---
        # These should ideally represent distributions over the final VOCAB IDs (97-100 for nodes)
        # and the number of nodes per graph.
        # We will attempt to load them from a standard location or use defaults.
        self.n_nodes = None         # Distribution of node counts per graph
        self.node_types = None      # Distribution of node type VOCAB IDs (e.g., counts for ID 97, 98, 99, 100)
        self.edge_types = None      # Distribution of edge type VOCAB IDs (e.g., counts for ID 101, 102 + no-edge)
        self.max_n_nodes = 64     # Default max_n_nodes, will be updated from n_nodes distribution

        # Define path for potential pre-computed statistics
        # Assumes stats are saved relative to the *final* data dir (e.g., './aig/')
        final_data_dir = cfg.dataset.get('final_output_dir', '../aig') # Get from cfg or default
        stats_dir = os.path.join(final_data_dir, 'stats')
        os.makedirs(stats_dir, exist_ok=True) # Ensure stats dir exists

        meta_files = {
            "n_nodes": os.path.join(stats_dir, f'{self.name}_n_counts.txt'),
            "node_types": os.path.join(stats_dir, f'{self.name}_node_types_vocab_dist.txt'), # Indicate vocab ID dist
            "edge_types": os.path.join(stats_dir, f'{self.name}_edge_types_vocab_dist.txt'), # Indicate vocab ID dist
        }

        # --- Load or Compute Statistics ---
        # Option 1: Load pre-computed stats if they exist
        stats_loaded = False
        if not recompute_statistics and all(os.path.exists(p) for p in meta_files.values()):
             try:
                 self.n_nodes = torch.from_numpy(np.loadtxt(meta_files["n_nodes"])).float()
                 self.node_types = torch.from_numpy(np.loadtxt(meta_files["node_types"])).float()
                 self.edge_types = torch.from_numpy(np.loadtxt(meta_files["edge_types"])).float()
                 print(f"Loaded pre-computed statistics for AIG from {stats_dir}")
                 stats_loaded = True
             except Exception as e:
                 print(f"Warning: Failed to load statistics files from {stats_dir}: {e}. Will try to compute or use defaults.")
                 stats_loaded = False

        # Option 2: Compute stats from datamodule (if not loaded and possible)
        # Note: This requires dataloaders to be ready and might be slow.
        # Also, the default node_types()/edge_counts() compute stats based on *features*,
        # which would need re-mapping to vocab IDs. We skip direct computation here
        # and rely on pre-computed files or defaults for simplicity, like the original AIGinfos.
        if not stats_loaded:
            print("Statistics files not found or failed to load. Using default placeholders.")
            print(f"(To compute stats, ensure files exist in {stats_dir} or implement computation logic)")
            # Load Default Stats (Placeholder)
            self._load_default_stats() # Ensure self.n_nodes, self.node_types are set

        # --- Finalize Basic Info ---
        # Update max_n_nodes based on the distribution
        if self.n_nodes is not None and len(self.n_nodes) > 1:
            try:
                nz_indices = torch.nonzero(self.n_nodes).squeeze()
                if nz_indices.numel() > 0:
                    # Find the index of the last non-zero count, which corresponds to max_n_nodes
                    self.max_n_nodes = nz_indices.max().item()
                else:
                    self.max_n_nodes = 32 # Fallback if distribution is empty/zero
                print(f"Determined max_n_nodes from distribution: {self.max_n_nodes}")
            except Exception as e:
                print(f"Warning: Could not determine max_n_nodes from distribution: {e}. Using default: {self.max_n_nodes}")
                self.max_n_nodes = 64 # Fallback
        else:
            print(f"Warning: n_nodes distribution not available or invalid. Using default max_n_nodes: {self.max_n_nodes}")
            self.max_n_nodes = 64 # Fallback

        # !!! Crucial: Call complete_infos from the base class !!!
        # It requires self.n_nodes and self.node_types.
        # self.node_types MUST represent the distribution over VOCAB IDs (97-100).
        if self.n_nodes is None or self.node_types is None:
             print("ERROR: n_nodes or node_types distribution is None. Cannot complete infos.")
             # Handle error appropriately, maybe raise exception or load defaults again
             self._load_default_stats() # Ensure they are tensors

        # Ensure node_types tensor length matches vocab size if loaded from file
        expected_node_vocab_size = max(self.feature_index_to_vocab_id.values()) + 1 # Assumes dense IDs up to max
        if self.node_types.shape[0] < expected_node_vocab_size:
             print(f"Warning: Loaded node_types distribution length ({self.node_types.shape[0]}) is smaller than expected vocab size ({expected_node_vocab_size}). Padding with zeros.")
             # Pad the tensor if loaded distribution doesn't cover full vocab range
             padding_size = expected_node_vocab_size - self.node_types.shape[0]
             self.node_types = torch.cat((self.node_types, torch.zeros(padding_size, dtype=self.node_types.dtype)), dim=0)


        # This needs the distribution of node counts and the distribution over node *types*
        # Make sure self.node_types represents the distribution over vocab IDs 97-100 correctly.
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        print(f"AIGDatasetInfos initialized: max_n_nodes={self.max_n_nodes}, num_node_types={self.num_atom_types}")


    def _load_default_stats(self):
        """Load placeholder statistics if files are missing or computation fails."""
        print("Loading default placeholder statistics for AIG.")
        # Default max_n_nodes if not determined otherwise
        max_nodes = getattr(self, 'max_n_nodes', 64)
        self.n_nodes = torch.zeros(max_nodes + 1, dtype=torch.float)
        # Simple uniform distribution over assumed typical sizes
        if max_nodes >= 20:
            self.n_nodes[16:min(21, max_nodes+1)].fill_(1.0 / 5.0) # Fill range [16, 20]
        elif max_nodes > 0:
            self.n_nodes[1:max_nodes+1].fill_(1.0 / max_nodes)

        # Default node type distribution (uniform over vocab IDs 97-100)
        # Need a tensor representing counts/probs for *all* vocab IDs up to max AIG ID
        max_node_vocab_id = max(self.feature_index_to_vocab_id.values())
        self.node_types = torch.zeros(max_node_vocab_id + 1, dtype=torch.float)
        # Assign uniform probability to the actual AIG node IDs
        num_aig_node_types = len(self.feature_index_to_vocab_id)
        if num_aig_node_types > 0:
             uniform_prob = 1.0 / num_aig_node_types
             for vocab_id in self.feature_index_to_vocab_id.values():
                 self.node_types[vocab_id] = uniform_prob

        # Default edge type distribution (uniform over vocab IDs 101-102 + maybe 0 for no-edge)
        # The structure depends on how edge_counts() would format it.
        # Let's assume it creates entries up to max vocab ID.
        max_edge_vocab_id = max(self.edge_feature_index_to_vocab_id.values())
        self.edge_types = torch.zeros(max_edge_vocab_id + 1, dtype=torch.float)
        # Simple default: assume equal probability for INV and REG edges, ignore no-edge for now.
        num_aig_edge_types = len(self.edge_feature_index_to_vocab_id)
        if num_aig_edge_types > 0:
            uniform_prob = 1.0 / num_aig_edge_types
            for vocab_id in self.edge_feature_index_to_vocab_id.values():
                 self.edge_types[vocab_id] = uniform_prob