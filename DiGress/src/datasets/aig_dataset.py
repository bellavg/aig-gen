import os
import torch
import re # For sorting filenames naturally
from torch_geometric.data import Data, Dataset
import pathlib
# Assuming your aig_cfg is accessible
try:
    import G2PT.configs.aig as aig_cfg
except ImportError:
    print("Error importing G2PT.configs.aig. Using fallback.")
    # Fallback definitions (same as before)
    class FallbackAigCfg:
        MAX_NODE_COUNT = 64
        NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
        EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]
        NODE_TYPE_ENCODING = {
            "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], "NODE_PI": [0.0, 1.0, 0.0, 0.0],
            "NODE_AND": [0.0, 0.0, 1.0, 0.0], "NODE_PO": [0.0, 0.0, 0.0, 1.0]
        }
        EDGE_LABEL_ENCODING = {
            "EDGE_REG": [1.0, 0.0], "EDGE_INV": [0.0, 1.0]
        }
    aig_cfg = FallbackAigCfg()

from .abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# --- AIG Dataset Class (Using Standard Dataset) ---

class AIGDataset(Dataset):
    """
    Dataset class for AIG graphs, loading individually preprocessed .pt files.
    Assumes a one-time preprocessing script has converted raw pickles into:
        <root>/processed/train/graph_*.pt
        <root>/processed/val/graph_*.pt
        <root>/processed/test/graph_*.pt
    """
    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

        # Determine the directory containing the processed files for this split
        self.processed_split_dir = os.path.join(self.processed_dir, self.split)

        if not os.path.exists(self.processed_split_dir):
            raise RuntimeError(
                f"Processed directory for split '{self.split}' not found at "
                f"'{self.processed_split_dir}'. "
                f"Please run the one-time preprocessing script first.")

        # List all processed graph files for this split and sort them naturally
        self._processed_file_names = sorted(
            [f for f in os.listdir(self.processed_split_dir) if f.startswith('graph_') and f.endswith('.pt')],
            key=lambda x: int(re.search(r'\d+', x).group()) # Natural sort by number in filename
        )

        if not self._processed_file_names:
             print(f"WARNING: No processed 'graph_*.pt' files found in {self.processed_split_dir} for split '{self.split}'.")


    @property
    def raw_file_names(self):
        """
        Indicates the raw files this dataset *would* process if it were
        an InMemoryDataset. Not strictly needed for standard Dataset,
        but good practice for potential checks.
        """
        base = "real_aigs"
        total = 6
        if self.split == 'train':
            return [f"{base}_part_{i}_of_{total}.pkl" for i in range(1, 5)]
        elif self.split == 'val':
            return [f"{base}_part_6_of_{total}.pkl"]
        elif self.split == 'test':
            return [f"{base}_part_5_of_{total}.pkl"]
        else:
            return []

    @property
    def processed_file_names(self):
        """
        Returns the list of individual processed .pt files for this split.
        Used by PyG to check if processing is needed (though we handle the check manually).
        """
        return self._processed_file_names

    def len(self):
        """Returns the number of graphs in the dataset (number of processed files)."""
        return len(self._processed_file_names)

    def get(self, idx):
        """Loads and returns a single preprocessed graph by index."""
        if idx >= len(self._processed_file_names):
            raise IndexError("Index out of bounds")
        # Construct the full path to the individual graph file
        file_path = os.path.join(self.processed_split_dir, self._processed_file_names[idx])
        try:
            # Load the Data object from the .pt file
            data = torch.load(file_path)
            return data
        except Exception as e:
            print(f"Error loading processed file {file_path} at index {idx}: {e}")
            # Handle error appropriately, e.g., return None or raise exception
            return None # Or raise an error

    # download() and process() methods are not typically needed for standard Dataset
    # as preprocessing is assumed to be done separately.

# --- AIGDataModule and AIGDatasetInfos remain the same as the previous version ---
# (They will now use the updated AIGDataset class)

class AIGDataModule(AbstractDataModule):
    """DataModule for the AIG dataset using the standard Dataset class."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        print(f"AIGDataModule using root path: {root_path}")

        datasets = {
            'train': AIGDataset(root=root_path, split='train'),
            'val': AIGDataset(root=root_path, split='val'),
            'test': AIGDataset(root=root_path, split='test')
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class AIGDatasetInfos(AbstractDatasetInfos):
    """Provides metadata and distributions for the AIG dataset."""
    def __init__(self, datamodule, cfg):
        self.datamodule = datamodule
        self.name = 'aig'
        self.atom_encoder = {key: i for i, key in enumerate(aig_cfg.NODE_TYPE_KEYS)}
        self.atom_decoder = list(aig_cfg.NODE_TYPE_KEYS)
        self.num_atom_types = len(self.atom_decoder)
        self.edge_attrs = ['NoEdge'] + list(aig_cfg.EDGE_TYPE_KEYS)
        self.num_edge_types = len(self.edge_attrs)
        self.max_n_nodes = aig_cfg.MAX_NODE_COUNT

        print("Computing AIG dataset statistics (using training split)...")
        # Ensure dataloaders are created to access the dataset
        self.datamodule.setup('fit') # Or appropriate stage

        print("Computing node counts...")
        # Need a way to access the dataset object from the datamodule/dataloader
        # This might require adjusting AbstractDataModule or how helpers are called
        # For now, assume direct access for calculation (may need refinement)
        try:
             # Accessing the underlying dataset directly
             train_dataset = self.datamodule.train_dataset
             self.n_nodes = train_dataset.node_counts(max_nodes_possible=self.max_n_nodes + 1)
             print(f"Node count distribution computed (length {len(self.n_nodes)}).")
             self.node_types = train_dataset.node_types()
             print(f"Node type distribution computed: {self.node_types}")
             self.edge_types = train_dataset.edge_counts()
             print(f"Edge type distribution computed: {self.edge_types}")
        except AttributeError:
             print("Warning: Could not compute dataset statistics automatically. Need to implement helper functions in AIGDataset or adjust access.")
             # Initialize with None or defaults if stats computation fails
             self.n_nodes = torch.tensor([1.0]) # Placeholder
             self.node_types = torch.ones(self.num_atom_types) / self.num_atom_types # Placeholder
             self.edge_types = torch.ones(self.num_edge_types) / self.num_edge_types # Placeholder


        super().complete_infos(self.n_nodes, self.node_types)
        print("AIGDatasetInfos initialization complete.")