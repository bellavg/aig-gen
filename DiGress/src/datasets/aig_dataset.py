import os
import torch
import re # For sorting filenames naturally
from torch_geometric.data import Data, Dataset
import pathlib
import pickle
from tqdm import tqdm

# Assuming your aig_cfg is accessible
try:
    # Adjust this import path if your config is located elsewhere
    import aig_config as aig_cfg
except ImportError:
    from . import aig_config as aig_cfg


# Make sure abstract_dataset is importable
try:
    from .abstract_dataset import AbstractDataModule, AbstractDatasetInfos
except ImportError:
    # Adjust path if aig_dataset.py is not in the same directory as abstract_dataset.py
    from abstract_dataset import AbstractDataModule, AbstractDatasetInfos


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
        try:
            self._processed_file_names = sorted(
                [f for f in os.listdir(self.processed_split_dir) if f.startswith('graph_') and f.endswith('.pt')],
                # Natural sort by number in filename, handles potential errors
                key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else -1
            )
        except FileNotFoundError:
             raise RuntimeError(f"Processed directory not found during file listing: {self.processed_split_dir}")
        except Exception as e:
             raise RuntimeError(f"Error listing processed files in {self.processed_split_dir}: {e}")


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
            raise IndexError(f"Index {idx} out of bounds for dataset split '{self.split}' with length {self.len()}")

        file_path = os.path.join(self.processed_split_dir, self._processed_file_names[idx])
        try:
            # Load the Data object from the .pt file
            # --- FIX: Added weights_only=False ---
            data = torch.load(file_path, weights_only=False)
            # --- End Fix ---

            # Basic check if loaded data is valid
            if not isinstance(data, Data):
                 print(f"Warning: Loaded file {file_path} at index {idx} is not a PyG Data object (type: {type(data)}). Returning None.")
                 return None
            return data
        except EOFError:
             print(f"Error loading processed file {file_path} at index {idx}: File is corrupted or empty (EOFError). Returning None.")
             return None
        except pickle.UnpicklingError as pe:
             print(f"Error loading processed file {file_path} at index {idx}: Unpickling error ({pe}). Returning None.")
             return None
        except Exception as e:
            # Catch other potential errors during loading
            print(f"Error loading processed file {file_path} at index {idx}: {type(e).__name__} - {e}. Returning None.")
            return None

    # --- Add helper methods for statistics calculation ---
    # These methods iterate through the dataset using self.get(idx)
    def node_counts(self, max_nodes_possible=300):
        """Calculates node count distribution by loading graphs individually."""
        print(f"Calculating node counts for split '{self.split}'...")
        all_counts = torch.zeros(max_nodes_possible)
        num_graphs = self.len()
        for i in tqdm(range(num_graphs), desc=f"Node counts ({self.split})", leave=False):
            data = self.get(i)
            if data is not None and hasattr(data, 'num_nodes'):
                count = data.num_nodes
                if count < max_nodes_possible:
                    all_counts[count] += 1
                else:
                    print(f"Warning: Graph {i} has {count} nodes, exceeding max_nodes_possible {max_nodes_possible}. Clamping.")
                    all_counts[max_nodes_possible - 1] += 1 # Clamp to last bin
            elif data is not None:
                 print(f"Warning: Graph {i} missing 'num_nodes' attribute.")
            # else: Error already printed by get()

        # Find the highest index with a non-zero count
        nz = all_counts.nonzero().squeeze()
        max_index = nz.max().item() if nz.numel() > 0 else 0 # Handle empty dataset case
        all_counts = all_counts[:max_index + 1]

        total_sum = all_counts.sum()
        if total_sum > 0:
            all_counts = all_counts / total_sum
        else:
            print(f"Warning: Total node count sum is zero for split '{self.split}'. Returning zeros.")
        return all_counts

    def node_types(self):
        """Calculates node type distribution by loading graphs individually."""
        print(f"Calculating node types for split '{self.split}'...")
        num_classes = None
        temp_data = self.get(0) # Get first graph to determine num_classes
        if temp_data is not None and hasattr(temp_data, 'x'):
            num_classes = temp_data.x.shape[1]
        else:
            print(f"Warning: Could not determine number of node classes for split '{self.split}'. Returning None.")
            return None # Cannot proceed without knowing num_classes

        counts = torch.zeros(num_classes)
        num_graphs = self.len()
        total_nodes = 0
        for i in tqdm(range(num_graphs), desc=f"Node types ({self.split})", leave=False):
            data = self.get(i)
            if data is not None and hasattr(data, 'x'):
                counts += data.x.sum(dim=0)
                total_nodes += data.x.shape[0]
            # else: Error already printed by get()

        if total_nodes > 0:
            counts = counts / total_nodes # Normalize by total nodes, not sum of features
        else:
            print(f"Warning: Total node count is zero for split '{self.split}'. Returning zeros.")
        return counts

    def edge_counts(self):
        """Calculates edge type distribution by loading graphs individually."""
        print(f"Calculating edge types for split '{self.split}'...")
        num_classes = None
        temp_data = self.get(0) # Get first graph to determine num_classes
        if temp_data is not None and hasattr(temp_data, 'edge_attr'):
            num_classes = temp_data.edge_attr.shape[1]
        else:
            print(f"Warning: Could not determine number of edge classes for split '{self.split}'. Returning None.")
            return None

        d = torch.zeros(num_classes, dtype=torch.float)
        total_possible_edges = 0
        total_actual_edges = 0

        num_graphs = self.len()
        for i in tqdm(range(num_graphs), desc=f"Edge types ({self.split})", leave=False):
            data = self.get(i)
            if data is not None and hasattr(data, 'num_nodes') and hasattr(data, 'edge_index') and hasattr(data, 'edge_attr'):
                n = data.num_nodes
                # Calculate possible edges (excluding self-loops) for this graph
                possible_edges_in_graph = n * (n - 1)
                total_possible_edges += possible_edges_in_graph

                num_edges_in_graph = data.edge_index.shape[1]
                total_actual_edges += num_edges_in_graph

                # Sum edge attributes for existing edges
                if num_edges_in_graph > 0:
                    edge_types = data.edge_attr.sum(dim=0)
                    d += edge_types # Accumulate counts for existing edge types
            # else: Error already printed by get()

        # Calculate total non-edges across the dataset
        num_non_edges = total_possible_edges - total_actual_edges
        if num_non_edges < 0:
             print(f"Warning: Calculated negative non-edges ({num_non_edges}). Check logic. Setting to 0.")
             num_non_edges = 0

        # Assign non-edge count to the first class (index 0)
        if num_classes > 0:
             d[0] += num_non_edges
        else:
             print("Warning: num_classes is 0, cannot assign non-edge count.")


        total_sum = d.sum()
        if total_sum > 0:
            d = d / total_sum
        else:
            print(f"Warning: Total edge count sum is zero for split '{self.split}'. Returning zeros.")
        return d


# --- AIGDataModule ---
class AIGDataModule(AbstractDataModule):
    """DataModule for the AIG dataset using the standard Dataset class."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        # Construct the root path relative to the project structure
        # Assumes this script is run from a location where 'data/aig' is accessible
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2] # Adjust if script location changes
        root_path = os.path.join(base_path, self.datadir)
        print(f"AIGDataModule using root path: {root_path}")

        datasets = {
            'train': AIGDataset(root=root_path, split='train'),
            'val': AIGDataset(root=root_path, split='val'),
            'test': AIGDataset(root=root_path, split='test')
        }
        super().__init__(cfg, datasets)
        # self.inner = self.train_dataset # 'inner' might not be necessary

    # __getitem__ is inherited from AbstractDataModule/LightningDataset

# --- AIGDatasetInfos ---
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
        # Ensure dataloaders/datasets are accessible
        # Setup might be needed if accessing directly like this
        # self.datamodule.setup('fit') # Or appropriate stage

        # Access the dataset object directly from the datamodule
        # Note: This relies on the structure of LightningDataset/DataModule
        train_dataset = self.datamodule.train_dataloader().dataset

        print("Computing node counts...")
        self.n_nodes = train_dataset.node_counts(max_nodes_possible=self.max_n_nodes + 1)
        if self.n_nodes is None: raise RuntimeError("Failed to compute node counts.")
        print(f"Node count distribution computed (length {len(self.n_nodes)}).")

        print("Computing node types...")
        self.node_types = train_dataset.node_types()
        if self.node_types is None: raise RuntimeError("Failed to compute node types.")
        print(f"Node type distribution computed: {self.node_types}")

        print("Computing edge types...")
        self.edge_types = train_dataset.edge_counts()
        if self.edge_types is None: raise RuntimeError("Failed to compute edge types.")
        print(f"Edge type distribution computed: {self.edge_types}")

        # Finalize metadata using the computed distributions
        super().complete_infos(self.n_nodes, self.node_types)
        print("AIGDatasetInfos initialization complete.")

