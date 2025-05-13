import os
import os.path as osp
import pickle
import warnings  # For warnings
import torch
import numpy as np  # For splitting and random permutation
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm  # For progress bars
import pathlib  # For robust path handling
from sklearn.model_selection import train_test_split  # For easy data splitting

# Assuming aig_config.py is in the same directory or accessible in PYTHONPATH
# You MUST create this file (aig_config.py) in src/datasets/
# with the following content (adjust values as needed for your AIGs):
#
# NUM_NODE_FEATURES = 3  # Example: Number of features for your nodes
# NUM_EDGE_FEATURES = 1  # Example: Number of features for your *actual* edges
#
from src.aig_config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES

# Import AbstractDataset classes from DeFoG
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.dataset_utils import DistributionNodes


class AIGDataset(InMemoryDataset):
    """
    Dataset class for AIGs.
    Loads a single pre-processed file (e.g., "aig.pt") containing a list of
    PyTorch Geometric Data objects. It then splits this list into train,
    validation, and test sets, saving them as separate .pt files in the
    processed_dir. This processing step is done only once.
    """
    # Name of your combined raw file (output from your data prep script)
    raw_dataset_filename = "aig.pt"

    # Names for the processed split files
    processed_train_filename = "aig_train_processed.pt"
    processed_val_filename = "aig_val_processed.pt"
    processed_test_filename = "aig_test_processed.pt"

    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None,
                 train_ratio=0.8, val_ratio=0.1, random_seed=42):  # Added ratios and seed
        self.stage = stage  # 'train', 'val', or 'test'
        self.dataset_name = "aig"  # Internal name for this dataset type
        self.num_node_features_config = NUM_NODE_FEATURES
        self.num_edge_features_config = NUM_EDGE_FEATURES

        # Store split parameters
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        current_sum = train_ratio + val_ratio
        if not (0.0 <= current_sum <= 1.0):
            raise ValueError("The sum of train_ratio and val_ratio must be between 0.0 and 1.0.")
        self.test_ratio = 1.0 - current_sum
        self.random_seed = random_seed

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the correct processed split file based on the stage
        if self.stage == 'train':
            data_path = osp.join(self.processed_dir, self.processed_train_filename)
        elif self.stage == 'val':
            data_path = osp.join(self.processed_dir, self.processed_val_filename)
        elif self.stage == 'test':
            data_path = osp.join(self.processed_dir, self.processed_test_filename)
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train', 'val', or 'test'.")

        if not osp.exists(data_path):
            raise FileNotFoundError(
                f"Processed file for stage '{self.stage}' not found at {data_path}. "
                f"This might indicate that the 'process' method hasn't been run or "
                f"the raw file '{self.raw_dataset_filename}' was not found in '{self.raw_dir}'.")
        self.data, self.slices = torch.load(data_path)
        # self.num_graphs = len(self.data.num_nodes) # PyG InMemoryDataset usually handles this via len(self)

    @property
    def raw_file_names(self):
        # This dataset expects one raw file: your combined "aig.pt"
        return [self.raw_dataset_filename]

    @property
    def processed_file_names(self):
        # These are the files that this class will create in its process() method.
        # If these files exist, process() will be skipped by InMemoryDataset.
        # For loading, __init__ picks the correct file based on self.stage.
        return [self.processed_train_filename, self.processed_val_filename, self.processed_test_filename]

    def download(self):
        # This method is typically for downloading data if it doesn't exist.
        # In your case, it serves to check if your pre-made "aig.pt" is in the self.raw_dir.
        raw_path = osp.join(self.raw_dir, self.raw_dataset_filename)
        if not osp.exists(raw_path):
            raise FileNotFoundError(
                f"Your raw data file '{self.raw_dataset_filename}' was not found in the raw directory: {self.raw_dir}. "
                "Please ensure your 'aig.pt' (containing a list of PyTorch Geometric Data objects) "
                "is placed there before running the dataset initialization."
            )
        print(f"Raw data file '{self.raw_dataset_filename}' found at: {raw_path}")

    def process(self):
        # This method is called by InMemoryDataset if the files listed in
        # self.processed_file_names do not all exist in self.processed_dir.
        # It loads the combined raw data, splits it, and saves the individual splits.

        print(f"Processing raw data from '{self.raw_dataset_filename}' to create train/val/test splits...")
        raw_path = osp.join(self.raw_dir, self.raw_dataset_filename)

        # Load the list of PyG Data objects.
        # Ensure 'aig.pt' contains a list of torch_geometric.data.Data objects.
        all_pyg_data_objects = torch.load(raw_path)

        if not isinstance(all_pyg_data_objects, list) or \
                (len(all_pyg_data_objects) > 0 and not isinstance(all_pyg_data_objects[0], Data)):
            raise TypeError(
                f"The raw file '{raw_path}' is expected to contain a list of "
                f"PyTorch Geometric Data objects. Found: {type(all_pyg_data_objects)}"
            )
        print(f"Loaded {len(all_pyg_data_objects)} PyG Data objects from the raw file.")

        # Split the data
        # First, separate out the training set
        train_data, remaining_data = train_test_split(
            all_pyg_data_objects,
            train_size=self.train_ratio,
            random_state=self.random_seed,
            shuffle=True  # Shuffle once before all splits
        )

        # Then, split the remainder into validation and test sets
        if self.val_ratio + self.test_ratio > 0 and len(remaining_data) > 0:
            # Calculate val_ratio relative to the size of remaining_data
            # to ensure the original val_ratio (of total) is approximately met.
            # This handles cases where test_ratio might be 0.
            if np.isclose(self.val_ratio + self.test_ratio,
                          0.0):  # Should not happen due to earlier check, but defensive
                val_relative_ratio = 0.0
            else:
                val_relative_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)

            if np.isclose(val_relative_ratio, 1.0):  # If only validation set from remaining
                val_data = remaining_data
                test_data = []
            elif np.isclose(val_relative_ratio, 0.0):  # If only test set from remaining
                val_data = []
                test_data = remaining_data
            else:
                val_data, test_data = train_test_split(
                    remaining_data,
                    train_size=val_relative_ratio,
                    random_state=self.random_seed  # Use same seed for consistent split of remainder
                )
        else:  # If no validation or test set needed from remainder
            val_data = []
            test_data = remaining_data if self.test_ratio > 0 else []

        del all_pyg_data_objects, remaining_data  # Free memory
        gc.collect()

        print(f"Data split into: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test graphs.")

        # Apply pre_transform and pre_filter if they are defined
        # This is standard PyG InMemoryDataset practice
        if self.pre_filter is not None:
            train_data = [d for d in train_data if self.pre_filter(d)]
            val_data = [d for d in val_data if self.pre_filter(d)]
            test_data = [d for d in test_data if self.pre_filter(d)]
            print("Applied pre_filter.")

        if self.pre_transform is not None:
            print("Applying pre_transform (this can take a while)...")
            train_data = [self.pre_transform(d) for d in tqdm(train_data, desc="Pre-transforming train")]
            val_data = [self.pre_transform(d) for d in tqdm(val_data, desc="Pre-transforming val")]
            test_data = [self.pre_transform(d) for d in tqdm(test_data, desc="Pre-transforming test")]
            print("Finished applying pre_transform.")

        # Save the processed splits
        torch.save(self.collate(train_data), osp.join(self.processed_dir, self.processed_train_filename))
        print(f"Saved processed training data to: {osp.join(self.processed_dir, self.processed_train_filename)}")
        torch.save(self.collate(val_data), osp.join(self.processed_dir, self.processed_val_filename))
        print(f"Saved processed validation data to: {osp.join(self.processed_dir, self.processed_val_filename)}")
        torch.save(self.collate(test_data), osp.join(self.processed_dir, self.processed_test_filename))
        print(f"Saved processed test data to: {osp.join(self.processed_dir, self.processed_test_filename)}")


class AIGDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg.dataset.name
        self.datadir = cfg.dataset.datadir

        # Determine the root path for PyG InMemoryDataset.
        # This 'root' directory will contain 'raw' and 'processed' subdirectories.
        current_working_dir = pathlib.Path.cwd()
        root_path_for_pyg = str(current_working_dir / self.datadir)

        # Ensure the raw directory exists inside the root, as InMemoryDataset expects it
        # Your combined 'aig.pt' should be placed in this raw_dir by the user.
        os.makedirs(osp.join(root_path_for_pyg, 'raw'), exist_ok=True)
        # The processed directory will be created by InMemoryDataset if it doesn't exist.

        print(f"AIGDataModule initialized. Data root: {root_path_for_pyg}")
        print(f"  Raw data expected in: {osp.join(root_path_for_pyg, 'raw')}")
        print(f"  Processed data will be in: {osp.join(root_path_for_pyg, 'processed')}")

        transform = None  # Add any PyG transforms if needed (applied on data access)

        # Get split ratios and seed from config, with defaults
        train_ratio = getattr(cfg.dataset, "train_ratio", 0.8)
        val_ratio = getattr(cfg.dataset, "val_ratio", 0.1)
        # test_ratio is inferred by AIGDataset
        random_seed = getattr(cfg.train, "seed", 42)

        datasets = {
            "train": AIGDataset(stage="train", root=root_path_for_pyg, transform=transform,
                                train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed),
            "val": AIGDataset(stage="val", root=root_path_for_pyg, transform=transform,
                              train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed),
            "test": AIGDataset(stage="test", root=root_path_for_pyg, transform=transform,
                               train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed),
        }

        super().__init__(cfg, datasets)
        self.inner = datasets["train"]


class AIGDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: AIGDataModule, cfg):
        self.datamodule = datamodule
        self.dataset_name = cfg.dataset.name
        self.num_node_types = NUM_NODE_FEATURES  # From your aig_config
        self.num_edge_types = NUM_EDGE_FEATURES  # Actual edge types from aig_config

        print(f"Initializing AIGDatasetInfos for '{self.dataset_name}'...")
        # These methods compute statistics based on the loaded (and split) data.
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()  # Includes "no edge" type

        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None and len(self.n_nodes) > 0 else 0

        print(f"  Dataset Name: {self.dataset_name}")
        print(f"  Configured NUM_NODE_FEATURES: {self.num_node_types}")
        print(f"  Configured NUM_EDGE_FEATURES (actual types): {self.num_edge_types}")
        print(f"  Computed max number of nodes in dataset: {self.max_n_nodes}")
        print(
            f"  Computed node counts distribution (shape): {self.n_nodes.shape if self.n_nodes is not None else 'N/A'}")
        print(
            f"  Computed node types distribution (shape): {self.node_types.shape if self.node_types is not None else 'N/A'}")
        print(
            f"  Computed edge types distribution (incl. no-edge) (shape): {self.edge_types.shape if self.edge_types is not None else 'N/A'}")

        super().complete_infos(self.n_nodes, self.node_types)

        # These might not be relevant for AIGs or need custom calculation
        self.valencies = None
        self.atom_weights = None
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2) if self.max_n_nodes > 0 else torch.zeros(1)
        self.atom_decoder = [f"aig_node_type_{i}" for i in range(self.num_node_types)]
        print("AIGDatasetInfos initialization complete.")


if __name__ == '__main__':
    warnings.warn("Running aig_dataset.py directly for testing purposes.")

    # --- Create dummy aig_config.py for testing ---
    print("Creating dummy 'aig_config.py' for the test...")
    with open("aig_config.py", "w") as f:
        f.write("NUM_NODE_FEATURES = 3\n")  # Example
        f.write("NUM_EDGE_FEATURES = 1\n")  # Example: 1 actual edge type
    print("Dummy 'aig_config.py' created.")
    # Re-import to ensure the test uses these values if already imported.
    import importlib
    import aig_config

    importlib.reload(aig_config)
    from aig_config import NUM_NODE_FEATURES as NNF_TEST, NUM_EDGE_FEATURES as NEF_TEST

    print(f"Test using NUM_NODE_FEATURES={NNF_TEST}, NUM_EDGE_FEATURES={NEF_TEST}")


    # --- Dummy PyG Data creation function (simplified) ---
    def create_test_pyg_data():
        num_nodes = np.random.randint(5, 10)
        # Ensure x is one-hot
        x_indices = torch.randint(0, NNF_TEST, (num_nodes,))
        x = torch.nn.functional.one_hot(x_indices, num_classes=NNF_TEST).float()

        edge_src, edge_tgt, edge_attr_list = [], [], []
        if num_nodes > 1:
            num_edges = np.random.randint(num_nodes - 1, num_nodes * 2 if num_nodes > 1 else 1)
            for _ in range(num_edges):
                u, v = np.random.choice(num_nodes, 2, replace=False)
                edge_src.append(u)
                edge_tgt.append(v)
                # Ensure edge_attr is one-hot for actual edge types
                attr_indices = torch.randint(0, NEF_TEST, (1,))
                attr = torch.nn.functional.one_hot(attr_indices, num_classes=NEF_TEST).float().squeeze(0)
                edge_attr_list.append(attr)

        edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        edge_attr = torch.stack(edge_attr_list) if edge_attr_list else torch.empty((0, NEF_TEST), dtype=torch.float)

        # Add dummy y as DeFoG's AbstractDatasetInfos.compute_input_output_dims expects it
        y_tensor = torch.zeros((1, 0), dtype=torch.float)  # For graph-level properties if any
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor, num_nodes=torch.tensor(num_nodes))


    # --- Configuration for the test ---
    from omegaconf import DictConfig

    test_cfg = DictConfig({
        'dataset': {
            'name': 'aig',
            'datadir': 'temp_aig_data_test',  # Use a temporary directory for testing
            'train_ratio': 0.02,  # Very small for quick test
            'val_ratio': 0.01,
        },
        'train': {'batch_size': 2, 'num_workers': 0, 'seed': 123},
        'general': {'name': 'aig_direct_test'}
    })

    # --- Setup test directories ---
    test_root_dir = pathlib.Path(test_cfg.dataset.datadir).resolve()
    test_raw_dir = test_root_dir / 'raw'
    test_processed_dir = test_root_dir / 'processed'

    os.makedirs(test_raw_dir, exist_ok=True)
    os.makedirs(test_processed_dir, exist_ok=True)
    print(f"Test data will use root: {test_root_dir}")

    # --- Create dummy combined 'aig.pt' raw file ---
    print("Creating dummy raw 'aig.pt' file...")
    dummy_all_pyg_graphs = [create_test_pyg_data() for _ in range(100)]  # 100 dummy graphs
    raw_pt_path = test_raw_dir / AIGDataset.raw_dataset_filename
    torch.save(dummy_all_pyg_graphs, raw_pt_path)
    print(f"Saved {len(dummy_all_pyg_graphs)} dummy PyG graphs to {raw_pt_path}")

    # --- Clean up any old processed files to force reprocessing ---
    for f_name in [AIGDataset.processed_train_filename, AIGDataset.processed_val_filename,
                   AIGDataset.processed_test_filename]:
        if osp.exists(osp.join(test_processed_dir, f_name)):
            os.remove(osp.join(test_processed_dir, f_name))
    print(f"Cleaned old processed files from {test_processed_dir} (if any).")

    # --- Test the dataset classes ---
    try:
        print("\nInstantiating AIGDataModule (this will trigger AIGDataset.process)...")
        aig_dm_test = AIGDataModule(test_cfg)

        print("\nInstantiating AIGDatasetInfos...")
        aig_di_test = AIGDatasetInfos(aig_dm_test, test_cfg)

        print("\nChecking dataloaders accessibility...")
        train_dl = aig_dm_test.train_dataloader()
        val_dl = aig_dm_test.val_dataloader()
        test_dl = aig_dm_test.test_dataloader()
        print(f"  Num train batches: {len(train_dl)}")
        print(f"  Num val batches: {len(val_dl)}")
        print(f"  Num test batches: {len(test_dl)}")

        if len(train_dl) > 0:
            sample_batch_item = next(iter(train_dl))
            print("\nSample batch from training data:")
            print(sample_batch_item)
            print(f"  Batch x shape: {sample_batch_item.x.shape}, dtype: {sample_batch_item.x.dtype}")
            print(
                f"  Batch edge_attr shape: {sample_batch_item.edge_attr.shape}, dtype: {sample_batch_item.edge_attr.dtype}")
            print(f"  Batch y shape: {sample_batch_item.y.shape}")
        else:
            warnings.warn("  Training dataloader is empty. Check split ratios or raw data count.")

        print("\n--- Test Run Finished Successfully ---")

    except Exception as e_test:
        print(f"\n--- TEST RUN FAILED ---")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up dummy config and data directory
        if osp.exists("aig_config.py"):
            os.remove("aig_config.py")
        import shutil

        if osp.exists(test_root_dir):
            shutil.rmtree(test_root_dir)
        print("Cleaned up dummy 'aig_config.py' and test data directory.")