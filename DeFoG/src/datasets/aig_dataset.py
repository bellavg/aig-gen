#!/usr/bin/env python3
import os
import os.path as osp
import pickle
import warnings
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import gc
import pathlib
from sklearn.model_selection import train_test_split

# Assuming aig_config.py is in the same directory or accessible in PYTHONPATH
# Create this file (aig_config.py) in the same directory as aig_dataset.py
# with the following content (adjust values as needed):
#
# NUM_NODE_FEATURES = 3  # Example: Number of features for your nodes
# NUM_EDGE_FEATURES = 1  # Example: Number of features for your actual edges
#
from src.aig_config import NUM_NODE_FEATURES, NUM_EDGE_FEATURES

# Import AbstractDataset classes from DeFoG
# Ensure your PYTHONPATH is set up correctly if these are in a different relative location
# when running this script directly vs. when DeFoG's main.py runs it.
# For DeFoG, these imports should work if aig_dataset.py is in src/datasets/
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.dataset_utils import DistributionNodes  # , load_pickle, save_pickle, to_list (not strictly needed here but good practice)


class AIGDataset(InMemoryDataset):
    """
    Dataset class for AIGs.
    It expects a single raw file (e.g., "aig.pt") containing a list of
    PyTorch Geometric Data objects.
    It processes this file once to create train, validation, and test splits,
    saving them as separate .pt files in the processed_dir.
    """
    # Define train, validation, and test file names for processed PyG Data lists
    processed_train_file = "aig_train_pyg.pt"
    processed_val_file = "aig_val_pyg.pt"
    processed_test_file = "aig_test_pyg.pt"
    # Combined raw file produced by your script
    combined_raw_file_name = "aig.pt"

    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None,
                 train_ratio=0.8, val_ratio=0.1, random_seed=42):
        self.stage = stage
        self.dataset_name = "aig"
        self.num_node_features = NUM_NODE_FEATURES
        self.num_edge_features = NUM_EDGE_FEATURES  # Number of *actual* edge types

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        # Test ratio is inferred as 1.0 - train_ratio - val_ratio
        if not np.isclose(train_ratio + val_ratio, 1.0) and not np.isclose(train_ratio + val_ratio, 0.0) and (
                1.0 - train_ratio - val_ratio < 0):
            raise ValueError("Sum of train_ratio and val_ratio cannot exceed 1.0 if test set is desired.")
        self.test_ratio = 1.0 - train_ratio - val_ratio

        self.random_seed = random_seed

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the appropriate processed split
        if self.stage == 'train':
            data_path = osp.join(self.processed_dir, self.processed_train_file)
        elif self.stage == 'val':
            data_path = osp.join(self.processed_dir, self.processed_val_file)
        elif self.stage == 'test':
            data_path = osp.join(self.processed_dir, self.processed_test_file)
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

        if not osp.exists(data_path):
            raise FileNotFoundError(f"Processed file for stage {self.stage} not found at {data_path}. "
                                    "Ensure process() method has run successfully.")
        self.data, self.slices = torch.load(data_path)

    @property
    def raw_file_names(self):
        return [self.combined_raw_file_name]

    @property
    def processed_file_names(self):
        # This method should ideally return a list of ALL files that process() creates,
        # or dynamically return the one for the current stage.
        # PyG's InMemoryDataset checks if these exist to skip processing.
        # To ensure process() runs only once to create all splits, we check for all of them.
        # However, for loading, __init__ will pick the correct one based on stage.
        if self.stage == 'train':
            return [self.processed_train_file]
        elif self.stage == 'val':
            return [self.processed_val_file]
        elif self.stage == 'test':
            return [self.processed_test_file]
        # Fallback for the check in super().__init__ to see if processing is needed at all.
        # This list tells InMemoryDataset what files to expect from process().
        return [self.processed_train_file, self.processed_val_file, self.processed_test_file]

    def download(self):
        combined_raw_path = osp.join(self.raw_dir, self.combined_raw_file_name)
        if not osp.exists(combined_raw_path):
            raise FileNotFoundError(
                f"Combined raw data file '{self.combined_raw_file_name}' not found in {self.raw_dir}. "
                f"Please ensure your script has generated it and it's placed in the correct raw directory."
            )
        print(f"Found combined raw data file: {combined_raw_path}")

    def process(self):
        # This method processes the single raw "aig.pt" and creates the three split files.
        # It's called if any of the files in `processed_file_names` (when checked by base class) are missing.

        path_train = osp.join(self.processed_dir, self.processed_train_file)
        path_val = osp.join(self.processed_dir, self.processed_val_file)
        path_test = osp.join(self.processed_dir, self.processed_test_file)

        # If all split files already exist, we might not need to reprocess.
        # The InMemoryDataset logic usually handles this, but an explicit check can be clearer.
        if osp.exists(path_train) and osp.exists(path_val) and osp.exists(path_test):
            print("All processed split files already exist. Skipping processing.")
            return

        print(f"Processing raw data to create train/val/test splits...")
        combined_raw_path = osp.join(self.raw_dir, self.combined_raw_file_name)
        all_pyg_data_objects = torch.load(combined_raw_path)

        if not isinstance(all_pyg_data_objects, list) or \
                (len(all_pyg_data_objects) > 0 and not isinstance(all_pyg_data_objects[0], Data)):
            raise TypeError(f"Expected {combined_raw_path} to be a list of PyTorch Geometric Data objects.")

        print(f"Loaded {len(all_pyg_data_objects)} PyG Data objects from {combined_raw_path}.")

        # Shuffle data before splitting
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(all_pyg_data_objects))
        all_pyg_data_objects = [all_pyg_data_objects[i] for i in indices]

        num_total = len(all_pyg_data_objects)
        num_train = int(self.train_ratio * num_total)
        num_val = int(self.val_ratio * num_total)

        train_data = all_pyg_data_objects[:num_train]
        val_data = all_pyg_data_objects[num_train: num_train + num_val]
        test_data = all_pyg_data_objects[num_train + num_val:]

        del all_pyg_data_objects  # Free memory
        gc.collect()

        print(f"Splitting data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")

        # Apply pre_transform and pre_filter if they exist
        if self.pre_filter is not None:
            train_data = [d for d in train_data if self.pre_filter(d)]
            val_data = [d for d in val_data if self.pre_filter(d)]
            test_data = [d for d in test_data if self.pre_filter(d)]

        if self.pre_transform is not None:
            train_data = [self.pre_transform(d) for d in train_data]
            val_data = [self.pre_transform(d) for d in val_data]
            test_data = [self.pre_transform(d) for d in test_data]

        torch.save(self.collate(train_data), path_train)
        print(f"Saved training data to {path_train}")
        torch.save(self.collate(val_data), path_val)
        print(f"Saved validation data to {path_val}")
        torch.save(self.collate(test_data), path_test)
        print(f"Saved test data to {path_test}")


class AIGDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg.dataset.name
        self.datadir = cfg.dataset.datadir

        # Path logic: PyG InMemoryDataset expects 'root'. It creates 'root/raw' and 'root/processed'.
        # Your 'aig.pt' should be in 'root/raw/aig.pt'.
        # The split files will be saved in 'root/processed/'.
        current_working_dir = pathlib.Path.cwd()
        root_path_for_pyg = str(current_working_dir / self.datadir)
        print(f"AIGDataModule: Root path for PyG InMemoryDataset: {root_path_for_pyg}")
        # Ensure raw directory exists within the root_path_for_pyg for AIGDataset
        os.makedirs(osp.join(root_path_for_pyg, 'raw'), exist_ok=True)

        transform = None  # Define any PyG transforms you need for each Data object on access

        train_ratio = getattr(cfg.dataset, "train_ratio", 0.8)
        val_ratio = getattr(cfg.dataset, "val_ratio", 0.1)
        # test_ratio is inferred
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
    def __init__(self, datamodule: AIGDataModule, cfg):  # Added cfg for consistency
        self.datamodule = datamodule
        self.dataset_name = cfg.dataset.name
        self.num_node_types = NUM_NODE_FEATURES
        self.num_edge_types = NUM_EDGE_FEATURES  # This is the number of *actual* edge types

        print(f"Computing dataset statistics for {self.dataset_name}...")
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()  # This will include the implicit "no edge" type

        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None and len(self.n_nodes) > 0 else 0

        print(f"  Number of node features defined in aig_config: {self.num_node_types}")
        print(f"  Number of actual edge features in aig_config: {self.num_edge_types}")
        print(f"  Calculated max number of nodes in dataset: {self.max_n_nodes}")
        print(f"  Calculated node counts distribution shape: {self.n_nodes.shape}")
        print(f"  Calculated node types (features) distribution shape: {self.node_types.shape}")
        print(f"  Calculated edge types (features, incl. no-edge) distribution shape: {self.edge_types.shape}")

        super().complete_infos(self.n_nodes, self.node_types)

        # These might not be directly applicable or need custom computation for AIGs
        self.valencies = None
        self.atom_weights = None  # Not applicable for AIGs
        # Placeholder for valency distribution, can be computed if needed
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2) if self.max_n_nodes > 0 else torch.zeros(1)
        # Define a generic decoder if needed for debugging/visualization
        self.atom_decoder = [f"node_feat_val_{i}" for i in range(self.num_node_types)]

#
# if __name__ == '__main__':
#     print("Example usage of AIGDataset, AIGDataModule, and AIGDatasetInfos.")
#     print("This test expects 'aig_config.py' to exist in the same directory,")
#     print("and a dummy 'aig.pt' (list of PyG Data objects) to be created in './test_data_aig/raw/'.")
#
#     # --- Create dummy aig_config.py for testing ---
#     with open("aig_config.py", "w") as f:
#         f.write("NUM_NODE_FEATURES = 3\n")
#         f.write("NUM_EDGE_FEATURES = 2\n")  # Example: 2 actual edge types
#     print("Created dummy aig_config.py with NUM_NODE_FEATURES=3, NUM_EDGE_FEATURES=2")
#
#     from aig_config import NUM_NODE_FEATURES as NNF_TEST, NUM_EDGE_FEATURES as NEF_TEST
#
#
#     # --- Dummy PyG Data creation function (simplified from your script) ---
#     def create_dummy_pyg_data_for_aig():
#         num_nodes = np.random.randint(5, 15)
#         x = torch.randn(num_nodes, NNF_TEST).softmax(dim=-1)  # Random one-hot like
#         x = torch.nn.functional.one_hot(x.argmax(dim=-1), num_classes=NNF_TEST).float()
#
#         edge_src = []
#         edge_tgt = []
#         edge_attr_list = []
#         num_edges = np.random.randint(num_nodes - 1, num_nodes * 2)
#         for _ in range(num_edges):
#             u, v = np.random.choice(num_nodes, 2, replace=False)
#             edge_src.append(u)
#             edge_tgt.append(v)
#             # Example: Random one-hot edge features for actual edge types
#             attr = torch.zeros(NEF_TEST)
#             if NEF_TEST > 0:
#                 attr[np.random.randint(0, NEF_TEST)] = 1.0
#             edge_attr_list.append(attr)
#
#         edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
#         edge_attr = torch.stack(edge_attr_list) if edge_attr_list else torch.empty((0, NEF_TEST), dtype=torch.float)
#
#         # Add dummy y as DeFoG's AbstractDatasetInfos.compute_input_output_dims expects it
#         y_tensor = torch.zeros((1, 0), dtype=torch.float)
#
#         return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=torch.tensor(num_nodes), y=y_tensor)
#
#
#     # --- Configuration for the test ---
#     from omegaconf import DictConfig
#
#     dummy_cfg_test = DictConfig({
#         'dataset': {
#             'name': 'aig',
#             'datadir': 'test_data_aig',  # Root directory for this test
#             'train_ratio': 0.7,
#             'val_ratio': 0.15,
#             # test_ratio will be 0.15
#         },
#         'train': {
#             'batch_size': 4,
#             'num_workers': 0,
#             'seed': 42
#         },
#         'general': {'name': 'aig_test_run'}
#     })
#
#     # --- Setup test directories ---
#     project_root_test = pathlib.Path(".").resolve()  # current dir for test
#     test_data_dir = project_root_test / dummy_cfg_test.dataset.datadir
#     raw_dir_test = test_data_dir / 'raw'
#     processed_dir_test = test_data_dir / 'processed'
#
#     os.makedirs(raw_dir_test, exist_ok=True)
#     os.makedirs(processed_dir_test, exist_ok=True)
#
#     # --- Create dummy combined 'aig.pt' ---
#     dummy_all_graphs = [create_dummy_pyg_data_for_aig() for _ in range(100)]  # 100 dummy graphs
#     combined_pt_file_path = raw_dir_test / AIGDataset.combined_raw_file_name
#     torch.save(dummy_all_graphs, combined_pt_file_path)
#     print(f"Saved {len(dummy_all_graphs)} dummy PyG graphs to {combined_pt_file_path}")
#
#     # --- Clean up old processed files to ensure process() is called ---
#     for f_name in [AIGDataset.processed_train_file, AIGDataset.processed_val_file, AIGDataset.processed_test_file]:
#         if osp.exists(osp.join(processed_dir_test, f_name)):
#             os.remove(osp.join(processed_dir_test, f_name))
#     print(f"Cleaned old processed files from {processed_dir_test} (if any).")
#
#     # --- Test the dataset classes ---
#     try:
#         print("\nInstantiating AIGDataModule (this will trigger AIGDataset.process if needed)...")
#         aig_datamodule_test = AIGDataModule(dummy_cfg_test)
#
#         print("\nInstantiating AIGDatasetInfos...")
#         aig_infos_test = AIGDatasetInfos(aig_datamodule_test, dummy_cfg_test)
#
#         print("\nChecking dataloaders...")
#         train_loader_test = aig_datamodule_test.train_dataloader()
#         val_loader_test = aig_datamodule_test.val_dataloader()
#         test_loader_test = aig_datamodule_test.test_dataloader()
#
#         print(
#             f"  Num train batches: {len(train_loader_test)}, Num val batches: {len(val_loader_test)}, Num test batches: {len(test_loader_test)}")
#
#         if len(train_loader_test) > 0:
#             sample_batch_test = next(iter(train_loader_test))
#             print("\nSample batch from training data:")
#             print(sample_batch_test)
#             print(f"  Batch x shape: {sample_batch_test.x.shape}")
#             print(f"  Batch edge_index shape: {sample_batch_test.edge_index.shape}")
#             print(f"  Batch edge_attr shape: {sample_batch_test.edge_attr.shape}")
#         else:
#             print("  Train loader is empty!")
#
#         print("\n--- Test Run Finished ---")
#
#     except Exception as e:
#         print(f"\n--- TEST RUN FAILED ---")
#         import traceback
#
#         traceback.print_exc()
#     finally:
#         # Clean up dummy files
#         if osp.exists("aig_config.py"):
#             os.remove("aig_config.py")
#         # import shutil
#         # if osp.exists(test_data_dir):
#         #     shutil.rmtree(test_data_dir)
#         # print("Cleaned up dummy aig_config.py and test_data_aig directory.")