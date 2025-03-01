import pickle
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import torch.serialization
from torch.utils.data import random_split


class AIGDataset(Dataset):
    def __init__(self, file_path, num_graphs=None, transform=None, pre_transform=None):
        """
        Dataset for And-Inverter Graphs (AIGs) with an option to limit the number of graphs.

        Args:
            file_path (str): Path to the pickle file containing the AIGs.
            num_graphs (int, optional): Number of graphs to load (default: None, loads all).
            transform (callable, optional): A function/transform applied to a `torch_geometric.data.Data` object.
            pre_transform (callable, optional): A function/transform applied before saving data.
        """
        self.file_path = file_path
        self.num_graphs = num_graphs  # How many graphs to load
        self.processed_file = "data.pt"

        super().__init__(root=".", transform=transform, pre_transform=pre_transform)

    @property
    def processed_file_names(self):
        """Required for PyG to avoid NotImplementedError."""
        return [self.processed_file]

    def process(self):
        """Loads the AIGs from the pickle file and converts them to PyTorch Geometric Data objects."""
        with open(self.file_path, 'rb') as f:
            nx_graphs = pickle.load(f)

        # If num_graphs is specified, take only the first `num_graphs` graphs
        if self.num_graphs is not None:
            nx_graphs = nx_graphs[:self.num_graphs]  # ✅ Take only the first `num_graphs` graphs

        data_list = []
        for nx_graph in nx_graphs:
            data = self.convert_to_pyg_data(nx_graph)
            data_list.append(data)

        # ✅ Fix: Use torch.save() instead of self.collate()
        torch.save(data_list, self.processed_paths[0])  # ✅ Now correctly saves the processed dataset

    def convert_to_pyg_data(self, nx_graph):
        node_features = []
        for n in nx_graph.nodes():
            node_type = np.array(nx_graph.nodes[n]['type'], dtype=np.float32)
            node_feature = np.array([nx_graph.nodes[n]['feature']], dtype=np.float32)
            full_feature = np.concatenate([node_type, node_feature])
            node_features.append(full_feature)

        x = torch.tensor(np.array(node_features), dtype=torch.float)

        edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()

        # 🔍 Debug edge attributes
        edge_attr_list = [nx_graph[u][v]['type'] for u, v in nx_graph.edges]
        edge_attr = torch.tensor(np.array(edge_attr_list, dtype=np.float32), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


    def len(self):
        """Returns the number of AIGs in the dataset."""
        with torch.serialization.safe_globals(["torch_geometric.data.Data"]):  # ✅ Fix for PyG Data object
            data_list = torch.load(self.processed_paths[0], weights_only=False)  # ✅ Ensure full data loading
        return len(data_list)

    def get(self, idx):
        """Gets the AIG at the specified index."""
        with torch.serialization.safe_globals(["torch_geometric.data.Data"]):  # ✅ Fix for PyG Data object
            data_list = torch.load(self.processed_paths[0], weights_only=False)  # ✅ Ensure full data loading
        return data_list[idx]

