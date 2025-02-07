import os
import pickle
import pathlib as pl
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import networkx as nx
from torch_geometric.data import Data, Batch

class AIGDataset(Dataset):
    def __init__(self, root='./', conf_dict=None):
        """
        Dataset for And-Inverter Graphs (AIGs).

        Args:
            root (str): Root directory of the dataset.
            conf_dict (dict): Configuration dictionary for dataset parameters.
        """
        # Default configuration if none provided
        self.conf_dict = conf_dict or {
            "num_max_nodes": 120,  # Maximum number of nodes per graph
            "node_dim": 4,         # Node feature dimension (e.g. one-hot encoded)
            "edge_dim": 2,         # Edge feature dimension (e.g. one-hot encoded edge labels)
            "raw_file_name": "condition_graphs_0.pkl",
            "train_split_ratio": 0.8,  # Train-validation split ratio
            "dataset_size": 10000
        }

        self.num_max_nodes = self.conf_dict["num_max_nodes"]
        self.node_dim = self.conf_dict["node_dim"]
        self.edge_dim = self.conf_dict["edge_dim"]
        # Placeholder for edge_unroll (if you later wish to compute maximum fanout, etc.)
        self.edge_unroll = None
        self.raw_file_name = self.conf_dict["raw_file_name"]
        self.train_split_ratio = self.conf_dict["train_split_ratio"]
        self.dataset_size = self.conf_dict["dataset_size"]

        self.root = root
        self.raw_path = pl.PurePath(self.root, self.raw_file_name)

        # Process and store the dataset
        self.data_list = self.process()

    def process(self):
        """
        Processes the dataset by loading raw graphs and ensuring consistent padding
        for node features, adjacency matrices, and truth tables (conditions).

        Returns:
            list: A list of processed PyTorch Geometric Data objects.
        """


        data_list = []
        with open(self.raw_path, "rb") as f:
            all_graphs = pickle.load(f)  # Load raw graphs (NetworkX format)

            # Ensure we only process 10,000 random graphs out of 20,000
            if len(all_graphs) > self.dataset_size:
                all_graphs = random.sample(all_graphs, self.dataset_size)

            for nx_graph in all_graphs:
                data_list.append(self._process_graph(nx_graph))

        print(f"Processed {len(data_list)} graphs.")
        return data_list

    def _process_graph(self, nx_graph):
        """
        Convert a single NetworkX graph to a PyTorch Geometric Data object.
        This function:
          - Pads node features to (num_max_nodes, node_dim)
          - Constructs an adjacency tensor of shape (edge_dim, num_max_nodes, num_max_nodes)
          - Processes the truth table (condition) and pads it to length num_max_nodes
          - Retrieves input and output counts from the graph attributes

        Args:
            nx_graph (networkx.Graph): A NetworkX graph object.

        Returns:
            Data: PyTorch Geometric Data object with attributes:
                - x: Node features tensor (num_max_nodes, node_dim)
                - adj: Adjacency tensor (edge_dim, num_max_nodes, num_max_nodes)
                - tts: Truth table / condition vector (num_max_nodes,)
                - input_count: Number of primary inputs (as a tensor)
                - output_count: Number of primary outputs (as a tensor)
                - num_nodes: Original number of nodes in the graph (for reference)
        """
        # ----- Process Node Features -----
        node_features = torch.zeros((self.num_max_nodes, self.node_dim), dtype=torch.float)
        nodes = list(nx_graph.nodes)
        num_nodes_in_graph = len(nodes)
        for i, n in enumerate(nodes):
            if i >= self.num_max_nodes:
                break
            # Retrieve node feature (default to a zero vector if missing)
            feature = nx_graph.nodes[n].get("feature", [0] * self.node_dim)
            node_features[i] = torch.tensor(feature, dtype=torch.float)

        # ----- Process Edge Features (Adjacency Matrix) -----
        # The adjacency tensor is of shape (edge_dim, num_max_nodes, num_max_nodes)
        adj_matrix = torch.zeros((self.edge_dim, self.num_max_nodes, self.num_max_nodes), dtype=torch.float)
        for u, v, attrs in nx_graph.edges(data=True):
            # Retrieve the edge feature (expected as a one-hot list)
            edge_feat = attrs.get("feature", [0] * self.edge_dim)
            # Determine the edge type (index of the 1 in the one-hot vector)
            edge_type = int(torch.argmax(torch.tensor(edge_feat, dtype=torch.float)).item())
            if u < self.num_max_nodes and v < self.num_max_nodes:
                adj_matrix[edge_type, u, v] = 1.0

        # ----- Process Truth Table / Condition (tts) -----
        # Retrieve the truth table stored in the graph attributes.
        # If not provided, default to an all-zero vector.
        condition = nx_graph.graph.get("tts", [])
        # Ensure that the condition is padded (or truncated) to exactly num_max_nodes entries
        if len(condition) < self.num_max_nodes:
            condition = condition + [0.0] * (self.num_max_nodes - len(condition))
        else:
            condition = condition[:self.num_max_nodes]
        tts_tensor = torch.tensor(condition, dtype=torch.float)

        # ----- Process Input and Output Counts -----
        # These values should be stored in the graph attributes (e.g., by your preprocessing function)
        input_count = torch.tensor([nx_graph.graph.get("inputs", 0)], dtype=torch.long)
        output_count = torch.tensor([nx_graph.graph.get("outputs", 0)], dtype=torch.long)

        # ----- Create and Return the Data Object -----
        data = Data(
            x=node_features,              # (num_max_nodes, node_dim)
            adj=adj_matrix,               # (edge_dim, num_max_nodes, num_max_nodes)
            tts=tts_tensor,               # Truth table / condition vector (num_max_nodes,)
            num_nodes=num_nodes_in_graph, # Original number of nodes in the graph
            input_count=input_count,      # Number of primary inputs (PI)
            output_count=output_count     # Number of primary outputs (PO)
        )
        return data

    def get_split_idx(self):
        """
        Generate train-validation split indices dynamically.

        Returns:
            dict: A dictionary with keys 'train_idx' and 'valid_idx'.
        """
        total_graphs = len(self.data_list)
        indices = np.arange(total_graphs)
        np.random.shuffle(indices)
        train_size = int(total_graphs * self.train_split_ratio)
        train_idx = indices[:train_size]
        valid_idx = indices[train_size:]
        return {
            'train_idx': torch.tensor(train_idx, dtype=torch.long),
            'valid_idx': torch.tensor(valid_idx, dtype=torch.long)
        }

    def __len__(self):
        """
        Return the total number of graphs in the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieve a single data object by index.

        Args:
            idx (int): Index of the data object.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        return self.data_list[idx]

    def collate(self, data_list):
        """
        Combines a list of PyTorch Geometric Data objects into a single batch.

        Args:
            data_list (list): List of Data objects.

        Returns:
            Batch: A batch of data.
        """
        return Batch.from_data_list(data_list)
