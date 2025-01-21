import os
import pathlib as pl
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch

# Node type encoding
node_type_encoding = {
    "PI": [1, 0, 0, 0],  # Primary Input
    "AND": [0, 1, 0, 0],  # AND Gate
    "PO": [0, 0, 1, 0],  # Primary Output
    "0": [0, 0, 0, 1],  # Constant Zero
}


class AIGDataset(Dataset):
    def __init__(self, root='./', conf_dict=None):
        """
        Dataset for And-Inverter Graphs (AIGs).

        Args:
            root (str): Root directory of the dataset.
            conf_dict (dict): Configuration dictionary for dataset parameters.
        """
        self.conf_dict = conf_dict or {
            "num_max_nodes": 120,  # Max number of nodes
            "node_dim": 4,  # Node feature dimension
            "edge_dim": 2,  # Number of edge features
            "max_pis": 8,  # Max number of primary inputs
            "edge_unroll": 14,  # Max fanout (edges per node)
            "pad_value": -1,  # Padding value
            "raw_file_name": "more_graphs_depth_pad_tts.pkl",
            "train_split_ratio": 0.8  # Train-validation split ratio
        }

        self.num_max_nodes = self.conf_dict["num_max_nodes"]
        self.node_dim = self.conf_dict["node_dim"]
        self.edge_dim = self.conf_dict["edge_dim"]
        self.max_pis = self.conf_dict["max_pis"]
        self.edge_unroll = self.conf_dict["edge_unroll"]
        self.pad_value = self.conf_dict["pad_value"]
        self.raw_file_name = self.conf_dict["raw_file_name"]
        self.train_split_ratio = self.conf_dict["train_split_ratio"]

        self.root = root
        self.raw_path = pl.PurePath(self.root, self.raw_file_name)

        # Load and process data
        self.data_list = self.process()  # Assign the returned value from `process` to `self.data_list`


    def process(self):
        """
        Processes the dataset by loading raw graphs and ensuring consistent padding
        for truth tables across both dimensions (outputs and entries).
        """
        raw_path = pl.PurePath(self.root, self.raw_file_name)
        with open(raw_path, "rb") as f:
            all_graphs = pickle.load(f)  # Load raw graphs (NetworkX format)

        # Determine max truth table size and max number of outputs across graphs
        max_truth_table_size = 2 ** self.max_pis  # Based on the number of primary inputs
        max_outputs = max(len(graph.graph["tts"]) for graph in all_graphs)  # Maximum number of outputs

        print(f"Padding truth tables to {max_truth_table_size} columns and {max_outputs} rows.")

        data_list = []
        for nx_graph in all_graphs:
            # Extract and pad truth tables
            truth_tables = nx_graph.graph["tts"]
            padded_truth_tables = [
                truth_table + [-np.inf] * (max_truth_table_size - len(truth_table))
                for truth_table in truth_tables
            ]

            # Pad rows to match max_outputs
            while len(padded_truth_tables) < max_outputs:
                padded_truth_tables.append([-np.inf] * max_truth_table_size)

            # Create truth table mask
            truth_table_mask = [
                [1] * len(row[:max_truth_table_size]) + [0] * (max_truth_table_size - len(row[:max_truth_table_size]))
                for row in truth_tables
            ]
            while len(truth_table_mask) < max_outputs:
                truth_table_mask.append([0] * max_truth_table_size)

            # Update graph with padded truth tables and masks
            nx_graph.graph["tts"] = padded_truth_tables
            nx_graph.graph["tts_masks"] = truth_table_mask

            # Process the graph into PyTorch Geometric Data
            data_list.append(self._process_graph(nx_graph))

        print(f"Processed {len(data_list)} graphs.")
        return data_list  # Return the processed data list


    def _process_graph(self, nx_graph):
        """
        Convert a single NetworkX graph to PyTorch Geometric Data.

        Args:
            nx_graph: A NetworkX graph object.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        # Node features
        node_features = torch.zeros((self.num_max_nodes, self.node_dim), dtype=torch.float)
        for i, n in enumerate(nx_graph.nodes):
            if i >= self.num_max_nodes:
                break
            feature = nx_graph.nodes[n]["feature"]
            node_features[i] = torch.tensor(feature, dtype=torch.float)

        # Edge features
        adj_matrix = torch.zeros((self.edge_dim, self.num_max_nodes, self.num_max_nodes), dtype=torch.float)
        for u, v, attrs in nx_graph.edges(data=True):
            edge_type = torch.argmax(torch.tensor(attrs['feature'], dtype=torch.long)).item()  # Edge type (0 or 1)
            if u < self.num_max_nodes and v < self.num_max_nodes:
                adj_matrix[edge_type, u, v] = 1.0

        # Truth tables
        truth_tables = torch.tensor(nx_graph.graph["tts"], dtype=torch.float32)  # Padded truth tables
        truth_table_mask = torch.tensor(nx_graph.graph["tts_masks"], dtype=torch.bool)  # Corresponding mask

        # Output node mask
        output_node_mask = torch.zeros(self.num_max_nodes, dtype=torch.bool)
        for i, n in enumerate(nx_graph.nodes):
            if i >= self.num_max_nodes:
                break
            if nx_graph.nodes[n]["feature"] == node_type_encoding["PO"]:
                output_node_mask[i] = True

        # Create a PyTorch Geometric Data object
        return Data(
            x=node_features,  # Node features (num_max_nodes, node_dim)
            adj=adj_matrix,  # Adjacency tensor (edge_dim, num_max_nodes, num_max_nodes)
            tts=truth_tables,  # Truth tables (padded)
            tts_mask=truth_table_mask,  # Truth table mask
            output_node_mask=output_node_mask,  # Mask for output nodes
            num_nodes=nx_graph.number_of_nodes()  # Original number of nodes
        )

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
        Get a single data object by index.

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
            data_list (list): List of `Data` objects.

        Returns:
            Batch: A batch of data.
        """
        return Batch.from_data_list(data_list)
