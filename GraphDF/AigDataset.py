import os.path as osp
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data

class AIGDataset(InMemoryDataset):
    def __init__(self, root, num_max_node=100, max_pis=8):
        self.num_max_node = num_max_node  # Maximum number of nodes
        self.node_type_dim = 4  # Node types: PI, AND, PO, 0 (one-hot)

        #TODO: redo
        self.node_feature_dimension = self.node_type_dim # + 2**max_pis  # Node feature dimension
        super(AIGDataset, self).__init__(root)


        self.process()

    @property
    def raw_file_names(self):
        return ["all_6x6_graphs.pkl"]  # Input file with raw NetworkX graphs

    @property
    def processed_file_names(self):
        return ["data.pt"]  # Output file with processed PyG Data objects

    def process(self):
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_path, "rb") as f:
            all_graphs = pickle.load(f)

        data_list = []
        for graph in all_graphs:
            if len(graph.nodes) > self.num_max_node:
                print(f"Skipping graph with {len(graph.nodes)} nodes (max allowed: {self.num_max_node})")
                continue
            data_list.append(self._process_graph(graph))

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _process_graph(self, graph):
        """Convert a NetworkX graph to PyTorch Geometric Data with padded node features and directed edges."""
        # Step 1: Node Features - Use `attrs['feature']` directly
        node_features = []
        for _, attrs in graph.nodes(data=True):
            #TODO only node type not tts
            node_feature = torch.tensor(attrs['feature'][:4], dtype=torch.float)  # Entire feature vector
            node_features.append(node_feature)

        # Handle empty node list
        if len(node_features) == 0:
            node_features = torch.zeros((self.num_max_node, self.node_feature_dimension), dtype=torch.float)
        else:
            node_features = torch.stack(node_features)  # Shape: (num_nodes_in_graph, node_dim)
            node_features = F.pad(
                node_features, (0, (self.node_feature_dimension - node_features.size(1)), 0, (self.num_max_node - node_features.size(0))), "constant", 0)

        # Step 2: Edge Features - Directed adjacency matrix
        adj_matrix = torch.zeros((3, self.num_max_node, self.num_max_node), dtype=torch.float)
        for u, v, attrs in graph.edges(data=True):
            edge_type = torch.argmax(torch.tensor(attrs['feature'], dtype=torch.long)).item()  # 0 or 1
            if u < self.num_max_node and v < self.num_max_node:  # Bounds check
                adj_matrix[edge_type, u, v] = 1.0  # Real edges

        # Add virtual edge channel (zeros by default)
        adj_matrix[2, :, :] = 0.0

        # Step 3: Create PyTorch Geometric Data object
        data = Data(
            x=node_features,  # Padded node features (num_max_node, node_dim)
            adj=adj_matrix,   # Directed adjacency matrix (2, num_max_node, num_max_node)
            num_nodes=graph.number_of_nodes()
        )
        return data


