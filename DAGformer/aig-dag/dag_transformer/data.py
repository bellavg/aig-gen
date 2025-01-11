import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx, to_undirected
import networkx as nx


import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected
import networkx as nx

node_type_encoding = {
    "PI": [1, 0, 0, 0],  # Primary Input
    "AND": [0, 1, 0, 0],  # AND Gate
    "PO": [0, 0, 1, 0],  # Primary Output
    "0": [0, 0, 0, 1]     # Constant Zero
}

class GraphDataset:
    def __init__(self, dataset, use_mpnn=True, k=1000):
        """
        Dataset for processing AIGs for truth table prediction.

        Args:
            dataset: List of networkx graphs. Each graph has:
                     - Node features stored in `node["feature"]`.
                     - Edge features stored in `graph.edges[edge]["feature"]`.
                     - Truth table stored in `graph.graph["truth_tables"]`.
            use_mpnn: Whether to use MPNN for DAG attention.
            k: Maximum k-hop neighborhood for transitive closure.
        """
        self.dataset = dataset
        self.max_num_nodes = max(graph.number_of_nodes() for graph in dataset)
        self.use_mpnn = use_mpnn
        self.k = k
        self.max_ins = 8

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Load the networkx graph
        nx_graph = self.dataset[index]

        # Convert networkx graph to PyTorch Geometric Data object
        data = from_networkx(nx_graph)

        # Set node features
        data.x = torch.tensor(
            [nx_graph.nodes[n]["feature"] for n in nx_graph.nodes], dtype=torch.float32
        )

        # Add node depth as a separate attribute
        assert all("depth" in nx_graph.nodes[n] for n in nx_graph.nodes), \
            "Missing 'depth' attribute for one or more nodes!"
        data.node_depth = torch.tensor(
            [nx_graph.nodes[n]["depth"] for n in nx_graph.nodes], dtype=torch.long
        )

        # Set edge index (must be integers)
        data.edge_index = torch.tensor(
            list(nx_graph.edges), dtype=torch.long
        ).t().contiguous()

        # Set edge attributes
        data.edge_attr = torch.tensor(
            [nx_graph.edges[e]["feature"] for e in nx_graph.edges], dtype=torch.float32
        )
        assert data.edge_index.shape[1] == data.edge_attr.shape[0], \
            "Mismatch between number of edges in edge_index and edge_attr!"

        # Convert num_outs and num_ins to integers
        data.num_outs = int(nx_graph.graph["outputs"])
        data.num_ins = int(nx_graph.graph["inputs"])

        # Process truth tables (y) and create a mask
        truth_tables = torch.tensor(nx_graph.graph["tts"], dtype=torch.float32)

        # Create a mask to identify valid entries (non -inf)
        mask = truth_tables != -float("inf")
        truth_tables[~mask] = -1e9  # Replace -inf with -1e9 for invalid entries

        # Assign directly (truth tables are already padded)
        data.y = truth_tables
        data.y_mask = mask

        # Create output_node_mask to identify output nodes
        output_nodes = [n for n in nx_graph.nodes if nx_graph.nodes[n]["feature"] == node_type_encoding["PO"]]
        data.output_node_mask = torch.tensor(
            [1 if n in output_nodes else 0 for n in nx_graph.nodes], dtype=torch.bool
        )

        # Ensure the mask is consistent with the number of nodes
        assert data.output_node_mask.shape[0] == data.num_nodes, \
            "Output node mask size mismatch with number of nodes!"

        edge_index_dag = data.edge_index

        if self.use_mpnn:
            # Use MPNN-based DAG attention (reverse edges for bidirectional propagation)
            data.dag_rr_edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
            assert data.dag_rr_edge_index.shape[1] == 2 * data.edge_index.shape[1], \
                "Mismatch between edge_index and dag_rr_edge_index!"
        else:
            # using mask to implement DAG attention
            num_nodes = data.num_nodes
            if (num_nodes <= 120):
                max_num_nodes = 120
                mask_rc = torch.tensor([]).new_zeros(max_num_nodes, max_num_nodes).bool()
                for index1 in range(num_nodes):
                    ne_idx = edge_index_dag[0] == index1
                    le_idx = ne_idx.nonzero(as_tuple=True)[0]
                    lp_edge_index = edge_index_dag[1, le_idx]
                    ne_idx_inverse = edge_index_dag[1] == index1
                    le_idx_inverse = ne_idx_inverse.nonzero(as_tuple=True)[0]
                    lp_edge_index_inverse = edge_index_dag[0, le_idx_inverse]
                    mask_r = torch.tensor([]).new_zeros(max_num_nodes).bool()
                    mask_r[lp_edge_index] = True
                    mask_r[lp_edge_index_inverse] = True
                    mask_rc[index1] = ~mask_r
            data.mask_rc = mask_rc

        return data
