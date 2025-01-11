# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, Linear, ReLU
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from typing import Optional, List, Dict
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree



GNN_TYPES = [
    'graph', 'graphsage', 'gcn',
    'gin', 'gine',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4',
    'rwgnn', 'khopgnn'
]

EDGE_GNN_TYPES = [
    'gine', 'gcn',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4'
]

def create_pna_layer(gnn_type, embed_dim, edge_dim, aggregators=None, scalers=None, **kwargs):
    """
    Creates a PNAConv layer with specified aggregators and scalers.

    Args:
        gnn_type (str): Type of PNA layer (e.g., "pna", "pna2").
        embed_dim (int): Dimensionality of node embeddings.
        edge_dim (int): Dimensionality of edge features.
        aggregators (list, optional): Aggregators for PNAConv.
        scalers (list, optional): Scalers for PNAConv.
        **kwargs: Additional arguments.

    Returns:
        nn.Module: PNAConv layer.
    """
    if aggregators is None:
        aggregators = ['mean', 'min', 'max', 'std']
    if scalers is None:
        scalers = ['identity']

    deg = kwargs.get('deg', None)
    towers = kwargs.get('towers', 4)
    divide_input = kwargs.get('divide_input', True)

    return gnn.PNAConv(
        embed_dim, embed_dim,
        aggregators=aggregators,
        scalers=scalers,
        deg=deg,
        towers=towers,
        pre_layers=1,
        post_layers=1,
        divide_input=divide_input,
        edge_dim=edge_dim
    )



def get_simple_gnn_layer(gnn_type, embed_dim, **kwargs):
    """
    Returns a GNN layer based on the specified type.

    Args:
        gnn_type (str): Type of GNN layer (e.g., "gcn", "graphsage").
        embed_dim (int): Dimensionality of node embeddings.
        **kwargs: Additional arguments (e.g., edge_dim, deg).

    Returns:
        nn.Module: GNN layer.
    """
    edge_dim = kwargs.get('edge_dim', None)

    if gnn_type == "graph":
        return gnn.GraphConv(embed_dim, embed_dim)
    elif gnn_type == "graphsage":
        return gnn.SAGEConv(embed_dim, embed_dim)
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, embed_dim)
        return GCNConv(embed_dim, edge_dim)
    elif gnn_type == "gin":
        mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    elif gnn_type == "gine":
        if edge_dim is None:
            raise ValueError("edge_dim must be provided for GINEConv.")
        mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    elif gnn_type.startswith("pna"):
        return create_pna_layer(gnn_type, embed_dim, edge_dim, **kwargs)
    elif gnn_type == "mpnn":
        return create_pna_layer("mpnn", embed_dim, edge_dim, aggregators=["sum"], scalers=["identity"], **kwargs)
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")




class GCNConv(gnn.MessagePassing):
    def __init__(self, embed_dim, edge_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)


        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = nn.Linear(int(embed_dim/2), embed_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = utils.degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
