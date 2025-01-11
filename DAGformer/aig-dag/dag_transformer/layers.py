# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from .utils import pad_batch, pad_batch2, unpad_batch
from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES


class Attention(gnn.MessagePassing):
    """Multi-head DAG attention using PyG interface."""

    def __init__(self, embed_dim, num_heads=8, dropout=0., batch_first=False, bias=False,
                 gnn_type="gcn", **kwargs):
        super().__init__(node_dim=0, aggr='add')
        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.gnn_type = gnn_type
        self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type, **kwargs)
        self.attend = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        # Layers for Q, K, V
        self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_tqk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)

        # Structure extractor for SAT
        self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type, **kwargs)

        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, ptr=None, return_attn=False):
        """
        Forward pass for multi-head DAG attention.

        Args:
            x: Node features [num_nodes, embed_dim].
            SAT: Whether to use structure extraction.
            edge_index: Edge index for message passing.
            mask_dag_: DAG mask for attention.
            dag_rr_edge_index: Reversed edge index for DAG attention.
            edge_attr: Edge features (optional).
            ptr: Batch-wise indices (optional).
            return_attn: Whether to return attention weights.

        Returns:
            Updated node features, and optionally attention weights.
        """

        v = self.to_v(x)

        if SAT:
            x_struct = self.structure_extractor(x, edge_index, edge_attr)
            x = x + x_struct  # Combine original features with structural features

        # Compute Q, K, V
        qk = self.to_qk(x).chunk(2, dim=-1)

        attn = None
        if dag_rr_edge_index is not None:
            # print(dag_rr_edge_index.shape)
            out = self.propagate(dag_rr_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                                 return_attn=return_attn)
            if return_attn:
                attn = self._attn
                self._attn = None
                attn = torch.sparse_coo_tensor(
                    dag_rr_edge_index,
                    attn,
                ).to_dense().transpose(0, 1)
            out = rearrange(out, 'n h d -> n (h d)')
        else:
            out, attn = self.self_attn(qk, v, ptr, mask_dag_, return_attn=return_attn)
        return self.out_proj(out), attn

    def self_attn(self, qk, v, ptr, mask_dag_, return_attn=False):
        """ Self attention based on mask matrix"""

        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        # DAG mask
        mask_dag_ = mask_dag_.reshape(dots.shape[0], mask_dag_.shape[1], mask_dag_.shape[1])
        mask_dag_ = mask_dag_[:, :dots.shape[2], :dots.shape[3]]
        dots = dots.masked_fill(
            mask_dag_.unsqueeze(1),
            float('-inf'),
        )
        dots = self.attend(dots)
        dots = self.attn_dropout(dots)
        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None

    def message(self, v_j, q_i, k_j, index, ptr, size_i, edge_attr=None, return_attn=False):
        """Message-passing phase for sparse attention."""
        # Compute attention scores
        attn = (q_i * k_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn += edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        attn = self.dropout(attn)

        # Weighted sum of values
        if return_attn:
            self._attn = attn
        return v_j * attn.unsqueeze(-1)


class StructureExtractor(nn.Module):
    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):
        """
        StructureExtractor: Extracts structural features using stacked GNN layers.

        Args:
            embed_dim (int): Dimensionality of node embeddings.
            gnn_type (str): Type of GNN layer (e.g., "gcn", "gat").
            num_layers (int): Number of GNN layers.
            batch_norm (bool): Apply batch normalization to the final output.
            concat (bool): Concatenate intermediate layer outputs.
            khopgnn (bool): Apply k-hop aggregation using scatter functions.
            **kwargs: Additional arguments for GNN layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type

        # Build GNN layers
        self.gcn = nn.ModuleList([
            get_simple_gnn_layer(gnn_type, embed_dim, **kwargs)
            for _ in range(num_layers)
        ])

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        self.inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        # Optional BatchNorm after concatenation
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.inner_dim)

        # Final projection layer
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
                subgraph_indicator_index=None, agg="sum", return_intermediate=False):
        """
        Forward pass for structure extraction.

        Args:
            x (Tensor): Node features [num_nodes, embed_dim].
            edge_index (Tensor): Edge indices [2, num_edges].
            edge_attr (Tensor, optional): Edge features [num_edges, ...].
            subgraph_indicator_index (Tensor, optional): Subgraph indices for k-hop aggregation.
            agg (str, optional): Aggregation type ("sum", "mean", "max").
            return_intermediate (bool, optional): If True, return intermediate representations.

        Returns:
            Tensor: Extracted node or subgraph features.
        """
        intermediate_representations = []  # Store intermediate representations
        x_cat = [x]  # Store initial input features

        # Apply GNN layers
        for gcn_layer in self.gcn:
            if self.gnn_type in EDGE_GNN_TYPES and edge_attr is not None:
                x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if return_intermediate:
                intermediate_representations.append(x)
            if self.concat:
                x_cat.append(x)

        # Concatenate intermediate features if required
        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        # k-hop Aggregation (if enabled)
        if self.khopgnn:
            if subgraph_indicator_index is None:
                raise ValueError("subgraph_indicator_index is required for k-hop aggregation.")
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            elif agg == "max":
                x = scatter_max(x, subgraph_indicator_index, dim=0)[0]
            else:
                raise ValueError(f"Invalid aggregation type: {agg}. Use 'sum', 'mean', or 'max'.")
            return x

        # Apply batch normalization (if enabled)
        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        # Final projection
        x = self.out_proj(x)

        if return_intermediate:
            return x, intermediate_representations
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", gnn_type="gcn", **kwargs):
        super().__init__()

        # Attention Layer
        self.self_attn = Attention(
            d_model, nhead, dropout=dropout, bias=False, gnn_type=gnn_type, **kwargs
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.d_model = d_model

        # Feedforward submodule
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, ptr=None, return_attn=False):
        # Attention Layer
        attn_output, attn_weights = self.self_attn(
            x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
            edge_attr=edge_attr, ptr=ptr, return_attn=return_attn
        )
        x = x + self.dropout1(attn_output)  # Residual connection (Add)
        x = self.norm1(x)  # Normalize (Norm)

        # Feedforward submodule
        feedforward_output = self.linear2(
            self.dropout1(self.activation(self.linear1(x)))
        )
        x = x + self.dropout2(feedforward_output)  # Residual connection (Add)
        x = self.norm2(x)  # Normalize (Norm)

        return x

