# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric.nn as gnn
from .layers import TransformerEncoderLayer
from einops import repeat
import math
import numpy as np

from torch_geometric.utils import *


class GraphTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)


    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, ptr=None, return_attn=False):
        output = x
        for layer in self.layers:
            output = layer(
                output,
                SAT,
                edge_index,
                mask_dag_,
                dag_rr_edge_index,
                edge_attr=edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        return self.norm(output)



from torch_geometric.nn.encoding import PositionalEncoding


class GraphTransformer(nn.Module):
    def __init__(self, in_size, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, gnn_type="gcn", use_edge_attr=True,
                 num_edge_features=2,
                 use_global_pool=True, global_pool='mean', SAT=True, **kwargs):
        super().__init__()

        self.SAT = SAT
        self.dropout = nn.Dropout(0.1)

        # Input Embedding
        self.embedding = nn.Linear(in_features=in_size + 1, out_features=d_model, bias=False)

        # Sinusoidal Positional Encoding (from PyTorch Geometric)
        self.positional_encoding = PositionalEncoding(out_channels=d_model)

        # Edge Embedding
        self.use_edge_attr = use_edge_attr
        self.embedding_edge = nn.Linear(in_features=num_edge_features, out_features=int(d_model / 2), bias=False)

        # Transformer Encoder
        self.encoder = GraphTransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward, dropout,
                batch_norm=batch_norm, gnn_type=gnn_type, **kwargs
            ),
            num_layers=num_layers
        )
        # Global Pooling
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.shared_output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 256)  # Fixed maximum output size
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        mask_dag_ = data.mask_rc if hasattr(data, 'mask_rc') else None
        dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None

        # Concatenate node depth as a new feature
        x = torch.cat([x, node_depth.view(-1, 1)], dim=-1)

        # Apply input embedding
        output = self.embedding(x)

        # Add positional encoding using PyTorch Geometric
        if node_depth is not None:
            pos_enc = self.positional_encoding(node_depth.view(-1, 1).float())
            output = output + pos_enc

        output = self.dropout(output)

        # Edge Embedding
        edge_attr = self.embedding_edge(edge_attr)

        # Transformer Encoder
        output = self.encoder(
            output,
            self.SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=data.ptr
        )

        # Use the batch mask to select output node embeddings
        output_node_embeddings = output[data.output_node_mask]  # Shape: [num_output_nodes, d_model]
        output_node_batch = data.batch[data.output_node_mask]  # Batch assignment for output nodes

        # Initialize a list to hold predictions for all graphs
        all_preds = []

        # Iterate over graphs in the batch
        for graph_idx in range(data.num_graphs):
            # Select output nodes for the current graph
            graph_output_mask = output_node_batch == graph_idx
            graph_output_nodes = output_node_embeddings[graph_output_mask]  # Shape: [num_outs, d_model]

            # Determine truth table size for the current graph
            num_outs = data.num_outs[graph_idx]
            num_inputs = data.num_ins[graph_idx]
            output_size = 2 ** num_inputs  # Truth table size for this graph

            # Pass through the shared output head
            graph_output = self.shared_output_head(graph_output_nodes)  # Shape: [num_outs, 256]

            # Append the predictions for this graph to the list
            all_preds.append(graph_output)

        # Concatenate all predictions along the first dimension (output nodes)
        final_preds = torch.cat(all_preds, dim=0)  # Shape: [total_output_nodes, max_output_size]

        return final_preds

# class GraphTransformer(nn.Module):
#     def __init__(self, in_size, num_class, d_model, num_heads=8,
#                  dim_feedforward=512, dropout=0.0, num_layers=4,
#                  batch_norm=False,
#                  gnn_type="gcn", use_edge_attr=False, num_edge_features=4,
#                  in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
#                  global_pool='mean', SAT=False, **kwargs):
#         super().__init__()
#
#         self.SAT = SAT
#         self.dropout = nn.Dropout(0.1)
#         if in_embed:
#             if isinstance(in_size, int):
#                 self.embedding = nn.Embedding(in_size, d_model)
#             elif isinstance(in_size, nn.Module):
#                 self.embedding = in_size
#             else:
#                 raise ValueError("Not implemented!")
#         else:
#             self.embedding = nn.Linear(in_features=in_size,
#                                        out_features=d_model,
#                                        bias=False)
#
#         self.use_edge_attr = use_edge_attr
#         if use_edge_attr:
#             edge_dim = kwargs.get('edge_dim', 32)
#             if edge_embed:
#                 if isinstance(num_edge_features, int):
#                     self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
#                 else:
#                     raise ValueError("Not implemented!")
#             else:
#                 self.embedding_edge = nn.Linear(in_features=num_edge_features,
#                     out_features=edge_dim, bias=False)
#         else:
#             kwargs['edge_dim'] = None
#
#         self.gnn_type = gnn_type
#         encoder_layer = TransformerEncoderLayer(
#             d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
#             gnn_type=gnn_type, **kwargs)
#         self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
#         self.global_pool = global_pool
#         if global_pool == 'mean':
#             self.pooling = gnn.global_mean_pool
#         elif global_pool == 'add':
#             self.pooling = gnn.global_add_pool
#         elif global_pool == 'cls':
#             self.cls_token = nn.Parameter(torch.randn(1, d_model))
#             self.pooling = None
#         self.use_global_pool = use_global_pool
#
#         self.max_seq_len = max_seq_len
#         if max_seq_len is None:
#             self.classifier = nn.Sequential(
#                 nn.Linear(d_model, d_model),
#                 nn.ReLU(True),
#                 nn.Linear(d_model, num_class)
#             )
#         else:
#             self.classifier = nn.ModuleList()
#             for i in range(max_seq_len):
#                 self.classifier.append(nn.Linear(d_model, num_class))
#
#     def forward(self, data, return_attn=False):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         node_depth = data.node_depth if hasattr(data, "node_depth") else None
#         mask_dag_ = data.mask_rc if hasattr(data, 'mask_rc') else None
#         dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None
#         output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
#         output = self.dropout(output)
#
#         if self.use_edge_attr and edge_attr is not None:
#             edge_attr = self.embedding_edge(edge_attr)
#         else:
#             edge_attr = None
#
#         output = self.encoder(
#             output,
#             self.SAT,
#             edge_index,
#             mask_dag_,
#             dag_rr_edge_index,
#             edge_attr=edge_attr,
#             ptr=data.ptr,
#             return_attn=return_attn
#         )
#         # readout step
#         if self.use_global_pool:
#             output = self.pooling(output, data.batch)
#         if self.max_seq_len is not None:
#             pred_list = []
#             for i in range(self.max_seq_len):
#                 pred_list.append(self.classifier[i](output))
#             return pred_list
#         return self.classifier(output)
