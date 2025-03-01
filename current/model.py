# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn.conv import GraphConv
# from torch_geometric.nn import GAT
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class EdgeAwareGraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=2, num_heads=8, dropout=0.1):
        super(EdgeAwareGraphTransformerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, num_heads)

        # Multi-head attention with edge features
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.ReLU(),
            nn.Linear(4 * out_dim, out_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # Projection if dimensions change
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch=None, mask=None):
        """
        x: Node features [num_nodes, in_dim]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge features [num_edges, edge_dim]
        batch: Batch assignment for nodes [num_nodes]
        mask: Attention mask [num_nodes]
        """
        # Apply layer normalization
        x_norm = self.norm1(x)

        # Project queries, keys, values
        q = self.q_proj(x_norm)  # [num_nodes, out_dim]
        k = self.k_proj(x_norm)  # [num_nodes, out_dim]
        v = self.v_proj(x_norm)  # [num_nodes, out_dim]

        # Project edge features
        edge_weights = self.edge_proj(edge_attr)  # [num_edges, num_heads]

        # Process each graph in the batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        out = torch.zeros_like(v)

        for g in range(num_graphs):
            # Get nodes for this graph
            graph_mask = batch == g
            graph_nodes = torch.nonzero(graph_mask).squeeze(1)

            if len(graph_nodes) == 0:
                continue

            # Get node indices in the original tensor
            node_indices = {n.item(): i for i, n in enumerate(graph_nodes)}

            # Get node features for this graph
            graph_q = q[graph_nodes].view(len(graph_nodes), self.num_heads, self.head_dim)
            graph_k = k[graph_nodes].view(len(graph_nodes), self.num_heads, self.head_dim)
            graph_v = v[graph_nodes].view(len(graph_nodes), self.num_heads, self.head_dim)

            # Reshape for multi-head attention
            graph_q = graph_q.permute(1, 0, 2)  # [heads, nodes, head_dim]
            graph_k = graph_k.permute(1, 0, 2)  # [heads, nodes, head_dim]
            graph_v = graph_v.permute(1, 0, 2)  # [heads, nodes, head_dim]

            # Compute attention scores
            attn_scores = torch.matmul(graph_q, graph_k.transpose(-2, -1)) / (
                        self.head_dim ** 0.5)  # [heads, nodes, nodes]

            # Get edges for this graph
            graph_edge_mask = (edge_index[0, :] >= graph_nodes.min()) & \
                              (edge_index[0, :] <= graph_nodes.max()) & \
                              (edge_index[1, :] >= graph_nodes.min()) & \
                              (edge_index[1, :] <= graph_nodes.max())

            if graph_edge_mask.sum() > 0:
                graph_edges = edge_index[:, graph_edge_mask]
                graph_edge_weights = edge_weights[graph_edge_mask]

                # Adjust edge indices to local graph indices
                local_src = torch.tensor([node_indices[n.item()] for n in graph_edges[0]], device=x.device)
                local_dst = torch.tensor([node_indices[n.item()] for n in graph_edges[1]], device=x.device)

                # Add edge weights to attention scores
                for h in range(self.num_heads):
                    attn_scores[h, local_src, local_dst] += graph_edge_weights[:, h]

            # Apply mask if provided
            if mask is not None:
                graph_node_mask = mask[graph_nodes]
                attn_mask = ~graph_node_mask.bool().unsqueeze(0).unsqueeze(-1)
                attn_scores = attn_scores.masked_fill(attn_mask, -1e9)

            # Apply softmax and dropout
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Apply attention to values
            graph_out = torch.matmul(attn_probs, graph_v)  # [heads, nodes, head_dim]
            graph_out = graph_out.permute(1, 0, 2).reshape(len(graph_nodes), self.out_dim)  # [nodes, out_dim]

            # Store output for this graph
            out[graph_nodes] = graph_out

        # Residual connection and projection
        x_out = self.proj(x) + self.dropout(out)

        # Feed-forward with residual
        x_out = x_out + self.dropout(self.ff(self.norm2(x_out)))

        return x_out

class AIGTransformer(nn.Module):
    def __init__(
            self,
            node_features=4,
            edge_features=2,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            max_nodes=120
    ):
        super(AIGTransformer, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Node feature embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)

        # Positional encoding (will be applied per graph)
        self.pos_encoding = nn.Parameter(torch.randn(max_nodes, hidden_dim))

        # Graph transformer layers
        self.layers = nn.ModuleList([
            EdgeAwareGraphTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                edge_dim=edge_features,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Task-specific head
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass for batched PyG Data objects

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_features]
                - batch: Batch indices [num_nodes]
                - node_mask: Masking information [num_nodes]
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None
        node_mask = data.node_mask if hasattr(data, 'node_mask') else None

        # Node feature embedding
        x = self.node_embedding(x)

        # Add positional encoding based on node position within each graph
        if batch is not None:
            for g in range(batch.max().item() + 1):
                graph_mask = batch == g
                num_nodes = graph_mask.sum().item()
                x[graph_mask] = x[graph_mask] + self.pos_encoding[:num_nodes]
        else:
            # Single graph case
            x = x + self.pos_encoding[:x.size(0)]

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch, node_mask)

        # Predict node features
        node_out = self.node_predictor(x)

        return {
            'node_features': node_out,
            'mask': node_mask
        }

    def compute_loss(self, outputs, targets):
        """
        Compute loss for masked node prediction task

        Args:
            outputs: Dict with 'node_features' and 'mask'
            targets: Dict with 'node_features', 'edge_index', 'edge_attr'
        """
        pred_nodes = outputs['node_features']
        mask = outputs['mask']
        target_nodes = targets['node_features']

        # Only compute loss for masked nodes
        node_loss = F.binary_cross_entropy_with_logits(
            pred_nodes[mask],
            target_nodes[mask]
        )

        return node_loss, {"node_loss": node_loss}

#     def generate_aig(self, partial_graph, num_steps=10):
#         """Incrementally generate a complete AIG from a partial graph"""
#         # Start with the partial graph
#         current_graph = partial_graph.clone()
#
#         for step in range(num_steps):
#             # Forward pass to get predictions
#             outputs = self.forward([current_graph])
#
#             # Get node and edge predictions
#             node_preds = outputs['node_features'][0]
#             edge_preds = outputs['edge_features'][0] if outputs['edge_features'] is not None else None
#
#             # For "node_replacement" task
#             if self.current_task == "node_replacement":
#                 # Find missing nodes and replace them
#                 missing_mask = ~current_graph.mask
#                 if missing_mask.sum() > 0:
#                     # Replace node features for missing nodes
#                     missing_indices = torch.where(missing_mask)[0]
#                     current_graph.x[missing_indices] = (node_preds[missing_indices] > 0).float()
#
#                     # Update mask
#                     current_graph.mask[missing_indices] = True
#
#             # For "missing_level" or "and_gate" tasks
#             elif self.current_task in ["missing_level", "and_gate"]:
#                 # Add new nodes and edges
#                 # This is more complex and depends on the specific AIG structure
#                 # Threshold prediction to make binary decisions
#                 new_node_features = (node_preds > 0).float()
#                 new_edges = []
#                 new_edge_attrs = []
#
#                 if edge_preds is not None:
#                     # Threshold edge predictions
#                     potential_edges = (edge_preds > 0).float()
#
#                     # Add most confident new edges
#                     confidence = torch.sigmoid(edge_preds)
#                     for i in range(current_graph.num_nodes):
#                         for j in range(current_graph.num_nodes):
#                             if confidence[i, j, 0] > 0.8:  # High confidence threshold
#                                 new_edges.append([i, j])
#                                 new_edge_attrs.append(potential_edges[i, j])
#
#                 # Update the graph with new edges
#                 if new_edges:
#                     new_edge_index = torch.tensor(new_edges).t()
#                     new_edge_attrs = torch.stack(new_edge_attrs)
#
#                     # Combine with existing edges
#                     current_graph.edge_index = torch.cat([current_graph.edge_index, new_edge_index], dim=1)
#                     current_graph.edge_attr = torch.cat([current_graph.edge_attr, new_edge_attrs], dim=0)
#
#             # Verify functional correctness
#             # Simulate truth table and compare with target
#             # This would require a custom AIG simulator function
#
#         return current_graph
# #
# class AIGModel(nn.Module):
#     def __init__(self, node_features, edge_features,
#                  hidden_dim=256, num_layers=4, heads=8, dropout=0.1):
#         super(AIGModel, self).__init__()
#
#         self.node_encoder = nn.Linear(node_features, hidden_dim)
#         self.edge_encoder = nn.Linear(edge_features, hidden_dim) if edge_features > 0 else None
#
#         # This GAT model *automatically* applies `num_layers` internal GATConv layers
#         self.gat = GAT(
#             in_channels=hidden_dim,
#             hidden_channels=hidden_dim,
#             num_layers=num_layers,
#             out_channels=hidden_dim,
#             dropout=dropout,
#             heads=heads,
#             edge_dim=hidden_dim if edge_features > 0 else None
#         )
#
#         self.node_decoder = nn.Linear(hidden_dim, node_features)
#
#         self.edge_existence_predictor = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
#
#         if edge_features > 0:
#             self.edge_feature_predictor = nn.Linear(hidden_dim * 2, edge_features)
#
#     def forward(self, data):
#         """
#         Forward pass for joint node and edge prediction
#         """
#         x, edge_index = data.x, data.edge_index
#         edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
#
#         # Get full edge candidates for prediction
#         all_node_pairs = self._get_candidate_edges(data)
#
#         # Initial embeddings
#         x = self.node_encoder(x)
#
#         if edge_attr is not None and self.edge_encoder is not None:
#             edge_embedding = self.edge_encoder(edge_attr)
#         else:
#             edge_embedding = None
#
#         # Apply graph attention layers
#
#         x = self.gat(x, edge_index, edge_attr=edge_embedding)
#
#         # Decode node features
#         node_pred = self.node_decoder(x)
#
#         # Predict edges between all node pairs
#         edge_existence, edge_features = self._predict_edges(x, all_node_pairs)
#
#         return {
#             'node_pred': node_pred,
#             'edge_existence': edge_existence,
#             'edge_features': edge_features,
#             'candidate_edges': all_node_pairs
#         }
#
#     def _get_candidate_edges(self, data):
#         """
#         Return only the edges connected to (at least one) masked node.
#         """
#         # 1) Identify which nodes are masked.
#         #    Here we assume data.node_mask is a boolean tensor of length [num_nodes].
#         node_mask = data.node_mask if hasattr(data, 'node_mask') else None
#
#         if node_mask is None:
#             # If there's no node_mask, just return existing or target edges as a fallback
#             existing_edges = (
#                 data.edge_index_target if hasattr(data, 'edge_index_target')
#                 else data.edge_index
#             )
#             return existing_edges
#
#         # 2) Decide on the "base" edge set to consider
#         #    Typically you'd use data.edge_index_target if it exists,
#         #    otherwise the original data.edge_index.
#         existing_edges = (
#             data.edge_index_target if hasattr(data, 'edge_index_target')
#             else data.edge_index
#         )
#
#         # 3) For each edge, see if src or dst is masked
#         src, dst = existing_edges
#         src_masked = node_mask[src]
#         dst_masked = node_mask[dst]
#         keep_edges = src_masked | dst_masked  # OR => keep if either end is masked
#
#         # 4) Filter
#         filtered_edges = existing_edges[:, keep_edges]
#
#         return filtered_edges
#
#     def _predict_edges(self, node_embeddings, candidate_edges):
#         """Predict edge existence and features for candidate edges"""
#         src, dst = candidate_edges
#
#         # Create edge features by concatenating node embeddings
#         edge_features_input = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=1)
#
#         # Predict edge existence
#         edge_existence = self.edge_existence_predictor(edge_features_input)
#
#         # Predict edge features if needed
#         if hasattr(self, 'edge_feature_predictor'):
#             edge_features = self.edge_feature_predictor(edge_features_input)
#         else:
#             edge_features = None
#
#         return edge_existence, edge_features
#
#     def compute_loss(self, pred, data):
#         """Compute loss for masked node and edge prediction"""
#         losses = {}
#
#         # Node reconstruction loss (only for masked nodes)
#         if hasattr(data, 'node_mask') and data.node_mask.sum() > 0:
#             node_loss = F.mse_loss(
#                 pred['node_pred'][data.node_mask],
#                 data.x_target[data.node_mask]
#             )
#             losses['node_loss'] = node_loss
#         else:
#             losses['node_loss'] = torch.tensor(0.0, device=pred['node_pred'].device)
#
#         # Edge existence loss
#         if pred['edge_existence'] is not None and hasattr(data, 'edge_index_target'):
#             # Create target tensor (1 for edges that exist in the original graph)
#             all_edges = pred['candidate_edges']
#             target_edges = data.edge_index_target
#
#             # Create a mapping for target edges
#             edge_map = {(src.item(), dst.item()): 1 for src, dst in zip(target_edges[0], target_edges[1])}
#
#             # Create target tensor
#             edge_targets = torch.zeros(all_edges.size(1), device=all_edges.device)
#             for i in range(all_edges.size(1)):
#                 src, dst = all_edges[0, i].item(), all_edges[1, i].item()
#                 if (src, dst) in edge_map:
#                     edge_targets[i] = 1.0
#
#             # Compute binary cross entropy loss
#             edge_existence_loss = F.binary_cross_entropy(
#                 pred['edge_existence'].squeeze(),
#                 edge_targets
#             )
#             losses['edge_existence_loss'] = edge_existence_loss
#         else:
#             losses['edge_existence_loss'] = torch.tensor(0.0, device=pred['node_pred'].device)
#
#         # Edge feature loss (if applicable)
#         if pred['edge_features'] is not None and hasattr(data, 'edge_attr_target'):
#             # For simplicity, we're only computing loss for edges that exist in the target
#             # In a more complete implementation, you'd filter based on predicted existence
#             # This is a simplified version
#             edge_feature_loss = F.mse_loss(
#                 pred['edge_features'],
#                 data.edge_attr_target
#             )
#             losses['edge_feature_loss'] = edge_feature_loss
#         else:
#             losses['edge_feature_loss'] = torch.tensor(0.0, device=pred['node_pred'].device)
#
#         # Total loss
#         total_loss = losses['node_loss'] + losses['edge_existence_loss']
#         if 'edge_feature_loss' in losses:
#             total_loss += losses['edge_feature_loss']
#
#         losses['total_loss'] = total_loss
#
#         return total_loss, {k: v.item() for k, v in losses.items()}
#
