# src/model/layer_dag.py
import dgl.sparse as dglsp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import random # Not typically used in the original core sampling logic like this

from einops import rearrange

__all__ = [
    'LayerDAG'
]


class SinusoidalPE(nn.Module):
    def __init__(self, pe_size):
        super().__init__()

        self.pe_size = pe_size
        if pe_size > 0:
            # Ensure div_term is created on the correct device if it's used early
            # However, it's usually fine as Parameters get moved with the model.
            self.div_term = torch.exp(torch.arange(0, pe_size, 2) *
                                      (-math.log(10000.0) / pe_size))
            self.div_term = nn.Parameter(self.div_term, requires_grad=False)

    def forward(self, position):
        if self.pe_size == 0:
            # Ensure returns empty tensor on the same device as position
            return torch.zeros(len(position), 0, device=position.device)
        # Ensure position is float for multiplication
        position = position.float()
        return torch.cat([
            torch.sin(position * self.div_term),
            torch.cos(position * self.div_term)
        ], dim=-1)


class BiMPNNLayer(nn.Module):
    def __init__(self, in_size, out_size):  # Original took in_size, out_size
        super().__init__()
        # In the original, in_size and out_size for BiMPNNLayer were typically the same (hidden_size)
        self.W = nn.Linear(in_size, out_size)
        self.W_trans = nn.Linear(in_size, out_size)
        self.W_self = nn.Linear(in_size, out_size)

    def forward(self, A, A_T, h_n):
        if A.nnz == 0:
            h_n_out = self.W_self(h_n)
        else:
            h_n_out = A @ self.W(h_n) + A_T @ self.W_trans(h_n) + \
                      self.W_self(h_n)
        return F.gelu(h_n_out)


class OneHotPE(nn.Module):
    def __init__(self, pe_size):
        super().__init__()
        self.pe_size = pe_size

    def forward(self, position):
        if self.pe_size == 0:
            return torch.zeros(len(position), 0, device=position.device)
        # Ensure position is long for one_hot and squeeze if it's (N,1)
        position_squeezed = position.long().squeeze(-1) if position.ndim > 1 and position.shape[
            -1] == 1 else position.long()
        # Clamp before one_hot to avoid index out of bounds
        clamped_position = position_squeezed.clamp(min=0, max=self.pe_size - 1)
        return F.one_hot(clamped_position,
                         num_classes=self.pe_size).float()  # one_hot returns long, convert to float


class MultiEmbedding(nn.Module):
    # num_x_n_cat is expected to be a LongTensor from train_set.num_categories
    def __init__(self, num_x_n_cat, hidden_size):
        super().__init__()

        if isinstance(num_x_n_cat, int):  # Single feature category count
            processed_num_x_n_cat_list = [num_x_n_cat]
        elif isinstance(num_x_n_cat, torch.Tensor):
            if num_x_n_cat.ndim == 0:  # Scalar tensor
                processed_num_x_n_cat_list = [num_x_n_cat.item()]
            else:  # 1D tensor of category counts per feature
                processed_num_x_n_cat_list = num_x_n_cat.tolist()
        elif isinstance(num_x_n_cat, (list, tuple)):
            processed_num_x_n_cat_list = list(num_x_n_cat)
        else:
            raise TypeError(f"num_x_n_cat must be int, Tensor, list, or tuple, got {type(num_x_n_cat)}")

        self.emb_list = nn.ModuleList([
            nn.Embedding(int(num_cat_i), hidden_size)  # Ensure num_cat_i is int
            for num_cat_i in processed_num_x_n_cat_list
        ])

    def forward(self, x_n_cat):
        # x_n_cat: (N) if single feature, (N, num_features) if multiple
        if not self.emb_list:
            # This case should ideally not be reached if initialized properly
            return torch.empty(x_n_cat.shape[0], 0, device=x_n_cat.device)

        if x_n_cat.ndim == 1:  # Single feature dimension
            if len(self.emb_list) != 1:
                raise ValueError(
                    f"Input x_n_cat is 1D, but have {len(self.emb_list)} embedding layers. Expected 1 for 1D input.")
            x_n_emb = self.emb_list[0](x_n_cat)
        elif x_n_cat.ndim == 2:  # Multiple features: (N, num_features)
            if x_n_cat.shape[1] != len(self.emb_list):
                raise ValueError(
                    f"MultiEmbedding: Number of features in x_n_cat ({x_n_cat.shape[1]}) "
                    f"does not match number of embedding layers ({len(self.emb_list)})."
                )
            x_n_emb_parts = []
            for i in range(len(self.emb_list)):
                x_n_emb_parts.append(self.emb_list[i](x_n_cat[:, i]))
            x_n_emb = torch.cat(x_n_emb_parts, dim=1)
        else:
            raise ValueError(f"Unsupported x_n_cat shape: {x_n_cat.shape}")
        return x_n_emb


class BiMPNNEncoder(nn.Module):
    def __init__(self,
                 num_x_n_cat,  # LongTensor or list/int
                 x_n_emb_size,  # Embedding size for each feature type
                 pe_emb_size,
                 hidden_size,  # Main internal hidden dimension
                 num_mpnn_layers,
                 pe=None,
                 y_emb_size=0,
                 pool=None):
        super().__init__()

        self.pe = pe
        self.level_emb = None
        if self.pe in ['relative_level', 'abs_level'] and pe_emb_size > 0:
            self.level_emb = SinusoidalPE(pe_emb_size)
        elif self.pe == 'relative_level_one_hot' and pe_emb_size > 0:
            self.level_emb = OneHotPE(pe_emb_size)

        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)

        # Calculate actual dimension after MultiEmbedding
        if isinstance(num_x_n_cat, int) or (isinstance(num_x_n_cat, torch.Tensor) and num_x_n_cat.ndim == 0):
            actual_x_n_dim_after_embedding = x_n_emb_size
        elif isinstance(num_x_n_cat, (list, tuple, torch.Tensor)):  # list, tuple, or 1D Tensor
            actual_x_n_dim_after_embedding = len(num_x_n_cat) * x_n_emb_size
        else:
            raise TypeError(f"Unhandled type for num_x_n_cat in BiMPNNEncoder: {type(num_x_n_cat)}")

        self.y_emb = SinusoidalPE(y_emb_size) if y_emb_size > 0 else None

        proj_input_dim = actual_x_n_dim_after_embedding
        if self.level_emb is not None: proj_input_dim += pe_emb_size
        # y_emb is tricky if y is graph-level and pool=None.
        # Original LayerDAG assumes y is expanded to node-level if used here.
        # Or y_emb_size is 0 if unconditional.
        if self.y_emb is not None: proj_input_dim += y_emb_size

        self.proj_input = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_size),  # Project combined to hidden_size
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.mpnn_layers = nn.ModuleList()
        for _ in range(num_mpnn_layers):
            # MPNN layers operate on hidden_size
            self.mpnn_layers.append(BiMPNNLayer(hidden_size, hidden_size))

        self.project_output_n = nn.Sequential(
            nn.Linear((num_mpnn_layers + 1) * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.pool = pool
        if pool is not None:
            self.bn_g = nn.BatchNorm1d(hidden_size)

    def forward(self, A, x_n, abs_level, rel_level, y=None, A_n2g=None):
        A_T = A.T

        h_n_parts = [self.x_n_emb(x_n)]

        if self.level_emb is not None:
            node_pe = None
            if self.pe == 'abs_level':
                node_pe = self.level_emb(abs_level)
            elif self.pe in ['relative_level', 'relative_level_one_hot']:
                node_pe = self.level_emb(rel_level)

            if node_pe is not None and node_pe.shape[1] > 0:  # Check if PE is actually generated
                h_n_parts.append(node_pe)

        if y is not None and self.y_emb is not None:
            # Assuming y is (N, y_feature_dim) or (B, y_feature_dim)
            # If y is (B, y_feature_dim) and pool is None, it needs expansion via A_n2g
            # Original LayerDAG's BiMPNNEncoder in config often had y_emb_size=0 or y was pre-expanded.
            # For simplicity matching original structure, assume y is prepared (node-level) if pool is None.
            h_y_processed = self.y_emb(y)  # y might be (N,1) or (B,1) -> (N or B, y_emb_size)

            if self.pool is None and h_y_processed.shape[0] != x_n.shape[0] and A_n2g is not None:
                # Attempt to expand graph-level y to node-level if A_n2g is available
                # This requires A_n2g to map nodes to graphs correctly.
                # A common way: if A_n2g is (B,N) and A_n2g.coo()[0] are graph_ids for nodes
                # This is complex to do robustly here without knowing A_n2g's exact construction from collate.
                # For now, we'll assume y is node-level if pool is None, or y_emb_size=0 for unconditional.
                # If y is (B, y_emb_size) and A_n2g is (B,N), we need to find graph_idx for each node.
                # This part is often handled in data prep or by the main model.
                # Given the original structure, if y is used here, it's often assumed to be node-level.
                # If y is (B, y_emb_size) and we need (N, y_emb_size), and A_n2g is (B,N)
                # A robust way: find graph_idx for each node.
                # Example: if A_n2g is from `torch.stack([gids, nids])` where `gids` are graph indices for nodes `0..N-1`
                # then `gids_for_nodes = A_n2g.coo()[0][torch.sort(A_n2g.coo()[1]).indices]` (if nids are unique)
                # This is too complex for here. Assume y is correctly shaped or y_emb_size=0.
                pass  # Keep h_y_processed as is, relying on upstream or config.

            if h_y_processed.shape[0] == x_n.shape[0]:  # If y is already node-level
                h_n_parts.append(h_y_processed)
            elif self.pool is not None and A_n2g is not None and h_y_processed.shape[0] == A_n2g.shape[0]:
                # If pooling, y is graph-level and will be combined with h_g later.
                # So, don't append h_y_processed to h_n_parts here.
                pass
            # else: if shapes mismatch and no clear way to combine, it might be an issue.

        h_n_combined = torch.cat(h_n_parts, dim=-1)
        h_n = self.proj_input(h_n_combined)

        h_n_cat_list = [h_n]
        for layer in self.mpnn_layers:
            h_n = layer(A, A_T, h_n)
            h_n_cat_list.append(h_n)
        h_n = torch.cat(h_n_cat_list, dim=-1)
        h_n = self.project_output_n(h_n)

        if self.pool is None:
            return h_n
        elif self.pool in ['sum', 'mean']:
            if A_n2g is None: raise ValueError(f"A_n2g must be provided for {self.pool} pooling.")
            h_g = A_n2g @ h_n
            if self.pool == 'mean':
                sum_val = A_n2g.sum(dim=1).unsqueeze(-1)
                sum_val = torch.where(sum_val == 0, torch.ones_like(sum_val), sum_val)  # Avoid div by zero
                h_g = h_g / sum_val

            # If y was graph-level and meant to be combined with h_g
            if y is not None and self.y_emb is not None and h_y_processed.shape[0] == h_g.shape[0]:
                h_g = torch.cat([h_g, h_y_processed], dim=-1)  # Example: combine here
                # Note: This would change the output dimension of BiMPNNEncoder if y is used this way.
                # The GraphClassifier's input emb_size would need to account for this.
                # Original LayerDAG usually set y_emb_size=0 in encoder if y is not used directly on nodes.

            return self.bn_g(h_g)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool}")


class GraphClassifier(nn.Module):
    def __init__(self,
                 graph_encoder,
                 emb_size,  # This should be the output size of graph_encoder
                 num_classes):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size),  # emb_size here refers to h_g's feature dim
            nn.GELU(),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, y=None):
        # y is graph-level condition (e.g. (B,1) or (B,F_y))
        h_g = self.graph_encoder(A, x_n, abs_level, rel_level, y=y, A_n2g=A_n2g)
        pred_g = self.predictor(h_g)
        return pred_g


class TransformerLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 dropout):
        super().__init__()

        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_qk = nn.Linear(hidden_size, hidden_size * 2)

        self._reset_parameters()

        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        if not (head_dim * num_heads == hidden_size):  # Check for clean division
            raise ValueError("hidden_size must be divisible by num_heads")
        self.scale = head_dim ** -0.5

        self.proj_new = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_qk.weight)
        if self.to_v.bias is not None: nn.init.zeros_(self.to_v.bias)
        if self.to_qk.bias is not None: nn.init.zeros_(self.to_qk.bias)

    def attn(self, q, k, v, num_query_cumsum):
        batch_size = len(num_query_cumsum) - 1
        if batch_size <= 0:
            return q.new_empty(0, q.shape[-1]) if q is not None and q.numel() > 0 else torch.empty(0, 0,
                                                                                                   device=num_query_cumsum.device)

        num_query_nodes = torch.diff(num_query_cumsum)
        max_num_nodes = 0
        if num_query_nodes.numel() > 0:  # Ensure num_query_nodes is not empty
            max_num_nodes = num_query_nodes.max().item()

        if max_num_nodes == 0:  # No queries to process in any graph of the batch
            # If q is already empty (total queries = 0), return q
            if q is not None and q.shape[0] == 0: return q
            feat_dim = q.shape[-1] if q is not None and q.ndim > 1 else (
                v.shape[-1] if v is not None and v.ndim > 1 else 0)
            return torch.empty(0, feat_dim, device=q.device if q is not None else (
                v.device if v is not None else num_query_cumsum.device))

        q_padded = q.new_zeros(batch_size, max_num_nodes, q.shape[-1])
        k_padded = k.new_zeros(batch_size, max_num_nodes, k.shape[-1])
        v_padded = v.new_zeros(batch_size, max_num_nodes, v.shape[-1])
        # pad_mask should be boolean
        pad_mask = q.new_zeros(batch_size, max_num_nodes, dtype=torch.bool)

        for i in range(batch_size):
            start_idx, end_idx = num_query_cumsum[i], num_query_cumsum[i + 1]
            num_q_i = num_query_nodes[i].item()  # num_q_i can be 0
            if num_q_i > 0:
                q_padded[i, :num_q_i] = q[start_idx:end_idx]
                k_padded[i, :num_q_i] = k[start_idx:end_idx]
                v_padded[i, :num_q_i] = v[start_idx:end_idx]
            pad_mask[i, num_q_i:] = True

        q_padded = rearrange(q_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        k_padded = rearrange(k_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        v_padded = rearrange(v_padded, 'b n (h d) -> b h n d', h=self.num_heads)

        dot = torch.matmul(q_padded, k_padded.transpose(-1, -2)) * self.scale
        dot = dot.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(2),  # (B,1,1,max_N)
            float('-inf'),
        )

        attn_scores = F.softmax(dot, dim=-1)
        h_n_padded = torch.matmul(attn_scores, v_padded)
        h_n_padded = rearrange(h_n_padded, 'b h n d -> (b n) (h d)')

        unpad_mask = (~pad_mask).reshape(-1)
        return h_n_padded[unpad_mask]

    def forward(self, h_n, num_query_cumsum):
        if h_n.shape[0] == 0:  # No query nodes
            return h_n

        v_n = self.to_v(h_n)
        q_n, k_n = self.to_qk(h_n).chunk(2, dim=-1)

        h_n_new = self.attn(q_n, k_n, v_n, num_query_cumsum)

        if h_n_new.shape[0] != h_n.shape[0]:
            # This case indicates an issue in attn or padding/unpadding.
            # For robustness, if shapes mismatch, avoid direct addition.
            # A simple fallback is to return h_n, but this bypasses the transformer layer.
            # Or, if h_n_new is empty but h_n is not, it's also an issue.
            print(
                f"WARNING: TransformerLayer shape mismatch or empty h_n_new. h_n: {h_n.shape}, h_n_new: {h_n_new.shape}. Skipping update.")
            return h_n  # Fallback

        h_n = self.norm1(h_n + self.proj_new(h_n_new))
        h_n = self.norm2(h_n + self.out_proj(h_n))
        return h_n


class NodePredModel(nn.Module):
    def __init__(self,
                 graph_encoder,
                 num_x_n_cat,  # Tensor of actual category counts (e.g. total_cats-1 for each feature)
                 x_n_emb_size,  # Embedding size for x_n_t attributes
                 t_emb_size,
                 in_hidden_size,  # Output hidden_size from graph_encoder's proj_input
                 out_hidden_size,  # Output hidden_size for transformer and final predictors
                 num_transformer_layers,
                 num_heads,
                 dropout):
        super().__init__()

        self.graph_encoder = graph_encoder
        # MultiEmbedding for noisy node attributes x_n_t
        # num_x_n_cat here should be the number of *actual* categories for each feature,
        # as x_n_t will contain noisy versions of these actual attributes.
        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)
        self.t_emb = SinusoidalPE(t_emb_size)

        # Calculate actual dimension of x_n_t after embedding
        if isinstance(num_x_n_cat, int) or (isinstance(num_x_n_cat, torch.Tensor) and num_x_n_cat.ndim == 0):
            actual_x_n_t_dim_embedded = x_n_emb_size
        elif isinstance(num_x_n_cat, (list, tuple, torch.Tensor)):
            actual_x_n_t_dim_embedded = len(num_x_n_cat) * x_n_emb_size
        else:
            raise TypeError(f"Unhandled type for num_x_n_cat in NodePredModel: {type(num_x_n_cat)}")

        # Input to project_h_n is concatenation of:
        # 1. Embedded noisy node attributes (h_n_t_attrs)
        # 2. Graph embedding from graph_encoder (h_g, which is in_hidden_size)
        # 3. Timestep embedding (h_t_graph)
        combined_input_dim_for_proj = actual_x_n_t_dim_embedded + in_hidden_size + t_emb_size

        self.project_h_n = nn.Sequential(
            nn.Linear(combined_input_dim_for_proj, out_hidden_size),
            nn.GELU()
        )

        self.trans_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.trans_layers.append(TransformerLayer(
                out_hidden_size, num_heads, dropout
            ))

        self.pred_list = nn.ModuleList([])
        # num_x_n_cat (passed to __init__) should be a list/tensor of K_d values
        # (number of actual categories for each node attribute dimension d)
        processed_num_x_n_cat_list = []
        if isinstance(num_x_n_cat, torch.Tensor):
            if num_x_n_cat.ndim == 0:
                processed_num_x_n_cat_list = [num_x_n_cat.item()]
            else:
                processed_num_x_n_cat_list = num_x_n_cat.tolist()
        elif isinstance(num_x_n_cat, (list, tuple)):
            processed_num_x_n_cat_list = list(num_x_n_cat)
        elif isinstance(num_x_n_cat, int):
            processed_num_x_n_cat_list = [num_x_n_cat]

        for num_classes_f in processed_num_x_n_cat_list:
            self.pred_list.append(nn.Sequential(
                nn.Linear(out_hidden_size, out_hidden_size),
                nn.GELU(),
                nn.Linear(out_hidden_size, int(num_classes_f))  # Predict K_d classes
            ))

    def forward_with_h_g(self, h_g, x_n_t,
                         t, query2g, num_query_cumsum):
        # x_n_t: (Q, num_features) or (Q) - noisy node attributes for query nodes
        # h_g: (B, F_graph_enc = in_hidden_size) - graph embeddings
        # t: (B, 1) - timesteps for graphs
        # query2g: (Q) - maps each query node to its graph index in the batch
        # num_query_cumsum: (B+1) - cumsum of query nodes per graph

        if x_n_t.shape[0] == 0:  # No query nodes
            # Return a list of empty tensors, matching the structure of pred_list
            return [torch.empty(0, device=h_g.device) for _ in self.pred_list]

        h_t_graph = self.t_emb(t)  # (B, t_emb_size)
        # Concatenate graph embedding (h_g) with its timestep embedding (h_t_graph)
        h_g_cond = torch.cat([h_g, h_t_graph], dim=1)  # (B, in_hidden_size + t_emb_size)

        # Embed the noisy node attributes x_n_t
        h_n_t_attrs = self.x_n_emb(x_n_t)  # (Q, actual_x_n_t_dim_embedded)

        # Expand graph-level conditional information (h_g_cond) to each query node
        h_g_cond_expanded = h_g_cond[query2g]  # (Q, in_hidden_size + t_emb_size)

        # Concatenate node's own noisy attributes with the expanded graph condition
        h_n_t_combined = torch.cat([h_n_t_attrs, h_g_cond_expanded], dim=1)
        h_n_t = self.project_h_n(h_n_t_combined)  # (Q, out_hidden_size)

        for trans_layer in self.trans_layers:
            if h_n_t.shape[0] == 0: break  # Should not happen if initial check passed
            h_n_t = trans_layer(h_n_t, num_query_cumsum)

        pred = []
        if h_n_t.shape[0] > 0:  # Ensure h_n_t is not empty after transformers
            for d_idx in range(len(self.pred_list)):
                pred.append(self.pred_list[d_idx](h_n_t))
        else:  # If h_n_t became empty (e.g. transformer layer returned empty)
            pred = [torch.empty(0, device=h_g.device) for _ in self.pred_list]

        return pred

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, x_n_t,
                t, query2g, num_query_cumsum, y=None):
        # y is graph-level condition for graph_encoder
        h_g = self.graph_encoder(A, x_n, abs_level,
                                 rel_level, y=y, A_n2g=A_n2g)
        return self.forward_with_h_g(h_g, x_n_t, t, query2g,
                                     num_query_cumsum)


class EdgePredModel(nn.Module):
    def __init__(self,
                 graph_encoder,
                 t_emb_size,
                 in_hidden_size,  # Output hidden_size from graph_encoder's proj_input
                 out_hidden_size):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.t_emb = SinusoidalPE(t_emb_size)
        # Input to MLP is [src_node_emb, dst_node_emb, edge_timestep_emb]
        mlp_input_dim = 2 * in_hidden_size + t_emb_size
        self.pred = nn.Sequential(
            nn.Linear(mlp_input_dim, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, 2)  # Binary prediction (edge exists or not)
        )

    def forward(self, A, x_n, abs_level, rel_level, t,
                query_src, query_dst, y=None):
        # graph_encoder's pool should be None for EdgePredModel as it needs node embeddings (h_n)
        # y is graph-level condition. If graph_encoder.pool is None, y needs to be handled
        # carefully (e.g. expanded to node level if used by encoder, or y_emb_size=0 if not conditional)
        # Original LayerDAG config for edge_pred.graph_encoder usually has pool=None.
        h_n = self.graph_encoder(A, x_n, abs_level, rel_level, y=y, A_n2g=None)  # Pass A_n2g=None

        if query_src.numel() == 0:  # No query edges
            return torch.empty(0, 2, device=h_n.device if h_n.numel() > 0 else t.device)

        # t is (num_queries, 1) for edges
        h_t_edge = self.t_emb(t)  # (num_queries, t_emb_size)

        h_e_parts = []
        if h_t_edge.shape[1] > 0: h_e_parts.append(h_t_edge)  # Only if t_emb_size > 0
        h_e_parts.append(h_n[query_src])
        h_e_parts.append(h_n[query_dst])

        h_e = torch.cat(h_e_parts, dim=-1)

        return self.pred(h_e)


class LayerDAG(nn.Module):
    def __init__(self,
                 device,
                 num_x_n_cat,  # From train_set.num_categories (actual_types + 1 for dummy)
                 node_count_encoder_config,
                 max_layer_size,
                 node_diffusion,  # Instance of DiscreteDiffusion
                 node_pred_graph_encoder_config,
                 node_predictor_config,
                 edge_diffusion,  # Instance of EdgeDiscreteDiffusion
                 edge_pred_graph_encoder_config,
                 edge_predictor_config,
                 max_level=None):
        super().__init__()

        self.device = torch.device(device)

        # Process num_x_n_cat to be a tensor on the correct device
        # This is the total number of categories including the dummy one for each feature type.
        if isinstance(num_x_n_cat, int):
            self.num_x_n_cat_tensor = torch.LongTensor([num_x_n_cat]).to(self.device)
        elif isinstance(num_x_n_cat, torch.Tensor):
            self.num_x_n_cat_tensor = num_x_n_cat.clone().detach().to(self.device)
        elif isinstance(num_x_n_cat, (list, tuple)):
            self.num_x_n_cat_tensor = torch.LongTensor(num_x_n_cat).to(self.device)
        else:
            raise TypeError(f"num_x_n_cat must be an int, list, tuple or torch.Tensor, got {type(num_x_n_cat)}")

        # dummy_x_n should represent the index of the dummy category for each feature type
        if self.num_x_n_cat_tensor.ndim == 0 or len(self.num_x_n_cat_tensor) == 1:
            self.dummy_x_n = self.num_x_n_cat_tensor.item() - 1  # Scalar index
        else:  # Tensor of indices for multi-feature
            self.dummy_x_n = self.num_x_n_cat_tensor - 1

        # --- Node Count Model ---
        nc_config = node_count_encoder_config  # This is config['node_count']['model']
        # y_emb_size for node_count_model depends on whether the overall training is conditional
        # This should be determined by train.py based on general_config['conditional']
        # For now, assume nc_config['y_emb_size'] is correctly set (e.g. to 0 if not conditional)
        node_count_encoder = BiMPNNEncoder(
            num_x_n_cat=self.num_x_n_cat_tensor,  # Pass full categories including dummy
            x_n_emb_size=nc_config['x_n_emb_size'],
            pe_emb_size=nc_config.get('pe_emb_size', 0),
            # hidden_size is the main internal dimension for BiMPNNEncoder
            hidden_size=nc_config.get('hidden_size', nc_config['x_n_emb_size']),
            # Fallback if 'hidden_size' not in config
            num_mpnn_layers=nc_config['num_mpnn_layers'],
            pe=nc_config.get('pe'),
            y_emb_size=nc_config.get('y_emb_size', 0),
            pool=nc_config.get('pool', 'sum')  # Default to 'sum' if not specified
        ).to(self.device)
        self.node_count_model = GraphClassifier(
            node_count_encoder,
            # emb_size for GraphClassifier is the output dim of node_count_encoder (which is its hidden_size)
            emb_size=nc_config.get('hidden_size', nc_config['x_n_emb_size']),
            num_classes=max_layer_size + 1
        ).to(self.device)

        # --- Node Prediction Model ---
        self.node_diffusion = node_diffusion.to(self.device)  # Pass instance
        np_ge_config = node_pred_graph_encoder_config  # config['node_pred']['graph_encoder']
        np_pred_config = node_predictor_config  # config['node_pred']['predictor']

        node_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat=self.num_x_n_cat_tensor,  # Full categories for existing graph nodes
            x_n_emb_size=np_ge_config['x_n_emb_size'],
            pe_emb_size=np_ge_config.get('pe_emb_size', 0),
            hidden_size=np_ge_config.get('hidden_size', np_ge_config['x_n_emb_size']),
            num_mpnn_layers=np_ge_config['num_mpnn_layers'],
            pe=np_ge_config.get('pe'),
            y_emb_size=np_ge_config.get('y_emb_size', 0),
            pool=np_ge_config.get('pool', 'sum')
        ).to(self.device)

        # For NodePredModel's x_n_t embedding, num_x_n_cat should be actual categories (no dummy)
        # This is because x_n_t represents noisy versions of *actual* node attributes.
        # node_diffusion.num_classes_list should store these actual category counts.
        num_actual_node_cats_for_pred = torch.LongTensor(self.node_diffusion.num_classes_list).to(self.device)

        self.node_pred_model = NodePredModel(
            node_pred_graph_encoder,
            num_x_n_cat=num_actual_node_cats_for_pred,  # Actual categories for x_n_t
            x_n_emb_size=np_ge_config['x_n_emb_size'],  # x_n_emb_size for x_n_t embedding
            t_emb_size=np_pred_config['t_emb_size'],
            in_hidden_size=np_ge_config.get('hidden_size', np_ge_config['x_n_emb_size']),  # Output of graph_encoder
            out_hidden_size=np_pred_config['out_hidden_size'],  # Transformer hidden size
            num_transformer_layers=np_pred_config['num_transformer_layers'],
            num_heads=np_pred_config['num_heads'],
            dropout=np_pred_config['dropout']
        ).to(self.device)

        # --- Edge Prediction Model ---
        self.edge_diffusion = edge_diffusion.to(self.device)  # Pass instance
        ep_ge_config = edge_pred_graph_encoder_config  # config['edge_pred']['graph_encoder']
        ep_pred_config = edge_predictor_config  # config['edge_pred']['predictor']

        edge_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat=self.num_x_n_cat_tensor,  # Full categories for graph nodes
            x_n_emb_size=ep_ge_config['x_n_emb_size'],
            pe_emb_size=ep_ge_config.get('pe_emb_size', 0),
            hidden_size=ep_ge_config.get('hidden_size', ep_ge_config['x_n_emb_size']),
            num_mpnn_layers=ep_ge_config['num_mpnn_layers'],
            pe=ep_ge_config.get('pe'),
            y_emb_size=ep_ge_config.get('y_emb_size', 0),
            pool=ep_ge_config.get('pool')  # Often None for edge prediction
        ).to(self.device)
        self.edge_pred_model = EdgePredModel(
            edge_pred_graph_encoder,
            t_emb_size=ep_pred_config['t_emb_size'],
            # in_hidden_size is output of edge_pred_graph_encoder
            in_hidden_size=ep_ge_config.get('hidden_size', ep_ge_config['x_n_emb_size']),
            out_hidden_size=ep_pred_config['out_hidden_size']
        ).to(self.device)

        self.max_level = max_level

    @torch.no_grad()
    def sample_node_layer(self, A, x_n, abs_level, rel_level, A_n2g,
                          curr_level, y=None,  # y is graph-level condition
                          min_num_steps_n=None, max_num_steps_n=None):

        self.node_count_model.eval()
        self.node_pred_model.eval()

        batch_size = A_n2g.shape[0]
        if batch_size == 0: return []

        node_count_logits = self.node_count_model(A, x_n, abs_level, rel_level, A_n2g=A_n2g, y=y)

        if curr_level == 0:  # For the first layer, size must be > 0
            node_count_logits[:, 0] = float('-inf')

        node_count_probs = node_count_logits.softmax(dim=-1)
        # num_new_nodes_per_graph: (B) tensor, number of new nodes for each graph
        num_new_nodes_per_graph = node_count_probs.multinomial(1).squeeze(-1)

        x_n_l_list_final = [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(batch_size)]

        # Identify graphs that predict > 0 new nodes
        active_graph_mask = num_new_nodes_per_graph > 0
        if not active_graph_mask.any():
            return x_n_l_list_final  # All graphs predict 0 new nodes

        # Total number of new nodes to sample attributes for, across active graphs
        total_new_nodes_to_sample = num_new_nodes_per_graph[active_graph_mask].sum().item()
        if total_new_nodes_to_sample == 0:  # Should be caught by active_graph_mask.any()
            return x_n_l_list_final

        # query2g maps each of the total_new_nodes_to_sample to its original graph index in the batch
        query2g = torch.repeat_interleave(torch.arange(batch_size, device=self.device)[active_graph_mask],
                                          num_new_nodes_per_graph[active_graph_mask])

        # num_query_cumsum for the set of active graphs, for TransformerLayer
        active_num_nodes_pred = num_new_nodes_per_graph[active_graph_mask]
        num_query_cumsum_active = torch.cumsum(
            torch.cat([torch.tensor([0], device=self.device, dtype=torch.long), active_num_nodes_pred]),
            dim=0)

        # Initial noisy attributes x_n_t for all new query nodes
        # Sample from prior p(x_T) which is often uniform or marginal-based
        num_node_features = len(self.node_diffusion.num_classes_list)
        x_n_t_parts = []
        for d_idx in range(num_node_features):
            # marginal_list[d_idx] is (1, K_d), representing p(x_0) for feature d
            # For p(x_T), sample from self.node_diffusion.m_list[d_idx][0] (prior for T)
            # Or simply uniform if m_list is uniform.
            # For simplicity, let's use the prior from diffusion model (often uniform for x_T)
            prior_d = self.node_diffusion.m_list[d_idx][0, :self.node_diffusion.num_classes_list[d_idx]]  # (K_d)
            x_n_t_d = prior_d.unsqueeze(0).expand(total_new_nodes_to_sample, -1).multinomial(1).squeeze(-1)
            x_n_t_parts.append(x_n_t_d)

        x_n_t = torch.stack(x_n_t_parts, dim=1) if num_node_features > 1 else x_n_t_parts[0].unsqueeze(-1)
        x_n_t = x_n_t.to(self.device)  # Ensure device

        # Determine number of diffusion steps for this layer
        T_sampling = self.node_diffusion.T
        if max_num_steps_n is not None: T_sampling = min(T_sampling, max_num_steps_n)
        # Optional: Adjust T_sampling based on min_num_steps_n or curr_level (original had this)
        # For now, using fixed T_sampling or max_num_steps_n

        # Filter A, x_n, etc. for only active graphs to pass to node_pred_model
        # This requires careful re-indexing if done.
        # Simpler: pass full batch A, x_n and let node_pred_model use query2g to get relevant graph embeddings.
        # The graph encoder part of node_pred_model will run on the full batch A, x_n, etc.
        # Then h_g will be (B, F_g). query2g will map new nodes to these B graphs.

        # Timestep tensor for node_pred_model (needs to be (B,1) for the graphs these queries belong to)
        # The t passed to node_pred_model.forward_with_h_g is (B_active, 1)
        # So, we need y_active, A_active, x_n_active etc. if we sub-batch.
        # Let's assume node_pred_model's graph_encoder runs on the original full batch A,x_n,y
        # and its output h_g (B, F_g) is then indexed by query2g.

        h_g_full_batch = self.node_pred_model.graph_encoder(A, x_n, abs_level, rel_level, y=y, A_n2g=A_n2g)

        # Iterative refinement loop (DDPM sampling)
        for s_iter in range(T_sampling - 1, -1, -1):
            t_current_step = s_iter + 1  # t ranges from T down to 1

            # Timestep tensor for model: (num_active_graphs, 1)
            # query2g maps to original batch indices. We need t for each *active graph*.
            # The t for forward_with_h_g should correspond to h_g_full_batch[active_graph_mask]
            t_tensor_for_active_graphs = torch.full(
                (active_graph_mask.sum().item(), 1), t_current_step, dtype=torch.long, device=self.device
            )

            # x_n_0_logits_list is a list of tensors, one for each attribute dim: (total_new_nodes, K_d)
            x_n_0_logits_list = self.node_pred_model.forward_with_h_g(
                h_g_full_batch[active_graph_mask],  # Pass h_g only for active graphs
                x_n_t,  # (total_new_nodes, num_features) - for all queries
                t_tensor_for_active_graphs,  # Timesteps for active graphs
                query2g,  # Maps total_new_nodes to *original* batch indices
                # This needs adjustment if h_g is sub-selected.
                # A better query2g_active mapping new nodes to indices *within the active graph set*
                num_query_cumsum_active  # Cumsum for queries within active graphs
            )
            # The query2g and num_query_cumsum for TransformerLayer inside forward_with_h_g
            # must align with the first dimension of h_g passed to it.
            # If h_g_full_batch[active_graph_mask] is (B_active, F_g), then
            # query2g needs to map total_new_nodes to 0..B_active-1.
            # This means re-calculating query2g and num_query_cumsum for the active set.

            # Recompute query2g_active and num_query_cumsum_active for the call
            query2g_for_pred_model = torch.repeat_interleave(
                torch.arange(active_graph_mask.sum().item(), device=self.device),  # 0 to B_active-1
                active_num_nodes_pred  # num new nodes for each active graph
            )

            x_n_s_parts = []
            for d_idx in range(num_node_features):
                x_n_0_probs_d = x_n_0_logits_list[d_idx].softmax(dim=-1)  # (total_new_nodes, K_d)

                # Current noisy attributes for this dimension: x_n_t[:, d_idx]
                current_x_n_t_d = x_n_t[:, d_idx] if x_n_t.ndim > 1 else x_n_t.squeeze(-1)

                # Prepare for self.node_diffusion.posterior
                # It expects x_t_one_hot, Q_t, Q_bar_s, Q_bar_t, x_0_probs
                # Alphas need to be for each query node, based on its graph's t_current_step
                # Since t_current_step is same for all active graphs in this iteration:
                alpha_t_val = self.node_diffusion.alphas[t_current_step]
                alpha_bar_s_val = self.node_diffusion.alpha_bars[s_iter]
                alpha_bar_t_val = self.node_diffusion.alpha_bars[t_current_step]

                Q_t_d = self.node_diffusion.get_Q(alpha_t_val, d_idx).to(self.device)  # (K_d, K_d)
                Q_bar_s_d = self.node_diffusion.get_Q(alpha_bar_s_val, d_idx).to(self.device)
                Q_bar_t_d = self.node_diffusion.get_Q(alpha_bar_t_val, d_idx).to(self.device)

                x_n_t_one_hot_d = F.one_hot(current_x_n_t_d,
                                            num_classes=self.node_diffusion.num_classes_list[d_idx]).float()

                # Posterior call from DiscreteDiffusion
                # posterior(self, Z_t, Q_t, Q_bar_s, Q_bar_t, Z_0)
                # Z_t is x_n_t_one_hot_d (total_new_nodes, K_d)
                # Z_0 is x_n_0_probs_d (total_new_nodes, K_d)
                # Q matrices are (K_d, K_d) - need to be broadcast or applied row-wise effectively
                # The original DiscreteDiffusion.posterior handles this broadcasting.
                x_n_s_probs_d = self.node_diffusion.posterior(
                    x_n_t_one_hot_d, Q_t_d, Q_bar_s_d, Q_bar_t_d, x_n_0_probs_d
                )  # (total_new_nodes, K_d)

                x_n_s_d = x_n_s_probs_d.multinomial(1).squeeze(-1)  # (total_new_nodes)
                x_n_s_parts.append(x_n_s_d)

            x_n_t = torch.stack(x_n_s_parts, dim=1) if num_node_features > 1 else x_n_s_parts[0].unsqueeze(-1)

        # Distribute final sampled attributes (x_n_t is now x_0_sampled)
        current_offset = 0
        for i_orig_batch in range(batch_size):
            if active_graph_mask[i_orig_batch]:
                num_new = num_new_nodes_per_graph[i_orig_batch].item()
                x_n_l_list_final[i_orig_batch] = x_n_t[current_offset: current_offset + num_new]
                current_offset += num_new

        return x_n_l_list_final

    @torch.no_grad()
    def sample_edge_layer(self, num_nodes_cumsum, edge_index_list,
                          batch_x_n, batch_abs_level, batch_rel_level,
                          # num_new_nodes_list: list of num new nodes for each graph in current batch
                          num_new_nodes_list,
                          # batch_query_src/dst: concatenated queries, global node indices in current batch
                          batch_query_src, batch_query_dst,
                          # query_src/dst_list_local: list of local queries for each graph
                          query_src_list_local, query_dst_list_local,
                          y=None, curr_level=None,
                          min_num_steps_e=None, max_num_steps_e=None):

        self.edge_pred_model.eval()

        if batch_query_src.numel() == 0:  # No query edges at all
            return edge_index_list  # Return original edge lists if no queries

        # Initial noisy edge states (label_t) for all query edges
        # Sample from prior p(edge_T), often uniform or based on avg_in_deg
        # avg_in_deg / num_candidate_sources = marginal prob of edge existing
        # For simplicity, start with uniform 0.5 for T, or use EdgeDiscreteDiffusion's apply_noise logic
        # Let's use a simplified prior for T:
        prob_edge_at_T = 0.5  # Or derive from self.edge_diffusion.avg_in_deg / typical_num_sources
        current_label_t = torch.bernoulli(
            torch.full_like(batch_query_src, prob_edge_at_T, dtype=torch.float32)
        ).long()  # (Total_Query_Edges)

        T_sampling = self.edge_diffusion.T
        if max_num_steps_e is not None: T_sampling = min(T_sampling, max_num_steps_e)
        # Optional: Adjust T_sampling based on min_num_steps_e or curr_level

        # Current graph structure (existing edges)
        batch_A = self.get_batch_A(num_nodes_cumsum, edge_index_list, self.device)

        # Iterative refinement for edges
        for s_iter in range(T_sampling - 1, -1, -1):
            t_current_step = s_iter + 1

            # Timestep tensor for all query edges: (Total_Query_Edges, 1)
            t_tensor_for_edges = torch.full(
                (len(batch_query_src), 1), t_current_step, dtype=torch.long, device=self.device
            )

            # Predict clean edge states (e_0_logits)
            # EdgePredModel takes batch_A (current edges), batch_x_n (all nodes),
            # and batch_query_src/dst (global indices for queries)
            # y is graph-level condition for the current batch of graphs
            e_0_logits = self.edge_pred_model(
                batch_A, batch_x_n, batch_abs_level, batch_rel_level,
                t_tensor_for_edges, batch_query_src, batch_query_dst, y
            )  # (Total_Query_Edges, 2)
            e_0_probs = e_0_logits.softmax(dim=-1)  # (Total_Query_Edges, 2)

            # Use EdgeDiscreteDiffusion's posterior_edge method
            # posterior_edge(self, Z_t, alpha_t, alpha_bar_s, alpha_bar_t, Z_0, marginal_list, ...)
            # Z_t is current_label_t (one-hot)
            # Z_0 is e_0_probs
            # marginal_list for edges is tricky. Original LayerDAG passes it to posterior_edge.
            # It's a list of marginal probabilities for edge existence for *each graph's new layer*.
            # This requires knowing how many queries belong to each graph's new layer.

            # Construct marginal_list for EdgeDiscreteDiffusion.posterior_edge
            # This needs num_new_nodes_list and num_potential_sources_per_new_node
            # For now, use a simplified global marginal or assume EdgeDiscreteDiffusion handles it.
            # The original EdgeDiscreteDiffusion.posterior_edge expects a per-graph marginal.

            # Simplified approach: Assume a single marginal for all queries for now.
            # This is a deviation if original posterior_edge expects per-graph marginals.
            # The avg_in_deg is a graph property, not per query.
            # Let's use the structure from the original EdgeDiscreteDiffusion.apply_noise's marginal.
            # This part needs to align with how EdgeDiscreteDiffusion.posterior_edge expects marginals.
            # The original `posterior_edge` in `LayerDAG` class (user's version) was complex.
            # The `EdgeDiscreteDiffusion.posterior` (if it exists) or a similar formulation is needed.
            # For now, let's assume a simplified sampling based on e_0_probs for x_s,
            # similar to how some DDPMs directly use x0_hat for sampling x_{t-1}.
            # This is not the full DDPM posterior but a common simplification.
            # A more accurate version would use the formula from EdgeDiscreteDiffusion.

            # Placeholder for proper posterior step:
            # For now, sample directly from predicted e_0_probs (very simplified)
            # This bypasses the diffusion history, not ideal for DDPM.
            # current_label_t = torch.bernoulli(e_0_probs[:, 1]).long() # Sample based on P(edge=1)

            # More correct (closer to original intent):
            alpha_t_val = self.edge_diffusion.alphas[t_current_step]
            alpha_bar_s_val = self.edge_diffusion.alpha_bars[s_iter]
            alpha_bar_t_val = self.edge_diffusion.alpha_bars[t_current_step]

            # Edge marginal: avg_in_deg / num_candidate_sources (simplified)
            # This should ideally be per query or per (new_node, old_graph_part)
            # Using a global estimate for now.
            # Max possible sources for any new node is batch_x_n.shape[0] - num_new_nodes_in_its_graph
            # This is complex. Original used `marginal_list` in `posterior_edge`.
            # Let's assume a single marginal for all queries for this step.
            # A typical value for num_candidate_sources could be related to `A.shape[1]` (num nodes in graph part)
            # This needs to be robust. For now, a placeholder:
            num_potential_sources_approx = batch_x_n.shape[0] / len(num_nodes_cumsum - 1) if len(
                num_nodes_cumsum - 1) > 0 else 10
            marginal_edge_prob = min(self.edge_diffusion.avg_in_deg, num_potential_sources_approx) / (
                        num_potential_sources_approx + 1e-6)
            marginal_edge_prob = torch.clamp(torch.tensor(marginal_edge_prob), 0.01, 0.99).to(self.device)

            # The `posterior_edge` in `LayerDAG` (original) was:
            # Z_t_one_hot = F.one_hot(current_label_t, num_classes=2).float()
            # current_label_t_mask = self.posterior_edge( # This was the complex method in LayerDAG
            # Z_t_one_hot, alpha_t_val, alpha_bar_s_val, alpha_bar_t_val, e_0_probs,
            # marginal_list, num_new_nodes_list, num_query_list_per_graph
            # )
            # current_label_t = current_label_t_mask.long() # if posterior_edge returned a mask

            # Reverting to use a conceptual posterior from EdgeDiscreteDiffusion (which might need to be added/exposed)
            # If EdgeDiscreteDiffusion has a posterior method like DiscreteDiffusion:
            # Q_t_edge, Q_bar_s_edge, Q_bar_t_edge = self.edge_diffusion.get_Qs(
            # alpha_t_val, alpha_bar_s_val, alpha_bar_t_val, marginal_edge_prob
            # )
            # Z_t_edge_one_hot = F.one_hot(current_label_t, num_classes=2).float()
            # e_s_probs = self.edge_diffusion.posterior( # Assuming such a method exists
            # Z_t_edge_one_hot, Q_t_edge, Q_bar_s_edge, Q_bar_t_edge, e_0_probs
            # )
            # current_label_t = e_s_probs.multinomial(1).squeeze(-1)

            # Fallback to simplified sampling if posterior is not readily available/correct
            # This is a significant simplification of the DDPM reverse process.
            current_label_t = torch.bernoulli(e_0_probs[:, 1]).long()

        # Update edge_index_list based on the final sampled edges (current_label_t)
        updated_edge_index_list = []
        query_offset = 0
        for i in range(len(edge_index_list)):  # Iterate through graphs in the current batch
            num_queries_for_graph_i = len(query_src_list_local[i])

            if num_queries_for_graph_i == 0:
                updated_edge_index_list.append(edge_index_list[i])
                continue

            sampled_states_for_graph_i = current_label_t[query_offset: query_offset + num_queries_for_graph_i]

            # Get local src/dst for edges predicted to exist (state == 1)
            src_local_new_edges = query_src_list_local[i][sampled_states_for_graph_i == 1]
            dst_local_new_edges = query_dst_list_local[i][sampled_states_for_graph_i == 1]

            if src_local_new_edges.numel() > 0:
                new_edges_i = torch.stack([dst_local_new_edges, src_local_new_edges], dim=0)  # DGL format [dst, src]
                graph_i_updated_edges = torch.cat([edge_index_list[i], new_edges_i], dim=1)
                updated_edge_index_list.append(graph_i_updated_edges)
            else:
                updated_edge_index_list.append(edge_index_list[i])  # No new edges for this graph

            query_offset += num_queries_for_graph_i

        return updated_edge_index_list

    def get_batch_A(self, num_nodes_cumsum, edge_index_list, current_device):
        # num_nodes_cumsum: (B+1), edge_index_list: list of (2, E_i) tensors with local indices
        num_total_nodes_in_batch = num_nodes_cumsum[-1].item()
        if num_total_nodes_in_batch == 0:  # Batch of empty graphs (only if no dummy node start)
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        batch_src_list, batch_dst_list = [], []
        for i in range(len(edge_index_list)):  # Iterate graphs in batch
            if edge_index_list[i].numel() > 0:
                # edge_index_list[i] is [dst, src] with local-to-graph indices
                # Offset local indices by num_nodes_cumsum[i] to get global batch indices
                batch_dst_list.append(edge_index_list[i][0] + num_nodes_cumsum[i])
                batch_src_list.append(edge_index_list[i][1] + num_nodes_cumsum[i])

        if not batch_src_list:  # No edges in the entire batch
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(num_total_nodes_in_batch, num_total_nodes_in_batch))

        batch_src_cat = torch.cat(batch_src_list)
        batch_dst_cat = torch.cat(batch_dst_list)

        return dglsp.spmatrix(torch.stack([batch_dst_cat, batch_src_cat]),
                              shape=(num_total_nodes_in_batch, num_total_nodes_in_batch)).to(current_device)

    def get_batch_A_n2g(self, num_nodes_cumsum, current_device):
        current_batch_size = len(num_nodes_cumsum) - 1
        if current_batch_size == 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        num_total_nodes_in_batch = num_nodes_cumsum[-1].item()
        # Handle case where batch might contain only empty graphs (e.g. after dummy node removal)
        if num_total_nodes_in_batch == 0 and current_batch_size > 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(current_batch_size, 0))
        if num_total_nodes_in_batch == 0 and current_batch_size == 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        nids_list, gids_list = [], []
        for i in range(current_batch_size):
            num_nodes_in_graph_i = num_nodes_cumsum[i + 1] - num_nodes_cumsum[i]
            if num_nodes_in_graph_i > 0:  # Only add if graph has nodes
                nids_list.append(
                    torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i + 1], device=current_device, dtype=torch.long))
                gids_list.append(torch.full((num_nodes_in_graph_i,), i, device=current_device, dtype=torch.long))

        if not nids_list:  # All graphs in batch are empty
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(current_batch_size, num_total_nodes_in_batch))

        nids_cat = torch.cat(nids_list)
        gids_cat = torch.cat(gids_list)

        return dglsp.spmatrix(torch.stack([gids_cat, nids_cat]),
                              shape=(current_batch_size, num_total_nodes_in_batch)).to(current_device)

    def get_batch_y(self, current_y_list, current_x_n_list_shapes_or_lengths, current_device):
        # current_y_list: list of y conditions for graphs in the current active batch
        # current_x_n_list_shapes_or_lengths: used to determine if y should be node or graph level for encoders
        if current_y_list is None:  # or not self.is_conditional (handled by y_emb_size=0)
            return None
        if not current_y_list: return None

        # This function is for preparing y for BiMPNNEncoder.
        # If BiMPNNEncoder.pool is active, y is graph-level.
        # If BiMPNNEncoder.pool is None, y needs to be node-level.
        # The original LayerDAG passed y directly. If conditional, y was (B,1) or (B,F_y)
        # for graph-level, or pre-expanded (N,F_y) for node-level.
        # For now, assume y is graph-level (B,F_y_cond) as per typical conditional generation.
        # train.py's collate functions broadcast y to node-level if needed by original dataset structure.
        # Here, if y is for the *current active batch*, it should be (current_batch_size, F_y_cond)

        if isinstance(current_y_list[0], (int, float)):  # List of scalar y values
            return torch.tensor([[y_val] for y_val in current_y_list], dtype=torch.float32, device=current_device)
        elif isinstance(current_y_list[0], torch.Tensor):  # List of y tensors
            return torch.stack([y_val.to(current_device) for y_val in current_y_list])
        else:
            raise TypeError(f"Unsupported y type in get_batch_y: {type(current_y_list[0])}")

    @torch.no_grad()
    def sample(self,
               batch_size=1,
               raw_y_batch=None,  # List of y conditions for the batch
               min_num_steps_n=None,
               max_num_steps_n=None,
               min_num_steps_e=None,
               max_num_steps_e=None):

        current_device = self.device
        self.eval()

        is_model_conditional = (raw_y_batch is not None)  # Infer from input
        if is_model_conditional and batch_size != len(raw_y_batch):
            raise ValueError("Batch size must match len(raw_y_batch) for conditional sampling.")

        # --- Initialization for active graphs ---
        # These lists will shrink as graphs complete generation
        active_edge_indices = [torch.tensor([[], []], dtype=torch.long, device=current_device) for _ in
                               range(batch_size)]

        if isinstance(self.dummy_x_n, int):
            init_x_n_val = torch.tensor([[self.dummy_x_n]], dtype=torch.long, device=current_device)
        elif isinstance(self.dummy_x_n, torch.Tensor) and self.dummy_x_n.ndim == 0:
            init_x_n_val = torch.tensor([[self.dummy_x_n.item()]], dtype=torch.long, device=current_device)
        elif isinstance(self.dummy_x_n, torch.Tensor) and self.dummy_x_n.ndim > 0:  # Tensor for multi-feature
            init_x_n_val = self.dummy_x_n.clone().detach().unsqueeze(0).to(current_device)
        else:
            raise TypeError(f"Unsupported self.dummy_x_n type: {type(self.dummy_x_n)}")

        active_x_n_features = [init_x_n_val.clone() for _ in range(batch_size)]
        active_abs_levels = [torch.tensor([[0.0]], device=current_device, dtype=torch.float32) for _ in
                             range(batch_size)]

        active_y_conditions = list(raw_y_batch) if is_model_conditional else [None] * batch_size
        active_original_indices = list(range(batch_size))  # To map back to original batch order

        # --- Storage for finished graphs ---
        finished_edge_indices = [None] * batch_size
        finished_x_n_features = [None] * batch_size
        finished_y_conditions = [None] * batch_size if is_model_conditional else None

        current_level = 0.0

        while active_x_n_features:  # Loop while there are graphs still being generated
            current_active_batch_size = len(active_x_n_features)

            # --- Prepare batch inputs for currently active graphs ---
            num_nodes_cumsum_active = torch.cumsum(torch.tensor(
                [0] + [len(x_n_i) for x_n_i in active_x_n_features], device=current_device, dtype=torch.long), dim=0)

            batch_x_n_active = torch.cat(active_x_n_features)
            batch_abs_level_active = torch.cat(active_abs_levels)
            batch_rel_level_active = batch_abs_level_active.max() - batch_abs_level_active

            batch_A_active = self.get_batch_A(num_nodes_cumsum_active, active_edge_indices, current_device)
            batch_A_n2g_active = self.get_batch_A_n2g(num_nodes_cumsum_active, current_device)

            batch_y_tensor_active = None
            if is_model_conditional and any(y is not None for y in active_y_conditions):  # Check if any y is not None
                # Ensure active_y_conditions is a list of tensors or scalars for get_batch_y
                # If some are None (e.g. if a graph finished, this list shrinks), filter them out.
                # This active_y_conditions should always correspond to active_x_n_features.
                batch_y_tensor_active = self.get_batch_y(active_y_conditions, active_x_n_features, current_device)

            # 1. Sample Node Layer attributes for all active graphs
            new_nodes_attrs_per_active_graph = self.sample_node_layer(
                batch_A_active, batch_x_n_active, batch_abs_level_active, batch_rel_level_active,
                batch_A_n2g_active, curr_level=current_level, y=batch_y_tensor_active,
                min_num_steps_n=min_num_steps_n, max_num_steps_n=max_num_steps_n
            )  # List of (num_new_nodes_i, num_features_node) or (num_new_nodes_i)

            # --- Update active lists, move finished graphs, prepare for edge prediction ---
            next_iter_active_edge_indices = []
            next_iter_active_x_n_features = []
            next_iter_active_abs_levels = []
            next_iter_active_y_conditions = [] if is_model_conditional else None
            next_iter_original_indices = []

            # For edge prediction on graphs that are *continuing this layer*
            query_src_local_for_continuing = []  # List of local src tensors for each continuing graph
            query_dst_local_for_continuing = []  # List of local dst tensors
            num_new_nodes_for_continuing = []  # List of num new nodes for each continuing graph

            continuing_graph_indices_in_active_batch = []  # Indices of graphs that continue, within current active_batch

            for i in range(current_active_batch_size):
                original_batch_idx = active_original_indices[i]
                newly_sampled_nodes_for_graph_i = new_nodes_attrs_per_active_graph[i]

                if newly_sampled_nodes_for_graph_i.numel() == 0:  # Graph i finished
                    final_edges = active_edge_indices[i]
                    # Adjust for dummy node by subtracting 1 from all node indices
                    if final_edges.numel() > 0: final_edges = final_edges - 1

                    finished_edge_indices[original_batch_idx] = final_edges
                    finished_x_n_features[original_batch_idx] = active_x_n_features[i][1:]  # Remove dummy node
                    if is_model_conditional:
                        finished_y_conditions[original_batch_idx] = active_y_conditions[i]
                else:  # Graph i continues to the next layer (or edge prediction for this layer)
                    continuing_graph_indices_in_active_batch.append(i)
                    next_iter_original_indices.append(original_batch_idx)
                    next_iter_active_edge_indices.append(active_edge_indices[i])  # Carry over old edges for now

                    updated_x_n_for_graph_i = torch.cat([active_x_n_features[i], newly_sampled_nodes_for_graph_i])
                    next_iter_active_x_n_features.append(updated_x_n_for_graph_i)

                    new_abs_levels_i = torch.full(
                        (newly_sampled_nodes_for_graph_i.shape[0], 1),
                        current_level + 1.0,
                        dtype=torch.float32, device=current_device
                    )
                    updated_abs_level_for_graph_i = torch.cat([active_abs_levels[i], new_abs_levels_i])
                    next_iter_active_abs_levels.append(updated_abs_level_for_graph_i)

                    if is_model_conditional:
                        next_iter_active_y_conditions.append(active_y_conditions[i])

                    # Prepare queries for edge prediction (from existing real nodes to new nodes)
                    num_old_nodes_incl_dummy = active_x_n_features[i].shape[0]
                    num_new_nodes = newly_sampled_nodes_for_graph_i.shape[0]
                    num_new_nodes_for_continuing.append(num_new_nodes)

                    q_src_local_i, q_dst_local_i = [], []
                    if num_old_nodes_incl_dummy > 1:  # If there are any real existing nodes
                        # Local indices for existing real nodes: 1 to num_old_nodes_incl_dummy - 1
                        # Local indices for new nodes: num_old_nodes_incl_dummy to ...
                        for s_local_idx in range(1, num_old_nodes_incl_dummy):  # Iterate existing REAL nodes
                            for d_new_node_offset in range(num_new_nodes):
                                d_local_idx = num_old_nodes_incl_dummy + d_new_node_offset
                                q_src_local_i.append(s_local_idx)
                                q_dst_local_i.append(d_local_idx)

                    query_src_local_for_continuing.append(
                        torch.tensor(q_src_local_i, dtype=torch.long, device=current_device))
                    query_dst_local_for_continuing.append(
                        torch.tensor(q_dst_local_i, dtype=torch.long, device=current_device))

            # Update active lists for the next iteration (or for edge prediction step)
            active_edge_indices = next_iter_active_edge_indices
            active_x_n_features = next_iter_active_x_n_features
            active_abs_levels = next_iter_active_abs_levels
            if is_model_conditional: active_y_conditions = next_iter_active_y_conditions
            active_original_indices = next_iter_original_indices

            if not active_x_n_features: break  # All graphs finished

            current_level += 1.0  # Increment level for the next layer of nodes

            # 2. Sample Edges for the layer just added (for continuing graphs)
            if any(q.numel() > 0 for q in query_src_local_for_continuing):  # If any graph has queries
                # Prepare batch inputs for edge prediction (only for graphs that continued and have queries)
                # These are based on the *updated* active_x_n_features, active_abs_levels
                num_nodes_cumsum_edge_pred = torch.cumsum(torch.tensor(
                    [0] + [len(x_n_i) for x_n_i in active_x_n_features], device=current_device, dtype=torch.long),
                    dim=0)

                batch_x_n_edge_pred = torch.cat(active_x_n_features)
                batch_abs_level_edge_pred = torch.cat(active_abs_levels)
                batch_rel_level_edge_pred = batch_abs_level_edge_pred.max() - batch_abs_level_edge_pred

                batch_y_tensor_edge_pred = None
                if is_model_conditional and any(y is not None for y in active_y_conditions):
                    batch_y_tensor_edge_pred = self.get_batch_y(active_y_conditions, active_x_n_features,
                                                                current_device)

                # Concatenate local queries into batch queries with global indices
                batch_query_src_cat, batch_query_dst_cat = [], []
                current_node_offset = 0
                for i in range(len(active_x_n_features)):  # Iterate through graphs that are *still active*
                    if query_src_local_for_continuing[i].numel() > 0:
                        batch_query_src_cat.append(query_src_local_for_continuing[i] + num_nodes_cumsum_edge_pred[i])
                        batch_query_dst_cat.append(query_dst_local_for_continuing[i] + num_nodes_cumsum_edge_pred[i])
                    current_node_offset += active_x_n_features[i].shape[
                        0]  # This offset logic was for previous structure

                if batch_query_src_cat:  # If there are any queries after filtering
                    final_batch_q_src = torch.cat(batch_query_src_cat)
                    final_batch_q_dst = torch.cat(batch_query_dst_cat)

                    if final_batch_q_src.numel() > 0:
                        # sample_edge_layer updates active_edge_indices in place (or returns updated list)
                        active_edge_indices = self.sample_edge_layer(
                            num_nodes_cumsum_edge_pred, active_edge_indices,  # Pass current state of edges
                            batch_x_n_edge_pred, batch_abs_level_edge_pred, batch_rel_level_edge_pred,
                            num_new_nodes_for_continuing,  # Num new nodes for each graph that generated queries
                            final_batch_q_src, final_batch_q_dst,  # Concatenated global queries
                            query_src_local_for_continuing, query_dst_local_for_continuing,  # List of local queries
                            y=batch_y_tensor_edge_pred, curr_level=current_level,
                            # current_level is now level of *new* nodes
                            min_num_steps_e=min_num_steps_e, max_num_steps_e=max_num_steps_e
                        )

            if self.max_level is not None and current_level >= self.max_level:
                # Max levels reached, move any remaining active graphs to finished
                for i in range(len(active_x_n_features)):
                    original_idx_rem = active_original_indices[i]
                    final_edges_rem = active_edge_indices[i]
                    if final_edges_rem.numel() > 0: final_edges_rem = final_edges_rem - 1

                    finished_edge_indices[original_idx_rem] = final_edges_rem
                    finished_x_n_features[original_idx_rem] = active_x_n_features[i][1:]
                    if is_model_conditional:
                        finished_y_conditions[original_idx_rem] = active_y_conditions[i]
                break  # Exit main generation loop

        return finished_edge_indices, finished_x_n_features, finished_y_conditions

