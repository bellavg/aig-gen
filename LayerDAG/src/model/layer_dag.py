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
        if A.nnz == 0:  # Handle case with no edges
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
            x_n_emb = torch.cat(x_n_emb_parts, dim=1)  # This was sum, should be cat if hidden_size is per feature
            # If hidden_size is total, then sum is okay.
            # The BiMPNNEncoder's actual_x_n_dim_after_embedding
            # assumes concatenation (len(num_x_n_cat) * x_n_emb_size)
            # So, this should be torch.cat
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
                 y_emb_size=0,  # This is the y_emb_size from config, potentially overridden if unconditional
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
        self.effective_y_emb_size = y_emb_size if self.y_emb is not None else 0

        proj_input_dim = actual_x_n_dim_after_embedding
        if self.level_emb is not None: proj_input_dim += pe_emb_size

        # Only add y_emb_size to proj_input_dim if self.y_emb is actually created
        if self.y_emb is not None:  # This check is based on the y_emb_size passed to __init__
            proj_input_dim += self.effective_y_emb_size

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
            self.bn_g = nn.BatchNorm1d(hidden_size)  # BN on graph embedding h_g
            # If y is concatenated to h_g, this size might need adjustment
            # or BN applied before y concat.
            # Original paper doesn't explicitly show y concat with h_g before BN.

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

        # Crucial part: only append y_embedding if y is provided AND self.y_emb was initialized
        if y is not None and self.y_emb is not None:
            h_y_processed = self.y_emb(y)

            if self.pool is None:  # Node-level task, y should be node-level or expandable
                if h_y_processed.shape[0] == x_n.shape[0]:  # If y is already node-level
                    h_n_parts.append(h_y_processed)
                elif A_n2g is not None and h_y_processed.shape[0] == A_n2g.shape[0]:  # y is graph-level, expand
                    # A_n2g maps nodes to graphs. gids = A_n2g.coo()[0]
                    # This assumes A_n2g is (B, N_total_batch)
                    # A more robust way if A_n2g is [gids, nids] from collate_common:
                    # graph_indices_for_nodes = A_n2g[0, torch.sort(A_n2g[1]).indices] # if A_n2g is index tensor
                    # For dglsp.spmatrix, it's more complex.
                    # Assuming y is (B, Y_feat) and we need (N_total_batch, Y_feat)
                    # This expansion is typically done in the main model or data prep.
                    # For now, we'll rely on y being correctly shaped if self.pool is None.
                    # If y is (B,F_y) and A_n2g is (B,N), we need to find graph_idx for each node.
                    # This is often done by:
                    #   g_idx = torch.zeros(x_n.shape[0], dtype=torch.long, device=x_n.device)
                    #   for i_batch in range(A_n2g.shape[0]): # B
                    #       nodes_in_graph_i_mask = (A_n2g.to_dense()[i_batch] == 1) # if A_n2g is BxN indicator
                    #       g_idx[nodes_in_graph_i_mask] = i_batch
                    #   h_y_expanded = h_y_processed[g_idx]
                    #   h_n_parts.append(h_y_expanded)
                    # This logic is complex for a general encoder.
                    # The original paper implies y is handled by the main LayerDAG model structure.
                    # If y is (B,F_y) and pool is None, it's an issue unless y_emb_size was 0.
                    pass  # Avoid appending if y is graph-level and no pooling
            # If pooling, y is graph-level. It might be concatenated *after* pooling.
            # The current proj_input_dim calculation assumes y is part of node features *before* MPNN if y_emb_size > 0.

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

            # If y was graph-level and self.y_emb exists (meaning y_emb_size > 0 during init)
            # And y was provided to forward.
            # The paper's Fig 5 suggests y is incorporated into context h_t_G<=l.
            # Here, if pooling, y is graph-level. It could be concatenated to h_g.
            if y is not None and self.y_emb is not None and h_y_processed.shape[0] == h_g.shape[0]:
                # This changes output dim of BiMPNNEncoder if y is used this way.
                # GraphClassifier's emb_size would need to account for this.
                # For now, let's assume y is handled by GraphClassifier if pooled.
                pass

            return self.bn_g(h_g)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool}")


class GraphClassifier(nn.Module):
    def __init__(self,
                 graph_encoder,
                 emb_size,  # This should be the output size of graph_encoder (its hidden_size)
                 # Potentially hidden_size + y_emb_size if y is concatenated after pooling
                 num_classes):
        super().__init__()

        self.graph_encoder = graph_encoder
        # If y is concatenated to h_g *after* BiMPNNEncoder's pooling and BN,
        # then emb_size here should be graph_encoder.hidden_size + actual_y_emb_size_used
        # For simplicity, assume emb_size passed here is the final dimension before predictor.
        self.predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, y=None):
        # y is graph-level condition (e.g. (B,1) or (B,F_y))
        # BiMPNNEncoder's `pool` should be active (e.g. 'sum')
        h_g = self.graph_encoder(A, x_n, abs_level, rel_level, y=y, A_n2g=A_n2g)

        # If y was meant to be combined here (after pooling and BN in encoder):
        # if y is not None and self.graph_encoder.y_emb is not None: # Check if encoder has y_emb
        #    h_y_graph = self.graph_encoder.y_emb(y)
        #    if h_y_graph.shape[0] == h_g.shape[0]:
        #        h_g = torch.cat([h_g, h_y_graph], dim=-1)
        # This would require emb_size for self.predictor to be pre-calculated correctly.

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
            if q is not None and q.shape[0] == 0: return q
            feat_dim = q.shape[-1] if q is not None and q.ndim > 1 else (
                v.shape[-1] if v is not None and v.ndim > 1 else 0)
            return torch.empty(0, feat_dim, device=q.device if q is not None else (
                v.device if v is not None else num_query_cumsum.device))

        q_padded = q.new_zeros(batch_size, max_num_nodes, q.shape[-1])
        k_padded = k.new_zeros(batch_size, max_num_nodes, k.shape[-1])
        v_padded = v.new_zeros(batch_size, max_num_nodes, v.shape[-1])
        pad_mask = q.new_zeros(batch_size, max_num_nodes, dtype=torch.bool)

        for i in range(batch_size):
            start_idx, end_idx = num_query_cumsum[i], num_query_cumsum[i + 1]
            num_q_i = num_query_nodes[i].item()
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
            pad_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        attn_scores = F.softmax(dot, dim=-1)
        h_n_padded = torch.matmul(attn_scores, v_padded)
        h_n_padded = rearrange(h_n_padded, 'b h n d -> (b n) (h d)')

        unpad_mask = (~pad_mask).reshape(-1)
        return h_n_padded[unpad_mask]

    def forward(self, h_n, num_query_cumsum):
        if h_n.shape[0] == 0:
            return h_n

        v_n = self.to_v(h_n)
        q_n, k_n = self.to_qk(h_n).chunk(2, dim=-1)

        h_n_new = self.attn(q_n, k_n, v_n, num_query_cumsum)

        if h_n_new.shape[0] != h_n.shape[0]:
            print(
                f"WARNING: TransformerLayer shape mismatch or empty h_n_new. h_n: {h_n.shape}, h_n_new: {h_n_new.shape}. Skipping update.")
            return h_n

        h_n = self.norm1(h_n + self.proj_new(h_n_new))
        h_n = self.norm2(h_n + self.out_proj(h_n))
        return h_n


class NodePredModel(nn.Module):
    def __init__(self,
                 graph_encoder,
                 num_x_n_cat,
                 x_n_emb_size,
                 t_emb_size,
                 in_hidden_size,
                 out_hidden_size,
                 num_transformer_layers,
                 num_heads,
                 dropout):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)
        self.t_emb = SinusoidalPE(t_emb_size)

        if isinstance(num_x_n_cat, int) or (isinstance(num_x_n_cat, torch.Tensor) and num_x_n_cat.ndim == 0):
            actual_x_n_t_dim_embedded = x_n_emb_size
        elif isinstance(num_x_n_cat, (list, tuple, torch.Tensor)):
            actual_x_n_t_dim_embedded = len(num_x_n_cat) * x_n_emb_size
        else:
            raise TypeError(f"Unhandled type for num_x_n_cat in NodePredModel: {type(num_x_n_cat)}")

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
                nn.Linear(out_hidden_size, int(num_classes_f))
            ))

    def forward_with_h_g(self, h_g, x_n_t,
                         t, query2g, num_query_cumsum):
        if x_n_t.shape[0] == 0:
            return [torch.empty(0, device=h_g.device) for _ in self.pred_list]

        h_t_graph = self.t_emb(t)
        h_g_cond = torch.cat([h_g, h_t_graph], dim=1)


        h_n_t_attrs = self.x_n_emb(x_n_t)
        h_g_cond_expanded = h_g_cond[query2g]

        # ---- ADD THIS DEBUG CHECK ----
        if x_n_t.numel() > 0:  # Only check if x_n_t is not empty
            num_features = x_n_t.shape[1]
            # num_classes_list should be accessible here, e.g., self.num_x_n_cat (the one used for this embedding)
            # Assuming self.x_n_emb was initialized with num_actual_node_cats_for_pred
            # You might need to access self.node_diffusion.num_classes_list if it's not stored directly
            # For this example, let's assume self.num_x_n_cat used for self.x_n_emb is correct
            # The self.x_n_emb in NodePredModel has its own num_x_n_cat passed during init
            # This was: num_x_n_cat=num_actual_node_cats_for_pred in LayerDAG constructor
            # num_actual_node_cats_for_pred = torch.LongTensor(self.node_diffusion.num_classes_list)

            # Let's assume self.x_n_emb.emb_list[d].num_embeddings gives the K for feature d
            for d_idx in range(num_features):
                max_val_in_feature = x_n_t[:, d_idx].max().item()
                min_val_in_feature = x_n_t[:, d_idx].min().item()
                num_categories_for_feature = self.x_n_emb.emb_list[d_idx].num_embeddings
                if max_val_in_feature >= num_categories_for_feature:
                    print(
                        f"ERROR: Feature {d_idx} in x_n_t has max value {max_val_in_feature} >= num_categories {num_categories_for_feature}")
                    # Potentially raise an error or breakpoint
                if min_val_in_feature < 0:
                    print(f"ERROR: Feature {d_idx} in x_n_t has min value {min_val_in_feature} < 0")
                    # Potentially raise an error or breakpoint
        # ---- END DEBUG CHECK ----

        h_n_t_combined = torch.cat([h_n_t_attrs, h_g_cond_expanded], dim=1)
        h_n_t = self.project_h_n(h_n_t_combined)

        for trans_layer in self.trans_layers:
            if h_n_t.shape[0] == 0: break
            h_n_t = trans_layer(h_n_t, num_query_cumsum)

        pred = []
        if h_n_t.shape[0] > 0:
            for d_idx in range(len(self.pred_list)):
                pred.append(self.pred_list[d_idx](h_n_t))
        else:
            pred = [torch.empty(0, device=h_g.device) for _ in self.pred_list]
        return pred

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, x_n_t,
                t, query2g, num_query_cumsum, y=None):
        h_g = self.graph_encoder(A, x_n, abs_level,
                                 rel_level, y=y, A_n2g=A_n2g)
        return self.forward_with_h_g(h_g, x_n_t, t, query2g,
                                     num_query_cumsum)


class EdgePredModel(nn.Module):
    def __init__(self,
                 graph_encoder,
                 t_emb_size,
                 in_hidden_size,
                 out_hidden_size):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.t_emb = SinusoidalPE(t_emb_size)
        mlp_input_dim = 2 * in_hidden_size + t_emb_size
        self.pred = nn.Sequential(
            nn.Linear(mlp_input_dim, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, 2)
        )

    def forward(self, A, x_n, abs_level, rel_level, t,
                query_src, query_dst, y=None):
        h_n = self.graph_encoder(A, x_n, abs_level, rel_level, y=y, A_n2g=None)

        if query_src.numel() == 0:
            return torch.empty(0, 2, device=h_n.device if h_n.numel() > 0 else t.device)

        h_t_edge = self.t_emb(t)

        h_e_parts = []
        if h_t_edge.shape[1] > 0: h_e_parts.append(h_t_edge)
        h_e_parts.append(h_n[query_src])
        h_e_parts.append(h_n[query_dst])

        h_e = torch.cat(h_e_parts, dim=-1)
        return self.pred(h_e)


class LayerDAG(nn.Module):
    def __init__(self,
                 device,
                 num_x_n_cat,
                 node_count_encoder_config,
                 max_layer_size,
                 node_diffusion,
                 node_pred_graph_encoder_config,
                 node_predictor_config,
                 edge_diffusion,
                 edge_pred_graph_encoder_config,
                 edge_predictor_config,
                 is_model_conditional,  # ADDED: Flag to indicate overall model conditionality
                 max_level=None):
        super().__init__()

        self.device = torch.device(device)
        self.is_model_conditional = is_model_conditional  # Store the flag

        if isinstance(num_x_n_cat, int):
            self.num_x_n_cat_tensor = torch.LongTensor([num_x_n_cat]).to(self.device)
        elif isinstance(num_x_n_cat, torch.Tensor):
            self.num_x_n_cat_tensor = num_x_n_cat.clone().detach().to(self.device)
        elif isinstance(num_x_n_cat, (list, tuple)):
            self.num_x_n_cat_tensor = torch.LongTensor(num_x_n_cat).to(self.device)
        else:
            raise TypeError(f"num_x_n_cat must be an int, list, tuple or torch.Tensor, got {type(num_x_n_cat)}")

        if self.num_x_n_cat_tensor.ndim == 0 or len(self.num_x_n_cat_tensor) == 1:
            self.dummy_x_n = self.num_x_n_cat_tensor.item() - 1
        else:
            self.dummy_x_n = self.num_x_n_cat_tensor - 1

        # --- Node Count Model ---
        nc_config = node_count_encoder_config
        nc_y_emb_size = nc_config.get('y_emb_size', 0)
        if not self.is_model_conditional:  # If model is unconditional, force y_emb_size to 0
            nc_y_emb_size = 0

        node_count_encoder = BiMPNNEncoder(
            num_x_n_cat=self.num_x_n_cat_tensor,
            x_n_emb_size=nc_config['x_n_emb_size'],
            pe_emb_size=nc_config.get('pe_emb_size', 0),
            hidden_size=nc_config.get('hidden_size', nc_config['x_n_emb_size']),
            num_mpnn_layers=nc_config['num_mpnn_layers'],
            pe=nc_config.get('pe'),
            y_emb_size=nc_y_emb_size,  # Use potentially overridden y_emb_size
            pool=nc_config.get('pool', 'sum')
        ).to(self.device)
        self.node_count_model = GraphClassifier(
            node_count_encoder,
            emb_size=nc_config.get('hidden_size', nc_config['x_n_emb_size']),
            num_classes=max_layer_size + 1
        ).to(self.device)

        # --- Node Prediction Model ---
        self.node_diffusion = node_diffusion.to(self.device)
        np_ge_config = node_pred_graph_encoder_config
        np_pred_config = node_predictor_config

        np_ge_y_emb_size = np_ge_config.get('y_emb_size', 0)
        if not self.is_model_conditional:  # Override if unconditional
            np_ge_y_emb_size = 0

        node_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat=self.num_x_n_cat_tensor,
            x_n_emb_size=np_ge_config['x_n_emb_size'],
            pe_emb_size=np_ge_config.get('pe_emb_size', 0),
            hidden_size=np_ge_config.get('hidden_size', np_ge_config['x_n_emb_size']),
            num_mpnn_layers=np_ge_config['num_mpnn_layers'],
            pe=np_ge_config.get('pe'),
            y_emb_size=np_ge_y_emb_size,  # Use potentially overridden y_emb_size
            pool=np_ge_config.get('pool', 'sum')
        ).to(self.device)

        num_actual_node_cats_for_pred = torch.LongTensor(self.node_diffusion.num_classes_list).to(self.device)
        self.node_pred_model = NodePredModel(
            node_pred_graph_encoder,
            num_x_n_cat=num_actual_node_cats_for_pred,
            x_n_emb_size=np_ge_config['x_n_emb_size'],
            t_emb_size=np_pred_config['t_emb_size'],
            in_hidden_size=np_ge_config.get('hidden_size', np_ge_config['x_n_emb_size']),
            out_hidden_size=np_pred_config['out_hidden_size'],
            num_transformer_layers=np_pred_config['num_transformer_layers'],
            num_heads=np_pred_config['num_heads'],
            dropout=np_pred_config['dropout']
        ).to(self.device)

        # --- Edge Prediction Model ---
        self.edge_diffusion = edge_diffusion.to(self.device)
        ep_ge_config = edge_pred_graph_encoder_config
        ep_pred_config = edge_predictor_config

        ep_ge_y_emb_size = ep_ge_config.get('y_emb_size', 0)
        if not self.is_model_conditional:  # Override if unconditional
            ep_ge_y_emb_size = 0

        edge_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat=self.num_x_n_cat_tensor,
            x_n_emb_size=ep_ge_config['x_n_emb_size'],
            pe_emb_size=ep_ge_config.get('pe_emb_size', 0),
            hidden_size=ep_ge_config.get('hidden_size', ep_ge_config['x_n_emb_size']),
            num_mpnn_layers=ep_ge_config['num_mpnn_layers'],
            pe=ep_ge_config.get('pe'),
            y_emb_size=ep_ge_y_emb_size,  # Use potentially overridden y_emb_size
            pool=ep_ge_config.get('pool')
        ).to(self.device)
        self.edge_pred_model = EdgePredModel(
            edge_pred_graph_encoder,
            t_emb_size=ep_pred_config['t_emb_size'],
            in_hidden_size=ep_ge_config.get('hidden_size', ep_ge_config['x_n_emb_size']),
            out_hidden_size=ep_pred_config['out_hidden_size']
        ).to(self.device)

        self.max_level = max_level

    @torch.no_grad()
    def sample_node_layer(self, A, x_n, abs_level, rel_level, A_n2g,
                          curr_level, y=None,
                          min_num_steps_n=None, max_num_steps_n=None):

        self.node_count_model.eval()
        self.node_pred_model.eval()

        batch_size = A_n2g.shape[0]
        if batch_size == 0: return []

        node_count_logits = self.node_count_model(A, x_n, abs_level, rel_level, A_n2g=A_n2g, y=y)

        if curr_level == 0:
            node_count_logits[:, 0] = float('-inf')

        node_count_probs = node_count_logits.softmax(dim=-1)
        num_new_nodes_per_graph = node_count_probs.multinomial(1).squeeze(-1)
        x_n_l_list_final = [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(batch_size)]
        active_graph_mask = num_new_nodes_per_graph > 0
        if not active_graph_mask.any():
            return x_n_l_list_final

        total_new_nodes_to_sample = num_new_nodes_per_graph[active_graph_mask].sum().item()
        if total_new_nodes_to_sample == 0:
            return x_n_l_list_final

        query2g = torch.repeat_interleave(torch.arange(batch_size, device=self.device)[active_graph_mask],
                                          num_new_nodes_per_graph[active_graph_mask])
        active_num_nodes_pred = num_new_nodes_per_graph[active_graph_mask]
        num_query_cumsum_active = torch.cumsum(
            torch.cat([torch.tensor([0], device=self.device, dtype=torch.long), active_num_nodes_pred]),
            dim=0)

        num_node_features = len(self.node_diffusion.num_classes_list)
        x_n_t_parts = []
        for d_idx in range(num_node_features):
            prior_d = self.node_diffusion.m_list[d_idx][0, :self.node_diffusion.num_classes_list[d_idx]]
            x_n_t_d = prior_d.unsqueeze(0).expand(total_new_nodes_to_sample, -1).multinomial(1).squeeze(-1)
            x_n_t_parts.append(x_n_t_d)

        x_n_t = torch.stack(x_n_t_parts, dim=1) if num_node_features > 1 else x_n_t_parts[0].unsqueeze(-1)
        x_n_t = x_n_t.to(self.device)

        T_sampling = self.node_diffusion.T
        if max_num_steps_n is not None: T_sampling = min(T_sampling, max_num_steps_n)

        h_g_full_batch = self.node_pred_model.graph_encoder(A, x_n, abs_level, rel_level, y=y, A_n2g=A_n2g)

        # We need h_g for *active* graphs to pass to forward_with_h_g
        h_g_active = h_g_full_batch[active_graph_mask]

        # query2g for forward_with_h_g should map to indices *within the active set*
        query2g_for_pred_model = torch.repeat_interleave(
            torch.arange(h_g_active.shape[0], device=self.device),
            active_num_nodes_pred
        )

        for s_iter in range(T_sampling - 1, -1, -1):
            t_current_step = s_iter + 1
            t_tensor_for_active_graphs = torch.full(
                (h_g_active.shape[0], 1), t_current_step, dtype=torch.long, device=self.device
            )

            x_n_0_logits_list = self.node_pred_model.forward_with_h_g(
                h_g_active,  # Pass h_g only for active graphs
                x_n_t,
                t_tensor_for_active_graphs,
                query2g_for_pred_model,  # Use remapped query2g
                num_query_cumsum_active
            )

            x_n_s_parts = []
            for d_idx in range(num_node_features):
                x_n_0_probs_d = x_n_0_logits_list[d_idx].softmax(dim=-1)
                current_x_n_t_d = x_n_t[:, d_idx] if x_n_t.ndim > 1 else x_n_t.squeeze(-1)
                alpha_t_val = self.node_diffusion.alphas[t_current_step]
                alpha_bar_s_val = self.node_diffusion.alpha_bars[s_iter]
                alpha_bar_t_val = self.node_diffusion.alpha_bars[t_current_step]
                Q_t_d = self.node_diffusion.get_Q(alpha_t_val, d_idx).to(self.device)
                Q_bar_s_d = self.node_diffusion.get_Q(alpha_bar_s_val, d_idx).to(self.device)
                Q_bar_t_d = self.node_diffusion.get_Q(alpha_bar_t_val, d_idx).to(self.device)
                x_n_t_one_hot_d = F.one_hot(current_x_n_t_d,
                                            num_classes=self.node_diffusion.num_classes_list[d_idx]).float()
                x_n_s_probs_d = self.node_diffusion.posterior(
                    x_n_t_one_hot_d, Q_t_d, Q_bar_s_d, Q_bar_t_d, x_n_0_probs_d
                )
                x_n_s_d = x_n_s_probs_d.multinomial(1).squeeze(-1)
                x_n_s_parts.append(x_n_s_d)
            x_n_t = torch.stack(x_n_s_parts, dim=1) if num_node_features > 1 else x_n_s_parts[0].unsqueeze(-1)

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
                          num_new_nodes_list,
                          batch_query_src, batch_query_dst,
                          query_src_list_local, query_dst_list_local,
                          y=None, curr_level=None,
                          min_num_steps_e=None, max_num_steps_e=None):

        self.edge_pred_model.eval()
        if batch_query_src.numel() == 0:
            return edge_index_list

        prob_edge_at_T = 0.5
        current_label_t = torch.bernoulli(
            torch.full_like(batch_query_src, prob_edge_at_T, dtype=torch.float32)
        ).long()

        T_sampling = self.edge_diffusion.T
        if max_num_steps_e is not None: T_sampling = min(T_sampling, max_num_steps_e)

        batch_A = self.get_batch_A(num_nodes_cumsum, edge_index_list, self.device)

        for s_iter in range(T_sampling - 1, -1, -1):
            t_current_step = s_iter + 1
            t_tensor_for_edges = torch.full(
                (len(batch_query_src), 1), t_current_step, dtype=torch.long, device=self.device
            )
            e_0_logits = self.edge_pred_model(
                batch_A, batch_x_n, batch_abs_level, batch_rel_level,
                t_tensor_for_edges, batch_query_src, batch_query_dst, y
            )
            e_0_probs = e_0_logits.softmax(dim=-1)

            # Simplified sampling, replace with full posterior if available and correct
            current_label_t = torch.bernoulli(e_0_probs[:, 1]).long()

        updated_edge_index_list = []
        query_offset = 0
        for i in range(len(edge_index_list)):
            num_queries_for_graph_i = len(query_src_list_local[i])
            if num_queries_for_graph_i == 0:
                updated_edge_index_list.append(edge_index_list[i])
                continue
            sampled_states_for_graph_i = current_label_t[query_offset: query_offset + num_queries_for_graph_i]
            src_local_new_edges = query_src_list_local[i][sampled_states_for_graph_i == 1]
            dst_local_new_edges = query_dst_list_local[i][sampled_states_for_graph_i == 1]
            if src_local_new_edges.numel() > 0:
                new_edges_i = torch.stack([dst_local_new_edges, src_local_new_edges], dim=0)
                graph_i_updated_edges = torch.cat([edge_index_list[i], new_edges_i], dim=1)
                updated_edge_index_list.append(graph_i_updated_edges)
            else:
                updated_edge_index_list.append(edge_index_list[i])
            query_offset += num_queries_for_graph_i
        return updated_edge_index_list

    def get_batch_A(self, num_nodes_cumsum, edge_index_list, current_device):
        num_total_nodes_in_batch = num_nodes_cumsum[-1].item()
        if num_total_nodes_in_batch == 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        batch_src_list, batch_dst_list = [], []
        for i in range(len(edge_index_list)):
            if edge_index_list[i].numel() > 0:
                batch_dst_list.append(edge_index_list[i][0] + num_nodes_cumsum[i])
                batch_src_list.append(edge_index_list[i][1] + num_nodes_cumsum[i])

        if not batch_src_list:
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
        if num_total_nodes_in_batch == 0 and current_batch_size > 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(current_batch_size, 0))
        if num_total_nodes_in_batch == 0 and current_batch_size == 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        nids_list, gids_list = [], []
        for i in range(current_batch_size):
            num_nodes_in_graph_i = num_nodes_cumsum[i + 1] - num_nodes_cumsum[i]
            if num_nodes_in_graph_i > 0:
                nids_list.append(
                    torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i + 1], device=current_device, dtype=torch.long))
                gids_list.append(torch.full((num_nodes_in_graph_i,), i, device=current_device, dtype=torch.long))

        if not nids_list:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(current_batch_size, num_total_nodes_in_batch))

        nids_cat = torch.cat(nids_list)
        gids_cat = torch.cat(gids_list)
        return dglsp.spmatrix(torch.stack([gids_cat, nids_cat]),
                              shape=(current_batch_size, num_total_nodes_in_batch)).to(current_device)

    def get_batch_y(self, current_y_list, current_x_n_list_shapes_or_lengths, current_device):
        if current_y_list is None:
            return None
        if not current_y_list: return None

        # Ensure all y values are tensors and on the correct device
        processed_y_list = []
        for y_val in current_y_list:
            if y_val is None:  # Should not happen if active_y_conditions is filtered
                # This case needs to be handled based on expected y_feature_dim if some are None
                # For now, assume y_val is not None if current_y_list is not empty
                # Or, if it can be None, the y_emb_size for this encoder should have been 0.
                # This implies a per-sample conditionality, which is more complex.
                # The current fix assumes batch-level conditionality.
                continue

            y_tensor = y_val if isinstance(y_val, torch.Tensor) else torch.tensor(y_val)
            processed_y_list.append(y_tensor.to(current_device).float())  # Ensure float for SinusoidalPE

        if not processed_y_list: return None

        # Stack them. If they are scalars, they become (B,). Unsqueeze to (B,1).
        try:
            stacked_y = torch.stack(processed_y_list)
        except RuntimeError:  # Mismatched shapes
            # If y can have variable feature dimensions per sample, SinusoidalPE might not work directly.
            # Assuming for now they are all meant to be scalars or same-dim vectors.
            # If scalar, make them (1,) before stacking.
            processed_y_list_1d = [y.unsqueeze(0) if y.ndim == 0 else y for y in processed_y_list]
            stacked_y = torch.stack(processed_y_list_1d)

        if stacked_y.ndim == 1:  # If original y were scalars, stacked_y is (B,)
            stacked_y = stacked_y.unsqueeze(-1)  # Make it (B,1) for SinusoidalPE
        return stacked_y

    @torch.no_grad()
    def sample(self,
               batch_size=1,  # Changed from device to num_samples to match paper/usage
               raw_y_batch=None,
               min_num_steps_n=None,
               max_num_steps_n=None,
               min_num_steps_e=None,
               max_num_steps_e=None):

        current_device = self.device  # Use device from LayerDAG instance
        self.eval()

        # is_model_conditional is an attribute of LayerDAG instance, set during __init__
        if self.is_model_conditional and raw_y_batch is None:
            raise ValueError("raw_y_batch must be provided for conditional sampling if model is conditional.")
        if not self.is_model_conditional and raw_y_batch is not None:
            print("Warning: raw_y_batch provided for an unconditionally trained model. It will be ignored.")
            raw_y_batch = None  # Ignore it

        if raw_y_batch is not None and batch_size != len(raw_y_batch):
            raise ValueError("Batch size must match len(raw_y_batch) for conditional sampling.")

        active_edge_indices = [torch.tensor([[], []], dtype=torch.long, device=current_device) for _ in
                               range(batch_size)]

        if isinstance(self.dummy_x_n, int):
            init_x_n_val = torch.tensor([[self.dummy_x_n]], dtype=torch.long, device=current_device)
        elif isinstance(self.dummy_x_n, torch.Tensor) and self.dummy_x_n.ndim == 0:
            init_x_n_val = torch.tensor([[self.dummy_x_n.item()]], dtype=torch.long, device=current_device)
        elif isinstance(self.dummy_x_n, torch.Tensor) and self.dummy_x_n.ndim > 0:
            init_x_n_val = self.dummy_x_n.clone().detach().unsqueeze(0).to(current_device)
        else:
            raise TypeError(f"Unsupported self.dummy_x_n type: {type(self.dummy_x_n)}")

        active_x_n_features = [init_x_n_val.clone() for _ in range(batch_size)]
        active_abs_levels = [torch.tensor([[0.0]], device=current_device, dtype=torch.float32) for _ in
                             range(batch_size)]

        active_y_conditions = list(raw_y_batch) if self.is_model_conditional and raw_y_batch is not None else [
                                                                                                                  None] * batch_size
        active_original_indices = list(range(batch_size))

        finished_edge_indices = [None] * batch_size
        finished_x_n_features = [None] * batch_size
        finished_y_conditions = [None] * batch_size if self.is_model_conditional else None

        current_level = 0.0

        while active_x_n_features:
            current_active_batch_size = len(active_x_n_features)
            if current_active_batch_size == 0: break

            num_nodes_cumsum_active = torch.cumsum(torch.tensor(
                [0] + [len(x_n_i) for x_n_i in active_x_n_features], device=current_device, dtype=torch.long), dim=0)

            batch_x_n_active = torch.cat(active_x_n_features)
            batch_abs_level_active = torch.cat(active_abs_levels)
            batch_rel_level_active = batch_abs_level_active.max() - batch_abs_level_active if batch_abs_level_active.numel() > 0 else torch.empty_like(
                batch_abs_level_active)

            batch_A_active = self.get_batch_A(num_nodes_cumsum_active, active_edge_indices, current_device)
            batch_A_n2g_active = self.get_batch_A_n2g(num_nodes_cumsum_active, current_device)

            batch_y_tensor_active = None
            if self.is_model_conditional and any(y is not None for y in active_y_conditions):
                # Filter None y's if some graphs finished early and active_y_conditions was not perfectly synced
                # (though logic below tries to keep them synced)
                current_active_y_for_batch = [active_y_conditions[i] for i in range(current_active_batch_size) if
                                              active_y_conditions[i] is not None]
                if len(current_active_y_for_batch) == current_active_batch_size:  # All active graphs have y
                    batch_y_tensor_active = self.get_batch_y(active_y_conditions[:current_active_batch_size],
                                                             active_x_n_features, current_device)
                elif not current_active_y_for_batch:  # No active graphs have y (should not happen if is_model_conditional and raw_y_batch was provided)
                    pass  # batch_y_tensor_active remains None
                else:  # Mixed case - this implies per-sample conditionality which is complex.
                    # For now, if any y is None in a conditional batch, it might cause issues or require y_emb_size=0 for those.
                    # The current setup assumes all active graphs in a conditional batch have a y.
                    # If active_y_conditions can have Nones, get_batch_y needs to handle it or error out.
                    # Safest for now: if any y is None in active_y_conditions for a conditional model, this is an issue.
                    # However, active_y_conditions should always correspond to active_x_n_features.
                    batch_y_tensor_active = self.get_batch_y(active_y_conditions, active_x_n_features, current_device)

            new_nodes_attrs_per_active_graph = self.sample_node_layer(
                batch_A_active, batch_x_n_active, batch_abs_level_active, batch_rel_level_active,
                batch_A_n2g_active, curr_level=current_level, y=batch_y_tensor_active,
                min_num_steps_n=min_num_steps_n, max_num_steps_n=max_num_steps_n
            )

            next_iter_active_edge_indices = []
            next_iter_active_x_n_features = []
            next_iter_active_abs_levels = []
            next_iter_active_y_conditions = [] if self.is_model_conditional else None
            next_iter_original_indices = []

            query_src_local_for_continuing = []
            query_dst_local_for_continuing = []
            num_new_nodes_for_continuing = []

            continuing_graph_indices_in_active_batch = []

            for i in range(current_active_batch_size):
                original_batch_idx = active_original_indices[i]
                newly_sampled_nodes_for_graph_i = new_nodes_attrs_per_active_graph[i]

                if newly_sampled_nodes_for_graph_i.numel() == 0:
                    final_edges = active_edge_indices[i]
                    if final_edges.numel() > 0: final_edges = final_edges - 1

                    finished_edge_indices[original_batch_idx] = final_edges
                    finished_x_n_features[original_batch_idx] = active_x_n_features[i][1:]
                    if self.is_model_conditional:
                        finished_y_conditions[original_batch_idx] = active_y_conditions[i]
                else:
                    continuing_graph_indices_in_active_batch.append(i)
                    next_iter_original_indices.append(original_batch_idx)
                    next_iter_active_edge_indices.append(active_edge_indices[i])

                    updated_x_n_for_graph_i = torch.cat([active_x_n_features[i], newly_sampled_nodes_for_graph_i])
                    next_iter_active_x_n_features.append(updated_x_n_for_graph_i)

                    new_abs_levels_i = torch.full(
                        (newly_sampled_nodes_for_graph_i.shape[0], 1),
                        current_level + 1.0,
                        dtype=torch.float32, device=current_device
                    )
                    updated_abs_level_for_graph_i = torch.cat([active_abs_levels[i], new_abs_levels_i])
                    next_iter_active_abs_levels.append(updated_abs_level_for_graph_i)

                    if self.is_model_conditional:
                        next_iter_active_y_conditions.append(active_y_conditions[i])

                    num_old_nodes_incl_dummy = active_x_n_features[i].shape[0]
                    num_new_nodes = newly_sampled_nodes_for_graph_i.shape[0]
                    num_new_nodes_for_continuing.append(num_new_nodes)

                    q_src_local_i, q_dst_local_i = [], []
                    if num_old_nodes_incl_dummy > 1:
                        for s_local_idx in range(1, num_old_nodes_incl_dummy):
                            for d_new_node_offset in range(num_new_nodes):
                                d_local_idx = num_old_nodes_incl_dummy + d_new_node_offset
                                q_src_local_i.append(s_local_idx)
                                q_dst_local_i.append(d_local_idx)

                    query_src_local_for_continuing.append(
                        torch.tensor(q_src_local_i, dtype=torch.long, device=current_device))
                    query_dst_local_for_continuing.append(
                        torch.tensor(q_dst_local_i, dtype=torch.long, device=current_device))

            active_edge_indices = next_iter_active_edge_indices
            active_x_n_features = next_iter_active_x_n_features
            active_abs_levels = next_iter_active_abs_levels
            if self.is_model_conditional: active_y_conditions = next_iter_active_y_conditions
            active_original_indices = next_iter_original_indices

            if not active_x_n_features: break

            current_level += 1.0

            if any(q.numel() > 0 for q in query_src_local_for_continuing):
                num_nodes_cumsum_edge_pred = torch.cumsum(torch.tensor(
                    [0] + [len(x_n_i) for x_n_i in active_x_n_features], device=current_device, dtype=torch.long),
                    dim=0)

                batch_x_n_edge_pred = torch.cat(active_x_n_features)
                batch_abs_level_edge_pred = torch.cat(active_abs_levels)
                batch_rel_level_edge_pred = batch_abs_level_edge_pred.max() - batch_abs_level_edge_pred if batch_abs_level_edge_pred.numel() > 0 else torch.empty_like(
                    batch_abs_level_edge_pred)

                batch_y_tensor_edge_pred = None
                if self.is_model_conditional and any(y is not None for y in active_y_conditions):
                    batch_y_tensor_edge_pred = self.get_batch_y(active_y_conditions, active_x_n_features,
                                                                current_device)

                batch_query_src_cat, batch_query_dst_cat = [], []
                for i in range(len(active_x_n_features)):
                    if query_src_local_for_continuing[i].numel() > 0:
                        batch_query_src_cat.append(query_src_local_for_continuing[i] + num_nodes_cumsum_edge_pred[i])
                        batch_query_dst_cat.append(query_dst_local_for_continuing[i] + num_nodes_cumsum_edge_pred[i])

                if batch_query_src_cat:
                    final_batch_q_src = torch.cat(batch_query_src_cat)
                    final_batch_q_dst = torch.cat(batch_query_dst_cat)

                    if final_batch_q_src.numel() > 0:
                        active_edge_indices = self.sample_edge_layer(
                            num_nodes_cumsum_edge_pred, active_edge_indices,
                            batch_x_n_edge_pred, batch_abs_level_edge_pred, batch_rel_level_edge_pred,
                            num_new_nodes_for_continuing,
                            final_batch_q_src, final_batch_q_dst,
                            query_src_local_for_continuing, query_dst_local_for_continuing,
                            y=batch_y_tensor_edge_pred, curr_level=current_level,
                            min_num_steps_e=min_num_steps_e, max_num_steps_e=max_num_steps_e
                        )

            if self.max_level is not None and current_level >= self.max_level:
                for i in range(len(active_x_n_features)):
                    original_idx_rem = active_original_indices[i]
                    final_edges_rem = active_edge_indices[i]
                    if final_edges_rem.numel() > 0: final_edges_rem = final_edges_rem - 1

                    finished_edge_indices[original_idx_rem] = final_edges_rem
                    finished_x_n_features[original_idx_rem] = active_x_n_features[i][1:]
                    if self.is_model_conditional:
                        finished_y_conditions[original_idx_rem] = active_y_conditions[i]
                break

        return finished_edge_indices, finished_x_n_features, finished_y_conditions


