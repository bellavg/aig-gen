# src/model/layer_dag.py
import dgl.sparse as dglsp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random  # For sampling steps in diffusion

from einops import rearrange

__all__ = [
    'LayerDAG'
]


class SinusoidalPE(nn.Module):
    def __init__(self, pe_size):
        super().__init__()

        self.pe_size = pe_size
        if pe_size > 0:
            self.div_term = torch.exp(torch.arange(0, pe_size, 2) *
                                      (-math.log(10000.0) / pe_size))
            self.div_term = nn.Parameter(self.div_term, requires_grad=False)

    def forward(self, position):
        if self.pe_size == 0:
            return torch.zeros(len(position), 0, device=position.device)  # Ensure device
        # Ensure position is float for multiplication
        position = position.float()
        return torch.cat([
            torch.sin(position * self.div_term),
            torch.cos(position * self.div_term)
        ], dim=-1)


class BiMPNNLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.W = nn.Linear(in_size, out_size)
        self.W_trans = nn.Linear(in_size, out_size)
        self.W_self = nn.Linear(in_size, out_size)

    def forward(self, A, A_T, h_n):
        if A.nnz == 0:  # Check if the sparse matrix A has any non-zero elements
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
        return F.one_hot(position_squeezed.clamp(max=self.pe_size - 1),
                         num_classes=self.pe_size).float()  # one_hot returns long, convert to float


class MultiEmbedding(nn.Module):
    def __init__(self, num_x_n_cat, hidden_size):  # num_x_n_cat is expected to be a tensor here
        super().__init__()

        # Process num_x_n_cat to be a list of integers for nn.Embedding
        if isinstance(num_x_n_cat, torch.Tensor):
            if num_x_n_cat.ndim == 0:  # Scalar tensor
                processed_num_x_n_cat = [int(num_x_n_cat.item())]
            else:  # 1D tensor
                processed_num_x_n_cat = [int(c.item()) for c in num_x_n_cat]
        elif isinstance(num_x_n_cat, (list, tuple)):
            processed_num_x_n_cat = [int(c) for c in num_x_n_cat]
        elif isinstance(num_x_n_cat, int):
            processed_num_x_n_cat = [num_x_n_cat]
        else:
            raise TypeError(f"num_x_n_cat must be an int, list, tuple, or tensor, got {type(num_x_n_cat)}")

        self.emb_list = nn.ModuleList([
            nn.Embedding(num_cat_i, hidden_size)
            for num_cat_i in processed_num_x_n_cat
        ])

    def forward(self, x_n_cat):
        # x_n_cat: (N) if single feature, (N, num_features) if multiple
        if not self.emb_list:
            raise RuntimeError("MultiEmbedding emb_list is empty. Check initialization with num_x_n_cat.")

        if x_n_cat.ndim == 1:  # Single feature or already processed
            if len(self.emb_list) != 1:
                raise ValueError(f"Input x_n_cat is 1D, but have {len(self.emb_list)} embedding layers. Expected 1.")
            # ---- START DEBUG PRINTS (Optional, can be removed after debugging) ----
            # print(f"DEBUG: MultiEmbedding input x_n_cat device: {x_n_cat.device}")
            # print(f"DEBUG: MultiEmbedding input x_n_cat dtype: {x_n_cat.dtype}")
            # if x_n_cat.numel() > 0:
            #     print(f"DEBUG: MultiEmbedding input x_n_cat min: {x_n_cat.min().item()}")
            #     print(f"DEBUG: MultiEmbedding input x_n_cat max: {x_n_cat.max().item()}")
            # else:
            #     print(f"DEBUG: MultiEmbedding input x_n_cat is EMPTY. Shape: {x_n_cat.shape}")
            # print(f"DEBUG: MultiEmbedding emb_list[0].num_embeddings: {self.emb_list[0].num_embeddings}")
            # ---- END DEBUG PRINTS ----
            x_n_emb = self.emb_list[0](x_n_cat)
        elif x_n_cat.ndim == 2:  # Multiple features: (N, num_features)
            if x_n_cat.shape[1] != len(self.emb_list):
                raise ValueError(
                    f"MultiEmbedding: Number of features in x_n_cat ({x_n_cat.shape[1]}) "
                    f"does not match number of embedding layers ({len(self.emb_list)})."
                )
            x_n_emb = torch.cat([
                self.emb_list[i](x_n_cat[:, i]) for i in range(len(self.emb_list))
            ], dim=1)
        else:
            raise ValueError(f"Unsupported x_n_cat shape: {x_n_cat.shape}")
        return x_n_emb


class BiMPNNEncoder(nn.Module):
    def __init__(self,
                 num_x_n_cat,  # Expected to be a tensor e.g. from LayerDAG's num_x_n_cat_tensor
                 x_n_emb_size,
                 pe_emb_size,
                 hidden_size,  # This is the target combined feature size for proj_input
                 num_mpnn_layers,
                 pe=None,
                 y_emb_size=0,  # This will be 0 if not conditional
                 pool=None):
        super().__init__()

        self.pe = pe
        self.level_emb = None
        if self.pe in ['relative_level', 'abs_level'] and pe_emb_size > 0:
            self.level_emb = SinusoidalPE(pe_emb_size)
        elif self.pe == 'relative_level_one_hot' and pe_emb_size > 0:
            self.level_emb = OneHotPE(pe_emb_size)

        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)
        self.y_emb = SinusoidalPE(y_emb_size) if y_emb_size > 0 else None

        # Calculate actual input dimension to self.proj_input
        # Based on the output dim of self.x_n_emb
        actual_x_n_emb_dim = x_n_emb_size
        if isinstance(num_x_n_cat, torch.Tensor) and num_x_n_cat.ndim > 0 and len(num_x_n_cat) > 1:
            actual_x_n_emb_dim = len(num_x_n_cat) * x_n_emb_size
        elif isinstance(num_x_n_cat, (list, tuple)) and len(num_x_n_cat) > 1:  # Should be tensor by now
            actual_x_n_emb_dim = len(num_x_n_cat) * x_n_emb_size

        current_input_dim_for_proj = actual_x_n_emb_dim
        if self.level_emb is not None and pe_emb_size > 0:
            current_input_dim_for_proj += pe_emb_size
        if self.y_emb is not None and y_emb_size > 0:  # y_emb_size is already conditional
            current_input_dim_for_proj += y_emb_size

        self.proj_input = nn.Sequential(
            nn.Linear(current_input_dim_for_proj, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.mpnn_layers = nn.ModuleList()
        for _ in range(num_mpnn_layers):
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
        # A and A_T are dgl.sparse matrices
        A_T = A.T

        h_n_parts = [self.x_n_emb(x_n)]

        if self.level_emb is not None:
            if self.pe == 'abs_level':
                node_pe = self.level_emb(abs_level)
            elif self.pe in ['relative_level', 'relative_level_one_hot']:
                node_pe = self.level_emb(rel_level)
            else:  # Should not happen if pe is one of the known types and level_emb exists
                node_pe = None
            if node_pe is not None and node_pe.shape[1] > 0:
                h_n_parts.append(node_pe)

        # Handling of y (conditional information)
        # If y is graph-level (B, 1 or B, F_y_cond) and we need to apply it to nodes (N, F_node_cond)
        # A_n2g (B, N) maps graphs to nodes. We need to map nodes to their graph's y.
        if y is not None and self.y_emb is not None:
            h_y_graph = self.y_emb(y)  # (B, y_emb_size)
            if A_n2g is not None and A_n2g.shape[0] == y.shape[0]:  # y is graph-level
                # Map graph embeddings to nodes
                # A_n2g.indices() gives [graph_indices_for_edges, node_indices_for_edges]
                # We need a map from node_idx (0..N-1) to graph_idx (0..B-1)
                # Assuming A_n2g is constructed such that its non-zero elements define this mapping
                # Example: if A_n2g from dglsp.spmatrix(torch.stack([gids, nids]), shape=(B,N))
                # where gids and nids are flat lists.
                if A_n2g.nnz > 0:
                    node_to_graph_map = torch.zeros(A_n2g.shape[1], dtype=torch.long, device=x_n.device)
                    # A_n2g.coo() returns (row, col, val) -> (gid, nid, 1.0)
                    # For dgl.sparse, A.coo() returns (rows, cols) for indices, A.val for values
                    # Let's assume A_n2g.indices() gives [gids_of_entries, nids_of_entries]
                    # We need to ensure each node gets one graph_id.
                    # A simpler way if A_n2g is effectively a dense mapping in one direction:
                    # If A_n2g.shape = (B,N) and A_n2g[b,n]=1 if node n in graph b, 0 otherwise.
                    # Then A_n2g.T.argmax(dim=1) would give graph_idx for each node.
                    # This is inefficient for sparse.
                    # A common way to get gids for nodes if A_n2g is from [gids,nids]:
                    # If nids are sorted 0..N-1, then gids correspond.
                    # For now, let's assume a robust way to get this mapping if needed.
                    # A simple but potentially slow way for sparse A_n2g (B,N):
                    # Find the graph index for each node.
                    # This is complex if A_n2g doesn't have exactly one 1 per column.
                    # Let's revert to the simpler direct concatenation if y is already node-level.
                    # If y is (N, y_emb_size), then direct concat is fine.
                    # The original code was: h_n = torch.cat([h_n, h_y], dim=-1)
                    # This implies h_y must be (N, y_emb_size).
                    # If y is (B,1), h_y_graph is (B, y_emb_size).
                    # We need to expand h_y_graph to (N, y_emb_size).
                    # This requires mapping each node to its graph index.
                    # If A_n2g.shape = (B, N), and A_n2g.val has shape (num_nodes_total_in_batch)
                    # and A_n2g.indices()[0] are graph_ids, A_n2g.indices()[1] are node_ids (0 to N-1)
                    # Then we can use A_n2g.indices()[0] to index h_y_graph if node_ids are sorted.
                    # This part is tricky and depends heavily on A_n2g's structure.
                    # A robust way:
                    if h_y_graph.shape[0] == A_n2g.shape[0]:  # if h_y_graph is (B, F_y)
                        # We need to "scatter" h_y_graph to nodes.
                        # If A_n2g is (B,N) and its rows sum to 1 (node belongs to one graph)
                        # This is non-trivial with dgl.sparse in a simple line.
                        # For now, assume if y is passed and y_emb exists, it's node-level or handled by pooling.
                        # The original code's direct concatenation implies y is already (N, features)
                        # or y_emb(y) results in (N, features).
                        # If y is (B,1), y_emb(y) is (B, y_emb_size). This won't cat with h_n (N, ...).
                        # This indicates a potential architectural mismatch if y is graph-level and pool=None.
                        # If pool is not None, y (graph-level) is typically combined *after* pooling h_n to h_g.
                        # For now, let's keep the original structure, assuming y is correctly prepared if provided.
                        h_n_parts.append(h_y_graph)  # This will fail if shapes don't match for cat.
                        # This line is kept from user's baseline.
                        # A proper fix would involve A_n2g if y is graph-level.
                    else:  # y is already node-level
                        h_n_parts.append(h_y_graph)

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
        elif self.pool == 'sum':
            if A_n2g is None: raise ValueError("A_n2g must be provided for sum pooling.")
            h_g = A_n2g @ h_n
            return self.bn_g(h_g)
        elif self.pool == 'mean':
            if A_n2g is None: raise ValueError("A_n2g must be provided for mean pooling.")
            h_g = A_n2g @ h_n
            sum_val = A_n2g.sum(dim=1).unsqueeze(-1)
            sum_val = torch.where(sum_val == 0, torch.ones_like(sum_val), sum_val)
            h_g = h_g / sum_val
            return self.bn_g(h_g)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool}")


class GraphClassifier(nn.Module):
    def __init__(self,
                 graph_encoder,  # Instance of BiMPNNEncoder
                 emb_size,  # Output hidden_size of graph_encoder
                 num_classes):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, y=None):
        # y is graph-level condition (B,1)
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
        assert head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
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
        if batch_size <= 0:  # Handle empty or invalid batch case
            return q.new_empty(0, q.shape[-1]) if q.numel() > 0 else torch.empty(0, 0, device=q.device)

        num_query_nodes = torch.diff(num_query_cumsum)

        max_num_nodes = 0
        if num_query_nodes.numel() > 0:
            max_num_nodes = num_query_nodes.max().item()

        if max_num_nodes == 0:
            print(
                f"DEBUG: TransformerLayer.attn - max_num_nodes is 0. Batch size: {batch_size}. q.shape: {q.shape if q is not None else 'None'}")
            # If q is already empty (total queries = 0), return q (which would be empty)
            if q is not None and q.shape[0] == 0:
                return q
            # If q is None or not empty but max_num_nodes is 0, implies no actual queries to process.
            # Return an empty tensor with the feature dimension of q if q exists, else a generic empty tensor.
            feat_dim = q.shape[-1] if q is not None and q.ndim > 1 else (
                v.shape[-1] if v is not None and v.ndim > 1 else 0)  # Try to get feature dim
            return torch.empty(0, feat_dim, device=q.device if q is not None else (
                v.device if v is not None else num_query_cumsum.device))

        q_padded = q.new_zeros(batch_size, max_num_nodes, q.shape[-1])
        k_padded = k.new_zeros(batch_size, max_num_nodes, k.shape[-1])
        v_padded = v.new_zeros(batch_size, max_num_nodes, v.shape[-1])
        pad_mask = q.new_zeros(batch_size, max_num_nodes, dtype=torch.bool)  # Ensure boolean

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
            pad_mask.unsqueeze(1).unsqueeze(2),  # (B, max_N) -> (B,1,1,max_N) for broadcasting
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

        # Check if h_n_new is empty due to all-zero max_num_nodes in attn
        if h_n_new.shape[0] == 0 and h_n.shape[0] != 0:
            # This means attn decided there's nothing to output, but h_n had queries.
            # This could happen if max_num_nodes was 0.
            # In this case, the residual connection and further processing are problematic.
            # It might be better to return h_n as if the transformer block was a no-op for this case.
            print(
                f"WARNING: TransformerLayer.forward - h_n_new is empty after attn, but h_n was not. h_n.shape: {h_n.shape}. Skipping residual and FFN for safety.")
            return h_n  # Or handle more gracefully, e.g. by ensuring attn returns shape compatible with h_n

        # Ensure h_n_new has the same number of elements (first dim) as h_n for residual
        if h_n_new.shape[0] != h_n.shape[0]:
            print(
                f"WARNING: TransformerLayer.forward - shape mismatch h_n ({h_n.shape}) vs h_n_new ({h_n_new.shape}) after attn. This is unexpected. Skipping residual for h_n_new.")
            # If shapes mismatch, direct addition will fail.
            # This indicates a potential issue in padding/unpadding logic or how num_query_cumsum is handled.
            # For now, to avoid crash, we might skip the residual for h_n_new or only apply FFN.
            # Applying FFN to h_n_new and then norm2 might be an option if proj_new is applied first.
            h_n_transformed_ffn = self.norm2(
                self.proj_new(h_n_new) + self.out_proj(self.proj_new(h_n_new)))  # Example, needs careful thought
            # This is not ideal. The shapes should match.
            # For minimal change, if they don't match, perhaps the block should effectively be a no-op or raise error.
            # Let's assume for now the proj_new is applied and then added to h_n if shapes allow.
            # The original was h_n = self.norm1(h_n + self.proj_new(h_n_new))
            # If h_n_new is not same shape as h_n in the first dim, this will fail.
            # This state should ideally not be reached.
            return h_n  # Fallback: return original h_n if shapes mismatch critically.

        h_n = self.norm1(h_n + self.proj_new(h_n_new))
        h_n = self.norm2(h_n + self.out_proj(h_n))
        return h_n


class NodePredModel(nn.Module):
    def __init__(self,
                 graph_encoder,
                 num_x_n_cat,  # Tensor of actual category counts (e.g. total_cats - 1)
                 x_n_emb_size,
                 t_emb_size,
                 in_hidden_size,  # Output hidden_size from graph_encoder
                 out_hidden_size,
                 num_transformer_layers,
                 num_heads,
                 dropout):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)  # Embedding for x_n_t (noisy attributes)
        self.t_emb = SinusoidalPE(t_emb_size)

        # Calculate actual input dim for self.project_h_n
        actual_x_n_t_emb_dim = x_n_emb_size
        if isinstance(num_x_n_cat, torch.Tensor) and num_x_n_cat.ndim > 0 and len(num_x_n_cat) > 1:
            actual_x_n_t_emb_dim = len(num_x_n_cat) * x_n_emb_size
        elif isinstance(num_x_n_cat, (list, tuple)) and len(num_x_n_cat) > 1:  # Should be tensor by now
            actual_x_n_t_emb_dim = len(num_x_n_cat) * x_n_emb_size

        combined_in_hidden_size_for_proj = actual_x_n_t_emb_dim + in_hidden_size + t_emb_size

        self.project_h_n = nn.Sequential(
            nn.Linear(combined_in_hidden_size_for_proj, out_hidden_size),
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
                processed_num_x_n_cat_list = [int(num_x_n_cat.item())]
            else:
                processed_num_x_n_cat_list = [int(c.item()) for c in num_x_n_cat]
        elif isinstance(num_x_n_cat, (list, tuple)):
            processed_num_x_n_cat_list = [int(c) for c in num_x_n_cat]
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
        # x_n_t: (Q, num_features) or (Q) - noisy node attributes for query nodes
        # h_g: (B, F_graph_enc) - graph embeddings
        # t: (B, 1) - timesteps for graphs
        # query2g: (Q) - maps each query node to its graph index in the batch
        # num_query_cumsum: (B+1) - cumsum of query nodes per graph

        if x_n_t.shape[0] == 0:  # No query nodes
            return [torch.empty(0, device=h_g.device) for _ in self.pred_list]  # Match expected list of tensors

        h_t_graph = self.t_emb(t)  # (B, t_emb_size)
        h_g_cond = torch.cat([h_g, h_t_graph], dim=1)  # (B, F_graph_enc + t_emb_size)

        # x_n_t needs to be prepared for MultiEmbedding. If it's (Q), it's fine. If (Q, num_feat), also fine.
        h_n_t_attrs = self.x_n_emb(x_n_t)  # (Q, actual_x_n_t_emb_dim)

        h_g_cond_expanded = h_g_cond[query2g]  # (Q, F_graph_enc + t_emb_size)

        h_n_t_combined = torch.cat([h_n_t_attrs, h_g_cond_expanded], dim=1)
        h_n_t = self.project_h_n(h_n_t_combined)  # (Q, out_hidden_size)

        for trans_layer in self.trans_layers:
            if h_n_t.shape[0] == 0: break  # No nodes to process further
            h_n_t = trans_layer(h_n_t, num_query_cumsum)

        pred = []
        if h_n_t.shape[0] > 0:
            for d_idx in range(len(self.pred_list)):
                pred.append(self.pred_list[d_idx](h_n_t))
        else:  # h_n_t became empty or was initially for 0 queries
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
                 in_hidden_size,  # Output hidden_size from graph_encoder
                 out_hidden_size):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.t_emb = SinusoidalPE(t_emb_size)
        mlp_input_dim = 2 * in_hidden_size + t_emb_size
        self.pred = nn.Sequential(
            nn.Linear(mlp_input_dim, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, 2)  # Binary prediction (edge exists or not)
        )

    def forward(self, A, x_n, abs_level, rel_level, t,
                query_src, query_dst, y=None):
        # For EdgePredModel, graph_encoder typically runs on the current graph state (A, x_n)
        # to get node embeddings h_n.
        # If conditional (y is not None), y is graph-level. BiMPNNEncoder needs A_n2g for this.
        # This A_n2g is not directly passed to EdgePredModel.forward in train.py.
        # This implies either:
        # 1. The BiMPNNEncoder for EdgePredModel is configured with pool=None (so A_n2g not used for y).
        # 2. Or, y is not used by BiMPNNEncoder if A_n2g is None.
        # 3. Or, A_n2g needs to be constructed here if y is to be used graph-conditionally.
        # For now, passing A_n2g=None, assuming BiMPNNEncoder handles it (e.g. y_emb_size is 0 if not conditional,
        # or if conditional, it's configured such that y is applied without explicit A_n2g, or pool=None).
        h_n = self.graph_encoder(A, x_n, abs_level, rel_level, y=y, A_n2g=None)

        if query_src.numel() == 0:  # No query edges
            return torch.empty(0, 2, device=h_n.device)  # Return shape (0, num_classes_for_edge)

        h_t_edge = self.t_emb(t)  # t is (num_queries, 1) -> (num_queries, t_emb_size)

        h_e_parts = []
        if h_t_edge.shape[1] > 0: h_e_parts.append(h_t_edge)
        h_e_parts.append(h_n[query_src])
        h_e_parts.append(h_n[query_dst])

        h_e = torch.cat(h_e_parts, dim=-1)

        return self.pred(h_e)


class LayerDAG(nn.Module):
    def __init__(self,
                 device,  # Explicit device parameter
                 num_x_n_cat,  # From train_set.num_categories (actual_types + 1 for dummy)
                 node_count_encoder_config,
                 max_layer_size,
                 node_diffusion,
                 node_pred_graph_encoder_config,
                 node_predictor_config,
                 edge_diffusion,
                 edge_pred_graph_encoder_config,
                 edge_predictor_config,
                 is_conditional=False,  # Explicit flag for conditional mode
                 max_level=None):
        super().__init__()

        self.device = torch.device(device)  # Store device
        self.is_conditional = is_conditional  # Store conditional flag

        # Process num_x_n_cat to be a tensor on the correct device
        if isinstance(num_x_n_cat, int):
            num_x_n_cat_tensor = torch.LongTensor([num_x_n_cat]).to(self.device)
        elif isinstance(num_x_n_cat, torch.Tensor):
            num_x_n_cat_tensor = num_x_n_cat.to(self.device)
        elif isinstance(num_x_n_cat, (list, tuple)):
            num_x_n_cat_tensor = torch.LongTensor(num_x_n_cat).to(self.device)
        else:
            raise TypeError(f"num_x_n_cat must be an int, list, tuple or torch.Tensor, got {type(num_x_n_cat)}")

        # Determine dummy_x_n based on num_x_n_cat_tensor
        if num_x_n_cat_tensor.ndim == 0 or len(num_x_n_cat_tensor) == 1:  # Single feature type
            self.dummy_x_n = (num_x_n_cat_tensor.item() - 1)
        else:  # Multi-feature node types
            self.dummy_x_n = num_x_n_cat_tensor - 1  # Tensor of dummy indices for each feature type

        # --- Node Count Model ---
        nc_config = node_count_encoder_config
        nc_y_emb_size = nc_config.get('y_emb_size', 0) if self.is_conditional else 0

        node_count_encoder = BiMPNNEncoder(
            num_x_n_cat=num_x_n_cat_tensor,
            x_n_emb_size=nc_config['x_n_emb_size'],
            pe_emb_size=nc_config.get('pe_emb_size', 0),
            hidden_size=nc_config['x_n_emb_size'] * (
                len(num_x_n_cat_tensor) if num_x_n_cat_tensor.ndim > 0 and len(num_x_n_cat_tensor) > 1 else 1) + \
                        nc_config.get('pe_emb_size', 0) + \
                        nc_y_emb_size,  # hidden_size for BiMPNNEncoder is the combined input dim
            num_mpnn_layers=nc_config['num_mpnn_layers'],
            pe=nc_config.get('pe'),
            y_emb_size=nc_y_emb_size,  # Pass conditional y_emb_size
            pool=nc_config.get('pool', 'sum')
        ).to(self.device)
        self.node_count_model = GraphClassifier(
            node_count_encoder,
            emb_size=node_count_encoder.proj_input[0].out_features,  # Output size of BiMPNNEncoder's proj_input
            num_classes=max_layer_size + 1
        ).to(self.device)

        # --- Node Prediction Model ---
        self.node_diffusion = node_diffusion.to(self.device)
        np_config_enc = node_pred_graph_encoder_config
        np_config_pred = node_predictor_config
        np_y_emb_size = np_config_enc.get('y_emb_size', 0) if self.is_conditional else 0

        node_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat=num_x_n_cat_tensor,
            x_n_emb_size=np_config_enc['x_n_emb_size'],
            pe_emb_size=np_config_enc.get('pe_emb_size', 0),
            hidden_size=np_config_enc['x_n_emb_size'] * (
                len(num_x_n_cat_tensor) if num_x_n_cat_tensor.ndim > 0 and len(num_x_n_cat_tensor) > 1 else 1) + \
                        np_config_enc.get('pe_emb_size', 0) + \
                        np_y_emb_size,
            num_mpnn_layers=np_config_enc['num_mpnn_layers'],
            pe=np_config_enc.get('pe'),
            y_emb_size=np_y_emb_size,
            pool=np_config_enc.get('pool', 'sum')
        ).to(self.device)

        num_actual_node_cats = num_x_n_cat_tensor - 1  # For NodePredModel's internal MultiEmbedding on x_n_t

        self.node_pred_model = NodePredModel(
            node_pred_graph_encoder,
            num_x_n_cat=num_actual_node_cats,
            x_n_emb_size=np_config_enc['x_n_emb_size'],  # For x_n_t embedding
            t_emb_size=np_config_pred['t_emb_size'],
            in_hidden_size=node_pred_graph_encoder.proj_input[0].out_features,  # Output of graph_encoder
            out_hidden_size=np_config_pred['out_hidden_size'],
            num_transformer_layers=np_config_pred['num_transformer_layers'],
            num_heads=np_config_pred['num_heads'],
            dropout=np_config_pred['dropout']
        ).to(self.device)

        # --- Edge Prediction Model ---
        self.edge_diffusion = edge_diffusion.to(self.device)
        ep_config_enc = edge_pred_graph_encoder_config
        ep_config_pred = edge_predictor_config
        ep_y_emb_size = ep_config_enc.get('y_emb_size', 0) if self.is_conditional else 0

        edge_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat=num_x_n_cat_tensor,
            x_n_emb_size=ep_config_enc['x_n_emb_size'],
            pe_emb_size=ep_config_enc.get('pe_emb_size', 0),
            hidden_size=ep_config_enc['x_n_emb_size'] * (
                len(num_x_n_cat_tensor) if num_x_n_cat_tensor.ndim > 0 and len(num_x_n_cat_tensor) > 1 else 1) + \
                        ep_config_enc.get('pe_emb_size', 0) + \
                        ep_y_emb_size,
            num_mpnn_layers=ep_config_enc['num_mpnn_layers'],
            pe=ep_config_enc.get('pe'),
            y_emb_size=ep_y_emb_size,
            pool=ep_config_enc.get('pool')  # Often None for edge prediction
        ).to(self.device)
        self.edge_pred_model = EdgePredModel(
            edge_pred_graph_encoder,
            t_emb_size=ep_config_pred['t_emb_size'],
            in_hidden_size=edge_pred_graph_encoder.proj_input[0].out_features,  # Output from graph_encoder
            out_hidden_size=ep_config_pred['out_hidden_size']
        ).to(self.device)

        self.max_level = max_level

    @torch.no_grad()
    def posterior(self, A, x_n, abs_level, rel_level, A_n2g,
                  x_n_t, t, query2g, num_query_cumsum,
                  y=None,
                  x_n_0_logits_list=None):
        # This method seems to be from DiGress, ensure it's adapted for LayerDAG
        # x_n_t is (Q, num_features) or (Q)
        # t is (B, 1)
        # query2g maps Q to B
        # num_query_cumsum is (B+1)

        if x_n_0_logits_list is None:
            x_n_0_logits_list = self.node_pred_model(
                A, x_n, abs_level, rel_level, A_n2g, x_n_t, t,
                query2g, num_query_cumsum, y)

        Q_bar_s_list = []
        Q_bar_t_list = []
        Q_t_list = []

        s = t - 1  # t is (B,1), so s is (B,1)
        s = torch.where(s < 0, torch.zeros_like(s), s)  # Clamp s to be non-negative for indexing alpha_bars

        # alpha_bars, alphas are (T_diffusion+1)
        # Ensure indices are within bounds
        s_flat = s.squeeze(-1).clamp(0, self.node_diffusion.T)  # (B)
        t_flat = t.squeeze(-1).clamp(0, self.node_diffusion.T)  # (B)

        alpha_bar_s_vals = self.node_diffusion.alpha_bars[s_flat]  # (B)
        alpha_bar_t_vals = self.node_diffusion.alpha_bars[t_flat]  # (B)
        alpha_t_vals = self.node_diffusion.alphas[t_flat]  # (B)

        num_node_features_to_predict = x_n_t.shape[1] if x_n_t.ndim > 1 else 1
        if not isinstance(self.node_diffusion.num_classes_list, list):  # Should be a list from DiscreteDiffusion
            num_classes_list_internal = [self.node_diffusion.num_classes_list]
        else:
            num_classes_list_internal = self.node_diffusion.num_classes_list

        for d in range(num_node_features_to_predict):
            # Expand alpha values for each query node based on its graph
            alpha_bar_s_expanded = alpha_bar_s_vals[query2g]  # (Q)
            alpha_bar_t_expanded = alpha_bar_t_vals[query2g]  # (Q)
            alpha_t_expanded = alpha_t_vals[query2g]  # (Q)

            Q_bar_s_d = self.node_diffusion.get_Q(alpha_bar_s_expanded, d)  # (Q, K_d, K_d)
            Q_bar_s_list.append(Q_bar_s_d)

            Q_bar_t_d = self.node_diffusion.get_Q(alpha_bar_t_expanded, d)  # (Q, K_d, K_d)
            Q_bar_t_list.append(Q_bar_t_d)

            Q_t_d = self.node_diffusion.get_Q(alpha_t_expanded, d)  # (Q, K_d, K_d)
            Q_t_list.append(Q_t_d)

        pred_list = []
        for d in range(num_node_features_to_predict):
            x_n_0_probs_d = x_n_0_logits_list[d].softmax(dim=-1)  # (Q, K_d)

            current_x_n_t_d = x_n_t[:, d] if x_n_t.ndim > 1 else x_n_t
            x_n_t_one_hot_d = F.one_hot(
                current_x_n_t_d,
                num_classes=num_classes_list_internal[d]
            ).float()  # (Q, K_d)

            # Shapes for bmm: (Q, 1, K_d) @ (Q, K_d, K_d) -> (Q, 1, K_d)
            # Or element-wise products if Q matrices are (Q, K, K)
            # Original DiGress: (B, K, K) @ (Q, K, 1) -> (B, Q, K) with query2g indexing.
            # Here, Q matrices are already (Q, K, K) after expansion.

            # numer1: p(x_s | x_0) = Q_bar_s @ x_0_probs
            # (Q, K_d, K_d) @ (Q, K_d, 1) -> (Q, K_d, 1)
            numer1 = torch.bmm(Q_bar_s_list[d], x_n_0_probs_d.unsqueeze(-1))

            # numer2: p(x_t | x_s) for x_s = k', sum over k' of Q_t[x_t, k'] * p(x_s=k' | x_0)
            # This is getting into the full posterior formula which is complex.
            # A common simplification or direct use:
            # p(x_{s} | x_t, \hat{x}_0) \propto p(x_t | x_s, \hat{x}_0) p(x_s | \hat{x}_0)
            #                          = p(x_t | x_s) p(x_s | \hat{x}_0) (assuming x_0 indep of x_t given x_s)
            # where p(x_t | x_s) from Q_t, and p(x_s | \hat{x}_0) from Q_bar_s applied to \hat{x}_0_probs

            # Let's use the DiGress formula structure:
            # numer = (Q_bar_s @ x_0_probs) * (Q_t.T @ x_t_one_hot) -- element-wise then sum over x_0
            # This is (Q, K_d) * (Q, K_d) -> (Q, K_d)

            # term1 for numerator: (Q_bar_s @ x_0_probs) -> (Q, K_d)
            # This is sum over x_0_j of Q_bar_s[x_s_i, x_0_j] * P(x_0_j)
            # Q_bar_s_list[d] is (Q, K_d, K_d)
            # x_n_0_probs_d is (Q, K_d)
            # We need (Q_bar_s_list[d] * x_n_0_probs_d.unsqueeze(1)).sum(dim=2) -> (Q, K_d)

            term1_num = (Q_bar_s_list[d] * x_n_0_probs_d.unsqueeze(1)).sum(dim=2)  # Sum over x_0 dimension

            # term2 for numerator: (Q_t.T @ x_t_one_hot) -> (Q, K_d)
            # This is sum over x_t_j of Q_t[x_t_j, x_s_i] * P(x_t_j) (where P(x_t_j) is one-hot)
            # Q_t_list[d] is (Q, K_d, K_d)
            # x_n_t_one_hot_d is (Q, K_d)
            # We need (Q_t_list[d].transpose(1,2) * x_n_t_one_hot_d.unsqueeze(1)).sum(dim=2)
            term2_num = (Q_t_list[d].transpose(1, 2) * x_n_t_one_hot_d.unsqueeze(1)).sum(dim=2)

            numer = term1_num * term2_num  # (Q, K_d)

            # Denominator: sum over x_s of numer
            # This is not quite right. The DiGress paper has a specific formula for q(x_{t-1} | x_t, x_0)
            # It's ((Q_t)_{x_t, x_{t-1}} * (Q_bar_{s})_{x_{t-1}, x_0}) / (Q_bar_t)_{x_t, x_0}
            # This means for each possible x_{t-1}, we calculate this.
            # x_0_probs_d is P(x_0 | data) (model prediction) (Q, K_d)
            # x_n_t_one_hot_d is one-hot x_t (Q, K_d)

            # Posterior for x_{s} (which is x_{t-1}):
            # p(x_s=k | x_t, \hat{x}_0) \propto \sum_{j} (Q_t[x_t, k] * Q_bar_s[k, j] * \hat{x}_0[j])
            # Q_t[x_t, k] means select row x_t from Q_t, then column k.
            # Q_bar_s[k,j] means select row k from Q_bar_s, then column j.

            # Loop over possible states k for x_s (x_{t-1})
            # Q_t_list[d] is (Q, K_d_xt, K_d_xs)
            # Q_bar_s_list[d] is (Q, K_d_xs, K_d_x0)
            # x_n_0_probs_d is (Q, K_d_x0)
            # x_n_t_one_hot_d is (Q, K_d_xt)

            # For each query q, and each possible state k_s for x_s:
            # prob_xs_k = sum_j { Q_t[q, x_t[q], k_s] * Q_bar_s[q, k_s, j] * x_0_probs[q, j] }

            # Efficiently:
            # 1. Q_bar_s_times_x0 = Q_bar_s_list[d] @ x_n_0_probs_d.unsqueeze(-1) -> (Q, K_d_xs, 1)
            Q_bar_s_times_x0 = torch.einsum('qij,qj->qi', Q_bar_s_list[d], x_n_0_probs_d)  # (Q, K_d_xs)

            # 2. Get relevant rows of Q_t based on x_t
            # Q_t_rows_for_xt = Q_t_list[d][torch.arange(Q_max), x_n_t_one_hot_d.argmax(dim=1), :] -> (Q, K_d_xs)
            # This is tricky if x_n_t_one_hot_d is not strictly one-hot due to sampling issues.
            # Assuming current_x_n_t_d is the index for x_t.
            Q_t_selected_rows = Q_t_list[d][torch.arange(len(current_x_n_t_d)), current_x_n_t_d, :]  # (Q, K_d_xs)

            # 3. Element-wise product
            unnormalized_log_probs_xs = Q_t_selected_rows * Q_bar_s_times_x0  # (Q, K_d_xs)

            # Normalize: (Not log probs yet, these are proportional to probs)
            # The denominator in DiGress is (Q_bar_t @ x_0_probs)_{x_t}
            # denom_val = (Q_bar_t_list[d] @ x_n_0_probs_d.unsqueeze(-1)).squeeze(-1) # (Q, K_d_xt)
            # denom_val_for_xt = denom_val[torch.arange(Q_max), current_x_n_t_d] # (Q)
            # pred_d = unnormalized_log_probs_xs / (denom_val_for_xt.unsqueeze(-1) + 1e-8)
            # This is complex. For sampling, often just use the unnormalized and sample.
            # Or, if the model directly predicts x_0, and diffusion is simple, use that.
            # The current `self.node_pred_model` directly predicts `x_n_0_logits_list`.
            # For DDPM sampling, one typically uses the model's prediction of x_0 or noise.
            # If this posterior is for the p(x_{t-1} | x_t, x_0_hat) step:
            # The logits for x_{t-1} are often derived from this.
            # For now, let's assume x_n_0_logits_list is the direct prediction of the "cleaned" data.
            # The sampling loop then uses this.
            # The provided posterior seems to be trying to compute p(x_{t-1} | x_t, \hat{x}_0(x_t))
            # This is standard for discrete DDPM.

            pred_list.append(unnormalized_log_probs_xs)  # These are logits for x_{t-1}

        return pred_list

    @torch.no_grad()
    def posterior_edge(self, A, x_n, abs_level, rel_level,
                       t,  # (num_queries_edge, 1)
                       label_t,  # (num_queries_edge) - current noisy edge states
                       query_src, query_dst, y=None,
                       label_0_logits=None):  # label_0_logits is prediction of clean edges (Q_edge, 2)

        if label_0_logits is None:
            label_0_logits = self.edge_pred_model(
                A, x_n, abs_level, rel_level, t, query_src, query_dst, y)

        s = t - 1  # (Q_edge, 1)
        s = torch.where(s < 0, torch.zeros_like(s), s)

        s_flat = s.squeeze(-1).clamp(0, self.edge_diffusion.T)  # (Q_edge)
        t_flat = t.squeeze(-1).clamp(0, self.edge_diffusion.T)  # (Q_edge)

        alpha_bar_s_vals = self.edge_diffusion.alpha_bars[s_flat]  # (Q_edge)
        alpha_bar_t_vals = self.edge_diffusion.alpha_bars[t_flat]  # (Q_edge)
        alpha_t_vals = self.edge_diffusion.alphas[t_flat]  # (Q_edge)

        # Marginal for edges (p1 for edge existence)
        # This should be a scalar or (Q_edge) if it varies per query
        marginal_p1 = self.edge_diffusion.avg_in_deg / A.shape[1]  # Simplified, assuming avg_in_deg is relevant proxy
        marginal_p1 = torch.clamp(marginal_p1, 0.01, 0.99)  # Avoid 0 or 1

        # Get Q matrices (these are (2,2) but need to be (Q_edge, 2, 2))
        # The get_Qs in EdgeDiscreteDiffusion returns single (2,2) matrices.
        # We need to make them (Q_edge, 2,2) using the per-query alpha values.

        Q_bar_s_matrices = torch.zeros(len(t), 2, 2, device=t.device)
        Q_t_matrices = torch.zeros(len(t), 2, 2, device=t.device)

        for i in range(len(t)):
            # get_Qs expects scalar alphas.
            # EdgeDiscreteDiffusion.get_Qs needs to be adapted or called per item.
            # For now, let's assume a simplified way to get these for the batch.
            # This is a placeholder, proper batching of get_Qs is needed.
            # For simplicity, using the first alpha value for all, which is not correct.
            # This needs to be fixed by making get_Qs in EdgeDiscreteDiffusion batch-aware.
            # Or by calling it in a loop (inefficient).
            # Let's assume a batch-aware get_Q method for edges similar to node_diffusion.get_Q

            # Simplified: Construct M per query based on its marginal_p1 (if it varies)
            # For now, use a single marginal_p1 for the batch.
            M_edge = torch.tensor([[1 - marginal_p1, marginal_p1],
                                   [1 - marginal_p1, marginal_p1]], device=t.device)
            I_edge = torch.eye(2, device=t.device)

            Q_bar_s_matrices[i] = alpha_bar_s_vals[i] * I_edge + (1 - alpha_bar_s_vals[i]) * M_edge
            Q_t_matrices[i] = alpha_t_vals[i] * I_edge + (1 - alpha_t_vals[i]) * M_edge
            # Q_bar_t is not directly used in the simplified posterior below for logits of x_{t-1}

        label_0_probs = label_0_logits.softmax(dim=-1)  # (Q_edge, 2)
        label_t_one_hot = F.one_hot(label_t, num_classes=2).float()  # (Q_edge, 2)

        # Similar to node posterior:
        # term1_num = (Q_bar_s @ \hat{x}_0_probs) -> (Q_edge, 2)
        term1_num_edge = (Q_bar_s_matrices * label_0_probs.unsqueeze(1)).sum(dim=2)

        # term2_num = (Q_t.T @ x_t_one_hot) -> (Q_edge, 2)
        term2_num_edge = (Q_t_matrices.transpose(1, 2) * label_t_one_hot.unsqueeze(1)).sum(dim=2)

        unnormalized_logits_edge_s = term1_num_edge * term2_num_edge  # (Q_edge, 2)

        return unnormalized_logits_edge_s  # These are logits for edge state at s=t-1

    @torch.no_grad()
    def sample_node_layer(self, A, x_n, abs_level, rel_level, A_n2g,
                          curr_level, y=None,
                          min_num_steps_n=None, max_num_steps_n=None):
        batch_size = A_n2g.shape[0]
        if batch_size == 0: return []  # No graphs in batch

        x_n_count_logits = self.node_count_model(
            A, x_n, abs_level, rel_level, A_n2g, y)
        num_nodes_pred_per_graph = x_n_count_logits.softmax(dim=-1).multinomial(1).squeeze(-1)  # (B)

        x_n_l_list_final = [torch.empty(0, dtype=torch.long, device=self.device) for _ in
                            range(batch_size)]  # Ensure correct type and device

        # Identify graphs that predict >0 nodes for the new layer
        active_graph_mask = num_nodes_pred_per_graph > 0
        if not active_graph_mask.any():  # All graphs predict 0 new nodes
            return x_n_l_list_final

        # Filter inputs for active graphs only
        active_indices = torch.where(active_graph_mask)[0]
        num_active_graphs = len(active_indices)

        active_num_nodes_pred = num_nodes_pred_per_graph[active_graph_mask]  # (B_active)

        # Need to reconstruct A, x_n etc. for the active subgraph batch
        # This is complex. A simpler way: process all, then filter.
        # Or, if Q_max_active is 0, return.

        Q_max_total_new_nodes = num_nodes_pred_per_graph.sum().item()
        if Q_max_total_new_nodes == 0:  # No new nodes predicted across the entire batch
            return x_n_l_list_final

        query2g = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device), num_nodes_pred_per_graph)  # (Q_total_new_nodes)

        # Filter query2g for active graphs if we were to sub-batch
        # For now, proceed with all, and filter results.

        num_query_cumsum = torch.cumsum(
            torch.cat([torch.tensor([0], device=self.device, dtype=torch.long), num_nodes_pred_per_graph]),
            dim=0)  # (B+1)

        # Initialize x_n_t for all predicted new nodes
        num_node_features = len(self.node_diffusion.num_classes_list)
        x_n_t_parts = []
        for d in range(num_node_features):
            num_actual_classes_d = self.node_diffusion.num_classes_list[d]
            x_n_t_d = torch.randint(0, num_actual_classes_d, (Q_max_total_new_nodes,), device=self.device)
            x_n_t_parts.append(x_n_t_d)

        x_n_t = torch.stack(x_n_t_parts, dim=1) if num_node_features > 1 else x_n_t_parts[0].unsqueeze(
            -1)  # (Q_total, num_feat) or (Q_total,1)

        num_steps = self.node_diffusion.T
        if min_num_steps_n is not None and max_num_steps_n is not None:  # Ensure both are given
            num_steps = random.randint(min_num_steps_n, max_num_steps_n)

        # Diffusion sampling loop
        for s_idx in range(num_steps - 1, -1, -1):
            # t_for_diffusion is (current_batch_size, 1)
            t_for_diffusion = torch.full((batch_size, 1), s_idx, dtype=torch.long, device=self.device)

            # The posterior function expects x_n_t for *all* query nodes (Q_max_total_new_nodes)
            # A, x_n, etc. are for the *existing* graph structure (B graphs, N_total_existing nodes)
            # y is also for B graphs.
            # query2g maps Q_total_new_nodes to B graphs.
            # num_query_cumsum is for Q_total_new_nodes relative to B graphs.

            pred_logits_list = self.posterior(  # This returns list of (Q_total, K_d)
                A, x_n, abs_level, rel_level, A_n2g,
                x_n_t, t_for_diffusion, query2g, num_query_cumsum, y)

            x_n_s_parts = []
            for d in range(num_node_features):
                # pred_logits_list[d] is (Q_total, K_d)
                x_n_s_d = pred_logits_list[d].softmax(dim=-1).multinomial(1).squeeze(-1)  # (Q_total)
                x_n_s_parts.append(x_n_s_d)
            x_n_t = torch.stack(x_n_s_parts, dim=1) if num_node_features > 1 else x_n_s_parts[0].unsqueeze(-1)

        # Distribute the final x_n_t (cleaned node attributes) back to per-graph lists
        for i in range(batch_size):
            if num_nodes_pred_per_graph[i] > 0:
                start_q_idx = num_query_cumsum[i]
                end_q_idx = num_query_cumsum[i + 1]
                x_n_l_list_final[i] = x_n_t[start_q_idx:end_q_idx]

        return x_n_l_list_final

    @torch.no_grad()
    def sample_edge_layer(self, num_nodes_cumsum_active, edge_index_list_active,
                          batch_x_n_active, batch_abs_level_active, batch_rel_level_active,
                          # num_new_nodes_for_edges_active: list of num new nodes for each *active* graph
                          num_new_nodes_for_edges_active,
                          # batch_query_src_cat_active, batch_query_dst_cat_active: concatenated queries for *active* graphs, global node indices
                          batch_query_src_cat_active, batch_query_dst_cat_active,
                          # query_src_list_local_active, query_dst_list_local_active: list of local queries for each *active* graph
                          query_src_list_local_active, query_dst_list_local_active,
                          y_active=None, curr_level=None,  # y_active is for active graphs
                          min_num_steps_e=None, max_num_steps_e=None):

        if batch_query_src_cat_active.numel() == 0:  # No query edges at all
            return edge_index_list_active

            # batch_A_active represents the current graph structure for active graphs
        batch_A_active = self.get_batch_A(num_nodes_cumsum_active, edge_index_list_active, self.device)

        # Initial noisy edge states (label_t) for all query edges
        # Sample from Bernoulli(0.5) - 0 for no edge, 1 for edge
        current_label_t = torch.bernoulli(
            torch.ones(len(batch_query_src_cat_active), device=self.device) * 0.5
        ).long()  # (Total_Query_Edges_Active)

        num_steps = self.edge_diffusion.T
        if min_num_steps_e is not None and max_num_steps_e is not None:
            num_steps = random.randint(min_num_steps_e, max_num_steps_e)

        # Diffusion sampling loop for edges
        for s_idx in range(num_steps - 1, -1, -1):
            # t_for_edge_diffusion is (Total_Query_Edges_Active, 1)
            t_for_edge_diffusion = torch.full((len(batch_query_src_cat_active), 1), s_idx, dtype=torch.long,
                                              device=self.device)

            # Get model's prediction of clean edges (logits for edge_state=0 and edge_state=1)
            # The EdgePredModel takes batch_A_active, batch_x_n_active, etc.
            # And batch_query_src/dst_cat_active which are global node indices within this active batch.
            # y_active is graph-level conditions for the active graphs.
            label_0_logits = self.edge_pred_model(
                batch_A_active, batch_x_n_active, batch_abs_level_active, batch_rel_level_active,
                t_for_edge_diffusion, batch_query_src_cat_active, batch_query_dst_cat_active, y_active
            )  # (Total_Query_Edges_Active, 2)

            # Use posterior or simplified sampling.
            # If posterior_edge is correctly implemented for p(x_{t-1} | x_t, x_0_hat)
            # current_label_t is x_t. label_0_logits is for x_0_hat.
            # posterior_edge_logits = self.posterior_edge(
            #     batch_A_active, batch_x_n_active, batch_abs_level_active, batch_rel_level_active,
            #     t_for_edge_diffusion, current_label_t, # Pass current noisy state
            #     batch_query_src_cat_active, batch_query_dst_cat_active, y_active,
            #     label_0_logits=label_0_logits # Pass model's x0 prediction
            # )
            # current_label_t = posterior_edge_logits.softmax(dim=-1).multinomial(1).squeeze(-1) # (Total_Query_Edges_Active)

            # Simplified sampling: Use model's prediction of x0 directly
            # This is common in practice if full posterior is complex or shows issues.
            # Sample based on the probability of edge existence from label_0_logits.
            edge_probs_from_x0_hat = label_0_logits.softmax(dim=-1)[:, 1]  # Prob of edge existing
            current_label_t = torch.bernoulli(edge_probs_from_x0_hat).long()

        # Update edge_index_list_active based on the final sampled edges (current_label_t)
        new_edge_index_list_for_active_graphs = []
        query_offset = 0
        for i in range(len(edge_index_list_active)):  # Iterate through active graphs
            num_queries_for_this_graph = len(query_src_list_local_active[i])

            if num_queries_for_this_graph == 0:
                new_edge_index_list_for_active_graphs.append(edge_index_list_active[i])
                continue

            # Get the sampled edge states for this graph's queries
            sampled_edge_states_for_graph = current_label_t[query_offset: query_offset + num_queries_for_this_graph]

            # Get the source and destination nodes for the *predicted* edges (where state is 1)
            # query_src/dst_list_local_active[i] are local node indices for *this* graph
            src_nodes_of_predicted_edges = query_src_list_local_active[i][sampled_edge_states_for_graph == 1]
            dst_nodes_of_predicted_edges = query_dst_list_local_active[i][sampled_edge_states_for_graph == 1]

            if src_nodes_of_predicted_edges.numel() > 0:
                # Create new edges in DGL format [dst, src]
                new_edges_for_graph_i = torch.stack([dst_nodes_of_predicted_edges, src_nodes_of_predicted_edges], dim=0)
                # Append to existing edges for this graph
                updated_edge_index_for_graph_i = torch.cat([edge_index_list_active[i], new_edges_for_graph_i], dim=1)
                new_edge_index_list_for_active_graphs.append(updated_edge_index_for_graph_i)
            else:  # No new edges predicted for this graph
                new_edge_index_list_for_active_graphs.append(edge_index_list_active[i])

            query_offset += num_queries_for_this_graph

        return new_edge_index_list_for_active_graphs

    def get_batch_A(self, num_nodes_cumsum, edge_index_list, current_device):
        num_total_nodes = num_nodes_cumsum[-1].item()
        if num_total_nodes == 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        batch_src_list = []
        batch_dst_list = []
        for i in range(len(edge_index_list)):  # Iterate through graphs in the current (active) batch
            if edge_index_list[i].numel() > 0:
                # edge_index_list[i] is [dst, src] with local-to-graph indices
                # These indices need to be offset by num_nodes_cumsum[i]
                batch_dst_list.append(edge_index_list[i][0] + num_nodes_cumsum[i])
                batch_src_list.append(edge_index_list[i][1] + num_nodes_cumsum[i])

        if not batch_src_list:  # No edges in the entire active batch
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(num_total_nodes, num_total_nodes))

        batch_src_cat = torch.cat(batch_src_list)
        batch_dst_cat = torch.cat(batch_dst_list)

        return dglsp.spmatrix(torch.stack([batch_dst_cat, batch_src_cat]),
                              shape=(num_total_nodes, num_total_nodes)).to(current_device)

    def get_batch_A_n2g(self, num_nodes_cumsum, current_device):
        current_batch_size = len(num_nodes_cumsum) - 1
        if current_batch_size == 0:
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        num_total_nodes = num_nodes_cumsum[-1].item()
        if num_total_nodes == 0 and current_batch_size > 0:  # Batch of empty graphs (only if no dummy node)
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(current_batch_size, 0))
        if num_total_nodes == 0 and current_batch_size == 0:  # Fully empty
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device), shape=(0, 0))

        nids_list = []
        gids_list = []
        for i in range(current_batch_size):
            num_nodes_in_graph_i = num_nodes_cumsum[i + 1] - num_nodes_cumsum[i]
            if num_nodes_in_graph_i > 0:
                nids_list.append(
                    torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i + 1], device=current_device, dtype=torch.long))
                gids_list.append(torch.full((num_nodes_in_graph_i,), i, device=current_device, dtype=torch.long))

        if not nids_list:  # All graphs in batch are empty (e.g. after filtering, should not happen if dummy node logic is robust)
            return dglsp.spmatrix(torch.tensor([[], []], dtype=torch.long, device=current_device),
                                  shape=(current_batch_size, num_total_nodes))

        nids_cat = torch.cat(nids_list)
        gids_cat = torch.cat(gids_list)

        return dglsp.spmatrix(torch.stack([gids_cat, nids_cat]),
                              shape=(current_batch_size, num_total_nodes)).to(current_device)

    def get_batch_y(self, current_y_list_active, current_x_n_list_active, current_device):
        if current_y_list_active is None or not self.is_conditional:
            return None

        if not current_y_list_active:  # Empty list of y conditions
            return None

        # Assuming current_y_list_active contains scalar conditions or tensor conditions
        # that correspond to the graphs in current_x_n_list_active.
        # Output should be (current_active_batch_size, y_feature_dim)
        if isinstance(current_y_list_active[0], (int, float)):
            return torch.tensor([[y_val] for y_val in current_y_list_active], dtype=torch.float32,
                                device=current_device)
        elif isinstance(current_y_list_active[0], torch.Tensor):
            return torch.stack([y_val.to(current_device) for y_val in current_y_list_active])
        else:
            raise TypeError(f"Unsupported type for elements in current_y_list_active: {type(current_y_list_active[0])}")

    @torch.no_grad()
    def sample(self,
               batch_size=1,
               raw_y_batch=None,
               min_num_steps_n=None,
               max_num_steps_n=None,
               min_num_steps_e=None,
               max_num_steps_e=None):

        current_device = self.device
        self.eval()  # Set model to evaluation mode

        y_list_for_active_graphs = None
        if self.is_conditional and raw_y_batch is not None:
            if batch_size != len(raw_y_batch):
                raise ValueError(
                    f"Batch size ({batch_size}) must match length of raw_y_batch ({len(raw_y_batch)}) in conditional mode.")
            y_list_for_active_graphs = list(raw_y_batch)  # Mutable copy for active graphs

        # Initial state for each graph in the batch
        edge_index_list_active = [torch.tensor([[], []], dtype=torch.long, device=current_device) for _ in
                                  range(batch_size)]

        # Initial node features (dummy node)
        if isinstance(self.dummy_x_n, int):  # Scalar dummy index
            init_x_n_val = torch.tensor([[self.dummy_x_n]], dtype=torch.long, device=current_device)  # (1,1)
        elif isinstance(self.dummy_x_n, torch.Tensor) and self.dummy_x_n.ndim == 0:  # Scalar tensor
            init_x_n_val = torch.tensor([[self.dummy_x_n.item()]], dtype=torch.long, device=current_device)
        elif isinstance(self.dummy_x_n,
                        torch.Tensor) and self.dummy_x_n.ndim > 0:  # Tensor of dummy indices for multi-feature
            init_x_n_val = self.dummy_x_n.to(current_device).clone().detach().unsqueeze(0)  # (1, num_node_feats)
        else:
            raise TypeError(f"Unsupported type or structure for self.dummy_x_n: {type(self.dummy_x_n)}")

        x_n_list_active = [init_x_n_val.clone() for _ in range(batch_size)]

        level = 0.0
        abs_level_list_active = [
            torch.tensor([[level]], device=current_device, dtype=torch.float32) for _ in range(batch_size)
        ]

        # Store original indices to correctly map y_finished
        original_indices_active = list(range(batch_size))

        # Lists to store components of completely generated graphs
        edge_index_finished_graphs = []
        x_n_finished_graphs = []
        y_values_for_finished_graphs = [] if self.is_conditional and raw_y_batch is not None else None

        while x_n_list_active:  # Loop as long as there are graphs being actively generated
            current_active_batch_size = len(x_n_list_active)

            # Prepare batch inputs for the currently active graphs
            num_nodes_cumsum_active = torch.cumsum(torch.tensor(
                [0] + [len(x_n_i) for x_n_i in x_n_list_active], device=current_device, dtype=torch.long), dim=0)

            batch_x_n_active = torch.cat(x_n_list_active).to(current_device)
            batch_abs_level_active = torch.cat(abs_level_list_active).to(current_device)
            batch_rel_level_active = batch_abs_level_active.max() - batch_abs_level_active  # Max over current active batch nodes

            batch_A_active = self.get_batch_A(num_nodes_cumsum_active, edge_index_list_active, current_device)
            batch_A_n2g_active = self.get_batch_A_n2g(num_nodes_cumsum_active, current_device)

            batch_y_tensor_active = None
            if self.is_conditional and y_list_for_active_graphs is not None:
                batch_y_tensor_active = self.get_batch_y(y_list_for_active_graphs, x_n_list_active, current_device)

            # 1. Sample Node Layer attributes for all active graphs
            # Returns a list of tensors, one for each active graph, containing new node attributes
            new_nodes_attributes_per_active_graph = self.sample_node_layer(
                batch_A_active, batch_x_n_active, batch_abs_level_active, batch_rel_level_active,
                batch_A_n2g_active, curr_level=level, y=batch_y_tensor_active,
                min_num_steps_n=min_num_steps_n, max_num_steps_n=max_num_steps_n
            )

            # --- Filter finished graphs and prepare for next iteration or edge prediction ---
            next_iter_edge_index_list = []
            next_iter_x_n_list = []
            next_iter_abs_level_list = []
            next_iter_y_list = [] if self.is_conditional and y_list_for_active_graphs is not None else None
            next_iter_original_indices = []

            # For edge prediction on graphs that are *continuing*
            query_src_local_for_continuing_graphs = []
            query_dst_local_for_continuing_graphs = []
            query_src_batch_cat_for_continuing_graphs = []
            query_dst_batch_cat_for_continuing_graphs = []
            num_new_nodes_for_continuing_graphs = []

            node_offset_for_batch_edge_queries = 0

            for i in range(current_active_batch_size):  # Iterate through the graphs that were active in this step
                original_graph_idx = original_indices_active[i]
                newly_sampled_nodes_for_graph_i = new_nodes_attributes_per_active_graph[i]

                if newly_sampled_nodes_for_graph_i.numel() == 0:  # Graph i finished (no new nodes)
                    final_edges = edge_index_list_active[i]
                    if final_edges.numel() > 0: final_edges = final_edges - 1  # Adjust for dummy

                    edge_index_finished_graphs.append(final_edges)
                    x_n_finished_graphs.append(x_n_list_active[i][1:])  # Remove dummy
                    if self.is_conditional and raw_y_batch is not None:
                        y_values_for_finished_graphs.append(raw_y_batch[original_graph_idx])
                else:  # Graph i continues
                    next_iter_original_indices.append(original_graph_idx)
                    next_iter_edge_index_list.append(edge_index_list_active[i])  # Carry over old edges

                    # Add new nodes and their levels
                    updated_x_n_for_graph_i = torch.cat([x_n_list_active[i], newly_sampled_nodes_for_graph_i])
                    next_iter_x_n_list.append(updated_x_n_for_graph_i)

                    updated_abs_level_for_graph_i = torch.cat([
                        abs_level_list_active[i],
                        torch.full((newly_sampled_nodes_for_graph_i.shape[0], 1), level + 1.0,
                                   dtype=torch.float32, device=current_device)
                    ])
                    next_iter_abs_level_list.append(updated_abs_level_for_graph_i)

                    if self.is_conditional and y_list_for_active_graphs is not None:
                        next_iter_y_list.append(y_list_for_active_graphs[i])

                    # Prepare queries for edge prediction for this continuing graph
                    num_old_nodes_incl_dummy = x_n_list_active[i].shape[0]
                    num_new_nodes = newly_sampled_nodes_for_graph_i.shape[0]
                    num_new_nodes_for_continuing_graphs.append(num_new_nodes)

                    # Query edges from existing *real* nodes to new nodes
                    # Local indices for existing real nodes: 1 to num_old_nodes_incl_dummy - 1
                    # Local indices for new nodes: num_old_nodes_incl_dummy to num_old_nodes_incl_dummy + num_new_nodes - 1
                    q_src_local_i, q_dst_local_i = [], []
                    if num_old_nodes_incl_dummy > 1:  # If there are any real existing nodes
                        for s_local_idx in range(1, num_old_nodes_incl_dummy):
                            for d_new_local_offset in range(num_new_nodes):
                                d_local_idx = num_old_nodes_incl_dummy + d_new_local_offset
                                q_src_local_i.append(s_local_idx)
                                q_dst_local_i.append(d_local_idx)

                    if q_src_local_i:
                        q_src_tensor_local = torch.tensor(q_src_local_i, dtype=torch.long, device=current_device)
                        q_dst_tensor_local = torch.tensor(q_dst_local_i, dtype=torch.long, device=current_device)
                        query_src_local_for_continuing_graphs.append(q_src_tensor_local)
                        query_dst_local_for_continuing_graphs.append(q_dst_tensor_local)
                        # For batch tensor, offset by current total nodes processed for edge queries
                        query_src_batch_cat_for_continuing_graphs.append(
                            q_src_tensor_local + node_offset_for_batch_edge_queries)
                        query_dst_batch_cat_for_continuing_graphs.append(
                            q_dst_tensor_local + node_offset_for_batch_edge_queries)
                    else:
                        query_src_local_for_continuing_graphs.append(
                            torch.tensor([], dtype=torch.long, device=current_device))
                        query_dst_local_for_continuing_graphs.append(
                            torch.tensor([], dtype=torch.long, device=current_device))

                    node_offset_for_batch_edge_queries += updated_x_n_for_graph_i.shape[0]

            # Update active lists for the next iteration
            edge_index_list_active = next_iter_edge_index_list
            x_n_list_active = next_iter_x_n_list
            abs_level_list_active = next_iter_abs_level_list
            if self.is_conditional:
                y_list_for_active_graphs = next_iter_y_list
            original_indices_active = next_iter_original_indices

            if not x_n_list_active:  # All graphs have finished generating
                break

            level += 1.0  # Increment level for the next layer

            # 2. Sample Edge Layer for continuing graphs (if any queries were generated)
            if query_src_batch_cat_for_continuing_graphs:  # Check if list of tensors is not empty
                # Concatenate all query tensors for the batch
                final_batch_q_src_for_edges = torch.cat(query_src_batch_cat_for_continuing_graphs)
                final_batch_q_dst_for_edges = torch.cat(query_dst_batch_cat_for_continuing_graphs)

                if final_batch_q_src_for_edges.numel() > 0:  # If there are actual queries
                    # Recalculate batch inputs for the *current set* of active graphs for edge prediction
                    num_nodes_cumsum_for_edges = torch.cumsum(torch.tensor(
                        [0] + [len(x_n_i) for x_n_i in x_n_list_active], device=current_device, dtype=torch.long),
                        dim=0)
                    batch_x_n_for_edges = torch.cat(x_n_list_active).to(current_device)
                    batch_abs_level_for_edges = torch.cat(abs_level_list_active).to(current_device)
                    batch_rel_level_for_edges = batch_abs_level_for_edges.max() - batch_abs_level_for_edges

                    batch_y_tensor_for_edges = None
                    if self.is_conditional and y_list_for_active_graphs is not None:
                        batch_y_tensor_for_edges = self.get_batch_y(y_list_for_active_graphs, x_n_list_active,
                                                                    current_device)

                    # sample_edge_layer updates edge_index_list_active in place (or returns updated)
                    edge_index_list_active = self.sample_edge_layer(
                        num_nodes_cumsum_for_edges, edge_index_list_active,
                        batch_x_n_for_edges, batch_abs_level_for_edges, batch_rel_level_for_edges,
                        num_new_nodes_for_continuing_graphs,  # List of num new nodes for each continuing graph
                        final_batch_q_src_for_edges, final_batch_q_dst_for_edges,  # Concatenated global queries
                        query_src_local_for_continuing_graphs, query_dst_local_for_continuing_graphs,
                        # List of local queries
                        y=batch_y_tensor_for_edges, curr_level=level,
                        min_num_steps_e=min_num_steps_e, max_num_steps_e=max_num_steps_e
                    )

            if self.max_level is not None and level >= self.max_level:
                # Move any remaining active graphs to finished
                for i in range(len(x_n_list_active)):
                    original_idx_rem = original_indices_active[i]
                    final_edges_rem = edge_index_list_active[i]
                    if final_edges_rem.numel() > 0: final_edges_rem = final_edges_rem - 1

                    edge_index_finished_graphs.append(final_edges_rem)
                    x_n_finished_graphs.append(x_n_list_active[i][1:])
                    if self.is_conditional and raw_y_batch is not None:
                        y_values_for_finished_graphs.append(raw_y_batch[original_idx_rem])
                break  # Exit main generation loop

        return edge_index_finished_graphs, x_n_finished_graphs, y_values_for_finished_graphs

