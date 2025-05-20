# src/model/layer_dag.py
import dgl.sparse as dglsp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            return torch.zeros(len(position), 0).to(position.device)

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
        if A.nnz == 0:
            h_n_out = self.W_self(h_n)
        else:
            h_n_out = A @ self.W(h_n) + A_T @ self.W_trans(h_n) +\
                self.W_self(h_n)
        return F.gelu(h_n_out)

class OneHotPE(nn.Module):
    def __init__(self, pe_size):
        super().__init__()

        self.pe_size = pe_size

    def forward(self, position):
        if self.pe_size == 0:
            return torch.zeros(len(position), 0).to(position.device)

        return F.one_hot(position.clamp(max=self.pe_size - 1).long().squeeze(-1),
                         num_classes=self.pe_size)

class MultiEmbedding(nn.Module):
    def __init__(self, num_x_n_cat, hidden_size):
        super().__init__()

        self.emb_list = nn.ModuleList([
            nn.Embedding(num_x_n_cat_i, hidden_size)
            for num_x_n_cat_i in num_x_n_cat.tolist()
        ])

    def forward(self, x_n_cat):
        if len(x_n_cat.shape) == 1:
            # ---- START DEBUG PRINTS ----
            print(f"DEBUG: MultiEmbedding input x_n_cat device: {x_n_cat.device}")
            print(f"DEBUG: MultiEmbedding input x_n_cat dtype: {x_n_cat.dtype}")
            if x_n_cat.numel() > 0:  # Avoid errors on empty tensors
                print(f"DEBUG: MultiEmbedding input x_n_cat min: {x_n_cat.min().item()}")
                print(f"DEBUG: MultiEmbedding input x_n_cat max: {x_n_cat.max().item()}")
            else:
                print(f"DEBUG: MultiEmbedding input x_n_cat is EMPTY. Shape: {x_n_cat.shape}")
            print(f"DEBUG: MultiEmbedding emb_list[0].num_embeddings: {self.emb_list[0].num_embeddings}")
            # ---- END DEBUG PRINTS ----
            x_n_emb = self.emb_list[0](x_n_cat)
        else:
            x_n_emb = torch.cat([
                self.emb_list[i](x_n_cat[:, i]) for i in range(len(self.emb_list))
            ], dim=1)

        return x_n_emb

class BiMPNNEncoder(nn.Module):
    def __init__(self,
                 num_x_n_cat,
                 x_n_emb_size,
                 pe_emb_size,
                 hidden_size,
                 num_mpnn_layers,
                 pe=None,
                 y_emb_size=0,
                 pool=None):
        super().__init__()

        self.pe = pe
        if self.pe in ['relative_level', 'abs_level']:
            self.level_emb = SinusoidalPE(pe_emb_size)
        elif self.pe == 'relative_level_one_hot':
            self.level_emb = OneHotPE(pe_emb_size)

        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)
        self.y_emb = SinusoidalPE(y_emb_size)

        self.proj_input = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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
        A_T = A.T
        h_n = self.x_n_emb(x_n)

        if self.pe == 'abs_level':
            node_pe = self.level_emb(abs_level)

        if self.pe in ['relative_level', 'relative_level_one_hot']:
            node_pe = self.level_emb(rel_level)

        if self.pe is not None:
            h_n = torch.cat([h_n, node_pe], dim=-1)

        if y is not None:
            h_y = self.y_emb(y)
            h_n = torch.cat([h_n, h_y], dim=-1)

        h_n = self.proj_input(h_n)
        h_n_cat = [h_n]
        for layer in self.mpnn_layers:
            h_n = layer(A, A_T, h_n)
            h_n_cat.append(h_n)
        h_n = torch.cat(h_n_cat, dim=-1)
        h_n = self.project_output_n(h_n)

        if self.pool is None:
            return h_n
        elif self.pool == 'sum':
            h_g = A_n2g @ h_n
            return self.bn_g(h_g)
        elif self.pool == 'mean':
            h_g = A_n2g @ h_n
            h_g = h_g / A_n2g.sum(dim=1).unsqueeze(-1)
            return self.bn_g(h_g)

class GraphClassifier(nn.Module):
    def __init__(self,
                 graph_encoder,
                 emb_size,
                 num_classes):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.predictor = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, y=None):
        h_g = self.graph_encoder(A, x_n, abs_level, rel_level, y, A_n2g)
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

    def attn(self, q, k, v, num_query_cumsum):
        """
        Parameters
        ----------
        q : torch.Tensor of shape (N, F)
            Query matrix for node representations.
        k : torch.Tensor of shape (N, F)
            Key matrix for node representations.
        v : torch.Tensor of shape (N, F)
            Value matrix for node representations.
        num_query_cumsum : torch.Tensor of shape (B + 1)
            num_query_cumsum[0] is 0, num_query_cumsum[i] is the number of queries
            for the first i graphs in the batch for i > 0.

        Returns
        -------
        torch.Tensor of shape (N, F)
            Updated hidden representations of query nodes for the batch of graphs.
        """
        # Handle different numbers of query nodes in the batch with padding.
        batch_size = len(num_query_cumsum) - 1
        num_query_nodes = torch.diff(num_query_cumsum)
        max_num_nodes = num_query_nodes.max().item()

        q_padded = q.new_zeros(batch_size, max_num_nodes, q.shape[-1])
        k_padded = k.new_zeros(batch_size, max_num_nodes, k.shape[-1])
        v_padded = v.new_zeros(batch_size, max_num_nodes, v.shape[-1])
        pad_mask = q.new_zeros(batch_size, max_num_nodes).bool()

        for i in range(batch_size):
            q_padded[i, :num_query_nodes[i]] = q[num_query_cumsum[i]:num_query_cumsum[i + 1]]
            k_padded[i, :num_query_nodes[i]] = k[num_query_cumsum[i]:num_query_cumsum[i + 1]]
            v_padded[i, :num_query_nodes[i]] = v[num_query_cumsum[i]:num_query_cumsum[i + 1]]
            pad_mask[i, num_query_nodes[i]:] = True

        # Split F into H * D, where H is the number of heads
        # D is the dimension per head.

        # (B, H, max_num_nodes, D)
        q_padded = rearrange(q_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        # (B, H, max_num_nodes, D)
        k_padded = rearrange(k_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        # (B, H, max_num_nodes, D)
        v_padded = rearrange(v_padded, 'b n (h d) -> b h n d', h=self.num_heads)

        # Q * K^T / sqrt(D)
        # (B, H, max_num_nodes, max_num_nodes)
        dot = torch.matmul(q_padded, k_padded.transpose(-1, -2)) * self.scale
        # Mask unnormalized attention logits for non-existent nodes.
        dot = dot.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        attn_scores = F.softmax(dot, dim=-1)
        # (B, H, max_num_nodes, D)
        h_n_padded = torch.matmul(attn_scores, v_padded)
        # (B * max_num_nodes, H * D) = (B * max_num_nodes, F)
        h_n_padded = rearrange(h_n_padded, 'b h n d -> (b n) (h d)')

        # Unpad the aggregation results.
        # (N, F)
        pad_mask = (~pad_mask).reshape(-1)
        return h_n_padded[pad_mask]

    def forward(self, h_n, num_query_cumsum):
        # Compute value matrix
        v_n = self.to_v(h_n)

        # Compute query and key matrices
        q_n, k_n = self.to_qk(h_n).chunk(2, dim=-1)

        h_n_new = self.attn(q_n, k_n, v_n, num_query_cumsum)
        h_n_new = self.proj_new(h_n_new)

        # Add & Norm
        h_n = self.norm1(h_n + h_n_new)
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
        num_real_classes = num_x_n_cat - 1
        self.x_n_emb = MultiEmbedding(num_real_classes, x_n_emb_size)
        self.t_emb = SinusoidalPE(t_emb_size)
        in_hidden_size = in_hidden_size + t_emb_size + len(num_real_classes) * x_n_emb_size
        self.project_h_n = nn.Sequential(
            nn.Linear(in_hidden_size, out_hidden_size),
            nn.GELU()
        )

        self.trans_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.trans_layers.append(TransformerLayer(
                out_hidden_size, num_heads, dropout
            ))

        self.pred_list = nn.ModuleList([])
        num_real_classes = num_real_classes.tolist()
        for num_classes_f in num_real_classes:
            self.pred_list.append(nn.Sequential(
                nn.Linear(out_hidden_size, out_hidden_size),
                nn.GELU(),
                nn.Linear(out_hidden_size, num_classes_f)
            ))

    def forward_with_h_g(self, h_g, x_n_t,
                         t, query2g, num_query_cumsum):
        h_t = self.t_emb(t)
        h_g = torch.cat([h_g, h_t], dim=1)

        h_n_t = self.x_n_emb(x_n_t)
        h_n_t = torch.cat([h_n_t, h_g[query2g]], dim=1)
        h_n_t = self.project_h_n(h_n_t)

        for trans_layer in self.trans_layers:
            h_n_t = trans_layer(h_n_t, num_query_cumsum)

        pred = []
        for d in range(len(self.pred_list)):
            pred.append(self.pred_list[d](h_n_t))

        return pred

    def forward(self, A, x_n, abs_level, rel_level, A_n2g, x_n_t,
                t, query2g, num_query_cumsum, y=None):
        """
        Parameters
        ----------
        x_n_t : torch.LongTensor of shape (Q)
        t : torch.LongTensor of shape (B, 1)
        query2g : torch.LongTensor of shape (Q)
        num_query_cumsum : torch.LongTensor of shape (B + 1)
        """
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
        self.pred = nn.Sequential(
            nn.Linear(2 * in_hidden_size + t_emb_size, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, 2)
        )

    def forward(self, A, x_n, abs_level, rel_level, t,
                query_src, query_dst, y=None):
        """
        t : torch.tensor of shape (num_queries, 1)
        """
        h_n = self.graph_encoder(A, x_n, abs_level, rel_level, y=y)

        h_e = torch.cat([
            self.t_emb(t),
            h_n[query_src],
            h_n[query_dst]
        ], dim=-1)

        return self.pred(h_e)

# ... (SinusoidalPE, BiMPNNLayer, OneHotPE, MultiEmbedding, BiMPNNEncoder, GraphClassifier, TransformerLayer, NodePredModel, EdgePredModel classes remain the same) ...

class LayerDAG(nn.Module):
    def __init__(self,
                 device,  # Added device argument here
                 num_x_n_cat,
                 node_count_encoder_config,
                 max_layer_size,
                 node_diffusion,
                 node_pred_graph_encoder_config,
                 node_predictor_config,
                 edge_diffusion,
                 edge_pred_graph_encoder_config,
                 edge_predictor_config,
                 is_conditional,  # <--- ADD THIS ARGUMENT
                 max_level=None):
        """
        Parameters
        ----------
        device : torch.device
            The device to place the model and its submodules on.
        num_x_n_cat :
            Case1: int
            Case2: torch.LongTensor of shape (num_feats)
        is_conditional : bool
            Flag indicating whether the model operates in conditional mode.
        """
        super().__init__()

        self.device = device  # Store the device
        self.is_conditional = is_conditional  # <--- STORE THIS FLAG

        if isinstance(num_x_n_cat, int):
            # Ensure num_x_n_cat is a tensor for MultiEmbedding and other parts
            num_x_n_cat_tensor = torch.LongTensor([num_x_n_cat])
        elif isinstance(num_x_n_cat, torch.Tensor):
            num_x_n_cat_tensor = num_x_n_cat
        else:
            raise TypeError("num_x_n_cat must be an int or a torch.LongTensor")

        # Ensure dummy_x_n is correctly derived for MultiEmbedding compatibility
        # If num_x_n_cat_tensor represents [actual_cat_count_feat1, actual_cat_count_feat2, ...]
        # and DAGDataset's num_categories is actual_cat_count + 1 (for the dummy)
        # then the input to MultiEmbedding (num_real_classes) in NodePredModel needs to be actual_cat_count.
        # self.dummy_x_n is used for initializing the first node.
        # If num_x_n_cat_tensor has one element (total categories including dummy),
        # then dummy is num_x_n_cat_tensor[0] - 1.
        if len(num_x_n_cat_tensor) == 1:
            self.dummy_x_n = (num_x_n_cat_tensor[0] - 1).item()  # a scalar index for the dummy category
        else:
            # For multi-feature, dummy_x_n could be a tensor of dummy indices for each feature
            self.dummy_x_n = num_x_n_cat_tensor - 1  # This assumes num_x_n_cat_tensor are counts *including* a dummy placeholder per feature
            # Or, if num_x_n_cat_tensor is from train_set.num_categories (which is actual+1)
            # and it's a single value, then self.dummy_x_n should be num_x_n_cat_tensor -1
            # The original code had: self.dummy_x_n = num_x_n_cat - 1
            # Let's stick to the simpler interpretation for now, assuming num_x_n_cat
            # passed here is the one from model_config which is train_set.num_categories
            # (which is actual_node_types + 1)
            self.dummy_x_n = num_x_n_cat_tensor - 1  # This will be a tensor if num_x_n_cat_tensor is a tensor

        # Node Count Model
        # Calculate hidden_size based on actual embedding output dimensions
        # BiMPNNEncoder's x_n_emb is MultiEmbedding(num_x_n_cat_tensor, x_n_emb_size)
        # If num_x_n_cat_tensor is [C1, C2], x_n_emb_size is H, output is len(num_x_n_cat_tensor) * H
        # If num_x_n_cat_tensor is a single value C (total cats incl dummy), output is H

        # The original code's hidden_size calculation for BiMPNNEncoder:
        # hidden_size = len(num_x_n_cat) * node_count_encoder_config['x_n_emb_size'] +\
        #     node_count_encoder_config['pe_emb_size'] +\
        #     node_count_encoder_config['y_emb_size']
        # This assumes num_x_n_cat is a list/tensor of categories per feature.
        # If num_x_n_cat (from model_config) is a single int (total_cats_incl_dummy),
        # then MultiEmbedding(num_x_n_cat_tensor, x_n_emb_size) will use num_x_n_cat_tensor[0]
        # and output x_n_emb_size.

        # Let's adjust hidden_size calculation for BiMPNNEncoder input projection
        # The x_n_emb output dim for MultiEmbedding is:
        #   - x_n_emb_size if num_x_n_cat_tensor is a scalar/single-element tensor
        #   - len(num_x_n_cat_tensor) * x_n_emb_size if num_x_n_cat_tensor has multiple elements

        # For node_count_model
        nc_x_n_actual_emb_size = node_count_encoder_config['x_n_emb_size']
        if len(num_x_n_cat_tensor) > 1 and num_x_n_cat_tensor.ndim > 0:  # If num_x_n_cat is a list of counts per feature
            nc_x_n_actual_emb_size *= len(num_x_n_cat_tensor)

        nc_pe_emb_size = node_count_encoder_config.get('pe_emb_size', 0)  # Default to 0 if not specified
        nc_y_emb_size = node_count_encoder_config.get('y_emb_size', 0) if self.is_conditional else 0

        nc_hidden_size_input_to_bimpnn = nc_x_n_actual_emb_size + nc_pe_emb_size + nc_y_emb_size

        node_count_encoder = BiMPNNEncoder(
            num_x_n_cat_tensor,  # Pass the tensor here
            hidden_size=nc_hidden_size_input_to_bimpnn,
            # This is the combined input feature dim for BiMPNN's proj_input
            **node_count_encoder_config
        ).to(self.device)

        self.node_count_model = GraphClassifier(
            node_count_encoder,
            emb_size=nc_hidden_size_input_to_bimpnn,
            # The output size of BiMPNNEncoder (before pooling if any, or after if it's the feature dim)
            # GraphClassifier takes graph_encoder's output, which is hidden_size
            num_classes=max_layer_size + 1
        ).to(self.device)

        # Node Prediction Model
        self.node_diffusion = node_diffusion
        np_x_n_actual_emb_size = node_pred_graph_encoder_config['x_n_emb_size']
        if len(num_x_n_cat_tensor) > 1 and num_x_n_cat_tensor.ndim > 0:
            np_x_n_actual_emb_size *= len(num_x_n_cat_tensor)

        np_pe_emb_size = node_pred_graph_encoder_config.get('pe_emb_size', 0)
        np_y_emb_size = node_pred_graph_encoder_config.get('y_emb_size', 0) if self.is_conditional else 0
        np_hidden_size_input_to_bimpnn = np_x_n_actual_emb_size + np_pe_emb_size + np_y_emb_size

        node_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat_tensor,
            hidden_size=np_hidden_size_input_to_bimpnn,
            **node_pred_graph_encoder_config
        ).to(self.device)

        # NodePredModel's num_x_n_cat should be the actual categories (num_x_n_cat_tensor - 1 if dummy is included)
        # And its x_n_emb_size is the one from node_pred_graph_encoder_config
        self.node_pred_model = NodePredModel(
            node_pred_graph_encoder,
            num_x_n_cat=num_x_n_cat_tensor - 1,  # Pass actual categories (num_total_categories - dummy_category)
            x_n_emb_size=node_pred_graph_encoder_config['x_n_emb_size'],
            # This is per-feature embedding size for its MultiEmbedding
            in_hidden_size=np_hidden_size_input_to_bimpnn,  # Output of graph_encoder
            **node_predictor_config
        ).to(self.device)

        # Edge Prediction Model
        self.edge_diffusion = edge_diffusion
        ep_x_n_actual_emb_size = edge_pred_graph_encoder_config['x_n_emb_size']
        if len(num_x_n_cat_tensor) > 1 and num_x_n_cat_tensor.ndim > 0:
            ep_x_n_actual_emb_size *= len(num_x_n_cat_tensor)

        ep_pe_emb_size = edge_pred_graph_encoder_config.get('pe_emb_size', 0)
        ep_y_emb_size = edge_pred_graph_encoder_config.get('y_emb_size', 0) if self.is_conditional else 0
        ep_hidden_size_input_to_bimpnn = ep_x_n_actual_emb_size + ep_pe_emb_size + ep_y_emb_size

        edge_pred_graph_encoder = BiMPNNEncoder(
            num_x_n_cat_tensor,
            hidden_size=ep_hidden_size_input_to_bimpnn,
            **edge_pred_graph_encoder_config
        ).to(self.device)

        self.edge_pred_model = EdgePredModel(
            edge_pred_graph_encoder,
            in_hidden_size=ep_hidden_size_input_to_bimpnn,  # Output of graph_encoder
            **edge_predictor_config
        ).to(self.device)

        self.max_level = max_level

    # ... (posterior, posterior_edge, sample_node_layer, sample_edge_layer, get_batch_A, get_batch_A_n2g, get_batch_y, sample methods remain the same) ...

    # Ensure other methods like sample also use self.is_conditional if they handle 'y'
    @torch.no_grad()
    def sample(self,
               # device, # device is now self.device
               batch_size=1,
               y=None,  # This is raw_y_batch
               min_num_steps_n=None,
               max_num_steps_n=None,
               min_num_steps_e=None,
               max_num_steps_e=None):

        current_device = self.device  # Use stored device

        if self.is_conditional and y is not None:
            if batch_size != len(y):  # Check consistency if conditional and y is provided
                raise ValueError(f"Batch size ({batch_size}) must match length of y ({len(y)}) in conditional mode.")
        elif not self.is_conditional:
            y = None  # Ensure y is None if not conditional, regardless of input

        y_list = y  # y_list will be None if not self.is_conditional or y was None

        edge_index_list = [
            torch.LongTensor([[], []]).to(current_device)  # Use current_device
            for _ in range(batch_size)
        ]

        # Handle dummy_x_n based on whether it's a scalar or tensor
        if isinstance(self.dummy_x_n, (int, float)) or (
                isinstance(self.dummy_x_n, torch.Tensor) and self.dummy_x_n.ndim == 0):
            init_x_n_val = torch.LongTensor([[int(self.dummy_x_n)]]).to(current_device)
        elif isinstance(self.dummy_x_n,
                        torch.Tensor) and self.dummy_x_n.ndim > 0:  # If dummy_x_n is a tensor of dummies for multi-feature
            init_x_n_val = self.dummy_x_n.to(current_device).unsqueeze(0)
        else:
            raise TypeError(f"Unsupported type for self.dummy_x_n: {type(self.dummy_x_n)}")

        x_n_list = [init_x_n_val for _ in range(batch_size)]
        batch_x_n = torch.cat(x_n_list)

        # Pass self.is_conditional to get_batch_y or handle y directly
        batch_y_tensor = None
        if self.is_conditional and y_list is not None:
            batch_y_tensor = self.get_batch_y(y_list, x_n_list, current_device)

        level = 0.
        abs_level_list = [
            torch.tensor([[level]], device=current_device)  # Use current_device
            for _ in range(batch_size)
        ]
        batch_abs_level = torch.cat(abs_level_list)
        batch_rel_level = batch_abs_level.max() - batch_abs_level  # This is fine

        edge_index_finished = []
        x_n_finished = []
        y_finished = []  # Initialize y_finished, will only be populated if conditional

        num_nodes_cumsum = torch.cumsum(torch.tensor(
            [0] + [len(x_n_i) for x_n_i in x_n_list]), dim=0).to(current_device)  # Move to device

        # --- Main generation loop ---
        while True:
            # Ensure all tensors for model input are on self.device
            batch_A = self.get_batch_A(num_nodes_cumsum, edge_index_list, current_device)
            batch_A_n2g = self.get_batch_A_n2g(num_nodes_cumsum, current_device)

            # Current batch_x_n, batch_abs_level, batch_rel_level are already constructed
            # batch_y_tensor is used for conditioning if applicable

            x_n_l_list = self.sample_node_layer(
                batch_A, batch_x_n.to(current_device),
                batch_abs_level.to(current_device), batch_rel_level.to(current_device),
                batch_A_n2g, curr_level=level,
                y=batch_y_tensor,  # Pass the tensor batch_y_tensor
                min_num_steps_n=min_num_steps_n,
                max_num_steps_n=max_num_steps_n)

            # --- Processing after sampling a node layer ---
            edge_index_list_next_iter = []
            x_n_list_next_iter = []
            abs_level_list_next_iter = []

            query_src_list_for_edges = []
            query_dst_list_for_edges = []
            num_new_nodes_for_edges_list = []
            batch_query_src_for_edges = []
            batch_query_dst_for_edges = []

            y_list_next_iter = [] if self.is_conditional and y_list is not None else None

            active_indices_in_batch = []  # Keep track of which original batch items are still generating

            current_total_node_offset = 0  # For constructing batch_query_src/dst

            for i_original_batch in range(len(x_n_l_list)):  # Iterate based on original batch size before filtering
                # Check if this item from original batch is still active
                # This logic needs to be based on which items were passed to sample_node_layer
                # The loop for sample_node_layer processes all items currently in x_n_list
                # So, i_original_batch here corresponds to the index in the *current* active batch

                x_n_l_i = x_n_l_list[i_original_batch]  # Nodes for the new layer for graph i

                if len(x_n_l_i) == 0:  # Termination condition for this graph
                    # Move graph to finished list
                    # Node indices are 1-based in original DAGs, 0 is dummy.
                    # sample_node_layer returns attributes for new nodes.
                    # x_n_list[i_original_batch] contains all nodes so far including dummy
                    edge_index_finished.append(edge_index_list[i_original_batch] - 1)  # Adjust by dummy
                    x_n_finished.append(x_n_list[i_original_batch][1:])  # Remove dummy node
                    if self.is_conditional and y_list is not None:
                        y_finished.append(y_list[i_original_batch])
                else:
                    # Graph continues generation
                    active_indices_in_batch.append(i_original_batch)  # Log that this original index is still active

                    edge_index_list_next_iter.append(edge_index_list[i_original_batch])

                    new_full_x_n_i = torch.cat([x_n_list[i_original_batch], x_n_l_i.to(current_device)])
                    x_n_list_next_iter.append(new_full_x_n_i)

                    if self.is_conditional and y_list is not None:
                        y_list_next_iter.append(y_list[i_original_batch])

                    new_abs_level_i = torch.cat([
                        abs_level_list[i_original_batch],
                        torch.zeros(len(x_n_l_i), 1, device=current_device).fill_(level + 1)
                        # level will be incremented later
                    ])
                    abs_level_list_next_iter.append(new_abs_level_i)

                    # Prepare queries for edge prediction for this graph
                    N_old_i = len(x_n_list[i_original_batch])  # Nodes before adding new layer (incl dummy)
                    N_new_layer_i = len(x_n_l_i)  # Nodes in the new layer

                    current_query_src_i = []
                    current_query_dst_i = []

                    # Nodes from previous layers (excluding dummy) can be sources
                    # Dummy node is at index 0. Real nodes start from index 1.
                    src_candidates_i = list(range(1, N_old_i))

                    # New nodes are destinations
                    for dst_node_local_idx in range(N_new_layer_i):
                        # Global index for the new destination node
                        dst_node_global_idx = N_old_i + dst_node_local_idx
                        current_query_src_i.extend(src_candidates_i)
                        current_query_dst_i.extend([dst_node_global_idx] * len(src_candidates_i))

                    if current_query_src_i:  # Only if there are potential source nodes and new nodes
                        query_src_list_for_edges.append(torch.LongTensor(current_query_src_i).to(current_device))
                        query_dst_list_for_edges.append(torch.LongTensor(current_query_dst_i).to(current_device))

                        # For batch_query_src/dst, indices need to be offset by current_total_node_offset
                        batch_query_src_for_edges.append(
                            torch.LongTensor(current_query_src_i).to(current_device) + current_total_node_offset)
                        batch_query_dst_for_edges.append(
                            torch.LongTensor(current_query_dst_i).to(current_device) + current_total_node_offset)
                    else:  # Handle cases with no valid queries (e.g. first layer has no prior nodes to connect from, or no new nodes)
                        query_src_list_for_edges.append(torch.LongTensor([]).to(current_device))
                        query_dst_list_for_edges.append(torch.LongTensor([]).to(current_device))
                        # batch_query_src/dst will also be empty for this graph, handled by cat later

                    num_new_nodes_for_edges_list.append(N_new_layer_i)
                    current_total_node_offset += N_old_i + N_new_layer_i

            # Update lists for the next iteration with only active graphs
            edge_index_list = edge_index_list_next_iter
            x_n_list = x_n_list_next_iter
            abs_level_list = abs_level_list_next_iter
            if self.is_conditional:
                y_list = y_list_next_iter  # This will be None if not conditional from the start

            if not edge_index_list:  # All graphs have finished
                break

            level += 1  # Increment level for the new layer being processed

            # Re-calculate cumsum, batch_x_n, etc. for the *active* graphs
            num_nodes_cumsum = torch.cumsum(torch.tensor(
                [0] + [len(x_n_i) for x_n_i in x_n_list]), dim=0).to(current_device)
            batch_x_n = torch.cat(x_n_list)
            batch_abs_level = torch.cat(abs_level_list)
            batch_rel_level = batch_abs_level.max() - batch_abs_level  # Max is over all nodes in the current active batch

            batch_y_tensor = None  # Reset for current active batch
            if self.is_conditional and y_list is not None:
                batch_y_tensor = self.get_batch_y(y_list, x_n_list, current_device)

            if level == 1 and not query_src_list_for_edges:  # First layer nodes are added, but no edges to predict yet (no prior nodes)
                # OR if all query_src_list_for_edges were empty
                # This condition might need refinement: if all query lists are empty, skip edge prediction
                all_queries_empty = all(len(qs) == 0 for qs in query_src_list_for_edges)
                if all_queries_empty:
                    continue

            # Concatenate query lists only if they are not all empty
            if batch_query_src_for_edges and batch_query_dst_for_edges and \
                    any(len(qs) > 0 for qs in query_src_list_for_edges):  # Check if there's anything to concatenate

                batch_query_src_tensor = torch.cat(batch_query_src_for_edges)
                batch_query_dst_tensor = torch.cat(batch_query_dst_for_edges)

                edge_index_list = self.sample_edge_layer(
                    num_nodes_cumsum, edge_index_list, batch_x_n, batch_abs_level,
                    batch_rel_level, num_new_nodes_for_edges_list,
                    batch_query_src_tensor, batch_query_dst_tensor,
                    query_src_list_for_edges, query_dst_list_for_edges,
                    batch_y_tensor,  # Pass the tensor
                    curr_level=level,
                    min_num_steps_e=min_num_steps_e,
                    max_num_steps_e=max_num_steps_e)
            # else:
            # print(f"Level {level}: Skipping edge prediction as there are no queries.")

            if self.max_level is not None and level >= self.max_level:  # Use >= to stop after processing max_level
                # If max_level is reached, move remaining active graphs to finished
                for i in range(len(edge_index_list)):
                    edge_index_finished.append(edge_index_list[i] - 1)
                    x_n_finished.append(x_n_list[i][1:])
                    if self.is_conditional and y_list is not None:
                        y_finished.append(y_list[i])
                break  # Exit main generation loop

        # After loop, if any graphs were terminated by max_level, they are already added.
        # No need for the extra loop here if max_level logic is inside the while.

        if self.is_conditional:  # Only return y_finished if model is conditional
            return edge_index_finished, x_n_finished, y_finished
        else:
            return edge_index_finished, x_n_finished
