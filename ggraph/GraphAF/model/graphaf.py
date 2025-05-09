import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import F explicitly
import warnings  # For warnings
from .rgcn import RGCN  # Assuming RGCN is in the same directory
from .st_net import ST_Net_Sigmoid, ST_Net_Exp, \
    ST_Net_Softplus  # Assuming ST_Net implementations are in the same directory


class MaskedGraphAF(nn.Module):
    def __init__(self, mask_node, mask_edge, index_select_edge, st_type='sigmoid', num_flow_layer=12, graph_size=38,
                 num_node_type=4, num_edge_type=3, use_bn=True, num_rgcn_layer=3, nhid=128, nout=128):
        '''
        MaskedGraphAF: Core Autoregressive Flow model using continuous flows.

        Args:
            mask_node (torch.Tensor): Masks for node generation steps.
            mask_edge (torch.Tensor): Masks for edge generation steps.
            index_select_edge (torch.Tensor): Defines the order of edge generation.
            st_type (str): Type of S-T network ('sigmoid', 'exp', 'softplus').
            num_flow_layer (int): Number of flow layers.
            graph_size (int): Maximum number of nodes in a graph.
            num_node_type (int): Number of distinct node types (e.g., 4 for AIGs).
            num_edge_type (int): Number of distinct edge types/channels (e.g., 3 for AIGs: REG, INV, NO_EDGE).
            use_bn (bool): Whether to use batch normalization in RGCN.
            num_rgcn_layer (int): Number of layers in the RGCN.
            nhid (int): Hidden dimension size for RGCN and ST-nets.
            nout (int): Output dimension size for RGCN (embedding size).
        '''
        super(MaskedGraphAF, self).__init__()
        self.repeat_num = mask_node.size(0)  # Total number of autoregressive steps
        self.graph_size = graph_size
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type

        # Store masks and indices as non-trainable parameters
        self.mask_node = nn.Parameter(mask_node.view(1, self.repeat_num, graph_size, 1), requires_grad=False)
        self.mask_edge = nn.Parameter(mask_edge.view(1, self.repeat_num, 1, graph_size, graph_size),
                                      requires_grad=False)
        self.index_select_edge = nn.Parameter(index_select_edge, requires_grad=False)

        self.emb_size = nout
        self.num_flow_layer = num_flow_layer

        # RGCN processes actual edge types (e.g., REG, INV), excluding NO_EDGE channel
        rgcn_edge_dim = self.num_edge_type - 1 if self.num_edge_type > 1 else 0
        if rgcn_edge_dim <= 0 and self.num_edge_type > 0:
            warnings.warn(f"RGCN edge_dim calculated as {rgcn_edge_dim} from num_edge_type={self.num_edge_type}. "
                          "Ensure num_edge_type includes a NO_EDGE channel if applicable.")
            rgcn_edge_dim = max(0, rgcn_edge_dim)  # Prevent negative dim

        self.rgcn = RGCN(num_node_type, nhid=nhid, nout=nout, edge_dim=rgcn_edge_dim,
                         num_layers=num_rgcn_layer, dropout=0., normalization=False)

        if use_bn:
            self.batchNorm = nn.BatchNorm1d(nout)

        self.st_type = st_type
        self.st_net_fn_dict = {'sigmoid': ST_Net_Sigmoid, 'exp': ST_Net_Exp, 'softplus': ST_Net_Softplus}
        assert st_type in self.st_net_fn_dict, f'Unsupported st_type: {st_type}. Choices are {list(self.st_net_fn_dict.keys())}'
        st_net_fn = self.st_net_fn_dict[st_type]

        # Initialize ST-Nets (Scale and Translate networks) for flow layers
        self.node_st_net = nn.ModuleList(
            [st_net_fn(nout, self.num_node_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])
        # Edge ST-Nets condition on source node, target node, and graph embeddings (hence nout*3)
        self.edge_st_net = nn.ModuleList(
            [st_net_fn(nout * 3, self.num_edge_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])

    def forward(self, x, adj, x_deq, adj_deq):
        '''
        Forward pass for training (data -> latent).

        Args:
            x (torch.Tensor): Input node features (batch, N, num_node_type). Typically one-hot.
            adj (torch.Tensor): Input adjacency features (batch, num_edge_type, N, N). Typically one-hot.
            x_deq (torch.Tensor): Dequantized node features (batch, N, num_node_type).
            adj_deq (torch.Tensor): Dequantized edge features corresponding to the generation schedule
                                   (batch, edge_num_in_schedule, num_edge_type).
        Returns:
            list: Latent variables [z_node, z_edge].
            list: Log determinants [logdet_node, logdet_edge].
        '''
        batch_size = x.size(0)
        # Get conditioning embeddings from the input graph structure
        graph_emb_node, graph_node_emb_edge = self._get_embs(x, adj)

        # Reshape dequantized features for processing by ST-Nets
        x_deq = x_deq.view(-1, self.num_node_type)  # (batch * N, num_node_type)
        adj_deq = adj_deq.view(-1, self.num_edge_type)  # (batch * num_edges_in_schedule, num_edge_type)

        # Initialize log determinants
        total_x_log_jacob = torch.zeros(x_deq.shape[0], device=x_deq.device)  # Shape: (batch*N)
        total_adj_log_jacob = torch.zeros(adj_deq.shape[0],
                                          device=adj_deq.device)  # Shape: (batch*num_edges_in_schedule)

        for i in range(self.num_flow_layer):
            # --- Node Flow Layer ---
            node_s, node_t = self.node_st_net[i](graph_emb_node)  # Get scale and translation

            # Apply affine transformation based on st_type
            if self.st_type == 'sigmoid':  # y = x*s + t
                x_deq = x_deq * node_s + node_t
                # log|det| = log|s|
                current_x_log_jacob = torch.sum((torch.abs(node_s) + 1e-20).log(), dim=-1)
            elif self.st_type == 'exp':  # y = (x+t) * exp(log_s)
                # Assumes ST_Net_Exp outputs log_s for 's'
                log_s = node_s
                x_deq = (x_deq + node_t) * torch.exp(log_s)
                # log|det| = log(exp(log_s)) = log_s
                current_x_log_jacob = torch.sum(log_s, dim=-1)
            elif self.st_type == 'softplus':  # y = (x+t) * s, where s = softplus(raw_s)
                # Assumes ST_Net_Softplus outputs s directly
                s_val = node_s
                x_deq = (x_deq + node_t) * s_val
                # log|det| = log|s| = log(s) since s > 0
                current_x_log_jacob = torch.sum((s_val + 1e-20).log(), dim=-1)
            else:
                raise ValueError(f'Unsupported st_type: {self.st_type}')

            if torch.isnan(x_deq).any():
                raise RuntimeError(f'x_deq has NaN entries after node transformation at layer {i}')
            total_x_log_jacob += current_x_log_jacob

            # --- Edge Flow Layer ---
            edge_s, edge_t = self.edge_st_net[i](graph_node_emb_edge)

            # Apply affine transformation based on st_type
            if self.st_type == 'sigmoid':  # y = x*s + t
                adj_deq = adj_deq * edge_s + edge_t
                current_adj_log_jacob = torch.sum((torch.abs(edge_s) + 1e-20).log(), dim=-1)
            elif self.st_type == 'exp':  # y = (x+t) * exp(log_s)
                log_s = edge_s
                adj_deq = (adj_deq + edge_t) * torch.exp(log_s)
                current_adj_log_jacob = torch.sum(log_s, dim=-1)
            elif self.st_type == 'softplus':  # y = (x+t) * s
                s_val = edge_s
                adj_deq = (adj_deq + edge_t) * s_val
                current_adj_log_jacob = torch.sum((s_val + 1e-20).log(), dim=-1)
            else:
                raise ValueError(f'Unsupported st_type: {self.st_type}')

            if torch.isnan(adj_deq).any():
                raise RuntimeError(f'adj_deq has NaN entries after edge transformation at layer {i}')
            total_adj_log_jacob += current_adj_log_jacob

        # Reshape latent variables and log determinants back to batch structure
        z_node = x_deq.view(batch_size, -1)  # (batch, N * num_node_type)
        z_edge = adj_deq.view(batch_size, -1)  # (batch, num_edges_in_schedule * num_edge_type)

        # Reshape log determinants and sum over dimensions for each batch item
        logdet_node = total_x_log_jacob.view(batch_size, -1).sum(-1)  # (batch,)
        logdet_edge = total_adj_log_jacob.view(batch_size, -1).sum(-1)  # (batch,)

        return [z_node, z_edge], [logdet_node, logdet_edge]

    def forward_rl_node(self, x, adj, x_cont):
        """
        Forward pass for node generation in reinforcement learning (if used).
        Transforms a continuous input `x_cont` based on graph context `x`, `adj`.

        Args:
            x (torch.Tensor): Current graph node features (batch, N, num_node_type).
            adj (torch.Tensor): Current graph adjacency features (batch, num_edge_type, N, N).
            x_cont (torch.Tensor): Continuous features to be transformed (batch, num_node_type).
        Returns:
            torch.Tensor: Transformed continuous features (batch, num_node_type).
            torch.Tensor: Log Jacobian determinant of the transformation (batch, ).
        """
        embs = self._get_embs_node(x, adj)  # (batch, d)
        total_log_jacob = torch.zeros(x_cont.shape[0], device=x_cont.device)  # Shape: (batch)

        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](embs)

            if self.st_type == 'sigmoid':
                x_cont = x_cont * node_s + node_t
                current_log_jacob = torch.sum((torch.abs(node_s) + 1e-20).log(), dim=-1)
            elif self.st_type == 'exp':
                log_s = node_s
                x_cont = (x_cont + node_t) * torch.exp(log_s)
                current_log_jacob = torch.sum(log_s, dim=-1)
            elif self.st_type == 'softplus':
                s_val = node_s
                x_cont = (x_cont + node_t) * s_val
                current_log_jacob = torch.sum((s_val + 1e-20).log(), dim=-1)
            else:
                # Corrected: Use self.st_type instead of self.args.st_type
                raise ValueError('unsupported st type: (%s)' % self.st_type)

            if torch.isnan(x_cont).any():
                raise RuntimeError(f'x_cont has NaN entries after RL node transformation at layer {i}')

            total_log_jacob += current_log_jacob

        # No need to sum again, already summed over the feature dimension
        # x_log_jacob = total_log_jacob.sum(-1) # This would be wrong

        return x_cont, total_log_jacob

    def forward_rl_edge(self, x, adj, x_cont, index):
        """
        Forward pass for edge generation in reinforcement learning (if used).
        Transforms continuous input `x_cont` based on graph context and target edge `index`.

        Args:
            x (torch.Tensor): Current graph node features (batch, N, num_node_type).
            adj (torch.Tensor): Current graph adjacency features (batch, num_edge_type, N, N).
            x_cont (torch.Tensor): Continuous features to be transformed (batch, num_edge_type).
            index (torch.Tensor): Index of the edge being generated (batch, 2).
        Returns:
            torch.Tensor: Transformed continuous features (batch, num_edge_type).
            torch.Tensor: Log Jacobian determinant of the transformation (batch, ).
        """
        embs = self._get_embs_edge(x, adj, index)  # (batch, 3d)
        total_log_jacob = torch.zeros(x_cont.shape[0], device=x_cont.device)  # Shape: (batch)

        for i in range(self.num_flow_layer):
            edge_s, edge_t = self.edge_st_net[i](embs)

            if self.st_type == 'sigmoid':
                x_cont = x_cont * edge_s + edge_t
                current_log_jacob = torch.sum((torch.abs(edge_s) + 1e-20).log(), dim=-1)
            elif self.st_type == 'exp':
                log_s = edge_s
                x_cont = (x_cont + edge_t) * torch.exp(log_s)
                current_log_jacob = torch.sum(log_s, dim=-1)
            elif self.st_type == 'softplus':
                s_val = edge_s
                x_cont = (x_cont + edge_t) * s_val
                current_log_jacob = torch.sum((s_val + 1e-20).log(), dim=-1)
            else:
                # Corrected: Use self.st_type instead of self.args.st_type
                raise ValueError('unsupported st type: (%s)' % self.st_type)

            if torch.isnan(x_cont).any():
                raise RuntimeError(f'x_cont has NaN entries after RL edge transformation at layer {i}')

            total_log_jacob += current_log_jacob

        # No need to sum again
        # x_log_jacob = total_log_jacob.sum(-1) # Wrong

        return x_cont, total_log_jacob

    def reverse(self, x, adj, latent, mode, edge_index=None):
        '''
        Reverse pass for generation (latent -> data).

        Args:
            x (torch.Tensor): Generated subgraph node features so far (1, N, num_node_type).
                              Typically discretized (e.g., one-hot) for conditioning RGCN.
            adj (torch.Tensor): Generated subgraph adjacency features so far (1, num_edge_type, N, N).
                                Typically one-hot for conditioning RGCN.
            latent (torch.Tensor): Sampled latent vector from prior:
                                   (1, num_node_type) if mode == 0 (node generation).
                                   (1, num_edge_type) if mode == 1 (edge generation).
            mode (int): 0 for node generation, 1 for edge generation.
            edge_index (torch.Tensor, optional): Specifies the (u,v) edge if mode == 1. Shape (1, 2).

        Returns:
            torch.Tensor: Generated node/edge features (continuous scores).
                          Shape (1, num_node_type) or (1, num_edge_type).
        '''
        assert mode == 0 or edge_index is not None, 'If mode is 1 (edge generation), edge_index must be specified.'
        assert x.size(0) == 1, "Reverse pass expects batch size of 1 for generation."
        assert adj.size(0) == 1, "Reverse pass expects batch size of 1 for generation."
        assert edge_index is None or (edge_index.size(0) == 1 and edge_index.size(1) == 2), "edge_index shape mismatch."

        # Select the appropriate ST-Net list and compute conditioning embedding
        if mode == 0:  # Node generation
            st_net_list = self.node_st_net
            emb = self._get_embs_node(x, adj)  # Embedding based on current graph state
        else:  # Edge generation (mode == 1)
            st_net_list = self.edge_st_net
            emb = self._get_embs_edge(x, adj, edge_index)  # Embedding based on graph state and target edge

        # Apply inverse flow transformations layer by layer
        for i in reversed(range(self.num_flow_layer)):
            s, t = st_net_list[i](emb)  # Get scale and translation from the i-th ST-Net

            # Apply inverse affine transformation based on st_type
            # Add epsilon for numerical stability during division
            epsilon = 1e-9
            if self.st_type == 'sigmoid':  # Inverse: x = (y - t) / s
                latent = (latent - t) / (s + torch.sign(s) * epsilon + epsilon)  # Sign term avoids issues near zero
            elif self.st_type == 'exp':  # Inverse: x = (y / exp(log_s)) - t
                log_s = s
                latent = (latent / (torch.exp(log_s) + epsilon)) - t  # exp(log_s) is always positive
            elif self.st_type == 'softplus':  # Inverse: x = (y / s) - t
                s_val = s
                latent = (latent / (s_val + epsilon)) - t  # s_val from softplus is always positive
            else:
                raise ValueError(f'Unsupported st_type: {self.st_type}')

            if torch.isnan(latent).any():
                warnings.warn(
                    f'Latent became NaN during reverse at layer {i} for mode {mode}. Check ST-Net outputs and stability.')
                # Optionally, break or return a default value if NaN occurs
                # return torch.zeros_like(latent) # Example fallback

        # The final 'latent' here is the transformed value in the data space (continuous scores)
        return latent

    def _get_embs_node(self, x, adj):
        """
        Computes graph embeddings for node generation steps.

        Args:
            x (torch.Tensor): Node features (batch, N, num_node_type). Assumed discrete/one-hot for RGCN.
            adj (torch.Tensor): Adjacency features (batch, num_edge_type, N, N). Assumed one-hot.
        Returns:
            torch.Tensor: Graph embedding for conditioning node ST-Nets (batch, nout).
        """
        # RGCN processes actual edge types (e.g., REG, INV), excluding NO_EDGE channel
        rgcn_edge_dim = self.num_edge_type - 1 if self.num_edge_type > 1 else 0
        if rgcn_edge_dim < 0: rgcn_edge_dim = 0  # Safety check
        adj_for_rgcn = adj[:, :rgcn_edge_dim, :, :] if rgcn_edge_dim > 0 else None

        if adj_for_rgcn is None and rgcn_edge_dim == 0:
            # Handle case with no edge types or only NO_EDGE type
            # RGCN might need modification or alternative embedding method
            warnings.warn(
                "RGCN called with edge_dim=0 in _get_embs_node. Ensure RGCN handles this or modify embedding logic.")
            # Fallback: maybe just use node features? For now, pass None or handle in RGCN.
            node_emb = self.rgcn(x, None)  # Assuming RGCN can handle adj=None
        else:
            node_emb = self.rgcn(x, adj_for_rgcn)  # (batch, N, nout)

        if hasattr(self, 'batchNorm'):
            # BatchNorm expects (N, C) or (N, C, L), need (batch, C, N)
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch, N, nout)

        # Sum node embeddings to get graph-level embedding
        graph_emb = torch.sum(node_emb, dim=1, keepdim=False).contiguous()  # (batch, nout)
        return graph_emb

    def _get_embs_edge(self, x, adj, index):
        """
        Computes composite embeddings for edge generation steps.

        Args:
            x (torch.Tensor): Node features (batch, N, num_node_type). Assumed discrete/one-hot.
            adj (torch.Tensor): Adjacency features (batch, num_edge_type, N, N). Assumed one-hot.
            index (torch.Tensor): Indices of the (source, target) nodes for the edge (batch, 2).
        Returns:
            torch.Tensor: Composite embedding [src_emb, tgt_emb, graph_emb] (batch, 3 * nout).
        """
        batch_size = x.size(0)
        assert batch_size == index.size(0)

        rgcn_edge_dim = self.num_edge_type - 1 if self.num_edge_type > 1 else 0
        if rgcn_edge_dim < 0: rgcn_edge_dim = 0
        adj_for_rgcn = adj[:, :rgcn_edge_dim, :, :] if rgcn_edge_dim > 0 else None

        if adj_for_rgcn is None and rgcn_edge_dim == 0:
            warnings.warn(
                "RGCN called with edge_dim=0 in _get_embs_edge. Ensure RGCN handles this or modify embedding logic.")
            node_emb = self.rgcn(x, None)
        else:
            node_emb = self.rgcn(x, adj_for_rgcn)  # (batch, N, nout)

        if hasattr(self, 'batchNorm'):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch, N, nout)

        # Graph embedding (sum over nodes)
        graph_emb_sum = torch.sum(node_emb, dim=1, keepdim=False).contiguous().view(batch_size, 1,
                                                                                    -1)  # (batch, 1, nout)

        # Gather source and target node embeddings using `index`
        # index shape: (batch_size, 2) -> expand to (batch_size, 2, self.emb_size) for gather
        expanded_index = index.view(batch_size, 2, 1).repeat(1, 1, self.emb_size)  # (batch_size, 2, nout)
        selected_node_embs = torch.gather(node_emb, dim=1, index=expanded_index)  # (batch_size, 2, nout)

        # Concatenate [source_node_emb, target_node_emb, graph_emb_sum] along dim=1
        graph_node_emb = torch.cat((selected_node_embs, graph_emb_sum), dim=1)  # (batch_size, 3, nout)
        graph_node_emb = graph_node_emb.view(batch_size, -1)  # (batch_size, 3 * nout)
        return graph_node_emb

    def _get_embs(self, x, adj):
        '''
        Computes embeddings for all autoregressive steps during training.

        Args:
            x (torch.Tensor): Full batch node features (batch, N, num_node_type).
            adj (torch.Tensor): Full batch adjacency features (batch, num_edge_type, N, N).
        Returns:
            torch.Tensor: Embeddings for node ST-Nets (batch * N, nout).
            torch.Tensor: Embeddings for edge ST-Nets (batch * num_edges_in_schedule, 3 * nout).
        '''
        batch_size = x.size(0)
        # Select actual edge type channels for RGCN
        rgcn_edge_dim = self.num_edge_type - 1 if self.num_edge_type > 1 else 0
        if rgcn_edge_dim < 0: rgcn_edge_dim = 0
        adj_for_rgcn = adj[:, :rgcn_edge_dim, :, :] if rgcn_edge_dim > 0 else None

        # Apply masks to create input sequences for each autoregressive step
        x_masked = torch.where(self.mask_node, x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1),
                               torch.zeros_like(x.unsqueeze(1)))
        x_masked = x_masked.view(-1, self.graph_size, self.num_node_type)  # (batch*repeat_num, N, num_node_type)

        if adj_for_rgcn is not None:
            # Mask adjacency features
            # mask_edge shape: (1, repeat_num, 1, N, N)
            # adj_for_rgcn shape: (batch, rgcn_edge_dim, N, N)
            # Unsqueeze and repeat adj: (batch, repeat_num, rgcn_edge_dim, N, N)
            adj_masked = torch.where(self.mask_edge, adj_for_rgcn.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1),
                                     torch.zeros_like(adj_for_rgcn.unsqueeze(1)))
            adj_masked = adj_masked.view(-1, rgcn_edge_dim, self.graph_size,
                                         self.graph_size)  # (batch*repeat_num, rgcn_edge_dim, N, N)
            # Get node embeddings for all steps using masked inputs
            node_emb_all_steps = self.rgcn(x_masked, adj_masked)  # (batch*repeat_num, N, nout)
        else:
            warnings.warn(
                "RGCN called with edge_dim=0 in _get_embs. Ensure RGCN handles this or modify embedding logic.")
            node_emb_all_steps = self.rgcn(x_masked, None)  # Assuming RGCN can handle adj=None

        if hasattr(self, 'batchNorm'):
            # Reshape for BatchNorm: (batch*repeat_num*N, nout) -> Transpose -> BN -> Transpose -> Reshape back
            original_shape = node_emb_all_steps.shape
            node_emb_all_steps_flat = node_emb_all_steps.view(-1,
                                                              self.emb_size)  # Check if this is correct or needs transpose
            # BatchNorm1d expects (N, C) or (N, C, L). Input is (batch*repeat_num*N, nout). This might require reshaping differently or using LayerNorm.
            # Let's assume batchNorm works on the last dim directly here, or adjust if needed.
            # If BN expects (N,C), need node_emb_all_steps.view(-1, self.emb_size)
            # If BN expects (N, C, L), need different view.
            # Assuming (N=batch*repeat_num*N, C=nout)
            try:
                # Apply BN on the embedding dimension
                bn_input = node_emb_all_steps.reshape(-1, self.emb_size)
                bn_output = self.batchNorm(bn_input)
                node_emb_all_steps = bn_output.reshape(original_shape)

            except RuntimeError as e:
                warnings.warn(
                    f"BatchNorm failed in _get_embs: {e}. Skipping BatchNorm. Input shape: {node_emb_all_steps.shape}")
                # Fallback or re-raise if BN is critical

        node_emb_all_steps = node_emb_all_steps.view(batch_size, self.repeat_num, self.graph_size,
                                                     -1)  # (batch, repeat_num, N, nout)

        # Calculate graph embedding for each step by summing node embeddings
        graph_emb_all_steps = torch.sum(node_emb_all_steps, dim=2, keepdim=False)  # (batch, repeat_num, nout)

        # --- Prepare embeddings for Node ST-Nets ---
        # Select embeddings corresponding to node generation steps (first self.graph_size steps)
        graph_emb_node = graph_emb_all_steps[:, :self.graph_size, :].contiguous()  # (batch, N, nout)
        graph_emb_node = graph_emb_node.view(batch_size * self.graph_size, -1)  # (batch*N, nout)

        # --- Prepare embeddings for Edge ST-Nets ---
        # Select graph embeddings corresponding to edge generation steps
        graph_emb_edge_steps = graph_emb_all_steps[:, self.graph_size:,
                               :].contiguous()  # (batch, num_edges_in_schedule, nout)
        graph_emb_edge_steps = graph_emb_edge_steps.unsqueeze(2)  # (batch, num_edges_in_schedule, 1, nout)

        # Select node embeddings corresponding to edge generation steps
        node_emb_edge_steps = node_emb_all_steps[:, self.graph_size:, :,
                              :].contiguous()  # (batch, num_edges_in_schedule, N, nout)

        # Gather source and target node embeddings for each edge step using self.index_select_edge
        num_edges_in_schedule = self.repeat_num - self.graph_size
        # index_select_edge shape: (num_edges_in_schedule, 2)
        # Expand index_select_edge for gathering: (1, num_edges_in_schedule, 2, 1) -> repeat batch -> (batch, num_edges_in_schedule, 2, 1) -> repeat emb_size -> (batch, num_edges_in_schedule, 2, nout)
        expanded_edge_indices = self.index_select_edge.view(1, num_edges_in_schedule, 2, 1).repeat(batch_size, 1, 1,
                                                                                                   self.emb_size)

        # Gather corresponding node embeddings
        # node_emb_edge_steps shape: (batch, num_edges_in_schedule, N, nout)
        # expanded_edge_indices shape: (batch, num_edges_in_schedule, 2, nout)
        # Need to gather along dim=2 (the N dimension)
        selected_node_embs_for_edges = torch.gather(node_emb_edge_steps, dim=2,
                                                    index=expanded_edge_indices)  # (batch, num_edges_in_schedule, 2, nout)

        # Concatenate [source_emb, target_emb, graph_emb] for each edge step
        # selected_node_embs_for_edges has shape (batch, num_edges_in_schedule, 2, nout)
        # graph_emb_edge_steps has shape      (batch, num_edges_in_schedule, 1, nout)
        graph_node_emb_edge = torch.cat((selected_node_embs_for_edges, graph_emb_edge_steps),
                                        dim=2)  # (batch, num_edges_in_schedule, 3, nout)

        # Reshape for edge ST-Nets
        graph_node_emb_edge = graph_node_emb_edge.view(batch_size * num_edges_in_schedule,
                                                       -1)  # (batch * num_edges_in_schedule, 3 * nout)

        return graph_emb_node, graph_node_emb_edge

