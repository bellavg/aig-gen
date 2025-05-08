import torch
from torch import nn
from .st_net import ST_Dis  # Assuming ST_Dis is in .st_net
from .df_utils import one_hot_add, one_hot_minus  # Assuming these are in .df_utils
from .rgcn import RGCN  # Assuming RGCN is in .rgcn


class DisGraphAF(nn.Module):
    def __init__(self, mask_node, mask_edge, index_select_edge, num_flow_layer=12, graph_size=64,
                 num_node_type=4, num_edge_type=3, use_bn=True, num_rgcn_layer=3, nhid=128, nout=128):
        '''
        Discrete Graph Autoregressive Flow model.
        Args:
            mask_node (Tensor): Mask for node generation steps.
            mask_edge (Tensor): Mask for edge generation steps.
            index_select_edge (Tensor): Indices for selecting edges during generation.
            num_flow_layer (int): Number of flow layers.
            graph_size (int): Maximum number of nodes in a graph.
            num_node_type (int): Number of distinct node types.
            num_edge_type (int): Number of distinct edge types (including no-edge if applicable).
            use_bn (bool): Whether to use batch normalization in RGCN.
            num_rgcn_layer (int): Number of layers in the RGCN.
            nhid (int): Hidden dimension size for RGCN and ST_Dis.
            nout (int): Output dimension size for RGCN (embedding size).
        '''
        super(DisGraphAF, self).__init__()
        self.repeat_num = mask_node.size(0)
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

        # Relational Graph Convolutional Network
        # edge_dim for RGCN is typically num_explicit_edge_types (e.g., REG, INV), so num_edge_type - 1 if one channel is for NO_EDGE.
        self.rgcn = RGCN(num_node_type, nhid=nhid, nout=nout, edge_dim=self.num_edge_type - 1,
                         num_layers=num_rgcn_layer, dropout=0., normalization=False)

        if use_bn:
            self.batchNorm = nn.BatchNorm1d(nout)

        # ST (Scaling and Translation) networks for discrete flow transformations
        self.node_st_net = nn.ModuleList(
            [ST_Dis(nout, self.num_node_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])
        self.edge_st_net = nn.ModuleList(
            [ST_Dis(nout * 3, self.num_edge_type, hid_dim=nhid, bias=True) for _ in range(num_flow_layer)])

    def forward(self, x, adj, x_deq, adj_deq):
        '''
        Forward pass for training. Transforms input one-hot vectors.
        Args:
            x (Tensor): Input node features (batch, N, num_node_type), typically one-hot.
            adj (Tensor): Input adjacency features (batch, num_edge_type, N, N), typically one-hot.
            x_deq (Tensor): Dequantized/continuous node features (batch, N, num_node_type).
                            In GraphDF, this might be the same as x if no dequantization is used before flow.
            adj_deq (Tensor): Dequantized/continuous edge features (batch, num_modeled_edges, num_edge_type).
                              In GraphDF, this might be the same as adj (after selection) if no dequantization.
        Returns:
            list: [transformed_x_deq, transformed_adj_deq]
        '''
        # Get embeddings based on the current graph state (x, adj)
        graph_emb_node, graph_node_emb_edge = self._get_embs(x, adj)

        # Apply flow layers
        for i in range(self.num_flow_layer):
            # Update node features
            # node_t are the parameters for the discrete transformation (e.g., shifts in logit space)
            node_t = self.node_st_net[i](graph_emb_node).type(x_deq.dtype)  # Ensure type consistency
            x_deq = one_hot_add(x_deq, node_t)  # Apply discrete transformation

            # Update edge features
            edge_t = self.edge_st_net[i](graph_node_emb_edge).type(adj_deq.dtype)  # Ensure type consistency
            adj_deq = one_hot_add(adj_deq, edge_t)  # Apply discrete transformation

        return [x_deq, adj_deq]

    def forward_rl_node(self, x, adj, x_cont):
        """
        Forward pass for reinforcement learning (node generation step).
        Args:
            x (Tensor): Current node features (batch, N, num_node_type).
            adj (Tensor): Current adjacency features (batch, num_edge_type, N, N).
            x_cont (Tensor): Node features to be transformed (batch, num_node_type).
        Returns:
            Tensor: Transformed node features.
            None: Placeholder for log_jacob (not used in discrete flows).
        """
        embs = self._get_embs_node(x, adj)  # (batch, d)
        for i in range(self.num_flow_layer):
            node_t = self.node_st_net[i](embs)
            x_cont = one_hot_add(x_cont, node_t)
        return x_cont, None

    def forward_rl_edge(self, x, adj, x_cont, index):
        """
        Forward pass for reinforcement learning (edge generation step).
        Args:
            x (Tensor): Current node features (batch, N, num_node_type).
            adj (Tensor): Current adjacency features (batch, num_edge_type, N, N).
            x_cont (Tensor): Edge features to be transformed (batch, num_edge_type).
            index (Tensor): Index of the edge being generated (batch, 2).
        Returns:
            Tensor: Transformed edge features.
            None: Placeholder for log_jacob (not used in discrete flows).
        """
        embs = self._get_embs_edge(x, adj, index)  # (batch, 3d)
        for i in range(self.num_flow_layer):
            edge_t = self.edge_st_net[i](embs)
            x_cont = one_hot_add(x_cont, edge_t)
        return x_cont, None

    # ***** CORRECTED METHOD SIGNATURE *****
    def reverse(self, x_cond_onehot, adj_cond_onehot, z_onehot, mode, edge_index=None):
        '''
        Reverse pass for generation (sampling).
        Args:
            x_cond_onehot (Tensor): Generated subgraph node features (one-hot) so far.
                                    Shape (1, N, num_node_type). Some parts are masked/zero.
            adj_cond_onehot (Tensor): Generated subgraph adjacency features (one-hot) so far.
                                      Shape (1, num_edge_type, N, N). Some parts are masked/zero.
            z_onehot (Tensor): Sampled one-hot latent vector from the base distribution.
                               Shape (1, num_node_type) for node mode, or (1, num_edge_type) for edge mode.
            mode (int): Generation mode. 0 for node generation, 1 for edge generation.
            edge_index (Tensor, optional): Index of the edge being generated if mode is 1. Shape (1, 2).
        Returns:
            Tensor: Output logits for the new node/edge type.
                    Shape (1, num_node_type) for node mode, or (1, num_edge_type) for edge mode.
        '''
        # Assertions to ensure correct input shapes and conditions
        assert mode == 0 or edge_index is not None, 'If mode is 1 (edge generation), edge_index must be provided.'
        assert x_cond_onehot.size(0) == 1, "Batch size for reverse pass must be 1."
        assert adj_cond_onehot.size(0) == 1, "Batch size for reverse pass must be 1."
        if edge_index is not None:
            assert edge_index.size(0) == 1 and edge_index.size(1) == 2, "edge_index shape must be (1, 2)."

        # Determine the appropriate ST network and conditioning embeddings based on mode
        if mode == 0:  # Node generation
            st_net_list = self.node_st_net
            # Use x_cond_onehot and adj_cond_onehot for conditioning
            conditioning_embedding = self._get_embs_node(x_cond_onehot, adj_cond_onehot)
        else:  # Edge generation
            st_net_list = self.edge_st_net
            # Use x_cond_onehot, adj_cond_onehot, and edge_index for conditioning
            conditioning_embedding = self._get_embs_edge(x_cond_onehot, adj_cond_onehot, edge_index)

        # Apply flow layers in reverse
        transformed_z = z_onehot  # Start with the latent sample
        for i in reversed(range(self.num_flow_layer)):
            # Get transformation parameters from the ST network
            transformation_params = st_net_list[i](conditioning_embedding)
            # Apply the inverse discrete transformation
            transformed_z = one_hot_minus(transformed_z, transformation_params)

        return transformed_z  # These are now the output logits

    def _get_embs_node(self, x, adj):
        """
        Helper function to get graph embeddings for node feature updates.
        Args:
            x (Tensor): Current node features (batch, N, num_node_type).
            adj (Tensor): Current adjacency features (batch, num_edge_type, N, N).
        Returns:
            Tensor: Graph embedding for node updates (batch, nout).
        """
        # Consider only explicit edge types for RGCN
        num_explicit_edges = self.num_edge_type - 1  # Assuming last channel is NO_EDGE or similar
        adj_for_rgcn = adj[:, :num_explicit_edges]

        node_emb = self.rgcn(x, adj_for_rgcn)  # (batch, N, nout)
        if hasattr(self, 'batchNorm'):
            # BatchNorm1d expects (batch, channels, length)
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)

            # Sum pooling to get graph-level embedding
        graph_emb = torch.sum(node_emb, dim=1, keepdim=False).contiguous()  # (batch, nout)
        return graph_emb

    def _get_embs_edge(self, x, adj, index):
        """
        Helper function to get embeddings for edge feature updates.
        Concatenates graph embedding, start node embedding, and end node embedding.
        Args:
            x (Tensor): Current node features (batch, N, num_node_type).
            adj (Tensor): Current adjacency features (batch, num_edge_type, N, N).
            index (Tensor): Edge indices (u, v) for which to generate embeddings (batch, 2).
        Returns:
            Tensor: Concatenated embeddings for edge updates (batch, 3 * nout).
        """
        batch_size = x.size(0)
        assert batch_size == index.size(0)

        num_explicit_edges = self.num_edge_type - 1
        adj_for_rgcn = adj[:, :num_explicit_edges]

        node_emb = self.rgcn(x, adj_for_rgcn)  # (batch, N, nout)
        if hasattr(self, 'batchNorm'):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)

        # Graph-level embedding
        graph_emb_sum = torch.sum(node_emb, dim=1, keepdim=False).contiguous().view(batch_size, 1,
                                                                                    -1)  # (batch, 1, nout)

        # Gather embeddings for start and end nodes of the specified edges
        # index shape: (batch, 2), needs to be (batch, 2, nout) for gather
        expanded_index = index.view(batch_size, 2, 1).repeat(1, 1, self.emb_size)  # (batch, 2, nout)
        edge_node_embs = torch.gather(node_emb, dim=1, index=expanded_index)  # (batch, 2, nout)

        # Concatenate: [start_node_emb, end_node_emb, graph_emb]
        # Reshape edge_node_embs from (batch, 2, nout) to (batch, 2*nout)
        # Or, more robustly, concatenate along the feature dimension
        start_node_emb = edge_node_embs[:, 0, :]  # (batch, nout)
        end_node_emb = edge_node_embs[:, 1, :]  # (batch, nout)

        # Concatenate along the last dimension
        # Make sure graph_emb_sum is (batch, nout)
        concatenated_embs = torch.cat((start_node_emb, end_node_emb, graph_emb_sum.squeeze(1)),
                                      dim=1)  # (batch, 3 * nout)

        return concatenated_embs

    def _get_embs(self, x, adj):
        '''
        Helper function to get embeddings for all node and edge generation steps,
        considering the autoregressive masks.
        Args:
            x (Tensor): Initial node features (batch, N, num_node_type).
            adj (Tensor): Initial adjacency features (batch, num_edge_type, N, N).
        Returns:
            Tensor: Embeddings for node generation steps (batch, N, nout).
            Tensor: Embeddings for edge generation steps (batch, num_edge_steps, 3 * nout).
        '''
        batch_size = x.size(0)
        num_explicit_edges = self.num_edge_type - 1

        # Apply masks to get features for each step in the autoregressive process
        # x_masked: (batch * repeat_num, N, num_node_type)
        # adj_masked: (batch * repeat_num, num_explicit_edges, N, N)
        x_masked = torch.where(self.mask_node, x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1),
                               torch.zeros_like(x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1))).view(
            batch_size * self.repeat_num, self.graph_size, self.num_node_type)

        adj_masked_for_rgcn = adj[:, :num_explicit_edges]  # Select explicit edge types
        adj_masked = torch.where(self.mask_edge, adj_masked_for_rgcn.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1),
                                 torch.zeros_like(
                                     adj_masked_for_rgcn.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1))).view(
            batch_size * self.repeat_num, num_explicit_edges, self.graph_size, self.graph_size)

        # Get node embeddings from RGCN for all masked steps
        node_emb_all_steps = self.rgcn(x_masked, adj_masked)  # (batch * repeat_num, N, nout)

        if hasattr(self, 'batchNorm'):
            node_emb_all_steps = self.batchNorm(node_emb_all_steps.transpose(1, 2)).transpose(1, 2)

        # Reshape to separate batch and repeat_num dimensions
        node_emb_all_steps = node_emb_all_steps.view(batch_size, self.repeat_num, self.graph_size,
                                                     -1)  # (batch, repeat_num, N, nout)

        # Sum node embeddings to get graph-level embeddings for each step
        graph_emb_all_steps = torch.sum(node_emb_all_steps, dim=2, keepdim=False)  # (batch, repeat_num, nout)

        # --- Embeddings for Node Generation ST-Nets ---
        # Select embeddings corresponding to node generation steps (first self.graph_size steps)
        graph_emb_for_nodes = graph_emb_all_steps[:, :self.graph_size, :].contiguous()  # (batch, N, nout)
        # No need to reshape to (batch*N, d) here, ST_Dis likely handles batch processing.

        # --- Embeddings for Edge Generation ST-Nets ---
        # Select graph-level embeddings for edge generation steps
        graph_emb_for_edges_sum = graph_emb_all_steps[:, self.graph_size:,
                                  :].contiguous()  # (batch, num_edge_steps, nout)
        graph_emb_for_edges_sum = graph_emb_for_edges_sum.unsqueeze(2)  # (batch, num_edge_steps, 1, nout)

        # Select node-level embeddings for edge generation steps
        node_embs_for_edges = node_emb_all_steps[:, self.graph_size:, :, :]  # (batch, num_edge_steps, N, nout)

        # Use self.index_select_edge to gather start/end node embeddings for each edge step
        # self.index_select_edge: (num_edge_steps, 2)
        # Expand index_select_edge for batching and embedding dimension
        num_edge_steps = self.index_select_edge.size(0)  # Should be self.repeat_num - self.graph_size

        # Prepare indices for torch.gather
        # index_select_edge_expanded: (1, num_edge_steps, 2, 1) -> (batch_size, num_edge_steps, 2, nout)
        index_select_edge_expanded = self.index_select_edge.view(1, num_edge_steps, 2, 1).repeat(batch_size, 1, 1,
                                                                                                 self.emb_size)

        # Gather start and end node embeddings
        # node_embs_for_edges: (batch, num_edge_steps, N, nout)
        # index_select_edge_expanded: (batch, num_edge_steps, 2, nout)
        # gathered_node_embs: (batch, num_edge_steps, 2, nout)
        gathered_node_embs = torch.gather(node_embs_for_edges, dim=2, index=index_select_edge_expanded)

        # Concatenate [start_node_emb, end_node_emb, graph_emb] for each edge step
        # gathered_node_embs[:,:,0,:] is start_node_emb (batch, num_edge_steps, nout)
        # gathered_node_embs[:,:,1,:] is end_node_emb   (batch, num_edge_steps, nout)
        # graph_emb_for_edges_sum.squeeze(2) is graph_emb (batch, num_edge_steps, nout)

        start_nodes = gathered_node_embs[:, :, 0, :]
        end_nodes = gathered_node_embs[:, :, 1, :]
        graph_sum = graph_emb_for_edges_sum.squeeze(2)

        conditioning_embs_for_edges = torch.cat((start_nodes, end_nodes, graph_sum),
                                                dim=2)  # (batch, num_edge_steps, 3 * nout)

        return graph_emb_for_nodes, conditioning_embs_for_edges
