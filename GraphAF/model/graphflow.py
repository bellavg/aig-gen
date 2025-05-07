import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from  rdkit import Chem # Not needed for AIG generation
# from dig.ggraph.utils import check_valency, convert_radical_electrons_to_hydrogens # Not needed for AIG
from .graphaf import MaskedGraphAF  # Assuming this is in the same directory (model/)


class GraphFlowModel(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict.get('edge_unroll', 12)  # Default if not in conf
        self.node_dim = model_conf_dict['node_dim']
        self.bond_dim = model_conf_dict['bond_dim']  # For AIG, this is num_edge_types (e.g., 3 for REG, INV, NO_EDGE)
        self.deq_coeff = model_conf_dict.get('deq_coeff', 0.9)  # Default if not in conf

        # Initialize masks for the autoregressive generation process
        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = \
            self.initialize_masks(max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(0)
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim

        # Device configuration
        self.use_gpu = model_conf_dict.get('use_gpu', False)  # Renamed from self.dp for clarity

        # Flow core model (MaskedGraphAF)
        # Ensure all required keys are present in model_conf_dict or provide defaults
        self.flow_core = MaskedGraphAF(
            mask_node=node_masks,
            mask_edge=adj_masks,
            index_select_edge=link_prediction_index,
            st_type=model_conf_dict.get('st_type', 'sigmoid'),  # Default st_type
            num_flow_layer=model_conf_dict.get('num_flow_layer', 12),
            graph_size=self.max_size,
            num_node_type=self.node_dim,
            num_edge_type=self.bond_dim,  # This is num_channels for adj matrix (e.g., 3 for AIG)
            num_rgcn_layer=model_conf_dict.get('num_rgcn_layer', 3),
            nhid=model_conf_dict.get('nhid', 128),
            nout=model_conf_dict.get('nout', 128)
        )

        # Parameters for log probability calculation
        constant_pi = torch.Tensor([3.1415926535])
        prior_ln_var = torch.zeros([1])

        if self.use_gpu and torch.cuda.is_available():
            self.flow_core = nn.DataParallel(self.flow_core)  # Use DataParallel if use_gpu is true
            self.constant_pi = nn.Parameter(constant_pi.cuda(), requires_grad=False)
            self.prior_ln_var = nn.Parameter(prior_ln_var.cuda(), requires_grad=False)
        else:
            self.constant_pi = nn.Parameter(constant_pi, requires_grad=False)
            self.prior_ln_var = nn.Parameter(prior_ln_var, requires_grad=False)

    def forward(self, inp_node_features, inp_adj_features):
        """
        Forward pass for training.
        Args:
            inp_node_features: (B, N, node_dim)
            inp_adj_features: (B, bond_dim, N, N) - Note: bond_dim here is num_edge_types
        Returns:
            z: Latent variables [(B, node_num*node_dim), (B, edge_num*bond_dim)]
            logdet: Log determinants ([B], [B])
        """
        # Dequantization for continuous flow
        inp_node_features_cont = inp_node_features.clone()
        # The flow_core_edge_masks select the upper triangular part of adj for autoregressive modeling
        inp_adj_features_cont = inp_adj_features[:, :, self.flow_core_edge_masks].clone()
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()  # (B, num_modeled_edges, bond_dim)

        # Add noise for dequantization
        current_device = inp_node_features.device  # Get device from input tensor
        inp_node_features_cont += self.deq_coeff * torch.rand(inp_node_features_cont.size(), device=current_device)
        inp_adj_features_cont += self.deq_coeff * torch.rand(inp_adj_features_cont.size(), device=current_device)

        # Pass through the flow core
        z, logdet = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        return z, logdet

    def generate_aig_raw_data(self, max_nodes, temperature, device):
        """
        Generates raw node features and a list of typed edges for a single AIG.
        This implements the core autoregressive generation logic of GraphAF,
        adapted for AIGs (directed graphs, specific node types).

        Args:
            max_nodes (int): The maximum number of nodes to generate (model's capacity, self.max_size).
            temperature (float): Sampling temperature.
            device (torch.device): The device to perform generation on.

        Returns:
            tuple:
                - raw_node_features (torch.Tensor): Shape (max_nodes, self.node_dim).
                                                    Contains scores/probabilities for each node type.
                - typed_edges_generated (list): List of tuples (source_idx, target_idx, edge_type_idx),
                                                where edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
                - actual_num_nodes (int): The actual number of nodes generated for this AIG.
        """
        with torch.no_grad():
            # Ensure the flow_core (MaskedGraphAF) is on the correct device
            flow_core_model = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core
            # The model should already be on a device from __init__ or by manual .to(device) call on GraphFlowModel instance
            # flow_core_model.to(device) # This line might be redundant if GraphFlowModel instance is already moved.

            # 1. Initialize prior distributions for node and edge latents
            # self.node_dim is expected to be 4 for AIG (CONST0, PI, AND, PO)
            prior_node_dist = torch.distributions.normal.Normal(
                torch.zeros([self.node_dim], device=device),
                temperature * torch.ones([self.node_dim], device=device)
            )
            # self.bond_dim is expected to be 3 for AIG (REG, INV, NO_EDGE channels)
            prior_edge_dist = torch.distributions.normal.Normal(
                torch.zeros([self.bond_dim], device=device),
                temperature * torch.ones([self.bond_dim], device=device)
            )

            # 2. Initialize current node features and adjacency matrix for flow input
            # Node features for flow_core.reverse: (1, max_nodes, self.node_dim)
            cur_node_features = torch.zeros([1, max_nodes, self.node_dim], device=device)
            # Adjacency features for flow_core.reverse: (1, self.bond_dim, max_nodes, max_nodes)
            # This represents the one-hot encoding of edge states (REG, INV, NO_EDGE)
            cur_adj_features_for_flow = torch.zeros([1, self.bond_dim, max_nodes, max_nodes], device=device)

            # Store the actual generated typed edges (source, target, type_index)
            # type_index: 0 for 'EDGE_REG', 1 for 'EDGE_INV' (as per aig_config.EDGE_TYPE_KEYS)
            typed_edges_generated = []

            actual_num_nodes = 0
            # AIG_STOP_NODE_TYPE_INDEX = -1 # Define if you have a special stop node type

            # 3. Autoregressive generation loop
            for i in range(max_nodes):  # Iterate up to the model's capacity (self.max_size)
                # a. Generate node `i`
                latent_node_sample = prior_node_dist.sample().view(1, -1)  # Shape (1, self.node_dim)

                # The reverse method expects inputs shaped for a batch size of 1
                generated_node_i_features = flow_core_model.reverse(
                    x=cur_node_features,  # Current state of all node features (1, max_nodes, node_dim)
                    adj=cur_adj_features_for_flow,
                    # Current state of all edge features (1, bond_dim, max_nodes, max_nodes)
                    latent=latent_node_sample,  # Latent sample for current node (1, node_dim)
                    mode=0  # Mode 0 for node generation
                ).view(-1)  # Flatten to (self.node_dim)

                cur_node_features[0, i, :] = generated_node_i_features
                actual_num_nodes = i + 1

                # Optional: Implement a stopping condition based on a special "stop" node type
                # current_node_type_idx = torch.argmax(generated_node_i_features).item()
                # if current_node_type_idx == AIG_STOP_NODE_TYPE_INDEX:
                #     actual_num_nodes = i # i nodes were generated before stop
                #     break

                # b. Generate edges for node `i` from/to previous nodes (j < i)
                if i > 0:  # No edges for the first node
                    start_node_for_edges = 0
                    if i >= self.edge_unroll:
                        start_node_for_edges = i - self.edge_unroll

                    for prev_node_idx in range(start_node_for_edges, i):
                        latent_edge_sample = prior_edge_dist.sample().view(1, -1)  # Shape (1, self.bond_dim)

                        # edge_index for AIG: [source, target]. Here, prev_node_idx -> i
                        edge_idx_tensor = torch.tensor([[prev_node_idx, i]], device=device).long()

                        # generated_edge_features will have shape (self.bond_dim), e.g., 3 for AIG
                        # These are scores for each category (e.g., REG, INV, NO_EDGE)
                        generated_edge_features = flow_core_model.reverse(
                            x=cur_node_features,
                            adj=cur_adj_features_for_flow,
                            latent=latent_edge_sample,  # Latent sample for current edge (1, bond_dim)
                            mode=1,  # Mode 1 for edge generation
                            edge_index=edge_idx_tensor  # Specifies which edge (u,v) to generate
                        ).view(-1)  # Flatten to (self.bond_dim)

                        # chosen_edge_category_idx: 0 for REG, 1 for INV, 2 for NO_EDGE (if bond_dim=3)
                        # This index corresponds to the channel in cur_adj_features_for_flow
                        chosen_edge_category_idx = torch.argmax(generated_edge_features).item()

                        # --- AIG Edge Interpretation Logic ---
                        # Map chosen_edge_category_idx to actual_edge_type_index for storage in NetworkX graph
                        # actual_edge_type_index: 0 for 'EDGE_REG', 1 for 'EDGE_INV'
                        actual_aig_edge_type_index = -1

                        # Assuming self.bond_dim is 3, representing channels for:
                        # Channel 0: EDGE_REG
                        # Channel 1: EDGE_INV
                        # Channel 2: NO_EDGE
                        # This order must match how the training data (inp_adj_features) was constructed.
                        if self.bond_dim == 3:
                            if chosen_edge_category_idx == 0:  # Corresponds to 'EDGE_REG'
                                actual_aig_edge_type_index = 0
                            elif chosen_edge_category_idx == 1:  # Corresponds to 'EDGE_INV'
                                actual_aig_edge_type_index = 1
                            # If chosen_edge_category_idx == 2 (NO_EDGE), actual_aig_edge_type_index remains -1

                        elif self.bond_dim == 2:
                            # This case implies a different encoding, e.g.:
                            # Channel 0: EDGE_REG, Channel 1: EDGE_INV (NO_EDGE is implicit if neither is chosen, or handled differently)
                            # OR Channel 0: EDGE_EXISTS (type decided by another mechanism), Channel 1: NO_EDGE
                            # For AIGs with distinct REG/INV types, bond_dim=3 (REG, INV, NO_EDGE) is clearer.
                            # If bond_dim is 2, the mapping to 'EDGE_REG'/'EDGE_INV' needs to be defined.
                            # Example: if 0 means REG and 1 means INV.
                            if chosen_edge_category_idx == 0:
                                actual_aig_edge_type_index = 0  # Example: REG
                            elif chosen_edge_category_idx == 1:
                                actual_aig_edge_type_index = 1  # Example: INV
                            print(
                                f"Warning: AIG generation with bond_dim={self.bond_dim}. Ensure mapping to REG/INV is correct.")

                        else:
                            print(
                                f"Error: Unsupported bond_dim ({self.bond_dim}) for AIG edge type interpretation. Expected 2 or 3.")
                            # Handle error or skip edge, or default to no edge

                        # If an actual AIG edge type (REG or INV) was chosen:
                        if actual_aig_edge_type_index != -1:
                            typed_edges_generated.append((prev_node_idx, i, actual_aig_edge_type_index))

                        # Update cur_adj_features_for_flow with the one-hot encoding of the *chosen category*
                        # This includes REG, INV, or NO_EDGE, to condition subsequent generation steps.
                        # chosen_edge_category_idx is the index in the one-hot vector for cur_adj_features_for_flow.
                        if 0 <= chosen_edge_category_idx < self.bond_dim:
                            cur_adj_features_for_flow[0, chosen_edge_category_idx, prev_node_idx, i] = 1.0
                            # For directed AIGs, we only set adj[u,v].
                        else:
                            # This should not happen if chosen_edge_category_idx comes from argmax of scores of size bond_dim
                            print(
                                f"Warning: chosen_edge_category_idx {chosen_edge_category_idx} is out of bounds for bond_dim {self.bond_dim}.")

            # 4. Prepare outputs
            raw_node_features_output = cur_node_features.squeeze(0)  # Shape (max_nodes, self.node_dim)

            return raw_node_features_output, typed_edges_generated, actual_num_nodes

    def initialize_masks(self, max_node_unroll, max_edge_unroll):
        """
        Initializes masks for the autoregressive generation process.
        This defines the order of node and edge generation.
        Args:
            max_node_unroll (int): Max number of nodes (self.max_size).
            max_edge_unroll (int): Max number of edges to predict for each new node (self.edge_unroll).
        Returns:
            tuple: node_masks, adj_masks, link_prediction_index, flow_core_edge_masks
        """
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (
                max_node_unroll - max_edge_unroll) * max_edge_unroll)
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).bool()
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()

        # flow_core_edge_masks: defines which entries of the adjacency matrix are modeled by the flow
        # Typically the upper or lower triangular part, excluding the diagonal.
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        cnt_node_step = 0  # Counter for node generation steps
        cnt_edge_step = 0  # Counter for edge generation steps

        for i in range(max_node_unroll):  # For each new node 'i' being added
            # Mask for node 'i' generation:
            # Nodes 0 to i-1 are visible.
            node_masks1[cnt_node_step, :i] = True
            # Adjacency matrix among nodes 0 to i-1 is visible.
            adj_masks1[cnt_node_step, :i, :i] = True
            cnt_node_step += 1

            # Determine edges to generate for this new node 'i'
            # Edges connect node 'i' to a subset of previous nodes (0 to i-1)
            # The subset is determined by 'max_edge_unroll'.
            num_edges_to_generate_for_node_i = 0
            start_prev_node_idx = 0  # Start index of previous nodes to connect to
            if i < max_edge_unroll:  # If current node index is less than unroll factor
                num_edges_to_generate_for_node_i = i  # Connect to all previous nodes (0 to i-1)
            else:  # If current node index is >= unroll factor
                num_edges_to_generate_for_node_i = max_edge_unroll  # Connect to 'max_edge_unroll' previous nodes
                start_prev_node_idx = i - max_edge_unroll  # Starting from (i - max_edge_unroll)

            for j in range(num_edges_to_generate_for_node_i):
                prev_node_idx = start_prev_node_idx + j  # The specific previous node

                # Mask for generating edge between 'prev_node_idx' and 'i':
                # Nodes 0 to 'i' are visible (node 'i' itself is now known).
                node_masks2[cnt_edge_step, :i + 1] = True
                # Adjacency matrix:
                # - Connections among 0 to i-1 are visible.
                # - Node 'i' is visible (e.g., self-loop info if modeled, or just its presence).
                # - Edges already generated for node 'i' (to prev_nodes < prev_node_idx) are visible.
                if j == 0:  # First edge for node 'i'
                    adj_masks2[cnt_edge_step] = adj_masks1[cnt_node_step - 1].clone()  # Start with adj among 0..i-1
                    adj_masks2[cnt_edge_step, i, i] = True  # Node 'i' is now part of the conditioning context
                else:  # Subsequent edges for node 'i'
                    adj_masks2[cnt_edge_step] = adj_masks2[cnt_edge_step - 1].clone()
                    # The edge (start_prev_node_idx + j - 1) <-> i was just added in the previous edge step
                    # So, make it visible in the current adjacency mask.
                    # This assumes directed edges, e.g. prev_node_idx -> i
                    adj_masks2[cnt_edge_step, start_prev_node_idx + j - 1, i] = True
                    # If modeling undirected, also set [i, start_prev_node_idx + j - 1]
                    # adj_masks2[cnt_edge_step, i, start_prev_node_idx + j - 1] = True

                link_prediction_index[cnt_edge_step, 0] = prev_node_idx
                link_prediction_index[cnt_edge_step, 1] = i
                cnt_edge_step += 1

        assert cnt_node_step == max_node_unroll
        assert cnt_edge_step == num_mask_edge

        # flow_core_edge_masks: defines which part of the full adjacency matrix the flow models.
        # For an autoregressive model generating node i and then its edges to j < i,
        # this typically means the lower triangular part (if adj[row, col] means row -> col).
        # Or upper triangular if adj[row, col] means col -> row.
        # The link_prediction_index stores pairs (prev_node_idx, i) where prev_node_idx < i.
        # So, flow_core_edge_masks should mark these positions.
        for k in range(num_mask_edge):
            u, v = link_prediction_index[k, 0].item(), link_prediction_index[k, 1].item()
            # Assuming link_prediction_index[k] = [source, target] for a directed edge
            flow_core_edge_masks[u, v] = True

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        # Convert to Parameters so they are part of the model's state_dict but not trained
        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks

    def log_prob(self, z, logdet):
        """
        Calculates the log probability for training.
        Args:
            z (list): List of latent variables [z_node, z_edge].
            logdet (list): List of log determinants [logdet_node, logdet_edge].
        Returns:
            torch.Tensor: The negative log-likelihood loss.
        """
        # Adjust logdet for the volume change in dequantization (subtracting a constant)
        logdet_node, logdet_edge = logdet[0], logdet[1]
        # These subtractions are related to the volume of the dequantization noise hypercube.
        # For a uniform dequantization U(-0.5, 0.5) scaled by deq_coeff, the log volume is log(deq_coeff).
        # The original paper might have specific reasons for latent_node_length.
        # If deq_coeff is applied to each dimension, then it's N * D * log(deq_coeff).
        # Here, it seems to be a fixed offset based on total latent dimensions.
        logdet_node = logdet_node - self.latent_node_length * np.log(
            self.deq_coeff) if self.deq_coeff > 0 else logdet_node
        logdet_edge = logdet_edge - self.latent_edge_length * np.log(
            self.deq_coeff) if self.deq_coeff > 0 else logdet_edge

        # Log-likelihood of latent z under a standard Gaussian prior N(0,I)
        # log p(z) = -0.5 * (log(2*pi) + z^2) per dimension
        # self.prior_ln_var is 0, so prior variance is 1.
        ll_node = -0.5 * (
                torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0] ** 2))
        ll_node = ll_node.sum(-1)  # Sum over latent dimensions for each sample in batch (B)

        ll_edge = -0.5 * (
                torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[1] ** 2))
        ll_edge = ll_edge.sum(-1)  # Sum over latent dimensions for each sample in batch (B)

        # Total log-likelihood log p(x) = log p(z) + log |det(dz/dx)|
        ll_node += logdet_node  # (B)
        ll_edge += logdet_edge  # (B)

        # Average negative log-likelihood per modeled latent dimension
        # This is the loss to be minimized.
        total_latent_dims = self.latent_node_length + self.latent_edge_length
        if total_latent_dims == 0: return torch.tensor(0.0, device=ll_node.device)  # Avoid division by zero

        return -(torch.mean(ll_node + ll_edge) / total_latent_dims)

    # dis_log_prob might be for a discrete flow version, not directly used if use_df=False
    def dis_log_prob(self, z):
        # This method seems to assume z contains dequantized one-hot vectors
        # and calculates log prob based on learned categorical distribution parameters.
        # It's likely for a different variant or an extension not fully used in the continuous flow.
        x_deq, adj_deq = z  # Assuming these are (B, N, node_dim) and (B, E, bond_dim)

        # Check if base_log_probs are defined; if not, this method is not applicable
        if not hasattr(self, 'node_base_log_probs') or not hasattr(self, 'edge_base_log_probs'):
            raise NotImplementedError("Discrete log probability is not configured (base_log_probs missing).")

        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        ll_node = torch.sum(x_deq * node_base_log_probs_sm, dim=(-1, -2))  # Sum over node_dim and N

        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        ll_edge = torch.sum(adj_deq * edge_base_log_probs_sm, dim=(-1, -2))  # Sum over bond_dim and E

        total_latent_dims = self.latent_node_length + self.latent_edge_length
        if total_latent_dims == 0: return torch.tensor(0.0, device=ll_node.device)

        return -(torch.mean(ll_node + ll_edge) / total_latent_dims)
