import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  # Added for F.one_hot

# Assuming DisGraphAF is defined in .disgraphaf
# If GraphDF uses a different flow core, import it accordingly.
try:
    from .disgraphaf import DisGraphAF  # For GraphDF
except ImportError:
    # Fallback or error if DisGraphAF is structured differently
    print("Warning: DisGraphAF not found directly. Ensure it's correctly placed for GraphDF's GraphFlowModel.")


    class DisGraphAF(nn.Module):  # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); raise NotImplementedError(
            "DisGraphAF placeholder used.")

        def forward(self, *args, **kwargs): raise NotImplementedError("DisGraphAF placeholder used.")

        def reverse(self, *args, **kwargs): raise NotImplementedError("DisGraphAF placeholder used.")


class GraphFlowModel(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        # --- Original __init__ content ---
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict.get('edge_unroll', 12)  # Use .get for robustness
        self.node_dim = model_conf_dict['node_dim']  # Expected to be 4 for AIGs
        self.bond_dim = model_conf_dict['bond_dim']  # Expected to be 3 for AIGs (REG, INV, NO_EDGE)

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
            max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(0)
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim

        self.use_gpu = model_conf_dict.get('use_gpu', False)  # Renamed from self.dp

        node_base_log_probs_init = torch.randn(self.max_size, self.node_dim)
        num_mask_edge = int(self.latent_step - self.max_size)  # Calculate num_mask_edge here
        edge_base_log_probs_init = torch.randn(num_mask_edge, self.bond_dim)  # Use num_mask_edge

        # Instantiate DisGraphAF flow core
        self.flow_core = DisGraphAF(
            mask_node=node_masks,
            mask_edge=adj_masks,
            index_select_edge=link_prediction_index,
            num_flow_layer=model_conf_dict.get('num_flow_layer', 12),  # Use .get
            graph_size=self.max_size,
            num_node_type=self.node_dim,
            num_edge_type=self.bond_dim,
            num_rgcn_layer=model_conf_dict.get('num_rgcn_layer', 3),  # Use .get
            nhid=model_conf_dict.get('nhid', 128),  # Use .get
            nout=model_conf_dict.get('nout', 128)  # Use .get
        )

        # Handle device placement and DataParallel
        if self.use_gpu and torch.cuda.is_available():
            target_device = torch.device("cuda")
            # Apply DataParallel *before* moving parameters to CUDA
            self.flow_core = nn.DataParallel(self.flow_core)
            self.node_base_log_probs = nn.Parameter(node_base_log_probs_init.to(target_device), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs_init.to(target_device), requires_grad=True)
        else:
            target_device = torch.device("cpu")
            # No DataParallel needed for CPU
            self.node_base_log_probs = nn.Parameter(node_base_log_probs_init.to(target_device), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs_init.to(target_device), requires_grad=True)

        # Move the flow core itself to the target device after handling DataParallel
        self.flow_core.to(target_device)
        # --- End Original __init__ content ---

    def forward(self, inp_node_features, inp_adj_features):
        """
        Forward pass for GraphDF training.
        Args:
            inp_node_features: (B, N, node_dim) - Expected one-hot
            inp_adj_features: (B, bond_dim, N, N) - Expected one-hot
        Returns:
            z (tuple): Output from DisGraphAF, expected by dis_log_prob.
                       (z_node_transformed, z_edge_transformed)
        """
        # --- Original forward content ---
        # These clones might be needed if DisGraphAF expects dequantized inputs
        # even though GraphDF focuses on the discrete loss. Check DisGraphAF's forward signature.
        inp_node_features_cont = inp_node_features.clone()  # (B, N, node_dim)

        inp_adj_features_cont = inp_adj_features[:, :,
                                self.flow_core_edge_masks].clone()  # (B, bond_dim, num_modeled_edges)
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()  # (B, num_modeled_edges, bond_dim)

        # Call the flow core (DisGraphAF)
        # Assuming it takes these 4 arguments based on the original snippet provided
        z = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        # --- End Original forward content ---
        return z

    # --- MODIFIED AIG Generation Method ---
    def generate_aig_discrete_raw_data(self, max_nodes, temperature_node, temperature_edge, device):
        """
        Generates raw node features (one-hot) and a list of typed AIG edges using discrete sampling.
        Args:
            max_nodes (int): Max number of nodes (model's capacity, self.max_size).
            temperature_node (float): Temperature for node type sampling. Should be > 0.
            temperature_edge (float): Temperature for edge type sampling. Should be > 0.
            device (torch.device): Device for generation.
        Returns:
            tuple:
                - raw_node_features_one_hot (torch.Tensor): Shape (max_nodes, self.node_dim), one-hot.
                - typed_edges_generated (list): List of (source, target, actual_aig_edge_type_idx)
                                                where actual_aig_edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
                - actual_num_nodes (int): Actual number of nodes generated.
        """
        # Ensure temperatures are positive to avoid division by zero or invalid logits
        if temperature_node <= 1e-9: temperature_node = 1e-9  # Small positive value
        if temperature_edge <= 1e-9: temperature_edge = 1e-9  # Small positive value

        with torch.no_grad():
            flow_core_model = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core
            # Ensure model components like base_log_probs are on the correct device
            # This should be handled by the main model's .to(device) call.

            # Initialize tensors to store the generated graph state (one-hot) for conditioning
            cur_node_features_one_hot = torch.zeros([1, max_nodes, self.node_dim], device=device)
            # cur_adj_features_one_hot stores the one-hot state of each edge slot (REG, INV, NO_EDGE)
            cur_adj_features_one_hot = torch.zeros([1, self.bond_dim, max_nodes, max_nodes], device=device)

            # Store the final AIG typed edges
            typed_edges_generated = []  # List of (u, v, type_idx) where type_idx is 0 for REG, 1 for INV

            actual_num_nodes = 0
            edge_step_idx = 0  # Counter for edge generation steps, to index self.edge_base_log_probs

            for i in range(max_nodes):  # Generate node i
                # --- Node Generation ---
                # 1. Sample initial latent node type from the learned base distribution for node i
                # self.node_base_log_probs has shape (max_size, node_dim)
                node_logits = self.node_base_log_probs[i] / temperature_node
                prior_node_dist = torch.distributions.Categorical(logits=node_logits)
                latent_node_idx = prior_node_dist.sample()  # Sampled index
                latent_node_one_hot = F.one_hot(latent_node_idx, num_classes=self.node_dim).float().unsqueeze(
                    0)  # Shape (1, node_dim)

                # 2. Pass through the flow's reverse function to get final logits for node i's type
                # DisGraphAF.reverse(x_cond_onehot, adj_cond_onehot, z_onehot, mode, edge_index=None)
                output_node_logits = flow_core_model.reverse(
                    x_cond_onehot=cur_node_features_one_hot,  # Graph state so far (nodes)
                    adj_cond_onehot=cur_adj_features_one_hot,  # Graph state so far (edges)
                    z_onehot=latent_node_one_hot,  # Sampled base latent for current node
                    mode=0  # Node generation mode
                ).view(-1)  # Output logits for node type, shape (node_dim)

                # 3. Sample the final node type for node 'i'
                final_node_type_idx = torch.distributions.Categorical(logits=output_node_logits).sample()
                cur_node_features_one_hot[0, i, :] = F.one_hot(final_node_type_idx, num_classes=self.node_dim).float()
                actual_num_nodes = i + 1

                # Optional: Implement a stop condition based on a special AIG node type
                # e.g., if AIG_NODE_TYPE_KEYS.index('NODE_PO') is considered a stop for some logic,
                # or if a dedicated stop token/type exists. For now, generate up to max_nodes.
                # if final_node_type_idx == AIG_STOP_NODE_TYPE_INDEX:
                #     actual_num_nodes = i # Correct count if stopping before filling max_nodes
                #     break

                # --- Edge Generation for node 'i' ---
                # Generate edges connecting node 'i' to relevant previous nodes 'prev_node_idx' < i
                if i > 0:  # Only generate edges if there are previous nodes
                    # Determine the range of previous nodes based on edge_unroll
                    start_prev_node_idx_for_edges = 0
                    if i >= self.edge_unroll:
                        start_prev_node_idx_for_edges = i - self.edge_unroll

                    for prev_node_idx in range(start_prev_node_idx_for_edges, i):
                        # Ensure edge_step_idx is within bounds for self.edge_base_log_probs
                        if edge_step_idx >= self.edge_base_log_probs.shape[0]:
                            print(f"Warning: edge_step_idx ({edge_step_idx}) is out of bounds for "
                                  f"edge_base_log_probs (size: {self.edge_base_log_probs.shape[0]}). "
                                  f"Stopping edge generation for current node {i}.")
                            break  # Stop generating more edges for this current node i

                        # 1. Sample initial latent edge category from base distribution for this edge step
                        # self.edge_base_log_probs has shape (num_mask_edge, bond_dim)
                        edge_category_logits = self.edge_base_log_probs[edge_step_idx] / temperature_edge
                        prior_edge_category_dist = torch.distributions.Categorical(logits=edge_category_logits)
                        latent_edge_category_idx = prior_edge_category_dist.sample()  # Sampled index (0, 1, or 2 if bond_dim=3)
                        latent_edge_category_one_hot = F.one_hot(latent_edge_category_idx,
                                                                 num_classes=self.bond_dim).float().unsqueeze(
                            0)  # Shape (1, bond_dim)

                        # 2. Pass through flow's reverse function to get final logits for this edge's category
                        edge_idx_tensor = torch.tensor([[prev_node_idx, i]],
                                                       device=device).long()  # Edge: prev_node_idx -> i

                        output_edge_category_logits = flow_core_model.reverse(
                            x_cond_onehot=cur_node_features_one_hot,  # Current node states
                            adj_cond_onehot=cur_adj_features_one_hot,  # Current edge states (one-hot categories)
                            z_onehot=latent_edge_category_one_hot,  # Sampled base latent for current edge category
                            mode=1,  # Edge generation mode
                            edge_index=edge_idx_tensor  # Specifies which edge (u,v)
                        ).view(-1)  # Output logits for edge category, shape (bond_dim)

                        # 3. Sample final edge category (0 for REG, 1 for INV, 2 for NO_EDGE if bond_dim=3)
                        final_edge_category_idx = torch.distributions.Categorical(
                            logits=output_edge_category_logits).sample().item()

                        # 4. AIG Edge Interpretation & Update Output List
                        # actual_aig_edge_type_index: 0 for 'EDGE_REG', 1 for 'EDGE_INV'
                        actual_aig_edge_type_index = -1  # Default to no AIG-specific edge

                        if self.bond_dim == 3:  # Assuming AIG uses bond_dim=3 for [REG, INV, NO_EDGE]
                            if final_edge_category_idx == 0:  # REG category
                                actual_aig_edge_type_index = 0
                            elif final_edge_category_idx == 1:  # INV category
                                actual_aig_edge_type_index = 1
                            # If final_edge_category_idx == 2 (NO_EDGE), actual_aig_edge_type_index remains -1
                        elif self.bond_dim == 2:  # E.g. [REG, INV] and NO_EDGE is implicit or handled by scores
                            if final_edge_category_idx == 0:
                                actual_aig_edge_type_index = 0
                            elif final_edge_category_idx == 1:
                                actual_aig_edge_type_index = 1
                            print(
                                f"Warning: AIG generation with bond_dim={self.bond_dim}. Ensure mapping to REG/INV is correct.")
                        else:
                            print(
                                f"Error: GraphDF AIG generation expects bond_dim of 2 or 3 for REG/INV types, got {self.bond_dim}.")

                        if actual_aig_edge_type_index != -1:
                            typed_edges_generated.append((prev_node_idx, i, actual_aig_edge_type_index))

                        # 5. Update conditioning tensor `cur_adj_features_one_hot`
                        # This must reflect the actual sampled category (0, 1, or 2) for the flow's conditioning.
                        if 0 <= final_edge_category_idx < self.bond_dim:
                            cur_adj_features_one_hot[0, final_edge_category_idx, prev_node_idx, i] = 1.0
                        else:
                            # This case should ideally not be reached if sampling is correct
                            print(
                                f"Warning: final_edge_category_idx {final_edge_category_idx} is out of bounds for bond_dim {self.bond_dim}.")
                            # Fallback: assume NO_EDGE if category is invalid
                            if self.bond_dim > 0:  # Check if bond_dim is valid
                                cur_adj_features_one_hot[
                                    0, self.bond_dim - 1, prev_node_idx, i] = 1.0  # Mark last channel (assumed NO_EDGE)

                        edge_step_idx += 1  # Increment for the next edge generation step

            # Prepare final outputs
            raw_node_features_output_one_hot = cur_node_features_one_hot.squeeze(0)  # Remove batch dim

            return raw_node_features_output_one_hot, typed_edges_generated, actual_num_nodes

    # --- End MODIFIED Method ---

    def initialize_masks(self, max_node_unroll, max_edge_unroll):
        """
        Initializes masks for the autoregressive generation process.
        (Identical logic to the original GraphDF snippet provided by the user)
        """
        # --- Original initialize_masks content ---
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (
                max_node_unroll - max_edge_unroll) * max_edge_unroll)
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).bool()
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                prev_node_idx = start + j
                if j == 0:
                    node_masks2[cnt_edge][:i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node - 1].clone()
                    adj_masks2[cnt_edge][i, i] = 1
                else:
                    node_masks2[cnt_edge][:i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge - 1].clone()
                    adj_masks2[cnt_edge][start + j - 1, i] = 1
                cnt += 1
                cnt_edge += 1

        # These assertions were not in the original user snippet but are good checks
        assert cnt_node == max_node_unroll, f"Node mask step count: {cnt_node} vs expected {max_node_unroll}"
        assert cnt_edge == num_mask_edge, f"Edge mask step count: {cnt_edge} vs expected {num_mask_edge}"
        assert cnt == num_masks, f"Total mask steps: {cnt} vs expected {num_masks}"

        cnt = 0  # Reset counter for populating link_prediction_index
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll

            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1

        assert cnt == num_mask_edge, f"Link prediction index count: {cnt} vs expected {num_mask_edge}"

        for k_idx in range(num_mask_edge):  # Use a different loop variable name
            u, v = link_prediction_index[k_idx, 0].item(), link_prediction_index[k_idx, 1].item()
            if 0 <= u < max_node_unroll and 0 <= v < max_node_unroll:
                flow_core_edge_masks[u, v] = True
            else:
                # This case should ideally not happen if link_prediction_index is populated correctly
                print(
                    f"Warning: Invalid indices ({u}, {v}) from link_prediction_index at step {k_idx} when setting flow_core_edge_masks.")

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)
        # --- End Original initialize_masks content ---

        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks

    def dis_log_prob(self, z_tuple):
        """
        Calculates discrete log probability for training GraphDF.
        Args:
            z_tuple (tuple): (z_node_transformed, z_edge_transformed) from forward pass.
        Returns:
            torch.Tensor: The negative log-likelihood loss.
        """
        # --- Original dis_log_prob content ---
        z_node_transformed, z_edge_transformed = z_tuple

        # Reshape if necessary (assuming forward pass might flatten)
        if z_node_transformed.ndim == 2:  # (B * max_size, node_dim) or (B, max_size * node_dim)
            # Assuming z_node_transformed from DisGraphAF is (B, max_size * node_dim)
            # and needs to be reshaped to (B, max_size, node_dim) for element-wise product with base probs
            z_node_transformed = z_node_transformed.view(-1, self.max_size, self.node_dim)

        if z_edge_transformed.ndim == 2:  # (B * num_modeled_edges, bond_dim) or (B, num_modeled_edges * bond_dim)
            # Assuming z_edge_transformed from DisGraphAF is (B, num_modeled_edges * bond_dim)
            num_modeled_edges = self.edge_base_log_probs.shape[0]
            z_edge_transformed = z_edge_transformed.view(-1, num_modeled_edges, self.bond_dim)

        # Calculate log-likelihood based on base distributions
        # self.node_base_log_probs is (max_size, node_dim)
        # self.edge_base_log_probs is (num_modeled_edges, bond_dim)
        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        # z_node_transformed is (B, max_size, node_dim), node_base_log_probs_sm is (max_size, node_dim)
        # Broadcasting unsqueeze(0) for batch dimension on base_log_probs
        ll_node = torch.sum(z_node_transformed * node_base_log_probs_sm.unsqueeze(0),
                            dim=(-1, -2))  # Sum over node_dim and max_size

        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        # z_edge_transformed is (B, num_modeled_edges, bond_dim), edge_base_log_probs_sm is (num_modeled_edges, bond_dim)
        ll_edge = torch.sum(z_edge_transformed * edge_base_log_probs_sm.unsqueeze(0),
                            dim=(-1, -2))  # Sum over bond_dim and num_modeled_edges

        # Normalize loss by total number of latent dimensions modeled
        total_dims = self.latent_node_length + self.latent_edge_length  # These are total possible one-hot elements
        if total_dims == 0:
            # Determine device safely
            current_device = torch.device("cpu")
            if hasattr(ll_node, 'device'):
                current_device = ll_node.device
            elif hasattr(ll_edge, 'device'):
                current_device = ll_edge.device
            return torch.tensor(0.0, device=current_device, requires_grad=True)

        loss = -(torch.mean(ll_node + ll_edge) / total_dims)
        # --- End Original dis_log_prob content ---
        return loss
