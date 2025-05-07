import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import defaultdict # Added for current_in_degrees

# Assuming DisGraphAF is defined in .disgraphaf
try:
    from .disgraphaf import DisGraphAF
except ImportError:
    print("Warning: DisGraphAF not found directly. Ensure it's correctly placed for GraphDF's GraphFlowModel.")
    class DisGraphAF(nn.Module): # Placeholder
        def __init__(self, *args, **kwargs): super().__init__(); raise NotImplementedError("DisGraphAF placeholder used.")
        def forward(self, *args, **kwargs): raise NotImplementedError("DisGraphAF placeholder used.")
        def reverse(self, *args, **kwargs): raise NotImplementedError("DisGraphAF placeholder used.")

# Attempt to import AIG config for node/edge type keys
try:
    from G2PT.configs import aig as aig_config
    AIG_NODE_TYPE_KEYS = aig_config.NODE_TYPE_KEYS
    AIG_EDGE_TYPE_KEYS = aig_config.EDGE_TYPE_KEYS # REG, INV
    print("Successfully imported AIG type keys from G2PT.configs.aig for GraphFlowModel.")
except ImportError:
    print("Warning (GraphFlowModel): G2PT.configs.aig not found. Using default AIG type keys.")
    AIG_NODE_TYPE_KEYS = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']  # Default
    AIG_EDGE_TYPE_KEYS = ['EDGE_REG', 'EDGE_INV']  # Default


class GraphFlowModel(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict.get('edge_unroll', 12)
        self.node_dim = model_conf_dict['node_dim'] # Expected 4 for AIG
        self.bond_dim = model_conf_dict['bond_dim'] # Expected 3 for AIG (REG, INV, NO_EDGE categories)

        # --- DEBUG PRINT in __init__ ---
        print(f"GraphFlowModel __init__: max_size={self.max_size}, edge_unroll={self.edge_unroll}")
        # --- End DEBUG ---

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
            max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(0)
        self.latent_node_length = self.max_size * self.node_dim
        num_mask_edge = int(self.latent_step - self.max_size) # Calculate num_mask_edge
        self.latent_edge_length = num_mask_edge * self.bond_dim # Correct calculation using num_mask_edge

        # --- DEBUG PRINT in __init__ ---
        print(f"GraphFlowModel __init__: latent_step={self.latent_step}, num_mask_edge={num_mask_edge}")
        # --- End DEBUG ---

        self.use_gpu = model_conf_dict.get('use_gpu', False)

        node_base_log_probs_init = torch.randn(self.max_size, self.node_dim)
        # Use the calculated num_mask_edge for edge_base_log_probs size
        edge_base_log_probs_init = torch.randn(num_mask_edge, self.bond_dim)

        # --- DEBUG PRINT in __init__ ---
        print(f"GraphFlowModel __init__: Initializing edge_base_log_probs with shape: {edge_base_log_probs_init.shape}")
        # --- End DEBUG ---


        self.flow_core = DisGraphAF(
            mask_node=node_masks,
            mask_edge=adj_masks,
            index_select_edge=link_prediction_index,
            num_flow_layer=model_conf_dict.get('num_flow_layer', 12),
            graph_size=self.max_size,
            num_node_type=self.node_dim,
            num_edge_type=self.bond_dim, # Pass bond_dim (e.g., 3) to DisGraphAF
            num_rgcn_layer=model_conf_dict.get('num_rgcn_layer', 3),
            nhid=model_conf_dict.get('nhid', 128),
            nout=model_conf_dict.get('nout', 128)
        )

        if self.use_gpu and torch.cuda.is_available():
            target_device = torch.device("cuda")
            if torch.cuda.device_count() > 1:
                print(f"GraphFlowModel: Using nn.DataParallel for flow_core across {torch.cuda.device_count()} GPUs.")
                self.flow_core = nn.DataParallel(self.flow_core)
            else:
                print("GraphFlowModel: Single GPU detected or DataParallel not explicitly enabled for 1 GPU. Not wrapping flow_core.")
            self.node_base_log_probs = nn.Parameter(node_base_log_probs_init.to(target_device), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs_init.to(target_device), requires_grad=True)
        else:
            target_device = torch.device("cpu")
            print("GraphFlowModel: Using CPU.")
            self.node_base_log_probs = nn.Parameter(node_base_log_probs_init.to(target_device), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs_init.to(target_device), requires_grad=True)

        self.flow_core.to(target_device)

    def forward(self, inp_node_features, inp_adj_features):
        inp_node_features_cont = inp_node_features.clone()
        # Ensure flow_core_edge_masks is boolean before indexing
        if hasattr(self.flow_core_edge_masks, 'dtype') and self.flow_core_edge_masks.dtype != torch.bool:
             flow_core_edge_masks_bool = self.flow_core_edge_masks.bool()
        else:
             flow_core_edge_masks_bool = self.flow_core_edge_masks

        inp_adj_features_cont = inp_adj_features[:, :, flow_core_edge_masks_bool].clone()
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()
        z = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        return z

    def generate_aig_discrete_raw_data(self, max_nodes, temperature_node, temperature_edge, device):
        """
        Generates raw node features (one-hot) and a list of typed AIG edges using discrete sampling
        with validity checking and resampling for edges.
        Args:
            max_nodes (int): Max number of nodes (model's capacity, self.max_size).
            temperature_node (float): Temperature for node type sampling.
            temperature_edge (float): Temperature for edge type sampling.
            device (torch.device): Device for generation.
        Returns:
            tuple:
                - raw_node_features_one_hot (torch.Tensor): Shape (max_nodes, self.node_dim), one-hot.
                - typed_edges_generated (list): List of (source, target, actual_aig_edge_type_idx)
                                                where actual_aig_edge_type_idx is 0 for 'EDGE_REG', 1 for 'EDGE_INV'.
                - actual_num_nodes (int): Actual number of nodes generated.
        """
        if temperature_node <= 1e-9: temperature_node = 1e-9
        if temperature_edge <= 1e-9: temperature_edge = 1e-9

        with torch.no_grad():
            flow_core_model = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core

            cur_node_features_one_hot = torch.zeros([1, max_nodes, self.node_dim], device=device)
            cur_adj_features_one_hot = torch.zeros([1, self.bond_dim, max_nodes, max_nodes], device=device)

            typed_edges_generated = []
            current_node_types = {}
            current_in_degrees = defaultdict(int)

            actual_num_nodes = 0
            edge_step_idx = 0
            max_resamples_per_edge_slot = 5

            # --- DEBUG: Get edge_base_log_probs shape once ---
            edge_base_probs_shape_0 = self.edge_base_log_probs.shape[0]
            print(f"--- Starting Generation: edge_base_log_probs shape[0] = {edge_base_probs_shape_0} ---")
            # --- End DEBUG ---

            for i in range(max_nodes):
                # --- Node Generation ---
                node_logits_prior = self.node_base_log_probs[i] / temperature_node
                prior_node_dist = torch.distributions.Categorical(logits=node_logits_prior)
                latent_node_idx_prior = prior_node_dist.sample()
                latent_node_one_hot_prior = F.one_hot(latent_node_idx_prior, num_classes=self.node_dim).float().unsqueeze(0).to(device)

                output_node_logits = flow_core_model.reverse(
                    x_cond_onehot=cur_node_features_one_hot,
                    adj_cond_onehot=cur_adj_features_one_hot,
                    z_onehot=latent_node_one_hot_prior,
                    mode=0
                ).view(-1)

                final_node_type_idx = torch.distributions.Categorical(logits=output_node_logits).sample().item()
                cur_node_features_one_hot[0, i, final_node_type_idx] = 1.0
                current_node_types[i] = AIG_NODE_TYPE_KEYS[final_node_type_idx] if 0 <= final_node_type_idx < len(AIG_NODE_TYPE_KEYS) else "UNKNOWN_NODE"
                actual_num_nodes = i + 1

                # --- Edge Generation ---
                if i > 0:
                    start_prev_node_idx_for_edges = max(0, i - self.edge_unroll)
                    for prev_node_idx in range(start_prev_node_idx_for_edges, i):

                        # --- Moved and Enhanced Debug Print ---
                        print(f"DEBUG: Node i={i}, Edge ({prev_node_idx} -> {i}). Current edge_step_idx={edge_step_idx}. Max edge steps={edge_base_probs_shape_0}")
                        # --- End Debug Print ---

                        # Check BEFORE attempting access
                        if edge_step_idx >= edge_base_probs_shape_0:
                            print(f"CRITICAL WARNING: edge_step_idx ({edge_step_idx}) >= edge_base_log_probs size ({edge_base_probs_shape_0}). Breaking inner edge loop for node {i}.")
                            break # Break the inner loop (for prev_node_idx)

                        valid_edge_category_found_for_slot = False
                        resample_count_this_slot = 0
                        invalid_categories_tried_this_slot = set()

                        # Access edge_base_log_probs ONLY if index is valid
                        current_edge_slot_base_logits = self.edge_base_log_probs[edge_step_idx].clone().to(device)

                        while not valid_edge_category_found_for_slot and \
                              resample_count_this_slot < max_resamples_per_edge_slot and \
                              len(invalid_categories_tried_this_slot) < self.bond_dim:

                            temp_logits_for_sampling = current_edge_slot_base_logits.clone()
                            for tried_invalid_cat in invalid_categories_tried_this_slot:
                                if 0 <= tried_invalid_cat < self.bond_dim:
                                    temp_logits_for_sampling[tried_invalid_cat] = -float('inf')

                            if torch.isinf(temp_logits_for_sampling).all():
                                print(f"Debug: Node {i}, Edge ({prev_node_idx},{i}): All categories masked or invalid. Forcing NO_EDGE.")
                                sampled_final_edge_category_idx = self.bond_dim - 1
                                break

                            prior_edge_dist = torch.distributions.Categorical(logits=temp_logits_for_sampling / temperature_edge)
                            latent_edge_idx_prior = prior_edge_dist.sample()
                            latent_edge_one_hot_prior = F.one_hot(latent_edge_idx_prior, num_classes=self.bond_dim).float().unsqueeze(0).to(device)

                            edge_indices_for_reverse = torch.tensor([[prev_node_idx, i]], device=device).long()
                            output_edge_category_logits = flow_core_model.reverse(
                                x_cond_onehot=cur_node_features_one_hot,
                                adj_cond_onehot=cur_adj_features_one_hot,
                                z_onehot=latent_edge_one_hot_prior,
                                mode=1,
                                edge_index=edge_indices_for_reverse
                            ).view(-1)

                            final_logits_for_sampling = output_edge_category_logits.clone()
                            for tried_invalid_cat_runtime in invalid_categories_tried_this_slot:
                                if 0 <= tried_invalid_cat_runtime < self.bond_dim:
                                     final_logits_for_sampling[tried_invalid_cat_runtime] = -float('inf')

                            if torch.isinf(final_logits_for_sampling).all():
                                print(f"Debug: Node {i}, Edge ({prev_node_idx},{i}): All categories masked post-flow. Forcing NO_EDGE.")
                                sampled_final_edge_category_idx = self.bond_dim - 1
                                break

                            sampled_final_edge_category_idx = torch.distributions.Categorical(logits=final_logits_for_sampling).sample().item()

                            is_locally_valid = True
                            tentative_aig_edge_type = -1

                            if sampled_final_edge_category_idx == 0: tentative_aig_edge_type = 0
                            elif sampled_final_edge_category_idx == 1: tentative_aig_edge_type = 1

                            if tentative_aig_edge_type != -1:
                                target_node_type_str = current_node_types.get(i, "UNKNOWN_NODE")
                                target_node_current_in_degree = current_in_degrees.get(i, 0)

                                if target_node_type_str == "NODE_PI" or target_node_type_str == "NODE_CONST0":
                                    is_locally_valid = False
                                elif target_node_type_str == "NODE_AND":
                                    if target_node_current_in_degree >= 2:
                                        is_locally_valid = False

                            if is_locally_valid:
                                valid_edge_category_found_for_slot = True
                                cur_adj_features_one_hot[0, sampled_final_edge_category_idx, i, prev_node_idx] = 1.0
                                cur_adj_features_one_hot[0, sampled_final_edge_category_idx, prev_node_idx, i] = 1.0

                                if tentative_aig_edge_type != -1:
                                    typed_edges_generated.append((prev_node_idx, i, tentative_aig_edge_type))
                                    current_in_degrees[i] += 1
                            else:
                                invalid_categories_tried_this_slot.add(sampled_final_edge_category_idx)
                                resample_count_this_slot += 1
                                # print(f"Debug: Node {i}, Edge ({prev_node_idx} -> {i}): Category {sampled_final_edge_category_idx} (AIG type {tentative_aig_edge_type}) failed local AIG check. "
                                #       f"Target type: {current_node_types.get(i)}, In-degree: {current_in_degrees.get(i,0)}. Resampling ({resample_count_this_slot}/{max_resamples_per_edge_slot}).") # Optional: Reduce verbosity
                        # End of while loop

                        if not valid_edge_category_found_for_slot:
                            no_edge_category_model_idx = self.bond_dim - 1
                            cur_adj_features_one_hot[0, no_edge_category_model_idx, i, prev_node_idx] = 1.0
                            cur_adj_features_one_hot[0, no_edge_category_model_idx, prev_node_idx, i] = 1.0
                            # print(f"Debug: Node {i}, Edge ({prev_node_idx} -> {i}): Defaulted to NO_EDGE category for conditioning after failing to find a valid AIG edge.") # Optional: Reduce verbosity

                        # Increment edge_idx AFTER processing the current edge slot
                        edge_step_idx += 1
                    # End inner loop (for prev_node_idx)
            # End outer loop (for i)

            raw_node_features_output_one_hot = cur_node_features_one_hot.squeeze(0)
            return raw_node_features_output_one_hot, typed_edges_generated, actual_num_nodes

    def initialize_masks(self, max_node_unroll, max_edge_unroll):
        # --- DEBUG PRINT in initialize_masks ---
        print(f"initialize_masks: max_node_unroll={max_node_unroll}, max_edge_unroll={max_edge_unroll}")
        # --- End DEBUG ---
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (
                max_node_unroll - max_edge_unroll) * max_edge_unroll)
        num_mask_edge = int(num_masks - max_node_unroll)
        # --- DEBUG PRINT in initialize_masks ---
        print(f"initialize_masks: Calculated num_masks={num_masks}, num_mask_edge={num_mask_edge}")
        # --- End DEBUG ---

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

            edge_total_for_node_i = 0
            start_prev_node_for_edges = 0
            if i < max_edge_unroll:
                start_prev_node_for_edges = 0
                edge_total_for_node_i = i
            else:
                start_prev_node_for_edges = i - max_edge_unroll
                edge_total_for_node_i = max_edge_unroll

            for j in range(edge_total_for_node_i):
                prev_node_connected_idx = start_prev_node_for_edges + j
                node_masks2[cnt_edge][:i + 1] = 1

                if j == 0:
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node - 1].clone()
                    adj_masks2[cnt_edge][i,i] = 1
                else:
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge - 1].clone()
                    # Make edge decided in previous step visible (symmetric)
                    adj_masks2[cnt_edge][i, start_prev_node_for_edges + (j-1)] = 1
                    adj_masks2[cnt_edge][start_prev_node_for_edges + (j-1), i] = 1

                link_prediction_index[cnt_edge][0] = prev_node_connected_idx
                link_prediction_index[cnt_edge][1] = i
                cnt += 1
                cnt_edge += 1

        # --- DEBUG Assertions in initialize_masks ---
        assert cnt == num_masks, f'Masks count mismatch: total {cnt} vs expected {num_masks}'
        assert cnt_node == max_node_unroll, f'Node masks count mismatch: {cnt_node} vs expected {max_node_unroll}'
        assert cnt_edge == num_mask_edge, f'Edge masks count mismatch: {cnt_edge} vs expected {num_mask_edge}'
        print(f"initialize_masks: Assertions passed. Final cnt_edge={cnt_edge}")
        # --- End DEBUG ---

        for k_idx in range(num_mask_edge):
            u, v = link_prediction_index[k_idx, 0].item(), link_prediction_index[k_idx, 1].item()
            # Ensure indices are within bounds before assignment
            if 0 <= u < max_node_unroll and 0 <= v < max_node_unroll:
                 flow_core_edge_masks[u, v] = True
            else:
                 # This should ideally not happen if link_prediction_index is correct
                 print(f"WARNING (initialize_masks): Invalid indices ({u},{v}) from link_prediction_index at step {k_idx} while setting flow_core_edge_masks.")


        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks


    def dis_log_prob(self, z_tuple):
        z_node_transformed, z_edge_transformed = z_tuple
        if z_node_transformed.ndim == 2:
            z_node_transformed = z_node_transformed.view(-1, self.max_size, self.node_dim)
        if z_edge_transformed.ndim == 2:
            num_modeled_edges = self.edge_base_log_probs.shape[0]
            z_edge_transformed = z_edge_transformed.view(-1, num_modeled_edges, self.bond_dim)

        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        ll_node = torch.sum(z_node_transformed * node_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        ll_edge = torch.sum(z_edge_transformed * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        total_dims = self.latent_node_length + self.latent_edge_length
        if total_dims == 0: return torch.tensor(0.0, device=ll_node.device if hasattr(ll_node, 'device') else torch.device("cpu"), requires_grad=True)

        loss = -(torch.mean(ll_node + ll_edge) / total_dims)
        return loss
