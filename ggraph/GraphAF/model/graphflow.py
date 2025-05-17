import numpy as np
import torch
import torch.nn as nn
from .graphaf import MaskedGraphAF  # Assuming .graphaf contains MaskedGraphAF
from aig_config import (
    NUM2EDGETYPE,
    NUM2NODETYPE,
    NO_EDGE_CHANNEL,  # Index for the "no edge" or "virtual edge" type
    check_validity,  # General validity check, can still be used for final validation if needed
    check_interim_validity,  # Import for the specific interim check
    to_directed_aig,
    remove_padding_nodes,
    display_graph_details  # For debugging if needed
)
import networkx as nx
import warnings


class GraphFlowModel(nn.Module):
    # Class attribute to track if detailed failure prints have occurred for the first graph
    _detailed_failure_printed_for_first_graph_af = False  # Specific to GraphAF

    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict['edge_unroll']
        self.node_dim = model_conf_dict['node_dim']
        self.bond_dim = model_conf_dict['bond_dim']
        self.deq_coeff = model_conf_dict['deq_coeff']

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
            max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(0)
        self.latent_node_length = self.max_size * self.node_dim

        num_edge_steps = 0
        if self.max_size <= 0:
            num_edge_steps = 0
        elif self.edge_unroll >= self.max_size - 1:
            num_edge_steps = int((self.max_size - 1) * self.max_size / 2) if self.max_size > 0 else 0
        else:
            num_edge_steps = int(
                (self.edge_unroll - 1) * self.edge_unroll / 2 +
                (self.max_size - self.edge_unroll) * self.edge_unroll
            )
            if self.edge_unroll == 0:  # Ensure no negative or NaN if edge_unroll is 0
                num_edge_steps = 0

        self.latent_edge_length = num_edge_steps * self.bond_dim
        self.dp = model_conf_dict['use_gpu']

        constant_pi = torch.Tensor([3.1415926535])
        prior_ln_var = torch.zeros([1])

        self.flow_core = MaskedGraphAF(node_masks, adj_masks, link_prediction_index, st_type=model_conf_dict['st_type'],
                                       num_flow_layer=model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                       num_node_type=self.node_dim, num_edge_type=self.bond_dim,
                                       num_rgcn_layer=model_conf_dict['num_rgcn_layer'], nhid=model_conf_dict['nhid'],
                                       nout=model_conf_dict['nout'])

        current_device = torch.device("cuda" if self.dp and torch.cuda.is_available() else "cpu")
        if self.dp and not torch.cuda.is_available():
            warnings.warn("CUDA not available, running GraphAF model on CPU despite use_gpu=True.")
            self.dp = False

        self.constant_pi = nn.Parameter(constant_pi.to(current_device), requires_grad=False)
        self.prior_ln_var = nn.Parameter(prior_ln_var.to(current_device), requires_grad=False)

        if self.dp:
            if current_device.type == 'cuda':
                self.flow_core = nn.DataParallel(self.flow_core)
            else:  # Should not happen if dp is true but cuda not avail due to warning above
                self.dp = False

    def forward(self, inp_node_features, inp_adj_features):
        current_device = inp_node_features.device  # Get device from input tensor
        inp_node_features_cont = inp_node_features.clone()

        selected_edges_list = []
        if self.flow_core_edge_masks.numel() > 0:  # Check if mask is not empty
            for b in range(inp_adj_features.size(0)):
                batch_edges = inp_adj_features[b, :, self.flow_core_edge_masks]
                selected_edges_list.append(batch_edges)

            if selected_edges_list:  # Ensure list is not empty before stacking
                inp_adj_features_cont = torch.stack(selected_edges_list)
                inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()
            else:  # Fallback if mask is all False or leads to empty selection
                batch_size = inp_adj_features.size(0)
                num_edge_steps = self.latent_edge_length // self.bond_dim if self.bond_dim > 0 else 0
                inp_adj_features_cont = torch.empty(batch_size, num_edge_steps, self.bond_dim,
                                                    device=current_device,
                                                    dtype=inp_adj_features.dtype)
        else:  # No edges are modeled by flow_core_edge_masks
            batch_size = inp_adj_features.size(0)
            # Calculate num_edge_steps based on latent_edge_length and bond_dim
            num_edge_steps = self.latent_edge_length // self.bond_dim if self.bond_dim > 0 else 0
            inp_adj_features_cont = torch.empty(batch_size, num_edge_steps, self.bond_dim,
                                                device=current_device,
                                                dtype=inp_adj_features.dtype)

        inp_node_features_cont += self.deq_coeff * torch.rand(inp_node_features_cont.size(), device=current_device)
        inp_adj_features_cont += self.deq_coeff * torch.rand(inp_adj_features_cont.size(), device=current_device)

        z, logdet = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        return z, logdet

    def generate(self, temperature=0.75, min_atoms=5, max_atoms=64, disconnection_patience=20):
        disconnection_streak = 0
        current_device = self.prior_ln_var.device  # Use device of a model parameter

        with torch.no_grad():
            num2edge_type_map = NUM2EDGETYPE
            num2node_type_map = NUM2NODETYPE

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)

            # Using NetworkX Graph for building the AIG structure
            graph_representation = nx.Graph()
            graph_backup_on_connect = None  # Backup of the graph state when a node connects

            is_continue_generation = True
            total_resamples_count = 0  # Renamed for clarity

            model_to_call = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core

            for i in range(max_atoms):  # Iterate for each potential node
                if not is_continue_generation:
                    break

                # Determine how many previous nodes to consider connecting to
                if i < self.edge_unroll:
                    num_sources_to_consider = i
                    source_node_start_index = 0
                else:
                    num_sources_to_consider = self.edge_unroll
                    source_node_start_index = i - self.edge_unroll

                # 1. Generate Node Type (using Normal distribution for GraphAF)
                prior_node_dist = torch.distributions.normal.Normal(
                    torch.zeros([self.node_dim], device=current_device),
                    temperature * torch.ones([self.node_dim], device=current_device)
                )
                latent_node_sample = prior_node_dist.sample().view(1, -1)

                # Reverse flow for node type
                resolved_node_type_latent = model_to_call.reverse(
                    cur_node_features, cur_adj_features, latent_node_sample, mode=0
                ).view(-1)

                node_feature_id = torch.argmax(resolved_node_type_latent).item()

                # Update node features and add node to graph
                cur_node_features[0, i, node_feature_id] = 1.0
                # For self-loop, use NO_EDGE_CHANNEL as per AIG convention for GraphAF
                cur_adj_features[0, NO_EDGE_CHANNEL, i, i] = 1.0

                node_type_str = num2node_type_map.get(node_feature_id, f"UNKNOWN_NODE_ID_{node_feature_id}")
                graph_representation.add_node(i, type=node_type_str)

                current_node_is_connected = (i == 0)

                # 2. Generate Edges to Previous Nodes
                for j_offset in range(num_sources_to_consider):
                    actual_source_node_idx = source_node_start_index + j_offset

                    if actual_source_node_idx >= i or actual_source_node_idx < 0 or not graph_representation.has_node(
                            actual_source_node_idx):
                        warnings.warn(
                            f"GraphAF: Skipping edge attempt: Source {actual_source_node_idx} invalid for target {i}.")
                        continue

                    edge_successfully_placed = False
                    resample_edge_attempts = 0
                    # invalid_edge_types_tried = set() # Not used in GraphAF's typical resampling

                    while not edge_successfully_placed:
                        # Sample edge type from Normal distribution
                        prior_edge_dist = torch.distributions.normal.Normal(
                            torch.zeros([self.bond_dim], device=current_device),
                            temperature * torch.ones([self.bond_dim], device=current_device)
                        )
                        latent_edge_sample = prior_edge_dist.sample().view(1, -1)

                        # Force NO_EDGE_CHANNEL if too many resamples
                        if resample_edge_attempts > 100:  # Resample limit
                            chosen_edge_type_id = NO_EDGE_CHANNEL
                        else:
                            # Reverse flow for edge type
                            resolved_edge_type_latent = model_to_call.reverse(
                                cur_node_features, cur_adj_features, latent_edge_sample, mode=1,
                                edge_index=torch.tensor([[actual_source_node_idx, i]], device=current_device).long()
                            ).view(-1)
                            chosen_edge_type_id = torch.argmax(resolved_edge_type_latent).item()

                        # Update adjacency features (tentatively)
                        cur_adj_features[0, chosen_edge_type_id, i, actual_source_node_idx] = 1.0
                        cur_adj_features[0, chosen_edge_type_id, actual_source_node_idx, i] = 1.0

                        current_edge_type_str = num2edge_type_map.get(chosen_edge_type_id,
                                                                      f"UNKNOWN_EDGE_ID_{chosen_edge_type_id}")

                        if chosen_edge_type_id == NO_EDGE_CHANNEL:
                            edge_successfully_placed = True
                        else:
                            graph_representation.add_edge(i, actual_source_node_idx, type=current_edge_type_str)

                            # Use check_interim_validity for AIGs
                            is_structurally_valid = check_interim_validity(graph_representation, actual_source_node_idx,
                                                                           i)

                            if is_structurally_valid:
                                edge_successfully_placed = True
                                current_node_is_connected = True
                            else:
                                # Backtrack
                                if not GraphFlowModel._detailed_failure_printed_for_first_graph_af:  # Use AF specific flag
                                    print(
                                        f"\n--- [GraphAF First Ever Failing Edge Detail] Attempting edge ({actual_source_node_idx} -- {i}) type {current_edge_type_str} ---")
                                    display_graph_details(graph_representation,
                                                          f"GraphAF state BEFORE check_interim_validity for edge ({actual_source_node_idx}--{i}) that FAILED")
                                    print(
                                        f"--- [GraphAF First Ever Failing Edge Detail] check_interim_validity FAILED for edge ({actual_source_node_idx} -- {i}) ---")
                                    GraphFlowModel._detailed_failure_printed_for_first_graph_af = True

                                graph_representation.remove_edge(i, actual_source_node_idx)
                                cur_adj_features[0, chosen_edge_type_id, i, actual_source_node_idx] = 0.0
                                cur_adj_features[0, chosen_edge_type_id, actual_source_node_idx, i] = 0.0
                                total_resamples_count += 1
                                resample_edge_attempts += 1
                                # GraphAF typically does not modify the prior logits for resampling the same edge slot.
                                # It relies on fresh samples from the Normal distribution.

                # After processing all potential edges for node i
                if current_node_is_connected:
                    is_continue_generation = True
                    graph_backup_on_connect = graph_representation.copy()
                    # No need to backup cur_node_features/cur_adj_features here as they are built incrementally
                    # and restored only if generation fully stops due to disconnection.
                    disconnection_streak = 0
                elif not current_node_is_connected and disconnection_streak < disconnection_patience:
                    is_continue_generation = True
                    if graph_backup_on_connect is None:  # Should only happen if i=0 was somehow disconnected
                        graph_backup_on_connect = graph_representation.copy()
                    disconnection_streak += 1
                else:  # Node i is not connected AND disconnection patience exceeded
                    is_continue_generation = False
                    if graph_backup_on_connect is not None:
                        graph_representation = graph_backup_on_connect.copy()
                        # Note: cur_node_features and cur_adj_features might be out of sync with the restored graph_representation here.
                        # This might be acceptable if generation stops, as only graph_representation is used.
                        # For strict consistency, one might need to rebuild features or store feature backups.
                        # However, the original GraphAF also doesn't explicitly restore feature tensors here.

            # Final graph processing
            final_graph_nx = graph_representation
            # If generation stopped due to disconnection and there was a backup, it's already in final_graph_nx.
            # If the loop completed, final_graph_nx is the result.

            if final_graph_nx is None or final_graph_nx.number_of_nodes() == 0:
                warnings.warn("GraphAF: Generated graph is empty or None at the end.")
                final_graph_nx = graph_backup_on_connect if graph_backup_on_connect is not None else nx.Graph()
                if final_graph_nx.number_of_nodes() == 0:
                    return None, 0, 0

            final_aig_directed = to_directed_aig(final_graph_nx)
            if final_aig_directed:
                final_aig_processed = remove_padding_nodes(final_aig_directed)
                if final_aig_processed is None:  # If padding removal results in None (e.g. empty graph)
                    final_aig_processed = final_aig_directed
            else:
                warnings.warn("GraphAF: to_directed_aig failed for the generated graph.")
                final_aig_processed = final_graph_nx  # Fallback

            num_nodes_after_processing = final_aig_processed.number_of_nodes() if final_aig_processed else 0

            if num_nodes_after_processing < min_atoms:
                return None, (total_resamples_count == 0), num_nodes_after_processing

            pure_valid_flag = 1.0 if total_resamples_count == 0 else 0.0
            return final_aig_processed, pure_valid_flag, num_nodes_after_processing

    def initialize_masks(self, max_node_unroll=64, max_edge_unroll=25):
        # This mask initialization logic appears consistent with GraphDF and general flow-based autoregressive models
        num_edge_steps = 0
        if max_node_unroll <= 0:
            num_edge_steps = 0
        elif max_edge_unroll >= max_node_unroll - 1:
            num_edge_steps = int((max_node_unroll - 1) * max_node_unroll / 2) if max_node_unroll > 0 else 0
        else:
            num_edge_steps = int(
                (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (max_edge_unroll)
            ) if max_edge_unroll > 0 else 0
            if max_edge_unroll == 0: num_edge_steps = 0

        node_masks_for_node_gen_phase = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks_for_node_gen_phase = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()

        if num_edge_steps > 0:
            node_masks_for_edge_gen_phase = torch.zeros([num_edge_steps, max_node_unroll]).bool()
            adj_masks_for_edge_gen_phase = torch.zeros([num_edge_steps, max_node_unroll, max_node_unroll]).bool()
            link_prediction_index = torch.zeros([num_edge_steps, 2]).long()
        else:
            node_masks_for_edge_gen_phase = torch.empty(0, max_node_unroll).bool()
            adj_masks_for_edge_gen_phase = torch.empty(0, max_node_unroll, max_node_unroll).bool()
            link_prediction_index = torch.empty(0, 2).long()

        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        node_step_global_idx = 0
        edge_step_global_idx = 0

        for i in range(max_node_unroll):
            # Node generation step mask for node i
            if node_step_global_idx < max_node_unroll:
                node_masks_for_node_gen_phase[node_step_global_idx, :i] = True
                adj_masks_for_node_gen_phase[node_step_global_idx, :i, :i] = True
                node_step_global_idx += 1

            # Edge generation step masks for edges connecting to node i
            num_sources_for_node_i = 0
            source_start_idx = 0
            if i == 0:
                num_sources_for_node_i = 0
            elif i < max_edge_unroll:
                source_start_idx = 0
                num_sources_for_node_i = i
            else:
                source_start_idx = i - max_edge_unroll
                num_sources_for_node_i = max_edge_unroll

            current_adj_mask_for_node_i_edges = adj_masks_for_node_gen_phase[i, :,
                                                :].clone() if i < max_node_unroll else torch.zeros(max_node_unroll,
                                                                                                   max_node_unroll).bool()
            current_adj_mask_for_node_i_edges[i, i] = True  # Self-loop info is part of conditioning context

            for j_offset in range(num_sources_for_node_i):
                actual_source_node_idx = source_start_idx + j_offset

                if edge_step_global_idx < num_edge_steps:
                    node_masks_for_edge_gen_phase[edge_step_global_idx, :i + 1] = True  # Nodes 0..i are visible
                    adj_masks_for_edge_gen_phase[edge_step_global_idx, :, :] = current_adj_mask_for_node_i_edges

                    link_prediction_index[edge_step_global_idx, 0] = actual_source_node_idx
                    link_prediction_index[edge_step_global_idx, 1] = i
                    flow_core_edge_masks[i, actual_source_node_idx] = True  # Mark this edge for flow_core processing

                    edge_step_global_idx += 1

                    # Update conditioning mask for the *next* edge prediction for node i
                    current_adj_mask_for_node_i_edges[i, actual_source_node_idx] = True
                    current_adj_mask_for_node_i_edges[actual_source_node_idx, i] = True
                # else: (Warning for exceeding num_edge_steps can be added if necessary)

        if not (node_step_global_idx == max_node_unroll):
            warnings.warn(
                f"GraphAF Mask Init: Node step count mismatch. Expected {max_node_unroll}, got {node_step_global_idx}")
        if not (edge_step_global_idx == num_edge_steps):
            warnings.warn(
                f"GraphAF Mask Init: Edge step count mismatch. Expected {num_edge_steps}, got {edge_step_global_idx}. This can happen if max_node_unroll is small.")

        final_node_masks = torch.cat((node_masks_for_node_gen_phase, node_masks_for_edge_gen_phase), dim=0)
        final_adj_masks = torch.cat((adj_masks_for_node_gen_phase, adj_masks_for_edge_gen_phase), dim=0)

        final_node_masks = nn.Parameter(final_node_masks, requires_grad=False)
        final_adj_masks = nn.Parameter(final_adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return final_node_masks, final_adj_masks, link_prediction_index, flow_core_edge_masks

    def log_prob(self, z, logdet):
        # Dequantization adjustment to log determinant (subtracts log(deq_coeff) for each variable)
        # Number of node variables = self.max_size * self.node_dim
        # Number of edge variables = num_edge_steps * self.bond_dim
        # (where num_edge_steps is self.latent_edge_length / self.bond_dim)

        deq_adj_nodes = self.latent_node_length * np.log(self.deq_coeff)
        deq_adj_edges = self.latent_edge_length * np.log(self.deq_coeff)

        logdet[0] = logdet[0] - deq_adj_nodes
        logdet[1] = logdet[1] - deq_adj_edges

        # Log probability under N(0, temp^2 * I) prior (where temp^2 = exp(prior_ln_var))
        ll_node = -0.5 * (
                (z[0] ** 2) / torch.exp(self.prior_ln_var) + self.prior_ln_var + torch.log(2 * self.constant_pi))
        ll_node = ll_node.sum(-1)  # Sum over feature dimensions for each sample in batch (B)

        ll_edge = -0.5 * (
                (z[1] ** 2) / torch.exp(self.prior_ln_var) + self.prior_ln_var + torch.log(2 * self.constant_pi))
        ll_edge = ll_edge.sum(-1)  # Sum over feature dimensions for each sample in batch (B)

        ll_node += logdet[0]  # Add log determinant for node transformations
        ll_edge += logdet[1]  # Add log determinant for edge transformations

        total_log_likelihood = torch.mean(ll_node + ll_edge)  # Mean over batch

        # Normalize by total number of latent variables (nodes + edges)
        # This normalization is by the total number of *dimensions* in the latent space
        norm_factor = self.latent_node_length + self.latent_edge_length
        if norm_factor == 0:  # Avoid division by zero if max_size is 0 or no edges
            return -torch.tensor(0.0,
                                 device=total_log_likelihood.device) if total_log_likelihood == 0 else -total_log_likelihood

        # Return negative average log-likelihood (bits per dimension)
        return -(total_log_likelihood / norm_factor)

    def dis_log_prob(self, z):
        # This method is for discrete flows (like GraphDF) and should not be called by GraphAF.
        raise NotImplementedError(
            "dis_log_prob is for discrete flows and not implemented for GraphAF (continuous flow).")
