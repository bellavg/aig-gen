import numpy as np
import torch
import torch.nn as nn
from .graphaf import MaskedGraphAF  # Assuming .graphaf contains MaskedGraphAF
from aig_config import (
    NUM2EDGETYPE,
    NUM2NODETYPE,
    NO_EDGE_CHANNEL,  # Index for the "no edge" or "virtual edge" type
    check_validity,
    to_directed_aig,
    remove_padding_nodes
    # display_graph_details # Removed from imports as it's no longer called here
)
import networkx as nx
import warnings


class GraphFlowModel(nn.Module):
    # Class attribute to track if detailed failure prints have occurred (not used in this version)
    # _detailed_failure_printed_for_first_graph_af = False

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
            if self.edge_unroll == 0:
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
            else:
                self.dp = False

    def forward(self, inp_node_features, inp_adj_features):
        current_device = inp_node_features.device
        inp_node_features_cont = inp_node_features.clone()

        selected_edges_list = []
        for b in range(inp_adj_features.size(0)):
            batch_edges = inp_adj_features[b, :, self.flow_core_edge_masks]
            selected_edges_list.append(batch_edges)

        inp_adj_features_cont = torch.stack(selected_edges_list)
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()

        inp_node_features_cont += self.deq_coeff * torch.rand(inp_node_features_cont.size(), device=current_device)
        inp_adj_features_cont += self.deq_coeff * torch.rand(inp_adj_features_cont.size(), device=current_device)

        z, logdet = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        return z, logdet

    def generate(self, temperature=0.75, min_atoms=5, max_atoms=64, disconnection_patience=20):
        disconnection_streak = 0
        current_device = self.prior_ln_var.device

        with torch.no_grad():
            num2bond_map = NUM2EDGETYPE
            num2atom_map = NUM2NODETYPE

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)

            graph = nx.Graph()
            graph_backup_on_connect = None

            is_continue = True
            total_resamples = 0

            model_to_call = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core

            for i in range(max_atoms):
                if not is_continue:
                    break

                if i < self.edge_unroll:
                    num_sources_to_consider = i
                    source_node_start_index = 0
                else:
                    num_sources_to_consider = self.edge_unroll
                    source_node_start_index = i - self.edge_unroll

                prior_node_dist = torch.distributions.normal.Normal(
                    torch.zeros([self.node_dim], device=current_device),
                    temperature * torch.ones([self.node_dim], device=current_device)
                )
                latent_node_sample = prior_node_dist.sample().view(1, -1)

                resolved_node_type_latent = model_to_call.reverse(
                    cur_node_features, cur_adj_features, latent_node_sample, mode=0
                ).view(-1)

                node_feature_id = torch.argmax(resolved_node_type_latent).item()
                cur_node_features[0, i, node_feature_id] = 1.0
                cur_adj_features[0, NO_EDGE_CHANNEL, i, i] = 1.0

                node_type_str = num2atom_map.get(node_feature_id, f"UNKNOWN_NODE_ID_{node_feature_id}")
                graph.add_node(i, type=node_type_str)

                if i == 0:
                    is_connect = True
                else:
                    is_connect = False

                for j_offset in range(num_sources_to_consider):
                    actual_source_node_idx = source_node_start_index + j_offset

                    if actual_source_node_idx >= i or actual_source_node_idx < 0 or not graph.has_node(
                            actual_source_node_idx):
                        warnings.warn(f"Skipping edge attempt: Source {actual_source_node_idx} invalid for target {i}.")
                        continue

                    valid_edge_found = False
                    resample_edge_count = 0

                    while not valid_edge_found:
                        prior_edge_dist = torch.distributions.normal.Normal(
                            torch.zeros([self.bond_dim], device=current_device),
                            temperature * torch.ones([self.bond_dim], device=current_device)
                        )
                        latent_edge_sample = prior_edge_dist.sample().view(1, -1)

                        if resample_edge_count > 100:
                            edge_discrete_id = NO_EDGE_CHANNEL
                        else:
                            resolved_edge_type_latent = model_to_call.reverse(
                                cur_node_features, cur_adj_features, latent_edge_sample, mode=1,
                                edge_index=torch.tensor([[actual_source_node_idx, i]], device=current_device).long()
                            ).view(-1)
                            edge_discrete_id = torch.argmax(resolved_edge_type_latent).item()

                        cur_adj_features[0, edge_discrete_id, i, actual_source_node_idx] = 1.0
                        cur_adj_features[0, edge_discrete_id, actual_source_node_idx, i] = 1.0

                        current_edge_type_str = num2bond_map.get(edge_discrete_id,
                                                                 f"UNKNOWN_EDGE_ID_{edge_discrete_id}")

                        if edge_discrete_id == NO_EDGE_CHANNEL:
                            valid_edge_found = True
                        else:
                            graph.add_edge(i, actual_source_node_idx, type=current_edge_type_str)

                            is_structurally_valid = check_validity(graph)

                            if is_structurally_valid:
                                valid_edge_found = True
                                is_connect = True
                            else:
                                # No detailed display_graph_details calls here anymore
                                graph.remove_edge(i, actual_source_node_idx)

                                cur_adj_features[0, edge_discrete_id, i, actual_source_node_idx] = 0.0
                                cur_adj_features[0, edge_discrete_id, actual_source_node_idx, i] = 0.0

                                total_resamples += 1
                                resample_edge_count += 1

                if is_connect:
                    is_continue = True
                    graph_backup_on_connect = graph.copy()
                    disconnection_streak = 0
                elif not is_connect and disconnection_streak < disconnection_patience:
                    is_continue = True
                    if graph_backup_on_connect is None:
                        graph_backup_on_connect = graph.copy()
                    disconnection_streak += 1
                else:
                    is_continue = False
                    if graph_backup_on_connect is None:
                        graph_backup_on_connect = graph.copy()

            final_graph_nx = graph
            if graph_backup_on_connect is not None and not is_connect and disconnection_streak >= disconnection_patience:
                pass

            if final_graph_nx is None or final_graph_nx.number_of_nodes() == 0:
                warnings.warn("GraphAF: Generated graph is empty or None at the end.")
                return None, 0, 0

            final_aig_directed = to_directed_aig(final_graph_nx)
            if final_aig_directed:
                final_aig_processed = remove_padding_nodes(final_aig_directed)
                if final_aig_processed is None:
                    final_aig_processed = final_aig_directed
            else:
                warnings.warn("GraphAF: to_directed_aig failed for the generated graph.")
                final_aig_processed = final_graph_nx

            num_nodes_after_processing = final_aig_processed.number_of_nodes() if final_aig_processed else 0

            if num_nodes_after_processing < min_atoms:
                return None, (total_resamples == 0), num_nodes_after_processing

            pure_valid_flag = 1.0 if total_resamples == 0 else 0.0
            return final_aig_processed, pure_valid_flag, num_nodes_after_processing

    def initialize_masks(self, max_node_unroll=64, max_edge_unroll=25):
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
            if node_step_global_idx < max_node_unroll:
                node_masks_for_node_gen_phase[node_step_global_idx, :i] = True
                adj_masks_for_node_gen_phase[node_step_global_idx, :i, :i] = True
                node_step_global_idx += 1

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
            current_adj_mask_for_node_i_edges[i, i] = True

            for j_offset in range(num_sources_for_node_i):
                actual_source_node_idx = source_start_idx + j_offset

                if edge_step_global_idx < num_edge_steps:
                    node_masks_for_edge_gen_phase[edge_step_global_idx, :i + 1] = True
                    adj_masks_for_edge_gen_phase[edge_step_global_idx, :, :] = current_adj_mask_for_node_i_edges
                    link_prediction_index[edge_step_global_idx, 0] = actual_source_node_idx
                    link_prediction_index[edge_step_global_idx, 1] = i
                    flow_core_edge_masks[i, actual_source_node_idx] = True
                    edge_step_global_idx += 1
                    current_adj_mask_for_node_i_edges[i, actual_source_node_idx] = True
                    current_adj_mask_for_node_i_edges[actual_source_node_idx, i] = True

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
        ll_node = -0.5 * (
                    (z[0] ** 2) / torch.exp(self.prior_ln_var) + self.prior_ln_var + torch.log(2 * self.constant_pi))
        ll_node = ll_node.sum(-1)

        ll_edge = -0.5 * (
                    (z[1] ** 2) / torch.exp(self.prior_ln_var) + self.prior_ln_var + torch.log(2 * self.constant_pi))
        ll_edge = ll_edge.sum(-1)

        ll_node += logdet[0]
        ll_edge += logdet[1]

        total_log_likelihood = torch.mean(ll_node + ll_edge)

        norm_factor = self.latent_node_length + self.latent_edge_length
        if norm_factor == 0:
            return -torch.tensor(0.0,
                                 device=total_log_likelihood.device) if total_log_likelihood == 0 else -total_log_likelihood

        return -(total_log_likelihood / norm_factor)

    def dis_log_prob(self, z):
        raise NotImplementedError(
            "dis_log_prob is for discrete flows and not implemented for GraphAF (continuous flow).")

