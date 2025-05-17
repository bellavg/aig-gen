import torch
import torch.nn as nn

from .disgraphaf import DisGraphAF  # Assuming this is in the same directory
# Ensure all necessary imports from aig_config are present
from aig_config import (
    NUM2EDGETYPE,
    NUM2NODETYPE,
    NO_EDGE_CHANNEL,
    check_validity,
    to_directed_aig,
    remove_padding_nodes,
    display_graph_details  # For debugging the first failure
)
import networkx as nx
import numpy as np
import warnings


class GraphFlowModel(nn.Module):
    # Class attribute to track if detailed failure prints have occurred for the first graph
    _detailed_failure_printed_for_first_graph = False

    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict['edge_unroll']
        self.node_dim = model_conf_dict['node_dim']
        self.bond_dim = model_conf_dict['bond_dim']

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
            max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(
            0)
        self.latent_node_length = self.max_size * self.node_dim

        # Calculate the number of edge generation steps based on how initialize_masks works
        num_edge_steps = 0
        if self.max_size <= 0:  # Handle case with no nodes
            num_edge_steps = 0
        elif self.edge_unroll >= self.max_size - 1:  # If edge_unroll is large enough to connect to all previous nodes
            num_edge_steps = int((self.max_size - 1) * self.max_size / 2) if self.max_size > 0 else 0
        else:  # Standard case
            num_edge_steps = int(
                (self.edge_unroll - 1) * self.edge_unroll / 2 +  # Edges for the first 'max_edge_unroll' nodes
                (self.max_size - self.edge_unroll) * self.edge_unroll  # Edges for the remaining nodes
            )
            if self.edge_unroll == 0:  # if edge_unroll is 0
                num_edge_steps = 0
        self.latent_edge_length = num_edge_steps * self.bond_dim

        self.dp = model_conf_dict['use_gpu']

        node_base_log_probs = torch.randn(self.max_size, self.node_dim)
        # Ensure edge_base_log_probs has the correct size, even if num_edge_steps is 0
        if num_edge_steps > 0:
            edge_base_log_probs = torch.randn(num_edge_steps, self.bond_dim)
        else:
            edge_base_log_probs = torch.empty((0, self.bond_dim))  # Empty tensor if no edge steps

        self.flow_core = DisGraphAF(node_masks, adj_masks, link_prediction_index,
                                    num_flow_layer=model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                    num_node_type=self.node_dim, num_edge_type=self.bond_dim,
                                    num_rgcn_layer=model_conf_dict['num_rgcn_layer'],
                                    nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])

        current_device = torch.device("cuda" if self.dp and torch.cuda.is_available() else "cpu")

        if self.dp and not torch.cuda.is_available():
            warnings.warn("CUDA not available, running GraphDF model on CPU despite use_gpu=True.")
            self.dp = False  # Correct the flag

        self.node_base_log_probs = nn.Parameter(node_base_log_probs.to(current_device), requires_grad=True)
        self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.to(current_device), requires_grad=True)

        if self.dp:  # Apply DataParallel if dp is True and Cuda is available
            if current_device.type == 'cuda':
                self.flow_core = nn.DataParallel(self.flow_core)
            else:  # Should not happen if dp is true but cuda not avail due to warning above
                self.dp = False

    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (B, N, node_dim) - N is max_size
            inp_adj_features: (B, bond_dim, N, N) - N is max_size

        Returns:
            z: [(B, max_size, node_dim), (B, num_edge_steps, bond_dim)]
               Output from DisGraphAF, which are the transformed one-hot vectors.
        """
        inp_node_features_one_hot = inp_node_features.clone()

        selected_edges_list = []
        if self.flow_core_edge_masks.numel() > 0:  # Only select if there are edges to select
            for b in range(inp_adj_features.size(0)):
                batch_edges = inp_adj_features[b, :, self.flow_core_edge_masks]
                selected_edges_list.append(batch_edges)
            inp_adj_features_one_hot = torch.stack(selected_edges_list)
            inp_adj_features_one_hot = inp_adj_features_one_hot.permute(0, 2, 1).contiguous()
        else:  # No edges are modeled by flow_core_edge_masks (e.g. max_size is 0 or 1)
            # Create an empty tensor with the expected shape (B, 0, bond_dim)
            batch_size = inp_adj_features.size(0)
            inp_adj_features_one_hot = torch.empty(batch_size, 0, self.bond_dim,
                                                   device=inp_adj_features.device,
                                                   dtype=inp_adj_features.dtype)

        z = self.flow_core(inp_node_features, inp_adj_features,
                           inp_node_features_one_hot, inp_adj_features_one_hot)
        return z

    def generate(self, temperature=[0.3, 0.3], min_atoms=5, max_atoms=64, disconnection_patience=20):
        """
        Inverse flow to generate AIGs.
        """
        # This flag is checked before printing detailed failure info.
        # It's set to True by the first graph that *does* print such details.
        # current_graph_instance_prints_details_on_failure = not GraphFlowModel._detailed_failure_printed_for_first_graph

        disconnection_streak = 0
        current_device = self.node_base_log_probs.device

        with torch.no_grad():
            num2bond_map = NUM2EDGETYPE
            num2atom_map = NUM2NODETYPE

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)

            graph_backup = None
            aig = nx.Graph()

            is_continue = True
            edge_idx_for_sampling = 0
            total_resamples = 0

            model_to_call = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core

            for i in range(max_atoms):
                if not is_continue:
                    break

                if i < self.edge_unroll:
                    edge_total_to_consider = i
                    source_start_offset = 0
                else:
                    edge_total_to_consider = self.edge_unroll
                    source_start_offset = i - self.edge_unroll

                node_logits = self.node_base_log_probs[i].to(current_device)
                if temperature[0] <= 0:
                    latent_node_type_idx = torch.argmax(node_logits)
                    latent_node_sample = torch.nn.functional.one_hot(latent_node_type_idx,
                                                                     num_classes=self.node_dim).float().view(1, -1)
                else:
                    prior_node_dist = torch.distributions.OneHotCategorical(logits=node_logits / temperature[0])
                    latent_node_sample = prior_node_dist.sample().view(1, -1)

                resolved_node_type_latent = model_to_call.reverse(
                    cur_node_features, cur_adj_features, latent_node_sample, mode=0).view(-1)

                node_feature_id = torch.argmax(resolved_node_type_latent).item()
                cur_node_features[0, i, node_feature_id] = 1.0
                cur_adj_features[0, NO_EDGE_CHANNEL, i, i] = 1.0

                node_type_str = num2atom_map.get(node_feature_id, f"UNKNOWN_NODE_ID_{node_feature_id}")
                aig.add_node(i, type=node_type_str)

                if i == 0:
                    is_connect = True
                else:
                    is_connect = False

                for j_offset in range(edge_total_to_consider):
                    actual_source_node_idx = source_start_offset + j_offset

                    if actual_source_node_idx >= i or actual_source_node_idx < 0:
                        warnings.warn(
                            f"Invalid actual_source_node_idx {actual_source_node_idx} for target node {i}. Skipping edge.")
                        if edge_idx_for_sampling < self.edge_base_log_probs.shape[0]:  # Check bounds
                            edge_idx_for_sampling += 1
                        continue
                    if not aig.has_node(actual_source_node_idx):
                        warnings.warn(
                            f"Source node {actual_source_node_idx} does not exist in AIG when trying to connect to target {i}. Skipping edge.")
                        if edge_idx_for_sampling < self.edge_base_log_probs.shape[0]:  # Check bounds
                            edge_idx_for_sampling += 1
                        continue

                    if edge_idx_for_sampling >= self.edge_base_log_probs.shape[0]:
                        warnings.warn(
                            f"edge_idx_for_sampling {edge_idx_for_sampling} out of bounds for edge_base_log_probs size {self.edge_base_log_probs.shape[0]}. Stopping edge generation for this node.")
                        break  # Stop trying to add edges for this node i

                    valid_edge_found = False
                    resample_edge_count = 0
                    current_edge_logits = self.edge_base_log_probs[edge_idx_for_sampling].to(current_device).clone()

                    while not valid_edge_found:
                        if resample_edge_count > 100 or torch.isneginf(current_edge_logits).all():
                            edge_discrete_id = NO_EDGE_CHANNEL
                        else:
                            if temperature[1] <= 0:
                                latent_edge_idx = torch.argmax(current_edge_logits)
                                latent_edge_sample = torch.nn.functional.one_hot(latent_edge_idx,
                                                                                 num_classes=self.bond_dim).float().view(
                                    1, -1)
                            else:
                                prior_edge_dist = torch.distributions.OneHotCategorical(
                                    logits=current_edge_logits / temperature[1])
                                latent_edge_sample = prior_edge_dist.sample().view(1, -1)

                            resolved_edge_type_latent = model_to_call.reverse(
                                cur_node_features, cur_adj_features, latent_edge_sample,
                                mode=1,
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
                            aig.add_edge(i, actual_source_node_idx, type=current_edge_type_str)
                            is_structurally_valid = check_validity(aig)

                            if is_structurally_valid:
                                valid_edge_found = True
                                is_connect = True
                            else:
                                # --- Conditional Debug Print on First Ever Failure ---
                                if not GraphFlowModel._detailed_failure_printed_for_first_graph:
                                    print(
                                        f"\n--- [GraphDF First Ever Failing Edge Detail] Attempting edge ({actual_source_node_idx} -- {i}) type {current_edge_type_str} ---")
                                    display_graph_details(aig,
                                                          f"GraphDF state BEFORE check_validity for edge ({actual_source_node_idx}--{i}) that FAILED")
                                    print(
                                        f"--- [GraphDF First Ever Failing Edge Detail] check_validity FAILED for edge ({actual_source_node_idx} -- {i}) ---")
                                    # display_graph_details(aig, f"GraphDF state AFTER FAILED check_validity for edge ({actual_source_node_idx}--{i}) (before removal)") # Redundant if next print shows removal
                                    GraphFlowModel._detailed_failure_printed_for_first_graph = True  # Set flag immediately
                                # --- End Conditional Debug Print ---

                                aig.remove_edge(i, actual_source_node_idx)
                                cur_adj_features[0, edge_discrete_id, i, actual_source_node_idx] = 0.0
                                cur_adj_features[0, edge_discrete_id, actual_source_node_idx, i] = 0.0
                                total_resamples += 1.0
                                resample_edge_count += 1
                                current_edge_logits[edge_discrete_id] = -float('inf')

                                # if GraphFlowModel._detailed_failure_printed_for_first_graph and not printed_for_this_call: # This logic is now simpler
                                # display_graph_details(aig, f"GraphDF state AFTER FAILED check_validity AND edge ({actual_source_node_idx}--{i}) REMOVAL (First Failure Instance)")

                    edge_idx_for_sampling += 1

                if is_connect:
                    is_continue = True
                    graph_backup = aig.copy()
                    disconnection_streak = 0
                elif not is_connect and disconnection_streak < disconnection_patience:
                    is_continue = True
                    if graph_backup is None:
                        graph_backup = aig.copy()
                    disconnection_streak += 1
                else:
                    is_continue = False
                    if graph_backup is None:
                        graph_backup = aig.copy()

            final_graph_nx = aig
            if final_graph_nx is None or final_graph_nx.number_of_nodes() == 0:
                warnings.warn("Generated graph `aig` is empty or None at the end.")
                final_graph_nx = graph_backup if graph_backup is not None else nx.Graph()
                if final_graph_nx.number_of_nodes() == 0:
                    return None, 0, 0

            final_aig_directed = to_directed_aig(final_graph_nx)
            if final_aig_directed:
                final_aig_processed = remove_padding_nodes(final_aig_directed)
                if final_aig_processed is None:
                    final_aig_processed = final_aig_directed
            else:
                warnings.warn(f"to_directed_aig failed for generated graph.")
                final_aig_processed = final_graph_nx

            num_nodes_after_processing = final_aig_processed.number_of_nodes() if final_aig_processed else 0

            if num_nodes_after_processing < min_atoms:
                return None, total_resamples == 0, num_nodes_after_processing

            pure_valid_flag = 1.0 if total_resamples == 0 else 0.0
            return final_aig_processed, pure_valid_flag, num_nodes_after_processing

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=25):
        num_edge_steps = 0
        if max_node_unroll <= 0:
            num_edge_steps = 0
        elif max_edge_unroll >= max_node_unroll - 1:  # Corrected condition
            num_edge_steps = int((max_node_unroll - 1) * max_node_unroll / 2) if max_node_unroll > 0 else 0
        else:
            num_edge_steps = int(
                (max_edge_unroll - 1) * max_edge_unroll / 2 +
                (max_node_unroll - max_edge_unroll) * max_edge_unroll
            ) if max_edge_unroll > 0 else 0  # Ensure max_edge_unroll > 0 for first term
            if max_edge_unroll == 0:  # if edge_unroll is 0
                num_edge_steps = 0

        node_masks_for_node_step = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks_for_node_step = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()

        if num_edge_steps > 0:
            node_masks_for_edge_step = torch.zeros([num_edge_steps, max_node_unroll]).bool()
            adj_masks_for_edge_step = torch.zeros([num_edge_steps, max_node_unroll, max_node_unroll]).bool()
            link_prediction_index = torch.zeros([num_edge_steps, 2]).long()
        else:
            node_masks_for_edge_step = torch.empty([0, max_node_unroll]).bool()
            adj_masks_for_edge_step = torch.empty([0, max_node_unroll, max_node_unroll]).bool()
            link_prediction_index = torch.empty([0, 2]).long()

        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        current_node_step_idx = 0
        current_edge_step_idx = 0

        for i in range(max_node_unroll):
            if current_node_step_idx < max_node_unroll:
                node_masks_for_node_step[current_node_step_idx, :i] = 1
                adj_masks_for_node_step[current_node_step_idx, :i, :i] = 1
                current_node_step_idx += 1

            num_sources_to_connect_to = 0
            source_start_node_idx = 0
            if i == 0:
                num_sources_to_connect_to = 0
            elif i < max_edge_unroll:
                source_start_node_idx = 0
                num_sources_to_connect_to = i
            else:
                source_start_node_idx = i - max_edge_unroll
                num_sources_to_connect_to = max_edge_unroll

            current_adj_mask_for_edges = adj_masks_for_node_step[i, :,
                                         :].clone() if i < max_node_unroll else torch.zeros(max_node_unroll,
                                                                                            max_node_unroll).bool()
            current_adj_mask_for_edges[i, i] = True  # Node i itself is now known for conditioning edges

            for j_offset in range(num_sources_to_connect_to):
                actual_source_node = source_start_node_idx + j_offset
                if current_edge_step_idx < num_edge_steps:
                    node_masks_for_edge_step[current_edge_step_idx, :i + 1] = 1  # Nodes 0..i are visible

                    # Adjacency state for this edge step:
                    # Includes connections among 0..i-1 (from node step)
                    # AND connections from i to 0..actual_source_node-1 (already decided edges for node i)
                    adj_masks_for_edge_step[current_edge_step_idx, :,
                    :] = current_adj_mask_for_edges  # Start with adj state after node i type is known

                    link_prediction_index[current_edge_step_idx, 0] = actual_source_node
                    link_prediction_index[current_edge_step_idx, 1] = i
                    flow_core_edge_masks[i, actual_source_node] = 1  # (target, source)
                    current_edge_step_idx += 1

                    # Update current_adj_mask for the *next* edge step of this node i
                    current_adj_mask_for_edges[i, actual_source_node] = True
                    current_adj_mask_for_edges[actual_source_node, i] = True

        if not (current_node_step_idx == max_node_unroll):
            warnings.warn(f"Node mask count mismatch: expected {max_node_unroll}, got {current_node_step_idx}")
        if not (current_edge_step_idx == num_edge_steps):
            warnings.warn(
                f"Edge mask count mismatch: expected {num_edge_steps}, got {current_edge_step_idx}. This can happen if max_node_unroll is small (e.g., 0 or 1).")

        node_masks_all = torch.cat((node_masks_for_node_step, node_masks_for_edge_step), dim=0)
        adj_masks_all = torch.cat((adj_masks_for_node_step, adj_masks_for_edge_step), dim=0)

        node_masks_all = nn.Parameter(node_masks_all, requires_grad=False)
        adj_masks_all = nn.Parameter(adj_masks_all, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return node_masks_all, adj_masks_all, link_prediction_index, flow_core_edge_masks

    def dis_log_prob(self, z):
        x_output, adj_output = z
        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)
        ll_node = torch.sum(x_output * node_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs, dim=-1)
        if adj_output.numel() == 0 and self.edge_base_log_probs.numel() == 0:  # Handle case with no edges
            ll_edge = torch.zeros_like(
                ll_node)  # Or handle as appropriate (e.g. if batch size > 0, (B,) tensor of zeros)
        elif adj_output.shape[1] == 0 and self.edge_base_log_probs.shape[0] == 0:  # No edge steps
            ll_edge = torch.zeros_like(ll_node)
        elif adj_output.shape[1] != self.edge_base_log_probs.shape[0]:
            warnings.warn(
                f"Shape mismatch in dis_log_prob for edges: adj_output {adj_output.shape}, edge_base_log_probs {self.edge_base_log_probs.shape}")
            # Fallback or error, for now, let's assume if adj_output is empty, ll_edge is 0
            if adj_output.numel() == 0:
                ll_edge = torch.zeros_like(ll_node)
            else:  # This case should ideally not be hit if masks and forward pass are correct
                ll_edge = torch.tensor(float('-inf'), device=ll_node.device).repeat(
                    ll_node.shape[0])  # Penalize heavily
        else:
            ll_edge = torch.sum(adj_output * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        mean_ll = torch.mean(ll_node + ll_edge)

        num_node_variables = self.max_size
        num_edge_variables = self.edge_base_log_probs.shape[0]

        if (num_node_variables + num_edge_variables) == 0:
            return -torch.tensor(0.0, device=mean_ll.device) if mean_ll == 0 else -mean_ll

        return -(mean_ll / (num_node_variables + num_edge_variables))
