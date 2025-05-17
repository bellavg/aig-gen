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
    display_graph_details  # Added for debugging
)
import networkx as nx
import numpy as np
import warnings


class GraphFlowModel(nn.Module):
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
        if self.edge_unroll >= self.max_size:  # handles cases where max_edge_unroll is effectively max_node_unroll or larger
            if self.max_size > 0:
                num_edge_steps = int((self.max_size - 1) * self.max_size / 2)
            else:
                num_edge_steps = 0
        else:
            num_edge_steps = int(
                (self.edge_unroll - 1) * self.edge_unroll / 2 +
                (self.max_size - self.edge_unroll) * self.edge_unroll
            )
        self.latent_edge_length = num_edge_steps * self.bond_dim

        self.dp = model_conf_dict['use_gpu']

        node_base_log_probs = torch.randn(self.max_size, self.node_dim)
        edge_base_log_probs = torch.randn(num_edge_steps, self.bond_dim)  # Corrected size

        self.flow_core = DisGraphAF(node_masks, adj_masks, link_prediction_index,
                                    num_flow_layer=model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                    num_node_type=self.node_dim, num_edge_type=self.bond_dim,
                                    num_rgcn_layer=model_conf_dict['num_rgcn_layer'],
                                    nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])
        if self.dp and torch.cuda.is_available():  # Check cuda availability
            self.flow_core = nn.DataParallel(self.flow_core)
            self.node_base_log_probs = nn.Parameter(node_base_log_probs.cuda(), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.cuda(), requires_grad=True)
        else:
            if self.dp and not torch.cuda.is_available():
                warnings.warn("CUDA not available, running GraphDF model on CPU despite use_gpu=True.")
                self.dp = False  # Correct the flag
            self.node_base_log_probs = nn.Parameter(node_base_log_probs, requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs, requires_grad=True)

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
        for b in range(inp_adj_features.size(0)):
            batch_edges = inp_adj_features[b, :, self.flow_core_edge_masks]
            selected_edges_list.append(batch_edges)

        inp_adj_features_one_hot = torch.stack(selected_edges_list)
        inp_adj_features_one_hot = inp_adj_features_one_hot.permute(0, 2, 1).contiguous()

        z = self.flow_core(inp_node_features, inp_adj_features,
                           inp_node_features_one_hot, inp_adj_features_one_hot)
        return z

    # Class attribute to track if detailed failure prints have occurred for the first graph
    _detailed_failure_printed_for_first_graph = False

    def generate(self, temperature=[0.3, 0.3], min_atoms=5, max_atoms=64, disconnection_patience=20):
        """
        Inverse flow to generate AIGs.
        """
        current_graph_instance_prints_details_on_failure = not GraphFlowModel._detailed_failure_printed_for_first_graph

        disconnection_streak = 0
        current_device = self.node_base_log_probs.device

        with torch.no_grad():
            num2bond_map = NUM2EDGETYPE
            num2atom_map = NUM2NODETYPE

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)

            # graph_backup now stores the latest version of 'aig' that was considered good (connected)
            # or just the current 'aig' if we are allowing disconnected parts to extend.
            graph_backup = None
            # We won't use node/adj_features_each_iter_backup for reverting features,
            # as 'aig' will be the single source of truth for graph structure being built.

            aig = nx.Graph()

            is_continue = True
            edge_idx_for_sampling = 0
            total_resamples = 0

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

                model_to_call = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core
                resolved_node_type_latent = model_to_call.reverse(
                    cur_node_features, cur_adj_features, latent_node_sample, mode=0).view(-1)

                node_feature_id = torch.argmax(resolved_node_type_latent).item()
                cur_node_features[0, i, node_feature_id] = 1.0  # Update feature matrix for model
                # Self-loops in adj_features usually marked as NO_EDGE or a special type.
                # This helps the model understand node i's type is now known.
                cur_adj_features[0, NO_EDGE_CHANNEL, i, i] = 1.0

                node_type_str = num2atom_map.get(node_feature_id, f"UNKNOWN_NODE_ID_{node_feature_id}")
                aig.add_node(i, type=node_type_str)  # Add node i to our NetworkX graph object

                if i == 0:
                    is_connect = True  # First node is considered "connected" to itself/start
                else:
                    is_connect = False  # Assume node i is not connected until an edge is formed

                for j_offset in range(edge_total_to_consider):
                    actual_source_node_idx = source_start_offset + j_offset

                    # Ensure actual_source_node_idx is valid and exists (should be < i)
                    if actual_source_node_idx >= i or actual_source_node_idx < 0:
                        # This case should not happen with correct loop bounds for j_offset
                        warnings.warn(
                            f"Invalid actual_source_node_idx {actual_source_node_idx} for target node {i}. Skipping edge.")
                        edge_idx_for_sampling += 1  # Still consume an edge logit
                        continue
                    if not aig.has_node(actual_source_node_idx):
                        # This would be very unusual if nodes are added sequentially 0 to i-1
                        warnings.warn(
                            f"Source node {actual_source_node_idx} does not exist in AIG when trying to connect to target {i}. Skipping edge.")
                        edge_idx_for_sampling += 1
                        continue

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

                        # Update current feature matrix (for model's next step)
                        cur_adj_features[0, edge_discrete_id, i, actual_source_node_idx] = 1.0
                        cur_adj_features[0, edge_discrete_id, actual_source_node_idx, i] = 1.0

                        current_edge_type_str = num2bond_map.get(edge_discrete_id,
                                                                 f"UNKNOWN_EDGE_ID_{edge_discrete_id}")

                        if edge_discrete_id == NO_EDGE_CHANNEL:
                            valid_edge_found = True
                        else:
                            # Add edge to NetworkX graph object
                            aig.add_edge(i, actual_source_node_idx, type=current_edge_type_str)

                            is_structurally_valid = check_validity(aig)  # Using the less strict check_validity

                            if is_structurally_valid:
                                valid_edge_found = True
                                is_connect = True  # An actual edge was formed connecting node i
                            else:
                                # --- Conditional Debug Print on Failure ---
                                if current_graph_instance_prints_details_on_failure:
                                    print(
                                        f"\n--- [First Failing Graph Detail] Attempting edge ({actual_source_node_idx} -- {i}) type {current_edge_type_str} ---")
                                    display_graph_details(aig,
                                                          f"Graph state BEFORE check_validity for edge ({actual_source_node_idx}--{i}) that FAILED")
                                    print(
                                        f"--- [First Failing Graph Detail] check_validity FAILED for edge ({actual_source_node_idx} -- {i}) ---")
                                # --- End Conditional Debug Print ---

                                aig.remove_edge(i, actual_source_node_idx)  # Remove from NetworkX graph

                                # Revert adj_features for this failed attempt
                                cur_adj_features[0, edge_discrete_id, i, actual_source_node_idx] = 0.0
                                cur_adj_features[0, edge_discrete_id, actual_source_node_idx, i] = 0.0
                                # Ensure NO_EDGE is marked for this pair if an explicit edge failed (optional, depends on model needs)
                                # cur_adj_features[0, NO_EDGE_CHANNEL, i, actual_source_node_idx] = 1.0
                                # cur_adj_features[0, NO_EDGE_CHANNEL, actual_source_node_idx, i] = 1.0

                                total_resamples += 1.0
                                resample_edge_count += 1
                                current_edge_logits[edge_discrete_id] = -float('inf')

                                if current_graph_instance_prints_details_on_failure:
                                    # display_graph_details(aig, f"Graph state AFTER FAILED check_validity AND edge ({actual_source_node_idx}--{i}) REMOVAL")
                                    GraphFlowModel._detailed_failure_printed_for_first_graph = True
                    edge_idx_for_sampling += 1

                    # After trying to connect node i to all its potential sources:
                if is_connect:
                    is_continue = True
                    # `aig` now contains node `i` and its successful connections.
                    # `graph_backup` will store this state as the last known "good" state.
                    graph_backup = aig.copy()
                    disconnection_streak = 0
                elif not is_connect and disconnection_streak < disconnection_patience:
                    # Node `i` was added but failed to connect to any previous nodes.
                    # We want to KEEP node `i` in `aig`.
                    is_continue = True  # Continue trying to add more nodes, hoping they connect to this segment.
                    # `aig` still contains node `i`.
                    # `graph_backup` remains the last state where a node successfully connected.
                    # If no node has connected yet (e.g. i=1, node 1 fails to connect to node 0),
                    # graph_backup might still be None or the state of node 0.
                    # We can choose to update graph_backup to include the disconnected 'i' if we want to keep it no matter what.
                    # For now, graph_backup holds the largest "successfully extended" graph.
                    # 'aig' holds the current attempt, including potentially isolated node 'i'.
                    if graph_backup is None:  # If this is the first node or no backup was ever made
                        graph_backup = aig.copy()  # Consider the current state (with isolated i) as a baseline

                    disconnection_streak += 1
                else:  # Node `i` is disconnected, and patience ran out.
                    is_continue = False  # Stop adding NEW nodes after this one.
                    # We KEEP node `i` in `aig`. `aig` is NOT reverted.
                    # `graph_backup` remains the last state where a node successfully connected.
                    # If no node ever connected, graph_backup might be the initial node or None.
                    # If we want the final graph to include this persistently disconnected node `i`:
                    # We could update graph_backup here if we decide this is the new "final" state.
                    # For now, `aig` holds this state.
                    if graph_backup is None:
                        graph_backup = aig.copy()

            # --- End of main loop for adding nodes ---

            # Determine the final graph to return
            # If we always want to return what was built in `aig`, even if disconnected at the end:
            final_graph_nx = aig
            if final_graph_nx is None or final_graph_nx.number_of_nodes() == 0:  # Should not be None if aig was used
                warnings.warn("Generated graph `aig` is empty or None at the end.")
                # Fallback if aig somehow became None; prefer graph_backup if it exists
                final_graph_nx = graph_backup if graph_backup is not None else nx.Graph()
                if final_graph_nx.number_of_nodes() == 0:
                    return None, 0, 0

            # Final processing: convert to directed, remove padding (if any was modeled)
            final_aig_directed = to_directed_aig(final_graph_nx)
            if final_aig_directed:
                final_aig_processed = remove_padding_nodes(final_aig_directed)
                if final_aig_processed is None:
                    final_aig_processed = final_aig_directed
            else:
                warnings.warn(f"to_directed_aig failed for generated graph.")
                final_aig_processed = final_graph_nx  # Fallback to the (likely undirected) graph

            num_nodes_after_processing = final_aig_processed.number_of_nodes() if final_aig_processed else 0

            if num_nodes_after_processing < min_atoms:
                return None, total_resamples == 0, num_nodes_after_processing

            pure_valid_flag = 1.0 if total_resamples == 0 else 0.0
            return final_aig_processed, pure_valid_flag, num_nodes_after_processing

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=25):
        """
        Initializes masks for the autoregressive generation process.
        Args:
            max_node_unroll (int): Maximum number of nodes in a graph.
            max_edge_unroll (int): Maximum number of previous nodes to consider for edge connections.
        Returns:
            tuple: Contains node_masks_all, adj_masks_all, link_prediction_index, flow_core_edge_masks.
        """
        num_edge_steps = 0
        if max_node_unroll <= 0:
            num_edge_steps = 0
        elif max_edge_unroll >= max_node_unroll - 1:
            num_edge_steps = int((max_node_unroll - 1) * max_node_unroll / 2) if max_node_unroll > 0 else 0
        else:
            num_edge_steps = int(
                (max_edge_unroll - 1) * max_edge_unroll / 2 +
                (max_node_unroll - max_edge_unroll) * max_edge_unroll
            )
            if max_edge_unroll == 0:
                num_edge_steps = (max_node_unroll - max_edge_unroll) * max_edge_unroll

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

            for j_offset in range(num_sources_to_connect_to):
                actual_source_node = source_start_node_idx + j_offset
                if current_edge_step_idx < num_edge_steps:
                    node_masks_for_edge_step[current_edge_step_idx, :i + 1] = 1
                    adj_masks_for_edge_step[current_edge_step_idx, :i, :i] = 1
                    adj_masks_for_edge_step[current_edge_step_idx, i, i] = 1

                    for k_prev_source_offset in range(j_offset):
                        prev_source_node = source_start_node_idx + k_prev_source_offset
                        adj_masks_for_edge_step[current_edge_step_idx, i, prev_source_node] = 1
                        adj_masks_for_edge_step[current_edge_step_idx, prev_source_node, i] = 1

                    link_prediction_index[current_edge_step_idx, 0] = actual_source_node
                    link_prediction_index[current_edge_step_idx, 1] = i
                    flow_core_edge_masks[i, actual_source_node] = 1
                    current_edge_step_idx += 1

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
        ll_edge = torch.sum(adj_output * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        mean_ll = torch.mean(ll_node + ll_edge)

        num_node_variables = self.max_size
        num_edge_variables = self.edge_base_log_probs.shape[0]

        if (num_node_variables + num_edge_variables) == 0:
            return -torch.tensor(0.0, device=mean_ll.device) if mean_ll == 0 else -mean_ll

        return -(mean_ll / (num_node_variables + num_edge_variables))
