import torch
import torch.nn as nn

from .disgraphaf import DisGraphAF  # Assuming this is in the same directory
from aig_config import *  # Make sure Fan_ins, NUM2NODETYPE, NUM2EDGETYPE, VIRTUAL_EDGE_INDEX, NUM_EXPLICIT_EDGE_TYPES, check_validity are here
import networkx as nx
import numpy as np
import warnings


class GraphFlowModel(nn.Module):
    def __init__(self, model_conf_dict):
        super(GraphFlowModel, self).__init__()
        self.max_size = model_conf_dict['max_size']
        self.edge_unroll = model_conf_dict['edge_unroll']
        self.node_dim = model_conf_dict['node_dim']
        self.bond_dim = model_conf_dict['bond_dim']  # This should be NUM_ADJ_CHANNELS

        node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
            max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = node_masks.size(
            0)
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim

        self.dp = model_conf_dict['use_gpu']

        node_base_log_probs = torch.randn(self.max_size, self.node_dim)
        # The size of edge_base_log_probs should correspond to the total number of edge sampling steps
        # This is related to num_mask_edge in initialize_masks
        num_mask_edge = int(
            self.max_size + (self.edge_unroll - 1) * self.edge_unroll / 2 + (self.max_size - self.edge_unroll) * (
                self.edge_unroll)) - self.max_size

        edge_base_log_probs = torch.randn(num_mask_edge, self.bond_dim)

        self.flow_core = DisGraphAF(node_masks, adj_masks, link_prediction_index,
                                    num_flow_layer=model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                    num_node_type=self.node_dim, num_edge_type=self.bond_dim,
                                    num_rgcn_layer=model_conf_dict['num_rgcn_layer'],
                                    nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)
            self.node_base_log_probs = nn.Parameter(node_base_log_probs.cuda(), requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.cuda(), requires_grad=True)
        else:
            self.node_base_log_probs = nn.Parameter(node_base_log_probs, requires_grad=True)
            self.edge_base_log_probs = nn.Parameter(edge_base_log_probs, requires_grad=True)

    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (B, N, node_dim)
            inp_adj_features: (B, bond_dim, N, N)

        Returns:
            z: [(B, node_num*node_dim), (B, edge_num*bond_dim)]
        """
        inp_node_features_cont = inp_node_features.clone()

        # flow_core_edge_masks is (max_size, max_size) bool
        # inp_adj_features is (B, bond_dim, N, N)
        # We need to select edges based on the mask. The original code implies
        # that flow_core_edge_masks identifies which (target, source) pairs are considered.
        # The permutation then makes it (B, edge_num, bond_dim).
        # This part seems to assume inp_adj_features is (B, N, N, bond_dim) or similar before selection.
        # Let's assume the original selection logic is correct for how flow_core_edge_masks is structured.
        # If inp_adj_features is (B, bond_dim, N, N), then:
        # inp_adj_features_perm = inp_adj_features.permute(0, 2, 3, 1) # (B, N, N, bond_dim)
        # inp_adj_features_cont = inp_adj_features_perm[:, self.flow_core_edge_masks].clone() # (B, edge_num, bond_dim)
        # This is a common way to handle it. The original code did:
        # inp_adj_features_cont = inp_adj_features[:, :, self.flow_core_edge_masks].clone() # (B, bond_dim, edge_num)
        # inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()  # (B, edge_num, bond_dim)
        # This implies flow_core_edge_masks is 1D or applied in a way that results in edge_num dimension.
        # Given flow_core_edge_masks is (max_size, max_size), the selection is likely:
        selected_edges_list = []
        for b in range(inp_adj_features.size(0)):
            # Select the upper triangle of edges defined by flow_core_edge_masks
            # self.flow_core_edge_masks is (max_node_unroll, max_node_unroll)
            # It's 1 for edges (target_idx, source_idx) that are considered in the autoregressive order.
            # For each target node `i`, it considers sources `start` to `end`.
            # We need to collect these edges.
            # The number of true values in self.flow_core_edge_masks is num_mask_edge.
            batch_edges = inp_adj_features[b, :, self.flow_core_edge_masks]  # (bond_dim, num_mask_edge)
            selected_edges_list.append(batch_edges)
        inp_adj_features_cont = torch.stack(selected_edges_list)  # (B, bond_dim, num_mask_edge)
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()  # (B, num_mask_edge, bond_dim)

        z = self.flow_core(inp_node_features, inp_adj_features, inp_node_features_cont, inp_adj_features_cont)
        return z

    def generate(self, temperature=[0.3, 0.3], min_atoms=5, max_atoms=64, disconnection_patience=20):
        """
        Inverse flow to generate AIGs.
        Attempts to satisfy in-degree for AND (2) and PO (1) nodes for the current node i.
        """
        disconnection_streak = 0
        current_device = torch.device("cuda" if self.dp and torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            num2bond_map = NUM2EDGETYPE  # e.g., {0: "EDGE_REG", 1: "EDGE_INV"}
            num2atom_map = NUM2NODETYPE  # e.g., {0: "NODE_CONST0", ..., 3: "NODE_PO"}

            # Initialize tensors for current graph features
            cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)

            # Backups for graph state if generation needs to revert due to disconnection
            node_features_each_iter_backup = cur_node_features.clone()
            adj_features_each_iter_backup = cur_adj_features.clone()

            aig = nx.DiGraph()  # The graph being built
            graph_backup_candidate = None  # Stores the last "good" AIG state

            is_generation_continue = True
            edge_idx_counter = 0  # Index for self.edge_base_log_probs
            total_resamples_count = 0

            # Main loop: Generate one node at a time
            for i in range(max_atoms):  # i is the current node index (target node)
                if not is_generation_continue:
                    break

                # Determine the range of previous nodes to connect to the current node i
                if i < self.edge_unroll:
                    num_potential_sources_to_try = i
                    source_node_idx_start = 0
                else:
                    num_potential_sources_to_try = self.edge_unroll
                    source_node_idx_start = i - self.edge_unroll

                # 1. Generate Node Type for current node i
                prior_node_dist = torch.distributions.OneHotCategorical(
                    logits=self.node_base_log_probs[i].to(current_device) * temperature[0])
                latent_node_type_sample = prior_node_dist.sample().view(1, -1)

                if self.dp:
                    resolved_node_type_latent = self.flow_core.module.reverse(
                        cur_node_features, cur_adj_features, latent_node_type_sample, mode=0).view(-1)
                else:
                    resolved_node_type_latent = self.flow_core.reverse(
                        cur_node_features, cur_adj_features, latent_node_type_sample, mode=0).view(-1)

                node_feature_id = torch.argmax(resolved_node_type_latent).item()
                cur_node_features[0, i, node_feature_id] = 1.0
                # Adjacency matrix for self-loops (usually not actual edges in AIGs, but might be for model conditioning)
                # If self.bond_dim includes a "no edge" or "self-loop" type, this needs care.
                # Assuming diagonal of adjacency is for model state, not literal self-loops.
                # The original code sets all bond_dim channels to 1.0 for (i,i).
                # This might be specific to how the model interprets it.
                # If bond_dim has a specific "no edge" or "padding" channel, only that should be set.
                # For now, replicating original:
                cur_adj_features[0, :, i, i] = 1.0  # This might need adjustment based on bond_dim meaning

                node_type_str = num2atom_map[node_feature_id]
                aig.add_node(i, type=node_type_str)

                # 2. Generate Edges to current node i from previous nodes
                actual_edges_added_to_node_i = 0
                is_node_i_connected = (i == 0)  # Node 0 is connected by definition (no prior nodes)

                # Determine desired in-degree for node i based on its type
                # Fan_ins should be like: {"NODE_AND": 2, "NODE_PO": 1, "NODE_PI": 0, ...}
                desired_in_degree_for_node_i = Fan_ins.get(node_type_str, 0)

                potential_source_indices = list(
                    range(source_node_idx_start, source_node_idx_start + num_potential_sources_to_try))
                # np.random.shuffle(potential_source_indices) # Optional: if order of trying sources matters

                for k_source_attempt in range(num_potential_sources_to_try):
                    source_prev_node_idx = potential_source_indices[k_source_attempt]

                    # If node 'i' has a specific in-degree target and it's already met,
                    # subsequent "attempts" for this node 'i' from other sources are forced to be virtual.
                    if desired_in_degree_for_node_i > 0 and actual_edges_added_to_node_i >= desired_in_degree_for_node_i:
                        chosen_edge_discrete_id = VIRTUAL_EDGE_INDEX
                        # Update cur_adj_features for the flow core. AIG is not changed for virtual.
                        cur_adj_features[0, chosen_edge_discrete_id, source_prev_node_idx, i] = 1.0
                        # valid_edge_found_for_this_source = True # Not strictly needed here
                    else:
                        # Attempt to sample a real or virtual edge
                        valid_edge_found_for_this_source = False
                        resample_edge_count = 0

                        if edge_idx_counter >= self.edge_base_log_probs.shape[0]:
                            warnings.warn(
                                f"edge_idx_counter {edge_idx_counter} is out of bounds for edge_base_log_probs (shape {self.edge_base_log_probs.shape[0]}). Forcing virtual edge.")
                            edge_log_probs_for_current_pair = None  # Cannot sample
                        else:
                            edge_log_probs_for_current_pair = self.edge_base_log_probs[edge_idx_counter].clone().to(
                                current_device)

                        invalid_edge_type_set_for_pair = set()

                        while not valid_edge_found_for_this_source:
                            if edge_log_probs_for_current_pair is None or \
                                    (
                                            len(invalid_edge_type_set_for_pair) >= self.bond_dim or resample_edge_count > 50):  # Tried all bond types or timed out
                                chosen_edge_discrete_id = VIRTUAL_EDGE_INDEX
                            else:
                                prior_edge_dist = torch.distributions.OneHotCategorical(
                                    logits=edge_log_probs_for_current_pair / temperature[1])
                                latent_edge_sample = prior_edge_dist.sample().view(1, -1)
                                sampled_edge_type_id_in_logits = torch.argmax(latent_edge_sample,
                                                                              dim=1).item()  # For blacklisting

                                if self.dp:
                                    resolved_edge_latent = self.flow_core.module.reverse(
                                        cur_node_features, cur_adj_features, latent_edge_sample, mode=1,
                                        edge_index=torch.tensor([[source_prev_node_idx, i]], dtype=torch.long,
                                                                device=current_device)
                                    ).view(-1)
                                else:
                                    resolved_edge_latent = self.flow_core.reverse(
                                        cur_node_features, cur_adj_features, latent_edge_sample, mode=1,
                                        edge_index=torch.tensor([[source_prev_node_idx, i]], dtype=torch.long,
                                                                device=current_device)
                                    ).view(-1)
                                chosen_edge_discrete_id = torch.argmax(resolved_edge_latent).item()

                            # Update cur_adj_features (model state)
                            cur_adj_features[0, chosen_edge_discrete_id, source_prev_node_idx, i] = 1.0

                            if chosen_edge_discrete_id == VIRTUAL_EDGE_INDEX:
                                valid_edge_found_for_this_source = True
                                if aig.has_edge(source_prev_node_idx,
                                                i):  # If a real edge was there from a failed attempt
                                    aig.remove_edge(source_prev_node_idx, i)
                            else:  # Actual edge type proposed
                                aig.add_edge(source_prev_node_idx, i, type=num2bond_map[chosen_edge_discrete_id])
                                is_current_aig_valid = check_validity(aig)

                                if is_current_aig_valid:
                                    valid_edge_found_for_this_source = True
                                    is_node_i_connected = True
                                    actual_edges_added_to_node_i += 1
                                else:  # Backtrack AIG and model state for this edge type
                                    if edge_log_probs_for_current_pair is not None:  # Check if we could sample
                                        edge_log_probs_for_current_pair[sampled_edge_type_id_in_logits] = float('-inf')

                                    aig.remove_edge(source_prev_node_idx, i)
                                    cur_adj_features[
                                        0, chosen_edge_discrete_id, source_prev_node_idx, i] = 0.0  # Reset adj

                                    total_resamples_count += 1
                                    resample_edge_count += 1
                                    invalid_edge_type_set_for_pair.add(chosen_edge_discrete_id)
                        # End of while not valid_edge_found_for_this_source
                    # End of else (attempt to sample real/virtual edge)
                    edge_idx_counter += 1  # Increment for each (source, target) pair considered
                # End of for k_source_attempt loop

                # "Patience" exhibited: if desired_in_degree not met, we tried.
                if desired_in_degree_for_node_i > 0 and actual_edges_added_to_node_i < desired_in_degree_for_node_i:
                    # print(f"Info: Node {i} ({node_type_str}) aimed for {desired_in_degree_for_node_i} edges, got {actual_edges_added_to_node_i}.")
                    pass

                # 3. Update graph backups and decide whether to continue generation
                if is_node_i_connected:
                    is_generation_continue = True
                    graph_backup_candidate = aig.copy()
                    node_features_each_iter_backup = cur_node_features.clone()
                    adj_features_each_iter_backup = cur_adj_features.clone()
                    disconnection_streak = 0
                elif not is_node_i_connected and disconnection_streak < disconnection_patience:
                    # Node i was not connected, but we might tolerate it for a few steps
                    is_generation_continue = True
                    # We still back up this state, as it's the current "best" even if node i is isolated
                    graph_backup_candidate = aig.copy()
                    node_features_each_iter_backup = cur_node_features.clone()
                    adj_features_each_iter_backup = cur_adj_features.clone()
                    disconnection_streak += 1
                    # print(f"Info: Node {i} is not connected. Disconnection streak: {disconnection_streak}")
                else:  # Node i not connected AND patience ran out
                    is_generation_continue = False
                    # print(f"Info: Node {i} not connected and disconnection patience {disconnection_patience} reached. Stopping generation.")
                    # The 'aig' at this point includes the disconnected node 'i'.
                    # 'graph_backup_candidate' holds the state *before* adding the problematic node 'i'
                    # if the previous iteration was connected.
                    # If the *very first node* (i=0) somehow fails to be 'is_node_i_connected' (shouldn't happen),
                    # graph_backup_candidate would be None.

                    # If stopping, we need to decide which graph to return.
                    # 'graph_backup_candidate' is likely the one to consider if 'aig' (with the new disconnected node) is bad.
                    # The logic below for final_graph_to_return handles this.
                    # If we stopped because node 'i' couldn't connect, 'aig' has node 'i',
                    # but 'graph_backup_candidate' (if not None) does not.
                    # We might want to revert to graph_backup_candidate.
                    # The current logic for choosing final_graph_to_return will prefer graph_backup_candidate if it's valid
                    # and aig (the one with the disconnected node i) is not or is too small.
                    pass  # Final graph selection happens after the loop

            # End of for i in range(max_atoms) loop

            # Determine the final graph to return
            final_graph_to_return = nx.DiGraph()
            # Prefer graph_backup_candidate if it exists, is large enough, and meets component minimums
            if graph_backup_candidate is not None and \
                    graph_backup_candidate.number_of_nodes() >= min_atoms and \
                    check_aig_component_minimums(graph_backup_candidate):
                final_graph_to_return = graph_backup_candidate
            # Fallback to current 'aig' if generation completed fully and 'aig' is valid
            elif is_generation_continue and \
                    aig.number_of_nodes() >= min_atoms and \
                    check_aig_component_minimums(aig):
                final_graph_to_return = aig
            # Additional fallback: if graph_backup_candidate was too small or invalid, but aig is better
            elif aig.number_of_nodes() >= min_atoms and \
                    check_aig_component_minimums(aig):
                final_graph_to_return = aig

            if final_graph_to_return.number_of_nodes() == 0 and min_atoms > 0:
                print(
                    f"Warning: Generated AIG is empty or did not meet all criteria "
                    f"(nodes: {final_graph_to_return.number_of_nodes()}), but min_nodes was {min_atoms}.")

            num_nodes_generated = final_graph_to_return.number_of_nodes()
            pure_valid_generation = 1.0 if total_resamples_count == 0 else 0.0

            return final_graph_to_return, pure_valid_generation, num_nodes_generated

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=12):
        """
        Args:
            max_node_unroll: Max number of nodes.
            max_edge_unroll: Max number of edges to predict for each generated node.
        Returns:
            Tuple of masks and indices for the flow core.
        """
        # Calculate total number of steps (node steps + edge steps)
        # Node steps = max_node_unroll
        # Edge steps calculation:
        # For first 'max_edge_unroll' nodes, node 'i' considers 'i' previous nodes. Sum = (max_edge_unroll-1)*max_edge_unroll/2
        # For remaining 'max_node_unroll - max_edge_unroll' nodes, each considers 'max_edge_unroll' previous nodes.
        # Sum = (max_node_unroll - max_edge_unroll) * max_edge_unroll
        num_edge_steps = int(
            (max_edge_unroll - 1) * max_edge_unroll / 2 +
            (max_node_unroll - max_edge_unroll) * max_edge_unroll
        )
        if max_edge_unroll > max_node_unroll:  # handles cases where max_edge_unroll is effectively max_node_unroll
            num_edge_steps = int((max_node_unroll - 1) * max_node_unroll / 2)

        num_masks = max_node_unroll + num_edge_steps

        # Masks for node generation steps
        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()

        # Masks for edge generation steps
        node_masks2 = torch.zeros([num_edge_steps, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_edge_steps, max_node_unroll, max_node_unroll]).bool()

        link_prediction_index = torch.zeros([num_edge_steps, 2]).long()

        # This mask identifies which entries in the full adjacency matrix correspond to the
        # edges considered by the autoregressive model (upper triangle based on edge_unroll).
        # It's used to extract the relevant part of the input adjacency matrix in the forward pass.
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()

        current_node_step_idx = 0
        current_edge_step_idx = 0
        for i in range(max_node_unroll):  # Current node being added (target)
            # Node update step for node i
            node_masks1[current_node_step_idx, :i] = 1  # Previous nodes are visible
            adj_masks1[current_node_step_idx, :i, :i] = 1  # Adjacency among previous nodes is visible
            current_node_step_idx += 1

            # Edge update steps for node i connecting to previous nodes
            num_sources_for_node_i = 0
            if i < max_edge_unroll:
                source_start_idx = 0
                num_sources_for_node_i = i
            else:
                source_start_idx = i - max_edge_unroll
                num_sources_for_node_i = max_edge_unroll

            for j in range(num_sources_for_node_i):  # j iterates over potential sources for current node i
                actual_source_node_idx = source_start_idx + j

                # For this edge step, nodes up to i are visible
                node_masks2[current_edge_step_idx, :i + 1] = 1
                # Adjacency matrix state before adding this edge:
                # Includes connections among nodes <i, and connections from <i to i already made in this inner loop
                if j == 0:  # First edge for node i
                    adj_masks2[current_edge_step_idx] = adj_masks1[
                        current_node_step_idx - 1].clone()  # State before any edges to i
                    adj_masks2[current_edge_step_idx, i, i] = 1  # Mask for node i itself (e.g. for type info)
                else:  # Subsequent edges for node i
                    adj_masks2[current_edge_step_idx] = adj_masks2[current_edge_step_idx - 1].clone()
                    # The previous edge (source_start_idx + j - 1 -> i) is now part of the visible graph
                    # adj_masks2[current_edge_step_idx, i, source_start_idx + j - 1] = 1 # Directed: source -> target
                    adj_masks2[
                        current_edge_step_idx, source_start_idx + j - 1, i] = 1  # Corrected: source -> target for (source, target) indexing

                link_prediction_index[current_edge_step_idx, 0] = actual_source_node_idx
                link_prediction_index[current_edge_step_idx, 1] = i
                current_edge_step_idx += 1

        assert current_node_step_idx == max_node_unroll, 'Node mask count wrong'
        assert current_edge_step_idx == num_edge_steps, 'Edge mask count wrong'

        # Populate flow_core_edge_masks: True for (target, source) pairs considered.
        # This mask is (N, N) where N is max_node_unroll.
        # It should mark adj[target_idx, source_idx] = True if source_idx is a potential source for target_idx.
        for target_idx in range(max_node_unroll):
            if target_idx == 0:
                continue

            num_sources_for_target = 0
            if target_idx < max_edge_unroll:
                source_start = 0
                num_sources_for_target = target_idx
            else:
                source_start = target_idx - max_edge_unroll
                num_sources_for_target = max_edge_unroll

            # flow_core_edge_masks[target_idx, source_start : source_start + num_sources_for_target] = 1
            # Transposed for (target, source) indexing if adj is (bond, target, source)
            # No, the mask should be (target, source) as it's applied to adj (N,N) slice.
            # The selection `adj[:, flow_core_edge_masks]` implies flow_core_edge_masks is 2D and selects elements.
            # If adj is (B, D, N, N), then adj[b, d, flow_core_edge_masks] would work if flow_core_edge_masks is (N,N)
            # The original code was: inp_adj_features[:, :, self.flow_core_edge_masks]
            # This means flow_core_edge_masks must be selecting along the last two N, N dimensions.
            # So, flow_core_edge_masks[target_idx, source_idx] = 1 is correct.
            for s_offset in range(num_sources_for_target):
                source_idx = source_start + s_offset
                flow_core_edge_masks[target_idx, source_idx] = 1

        node_masks_all = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks_all = torch.cat((adj_masks1, adj_masks2), dim=0)

        # Make them non-trainable parameters
        node_masks_all = nn.Parameter(node_masks_all, requires_grad=False)
        adj_masks_all = nn.Parameter(adj_masks_all, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)  # This is (N,N)

        return node_masks_all, adj_masks_all, link_prediction_index, flow_core_edge_masks

    def dis_log_prob(self, z):
        x_deq, adj_deq = z  # x_deq: (B, N, D_node), adj_deq: (B, num_edges, D_edge)

        # node_base_log_probs is (max_size, node_dim)
        # x_deq is (B, N, node_dim) where N is actual number of nodes in each batch item.
        # This implies x_deq needs to be padded or selected to match max_size for direct multiplication.
        # However, the original sum is over (-1, -2), implying a sum over N and node_dim.
        # This means node_base_log_probs should be indexed or tiled.
        # Assuming x_deq is already matched/selected for the N nodes.
        # The sum should be over the actual nodes present, not max_size.

        # If x_deq is (B, max_size, node_dim) and masked for non-existent nodes:
        node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs,
                                                                 dim=-1)  # (max_size, node_dim)
        # ll_node = torch.sum(x_deq * node_base_log_probs_sm.unsqueeze(0), dim=(-1, -2)) # Sum over N and node_dim
        # Corrected: Ensure x_deq corresponds to the first N nodes, and node_base_log_probs also for first N.
        # This assumes x_deq from forward pass corresponds to the full max_size.
        ll_node = torch.sum(x_deq * node_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        # edge_base_log_probs is (num_total_edge_steps, bond_dim)
        # adj_deq is (B, num_autoregressive_edges_in_batch, bond_dim)
        # The number of edges in adj_deq corresponds to the number of True values in flow_core_edge_masks,
        # which is num_mask_edge.
        edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs,
                                                                 dim=-1)  # (num_mask_edge, bond_dim)
        ll_edge = torch.sum(adj_deq * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))  # Sum over Edges and bond_dim

        # Normalize by the total number of latent dimensions modeled
        # latent_node_length = max_size * node_dim
        # latent_edge_length = num_mask_edge * bond_dim
        return -(torch.mean(ll_node + ll_edge) / (self.latent_node_length + self.latent_edge_length))

