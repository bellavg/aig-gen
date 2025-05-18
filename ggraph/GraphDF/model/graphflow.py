import torch
import torch.nn as nn

from .disgraphaf import DisGraphAF  # Assuming this is in the same directory
# Ensure all necessary imports from aig_config are present
from aig_config import (
    NUM2EDGETYPE,
    NUM2NODETYPE,
    NO_EDGE_CHANNEL,  # Still used for explicit "no edge" choices
    check_interim_validity,  # This will be effectively bypassed
    to_directed_aig,
    check_validity,
    remove_padding_nodes,
    display_graph_details
)
import networkx as nx
import numpy as np
import warnings


class GraphFlowModel(nn.Module):
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

        node_base_log_probs = torch.randn(self.max_size, self.node_dim)
        if num_edge_steps > 0:
            edge_base_log_probs = torch.randn(num_edge_steps, self.bond_dim)
        else:
            edge_base_log_probs = torch.empty((0, self.bond_dim))

        self.flow_core = DisGraphAF(node_masks, adj_masks, link_prediction_index,
                                    num_flow_layer=model_conf_dict['num_flow_layer'], graph_size=self.max_size,
                                    num_node_type=self.node_dim, num_edge_type=self.bond_dim,
                                    num_rgcn_layer=model_conf_dict['num_rgcn_layer'],
                                    nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])

        current_device = torch.device("cuda" if self.dp and torch.cuda.is_available() else "cpu")

        if self.dp and not torch.cuda.is_available():
            warnings.warn("CUDA not available, running GraphDF model on CPU despite use_gpu=True.")
            self.dp = False

        self.node_base_log_probs = nn.Parameter(node_base_log_probs.to(current_device), requires_grad=True)
        self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.to(current_device), requires_grad=True)

        if self.dp:
            if current_device.type == 'cuda':
                self.flow_core = nn.DataParallel(self.flow_core)
            else:
                self.dp = False

    def forward(self, inp_node_features, inp_adj_features):
        inp_node_features_one_hot = inp_node_features.clone()

        selected_edges_list = []
        if self.flow_core_edge_masks.numel() > 0:
            for b in range(inp_adj_features.size(0)):
                batch_edges = inp_adj_features[b, :, self.flow_core_edge_masks]
                selected_edges_list.append(batch_edges)
            inp_adj_features_one_hot = torch.stack(selected_edges_list)
            inp_adj_features_one_hot = inp_adj_features_one_hot.permute(0, 2, 1).contiguous()
        else:
            batch_size = inp_adj_features.size(0)
            inp_adj_features_one_hot = torch.empty(batch_size, 0, self.bond_dim,
                                                   device=inp_adj_features.device,
                                                   dtype=inp_adj_features.dtype)

        z = self.flow_core(inp_node_features, inp_adj_features,
                           inp_node_features_one_hot, inp_adj_features_one_hot)
        return z

    def generate(self, temperature=[0.3, 0.3], min_atoms=5, max_atoms=64, disconnection_patience=50):
        disconnection_streak = 0
        current_device = self.node_base_log_probs.device

        with torch.no_grad():
            num2edge_type_map = NUM2EDGETYPE
            num2node_type_map = NUM2NODETYPE

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)

            node_features_each_iter_backup = cur_node_features.clone()
            adj_features_each_iter_backup = cur_adj_features.clone()

            graph_representation = nx.Graph()
            graph_backup_on_connect = None

            is_continue_generation = True
            edge_sampling_idx = 0
            total_resamples_count = 0  # This will remain 0 as we are removing resampling for validity

            model_to_call = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core

            for i in range(max_atoms):
                if not is_continue_generation:
                    break

                if i < self.edge_unroll:
                    num_edges_to_consider_for_node_i = i
                    source_node_start_index = 0
                else:
                    num_edges_to_consider_for_node_i = self.edge_unroll
                    source_node_start_index = i - self.edge_unroll

                prior_node_dist = torch.distributions.OneHotCategorical(
                    logits=self.node_base_log_probs[i] * temperature[0])
                latent_node_sample = prior_node_dist.sample().view(1, -1)

                resolved_node_type_latent = model_to_call.reverse(
                    cur_node_features, cur_adj_features, latent_node_sample, mode=0).view(-1)

                node_feature_id = torch.argmax(resolved_node_type_latent).item()
                cur_node_features[0, i, node_feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0  # Self-loop

                node_type_str = num2node_type_map.get(node_feature_id, f"UNKNOWN_NODE_ID_{node_feature_id}")
                graph_representation.add_node(i, type=node_type_str)
                current_node_is_connected = (i == 0)

                for j_offset in range(num_edges_to_consider_for_node_i):
                    actual_source_node_idx = source_node_start_index + j_offset

                    # Directly sample and add edge without validation loop
                    current_edge_logits_for_sampling = self.edge_base_log_probs[edge_sampling_idx].to(
                        current_device)  # No clone needed

                    if temperature[1] <= 0:
                        latent_edge_idx = torch.argmax(current_edge_logits_for_sampling)
                        latent_edge_sample = torch.nn.functional.one_hot(latent_edge_idx,
                                                                         num_classes=self.bond_dim).float().view(1, -1)
                    else:
                        prior_edge_dist = torch.distributions.OneHotCategorical(
                            logits=current_edge_logits_for_sampling / temperature[1])
                        latent_edge_sample = prior_edge_dist.sample().view(1, -1)

                    resolved_edge_type_latent = model_to_call.reverse(
                        cur_node_features, cur_adj_features, latent_edge_sample,
                        mode=1,
                        edge_index=torch.tensor([[actual_source_node_idx, i]], device=current_device).long()
                    ).view(-1)
                    chosen_edge_type_id = torch.argmax(resolved_edge_type_latent).item()

                    cur_adj_features[0, chosen_edge_type_id, i, actual_source_node_idx] = 1.0
                    cur_adj_features[0, chosen_edge_type_id, actual_source_node_idx, i] = 1.0
                    current_edge_type_str = num2edge_type_map.get(chosen_edge_type_id,
                                                                  f"UNKNOWN_EDGE_ID_{chosen_edge_type_id}")

                    if chosen_edge_type_id != NO_EDGE_CHANNEL:
                        graph_representation.add_edge(i, actual_source_node_idx, type=current_edge_type_str)
                        current_node_is_connected = True
                        # No validation check, edge is added directly

                    edge_sampling_idx += 1

                if current_node_is_connected:
                    is_continue_generation = True
                    graph_backup_on_connect = graph_representation.copy()
                    node_features_each_iter_backup = cur_node_features.clone()
                    adj_features_each_iter_backup = cur_adj_features.clone()
                    disconnection_streak = 0
                elif not current_node_is_connected and disconnection_streak < disconnection_patience:
                    is_continue_generation = True
                    disconnection_streak += 1
                    if graph_backup_on_connect is None: graph_backup_on_connect = graph_representation.copy()
                else:
                    is_continue_generation = False
                    if graph_backup_on_connect is not None:
                        graph_representation = graph_backup_on_connect.copy()
                        cur_node_features = node_features_each_iter_backup.clone()
                        cur_adj_features = adj_features_each_iter_backup.clone()

            final_graph_nx = graph_representation
            if final_graph_nx is None or final_graph_nx.number_of_nodes() == 0:
                warnings.warn("Generated graph is empty or None at the end.")
                final_graph_nx = graph_backup_on_connect if graph_backup_on_connect is not None else nx.Graph()
                if final_graph_nx.number_of_nodes() == 0:
                    return None, 0, 0

            final_aig_directed = to_directed_aig(final_graph_nx)
            if final_aig_directed:
                final_aig_processed = remove_padding_nodes(final_aig_directed)
                if final_aig_processed is None:
                    final_aig_processed = final_aig_directed
            else:
                warnings.warn(f"to_directed_aig failed for generated graph. Using original NetworkX graph.")
                final_aig_processed = final_graph_nx

            num_nodes_after_processing = final_aig_processed.number_of_nodes() if final_aig_processed else 0

            if num_nodes_after_processing < min_atoms:
                return None, 1.0, num_nodes_after_processing  # pure_valid_flag is 1.0 as no resampling for validity

            pure_valid_flag = 1.0  # No resampling based on validity

            return final_aig_processed, pure_valid_flag, num_nodes_after_processing

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=25):
        num_edge_steps = 0
        if max_node_unroll <= 0:
            num_edge_steps = 0
        elif max_edge_unroll >= max_node_unroll - 1:
            num_edge_steps = int((max_node_unroll - 1) * max_node_unroll / 2) if max_node_unroll > 0 else 0
        else:
            num_edge_steps = int(
                (max_edge_unroll - 1) * max_edge_unroll / 2 +
                (max_node_unroll - max_edge_unroll) * max_edge_unroll
            ) if max_edge_unroll > 0 else 0
            if max_edge_unroll == 0:
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

            current_adj_mask_for_edges_of_node_i = adj_masks_for_node_step[i, :,
                                                   :].clone() if i < max_node_unroll else torch.zeros(max_node_unroll,
                                                                                                      max_node_unroll).bool()
            current_adj_mask_for_edges_of_node_i[i, i] = True

            for j_offset in range(num_sources_to_connect_to):
                actual_source_node = source_start_node_idx + j_offset
                if current_edge_step_idx < num_edge_steps:
                    node_masks_for_edge_step[current_edge_step_idx, :i + 1] = 1
                    adj_masks_for_edge_step[current_edge_step_idx, :, :] = current_adj_mask_for_edges_of_node_i
                    link_prediction_index[current_edge_step_idx, 0] = actual_source_node
                    link_prediction_index[current_edge_step_idx, 1] = i
                    flow_core_edge_masks[i, actual_source_node] = 1
                    current_edge_step_idx += 1
                    current_adj_mask_for_edges_of_node_i[i, actual_source_node] = True
                    current_adj_mask_for_edges_of_node_i[actual_source_node, i] = True
                else:
                    if num_edge_steps > 0:
                        warnings.warn(
                            f"Mask Init: Trying to create edge step {current_edge_step_idx + 1} but only {num_edge_steps} are allocated. Max_node_unroll: {max_node_unroll}, Max_edge_unroll: {max_edge_unroll}")

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

        if adj_output.numel() == 0 and self.edge_base_log_probs.numel() == 0:
            ll_edge = torch.zeros_like(ll_node)
        elif adj_output.shape[1] == 0 and self.edge_base_log_probs.shape[0] == 0:
            ll_edge = torch.zeros_like(ll_node)
        elif adj_output.shape[1] != self.edge_base_log_probs.shape[0]:
            warnings.warn(
                f"Shape mismatch in dis_log_prob for edges: adj_output steps {adj_output.shape[1]}, edge_base_log_probs steps {self.edge_base_log_probs.shape[0]}")
            if adj_output.numel() == 0:
                ll_edge = torch.zeros_like(ll_node)
            else:
                ll_edge = torch.tensor(float('-inf'), device=ll_node.device).repeat(ll_node.shape[0])
        else:
            ll_edge = torch.sum(adj_output * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))

        mean_ll = torch.mean(ll_node + ll_edge)
        num_node_variables = self.max_size
        num_edge_variables = self.edge_base_log_probs.shape[0]
        total_variables = num_node_variables + num_edge_variables
        if total_variables == 0:
            return -torch.tensor(0.0, device=mean_ll.device) if mean_ll == 0 else -mean_ll
        return -(mean_ll / total_variables)


# import torch
# import torch.nn as nn
#
# from .disgraphaf import DisGraphAF  # Assuming this is in the same directory
# # Ensure all necessary imports from aig_config are present
# from aig_config import (
#     NUM2EDGETYPE,
#     NUM2NODETYPE,
#     NO_EDGE_CHANNEL,  # Still used for explicit "no edge" choices
#     check_interim_validity,  # Corrected call will be used
#     to_directed_aig,
#     remove_padding_nodes,
#     display_graph_details  # For debugging the first failure
# )
# import networkx as nx
# import numpy as np
# import warnings
#
#
# class GraphFlowModel(nn.Module):
#     # Class attribute to track if detailed failure prints have occurred for the first graph
#     _detailed_failure_printed_for_first_graph = False
#
#     def __init__(self, model_conf_dict):
#         super(GraphFlowModel, self).__init__()
#         self.max_size = model_conf_dict['max_size']
#         self.edge_unroll = model_conf_dict['edge_unroll']
#         self.node_dim = model_conf_dict['node_dim']
#         self.bond_dim = model_conf_dict['bond_dim']
#
#         node_masks, adj_masks, link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(
#             max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)
#
#         self.latent_step = node_masks.size(
#             0)
#         self.latent_node_length = self.max_size * self.node_dim
#
#         # Calculate the number of edge generation steps based on how initialize_masks works
#         num_edge_steps = 0
#         if self.max_size <= 0:  # Handle case with no nodes
#             num_edge_steps = 0
#         elif self.edge_unroll >= self.max_size - 1:  # If edge_unroll is large enough to connect to all previous nodes
#             num_edge_steps = int((self.max_size - 1) * self.max_size / 2) if self.max_size > 0 else 0
#         else:  # Standard case
#             num_edge_steps = int(
#                 (self.edge_unroll - 1) * self.edge_unroll / 2 +  # Edges for the first 'max_edge_unroll' nodes
#                 (self.max_size - self.edge_unroll) * self.edge_unroll  # Edges for the remaining nodes
#             )
#             if self.edge_unroll == 0:  # if edge_unroll is 0
#                 num_edge_steps = 0
#         self.latent_edge_length = num_edge_steps * self.bond_dim
#
#         self.dp = model_conf_dict['use_gpu']
#
#         node_base_log_probs = torch.randn(self.max_size, self.node_dim)
#         # Ensure edge_base_log_probs has the correct size, even if num_edge_steps is 0
#         if num_edge_steps > 0:
#             edge_base_log_probs = torch.randn(num_edge_steps, self.bond_dim)
#         else:
#             edge_base_log_probs = torch.empty((0, self.bond_dim))  # Empty tensor if no edge steps
#
#         self.flow_core = DisGraphAF(node_masks, adj_masks, link_prediction_index,
#                                     num_flow_layer=model_conf_dict['num_flow_layer'], graph_size=self.max_size,
#                                     num_node_type=self.node_dim, num_edge_type=self.bond_dim,
#                                     num_rgcn_layer=model_conf_dict['num_rgcn_layer'],
#                                     nhid=model_conf_dict['nhid'], nout=model_conf_dict['nout'])
#
#         current_device = torch.device("cuda" if self.dp and torch.cuda.is_available() else "cpu")
#
#         if self.dp and not torch.cuda.is_available():
#             warnings.warn("CUDA not available, running GraphDF model on CPU despite use_gpu=True.")
#             self.dp = False  # Correct the flag
#
#         self.node_base_log_probs = nn.Parameter(node_base_log_probs.to(current_device), requires_grad=True)
#         self.edge_base_log_probs = nn.Parameter(edge_base_log_probs.to(current_device), requires_grad=True)
#
#         if self.dp:  # Apply DataParallel if dp is True and Cuda is available
#             if current_device.type == 'cuda':
#                 self.flow_core = nn.DataParallel(self.flow_core)
#             else:
#                 self.dp = False
#
#     def forward(self, inp_node_features, inp_adj_features):
#         """
#         Args:
#             inp_node_features: (B, N, node_dim) - N is max_size
#             inp_adj_features: (B, bond_dim, N, N) - N is max_size
#
#         Returns:
#             z: [(B, max_size, node_dim), (B, num_edge_steps, bond_dim)]
#                Output from DisGraphAF, which are the transformed one-hot vectors.
#         """
#         inp_node_features_one_hot = inp_node_features.clone()
#
#         selected_edges_list = []
#         if self.flow_core_edge_masks.numel() > 0:  # Only select if there are edges to select
#             for b in range(inp_adj_features.size(0)):
#                 batch_edges = inp_adj_features[b, :, self.flow_core_edge_masks]
#                 selected_edges_list.append(batch_edges)
#             inp_adj_features_one_hot = torch.stack(selected_edges_list)
#             inp_adj_features_one_hot = inp_adj_features_one_hot.permute(0, 2, 1).contiguous()
#         else:  # No edges are modeled by flow_core_edge_masks (e.g. max_size is 0 or 1)
#             # Create an empty tensor with the expected shape (B, 0, bond_dim)
#             batch_size = inp_adj_features.size(0)
#             inp_adj_features_one_hot = torch.empty(batch_size, 0, self.bond_dim,
#                                                    device=inp_adj_features.device,
#                                                    dtype=inp_adj_features.dtype)
#
#         z = self.flow_core(inp_node_features, inp_adj_features,
#                            inp_node_features_one_hot, inp_adj_features_one_hot)
#         return z
#
#     def generate(self, temperature=[0.3, 0.3], min_atoms=5, max_atoms=64, disconnection_patience=50):
#         """
#         Inverse flow to generate AIGs.
#         This version aligns more closely with the original RDKit-based generation logic
#         for self-loop adjacency initialization and validity checking.
#         """
#         disconnection_streak = 0
#         current_device = self.node_base_log_probs.device
#
#         with torch.no_grad():
#             # AIG-specific type mappings (replaces atom_list and num2bond from RDKit original)
#             num2edge_type_map = NUM2EDGETYPE
#             num2node_type_map = NUM2NODETYPE
#
#             # Initialize feature matrices
#             cur_node_features = torch.zeros([1, max_atoms, self.node_dim], device=current_device)
#             cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms], device=current_device)
#
#             # Backup features (similar to original)
#             node_features_each_iter_backup = cur_node_features.clone()
#             adj_features_each_iter_backup = cur_adj_features.clone()
#
#             # AIG representation (replaces RDKit's RWMol)
#             # Using NetworkX Graph. It will be converted to DiGraph later if needed by validity checks.
#             # The original used RWMol which is mutable. NetworkX graphs are also mutable.
#             graph_representation = nx.Graph()  # Changed variable name for clarity
#             graph_backup_on_connect = None  # Backup of the graph state
#
#             is_continue_generation = True  # Renamed for clarity
#             edge_sampling_idx = 0  # Renamed for clarity
#             total_resamples_count = 0  # Renamed for clarity
#             # each_node_resample = np.zeros([max_atoms]) # This was in original, can be re-added if detailed stats are needed
#
#             model_to_call = self.flow_core.module if isinstance(self.flow_core, nn.DataParallel) else self.flow_core
#
#             for i in range(max_atoms):  # Iterate up to max_atoms (potential nodes)
#                 if not is_continue_generation:
#                     break
#
#                 # Determine how many previous nodes to consider connecting to
#                 if i < self.edge_unroll:
#                     num_edges_to_consider_for_node_i = i
#                     source_node_start_index = 0
#                 else:
#                     num_edges_to_consider_for_node_i = self.edge_unroll
#                     source_node_start_index = i - self.edge_unroll
#
#                 # 1. Generate Node Type
#
#                 prior_node_dist = torch.distributions.OneHotCategorical(logits=self.node_base_log_probs[i]*temperature[0])
#                 latent_node_sample = prior_node_dist.sample().view(1, -1)
#
#                 # Reverse flow for node type
#                 resolved_node_type_latent = model_to_call.reverse(
#                     cur_node_features, cur_adj_features, latent_node_sample, mode=0).view(-1)
#
#                 node_feature_id = torch.argmax(resolved_node_type_latent).item()  # Get the chosen node type index
#
#                 # Update node features and add node to graph
#                 cur_node_features[0, i, node_feature_id] = 1.0
#
#                 # ORIGINAL LOGIC FOR SELF-LOOP: Set all bond dimensions for self-loop
#                 cur_adj_features[0, :, i, i] = 1.0
#
#                 node_type_str = num2node_type_map.get(node_feature_id, f"UNKNOWN_NODE_ID_{node_feature_id}")
#                 graph_representation.add_node(i, type=node_type_str)
#
#                 # Connectivity flag for the current node
#                 current_node_is_connected = (i == 0)  # First node is considered connected by default
#
#                 # 2. Generate Edges to Previous Nodes
#                 for j_offset in range(num_edges_to_consider_for_node_i):
#                     actual_source_node_idx = source_node_start_index + j_offset
#
#                         # Edge generation loop with resampling for validity
#                     edge_successfully_placed = False
#                     resample_edge_attempts = 0
#                     # Clone base logits for current edge step to allow modification (marking invalid choices)
#                     current_edge_logits_for_sampling = self.edge_base_log_probs[edge_sampling_idx].to(
#                         current_device).clone()
#                     invalid_edge_types_tried = set()
#
#                     while not edge_successfully_placed:
#                         # Force NO_EDGE_CHANNEL if too many resamples or all other types are invalid
#                         if resample_edge_attempts > 50 or len(invalid_edge_types_tried) >= (
#                                 self.bond_dim - 1) or torch.isneginf(current_edge_logits_for_sampling).all():
#                             chosen_edge_type_id = NO_EDGE_CHANNEL
#                         else:
#                             if temperature[1] <= 0:  # Greedy edge type sampling
#                                 latent_edge_idx = torch.argmax(current_edge_logits_for_sampling)
#                                 latent_edge_sample = torch.nn.functional.one_hot(latent_edge_idx,
#                                                                                  num_classes=self.bond_dim).float().view(
#                                     1, -1)
#                             else:  # Sample edge type
#                                 prior_edge_dist = torch.distributions.OneHotCategorical(
#                                     logits=current_edge_logits_for_sampling / temperature[1])
#                                 latent_edge_sample = prior_edge_dist.sample().view(1, -1)
#
#                             # Reverse flow for edge type
#                             resolved_edge_type_latent = model_to_call.reverse(
#                                 cur_node_features, cur_adj_features, latent_edge_sample,
#                                 mode=1,
#                                 edge_index=torch.tensor([[actual_source_node_idx, i]], device=current_device).long()
#                             ).view(-1)
#                             chosen_edge_type_id = torch.argmax(resolved_edge_type_latent).item()
#
#                         # Update adjacency features
#                         cur_adj_features[0, chosen_edge_type_id, i, actual_source_node_idx] = 1.0
#                         cur_adj_features[0, chosen_edge_type_id, actual_source_node_idx, i] = 1.0
#
#                         current_edge_type_str = num2edge_type_map.get(chosen_edge_type_id,
#                                                                       f"UNKNOWN_EDGE_ID_{chosen_edge_type_id}")
#
#                         if chosen_edge_type_id == NO_EDGE_CHANNEL:  # No functional edge chosen
#                             edge_successfully_placed = True
#                         else:  # A functional edge type was chosen, add to graph and check validity
#                             graph_representation.add_edge(i, actual_source_node_idx, type=current_edge_type_str)
#
#                             # CORRECTED VALIDITY CHECK CALL
#                             # Pass source and target for directed graph check context
#                             is_structurally_valid = check_interim_validity(graph_representation, actual_source_node_idx,
#                                                                            i)
#
#                             if is_structurally_valid:
#                                 edge_successfully_placed = True
#                                 current_node_is_connected = True  # Mark current node `i` as connected
#                             else:  # Backtrack: remove edge, reset features, penalize this edge type choice
#                                 if not GraphFlowModel._detailed_failure_printed_for_first_graph:
#                                     print(
#                                         f"\n--- [GraphDF First Ever Failing Edge Detail] Attempting edge ({actual_source_node_idx} -- {i}) type {current_edge_type_str} ---")
#                                     display_graph_details(graph_representation,
#                                                           f"GraphDF state BEFORE check_interim_validity for edge ({actual_source_node_idx}--{i}) that FAILED")
#                                     print(
#                                         f"--- [GraphDF First Ever Failing Edge Detail] check_interim_validity FAILED for edge ({actual_source_node_idx} -- {i}) ---")
#                                     GraphFlowModel._detailed_failure_printed_for_first_graph = True
#
#                                 graph_representation.remove_edge(i, actual_source_node_idx)
#                                 cur_adj_features[0, chosen_edge_type_id, i, actual_source_node_idx] = 0.0
#                                 cur_adj_features[0, chosen_edge_type_id, actual_source_node_idx, i] = 0.0
#                                 total_resamples_count += 1.0
#                                 resample_edge_attempts += 1
#                                 current_edge_logits_for_sampling[chosen_edge_type_id] = -float(
#                                     'inf')  # Penalize this choice
#                                 invalid_edge_types_tried.add(chosen_edge_type_id)
#
#                     edge_sampling_idx += 1  # Move to the next edge slot for base probabilities
#
#                 # After trying to connect node `i` to all considered previous nodes
#                 if current_node_is_connected:
#                     is_continue_generation = True
#                     graph_backup_on_connect = graph_representation.copy()  # Backup successful state
#                     node_features_each_iter_backup = cur_node_features.clone()
#                     adj_features_each_iter_backup = cur_adj_features.clone()
#                     disconnection_streak = 0
#                 elif not current_node_is_connected and disconnection_streak < disconnection_patience:
#                     # Node `i` was added but couldn't connect. If patience not exceeded, continue adding nodes.
#                     # The graph state remains as is (with disconnected node `i`), but backups are NOT updated.
#                     is_continue_generation = True
#                     disconnection_streak += 1
#                     # Restore from backup if we want to discard the disconnected node `i` immediately.
#                     # For now, aligns with original: keep adding nodes and stop if streak too long.
#                     # If graph_backup_on_connect is None (e.g. first node was disconnected, which shouldn't happen),
#                     # then just use the current graph_representation as the backup.
#                     if graph_backup_on_connect is None: graph_backup_on_connect = graph_representation.copy()
#
#                 else:  # Node `i` is not connected AND disconnection patience exceeded
#                     is_continue_generation = False
#                     # Restore from the last known good connected state
#                     if graph_backup_on_connect is not None:
#                         graph_representation = graph_backup_on_connect.copy()
#                         cur_node_features = node_features_each_iter_backup.clone()
#                         cur_adj_features = adj_features_each_iter_backup.clone()
#                     # If graph_backup_on_connect is None (should be rare), then we might end up with a small/empty graph.
#
#             # Final processing of the generated graph
#             final_graph_nx = graph_representation  # This is the graph after the loop
#             if final_graph_nx is None or final_graph_nx.number_of_nodes() == 0:
#                 warnings.warn("Generated graph is empty or None at the end.")
#                 # Try to use backup if primary is empty and backup exists
#                 final_graph_nx = graph_backup_on_connect if graph_backup_on_connect is not None else nx.Graph()
#                 if final_graph_nx.number_of_nodes() == 0:
#                     return None, 0, 0  # Return None if still empty
#
#             # Convert to directed AIG and remove padding (AIG-specific post-processing)
#             final_aig_directed = to_directed_aig(final_graph_nx)
#             if final_aig_directed:
#                 final_aig_processed = remove_padding_nodes(final_aig_directed)
#                 # If remove_padding_nodes returns None (e.g. empty graph after padding removal),
#                 # fall back to the directed graph before padding removal.
#                 if final_aig_processed is None:
#                     final_aig_processed = final_aig_directed
#             else:
#                 warnings.warn(f"to_directed_aig failed for generated graph. Using original NetworkX graph.")
#                 final_aig_processed = final_graph_nx  # Fallback to the undirected graph if conversion fails
#
#             num_nodes_after_processing = final_aig_processed.number_of_nodes() if final_aig_processed else 0
#
#             # Check against min_atoms
#             if num_nodes_after_processing < min_atoms:
#                 return None, (total_resamples_count == 0), num_nodes_after_processing
#
#             pure_valid_flag = 1.0 if total_resamples_count == 0 else 0.0
#             return final_aig_processed, pure_valid_flag, num_nodes_after_processing
#
#     def initialize_masks(self, max_node_unroll=38, max_edge_unroll=25):
#         # This mask initialization logic seems to be consistent with the original's intent
#         # for ordering node and edge generation steps.
#         # The calculation of num_edge_steps here should match self.latent_edge_length calculation in __init__.
#         num_edge_steps = 0
#         if max_node_unroll <= 0:
#             num_edge_steps = 0
#         elif max_edge_unroll >= max_node_unroll - 1:
#             num_edge_steps = int((max_node_unroll - 1) * max_node_unroll / 2) if max_node_unroll > 0 else 0
#         else:
#             num_edge_steps = int(
#                 (max_edge_unroll - 1) * max_edge_unroll / 2 +
#                 (max_node_unroll - max_edge_unroll) * max_edge_unroll
#             ) if max_edge_unroll > 0 else 0
#             if max_edge_unroll == 0:
#                 num_edge_steps = 0
#
#         node_masks_for_node_step = torch.zeros([max_node_unroll, max_node_unroll]).bool()
#         adj_masks_for_node_step = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()
#
#         if num_edge_steps > 0:
#             node_masks_for_edge_step = torch.zeros([num_edge_steps, max_node_unroll]).bool()
#             adj_masks_for_edge_step = torch.zeros([num_edge_steps, max_node_unroll, max_node_unroll]).bool()
#             link_prediction_index = torch.zeros([num_edge_steps, 2]).long()
#         else:  # Handle cases where num_edge_steps could be 0
#             node_masks_for_edge_step = torch.empty([0, max_node_unroll]).bool()
#             adj_masks_for_edge_step = torch.empty([0, max_node_unroll, max_node_unroll]).bool()
#             link_prediction_index = torch.empty([0, 2]).long()
#
#         flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()
#
#         current_node_step_idx = 0
#         current_edge_step_idx = 0
#
#         for i in range(max_node_unroll):  # For each potential node to be added
#             # === Node Generation Step Mask for node i ===
#             if current_node_step_idx < max_node_unroll:  # Should always be true within this loop
#                 # Visible nodes: 0 to i-1
#                 node_masks_for_node_step[current_node_step_idx, :i] = 1
#                 # Visible adjacency: connections among nodes 0 to i-1
#                 adj_masks_for_node_step[current_node_step_idx, :i, :i] = 1
#                 current_node_step_idx += 1
#
#             # === Edge Generation Step Masks for edges connecting to node i ===
#             num_sources_to_connect_to = 0
#             source_start_node_idx = 0
#             if i == 0:  # First node has no previous nodes to connect to
#                 num_sources_to_connect_to = 0
#             elif i < max_edge_unroll:  # Node i connects to all previous nodes 0 to i-1
#                 source_start_node_idx = 0
#                 num_sources_to_connect_to = i
#             else:  # Node i connects to `max_edge_unroll` previous nodes
#                 source_start_node_idx = i - max_edge_unroll
#                 num_sources_to_connect_to = max_edge_unroll
#
#             # This is the state of the adjacency matrix *after* node i's type is known,
#             # but *before* any of its edges (to 0..i-1) are decided.
#             # It includes connections among 0..i-1 (from node step) and self-loop for i.
#             current_adj_mask_for_edges_of_node_i = adj_masks_for_node_step[i, :,
#                                                    :].clone() if i < max_node_unroll else torch.zeros(max_node_unroll,
#                                                                                                       max_node_unroll).bool()
#             current_adj_mask_for_edges_of_node_i[i, i] = True
#
#             for j_offset in range(num_sources_to_connect_to):
#                 actual_source_node = source_start_node_idx + j_offset
#
#                 if current_edge_step_idx < num_edge_steps:  # Ensure we don't exceed allocated mask size
#                     # Visible nodes: 0 to i (node i's type is now known)
#                     node_masks_for_edge_step[current_edge_step_idx, :i + 1] = 1
#
#                     # Visible adjacency: state captured in current_adj_mask_for_edges_of_node_i
#                     adj_masks_for_edge_step[current_edge_step_idx, :, :] = current_adj_mask_for_edges_of_node_i
#
#                     link_prediction_index[
#                         current_edge_step_idx, 0] = actual_source_node  # Source of the edge being predicted
#                     link_prediction_index[current_edge_step_idx, 1] = i  # Target of the edge being predicted
#
#                     # Mark this edge (i, actual_source_node) as one that flow_core will model
#                     flow_core_edge_masks[i, actual_source_node] = 1
#
#                     current_edge_step_idx += 1
#
#                     # Update current_adj_mask_for_edges_of_node_i for the *next* edge prediction step for node i:
#                     # The edge (i, actual_source_node) is now considered known.
#                     current_adj_mask_for_edges_of_node_i[i, actual_source_node] = True
#                     current_adj_mask_for_edges_of_node_i[actual_source_node, i] = True
#                 else:
#                     # This case should ideally not be hit if num_edge_steps is calculated correctly.
#                     # If it is, it means we are trying to define more edge steps than allocated.
#                     if num_edge_steps > 0:  # Only warn if we expected edge steps
#                         warnings.warn(
#                             f"Mask Init: Trying to create edge step {current_edge_step_idx + 1} but only {num_edge_steps} are allocated. Max_node_unroll: {max_node_unroll}, Max_edge_unroll: {max_edge_unroll}")
#
#         # Final checks for mask counts
#         if not (current_node_step_idx == max_node_unroll):
#             warnings.warn(f"Node mask count mismatch: expected {max_node_unroll}, got {current_node_step_idx}")
#         if not (current_edge_step_idx == num_edge_steps):
#             warnings.warn(
#                 f"Edge mask count mismatch: expected {num_edge_steps}, got {current_edge_step_idx}. This can happen if max_node_unroll is small (e.g., 0 or 1).")
#
#         # Concatenate node-step masks and edge-step masks
#         node_masks_all = torch.cat((node_masks_for_node_step, node_masks_for_edge_step), dim=0)
#         adj_masks_all = torch.cat((adj_masks_for_node_step, adj_masks_for_edge_step), dim=0)
#
#         # Set as non-trainable parameters
#         node_masks_all = nn.Parameter(node_masks_all, requires_grad=False)
#         adj_masks_all = nn.Parameter(adj_masks_all, requires_grad=False)
#         link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
#         flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)
#
#         return node_masks_all, adj_masks_all, link_prediction_index, flow_core_edge_masks
#
#     def dis_log_prob(self, z):
#         x_output, adj_output = z  # x_output: (B, N, node_dim), adj_output: (B, num_edge_steps, bond_dim)
#
#         # Node likelihood
#         node_base_log_probs_sm = torch.nn.functional.log_softmax(self.node_base_log_probs, dim=-1)  # (N, node_dim)
#         # Sum over node_dim and N (max_size)
#         ll_node = torch.sum(x_output * node_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))  # (B)
#
#         # Edge likelihood
#         edge_base_log_probs_sm = torch.nn.functional.log_softmax(self.edge_base_log_probs,
#                                                                  dim=-1)  # (num_edge_steps, bond_dim)
#
#         if adj_output.numel() == 0 and self.edge_base_log_probs.numel() == 0:
#             ll_edge = torch.zeros_like(ll_node)
#         elif adj_output.shape[1] == 0 and self.edge_base_log_probs.shape[0] == 0:  # No edge steps defined
#             ll_edge = torch.zeros_like(ll_node)
#         elif adj_output.shape[1] != self.edge_base_log_probs.shape[0]:  # Mismatch in number of edge steps
#             warnings.warn(
#                 f"Shape mismatch in dis_log_prob for edges: adj_output steps {adj_output.shape[1]}, edge_base_log_probs steps {self.edge_base_log_probs.shape[0]}")
#             if adj_output.numel() == 0:  # If adj_output is empty due to no actual edges being formed in this batch/step
#                 ll_edge = torch.zeros_like(ll_node)
#             else:  # This is a more problematic mismatch
#                 ll_edge = torch.tensor(float('-inf'), device=ll_node.device).repeat(ll_node.shape[0])
#         else:
#             # Sum over bond_dim and num_edge_steps
#             ll_edge = torch.sum(adj_output * edge_base_log_probs_sm.unsqueeze(0), dim=(-1, -2))  # (B)
#
#         mean_ll = torch.mean(ll_node + ll_edge)  # Scalar
#
#         # Normalization factor: total number of variables predicted (nodes + edges)
#         # This should align with how latent_node_length and latent_edge_length are calculated in __init__
#         # latent_node_length = self.max_size * self.node_dim -> but here we sum over node_dim in ll_node
#         # So, for normalization, we consider number of node variables = self.max_size
#         # And number of edge variables = num_edge_steps (self.edge_base_log_probs.shape[0])
#
#         num_node_variables = self.max_size
#         num_edge_variables = self.edge_base_log_probs.shape[0]  # Number of edge generation steps
#
#         total_variables = num_node_variables + num_edge_variables
#         if total_variables == 0:  # Avoid division by zero if max_size is 0
#             return -torch.tensor(0.0, device=mean_ll.device) if mean_ll == 0 else -mean_ll
#
#         # Negative log-likelihood, normalized per variable
#         return -(mean_ll / total_variables)
