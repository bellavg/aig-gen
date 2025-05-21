import torch

from collections import defaultdict
from torch.utils.data import Dataset

__all__ = ['LayerDAGNodeCountDataset',
           'LayerDAGNodePredDataset',
           'LayerDAGEdgePredDataset',
           'collate_node_count',
           'collate_node_pred',
           'collate_edge_pred']


class LayerDAGBaseDataset(Dataset):
    def __init__(self, conditional=False):
        self.input_src = []
        self.input_dst = []
        self.input_x_n = []
        self.input_level = []

        self.input_e_start = []
        self.input_e_end = []
        self.input_n_start = []
        self.input_n_end = []

        self.conditional = conditional
        if conditional:
            self.input_y = []
            self.input_g = []

    def get_in_deg(self, dst, num_nodes):
        return torch.bincount(dst, minlength=num_nodes).tolist()

    def get_out_adj_list(self, src, dst):
        out_adj_list = defaultdict(list)
        num_edges = len(src)
        for i in range(num_edges):
            out_adj_list[src[i]].append(dst[i])
        return out_adj_list

    def get_in_adj_list(self, src, dst):
        in_adj_list = defaultdict(list)
        num_edges = len(src)
        for i in range(num_edges):
            in_adj_list[dst[i]].append(src[i])
        return in_adj_list

    def base_postprocess(self):
        self.input_src = torch.LongTensor(self.input_src)
        self.input_dst = torch.LongTensor(self.input_dst)
        self.input_x_n = torch.LongTensor(self.input_x_n)
        self.input_level = torch.LongTensor(self.input_level)
        self.input_e_start = torch.LongTensor(self.input_e_start)
        self.input_e_end = torch.LongTensor(self.input_e_end)
        self.input_n_start = torch.LongTensor(self.input_n_start)
        self.input_n_end = torch.LongTensor(self.input_n_end)

        if self.conditional:
            if isinstance(self.input_y, list) and self.input_y:
                if all(isinstance(item, (int, float)) for item in self.input_y):
                    self.input_y = torch.tensor(self.input_y)
                elif all(isinstance(item, torch.Tensor) for item in self.input_y):
                    try:
                        self.input_y = torch.stack(self.input_y)
                    except RuntimeError:
                        pass  # Keep as list if shapes differ
            elif not isinstance(self.input_y, torch.Tensor):
                self.input_y = torch.tensor(self.input_y)
            self.input_g = torch.LongTensor(self.input_g)


class LayerDAGNodeCountDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.label = []

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            # Offset for the current original graph sample's nodes in the global self.input_x_n list
            current_original_sample_node_base_idx = len(self.input_x_n)

            # --- For each step/item generated from this original graph sample ---
            # input_n_start_val will be the global index of the dummy node for this step's input graph
            # input_n_end_val will be the global index after the last node of this step's input graph
            # input_e_start_val will be the global index for edges of this step's input graph
            # input_e_end_val will be the global index after the last edge of this step's input graph

            # State for the current step's input graph construction
            step_input_n_start = current_original_sample_node_base_idx
            step_input_n_end = current_original_sample_node_base_idx  # Will be incremented
            step_input_e_start = len(self.input_src)
            step_input_e_end = len(self.input_src)  # Will be incremented

            # Add dummy node for this original sample's processing. Its global index is current_original_sample_node_base_idx
            self.input_x_n.append(dag_dataset.dummy_category)
            step_input_n_end += 1  # The dummy node is part of the input for the first step

            # Shift original 0-indexed src/dst to be 1-indexed relative to this sample's dummy
            src_plus_dummy = src + 1
            dst_plus_dummy = dst + 1

            level = 0
            self.input_level.append(level)  # Level of the dummy node

            num_nodes_in_original_sample_with_dummy = len(x_n) + 1
            in_deg = self.get_in_deg(dst_plus_dummy, num_nodes_in_original_sample_with_dummy)

            x_n_list = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())
            in_adj_list = self.get_in_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())

            frontiers = [
                u for u in range(1, num_nodes_in_original_sample_with_dummy) if in_deg[u] == 0
            ]  # u is 1-indexed relative to this sample's dummy
            frontier_size = len(frontiers)

            while frontier_size > 0:
                level += 1
                self.input_e_start.append(step_input_e_start)
                self.input_e_end.append(step_input_e_end)
                self.input_n_start.append(step_input_n_start)
                self.input_n_end.append(step_input_n_end)

                if conditional:
                    self.input_g.append(input_g)
                self.label.append(frontier_size)

                next_frontiers = []
                for u_frontier_node_local in frontiers:  # u_frontier_node_local is 1-indexed for this sample
                    # Add node attribute to global list
                    # Its global index will be current_original_sample_node_base_idx + u_frontier_node_local
                    self.input_x_n.append(x_n_list[u_frontier_node_local - 1])
                    self.input_level.append(level)

                    # Add edges to global list, converting local 1-indexed to global 0-indexed
                    for t_source_node_local in in_adj_list[u_frontier_node_local]:
                        # global_t_idx = current_original_sample_node_base_idx + t_source_node_local (if dummy is 0)
                        # global_u_idx = current_original_sample_node_base_idx + u_frontier_node_local
                        # The dummy node is at current_original_sample_node_base_idx
                        # Node t_source_node_local (1-indexed) is at global current_original_sample_node_base_idx + t_source_node_local
                        # Node u_frontier_node_local (1-indexed) is at global current_original_sample_node_base_idx + u_frontier_node_local
                        self.input_src.append(current_original_sample_node_base_idx + t_source_node_local)
                        self.input_dst.append(current_original_sample_node_base_idx + u_frontier_node_local)
                        step_input_e_end += 1

                    for v_target_node_local in out_adj_list[u_frontier_node_local]:
                        in_deg[v_target_node_local] -= 1
                        if in_deg[v_target_node_local] == 0:
                            next_frontiers.append(v_target_node_local)

                # Nodes of the current frontier are now part of the "input graph" for the next step
                step_input_n_end += frontier_size

                frontiers = next_frontiers
                frontier_size = len(frontiers)

            # Termination step
            self.input_e_start.append(step_input_e_start)
            self.input_e_end.append(step_input_e_end)
            self.input_n_start.append(step_input_n_start)
            self.input_n_end.append(step_input_n_end)
            if conditional:
                self.input_g.append(input_g)
            self.label.append(frontier_size)  # Should be 0

        self.base_postprocess()
        self.label = torch.LongTensor(self.label)
        if len(self.label) > 0:
            self.max_layer_size = self.label.max().item()
        else:
            self.max_layer_size = 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        input_e_start_idx = self.input_e_start[index]
        input_e_end_idx = self.input_e_end[index]
        input_n_start_idx = self.input_n_start[index]  # Global start index for this item's nodes
        input_n_end_idx = self.input_n_end[index]

        input_x_n_slice = self.input_x_n[input_n_start_idx:input_n_end_idx]
        input_abs_level_slice = self.input_level[input_n_start_idx:input_n_end_idx]

        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        # src_for_item and dst_for_item now contain GLOBAL indices
        src_for_item = self.input_src[input_e_start_idx:input_e_end_idx]
        dst_for_item = self.input_dst[input_e_start_idx:input_e_end_idx]

        # Re-index global edge indices to be 0-based local to input_x_n_slice
        src_reindexed = src_for_item - input_n_start_idx
        dst_reindexed = dst_for_item - input_n_start_idx

        if self.conditional:
            input_g_idx = self.input_g[index]
            input_y_val = self.input_y[input_g_idx].item() if isinstance(self.input_y[input_g_idx], torch.Tensor) and \
                                                              self.input_y[input_g_idx].numel() == 1 else self.input_y[
                input_g_idx]
            return src_reindexed, dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                input_y_val, self.label[index]
        else:
            return src_reindexed, dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                self.label[index]


class LayerDAGNodePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False, get_marginal=True):
        super().__init__(conditional)
        self.label_start = []  # Global start index in self.input_x_n for label nodes
        self.label_end = []  # Global end index in self.input_x_n for label nodes

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            current_original_sample_node_base_idx = len(self.input_x_n)

            step_input_n_start = current_original_sample_node_base_idx
            step_input_n_end = current_original_sample_node_base_idx
            step_input_e_start = len(self.input_src)
            step_input_e_end = len(self.input_src)

            # Add dummy node for model input part of this original sample's processing
            self.input_x_n.append(dag_dataset.dummy_category)
            step_input_n_end += 1

            # Labels for this sample will start after all its input nodes (including dummy)
            # The label nodes are also added to self.input_x_n
            current_label_global_start_idx = len(self.input_x_n)

            src_plus_dummy = src + 1
            dst_plus_dummy = dst + 1

            level = 0
            self.input_level.append(level)  # Level for the dummy node

            num_nodes_in_original_sample_with_dummy = len(x_n) + 1
            in_deg = self.get_in_deg(dst_plus_dummy, num_nodes_in_original_sample_with_dummy)

            x_n_list = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())
            in_adj_list = self.get_in_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())

            frontiers = [
                u for u in range(1, num_nodes_in_original_sample_with_dummy) if in_deg[u] == 0
            ]
            frontier_size = len(frontiers)

            while frontier_size > 0:
                level += 1
                self.input_e_start.append(step_input_e_start)
                self.input_e_end.append(step_input_e_end)
                self.input_n_start.append(step_input_n_start)  # Global start of input nodes for this step
                self.input_n_end.append(step_input_n_end)  # Global end of input nodes for this step

                if conditional:
                    self.input_g.append(input_g)

                # Record global indices for the label nodes (current frontier)
                self.label_start.append(current_label_global_start_idx)
                current_label_global_end_idx = current_label_global_start_idx + frontier_size
                self.label_end.append(current_label_global_end_idx)
                current_label_global_start_idx = current_label_global_end_idx  # For next iteration's labels

                next_frontiers = []
                for u_frontier_node_local in frontiers:  # 1-indexed local to this sample
                    # Add current frontier node attributes to self.input_x_n (these are the labels z)
                    self.input_x_n.append(x_n_list[u_frontier_node_local - 1])
                    self.input_level.append(level)  # Level for this label node

                    # Add edges for model input (edges leading to this new layer)
                    for t_source_node_local in in_adj_list[u_frontier_node_local]:
                        self.input_src.append(current_original_sample_node_base_idx + t_source_node_local)
                        self.input_dst.append(current_original_sample_node_base_idx + u_frontier_node_local)
                        step_input_e_end += 1

                    for v_target_node_local in out_adj_list[u_frontier_node_local]:
                        in_deg[v_target_node_local] -= 1
                        if in_deg[v_target_node_local] == 0:
                            next_frontiers.append(v_target_node_local)

                # The nodes of the current frontier (which were labels for this step)
                # become part of the "input graph" for the *next* prediction step.
                # Their global indices range from step_input_n_end to step_input_n_end + frontier_size -1
                step_input_n_end += frontier_size

                frontiers = next_frontiers
                frontier_size = len(frontiers)

        self.base_postprocess()
        self.label_start = torch.LongTensor(self.label_start)
        self.label_end = torch.LongTensor(self.label_end)

        if get_marginal:
            input_x_n_for_marginal = self.input_x_n
            if input_x_n_for_marginal.ndim == 1:
                input_x_n_for_marginal = input_x_n_for_marginal.unsqueeze(-1)
            num_feats = input_x_n_for_marginal.shape[-1]
            x_n_marginal = []
            for f in range(num_feats):
                input_x_n_f = input_x_n_for_marginal[:, f]
                unique_x_n_f, x_n_count_f = torch.unique(input_x_n_f, return_counts=True)
                num_actual_types_f = dag_dataset.num_categories
                x_n_marginal_f = torch.zeros(num_actual_types_f)
                for c_idx in range(len(unique_x_n_f)):
                    x_n_type_val = unique_x_n_f[c_idx].item()
                    if x_n_type_val < num_actual_types_f:
                        x_n_marginal_f[x_n_type_val] += x_n_count_f[c_idx].item()
                x_n_marginal_f /= (x_n_marginal_f.sum() + 1e-8)
                x_n_marginal.append(x_n_marginal_f)
            self.x_n_marginal = x_n_marginal

    def __len__(self):
        return len(self.label_start)

    def __getitem__(self, index):
        input_e_start_idx = self.input_e_start[index]
        input_e_end_idx = self.input_e_end[index]
        input_n_start_idx = self.input_n_start[index]  # Global start index for input nodes
        input_n_end_idx = self.input_n_end[index]  # Global end index for input nodes

        label_start_global_idx = self.label_start[index]  # Global start index for label nodes
        label_end_global_idx = self.label_end[index]  # Global end index for label nodes

        input_x_n_slice = self.input_x_n[input_n_start_idx:input_n_end_idx]
        input_abs_level_slice = self.input_level[input_n_start_idx:input_n_end_idx]
        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        src_for_item = self.input_src[input_e_start_idx:input_e_end_idx]  # Global indices
        dst_for_item = self.input_dst[input_e_start_idx:input_e_end_idx]  # Global indices

        src_reindexed = src_for_item - input_n_start_idx  # Local to input_x_n_slice
        dst_reindexed = dst_for_item - input_n_start_idx  # Local to input_x_n_slice

        z = self.input_x_n[label_start_global_idx:label_end_global_idx]  # Ground truth attributes
        t, z_t = self.node_diffusion.apply_noise(z)

        if self.conditional:
            input_g_idx = self.input_g[index]
            input_y_val = self.input_y[input_g_idx].item() if isinstance(self.input_y[input_g_idx], torch.Tensor) and \
                                                              self.input_y[input_g_idx].numel() == 1 else self.input_y[
                input_g_idx]
            return src_reindexed, dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                z_t, t, input_y_val, z
        else:
            return src_reindexed, dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                z_t, t, z


class LayerDAGEdgePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.query_src_list = []
        self.query_dst_list = []
        self.label_list = []
        self.query_start_indices = []
        self.query_end_indices = []

        num_total_edges_processed = 0
        num_total_nonsrc_nodes = 0

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            current_original_sample_node_base_idx = len(self.input_x_n)

            step_input_n_start = current_original_sample_node_base_idx
            step_input_n_end = current_original_sample_node_base_idx
            step_input_e_start = len(self.input_src)
            step_input_e_end = len(self.input_src)

            current_query_global_start_idx = len(self.query_src_list)

            self.input_x_n.append(dag_dataset.dummy_category)
            step_input_n_end += 1
            src_plus_dummy = src + 1
            dst_plus_dummy = dst + 1

            level = 0
            self.input_level.append(level)

            num_nodes_in_original_sample_with_dummy = len(x_n) + 1
            in_deg = self.get_in_deg(dst_plus_dummy, num_nodes_in_original_sample_with_dummy)
            x_n_list = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())
            in_adj_list = self.get_in_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())

            prev_frontiers_nodes_local = [
                u for u in range(1, num_nodes_in_original_sample_with_dummy) if in_deg[u] == 0
            ]

            current_frontiers_nodes_local = []
            level += 1

            num_total_edges_processed += len(src)
            num_total_nonsrc_nodes += len(x_n) - len(prev_frontiers_nodes_local)

            for u_node_local in prev_frontiers_nodes_local:
                self.input_x_n.append(x_n_list[u_node_local - 1])
                self.input_level.append(level)
            step_input_n_end += len(prev_frontiers_nodes_local)

            src_candidates_for_queries_global = [current_original_sample_node_base_idx + u_local for u_local in
                                                 prev_frontiers_nodes_local]

            for u_node_local in prev_frontiers_nodes_local:
                for v_target_node_local in out_adj_list[u_node_local]:
                    in_deg[v_target_node_local] -= 1
                    if in_deg[v_target_node_local] == 0:
                        current_frontiers_nodes_local.append(v_target_node_local)

            while len(current_frontiers_nodes_local) > 0:
                level += 1
                self.input_e_start.append(step_input_e_start)
                self.input_e_end.append(step_input_e_end)
                self.input_n_start.append(step_input_n_start)
                self.input_n_end.append(step_input_n_end)
                if conditional:
                    self.input_g.append(input_g)

                self.query_start_indices.append(current_query_global_start_idx)

                num_queries_this_step = 0
                temp_edges_added_to_input_this_step = 0
                next_frontiers_nodes_local_buffer = []

                for u_target_node_local in current_frontiers_nodes_local:
                    self.input_x_n.append(x_n_list[u_target_node_local - 1])
                    self.input_level.append(level)

                    global_u_target_node_idx = current_original_sample_node_base_idx + u_target_node_local

                    for global_t_candidate_source_idx in src_candidates_for_queries_global:
                        self.query_src_list.append(global_t_candidate_source_idx)
                        self.query_dst_list.append(global_u_target_node_idx)
                        num_queries_this_step += 1

                        # Check original connectivity based on local indices for in_adj_list
                        # t_candidate_source_local is global_t_candidate_source_idx - current_original_sample_node_base_idx
                        t_candidate_source_local_for_check = global_t_candidate_source_idx - current_original_sample_node_base_idx
                        if t_candidate_source_local_for_check in in_adj_list[u_target_node_local]:
                            self.input_src.append(global_t_candidate_source_idx)
                            self.input_dst.append(global_u_target_node_idx)
                            temp_edges_added_to_input_this_step += 1
                            self.label_list.append(1)
                        else:
                            self.label_list.append(0)

                    for v_next_target_local in out_adj_list[u_target_node_local]:
                        in_deg[v_next_target_local] -= 1
                        if in_deg[v_next_target_local] == 0:
                            next_frontiers_nodes_local_buffer.append(v_next_target_local)

                step_input_n_end += len(current_frontiers_nodes_local)
                step_input_e_end += temp_edges_added_to_input_this_step

                current_query_global_start_idx += num_queries_this_step
                self.query_end_indices.append(current_query_global_start_idx)

                src_candidates_for_queries_global.extend(
                    [current_original_sample_node_base_idx + u_local for u_local in current_frontiers_nodes_local])
                current_frontiers_nodes_local = list(set(next_frontiers_nodes_local_buffer))

        self.base_postprocess()
        self.query_src_list = torch.LongTensor(self.query_src_list)
        self.query_dst_list = torch.LongTensor(self.query_dst_list)
        self.label_list = torch.LongTensor(self.label_list)
        self.query_start_indices = torch.LongTensor(self.query_start_indices)
        self.query_end_indices = torch.LongTensor(self.query_end_indices)

        if num_total_nonsrc_nodes > 0:
            self.avg_in_deg = num_total_edges_processed / num_total_nonsrc_nodes
        else:
            self.avg_in_deg = 0.0

    def __len__(self):
        return len(self.query_start_indices)

    def __getitem__(self, index):
        input_e_start_idx = self.input_e_start[index]
        input_e_end_idx = self.input_e_end[index]
        input_n_start_idx = self.input_n_start[index]  # Global start index for input nodes
        input_n_end_idx = self.input_n_end[index]

        input_x_n_slice = self.input_x_n[input_n_start_idx:input_n_end_idx]
        input_abs_level_slice = self.input_level[input_n_start_idx:input_n_end_idx]
        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        src_for_item = self.input_src[input_e_start_idx:input_e_end_idx]  # Global indices
        dst_for_item = self.input_dst[input_e_start_idx:input_e_end_idx]  # Global indices
        src_reindexed = src_for_item - input_n_start_idx  # Local to input_x_n_slice
        dst_reindexed = dst_for_item - input_n_start_idx  # Local to input_x_n_slice

        query_s_global_idx = self.query_start_indices[index]
        query_e_global_idx = self.query_end_indices[index]

        query_src_global = self.query_src_list[query_s_global_idx:query_e_global_idx]
        query_dst_global = self.query_dst_list[query_s_global_idx:query_e_global_idx]
        label_for_queries = self.label_list[query_s_global_idx:query_e_global_idx]

        query_src_reindexed = query_src_global - input_n_start_idx  # Local to input_x_n_slice
        query_dst_reindexed = query_dst_global - input_n_start_idx  # Local to input_x_n_slice

        # Placeholder for noise application for edges
        if label_for_queries.numel() > 0:
            # This part needs to correctly call self.edge_diffusion.apply_noise
            # The original code reshaped label_for_queries into an adjacency matrix.
            # If self.edge_diffusion.apply_noise expects a flat list of labels and returns flat noisy labels:
            # t_scalar, label_t_flat = self.edge_diffusion.apply_noise(label_for_queries) # Assuming this interface

            # For now, simplified placeholder:
            t_scalar = torch.randint(0, self.edge_diffusion.T + 1, (1,)).item() if hasattr(self.edge_diffusion,
                                                                                           'T') else 0
            label_t_flat = label_for_queries  # Replace with actual noisy labels from diffusion model

            t_tensor = torch.full((label_for_queries.shape[0], 1), t_scalar, dtype=torch.long)
            mask = (label_t_flat == 1)
            noisy_src_reindexed = query_src_reindexed[mask]
            noisy_dst_reindexed = query_dst_reindexed[mask]
        else:
            t_tensor = torch.empty((0, 1), dtype=torch.long)
            noisy_src_reindexed = torch.empty_like(query_src_reindexed)
            noisy_dst_reindexed = torch.empty_like(query_dst_reindexed)

        if self.conditional:
            input_g_idx = self.input_g[index]
            input_y_val = self.input_y[input_g_idx].item() if isinstance(self.input_y[input_g_idx], torch.Tensor) and \
                                                              self.input_y[input_g_idx].numel() == 1 else self.input_y[
                input_g_idx]
            return src_reindexed, dst_reindexed, \
                noisy_src_reindexed, noisy_dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                t_tensor, input_y_val, \
                query_src_reindexed, query_dst_reindexed, label_for_queries
        else:
            return src_reindexed, dst_reindexed, \
                noisy_src_reindexed, noisy_dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                t_tensor, \
                query_src_reindexed, query_dst_reindexed, label_for_queries


def collate_common(src, dst, x_n, abs_level, rel_level):
    valid_indices = [i for i, item_x_n in enumerate(x_n) if item_x_n is not None and len(item_x_n) > 0]
    if not valid_indices:
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long)

    src = [src[i] for i in valid_indices]
    dst = [dst[i] for i in valid_indices]
    x_n = [x_n[i] for i in valid_indices]
    abs_level = [abs_level[i] for i in valid_indices]
    rel_level = [rel_level[i] for i in valid_indices]

    if not x_n:
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long)

    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in x_n]), dim=0)
    batch_size = len(x_n)
    src_ = []
    dst_ = []
    for i in range(batch_size):
        src_i = src[i] if isinstance(src[i], torch.Tensor) else torch.LongTensor(src[i])
        dst_i = dst[i] if isinstance(dst[i], torch.Tensor) else torch.LongTensor(dst[i])
        if src_i.numel() > 0 or dst_i.numel() > 0:
            src_.append(src_i + num_nodes_cumsum[i])
            dst_.append(dst_i + num_nodes_cumsum[i])

    if src_:
        src = torch.cat(src_, dim=0)
        dst = torch.cat(dst_, dim=0)
        edge_index = torch.stack([dst, src])
    else:
        src = torch.empty((0,), dtype=torch.long)
        dst = torch.empty((0,), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x_n = torch.cat(x_n, dim=0).long()
    abs_level = torch.cat(abs_level, dim=0).float().unsqueeze(-1)
    rel_level = torch.cat(rel_level, dim=0).float().unsqueeze(-1)

    nids = []
    gids = []
    for i in range(batch_size):
        num_nodes_in_graph_i = num_nodes_cumsum[i + 1] - num_nodes_cumsum[i]
        if num_nodes_in_graph_i > 0:
            nids.append(torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i + 1]).long())
            gids.append(torch.ones(num_nodes_in_graph_i).fill_(i).long())

    if nids:
        nids = torch.cat(nids, dim=0)
        gids = torch.cat(gids, dim=0)
        n2g_index = torch.stack([gids, nids])
    else:
        nids = torch.empty((0,), dtype=torch.long)
        gids = torch.empty((0,), dtype=torch.long)
        n2g_index = torch.empty((2, 0), dtype=torch.long)

    return batch_size, edge_index, x_n, abs_level, rel_level, n2g_index


def collate_node_count(data):
    data = [d for d in data if d is not None]
    if not data:
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    if len(data[0]) == 7:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_y_list, batch_label = map(list,
                                                                                                           zip(*data))
        batch_y_tensor = torch.tensor(batch_y_list)
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_label = map(list, zip(*data))
        batch_y_tensor = None

    batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, batch_rel_level_cat, \
        batch_n2g_index = collate_common(
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)
    batch_label_stacked = torch.stack(batch_label) if batch_label and all(
        isinstance(l, torch.Tensor) for l in batch_label) else torch.LongTensor(
        batch_label) if batch_label else torch.empty(0, dtype=torch.long)

    if len(data[0]) == 7:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_y_tensor, batch_n2g_index, batch_label_stacked
    else:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_label_stacked


def collate_node_pred(data):
    data = [d for d in data if d is not None]
    if not data:
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), \
            torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), torch.empty((0, 0),
                                                                                                  dtype=torch.long)

    if len(data[0]) == 9:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, \
            batch_z_t, batch_t, batch_y_list, batch_z = map(list, zip(*data))
        batch_y_tensor = torch.tensor(batch_y_list)
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)
    elif len(data[0]) == 8:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, \
            batch_z_t, batch_t, batch_z = map(list, zip(*data))
        batch_y_tensor = None
    else:
        raise ValueError(f"Unexpected number of items in data for collate_node_pred: {len(data[0])}")

    batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, batch_rel_level_cat, \
        batch_n2g_index = collate_common(
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)

    batch_z_t_filtered = [zt for zt in batch_z_t if zt.numel() > 0]
    batch_z_filtered = [z_val for z_val in batch_z if z_val.numel() > 0]

    # Determine expected shape for empty tensors if all are empty
    # Default to 0 features if all z_t or z are empty.
    z_t_feat_dim = batch_z_t[0].shape[1] if batch_z_t and batch_z_t[0].numel() > 0 and batch_z_t[0].ndim > 1 else 0
    z_feat_dim = batch_z[0].shape[1] if batch_z and batch_z[0].numel() > 0 and batch_z[0].ndim > 1 else (
        1 if batch_z and batch_z[0].numel() > 0 and batch_z[0].ndim == 1 else 0)

    batch_z_t_cat = torch.cat(batch_z_t_filtered) if batch_z_t_filtered else torch.empty((0, z_t_feat_dim),
                                                                                         dtype=batch_z_t[
                                                                                             0].dtype if batch_z_t else torch.long)
    batch_t_cat = torch.cat(batch_t).unsqueeze(-1) if batch_t and all(t.numel() > 0 for t in batch_t) else torch.empty(
        (0, 1), dtype=torch.long)
    batch_z_cat = torch.cat(batch_z_filtered) if batch_z_filtered else torch.empty((0, z_feat_dim), dtype=batch_z[
        0].dtype if batch_z else torch.long)

    num_query_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(z_t_i) for z_t_i in batch_z_t]), dim=0)
    query2g = []
    for i in range(batch_size):
        num_queries_i = num_query_cumsum[i + 1] - num_query_cumsum[i]
        if num_queries_i > 0:
            query2g.append(torch.ones(num_queries_i).fill_(i).long())
    if query2g:
        query2g = torch.cat(query2g)
    else:
        query2g = torch.empty((0,), dtype=torch.long)

    if batch_z_cat.ndim == 1 and batch_z_cat.numel() > 0:
        batch_z_cat = batch_z_cat.unsqueeze(-1)
    elif batch_z_cat.numel() == 0 and batch_z_cat.ndim == 1:
        batch_z_cat = batch_z_cat.view(0, z_feat_dim if z_feat_dim > 0 else 1)

    if len(data[0]) == 9:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_z_t_cat, batch_t_cat, batch_y_tensor, \
            query2g, num_query_cumsum, batch_z_cat
    else:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_z_t_cat, batch_t_cat, \
            query2g, num_query_cumsum, batch_z_cat


def collate_edge_pred(data):
    data = [d for d in data if d is not None]
    if not data:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((2, 0), dtype=torch.long), \
            torch.empty((0,), dtype=torch.long), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.long), \
            torch.empty((0, 0), dtype=torch.long), \
            torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0,), dtype=torch.long)

    if len(data[0]) == 12:
        batch_input_src, batch_input_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, \
            batch_abs_level, batch_rel_level, batch_t_list, batch_y_list, \
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        batch_y_tensor = torch.tensor(batch_y_list)
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)
    elif len(data[0]) == 11:
        batch_input_src, batch_input_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, \
            batch_abs_level, batch_rel_level, batch_t_list, \
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        batch_y_tensor = None
    else:
        raise ValueError(f"Unexpected number of items in data for collate_edge_pred: {len(data[0])}")

    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in batch_x_n]), dim=0)
    batch_x_n_cat = torch.cat(batch_x_n, dim=0).long()
    batch_abs_level_cat = torch.cat(batch_abs_level, dim=0).float().unsqueeze(-1)
    batch_rel_level_cat = torch.cat(batch_rel_level, dim=0).float().unsqueeze(-1)

    input_src_cat_list, input_dst_cat_list = [], []
    for i in range(len(batch_x_n)):
        if batch_input_src[i].numel() > 0:
            input_src_cat_list.append(batch_input_src[i] + num_nodes_cumsum[i])
            input_dst_cat_list.append(batch_input_dst[i] + num_nodes_cumsum[i])
    input_src_b = torch.cat(input_src_cat_list) if input_src_cat_list else torch.empty(0, dtype=torch.long)
    input_dst_b = torch.cat(input_dst_cat_list) if input_dst_cat_list else torch.empty(0, dtype=torch.long)
    input_edge_index_b = torch.stack([input_dst_b, input_src_b]) if input_src_b.numel() > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long)

    noisy_src_cat_list, noisy_dst_cat_list = [], []
    for i in range(len(batch_x_n)):
        if batch_noisy_src[i].numel() > 0:
            noisy_src_cat_list.append(batch_noisy_src[i] + num_nodes_cumsum[i])
            noisy_dst_cat_list.append(batch_noisy_dst[i] + num_nodes_cumsum[i])
    noisy_src_b = torch.cat(noisy_src_cat_list) if noisy_src_cat_list else torch.empty(0, dtype=torch.long)
    noisy_dst_b = torch.cat(noisy_dst_cat_list) if noisy_dst_cat_list else torch.empty(0, dtype=torch.long)
    noisy_edge_index_b = torch.stack([noisy_dst_b, noisy_src_b]) if noisy_src_b.numel() > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long)

    query_src_cat_list, query_dst_cat_list = [], []
    t_for_queries_list, labels_for_queries_list = [], []
    for i in range(len(batch_x_n)):
        if batch_query_src[i].numel() > 0:
            query_src_cat_list.append(batch_query_src[i] + num_nodes_cumsum[i])
            query_dst_cat_list.append(batch_query_dst[i] + num_nodes_cumsum[i])
            t_for_queries_list.append(batch_t_list[i])
            labels_for_queries_list.append(batch_label[i])
    query_src_b = torch.cat(query_src_cat_list) if query_src_cat_list else torch.empty(0, dtype=torch.long)
    query_dst_b = torch.cat(query_dst_cat_list) if query_dst_cat_list else torch.empty(0, dtype=torch.long)
    t_b = torch.cat(t_for_queries_list) if t_for_queries_list else torch.empty((0, 1), dtype=torch.long)
    label_b = torch.cat(labels_for_queries_list) if labels_for_queries_list else torch.empty(0, dtype=torch.long)

    if len(data[0]) == 12:
        return input_edge_index_b, noisy_edge_index_b, batch_x_n_cat, \
            batch_abs_level_cat, batch_rel_level_cat, \
            t_b, batch_y_tensor, \
            query_src_b, query_dst_b, label_b
    else:
        return input_edge_index_b, noisy_edge_index_b, batch_x_n_cat, \
            batch_abs_level_cat, batch_rel_level_cat, \
            t_b, \
            query_src_b, query_dst_b, label_b

