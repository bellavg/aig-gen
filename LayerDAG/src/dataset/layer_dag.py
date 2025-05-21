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
            elif not isinstance(self.input_y, torch.Tensor):  # Ensure it's a tensor if not already
                self.input_y = torch.tensor(self.input_y)
            self.input_g = torch.LongTensor(self.input_g)


class LayerDAGNodeCountDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.label = []
        self.item_to_original_graph_idx = []  # For debugging

        for i in range(len(dag_dataset)):  # Iterate over original graphs
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g_for_cond = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i
                input_g_for_cond = -1

            # Map from original graph's local node ID (0 to len(x_n)-1, and -1 for dummy)
            # to its actual global index in self.input_x_n
            original_graph_local_to_actual_global_idx_map = {}

            # Global start index in self.input_x_n for nodes of *this original graph i*
            # This is also the global index of the dummy node for this graph.
            current_graph_nodes_global_start_offset = len(self.input_x_n)

            # These track the extent of nodes and edges for the GNN input of a *single training item*.
            # An item is created for each layer of the original graph.
            # `item_input_nodes_global_end` is the current end of the GNN input node sequence for graph `i`.
            # `item_input_edges_global_end` is the current end of the GNN input edge sequence for graph `i`.
            item_input_nodes_global_end = current_graph_nodes_global_start_offset
            item_input_edges_global_end = len(self.input_src)

            # Add dummy node for this original graph `i`.
            # Its features are appended to self.input_x_n.
            # Its actual global index is `len(self.input_x_n)` before append.
            actual_global_idx_dummy = len(self.input_x_n)
            self.input_x_n.append(dag_dataset.dummy_category)
            self.input_level.append(0)
            original_graph_local_to_actual_global_idx_map[-1] = actual_global_idx_dummy  # -1 for dummy
            item_input_nodes_global_end += 1  # Dummy node is now part of this graph's processed nodes

            # Original src/dst are 0-indexed for graph `x_n`.
            # We will use these 0-indexed IDs with the map.
            # For adj list construction, we still use 1-indexed local to dummy for convenience.
            src_plus_dummy_for_adj = src + 1
            dst_plus_dummy_for_adj = dst + 1

            num_nodes_in_original_sample_incl_dummy = len(x_n) + 1
            in_deg_local_for_adj = self.get_in_deg(dst_plus_dummy_for_adj, num_nodes_in_original_sample_incl_dummy)
            x_n_list_original = x_n.tolist()
            out_adj_list_local_for_adj = self.get_out_adj_list(src_plus_dummy_for_adj.tolist(),
                                                               dst_plus_dummy_for_adj.tolist())
            in_adj_list_local_for_adj = self.get_in_adj_list(src_plus_dummy_for_adj.tolist(),
                                                             dst_plus_dummy_for_adj.tolist())

            frontiers_local_indices_for_adj = [  # These are 1-indexed relative to dummy
                u_local for u_local in range(1, num_nodes_in_original_sample_incl_dummy) if
                in_deg_local_for_adj[u_local] == 0
            ]
            frontier_size = len(frontiers_local_indices_for_adj)
            current_item_level_in_original_graph = 0

            while frontier_size > 0:
                current_item_level_in_original_graph += 1

                # Define the GNN input scope for the current item.
                # Nodes: from `current_graph_nodes_global_start_offset` up to `item_input_nodes_global_end` (exclusive).
                # Edges: from `current_graph_edges_global_start_offset` (which is fixed for graph `i`)
                #        up to `item_input_edges_global_end` (exclusive).
                # Note: item_edges_start_global was implicitly len(self.input_src) at start of graph i.
                # For clarity, let's define it explicitly.
                if current_item_level_in_original_graph == 1:  # First real layer
                    current_graph_edges_global_start_offset = len(
                        self.input_src)  # No edges before first real layer processing

                self.input_n_start.append(current_graph_nodes_global_start_offset)
                self.input_n_end.append(item_input_nodes_global_end)  # Nodes processed up to previous layer
                self.input_e_start.append(
                    current_graph_edges_global_start_offset if current_item_level_in_original_graph > 1 else item_input_edges_global_end)  # Edges within previous layers
                self.input_e_end.append(item_input_edges_global_end)  # Edges processed up to previous layer

                self.item_to_original_graph_idx.append(i)
                if conditional:
                    self.input_g.append(input_g_for_cond)
                self.label.append(frontier_size)

                # Process current frontier nodes and edges leading to them
                next_frontiers_local_indices_for_adj_buffer = []
                for u_frontier_local_idx_for_adj in frontiers_local_indices_for_adj:  # 1-indexed
                    original_node_idx_0_based = u_frontier_local_idx_for_adj - 1

                    # Add current frontier node's features to self.input_x_n
                    actual_global_idx_for_u_frontier = len(self.input_x_n)
                    self.input_x_n.append(x_n_list_original[original_node_idx_0_based])
                    self.input_level.append(current_item_level_in_original_graph)
                    original_graph_local_to_actual_global_idx_map[
                        original_node_idx_0_based] = actual_global_idx_for_u_frontier

                    # Add edges from previous layers to this frontier node
                    for t_source_local_idx_for_adj in in_adj_list_local_for_adj[u_frontier_local_idx_for_adj]:
                        actual_global_dst_idx = actual_global_idx_for_u_frontier
                        if t_source_local_idx_for_adj == 0:  # Edge from dummy
                            actual_global_src_idx = original_graph_local_to_actual_global_idx_map[-1]
                        else:  # Edge from a real predecessor node
                            original_src_node_idx_0_based = t_source_local_idx_for_adj - 1
                            actual_global_src_idx = original_graph_local_to_actual_global_idx_map[
                                original_src_node_idx_0_based]

                        self.input_src.append(actual_global_src_idx)
                        self.input_dst.append(actual_global_dst_idx)
                        item_input_edges_global_end += 1  # This edge is now part of processed edges

                    # Update in-degrees for neighbors for next frontier calculation
                    for v_target_local_idx_for_adj in out_adj_list_local_for_adj[u_frontier_local_idx_for_adj]:
                        in_deg_local_for_adj[v_target_local_idx_for_adj] -= 1
                        if in_deg_local_for_adj[v_target_local_idx_for_adj] == 0:
                            next_frontiers_local_indices_for_adj_buffer.append(v_target_local_idx_for_adj)

                    item_input_nodes_global_end += 1  # This frontier node is now processed

                frontiers_local_indices_for_adj = next_frontiers_local_indices_for_adj_buffer
                frontier_size = len(frontiers_local_indices_for_adj)

            # Termination item for original graph `i`
            self.input_n_start.append(current_graph_nodes_global_start_offset)
            self.input_n_end.append(item_input_nodes_global_end)  # All nodes of graph i
            self.input_e_start.append(current_graph_edges_global_start_offset if len(
                x_n) > 0 else item_input_edges_global_end)  # All edges of graph i
            self.input_e_end.append(item_input_edges_global_end)
            self.item_to_original_graph_idx.append(i)
            if conditional:
                self.input_g.append(input_g_for_cond)
            self.label.append(0)

        self.base_postprocess()
        self.label = torch.LongTensor(self.label)
        if len(self.label) > 0:
            self.max_layer_size = self.label.max().item()
        else:
            self.max_layer_size = 0
        self.item_to_original_graph_idx = torch.LongTensor(self.item_to_original_graph_idx)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        item_edges_start_global_idx = self.input_e_start[index]
        item_edges_end_global_idx = self.input_e_end[index]
        item_nodes_start_global_idx = self.input_n_start[index]
        item_nodes_end_global_idx = self.input_n_end[index]

        input_x_n_slice = self.input_x_n[item_nodes_start_global_idx:item_nodes_end_global_idx]
        input_abs_level_slice = self.input_level[item_nodes_start_global_idx:item_nodes_end_global_idx]

        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        src_for_item_global = self.input_src[item_edges_start_global_idx:item_edges_end_global_idx]
        dst_for_item_global = self.input_dst[item_edges_start_global_idx:item_edges_end_global_idx]

        src_reindexed_local = src_for_item_global - item_nodes_start_global_idx
        dst_reindexed_local = dst_for_item_global - item_nodes_start_global_idx

        num_nodes_in_slice = len(input_x_n_slice)

        # --- Debug Checks for __getitem__ (can be kept or removed after fixing) ---
        valid_item = True
        if src_reindexed_local.numel() > 0:
            if src_reindexed_local.max() >= num_nodes_in_slice or src_reindexed_local.min() < 0:
                valid_item = False;
                print(
                    f"ERROR __getitem__ src invalid: index={index}, max_src={src_reindexed_local.max()}, slice_len={num_nodes_in_slice}")
        if dst_reindexed_local.numel() > 0:
            if dst_reindexed_local.max() >= num_nodes_in_slice or dst_reindexed_local.min() < 0:
                valid_item = False;
                print(
                    f"ERROR __getitem__ dst invalid: index={index}, max_dst={dst_reindexed_local.max()}, slice_len={num_nodes_in_slice}")
        if not valid_item:
            # You might want to raise an error or return None if an invalid item is found,
            # depending on how you want to handle it during development.
            # For now, just printing.
            pass
        # --- End Debug Checks ---

        if self.conditional:
            input_g_idx = self.input_g[index]
            y_val_raw = self.input_y[input_g_idx]
            input_y_val_tensor = y_val_raw if isinstance(y_val_raw, torch.Tensor) else torch.tensor(y_val_raw)
            if input_y_val_tensor.ndim == 0: input_y_val_tensor = input_y_val_tensor.unsqueeze(0)
            return src_reindexed_local, dst_reindexed_local, \
                input_x_n_slice, input_abs_level_slice, input_rel_level_slice, \
                input_y_val_tensor, self.label[index]
        else:
            return src_reindexed_local, dst_reindexed_local, \
                input_x_n_slice, input_abs_level_slice, input_rel_level_slice, \
                self.label[index]


class LayerDAGNodePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False, get_marginal=True):
        super().__init__(conditional)
        self.label_start = []
        self.label_end = []
        self.node_diffusion = None
        # For debugging
        self.item_to_original_graph_idx = []

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g_for_cond = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i
                input_g_for_cond = -1

            original_graph_local_to_actual_global_idx_map = {}
            current_graph_nodes_global_start_offset = len(self.input_x_n)

            # Tracks the GNN input graph for the current item being constructed
            item_input_nodes_global_end = current_graph_nodes_global_start_offset
            item_input_edges_global_end = len(self.input_src)

            # Tracks the start of the *label* nodes (current frontier) in self.input_x_n
            # These are added *after* the GNN input nodes for the current item.
            current_item_label_nodes_global_start = -1  # Will be set when first label node is added

            # Add dummy node for this original graph `i`.
            actual_global_idx_dummy = len(self.input_x_n)
            self.input_x_n.append(dag_dataset.dummy_category)
            self.input_level.append(0)
            original_graph_local_to_actual_global_idx_map[-1] = actual_global_idx_dummy
            item_input_nodes_global_end += 1

            src_plus_dummy_for_adj = src + 1
            dst_plus_dummy_for_adj = dst + 1
            current_item_level_in_original_graph = 0

            num_nodes_in_original_sample_incl_dummy = len(x_n) + 1
            in_deg_local_for_adj = self.get_in_deg(dst_plus_dummy_for_adj, num_nodes_in_original_sample_incl_dummy)
            x_n_list_original = x_n.tolist()
            out_adj_list_local_for_adj = self.get_out_adj_list(src_plus_dummy_for_adj.tolist(),
                                                               dst_plus_dummy_for_adj.tolist())
            in_adj_list_local_for_adj = self.get_in_adj_list(src_plus_dummy_for_adj.tolist(),
                                                             dst_plus_dummy_for_adj.tolist())

            frontiers_local_indices_for_adj = [
                u_local for u_local in range(1, num_nodes_in_original_sample_incl_dummy) if
                in_deg_local_for_adj[u_local] == 0
            ]
            frontier_size = len(frontiers_local_indices_for_adj)

            while frontier_size > 0:
                current_item_level_in_original_graph += 1

                # Define GNN input scope for the current item (predicting attributes for current frontier)
                self.input_n_start.append(current_graph_nodes_global_start_offset)
                self.input_n_end.append(item_input_nodes_global_end)  # Nodes processed up to previous layer
                # Edges for GNN input
                current_graph_edges_global_start_offset_for_item = len(
                    self.input_src) if current_item_level_in_original_graph == 1 and not self.input_src else (
                    self.input_e_start[-1] if self.input_e_start else 0)  # Simplified
                if current_item_level_in_original_graph == 1 and i == 0:  # First item of first graph
                    current_graph_edges_global_start_offset_for_item = 0

                self.input_e_start.append(
                    self.input_e_end[-1] if self.input_e_end else 0)  # Start where last item's edges ended
                self.input_e_end.append(item_input_edges_global_end)  # Edges processed up to previous layer

                self.item_to_original_graph_idx.append(i)
                if conditional:
                    self.input_g.append(input_g_for_cond)

                # Record global indices for the label nodes (current frontier)
                # These label nodes are added to self.input_x_n *now*.
                current_item_label_nodes_global_start = len(self.input_x_n)
                self.label_start.append(current_item_label_nodes_global_start)

                next_frontiers_local_indices_for_adj_buffer = []
                for u_frontier_local_idx_for_adj in frontiers_local_indices_for_adj:
                    original_node_idx_0_based = u_frontier_local_idx_for_adj - 1

                    # Add current frontier node's attributes to self.input_x_n (these are the labels z)
                    actual_global_idx_for_u_frontier_label = len(self.input_x_n)
                    self.input_x_n.append(x_n_list_original[original_node_idx_0_based])
                    self.input_level.append(current_item_level_in_original_graph)
                    original_graph_local_to_actual_global_idx_map[
                        original_node_idx_0_based] = actual_global_idx_for_u_frontier_label

                    # Add edges from previous layers to this frontier node (these become part of GNN input for *next* item)
                    for t_source_local_idx_for_adj in in_adj_list_local_for_adj[u_frontier_local_idx_for_adj]:
                        actual_global_dst_idx = actual_global_idx_for_u_frontier_label  # Edge to current label node
                        if t_source_local_idx_for_adj == 0:  # Edge from dummy
                            actual_global_src_idx = original_graph_local_to_actual_global_idx_map[-1]
                        else:  # Edge from a real predecessor node
                            original_src_node_idx_0_based = t_source_local_idx_for_adj - 1
                            actual_global_src_idx = original_graph_local_to_actual_global_idx_map[
                                original_src_node_idx_0_based]

                        self.input_src.append(actual_global_src_idx)
                        self.input_dst.append(actual_global_dst_idx)
                        item_input_edges_global_end += 1

                    for v_target_local_idx_for_adj in out_adj_list_local_for_adj[u_frontier_local_idx_for_adj]:
                        in_deg_local_for_adj[v_target_local_idx_for_adj] -= 1
                        if in_deg_local_for_adj[v_target_local_idx_for_adj] == 0:
                            next_frontiers_local_indices_for_adj_buffer.append(v_target_local_idx_for_adj)

                self.label_end.append(len(self.input_x_n))  # Labels end after all current frontier nodes are added

                # Nodes of the current frontier (which were labels) become part of GNN input for the next item.
                item_input_nodes_global_end = len(self.input_x_n)

                frontiers_local_indices_for_adj = next_frontiers_local_indices_for_adj_buffer
                frontier_size = len(frontiers_local_indices_for_adj)

        self.base_postprocess()
        self.label_start = torch.LongTensor(self.label_start)
        self.label_end = torch.LongTensor(self.label_end)
        self.item_to_original_graph_idx = torch.LongTensor(self.item_to_original_graph_idx)

        if get_marginal:
            input_x_n_for_marginal = self.input_x_n
            if input_x_n_for_marginal.ndim == 1:
                input_x_n_for_marginal = input_x_n_for_marginal.unsqueeze(-1)

            num_feats = input_x_n_for_marginal.shape[-1]
            x_n_marginal = []

            num_actual_categories_per_feat = [dag_dataset.num_categories] * num_feats

            for f_idx in range(num_feats):
                input_x_n_f = input_x_n_for_marginal[:, f_idx]
                actual_nodes_mask = (input_x_n_f != dag_dataset.dummy_category)
                actual_nodes_input_x_n_f = input_x_n_f[actual_nodes_mask]
                num_actual_types_this_feat = num_actual_categories_per_feat[f_idx]
                marginal_f = torch.zeros(num_actual_types_this_feat)
                if actual_nodes_input_x_n_f.numel() > 0:
                    unique_actual_vals, counts_actual_vals = torch.unique(actual_nodes_input_x_n_f, return_counts=True)
                    for val_idx, val_actual in enumerate(unique_actual_vals):
                        if 0 <= val_actual.item() < num_actual_types_this_feat:
                            marginal_f[val_actual.item()] += counts_actual_vals[val_idx].item()
                if marginal_f.sum() > 0:
                    marginal_f /= marginal_f.sum()
                else:
                    marginal_f.fill_(1.0 / num_actual_types_this_feat if num_actual_types_this_feat > 0 else 0)
                x_n_marginal.append(marginal_f)
            self.x_n_marginal = x_n_marginal

    def __len__(self):
        return len(self.label_start)

    def __getitem__(self, index):
        if self.node_diffusion is None:
            raise RuntimeError("node_diffusion not set for LayerDAGNodePredDataset instance.")

        input_e_start_idx = self.input_e_start[index]
        input_e_end_idx = self.input_e_end[index]
        input_n_start_idx = self.input_n_start[index]
        input_n_end_idx = self.input_n_end[index]

        label_start_global_idx = self.label_start[index]
        label_end_global_idx = self.label_end[index]

        input_x_n_slice = self.input_x_n[input_n_start_idx:input_n_end_idx]
        input_abs_level_slice = self.input_level[input_n_start_idx:input_n_end_idx]
        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        src_for_item = self.input_src[input_e_start_idx:input_e_end_idx]
        dst_for_item = self.input_dst[input_e_start_idx:input_e_end_idx]
        src_reindexed = src_for_item - input_n_start_idx
        dst_reindexed = dst_for_item - input_n_start_idx

        z = self.input_x_n[label_start_global_idx:label_end_global_idx]

        if z.numel() == 0:
            num_node_features = len(self.node_diffusion.num_classes_list) if self.node_diffusion.num_classes_list else 1
            z_t = torch.empty((0, num_node_features) if num_node_features > 1 else (0,), dtype=torch.long)
            t = torch.empty((1,), dtype=torch.long)
            z_reshaped = torch.empty((0, num_node_features) if num_node_features > 1 else (0,), dtype=torch.long)
        else:
            t, z_t = self.node_diffusion.apply_noise(z)
            z_reshaped = z

        if self.conditional:
            input_g_idx = self.input_g[index]
            y_val_raw = self.input_y[input_g_idx]
            input_y_val_tensor = y_val_raw if isinstance(y_val_raw, torch.Tensor) else torch.tensor(y_val_raw)
            if input_y_val_tensor.ndim == 0: input_y_val_tensor = input_y_val_tensor.unsqueeze(0)
            return src_reindexed, dst_reindexed, \
                input_x_n_slice, input_abs_level_slice, input_rel_level_slice, \
                z_t, t, input_y_val_tensor, z_reshaped
        else:
            return src_reindexed, dst_reindexed, \
                input_x_n_slice, input_abs_level_slice, input_rel_level_slice, \
                z_t, t, z_reshaped


class LayerDAGEdgePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.query_src_list = []
        self.query_dst_list = []
        self.label_list = []
        self.query_start_indices = []
        self.query_end_indices = []
        self.edge_diffusion = None
        self.item_to_original_graph_idx = []

        num_total_edges_processed_in_original_graphs = 0
        num_total_nonsrc_nodes_in_original_graphs = 0

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g_for_cond = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i
                input_g_for_cond = -1

            num_total_edges_processed_in_original_graphs += len(src)
            original_graph_local_to_actual_global_idx_map = {}
            current_graph_nodes_global_start_offset = len(self.input_x_n)

            item_input_nodes_global_end = current_graph_nodes_global_start_offset
            item_input_edges_global_end = len(self.input_src)
            current_item_query_list_start_idx = len(self.query_src_list)

            actual_global_idx_dummy = len(self.input_x_n)
            self.input_x_n.append(dag_dataset.dummy_category)
            self.input_level.append(0)
            original_graph_local_to_actual_global_idx_map[-1] = actual_global_idx_dummy
            item_input_nodes_global_end += 1

            src_plus_dummy_for_adj = src + 1
            dst_plus_dummy_for_adj = dst + 1
            current_item_level_in_original_graph = 0

            num_nodes_in_original_sample_incl_dummy = len(x_n) + 1
            in_deg_local_for_adj = self.get_in_deg(dst_plus_dummy_for_adj, num_nodes_in_original_sample_incl_dummy)
            x_n_list_original = x_n.tolist()
            out_adj_list_local_for_adj = self.get_out_adj_list(src_plus_dummy_for_adj.tolist(),
                                                               dst_plus_dummy_for_adj.tolist())
            in_adj_list_local_for_adj = self.get_in_adj_list(src_plus_dummy_for_adj.tolist(),
                                                             dst_plus_dummy_for_adj.tolist())

            potential_source_nodes_local_0_indexed_prev_layer = [-1]  # Start with dummy node (local ID -1)

            current_item_level_in_original_graph += 1  # First real layer is level 1
            # Process the first true frontier
            first_frontier_local_indices_for_adj = [
                u_local for u_local in range(1, num_nodes_in_original_sample_incl_dummy) if
                in_deg_local_for_adj[u_local] == 0
            ]

            for u_node_local_adj_idx in first_frontier_local_indices_for_adj:  # 1-indexed
                original_node_idx_0_based = u_node_local_adj_idx - 1
                actual_global_idx_for_u = len(self.input_x_n)
                self.input_x_n.append(x_n_list_original[original_node_idx_0_based])
                self.input_level.append(current_item_level_in_original_graph)
                original_graph_local_to_actual_global_idx_map[original_node_idx_0_based] = actual_global_idx_for_u
            item_input_nodes_global_end += len(first_frontier_local_indices_for_adj)
            potential_source_nodes_local_0_indexed_prev_layer.extend(
                [idx - 1 for idx in first_frontier_local_indices_for_adj])

            current_frontiers_nodes_local_indices_for_adj_next_layer = []
            for u_node_local_adj_idx in first_frontier_local_indices_for_adj:
                for v_target_local_adj_idx in out_adj_list_local_for_adj[u_node_local_adj_idx]:
                    in_deg_local_for_adj[v_target_local_adj_idx] -= 1
                    if in_deg_local_for_adj[v_target_local_adj_idx] == 0:
                        current_frontiers_nodes_local_indices_for_adj_next_layer.append(v_target_local_adj_idx)
            current_frontiers_nodes_local_indices_for_adj_next_layer = list(
                set(current_frontiers_nodes_local_indices_for_adj_next_layer))

            while len(current_frontiers_nodes_local_indices_for_adj_next_layer) > 0:
                current_item_level_in_original_graph += 1

                self.input_n_start.append(current_graph_nodes_global_start_offset)
                self.input_n_end.append(item_input_nodes_global_end)
                self.input_e_start.append(self.input_e_end[-1] if self.input_e_end else 0)
                self.input_e_end.append(item_input_edges_global_end)
                self.item_to_original_graph_idx.append(i)
                if conditional: self.input_g.append(input_g_for_cond)

                self.query_start_indices.append(current_item_query_list_start_idx)
                num_queries_this_item = 0
                next_frontiers_after_current_local_indices_buffer = []

                for u_target_local_adj_idx in current_frontiers_nodes_local_indices_for_adj_next_layer:  # 1-indexed
                    original_target_node_idx_0_based = u_target_local_adj_idx - 1
                    actual_global_idx_for_u_target = len(self.input_x_n)
                    self.input_x_n.append(x_n_list_original[original_target_node_idx_0_based])
                    self.input_level.append(current_item_level_in_original_graph)
                    original_graph_local_to_actual_global_idx_map[
                        original_target_node_idx_0_based] = actual_global_idx_for_u_target
                    num_total_nonsrc_nodes_in_original_graphs += 1

                    for t_source_orig_0_indexed_idx in potential_source_nodes_local_0_indexed_prev_layer:
                        actual_global_src_idx = original_graph_local_to_actual_global_idx_map[
                            t_source_orig_0_indexed_idx]
                        self.query_src_list.append(actual_global_src_idx)
                        self.query_dst_list.append(actual_global_idx_for_u_target)
                        num_queries_this_item += 1

                        # Check original connectivity (using adj-based local indices)
                        t_source_local_adj_idx = t_source_orig_0_indexed_idx + 1 if t_source_orig_0_indexed_idx != -1 else 0
                        if t_source_local_adj_idx in in_adj_list_local_for_adj[u_target_local_adj_idx]:
                            self.label_list.append(1)
                            self.input_src.append(actual_global_src_idx)
                            self.input_dst.append(actual_global_idx_for_u_target)
                            item_input_edges_global_end += 1
                        else:
                            self.label_list.append(0)

                    for v_next_target_local_adj_idx in out_adj_list_local_for_adj[u_target_local_adj_idx]:
                        in_deg_local_for_adj[v_next_target_local_adj_idx] -= 1
                        if in_deg_local_for_adj[v_next_target_local_adj_idx] == 0:
                            next_frontiers_after_current_local_indices_buffer.append(v_next_target_local_adj_idx)

                item_input_nodes_global_end += len(current_frontiers_nodes_local_indices_for_adj_next_layer)
                current_item_query_list_start_idx += num_queries_this_item
                self.query_end_indices.append(current_item_query_list_start_idx)

                potential_source_nodes_local_0_indexed_prev_layer.extend(
                    [idx - 1 for idx in current_frontiers_nodes_local_indices_for_adj_next_layer])
                current_frontiers_nodes_local_indices_for_adj_next_layer = list(
                    set(next_frontiers_after_current_local_indices_buffer))

        self.base_postprocess()
        self.query_src_list = torch.LongTensor(self.query_src_list)
        self.query_dst_list = torch.LongTensor(self.query_dst_list)
        self.label_list = torch.LongTensor(self.label_list)
        self.query_start_indices = torch.LongTensor(self.query_start_indices)
        self.query_end_indices = torch.LongTensor(self.query_end_indices)
        self.item_to_original_graph_idx = torch.LongTensor(self.item_to_original_graph_idx)

        if num_total_nonsrc_nodes_in_original_graphs > 0:
            self.avg_in_deg = num_total_edges_processed_in_original_graphs / num_total_nonsrc_nodes_in_original_graphs
        else:
            self.avg_in_deg = 0.0

    def __len__(self):
        return len(self.query_start_indices)

    def __getitem__(self, index):
        if self.edge_diffusion is None:
            raise RuntimeError("edge_diffusion not set for LayerDAGEdgePredDataset instance.")

        item_edges_start_global_idx = self.input_e_start[index]
        item_edges_end_global_idx = self.input_e_end[index]
        item_nodes_start_global_idx = self.input_n_start[index]
        item_nodes_end_global_idx = self.input_n_end[index]

        input_x_n_slice = self.input_x_n[item_nodes_start_global_idx:item_nodes_end_global_idx]
        input_abs_level_slice = self.input_level[item_nodes_start_global_idx:item_nodes_end_global_idx]
        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        src_for_item_gnn_input_global = self.input_src[item_edges_start_global_idx:item_edges_end_global_idx]
        dst_for_item_gnn_input_global = self.input_dst[item_edges_start_global_idx:item_edges_end_global_idx]
        src_reindexed_gnn_input = src_for_item_gnn_input_global - item_nodes_start_global_idx
        dst_reindexed_gnn_input = dst_for_item_gnn_input_global - item_nodes_start_global_idx

        query_list_start_for_item = self.query_start_indices[index]
        query_list_end_for_item = self.query_end_indices[index]

        query_src_global_for_item = self.query_src_list[query_list_start_for_item:query_list_end_for_item]
        query_dst_global_for_item = self.query_dst_list[query_list_start_for_item:query_list_end_for_item]
        label_for_query_edges = self.label_list[query_list_start_for_item:query_list_end_for_item]

        query_src_reindexed_local = query_src_global_for_item - item_nodes_start_global_idx
        query_dst_reindexed_local = query_dst_global_for_item - item_nodes_start_global_idx

        if label_for_query_edges.numel() > 0:
            num_candidate_sources_for_marginal_approx = len(input_x_n_slice)
            t_scalar, label_t_flat = self.edge_diffusion.apply_noise(
                label_for_query_edges,
                num_candidate_sources_for_marginal=num_candidate_sources_for_marginal_approx
            )
            t_tensor = torch.full((label_for_query_edges.shape[0], 1),
                                  t_scalar.item() if isinstance(t_scalar, torch.Tensor) else t_scalar, dtype=torch.long)
            noisy_mask = (label_t_flat == 1)
            noisy_src_reindexed = query_src_reindexed_local[noisy_mask]
            noisy_dst_reindexed = query_dst_reindexed_local[noisy_mask]
        else:
            t_tensor = torch.empty((0, 1), dtype=torch.long)
            noisy_src_reindexed = torch.empty_like(query_src_reindexed_local)
            noisy_dst_reindexed = torch.empty_like(query_dst_reindexed_local)

        if self.conditional:
            input_g_idx = self.input_g[index]
            y_val_raw = self.input_y[input_g_idx]
            input_y_val_tensor = y_val_raw if isinstance(y_val_raw, torch.Tensor) else torch.tensor(y_val_raw)
            if input_y_val_tensor.ndim == 0: input_y_val_tensor = input_y_val_tensor.unsqueeze(0)
            return src_reindexed_gnn_input, dst_reindexed_gnn_input, \
                noisy_src_reindexed, noisy_dst_reindexed, \
                input_x_n_slice, input_abs_level_slice, input_rel_level_slice, \
                t_tensor, input_y_val_tensor, \
                query_src_reindexed_local, query_dst_reindexed_local, label_for_query_edges
        else:
            return src_reindexed_gnn_input, dst_reindexed_gnn_input, \
                noisy_src_reindexed, noisy_dst_reindexed, \
                input_x_n_slice, input_abs_level_slice, input_rel_level_slice, \
                t_tensor, \
                query_src_reindexed_local, query_dst_reindexed_local, label_for_query_edges


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

    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in x_n]), dim=0)
    batch_size = len(x_n)

    src_ = []
    dst_ = []
    for i in range(batch_size):
        src_i = src[i] if isinstance(src[i], torch.Tensor) else torch.LongTensor(src[i])
        dst_i = dst[i] if isinstance(dst[i], torch.Tensor) else torch.LongTensor(dst[i])
        if src_i.numel() > 0 or dst_i.numel() > 0:
            src_i_flat = src_i.view(-1)
            dst_i_flat = dst_i.view(-1)
            src_.append(src_i_flat + num_nodes_cumsum[i])
            dst_.append(dst_i_flat + num_nodes_cumsum[i])

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
        num_elements = 7 if (len(data) > 0 and len(data[0]) == 7) else 6  # Check based on expected structure
        empty_batch = [
            0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float),
        ]
        if num_elements == 7:
            empty_batch.append(torch.empty((0, 1), dtype=torch.float))
        empty_batch.extend([
            torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
        ])
        return tuple(empty_batch)

    is_conditional_item = (len(data[0]) == 7)

    if is_conditional_item:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_y_list, batch_label = map(list,
                                                                                                           zip(*data))
        batch_y_tensorized = [y.clone().detach() if isinstance(y, torch.Tensor) else torch.tensor(y) for y in
                              batch_y_list]
        try:
            batch_y_tensor = torch.stack(batch_y_tensorized)
        except RuntimeError:
            batch_y_tensor = torch.stack([y.unsqueeze(0) if y.ndim == 0 else y for y in batch_y_tensorized])
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_label = map(list, zip(*data))
        batch_y_tensor = None

    batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, batch_rel_level_cat, \
        batch_n2g_index = collate_common(
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level
    )

    if batch_label and isinstance(batch_label[0], torch.Tensor):
        batch_label_stacked = torch.stack(batch_label)
    elif batch_label:
        batch_label_stacked = torch.LongTensor(batch_label)
    else:
        batch_label_stacked = torch.empty(0, dtype=torch.long)

    if is_conditional_item:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_y_tensor, batch_n2g_index, batch_label_stacked
    else:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_label_stacked


def collate_node_pred(data):
    data = [d for d in data if d is not None]
    if not data:
        num_elements = 9 if (len(data) > 0 and len(data[0]) == 9) else 8
        empty_batch = [
            0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float),
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.long),
        ]
        if num_elements == 9:
            empty_batch.append(torch.empty((0, 0), dtype=torch.float))
        empty_batch.extend([
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.long)
        ])
        return tuple(empty_batch)

    is_conditional_item = (len(data[0]) == 9)

    if is_conditional_item:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, \
            batch_z_t, batch_t, batch_y_list, batch_z = map(list, zip(*data))
        batch_y_tensorized = [y.clone().detach() if isinstance(y, torch.Tensor) else torch.tensor(y) for y in
                              batch_y_list]
        try:
            batch_y_tensor = torch.stack(batch_y_tensorized)
        except RuntimeError:
            batch_y_tensor = torch.stack([y.unsqueeze(0) if y.ndim == 0 else y for y in batch_y_tensorized])
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

    z_t_feat_dim = 1
    if batch_z_t and any(zt.numel() > 0 for zt in batch_z_t):
        first_valid_zt = next(zt for zt in batch_z_t if zt.numel() > 0)
        z_t_feat_dim = first_valid_zt.shape[1] if first_valid_zt.ndim > 1 else 1

    z_feat_dim = 1
    if batch_z and any(z_val.numel() > 0 for z_val in batch_z):
        first_valid_z = next(z_val for z_val in batch_z if z_val.numel() > 0)
        z_feat_dim = first_valid_z.shape[1] if first_valid_z.ndim > 1 else 1

    batch_z_t_processed = []
    for zt in batch_z_t:
        if zt.numel() == 0:
            batch_z_t_processed.append(torch.empty((0, z_t_feat_dim), dtype=zt.dtype))
        elif zt.ndim == 1:
            batch_z_t_processed.append(zt.unsqueeze(-1) if z_t_feat_dim == 1 else zt.view(len(zt), -1))
        else:
            batch_z_t_processed.append(zt)
    batch_z_t_cat = torch.cat(batch_z_t_processed) if batch_z_t_processed else torch.empty((0, z_t_feat_dim),
                                                                                           dtype=torch.long)

    batch_t_processed = [t.squeeze() if t.numel() > 0 else torch.empty(0, dtype=torch.long) for t in batch_t]
    batch_t_cat = torch.cat(batch_t_processed).unsqueeze(-1) if any(
        t.numel() > 0 for t in batch_t_processed) else torch.empty((0, 1), dtype=torch.long)

    batch_z_processed = []
    for z_val in batch_z:
        if z_val.numel() == 0:
            batch_z_processed.append(torch.empty((0, z_feat_dim), dtype=z_val.dtype))
        elif z_val.ndim == 1:
            batch_z_processed.append(z_val.unsqueeze(-1) if z_feat_dim == 1 else z_val.view(len(z_val), -1))
        else:
            batch_z_processed.append(z_val)
    batch_z_cat = torch.cat(batch_z_processed) if batch_z_processed else torch.empty((0, z_feat_dim), dtype=torch.long)

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

    if is_conditional_item:
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
        num_elements = 12 if (len(data) > 0 and len(data[0]) == 12) else 11
        empty_batch = [
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float),
            torch.empty((0, 1), dtype=torch.float),
            torch.empty((0, 1), dtype=torch.long),
        ]
        if num_elements == 12:
            empty_batch.append(torch.empty((0, 0), dtype=torch.float))
        empty_batch.extend([
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long)
        ])
        return tuple(empty_batch)

    is_conditional_item = (len(data[0]) == 12)

    if is_conditional_item:
        batch_input_src, batch_input_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, \
            batch_abs_level, batch_rel_level, batch_t_list, batch_y_list, \
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        batch_y_tensorized = [y.clone().detach() if isinstance(y, torch.Tensor) else torch.tensor(y) for y in
                              batch_y_list]
        try:
            batch_y_tensor = torch.stack(batch_y_tensorized)
        except RuntimeError:
            batch_y_tensor = torch.stack([y.unsqueeze(0) if y.ndim == 0 else y for y in batch_y_tensorized])
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)
    elif len(data[0]) == 11:
        batch_input_src, batch_input_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, \
            batch_abs_level, batch_rel_level, batch_t_list, \
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        batch_y_tensor = None
    else:
        raise ValueError(f"Unexpected number of items in data for collate_edge_pred: {len(data[0])}")

    valid_indices_nodes = [i for i, item_x_n in enumerate(batch_x_n) if item_x_n is not None and len(item_x_n) > 0]
    if not valid_indices_nodes:
        return collate_edge_pred([])

    batch_input_src = [batch_input_src[i] for i in valid_indices_nodes]
    batch_input_dst = [batch_input_dst[i] for i in valid_indices_nodes]
    batch_noisy_src = [batch_noisy_src[i] for i in valid_indices_nodes]
    batch_noisy_dst = [batch_noisy_dst[i] for i in valid_indices_nodes]
    batch_x_n = [batch_x_n[i] for i in valid_indices_nodes]
    batch_abs_level = [batch_abs_level[i] for i in valid_indices_nodes]
    batch_rel_level = [batch_rel_level[i] for i in valid_indices_nodes]
    batch_t_list = [batch_t_list[i] for i in valid_indices_nodes]
    if is_conditional_item:
        batch_y_tensor = batch_y_tensor[valid_indices_nodes] if batch_y_tensor is not None else None
    batch_query_src = [batch_query_src[i] for i in valid_indices_nodes]
    batch_query_dst = [batch_query_dst[i] for i in valid_indices_nodes]
    batch_label = [batch_label[i] for i in valid_indices_nodes]

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

    if is_conditional_item:
        return input_edge_index_b, noisy_edge_index_b, batch_x_n_cat, \
            batch_abs_level_cat, batch_rel_level_cat, \
            t_b, batch_y_tensor, \
            query_src_b, query_dst_b, label_b
    else:
        return input_edge_index_b, noisy_edge_index_b, batch_x_n_cat, \
            batch_abs_level_cat, batch_rel_level_cat, \
            t_b, \
            query_src_b, query_dst_b, label_b


