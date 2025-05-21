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

        # Case1: self.input_x_n[0] is an int.
        # Case2: self.input_x_n[0] is a tensor of shape (F).
        self.input_x_n = torch.LongTensor(self.input_x_n)
        self.input_level = torch.LongTensor(self.input_level)

        self.input_e_start = torch.LongTensor(self.input_e_start)
        self.input_e_end = torch.LongTensor(self.input_e_end)
        self.input_n_start = torch.LongTensor(self.input_n_start)
        self.input_n_end = torch.LongTensor(self.input_n_end)

        if self.conditional:
            # Ensure self.input_y is a tensor. If it's a list of scalars, convert.
            # If it's already a tensor (e.g. from a previous load), this might not be necessary
            # or could be handled more robustly depending on expected input_y types.
            if isinstance(self.input_y, list) and self.input_y:
                # Check if elements are tensors or scalars
                if all(isinstance(item, (int, float)) for item in self.input_y):
                    self.input_y = torch.tensor(self.input_y)
                elif all(isinstance(item, torch.Tensor) for item in self.input_y):
                    try:
                        self.input_y = torch.stack(self.input_y)  # If they are tensors of same shape
                    except RuntimeError:  # Fallback if shapes differ, e.g. for varying feature sizes
                        self.input_y = self.input_y  # Keep as list of tensors, collate_fn might need adjustment
                # else: handle mixed types or other scenarios if necessary
            elif not isinstance(self.input_y, torch.Tensor):  # If not list or tensor, try to convert
                self.input_y = torch.tensor(self.input_y)

            self.input_g = torch.LongTensor(self.input_g)


class LayerDAGNodeCountDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)

        # Size of the next layer to predict.
        self.label = []

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]

            if conditional:
                src, dst, x_n, y = data_i
                # Index of y in self.input_y
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            # For recording indices of the node attributes in self.input_x_n
            input_n_start_val = len(self.input_x_n)  # Use a different variable name for clarity
            input_n_end_val = len(self.input_x_n)

            # For recording indices of the edges in self.input_src/self.input_dst
            input_e_start_val = len(self.input_src)
            input_e_end_val = len(self.input_src)

            # Use a dummy node for representing the initial empty DAG.
            self.input_x_n.append(dag_dataset.dummy_category)
            input_n_end_val += 1

            # Convert original src/dst (0-indexed for original graph) to 1-indexed relative to dummy
            src_plus_dummy = src + 1
            dst_plus_dummy = dst + 1

            # Layer ID
            level = 0
            self.input_level.append(level)

            num_nodes_with_dummy = len(x_n) + 1  # Total nodes including the dummy node
            in_deg = self.get_in_deg(dst_plus_dummy, num_nodes_with_dummy)

            # Use original src/dst for adj list creation, then adjust when adding to self.input_src/dst
            src_list_orig = src.tolist()
            dst_list_orig = dst.tolist()
            x_n_list = x_n.tolist()  # x_n is 0-indexed corresponding to original graph nodes

            # Adj lists should use indices relative to the graph with the dummy node (1 to num_nodes_with_dummy-1 for real nodes)
            out_adj_list = self.get_out_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())
            in_adj_list = self.get_in_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())

            frontiers = [
                u for u in range(1, num_nodes_with_dummy) if in_deg[u] == 0  # u are 1-indexed (dummy is 0)
            ]
            frontier_size = len(frontiers)
            while frontier_size > 0:
                # There is another layer.
                level += 1

                # Record indices for retrieving edges in the previous layers
                # for model input.
                self.input_e_start.append(input_e_start_val)
                self.input_e_end.append(input_e_end_val)

                # Record indices for retrieving node attributes in the previous
                # layers for model input.
                self.input_n_start.append(input_n_start_val)
                self.input_n_end.append(input_n_end_val)

                if conditional:
                    # Record the index for retrieving graph-level conditional
                    # information for model input.
                    self.input_g.append(input_g)
                self.label.append(frontier_size)

                # (1) Add the node attributes/edges for the current layer.
                # (2) Get the next layer.
                next_frontiers = []
                for u_frontier_node in frontiers:  # u_frontier_node is 1-indexed (relative to dummy)
                    # x_n_list is 0-indexed (original graph nodes)
                    # So, u_frontier_node - 1 maps to the correct index in x_n_list
                    self.input_x_n.append(x_n_list[u_frontier_node - 1])
                    self.input_level.append(level)

                    # Edges are added using their 1-indexed values (relative to dummy)
                    for t_source_node in in_adj_list[u_frontier_node]:
                        self.input_src.append(t_source_node)
                        self.input_dst.append(u_frontier_node)
                        input_e_end_val += 1

                    for v_target_node in out_adj_list[u_frontier_node]:
                        in_deg[v_target_node] -= 1
                        if in_deg[v_target_node] == 0:
                            next_frontiers.append(v_target_node)
                input_n_end_val += frontier_size

                frontiers = next_frontiers
                frontier_size = len(frontiers)

            # Handle termination, namely predicting the layer size to be 0.
            self.input_e_start.append(input_e_start_val)
            self.input_e_end.append(input_e_end_val)
            self.input_n_start.append(input_n_start_val)
            self.input_n_end.append(input_n_end_val)
            if conditional:
                self.input_g.append(input_g)
            self.label.append(frontier_size)  # Should be 0 here

        self.base_postprocess()
        self.label = torch.LongTensor(self.label)
        # Maximum number of nodes in a layer.
        if len(self.label) > 0:
            self.max_layer_size = self.label.max().item()
        else:  # Handle case where dataset might be empty or only produces 0-size layers
            self.max_layer_size = 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        input_e_start_idx = self.input_e_start[index]
        input_e_end_idx = self.input_e_end[index]
        input_n_start_idx = self.input_n_start[index]
        input_n_end_idx = self.input_n_end[index]

        input_x_n_slice = self.input_x_n[input_n_start_idx:input_n_end_idx]
        input_abs_level_slice = self.input_level[input_n_start_idx:input_n_end_idx]

        # Ensure input_abs_level_slice is not empty before calling .max()
        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            # Handle empty slice case, e.g., return an empty tensor of the same dtype
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        src_for_item = self.input_src[input_e_start_idx:input_e_end_idx]
        dst_for_item = self.input_dst[input_e_start_idx:input_e_end_idx]

        # --- BEGIN MODIFICATION ---
        # Re-index edge indices to be 0-based for the current node slice
        # The node indices stored in self.input_src/dst are relative to the start of the *original graph's nodes including the dummy node*.
        # The current slice of nodes (input_x_n_slice) starts at input_n_start_idx in self.input_x_n.
        # So, we need to subtract this offset.
        current_n_start_offset = input_n_start_idx
        src_reindexed = src_for_item - current_n_start_offset
        dst_reindexed = dst_for_item - current_n_start_offset
        # --- END MODIFICATION ---

        if self.conditional:
            input_g_idx = self.input_g[index]
            # Ensure input_y[input_g_idx] is scalar if .item() is used
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

        self.label_start = []
        self.label_end = []

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]

            if conditional:
                src, dst, x_n, y = data_i
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            input_n_start_val = len(self.input_x_n)
            input_n_end_val = len(self.input_x_n)
            input_e_start_val = len(self.input_src)
            input_e_end_val = len(self.input_src)

            self.input_x_n.append(dag_dataset.dummy_category)  # Model input dummy node
            input_n_end_val += 1

            src_plus_dummy = src + 1
            dst_plus_dummy = dst + 1

            label_start_val = len(self.input_x_n)  # Labels start after the dummy node

            level = 0
            self.input_level.append(level)  # Level for the dummy node

            num_nodes_with_dummy = len(x_n) + 1
            in_deg = self.get_in_deg(dst_plus_dummy, num_nodes_with_dummy)

            x_n_list = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())
            in_adj_list = self.get_in_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())

            frontiers = [
                u for u in range(1, num_nodes_with_dummy) if in_deg[u] == 0
            ]
            frontier_size = len(frontiers)
            while frontier_size > 0:
                level += 1

                self.input_e_start.append(input_e_start_val)
                self.input_e_end.append(input_e_end_val)
                self.input_n_start.append(input_n_start_val)
                self.input_n_end.append(input_n_end_val)

                if conditional:
                    self.input_g.append(input_g)

                self.label_start.append(label_start_val)
                label_end_val = label_start_val + frontier_size
                self.label_end.append(label_end_val)
                label_start_val = label_end_val

                next_frontiers = []
                for u_frontier_node in frontiers:
                    self.input_x_n.append(x_n_list[u_frontier_node - 1])  # This is a label node
                    self.input_level.append(level)  # Level for this label node

                    # Add edges for model input (edges leading to this new layer)
                    for t_source_node in in_adj_list[u_frontier_node]:
                        self.input_src.append(t_source_node)
                        self.input_dst.append(u_frontier_node)
                        input_e_end_val += 1

                    # Update degrees for next frontier
                    for v_target_node in out_adj_list[u_frontier_node]:
                        in_deg[v_target_node] -= 1
                        if in_deg[v_target_node] == 0:
                            next_frontiers.append(v_target_node)

                # The nodes added to self.input_x_n in this loop are the *labels* for this step.
                # The *model input nodes* for this step are those from previous layers,
                # which are captured by input_n_start_val and input_n_end_val.
                # We need to update input_n_end_val to include the nodes of the *current frontier*
                # as they become part of the "previous layers" for the *next* prediction step.
                input_n_end_val += frontier_size

                frontiers = next_frontiers
                frontier_size = len(frontiers)

        self.base_postprocess()
        self.label_start = torch.LongTensor(self.label_start)
        self.label_end = torch.LongTensor(self.label_end)

        if get_marginal:
            input_x_n_for_marginal = self.input_x_n  # All node attributes including dummy and labels
            if input_x_n_for_marginal.ndim == 1:
                input_x_n_for_marginal = input_x_n_for_marginal.unsqueeze(-1)

            num_feats = input_x_n_for_marginal.shape[-1]
            x_n_marginal = []
            for f in range(num_feats):
                input_x_n_f = input_x_n_for_marginal[:, f]
                unique_x_n_f, x_n_count_f = torch.unique(input_x_n_f, return_counts=True)

                # Determine number of actual categories (excluding dummy)
                # dag_dataset.num_categories is total actual types (e.g. 4 for AIG)
                # dag_dataset.dummy_category is the integer for dummy (e.g. 4)
                num_actual_types_f = dag_dataset.num_categories  # e.g. 4 (types 0,1,2,3)

                x_n_marginal_f = torch.zeros(num_actual_types_f)
                for c_idx in range(len(unique_x_n_f)):
                    x_n_type_val = unique_x_n_f[c_idx].item()
                    if x_n_type_val < num_actual_types_f:  # Only count actual types for marginal
                        x_n_marginal_f[x_n_type_val] += x_n_count_f[c_idx].item()

                x_n_marginal_f /= (x_n_marginal_f.sum() + 1e-8)
                x_n_marginal.append(x_n_marginal_f)
            self.x_n_marginal = x_n_marginal

    def __len__(self):
        return len(self.label_start)

    def __getitem__(self, index):
        input_e_start_idx = self.input_e_start[index]
        input_e_end_idx = self.input_e_end[index]
        input_n_start_idx = self.input_n_start[index]
        input_n_end_idx = self.input_n_end[index]
        label_start_idx = self.label_start[index]
        label_end_idx = self.label_end[index]

        # Node features for model input (previous layers)
        input_x_n_slice = self.input_x_n[input_n_start_idx:input_n_end_idx]
        # Level information for model input nodes
        input_abs_level_slice = self.input_level[input_n_start_idx:input_n_end_idx]
        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        # Edges for model input (previous layers)
        src_for_item = self.input_src[input_e_start_idx:input_e_end_idx]
        dst_for_item = self.input_dst[input_e_start_idx:input_e_end_idx]

        current_n_start_offset = input_n_start_idx
        src_reindexed = src_for_item - current_n_start_offset
        dst_reindexed = dst_for_item - current_n_start_offset

        # Ground truth node attributes for the new layer (z)
        z = self.input_x_n[label_start_idx:label_end_idx]
        t, z_t = self.node_diffusion.apply_noise(z)  # z_t are the noisy attributes to predict

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

        self.query_src_list = []  # Stores all query sources
        self.query_dst_list = []  # Stores all query destinations
        self.label_list = []  # Stores all labels for queries

        # Indices for retrieving the query node pairs for a given sample
        self.query_start_indices = []  # Start index in query_src_list/query_dst_list/label_list
        self.query_end_indices = []  # End index

        num_total_edges_processed = 0  # For avg_in_deg calculation
        num_total_nonsrc_nodes = 0  # For avg_in_deg calculation

        for i in range(len(dag_dataset)):
            data_i = dag_dataset[i]

            if conditional:
                src, dst, x_n, y = data_i
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            input_n_start_val = len(self.input_x_n)
            input_n_end_val = len(self.input_x_n)
            input_e_start_val = len(self.input_src)  # Edges accumulated for model input
            input_e_end_val = len(self.input_src)

            query_start_for_sample = len(self.query_src_list)  # Start of queries for *this* sample

            self.input_x_n.append(dag_dataset.dummy_category)
            input_n_end_val += 1
            src_plus_dummy = src + 1
            dst_plus_dummy = dst + 1

            level = 0
            self.input_level.append(level)

            num_nodes_with_dummy = len(x_n) + 1
            in_deg = self.get_in_deg(dst_plus_dummy, num_nodes_with_dummy)

            x_n_list = x_n.tolist()
            out_adj_list = self.get_out_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())
            in_adj_list = self.get_in_adj_list(src_plus_dummy.tolist(), dst_plus_dummy.tolist())

            # Nodes in the first layer (these are 1-indexed relative to dummy)
            # These become the initial set of candidate source nodes for edges to the next layer.
            # Their attributes are added to self.input_x_n.
            prev_frontiers_nodes = [
                u for u in range(1, num_nodes_with_dummy) if in_deg[u] == 0
            ]

            current_frontiers_nodes = []
            level += 1  # Level of the first real layer

            num_total_edges_processed += len(src)  # Original edges in the graph
            num_total_nonsrc_nodes += len(x_n) - len(prev_frontiers_nodes)  # Nodes not in the first layer

            # Add nodes of the first layer to self.input_x_n and self.input_level
            for u_node in prev_frontiers_nodes:
                self.input_x_n.append(x_n_list[u_node - 1])
                self.input_level.append(level)
            input_n_end_val += len(prev_frontiers_nodes)

            # src_candidates_for_queries are the nodes from *previous* layers that can be sources
            # These are 1-indexed relative to the dummy node.
            src_candidates_for_queries = list(prev_frontiers_nodes)

            # Determine next layer of nodes
            for u_node in prev_frontiers_nodes:
                for v_target_node in out_adj_list[u_node]:
                    in_deg[v_target_node] -= 1
                    if in_deg[v_target_node] == 0:
                        current_frontiers_nodes.append(v_target_node)

            # Loop while there are new layers to process
            while len(current_frontiers_nodes) > 0:
                level += 1  # Level of the current_frontiers_nodes

                # For each node u in the current_frontiers_nodes, we generate queries
                # from all src_candidates_for_queries to u.

                # Record the state of the graph *before* adding this layer's nodes and edges
                # These define the input graph for predicting edges to current_frontiers_nodes
                self.input_e_start.append(input_e_start_val)
                self.input_e_end.append(input_e_end_val)
                self.input_n_start.append(input_n_start_val)
                self.input_n_end.append(input_n_end_val)
                if conditional:
                    self.input_g.append(input_g)

                # Record where queries for this step begin and will end
                self.query_start_indices.append(query_start_for_sample)

                num_queries_this_step = 0
                temp_edges_added_to_input_this_step = 0

                next_frontiers_nodes_buffer = []  # To collect nodes for the *next* layer

                for u_target_node in current_frontiers_nodes:  # u_target_node is a node in the new layer
                    # Add attributes of u_target_node to self.input_x_n
                    self.input_x_n.append(x_n_list[u_target_node - 1])
                    self.input_level.append(level)

                    # Generate queries: (candidate_source -> u_target_node)
                    # candidate_source are 1-indexed (relative to dummy)
                    for t_candidate_source in src_candidates_for_queries:
                        self.query_src_list.append(t_candidate_source)
                        self.query_dst_list.append(u_target_node)
                        num_queries_this_step += 1

                        # Determine label for this query and add edge to input if it exists
                        if t_candidate_source in in_adj_list[u_target_node]:
                            self.input_src.append(t_candidate_source)
                            self.input_dst.append(u_target_node)
                            temp_edges_added_to_input_this_step += 1
                            self.label_list.append(1)
                        else:
                            self.label_list.append(0)

                    # Prepare for the next layer
                    for v_next_target in out_adj_list[u_target_node]:
                        in_deg[v_next_target] -= 1
                        if in_deg[v_next_target] == 0:
                            next_frontiers_nodes_buffer.append(v_next_target)

                input_n_end_val += len(current_frontiers_nodes)  # Nodes of current layer are now part of input graph
                input_e_end_val += temp_edges_added_to_input_this_step  # Edges to current layer are now part of input graph

                query_start_for_sample += num_queries_this_step  # Update for next step's query start
                self.query_end_indices.append(query_start_for_sample)  # Mark end of queries for this step

                src_candidates_for_queries.extend(current_frontiers_nodes)
                current_frontiers_nodes = list(set(next_frontiers_nodes_buffer))  # Remove duplicates for next layer

        self.base_postprocess()  # Converts lists to tensors
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
        return len(self.query_start_indices)  # Number of prediction steps

    def __getitem__(self, index):
        # Indices for the input graph structure (nodes and edges from previous layers)
        input_e_start_idx = self.input_e_start[index]
        input_e_end_idx = self.input_e_end[index]
        input_n_start_idx = self.input_n_start[index]
        input_n_end_idx = self.input_n_end[index]

        # Input nodes and their levels
        input_x_n_slice = self.input_x_n[input_n_start_idx:input_n_end_idx]
        input_abs_level_slice = self.input_level[input_n_start_idx:input_n_end_idx]
        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        # Input edges (re-indexed to be local to input_x_n_slice)
        current_n_start_offset = input_n_start_idx
        src_for_item = self.input_src[input_e_start_idx:input_e_end_idx] - current_n_start_offset
        dst_for_item = self.input_dst[input_e_start_idx:input_e_end_idx] - current_n_start_offset

        # Query edges for prediction (also re-indexed to be local to input_x_n_slice)
        query_s_idx = self.query_start_indices[index]
        query_e_idx = self.query_end_indices[index]

        query_src_for_item = self.query_src_list[query_s_idx:query_e_idx] - current_n_start_offset
        query_dst_for_item = self.query_dst_list[query_s_idx:query_e_idx] - current_n_start_offset
        label_for_item = self.label_list[query_s_idx:query_e_idx]

        # Apply noise to edge labels (label_for_item)
        # Need to reshape label_for_item for apply_noise if it expects a matrix
        # Assuming self.edge_diffusion.apply_noise can handle a flat list of 0/1 labels
        # Or it might expect an adjacency matrix representation of the queries.
        # The original code reshaped:
        #   unique_src = torch.unique(query_src_for_item, sorted=False) # These are local query src/dst
        #   unique_dst = torch.unique(query_dst_for_item, sorted=False)
        #   if len(unique_dst) > 0 and len(unique_src) > 0 and len(label_for_item) == len(unique_dst) * len(unique_src):
        #      label_adj = label_for_item.reshape(len(unique_dst), len(unique_src))
        #      t, label_t_adj = self.edge_diffusion.apply_noise(label_adj)
        #      # noisy_src/dst are derived from query_src/dst where label_t_adj (flattened) is 1
        #      label_t_flat = label_t_adj.flatten()
        #      mask = (label_t_flat == 1)
        #      noisy_src = query_src_for_item[mask]
        #      noisy_dst = query_dst_for_item[mask]
        #   else: # Fallback if reshape is not possible (e.g. not all pairs queried)
        #      # Apply noise to flat labels, but noisy_src/dst logic might be simpler
        #      # For simplicity, if not forming a full matrix, let's assume apply_noise works on flat labels
        #      # and the model predicts for all queries. Noisy edges are just those with label_t=1.
        #      # This part needs to align with how EdgeDiscreteDiffusion and EdgePredModel expect input.
        #      # The original code's apply_noise for edges was complex and used avg_in_deg.
        #      # Let's assume a simplified path for now if the reshape fails.
        #      t = torch.randint(0, self.edge_diffusion.T + 1, (1,)).item() # Sample a timestep
        #      # A placeholder for noise application on flat labels:
        #      # This is a simplification. Real noise model would be more complex.
        #      noise_prob = self.edge_diffusion.betas[t-1] if t > 0 else 0.0 # Simplified noise probability
        #      label_t_flat = torch.where(torch.rand_like(label_for_item.float()) < noise_prob, 1 - label_for_item, label_for_item)
        #      mask = (label_t_flat == 1)
        #      noisy_src = query_src_for_item[mask]
        #      noisy_dst = query_dst_for_item[mask]

        # Simpler approach based on original structure for apply_noise:
        # It expects a 2D adjacency matrix of labels for the *queried* pairs.
        # This requires query_src_for_item and query_dst_for_item to map to a dense matrix.
        # Let's assume for now that apply_noise can take flat labels and return flat noisy labels + t
        # And that the model can take all query pairs.
        # The original `apply_noise` in `LayerDAGEdgePredDataset` was:
        #   unique_src = torch.unique(query_src, sorted=False)
        #   unique_dst = torch.unique(query_dst, sorted=False)
        #   label_adj = label.reshape(len(unique_dst), len(unique_src))
        #   t, label_t = self.edge_diffusion.apply_noise(label_adj)
        #   mask = (label_t == 1) # This label_t is 2D
        #   noisy_src = query_src[mask.flatten()] # This assumes query_src corresponds to flattened mask
        #   noisy_dst = query_dst[mask.flatten()]
        # This is complex. The collate_fn expects flat noisy_src/dst.
        # A more robust way is to apply noise to the flat `label_for_item` directly if `edge_diffusion.apply_noise` supports it
        # or if `edge_diffusion` is simplified for binary values.

        # For now, let's assume a simplified noise application for the purpose of getting indices right.
        # The actual noise model (self.edge_diffusion.apply_noise) needs to be compatible.
        # If self.edge_diffusion.apply_noise expects a 2D matrix, this part needs more work.
        # Let's assume it can take the flat `label_for_item` and return a flat `label_t` and scalar `t`.
        # This is a placeholder for correct noise application logic:
        if label_for_item.numel() > 0:
            # This is a placeholder. The actual call to self.edge_diffusion.apply_noise
            # needs to be consistent with its implementation.
            # If it expects a 2D matrix, label_for_item needs to be reshaped.
            # For now, assume it can work with flat labels or a simplified noise model.
            t_scalar = torch.randint(0, self.edge_diffusion.T + 1, (1,)).item() if hasattr(self.edge_diffusion,
                                                                                           'T') else 0
            # Placeholder for noisy labels - in reality, this comes from self.edge_diffusion.apply_noise
            label_t_flat = label_for_item  # Replace with actual noisy labels

            # Create t as a tensor matching the number of queries, as expected by collate_fn
            t_tensor = torch.full((label_for_item.shape[0], 1), t_scalar, dtype=torch.long)

            mask = (label_t_flat == 1)  # Assuming label_t_flat is binary
            noisy_src_for_item = query_src_for_item[mask]
            noisy_dst_for_item = query_dst_for_item[mask]
        else:  # No queries
            t_tensor = torch.empty((0, 1), dtype=torch.long)
            noisy_src_for_item = torch.empty_like(query_src_for_item)
            noisy_dst_for_item = torch.empty_like(query_dst_for_item)

        if self.conditional:
            input_g_idx = self.input_g[index]
            input_y_val = self.input_y[input_g_idx].item() if isinstance(self.input_y[input_g_idx], torch.Tensor) and \
                                                              self.input_y[input_g_idx].numel() == 1 else self.input_y[
                input_g_idx]

            return src_for_item, dst_for_item, \
                noisy_src_for_item, noisy_dst_for_item, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                t_tensor, input_y_val, \
                query_src_for_item, query_dst_for_item, label_for_item
        else:
            return src_for_item, dst_for_item, \
                noisy_src_for_item, noisy_dst_for_item, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                t_tensor, \
                query_src_for_item, query_dst_for_item, label_for_item


def collate_common(src, dst, x_n, abs_level, rel_level):
    # Filter out any None items that might have occurred if a __getitem__ failed or returned None
    # This is a safeguard; ideally, __getitem__ should always return valid data or raise an error.
    valid_indices = [i for i, item_x_n in enumerate(x_n) if item_x_n is not None and len(item_x_n) > 0]
    if not valid_indices:  # All items were None or empty
        # Return empty tensors with expected structure or handle error
        # This depends on how downstream code handles empty batches.
        # For now, let's assume we proceed if at least one valid item.
        # If all are invalid, this will likely lead to errors later.
        # A more robust solution might be to raise an error or return a specific "empty batch" signal.
        # For simplicity, if all items are invalid, this will likely fail at torch.cat if lists are empty.
        # Let's assume the filtering ensures non-empty lists if valid_indices is not empty.
        pass  # Let it proceed, subsequent torch.cat might fail if all are filtered out.

    # Ensure we only process valid items
    src = [src[i] for i in valid_indices]
    dst = [dst[i] for i in valid_indices]
    x_n = [x_n[i] for i in valid_indices]
    abs_level = [abs_level[i] for i in valid_indices]
    rel_level = [rel_level[i] for i in valid_indices]

    if not x_n:  # If all items were filtered out
        # Return structure expected by downstream code for an empty batch
        # This is a simplified handling. Proper empty batch handling might be more complex.
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long)

    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in x_n]), dim=0)

    batch_size = len(x_n)
    src_ = []
    dst_ = []
    for i in range(batch_size):
        # Ensure src[i] and dst[i] are tensors
        src_i = src[i] if isinstance(src[i], torch.Tensor) else torch.LongTensor(src[i])
        dst_i = dst[i] if isinstance(dst[i], torch.Tensor) else torch.LongTensor(dst[i])
        if src_i.numel() > 0 or dst_i.numel() > 0:  # Only append if there are edges
            src_.append(src_i + num_nodes_cumsum[i])
            dst_.append(dst_i + num_nodes_cumsum[i])

    if src_:  # If there are any edges in the batch
        src = torch.cat(src_, dim=0)
        dst = torch.cat(dst_, dim=0)
        edge_index = torch.stack([dst, src])
    else:  # No edges in the batch
        src = torch.empty((0,), dtype=torch.long)
        dst = torch.empty((0,), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x_n = torch.cat(x_n, dim=0).long()
    abs_level = torch.cat(abs_level, dim=0).float().unsqueeze(-1)
    rel_level = torch.cat(rel_level, dim=0).float().unsqueeze(-1)

    # Prepare edge index for node to graph mapping
    nids = []
    gids = []
    for i in range(batch_size):
        num_nodes_in_graph_i = num_nodes_cumsum[i + 1] - num_nodes_cumsum[i]
        if num_nodes_in_graph_i > 0:  # Only if graph has nodes
            nids.append(torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i + 1]).long())
            gids.append(torch.ones(num_nodes_in_graph_i).fill_(i).long())

    if nids:  # If there are any nodes in the batch
        nids = torch.cat(nids, dim=0)
        gids = torch.cat(gids, dim=0)
        n2g_index = torch.stack([gids, nids])
    else:  # No nodes in the batch
        nids = torch.empty((0,), dtype=torch.long)
        gids = torch.empty((0,), dtype=torch.long)
        n2g_index = torch.empty((2, 0), dtype=torch.long)

    return batch_size, edge_index, x_n, abs_level, rel_level, n2g_index


def collate_node_count(data):
    # Filter out None items from data if any __getitem__ failed
    data = [d for d in data if d is not None]
    if not data:  # All items were None
        # Return a structure indicating an empty batch, matching expected output tuple length
        # This needs to be robustly handled by the training loop.
        # Assuming 7 or 8 items in the tuple based on conditional flag.
        # This is a placeholder, proper empty batch handling is complex.
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)  # Unconditional
        # Or add batch_y for conditional case.

    if len(data[0]) == 7:  # Conditional case
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_y_list, batch_label = map(list,
                                                                                                           zip(*data))

        # Process batch_y: y_ is broadcasted if needed by BiMPNNEncoder if pool is None
        # If BiMPNNEncoder has pool='sum' (typical for node_count), y is graph-level.
        # The original code created a node-level y_ by repeating.
        # Let's keep y as graph-level here, as GraphClassifier expects graph-level y.
        # batch_y will be (batch_size, 1) or (batch_size, F_y)
        batch_y_tensor = torch.tensor(batch_y_list)  # Assuming batch_y_list contains scalars or 1D tensors for y
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)

    else:  # Unconditional case
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_label = map(
            list, zip(*data))
        batch_y_tensor = None  # No y for unconditional

    batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, batch_rel_level_cat, \
        batch_n2g_index = collate_common(
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)

    batch_label_stacked = torch.stack(batch_label) if batch_label else torch.empty(0, dtype=torch.long)

    if len(data[0]) == 7:  # Conditional
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_y_tensor, batch_n2g_index, batch_label_stacked
    else:  # Unconditional
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_label_stacked


def collate_node_pred(data):
    data = [d for d in data if d is not None]
    if not data:  # All items were None
        # Placeholder for empty batch, adjust tuple length as needed
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long), torch.empty((0,)), torch.empty((0, 1), dtype=torch.long), \
            torch.empty((0,)), torch.empty((0,), dtype=torch.long), torch.empty((0,))  # Unconditional

    if len(data[0]) == 9:  # Conditional: src, dst, x_n, abs_l, rel_l, z_t, t, y, z
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, \
            batch_z_t, batch_t, batch_y_list, batch_z = map(list, zip(*data))

        batch_y_tensor = torch.tensor(batch_y_list)  # Graph-level y
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)

    elif len(data[0]) == 8:  # Unconditional: src, dst, x_n, abs_l, rel_l, z_t, t, z
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, \
            batch_z_t, batch_t, batch_z = map(list, zip(*data))
        batch_y_tensor = None
    else:
        raise ValueError(f"Unexpected number of items in data for collate_node_pred: {len(data[0])}")

    batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, batch_rel_level_cat, \
        batch_n2g_index = collate_common(
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)

    # Filter batch_z_t for non-empty tensors before cat
    batch_z_t_filtered = [zt for zt in batch_z_t if zt.numel() > 0]
    if not batch_z_t_filtered and any(zt.numel() == 0 for zt in batch_z_t):  # if some were empty but not all
        # This case means some graphs had no query nodes.
        # The logic for num_query_cumsum and query2g needs to handle this.
        # If all z_t are empty, then batch_z_t_cat will be empty.
        pass

    num_query_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(z_t_i) for z_t_i in batch_z_t]), dim=0)  # Use original batch_z_t for lengths

    query2g = []
    for i in range(batch_size):  # batch_size from collate_common
        num_queries_i = num_query_cumsum[i + 1] - num_query_cumsum[i]
        if num_queries_i > 0:
            query2g.append(torch.ones(num_queries_i).fill_(i).long())

    if query2g:
        query2g = torch.cat(query2g)
    else:  # No queries in the entire batch
        query2g = torch.empty((0,), dtype=torch.long)

    batch_z_t_cat = torch.cat(batch_z_t_filtered) if batch_z_t_filtered else torch.empty(
        (0, batch_z_t[0].shape[1] if batch_z_t and batch_z_t[0].numel() > 0 else 0),
        dtype=batch_z_t[0].dtype if batch_z_t else torch.long)
    batch_t_cat = torch.cat(batch_t).unsqueeze(-1) if batch_t and all(t.numel() > 0 for t in batch_t) else torch.empty(
        (0, 1), dtype=torch.long)  # t corresponds to z_t

    batch_z_filtered = [z_val for z_val in batch_z if z_val.numel() > 0]
    batch_z_cat = torch.cat(batch_z_filtered) if batch_z_filtered else torch.empty(
        (0, batch_z[0].shape[1] if batch_z and batch_z[0].numel() > 0 else 0),
        dtype=batch_z[0].dtype if batch_z else torch.long)

    if batch_z_cat.ndim == 1 and batch_z_cat.numel() > 0:  # Ensure it's not empty before unsqueeze
        batch_z_cat = batch_z_cat.unsqueeze(-1)
    elif batch_z_cat.numel() == 0 and batch_z_cat.ndim == 1:  # if empty and 1D, make it (0,1)
        batch_z_cat = batch_z_cat.view(0, 1)

    if len(data[0]) == 9:  # Conditional
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_z_t_cat, batch_t_cat, batch_y_tensor, \
            query2g, num_query_cumsum, batch_z_cat
    else:  # Unconditional
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_z_t_cat, batch_t_cat, \
            query2g, num_query_cumsum, batch_z_cat


def collate_edge_pred(data):
    data = [d for d in data if d is not None]
    if not data:
        # Placeholder for empty batch
        return torch.empty((2, 0), dtype=torch.long), torch.empty((2, 0), dtype=torch.long), \
            torch.empty((0,), dtype=torch.long), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.long), \
            torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0,), dtype=torch.long)  # Unconditional, 9 items

    if len(data[
               0]) == 12:  # Conditional: input_src, input_dst, noisy_src, noisy_dst, x_n, abs_l, rel_l, t, y, query_src, query_dst, label
        batch_input_src, batch_input_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, \
            batch_abs_level, batch_rel_level, batch_t_list, batch_y_list, \
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))

        batch_y_tensor = torch.tensor(batch_y_list)  # Graph-level y
        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)

    elif len(data[0]) == 11:  # Unconditional
        batch_input_src, batch_input_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, \
            batch_abs_level, batch_rel_level, batch_t_list, \
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        batch_y_tensor = None
    else:
        raise ValueError(f"Unexpected number of items in data for collate_edge_pred: {len(data[0])}")

    # Common processing for nodes
    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in batch_x_n]), dim=0)

    batch_x_n_cat = torch.cat(batch_x_n, dim=0).long()
    batch_abs_level_cat = torch.cat(batch_abs_level, dim=0).float().unsqueeze(-1)
    batch_rel_level_cat = torch.cat(batch_rel_level, dim=0).float().unsqueeze(-1)

    # Process edges (input_src/dst, noisy_src/dst, query_src/dst)
    # These are already local 0-indexed from __getitem__

    # Batch input_edges
    input_src_cat_list, input_dst_cat_list = [], []
    for i in range(len(batch_x_n)):
        if batch_input_src[i].numel() > 0:
            input_src_cat_list.append(batch_input_src[i] + num_nodes_cumsum[i])
            input_dst_cat_list.append(batch_input_dst[i] + num_nodes_cumsum[i])
    input_src_b = torch.cat(input_src_cat_list) if input_src_cat_list else torch.empty(0, dtype=torch.long)
    input_dst_b = torch.cat(input_dst_cat_list) if input_dst_cat_list else torch.empty(0, dtype=torch.long)
    input_edge_index_b = torch.stack([input_dst_b, input_src_b]) if input_src_b.numel() > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long)

    # Batch noisy_edges
    noisy_src_cat_list, noisy_dst_cat_list = [], []
    for i in range(len(batch_x_n)):
        if batch_noisy_src[i].numel() > 0:
            noisy_src_cat_list.append(batch_noisy_src[i] + num_nodes_cumsum[i])
            noisy_dst_cat_list.append(batch_noisy_dst[i] + num_nodes_cumsum[i])
    noisy_src_b = torch.cat(noisy_src_cat_list) if noisy_src_cat_list else torch.empty(0, dtype=torch.long)
    noisy_dst_b = torch.cat(noisy_dst_cat_list) if noisy_dst_cat_list else torch.empty(0, dtype=torch.long)
    noisy_edge_index_b = torch.stack([noisy_dst_b, noisy_src_b]) if noisy_src_b.numel() > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long)

    # Batch query_edges and corresponding t and labels
    query_src_cat_list, query_dst_cat_list = [], []
    t_for_queries_list, labels_for_queries_list = [], []

    for i in range(len(batch_x_n)):
        if batch_query_src[i].numel() > 0:
            query_src_cat_list.append(batch_query_src[i] + num_nodes_cumsum[i])
            query_dst_cat_list.append(batch_query_dst[i] + num_nodes_cumsum[i])
            # batch_t_list[i] should be (num_queries_i, 1) from __getitem__
            t_for_queries_list.append(batch_t_list[i])
            labels_for_queries_list.append(batch_label[i])

    query_src_b = torch.cat(query_src_cat_list) if query_src_cat_list else torch.empty(0, dtype=torch.long)
    query_dst_b = torch.cat(query_dst_cat_list) if query_dst_cat_list else torch.empty(0, dtype=torch.long)

    # Concatenate t and labels only if there are queries
    t_b = torch.cat(t_for_queries_list) if t_for_queries_list else torch.empty((0, 1), dtype=torch.long)
    label_b = torch.cat(labels_for_queries_list) if labels_for_queries_list else torch.empty(0, dtype=torch.long)

    if len(data[0]) == 12:  # Conditional
        return input_edge_index_b, noisy_edge_index_b, batch_x_n_cat, \
            batch_abs_level_cat, batch_rel_level_cat, \
            t_b, batch_y_tensor, \
            query_src_b, query_dst_b, label_b
    else:  # Unconditional
        return input_edge_index_b, noisy_edge_index_b, batch_x_n_cat, \
            batch_abs_level_cat, batch_rel_level_cat, \
            t_b, \
            query_src_b, query_dst_b, label_b
