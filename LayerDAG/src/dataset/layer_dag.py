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
        # For debugging: store original graph index for each item
        self.item_to_original_graph_idx = []

        for i in range(len(dag_dataset)):  # Iterate over original graphs
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g = len(self.input_y)  # This is the graph index for conditional data
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i
                input_g = -1  # Placeholder if not conditional

            # This is the global index in self.input_x_n where nodes for the *current original graph i* start.
            current_original_sample_node_base_idx = len(self.input_x_n)

            # These track the extent of nodes and edges for the *input graph of a single training item*.
            # An item is created for each layer of the original graph.
            # `step_input_n_start` is the global index of the dummy node for this item.
            # `step_input_n_end` is the global index *after* the last node of this item's input graph.
            # `step_input_e_start` / `step_input_e_end` track edges for this item's input graph.

            item_nodes_start_global = current_original_sample_node_base_idx  # Dummy node for this item
            item_nodes_end_global = current_original_sample_node_base_idx  # Will be incremented
            item_edges_start_global = len(self.input_src)
            item_edges_end_global = len(self.input_src)

            # Add dummy node for this item. Its global index is `item_nodes_end_global`.
            self.input_x_n.append(dag_dataset.dummy_category)
            self.input_level.append(0)  # Level of the dummy node
            item_nodes_end_global += 1  # Dummy node is now part of this item's input graph

            # Original src/dst are 0-indexed for graph `x_n`.
            # We shift them to be 1-indexed relative to this item's dummy node (which is at local index 0).
            src_plus_dummy_local = src + 1
            dst_plus_dummy_local = dst + 1

            num_nodes_in_original_sample_incl_dummy = len(x_n) + 1  # Total nodes if all were laid out for this item
            in_deg_local = self.get_in_deg(dst_plus_dummy_local, num_nodes_in_original_sample_incl_dummy)

            x_n_list_original = x_n.tolist()  # Original node features (0-indexed)
            # Adj lists use local 1-indexed node IDs (relative to dummy)
            out_adj_list_local = self.get_out_adj_list(src_plus_dummy_local.tolist(), dst_plus_dummy_local.tolist())
            in_adj_list_local = self.get_in_adj_list(src_plus_dummy_local.tolist(), dst_plus_dummy_local.tolist())

            # `frontiers` contains local 1-indexed node IDs (relative to dummy)
            frontiers_local = [
                u_local for u_local in range(1, num_nodes_in_original_sample_incl_dummy) if in_deg_local[u_local] == 0
            ]
            frontier_size = len(frontiers_local)
            current_item_level = 0

            # Loop to generate one training item per layer of the original graph `i`
            while frontier_size > 0:
                current_item_level += 1  # Level of nodes in the current frontier

                # --- Create a training item for predicting `frontier_size` ---
                # The input graph for this item consists of nodes up to `item_nodes_end_global` (exclusive)
                # and edges up to `item_edges_end_global` (exclusive).
                self.input_e_start.append(item_edges_start_global)
                self.input_e_end.append(item_edges_end_global)
                self.input_n_start.append(item_nodes_start_global)
                self.input_n_end.append(item_nodes_end_global)
                self.item_to_original_graph_idx.append(i)  # Debug: track original graph

                if conditional:
                    self.input_g.append(input_g)
                self.label.append(frontier_size)  # Label for this item is the size of the current frontier

                # --- Prepare for the *next* item (or termination) ---
                # The nodes in the current `frontiers_local` will be added to the input graph
                # for the *next* item (if any).
                next_frontiers_local_buffer = []
                for u_frontier_node_local_idx in frontiers_local:  # u_frontier_node_local_idx is 1-indexed for this sample
                    # Add current frontier node's attribute to global list `self.input_x_n`.
                    # Its global index will be `item_nodes_end_global`.
                    # This node is at `current_item_level`.
                    self.input_x_n.append(
                        x_n_list_original[u_frontier_node_local_idx - 1])  # -1 to access 0-indexed x_n_list_original
                    self.input_level.append(current_item_level)

                    # Add edges connecting to this `u_frontier_node_local_idx` to global lists.
                    # These edges become part of the input graph for the *next* item.
                    # Source nodes `t_source_node_local_idx` are from previous levels (already in `self.input_x_n`).
                    for t_source_node_local_idx in in_adj_list_local[u_frontier_node_local_idx]:
                        # Convert local 1-indexed IDs to global 0-indexed IDs
                        global_t_src_idx = item_nodes_start_global + t_source_node_local_idx
                        global_u_dst_idx = item_nodes_start_global + u_frontier_node_local_idx  # This is the current frontier node

                        # --- DEBUG CHECK for edge addition ---
                        # `global_t_src_idx` should be < `item_nodes_end_global` (nodes already added)
                        # `global_u_dst_idx` is `item_nodes_end_global` (the node being added *now* to self.input_x_n)
                        # So, for the *next* item, this edge will be valid.
                        # The critical part is that `item_edges_end_global` correctly reflects edges
                        # *within* the nodes defined by `item_nodes_start_global` to `item_nodes_end_global`
                        # for the item currently being defined by `self.input_e_start.append(item_edges_start_global)`.
                        if not (item_nodes_start_global <= global_t_src_idx < item_nodes_end_global):
                            print(
                                f"DEBUG INITWARNING (Graph {i}, ItemLevel {current_item_level}): Edge source {global_t_src_idx} (local {t_source_node_local_idx}) "
                                f"is OUTSIDE current item's node range [{item_nodes_start_global}, {item_nodes_end_global - 1}] "
                                f"when adding edge to frontier node {global_u_dst_idx} (local {u_frontier_node_local_idx}).")
                        # This edge connects to `u_frontier_node_local_idx` which is *just being added*.
                        # So `global_u_dst_idx` (which is `item_nodes_end_global` at the moment of adding `u_frontier_node_local_idx`'s feature)
                        # will be part of the *next* item's node set.
                        # The edges added here are for the *next* item's input graph.

                        self.input_src.append(global_t_src_idx)
                        self.input_dst.append(global_u_dst_idx)  # This is the node whose feature was just appended
                        item_edges_end_global += 1  # This edge is now part of the *next* item's edge set

                    # Update in-degrees for neighbors of `u_frontier_node_local_idx`
                    for v_target_node_local_idx in out_adj_list_local[u_frontier_node_local_idx]:
                        in_deg_local[v_target_node_local_idx] -= 1
                        if in_deg_local[v_target_node_local_idx] == 0:
                            next_frontiers_local_buffer.append(v_target_node_local_idx)

                    item_nodes_end_global += 1  # Current frontier node is now processed and part of the input for the next step

                frontiers_local = next_frontiers_local_buffer
                frontier_size = len(frontiers_local)
                # `item_nodes_start_global`, `item_edges_start_global` remain the same for all items from this original graph `i`.
                # `item_nodes_end_global` and `item_edges_end_global` grow, defining the cumulative graph.

            # Termination step for the original graph `i`
            self.input_e_start.append(item_edges_start_global)
            self.input_e_end.append(item_edges_end_global)
            self.input_n_start.append(item_nodes_start_global)
            self.input_n_end.append(item_nodes_end_global)
            self.item_to_original_graph_idx.append(i)  # Debug

            if conditional:
                self.input_g.append(input_g)
            self.label.append(0)  # Label for termination is 0 new nodes

        self.base_postprocess()  # Convert lists to tensors
        self.label = torch.LongTensor(self.label)
        if len(self.label) > 0:
            self.max_layer_size = self.label.max().item()
        else:
            self.max_layer_size = 0
        self.item_to_original_graph_idx = torch.LongTensor(self.item_to_original_graph_idx)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # These are global indices into self.input_src/dst and self.input_x_n
        item_edges_start_global_idx = self.input_e_start[index]
        item_edges_end_global_idx = self.input_e_end[index]
        item_nodes_start_global_idx = self.input_n_start[index]
        item_nodes_end_global_idx = self.input_n_end[index]

        # Slice the global tensors to get data for this specific item
        # `input_x_n_slice` contains the node features for the GNN input of this item.
        input_x_n_slice = self.input_x_n[item_nodes_start_global_idx:item_nodes_end_global_idx]
        input_abs_level_slice = self.input_level[item_nodes_start_global_idx:item_nodes_end_global_idx]

        if input_abs_level_slice.numel() > 0:
            input_rel_level_slice = input_abs_level_slice.max() - input_abs_level_slice
        else:
            # Handle empty slice for levels, e.g. if item_nodes_start_global_idx == item_nodes_end_global_idx
            # (though this shouldn't happen if dummy node is always present)
            input_rel_level_slice = torch.empty_like(input_abs_level_slice)

        # `src_for_item_global` and `dst_for_item_global` contain GLOBAL node indices for edges of this item.
        src_for_item_global = self.input_src[item_edges_start_global_idx:item_edges_end_global_idx]
        dst_for_item_global = self.input_dst[item_edges_start_global_idx:item_edges_end_global_idx]

        # Re-index these global edge indices to be 0-based LOCAL to `input_x_n_slice`.
        # This is where the error likely occurs if global indices are outside the slice's range.
        src_reindexed_local = src_for_item_global - item_nodes_start_global_idx
        dst_reindexed_local = dst_for_item_global - item_nodes_start_global_idx

        num_nodes_in_slice = len(input_x_n_slice)

        # --- Start Debug Checks for __getitem__ ---
        valid_item = True
        if src_reindexed_local.numel() > 0:  # Only check if there are edges
            if src_reindexed_local.max() >= num_nodes_in_slice or src_reindexed_local.min() < 0:
                valid_item = False
                print(f"--- ERROR in LayerDAGNodeCountDataset.__getitem__(index={index}) ---")
                print(f"Original Graph Index for this item: {self.item_to_original_graph_idx[index].item()}")
                print(
                    f"Item's global node range: [{item_nodes_start_global_idx.item()}, {item_nodes_end_global_idx.item()}) -> num_nodes_in_slice = {num_nodes_in_slice}")
                print(
                    f"Item's global edge range: [{item_edges_start_global_idx.item()}, {item_edges_end_global_idx.item()})")
                print(
                    f"src_for_item_global (min/max): {src_for_item_global.min().item() if src_for_item_global.numel() > 0 else 'N/A'} / {src_for_item_global.max().item() if src_for_item_global.numel() > 0 else 'N/A'}")
                print(
                    f"src_reindexed_local (min/max): {src_reindexed_local.min().item()} / {src_reindexed_local.max().item()}")
                print(
                    f"   Problem: src_reindexed_local.max() ({src_reindexed_local.max().item()}) >= num_nodes_in_slice ({num_nodes_in_slice}) OR min < 0")
                # print(f"   src_for_item_global: {src_for_item_global}")
                # print(f"   src_reindexed_local: {src_reindexed_local}")

        if dst_reindexed_local.numel() > 0:  # Only check if there are edges
            if dst_reindexed_local.max() >= num_nodes_in_slice or dst_reindexed_local.min() < 0:
                valid_item = False
                print(f"--- ERROR in LayerDAGNodeCountDataset.__getitem__(index={index}) ---")
                print(f"Original Graph Index for this item: {self.item_to_original_graph_idx[index].item()}")
                print(
                    f"Item's global node range: [{item_nodes_start_global_idx.item()}, {item_nodes_end_global_idx.item()}) -> num_nodes_in_slice = {num_nodes_in_slice}")
                print(
                    f"Item's global edge range: [{item_edges_start_global_idx.item()}, {item_edges_end_global_idx.item()})")
                print(
                    f"dst_for_item_global (min/max): {dst_for_item_global.min().item() if dst_for_item_global.numel() > 0 else 'N/A'} / {dst_for_item_global.max().item() if dst_for_item_global.numel() > 0 else 'N/A'}")
                print(
                    f"dst_reindexed_local (min/max): {dst_reindexed_local.min().item()} / {dst_reindexed_local.max().item()}")
                print(
                    f"   Problem: dst_reindexed_local.max() ({dst_reindexed_local.max().item()}) >= num_nodes_in_slice ({num_nodes_in_slice}) OR min < 0")
                # print(f"   dst_for_item_global: {dst_for_item_global}")
                # print(f"   dst_reindexed_local: {dst_reindexed_local}")

        if not valid_item:
            # Optionally, raise an error here to stop immediately, or return None/dummy to see if collate handles it.
            # For now, let it pass to see the collate error, but the print above should give clues.
            # To make it fail here:
            # raise IndexError(f"Invalid edge index detected in __getitem__ for index {index}")
            pass
        # --- End Debug Checks ---

        if self.conditional:
            input_g_idx = self.input_g[index]
            # Ensure input_y_val is a tensor, even if scalar, for consistent batching
            y_val_raw = self.input_y[input_g_idx]
            input_y_val_tensor = y_val_raw if isinstance(y_val_raw, torch.Tensor) else torch.tensor(y_val_raw)
            if input_y_val_tensor.ndim == 0:  # Ensure it's at least 1D
                input_y_val_tensor = input_y_val_tensor.unsqueeze(0)

            return src_reindexed_local, dst_reindexed_local, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                input_y_val_tensor, self.label[index]
        else:
            return src_reindexed_local, dst_reindexed_local, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                self.label[index]


class LayerDAGNodePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False, get_marginal=True):
        super().__init__(conditional)
        self.label_start = []  # Global start index in self.input_x_n for label nodes
        self.label_end = []  # Global end index in self.input_x_n for label nodes
        self.node_diffusion = None  # Will be set after instantiation

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

            self.input_x_n.append(dag_dataset.dummy_category)
            self.input_level.append(0)
            step_input_n_end += 1

            current_label_global_start_idx = len(self.input_x_n)  # Labels start *after* all input nodes for this step

            src_plus_dummy = src + 1
            dst_plus_dummy = dst + 1
            level = 0

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
                self.input_n_start.append(step_input_n_start)
                self.input_n_end.append(step_input_n_end)

                if conditional:
                    self.input_g.append(input_g)

                self.label_start.append(current_label_global_start_idx)
                current_label_global_end_idx = current_label_global_start_idx + frontier_size
                self.label_end.append(current_label_global_end_idx)

                next_frontiers = []
                for u_frontier_node_local in frontiers:
                    # Add current frontier node attributes to self.input_x_n (these are the labels z)
                    # Their global indices start from `current_label_global_start_idx`
                    self.input_x_n.append(x_n_list[u_frontier_node_local - 1])
                    self.input_level.append(level)  # Level for this label node

                    for t_source_node_local in in_adj_list[u_frontier_node_local]:
                        self.input_src.append(
                            step_input_n_start + t_source_node_local)  # Edges connect to input graph nodes
                        self.input_dst.append(
                            step_input_n_start + u_frontier_node_local)  # u_frontier is now part of input graph for next step
                        step_input_e_end += 1

                    for v_target_node_local in out_adj_list[u_frontier_node_local]:
                        in_deg[v_target_node_local] -= 1
                        if in_deg[v_target_node_local] == 0:
                            next_frontiers.append(v_target_node_local)

                # Update for next iteration: labels of this step become input for the next
                step_input_n_end += frontier_size
                current_label_global_start_idx = current_label_global_end_idx  # Next labels start after current ones

                frontiers = next_frontiers
                frontier_size = len(frontiers)

        self.base_postprocess()
        self.label_start = torch.LongTensor(self.label_start)
        self.label_end = torch.LongTensor(self.label_end)

        if get_marginal:
            # Ensure self.input_x_n is used for marginal calculation
            # It should contain all node types including dummy and actual attributes
            input_x_n_for_marginal = self.input_x_n
            if input_x_n_for_marginal.ndim == 1:  # Ensure it's at least 2D for feature iteration
                input_x_n_for_marginal = input_x_n_for_marginal.unsqueeze(-1)

            num_feats = input_x_n_for_marginal.shape[-1]
            x_n_marginal = []

            # dag_dataset.num_categories is the number of *actual* node types (e.g., 4 for AIG)
            # The diffusion model's num_classes_list should correspond to these actual types.
            num_actual_categories_per_feat = [
                                                 dag_dataset.num_categories] * num_feats  # Assuming same for all feats if not multi-dim from config

            for f_idx in range(num_feats):
                input_x_n_f = input_x_n_for_marginal[:, f_idx]

                # We only want marginal of *actual* node types, exclude dummy for diffusion
                # The dummy category is `dag_dataset.dummy_category`
                actual_nodes_mask = (input_x_n_f != dag_dataset.dummy_category)
                actual_nodes_input_x_n_f = input_x_n_f[actual_nodes_mask]

                num_actual_types_this_feat = num_actual_categories_per_feat[f_idx]
                marginal_f = torch.zeros(num_actual_types_this_feat)

                if actual_nodes_input_x_n_f.numel() > 0:
                    unique_actual_vals, counts_actual_vals = torch.unique(actual_nodes_input_x_n_f, return_counts=True)
                    for val_idx, val_actual in enumerate(unique_actual_vals):
                        if 0 <= val_actual.item() < num_actual_types_this_feat:  # Ensure valid category index
                            marginal_f[val_actual.item()] += counts_actual_vals[val_idx].item()

                if marginal_f.sum() > 0:
                    marginal_f /= marginal_f.sum()
                else:  # Handle case with no actual nodes or all nodes are of types outside expected range
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

        z = self.input_x_n[label_start_global_idx:label_end_global_idx]  # Ground truth attributes for the new layer

        # Ensure z is not empty before applying noise
        if z.numel() == 0:  # No nodes in the layer to predict
            # Create empty tensors with appropriate shapes and types
            # Assuming z_t and z should have a feature dimension if multi-feature
            # node_diffusion.num_classes_list gives number of features
            num_node_features = len(self.node_diffusion.num_classes_list) if self.node_diffusion.num_classes_list else 1

            z_t = torch.empty((0, num_node_features) if num_node_features > 1 else (0,), dtype=torch.long)
            t = torch.empty((1,), dtype=torch.long)  # Timestep is usually scalar for the item
            # z should also match this structure if it was empty
            z_reshaped = torch.empty((0, num_node_features) if num_node_features > 1 else (0,), dtype=torch.long)

        else:
            t, z_t = self.node_diffusion.apply_noise(z)  # z_t will have same shape as z
            z_reshaped = z  # z is already correct

        if self.conditional:
            input_g_idx = self.input_g[index]
            y_val_raw = self.input_y[input_g_idx]
            input_y_val_tensor = y_val_raw if isinstance(y_val_raw, torch.Tensor) else torch.tensor(y_val_raw)
            if input_y_val_tensor.ndim == 0: input_y_val_tensor = input_y_val_tensor.unsqueeze(0)

            return src_reindexed, dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                z_t, t, input_y_val_tensor, z_reshaped  # Use z_reshaped
        else:
            return src_reindexed, dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                z_t, t, z_reshaped  # Use z_reshaped


class LayerDAGEdgePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.query_src_list = []
        self.query_dst_list = []
        self.label_list = []  # 0 or 1 for each query edge
        self.query_start_indices = []  # Start index in query_src/dst/label_list for an item
        self.query_end_indices = []  # End index for an item
        self.edge_diffusion = None  # Will be set after instantiation

        num_total_edges_processed_in_original_graphs = 0
        num_total_nonsrc_nodes_in_original_graphs = 0  # Nodes that are destinations of some edge

        for i in range(len(dag_dataset)):  # Iterate over original graphs
            data_i = dag_dataset[i]
            if conditional:
                src, dst, x_n, y = data_i
                input_g = len(self.input_y)
                self.input_y.append(y)
            else:
                src, dst, x_n = data_i

            num_total_edges_processed_in_original_graphs += len(src)

            current_original_sample_node_base_idx = len(self.input_x_n)
            item_nodes_start_global = current_original_sample_node_base_idx
            item_nodes_end_global = current_original_sample_node_base_idx
            item_edges_start_global = len(self.input_src)  # Edges for GNN input of current item
            item_edges_end_global = len(self.input_src)

            current_item_query_list_start_idx = len(self.query_src_list)  # For self.query_src/dst_list

            self.input_x_n.append(dag_dataset.dummy_category)
            self.input_level.append(0)
            item_nodes_end_global += 1

            src_plus_dummy_local = src + 1
            dst_plus_dummy_local = dst + 1
            level = 0

            num_nodes_in_original_sample_incl_dummy = len(x_n) + 1
            in_deg_local = self.get_in_deg(dst_plus_dummy_local, num_nodes_in_original_sample_incl_dummy)
            x_n_list_original = x_n.tolist()
            out_adj_list_local = self.get_out_adj_list(src_plus_dummy_local.tolist(), dst_plus_dummy_local.tolist())
            in_adj_list_local = self.get_in_adj_list(src_plus_dummy_local.tolist(), dst_plus_dummy_local.tolist())

            # Nodes from previous layer that can be sources for edges to current layer's nodes
            # Initially, these are the first true frontier (nodes with in-degree 0 in original graph)
            # Stored as local 1-indexed IDs.
            potential_source_nodes_local_prev_layer = [
                u_local for u_local in range(1, num_nodes_in_original_sample_incl_dummy) if in_deg_local[u_local] == 0
            ]

            # Add these initial frontier nodes to self.input_x_n as they form the first layer of the GNN input graph
            level += 1
            for u_node_local in potential_source_nodes_local_prev_layer:
                self.input_x_n.append(x_n_list_original[u_node_local - 1])
                self.input_level.append(level)
            item_nodes_end_global += len(potential_source_nodes_local_prev_layer)

            # Determine the next frontier based on the initial one
            current_frontiers_nodes_local_next_layer = []
            for u_node_local in potential_source_nodes_local_prev_layer:
                for v_target_node_local in out_adj_list_local[u_node_local]:
                    in_deg_local[v_target_node_local] -= 1
                    if in_deg_local[v_target_node_local] == 0:
                        current_frontiers_nodes_local_next_layer.append(v_target_node_local)
            current_frontiers_nodes_local_next_layer = list(set(current_frontiers_nodes_local_next_layer))

            # Loop while there are nodes in the current_frontiers_nodes_local_next_layer to process
            while len(current_frontiers_nodes_local_next_layer) > 0:
                level += 1  # This is the level of nodes in `current_frontiers_nodes_local_next_layer`

                # --- Create a training item ---
                # Input graph for this item: nodes up to `item_nodes_end_global`, edges up to `item_edges_end_global`
                self.input_e_start.append(item_edges_start_global)
                self.input_e_end.append(item_edges_end_global)
                self.input_n_start.append(item_nodes_start_global)
                self.input_n_end.append(item_nodes_end_global)
                if conditional: self.input_g.append(input_g)

                self.query_start_indices.append(current_item_query_list_start_idx)
                num_queries_this_item = 0

                # Buffer for nodes that will form the layer after `current_frontiers_nodes_local_next_layer`
                next_frontiers_after_current_local_buffer = []

                # Iterate over nodes in the current layer to be predicted (targets for query edges)
                for u_target_node_local in current_frontiers_nodes_local_next_layer:
                    # Add this target node's feature to self.input_x_n. It becomes part of GNN input for *next* item.
                    self.input_x_n.append(x_n_list_original[u_target_node_local - 1])
                    self.input_level.append(level)
                    global_u_target_node_idx = item_nodes_start_global + u_target_node_local

                    num_total_nonsrc_nodes_in_original_graphs += 1

                    # Create query edges from all `potential_source_nodes_local_prev_layer` to this `u_target_node_local`
                    for t_source_node_local in potential_source_nodes_local_prev_layer:
                        global_t_source_node_idx = item_nodes_start_global + t_source_node_local

                        self.query_src_list.append(global_t_source_node_idx)
                        self.query_dst_list.append(global_u_target_node_idx)
                        num_queries_this_item += 1

                        # Check if this edge actually existed in the original graph
                        if t_source_node_local in in_adj_list_local[u_target_node_local]:
                            self.label_list.append(1)
                            # This edge (t_source -> u_target) is now part of the GNN input graph for the *next* item
                            self.input_src.append(global_t_source_node_idx)
                            self.input_dst.append(global_u_target_node_idx)
                            item_edges_end_global += 1
                        else:
                            self.label_list.append(0)

                    # Find nodes for the layer after the current one
                    for v_next_target_local in out_adj_list_local[u_target_node_local]:
                        in_deg_local[v_next_target_local] -= 1
                        if in_deg_local[v_next_target_local] == 0:
                            next_frontiers_after_current_local_buffer.append(v_next_target_local)

                item_nodes_end_global += len(
                    current_frontiers_nodes_local_next_layer)  # Nodes of current frontier added to input
                current_item_query_list_start_idx += num_queries_this_item
                self.query_end_indices.append(current_item_query_list_start_idx)

                # Update for next iteration
                potential_source_nodes_local_prev_layer.extend(current_frontiers_nodes_local_next_layer)
                current_frontiers_nodes_local_next_layer = list(set(next_frontiers_after_current_local_buffer))

        self.base_postprocess()
        self.query_src_list = torch.LongTensor(self.query_src_list)
        self.query_dst_list = torch.LongTensor(self.query_dst_list)
        self.label_list = torch.LongTensor(self.label_list)
        self.query_start_indices = torch.LongTensor(self.query_start_indices)
        self.query_end_indices = torch.LongTensor(self.query_end_indices)

        if num_total_nonsrc_nodes_in_original_graphs > 0:
            self.avg_in_deg = num_total_edges_processed_in_original_graphs / num_total_nonsrc_nodes_in_original_graphs
        else:
            self.avg_in_deg = 0.0

    def __len__(self):
        return len(self.query_start_indices)  # Number of items generated

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

        # Edges for GNN input (already existing edges)
        src_for_item_gnn_input_global = self.input_src[item_edges_start_global_idx:item_edges_end_global_idx]
        dst_for_item_gnn_input_global = self.input_dst[item_edges_start_global_idx:item_edges_end_global_idx]
        src_reindexed_gnn_input = src_for_item_gnn_input_global - item_nodes_start_global_idx
        dst_reindexed_gnn_input = dst_for_item_gnn_input_global - item_nodes_start_global_idx

        # Query edges for prediction
        query_list_start_for_item = self.query_start_indices[index]
        query_list_end_for_item = self.query_end_indices[index]

        query_src_global_for_item = self.query_src_list[query_list_start_for_item:query_list_end_for_item]
        query_dst_global_for_item = self.query_dst_list[query_list_start_for_item:query_list_end_for_item]
        label_for_query_edges = self.label_list[query_list_start_for_item:query_list_end_for_item]

        # Re-index query edges to be local to input_x_n_slice
        query_src_reindexed_local = query_src_global_for_item - item_nodes_start_global_idx
        query_dst_reindexed_local = query_dst_global_for_item - item_nodes_start_global_idx

        # Apply noise to edge labels (label_for_query_edges)
        if label_for_query_edges.numel() > 0:
            # Determine num_candidate_sources for marginal in apply_noise
            # This is tricky. For a given query (src, dst), the number of candidate sources for dst
            # depends on the GNN input graph structure.
            # A simplified approach: use a characteristic of the input graph for this item.
            # For example, number of nodes in input_x_n_slice, or avg_in_deg of the dataset.
            # The original LayerDAGEdgePredDataset's apply_noise was complex.
            # Let's assume edge_diffusion.apply_noise can take a flat list of edge labels.

            # Simplified: pass number of nodes in the current GNN input graph as a proxy
            num_candidate_sources_for_marginal_approx = len(input_x_n_slice)

            t_scalar, label_t_flat = self.edge_diffusion.apply_noise(
                label_for_query_edges,
                num_candidate_sources_for_marginal=num_candidate_sources_for_marginal_approx
            )

            t_tensor = torch.full((label_for_query_edges.shape[0], 1),
                                  t_scalar.item() if isinstance(t_scalar, torch.Tensor) else t_scalar, dtype=torch.long)

            # noisy_edge_index: edges that are 1 (exist) in the noised labels
            noisy_mask = (label_t_flat == 1)
            noisy_src_reindexed = query_src_reindexed_local[noisy_mask]
            noisy_dst_reindexed = query_dst_reindexed_local[noisy_mask]
        else:  # No query edges for this item
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
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                t_tensor, input_y_val_tensor, \
                query_src_reindexed_local, query_dst_reindexed_local, label_for_query_edges
        else:
            return src_reindexed_gnn_input, dst_reindexed_gnn_input, \
                noisy_src_reindexed, noisy_dst_reindexed, \
                input_x_n_slice, \
                input_abs_level_slice, input_rel_level_slice, \
                t_tensor, \
                query_src_reindexed_local, query_dst_reindexed_local, label_for_query_edges


def collate_common(src, dst, x_n, abs_level, rel_level):
    # Filter out items where x_n is None or empty, as these cannot be processed
    valid_indices = [i for i, item_x_n in enumerate(x_n) if item_x_n is not None and len(item_x_n) > 0]

    if not valid_indices:  # If no valid items in the batch
        # Return empty tensors with correct number of dimensions but 0 size in batch dim
        # This structure should match what the training loop expects for an empty batch.
        # Assuming x_n is (num_nodes), abs_level/rel_level are (num_nodes, 1)
        return 0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), \
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float), \
            torch.empty((2, 0), dtype=torch.long)

    # Use only valid items for batching
    src = [src[i] for i in valid_indices]
    dst = [dst[i] for i in valid_indices]
    x_n = [x_n[i] for i in valid_indices]
    abs_level = [abs_level[i] for i in valid_indices]
    rel_level = [rel_level[i] for i in valid_indices]

    # At this point, x_n contains only non-empty tensors.
    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in x_n]), dim=0)
    batch_size = len(x_n)  # This is now the number of valid items

    src_ = []
    dst_ = []
    for i in range(batch_size):  # Iterate over valid items
        src_i = src[i] if isinstance(src[i], torch.Tensor) else torch.LongTensor(src[i])
        dst_i = dst[i] if isinstance(dst[i], torch.Tensor) else torch.LongTensor(dst[i])
        # Only append if there are edges for this item
        if src_i.numel() > 0 or dst_i.numel() > 0:  # Check if there are edges
            # Ensure src_i and dst_i are 1D before addition
            src_i_flat = src_i.view(-1)
            dst_i_flat = dst_i.view(-1)
            src_.append(src_i_flat + num_nodes_cumsum[i])
            dst_.append(dst_i_flat + num_nodes_cumsum[i])

    if src_:  # If there are any edges across all valid items
        src = torch.cat(src_, dim=0)
        dst = torch.cat(dst_, dim=0)
        edge_index = torch.stack([dst, src])  # DGL format: [dst, src]
    else:  # No edges in any valid item
        src = torch.empty((0,), dtype=torch.long)
        dst = torch.empty((0,), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x_n = torch.cat(x_n, dim=0).long()  # Node features
    abs_level = torch.cat(abs_level, dim=0).float().unsqueeze(-1)  # Absolute levels
    rel_level = torch.cat(rel_level, dim=0).float().unsqueeze(-1)  # Relative levels

    # Create n2g_index (node-to-graph mapping)
    nids = []
    gids = []
    for i in range(batch_size):  # Iterate over valid items
        num_nodes_in_graph_i = num_nodes_cumsum[i + 1] - num_nodes_cumsum[i]
        if num_nodes_in_graph_i > 0:  # Only if the item contributes nodes
            nids.append(torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i + 1]).long())
            gids.append(torch.ones(num_nodes_in_graph_i).fill_(i).long())

    if nids:  # If any nodes were contributed by valid items
        nids = torch.cat(nids, dim=0)
        gids = torch.cat(gids, dim=0)
        n2g_index = torch.stack([gids, nids])  # [graph_idx_in_batch, node_idx_in_batch]
    else:  # No nodes in any valid item (should ideally not happen if x_n filtering works)
        nids = torch.empty((0,), dtype=torch.long)
        gids = torch.empty((0,), dtype=torch.long)
        n2g_index = torch.empty((2, 0), dtype=torch.long)

    return batch_size, edge_index, x_n, abs_level, rel_level, n2g_index


def collate_node_count(data):
    # Filter out None items that might come from __getitem__ if an error occurred
    data = [d for d in data if d is not None]
    if not data:  # If all items were None or batch is empty
        # Return structure expected by the training loop for an empty batch
        # Adjust based on conditional or not
        num_elements = 7 if (len(data) > 0 and len(data[0]) == 7) else 6
        empty_batch = [
            0,  # batch_size
            torch.empty((2, 0), dtype=torch.long),  # edge_index
            torch.empty((0,), dtype=torch.long),  # x_n
            torch.empty((0, 1), dtype=torch.float),  # abs_level
            torch.empty((0, 1), dtype=torch.float),  # rel_level
        ]
        if num_elements == 7:  # Conditional
            empty_batch.append(torch.empty((0, 1), dtype=torch.float))  # y (assuming float, adjust if different)
        empty_batch.extend([
            torch.empty((2, 0), dtype=torch.long),  # n2g_index
            torch.empty((0,), dtype=torch.long)  # label
        ])
        return tuple(empty_batch)

    is_conditional_item = (len(data[0]) == 7)

    if is_conditional_item:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_y_list, batch_label = map(list,
                                                                                                           zip(*data))
        # Ensure batch_y_list items are tensors for robust stacking/conversion
        batch_y_tensorized = [y.clone().detach() if isinstance(y, torch.Tensor) else torch.tensor(y) for y in
                              batch_y_list]
        # Stack if all are same shape, otherwise handle carefully or ensure they are made consistent
        try:
            batch_y_tensor = torch.stack(batch_y_tensorized)
        except RuntimeError:  # If shapes mismatch (e.g. scalar vs tensor)
            # Attempt to make them all at least 1D
            batch_y_tensor = torch.stack([y.unsqueeze(0) if y.ndim == 0 else y for y in batch_y_tensorized])

        if batch_y_tensor.ndim == 1: batch_y_tensor = batch_y_tensor.unsqueeze(-1)  # Ensure (B, F_y)
    else:  # Not conditional
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_label = map(list, zip(*data))
        batch_y_tensor = None

    batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, batch_rel_level_cat, \
        batch_n2g_index = collate_common(
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level
    )

    # Ensure batch_label is a tensor
    if batch_label and isinstance(batch_label[0], torch.Tensor):
        batch_label_stacked = torch.stack(batch_label)
    elif batch_label:
        batch_label_stacked = torch.LongTensor(batch_label)
    else:  # batch_label is empty
        batch_label_stacked = torch.empty(0, dtype=torch.long)

    if is_conditional_item:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_y_tensor, batch_n2g_index, batch_label_stacked
    else:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_label_stacked


def collate_node_pred(data):
    data = [d for d in data if d is not None]
    if not data:  # Empty or all None batch
        num_elements = 9 if (len(data) > 0 and len(data[0]) == 9) else 8
        empty_batch = [
            0, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float), torch.empty((0, 1), dtype=torch.float),
            torch.empty((2, 0), dtype=torch.long),  # n2g_index
            torch.empty((0, 0), dtype=torch.long),  # z_t (assuming feature dim 0 if unknown)
            torch.empty((0, 1), dtype=torch.long),  # t
        ]
        if num_elements == 9:  # Conditional
            empty_batch.append(torch.empty((0, 0), dtype=torch.float))  # y
        empty_batch.extend([
            torch.empty((0,), dtype=torch.long),  # query2g
            torch.empty((0,), dtype=torch.long),  # num_query_cumsum
            torch.empty((0, 0), dtype=torch.long)  # z (assuming feature dim 0 if unknown)
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
    elif len(data[0]) == 8:  # Unconditional
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, \
            batch_z_t, batch_t, batch_z = map(list, zip(*data))
        batch_y_tensor = None
    else:
        raise ValueError(f"Unexpected number of items in data for collate_node_pred: {len(data[0])}")

    batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, batch_rel_level_cat, \
        batch_n2g_index = collate_common(
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)

    # Filter empty tensors from z_t and z before cat, but keep track of original structure for query2g
    # Determine feature dimension from first non-empty tensor or default to 1 if all are empty scalars
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
            batch_z_t_processed.append(zt.unsqueeze(-1) if z_t_feat_dim == 1 else zt.view(len(zt), -1))  # Ensure 2D
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
            batch_z_processed.append(
                z_val.unsqueeze(-1) if z_feat_dim == 1 else z_val.view(len(z_val), -1))  # Ensure 2D
        else:
            batch_z_processed.append(z_val)
    batch_z_cat = torch.cat(batch_z_processed) if batch_z_processed else torch.empty((0, z_feat_dim), dtype=torch.long)

    num_query_cumsum = torch.cumsum(torch.tensor(  # Number of query nodes per item in batch
        [0] + [len(z_t_i) for z_t_i in batch_z_t]), dim=0)  # Use original batch_z_t for lengths

    query2g = []  # Maps each query node (in concatenated batch_z_t_cat) to its graph index
    for i in range(batch_size):  # batch_size is number of valid items
        num_queries_i = num_query_cumsum[i + 1] - num_query_cumsum[i]
        if num_queries_i > 0:
            query2g.append(torch.ones(num_queries_i).fill_(i).long())

    if query2g:
        query2g = torch.cat(query2g)
    else:  # No query nodes in the entire batch
        query2g = torch.empty((0,), dtype=torch.long)

    if is_conditional_item:
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_z_t_cat, batch_t_cat, batch_y_tensor, \
            query2g, num_query_cumsum, batch_z_cat
    else:  # Unconditional
        return batch_size, batch_edge_index, batch_x_n_cat, batch_abs_level_cat, \
            batch_rel_level_cat, batch_n2g_index, batch_z_t_cat, batch_t_cat, \
            query2g, num_query_cumsum, batch_z_cat


def collate_edge_pred(data):
    data = [d for d in data if d is not None]
    if not data:  # Empty or all None batch
        num_elements = 12 if (len(data) > 0 and len(data[0]) == 12) else 11
        empty_batch = [
            torch.empty((2, 0), dtype=torch.long),  # input_edge_index
            torch.empty((2, 0), dtype=torch.long),  # noisy_edge_index
            torch.empty((0,), dtype=torch.long),  # x_n
            torch.empty((0, 1), dtype=torch.float),  # abs_level
            torch.empty((0, 1), dtype=torch.float),  # rel_level
            torch.empty((0, 1), dtype=torch.long),  # t (for queries)
        ]
        if num_elements == 12:  # Conditional
            empty_batch.append(torch.empty((0, 0), dtype=torch.float))  # y
        empty_batch.extend([
            torch.empty((0,), dtype=torch.long),  # query_src
            torch.empty((0,), dtype=torch.long),  # query_dst
            torch.empty((0,), dtype=torch.long)  # label (for queries)
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
    elif len(data[0]) == 11:  # Unconditional
        batch_input_src, batch_input_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, \
            batch_abs_level, batch_rel_level, batch_t_list, \
            batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        batch_y_tensor = None
    else:
        raise ValueError(f"Unexpected number of items in data for collate_edge_pred: {len(data[0])}")

    # Common processing for nodes (similar to collate_common but without returning batch_size and n2g_index here)
    valid_indices_nodes = [i for i, item_x_n in enumerate(batch_x_n) if item_x_n is not None and len(item_x_n) > 0]
    if not valid_indices_nodes:  # If no items have nodes, return empty structure
        # This case should be rare if __getitem__ filters properly or dag_dataset is non-empty
        # Re-use the empty batch logic from the start of the function
        return collate_edge_pred([])

    # Filter all lists based on valid_indices_nodes
    batch_input_src = [batch_input_src[i] for i in valid_indices_nodes]
    batch_input_dst = [batch_input_dst[i] for i in valid_indices_nodes]
    batch_noisy_src = [batch_noisy_src[i] for i in valid_indices_nodes]
    batch_noisy_dst = [batch_noisy_dst[i] for i in valid_indices_nodes]
    batch_x_n = [batch_x_n[i] for i in valid_indices_nodes]
    batch_abs_level = [batch_abs_level[i] for i in valid_indices_nodes]
    batch_rel_level = [batch_rel_level[i] for i in valid_indices_nodes]
    batch_t_list = [batch_t_list[i] for i in valid_indices_nodes]
    if is_conditional_item:
        batch_y_tensor = batch_y_tensor[valid_indices_nodes] if batch_y_tensor is not None else None  # Filter y as well
    batch_query_src = [batch_query_src[i] for i in valid_indices_nodes]
    batch_query_dst = [batch_query_dst[i] for i in valid_indices_nodes]
    batch_label = [batch_label[i] for i in valid_indices_nodes]

    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n_i) for x_n_i in batch_x_n]), dim=0)  # batch_x_n now only has valid items

    batch_x_n_cat = torch.cat(batch_x_n, dim=0).long()
    batch_abs_level_cat = torch.cat(batch_abs_level, dim=0).float().unsqueeze(-1)
    batch_rel_level_cat = torch.cat(batch_rel_level, dim=0).float().unsqueeze(-1)

    # Process GNN input edges
    input_src_cat_list, input_dst_cat_list = [], []
    for i in range(len(batch_x_n)):  # Iterate over valid items
        if batch_input_src[i].numel() > 0:
            input_src_cat_list.append(batch_input_src[i] + num_nodes_cumsum[i])
            input_dst_cat_list.append(batch_input_dst[i] + num_nodes_cumsum[i])
    input_src_b = torch.cat(input_src_cat_list) if input_src_cat_list else torch.empty(0, dtype=torch.long)
    input_dst_b = torch.cat(input_dst_cat_list) if input_dst_cat_list else torch.empty(0, dtype=torch.long)
    input_edge_index_b = torch.stack([input_dst_b, input_src_b]) if input_src_b.numel() > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long)

    # Process noisy edges (for diffusion model input)
    noisy_src_cat_list, noisy_dst_cat_list = [], []
    for i in range(len(batch_x_n)):
        if batch_noisy_src[i].numel() > 0:
            noisy_src_cat_list.append(batch_noisy_src[i] + num_nodes_cumsum[i])
            noisy_dst_cat_list.append(batch_noisy_dst[i] + num_nodes_cumsum[i])
    noisy_src_b = torch.cat(noisy_src_cat_list) if noisy_src_cat_list else torch.empty(0, dtype=torch.long)
    noisy_dst_b = torch.cat(noisy_dst_cat_list) if noisy_dst_cat_list else torch.empty(0, dtype=torch.long)
    noisy_edge_index_b = torch.stack([noisy_dst_b, noisy_src_b]) if noisy_src_b.numel() > 0 else torch.empty((2, 0),
                                                                                                             dtype=torch.long)

    # Process query edges and their associated timesteps and labels
    query_src_cat_list, query_dst_cat_list = [], []
    t_for_queries_list, labels_for_queries_list = [], []
    for i in range(len(batch_x_n)):  # Iterate over valid items
        if batch_query_src[i].numel() > 0:  # If this item has query edges
            query_src_cat_list.append(batch_query_src[i] + num_nodes_cumsum[i])
            query_dst_cat_list.append(batch_query_dst[i] + num_nodes_cumsum[i])
            t_for_queries_list.append(batch_t_list[i])  # batch_t_list[i] is already (num_queries_i, 1)
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
    else:  # Unconditional
        return input_edge_index_b, noisy_edge_index_b, batch_x_n_cat, \
            batch_abs_level_cat, batch_rel_level_cat, \
            t_b, \
            query_src_b, query_dst_b, label_b
