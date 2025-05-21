# extension_data.py
import numpy as np
import networkx as nx
import torch
import random

# Make sure aig_config.py is accessible if these are not passed explicitly
# For robust code, it's better if num_node_features and num_edge_features
# are always passed to __init__ from your config.
try:
    from aig_config import NUM_EDGE_FEATURES as DEFAULT_NUM_EDGE_FEATURES
    from aig_config import NUM_NODE_FEATURES as DEFAULT_NUM_NODE_FEATURES
except ImportError:
    # Define placeholders if aig_config is not found,
    # but this means num_node_features and num_edge_features MUST be passed to __init__
    DEFAULT_NUM_NODE_FEATURES = -1  # Placeholder, indicates it must be provided
    DEFAULT_NUM_EDGE_FEATURES = -1  # Placeholder
    print(
        "Warning: aig_config.py not found. NUM_NODE_FEATURES and NUM_EDGE_FEATURES must be passed to DirectedGraphDataSet.")

import os


# Helper function to convert a single PyG Data object to NetworkX DiGraph
def pyg_to_nx_directed_with_types(pyg_data, num_node_features, num_edge_features):
    """
    Converts a PyTorch Geometric Data object back to a NetworkX DiGraph
    with one-hot 'type' attributes on nodes and edges.
    Nodes in the returned graph will be integers 0 to N-1.
    Args:
        pyg_data: A PyTorch Geometric Data object.
        num_node_features (int): The length of the one-hot node feature vector.
        num_edge_features (int): The length of the one-hot edge feature vector.

    Returns:
        nx.DiGraph: A NetworkX DiGraph with 'type' attributes.
    """
    G = nx.DiGraph()

    # Ensure data is on CPU before converting to numpy
    node_features = pyg_data.x.cpu().numpy()
    num_nodes_in_pyg = node_features.shape[0]

    if num_node_features == -1 or num_edge_features == -1:
        raise ValueError(
            "num_node_features or num_edge_features was not properly initialized. Check aig_config.py or constructor arguments.")

    if node_features.shape[1] != num_node_features:
        raise ValueError(
            f"Node features in PyG data have {node_features.shape[1]} dimensions, but expected {num_node_features}")

    for i in range(num_nodes_in_pyg):
        G.add_node(i, type=list(node_features[i]))  # Nodes are 0, 1, ..., N-1

    edge_index = pyg_data.edge_index.cpu().numpy()
    if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None and pyg_data.edge_attr.shape[0] > 0:
        edge_attributes = pyg_data.edge_attr.cpu().numpy()
        if edge_attributes.shape[1] != num_edge_features:
            raise ValueError(
                f"Edge attributes in PyG data have {edge_attributes.shape[1]} features, but expected {num_edge_features}")
        for j in range(edge_index.shape[1]):
            src_node, tgt_node = edge_index[0, j].item(), edge_index[1, j].item()
            G.add_edge(src_node, tgt_node, type=list(edge_attributes[j]))
    elif edge_index.shape[1] > 0:  # Edges exist but no attributes
        print(
            f"Warning: Graph has {edge_index.shape[1]} edges but no/empty edge_attr. Edges in NetworkX graph will be assigned a default zero 'type'.")
        for j in range(edge_index.shape[1]):
            src_node, tgt_node = edge_index[0, j].item(), edge_index[1, j].item()
            G.add_edge(src_node, tgt_node, type=[0.0] * num_edge_features)

    return G


def original_topological_sort_for_binary_matrix(g):
    """
    Original topological_sort function from the provided script.
    Returns the adjacency matrix of G, reordered according to a random topological sort.
    This produces a binary adjacency matrix.
    """
    a = nx.to_numpy_array(g)
    reordering = np.random.permutation(list(g.nodes()))
    node_map = {node: i for i, node in enumerate(reordering)}
    temp_g = nx.DiGraph()
    temp_g.add_nodes_from(range(g.number_of_nodes()))
    temp_g.add_edges_from((node_map[u], node_map[v]) for u, v in g.edges())
    permuted_matrix_a = nx.to_numpy_array(temp_g, nodelist=sorted(temp_g.nodes()))
    try:
        topsort_indices = list(nx.lexicographical_topological_sort(temp_g))
    except nx.NetworkXUnfeasible:
        print(f"Warning: Graph {g} is not a DAG, cannot topologically sort. Returning identity permutation.")
        topsort_indices = list(range(g.number_of_nodes()))
    return permuted_matrix_a[topsort_indices][:, topsort_indices]


def bfs_permute(g):
    """Randomly permutes given DIRECTED graph, performs BFS, and returns
        the adjacency matrix reordered by a randomized BFS traversal.
    """
    a = nx.to_numpy_array(g)
    n = g.number_of_nodes()
    reordering = np.random.permutation(list(g.nodes()))
    node_map = {node: i for i, node in enumerate(reordering)}
    temp_g = nx.DiGraph()
    temp_g.add_nodes_from(range(n))
    temp_g.add_edges_from((node_map[u], node_map[v]) for u, v in g.edges())
    a_reord = nx.to_numpy_array(temp_g, nodelist=sorted(temp_g.nodes()))
    visited_nodes = set()
    unvisited_nodes = set(temp_g.nodes())
    traversal_indices = []
    while len(visited_nodes) < n:
        if not unvisited_nodes: break
        src = random.choice(tuple(unvisited_nodes))
        bfs_q = [src]
        visited_component = {src}
        head = 0
        while head < len(bfs_q):
            curr = bfs_q[head];
            head += 1
            if curr not in visited_nodes:
                traversal_indices.append(curr)
                visited_nodes.add(curr)
                if curr in unvisited_nodes: unvisited_nodes.remove(curr)
                for neighbor in temp_g.neighbors(curr):
                    if neighbor not in visited_component and neighbor not in visited_nodes:
                        visited_component.add(neighbor)
                        bfs_q.append(neighbor)
    if len(traversal_indices) < n:
        remaining_nodes = list(unvisited_nodes)
        random.shuffle(remaining_nodes)
        traversal_indices.extend(remaining_nodes)
    if len(traversal_indices) != n:
        traversal_indices = list(np.random.permutation(n))
    return a_reord[traversal_indices][:, traversal_indices]


class DirectedGraphDataSet(torch.utils.data.Dataset):
    """Dataset to handle directed DAGs, loading from a PyG file and converting on-the-fly."""

    def __init__(self, dataset_type, m, pyg_file_path,
                 num_node_features=DEFAULT_NUM_NODE_FEATURES,
                 num_edge_features=DEFAULT_NUM_EDGE_FEATURES,
                 training=True, train_split=0.8):
        self.dataset_type = dataset_type
        self.m = m
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        if self.num_node_features == -1 or self.num_edge_features == -1:
            raise ValueError("num_node_features and/or num_edge_features were not provided to DirectedGraphDataSet "
                             "and could not be imported from aig_config.py. Please provide them via config.")

        print(f"Loading PyG data from: {pyg_file_path}")
        try:
            self.pyg_data_list = torch.load(pyg_file_path, map_location='cpu')
            if not isinstance(self.pyg_data_list, list):
                raise TypeError(f"Expected {pyg_file_path} to contain a list of PyG Data objects.")
            print(f"Loaded {len(self.pyg_data_list)} PyG Data objects.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The PyG data file was not found at {pyg_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing {pyg_file_path}: {e}")

        if not self.pyg_data_list:
            print("Warning: No graphs loaded from PyG file. Dataset will be empty.")
            self.max_node_count = 0
        else:
            self.max_node_count = 0
            for pyg_data_item in self.pyg_data_list:
                if hasattr(pyg_data_item, 'num_nodes') and isinstance(pyg_data_item.num_nodes, (int, torch.Tensor)):
                    current_num_nodes = pyg_data_item.num_nodes.item() if isinstance(pyg_data_item.num_nodes,
                                                                                     torch.Tensor) else pyg_data_item.num_nodes
                    if current_num_nodes > self.max_node_count:
                        self.max_node_count = current_num_nodes
                elif hasattr(pyg_data_item, 'x') and pyg_data_item.x is not None:
                    if pyg_data_item.x.shape[0] > self.max_node_count:
                        self.max_node_count = pyg_data_item.x.shape[0]
                else:
                    print("Warning: PyG item found without 'num_nodes' or 'x' attribute for size calculation.")

        indices = list(range(len(self.pyg_data_list)))
        train_size = int(len(self.pyg_data_list) * train_split)

        if training:
            self.current_indices = indices[:train_size]
        else:
            self.current_indices = indices[train_size:]

        self.length = len(self.current_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx_in_split):
        if not (0 <= idx_in_split < self.length):
            raise IndexError(f"Index {idx_in_split} is out of bounds for the current split of length {self.length}")

        original_pyg_idx = self.current_indices[idx_in_split]
        pyg_data_item = self.pyg_data_list[original_pyg_idx]

        g = pyg_to_nx_directed_with_types(
            pyg_data_item,
            self.num_node_features,
            self.num_edge_features
        )

        num_nodes_in_graph = g.number_of_nodes()

        if num_nodes_in_graph == 0:
            return {
                'x_adj': torch.zeros(self.max_node_count, self.m, self.num_edge_features, dtype=torch.float),
                'node_attr_onehot': torch.zeros(self.max_node_count, self.num_node_features, dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long)
            }

        if self.dataset_type == 'aig-custom-topsort':
            # MODIFICATION: Use the inherent node order (0 to N-1) as the topological sort
            # This assumes the PyG data (and thus 'g') has nodes 0..N-1 in the desired topological order.
            if not nx.is_directed_acyclic_graph(g):
                print(
                    f"Warning: Graph (from PyG index {original_pyg_idx}) is not a DAG, though node order is preserved. "
                    "Sequential processing might be problematic.")
                # Depending on severity, you might return empty or try to proceed.
                # For now, we'll proceed, but this is a critical assumption for GraphRNN.

            # The nodes in 'g' are 0, 1, ..., num_nodes_in_graph - 1.
            # This list will serve as the processing order.
            processing_node_order = list(range(num_nodes_in_graph))

            node_attr_list = []
            for node_id in processing_node_order:  # Iterate in the 0 to N-1 order
                try:
                    attr = g.nodes[node_id]['type']
                    if not isinstance(attr, (list, np.ndarray)) or len(attr) != self.num_node_features:
                        raise ValueError(
                            f"Node {node_id} 'type' (from PyG index {original_pyg_idx}) has wrong format or length ({len(attr)}). Expected {self.num_node_features}")
                    node_attr_list.append(torch.tensor(attr, dtype=torch.float))
                except KeyError:
                    raise ValueError(
                        f"Node {node_id} (from PyG index {original_pyg_idx}) is missing 'type' attribute after PyG->NX conversion.")

            padded_node_attr_onehot = torch.zeros(self.max_node_count, self.num_node_features, dtype=torch.float)
            if node_attr_list:
                stacked_node_attrs = torch.stack(node_attr_list)
                padded_node_attr_onehot[:num_nodes_in_graph] = stacked_node_attrs

            x_adj_list = []
            # For node 'i' in the processing_node_order (which is node with original label 'i')
            # its connections S_i are to predecessors p_window_idx in that same order.
            for i_idx_in_order, current_node_label in enumerate(processing_node_order):
                s_i = torch.zeros(self.m, self.num_edge_features, dtype=torch.float)
                for p_window_idx in range(self.m):  # p_window_idx is 0 for immediate predecessor, 1 for next, etc.
                    # The actual label of the predecessor node in the processing_node_order
                    # is processing_node_order[i_idx_in_order - 1 - p_window_idx]
                    pred_relative_idx = i_idx_in_order - 1 - p_window_idx

                    if pred_relative_idx >= 0:
                        predecessor_node_label = processing_node_order[pred_relative_idx]

                        if g.has_edge(predecessor_node_label, current_node_label):
                            try:
                                edge_attr = g.edges[predecessor_node_label, current_node_label]['type']
                                if not isinstance(edge_attr, (list, np.ndarray)) or len(
                                        edge_attr) != self.num_edge_features:
                                    raise ValueError(
                                        f"Edge ({predecessor_node_label}-{current_node_label}) 'type' (from PyG index {original_pyg_idx}) has wrong format/length ({len(edge_attr)}). Expected {self.num_edge_features}")
                                # s_i stores connection to (immediate pred, pred-1, pred-2, ...)
                                # So s_i[0] is connection to immediate predecessor (p_window_idx=0)
                                s_i[p_window_idx, :] = torch.tensor(edge_attr, dtype=torch.float)
                            except KeyError:
                                raise ValueError(
                                    f"Edge ({predecessor_node_label}-{current_node_label}) (from PyG index {original_pyg_idx}) is missing 'type' attribute after PyG->NX conversion.")
                x_adj_list.append(s_i)

            padded_x_adj = torch.zeros(self.max_node_count, self.m, self.num_edge_features, dtype=torch.float)
            if x_adj_list:
                stacked_x_adj = torch.stack(x_adj_list)
                padded_x_adj[:num_nodes_in_graph] = stacked_x_adj

            return {
                'x_adj': padded_x_adj,
                'node_attr_onehot': padded_node_attr_onehot,
                'len': torch.tensor(num_nodes_in_graph, dtype=torch.long)
            }
        else:
            raise Exception(f"Unsupported dataset_type: {self.dataset_type} in __getitem__ of DirectedGraphDataSet")

