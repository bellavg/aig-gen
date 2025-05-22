# extension_data.py
import numpy as np
import networkx as nx
import torch
import random
import os

# Make sure aig_config.py is accessible and defines the new EDGE_TYPE_KEYS and ENCODING
try:
    from aig_config import (
        NUM_EDGE_FEATURES as DEFAULT_NUM_EDGE_FEATURES,
        NUM_NODE_FEATURES as DEFAULT_NUM_NODE_FEATURES,
        EDGE_LABEL_ENCODING,  # For NO_EDGE_ENCODING_TENSOR
        EDGE_TYPE_KEYS  # For consistency checks if needed
    )

    # Ensure EDGE_NO_EDGE is defined and is the first one for index 0
    if "EDGE_NO_EDGE" not in EDGE_LABEL_ENCODING or EDGE_TYPE_KEYS[0] != "EDGE_NO_EDGE":
        raise ImportError("EDGE_NO_EDGE not configured correctly as index 0 in aig_config.py")
    NO_EDGE_ENCODING_TENSOR = torch.tensor(EDGE_LABEL_ENCODING["EDGE_NO_EDGE"], dtype=torch.float)
    # Get specific encodings for REG and INV to use in the transformation
    EDGE_REG_ENCODING_3FEATURE = np.array(EDGE_LABEL_ENCODING.get("EDGE_REG", [0.0, 1.0, 0.0]), dtype=np.float32)
    EDGE_INV_ENCODING_3FEATURE = np.array(EDGE_LABEL_ENCODING.get("EDGE_INV", [0.0, 0.0, 1.0]), dtype=np.float32)


except ImportError as e:
    DEFAULT_NUM_NODE_FEATURES = -1
    DEFAULT_NUM_EDGE_FEATURES = -1
    NO_EDGE_ENCODING_TENSOR = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float)  # Fallback if aig_config fails
    EDGE_REG_ENCODING_3FEATURE = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Fallback
    EDGE_INV_ENCODING_3FEATURE = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Fallback
    print(
        f"Warning: Could not fully import from aig_config.py ({e}). "
        "NUM_NODE_FEATURES, NUM_EDGE_FEATURES, and encodings might use fallbacks."
        "Ensure aig_config.py is correct and accessible."
    )


# Helper function to convert a single PyG Data object to NetworkX DiGraph
def pyg_to_nx_directed_with_types(pyg_data, num_node_features, num_edge_features):
    """
    Converts a PyTorch Geometric Data object back to a NetworkX DiGraph
    with one-hot 'type' attributes on nodes and edges.
    Nodes in the returned graph will be integers 0 to N-1.
    Args:
        pyg_data: A PyTorch Geometric Data object.
        num_node_features (int): The length of the one-hot node feature vector (e.g., 4 for AIG nodes).
        num_edge_features (int): The length of the one-hot edge feature vector (e.g., 3 for AIG edges: NO_EDGE, REG, INV).

    Returns:
        nx.DiGraph: A NetworkX DiGraph with 'type' attributes.
    """
    G = nx.DiGraph()

    # Ensure data is on CPU before converting to numpy
    node_features = pyg_data.x.cpu().numpy()
    num_nodes_in_pyg = node_features.shape[0]

    if num_node_features == -1:  # Check if placeholder is used
        raise ValueError(
            "num_node_features was not properly initialized. Check aig_config.py or constructor arguments for DirectedGraphDataSet.")
    if num_edge_features == -1:  # Check if placeholder is used
        raise ValueError(
            "num_edge_features was not properly initialized. Check aig_config.py or constructor arguments for DirectedGraphDataSet.")

    if node_features.shape[1] != num_node_features:
        raise ValueError(
            f"Node features in PyG data have {node_features.shape[1]} dimensions, but expected {num_node_features}")

    for i in range(num_nodes_in_pyg):
        G.add_node(i, type=list(node_features[i]))

    edge_index = pyg_data.edge_index.cpu().numpy()
    if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None and pyg_data.edge_attr.shape[0] > 0:
        edge_attributes_original = pyg_data.edge_attr.cpu().numpy()

        # MODIFICATION: Handle 2-feature edge_attr and convert to 3-feature
        if edge_attributes_original.shape[1] == 2 and num_edge_features == 3:
            print(
                f"Info: Detected 2-feature edge_attr, converting to {num_edge_features}-feature for {edge_attributes_original.shape[0]} edges.")
            transformed_edge_attributes = np.zeros((edge_attributes_original.shape[0], num_edge_features),
                                                   dtype=np.float32)
            # Assuming original 2-feature: [1,0] for REG, [0,1] for INV
            # And target 3-feature encodings are from aig_config.py
            # EDGE_REG_ENCODING_3FEATURE (e.g., [0,1,0])
            # EDGE_INV_ENCODING_3FEATURE (e.g., [0,0,1])

            # Determine what [1,0] and [0,1] in 2-feature space mean.
            # If your data generation script used argmax, then [1,0] -> 0, [0,1] -> 1.
            # Let's assume 2-feature [1,0] was intended as the first type (REG) and [0,1] as the second type (INV).

            for i in range(edge_attributes_original.shape[0]):
                if np.array_equal(edge_attributes_original[i], [1.0, 0.0]):  # Was REG
                    transformed_edge_attributes[i] = EDGE_REG_ENCODING_3FEATURE
                elif np.array_equal(edge_attributes_original[i], [0.0, 1.0]):  # Was INV
                    transformed_edge_attributes[i] = EDGE_INV_ENCODING_3FEATURE
                else:
                    # Fallback or error for unexpected 2-feature vectors
                    print(
                        f"Warning: Unrecognized 2-feature vector {edge_attributes_original[i]} at edge index {i}. Defaulting to NO_EDGE encoding.")
                    transformed_edge_attributes[i] = np.array(EDGE_LABEL_ENCODING.get("EDGE_NO_EDGE"), dtype=np.float32)
            edge_attributes = transformed_edge_attributes
        else:
            edge_attributes = edge_attributes_original
        # END MODIFICATION

        if edge_attributes.shape[1] != num_edge_features:
            raise ValueError(
                f"Edge attributes in PyG data have {edge_attributes.shape[1]} features, but expected {num_edge_features} (for NO_EDGE, REG, INV). "
                "Ensure your PyG data's edge_attr uses the correct 3-feature one-hot encoding (e.g., REG=[0,1,0], INV=[0,0,1]).")
        for j in range(edge_index.shape[1]):
            src_node, tgt_node = edge_index[0, j].item(), edge_index[1, j].item()
            G.add_edge(src_node, tgt_node, type=list(edge_attributes[j]))
    elif edge_index.shape[1] > 0:  # Edges exist but no attributes
        print(
            f"Warning: Graph has {edge_index.shape[1]} edges but no/empty edge_attr. "
            f"Edges in NetworkX graph will be assigned a default 'type' of all zeros (length {num_edge_features}). "
            "This might not align with the NO_EDGE encoding [1,0,0].")
        # Defaulting to all zeros might be problematic if [1,0,0] is NO_EDGE.
        # It's better if PyG data always has edge_attr if edges exist.
        default_edge_type = [0.0] * num_edge_features
        for j in range(edge_index.shape[1]):
            src_node, tgt_node = edge_index[0, j].item(), edge_index[1, j].item()
            G.add_edge(src_node, tgt_node, type=default_edge_type)

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
    if len(traversal_indices) != n:  # Should not happen if graph is fully traversed
        traversal_indices = list(np.random.permutation(n))  # Fallback
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

        if self.num_node_features == -1 or self.num_edge_features == -1:  # Check if placeholders are still used
            raise ValueError("num_node_features and/or num_edge_features were not properly initialized. "
                             "Ensure aig_config.py is correct and accessible, or provide these values in the YAML config "
                             "and ensure train.py passes them to this constructor.")

        print(
            f"DirectedGraphDataSet initialized with: num_node_features={self.num_node_features}, num_edge_features={self.num_edge_features}")

        print(f"Loading PyG data from: {pyg_file_path}")
        try:
            self.pyg_data_list = torch.load(pyg_file_path, map_location='cpu', weights_only=False)
            if not isinstance(self.pyg_data_list, list):
                if 'torch_geometric.data.data.Data' in str(type(self.pyg_data_list)):
                    print("Note: Loaded a single PyG Data object, wrapping it in a list.")
                    self.pyg_data_list = [self.pyg_data_list]
                else:
                    raise TypeError(
                        f"Expected {pyg_file_path} to contain a list of PyG Data objects, but got {type(self.pyg_data_list)}.")
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
                current_num_nodes = 0
                if hasattr(pyg_data_item, 'num_nodes') and isinstance(pyg_data_item.num_nodes, (int, torch.Tensor)):
                    current_num_nodes = pyg_data_item.num_nodes.item() if isinstance(pyg_data_item.num_nodes,
                                                                                     torch.Tensor) else pyg_data_item.num_nodes
                elif hasattr(pyg_data_item, 'x') and pyg_data_item.x is not None:
                    current_num_nodes = pyg_data_item.x.shape[0]
                else:
                    print("Warning: PyG item found without 'num_nodes' or 'x' attribute for size calculation.")

                if current_num_nodes > self.max_node_count:
                    self.max_node_count = current_num_nodes
            print(f"Determined max_node_count: {self.max_node_count}")

        indices = list(range(len(self.pyg_data_list)))
        np.random.shuffle(indices)  # Shuffle indices for train/test split
        train_size = int(len(self.pyg_data_list) * train_split)

        if training:
            self.current_indices = indices[:train_size]
        else:
            self.current_indices = indices[train_size:]

        self.length = len(self.current_indices)
        print(f"Dataset split: {'Training' if training else 'Test'} size: {self.length}")

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
            self.num_edge_features  # This should be 3
        )

        num_nodes_in_graph = g.number_of_nodes()

        # Prepare padded node attributes (target for GraphLevelRNN's node attribute predictor)
        padded_node_attr_onehot = torch.zeros(self.max_node_count, self.num_node_features, dtype=torch.float)
        # Initialize padded x_adj (target for EdgeLevelRNN/MLP) with NO_EDGE encoding
        # NO_EDGE_ENCODING_TENSOR shape: [num_edge_features]
        padded_x_adj = NO_EDGE_ENCODING_TENSOR.clone().detach().reshape(1, 1, -1).expand(self.max_node_count, self.m,
                                                                                         self.num_edge_features)

        if num_nodes_in_graph == 0:
            return {
                'x_adj': padded_x_adj,  # Will be all NO_EDGE if graph is empty
                'node_attr_onehot': padded_node_attr_onehot,  # All zeros
                'len': torch.tensor(0, dtype=torch.long)
            }

        if self.dataset_type == 'aig-custom-topsort':
            # Assuming nodes in PyG data are already in topological order (0 to N-1)
            # If not, a topological sort should happen here or before pyg_to_nx_directed_with_types
            if not nx.is_directed_acyclic_graph(g):
                print(
                    f"Warning: Graph (from PyG index {original_pyg_idx}, NX graph has {g.number_of_nodes()} nodes) is not a DAG. "
                    "Sequential processing might be problematic.")

            # The nodes in 'g' are 0...num_nodes_in_graph-1, assumed to be in topological order
            processing_node_order = list(range(num_nodes_in_graph))

            node_attr_list = []
            for node_id in processing_node_order:  # node_id is 0, 1, ...
                try:
                    attr = g.nodes[node_id]['type']  # This is a list/numpy array from pyg_to_nx
                    if not isinstance(attr, (list, np.ndarray)) or len(attr) != self.num_node_features:
                        raise ValueError(
                            f"Node {node_id} 'type' (from PyG index {original_pyg_idx}) has wrong format or length ({len(attr)}). Expected {self.num_node_features}")
                    node_attr_list.append(torch.tensor(attr, dtype=torch.float))
                except KeyError:
                    raise ValueError(
                        f"Node {node_id} (from PyG index {original_pyg_idx}) is missing 'type' attribute after PyG->NX conversion.")

            if node_attr_list:  # Should always be true if num_nodes_in_graph > 0
                stacked_node_attrs = torch.stack(node_attr_list)
                padded_node_attr_onehot[:num_nodes_in_graph] = stacked_node_attrs

            # Construct x_adj_list (list of S_i tensors)
            x_adj_list_for_stacking = []
            for i_idx_in_order, current_node_label in enumerate(
                    processing_node_order):  # current_node_label is 0, 1, ...
                # s_i is the M-window adjacency vector for current_node_label
                # Initialize with NO_EDGE encoding
                s_i = NO_EDGE_ENCODING_TENSOR.clone().detach().unsqueeze(0).repeat(self.m,
                                                                                   1)  # Shape [m, num_edge_features]

                for p_window_idx in range(self.m):  # Iterate through M previous slots
                    # pred_relative_idx is the index in processing_node_order of the potential predecessor
                    # p_window_idx = 0 means immediate predecessor (current_node_label - 1)
                    # p_window_idx = 1 means (current_node_label - 2), etc.
                    pred_relative_idx = i_idx_in_order - 1 - p_window_idx

                    if pred_relative_idx >= 0:
                        predecessor_node_label_in_order = processing_node_order[pred_relative_idx]

                        # Check if an edge exists from that predecessor to the current node
                        if g.has_edge(predecessor_node_label_in_order, current_node_label):
                            try:
                                edge_attr = g.edges[predecessor_node_label_in_order, current_node_label]['type']
                                # edge_attr should be a list/numpy array from pyg_to_nx_directed_with_types
                                if not isinstance(edge_attr, (list, np.ndarray)) or len(
                                        edge_attr) != self.num_edge_features:
                                    raise ValueError(
                                        f"Edge ({predecessor_node_label_in_order}->{current_node_label}) 'type' (from PyG index {original_pyg_idx}) "
                                        f"has wrong format/length ({len(edge_attr)}). Expected {self.num_edge_features} (for NO_EDGE, REG, INV).")
                                s_i[p_window_idx, :] = torch.tensor(edge_attr, dtype=torch.float)
                            except KeyError:
                                # This should not happen if g.has_edge is true and pyg_to_nx works correctly
                                raise ValueError(
                                    f"Edge ({predecessor_node_label_in_order}->{current_node_label}) (from PyG index {original_pyg_idx}) "
                                    f"is missing 'type' attribute despite g.has_edge being true.")
                        # else: No edge from this predecessor in this M-window slot, s_i[p_window_idx, :] remains NO_EDGE_ENCODING_TENSOR
                x_adj_list_for_stacking.append(s_i)

            if x_adj_list_for_stacking:  # If there were any nodes
                stacked_x_adj = torch.stack(x_adj_list_for_stacking)  # Shape [num_nodes_in_graph, m, num_edge_features]
                padded_x_adj[:num_nodes_in_graph] = stacked_x_adj  # Fill in actual data

            return {
                'x_adj': padded_x_adj,
                'node_attr_onehot': padded_node_attr_onehot,
                'len': torch.tensor(num_nodes_in_graph, dtype=torch.long)
            }
        else:
            raise Exception(f"Unsupported dataset_type: {self.dataset_type} in __getitem__ of DirectedGraphDataSet")