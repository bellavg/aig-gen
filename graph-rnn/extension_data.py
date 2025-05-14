import numpy as np
import networkx as nx
import torch
import random
from aig_config import NUM_EDGE_FEATURES, NUM_NODE_FEATURES


# It's good practice to have these defined or passed if they vary
# For now, assuming they might be passed to __init__ or are globally accessible
# from your aig_config.py if this script is in the same environment.
# For this example, I'll add them as parameters to __init__.


def original_topological_sort_for_binary_matrix(g):
    """
    Original topological_sort function from the provided script.
    Returns the adjacency matrix of G, reordered according to a random topological sort.
    This produces a binary adjacency matrix.
    """
    a = nx.to_numpy_array(g)  # By default, uses weight 'weight', otherwise 1 for presence.
    # For unweighted graphs, this is a binary matrix.
    # n = g.number_of_nodes() # Not used in this function

    # Random permutation of node labels before topological sort to get a random valid topsort
    reordering = np.random.permutation(list(g.nodes()))

    # Create a temporary graph with integer labels 0..N-1 based on the permutation
    # This is important if original node labels are not integers or not contiguous
    node_map = {node: i for i, node in enumerate(reordering)}
    temp_g = nx.DiGraph()
    temp_g.add_nodes_from(range(g.number_of_nodes()))
    temp_g.add_edges_from((node_map[u], node_map[v]) for u, v in g.edges())

    # Get the adjacency matrix of the consistently-labeled temp_g
    # Ensure it's binary if no weights are intended.
    # If g.edges have 'weight', nx.to_numpy_array will use them.
    # For GraphRNN's original topsort, a binary matrix representing structure is typical.
    # We'll get the matrix from temp_g to match the reordering.
    # Adjacency matrix of the permuted graph (nodes are 0 to N-1)
    permuted_matrix_a = nx.to_numpy_array(temp_g, nodelist=sorted(temp_g.nodes()))

    # Perform topological sort on the temp_g (whose nodes are 0..N-1)
    # nx.lexicographical_topological_sort provides a canonical sort if multiple exist
    try:
        topsort_indices = list(nx.lexicographical_topological_sort(temp_g))
    except nx.NetworkXUnfeasible:  # Graph has a cycle
        # This should ideally not happen if graphs are guaranteed DAGs
        # Fallback or raise error
        print(f"Warning: Graph {g} is not a DAG, cannot topologically sort. Returning identity permutation.")
        topsort_indices = list(range(g.number_of_nodes()))

    # Reorder the permuted_matrix_a according to the topsort_indices
    return permuted_matrix_a[topsort_indices][:, topsort_indices]


def bfs_permute(g):
    """Randomly permutes given DIRECTED graph, performs BFS, and returns
        the adjacency matrix reordered by a randomized BFS traversal.
    """
    a = nx.to_numpy_array(g)
    n = g.number_of_nodes()

    reordering = np.random.permutation(list(g.nodes()))
    # Create a temporary graph with integer labels 0..N-1
    node_map = {node: i for i, node in enumerate(reordering)}
    temp_g = nx.DiGraph()
    temp_g.add_nodes_from(range(n))
    temp_g.add_edges_from((node_map[u], node_map[v]) for u, v in g.edges())

    # Adjacency matrix of the permuted graph (nodes are 0 to N-1)
    a_reord = nx.to_numpy_array(temp_g, nodelist=sorted(temp_g.nodes()))

    visited_nodes = set()
    # Use nodes from temp_g which are 0..n-1
    unvisited_nodes = set(temp_g.nodes())
    traversal_indices = []  # Stores the new order of indices (0 to n-1)

    while len(visited_nodes) < n:
        # Ensure unvisited_nodes is not empty before choosing
        if not unvisited_nodes:
            break
        src = random.choice(tuple(unvisited_nodes))

        # BFS successors in temp_g
        # nx.bfs_successors returns an iterator of (node, [children])
        # We need the actual traversal order

        # Perform BFS from src to find all reachable nodes in this component
        bfs_q = [src]
        visited_component = {src}  # Keep track of visited in current BFS traversal

        head = 0
        while head < len(bfs_q):
            curr = bfs_q[head]
            head += 1

            if curr not in visited_nodes:  # Process if not globally visited
                traversal_indices.append(curr)
                visited_nodes.add(curr)
                if curr in unvisited_nodes:  # Check before removing
                    unvisited_nodes.remove(curr)

                for neighbor in temp_g.neighbors(curr):  # Successors of curr
                    if neighbor not in visited_component and neighbor not in visited_nodes:
                        visited_component.add(neighbor)
                        bfs_q.append(neighbor)

    # If graph is not connected, traversal_indices might be shorter than n
    # Pad with remaining unvisited nodes if any (though BFS should cover components)
    if len(traversal_indices) < n:
        remaining_nodes = list(unvisited_nodes)
        random.shuffle(remaining_nodes)  # Add them in a random order
        traversal_indices.extend(remaining_nodes)

    if len(traversal_indices) != n:
        # This case should be rare if BFS is implemented correctly for all components
        # Fallback to a simple permutation if traversal is incomplete
        # print(f"Warning: BFS traversal incomplete ({len(traversal_indices)}/{n}). Using random permutation.")
        traversal_indices = list(np.random.permutation(n))

    return a_reord[traversal_indices][:, traversal_indices]


class DirectedGraphDataSet(torch.utils.data.Dataset):
    """Dataset to handle directed DAGs and ego-networks in various representations"""

    def __init__(self, dataset_type, m, graphs_data,  # `graphs_data` is the list of loaded nx.Graphs
                 num_node_features=NUM_NODE_FEATURES, num_edge_features=NUM_EDGE_FEATURES,  # NEW: for the new mode
                 training=True, train_split=0.8):
        """
        Args:
            dataset_type (str): Type of dataset/processing mode.
                                e.g., 'aig-custom-topsort', 'dag-multiclass', 'ego-directed-topsort'.
            m (int): The M-window size for GraphRNN adjacency sequence.
            graphs_data (list): A list of NetworkX DiGraph objects.
                                For 'aig-custom-topsort', these graphs must have 'type' attributes
                                on nodes and edges (as one-hot lists/arrays).
            num_node_features (int): Dimensionality of node features (e.g., number of node types).
            num_edge_features (int): Dimensionality of edge features (e.g., number of edge types).
            training (bool): Loads the training split if True.
            train_split (float): Percentage of data for training.
        """
        self.dataset_type = dataset_type
        self.m = m  # M-window for GraphRNN
        self.graphs = graphs_data  # Directly use provided graphs
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        # Determine max_node_count from the provided graphs
        self.max_node_count = 0
        if self.graphs:
            for g in self.graphs:
                if g.number_of_nodes() > self.max_node_count:
                    self.max_node_count = g.number_of_nodes()
        else:  # Handle case of empty graphs_data list
            self.max_node_count = 0  # Or a default like 1 if 0 causes issues downstream

        # Original dataset loading logic based on string names (can be removed if graphs_data is always provided)
        # np.random.seed(42)
        # if dataset_type == 'dag-multiclass':
        #     self.graphs =  self.load_DAG_dataset() # These would need to be adapted or replaced
        # elif 'ego' in dataset_type and 'multiclass' in dataset_type: # e.g. ego-directed-multiclass
        #     self.graphs = self.load_citeseer_ego_dags()
        # elif 'ego' in dataset_type and 'topsort' in dataset_type: # e.g. ego-directed-topsort (original binary)
        #     self.graphs = self.load_citeseer_ego_dags()
        # elif dataset_type == 'aig-custom-topsort':
        #     # Graphs are assumed to be passed via graphs_data for this mode
        #     if not self.graphs:
        #          raise ValueError("For 'aig-custom-topsort', `graphs_data` must be provided to __init__.")
        # else:
        #     raise Exception(f"No data-loader for dataset_type `{dataset_type}` or graphs not provided.")

        if not self.graphs and self.max_node_count == 0:  # If graphs_data was empty
            print("Warning: No graphs provided to DirectedGraphDataSet. Dataset will be empty.")

        # Shuffle for random train/test split
        # Ensure self.graphs is a list before shuffling
        if isinstance(self.graphs, list):
            np.random.shuffle(self.graphs)
        else:  # If self.graphs is not a list (e.g. None or other type)
            print(
                "Warning: self.graphs is not a list, skipping shuffle. This might be an issue if train/test split is expected.")

        train_size = int(len(self.graphs) * train_split) if self.graphs else 0
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else (len(self.graphs) - train_size) if self.graphs else 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        graph_idx = self.start_idx + idx
        if graph_idx >= len(self.graphs):
            raise IndexError(
                f"Index {idx} (absolute {graph_idx}) is out of bounds for graphs list length {len(self.graphs)}")

        g = self.graphs[graph_idx]
        num_nodes_in_graph = g.number_of_nodes()

        if num_nodes_in_graph == 0:  # Handle empty graph
            # Return tensors of zeros with correct dimensions for padding
            # This ensures collate_fn doesn't break
            return {
                'x_adj': torch.zeros(self.max_node_count, self.m, self.num_edge_features, dtype=torch.float),
                'node_attr_onehot': torch.zeros(self.max_node_count, self.num_node_features, dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long)
            }

        # --- New 'aig-custom-topsort' mode (or similar name you choose) ---
        if self.dataset_type == 'aig-custom-topsort':
            try:
                # Node ordering based on topological sort
                sorted_node_ids = list(nx.topological_sort(g))
            except nx.NetworkXUnfeasible:  # Should not happen for AIGs
                print(f"Graph {idx} is not a DAG! Skipping or returning empty.")
                # Fallback: return empty/zero tensors to avoid crashing batch processing
                return {
                    'x_adj': torch.zeros(self.max_node_count, self.m, self.num_edge_features, dtype=torch.float),
                    'node_attr_onehot': torch.zeros(self.max_node_count, self.num_node_features, dtype=torch.float),
                    'len': torch.tensor(0, dtype=torch.long)
                }

            node_to_topo_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}

            # 1. Prepare 'node_attr_onehot'
            node_attr_list = []
            for node_id in sorted_node_ids:
                try:
                    attr = g.nodes[node_id]['type']
                    if not isinstance(attr, (list, np.ndarray)) or len(attr) != self.num_node_features:
                        raise ValueError(
                            f"Node {node_id} 'type' attribute has wrong format or length. Expected {self.num_node_features}")
                    node_attr_list.append(torch.tensor(attr, dtype=torch.float))
                except KeyError:
                    # Handle missing 'type' attribute - critical error for this mode
                    raise ValueError(f"Node {node_id} is missing 'type' attribute in graph {idx}.")

            # Pad node_attr_list
            padded_node_attr_onehot = torch.zeros(self.max_node_count, self.num_node_features, dtype=torch.float)
            if node_attr_list:  # Only stack if list is not empty
                stacked_node_attrs = torch.stack(node_attr_list)
                padded_node_attr_onehot[:num_nodes_in_graph] = stacked_node_attrs

            # 2. Prepare 'x_adj' (sequence of S_i vectors)
            x_adj_list = []  # Will store S_i for each node i
            for i in range(num_nodes_in_graph):  # For node S_i (current node is sorted_node_ids[i])
                current_node_id_original_label = sorted_node_ids[i]
                s_i = torch.zeros(self.m, self.num_edge_features,
                                  dtype=torch.float)  # Adjacency vector for current node

                for p_window_idx in range(self.m):  # Window index for predecessors
                    # Topological index of potential predecessor: i (current) - 1 (previous) - p (window offset)
                    pred_topo_idx = i - 1 - p_window_idx

                    if pred_topo_idx >= 0:  # If this predecessor exists in the sorted list
                        pred_node_id_original_label = sorted_node_ids[pred_topo_idx]

                        if g.has_edge(pred_node_id_original_label, current_node_id_original_label):
                            try:
                                edge_attr = g.edges[pred_node_id_original_label, current_node_id_original_label]['type']
                                if not isinstance(edge_attr, (list, np.ndarray)) or len(
                                        edge_attr) != self.num_edge_features:
                                    raise ValueError(
                                        f"Edge ({pred_node_id_original_label}-{current_node_id_original_label}) 'type' has wrong format or length. Expected {self.num_edge_features}")
                                s_i[p_window_idx, :] = torch.tensor(edge_attr, dtype=torch.float)
                            except KeyError:
                                # Handle missing 'type' attribute on edge - critical
                                raise ValueError(
                                    f"Edge ({pred_node_id_original_label}-{current_node_id_original_label}) is missing 'type' attribute in graph {idx}.")
                x_adj_list.append(s_i)

            # Pad x_adj_list
            padded_x_adj = torch.zeros(self.max_node_count, self.m, self.num_edge_features, dtype=torch.float)
            if x_adj_list:  # Only stack if list is not empty
                stacked_x_adj = torch.stack(x_adj_list)
                padded_x_adj[:num_nodes_in_graph] = stacked_x_adj

            return {
                'x_adj': padded_x_adj,
                'node_attr_onehot': padded_node_attr_onehot,
                'len': torch.tensor(num_nodes_in_graph, dtype=torch.long)
            }

        # --- Original 'ego-directed-topsort' (binary matrix output) ---
        elif self.dataset_type == 'ego-directed-topsort':
            # This mode returns a binary adjacency matrix, not compatible with train_rnn_step's new needs
            # It's kept for backward compatibility if other parts of the code use it.
            # You would typically not use this dataset_type with the modified train_rnn_step.
            g_bin = self.graphs[self.start_idx + idx]  # g_bin to denote it's for binary processing
            n_bin = g_bin.number_of_nodes()
            if n_bin == 0:
                return {'x': torch.zeros(self.max_node_count, self.max_node_count), 'len': torch.tensor(0)}

            # Uses the original_topological_sort_for_binary_matrix
            permuted_matrix_binary = original_topological_sort_for_binary_matrix(g_bin)

            # Original logic: .T[1:] - this seems specific and might need review for GraphRNN input.
            # GraphRNN typically takes S_i sequences. A full matrix might be for a different model.
            # For GraphRNN, 'x' should be the sequence of M-vectors.
            # This path is likely incompatible with the new train_rnn_step.
            # If you need to use this, it would require its own processing logic in train.py.

            # Returning a padded full matrix as per original logic for this specific mode.
            # This is likely NOT what train_rnn_step expects for 'x_adj'.
            # The original GraphRNN paper's S_i is different from a full adj matrix.
            # This mode seems to provide S_i as rows of the lower/upper triangular matrix.
            # The .T[1:] operation is unusual for GraphRNN's typical BFS-based S_i.
            # For topsort, S_i would be incoming edges from M predecessors.

            # For now, replicating the padding of the full matrix.
            # This 'x' is not directly usable by the modified train_rnn_step.
            final_matrix_for_output = permuted_matrix_binary  # Not doing .T[1:] to keep it N x N for now
            # as .T[1:] would make it N-1 x N.

            # Pad to max_node_count x max_node_count
            padded_binary_matrix = np.zeros((self.max_node_count, self.max_node_count))
            padded_binary_matrix[:n_bin, :n_bin] = final_matrix_for_output

            return {'x': torch.from_numpy(padded_binary_matrix).float(),
                    'len': torch.tensor(n_bin)}  # original returned n-1

        # --- Original 'multiclass' using BFS (e.g., 'dag-multiclass', 'ego-directed-multiclass') ---
        elif 'multiclass' in self.dataset_type:
            # This mode uses BFS and the 4-class edge encoding (no edge, fwd, bwd, bi).
            # This is also NOT directly compatible with the 'aig-custom-topsort' data needs.
            # It produces 'x' as a sequence of M-vectors, but the features are the 4-class encoding.
            g_mc = self.graphs[self.start_idx + idx]  # mc for multiclass
            n_mc = g_mc.number_of_nodes()
            if n_mc == 0:
                # For multiclass, x has shape [max_N, M, 4]
                return {'x': torch.zeros(self.max_node_count, self.m, 4), 'len': torch.tensor(0)}

            permuted_matrix_mc = bfs_permute(g_mc)  # Uses BFS

            # Augmented matrix for 4-class edge encoding
            augmented_matrix = permuted_matrix_mc + 2 * permuted_matrix_mc.T
            augmented_matrix = augmented_matrix.astype(int)

            num_edge_classes_original_multiclass = 4  # Fixed for this mode
            # tnsr shape: [N, N, 4]
            tnsr = np.zeros((n_mc, n_mc, num_edge_classes_original_multiclass))

            for i in range(n_mc):
                for j in range(i):  # Lower triangular part for BFS sequence
                    tnsr[i, j, augmented_matrix[i, j]] = 1  # One-hot encode the 0-3 class

            # Create sequence of M-vectors (S_i)
            scratch_mc = []
            for i in range(1, n_mc):  # For S_1 to S_{N-1} (node i is connected by row i to cols 0..i-1)
                # Critical strip: row i, columns from max(0, i-M) to i-1
                # This is S_i, representing connections from node i to its M predecessors in BFS order
                critical_strip = tnsr[i, max(i - self.m, 0):i, :]
                m_dash = critical_strip.shape[0]  # Actual number of predecessors in this strip

                # Pad to self.m (M-window) and reverse order as per original GraphRNN
                # Padded shape: [M, 4]
                padded_strip = np.pad(critical_strip,
                                      [(self.m - m_dash, 0), (0, 0)],
                                      mode='constant', constant_values=0)[::-1, :]
                scratch_mc.append(padded_strip)

            result_mc = np.array(scratch_mc)  # Shape [N-1, M, 4] if N > 0, else [0, M, 4]

            # Pad sequence length to max_node_count (or max_node_count-1 if result_mc is N-1 long)
            # The 'len' returned is num_nodes_in_graph, but result_mc is for N-1 S_i vectors.
            # Original GraphRNN data.py returns N-1 vectors for S_1 to S_{N-1}.
            # The train script then adds SOS.
            # Let's pad to max_node_count, assuming the first S_0 might be all zeros or handled by SOS.
            # If result_mc is empty (n_mc <=1), result_mc.shape[0] is 0.

            # Pad the sequence of S_i vectors
            # Padded shape: [max_node_count, M, 4]
            # (Assuming S_0 is implicitly all zeros or handled by SOS in training)
            # The original code pads to self.max_node_count, and result.shape[0] is N-1.
            # So, if S_0 is needed, it's missing here.
            # The train_rnn_step in graphrnn_train_with_node_attrs expects data['x_adj'] to be S0..SN-1.
            # This original multiclass mode produces S1..SN-1.
            # This is a discrepancy to be aware of if using this mode with the new train script.

            # For now, pad what we have (S1..SN-1)
            num_s_vectors = result_mc.shape[0] if n_mc > 1 else 0
            padded_result_mc = np.zeros((self.max_node_count, self.m, num_edge_classes_original_multiclass))
            if num_s_vectors > 0:
                # This places S1..SN-1 at indices 0..N-2 of the padded array.
                # If train script expects S0 at index 0, this needs adjustment.
                # Or, the train script's handling of SOS and slicing needs to be robust.
                padded_result_mc[:num_s_vectors] = result_mc

            return {
                'x': torch.from_numpy(padded_result_mc).float(),
                'len': torch.tensor(n_mc)  # Number of actual nodes
                # This 'x' is NOT 'x_adj' and doesn't include node_attr_onehot
            }
        else:
            raise Exception(f"Unsupported dataset_type: {self.dataset_type} in __getitem__")

    # --- Original Data Loading Methods (Kept for reference, may need adaptation/removal) ---
    def load_DAG_dataset(self, graph_count=3000, min_nodes=4, max_nodes=4):
        """Generate random `graph_count` weakly-connected DAGs.
           NOTE: These graphs will NOT have 'type' attributes for nodes/edges
           unless this function is modified.
        """
        retval = []
        # This self.max_node_count update should ideally happen in __init__
        # current_max_nodes_for_this_load = 0

        while len(retval) < graph_count:
            n = np.random.choice((range(min_nodes, max_nodes + 1)))
            A = np.zeros((n, n))
            for i in range(1, n):
                for j in range(i):  # Ensures DAG by only adding edges from lower to higher index
                    A[i][j] = random.choice([0, 1])

            G = nx.to_networkx_graph(A, create_using=nx.DiGraph)
            if nx.is_weakly_connected(G):  # Ensure graph is connected
                # current_max_nodes_for_this_load = max(current_max_nodes_for_this_load, G.number_of_nodes())
                # Add dummy type attributes if this is to be used with new modes
                # For example:
                # for node_id in G.nodes():
                #    G.nodes[node_id]['type'] = [1.0, 0.0, 0.0] # Dummy node type
                # for u,v in G.edges():
                #    G.edges[u,v]['type'] = [1.0, 0.0] # Dummy edge type
                retval.append(G)

        # self.max_node_count = max(self.max_node_count, current_max_nodes_for_this_load)
        return retval

    def load_citeseer_ego_dags(self, min_nodes=7, max_nodes=30):
        """Loads citeseer network from disk, and generates 3-hop
         weakly-connected ego-DAGs centered at random nodes.
         NOTE: These graphs will NOT have 'type' attributes for nodes/edges
         unless this function is modified.
        """
        # This self.max_node_count update should ideally happen in __init__
        # current_max_nodes_for_this_load = 0
        edges = []
        # Ensure this path is correct or make it a parameter
        fpath = "dataset/EGO/citeseer.cites"
        if not os.path.exists(fpath):
            print(f"Warning: Citeseer data file not found at {fpath}. Cannot load citeseer ego dags.")
            return []

        for line in open(fpath):
            parts = line.strip("\n").split("\t")
            if len(parts) == 2:  # Ensure valid edge format
                edges.append(parts)

        full_g = nx.DiGraph()
        full_g.add_edges_from(edges)

        # Get the largest weakly connected component
        largest_wcc_nodes = max(nx.weakly_connected_components(full_g), key=len, default=set())
        if not largest_wcc_nodes:
            print("Warning: Citeseer graph has no weakly connected components or is empty.")
            return []

        g_component = full_g.subgraph(largest_wcc_nodes).copy()  # Use .copy() for modifications
        g_component = nx.convert_node_labels_to_integers(g_component, first_label=0)

        g_component.remove_edges_from(list(nx.selfloop_edges(g_component)))  # list() for iter safety

        dags = []
        # Iterate over a sample of nodes if graph is very large, or all nodes
        node_list_for_ego = list(g_component.nodes())
        random.shuffle(node_list_for_ego)  # Process in random order

        for node_idx, node in enumerate(node_list_for_ego):
            if len(dags) >= 200:  # Limit number of ego graphs generated
                break
            # Create ego graph (undirected=True makes it include predecessors as well for radius)
            # Then, we check if this subgraph of the original *directed* graph is a DAG.
            # The ego_graph itself from networkx is undirected if undirected=True.
            # We need to be careful: nx.ego_graph with undirected=True on a DiGraph
            # considers reachability in an undirected sense.
            # A better approach for directed ego graphs on a DiGraph:
            # Get nodes within radius, then form subgraph from original DiGraph.

            nodes_in_radius = set(
                nx.single_source_shortest_path_length(g_component.to_undirected(), node, radius=3).keys())
            ego_subgraph_directed = g_component.subgraph(nodes_in_radius).copy()

            if nx.is_directed_acyclic_graph(ego_subgraph_directed):
                n_nodes = ego_subgraph_directed.number_of_nodes()
                if (n_nodes >= min_nodes and n_nodes <= max_nodes):
                    # Add dummy type attributes if this is to be used with new modes
                    # For example:
                    # for node_id_ego in ego_subgraph_directed.nodes():
                    #    ego_subgraph_directed.nodes[node_id_ego]['type'] = [1.0, 0.0, 0.0]
                    # for u_ego,v_ego in ego_subgraph_directed.edges():
                    #    ego_subgraph_directed.edges[u_ego,v_ego]['type'] = [1.0, 0.0]
                    dags.append(ego_subgraph_directed)
                    # current_max_nodes_for_this_load = max(current_max_nodes_for_this_load, n_nodes)

        # self.max_node_count = max(self.max_node_count, current_max_nodes_for_this_load)
        # np.random.shuffle(dags) # Already shuffled node_list_for_ego
        return dags  # Returns up to 200 dags

