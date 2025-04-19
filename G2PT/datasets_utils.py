# G2PT/datasets_utils.py

# --- Keep all original imports ---
from torch.utils.data import Dataset
import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data
from collections import deque
import numpy as np
import os
from torch_geometric.utils import to_networkx
from torch_geometric.utils.convert import from_networkx
import re
from functools import partial
import json
import networkx as nx


# --- Move pre_tokenize_function to top level ---
def pre_tokenize_function(examples, tokenizer, order_function, atom_type, bond_type):
    """
    Top-level function to process data examples into tokenized sequences.
    Now accepts tokenizer and order_function as arguments.
    """
    # Ensure 'examples' dict contains integer tensors 'x', 'edge_index', 'edge_attr'
    # Use the passed order_function
    data = order_function(examples, atom_type, bond_type)
    # Tokenize the generated text sequence
    # Assuming tokenizer is pre-configured (e.g., from AutoTokenizer)
    tokenized_data = tokenizer(data['text'], padding='max_length', truncation=True,
                               return_tensors='pt')  # Added truncation
    # Ensure tensors are correctly shaped (remove batch dim if tokenizer adds one)
    input_ids = tokenized_data['input_ids'].squeeze(0)
    attention_mask = tokenized_data['attention_mask'].squeeze(0)
    # Create labels (shifted input_ids)
    labels = input_ids.clone()
    # G2PT typically uses the input_ids directly as labels, handle potential shifts if needed
    # For standard causal LM, labels = input_ids, loss calculated internally on shifted logits/labels
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# --- Keep other top-level functions like to_seq_aig_topo, seq_to_nxgraph etc. ---
# --- (Ensure they are defined before get_datasets if called within) ---

def to_seq_aig_topo(data, atom_type, bond_type):
    """
    Converts AIG data (nodes, directed edges) to G2PT sequence format
    using topological ordering.

    Args:
        data (dict): {'x': tensor (N,), 'edge_index': tensor (2, E), 'edge_attr': tensor (E,)}
                     Assumes tensors contain INTEGER VOCABULARY IDs (e.g., 97-102).
        atom_type (list): List of node type token strings ['NODE_CONST0', ...]
        bond_type (list): List of edge type token strings ['EDGE_INV', ...]

    Returns:
        dict: {"text": [sequence_string]}
    """
    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]

    if num_nodes == 0:
        return {"text": ["<boc> <eoc> <bog> <eog>"]}  # Handle empty graph

    # 1. Build NetworkX DiGraph from input tensors
    G = nx.DiGraph()
    node_idx_map = {}  # Map internal 0..N-1 index to IDX_n token
    node_id_to_token_map = {}  # Map vocab ID (e.g., 97) to token ('NODE_CONST0')
    node_vocab_offset = 97  # ID of NODE_CONST0

    for node_idx in range(num_nodes):
        G.add_node(node_idx)  # Add nodes using 0..N-1 indices
        node_idx_map[node_idx] = f'IDX_{node_idx}'
        node_id_val = x_ids[node_idx].item()
        node_token_index = node_id_val - node_vocab_offset
        if 0 <= node_token_index < len(atom_type):
            node_id_to_token_map[node_idx] = atom_type[node_token_index]
        else:
            print(f"Warning: Node {node_idx} has unexpected ID {node_id_val}. Assigning UNK type.")
            node_id_to_token_map[node_idx] = "[UNK]"  # Or handle differently

    # Add edges to the DiGraph
    edge_id_to_token_map = {}  # Store edge type tokens associated with (u, v) pairs
    edge_vocab_offset = 101  # ID of EDGE_INV
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        src_node_idx = edge_index[0, i].item()
        dst_node_idx = edge_index[1, i].item()
        edge_id_val = edge_attr_ids[i].item()

        # Ensure nodes exist before adding edge (handles potential filtering issues)
        if src_node_idx in G and dst_node_idx in G:
            G.add_edge(src_node_idx, dst_node_idx)
            # Map edge ID to token string
            edge_token_index = edge_id_val - edge_vocab_offset
            if 0 <= edge_token_index < len(bond_type):
                edge_id_to_token_map[(src_node_idx, dst_node_idx)] = bond_type[edge_token_index]
            else:
                print(
                    f"Warning: Edge ({src_node_idx}->{dst_node_idx}) has unexpected ID {edge_id_val}. Assigning UNK type.")
                edge_id_to_token_map[(src_node_idx, dst_node_idx)] = "[UNK]"  # Or handle differently
        else:
            print(f"Warning: Skipping edge ({src_node_idx}->{dst_node_idx}) due to missing node index.")

    # 2. Perform Topological Sort
    try:
        # Use standard topological sort
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print(
            "Warning: Graph contains a cycle, cannot perform topological sort. Falling back to BFS order for sequence generation.")
        # Fallback: Use directed BFS - find a root (node with in-degree 0) or start at 0
        roots = [n for n, d in G.in_degree() if d == 0]
        start_node = roots[0] if roots else 0
        if start_node not in G:  # Handle case where start_node might not exist
            start_node = next(iter(G.nodes())) if G.nodes() else -1

        if start_node != -1:
            # Perform directed BFS traversal to get node order
            topo_order = list(nx.bfs_tree(G, source=start_node))
            # Add remaining nodes from other components if graph is not connected
            if len(topo_order) < G.number_of_nodes():
                remaining_nodes = list(set(G.nodes()) - set(topo_order))
                # Could perform BFS on remaining components, adding for simplicity
                topo_order.extend(remaining_nodes)
        else:
            topo_order = list(G.nodes())  # Fallback to arbitrary node order if graph is empty

    # 3. Build Node Context (<boc>...<eoc>) based on Topological Order
    ctx = ['<boc>']
    for node_idx in topo_order:
        node_token = node_id_to_token_map.get(node_idx, "[UNK]")
        node_idx_token = node_idx_map.get(node_idx, "IDX_?")  # Should always exist if node is in topo_order
        ctx.extend(['<sepc>', node_token, node_idx_token])
    ctx.append('<eoc>')

    # 4. Build Edge Sequence (<bog>...<eog>)
    # Option: Iterate through nodes in topological order, then process their outgoing edges
    outputs = ['<bog>']
    processed_edges = set()
    for u in topo_order:
        # Process outgoing edges for node u
        # G.successors(u) or G.out_edges(u)
        for v in sorted(list(G.successors(u))):  # Sort successors for determinism
            edge_tuple = (u, v)
            if edge_tuple in edge_id_to_token_map:
                edge_token = edge_id_to_token_map[edge_tuple]
                src_token_str = node_idx_map.get(u)
                dst_token_str = node_idx_map.get(v)
                if src_token_str and dst_token_str:  # Ensure nodes are mapped
                    outputs.extend(['<sepg>', src_token_str, dst_token_str, edge_token])
                    processed_edges.add(edge_tuple)
            else:
                # This might happen if edges were filtered earlier
                print(f"Warning: Edge ({u}->{v}) found during traversal but missing from edge_id_to_token_map.")

    outputs.append('<eog>')
    if len(outputs) == 2:  # Only contains <bog> and <eog>
        outputs = ['<bog>', '<eog>']  # Ensure it's not empty if no edges

    return {"text": [" ".join(ctx + outputs)]}


def seq_to_nxgraph(seq_str):
    """
    Converts a G2PT sequence string back into a NetworkX graph.
    Modified to handle AIG tokens and create a directed graph (nx.DiGraph).
    """
    tokens = seq_str.split()

    try:
        ctx_start = tokens.index('<boc>') + 1
        ctx_end = tokens.index('<eoc>')
        bog_start = tokens.index('<bog>') + 1
        eog_end = tokens.index('<eog>')
    except ValueError:
        print(f"Warning: Malformed sequence missing <boc>/<eoc> or <bog>/<eog>. Sequence: {seq_str[:100]}...")
        return nx.DiGraph()  # Return empty graph

    ctx_tokens = tokens[ctx_start:ctx_end]
    edge_tokens = [token for token in tokens[bog_start:eog_end] if token != '<sepg>']

    G = nx.DiGraph()  # Create a directed graph

    # --- Parse Nodes ---
    node_map = {}  # Map IDX_n token string back to integer node index 0..N-1
    node_data = {}  # Store node attributes {node_idx: {'type': 'NODE_TYPE_TOKEN'}}
    idx_pattern = re.compile(r'IDX_(\d+)')
    # More general pattern to capture different node type prefixes
    node_type_pattern = re.compile(r'(NODE_[A-Z0-9]+|ATOM_[A-Za-z]+|NODE)')  # Add other prefixes if needed

    current_node_idx_str = None
    current_node_type_str = None
    node_counter = 0  # Assign sequential indices 0, 1, 2...
    processed_idx_tokens = set()

    for token in ctx_tokens:
        if token == '<sepc>':
            # Reset for next node entry, handle potential missing parts
            current_node_type_str = None
            current_node_idx_str = None
            continue

        node_match = node_type_pattern.match(token)
        idx_match = idx_pattern.match(token)

        if node_match:
            current_node_type_str = node_match.group(0)
        elif idx_match:
            current_node_idx_str = idx_match.group(0)
            # Assign node index only when both type and IDX are found for a node entry
            if current_node_type_str and current_node_idx_str:
                if current_node_idx_str not in processed_idx_tokens:
                    node_index = node_counter  # Use sequential index 0, 1,...
                    node_map[current_node_idx_str] = node_index  # Map IDX_n -> node_index
                    node_data[node_index] = {'type': current_node_type_str}  # Store type attribute
                    processed_idx_tokens.add(current_node_idx_str)
                    node_counter += 1
                # Reset immediately after processing a complete node entry
                current_node_type_str = None
                current_node_idx_str = None

    # Add nodes to graph
    G.add_nodes_from([(idx, data) for idx, data in node_data.items()])

    # --- Parse Edges ---
    # More general pattern for edge types
    edge_type_pattern = re.compile(r'(EDGE_[A-Z]+|BOND_[A-Z]+|EDGE)')

    # Ensure edge_tokens list has a multiple of 3 elements
    if len(edge_tokens) % 3 != 0:
        print(f"Warning: Malformed edge sequence part. Length is not a multiple of 3. Tokens: {edge_tokens}")
        # Decide handling: return graph as is, or try parsing partial edges? Returning as is.
        return G

    for i in range(0, len(edge_tokens), 3):
        src_id_str = edge_tokens[i]
        dest_id_str = edge_tokens[i + 1]
        edge_type_str = edge_tokens[i + 2]

        # Use the map to get integer node indices
        src_idx = node_map.get(src_id_str)
        dest_idx = node_map.get(dest_id_str)

        # Extract edge type token
        edge_type_match = edge_type_pattern.match(edge_type_str)
        edge_type = edge_type_match.group(0) if edge_type_match else 'UNKNOWN_EDGE'  # Assign a default or raise error

        # Add directed edge if both nodes were found
        if src_idx is not None and dest_idx is not None:
            # Check if nodes actually exist in the graph (they should based on map)
            if src_idx in G and dest_idx in G:
                G.add_edge(src_idx, dest_idx, type=edge_type)
            else:
                # This case indicates an issue with node parsing or mapping
                print(
                    f"Warning: Node index {src_idx} or {dest_idx} not found in graph G when adding edge {src_id_str}->{dest_id_str}.")
        else:
            print(f"Warning: Could not map edge tokens to node indices: {src_id_str} or {dest_id_str}. Skipping edge.")

    return G


# --- Keep NumpyBinDataset class as corrected in the previous step ---
class NumpyBinDataset(Dataset):
    """
    Loads graph data preprocessed into numpy memmap files (.bin).
    Modified __getitem__ to return integer vocabulary IDs.
    CORRECTED VERSION using 'edge_indices' and 'edge_attrs' consistently.
    """

    def __init__(self, path, num_data, num_node_class, num_edge_class, shape, process_fn=lambda x: x):
        self.path = path
        self.num_data = num_data
        self.num_node_class = num_node_class
        self.num_edge_class = num_edge_class
        self.process_fn = process_fn  # This will be the pre_tokenize_function

        # Make a copy to avoid modifying the original dict passed in
        local_shape = shape.copy()

        # --- Ensure shapes are tuples using CONSISTENT keys from data_meta.json ---
        try:
            local_shape['xs'] = tuple(local_shape['xs'])
            local_shape['edge_indices'] = tuple(local_shape['edge_indices'])  # Use PLURAL
            local_shape['edge_attrs'] = tuple(local_shape['edge_attrs'])  # Use PLURAL
        except KeyError as e:
            raise KeyError(
                f"Error converting shapes to tuples. Missing key {e} in shape dictionary: {local_shape}. Check data_meta.json.")
        except Exception as e:
            raise RuntimeError(f"Error processing shape dictionary {local_shape}: {e}")

        # --- Load Memory Mapped Files using CONSISTENT keys ---
        try:
            xs_path = os.path.join(path, 'xs.bin')
            edge_indices_path = os.path.join(path, 'edge_indices.bin')  # Path corresponds to PLURAL key
            edge_attrs_path = os.path.join(path, 'edge_attrs.bin')  # Path corresponds to PLURAL key

            self.xs = np.memmap(xs_path, dtype=np.int16, mode='r', shape=local_shape['xs'])
            # --- Use PLURAL key for shape ---
            self.edge_indices = np.memmap(edge_indices_path, dtype=np.int16, mode='r',
                                          shape=local_shape['edge_indices'])
            # --- Use PLURAL key for shape ---
            self.edge_attrs = np.memmap(edge_attrs_path, dtype=np.int16, mode='r', shape=local_shape['edge_attrs'])

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Error opening memmap files in {path}. Did prepare_aig.py run correctly and save files to this location? Details: {e}")
        except Exception as e:
            raise RuntimeError(f"Error setting up memmap in {path} with shapes {local_shape}. Details: {e}")

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # Load raw int16 data from memmap files for the given index
        try:
            raw_x = np.array(self.xs[idx]).astype(np.int64)
            # --- Use self.edge_indices (PLURAL) ---
            raw_edge_index = np.array(self.edge_indices[idx]).astype(np.int64)
            # --- Use self.edge_attrs (PLURAL) ---
            raw_edge_attr = np.array(self.edge_attrs[idx]).astype(np.int64)
        except IndexError:
            print(f"Error: Index {idx} out of bounds for memmap arrays (num_data={self.num_data}).")
            # Return an empty dict for process_fn to handle gracefully
            empty_data = {'x': torch.tensor([], dtype=torch.long),
                          'edge_index': torch.tensor([[], []], dtype=torch.long),
                          'edge_attr': torch.tensor([], dtype=torch.long)}
            return self.process_fn(empty_data)  # Use self.process_fn
        except Exception as e:
            print(f"Error accessing memmap data at index {idx}: {e}")
            empty_data = {'x': torch.tensor([], dtype=torch.long),
                          'edge_index': torch.tensor([[], []], dtype=torch.long),
                          'edge_attr': torch.tensor([], dtype=torch.long)}
            return self.process_fn(empty_data)  # Use self.process_fn

        # Filter padding (-100) from node features
        node_padding_mask = raw_x != -100
        x_ids = torch.from_numpy(raw_x[node_padding_mask])
        num_valid_nodes = len(x_ids)

        if num_valid_nodes == 0:
            # Handle graphs with no valid nodes after filtering padding
            empty_data = {'x': x_ids,
                          'edge_index': torch.tensor([[], []], dtype=torch.long),
                          'edge_attr': torch.tensor([], dtype=torch.long)}
            return self.process_fn(empty_data)  # Use self.process_fn

        # Filter edge attributes based on padding
        edge_padding_mask = raw_edge_attr != -100
        edge_attr_ids_filtered_by_attr = torch.from_numpy(raw_edge_attr[edge_padding_mask])

        # Filter edge indices based *only* on the edge attribute padding mask initially
        # raw_edge_index has shape [2, max_num_edges]
        # edge_padding_mask has shape [max_num_edges]
        edge_index_filtered_by_attr = torch.from_numpy(raw_edge_index[:, edge_padding_mask])

        # Further filter edges where BOTH source and destination nodes are valid
        # Node indices should be between 0 and num_valid_nodes - 1
        if edge_index_filtered_by_attr.numel() > 0:  # Check if there are any edges left
            src_nodes = edge_index_filtered_by_attr[0, :]
            dst_nodes = edge_index_filtered_by_attr[1, :]
            node_indices_valid_mask = (src_nodes < num_valid_nodes) & (dst_nodes < num_valid_nodes) & \
                                      (src_nodes >= 0) & (dst_nodes >= 0)

            # Apply the node validity mask to get final edge indices and attributes
            edge_index_final = edge_index_filtered_by_attr[:, node_indices_valid_mask]
            edge_attr_final = edge_attr_ids_filtered_by_attr[node_indices_valid_mask]
        else:
            # No edges were valid based on attribute padding
            edge_index_final = torch.tensor([[], []], dtype=torch.long)
            edge_attr_final = torch.tensor([], dtype=torch.long)

        # Return dictionary containing tensors with INTEGER vocabulary IDs
        data_dict = {'x': x_ids, 'edge_index': edge_index_final, 'edge_attr': edge_attr_final}

        # Pass the dictionary with integer IDs to the processing function (pre_tokenize_function)
        return self.process_fn(data_dict)  # Use self.process_fn


# --- Keep randperm_node, remove_edge_with_attr, bfs_with_all_edges, to_seq_by_bfs, to_seq_by_deg ---
# ... (these functions remain unchanged) ...
def randperm_node(x, edge_index):
    num_nodes = x.shape[0]

    perm = torch.randperm(num_nodes)

    # Create a mapping from old node indices to new node indices
    mapping = torch.empty_like(perm)
    mapping[perm] = torch.arange(num_nodes)

    # Permute node features
    new_x = x[perm]
    # Update edge indices using the mapping
    new_edge_index = mapping[edge_index]

    return new_x, new_edge_index


def remove_edge_with_attr(graph, edge_to_remove):
    """
    Remove an edge and its attributes from a PyTorch Geometric graph.

    Args:
        graph (torch_geometric.data.Data): Input graph.
        edge_to_remove (tuple): Edge to remove, specified as (source, target).

    Returns:
        torch_geometric.data.Data: Graph with the specified edge and its attributes removed.
    """
    new_graph = graph.clone()
    edge_index = new_graph.edge_index
    edge_attr = new_graph.edge_attr

    # Find edges to keep
    mask1 = ~((edge_index[0] == edge_to_remove[0]) & (edge_index[1] == edge_to_remove[1]))
    mask2 = ~((edge_index[1] == edge_to_remove[0]) & (edge_index[0] == edge_to_remove[1]))
    mask = mask1.logical_and(mask2)
    # Apply the mask to edge_index and edge_attr
    new_edge_index = edge_index[:, mask]
    if edge_attr is not None:
        new_edge_attr = edge_attr[mask]
    else:
        new_edge_attr = None
    if len(edge_attr.shape) == 2:  # one hot
        poped_edge_attr = edge_attr[~mask1].argmax().item()
    else:
        poped_edge_attr = edge_attr[~mask1].item()
    # Update the graph
    new_graph.edge_index = new_edge_index
    new_graph.edge_attr = new_edge_attr
    return new_graph, poped_edge_attr


def bfs_with_all_edges(G, source):
    visited = set()
    edges = set()
    edges_bfs = []

    queue = deque([source])
    visited.add(source)

    while queue:
        node = queue.popleft()
        for neighbor in G[node]:
            if neighbor not in visited:
                edges.add(tuple(sorted((node, neighbor))))
                edges_bfs.append((node, neighbor))

                visited.add(neighbor)
                queue.append(neighbor)
            else:
                if tuple(sorted((neighbor, node))) not in edges:
                    edges.add(tuple(sorted((neighbor, node))))
                    edges_bfs.append((node, neighbor))

    return edges_bfs


def to_seq_by_bfs(data, atom_type, bond_type):
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)
    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}'] for node_idx, node_type in
           enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []

    G = to_networkx(data)

    # get edge order from dfs,begin from node 0, G is nx graph
    # _,edges_order_dfs = dfs_with_all_edges(G,0)
    edges_order_bfs = bfs_with_all_edges(G, 0)
    for selected_source_node_idx, selected_dest_node_idx in edges_order_bfs:
        # get_edge_attr
        edge_mask = ((data.edge_index[0] == selected_source_node_idx) & (
                    data.edge_index[1] == selected_dest_node_idx)) | \
                    ((data.edge_index[0] == selected_dest_node_idx) & (data.edge_index[1] == selected_source_node_idx))
        edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        if len(edge_indices) > 0:
            removed_edge_type = data.edge_attr[edge_indices][0].argmax().item()
        outputs.append(['<sepg>', f'IDX_{selected_source_node_idx}', f'IDX_{selected_dest_node_idx}',
                        bond_type[removed_edge_type - 1]])

    ctx[0] = '<boc>'
    ctx.append('<eoc>')
    outputs = sum(outputs, [])
    outputs[0] = '<bog>'
    outputs.append('<eog>')

    return {"text": [" ".join(ctx + outputs)]}


def to_seq_by_deg(data, atom_type, bond_type):
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)
    num_nodes = x.shape[0]

    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}'] for node_idx, node_type in
           enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])
    data_t = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []
    INF = 100
    while True:
        source_nodes_t = data_t.edge_index[0]
        node_degrees_t = degree(source_nodes_t, num_nodes=num_nodes)
        if torch.all(node_degrees_t == 0):
            break
        node_degrees_t[node_degrees_t == 0] = INF
        # sample a source node with minimum deg
        candidate_source_nodes = torch.where(node_degrees_t == node_degrees_t.min())[0]
        selected_index = torch.randint(0, candidate_source_nodes.shape[0], (1,)).item()
        selected_source_node_idx = candidate_source_nodes[selected_index].item()

        # get the dest node with minimum deg
        source_node_mask = source_nodes_t == selected_source_node_idx
        candidate_dest_nodes = data_t.edge_index[1][source_node_mask].unique()

        candidate_dest_degrees = node_degrees_t[candidate_dest_nodes]
        min_dest_degree = candidate_dest_degrees.min()

        indices = torch.where(candidate_dest_degrees == min_dest_degree)[0]
        selected_index = indices[torch.randint(0, len(indices), (1,)).item()]
        selected_dest_node_idx = candidate_dest_nodes[selected_index].item()

        # get new graph at t-1
        data_tminus1, removed_edge_type = remove_edge_with_attr(data_t,
                                                                (selected_source_node_idx, selected_dest_node_idx))
        # selected_source_node_type = data.x[selected_source_node_idx].argmax(-1).item()
        # selected_dest_node_type = data.x[selected_dest_node_idx].argmax(-1).item()
        outputs.append(['<sepg>', f'IDX_{selected_source_node_idx}', f'IDX_{selected_dest_node_idx}',
                        bond_type[removed_edge_type - 1]])
        data_t = data_tminus1

    ctx[0] = '<boc>'
    ctx.append('<eoc>')
    outputs = outputs[::-1]
    outputs = sum(outputs, [])
    outputs[0] = '<bog>'
    outputs.append('<eog>')
    return {"text": [" ".join(ctx + outputs)]}


# --- Modified get_datasets function ---
def get_datasets(dataset_name, tokenizer, order='bfs'):
    # Select the sequence generation function based on dataset and order
    if dataset_name == 'aig' or order == 'topo':  # Use topo sort for AIG regardless of 'order' arg for now
        print(f"Using topological sequence generation logic for AIG dataset.")
        order_function = to_seq_aig_topo
        if dataset_name != 'aig':
            print(f"Warning: Using topological sort for non-AIG dataset '{dataset_name}' based on --ordering=topo.")
    elif order == 'bfs':
        order_function = to_seq_by_bfs
    elif order == 'deg':
        order_function = to_seq_by_deg
    else:
        raise NotImplementedError(f"Order function {order} is not implemented")

    # --- Dataset Specific Setups ---
    train_datasets = None
    eval_datasets = None
    ATOM_TYPE = None
    BOND_TYPE = None
    train_shape = None
    eval_shape = None
    data_dir = f'./datasets/{dataset_name}'  # Default data dir

    # Define ATOM_TYPE, BOND_TYPE, shapes, and data_dir for each dataset
    if dataset_name == 'aig':
        ATOM_TYPE = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
        BOND_TYPE = ['EDGE_INV', 'EDGE_REG']
        meta_path = os.path.join(data_dir, 'data_meta.json')
        try:
            with open(meta_path, 'r') as f:
                data_meta = json.load(f)
            train_shape = {k: tuple(v) for k, v in data_meta['train_shape'].items()}
            eval_shape = {k: tuple(v) for k, v in data_meta['eval_shape'].items()}
        except Exception as e:
            raise RuntimeError(f"Failed to load AIG meta {meta_path}: {e}")

    elif dataset_name == 'qm9':
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_O', 'ATOM_F']
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC']
        train_shape = {'xs': (97732, 9), 'edge_indices': (97732, 2, 28), 'edge_attrs': (97732, 28)}
        eval_shape = {'xs': (20042, 9), 'edge_indices': (20042, 2, 26), 'edge_attrs': (20042, 26)}

    elif dataset_name == 'tree':
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']
        train_shape = {'xs': (256, 64), 'edge_indices': (256, 2, 126), 'edge_attrs': (256, 126)}
        eval_shape = {'xs': (64, 64), 'edge_indices': (64, 2, 126), 'edge_attrs': (64, 126)}

    else:
        raise NotImplementedError(f"Dataset {dataset_name} setup is not implemented.")

    # --- Instantiate Datasets ---
    if ATOM_TYPE and BOND_TYPE and train_shape and eval_shape:
        num_train = train_shape['xs'][0]
        num_eval = eval_shape['xs'][0]
        num_node_classes = len(ATOM_TYPE)
        # +1 for padding/no-edge type implicitly handled by vocab size usually
        num_edge_classes = len(BOND_TYPE) + 1

        # Define the processing function using partial, binding the top-level pre_tokenize_function
        # Pass tokenizer and order_function explicitly here
        process_fn = partial(pre_tokenize_function,
                             tokenizer=tokenizer,
                             order_function=order_function,
                             atom_type=ATOM_TYPE,
                             bond_type=BOND_TYPE)

        train_datasets = NumpyBinDataset(os.path.join(data_dir, 'train'),
                                         num_train, num_node_classes, num_edge_classes,
                                         shape=train_shape,
                                         process_fn=process_fn)  # Pass the partial function
        eval_datasets = NumpyBinDataset(os.path.join(data_dir, 'eval'),
                                        num_eval, num_node_classes, num_edge_classes,
                                        shape=eval_shape,
                                        process_fn=process_fn)  # Pass the partial function
    else:
        raise RuntimeError(f"Missing configuration (ATOM_TYPE, BOND_TYPE, shapes) for dataset {dataset_name}")

    # Final check
    if train_datasets is None or eval_datasets is None:
        raise RuntimeError(f"Failed to initialize datasets for {dataset_name}")

    return train_datasets, eval_datasets