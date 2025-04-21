# G2PT/datasets_utils.py

# --- Keep all original imports ---
from torch.utils.data import Dataset
import torch
# from torch_geometric.utils import degree # We'll use networkx degree
from torch_geometric.data import Data
from collections import deque
import numpy as np
import os
# from torch_geometric.utils import to_networkx # Use nx directly
# from torch_geometric.utils.convert import from_networkx
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
    # The order_function now directly takes the dictionary with integer tensors
    data_dict = {'x': examples['x'], 'edge_index': examples['edge_index'], 'edge_attr': examples['edge_attr']}
    sequence_data = order_function(data_dict, atom_type, bond_type) # Pass atom/bond types

    # Tokenize the generated text sequence
    # Assuming tokenizer is pre-configured (e.g., from AutoTokenizer)
    tokenized_data = tokenizer(sequence_data['text'], padding='max_length', truncation=True,
                               return_tensors='pt')  # Added truncation

    # Ensure tensors are correctly shaped (remove batch dim if tokenizer adds one)
    input_ids = tokenized_data['input_ids'].squeeze(0) if tokenized_data['input_ids'].ndim > 1 else tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask'].squeeze(0) if tokenized_data['attention_mask'].ndim > 1 else tokenized_data['attention_mask']

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
    if 'x' not in data or 'edge_index' not in data or 'edge_attr' not in data:
        print("Warning (Topo): Input data dictionary missing required keys ('x', 'edge_index', 'edge_attr').")
        return {"text": ["<boc> <eoc> <bog> <eog>"]}

    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]

    if num_nodes == 0:
        return {"text": ["<boc> <eoc> <bog> <eog>"]}  # Handle empty graph

    # 1. Build NetworkX DiGraph from input tensors
    G = nx.DiGraph()
    node_idx_map = {}  # Map internal 0..N-1 index to IDX_n token
    node_id_to_token_map = {}  # Map node index (0..N-1) to token ('NODE_CONST0')
    node_vocab_offset = 97  # ID of NODE_CONST0

    for node_idx in range(num_nodes):
        G.add_node(node_idx)  # Add nodes using 0..N-1 indices
        node_idx_map[node_idx] = f'IDX_{node_idx}'
        node_id_val = x_ids[node_idx].item()
        node_token_index = node_id_val - node_vocab_offset
        if 0 <= node_token_index < len(atom_type):
            node_id_to_token_map[node_idx] = atom_type[node_token_index]
        else:
            print(f"Warning (Topo Node Type): Node {node_idx} has unexpected ID {node_id_val}. Assigning UNK type.")
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
                    f"Warning (Topo Edge Type): Edge ({src_node_idx}->{dst_node_idx}) has unexpected ID {edge_id_val}. Assigning UNK type.")
                edge_id_to_token_map[(src_node_idx, dst_node_idx)] = "[UNK]"  # Or handle differently
        else:
            print(f"Warning (Topo Edge Add): Skipping edge ({src_node_idx}->{dst_node_idx}) due to missing node index.")

    # 2. Perform Topological Sort
    try:
        # Use standard topological sort
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print(
            "Warning (Topo Sort): Graph contains a cycle, cannot perform topological sort. Falling back to BFS order for sequence generation.")
        # Fallback: Use directed BFS - find a root (node with in-degree 0) or start at 0
        roots = [n for n, d in G.in_degree() if d == 0]
        start_node = roots[0] if roots else 0
        if start_node not in G:  # Handle case where start_node might not exist
            start_node = next(iter(G.nodes())) if G.nodes() else -1

        if start_node != -1:
            # Perform directed BFS traversal to get node order
            bfs_nodes_order = list(nx.bfs_tree(G, source=start_node))
            # Add remaining nodes from other components if graph is not connected
            if len(bfs_nodes_order) < G.number_of_nodes():
                remaining_nodes = list(set(G.nodes()) - set(bfs_nodes_order))
                # Could perform BFS on remaining components, adding for simplicity
                bfs_nodes_order.extend(remaining_nodes)
            topo_order = bfs_nodes_order # Use BFS order as fallback
        else:
            topo_order = list(G.nodes())  # Fallback to arbitrary node order if graph is empty/malformed

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
                print(f"Warning (Topo Edge Seq): Edge ({u}->{v}) found during traversal but missing from edge_id_to_token_map.")

    outputs.append('<eog>')
    if len(outputs) == 2:  # Only contains <bog> and <eog>
        outputs = ['<bog>', '<eog>']  # Ensure it's not empty if no edges

    return {"text": [" ".join(ctx + outputs)]}


def seq_to_nxgraph(seq_str, parsing_mode='strict'):
    """
    Converts a G2PT sequence string back into a NetworkX graph.
    Allows different parsing modes for the edge sequence.

    Args:
        seq_str (str): The sequence string generated by the model.
        parsing_mode (str): 'strict' or 'robust'.
                           'strict': Requires edge tokens to be perfect triplets.
                           'robust': Tries to find valid triplets, skipping malformed parts.

    Returns:
        nx.DiGraph: The resulting graph.
    """
    tokens = seq_str.split()

    try:
        ctx_start = tokens.index('<boc>') + 1
        ctx_end = tokens.index('<eoc>')
        bog_start = tokens.index('<bog>') + 1
        eog_end = tokens.index('<eog>')
    except ValueError:
        print(f"Warning (Seq Parse): Malformed sequence missing <boc>/<eoc> or <bog>/<eog>. Seq: {seq_str[:100]}...")
        return nx.DiGraph()

    ctx_tokens = tokens[ctx_start:ctx_end]
    edge_tokens = [token for token in tokens[bog_start:eog_end] if token != '<sepg>']

    G = nx.DiGraph()
    node_map = {}
    node_data = {}
    idx_pattern = re.compile(r'IDX_(\d+)')
    node_type_pattern = re.compile(r'(NODE_[A-Z0-9]+|ATOM_[A-Za-z]+|NODE)')
    edge_type_pattern = re.compile(r'^(EDGE_[A-Z]+|BOND_[A-Z]+|EDGE)$') # Anchor regex
    node_idx_pattern = re.compile(r'^IDX_\d+$') # Anchor regex for IDX tokens

    # --- Parse Nodes (remains the same) ---
    current_node_idx_str = None
    current_node_type_str = None
    node_counter = 0
    processed_idx_tokens = set()
    for token in ctx_tokens:
        if token == '<sepc>': current_node_type_str = None; current_node_idx_str = None; continue
        node_match = node_type_pattern.match(token)
        idx_match = idx_pattern.match(token)
        if node_match: current_node_type_str = node_match.group(0)
        elif idx_match: current_node_idx_str = idx_match.group(0)
        if current_node_type_str and current_node_idx_str:
            if current_node_idx_str not in processed_idx_tokens:
                # Use the index from the token itself (IDX_*) if possible
                node_index_from_token = int(idx_match.group(1))
                # If multiple nodes map to the same IDX_*, this will overwrite, which
                # implies the sequence generation might be flawed. Robust parsing accepts this.
                node_map[current_node_idx_str] = node_index_from_token
                node_data[node_index_from_token] = {'type': current_node_type_str}
                processed_idx_tokens.add(current_node_idx_str)
                # Keep track of max node index seen for adding nodes later
                node_counter = max(node_counter, node_index_from_token + 1)
            current_node_type_str = None; current_node_idx_str = None
    # Add all nodes up to the maximum index encountered
    G.add_nodes_from([(idx, node_data.get(idx, {'type': 'UNKNOWN'})) for idx in range(node_counter)])
    # Update attributes for nodes that were actually defined
    nx.set_node_attributes(G, {idx: data for idx, data in node_data.items()})
    # --- End Node Parsing ---

    # --- Parse Edges (Conditional Logic) ---
    if parsing_mode == 'strict':
        # --- Original Strict Logic ---
        if len(edge_tokens) % 3 != 0:
            if edge_tokens: # Only warn if there were actually edge tokens
                 print(f"Warning (Strict Parse): Malformed edge sequence. Length ({len(edge_tokens)}) not multiple of 3. Discarding all edges.")
            # Return graph with only nodes if edge part is malformed in strict mode
            return G
        # Proceed with strict 3-step iteration if length is okay
        for i in range(0, len(edge_tokens), 3):
            src_id_str = edge_tokens[i]
            dest_id_str = edge_tokens[i + 1]
            edge_type_str = edge_tokens[i + 2]
            src_idx = node_map.get(src_id_str)
            dest_idx = node_map.get(dest_id_str)
            edge_type_match = edge_type_pattern.match(edge_type_str)
            edge_type = edge_type_match.group(0) if edge_type_match else 'UNKNOWN_EDGE'
            if src_idx is not None and dest_idx is not None:
                if src_idx in G and dest_idx in G: G.add_edge(src_idx, dest_idx, type=edge_type)
                else: print(f"Warning (Strict Parse): Node index {src_idx} or {dest_idx} invalid or missing from node context.")
            else: print(f"Warning (Strict Parse): Could not map edge tokens {src_id_str} or {dest_id_str} to node indices. Skipping.")
        # --- End Strict Logic ---

    elif parsing_mode == 'robust':
        # --- Proposed Robust Logic ---
        idx = 0
        while idx < len(edge_tokens):
            is_potential_triplet = False
            if idx + 2 < len(edge_tokens):
                src_candidate = edge_tokens[idx]
                dst_candidate = edge_tokens[idx+1]
                type_candidate = edge_tokens[idx+2]
                # Check if candidates look like IDX_*, IDX_*, EDGE_*
                if node_idx_pattern.match(src_candidate) and \
                   node_idx_pattern.match(dst_candidate) and \
                   edge_type_pattern.match(type_candidate):
                    is_potential_triplet = True

            if is_potential_triplet:
                src_id_str = edge_tokens[idx]
                dest_id_str = edge_tokens[idx+1]
                edge_type_str = edge_tokens[idx+2]
                src_idx = node_map.get(src_id_str)
                dest_idx = node_map.get(dest_id_str)
                edge_type = edge_type_str # Already matched by regex

                if src_idx is not None and dest_idx is not None:
                    # Ensure nodes actually exist in the graph before adding edge
                    if G.has_node(src_idx) and G.has_node(dest_idx):
                        G.add_edge(src_idx, dest_idx, type=edge_type)
                    else:
                         print(f"Warning (Robust Parse): Node index {src_idx} or {dest_idx} (from {src_id_str}/{dest_id_str}) not found in graph nodes despite being parsed. Skipping edge.")
                else:
                    print(f"Warning (Robust Parse): Could not map edge tokens {src_id_str} or {dest_id_str} to node indices. Skipping edge.")
                # Advance past the processed triplet
                idx += 3
            else:
                # Token at idx is not the start of a valid triplet, skip it
                # Optional: print(f"Skipping unexpected token in edge sequence: {edge_tokens[idx]}")
                idx += 1
        # --- End Robust Logic ---

    else:
        print(f"Error: Unknown parsing_mode '{parsing_mode}'. Using strict parsing.")
        # Optionally fall back to strict or raise an error
        # Fallback to strict:
        if len(edge_tokens) % 3 != 0: return G
        for i in range(0, len(edge_tokens), 3): # Simplified strict loop for fallback
             src_id_str = edge_tokens[i]; dest_id_str = edge_tokens[i+1]; edge_type_str = edge_tokens[i+2]
             src_idx = node_map.get(src_id_str); dest_idx = node_map.get(dest_id_str)
             edge_type = edge_type_str if edge_type_pattern.match(edge_type_str) else 'UNKNOWN_EDGE'
             if src_idx is not None and dest_idx is not None and G.has_node(src_idx) and G.has_node(dest_idx):
                 G.add_edge(src_idx, dest_idx, type=edge_type)

    return G

# --- Keep NumpyBinDataset class as corrected ---
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
            # Ensure keys exist before accessing them
            required_keys = ['xs', 'edge_indices', 'edge_attrs']
            for key in required_keys:
                if key not in local_shape:
                     raise KeyError(f"Missing key '{key}' in shape dictionary")
            local_shape['xs'] = tuple(local_shape['xs'])
            local_shape['edge_indices'] = tuple(local_shape['edge_indices'])  # Use PLURAL
            local_shape['edge_attrs'] = tuple(local_shape['edge_attrs'])  # Use PLURAL
        except KeyError as e:
            raise KeyError(
                f"Error converting shapes to tuples. {e} in shape dictionary: {local_shape}. Check data_meta.json.")
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
            # Slice the first dimension (batch)
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
            # Process the empty data using the bound pre_tokenize_function
            return self.process_fn(empty_data)
        except Exception as e:
            print(f"Error accessing memmap data at index {idx}: {e}")
            empty_data = {'x': torch.tensor([], dtype=torch.long),
                          'edge_index': torch.tensor([[], []], dtype=torch.long),
                          'edge_attr': torch.tensor([], dtype=torch.long)}
            # Process the empty data using the bound pre_tokenize_function
            return self.process_fn(empty_data)

        # Filter padding (-100) from node features
        node_padding_mask = raw_x != -100
        x_ids = torch.from_numpy(raw_x[node_padding_mask])
        num_valid_nodes = len(x_ids)

        # Create a mapping from old (padded) index to new (unpadded) index
        old_indices = np.arange(len(raw_x))
        new_indices_map = -np.ones_like(old_indices) # Initialize with -1
        new_indices_map[node_padding_mask] = np.arange(num_valid_nodes)

        if num_valid_nodes == 0:
            # Handle graphs with no valid nodes after filtering padding
            empty_data = {'x': x_ids,
                          'edge_index': torch.tensor([[], []], dtype=torch.long),
                          'edge_attr': torch.tensor([], dtype=torch.long)}
            # Process the empty data using the bound pre_tokenize_function
            return self.process_fn(empty_data)

        # Filter edge attributes based on padding (-100)
        # Ensure raw_edge_attr is 1D before applying mask
        if raw_edge_attr.ndim > 1:
             print(f"Warning: edge_attr has unexpected shape {raw_edge_attr.shape} at index {idx}. Flattening.")
             raw_edge_attr = raw_edge_attr.flatten()

        edge_padding_mask = raw_edge_attr != -100
        edge_attr_ids_filtered_by_attr = torch.from_numpy(raw_edge_attr[edge_padding_mask])

        # Filter edge indices based *only* on the edge attribute padding mask initially
        # raw_edge_index has shape [2, max_num_edges]
        # edge_padding_mask has shape [max_num_edges]
        # Ensure shapes match before filtering
        if edge_padding_mask.shape[0] != raw_edge_index.shape[1]:
             print(f"Warning: Mismatch between edge_attr padding mask ({edge_padding_mask.shape[0]}) and edge_index columns ({raw_edge_index.shape[1]}) at index {idx}. Skipping edge processing.")
             edge_index_filtered_by_attr = torch.tensor([[], []], dtype=torch.long)
             edge_attr_ids_filtered_by_attr = torch.tensor([], dtype=torch.long) # Ensure consistency
        else:
             edge_index_filtered_by_attr = torch.from_numpy(raw_edge_index[:, edge_padding_mask])


        # Remap edge indices to the new node indices (0 to num_valid_nodes - 1)
        # and filter edges connecting to padded nodes
        if edge_index_filtered_by_attr.numel() > 0: # Check if there are any edges left
             src_nodes_old = edge_index_filtered_by_attr[0, :].numpy()
             dst_nodes_old = edge_index_filtered_by_attr[1, :].numpy()

             # Map old indices to new indices
             src_nodes_new = new_indices_map[src_nodes_old]
             dst_nodes_new = new_indices_map[dst_nodes_old]

             # Create a mask for valid edges (where both src and dst map to non -1 indices)
             valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)

             # Apply the mask to get final edge indices and attributes
             edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask],
                                              dst_nodes_new[valid_edge_mask]], dtype=torch.long)
             edge_attr_final = edge_attr_ids_filtered_by_attr[valid_edge_mask]
        else:
             # No edges were valid based on attribute padding
             edge_index_final = torch.tensor([[], []], dtype=torch.long)
             edge_attr_final = torch.tensor([], dtype=torch.long)


        # Return dictionary containing tensors with INTEGER vocabulary IDs
        data_dict = {'x': x_ids, 'edge_index': edge_index_final, 'edge_attr': edge_attr_final}

        # Pass the dictionary with integer IDs to the processing function (pre_tokenize_function)
        return self.process_fn(data_dict)


# --- REMOVE unused randperm_node and remove_edge_with_attr ---
# def randperm_node(x, edge_index): ...
# def remove_edge_with_attr(graph, edge_to_remove): ...


# --- REFINED bfs_edge_order ---
# Renamed for clarity and simplified logic for directed graphs
def get_bfs_edge_order(G):
    """
    Performs BFS starting from all nodes with in-degree 0
    and returns a list of directed edges in the order they are traversed.
    Handles disconnected graphs by restarting BFS from unvisited nodes.

    Args:
        G (nx.DiGraph): Input directed graph.

    Returns:
        list: List of edge tuples (u, v) in BFS traversal order.
    """
    if not G or G.number_of_edges() == 0:
        return []

    edges_bfs = []
    visited_nodes = set()
    nodes_to_process = list(G.nodes()) # Keep track of nodes not yet started from
    processed_edges = set() # Keep track of edges already added

    while nodes_to_process:
        # Find a starting node for the next BFS component
        start_node = -1
        # Prioritize nodes with in-degree 0 among the remaining unvisited ones
        possible_starts = sorted([n for n in nodes_to_process if G.in_degree(n) == 0 and n not in visited_nodes])
        if possible_starts:
            start_node = possible_starts[0] # Pick lowest index source node
        else:
            # If no nodes with in-degree 0 left (e.g., cycles or only visited nodes remain)
            # just pick the first unvisited node from the remaining list (sorted for determinism)
            unvisited_remaining = sorted([n for n in nodes_to_process if n not in visited_nodes])
            if unvisited_remaining:
                 start_node = unvisited_remaining[0]
                 # This might indicate a cycle or unusual component starting point
                 # print(f"Warning (BFS Edge Order): No source node found. Starting BFS from node {start_node}")

        if start_node == -1: # All remaining nodes must have been visited already
            break

        # Start BFS from the selected start_node
        queue = deque([start_node])
        visited_nodes.add(start_node)
        nodes_to_process.remove(start_node) # Mark as processed for starting purposes

        component_edges = []
        while queue:
            u = queue.popleft()
            # Process neighbors in a deterministic order (sorted by node index)
            for v in sorted(list(G.successors(u))):
                edge = (u, v)
                # Record the directed edge only if not already processed
                if edge not in processed_edges:
                    component_edges.append(edge)
                    processed_edges.add(edge)

                if v not in visited_nodes:
                    visited_nodes.add(v)
                    if v in nodes_to_process: # Only remove if it was pending start
                         nodes_to_process.remove(v)
                    queue.append(v)

        edges_bfs.extend(component_edges) # Add edges from this component

    # Sanity check (optional): Ensure all edges were captured
    if len(processed_edges) != G.number_of_edges():
       print(f"Warning (BFS Edge Order): Number of traversed edges ({len(processed_edges)}) doesn't match graph edges ({G.number_of_edges()}). Possible disconnected graph or issue.")

    return edges_bfs


# --- REFINED to_seq_by_bfs ---
def to_seq_by_bfs(data, atom_type, bond_type):
    """
    Converts AIG data (nodes, directed edges) to G2PT sequence format
    using BFS ordering for edges. Nodes listed in index order.
    REFINED to use get_bfs_edge_order and corrected data handling.

    Args:
        data (dict): {'x': tensor (N,), 'edge_index': tensor (2, E), 'edge_attr': tensor (E,)}
                     Assumes tensors contain INTEGER VOCABULARY IDs (e.g., 97-102).
        atom_type (list): List of node type token strings ['NODE_CONST0', ...] (Indices 0-3 map to IDs 97-100)
        bond_type (list): List of edge type token strings ['EDGE_INV', 'EDGE_REG'] (Indices 0-1 map to IDs 101-102)

    Returns:
        dict: {"text": [sequence_string]}
    """
    if 'x' not in data or 'edge_index' not in data or 'edge_attr' not in data:
        print("Warning (BFS): Input data dictionary missing required keys ('x', 'edge_index', 'edge_attr').")
        return {"text": ["<boc> <eoc> <bog> <eog>"]}

    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]

    if num_nodes == 0:
        return {"text": ["<boc> <eoc> <bog> <eog>"]}

    # --- 1. Build Node Context (<boc>...<eoc>) ---
    # Node context lists nodes in their original 0..N-1 index order
    ctx = ['<boc>']
    node_indices_map = {} # Map node index (0..N-1) to IDX_n token string
    node_id_to_type_token = {} # Map node index to its type token string (for building G)
    node_vocab_offset = 97

    for node_idx in range(num_nodes):
        idx_token_str = f'IDX_{node_idx}'
        node_indices_map[node_idx] = idx_token_str

        node_vocab_id = x_ids[node_idx].item()
        node_token_index = node_vocab_id - node_vocab_offset
        if 0 <= node_token_index < len(atom_type):
            node_type_str = atom_type[node_token_index]
        else:
            print(f"Warning (BFS Node Ctx): Node {node_idx} unexpected ID {node_vocab_id}. UNK type.")
            node_type_str = "[UNK]"

        node_id_to_type_token[node_idx] = node_type_str
        ctx.extend(['<sepc>', node_type_str, idx_token_str])
    ctx.append('<eoc>')

    # --- 2. Build NetworkX DiGraph (for BFS traversal) ---
    G = nx.DiGraph()
    edge_data_map = {} # Store edge attributes keyed by (u,v) tuple for easy lookup
    edge_vocab_offset = 101

    # Add nodes with their type string as an attribute
    for node_idx in range(num_nodes):
        G.add_node(node_idx, type=node_id_to_type_token.get(node_idx, "[UNK]"))

    num_edges = edge_index.shape[1]
    if num_edges != edge_attr_ids.shape[0]:
         print(f"Warning (BFS Graph Build): Mismatch between edge_index count ({num_edges}) and edge_attr count ({edge_attr_ids.shape[0]}).")
         # Decide how to handle: proceed cautiously or return error? Let's proceed.

    for i in range(num_edges):
        src_node_idx = edge_index[0, i].item()
        dst_node_idx = edge_index[1, i].item()

        # Ensure indices are within the valid range of nodes added
        if src_node_idx >= 0 and src_node_idx < num_nodes and \
           dst_node_idx >= 0 and dst_node_idx < num_nodes:
            # Add edge to graph
            G.add_edge(src_node_idx, dst_node_idx)
            # Store edge type string in map
            edge_vocab_id = edge_attr_ids[i].item()
            edge_token_index = edge_vocab_id - edge_vocab_offset
            if 0 <= edge_token_index < len(bond_type):
                bond_type_str = bond_type[edge_token_index]
            else:
                print(f"Warning (BFS Edge Attr): Edge ({src_node_idx}->{dst_node_idx}) unexpected ID {edge_vocab_id}. UNK type.")
                bond_type_str = "[UNK]"
            # Store bond type in the map (overwrites if multiple edges exist, assumes simple graph for seq)
            edge_data_map[(src_node_idx, dst_node_idx)] = bond_type_str
        else:
            print(f"Warning (BFS Graph Build): Skipping edge ({src_node_idx}->{dst_node_idx}) due to invalid node index (max is {num_nodes-1}).")


    # --- 3. Perform BFS and Build Edge Sequence (<bog>...<eog>) ---
    outputs = ['<bog>']
    # Get edge order using the refined BFS function
    edges_order_bfs = get_bfs_edge_order(G)

    for src_idx, dest_idx in edges_order_bfs:
        src_token_str = node_indices_map.get(src_idx)
        dest_token_str = node_indices_map.get(dest_idx)
        # Retrieve edge type from the map built earlier
        bond_type_str = edge_data_map.get((src_idx, dest_idx))

        # Check if all components were found
        if src_token_str and dest_token_str and bond_type_str:
            outputs.extend(['<sepg>', src_token_str, dest_token_str, bond_type_str])
        else:
            # This indicates an edge from BFS wasn't properly recorded in edge_data_map or node_indices_map
            print(f"Warning (BFS Edge Seq): Missing data for edge ({src_idx}->{dest_idx}) from BFS order.")
            # Optional: Add more debugging if needed

    outputs.append('<eog>')
    if len(outputs) == 2: # Only <bog> and <eog>
        outputs = ['<bog>', '<eog>']

    # --- 4. Combine and Return ---
    return {"text": [" ".join(ctx + outputs)]}


# --- REWRITTEN to_seq_by_deg ---
def to_seq_by_deg(data, atom_type, bond_type):
    """
    Converts AIG data (nodes, directed edges) to G2PT sequence format
    using Degree-based ordering for edges. Nodes listed in index order.

    Edge Selection Strategy:
    1. Select source node `u` with the minimum *non-zero* out-degree. Break ties using the lowest node index.
    2. Among `u`'s neighbors `v`, select the destination node `v` with the minimum out-degree (can be zero). Break ties using the lowest node index.
    3. Add the edge `(u, v)` to the sequence.
    4. Remove the edge `(u, v)` from the graph.
    5. Repeat until no edges remain.

    Args:
        data (dict): {'x': tensor (N,), 'edge_index': tensor (2, E), 'edge_attr': tensor (E,)}
                     Assumes tensors contain INTEGER VOCABULARY IDs (e.g., 97-102).
        atom_type (list): List of node type token strings ['NODE_CONST0', ...]
        bond_type (list): List of edge type token strings ['EDGE_INV', 'EDGE_REG']

    Returns:
        dict: {"text": [sequence_string]}
    """
    if 'x' not in data or 'edge_index' not in data or 'edge_attr' not in data:
        print("Warning (Deg): Input data dictionary missing required keys ('x', 'edge_index', 'edge_attr').")
        return {"text": ["<boc> <eoc> <bog> <eog>"]}

    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]

    if num_nodes == 0:
        return {"text": ["<boc> <eoc> <bog> <eog>"]}

    # --- 1. Build Node Context (<boc>...<eoc>) ---
    # Node context lists nodes in their original 0..N-1 index order
    ctx = ['<boc>']
    node_indices_map = {} # Map node index (0..N-1) to IDX_n token string
    node_id_to_type_token = {} # Map node index to its type token string
    node_vocab_offset = 97

    for node_idx in range(num_nodes):
        idx_token_str = f'IDX_{node_idx}'
        node_indices_map[node_idx] = idx_token_str

        node_vocab_id = x_ids[node_idx].item()
        node_token_index = node_vocab_id - node_vocab_offset
        if 0 <= node_token_index < len(atom_type):
            node_type_str = atom_type[node_token_index]
        else:
            print(f"Warning (Deg Node Ctx): Node {node_idx} unexpected ID {node_vocab_id}. UNK type.")
            node_type_str = "[UNK]"

        node_id_to_type_token[node_idx] = node_type_str
        ctx.extend(['<sepc>', node_type_str, idx_token_str])
    ctx.append('<eoc>')

    # --- 2. Build NetworkX DiGraph ---
    # We build a mutable copy to remove edges during selection
    G = nx.DiGraph()
    original_edge_attributes = {} # Store original attributes keyed by (u,v)
    edge_vocab_offset = 101

    # Add nodes
    for node_idx in range(num_nodes):
        G.add_node(node_idx, type=node_id_to_type_token.get(node_idx, "[UNK]"))

    # Add edges and store attributes
    num_edges_original = edge_index.shape[1]
    if num_edges_original != edge_attr_ids.shape[0]:
         print(f"Warning (Deg Graph Build): Mismatch between edge_index count ({num_edges_original}) and edge_attr count ({edge_attr_ids.shape[0]}).")

    for i in range(num_edges_original):
        src_node_idx = edge_index[0, i].item()
        dst_node_idx = edge_index[1, i].item()

        # Ensure indices are valid
        if src_node_idx >= 0 and src_node_idx < num_nodes and \
           dst_node_idx >= 0 and dst_node_idx < num_nodes:
            # Add edge to the graph we'll modify
            G.add_edge(src_node_idx, dst_node_idx)

            # Store original edge type string in the map
            edge_vocab_id = edge_attr_ids[i].item()
            edge_token_index = edge_vocab_id - edge_vocab_offset
            if 0 <= edge_token_index < len(bond_type):
                bond_type_str = bond_type[edge_token_index]
            else:
                print(f"Warning (Deg Edge Attr): Edge ({src_node_idx}->{dst_node_idx}) unexpected ID {edge_vocab_id}. UNK type.")
                bond_type_str = "[UNK]"
            # Store the attribute (overwrites if parallel edges exist, assumes simple for seq)
            original_edge_attributes[(src_node_idx, dst_node_idx)] = bond_type_str
        else:
            print(f"Warning (Deg Graph Build): Skipping edge ({src_node_idx}->{dst_node_idx}) due to invalid node index (max is {num_nodes-1}).")

    # --- 3. Iteratively Select Edges based on Degree and Build Edge Sequence ---
    outputs = ['<bog>']
    ordered_edges = [] # List to store edges in the selected order

    current_num_edges = G.number_of_edges()
    while current_num_edges > 0:
        # Calculate current out-degrees for all nodes
        # G.out_degree() returns an iterator of (node, degree) pairs
        out_degrees = dict(G.out_degree())

        # Find source nodes with minimum *non-zero* out-degree
        min_out_degree = float('inf')
        candidate_sources = []
        for node, degree in out_degrees.items():
            if degree > 0: # Only consider nodes with outgoing edges
                if degree < min_out_degree:
                    min_out_degree = degree
                    candidate_sources = [node]
                elif degree == min_out_degree:
                    candidate_sources.append(node)

        if not candidate_sources:
            # Should not happen if current_num_edges > 0, but break defensively
            print("Warning (Deg Edge Sel): No source nodes with non-zero out-degree found, but edges remain.")
            break

        # Tie-breaking for source: choose the one with the smallest node index
        selected_source = min(candidate_sources)

        # Find neighbors of the selected source
        neighbors = list(G.successors(selected_source))
        if not neighbors:
             # Should not happen if source had non-zero out-degree
             print(f"Warning (Deg Edge Sel): Source {selected_source} selected with non-zero degree but has no successors.")
             # As a fallback, remove the node and try again? Or just break. Let's break.
             break


        # Find the neighbor(s) with the minimum out-degree (can be zero)
        min_neighbor_degree = float('inf')
        candidate_destinations = []
        for neighbor in neighbors:
            neighbor_degree = out_degrees.get(neighbor, 0) # Use .get() for nodes with 0 out-degree now
            if neighbor_degree < min_neighbor_degree:
                min_neighbor_degree = neighbor_degree
                candidate_destinations = [neighbor]
            elif neighbor_degree == min_neighbor_degree:
                candidate_destinations.append(neighbor)

        # Tie-breaking for destination: choose the one with the smallest node index
        selected_destination = min(candidate_destinations)

        # --- Record the selected edge ---
        edge_tuple = (selected_source, selected_destination)
        src_token_str = node_indices_map.get(selected_source)
        dest_token_str = node_indices_map.get(selected_destination)
        # Retrieve the original edge attribute
        bond_type_str = original_edge_attributes.get(edge_tuple)

        if src_token_str and dest_token_str and bond_type_str:
             outputs.extend(['<sepg>', src_token_str, dest_token_str, bond_type_str])
             ordered_edges.append(edge_tuple) # Keep track if needed
        else:
             print(f"Warning (Deg Edge Seq): Missing data for selected edge {edge_tuple}. Src:{src_token_str}, Dst:{dest_token_str}, Type:{bond_type_str}")

        # --- Remove the edge from the graph for the next iteration ---
        if G.has_edge(*edge_tuple):
            G.remove_edge(*edge_tuple)
        else:
            # This indicates a logic error or graph state issue
            print(f"Error (Deg Edge Sel): Attempted to remove non-existent edge {edge_tuple}.")
            break # Avoid infinite loop

        current_num_edges = G.number_of_edges() # Update edge count

    outputs.append('<eog>')
    if len(outputs) == 2: # Only <bog> and <eog>
        outputs = ['<bog>', '<eog>']

    # Sanity check (optional)
    if len(ordered_edges) != num_edges_original:
         print(f"Warning (Deg Edge Seq): Number of selected edges ({len(ordered_edges)}) does not match original edge count ({num_edges_original}).")


    # --- 4. Combine and Return ---
    return {"text": [" ".join(ctx + outputs)]}


# --- Modified get_datasets function ---
def get_datasets(dataset_name, tokenizer, order='bfs'):
    """
    Loads the specified dataset and prepares it for the model.

    Args:
        dataset_name (str): Name of the dataset ('aig', 'qm9', 'tree').
        tokenizer: Pre-initialized tokenizer instance.
        order (str): Node/edge ordering strategy ('bfs', 'topo', 'deg').

    Returns:
        tuple: (train_datasets, eval_datasets)
    """
    print(f"Loading dataset: {dataset_name} with ordering: {order}")

    # Select the sequence generation function based on dataset and order
    if order == 'topo':
        print(f"Using topological sequence generation logic.")
        order_function = to_seq_aig_topo
    elif order == 'bfs':
        print(f"Using BFS sequence generation logic.")
        order_function = to_seq_by_bfs
    elif order == 'deg':
        print(f"Using Degree-based sequence generation logic.")
        order_function = to_seq_by_deg
    else:
        # Default or fallback logic
        print(f"Warning: Unsupported order '{order}'. Defaulting to BFS.")
        order_function = to_seq_by_bfs # Default to BFS


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
        ATOM_TYPE = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'] # IDs 97, 98, 99, 100
        BOND_TYPE = ['EDGE_INV', 'EDGE_REG'] # IDs 101, 102
        meta_path = os.path.join(data_dir, 'data_meta.json')
        try:
            # Ensure data_dir exists
            if not os.path.isdir(data_dir):
                 raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"AIG metadata file not found: {meta_path}. Did prepare_aig.py run?")

            with open(meta_path, 'r') as f:
                data_meta = json.load(f)

            # Check if required keys exist in meta file
            if 'train_shape' not in data_meta or 'eval_shape' not in data_meta:
                 raise KeyError("Missing 'train_shape' or 'eval_shape' in data_meta.json")

            # Validate shape dictionary keys before tuple conversion
            required_shape_keys = ['xs', 'edge_indices', 'edge_attrs']
            for shape_dict in [data_meta['train_shape'], data_meta['eval_shape']]:
                 for key in required_shape_keys:
                      if key not in shape_dict:
                           raise KeyError(f"Missing key '{key}' in shape dictionary within data_meta.json")

            train_shape = {k: tuple(v) for k, v in data_meta['train_shape'].items()}
            eval_shape = {k: tuple(v) for k, v in data_meta['eval_shape'].items()}
        except FileNotFoundError as e:
             raise FileNotFoundError(f"Error accessing AIG dataset files: {e}")
        except KeyError as e:
             raise KeyError(f"Error reading AIG metadata from {meta_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse AIG meta {meta_path}: {e}")

    elif dataset_name == 'qm9':
        # Note: QM9 might require different logic if node/edge IDs differ
        print("Warning: QM9 dataset selected. Ensure atom/bond types and IDs match expectation if using.")
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_O', 'ATOM_F'] # Example, verify IDs
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC'] # Example, verify IDs
        # Placeholder shapes - replace with actual QM9 shapes if loading from files
        train_shape = {'xs': (97732, 9), 'edge_indices': (97732, 2, 28), 'edge_attrs': (97732, 28)}
        eval_shape = {'xs': (20042, 9), 'edge_indices': (20042, 2, 26), 'edge_attrs': (20042, 26)}
        # Need to ensure the processing functions handle QM9 IDs correctly

    elif dataset_name == 'tree':
         # Note: Tree dataset might require different logic
        print("Warning: Tree dataset selected. Ensure atom/bond types and IDs match expectation if using.")
        ATOM_TYPE = ['NODE'] # Example
        BOND_TYPE = ['EDGE'] # Example
        # Placeholder shapes - replace with actual Tree shapes if loading from files
        train_shape = {'xs': (256, 64), 'edge_indices': (256, 2, 126), 'edge_attrs': (256, 126)}
        eval_shape = {'xs': (64, 64), 'edge_indices': (64, 2, 126), 'edge_attrs': (64, 126)}
        # Need to ensure the processing functions handle Tree IDs correctly

    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' setup is not implemented.")

    # --- Instantiate Datasets ---
    if ATOM_TYPE and BOND_TYPE and train_shape and eval_shape:
        # Basic validation of shapes
        if not all(k in train_shape for k in ['xs', 'edge_indices', 'edge_attrs']) or \
           not all(k in eval_shape for k in ['xs', 'edge_indices', 'edge_attrs']):
            raise ValueError("Train or eval shape dictionary is missing required keys ('xs', 'edge_indices', 'edge_attrs')")

        num_train = train_shape['xs'][0]
        num_eval = eval_shape['xs'][0]
        num_node_classes = len(ATOM_TYPE)
        num_edge_classes = len(BOND_TYPE) # Note: Padding handling might differ based on vocab

        # Define the processing function using partial, binding the top-level pre_tokenize_function
        # Pass tokenizer, order_function, and types explicitly here
        process_fn = partial(pre_tokenize_function,
                             tokenizer=tokenizer,
                             order_function=order_function, # The selected function (bfs, topo, deg)
                             atom_type=ATOM_TYPE,
                             bond_type=BOND_TYPE)

        try:
            train_path = os.path.join(data_dir, 'train')
            eval_path = os.path.join(data_dir, 'eval')

            # Check if dataset paths exist
            if not os.path.isdir(train_path):
                 raise FileNotFoundError(f"Training data directory not found: {train_path}")
            if not os.path.isdir(eval_path):
                 raise FileNotFoundError(f"Evaluation data directory not found: {eval_path}")


            train_datasets = NumpyBinDataset(train_path,
                                             num_train, num_node_classes, num_edge_classes,
                                             shape=train_shape,
                                             process_fn=process_fn)  # Pass the partial function
            eval_datasets = NumpyBinDataset(eval_path,
                                            num_eval, num_node_classes, num_edge_classes,
                                            shape=eval_shape,
                                            process_fn=process_fn)  # Pass the partial function
        except FileNotFoundError as e:
             raise FileNotFoundError(f"Error initializing dataset paths: {e}. Ensure data is correctly placed.")
        except Exception as e:
             raise RuntimeError(f"Error creating NumpyBinDataset instances: {e}")

    else:
        raise RuntimeError(f"Missing configuration (ATOM_TYPE, BOND_TYPE, shapes) for dataset {dataset_name}")

    # Final check
    if train_datasets is None or eval_datasets is None:
        raise RuntimeError(f"Failed to initialize datasets for {dataset_name}")

    print(f"Successfully loaded {dataset_name} - Train: {len(train_datasets)} samples, Eval: {len(eval_datasets)} samples.")
    return train_datasets, eval_datasets