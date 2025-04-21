# G2PT/datasets_utils.py

# --- Keep all original imports ---
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
from collections import deque
import numpy as np
import os
import re
from functools import partial
import json
import networkx as nx
import random # <-- IMPORT RANDOM

# --- Modified pre_tokenize_function ---
def pre_tokenize_function(examples, tokenizer, order_function, atom_type, bond_type, aug_seed=None): # <-- Added aug_seed
    """
    Top-level function to process data examples into tokenized sequences.
    Now accepts tokenizer, order_function, and aug_seed as arguments.
    """
    # Ensure 'examples' dict contains integer tensors 'x', 'edge_index', 'edge_attr'
    # Use the passed order_function
    data_dict = {'x': examples['x'], 'edge_index': examples['edge_index'], 'edge_attr': examples['edge_attr']}
    # !!! Pass aug_seed to the order function !!!
    sequence_data = order_function(data_dict, atom_type, bond_type, aug_seed=aug_seed)

    # Tokenize the generated text sequence (expects list with one item)
    sequence_string = sequence_data.get("text", [""])[0]
    if not sequence_string:
         # Handle empty sequence generation if needed
         sequence_string = "<boc> <eoc> <bog> <eog>" # Default empty graph sequence

    tokenized_data = tokenizer(sequence_string, padding='max_length', truncation=True,
                               return_tensors='pt')

    # Ensure tensors are correctly shaped (remove batch dim if tokenizer adds one)
    input_ids = tokenized_data['input_ids'].squeeze(0) if tokenized_data['input_ids'].ndim > 1 else tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask'].squeeze(0) if tokenized_data['attention_mask'].ndim > 1 else tokenized_data['attention_mask']

    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# --- Modified to_seq_aig_topo ---
def to_seq_aig_topo(data, atom_type, bond_type, aug_seed=None): # <-- Added aug_seed
    """
    Converts AIG data to sequence using topological ordering.
    Introduces randomness in tie-breaking if aug_seed is provided.
    """
    # Initialize local random generator if seed is provided
    local_random = random.Random(aug_seed) if aug_seed is not None else random

    # --- (Initial checks and graph building remain mostly the same) ---
    if 'x' not in data or 'edge_index' not in data or 'edge_attr' not in data:
        print("Warning (Topo): Input data dictionary missing required keys ('x', 'edge_index', 'edge_attr').")
        return {"text": ["<boc> <eoc> <bog> <eog>"]}
    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]
    if num_nodes == 0: return {"text": ["<boc> <eoc> <bog> <eog>"]}

    G = nx.DiGraph()
    node_idx_map = {}
    node_id_to_token_map = {}
    node_vocab_offset = 97
    for node_idx in range(num_nodes):
        G.add_node(node_idx)
        node_idx_map[node_idx] = f'IDX_{node_idx}'
        node_id_val = x_ids[node_idx].item()
        node_token_index = node_id_val - node_vocab_offset
        node_id_to_token_map[node_idx] = atom_type[node_token_index] if 0 <= node_token_index < len(atom_type) else "[UNK]"

    edge_id_to_token_map = {}
    edge_vocab_offset = 101
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        src_node_idx, dst_node_idx = edge_index[0, i].item(), edge_index[1, i].item()
        edge_id_val = edge_attr_ids[i].item()
        if src_node_idx in G and dst_node_idx in G:
            G.add_edge(src_node_idx, dst_node_idx)
            edge_token_index = edge_id_val - edge_vocab_offset
            edge_id_to_token_map[(src_node_idx, dst_node_idx)] = bond_type[edge_token_index] if 0 <= edge_token_index < len(bond_type) else "[UNK]"

    # --- Topological Sort / BFS Fallback ---
    try:
        # NetworkX topological_sort doesn't easily expose tie-breaking control.
        # For augmentation, we rely more on the edge ordering randomization below.
        # If we *really* needed node order variations, we might implement a custom topo sort.
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Warning (Topo Sort): Graph contains a cycle. Falling back to BFS order for sequence generation.")
        # Fallback uses BFS - use the augmented get_bfs_edge_order logic indirectly
        # We need the node order from BFS here.
        roots = [n for n, d in G.in_degree() if d == 0]
        if aug_seed is not None and len(roots) > 1:
            start_node = local_random.choice(roots)
        elif roots:
            start_node = min(roots) # Deterministic fallback
        else: start_node = min(G.nodes()) if G.nodes() else -1

        if start_node != -1:
            bfs_nodes_order = list(nx.bfs_tree(G, source=start_node)) # Node order from standard BFS
            # Could potentially randomize neighbor order in nx.bfs_tree for more variation? Not standard.
            if len(bfs_nodes_order) < G.number_of_nodes():
                 remaining_nodes = list(set(G.nodes()) - set(bfs_nodes_order))
                 if aug_seed is not None: local_random.shuffle(remaining_nodes) # Shuffle remaining for augmentation
                 else: remaining_nodes.sort() # Deterministic order
                 bfs_nodes_order.extend(remaining_nodes)
            topo_order = bfs_nodes_order
        else: topo_order = list(G.nodes())


    # --- Build Node Context (<boc>...<eoc>) ---
    # Node context order depends on the determined topo_order (potentially randomized in cycle fallback)
    ctx = ['<boc>']
    for node_idx in topo_order:
        node_token = node_id_to_token_map.get(node_idx, "[UNK]")
        node_idx_token = node_idx_map.get(node_idx, "IDX_?")
        ctx.extend(['<sepc>', node_token, node_idx_token])
    ctx.append('<eoc>')

    # --- Build Edge Sequence (<bog>...<eog>) ---
    outputs = ['<bog>']
    processed_edges = set()
    # Iterate through nodes in the determined (potentially variable) topo_order
    for u in topo_order:
        successors = list(G.successors(u))
        # !!! Randomize successor order if augmenting !!!
        if aug_seed is not None:
            local_random.shuffle(successors)
        else:
            successors.sort() # Deterministic order

        for v in successors:
            edge_tuple = (u, v)
            if edge_tuple in edge_id_to_token_map:
                edge_token = edge_id_to_token_map[edge_tuple]
                src_token_str = node_idx_map.get(u)
                dst_token_str = node_idx_map.get(v)
                if src_token_str and dst_token_str:
                    outputs.extend(['<sepg>', src_token_str, dst_token_str, edge_token])
                    processed_edges.add(edge_tuple)
            # else: (Handle warning if needed)

    outputs.append('<eog>')
    if len(outputs) == 2: outputs = ['<bog>', '<eog>']

    # Return dict with list containing one sequence string
    return {"text": [" ".join(ctx + outputs)]}


# --- Modified seq_to_nxgraph (no changes needed for augmentation) ---
def seq_to_nxgraph(seq_str, parsing_mode='strict'):
    # (This function remains the same)
    # ... (previous implementation) ...
    tokens = seq_str.split()
    try:
        ctx_start = tokens.index('<boc>') + 1; ctx_end = tokens.index('<eoc>')
        bog_start = tokens.index('<bog>') + 1; eog_end = tokens.index('<eog>')
    except ValueError: return nx.DiGraph() # Malformed
    ctx_tokens = tokens[ctx_start:ctx_end]
    edge_tokens = [t for t in tokens[bog_start:eog_end] if t != '<sepg>']
    G = nx.DiGraph(); node_map = {}; node_data = {}
    idx_pattern = re.compile(r'IDX_(\d+)'); node_type_pattern = re.compile(r'(NODE_[A-Z0-9]+|ATOM_[A-Za-z]+|NODE)')
    edge_type_pattern = re.compile(r'^(EDGE_[A-Z]+|BOND_[A-Z]+|EDGE)$'); node_idx_pattern = re.compile(r'^IDX_\d+$')
    current_node_idx_str = None; current_node_type_str = None; node_counter = 0; processed_idx_tokens = set()
    for token in ctx_tokens: # Parse nodes
        if token == '<sepc>': current_node_type_str = None; current_node_idx_str = None; continue
        node_match = node_type_pattern.match(token); idx_match = idx_pattern.match(token)
        if node_match: current_node_type_str = node_match.group(0)
        elif idx_match: current_node_idx_str = idx_match.group(0)
        if current_node_type_str and current_node_idx_str:
            if current_node_idx_str not in processed_idx_tokens:
                try: node_index = int(idx_match.group(1))
                except: node_index = node_counter # Fallback if IDX_ token malformed
                node_map[current_node_idx_str] = node_index
                node_data[node_index] = {'type': current_node_type_str}; processed_idx_tokens.add(current_node_idx_str)
                node_counter = max(node_counter, node_index + 1)
            current_node_type_str = None; current_node_idx_str = None
    G.add_nodes_from([(idx, node_data.get(idx, {'type': 'UNKNOWN'})) for idx in range(node_counter)])
    nx.set_node_attributes(G, {idx: data for idx, data in node_data.items()})
    if parsing_mode == 'strict': # Parse edges (strict)
        if len(edge_tokens) % 3 != 0: return G # Malformed edges
        for i in range(0, len(edge_tokens), 3):
            src_id_str, dest_id_str, edge_type_str = edge_tokens[i], edge_tokens[i+1], edge_tokens[i+2]
            src_idx, dest_idx = node_map.get(src_id_str), node_map.get(dest_id_str)
            edge_type = edge_type_str if edge_type_pattern.match(edge_type_str) else 'UNKNOWN_EDGE'
            if src_idx is not None and dest_idx is not None and G.has_node(src_idx) and G.has_node(dest_idx): G.add_edge(src_idx, dest_idx, type=edge_type)
    elif parsing_mode == 'robust': # Parse edges (robust)
        idx = 0
        while idx < len(edge_tokens):
            is_triplet = False
            if idx + 2 < len(edge_tokens) and node_idx_pattern.match(edge_tokens[idx]) and node_idx_pattern.match(edge_tokens[idx+1]) and edge_type_pattern.match(edge_tokens[idx+2]): is_triplet = True
            if is_triplet:
                src_id_str, dest_id_str, edge_type_str = edge_tokens[idx], edge_tokens[idx+1], edge_tokens[idx+2]
                src_idx, dest_idx = node_map.get(src_id_str), node_map.get(dest_id_str)
                if src_idx is not None and dest_idx is not None and G.has_node(src_idx) and G.has_node(dest_idx): G.add_edge(src_idx, dest_idx, type=edge_type_str)
                idx += 3
            else: idx += 1 # Skip token
    return G

# --- Modified NumpyBinDataset ---
class NumpyBinDataset(Dataset):
    """
    Loads graph data preprocessed into numpy memmap files (.bin).
    Handles data augmentation by multiplying dataset size.
    """
    def __init__(self, path, num_data, num_node_class, num_edge_class, shape, process_fn, num_augmentations=1): # <-- Added num_augmentations
        self.path = path
        self.original_num_data = num_data # Store original count
        self.num_augmentations = max(1, num_augmentations) # Ensure at least 1
        self.num_data = self.original_num_data * self.num_augmentations # Effective size
        self.num_node_class = num_node_class
        self.num_edge_class = num_edge_class
        self.process_fn = process_fn # This will be the partially filled pre_tokenize_function

        local_shape = shape.copy()
        try: # Shape validation
            required_keys = ['xs', 'edge_indices', 'edge_attrs']
            for key in required_keys: local_shape[key] = tuple(local_shape[key])
        except KeyError as e: raise KeyError(f"Missing key {e} in shape dict: {local_shape}")
        except Exception as e: raise RuntimeError(f"Error processing shape dict {local_shape}: {e}")

        try: # Memmap loading
            self.xs = np.memmap(os.path.join(path, 'xs.bin'), dtype=np.int16, mode='r', shape=local_shape['xs'])
            self.edge_indices = np.memmap(os.path.join(path, 'edge_indices.bin'), dtype=np.int16, mode='r', shape=local_shape['edge_indices'])
            self.edge_attrs = np.memmap(os.path.join(path, 'edge_attrs.bin'), dtype=np.int16, mode='r', shape=local_shape['edge_attrs'])
        except FileNotFoundError as e: raise FileNotFoundError(f"Error opening memmap files in {path}: {e}")
        except Exception as e: raise RuntimeError(f"Error setting up memmap in {path} with shapes {local_shape}: {e}")

    def __len__(self):
        # Return augmented length
        return self.num_data

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_data:
            raise IndexError(f"Index {idx} out of bounds for augmented dataset size {self.num_data}")

        # Calculate original graph index and augmentation index
        graph_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations # Use this as the seed for this augmentation
        aug_seed = aug_idx # Pass augmentation index as the seed

        # Load raw data for the original graph index
        try:
            raw_x = np.array(self.xs[graph_idx]).astype(np.int64)
            raw_edge_index = np.array(self.edge_indices[graph_idx]).astype(np.int64)
            raw_edge_attr = np.array(self.edge_attrs[graph_idx]).astype(np.int64)
        except IndexError:
            # This might happen if original_num_data was wrong
            print(f"Error: Original graph index {graph_idx} (derived from augmented index {idx}) out of bounds for memmap arrays (original num_data={self.original_num_data}).")
            empty_data = {'x': torch.tensor([], dtype=torch.long), 'edge_index': torch.tensor([[], []], dtype=torch.long), 'edge_attr': torch.tensor([], dtype=torch.long)}
            # Process empty data, passing aug_seed (though it won't be used)
            return self.process_fn(empty_data, aug_seed=aug_seed) # <-- Pass aug_seed here
        except Exception as e:
            print(f"Error accessing memmap data at original index {graph_idx}: {e}")
            empty_data = {'x': torch.tensor([], dtype=torch.long), 'edge_index': torch.tensor([[], []], dtype=torch.long), 'edge_attr': torch.tensor([], dtype=torch.long)}
            return self.process_fn(empty_data, aug_seed=aug_seed) # <-- Pass aug_seed here


        # --- Preprocessing (filtering padding, remapping indices) ---
        # (This logic remains the same as before)
        node_padding_mask = raw_x != -100
        x_ids = torch.from_numpy(raw_x[node_padding_mask])
        num_valid_nodes = len(x_ids)
        old_indices = np.arange(len(raw_x)); new_indices_map = -np.ones_like(old_indices)
        new_indices_map[node_padding_mask] = np.arange(num_valid_nodes)
        if num_valid_nodes == 0:
            empty_data = {'x': x_ids, 'edge_index': torch.tensor([[], []], dtype=torch.long), 'edge_attr': torch.tensor([], dtype=torch.long)}
            return self.process_fn(empty_data, aug_seed=aug_seed) # <-- Pass aug_seed here

        if raw_edge_attr.ndim > 1: raw_edge_attr = raw_edge_attr.flatten()
        edge_padding_mask = raw_edge_attr != -100
        edge_attr_ids_filtered_by_attr = torch.from_numpy(raw_edge_attr[edge_padding_mask])
        if edge_padding_mask.shape[0] != raw_edge_index.shape[1]:
             edge_index_filtered_by_attr = torch.tensor([[], []], dtype=torch.long)
             edge_attr_ids_filtered_by_attr = torch.tensor([], dtype=torch.long)
        else: edge_index_filtered_by_attr = torch.from_numpy(raw_edge_index[:, edge_padding_mask])

        if edge_index_filtered_by_attr.numel() > 0:
             src_nodes_old = edge_index_filtered_by_attr[0, :].numpy(); dst_nodes_old = edge_index_filtered_by_attr[1, :].numpy()
             src_nodes_new = new_indices_map[src_nodes_old]; dst_nodes_new = new_indices_map[dst_nodes_old]
             valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)
             edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask], dst_nodes_new[valid_edge_mask]], dtype=torch.long)
             edge_attr_final = edge_attr_ids_filtered_by_attr[valid_edge_mask]
        else: edge_index_final = torch.tensor([[], []], dtype=torch.long); edge_attr_final = torch.tensor([], dtype=torch.long)
        # --- End Preprocessing ---

        # Prepare the final data dictionary for the sequence generation function
        data_dict = {'x': x_ids, 'edge_index': edge_index_final, 'edge_attr': edge_attr_final}

        # Pass the dictionary AND the aug_seed to the processing function
        # The process_fn is partial(pre_tokenize_function, tokenizer=..., order_function=...)
        # We need to add the aug_seed to the arguments passed by partial
        return self.process_fn(data_dict, aug_seed=aug_seed) # <-- Pass aug_seed here


# --- Modified get_bfs_edge_order ---
def get_bfs_edge_order(G, aug_seed=None): # <-- Added aug_seed
    """ Performs BFS, returns edges. Randomizes tie-breaks if aug_seed provided. """
    local_random = random.Random(aug_seed) if aug_seed is not None else random

    if not G or G.number_of_edges() == 0: return []
    edges_bfs = []; visited_nodes = set(); nodes_to_process = list(G.nodes()); processed_edges = set()

    while nodes_to_process:
        start_node = -1
        possible_starts = [n for n in nodes_to_process if G.in_degree(n) == 0 and n not in visited_nodes]

        # !!! Randomize start node selection if augmenting !!!
        if aug_seed is not None and len(possible_starts) > 1:
            start_node = local_random.choice(possible_starts)
        elif possible_starts:
            start_node = min(possible_starts) # Deterministic
        else: # Fallback if no zero in-degree nodes left (cycles?)
            unvisited_remaining = sorted([n for n in nodes_to_process if n not in visited_nodes])
            if unvisited_remaining: start_node = unvisited_remaining[0]

        if start_node == -1: break # All processed

        queue = deque([start_node]); visited_nodes.add(start_node); nodes_to_process.remove(start_node)
        component_edges = []
        while queue:
            u = queue.popleft()
            successors = list(G.successors(u))
            # !!! Randomize neighbor processing order if augmenting !!!
            if aug_seed is not None:
                local_random.shuffle(successors)
            else:
                successors.sort() # Deterministic

            for v in successors:
                edge = (u, v)
                if edge not in processed_edges:
                    component_edges.append(edge); processed_edges.add(edge)
                if v not in visited_nodes:
                    visited_nodes.add(v)
                    if v in nodes_to_process: nodes_to_process.remove(v)
                    queue.append(v)
        edges_bfs.extend(component_edges)

    # (Optional sanity check)
    # if len(processed_edges) != G.number_of_edges(): ...

    return edges_bfs

# --- Modified to_seq_by_bfs ---
def to_seq_by_bfs(data, atom_type, bond_type, aug_seed=None): # <-- Added aug_seed
    """ Converts AIG data to sequence using BFS ordering. Randomizes if aug_seed provided. """
    # (Initial checks and node context building remain the same)
    if 'x' not in data or 'edge_index' not in data or 'edge_attr' not in data: return {"text": ["<boc> <eoc> <bog> <eog>"]}
    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]
    if num_nodes == 0: return {"text": ["<boc> <eoc> <bog> <eog>"]}
    ctx = ['<boc>']; node_indices_map = {}; node_id_to_type_token = {}; node_vocab_offset = 97
    for node_idx in range(num_nodes): # Build node context (deterministic order 0..N-1)
        idx_token_str = f'IDX_{node_idx}'; node_indices_map[node_idx] = idx_token_str
        node_vocab_id = x_ids[node_idx].item(); node_token_index = node_vocab_id - node_vocab_offset
        node_type_str = atom_type[node_token_index] if 0 <= node_token_index < len(atom_type) else "[UNK]"
        node_id_to_type_token[node_idx] = node_type_str; ctx.extend(['<sepc>', node_type_str, idx_token_str])
    ctx.append('<eoc>')

    # (Graph building remains the same)
    G = nx.DiGraph(); edge_data_map = {}; edge_vocab_offset = 101
    for node_idx in range(num_nodes): G.add_node(node_idx, type=node_id_to_type_token.get(node_idx, "[UNK]"))
    num_edges = edge_index.shape[1]
    # if num_edges != edge_attr_ids.shape[0]: print(...)
    for i in range(num_edges):
        src_node_idx, dst_node_idx = edge_index[0, i].item(), edge_index[1, i].item()
        if src_node_idx >= 0 and src_node_idx < num_nodes and dst_node_idx >= 0 and dst_node_idx < num_nodes:
            G.add_edge(src_node_idx, dst_node_idx)
            edge_vocab_id = edge_attr_ids[i].item(); edge_token_index = edge_vocab_id - edge_vocab_offset
            bond_type_str = bond_type[edge_token_index] if 0 <= edge_token_index < len(bond_type) else "[UNK]"
            edge_data_map[(src_node_idx, dst_node_idx)] = bond_type_str
        # else: print(...)

    # --- Get BFS Edge Order (passing seed) ---
    outputs = ['<bog>']
    # !!! Pass aug_seed to the helper function !!!
    edges_order_bfs = get_bfs_edge_order(G, aug_seed=aug_seed)

    # (Building output sequence remains the same)
    for src_idx, dest_idx in edges_order_bfs:
        src_token_str = node_indices_map.get(src_idx); dest_token_str = node_indices_map.get(dest_idx)
        bond_type_str = edge_data_map.get((src_idx, dest_idx))
        if src_token_str and dest_token_str and bond_type_str: outputs.extend(['<sepg>', src_token_str, dest_token_str, bond_type_str])
        # else: print(...)
    outputs.append('<eog>')
    if len(outputs) == 2: outputs = ['<bog>', '<eog>']

    return {"text": [" ".join(ctx + outputs)]}


# --- Modified to_seq_by_deg ---
def to_seq_by_deg(data, atom_type, bond_type, aug_seed=None): # <-- Added aug_seed
    """ Converts AIG data using Degree-based ordering. Randomizes tie-breaks if aug_seed provided. """
    local_random = random.Random(aug_seed) if aug_seed is not None else random

    # (Initial checks and node context building remain the same)
    if 'x' not in data or 'edge_index' not in data or 'edge_attr' not in data: return {"text": ["<boc> <eoc> <bog> <eog>"]}
    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]
    if num_nodes == 0: return {"text": ["<boc> <eoc> <bog> <eog>"]}
    ctx = ['<boc>']; node_indices_map = {}; node_id_to_type_token = {}; node_vocab_offset = 97
    for node_idx in range(num_nodes): # Build node context (deterministic order 0..N-1)
        idx_token_str = f'IDX_{node_idx}'; node_indices_map[node_idx] = idx_token_str
        node_vocab_id = x_ids[node_idx].item(); node_token_index = node_vocab_id - node_vocab_offset
        node_type_str = atom_type[node_token_index] if 0 <= node_token_index < len(atom_type) else "[UNK]"
        node_id_to_type_token[node_idx] = node_type_str; ctx.extend(['<sepc>', node_type_str, idx_token_str])
    ctx.append('<eoc>')

    # (Graph building and attribute storage remain the same)
    G = nx.DiGraph(); original_edge_attributes = {}; edge_vocab_offset = 101
    for node_idx in range(num_nodes): G.add_node(node_idx, type=node_id_to_type_token.get(node_idx, "[UNK]"))
    num_edges_original = edge_index.shape[1]
    # if num_edges_original != edge_attr_ids.shape[0]: print(...)
    for i in range(num_edges_original):
        src_node_idx, dst_node_idx = edge_index[0, i].item(), edge_index[1, i].item()
        if src_node_idx >= 0 and src_node_idx < num_nodes and dst_node_idx >= 0 and dst_node_idx < num_nodes:
            G.add_edge(src_node_idx, dst_node_idx) # Add to mutable graph G
            edge_vocab_id = edge_attr_ids[i].item(); edge_token_index = edge_vocab_id - edge_vocab_offset
            bond_type_str = bond_type[edge_token_index] if 0 <= edge_token_index < len(bond_type) else "[UNK]"
            original_edge_attributes[(src_node_idx, dst_node_idx)] = bond_type_str
        # else: print(...)

    # --- Iterative Edge Selection (with randomization) ---
    outputs = ['<bog>']
    ordered_edges = []
    # Create a copy of the graph to modify if G needs to be preserved, or modify G directly
    # Modifying G directly is more efficient here.
    current_num_edges = G.number_of_edges()
    while current_num_edges > 0:
        out_degrees = dict(G.out_degree())
        min_out_degree = float('inf'); candidate_sources = []
        for node, degree in out_degrees.items():
            if degree > 0: # Only consider nodes with outgoing edges remaining
                if degree < min_out_degree: min_out_degree = degree; candidate_sources = [node]
                elif degree == min_out_degree: candidate_sources.append(node)
        if not candidate_sources: break # Should not happen

        # !!! Randomize source tie-breaking if augmenting !!!
        if aug_seed is not None and len(candidate_sources) > 1:
            selected_source = local_random.choice(candidate_sources)
        else:
            selected_source = min(candidate_sources) # Deterministic

        neighbors = list(G.successors(selected_source))
        if not neighbors: break # Should not happen

        min_neighbor_degree = float('inf'); candidate_destinations = []
        for neighbor in neighbors:
            neighbor_degree = out_degrees.get(neighbor, 0)
            if neighbor_degree < min_neighbor_degree: min_neighbor_degree = neighbor_degree; candidate_destinations = [neighbor]
            elif neighbor_degree == min_neighbor_degree: candidate_destinations.append(neighbor)

        # !!! Randomize destination tie-breaking if augmenting !!!
        if aug_seed is not None and len(candidate_destinations) > 1:
            selected_destination = local_random.choice(candidate_destinations)
        else:
            selected_destination = min(candidate_destinations) # Deterministic

        # Record edge
        edge_tuple = (selected_source, selected_destination)
        src_token_str = node_indices_map.get(selected_source); dest_token_str = node_indices_map.get(selected_destination)
        bond_type_str = original_edge_attributes.get(edge_tuple)
        if src_token_str and dest_token_str and bond_type_str: outputs.extend(['<sepg>', src_token_str, dest_token_str, bond_type_str]); ordered_edges.append(edge_tuple)
        # else: print(...)

        # Remove edge
        if G.has_edge(*edge_tuple): G.remove_edge(*edge_tuple)
        else: print(f"Error (Deg): Attempted to remove non-existent edge {edge_tuple}."); break
        current_num_edges = G.number_of_edges()

    outputs.append('<eog>')
    if len(outputs) == 2: outputs = ['<bog>', '<eog>']
    # (Optional sanity check)
    # if len(ordered_edges) != num_edges_original: print(...)

    return {"text": [" ".join(ctx + outputs)]}


# --- Modified get_datasets function ---
def get_datasets(dataset_name, tokenizer, order='bfs', num_augmentations=1): # <-- Added num_augmentations
    """
    Loads the specified dataset and prepares it for the model.
    Handles data augmentation.
    """
    print(f"Loading dataset: {dataset_name} with ordering: {order} (Num augmentations: {num_augmentations})") # <-- Log augmentations

    # Select the base sequence generation function
    order_function = None
    if order == 'topo': order_function = to_seq_aig_topo
    elif order == 'bfs': order_function = to_seq_by_bfs
    elif order == 'deg': order_function = to_seq_by_deg
    else: raise NotImplementedError(f"Order function {order} is not implemented.")

    # --- Dataset Specific Setups (remain the same) ---
    # ... (load ATOM_TYPE, BOND_TYPE, shapes, etc.) ...
    train_datasets = None; eval_datasets = None; ATOM_TYPE = None; BOND_TYPE = None
    train_shape = None; eval_shape = None; data_dir = f'./datasets/{dataset_name}'
    if dataset_name == 'aig':
        ATOM_TYPE = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']; BOND_TYPE = ['EDGE_INV', 'EDGE_REG']
        meta_path = os.path.join(data_dir, 'data_meta.json')
        try:
            if not os.path.isdir(data_dir): raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
            if not os.path.exists(meta_path): raise FileNotFoundError(f"AIG metadata file not found: {meta_path}")
            with open(meta_path, 'r') as f: data_meta = json.load(f)
            required_shape_keys = ['xs', 'edge_indices', 'edge_attrs']
            if 'train_shape' not in data_meta or 'eval_shape' not in data_meta: raise KeyError("Missing shapes in meta")
            for shape_dict in [data_meta['train_shape'], data_meta['eval_shape']]:
                 for key in required_shape_keys:
                      if key not in shape_dict: raise KeyError(f"Missing key '{key}' in shapes")
            train_shape = {k: tuple(v) for k, v in data_meta['train_shape'].items()}
            eval_shape = {k: tuple(v) for k, v in data_meta['eval_shape'].items()}
        except Exception as e: raise RuntimeError(f"Failed to load or parse AIG meta {meta_path}: {e}")
    # Add elif for 'qm9', 'tree' if needed, loading their types/shapes
    else: raise NotImplementedError(f"Dataset '{dataset_name}' setup is not implemented.")
    # --- End Dataset Specific Setups ---


    if ATOM_TYPE and BOND_TYPE and train_shape and eval_shape:
        num_train_original = train_shape['xs'][0]
        num_eval_original = eval_shape['xs'][0]
        num_node_classes = len(ATOM_TYPE)
        num_edge_classes = len(BOND_TYPE)

        # !!! Define the processing function using partial BUT WE NEED TO PASS aug_seed LATER !!!
        # The process_fn stored in the dataset will be the partially filled pre_tokenize_function
        # The __getitem__ method will call it and provide the final 'aug_seed' argument.
        process_fn = partial(pre_tokenize_function,
                             tokenizer=tokenizer,
                             order_function=order_function, # Pass the selected base order function
                             atom_type=ATOM_TYPE,
                             bond_type=BOND_TYPE)
                             # NOTE: aug_seed is NOT bound here, it's provided in __getitem__

        try:
            train_path = os.path.join(data_dir, 'train')
            eval_path = os.path.join(data_dir, 'eval')
            if not os.path.isdir(train_path): raise FileNotFoundError(f"Training data directory not found: {train_path}")
            if not os.path.isdir(eval_path): raise FileNotFoundError(f"Evaluation data directory not found: {eval_path}")

            # !!! Pass num_augmentations to the dataset constructor !!!
            train_datasets = NumpyBinDataset(train_path,
                                             num_train_original, num_node_classes, num_edge_classes,
                                             shape=train_shape,
                                             process_fn=process_fn,
                                             num_augmentations=num_augmentations) # <-- Pass augmentation count
            # Typically, augmentation is only applied to the training set
            eval_datasets = NumpyBinDataset(eval_path,
                                            num_eval_original, num_node_classes, num_edge_classes,
                                            shape=eval_shape,
                                            process_fn=process_fn,
                                            num_augmentations=1) # <-- Eval set usually has no augmentation (num_augmentations=1)

        except FileNotFoundError as e: raise FileNotFoundError(f"Error initializing dataset paths: {e}")
        except Exception as e: raise RuntimeError(f"Error creating NumpyBinDataset instances: {e}")

    else: raise RuntimeError(f"Missing configuration for dataset {dataset_name}")

    if train_datasets is None or eval_datasets is None: raise RuntimeError(f"Failed to initialize datasets")

    # Report augmented size for training set
    logger.info(f"Successfully loaded {dataset_name} - Train: {len(train_datasets)} samples (augmented), Eval: {len(eval_datasets)} samples.")
    return train_datasets, eval_datasets