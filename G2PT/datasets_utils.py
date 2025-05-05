# G2PT/datasets_utils.py

# --- Keep all original imports ---
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
from collections import deque
import numpy as np
import os # <-- Added import
import re
from functools import partial
import json
import random
from collections import deque
import networkx as nx
import random # <-- IMPORT RANDOM
import sys # <-- Added import

# --- Import AIG Config ---
import configs.aig as aig_cfg



# --- pre_tokenize_function (No changes) ---
def pre_tokenize_function(examples, tokenizer, order_function, atom_type, bond_type, aug_seed=None):
    data_dict = {'x': examples['x'], 'edge_index': examples['edge_index'], 'edge_attr': examples['edge_attr']}
    sequence_data = order_function(data_dict, atom_type, bond_type, aug_seed=aug_seed)
    sequence_string = sequence_data.get("text", [""])[0]
    if not sequence_string: sequence_string = "<boc> <eoc> <bog> <eog>"
    tokenized_data = tokenizer(sequence_string, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokenized_data['input_ids'].squeeze(0) if tokenized_data['input_ids'].ndim > 1 else tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask'].squeeze(0) if tokenized_data['attention_mask'].ndim > 1 else tokenized_data['attention_mask']
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# --- custom_randomized_topological_sort (No changes) ---
def custom_randomized_topological_sort(G, random_generator):
    if not G.is_directed(): raise nx.NetworkXError("Topological sort not defined for undirected graphs.")
    in_degree_map = {node: degree for node, degree in G.in_degree()}
    zero_in_degree_nodes = [node for node, degree in in_degree_map.items() if degree == 0]
    if len(zero_in_degree_nodes) > 1: random_generator.shuffle(zero_in_degree_nodes)
    queue = deque(zero_in_degree_nodes); result_order = []
    while queue:
        u = queue.popleft(); result_order.append(u)
        newly_zero_in_degree = []
        for v in sorted(list(G.successors(u))):
            in_degree_map[v] -= 1
            if in_degree_map[v] == 0: newly_zero_in_degree.append(v)
            elif in_degree_map[v] < 0: raise RuntimeError(f"In-degree became negative for node {v}.")
        if len(newly_zero_in_degree) > 1: random_generator.shuffle(newly_zero_in_degree)
        for node in newly_zero_in_degree: queue.append(node)
    if len(result_order) != G.number_of_nodes(): raise nx.NetworkXUnfeasible(f"Graph contains a cycle.")
    return result_order

# --- to_seq_aig_topo (No changes) ---
def to_seq_aig_topo(data, atom_type, bond_type, aug_seed=None):
    local_random = random.Random(aug_seed) if aug_seed is not None else random
    if 'x' not in data or 'edge_index' not in data or 'edge_attr' not in data: return {"text": ["<boc> <eoc> <bog> <eog>"]}
    x_ids, edge_index, edge_attr_ids = data['x'], data['edge_index'], data['edge_attr']
    num_nodes = x_ids.shape[0]
    if num_nodes == 0: return {"text": ["<boc> <eoc> <bog> <eog>"]}
    G = nx.DiGraph(); node_idx_map = {}; node_id_to_token_map = {}
    node_vocab_offset = aig_cfg.NODE_VOCAB_OFFSET
    for node_idx in range(num_nodes):
        G.add_node(node_idx); node_idx_map[node_idx] = f'IDX_{node_idx}'
        node_id_val = x_ids[node_idx].item(); node_token_index = node_id_val - node_vocab_offset
        node_id_to_token_map[node_idx] = atom_type[node_token_index] if 0 <= node_token_index < len(atom_type) else "[UNK]"
    edge_id_to_token_map = {}; edge_vocab_offset = aig_cfg.EDGE_VOCAB_OFFSET
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        src_node_idx, dst_node_idx = edge_index[0, i].item(), edge_index[1, i].item()
        edge_id_val = edge_attr_ids[i].item()
        if src_node_idx in G and dst_node_idx in G:
            G.add_edge(src_node_idx, dst_node_idx)
            edge_token_index = edge_id_val - edge_vocab_offset
            edge_id_to_token_map[(src_node_idx, dst_node_idx)] = bond_type[edge_token_index] if 0 <= edge_token_index < len(bond_type) else "[UNK]"
    try: topo_order = list(custom_randomized_topological_sort(G, local_random))
    except nx.NetworkXUnfeasible as e:
        print(f"Warning (Custom Topo Sort): {e}. Falling back to BFS order.")
        roots = [n for n, d in G.in_degree() if d == 0]; start_node = -1
        if roots: start_node = local_random.choice(roots) if aug_seed is not None and len(roots) > 1 else min(roots)
        elif G.nodes(): start_node = min(G.nodes()); print(f"Warning (Cycle Fallback): Starting BFS from node {start_node}.")
        if start_node != -1:
            bfs_nodes_order = list(nx.bfs_tree(G, source=start_node)); remaining_nodes = list(set(G.nodes()) - set(bfs_nodes_order))
            if remaining_nodes: remaining_nodes.sort();
            if aug_seed is not None: local_random.shuffle(remaining_nodes); bfs_nodes_order.extend(remaining_nodes)
            topo_order = bfs_nodes_order
        else: topo_order = []
    ctx = ['<boc>']
    for node_idx in topo_order: ctx.extend(['<sepc>', node_id_to_token_map.get(node_idx, "[UNK]"), node_idx_map.get(node_idx, "IDX_?")])
    ctx.append('<eoc>')
    outputs = ['<bog>']; processed_edges = set()
    for u in topo_order:
        successors = list(G.successors(u)); successors.sort()
        if aug_seed is not None: local_random.shuffle(successors)
        for v in successors:
            edge_tuple = (u, v)
            if edge_tuple in edge_id_to_token_map:
                edge_token = edge_id_to_token_map[edge_tuple]; src_token_str = node_idx_map.get(u); dst_token_str = node_idx_map.get(v)
                if src_token_str and dst_token_str: outputs.extend(['<sepg>', src_token_str, dst_token_str, edge_token]); processed_edges.add(edge_tuple)
    outputs.append('<eog>')
    if len(outputs) == 2 and len(processed_edges) > 0: outputs = ['<bog>', '<eog>']
    return {"text": [" ".join(ctx + outputs)]}

# --- seq_to_nxgraph ---
def seq_to_nxgraph(seq_str, parsing_mode='strict'):
    """
    Parses a sequence string back into a NetworkX DiGraph.
    Assigns node/edge types based on the tokens found.
    Maps unknown edge patterns to 'UNKNOWN_EDGE'.
    """
    tokens = seq_str.split()
    try:
        ctx_start = tokens.index('<boc>') + 1; ctx_end = tokens.index('<eoc>')
        bog_start = tokens.index('<bog>') + 1; eog_end = tokens.index('<eog>')
    except ValueError: return nx.DiGraph() # Malformed
    ctx_tokens = tokens[ctx_start:ctx_end]
    edge_tokens = [t for t in tokens[bog_start:eog_end] if t != '<sepg>']
    G = nx.DiGraph(); node_map = {}; node_data = {}
    idx_pattern = re.compile(r'^IDX_(\d+)$') # Stricter IDX pattern
    # Stricter NODE pattern (adjust if other patterns like ATOM_ needed)
    node_type_pattern = re.compile(r'^NODE_(CONST0|PI|AND|PO)$')
    # <<< CHANGE HERE: Stricter edge type pattern >>>
    # Only match known edge types exactly
    edge_type_pattern = re.compile(r'^(EDGE_REG|EDGE_INV)$')
    # --- End Change ---

    current_node_idx_str = None; current_node_type_str = None; node_counter = 0; processed_idx_tokens = set()
    for token in ctx_tokens: # Parse nodes
        if token == '<sepc>': current_node_type_str = None; current_node_idx_str = None; continue
        # Use fullmatch for stricter checking
        node_match = re.fullmatch(r'NODE_[A-Z0-9]+', token) # Allow any NODE_ pattern initially
        idx_match = idx_pattern.fullmatch(token)
        if node_match: current_node_type_str = node_match.group(0)
        elif idx_match: current_node_idx_str = idx_match.group(0)
        if current_node_type_str and current_node_idx_str:
            if current_node_idx_str not in processed_idx_tokens:
                try: node_index = int(idx_pattern.match(current_node_idx_str).group(1)) # Use matched group
                except: node_index = node_counter # Fallback
                node_map[current_node_idx_str] = node_index
                # Assign type directly, validation happens elsewhere
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
            # <<< CHANGE HERE: Use stricter edge pattern match >>>
            edge_type = edge_type_str if edge_type_pattern.fullmatch(edge_type_str) else 'UNKNOWN_EDGE'
            # --- End Change ---
            if src_idx is not None and dest_idx is not None and G.has_node(src_idx) and G.has_node(dest_idx): G.add_edge(src_idx, dest_idx, type=edge_type)
    elif parsing_mode == 'robust': # Parse edges (robust)
        idx = 0
        while idx < len(edge_tokens):
            is_triplet = False
            # Check if the next three tokens form a valid-looking triplet
            if idx + 2 < len(edge_tokens) and \
               idx_pattern.fullmatch(edge_tokens[idx]) and \
               idx_pattern.fullmatch(edge_tokens[idx+1]):
               # Check edge type loosely first for triplet structure
               if re.match(r'^EDGE_', edge_tokens[idx+2]): # Allow any EDGE_ for structure
                    is_triplet = True

            if is_triplet:
                src_id_str, dest_id_str, edge_type_str = edge_tokens[idx], edge_tokens[idx+1], edge_tokens[idx+2]
                src_idx, dest_idx = node_map.get(src_id_str), node_map.get(dest_id_str)
                # <<< CHANGE HERE: Use stricter edge pattern match for assigning type >>>
                edge_type = edge_type_str if edge_type_pattern.fullmatch(edge_type_str) else 'UNKNOWN_EDGE'
                # --- End Change ---
                if src_idx is not None and dest_idx is not None and G.has_node(src_idx) and G.has_node(dest_idx):
                    G.add_edge(src_idx, dest_idx, type=edge_type) # Assign potentially UNKNOWN_EDGE
                idx += 3
            else: idx += 1 # Skip token if not a valid triplet structure
    return G


# --- NumpyBinDataset (No changes) ---
class NumpyBinDataset(Dataset):
    def __init__(self, path, num_data, num_node_class, num_edge_class, shape, process_fn, num_augmentations=1):
        self.path = path; self.original_num_data = num_data; self.num_augmentations = max(1, num_augmentations)
        self.num_data = self.original_num_data * self.num_augmentations; self.num_node_class = num_node_class
        self.num_edge_class = num_edge_class; self.process_fn = process_fn; self.pad_value = aig_cfg.PAD_VALUE
        local_shape = shape.copy()
        self.xs = np.memmap(os.path.join(path, 'xs.bin'), dtype=np.int16, mode='r', shape=local_shape['xs'])
        self.edge_indices = np.memmap(os.path.join(path, 'edge_indices.bin'), dtype=np.int16, mode='r', shape=local_shape['edge_indices'])
        self.edge_attrs = np.memmap(os.path.join(path, 'edge_attrs.bin'), dtype=np.int16, mode='r', shape=local_shape['edge_attrs'])

    def __len__(self): return self.num_data
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_data: raise IndexError(f"Index {idx} out of bounds for augmented dataset size {self.num_data}")
        graph_idx = idx // self.num_augmentations; aug_seed = idx % self.num_augmentations
        try:
            raw_x = np.array(self.xs[graph_idx]).astype(np.int64); raw_edge_index = np.array(self.edge_indices[graph_idx]).astype(np.int64)
            raw_edge_attr = np.array(self.edge_attrs[graph_idx]).astype(np.int64)
        except IndexError:
            print(f"Error: Original graph index {graph_idx} (derived from augmented index {idx}) out of bounds for memmap arrays (original num_data={self.original_num_data}).")
            empty_data = {'x': torch.tensor([], dtype=torch.long), 'edge_index': torch.tensor([[], []], dtype=torch.long), 'edge_attr': torch.tensor([], dtype=torch.long)}
            return self.process_fn(empty_data, aug_seed=aug_seed)
        except Exception as e:
            print(f"Error accessing memmap data at original index {graph_idx}: {e}")
            empty_data = {'x': torch.tensor([], dtype=torch.long), 'edge_index': torch.tensor([[], []], dtype=torch.long), 'edge_attr': torch.tensor([], dtype=torch.long)}
            return self.process_fn(empty_data, aug_seed=aug_seed)
        node_padding_mask = raw_x != self.pad_value; x_ids = torch.from_numpy(raw_x[node_padding_mask]); num_valid_nodes = len(x_ids)
        old_indices = np.arange(len(raw_x)); new_indices_map = -np.ones_like(old_indices, dtype=np.int64); new_indices_map[node_padding_mask] = np.arange(num_valid_nodes)
        if num_valid_nodes == 0: empty_data = {'x': x_ids, 'edge_index': torch.tensor([[], []], dtype=torch.long), 'edge_attr': torch.tensor([], dtype=torch.long)}; return self.process_fn(empty_data, aug_seed=aug_seed)
        if raw_edge_attr.ndim > 1: raw_edge_attr = raw_edge_attr.flatten()
        edge_padding_mask = raw_edge_attr != self.pad_value
        if edge_padding_mask.shape[0] != raw_edge_index.shape[1]:
             print(f"Warning: Shape mismatch edge_attr ({edge_padding_mask.shape[0]}) vs edge_index ({raw_edge_index.shape[1]}) after filtering for graph {graph_idx}. Assuming no valid edges.")
             edge_index_filtered_by_attr = torch.tensor([[], []], dtype=torch.long); edge_attr_ids_filtered_by_attr = torch.tensor([], dtype=torch.long)
        else:
             edge_attr_ids_filtered_by_attr = torch.from_numpy(raw_edge_attr[edge_padding_mask]); edge_index_filtered_by_attr = torch.from_numpy(raw_edge_index[:, edge_padding_mask])
        if edge_index_filtered_by_attr.numel() > 0:
             src_nodes_old = edge_index_filtered_by_attr[0, :].numpy(); dst_nodes_old = edge_index_filtered_by_attr[1, :].numpy()
             src_nodes_old = np.clip(src_nodes_old, 0, len(new_indices_map) - 1); dst_nodes_old = np.clip(dst_nodes_old, 0, len(new_indices_map) - 1)
             src_nodes_new = new_indices_map[src_nodes_old]; dst_nodes_new = new_indices_map[dst_nodes_old]
             valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)
             edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask], dst_nodes_new[valid_edge_mask]], dtype=torch.long)
             edge_attr_final = edge_attr_ids_filtered_by_attr[valid_edge_mask]
        else: edge_index_final = torch.tensor([[], []], dtype=torch.long); edge_attr_final = torch.tensor([], dtype=torch.long)
        data_dict = {'x': x_ids, 'edge_index': edge_index_final, 'edge_attr': edge_attr_final}
        return self.process_fn(data_dict, aug_seed=aug_seed)

# --- Modified get_datasets function ---
def get_datasets(dataset_name, tokenizer, order='topo', num_augmentations=1):
    """
    Loads the specified dataset and prepares it for the model.
    Handles data augmentation.
    Uses aig_cfg for paths and types.
    Correctly constructs paths for metadata and splits.
    """
    print(f"Loading dataset: {dataset_name} with ordering: {order} (Num augmentations: {num_augmentations})")

    order_function = None
    if order == 'topo': order_function = to_seq_aig_topo
    else: raise NotImplementedError(f"Order function {order} is not implemented.")

    train_datasets = None; eval_datasets = None; ATOM_TYPE = None; BOND_TYPE = None
    train_shape = None; eval_shape = None

    #dataset_specific_dir = "/Users/bellavg/aig-gen/G2PT/datasets/aig"
    # Construct the path to the specific dataset directory (e.g., /path/to/project/datasets/aig/)
    dataset_specific_dir = os.path.join(aig_cfg.data_dir)
    # Construct the path to the metadata file *within* the specific dataset directory
    meta_path = os.path.join(dataset_specific_dir, 'data_meta.json')
    # --- End Path Change ---

    print(f"get_datasets: Using dataset directory: {dataset_specific_dir}")
    print(f"get_datasets: Looking for metadata at: {meta_path}")

    if dataset_name == 'aig':
        ATOM_TYPE = aig_cfg.NODE_TYPE_KEYS
        BOND_TYPE = aig_cfg.EDGE_TYPE_KEYS
        try:
            if not os.path.isdir(dataset_specific_dir): raise FileNotFoundError(f"Dataset directory not found: {dataset_specific_dir}")
            if not os.path.exists(meta_path): raise FileNotFoundError(f"AIG metadata file not found: {meta_path}")

            with open(meta_path, 'r') as f: data_meta = json.load(f)
            required_shape_keys = ['xs', 'edge_indices', 'edge_attrs']
            train_key = 'train_shape' if 'train_shape' in data_meta else None
            eval_key = 'eval_shape' if 'eval_shape' in data_meta else ('val_shape' if 'val_shape' in data_meta else None)

            if not train_key or not eval_key: raise KeyError("Missing train_shape or eval/val_shape in meta")

            for shape_dict in [data_meta[train_key], data_meta[eval_key]]:
                 for key in required_shape_keys:
                      if key not in shape_dict: raise KeyError(f"Missing key '{key}' in shapes")

            train_shape = {k: tuple(v) for k, v in data_meta[train_key].items()}
            eval_shape = {k: tuple(v) for k, v in data_meta[eval_key].items()}
        except Exception as e: raise RuntimeError(f"Failed to load or parse AIG meta {meta_path}: {e}")
    else: raise NotImplementedError(f"Dataset '{dataset_name}' setup is not implemented.")

    if ATOM_TYPE and BOND_TYPE and train_shape and eval_shape:
        num_train_original = train_shape['xs'][0]
        num_eval_original = eval_shape['xs'][0]
        num_node_classes = len(ATOM_TYPE)
        num_edge_classes = len(BOND_TYPE)

        process_fn = partial(pre_tokenize_function,
                             tokenizer=tokenizer,
                             order_function=order_function,
                             atom_type=ATOM_TYPE,
                             bond_type=BOND_TYPE)

        try:
            # Use dataset_specific_dir for split paths
            train_path = os.path.join(dataset_specific_dir, 'train')
            eval_split_name = 'eval' if 'eval_shape' in data_meta else 'val'
            eval_path = os.path.join(dataset_specific_dir, eval_split_name)

            if not os.path.isdir(train_path): raise FileNotFoundError(f"Training data directory not found: {train_path}")
            if not os.path.isdir(eval_path): raise FileNotFoundError(f"{eval_split_name.capitalize()} data directory not found: {eval_path}")

            train_datasets = NumpyBinDataset(train_path,
                                             num_train_original, num_node_classes, num_edge_classes,
                                             shape=train_shape,
                                             process_fn=process_fn,
                                             num_augmentations=num_augmentations)
            eval_datasets = NumpyBinDataset(eval_path,
                                            num_eval_original, num_node_classes, num_edge_classes,
                                            shape=eval_shape,
                                            process_fn=process_fn,
                                            num_augmentations=1)

        except FileNotFoundError as e: raise FileNotFoundError(f"Error initializing dataset paths: {e}")
        except Exception as e: raise RuntimeError(f"Error creating NumpyBinDataset instances: {e}")

    else: raise RuntimeError(f"Missing configuration for dataset {dataset_name}")

    if train_datasets is None or eval_datasets is None: raise RuntimeError(f"Failed to initialize datasets")

    return train_datasets, eval_datasets
