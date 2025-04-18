# from datasets import IterableDataset
# from rdkit import Chem
# from rdkit.Chem import rdchem
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


import networkx as nx # Ensure this is imported
import torch
from collections import deque # Ensure this is imported

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
        return {"text": ["<boc> <eoc> <bog> <eog>"]} # Handle empty graph

    # 1. Build NetworkX DiGraph from input tensors
    G = nx.DiGraph()
    node_idx_map = {} # Map internal 0..N-1 index to IDX_n token
    node_id_to_token_map = {} # Map vocab ID (e.g., 97) to token ('NODE_CONST0')
    node_vocab_offset = 97 # ID of NODE_CONST0

    for node_idx in range(num_nodes):
        G.add_node(node_idx) # Add nodes using 0..N-1 indices
        node_idx_map[node_idx] = f'IDX_{node_idx}'
        node_id_val = x_ids[node_idx].item()
        node_token_index = node_id_val - node_vocab_offset
        if 0 <= node_token_index < len(atom_type):
            node_id_to_token_map[node_idx] = atom_type[node_token_index]
        else:
            print(f"Warning: Node {node_idx} has unexpected ID {node_id_val}. Assigning UNK type.")
            node_id_to_token_map[node_idx] = "[UNK]" # Or handle differently

    # Add edges to the DiGraph
    edge_id_to_token_map = {} # Store edge type tokens associated with (u, v) pairs
    edge_vocab_offset = 101 # ID of EDGE_INV
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
                 print(f"Warning: Edge ({src_node_idx}->{dst_node_idx}) has unexpected ID {edge_id_val}. Assigning UNK type.")
                 edge_id_to_token_map[(src_node_idx, dst_node_idx)] = "[UNK]" # Or handle differently
        else:
            print(f"Warning: Skipping edge ({src_node_idx}->{dst_node_idx}) due to missing node index.")


    # 2. Perform Topological Sort
    try:
        # Use standard topological sort
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Warning: Graph contains a cycle, cannot perform topological sort. Falling back to BFS order for sequence generation.")
        # Fallback: Use directed BFS - find a root (node with in-degree 0) or start at 0
        roots = [n for n, d in G.in_degree() if d == 0]
        start_node = roots[0] if roots else 0
        if start_node not in G: # Handle case where start_node might not exist
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
            topo_order = list(G.nodes()) # Fallback to arbitrary node order if graph is empty

    # 3. Build Node Context (<boc>...<eoc>) based on Topological Order
    ctx = ['<boc>']
    for node_idx in topo_order:
        node_token = node_id_to_token_map.get(node_idx, "[UNK]")
        node_idx_token = node_idx_map.get(node_idx, "IDX_?") # Should always exist if node is in topo_order
        ctx.extend(['<sepc>', node_token, node_idx_token])
    ctx.append('<eoc>')

    # 4. Build Edge Sequence (<bog>...<eog>)
    # Option: Iterate through nodes in topological order, then process their outgoing edges
    outputs = ['<bog>']
    processed_edges = set()
    for u in topo_order:
        # Process outgoing edges for node u
        # G.successors(u) or G.out_edges(u)
        for v in sorted(list(G.successors(u))): # Sort successors for determinism
            edge_tuple = (u, v)
            if edge_tuple in edge_id_to_token_map:
                edge_token = edge_id_to_token_map[edge_tuple]
                src_token_str = node_idx_map.get(u)
                dst_token_str = node_idx_map.get(v)
                if src_token_str and dst_token_str: # Ensure nodes are mapped
                     outputs.extend(['<sepg>', src_token_str, dst_token_str, edge_token])
                     processed_edges.add(edge_tuple)
            else:
                 # This might happen if edges were filtered earlier
                 print(f"Warning: Edge ({u}->{v}) found during traversal but missing from edge_id_to_token_map.")

    # Optional: Add any edges missed if traversal didn't cover all (e.g., due to cycles or disconnected parts)
    # This part might be redundant if topo_order covers all nodes and we iterate G.successors
    # for u, v in G.edges():
    #     if (u, v) not in processed_edges and (u,v) in edge_id_to_token_map:
    #          edge_token = edge_id_to_token_map[(u, v)]
    #          src_token_str = node_idx_map.get(u)
    #          dst_token_str = node_idx_map.get(v)
    #          if src_token_str and dst_token_str:
    #               outputs.extend(['<sepg>', src_token_str, dst_token_str, edge_token])

    outputs.append('<eog>')
    if len(outputs) == 2: # Only contains <bog> and <eog>
         outputs = ['<bog>', '<eog>'] # Ensure it's not empty if no edges

    return {"text": [" ".join(ctx + outputs)]}

# def check_valency(mol):
#     try:
#         Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
#         return True, None
#     except ValueError as e:
#         e = str(e)
#         p = e.find('#')
#         e_sub = e[p:]
#         atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
#         return False, atomid_valence
        
# def mol2smiles(mol):
#     try:
#         Chem.SanitizeMol(mol)
#     except ValueError:
#         return None
#     return Chem.MolToSmiles(mol)
#
# def get_smiles(mol):
#     smiles = mol2smiles(mol)
#     if smiles is not None:
#         try:
#             mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
#             largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
#             smiles = mol2smiles(largest_mol)
#             return smiles
#         except Chem.rdchem.AtomValenceException:
#             print("Valence error in GetmolFrags")
#             return None
#         except Chem.rdchem.KekulizeException:
#             print("Can't kekulize molecule")
#             return None
#     else:
#         return None

# Add to or ensure these imports are at the top of G2PT/datasets_utils.py
import networkx as nx
import re

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
        return nx.DiGraph() # Return empty graph

    ctx_tokens = tokens[ctx_start:ctx_end]
    edge_tokens = [token for token in tokens[bog_start:eog_end] if token != '<sepg>']

    G = nx.DiGraph() # Create a directed graph

    # --- Parse Nodes ---
    node_map = {} # Map IDX_n token string back to integer node index 0..N-1
    node_data = {} # Store node attributes {node_idx: {'type': 'NODE_TYPE_TOKEN'}}
    idx_pattern = re.compile(r'IDX_(\d+)')
    # More general pattern to capture different node type prefixes
    node_type_pattern = re.compile(r'(NODE_[A-Z0-9]+|ATOM_[A-Za-z]+|NODE)') # Add other prefixes if needed

    current_node_idx_str = None
    current_node_type_str = None
    node_counter = 0 # Assign sequential indices 0, 1, 2...
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
                     node_index = node_counter # Use sequential index 0, 1,...
                     node_map[current_node_idx_str] = node_index # Map IDX_n -> node_index
                     node_data[node_index] = {'type': current_node_type_str} # Store type attribute
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
        edge_type_str = edge_tokens[i+2]

        # Use the map to get integer node indices
        src_idx = node_map.get(src_id_str)
        dest_idx = node_map.get(dest_id_str)

        # Extract edge type token
        edge_type_match = edge_type_pattern.match(edge_type_str)
        edge_type = edge_type_match.group(0) if edge_type_match else 'UNKNOWN_EDGE' # Assign a default or raise error

        # Add directed edge if both nodes were found
        if src_idx is not None and dest_idx is not None:
             # Check if nodes actually exist in the graph (they should based on map)
             if src_idx in G and dest_idx in G:
                  G.add_edge(src_idx, dest_idx, type=edge_type)
             else:
                  # This case indicates an issue with node parsing or mapping
                  print(f"Warning: Node index {src_idx} or {dest_idx} not found in graph G when adding edge {src_id_str}->{dest_id_str}.")
        else:
             print(f"Warning: Could not map edge tokens to node indices: {src_id_str} or {dest_id_str}. Skipping edge.")

    return G

# def seq_to_mol(seq_str):
#     tokens = seq_str.split()
#     mol = Chem.RWMol()
#
#     ctx_start = tokens.index('<boc>') + 1
#     ctx_end = tokens.index('<eoc>')
#     ctx_tokens = tokens[ctx_start:ctx_end+1]
#
#     id_atom_lookup = {}
#     for i in range(0, len(ctx_tokens), 3):
#         atom_type = ctx_tokens[i]
#         atom_id = ctx_tokens[i + 1]
#         atomic_symbol = atom_type.split('_')[1]
#         atomic_num = Chem.Atom(atomic_symbol).GetAtomicNum()
#         mol.AddAtom(Chem.Atom(atomic_num))
#         id_atom_lookup[atom_id] = mol.GetNumAtoms() - 1
#
#     # Extract bond tokens
#     bond_start = tokens.index('<bog>') + 1
#     bond_end = tokens.index('<eog>')
#     bond_tokens = [token for token in tokens[bond_start:bond_end] if token != '<sepg>']
#
#     for i in range(0, len(bond_tokens), 3):
#         src_id = bond_tokens[i]
#         dest_id = bond_tokens[i + 1]
#         bond_type = bond_tokens[i + 2]
#         bond_type_rdkit = {
#             'BOND_SINGLE': rdchem.BondType.SINGLE,
#             'BOND_DOUBLE': rdchem.BondType.DOUBLE,
#             'BOND_TRIPLE': rdchem.BondType.TRIPLE,
#             'BOND_AROMATIC': rdchem.BondType.AROMATIC
#         }[bond_type]
#
#         if src_id in id_atom_lookup and dest_id in id_atom_lookup:
#             mol.AddBond(id_atom_lookup[src_id], id_atom_lookup[dest_id], bond_type_rdkit)
#
#     return mol
#
# def seq_to_molecule_with_partial_charges(seq_str):
#     ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
#
#     tokens = seq_str.split()
#     mol = Chem.RWMol()
#
#     ctx_start = tokens.index('<boc>') + 1
#     ctx_end = tokens.index('<eoc>')
#     ctx_tokens = tokens[ctx_start:ctx_end+1]
#
#     id_atom_lookup = {}
#     for i in range(0, len(ctx_tokens), 3):
#         atom_type = ctx_tokens[i]
#         atom_id = ctx_tokens[i + 1]
#         atomic_symbol = atom_type.split('_')[1]
#         atomic_num = Chem.Atom(atomic_symbol).GetAtomicNum()
#         mol.AddAtom(Chem.Atom(atomic_num))
#         id_atom_lookup[atom_id] = mol.GetNumAtoms() - 1
#
#     # Extract bond tokens
#     bond_start = tokens.index('<bog>') + 1
#     bond_end = tokens.index('<eog>')
#     bond_tokens = [token for token in tokens[bond_start:bond_end] if token != '<sepg>']
#
#     for i in range(0, len(bond_tokens), 3):
#         src_id = bond_tokens[i]
#         dest_id = bond_tokens[i + 1]
#         bond_type = bond_tokens[i + 2]
#         bond_type_rdkit = {
#             'BOND_SINGLE': rdchem.BondType.SINGLE,
#             'BOND_DOUBLE': rdchem.BondType.DOUBLE,
#             'BOND_TRIPLE': rdchem.BondType.TRIPLE,
#             'BOND_AROMATIC': rdchem.BondType.AROMATIC
#         }[bond_type]
#
#         if src_id in id_atom_lookup and dest_id in id_atom_lookup:
#             mol.AddBond(id_atom_lookup[src_id], id_atom_lookup[dest_id], bond_type_rdkit)
#             flag, atomid_valence = check_valency(mol)
#             if flag:
#                 continue
#             else:
#                 assert len(atomid_valence) == 2
#                 idx = atomid_valence[0]
#                 v = atomid_valence[1]
#                 an = mol.GetAtomWithIdx(idx).GetAtomicNum()
#                 if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
#                     mol.GetAtomWithIdx(idx).SetFormalCharge(1)
#     return mol
#
# class LobsterDataset(Dataset):
#     def __init__(self, num_data, process_fn=lambda x: x, min_node = 10, max_node=100):
#         self.num_data = num_data
#         self.min_node = min_node
#         self.max_node = max_node
#         self.process_fn = process_fn
#         self.indices = torch.randperm(num_data)
#     def __len__(self):
#         return self.num_data
#
#     def __getitem__(self, idx):
#         if idx == len(self):
#             raise IndexError
#         while True:
#             G = nx.random_lobster(int((self.min_node+self.max_node)/2), 0.7, 0.7)
#             if len(G.nodes()) >= self.min_node and len(G.nodes()) <= self.max_node:
#                 break
#         pyg = from_networkx(G)
#         X = torch.ones(pyg.num_nodes, 1, dtype=torch.float)
#         edge_attr = torch.zeros(pyg.edge_index.shape[-1], 2, dtype=torch.float)
#         return self.process_fn({'x': X, 'edge_index': pyg.edge_index, 'edge_attr': edge_attr})

# Add to or ensure these imports are at the top of G2PT/datasets_utils.py
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class NumpyBinDataset(Dataset):
    """
    Loads graph data preprocessed into numpy memmap files (.bin).
    Modified __getitem__ to return integer vocabulary IDs.
    """
    def __init__(self, path, num_data, num_node_class, num_edge_class, shape, process_fn=lambda x: x):
        self.path = path
        self.num_data = num_data
        # num_node_class and num_edge_class are not strictly needed when returning integer IDs,
        # but kept for potential future use or consistency.
        self.num_node_class = num_node_class
        self.num_edge_class = num_edge_class

        self.process_fn = process_fn # This will be the pre_tokenize_function

        # Ensure shapes are tuples for memmap
        shape['xs'] = tuple(shape['xs'])
        shape['edge_index'] = tuple(shape['edge_index'])
        shape['edge_attr'] = tuple(shape['edge_attr'])

        # --- Load Memory Mapped Files ---
        try:
            xs_path = os.path.join(path, 'xs.bin')
            edge_indices_path = os.path.join(path, 'edge_indices.bin')
            edge_attrs_path = os.path.join(path, 'edge_attrs.bin')

            self.xs = np.memmap(xs_path, dtype=np.int16, mode='r', shape=shape['xs'])
            self.edge_indices = np.memmap(edge_indices_path, dtype=np.int16, mode='r', shape=shape['edge_index'])
            self.edge_attrs = np.memmap(edge_attrs_path, dtype=np.int16, mode='r', shape=shape['edge_attr'])
        except FileNotFoundError as e:
             raise FileNotFoundError(f"Error opening memmap files in {path}. Did prepare_aig.py run correctly and save files to this location? Details: {e}")
        except Exception as e:
             raise RuntimeError(f"Error setting up memmap in {path} with shapes {shape}. Details: {e}")

        # self.indices = torch.randperm(num_data) # Shuffling usually handled by DataLoader sampler

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # Load raw int16 data from memmap files for the given index
        try:
            raw_x = np.array(self.xs[idx]).astype(np.int64)
            raw_edge_index = np.array(self.edge_indices[idx]).astype(np.int64)
            raw_edge_attr = np.array(self.edge_attrs[idx]).astype(np.int64)
        except IndexError:
             print(f"Error: Index {idx} out of bounds for memmap arrays (num_data={self.num_data}).")
             # Return an empty dict for process_fn to handle gracefully
             return self.process_fn({'x': torch.tensor([], dtype=torch.long),
                                     'edge_index': torch.tensor([[],[]], dtype=torch.long),
                                     'edge_attr': torch.tensor([], dtype=torch.long)})
        except Exception as e:
            print(f"Error accessing memmap data at index {idx}: {e}")
            return self.process_fn({'x': torch.tensor([], dtype=torch.long),
                                    'edge_index': torch.tensor([[],[]], dtype=torch.long),
                                    'edge_attr': torch.tensor([], dtype=torch.long)})

        # Filter padding (-100) from node features
        node_padding_mask = raw_x != -100
        x_ids = torch.from_numpy(raw_x[node_padding_mask])
        num_valid_nodes = len(x_ids)

        if num_valid_nodes == 0:
             # Handle graphs with no valid nodes after filtering padding
             return self.process_fn({'x': x_ids,
                                     'edge_index': torch.tensor([[],[]], dtype=torch.long),
                                     'edge_attr': torch.tensor([], dtype=torch.long)})

        # Filter edge attributes based on padding
        edge_padding_mask = raw_edge_attr != -100
        edge_attr_ids_filtered_by_attr = torch.from_numpy(raw_edge_attr[edge_padding_mask])

        # Filter edge indices based *only* on the edge attribute padding mask initially
        # raw_edge_index has shape [2, max_num_edges]
        # edge_padding_mask has shape [max_num_edges]
        edge_index_filtered_by_attr = torch.from_numpy(raw_edge_index[:, edge_padding_mask])

        # Further filter edges where BOTH source and destination nodes are valid
        # Node indices should be between 0 and num_valid_nodes - 1
        if edge_index_filtered_by_attr.numel() > 0: # Check if there are any edges left
            src_nodes = edge_index_filtered_by_attr[0, :]
            dst_nodes = edge_index_filtered_by_attr[1, :]
            node_indices_valid_mask = (src_nodes < num_valid_nodes) & (dst_nodes < num_valid_nodes) & \
                                      (src_nodes >= 0) & (dst_nodes >= 0)

            # Apply the node validity mask to get final edge indices and attributes
            edge_index_final = edge_index_filtered_by_attr[:, node_indices_valid_mask]
            edge_attr_final = edge_attr_ids_filtered_by_attr[node_indices_valid_mask]
        else:
            # No edges were valid based on attribute padding
            edge_index_final = torch.tensor([[],[]], dtype=torch.long)
            edge_attr_final = torch.tensor([], dtype=torch.long)


        # Return dictionary containing tensors with INTEGER vocabulary IDs
        data_dict = {'x': x_ids, 'edge_index': edge_index_final, 'edge_attr': edge_attr_final}

        # Pass the dictionary with integer IDs to the processing function (pre_tokenize_function)
        return self.process_fn(data_dict)


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
    if len(edge_attr.shape) == 2:# one hot
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

    return  edges_bfs

def to_seq_by_bfs(data, atom_type, bond_type):
    
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)
    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}'] for node_idx, node_type in enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []
    
    G = to_networkx(data)

    #get edge order from dfs,begin from node 0, G is nx graph
    # _,edges_order_dfs = dfs_with_all_edges(G,0)
    edges_order_bfs = bfs_with_all_edges(G,0)
    for selected_source_node_idx, selected_dest_node_idx in edges_order_bfs:
        #get_edge_attr
        edge_mask = ((data.edge_index[0] == selected_source_node_idx) & (data.edge_index[1] == selected_dest_node_idx)) | \
            ((data.edge_index[0] == selected_dest_node_idx) & (data.edge_index[1] == selected_source_node_idx))  
        edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        if len(edge_indices) > 0:
            removed_edge_type = data.edge_attr[edge_indices][0].argmax().item()
        outputs.append(['<sepg>', f'IDX_{selected_source_node_idx}', f'IDX_{selected_dest_node_idx}', bond_type[removed_edge_type-1]])

    ctx[0] = '<boc>'
    ctx.append('<eoc>')
    outputs = sum(outputs,[])
    outputs[0] = '<bog>'
    outputs.append('<eog>')
    
    return {"text": [" ".join(ctx + outputs)]}

def to_seq_by_deg(data, atom_type, bond_type):
    
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)
    num_nodes = x.shape[0]

    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}'] for node_idx, node_type in enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])
    data_t = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []
    INF = 100
    while True:
        source_nodes_t = data_t.edge_index[0]
        node_degrees_t = degree(source_nodes_t, num_nodes=num_nodes)
        if torch.all(node_degrees_t==0):
            break
        node_degrees_t[node_degrees_t==0] = INF
        # sample a source node with minimum deg
        candidate_source_nodes = torch.where(node_degrees_t==node_degrees_t.min())[0]
        selected_index = torch.randint(0, candidate_source_nodes.shape[0], (1,)).item()
        selected_source_node_idx = candidate_source_nodes[selected_index].item()
        
        # get the dest node with minimum deg 
        source_node_mask = source_nodes_t==selected_source_node_idx
        candidate_dest_nodes = data_t.edge_index[1][source_node_mask].unique()
        
        candidate_dest_degrees = node_degrees_t[candidate_dest_nodes]
        min_dest_degree = candidate_dest_degrees.min()
        
        indices = torch.where(candidate_dest_degrees == min_dest_degree)[0]
        selected_index = indices[torch.randint(0, len(indices), (1,)).item()]
        selected_dest_node_idx = candidate_dest_nodes[selected_index].item()

        # get new graph at t-1
        data_tminus1, removed_edge_type = remove_edge_with_attr(data_t, (selected_source_node_idx, selected_dest_node_idx))
        # selected_source_node_type = data.x[selected_source_node_idx].argmax(-1).item()
        # selected_dest_node_type = data.x[selected_dest_node_idx].argmax(-1).item()
        outputs.append(['<sepg>', f'IDX_{selected_source_node_idx}', f'IDX_{selected_dest_node_idx}', bond_type[removed_edge_type-1]])
        data_t = data_tminus1
        
    ctx[0] = '<boc>'
    ctx.append('<eoc>')
    outputs = outputs[::-1]
    outputs = sum(outputs,[])
    outputs[0] = '<bog>'
    outputs.append('<eog>')
    return {"text": [" ".join(ctx + outputs)]}

def get_datasets(dataset_name, tokenizer, order='bfs'):

    # Select the sequence generation function
    if dataset_name == 'aig':
        print(f"Using custom sequence generation logic for AIG dataset.")
        order_function = to_seq_aig_topo
    elif order == 'bfs':
        order_function = to_seq_by_bfs
    elif order == 'deg':
        order_function = to_seq_by_deg
    else:
        raise NotImplementedError(f"Order function {order} is not implemented")

    # Define the tokenization function (process_fn)
    def pre_tokenize_function(examples, atom_type, bond_type):
        # Ensure 'examples' dict contains integer tensors 'x', 'edge_index', 'edge_attr'
        data = order_function(examples, atom_type, bond_type)
        # Tokenize the generated text sequence
        tokenized_data = tokenizer(data['text'], padding='max_length', truncation=True, return_tensors='pt') # Added truncation
        # Ensure tensors are correctly shaped (remove batch dim if tokenizer adds one)
        input_ids = tokenized_data['input_ids'].squeeze(0)
        attention_mask = tokenized_data['attention_mask'].squeeze(0)
        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        # G2PT typically uses the input_ids directly as labels, handle potential shifts if needed
        # For standard causal LM, labels = input_ids, loss calculated internally on shifted logits/labels
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


    # --- Dataset Specific Setups ---
    train_datasets = None
    eval_datasets = None

    # if dataset_name == 'lobster':
    #     ATOM_TYPE = ['NODE']
    #     BOND_TYPE = ['EDGE']
    #     train_datasets = LobsterDataset(num_data=256,
    #                                   process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))
    #     eval_datasets = LobsterDataset(num_data=64,
    #                                  process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))

    if dataset_name == 'aig':
        ATOM_TYPE = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
        BOND_TYPE = ['EDGE_INV', 'EDGE_REG']
        # Adjust this path if your data is elsewhere relative to the script's run location
        data_dir = './datasets/aig'
        meta_path = os.path.join(data_dir, 'data_meta.json')

        try:
            with open(meta_path, 'r') as f:
                data_meta = json.load(f)
            # Ensure shapes are tuples
            train_shape = {k: tuple(v) for k, v in data_meta['train_shape'].items()}
            eval_shape = {k: tuple(v) for k, v in data_meta['eval_shape'].items()}
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {meta_path} not found. Run prepare_aig.py first.")
        except Exception as e:
            raise RuntimeError(f"Error loading or parsing {meta_path}: {e}")

        num_train = train_shape['xs'][0]
        num_eval = eval_shape['xs'][0]
        # Pass the number of distinct node/edge types (classes)
        num_node_classes = len(ATOM_TYPE)
        num_edge_classes = len(BOND_TYPE) + 1 # +1 for padding/no-edge type

        train_datasets = NumpyBinDataset(os.path.join(data_dir, 'train'),
                                         num_train, num_node_classes, num_edge_classes,
                                         shape=train_shape,
                                         process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))
        eval_datasets = NumpyBinDataset(os.path.join(data_dir, 'eval'),
                                        num_eval, num_node_classes, num_edge_classes,
                                        shape=eval_shape,
                                        process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))

    # --- Other Dataset Setups (MOSES, GuacaMol, QM9, Tree, SBM, Planar) ---
    # Include the elif blocks for other datasets exactly as they were in the original file
    elif dataset_name == 'moses':
        # ... (original moses setup: ATOM_TYPE, BOND_TYPE, shapes) ...
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_S', 'ATOM_O', 'ATOM_F', 'ATOM_Cl', 'ATOM_Br', 'ATOM_H']
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC']
        train_shape = {'x': (1419512, 27), 'edge_index': (1419512, 2, 62), 'edge_attr': (1419512, 62)}
        eval_shape = {'x': (156176, 27), 'edge_index': (156176, 2, 62), 'edge_attr': (156176, 62)}
        # Common loading logic applies (see below)
    elif dataset_name == 'guacamol':
        # ... (original guacamol setup) ...
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_O', 'ATOM_F', 'ATOM_B', 'ATOM_Br', 'ATOM_Cl', 'ATOM_I', 'ATOM_P', 'ATOM_S', 'ATOM_Se', 'ATOM_Si']
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC']
        train_shape = {'x': (1118633, 88), 'edge_index': (1118633, 2, 174), 'edge_attr': (1118633, 174)}
        eval_shape = {'x': (69926, 76), 'edge_index': (69926, 2, 158), 'edge_attr': (69926, 158)}
        # Common loading logic applies
    elif dataset_name == 'qm9':
         # ... (original qm9 setup) ...
        ATOM_TYPE = ['ATOM_C', 'ATOM_N', 'ATOM_O', 'ATOM_F']
        BOND_TYPE = ['BOND_SINGLE', 'BOND_DOUBLE', 'BOND_TRIPLE', 'BOND_AROMATIC']
        train_shape = {'x': (97732, 9), 'edge_index': (97732, 2, 28), 'edge_attr': (97732, 28)}
        eval_shape = {'x': (20042, 9), 'edge_index': (20042, 2, 26), 'edge_attr': (20042, 26)}
         # Common loading logic applies
    elif dataset_name == 'tree':
         # ... (original tree setup) ...
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']
        train_shape = {'x': (256, 64), 'edge_index': (256, 2, 126), 'edge_attr': (256, 126)}
        eval_shape = {'x': (64, 64), 'edge_index': (64, 2, 126), 'edge_attr': (64, 126)}
         # Common loading logic applies
    elif dataset_name == 'sbm':
         # ... (original sbm setup) ...
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']
        train_shape = {'x': (256, 187), 'edge_index': (256, 2, 2258), 'edge_attr': (256, 2258)}
        eval_shape = {'x': (64, 172), 'edge_index': (64, 2, 1808), 'edge_attr': (64, 1808)}
         # Common loading logic applies
    elif dataset_name == 'planar':
         # ... (original planar setup) ...
        ATOM_TYPE = ['NODE']
        BOND_TYPE = ['EDGE']
        train_shape = {'x': (256, 64), 'edge_index': (256, 2, 362), 'edge_attr': (256, 362)}
        eval_shape = {'x': (64, 64), 'edge_index': (64, 2, 362), 'edge_attr': (64, 362)}
         # Common loading logic applies
    else:
        # If dataset name is not recognized after all checks
        if train_datasets is None: # Check if dataset wasn't handled
             raise NotImplementedError(f"Dataset {dataset_name} is not implemented or handled.")


    # --- Common Loading Logic for NumpyBinDataset (if not handled already) ---
    if dataset_name not in ['lobster', 'aig']: # Check if datasets were already created
        num_train = train_shape['xs'][0]
        num_eval = eval_shape['xs'][0]
        num_node_classes = len(ATOM_TYPE)
        num_edge_classes = len(BOND_TYPE) + 1

        # Instantiate datasets using the common logic
        train_datasets = NumpyBinDataset(f'./datasets/{dataset_name}/train', # Adjust path as needed
                                         num_train, num_node_classes, num_edge_classes,
                                         shape=train_shape,
                                         process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))
        eval_datasets = NumpyBinDataset(f'./datasets/{dataset_name}/eval', # Adjust path as needed
                                        num_eval, num_node_classes, num_edge_classes,
                                        shape=eval_shape,
                                        process_fn=partial(pre_tokenize_function, atom_type=ATOM_TYPE, bond_type=BOND_TYPE))

    # Final check before returning
    if train_datasets is None or eval_datasets is None:
        raise RuntimeError(f"Failed to initialize datasets for {dataset_name}")

    return train_datasets, eval_datasets
