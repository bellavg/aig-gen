# -*- coding: utf-8 -*-
import os
import pickle
import torch
import itertools
from torch_geometric.data import Data
import argparse
import numpy
import torch.nn.functional as F
from torch_geometric.utils.convert import to_networkx, from_networkx
import networkx as nx
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_undirected)
import glob

# --- Imports from external libraries ---
# These should be installed in your environment (e.g., pip install aigverse torch_geometric networkx)
try:
    from aigverse import read_aiger_into_aig, to_edge_list
except ImportError:
    print("Error: 'aigverse' library not found. Please install it.")
    exit(1)


# --- Utility functions for graph preprocessing (from your add_order_info implementation) ---

def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """Implement different eigenvector normalizations."""
    EigVals = EigVals.unsqueeze(0)
    if normalization == "L1":
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)
    elif normalization == "L2":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
    elif normalization == "abs-max":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")
    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom
    return EigVecs

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats."""
    N = len(evals)
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], numpy.real(evects[:, idx])
    evals = torch.from_numpy(numpy.real(evals)).clamp_min(0)
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects
    return EigVecs

def top_sort(edge_index, graph_size):
    """Performs a topological sort on the nodes of a graph."""
    node_ids = numpy.arange(graph_size, dtype=int)
    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)
    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]
    n = 0
    while unevaluated_nodes.any():
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        unready_children = child_nodes[unevaluated_mask]
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)
        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False
        n += 1
    return torch.from_numpy(node_order).long()

def add_order_info(graph):
    """
    Adds topological and spectral information to the PyG Data object.
    This includes positional encodings (abs_pe) and Laplacian eigenvectors.
    """
    num_nodes = graph.num_nodes
    
    pe = top_sort(graph.edge_index, num_nodes)
    graph.abs_pe = pe

    undir_edge_index = to_undirected(graph.edge_index)
    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=None, num_nodes=num_nodes)
    )
    evals, evects = numpy.linalg.eigh(L.toarray())
    graph.Eigvecs = get_lap_decomp_stats(
        evals=evals, evects=evects,
        max_freqs=8,
        eigvec_norm='L2'
    )

    data_for_nx = Data(edge_index=graph.edge_index, num_nodes=num_nodes)
    DG = to_networkx(data_for_nx, to_undirected=False)
    TC = nx.transitive_closure_dag(DG)
    edge_index_dag = from_networkx(TC).edge_index
    graph.dag_rr_edge_index = to_undirected(edge_index_dag)

    max_num_nodes = 8
    mask_rc = torch.zeros(max_num_nodes, max_num_nodes, dtype=torch.bool)
    if num_nodes <= max_num_nodes:
        for i in range(num_nodes):
            successors = edge_index_dag[1, edge_index_dag[0] == i]
            predecessors = edge_index_dag[0, edge_index_dag[1] == i]
            mask_r = torch.zeros(max_num_nodes, dtype=torch.bool)
            if successors.numel() > 0: mask_r[successors] = True
            if predecessors.numel() > 0: mask_r[predecessors] = True
            mask_rc[i] = ~mask_r
    graph.mask_rc = mask_rc
    
    return graph


# --- Placeholder for Target Vector Calculation ---
def get_structural_difference_vector(aig1, aig2):
    """
    !!! IMPORTANT PLACEHOLDER !!!
    You must replace this function with your actual implementation.
    """
    aig1_name = getattr(aig1, 'name', 'unknown_aig1')
    aig2_name = getattr(aig2, 'name', 'unknown_aig2')
    print(f"      (Placeholder) Calculating structural vector between {os.path.basename(aig1_name)} and {os.path.basename(aig2_name)}")
    return torch.randn(1, 16)


# --- Configuration ---
NODE_TYPE_ENCODING = {
    'CONST0': [1, 0, 0, 0], 'PI': [0, 1, 0, 0],
    'AND':    [0, 0, 1, 0], 'PO': [0, 0, 0, 1]
}
EDGE_TYPE_ENCODING = {
    'REGULAR':  [1, 0], 'INVERTED': [0, 1]
}

# --- Core Data Processing Function ---
def create_aig_pyg_data(aig):
    """
    Converts a single AIG object from aigverse into a torch_geometric.data.Data object.
    """
    try:
        node_mapping = {}
        def add_to_mapping(node_id):
            if node_id not in node_mapping:
                node_mapping[node_id] = len(node_mapping)

        add_to_mapping(0)
        for pi_id in aig.pis(): add_to_mapping(pi_id)
        for gate_id in aig.gates(): add_to_mapping(gate_id)

        node_features = [None] * len(node_mapping)
        node_features[node_mapping[0]] = NODE_TYPE_ENCODING['CONST0']
        for pi_id in aig.pis(): node_features[node_mapping[pi_id]] = NODE_TYPE_ENCODING['PI']
        for gate_id in aig.gates(): node_features[node_mapping[gate_id]] = NODE_TYPE_ENCODING['AND']
        
        po_start_idx = len(node_mapping)
        source_nodes, target_nodes, edge_features = [], [], []

        for i, po_literal in enumerate(aig.pos()):
            po_idx = po_start_idx + i
            node_features.append(NODE_TYPE_ENCODING['PO'])
            driver_node_id = aig.get_node(po_literal)
            if driver_node_id in node_mapping:
                source_nodes.append(node_mapping[driver_node_id])
                target_nodes.append(po_idx)
                is_inverted = aig.is_complemented(po_literal)
                edge_features.append(EDGE_TYPE_ENCODING['INVERTED'] if is_inverted else EDGE_TYPE_ENCODING['REGULAR'])

        x_tensor = torch.tensor(node_features, dtype=torch.float)

        for edge in to_edge_list(aig):
            if edge.source in node_mapping and edge.target in node_mapping:
                source_nodes.append(node_mapping[edge.source])
                target_nodes.append(node_mapping[edge.target])
                is_inverted = (edge.weight == 1)
                edge_features.append(EDGE_TYPE_ENCODING['INVERTED'] if is_inverted else EDGE_TYPE_ENCODING['REGULAR'])
        
        edge_index_tensor = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)

        pyg_data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
        pyg_data.aig_name = getattr(aig, 'name', 'unnamed_aig')
        
        pyg_data = add_order_info(pyg_data)
        return pyg_data
    except Exception as e:
        aig_name = getattr(aig, 'name', 'unnamed_aig')
        print(f"      [Error] Failed to process AIG: {aig_name}. Reason: {e}")
        return None

def _save_data_chunk(data_list, output_path):
    """Saves a list of data pairs to a pickle file."""
    if not data_list:
        return
    print(f"    Saving chunk of {len(data_list)} pairs to '{os.path.basename(output_path)}'...")
    with open(output_path, 'wb') as f:
        pickle.dump(data_list, f)

# --- Main Siamese Pair Processing Logic ---
def process_aig_pairs_directory(base_dir, output_file, chunk_size, use_chunking):
    """
    Processes all subdirectories in a base directory to create AIG pairs.
    """
    output_dir = os.path.dirname(output_file)
    base_filename, _ = os.path.splitext(os.path.basename(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    siamese_dataset = []
    chunk_count = 1
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    
    print(f"Found {len(subfolders)} subfolders. Chunking enabled: {use_chunking} (size: {chunk_size})")

    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        print(f"\n--- Processing subfolder: {folder_name} ---")
        
        aig_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.aig', '.aag'))]
        if len(aig_files) < 2:
            print(f"  -> Skipping '{folder_name}', found fewer than 2 AIG files.")
            continue
            
        print(f"  Found {len(aig_files)} AIGs. Creating pairs...")
        
        for file1, file2 in itertools.combinations(aig_files, 2):
            path1, path2 = os.path.join(folder_path, file1), os.path.join(folder_path, file2)
            print(f"  - Pairing '{file1}' and '{file2}'")

            try:
                aig1, aig2 = read_aiger_into_aig(path1), read_aiger_into_aig(path2)
                aig1.name, aig2.name = path1, path2

                pyg_data1, pyg_data2 = create_aig_pyg_data(aig1), create_aig_pyg_data(aig2)

                if not pyg_data1 or not pyg_data2:
                    print(f"    -> Skipping pair due to processing error.")
                    continue

                siamese_dataset.append({'graph1': pyg_data1, 'graph2': pyg_data2, 'y': get_structural_difference_vector(aig1, aig2)})
                siamese_dataset.append({'graph1': pyg_data2, 'graph2': pyg_data1, 'y': get_structural_difference_vector(aig2, aig1)})

                if use_chunking and len(siamese_dataset) >= chunk_size:
                    chunk_filename = f"{base_filename}_part_{chunk_count}.pkl"
                    _save_data_chunk(siamese_dataset, os.path.join(output_dir, chunk_filename))
                    siamese_dataset = []
                    chunk_count += 1

            except Exception as e:
                print(f"    -> [Critical Error] Could not process pair {file1}-{file2}. Error: {e}")

    # Save any remaining data
    if siamese_dataset:
        if use_chunking:
            chunk_filename = f"{base_filename}_part_{chunk_count}.pkl"
            _save_data_chunk(siamese_dataset, os.path.join(output_dir, chunk_filename))
        else: # No chunking, save all at once
             _save_data_chunk(siamese_dataset, output_file)

    print(f"\n--- Processing Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process paired AIG files for a Siamese network.")
    parser.add_argument('--base-input-dir', type=str, default="./sample_aig_sets", 
                        help="Base directory containing subfolders of functionally equivalent AIGs.")
    parser.add_argument('--output-file', type=str, default="./processed_siamese/aig_siamese_dataset.pkl", 
                        help="Path for the output pickle file (used as a base name for chunks).")
    parser.add_argument('--chunk-size', type=int, default=5000, help="Number of graph pairs per chunk file.")
    parser.add_argument('--no-chunking', action='store_true', help="Disable chunking and save all data to a single file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.base_input_dir):
        print(f"Creating dummy directory '{args.base_input_dir}' for demonstration.")
        os.makedirs(os.path.join(args.base_input_dir, "adder_designs"))
        os.makedirs(os.path.join(args.base_input_dir, "multiplier_designs"))
        for i in range(3):
            with open(os.path.join(args.base_input_dir, "adder_designs", f"adder_v{i+1}.aag"), "w") as f: f.write("aag 1 1 0 0 1\n2\n")
        for i in range(2):
            with open(os.path.join(args.base_input_dir, "multiplier_designs", f"mult_v{i+1}.aag"), "w") as f: f.write("aag 1 1 0 0 1\n2\n")
        print("Created dummy files for demonstration.")
    
    process_aig_pairs_directory(
        args.base_input_dir, 
        args.output_file,
        chunk_size=args.chunk_size,
        use_chunking=not args.no_chunking
    )

    print("\n--- To load the processed dataset in your training script, use: ---")
    output_dir = os.path.dirname(args.output_file)
    base_filename, _ = os.path.splitext(os.path.basename(args.output_file))
    
    if not args.no_chunking:
        print("import os, pickle, glob")
        print(f"data_files = sorted(glob.glob(os.path.join('{output_dir}', '{base_filename}_part_*.pkl')))")
        print("all_data = []")
        print("for file in data_files:")
        print("    with open(file, 'rb') as f:")
        print("        all_data.extend(pickle.load(f))")
        print("if all_data: print(f'Loaded {len(all_data)} pairs from {len(data_files)} files.')")
    else:
        print("import pickle")
        print(f"if os.path.exists('{args.output_file}'):")
        print(f"    with open('{args.output_file}', 'rb') as f:")
        print("        all_data = pickle.load(f)")
        print("    if all_data: print(f'Loaded {len(all_data)} pairs.')")
