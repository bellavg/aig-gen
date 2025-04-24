# G2PT/datasets/aig_pkl_to_pyg_with_all_tt_counts_v5.py # <-- Version bumped
# Stage 1: Convert NetworkX AIGs from .pkl to PyG Data objects,
# storing ALL provided truth table types and input/output counts.

import os
import pickle
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
import warnings
from types import SimpleNamespace


# --- Configuration ---
CFG = SimpleNamespace(
    # --- Paths ---
    input_pickle_path='epfl_aigs.pkl',     # Path to your input AIG pickle file
    output_pyg_dir='real_aigs/',             # Base directory for PyG dataset output

    # --- Data Handling ---
    split_ratios=(0.7, 0.15, 0.15),
    # *** Key names for graph attributes to fetch ***
    truth_table_attribute_key = 'tts',                # Input/Output TTs (0 for AND)
    truth_table_with_ssim_attribute_key = 'tts_with_ss', # Input/Output/SSIM TTs
    output_patterns_attribute_key='output_patterns',  # Output Patterns only TTs
    num_inputs_attribute_key='inputs',
    num_outputs_attribute_key='outputs',

    # --- Feature Mappings (Node) ---
    node_type_map={
        tuple([0, 0, 0]): [1.0, 0.0, 0.0, 0.0], # CONST0 -> Index 0
        tuple([1, 0, 0]): [0.0, 1.0, 0.0, 0.0], # PI -> Index 1
        tuple([0, 1, 0]): [0.0, 0.0, 1.0, 0.0], # AND -> Index 2
        tuple([0, 0, 1]): [0.0, 0.0, 0.0, 1.0], # PO -> Index 3
    },
    default_node_feature=[0.0, 0.0, 0.0, 0.0],
    num_node_features=4,

    # --- Feature Mappings (Edge) ---
    edge_type_map={
        tuple([1, 0]): [1.0, 0.0], # INV -> Index 0
        tuple([0, 1]): [0.0, 1.0], # REG -> Index 1
    },
    default_edge_feature=[0.0, 1.0],
    num_edge_features=2,
)

# --- Main Conversion Logic ---
if __name__ == '__main__':
    print(f"--- Stage 1: AIG .pkl to PyG .pt Conversion (All TT Types & Counts from Graph Attributes) ---")
    print(f"Input : {CFG.input_pickle_path}")
    print(f"Output Dir: {CFG.output_pyg_dir}")
    print(f"Expecting TT attributes: '{CFG.truth_table_attribute_key}', '{CFG.truth_table_with_ssim_attribute_key}', '{CFG.output_patterns_attribute_key}'")
    print(f"Expecting count attributes: '{CFG.num_inputs_attribute_key}', '{CFG.num_outputs_attribute_key}'")


    # --- Load Data ---
    try:
        with open(CFG.input_pickle_path, 'rb') as f:
            all_nx_graphs = pickle.load(f)
        print(f"Loaded {len(all_nx_graphs)} graphs from pickle.")
    except FileNotFoundError:
        print(f"Error: Input pickle file not found at {CFG.input_pickle_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading pickle file '{CFG.input_pickle_path}': {e}")
        exit(1)

    if not isinstance(all_nx_graphs, list):
        if isinstance(all_nx_graphs, nx.DiGraph):
             print("Warning: Loaded a single graph object, not a list. Processing it.")
             all_nx_graphs = [all_nx_graphs]
        else:
             print(f"Error: Expected a list of graphs, got {type(all_nx_graphs)}")
             exit(1)

    # Filter and Convert to PyG Data objects
    all_pyg_data = []
    skipped_items = 0
    missing_attr_count = 0
    missing_some_tt_count = 0
    print("Converting Graphs (reading attributes) to PyG Data objects...")

    for nx_graph in tqdm(all_nx_graphs, desc="Converting"):
        # Basic graph validation
        if not isinstance(nx_graph, nx.DiGraph):
            warnings.warn(f"Skipping item - not a NetworkX DiGraph.")
            skipped_items += 1
            continue
        if nx_graph.number_of_nodes() == 0:
            warnings.warn(f"Skipping item - graph has no nodes.")
            skipped_items += 1
            continue

        # *** MODIFICATION: Extract ALL THREE TT types, input count, and output count ***
        tt_io = nx_graph.graph.get(CFG.truth_table_attribute_key)
        tt_ssim = nx_graph.graph.get(CFG.truth_table_with_ssim_attribute_key)
        tt_out = nx_graph.graph.get(CFG.output_patterns_attribute_key)
        num_inputs = nx_graph.graph.get(CFG.num_inputs_attribute_key)
        num_outputs = nx_graph.graph.get(CFG.num_outputs_attribute_key)

        # Check if essential non-TT attributes are missing
        if num_inputs is None or num_outputs is None:
            missing = []
            if num_inputs is None: missing.append(f"'{CFG.num_inputs_attribute_key}'")
            if num_outputs is None: missing.append(f"'{CFG.num_outputs_attribute_key}'")
            warnings.warn(f"Skipping graph - missing essential count attributes: {', '.join(missing)}.")
            missing_attr_count += 1
            skipped_items += 1
            continue

        # Check if ALL TT attributes are missing (optional, could allow graphs with no TTs)
        if tt_io is None and tt_ssim is None and tt_out is None:
            warnings.warn(f"Skipping graph - missing ALL expected truth table attributes.")
            missing_attr_count += 1 # Count as missing required attributes
            skipped_items += 1
            continue
        # Warn if *some* TTs are missing, but proceed storing None for missing ones
        elif tt_io is None or tt_ssim is None or tt_out is None:
             missing_tts = []
             if tt_io is None: missing_tts.append(f"'{CFG.truth_table_attribute_key}'")
             if tt_ssim is None: missing_tts.append(f"'{CFG.truth_table_with_ssim_attribute_key}'")
             if tt_out is None: missing_tts.append(f"'{CFG.output_patterns_attribute_key}'")
             warnings.warn(f"Graph included, but missing some TT attributes: {', '.join(missing_tts)}. Storing None for these.")
             missing_some_tt_count += 1
        # --- End Attribute Fetching Modification ---


        # --- Manual Conversion from NetworkX DiGraph to PyG Components ---
        # (This part remains the same)
        node_list = list(nx_graph.nodes())
        node_map = {node_id: i for i, node_id in enumerate(node_list)}

        x_features = []
        valid_graph = True
        for node_id in node_list:
            nx_attrs = nx_graph.nodes[node_id]
            type_tuple = tuple(nx_attrs.get('type', []))
            feature = CFG.node_type_map.get(type_tuple)
            if feature is None:
                 warnings.warn(f"Graph skipped (node check): Node {node_id} unknown 'type' (value: {nx_attrs.get('type')}).")
                 valid_graph = False
                 break
            x_features.append(feature)
        if not valid_graph:
            skipped_items += 1
            continue

        edge_indices_list = []
        edge_attrs_list = []
        for u, v, nx_edge_data in nx_graph.edges(data=True):
            src_idx = node_map.get(u)
            dst_idx = node_map.get(v)
            if src_idx is None or dst_idx is None:
                 warnings.warn(f"Graph skipped (edge check): Edge ({u},{v}) has node not in map.")
                 valid_graph = False
                 break
            edge_type_tuple = tuple(nx_edge_data.get('type', CFG.default_edge_feature))
            edge_feature = CFG.edge_type_map.get(edge_type_tuple)
            if edge_feature is None:
                warnings.warn(f"Edge ({u},{v}) unknown 'type' {edge_type_tuple}. Using default.")
                edge_feature = CFG.default_edge_feature
            edge_indices_list.append([src_idx, dst_idx])
            edge_attrs_list.append(edge_feature)
        if not valid_graph:
            skipped_items += 1
            continue
        # --- End Manual Conversion ---


        # --- Create PyG Data object ---
        # *** MODIFICATION: Store each TT type separately ***
        data = Data(
            x=torch.tensor(x_features, dtype=torch.float).reshape(-1, CFG.num_node_features),
            edge_index=torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous() if edge_indices_list else torch.empty((2,0), dtype=torch.long),
            edge_attr=torch.tensor(edge_attrs_list, dtype=torch.float).reshape(-1, CFG.num_edge_features) if edge_attrs_list else torch.empty((0, CFG.num_edge_features), dtype=torch.float),
            y=torch.zeros((1, 0), dtype=torch.float), # Placeholder y
            # Store each truth table type (will be None if missing in the source graph)
            tts=tt_io,
            tts_with_ss=tt_ssim,
            output_patterns=tt_out,
            # Store counts
            num_inputs=num_inputs,
            num_outputs=num_outputs
        )
        all_pyg_data.append(data)
    # --- End Conversion Loop ---

    print(f"Successfully converted {len(all_pyg_data)} graphs.")
    if missing_attr_count > 0:
        print(f"Skipped {missing_attr_count} graphs due to missing essential attributes (counts or all TTs).")
    if missing_some_tt_count > 0:
         print(f"Included {missing_some_tt_count} graphs that were missing *some* TT attributes (stored as None).")
    other_skipped = skipped_items - missing_attr_count
    if other_skipped > 0:
        print(f"Skipped an additional {other_skipped} items due to other errors or invalid format.")
    if not all_pyg_data:
        print("Error: No graphs were converted successfully. Exiting.")
        exit(1)

    # --- Shuffle and Split ---
    # (Remains the same)
    print(f"Shuffling {len(all_pyg_data)} converted PyG Data objects...")
    np.random.shuffle(all_pyg_data)
    num_graphs = len(all_pyg_data)
    num_train = int(num_graphs * CFG.split_ratios[0])
    num_val = int(num_graphs * CFG.split_ratios[1])
    if num_train + num_val >= num_graphs:
        num_val = max(0, num_graphs - num_train)
        num_test = 0
    else:
        num_test = num_graphs - num_train - num_val

    datasets = {
        'train': all_pyg_data[:num_train],
        'val': all_pyg_data[num_train : num_train + num_val],
        'test': all_pyg_data[num_train + num_val :]
    }
    print(f"Split sizes - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

    # --- Save Splits ---
    # (Remains the same)
    output_raw_dir = os.path.join(CFG.output_pyg_dir, 'raw')
    os.makedirs(output_raw_dir, exist_ok=True)
    print(f"Saving splits to {output_raw_dir}...")

    for split_name, data_list in datasets.items():
        if not data_list and split_name != 'test': # Allow empty test split
             print(f"Warning: Split '{split_name}' is empty.")
             save_path = os.path.join(output_raw_dir, f'{split_name}.pt')
             torch.save([], save_path)
             print(f"Saved empty {split_name} split file to {save_path}")
             continue

        save_path = os.path.join(output_raw_dir, f'{split_name}.pt')
        try:
            torch.save(data_list, save_path)
            print(f"Saved {split_name} split ({len(data_list)} graphs) to {save_path}")
        except Exception as e:
            print(f"Error saving {split_name} split to {save_path}: {e}")

    print("--- Stage 1: Conversion to PyG .pt files (with All TT Types & Counts) finished. ---")
    print(f"PyG dataset raw files saved in: {output_raw_dir}")
    print("Next steps:")
    print("1. Update AIGPygDataset/AIGPygDataModule if needed to handle 'tts', 'tts_with_ss', 'output_patterns'.")
    print("2. Modify prepare_aig.py (or datasets_utils.py) to read these TT attributes and 'num_inputs', 'num_outputs' and generate the final sequence string, potentially prefixing context tokens based on these values.")