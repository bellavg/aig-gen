# G2PT/check_aig_to_seq_conv.py (MODIFIED TO TEST AUGMENTATION)
import os
import argparse
import torch
import networkx as nx
import numpy as np
import logging
from collections import defaultdict, Counter
import time # For timing info

# --- Import necessary functions and classes ---
try:
    from datasets_utils import (
        to_seq_by_bfs,
        to_seq_by_deg,
        to_seq_aig_topo,
        seq_to_nxgraph
    )
except ImportError as e:
     print(f"Error importing from datasets_utils: {e}")
     print("Make sure datasets_utils.py is in the correct path and contains all required functions.")
     exit(1)

try:
    from datasets.aig_dataset import AIGPygDataset
except ImportError:
     try: from aig_dataset import AIGPygDataset
     except ImportError: print("Error: Could not find AIGPygDataset."); exit(1)

try:
    from evaluate_aigs import calculate_structural_aig_metrics, VALID_AIG_NODE_TYPES
except ImportError:
    print("Warning: Could not import evaluate_aigs.py. Structural validity checks will be basic.")
    def calculate_structural_aig_metrics(G):
        metrics = {'is_structurally_valid': False, 'constraints_failed': ['Evaluation script not found']}
        if isinstance(G, nx.DiGraph) and G.number_of_nodes() > 0:
             if nx.is_directed_acyclic_graph(G):
                  node_types_present = {data.get('type', 'UNKNOWN') for _, data in G.nodes(data=True)}
                  if all(ntype in VALID_AIG_NODE_TYPES or ntype == 'UNKNOWN' for ntype in node_types_present):
                       metrics['is_structurally_valid'] = True; metrics['constraints_failed'] = []
                  else: metrics['constraints_failed'] = ['Unknown node types found (basic check)']
             else: metrics['constraints_failed'] = ['Not a DAG (basic check)']
        else: metrics['constraints_failed'] = ['Not a DiGraph or empty (basic check)']
        return metrics
    VALID_AIG_NODE_TYPES = {'NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'}


# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger("check_augmentation")

# --- Helper Function (pyg_data_to_nx) ---
# (Remains unchanged from previous version)
def pyg_data_to_nx(data):
    if not hasattr(data, 'x') or data.x is None: return None
    G = nx.DiGraph(); num_nodes = data.x.size(0)
    node_feature_idx_to_type = {0: 'NODE_CONST0', 1: 'NODE_PI', 2: 'NODE_AND', 3: 'NODE_PO'}
    vocab_id_to_feature_index = {97: 0, 98: 1, 99: 2, 100: 3}
    node_types = []; type_indices = []
    if data.x.numel() == 0: return G
    if data.x.dim() > 1 and data.x.shape[1] > 1 and data.x.dtype == torch.float: type_indices = torch.argmax(data.x, dim=1).tolist()
    elif data.x.dim() == 1:
        is_vocab_id = torch.all((data.x >= 97) & (data.x <= 100)).item(); is_feature_idx = torch.all((data.x >= 0) & (data.x <= 3)).item()
        if is_vocab_id: type_indices = [vocab_id_to_feature_index.get(vid.item(), -1) for vid in data.x]
        elif is_feature_idx: type_indices = data.x.tolist()
        else: logger.error(f"Unexpected int values in data.x: {data.x.unique().tolist()}"); return None
    else: logger.error(f"Unexpected shape/type for data.x: {data.x.shape}, dtype: {data.x.dtype}."); return None
    for i in range(num_nodes):
        if i < len(type_indices):
            node_type_idx = int(type_indices[i]); node_type_str = node_feature_idx_to_type.get(node_type_idx, 'UNKNOWN')
            if node_type_str == 'UNKNOWN': logger.warning(f"Unknown node idx {node_type_idx}.")
            G.add_node(i, type=node_type_str); node_types.append(node_type_str)
        else: logger.error(f"Node index mismatch: {num_nodes} vs {len(type_indices)}"); G.add_node(i, type='UNKNOWN')
    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        edge_index = data.edge_index.cpu().numpy(); num_edges = edge_index.shape[1]
        for i in range(num_edges):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u in G and v in G: G.add_edge(u, v)
            else: logger.warning(f"Invalid edge index: ({u}, {v}) for max node {num_nodes-1}. Skipping.")
    return G


# --- Main Function (Modified for Augmentation Testing) ---
def main(args):
    start_time = time.time()
    logger.info(f"Loading processed AIG dataset from: {args.data_dir}")
    try:
        dataset = AIGPygDataset(root=args.data_dir, split=args.split, transform=None)
    except Exception as e: logger.error(f"Failed to load dataset from {args.data_dir}: {e}"); return

    num_to_check = min(args.num_graphs, len(dataset))
    if num_to_check <= 0: logger.warning(f"No graphs in split '{args.split}'. Exiting."); return

    # --- Select Ordering Function ---
    order_function = None; order_name = ""
    if args.ordering == 'bfs': order_function = to_seq_by_bfs; order_name = "BFS"
    elif args.ordering == 'deg': order_function = to_seq_by_deg; order_name = "Degree"
    elif args.ordering == 'topo': order_function = to_seq_aig_topo; order_name = "Topological"
    else: logger.error(f"Unsupported ordering: {args.ordering}"); return

    logger.info(f"Testing {args.num_augmentations_to_test} augmentations per graph using {order_name} ordering.")
    logger.info(f"Checking {num_to_check} graphs from split '{args.split}'...")

    # --- Define Mappings & Type Lists ---
    node_feature_map = {0: 97, 1: 98, 2: 99, 3: 100}; edge_feature_map = {0: 101, 1: 102}
    actual_atom_type_list = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
    actual_bond_type_list = ['EDGE_INV', 'EDGE_REG']

    # --- Initialize counters ---
    conversion_warnings = 0; validity_failures = 0; isomorphism_failures = 0
    successful_checks = 0; processed_count = 0
    total_unique_sequences_generated = 0; graphs_with_multiple_sequences = 0

    # --- Main Loop (Iterate through graphs) ---
    for i in range(num_to_check):
        graph_start_time = time.time()
        try:
            original_pyg_data = dataset[i]
            processed_count += 1

            # --- Preprocessing: Convert PyG -> Dict with Vocab IDs ---
            processed_data_for_seq = {}; fatal_preprocessing_error = False
            # (Preprocessing logic - condensed for brevity, same as before)
            if hasattr(original_pyg_data, 'x') and original_pyg_data.x is not None and original_pyg_data.x.numel()>0:
                node_x = original_pyg_data.x; node_vocab_ids = None
                if node_x.dim()>1: node_feature_indices=node_x.argmax(dim=-1); node_vocab_ids=torch.tensor([node_feature_map.get(idx.item(), -1) for idx in node_feature_indices], dtype=torch.long)
                elif node_x.dim()==1:
                    if torch.all((node_x>=0)&(node_x<=3)).item(): node_vocab_ids=torch.tensor([node_feature_map.get(idx.item(),-1) for idx in node_x],dtype=torch.long)
                    elif torch.all((node_x>=97)&(node_x<=100)).item(): node_vocab_ids=node_x.long()
                    else: node_vocab_ids=torch.tensor([-1]*node_x.shape[0],dtype=torch.long)
                else: node_vocab_ids=torch.tensor([-1]*node_x.shape[0],dtype=torch.long)
                if torch.any(node_vocab_ids == -1): fatal_preprocessing_error = True
                processed_data_for_seq['x'] = node_vocab_ids
            else: fatal_preprocessing_error = True
            if fatal_preprocessing_error: logger.error(f"Graph {i}: Skipping due to node preprocessing error."); validity_failures+=1; continue
            if hasattr(original_pyg_data,'edge_index') and original_pyg_data.edge_index is not None:
                processed_data_for_seq['edge_index'] = original_pyg_data.edge_index
                if hasattr(original_pyg_data,'edge_attr') and original_pyg_data.edge_attr is not None and original_pyg_data.edge_attr.numel()>0:
                    edge_attr=original_pyg_data.edge_attr; edge_vocab_ids = None
                    if edge_attr.dim()>1: edge_feature_indices=edge_attr.argmax(dim=-1); edge_vocab_ids=torch.tensor([edge_feature_map.get(idx.item(),-1) for idx in edge_feature_indices],dtype=torch.long)
                    elif edge_attr.dim()==1:
                         if torch.all((edge_attr>=0)&(edge_attr<=1)).item(): edge_vocab_ids=torch.tensor([edge_feature_map.get(idx.item(),-1) for idx in edge_attr],dtype=torch.long)
                         elif torch.all((edge_attr>=101)&(edge_attr<=102)).item(): edge_vocab_ids=edge_attr.long()
                         else: edge_vocab_ids=torch.tensor([-1]*edge_attr.shape[0],dtype=torch.long)
                    else: edge_vocab_ids=torch.tensor([-1]*edge_attr.shape[0],dtype=torch.long)
                    #if torch.any(edge_vocab_ids == -1): logger.warning(...) # Don't make edge errors fatal here
                    processed_data_for_seq['edge_attr'] = edge_vocab_ids
                else: processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
            else: processed_data_for_seq['edge_index'] = torch.tensor([[],[]],dtype=torch.long); processed_data_for_seq['edge_attr'] = torch.tensor([],dtype=torch.long)
            # --- End Preprocessing ---

            generated_sequences = set()
            all_augmentations_valid = True
            first_reconstructed_graph = None # For isomorphism check

            # --- Inner Loop (Iterate through augmentations) ---
            for aug_idx in range(args.num_augmentations_to_test):
                # --- Step 1: Convert Processed Graph -> Sequence (with aug_seed) ---
                sequence_dict = order_function(
                    processed_data_for_seq,
                    actual_atom_type_list,
                    actual_bond_type_list,
                    aug_seed=aug_idx # Pass augmentation index as seed
                )
                sequence = sequence_dict.get("text", [""])[0]

                if not sequence or sequence == "<boc> <eoc> <bog> <eog>": # Check for empty/failed generation
                    logger.warning(f"Graph {i}, Aug {aug_idx}: Failed to generate valid sequence ({order_name}).")
                    all_augmentations_valid = False; break # Stop checking this graph if sequence fails

                generated_sequences.add(sequence) # Store unique sequences

                # --- Step 2: Convert Sequence -> Reconstructed Graph ---
                reconstructed_graph = None
                try:
                    reconstructed_graph = seq_to_nxgraph(sequence, parsing_mode='robust')
                    if reconstructed_graph is None: raise ValueError("seq_to_nxgraph returned None")
                    if aug_idx == 0: first_reconstructed_graph = reconstructed_graph # Save first for iso check
                except Exception as e:
                    logger.warning(f"Graph {i}, Aug {aug_idx}: Failed to convert {order_name} sequence back to graph. Error: {e}")
                    logger.debug(f"Failed sequence (Graph {i}, Aug {aug_idx}, {order_name}): {sequence[:300]}...")
                    conversion_warnings += 1; all_augmentations_valid = False; break

                # --- Step 3: Check Validity of Reconstructed Graph ---
                validity_metrics = calculate_structural_aig_metrics(reconstructed_graph)
                structurally_valid = validity_metrics.get('is_structurally_valid', False)
                if not structurally_valid:
                    logger.warning(f"Graph {i}, Aug {aug_idx}: Reconstructed graph (from {order_name}) FAILED structural validity checks.")
                    logger.debug(f"Failure reasons: {validity_metrics.get('constraints_failed', ['Unknown'])}")
                    logger.debug(f"Failed sequence (Graph {i}, Aug {aug_idx}, {order_name}): {sequence[:300]}...")
                    all_augmentations_valid = False; break

            # --- After checking all augmentations for graph i ---
            num_unique_sequences = len(generated_sequences)
            total_unique_sequences_generated += num_unique_sequences

            # Log augmentation results for this graph
            log_level = logging.INFO
            if num_unique_sequences == 1 and args.num_augmentations_to_test > 1 and all_augmentations_valid:
                 log_level = logging.WARNING # Warn if only one sequence generated when more expected
                 logger.log(log_level, f"Graph {i}: Only 1 unique sequence generated (expected up to {args.num_augmentations_to_test}). Structure may limit variability.")
                 # logger.debug(f"Graph {i} single sequence: {list(generated_sequences)[0][:200]}...")
            elif num_unique_sequences > 1:
                 logger.log(log_level, f"Graph {i}: Generated {num_unique_sequences}/{args.num_augmentations_to_test} unique valid sequences using {order_name}.")
                 graphs_with_multiple_sequences += 1
            # else: case where all_augmentations_valid is False (already logged warnings/errors)

            if not all_augmentations_valid:
                validity_failures += 1 # Mark graph as failed if any augmentation failed seq gen or validity
                continue # Move to next graph

            # --- Step 4: Check Isomorphism (Optional) ---
            # Compare the *first* reconstructed graph (aug_idx=0) to the original
            if args.check_isomorphism:
                if first_reconstructed_graph is None: # Should have been caught by all_augmentations_valid
                    logger.error(f"Graph {i}: Isomorphism check requested, but first reconstructed graph is missing (logic error?).")
                    validity_failures += 1; continue

                original_nx_graph = pyg_data_to_nx(original_pyg_data)
                if original_nx_graph is None:
                    logger.warning(f"Graph {i}: Could not convert original PyG data to NX for comparison.")
                    isomorphism_failures += 1; validity_failures += 1; continue

                if original_nx_graph.number_of_nodes() != first_reconstructed_graph.number_of_nodes() or \
                   original_nx_graph.number_of_edges() != first_reconstructed_graph.number_of_edges():
                    logger.warning(f"Graph {i}: First reconstr. ({order_name}, aug 0) VALID but node/edge count mismatch. Not isomorphic.")
                    isomorphism_failures += 1; validity_failures += 1; continue

                try:
                    nm = nx.isomorphism.categorical_node_match('type', default='UNKNOWN')
                    em = nx.isomorphism.generic_edge_match(None, None, default=True)
                    if nx.is_isomorphic(original_nx_graph, first_reconstructed_graph, node_match=nm, edge_match=em):
                        successful_checks += 1 # Success = all augs valid & first is isomorphic
                    else:
                        logger.warning(f"Graph {i}: First reconstr. ({order_name}, aug 0) VALID but NOT isomorphic.")
                        isomorphism_failures += 1; validity_failures += 1
                except Exception as e:
                    logger.error(f"Graph {i}: Error during isomorphism check ({order_name}, aug 0): {e}")
                    isomorphism_failures += 1; validity_failures += 1
            else:
                # Success = all augmentations were validly reconstructed
                successful_checks += 1

        except Exception as e:
            logger.error(f"Critical error processing graph index {i}: {e}", exc_info=True)
            validity_failures += 1

        # Log progress periodically
        if processed_count % max(1, num_to_check // 10) == 0 and processed_count > 0: # Log ~10 times
             logger.info(f"Checked {processed_count}/{num_to_check} graphs...")

    # --- Final Reporting ---
    end_time = time.time()
    logger.info(f"Finished checking graphs with {order_name} ordering.")
    print(f"\n--- Augmentation & Conversion Check Summary ({order_name} Ordering) ---")
    print(f"Graphs Checked:                    {processed_count}")
    print(f"Augmentations Requested per Graph: {args.num_augmentations_to_test}")
    print(f"Sequence->Graph Warnings:          {conversion_warnings}")
    print(f"Graphs Failing Validity/Iso:       {validity_failures}")
    if args.check_isomorphism:
        print(f"  (Isomorphism Failures Only):     {isomorphism_failures}")
    print(f"Graphs Passing Checks:             {successful_checks}")
    print("-" * 45)
    print(f"Graphs Producing >1 Sequence:      {graphs_with_multiple_sequences} / {processed_count}")
    avg_unique_seqs = (total_unique_sequences_generated / processed_count) if processed_count > 0 else 0
    print(f"Avg. Unique Sequences per Graph:   {avg_unique_seqs:.2f}")
    print("-" * 45)
    success_rate = (successful_checks / processed_count) * 100 if processed_count > 0 else 0
    print(f"Overall Success Rate (Validity/Iso): {success_rate:.2f}%")
    print(f"Total Time:                        {end_time - start_time:.2f} seconds")
    print("-----------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check AIG graph -> sequence -> graph round-trip conversion and augmentation.')
    parser.add_argument('--data_dir', type=str, default="./datasets/aig/", help='Base directory containing the AIG PyG data')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Which data split to check')
    parser.add_argument('--num_graphs', type=int, default=10, help='Number of graphs to check from the dataset') # Reduced default for faster testing
    parser.add_argument('--check_isomorphism', action='store_true', help='Perform isomorphism check (original vs. first reconstruction)')
    parser.add_argument('--ordering', type=str, default='topo', choices=['bfs', 'deg', 'topo'], help='Sequence ordering method to test')
    # --- ADDED ARGUMENT ---
    parser.add_argument('--num_augmentations_to_test', type=int, default=5, help='Number of augmentations to generate and test per graph')


    parsed_args = parser.parse_args()
    if parsed_args.num_augmentations_to_test < 1:
         logger.error("--num_augmentations_to_test must be at least 1.")
         exit(1)

    # Validate data_dir
    if not os.path.isdir(parsed_args.data_dir): logger.error(f"Data directory not found: {parsed_args.data_dir}"); exit(1)
    processed_dir = os.path.join(parsed_args.data_dir, "processed")
    expected_data_file = os.path.join(processed_dir, f'{parsed_args.split}.pt')
    if not os.path.isfile(expected_data_file):
         logger.warning(f"Processed data file not found: {expected_data_file}.")
         # Continue, AIGPygDataset might handle this if raw files exist

    main(parsed_args)