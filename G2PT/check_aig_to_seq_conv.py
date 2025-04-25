# G2PT/check_aig_to_seq_conv_original_plus_pipo_iso_mandatory.py
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
    # ENSURE THIS IMPORTS THE CURRENT datasets_utils.py with offsets 71/75
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

# Removed import for calculate_structural_aig_metrics as check is disabled

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger("check_original_pipo_iso")

# --- Helper Function (pyg_data_to_nx_with_counts) ---
# (Using the version from the previous response that counts PIs/POs)
def pyg_data_to_nx_with_counts(data):
    """Converts PyG Data to NetworkX DiGraph and counts PI/PO nodes."""
    if not hasattr(data, 'x') or data.x is None: return None, 0, 0
    G = nx.DiGraph(); num_nodes = data.x.size(0)
    node_feature_idx_to_type = {0: 'NODE_CONST0', 1: 'NODE_PI', 2: 'NODE_AND', 3: 'NODE_PO'}
    node_types = []
    original_pi_count = 0
    original_po_count = 0
    if data.x.numel() == 0: return G, 0, 0
    if data.x.dim() > 1 and data.x.shape[1] > 1 and data.x.dtype == torch.float:
        type_indices = torch.argmax(data.x, dim=1).tolist()
    elif data.x.dim() == 1 and data.x.dtype == torch.long:
        type_indices = data.x.tolist()
    else:
        logger.error(f"Unexpected shape/type for data.x: {data.x.shape}, dtype: {data.x.dtype}. Cannot determine node types.")
        return None, 0, 0

    for i in range(num_nodes):
        if i < len(type_indices):
            node_type_idx = int(type_indices[i])
            node_type_str = node_feature_idx_to_type.get(node_type_idx, 'UNKNOWN')
            if node_type_str == 'UNKNOWN': logger.warning(f"Unknown node feature index {node_type_idx}.")
            G.add_node(i, type=node_type_str)
            node_types.append(node_type_str)
            if node_type_str == 'NODE_PI': original_pi_count += 1
            elif node_type_str == 'NODE_PO': original_po_count += 1
        else:
            logger.error(f"Node index mismatch: {num_nodes} vs {len(type_indices)}")
            G.add_node(i, type='UNKNOWN')

    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        edge_index = data.edge_index.cpu().numpy(); num_edges = edge_index.shape[1]
        for i in range(num_edges):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u in G and v in G: G.add_edge(u, v)
            else: logger.warning(f"Invalid edge index: ({u}, {v}) for max node {num_nodes-1}. Skipping.")
    return G, original_pi_count, original_po_count


# --- Main Function (Based on User's Original + PI/PO + Correct Offsets + Iso Mandatory) ---
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
    logger.info(f"Will check input PI/PO consistency, MANDATORY isomorphism, and reconstructed PI/PO counts.")
    logger.warning("NOTE: Strict structural validity check (calculate_structural_aig_metrics) is DISABLED.")

    # --- Define Mappings & Type Lists (USING CURRENT VOCAB IDS 71-76) ---
    node_feature_map = {0: 71, 1: 72, 2: 73, 3: 74} # CORRECT MAPPING
    edge_feature_map = {0: 75, 1: 76}                # CORRECT MAPPING
    actual_atom_type_list = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
    actual_bond_type_list = ['EDGE_INV', 'EDGE_REG']

    # --- Initialize counters ---
    conversion_warnings = 0
    pipo_mismatch_input = 0
    pipo_mismatch_recon = 0
    isomorphism_failures = 0
    seq_gen_failures = 0
    reconstruction_failures = 0
    successful_checks = 0
    processed_count = 0
    total_unique_sequences_generated = 0
    graphs_with_multiple_sequences = 0
    other_failures = 0 # Counter for unexpected errors

    # --- Main Loop (Iterate through graphs) ---
    for i in range(num_to_check):
        graph_start_time = time.time()
        graph_check_failed = False # Flag for current graph failure
        try:
            original_pyg_data = dataset[i]
            processed_count += 1

            # --- 0. Get Original PI/PO counts ---
            original_pi_expected = -1; original_po_expected = -1
            original_nx_graph = None; original_pi_counted = -1; original_po_counted = -1

            try:
                original_pi_expected = int(original_pyg_data.num_inputs)
                original_po_expected = int(original_pyg_data.num_outputs)
            except Exception as e:
                 logger.error(f"Graph {i}: Skipping. Failed to get expected PI/PO counts: {e}")
                 graph_check_failed = True; continue

            try:
                original_nx_graph, original_pi_counted, original_po_counted = pyg_data_to_nx_with_counts(original_pyg_data)
                if original_nx_graph is None: raise ValueError("pyg_data_to_nx_with_counts returned None")
            except Exception as e:
                logger.error(f"Graph {i}: Skipping. Failed to convert original PyG to NX: {e}")
                graph_check_failed = True; continue

            # --- CHECK 1: Input Data Consistency ---
            if original_pi_expected != original_pi_counted or original_po_expected != original_po_counted:
                 logger.warning(f"Graph {i}: Input PI/PO count mismatch! "
                                f"Expected (PI={original_pi_expected}, PO={original_po_expected}), "
                                f"Counted in graph (PI={original_pi_counted}, PO={original_po_counted}).")
                 pipo_mismatch_input += 1
                 # graph_check_failed = True; continue # Optional: Make this fatal

            # --- 1. Preprocessing: Convert PyG -> Dict with Vocab IDs ---
            processed_data_for_seq = {}
            fatal_preprocessing_error = False
            if hasattr(original_pyg_data, 'x') and original_pyg_data.x is not None and original_pyg_data.x.numel()>0:
                 node_x = original_pyg_data.x; node_vocab_ids = None
                 if node_x.dim()>1: node_feature_indices=node_x.argmax(dim=-1); node_vocab_ids=torch.tensor([node_feature_map.get(idx.item(), -1) for idx in node_feature_indices], dtype=torch.long)
                 elif node_x.dim()==1 and node_x.dtype == torch.long: node_vocab_ids = torch.tensor([node_feature_map.get(idx.item(), -1) for idx in node_x], dtype=torch.long)
                 else: node_vocab_ids=torch.tensor([-1]*node_x.shape[0],dtype=torch.long)
                 if torch.any(node_vocab_ids == -1): fatal_preprocessing_error = True; logger.warning(f"Graph {i}: Unknown node feature index during mapping.")
                 processed_data_for_seq['x'] = node_vocab_ids
            else: fatal_preprocessing_error = True; logger.warning(f"Graph {i}: Missing or empty node features.")
            if fatal_preprocessing_error: logger.error(f"Graph {i}: Skipping due to node preprocessing error."); graph_check_failed=True; continue
            # Edge processing
            if hasattr(original_pyg_data,'edge_index') and original_pyg_data.edge_index is not None and original_pyg_data.edge_index.numel() > 0:
                 processed_data_for_seq['edge_index'] = original_pyg_data.edge_index
                 if hasattr(original_pyg_data,'edge_attr') and original_pyg_data.edge_attr is not None and original_pyg_data.edge_attr.numel()>0:
                     edge_attr = original_pyg_data.edge_attr; edge_vocab_ids = None
                     if edge_attr.dim()>1: edge_feature_indices=edge_attr.argmax(dim=-1); edge_vocab_ids=torch.tensor([edge_feature_map.get(idx.item(),-1) for idx in edge_feature_indices],dtype=torch.long)
                     elif edge_attr.dim()==1 and edge_attr.dtype == torch.long: edge_vocab_ids = torch.tensor([edge_feature_map.get(idx.item(), -1) for idx in edge_attr], dtype=torch.long)
                     else: edge_vocab_ids = torch.tensor([-1]*edge_attr.shape[0],dtype=torch.long)
                     processed_data_for_seq['edge_attr'] = edge_vocab_ids
                 else: processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
            else: processed_data_for_seq['edge_index'] = torch.tensor([[],[]],dtype=torch.long); processed_data_for_seq['edge_attr'] = torch.tensor([],dtype=torch.long)
            # --- End Preprocessing ---

            # --- 2. Generate Sequence(s) and Reconstruct First Graph ---
            generated_sequences = set()
            first_reconstructed_graph = None
            for aug_idx in range(args.num_augmentations_to_test):
                try:
                    sequence_dict = order_function(
                        processed_data_for_seq,
                        actual_atom_type_list,
                        actual_bond_type_list,
                        aug_seed=aug_idx
                    )
                    sequence = sequence_dict.get("text", [""])[0]
                    if not sequence or sequence == "<boc> <eoc> <bog> <eog>":
                        logger.warning(f"Graph {i}, Aug {aug_idx}: Failed to generate sequence.")
                        # Don't mark graph as failed here, just note sequence failure if it's the first aug
                        if aug_idx == 0: seq_gen_failures += 1; graph_check_failed = True
                        break # Stop trying augmentations for this graph

                    generated_sequences.add(sequence)

                    # Only reconstruct the first augmentation for iso/pipo checks
                    if aug_idx == 0:
                        first_reconstructed_graph = seq_to_nxgraph(sequence, parsing_mode='robust')
                        if first_reconstructed_graph is None:
                            logger.warning(f"Graph {i}, Aug 0: seq_to_nxgraph returned None.")
                            reconstruction_failures += 1; graph_check_failed = True; break

                except Exception as e:
                    logger.warning(f"Graph {i}, Aug {aug_idx}: Failed during seq generation or reconstruction. Error: {e}")
                    if aug_idx == 0: reconstruction_failures += 1 # Count failure if first augmentation fails
                    graph_check_failed = True; break # Stop checking this graph

            if graph_check_failed: continue # Skip iso/pipo checks if conversion failed

            # Check if first graph was reconstructed (should always exist if graph_check_failed is False)
            if first_reconstructed_graph is None:
                 logger.error(f"Graph {i}: First reconstructed graph is missing after loop (logic error?).")
                 graph_check_failed = True; continue

            # --- 3. Check Isomorphism (Mandatory) ---
            isomorphic = False # Default to false until proven
            if original_nx_graph is None: graph_check_failed=True; continue # Should be caught earlier

            if original_nx_graph.number_of_nodes() != first_reconstructed_graph.number_of_nodes() or \
               original_nx_graph.number_of_edges() != first_reconstructed_graph.number_of_edges():
                logger.warning(f"Graph {i}: Node/edge count mismatch. Not isomorphic.")
                isomorphism_failures += 1; graph_check_failed = True
            else:
                try:
                    # Use categorical node match based on 'type' attribute
                    nm = nx.isomorphism.categorical_node_match('type', default='UNKNOWN')
                    # Edge match doesn't need attributes for basic isomorphism check
                    if nx.is_isomorphic(original_nx_graph, first_reconstructed_graph, node_match=nm):
                        isomorphic = True # Passed!
                    else:
                        logger.warning(f"Graph {i}: Failed isomorphism check.")
                        isomorphism_failures += 1; graph_check_failed = True
                except Exception as e:
                    logger.error(f"Graph {i}: Error during isomorphism check: {e}")
                    isomorphism_failures += 1; graph_check_failed = True

            if not isomorphic: continue # Stop if isomorphism check failed

            # --- 4. Check Reconstructed PI/PO Counts (only if isomorphic) ---
            recon_pi_count = 0; recon_po_count = 0
            for _, node_data in first_reconstructed_graph.nodes(data=True):
                 node_type = node_data.get('type')
                 if node_type == 'NODE_PI': recon_pi_count += 1
                 elif node_type == 'NODE_PO': recon_po_count += 1

            if recon_pi_count != original_pi_expected or recon_po_count != original_po_expected:
                 logger.warning(f"Graph {i}: Isomorphic but PI/PO count mismatch! "
                                f"Expected (PI={original_pi_expected}, PO={original_po_expected}), "
                                f"Reconstructed (PI={recon_pi_count}, PO={recon_po_count}).")
                 pipo_mismatch_recon += 1
                 graph_check_failed = True; continue

            # --- Success (Passed Isomorphism and PI/PO counts) ---
            if not graph_check_failed:
                successful_checks += 1
                # Log augmentation variety if multiple were tested
                num_unique_sequences = len(generated_sequences)
                total_unique_sequences_generated += num_unique_sequences
                if args.num_augmentations_to_test > 1:
                     if num_unique_sequences > 1:
                          graphs_with_multiple_sequences += 1
                     # Optional: Log if only 1 seq generated when more expected
                     # elif num_unique_sequences == 1: logger.warning(...)

        except Exception as e:
            logger.error(f"Critical error processing graph index {i}: {e}", exc_info=True)
            if not graph_check_failed: other_failures +=1 # Use separate counter for unexpected errors

        # Log progress
        if processed_count % max(1, num_to_check // 10) == 0 and processed_count > 0:
             logger.info(f"Checked {processed_count}/{num_to_check} graphs...")

    # --- Final Reporting ---
    end_time = time.time()
    logger.info(f"Finished checking graphs with {order_name} ordering.")
    print(f"\n--- Conversion & PI/PO Check Summary ({order_name} Ordering - Iso Mandatory) ---")
    print(f"Graphs Checked:                    {processed_count}")
    print(f"Augmentations Tested per Graph:    {args.num_augmentations_to_test}")
    print(f"Seq Gen / Reconstruction Errors:   {seq_gen_failures + reconstruction_failures}")
    print(f"Input Data PI/PO Count Mismatches: {pipo_mismatch_input}")
    print(f"Graphs Failing Isomorphism Check:  {isomorphism_failures}")
    print(f"Graphs Failing Recon PI/PO Check:  {pipo_mismatch_recon}")
    print(f"Other Critical Errors:             {other_failures}")
    # Calculate total checks that failed iso or pipo recon
    total_failed_checks = isomorphism_failures + pipo_mismatch_recon + seq_gen_failures + reconstruction_failures + other_failures

    print(f"Graphs Passing Checks:             {successful_checks}")
    print("-" * 45)
    if args.num_augmentations_to_test > 1:
        print(f"Graphs Producing >1 Sequence:      {graphs_with_multiple_sequences} / {processed_count}")
        avg_unique_seqs = (total_unique_sequences_generated / processed_count) if processed_count > 0 else 0
        print(f"Avg. Unique Sequences per Graph:   {avg_unique_seqs:.2f}")
        print("-" * 45)
    success_rate = (successful_checks / processed_count) * 100 if processed_count > 0 else 0
    print(f"Overall Success Rate (Iso + PI/PO): {success_rate:.2f}%")
    print(f"Total Time:                        {end_time - start_time:.2f} seconds")
    print("-----------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check AIG graph -> sequence -> graph round-trip (Isomorphism/PIPO mandatory).')
    parser.add_argument('--data_dir', type=str, default="./datasets/aig/",
                        help='Directory containing the AIG PyG data (root with raw/processed subdirs)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Which data split to check')
    parser.add_argument('--num_graphs', type=int, default=100, help='Number of graphs to check from the dataset')
    # Isomorphism check is now mandatory
    parser.add_argument('--ordering', type=str, default='topo', choices=['bfs', 'deg', 'topo'], help='Sequence ordering method to test')
    parser.add_argument('--num_augmentations_to_test', type=int, default=1, help='Number of augmentations to generate per graph (only first checked for iso/pipo)')

    parsed_args = parser.parse_args()
    if parsed_args.num_augmentations_to_test < 1:
         logger.error("--num_augmentations_to_test must be at least 1.")
         exit(1)

    # Validate data_dir
    if not os.path.isdir(parsed_args.data_dir): logger.error(f"Data directory not found: {parsed_args.data_dir}"); exit(1)
    raw_dir = os.path.join(parsed_args.data_dir, "raw")
    processed_dir = os.path.join(parsed_args.data_dir, "processed")
    if not os.path.isdir(raw_dir): logger.warning(f"Raw data directory not found: {raw_dir}.")
    if not os.path.isdir(processed_dir): os.makedirs(processed_dir); logger.info(f"Created processed directory: {processed_dir}")

    main(parsed_args)