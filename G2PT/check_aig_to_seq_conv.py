# G2PT/check_aig_to_seq_conv.py (MODIFIED TO USE BFS)
import os
import argparse
import torch
import networkx as nx
import numpy as np
import logging
from collections import defaultdict, Counter

# --- Import necessary functions and classes ---
# !!! CHANGE: Import to_seq_by_bfs instead of to_seq_aig_topo !!!
from datasets_utils import to_seq_by_bfs, seq_to_nxgraph # <--- MODIFIED IMPORT
try:
    # Assuming datasets package is structured correctly relative to this script
    from datasets.aig_dataset import AIGPygDataset # To load processed data
except ImportError:
     # Fallback if run from within the datasets directory
     from aig_dataset import AIGPygDataset
try:
    # Assume evaluate_aigs.py is in the parent directory
    import sys
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Uncomment if needed
    from evaluate_aigs import calculate_structural_aig_metrics, VALID_AIG_NODE_TYPES
    # sys.path.pop(0) # Clean up path modification if done
except ImportError:
    print("Warning: Could not import evaluate_aigs.py. Structural validity checks will be basic.")
    # (Keep the fallback calculate_structural_aig_metrics function)
    def calculate_structural_aig_metrics(G):
        metrics = {'is_structurally_valid': False, 'constraints_failed': ['Evaluation script not found']}
        if isinstance(G, nx.DiGraph) and G.number_of_nodes() > 0:
             if nx.is_directed_acyclic_graph(G):
                  metrics['is_structurally_valid'] = True
                  metrics['constraints_failed'] = []
             else:
                  metrics['constraints_failed'] = ['Not a DAG (basic check)']
        return metrics
    VALID_AIG_NODE_TYPES = {'NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'}


# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("check_conversion")

# --- Helper Function (for isomorphism check) ---
def pyg_data_to_nx(data):
    # (This function remains unchanged)
    if not hasattr(data, 'edge_index') or not hasattr(data, 'x'):
        logger.warning("PyG Data object missing edge_index or x attribute.")
        return None

    G = nx.DiGraph()
    num_nodes = data.x.size(0)
    node_type_map = {0: 'NODE_CONST0', 1: 'NODE_PI', 2: 'NODE_AND', 3: 'NODE_PO'}
    node_types = []
    if hasattr(data, 'x') and data.x is not None and data.x.numel() > 0:
        if data.x.dim() > 1 and data.x.shape[1] > 1 and data.x.dtype == torch.float:
            type_indices = torch.argmax(data.x, dim=1).tolist()
        elif data.x.dim() == 1:
             is_vocab_id = torch.all((data.x >= 97) & (data.x <= 100)).item() if data.x.numel() > 0 else False
             if is_vocab_id:
                 vocab_id_to_feature_index = {97: 0, 98: 1, 99: 2, 100: 3}
                 type_indices = [vocab_id_to_feature_index.get(vid.item(), -1) for vid in data.x]
             else:
                 type_indices = data.x.tolist()
        else:
            logger.error(f"Unexpected shape or type for data.x: {data.x.shape}, dtype: {data.x.dtype}.")
            return None
    else:
         logger.warning("data.x is missing, None, or empty.")
         return None

    for i in range(num_nodes):
        if i < len(type_indices):
             node_type_idx = int(type_indices[i])
             node_type_str = node_type_map.get(node_type_idx, 'UNKNOWN')
             if node_type_str == 'UNKNOWN': logger.warning(f"Unknown node type index {node_type_idx}.")
             G.add_node(i, type=node_type_str)
             node_types.append(node_type_str)
        else:
            logger.error(f"Index mismatch: num_nodes={num_nodes}, len(type_indices)={len(type_indices)}")
            return None

    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if 0 <= u < num_nodes and 0 <= v < num_nodes: G.add_edge(u, v)
            else: logger.warning(f"Invalid edge index: ({u}, {v}) for num_nodes={num_nodes}")
    return G

# --- Main Function (Modified) ---
def main(args):
    logger.info(f"Loading processed AIG dataset from: {args.data_dir}")
    try:
        dataset = AIGPygDataset(root=args.data_dir, split=args.split)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.data_dir}: {e}")
        return

    num_to_check = min(args.num_graphs, len(dataset))
    if num_to_check <= 0:
        logger.warning(f"No graphs found or requested in the dataset split '{args.split}'. Exiting.")
        return

    # !!! Indicate BFS ordering is being used !!!
    logger.info(f"Checking round-trip conversion using BFS ordering for {num_to_check} graphs from split '{args.split}'...")

    # --- Define Mappings --- (Keep as is)
    node_feature_map = {0: 97, 1: 98, 2: 99, 3: 100}
    edge_feature_map = {0: 101, 1: 102}

    # --- Define Type Lists --- (Keep as is)
    max_vocab_id = 102
    full_vocab_list = [f"UNK_{i}" for i in range(max_vocab_id + 1)]
    full_vocab_list[97] = 'NODE_CONST0'; full_vocab_list[98] = 'NODE_PI'
    full_vocab_list[99] = 'NODE_AND'; full_vocab_list[100] = 'NODE_PO'
    full_vocab_list[101] = 'EDGE_INV'; full_vocab_list[102] = 'EDGE_REG'
    actual_atom_type_list = [full_vocab_list[97], full_vocab_list[98], full_vocab_list[99], full_vocab_list[100]]
    actual_bond_type_list = [full_vocab_list[101], full_vocab_list[102]]

    # --- Initialize counters --- (Keep as is)
    conversion_warnings = 0; validity_failures = 0; isomorphism_failures = 0
    successful_checks = 0; processed_count = 0

    # --- Main Loop ---
    for i in range(num_to_check):
        try:
            original_pyg_data = dataset[i]
            processed_count += 1

            # --- ** START PREPROCESSING ** --- (Keep as is)
            processed_data_for_seq = {} # Renamed variable for clarity
            fatal_preprocessing_error = False
            # (Keep the logic to convert original_pyg_data features to vocab IDs
            #  and store in processed_data_for_seq['x'], ['edge_index'], ['edge_attr'])
            # ... (Preprocessing logic remains the same as your previous script) ...
            if hasattr(original_pyg_data, 'x') and original_pyg_data.x is not None and original_pyg_data.x.numel() > 0:
                 node_x = original_pyg_data.x
                 if node_x.dim() > 1 and node_x.shape[1] > 1 and node_x.dtype == torch.float:
                     node_feature_indices = node_x.argmax(dim=-1)
                     node_vocab_ids = torch.tensor([node_feature_map.get(idx.item(), -1) for idx in node_feature_indices], dtype=torch.long)
                     if torch.any(node_vocab_ids == -1): logger.warning(f"Graph {i}: Unknown node feature index."); fatal_preprocessing_error = True
                     processed_data_for_seq['x'] = node_vocab_ids
                 elif node_x.dim() == 1: processed_data_for_seq['x'] = node_x.long()
                 else: logger.warning(f"Graph {i}: Unexpected node format."); fatal_preprocessing_error = True
            else: logger.warning(f"Graph {i}: Missing/empty node features."); fatal_preprocessing_error = True

            if not fatal_preprocessing_error:
                if hasattr(original_pyg_data, 'edge_index') and original_pyg_data.edge_index is not None:
                     processed_data_for_seq['edge_index'] = original_pyg_data.edge_index
                     if hasattr(original_pyg_data, 'edge_attr') and original_pyg_data.edge_attr is not None and original_pyg_data.edge_attr.numel() > 0:
                         edge_attr = original_pyg_data.edge_attr
                         if edge_attr.dim() > 1 and edge_attr.shape[1] > 1 and edge_attr.dtype == torch.float:
                             edge_feature_indices = edge_attr.argmax(dim=-1)
                             edge_vocab_ids = torch.tensor([edge_feature_map.get(idx.item(), -1) for idx in edge_feature_indices], dtype=torch.long)
                             if torch.any(edge_vocab_ids == -1): logger.warning(f"Graph {i}: Unknown edge feature index."); fatal_preprocessing_error = True
                             processed_data_for_seq['edge_attr'] = edge_vocab_ids
                         elif edge_attr.dim() == 1: processed_data_for_seq['edge_attr'] = edge_attr.long()
                         else: logger.warning(f"Graph {i}: Unexpected edge attr format."); processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
                     else: processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
                else: processed_data_for_seq['edge_index'] = torch.tensor([[],[]], dtype=torch.long); processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
            # --- ** END PREPROCESSING ** ---


            if fatal_preprocessing_error:
                 validity_failures += 1
                 continue # Skip this graph

            # --- Step 1: Convert Processed Graph -> Sequence ---
            # !!! CHANGE: Call to_seq_by_bfs instead of to_seq_aig_topo !!!
            sequence_dict = to_seq_by_bfs(processed_data_for_seq, actual_atom_type_list, actual_bond_type_list) # <--- MODIFIED CALL
            sequence = sequence_dict.get("text", [""])[0]

            if not sequence:
                 logger.warning(f"Graph {i}: Failed to generate sequence from processed PyG data (BFS).")
                 validity_failures += 1
                 continue

            # --- Step 2: Convert Sequence -> Reconstructed Graph --- (Keep as is)
            reconstructed_graph = None
            try:
                reconstructed_graph = seq_to_nxgraph(sequence)
                if reconstructed_graph is None: raise ValueError("seq_to_nxgraph returned None")
            except Exception as e:
                logger.warning(f"Graph {i}: Failed to convert BFS sequence back to graph. Error: {e}")
                logger.debug(f"Failed sequence (BFS): {sequence[:200]}...")
                conversion_warnings += 1; validity_failures += 1
                continue

            # --- Step 3: Check Validity of Reconstructed Graph --- (Keep as is)
            validity_metrics = calculate_structural_aig_metrics(reconstructed_graph)
            structurally_valid = validity_metrics.get('is_structurally_valid', False)
            if not structurally_valid:
                logger.warning(f"Graph {i}: Reconstructed graph (from BFS) FAILED structural validity checks.")
                logger.debug(f"Failure reasons: {validity_metrics.get('constraints_failed', ['Unknown'])}")
                validity_failures += 1
                continue

            # --- Step 4: Check Isomorphism (Optional) --- (Keep as is)
            if args.check_isomorphism:
                original_nx_graph = pyg_data_to_nx(original_pyg_data)
                if original_nx_graph is None:
                     logger.warning(f"Graph {i}: Could not convert original PyG data to NX for comparison.")
                     isomorphism_failures += 1; validity_failures += 1
                else:
                    try:
                        nm = nx.isomorphism.categorical_node_match('type', default='UNKNOWN')
                        em = nx.isomorphism.generic_edge_match(None, None, default=True)
                        if nx.is_isomorphic(original_nx_graph, reconstructed_graph, node_match=nm, edge_match=em):
                            successful_checks += 1
                        else:
                            logger.warning(f"Graph {i}: Reconstructed graph (BFS) VALID but NOT isomorphic.")
                            isomorphism_failures += 1; validity_failures += 1
                    except Exception as e:
                         logger.error(f"Graph {i}: Error during isomorphism check (BFS): {e}")
                         isomorphism_failures += 1; validity_failures += 1
            else:
                 successful_checks += 1

        except Exception as e:
            logger.error(f"Error processing graph index {i}: {e}", exc_info=True)
            validity_failures += 1

        if processed_count % 100 == 0 and processed_count > 0:
             logger.info(f"Checked {processed_count}/{num_to_check} graphs...")

    # --- Final Reporting --- (Keep as is)
    logger.info("Finished checking graphs.")
    print("\n--- Conversion Check Summary (BFS Ordering) ---") # <-- Added (BFS Ordering)
    print(f"Graphs Checked:           {processed_count}")
    print(f"Sequence->Graph Warnings: {conversion_warnings}")
    print(f"Validity/Iso Failures:    {validity_failures}")
    if args.check_isomorphism:
        print(f"  (Isomorphism Failures): {isomorphism_failures}")
    print(f"Successful Round-Trips:   {successful_checks}")
    success_rate = (successful_checks / processed_count) * 100 if processed_count > 0 else 0
    print(f"Success Rate:             {success_rate:.2f}%")
    print("------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check AIG graph -> sequence -> graph conversion (BFS default).') # <-- Changed description
    parser.add_argument('--data_dir', type=str, default="./datasets/aig/",
                        help='Base directory containing the AIG PyG data (e.g., datasets/aig/)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Which data split to check (default: train)')
    parser.add_argument('--num_graphs', type=int, default=100,
                        help='Number of graphs to check from the dataset (default: 100)')
    parser.add_argument('--check_isomorphism', action='store_true',
                        help='Perform isomorphism check between original and reconstructed graph')
    # Optional: Add an --ordering argument if you want to switch between topo/bfs easily
    # parser.add_argument('--ordering', type=str, default='bfs', choices=['topo', 'bfs'], help='Sequence ordering method')

    parsed_args = parser.parse_args()

    # Validate data_dir (Keep as is)
    if not os.path.isdir(parsed_args.data_dir): logger.error(f"Data directory not found: {parsed_args.data_dir}"); exit(1)
    processed_dir = os.path.join(parsed_args.data_dir, "processed")
    if not os.path.isdir(processed_dir): logger.warning(f"Processed directory not found: {processed_dir}.")

    # !!! IMPORTANT: Ensure datasets_utils.py has the corrected to_seq_by_bfs function available !!!
    try:
        from datasets_utils import to_seq_by_bfs
    except ImportError:
        logger.error("Could not import the required 'to_seq_by_bfs' function from datasets_utils.py.")
        logger.error("Make sure you have saved the modified 'to_seq_by_bfs' function in that file.")
        exit(1)


    main(parsed_args)