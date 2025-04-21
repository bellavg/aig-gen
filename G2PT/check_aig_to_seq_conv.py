# G2PT/check_aig_to_seq_conv.py (MODIFIED TO TEST BFS, DEG, TOPO)
import os
import argparse
import torch
import networkx as nx
import numpy as np
import logging
from collections import defaultdict, Counter

# --- Import necessary functions and classes ---
try:
    from datasets_utils import (
        to_seq_by_bfs,
        to_seq_by_deg,  # Import the degree function
        to_seq_aig_topo, # Import the topo function
        seq_to_nxgraph
    )
except ImportError as e:
     print(f"Error importing from datasets_utils: {e}")
     print("Make sure datasets_utils.py is in the correct path and contains all required functions.")
     exit(1)

try:
    # Assuming datasets package is structured correctly relative to this script
    from datasets.aig_dataset import AIGPygDataset # To load processed data
except ImportError:
     # Fallback if run from within the datasets directory or main G2PT dir
     try:
         from aig_dataset import AIGPygDataset
     except ImportError:
         print("Error: Could not find AIGPygDataset. Make sure datasets/aig_dataset.py exists and is accessible.")
         exit(1)

try:
    # Assume evaluate_aigs.py is in the parent directory or same directory
    import sys
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Add current dir
    from evaluate_aigs import calculate_structural_aig_metrics, VALID_AIG_NODE_TYPES
    # sys.path.pop(0) # Clean up path modification if done
except ImportError:
    print("Warning: Could not import evaluate_aigs.py. Structural validity checks will be basic.")
    # (Keep the fallback calculate_structural_aig_metrics function)
    def calculate_structural_aig_metrics(G):
        metrics = {'is_structurally_valid': False, 'constraints_failed': ['Evaluation script not found']}
        if isinstance(G, nx.DiGraph) and G.number_of_nodes() > 0:
             # Basic check: is it a DAG?
             if nx.is_directed_acyclic_graph(G):
                  # Add other basic checks if desired (e.g., node types present)
                  node_types_present = {data.get('type', 'UNKNOWN') for _, data in G.nodes(data=True)}
                  if all(ntype in VALID_AIG_NODE_TYPES or ntype == 'UNKNOWN' for ntype in node_types_present):
                       metrics['is_structurally_valid'] = True
                       metrics['constraints_failed'] = []
                  else:
                       metrics['constraints_failed'] = ['Unknown node types found (basic check)']

             else:
                  metrics['constraints_failed'] = ['Not a DAG (basic check)']
        else:
            metrics['constraints_failed'] = ['Not a DiGraph or empty (basic check)']
        return metrics
    # Define valid types if evaluate_aigs wasn't imported
    VALID_AIG_NODE_TYPES = {'NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'}


# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("check_conversion")

# --- Helper Function (for isomorphism check) ---
def pyg_data_to_nx(data):
    # (This function remains unchanged, handles vocab IDs)
    if not hasattr(data, 'x') or data.x is None:
        logger.warning("PyG Data object missing or has None 'x' attribute.")
        return None

    G = nx.DiGraph()
    num_nodes = data.x.size(0)
    # Map from feature *index* (0-3) to type string
    node_feature_idx_to_type = {0: 'NODE_CONST0', 1: 'NODE_PI', 2: 'NODE_AND', 3: 'NODE_PO'}
    # Map from vocab ID (97-100) to feature index (0-3)
    vocab_id_to_feature_index = {97: 0, 98: 1, 99: 2, 100: 3}
    node_types = []

    if data.x.numel() == 0:
        logger.warning("data.x is empty.")
        # Return empty graph? Or handle as error? Return empty graph for now.
        return G

    # Determine node types based on data.x format
    type_indices = []
    if data.x.dim() > 1 and data.x.shape[1] > 1 and data.x.dtype == torch.float:
        # Assumes one-hot float tensor
        type_indices = torch.argmax(data.x, dim=1).tolist()
    elif data.x.dim() == 1:
        # Assumes integer tensor: could be feature indices (0-3) or vocab IDs (97-100)
        is_vocab_id = torch.all((data.x >= 97) & (data.x <= 100)).item()
        is_feature_idx = torch.all((data.x >= 0) & (data.x <= 3)).item()

        if is_vocab_id:
            type_indices = [vocab_id_to_feature_index.get(vid.item(), -1) for vid in data.x]
        elif is_feature_idx:
            type_indices = data.x.tolist() # Assume they are already 0-3 indices
        else:
            logger.error(f"Unexpected integer values in data.x (dim=1): {data.x.unique().tolist()}")
            return None # Unknown integer format
    else:
        logger.error(f"Unexpected shape or type for data.x: {data.x.shape}, dtype: {data.x.dtype}.")
        return None

    # Add nodes to graph
    for i in range(num_nodes):
        if i < len(type_indices):
            node_type_idx = int(type_indices[i])
            node_type_str = node_feature_idx_to_type.get(node_type_idx, 'UNKNOWN')
            if node_type_str == 'UNKNOWN': logger.warning(f"Unknown node feature index {node_type_idx} encountered.")
            G.add_node(i, type=node_type_str)
            node_types.append(node_type_str)
        else:
            # This case should ideally not happen if data.x represents all nodes
            logger.error(f"Index mismatch: num_nodes={num_nodes}, len(type_indices)={len(type_indices)}")
            # Add remaining nodes as UNKNOWN? Or return error?
            # For robustness, add remaining nodes as UNKNOWN
            for j in range(len(type_indices), num_nodes):
                 G.add_node(j, type='UNKNOWN')
                 logger.warning(f"Added node {j} as UNKNOWN due to index mismatch.")


    # Add edges
    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        edge_index = data.edge_index.cpu().numpy()
        num_edges = edge_index.shape[1]
        for i in range(num_edges):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            # Check if node indices are valid *before* adding edge
            if u in G and v in G:
                G.add_edge(u, v)
            else:
                logger.warning(f"Invalid edge index: ({u}, {v}) points to non-existent node (max index {num_nodes-1}). Skipping edge.")
    return G


# --- Main Function (Modified) ---
def main(args):
    logger.info(f"Loading processed AIG dataset from: {args.data_dir}")
    try:
        # Ensure the dataset loader doesn't apply transforms that change node features yet
        dataset = AIGPygDataset(root=args.data_dir, split=args.split, transform=None)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.data_dir}: {e}")
        return

    num_to_check = min(args.num_graphs, len(dataset))
    if num_to_check <= 0:
        logger.warning(f"No graphs found or requested in the dataset split '{args.split}'. Exiting.")
        return

    # --- Select Ordering Function ---
    order_function = None
    order_name = ""
    if args.ordering == 'bfs':
        order_function = to_seq_by_bfs
        order_name = "BFS"
    elif args.ordering == 'deg':
        order_function = to_seq_by_deg
        order_name = "Degree"
    elif args.ordering == 'topo':
        order_function = to_seq_aig_topo
        order_name = "Topological"
    else:
        # Should not happen due to argparse 'choices'
        logger.error(f"Unsupported ordering specified: {args.ordering}")
        return

    logger.info(f"Checking round-trip conversion using {order_name} ordering for {num_to_check} graphs from split '{args.split}'...")

    # --- Define Mappings --- (Keep as is)
    node_feature_map = {0: 97, 1: 98, 2: 99, 3: 100} # feature_idx -> vocab_id
    edge_feature_map = {0: 101, 1: 102} # feature_idx -> vocab_id (0=inv, 1=reg)

    # --- Define Type Lists --- (Keep as is)
    # These are needed by the conversion functions
    actual_atom_type_list = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'] # Corresponds to vocab IDs 97-100
    actual_bond_type_list = ['EDGE_INV', 'EDGE_REG'] # Corresponds to vocab IDs 101-102

    # --- Initialize counters --- (Keep as is)
    conversion_warnings = 0; validity_failures = 0; isomorphism_failures = 0
    successful_checks = 0; processed_count = 0

    # --- Main Loop ---
    for i in range(num_to_check):
        try:
            original_pyg_data = dataset[i]
            processed_count += 1

            # --- ** START PREPROCESSING: Convert PyG features to vocab IDs ** ---
            # This step ensures the input to our seq functions matches expectations
            processed_data_for_seq = {}
            fatal_preprocessing_error = False

            # Process Node Features (x)
            if hasattr(original_pyg_data, 'x') and original_pyg_data.x is not None and original_pyg_data.x.numel() > 0:
                node_x = original_pyg_data.x
                if node_x.dim() > 1 and node_x.shape[1] > 1 and node_x.dtype == torch.float:
                    # Assume one-hot float, convert to feature index, then to vocab ID
                    node_feature_indices = node_x.argmax(dim=-1)
                    node_vocab_ids = torch.tensor([node_feature_map.get(idx.item(), -1) for idx in node_feature_indices], dtype=torch.long)
                elif node_x.dim() == 1:
                    # Assume integer tensor: Could be feature indices (0-3) or already vocab IDs (97-100)
                    if torch.all((node_x >= 0) & (node_x <= 3)).item(): # Check if feature indices
                        node_vocab_ids = torch.tensor([node_feature_map.get(idx.item(), -1) for idx in node_x], dtype=torch.long)
                    elif torch.all((node_x >= 97) & (node_x <= 100)).item(): # Check if vocab IDs
                         node_vocab_ids = node_x.long() # Already vocab IDs
                    else:
                         logger.warning(f"Graph {i}: Node features are 1D int but not recognized indices (0-3) or vocab IDs (97-100). Values: {node_x.unique().tolist()}")
                         node_vocab_ids = torch.tensor([-1] * node_x.shape[0], dtype=torch.long) # Mark as invalid
                else:
                    logger.warning(f"Graph {i}: Unexpected node feature format (Shape: {node_x.shape}, Dtype: {node_x.dtype}).")
                    node_vocab_ids = torch.tensor([-1] * node_x.shape[0], dtype=torch.long)

                if torch.any(node_vocab_ids == -1):
                    logger.warning(f"Graph {i}: Unknown node feature index or format detected during preprocessing.")
                    fatal_preprocessing_error = True
                processed_data_for_seq['x'] = node_vocab_ids

            else:
                logger.warning(f"Graph {i}: Missing/empty node features (x). Cannot generate sequence.")
                fatal_preprocessing_error = True


            # Process Edge Info (edge_index, edge_attr) - only if node processing succeeded
            if not fatal_preprocessing_error:
                if hasattr(original_pyg_data, 'edge_index') and original_pyg_data.edge_index is not None:
                    processed_data_for_seq['edge_index'] = original_pyg_data.edge_index
                    # Process Edge Attributes (edge_attr) - can be missing or empty
                    if hasattr(original_pyg_data, 'edge_attr') and original_pyg_data.edge_attr is not None and original_pyg_data.edge_attr.numel() > 0:
                        edge_attr = original_pyg_data.edge_attr
                        if edge_attr.dim() > 1 and edge_attr.shape[1] > 1 and edge_attr.dtype == torch.float:
                             # Assume one-hot float, convert to feature index (0 or 1), then to vocab ID
                            edge_feature_indices = edge_attr.argmax(dim=-1)
                            edge_vocab_ids = torch.tensor([edge_feature_map.get(idx.item(), -1) for idx in edge_feature_indices], dtype=torch.long)
                        elif edge_attr.dim() == 1:
                             # Assume integer tensor: Could be feature indices (0=inv, 1=reg) or vocab IDs (101=inv, 102=reg)
                             if torch.all((edge_attr >= 0) & (edge_attr <= 1)).item(): # Check if feature indices
                                 edge_vocab_ids = torch.tensor([edge_feature_map.get(idx.item(), -1) for idx in edge_attr], dtype=torch.long)
                             elif torch.all((edge_attr >= 101) & (edge_attr <= 102)).item(): # Check if vocab IDs
                                  edge_vocab_ids = edge_attr.long() # Already vocab IDs
                             else:
                                 logger.warning(f"Graph {i}: Edge attributes are 1D int but not recognized indices (0-1) or vocab IDs (101-102). Values: {edge_attr.unique().tolist()}")
                                 edge_vocab_ids = torch.tensor([-1] * edge_attr.shape[0], dtype=torch.long) # Mark as invalid
                        else:
                             logger.warning(f"Graph {i}: Unexpected edge attribute format (Shape: {edge_attr.shape}, Dtype: {edge_attr.dtype}).")
                             edge_vocab_ids = torch.tensor([-1] * edge_attr.shape[0], dtype=torch.long)

                        if torch.any(edge_vocab_ids == -1):
                             logger.warning(f"Graph {i}: Unknown edge feature index or format detected during preprocessing.")
                             # Don't make this fatal? Seq generation might handle unknown edge types.
                             # fatal_preprocessing_error = True # Optional: make it fatal
                        processed_data_for_seq['edge_attr'] = edge_vocab_ids
                    else:
                        # Handle missing or empty edge_attr - assign empty tensor
                         processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
                         # Adjust edge_index if edge_attr was expected but missing/empty? No, seq functions should handle mismatch.
                else:
                    # Handle missing edge_index
                    processed_data_for_seq['edge_index'] = torch.tensor([[],[]], dtype=torch.long)
                    processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long) # Must also be empty
            # --- ** END PREPROCESSING ** ---


            if fatal_preprocessing_error:
                logger.error(f"Graph {i}: Skipping due to fatal preprocessing error.")
                validity_failures += 1 # Count as failure
                continue # Skip this graph

            # --- Step 1: Convert Processed Graph -> Sequence ---
            # !!! Use the selected order_function !!!
            sequence_dict = order_function(processed_data_for_seq, actual_atom_type_list, actual_bond_type_list)
            sequence = sequence_dict.get("text", [""])[0]

            if not sequence:
                logger.warning(f"Graph {i}: Failed to generate sequence from processed PyG data ({order_name}).")
                validity_failures += 1
                continue

            # --- Step 2: Convert Sequence -> Reconstructed Graph --- (Keep as is)
            reconstructed_graph = None
            try:
                # Use robust parsing to handle potential minor sequence errors
                reconstructed_graph = seq_to_nxgraph(sequence, parsing_mode='robust')
                if reconstructed_graph is None: raise ValueError("seq_to_nxgraph returned None")
            except Exception as e:
                logger.warning(f"Graph {i}: Failed to convert {order_name} sequence back to graph. Error: {e}")
                logger.debug(f"Failed sequence ({order_name}): {sequence[:300]}...") # Log more sequence
                conversion_warnings += 1; validity_failures += 1
                continue

            # --- Step 3: Check Validity of Reconstructed Graph --- (Keep as is)
            validity_metrics = calculate_structural_aig_metrics(reconstructed_graph)
            structurally_valid = validity_metrics.get('is_structurally_valid', False)
            if not structurally_valid:
                logger.warning(f"Graph {i}: Reconstructed graph (from {order_name}) FAILED structural validity checks.")
                logger.debug(f"Failure reasons: {validity_metrics.get('constraints_failed', ['Unknown'])}")
                logger.debug(f"Failed sequence ({order_name}): {sequence[:300]}...") # Log sequence again on failure
                validity_failures += 1
                continue

            # --- Step 4: Check Isomorphism (Optional) ---
            if args.check_isomorphism:
                original_nx_graph = pyg_data_to_nx(original_pyg_data)
                if original_nx_graph is None:
                     logger.warning(f"Graph {i}: Could not convert original PyG data to NX for comparison.")
                     # Count as failure if isomorphism check was requested but couldn't be performed
                     isomorphism_failures += 1; validity_failures += 1
                     continue # Skip iso check for this graph

                # Check basic properties first
                if original_nx_graph.number_of_nodes() != reconstructed_graph.number_of_nodes() or \
                   original_nx_graph.number_of_edges() != reconstructed_graph.number_of_edges():
                     logger.warning(f"Graph {i}: Reconstructed graph ({order_name}) VALID but node/edge count mismatch ({original_nx_graph.number_of_nodes()}/{original_nx_graph.number_of_edges()} vs {reconstructed_graph.number_of_nodes()}/{reconstructed_graph.number_of_edges()}). Not isomorphic.")
                     isomorphism_failures += 1; validity_failures += 1
                     continue

                # Proceed with isomorphism check if counts match
                try:
                    # Match nodes based on their 'type' attribute
                    nm = nx.isomorphism.categorical_node_match('type', default='UNKNOWN')
                    # Edges don't have attributes in this conversion, so basic edge matching is fine
                    em = nx.isomorphism.generic_edge_match(None, None, default=True) # Or lambda u,v: True

                    if nx.is_isomorphic(original_nx_graph, reconstructed_graph, node_match=nm, edge_match=em):
                        successful_checks += 1
                    else:
                        logger.warning(f"Graph {i}: Reconstructed graph ({order_name}) VALID but NOT isomorphic.")
                        # Optional: Add debug info about node/edge differences here if needed
                        isomorphism_failures += 1; validity_failures += 1
                except Exception as e:
                    logger.error(f"Graph {i}: Error during isomorphism check ({order_name}): {e}")
                    isomorphism_failures += 1; validity_failures += 1
            else:
                # If isomorphism check is skipped, validity is enough for success
                successful_checks += 1

        except Exception as e:
            logger.error(f"Critical error processing graph index {i}: {e}", exc_info=True)
            validity_failures += 1 # Count unexpected errors as failures

        if processed_count % 100 == 0 and processed_count > 0:
            logger.info(f"Checked {processed_count}/{num_to_check} graphs...")

    # --- Final Reporting ---
    logger.info(f"Finished checking graphs with {order_name} ordering.")
    print(f"\n--- Conversion Check Summary ({order_name} Ordering) ---") # Reflect tested order
    print(f"Graphs Attempted:         {processed_count}")
    print(f"Sequence->Graph Warnings: {conversion_warnings}") # Warnings during seq -> graph step
    print(f"Validity/Iso Failures:    {validity_failures}")   # Graphs failing validity or isomorphism
    if args.check_isomorphism:
        print(f"  (Iso Failures Only):    {isomorphism_failures}") # Specifically isomorphism failures among valid graphs
    print(f"Successful Round-Trips:   {successful_checks}") # Passed validity (and iso if checked)
    success_rate = (successful_checks / processed_count) * 100 if processed_count > 0 else 0
    print(f"Success Rate:             {success_rate:.2f}%")
    print("---------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check AIG graph -> sequence -> graph round-trip conversion.')
    parser.add_argument('--data_dir', type=str, default="./datasets/aig/",
                        help='Base directory containing the AIG PyG data (e.g., datasets/aig/)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Which data split to check (default: train)')
    parser.add_argument('--num_graphs', type=int, default=100,
                        help='Number of graphs to check from the dataset (default: 100)')
    parser.add_argument('--check_isomorphism', action='store_true',
                        help='Perform isomorphism check between original and reconstructed graph')
    # --- ADDED ARGUMENT ---
    parser.add_argument('--ordering', type=str, default='deg', choices=['bfs', 'deg', 'topo'],
                        help='Sequence ordering method to test (default: bfs)')

    parsed_args = parser.parse_args()

    # Validate data_dir (Keep as is)
    if not os.path.isdir(parsed_args.data_dir): logger.error(f"Data directory not found: {parsed_args.data_dir}"); exit(1)
    processed_dir = os.path.join(parsed_args.data_dir, "processed")
    # Check if the expected processed file exists for the split
    expected_data_file = os.path.join(processed_dir, f'{parsed_args.split}.pt')
    if not os.path.isfile(expected_data_file):
         logger.warning(f"Processed data file not found: {expected_data_file}.")
         logger.warning("Please ensure the AIG dataset has been processed (e.g., using aig_pkl_to_pyg.py).")
         # Optionally exit if the file is strictly required by AIGPygDataset
         # exit(1)


    # --- Check if selected function is available ---
    # (Imports are already checked at the top)


    main(parsed_args)