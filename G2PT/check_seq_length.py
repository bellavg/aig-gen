# G2PT/check_max_sequence_length.py
import os
import argparse
import torch
import numpy as np
import logging
import warnings # <-- Added import
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Import necessary functions and classes ---
try:
    # Assuming running from G2PT directory or datasets_utils is in path
    from datasets_utils import (
        to_seq_by_bfs,
        to_seq_by_deg,
        to_seq_aig_topo,
    )
    print("Imported sequence functions from datasets_utils.")
except ImportError as e:
    print(f"Error importing from datasets_utils: {e}")
    print("Make sure datasets_utils.py is in the correct path and contains updated functions.")
    exit(1)


# Assuming datasets package is structured correctly relative to this script
from G2PT.datasets.aig_dataset import AIGPygDataset # To load processed data



# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("check_max_len")

# --- Configuration (Adapt these paths/values if needed) ---
# Use relative paths or allow full paths via arguments
DEFAULT_DATA_DIR = '/Users/bellavg/aig-gen/G2PT/datasets/real_aigs' # Example relative path
DEFAULT_TOKENIZER_PATH = '/Users/bellavg/aig-gen/G2PT/tokenizers/aig' # Example relative path

# --- Define Mappings/Types (Should match datasets_utils and prepare_aig_final) ---
# These mappings convert the *feature index* (0, 1, 2, 3) from PyG data
# to the *final vocabulary ID* used in the .bin files and sequence generation.
NODE_FEATURE_INDEX_TO_ID = { 0: 71, 1: 72, 2: 73, 3: 74} # CONST0, PI, AND, PO
EDGE_FEATURE_INDEX_TO_ID = { 0: 75, 1: 76 } # INV, REG

# These lists provide the *token strings* corresponding to the node/edge types
# used by the to_seq_* functions.
ATOM_TYPE_LIST = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
BOND_TYPE_LIST = ['EDGE_INV', 'EDGE_REG']

# --- Main Function ---
def main(args):
    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    try:
        # Use AutoTokenizer to load based on the JSON files
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        # Check for UNK token ID
        if tokenizer.unk_token_id is None:
             logger.warning("Tokenizer does not have an explicit unk_token_id defined.")
             # Attempt to get it from the token string if present
             if '[UNK]' in tokenizer.vocab:
                  tokenizer.unk_token_id = tokenizer.vocab['[UNK]']
                  logger.info("Found '[UNK]' token and set it as unk_token_id.")
             else:
                  logger.error("Cannot determine UNK token ID. Please ensure tokenizer has '[UNK]' or sets unk_token_id.")
                  return

    except Exception as e:
        logger.error(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
        return

    logger.info(f"Loading processed AIG PyG dataset from: {args.data_dir}")
    try:
        # Load the dataset without transforms initially
        # Assumes AIGPygDataset loads the .pt files created by aig_pkl_to_pyg.py
        # which should contain num_inputs/num_outputs if that script added them.
        dataset = AIGPygDataset(root=args.data_dir, split=args.split, transform=None)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.data_dir} for split '{args.split}': {e}")
        logger.error("Ensure the path is correct and the processed .pt files (e.g., aig_processed_val.pt) exist in the 'processed' subdirectory.")
        return

    num_to_check = len(dataset)
    if num_to_check <= 0:
        logger.warning(f"No graphs found in the dataset split '{args.split}'. Exiting.")
        return

    # --- Select Ordering Function ---
    order_function = None
    if args.ordering == 'bfs': order_function = to_seq_by_bfs
    elif args.ordering == 'deg': order_function = to_seq_by_deg
    elif args.ordering == 'topo': order_function = to_seq_aig_topo
    else: logger.error(f"Unsupported ordering: {args.ordering}"); return

    logger.info(f"Checking max sequence length using {args.ordering} ordering for {num_to_check} graphs from split '{args.split}'...")

    max_len = 0
    max_len_graph_idx = -1

    # --- Main Loop ---
    for i in tqdm(range(num_to_check), desc=f"Processing {args.split} split"):
        try:
            original_pyg_data = dataset[i]

            # --- Preprocessing: Convert PyG features to final vocab IDs ---
            # This step mimics how the data is prepared before being passed
            # to the sequence generation functions in datasets_utils.
            processed_data_for_seq = {}
            valid_graph = True

            # Nodes
            if hasattr(original_pyg_data, 'x') and original_pyg_data.x is not None and original_pyg_data.x.numel() > 0:
                node_x = original_pyg_data.x
                # Assuming node features are one-hot encoded floats from pyg conversion
                if node_x.dim() > 1 and node_x.shape[1] > 1 and node_x.dtype == torch.float:
                    node_feature_indices = node_x.argmax(dim=-1) # Get index (0-3)
                    # Map feature index to final vocab ID (71-74)
                    node_vocab_ids = torch.tensor([NODE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in node_feature_indices], dtype=torch.long)
                # Add handling if 'x' might already contain vocab IDs or feature indices directly
                elif node_x.dim() == 1:
                     # Check if they are already feature indices (0-3)
                     if torch.all((node_x >= 0) & (node_x < len(NODE_FEATURE_INDEX_TO_ID))).item():
                          node_vocab_ids = torch.tensor([NODE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in node_x], dtype=torch.long)
                     # Check if they are already vocab IDs (71-74)
                     elif torch.all((node_x >= min(NODE_FEATURE_INDEX_TO_ID.values())) & (node_x <= max(NODE_FEATURE_INDEX_TO_ID.values()))).item():
                          node_vocab_ids = node_x.long()
                     else: # Unrecognized format
                          node_vocab_ids = torch.full_like(node_x, -1, dtype=torch.long)
                else: # Unexpected shape
                     node_vocab_ids = torch.full((node_x.shape[0],), -1, dtype=torch.long)


                if torch.any(node_vocab_ids == -1):
                    valid_graph = False; logger.debug(f"Idx {i}: Invalid node feature index found during mapping.")
                processed_data_for_seq['x'] = node_vocab_ids
            else:
                valid_graph = False; logger.debug(f"Idx {i}: Missing or empty node features (data.x).")

            # Edges (only if nodes were valid)
            if valid_graph:
                if hasattr(original_pyg_data, 'edge_index') and original_pyg_data.edge_index is not None and original_pyg_data.edge_index.numel() > 0:
                    processed_data_for_seq['edge_index'] = original_pyg_data.edge_index # Shape [2, num_edges]

                    if hasattr(original_pyg_data, 'edge_attr') and original_pyg_data.edge_attr is not None and original_pyg_data.edge_attr.numel() > 0:
                        edge_attr = original_pyg_data.edge_attr
                        # Assuming edge features are one-hot encoded floats
                        if edge_attr.dim() > 1 and edge_attr.shape[1] > 1 and edge_attr.dtype == torch.float:
                            edge_feature_indices = edge_attr.argmax(dim=-1) # Get index (0-1)
                            # Map feature index to final vocab ID (75-76)
                            edge_vocab_ids = torch.tensor([EDGE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in edge_feature_indices], dtype=torch.long)
                        # Add handling if 'edge_attr' might already contain vocab IDs or feature indices
                        elif edge_attr.dim() == 1:
                             # Check if they are already feature indices (0-1)
                             if torch.all((edge_attr >= 0) & (edge_attr < len(EDGE_FEATURE_INDEX_TO_ID))).item():
                                  edge_vocab_ids = torch.tensor([EDGE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in edge_attr], dtype=torch.long)
                            # Check if they are already vocab IDs (75-76)
                             elif torch.all((edge_attr >= min(EDGE_FEATURE_INDEX_TO_ID.values())) & (edge_attr <= max(EDGE_FEATURE_INDEX_TO_ID.values()))).item():
                                  edge_vocab_ids = edge_attr.long()
                             else: # Unrecognized format
                                  edge_vocab_ids = torch.full_like(edge_attr, -1, dtype=torch.long)
                        else: # Unexpected shape
                             edge_vocab_ids = torch.full((edge_attr.shape[0],), -1, dtype=torch.long)


                        if torch.any(edge_vocab_ids == -1):
                             valid_graph = False; logger.debug(f"Idx {i}: Invalid edge feature index found during mapping.")
                        processed_data_for_seq['edge_attr'] = edge_vocab_ids # Shape [num_edges]
                    else: # Missing/empty edge_attr
                        # If edges exist but no attributes, the graph is likely invalid for sequence generation
                        valid_graph = False; logger.debug(f"Idx {i}: Graph has edges but no edge attributes.")
                        processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)

                else: # Missing/empty edge_index (graph with only nodes)
                    processed_data_for_seq['edge_index'] = torch.tensor([[],[]], dtype=torch.long)
                    processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
            # --- End Preprocessing ---


            # --- Fetch PI/PO Counts and Generate Context Tokens ---
            context_token_ids = torch.tensor([tokenizer.unk_token_id, tokenizer.unk_token_id], dtype=torch.long) # Default

            if valid_graph: # Only proceed if graph structure is valid so far
                try:
                    # Check if attributes exist directly on the Data object
                    if hasattr(original_pyg_data, 'num_inputs') and hasattr(original_pyg_data, 'num_outputs') and \
                       original_pyg_data.num_inputs is not None and original_pyg_data.num_outputs is not None:

                        pi_count = int(original_pyg_data.num_inputs)
                        po_count = int(original_pyg_data.num_outputs)

                        # Convert counts to token strings
                        pi_token_str = f"PI_COUNT_{pi_count}"
                        po_token_str = f"PO_COUNT_{po_count}"

                        # Get token IDs, fallback to UNK token
                        pi_token_id = tokenizer.encode(pi_token_str, add_special_tokens=False)[0] \
                            if pi_token_str in tokenizer.vocab else tokenizer.unk_token_id
                        po_token_id = tokenizer.encode(po_token_str, add_special_tokens=False)[0] \
                            if po_token_str in tokenizer.vocab else tokenizer.unk_token_id

                        if pi_token_id == tokenizer.unk_token_id:
                            logger.warning(f"Idx {i}: Token '{pi_token_str}' not found in vocab. Using UNK.")
                        if po_token_id == tokenizer.unk_token_id:
                            logger.warning(f"Idx {i}: Token '{po_token_str}' not found in vocab. Using UNK.")

                        context_token_ids = torch.tensor([pi_token_id, po_token_id], dtype=torch.long)
                    else:
                        logger.warning(f"Graph index {i}: Missing 'num_inputs' or 'num_outputs' attribute on PyG Data object. Cannot determine context tokens.")
                        valid_graph = False # Mark graph as invalid if counts are missing

                except Exception as e:
                    logger.error(f"Error getting PI/PO counts or token IDs for graph index {i}: {e}")
                    valid_graph = False # Mark as invalid if counts fail
            # --- End Fetch PI/PO Counts ---


            if not valid_graph:
                logger.warning(f"Skipping graph index {i} due to preprocessing or count issues.")
                continue

            # --- Generate Sequence (using preprocessed graph data) ---
            # Pass the data dict containing final vocab IDs
            sequence_dict = order_function(processed_data_for_seq, ATOM_TYPE_LIST, BOND_TYPE_LIST)
            sequence_string = sequence_dict.get("text", [""])[0]

            if not sequence_string:
                logger.warning(f"Graph {i}: Failed to generate sequence string from processed data.")
                continue

            # --- Tokenize Graph Sequence ---
            tokenized_graph_output = tokenizer(sequence_string, return_tensors=None, add_special_tokens=False)
            graph_token_ids = tokenized_graph_output['input_ids']

            # --- Prepend Context Tokens and Check Length ---
            final_token_ids = context_token_ids.tolist() + graph_token_ids # Combine lists
            current_len = len(final_token_ids) # Calculate length *after* prepending

            if current_len > max_len:
                max_len = current_len
                max_len_graph_idx = i
                # Log less frequently to avoid spamming for large datasets
                if max_len % 10 == 0 or max_len < 50:
                     logger.info(f"New max length found: {max_len} (Graph Index: {i})")

        except Exception as e:
            # Log general errors during the loop for a specific graph
            logger.error(f"General error processing graph index {i}: {e}", exc_info=True)

    # --- Final Reporting ---
    print(f"\n--- Maximum Sequence Length Check Summary ---")
    print(f"Dataset Split: {args.split}")
    print(f"Ordering Method: {args.ordering}")
    print(f"Graphs Checked: {num_to_check}")
    print(f"Maximum Token Sequence Length Found (incl. 2 context tokens): {max_len}") # Updated message
    print(f"(Found in graph index: {max_len_graph_idx})")
    print("---------------------------------------------")
    print(f"Recommended 'block_size' based on this check: >= {max_len}")
    print(f"Consider rounding up (e.g., {2**int(np.ceil(np.log2(max_len))) if max_len > 0 else 512}) that fits memory and is convenient.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check maximum sequence length for AIG dataset including context tokens.')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Base directory containing the AIG PyG data (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH,
                        help=f'Path to the saved tokenizer directory (default: {DEFAULT_TOKENIZER_PATH})')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which data split to check (default: val)')
    parser.add_argument('--ordering', type=str, default='topo', choices=['bfs', 'deg', 'topo'],
                        help='Sequence ordering method used (must match datasets_utils function)')

    parsed_args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(parsed_args.data_dir): logger.error(f"Data directory not found: {parsed_args.data_dir}"); exit(1)
    # Check for the 'processed' subdir specifically, as that's where AIGPygDataset looks
    processed_dir = os.path.join(parsed_args.data_dir, 'processed')
    if not os.path.isdir(processed_dir): logger.error(f"Processed data directory not found: {processed_dir}"); exit(1)

    if not os.path.isdir(parsed_args.tokenizer_path): logger.error(f"Tokenizer directory not found: {parsed_args.tokenizer_path}"); exit(1)

    main(parsed_args)