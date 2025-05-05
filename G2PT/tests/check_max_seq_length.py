# G2PT/check_max_sequence_length.py
# Checks the maximum token sequence length for a given split of the AIG dataset,
# including two prepended context tokens (PI/PO counts).
# Loads processed PyG data (.pt files).
# Uses sequence generation functions from datasets_utils.
# Derives mappings and types directly from G2PT.configs.aig.

import os
import argparse
import torch
import numpy as np
import logging
import warnings
import sys
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Import necessary AIG config first ---
try:
    import G2PT.configs.aig as aig_cfg
    print("Successfully imported G2PT.configs.aig")
except ImportError:
    print("Could not import G2PT.configs.aig directly. Adjusting sys.path.")
    # Attempt to adjust path assuming standard project structure
    script_dir = os.path.dirname(os.path.realpath(__file__)) # G2PT/datasets or G2PT/
    g2pt_root = script_dir if os.path.basename(script_dir) == 'G2PT' else os.path.dirname(script_dir)
    if g2pt_root not in sys.path:
        sys.path.insert(0, g2pt_root) # Prepend G2PT root to path
    try:
        import G2PT.configs.aig as aig_cfg
        print(f"Successfully imported G2PT.configs.aig after path adjustment.")
    except ImportError as e:
        print(f"Error importing G2PT.configs.aig even after path adjustment: {e}")
        sys.exit(1)

# --- Import necessary functions and classes AFTER potentially adjusting path ---
try:
    # Assuming datasets_utils is in G2PT/ or accessible via path
    from G2PT.datasets_utils import to_seq_aig_topo # Add other functions (bfs, deg) if needed
    print("Imported sequence functions from G2PT.datasets_utils.")
    # Assuming datasets package is structured correctly relative to G2PT root
    from G2PT.datasets.aig_dataset import AIGPygDataset # To load processed data
    print("Imported AIGPygDataset from G2PT.datasets.aig_dataset.")
except ImportError as e:
    print(f"Error importing from G2PT.datasets_utils or G2PT.datasets.aig_dataset: {e}")
    print("Make sure these files exist and the G2PT directory is in your Python path.")
    exit(1)


# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("check_max_len")

# --- Configuration (Adapt these paths/values if needed) ---
# Use relative paths assuming script might be run from G2PT root or similar
# Or provide absolute paths via command line arguments.
DEFAULT_DATA_DIR = 'datasets/aig' # Relative path from G2PT root
DEFAULT_TOKENIZER_PATH = 'datasets/aig/tokenizer' # Relative path from G2PT root

# --- Derive Mappings/Types from Config (Safer than hardcoding) ---
# These mappings convert the *feature index* (0, 1, 2, 3 or 0, 1) from PyG data
# to the *final vocabulary ID* used in the .bin files and sequence generation.
try:
    NODE_FEATURE_INDEX_TO_ID = aig_cfg.NODE_FEATURE_INDEX_TO_VOCAB
    EDGE_FEATURE_INDEX_TO_ID = aig_cfg.EDGE_FEATURE_INDEX_TO_VOCAB

    # These lists provide the *token strings* corresponding to the node/edge types
    # used by the to_seq_* functions. Get them directly from config keys.
    ATOM_TYPE_LIST = list(aig_cfg.NODE_TYPE_KEYS) # ['NODE_CONST0', 'NODE_PI', ...]
    BOND_TYPE_LIST = list(aig_cfg.EDGE_TYPE_KEYS) # ['EDGE_REG', 'EDGE_INV']
    print("Derived ID mappings and type lists from aig_cfg.")
    # Optional: Print mappings for verification
    # print("Node Mapping (Feature Idx -> Vocab ID):", NODE_FEATURE_INDEX_TO_ID)
    # print("Edge Mapping (Feature Idx -> Vocab ID):", EDGE_FEATURE_INDEX_TO_ID)
    # print("Node Type List:", ATOM_TYPE_LIST)
    # print("Edge Type List:", BOND_TYPE_LIST)
except AttributeError as e:
     print(f"Error accessing mapping/key attributes from imported aig_cfg: {e}")
     print("Ensure G2PT.configs.aig defines NODE_FEATURE_INDEX_TO_VOCAB, EDGE_FEATURE_INDEX_TO_VOCAB, NODE_TYPE_KEYS, EDGE_TYPE_KEYS.")
     exit(1)

# --- Main Function ---
def main(args):
    # Resolve potential relative paths from arguments
    data_dir = os.path.abspath(args.data_dir)
    tokenizer_path = os.path.abspath(args.tokenizer_path)

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    try:
        # Use AutoTokenizer to load based on the JSON files in the directory
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        # Ensure UNK token ID is available
        if tokenizer.unk_token_id is None:
             logger.warning("Tokenizer does not have an explicit unk_token_id defined.")
             # Attempt to get it from the token string if present in vocab
             if tokenizer.unk_token in tokenizer.vocab:
                  tokenizer.unk_token_id = tokenizer.vocab[tokenizer.unk_token]
                  logger.info(f"Found '{tokenizer.unk_token}' token and set it as unk_token_id ({tokenizer.unk_token_id}).")
             else:
                  logger.error(f"Cannot determine UNK token ID. Please ensure tokenizer has '{tokenizer.unk_token}' or sets unk_token_id.")
                  return
        else:
             logger.info(f"Tokenizer UNK token ID: {tokenizer.unk_token_id}")

    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        return

    logger.info(f"Loading processed AIG PyG dataset from: {data_dir} (Split: {args.split})")
    try:
        # Load the dataset using AIGPygDataset
        # It expects data_dir to contain raw/ and processed/ subdirs
        # It will load root/processed/aig_processed_{split}.pt
        dataset = AIGPygDataset(root=data_dir, split=args.split, transform=None)
        logger.info(f"Dataset loaded successfully. Number of graphs: {len(dataset)}")
    except FileNotFoundError as e:
         logger.error(f"Failed to load dataset: {e}")
         logger.error(f"Please ensure the processed file exists: {os.path.join(data_dir, 'processed', f'aig_processed_{args.split}.pt')}")
         logger.error("(This file is created by AIGPygDataset the first time it's initialized if the raw file exists)")
         return
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_dir} for split '{args.split}': {e}")
        return

    num_to_check = len(dataset)
    if num_to_check <= 0:
        logger.warning(f"No graphs found in the dataset split '{args.split}'. Exiting.")
        return

    # --- Select Ordering Function ---
    order_function = None
    if args.ordering == 'topo': order_function = to_seq_aig_topo
    # Add elif blocks here if other ordering functions (bfs, deg) are implemented in datasets_utils
    # elif args.ordering == 'bfs': order_function = to_seq_aig_bfs
    # elif args.ordering == 'deg': order_function = to_seq_aig_deg
    else:
        logger.error(f"Unsupported ordering: {args.ordering}. Available: 'topo' (add others if implemented).")
        return

    logger.info(f"Checking max sequence length using '{args.ordering}' ordering for {num_to_check} graphs from split '{args.split}'...")

    max_len = 0
    max_len_graph_idx = -1
    skipped_graphs = 0

    # --- Main Loop ---
    for i in tqdm(range(num_to_check), desc=f"Processing {args.split} split"):
        try:
            # Get the raw PyG Data object
            original_pyg_data = dataset[i]

            # --- Preprocessing: Convert PyG features to final vocab IDs ---
            # This step mimics the data preparation for sequence generation.
            processed_data_for_seq = {}
            valid_graph = True

            # Nodes
            if hasattr(original_pyg_data, 'x') and original_pyg_data.x is not None and original_pyg_data.x.numel() > 0:
                node_x = original_pyg_data.x
                # Assuming node features are one-hot encoded floats from pyg conversion
                if node_x.dim() > 1 and node_x.shape[1] > 1 and node_x.dtype == torch.float:
                    node_feature_indices = node_x.argmax(dim=-1) # Get index (0-3)
                    # Map feature index to final vocab ID using derived mapping
                    node_vocab_ids = torch.tensor([NODE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in node_feature_indices], dtype=torch.long)
                # Handle cases where 'x' might already contain feature indices or vocab IDs
                elif node_x.dim() == 1:
                     if torch.all((node_x >= 0) & (node_x < len(NODE_FEATURE_INDEX_TO_ID))).item(): # Check if feature indices (0-3)
                          node_vocab_ids = torch.tensor([NODE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in node_x], dtype=torch.long)
                     elif torch.all((node_x >= min(NODE_FEATURE_INDEX_TO_ID.values())) & (node_x <= max(NODE_FEATURE_INDEX_TO_ID.values()))).item(): # Check if vocab IDs
                          node_vocab_ids = node_x.long()
                     else: node_vocab_ids = torch.full_like(node_x, -1, dtype=torch.long) # Unrecognized 1D format
                else: node_vocab_ids = torch.full((node_x.shape[0],), -1, dtype=torch.long) # Unexpected shape

                if torch.any(node_vocab_ids == -1):
                    valid_graph = False; logger.debug(f"Idx {i}: Invalid node feature index/value found during mapping.")
                processed_data_for_seq['x'] = node_vocab_ids # Store final vocab IDs
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
                            # Map feature index to final vocab ID using derived mapping
                            edge_vocab_ids = torch.tensor([EDGE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in edge_feature_indices], dtype=torch.long)
                        # Handle cases where 'edge_attr' might already contain feature indices or vocab IDs
                        elif edge_attr.dim() == 1:
                             if torch.all((edge_attr >= 0) & (edge_attr < len(EDGE_FEATURE_INDEX_TO_ID))).item(): # Check if feature indices (0-1)
                                  edge_vocab_ids = torch.tensor([EDGE_FEATURE_INDEX_TO_ID.get(idx.item(), -1) for idx in edge_attr], dtype=torch.long)
                             elif torch.all((edge_attr >= min(EDGE_FEATURE_INDEX_TO_ID.values())) & (edge_attr <= max(EDGE_FEATURE_INDEX_TO_ID.values()))).item(): # Check if vocab IDs
                                  edge_vocab_ids = edge_attr.long()
                             else: edge_vocab_ids = torch.full_like(edge_attr, -1, dtype=torch.long) # Unrecognized 1D format
                        else: edge_vocab_ids = torch.full((edge_attr.shape[0],), -1, dtype=torch.long) # Unexpected shape

                        if torch.any(edge_vocab_ids == -1):
                             valid_graph = False; logger.debug(f"Idx {i}: Invalid edge feature index/value found during mapping.")
                        processed_data_for_seq['edge_attr'] = edge_vocab_ids # Store final vocab IDs, Shape [num_edges]
                    else: # Missing/empty edge_attr
                        valid_graph = False; logger.debug(f"Idx {i}: Graph has edges but no edge attributes.")
                        processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
                else: # Missing/empty edge_index (graph with only nodes)
                    processed_data_for_seq['edge_index'] = torch.tensor([[],[]], dtype=torch.long)
                    processed_data_for_seq['edge_attr'] = torch.tensor([], dtype=torch.long)
            # --- End Preprocessing ---


            # --- Fetch PI/PO Counts and Generate Context Tokens ---
            # Default to UNK tokens if counts are missing or invalid
            context_token_ids = torch.tensor([tokenizer.unk_token_id, tokenizer.unk_token_id], dtype=torch.long)

            if valid_graph: # Only proceed if graph structure is valid so far
                try:
                    # Check if attributes exist directly on the Data object
                    if hasattr(original_pyg_data, 'num_inputs') and hasattr(original_pyg_data, 'num_outputs') and \
                       original_pyg_data.num_inputs is not None and original_pyg_data.num_outputs is not None:

                        pi_count = int(original_pyg_data.num_inputs)
                        po_count = int(original_pyg_data.num_outputs)

                        # Convert counts to token strings (e.g., "PI_COUNT_5")
                        pi_token_str = f"PI_COUNT_{pi_count}"
                        po_token_str = f"PO_COUNT_{po_count}"

                        # Get token IDs from tokenizer, fallback to UNK token ID
                        # Use tokenizer.convert_tokens_to_ids for single tokens
                        pi_token_id = tokenizer.convert_tokens_to_ids(pi_token_str) \
                            if pi_token_str in tokenizer.vocab else tokenizer.unk_token_id
                        po_token_id = tokenizer.convert_tokens_to_ids(po_token_str) \
                            if po_token_str in tokenizer.vocab else tokenizer.unk_token_id

                        # Log warnings if context tokens are unknown
                        if pi_token_id == tokenizer.unk_token_id:
                            logger.warning(f"Idx {i}: Context token '{pi_token_str}' not found in tokenizer vocab. Using UNK.")
                        if po_token_id == tokenizer.unk_token_id:
                            logger.warning(f"Idx {i}: Context token '{po_token_str}' not found in tokenizer vocab. Using UNK.")

                        context_token_ids = torch.tensor([pi_token_id, po_token_id], dtype=torch.long)
                    else:
                        logger.warning(f"Graph index {i}: Missing 'num_inputs' or 'num_outputs' attribute on PyG Data object. Cannot determine context tokens. Using UNK.")
                        # Don't mark graph as invalid here, just use UNK context tokens

                except Exception as e:
                    logger.error(f"Error getting PI/PO counts or token IDs for graph index {i}: {e}. Using UNK context tokens.")
                    # Don't mark graph as invalid here, just use UNK context tokens
            # --- End Fetch PI/PO Counts ---


            if not valid_graph:
                # This only triggers if node/edge preprocessing failed
                logger.warning(f"Skipping graph index {i} due to node/edge preprocessing issues.")
                skipped_graphs += 1
                continue

            # --- Generate Sequence (using preprocessed graph data with final vocab IDs) ---
            # Pass the data dict containing final vocab IDs and the type lists derived from config
            sequence_dict = order_function(processed_data_for_seq, ATOM_TYPE_LIST, BOND_TYPE_LIST)
            # Expecting the function to return a dict with a 'text' key containing a list of strings
            sequence_string = sequence_dict.get("text", [""])[0] # Get the first (and likely only) sequence string

            if not sequence_string:
                logger.warning(f"Graph {i}: Failed to generate sequence string from processed data using {args.ordering} ordering.")
                skipped_graphs += 1
                continue

            # --- Tokenize Graph Sequence ---
            # Tokenize the main graph sequence string (without context tokens yet)
            try:
                # Use batch_encode_plus for potentially more robust handling, though single string is fine
                tokenized_graph_output = tokenizer(sequence_string, return_tensors=None, add_special_tokens=False)
                graph_token_ids = tokenized_graph_output['input_ids']
            except Exception as e:
                 logger.error(f"Error tokenizing sequence string for graph index {i}: {e}")
                 logger.error(f"Sequence string was: '{sequence_string[:100]}...'") # Log part of the string
                 skipped_graphs += 1
                 continue

            # --- Prepend Context Tokens and Check Length ---
            final_token_ids = context_token_ids.tolist() + graph_token_ids # Combine context and graph token ID lists
            current_len = len(final_token_ids) # Calculate total length

            # Update maximum length found
            if current_len > max_len:
                max_len = current_len
                max_len_graph_idx = i
                # Log new maximum found (less frequently for large datasets)
                if max_len % 50 == 0 or max_len < 100 or max_len > 1000: # Adjust logging frequency
                     logger.info(f"New max length found: {max_len} (Graph Index: {i})")

        except Exception as e:
            # Log general errors during the loop for a specific graph
            logger.error(f"General error processing graph index {i}: {e}", exc_info=False) # Set exc_info=True for traceback
            skipped_graphs += 1

    # --- Final Reporting ---
    print(f"\n--- Maximum Sequence Length Check Summary ---")
    print(f"Dataset Split: {args.split}")
    print(f"Ordering Method: {args.ordering}")
    print(f"Graphs Checked: {num_to_check}")
    if skipped_graphs > 0:
        print(f"Graphs Skipped due to errors: {skipped_graphs}")
    print(f"Maximum Token Sequence Length Found (incl. 2 context tokens): {max_len}")
    if max_len_graph_idx != -1:
        print(f"(Found in graph index: {max_len_graph_idx})")
    else:
        print("(No valid graphs processed or max length remained 0)")
    print("---------------------------------------------")
    if max_len > 0:
        recommended_block_size = 2**int(np.ceil(np.log2(max_len)))
        print(f"Recommended 'block_size' based on this check: >= {max_len}")
        print(f"Consider rounding up (e.g., to the next power of 2: {recommended_block_size}) that fits memory.")
    else:
        print("Could not determine a recommended block_size as max length is 0.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check maximum sequence length for AIG dataset including context tokens.')
    # Use more descriptive help messages
    parser.add_argument('--data_dir', type=str, default="../" + DEFAULT_DATA_DIR,
                        help=f'Root directory containing the AIG PyG data (expects raw/ and processed/ subdirs). Default: {DEFAULT_DATA_DIR}')
    parser.add_argument('--tokenizer_path', type=str, default="../" + DEFAULT_TOKENIZER_PATH,
                        help=f'Path to the saved Hugging Face tokenizer directory. Default: {DEFAULT_TOKENIZER_PATH}')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which data split to check (default: val)')
    parser.add_argument('--ordering', type=str, default='topo', choices=['topo'], # Add 'bfs', 'deg' if implemented
                        help='Sequence ordering method (must match datasets_utils function). Default: topo')

    parsed_args = parser.parse_args()

    # Resolve potential relative paths *before* validation
    abs_data_dir = parsed_args.data_dir
    abs_tokenizer_path = parsed_args.tokenizer_path

    # Validate paths
    if not os.path.isdir(abs_data_dir):
        logger.error(f"Data directory not found: {abs_data_dir}")
        exit(1)
    # Check for the 'processed' subdir specifically, as that's where AIGPygDataset looks for processed files
    processed_dir = os.path.join(abs_data_dir, 'processed')
    if not os.path.isdir(processed_dir):
        logger.warning(f"Processed data directory not found: {processed_dir}. AIGPygDataset will attempt to process raw data.")
        # Check if raw data exists if processed is missing
        raw_file = os.path.join(abs_data_dir, 'raw', f'{parsed_args.split}.pt')
        if not os.path.exists(raw_file):
             logger.error(f"Neither processed directory nor raw file found: {raw_file}")
             exit(1)

    if not os.path.isdir(abs_tokenizer_path):
        logger.error(f"Tokenizer directory not found: {abs_tokenizer_path}")
        exit(1)

    # Pass the absolute paths to main
    parsed_args.data_dir = abs_data_dir
    parsed_args.tokenizer_path = abs_tokenizer_path

    main(parsed_args)
