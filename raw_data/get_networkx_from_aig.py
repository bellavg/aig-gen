#!/usr/bin/env python3
"""
AIG Graph Dataset Generator (Configurable, Chunked Output)

This script processes AIG files from a specified folder, converts them to
graph representations using configurations from G2PT.configs.aig, and saves
them incrementally as multiple pickle files (chunks) to manage memory usage.
It shuffles the input files before processing.
It also saves a summary text file containing processing statistics (counts, issues).
It enforces constraints on input/output sizes and total node count based on the config.
It logs warnings for non-DAG graphs or graphs with isolated nodes (excluding Const0)
but does *not* reject them based on these checks alone.
Output patterns are padded to a consistent length.

Potential Fix for SIGSEGV: Modified get_edges to correctly resolve PO driver nodes
using aig.get_node() and aig.is_complemented(), similar to the user's
original working script.
"""

import os
import sys
import pickle
import networkx as nx
# Assuming aigverse is installed or available in the path
try:
    # Import necessary functions from aigverse
    from aigverse import read_aiger_into_aig, to_edge_list, simulate, simulate_nodes
except ImportError:
    # Provide helpful error message if aigverse is not found
    print("Error: 'aigverse' library not found. Please install it (e.g., pip install aigverse)")
    sys.exit(1)
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter
import time
import logging
import math # Needed for ceil calculation
import io # Needed for capturing print output to string
import random # Needed for shuffling file list

# --- Add G2PT root to path to import config ---
# Try importing directly first. If it fails, assume the script is potentially
# one level above the G2PT directory or G2PT is in the Python path.
try:
    # Attempt to import the configuration module
    import G2PT.configs.aig as aig_cfg
    print(f"Successfully imported AIG config from: G2PT.configs.aig")
except ImportError:
    print("Could not import G2PT.configs.aig directly.")
    # Try adding the parent directory of the script's location to the path
    # This assumes the script might be in a directory parallel to G2PT
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)
    print(f"Adding '{parent_dir}' to sys.path to find G2PT.")
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        # Retry import after path adjustment
        import G2PT.configs.aig as aig_cfg
        print(f"Successfully imported AIG config after path adjustment.")
    except ImportError as e:
        print(f"Error importing G2PT.configs.aig even after path adjustment: {e}")
        print(f"Current sys.path: {sys.path}")
        print("Ensure the G2PT root directory is accessible via the Python path or installed.")
        sys.exit(1)
except AttributeError as e:
     # Catch cases where the module is imported but lacks expected attributes
     print(f"Error accessing attributes from aig_cfg: {e}")
     print("Ensure G2PT.configs.aig defines necessary constants (e.g., MAX_NODE_COUNT, NODE_TYPE_KEYS).")
     sys.exit(1)

# --- Logger Setup ---
# Configure logging level (e.g., INFO, DEBUG)
LOG_LEVEL = logging.INFO
# Set format for log messages
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
# Get logger instance
logger = logging.getLogger("aig_dataset_generator")


# =============================================================================
# CONFIGURATION CONSTANTS (Partially Hardcoded, Partially from aig_cfg)
# =============================================================================

# --- Path configuration ---
# !! IMPORTANT !! Using the absolute path provided by the user.
INPUT_PATH = "/raw_data/raw_aigs/"
# Define base output file name and directory
OUTPUT_FILENAME_BASE = "real_aigs" # Base name for chunked files and stats file
# Save output files and stats file in the same directory as the input AIGs
OUTPUT_DIR = INPUT_PATH
# Define the name for the statistics summary text file
STATS_FILENAME = f"{OUTPUT_FILENAME_BASE}_stats_summary.txt"
# Construct the full path for the statistics file
STATS_FILE_PATH = os.path.join(OUTPUT_DIR, STATS_FILENAME)

# Log the configured paths (using absolute paths for clarity)
logger.info(f"Using INPUT_PATH: {os.path.abspath(INPUT_PATH)}")
logger.info(f"Output files will be saved in: {os.path.abspath(OUTPUT_DIR)} with base name: {OUTPUT_FILENAME_BASE}")
logger.info(f"Statistics summary will be saved to: {os.path.abspath(STATS_FILE_PATH)}")

# --- Dataset Size and Chunking Configuration (User Defined) ---
# Total number of graphs expected to be applicable/accepted after filtering
# !! Adjust this value based on your pre-analysis or estimation !!
TOTAL_APPLICABLE_GRAPHS = 156042
# Desired number of output pickle files
NUM_OUTPUT_FILES = 6
# Calculate chunk size (number of graphs per file)
# Use math.ceil to ensure all graphs are saved, even if the division isn't perfect
CHUNK_SIZE = math.ceil(TOTAL_APPLICABLE_GRAPHS / NUM_OUTPUT_FILES) if NUM_OUTPUT_FILES > 0 and TOTAL_APPLICABLE_GRAPHS > 0 else TOTAL_APPLICABLE_GRAPHS if TOTAL_APPLICABLE_GRAPHS > 0 else 1
logger.info(f"Expecting ~{TOTAL_APPLICABLE_GRAPHS} applicable graphs.")
logger.info(f"Configured to save into {NUM_OUTPUT_FILES} chunk files.")
logger.info(f"Calculated chunk size: {CHUNK_SIZE} graphs per file.")

# --- Filter constraints (from aig_cfg) ---
MAX_SIZE = aig_cfg.MAX_NODE_COUNT      # Maximum number of nodes in graph
MAX_INPUTS = aig_cfg.MAX_PI_COUNT     # Maximum number of inputs
MAX_OUTPUTS = aig_cfg.MAX_PO_COUNT    # Maximum number of outputs

# --- Truth table configuration (from aig_cfg) ---
# Maximum truth table length derived from max inputs
try:
    # Calculate 2^MAX_INPUTS safely
    MAX_TT_LENGTH = 2 ** MAX_INPUTS
except OverflowError:
    # Handle cases where MAX_INPUTS is too large for standard integer exponentiation
    logger.warning(f"MAX_INPUTS ({MAX_INPUTS}) is large, MAX_TT_LENGTH might overflow standard integers.")
    MAX_TT_LENGTH = float('inf') # Use infinity as a fallback
except TypeError:
    # Handle cases where MAX_INPUTS is not a valid number
    logger.error(f"MAX_INPUTS is not a valid number in aig_cfg ({MAX_INPUTS}). Cannot calculate MAX_TT_LENGTH.")
    sys.exit(1)


# Padding value for truth tables (from aig_cfg)
TT_PADDING_VALUE = aig_cfg.PAD_VALUE

# --- Node and edge type encodings (Using keys and dicts from config) ---
# Validate that necessary keys and dictionaries exist in the config
required_node_keys = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
required_edge_keys = ["EDGE_REG", "EDGE_INV"]

# Check for presence and basic structure of config attributes
if not hasattr(aig_cfg, 'NODE_TYPE_KEYS') or not all(k in aig_cfg.NODE_TYPE_KEYS for k in required_node_keys):
     logger.error("aig_cfg.NODE_TYPE_KEYS is missing or incomplete.")
     sys.exit(1)
if not hasattr(aig_cfg, 'EDGE_TYPE_KEYS') or not all(k in aig_cfg.EDGE_TYPE_KEYS for k in required_edge_keys):
     logger.error("aig_cfg.EDGE_TYPE_KEYS is missing or incomplete.")
     sys.exit(1)
if not hasattr(aig_cfg, 'NODE_TYPE_ENCODING') or not isinstance(aig_cfg.NODE_TYPE_ENCODING, dict):
     logger.error("aig_cfg.NODE_TYPE_ENCODING is missing or not a dictionary.")
     sys.exit(1)
if not hasattr(aig_cfg, 'EDGE_LABEL_ENCODING') or not isinstance(aig_cfg.EDGE_LABEL_ENCODING, dict):
     logger.error("aig_cfg.EDGE_LABEL_ENCODING is missing or not a dictionary.")
     sys.exit(1)

# Get keys in the order defined in the config (assuming order matters and keys exist)
# These keys are used to look up encoding vectors later
NODE_CONST0_KEY = aig_cfg.NODE_TYPE_KEYS[0] # e.g., "NODE_CONST0"
NODE_PI_KEY = aig_cfg.NODE_TYPE_KEYS[1]     # e.g., "NODE_PI"
NODE_AND_KEY = aig_cfg.NODE_TYPE_KEYS[2]    # e.g., "NODE_AND"
NODE_PO_KEY = aig_cfg.NODE_TYPE_KEYS[3]     # e.g., "NODE_PO"

EDGE_REG_KEY = aig_cfg.EDGE_TYPE_KEYS[0]    # e.g., "EDGE_REG"
EDGE_INV_KEY = aig_cfg.EDGE_TYPE_KEYS[1]    # e.g., "EDGE_INV"

# Use the encoding dictionaries directly from the config
NODE_TYPE_ENCODING = aig_cfg.NODE_TYPE_ENCODING
EDGE_LABEL_ENCODING = aig_cfg.EDGE_LABEL_ENCODING

# Define the node ID assumed to be Const0 for exclusion in isolated check
CONST0_NODE_ID = 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_graph_chunk(graph_chunk: List[nx.DiGraph], output_dir: str, base_filename: str, chunk_num: int, total_chunks: int) -> None:
    """
    Saves a single chunk of graphs to a pickle file. Ensures directory exists.

    Args:
        graph_chunk: List of DiGraph objects in the current chunk.
        output_dir: Directory where the file will be saved.
        base_filename: Base name for the output files (e.g., 'generated_aigs').
        chunk_num: The current chunk number (1-based index).
        total_chunks: The total number of expected chunks.
    """
    if not graph_chunk:
        logger.warning(f"Attempted to save an empty chunk (Chunk {chunk_num}/{total_chunks}). Skipping.")
        return

    # Ensure the output directory exists
    try:
        # Use absolute path for directory creation
        abs_output_dir = os.path.abspath(output_dir)
        if abs_output_dir:
             os.makedirs(abs_output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {abs_output_dir}: {e}")
        return # Cannot proceed without output directory

    # Construct filename: <output_dir>/<base_filename>_part_<chunk_num>_of_<total_chunks>.pkl
    output_filename = f"{base_filename}_part_{chunk_num}_of_{total_chunks}.pkl"
    # Use absolute path for saving
    output_file_path = os.path.join(abs_output_dir, output_filename)

    logger.info(f"Saving {len(graph_chunk)} graphs to {output_file_path} (Chunk {chunk_num}/{total_chunks})...")
    try:
        # Open file in binary write mode and dump the chunk
        with open(output_file_path, "wb") as f:
            pickle.dump(graph_chunk, f)
        logger.info(f"Successfully saved chunk {chunk_num} to {output_file_path}")
    except Exception as e:
        # Log errors during saving
        logger.error(f"Failed to save graph chunk {chunk_num} to {output_file_path}: {e}")


def save_stats_to_file(stats_content: str, output_file_path: str) -> None:
    """
    Saves the provided statistics content string to a text file.

    Args:
        stats_content: A string containing the formatted statistics summary.
        output_file_path: The full path to the output text file.
    """
    try:
        # Ensure the output directory exists using absolute path
        abs_output_file_path = os.path.abspath(output_file_path)
        output_dir = os.path.dirname(abs_output_file_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)

        # Open the file in write mode with UTF-8 encoding and write content
        with open(abs_output_file_path, "w", encoding="utf-8") as f:
            f.write(stats_content)
        logger.info(f"Successfully saved statistics summary to {abs_output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save statistics summary to {abs_output_file_path}: {e}")


def generate_binary_inputs(num_inputs: int) -> List[List[int]]:
    """
    Generates all possible binary input combinations for a given number of inputs.
    Warns if num_inputs exceeds configured MAX_INPUTS.

    Args:
        num_inputs: Number of input variables

    Returns:
        List of all possible binary input combinations
    """
    # Warn if the number of inputs exceeds the configured maximum
    if num_inputs > MAX_INPUTS:
        logger.warning(f"Generating inputs for {num_inputs} inputs, which exceeds MAX_INPUTS ({MAX_INPUTS}). This might lead to large truth tables.")
    # Handle the edge case of zero inputs
    if num_inputs == 0:
        return [[]] # A single empty input pattern
    try:
        # Calculate 2^num_inputs safely using bit shift
        num_combinations = 1 << num_inputs
        # Add a sanity check for potentially very large numbers of combinations
        if num_combinations > (1 << 20): # e.g., more than a million
             logger.warning(f"Generating a very large number of input combinations: {num_combinations}")

        # Generate combinations using list comprehension and bit manipulation
        # Creates [[0,0,0], [0,0,1], ..., [1,1,1]] for num_inputs=3
        return [[(i >> bit) & 1 for bit in range(num_inputs - 1, -1, -1)]
                for i in range(num_combinations)]
    except OverflowError:
        logger.error(f"Cannot generate binary inputs for {num_inputs} inputs due to overflow.")
        return []
    except TypeError:
         logger.error(f"Invalid num_inputs provided: {num_inputs}. Must be an integer.")
         return []


def get_padded_truth_table(tt_binary: str, num_inputs_for_aig: int) -> List[int]:
    """
    Convert binary truth table string to list and pad with config PAD_VALUE
    to the length determined by MAX_INPUTS from config (MAX_TT_LENGTH).

    Args:
        tt_binary: Binary string representation of truth table.
        num_inputs_for_aig: The actual number of inputs for the specific AIG
                           (used for warning if TT length mismatch).

    Returns:
        Padded truth table as a list of integers, or truncated list if too long.
    """
    # Handle the case where padding is effectively disabled due to overflow
    if MAX_TT_LENGTH == float('inf'):
        logger.debug("MAX_TT_LENGTH is infinite, cannot pad. Returning original TT as list of ints.")
        try:
            return [int(bit) for bit in tt_binary]
        except ValueError:
             logger.error(f"Invalid character found in truth table binary string: '{tt_binary}'. Returning empty list.")
             return []


    # Expected length based on the AIG's actual inputs (capped by MAX_INPUTS for TT calculation)
    effective_inputs = min(num_inputs_for_aig, MAX_INPUTS)
    # Calculate 2^effective_inputs safely
    expected_len = 1 << effective_inputs if effective_inputs >= 0 else 0

    # Log potential mismatch between actual TT length and expected length before padding
    if len(tt_binary) != expected_len:
        logger.debug(f"Input tt_binary length ({len(tt_binary)}) differs from expected length ({expected_len}) for {num_inputs_for_aig} inputs (before padding/truncation to MAX_TT_LENGTH).")

    # Convert binary string to list of integers
    try:
        tt_list = [int(bit) for bit in tt_binary]
    except ValueError:
        logger.error(f"Invalid character found in truth table binary string: '{tt_binary}'. Returning empty list.")
        return []


    # Calculate padding needed based on the global MAX_TT_LENGTH
    padding_length = MAX_TT_LENGTH - len(tt_list)

    if padding_length < 0:
        # Truncate if the truth table is longer than MAX_TT_LENGTH
        logger.debug(f"Truth table length ({len(tt_list)}) exceeds MAX_TT_LENGTH ({MAX_TT_LENGTH}). Truncating.")
        return tt_list[:MAX_TT_LENGTH]
    elif padding_length > 0:
        # Pad with the configured padding value if shorter
        try:
            padding = [TT_PADDING_VALUE] * padding_length
            return tt_list + padding
        except TypeError:
             # Handle potential errors if padding value or length is invalid
             logger.error(f"Invalid TT_PADDING_VALUE ({TT_PADDING_VALUE}) or padding_length ({padding_length}). Cannot pad.")
             return tt_list # Return unpadded list as fallback
    else:
        # Return as is if length matches MAX_TT_LENGTH
        return tt_list


def get_nodes(aig: Any, G: nx.DiGraph, pad: bool = True) -> nx.DiGraph:
    """
    Add nodes (Const0, PI, AND) to the graph with features and types from config.

    Args:
        aig: AIG object from aigverse.
        G: NetworkX DiGraph to add nodes to.
        pad: Whether to pad truth tables to max length (using config MAX_TT_LENGTH).

    Returns:
        Updated DiGraph with nodes added.
    """
    try:
        # Get number of primary inputs from AIG object
        num_inputs = aig.num_pis()
    except Exception as e:
        logger.error(f"Error getting number of PIs from AIG object: {e}. Cannot proceed with node addition.")
        return G # Return unchanged graph

    # Log if number of inputs exceeds configured maximum
    if num_inputs > MAX_INPUTS:
        logger.debug(f"AIG has {num_inputs} inputs, exceeding MAX_INPUTS ({MAX_INPUTS}). Truth tables will be large or truncated.")

    # Generate all binary input patterns for simulation
    input_patterns = list(zip(*generate_binary_inputs(num_inputs)))

    # Calculate expected truth table length for *this specific AIG* (capped by MAX_INPUTS)
    effective_inputs = min(num_inputs, MAX_INPUTS)
    current_tt_len = 1 << effective_inputs if effective_inputs >= 0 else 0

    # Helper function to format truth table (pad or just convert to int list)
    def format_tt(binary_str):
        return get_padded_truth_table(binary_str, num_inputs) if pad else [int(b) for b in binary_str]

    # --- Add Constant 0 Node ---
    try:
        # Generate zero truth table of the appropriate length
        zero_tt_str = "0" * current_tt_len if current_tt_len > 0 else "0"
        zero_tt = format_tt(zero_tt_str)
        # Get node type encoding from config
        node_type_array = np.array(NODE_TYPE_ENCODING[NODE_CONST0_KEY], dtype=np.float32)
        # Add node to graph with type and feature (truth table)
        G.add_node(CONST0_NODE_ID, type=node_type_array, feature=zero_tt)
        logger.debug(f"Added Const0 node {CONST0_NODE_ID} with feature length {len(zero_tt)}")
    except KeyError:
         logger.error(f"Node type key '{NODE_CONST0_KEY}' not found in NODE_TYPE_ENCODING. Cannot add Const0 node.")
    except Exception as e:
         logger.error(f"Error adding Const0 node: {e}")


    # --- Add PI Nodes ---
    try:
        # Get PI node type encoding
        pi_node_type_array = np.array(NODE_TYPE_ENCODING[NODE_PI_KEY], dtype=np.float32)
        # Iterate through primary inputs of the AIG
        for idx, pi_node_id in enumerate(aig.pis()):
            # Ensure we have a corresponding input pattern
            if idx < len(input_patterns):
                # Create the truth table string for this PI
                binary_inputs = "".join(map(str, list(input_patterns[idx])))
                # Format (pad) the truth table
                pi_tt = format_tt(binary_inputs)
                # Add PI node to graph
                G.add_node(pi_node_id, type=pi_node_type_array, feature=pi_tt)
                logger.debug(f"Added PI node {pi_node_id} with feature length {len(pi_tt)}")
            else:
                # Log error if mismatch occurs (should not happen with correct generate_binary_inputs)
                logger.error(f"Mismatch between number of PIs ({num_inputs}) and generated input patterns ({len(input_patterns)}). Skipping PI node {pi_node_id}.")
                continue
    except KeyError:
         logger.error(f"Node type key '{NODE_PI_KEY}' not found in NODE_TYPE_ENCODING. Cannot add PI nodes.")
    except Exception as e:
         logger.error(f"Error adding PI nodes: {e}")


    # --- Add AND Gate Nodes ---
    try:
        # Get AND node type encoding
        and_node_type_array = np.array(NODE_TYPE_ENCODING[NODE_AND_KEY], dtype=np.float32)
        # Simulate internal nodes to get their truth tables
        n_to_tt = simulate_nodes(aig)
        # Iterate through AND gates of the AIG
        for gate_node_id in aig.gates():
            # Check if simulation result exists for this gate
            if gate_node_id in n_to_tt:
                # Get binary truth table string
                binary_truths = n_to_tt[gate_node_id].to_binary()
                # Format (pad) the truth table
                and_tt = format_tt(binary_truths)
                # Add AND node to graph
                G.add_node(gate_node_id, type=and_node_type_array, feature=and_tt)
                logger.debug(f"Added AND node {gate_node_id} with feature length {len(and_tt)}")
            else:
                # Warn if simulation result is missing for a gate
                logger.warning(f"Gate node {gate_node_id} not found in simulation results. Skipping node addition.")
    except KeyError:
         logger.error(f"Node type key '{NODE_AND_KEY}' not found in NODE_TYPE_ENCODING. Cannot add AND nodes.")
    except Exception as e:
        # Log any errors during simulation or node addition
        logger.error(f"Error during node simulation or adding AND gates: {e}", exc_info=True)

    return G


def get_outs(aig: Any, G: nx.DiGraph, size_before_pos: int) -> nx.DiGraph:
    """
    Add output nodes (PO) to the graph using config encodings.
    Node IDs for POs start after Const0, PIs, and AND gates.

    Args:
        aig: AIG object.
        G: NetworkX DiGraph.
        size_before_pos: Number of nodes before adding POs (Const0 + PIs + Gates).

    Returns:
        Updated DiGraph with output nodes added.
    """
    try:
        # Get number of inputs and outputs
        num_inputs = aig.num_pis()
        num_pos = aig.num_pos()
    except Exception as e:
        logger.error(f"Error getting PI/PO counts from AIG object: {e}. Cannot add PO nodes.")
        return G

    try:
        # Get PO node type encoding
        po_node_type_array = np.array(NODE_TYPE_ENCODING[NODE_PO_KEY], dtype=np.float32)
        # Simulate all primary outputs to get their truth tables
        tts = simulate(aig)

        # Check if the number of simulated TTs matches the number of POs
        if len(tts) != num_pos:
            logger.error(f"Mismatch between simulated truth tables ({len(tts)}) and number of POs ({num_pos}). Cannot reliably add PO nodes.")
            return G

        # Iterate through the primary outputs (using index and driver info)
        for ind, po_literal in enumerate(aig.pos()): # Use po_literal as it might not be just an ID
            # Determine the unique node ID for this PO node
            new_out_node_id = size_before_pos + ind
            # Get the binary truth table string for this output
            binary_truths = tts[ind].to_binary()

            # Format (pad) the PO node's feature truth table
            po_tt = get_padded_truth_table(binary_truths, num_inputs)

            # Add the PO node to the graph
            G.add_node(new_out_node_id,
                       type=po_node_type_array,
                       feature=po_tt)
            logger.debug(f"Added PO node {new_out_node_id} with feature length {len(po_tt)}")

            # Note: Edges connecting to these PO nodes are added in get_edges

    except KeyError:
         logger.error(f"Node type key '{NODE_PO_KEY}' not found in NODE_TYPE_ENCODING. Cannot add PO nodes.")
    except Exception as e:
        logger.error(f"Error during output simulation or adding PO nodes: {e}", exc_info=True)
    return G


def get_edges(aig: Any, G: nx.DiGraph) -> nx.DiGraph:
    """
    Add edges to the graph with one-hot encoded edge labels from config.
    This now runs AFTER all nodes (including POs) are added.
    Correctly handles PO driver resolution using aig.get_node/is_complemented.

    Args:
        aig: AIG object.
        G: NetworkX DiGraph (should contain all nodes at this point).

    Returns:
        Updated DiGraph with edges added.
    """
    logger.debug("Adding edges to the graph...")
    edges_added_count = 0
    try:
        # Get edge encodings from config and convert to numpy arrays
        edge_reg_label = np.array(EDGE_LABEL_ENCODING[EDGE_REG_KEY], dtype=np.float32)
        edge_inv_label = np.array(EDGE_LABEL_ENCODING[EDGE_INV_KEY], dtype=np.float32)

        # Get internal edges (between Const0, PIs, ANDs) using aigverse helper
        # Use weight 1 for inverted, 0 for regular
        internal_edges = to_edge_list(aig, inverted_weight=1, regular_weight=0)
        logger.debug(f"Retrieved {len(internal_edges)} internal edges from to_edge_list.")

        # --- Add internal edges (PI/AND -> AND) ---
        for e in internal_edges:
            # Check if source and target nodes exist in the graph before adding edge
            if not G.has_node(e.source):
                logger.warning(f"Source node {e.source} for internal edge ({e.source} -> {e.target}) does not exist. Skipping.")
                continue
            if not G.has_node(e.target):
                logger.warning(f"Target node {e.target} for internal edge ({e.source} -> {e.target}) does not exist. Skipping.")
                continue

            # Assign one-hot encoded edge labels based on weight (inversion)
            onehot_label = edge_inv_label if e.weight == 1 else edge_reg_label
            G.add_edge(e.source, e.target, type=onehot_label)
            edges_added_count += 1
            logger.debug(f"Added internal edge {e.source} -> {e.target} with type {'INV' if e.weight==1 else 'REG'}")

        # --- Add edges connecting to PO nodes ---
        # POs themselves are nodes. The 'edge' is from their driver node to the PO node.
        num_pis = aig.num_pis()
        num_gates = aig.num_gates()
        # Calculate the starting node ID for POs
        size_before_pos = 1 + num_pis + num_gates

        # Iterate through the primary outputs using enumerate to get index and the PO literal
        for ind, po_literal in enumerate(aig.pos()):
            # Calculate the node ID of the PO node itself
            po_node_id = size_before_pos + ind

            try:
                # *** CORRECTED LOGIC ***
                # Use aig methods to resolve the actual driver node ID and inversion status
                # This assumes po_literal is the object/representation expected by these methods
                driver_node_id = aig.get_node(po_literal)
                is_inverted = aig.is_complemented(po_literal)

                # Check if the resolved driver node exists in the graph
                if not G.has_node(driver_node_id):
                     logger.warning(f"Resolved driver node {driver_node_id} (from literal {po_literal}) for PO {po_node_id} does not exist in graph G. Skipping PO edge.")
                     continue
                # Check if the PO node itself exists (should have been added in get_outs)
                if not G.has_node(po_node_id):
                     logger.warning(f"PO node {po_node_id} itself does not exist. Skipping PO edge.")
                     continue

                # Add edge from the resolved driver to the PO node
                onehot_label = edge_inv_label if is_inverted else edge_reg_label
                G.add_edge(driver_node_id, po_node_id, type=onehot_label)
                edges_added_count += 1
                logger.debug(f"Added PO edge {driver_node_id} -> {po_node_id} with type {'INV' if is_inverted else 'REG'}")

            except Exception as po_edge_e:
                 # Log errors encountered while processing a specific PO's edge
                 logger.error(f"Error processing PO literal {po_literal} for PO node {po_node_id}: {po_edge_e}")
                 continue # Skip this edge on error

        logger.debug(f"Successfully added {edges_added_count} edges in total.")

    except KeyError as ke:
         # Handle missing keys in encoding dictionaries
         logger.error(f"Edge type key missing in EDGE_LABEL_ENCODING: {ke}. Cannot add edges.")
    except Exception as e:
        # Catch-all for other errors during edge processing
        logger.error(f"Error processing or adding edges: {e}", exc_info=True)
    return G


def get_condition(aig: Any, graph_size: int, pad: bool = True) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Get condition lists (truth tables for specific nodes) for the graph.
    'condition_list' has TTs only for inputs/outputs.
    'full_condition_list' has TTs for all nodes. Uses config padding.
    Adjusts list sizes to match the final graph_size.

    Args:
        aig: AIG object.
        graph_size: Expected final size of the graph (for adjustments).
        pad: Whether to pad truth tables.

    Returns:
        Tuple of (condition_list, full_condition_list). Returns ([], []) on critical error.
    """
    try:
        # Get AIG properties needed for simulation and list sizing
        num_inputs = aig.num_pis()
        num_pos = aig.num_pos()
        num_gates = aig.num_gates()
    except Exception as e:
        logger.error(f"Error getting PI/PO/Gate counts from AIG: {e}. Cannot generate conditions.")
        return [], []

    # Log if number of inputs exceeds configured maximum
    if num_inputs > MAX_INPUTS:
        logger.debug(f"get_condition: AIG has {num_inputs} inputs, exceeding MAX_INPUTS ({MAX_INPUTS}).")

    # Calculate effective number of inputs for TT length (capped by MAX_INPUTS)
    effective_inputs = min(num_inputs, MAX_INPUTS)
    # Calculate TT length based on effective inputs
    current_tt_len = 1 << effective_inputs if effective_inputs >= 0 else 0

    # Helper function to format truth table string (pad or just convert)
    def format_tt(binary_str):
        return get_padded_truth_table(binary_str, num_inputs) if pad else [int(b) for b in binary_str]

    # Generate zero TT based on effective length
    zero_tt_str = "0" * current_tt_len if current_tt_len > 0 else "0"
    zero_tt = format_tt(zero_tt_str) # Format the zero TT (padding etc.)

    # Initialize lists for conditions
    condition_list = []      # Sparse list (TTs for Const0, PIs, POs only)
    full_condition_list = [] # Dense list (TTs for all nodes)

    # 1. Const0 Node (Node ID 0)
    condition_list.append(zero_tt)
    full_condition_list.append(zero_tt)

    # 2. PI Nodes (Node IDs 1 to num_inputs)
    input_patterns = list(zip(*generate_binary_inputs(num_inputs)))
    for idx, pi_node_id in enumerate(aig.pis()):
        if idx < len(input_patterns):
             # Generate TT string from input patterns
             binary_inputs = "".join(map(str, list(input_patterns[idx])))
             # Format the TT
             tt = format_tt(binary_inputs)
             # Append to both lists
             condition_list.append(tt)
             full_condition_list.append(tt)
        else:
             # Handle potential mismatch (shouldn't happen)
             logger.error(f"Mismatch between PIs ({num_inputs}) and patterns ({len(input_patterns)}) in get_condition. Appending zero TT for PI {pi_node_id}.")
             condition_list.append(zero_tt)
             full_condition_list.append(zero_tt)

    # 3. AND Gate Nodes (Node IDs num_inputs+1 to num_inputs+num_gates)
    try:
        # Simulate internal nodes
        n_to_tt = simulate_nodes(aig)
        # Iterate through gates
        for gate_node_id in aig.gates():
             if gate_node_id in n_to_tt:
                # Get and format TT if simulation result exists
                binary_t = n_to_tt[gate_node_id].to_binary()
                tt = format_tt(binary_t)
                full_condition_list.append(tt) # Actual TT for full list
                condition_list.append(zero_tt)   # Placeholder (zero TT) for sparse list
             else:
                 # Append placeholders if simulation result missing
                 logger.warning(f"Gate node {gate_node_id} not found in simulation results during condition generation. Appending zero TT.")
                 full_condition_list.append(zero_tt)
                 condition_list.append(zero_tt)
    except Exception as e:
        logger.error(f"Error simulating nodes for condition lists: {e}")
        # Append placeholders if simulation fails mid-way to maintain structure
        num_gates_added = len(full_condition_list) - (1 + num_inputs) # Nodes added so far
        num_gates_missing = num_gates - num_gates_added
        if num_gates_missing > 0:
            logger.warning(f"Appending {num_gates_missing} zero TTs for gates due to simulation error.")
            full_condition_list.extend([zero_tt] * num_gates_missing)
            condition_list.extend([zero_tt] * num_gates_missing)


    # 4. PO Nodes (Node IDs num_inputs+num_gates+1 to num_inputs+num_gates+num_pos)
    try:
        # Simulate primary outputs
        tts = simulate(aig)
        # Check for mismatch between simulation results and expected number of POs
        if len(tts) != num_pos:
             logger.error(f"Mismatch simulated TTs ({len(tts)}) and POs ({num_pos}) in get_condition.")
             num_missing = num_pos - len(tts)
             # Pad with placeholders if simulation returned too few TTs
             if num_missing > 0:
                 logger.warning(f"Padding condition lists with {num_missing} zero TTs for POs due to simulation mismatch.")
                 condition_list.extend([zero_tt] * num_missing)
                 full_condition_list.extend([zero_tt] * num_missing)
             # Truncate if simulation returned too many TTs (less likely)
             tts = tts[:num_pos]

        # Add TTs for the outputs we have simulations for
        for tt_simulated in tts:
            binary_truths = tt_simulated.to_binary()
            tt = format_tt(binary_truths)
            condition_list.append(tt) # Actual TT for sparse list
            full_condition_list.append(tt) # Actual TT for full list

    except Exception as e:
        logger.error(f"Error simulating outputs for condition lists: {e}")
        # Append placeholders if simulation fails mid-way
        num_pos_added = len(full_condition_list) - (1 + num_inputs + num_gates)
        num_pos_missing = num_pos - num_pos_added
        if num_pos_missing > 0:
            logger.warning(f"Appending {num_pos_missing} zero TTs for POs due to simulation error.")
            condition_list.extend([zero_tt] * num_pos_missing)
            full_condition_list.extend([zero_tt] * num_pos_missing)


    # Final size adjustment relative to the *provided* graph_size
    # This handles cases where node addition might have failed or graph_size was calculated differently
    current_len = len(full_condition_list)
    if current_len != graph_size:
         logger.warning(f"Condition list length ({current_len}) differs from target graph_size ({graph_size}). Adjusting...")
         # Pad with zero TT if lists are too short
         while len(condition_list) < graph_size:
             condition_list.append(zero_tt)
         while len(full_condition_list) < graph_size:
             full_condition_list.append(zero_tt)
         # Truncate if lists are too long
         condition_list = condition_list[:graph_size]
         full_condition_list = full_condition_list[:graph_size]

    # Final check of list lengths against target size
    if len(condition_list) != graph_size:
         logger.error(f"Final condition_list length ({len(condition_list)}) still does not match target graph_size ({graph_size}) after adjustments.")
    if len(full_condition_list) != graph_size:
         logger.error(f"Final full_condition_list length ({len(full_condition_list)}) still does not match target graph_size ({graph_size}) after adjustments.")

    return condition_list, full_condition_list


def get_graph(aig: Any, expected_graph_size: int, pad: bool = True) -> nx.DiGraph | None:
    """
    Create a complete graph representation of the AIG using config settings.
    Adds all nodes first, then all edges. Includes graph metadata.

    Args:
        aig: AIG object.
        expected_graph_size: Expected final number of nodes (for initial checks).
        pad: Whether to pad truth tables (applies to features and conditions).

    Returns:
        NetworkX DiGraph representing the AIG, or None if a critical error occurs.
    """
    # Log start of graph creation for a specific AIG
    try:
        logger.debug(f"Creating graph for AIG with {aig.num_pis()} PIs, {aig.num_gates()} Gates, {aig.num_pos()} POs.")
        num_inputs = aig.num_pis()
        num_gates = aig.num_gates()
        num_pos = aig.num_pos()
    except Exception as e:
         logger.error(f"Error getting basic AIG properties: {e}. Cannot create graph.")
         return None

    try:
        # Simulate outputs once for storing patterns in graph metadata
        simulated_outputs = simulate(aig)
        padded_output_patterns = []
        if len(simulated_outputs) == num_pos:
             # Pad each output TT and store
             for tt in simulated_outputs:
                 binary_str = tt.to_binary()
                 padded_tt = get_padded_truth_table(binary_str, num_inputs)
                 padded_output_patterns.append(padded_tt)
             logger.debug(f"Generated {len(padded_output_patterns)} padded output patterns.")
        else:
             # Log error if simulation results don't match expected PO count
             logger.error(f"Output pattern simulation mismatch ({len(simulated_outputs)} vs {num_pos}), storing empty list for output_patterns.")

        # Initialize the graph with metadata (defer conditions until after node addition)
        G = nx.DiGraph(
            inputs=num_inputs,
            outputs=num_pos,
            gates = num_gates,
            tts=None, # Placeholder for sparse condition list
            tts_with_ss=None, # Placeholder for full condition list
            output_patterns=padded_output_patterns # Store the padded patterns
        )

        # Calculate size before adding POs for node ID assignment
        # Node ID order: 0 (Const0), PIs, ANDs, POs
        size_before_pos = 1 + num_inputs + num_gates

        # --- Node Addition ---
        # Add Const0, PI, and AND nodes
        G = get_nodes(aig, G, pad=pad)
        # Add PO nodes
        G = get_outs(aig, G, size_before_pos)

        # --- Edge Addition ---
        # Add edges between all existing nodes (including PO connections)
        G = get_edges(aig, G)

        # --- Final Checks & Condition Addition ---
        # Get the actual number of nodes added to the graph
        actual_nodes = G.number_of_nodes()
        # Use the actual node count for generating condition lists
        final_graph_size_for_conditions = actual_nodes

        # Log a warning if the actual node count differs from the initially expected size
        if actual_nodes != expected_graph_size:
             logger.warning(f"Final graph node count mismatch: Expected {expected_graph_size}, Got {actual_nodes}. Using actual count ({actual_nodes}) for condition lists.")

        # --- Add Condition Lists (using the actual graph size) ---
        try:
            # Generate the condition lists based on the final graph size
            condition, full_condition = get_condition(aig, final_graph_size_for_conditions, pad=pad)
            # Assign lists to graph attributes if successfully generated
            if condition is not None and full_condition is not None:
                G.graph['tts'] = condition
                G.graph['tts_with_ss'] = full_condition
                logger.debug(f"Added condition lists with actual size {final_graph_size_for_conditions}")
            else:
                 # Log error if condition list generation failed
                 logger.error("Condition list generation failed, 'tts' and 'tts_with_ss' attributes will be None or empty.")
        except Exception as cond_e:
            # Log errors during condition list generation/assignment
            logger.error(f"Failed to generate or add condition lists after graph creation: {cond_e}", exc_info=True)


        # Optional: Verify counts of each node type added vs. AIG properties
        # ... (detailed counts check code could be added here) ...

        return G

    except Exception as e:
        # Catch-all for critical errors during graph creation process
        logger.error(f"Critical error during graph creation for AIG: {e}", exc_info=True)
        return None


def find_all_aig_files(root_dir: str) -> List[str]:
    """
    Recursively find all .aig and .aag files in the given directory.

    Args:
        root_dir: Root directory to start the search from.

    Returns:
        List of paths to all found AIGER files.
    """
    aig_files = []
    abs_root_dir = os.path.abspath(root_dir) # Use absolute path for checking
    logger.info(f"Searching for .aig and .aag files in '{abs_root_dir}'...")
    # Check if the provided root directory is valid
    if not os.path.isdir(abs_root_dir):
        logger.error(f"Provided root directory '{abs_root_dir}' is not a valid directory or is inaccessible.")
        return []
    try:
        # Walk through the directory tree
        for dirpath, _, filenames in os.walk(abs_root_dir):
            for filename in filenames:
                # Check if filename ends with .aig or .aag (case-insensitive)
                if filename.lower().endswith(('.aig', '.aag')):
                    # Construct full path and add to list
                    full_path = os.path.join(dirpath, filename)
                    aig_files.append(full_path)
    except Exception as e:
        logger.error(f"Error walking directory {abs_root_dir}: {e}")
        return []
    # **ADDED LOGGING**
    logger.info(f"Finished search. Found {len(aig_files)} '.aig' or '.aag' files in '{abs_root_dir}'.")
    return aig_files


def main():
    """Main function to process AIG files using config and create graph dataset."""
    start_time = time.time() # Record start time for duration calculation
    current_chunk_graphs = [] # List to store graphs for the current chunk
    current_chunk_num = 1     # 1-based index for the current output file chunk
    accepted_in_chunk = 0     # Counter for accepted graphs within the current chunk
    total_processed = 0       # Counter for total files scanned
    total_accepted = 0        # Counter for graphs that pass filters
    # Counter for different rejection reasons and warnings
    issue_reasons = Counter({
        "too_many_inputs": 0, "too_many_outputs": 0, "too_many_nodes": 0, # Size filter rejections
        "load_error": 0, "graph_creation_error": 0, "aig_property_error": 0, # Processing errors
        "final_node_count_mismatch": 0, # Rejection if final size > max_nodes
        "warn_not_a_dag": 0,             # Warning: Graph is not a DAG
        "warn_isolated_nodes": 0,        # Warning: Graph has isolated nodes (excluding Const0)
        "warn_node_count_discrepancy": 0 # Warning: Final node count != expected, but within limits
    })
    # Dictionary to store statistics per subfolder {subfolder_name: Counter}
    subfolder_stats = {}

    # --- Use Configured Constraints and Hardcoded Path ---
    input_dir = INPUT_PATH
    output_dir = OUTPUT_DIR
    output_base_name = OUTPUT_FILENAME_BASE
    stats_file_path = STATS_FILE_PATH
    # Use chunking config defined globally
    num_output_files = NUM_OUTPUT_FILES
    chunk_size = CHUNK_SIZE
    # Filter limits
    max_inputs = MAX_INPUTS
    max_outputs = MAX_OUTPUTS
    max_nodes = MAX_SIZE
    # ----------------------------------------------------

    # --- Log Initial Configuration to String Buffer ---
    config_log = io.StringIO()
    print("--- Configuration ---", file=config_log)
    print(f"Input Directory : {os.path.abspath(input_dir)}", file=config_log) # Log absolute path
    print(f"Output Directory: {os.path.abspath(output_dir)}", file=config_log) # Log absolute path
    print(f"Output Base Name: {output_base_name}", file=config_log)
    print(f"Target Files    : {num_output_files}", file=config_log)
    print(f"Graphs per File : {chunk_size} (approx)", file=config_log)
    print(f"Stats File      : {os.path.abspath(stats_file_path)}", file=config_log) # Log absolute path
    print(f"Max Inputs      : {max_inputs}", file=config_log)
    print(f"Max Outputs     : {max_outputs}", file=config_log)
    print(f"Max Nodes       : {max_nodes}", file=config_log)
    print(f"Max TT Length   : {MAX_TT_LENGTH if MAX_TT_LENGTH != float('inf') else 'Infinite'}", file=config_log)
    print(f"TT Pad Value    : {TT_PADDING_VALUE}", file=config_log)
    print("-" * 20, file=config_log)
    config_summary = config_log.getvalue() # Get string from buffer
    # Log the configuration summary
    logger.info(f"Starting AIG processing with configuration:\n{config_summary}")


    # --- Find and Shuffle AIG files ---
    all_aig_files = find_all_aig_files(input_dir)
    # **ADDED CHECK AND LOGGING**
    if not all_aig_files:
        logger.error(f"CRITICAL: No .aig or .aag files found in the specified input directory: {os.path.abspath(input_dir)}. Please check the INPUT_PATH variable and the directory contents.")
        logger.error("Exiting script.")
        # Save just the config summary if no files found
        save_stats_to_file(config_summary + f"\nCRITICAL: No .aig or .aag files found in {os.path.abspath(input_dir)}. No processing done.", stats_file_path)
        return # Exit early

    logger.info(f"Successfully found {len(all_aig_files)} files. Shuffling file order...")
    random.shuffle(all_aig_files) # Shuffle the list in place
    logger.info("File order shuffled. Starting processing loop...")

    # --- File Processing Loop ---
    total_files_to_process = len(all_aig_files)
    # **ADDED LOGGING**
    logger.info(f"Beginning processing of {total_files_to_process} files...")
    for file_idx, file_path in enumerate(all_aig_files):
        # **ADDED LOGGING**
        logger.debug(f"--- Processing file {file_idx+1}/{total_files_to_process}: {os.path.basename(file_path)} ---")
        try:
            # Use absolute path for relpath calculation base for robustness
            abs_input_dir = os.path.abspath(input_dir)
            rel_path = os.path.relpath(file_path, abs_input_dir)
            subfolder = os.path.dirname(rel_path) if os.path.dirname(rel_path) else "root"
        except ValueError:
             subfolder = "unknown_relative_path"
             logger.warning(f"Could not determine relative path for {file_path} against {abs_input_dir}")

        subfolder_stats.setdefault(subfolder, Counter({"total": 0, "accepted": 0, "rejected": 0}))
        subfolder_stats[subfolder]["total"] += 1

        if file_idx > 0 and file_idx % 1000 == 0: # Log progress
            elapsed_time = time.time() - start_time
            files_per_second = file_idx / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"Progress: Processed {file_idx}/{total_files_to_process} files. "
                        f"Accepted {total_accepted} graphs (current chunk: {accepted_in_chunk}/{chunk_size}). "
                        f"Saved {current_chunk_num-1}/{num_output_files} chunks. "
                        f"({files_per_second:.1f} files/sec)")

        total_processed += 1
        rejected = False
        aig = None
        Graph = None
        expected_graph_size = -1

        # --- Step 1: Load AIG ---
        try:
            aig = read_aiger_into_aig(file_path)
            logger.debug(f"Successfully loaded: {os.path.basename(file_path)}")
        except Exception as e:
            if issue_reasons["load_error"] < 5 or issue_reasons["load_error"] % 100 == 0:
                logger.warning(f"Failed to load {file_path}: {e}")
            issue_reasons["load_error"] += 1
            rejected = True

        # --- Step 2: Apply Size/IO Filters ---
        if not rejected and aig:
            try:
                num_pis, num_pos, num_gates = aig.num_pis(), aig.num_pos(), aig.num_gates()
                expected_graph_size = 1 + num_pis + num_gates + num_pos

                if num_pis > max_inputs:
                    logger.debug(f"Rejected {os.path.basename(file_path)}: Too many inputs ({num_pis} > {max_inputs})")
                    issue_reasons["too_many_inputs"] += 1; rejected = True
                elif num_pos > max_outputs:
                    logger.debug(f"Rejected {os.path.basename(file_path)}: Too many outputs ({num_pos} > {max_outputs})")
                    issue_reasons["too_many_outputs"] += 1; rejected = True
                elif expected_graph_size > max_nodes:
                    logger.debug(f"Rejected {os.path.basename(file_path)}: Expected node count too high ({expected_graph_size} > {max_nodes})")
                    issue_reasons["too_many_nodes"] += 1; rejected = True
            except Exception as filter_e:
                 logger.error(f"Error getting properties from AIG {os.path.basename(file_path)} for filtering: {filter_e}")
                 issue_reasons["aig_property_error"] += 1
                 rejected = True
        elif not rejected and aig is None:
             logger.error(f"AIG object is None for {file_path} after supposed successful load. Rejecting.")
             issue_reasons["load_error"] += 1
             rejected = True

        # --- Step 3: Create Graph ---
        if not rejected and aig and expected_graph_size != -1:
            Graph = get_graph(aig, expected_graph_size, pad=True)
            if Graph is None:
                 logger.warning(f"Graph creation failed for {os.path.basename(file_path)}.")
                 issue_reasons["graph_creation_error"] += 1; rejected = True
            else:
                 # --- Step 3a: Check final node count ---
                 actual_node_count = Graph.number_of_nodes()
                 if actual_node_count > max_nodes:
                     logger.debug(f"Rejected {os.path.basename(file_path)}: Final graph node count too high ({actual_node_count} > {max_nodes}) after creation.")
                     issue_reasons["final_node_count_mismatch"] += 1; rejected = True
                 elif actual_node_count != expected_graph_size:
                      issue_reasons["warn_node_count_discrepancy"] += 1

                 # --- Step 3b: DAG Check (Warning only) ---
                 if not rejected:
                     try:
                         if not nx.is_directed_acyclic_graph(Graph):
                             logger.warning(f"Graph for {os.path.basename(file_path)} is not a DAG.")
                             issue_reasons["warn_not_a_dag"] += 1
                     except Exception as dag_e:
                         logger.error(f"Error during DAG check for {os.path.basename(file_path)}: {dag_e}")
                         issue_reasons["warn_not_a_dag"] += 1 # Count error as warning

                 # --- Step 3c: Isolated Node Check (Warning only) ---
                 if not rejected:
                     try:
                         isolates = list(nx.isolates(Graph))
                         other_isolates = [node for node in isolates if node != CONST0_NODE_ID]
                         if other_isolates:
                             logger.warning(f"Graph for {os.path.basename(file_path)} has isolated nodes (excluding {CONST0_NODE_ID}): {other_isolates}")
                             issue_reasons["warn_isolated_nodes"] += 1
                     except Exception as iso_e:
                         logger.error(f"Error during isolated node check for {os.path.basename(file_path)}: {iso_e}")
                         issue_reasons["warn_isolated_nodes"] += 1 # Count error as warning


        # --- Step 4: Final Decision & Chunk Management ---
        if rejected:
            subfolder_stats[subfolder]["rejected"] += 1
            logger.debug(f"Rejected file: {os.path.basename(file_path)}")
        elif Graph is not None:
            # --- Graph Accepted ---
            current_chunk_graphs.append(Graph) # Add accepted graph to current chunk
            total_accepted += 1
            accepted_in_chunk += 1
            subfolder_stats[subfolder]["accepted"] += 1
            logger.debug(f"Accepted and added graph for: {os.path.basename(file_path)} (Total Accepted: {total_accepted}, Chunk {current_chunk_num}: {accepted_in_chunk}/{chunk_size})")

            # --- Check if Chunk is Full ---
            # Save if chunk is full AND we haven't already saved the maximum number of chunks
            if accepted_in_chunk >= chunk_size and current_chunk_num <= num_output_files:
                save_graph_chunk(current_chunk_graphs, output_dir, output_base_name, current_chunk_num, num_output_files)
                # Reset for next chunk
                current_chunk_graphs = []
                accepted_in_chunk = 0
                current_chunk_num += 1

        else: # Handle unexpected cases where Graph is None but not rejected
             if not rejected:
                 logger.error(f"File {os.path.basename(file_path)} was not rejected, but Graph object is None. Counting as graph_creation_error.")
                 issue_reasons["graph_creation_error"] += 1
                 subfolder_stats[subfolder]["rejected"] += 1
                 logger.debug(f"Rejected file (graph creation error): {os.path.basename(file_path)}")


    # --- After the Loop ---
    # Save any remaining graphs in the last chunk, if any exist and we haven't exceeded the target file count
    if current_chunk_graphs and current_chunk_num <= num_output_files:
        logger.info(f"Saving final chunk {current_chunk_num} with {len(current_chunk_graphs)} graphs...")
        save_graph_chunk(current_chunk_graphs, output_dir, output_base_name, current_chunk_num, num_output_files)
    elif current_chunk_graphs:
         logger.warning(f"There are {len(current_chunk_graphs)} remaining graphs, but the target number of files ({num_output_files}) has already been saved. These graphs will not be saved.")


    # --- Processing Finished ---
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"\n--- Processing Summary ---")
    logger.info(f"Processing complete in {total_time:.2f} seconds")
    logger.info(f"Total files scanned   : {total_processed}")
    logger.info(f"Total graphs accepted : {total_accepted}")
    final_chunks_saved = current_chunk_num -1 if not current_chunk_graphs or current_chunk_num > num_output_files else current_chunk_num
    logger.info(f"Chunks saved          : {final_chunks_saved} / {num_output_files}") # Log final chunk count

    # --- Prepare Statistics String for File ---
    stats_output_buffer = io.StringIO()
    print(config_summary, file=stats_output_buffer)

    # --- Processing Summary ---
    print("\n--- PROCESSING SUMMARY ---", file=stats_output_buffer)
    print(f"Processing complete in: {total_time:.2f} seconds", file=stats_output_buffer)
    print(f"Total files scanned   : {total_processed}", file=stats_output_buffer)
    print(f"Total graphs accepted : {total_accepted}", file=stats_output_buffer)
    # final_chunks_saved calculation moved up for logging consistency
    print(f"Chunks saved          : {final_chunks_saved} / {num_output_files}", file=stats_output_buffer)
    total_rejected_final = sum(count for reason, count in issue_reasons.items() if not reason.startswith("warn_"))
    print(f"Total files rejected  : {total_rejected_final}", file=stats_output_buffer)
    total_warnings_final = issue_reasons['warn_not_a_dag'] + issue_reasons['warn_isolated_nodes'] + issue_reasons['warn_node_count_discrepancy']
    print(f"Total warnings issued : {total_warnings_final}", file=stats_output_buffer)


    # --- Issue Statistics (Rejections and Warnings) ---
    print("\n=== ISSUE STATISTICS (Rejections and Warnings) ===", file=stats_output_buffer)
    print(f"Breakdown:")
    if total_processed > 0:
        for reason, count in sorted(issue_reasons.items()):
            if count > 0:
                percentage = (count / total_processed) * 100
                issue_type = "Warning" if reason.startswith("warn_") else "Reject"
                print(f"  - {reason:<30}: {count:<7} ({percentage:.1f}%) [{issue_type}]", file=stats_output_buffer)
    else: print("  No files were processed.", file=stats_output_buffer)


    # --- Subfolder Statistics ---
    print("\n=== SUBFOLDER STATISTICS ===", file=stats_output_buffer)
    if not subfolder_stats: print("  No subfolders processed or stats collected.", file=stats_output_buffer)
    else:
        for subfolder, stats_counter in sorted(subfolder_stats.items()):
            total = stats_counter.get("total", 0)
            accepted = stats_counter.get("accepted", 0)
            rejected = stats_counter.get("rejected", 0)
            acceptance_rate = (accepted / total) * 100 if total > 0 else 0
            print(f"  {subfolder:<40}: Processed={total:<6}, Accepted={accepted:<6}, Rejected={rejected:<6} ({acceptance_rate:.1f}% acceptance)", file=stats_output_buffer)


    # --- Overall Dataset Statistics (Removed) ---
    # We cannot calculate detailed stats (min/max/avg/median) easily without loading all graphs.
    # Reporting only counts based summary.
    print("\n--- STATISTICS FOR ACCEPTED GRAPH DATASET ---", file=stats_output_buffer)
    if total_accepted > 0:
         print(f"\nDetailed dataset statistics (min/max/avg/median) were not calculated due to chunked processing.", file=stats_output_buffer)
         print(f"Refer to the processing summary above for total accepted graph count.", file=stats_output_buffer)
    else:
        # logger.warning already handled if total_accepted is 0
        print("\nNo graphs were accepted. No dataset statistics available.", file=stats_output_buffer)


    # --- Save the Combined Statistics to File ---
    final_stats_content = stats_output_buffer.getvalue()
    save_stats_to_file(final_stats_content, stats_file_path)

    logger.info("Script finished.")


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()
