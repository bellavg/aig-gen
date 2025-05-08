import os
import sys
import pickle
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import logging
import time

# --- Configuration ---

# !!! IMPORTANT: SET THIS TO YOUR DATASET ROOT DIRECTORY !!!
# This directory should contain 'raw/' (with your .pkl files)
# and will contain the created 'processed/' directory.
DATASET_ROOT_DIR = "./" # Example: Adjust this path

RAW_DIR = os.path.join(DATASET_ROOT_DIR, 'raw_aigs')
PROCESSED_DIR = os.path.join("./digress_dataset", 'processed')
RAW_FILENAME_BASE = "real_aigs"
TOTAL_PARTS = 6

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aig_preprocessor")

# --- Attempt to import AIG config ---
try:
    # Assuming the script is run from a location where G2PT is accessible
    # Or that G2PT is in the Python path. Adjust if necessary.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    g2pt_parent_dir = os.path.dirname(script_dir) # Go one level up potentially
    if g2pt_parent_dir not in sys.path:
        sys.path.insert(0, g2pt_parent_dir)
        logger.info(f"Added '{g2pt_parent_dir}' to sys.path for G2PT import.")

    import G2PT.configs.aig as aig_cfg
    logger.info("Successfully imported G2PT.configs.aig")
except ImportError as e:
    logger.error(f"Fatal Error: Could not import G2PT.configs.aig: {e}")
    logger.error("Ensure the G2PT package is installed or accessible in your PYTHONPATH.")
    logger.error("Using fallback config - this might lead to incorrect processing!")
    # Fallback definitions (should not be relied upon in production)
    class FallbackAigCfg:
        NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
        EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]
        NODE_TYPE_ENCODING = {
            "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], "NODE_PI": [0.0, 1.0, 0.0, 0.0],
            "NODE_AND": [0.0, 0.0, 1.0, 0.0], "NODE_PO": [0.0, 0.0, 0.0, 1.0]
        }
        EDGE_LABEL_ENCODING = {"EDGE_REG": [1.0, 0.0], "EDGE_INV": [0.0, 1.0]}
    aig_cfg = FallbackAigCfg()
except AttributeError as e:
     logger.error(f"Fatal Error: Attribute missing from G2PT.configs.aig: {e}")
     logger.error("Please ensure your aig.py config file is complete.")
     sys.exit(1)


# --- Mappings (derived from imported config) ---
try:
    NODE_TYPE_TO_INDEX = {key: i for i, key in enumerate(aig_cfg.NODE_TYPE_KEYS)}
    NUM_NODE_TYPES = len(aig_cfg.NODE_TYPE_KEYS)
    EDGE_TYPE_TO_INDEX = {key: i + 1 for i, key in enumerate(aig_cfg.EDGE_TYPE_KEYS)}
    NUM_EDGE_TYPES = len(aig_cfg.EDGE_TYPE_KEYS) + 1 # +1 for "no edge"

    # Node features: Based on the order defined in aig_cfg.NODE_TYPE_KEYS
    node_encoding_map = {
        NODE_TYPE_TO_INDEX[aig_cfg.NODE_TYPE_KEYS[0]]: [1., 0., 0., 0.], # CONST0
        NODE_TYPE_TO_INDEX[aig_cfg.NODE_TYPE_KEYS[1]]: [0., 1., 0., 0.], # PI
        NODE_TYPE_TO_INDEX[aig_cfg.NODE_TYPE_KEYS[2]]: [0., 0., 1., 0.], # AND
        NODE_TYPE_TO_INDEX[aig_cfg.NODE_TYPE_KEYS[3]]: [0., 0., 0., 1.]  # PO
    }

    # Edge features: Index 0 = NoEdge, Index 1 = REG, Index 2 = INV
    edge_encoding_map = {
        0:                                      [1., 0., 0.], # No edge
        EDGE_TYPE_TO_INDEX[aig_cfg.EDGE_TYPE_KEYS[0]]: [0., 1., 0.], # REG
        EDGE_TYPE_TO_INDEX[aig_cfg.EDGE_TYPE_KEYS[1]]: [0., 0., 1.]  # INV
    }
    logger.info("Derived node/edge mappings and encodings from config.")

except Exception as config_e:
     logger.error(f"Fatal Error: Could not derive mappings from aig_cfg: {config_e}")
     sys.exit(1)


# --- Helper Function to Convert NX to PyG Data ---
def convert_nx_to_pyg(nx_graph: nx.DiGraph, graph_index: int) -> Data | None:
    """Converts a single NetworkX AIG graph to a PyG Data object."""
    try:
        # Node Features
        node_features = []
        node_mapping = {}
        current_index = 0
        # Ensure consistent node order if node IDs aren't sequential 0-based
        sorted_nodes = sorted(list(nx_graph.nodes(data=True)), key=lambda x: x[0])

        for node_id, node_data in sorted_nodes:
            one_hot_type = node_data.get('type')
            if one_hot_type is None:
                raise ValueError(f"Node {node_id} missing 'type' attribute")

            # Convert numpy array/list to tuple for dict lookup
            one_hot_tuple = tuple(one_hot_type)
            node_type_key = next((k for k, v in aig_cfg.NODE_TYPE_ENCODING.items() if tuple(v) == one_hot_tuple), None)
            if node_type_key is None:
                raise ValueError(f"Unknown node type encoding {one_hot_type} for node {node_id}")

            node_type_index = NODE_TYPE_TO_INDEX.get(node_type_key)
            if node_type_index is None:
                raise ValueError(f"Node type key '{node_type_key}' not mapped to index")

            encoded_feature = node_encoding_map.get(node_type_index)
            if encoded_feature is None:
                raise ValueError(f"Encoding not found for node type index {node_type_index}")

            node_features.append(encoded_feature)
            node_mapping[node_id] = current_index
            current_index += 1

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge Index and Features
        edge_source, edge_target, edge_features = [], [], []
        for u, v, edge_data in nx_graph.edges(data=True):
            mapped_u, mapped_v = node_mapping.get(u), node_mapping.get(v)
            if mapped_u is None or mapped_v is None:
                logger.warning(f"Graph {graph_index}: Edge ({u}->{v}) has node not in mapping. Skipping edge.")
                continue

            edge_source.append(mapped_u)
            edge_target.append(mapped_v)

            one_hot_edge_type = edge_data.get('type')
            if one_hot_edge_type is None:
                raise ValueError(f"Edge ({u},{v}) missing 'type' attribute")

            one_hot_tuple = tuple(one_hot_edge_type)
            edge_type_key = next((k for k, v in aig_cfg.EDGE_LABEL_ENCODING.items() if tuple(v) == one_hot_tuple), None)
            # Assume REG if key not found (or handle stricter)
            if edge_type_key is None:
                 logger.warning(f"Graph {graph_index}: Unknown edge encoding {one_hot_edge_type}. Assuming REGULAR.")
                 edge_type_key = aig_cfg.EDGE_TYPE_KEYS[0] # Assumes REG is first key

            edge_type_index = EDGE_TYPE_TO_INDEX.get(edge_type_key)
            if edge_type_index is None:
                 raise ValueError(f"Edge type key '{edge_type_key}' not mapped to index")

            encoded_edge_feature = edge_encoding_map.get(edge_type_index)
            if encoded_edge_feature is None:
                 raise ValueError(f"Encoding not found for edge type index {edge_type_index}")
            edge_features.append(encoded_edge_feature)

        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Global Features
        y = torch.zeros((1, 0), dtype=torch.float) # No global features for AIGs typically

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=graph_index)

    except Exception as e:
        logger.error(f"Error converting NX graph {graph_index} to PyG Data: {e}")
        return None


# --- Main Preprocessing Logic ---
def main():
    start_time = time.time()
    logger.info(f"Starting AIG dataset preprocessing.")
    logger.info(f"Raw data directory: {RAW_DIR}")
    logger.info(f"Processed data directory: {PROCESSED_DIR}")

    split_definitions = {
        'train': [f"{RAW_FILENAME_BASE}_part_{i}_of_{TOTAL_PARTS}.pkl" for i in range(1, 5)],
        'val': [f"{RAW_FILENAME_BASE}_part_6_of_{TOTAL_PARTS}.pkl"],
        'test': [f"{RAW_FILENAME_BASE}_part_5_of_{TOTAL_PARTS}.pkl"]
    }

    global_graph_idx = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    total_errors = 0

    # Process each split
    for split, raw_filenames in split_definitions.items():
        logger.info(f"\n--- Processing split: {split} ---")
        split_processed_dir = os.path.join(PROCESSED_DIR, split)
        os.makedirs(split_processed_dir, exist_ok=True)
        logger.info(f"Output directory for this split: {split_processed_dir}")

        split_graph_count = 0
        split_error_count = 0

        # Process raw files for this split
        for filename in raw_filenames:
            raw_path = os.path.join(RAW_DIR, filename)
            logger.info(f"Loading raw file: {raw_path}...")

            if not os.path.exists(raw_path):
                logger.warning(f"File not found: {raw_path}. Skipping.")
                continue

            try:
                with open(raw_path, 'rb') as f:
                    graph_chunk = pickle.load(f)
                logger.info(f"Loaded {len(graph_chunk)} graphs from {filename}.")
            except Exception as e:
                logger.error(f"Error loading {raw_path}: {e}. Skipping file.")
                total_errors += 1 # Count file load error
                continue

            # Convert and save graphs in the chunk
            for nx_graph in tqdm(graph_chunk, desc=f"Converting graphs from {filename}"):
                data_object = convert_nx_to_pyg(nx_graph, global_graph_idx)

                if data_object is not None:
                    try:
                        save_path = os.path.join(split_processed_dir, f'graph_{global_graph_idx}.pt')
                        torch.save(data_object, save_path)
                        global_graph_idx += 1
                        split_graph_count += 1
                    except Exception as save_e:
                        logger.error(f"Error saving processed graph {global_graph_idx}: {save_e}")
                        split_error_count += 1
                else:
                    # Error occurred during conversion (already logged in helper)
                    split_error_count += 1

        logger.info(f"Finished processing split '{split}'.")
        logger.info(f"  Successfully saved graphs: {split_graph_count}")
        if split_error_count > 0:
            logger.warning(f"  Errors encountered (graphs skipped): {split_error_count}")
        split_counts[split] = split_graph_count
        total_errors += split_error_count

    # --- Final Summary ---
    end_time = time.time()
    logger.info("\n--- Preprocessing Summary ---")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Total graphs processed and saved: {global_graph_idx}")
    logger.info(f"  Train split graphs: {split_counts['train']}")
    logger.info(f"  Validation split graphs: {split_counts['val']}")
    logger.info(f"  Test split graphs: {split_counts['test']}")
    if total_errors > 0:
        logger.warning(f"Total errors encountered (files/graphs skipped): {total_errors}")
    logger.info(f"Processed data saved in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()