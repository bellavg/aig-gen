# G2PT/configs/aig.py
import math
import os # Added for potential path joining if needed
import networkx as nx
from typing import Union # Added for type hinting

# --- Primary Configuration Constants ---
dataset = 'aig'

# Consider using os.path.join for better path handling
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets G2PT directory
# data_dir = os.path.join(base_dir, 'datasets', 'aig')
# tokenizer_path = os.path.join(base_dir, 'tokenizers', 'aig')
#

# AIG constraint constants
MAX_NODE_COUNT = 64
MIN_PI_COUNT = 2
MAX_PI_COUNT = 8
MIN_PO_COUNT = 1
MAX_PO_COUNT = 8
MIN_AND_COUNT = 1 # Assuming at least one AND gate needed



NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]



# Derive feature counts from the size of the derived vocabularies
NUM_NODE_FEATURES = len(NODE_TYPE_KEYS) # Should be 4
NUM_EDGE_FEATURES = len(EDGE_TYPE_KEYS) # Should be 2


# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], # Index 0 feature
    "NODE_PI":     [0.0, 1.0, 0.0, 0.0], # Index 1 feature
    "NODE_AND":    [0.0, 0.0, 1.0, 0.0], # Index 2 feature
    "NODE_PO":     [0.0, 0.0, 0.0, 1.0]  # Index 3 feature
}

# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_LABEL_ENCODING = {
    "EDGE_REG": [1.0, 0.0],  # Index 0 feature
    "EDGE_INV": [0.0, 1.0]   # Index 1 feature
}

# --- Final Vocab Size Calculation ---
# Determine the highest ID used across both the main vocab and special tokens
# In src/aig_config.py

import networkx as nx
from typing import Union

# Make sure NODE_TYPE_KEYS, EDGE_TYPE_KEYS are defined as before
# NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
# EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]


# Define these keys globally or ensure they are accessible
VALIDITY_ERROR_KEYS = [
    "VALID", "NOT_DAG", "MISSING_NODE_TYPE", "UNKNOWN_NODE_TYPE_ATTR",
    "INVALID_NODE_TYPE_KEY", "CONST0_IN_DEGREE", "PI_IN_DEGREE",
    "AND_IN_DEGREE", "PO_IN_DEGREE", "PO_OUT_DEGREE", "MISSING_EDGE_TYPE",
    "UNKNOWN_EDGE_TYPE_ATTR", "INVALID_EDGE_TYPE_KEY", "NOT_DIGRAPH"
]


def check_validity(graph: Union[nx.Graph, nx.DiGraph], return_error_code: bool = False) -> Union[bool, str]:
    """
    Checks the structural validity of an AIG.
    If return_error_code is True, returns a string code from VALIDITY_ERROR_KEYS.
    Otherwise, returns bool.
    """
    error_code_mode = return_error_code  # Store for clarity

    if not isinstance(graph, nx.DiGraph):
        return VALIDITY_ERROR_KEYS[13] if error_code_mode else False  # "NOT_DIGRAPH"

    working_graph = graph

    if not nx.is_directed_acyclic_graph(working_graph):
        return VALIDITY_ERROR_KEYS[1] if error_code_mode else False  # "NOT_DAG"

    # Node checks
    for node, data in working_graph.nodes(data=True):
        if 'type' not in data:
            return VALIDITY_ERROR_KEYS[2] if error_code_mode else False  # "MISSING_NODE_TYPE"
        node_type = data['type']

        if node_type == "UNKNOWN_TYPE_ATTRIBUTE":  # Ensure this matches how unknown types are marked
            return VALIDITY_ERROR_KEYS[3] if error_code_mode else False
        if node_type not in NODE_TYPE_KEYS:
            return VALIDITY_ERROR_KEYS[4] if error_code_mode else False

        in_degree = working_graph.in_degree(node)
        out_degree = working_graph.out_degree(node)

        if node_type == NODE_TYPE_KEYS[0]:  # NODE_CONST0
            if in_degree != 0:
                return VALIDITY_ERROR_KEYS[5] if error_code_mode else False
        elif node_type == NODE_TYPE_KEYS[1]:  # NODE_PI
            if in_degree != 0:
                return VALIDITY_ERROR_KEYS[6] if error_code_mode else False
        elif node_type == NODE_TYPE_KEYS[2]:  # NODE_AND
            if in_degree > 2:  # Or == 2 if strictly two inputs
                return VALIDITY_ERROR_KEYS[7] if error_code_mode else False
        elif node_type == NODE_TYPE_KEYS[3]:  # NODE_PO
            if in_degree > 1:  # Or == 1
                return VALIDITY_ERROR_KEYS[8] if error_code_mode else False
            if out_degree != 0:
                return VALIDITY_ERROR_KEYS[9] if error_code_mode else False

    # Edge checks
    for u, v, data in working_graph.edges(data=True):
        if 'type' not in data:
            return VALIDITY_ERROR_KEYS[10] if error_code_mode else False  # "MISSING_EDGE_TYPE"
        edge_type = data['type']
        # Assuming "EDGE_GENERIC_OR_PADDING" is filtered out before this check by the converter
        if edge_type not in EDGE_TYPE_KEYS:
            return VALIDITY_ERROR_KEYS[12] if error_code_mode else False  # "INVALID_EDGE_TYPE_KEY"

    return VALIDITY_ERROR_KEYS[0] if error_code_mode else True  # "VALID"

# You might also want to pass NODE_TYPE_KEYS and EDGE_TYPE_KEYS explicitly
# or ensure they are correctly imported within check_validity if it's complex.