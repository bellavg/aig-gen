# G2PT/configs/aig.py
import math
import os # Added for potential path joining if needed
import networkx as nx

# --- Primary Configuration Constants ---
dataset = 'aig'

# Consider using os.path.join for better path handling
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets G2PT directory
# data_dir = os.path.join(base_dir, 'datasets', 'aig')
# tokenizer_path = os.path.join(base_dir, 'tokenizers', 'aig')


# AIG constraint constants
MAX_NODE_COUNT = 64
MIN_PI_COUNT = 2
MAX_PI_COUNT = 8
MIN_PO_COUNT = 1
MAX_PO_COUNT = 8
MIN_AND_COUNT = 1 # Assuming at least one AND gate needed



NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]


NODE_TYPE_ENCODING = {
    "NODE_CONST0": [1.0, 0.0, 0.0, 0.0],
    "NODE_PI":     [0.0, 1.0, 0.0, 0.0],
    "NODE_AND":    [0.0, 0.0, 1.0, 0.0],
    "NODE_PO":     [0.0, 0.0, 0.0, 1.0]
}

DECODING_NODE_TYPE_NX = {
    (1.0, 0.0, 0.0, 0.0): "NODE_CONST0",
    (0.0, 1.0, 0.0, 0.0): "NODE_PI",
    (0.0, 0.0, 1.0, 0.0): "NODE_AND",
    (0.0, 0.0, 0.0, 1.0): "NODE_PO"
}
# Derive feature counts from the size of the derived vocabularies
NUM_NODE_FEATURES = len(NODE_TYPE_KEYS) # Should be 4
NUM_EDGE_FEATURES = len(EDGE_TYPE_KEYS) # Should be 2



# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_LABEL_ENCODING = {
    "EDGE_REG": [1.0, 0.0],  # Index 0 feature
    "EDGE_INV": [0.0, 1.0]   # Index 1 feature
}

DECODING_EDGE_TYPE_NX = {
    (1.0, 0.0): "EDGE_REG" ,  # Index 0 feature
    (0.0, 1.0) :"EDGE_INV"  # Index 1 feature
}


# --- Final Vocab Size Calculation ---
# Determine the highest ID used across both the main vocab and special tokens



def check_validity(graph: nx.DiGraph) -> bool:
    """
    Checks the structural validity of a (potentially partially built) AIG.
    This function is called during the generation process.

    Args:
        graph (nx.DiGraph): The AIG graph to validate. It's assumed that
                            node and edge 'type' attributes are strings
                            (e.g., "NODE_PI", "EDGE_REG").

    Returns:
        bool: True if the graph is currently valid according to AIG rules, False otherwise.
    """
    if not graph: # Handle empty graph case if necessary
        return True # An empty graph might be considered valid at the start

    # 1. Check DAG property
    if not nx.is_directed_acyclic_graph(graph):
        # print("Debug: Validity Check Failed - Not a DAG")
        return False

    # Ensure NODE_TYPE_KEYS is accessible in this scope (it should be global in aig_config.py)
    # These string constants must match the 'type' attributes set on nodes/edges
    # NODE_TYPE_KEYS should be defined in aig_config.py as e.g.
    # NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
    # EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]
    # If these are not defined, this function will error or behave unexpectedly.

    NODE_CONST0_STR = NODE_TYPE_KEYS[0]
    NODE_PI_STR = NODE_TYPE_KEYS[1]
    NODE_AND_STR = NODE_TYPE_KEYS[2]
    NODE_PO_STR = NODE_TYPE_KEYS[3]

    for node, data in graph.nodes(data=True):
        node_type = data['type']

        if node_type == "UNKNOWN_TYPE_ATTRIBUTE":
            print(f"Debug: Validity Check Failed - Node {node} has missing or malformed 'type' attribute: {data}")
            return False
        if node_type not in NODE_TYPE_KEYS:
            print(f"Debug: Validity Check Failed - Node {node} has type '{node_type}' not in defined NODE_TYPE_KEYS.")
            return False

        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        if node_type == NODE_CONST0_STR:
            if in_degree != 0:
                # print(f"Debug: Validity Check Failed - CONST0 node {node} (type: {node_type}) has in-degree {in_degree} (should be 0).")
                return False
        elif node_type == NODE_PI_STR:
            if in_degree != 0:
                # print(f"Debug: Validity Check Failed - PI node {node} (type: {node_type}) has in-degree {in_degree} (should be 0).")
                return False
        elif node_type == NODE_AND_STR:
            # For AND gates during generation, in-degree can be 0, 1, or 2.
            # The final check in evaluation_aigs.py will be stricter (must be 2).
            if in_degree > 2:
                # print(f"Debug: Validity Check Failed - AND node {node} (type: {node_type}) has in-degree {in_degree} (should be <= 2).")
                return False
        elif node_type == NODE_PO_STR:
            # For PO gates during generation, in-degree can be 0 or 1.
            # The final check in evaluation_aigs.py will be stricter (must be 1).
            if in_degree > 1:
                # print(f"Debug: Validity Check Failed - PO node {node} (type: {node_type}) has in-degree {in_degree} (should be <= 1).")
                return False
            if out_degree != 0:
                # print(f"Debug: Validity Check Failed - PO node {node} (type: {node_type}) has out-degree {out_degree} (should be 0).")
                return False


    # 3. Check Edge Types
    for u, v, data in graph.edges(data=True):
        edge_type = data['type']
        if edge_type == "UNKNOWN_TYPE_ATTRIBUTE":
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has missing or malformed 'type' attribute: {data}")
            return False
        if edge_type not in EDGE_TYPE_KEYS: # EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has type '{edge_type}' not in defined EDGE_TYPE_KEYS.")
            return False

    return True