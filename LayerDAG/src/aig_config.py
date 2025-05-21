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


# AIG constraint constants
MAX_NODE_COUNT = 64
MIN_PI_COUNT = 2
MAX_PI_COUNT = 8
MIN_PO_COUNT = 1
MAX_PO_COUNT = 8
MIN_AND_COUNT = 1 # Assuming at least one AND gate needed



NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EDGE_TYPE_KEYS = ["EDGE_REG"]



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
    "EDGE_REG": [1.0] # Or simply ensure "EDGE_REG" string is used
}
# --- Final Vocab Size Calculation ---
# Determine the highest ID used across both the main vocab and special tokens



def check_validity(graph: Union[nx.Graph, nx.DiGraph]) -> bool: # Changed type hint
    """
    Checks the structural validity of a (potentially partially built) AIG.
    If the input graph is undirected, it's first converted to a directed graph
    where edges go from smaller node ID to larger node ID.

    Args:
        graph (Union[nx.Graph, nx.DiGraph]): The AIG graph to validate.
                                             Node and edge 'type' attributes are assumed
                                             to be strings.

    Returns:
        bool: True if the graph is currently valid according to AIG rules, False otherwise.
    """
    if not graph: # Handle empty graph case
        return True

    working_graph = nx.DiGraph() # Ensure we are working with a DiGraph

    if isinstance(graph, nx.Graph) and not graph.is_directed():
        # Input is undirected, convert to directed (smaller_id -> larger_id)
        working_graph.add_nodes_from(graph.nodes(data=True)) # Copy nodes and their attributes
        for u, v, data in graph.edges(data=True):
            # data is a dictionary of edge attributes, make sure to copy them
            edge_attrs = data.copy()
            if u < v:
                working_graph.add_edge(u, v, **edge_attrs)
            else: # v < u (or u == v for self-loops, though less common for AIG edges from undirected)
                working_graph.add_edge(v, u, **edge_attrs)
    elif isinstance(graph, nx.DiGraph):
        working_graph = graph # Use the graph directly if it's already a DiGraph
    else:
        # Should not happen if type hint is respected, but as a safeguard
        print("Debug: Validity Check Failed - Input graph is neither nx.Graph nor nx.DiGraph.")
        return False


    # 1. Check DAG property (on the now guaranteed DiGraph)
    if not nx.is_directed_acyclic_graph(working_graph):
        # print("Debug: Validity Check Failed - Not a DAG")
        return False

    # Ensure NODE_TYPE_KEYS and EDGE_TYPE_KEYS are accessible
    # These constants should be defined in your aig_config.py or globally
    # For example:
    # NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
    # EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]

    # It's safer to ensure they are defined, or pass them as arguments,
    # but following the original structure:
    try:
        NODE_CONST0_STR = NODE_TYPE_KEYS[0]
        NODE_PI_STR = NODE_TYPE_KEYS[1]
        NODE_AND_STR = NODE_TYPE_KEYS[2]
        NODE_PO_STR = NODE_TYPE_KEYS[3]
    except (NameError, IndexError) as e:
        print(f"Debug: Validity Check Failed - NODE_TYPE_KEYS not properly defined or accessible: {e}")
        return False


    for node, data in working_graph.nodes(data=True):
        if 'type' not in data:
            print(f"Debug: Validity Check Failed - Node {node} is missing 'type' attribute.")
            return False
        node_type = data['type']

        # This check was "UNKNOWN_TYPE_ATTRIBUTE", which might be too specific if 'type' is just missing
        # Keeping similar logic for now.
        if node_type == "UNKNOWN_TYPE_ATTRIBUTE": # Consider changing this if 'type' can be missing
            print(f"Debug: Validity Check Failed - Node {node} has explicit 'UNKNOWN_TYPE_ATTRIBUTE': {data}")
            return False
        if node_type not in NODE_TYPE_KEYS:
            print(f"Debug: Validity Check Failed - Node {node} has type '{node_type}' not in defined NODE_TYPE_KEYS.")
            return False

        in_degree = working_graph.in_degree(node)
        out_degree = working_graph.out_degree(node)

        if node_type == NODE_CONST0_STR:
            if in_degree != 0:
                # print(f"Debug: Validity Check Failed - CONST0 node {node} (type: {node_type}) has in-degree {in_degree} (should be 0).")
                return False
        elif node_type == NODE_PI_STR:
            if in_degree != 0:
                # print(f"Debug: Validity Check Failed - PI node {node} (type: {node_type}) has in-degree {in_degree} (should be 0).")
                return False
        elif node_type == NODE_AND_STR:
            if in_degree > 2:
                # print(f"Debug: Validity Check Failed - AND node {node} (type: {node_type}) has in-degree {in_degree} (should be <= 2).")
                return False
        elif node_type == NODE_PO_STR:
            if in_degree > 1:
                # print(f"Debug: Validity Check Failed - PO node {node} (type: {node_type}) has in-degree {in_degree} (should be <= 1).")
                return False
            if out_degree != 0:
                # print(f"Debug: Validity Check Failed - PO node {node} (type: {node_type}) has out-degree {out_degree} (should be 0).")
                return False

    # 3. Check Edge Types
    try:
        _ = EDGE_TYPE_KEYS # Check if EDGE_TYPE_KEYS is defined
    except NameError as e:
        print(f"Debug: Validity Check Failed - EDGE_TYPE_KEYS not defined or accessible: {e}")
        return False

    for u, v, data in working_graph.edges(data=True):
        if 'type' not in data:
            print(f"Debug: Validity Check Failed - Edge ({u}-{v}) is missing 'type' attribute.")
            return False
        edge_type = data['type']

        if edge_type == "UNKNOWN_TYPE_ATTRIBUTE": # Similar to node type check
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has explicit 'UNKNOWN_TYPE_ATTRIBUTE': {data}")
            return False
        if edge_type not in EDGE_TYPE_KEYS:
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has type '{edge_type}' not in defined EDGE_TYPE_KEYS.")
            return False

    return True