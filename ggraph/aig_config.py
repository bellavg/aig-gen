# G2PT/configs/aig.py
import math
import os # Added for potential path joining if needed
import networkx as nx
from typing import Union # Added for type hinting
import warnings
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


NUM_ADJ_CHANNELS = 3

EXPLICIT_NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EXPLICIT_EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]

ALL_NODE_KEYS = EXPLICIT_NODE_TYPE_KEYS + ["PADDING_NODE"]
ALL_EDGE_KEYS = EXPLICIT_EDGE_TYPE_KEYS + ["NONE"]

NUM2EDGETYPE = {0: ALL_EDGE_KEYS[0], 1: ALL_EDGE_KEYS[1], 2: ALL_EDGE_KEYS[2]}
NUM2NODETYPE = {0: ALL_NODE_KEYS[0], 1: ALL_NODE_KEYS[1], 2: ALL_NODE_KEYS[2], 3: ALL_NODE_KEYS[3], 4: ALL_NODE_KEYS[4]}



# Derive feature counts from the size of the derived vocabularies
NUM_EXPLICIT_NODE_FEATURES = len(EXPLICIT_NODE_TYPE_KEYS) # Should be 4
NUM_EXPLICIT_EDGE_FEATURES = len(EXPLICIT_EDGE_TYPE_KEYS) # Should be 2

NUM_NODE_ATTRIBUTES = NUM_EXPLICIT_NODE_FEATURES + 1 # for none type
NUM_EDGE_ATTRIBUTES = NUM_EXPLICIT_EDGE_FEATURES + 1 # for virtual no edge
NO_EDGE_CHANNEL = NUM_EXPLICIT_EDGE_FEATURES
PADDING_NODE_CHANNEL = NUM_EXPLICIT_NODE_FEATURES


# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_NX = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], # Index 0 feature
    "NODE_PI":     [0.0, 1.0, 0.0, 0.0], # Index 1 feature
    "NODE_AND":    [0.0, 0.0, 1.0, 0.0], # Index 2 feature
    "NODE_PO":     [0.0, 0.0, 0.0, 1.0]  # Index 3 feature
}

# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_TYPE_ENCODING_NX = {
    "EDGE_REG": [1.0, 0.0],  # Index 0 feature
    "EDGE_INV": [0.0, 1.0]   # Index 1 feature
}


DECODING_NODE_TYPE_NX = {
    tuple(v): k for k, v in NODE_TYPE_ENCODING_NX.items()
}

DECODING_EDGE_TYPE_NX = {
    tuple(v): k for k, v in EDGE_TYPE_ENCODING_NX.items()
}




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
        NODE_CONST0_STR = EXPLICIT_NODE_TYPE_KEYS[0]
        NODE_PI_STR = EXPLICIT_NODE_TYPE_KEYS[1]
        NODE_AND_STR = EXPLICIT_NODE_TYPE_KEYS[2]
        NODE_PO_STR = EXPLICIT_NODE_TYPE_KEYS[3]
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
        if node_type not in EXPLICIT_NODE_TYPE_KEYS:
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
        _ = EXPLICIT_EDGE_TYPE_KEYS # Check if EDGE_TYPE_KEYS is defined
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
        if edge_type not in EXPLICIT_EDGE_TYPE_KEYS:
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has type '{edge_type}' not in defined EDGE_TYPE_KEYS.")
            return False

    return True


def check_aig_component_minimums(current_aig_graph: nx.DiGraph) -> bool:
    #TODO defintiely needs to be changed
    """
    Checks if the given AIG meets the minimum component criteria.
    - At least 1 "NODE_AND"
    - At least 2 "NODE_PI"
    - At least 1 "NODE_PO"
    """
    if current_aig_graph is None or current_aig_graph.number_of_nodes() == 0:
        return False

    type_counts = {"NODE_AND": 0, "NODE_PI": 0, "NODE_PO": 0}
    # Node types are stored as strings like "NODE_PI", "NODE_AND" etc.
    # These strings are the values in the NUM2NODETYPE dictionary.

    # We need to find the string values for PI, PO, AND from NUM2NODETYPE
    # This assumes NUM2NODETYPE is {0: "NODE_CONST0", 1: "NODE_PI", 2: "NODE_AND", 3: "NODE_PO"}
    # A more robust way would be to iterate NUM2NODETYPE's values if keys are not fixed.
    # For now, assuming fixed mapping or direct string comparison.

    # String representations for target node types
    # These should match the values in your NUM2NODETYPE from aig_config.py
    # Example: if NUM2NODETYPE = {0:"const0", 1:"pi", 2:"and", 3:"po"}
    # then target_node_type_and = "and", etc.
    # For this example, I'll assume the strings are exactly "NODE_AND", "NODE_PI", "NODE_PO"
    # as per common naming, but ensure these match your config's string values.

    # Find the string representation for AND, PI, PO from NUM2NODETYPE
    str_node_and = None
    str_node_pi = None
    str_node_po = None

    for key, value in NUM2NODETYPE.items():
        if "AND" in value.upper():  # Case-insensitive check for "AND"
            str_node_and = value
        elif "PI" in value.upper():  # Case-insensitive check for "PI"
            str_node_pi = value
        elif "PO" in value.upper():  # Case-insensitive check for "PO"
            str_node_po = value

    if not all([str_node_and, str_node_pi, str_node_po]):
        warnings.warn(
            f"Could not find all required node type strings (AND, PI, PO) in NUM2NODETYPE: {NUM2NODETYPE}. Component check might be inaccurate.")
        # Fallback to direct string comparison if mapping is complex
        str_node_and = str_node_and or "NODE_AND"
        str_node_pi = str_node_pi or "NODE_PI"
        str_node_po = str_node_po or "NODE_PO"

    for _, data in current_aig_graph.nodes(data=True):
        node_type_str = data.get('type')
        if node_type_str == str_node_and:
            type_counts["NODE_AND"] += 1
        elif node_type_str == str_node_pi:
            type_counts["NODE_PI"] += 1
        elif node_type_str == str_node_po:
            type_counts["NODE_PO"] += 1

    meets_criteria = (
            type_counts["NODE_AND"] >= 1 and
            type_counts["NODE_PI"] >= 2 and
            type_counts["NODE_PO"] >= 1
    )
    return meets_criteria


# --- Base Configuration Dictionary (Defaults) ---
base_conf = {
    "data_name": "aig",  # This will be overridden by args.dataset_name
    "model": {
        "max_size": MAX_NODE_COUNT, "node_dim": NUM_NODE_ATTRIBUTES, "bond_dim": NUM_ADJ_CHANNELS, "use_gpu": True,
        "edge_unroll": 25, "num_flow_layer": 12, "num_rgcn_layer": 3,
        "nhid": 128, "nout": 128,
        "deq_coeff": 0.9, "st_type": "exp", "use_df": False,
        "c": 0.05,  # Corresponds to former 'dequantization_scale_c'
        "ld_step": 60,  # Corresponds to former 'langevin_steps_k'
        "ld_noise": 0.005,  # Corresponds to former 'langevin_noise_sigma'
        "ld_step_size": 30.0,  # Corresponds to former 'langevin_step_size_lambda_half'
        "clamp": True,  # Corresponds to former 'langevin_clamp_grad' (user requested 'clamp')
        "alpha": 1.0,  # Corresponds to former 'loss_alpha_reg_weight'
        "hidden": 64

    },
    "lr": 0.001, "weight_decay": 1e-5, "batch_size": 32, "max_epochs": 30,
    "save_interval": 3, "grad_clip_value": 1.0,
}

