# G2PT/configs/aig.py
import math
import os # Added for potential path joining if needed
import networkx as nx
from typing import Union, Optional # Added for type hinting
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


def to_directed_aig(graph: Union[nx.Graph, nx.DiGraph]) -> Optional[nx.DiGraph]:
    """
    Converts an input graph to a directed AIG representation.
    - If the input is already a DiGraph, it's returned as is.
    - If the input is an undirected Graph, it's converted to a DiGraph
      with edges pointing from the node with the smaller ID to the node
      with the larger ID. Node and edge attributes are preserved.
    - If the input is not a NetworkX Graph object, None is returned.

    Args:
        graph (Union[nx.Graph, nx.DiGraph]): The input graph.

    Returns:
        Optional[nx.DiGraph]: The directed graph, or None if conversion is not possible.
    """
    if isinstance(graph, nx.DiGraph):
        return graph  # Already a DiGraph

    if isinstance(graph, nx.Graph):  # Covers undirected nx.Graph
        if graph.is_directed():  # Should have been caught by first isinstance, but as a safeguard
            warnings.warn("Input graph to to_directed_aig was nx.Graph but already directed. Returning as is.")
            return graph  # Technically a DiGraph if it passes is_directed()

        # Convert undirected graph to directed (smaller_id -> larger_id)
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(graph.nodes(data=True))  # Copy nodes and their attributes

        for u, v, data in graph.edges(data=True):
            edge_attrs = data.copy()  # Ensure edge attributes are copied
            if u == v:  # Handle self-loops if they exist (though not typical for AIG edges from undirected)
                directed_graph.add_edge(u, v, **edge_attrs)
            elif u < v:
                directed_graph.add_edge(u, v, **edge_attrs)
            else:  # v < u
                directed_graph.add_edge(v, u, **edge_attrs)
        return directed_graph

    # Input is not a recognized NetworkX graph type for this conversion
    return None


def check_validity(graph_input: Union[nx.Graph, nx.DiGraph, None]) -> bool:
    """
    Checks the structural validity of a (potentially partially built) AIG.
    It first ensures the graph is directed (handling undirected inputs)
    and then applies AIG-specific validation rules.

    Args:
        graph_input (Union[nx.Graph, nx.DiGraph, None]): The AIG graph to validate.
                                                       Node and edge 'type' attributes
                                                       are assumed to be strings.
                                                       None or an empty graph is considered valid.

    Returns:
        bool: True if the graph is valid according to AIG rules, False otherwise.
    """
    # Handle None or empty graph objects (which might evaluate to False)
    # An empty graph (e.g., nx.Graph() or nx.DiGraph()) is considered valid.
    if not graph_input:
        # This covers graph_input being None or an empty graph container that evaluates to False.
        # print("Debug: Validity Check - Input graph is None or empty. Considered valid.")
        return True

    # Convert to a working DiGraph; to_directed_aig handles type checking.
    working_graph = to_directed_aig(graph_input)

    if working_graph is None:
        # This means to_directed_aig received a non-graph type it couldn't process.
        # An empty nx.Graph() would have been successfully converted to an empty nx.DiGraph().
        print(
            f"Debug: Validity Check Failed - Input graph type '{type(graph_input)}' could not be processed into a DiGraph.")
        return False

    # If working_graph is an empty DiGraph (e.g., from an empty input graph),
    # it's a DAG, and loops below won't run, so it will correctly return True.

    # 1. Check DAG property
    if not nx.is_directed_acyclic_graph(working_graph):
        # print("Debug: Validity Check Failed - Not a DAG")
        return False

    # Access EXPLICIT_NODE_TYPE_KEYS and EXPLICIT_EDGE_TYPE_KEYS
    # These are expected to be defined in the global scope (e.g., in aig_config.py)
    try:
        # These variables are used for comparison, ensure they are defined.
        # (Assuming they are imported or defined in the same file as this function)
        _NODE_CONST0_STR = EXPLICIT_NODE_TYPE_KEYS[0]
        _NODE_PI_STR = EXPLICIT_NODE_TYPE_KEYS[1]
        _NODE_AND_STR = EXPLICIT_NODE_TYPE_KEYS[2]
        _NODE_PO_STR = EXPLICIT_NODE_TYPE_KEYS[3]
    except NameError:
        print(
            "Debug: Validity Check Failed - EXPLICIT_NODE_TYPE_KEYS is not defined. Cannot perform detailed node type checks.")
        return False  # Cannot validate without type definitions
    except IndexError:
        print(
            "Debug: Validity Check Failed - EXPLICIT_NODE_TYPE_KEYS does not contain enough elements. Cannot perform detailed node type checks.")
        return False

    # 2. Check Node Types and Degrees
    for node, data in working_graph.nodes(data=True):
        if 'type' not in data:
            print(f"Debug: Validity Check Failed - Node {node} is missing 'type' attribute.")
            return False
        node_type = data['type']

        if node_type == "UNKNOWN_TYPE_ATTRIBUTE":  # Specific string to check against
            print(f"Debug: Validity Check Failed - Node {node} has explicit 'UNKNOWN_TYPE_ATTRIBUTE': {data}")
            return False
        if node_type not in EXPLICIT_NODE_TYPE_KEYS:
            print(
                f"Debug: Validity Check Failed - Node {node} has type '{node_type}' not in defined EXPLICIT_NODE_TYPE_KEYS.")
            return False

        in_degree = working_graph.in_degree(node)
        out_degree = working_graph.out_degree(node)

        if node_type == _NODE_CONST0_STR:
            if in_degree != 0:
                # print(f"Debug: Validity Check Failed - CONST0 node {node} (type: {node_type}) has in-degree {in_degree} (should be 0).")
                return False
        elif node_type == _NODE_PI_STR:
            if in_degree != 0:
                # print(f"Debug: Validity Check Failed - PI node {node} (type: {node_type}) has in-degree {in_degree} (should be 0).")
                return False
        elif node_type == _NODE_AND_STR:
            # For AND gates, typical AIGs have exactly 2 inputs.
            # Some definitions might allow more, but the original code checked for > 2.
            # If it's strictly 2, change to `if in_degree != 2:`.
            # Sticking to original logic:
            if in_degree > 2:  # Or in_degree != 2 if strictly two inputs
                # print(f"Debug: Validity Check Failed - AND node {node} (type: {node_type}) has in-degree {in_degree} (should be <= 2 or exactly 2).")
                return False
        elif node_type == _NODE_PO_STR:
            # For PO nodes, typical AIGs have exactly 1 input.
            # The original code checked for > 1.
            # If it's strictly 1, change to `if in_degree != 1:`.
            # Sticking to original logic:
            if in_degree > 1:  # Or in_degree != 1 if strictly one input
                # print(f"Debug: Validity Check Failed - PO node {node} (type: {node_type}) has in-degree {in_degree} (should be <= 1 or exactly 1).")
                return False
            if out_degree != 0:
                # print(f"Debug: Validity Check Failed - PO node {node} (type: {node_type}) has out-degree {out_degree} (should be 0).")
                return False

    # 3. Check Edge Types
    try:
        # Ensure EXPLICIT_EDGE_TYPE_KEYS is defined for comparison
        _ = EXPLICIT_EDGE_TYPE_KEYS
    except NameError:
        print(
            "Debug: Validity Check Failed - EXPLICIT_EDGE_TYPE_KEYS is not defined. Cannot perform detailed edge type checks.")
        return False  # Cannot validate without type definitions

    for u, v, data in working_graph.edges(data=True):
        if 'type' not in data:
            print(f"Debug: Validity Check Failed - Edge ({u}-{v}) is missing 'type' attribute.")
            return False
        edge_type = data['type']

        if edge_type == "UNKNOWN_TYPE_ATTRIBUTE":  # Specific string to check against
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has explicit 'UNKNOWN_TYPE_ATTRIBUTE': {data}")
            return False
        if edge_type not in EXPLICIT_EDGE_TYPE_KEYS:
            # print(f"Debug: Validity Check Failed - Edge ({u}-{v}) has type '{edge_type}' not in defined EXPLICIT_EDGE_TYPE_KEYS.")
            return False

    return True

def check_aig_component_minimums(current_aig_graph: nx.Graph) -> bool:
    #TODO defintiely needs to be changed
    """
    Checks if the given AIG meets the minimum component criteria.
    - At least 1 "NODE_AND"
    - At least 2 "NODE_PI"
    - At least 1 "NODE_PO"
    """
    if current_aig_graph is None or current_aig_graph.number_of_nodes() == 0:
        return False

    # Convert to a working DiGraph; to_directed_aig handles type checking.
    current_aig_graph = to_directed_aig(current_aig_graph)


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

