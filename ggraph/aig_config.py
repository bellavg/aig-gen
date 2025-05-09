import networkx as nx
import warnings

# --- Primary Configuration Constants ---
dataset = 'aig'

# AIG constraint constants
MAX_NODE_COUNT = 64
MIN_PI_COUNT = 2
MAX_PI_COUNT = 8
MIN_PO_COUNT = 1
MAX_PO_COUNT = 8
MIN_AND_COUNT = 1 # Assuming at least one AND gate needed


NUM_ADJ_CHANNELS = 3

NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]


NUM_NODE_FEATURES = len(NODE_TYPE_KEYS) # Should be 4
NUM_EXPLICIT_EDGE_TYPES = len(EDGE_TYPE_KEYS) # Should be 2

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


# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_PYG = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": 0, # Index 0 feature
    "NODE_PI":     1, # Index 1 feature
    "NODE_AND":    2, # Index 2 feature
    "NODE_PO":     3  # Index 3 feature
}

# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NUM2NODETYPE = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    0: "NODE_CONST0", # Index 0 feature
    1: "NODE_PI", # Index 1 feature
    2: "NODE_AND", # Index 2 feature
    3: "NODE_PO"  # Index 3 feature
}
# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_LABEL_ENCODING_PYG = {
    "EDGE_REG": 0,  # Index 0 feature
    "EDGE_INV": 1,   # Index 1 feature
    "NONE" : 2
}
VIRTUAL_EDGE_INDEX = EDGE_LABEL_ENCODING_PYG['NONE']
# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
NUM2EDGETYPE = {
    0: "EDGE_REG",  # Index 0 feature
    1: "EDGE_INV",   # Index 1 feature
}


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


def check_aig_component_minimums(current_aig_graph: nx.DiGraph) -> bool:
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
        "max_size": MAX_NODE_COUNT, "node_dim": NUM_NODE_FEATURES, "bond_dim": NUM_ADJ_CHANNELS, "use_gpu": True,
        "edge_unroll": 25, "num_flow_layer": 12, "num_rgcn_layer": 3,
        "nhid": 128, "nout": 128,
        "deq_coeff": 0.9, "st_type": "exp", "use_df": False
    },
    # "model_ebm": {
    #     "hidden": 64, "depth": 2, "swish_act": True, "add_self": False,
    #     "dropout": 0.0, "n_power_iterations": 1
    # },
    "lr": 0.001, "weight_decay": 1e-5, "batch_size": 32, "max_epochs": 30,
    "save_interval": 3, "grad_clip_value": 1.0,
    # "train_ebm": {
    #     "c": 0.0, "ld_step": 150, "ld_noise": 0.005, "ld_step_size": 30,
    #     "clamp_lgd_grad": True, "alpha": 1.0
    # }
}

